"""
Sinh dữ liệu "câu/dòng" bằng cách ghép (stitch) nhiều ảnh từ-level thành một ảnh dòng hoàn chỉnh.

Mục tiêu:
- Tận dụng tập ảnh từ đơn (word images) + nhãn tương ứng để tạo ảnh dòng (sentence/line image).
- Mỗi dòng được tạo bằng cách chọn ngẫu nhiên 3..7 từ, binarize, cắt khoảng trắng thừa,
  rồi ghép ngang (np.hstack) và chèn khoảng trắng giữa các từ.

Đầu vào:
- WORD_IMG_DIR: thư mục chứa ảnh từ-level.
- WORD_LABEL_FILE: file labels JSON mapping {filename: word_text}.

Đầu ra:
- OUTPUT_DIR: thư mục chứa ảnh dòng đã ghép.
- OUTPUT_LABEL_FILE: labels JSON mapping {filename: sentence_text}.

Ghi chú:
- Ảnh được xử lý grayscale, resize về cùng chiều cao `IMG_HEIGHT`, sau đó Otsu threshold để binarize.
- Hàm `trim_whitespace` chỉ cắt biên theo trục ngang (x) để loại vùng trắng vô nghĩa.
- Script bỏ qua các ảnh không đọc được hoặc thiếu file tương ứng.
"""

import os
import random
import json
import cv2
import numpy as np
from tqdm import tqdm
import sys

# Import config bằng cách thêm thư mục cha vào sys.path theo cấu trúc dự án.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- CẤU HÌNH ---
# Dataset từ-level và labels tương ứng.
WORD_IMG_DIR = os.path.join(config.OCR_DATASET_DIR, "word_test")
WORD_LABEL_FILE = os.path.join(WORD_IMG_DIR, "labels_merged.json")

# Thư mục và labels đầu ra cho dữ liệu dòng/câu.
OUTPUT_DIR = os.path.join(config.OCR_DATASET_DIR, "sentence_test")
OUTPUT_LABEL_FILE = os.path.join(config.OCR_DATASET_DIR, "labels_sentence_test.json")

# Số lượng dòng cần sinh và chiều cao chuẩn hoá.
NUM_LINES_TO_GEN = 1000
IMG_HEIGHT = 118


def resize_keep_ratio(img, target_h):
    """Resize ảnh theo chiều cao mục tiêu, giữ nguyên tỷ lệ khung hình."""
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def clean_and_binarize(img):
    """Chuẩn hoá kích thước và nhị phân hoá ảnh bằng Otsu threshold."""
    img = resize_keep_ratio(img, IMG_HEIGHT)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_bin


def trim_whitespace(img):
    """Cắt khoảng trắng thừa theo chiều ngang dựa trên bounding box vùng mực.

    - Đảo ảnh (invert) để coi vùng chữ là pixel khác 0.
    - Tìm bounding box, sau đó cắt theo trục x với padding nhỏ 2px.
    """
    img_inv = 255 - img
    coords = cv2.findNonZero(img_inv)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[:, max(0, x - 2) : min(img.shape[1], x + w + 2)]


def main():
    """Điểm vào chính: nạp vocab, chọn từ ngẫu nhiên, ghép ảnh và ghi labels."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"[INFO] Đang đọc từ vựng: {WORD_LABEL_FILE}")
    try:
        with open(WORD_LABEL_FILE, "r", encoding="utf-8") as f:
            word_dict = json.load(f)
    except FileNotFoundError:
        print("[ERR] Không tìm thấy file nhãn word.")
        return

    # Tạo mapping text -> list ảnh để có thể chọn ảnh ngẫu nhiên cho cùng một từ.
    text_to_images = {}
    for fname, text in word_dict.items():
        if text not in text_to_images:
            text_to_images[text] = []
        text_to_images[text].append(fname)

    unique_texts = list(text_to_images.keys())
    new_labels = {}

    print(f"[INFO] Sinh {NUM_LINES_TO_GEN} dòng...")
    for i in tqdm(range(NUM_LINES_TO_GEN)):
        # Chọn số từ trong một dòng (giới hạn theo số unique_texts thực tế).
        num_words = random.randint(3, 7)
        selected_texts = random.sample(unique_texts, k=min(num_words, len(unique_texts)))

        parts, final_texts = [], []
        for text in selected_texts:
            # Chọn ngẫu nhiên một ảnh đại diện cho từ này.
            img_name = random.choice(text_to_images[text])
            path = os.path.join(WORD_IMG_DIR, img_name)
            if not os.path.exists(path):
                continue

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Tiền xử lý ảnh từ-level trước khi ghép.
            img_bin = clean_and_binarize(img)
            parts.append(trim_whitespace(img_bin))

            # Lưu text theo đúng entry đã chọn (đảm bảo consistency).
            final_texts.append(word_dict[img_name])

            # Chèn một block trắng để tạo khoảng cách giữa các từ.
            parts.append(np.full((IMG_HEIGHT, random.randint(10, 25)), 255, dtype=np.uint8))

        if not parts:
            continue

        # Bỏ khoảng trắng cuối để tránh dư vùng trắng ở cuối dòng.
        parts = parts[:-1]

        try:
            full_line = np.hstack(parts)
            fname = f"stitched_{i:06d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, fname), full_line)

            # Nhãn dòng là chuỗi ghép các từ theo đúng thứ tự đã dựng.
            new_labels[fname] = " ".join(final_texts)
        except:
            continue

    # Ghi labels JSON cho tập dữ liệu dòng/câu.
    with open(OUTPUT_LABEL_FILE, "w", encoding="utf-8") as f:
        json.dump(new_labels, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
