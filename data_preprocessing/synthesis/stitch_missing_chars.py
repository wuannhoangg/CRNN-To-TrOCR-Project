"""
Sinh dữ liệu OCR tổng hợp (synthetic) theo dạng "dòng văn bản có ngữ cảnh" để bù ký tự thiếu.

Mục tiêu:
- So sánh bộ ký tự chuẩn (PRETRAIN_ALPHABET) với bộ ký tự hiện có (REAL_CHARSET_JSON).
- Tìm các ký tự còn thiếu trong dataset hiện tại.
- Sinh NUM_LINES ảnh dạng dòng text, trong đó mỗi dòng đảm bảo chứa (ít nhất) một ký tự thiếu.
- Mỗi dòng gồm từ đệm (filler words) lấy từ từ điển thật + một từ mục tiêu có chứa ký tự thiếu.
- Render bằng nhiều font để tăng tính đa dạng, và thêm augmentation nhẹ (noise/blur).

Đầu vào:
- PRETRAIN_ALPHABET: file alphabet định nghĩa tập ký tự mục tiêu.
- REAL_CHARSET_JSON: JSON hiện tại (có thể là list charset hoặc dict nhãn) để suy ra tập ký tự đang có.
- WORD_LABEL_SRC: labels JSON từ dataset thật để nạp "vốn từ" (vocabulary).

Đầu ra:
- OUTPUT_DIR: thư mục chứa ảnh synthetic.
- OUTPUT_LABEL: file JSON mapping {filename: text} của ảnh synthetic.
"""

import os
import json
import random
import unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import sys

# Import config dự án bằng cách thêm đường dẫn cha vào sys.path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- CẤU HÌNH ---
# File alphabet mục tiêu (ground-truth charset muốn cover).
PRETRAIN_ALPHABET = config.ALPHABET_FILE

# JSON charset hiện tại (có thể là list charset hoặc dict nhãn).
REAL_CHARSET_JSON = config.CURRENT_CHARSET_JSON

# File từ điển mẫu (labels JSON) để lấy vocabulary thực tế.
WORD_LABEL_SRC = os.path.join(config.OCR_DATASET_DIR, "vi_word_all", "labels_merged.json")

# Thư mục và file labels đầu ra cho dữ liệu synthetic.
OUTPUT_DIR = os.path.join(config.OCR_DATASET_DIR, "synthetic_lines_context")
OUTPUT_LABEL = os.path.join(config.OCR_DATASET_DIR, "labels_lines_context.json")

# Số lượng dòng ảnh cần sinh.
NUM_LINES = 5000

# Chiều cao ảnh cố định (phù hợp pipeline OCR).
IMG_H = 118

# Danh sách font để render; lấy từ config để quản lý tập trung.
FONTS = config.FONT_PATHS


def normalize_text(text):
    """Chuẩn hoá Unicode về dạng NFC để đồng nhất ký tự (tránh lệch tổ hợp dấu)."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def load_charset(path, is_json=False):
    """Nạp tập ký tự từ file alphabet hoặc từ JSON.

    - Nếu `is_json=False`: đọc từng dòng trong file alphabet; '<space>' được ánh xạ thành ' '.
    - Nếu `is_json=True`:
      - Nếu JSON là list: coi là danh sách charset (bỏ token đặc biệt).
      - Nếu JSON là dict: coi value là text nhãn; trích toàn bộ ký tự từ nhãn.

    Args:
        path (str): Đường dẫn file charset/alphabet hoặc JSON.
        is_json (bool): True nếu đầu vào là JSON, False nếu là file alphabet dạng text.

    Returns:
        set: Tập ký tự (đã normalize NFC).
    """
    chars = set()
    if is_json:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for c in data:
                if c not in ["[SOS]", "[EOS]", "[PAD]"]:
                    chars.add(normalize_text(c))
        elif isinstance(data, dict):
            for v in data.values():
                for c in normalize_text(v):
                    chars.add(c)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                c = line.strip()
                if c:
                    if c == "<space>":
                        chars.add(" ")
                    else:
                        chars.add(normalize_text(c))
    return chars


def load_vocabulary(path):
    """Nạp danh sách từ vựng từ file labels JSON nguồn.

    - Chỉ lấy text (value), không dùng key (tên ảnh).
    - Lọc những mục có độ dài > 1 để ưu tiên từ/cụm từ có nghĩa.

    Args:
        path (str): Đường dẫn file labels JSON.

    Returns:
        list: Danh sách chuỗi từ vựng đã normalize NFC.
    """
    print(f"[INFO] Đang nạp từ điển...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = list(data.values())
    words = [normalize_text(w) for w in words if len(w) > 1]
    print(f" -> Đã nạp {len(words)} từ vựng.")
    return words


def create_target_text(missing_char, vocab):
    """Tạo một từ/cụm từ đảm bảo có chứa ký tự thiếu.

    Chiến thuật:
    1) Tìm trong vocab các từ có chứa ký tự này (so sánh không phân biệt hoa/thường).
    2) Nếu ký tự thiếu là chữ in hoa và tìm được ứng viên: chọn 1 từ và viết HOA toàn bộ.
    3) Nếu không có từ phù hợp: sinh chuỗi fallback cho một số ký tự đặc biệt,
       hoặc chèn ký tự thiếu vào một cụm đơn giản.

    Args:
        missing_char (str): Ký tự đang thiếu cần được đưa vào dòng text.
        vocab (list): Danh sách từ vựng.

    Returns:
        str: Chuỗi mục tiêu có chứa ký tự thiếu.
    """
    missing_char_lower = missing_char.lower()

    # Chọn các từ mà khi lower() có chứa ký tự thiếu.
    candidates = [w for w in vocab if missing_char_lower in w.lower()]

    if candidates and missing_char.isupper():
        word = random.choice(candidates).upper()
        return word
    elif candidates:
        return random.choice(candidates)
    else:
        # Fallback cho một số trường hợp ký tự không có trong từ điển tiếng Việt.
        if missing_char == "$":
            return f"{random.randint(10, 999)}$"
        if missing_char == "Z":
            return random.choice(["Zebra", "Zone", "Size", "Pizza", "Z"])
        if missing_char == "J":
            return random.choice(["Jack", "Just", "Job", "J"])
        if missing_char == "W":
            return random.choice(["Wow", "Web", "Wifi", "W"])
        if missing_char == "F":
            return random.choice(["Facebook", "Fix", "F"])

        # Fallback cuối: ghép ký tự vào cuối một từ ngẫu nhiên để đảm bảo xuất hiện.
        w1 = random.choice(vocab)
        return f"{w1} {missing_char}"


def render_line_image(text, font_paths, height):
    """Render một dòng text thành ảnh grayscale với augmentation nhẹ.

    Các điểm chính:
    - Random font + random font size để đa dạng hoá.
    - Nền trắng/ghi nhạt, chữ đen/xám đậm.
    - Thêm noise Gaussian và blur nhẹ theo xác suất để mô phỏng ảnh chụp/scan.

    Args:
        text (str): Nội dung cần render.
        font_paths (list): Danh sách đường dẫn font.
        height (int): Chiều cao ảnh đầu ra.

    Returns:
        np.ndarray | None: Ảnh grayscale dạng numpy hoặc None nếu không render được.
    """
    font_path = random.choice(font_paths)
    font_size = int(height * random.uniform(0.5, 0.8))

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        return None

    # Tính bounding box của text để ước lượng kích thước canvas.
    bbox = font.getbbox(text)
    if not bbox:
        return None

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Tạo chiều rộng có padding để tránh text sát biên.
    img_w = text_w + random.randint(40, 100)

    # Nền trắng/ghi nhạt để mô phỏng nền giấy.
    bg_color = random.randint(230, 255)
    image = Image.new("L", (img_w, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Canh giữa và tạo offset nhỏ để tránh "quá chuẩn".
    x = (img_w - text_w) // 2
    y = (height - text_h) // 2 - (text_h * 0.1)

    # Màu chữ đen/xám đậm.
    text_color = random.randint(0, 50)
    draw.text((x, y), text, font=font, fill=text_color)

    # Chuyển sang numpy để áp dụng augmentation bằng OpenCV/Numpy.
    np_img = np.array(image)

    # Thêm noise Gaussian nhẹ.
    if random.random() < 0.7:
        noise = np.random.normal(0, 5, np_img.shape).astype("uint8")
        np_img = np.clip(np_img.astype(int) - noise, 0, 255).astype("uint8")

    # Blur nhẹ để mô phỏng mất nét.
    if random.random() < 0.3:
        kernel_size = random.choice([3])
        np_img = cv2.GaussianBlur(np_img, (kernel_size, kernel_size), 0)

    return np_img


def main():
    """Điểm vào chính: xác định ký tự thiếu, sinh text, render ảnh và lưu labels."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1) Xác định tập ký tự đầy đủ (mục tiêu) và tập hiện có, sau đó lấy phần thiếu.
    full_set = load_charset(PRETRAIN_ALPHABET, is_json=False)
    curr_set = load_charset(REAL_CHARSET_JSON, is_json=True)

    # Token đặc biệt được loại khỏi danh sách ký tự cần bổ sung.
    full_set.update(["[SOS]", "[EOS]", "[PAD]"])
    missing_chars = sorted(list(full_set - curr_set))
    missing_chars = [c for c in missing_chars if c not in ["[SOS]", "[EOS]", "[PAD]"]]

    if not missing_chars:
        print("[INFO] Không thiếu ký tự nào.")
        return

    print(f"[INFO] Tìm thấy {len(missing_chars)} ký tự thiếu: {missing_chars}")

    # 2) Nạp từ vựng từ dataset thật để làm từ đệm, giúp câu có ngữ cảnh tự nhiên hơn.
    vocab = load_vocabulary(WORD_LABEL_SRC)

    print(f"[INFO] Đang sinh {NUM_LINES} dòng văn bản (Render bằng Font)...")

    new_labels = {}

    # 3) Sinh dữ liệu: mỗi dòng chọn ngẫu nhiên một ký tự thiếu làm trọng tâm.
    for i in tqdm(range(NUM_LINES)):
        target_char = random.choice(missing_chars)
        target_word = create_target_text(target_char, vocab)

        # Tạo 2-4 từ đệm để ghép thành "câu" ngắn.
        num_fillers = random.randint(2, 4)
        fillers = random.choices(vocab, k=num_fillers)

        # Trộn từ và ghép thành một dòng text.
        words_in_line = fillers + [target_word]
        random.shuffle(words_in_line)
        full_text = " ".join(words_in_line)
        full_text = normalize_text(full_text)

        # Render text thành ảnh; nếu thất bại thì bỏ qua mẫu này.
        img_np = render_line_image(full_text, FONTS, IMG_H)
        if img_np is None:
            continue

        # Lưu ảnh và labels theo tên file chuẩn hoá.
        fname = f"synth_line_{i:05d}.jpg"
        save_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(save_path, img_np)

        new_labels[fname] = full_text

    # 4) Ghi labels JSON cho dữ liệu synthetic.
    with open(OUTPUT_LABEL, "w", encoding="utf-8") as f:
        json.dump(new_labels, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Đã tạo {len(new_labels)} ảnh tại '{OUTPUT_DIR}'.")
    print(f" -> Các ảnh này là dòng câu hoàn chỉnh, chứa các ký tự thiếu '{missing_chars}'.")


if __name__ == "__main__":
    main()
