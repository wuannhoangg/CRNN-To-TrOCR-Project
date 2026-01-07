"""
Di chuyển tập con ảnh theo danh sách key trong một file labels JSON sang thư mục đích.

Luồng xử lý:
- Đọc file JSON `INPUT_JSON_LIST` (mapping: filename -> text).
- Với mỗi key (đường dẫn ảnh tương đối hoặc filename):
  - Tìm file nguồn trong `SOURCE_DIR` theo 2 cách:
    1) Giữ nguyên `img_rel_path` (trường hợp SOURCE_DIR có cấu trúc thư mục con).
    2) Fallback basename (trường hợp ảnh bị "flatten" vào 1 thư mục).
  - Tạo thư mục đích tương ứng trong `DEST_DIR` và di chuyển ảnh sang đó.
- In ra tổng số ảnh di chuyển thành công.

Ghi chú:
- Script không cập nhật lại labels JSON; mục đích là tách/chuẩn hoá tập ảnh vật lý.
- Nếu file JSON không đọc được, script sẽ thoát sớm (return).
- Nếu file đích đã tồn tại, `shutil.move` có thể ghi đè/hoặc lỗi tuỳ hệ điều hành.
"""

import json
import os
import shutil
from tqdm import tqdm
import sys

# Bổ sung thư mục cha để import `config` theo cấu trúc dự án.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# File JSON chứa danh sách ảnh cần lấy (key là đường dẫn/filename ảnh).
INPUT_JSON_LIST = os.path.join(config.OCR_DATASET_DIR, "labels_pretrain_500k.json")

# Thư mục nguồn đang chứa ảnh lộn xộn (có thể có cấu trúc thư mục con hoặc bị flatten).
SOURCE_DIR = os.path.join(config.OCR_DATASET_DIR, "merged_images")

# Thư mục đích (được coi là "sạch") để đưa ảnh pretrain vào.
DEST_DIR = config.PRETRAIN_DATA_DIR


def main():
    """Điểm vào chính: đọc danh sách từ JSON và di chuyển ảnh sang thư mục đích."""
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    print(f"[INFO] Đọc list từ: {INPUT_JSON_LIST}")
    try:
        with open(INPUT_JSON_LIST, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        # Nếu không đọc được JSON thì dừng để tránh xử lý sai.
        return

    print(f"[INFO] Di chuyển {len(data)} ảnh sang '{DEST_DIR}'...")
    success = 0

    # Duyệt theo danh sách key trong JSON; chỉ quan tâm key vì value là text nhãn.
    for img_rel_path in tqdm(data.keys()):
        src = os.path.join(SOURCE_DIR, img_rel_path)
        dst = os.path.join(DEST_DIR, img_rel_path)

        # Tạo thư mục cha cho file đích để đảm bảo move không bị lỗi do thiếu folder.
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.exists(src):
            shutil.move(src, dst)
            success += 1
        else:
            # Fallback khi ảnh không nằm theo cấu trúc thư mục mà bị gom phẳng (flatten).
            flat_src = os.path.join(SOURCE_DIR, os.path.basename(img_rel_path))
            if os.path.exists(flat_src):
                shutil.move(flat_src, dst)
                success += 1

    print(f"Đã chuyển: {success} ảnh.")


if __name__ == "__main__":
    main()
