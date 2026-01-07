"""
Kiểm tra mức độ bao phủ bộ ký tự (alphabet coverage) của dataset OCR.

Mục tiêu:
- Đọc tập ký tự mục tiêu từ file alphabet (theo định dạng mỗi dòng 1 ký tự,
  riêng '<space>' được quy ước là dấu cách).
- Quét toàn bộ nhãn trong file JSON (label_file) để thu tập ký tự thực tế xuất hiện.
- Báo cáo:
  - Các ký tự còn thiếu so với alphabet mục tiêu.
  - Các ký tự “lạ” xuất hiện trong dataset nhưng không nằm trong alphabet.

Ghi chú triển khai:
- Script thêm thư mục cha của file hiện tại vào sys.path để import `config`.
- `config` được kỳ vọng cung cấp:
  - FINAL_LABEL_FILE: đường dẫn tới file labels JSON (dict filename -> text).
  - ALPHABET_FILE: đường dẫn tới file alphabet.
"""

import sys
import os
import json
from tqdm import tqdm

# Thêm đường dẫn cha để import module `config` cùng cấp dự án.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_target_alphabet(path):
    """Đọc file alphabet và trả về tập ký tự mục tiêu.

    Quy ước:
    - Mỗi dòng là một ký tự.
    - Nếu gặp '<space>' thì ánh xạ thành ký tự ' '.
    - Luôn đảm bảo tập kết quả có chứa ký tự space.

    Args:
        path (str): Đường dẫn tới file alphabet.

    Returns:
        set: Tập ký tự mục tiêu. Nếu file không tồn tại, trả về set rỗng.
    """
    if not os.path.exists(path):
        print(f"[ERR] Không tìm thấy file alphabet: {path}")
        return set()

    chars = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            ch = line.strip()
            if ch:
                if ch == '<space>':
                    chars.add(' ')
                else:
                    chars.add(ch)

    if ' ' not in chars:
        chars.add(' ')

    return chars


def main():
    """Chạy kiểm tra coverage: target alphabet vs ký tự xuất hiện trong dataset."""
    # Đọc đường dẫn từ cấu hình dự án.
    label_file = config.FINAL_LABEL_FILE
    alphabet_file = config.ALPHABET_FILE

    print(f"[1] Đang đọc file mục tiêu: {alphabet_file}")
    target_set = load_target_alphabet(alphabet_file)
    print(f" -> Tổng số ký tự mong muốn: {len(target_set)}")

    print(f"\n[2] Đang quét file dữ liệu: {label_file}")
    if not os.path.exists(label_file):
        print("[ERR] File JSON không tồn tại!")
        return

    with open(label_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Tập ký tự thực tế xuất hiện trong toàn bộ nhãn của dataset.
    found_set = set()
    for text in tqdm(data.values(), desc="Scanning chars"):
        found_set.update(list(text))

    print(f" -> Tổng số ký tự thực tế tìm thấy: {len(found_set)}")

    print("\n" + "=" * 40)
    missing_chars = target_set - found_set
    extra_chars = found_set - target_set

    # Báo cáo coverage theo hướng phục vụ kiểm tra chất lượng dữ liệu.
    if len(missing_chars) == 0:
        print("Dataset đã bao phủ ĐỦ 100% bộ ký tự.")
    else:
        print(f"Thiếu {len(missing_chars)} ký tự!")
        print(f"   {sorted(list(missing_chars))}")

    if len(extra_chars) > 0:
        print(f"Có {len(extra_chars)} ký tự LẠ (ngoài alphabet).")
        print(f"   Ví dụ: {sorted(list(extra_chars))[:20]}")


if __name__ == "__main__":
    main()
