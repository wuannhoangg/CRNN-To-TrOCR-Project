"""
Gộp nhiều bộ dữ liệu OCR (ảnh + labels JSON) về một thư mục đích duy nhất và tạo labels tổng.

Luồng xử lý tổng quát:
- Với mỗi dataset nguồn trong `SOURCE_DATASETS`:
  1) Đọc file nhãn JSON (mapping: filename -> text).
  2) Với mỗi entry:
     - Tìm ảnh tương ứng trong `img_dir` (ưu tiên đường dẫn trực tiếp, fallback basename).
     - Sinh tên file mới theo chuẩn đánh số tăng dần: img_{index:07d}{ext}.
     - Thực hiện move/copy ảnh sang `OUTPUT_DIR`.
     - Ghi nhãn vào dict tổng với key là tên mới.
- Sau khi xử lý tất cả dataset, lưu `final_labels` ra `OUTPUT_JSON`.

Ghi chú vận hành:
- Tên ảnh đầu ra được chuẩn hoá để tránh trùng lặp giữa các nguồn.
- Biến `MODE` cho phép chọn 'move' (mặc định) hoặc 'copy' để giữ dữ liệu gốc.
- Script không tự xử lý trường hợp file đích đã tồn tại (có thể gây lỗi khi move/copy).
"""

import os
import json
import shutil
from tqdm import tqdm
import sys

# Bổ sung thư mục cha để import module cấu hình dự án.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ================= CẤU HÌNH TỪ CONFIG =================
# Danh sách các dataset nguồn cần gộp. Mỗi phần tử gồm thư mục ảnh và đường dẫn labels JSON.
SOURCE_DATASETS = [
    {
        "dir": os.path.join(config.OCR_DATASET_DIR, "synthetic_lines_stitched"),
        "label": os.path.join(config.OCR_DATASET_DIR, "labels_stitched.json"),
    },
    {
        "dir": config.PRETRAIN_DATA_DIR,
        "label": os.path.join(config.OCR_DATASET_DIR, "labels_pretrain_500k.json"),
    },
]

# Thư mục đích và file labels tổng đầu ra.
OUTPUT_DIR = config.FINAL_DATASET_DIR
OUTPUT_JSON = config.FINAL_LABEL_FILE

# Chế độ xử lý file: 'move' để di chuyển, 'copy' để sao chép.
MODE = "move"
# ======================================================


def process_dataset(img_dir, json_file, start_index, final_dict):
    """Xử lý một dataset nguồn và cập nhật labels tổng.

    Args:
        img_dir (str): Thư mục chứa ảnh nguồn.
        json_file (str): Đường dẫn file labels JSON của dataset nguồn.
        start_index (int): Chỉ số bắt đầu để đặt tên ảnh đầu ra.
        final_dict (dict): Dict labels tổng được cập nhật theo tên file mới.

    Returns:
        int: Chỉ số tiếp theo sau khi xử lý xong dataset hiện tại.
    """
    print(f"\n[INFO] Xử lý: {json_file} ...")
    if not os.path.exists(json_file):
        print(f"[WARN] Không thấy file nhãn: {json_file}")
        return start_index

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    current_idx = start_index
    count_ok = 0

    # Duyệt từng entry nhãn và tìm ảnh tương ứng để gom vào thư mục đích.
    for old_fname, text in tqdm(data.items()):
        src_path = os.path.join(img_dir, old_fname)
        if not os.path.exists(src_path):
            # Fallback: dùng basename để xử lý trường hợp labels lưu path thay vì filename.
            src_path = os.path.join(img_dir, os.path.basename(old_fname))

        if os.path.exists(src_path):
            # Giữ extension gốc; nếu không có extension thì fallback ".jpg".
            ext = os.path.splitext(old_fname)[1] or ".jpg"
            new_fname = f"img_{current_idx:07d}{ext}"
            dst_path = os.path.join(OUTPUT_DIR, new_fname)

            try:
                # Move/Copy theo MODE đã cấu hình.
                if MODE == "move":
                    shutil.move(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)

                # Cập nhật labels tổng theo tên file mới.
                final_dict[new_fname] = text
                current_idx += 1
                count_ok += 1
            except Exception as e:
                # Báo lỗi và tiếp tục để không làm gián đoạn batch lớn.
                print(f"[ERR] {e}")

    return current_idx


def main():
    """Điểm vào chính: tạo thư mục đích, lần lượt xử lý từng dataset và ghi labels tổng."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    final_labels = {}
    current_index = 1

    # Xử lý lần lượt các nguồn, đảm bảo index tăng liên tục xuyên suốt.
    for dataset in SOURCE_DATASETS:
        current_index = process_dataset(dataset["dir"], dataset["label"], current_index, final_labels)

    print(f"\n[INFO] Lưu JSON tổng: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_labels, f, ensure_ascii=False, indent=2)

    print(f"HOÀN TẤT! Tổng {len(final_labels)} ảnh.")


if __name__ == "__main__":
    main()
