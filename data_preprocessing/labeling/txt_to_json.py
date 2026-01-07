"""
Tạo file labels JSON (filename -> text) từ một thư mục chứa cặp ảnh + file .txt.

Mục tiêu:
- Duyệt toàn bộ file .txt trong `folder_path`.
- Với mỗi file .txt:
  - Đọc nội dung text (UTF-8) và strip khoảng trắng đầu/cuối.
  - Tìm file ảnh có cùng basename với một trong các đuôi hợp lệ.
  - Nếu tìm thấy ảnh tương ứng, thêm entry vào dict kết quả:
    { "<image_filename>": "<content>" }.
- Ghi dict ra file JSON tại `output_json_path`.

Ghi chú:
- Script hiện chỉ kiểm tra ảnh trong cùng một thư mục (không duyệt đệ quy).
- Danh sách đuôi ảnh cho phép gồm cả dạng viết hoa để tương thích dữ liệu thực tế.
- Phần `__main__` thêm đường dẫn cha vào sys.path để import `config` và lấy base path.
"""

import os
import json
import sys


def create_json_labels(folder_path, output_json_path):
    """Tạo labels JSON từ thư mục ảnh + file nhãn text.

    Args:
        folder_path (str): Đường dẫn thư mục chứa ảnh và các file .txt.
        output_json_path (str): Đường dẫn file JSON đầu ra để lưu nhãn.
    """
    data = {}
    valid_exts = ['.png', '.jpg', '.jpeg', '.JPG', '.PNG']

    # Kiểm tra thư mục dữ liệu đầu vào.
    if not os.path.exists(folder_path):
        print(f"Lỗi: '{folder_path}' không tồn tại.")
        return

    print(f"Xử lý: {folder_path} ...")
    all_files = os.listdir(folder_path)
    text_files = [f for f in all_files if f.endswith('.txt')]
    count = 0

    # Duyệt từng file .txt và tìm ảnh cùng basename.
    for text_file in text_files:
        base_name = os.path.splitext(text_file)[0]
        text_path = os.path.join(folder_path, text_file)

        # Đọc nội dung nhãn; bỏ qua file lỗi để tránh dừng toàn bộ batch.
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except:
            continue

        # Tìm ảnh ứng với basename theo danh sách extension hợp lệ.
        image_found = None
        for ext in valid_exts:
            if (base_name + ext) in all_files:
                image_found = base_name + ext
                break

        # Nếu có ảnh, thêm entry vào labels.
        if image_found:
            data[image_found] = content
            count += 1

    # Ghi file JSON đầu ra.
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Xong! Lưu {count} nhãn tại '{output_json_path}'.\n")
    except Exception as e:
        print(f"Lỗi ghi JSON: {e}")


if __name__ == "__main__":
    # Thêm thư mục cha để có thể import `config` theo cấu trúc dự án.
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

    # Cấu hình batch: mỗi phần tử tương ứng một thư mục con cần tạo labels JSON.
    base = config.OCR_DATASET_DIR
    batch_config = [
        {
            "folder": os.path.join(base, "en_00"),
            "output": os.path.join(base, "labels_en00.json"),
        }
    ]

    # Chạy lần lượt các batch theo cấu hình.
    for conf in batch_config:
        create_json_labels(conf["folder"], conf["output"])
