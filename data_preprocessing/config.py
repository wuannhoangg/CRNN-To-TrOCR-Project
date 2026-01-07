"""
Cấu hình đường dẫn (paths) dùng chung cho pipeline tiền xử lý dữ liệu OCR.

Mục tiêu:
- Tập trung hoá toàn bộ đường dẫn dataset, file cấu hình và tài nguyên (fonts)
  để các script trong `data_preprocessing/` và các module khác có thể import dùng lại.

Lưu ý sử dụng:
- Khi chuyển máy hoặc thay đổi vị trí dự án, chỉ cần cập nhật `BASE_PROJECT_PATH`.
- Các biến còn lại được xây dựng tương đối dựa trên `BASE_PROJECT_PATH` để giảm lỗi cấu hình.

Thành phần chính:
- Dataset paths: thư mục chứa dữ liệu OCR, dữ liệu pretrain, dữ liệu final.
- Config files: alphabet.txt (charset mục tiêu), charset.json hiện có để kiểm tra thiếu ký tự.
- Fonts: danh sách font hệ thống dùng cho các script sinh dữ liệu synthetic.
"""

import os

# --- ĐƯỜNG DẪN GỐC ---
# Thay đổi đường dẫn này thành folder dự án trên máy.
BASE_PROJECT_PATH = r"C:\Users\admin\HCMUT\DATH\TrOCR_DATH"

# --- DATASET PATHS ---
# Thư mục gốc chứa toàn bộ dataset OCR của dự án.
OCR_DATASET_DIR = os.path.join(BASE_PROJECT_PATH, "ocr_dataset")

# Folder chứa ảnh Pretrain sạch (đích đến của script move_subset).
PRETRAIN_DATA_DIR = os.path.join(OCR_DATASET_DIR, "pretrain_data_clean")

# Folder chứa ảnh đã gộp cuối cùng (ví dụ: bộ 1M samples).
FINAL_DATASET_DIR = os.path.join(OCR_DATASET_DIR, "1MData")

# File labels JSON tương ứng với FINAL_DATASET_DIR.
FINAL_LABEL_FILE = os.path.join(OCR_DATASET_DIR, "labels_1M_final.json")

# --- FILES CẤU HÌNH ---
# File alphabet định nghĩa bộ ký tự mục tiêu (dùng cho xây vocab/kiểm tra cover charset).
ALPHABET_FILE = os.path.join(BASE_PROJECT_PATH, "alphabet.txt")

# File charset thực tế sau khi train (dùng để check missing charset so với alphabet).
CURRENT_CHARSET_JSON = os.path.join(
    BASE_PROJECT_PATH,
    "outputs_transformer_v5_finetuned_with_aug",
    "charset.json",
)

# --- FONTS (Cho phần sinh dữ liệu) ---
# Danh sách font dùng để render chữ khi sinh dữ liệu synthetic.
# Khuyến nghị: đảm bảo các font này tồn tại trên Windows, hoặc thay bằng đường dẫn font bạn cài.
FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
    # Thêm font viết tay vào đây nếu có.
]
