# Thư mục Dữ liệu

Thư mục này chứa các tài nguyên dữ liệu cần thiết cho dự án **OCR chữ viết tay tiếng Việt**.

> **LƯU Ý QUAN TRỌNG:** Do kích thước lớn, thư mục `images/` và file `labels.json` **KHÔNG** được đưa lên Git. Bạn phải tải chúng về và đặt thủ công theo cấu trúc bên dưới.

-----

## 📁 Cấu trúc thư mục

Để các kịch bản (script) huấn luyện và đánh giá chạy đúng, cấu trúc thư mục `data/` phải như sau:

```
data/
├── alphabet_vi_full.txt     # Bảng chữ cái (mỗi dòng 1 ký tự, UTF-8)
├── images/                  # ẢNH TRAIN/VAL (PNG/JPG/JPEG)
│   ├── image_0001.png
│   ├── image_0002.jpg
│   └── ...
├── labels.json              # NHÃN TRAIN/VAL (map từ tên file → ground-truth)
│
├── test/                    # (Tùy chọn) Bộ test độc lập
│   ├── images/
│   │   ├── t_image_0001.png
│   │   └── ...
│   └── labels.json
│
└── README.md                # File hướng dẫn này
```

**Ghi chú về file:**

  * **`alphabet_vi_full.txt`**: Định nghĩa bộ ký tự cho model.

      * Mỗi dòng là **một** ký tự.
      * Các script trong dự án này (như `vn_ocr_transformer_v5.5.py`) đã được thiết kế để xử lý **khoảng trắng thật** (" "), không phải token `<space>`.

  * **`labels.json`**: File JSON map từ tên file ảnh (basename) sang nhãn (ground-truth).

      * Nhãn phải được chuẩn hóa Unicode **NFC**.

    <!-- end list -->

    ```json
    {
      "image_0001.png": "Xin chào thế giới",
      "image_0002.jpg": "Tôi yêu học máy"
    }
    ```

-----

## ⬇️ Nguồn & Tải về

### Nguồn dữ liệu gốc

Bộ dữ liệu này được tổng hợp và chuẩn hoá từ nhiều nguồn công khai, bao gồm:

1.  **TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR**: [https://github.com/TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR/](https://github.com/TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR/)
2.  **nghiangh/UIT-HWDB-dataset**: [https://github.com/nghiangh/UIT-HWDB-dataset](https://github.com/nghiangh/UIT-HWDB-dataset)

### Dữ liệu đã xử lý (Download)

Để tiện lợi, bạn có thể tải các bộ dữ liệu đã được gộp, làm sạch và chuẩn hóa sẵn từ link dưới đây:

  * **Train/Val (đã gộp & chuẩn hoá)**:
    [https://drive.google.com/file/d/1qQ-r0ZERGDyHSkgJn3kZ7J3NAdKK7sBe/view?usp=sharing](https://drive.google.com/file/d/1qQ-r0ZERGDyHSkgJn3kZ7J3NAdKK7sBe/view?usp=sharing)
    **Link Label cho tập Train/Val**:
    https://drive.google.com/file/d/1FCw5bKxCDbxw7WiH-HnB4WwUSfavluTm/view?usp=sharing

  * **Test set (độc lập)**:
    [https://drive.google.com/file/d/1muDyBA_11GSnAVI19jURIRLPd6oVMbWG/view?usp=sharing](https://drive.google.com/file/d/1muDyBA_11GSnAVI19jURIRLPd6oVMbWG/view?usp=sharing)

**Cách sử dụng:**
Tải các file nén từ link trên và giải nén vào thư mục `data/` này.

```bash
# Ví dụ
unzip trainval.zip -d data/
unzip test.zip -d data/
```

-----

## ✅ Kiểm tra Dữ liệu (Sanity Check)

Sau khi giải nén, hãy chạy đoạn script Python sau từ thư mục gốc (`vn-handwriting-ocr/`) để kiểm tra tính toàn vẹn của dữ liệu.

*Script này sẽ kiểm tra:*

1.  File nhãn và alphabet có tồn tại không.
2.  Ảnh có bị thiếu so với nhãn không.
3.  Các nhãn có bị rỗng không.
4.  Có ký tự nào trong nhãn nằm ngoài bảng chữ cái (`alphabet_vi_full.txt`) không.

<!-- end list -->

```bash
python - <<'PY'
import os, json, unicodedata, io

DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
ALPH = os.path.join(DATA_DIR, "alphabet_vi_full.txt")
LABELS = os.path.join(DATA_DIR, "labels.json")

print(f"--- Đang kiểm tra thư mục: {DATA_DIR} ---")

if not os.path.exists(ALPH):
    print(f"[LỖI] Không tìm thấy file alphabet: {ALPH}")
    exit()
if not os.path.exists(LABELS):
    print(f"[LỖI] Không tìm thấy file nhãn: {LABELS}")
    exit()
if not os.path.exists(IMG_DIR):
    print(f"[LỖI] Không tìm thấy thư mục ảnh: {IMG_DIR}")
    exit()

with io.open(ALPH, "r", encoding="utf-8") as f:
    alphabet = f.read().splitlines()
alphabet = [(" " if ch == "<space>" else ch) for ch in alphabet]
charset = set(alphabet)
if " " not in charset:
    print("[WARN] Alphabet chưa có khoảng trắng. Thêm ' ' vào danh sách.")

with io.open(LABELS, "r", encoding="utf-8") as f:
    labels = json.load(f)

missing_imgs, bad_keys, empty_labels = [], [], []
for k, v in labels.items():
    if os.path.basename(k) != k:
        bad_keys.append(k)
    if not v.strip():
        empty_labels.append(k)
    if not os.path.exists(os.path.join(IMG_DIR, k)):
        missing_imgs.append(k)

oov = set()
for v in labels.values():
    v = unicodedata.normalize("NFC", v)
    for ch in v:
        if ch not in charset:
            oov.add(ch)

print(f"[OK] Số nhãn: {len(labels)}")
print(f"[CHECK] Thiếu ảnh: {len(missing_imgs)}")
print(f"[CHECK] Key không phải basename: {len(bad_keys)}")
print(f"[CHECK] Nhãn rỗng: {len(empty_labels)}")
if oov:
    print(f"[!!WARN!!] Ký tự ngoài alphabet: {len(oov)} -> {sorted(list(oov))[:50]}")
else:
    print("[OK] Toàn bộ ký tự đều nằm trong alphabet.")
print("--- Kiểm tra hoàn tất ---")
PY
```

-----

## Attribution
Nếu bạn sử dụng bộ dữ liệu đã được tổng hợp/chuẩn hoá này, vui lòng trích dẫn (cite) repo của dự án này và các nguồn dữ liệu gốc (đã liệt kê ở Mục B).
