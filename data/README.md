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
├── test/                    # Bộ test độc lập
│   ├── images/
│   │   ├── t_image_0001.png
│   │   └── ...
│   
│── labels_test.json
└── README.md                
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

Bộ dữ liệu này được tổng hợp và chuẩn hoá từ nhiều nguồn công khai cùng các dữ liệu tự viết tay, bao gồm:

1.  **TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR**: [https://github.com/TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR/](https://github.com/TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR/)
2.  **nghiangh/UIT-HWDB-dataset**: [https://github.com/nghiangh/UIT-HWDB-dataset](https://github.com/nghiangh/UIT-HWDB-dataset)
3.  **pbcquoc/vietocr** https://github.com/pbcquoc/vietocr?tab=readme-ov-file


### Dữ liệu đã xử lý (Download)

Để tiện lợi, bạn có thể tải các bộ dữ liệu đã được gộp, làm sạch và chuẩn hóa sẵn từ link dưới đây:

  * **Data 3 giai doan**:
    [https://drive.google.com/drive/folders/1r9xUVx5-SSoBBR7uUHyQNg_ze8bevslG?usp=sharing](https://drive.google.com/file/d/1qQ-r0ZERGDyHSkgJn3kZ7J3NAdKK7sBe/view?usp=sharing)

**Cách sử dụng:**
Tải các file nén từ link trên và giải nén vào thư mục `data/` này.

```bash
# Ví dụ
unzip trainval.zip -d data/
unzip test.zip -d data/
```

-----


## Attribution
Nếu bạn sử dụng bộ dữ liệu đã được tổng hợp/chuẩn hoá này, vui lòng trích dẫn (cite) repo của dự án này và các nguồn dữ liệu gốc (đã liệt kê ở Mục B).
