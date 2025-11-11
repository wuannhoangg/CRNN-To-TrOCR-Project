# Đồ án: Cải tiến OCR chữ viết tay tiếng Việt với TrOCR và Hậu xử lý Ngôn ngữ
[cite_start]Dự án này là đồ án môn học, nhằm mục tiêu cải tiến hệ thống nhận dạng chữ viết tay tiếng Việt bằng cách[cite: 3]:

1.  [cite_start]Sử dụng mô hình Transformer (TrOCR) làm mô hình chính[cite: 3].
2.  [cite_start]Tích hợp mô hình ngôn ngữ (KenLM) để hậu xử lý, tăng độ chính xác[cite: 3, 9].
3.  [cite_start]Xây dựng pipeline End-to-End (E2E) với Text Detection (DBNet)[cite: 3, 9].

Dự án được xây dựng và cải tiến dựa trên mã nguồn CRNN baseline của **TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR**.

-----

## 🚀 Demo Trực Tiếp (Web Application)

[cite_start]Bạn có thể trải nghiệm pipeline hoàn chỉnh (Detection + Recognition + Language Model) tại web demo do nhóm triển khai[cite: 6, 9].

**Link Demo:** [https://huggingface.co/spaces/wuann/TrOCR_Demo](https://huggingface.co/spaces/wuann/TrOCR_Demo)
-----

## 📊 Kết quả So sánh

[cite_start]Kết quả đánh giá trên tập test độc lập cho thấy mô hình Transformer (TrOCR) kết hợp với Hậu xử lý Ngôn ngữ (LM) cho kết quả vượt trội so với baseline CRNN[cite: 6, 9].

| Mô hình | CER (Lỗi Ký tự) ⬇️ | WER (Lỗi Từ) ⬇️ |
| :--- | :---: | :---: |
| CRNN (Baseline) | *9.56%* | *27.52%* |
| **TrOCR (Cải tiến)** | *9.01%* | *19.43%* |


-----

## 📁 Cấu trúc thư mục

```
vn-handwriting-ocr/
├── configs/                # Chứa file .yml config cho training
├── data/                   # Nơi chứa dữ liệu (BỊ GIT BỎ QUA)
│   ├── alphabet_vi_full.txt
│   └── README.md           # Hướng dẫn tải data
├── models/                 # Nơi chứa checkpoint (BỊ GIT BỎ QUA)
│   └── README.md           # Hướng dẫn tải model
├── src/                    # TOÀN BỘ CODE HUẤN LUYỆN & ĐÁNH GIÁ
│   ├── crnn/               # Module cho model CRNN baseline
│   ├── transformer/        # Module cho model TrOCR cải tiến
│   └── utils/              # Các hàm dùng chung (metrics, dataset,...)
│
├── web_demo/               # CODE DEMO (FastAPI + Docker)
│   ├── app/                # Code FastAPI (main.py, pipeline.py)
│   ├── models/             # Models dùng cho demo (được LFS theo dõi)
│   ├── Dockerfile
│   └── requirements.txt
│
├── .gitignore              # File bỏ qua của Git
├── README.md               # File này
└── requirements.txt        # Thư viện cho huấn luyện (src/)
```

-----

## ⚙️ Cài đặt & Hướng dẫn sử dụng

### A. Chuẩn bị (Bắt buộc)

1.  **Clone dự án:**

    ```bash
    git clone https://github.com/ten-ban/vn-handwriting-ocr
    cd vn-handwriting-ocr
    ```

2.  **Cài đặt Git LFS (để tải model cho `web_demo`):**

    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Thiết lập môi trường (Windows):**

      * (Tùy chọn) Sửa lỗi hiển thị UTF-8 trên CMD:
        ```cmd
        chcp 65001 >NUL
        set PYTHONIOENCODING=utf-8
        ```
      * Kích hoạt môi trường ảo (ví dụ của bạn):
        ```cmd
        .\.venv-py311\Scripts\activate
        ```
      * Cài đặt thư viện cho Huấn luyện & Đánh giá:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Tải Dữ liệu (Data):**

      * **ĐỌC KỸ:** `data/README.md`.
      * Bạn cần tải và giải nén dữ liệu vào thư mục `data/` trước khi train.

5.  **Tải Model (Checkpoints):**

      * **ĐỌC KỸ:** `models/README.md`.
      * Bạn cần tải các model đã huấn luyện (ví dụ: `best_transformer.pt`, `best_crnn.pt`, `3-gram-lm.binary`) và đặt vào thư mục `models/`.

### B. Huấn luyện (Training)

(Các lệnh được chạy từ thư mục gốc `vn-handwriting-ocr/`)

#### 1\. Huấn luyện Transformer (TrOCR)

Chạy training bằng cách gọi module `src.transformer.train`:

```cmd
python -m src.transformer.train ^
    --config "configs/transformer_config.yml" ^
    --resume_from "models/best_transformer.pt"
```

**Chú thích tham số:**

  * `--config`: (Bắt buộc) Chỉ định file config YAML chứa mọi cài đặt (đường dẫn data, learning rate, batch size...).
  * `--resume_from`: (Tùy chọn)
      * [cite_start]**Để huấn luyện tiếp (fine-tune):** Dùng tham số này và trỏ đến file `best_transformer.pt` đã có[cite: 9].
      * **Để huấn luyện từ đầu (from scratch):** **Xóa** tham số này đi.

#### 2\. Huấn luyện CRNN (Baseline)

Chạy training bằng cách gọi module `src.crnn.train`:

```cmd
python -m src.crnn.train ^
    --images_dir "data/images" ^
    --labels_json "data/labels.json" ^
    --output_dir "models/checkpoints_crnn_new" ^
    --device cuda --amp
```

**Chú thích tham số:**

  * `--images_dir`, `--labels_json`: Đường dẫn đến dữ liệu huấn luyện.
  * `--output_dir`: Thư mục để lưu checkpoint `best.pt` mới.
  * `--device cuda --amp`: Tăng tốc training nếu bạn có GPU (khuyến khích).

### C. Đánh giá (Evaluation)

(Giả sử bạn đã đặt file test trong `data/test/images` và `data/test/labels.json`)

#### 1\. Đánh giá Transformer (TrOCR) + Language Model

Chạy đánh giá bằng cách gọi module `src.transformer.eval_lm`:

```cmd
python -m src.transformer.eval_lm ^
    --checkpoint "models/best_transformer.pt" ^
    --test_images_dir "data/test/images" ^
    --test_labels_json "data/test/labels.json" ^
    --lm_path "models/3-gram-lm.binary" ^
    --output_file "evaluation_results.json" ^
    --beam_width 10 ^
    --lm_alpha 0.5 ^
    --lm_beta 0.2
```

**Chú thích tham số (Rất quan trọng):**

  * `--checkpoint`: Trỏ đến file model TrOCR (`.pt`) bạn muốn đánh giá.
  * `--lm_path`: Trỏ đến file model ngôn ngữ (`.binary`).
  * `--output_file`: (Tùy chọn) Lưu kết quả dự đoán chi tiết ra file JSON.
  * `--beam_width 10`: Tăng số "beam" cho kết quả tốt hơn (nhưng chậm hơn).
  * `--lm_alpha 0.5`, `--lm_beta 0.2`:
      * `lm_alpha`: Sức mạnh của model ngôn ngữ (cao hơn = ưu tiên ngữ pháp hơn).
      * `lm_beta`: Thưởng/phạt cho độ dài (word count).
      * [cite_start]**Ghi chú:** Qua kiểm thử (Task C3), bộ tham số `alpha=0.5` và `beta=0.2` cho kết quả CER/WER cân bằng và tốt nhất trên tập test[cite: 9].

#### 2\. Đánh giá CRNN (Baseline)

Chạy đánh giá bằng cách gọi module `src.crnn.eval`:

```cmd
python -m src.crnn.eval ^
    --weights "models/best_crnn.pt" ^
    --images_dir "data/test/images" ^
    --labels_json "data/test/labels.json" ^
    --device cuda --amp ^
    --out_dir "outputs_eval_crnn"
```

**Chú thích tham số:**

  * `--weights`: Trỏ đến file model CRNN (`.pt`) bạn muốn đánh giá.
  * `--out_dir`: Nơi lưu file kết quả `preds.csv` và `metrics.txt`.

### D. Dự đoán 1 ảnh (Predict)

#### 1\. Dự đoán (Transformer + LM)

Chạy dự đoán 1 ảnh bằng cách gọi module `src.transformer.predict_lm`:

```cmd
python -m src.transformer.predict_lm ^
    --checkpoint "models/best_transformer.pt" ^
    --image "data/test/images/t1.jpg" ^
    --lm_path "models/3-gram-lm.binary" ^
    --beam_width 10 ^
    --lm_alpha 0.5 ^
    --lm_beta 0.5
```

**Chú thích tham số:**

  * `--image`: Đường dẫn đến ảnh bạn muốn dự đoán.
  * `--lm_alpha 0.5`, `--lm_beta 0.5`: Tham số alpha/beta khi dự đoán 1 ảnh có thể cần tinh chỉnh khác với khi đánh giá hàng loạt (ví dụ: `0.5`/`0.5`).

#### 2\. Dự đoán (CRNN)

Chạy dự đoán 1 ảnh bằng cách gọi module `src.crnn.predict`:

```cmd
python -m src.crnn.predict ^
    --weights "models/best_crnn.pt" ^
    --image "data/test/images/15520_samples.jpg" ^
    --device cuda --amp
```

### E. Chạy Web Demo (Local)

1.  **Chạy bằng Docker (Khuyến khích):**

      * (Đảm bảo bạn đã `git lfs pull` để có model trong `web_demo/models/`)

    ```bash
    cd web_demo
    docker build -t ocr-app .
    docker run -p 8000:8000 ocr-app
    ```

2.  **Chạy thủ công (Local):**

      * Cài đặt các thư viện riêng của web demo:
        ```bash
        pip install -r web_demo/requirements.txt
        ```
      * Chạy server FastAPI:
        ```bash
        # Chạy từ thư mục gốc vn-handwriting-ocr/
        python -m uvicorn web_demo.app.main:app --host 0.0.0.0 --port 8000
        ```
      * Mở trình duyệt tại: `http://localhost:8000`

-----

## Attribute

  * **Baseline CRNN:** [TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR](https://github.com/TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR)
  * **Dữ liệu:** [nghiangh/UIT-HWDB-dataset](https://github.com/nghiangh/UIT-HWDB-dataset) và các nguồn khác.
