# TrOCR_VN_Handwriting — Cải tiến OCR chữ viết tay tiếng Việt với TrOCR + KenLM + DBNet

## 1) Tổng quan
Dự án tập trung cải tiến hệ thống OCR chữ viết tay tiếng Việt theo hướng **End-to-End (E2E)**:

- **Nhận dạng (Recognition):** Vision Transformer OCR (TrOCR-style) là mô hình chính.
- **Hậu xử lý ngôn ngữ:** **KenLM (n-gram)** được tích hợp trong **Beam Search** để tối ưu chuỗi đầu ra.
- **Phát hiện văn bản (Detection):** **DBNet (OpenCV DNN, ONNX)** dùng để tách dòng chữ khi input là ảnh “full page”.
- **Baseline đối chứng:** CRNN + CTC.

> Nền tảng mã nguồn ban đầu tham khảo từ CRNN baseline (TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR) và được mở rộng thành pipeline hoàn chỉnh.

---

## 2) Demo (đã deploy)
Pipeline hoàn chỉnh (Detection + Recognition + LM) được deploy tại Hugging Face Spaces:

- **Demo:** https://huggingface.co/spaces/wuann/TrOCR_VN_Handwritting

---

## 3) Kết quả (tham khảo)
Kết quả trên tập test độc lập (1k2 ảnh test):

| Mô hình | CER (↓) | WER (↓) |
|---|---:|---:|
| CRNN (Baseline) | 21.54% | 52.21% |
| TrOCR (Cải tiến) | 10.88% | 26.90% |

---

## 4) Cấu trúc thư mục (tóm tắt)
```
CRNN-TO-TROCR-PROJECT/
├── configs/                 # YAML configs (training/eval)
├── data/                    # Dữ liệu
│   └── README.md
├── data_preprocessing/      # Script tiền xử lý, tạo nhãn, tổ chức dataset
│   └── README.md
├── models/                  # Checkpoint + LM + DBNet 
│   └── README.md
├── src/
│   ├── crnn/                # CRNN baseline (train/eval/predict)
│   ├── transformer/         # Transformer OCR + LM (train/eval/predict)
│   └── utils/               # Dataset, decoding, preprocessing, metrics...
├── requirements.txt         # Thư viện cho training/eval (không gồm PyTorch)
└── README.md                
```

---

## 5) Cài đặt môi trường

### 5.1 Yêu cầu
- Python **3.10–3.11** (khuyến nghị 3.11 nếu bạn đang dùng `.venv-3.11`).
- Windows/Linux đều hỗ trợ (lệnh bên dưới minh họa Windows).
- (Tuỳ chọn) GPU NVIDIA + CUDA để tăng tốc.

### 5.2 Tạo môi trường ảo và cài thư viện
Từ thư mục gốc dự án:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### 5.3 Cài PyTorch (CPU/GPU)
Dự án **không cố định phiên bản torch trong `requirements.txt`** để bạn chủ động chọn đúng build (CPU/CUDA).

- **GPU:** cài theo hướng dẫn chính thức tại: https://pytorch.org/get-started/locally/
- **CPU:** có thể cài trực tiếp:
```bash
pip install torch torchvision
```

---

## 6) Chuẩn bị Data và Models

### 6.1 Data
- Làm theo `data/README.md` để tải/giải nén và tạo cấu trúc dữ liệu.
- Tối thiểu cần: thư mục ảnh (`images_dir`) và file nhãn JSON (`labels_json`).

### 6.2 Models (checkpoint + LM + DBNet)
- Tải theo hướng dẫn trong `models/README.md`.
- Link Google Drive:
  - https://drive.google.com/drive/folders/1kS9x2vLasqhu5VRu-GuxH5Wh9MxmH9-r?usp=sharing



---

## 7) Cấu hình YAML (configs/)
Tinh chỉnh lại config trước khi train cho phù hợp với môi trường local trên máy. Chi tiết xem ở file config ở folder config.

---

## 8) Huấn luyện (Training)

> Tất cả lệnh dưới đây chạy từ **thư mục gốc** dự án.

### 8.1 Train / Fine-tune Transformer OCR
```bash
python -m src.transformer.train --config "configs/transformer_config.yml"
```

Fine-tune từ checkpoint có sẵn:
```bash
python -m src.transformer.train --config "configs/transformer_config.yml" --resume_from "models/best_transformer.pt"
```

### 8.2 Train CRNN baseline
```bash
python -m src.crnn.train ^
  --images_dir "C:/.../train_images" ^
  --labels_json "C:/.../labels.json" ^
  --output_dir "models/checkpoints/crnn" ^
  --device cuda --amp
```

---

## 9) Đánh giá (Evaluation)

### 9.1 Đánh giá Transformer + KenLM (Beam Search + LM)
Lệnh chạy phụ thuộc đúng tên tham số trong `src/transformer/eval_lm.py` (khuyến nghị xem `--help` nếu bạn đã chỉnh sửa script). Ví dụ:

```bash
python -m src.transformer.eval_lm ^
  --checkpoint "models/best_transformer.pt" ^
  --test_images_dir "data/test/images" ^
  --test_labels_json "data/test/labels.json" ^
  --lm_path "models/3-gram-lm.binary" ^
  --beam_width 10 --lm_alpha 0.5 --lm_beta 0.2 ^
  --output_file "outputs_eval/transformer_preds.json"
```

### 9.2 Đánh giá CRNN (CTC)
```bash
python -m src.crnn.eval ^
  --weights "models/best_crnn.pt" ^
  --images_dir "data/test/images" ^
  --labels_json "data/test/labels.json" ^
  --device cuda --amp ^
  --out_dir "outputs_eval/crnn"
```

---

## 10) Suy luận (Inference / Predict)

### 10.1 Predict 1 ảnh (Transformer + KenLM + DBNet)
Script `src/transformer/predict_lm.py` tự động chọn chế độ:
- Nếu ảnh **đã crop sẵn 1 dòng** → bỏ qua detection.
- Nếu ảnh **full page** → chạy DBNet để cắt dòng rồi nhận dạng từng dòng.

Ví dụ chạy:
```bash
python -m src.transformer.predict_lm ^
  --image "data/test/page_01.jpg" ^
  --ocr_checkpoint "models/best_transformer.pt" ^
  --lm_model "models/3-gram-lm.binary" ^
  --det_model "models/DB_TD500_resnet50.onnx" ^
  --beam_width 10 --lm_alpha 0.5 --lm_beta 0.2 ^
  --output_file "outputs_eval/page_01.txt"
```

### 10.2 Predict 1 ảnh (CRNN)
```bash
python -m src.crnn.predict ^
  --weights "models/best_crnn.pt" ^
  --image "data/test/line_01.jpg" ^
  --device cuda --amp
```

---

## 11) Data preprocessing
Toàn bộ script tiền xử lý nằm trong `data_preprocessing/` (check charset, gộp nhãn, đổi tên batch, gộp dataset, sinh dữ liệu tổng hợp...).  
Xem hướng dẫn chi tiết tại: `data_preprocessing/README.md`.

---


## 13) Tham chiếu
- CRNN baseline: TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR  
- Một phần dữ liệu tham khảo: nghiangh/UIT-HWDB-dataset và các nguồn tổng hợp khác theo báo cáo.
