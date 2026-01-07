# src/crnn/eval.py
"""
Script đánh giá (eval) cho mô hình CRNN OCR tiếng Việt.

Chức năng chính:
- Nạp checkpoint đã train (weights).
- Tiền xử lý ảnh test giống pipeline lúc train.
- Chạy inference + giải mã CTC (greedy decode) để ra chuỗi ký tự.
- Tính các chỉ số CER/WER/SER và lưu dự đoán ra CSV.
"""

import os, json, argparse, csv, pathlib, unicodedata
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.preprocessing import (
    read_labels,
    resize_keep_height,
    gaussian_blur_and_adapt_thresh
)
from ..utils.decoding import greedy_decode
from ..utils.metrics import calculate_metrics as metrics

from .model import CRNN


def preprocess_image(path, height, pad_w):
    """Đọc và tiền xử lý 1 ảnh (grayscale) để đưa vào CRNN.

    Pipeline:
    - Đọc ảnh dạng grayscale bằng OpenCV.
    - Resize giữ nguyên tỉ lệ, ép về chiều cao `height`.
    - Nếu width < pad_w thì pad (mode='median'); nếu lớn hơn thì resize về (pad_w, height).
    - Áp dụng làm mờ + adaptive threshold (giống lúc train).
    - Chuẩn hoá về float32 [0, 1] và thêm chiều channel.

    Args:
        path (str): Đường dẫn ảnh input.
        height (int): Chiều cao target (giống khi train).
        pad_w (int): Chiều rộng đã pad/cố định (giống khi train).

    Returns:
        torch.Tensor: Tensor ảnh shape (1, H, W), dtype float32.

    Raises:
        FileNotFoundError: Nếu OpenCV không đọc được ảnh.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    # Resize giữ chiều cao cố định, giữ nguyên tỉ lệ ảnh.
    img = resize_keep_height(img, height)
    h, w = img.shape

    # Đảm bảo đầu vào luôn có width = pad_w (đúng định dạng model mong đợi).
    if w < pad_w:
        img = np.pad(img, ((0, 0), (0, pad_w - w)), mode='median')
    else:
        img = cv2.resize(img, (pad_w, height))

    # Tiền xử lý ảnh (blur + adaptive threshold) để tăng tương phản ký tự.
    img = gaussian_blur_and_adapt_thresh(img)

    # Chuẩn hoá và thêm chiều channel: (1, H, W).
    img = (img.astype(np.float32) / 255.0)[None, ...]  # (1,H,W)
    return torch.from_numpy(img)                       # (1,H,W)


def main():
    """Hàm main: parse args, load model, chạy eval, lưu kết quả.

    Checkpoint kỳ vọng có các key (tối thiểu):
    - model_state: state_dict của model
    - charset: danh sách ký tự dùng encode/decode
    - height: chiều cao ảnh lúc train (không có thì default 118)
    - pad_w: chiều rộng pad/cố định lúc train (không có thì default 2167)
    - blank_id: index của ký tự blank cho CTC (không có thì default len(charset))
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="outputs_torch/best.pt")
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--labels_json", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out_dir", type=str, default="./outputs_eval")
    ap.add_argument("--norm_accent", action="store_true")
    ap.add_argument("--norm_punct", action="store_true")
    args = ap.parse_args()

    # Chọn device: ưu tiên CUDA nếu người dùng chọn cuda và máy có GPU.
    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")

    # Tạo thư mục output nếu chưa có.
    os.makedirs(args.out_dir, exist_ok=True)

    # Load checkpoint lên CPU trước để an toàn, sau đó chuyển model sang device mong muốn.
    ckpt = torch.load(args.weights, map_location="cpu")

    # Lấy cấu hình từ checkpoint (nếu thiếu thì dùng default).
    charset  = ckpt.get("charset")
    height   = int(ckpt.get("height", 118))
    pad_w    = int(ckpt.get("pad_w", 2167))
    blank_id = int(ckpt.get("blank_id", len(charset)))

    # Tổng số lớp = số ký tự + 1 lớp blank cho CTC.
    num_classes = len(charset) + 1

    # Bảng ánh xạ index -> ký tự để dựng chuỗi output.
    idx_to_char = {i:c for i,c in enumerate(charset)}

    # Khởi tạo model và nạp trọng số.
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Đọc labels từ JSON và map theo basename của file ảnh.
    labels_map = read_labels(args.labels_json, normalize=False)
    labels_map = {os.path.basename(k): v for k,v in labels_map.items()}

    # Danh sách phần mở rộng file ảnh chấp nhận.
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}

    # Quét ảnh trong thư mục test (không đệ quy).
    images = []
    for p in pathlib.Path(args.images_dir).glob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(str(p))
    images.sort()

    # Lọc ra các ảnh có nhãn tương ứng trong labels_map.
    imgs_kept, gts = [], []
    for path in images:
        name = os.path.basename(path)
        # Điều kiện (labels_map[name] hoặc labels_map[name] == "") để giữ cả nhãn rỗng.
        if name in labels_map and (labels_map[name] or labels_map[name]==""):
            imgs_kept.append(path)
            gts.append(labels_map[name])

    # Nếu không có cặp (ảnh, nhãn) nào khớp thì dừng.
    if not imgs_kept:
        raise SystemExit("[ERR] Không tìm thấy cặp (ảnh, nhãn) khớp trong thư mục test.")

    print(f"[INFO] Test items: {len(imgs_kept)} | height={height} pad_w={pad_w} | device={device}")

    preds = []
    rows  = []  # Lưu từng dòng để ghi CSV: filename, gt, pred

    # AMP chỉ có ý nghĩa khi chạy trên GPU.
    use_amp = (device.type=="cuda" and args.amp)

    with torch.no_grad():
        for path, gt in zip(imgs_kept, gts):
            # Tiền xử lý ảnh và thêm batch dimension: (B=1, C=1, H, W).
            x = preprocess_image(path, height, pad_w).unsqueeze(0).to(device) # (B=1,1,H,W)

            # Chạy model với autocast (nếu bật AMP trên GPU) để tăng tốc/giảm RAM.
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits = model(x) # (T,B=1,C)
                log_probs = F.log_softmax(logits, dim=-1)

            # Giải mã CTC theo greedy: lấy chuỗi index và bỏ blank/duplicate theo hàm greedy_decode.
            seq = greedy_decode(log_probs, blank_id=blank_id)[0]

            # Ghép index -> ký tự thành chuỗi dự đoán.
            pred = "".join(idx_to_char.get(i,"") for i in seq)

            preds.append(pred)
            rows.append([os.path.basename(path), gt, pred])

    # Tính metric: CER/WER/SER (có thể bật chuẩn hoá dấu/câu).
    cer, wer, ser = metrics(preds, gts, norm_accent=args.norm_accent, norm_punct=args.norm_punct)
    print("========== EVAL ==========")
    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")
    print(f"SER: {ser:.4f}")

    # Ghi file CSV dự đoán để dễ inspect từng sample.
    csv_path = os.path.join(args.out_dir, "preds.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename","ground_truth","prediction"])
        w.writerows(rows)
    print(f"[OK] Saved predictions -> {csv_path}")

    # Ghi file txt lưu metrics và các option normalize.
    with open(os.path.join(args.out_dir,"metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([
            f"items={len(rows)}",
            f"CER={cer}",
            f"WER={wer}",
            f"SER={ser}",
            f"norm_accent={args.norm_accent}",
            f"norm_punct={args.norm_punct}",
        ]))
    print(f"[OK] Saved metrics -> {os.path.join(args.out_dir,'metrics.txt')}")


if __name__ == "__main__":
    # Điểm vào chương trình khi chạy từ CLI: python -m src.crnn.eval ...
    main()
