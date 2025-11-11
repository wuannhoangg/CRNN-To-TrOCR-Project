# src/crnn/eval.py
import os, json, argparse, csv, pathlib, unicodedata
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Import từ thư mục utils ---
from ..utils.preprocessing import (
    read_labels,
    resize_keep_height,
    gaussian_blur_and_adapt_thresh
)
from ..utils.decoding import greedy_decode
from ..utils.metrics import calculate_metrics as metrics # Dùng `as` để giữ tên hàm gốc

# --- Import model từ file cùng cấp ---
from .model import CRNN

# Hàm này là logic riêng của eval/predict, ta giữ nó ở đây
def preprocess_image(path, height, pad_w):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = resize_keep_height(img, height) # Đã import
    h, w = img.shape
    if w < pad_w:
        img = np.pad(img, ((0,0),(0, pad_w - w)), mode='median')
    else:
        img = cv2.resize(img, (pad_w, height))
    img = gaussian_blur_and_adapt_thresh(img) # Đã import
    img = (img.astype(np.float32) / 255.0)[None, ...]  # (1,H,W)
    return torch.from_numpy(img)                       # (1,H,W)

# --- Main logic từ eval_ocr_torch.py ---
def main():
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

    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.weights, map_location="cpu")
    charset  = ckpt.get("charset")
    height   = int(ckpt.get("height", 118))
    pad_w    = int(ckpt.get("pad_w", 2167))
    blank_id = int(ckpt.get("blank_id", len(charset)))
    num_classes = len(charset) + 1
    idx_to_char = {i:c for i,c in enumerate(charset)}

    # Build model (đã import CRNN)
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Load labels (đã import read_labels)
    labels_map = read_labels(args.labels_json, normalize=False)
    labels_map = {os.path.basename(k): v for k,v in labels_map.items()}

    # Collect images
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
    images = []
    for p in pathlib.Path(args.images_dir).glob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(str(p))
    images.sort()

    # Intersect with labels
    imgs_kept, gts = [], []
    for path in images:
        name = os.path.basename(path)
        if name in labels_map and (labels_map[name] or labels_map[name]==""):
            imgs_kept.append(path)
            gts.append(labels_map[name])
    if not imgs_kept:
        raise SystemExit("[ERR] Không tìm thấy cặp (ảnh, nhãn) khớp trong thư mục test.")

    print(f"[INFO] Test items: {len(imgs_kept)} | height={height} pad_w={pad_w} | device={device}")

    preds = []
    rows  = []  # for CSV
    use_amp = (device.type=="cuda" and args.amp)

    with torch.no_grad():
        for path, gt in zip(imgs_kept, gts):
            x = preprocess_image(path, height, pad_w).unsqueeze(0).to(device) # (B=1,1,H,W)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits = model(x)                 # (T,B=1,C)
                log_probs = F.log_softmax(logits, dim=-1)
            seq = greedy_decode(log_probs, blank_id=blank_id)[0] # Đã import
            pred = "".join(idx_to_char.get(i,"") for i in seq)
            preds.append(pred)
            rows.append([os.path.basename(path), gt, pred])

    # Metrics (đã import metrics)
    cer, wer, ser = metrics(preds, gts, norm_accent=args.norm_accent, norm_punct=args.norm_punct)
    print("========== EVAL ==========")
    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")
    print(f"SER: {ser:.4f}")

    # Save CSV
    csv_path = os.path.join(args.out_dir, "preds.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename","ground_truth","prediction"])
        w.writerows(rows)
    print(f"[OK] Saved predictions -> {csv_path}")

    # Save metrics txt
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
    main()