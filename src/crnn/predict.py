# src/crnn/predict.py
import os, argparse, json, pathlib
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Import từ thư mục utils ---
from ..utils.preprocessing import (
    resize_keep_height,
    gaussian_blur_and_adapt_thresh
)
from ..utils.decoding import greedy_decode

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='outputs_torch/best.pt')
    ap.add_argument('--image', type=str, help='Đường dẫn 1 ảnh để nhận dạng')
    ap.add_argument('--images_dir', type=str, help='Hoặc thư mục ảnh để nhận dạng hàng loạt')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--amp', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if args.device=='cuda' and torch.cuda.is_available() else 'cpu')

    # Load checkpoint đã train
    ckpt = torch.load(args.weights, map_location='cpu')
    charset  = ckpt.get('charset')
    height   = int(ckpt.get('height', 118))
    pad_w    = int(ckpt.get('pad_w', 2167))
    blank_id = int(ckpt.get('blank_id', len(charset)))

    num_classes = len(charset) + 1
    idx_to_char = {i:c for i,c in enumerate(charset)}

    model = CRNN(num_classes=num_classes).to(device) # Đã import
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval()

    # Gom danh sách ảnh (giữ y nguyên)
    paths=[]
    if args.image:
        paths=[args.image]
    elif args.images_dir:
        for p in pathlib.Path(args.images_dir).glob('*'):
            if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.gif'}:
                paths.append(str(p))
    else:
        raise SystemError('Cần truyền --image hoặc --images_dir')

    with torch.no_grad():
        use_amp = (device.type=='cuda' and args.amp)
        for p in paths:
            x = preprocess_image(p, height, pad_w).unsqueeze(0).to(device)  # (B=1,1,H,W)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                logits = model(x)                      # (T,B=1,C)
                log_probs = F.log_softmax(logits, -1)
            seq = greedy_decode(log_probs, blank_id=blank_id)[0] # Đã import
            pred = ''.join(idx_to_char.get(i, '') for i in seq)
            print(f'{os.path.basename(p)} => {pred}')

if __name__ == '__main__':
    main()