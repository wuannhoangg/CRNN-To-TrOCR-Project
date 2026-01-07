# src/crnn/train.py
"""
Huấn luyện mô hình CRNN cho bài toán OCR theo CTC.

Luồng chính:
- Đọc nhãn từ labels.json và xây dựng charset.
- Quét ảnh trong thư mục dữ liệu, chia train/val.
- Tính pad_w để chuẩn hoá width đầu vào.
- Huấn luyện với CTCLoss, lưu checkpoint mỗi epoch và lưu best checkpoint.
- Cuối cùng chạy suy luận trên tập validation và tính CER/WER/SER.
"""

import os
import argparse
import json
import random
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ..utils.preprocessing import (
    gpu_check,
    ensure_dir,
    read_labels,
    build_char_list,
    calc_max_padded_width,
    auto_find_labels_json,
)
from ..utils.dataset import CTCDataset, ctc_collate
from ..utils.metrics import calculate_metrics
from ..utils.decoding import greedy_decode

from .model import CRNN


def main():
    """Parse tham số CLI và thực thi toàn bộ pipeline huấn luyện/evaluate."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--images_dir',
        type=str,
        default=r"C:\Users\admin\HCMUT\DATH\TrOCR_DATH\data",
        help='Thư mục chứa ảnh huấn luyện/đánh giá.',
    )
    ap.add_argument(
        '--labels_json',
        type=str,
        default=r"C:\Users\admin\HCMUT\DATH\TrOCR_DATH\labels.json",
        help='Có thể truyền file labels.json hoặc thư mục chứa nó.',
    )
    ap.add_argument('--output_dir', type=str, default='./outputs_torch')
    ap.add_argument('--height', type=int, default=118)
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--cpu', action='store_true', help='Ép chạy CPU (không dùng GPU).')
    ap.add_argument('--require_gpu', action='store_true', help='Báo lỗi nếu không có GPU.')
    ap.add_argument('--seed', type=int, default=42, help='Seed để tái lập kết quả.')
    ap.add_argument('--device', type=str, default='cuda', help='cuda hoặc cpu.')
    ap.add_argument('--amp', action='store_true', help='Dùng mixed precision để tăng tốc/giảm VRAM.')
    args = ap.parse_args()

    # Thiết lập seed cho reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Kiểm tra GPU/CPU theo cấu hình người dùng.
    mode = gpu_check(force_cpu=args.cpu, require_gpu=args.require_gpu)
    use_cuda = (mode in ('gpu', 'cuda') and args.device == 'cuda' and torch.cuda.is_available())
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'[INFO] Running on: {device}')

    # Tạo thư mục output (checkpoint, log) nếu chưa tồn tại.
    ensure_dir(args.output_dir)

    # Tự động dò labels.json nếu người dùng truyền vào thư mục hoặc đường dẫn không hợp lệ.
    if (not args.labels_json.lower().endswith('.json')) or (not os.path.isfile(args.labels_json)):
        auto_lbl = auto_find_labels_json(args.labels_json) or auto_find_labels_json(args.images_dir)
        if auto_lbl:
            print(f'[INFO] Dùng file nhãn phát hiện: {auto_lbl}')
            args.labels_json = auto_lbl
        else:
            raise FileNotFoundError('Không tìm thấy labels.json. Truyền --labels_json "Đường\\đến\\labels.json"')

    print(f'[INFO] labels_json = {args.labels_json}')

    # CRNN dùng label gốc (không normalize) để huấn luyện theo charset thực tế.
    labels_map = read_labels(args.labels_json, normalize=False)

    # Xây dựng charset và các bảng ánh xạ ký tự <-> index cho CTC.
    char_list = build_char_list(labels_map)
    num_classes = len(char_list) + 1
    blank_id = num_classes - 1
    char_to_idx = {c: i for i, c in enumerate(char_list)}
    idx_to_char = {i: c for i, c in enumerate(char_list)}

    # Quét toàn bộ ảnh trong images_dir (đệ quy).
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif'}
    images = []
    base_path = pathlib.Path(args.images_dir)
    if not base_path.is_dir():
        raise FileNotFoundError(f'images_dir không phải thư mục: {args.images_dir}')
    for item in base_path.glob('**/*'):
        if item.is_file() and item.suffix.lower() in image_exts:
            images.append(str(item))
    if not images:
        raise FileNotFoundError(f'Không tìm thấy ảnh trong {args.images_dir}')
    print(f'[INFO] Found {len(images)} images')

    # Chia tập train/validation.
    train_paths, val_paths = train_test_split(images, test_size=args.test_size, random_state=args.seed)

    # Tính pad_w để chuẩn hoá width đầu vào trên cả train/val.
    pad_w = calc_max_padded_width(train_paths + val_paths, args.height)
    print(f'[INFO] dynamic pad width = {pad_w}')

    # Khởi tạo dataset và dataloader cho CTC training.
    train_ds = CTCDataset(train_paths, labels_map, char_to_idx, args.height, pad_w)
    val_ds = CTCDataset(val_paths, labels_map, char_to_idx, args.height, pad_w)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=ctc_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=ctc_collate,
    )

    # Khởi tạo model, optimizer, loss và AMP scaler (nếu bật).
    model = CRNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.amp))

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        # ------------------------
        # Training phase
        # ------------------------
        model.train()
        running = 0.0
        for imgs, targets, target_lengths, _ in train_loader:
            # Bỏ qua batch rỗng/bị lỗi từ collate.
            if imgs.numel() == 0:
                continue

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward + CTC loss (có thể dùng autocast nếu bật AMP).
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and args.amp)):
                logits = model(imgs)  # (T, B, C)
                log_probs = F.log_softmax(logits, dim=-1)
                T, B, C = log_probs.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # Backward + cập nhật trọng số (kèm gradient scaling nếu AMP).
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        # ------------------------
        # Validation phase
        # ------------------------
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for imgs, targets, target_lengths, _ in val_loader:
                if imgs.numel() == 0:
                    continue

                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                target_lengths = target_lengths.to(device, non_blocking=True)

                logits = model(imgs)
                log_probs = F.log_softmax(logits, dim=-1)
                T, B, C = log_probs.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                vloss += loss.item()

        val_loss = vloss / max(1, len(val_loader))

        print(f'Epoch {epoch}/{args.epochs} | train_ctc: {train_loss:.4f} | val_ctc: {val_loss:.4f}')

        # ------------------------
        # Checkpointing
        # ------------------------
        ckpt = {
            'model_state': model.state_dict(),
            'charset': char_list,
            'height': args.height,
            'pad_w': pad_w,
            'blank_id': blank_id,
            'epoch': epoch,
        }

        torch.save(ckpt, os.path.join(args.output_dir, f'crnn_ep{epoch:03d}.pt'))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.output_dir, 'best.pt'))
            with open(os.path.join(args.output_dir, 'charset.json'), 'w', encoding='utf-8') as f:
                json.dump(char_list, f, ensure_ascii=False, indent=2)
            print('  -> Saved best checkpoint')

    # ------------------------
    # Post-train evaluation (greedy decode)
    # ------------------------
    print('[INFO] Running validation inference...')
    pred_texts = []
    gt_texts = []
    with torch.no_grad():
        for imgs, _, _, texts in val_loader:
            if imgs.numel() == 0:
                continue

            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)  # (T, B, C)

            # Lưu ý: greedy_decode trong dự án này nhận logits/log_probs tuỳ implementation.
            preds = greedy_decode(logits, blank_id=blank_id)

            for seq, gt in zip(preds, texts):
                pred = ''.join(idx_to_char.get(i, '') for i in seq)
                pred_texts.append(pred)
                gt_texts.append(gt)

    CER, WER, SER = calculate_metrics(pred_texts, gt_texts)
    print('--- Metrics ---')
    print(f'CER: {CER}')
    print(f'WER: {WER}')
    print(f'SER: {SER}')


if __name__ == '__main__':
    main()
