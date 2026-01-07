# src/transformer/train.py
# -*- coding: utf-8 -*-

"""
Huấn luyện mô hình Vision Transformer OCR.

Luồng chính:
1) Nạp cấu hình từ YAML và hợp nhất với cấu hình mặc định.
2) Chuẩn bị dữ liệu: đọc labels, quét ảnh, chia train/val, tạo DataLoader.
3) Khởi tạo mô hình và (tuỳ chọn) nạp trọng số từ checkpoint để fine-tune.
4) Huấn luyện theo teacher forcing với CrossEntropyLoss (ignore PAD).
5) Validation gồm:
   - Tính val loss (teacher forcing).
   - Inference autoregressive bằng beam search để tính CER/WER.
6) Lưu checkpoint tốt nhất theo val loss.

Usage:
    python -m src.transformer.train --config configs/base.yaml
"""

import os
import argparse
import json
import random
import pathlib
import math
import time
import unicodedata
import yaml
from typing import List, Dict, Tuple, Any, Optional
from functools import partial

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ..utils.preprocessing import (
    gpu_check,
    ensure_dir,
    read_labels,
    read_alphabet,
    resize_keep_height,
    gaussian_blur_and_adapt_thresh,
    calc_max_padded_width,
    apply_augs_pil,
)
from ..utils.decoding import beam_search_decode

from .model import VisionTransformerOCR
from ..utils.dataset import TransformerDataset as OCRDataset, transformer_collate


DEFAULT_CONFIG = {
    'data': {
        'alphabet': None,
        'img_height': 118,
        'simple_preprocess': False,
    },
    'training': {
        'transformer': {
            'batch_size': 16,
            'epochs': 50,
            'lr': 3e-4,
            'test_size': 0.1,
        }
    },
    'evaluation': {
        'beam_width': 3,
        'beam_alpha': 0.7,
    },
    'seed': 42,
}


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Hợp nhất đệ quy hai dictionary cấu hình.

    Các giá trị trong `override_config` sẽ ghi đè lên `base_config`.

    Args:
        base_config (Dict): Cấu hình gốc (mặc định).
        override_config (Dict): Cấu hình ghi đè (từ YAML).

    Returns:
        Dict: Cấu hình đã hợp nhất.
    """
    for key, value in override_config.items():
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            merge_configs(base_config[key], value)
        else:
            base_config[key] = value
    return base_config


def cer_wer_ser(pred_texts: List[str], gt_texts: List[str]) -> Tuple[float, float, float]:
    """Tính CER/WER/SER dựa trên Levenshtein distance.

    Args:
        pred_texts (List[str]): Danh sách chuỗi dự đoán.
        gt_texts (List[str]): Danh sách chuỗi ground truth.

    Returns:
        Tuple[float, float, float]: (CER, WER, SER) theo trung bình.
    """

    def lev(a, b):
        """Tính khoảng cách Levenshtein (edit distance)."""
        n, m = len(a), len(b)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                tmp = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = tmp
        return dp[m]

    cer, wer, ser = [], [], []
    for pd, gt in zip(pred_texts, gt_texts):
        if not gt:
            continue

        cer.append(lev(list(pd.lower()), list(gt.lower())) / len(gt))

        pw, gw = pd.lower().split(), gt.lower().split()
        if len(gw) > 0:
            wer.append(lev(pw, gw) / len(gw))

        ser.append(0.0 if pd == gt else 1.0)

    return (
        float(np.mean(cer if cer else [1.0])),
        float(np.mean(wer if wer else [1.0])),
        float(np.mean(ser if ser else [1.0])),
    )


def main():
    """Điểm vào chính: nạp config, chuẩn bị dữ liệu, huấn luyện và lưu checkpoint."""
    ap = argparse.ArgumentParser(description="Train Vision Transformer OCR Model (v5.5 Stable + YAML/Alphabet)")
    ap.add_argument('--config', type=str, required=True, help='Đường dẫn file cấu hình YAML.')

    ap.add_argument('--images_dir', type=str, default=None, help='Ghi đè đường dẫn thư mục ảnh.')
    ap.add_argument('--labels_json', type=str, default=None, help='Ghi đè đường dẫn file labels.')
    ap.add_argument('--output_dir', type=str, default=None, help='Ghi đè thư mục lưu output.')
    ap.add_argument('--resume_from', type=str, default=None, help='Đường dẫn checkpoint để train tiếp.')
    ap.add_argument('--lr', type=float, default=None, help='Ghi đè Learning Rate.')
    ap.add_argument('--epochs', type=int, default=None, help='Ghi đè số Epochs.')
    ap.add_argument('--batch_size', type=int, default=None, help='Ghi đè Batch Size.')
    args = ap.parse_args()

    # Nạp YAML config và hợp nhất với cấu hình mặc định.
    print(f"[INFO] Loading configuration from: {args.config}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"YAML config file not found: {args.config}")
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
    except Exception as e:
        raise IOError(f"Failed to read/parse YAML config file {args.config}: {e}")

    config = merge_configs(DEFAULT_CONFIG.copy(), yaml_config)

    # Tổng hợp tham số thực thi: ưu tiên CLI > YAML.
    cli_args = {}
    cli_args['images_dir'] = args.images_dir or config['data'].get('images_dir')
    cli_args['labels_json'] = args.labels_json or config['data'].get('labels_json')
    cli_args['output_dir'] = args.output_dir or config['training'].get('output_dir', './outputs_transformer_yaml')
    cli_args['resume_from'] = args.resume_from

    cfg_data = config.get('data', {})
    cfg_train = config.get('training', {}).get('transformer', {})
    cfg_eval = config.get('evaluation', {})
    cfg_seed = config.get('seed', 42)

    cli_args['alphabet_path'] = cfg_data.get('alphabet')
    cli_args['height'] = cfg_data.get('img_height', 118)
    cli_args['simple_preprocess'] = cfg_data.get('simple_preprocess', False)
    cli_args['test_size'] = cfg_train.get('test_size', 0.1)
    cli_args['batch_size'] = args.batch_size or cfg_train.get('batch_size', 16)
    cli_args['epochs'] = args.epochs or cfg_train.get('epochs', 50)

    try:
        cli_args['lr'] = float(args.lr or cfg_train.get('lr', 3e-4))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid learning rate value: {args.lr or cfg_train.get('lr')}")

    cli_args['beam_width'] = cfg_eval.get('beam_width', 3)
    cli_args['beam_alpha'] = cfg_eval.get('beam_alpha', 0.7)

    if not cli_args['images_dir']:
        raise ValueError("Config/CLI error: 'images_dir' is required.")
    if not cli_args['labels_json']:
        raise ValueError("Config/CLI error: 'labels_json' is required.")
    if not cli_args['alphabet_path']:
        raise ValueError("Config error: 'data.alphabet' path is required.")

    # Thiết lập seed cho reproducibility.
    random.seed(cfg_seed)
    np.random.seed(cfg_seed)
    torch.manual_seed(cfg_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg_seed)

    device = torch.device(gpu_check())
    ensure_dir(cli_args['output_dir'])

    # Chuẩn bị vocabulary và checkpoint (nếu fine-tune/resume).
    best_val_loss = float('inf')

    base_char_list = read_alphabet(cli_args['alphabet_path'])
    special_tokens = ['[SOS]', '[EOS]', '[PAD]']
    char_list = special_tokens + base_char_list

    if cli_args['resume_from'] and os.path.exists(cli_args['resume_from']):
        print(f"[INFO] Loading weights from checkpoint: {cli_args['resume_from']}")
        ckpt = torch.load(cli_args['resume_from'], map_location=device)
        if 'charset' in ckpt and set(char_list) != set(ckpt['charset']):
            print("[WARN] Checkpoint charset mismatch with alphabet file! Re-initializing classification head.")
        print("[INFO] Starting new fine-tuning run from epoch 1 (Reset Optimizer).")
    else:
        ckpt = None

    char_to_idx = {c: i for i, c in enumerate(char_list)}
    idx_to_char = {i: c for i, c in char_to_idx.items()}
    SOS_IDX, EOS_IDX, PAD_IDX = char_to_idx['[SOS]'], char_to_idx['[EOS]'], char_to_idx['[PAD]']
    num_classes = len(char_list)
    print(f'[INFO] Vocabulary size: {num_classes}')

    # Chuẩn bị dữ liệu và dataloader.
    labels_map = read_labels(cli_args['labels_json'], normalize=True)

    image_paths = [
        str(p) for p in pathlib.Path(cli_args['images_dir']).glob('**/*')
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    ]
    image_paths = [p for p in image_paths if os.path.basename(p) in labels_map]
    if not image_paths:
        raise ValueError("No valid image paths found matching the labels JSON.")

    train_paths, val_paths = train_test_split(
        image_paths,
        test_size=cli_args['test_size'],
        random_state=cfg_seed,
    )

    # Chọn pad_w: ưu tiên pad_w trong checkpoint để đảm bảo tương thích.
    if ckpt and 'pad_w' in ckpt:
        pad_w = ckpt['pad_w']
        print(f'[INFO] Using padded width from checkpoint: {pad_w}')
    else:
        pad_w = calc_max_padded_width(image_paths, cli_args['height'])
    print(f'[INFO] Max padded width = {pad_w}')

    train_ds = OCRDataset(train_paths, labels_map, cli_args['height'], pad_w, is_training=True)
    val_ds = OCRDataset(val_paths, labels_map, cli_args['height'], pad_w, is_training=False)

    collate_fn = partial(
        transformer_collate,
        char_to_idx=char_to_idx,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX,
        pad_idx=PAD_IDX,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cli_args['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cli_args['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    # Khởi tạo model và nạp trọng số (strict=False) nếu fine-tune.
    model = VisionTransformerOCR(num_classes=num_classes).to(device)
    if ckpt:
        print("[INFO] Loading weights with vocabulary change...")
        checkpoint_state_dict = ckpt['model_state']
        keys_to_remove = ['embedding.weight', 'fc_out.weight', 'fc_out.bias']
        filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k not in keys_to_remove}
        try:
            model.load_state_dict(filtered_state_dict, strict=False)
            print("[INFO] Model weights loaded successfully (strict=False).")
        except Exception as e:
            print(f"[ERROR] Failed to load weights even after filtering: {e}")

    # Optimizer, scheduler, loss.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cli_args['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Vòng lặp huấn luyện.
    print(f"[INFO] Starting new run from epoch 1 up to {cli_args['epochs']} epochs.")
    for epoch in range(1, cli_args['epochs'] + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cli_args['epochs']} [Train]", leave=False)
        for batch_data in train_pbar:
            if batch_data[0] is None:
                continue
            imgs, src_key_padding_mask, targets, _ = batch_data
            if imgs is None or imgs.numel() == 0:
                continue

            imgs = imgs.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            targets = targets.to(device)

            # Teacher forcing: decoder input là chuỗi dịch phải, expected là chuỗi dịch trái.
            tgt_input, tgt_expected = targets[:, :-1], targets[:, 1:]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            tgt_padding_mask = (tgt_input == PAD_IDX)

            optimizer.zero_grad()

            # AMP: chọn bfloat16 nếu phần cứng hỗ trợ, ngược lại dùng float16.
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                logits = model(imgs, src_key_padding_mask, tgt_input, tgt_mask, tgt_padding_mask)
                loss = criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt_expected.reshape(-1),
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss
            train_pbar.set_postfix({'loss': f"{current_loss:.4f}"})

        train_loss = 0.0 if len(train_loader) == 0 else running_loss / len(train_loader)

        # Validation: vừa tính loss, vừa decode autoregressive để tính CER/WER.
        model.eval()
        val_loss_sum = 0.0
        pred_texts, gt_texts = [], []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{cli_args['epochs']} [Val]", leave=False)
        with torch.no_grad():
            for batch_data in val_pbar:
                if batch_data[0] is None:
                    continue
                imgs, src_key_padding_mask, targets, gts = batch_data
                if imgs is None or imgs.numel() == 0:
                    continue

                imgs = imgs.to(device)
                src_key_padding_mask = src_key_padding_mask.to(device)
                targets = targets.to(device)

                tgt_input, tgt_expected = targets[:, :-1], targets[:, 1:]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                tgt_padding_mask = (tgt_input == PAD_IDX)

                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    logits = model(imgs, src_key_padding_mask, tgt_input, tgt_mask, tgt_padding_mask)
                    loss_v = criterion(
                        logits.reshape(-1, logits.shape[-1]),
                        tgt_expected.reshape(-1),
                    )
                val_loss_sum += loss_v.item()

                memory = model.encode(imgs, src_key_padding_mask)

                hyps = beam_search_decode(
                    model,
                    memory,
                    src_key_padding_mask,
                    SOS_IDX,
                    EOS_IDX,
                    150,
                    cli_args['beam_width'],
                    cli_args['beam_alpha'],
                    repetition_penalty=1.2,
                )

                for seq in hyps:
                    pred_indices = seq[1:]
                    try:
                        eos_pos = pred_indices.index(EOS_IDX)
                        pred_indices = pred_indices[:eos_pos]
                    except ValueError:
                        pass
                    pred_texts.append("".join(idx_to_char.get(idx, "") for idx in pred_indices))
                gt_texts.extend(gts)

        val_loss = val_loss_sum / len(val_loader)
        val_cer, val_wer, _ = cer_wer_ser(pred_texts, gt_texts)

        epoch_mins, epoch_secs = divmod(time.time() - start_time, 60)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch}/{cli_args['epochs']} | Time: {int(epoch_mins)}m {int(epoch_secs)}s | "
            f"LR: {current_lr:.1e} | TLoss: {train_loss:.4f} | VLoss: {val_loss:.4f} | "
            f"CER: {val_cer:.4f} | WER: {val_wer:.4f}"
        )

        # Lưu checkpoint tốt nhất theo val loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'charset': char_list,
                'height': cli_args['height'],
                'pad_w': pad_w,
                'best_val_loss': best_val_loss,
                'simple_preprocess': cli_args['simple_preprocess'],
            }
            save_path = os.path.join(cli_args['output_dir'], 'best_transformer.pt')
            torch.save(ckpt, save_path)

            with open(os.path.join(cli_args['output_dir'], 'charset.json'), 'w', encoding='utf-8') as f:
                json.dump(char_list, f, ensure_ascii=False, indent=2)

            print(f" -> Saved best checkpoint (Loss: {best_val_loss:.4f}) to {save_path}")


if __name__ == '__main__':
    main()
