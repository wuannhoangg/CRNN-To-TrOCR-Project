# src/transformer/predict_lm.py
import os
import argparse
import json
import math
from typing import List, Dict, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import kenlm
except ImportError:
    print("Lỗi: Gói 'kenlm' chưa được cài đặt.")
    exit()

# --- Import từ thư mục utils ---
from ..utils.preprocessing import (
    gpu_check,
    resize_keep_height,
    gaussian_blur_and_adapt_thresh
)
from ..utils.decoding import (
    get_word_from_indices,
    beam_search_decode_with_lm
)

# --- Import model từ file cùng cấp ---
from .model import PositionalEncoding, VisionTransformerOCR

# --- Main Prediction Logic ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint.')
    ap.add_argument('--image', type=str, required=True, help='Path to image.')
    ap.add_argument('--beam_width', type=int, default=30, help='Beam width for decoding.')
    
    ap.add_argument('--lm_path', type=str, required=True, help='Path to the KenLM language model file (.arpa or .binary).')
    ap.add_argument('--lm_alpha', type=float, default=0.5, help='Weight (alpha) for the language model score.')
    ap.add_argument('--lm_beta', type=float, default=0.2, help='Weight (beta) for the word count / length penalty.')

    args = ap.parse_args()
    device = torch.device(gpu_check())
    print(f'[INFO] Using device: {device}')

    # --- Tải Model OCR ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    print('[INFO] Checkpoint loaded successfully.')

    char_list = ckpt['charset']
    char_to_idx = {c: i for i, c in enumerate(char_list)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    SOS_IDX, EOS_IDX, PAD_IDX = char_to_idx['[SOS]'], char_to_idx['[EOS]'], char_to_idx['[PAD]']
    num_classes = len(char_list)
    
    model = VisionTransformerOCR(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    print('[INFO] Model OCR recreated and weights loaded.')

    # --- Tải Language Model ---
    print(f"[INFO] Đang tải Language Model từ: {args.lm_path}...")
    try:
        lm_model = kenlm.Model(args.lm_path)
        print("[INFO] Language Model loaded successfully.")
    except Exception as e:
        print(f"[LỖI] Không thể tải KenLM model: {e}")
        return

    img_h, pad_w = ckpt['height'], ckpt['pad_w']

    # --- Xử lý ảnh ---
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    original_size = img.shape[::-1]
    
    img_resized = resize_keep_height(img, img_h)
    original_width_after_resize = img_resized.shape[1]

    h, w = img_resized.shape
    padded_img = np.pad(img_resized, ((0,0),(0, pad_w - w)), mode='median') if w < pad_w else cv2.resize(img_resized, (pad_w, img_h))

    final_img = (gaussian_blur_and_adapt_thresh(padded_img).astype(np.float32) / 255.0)[None, ...]
    img_tensor = torch.from_numpy(final_img).unsqueeze(0).to(device) # Batch size = 1
    
    print(f'[INFO] Image preprocessed. Original size: {original_size}, Padded to: ({img_h}, {pad_w})')

    cnn_output_width = pad_w // 8
    src_key_padding_mask = torch.ones(1, cnn_output_width, dtype=torch.bool, device=device)
    valid_len = original_width_after_resize // 8
    if valid_len > 0 and valid_len <= cnn_output_width:
        src_key_padding_mask[0, :valid_len] = False
    else:
        src_key_padding_mask[0, :] = False

    # --- Chạy Beam Search MỚI ---
    best_sequence = beam_search_decode_with_lm(
        model, img_tensor, src_key_padding_mask,
        SOS_IDX, EOS_IDX, PAD_IDX,
        150, args.beam_width,
        idx_to_char,
        lm_model,
        args.lm_alpha,
        args.lm_beta
    )
    
    pred_indices = best_sequence[1:] # Bỏ [SOS]
    try:
        eos_pos = pred_indices.index(EOS_IDX)
        pred_indices = pred_indices[:eos_pos]
    except ValueError:
        pass
    
    final_indices = [idx for idx in pred_indices if idx != PAD_IDX]
    predicted_text = "".join(idx_to_char.get(idx, "") for idx in final_indices)

    print("\n" + "="*30)
    print(f"PREDICTION RESULT (Beam: {args.beam_width}, LM Alpha: {args.lm_alpha}, LM Beta: {args.lm_beta})")
    print("="*30)
    print(f"Predicted Text: {predicted_text}")
    print("="*30)

if __name__ == '__main__':
    main()