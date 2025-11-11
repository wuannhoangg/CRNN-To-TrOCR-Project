# src/transformer/eval_lm.py
import os
import argparse
import json
import math
import pathlib
from typing import List, Dict, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import kenlm
except ImportError:
    print("Lỗi: Gói 'kenlm' chưa được cài đặt.")
    exit()

# --- Import từ thư mục utils ---
from ..utils.preprocessing import (
    gpu_check,
    read_labels,
    resize_keep_height,
    gaussian_blur_and_adapt_thresh
)
from ..utils.decoding import (
    get_word_from_indices,
    beam_search_decode_with_lm
)

# --- Import model từ file cùng cấp ---
from .model import PositionalEncoding, VisionTransformerOCR

# --- Dataset & Collate ---
class OCRDataset(Dataset):
    def __init__(self, paths: List[str], labels_map: Dict[str, str], height: int, pad_w: int):
        self.paths = paths
        self.labels_map = labels_map
        self.height = height
        self.pad_w = pad_w
        self.image_exts = {'.png','.jpg','.jpeg'}
        self.filtered = []
        for p in paths:
            fname = os.path.basename(p)
            label = labels_map.get(fname)
            if label is not None:
                self.filtered.append(p)

    def __len__(self):
        return len(self.filtered)

    def __getitem__(self, idx):
        path = self.filtered[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, "", 0, "" # Trả về None để collate bỏ qua
            
        img_resized = resize_keep_height(img, self.height)
        original_width_after_resize = img_resized.shape[1]
        h, w = img_resized.shape
        padded_img = np.pad(img_resized, ((0,0),(0, self.pad_w - w)), mode='median') if w < self.pad_w else cv2.resize(img_resized, (self.pad_w, self.height))
        final_img = (gaussian_blur_and_adapt_thresh(padded_img).astype(np.float32) / 255.0)[None, ...]
        text = self.labels_map.get(os.path.basename(path), '')
        return torch.from_numpy(final_img), text, original_width_after_resize, os.path.basename(path)

def transformer_collate(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None, None, None
        
    imgs, texts, original_widths, fnames = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    cnn_output_width = imgs.shape[3] // 8
    src_key_padding_mask = torch.ones(len(texts), cnn_output_width, dtype=torch.bool)
    for i, w in enumerate(original_widths):
        valid_len = w // 8
        if valid_len > 0 and valid_len <= cnn_output_width:
            src_key_padding_mask[i, :valid_len] = False
    return imgs, src_key_padding_mask, texts, fnames

# --- Metrics ---
def cer_wer_ser(pred_texts: List[str], gt_texts: List[str]):
    def lev(a, b):
        n, m = len(a), len(b)
        dp = list(range(m+1))
        for i in range(1, n+1):
            prev = dp[0]; dp[0] = i
            for j in range(1, m+1):
                tmp = dp[j]
                dp[j] = min(dp[j]+1, dp[j-1]+1, prev + (0 if a[i-1]==b[j-1] else 1))
                prev = tmp
        return dp[m]
    cer, wer, ser = [], [], []
    for pd, gt in zip(pred_texts, gt_texts):
        if not gt: continue
        cer.append(lev(list(pd.lower()), list(gt.lower())) / len(gt))
        pw, gw = pd.lower().split(), gt.lower().split()
        if len(gw) > 0: wer.append(lev(pw, gw) / len(gw))
        ser.append(0.0 if pd == gt else 1.0)
    return float(np.mean(cer if cer else [1.0])), float(np.mean(wer if wer else [1.0])), float(np.mean(ser if ser else [1.0]))

# --------------------- Main Evaluation Logic ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pt file).')
    ap.add_argument('--test_images_dir', type=str, required=True, help='Path to the directory containing test images.')
    ap.add_argument('--test_labels_json', type=str, required=True, help='Path to the JSON file containing test labels.')
    ap.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
    ap.add_argument('--beam_width', type=int, default=30, help='Beam width for decoding.')
    ap.add_argument('--output_file', type=str, default=None, help='(Optional) Path to save detailed prediction results as a JSON file.')
    
    # --- Tham số cho Language Model ---
    ap.add_argument('--lm_path', type=str, required=True, help='Path to the KenLM language model file (.arpa or .binary).')
    ap.add_argument('--lm_alpha', type=float, default=0.5, help='Weight (alpha) for the language model score.')
    ap.add_argument('--lm_beta', type=float, default=0.2, help='Weight (beta) for the word count / length penalty.')

    args = ap.parse_args()
    device = torch.device(gpu_check())
    print(f'[INFO] Using device: {device}')

    # --- TẢI MODEL OCR (Transformer) ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f'[INFO] Checkpoint "{args.checkpoint}" loaded successfully.')

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

    img_h, pad_w = ckpt['height'], ckpt['pad_w']

    # --- TẢI MODEL NGÔN NGỮ (KenLM) ---
    print(f"[INFO] Đang tải Language Model từ: {args.lm_path}...")
    try:
        lm_model = kenlm.Model(args.lm_path)
        print("[INFO] Language Model loaded successfully.")
    except Exception as e:
        print(f"[LỖI] Không thể tải KenLM model: {e}")
        return

    # --- Tải Dataset ---
    labels_map = read_labels(args.test_labels_json, normalize=True) # Normalize text
    image_paths = [str(p) for p in pathlib.Path(args.test_images_dir).glob('**/*') if p.suffix.lower() in {'.png','.jpg','.jpeg'}]
    
    test_ds = OCRDataset(image_paths, labels_map, img_h, pad_w)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=transformer_collate)

    print(f"\n[INFO] Starting evaluation on {len(test_ds)} test samples...")
    pred_texts, gt_texts, filenames = [], [], []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            if batch_data[0] is None: continue
            imgs, src_key_padding_mask, gts, fns = batch_data
                
            imgs, src_key_padding_mask = imgs.to(device), src_key_padding_mask.to(device)
            memory = model.encode(imgs, src_key_padding_mask)
            
            hyps = beam_search_decode_with_lm(
                model, memory, src_key_padding_mask,
                SOS_IDX, EOS_IDX, PAD_IDX,
                150, args.beam_width,
                idx_to_char,
                lm_model,
                args.lm_alpha,
                args.lm_beta
            )
            
            for seq in hyps:
                pred_indices = seq[1:] # Bỏ [SOS]
                try:
                    eos_pos = pred_indices.index(EOS_IDX)
                    pred_indices = pred_indices[:eos_pos]
                except ValueError: pass
                
                final_indices = [idx for idx in pred_indices if idx != PAD_IDX]
                pred_texts.append("".join(idx_to_char.get(idx, "") for idx in final_indices))
                
            gt_texts.extend(gts)
            filenames.extend(fns)

    final_cer, final_wer, final_ser = cer_wer_ser(pred_texts, gt_texts)
    
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS (with LM)")
    print("="*30)
    print(f"Character Error Rate (CER): {final_cer:.4f}")
    print(f"Word Error Rate (WER)   : {final_wer:.4f}")
    print(f"Sentence Error Rate (SER) : {final_ser:.4f}")
    print("="*30)
    print(f"(LM_ALPHA={args.lm_alpha}, LM_BETA={args.lm_beta}, BEAM_WIDTH={args.beam_width})")

    if args.output_file:
        results = []
        for i in range(len(filenames)):
            results.append({
                "filename": filenames[i],
                "ground_truth": gt_texts[i],
                "prediction": pred_texts[i]
            })
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Detailed prediction results saved to: {args.output_file}")

if __name__ == '__main__':
    main()