# utils/dataset.py

import os
import random
from typing import List, Dict
from functools import partial

import numpy as np
import cv2
import torch # <-- ĐÃ THÊM IMPORT
from torch.utils.data import Dataset
from PIL import Image

# Import các hàm utils chúng ta vừa tạo
from .preprocessing import (
    resize_keep_height,
    gaussian_blur_and_adapt_thresh,
    apply_augs_pil,
    encode_to_labels
)

class CTCDataset(Dataset):
    """
    Dataset cho model CRNN (CTC).
    """
    def __init__(self, paths: List[str], labels_map: Dict[str, str],
                char_to_idx: Dict[str,int], height: int, pad_w: int):
        self.paths = paths
        self.labels_map = labels_map
        self.char_to_idx = char_to_idx
        self.height = height
        self.pad_w = pad_w
        self.image_exts = {'.png','.jpg','.jpeg'}

        self.filtered = []
        for p in self.paths:
            fname = os.path.basename(p)
            label = labels_map.get(fname, '')
            if label and label.strip() and os.path.splitext(fname)[1].lower() in self.image_exts:
                self.filtered.append(p)

    def __len__(self):
        return len(self.filtered)

    def __getitem__(self, idx):
        path = self.filtered[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Không thể đọc ảnh: {path}. Bỏ qua.")
            return self.fallback_item()

        img = resize_keep_height(img, self.height)
        h2, w2 = img.shape

        if w2 < self.pad_w:
            img = np.pad(img, ((0,0),(0, self.pad_w - w2)), mode='median')
        else:
            img = cv2.resize(img, (self.pad_w, self.height))

        img = gaussian_blur_and_adapt_thresh(img)
        img = (img.astype(np.float32) / 255.0)[None, ...]  # (1,H,W)

        fname = os.path.basename(path)
        text = self.labels_map.get(fname, '')
        target = encode_to_labels(text, self.char_to_idx)
        if len(target) == 0:
            target = [0] # Phải có ít nhất 1 ký tự (có thể là ký tự đầu tiên)

        return torch.from_numpy(img), torch.tensor(target, dtype=torch.long), text

    def fallback_item(self):
        """Tr_item rỗng nếu ảnh lỗi"""
        img = torch.zeros((1, self.height, self.pad_w), dtype=torch.float32)
        target = torch.tensor([0], dtype=torch.long)
        text = ""
        return img, target, text

def ctc_collate(batch):
    batch = [item for item in batch if item[0].numel() > 0]
    if not batch:
        return torch.Tensor(), torch.Tensor(), torch.Tensor(), []
        
    imgs, labels, texts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    
    if sum(label_lengths).item() > 0:
        targets = torch.cat(labels)
    else:
        targets = torch.zeros((0,), dtype=torch.long)
        
    return imgs, targets, label_lengths, texts


class TransformerDataset(Dataset):
    """
    Dataset cho model Transformer (Seq2Seq).
    """
    def __init__(self, paths: List[str], labels_map: Dict[str, str], height: int, pad_w: int, is_training: bool = False):
        self.paths = paths
        self.labels_map = labels_map
        self.height = height
        self.pad_w = pad_w
        self.is_training = is_training
        self.filtered = [p for p in paths if os.path.basename(p) in labels_map]

    def __len__(self):
        return len(self.filtered)

    def __getitem__(self, idx):
        path = self.filtered[idx]
        try:
            img_bytes = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None: raise IOError("cv2.imdecode returned None")
        except Exception as e:
            print(f"[WARN] Không thể đọc ảnh {path}: {e}. Bỏ qua.")
            return torch.Tensor(), "", 0
        
        img_resized = resize_keep_height(img, self.height)

        if self.is_training and random.random() < 0.8:
            try:
                img_pil = Image.fromarray(img_resized)
                img_pil = apply_augs_pil(img_pil)
                img_resized = np.array(img_pil)
            except Exception as e:
                print(f"[WARN] Augmentation thất bại cho {path}: {e}")

        original_width_after_resize = img_resized.shape[1]
        h, w = img_resized.shape

        if w < self.pad_w:
            padded_img = np.pad(img_resized, ((0,0),(0, self.pad_w - w)), mode='median')
        elif w > self.pad_w:
            padded_img = cv2.resize(img_resized, (self.pad_w, self.height))
        else:
            padded_img = img_resized

        final_img = gaussian_blur_and_adapt_thresh(padded_img)
        final_img = (final_img.astype(np.float32) / 255.0)[None, ...]
        
        text = self.labels_map.get(os.path.basename(path), '')
        return torch.from_numpy(final_img), text, original_width_after_resize

def transformer_collate(batch, char_to_idx, sos_idx, eos_idx, pad_idx):
    batch = [item for item in batch if item[0].numel() > 0]
    if not batch:
        return None, None, None, None

    imgs, texts, original_widths = zip(*batch); imgs = torch.stack(imgs, dim=0)
    
    if len(imgs.shape) != 4:
        return None, None, None, None

    cnn_output_width = imgs.shape[3] // 8
    src_key_padding_mask = torch.ones(len(texts), cnn_output_width, dtype=torch.bool)
    for i, w in enumerate(original_widths):
        valid_len = w // 8
        if valid_len > 0 and valid_len <= cnn_output_width:
            src_key_padding_mask[i, :valid_len] = False
    
    max_len = max(len(s) for s in texts) + 2 if texts else 2
    
    tgt_padded = torch.full((len(texts), max_len), pad_idx, dtype=torch.long)
    for i, txt in enumerate(texts):
        encoded = [sos_idx] + [char_to_idx.get(c, pad_idx) for c in txt] + [eos_idx]
        encoded_len = min(len(encoded), max_len)
        tgt_padded[i, :encoded_len] = torch.tensor(encoded[:encoded_len], dtype=torch.long)
        
    return imgs, src_key_padding_mask, tgt_padded, texts