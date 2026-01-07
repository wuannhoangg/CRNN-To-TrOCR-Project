# utils/dataset.py
"""
Các lớp Dataset và hàm collate cho hai pipeline OCR:
- CRNN (CTC): trả về ảnh tensor + nhãn dạng indices (targets) + text gốc.
- Transformer (Seq2Seq): trả về ảnh tensor + text gốc + width thực tế để tạo mask.

Ghi chú thiết kế:
- Dataset CRNN ở đây sử dụng padding trắng (255) và binarization trước khi chuẩn hoá.
- Dataset Transformer hỗ trợ augmentation (PIL) khi training và tối ưu Dynamic Batch Width
  để giảm chi phí tính toán trên vùng padding.
"""

import os
import random
import math
from typing import List, Dict
from functools import partial

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

from .preprocessing import (
    resize_keep_height,
    gaussian_blur_and_adapt_thresh,
    apply_augs_pil,
    encode_to_labels,
)


class CTCDataset(Dataset):
    """Dataset cho mô hình CRNN huấn luyện theo CTCLoss.

    Args:
        paths (List[str]): Danh sách đường dẫn ảnh.
        labels_map (Dict[str, str]): Ánh xạ filename -> ground truth.
        char_to_idx (Dict[str, int]): Bảng mã ký tự -> index.
        height (int): Chiều cao chuẩn hoá sau resize.
        pad_w (int): Chiều rộng cố định/padding sau resize.
    """

    def __init__(
        self,
        paths: List[str],
        labels_map: Dict[str, str],
        char_to_idx: Dict[str, int],
        height: int,
        pad_w: int,
    ):
        self.paths = paths
        self.labels_map = labels_map
        self.char_to_idx = char_to_idx
        self.height = height
        self.pad_w = pad_w
        self.image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif'}

        # Chỉ giữ các mẫu có label hợp lệ và đúng định dạng ảnh.
        self.filtered = []
        for p in self.paths:
            fname = os.path.basename(p)
            label = labels_map.get(fname, '')
            if label and label.strip() and os.path.splitext(fname)[1].lower() in self.image_exts:
                self.filtered.append(p)

    def __len__(self):
        return len(self.filtered)

    def __getitem__(self, idx):
        """Trả về 1 sample: (img_tensor, target_tensor, raw_text)."""
        path = self.filtered[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Không raise để tránh dừng training; thay vào đó trả fallback sample.
            print(f"[WARN] Không thể đọc ảnh: {path}. Bỏ qua.")
            return self.fallback_item()

        img = resize_keep_height(img, self.height)
        h2, w2 = img.shape

        # Chuẩn hoá width: pad trắng hoặc resize cưỡng ép về pad_w.
        if w2 < self.pad_w:
            img = np.pad(
                img,
                ((0, 0), (0, self.pad_w - w2)),
                mode='constant',
                constant_values=255,
            )
        else:
            img = cv2.resize(img, (self.pad_w, self.height))

        # Binarization + normalization về [0, 1], thêm chiều channel.
        img = gaussian_blur_and_adapt_thresh(img)
        img = (img.astype(np.float32) / 255.0)[None, ...]  # (1, H, W)

        fname = os.path.basename(path)
        text = self.labels_map.get(fname, '')
        target = encode_to_labels(text, self.char_to_idx)

        # CTCLoss yêu cầu target_lengths > 0, do đó đảm bảo target có ít nhất 1 phần tử.
        if len(target) == 0:
            target = [0]

        return torch.from_numpy(img), torch.tensor(target, dtype=torch.long), text

    def fallback_item(self):
        """Tạo sample tối thiểu trong trường hợp đọc ảnh lỗi."""
        img = torch.zeros((1, self.height, self.pad_w), dtype=torch.float32)
        target = torch.tensor([0], dtype=torch.long)
        text = ""
        return img, target, text


def ctc_collate(batch):
    """Collate function cho CTCDataset.

    Quy ước output:
    - imgs: (B, 1, H, W)
    - targets: tensor 1D là concat của toàn bộ label indices trong batch
    - label_lengths: (B,) độ dài nhãn của từng sample
    - texts: list ground truth string
    """
    batch = [item for item in batch if item[0].numel() > 0]
    if not batch:
        return torch.Tensor(), torch.Tensor(), torch.Tensor(), []

    imgs, labels, texts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    # `targets` là concat toàn bộ label indices (CTCLoss yêu cầu format này).
    if sum(label_lengths).item() > 0:
        targets = torch.cat(labels)
    else:
        targets = torch.zeros((0,), dtype=torch.long)

    return imgs, targets, label_lengths, texts


class TransformerDataset(Dataset):
    """Dataset cho mô hình Transformer OCR (Seq2Seq).

    Args:
        paths (List[str]): Danh sách đường dẫn ảnh.
        labels_map (Dict[str, str]): Ánh xạ filename -> ground truth.
        height (int): Chiều cao sau resize.
        pad_w (int): Chiều rộng pad/resize.
        is_training (bool): Bật augmentation khi training.
    """

    def __init__(
        self,
        paths: List[str],
        labels_map: Dict[str, str],
        height: int,
        pad_w: int,
        is_training: bool = False,
    ):
        self.paths = paths
        self.labels_map = labels_map
        self.height = height
        self.pad_w = pad_w
        self.is_training = is_training
        self.filtered = [p for p in paths if os.path.basename(p) in labels_map]

    def __len__(self):
        return len(self.filtered)

    def __getitem__(self, idx):
        """Trả về 1 sample: (img_tensor, raw_text, width_after_resize)."""
        path = self.filtered[idx]
        try:
            img_bytes = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError("cv2.imdecode returned None")
        except Exception as e:
            print(f"[WARN] Không thể đọc ảnh {path}: {e}. Bỏ qua.")
            return torch.Tensor(), "", 0

        img_resized = resize_keep_height(img, self.height)

        # Augmentation (PIL) chỉ áp dụng trong chế độ training.
        if self.is_training and random.random() < 0.8:
            try:
                img_pil = Image.fromarray(img_resized)
                img_pil = apply_augs_pil(img_pil)
                img_resized = np.array(img_pil)
            except Exception as e:
                print(f"[WARN] Augmentation thất bại cho {path}: {e}")

        # Cắt khoảng trắng thừa bên phải nhằm giảm vùng padding không cần thiết.
        if img_resized.shape[1] > 100:
            col_sum = np.sum(img_resized < 200, axis=0)
            non_empty_cols = np.where(col_sum > 2)[0]
            if len(non_empty_cols) > 0:
                max_col = non_empty_cols[-1] + 10
                max_col = min(max_col, img_resized.shape[1])
                img_resized = img_resized[:, :max_col]

        original_width_after_resize = img_resized.shape[1]
        h, w = img_resized.shape

        # Chuẩn hoá width: pad trắng hoặc resize cưỡng ép về pad_w.
        if w < self.pad_w:
            padded_img = np.pad(
                img_resized,
                ((0, 0), (0, self.pad_w - w)),
                mode='constant',
                constant_values=255,
            )
        elif w > self.pad_w:
            padded_img = cv2.resize(img_resized, (self.pad_w, self.height))
        else:
            padded_img = img_resized

        # Binarization + normalization về [0, 1], thêm chiều channel.
        final_img = gaussian_blur_and_adapt_thresh(padded_img)
        final_img = (final_img.astype(np.float32) / 255.0)[None, ...]

        text = self.labels_map.get(os.path.basename(path), '')
        return torch.from_numpy(final_img), text, original_width_after_resize


def transformer_collate(batch, char_to_idx, sos_idx, eos_idx, pad_idx):
    """Collate function cho TransformerDataset.

    Các tối ưu chính:
    - Loại bỏ sample lỗi.
    - Dynamic Batch Width: cắt batch theo width lớn nhất thực tế (làm tròn bội 8).
    - Tạo src_key_padding_mask dựa trên width hợp lệ của từng ảnh.
    - Pad target sequence theo max_len trong batch và thêm [SOS]/[EOS].

    Returns:
        imgs (torch.Tensor): (B, 1, H, W')
        src_key_padding_mask (torch.BoolTensor): (B, S)
        tgt_padded (torch.LongTensor): (B, L)
        texts (Tuple[str]): ground truth strings
    """
    batch = [item for item in batch if item[0].numel() > 0]
    if not batch:
        return None, None, None, None

    imgs, texts, original_widths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    # Dynamic width: giới hạn tính toán theo max width thực tế trong batch.
    max_w_in_batch = max(original_widths)
    max_w_in_batch = math.ceil(max_w_in_batch / 8) * 8
    imgs = imgs[:, :, :, :max_w_in_batch]

    if len(imgs.shape) != 4:
        return None, None, None, None

    # Mask theo chiều width sau backbone (downsample x8).
    cnn_output_width = imgs.shape[3] // 8
    src_key_padding_mask = torch.ones(len(texts), cnn_output_width, dtype=torch.bool)
    for i, w in enumerate(original_widths):
        valid_len = w // 8
        if valid_len > 0 and valid_len <= cnn_output_width:
            src_key_padding_mask[i, :valid_len] = False

    # Target padding: [SOS] + tokens + [EOS], pad tới max_len trong batch.
    max_len = max(len(s) for s in texts) + 2 if texts else 2
    tgt_padded = torch.full((len(texts), max_len), pad_idx, dtype=torch.long)

    for i, txt in enumerate(texts):
        encoded = [sos_idx] + [char_to_idx.get(c, pad_idx) for c in txt] + [eos_idx]
        encoded_len = min(len(encoded), max_len)
        tgt_padded[i, :encoded_len] = torch.tensor(encoded[:encoded_len], dtype=torch.long)

    return imgs, src_key_padding_mask, tgt_padded, texts
