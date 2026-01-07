# src/transformer/eval_lm.py
# -*- coding: utf-8 -*-

"""
Đánh giá mô hình Transformer OCR với Beam Search kết hợp Language Model (KenLM).

Luồng xử lý:
1) Nạp dữ liệu test và tiền xử lý ảnh (resize theo height, trim khoảng trắng, padding, binarization).
2) Nạp mô hình OCR (VisionTransformerOCR) từ checkpoint và nạp KenLM.
3) Chạy inference theo batch (encode ảnh -> beam search decode có LM).
4) Tính CER/WER/SER và (tuỳ chọn) lưu kết quả chi tiết ra JSON.

Ghi chú:
- Script này giả định checkpoint chứa `charset`, `model_state`, `height` và (tuỳ chọn) `pad_w`.
- Việc decode phụ thuộc vào hàm `beam_search_decode_with_lm` trong utils.decoding.
"""

import os
import argparse
import json
import math
import pathlib
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import kenlm
except ImportError:
    print("Lỗi: Gói 'kenlm' chưa được cài đặt. Vui lòng cài đặt để sử dụng tính năng này.")
    exit()

from ..utils.preprocessing import (
    gpu_check,
    read_labels,
    resize_keep_height,
    gaussian_blur_and_adapt_thresh,
)
from ..utils.decoding import (
    beam_search_decode_with_lm,
)

from .model import VisionTransformerOCR


class EvalOCRDataset(Dataset):
    """Dataset phục vụ đánh giá mô hình OCR.

    Dataset này:
    - Đọc ảnh từ đường dẫn (hỗ trợ path có ký tự Unicode bằng np.fromfile + cv2.imdecode).
    - Resize giữ tỷ lệ theo `height`.
    - Trim khoảng trắng thừa bên phải để giảm vùng padding không cần thiết.
    - Padding/resize theo `pad_w`.
    - Binarization + normalization để tạo tensor input.

    Attributes:
        paths (List[str]): Danh sách đường dẫn ảnh.
        labels_map (Dict[str, str]): Ánh xạ filename -> ground truth.
        height (int): Chiều cao sau resize.
        pad_w (int): Chiều rộng pad/resize thống nhất.
    """

    def __init__(self, paths: List[str], labels_map: Dict[str, str], height: int, pad_w: int):
        self.paths = paths
        self.labels_map = labels_map
        self.height = height
        self.pad_w = pad_w
        self.image_exts = {'.png', '.jpg', '.jpeg'}

        # Chỉ giữ các ảnh có entry trong labels_map (label có thể rỗng).
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

        # Đọc ảnh grayscale; dùng np.fromfile để tránh lỗi với đường dẫn Unicode.
        try:
            img_bytes = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError("cv2.imdecode returned None")
        except Exception:
            # Trả dummy sample để tránh làm crash luồng đánh giá.
            return None, "", 0, ""

        # Resize ảnh theo chiều cao cố định, giữ nguyên tỷ lệ.
        img_resized = resize_keep_height(img, self.height)

        # Trim khoảng trắng thừa bên phải bằng cách tìm cột cuối có thông tin.
        if img_resized.shape[1] > 100:
            col_sum = np.sum(img_resized < 200, axis=0)
            non_empty_cols = np.where(col_sum > 2)[0]
            if len(non_empty_cols) > 0:
                max_col = non_empty_cols[-1] + 10
                max_col = min(max_col, img_resized.shape[1])
                img_resized = img_resized[:, :max_col]

        original_width_after_resize = img_resized.shape[1]
        h, w = img_resized.shape

        # Chuẩn hoá width: pad trắng bên phải hoặc resize cưỡng ép khi quá dài.
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

        # Trả về: ảnh tensor, ground truth, width thực tế sau resize, filename.
        return (
            torch.from_numpy(final_img),
            text,
            original_width_after_resize,
            os.path.basename(path),
        )


def eval_collate(batch):
    """Collate function cho DataLoader đánh giá.

    Mục tiêu:
    - Loại bỏ sample lỗi (None).
    - Áp dụng tối ưu "Dynamic Batch Width": cắt batch theo width lớn nhất thật sự
      trong batch để giảm tính toán trên vùng padding.
    - Tạo `src_key_padding_mask` cho Transformer encoder/decoder.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[str], Tuple[str]] hoặc (None, None, None, None)
    """
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None, None, None

    imgs, texts, original_widths, fnames = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    # Dynamic width: cắt theo max width thực tế (làm tròn bội 8 để khớp downsample CNN).
    max_w_in_batch = max(original_widths)
    max_w_in_batch = math.ceil(max_w_in_batch / 8) * 8
    imgs = imgs[:, :, :, :max_w_in_batch]

    # Mask theo chiều width sau backbone (downsample x8).
    cnn_output_width = imgs.shape[3] // 8
    src_key_padding_mask = torch.ones(len(texts), cnn_output_width, dtype=torch.bool)

    for i, w in enumerate(original_widths):
        valid_len = w // 8
        if valid_len > 0 and valid_len <= cnn_output_width:
            src_key_padding_mask[i, :valid_len] = False

    return imgs, src_key_padding_mask, texts, fnames


def cer_wer_ser(pred_texts: List[str], gt_texts: List[str]) -> Tuple[float, float, float]:
    """Tính CER, WER, SER theo trung bình trên tập mẫu hợp lệ (gt không rỗng).

    Args:
        pred_texts (List[str]): Chuỗi dự đoán.
        gt_texts (List[str]): Chuỗi ground truth.

    Returns:
        Tuple[float, float, float]: (CER, WER, SER).
    """

    def lev(a, b):
        """Khoảng cách Levenshtein cho chuỗi/list."""
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

    # Nếu không có mẫu hợp lệ, trả về 1.0 (lỗi tối đa) để tránh chia 0.
    return (
        float(np.mean(cer if cer else [1.0])),
        float(np.mean(wer if wer else [1.0])),
        float(np.mean(ser if ser else [1.0])),
    )


def main():
    """Điểm vào chính: nạp model + LM, chạy evaluation loop và báo cáo kết quả."""
    ap = argparse.ArgumentParser(description="Evaluate OCR Model using Beam Search + KenLM.")

    ap.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn checkpoint model (.pt).')
    ap.add_argument('--test_images_dir', type=str, required=True, help='Thư mục chứa ảnh test.')
    ap.add_argument('--test_labels_json', type=str, required=True, help='JSON nhãn ground truth.')
    ap.add_argument('--batch_size', type=int, default=16, help='Kích thước batch.')
    ap.add_argument('--beam_width', type=int, default=10, help='Độ rộng beam search.')
    ap.add_argument('--output_file', type=str, default=None, help='(Tuỳ chọn) JSON lưu kết quả chi tiết.')

    ap.add_argument('--lm_path', type=str, required=True, help='Đường dẫn KenLM (.binary hoặc .arpa).')
    ap.add_argument('--lm_alpha', type=float, default=0.5, help='Trọng số LM score.')
    ap.add_argument('--lm_beta', type=float, default=0.2, help='Length penalty.')

    args = ap.parse_args()

    # Chọn device (GPU/CPU) theo cấu hình runtime.
    device = torch.device(gpu_check())
    print(f'[INFO] Using device: {device}')

    # Nạp checkpoint và khôi phục charset/token indices.
    print(f'[INFO] Loading Checkpoint "{args.checkpoint}"...')
    ckpt = torch.load(args.checkpoint, map_location=device)

    char_list = ckpt['charset']
    char_to_idx = {c: i for i, c in enumerate(char_list)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    SOS_IDX = char_to_idx['[SOS]']
    EOS_IDX = char_to_idx['[EOS]']
    PAD_IDX = char_to_idx['[PAD]']
    num_classes = len(char_list)

    # Khởi tạo kiến trúc model và nạp trọng số.
    model = VisionTransformerOCR(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    print('[INFO] OCR Model loaded successfully.')

    img_h = ckpt['height']
    pad_w = ckpt.get('pad_w', 3500)

    # Nạp KenLM.
    print(f"[INFO] Loading Language Model from: {args.lm_path}...")
    try:
        lm_model = kenlm.Model(args.lm_path)
        print("[INFO] Language Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load KenLM: {e}")
        return

    # Đọc nhãn test với normalize=True để thống nhất chuẩn so sánh.
    labels_map = read_labels(args.test_labels_json, normalize=True)

    # Quét toàn bộ ảnh test (đệ quy).
    image_paths = [
        str(p) for p in pathlib.Path(args.test_images_dir).glob('**/*')
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    ]

    test_ds = EvalOCRDataset(image_paths, labels_map, img_h, pad_w)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=eval_collate,
    )

    print(f"\n[INFO] Starting evaluation on {len(test_ds)} test samples...")
    pred_texts, gt_texts, filenames = [], [], []

    # Evaluation loop: encode -> beam search decode có LM.
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            if batch_data[0] is None:
                continue

            imgs, src_key_padding_mask, gts, fns = batch_data
            imgs = imgs.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)

            memory = model.encode(imgs, src_key_padding_mask)

            hyps = beam_search_decode_with_lm(
                model,
                memory,
                src_key_padding_mask,
                SOS_IDX,
                EOS_IDX,
                PAD_IDX,
                150,
                args.beam_width,
                idx_to_char,
                lm_model,
                args.lm_alpha,
                args.lm_beta,
            )

            for seq in hyps:
                pred_indices = seq[1:]
                try:
                    eos_pos = pred_indices.index(EOS_IDX)
                    pred_indices = pred_indices[:eos_pos]
                except ValueError:
                    pass

                final_indices = [idx for idx in pred_indices if idx != PAD_IDX]
                decoded_text = "".join(idx_to_char.get(idx, "") for idx in final_indices)
                pred_texts.append(decoded_text)

            gt_texts.extend(gts)
            filenames.extend(fns)

    final_cer, final_wer, final_ser = cer_wer_ser(pred_texts, gt_texts)

    print("\n" + "=" * 30)
    print("FINAL EVALUATION RESULTS (with LM)")
    print("=" * 30)
    print(f"Character Error Rate (CER): {final_cer:.4f}")
    print(f"Word Error Rate (WER)   : {final_wer:.4f}")
    print(f"Sentence Error Rate (SER) : {final_ser:.4f}")
    print("=" * 30)
    print(f"(LM_ALPHA={args.lm_alpha}, LM_BETA={args.lm_beta}, BEAM_WIDTH={args.beam_width})")

    # Lưu kết quả chi tiết nếu được yêu cầu.
    if args.output_file:
        results = []
        for i in range(len(filenames)):
            results.append({
                "filename": filenames[i],
                "ground_truth": gt_texts[i],
                "prediction": pred_texts[i],
            })

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Detailed prediction results saved to: {args.output_file}")


if __name__ == '__main__':
    main()
