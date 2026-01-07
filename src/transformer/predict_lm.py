# src/transformer/predict_lm.py
# -*- coding: utf-8 -*-
"""
Suy luận OCR theo pipeline hợp nhất (phát hiện dòng + nhận dạng) với Transformer OCR
kết hợp Language Model (KenLM).

Luồng xử lý chính:
1) Nạp checkpoint Transformer OCR và khôi phục charset/[SOS]/[EOS]/[PAD].
2) Nạp KenLM để chấm điểm chuỗi trong quá trình beam search.
3) Đọc ảnh đầu vào:
   - Nếu ảnh đã được cắt sẵn dạng 1 dòng (pre-cropped) -> bỏ qua detection.
   - Nếu là ảnh trang đầy đủ -> chạy DBNet (OpenCV DNN) để phát hiện các dòng text.
4) Cắt từng vùng text, tiền xử lý (resize/pad/binarize), tạo mask theo chiều rộng hợp lệ.
5) Decode bằng beam search có LM, ghép kết quả thành nhiều dòng và in/ghi file.

Ghi chú:
- Script này có phụ thuộc `shapely` để lọc box chồng lấn theo diện tích giao.
- DBNet sử dụng OpenCV `cv2.dnn_TextDetectionModel_DB` với file ONNX.
"""

import os
import argparse
import json
import math
import time
import traceback
from typing import List, Dict, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Polygon

try:
    import kenlm
except ImportError:
    print("[WARN] Chưa cài đặt 'kenlm'. Chức năng LM sẽ không hoạt động.")
    exit()

from ..utils.preprocessing import (
    gpu_check,
    resize_keep_height,
    gaussian_blur_and_adapt_thresh,
)
from ..utils.decoding import (
    beam_search_decode_with_lm,
)
from .model import VisionTransformerOCR


def is_pre_cropped(image: np.ndarray, pixel_threshold=1.93, aspect_ratio_threshold=5.0) -> bool:
    """Heuristic kiểm tra ảnh có phải đã cắt sẵn dạng 1 dòng text hay không.

    Tiêu chí:
    - Tỷ lệ W/H đủ lớn (dạng "dài ngang").
    - Tỷ lệ pixel chữ (sau binarize) vượt ngưỡng tối thiểu.

    Args:
        image (np.ndarray): Ảnh đầu vào (BGR hoặc grayscale).
        pixel_threshold (float): Ngưỡng phần trăm pixel chữ tối thiểu.
        aspect_ratio_threshold (float): Ngưỡng tỷ lệ W/H tối thiểu.

    Returns:
        bool: True nếu ảnh có khả năng là pre-cropped.
    """
    try:
        h, w = image.shape[:2]
        aspect_ratio = w / h
        if aspect_ratio < aspect_ratio_threshold:
            return False

        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        thresh = gaussian_blur_and_adapt_thresh(gray)
        total_pixels = thresh.size
        text_pixels = np.count_nonzero(thresh)
        if total_pixels == 0:
            return False
        percentage = (text_pixels / total_pixels) * 100

        return (percentage > pixel_threshold)
    except:
        return False


def preprocess_for_dbnet(image, target_size=736, multiple_of=32):
    """Resize + pad ảnh đầu vào để phù hợp với DBNet.

    Mục tiêu:
    - Scale theo cạnh lớn nhất về target_size.
    - Pad lên bội số `multiple_of` để tương thích cấu trúc CNN.

    Args:
        image (np.ndarray): Ảnh BGR đầu vào.
        target_size (int): Kích thước mục tiêu theo cạnh lớn nhất.
        multiple_of (int): Bội số cần pad.

    Returns:
        np.ndarray: Ảnh đã resize/pad.
    """
    try:
        h_orig, w_orig = image.shape[:2]
        scale = target_size / max(h_orig, w_orig)
        if scale * w_orig < multiple_of:
            scale = multiple_of / w_orig
        if scale * h_orig < multiple_of:
            scale = multiple_of / h_orig
        img_resized = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)))
        h_new, w_new = img_resized.shape[:2]
        pad_h = (multiple_of - (h_new % multiple_of)) % multiple_of
        pad_w = (multiple_of - (w_new % multiple_of)) % multiple_of
        padded_image = np.zeros((h_new + pad_h, w_new + pad_w, 3), dtype=np.uint8)
        padded_image[0:h_new, 0:w_new, :] = img_resized
        return padded_image
    except:
        return image


def get_cropped_image(image, box):
    """Cắt ảnh theo rotated rectangle (minAreaRect) bằng perspective transform.

    Args:
        image (np.ndarray): Ảnh gốc.
        box: Rotated rect theo format ((cx, cy), (w, h), angle).

    Returns:
        np.ndarray: Ảnh crop đã được warp về hình chữ nhật thẳng.
    """
    points = cv2.boxPoints(box)
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def sort_boxes(boxes):
    """Sắp xếp box theo toạ độ y của tâm để đọc theo thứ tự từ trên xuống."""
    return sorted(boxes, key=lambda b: b[0][1])


def filter_boxes_shapely(boxes, overlap_threshold=0.4):
    """Loại bỏ các box chồng lấn mạnh dựa trên tỉ lệ overlap area.

    Chiến lược:
    - Duyệt box theo diện tích giảm dần.
    - Bỏ box nếu overlap_ratio (intersection/current_area) vượt ngưỡng.

    Args:
        boxes (List): Danh sách rotated rect theo format OpenCV.
        overlap_threshold (float): Ngưỡng overlap tối đa cho phép.

    Returns:
        List: Danh sách box đã lọc.
    """
    if len(boxes) == 0:
        return []
    sorted_indices = sorted(
        range(len(boxes)),
        key=lambda i: boxes[i][1][0] * boxes[i][1][1],
        reverse=True,
    )
    max_area = boxes[sorted_indices[0]][1][0] * boxes[sorted_indices[0]][1][1]
    keep_indices = []

    for i in sorted_indices:
        current_box = boxes[i]
        (cx, cy), (w, h), angle = current_box
        current_area = w * h

        # Bỏ các box quá nhỏ so với box lớn nhất để giảm nhiễu.
        if current_area < max_area * 0.05:
            continue

        is_bad_box = False
        for j in keep_indices:
            kept_box = boxes[j]
            try:
                p1 = Polygon(cv2.boxPoints(current_box))
                p2 = Polygon(cv2.boxPoints(kept_box))
                if not p1.is_valid or not p2.is_valid:
                    continue
                inter_area = p1.intersection(p2).area
                overlap_ratio = inter_area / current_area if current_area > 0 else 0
                if overlap_ratio > overlap_threshold:
                    is_bad_box = True
                    break
            except:
                pass

        if not is_bad_box:
            keep_indices.append(i)

    return [boxes[i] for i in keep_indices]


def main():
    """Điểm vào chính: load model, chạy detection (nếu cần), nhận dạng và xuất kết quả."""
    parser = argparse.ArgumentParser(description="Unified OCR Prediction (No T5 Correction)")

    parser.add_argument('--image', required=True, help='Path to input image.')
    parser.add_argument('--output_file', default=None, help='Path to save text output.')

    parser.add_argument('--ocr_checkpoint', required=True, help='Path to Transformer OCR checkpoint (.pt).')
    parser.add_argument('--lm_model', required=True, help='Path to KenLM binary (.binary).')
    parser.add_argument('--det_model', default='models/DB_TD500_resnet50.onnx', help='Path to DBNet ONNX model.')

    parser.add_argument('--beam_width', type=int, default=10, help='Beam width.')
    parser.add_argument('--lm_alpha', type=float, default=0.5, help='LM Alpha.')
    parser.add_argument('--lm_beta', type=float, default=0.2, help='LM Beta (length penalty).')

    args = parser.parse_args()
    device = torch.device(gpu_check())
    print(f'[INFO] Device: {device}')

    # Nạp OCR checkpoint và khôi phục vocabulary/tokens.
    print(f"[INFO] Loading OCR Model from {args.ocr_checkpoint}...")
    ckpt = torch.load(args.ocr_checkpoint, map_location=device)
    char_list = ckpt['charset']
    char_to_idx = {c: i for i, c in enumerate(char_list)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    SOS_IDX, EOS_IDX, PAD_IDX = char_to_idx['[SOS]'], char_to_idx['[EOS]'], char_to_idx['[PAD]']

    ocr_model = VisionTransformerOCR(num_classes=len(char_list))
    ocr_model.load_state_dict(ckpt['model_state'])
    ocr_model.to(device)
    ocr_model.eval()

    IMG_H = ckpt['height']
    PAD_W = ckpt['pad_w']

    # Nạp KenLM để dùng trong beam search.
    print(f"[INFO] Loading KenLM from {args.lm_model}...")
    try:
        lm_model = kenlm.Model(args.lm_model)
    except Exception as e:
        print(f"[ERROR] Failed to load KenLM: {e}")
        return

    # Đọc ảnh đầu vào.
    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] Image not found: {args.image}")
        return

    original_image = image.copy()
    crops_to_process = []

    # Chọn chế độ xử lý: line-level (pre-cropped) hoặc page-level (cần detection).
    if is_pre_cropped(original_image):
        print("[INFO] Mode: Single Line (Pre-cropped). Skipping Detection.")
        crops_to_process.append(original_image)
    else:
        print("[INFO] Mode: Full Page. Running DBNet Detection...")
        if not os.path.exists(args.det_model):
            print(f"[WARN] DBNet model not found at {args.det_model}. Fallback to full image processing.")
            crops_to_process.append(original_image)
        else:
            try:
                text_detector = cv2.dnn_TextDetectionModel_DB(args.det_model)
                text_detector.setInputParams(
                    1.0 / 255.0,
                    (960, 960),
                    (122.6789, 116.6687, 104.0069),
                    True,
                    False,
                )
                text_detector.setBinaryThreshold(0.3)
                text_detector.setPolygonThreshold(0.2)
                text_detector.setUnclipRatio(2.0)

                db_input = preprocess_for_dbnet(original_image, target_size=736)
                rects, _ = text_detector.detectTextRectangles(db_input)

                rects = filter_boxes_shapely(rects)

                h_orig, w_orig = original_image.shape[:2]
                h_db, w_db = db_input.shape[:2]
                scale_h, scale_w = h_orig / h_db, w_orig / w_db

                scaled_rects = []
                for box in rects:
                    (cx, cy), (w, h), angle = box
                    new_w = w * scale_w * 1.15
                    new_h = h * scale_h * 1.10
                    new_cx, new_cy = cx * scale_w, cy * scale_h
                    scaled_rects.append(((new_cx, new_cy), (new_w, new_h), angle))

                scaled_rects = sort_boxes(scaled_rects)

                if not scaled_rects:
                    print("[WARN] No text detected. Fallback to full image.")
                    crops_to_process.append(original_image)
                else:
                    print(f"[INFO] Detected {len(scaled_rects)} text lines.")
                    for box in scaled_rects:
                        crops_to_process.append(get_cropped_image(original_image, box))

            except Exception as e:
                print(f"[ERROR] DBNet failed: {e}. Fallback to full image.")
                traceback.print_exc()
                crops_to_process.append(original_image)

    # Vòng lặp nhận dạng: tiền xử lý -> encode -> beam search + LM -> decode text.
    final_lines = []
    print(f"[INFO] Running Recognition on {len(crops_to_process)} segments...")

    for i, crop in enumerate(crops_to_process):
        try:
            if len(crop.shape) == 3 and crop.shape[2] == 3:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                crop_gray = crop

            img_resized = resize_keep_height(crop_gray, IMG_H)
            h, w = img_resized.shape

            if w < PAD_W:
                padded_img = np.pad(
                    img_resized,
                    ((0, 0), (0, PAD_W - w)),
                    mode='constant',
                    constant_values=255,
                )
            elif w > PAD_W:
                padded_img = cv2.resize(img_resized, (PAD_W, IMG_H))
            else:
                padded_img = img_resized

            # Debug: lưu ảnh sau binarization để kiểm tra pipeline tiền xử lý.
            final_img = gaussian_blur_and_adapt_thresh(padded_img)
            cv2.imwrite(f"debug_input_tensor_{i}.jpg", final_img)

            tensor = torch.from_numpy((final_img.astype(np.float32) / 255.0)[None, ...])
            tensor = tensor.unsqueeze(0).to(device)

            # Tạo mask theo width hợp lệ (downsample x8) để bỏ vùng padding.
            cnn_output_w = PAD_W // 8
            valid_len = min(img_resized.shape[1] // 8, cnn_output_w)
            mask = torch.ones((1, cnn_output_w), dtype=torch.bool, device=device)
            if valid_len > 0:
                mask[0, :valid_len] = False

            with torch.no_grad():
                memory = ocr_model.encode(tensor, mask)
                best_seq_batch = beam_search_decode_with_lm(
                    ocr_model,
                    memory,
                    mask,
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

            best_seq = best_seq_batch[0]

            pred_indices = best_seq[1:]
            try:
                eos_pos = pred_indices.index(EOS_IDX)
                pred_indices = pred_indices[:eos_pos]
            except ValueError:
                pass

            final_indices = [idx for idx in pred_indices if idx != PAD_IDX]
            text = "".join(idx_to_char.get(idx, "") for idx in final_indices)
            final_lines.append(text)

        except Exception as e:
            print(f"[ERROR] Failed on crop {i}: {e}")

    # Ghép các dòng nhận dạng và xuất kết quả.
    full_text = "\n".join(final_lines)
    print("\n" + "=" * 30)
    print("FINAL PREDICTION")
    print("=" * 30)
    print(full_text)
    print("=" * 30)

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"[INFO] Saved to {args.output_file}")


if __name__ == '__main__':
    main()
