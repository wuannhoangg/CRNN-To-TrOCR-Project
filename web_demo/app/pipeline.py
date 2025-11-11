#!/usr/bin/env python
# -*- coding: utf-8 -*-
# """
# HỆ THỐNG OCR HOÀN CHỈNH (PIPELINE 3 GIAI ĐOẠN)
# PHIÊN BẢN V4 - Tự động phát hiện loại ảnh (A4 hay CROP)
# """
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
from tqdm import tqdm
import traceback

# --- Import KenLM ---
try:
    import kenlm
except ImportError:
    print("Lỗi: Không tìm thấy thư viện KenLM.")
    print("Vui lòng kích hoạt môi trường .venv-py311 và chạy: pip install kenlm")
    exit()

def gpu_check() -> str:
    if torch.cuda.is_available(): return 'cuda'
    else: return 'cpu'
def resize_keep_height(gray_img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = gray_img.shape[:2]
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(gray_img, (new_w, target_h))
def gaussian_blur_and_adapt_thresh(gray_img: np.ndarray) -> np.ndarray:
    img = cv2.GaussianBlur(gray_img, (5,5), 0)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
def get_cropped_image(image, box):
    points = cv2.boxPoints(box)
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1); rect[0] = points[np.argmin(s)]; rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1); rect[1] = points[np.argmin(diff)]; rect[3] = points[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl); widthB = np.linalg.norm(tr - tl); maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br); heightB = np.linalg.norm(tl - bl); maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__(); self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1); div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model); pe[:, 0, 0::2] = torch.sin(position * div_term); pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0); x = x + self.pe[:seq_len]; return self.dropout(x)
class VisionTransformerOCR(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 3, num_decoder_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.2):
        super().__init__(); self.d_model = d_model
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2,2)))
        self.input_proj = nn.Linear(512 * 3, d_model); self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(num_classes, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation='gelu', batch_first=False)
        self.fc_out = nn.Linear(d_model, num_classes)
    def process_src(self, src: torch.Tensor) -> torch.Tensor:
        src_features = self.cnn_backbone(src); bs, c, h, w = src_features.shape
        if h != 3: 
            if not hasattr(self, 'adaptive_pool'): self.adaptive_pool = nn.AdaptiveAvgPool2d((3, w))
            src_features = self.adaptive_pool(src_features);
        src_features = src_features.view(bs, c * h, w).permute(2, 0, 1); src_features = self.input_proj(src_features); src_features = self.pos_encoder(src_features)
        return src_features
    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        src_processed = self.process_src(src); return self.transformer.encoder(src_processed, src_key_padding_mask=src_key_padding_mask)
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model); tgt_emb = self.pos_encoder(tgt_emb)
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
def get_word_from_indices(indices: List[int], idx_to_char_map: Dict[int, str], space_char: str) -> str:
    return "".join(idx_to_char_map.get(i, "") for i in indices if idx_to_char_map.get(i, "") != space_char)
def beam_search_decode_with_lm(model, memory, src_key_padding_mask, sos_idx, eos_idx, pad_idx, max_len, beam_width, idx_to_char_map: Dict[int, str], lm_model: kenlm.Model, lm_alpha: float, lm_beta: float):
    device = memory.device; bs = memory.size(1); SPACE_CHAR = " "; SPACE_IDX = -1
    for idx, char in idx_to_char_map.items():
        if char == SPACE_CHAR: SPACE_IDX = idx; break
    if bs != 1: raise ValueError("Pipeline này chỉ hỗ trợ batch size = 1 cho việc nhận diện từng crop.")
    mem = memory; mem_pad_mask = src_key_padding_mask; initial_lm_state = kenlm.State(); lm_model.BeginSentenceWrite(initial_lm_state)
    sequences: List[Tuple[List[int], float, kenlm.State, float, List[int]]] = [([sos_idx], 0.0, initial_lm_state, 0.0, [])]
    for step in range(max_len):
        all_candidates = []; 
        for seq, ocr_score, prev_lm_state, total_lm_score, current_word in sequences:
            last_idx = seq[-1]
            if last_idx == eos_idx:
                combined_score = ocr_score + lm_alpha * total_lm_score + lm_beta * len(seq); all_candidates.append((seq, ocr_score, prev_lm_state, total_lm_score, current_word, combined_score))
                continue
            tgt_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(1); tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)
            with torch.no_grad():
                out = model.decode(tgt_tensor, mem, tgt_mask, mem_pad_mask); ocr_log_probs = F.log_softmax(model.fc_out(out[-1, :, :]), dim=-1)
            top_k_ocr_scores, top_k_indices = ocr_log_probs.topk(beam_width)
            for k in range(beam_width):
                new_idx = top_k_indices[0, k].item()
                if new_idx == pad_idx: continue
                new_char = idx_to_char_map.get(new_idx, ""); new_ocr_score_char = top_k_ocr_scores[0, k].item(); new_seq = seq + [new_idx]; new_ocr_score = ocr_score + new_ocr_score_char
                if new_char == SPACE_CHAR or new_idx == eos_idx:
                    word = get_word_from_indices(current_word, idx_to_char_map, SPACE_CHAR); new_lm_state = kenlm.State(); word_lm_score = 0.0
                    if word: word_lm_score = lm_model.BaseScore(prev_lm_state, word, new_lm_state)
                    else: new_lm_state = prev_lm_state
                    new_total_lm_score = total_lm_score + word_lm_score
                    if new_idx == eos_idx:
                        final_lm_state = kenlm.State(); eos_score = lm_model.BaseScore(new_lm_state, "</s>", final_lm_state)
                        new_total_lm_score += eos_score; new_lm_state = final_lm_state
                    new_current_word = []; combined_score = new_ocr_score + lm_alpha * new_total_lm_score + lm_beta * len(new_seq)
                    all_candidates.append((new_seq, new_ocr_score, new_lm_state, new_total_lm_score, new_current_word, combined_score))
                else:
                    new_current_word = current_word + [new_idx]; new_total_lm_score = total_lm_score; new_lm_state = prev_lm_state
                    combined_score = new_ocr_score + lm_alpha * new_total_lm_score + lm_beta * len(new_seq)
                    all_candidates.append((new_seq, new_ocr_score, new_lm_state, new_total_lm_score, new_current_word, combined_score))
        if not all_candidates: break
        ordered = sorted(all_candidates, key=lambda x: x[5], reverse=True); sequences = [(h[0], h[1], h[2], h[3], h[4]) for h in ordered[:beam_width]] 
        if all(s[0][-1] == eos_idx for s in sequences): break
    return sequences[0][0] if sequences else [sos_idx, eos_idx]

# -------------------------------------------------------------
# --- HÀM MAIN: CHẠY PIPELINE HOÀN CHỈNH ---
# -------------------------------------------------------------

def run_pipeline(args):
    device = torch.device("cpu")
    print(f'[INFO] Using device: {device}')

    # --- BƯỚC 0: TẢI MODEL ---

    # 1. Tải model Recognition (BẮT BUỘC)
    print(f"[INFO] Đang tải Recognition model (Transformer) từ: {args.recognition_model}")
    try:
        ckpt = torch.load(args.recognition_model, map_location=device)
        char_list = ckpt['charset']
        char_to_idx = {c: i for i, c in enumerate(char_list)}
        idx_to_char = {i: c for c, i in char_to_idx.items()}
        SOS_IDX, EOS_IDX, PAD_IDX = char_to_idx['[SOS]'], char_to_idx['[EOS]'], char_to_idx['[PAD]']
        num_classes = len(char_list)
        
        ocr_model = VisionTransformerOCR(num_classes=num_classes)
        ocr_model.load_state_dict(ckpt['model_state'])
        ocr_model.to(device)
        ocr_model.eval()
        
        IMG_H = ckpt['height']
        PAD_W = ckpt['pad_w']
        CNN_OUTPUT_W = PAD_W // 8
        print('[INFO] Recognition model (Transformer) đã tải thành công.')
    except Exception as e:
        print(f"[LỖI] Không thể tải Recognition checkpoint. Lỗi: {e}")
        traceback.print_exc()
        return

    # 2. Tải model Language Model (BẮT BUỘC)
    print(f"[INFO] Đang tải Language Model (KenLM) từ: {args.lm_model}")
    try:
        lm_model = kenlm.Model(args.lm_model)
        print('[INFO] Language Model (KenLM) đã tải thành công.')
    except Exception as e:
        print(f"[LỖI] Không thể tải KenLM model. Lỗi: {e}")
        traceback.print_exc()
        return
        
    # 3. Tải model Detection (BẮT BUỘC)
    print(f"[INFO] Đang tải Detection model (DBNet) từ: {args.detection_model}")
    try:
        text_detector = cv2.dnn_TextDetectionModel_DB(args.detection_model)
        text_detector.setInputParams(1.0 / 255.0, (736, 736), (122.6789, 116.6687, 104.0069), True, False)
        # Dùng giá trị đã sửa, "nhạy" hơn
        text_detector.setBinaryThreshold(0.2)
        text_detector.setPolygonThreshold(0.3)
        text_detector.setUnclipRatio(4.0)
    except Exception as e:
        print(f"[LỖI] Không thể tải DBNet model. Bạn đã tải file '.onnx' chưa? Lỗi: {e}")
        traceback.print_exc()
        return

    # --- BƯỚC 1: ĐỌC ẢNH VÀ CHẠY DETECTION (LOGIC TỰ ĐỘNG) ---
    print(f"\n[INFO] Đang xử lý ảnh: {args.input_image}")
    image = cv2.imread(args.input_image)
    if image is None:
        print(f"[LỖI] Không thể đọc ảnh đầu vào.")
        return
        
    original_image_for_cropping = image.copy()
    
    # Danh sách các ảnh con để đưa vào Giai đoạn 2
    all_crops = []
    
    print("[INFO] Giai đoạn 1: Đang phát hiện văn bản (Detection)...")
    rotated_rects, _ = text_detector.detectTextRectangles(image)
    print(f"[INFO] Đã tìm thấy {len(rotated_rects)} dòng văn bản.")
    
    if len(rotated_rects) == 0:
        # --- LOGIC TỰ ĐỘNG ---
        print("[WARN] Không tìm thấy box nào. Giả định đây là ảnh đã cắt (pre-cropped).")
        print("[INFO] Chuyển sang Giai đoạn 2 với toàn bộ ảnh.")
        all_crops.append(original_image_for_cropping)
        # Không cần lưu ảnh debug vì không có box
    else:
        # --- LOGIC BÌNH THƯỜNG ---
        print(f"[INFO] Đã tìm thấy {len(rotated_rects)} box. Tiến hành cắt và nhận diện.")
        image_with_boxes = original_image_for_cropping.copy()
        for box in rotated_rects:
            points = cv2.boxPoints(box).astype(int)
            cv2.drawContours(image_with_boxes, [points], 0, (0, 255, 0), 2)
        cv2.imwrite("output_detected_boxes.jpg", image_with_boxes)
        print("[INFO] Đã lưu ảnh debug Giai đoạn 1 tại: output_detected_boxes.jpg")
        
        # Lấy tất cả ảnh crop
        for box in rotated_rects:
            all_crops.append(get_cropped_image(original_image_for_cropping, box))
            
    
    final_lines = []
    
    # --- BƯỚC 2 & 3: CHẠY RECOGNITION + LM TRÊN TỪNG CROP ---
    print(f"[INFO] Giai đoạn 2 & 3: Đang nhận diện {len(all_crops)} ảnh crop...")
    
    for i, crop in enumerate(tqdm(all_crops, desc="Đang đọc từng dòng")):
        try:
            # 1. Cắt ảnh con (đã có, là `crop`)
            
            # 2. Tiền xử lý ảnh con
            if len(crop.shape) == 3 and crop.shape[2] == 3:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                crop_gray = crop # Giả sử nó đã là ảnh xám

            img_resized = resize_keep_height(crop_gray, IMG_H)
            original_width_after_resize = img_resized.shape[1]
            
            h, w = img_resized.shape
            if w < PAD_W:
                padded_img = np.pad(img_resized, ((0,0),(0, PAD_W - w)), mode='median')
            elif w > PAD_W:
                padded_img = cv2.resize(img_resized, (PAD_W, IMG_H))
            else:
                padded_img = img_resized

            final_img_processed = gaussian_blur_and_adapt_thresh(padded_img)
            tensor = torch.from_numpy((final_img_processed.astype(np.float32) / 255.0)[None, ...])
            tensor = tensor.unsqueeze(0).to(device) # Shape: [1, 1, IMG_H, PAD_W]

            # 3. Tạo padding mask
            valid_len = min(original_width_after_resize // 8, CNN_OUTPUT_W)
            mask = torch.ones((1, CNN_OUTPUT_W), dtype=torch.bool, device=device)
            if valid_len > 0:
                mask[0, :valid_len] = False

            # 4. Chạy Giai đoạn 2 (Encode) và 3 (Decode + LM)
            with torch.no_grad():
                memory = ocr_model.encode(tensor, mask)
                
                seq_indices = beam_search_decode_with_lm(
                    ocr_model, memory, mask, 
                    SOS_IDX, EOS_IDX, PAD_IDX, 
                    150, args.beam_width, 
                    idx_to_char,
                    lm_model,
                    args.lm_alpha,
                    args.lm_beta
                )
            
            # 5. Dọn dẹp kết quả
            pred_indices = seq_indices[1:] 
            try:
                eos_pos = pred_indices.index(EOS_IDX)
                pred_indices = pred_indices[:eos_pos]
            except ValueError: pass
            
            final_indices = [idx for idx in pred_indices if idx != PAD_IDX]
            text = "".join(idx_to_char.get(idx, "") for idx in final_indices)
            
            final_lines.append(text)

        except Exception as e:
            print(f"\n[LỖI NGHIÊM TRỌNG] Gặp lỗi khi xử lý dòng {i+1}. Bỏ qua dòng này.")
            traceback.print_exc() 
            
    # --- BƯỚC 4: HOÀN TẤT ---
    full_text_result = "\n".join(final_lines)
    
    print("\n" + "="*30)
    print(" KẾT QUẢ OCR HOÀN CHỈNH")
    print("="*30)
    print(full_text_result)
    print("="*30)
    
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(full_text_result)
            print(f"[INFO] Đã lưu kết quả vào file: {args.output_file}")
        except Exception as e:
            print(f"[LỖI] Không thể lưu file output: {e}")
