# utils/preprocessing.py

import os
import json
import unicodedata
import math
import random
from typing import List, Dict

import numpy as np
import cv2
import torch
from tqdm import tqdm
try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    print("Lỗi: Cần cài đặt Pillow: pip install Pillow")


def gpu_check(force_cpu: bool = False, require_gpu: bool = False) -> str:
    """
    Kiểm tra GPU.
    """
    if force_cpu:
        print('[INFO] Forced CPU mode (--cpu).')
        return 'cpu'
    if torch.cuda.is_available():
        print(f'[OK] Found GPU: {torch.cuda.get_device_name(0)}')
        return 'cuda'
    else:
        if require_gpu:
            raise SystemError('GPU required (--require_gpu) nhưng không tìm thấy.')
        print('[WARN] No GPU detected. Falling back to CPU.')
        return 'cpu'

def ensure_dir(p: str):
    """
    Đảm bảo thư mục tồn tại.
    """
    os.makedirs(p, exist_ok=True)

def auto_find_labels_json(base: str) -> str:
    """
    Tự động tìm file labels.json.
    """
    if os.path.isfile(base) and base.lower().endswith('.json'):
        return base
    search_dir = base if os.path.isdir(base) else os.path.dirname(base)
    if not os.path.isdir(search_dir):
        return ''
    cands = [str(p) for p in pathlib.Path(search_dir).glob('*.json')]
    if not cands:
        return ''
    label_like = [c for c in cands if 'label' in os.path.basename(c).lower()]
    return label_like[0] if label_like else cands[0]

# --- Các hàm xử lý Ảnh ---

def resize_keep_height(gray_img: np.ndarray, target_h: int) -> np.ndarray:
    """
    Thay đổi kích thước ảnh về chiều cao `target_h` và giữ nguyên tỷ lệ.
    """
    h, w = gray_img.shape[:2]
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(gray_img, (new_w, target_h))

def gaussian_blur_and_adapt_thresh(gray_img: np.ndarray) -> np.ndarray:
    """
    Áp dụng Gaussian Blur và Adaptive Threshold.
    Đây là bước tiền xử lý chính cho cả CRNN và Transformer.
    """
    img = cv2.GaussianBlur(gray_img, (5,5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 4)
    return img

def apply_augs_pil(img_pil: Image.Image) -> Image.Image:
    """
    Áp dụng các phép tăng cường (augmentation) bằng PIL.
    Dùng cho TransformerDataset.
    """
    fill_c = 255 # Giả định nền trắng sau khi resize
    
    if random.random() < 0.5:
        angle = random.uniform(-3, 3)
        img_pil = img_pil.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=fill_c)

    if random.random() < 0.5:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.8, 1.2))

    if random.random() < 0.5:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.8, 1.2))

    if random.random() < 0.3:
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
    
    return img_pil

# --- Các hàm đọc/chuẩn hóa Dữ liệu ---

def normalize_text(s: str) -> str:
    """
    Chuẩn hóa text (Unicode NFC, xóa khoảng trắng thừa).
    """
    if s is None: return ""
    s = unicodedata.normalize("NFC", s)
    return " ".join(s.strip().split())

def read_labels(labels_path: str, normalize: bool = False) -> Dict[str, str]:
    """
    Đọc file labels.json và trả về dict {basename: label}.
    """
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f'labels_json không tồn tại: {labels_path}')
    
    with open(labels_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    normalized_data = {}
    for k, v in data.items():
        base_k = os.path.basename(k)
        label = normalize_text(v) if normalize else v
        if label or label == "": # Chấp nhận cả label rỗng
            normalized_data[base_k] = label
            
    return normalized_data

def read_alphabet(alphabet_path: str) -> List[str]:
    """
    Đọc file alphabet (dùng cho Transformer).
    """
    if not os.path.isfile(alphabet_path):
        raise FileNotFoundError(f'Alphabet file not found: {alphabet_path}')
    charset = set()
    with open(alphabet_path, 'r', encoding='utf8') as f:
        for line in f:
            char = line.strip()
            if char:
                if char == '<space>':
                    charset.add(' ')
                else:
                    charset.add(char)
    if ' ' not in charset:
        charset.add(' ')
    return sorted(list(charset))

def build_char_list(labels: Dict[str, str]) -> List[str]:
    """
    Tự động xây dựng bộ ký tự từ labels (dùng cho CRNN).
    """
    charset = set()
    for txt in labels.values():
        if not txt: continue
        charset.update(list(txt))
    return sorted(list(charset))

def encode_to_labels(txt: str, char_to_idx: Dict[str,int]) -> List[int]:
    """
    Mã hóa chuỗi text sang list các
    index (dùng cho CRNN).
    """
    return [char_to_idx[ch] for ch in txt if ch in char_to_idx]

def calc_max_padded_width(image_paths: List[str], target_h: int) -> int:
    """
    Tính toán độ rộng (width) tối đa sau khi resize
    để dùng cho việc padding.
    """
    max_w = 1
    print("[INFO] Đang tính toán max padded width...")
    for p in tqdm(image_paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img_resized = resize_keep_height(img, target_h)
        max_w = max(max_w, img_resized.shape[1])
    
    # Kích thước ảnh đầu vào CNN của Transformer là 8
    # Của CRNN cũng là 8 (3 lần MaxPool 3x3 rồi squeeze(2) và pool (3,1))
    # => Chúng ta dùng chung downsample_factor = 8
    downsample_factor = 8
    
    # Làm tròn lên bội số gần nhất của 8
    return (max_w + downsample_factor - 1) // downsample_factor * downsample_factor
