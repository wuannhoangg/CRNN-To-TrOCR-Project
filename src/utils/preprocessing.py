# utils/preprocessing.py
"""
Tiện ích tiền xử lý dữ liệu cho OCR.

Bao gồm:
- Kiểm tra thiết bị chạy (GPU/CPU).
- Các hàm xử lý ảnh: resize theo height, binarization (blur + adaptive threshold),
  augmentation bằng PIL (phục vụ Transformer).
- Các hàm đọc/chuẩn hoá nhãn và xây dựng charset.
- Tính toán pad width tối đa sau resize để thống nhất kích thước đầu vào.

Ghi chú:
- Các hàm trong module này được dùng chung cho cả pipeline CRNN (CTC) và Transformer (Seq2Seq).
"""

import os
import json
import unicodedata
import math
import random
import pathlib
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
    """Kiểm tra khả năng sử dụng GPU và trả về chế độ chạy.

    Args:
        force_cpu (bool): Nếu True, ép chạy CPU (bỏ qua GPU).
        require_gpu (bool): Nếu True, báo lỗi khi không có GPU.

    Returns:
        str: 'cuda' nếu có GPU và không bị ép CPU, ngược lại 'cpu'.

    Raises:
        SystemError: Nếu require_gpu=True nhưng không phát hiện GPU.
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
    """Đảm bảo thư mục tồn tại; tạo mới nếu chưa có.

    Args:
        p (str): Đường dẫn thư mục.
    """
    os.makedirs(p, exist_ok=True)


def auto_find_labels_json(base: str) -> str:
    """Tự động tìm file JSON nhãn trong một thư mục (hoặc từ đường dẫn gợi ý).

    Quy tắc:
    - Nếu `base` là file .json hợp lệ thì trả về luôn.
    - Nếu `base` là thư mục (hoặc file path), tìm các .json trong thư mục đó.
    - Ưu tiên file có chứa 'label' trong tên, nếu không có thì chọn file .json đầu tiên.

    Args:
        base (str): Đường dẫn file hoặc thư mục.

    Returns:
        str: Đường dẫn labels.json nếu tìm thấy, ngược lại trả về chuỗi rỗng.
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


def resize_keep_height(gray_img: np.ndarray, target_h: int) -> np.ndarray:
    """Resize ảnh grayscale về chiều cao `target_h` và giữ nguyên tỷ lệ.

    Args:
        gray_img (np.ndarray): Ảnh grayscale (H, W).
        target_h (int): Chiều cao mục tiêu.

    Returns:
        np.ndarray: Ảnh đã resize có shape (target_h, new_w).
    """
    h, w = gray_img.shape[:2]
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(gray_img, (new_w, target_h))


def gaussian_blur_and_adapt_thresh(gray_img: np.ndarray) -> np.ndarray:
    """Tiền xử lý ảnh bằng Gaussian Blur và Adaptive Threshold (binary invert).

    Mục tiêu: làm rõ nét vùng chữ để hỗ trợ OCR.

    Args:
        gray_img (np.ndarray): Ảnh grayscale.

    Returns:
        np.ndarray: Ảnh nhị phân (0/255) sau adaptive threshold.
    """
    img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        4,
    )
    return img


def apply_augs_pil(img_pil: Image.Image) -> Image.Image:
    """Áp dụng augmentation bằng PIL (phục vụ TransformerDataset).

    Các phép tăng cường có thể bao gồm:
    - Xoay nhẹ (rotation)
    - Thay đổi độ sáng (brightness)
    - Thay đổi độ tương phản (contrast)
    - Làm mờ nhẹ (gaussian blur)

    Args:
        img_pil (PIL.Image.Image): Ảnh đầu vào dạng PIL.

    Returns:
        PIL.Image.Image: Ảnh sau augmentation.
    """
    fill_c = 255  # Màu nền (giả định nền trắng)

    if random.random() < 0.5:
        angle = random.uniform(-3, 3)
        img_pil = img_pil.rotate(
            angle,
            resample=Image.BILINEAR,
            expand=False,
            fillcolor=fill_c,
        )

    if random.random() < 0.5:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.8, 1.2))

    if random.random() < 0.5:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.8, 1.2))

    if random.random() < 0.3:
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))

    return img_pil


def normalize_text(s: str) -> str:
    """Chuẩn hoá chuỗi text: Unicode NFC và chuẩn hoá khoảng trắng.

    Args:
        s (str): Chuỗi đầu vào.

    Returns:
        str: Chuỗi đã chuẩn hoá.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    return " ".join(s.strip().split())


def read_labels(labels_path: str, normalize: bool = False) -> Dict[str, str]:
    """Đọc labels.json và trả về dict ánh xạ basename -> label.

    Args:
        labels_path (str): Đường dẫn file labels.json.
        normalize (bool): Nếu True, chuẩn hoá text bằng `normalize_text`.

    Returns:
        Dict[str, str]: Mapping từ tên file (basename) sang label.

    Raises:
        FileNotFoundError: Nếu file labels.json không tồn tại.
    """
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f'labels_json không tồn tại: {labels_path}')

    with open(labels_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    normalized_data = {}
    for k, v in data.items():
        base_k = os.path.basename(k)
        label = normalize_text(v) if normalize else v
        # Chấp nhận cả label rỗng để hỗ trợ bộ test blind set.
        if label or label == "":
            normalized_data[base_k] = label

    return normalized_data


def read_alphabet(alphabet_path: str) -> List[str]:
    """Đọc file alphabet để tạo charset cho Transformer.

    Quy ước:
    - '<space>' được ánh xạ thành ký tự ' ' (space).
    - Luôn đảm bảo charset có chứa ký tự space.

    Args:
        alphabet_path (str): Đường dẫn file alphabet.

    Returns:
        List[str]: Danh sách ký tự (đã sort).
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
    """Tự động xây dựng charset từ tập nhãn (CRNN).

    Args:
        labels (Dict[str, str]): Mapping filename -> label.

    Returns:
        List[str]: Danh sách ký tự (đã sort) xuất hiện trong labels.
    """
    charset = set()
    for txt in labels.values():
        if not txt:
            continue
        charset.update(list(txt))
    return sorted(list(charset))


def encode_to_labels(txt: str, char_to_idx: Dict[str, int]) -> List[int]:
    """Mã hoá chuỗi text thành list indices theo charset (CRNN).

    Args:
        txt (str): Chuỗi đầu vào.
        char_to_idx (Dict[str, int]): Bảng mã ký tự -> index.

    Returns:
        List[int]: Danh sách indices tương ứng với các ký tự có trong charset.
    """
    return [char_to_idx[ch] for ch in txt if ch in char_to_idx]


def calc_max_padded_width(image_paths: List[str], target_h: int) -> int:
    """Tính max width sau resize (theo `target_h`) để dùng cho padding.

    Ghi chú:
    - Hàm này làm tròn lên bội số của 8 để phù hợp với downsample của backbone CNN.

    Args:
        image_paths (List[str]): Danh sách đường dẫn ảnh.
        target_h (int): Chiều cao resize.

    Returns:
        int: Max width sau resize (đã làm tròn bội 8).
    """
    max_w = 1
    print("[INFO] Đang tính toán max padded width...")
    for p in tqdm(image_paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_resized = resize_keep_height(img, target_h)
        max_w = max(max_w, img_resized.shape[1])

    downsample_factor = 8
    return (max_w + downsample_factor - 1) // downsample_factor * downsample_factor
