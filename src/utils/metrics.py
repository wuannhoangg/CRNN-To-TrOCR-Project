# utils/metrics.py
"""
Tính toán các chỉ số đánh giá OCR.

Bao gồm:
- Levenshtein distance (edit distance) cho list/chuỗi.
- CER (Character Error Rate), WER (Word Error Rate), SER (Sentence Error Rate).

Tùy chọn chuẩn hoá:
- norm_accent: bỏ dấu tiếng Việt (unicode NFKD -> ASCII).
- norm_punct: loại bỏ dấu câu theo `string.punctuation`.

Ghi chú:
- SER trong module này được tính theo exact match trên chuỗi gốc (không normalize),
  nhằm phản ánh độ chính xác tuyệt đối theo nhãn ban đầu.
"""

import unicodedata
import string
from typing import List, Tuple


def levenshtein(a: List, b: List) -> int:
    """Tính khoảng cách Levenshtein (edit distance) giữa hai list.

    Args:
        a (List): Dãy đầu vào thứ nhất (ký tự hoặc token).
        b (List): Dãy đầu vào thứ hai (ký tự hoặc token).

    Returns:
        int: Số phép biến đổi tối thiểu (chèn/xoá/thay) để chuyển a -> b.
    """
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if a[i - 1] == b[j - 1] else 1),
            )
            prev = tmp
    return dp[m]


def calculate_metrics(
    preds: List[str],
    gts: List[str],
    norm_accent: bool = False,
    norm_punct: bool = False,
) -> Tuple[float, float, float]:
    """Tính CER, WER, SER cho danh sách dự đoán và ground truth.

    CER/WER được tính trên chuỗi đã normalize (tuỳ chọn) và lower-case.
    SER được tính theo exact match trên chuỗi gốc (không normalize).

    Args:
        preds (List[str]): Danh sách chuỗi dự đoán.
        gts (List[str]): Danh sách chuỗi ground truth.
        norm_accent (bool): Nếu True, loại bỏ dấu (NFKD -> ASCII).
        norm_punct (bool): Nếu True, loại bỏ dấu câu.

    Returns:
        Tuple[float, float, float]: (CER_mean, WER_mean, SER_mean).
    """

    def norm(s: str) -> str:
        """Chuẩn hoá chuỗi theo tuỳ chọn."""
        if norm_accent:
            s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
        if norm_punct:
            s = s.translate(str.maketrans("", "", string.punctuation))
        return s

    CER, WER, SER = [], [], []
    for p, g in zip(preds, gts):
        p2, g2 = norm(p).lower(), norm(g).lower()

        # CER: lỗi trên mức ký tự.
        if len(g2) > 0:
            cer = levenshtein(list(p2), list(g2)) / len(g2)
        elif len(p2) > 0:
            cer = 1.0
        else:
            cer = 0.0

        # WER: lỗi trên mức từ.
        pw, gw = p2.split(), g2.split()
        if len(gw) > 0:
            wer = levenshtein(pw, gw) / len(gw)
        elif len(pw) > 0:
            wer = 1.0
        else:
            wer = 0.0

        # SER: so khớp tuyệt đối theo chuỗi gốc.
        ser = 0.0 if p == g else 1.0

        CER.append(cer)
        WER.append(wer)
        SER.append(ser)

    return (
        float(sum(CER) / len(CER)),
        float(sum(WER) / len(WER)),
        float(sum(SER) / len(SER)),
    )
