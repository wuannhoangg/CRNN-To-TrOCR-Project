# utils/metrics.py

import unicodedata
import string
from typing import List, Tuple

def levenshtein(a: List, b: List) -> int:
    """
    Tính khoảng cách Levenshtein (edit distance) giữa 2 list.
    """
    n,m = len(a), len(b)
    dp = list(range(m+1))
    for i in range(1,n+1):
        prev = dp[0]
        dp[0] = i
        for j in range(1,m+1):
            tmp = dp[j]
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev + (0 if a[i-1]==b[j-1] else 1))
            prev = tmp
    return dp[m]

def calculate_metrics(
    preds: List[str], 
    gts: List[str], 
    norm_accent: bool = False, 
    norm_punct: bool = False
) -> Tuple[float,float,float]:
    """
    Tính toán CER, WER, SER.
    """
    def norm(s: str) -> str:
        if norm_accent:
            s = unicodedata.normalize("NFKD", s).encode("ASCII","ignore").decode("ASCII")
        if norm_punct:
            s = s.translate(str.maketrans("", "", string.punctuation))
        return s
        
    CER, WER, SER = [], [], []
    for p, g in zip(preds, gts):
        p2, g2 = norm(p).lower(), norm(g).lower()
        
        # Tính CER
        if len(g2) > 0:
            cer = levenshtein(list(p2), list(g2)) / len(g2)
        elif len(p2) > 0:
            cer = 1.0 # Ground truth rỗng, dự đoán có chữ
        else:
            cer = 0.0 # Cả hai đều rỗng
        
        # Tính WER
        pw, gw = p2.split(), g2.split()
        if len(gw) > 0:
            wer = levenshtein(pw, gw) / len(gw)
        elif len(pw) > 0:
            wer = 1.0
        else:
            wer = 0.0
            
        # Tính SER (Exact match, không normalize)
        ser = 0.0 if p == g else 1.0
        
        CER.append(cer); WER.append(wer); SER.append(ser)
        
    return float(sum(CER)/len(CER)), float(sum(WER)/len(WER)), float(sum(SER)/len(SER))