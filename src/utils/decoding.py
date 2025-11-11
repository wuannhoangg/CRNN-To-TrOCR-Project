# utils/decoding.py

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

try:
    import kenlm
except ImportError:
    print("[WARN] Gói 'kenlm' chưa được cài đặt. Chức năng decode với LM sẽ không hoạt động.")
    kenlm = None

def greedy_decode(logits: torch.Tensor, blank_id: int) -> List[List[int]]:
    """
    Giải mã Greedy cho CTC (dùng cho CRNN).
    """
    preds = logits.detach().argmax(-1)  # (T,B)
    T, B = preds.shape
    seqs = []
    for b in range(B):
        prev = -1
        seq = []
        for t in range(T):
            p = preds[t, b].item()
            if p != blank_id and p != prev:
                seq.append(p)
            prev = p
        seqs.append(seq)
    return seqs

def beam_search_decode(
    model,
    memory,
    src_key_padding_mask,
    sos_idx,
    eos_idx,
    max_len,
    beam_width,
    alpha=0.7
) -> List[List[int]]:
    """
    Giải mã Beam Search (không có LM) cho Transformer.
    """
    device = memory.device
    bs = memory.size(1)
    final_hypotheses = [[] for _ in range(bs)]

    for i in range(bs):
        mem = memory[:, i:i+1, :]
        mem_pad_mask = src_key_padding_mask[i:i+1, :]
        sequences = [[[sos_idx], 0.0]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == eos_idx:
                    all_candidates.append([seq, score])
                    continue

                tgt_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(1)
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)

                with torch.no_grad():
                    out = model.decode(tgt_tensor, mem, tgt_mask, mem_pad_mask)
                    prob = F.log_softmax(model.fc_out(out[-1, :, :]), dim=-1)

                top_k_scores, top_k_indices = prob.topk(beam_width)
                
                for k in range(beam_width):
                    new_seq = seq + [top_k_indices[0, k].item()]
                    new_score = score + top_k_scores[0, k].item()
                    all_candidates.append([new_seq, new_score])
            
            ordered = sorted(all_candidates, key=lambda x: x[1] / ((len(x[0]) ** alpha) + 1e-9), reverse=True)
            sequences = ordered[:beam_width]

            if all(s[0][-1] == eos_idx for s in sequences):
                break
        
        final_hypotheses[i] = sequences[0][0]
    return final_hypotheses

# --- Các hàm cho Beam Search với LM ---

def get_word_from_indices(indices: List[int], idx_to_char_map: Dict[int, str], space_char: str) -> str:
    """Helper: Chuyển list index (của 1 từ) sang string"""
    return "".join(idx_to_char_map.get(i, "") for i in indices if idx_to_char_map.get(i, "") != space_char)

def beam_search_decode_with_lm(
    model,
    memory,
    src_key_padding_mask,
    sos_idx,
    eos_idx,
    pad_idx,
    max_len,
    beam_width,
    idx_to_char_map: Dict[int, str],
    lm_model: 'kenlm.Model',
    lm_alpha: float,
    lm_beta: float
) -> List[List[int]]:
    """
    Giải mã Beam Search (CÓ LM) cho Transformer.
    Trích từ: evaluate_with_lm.py
    """
    if kenlm is None:
        raise ImportError("Gói KenLM là bắt buộc. Vui lòng cài đặt.")
        
    device = memory.device
    bs = memory.size(1)
    final_hypotheses = [[] for _ in range(bs)]
    
    SPACE_CHAR = " "
    SPACE_IDX = -1
    for idx, char in idx_to_char_map.items():
        if char == SPACE_CHAR:
            SPACE_IDX = idx
            break
    if SPACE_IDX == -1:
        print("[WARN] Không tìm thấy dấu cách (' ') trong idx_to_char_map.")

    for i in range(bs):
        mem = memory[:, i:i+1, :]
        mem_pad_mask = src_key_padding_mask[i:i+1, :]
        
        initial_lm_state = kenlm.State()
        lm_model.BeginSentenceWrite(initial_lm_state)
        
        sequences: List[Tuple[List[int], float, kenlm.State, float, List[int]]] = [
            ([sos_idx], 0.0, initial_lm_state, 0.0, [])
        ]

        for step in range(max_len):
            all_candidates = []
            
            for seq, ocr_score, prev_lm_state, total_lm_score, current_word in sequences:
                last_idx = seq[-1]
                if last_idx == eos_idx:
                    combined_score = ocr_score + lm_alpha * total_lm_score + lm_beta * len(seq)
                    all_candidates.append((seq, ocr_score, prev_lm_state, total_lm_score, current_word, combined_score))
                    continue

                tgt_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(1)
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)
                
                with torch.no_grad():
                    out = model.decode(tgt_tensor, mem, tgt_mask, mem_pad_mask)
                    ocr_log_probs = F.log_softmax(model.fc_out(out[-1, :, :]), dim=-1)

                top_k_ocr_scores, top_k_indices = ocr_log_probs.topk(beam_width)
                
                for k in range(beam_width):
                    new_idx = top_k_indices[0, k].item()
                    if new_idx == pad_idx: continue
                    
                    new_char = idx_to_char_map.get(new_idx, "")
                    new_ocr_score_char = top_k_ocr_scores[0, k].item()
                    new_seq = seq + [new_idx]
                    new_ocr_score = ocr_score + new_ocr_score_char

                    if new_char == SPACE_CHAR or new_idx == eos_idx:
                        word = get_word_from_indices(current_word, idx_to_char_map, SPACE_CHAR)
                        new_lm_state = kenlm.State()
                        word_lm_score = 0.0
                        if word:
                            word_lm_score = lm_model.BaseScore(prev_lm_state, word, new_lm_state)
                        else:
                            new_lm_state = prev_lm_state
                        
                        new_total_lm_score = total_lm_score + word_lm_score
                        
                        if new_idx == eos_idx:
                            final_lm_state = kenlm.State()
                            eos_score = lm_model.BaseScore(new_lm_state, "</s>", final_lm_state)
                            new_total_lm_score += eos_score
                            new_lm_state = final_lm_state
                        
                        new_current_word = []
                        combined_score = new_ocr_score + lm_alpha * new_total_lm_score + lm_beta * len(new_seq)
                        all_candidates.append((new_seq, new_ocr_score, new_lm_state, new_total_lm_score, new_current_word, combined_score))
                    else:
                        new_current_word = current_word + [new_idx]
                        new_total_lm_score = total_lm_score
                        new_lm_state = prev_lm_state
                        combined_score = new_ocr_score + lm_alpha * new_total_lm_score + lm_beta * len(new_seq)
                        all_candidates.append((new_seq, new_ocr_score, new_lm_state, new_total_lm_score, new_current_word, combined_score))
            
            if not all_candidates:
                break # Không có ứng cử viên nào
                
            ordered = sorted(all_candidates, key=lambda x: x[5], reverse=True)
            sequences = [(h[0], h[1], h[2], h[3], h[4]) for h in ordered[:beam_width]]
            
            if all(s[0][-1] == eos_idx for s in sequences):
                break
        
        final_hypotheses[i] = sequences[0][0]
    return final_hypotheses