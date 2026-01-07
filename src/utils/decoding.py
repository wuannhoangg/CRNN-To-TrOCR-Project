# src/utils/decoding.py
"""
Các hàm giải mã (decoding) cho OCR.

Bao gồm:
- Greedy decoding cho CTC (CRNN).
- Beam search decoding cho Transformer (không dùng LM).
- Beam search decoding cho Transformer có tích hợp KenLM để cải thiện độ chính xác.

Ghi chú:
- Các hàm beam search ở đây giả định `model.decode(...)` trả về hidden states của decoder
  và `model.fc_out(...)` ánh xạ hidden state -> logits theo vocabulary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

try:
    import kenlm
except ImportError:
    print("[WARN] Gói 'kenlm' chưa được cài đặt. Chức năng decode với LM sẽ không hoạt động.")
    kenlm = None


def greedy_decode(logits: torch.Tensor, blank_id: int) -> List[List[int]]:
    """Giải mã greedy cho CTC (CRNN).

    Quy tắc:
    - Lấy argmax theo lớp ở mỗi timestep.
    - Loại bỏ lặp liên tiếp (collapse repeats).
    - Bỏ token blank.

    Args:
        logits (torch.Tensor): Tensor shape (T, B, C) hoặc (T, B) sau argmax.
        blank_id (int): Index của lớp blank trong CTC.

    Returns:
        List[List[int]]: Danh sách chuỗi indices cho từng phần tử trong batch.
    """
    preds = logits.detach().argmax(-1)  # (T, B)
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
    alpha=0.7,
    repetition_penalty=1.0,
):
    """Beam search decoding tiêu chuẩn cho Transformer (batch-first).

    Args:
        model: Mô hình Transformer OCR (cần có `decode(...)` và `fc_out(...)`).
        memory (torch.Tensor): Output encoder (batch-first).
        src_key_padding_mask (torch.BoolTensor): Mask padding cho encoder (B, S).
        sos_idx (int): Index token [SOS].
        eos_idx (int): Index token [EOS].
        max_len (int): Độ dài tối đa khi decode.
        beam_width (int): Số nhánh giữ lại ở mỗi bước.
        alpha (float): Hệ số length normalization.
        repetition_penalty (float): Hệ số phạt lặp token (>=1.0).

    Returns:
        List[List[int]]: Chuỗi token indices cho từng mẫu trong batch.
    """
    device = memory.device
    bs = memory.size(0)
    final_hypotheses = [[] for _ in range(bs)]

    for i in range(bs):
        mem = memory[i:i + 1, :, :].contiguous()
        mem_pad_mask = src_key_padding_mask[i:i + 1, :]
        sequences = [[[sos_idx], 0.0]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == eos_idx:
                    all_candidates.append([seq, score])
                    continue

                # Tạo input decoder dạng (B=1, L) theo batch-first.
                tgt_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)

                with torch.no_grad():
                    out = model.decode(tgt_tensor, mem, tgt_mask, mem_pad_mask)
                    prob = F.log_softmax(model.fc_out(out[:, -1, :]), dim=-1)

                # Phạt token đã xuất hiện để giảm lặp (tuỳ chọn).
                if repetition_penalty > 1.0:
                    previous_tokens = set(seq)
                    for token_id in previous_tokens:
                        prob[0, token_id] *= repetition_penalty

                top_k_scores, top_k_indices = prob.topk(beam_width)

                for k in range(beam_width):
                    new_seq = seq + [top_k_indices[0, k].item()]
                    new_score = score + top_k_scores[0, k].item()
                    all_candidates.append([new_seq, new_score])

            # Xếp hạng theo score có length normalization để tránh ưu tiên chuỗi quá ngắn.
            ordered = sorted(
                all_candidates,
                key=lambda x: x[1] / ((len(x[0]) ** alpha) + 1e-9),
                reverse=True,
            )
            sequences = ordered[:beam_width]

            # Dừng sớm nếu tất cả hypothesis đều kết thúc bằng EOS.
            if all(s[0][-1] == eos_idx for s in sequences):
                break

        final_hypotheses[i] = sequences[0][0]

    return final_hypotheses


def get_word_from_indices(
    indices: List[int],
    idx_to_char_map: Dict[int, str],
    space_char: str,
) -> str:
    """Chuyển list indices của một từ thành chuỗi (bỏ ký tự space nếu có).

    Args:
        indices (List[int]): Danh sách token indices thuộc một "từ".
        idx_to_char_map (Dict[int, str]): Bảng ánh xạ index -> ký tự.
        space_char (str): Ký tự được coi là dấu cách.

    Returns:
        str: Từ đã ghép từ indices.
    """
    return "".join(
        idx_to_char_map.get(i, "")
        for i in indices
        if idx_to_char_map.get(i, "") != space_char
    )


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
    lm_beta: float,
):
    """Beam search decoding có tích hợp KenLM (batch-first).

    Ý tưởng:
    - Score tổng = OCR log-prob + lm_alpha * LM_score + lm_beta * length_bonus.
    - LM_score được cộng theo "từ" (khi gặp space hoặc EOS) để giảm số lần gọi KenLM.

    Args:
        model: Mô hình Transformer OCR.
        memory (torch.Tensor): Output encoder.
        src_key_padding_mask (torch.BoolTensor): Mask padding (B, S).
        sos_idx (int): Token [SOS].
        eos_idx (int): Token [EOS].
        pad_idx (int): Token [PAD] (bị bỏ qua khi decode).
        max_len (int): Độ dài tối đa khi decode.
        beam_width (int): Beam width.
        idx_to_char_map (Dict[int, str]): Bảng ánh xạ token index -> ký tự.
        lm_model (kenlm.Model): KenLM model đã load.
        lm_alpha (float): Trọng số LM score.
        lm_beta (float): Trọng số thưởng độ dài (length bonus).

    Returns:
        List[List[int]]: Chuỗi token indices cho từng mẫu trong batch.

    Raises:
        ImportError: Nếu KenLM chưa được cài đặt.
    """
    if kenlm is None:
        raise ImportError("Gói KenLM là bắt buộc. Vui lòng cài đặt.")

    device = memory.device
    bs = memory.size(0)
    final_hypotheses = [[] for _ in range(bs)]

    # Xác định token space (nếu tồn tại trong vocab) để chốt "từ" khi tính LM.
    SPACE_CHAR = " "
    SPACE_IDX = -1
    for idx, char in idx_to_char_map.items():
        if char == SPACE_CHAR:
            SPACE_IDX = idx
            break

    for i in range(bs):
        mem = memory[i:i + 1, :, :].contiguous()
        mem_pad_mask = src_key_padding_mask[i:i + 1, :]

        # Trạng thái LM ban đầu: bắt đầu câu.
        initial_lm_state = kenlm.State()
        lm_model.BeginSentenceWrite(initial_lm_state)

        # Mỗi hypothesis giữ: seq, ocr_score, lm_state, total_lm_score, current_word_indices.
        sequences = [([sos_idx], 0.0, initial_lm_state, 0.0, [])]

        for step in range(max_len):
            all_candidates = []

            for seq, ocr_score, prev_lm_state, total_lm_score, current_word in sequences:
                last_idx = seq[-1]

                # Nếu đã kết thúc câu, chỉ tính score tổng và giữ lại.
                if last_idx == eos_idx:
                    combined_score = ocr_score + lm_alpha * total_lm_score + lm_beta * len(seq)
                    all_candidates.append(
                        (seq, ocr_score, prev_lm_state, total_lm_score, current_word, combined_score)
                    )
                    continue

                tgt_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)

                with torch.no_grad():
                    out = model.decode(tgt_tensor, mem, tgt_mask, mem_pad_mask)
                    ocr_log_probs = F.log_softmax(model.fc_out(out[:, -1, :]), dim=-1)

                top_k_ocr_scores, top_k_indices = ocr_log_probs.topk(beam_width)

                for k in range(beam_width):
                    new_idx = top_k_indices[0, k].item()
                    if new_idx == pad_idx:
                        continue

                    new_char = idx_to_char_map.get(new_idx, "")
                    new_ocr_score_char = top_k_ocr_scores[0, k].item()

                    new_seq = seq + [new_idx]
                    new_ocr_score = ocr_score + new_ocr_score_char

                    # Khi gặp space hoặc EOS, chốt từ hiện tại để cập nhật LM score.
                    if new_char == SPACE_CHAR or new_idx == eos_idx:
                        word = get_word_from_indices(current_word, idx_to_char_map, SPACE_CHAR)

                        new_lm_state = kenlm.State()
                        word_lm_score = 0.0
                        if word:
                            word_lm_score = lm_model.BaseScore(prev_lm_state, word, new_lm_state)
                        else:
                            new_lm_state = prev_lm_state

                        new_total_lm_score = total_lm_score + word_lm_score

                        # Nếu kết thúc câu thì cộng thêm score cho token </s>.
                        if new_idx == eos_idx:
                            final_lm_state = kenlm.State()
                            eos_score = lm_model.BaseScore(new_lm_state, "</s>", final_lm_state)
                            new_total_lm_score += eos_score
                            new_lm_state = final_lm_state

                        new_current_word = []
                        combined_score = (
                            new_ocr_score
                            + lm_alpha * new_total_lm_score
                            + lm_beta * len(new_seq)
                        )
                        all_candidates.append(
                            (new_seq, new_ocr_score, new_lm_state, new_total_lm_score, new_current_word, combined_score)
                        )
                    else:
                        # Nếu chưa kết thúc từ, chỉ mở rộng current_word và giữ nguyên LM state.
                        new_current_word = current_word + [new_idx]
                        new_total_lm_score = total_lm_score
                        new_lm_state = prev_lm_state
                        combined_score = (
                            new_ocr_score
                            + lm_alpha * new_total_lm_score
                            + lm_beta * len(new_seq)
                        )
                        all_candidates.append(
                            (new_seq, new_ocr_score, new_lm_state, new_total_lm_score, new_current_word, combined_score)
                        )

            if not all_candidates:
                break

            # Chọn top hypotheses theo combined_score.
            ordered = sorted(all_candidates, key=lambda x: x[5], reverse=True)
            sequences = [(h[0], h[1], h[2], h[3], h[4]) for h in ordered[:beam_width]]

            if all(s[0][-1] == eos_idx for s in sequences):
                break

        final_hypotheses[i] = sequences[0][0]

    return final_hypotheses
