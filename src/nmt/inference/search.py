"""TASK 16 — Greedy Search.       Người làm: My.   [Evaluation • Bắt buộc]
TASK 19 — Beam Search & KV Cache. Người làm: Phú.  [Inference • ƯU TIÊN THẤP]

Theo nhận xét 6 của mentor: Beam Search và KV Cache chưa cần vội, ưu tiên chạy
được model trước. Chấm điểm baseline ở TASK 16 dùng Greedy.
TASK 19 là task ĐẦU TIÊN BỊ CẮT nếu trễ tiến độ.

Bài test rẻ nhất mà bắt được hầu hết lỗi beam search:
    đặt beam = 1 thì Beam Search phải cho kết quả TRÙNG KHỚP TỪNG CHỮ với Greedy.
    Nằm ở tests/test_beam_search.py.

Yêu cầu của TASK 19: Beam phải cho BLEU cao hơn Greedy ít nhất 0,5 điểm.
Không cao hơn thì gần như chắc chắn Beam Search đang có lỗi, phải kiểm tra lại
chứ đừng ghi vào báo cáo là "beam không hiệu quả".
"""

from __future__ import annotations

import torch


@torch.no_grad()
def greedy_search(model, src_ids, src_mask, bos_id: int, eos_id: int, do_dai_toi_da: int = 128):
    """Mỗi bước chọn luôn từ có xác suất cao nhất."""
    from nmt.model.masking import tao_causal_mask

    batch_size = src_ids.size(0)
    device = src_ids.device

    # Tiền tính toán bo_nho_encoder (chỉ làm 1 lần)
    bo_nho_encoder = model.encode(src_ids, src_mask)

    # Khởi tạo chuỗi đích với bos_id
    tgt_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

    # Đánh dấu các sequence đã sinh ra eos_id
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(do_dai_toi_da):
        seq_len = tgt_ids.size(1)
        # Tạo causal mask cho tgt_ids hiện tại
        tgt_mask = tao_causal_mask(seq_len, device=device)

        # Giải mã một bước
        decoder_output = model.decode(tgt_ids, bo_nho_encoder, tgt_mask, src_mask)
        logits = model.output_projection(decoder_output)

        # Lấy logits của token cuối cùng vừa được sinh ra
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # Cập nhật kết quả: nếu đã finished thì ghi đè bằng pad_id (ở đây dùng eos_id làm pad cho output)
        # hoặc có thể giữ nguyên next_tokens cũng được vì đằng nào ta cũng sẽ cắt chuỗi ở eos_id.
        next_tokens = next_tokens.masked_fill(finished, eos_id)

        # Gắn token mới vào tgt_ids
        tgt_ids = torch.cat([tgt_ids, next_tokens.unsqueeze(-1)], dim=-1)

        # Cập nhật trạng thái finished
        finished |= (next_tokens == eos_id)

        # Dừng sớm nếu tất cả sequence đều đã sinh ra eos_id
        if finished.all():
            break

    return tgt_ids


@torch.no_grad()
def beam_search(
    model, src_ids, src_mask, bos_id: int, eos_id: int,
    beam_size: int = 4, he_so_phat_do_dai: float = 1.0, do_dai_toi_da: int = 128,
):
    """Giữ lại beam_size phương án tốt nhất ở mỗi bước.

    Kèm hệ số phạt độ dài để mô hình không thiên vị câu quá ngắn.
    """
    raise NotImplementedError("TASK 19 — Phú")
