"""TASK 04 — Module tạo mask.  Người làm: My.  [Data Pipeline • Bắt buộc]

QUY ƯỚC THỐNG NHẤT CHO CẢ NHÓM. Sai quy ước này là sai cả mô hình:

    True (hoặc 1)  nghĩa là ĐƯỢC PHÉP NHÌN
    False (hoặc 0) nghĩa là BỊ CHE

Trong attention thì che bằng torch.finfo(scores.dtype).min, KHÔNG dùng -1e9.

Lỗi mask bị lật ngược là lỗi nguy hiểm nhất của cả đồ án: mô hình nhìn trộm
được đáp án nên loss lúc train giảm rất đẹp, nhưng lúc dịch thật thì ra rác.
Bài test 1 và bài test 2 bắt đúng loại lỗi này.
"""

from __future__ import annotations

import torch


def tao_padding_mask(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(batch, seq_len) -> (batch, 1, 1, seq_len), True ở vị trí token thật."""
    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)


def tao_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """(1, 1, seq_len, seq_len) tam giác dưới — decoder không nhìn thấy từ phía sau."""
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    return mask.unsqueeze(0).unsqueeze(0)


def gop_mask(padding_mask: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
    """Gộp hai mask bằng phép AND theo từng phần tử, có broadcast."""
    return padding_mask & causal_mask
