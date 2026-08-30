"""TASK 13 — Hàm loss.  Người làm: Quân.  [Training • Bắt buộc]

Label Smoothing MẶC ĐỊNH TẮT (label_smoothing = 0.0) theo nhận xét 1 của mentor.
Đặt 0.1 thì bật, khi đó không bắt mô hình tin tuyệt đối vào một đáp án duy nhất.
Ablation A3 (TASK 18) quyết định giữ hay bỏ.

Nhớ bỏ qua vị trí <pad> khi tính loss, nếu không thì mô hình được thưởng vì
đoán đúng token đệm và loss trông đẹp hơn thực tế.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Cross entropy có label smoothing, bỏ qua vị trí padding.

    smoothing = 0.0 thì phải cho ra kết quả GIỐNG HỆT cross entropy thường —
    đây là phép kiểm rẻ tiền nên viết luôn thành unit test.

    Công thức, tính trên từng token rồi mới lấy trung bình:

        loss = (1 - eps) * (-log p[đáp_án])  +  eps * trung_bình_mọi_lớp(-log p)

    Vế đầu là cross entropy thường. Vế sau kéo một phần khối lượng xác suất rải
    đều sang các lớp khác, để mô hình không bị ép tin tuyệt đối vào một đáp án.
    Đặt eps = 0 thì vế sau biến mất và còn đúng cross entropy — chính là tính
    chất mà bài test đối chiếu với F.cross_entropy.

    Cách rải eps đều lên TOÀN BỘ V lớp (thay vì V-1 lớp còn lại) là để khớp đúng
    với `F.cross_entropy(label_smoothing=...)` của PyTorch, nhờ vậy bài test so
    được hai bên ở mọi giá trị eps chứ không riêng eps = 0.

    Trung bình chỉ lấy trên các vị trí THẬT. Chia cho tổng số ô kể cả ô đệm thì
    loss bị pha loãng, batch nhiều đệm trông "tốt" hơn batch ít đệm dù mô hình y
    hệt — một kiểu hỏng im lặng.
    """

    def __init__(self, vocab_size: int, pad_id: int, smoothing: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(
                f"smoothing phải nằm trong [0, 1), nhận được {smoothing}. "
                "Sửa khóa toi_uu.label_smoothing trong YAML."
            )
        if vocab_size <= 1:
            raise ValueError(f"vocab_size phải lớn hơn 1, nhận được {vocab_size}")

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, nhan: torch.Tensor) -> torch.Tensor:
        """logits: (batch, len, vocab) hoặc (N, vocab). nhan: (batch, len) hoặc (N,)."""
        if logits.size(-1) != self.vocab_size:
            raise ValueError(
                f"Chiều cuối của logits là {logits.size(-1)}, không khớp vocab_size "
                f"{self.vocab_size}."
            )

        logits = logits.reshape(-1, self.vocab_size)
        nhan = nhan.reshape(-1)

        # Ép về float32 trước log_softmax. Ở fp16, log_softmax trên bảng từ vựng
        # 32 nghìn lớp rất dễ mất chính xác rồi ra NaN — cùng họ với cái bẫy fp16
        # số 3 của RMSNorm. autocast thường tự làm việc này, nhưng viết hẳn ra đây
        # thì kể cả gọi ngoài autocast cũng an toàn.
        log_xac_suat = F.log_softmax(logits.float(), dim=-1)

        vi_tri_that = nhan != self.pad_id
        so_token_that = int(vi_tri_that.sum())
        if so_token_that == 0:
            # Cả batch toàn đệm. Trả về 0 CÓ NỐI VÀO ĐỒ THỊ để backward không nổ.
            return log_xac_suat.sum() * 0.0

        # gather cần chỉ số hợp lệ ở mọi hàng, kể cả hàng sắp bị loại, nên tạm
        # thay pad_id bằng 0 rồi lọc lại bằng mặt nạ.
        nhan_an_toan = nhan.clone()
        nhan_an_toan[~vi_tri_that] = 0

        nll = -log_xac_suat.gather(dim=-1, index=nhan_an_toan.unsqueeze(-1)).squeeze(-1)
        deu = -log_xac_suat.mean(dim=-1)

        loss_moi_token = (1.0 - self.smoothing) * nll + self.smoothing * deu
        return loss_moi_token[vi_tri_that].mean()

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, pad_id={self.pad_id}, "
            f"smoothing={self.smoothing}"
        )


def tao_loss(cfg, vocab_size: int | None = None, pad_id: int = 0) -> LabelSmoothingLoss:
    """Factory đọc `toi_uu.label_smoothing` từ YAML.

    Có hàm này thì ablation A3 chỉ cần đổi một dòng cấu hình, đúng như lời hứa
    trong README, và trainer không phải tự đi đọc khóa cấu hình.
    """
    return LabelSmoothingLoss(
        vocab_size=vocab_size if vocab_size is not None else cfg.du_lieu.vocab_size,
        pad_id=pad_id,
        smoothing=cfg.toi_uu.label_smoothing,
    )
