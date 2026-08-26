"""TASK 04 — DataLoader gom batch theo số token.  Người làm: My.  [Data Pipeline • Bắt buộc]

Xong khi: batch mẫu đúng kích thước, tỉ lệ ô đệm thừa dưới 25 phần trăm.

Gom batch theo TỔNG SỐ TOKEN chứ không theo số câu. Gom theo số câu thì batch
toàn câu ngắn sẽ lãng phí bộ nhớ, còn batch toàn câu dài thì tràn bộ nhớ.
Xếp các câu có độ dài gần nhau vào cùng một batch để đỡ phí padding.

Nhớ truyền seed_cho_worker và sinh_generator từ nmt.utils.seed vào DataLoader,
nếu không thì thứ tự trộn dữ liệu đổi theo số worker và mất tính tái lập.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, BatchSampler, Sampler

if TYPE_CHECKING:
    from nmt.data.dataset import DuLieuSongNgu

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3


# ---------------------------------------------------------------------------
# Sampler gom batch theo số token (token-bucket batching)
# ---------------------------------------------------------------------------

class GomBatchTheoTokenSampler(Sampler):
    """BatchSampler gom các câu CÓ ĐỘ DÀI GẦN NHAU vào cùng batch.

    Thuật toán:
        1. Sắp xếp toàn bộ chỉ số theo độ dài câu src (gom_theo_do_dai=True)
           hoặc giữ nguyên thứ tự ngẫu nhiên (gom_theo_do_dai=False).
        2. Duyệt tuần tự, cộng dồn số token (max_len_trong_batch × so_cau),
           khi vượt so_token_moi_batch thì đóng batch và mở batch mới.
        3. Cuối epoch xáo trộn thứ tự CÁC BATCH để mô hình không nhìn thấy
           cùng một phân phối độ dài câu mỗi epoch.

    Lý do dùng max_len × so_cau thay vì tổng token thực tế:
        Với padding, thực tế bộ nhớ chiếm bởi một batch là max_len × so_cau.
        Dùng tổng token thực tế thì batch cuối cùng của các câu dài vẫn có thể
        tràn VRAM nếu có một câu rất dài kéo max_len lên cao.
    """

    def __init__(
        self,
        dataset: "DuLieuSongNgu",
        so_token_moi_batch: int = 4096,
        gom_theo_do_dai: bool = True,
        tron: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        self._dataset = dataset
        self._so_token = so_token_moi_batch
        self._gom = gom_theo_do_dai
        self._tron = tron
        self._generator = generator
        self._batches = self._xay_cac_batch()

    def _xay_cac_batch(self) -> list[list[int]]:
        n = len(self._dataset)

        if self._gom:
            # Sắp xếp theo độ dài câu src
            chi_so = sorted(range(n), key=lambda i: self._dataset.do_dai_src(i))
        else:
            # Thứ tự nguyên
            chi_so = list(range(n))

        cac_batch: list[list[int]] = []
        batch_hien_tai: list[int] = []
        max_len_hien_tai = 0

        for i in chi_so:
            do_dai = self._dataset.do_dai_src(i)
            max_len_moi = max(max_len_hien_tai, do_dai)
            so_cau_moi = len(batch_hien_tai) + 1

            if max_len_moi * so_cau_moi > self._so_token and batch_hien_tai:
                # Đóng batch hiện tại
                cac_batch.append(batch_hien_tai)
                batch_hien_tai = [i]
                max_len_hien_tai = do_dai
            else:
                batch_hien_tai.append(i)
                max_len_hien_tai = max_len_moi

        if batch_hien_tai:
            cac_batch.append(batch_hien_tai)

        # Xáo trộn THỨ TỰ CÁC BATCH (không phải câu trong batch)
        if self._tron:
            thu_tu = torch.randperm(
                len(cac_batch), generator=self._generator
            ).tolist()
            cac_batch = [cac_batch[i] for i in thu_tu]

        return cac_batch

    def __iter__(self):
        # Cuối mỗi epoch xây lại danh sách batch (có xáo trộn thứ tự nếu self._tron=True)
        # để mô hình không thấy cùng một thứ tự batch ở mọi epoch.
        yield from self._xay_cac_batch()

    def __len__(self) -> int:
        return len(self._batches)


# ---------------------------------------------------------------------------
# collate_fn: ghép các mẫu thành batch tensor + sinh mask
# ---------------------------------------------------------------------------

def ghep_batch(cac_mau: list[dict], pad_id: int = PAD_ID) -> dict[str, torch.Tensor]:
    """collate_fn: đệm cho bằng nhau, sinh padding mask và causal mask.

    Đầu vào: list các dict {"src_ids": Tensor, "tgt_ids": Tensor}

    Trả về dict:
        src_ids    : [B, S]      token IDs nguồn (EN), đã padding
        tgt_input  : [B, T]      decoder input = tgt[:-1] (BOS…, không EOS)
        labels     : [B, T]      nhãn = tgt[1:] (không BOS, …EOS)
        src_mask   : [B, 1, 1, S] True ở vị trí THẬT, False ở PAD
                                  (broadcast cho multi-head attention)
        tgt_mask   : [B, 1, T, T] kết hợp padding mask + causal mask

    Thiết kế mask dùng True=giữ, False=che — nhất quán với
    F.scaled_dot_product_attention(attn_mask=...) khi is_causal=False.
    Trong attention score: score.masked_fill(~mask, -inf)
    """
    # --- Tách src và tgt ---
    src_list = [m["src_ids"] for m in cac_mau]
    tgt_list = [m["tgt_ids"] for m in cac_mau]   # có BOS ở đầu và EOS ở cuối

    # --- Padding ---
    # pad_sequence mặc định padding_value=0
    src_padded = _pad(src_list, pad_id)   # [B, S]
    tgt_padded = _pad(tgt_list, pad_id)   # [B, T+1] (T = max tgt len)

    # Teacher forcing: input cho decoder là tgt[:-1], nhãn là tgt[1:]
    tgt_input = tgt_padded[:, :-1]        # [B, T]   — có BOS, không EOS
    labels    = tgt_padded[:, 1:]         # [B, T]   — không BOS, có EOS

    B, S = src_padded.shape
    _, T = tgt_input.shape

    # --- src_mask: [B, 1, 1, S] — True tại vị trí THẬT ---
    src_mask = (src_padded != pad_id).unsqueeze(1).unsqueeze(2)   # [B, 1, 1, S]

    # --- tgt_mask: [B, 1, T, T] — padding mask AND causal mask ---
    # padding mask: [B, 1, T, T] — True tại các vị trí không phải PAD
    pad_mask = (tgt_input != pad_id).unsqueeze(1).unsqueeze(2)    # [B, 1, 1, T]
    pad_mask = pad_mask.expand(B, 1, T, T)

    # causal mask: [T, T] — True tại tam giác dưới (bao gồm đường chéo)
    causal = torch.ones(T, T, dtype=torch.bool).tril()             # [T, T]

    # Kết hợp: vị trí được nhìn = không phải PAD VÀ không phải tương lai
    tgt_mask = pad_mask & causal.unsqueeze(0).unsqueeze(0)         # [B, 1, T, T]

    return {
        "src_ids":   src_padded,   # [B, S]
        "tgt_input": tgt_input,    # [B, T]
        "labels":    labels,       # [B, T]
        "src_mask":  src_mask,     # [B, 1, 1, S]
        "tgt_mask":  tgt_mask,     # [B, 1, T, T]
    }


def _pad(tensors: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    """Padding một list Tensor 1-D thành ma trận [B, L_max]."""
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(tensors):
        out[i, : t.size(0)] = t
    return out


# ---------------------------------------------------------------------------
# Hàm tiện ích đo tỉ lệ padding
# ---------------------------------------------------------------------------

def ti_le_o_dem(batch: dict[str, torch.Tensor], pad_id: int = PAD_ID) -> float:
    """Tỉ lệ phần trăm ô đệm thừa trong batch. Yêu cầu dưới 25%.

    Tính trên cả src và tgt_input để phản ánh toàn bộ bộ nhớ chiếm dụng.
    """
    tong_o = 0
    tong_dem = 0
    for ten in ("src_ids", "tgt_input"):
        if ten in batch:
            t = batch[ten]
            tong_o += t.numel()
            tong_dem += (t == pad_id).sum().item()
    return tong_dem / tong_o if tong_o > 0 else 0.0


# ---------------------------------------------------------------------------
# Hàm tạo DataLoader hoàn chỉnh
# ---------------------------------------------------------------------------

def tao_dataloader(
    dataset: "DuLieuSongNgu",
    so_token_moi_batch: int = 4096,
    gom_theo_do_dai: bool = True,
    so_worker: int = 2,
    tron: bool = True,
    generator: torch.Generator | None = None,
    pad_id: int = PAD_ID,
) -> DataLoader:
    """Tạo DataLoader với token-bucket batching.

    Args:
        dataset: instance của DuLieuSongNgu
        so_token_moi_batch: tổng số token tối đa mỗi batch (đọc từ config)
        gom_theo_do_dai: sắp xếp câu theo độ dài trước khi gom batch
        so_worker: số worker cho DataLoader (đọc từ config)
        tron: xáo trộn thứ tự batch mỗi epoch
        generator: torch.Generator để tái lập (từ nmt.utils.sinh_generator)
        pad_id: ID của token padding

    Example::
        from nmt.utils import sinh_generator
        gen = sinh_generator(seed=42)
        loader = tao_dataloader(dataset, generator=gen)
        for batch in loader:
            ...
    """
    from nmt.utils import seed_cho_worker

    sampler = GomBatchTheoTokenSampler(
        dataset=dataset,
        so_token_moi_batch=so_token_moi_batch,
        gom_theo_do_dai=gom_theo_do_dai,
        tron=tron,
        generator=generator,
    )

    from functools import partial
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=so_worker,
        collate_fn=partial(ghep_batch, pad_id=pad_id),
        worker_init_fn=seed_cho_worker,
        # Chỉ ghim bộ nhớ khi thật sự có GPU để chép sang. Bật vô điều kiện thì
        # máy chạy CPU in cảnh báo ở mỗi lần tạo DataLoader, và nhiễu như vậy
        # lâu dần khiến cả nhóm bỏ qua luôn những cảnh báo có thật.
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(so_worker > 0),
    )


# ---------------------------------------------------------------------------
# Hàm gom batch đơn giản (không dùng sampler — dành cho dev/test)
# ---------------------------------------------------------------------------

def gom_batch_theo_token(
    dataset: "DuLieuSongNgu",
    so_token_moi_batch: int = 4096,
) -> list[list[int]]:
    """Trả về danh sách các batch, mỗi batch là danh sách chỉ số câu.

    Hàm này chỉ TÍNH TOÁN các batch, không tạo DataLoader.
    Dùng khi cần biết trước số lượng batch hoặc debug phân phối độ dài.
    """
    n = len(dataset)
    chi_so = sorted(range(n), key=lambda i: dataset.do_dai_src(i))

    cac_batch: list[list[int]] = []
    batch_hien_tai: list[int] = []
    max_len_hien_tai = 0

    for i in chi_so:
        do_dai = dataset.do_dai_src(i)
        max_len_moi = max(max_len_hien_tai, do_dai)
        so_cau_moi = len(batch_hien_tai) + 1

        if max_len_moi * so_cau_moi > so_token_moi_batch and batch_hien_tai:
            cac_batch.append(batch_hien_tai)
            batch_hien_tai = [i]
            max_len_hien_tai = do_dai
        else:
            batch_hien_tai.append(i)
            max_len_hien_tai = max_len_moi

    if batch_hien_tai:
        cac_batch.append(batch_hien_tai)

    return cac_batch
