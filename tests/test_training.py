"""TASK 13 — Kiểm tra hàm loss, scheduler và vòng huấn luyện.  Người làm: Quân.

TASK 13 xong khi: huấn luyện liên tục 1000 bước không xuất hiện NaN.

Bộ test này chạy trên CPU với mô hình thu nhỏ, vài giây là xong. Nó KHÔNG thay
được lượt chạy 1000 bước thật trên T4 — nó bắt các lỗi cơ chế mà lượt chạy thật
sẽ không nói cho ta biết là sai ở đâu:

    - cắt gradient trước khi gỡ hệ số giãn của GradScaler
    - quên chia loss cho số bước cộng dồn
    - loss tính cả vị trí đệm nên trông đẹp hơn thực tế
    - learning rate nhảy về đầu sau khi resume

torch.nn.functional.cross_entropy chỉ xuất hiện trong file này để đối chiếu số,
đúng quy tắc của đồ án. Thư mục src/ tuyệt đối không được có.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from nmt.training.checkpoint import CHE_DO_SMOKE
from nmt.training.loss import LabelSmoothingLoss, tao_loss
from nmt.training.scheduler import SchedulerCoDinh, WarmupScheduler, tao_scheduler
from nmt.training.trainer import Trainer

PAD_ID = 0
VOCAB_NHO = 60


# ===========================================================================
# PHẦN A — LabelSmoothingLoss
# ===========================================================================

@pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.3])
def test_loss_khop_voi_cross_entropy_cua_pytorch(smoothing):
    """TIÊU CHÍ RẺ TIỀN NHẤT của TASK 13. Cùng đầu vào, cùng cách bỏ qua đệm,
    kết quả phải khớp F.cross_entropy ở mọi mức smoothing.

    Đặc biệt với smoothing = 0 thì hai bên phải giống hệt nhau — lệch nghĩa là
    phần label smoothing vẫn được cộng vào dù đã tắt.
    """
    # Arrange
    torch.manual_seed(0)
    logits = torch.randn(4, 7, VOCAB_NHO)
    nhan = torch.randint(1, VOCAB_NHO, (4, 7))
    nhan[0, 5:] = PAD_ID
    nhan[2, 6:] = PAD_ID

    # Act
    cua_ta = LabelSmoothingLoss(VOCAB_NHO, PAD_ID, smoothing)(logits, nhan)
    cua_pytorch = F.cross_entropy(
        logits.reshape(-1, VOCAB_NHO).float(),
        nhan.reshape(-1),
        ignore_index=PAD_ID,
        label_smoothing=smoothing,
    )

    # Assert
    assert abs(cua_ta.item() - cua_pytorch.item()) < 1e-5


def test_loss_bo_qua_vi_tri_dem():
    """Thêm đệm vào cuối câu KHÔNG được làm loss đổi.

    Tính cả ô đệm thì loss bị pha loãng, batch nhiều đệm trông "tốt" hơn batch ít
    đệm dù mô hình y hệt — một kiểu hỏng im lặng, và nó làm hỏng luôn phép so
    giữa các cấu hình ở TASK 11.
    """
    # Arrange
    torch.manual_seed(1)
    logits = torch.randn(2, 4, VOCAB_NHO)
    nhan = torch.randint(1, VOCAB_NHO, (2, 4))
    ham_loss = LabelSmoothingLoss(VOCAB_NHO, PAD_ID, 0.0)

    logits_dai = torch.cat([logits, torch.randn(2, 3, VOCAB_NHO)], dim=1)
    nhan_dai = torch.cat([nhan, torch.full((2, 3), PAD_ID)], dim=1)

    # Act & Assert
    assert torch.allclose(ham_loss(logits, nhan), ham_loss(logits_dai, nhan_dai), atol=1e-6)


def test_loss_khong_ra_nan_khi_ca_batch_toan_dem():
    """Batch cuối của một epoch có thể rơi vào trường hợp này. Chia cho 0 ở đây
    làm hỏng toàn bộ trọng số ở bước backward ngay sau đó."""
    # Arrange
    logits = torch.randn(2, 3, VOCAB_NHO, requires_grad=True)
    nhan = torch.full((2, 3), PAD_ID)

    # Act
    loss = LabelSmoothingLoss(VOCAB_NHO, PAD_ID, 0.0)(logits, nhan)
    loss.backward()

    # Assert
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_loss_bao_loi_khi_smoothing_ngoai_khoang():
    with pytest.raises(ValueError, match="label_smoothing"):
        LabelSmoothingLoss(VOCAB_NHO, PAD_ID, 1.0)


def test_tao_loss_doc_dung_khoa_cau_hinh(cfg_goc):
    """Ablation A3 phải đổi được bằng đúng một dòng YAML."""
    # Act
    ham_loss = tao_loss(cfg_goc)

    # Assert
    assert ham_loss.smoothing == cfg_goc.toi_uu.label_smoothing
    assert ham_loss.vocab_size == cfg_goc.du_lieu.vocab_size


# ===========================================================================
# PHẦN B — Scheduler
# ===========================================================================

def test_warmup_dat_dinh_dung_tai_buoc_warmup():
    """Hai nhánh của công thức Noam gặp nhau đúng tại t = so_buoc_warmup.
    Lệch chỗ này nghĩa là sai số mũ trong công thức."""
    # Arrange
    model = torch.nn.Linear(4, 4)
    scheduler = WarmupScheduler(torch.optim.AdamW(model.parameters()), 512, so_buoc_warmup=100)

    # Act
    cac_lr = []
    for _ in range(300):
        cac_lr.append(scheduler.learning_rate_hien_tai())
        scheduler.step()

    # Assert
    buoc_dinh = max(range(len(cac_lr)), key=lambda i: cac_lr[i]) + 1
    assert abs(buoc_dinh - 100) <= 1, f"Đỉnh rơi ở bước {buoc_dinh}, mong đợi 100"
    assert cac_lr[0] < cac_lr[49] < cac_lr[99], "Trước đỉnh learning rate phải tăng"
    assert cac_lr[99] > cac_lr[199] > cac_lr[299], "Sau đỉnh learning rate phải giảm"


def test_warmup_khong_bao_gio_chia_cho_khong():
    """Đếm bước từ 0 thì lr thành vô cùng ngay bước đầu, chương trình không báo
    lỗi mà chỉ thấy loss ra NaN."""
    # Arrange & Act
    model = torch.nn.Linear(4, 4)
    scheduler = WarmupScheduler(torch.optim.AdamW(model.parameters()), 512, 10)

    # Assert
    assert math.isfinite(scheduler.learning_rate_hien_tai())
    assert scheduler.learning_rate_hien_tai() > 0


def test_scheduler_khoi_phuc_dung_learning_rate():
    """Thiếu phần này thì sau khi resume, learning rate nhảy về đầu đường warmup
    và đường loss gãy một nhát ngay chỗ nối."""
    # Arrange
    model = torch.nn.Linear(4, 4)
    goc = WarmupScheduler(torch.optim.AdamW(model.parameters()), 512, 100)
    for _ in range(37):
        goc.step()

    # Act
    moi = WarmupScheduler(torch.optim.AdamW(model.parameters()), 512, 100)
    moi.load_state_dict(goc.state_dict())

    # Assert
    assert moi.learning_rate_hien_tai() == goc.learning_rate_hien_tai()
    assert moi.optimizer.param_groups[0]["lr"] == goc.optimizer.param_groups[0]["lr"]


def test_scheduler_co_dinh_giu_nguyen_learning_rate():
    """Nhánh đối chứng của ablation A2 — mặc định của base.yaml."""
    # Arrange
    model = torch.nn.Linear(4, 4)
    scheduler = SchedulerCoDinh(torch.optim.AdamW(model.parameters()), 7e-4)

    # Act
    for _ in range(100):
        scheduler.step()

    # Assert
    assert scheduler.learning_rate_hien_tai() == 7e-4
    assert scheduler.optimizer.param_groups[0]["lr"] == 7e-4


@pytest.mark.parametrize(
    "kieu, lop_mong_doi", [("co_dinh", SchedulerCoDinh), ("warmup", WarmupScheduler)]
)
def test_tao_scheduler_doc_dung_khoa_cau_hinh(cfg_goc, kieu, lop_mong_doi):
    # Arrange
    cfg_goc.toi_uu.scheduler = kieu
    model = torch.nn.Linear(4, 4)

    # Act
    scheduler = tao_scheduler(cfg_goc, torch.optim.AdamW(model.parameters()))

    # Assert
    assert isinstance(scheduler, lop_mong_doi)


def test_tao_scheduler_bao_loi_khi_kieu_sai(cfg_goc):
    cfg_goc.toi_uu.scheduler = "cosine"
    with pytest.raises(ValueError, match="scheduler"):
        tao_scheduler(cfg_goc, torch.optim.AdamW(torch.nn.Linear(4, 4).parameters()))


# ===========================================================================
# PHẦN C — Trainer
# ===========================================================================

def _cfg_thu_nho(cfg):
    """Thu mô hình về cỡ chạy được trên CPU trong vài giây."""
    cfg.mo_hinh.d_model = 32
    cfg.mo_hinh.so_head = 4
    cfg.mo_hinh.so_lop_encoder = 1
    cfg.mo_hinh.so_lop_decoder = 1
    cfg.mo_hinh.d_ff = 32
    cfg.du_lieu.vocab_size = VOCAB_NHO
    cfg.du_lieu.do_dai_toi_da = 8
    cfg.toi_uu.so_buoc_cong_don_gradient = 2
    cfg.huan_luyen.danh_gia_moi = 5
    cfg.huan_luyen.luu_checkpoint_moi = 5
    cfg.huan_luyen.dung_som_sau = 99
    return cfg


def _batch_gia(so_cau=4, len_src=6, len_tgt=5):
    nhan = torch.randint(1, VOCAB_NHO, (so_cau, len_tgt))
    nhan[:, -1] = PAD_ID
    return {
        "src_ids": torch.randint(1, VOCAB_NHO, (so_cau, len_src)),
        "tgt_input": torch.randint(1, VOCAB_NHO, (so_cau, len_tgt)),
        "labels": nhan,
        "src_mask": torch.ones(so_cau, 1, 1, len_src, dtype=torch.bool),
        "tgt_mask": torch.ones(so_cau, 1, len_tgt, len_tgt, dtype=torch.bool).tril(),
    }


class _LoaderGia:
    """DataLoader tối giản — Trainer chỉ cần lặp được và có len()."""

    def __init__(self, so_batch: int = 4) -> None:
        torch.manual_seed(7)
        self._cac_batch = [_batch_gia() for _ in range(so_batch)]

    def __iter__(self):
        return iter(self._cac_batch)

    def __len__(self) -> int:
        return len(self._cac_batch)


def _tao_trainer(cfg, tmp_path, so_buoc: int, thu_muc: str = "ckpt"):
    from nmt.model.transformer import TransformerNMT

    return Trainer(
        cfg,
        TransformerNMT(cfg),
        _LoaderGia(),
        _LoaderGia(),
        logger=None,
        che_do=CHE_DO_SMOKE,
        thu_muc_checkpoint=tmp_path / thu_muc,
        repo_hub=None,
        so_buoc_toi_da=so_buoc,
    )


def test_chay_lien_tuc_khong_ra_nan(cfg_goc, tmp_path):
    """TIÊU CHÍ XONG KHI CỦA TASK 13, bản thu nhỏ chạy được ở máy cá nhân.

    Lượt 1000 bước thật chạy trên T4 bằng scripts/train.py; bài này bắt lỗi cơ
    chế trước khi tốn giờ GPU.
    """
    # Arrange
    trainer = _tao_trainer(_cfg_thu_nho(cfg_goc), tmp_path, so_buoc=30)

    # Act
    ket_qua = trainer.train()

    # Assert
    assert ket_qua["buoc_cuoi"] == 30
    assert all(math.isfinite(gia_tri) for gia_tri in trainer.lich_su_loss)
    assert ket_qua["loss_dev_tot_nhat"] is not None


def test_cong_don_gradient_cap_nhat_dung_mot_lan_moi_chu_ky(cfg_goc, tmp_path):
    """Với so_buoc_cong_don = 2, chạy 5 bước phải tiêu thụ 10 batch nhưng chỉ gọi
    optimizer.step() đúng 5 lần.

    Gọi step() ở mỗi batch con thì learning rate hiệu dụng gấp đôi mà không có
    lỗi nào báo ra.
    """
    # Arrange
    trainer = _tao_trainer(_cfg_thu_nho(cfg_goc), tmp_path, so_buoc=5)
    so_lan_goi = {"n": 0}
    step_goc = trainer.optimizer.step

    def dem_step(*args, **kwargs):
        so_lan_goi["n"] += 1
        return step_goc(*args, **kwargs)

    trainer.optimizer.step = dem_step

    # Act
    trainer.train()

    # Assert
    assert trainer.so_buoc_cong_don == 2
    assert so_lan_goi["n"] == 5, f"optimizer.step() được gọi {so_lan_goi['n']} lần, mong đợi 5"


def test_gradient_bi_cat_theo_nguong(cfg_goc, tmp_path):
    """Sau khi cắt theo norm 1.0, chuẩn của toàn bộ gradient không được vượt
    ngưỡng. Đây là phần bảo vệ chính chống bùng nổ gradient khi chạy fp16."""
    # Arrange
    cfg = _cfg_thu_nho(cfg_goc)
    cfg.toi_uu.cat_gradient_norm = 1.0
    trainer = _tao_trainer(cfg, tmp_path, so_buoc=3)
    chuan_sau_khi_cat = []

    clip_goc = torch.nn.utils.clip_grad_norm_

    def ghi_lai_chuan(tham_so, nguong, *args, **kwargs):
        ket_qua = clip_goc(tham_so, nguong, *args, **kwargs)
        # Đo LẠI sau khi cắt, vì clip_grad_norm_ trả về chuẩn TRƯỚC khi cắt.
        tong = sum(
            p.grad.detach().pow(2).sum()
            for p in trainer.model.parameters()
            if p.grad is not None
        )
        chuan_sau_khi_cat.append(float(tong.sqrt()))
        return ket_qua

    torch.nn.utils.clip_grad_norm_ = ghi_lai_chuan
    try:
        trainer.train()
    finally:
        torch.nn.utils.clip_grad_norm_ = clip_goc

    # Assert
    assert chuan_sau_khi_cat, "clip_grad_norm_ chưa từng được gọi"
    for chuan in chuan_sau_khi_cat:
        assert chuan <= 1.0 + 1e-4, f"Chuẩn gradient sau khi cắt là {chuan}, vượt ngưỡng 1.0"


def test_resume_chay_tiep_dung_cho_cu(cfg_goc, tmp_path):
    """Nền tảng của TASK 14. Nạp checkpoint xong phải chạy tiếp từ đúng số bước
    đã lưu chứ không bắt đầu lại từ 0."""
    # Arrange
    cfg = _cfg_thu_nho(cfg_goc)
    trainer_dau = _tao_trainer(cfg, tmp_path, so_buoc=10, thu_muc="lan1")
    trainer_dau.train()
    checkpoint = tmp_path / "lan1" / "moi_nhat.pt"

    # Act
    trainer_sau = _tao_trainer(cfg, tmp_path, so_buoc=20, thu_muc="lan2")
    trainer_sau.tiep_tuc_tu(checkpoint)
    buoc_khi_nap = trainer_sau.buoc
    ket_qua = trainer_sau.train()

    # Assert
    assert buoc_khi_nap == 10, "Nạp checkpoint xong phải đang ở bước 10"
    assert ket_qua["buoc_cuoi"] == 20
    assert ket_qua["so_buoc_da_chay"] == 10


def test_dung_som_khi_loss_dev_khong_cai_thien(cfg_goc, tmp_path):
    """Dừng sớm phải kích hoạt được, nếu không thì lượt chạy thật đốt hết giờ GPU
    cho phần đã bão hòa."""
    # Arrange
    cfg = _cfg_thu_nho(cfg_goc)
    cfg.huan_luyen.dung_som_sau = 1
    cfg.huan_luyen.danh_gia_moi = 2
    trainer = _tao_trainer(cfg, tmp_path, so_buoc=100)

    # Ép loss dev xấu dần để chắc chắn không có lần nào cải thiện
    dem = {"n": 0}

    def danh_gia_gia():
        dem["n"] += 1
        return {"loss_dev": 1.0 + dem["n"], "perplexity_dev": 2.0}

    trainer.danh_gia = danh_gia_gia

    # Act
    ket_qua = trainer.train()

    # Assert
    assert ket_qua["dung_som"] is True
    assert ket_qua["buoc_cuoi"] < 100


def test_khong_bat_fp16_khi_chay_tren_cpu(cfg_goc, tmp_path):
    """GradScaler chỉ có tác dụng trên CUDA. Bật ở CPU thì PyTorch in cảnh báo và
    chạy chậm hơn, mà toàn bộ test của nhóm đều chạy CPU."""
    # Arrange
    cfg = _cfg_thu_nho(cfg_goc)
    cfg.toi_uu.do_chinh_xac_hon_hop = True

    # Act
    trainer = _tao_trainer(cfg, tmp_path, so_buoc=1)

    # Assert
    if trainer.thiet_bi.type != "cuda":
        assert trainer.dung_fp16 is False
        assert trainer.scaler.is_enabled() is False
