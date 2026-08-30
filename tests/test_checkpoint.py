"""TASK 12 + TASK 14 — Kiểm tra checkpoint và phục hồi.  Người làm: Quân.

TASK 14 xong khi: hai đường loss chênh nhau dưới 1 phần trăm tại cùng một bước.

Hình hai đường loss chồng lên nhau của thí nghiệm giết phiên là HÌNH CÓ GIÁ TRỊ
NHẤT của cả đồ án — dùng cho báo cáo, slide và cả CV — vì nó chứng minh cơ chế
phục hồi hoạt động thật chứ không chỉ nói suông.

Bộ test này chạy hoàn toàn trên CPU với mô hình thu nhỏ nên máy nào cũng xong
trong vài giây. Lượt chạy thật trên T4 do `scripts/thi_nghiem_phuc_hoi.py` lo.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nmt.training.checkpoint import (
    CHE_DO_SMOKE,
    CHE_DO_THAT,
    doc_thong_tin,
    luu_checkpoint,
    nap_checkpoint,
)
from nmt.training.scheduler import WarmupScheduler


def _bo_ba_de_test(lr: float = 1e-3):
    """Một mô hình bé xíu kèm optimizer, scheduler, scaler — đủ để kiểm cơ chế."""
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, d_model=64, so_buoc_warmup=10)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scheduler, scaler


def _chay_vai_buoc(model, optimizer, scheduler, so_buoc: int = 5) -> None:
    """Chạy vài bước thật để optimizer có momentum và scheduler nhích khỏi bước 1.

    Lưu checkpoint của mô hình vừa khởi tạo thì mọi trạng thái đều rỗng, và bài
    test sẽ báo ĐẠT ngay cả khi phần lưu trạng thái optimizer bị bỏ quên.
    """
    for _ in range(so_buoc):
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(4, 8)).pow(2).mean().backward()
        optimizer.step()
        scheduler.step()


def test_luu_roi_nap_lai_ra_dung_trong_so(tmp_path, cfg_goc):
    """Lưu rồi nạp lại, mọi tensor trọng số phải giống hệt từng phần tử."""
    # Arrange
    model, optimizer, scheduler, scaler = _bo_ba_de_test()
    _chay_vai_buoc(model, optimizer, scheduler)
    duong_dan = tmp_path / "ck.pt"

    # Act
    luu_checkpoint(duong_dan, model, optimizer, scheduler, scaler,
                   buoc=5, epoch=1, cfg=cfg_goc, che_do=CHE_DO_SMOKE)

    model_moi, opt_moi, sch_moi, scaler_moi = _bo_ba_de_test()
    thong_tin = nap_checkpoint(duong_dan, model_moi, opt_moi, sch_moi, scaler_moi,
                               map_location="cpu", che_do_mong_doi=CHE_DO_SMOKE)

    # Assert
    for (ten, goc), (_, nap_lai) in zip(
        model.state_dict().items(), model_moi.state_dict().items()
    ):
        assert torch.equal(goc, nap_lai), f"Tensor {ten} khác nhau sau khi nạp lại"
    assert thong_tin["buoc"] == 5
    assert thong_tin["epoch"] == 1


def test_nap_lai_du_trang_thai_optimizer_va_scheduler(tmp_path, cfg_goc):
    """Thiếu trạng thái optimizer thì momentum của AdamW về 0 và loss giật lên
    ngay sau khi resume — nhìn đường loss là thấy một vết gãy.
    Thiếu trạng thái scheduler thì learning rate nhảy về đầu."""
    # Arrange
    model, optimizer, scheduler, scaler = _bo_ba_de_test()
    _chay_vai_buoc(model, optimizer, scheduler, so_buoc=7)
    lr_truoc_khi_luu = scheduler.learning_rate_hien_tai()
    duong_dan = tmp_path / "ck.pt"
    luu_checkpoint(duong_dan, model, optimizer, scheduler, scaler,
                   buoc=7, epoch=0, cfg=cfg_goc, che_do=CHE_DO_SMOKE)

    # Act
    model_moi, opt_moi, sch_moi, scaler_moi = _bo_ba_de_test()
    lr_khi_moi_tao = sch_moi.learning_rate_hien_tai()
    nap_checkpoint(duong_dan, model_moi, opt_moi, sch_moi, scaler_moi,
                   map_location="cpu", che_do_mong_doi=CHE_DO_SMOKE)

    # Assert — scheduler quay lại đúng chỗ, không nhảy về đầu đường warmup
    assert lr_khi_moi_tao != lr_truoc_khi_luu, "Bài test tự nó hỏng: hai lr phải khác nhau"
    assert sch_moi.learning_rate_hien_tai() == lr_truoc_khi_luu

    # Assert — momentum của AdamW còn nguyên, không bị về 0
    trang_thai = opt_moi.state_dict()["state"]
    assert trang_thai, "Trạng thái optimizer rỗng, momentum AdamW đã bị mất"
    for gia_tri in trang_thai.values():
        assert gia_tri["exp_avg"].abs().sum() > 0


def test_ghi_an_toan_khong_lam_hong_checkpoint_cu(tmp_path, cfg_goc, monkeypatch):
    """Mô phỏng bị ngắt giữa lúc đang ghi. Checkpoint CŨ phải còn nguyên vẹn và
    nạp lại được. Đây là lý do phải ghi ra file tạm rồi mới đổi tên."""
    # Arrange — đã có sẵn một checkpoint tốt ở bước 5
    model, optimizer, scheduler, scaler = _bo_ba_de_test()
    _chay_vai_buoc(model, optimizer, scheduler)
    duong_dan = tmp_path / "ck.pt"
    luu_checkpoint(duong_dan, model, optimizer, scheduler, scaler,
                   buoc=5, epoch=0, cfg=cfg_goc, che_do=CHE_DO_SMOKE)
    kich_thuoc_cu = duong_dan.stat().st_size

    # Act — lần ghi thứ hai chết đúng lúc torch.save đang chạy
    def torch_save_chet(*_args, **_kwargs):
        raise KeyboardInterrupt("phiên Kaggle bị ngắt giữa lúc đang ghi")

    monkeypatch.setattr(torch, "save", torch_save_chet)
    with pytest.raises(KeyboardInterrupt):
        luu_checkpoint(duong_dan, model, optimizer, scheduler, scaler,
                       buoc=99, epoch=9, cfg=cfg_goc, che_do=CHE_DO_SMOKE)
    monkeypatch.undo()

    # Assert — bản cũ còn nguyên, vẫn nạp được, vẫn là bước 5
    assert duong_dan.exists()
    assert duong_dan.stat().st_size == kich_thuoc_cu
    assert doc_thong_tin(duong_dan)["buoc"] == 5

    model_moi, *_ = _bo_ba_de_test()
    assert nap_checkpoint(duong_dan, model_moi, map_location="cpu")["buoc"] == 5


def test_chan_checkpoint_smoke_lan_vao_luot_chay_that(tmp_path, cfg_goc):
    """Bài học mục 1.8 của `Sưu tập lỗi.md`: một lượt smoke test từng đè lên
    checkpoint thật, hôm sau cả pipeline chạy bằng mô hình đồ chơi mà không có
    cảnh báo nào. Nạp nhầm chế độ phải NÉM LỖI chứ không được đi tiếp."""
    # Arrange
    model, optimizer, scheduler, scaler = _bo_ba_de_test()
    duong_dan = tmp_path / "ck_smoke.pt"
    luu_checkpoint(duong_dan, model, optimizer, scheduler, scaler,
                   buoc=3, epoch=0, cfg=cfg_goc, che_do=CHE_DO_SMOKE)

    # Act & Assert
    model_moi, *_ = _bo_ba_de_test()
    with pytest.raises(RuntimeError, match="smoke"):
        nap_checkpoint(duong_dan, model_moi, map_location="cpu",
                       che_do_mong_doi=CHE_DO_THAT)

    # Đúng chế độ thì vẫn nạp bình thường
    assert nap_checkpoint(duong_dan, model_moi, map_location="cpu",
                          che_do_mong_doi=CHE_DO_SMOKE)["buoc"] == 3


def test_luu_du_cac_mon_bat_buoc(tmp_path, cfg_goc):
    """Thiếu một món là resume sai mà không báo lỗi, nên kiểm bằng danh sách chứ
    đừng tin là mình đã lưu đủ."""
    # Arrange
    model, optimizer, scheduler, scaler = _bo_ba_de_test()
    duong_dan = tmp_path / "ck.pt"

    # Act
    luu_checkpoint(duong_dan, model, optimizer, scheduler, scaler,
                   buoc=1, epoch=0, cfg=cfg_goc, che_do=CHE_DO_SMOKE)
    goi = torch.load(duong_dan, map_location="cpu", weights_only=False)

    # Assert
    for khoa in ("model", "optimizer", "scheduler", "scaler", "buoc", "epoch", "cfg", "rng"):
        assert khoa in goi, f"Checkpoint thiếu {khoa}"
    for nguon in ("python", "numpy", "torch"):
        assert nguon in goi["rng"], f"Thiếu trạng thái RNG của {nguon}"
    # cfg phải là dict THUẦN để máy chưa cài package nmt vẫn nạp lại được
    assert type(goi["cfg"]) is dict
    assert goi["cfg"]["mo_hinh"]["d_model"] == cfg_goc.mo_hinh.d_model


def test_khoi_phuc_rng_cho_ra_cung_day_so_ngau_nhien(tmp_path, cfg_goc):
    """Thiếu trạng thái RNG thì sau khi resume, dropout và thứ tự trộn dữ liệu
    khác lượt chạy liền mạch, nên hai đường loss của TASK 14 không trùng khít."""
    # Arrange
    model, optimizer, scheduler, scaler = _bo_ba_de_test()
    duong_dan = tmp_path / "ck.pt"
    torch.manual_seed(1234)
    luu_checkpoint(duong_dan, model, optimizer, scheduler, scaler,
                   buoc=1, epoch=0, cfg=cfg_goc, che_do=CHE_DO_SMOKE)
    mong_doi = torch.randn(5)

    # Act — làm nhiễu RNG rồi nạp lại
    torch.manual_seed(999)
    torch.randn(100)
    model_moi, *_ = _bo_ba_de_test()
    nap_checkpoint(duong_dan, model_moi, map_location="cpu", khoi_phuc_rng=True)

    # Assert
    assert torch.allclose(torch.randn(5), mong_doi), (
        "Sau khi khôi phục RNG, dãy số ngẫu nhiên phải lặp lại y hệt"
    )
