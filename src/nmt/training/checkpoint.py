"""TASK 12 — Cơ chế lưu checkpoint.  Người làm: Bảo.  [Training Infra • Bắt buộc]

Xong khi: thời gian đồng bộ checkpoint chiếm dưới 5 phần trăm tổng thời gian huấn luyện.

Lưu ĐẦY ĐỦ, thiếu một món là resume sai mà không báo lỗi:
    - trọng số mô hình
    - trạng thái optimizer   (thiếu cái này thì momentum của AdamW về 0, loss giật lên)
    - trạng thái scheduler
    - trạng thái GradScaler  (fp16)
    - số bước đã chạy, số epoch
    - bản sao cấu hình đã gộp
    - seed và trạng thái RNG (để resume ra đúng cùng một đường loss)

GHI FILE THEO KIỂU AN TOÀN: ghi ra file tạm rồi mới đổi tên (atomic rename).
Phiên Kaggle bị ngắt đúng lúc đang ghi mà ghi đè trực tiếp thì mất luôn cả
checkpoint cũ lẫn mới.

VÂN TAY CHẾ ĐỘ CHẠY — rút từ mục 1.8 và 1.9 của `Sưu tập lỗi.md`:

    Ở đồ án trước, một lượt smoke test đã ĐÈ LÊN checkpoint thật trên Hugging Face
    vì hai bên dùng chung đường dẫn. Hôm sau notebook kéo về đúng mô hình đồ chơi
    huấn luyện trên 1.296 mẫu thay vì 26.148 mẫu, mà KHÔNG có cảnh báo nào.

    Nên mỗi checkpoint ở đây mang theo trường `che_do` ("that" hoặc "smoke"), và
    `nap_checkpoint` sẽ NÉM LỖI khi lượt chạy thật vô tình nạp phải checkpoint
    smoke. Chặn ồn ào còn hơn hỏng im lặng.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nmt.utils.config import ve_dict_thuan

# Hai chế độ chạy. Smoke test chỉ để kiểm notebook không crash, tuyệt đối không
# được lẫn với checkpoint thật.
CHE_DO_THAT = "that"
CHE_DO_SMOKE = "smoke"

# Phiên bản định dạng checkpoint. Đổi cấu trúc thì tăng số này lên, nhờ vậy nạp
# phải checkpoint đời cũ sẽ báo lỗi rõ ràng thay vì thiếu khóa giữa chừng.
PHIEN_BAN_DINH_DANG = 1

# Bảy món bắt buộc. Kiểm bằng danh sách này thay vì tin rằng mình đã lưu đủ.
CAC_KHOA_BAT_BUOC = (
    "phien_ban",
    "che_do",
    "buoc",
    "epoch",
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "cfg",
    "rng",
)


def _gom_trang_thai_rng() -> dict[str, Any]:
    """Gom trạng thái sinh số ngẫu nhiên của cả bốn nguồn.

    Thiếu phần này thì sau khi resume, thứ tự trộn dữ liệu và mặt nạ dropout đều
    khác lượt chạy liền mạch, nên hai đường loss của TASK 14 không thể trùng khít
    và tiêu chí "chênh nhau dưới 1 phần trăm" sẽ trượt.
    """
    trang_thai: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        trang_thai["torch_cuda"] = torch.cuda.get_rng_state_all()
    return trang_thai


def _khoi_phuc_trang_thai_rng(trang_thai: dict[str, Any] | None) -> None:
    """Đặt lại đúng trạng thái RNG đã lưu. Thiếu khóa nào thì bỏ qua khóa đó."""
    if not trang_thai:
        return

    if "python" in trang_thai:
        random.setstate(trang_thai["python"])
    if "numpy" in trang_thai:
        np.random.set_state(trang_thai["numpy"])
    if "torch" in trang_thai:
        # torch yêu cầu ByteTensor nằm trên CPU, mà torch.load có thể đã đẩy
        # tensor lên GPU nếu map_location trỏ vào cuda.
        torch.set_rng_state(trang_thai["torch"].cpu().to(torch.uint8))
    if "torch_cuda" in trang_thai and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(
                [t.cpu().to(torch.uint8) for t in trang_thai["torch_cuda"]]
            )
        except (RuntimeError, ValueError) as loi:
            # Lưu trên máy 1 GPU rồi nạp trên máy 2 GPU (Kaggle hay cấp T4x2) thì
            # số lượng trạng thái lệch nhau. Không phải lỗi chặn, nhưng phải nói
            # ra chứ không nuốt im lặng, để không ai tưởng lượt này tái lập y hệt.
            print(
                f"[checkpoint] Không khôi phục được RNG của CUDA ({loi}). "
                "Thường do số GPU lúc lưu khác lúc nạp; phần còn lại vẫn đúng."
            )


def luu_checkpoint(
    duong_dan: str | Path,
    model,
    optimizer,
    scheduler,
    scaler,
    buoc: int,
    epoch: int,
    cfg,
    *,
    loss_dev: float | None = None,
    che_do: str = CHE_DO_THAT,
    so_lieu_them: dict[str, Any] | None = None,
) -> Path:
    """Ghi checkpoint đầy đủ theo kiểu an toàn, trả về đường dẫn đã ghi.

    Args:
        duong_dan: đường dẫn file .pt đích.
        model, optimizer, scheduler, scaler: bốn đối tượng cần lưu trạng thái.
            `scheduler` và `scaler` có thể là None (chạy fp32, hoặc không dùng
            scheduler) — khi đó lưu None và lúc nạp cũng bỏ qua.
        buoc: số bước huấn luyện toàn cục ĐÃ chạy xong.
        epoch: số epoch đã chạy xong.
        cfg: cấu hình đã gộp, lưu kèm để ba tháng sau còn biết lượt này chạy gì.
        loss_dev: loss trên tập dev tại thời điểm lưu, dùng để chọn bản tốt nhất.
        che_do: CHE_DO_THAT hoặc CHE_DO_SMOKE. Xem phần vân tay ở đầu file.
        so_lieu_them: số liệu phụ muốn ghi kèm (bleu, thời gian chạy...).

    Ghi ra `<duong_dan>.tmp` rồi mới `os.replace` sang tên thật. `os.replace` là
    thao tác nguyên tử trên cùng một phân vùng, nên phiên Kaggle chết giữa chừng
    thì cùng lắm mất file tạm, còn checkpoint cũ vẫn nguyên vẹn để chạy tiếp.
    """
    if che_do not in (CHE_DO_THAT, CHE_DO_SMOKE):
        raise ValueError(
            f"che_do không hợp lệ: {che_do!r}. Phải là {CHE_DO_THAT!r} hoặc {CHE_DO_SMOKE!r}."
        )

    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)

    noi_dung: dict[str, Any] = {
        "phien_ban": PHIEN_BAN_DINH_DANG,
        "che_do": che_do,
        "buoc": buoc,
        "epoch": epoch,
        "loss_dev": loss_dev,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        # ve_dict_thuan vì Config là lớp con của dict. torch.save vẫn lưu được,
        # nhưng đổi về dict thuần thì máy nào cũng nạp lại được kể cả khi chưa
        # cài package nmt.
        "cfg": ve_dict_thuan(cfg),
        "seed": cfg.get_sau("thi_nghiem.seed") if hasattr(cfg, "get_sau") else None,
        "rng": _gom_trang_thai_rng(),
        **(so_lieu_them or {}),
    }

    duong_dan_tam = duong_dan.with_suffix(duong_dan.suffix + ".tmp")
    torch.save(noi_dung, duong_dan_tam)
    os.replace(duong_dan_tam, duong_dan)
    return duong_dan


def nap_checkpoint(
    duong_dan: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    *,
    map_location: str | torch.device | None = None,
    che_do_mong_doi: str | None = None,
    khoi_phuc_rng: bool = True,
) -> dict:
    """Nạp checkpoint và đặt lại trạng thái cho các đối tượng được truyền vào.

    Returns:
        dict có ít nhất hai khóa `buoc` và `epoch` để chạy tiếp đúng chỗ cũ,
        kèm `che_do`, `loss_dev` và `cfg` của lượt chạy đã lưu.

    Args:
        che_do_mong_doi: truyền CHE_DO_THAT ở lượt chạy thật để chặn việc vô tình
            nạp phải checkpoint của smoke test. Đây là bài học mục 1.8 và 1.9 của
            `Sưu tập lỗi.md` — lỗi đó từng làm hỏng cả một bài nộp mà không báo gì.
    """
    duong_dan = Path(duong_dan)
    if not duong_dan.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {duong_dan}")

    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    # weights_only=False vì checkpoint chứa cả trạng thái RNG của numpy và vài
    # đối tượng Python khác, không chỉ mỗi tensor. File này do chính nhóm sinh ra
    # nên nguồn gốc tin được.
    goi = torch.load(duong_dan, map_location=map_location, weights_only=False)

    thieu = [khoa for khoa in CAC_KHOA_BAT_BUOC if khoa not in goi]
    if thieu:
        raise RuntimeError(
            f"Checkpoint {duong_dan} thiếu các khóa {thieu}. "
            "Nhiều khả năng là file của phiên bản cũ hoặc bị ghi dở. "
            "Xóa đi rồi lấy lại bản trên Hugging Face Hub."
        )

    if goi["phien_ban"] != PHIEN_BAN_DINH_DANG:
        raise RuntimeError(
            f"Checkpoint {duong_dan} theo định dạng phiên bản {goi['phien_ban']}, "
            f"mã hiện tại đọc phiên bản {PHIEN_BAN_DINH_DANG}."
        )

    if che_do_mong_doi is not None and goi["che_do"] != che_do_mong_doi:
        raise RuntimeError(
            f"Checkpoint {duong_dan} thuộc chế độ {goi['che_do']!r} nhưng lượt chạy "
            f"này cần {che_do_mong_doi!r}.\n"
            "Đây đúng là cái bẫy mục 1.8 trong 'Sưu tập lỗi.md': một lượt smoke test "
            "đè lên checkpoint thật, hôm sau cả pipeline chạy bằng mô hình đồ chơi mà "
            "không có cảnh báo nào. Kiểm lại đường dẫn, hoặc xóa checkpoint smoke đi."
        )

    model.load_state_dict(goi["model"])

    # Ba đối tượng dưới đây có thể không có ở lượt chạy chỉ để suy luận, nên chỉ
    # khôi phục khi người gọi thật sự truyền vào VÀ checkpoint có lưu.
    if optimizer is not None and goi["optimizer"] is not None:
        optimizer.load_state_dict(goi["optimizer"])
    if scheduler is not None and goi["scheduler"] is not None:
        scheduler.load_state_dict(goi["scheduler"])
    if scaler is not None and goi["scaler"] is not None:
        scaler.load_state_dict(goi["scaler"])

    if khoi_phuc_rng:
        _khoi_phuc_trang_thai_rng(goi.get("rng"))

    return {
        "buoc": goi["buoc"],
        "epoch": goi["epoch"],
        "che_do": goi["che_do"],
        "loss_dev": goi.get("loss_dev"),
        "cfg": goi.get("cfg"),
    }


def doc_thong_tin(duong_dan: str | Path) -> dict[str, Any]:
    """Đọc phần mô tả của checkpoint mà KHÔNG đụng tới mô hình.

    Dùng để kiểm nhanh "file này là smoke hay thật, dừng ở bước nào" trước khi
    quyết định có nạp hay không — đúng thao tác mà mục 1.7 của `Sưu tập lỗi.md`
    khuyên: luôn nhìn xem thứ đang có là gì, đừng đoán.
    """
    goi = torch.load(Path(duong_dan), map_location="cpu", weights_only=False)
    return {
        "phien_ban": goi.get("phien_ban"),
        "che_do": goi.get("che_do"),
        "buoc": goi.get("buoc"),
        "epoch": goi.get("epoch"),
        "loss_dev": goi.get("loss_dev"),
        "seed": goi.get("seed"),
    }
