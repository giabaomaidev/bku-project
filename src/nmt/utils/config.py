"""Hệ thống cấu hình YAML — TASK 01 (Phú).

Mọi script đều nhận `--config <duong-dan>` và không nhận hyperparameter rời rạc
qua dòng lệnh. Lý do: khi viết báo cáo phải nói được chính xác lần chạy đó dùng
cấu hình nào, mà điều đó chỉ đúng khi cấu hình nằm trọn trong một file.

Hai tính năng:

1. Kế thừa qua khóa ``ke_thua``. File ablation chỉ ghi đúng phần nó đổi, phần
   còn lại lấy từ ``base.yaml``. Nhờ vậy khi sửa base thì cả 6 file ablation
   được sửa theo, không có chuyện 6 file lệch nhau âm thầm.
2. Truy cập bằng dấu chấm: ``cfg.mo_hinh.d_model`` thay vì ``cfg["mo_hinh"]["d_model"]``.
   Gõ sai tên khóa thì báo lỗi ngay chứ không trả về None.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict cho phép truy cập bằng dấu chấm, lồng nhau bao nhiêu tầng cũng được."""

    def __getattr__(self, ten: str) -> Any:
        try:
            gia_tri = self[ten]
        except KeyError:
            raise AttributeError(
                f"Cấu hình không có khóa '{ten}'. Các khóa hiện có: {sorted(self.keys())}"
            ) from None

        # Bọc dict con thành Config rồi GHI NGƯỢC LẠI, để những lần truy cập sau
        # trả về ĐÚNG object đó chứ không phải một bản sao mới.
        #
        # Bản cũ trả thẳng `Config(gia_tri)`, tức mỗi lần gọi lại tạo một bản sao,
        # nên `cfg.mo_hinh.dropout = 0.0` ghi vào bản sao rồi vứt đi — cấu hình gốc
        # không đổi mà KHÔNG có lỗi nào. scripts/overfit_sanity.py dính đúng bẫy
        # này: tưởng đã tắt dropout cho cổng chặn overfit nhưng thực tế vẫn chạy
        # dropout 0.3, khiến loss không thể xuống dưới 0,05 và cổng chặn báo TRƯỢT
        # cho một kiến trúc vốn đúng.
        if isinstance(gia_tri, dict) and not isinstance(gia_tri, Config):
            gia_tri = Config(gia_tri)
            self[ten] = gia_tri
        return gia_tri

    def __setattr__(self, ten: str, gia_tri: Any) -> None:
        self[ten] = gia_tri

    def get_sau(self, duong_dan: str, mac_dinh: Any = None) -> Any:
        """Lấy giá trị lồng sâu: ``cfg.get_sau("mo_hinh.d_model")``."""
        nut: Any = self
        for phan in duong_dan.split("."):
            if not isinstance(nut, dict) or phan not in nut:
                return mac_dinh
            nut = nut[phan]
        return nut


def _gop_sau(goc: dict, ghi_de: dict) -> dict:
    """Gộp đệ quy. Giá trị trong `ghi_de` thắng, nhưng chỉ ở đúng nhánh của nó.

    Đây là điểm mấu chốt của cơ chế ablation: file A1 chỉ ghi
    ``mo_hinh.kieu_chuan_hoa`` thì toàn bộ phần còn lại của ``mo_hinh``
    (d_model, so_head, dropout...) vẫn giữ nguyên từ base.
    """
    ket_qua = copy.deepcopy(goc)
    for khoa, gia_tri in ghi_de.items():
        if khoa in ket_qua and isinstance(ket_qua[khoa], dict) and isinstance(gia_tri, dict):
            ket_qua[khoa] = _gop_sau(ket_qua[khoa], gia_tri)
        else:
            ket_qua[khoa] = copy.deepcopy(gia_tri)
    return ket_qua


def nap_config(duong_dan: str | Path, _da_di_qua: set[Path] | None = None) -> Config:
    """Đọc file YAML, giải quyết chuỗi kế thừa ``ke_thua``, trả về Config.

    Args:
        duong_dan: đường dẫn tới file .yaml
        _da_di_qua: dùng nội bộ để bắt kế thừa vòng (A kế thừa B, B kế thừa A)

    Ví dụ:
        >>> cfg = nap_config("configs/ablation_a1_layernorm.yaml")
        >>> cfg.mo_hinh.kieu_chuan_hoa
        'layernorm'
        >>> cfg.mo_hinh.d_model          # lấy từ base.yaml
        512
    """
    duong_dan = Path(duong_dan).resolve()
    if not duong_dan.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {duong_dan}")

    _da_di_qua = _da_di_qua or set()
    if duong_dan in _da_di_qua:
        raise ValueError(f"Kế thừa vòng trong cấu hình, phát hiện tại: {duong_dan}")
    _da_di_qua.add(duong_dan)

    with duong_dan.open("r", encoding="utf-8") as f:
        noi_dung = yaml.safe_load(f) or {}

    cha = noi_dung.pop("ke_thua", None)
    if cha is not None:
        goc = nap_config(duong_dan.parent / cha, _da_di_qua)
        noi_dung = _gop_sau(dict(goc), noi_dung)

    cfg = Config(noi_dung)
    _kiem_tra_rang_buoc(cfg)
    return cfg


def _kiem_tra_rang_buoc(cfg: Config) -> None:
    """Chặn sớm những cấu hình sai mà lúc chạy sẽ không báo lỗi gì cả.

    Đây đều là các lỗi "chạy được nhưng sai" — loại nguy hiểm nhất của đồ án này.
    """
    mo_hinh = cfg.get("mo_hinh") or {}

    d_model = mo_hinh.get("d_model")
    so_head = mo_hinh.get("so_head")
    if d_model and so_head and d_model % so_head != 0:
        raise ValueError(
            f"d_model ({d_model}) phải chia hết cho so_head ({so_head}); "
            f"mỗi head sẽ phụ trách {d_model / so_head} chiều — không phải số nguyên."
        )

    # Với Pre-Norm thì luồng residual không bao giờ được chuẩn hóa nên giá trị
    # phình dần qua từng lớp. Quên lớp chuẩn hóa cuối KHÔNG làm chương trình báo
    # lỗi, chỉ khiến mô hình mất ổn định mà không rõ nguyên nhân.
    if mo_hinh.get("vi_tri_chuan_hoa") == "pre" and not mo_hinh.get("co_chuan_hoa_cuoi"):
        raise ValueError(
            "vi_tri_chuan_hoa = 'pre' thì BẮT BUỘC co_chuan_hoa_cuoi = true. "
            "Xem bài kiểm tra số 10 trong tests/test_model_correctness.py."
        )

    # RoPE mã hóa khoảng cách tương đối giữa hai vị trí TRONG CÙNG MỘT CHUỖI.
    # Ở cross-attention, query nằm ở câu tiếng Việt còn key nằm ở câu tiếng Anh
    # nên khoảng cách giữa chúng không mang ý nghĩa gì.
    if mo_hinh.get("rope_ap_cho_cross_attention"):
        raise ValueError(
            "rope_ap_cho_cross_attention phải là false. RoPE chỉ áp cho self-attention. "
            "Xem Mục 4.2 và bài kiểm tra số 11."
        )

    hop_le = {
        "ma_hoa_vi_tri": {"rope", "sinusoidal"},
        "vi_tri_chuan_hoa": {"pre", "post"},
        "kieu_chuan_hoa": {"rmsnorm", "layernorm"},
        "kieu_ffn": {"swiglu", "relu"},
    }
    for khoa, tap in hop_le.items():
        gia_tri = mo_hinh.get(khoa)
        if gia_tri is not None and gia_tri not in tap:
            raise ValueError(f"mo_hinh.{khoa} = '{gia_tri}' không hợp lệ, phải thuộc {sorted(tap)}")

    toi_uu = cfg.get("toi_uu") or {}
    if toi_uu.get("kieu_do_chinh_xac") == "bf16":
        raise ValueError(
            "KHÔNG dùng bf16. GPU T4 của Kaggle là kiến trúc Turing (compute capability 7.5), "
            "không có phần cứng bf16. Trường hợp nhẹ là PyTorch báo lỗi, trường hợp nặng là "
            "chương trình vẫn chạy bằng giả lập phần mềm, chậm hơn cả fp32, khiến nhóm đo tốc "
            "độ rồi kết luận nhầm là T4 không đủ sức. Xem Ghi chú 4."
        )


def ve_dict_thuan(nut: Any) -> Any:
    """Đổi Config lồng nhau về dict thuần, đệ quy xuống cả list.

    `yaml.safe_dump` chỉ biết biểu diễn đúng kiểu `dict`, gặp lớp con của dict là
    ném RepresenterError. Mà `Config.__getattr__` có bọc dict con thành Config,
    nên cấu hình đã bị đọc qua sẽ chứa Config ở bên trong.
    """
    if isinstance(nut, dict):
        return {khoa: ve_dict_thuan(gia_tri) for khoa, gia_tri in nut.items()}
    if isinstance(nut, (list, tuple)):
        return [ve_dict_thuan(phan_tu) for phan_tu in nut]
    return nut


def luu_config(cfg: Config, duong_dan: str | Path) -> None:
    """Ghi lại cấu hình ĐÃ GỘP cạnh checkpoint và log.

    Bắt buộc gọi hàm này ở đầu mỗi lần chạy. Ba tháng sau nhìn lại một con số
    trong báo cáo, cái duy nhất trả lời được "chạy bằng cấu hình nào" là file này.
    """
    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    with duong_dan.open("w", encoding="utf-8") as f:
        yaml.safe_dump(ve_dict_thuan(cfg), f, allow_unicode=True, sort_keys=False)
