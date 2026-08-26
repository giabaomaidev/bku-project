"""TASK 11 — Đo tốc độ thật trên Kaggle TRƯỚC KHI cam kết phạm vi.  Người làm: Quân.

Đây là việc trả lời trực tiếp nhận xét thứ nhất của anh Huy, và phải làm ĐẦU TIÊN
của Phase 3. Chạy thử 2000 bước trên tập con khoảng 10 nghìn câu, đo:
    - số giây trên mỗi bước
    - số token xử lý được mỗi giây
    - bộ nhớ GPU dùng đỉnh điểm
Thử vài mức so_token_moi_batch để chọn mức tốt nhất, rồi nhân ra tổng số giờ GPU
cần cho toàn bộ quá trình huấn luyện IWSLT.

Đồng thời quét cấu hình: số lớp (4 và 6), số head (4 và 8, tức mỗi head 128 và 64 chiều).
Đây là căn cứ trả lời câu hỏi của mentor về số layer và số head.

Sinh ra: results/ngan_sach_tinh_toan.csv, results/khao_sat_cau_hinh.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Chạy được ngay cả khi chưa `pip install -e .` (tiện khi làm trên Kaggle).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Console Windows mặc định dùng bảng mã cp1252, in chữ tiếng Việt ra là vỡ chữ
# hoặc UnicodeEncodeError. Ép về UTF-8 để cả nhóm đọc được log giống nhau.
for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

from nmt.utils import dat_seed, luu_config, nap_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None, help="ghi đè thi_nghiem.seed")
    args = parser.parse_args()

    cfg = nap_config(args.config)
    if args.seed is not None:
        cfg["thi_nghiem"]["seed"] = args.seed
    dat_seed(cfg.thi_nghiem.seed, cfg.thi_nghiem.deterministic)

    raise NotImplementedError("TASK 11 — Quân")


if __name__ == "__main__":
    main()
