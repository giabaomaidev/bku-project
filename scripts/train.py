"""TASK 15 — Huấn luyện chính thức trên IWSLT.  Người làm: Bảo.

Dùng: python scripts/train.py --config configs/base.yaml
Ablation: python scripts/train.py --config configs/ablation_a1_layernorm.yaml --seed 42
          python scripts/train.py --config configs/ablation_a1_layernorm.yaml --seed 1337

Mỗi thí nghiệm ablation chạy TỐI THIỂU 2 seed rồi báo cáo cả độ lệch giữa các seed.
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

    raise NotImplementedError("TASK 15 — Bảo")


if __name__ == "__main__":
    main()
