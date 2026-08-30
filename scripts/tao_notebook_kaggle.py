"""Sinh notebook Kaggle `notebooks/02_kaggle_train.ipynb` bằng mã.

VÌ SAO SINH BẰNG MÃ CHỨ KHÔNG SỬA TAY — mục 2.7 của `Sưu tập lỗi.md`:
    Ở đồ án trước, sửa JSON của .ipynb bằng script vá tại chỗ đã xóa mất nguyên
    một cell, khiến cell sau dùng biến chưa định nghĩa. Notebook là JSON, sửa tay
    rất dễ hỏng mà không thấy ngay.

Chạy:  python scripts/tao_notebook_kaggle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

GOC = Path(__file__).resolve().parents[1]
DUONG_DAN = GOC / "notebooks" / "02_kaggle_train.ipynb"


def md(noi_dung: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": noi_dung.strip("\n").splitlines(keepends=True)}


def code(noi_dung: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": noi_dung.strip("\n").splitlines(keepends=True)}


CAC_CELL = [
    md(r"""
# ENVI-NMT — Huấn luyện trên Kaggle T4

**Phase 3 — TASK 11 tới TASK 15.** Notebook này chạy được từ đầu tới cuối bằng Run All.

## Trước khi bấm Run All, kiểm đủ 4 thứ

| # | Việc | Chỗ làm |
|---|---|---|
| 1 | **GPU T4** | panel phải → Accelerator → **GPU T4 x2** |
| 2 | **Internet BẬT** | panel phải → Settings → Internet → **On** (mặc định TẮT) |
| 3 | **Dataset code** | Add Input → Datasets → thêm bản zip repo đã upload |
| 4 | **HF_TOKEN** | Add-ons → Secrets → thêm `HF_TOKEN` (loại **Write**) → **bật công tắc cho notebook này** |

> Nếu gắn Secret **sau** khi phiên đã khởi động thì phải **Run → Restart session**,
> không thì notebook vẫn báo không có token (mục 1.4 của `Sưu tập lỗi.md`).

## Thứ tự chạy

1. Cell cài đặt (mọi thư viện nằm ở đây, không rải rác)
2. Dò môi trường và đường dẫn
3. Kiểm GPU + token
4. Chuẩn bị dữ liệu và tokenizer
5. **SMOKE TEST** — chạy ngắn toàn bộ pipeline, chỉ để chắc Run All không crash
6. Cổng chặn overfit 50 câu (VIỆC SỐ 0)
7. TASK 11 — khảo sát cấu hình
8. TASK 15 — huấn luyện thật
9. TASK 14 — thí nghiệm giết phiên

**Smoke test KHÔNG đẩy gì lên Hugging Face** — mục 1.8: ở đồ án trước một lượt
smoke test đã đè mất checkpoint thật.
"""),

    md("## Cell 1 — Cài đặt. TOÀN BỘ thư viện nằm ở đây."),
    code(r'''
# Mọi thư viện cài ở ĐÚNG MỘT CHỖ này. Rải rác giữa notebook thì tới cell thứ 8
# mới phát hiện thiếu gói, mà lúc đó đã tốn 40 phút GPU.
#
# -q cho đỡ ngập log, nhưng KHÔNG nuốt lỗi: ngay bên dưới in phiên bản thật của
# từng gói để đối chiếu.
!pip install -q tokenizers==0.21.0 sacrebleu==2.4.3 huggingface_hub==0.27.1 PyYAML==6.0.2

import importlib

print("Phiên bản thật đang dùng:")
for ten in ["torch", "numpy", "tokenizers", "sacrebleu", "huggingface_hub", "yaml", "matplotlib"]:
    try:
        m = importlib.import_module(ten)
        print(f"  {ten:18s} {getattr(m, '__version__', '(không có __version__)')}")
    except ImportError as loi:
        print(f"  {ten:18s} THIẾU — {loi}")
        raise
print("\nCài đặt xong.")
'''),

    md("## Cell 2 — Dò môi trường, tách chỗ ĐỌC khỏi chỗ GHI"),
    code(r'''
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Mục 1.3 của Sưu tập lỗi.md: /kaggle/input CHỈ ĐỌC. Viết theo tư duy Colab, nơi
# một thư mục vừa là nguồn vừa là chỗ ghi, sẽ chết ngay ở lệnh tạo thư mục đầu tiên.
IS_KAGGLE = os.path.isdir("/kaggle/working")
OUT_BASE = Path("/kaggle/working") if IS_KAGGLE else Path.cwd()

# Mục 1.2: độ sâu của dataset trong /kaggle/input KHÔNG cố định, tùy cách tạo.
# Quét 4 cấp thay vì đoán, và in ra những gì đã quét để còn đối chiếu.
CAC_MAU = ["/kaggle/input/*", "/kaggle/input/*/*",
           "/kaggle/input/*/*/*", "/kaggle/input/*/*/*/*"]


def tim_goc_repo():
    if os.environ.get("NMT_ROOT"):
        return Path(os.environ["NMT_ROOT"])
    for mau in CAC_MAU:
        for duong_dan in sorted(glob.glob(mau)):
            if (Path(duong_dan) / "src" / "nmt").is_dir():
                return Path(duong_dan)
    return None


REPO = tim_goc_repo()
if REPO is None and (Path.cwd() / "src" / "nmt").is_dir():
    REPO = Path.cwd()

if REPO is None:
    print("KHÔNG TÌM THẤY repo. Các thư mục đã quét:")
    for mau in CAC_MAU:
        for d in sorted(glob.glob(mau))[:40]:
            print("   ", d)
    raise SystemExit(
        "Add Input > Datasets > thêm bản zip repo, hoặc đặt biến môi trường NMT_ROOT."
    )

# Repo nằm trong /kaggle/input thì CHỈ ĐỌC, mà script cần ghi results/ và
# artifacts/. Nên chép sang chỗ ghi được rồi chạy ở đó.
if IS_KAGGLE and str(REPO).startswith("/kaggle/input"):
    DICH = OUT_BASE / "bku-project"
    if not DICH.exists():
        shutil.copytree(REPO, DICH)
    REPO = DICH

os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
print(f"IS_KAGGLE = {IS_KAGGLE}")
print(f"REPO      = {REPO}   (ghi được: {os.access(REPO, os.W_OK)})")
print(f"OUT_BASE  = {OUT_BASE}")
'''),

    md("## Cell 3 — Kiểm GPU và HF_TOKEN TRƯỚC khi tốn thời gian"),
    code(r'''
import torch

print(f"CUDA có sẵn : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    kha_nang = torch.cuda.get_device_capability()
    print(f"Compute cap : {kha_nang}")
    # T4 là Turing (7.5) — KHÔNG có phần cứng bf16. Đừng dùng
    # torch.cuda.is_bf16_supported() để tự chọn: hàm đó tính cả trường hợp giả lập
    # phần mềm nên trả True ngay trên T4, chạy được nhưng chậm hơn cả fp32.
    print(f"Dùng fp16   : {'ĐÚNG (T4 không có bf16)' if kha_nang[0] < 8 else 'máy này có bf16'}")
else:
    print("KHÔNG CÓ GPU — bật panel phải > Accelerator > GPU T4 x2")

# Kiểm token SỚM. Để tới cuối mới phát hiện thiếu token là mất trắng lượt train.
from nmt.training.hub_sync import doc_token

TOKEN = doc_token(bat_buoc=False)
print(f"\nHF_TOKEN    : {'CÓ' if TOKEN else 'KHÔNG CÓ — sẽ không đồng bộ được'}")

# ĐỔI THÀNH TÊN TÀI KHOẢN HUGGING FACE CỦA CẬU
REPO_HUB = "giabaomaidev/envi-nmt-scratch-transformer"
print(f"Repo Hub    : {REPO_HUB}")
'''),

    md("## Cell 4 — Chuẩn bị dữ liệu và tokenizer"),
    code(r'''
def chay(lenh, mo_ta=""):
    """Chạy một lệnh, in log trực tiếp, dừng notebook nếu lệnh thất bại."""
    print(f"\n{'=' * 70}\n$ {lenh}\n{'=' * 70}", flush=True)
    ket_qua = subprocess.run(lenh, shell=True)
    if ket_qua.returncode != 0:
        raise RuntimeError(f"Lệnh thất bại (mã {ket_qua.returncode}): {mo_ta or lenh}")
    return ket_qua


if not Path("data/processed/train.en").exists():
    chay("python scripts/prepare_data.py --config configs/base.yaml")
else:
    print("Dữ liệu đã có, bỏ qua bước tải.")

if not Path("artifacts/tokenizer/tokenizer.json").exists():
    chay("python scripts/train_tokenizer.py --config configs/base.yaml")
else:
    print("Tokenizer đã có, bỏ qua.")

for f in ["data/processed/train.en", "data/processed/train.vi",
          "data/processed/tst2012.en", "artifacts/tokenizer/tokenizer.json"]:
    p = Path(f)
    print(f"  {'có  ' if p.exists() else 'THIẾU'} {f}")
'''),

    md(r"""
## Cell 5 — SMOKE TEST

Chạy ngắn toàn bộ pipeline. Mục đích **duy nhất**: chắc chắn Run All không crash.
Không học được gì, và **không đẩy gì lên Hugging Face**.

Cell này mà đỏ thì đừng chạy tiếp — sửa xong hãy đi tiếp, đừng đốt giờ GPU.
"""),
    code(r'''
import time

bat_dau = time.perf_counter()

chay("python scripts/benchmark_speed.py --config configs/base.yaml --nhanh",
     "TASK 11 smoke")
chay("python scripts/train.py --config configs/base.yaml --smoke",
     "TASK 15 smoke")
chay("python scripts/thi_nghiem_phuc_hoi.py --config configs/base.yaml --nhanh",
     "TASK 14 smoke")

print(f"\n{'=' * 70}")
print(f"SMOKE TEST QUA — {(time.perf_counter() - bat_dau) / 60:.1f} phút")
print("Toàn bộ pipeline chạy được. Giờ mới sang phần thật.")
print("=" * 70)
'''),

    md(r"""
## Cell 6 — VIỆC SỐ 0: cổng chặn overfit 50 câu

**Mốc bắt buộc cuối Tuần 2.** Chưa qua thì không được sang TASK 11 hay TASK 15.

Script tự thoát khác 0 khi không đạt, nên cell này đỏ nghĩa là kiến trúc có lỗi
thật — đọc phần chẩn đoán nó in ra rồi báo cả nhóm, **đừng tự sửa `src/` một mình**.
"""),
    code(r'''
chay("python scripts/overfit_sanity.py --config configs/base.yaml",
     "cổng chặn overfit 50 câu")

from IPython.display import Image, display

if Path("results/overfit_loss.png").exists():
    display(Image("results/overfit_loss.png"))
'''),

    md("## Cell 7 — TASK 11: khảo sát cấu hình và ngân sách GPU"),
    code(r'''
chay("python scripts/benchmark_speed.py --config configs/base.yaml --so-buoc 30",
     "TASK 11")

print("\n" + Path("docs/ngan_sach_tinh_toan.md").read_text(encoding="utf-8"))
'''),

    md(r"""
## Cell 8 — TASK 15: huấn luyện thật

Checkpoint và log được đẩy lên Hugging Face **sau mỗi mốc**, không đợi tới cuối.
Kaggle ngắt phiên giữa chừng thì chạy lại notebook với `--tiep-tuc` là chạy tiếp
đúng chỗ cũ — người khác cũng dùng được, không cần máy của người trước.
"""),
    code(r'''
# Lần đầu: bỏ --tiep-tuc. Lần sau, hoặc người khác chạy tiếp: THÊM --tiep-tuc.
TIEP_TUC = Path("artifacts/checkpoints").exists() and any(
    Path("artifacts/checkpoints").rglob("moi_nhat.pt")
)
co_tiep_tuc = "--tiep-tuc" if TIEP_TUC else ""
print(f"Chế độ: {'CHẠY TIẾP từ checkpoint' if TIEP_TUC else 'huấn luyện từ đầu'}")

chay(f"python scripts/train.py --config configs/base.yaml --seed 42 "
     f"--repo-hub {REPO_HUB} {co_tiep_tuc}", "TASK 15")

print("\n" + Path("docs/bao_cao_huan_luyen.md").read_text(encoding="utf-8"))
'''),

    md("## Cell 9 — TASK 14: thí nghiệm giết phiên và phục hồi"),
    code(r'''
chay("python scripts/thi_nghiem_phuc_hoi.py --config configs/base.yaml --so-buoc 60",
     "TASK 14")

from IPython.display import Image, display

display(Image("results/thi_nghiem_phuc_hoi.png"))
print(Path("docs/thi_nghiem_phuc_hoi.md").read_text(encoding="utf-8"))
'''),

    md("## Cell 10 — Đẩy nốt kết quả lên Hub và tổng kết"),
    code(r'''
from nmt.training.hub_sync import dam_bao_repo, day_thu_muc_len_hub, liet_ke_file

if TOKEN:
    dam_bao_repo(REPO_HUB)
    day_thu_muc_len_hub("results", REPO_HUB, "results")
    day_thu_muc_len_hub("docs", REPO_HUB, "docs")

    # Mục 1.7: LIỆT KÊ repo để tự mắt thấy file đã lên đúng chỗ, đừng đoán.
    print("\nFile đang có trên Hub:")
    for f in sorted(liet_ke_file(REPO_HUB)):
        print("   ", f)
else:
    print("Không có HF_TOKEN nên bỏ qua bước đẩy lên Hub.")

print(f"\n{'=' * 70}")
print("XONG PHASE 3. Nhớ bấm SAVE VERSION, nếu không thì /kaggle/working mất sạch")
print("khi đóng trình duyệt (mục 1.5 — lần mất đầu tiên tốn 3 giờ GPU).")
print("=" * 70)
'''),
]

NOTEBOOK = {
    "cells": CAC_CELL,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    DUONG_DAN.parent.mkdir(parents=True, exist_ok=True)
    DUONG_DAN.write_text(
        json.dumps(NOTEBOOK, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    so_code = sum(1 for c in CAC_CELL if c["cell_type"] == "code")
    print(f"Đã sinh {DUONG_DAN}")
    print(f"  {len(CAC_CELL)} cell ({so_code} cell code, {len(CAC_CELL) - so_code} cell chữ)")


if __name__ == "__main__":
    main()
