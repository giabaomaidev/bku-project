"""Sinh notebook Kaggle `notebooks/02_kaggle_train.ipynb` bằng mã.

VÌ SAO SINH BẰNG MÃ CHỨ KHÔNG SỬA TAY — mục 2.7 của `Sưu tập lỗi.md`:
    Ở đồ án trước, sửa JSON của .ipynb bằng script vá tại chỗ đã xóa mất nguyên
    một cell, khiến cell sau dùng biến chưa định nghĩa.

BỐ CỤC NOTEBOOK — học từ notebook LegalIR đã chạy thành công:
    - MỘT cell cấu hình duy nhất ở đầu, chứa mọi công tắc
    - SMOKE_TEST là công tắc bật/tắt, KHÔNG phải một bước riêng trong cùng lượt
      Run All. Chạy Run All với True, xanh hết thì đổi thành False rồi Run All lại
    - Smoke test đẩy lên Hub ở nhánh `smoke/` tách hẳn, nhờ vậy kiểm được luôn cơ
      chế đẩy mà không đụng vào dữ liệu thật
    - Thử ghi lên Hub NGAY từ đầu, trước khi tốn một giây GPU nào

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

**Phase 3 — TASK 11 tới TASK 15.**

## Cách dùng: chạy Run All HAI LẦN

| Lượt | Đặt gì ở Cell 2 | Mất bao lâu | Để làm gì |
|---|---|---|---|
| **1** | `SMOKE_TEST = True` | ~5 phút | Chứng minh Run All không crash. Đẩy lên nhánh `smoke/` trên Hub để kiểm luôn cơ chế đẩy |
| **2** | `SMOKE_TEST = False` | ~10,5 giờ mỗi phiên | Huấn luyện thật |

Lượt 1 mà đỏ ở bất kỳ đâu thì **sửa xong hãy sang lượt 2**. Đó là toàn bộ lý do
smoke test tồn tại — bắt lỗi trước khi đốt giờ GPU.

## Trước khi bấm Run All, kiểm đủ 4 thứ

| # | Việc | Chỗ làm |
|---|---|---|
| 1 | **GPU T4** | panel phải → Accelerator → **GPU T4 x2** |
| 2 | **Internet BẬT** | panel phải → Settings → Internet → **On** (mặc định TẮT) |
| 3 | **Dataset code** | Add Input → Datasets → bản zip repo đã upload |
| 4 | **HF_TOKEN** | Add-ons → Secrets → `HF_TOKEN` loại **Write** → **bật công tắc cho notebook này** |

> Gắn Secret **sau** khi phiên đã khởi động thì phải **Run → Restart session**.

## Huấn luyện dài chạy nhiều phiên

Phiên GPU Kaggle bị cắt ở khoảng **12 giờ**, mà 60.000 bước mất hơn 13 giờ.
Nên `GIO_TOI_DA = 10.5`: trainer tự dừng khi còn sống, lưu checkpoint, đẩy lên
Hugging Face rồi thoát sạch. Phiên sau chạy lại notebook này là tự kéo về chạy tiếp.

**Hết quota thì đổi tài khoản, gắn `HF_TOKEN` của người đó, Run All lại.**
Không cần máy của người trước — toàn bộ trạng thái nằm trên Hub.
"""),

    md("## Cell 1 — Cài đặt. TOÀN BỘ thư viện nằm ở đây."),
    code(r'''
# Mọi thư viện cài ở ĐÚNG MỘT CHỖ này. Rải rác giữa notebook thì tới cell thứ 8
# mới phát hiện thiếu gói, mà lúc đó đã tốn 40 phút GPU.
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

    md(r"""
## Cell 2 — CẤU HÌNH. Mọi công tắc nằm ở đây.

Đây là cell duy nhất cần sửa tay.
"""),
    code(r'''
# ===================== CÔNG TẮC =====================

SMOKE_TEST = True      # True: chạy nháp ~5 phút. False: huấn luyện thật.

REPO_HUB   = "mgbao/envi-nmt-scratch-transformer"    # ĐỔI THÀNH TÀI KHOẢN HF CỦA CẬU
SEED       = 42
GIO_TOI_DA = 10.5      # ngân sách mỗi phiên, dưới ngưỡng cắt 12 giờ của Kaggle

# ====================================================

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Mục 1.3 của Sưu tập lỗi.md: /kaggle/input CHỈ ĐỌC. Viết theo tư duy Colab, nơi
# một thư mục vừa là nguồn vừa là chỗ ghi, sẽ chết ngay ở lệnh tạo thư mục đầu.
IS_KAGGLE = os.path.isdir("/kaggle/working")
OUT_BASE = Path("/kaggle/working") if IS_KAGGLE else Path.cwd()

# Mục 1.2: độ sâu dataset trong /kaggle/input KHÔNG cố định. Quét 4 cấp thay vì
# đoán, và in ra những gì đã quét để còn đối chiếu.
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
    raise SystemExit("Add Input > Datasets > thêm bản zip repo, hoặc đặt NMT_ROOT.")

# Repo nằm trong /kaggle/input thì CHỈ ĐỌC, mà script cần ghi results/ và
# artifacts/. Chép sang chỗ ghi được rồi chạy ở đó.
if IS_KAGGLE and str(REPO).startswith("/kaggle/input"):
    DICH = OUT_BASE / "bku-project"
    if not DICH.exists():
        shutil.copytree(REPO, DICH)
    REPO = DICH

os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))


def chay(lenh, mo_ta=""):
    """Chạy một lệnh, in log trực tiếp, dừng notebook nếu lệnh thất bại."""
    print(f"\n{'=' * 70}\n$ {lenh}\n{'=' * 70}", flush=True)
    ket_qua = subprocess.run(lenh, shell=True)
    if ket_qua.returncode != 0:
        raise RuntimeError(f"Lệnh thất bại (mã {ket_qua.returncode}): {mo_ta or lenh}")
    return ket_qua


print(f"CHẾ ĐỘ    : {'SMOKE TEST (chạy nháp)' if SMOKE_TEST else 'HUẤN LUYỆN THẬT'}")
print(f"IS_KAGGLE : {IS_KAGGLE}")
print(f"REPO      : {REPO}   (ghi được: {os.access(REPO, os.W_OK)})")
print(f"Nhánh Hub : {'smoke/...' if SMOKE_TEST else '(gốc repo)'}")
'''),

    md(r"""
## Cell 3 — Kiểm GPU, token, và THỬ GHI THẬT lên Hub

Cell này tốn 10 giây và nó thay thế cho bài học đắt nhất: một lượt chạy 13 tiếng
đã mất trắng vì repo trên Hub chưa được tạo, nên mọi lần đẩy checkpoint đều
thất bại âm thầm suốt cả lượt.
"""),
    code(r'''
import torch

print(f"CUDA có sẵn : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    kha_nang = torch.cuda.get_device_capability()
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"Compute cap : {kha_nang}")
    # T4 là Turing (7.5) — KHÔNG có phần cứng bf16. Đừng dùng
    # torch.cuda.is_bf16_supported() để tự chọn: hàm đó tính cả trường hợp giả lập
    # phần mềm nên trả True ngay trên T4, chạy được nhưng chậm hơn cả fp32.
    print(f"Dùng fp16   : {'ĐÚNG (T4 không có bf16)' if kha_nang[0] < 8 else 'máy này có bf16'}")
elif not SMOKE_TEST:
    raise SystemExit("KHÔNG CÓ GPU — panel phải > Accelerator > GPU T4 x2")

from nmt.training.hub_sync import doc_token

TOKEN = doc_token(bat_buoc=False)
if not TOKEN:
    raise SystemExit(
        "KHÔNG CÓ HF_TOKEN — dừng ở đây thay vì train xong rồi mất trắng.\n"
        "Add-ons > Secrets > thêm HF_TOKEN loại Write > BẬT công tắc cho notebook "
        "này > Run > Restart session."
    )

# THỬ GHI THẬT. Chứng minh cùng lúc ba thứ: mạng thông, token có quyền Write,
# repo tồn tại. Cả ba đều là thứ chỉ lộ ra sau nhiều giờ nếu không kiểm ở đây.
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id=REPO_HUB, token=TOKEN, private=True, exist_ok=True)
Path("kiem_tra_ghi.txt").write_text("thử quyền ghi", encoding="utf-8")
api.upload_file(
    path_or_fileobj="kiem_tra_ghi.txt",
    path_in_repo=("smoke/" if SMOKE_TEST else "") + "kiem_tra_ghi.txt",
    repo_id=REPO_HUB,
    token=TOKEN,
    commit_message="thử quyền ghi trước khi train",
)
print(f"\nGHI THỬ LÊN HUB: THÀNH CÔNG -> {REPO_HUB}")

# Mục 1.7: LIỆT KÊ repo để tự mắt thấy đang có gì, đừng bao giờ đoán.
print("\nFile hiện có trên Hub:")
for f in sorted(api.list_repo_files(repo_id=REPO_HUB, token=TOKEN)):
    print("   ", f)
'''),

    md("## Cell 4 — Chuẩn bị dữ liệu và tokenizer"),
    code(r'''
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

# Tokenizer phải đi cùng checkpoint. Mỗi người tự chạy train_tokenizer.py có thể
# ra file khác nhau nếu lệch phiên bản thư viện, khi đó checkpoint nạp vào dữ liệu
# người khác sẽ ra rác vì token ID không khớp.
from nmt.training.hub_sync import TEN_TOKENIZER, day_len_hub
from nmt.training.checkpoint import CHE_DO_SMOKE, CHE_DO_THAT

CHE_DO = CHE_DO_SMOKE if SMOKE_TEST else CHE_DO_THAT
day_len_hub("artifacts/tokenizer/tokenizer.json", REPO_HUB, TEN_TOKENIZER, che_do=CHE_DO)
'''),

    md("## Cell 5 — TASK 11: khảo sát cấu hình và ngân sách GPU"),
    code(r'''
co_nhanh = "--nhanh" if SMOKE_TEST else "--so-buoc 30"
chay(f"python scripts/benchmark_speed.py --config configs/base.yaml {co_nhanh}",
     "TASK 11")

print("\n" + Path("docs/ngan_sach_tinh_toan.md").read_text(encoding="utf-8"))
'''),

    md(r"""
## Cell 6 — VIỆC SỐ 0: cổng chặn overfit 50 câu

**Mốc bắt buộc cuối Tuần 2.** Bỏ qua ở lượt smoke vì nó cần chạy thật mới có ý
nghĩa. Cell này đỏ nghĩa là kiến trúc có lỗi thật — đọc phần chẩn đoán nó in ra
rồi báo cả nhóm, **đừng tự sửa `src/` một mình**.
"""),
    code(r'''
if SMOKE_TEST:
    print("Bỏ qua cổng chặn overfit ở lượt smoke — nó cần chạy đủ mới có ý nghĩa.")
else:
    chay("python scripts/overfit_sanity.py --config configs/base.yaml",
         "cổng chặn overfit 50 câu")

    from IPython.display import Image, display

    if Path("results/overfit_loss.png").exists():
        display(Image("results/overfit_loss.png"))
'''),

    md(r"""
## Cell 7 — TASK 15: huấn luyện

`--tiep-tuc` LUÔN bật. Chưa có checkpoint thì script tự huấn luyện từ đầu; có rồi
thì kéo từ Hub về chạy tiếp đúng bước đang dở.

Đừng kiểm bằng đĩa cục bộ để quyết định bật hay không: phiên Kaggle mới luôn có
`/kaggle/working` trắng, nên kiểm kiểu đó lúc nào cũng ra "chưa có" và lượt chạy
tiếp theo lại bắt đầu lại từ số 0.
"""),
    code(r'''
if SMOKE_TEST:
    # Vài bước trên vài trăm câu. Không học được gì, chỉ để chắc không crash.
    chay(f"python scripts/train.py --config configs/base.yaml --seed {SEED} "
         f"--repo-hub {REPO_HUB} --smoke", "TASK 15 smoke")
else:
    chay(f"python scripts/train.py --config configs/base.yaml --seed {SEED} "
         f"--repo-hub {REPO_HUB} --tiep-tuc --gio-toi-da {GIO_TOI_DA}", "TASK 15")

print("\n" + Path("docs/bao_cao_huan_luyen.md").read_text(encoding="utf-8"))
'''),

    md("## Cell 8 — TASK 14: thí nghiệm giết phiên và phục hồi"),
    code(r'''
co_nhanh = "--nhanh" if SMOKE_TEST else "--so-buoc 60"
chay(f"python scripts/thi_nghiem_phuc_hoi.py --config configs/base.yaml {co_nhanh}",
     "TASK 14")

from IPython.display import Image, display

display(Image("results/thi_nghiem_phuc_hoi.png"))
print(Path("docs/thi_nghiem_phuc_hoi.md").read_text(encoding="utf-8"))
'''),

    md("## Cell 9 — Đẩy kết quả lên Hub và tổng kết"),
    code(r'''
from nmt.training.hub_sync import day_thu_muc_len_hub, liet_ke_file

day_thu_muc_len_hub("results", REPO_HUB, "results", che_do=CHE_DO)
day_thu_muc_len_hub("docs", REPO_HUB, "docs", che_do=CHE_DO)
day_thu_muc_len_hub("artifacts/checkpoints", REPO_HUB, "checkpoints", che_do=CHE_DO)

print("\nFile đang có trên Hub:")
for f in sorted(liet_ke_file(REPO_HUB)):
    print("   ", f)

print(f"\n{'=' * 70}")
if SMOKE_TEST:
    print("SMOKE TEST QUA. Toàn bộ pipeline chạy được và đẩy Hub được.")
    print()
    print("BƯỚC TIẾP THEO:")
    print("  1. Lên Cell 2, đổi SMOKE_TEST = False")
    print("  2. Run All lại")
    print("  3. Nhớ Save Version, nếu không thì /kaggle/working mất sạch")
else:
    print("XONG MỘT PHIÊN HUẤN LUYỆN.")
    print()
    print("Đọc dòng cuối của Cell 7: còn bước thì mở phiên Kaggle mới, Run All lại.")
    print("Checkpoint đã nằm trên Hub nên tài khoản nào chạy tiếp cũng được.")
    print()
    print("NHỚ BẤM SAVE VERSION.")
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
