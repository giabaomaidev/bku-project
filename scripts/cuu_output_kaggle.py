"""Cứu output của một notebook Kaggle đã Save Version, đẩy lên Hugging Face Hub.

VÌ SAO CẦN FILE NÀY:
    Notebook đã bấm Save Version thì KHÔNG thêm cell được nữa. Nếu lượt chạy đó
    huấn luyện xong mà chưa đẩy được checkpoint lên Hub, cách duy nhất lấy lại là
    mở một notebook MỚI, gắn output của notebook cũ vào Input, rồi đẩy từ đó.

    Đúng tình huống ngày 2026-09-03: lượt train 13 tiếng chạy xong nhưng Hub trống
    vì `dam_bao_repo()` bị gọi SAU `trainer.train()`, nên mọi lần đẩy trong lúc
    huấn luyện đều rơi vào RepositoryNotFoundError. Lỗi đó đã sửa, file này để cứu
    những lượt chạy trót dính.

CỐ Ý KHÔNG IMPORT `nmt`:
    Output của notebook cũ có kèm bản `src/` đời cũ. Phụ thuộc vào nó là tự trói
    tay vào đúng phiên bản đang lỗi. File này chỉ cần `huggingface_hub`.

CÁCH DÙNG trên Kaggle:
    1. Tạo notebook MỚI
    2. Add Input -> Notebook Output -> chọn notebook cũ
    3. Add-ons -> Secrets -> bật HF_TOKEN (loại Write)
    4. Settings -> Internet -> On
    5. Dán nội dung file này vào một cell, sửa REPO_HUB, rồi chạy

Hoặc chạy như script:
    python scripts/cuu_output_kaggle.py --repo-hub mgbao/envi-nmt-scratch-transformer
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

TEN_BIEN_TOKEN = "HF_TOKEN"

# Độ sâu của output notebook trong /kaggle/input không cố định, tùy cách Kaggle
# đặt tên. Quét nhiều cấp thay vì đoán — bài học mục 1.2 của `Sưu tập lỗi.md`.
CAC_MAU_QUET = [
    "/kaggle/input/*",
    "/kaggle/input/*/*",
    "/kaggle/input/*/*/*",
    "/kaggle/input/*/*/*/*",
]

# Ba thư mục cần cứu, kèm chỗ đặt tương ứng trên Hub.
CAN_CUU = [
    ("artifacts/checkpoints", "checkpoints"),
    ("results", "results"),
    ("docs", "docs"),
]


def doc_token() -> str:
    """Kaggle Secrets -> biến môi trường -> báo lỗi rõ ràng."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        token = UserSecretsClient().get_secret(TEN_BIEN_TOKEN)
        if token:
            return token
    except ImportError:
        pass
    except Exception as loi:
        print(f"[cuu] Đọc Kaggle Secret {TEN_BIEN_TOKEN!r} thất bại: "
              f"{type(loi).__name__}: {loi}")
        print("[cuu] Sửa: Add-ons > Secrets > BẬT CÔNG TẮC cho notebook này, "
              "rồi Run > Restart session.")

    token = os.environ.get(TEN_BIEN_TOKEN)
    if token:
        return token

    raise SystemExit(
        f"Không tìm thấy {TEN_BIEN_TOKEN}. Add-ons > Secrets > thêm token loại "
        "Write, bật công tắc cho notebook này, rồi Restart session."
    )


def tim_thu_muc_co_checkpoint() -> list[Path]:
    """Tìm mọi thư mục trong /kaggle/input có chứa file .pt.

    Trả về danh sách chứ không phải một kết quả, vì có thể gắn nhiều notebook
    output cùng lúc. In hết ra để tự mắt chọn, đừng để script đoán hộ.
    """
    ung_vien: list[Path] = []
    for mau in CAC_MAU_QUET:
        for duong_dan in sorted(glob.glob(mau)):
            p = Path(duong_dan)
            if p.is_dir() and any(p.rglob("*.pt")):
                ung_vien.append(p)
    return ung_vien


def tim_goc_du_an(thu_muc: Path) -> Path:
    """Tìm thư mục gốc chứa artifacts/ hoặc results/ bên trong output."""
    if (thu_muc / "artifacts").is_dir() or (thu_muc / "results").is_dir():
        return thu_muc
    for con in sorted(thu_muc.rglob("*")):
        if con.is_dir() and ((con / "artifacts").is_dir() or (con / "results").is_dir()):
            return con
    return thu_muc


def day_thu_muc(api, thu_muc: Path, repo_id: str, tien_to: str, token: str) -> bool:
    if not thu_muc.is_dir():
        print(f"  bỏ qua (không có): {thu_muc}")
        return False

    so_file = sum(1 for _ in thu_muc.rglob("*") if _.is_file())
    dung_luong = sum(f.stat().st_size for f in thu_muc.rglob("*") if f.is_file())
    print(f"  đẩy {thu_muc}  ->  {tien_to}/   "
          f"({so_file} file, {dung_luong / (1024 ** 2):.1f} MB)", flush=True)

    try:
        api.upload_folder(
            folder_path=str(thu_muc),
            path_in_repo=tien_to,
            repo_id=repo_id,
            token=token,
            commit_message=f"cứu {tien_to} từ output notebook Kaggle",
        )
        return True
    except Exception as loi:
        print(f"  HỎNG: {type(loi).__name__}: {loi}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-hub", required=True,
                        help="ví dụ mgbao/envi-nmt-scratch-transformer")
    parser.add_argument("--goc", default=None,
                        help="chỉ định tay thư mục output, bỏ qua bước dò tự động")
    args = parser.parse_args()

    token = doc_token()

    from huggingface_hub import HfApi

    api = HfApi()

    # --- Tìm output của notebook cũ ------------------------------------------
    if args.goc:
        goc = Path(args.goc)
    else:
        ung_vien = tim_thu_muc_co_checkpoint()
        if not ung_vien:
            print("KHÔNG thấy thư mục nào chứa file .pt. Các thư mục đã quét:")
            for mau in CAC_MAU_QUET[:2]:
                for d in sorted(glob.glob(mau))[:30]:
                    print("   ", d)
            raise SystemExit(
                "Add Input > Notebook Output > chọn notebook cũ, hoặc truyền --goc."
            )
        print("Các thư mục có checkpoint:")
        for p in ung_vien:
            print("   ", p)
        goc = tim_goc_du_an(ung_vien[0])

    print(f"\nGốc dự án: {goc}")
    for f in sorted(goc.rglob("*.pt")):
        print(f"  checkpoint: {f.relative_to(goc)}  "
              f"({f.stat().st_size / (1024 ** 2):.1f} MB)")

    # --- Tạo repo TRƯỚC rồi mới đẩy ------------------------------------------
    print(f"\nTạo/mở repo {args.repo_hub} ...")
    api.create_repo(repo_id=args.repo_hub, token=token, private=True, exist_ok=True)
    print("Repo sẵn sàng.\n")

    for duong_dan_con, tien_to in CAN_CUU:
        day_thu_muc(api, goc / duong_dan_con, args.repo_hub, tien_to, token)

    # --- Liệt kê để tự mắt thấy, đừng đoán ------------------------------------
    print(f"\n{'=' * 60}\nFile hiện có trên {args.repo_hub}:\n{'=' * 60}")
    for f in sorted(api.list_repo_files(repo_id=args.repo_hub, token=token)):
        print("   ", f)

    print("\nXong. Giờ chạy tiếp ở máy/tài khoản khác bằng:")
    print(f"  python scripts/train.py --config configs/base.yaml --seed 42 \\")
    print(f"      --repo-hub {args.repo_hub} --tiep-tuc")


if __name__ == "__main__":
    main()
