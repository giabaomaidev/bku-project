"""Đóng gói output Kaggle thành một zip nhỏ để train tiếp ở phiên/tài khoản khác.

VÌ SAO CẦN:
    Notebook đã Save Version thì không thêm cell được nữa, nên không đẩy được
    checkpoint lên Hugging Face từ chính nó. Cách còn lại là tải output về máy,
    lọc lấy đúng thứ cần, rồi upload thành một Kaggle Dataset mới.

    Output thô nặng khoảng 1,8 GB nhưng phần THẬT SỰ cần chỉ vài trăm MB.

LỌC THEO VÂN TAY, KHÔNG THEO TÊN FILE — bài học mục 1.8 và 1.9 của `Sưu tập lỗi.md`:
    Ở đồ án trước, một checkpoint smoke test đã bị dùng nhầm cho lượt chạy thật vì
    chỗ kiểm chỉ nhìn tên file. Script này MỞ từng checkpoint ra đọc trường
    `che_do` rồi mới quyết định, nên bản smoke không có cửa lọt vào.

Chạy:
    python scripts/dong_goi_tiep_tuc.py
    python scripts/dong_goi_tiep_tuc.py --nguon OUTPUT_KAGGEL --kem-du-lieu
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

from nmt.training.checkpoint import CHE_DO_THAT, doc_thong_tin

GOC = Path(__file__).resolve().parents[1]

# Checkpoint của thí nghiệm giết phiên chỉ là sản phẩm phụ, không dùng để train
# tiếp. Bỏ ra cho nhẹ.
THU_MUC_BO_QUA = {"thi_nghiem_phuc_hoi"}


def tim_goc_du_an(nguon: Path) -> Path:
    """Output Kaggle thường bọc thêm một tầng thư mục tên dự án."""
    if (nguon / "artifacts").is_dir():
        return nguon
    for con in sorted(nguon.iterdir()) if nguon.is_dir() else []:
        if con.is_dir() and (con / "artifacts").is_dir():
            return con
    return nguon


def chon_checkpoint(thu_muc_ckpt: Path) -> list[tuple[Path, dict]]:
    """Trả về các checkpoint HỢP LỆ để train tiếp, kèm thông tin của chúng.

    Hợp lệ nghĩa là: đọc được, và `che_do` đúng bằng "that". Bản smoke bị loại
    ngay tại đây — mở file ra đọc chứ không đoán qua tên.
    """
    hop_le: list[tuple[Path, dict]] = []
    if not thu_muc_ckpt.is_dir():
        return hop_le

    for duong_dan in sorted(thu_muc_ckpt.rglob("*.pt")):
        if any(phan in THU_MUC_BO_QUA for phan in duong_dan.parts):
            print(f"  bỏ (sản phẩm phụ) : {duong_dan.name}  [{duong_dan.parent.name}]")
            continue

        try:
            thong_tin = doc_thong_tin(duong_dan)
        except Exception as loi:
            print(f"  bỏ (đọc hỏng)     : {duong_dan.name}  {type(loi).__name__}")
            continue

        mb = duong_dan.stat().st_size / (1024 ** 2)
        if thong_tin.get("che_do") != CHE_DO_THAT:
            print(f"  BỎ (chế độ {thong_tin.get('che_do')!r}) : {duong_dan.name} "
                  f"[{duong_dan.parent.name}]  {mb:.0f} MB")
            continue

        print(f"  GIỮ  bước {thong_tin['buoc']:>6,} · epoch {thong_tin['epoch']:>3} "
              f"· {mb:.0f} MB  ->  {duong_dan.parent.name}/{duong_dan.name}")
        hop_le.append((duong_dan, thong_tin))

    return hop_le


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nguon", default="OUTPUT_KAGGEL",
                        help="thư mục output Kaggle đã tải về")
    parser.add_argument("--dich", default="bku-project-tiep-tuc.zip")
    parser.add_argument("--kem-du-lieu", action="store_true",
                        help="gói cả data/processed. Không gói thì notebook tự tải "
                             "lại mất khoảng 10 phút, nhưng zip nhẹ đi nhiều")
    parser.add_argument("--kem-tot-nhat", action="store_true",
                        help="gói cả tot_nhat.pt. Thường không cần: lần đánh giá "
                             "đầu của phiên sau sẽ sinh bản tốt nhất mới")
    args = parser.parse_args()

    nguon = tim_goc_du_an(Path(args.nguon))
    if not nguon.is_dir():
        raise SystemExit(f"Không thấy thư mục nguồn: {args.nguon}")
    print(f"Nguồn: {nguon}\n")

    # --- Chọn checkpoint ------------------------------------------------------
    print("CHECKPOINT")
    cac_checkpoint = chon_checkpoint(nguon / "artifacts" / "checkpoints")
    if not cac_checkpoint:
        raise SystemExit(
            "\nKhông có checkpoint hợp lệ nào (chế độ 'that'). Không đóng gói được.\n"
            "Nếu chỉ có bản smoke thì phải huấn luyện lại từ đầu."
        )

    if not args.kem_tot_nhat:
        cac_checkpoint = [
            (p, t) for p, t in cac_checkpoint if p.name != "tot_nhat.pt"
        ]
        print("  (bỏ tot_nhat.pt cho nhẹ — dùng --kem-tot-nhat nếu muốn giữ)")

    # --- Gom danh sách file ---------------------------------------------------
    can_gom: list[tuple[Path, str]] = []
    for duong_dan, _ in cac_checkpoint:
        can_gom.append((duong_dan, str(duong_dan.relative_to(nguon)).replace("\\", "/")))

    tokenizer = nguon / "artifacts" / "tokenizer" / "tokenizer.json"
    if tokenizer.is_file():
        can_gom.append((tokenizer, "artifacts/tokenizer/tokenizer.json"))
    else:
        # Tokenizer lệch nhau giữa các máy thì token ID không khớp và checkpoint
        # nạp vào ra rác, mà không có lỗi nào báo. Thiếu nó là hỏng cả gói.
        raise SystemExit("THIẾU artifacts/tokenizer/tokenizer.json — gói này vô dụng.")

    # Log cũ, để phiên sau nối tiếp đường loss thay vì vẽ lại từ đầu.
    for csv in sorted((nguon / "results" / "logs").rglob("*.csv")):
        can_gom.append((csv, str(csv.relative_to(nguon)).replace("\\", "/")))

    if args.kem_du_lieu:
        for f in sorted((nguon / "data" / "processed").glob("*")):
            if f.is_file() and f.suffix in (".en", ".vi"):
                can_gom.append((f, str(f.relative_to(nguon)).replace("\\", "/")))

    # --- Ghi zip --------------------------------------------------------------
    dich = GOC.parent / args.dich
    print(f"\nĐANG GÓI -> {dich}")
    tong = 0
    with zipfile.ZipFile(dich, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for duong_dan, ten_trong_zip in can_gom:
            mb = duong_dan.stat().st_size / (1024 ** 2)
            tong += duong_dan.stat().st_size
            print(f"  {mb:8.1f} MB  {ten_trong_zip}", flush=True)
            z.write(duong_dan, ten_trong_zip)

    kich_thuoc_zip = dich.stat().st_size / (1024 ** 2)
    print(f"\nXONG: {dich}")
    print(f"  {len(can_gom)} file · gốc {tong / (1024 ** 2):.0f} MB "
          f"· nén còn {kich_thuoc_zip:.0f} MB")

    buoc_xa_nhat = max(t["buoc"] for _, t in cac_checkpoint)
    print(f"\nCheckpoint xa nhất: bước {buoc_xa_nhat:,}")
    print("\nBƯỚC TIẾP THEO")
    print("  1. kaggle.com/datasets > New Dataset > upload file zip này")
    print("     Đặt tên: bku-project-tiep-tuc")
    print("  2. Notebook mới: Add Input > cả HAI dataset")
    print("       bku-project            (mã nguồn)")
    print("       bku-project-tiep-tuc   (checkpoint + tokenizer)")
    print("  3. Cell 2 đặt SMOKE_TEST = False, rồi Run All")
    print("     Cell 7 tự thấy checkpoint và chạy tiếp từ bước "
          f"{buoc_xa_nhat:,}")


if __name__ == "__main__":
    main()
