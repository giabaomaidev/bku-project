"""TASK 15 — Huấn luyện chính thức trên IWSLT.  Người làm: Bảo.

Dùng: python scripts/train.py --config configs/base.yaml
Ablation: python scripts/train.py --config configs/ablation_a1_layernorm.yaml --seed 42
          python scripts/train.py --config configs/ablation_a1_layernorm.yaml --seed 1337

Mỗi thí nghiệm ablation chạy TỐI THIỂU 2 seed rồi báo cáo cả độ lệch giữa các seed.

BA ĐIỀU KHIẾN LƯỢT CHẠY NÀY KHÔNG MẤT TRẮNG KHI KAGGLE NGẮT PHIÊN:

1. `--tiep-tuc` tự kéo checkpoint mới nhất từ Hugging Face Hub về rồi chạy tiếp
   đúng chỗ cũ. Nhờ vậy người hết quota bàn lại cho người khác chạy tiếp được, và
   giám khảo muốn dựng lại từ đầu cũng chỉ cần clone repo trên Hub.

2. Checkpoint và CẢ thư mục log được đẩy lên Hub sau mỗi mốc chứ không đợi tới
   cuối. Mục 1.5 của `Sưu tập lỗi.md`: /kaggle/working chỉ tồn tại trong phiên,
   lần mất đầu tiên tốn 3 giờ GPU.

3. `--smoke` chạy vài bước để kiểm notebook không crash, và KHÔNG đẩy gì lên Hub.
   Mục 1.8: ở đồ án trước, một lượt smoke test đã đè mất checkpoint thật.

Sinh ra:
    artifacts/checkpoints/<tên lượt chạy>/{moi_nhat,tot_nhat}.pt
    results/logs/<tên lượt chạy>/metrics.csv
    docs/bao_cao_huan_luyen.md      <- báo cáo TỰ ĐIỀN
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Chạy được ngay cả khi chưa `pip install -e .` (tiện khi làm trên Kaggle).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Console Windows mặc định dùng bảng mã cp1252, in chữ tiếng Việt ra là vỡ chữ
# hoặc UnicodeEncodeError. Ép về UTF-8 để cả nhóm đọc được log giống nhau.
for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

from nmt.utils import BoGhiLog, dat_seed, luu_config, nap_config

GOC = Path(__file__).resolve().parents[1]

# Smoke test chỉ cần đủ để khẳng định "Run All không crash". Để số này nhỏ, vì
# mục đích của nó KHÔNG phải là học được gì.
SO_BUOC_SMOKE = 6
SO_CAU_SMOKE = 200


def nap_du_lieu(cfg, smoke: bool):
    """Trả về (train_loader, dev_loader, tokenizer). Thiếu file thì báo lỗi rõ ràng."""
    from nmt.data import DuLieuSongNgu, nap_tokenizer, tao_dataloader
    from nmt.utils import sinh_generator

    duong_dan_tokenizer = Path(cfg.du_lieu.tokenizer)
    thieu = [
        p for p in (
            Path(str(cfg.du_lieu.train) + ".en"),
            Path(str(cfg.du_lieu.train) + ".vi"),
            Path(str(cfg.du_lieu.dev) + ".en"),
            Path(str(cfg.du_lieu.dev) + ".vi"),
            duong_dan_tokenizer,
        ) if not p.exists()
    ]
    if thieu:
        raise FileNotFoundError(
            "Thiếu các file sau:\n  " + "\n  ".join(str(p) for p in thieu) + "\n"
            "Chạy trước hai lệnh này:\n"
            "  python scripts/prepare_data.py    --config configs/base.yaml\n"
            "  python scripts/train_tokenizer.py --config configs/base.yaml"
        )

    tokenizer = nap_tokenizer(str(duong_dan_tokenizer))
    train_set = DuLieuSongNgu(
        str(cfg.du_lieu.train) + ".en", str(cfg.du_lieu.train) + ".vi",
        tokenizer, cfg.du_lieu.do_dai_toi_da,
    )
    dev_set = DuLieuSongNgu(
        str(cfg.du_lieu.dev) + ".en", str(cfg.du_lieu.dev) + ".vi",
        tokenizer, cfg.du_lieu.do_dai_toi_da,
    )

    if smoke:
        # Cắt nhỏ để Run All xong trong vài phút. Smoke test không cần học được gì.
        for tap in (train_set, dev_set):
            tap._src = tap._src[:SO_CAU_SMOKE]
            tap._tgt = tap._tgt[:SO_CAU_SMOKE]

    generator = sinh_generator(cfg.thi_nghiem.seed)
    train_loader = tao_dataloader(
        train_set,
        so_token_moi_batch=cfg.du_lieu.so_token_moi_batch,
        gom_theo_do_dai=cfg.du_lieu.gom_theo_do_dai,
        so_worker=cfg.du_lieu.so_worker,
        generator=generator,
    )
    dev_loader = tao_dataloader(
        dev_set,
        so_token_moi_batch=cfg.du_lieu.so_token_moi_batch,
        gom_theo_do_dai=True,
        so_worker=cfg.du_lieu.so_worker,
        tron=False,
        generator=generator,
    )
    print(f"[train] train {len(train_set):,} câu · {len(train_loader):,} batch | "
          f"dev {len(dev_set):,} câu · {len(dev_loader):,} batch")
    return train_loader, dev_loader, tokenizer


def viet_bao_cao(duong_dan: Path, cfg, model, ket_qua: dict, giay_chay: float,
                 giay_dong_bo: float, smoke: bool, duong_dan_log: Path) -> None:
    """Ghi docs/bao_cao_huan_luyen.md từ số liệu THẬT của lượt chạy vừa xong."""
    tham_so = model.dem_tham_so()
    ty_le_dong_bo = giay_dong_bo / giay_chay * 100 if giay_chay > 0 else 0.0

    dong: list[str] = []
    ghi = dong.append

    ghi("# Báo cáo huấn luyện — TASK 13 + TASK 15\n")
    ghi("> File này do `scripts/train.py` TỰ SINH sau mỗi lượt chạy.")
    ghi("> Đừng sửa tay: chạy lại là mọi con số được cập nhật cùng lúc.\n")

    if smoke:
        ghi("> ⚠️ **ĐÂY LÀ LƯỢT SMOKE TEST**, không phải kết quả thật. Nó chỉ chứng minh")
        ghi("> notebook chạy Run All không crash, và không đẩy gì lên Hugging Face.\n")

    ghi("## Cấu hình lượt chạy\n")
    ghi(f"- Thí nghiệm: `{cfg.thi_nghiem.ten}`")
    ghi(f"- Seed: **{cfg.thi_nghiem.seed}** · deterministic: {cfg.thi_nghiem.deterministic}")
    ghi(f"- Kiến trúc: {cfg.mo_hinh.so_lop_encoder} lớp encoder + "
        f"{cfg.mo_hinh.so_lop_decoder} lớp decoder · d_model {cfg.mo_hinh.d_model} · "
        f"{cfg.mo_hinh.so_head} head")
    ghi(f"- Chuẩn hóa: {cfg.mo_hinh.kieu_chuan_hoa} / {cfg.mo_hinh.vi_tri_chuan_hoa}-norm · "
        f"FFN: {cfg.mo_hinh.kieu_ffn} (d_ff {cfg.mo_hinh.d_ff}) · "
        f"mã hóa vị trí: {cfg.mo_hinh.ma_hoa_vi_tri}")
    ghi(f"- Optimizer: {cfg.toi_uu.optimizer} lr {cfg.toi_uu.learning_rate} · "
        f"scheduler {cfg.toi_uu.scheduler} · label smoothing {cfg.toi_uu.label_smoothing}")
    ghi(f"- fp16: {cfg.toi_uu.do_chinh_xac_hon_hop} · "
        f"cộng dồn gradient {cfg.toi_uu.so_buoc_cong_don_gradient} · "
        f"cắt gradient {cfg.toi_uu.cat_gradient_norm}\n")

    ghi("## Số tham số\n")
    ghi("| thành phần | số tham số |")
    ghi("|---|---|")
    for ten, gia_tri in tham_so.items():
        ghi(f"| {ten} | {gia_tri:,} |")
    ghi("")

    ghi("## Kết quả\n")
    ghi("| chỉ số | giá trị |")
    ghi("|---|---|")
    ghi(f"| số bước đã chạy | {ket_qua['so_buoc_da_chay']:,} |")
    ghi(f"| bước cuối | {ket_qua['buoc_cuoi']:,} |")
    ghi(f"| epoch | {ket_qua['epoch']} |")
    if ket_qua.get("loss_train_cuoi") is not None:
        ghi(f"| loss train cuối | {ket_qua['loss_train_cuoi']:.4f} |")
    if ket_qua.get("loss_dev_tot_nhat") is not None:
        ppl = math.exp(min(ket_qua["loss_dev_tot_nhat"], 20.0))
        ghi(f"| **loss dev tốt nhất** | **{ket_qua['loss_dev_tot_nhat']:.4f}** |")
        ghi(f"| perplexity dev | {ppl:.2f} |")
    ghi(f"| dừng sớm | {'có' if ket_qua['dung_som'] else 'không'} |")
    ghi(f"| thời gian chạy | {giay_chay / 60:.1f} phút |")
    ghi("")

    ghi("## Tiêu chí XONG KHI\n")
    khong_nan = ket_qua.get("loss_train_cuoi") is not None
    ghi(f"- **TASK 13** — huấn luyện liên tục không xuất hiện NaN: "
        f"**{'ĐẠT' if khong_nan else 'CHƯA'}** "
        f"({ket_qua['so_buoc_da_chay']:,} bước liên tục)")
    ghi(f"- **TASK 12** — thời gian đồng bộ checkpoint dưới 5% tổng thời gian: "
        f"**{ty_le_dong_bo:.2f}%** → **{'ĐẠT' if ty_le_dong_bo < 5 else 'CHƯA ĐẠT'}**\n")

    ghi("## File sinh ra\n")
    ghi(f"- `{ket_qua['checkpoint_moi_nhat']}`")
    ghi(f"- `{duong_dan_log / 'metrics.csv'}` — dùng để vẽ đường loss cho báo cáo")
    ghi("- `results/cau_hinh_da_gop.yaml` — cấu hình đầy đủ của lượt chạy này\n")

    ghi("## Cách chạy lại y hệt\n")
    ghi("```bash")
    ghi(f"python scripts/train.py "
        f"--config {cfg.thi_nghiem.get('duong_dan_config', 'configs/base.yaml')} "
        f"--seed {cfg.thi_nghiem.seed}")
    ghi("```\n")
    ghi("Toàn bộ trọng số, log và cấu hình đều nằm trên Hugging Face Hub, nên người khác")
    ghi("chỉ cần thêm `--tiep-tuc` là chạy tiếp đúng chỗ cũ, không cần máy của người trước.")

    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    duong_dan.write_text("\n".join(dong) + "\n", encoding="utf-8")
    print(f"[train] Đã ghi báo cáo: {duong_dan}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None, help="ghi đè thi_nghiem.seed")
    parser.add_argument("--smoke", action="store_true",
                        help="chạy vài bước để kiểm Run All không crash, KHÔNG đẩy lên Hub")
    parser.add_argument("--tiep-tuc", action="store_true",
                        help="kéo checkpoint mới nhất từ Hub rồi chạy tiếp đúng chỗ cũ")
    parser.add_argument("--so-buoc", type=int, default=None,
                        help="ghi đè huan_luyen.so_buoc_toi_da")
    parser.add_argument("--repo-hub", default=None,
                        help="ghi đè thi_nghiem.hub_repo; để trống là không đồng bộ")
    args = parser.parse_args()

    cfg = nap_config(args.config)
    if args.seed is not None:
        cfg["thi_nghiem"]["seed"] = args.seed
    cfg["thi_nghiem"]["duong_dan_config"] = args.config
    dat_seed(cfg.thi_nghiem.seed, cfg.thi_nghiem.deterministic)

    from nmt.model.transformer import TransformerNMT
    from nmt.training.checkpoint import CHE_DO_SMOKE, CHE_DO_THAT
    from nmt.training.trainer import Trainer

    che_do = CHE_DO_SMOKE if args.smoke else CHE_DO_THAT

    # Repo Hub: smoke test KHÔNG đồng bộ; và nếu cấu hình mẫu còn để chỗ trống
    # thì cũng bỏ qua, thay vì cố đẩy lên một repo tên "<ten-tai-khoan-hf>/...".
    repo_hub = args.repo_hub or cfg.thi_nghiem.get("hub_repo")
    if args.smoke or not repo_hub or "<" in str(repo_hub):
        if not args.smoke and repo_hub and "<" in str(repo_hub):
            print(f"[train] thi_nghiem.hub_repo vẫn là chỗ trống ({repo_hub!r}) nên bỏ qua "
                  "đồng bộ Hub. Sửa configs/base.yaml hoặc truyền --repo-hub.")
        repo_hub = None

    ten_chay = f"{cfg.thi_nghiem.ten}_seed{cfg.thi_nghiem.seed}"
    if args.smoke:
        ten_chay = "smoke_" + ten_chay
    duong_dan_log = GOC / cfg.thi_nghiem.thu_muc_log / ten_chay
    thu_muc_checkpoint = GOC / cfg.thi_nghiem.thu_muc_checkpoint / ten_chay

    print("=" * 70)
    print(f"TASK 15 — HUẤN LUYỆN · {ten_chay} · chế độ {che_do}")
    print(f"Hub: {repo_hub or '(không đồng bộ)'}")
    print("=" * 70)

    train_loader, dev_loader, _ = nap_du_lieu(cfg, args.smoke)
    model = TransformerNMT(cfg)
    print(f"[train] Số tham số: {model.dem_tham_so()['tong']:,}")

    so_buoc = args.so_buoc or (SO_BUOC_SMOKE if args.smoke else None)
    logger = BoGhiLog(duong_dan_log)
    trainer = Trainer(
        cfg, model, train_loader, dev_loader, logger,
        che_do=che_do,
        thu_muc_checkpoint=thu_muc_checkpoint,
        repo_hub=repo_hub,
        so_buoc_toi_da=so_buoc,
    )

    # --- Chạy tiếp từ Hub nếu được yêu cầu -----------------------------------
    if args.tiep_tuc:
        from nmt.training.hub_sync import tai_checkpoint_moi_nhat

        duong_dan_cuc_bo = thu_muc_checkpoint / "moi_nhat.pt"
        if not duong_dan_cuc_bo.exists() and repo_hub:
            print("[train] Không có checkpoint ở đĩa, thử kéo từ Hugging Face Hub...")
            tai_ve = tai_checkpoint_moi_nhat(repo_hub, thu_muc_checkpoint)
            if tai_ve is not None:
                duong_dan_cuc_bo = tai_ve

        if duong_dan_cuc_bo.exists():
            trainer.tiep_tuc_tu(duong_dan_cuc_bo)
        else:
            print("[train] Không tìm thấy checkpoint nào — huấn luyện từ đầu.")

    # --- Huấn luyện -----------------------------------------------------------
    luu_config(cfg, GOC / "results" / "cau_hinh_da_gop.yaml")
    bat_dau = time.perf_counter()
    ket_qua = trainer.train()
    giay_chay = time.perf_counter() - bat_dau

    # --- Đẩy nốt cấu hình và tokenizer lên Hub --------------------------------
    giay_dong_bo = 0.0
    if repo_hub:
        from nmt.training.hub_sync import (
            TEN_TOKENIZER, TIEN_TO_CAU_HINH, dam_bao_repo, day_len_hub,
        )

        moc = time.perf_counter()
        dam_bao_repo(repo_hub)
        day_len_hub(GOC / "results" / "cau_hinh_da_gop.yaml", repo_hub,
                    f"{TIEN_TO_CAU_HINH}/cau_hinh_da_gop.yaml", che_do=che_do)
        # Tokenizer phải đi cùng checkpoint. Mỗi người tự chạy train_tokenizer.py
        # sẽ ra file khác nhau nếu lệch phiên bản thư viện, và khi đó checkpoint
        # nạp vào dữ liệu của người khác ra rác vì token ID không khớp.
        day_len_hub(Path(cfg.du_lieu.tokenizer), repo_hub, TEN_TOKENIZER, che_do=che_do)
        giay_dong_bo = time.perf_counter() - moc

    viet_bao_cao(
        GOC / "docs" / "bao_cao_huan_luyen.md",
        cfg, model, ket_qua, giay_chay, giay_dong_bo, args.smoke, duong_dan_log,
    )

    print("\n" + "=" * 70)
    print(f"XONG sau {giay_chay / 60:.1f} phút · {ket_qua['so_buoc_da_chay']:,} bước")
    if ket_qua.get("loss_dev_tot_nhat") is not None:
        print(f"loss dev tốt nhất: {ket_qua['loss_dev_tot_nhat']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
