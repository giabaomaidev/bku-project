"""TASK 14 — Thí nghiệm giết phiên và phục hồi.  Người làm: Bảo.

Xong khi: hai đường loss chênh nhau dưới 1 phần trăm tại cùng một bước.

Hình hai đường loss chồng lên nhau là HÌNH CÓ GIÁ TRỊ NHẤT của cả đồ án — dùng
cho báo cáo, slide và cả CV — vì nó chứng minh cơ chế phục hồi hoạt động thật
chứ không chỉ nói suông.

Cách làm:
    Lượt A  chạy liền một mạch N bước, ghi lại loss từng bước.
    Lượt B  chạy N bước nhưng BỊ GIẾT ở bước N/3 và 2N/3. Mỗi lần bị giết thì
            vứt bỏ toàn bộ đối tượng trong bộ nhớ, dựng lại từ số 0, nạp
            checkpoint rồi chạy tiếp — đúng như khi Kaggle ngắt phiên thật.
    So sánh loss tại từng bước của hai lượt.

Muốn hai đường trùng khít thì checkpoint phải giữ đủ: trọng số, trạng thái
optimizer, scheduler, GradScaler, VÀ trạng thái RNG. Thiếu bất kỳ món nào thì hai
đường tách nhau dần mà không có lỗi nào báo ra — chính vì vậy mới cần bài này.

Sinh ra:
    results/thi_nghiem_phuc_hoi.png    <- hình cho báo cáo
    results/thi_nghiem_phuc_hoi.csv
    docs/thi_nghiem_phuc_hoi.md        <- báo cáo TỰ ĐIỀN

Dùng:
    python scripts/thi_nghiem_phuc_hoi.py --config configs/base.yaml --so-buoc 60
    python scripts/thi_nghiem_phuc_hoi.py --nhanh      # smoke test
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

# Chạy được ngay cả khi chưa `pip install -e .` (tiện khi làm trên Kaggle).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Console Windows mặc định dùng bảng mã cp1252, in chữ tiếng Việt ra là vỡ chữ
# hoặc UnicodeEncodeError. Ép về UTF-8 để cả nhóm đọc được log giống nhau.
for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

from nmt.utils import dat_seed, nap_config

GOC = Path(__file__).resolve().parents[1]

# Ngưỡng của tiêu chí XONG KHI. Để một chỗ duy nhất vì cả script lẫn báo cáo đều
# dẫn về đây.
NGUONG_CHENH_LECH_PHAN_TRAM = 1.0

SO_CAU_DE_CHAY = 400


def _nap_hoac_gia_lap_du_lieu(cfg, so_cau: int):
    """Dùng dữ liệu thật nếu có; không thì batch giả.

    Thí nghiệm này kiểm CƠ CHẾ phục hồi chứ không kiểm chất lượng dịch, nên batch
    giả vẫn cho kết luận đúng: hai đường loss phải trùng khít bất kể dữ liệu là gì.
    """
    from nmt.data import DuLieuSongNgu, nap_tokenizer

    en = Path(str(cfg.du_lieu.train) + ".en")
    vi = Path(str(cfg.du_lieu.train) + ".vi")
    tok = Path(cfg.du_lieu.tokenizer)

    if en.exists() and vi.exists() and tok.exists():
        dataset = DuLieuSongNgu(en, vi, nap_tokenizer(str(tok)), cfg.du_lieu.do_dai_toi_da)
        dataset._src = dataset._src[:so_cau]
        dataset._tgt = dataset._tgt[:so_cau]
        return dataset, True

    print("[phuc_hoi] Không thấy dữ liệu thật — dùng batch giả. "
          "Kết luận về cơ chế phục hồi vẫn đúng.")
    return None, False


def _kiem_tra_vocab(dataset, vocab_size: int) -> None:
    """Chặn sớm khi token ID vượt quá số hàng của ma trận embedding.

    Không có chốt này thì lỗi rơi xuống tận nhân CUDA và biểu hiện là:

        IndexKernelUtils.cu:16: vectorized_gather_kernel:
        Assertion `ind >= 0 && ind < ind_dim_size` failed
        Aborted (core dumped)          -> mã thoát 134

    Tra ngược từ dòng đó về nguyên nhân rất mất thời gian, mà tệ hơn là CUDA
    context hỏng luôn nên phải khởi động lại kernel. Kiểm ở đây tốn vài mili giây
    và cho ra một câu tiếng Việt chỉ thẳng chỗ sai.
    """
    if dataset is None:
        return

    id_lon_nhat = max(
        max((max(cau) for cau in dataset._src if cau), default=0),
        max((max(cau) for cau in dataset._tgt if cau), default=0),
    )
    if id_lon_nhat >= vocab_size:
        raise ValueError(
            f"Token ID lớn nhất trong dữ liệu là {id_lon_nhat}, nhưng "
            f"du_lieu.vocab_size chỉ có {vocab_size}.\n"
            "Ma trận embedding không đủ hàng, và nếu để chạy tiếp thì CUDA sẽ "
            "abort với mã 134 chứ không báo gì dễ hiểu.\n"
            "Nguyên nhân thường gặp: thu nhỏ vocab_size cho chạy nhanh trong khi "
            "vẫn nạp dữ liệu thật. vocab_size PHẢI luôn khớp với tokenizer."
        )


class _LoaderCoDinh:
    """Danh sách batch cố định. Thứ tự luôn như nhau nên hai lượt so được với nhau."""

    def __init__(self, cfg, so_batch: int = 8) -> None:
        import torch

        torch.manual_seed(cfg.thi_nghiem.seed)
        v = cfg.du_lieu.vocab_size
        self._cac_batch = []
        for _ in range(so_batch):
            nhan = torch.randint(4, v, (8, 12))
            nhan[:, -1] = 0
            self._cac_batch.append({
                "src_ids": torch.randint(4, v, (8, 14)),
                "tgt_input": torch.randint(4, v, (8, 12)),
                "labels": nhan,
                "src_mask": torch.ones(8, 1, 1, 14, dtype=torch.bool),
                "tgt_mask": torch.ones(8, 1, 12, 12, dtype=torch.bool).tril(),
            })

    def __iter__(self):
        return iter(self._cac_batch)

    def __len__(self) -> int:
        return len(self._cac_batch)


def _tao_trainer(cfg, dataset, thu_muc_ckpt: Path, so_buoc: int):
    """Dựng một Trainer hoàn toàn mới — mô phỏng phiên Kaggle vừa khởi động lại."""
    from nmt.data import tao_dataloader
    from nmt.model.transformer import TransformerNMT
    from nmt.training.checkpoint import CHE_DO_SMOKE
    from nmt.training.trainer import Trainer
    from nmt.utils import sinh_generator

    dat_seed(cfg.thi_nghiem.seed, cfg.thi_nghiem.deterministic)

    if dataset is not None:
        loader = tao_dataloader(
            dataset,
            so_token_moi_batch=cfg.du_lieu.so_token_moi_batch,
            so_worker=0,
            generator=sinh_generator(cfg.thi_nghiem.seed),
        )
    else:
        loader = _LoaderCoDinh(cfg)

    return Trainer(
        cfg, TransformerNMT(cfg), loader, None, logger=None,
        che_do=CHE_DO_SMOKE,           # thí nghiệm nội bộ, không đụng tới Hub
        thu_muc_checkpoint=thu_muc_ckpt,
        repo_hub=None,
        so_buoc_toi_da=so_buoc,
    )


def _chay_lien_tuc(cfg, dataset, thu_muc: Path, so_buoc: int) -> list[float]:
    print(f"\n[lượt A] Chạy liền một mạch {so_buoc} bước...")
    trainer = _tao_trainer(cfg, dataset, thu_muc / "lien_tuc", so_buoc)
    trainer.train()
    lich_su = list(trainer.lich_su_loss)
    del trainer
    gc.collect()
    return lich_su


def _chay_bi_giet(cfg, dataset, thu_muc: Path, so_buoc: int,
                  cac_moc_giet: list[int]) -> list[float]:
    """Chạy nhưng bị giết ở các mốc cho trước, mỗi lần dựng lại từ số 0."""
    lich_su: list[float] = []
    thu_muc_ckpt = thu_muc / "bi_giet"
    checkpoint = thu_muc_ckpt / "moi_nhat.pt"
    cac_chang = [*cac_moc_giet, so_buoc]

    for chi_so, moc in enumerate(cac_chang):
        print(f"\n[lượt B · chặng {chi_so + 1}/{len(cac_chang)}] chạy tới bước {moc}...")
        trainer = _tao_trainer(cfg, dataset, thu_muc_ckpt, moc)

        if chi_so > 0:
            # Đây là chỗ mô phỏng "phiên Kaggle mới": mọi thứ trong RAM đã mất,
            # chỉ còn đúng file checkpoint nằm trên đĩa.
            trainer.tiep_tuc_tu(checkpoint)

        trainer.train()
        lich_su.extend(trainer.lich_su_loss)

        if chi_so < len(cac_chang) - 1:
            print(f"[lượt B] *** GIẾT PHIÊN tại bước {moc} ***")

        # Vứt sạch khỏi bộ nhớ để chặng sau thật sự bắt đầu từ con số 0.
        del trainer
        gc.collect()

    return lich_su


def _ve_hinh(duong_dan: Path, loss_a: list[float], loss_b: list[float],
             cac_moc_giet: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")   # Kaggle không có màn hình, phải chọn backend không cửa sổ
    import matplotlib.pyplot as plt

    buoc = range(1, len(loss_a) + 1)
    fig, (tren, duoi) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    tren.plot(buoc, loss_a, label="Chạy liền một mạch", linewidth=2.4, alpha=0.85)
    tren.plot(buoc, loss_b, label="Bị giết 2 lần rồi chạy tiếp",
              linewidth=1.4, linestyle="--", color="crimson")
    for moc in cac_moc_giet:
        tren.axvline(moc, color="gray", linestyle=":", linewidth=1.6)
        tren.annotate("giết phiên", xy=(moc, max(loss_a)), rotation=90,
                      va="top", ha="right", fontsize=9, color="gray")
    tren.set_ylabel("Loss huấn luyện")
    tren.set_title("TASK 14 — Giết phiên rồi phục hồi: hai đường loss phải trùng khít")
    tren.legend()
    tren.grid(alpha=0.3)

    chenh = [
        abs(a - b) / abs(a) * 100 if a != 0 else 0.0 for a, b in zip(loss_a, loss_b)
    ]
    duoi.plot(buoc, chenh, color="darkorange", linewidth=1.4)
    duoi.axhline(NGUONG_CHENH_LECH_PHAN_TRAM, color="red", linestyle="--", linewidth=1.2,
                 label=f"ngưỡng {NGUONG_CHENH_LECH_PHAN_TRAM}%")
    for moc in cac_moc_giet:
        duoi.axvline(moc, color="gray", linestyle=":", linewidth=1.6)
    duoi.set_xlabel("Bước huấn luyện")
    duoi.set_ylabel("Chênh lệch (%)")
    duoi.legend()
    duoi.grid(alpha=0.3)

    fig.tight_layout()
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(duong_dan, dpi=150)
    plt.close(fig)
    print(f"[phuc_hoi] Đã vẽ hình: {duong_dan}")


def viet_bao_cao(duong_dan: Path, cfg, loss_a: list[float], cac_moc_giet: list[int],
                 co_du_lieu_that: bool, dat: bool, chenh_lon_nhat: float,
                 chenh_trung_binh: float) -> None:
    dong: list[str] = []
    ghi = dong.append

    ghi("# Thí nghiệm giết phiên và phục hồi — TASK 14\n")
    ghi("> File này do `scripts/thi_nghiem_phuc_hoi.py` TỰ SINH từ số đo thật.\n")

    ghi("## Cách làm\n")
    ghi(f"- **Lượt A**: chạy liền một mạch {len(loss_a)} bước.")
    ghi(f"- **Lượt B**: cùng seed, nhưng bị giết ở bước "
        f"{' và '.join(str(m) for m in cac_moc_giet)}. Mỗi lần bị giết thì mọi đối tượng")
    ghi("  trong bộ nhớ bị vứt bỏ, dựng lại từ số 0 rồi nạp checkpoint chạy tiếp — đúng")
    ghi("  như khi Kaggle ngắt phiên thật.")
    ghi(f"- Seed: {cfg.thi_nghiem.seed} · dữ liệu: "
        f"{'tập con IWSLT thật' if co_du_lieu_that else 'batch giả'}")
    ghi(f"- Cấu hình: `{cfg.thi_nghiem.get('duong_dan_config', 'configs/base.yaml')}` "
        f"· thí nghiệm `{cfg.thi_nghiem.ten}`\n")

    ghi("## Kết quả\n")
    ghi("| chỉ số | giá trị |")
    ghi("|---|---|")
    ghi(f"| số bước so sánh | {len(loss_a)} |")
    ghi(f"| chênh lệch trung bình | {chenh_trung_binh:.4f}% |")
    ghi(f"| **chênh lệch lớn nhất** | **{chenh_lon_nhat:.4f}%** |")
    ghi(f"| ngưỡng yêu cầu | {NGUONG_CHENH_LECH_PHAN_TRAM}% |")
    ghi(f"| **kết luận** | **{'ĐẠT' if dat else 'CHƯA ĐẠT'}** |")
    ghi("")

    ghi("![Hai đường loss chồng lên nhau](../results/thi_nghiem_phuc_hoi.png)\n")

    if dat:
        ghi("Hai đường trùng khít nghĩa là checkpoint đã giữ đủ **cả bảy món**: trọng số,")
        ghi("trạng thái optimizer, scheduler, GradScaler, số bước/epoch, cấu hình và trạng")
        ghi("thái RNG. Thiếu bất kỳ món nào thì hai đường sẽ tách dần ra:\n")
        ghi("- thiếu trạng thái optimizer → momentum AdamW về 0, loss giật lên ngay chỗ nối")
        ghi("- thiếu trạng thái scheduler → learning rate nhảy về đầu")
        ghi("- thiếu trạng thái RNG → dropout và thứ tự batch khác đi, hai đường lệch dần\n")
    else:
        ghi("**CHƯA ĐẠT.** Nhìn đồ thị chênh lệch ở nửa dưới của hình để biết hai đường")
        ghi("bắt đầu tách ở đâu:\n")
        ghi("- tách **ngay tại mốc giết phiên** → thiếu trạng thái optimizer hoặc scheduler")
        ghi("- tách **từ từ sau mốc** → thiếu trạng thái RNG, hoặc thứ tự batch sau khi")
        ghi("  resume không khớp (xem `_luong_batch` trong `trainer.py`)\n")

    ghi("## Còn thiếu để nộp\n")
    ghi("- [ ] Ảnh chụp màn hình repo Hugging Face cho thấy các file đã đẩy lên đúng chỗ")
    ghi("- [ ] Tỉ lệ % thời gian tốn cho việc lưu và đẩy checkpoint — lấy ở")
    ghi("      `docs/bao_cao_huan_luyen.md`, mục Tiêu chí XONG KHI (yêu cầu dưới 5%)\n")

    ghi("## File số liệu gốc\n")
    ghi("- `results/thi_nghiem_phuc_hoi.csv`")
    ghi("- `results/thi_nghiem_phuc_hoi.png`")

    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    duong_dan.write_text("\n".join(dong) + "\n", encoding="utf-8")
    print(f"[phuc_hoi] Đã ghi báo cáo: {duong_dan}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--so-buoc", type=int, default=60)
    parser.add_argument("--nhanh", action="store_true",
                        help="smoke test: 9 bước, chỉ để kiểm không crash")
    args = parser.parse_args()

    cfg = nap_config(args.config)
    if args.seed is not None:
        cfg["thi_nghiem"]["seed"] = args.seed
    # Ghi lại chính file cấu hình đã dùng, để báo cáo nói được nó đo cái gì.
    cfg["thi_nghiem"]["duong_dan_config"] = args.config

    so_buoc = 9 if args.nhanh else args.so_buoc
    cac_moc_giet = [so_buoc // 3, so_buoc * 2 // 3]

    # Đánh giá dev không liên quan tới thí nghiệm này; tắt đi cho nhanh và để
    # loại bớt một nguồn ngẫu nhiên.
    cfg.huan_luyen.danh_gia_moi = so_buoc * 10
    cfg.huan_luyen.luu_checkpoint_moi = so_buoc * 10
    cfg.huan_luyen.dung_som_sau = 10_000

    if args.nhanh:
        # Thu nhỏ MÔ HÌNH cho nhanh. Nhưng TUYỆT ĐỐI KHÔNG đụng vào
        # du_lieu.vocab_size — nó phải luôn khớp với tokenizer.
        #
        # Bản đầu có thu nhỏ vocab_size xuống 200, và nó nổ trên Kaggle: hàm nạp
        # dữ liệu ưu tiên dùng dữ liệu THẬT khi có, nên token ID lên tới 32.000
        # trong khi ma trận embedding chỉ còn 200 hàng. Kết quả là CUDA assert
        # "index out of bounds" rồi abort với mã thoát 134.
        #
        # Máy cá nhân không bắt được lỗi này vì ở đó không có dữ liệu, script rơi
        # vào nhánh batch giả và batch giả sinh ID theo đúng vocab_size đã thu nhỏ.
        # Hai môi trường chạy hai nhánh khác nhau — bài học: thu nhỏ cấu hình thì
        # chỉ được đụng vào những khóa KHÔNG ràng buộc với dữ liệu bên ngoài.
        cfg.mo_hinh.d_model = 64
        cfg.mo_hinh.so_head = 4
        cfg.mo_hinh.so_lop_encoder = 1
        cfg.mo_hinh.so_lop_decoder = 1
        cfg.mo_hinh.d_ff = 64

    print("=" * 70)
    print(f"TASK 14 — GIẾT PHIÊN & PHỤC HỒI · {so_buoc} bước · giết tại {cac_moc_giet}")
    print("=" * 70)

    dataset, co_du_lieu_that = _nap_hoac_gia_lap_du_lieu(
        cfg, 100 if args.nhanh else SO_CAU_DE_CHAY
    )
    _kiem_tra_vocab(dataset, cfg.du_lieu.vocab_size)
    thu_muc = GOC / "artifacts" / "checkpoints" / "thi_nghiem_phuc_hoi"

    loss_a = _chay_lien_tuc(cfg, dataset, thu_muc, so_buoc)
    loss_b = _chay_bi_giet(cfg, dataset, thu_muc, so_buoc, cac_moc_giet)

    # Hai lượt phải ra đúng cùng số bước, nếu không thì so lệch chỉ số.
    n = min(len(loss_a), len(loss_b))
    loss_a, loss_b = loss_a[:n], loss_b[:n]

    chenh = [
        abs(a - b) / abs(a) * 100 if a != 0 else 0.0 for a, b in zip(loss_a, loss_b)
    ]
    chenh_lon_nhat = max(chenh) if chenh else 0.0
    chenh_trung_binh = sum(chenh) / len(chenh) if chenh else 0.0
    dat = chenh_lon_nhat < NGUONG_CHENH_LECH_PHAN_TRAM

    duong_dan_csv = GOC / "results" / "thi_nghiem_phuc_hoi.csv"
    duong_dan_csv.parent.mkdir(parents=True, exist_ok=True)
    with duong_dan_csv.open("w", encoding="utf-8", newline="") as f:
        ghi = csv.writer(f)
        ghi.writerow(["buoc", "loss_lien_tuc", "loss_bi_giet", "chenh_lech_phan_tram"])
        for i, (a, b, c) in enumerate(zip(loss_a, loss_b, chenh), start=1):
            ghi.writerow([i, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}"])

    _ve_hinh(GOC / "results" / "thi_nghiem_phuc_hoi.png", loss_a, loss_b, cac_moc_giet)
    viet_bao_cao(
        GOC / "docs" / "thi_nghiem_phuc_hoi.md",
        cfg, loss_a, cac_moc_giet, co_du_lieu_that, dat, chenh_lon_nhat, chenh_trung_binh,
    )

    print("\n" + "=" * 70)
    print("KẾT QUẢ — TASK 14")
    print("=" * 70)
    print(f"  chênh lệch trung bình : {chenh_trung_binh:.4f}%")
    print(f"  chênh lệch lớn nhất   : {chenh_lon_nhat:.4f}%   "
          f"(yêu cầu < {NGUONG_CHENH_LECH_PHAN_TRAM}%)")
    print(f"  kết luận              : {'ĐẠT' if dat else 'CHƯA ĐẠT'}")
    print("=" * 70)

    if not dat:
        print("\nHai đường tách nhau. Nhìn đồ thị chênh lệch ở nửa dưới của hình:")
        print("  - tách NGAY tại mốc giết  -> thiếu trạng thái optimizer hoặc scheduler")
        print("  - tách TỪ TỪ sau mốc      -> thiếu trạng thái RNG, hoặc thứ tự batch")
        print("                               sau khi resume không khớp")
        sys.exit(1)


if __name__ == "__main__":
    main()
