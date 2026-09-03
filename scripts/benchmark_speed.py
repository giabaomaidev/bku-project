"""TASK 11 — Đo tốc độ thật trên Kaggle TRƯỚC KHI cam kết phạm vi.  Người làm: Quân.

Đây là việc trả lời trực tiếp nhận xét 4 của mentor — PHẢI KHẢO SÁT BẰNG SỐ LIỆU,
không lấy nguyên cấu hình từ bài báo — và phải làm ĐẦU TIÊN của Phase 3.

Quét bốn trục, mỗi cấu hình chạy CÙNG MỘT ngân sách bước:
    số lớp    4 và 6
    số head   4 (mỗi head 128 chiều) và 8 (mỗi head 64 chiều)
    d_ff      quanh 688
    dropout   quanh 0.3
Đo trên T4: số giây mỗi bước, số token mỗi giây, bộ nhớ GPU đỉnh.
Thử vài mức so_token_moi_batch để chọn mức tốt nhất, rồi nhân ra tổng số giờ GPU
cần cho toàn bộ quá trình huấn luyện IWSLT.

Sinh ra:
    results/khao_sat_cau_hinh.csv
    results/ngan_sach_tinh_toan.csv
    docs/ngan_sach_tinh_toan.md      <- báo cáo TỰ ĐIỀN, không phải gõ tay

Dùng:
    python scripts/benchmark_speed.py --config configs/base.yaml
    python scripts/benchmark_speed.py --config configs/base.yaml --nhanh   # smoke test
"""

from __future__ import annotations

import argparse
import csv
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

from nmt.utils import dat_seed, luu_config, nap_config

GOC = Path(__file__).resolve().parents[1]

# Ngân sách GPU của nhóm. Kaggle bản free cấp 30 giờ GPU mỗi tuần cho mỗi tài
# khoản; chỉ tính 1 tài khoản để ước lượng cho an toàn.
NGAN_SACH_GIO_MOI_TUAN = 30.0

# Số bước bỏ đi ở đầu mỗi phép đo. Lượt chạy đầu trên GPU còn phải cấp phát bộ
# nhớ, biên dịch nhân CUDA và nạp dữ liệu nên chậm hơn hẳn phần sau. Tính cả mấy
# bước đó vào thì mọi con số bị thổi phồng và bảng khảo sát mất ý nghĩa.
SO_BUOC_KHOI_DONG = 5

# Kích thước tập con để đo. Đo trên toàn bộ 131k câu vừa lâu vừa không cần thiết,
# vì tốc độ mỗi bước không phụ thuộc số câu có trong tập.
SO_CAU_DE_DO = 10_000

# Tổng số câu của tập train sau khi lọc ở TASK 02 — dùng để quy đổi số batch đo
# trên tập con ra số batch của cả epoch.
TONG_SO_CAU_TRAIN = 131_400


def _lay_cau_hinh_quet(nhanh: bool) -> list[dict]:
    """Danh sách cấu hình cần quét.

    Trục số lớp và số head là hai câu hỏi mentor hỏi thẳng nên quét đủ 2x2.
    Trục d_ff và dropout chỉ quét quanh giá trị đang chốt để biết độ nhạy, không
    quét dày — chúng ảnh hưởng tốc độ ít hơn hẳn hai trục kia.
    """
    if nhanh:
        return [{"so_lop": 2, "so_head": 4, "d_ff": 688, "dropout": 0.3}]

    cau_hinh = []
    for so_lop in (4, 6):
        for so_head in (4, 8):
            cau_hinh.append(
                {"so_lop": so_lop, "so_head": so_head, "d_ff": 688, "dropout": 0.3}
            )
    for d_ff in (512, 1024):
        cau_hinh.append({"so_lop": 6, "so_head": 8, "d_ff": d_ff, "dropout": 0.3})
    for dropout in (0.1, 0.5):
        cau_hinh.append({"so_lop": 6, "so_head": 8, "d_ff": 688, "dropout": dropout})
    return cau_hinh


def _nap_du_lieu(cfg, nhanh: bool):
    """Trả về (dataset, có_dữ_liệu_thật). Thiếu dữ liệu thì dùng batch giả.

    Đo bằng dữ liệu giả vẫn cho số giây mỗi bước và bộ nhớ đỉnh đúng, vì hai con
    số đó chỉ phụ thuộc hình dạng tensor. Nhưng phải NÓI RÕ trong báo cáo, không
    được để người đọc tưởng đây là số đo trên dữ liệu thật.
    """
    from nmt.data import DuLieuSongNgu, nap_tokenizer

    duong_dan_en = Path(str(cfg.du_lieu.train) + ".en")
    duong_dan_vi = Path(str(cfg.du_lieu.train) + ".vi")
    duong_dan_tokenizer = Path(cfg.du_lieu.tokenizer)

    if not (duong_dan_en.exists() and duong_dan_vi.exists() and duong_dan_tokenizer.exists()):
        print("[benchmark] KHÔNG thấy dữ liệu đã xử lý hoặc tokenizer.")
        print(f"[benchmark]   {duong_dan_en}  {'có' if duong_dan_en.exists() else 'THIẾU'}")
        print(f"[benchmark]   {duong_dan_tokenizer}  "
              f"{'có' if duong_dan_tokenizer.exists() else 'THIẾU'}")
        print("[benchmark] Chuyển sang batch giả — số giây mỗi bước và bộ nhớ vẫn đúng, "
              "nhưng báo cáo sẽ ghi rõ là đo trên dữ liệu giả.")
        return None, False

    tokenizer = nap_tokenizer(str(duong_dan_tokenizer))
    dataset = DuLieuSongNgu(duong_dan_en, duong_dan_vi, tokenizer, cfg.du_lieu.do_dai_toi_da)
    gioi_han = 200 if nhanh else SO_CAU_DE_DO
    dataset._src = dataset._src[:gioi_han]
    dataset._tgt = dataset._tgt[:gioi_han]
    return dataset, True


def _batch_gia(so_token_moi_batch: int, vocab_size: int, do_dai: int = 40):
    """Sinh một batch giả có tổng số ô xấp xỉ so_token_moi_batch."""
    import torch

    so_cau = max(1, so_token_moi_batch // do_dai)
    nhan = torch.randint(4, vocab_size, (so_cau, do_dai))
    return {
        "src_ids": torch.randint(4, vocab_size, (so_cau, do_dai)),
        "tgt_input": torch.randint(4, vocab_size, (so_cau, do_dai)),
        "labels": nhan,
        "src_mask": torch.ones(so_cau, 1, 1, do_dai, dtype=torch.bool),
        "tgt_mask": torch.ones(so_cau, 1, do_dai, do_dai, dtype=torch.bool).tril(),
    }


def do_mot_cau_hinh(cfg, dataset, so_token_moi_batch: int, so_buoc_do: int) -> dict:
    """Đo một cấu hình. Trả về dict số liệu, hoặc lý do thất bại nếu tràn bộ nhớ."""
    import torch

    from nmt.data import tao_dataloader
    from nmt.model.transformer import TransformerNMT
    from nmt.training.loss import tao_loss
    from nmt.training.trainer import _co_the_dung_fp16, chon_thiet_bi

    thiet_bi = chon_thiet_bi()
    dung_fp16 = _co_the_dung_fp16(cfg, thiet_bi)

    if thiet_bi.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = TransformerNMT(cfg).to(thiet_bi)
    model.train()
    so_tham_so = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.toi_uu.learning_rate)
    criterion = tao_loss(cfg, vocab_size=cfg.du_lieu.vocab_size, pad_id=model.pad_id)
    scaler = torch.amp.GradScaler("cuda", enabled=dung_fp16)

    loader = None
    if dataset is not None:
        from nmt.utils import sinh_generator

        loader = tao_dataloader(
            dataset,
            so_token_moi_batch=so_token_moi_batch,
            so_worker=0,
            generator=sinh_generator(cfg.thi_nghiem.seed),
        )

        def lay_batch():
            while True:
                yield from loader
    else:
        batch_co_dinh = _batch_gia(so_token_moi_batch, cfg.du_lieu.vocab_size)

        def lay_batch():
            while True:
                yield batch_co_dinh

    luong = lay_batch()

    def mot_buoc() -> int:
        batch = next(luong)
        batch = {ten: gt.to(thiet_bi, non_blocking=True) for ten, gt in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=thiet_bi.type, dtype=torch.float16, enabled=dung_fp16):
            logits = model(batch["src_ids"], batch["tgt_input"],
                           batch["src_mask"], batch["tgt_mask"])
            loss = criterion(logits, batch["labels"])
        scaler.scale(loss).backward()
        if dung_fp16:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.toi_uu.cat_gradient_norm)
        scaler.step(optimizer)
        scaler.update()
        return int((batch["labels"] != model.pad_id).sum())

    try:
        for _ in range(SO_BUOC_KHOI_DONG):
            mot_buoc()

        if thiet_bi.type == "cuda":
            torch.cuda.synchronize()
        bat_dau = time.perf_counter()
        tong_token = 0
        for _ in range(so_buoc_do):
            tong_token += mot_buoc()
        if thiet_bi.type == "cuda":
            torch.cuda.synchronize()
        giay_troi = time.perf_counter() - bat_dau

    except torch.cuda.OutOfMemoryError:
        # Tràn bộ nhớ là một KẾT QUẢ chứ không phải sự cố. Ghi lại rồi đi tiếp,
        # vì đúng cái ngưỡng này mới trả lời được "T4 chịu được mức batch nào".
        del model, optimizer
        torch.cuda.empty_cache()
        return {"loi": "tràn bộ nhớ GPU", "so_tham_so": so_tham_so}

    bo_nho_dinh_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if thiet_bi.type == "cuda" else 0.0
    )
    so_batch = len(loader) if loader is not None else 0

    del model, optimizer
    if thiet_bi.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "giay_moi_buoc": giay_troi / so_buoc_do,
        "token_moi_giay": tong_token / giay_troi,
        "bo_nho_dinh_mb": bo_nho_dinh_mb,
        "so_tham_so": so_tham_so,
        "so_batch_moi_epoch": so_batch,
    }


def _ghi_csv(duong_dan: Path, cac_hang: list[dict]) -> None:
    if not cac_hang:
        return
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    # Gộp khóa của mọi hàng, vì hàng bị tràn bộ nhớ thiếu vài cột so với hàng đạt.
    cot: list[str] = []
    for hang in cac_hang:
        for khoa in hang:
            if khoa not in cot:
                cot.append(khoa)
    with duong_dan.open("w", encoding="utf-8", newline="") as f:
        ghi = csv.DictWriter(f, fieldnames=cot, extrasaction="ignore")
        ghi.writeheader()
        ghi.writerows(cac_hang)


def _ten_gpu() -> str:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU (không có GPU)"


def viet_bao_cao(
    duong_dan: Path,
    cfg,
    khao_sat: list[dict],
    ngan_sach: list[dict],
    co_du_lieu_that: bool,
    so_buoc_do: int,
) -> None:
    """Ghi docs/ngan_sach_tinh_toan.md từ số đo THẬT.

    Viết bằng code thay vì gõ tay để con số trong báo cáo luôn khớp với lần chạy
    gần nhất. Gõ tay thì sửa cấu hình xong quên cập nhật báo cáo là chuyện sớm
    muộn, và đó đúng kiểu sai mà mentor bắt được ngay.
    """
    dat = [h for h in khao_sat if "loi" not in h]
    nhanh_nhat = min(dat, key=lambda h: h["giay_moi_buoc"]) if dat else None
    ngan_sach_dat = [h for h in ngan_sach if "loi" not in h]
    batch_tot_nhat = (
        max(ngan_sach_dat, key=lambda h: h["token_moi_giay"]) if ngan_sach_dat else None
    )

    dong: list[str] = []
    ghi = dong.append

    ghi("# Ngân sách tính toán — TASK 11\n")
    ghi("> File này do `scripts/benchmark_speed.py` TỰ SINH từ số đo thật.")
    ghi("> Đừng sửa tay: chạy lại script là mọi con số được cập nhật cùng lúc.\n")

    ghi("## Điều kiện đo\n")
    ghi(f"- Cấu hình: `{cfg.thi_nghiem.get('duong_dan_config', 'configs/base.yaml')}` "
        f"· thí nghiệm `{cfg.thi_nghiem.ten}`")
    ghi(f"- GPU: **{_ten_gpu()}**")
    ghi(f"- Độ chính xác: {'fp16 + GradScaler' if cfg.toi_uu.do_chinh_xac_hon_hop else 'fp32'}")
    ghi(f"- Số bước đo mỗi cấu hình: {so_buoc_do} (đã bỏ {SO_BUOC_KHOI_DONG} bước khởi động đầu)")
    ghi(f"- Nguồn dữ liệu: {'tập con IWSLT thật' if co_du_lieu_that else '**batch giả**'}")
    if not co_du_lieu_that:
        ghi("  - Chưa chạy `prepare_data.py` nên số liệu đo trên batch giả. Số giây mỗi")
        ghi("    bước và bộ nhớ đỉnh vẫn đúng vì chỉ phụ thuộc hình dạng tensor, nhưng")
        ghi("    **số token mỗi giây thì không đại diện** cho dữ liệu thật.")
    ghi(f"- Seed: {cfg.thi_nghiem.seed}\n")

    ghi("## Bảng khảo sát cấu hình\n")
    ghi("Trả lời trực tiếp nhận xét 4 của mentor: chốt cấu hình bằng SỐ ĐO, không lấy")
    ghi("nguyên từ bài báo.\n")
    ghi("| số lớp | số head | chiều mỗi head | d_ff | dropout | tham số | giây/bước "
        "| token/giây | VRAM đỉnh (MB) |")
    ghi("|---|---|---|---|---|---|---|---|---|")
    for hang in khao_sat:
        chieu_moi_head = cfg.mo_hinh.d_model // hang["so_head"]
        if "loi" in hang:
            ghi(f"| {hang['so_lop']} | {hang['so_head']} | {chieu_moi_head} | {hang['d_ff']} "
                f"| {hang['dropout']} | {hang['so_tham_so']:,} | — | — | **{hang['loi']}** |")
        else:
            ghi(f"| {hang['so_lop']} | {hang['so_head']} | {chieu_moi_head} | {hang['d_ff']} "
                f"| {hang['dropout']} | {hang['so_tham_so']:,} | {hang['giay_moi_buoc']:.4f} "
                f"| {hang['token_moi_giay']:,.0f} | {hang['bo_nho_dinh_mb']:.0f} |")
    ghi("")

    ghi("## Bảng ngân sách theo số token mỗi batch\n")
    ghi("Lưu ý cách đọc: **giây/micro-batch** là thời gian một lượt forward+backward,")
    ghi("còn **giây/bước** là thời gian một bước optimizer đầy đủ, tức đã nhân với")
    ghi(f"`so_buoc_cong_don_gradient` = {cfg.toi_uu.so_buoc_cong_don_gradient}. Cột tổng giờ")
    ghi("GPU tính theo cột thứ hai, vì `huan_luyen.so_buoc_toi_da` đếm bước optimizer.\n")
    ghi("| số token/batch | giây/micro-batch | giây/bước | token/giây | VRAM đỉnh (MB) "
        "| bước/epoch | giờ/epoch | số epoch | tổng giờ GPU | ngân sách | khả thi |")
    ghi("|---|---|---|---|---|---|---|---|---|---|---|")
    for hang in ngan_sach:
        if "loi" in hang:
            ghi(f"| {hang['so_token_moi_batch']} | — | — | — | — | — | — | — | — "
                f"| {NGAN_SACH_GIO_MOI_TUAN:.0f} h | **{hang['loi']}** |")
            continue
        kha_thi = "CÓ" if hang["tong_gio_gpu"] <= NGAN_SACH_GIO_MOI_TUAN else "KHÔNG"
        ghi(f"| {hang['so_token_moi_batch']} | {hang['giay_moi_buoc']:.4f} "
            f"| {hang['giay_moi_buoc_optimizer']:.4f} "
            f"| {hang['token_moi_giay']:,.0f} | {hang['bo_nho_dinh_mb']:.0f} "
            f"| {hang['buoc_moi_epoch']:,} | {hang['gio_moi_epoch']:.2f} "
            f"| {hang['so_epoch']} | {hang['tong_gio_gpu']:.2f} "
            f"| {NGAN_SACH_GIO_MOI_TUAN:.0f} h | **{kha_thi}** |")
    ghi("")

    ghi("## Kết luận\n")
    if batch_tot_nhat is not None:
        kha_thi = batch_tot_nhat["tong_gio_gpu"] <= NGAN_SACH_GIO_MOI_TUAN
        ghi(f"Chạy **{batch_tot_nhat['so_epoch']} epoch** trên IWSLT với "
            f"`so_token_moi_batch = {batch_tot_nhat['so_token_moi_batch']}` tốn khoảng "
            f"**{batch_tot_nhat['tong_gio_gpu']:.2f} giờ GPU**. Ngân sách của nhóm là "
            f"{NGAN_SACH_GIO_MOI_TUAN:.0f} giờ mỗi tuần nên phương án này "
            f"**{'khả thi' if kha_thi else 'CHƯA khả thi — phải giảm số bước hoặc cỡ mô hình'}**.\n")
    if nhanh_nhat is not None:
        ghi(f"Cấu hình nhanh nhất trong bảng khảo sát: **{nhanh_nhat['so_lop']} lớp, "
            f"{nhanh_nhat['so_head']} head, d_ff {nhanh_nhat['d_ff']}** — "
            f"{nhanh_nhat['giay_moi_buoc']:.4f} giây mỗi bước, "
            f"{nhanh_nhat['bo_nho_dinh_mb']:.0f} MB VRAM đỉnh.\n")
        ghi("Lưu ý khi đọc bảng: nhanh nhất KHÔNG đồng nghĩa với tốt nhất. Cấu hình ít lớp")
        ghi("chạy nhanh hơn nhưng sức biểu diễn kém hơn, nên con số cuối cùng phải chốt cùng")
        ghi("với BLEU của TASK 15 chứ không chỉ nhìn mỗi tốc độ.\n")

    ghi("## File số liệu gốc\n")
    ghi("- `results/khao_sat_cau_hinh.csv`")
    ghi("- `results/ngan_sach_tinh_toan.csv`")
    ghi("- `results/cau_hinh_benchmark.yaml` — cấu hình đã gộp của lần đo này")

    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    duong_dan.write_text("\n".join(dong) + "\n", encoding="utf-8")
    print(f"[benchmark] Đã ghi báo cáo: {duong_dan}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None, help="ghi đè thi_nghiem.seed")
    parser.add_argument("--so-buoc", type=int, default=30,
                        help="số bước đo mỗi cấu hình, chưa tính bước khởi động")
    parser.add_argument("--nhanh", action="store_true",
                        help="smoke test: một cấu hình, vài bước, chỉ để kiểm không crash")
    args = parser.parse_args()

    cfg = nap_config(args.config)
    if args.seed is not None:
        cfg["thi_nghiem"]["seed"] = args.seed
    dat_seed(cfg.thi_nghiem.seed, cfg.thi_nghiem.deterministic)

    so_buoc_do = 3 if args.nhanh else args.so_buoc
    cac_muc_token = [2048] if args.nhanh else [2048, 4096, 8192]

    dataset, co_du_lieu_that = _nap_du_lieu(cfg, args.nhanh)

    print("=" * 70)
    print(f"TASK 11 — KHẢO SÁT CẤU HÌNH · GPU: {_ten_gpu()}")
    print("=" * 70)

    # --- Quét cấu hình, cùng một ngân sách bước cho mọi cấu hình --------------
    khao_sat: list[dict] = []
    for muc in _lay_cau_hinh_quet(args.nhanh):
        cfg.mo_hinh.so_lop_encoder = muc["so_lop"]
        cfg.mo_hinh.so_lop_decoder = muc["so_lop"]
        cfg.mo_hinh.so_head = muc["so_head"]
        cfg.mo_hinh.d_ff = muc["d_ff"]
        cfg.mo_hinh.dropout = muc["dropout"]

        print(f"\n  {muc['so_lop']} lớp · {muc['so_head']} head · d_ff {muc['d_ff']} "
              f"· dropout {muc['dropout']}", flush=True)
        ket_qua = do_mot_cau_hinh(cfg, dataset, cfg.du_lieu.so_token_moi_batch, so_buoc_do)
        khao_sat.append({**muc, **ket_qua})

        if "loi" in ket_qua:
            print(f"    -> {ket_qua['loi']}")
        else:
            print(f"    -> {ket_qua['giay_moi_buoc']:.4f} giây/bước · "
                  f"{ket_qua['token_moi_giay']:,.0f} token/giây · "
                  f"{ket_qua['bo_nho_dinh_mb']:.0f} MB")

    # --- Quét mức token mỗi batch trên cấu hình gốc ---------------------------
    cfg_goc = nap_config(args.config)
    if args.seed is not None:
        cfg_goc["thi_nghiem"]["seed"] = args.seed
    # Ghi lại chính file cấu hình đã dùng, để báo cáo nói được nó đo cái gì.
    cfg_goc["thi_nghiem"]["duong_dan_config"] = args.config

    print("\n" + "=" * 70)
    print("NGÂN SÁCH THEO SỐ TOKEN MỖI BATCH")
    print("=" * 70)

    ngan_sach: list[dict] = []
    for so_token in cac_muc_token:
        print(f"\n  so_token_moi_batch = {so_token}", flush=True)
        ket_qua = do_mot_cau_hinh(cfg_goc, dataset, so_token, so_buoc_do)

        if "loi" in ket_qua:
            ngan_sach.append({"so_token_moi_batch": so_token, **ket_qua})
            print(f"    -> {ket_qua['loi']}")
            continue

        # Số bước mỗi epoch phải tính trên TOÀN BỘ tập train chứ không phải tập
        # con đang đo. Quên quy đổi lại là ước lượng thấp hơn thực tế cả chục lần.
        so_batch_do_duoc = ket_qua["so_batch_moi_epoch"]
        if co_du_lieu_that and so_batch_do_duoc:
            ty_le = TONG_SO_CAU_TRAIN / min(SO_CAU_DE_DO, TONG_SO_CAU_TRAIN)
            buoc_moi_epoch = int(
                so_batch_do_duoc * ty_le / cfg_goc.toi_uu.so_buoc_cong_don_gradient
            )
        else:
            buoc_moi_epoch = 0

        # HAI CHỮ "BƯỚC" KHÁC NHAU — chỗ này từng tính sai và làm báo cáo lệch 4 lần.
        #
        #   do_mot_cau_hinh đo thời gian mỗi MICRO-BATCH (một lượt forward+backward)
        #   huan_luyen.so_buoc_toi_da đếm số BƯỚC OPTIMIZER
        #
        # Mà mỗi bước optimizer gồm so_buoc_cong_don_gradient micro-batch. Nhân
        # thẳng hai con số với nhau là ước lượng thấp đi đúng bằng hệ số cộng dồn.
        # Bản đầu báo "2,60 giờ GPU" trong khi lượt chạy thật mất hơn 13 tiếng.
        so_cong_don = cfg_goc.toi_uu.so_buoc_cong_don_gradient
        giay_moi_buoc_optimizer = ket_qua["giay_moi_buoc"] * so_cong_don

        gio_moi_epoch = buoc_moi_epoch * giay_moi_buoc_optimizer / 3600
        so_epoch = (
            max(1, round(cfg_goc.huan_luyen.so_buoc_toi_da / buoc_moi_epoch))
            if buoc_moi_epoch else 0
        )
        tong_gio = cfg_goc.huan_luyen.so_buoc_toi_da * giay_moi_buoc_optimizer / 3600

        ngan_sach.append({
            "so_token_moi_batch": so_token,
            **ket_qua,
            "so_buoc_cong_don": so_cong_don,
            "giay_moi_buoc_optimizer": giay_moi_buoc_optimizer,
            "buoc_moi_epoch": buoc_moi_epoch,
            "gio_moi_epoch": gio_moi_epoch,
            "so_epoch": so_epoch,
            "tong_gio_gpu": tong_gio,
        })
        print(f"    -> {ket_qua['giay_moi_buoc']:.4f} giây/bước · "
              f"{ket_qua['bo_nho_dinh_mb']:.0f} MB · tổng {tong_gio:.2f} giờ GPU")

    # --- Ghi kết quả ----------------------------------------------------------
    _ghi_csv(GOC / "results" / "khao_sat_cau_hinh.csv", khao_sat)
    _ghi_csv(GOC / "results" / "ngan_sach_tinh_toan.csv", ngan_sach)
    luu_config(cfg_goc, GOC / "results" / "cau_hinh_benchmark.yaml")
    viet_bao_cao(
        GOC / "docs" / "ngan_sach_tinh_toan.md",
        cfg_goc, khao_sat, ngan_sach, co_du_lieu_that, so_buoc_do,
    )

    print("\n" + "=" * 70)
    print("XONG. Đọc docs/ngan_sach_tinh_toan.md để lấy bảng cho báo cáo.")
    print("=" * 70)


if __name__ == "__main__":
    main()
