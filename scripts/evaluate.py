"""TASK 16 — Chấm BLEU và chrF++ bằng sacrebleu trên dev và test.  Người làm: My.

Ghi lại NGUYÊN VĂN chuỗi chữ ký của sacrebleu vào results/diem_chinh.csv.
Ghi rõ hướng dịch là từ Anh sang Việt, tên tập test và số câu.
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
    parser.add_argument("--checkpoint", type=str, help="Đường dẫn tới checkpoint (.pt) đã huấn luyện")
    parser.add_argument("--split", type=str, choices=["dev", "test"], default="test", help="Tập đánh giá (dev hoặc test)")
    args = parser.parse_args()

    cfg = nap_config(args.config)
    if args.seed is not None:
        cfg["thi_nghiem"]["seed"] = args.seed
    dat_seed(cfg.thi_nghiem.seed, cfg.thi_nghiem.deterministic)

    import torch
    import pandas as pd
    import os
    from datetime import datetime
    from nmt.data import nap_tokenizer, DuLieuSongNgu, tao_dataloader, PAD_ID, BOS_ID, EOS_ID
    from nmt.model.transformer import TransformerNMT
    from nmt.inference.search import greedy_search
    from nmt.eval.metrics import cham_bleu, cham_chrf

    tokenizer = nap_tokenizer(cfg.du_lieu.tokenizer)
    
    split_path = cfg.du_lieu.dev if args.split == "dev" else cfg.du_lieu.test
    duong_dan_en = f"{split_path}.en"
    duong_dan_vi = f"{split_path}.vi"
    
    dataset = DuLieuSongNgu(duong_dan_en, duong_dan_vi, tokenizer)
    loader = tao_dataloader(dataset, so_token_moi_batch=cfg.du_lieu.so_token_moi_batch, gom_theo_do_dai=False, tron=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = TransformerNMT(cfg).to(device)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        print(f"Đã nạp checkpoint từ {args.checkpoint}")
    else:
        print("CẢNH BÁO: Không có checkpoint, mô hình đang chạy bằng trọng số ngẫu nhiên!")

    model.eval()
    
    du_doan = []
    tham_chieu = []
    
    print(f"Bắt đầu dịch tập {args.split} ({len(dataset)} câu)...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            src_ids = batch["src_ids"].to(device)
            src_mask = batch["src_mask"].to(device)
            labels = batch["labels"].to(device)
            
            preds_ids = greedy_search(model, src_ids, src_mask, BOS_ID, EOS_ID, do_dai_toi_da=cfg.sinh_cau.do_dai_toi_da_khi_dich)
            
            for p_ids, l_ids in zip(preds_ids, labels):
                p_list = p_ids.tolist()
                if BOS_ID in p_list: p_list.remove(BOS_ID)
                if EOS_ID in p_list: p_list = p_list[:p_list.index(EOS_ID)]
                du_doan.append(tokenizer.decode(p_list, skip_special_tokens=True))
                
                l_list = l_ids.tolist()
                if EOS_ID in l_list: l_list = l_list[:l_list.index(EOS_ID)]
                tham_chieu.append(tokenizer.decode(l_list, skip_special_tokens=True))

            if (i + 1) % 10 == 0:
                print(f" Đã dịch xong batch {i + 1}/{len(loader)}")

    bleu, bleu_sig = cham_bleu(du_doan, tham_chieu)
    chrf, chrf_sig = cham_chrf(du_doan, tham_chieu)
    
    print(f"\nKẾT QUẢ TRÊN TẬP {args.split.upper()}:")
    print(f"BLEU:   {bleu:.2f}  (Chữ ký: {bleu_sig})")
    print(f"chrF++: {chrf:.2f}  (Chữ ký: {chrf_sig})")
    
    os.makedirs("results", exist_ok=True)
    csv_path = "results/diem_chinh.csv"
    
    row = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Tập test": args.split,
        "Hướng dịch": "En-Vi",
        "Số câu": len(dataset),
        "BLEU": bleu,
        "chrF++": chrf,
        "Chữ ký BLEU": bleu_sig,
        "Chữ ký chrF++": chrf_sig,
        "Checkpoint": args.checkpoint or "Random weights"
    }])
    
    # Nối thêm (append) nếu file đã tồn tại
    if os.path.exists(csv_path):
        row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        row.to_csv(csv_path, index=False)
        
    print(f"Đã lưu kết quả vào {csv_path}")


if __name__ == "__main__":
    main()
