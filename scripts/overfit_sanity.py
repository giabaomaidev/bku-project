"""TASK 10 — Bài test học thuộc 50 câu — CỔNG CHẶN cuối Tuần 2.  Người làm: My.

Lấy 50 cặp câu từ tập train, TẮT dropout, TẮT label smoothing, learning rate 1e-4,
huấn luyện tới khi loss xuống thật thấp, rồi dịch lại đúng 50 câu đó và chấm BLEU.

Yêu cầu: loss dưới 0,05 trong tối đa 500 bước, và BLEU trên 90.

Nếu đường loss chững lại ở mức 2 hay 3 thì CHẮC CHẮN có lỗi, thường là:
    - mask sai chiều
    - quên chia cho căn bậc hai của d_k
    - nhầm trục khi tính softmax
    - lệch một vị trí khi ghép cặp đầu vào và đầu ra của decoder

Nếu loss về 0 mà BLEU vẫn thấp thì lỗi nằm ở khâu giải mã hoặc ghép lại chữ,
không phải ở mô hình.

Sinh ra: results/overfit_loss.png, results/overfit_vi_du.csv
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

    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import pandas as pd
    from nmt.data import nap_tokenizer, DuLieuSongNgu, tao_dataloader, PAD_ID, BOS_ID, EOS_ID
    from nmt.model.transformer import TransformerNMT
    from nmt.inference.search import greedy_search
    from nmt.eval.metrics import cham_bleu
    import os

    cfg.mo_hinh.dropout = 0.0
    cfg.huan_luyen.nhan_tron = 0.0

    tokenizer = nap_tokenizer(cfg.du_lieu.tokenizer)
    dataset = DuLieuSongNgu(cfg.du_lieu.train + ".en", cfg.du_lieu.train + ".vi", tokenizer)
    dataset._src = dataset._src[:50]
    dataset._tgt = dataset._tgt[:50]
    loader = tao_dataloader(dataset, so_token_moi_batch=100000, gom_theo_do_dai=False, tron=False)

    batch = next(iter(loader))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    src_ids = batch["src_ids"].to(device)
    tgt_input = batch["tgt_input"].to(device)
    labels = batch["labels"].to(device)
    src_mask = batch["src_mask"].to(device)
    tgt_mask = batch["tgt_mask"].to(device)

    model = TransformerNMT(cfg).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    losses = []
    print("Bắt đầu overfit 50 câu...")
    for step in range(1, 501):
        optimizer.zero_grad()
        logits = model(src_ids, tgt_input, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.4f}")
        
        if loss.item() < 0.05:
            print(f"Đạt loss < 0.05 tại bước {step}!")
            break

    os.makedirs("results", exist_ok=True)
    plt.plot(losses)
    plt.title("Overfit 50 câu - Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("results/overfit_loss.png")
    plt.close()

    model.eval()
    print("Dịch 50 câu bằng Greedy Search...")
    with torch.no_grad():
        preds_ids = greedy_search(model, src_ids, src_mask, BOS_ID, EOS_ID, do_dai_toi_da=128)
    
    du_doan = []
    tham_chieu = []
    
    for p_ids, l_ids in zip(preds_ids, labels):
        p_list = p_ids.tolist()
        if BOS_ID in p_list: p_list.remove(BOS_ID)
        if EOS_ID in p_list: p_list = p_list[:p_list.index(EOS_ID)]
        pred_text = tokenizer.decode(p_list, skip_special_tokens=True)
        
        l_list = l_ids.tolist()
        if EOS_ID in l_list: l_list = l_list[:l_list.index(EOS_ID)]
        ref_text = tokenizer.decode(l_list, skip_special_tokens=True)
        
        du_doan.append(pred_text)
        tham_chieu.append(ref_text)

    bleu, signature = cham_bleu(du_doan, tham_chieu)
    print(f"BLEU overfit: {bleu:.2f} (Signature: {signature})")
    
    df = pd.DataFrame({"Tham chiếu": tham_chieu, "Dự đoán": du_doan})
    df.to_csv("results/overfit_vi_du.csv", index=False)


if __name__ == "__main__":
    main()
