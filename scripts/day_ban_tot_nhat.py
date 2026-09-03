"""So mọi checkpoint đang có rồi đẩy BẢN TỐT NHẤT THẬT lên Hugging Face Hub.

VÌ SAO CẦN:
    Lượt chạy thứ hai resume từ moi_nhat.pt, mà file đó lưu loss_dev = None, nên
    mốc "tốt nhất" bị reset về vô cùng. Lần đánh giá đầu của phiên mới — dù TỆ HƠN
    — vẫn được coi là tốt nhất và ghi đè tot_nhat.pt trên Hub.

    Số thật đo được:
        lượt 1  bước  6.000  loss_dev 2,2364   <- tốt nhất thật
        lượt 2  bước 12.000  loss_dev 2,3866   <- đang nằm trên Hub
        lượt 2  bước 17.000  loss_dev 2,5630   (dừng sớm ở đây)

    Lỗi gốc đã sửa trong trainer.py, nhưng bản sai vẫn nằm trên Hub nên cần script
    này để đặt lại cho đúng.

DÙNG LẠI ĐƯỢC CHO ABLATION: TASK 17 chạy 12 lượt, mỗi lượt nhiều phiên. Chạy script
này sau mỗi thí nghiệm là chắc chắn bản trên Hub đúng là bản tốt nhất.

Chạy:
    set HF_TOKEN=hf_...        (Windows)   /   export HF_TOKEN=hf_...  (Linux)
    python scripts/day_ban_tot_nhat.py --repo-hub mgbao/envi-nmt-scratch-transformer
    python scripts/day_ban_tot_nhat.py --repo-hub ... --chi-xem     # chỉ so, không đẩy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

from nmt.training.checkpoint import CHE_DO_THAT, doc_thong_tin

GOC = Path(__file__).resolve().parents[1]

# Nơi thường chứa output tải từ Kaggle về. Thêm thư mục khác thì truyền --tim-trong.
THU_MUC_MAC_DINH = ["OUTPUT_KAGGEL", "output_kaggle_huggingface", "artifacts"]

MODEL_CARD = """---
language: [en, vi]
tags: [translation, transformer, from-scratch, pytorch]
license: mit
---

# ENVI-NMT — Transformer En→Vi tự viết từ đầu

Đồ án môn học. **Toàn bộ kiến trúc do nhóm tự cài bằng PyTorch thuần** — không dùng
`nn.Transformer`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`,
`nn.LayerNorm` hay `nn.RMSNorm`.

| | |
|---|---|
| Kiến trúc | Transformer Encoder–Decoder, {so_tham_so} tham số |
| Chuẩn hóa | RMSNorm, Pre-Norm |
| Feed-forward | SwiGLU (d_ff 688) |
| Mã hóa vị trí | RoPE (chỉ cho self-attention) |
| Dữ liệu | IWSLT 2015 En-Vi, {so_cau} cặp câu |
| Tokenizer | BPE 32k dùng chung En+Vi |
| Seed | 42 |

## ⚠️ ĐÂY KHÔNG PHẢI MODEL CỦA `transformers`

Không dùng được `AutoModel.from_pretrained(...)`. Kiến trúc là mã tự viết của nhóm
nên phải có `src/nmt/` mới dựng lại được mô hình.

## Cách tải và dùng

```bash
git clone https://github.com/giabaomaidev/bku-project
cd bku-project
pip install -r requirements.txt
```

```python
from huggingface_hub import hf_hub_download
import sys; sys.path.insert(0, "src")

from nmt.utils import nap_config
from nmt.model.transformer import TransformerNMT
from nmt.training.checkpoint import nap_checkpoint
from nmt.data import nap_tokenizer

REPO = "{repo_id}"

# Tokenizer PHẢI lấy đúng bản này. Tự train lại sẽ ra token ID khác và mô hình
# nạp vào cho ra rác mà không báo lỗi gì.
duong_dan_tok = hf_hub_download(REPO, "artifacts/tokenizer/tokenizer.json")
duong_dan_ck  = hf_hub_download(REPO, "{ten_tren_hub}")

cfg = nap_config("configs/base.yaml")
model = TransformerNMT(cfg)
thong_tin = nap_checkpoint(duong_dan_ck, model, map_location="cpu")
model.eval()

print(thong_tin)   # buoc, epoch, loss_dev, che_do
```

Dịch thử:

```python
import torch
from nmt.inference.search import greedy_search

tok = nap_tokenizer(duong_dan_tok)
ids = torch.tensor([tok.encode("I love machine translation.").ids])
mask = torch.ones(1, 1, 1, ids.size(1), dtype=torch.bool)
ket_qua = greedy_search(model, ids, mask, bos_id=2, eos_id=3, do_dai_toi_da=128)
print(tok.decode(ket_qua[0].tolist(), skip_special_tokens=True))
```

## Huấn luyện lại từ đầu

```bash
python scripts/prepare_data.py    --config configs/base.yaml
python scripts/train_tokenizer.py --config configs/base.yaml
python scripts/train.py --config configs/base.yaml --seed 42 --tu-dau
```

## Bố cục repo này

```
checkpoints/<tên lượt chạy>/moi_nhat.pt    bản mới nhất, để chạy tiếp
checkpoints/<tên lượt chạy>/tot_nhat.pt    bản tốt nhất theo loss dev
logs/<tên lượt chạy>/metrics.csv           đường loss
configs/<tên lượt chạy>.yaml               cấu hình đã gộp của lượt đó
artifacts/tokenizer/tokenizer.json         DÙNG CHUNG cho mọi lượt
smoke/...                                  smoke test, KHÔNG phải kết quả thật
```
"""


def tim_checkpoint(cac_thu_muc: list[Path]) -> list[tuple[Path, dict]]:
    """Quét mọi .pt, giữ lại bản chế độ 'that' CÓ loss_dev để so được với nhau."""
    tim_thay = []
    for thu_muc in cac_thu_muc:
        if not thu_muc.exists():
            continue
        for f in sorted(thu_muc.rglob("*.pt")):
            try:
                t = doc_thong_tin(f)
            except Exception as loi:
                print(f"  bỏ (đọc hỏng): {f.name} — {type(loi).__name__}")
                continue
            tim_thay.append((f, t))
    return tim_thay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-hub", required=True)
    parser.add_argument("--ten-chay", default="iwslt_base_v1_seed42",
                        help="tên lượt chạy, quyết định đường dẫn trên Hub")
    parser.add_argument("--tim-trong", nargs="*", default=None,
                        help="thư mục cần quét, mặc định quét các thư mục output quen thuộc")
    parser.add_argument("--chi-xem", action="store_true",
                        help="chỉ so sánh và in ra, KHÔNG đẩy gì lên Hub")
    args = parser.parse_args()

    thu_muc = [GOC / t for t in (args.tim_trong or THU_MUC_MAC_DINH)]
    print("Quét:", ", ".join(str(t.name) for t in thu_muc), "\n")

    tat_ca = tim_checkpoint(thu_muc)
    if not tat_ca:
        raise SystemExit("Không tìm thấy checkpoint nào.")

    print(f"{'chế độ':<7} {'bước':>7} {'epoch':>6} {'loss_dev':>10}  đường dẫn")
    print("-" * 78)
    for f, t in sorted(tat_ca, key=lambda x: (x[1]["loss_dev"] is None, x[1]["loss_dev"] or 0)):
        loss = f"{t['loss_dev']:.4f}" if t["loss_dev"] is not None else "—"
        print(f"{t['che_do']:<7} {t['buoc']:>7,} {t['epoch']:>6} {loss:>10}  "
              f"{f.relative_to(GOC)}")

    # Chỉ so bản chế độ THẬT và CÓ loss_dev. moi_nhat.pt luôn có loss_dev = None
    # nên không so được, và bản smoke thì không phải kết quả thật.
    ung_vien = [
        (f, t) for f, t in tat_ca
        if t["che_do"] == CHE_DO_THAT and t["loss_dev"] is not None
    ]
    if not ung_vien:
        raise SystemExit(
            "\nKhông có checkpoint nào vừa ở chế độ 'that' vừa có loss_dev để so."
        )

    tot_nhat, thong_tin = min(ung_vien, key=lambda x: x[1]["loss_dev"])
    print(f"\n{'=' * 78}")
    print(f"TỐT NHẤT: bước {thong_tin['buoc']:,} · epoch {thong_tin['epoch']} · "
          f"loss_dev {thong_tin['loss_dev']:.4f}")
    print(f"          {tot_nhat.relative_to(GOC)}")
    print("=" * 78)

    if args.chi_xem:
        print("\n--chi-xem: dừng ở đây, không đẩy gì lên Hub.")
        return

    ten_tren_hub = f"checkpoints/{args.ten_chay}/tot_nhat.pt"

    from nmt.training.hub_sync import doc_token
    from huggingface_hub import HfApi

    token = doc_token()
    api = HfApi()
    api.create_repo(repo_id=args.repo_hub, token=token, private=True, exist_ok=True)

    print(f"\nĐang đẩy -> {args.repo_hub}/{ten_tren_hub} "
          f"({tot_nhat.stat().st_size / 1024 ** 2:.0f} MB)...", flush=True)
    api.upload_file(
        path_or_fileobj=str(tot_nhat),
        path_in_repo=ten_tren_hub,
        repo_id=args.repo_hub,
        token=token,
        commit_message=(
            f"đặt lại bản tốt nhất THẬT: bước {thong_tin['buoc']}, "
            f"loss_dev {thong_tin['loss_dev']:.4f}"
        ),
    )
    print("Đã đẩy checkpoint.")

    # Model card — để người khác biết cách dùng, vì đây KHÔNG phải model của
    # transformers nên AutoModel.from_pretrained() không chạy được.
    noi_dung = MODEL_CARD.format(
        so_tham_so="47.955.968",
        so_cau="131.339",
        repo_id=args.repo_hub,
        ten_tren_hub=ten_tren_hub,
    )
    duong_dan_card = GOC / "results" / "README_hub.md"
    duong_dan_card.parent.mkdir(parents=True, exist_ok=True)
    duong_dan_card.write_text(noi_dung, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(duong_dan_card),
        path_in_repo="README.md",
        repo_id=args.repo_hub,
        token=token,
        commit_message="model card: cách tải và dùng",
    )
    print("Đã đẩy model card (README.md).")

    print(f"\nXem tại: https://huggingface.co/{args.repo_hub}")


if __name__ == "__main__":
    main()
