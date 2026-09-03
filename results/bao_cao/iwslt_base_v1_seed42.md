# Báo cáo huấn luyện — TASK 13 + TASK 15

> File này do `scripts/train.py` TỰ SINH sau mỗi lượt chạy.
> Đừng sửa tay: chạy lại là mọi con số được cập nhật cùng lúc.

## Cấu hình lượt chạy

- Thí nghiệm: `iwslt_base_v1`
- Seed: **42** · deterministic: True
- Kiến trúc: 6 lớp encoder + 6 lớp decoder · d_model 512 · 8 head
- Chuẩn hóa: rmsnorm / pre-norm · FFN: swiglu (d_ff 688) · mã hóa vị trí: rope
- Optimizer: adamw lr 0.0007 · scheduler co_dinh · label smoothing 0.0
- fp16: True · cộng dồn gradient 4 · cắt gradient 1.0

## Số tham số

| thành phần | số tham số |
|---|---|
| embedding | 16,384,000 |
| encoder | 12,638,720 |
| decoder | 18,933,248 |
| lop_xuat | 0 |
| tong | 47,955,968 |

## Kết quả

| chỉ số | giá trị |
|---|---|
| số bước đã chạy | 6,000 |
| bước cuối | 17,000 |
| epoch | 98 |
| loss train cuối | 0.9983 |
| **loss dev tốt nhất** | **2.3866** |
| perplexity dev | 10.88 |
| dừng sớm | có |
| thời gian chạy | 144.7 phút |

## Tiêu chí XONG KHI

- **TASK 13** — huấn luyện liên tục không xuất hiện NaN: **ĐẠT** (6,000 bước liên tục)
- **TASK 12** — thời gian đồng bộ checkpoint dưới 5% tổng thời gian: **0.01%** → **ĐẠT**

## File sinh ra

- `/kaggle/working/bku-project/artifacts/checkpoints/iwslt_base_v1_seed42/moi_nhat.pt`
- `/kaggle/working/bku-project/results/logs/iwslt_base_v1_seed42/metrics.csv` — dùng để vẽ đường loss cho báo cáo
- `results/cau_hinh/iwslt_base_v1_seed42.yaml` — cấu hình đầy đủ của lượt chạy này
- `results/bao_cao/iwslt_base_v1_seed42.md` — chính là file này, bản riêng không bị đè
- Trên Hub: `checkpoints/iwslt_base_v1_seed42/`, `logs/iwslt_base_v1_seed42/`, `configs/iwslt_base_v1_seed42.yaml`

## Cách chạy lại y hệt

```bash
python scripts/train.py --config configs/base.yaml --seed 42
```

Toàn bộ trọng số, log và cấu hình đều nằm trên Hugging Face Hub, nên người khác
chỉ cần thêm `--tiep-tuc` là chạy tiếp đúng chỗ cũ, không cần máy của người trước.
