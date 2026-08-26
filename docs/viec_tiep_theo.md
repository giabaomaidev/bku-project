# Việc tiếp theo — Quân và Bảo

Cập nhật: 2026-08-26 · `main` đang ở commit `dce1763`

---

## Trạng thái hiện tại

Phase 1 và Phase 2 đã xong và đã merge vào `main`:

| Task | Người | Trạng thái |
|---|---|---|
| 01 Repo & tái lập | Phú | Xong |
| 02 · 03 · 04 Dữ liệu, tokenizer, DataLoader | My | Xong |
| 05 · 06 Attention, RoPE + sin-cos | Quân | Xong |
| 07 · 08 RMSNorm + LayerNorm, SwiGLU + ReLU | Bảo | Xong |
| 09 Ghép Encoder–Decoder | Quân | Xong |
| 10 Bộ 12 bài kiểm tra kiến trúc | My | Xong (Phú sửa 3 lỗi) |
| 16 Greedy Search + BLEU/chrF++ | My | Xong (Phú sửa 3 lỗi) |

Bắt đầu làm:

```bash
git checkout main && git pull
pytest -q          # phải ra: 68 passed, 9 skipped
```

Đỏ ngay từ đầu thì báo Phú, đừng code tiếp.

---

## VIỆC SỐ 0 — Ai có GPU làm trước, cả nhóm đang chờ

**Chạy cổng chặn overfit 50 câu trên Kaggle.** Đây là mốc bắt buộc cuối Tuần 2.
Chưa qua thì **không được** sang TASK 11 hay TASK 15.

```bash
python scripts/prepare_data.py    --config configs/base.yaml
python scripts/train_tokenizer.py --config configs/base.yaml
python scripts/overfit_sanity.py  --config configs/base.yaml
```

Script tự in kết quả và tự thoát khác 0 nếu không đạt:

```
KẾT QUẢ CỔNG CHẶN — bài kiểm tra số 7
  loss cuối : 0.0xxx   (yêu cầu < 0.05)   ĐẠT
  BLEU      : 9x.xx    (yêu cầu > 90)     ĐẠT
CỔNG CHẶN ĐÃ QUA.
```

Chưa ai chạy được bước này vì cả nhóm chưa ai chạy trên GPU. Phú thử trên CPU
nhưng mô hình 48 triệu tham số quá chậm, hơn 20 phút chưa xong một vòng.

Ra `CHƯA QUA` thì script in luôn chẩn đoán nguyên nhân — đọc rồi báo cả nhóm,
đừng tự sửa `src/` một mình vì đó là kiến trúc dùng chung.

Sinh ra: `results/overfit_loss.png` và `results/overfit_vi_du.csv` — hai file này
là hình quan trọng nhất của Phase 2 trong báo cáo.

---

## QUÂN — TASK 11, rồi TASK 13

### TASK 11 — Khảo sát cấu hình & ngân sách GPU  `scripts/benchmark_speed.py`

Trả lời trực tiếp nhận xét 4 của mentor: **phải khảo sát bằng số liệu, không lấy
nguyên cấu hình từ bài báo.**

Quét bốn trục, mỗi cấu hình chạy **cùng một ngân sách bước**:

| Trục | Các mức |
|---|---|
| số lớp | 4 và 6 |
| số head | 4 (mỗi head 128 chiều) và 8 (mỗi head 64 chiều) |
| `d_ff` | quanh 688 |
| dropout | quanh 0.3 |

Đo trên T4: **số giây mỗi bước · số token mỗi giây · bộ nhớ GPU đỉnh**. Thử vài
mức `so_token_moi_batch` để chọn mức tốt nhất, rồi nhân ra tổng số giờ GPU cho
toàn bộ quá trình huấn luyện IWSLT.

**Xong khi:** bảng khảo sát đầy đủ, và cấu hình cuối cùng được chốt **dựa trên số
liệu**. Ghi vào `docs/ngan_sach_tinh_toan.md`, kèm một câu kết luận rõ ràng dạng
*"chạy N epoch tốn khoảng X giờ GPU, ngân sách nhóm là Y giờ nên khả thi"*.

Đây là **mốc chặn đầu Tuần 3** và là căn cứ trả lời mentor về số layer, số head.

### TASK 13 — Training Loop  `src/nmt/training/{trainer,scheduler,loss}.py`

Cần TASK 09 (xong) và TASK 12 của Bảo.

AdamW · cộng dồn gradient · cắt gradient theo norm 1.0 · fp16 + GradScaler.
Warmup Scheduler và Label Smoothing cài ở dạng **TÙY CHỌN, mặc định TẮT** —
theo nhận xét 1 của mentor, hai kỹ thuật này không còn là thành phần bắt buộc.

**Hai chi tiết GradScaler dễ sai, cả hai đều KHÔNG báo lỗi gì:**

1. Phải gọi `scaler.unscale_(optimizer)` **trước** khi cắt gradient theo norm.
   Quên thì đang cắt gradient đã bị nhân hệ số giãn, ngưỡng 1.0 vô nghĩa.
2. Khi cộng dồn gradient thì chia loss cho số bước cộng dồn, và **chỉ** gọi
   `scaler.step()` cùng `scaler.update()` ở bước cuối mỗi chu kỳ.

Thứ tự đúng một chu kỳ:

```python
for i in range(so_buoc_cong_don):
    with autocast(dtype=torch.float16):
        loss = tinh_loss(...) / so_buoc_cong_don
    scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer); scaler.update()
optimizer.zero_grad(set_to_none=True)
scheduler.step()
```

**TUYỆT ĐỐI không gọi `model.half()`.** Dùng `torch.autocast`. Gọi `.half()` sẽ
ép bảng góc quay `cos_cache`/`inv_freq` của RoPE xuống fp16 và làm hỏng đúng cái
bẫy fp16 số 2 mà Quân đã cẩn thận tránh khi viết TASK 06.

**Xong khi:** huấn luyện liên tục 1000 bước không xuất hiện NaN.

---

## BẢO — TASK 12 làm được NGAY, không chờ ai

### TASK 12 — Checkpoint & đồng bộ HF Hub  `src/nmt/training/{checkpoint,hub_sync}.py`

Task này **không phụ thuộc model hay data**, bắt đầu song song với TASK 11 của
Quân được luôn.

Lưu **đầy đủ**, thiếu một món là resume sai mà không báo lỗi:

- trọng số mô hình
- trạng thái optimizer *(thiếu thì momentum AdamW về 0, đường loss gãy một nhát)*
- trạng thái scheduler *(thiếu thì learning rate nhảy về đầu)*
- trạng thái GradScaler
- số bước, số epoch
- bản sao cấu hình đã gộp
- seed và trạng thái RNG *(thiếu thì hai đường loss của TASK 14 không trùng khít)*

**Ghi file an toàn:** ghi ra file tạm rồi mới đổi tên. Phiên Kaggle bị ngắt đúng
lúc đang ghi mà ghi đè trực tiếp thì mất luôn cả checkpoint cũ lẫn mới.

**Bảo mật token:** đọc theo thứ tự Kaggle Secrets → biến môi trường `HF_TOKEN` →
báo lỗi rõ ràng. Không viết token vào code, không commit.

**Đẩy cả thư mục log lên Hub** cùng checkpoint. Thiếu bước này thì kernel chết là
mất log, và TASK 14 không vẽ được đường loss liền mạch qua các lần bị giết.

**Xong khi:** thời gian đồng bộ checkpoint chiếm dưới 5% tổng thời gian huấn luyện.

### TASK 14 — Thí nghiệm giết phiên & phục hồi

Cần TASK 12 và 13 xong. Chạy một lần liên tục, một lần bị giết giữa chừng hai lần
rồi resume, vẽ hai đường loss chồng lên nhau, đánh dấu đường kẻ dọc ở hai chỗ bị giết.

**Xong khi:** hai đường chênh nhau dưới 1% tại cùng một bước.

Đây là **hình có giá trị nhất của cả đồ án** — dùng cho báo cáo, slide và CV.

### TASK 15 — Huấn luyện chính thức  `scripts/train.py`

Chạy với cấu hình Quân đã chốt ở TASK 11. Theo dõi loss dev, dừng sớm khi không
cải thiện, lưu bản tốt nhất lên HF Hub.

**Mốc chặn cuối Tuần 3.** Trễ mốc này thì cắt TASK 19 và thu gọn ablation.

Xong rồi báo My để My làm TASK 16 trên checkpoint thật.

---

## Ba điều cả hai cùng nhớ

**Đừng gọi `model.half()`.** Dùng `torch.autocast` + `GradScaler`.

**fp16 chứ không bf16.** T4 của Kaggle là kiến trúc Turing, compute capability
7.5, không có phần cứng bf16. Đừng dùng `torch.cuda.is_bf16_supported()` để tự
chọn — hàm này tính cả trường hợp giả lập nên trả về `True` ngay trên T4, chạy
được nhưng chậm hơn cả fp32. Muốn kiểm tự động thì dùng
`torch.cuda.get_device_capability()[0] >= 8`.

**Tokenizer phải dùng chung một bản.** Mỗi người tự chạy `train_tokenizer.py` sẽ
ra file khác nhau nếu phiên bản thư viện `tokenizers` lệch. Khi đó checkpoint của
Bảo load vào dữ liệu của My sẽ ra rác vì token ID không khớp. Bảo lo TASK 12 nên
tiện thể **đẩy luôn `artifacts/tokenizer/tokenizer.json` lên HF Hub**, cả nhóm tải
cùng một file về dùng.

---

## Quy trình

Mỗi người một nhánh, không push thẳng vào `main`:

```bash
git checkout main && git pull
git checkout -b feat/training
# ... code ...
pytest -q                  # phải xanh trước khi push
git status                 # không để lọt .pt, checkpoint, data/, token
git commit -m "TASK NN: mo ta"
git push -u origin feat/training
```

Rồi mở PR trên GitHub. Phú review và merge.
