# Thí nghiệm giết phiên và phục hồi — TASK 14

> File này do `scripts/thi_nghiem_phuc_hoi.py` TỰ SINH từ số đo thật.

## Cách làm

- **Lượt A**: chạy liền một mạch 60 bước.
- **Lượt B**: cùng seed, nhưng bị giết ở bước 20 và 40. Mỗi lần bị giết thì mọi đối tượng
  trong bộ nhớ bị vứt bỏ, dựng lại từ số 0 rồi nạp checkpoint chạy tiếp — đúng
  như khi Kaggle ngắt phiên thật.
- Seed: 42 · dữ liệu: tập con IWSLT thật
- Cấu hình: `configs/base.yaml` · thí nghiệm `iwslt_base_v1`

## Kết quả

| chỉ số | giá trị |
|---|---|
| số bước so sánh | 60 |
| chênh lệch trung bình | 0.0000% |
| **chênh lệch lớn nhất** | **0.0000%** |
| ngưỡng yêu cầu | 1.0% |
| **kết luận** | **ĐẠT** |

![Hai đường loss chồng lên nhau](../results/thi_nghiem_phuc_hoi.png)

Hai đường trùng khít nghĩa là checkpoint đã giữ đủ **cả bảy món**: trọng số,
trạng thái optimizer, scheduler, GradScaler, số bước/epoch, cấu hình và trạng
thái RNG. Thiếu bất kỳ món nào thì hai đường sẽ tách dần ra:

- thiếu trạng thái optimizer → momentum AdamW về 0, loss giật lên ngay chỗ nối
- thiếu trạng thái scheduler → learning rate nhảy về đầu
- thiếu trạng thái RNG → dropout và thứ tự batch khác đi, hai đường lệch dần

## Còn thiếu để nộp

- [ ] Ảnh chụp màn hình repo Hugging Face cho thấy các file đã đẩy lên đúng chỗ
- [ ] Tỉ lệ % thời gian tốn cho việc lưu và đẩy checkpoint — lấy ở
      `docs/bao_cao_huan_luyen.md`, mục Tiêu chí XONG KHI (yêu cầu dưới 5%)

## File số liệu gốc

- `results/thi_nghiem_phuc_hoi.csv`
- `results/thi_nghiem_phuc_hoi.png`
