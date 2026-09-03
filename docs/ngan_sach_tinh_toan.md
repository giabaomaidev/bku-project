# Ngân sách tính toán — TASK 11

> File này do `scripts/benchmark_speed.py` TỰ SINH từ số đo thật.
> Đừng sửa tay: chạy lại script là mọi con số được cập nhật cùng lúc.

## Điều kiện đo

- Cấu hình: `configs/base.yaml` · thí nghiệm `iwslt_base_v1`
- GPU: **Tesla T4**
- Độ chính xác: fp16 + GradScaler
- Số bước đo mỗi cấu hình: 30 (đã bỏ 5 bước khởi động đầu)
- Nguồn dữ liệu: tập con IWSLT thật
- Seed: 42

## Bảng khảo sát cấu hình

Trả lời trực tiếp nhận xét 4 của mentor: chốt cấu hình bằng SỐ ĐO, không lấy
nguyên từ bài báo.

| số lớp | số head | chiều mỗi head | d_ff | dropout | tham số | giây/bước | token/giây | VRAM đỉnh (MB) |
|---|---|---|---|---|---|---|---|---|
| 4 | 4 | 128 | 688 | 0.3 | 37,432,320 | 0.2728 | 17,814 | 6959 |
| 4 | 8 | 64 | 688 | 0.3 | 37,432,320 | 0.2839 | 17,114 | 7025 |
| 6 | 4 | 128 | 688 | 0.3 | 47,955,968 | 0.3572 | 13,604 | 8016 |
| 6 | 8 | 64 | 688 | 0.3 | 47,955,968 | 0.3710 | 13,100 | 8112 |
| 6 | 8 | 64 | 512 | 0.3 | 44,711,936 | 0.3655 | 13,295 | 7924 |
| 6 | 8 | 64 | 1024 | 0.3 | 54,149,120 | 0.3951 | 12,298 | 8435 |
| 6 | 8 | 64 | 688 | 0.1 | 47,955,968 | 0.3812 | 12,746 | 8112 |
| 6 | 8 | 64 | 688 | 0.5 | 47,955,968 | 0.3765 | 12,907 | 8112 |

## Bảng ngân sách theo số token mỗi batch

Lưu ý cách đọc: **giây/micro-batch** là thời gian một lượt forward+backward,
còn **giây/bước** là thời gian một bước optimizer đầy đủ, tức đã nhân với
`so_buoc_cong_don_gradient` = 4. Cột tổng giờ
GPU tính theo cột thứ hai, vì `huan_luyen.so_buoc_toi_da` đếm bước optimizer.

| số token/batch | giây/micro-batch | giây/bước | token/giây | VRAM đỉnh (MB) | bước/epoch | giờ/epoch | số epoch | tổng giờ GPU | ngân sách | khả thi |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 0.2005 | 0.8019 | 11,708 | 4411 | 351 | 0.08 | 171 | 13.37 | 30 h | **CÓ** |
| 4096 | 0.3750 | 1.5000 | 12,958 | 8112 | 177 | 0.07 | 339 | 25.00 | 30 h | **CÓ** |
| 8192 | — | — | — | — | — | — | — | — | 30 h | **tràn bộ nhớ GPU** |

## Kết luận

Chạy **339 epoch** trên IWSLT với `so_token_moi_batch = 4096` tốn khoảng **25.00 giờ GPU**. Ngân sách của nhóm là 30 giờ mỗi tuần nên phương án này **khả thi**.

Cấu hình nhanh nhất trong bảng khảo sát: **4 lớp, 4 head, d_ff 688** — 0.2728 giây mỗi bước, 6959 MB VRAM đỉnh.

Lưu ý khi đọc bảng: nhanh nhất KHÔNG đồng nghĩa với tốt nhất. Cấu hình ít lớp
chạy nhanh hơn nhưng sức biểu diễn kém hơn, nên con số cuối cùng phải chốt cùng
với BLEU của TASK 15 chứ không chỉ nhìn mỗi tốc độ.

## File số liệu gốc

- `results/khao_sat_cau_hinh.csv`
- `results/ngan_sach_tinh_toan.csv`
- `results/cau_hinh_benchmark.yaml` — cấu hình đã gộp của lần đo này
