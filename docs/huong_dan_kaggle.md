# Hướng dẫn chạy trên Kaggle — Phase 3 (TASK 11 → 15)

Tài liệu này trả lời đúng ba câu: **upload cái gì**, **gắn vào đâu**, **bấm gì**.

---

## Bước 1 — Tạo file zip để upload

Chạy ở máy cá nhân, trong thư mục `bku-project`:

```bash
git archive --format=zip -o ../bku-project.zip HEAD
```

Lệnh này đóng gói **đúng những gì đã commit**, nên tự động bỏ qua `data/`,
`artifacts/`, `results/` và mọi file rác. File ra khoảng vài trăm KB.

> Đừng nén cả thư mục bằng chuột phải → Send to → Compressed. Cách đó gói luôn
> `.git/`, `__pycache__/` và checkpoint nếu có — vừa nặng vừa dễ lọt file không
> nên công khai.

Không dùng git thì nén tay, và **chỉ lấy 5 thứ này**:

```
configs/    scripts/    src/    tests/    pyproject.toml
```

---

## Bước 2 — Upload lên Kaggle Datasets

1. Vào **kaggle.com/datasets** → **New Dataset**
2. Kéo `bku-project.zip` vào
3. Đặt tên: `bku-project` (nhớ đúng tên này để lần sau cập nhật cho dễ)
4. **Create**

> Kaggle tự giải nén file zip, nên trong `/kaggle/input` sẽ thấy cây thư mục chứ
> không phải một file `.zip`.

**Lần sau sửa code thì đừng tạo dataset mới** — vào dataset cũ → **New Version**
→ upload zip mới. Tạo mới mỗi lần sẽ đẻ ra 5 dataset trùng tên và không ai biết
cái nào mới nhất.

---

## Bước 3 — Tạo notebook trên Kaggle

Có hai cách, chọn một:

### Cách A — Import notebook có sẵn (khuyên dùng)

1. **kaggle.com/code** → **New Notebook**
2. **File → Import Notebook**
3. Chọn file **`notebooks/02_kaggle_train.ipynb`** trong repo

> Đây là notebook **duy nhất** cần import. Hai file `00_kiem_tra_du_lieu.ipynb`
> và `01_giai_thich_kien_truc.ipynb` là tài liệu đọc ở máy, không chạy trên Kaggle.

### Cách B — Tạo trắng rồi dán

Tạo notebook trắng rồi copy từng cell từ `notebooks/02_kaggle_train.ipynb`.
Chậm hơn và dễ sót cell, chỉ dùng khi cách A hỏng.

---

## Bước 4 — Gắn bốn thứ vào notebook

Mở notebook trên Kaggle, panel bên phải:

| # | Mục | Chọn gì |
|---|---|---|
| 1 | **Accelerator** | **GPU T4 x2** |
| 2 | **Internet** | **On** — mặc định TẮT; không bật thì vừa không tải được dữ liệu vừa không đẩy được lên Hugging Face |
| 3 | **Input** → Add Input → Datasets | dataset **`bku-project`** vừa upload ở bước 2 |
| 4 | **Add-ons → Secrets** | thêm secret tên **`HF_TOKEN`** |

### Về `HF_TOKEN` — chỗ này hay sai nhất

1. Lên **huggingface.co/settings/tokens** → New token → chọn loại **Write**
   (token **Read** đẩy file lên sẽ bị lỗi 403)
2. Kaggle → **Add-ons → Secrets → Add a new secret**
   - Label: **`HF_TOKEN`** (đúng chính tả, viết hoa)
   - Value: dán token
3. **BẬT CÔNG TẮC cho đúng notebook này.** Tạo secret thôi chưa đủ — mỗi notebook
   có công tắc riêng.
4. Nếu phiên đã khởi động **trước** khi gắn secret thì phải **Run → Restart session**.

> Hai bước 3 và 4 chính là mục 1.4 trong `Sưu tập lỗi.md`. Lần trước nhóm mất khá
> lâu chỉ vì tưởng đã gắn token xong rồi.

---

## Bước 5 — Sửa một dòng trong notebook

Cell 3 có dòng này, đổi thành tài khoản Hugging Face của cậu:

```python
REPO_HUB = "giabaomaidev/envi-nmt-scratch-transformer"
```

Repo chưa tồn tại cũng không sao, script tự tạo và để ở chế độ riêng tư.

---

## Bước 6 — Run All

**Run → Run All**. Thứ tự notebook sẽ chạy:

| Cell | Việc | Thời gian ước tính |
|---|---|---|
| 1 | Cài thư viện | ~1 phút |
| 2–3 | Dò đường dẫn, kiểm GPU và token | vài giây |
| 4 | Tải và làm sạch dữ liệu, train tokenizer | ~10 phút (lần đầu) |
| 5 | **SMOKE TEST** — chạy ngắn cả pipeline | ~3 phút |
| 6 | Cổng chặn overfit 50 câu | ~5 phút |
| 7 | TASK 11 — khảo sát cấu hình | ~15 phút |
| 8 | TASK 15 — huấn luyện thật | nhiều giờ |
| 9 | TASK 14 — thí nghiệm giết phiên | ~10 phút |
| 10 | Đẩy kết quả lên Hub | ~2 phút |

**Cell 5 đỏ thì dừng lại sửa, đừng chạy tiếp.** Smoke test sinh ra chính là để
bắt lỗi trước khi đốt giờ GPU.

---

## Bước 7 — SAVE VERSION, đừng quên

`/kaggle/working` **chỉ tồn tại trong phiên**. Đóng trình duyệt, hết quota, hoặc
hết giờ là mất sạch.

- Lượt chạy dài: dùng **Save Version → Save & Run All (Commit)** thay vì chạy tương tác
- Chạy tương tác thì trước khi đóng máy phải bấm **Save Version**

> Mục 1.5 của `Sưu tập lỗi.md`: lần mất đầu tiên tốn 3 giờ GPU.

Kể cả quên thì cũng chưa mất hết — checkpoint đã được đẩy lên Hugging Face sau
mỗi mốc, chạy lại notebook là kéo về chạy tiếp được.

---

## Chạy tiếp khi hết quota

Máy cậu hết quota thì bạn khác chạy tiếp được, **không cần máy của cậu**:

1. Người đó mở notebook (Copy & Edit, hoặc import lại)
2. Gắn **`HF_TOKEN` của chính họ**, có quyền Write vào repo đó
3. Run All

Cell 8 tự phát hiện đã có checkpoint và thêm cờ `--tiep-tuc`, script sẽ kéo
`checkpoints/moi_nhat.pt` từ Hub về rồi chạy tiếp **đúng bước đang dở**, giữ
nguyên trạng thái optimizer, scheduler, GradScaler và cả RNG.

Giám khảo muốn dựng lại từ đầu thì cũng chỉ cần repo Hub: trong đó có trọng số,
log, cấu hình đã gộp và tokenizer — đủ để tái lập với `seed = 42`.

---

## Bảng tra lỗi nhanh

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `KHÔNG TÌM THẤY repo` | chưa Add Input, hoặc dataset thiếu `src/nmt` | Add Input → Datasets → `bku-project`; xem danh sách thư mục notebook in ra |
| `Không có HF_TOKEN` | chưa bật công tắc secret cho notebook này | Add-ons → Secrets → bật công tắc → Restart session |
| Lỗi 401 / 403 khi đẩy file | token loại **Read** | tạo lại token loại **Write** |
| `ConnectionError`, `resolve` | Internet đang tắt | Settings → Internet → On |
| `Read-only file system` | đang ghi vào `/kaggle/input` | notebook đã tự chép repo sang `/kaggle/working`; vẫn lỗi thì kiểm lại cell 2 |
| `OutOfMemoryError` | batch quá lớn | giảm `du_lieu.so_token_moi_batch` trong `configs/base.yaml` (xem bảng TASK 11) |
| Không có GPU | quên chọn accelerator | panel phải → Accelerator → GPU T4 x2 |

---

## Tóm tắt một dòng

Upload **`bku-project.zip`** vào **Datasets** rồi Add Input;
import **`notebooks/02_kaggle_train.ipynb`** vào **Code**;
bật **GPU T4 + Internet**; thêm secret **`HF_TOKEN`** loại Write và bật công tắc;
sửa `REPO_HUB`; **Run All**; xong nhớ **Save Version**.
