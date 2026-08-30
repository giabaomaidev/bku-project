"""ENVI-NMT — English-Vietnamese Neural Machine Translation.

Transformer hiện đại tự viết từ đầu bằng PyTorch thuần.

QUY TẮC BẤT DI BẤT DỊCH cho toàn bộ package này:
    Trong src/ chỉ được dùng nn.Linear, nn.Embedding, nn.Dropout, nn.Parameter
    và các phép tensor cơ bản (F.softmax, F.silu, F.pad, torch.matmul...).

    KHÔNG dùng: nn.Transformer, nn.TransformerEncoder, nn.MultiheadAttention,
                F.scaled_dot_product_attention, nn.LayerNorm, nn.RMSNorm.

    Các lớp tham chiếu của PyTorch CHỈ được xuất hiện trong tests/ để đối chiếu số.
"""

__version__ = "0.1.0"


# Console Windows mặc định dùng bảng mã cp1252, in chữ tiếng Việt ra là
# UnicodeEncodeError. Sáu file trong scripts/ đều đã tự làm việc này ở đầu file,
# nhưng thư viện còn được gọi từ notebook và từ `python -c`, mà ở đó không ai
# chạy khối đó. Đặt ở đây một lần cho mọi đường vào.
#
# Không phải chuyện thẩm mỹ: trainer.py in tiến độ ở mỗi mốc, nên thiếu dòng này
# là lượt huấn luyện dài chết giữa chừng chỉ vì một câu print.
def _ep_stdout_ve_utf8() -> None:
    import sys

    for luong in (sys.stdout, sys.stderr):
        if hasattr(luong, "reconfigure"):
            try:
                luong.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Luồng đã bị thay bằng thứ không cấu hình lại được (pytest bắt
                # đầu ra, IDE bọc console...). Không phải lỗi chặn.
                pass


_ep_stdout_ve_utf8()
