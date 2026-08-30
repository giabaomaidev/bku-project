"""TASK 13 — Learning rate scheduler.  Người làm: Quân.  [Training • Bắt buộc]

Theo nhận xét 1 của mentor: Warmup KHÔNG còn là thành phần bắt buộc.
Mặc định trong base.yaml là `scheduler: co_dinh`. Ablation A2 (TASK 18) quyết
định có giữ warmup hay không.

Lưu ý cho A2: bài On Layer Normalization in the Transformer Architecture
(arxiv 2002.04745) cho thấy Pre-Norm vốn đã làm giảm nhu cầu warmup. Nên chênh
lệch giữa bật và tắt có thể rất nhỏ. Đó là điều bài báo dự đoán, không phải bug.
Ngược lại, Post-Norm (A6) gần như bắt buộc phải có warmup.
"""

from __future__ import annotations


class WarmupScheduler:
    """Learning rate tăng dần trong so_buoc_warmup bước đầu rồi giảm dần.

    Mục đích của phần tăng dần là tránh làm hỏng ma trận trọng số lúc mới khởi tạo.

    Công thức Noam của bài 2017:

        lr(t) = d_model^(-0.5) * min( t^(-0.5),  t * warmup^(-1.5) )

    Hai nhánh gặp nhau đúng tại t = warmup nên đường learning rate liền lạc:
    trước đó `t * warmup^(-1.5)` nhỏ hơn nên lr tăng tuyến tính theo t; sau đó
    `t^(-0.5)` nhỏ hơn nên lr giảm dần theo nghịch đảo căn bậc hai.

    Đếm bước bắt đầu từ 1. Để t = 0 thì `t^(-0.5)` chia cho 0 và lr thành vô cùng
    ngay bước đầu tiên — chương trình KHÔNG báo lỗi, chỉ thấy loss ra NaN.

    Lưu ý: công thức này TỰ quyết định learning rate nên nó BỎ QUA khóa
    `toi_uu.learning_rate` trong YAML. Đó là đúng thiết kế của Noam chứ không phải
    quên đọc cấu hình — đỉnh của đường lr nằm ở d_model^(-0.5) * warmup^(-0.5).
    """

    def __init__(self, optimizer, d_model: int, so_buoc_warmup: int = 4000) -> None:
        if d_model <= 0:
            raise ValueError(f"d_model phải dương, nhận được {d_model}")
        if so_buoc_warmup <= 0:
            raise ValueError(
                f"so_buoc_warmup phải dương, nhận được {so_buoc_warmup}. "
                "Muốn tắt warmup thì đặt toi_uu.scheduler = co_dinh."
            )

        self.optimizer = optimizer
        self.d_model = d_model
        self.so_buoc_warmup = so_buoc_warmup

        # Đặt luôn learning rate của bước 1, để lần cập nhật đầu tiên đã dùng
        # đúng giá trị chứ không dùng lr mặc định còn sót của optimizer.
        self._buoc = 1
        self._ap_dung()

    def _tinh_lr(self, buoc: int) -> float:
        buoc = max(buoc, 1)
        return (self.d_model ** -0.5) * min(
            buoc ** -0.5,
            buoc * (self.so_buoc_warmup ** -1.5),
        )

    def _ap_dung(self) -> None:
        lr = self._tinh_lr(self._buoc)
        for nhom in self.optimizer.param_groups:
            nhom["lr"] = lr

    def step(self) -> None:
        self._buoc += 1
        self._ap_dung()

    def learning_rate_hien_tai(self) -> float:
        """Learning rate đang áp dụng — trainer ghi vào log để vẽ đường lr."""
        return self._tinh_lr(self._buoc)

    def state_dict(self) -> dict:
        return {"buoc": self._buoc}

    def load_state_dict(self, state: dict) -> None:
        # Thiếu phần khôi phục này thì sau khi resume, learning rate nhảy về đầu
        # đường warmup và đường loss gãy một nhát ngay chỗ nối.
        self._buoc = state["buoc"]
        self._ap_dung()

    def __repr__(self) -> str:
        return (
            f"WarmupScheduler(d_model={self.d_model}, so_buoc_warmup={self.so_buoc_warmup}, "
            f"buoc={self._buoc}, lr={self.learning_rate_hien_tai():.3e})"
        )


class SchedulerCoDinh:
    """Learning rate giữ nguyên — mặc định, và là nhánh đối chứng của A2."""

    def __init__(self, optimizer, learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate phải dương, nhận được {learning_rate}")

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        for nhom in self.optimizer.param_groups:
            nhom["lr"] = learning_rate

    def step(self) -> None:
        pass

    def learning_rate_hien_tai(self) -> float:
        return self.learning_rate

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def __repr__(self) -> str:
        return f"SchedulerCoDinh(learning_rate={self.learning_rate:.3e})"


def tao_scheduler(cfg, optimizer):
    """Factory đọc `toi_uu.scheduler` từ YAML."""
    kieu = cfg.toi_uu.scheduler
    if kieu == "warmup":
        return WarmupScheduler(optimizer, cfg.mo_hinh.d_model, cfg.toi_uu.so_buoc_warmup)
    if kieu == "co_dinh":
        return SchedulerCoDinh(optimizer, cfg.toi_uu.learning_rate)
    raise ValueError(f"toi_uu.scheduler khong hop le: {kieu} (phai la warmup hoac co_dinh)")
