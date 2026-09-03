"""TASK 13 — Training Loop.  Người làm: Quân.  [Training • Bắt buộc]

Xong khi: huấn luyện liên tục 1000 bước không xuất hiện NaN.

Thành phần: AdamW, cộng dồn gradient, cắt gradient theo norm 1.0,
độ chính xác hỗn hợp fp16 kèm GradScaler.

HAI CHI TIẾT DỄ SAI KHI GHÉP GradScaler — cả hai đều KHÔNG báo lỗi gì:

1. Phải gọi scaler.unscale_(optimizer) TRƯỚC khi cắt gradient theo norm.
   Quên thì đang cắt gradient đã bị nhân hệ số giãn, ngưỡng 1.0 trở nên vô nghĩa.

2. Khi cộng dồn gradient thì chia loss cho số bước cộng dồn, và CHỈ gọi
   scaler.step() cùng scaler.update() ở bước cuối của mỗi chu kỳ.

Thứ tự đúng trong một chu kỳ cộng dồn:
    for i in range(so_buoc_cong_don):
        with autocast(dtype=float16):
            loss = tinh_loss(...) / so_buoc_cong_don
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

TUYỆT ĐỐI KHÔNG GỌI model.half(). Dùng torch.autocast. Gọi .half() sẽ ép bảng
góc quay cos_cache/inv_freq của RoPE xuống fp16 và làm hỏng đúng cái bẫy fp16
số 2 mà TASK 06 đã cẩn thận tránh.

TÁI LẬP CHO TASK 14 — vì sao có `_luong_batch`:
    Muốn hai đường loss của thí nghiệm giết phiên trùng khít thì sau khi resume,
    mô hình phải gặp ĐÚNG thứ tự batch như lượt chạy liền mạch. Nên thứ tự batch
    của mỗi epoch được ghim theo (seed, epoch), và khi resume giữa chừng epoch
    thì tua nhanh qua đúng số batch đã tiêu thụ. Thiếu phần này thì hai đường
    tách nhau dần mà không có lỗi nào báo ra.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from nmt.training.checkpoint import CHE_DO_SMOKE, CHE_DO_THAT, luu_checkpoint, nap_checkpoint
from nmt.training.loss import tao_loss
from nmt.training.scheduler import tao_scheduler

TEN_FILE_MOI_NHAT = "moi_nhat.pt"
TEN_FILE_TOT_NHAT = "tot_nhat.pt"

# TOÀN BỘ cột của file log, khai báo sẵn ở một chỗ.
#
# BoGhiLog chốt header theo DÒNG GHI ĐẦU TIÊN rồi dùng extrasaction="ignore" cho
# mọi dòng sau. Dòng đầu là dòng train (không có loss_dev), nên mọi dòng đánh giá
# về sau bị vứt mất loss_dev và perplexity_dev — im lặng, không lỗi.
#
# Lượt chạy 11.000 bước đầu tiên dính đúng lỗi này: metrics.csv chỉ còn cột train,
# các dòng đánh giá ra thành "11000,,,,". Dừng sớm vẫn đúng vì nó đọc giá trị
# trong bộ nhớ, nhưng báo cáo mất hẳn đường loss dev.
#
# Cách chữa: mọi lần ghi đều truyền ĐỦ danh sách cột này, thiếu thì để None.
CAC_COT_LOG = (
    "loss_train",
    "loss_dev",
    "perplexity_dev",
    "learning_rate",
    "giay_moi_buoc",
    "token_moi_giay",
)


def chon_thiet_bi() -> torch.device:
    """CUDA nếu có, rồi tới MPS của máy Mac, cuối cùng là CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _co_the_dung_fp16(cfg, thiet_bi: torch.device) -> bool:
    """fp16 chỉ bật khi cấu hình yêu cầu VÀ đang thật sự chạy trên CUDA.

    GradScaler chỉ có tác dụng trên CUDA. Bật ở CPU thì PyTorch vừa in cảnh báo
    vừa chạy chậm hơn, mà bài test ở máy cá nhân lại toàn chạy CPU.
    """
    if thiet_bi.type != "cuda":
        return False
    if not cfg.toi_uu.get("do_chinh_xac_hon_hop", False):
        return False
    # Ràng buộc chặn bf16 đã nằm ở nap_config, đây chỉ khẳng định lại.
    return cfg.toi_uu.get("kieu_do_chinh_xac", "fp16") == "fp16"


class Trainer:
    """Vòng huấn luyện đầy đủ: cộng dồn gradient, fp16, checkpoint, đồng bộ Hub.

    Args:
        cfg: cấu hình đã gộp.
        model: TransformerNMT, hoặc bất kỳ nn.Module nào cùng chữ ký forward.
        train_loader, dev_loader: DataLoader sinh từ nmt.data.tao_dataloader.
        logger: BoGhiLog, hoặc None nếu không cần ghi log.
        che_do: CHE_DO_THAT hoặc CHE_DO_SMOKE. Smoke test KHÔNG đẩy gì lên Hub.
        thu_muc_checkpoint: nơi ghi checkpoint. Mặc định lấy từ cấu hình.
        repo_hub: repo Hugging Face để đồng bộ. None thì bỏ qua phần đồng bộ.
        so_buoc_toi_da: ghi đè huan_luyen.so_buoc_toi_da — dùng cho smoke test
            và cho thí nghiệm giết phiên của TASK 14.
    """

    def __init__(
        self,
        cfg,
        model,
        train_loader,
        dev_loader,
        logger=None,
        *,
        che_do: str = CHE_DO_THAT,
        thu_muc_checkpoint: str | Path | None = None,
        repo_hub: str | None = None,
        so_buoc_toi_da: int | None = None,
        gio_toi_da: float | None = None,
        ten_chay: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.che_do = che_do

        # TÊN LƯỢT CHẠY — quyết định đường dẫn trên Hub.
        #
        # Bản đầu viết cứng "checkpoints/moi_nhat.pt", nghĩa là MỌI lượt chạy đều
        # ghi đè lên nhau. Với TASK 17 có 6 thí nghiệm ablation nhân 2 seed là 12
        # lượt cùng bắn vào một chỗ — bảng ablation sẽ toàn số của lượt chạy cuối
        # cùng mà không ai nhận ra. Đúng lại kiểu hỏng im lặng của mục 1.8.
        self.ten_chay = ten_chay or f"{cfg.thi_nghiem.ten}_seed{cfg.thi_nghiem.seed}"
        self.logger = logger
        self.train_loader = train_loader
        self.dev_loader = dev_loader

        self.thiet_bi = chon_thiet_bi()
        self.model = model.to(self.thiet_bi)

        # --- Optimizer -------------------------------------------------------
        if cfg.toi_uu.optimizer != "adamw":
            raise ValueError(
                f"toi_uu.optimizer = {cfg.toi_uu.optimizer!r} chưa được cài. "
                "Hiện chỉ hỗ trợ adamw."
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.toi_uu.learning_rate,
            betas=tuple(cfg.toi_uu.betas),
            eps=cfg.toi_uu.eps,
            weight_decay=cfg.toi_uu.weight_decay,
        )

        self.scheduler = tao_scheduler(cfg, self.optimizer)
        self.criterion = tao_loss(cfg, vocab_size=cfg.du_lieu.vocab_size, pad_id=model.pad_id)

        # --- fp16 ------------------------------------------------------------
        self.dung_fp16 = _co_the_dung_fp16(cfg, self.thiet_bi)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=cfg.toi_uu.get("grad_scaler_init", 65536.0),
            enabled=self.dung_fp16,
        )

        # --- Các mốc ---------------------------------------------------------
        self.so_buoc_cong_don = max(1, int(cfg.toi_uu.get("so_buoc_cong_don_gradient", 1)))
        self.nguong_cat_gradient = cfg.toi_uu.get("cat_gradient_norm", 1.0)
        self.so_buoc_toi_da = so_buoc_toi_da or cfg.huan_luyen.so_buoc_toi_da
        self.danh_gia_moi = cfg.huan_luyen.danh_gia_moi
        self.luu_checkpoint_moi = cfg.huan_luyen.luu_checkpoint_moi
        self.dung_som_sau = cfg.huan_luyen.dung_som_sau

        self.thu_muc_checkpoint = Path(
            thu_muc_checkpoint or cfg.thi_nghiem.thu_muc_checkpoint
        )
        self.thu_muc_checkpoint.mkdir(parents=True, exist_ok=True)
        self.repo_hub = repo_hub

        # NGÂN SÁCH THỜI GIAN — thứ khiến lượt train dài chạy được trên Kaggle free.
        #
        # Phiên GPU của Kaggle bị cắt ở khoảng 12 tiếng. Lượt chạy 60.000 bước mất
        # hơn 13 tiếng nên chắc chắn bị giết giữa chừng, và nếu đúng lúc đó chưa
        # đẩy được checkpoint lên Hub thì mất trắng cả lượt.
        #
        # Đặt ngân sách thấp hơn giới hạn phiên thì trainer tự dừng, tự lưu, tự
        # đẩy lên Hub trong lúc còn sống. Phiên sau chỉ cần --tiep-tuc là chạy tiếp
        # đúng chỗ cũ. Huấn luyện dài trở thành nhiều chặng ngắn nối nhau.
        self.gio_toi_da = gio_toi_da

        # --- Trạng thái chạy -------------------------------------------------
        self.buoc = 0
        self.epoch = 0
        self.buoc_trong_epoch = 0
        self.loss_dev_tot_nhat = math.inf
        self.so_lan_khong_cai_thien = 0
        self.lich_su_loss: list[float] = []

    # ------------------------------------------------------------------ dữ liệu

    def _ghim_thu_tu_batch(self) -> None:
        """Ghim thứ tự batch của epoch hiện tại theo (seed, epoch).

        Nhờ vậy lượt chạy liền mạch và lượt bị giết rồi resume gặp ĐÚNG cùng một
        dãy batch — điều kiện để hai đường loss của TASK 14 trùng khít.
        """
        sampler = getattr(self.train_loader, "batch_sampler", None)
        generator = getattr(sampler, "_generator", None)
        if generator is not None:
            generator.manual_seed(self.cfg.thi_nghiem.seed * 100_003 + self.epoch)

    def _luong_batch(self):
        """Sinh batch liên tục qua nhiều epoch, tự tua nhanh khi resume."""
        while True:
            self._ghim_thu_tu_batch()
            can_bo_qua = self.buoc_trong_epoch

            for chi_so, batch in enumerate(self.train_loader):
                # Tua nhanh phần đã tiêu thụ trước khi bị giết phiên.
                if chi_so < can_bo_qua:
                    continue
                self.buoc_trong_epoch = chi_so + 1
                yield batch

            self.epoch += 1
            self.buoc_trong_epoch = 0

    def _ghi_log(self, buoc: int, **so_lieu) -> None:
        """Ghi log với ĐỦ mọi cột, thiếu thì để None.

        Xem chú thích của CAC_COT_LOG: không làm vậy thì cột nào không có mặt ở
        dòng ghi đầu tiên sẽ bị vứt im lặng ở mọi dòng sau.
        """
        if self.logger is None:
            return
        day_du = {ten: so_lieu.get(ten) for ten in CAC_COT_LOG}
        self.logger.ghi(buoc=buoc, **day_du)

    def _chuyen_len_thiet_bi(self, batch: dict) -> dict:
        return {
            ten: gia_tri.to(self.thiet_bi, non_blocking=True)
            for ten, gia_tri in batch.items()
        }

    # ------------------------------------------------------------------ một bước

    def _tinh_loss(self, batch: dict) -> torch.Tensor:
        logits = self.model(
            batch["src_ids"], batch["tgt_input"], batch["src_mask"], batch["tgt_mask"]
        )
        return self.criterion(logits, batch["labels"])

    def _mot_buoc_cap_nhat(self, luong) -> tuple[float, int]:
        """Chạy trọn một chu kỳ cộng dồn rồi cập nhật trọng số đúng MỘT lần.

        Returns:
            (loss trung bình của chu kỳ, số token thật đã xử lý)
        """
        self.optimizer.zero_grad(set_to_none=True)
        tong_loss = 0.0
        tong_token = 0

        for _ in range(self.so_buoc_cong_don):
            batch = self._chuyen_len_thiet_bi(next(luong))

            with torch.autocast(
                device_type=self.thiet_bi.type,
                dtype=torch.float16,
                enabled=self.dung_fp16,
            ):
                # Chia cho số bước cộng dồn NGAY TẠI ĐÂY. Cộng dồn gradient của n
                # lượt mà không chia thì độ lớn gradient gấp n lần, ngưỡng cắt 1.0
                # sẽ cắt gần hết, và mô hình học rất chậm mà không rõ vì sao.
                loss = self._tinh_loss(batch) / self.so_buoc_cong_don

            # Chặn NaN NGAY tại nguồn. Để nó chảy vào backward thì toàn bộ trọng
            # số thành NaN và mọi bước sau đều vô nghĩa, trong khi loss in ra vẫn
            # là một con số trông bình thường.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Loss ra {loss.item()} tại bước {self.buoc}. "
                    "Kiểm ba chỗ: learning rate quá lớn; mặt nạ che nhầm khiến cả "
                    "một hàng bị che (softmax của toàn -inf ra NaN); hoặc dữ liệu "
                    "có câu rỗng."
                )

            self.scaler.scale(loss).backward()

            tong_loss += loss.item() * self.so_buoc_cong_don
            tong_token += int((batch["labels"] != self.model.pad_id).sum())

        # CHI TIẾT DỄ SAI SỐ 1: phải gỡ hệ số giãn TRƯỚC khi cắt theo norm.
        # Quên dòng này thì đang cắt gradient đã nhân 65536, ngưỡng 1.0 vô nghĩa,
        # và chương trình không báo gì cả.
        if self.dung_fp16:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.nguong_cat_gradient)

        # CHI TIẾT DỄ SAI SỐ 2: step và update chỉ gọi MỘT lần ở cuối chu kỳ.
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return tong_loss / self.so_buoc_cong_don, tong_token

    # ------------------------------------------------------------------ đánh giá

    @torch.no_grad()
    def danh_gia(self) -> dict:
        """Loss trên tập dev. Trả về dict để ghi log."""
        if self.dev_loader is None:
            return {}

        self.model.eval()
        tong_loss = 0.0
        so_batch = 0

        for batch in self.dev_loader:
            batch = self._chuyen_len_thiet_bi(batch)
            with torch.autocast(
                device_type=self.thiet_bi.type,
                dtype=torch.float16,
                enabled=self.dung_fp16,
            ):
                loss = self._tinh_loss(batch)
            tong_loss += loss.item()
            so_batch += 1

        self.model.train()
        if so_batch == 0:
            return {}

        loss_dev = tong_loss / so_batch
        return {
            "loss_dev": loss_dev,
            # Perplexity dễ đọc hơn loss khi so giữa các lần chạy, và là con số
            # mà báo cáo dịch máy nào cũng có.
            "perplexity_dev": math.exp(min(loss_dev, 20.0)),
        }

    # ------------------------------------------------------------------ checkpoint

    def _luu(self, ten_file: str, loss_dev: float | None) -> Path:
        return luu_checkpoint(
            self.thu_muc_checkpoint / ten_file,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            buoc=self.buoc,
            epoch=self.epoch,
            cfg=self.cfg,
            loss_dev=loss_dev,
            che_do=self.che_do,
            so_lieu_them={"buoc_trong_epoch": self.buoc_trong_epoch},
        )

    def duong_dan_hub(self, ten_file: str) -> str:
        """Đường dẫn trên Hub, CÓ KÈM tên lượt chạy.

        Nhờ vậy 12 lượt ablation của TASK 17 nằm ở 12 chỗ khác nhau thay vì đè lên
        nhau. Dùng cả lúc đẩy lẫn lúc kéo về, nên hai bên không thể lệch nhau.
        """
        return f"checkpoints/{self.ten_chay}/{ten_file}"

    def _dong_bo_hub(self, duong_dan: Path, ten_file: str) -> None:
        """Đẩy checkpoint và log lên Hub.

        Smoke test CŨNG đẩy, nhưng vào nhánh `smoke/` tách hẳn — có vậy lượt smoke
        mới kiểm được luôn cơ chế đẩy, thứ đã âm thầm hỏng suốt 13 tiếng vì repo
        chưa được tạo. Chuyện smoke đè lên bản thật đã có hai lớp chặn: tiền tố
        riêng trong hub_sync, và `che_do_mong_doi` trong nap_checkpoint.
        """
        if not self.repo_hub:
            return

        from nmt.training.hub_sync import TIEN_TO_LOG, day_len_hub, day_thu_muc_len_hub

        day_len_hub(duong_dan, self.repo_hub, self.duong_dan_hub(ten_file),
                    che_do=self.che_do)

        # Đẩy CẢ log, nếu không thì kernel Kaggle chết là mất log và TASK 14
        # không vẽ được đường loss liền mạch qua các lần bị giết.
        if self.logger is not None:
            # Log cũng phải tách theo lượt chạy, nếu không thì đường loss của 12
            # lượt ablation trộn lẫn vào nhau trong cùng một file.
            day_thu_muc_len_hub(
                self.logger.thu_muc, self.repo_hub,
                f"{TIEN_TO_LOG}/{self.ten_chay}", che_do=self.che_do,
            )

    def tiep_tuc_tu(self, duong_dan: str | Path) -> dict:
        """Nạp checkpoint rồi chạy tiếp đúng chỗ cũ."""
        thong_tin = nap_checkpoint(
            duong_dan,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            map_location=self.thiet_bi,
            # Lượt chạy thật KHÔNG được vô tình nạp checkpoint của smoke test.
            # Đây là bài học mục 1.8 trong `Sưu tập lỗi.md`.
            che_do_mong_doi=self.che_do,
        )
        self.buoc = thong_tin["buoc"]
        self.epoch = thong_tin["epoch"]

        goi = torch.load(Path(duong_dan), map_location="cpu", weights_only=False)
        self.buoc_trong_epoch = goi.get("buoc_trong_epoch", 0)

        if thong_tin["loss_dev"] is not None:
            self.loss_dev_tot_nhat = thong_tin["loss_dev"]

        print(f"[trainer] Chạy tiếp từ bước {self.buoc} (epoch {self.epoch}, "
              f"batch {self.buoc_trong_epoch} trong epoch).")
        return thong_tin

    # ------------------------------------------------------------------ vòng chính

    def train(self) -> dict:
        """Chạy tới so_buoc_toi_da hoặc tới khi dừng sớm. Trả về tóm tắt lượt chạy."""
        self.model.train()
        luong = self._luong_batch()

        moc_thoi_gian = time.perf_counter()
        moc_bat_dau_chay = time.perf_counter()
        token_tu_lan_ghi_truoc = 0
        buoc_bat_dau = self.buoc
        dung_som = False
        het_gio = False

        print(f"[trainer] Thiết bị {self.thiet_bi} · fp16 {'BẬT' if self.dung_fp16 else 'TẮT'} · "
              f"cộng dồn {self.so_buoc_cong_don} · chế độ {self.che_do}")
        if self.gio_toi_da:
            print(f"[trainer] Ngân sách thời gian: {self.gio_toi_da:.2f} giờ. "
                  "Hết giờ sẽ tự lưu và dừng để phiên sau chạy tiếp.")

        while self.buoc < self.so_buoc_toi_da:
            loss_train, so_token = self._mot_buoc_cap_nhat(luong)
            self.buoc += 1
            self.lich_su_loss.append(loss_train)
            token_tu_lan_ghi_truoc += so_token

            # --- ghi log -----------------------------------------------------
            if self.logger is not None and self.buoc % 50 == 0:
                giay_troi = time.perf_counter() - moc_thoi_gian
                self._ghi_log(
                    self.buoc,
                    loss_train=loss_train,
                    learning_rate=self.scheduler.learning_rate_hien_tai(),
                    giay_moi_buoc=giay_troi / 50,
                    token_moi_giay=token_tu_lan_ghi_truoc / max(giay_troi, 1e-9),
                )
                moc_thoi_gian = time.perf_counter()
                token_tu_lan_ghi_truoc = 0

            if self.buoc % max(1, self.danh_gia_moi // 10) == 0:
                print(f"  bước {self.buoc:>6} · loss {loss_train:.4f} · "
                      f"lr {self.scheduler.learning_rate_hien_tai():.2e}")

            # --- đánh giá + dừng sớm ------------------------------------------
            if self.buoc % self.danh_gia_moi == 0:
                so_lieu = self.danh_gia()
                if so_lieu:
                    loss_dev = so_lieu["loss_dev"]
                    print(f"  [đánh giá] bước {self.buoc} · loss_dev {loss_dev:.4f} · "
                          f"ppl {so_lieu['perplexity_dev']:.2f}")
                    self._ghi_log(self.buoc, **so_lieu)

                    if loss_dev < self.loss_dev_tot_nhat:
                        self.loss_dev_tot_nhat = loss_dev
                        self.so_lan_khong_cai_thien = 0
                        duong_dan = self._luu(TEN_FILE_TOT_NHAT, loss_dev)
                        self._dong_bo_hub(duong_dan, TEN_FILE_TOT_NHAT)
                    else:
                        self.so_lan_khong_cai_thien += 1
                        if self.so_lan_khong_cai_thien >= self.dung_som_sau:
                            print(f"[trainer] Dừng sớm: {self.dung_som_sau} lần đánh giá "
                                  "liên tiếp không cải thiện.")
                            dung_som = True

            # --- hết ngân sách thời gian --------------------------------------
            #
            # Kiểm ở đây, TRƯỚC khối lưu checkpoint bên dưới, để lượt chạy luôn
            # kết thúc bằng một checkpoint đã đẩy lên Hub thành công.
            if self.gio_toi_da is not None:
                gio_da_chay = (time.perf_counter() - moc_bat_dau_chay) / 3600
                if gio_da_chay >= self.gio_toi_da:
                    print(f"[trainer] Hết ngân sách {self.gio_toi_da:.2f} giờ tại bước "
                          f"{self.buoc}/{self.so_buoc_toi_da}. Lưu lại rồi dừng.")
                    print("[trainer] Phiên sau chạy tiếp bằng cờ --tiep-tuc.")
                    het_gio = True

            # --- lưu checkpoint định kỳ ---------------------------------------
            if self.buoc % self.luu_checkpoint_moi == 0 or dung_som or het_gio:
                duong_dan = self._luu(TEN_FILE_MOI_NHAT, None)
                self._dong_bo_hub(duong_dan, TEN_FILE_MOI_NHAT)

            if dung_som or het_gio:
                break

        # Luôn lưu một bản cuối, kể cả khi chạy hết số bước mà không rơi đúng vào
        # mốc luu_checkpoint_moi — thiếu bản này là mất trắng phần chạy sau mốc cuối.
        duong_dan_cuoi = self._luu(TEN_FILE_MOI_NHAT, None)
        self._dong_bo_hub(duong_dan_cuoi, TEN_FILE_MOI_NHAT)
        if self.logger is not None:
            self.logger.dong()

        return {
            "buoc_cuoi": self.buoc,
            "so_buoc_da_chay": self.buoc - buoc_bat_dau,
            "epoch": self.epoch,
            "loss_train_cuoi": self.lich_su_loss[-1] if self.lich_su_loss else None,
            "loss_dev_tot_nhat": (
                self.loss_dev_tot_nhat if math.isfinite(self.loss_dev_tot_nhat) else None
            ),
            "dung_som": dung_som,
            "het_gio": het_gio,
            "da_xong_toan_bo": self.buoc >= self.so_buoc_toi_da or dung_som,
            "checkpoint_moi_nhat": str(duong_dan_cuoi),
        }
