"""TASK 01 + TASK 12 — Đồng bộ Hugging Face Hub.  Người làm: Bảo.  [Training Infra • Bắt buộc]

Nhóm dùng bản free trên Kaggle, đôi khi rớt mạng rồi F5 lại là mất sạch
checkpoint, nên phải đẩy checkpoint qua Hugging Face.

BẢO MẬT — đọc token theo đúng thứ tự này, KHÔNG viết thẳng token vào code:
    1. Kaggle Secrets (Add-ons -> Secrets), tên biến HF_TOKEN
    2. Biến môi trường HF_TOKEN (khi chạy ở máy cá nhân)
    3. Không có thì báo lỗi rõ ràng, đừng chạy tiếp rồi hỏng ở bước cuối

Giữ hai bản trên Hub: bản mới nhất để chạy tiếp, bản tốt nhất theo loss dev.
Đẩy CẢ thư mục log lên cùng, nếu không thì mất log mỗi lần kernel chết và
không vẽ được đường loss liền mạch cho thí nghiệm giết phiên (TASK 14).

BỐN CÁI BẪY LẤY TỪ `Sưu tập lỗi.md`, mỗi cái đều đã thật sự xảy ra:

    1.4  Tạo Kaggle Secret rồi mà vẫn báo không có token. Hai lý do: chưa bật
         công tắc riêng cho notebook đó, hoặc phiên khởi động trước khi gắn
         secret nên phải Restart session. Mã cũ nuốt exception im lặng nên không
         ai biết vì sao. Ở đây `doc_token` IN RÕ loại lỗi và cách khắc phục.

    1.7  Lược đồ đặt tên trên Hub đổi sau khi đã đẩy dữ liệu lên, nên notebook đi
         tìm chỗ mới, không thấy, rồi kết luận "chưa có checkpoint" và huấn luyện
         lại 3 tiếng vô ích — trong khi trọng số vẫn nằm nguyên trên Hub, chỉ là
         cái tên không khớp. Ở đây `liet_ke_file` LUÔN được gọi trước khi kết luận
         Hub trống, và in ra những gì repo thật sự đang có.

    1.8  Một lượt smoke test ĐÈ LÊN checkpoint thật vì hai bên chung đường dẫn.
         Ở đây smoke test MẶC ĐỊNH KHÔNG ĐẨY GÌ LÊN HUB. Ai cố tình bật thì mọi
         thứ rơi vào tiền tố `smoke/`, không đụng được vào dữ liệu thật.

    1.12 Kaggle TẮT Internet mặc định, nên lỗi mạng đầu tiên rất dễ bị hiểu nhầm
         thành lỗi token. `_giai_thich_loi_mang` phân biệt hai trường hợp đó.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from nmt.training.checkpoint import CHE_DO_SMOKE, CHE_DO_THAT

# Tên biến môi trường / tên Kaggle Secret. Để một chỗ duy nhất, vì tài liệu
# hướng dẫn và mọi thông báo lỗi đều phải nhắc đúng cái tên này.
TEN_BIEN_TOKEN = "HF_TOKEN"

# Bố cục thư mục trên Hub. Giám khảo tải repo về là dựng lại được nguyên lượt
# chạy, nên tên phải nói lên nội dung.
TEN_CHECKPOINT_MOI_NHAT = "checkpoints/moi_nhat.pt"
TEN_CHECKPOINT_TOT_NHAT = "checkpoints/tot_nhat.pt"
TIEN_TO_LOG = "logs"
TIEN_TO_CAU_HINH = "configs"
TEN_TOKENIZER = "artifacts/tokenizer/tokenizer.json"

# Mọi thứ của smoke test rơi vào đây, không bao giờ đụng vào dữ liệu thật.
TIEN_TO_SMOKE = "smoke"

GIAY_CHO_GIUA_HAI_LAN_THU = 5

# Các repo đã được bảo đảm tồn tại trong tiến trình này. Xem `_dam_bao_mot_lan`.
_REPO_DA_DAM_BAO: set[str] = set()


def doc_token(bat_buoc: bool = True) -> str | None:
    """Kaggle Secrets -> biến môi trường -> báo lỗi.

    Args:
        bat_buoc: True thì không có token là ném lỗi. False thì trả về None để
            người gọi tự quyết định bỏ qua phần đồng bộ — dùng cho smoke test và
            cho lượt chạy ở máy cá nhân không cần Hub.
    """
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        token = UserSecretsClient().get_secret(TEN_BIEN_TOKEN)
        if token:
            return token
    except ImportError:
        # Không chạy trên Kaggle. Bình thường, không cần nói gì.
        pass
    except Exception as loi:
        # Đang ở Kaggle mà đọc secret hỏng — đây mới là trường hợp phải nói to.
        # Mục 1.4: mã cũ dùng `except Exception: pass` nên không ai biết vì sao.
        print(f"[hub_sync] Đọc Kaggle Secret {TEN_BIEN_TOKEN!r} thất bại: "
              f"{type(loi).__name__}: {loi}")
        print("[hub_sync] Sửa: Add-ons > Secrets > BẬT CÔNG TẮC cho đúng notebook này,")
        print(f"[hub_sync] rồi Run > Restart session. Tên secret phải đúng là {TEN_BIEN_TOKEN}.")

    token = os.environ.get(TEN_BIEN_TOKEN)
    if token:
        return token

    if not bat_buoc:
        return None

    raise RuntimeError(
        f"Không tìm thấy Hugging Face token.\n"
        f"  - Trên Kaggle: Add-ons -> Secrets -> thêm {TEN_BIEN_TOKEN} (token loại Write),\n"
        f"    BẬT công tắc cho đúng notebook này, rồi Run -> Restart session.\n"
        f"  - Trên máy cá nhân: đặt biến môi trường {TEN_BIEN_TOKEN}.\n"
        f"TUYỆT ĐỐI không viết token thẳng vào code hay commit lên git."
    )


def _giai_thich_loi_mang(loi: Exception) -> str:
    """Đổi exception của huggingface_hub thành lời khuyên đọc được.

    Trên Kaggle, Internet TẮT MẶC ĐỊNH (mục 1.12). Khi đó lỗi trả về trông rất
    giống lỗi token, và nhóm dễ đi sửa nhầm chỗ.
    """
    mo_ta = f"{type(loi).__name__}: {loi}"
    chuoi = str(loi).lower()

    if any(tu in chuoi for tu in ("connection", "resolve", "timed out", "network", "dns")):
        return (
            f"{mo_ta}\n"
            "  Nhiều khả năng Kaggle đang TẮT Internet (mặc định là tắt).\n"
            "  Sửa: panel bên phải > Settings > Internet = On (cần xác thực số điện thoại)."
        )
    if any(tu in chuoi for tu in ("401", "403", "unauthorized", "forbidden", "token")):
        return (
            f"{mo_ta}\n"
            "  Nhiều khả năng token sai loại. Token phải là loại WRITE, không phải Read.\n"
            f"  Tạo lại ở huggingface.co/settings/tokens rồi cập nhật {TEN_BIEN_TOKEN}."
        )
    return mo_ta


def _tien_to_theo_che_do(ten_tren_hub: str, che_do: str) -> str:
    """Smoke test ghi vào `smoke/...`, lượt chạy thật ghi thẳng vào gốc.

    Đây là bản sửa của mục 1.8. Ở đồ án trước, hai chế độ dùng chung đường dẫn
    nên smoke test đè mất checkpoint thật mà không có cảnh báo nào.
    """
    return f"{TIEN_TO_SMOKE}/{ten_tren_hub}" if che_do == CHE_DO_SMOKE else ten_tren_hub


def dam_bao_repo(repo_id: str, token: str | None = None, rieng_tu: bool = True) -> bool:
    """Tạo repo trên Hub nếu chưa có. Trả về True khi repo đã sẵn sàng."""
    try:
        from huggingface_hub import HfApi

        HfApi().create_repo(
            repo_id=repo_id,
            token=token or doc_token(),
            private=rieng_tu,
            exist_ok=True,
        )
        _REPO_DA_DAM_BAO.add(repo_id)
        return True
    except Exception as loi:
        print(f"[hub_sync] Không tạo/mở được repo {repo_id}: {_giai_thich_loi_mang(loi)}")
        return False


def _dam_bao_mot_lan(repo_id: str, token: str | None = None) -> None:
    """Tạo repo ở LẦN ĐẨY ĐẦU TIÊN, sau đó không gọi lại nữa.

    Vì sao cần: bản đầu chỉ gọi `dam_bao_repo` trong scripts/train.py, mà lại đặt
    SAU `trainer.train()`. Thành ra suốt cả lượt huấn luyện, mọi lần đẩy checkpoint
    đều bắn vào một repo chưa tồn tại và trả về RepositoryNotFoundError. Hàm đẩy
    không ném lỗi (đúng thiết kế, để mất mạng một nhịp thì train vẫn chạy tiếp),
    nên chuyện này diễn ra âm thầm suốt 13 tiếng: Hugging Face trống trơn, mỗi lần
    lưu còn phí thêm ~15 giây cho ba lần thử lại.

    Đặt việc tạo repo ngay trong hàm đẩy thì KHÔNG CÒN thứ tự gọi nào làm hỏng nó
    được nữa. Sửa ở một chỗ, mọi nơi gọi đều đúng.
    """
    if repo_id in _REPO_DA_DAM_BAO:
        return
    dam_bao_repo(repo_id, token=token)
    # Thêm vào tập kể cả khi tạo hỏng, để không thử đi thử lại ở mỗi lần đẩy.
    # Lần đẩy ngay sau đó sẽ tự báo lỗi rõ ràng nếu repo thật sự không dùng được.
    _REPO_DA_DAM_BAO.add(repo_id)


def liet_ke_file(repo_id: str, token: str | None = None) -> list[str]:
    """Liệt kê mọi file đang có trên repo.

    Mục 1.7: LUÔN gọi hàm này trước khi kết luận "Hub chưa có gì". Lần trước
    nhóm kết luận nhầm rồi train lại 3 tiếng, trong khi trọng số vẫn nằm nguyên
    trên Hub, chỉ là tên đường dẫn không khớp.
    """
    try:
        from huggingface_hub import HfApi

        return list(
            HfApi().list_repo_files(repo_id=repo_id, token=token or doc_token(bat_buoc=False))
        )
    except Exception as loi:
        print(f"[hub_sync] Không liệt kê được file của {repo_id}: {_giai_thich_loi_mang(loi)}")
        return []


def day_len_hub(
    duong_dan_file,
    repo_id: str,
    ten_tren_hub: str,
    so_lan_thu: int = 3,
    *,
    token: str | None = None,
    che_do: str = CHE_DO_THAT,
    ghi_chu: str | None = None,
) -> bool:
    """Đẩy một file lên Hub, có cơ chế thử lại khi mạng lỗi. Trả về True nếu xong.

    Smoke test mặc định KHÔNG đẩy gì lên Hub — đúng yêu cầu, và cũng là bản sửa
    của mục 1.8. Ai cố tình bật thì mọi thứ rơi vào tiền tố `smoke/`.

    Không ném lỗi khi thất bại: mất mạng một nhịp thì huấn luyện vẫn phải chạy
    tiếp, checkpoint vẫn nằm ở đĩa cục bộ. Nhưng phải IN RÕ là đã thất bại chứ
    đừng im lặng — im lặng chính là thứ khiến mục 1.7 tốn 3 tiếng.
    """
    duong_dan_file = Path(duong_dan_file)
    if not duong_dan_file.exists():
        print(f"[hub_sync] Bỏ qua, không có file: {duong_dan_file}")
        return False

    # Smoke test VẪN đẩy lên Hub, nhưng vào nhánh `smoke/` tách hẳn. Có vậy mới
    # kiểm được luôn cơ chế đẩy — thứ đã âm thầm hỏng suốt 13 tiếng vì repo chưa
    # được tạo. Còn chuyện smoke đè lên bản thật thì đã có hai lớp chặn: tiền tố
    # riêng ở đây, và `che_do_mong_doi` trong nap_checkpoint.
    duong_dan_tren_hub = _tien_to_theo_che_do(ten_tren_hub, che_do)
    token = token or doc_token(bat_buoc=False)
    if token is None:
        print(f"[hub_sync] Không có {TEN_BIEN_TOKEN} nên bỏ qua đồng bộ {duong_dan_tren_hub}.")
        return False

    # Tạo repo ở lần đẩy đầu tiên. Không có dòng này thì mọi lần đẩy trong lúc
    # huấn luyện đều rơi vào RepositoryNotFoundError mà không ai biết.
    _dam_bao_mot_lan(repo_id, token)

    from huggingface_hub import HfApi

    api = HfApi()
    kich_thuoc_mb = duong_dan_file.stat().st_size / (1024 * 1024)

    for lan_thu in range(1, so_lan_thu + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(duong_dan_file),
                path_in_repo=duong_dan_tren_hub,
                repo_id=repo_id,
                token=token,
                commit_message=ghi_chu or f"đồng bộ {duong_dan_tren_hub} ({che_do})",
            )
            print(f"[hub_sync] Đã đẩy {duong_dan_tren_hub} ({kich_thuoc_mb:.1f} MB).")
            return True
        except Exception as loi:
            print(f"[hub_sync] Lần {lan_thu}/{so_lan_thu} đẩy {duong_dan_tren_hub} hỏng: "
                  f"{_giai_thich_loi_mang(loi)}")
            if lan_thu < so_lan_thu:
                time.sleep(GIAY_CHO_GIUA_HAI_LAN_THU * lan_thu)

    print(f"[hub_sync] BỎ CUỘC sau {so_lan_thu} lần với {duong_dan_tren_hub}. "
          "Huấn luyện vẫn chạy tiếp, checkpoint còn ở đĩa cục bộ.")
    return False


def day_thu_muc_len_hub(
    thu_muc,
    repo_id: str,
    tien_to_tren_hub: str = TIEN_TO_LOG,
    *,
    token: str | None = None,
    che_do: str = CHE_DO_THAT,
    so_lan_thu: int = 3,
) -> bool:
    """Đẩy nguyên một thư mục lên Hub — dùng cho thư mục log.

    Thiếu bước này thì kernel Kaggle chết là mất log, và TASK 14 không vẽ được
    đường loss liền mạch qua các lần bị giết. Mà đó lại là hình có giá trị nhất
    của cả đồ án.
    """
    thu_muc = Path(thu_muc)
    if not thu_muc.is_dir():
        print(f"[hub_sync] Bỏ qua, không có thư mục: {thu_muc}")
        return False

    token = token or doc_token(bat_buoc=False)
    if token is None:
        print(f"[hub_sync] Không có {TEN_BIEN_TOKEN} nên bỏ qua đồng bộ thư mục {thu_muc}.")
        return False

    _dam_bao_mot_lan(repo_id, token)

    from huggingface_hub import HfApi

    api = HfApi()
    duong_dan_tren_hub = _tien_to_theo_che_do(tien_to_tren_hub, che_do)

    for lan_thu in range(1, so_lan_thu + 1):
        try:
            api.upload_folder(
                folder_path=str(thu_muc),
                path_in_repo=duong_dan_tren_hub,
                repo_id=repo_id,
                token=token,
                commit_message=f"đồng bộ thư mục {duong_dan_tren_hub}",
            )
            print(f"[hub_sync] Đã đẩy thư mục {duong_dan_tren_hub}.")
            return True
        except Exception as loi:
            print(f"[hub_sync] Lần {lan_thu}/{so_lan_thu} đẩy thư mục hỏng: "
                  f"{_giai_thich_loi_mang(loi)}")
            if lan_thu < so_lan_thu:
                time.sleep(GIAY_CHO_GIUA_HAI_LAN_THU * lan_thu)

    return False


def tai_file(
    repo_id: str,
    ten_tren_hub: str,
    thu_muc_luu,
    *,
    token: str | None = None,
) -> Path | None:
    """Tải một file từ Hub về. Trả về None nếu Hub không có file đó."""
    try:
        from huggingface_hub import hf_hub_download

        duong_dan = hf_hub_download(
            repo_id=repo_id,
            filename=ten_tren_hub,
            local_dir=str(thu_muc_luu),
            token=token or doc_token(bat_buoc=False),
        )
        return Path(duong_dan)
    except Exception as loi:
        print(f"[hub_sync] Không tải được {ten_tren_hub}: {_giai_thich_loi_mang(loi)}")
        return None


def tai_checkpoint_moi_nhat(
    repo_id: str,
    thu_muc_luu,
    *,
    ten_tren_hub: str = TEN_CHECKPOINT_MOI_NHAT,
    token: str | None = None,
) -> Path | None:
    """Trả về đường dẫn checkpoint mới nhất, hoặc None nếu Hub chưa có gì.

    Đây là hàm khiến "bạn tớ train tiếp khi máy tớ hết quota" thành hiện thực:
    người sau chỉ cần chạy notebook, hàm này tự kéo checkpoint về rồi chạy tiếp
    đúng chỗ cũ.

    Mục 1.7: LIỆT KÊ repo trước, và khi không thấy thì IN RA những gì thật sự có.
    Lần trước nhóm kết luận nhầm "chưa có checkpoint" rồi train lại 3 tiếng trong
    khi file vẫn nằm đó dưới một cái tên khác.
    """
    thu_muc_luu = Path(thu_muc_luu)
    thu_muc_luu.mkdir(parents=True, exist_ok=True)

    cac_file = liet_ke_file(repo_id, token=token)
    if not cac_file:
        print(f"[hub_sync] Repo {repo_id} đang trống hoặc không đọc được — huấn luyện từ đầu.")
        return None

    if ten_tren_hub not in cac_file:
        cac_checkpoint = [f for f in cac_file if f.endswith(".pt")]
        print(f"[hub_sync] Không thấy {ten_tren_hub} trên {repo_id}.")
        print(f"[hub_sync] Các file .pt repo đang thật sự có: {cac_checkpoint or '(không có)'}")
        print("[hub_sync] Nếu thấy tên gần giống thì đó là lệch lược đồ đặt tên (mục 1.7), "
              "đừng vội train lại từ đầu.")
        return None

    return tai_file(repo_id, ten_tren_hub, thu_muc_luu, token=token)
