"""TASK 16 — Kiểm hàm chấm điểm BLEU và chrF++.

Vì sao file này tồn tại: bản đầu của `metrics.py` gọi `corpus_bleu(...).signature`,
mà `BLEUScore` không có thuộc tính đó — hàm ném AttributeError ngay lần gọi đầu.
Lỗi lọt được vì không có bài test nào gọi thử hai hàm này; cả `evaluate.py` lẫn
`overfit_sanity.py` chỉ gọi chúng ở cuối một lần chạy dài, nên không ai chạy tới.

Bài học đóng lại thành quy tắc: mọi hàm nằm trên đường đi tới một con số của báo
cáo đều phải có một bài test gọi nó với dữ liệu bé xíu, chạy trong một phần nghìn
giây. Số liệu báo cáo không được phép phụ thuộc vào một đường code chưa từng chạy.
"""

from __future__ import annotations

import pytest

from nmt.eval.metrics import cham_bleu, cham_chrf

# Một câu trùng khít, một câu lệch — đủ để điểm nằm trong khoảng (0, 100).
DU_DOAN = ["tôi yêu bạn", "hôm nay trời đẹp"]
THAM_CHIEU = ["tôi yêu bạn", "hôm nay trời rất đẹp"]


# --------------------------------------------------------------- chạy được
def test_cham_bleu_chay_duoc_va_tra_ve_dung_kieu():
    """Bài test đáng lẽ phải bắt được lỗi `.signature` ngay từ đầu."""
    diem, chu_ky = cham_bleu(DU_DOAN, THAM_CHIEU)

    assert isinstance(diem, float)
    assert 0.0 <= diem <= 100.0
    # Chữ ký PHẢI là chuỗi. get_signature() trả về đối tượng Signature, quên
    # str() thì lúc ghi CSV nhận về "<... object at 0x...>" và con số mất luôn
    # khả năng đối chiếu với bài báo.
    assert isinstance(chu_ky, str)
    assert chu_ky, "Chữ ký rỗng — tiêu chí 'Xong khi' của TASK 16 yêu cầu ghi nguyên văn."


def test_cham_chrf_chay_duoc_va_tra_ve_dung_kieu():
    diem, chu_ky = cham_chrf(DU_DOAN, THAM_CHIEU)

    assert isinstance(diem, float)
    assert 0.0 <= diem <= 100.0
    assert isinstance(chu_ky, str) and chu_ky


# --------------------------------------------------------------- chữ ký
def test_chu_ky_bleu_co_du_thong_tin_de_doi_chieu():
    """Chuỗi chữ ký là thứ khiến điểm BLEU so sánh được với bài báo.

    Thiếu `tok:` thì không ai biết bản dịch được tách từ kiểu gì, và con số trở
    thành vô nghĩa khi đặt cạnh bảng của người khác.
    """
    _, chu_ky = cham_bleu(DU_DOAN, THAM_CHIEU)

    for phan in ("nrefs:", "case:", "tok:", "version:"):
        assert phan in chu_ky, f"Chữ ký BLEU thiếu '{phan}': {chu_ky}"


def test_chu_ky_chrf_ghi_dung_la_chrf_cong_cong():
    """chrF++ khác chrF thường ở chỗ có tính n-gram TỪ tới bậc 2.

    Trong chữ ký, phần đó hiện ra là `nw:2`. Nếu ai đó lỡ bỏ `word_order=2` thì
    chữ ký thành `nw:0` — vẫn ra một con số hợp lý, nhưng là chrF chứ không phải
    chrF++, và không so được với bảng nào ghi chrF++.
    """
    _, chu_ky = cham_chrf(DU_DOAN, THAM_CHIEU)
    assert "nw:2" in chu_ky, f"Đây không phải chrF++ (thiếu nw:2): {chu_ky}"


# --------------------------------------------------------------- tính chất
def test_dich_trung_khit_cho_diem_toi_da():
    diem_bleu, _ = cham_bleu(THAM_CHIEU, THAM_CHIEU)
    diem_chrf, _ = cham_chrf(THAM_CHIEU, THAM_CHIEU)

    assert diem_bleu == pytest.approx(100.0, abs=1e-6)
    assert diem_chrf == pytest.approx(100.0, abs=1e-6)


def test_dich_sai_hoan_toan_cho_diem_thap():
    rac = ["xyz qwe rty", "abc def ghi"]
    diem, _ = cham_bleu(rac, THAM_CHIEU)
    assert diem < 5.0


def test_diem_khong_phu_thuoc_thu_tu_goi():
    """Gọi hai lần trên cùng dữ liệu phải ra cùng một con số.

    Chặn trường hợp ai đó dựng metric ở phạm vi module rồi vô tình để nó tích
    lũy trạng thái giữa các lần chấm.
    """
    lan_1, _ = cham_bleu(DU_DOAN, THAM_CHIEU)
    cham_bleu(["rác", "rác"], THAM_CHIEU)      # một lần chấm khác xen vào
    lan_2, _ = cham_bleu(DU_DOAN, THAM_CHIEU)

    assert lan_1 == pytest.approx(lan_2)


# --------------------------------------------------------------- chặn lỗi
def test_chan_lech_so_cau():
    """Lệch số câu là lỗi im lặng nguy hiểm nhất của khâu chấm điểm.

    sacrebleu vẫn chấm được và vẫn ra một con số trông hợp lý, nhưng câu thứ i
    của bản dịch đang bị so với câu thứ i của tham chiếu ở một vị trí đã lệch.
    """
    with pytest.raises(ValueError, match="khác số câu"):
        cham_bleu(["một câu"], THAM_CHIEU)

    with pytest.raises(ValueError, match="khác số câu"):
        cham_chrf(["một câu"], THAM_CHIEU)


def test_chan_tap_rong():
    with pytest.raises(ValueError, match="Không có câu nào"):
        cham_bleu([], [])
