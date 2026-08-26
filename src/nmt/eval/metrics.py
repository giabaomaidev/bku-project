"""TASK 16 — Chấm điểm BLEU và chrF++.  Người làm: My.  [Evaluation • Bắt buộc]

Xong khi: BLEU trên tst2013 tối thiểu 19, mức tốt là từ 22 trở lên.

BẮT BUỘC ghi lại nguyên văn CHUỖI CHỮ KÝ (signature) mà sacrebleu xuất ra.
Thiếu chuỗi này thì điểm BLEU không so sánh được với bất kỳ bài báo nào, vì
BLEU chỉ so sánh được khi hai bên dùng cùng một cách tokenize.

chrF++ đo mức trùng khớp ở cấp nhóm ký tự. Thước đo này quan trọng với tiếng
Việt vì tiếng Việt viết rời từng âm tiết.

LẤY CHỮ KÝ Ở ĐÂU. Chữ ký thuộc về đối tượng METRIC, không thuộc về đối tượng
điểm số. `sacrebleu.corpus_bleu(...)` trả về một `BLEUScore` — nó có `.score`
nhưng KHÔNG có `.signature`, nên gọi kiểu đó là AttributeError ngay lần chạy
đầu tiên. Đường đúng là dựng `BLEU()` / `CHRF()` rồi hỏi `get_signature()`:

    metric = sacrebleu.BLEU()
    diem   = metric.corpus_score(du_doan, [tham_chieu])
    chu_ky = str(metric.get_signature())

`get_signature()` trả về một đối tượng Signature chứ không phải chuỗi, nên phải
`str()` lại — nếu không thì lúc ghi ra CSV sẽ nhận về dạng `<... object at 0x...>`
và con số mất luôn khả năng đối chiếu.
"""

from __future__ import annotations

import sacrebleu


def cham_bleu(du_doan: list[str], tham_chieu: list[str]) -> tuple[float, str]:
    """Returns: (điểm BLEU, chuỗi chữ ký của sacrebleu).

    sacrebleu nhận NHIỀU bản dịch chuẩn cho mỗi câu, nên tham chiếu phải bọc
    thêm một lớp list: `[tham_chieu]` nghĩa là "một bộ tham chiếu duy nhất".
    """
    _kiem_dau_vao(du_doan, tham_chieu)
    metric = sacrebleu.BLEU()
    diem = metric.corpus_score(du_doan, [tham_chieu])
    return diem.score, str(metric.get_signature())


def cham_chrf(du_doan: list[str], tham_chieu: list[str]) -> tuple[float, str]:
    """chrF++ (word_order=2). Returns: (điểm, chuỗi chữ ký).

    `word_order=2` chính là phần "++" của chrF++: ngoài n-gram ký tự, tính thêm
    n-gram từ tới bậc 2. Bỏ tham số này đi là ra chrF thường, một thước đo khác,
    và con số sẽ không so được với bảng nào ghi chrF++.
    """
    _kiem_dau_vao(du_doan, tham_chieu)
    metric = sacrebleu.CHRF(word_order=2)
    diem = metric.corpus_score(du_doan, [tham_chieu])
    return diem.score, str(metric.get_signature())


def _kiem_dau_vao(du_doan: list[str], tham_chieu: list[str]) -> None:
    """Chặn hai lỗi làm điểm sai mà sacrebleu không hề báo.

    Lệch số câu là lỗi nguy hiểm nhất: sacrebleu vẫn chấm được và vẫn ra một
    con số trông hợp lý, nhưng câu thứ i của bản dịch đang bị so với câu thứ i
    của tham chiếu ở một vị trí đã lệch. Điểm ra thấp bất thường mà không ai
    hiểu vì sao.
    """
    if len(du_doan) != len(tham_chieu):
        raise ValueError(
            f"Số câu dịch ({len(du_doan)}) khác số câu tham chiếu ({len(tham_chieu)}). "
            "Chấm kiểu này vẫn ra điểm nhưng là điểm của hai tập lệch nhau."
        )
    if not du_doan:
        raise ValueError("Không có câu nào để chấm.")
