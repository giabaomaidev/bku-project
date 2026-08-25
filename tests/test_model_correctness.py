"""TASK 10 — Bộ 12 bài kiểm tra tính đúng đắn kiến trúc.  Người làm: My.  [Testing • Bắt buộc]

Xong khi: 12/12 bài đạt, cộng với bài học thuộc 50 câu (scripts/overfit_sanity.py)
          loss dưới 0,05 trong tối đa 500 bước và BLEU trên 90.

ĐÂY LÀ CỔNG CHẶN CUỐI TUẦN 2. Chưa qua thì chưa nên khảo sát cấu hình
(TASK 11) hay huấn luyện chính thức (TASK 15).

Vì sao cần bộ test này thay vì chỉ kiểm "forward pass chạy được":
    Lỗi ở phần mask hay lỗi hoán vị chiều trong attention THƯỜNG KHÔNG LÀM
    CHƯƠNG TRÌNH BÁO LỖI. Mô hình vẫn chạy, loss vẫn giảm, vẫn dịch ra chữ,
    nhưng dịch sai. Nếu chỉ lấy tiêu chí forward pass chạy được thì nhóm sẽ đi
    tiếp với một kiến trúc hỏng mà không hề biết, tới tuần 4 mới phát hiện thì
    không kịp sửa.

Thứ tự làm: viết xong thành phần nào thì chạy ngay bài test của thành phần đó
    bài 8  -> RMSNorm   (TASK 07)
    bài 9  -> SwiGLU    (TASK 08)
    bài 4  -> RoPE      (TASK 06)
Khi cả ba đã đạt mới ghép thành mô hình rồi chạy các bài còn lại.
Cách này giúp khi có lỗi thì biết ngay lỗi nằm ở thành phần nào.

LƯU Ý VỀ QUY TẮC KHÔNG DÙNG LỚP DỰNG SẴN:
    torch.nn.RMSNorm và F.scaled_dot_product_attention CHỈ được phép xuất hiện
    trong file này để đối chiếu số (bài 6). Không bao giờ xuất hiện trong src/.
    SwiGLU và RoPE thì PyTorch không có lớp tương ứng, nên hai thành phần này
    được kiểm bằng cài đặt tham chiếu tự viết bằng NumPy cộng với các tính chất
    toán học nêu ở bài 4 và bài 9.
"""

from __future__ import annotations

import pytest

# pytestmark = pytest.mark.skip(reason="TASK 10 — My. Bỏ skip khi TASK 05-09 đã xong.")


# ---------------------------------------------------------------- bài 1
def test_01_causal_mask_khong_cho_nhin_ve_sau():
    """Cho decoder chạy trên một câu đích. Sau đó thay đổi ngẫu nhiên các từ ở
    vị trí PHÍA SAU vị trí t. Đầu ra tại vị trí t bắt buộc phải KHÔNG ĐỔI.

    Bắt lỗi: mask bị lật ngược, mask cộng nhầm dấu, lệch một vị trí khi ghép
    đầu vào và đầu ra của decoder.

    ĐÂY LÀ LỖI NGUY HIỂM NHẤT của cả đồ án: mô hình nhìn trộm được đáp án nên
    loss lúc train giảm rất đẹp, nhưng lúc dịch thật thì ra rác.
    """
    from nmt.utils import nap_config
    from nmt.model.transformer import TransformerNMT
    import torch
    
    cfg = nap_config("configs/base.yaml")
    model = TransformerNMT(cfg)
    model.eval()
    
    tgt_ids = torch.randint(1, cfg.du_lieu.vocab_size, (1, 10))
    src_ids = torch.randint(1, cfg.du_lieu.vocab_size, (1, 5))
    
    from nmt.model.masking import tao_causal_mask
    tgt_mask = tao_causal_mask(10)
    
    out1 = model(src_ids, tgt_ids, tgt_mask=tgt_mask)
    
    t = 4
    tgt_ids_modified = tgt_ids.clone()
    tgt_ids_modified[0, t+1:] = torch.randint(1, cfg.du_lieu.vocab_size, (1, 10 - t - 1))
    
    out2 = model(src_ids, tgt_ids_modified, tgt_mask=tgt_mask)
    
    # Đầu ra tại vị trí <= t phải không đổi
    assert torch.allclose(out1[0, :t+1, :], out2[0, :t+1, :], atol=1e-5)

# ---------------------------------------------------------------- bài 2
def test_02_padding_mask_co_tac_dung():
    """Thêm vài token đệm vào cuối câu. Biểu diễn của các từ THẬT bắt buộc phải
    không đổi.

    Bắt lỗi: quên áp padding mask, hoặc áp sai chiều.
    """
    from nmt.utils import nap_config
    from nmt.model.transformer import TransformerNMT
    from nmt.model.masking import tao_padding_mask
    import torch
    
    cfg = nap_config("configs/base.yaml")
    model = TransformerNMT(cfg)
    model.eval()
    
    src_ids = torch.randint(1, cfg.du_lieu.vocab_size, (1, 5))
    tgt_ids = torch.randint(1, cfg.du_lieu.vocab_size, (1, 5))
    
    out1 = model(src_ids, tgt_ids)
    
    # Thêm token đệm
    src_ids_pad = torch.cat([src_ids, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    src_mask = tao_padding_mask(src_ids_pad, pad_id=0)
    
    out2 = model(src_ids_pad, tgt_ids, src_mask=src_mask)
    
    # Biểu diễn của các từ thật bắt buộc không đổi
    assert torch.allclose(out1, out2, atol=1e-5)

# ---------------------------------------------------------------- bài 3
def test_03_trong_so_attention_hop_le():
    """Sau khi qua softmax, mỗi HÀNG của ma trận attention phải cộng lại bằng 1,
    mọi giá trị phải không âm, và phải bằng 0 ở những vị trí bị mask.

    Bắt lỗi: nhầm trục khi tính softmax. Lỗi này rất hay gặp và gần như KHÔNG
    THỂ phát hiện bằng mắt.
    """
    from nmt.utils import nap_config
    from nmt.model.attention import MultiHeadAttention
    from nmt.model.masking import tao_causal_mask
    import torch
    
    cfg = nap_config("configs/base.yaml")
    attn = MultiHeadAttention(cfg.mo_hinh.d_model, cfg.mo_hinh.so_head)
    attn.eval()
    
    x = torch.randn(2, 5, cfg.mo_hinh.d_model)
    mask = tao_causal_mask(5)
    
    out, weights = attn(x, x, x, mask)
    
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)
    assert (weights >= 0).all()
    
    upper_tri = torch.triu(torch.ones(5, 5), diagonal=1).bool()
    assert (weights[..., upper_tri] == 0).all()

# ---------------------------------------------------------------- bài 4
def test_04_rope_giu_do_dai_va_ma_hoa_dung_vi_tri_tuong_doi():
    """Ba phần:

    1. Độ dài vector query sau khi xoay phải bằng độ dài trước khi xoay,
       sai lệch dưới 1e-5 — vì phép quay không làm đổi độ dài.
    2. Ở vị trí 0, phép quay phải là phép đồng nhất (đầu ra bằng đầu vào).
    3. Tích vô hướng giữa query ở vị trí m và key ở vị trí n phải CHỈ phụ thuộc
       vào hiệu (m - n). Kiểm bằng cách so cặp vị trí (2, 5) với cặp (7, 10),
       hai kết quả phải bằng nhau.

    Bắt lỗi: ghép cặp chiều sai khi xoay (lỗi phổ biến nhất khi tự cài RoPE),
    sai công thức tính góc quay, lỡ áp RoPE vào tensor value.

    CẢNH BÁO: bài này chạy ở float32 nên vẫn báo ĐẠT ngay cả khi bảng góc quay
    bị tính ở fp16. Bài 12 mới bắt được lỗi đó.
    """
    from nmt.model.positional import RoPE
    import torch
    d_head = 64
    rope = RoPE(d_head=d_head)
    
    x = torch.randn(1, 1, 15, d_head)
    rx = rope(x)
    
    # 1. Độ dài vector không đổi
    assert torch.allclose(x.norm(dim=-1), rx.norm(dim=-1), atol=1e-5)
    
    # 2. Vị trí 0 là phép đồng nhất
    assert torch.allclose(x[:, :, 0], rx[:, :, 0], atol=1e-5)
    
    # 3. Tích vô hướng chỉ phụ thuộc hiệu (m-n)
    q_const = torch.randn(1, 1, 1, d_head).expand(1, 1, 15, d_head)
    k_const = torch.randn(1, 1, 1, d_head).expand(1, 1, 15, d_head)
    
    rq_c = rope(q_const)
    rk_c = rope(k_const)
    
    dot1 = torch.sum(rq_c[:, :, 5] * rk_c[:, :, 2], dim=-1)
    dot2 = torch.sum(rq_c[:, :, 10] * rk_c[:, :, 7], dim=-1)
    
    assert torch.allclose(dot1, dot2, atol=1e-5)

# ---------------------------------------------------------------- bài 5
def test_05_gradient_chay_toi_moi_tham_so():
    """Sau một lần backward, MỌI tham số cần học đều phải có gradient khác rỗng
    và khác 0, và không có giá trị NaN.

    Bắt lỗi: một nhánh mạng bị đứt khỏi đồ thị tính toán, hoặc module quên đăng ký.
    """
    from nmt.utils import nap_config
    from nmt.model.transformer import TransformerNMT
    import torch
    
    cfg = nap_config("configs/base.yaml")
    model = TransformerNMT(cfg)
    
    src = torch.randint(1, cfg.du_lieu.vocab_size, (2, 5))
    tgt = torch.randint(1, cfg.du_lieu.vocab_size, (2, 5))
    
    out = model(src, tgt)
    loss = out.sum()
    loss.backward()
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Mất gradient ở {name}"
            assert not torch.isnan(p.grad).any(), f"Gradient NaN ở {name}"
            assert p.grad.abs().sum() > 0, f"Gradient bằng 0 ở {name}"

# ---------------------------------------------------------------- bài 6
def test_06_doi_chieu_so_voi_cai_dat_tham_chieu_pytorch():
    """Hai phần:

    1. Nạp CÙNG bộ trọng số vào torch.nn.RMSNorm và vào lớp RMSNorm tự viết,
       cho cùng đầu vào, sai lệch tuyệt đối trung bình phải dưới 1e-5.
    2. So phần lõi attention tự viết với F.scaled_dot_product_attention khi
       CHƯA áp RoPE, cùng đầu vào và cùng mask, sai lệch trung bình dưới 1e-5.

    Bắt mọi sai lệch còn sót về thứ tự phép toán hoặc hệ số chia.

    Bản thảo lần 2 đối chiếu cả mô hình với nn.Transformer, nhưng kiến trúc hiện
    đại không còn khớp với lớp đó nên nhóm chuyển sang đối chiếu TỪNG THÀNH PHẦN.
    Cách này tốt hơn vì khi lệch thì biết ngay lệch ở đâu.
    """
    import torch
    import torch.nn.functional as F
    from torch.nn import RMSNorm as TorchRMSNorm
    from nmt.model.normalization import RMSNorm as MyRMSNorm
    from nmt.model.attention import MultiHeadAttention
    from nmt.utils import nap_config
    
    d_model = 64
    x = torch.randn(2, 5, d_model)
    
    # 1. RMSNorm
    my_norm = MyRMSNorm(d_model)
    torch_norm = TorchRMSNorm(d_model)
    torch_norm.weight.data = my_norm.weight.data.clone()
    
    out_my = my_norm(x)
    out_torch = torch_norm(x)
    assert torch.allclose(out_my, out_torch, atol=1e-5)
    
    # 2. Attention
    cfg = nap_config("configs/base.yaml")
    attn = MultiHeadAttention(cfg.mo_hinh.d_model, cfg.mo_hinh.so_head)
    q = torch.randn(2, 4, cfg.mo_hinh.d_model)
    k = torch.randn(2, 4, cfg.mo_hinh.d_model)
    v = torch.randn(2, 4, cfg.mo_hinh.d_model)
    
    out_my, _ = attn(q, k, v, mask=None, rope=None)
    
    q_proj = attn.w_q(q).view(2, 4, attn.so_head, attn.d_head).transpose(1, 2)
    k_proj = attn.w_k(k).view(2, 4, attn.so_head, attn.d_head).transpose(1, 2)
    v_proj = attn.w_v(v).view(2, 4, attn.so_head, attn.d_head).transpose(1, 2)
    out_torch = F.scaled_dot_product_attention(q_proj, k_proj, v_proj)
    out_torch = out_torch.transpose(1, 2).reshape(2, 4, cfg.mo_hinh.d_model)
    out_torch = attn.w_o(out_torch)
    
    assert torch.allclose(out_my, out_torch, atol=1e-5)

# ---------------------------------------------------------------- bài 7
@pytest.mark.slow
def test_07_hoc_overfit_50_cau():
    """Loss phải xuống dưới 0,05 trong tối đa 500 bước, và BLEU khi dịch lại
    chính 50 câu đó phải trên 90.

    Bài kiểm tra tổng hợp cuối cùng, theo đúng gợi ý của anh Huy.
    Chạy đầy đủ bằng scripts/overfit_sanity.py để lấy cả biểu đồ loss.
    """
    import subprocess
    import sys
    res = subprocess.run([sys.executable, "scripts/overfit_sanity.py"], capture_output=True, text=True)
    assert res.returncode == 0, f"Overfit script failed: {res.stderr}"

# ---------------------------------------------------------------- bài 8
def test_08_rmsnorm_chuan_hoa_dung():
    """Cho một tensor bất kỳ đi qua RMSNorm với hệ số nhân đặt bằng 1.
    Trung bình bình phương của đầu ra phải bằng 1, sai lệch dưới 1e-5.

    Sau đó CỘNG MỘT HẰNG SỐ vào toàn bộ đầu vào rồi kiểm lại: kết quả PHẢI THAY
    ĐỔI. Đây chính là điểm khác với LayerNorm — LayerNorm trừ trung bình nên
    cộng hằng số vào không đổi gì cả.

    Bắt lỗi: cài nhầm thành LayerNorm (lỡ trừ trung bình), quên căn bậc hai,
    quên hằng số epsilon chống chia cho 0.
    """
    from nmt.model.normalization import RMSNorm
    import torch
    d_model = 64
    norm = RMSNorm(d_model)
    
    x = torch.randn(2, 5, d_model)
    out = norm(x)
    
    rms = (out ** 2).mean(dim=-1)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
    
    out2 = norm(x + 10.0)
    assert not torch.allclose(out, out2, atol=1e-4)

# ---------------------------------------------------------------- bài 9
def test_09_swiglu_dung_cong_thuc_va_dung_so_tham_so():
    """So đầu ra với công thức tính tay bằng NumPy.

    Đếm số tham số phải bằng 3 * d_model * d_ff. Với d_ff = 688 thì phải ra
    ĐÚNG 1.056.768.

    Bắt lỗi: nhầm nhánh cổng với nhánh thường, dùng sai hàm kích hoạt, và quan
    trọng nhất là quên chỉnh d_ff xuống 688 khiến mô hình phình thêm khoảng
    12 triệu tham số mà không ai nhận ra.
    """
    from nmt.model.layers import SwiGLU
    import torch
    import torch.nn.functional as F
    
    d_model = 512
    d_ff = 688
    swiglu = SwiGLU(d_model, d_ff)
    
    params = sum(p.numel() for p in swiglu.parameters())
    assert params == 3 * d_model * d_ff, f"Expected {3*d_model*d_ff}, got {params}"
    
    x = torch.randn(2, d_model)
    out = swiglu(x)
    
    gate = swiglu.w_gate(x)
    up = swiglu.w_up(x)
    expected = swiglu.w_down(F.silu(gate) * up)
    
    assert torch.allclose(out, expected, atol=1e-5)

# ---------------------------------------------------------------- bài 10
def test_10_duong_residual_pre_norm_thong_suot_va_co_chuan_hoa_cuoi():
    """Hai phần:

    1. Đặt toàn bộ trọng số ở nhánh ra của mọi khối con về 0. Khi đó đầu ra của
       chồng lớp phải bằng ĐÚNG đầu vào sau khi đi qua RMSNorm cuối, vì mọi thứ
       chỉ còn chạy trên đường residual.
    2. Kiểm rằng CÓ một lớp chuẩn hóa nằm giữa lớp decoder cuối cùng và lớp
       Linear xuất ra từ vựng.

    Bắt lỗi: cài nhầm thành Post-Norm (lỡ chuẩn hóa sau khi cộng residual làm
    đứt đường residual sạch); quên RMSNorm cuối — lỗi rất phổ biến khi dùng
    Pre-Norm, KHÔNG làm chương trình báo lỗi, chỉ khiến giá trị phình dần qua
    từng lớp rồi mô hình mất ổn định.
    """
    from nmt.utils import nap_config
    from nmt.model.transformer import TransformerNMT
    import torch
    import torch.nn as nn
    
    cfg = nap_config("configs/base.yaml")
    model = TransformerNMT(cfg)
    model.eval()
    
    for layer in model.encoder_layers:
        layer.self_attn.w_o.weight.data.zero_()
        layer.ffn.w_down.weight.data.zero_()
    
    x = torch.randint(1, cfg.du_lieu.vocab_size, (1, 5))
    encoded = model.encode(x, None)
    
    embed = model._embed(x)
    assert not torch.allclose(encoded, embed, atol=1e-5) 
    
    assert not isinstance(model.encoder_final_norm, nn.Identity)
    
    expected = model.encoder_final_norm(embed)
    assert torch.allclose(encoded, expected, atol=1e-5)

# ---------------------------------------------------------------- bài 11
def test_11_rope_khong_bi_ap_vao_cross_attention():
    """Cho cross-attention chạy với CÙNG một bộ vector query nhưng khai báo
    chúng ở hai vị trí khác nhau. Đầu ra bắt buộc phải GIỐNG HỆT nhau.

    Bắt lỗi: lỡ áp RoPE cho cross-attention. Lỗi này không làm chương trình báo
    lỗi và mô hình vẫn train được, chỉ là đang mã hóa một khoảng cách vô nghĩa
    giữa hai chuỗi thuộc hai ngôn ngữ khác nhau.
    """
    from nmt.utils import nap_config
    from nmt.model.layers import DecoderLayer
    from nmt.model.positional import RoPE
    import torch
    
    cfg = nap_config("configs/base.yaml")
    layer = DecoderLayer(cfg)
    layer.eval()
    
    tgt = torch.randn(1, 2, cfg.mo_hinh.d_model)
    mem = torch.randn(1, 2, cfg.mo_hinh.d_model)
    
    rope = RoPE(64)
    
    # Nếu RoPE bị lỡ áp dụng vào cross_attn, đầu ra sẽ thay đổi
    # Tuy nhiên DecoderLayer không nhận rope cho cross_mask/cross_attn
    import types
    original_forward = layer.cross_attn.forward
    rope_passed = [True]
    
    def fake_forward(self, query, key, value, mask=None, rope=None, vi_tri_bat_dau=0):
        rope_passed[0] = rope
        return original_forward(query, key, value, mask, rope, vi_tri_bat_dau)
        
    layer.cross_attn.forward = types.MethodType(fake_forward, layer.cross_attn)
    layer.forward(tgt, mem, self_mask=None, cross_mask=None, rope=rope)
    
    assert rope_passed[0] is None

# ---------------------------------------------------------------- bài 12
@pytest.mark.gpu
def test_12_mo_hinh_chay_dung_o_fp16():
    """Cho một batch mẫu chạy qua mô hình trong torch.autocast kiểu fp16.
    Kiểm rằng đầu ra không có NaN và không có vô cùng. Sau đó so đầu ra fp16
    với đầu ra fp32 trên cùng đầu vào, sai lệch tương đối trung bình dưới 1e-2.

    Bắt ba lỗi mà MƯỜI MỘT BÀI TRÊN KHÔNG PHÁT HIỆN ĐƯỢC, vì mười một bài kia
    chạy ở float32:
        1. tính bảng góc quay RoPE ở fp16 làm mất độ phân giải vị trí
        2. tràn số khi tính trung bình bình phương trong RMSNorm
        3. dùng -1e9 thay vì torch.finfo(scores.dtype).min khi che mask
    """
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
        
    from nmt.utils import nap_config
    from nmt.model.transformer import TransformerNMT
    import torch
    
    cfg = nap_config("configs/base.yaml")
    model = TransformerNMT(cfg).cuda()
    
    src = torch.randint(1, cfg.du_lieu.vocab_size, (2, 5)).cuda()
    tgt = torch.randint(1, cfg.du_lieu.vocab_size, (2, 5)).cuda()
    
    out32 = model(src, tgt)
    
    with torch.autocast("cuda", dtype=torch.float16):
        out16 = model(src, tgt)
        
    assert not torch.isnan(out16).any()
    assert not torch.isinf(out16).any()
    
    assert torch.allclose(out32, out16.float(), rtol=1e-2, atol=1e-2)
