"""Unit tests of the attention modules: shapes, masking, gradient flow, and the corrected behaviour (R0.1, R0.2, R0.19)."""
from __future__ import annotations

import pytest
import torch

from saca_rul.attention import ExactAttention, MultiheadSelfAttention, ProbAttention, ProbMask, set_probsparse


def qkv(B: int = 2, L: int = 12, H: int = 4, D: int = 8, seed: int = 0) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, L, H, D, generator=g, requires_grad=True) for _ in range(3))


# ----------------------------------------------------------------------------- shapes
@pytest.mark.parametrize("cls", [ProbAttention, ExactAttention])
@pytest.mark.parametrize("L", [1, 7, 40, 150])
def test_core_output_shape(cls: type, L: int) -> None:
    q, k, v = qkv(L=L)
    m = cls(mask_flag=False) if cls is ProbAttention else cls()
    out, attn = m(q, k, v)
    assert out.shape == q.shape
    assert attn is None


@pytest.mark.parametrize("kind", ["prob", "exact"])
def test_multihead_shape(kind: str) -> None:
    mha = MultiheadSelfAttention(32, 4, kind=kind)
    x = torch.randn(3, 20, 32)
    out, _ = mha(x, x, x)
    assert out.shape == (3, 20, 32)


def test_attention_weights_shape_and_rows_sum_to_one() -> None:
    q, k, v = qkv(L=9)
    out, attn = ExactAttention(output_attention=True)(q, k, v)
    assert attn is not None and attn.shape == (2, 4, 9, 9)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 4, 9), atol=1e-5)
    pa = ProbAttention(mask_flag=False, output_attention=True)
    _, a2 = pa(q, k, v)
    assert a2 is not None and a2.shape == (2, 4, 9, 9)


def test_exact_matches_reference_softmax() -> None:
    q, k, v = qkv(L=10)
    out, _ = ExactAttention()(q, k, v)
    qq, kk, vv = (t.transpose(1, 2) for t in (q, k, v))
    ref = torch.softmax(qq @ kk.transpose(-2, -1) / 8 ** 0.5, -1) @ vv
    assert torch.allclose(out.transpose(1, 2), ref, atol=1e-5)


# ----------------------------------------------------------------------------- R0.1: attention runs over tokens
def test_tokens_mix_across_the_sequence() -> None:
    """Changing another token changes token 0's output (the submitted wrapper made this change 3e-8)."""
    torch.manual_seed(0)
    mha = MultiheadSelfAttention(32, 8, kind="prob").eval()
    x = torch.randn(1, 20, 32)
    y0 = mha(x, x, x)[0][0, 0]
    x2 = x.clone()
    x2[:, 1:] = torch.randn(1, 19, 32)
    y1 = mha(x2, x2, x2)[0][0, 0]
    assert (y0 - y1).abs().max() > 1e-3


# ----------------------------------------------------------------------------- R0.2: no batch-row leakage
@pytest.mark.parametrize("kind", ["prob", "exact"])
def test_prediction_is_independent_of_other_batch_members(kind: str) -> None:
    torch.manual_seed(0)
    mha = MultiheadSelfAttention(32, 4, kind=kind).eval()
    x = torch.randn(6, 25, 32)
    full = mha(x, x, x)[0]
    alone = torch.stack([mha(x[i:i + 1], x[i:i + 1], x[i:i + 1])[0][0] for i in range(6)])
    assert torch.allclose(full, alone, atol=1e-5)
    x2 = x.clone()
    x2[0] = torch.randn(25, 32)
    assert torch.allclose(mha(x2, x2, x2)[0][1:], full[1:], atol=1e-5)


def test_prob_eval_is_deterministic_and_ignores_global_rng() -> None:
    mha = MultiheadSelfAttention(32, 4, kind="prob").eval()
    x = torch.randn(2, 200, 32)  # long enough that u < L (sparse path)
    torch.manual_seed(1)
    a = mha(x, x, x)[0]
    torch.manual_seed(2)
    b = mha(x, x, x)[0]
    assert torch.equal(a, b)


def test_prob_train_uses_global_rng() -> None:
    mha = MultiheadSelfAttention(32, 4, kind="prob").train()
    x = torch.randn(2, 200, 32)
    torch.manual_seed(1)
    a = mha(x, x, x)[0]
    torch.manual_seed(2)
    b = mha(x, x, x)[0]
    assert not torch.equal(a, b)


# ----------------------------------------------------------------------------- masking
def test_masks_are_rejected_by_the_wrapper() -> None:
    mha = MultiheadSelfAttention(16, 2)
    x = torch.randn(1, 5, 16)
    with pytest.raises(NotImplementedError):
        mha(x, x, x, attn_mask=torch.zeros(5, 5))
    with pytest.raises(NotImplementedError):
        mha(x, x, x, key_padding_mask=torch.zeros(1, 5, dtype=torch.bool))
    with pytest.raises(NotImplementedError):
        ExactAttention()(*qkv(), attn_mask=torch.zeros(1))


def test_prob_mask_is_strictly_upper_triangular_on_selected_rows() -> None:
    B, H, L = 1, 2, 6
    index = torch.tensor([[[0, 2, 5], [1, 3, 4]]])
    scores = torch.zeros(B, H, 3, L)
    mask = ProbMask(B, H, L, index, scores).mask
    assert mask.shape == scores.shape
    for h in range(H):
        for j, row in enumerate(index[0, h].tolist()):
            assert mask[0, h, j].tolist() == [c > row for c in range(L)]


def test_causal_probattention_does_not_see_the_future() -> None:
    """mask_flag=True path (not used by any encoder): the output at position i depends only on tokens <= i."""
    torch.manual_seed(0)
    pa = ProbAttention(mask_flag=True, exact_when_full=True).eval()
    q, k, v = (torch.randn(1, 8, 2, 4) for _ in range(3))
    o1, _ = pa(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, 7] += 5.0
    v2[:, 7] += 5.0
    o2, _ = pa(q, k2, v2)
    assert torch.allclose(o1[:, :7], o2[:, :7], atol=1e-5)


# ----------------------------------------------------------------------------- gradient flow
@pytest.mark.parametrize("kind", ["prob", "exact"])
def test_gradients_reach_all_parameters_and_inputs(kind: str) -> None:
    torch.manual_seed(0)
    mha = MultiheadSelfAttention(32, 4, kind=kind).train()
    x = torch.randn(3, 20, 32, requires_grad=True)
    mha(x, x, x)[0].square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name, p in mha.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


@pytest.mark.parametrize("cls", [ProbAttention, ExactAttention])
def test_core_gradients_flow_to_q_k_v(cls: type) -> None:
    q, k, v = qkv(L=16)
    m = cls(mask_flag=False) if cls is ProbAttention else cls()
    m(q, k, v)[0].sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and t.grad.abs().sum() > 0


def test_set_probsparse_counts_modules() -> None:
    from saca_rul.encoder import encoder_stack

    st = encoder_stack(16, 2, 32, 0.1, 3, attention="prob")
    assert set_probsparse(st, eval_seed=5, exact_when_full=False) == 3
    assert all(m.eval_seed == 5 for m in st.modules() if isinstance(m, ProbAttention))
