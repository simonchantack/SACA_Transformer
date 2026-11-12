"""Unit tests: embeddings, encoder layers, heads, theta*, D1-D4, baselines (shapes, order invariance, gradients)."""
from __future__ import annotations

import pytest
import torch

from saca_rul import baselines as B
from saca_rul import model as M
from saca_rul.embeddings import (
    EndPaddedPatchEmbedding,
    PatchEmbedding,
    n_patches,
    n_patches_padded,
    sensor_ids_from_features,
    window_stats,
)
from saca_rul.encoder import ChannelLayerNorm, TransformerBatchNormEncoderLayer
from saca_rul.search_space import Infeasible, materialize

C, L = 6, 30
SIDS = (2, 3, 4, 7, 11, 12)


def cfg(**kw: object) -> M.ModelConfig:
    base = dict(C=C, L=L, patch_len_t=10, stride_t=10, patch_len_c=3, stride_c=1, d_model_t=16, n_heads_t=2,
                n_layers_t=2, d_ff_t=32, dropout_t=0.1, d_model_c=16, n_heads_c=2, n_layers_c=2, d_ff_c=32,
                dropout_c=0.1, head_hidden=16)
    base.update(kw)
    return M.ModelConfig(**base)  # type: ignore[arg-type]


def test_patch_count_formulas() -> None:
    assert n_patches(50, 15, 15) == 3 and n_patches_padded(50, 15, 15) == 4
    assert n_patches(50, 20, 20) == 2 and n_patches_padded(50, 20, 20) == 3
    assert n_patches(30, 20, 20) == 1 and n_patches_padded(30, 20, 20) == 2
    assert n_patches(10, 20, 5) == 0


def test_patch_embeddings() -> None:
    x = torch.randn(4, 50)
    assert PatchEmbedding(15, 15, 8)(x).shape == (4, 3, 8)
    pe = EndPaddedPatchEmbedding(15, 15, 8)
    assert pe(x).shape == (4, 4, 8)
    p = pe.patches(x)
    assert torch.equal(p[:, -1, 5:], x[:, -1:].expand(4, 10))  # padding repeats the last value
    assert torch.equal(p[:, -1, :5], x[:, 45:])  # newest cycle lies in the last patch
    with pytest.raises(ValueError):
        PatchEmbedding(60, 1, 8)(x)


def test_window_stats_equals_instance_norm() -> None:
    x = torch.randn(3, 5, 40)
    xt, mu, sigma = window_stats(x)
    ref = torch.nn.InstanceNorm1d(1, affine=False, eps=1e-6)(x.reshape(15, 1, 40)).reshape(3, 5, 40)
    assert torch.allclose(xt, ref, atol=1e-5)
    assert mu.shape == sigma.shape == (3, 5)


def test_sensor_ids() -> None:
    assert sensor_ids_from_features(["sm_2", "sm_11"]) == (2, 11)
    with pytest.raises(ValueError):
        sensor_ids_from_features(["op_setting_1"])
    with pytest.raises(ValueError):
        sensor_ids_from_features(["sm_2", "sm_2"])


@pytest.mark.parametrize("norm", ["batch", "layer"])
def test_encoder_layer_shape_and_grad(norm: str) -> None:
    layer = TransformerBatchNormEncoderLayer(16, 2, 32, 0.1, attention="exact", norm=norm).train()
    x = torch.randn(4, 7, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.square().sum().backward()
    assert all(p.grad is not None for p in layer.parameters())
    assert ("running_mean" in layer.state_dict().__repr__()) == (norm == "batch")


def test_layernorm_normalises_per_token() -> None:
    ln = ChannelLayerNorm(8)
    z = torch.randn(5, 8, 3)  # (S, d, B)
    y = ln(z)
    assert torch.allclose(y.mean(1), torch.zeros(5, 3), atol=1e-5)
    with pytest.raises(ValueError):
        ln(torch.randn(5, 7, 3))


@pytest.mark.parametrize("design", ["theta", "D1", "D2", "D3", "D4"])
def test_design_output_shape_and_gradients(design: str) -> None:
    torch.manual_seed(0)
    m = M.build_model(cfg(), design, SIDS).train()
    x = torch.randn(8, C, L)
    y = m(x)
    assert y.shape == (8,)
    y.square().mean().backward()
    bad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, bad


def test_theta_has_probsparse_and_d1_does_not() -> None:
    assert M.count_prob_attention(M.build_model(cfg(), "theta")) == 4
    assert M.count_prob_attention(M.build_model(cfg(), "D1")) == 0
    th, d1 = M.build_model(cfg(), "theta"), M.build_model(cfg(), "D1")
    assert list(th.state_dict()) == list(d1.state_dict())  # same keys


def test_d4_token_shapes() -> None:
    m = M.build_model(cfg(), "D4", SIDS)
    a, p = m.initial_tokens(torch.randn(5, C, L))  # type: ignore[operator]
    assert a.shape == (5, C, 16) and p.shape == (5 * C, n_patches_padded(L, 10, 10), 16)
    assert m.encode(torch.randn(5, C, L)).shape == (5, C, 16)  # type: ignore[operator]


def test_d4_is_invariant_to_sensor_order() -> None:
    """Permuting rows together with their sensor ids leaves the output unchanged (R2.16, R1.39), float64."""
    torch.manual_seed(0)
    m = M.build_model(cfg(), "D4", SIDS).double().eval()
    x = torch.randn(4, C, L, dtype=torch.float64)
    perm = [3, 0, 5, 1, 4, 2]
    m2 = M.build_model(cfg(), "D4", tuple(SIDS[i] for i in perm)).double().eval()
    m2.load_state_dict(m.state_dict())
    assert torch.allclose(m(x), m2(x[:, perm]), atol=1e-9)


def test_d4_eval_is_batch_independent() -> None:
    torch.manual_seed(0)
    m = M.build_model(cfg(), "D4", SIDS).eval()
    x = torch.randn(6, C, L)
    full = m(x)
    single = torch.cat([m(x[i:i + 1]) for i in range(6)])
    assert torch.allclose(full, single, atol=1e-5)


def test_d4_requires_matching_widths_and_ids() -> None:
    with pytest.raises(ValueError):
        M.build_model(cfg(d_model_c=8), "D4", SIDS)
    with pytest.raises(ValueError):
        M.build_model(cfg(), "D4", None)
    with pytest.raises(ValueError):
        M.build_model(cfg(), "D4", (1, 2, 3))


def test_checkpoint_roundtrip(tmp_path: object) -> None:
    from pathlib import Path

    p = Path(str(tmp_path)) / "ck.pt"
    for design in ("theta", "D4"):
        m = M.build_model(cfg(), design, SIDS).eval()
        M.save_checkpoint(p, m, {"lr": 1e-3}, {"features": ["sm_2"]})
        m2, meta = M.load_checkpoint(p)
        x = torch.randn(3, C, L)
        assert torch.allclose(m(x), m2(x), atol=1e-6)
        assert meta["design"] == design


def test_materialize_rules() -> None:
    p = {"L": 30, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 64, "d_model": 32, "n_heads": 4, "n_layers_t": 1,
         "n_layers_c": 1, "d_ff": 64, "head_hidden": 64, "dropout_t": 0.1, "dropout_c": 0.1, "patch_t": 20,
         "r_t": 1.0, "patch_c": 5, "stride_c": 2}
    with pytest.raises(Infeasible):  # N_t = 1 < 2
        materialize("r3_D4", p, 14)
    cfg_b, L_b, _ = materialize("r3_D4", p, 14, bypass_buildability=True, bypass_L=30)
    assert L_b == 30 and cfg_b["L"] == 30
    with pytest.raises(Infeasible):  # the bypass is for D4 at the named L only
        materialize("dualdim", p, 14, bypass_buildability=True, bypass_L=30)
    with pytest.raises(Infeasible):
        materialize("r3_D4", dict(p, L=50, patch_c=9), 5)  # patch_c > C


@pytest.mark.parametrize("name,cfg_d", [
    ("patchtst", dict(patch_len=8, r=0.5, d_model=16, n_heads=4, e_layers=1, d_ff=32, dropout=0.1, revin=True)),
    ("itransformer", dict(d_model=32, m=2, e_layers=1, n_heads=4, dropout=0.1, use_norm=True)),
    ("crossformer", dict(seg_len=5, win_size=2, factor=5, d_model=16, n_heads=2, e_layers=2, dropout=0.1)),
    ("lstm", dict(hidden=16, layers=2, dropout=0.1)),
    ("ridge", dict(alpha=1.0)),
])
def test_baselines_shape_and_gradient(name: str, cfg_d: dict[str, object]) -> None:
    m = B.build_baseline(name, cfg_d, C, L)
    x = torch.randn(5, C, L)
    y = m(x)
    assert y.shape == (5,)
    if name != "ridge":
        y.square().mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_lstm_parameter_count_formula() -> None:
    m = B.build_baseline("lstm", dict(hidden=16, layers=2, dropout=0.1), C, L)
    assert sum(p.numel() for p in m.parameters()) == B.lstm_n_params(C, 16, 2)
