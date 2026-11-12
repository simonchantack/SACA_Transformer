"""Search space of protocol P2: sampling, feasibility and the mapping from a draw to (model config, L, recipe).

One file for every model (our model, the three external baselines, LSTM, ridge): the draws come from
``configs/protocols/protocol_P2*.yaml`` in YAML order, so the Optuna ``RandomSampler`` stream is the one that
produced the frozen selections.
"""
from __future__ import annotations

import math
from typing import Any

R3_FAMILIES = ("r3_D1", "r3_D2", "r3_D3", "r3_D4")
FAMILIES = ("dualdim", "patchtst", "itransformer", "crossformer", "lstm", "ridge") + R3_FAMILIES


class Infeasible(ValueError):
    """A sampled configuration that cannot be built (rejected at trial start, zero GPU cost)."""


class InfeasibleConfig(Infeasible):
    """Raised by the model builders for configurations that cannot be built."""


def space_family(family: str) -> str:
    """Every D1-D4 design samples the ``dualdim`` space (same RandomSampler stream as theta*)."""
    return "dualdim" if family in R3_FAMILIES else family


def is_r3(family: str) -> bool:
    return family in R3_FAMILIES


def round_half_up(x: float) -> int:
    """Strides use half-up rounding (2.5 -> 3); Python's ``round`` is banker's rounding."""
    return int(math.floor(float(x) + 0.5))


def n_patches(L: int, p: int, s: int) -> int:
    return (L - p) // s + 1 if L >= p else 0


def suggest(trial: Any, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sample every entry of ``space`` with the Optuna trial, in YAML order."""
    out: dict[str, Any] = {}
    for name, s in space.items():
        t = s["type"]
        if t == "categorical":
            out[name] = trial.suggest_categorical(name, list(s["choices"]))
        elif t == "float":
            out[name] = trial.suggest_float(name, float(s["low"]), float(s["high"]), log=bool(s.get("log", False)))
        elif t == "int":
            out[name] = trial.suggest_int(name, int(s["low"]), int(s["high"]), log=bool(s.get("log", False)))
        elif t == "fixed":
            out[name] = s["value"]
        else:
            raise ValueError(f"unknown space type {t} for {name}")
    return out


def materialize(family: str, params: dict[str, Any], C: int, fixed: dict[str, Any] | None = None,
                bypass_buildability: bool = False, bypass_L: int | None = None
                ) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Map sampled params to ``(model_cfg dict, L, recipe dict)``. Raises ``Infeasible`` for unbuildable draws.

    ``bypass_buildability`` (D4 window arm, L = 30 on FD002-FD004 only): skip the P2 ``N_t >= 2`` check, which the
    end-padded D4 does not need (it has N' >= 2). Every other check runs at the frozen L = 50.

    Shared dimensions (all families): L, lr, weight_decay, batch_size."""
    fixed = dict(fixed or {})
    p = dict(params)
    L = int(p.pop("L"))
    recipe = {"lr": float(p.pop("lr")), "weight_decay": float(p.pop("weight_decay")),
              "batch_size": int(p.pop("batch_size"))}
    if family == "lstm":
        cfg_l = {"hidden": int(p.pop("hidden")), "layers": int(p.pop("layers")), "dropout": float(p.pop("dropout"))}
        if p:
            raise ValueError(f"unexpected lstm params {sorted(p)}")
        cfg_l.update(fixed)
        return cfg_l, L, recipe
    if space_family(family) == "dualdim":
        pt = int(p["patch_t"])
        st = max(1, round_half_up(float(p["r_t"]) * pt))
        nt = n_patches(L, pt, st)
        if nt < 2:
            if bypass_buildability and family == "r3_D4" and bypass_L is not None and L == int(bypass_L):
                # every other check at the frozen L = 50, then the frozen L is put back
                cfg50, _, rec50 = materialize(family, dict(params, L=50), C, fixed)
                return dict(cfg50, L=L), L, rec50
            raise Infeasible(f"N_t = {nt} < 2 (L {L}, patch_t {pt}, stride_t {st})")
        pc, sc = int(p["patch_c"]), int(p["stride_c"])
        if pc > C:
            raise Infeasible(f"patch_c {pc} > C {C}")
        if sc > pc:
            raise Infeasible(f"stride_c {sc} > patch_c {pc}")
        d, h = int(p["d_model"]), int(p["n_heads"])
        if d % h:
            raise Infeasible(f"n_heads {h} does not divide d_model {d}")
        cfg = dict(L=L, patch_len_t=pt, stride_t=st, patch_len_c=pc, stride_c=sc,
                   d_model_t=d, n_heads_t=h, n_layers_t=int(p["n_layers_t"]), d_ff_t=int(p["d_ff"]),
                   dropout_t=float(p["dropout_t"]),
                   d_model_c=d, n_heads_c=h, n_layers_c=int(p["n_layers_c"]), d_ff_c=int(p["d_ff"]),
                   dropout_c=float(p["dropout_c"]), head_hidden=int(p["head_hidden"]))
        cfg.update(fixed)
        return cfg, L, recipe
    cfg = dict(p)
    if family == "patchtst":
        stride = max(1, round_half_up(float(cfg["r"]) * int(cfg["patch_len"])))
        npch = n_patches(L, int(cfg["patch_len"]), stride)
        if npch < 2:
            raise Infeasible(f"num_patch = {npch} < 2")
    elif family == "crossformer":
        if int(cfg["seg_len"]) > L:
            raise Infeasible(f"seg_len {cfg['seg_len']} > L {L}")
    elif family not in ("itransformer", "ridge"):
        raise ValueError(family)
    if family != "ridge" and int(cfg["d_model"]) % int(cfg["n_heads"]):
        raise Infeasible("n_heads does not divide d_model")
    cfg.update(fixed)
    return cfg, L, recipe


def resolve_space(protocol: dict[str, Any], family: str, dry: bool = False
                  ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Shared dimensions (batch choices from the budget option) followed by the family dimensions."""
    opt = protocol["options"][protocol["budget_option"]]
    space: dict[str, dict[str, Any]] = {}
    for k, v in protocol["shared_space"].items():
        v = dict(v)
        if v.get("choices") == "OPTION":
            v["choices"] = list(opt["batch_choices"])
        space[k] = v
    sf = space_family(family)
    space.update({k: dict(v) for k, v in protocol["spaces"][sf].items()})
    if dry:
        space.update(protocol["dry_run"].get("space_override", {}).get(sf, {}))
    return space, opt
