"""CPU shape script for D4 (R1.3, R1.31, R0.11): patch-count formula, numeric tensor shapes and token counts.

It fits the training pipeline on TR of each split seed (training file only, to get the kept-sensor count C), builds
D4 at the frozen configuration of exp_B16 (and the exp_A12 L = 30 / 40 cells), pushes a zero tensor through it with
forward hooks, and prints what it measured. No training, no GPU, no test file, no result: the output is structural.

Patch count with PatchTST end padding (submitted Eq. 10 has no padding):
    N  = floor((L - P) / S) + 1      (submitted code: newest (L - P) mod S cycles dropped)
    N' = floor((L - P) / S) + 2      (D4: S copies of the last value appended)
D4 attention operation counts per layer and sample (score entries x d): time stage C (N'+1)^2 d, sensor stage C^2 d.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from . import model as M
from .config import ROOT, load_config, resolve_path
from .data import DataCache
from .embeddings import n_patches, n_patches_padded, sensor_ids_from_features
from .runner import selection_for
from .search_space import materialize

CELLS = [("exp_B16_d4_main.yaml", "exp_B16 (frozen, L=50)"), ("exp_A12_L40.yaml", "exp_A12 L=40"),
         ("exp_A12_L30.yaml", "exp_A12 L=30")]


def shapes_for(cfg_file: str, subset: str, seed: int, batch: int | None = None) -> dict[str, Any]:
    cfg = load_config(ROOT / "configs" / cfg_file)
    ent, _, _ = selection_for(cfg, subset)
    paths = cfg["paths"]
    arm = cfg.get("arm") or {}
    cache = DataCache(subset, seed, paths["data_dir"], paths["splits"])
    from .search_space import resolve_space  # noqa: F401  (protocol-driven fixed block)
    from .training import load_protocol
    pr = load_protocol(cfg["protocol"])
    mcfg, L, rec = materialize(cfg["family"], ent["params"], cache.C, pr["fixed"].get("dualdim"),
                               bypass_buildability=bool(arm.get("bypass_buildability")), bypass_L=arm.get("bypass_L"))
    mc = M.resolve(mcfg, C=cache.C, L=L)
    model: Any = M.build_model(mc, "D4", sensor_ids_from_features(cache.features)).eval()
    B = int(batch or rec["batch_size"])
    P, S, d, C = mc.patch_len_t, mc.stride_t, mc.d_model_t, mc.C
    Np = n_patches_padded(L, P, S)
    seen: dict[str, tuple[int, ...]] = {}
    stages: list[dict[str, Any]] = []
    hooks = []

    def rec_stage(stage: str, tensor: str, shape: tuple[int, ...] | list[int], kind: str = "measured") -> None:
        stages.append({"stage": stage, "tensor": tensor, "shape": [int(v) for v in shape], "kind": kind})

    def hook(name: str) -> Any:
        def fn(_m: Any, _i: Any, out: Any) -> None:
            seen[name] = tuple(out.shape)
        return fn

    hooks.append(model.patch_embed.register_forward_hook(hook("patch_tokens_p0 (B*C, N', d)")))
    hooks.append(model.time_layers[0].register_forward_hook(hook("time_stage_out (B*C, N'+1, d)")))
    hooks.append(model.sensor_layers[0].register_forward_hook(hook("sensor_stage_out (B, C, d)")))
    hooks.append(model.head.register_forward_hook(hook("head_out (B,)")))

    def shape_of(t: Any) -> tuple[int, ...]:
        return tuple((t[0] if isinstance(t, tuple) else t).shape)

    def add_pre(mod: Any, stage: str, tensor: str, scores: tuple[Any, ...] | None = None) -> None:
        def fn(_m: Any, i: Any) -> None:
            rec_stage(stage, tensor, shape_of(i[0]))
            if scores is not None:
                rec_stage(stage, "attention scores per head, " + str(scores[0]), scores[1:], "derived")

        hooks.append(mod.register_forward_pre_hook(fn))

    def add_out(mod: Any, stage: str, tensor: str) -> None:
        hooks.append(mod.register_forward_hook(lambda _m, _i, o: rec_stage(stage, tensor, shape_of(o))))

    # Stage order follows SACAEncoder.initial_tokens / forward; forward hooks fire in execution order.
    # "derived" rows are computed from the measured shapes (attention scores are not materialised as module outputs).
    rec_stage("0 input", "x (B, C, L)", (B, C, L))
    add_out(model.sensor_emb, "1 anchor init", "sensor-id embedding E[id] (C, d)")
    add_out(model.stats_proj, "1 anchor init", "window statistics projection W_s [mu, log sigma] (B, C, d)")
    add_out(model.inp, "1 anchor init", "W_in x (B, C, d)")
    add_pre(model.patch_embed, "2 patching", "instance-normalised window, rows = (b, sensor) (B*C, L)")
    add_pre(model.patch_embed.proj, "2 patching", "after S end-padding copies and unfold (P, S): (B*C, N', P)")
    add_out(model.patch_embed.proj, "3 projection", "Linear(P, d) patch tokens (B*C, N', d)")
    add_out(model.pos_enc, "3 projection", "+ sinusoidal PE = p^0 (B*C, N', d)")
    for li in range(len(model.time_layers)):
        tl, sl = model.time_layers[li], model.sensor_layers[li]
        add_pre(tl, f"4 time stage, layer {li + 1}", "[anchor ; patches] per sensor (B*C, N'+1, d)",
                ("(B*C, heads_t, N'+1, N'+1)", B * C, mc.n_heads_t, Np + 1, Np + 1))
        add_out(tl.self_attn, f"4 time stage, layer {li + 1}", "attention output (B*C, N'+1, d)")
        add_out(tl, f"4 time stage, layer {li + 1}", "layer output (B*C, N'+1, d)")
        add_pre(sl, f"5 sensor stage, layer {li + 1}", "anchors only (B, C, d)",
                ("(B, heads_c, C, C)", B, mc.n_heads_c, C, C))
        add_out(sl.self_attn, f"5 sensor stage, layer {li + 1}", "attention output (B, C, d)")
        add_out(sl, f"5 sensor stage, layer {li + 1}", "layer output = new anchors (B, C, d)")
    add_pre(model.head.mlp[0], "6 head", "sensors in id order, flattened (B, C*d)")
    add_out(model.head, "6 head", "RUL prediction (B,)")
    with torch.no_grad():
        y = model(torch.zeros(B, C, L))
    for h in hooks:
        h.remove()
    return {"subset": subset, "seed": seed, "L": L, "C": C, "P_t": P, "S_t": S, "d_model": d, "K_layers": mc.n_layers_t,
            "heads": mc.n_heads_t, "heads_c": mc.n_heads_c, "batch": B, "N_no_padding": n_patches(L, P, S),
            "N_prime_end_padding": Np,
            "time_tokens_per_sensor_incl_anchor": Np + 1, "sensor_stage_tokens": C,
            "input": (B, C, L), "output": tuple(y.shape), "measured_shapes": seen,
            "attn_ops_time_per_layer_sample": C * (Np + 1) ** 2 * d, "attn_ops_sensor_per_layer_sample": C ** 2 * d,
            "n_params": M.n_params(model), "stages": stages}


def stage_tables(out: list[dict[str, Any]]) -> list[str]:
    """Markdown: one stage-by-stage tensor-shape table per (cell, subset) at seed 0, from the forward hooks."""
    lines = ["", "## Stage-by-stage tensor shapes (R1.31)", "",
             "Measured with forward hooks on the instantiated D4 (CPU, zero input, no training, no test file). "
             "B = batch, C = kept sensors, L = window, P = S = patch length and stride, N' = floor((L-P)/S) + 2 patches "
             "with end padding, d = d_model, K = layers, heads_t / heads_c = attention heads of the time / sensor stage. "
             "Rows marked `derived` are computed from the measured shapes (the attention score tensors are not module "
             "outputs). The time stage runs per sensor on [anchor ; N' patch tokens]; the sensor stage runs on the C "
             "anchors only. Cells whose kept-sensor count differs between seeds are listed under each table.", ""]
    for cell in dict.fromkeys(r["cell"] for r in out):
        for sub in ("FD001", "FD002", "FD003", "FD004"):
            recs = [r for r in out if r["cell"] == cell and r["subset"] == sub]
            r0 = next(r for r in recs if r["seed"] == 0)
            lines += [f"### {cell}, {sub}, seed 0", "",
                      f"B = {r0['batch']}, C = {r0['C']}, L = {r0['L']}, P = S = {r0['P_t']}, "
                      f"N' = {r0['N_prime_end_padding']}, d = {r0['d_model']}, K = {r0['K_layers']}, "
                      f"heads_t = {r0['heads']}, heads_c = {r0['heads_c']}", "",
                      "| stage | tensor | shape | source |", "|---|---|---|---|"]
            for st in r0["stages"]:
                shape = ", ".join(str(v) for v in st["shape"]) + ("," if len(st["shape"]) == 1 else "")
                lines.append(f"| {st['stage']} | {st['tensor']} | ({shape}) | {st['kind']} |")
            other = [f"seed {r['seed']}: C = {r['C']}" for r in recs if r["C"] != r0["C"]]
            if other:
                lines += ["", "Other kept-sensor counts: " + "; ".join(other) + " (shapes in `d4_shapes.json`)."]
            lines.append("")
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(resolve_path("results/shapes")))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    out: list[dict[str, Any]] = []
    for cfg_file, label in CELLS:
        for sub in ("FD001", "FD002", "FD003", "FD004"):
            for s in seeds:
                r = shapes_for(cfg_file, sub, s)
                r["cell"] = label
                out.append(r)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "d4_shapes.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = ["| cell | subset | seed | C | L | P=S | N (no pad) | N' (end pad) | tokens/sensor (N'+1) | d | K | heads | batch "
             "| time-stage ops | sensor-stage ops | params |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['cell']} | {r['subset']} | {r['seed']} | {r['C']} | {r['L']} | {r['P_t']}/{r['S_t']} | "
                     f"{r['N_no_padding']} | {r['N_prime_end_padding']} | {r['time_tokens_per_sensor_incl_anchor']} | "
                     f"{r['d_model']} | {r['K_layers']} | {r['heads']} | {r['batch']} | "
                     f"{r['attn_ops_time_per_layer_sample']} | {r['attn_ops_sensor_per_layer_sample']} | {r['n_params']} |")
    lines += stage_tables(out)
    (od / "d4_shapes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for r in out[:1] + [x for x in out if x["cell"].endswith("L=30") and x["subset"] != "FD001"][:1]:
        print(json.dumps({k: r[k] for k in ("cell", "subset", "input", "measured_shapes", "output")}))


if __name__ == "__main__":
    main()
