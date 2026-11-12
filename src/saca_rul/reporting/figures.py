"""Figures for the response to reviewers and the paper (R1.23): PNG at 300 dpi and PDF, one function per figure.

Port of ``docs/assets/make_figs_verified_v01.py``, ``make_figs_v03.py`` and ``make_figs_v04.py``. Data access is unchanged:
every number is read from a frozen artifact (named in each figure) or is a formula; figures marked ILLUSTRATIVE use
synthetic toy data and say so in the figure itself. Planning graphics (GPU-budget projections) are not regenerated.

Layout (fix attempt 1, R1.23): every figure is drawn at its final print size, 3.5 in (one ESWA column) or 7.2 in (two
columns), so that nothing is scaled when it is placed in the paper. All text is at least 7 pt at that size; ``save``
refuses to write a figure that is wider than 7.2 in or has any visible text below 7 pt (tick labels and offset text included;
mathtext super/subscripts count at 70 % of their nominal size, so log axes use plain decimal labels). The ``fig_r04_4`` padding bar is
read from the shape-script output (``results/shapes/d4_shapes.json``): its length is ``N' * S - L`` cycles.

``python -m saca_rul.reporting.figures --out-dir figures`` writes every figure; ``--only NAME`` writes one.
"""
import argparse
import json
import os
import textwrap
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from scipy import stats  # noqa: E402

from ..config import resolve_path  # noqa: E402

WS = str(resolve_path("${SACA_WORKSPACE:-../workspace}"))
OUT = str(resolve_path("figures"))
SHAPES_JSON = str(resolve_path("results/shapes/d4_shapes.json"))
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
BLUE, ORANGE, SKY, GREY = "#0072B2", "#E69F00", "#56B4E9", "#666666"  # Okabe-Ito; no red/green pairs (RULES section 37)
DARK_ORANGE = "#8F5A00"
SUBSET_STYLE = [(BLUE, "o", "-"), (ORANGE, "s", "--"), (SKY, "^", ":"), (GREY, "D", "-.")]  # colour, marker, line style per subset
SUB = SUBSETS
W1, W2 = 3.5, 7.2  # one column, two columns (inches)
MIN_PT = 7.0
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.titlesize": 8, "figure.dpi": 130, "pdf.fonttype": 42, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
})


def load(rel):
    with open(os.path.join(WS, rel), encoding="utf-8") as f:
        return json.load(f)


def wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.fill(line, width) for line in s.split("\n"))


def new_fig(width: float, height: float, **kw):
    return plt.subplots(figsize=(width, height), layout="constrained", **kw)


def effective_pt(t: Text) -> float:
    """Rendered size of a text object: mathtext super/subscripts are drawn at 70 % of the nominal size."""
    s = t.get_text()
    if "$" in s and any(c in s for c in "^_"):
        return float(t.get_fontsize()) * 0.7
    return float(t.get_fontsize())


def plain_log_axis(axis) -> None:
    """Plain decimal tick labels (100, 1,000, ...) on a log axis, so that no mathtext exponent (drawn at 70 %) is used."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def check_figure(fig, name: str) -> None:
    """Print-size gate: width at most 7.2 in and every visible text (tick labels, offset text, mathtext scripts) at least 7 pt."""
    w, _ = fig.get_size_inches()
    if w > W2 + 1e-6:
        raise ValueError(f"{name}: {w:.2f} in wide (limit {W2} in)")
    fig.canvas.draw()
    small = [(t.get_text()[:30], effective_pt(t)) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and effective_pt(t) < MIN_PT - 1e-6]
    if small:
        raise ValueError(f"{name}: text below {MIN_PT} pt: {small[:4]}")


def save(fig, name):
    """Write ``name``.png (300 dpi) and ``name``.pdf into the output folder (exact figure size, no cropping)."""
    check_figure(fig, name)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def _v4():
    v4 = load("experiments/final/combined_table_v4.json")
    rows = [("d4", "D4"), ("theta_star", "θ*"), ("patchtst", "PatchTST"), ("itransformer", "iTransformer"),
            ("crossformer", "Crossformer"), ("lstm", "LSTM"), ("ridge", "Ridge")]
    return v4, rows, v4["best_baseline_by_validation"]


def fig_v01_1_final_rmse() -> None:
    v4, ROWS, BK = _v4()
    fig, axes = new_fig(W2, 4.5, nrows=2, ncols=2)
    for ax, s in zip(axes.ravel(), SUBSETS):
        for i, (key, lab) in enumerate(ROWS):
            ps = v4["rows"][key]["per_subset"][s]["per_seed"]["RMSE"]
            col = "#0b6e99" if key == "d4" else ("#a15c00" if key == BK[s] else "#777777")
            ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), ps, s=9, color=col, alpha=0.8)
            ax.hlines(np.mean(ps), i - 0.3, i + 0.3, color=col, lw=1.8)
        ax.set_xticks(range(len(ROWS)))
        ax.set_xticklabels([r[1] for r in ROWS], rotation=35, ha="right")
        ax.set_title(f"{s} (B_k = {BK[s]})")
        ax.set_ylabel("test RMSE (cycles)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(wrap("Final rows, test set (combined_table_v4.json). Dots = split seeds 0-4, bar = mean. "
                      "Blue = D4, orange = validation-chosen best baseline B_k, grey = other rows.", 100))
    save(fig, "fig_v01_1_final_rmse")


def fig_v01_2_primary_pairs() -> None:
    v4, ROWS, BK = _v4()
    prim = v4["preregistered_primary_exp_B16"]
    ni = v4["non_inferiority_exp_B16"]
    pairs = prim["pairs"]
    fig, ax = new_fig(W2, 3.2)
    xs, ys, cs = [], [], []
    for j, p in enumerate(pairs):
        xs.append(j)
        ys.append(p["delta"])
        cs.append(BLUE if p["delta"] < 0 else ORANGE)  # blue: D4 lower; orange (hatched): D4 higher
    bars = ax.bar(xs, ys, color=cs, edgecolor="#333333", linewidth=0.4)
    for bar, y in zip(bars, ys):
        if y >= 0:
            bar.set_hatch("//")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(ni["margin_rmse"], color="#a15c00", ls="--", lw=1.2, label=f"non-inferiority margin +{ni['margin_rmse']}")
    ax.axhline(prim["mean_delta"], color="#0b6e99", ls=":", lw=1.2, label=f"mean Δ {prim['mean_delta']:+.2f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p['subset'][2:]}\ns{p['seed']}" for p in pairs])
    ax.set_ylabel("Δ test RMSE, D4 − B_k (cycles)")
    ax.set_title(wrap(f"Preregistered primary: {prim['n_dstar_better']}/20 pairs D4 lower; exact Wilcoxon p = "
                      f"{prim['wilcoxon_exact_two_sided_p']:.4f} → {prim['outcome']}; NI one-sided p = "
                      f"{ni['p_one_sided']:.5f}", 95))
    ax.legend(loc="lower left")
    save(fig, "fig_v01_2_primary_pairs")


def fig_v01_3_hier_bootstrap() -> None:
    hb = load("experiments/final/posthoc_R2.10/hier_bootstrap_results.json")
    COMP = [("theta_star", "θ*"), ("patchtst", "PatchTST"), ("itransformer", "iTransformer"),
            ("crossformer", "Crossformer"), ("lstm", "LSTM"), ("ridge", "Ridge"), ("best_baseline", "B_k")]
    fig, axes = new_fig(W2, 4.8, nrows=2, ncols=2, sharey=True)
    for ax, s in zip(axes.ravel(), SUBSETS):
        for i, (key, lab) in enumerate(COMP):
            r = hb["per_subset"][s][key]["RMSE"]
            lo, hi = r["ci95"]
            excl = (hi < 0) or (lo > 0)
            border = (s == "FD001" and key == "ridge")
            col, mk, mfc = (ORANGE, "D", ORANGE) if border else ((BLUE, "o", BLUE) if excl else (GREY, "s", "white"))
            ax.errorbar(r["delta"], i, xerr=[[r["delta"] - lo], [hi - r["delta"]]], fmt=mk, color=col, mfc=mfc, capsize=2, ms=3.5)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(s)
        ax.set_xlabel("Δ RMSE, D4 − row")
        ax.grid(axis="x", alpha=0.3)
        ax.set_yticks(range(len(COMP)))
        ax.set_yticklabels([c[1] for c in COMP])
        ax.tick_params(labelleft=True)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Hierarchical seeds × engines bootstrap, 95% percentile CI (conservative)")
    fig.supxlabel(wrap("Blue circle: CI excludes 0. Orange diamond: FD001 ridge, borderline (not cited as excluding 0). "
                       "Grey open square: CI covers 0 (never read as 'no difference').", 105), fontsize=7)
    save(fig, "fig_v01_3_hier_bootstrap")


def fig_v01_4_nosmooth_padding() -> None:
    pc = load("experiments/final/posthoc_R0.23_nosmooth_c/padding_nosmooth_results.json")
    vb = load("experiments/final/nosmooth_b_valsim/valsim_results.json")
    fig, ax = new_fig(W2, 3.4, ncols=2)
    w = 0.38
    x = np.arange(4)
    for k, (src, title) in enumerate([("c", "(a) Test: post-hoc split of saved predictions (option c)"),
                                      ("b", "(b) Validation-only simulation, test-like cuts (option b)")]):
        pad, unpad, pad_ci, unpad_ci = [], [], [], []
        for s in SUBSETS:
            if src == "c":
                g = pc["per_subset"][s]["B_nosmooth_gap"]
                gp, gu = g["gap_none_minus_ema_RMSE_padded"], g["gap_none_minus_ema_RMSE_unpadded"]
            else:
                g = vb["summary"][s]
                gp, gu = g["padded"]["gap_none_minus_ema"], g["unpadded"]["gap_none_minus_ema"]
            pad.append(gp["mean"]); pad_ci.append(gp["t95"])
            unpad.append(gu["mean"]); unpad_ci.append(gu["t95"])
        pe = np.array([[m - c[0], c[1] - m] for m, c in zip(pad, pad_ci)]).T
        ue = np.array([[m - c[0], c[1] - m] for m, c in zip(unpad, unpad_ci)]).T
        ax[k].bar(x - w / 2, pad, w, yerr=pe, capsize=2, color=ORANGE, hatch="//", edgecolor=DARK_ORANGE, linewidth=0.5, label="end-padded windows (T < 50)")
        ax[k].bar(x + w / 2, unpad, w, yerr=ue, capsize=2, color=BLUE, label="unpadded windows")
        for i in range(4):
            ax[k].text(x[i] + w / 2, max(unpad[i], 0) + 1.0, f"{unpad[i]:+.2f}", ha="center")
        ax[k].axhline(0, color="k", lw=0.6)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(SUBSETS)
        ax[k].set_ylabel("RMSE gap, none − EMA (cycles)\nmean, t 95% CI")
        ax[k].set_title(wrap(title, 42))
        ax[k].margins(y=0.2)
        ax[k].legend(loc="upper left")
    fig.suptitle("Where the no-smoothing excess error sits (descriptive; not a tested model mechanism)")
    save(fig, "fig_v01_4_nosmooth_padding")


def fig_v01_5_size_curve() -> None:
    a10 = load("experiments/round_04/exp_A10/metrics.json")
    fs = [0.25, 0.5, 0.75, 1.0]
    arm = {0.25: "f0.25", 0.5: "f0.5", 0.75: "f0.75", 1.0: "control"}
    fig, ax = new_fig(W2, 3.6, ncols=2)
    for k, split in enumerate(["val", "test"]):
        for s, mk in zip(SUBSETS, ["o", "s", "^", "D"]):
            means, sds = [], []
            for f in fs:
                m = a10["per_subset"][s][arm[f]][f"{split}_RMSE"]
                means.append(m["mean"]); sds.append(m["std"])
            col, _, ls = SUBSET_STYLE[SUBSETS.index(s)]
            ax[k].errorbar(fs, means, yerr=sds, marker=mk, color=col, ls=ls, capsize=2, ms=3, label=s)
            if s == "FD001":
                med = np.median(a10["per_subset"][s]["f0.25"][f"{split}_RMSE"]["per_seed"])
                ax[k].scatter([0.25], [med], marker="x", s=30, color="k", zorder=5,
                              label="FD001 f 0.25 median\n(seed 2 failed)")
        ax[k].set_xticks(fs)
        ax[k].set_xlabel("fraction f of training engines")
        ax[k].set_ylabel(f"{'validation' if split == 'val' else 'test'} RMSE, mean ± std (5 seeds)")
        ax[k].set_title(f"({'a' if k == 0 else 'b'}) {'Validation' if split == 'val' else 'Test'} (exp_A10; f = 1.0 is exp_B16)")
        ax[k].grid(alpha=0.3)
        ax[k].legend()
    save(fig, "fig_v01_5_size_curve")


def fig_v01_6_win_drop() -> None:
    a12 = load("experiments/round_04/exp_A12/metrics.json")
    a13 = load("experiments/round_04/exp_A13/metrics.json")
    arms = [("L30", a12), ("L40", a12), ("p0.05", a13), ("p0.1", a13), ("p0.3", a13)]
    fig, ax = new_fig(W2, 3.6)
    width = 0.16
    for j, (a, src) in enumerate(arms):
        for i, s in enumerate(SUBSETS):
            d = np.array(src["per_subset"][s][a]["val_RMSE"]["per_seed"]) - np.array(src["per_subset"][s]["control"]["val_RMSE"]["per_seed"])
            m, lo, hi = tci(d)
            xpos = j + (i - 1.5) * width
            ax.errorbar(xpos, m, yerr=[[m - lo], [hi - m]], fmt=SUBSET_STYLE[i][1], capsize=2, ms=3.5,
                        color=SUBSET_STYLE[i][0], label=s if j == 0 else None)
            if a == "L30" and s != "FD001":
                ax.text(xpos, hi + 0.08, "†", ha="center")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(["L = 30", "L = 40", "tied p = 0.05", "tied p = 0.10", "tied p = 0.30"])
    ax.set_ylabel("Δ validation RMSE, arm − frozen D4\n(paired t 95% CI)")
    ax.set_title(wrap("Validation-only sensitivity of the frozen D4 (exp_A12 window, exp_A13 dropout). Positive = arm worse. "
                      "† = end-padded L = 30 bypass cells (RULES §22d)", 95))
    ax.margins(y=0.12)
    ax.legend(ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_v01_6_win_drop")


def fig_v01_7_fd004_regime() -> None:
    rr = load("experiments/final/posthoc_R1.24/regime_results.json")
    names = rr["regime_names"]
    keys = sorted(names.keys(), key=int)
    th = [rr["summary"]["regimes"][k]["Score_share"]["theta_star"] for k in keys]
    d4 = [rr["summary"]["regimes"][k]["Score_share"]["d4"] for k in keys]
    fig, ax = new_fig(W2, 3.2)
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [100 * t[0] for t in th], 0.4, yerr=[100 * t[1] for t in th], capsize=2, color="#777777", label="θ* (exp_B6)")
    ax.bar(x + 0.2, [100 * t[0] for t in d4], 0.4, yerr=[100 * t[1] for t in d4], capsize=2, color="#0b6e99", label="D4 (exp_B16)")
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(names[k], 14) for k in keys])
    ax.set_ylabel("share of FD004 test Score (%)\nmean ± std over 5 seeds")
    ax.set_title("FD004 Score share by operating regime (post-hoc, descriptive; θ* vs D4 only)")
    ax.legend()
    save(fig, "fig_v01_7_fd004_regime")


def fig_r03_1_diagnosis() -> None:
    # SOURCE: experiments/round_03/exp_A9/length_u.md (lazy rows / n), exp_A9/shapes.md (pooling shares)
    lazy = [210 / 240, 170 / 200, 120 / 150, 120 / 150]
    share_theta = [0.016, 0.015, 0.020, 0.096]
    share_d2 = [0.235, 0.158, 0.214, 0.516]
    fig, ax = new_fig(W2, 3.2, ncols=2)
    x = np.arange(4)
    b = ax[0].bar(x, [100 * v for v in lazy], color=ORANGE, hatch="//", edgecolor=DARK_ORANGE, linewidth=0.5)
    ax[0].set_xticks(x, SUB); ax[0].set_ylim(0, 100)
    ax[0].set_ylabel("% of channel query rows")
    ax[0].set_title(wrap("(a) θ* channel encoder: query rows that output mean(V)", 42))
    for r, v in zip(b, lazy):
        ax[0].text(r.get_x() + r.get_width() / 2, 100 * v + 2, f"{100*v:.1f}%", ha="center")
    w = 0.38
    b1 = ax[1].bar(x - w / 2, share_theta, w, label="θ* (and D1)", color="#5b6675")
    b2 = ax[1].bar(x + w / 2, share_d2, w, label="D2 (sensor tokens)", color="#0b6e99")
    ax[1].set_xticks(x, SUB); ax[1].set_ylim(0, 0.65)
    ax[1].set_ylabel("temporal share of fusion mean")
    ax[1].set_title(wrap("(b) weight of the temporal branch in the pooled vector", 42))
    for bars in (b1, b2):
        for r in bars:
            ax[1].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.3f}", ha="center")
    ax[1].legend(loc="upper left")
    fig.suptitle("Computed from the model code at the θ* configurations (exp_A9, CPU; no training)")
    save(fig, "fig_r03_1_diagnosis")


def fig_r03_2_probe() -> None:
    # SOURCE: experiments/round_03/exp_A5/plan.md section 6b (exp_A9/stage0_probe/probe_results.jsonl)
    cfg = ["θ* FD001\n(4/240 tok, b256)", "θ* FD002\n(3/200, b128)", "θ* FD003\n(3/150, b128)",
           "θ* FD004\n(16/150, b64)", "space max\n(16/700, b128)"]
    ms_t = [20.3, 48.8, 40.7, 15.9, 257.6]; ms_d = [29.4, 46.0, 32.4, 15.1, 382.9]
    mem_t = [533, 1075, 809, 436, 5424]; mem_d = [316, 802, 597, 389, 3769]
    fig, ax = new_fig(W2, 4.6, nrows=2)
    x = np.arange(5); w = 0.38
    ax[0].bar(x - w / 2, ms_t, w, label="ProbSparse (θ*)", color="#5b6675")
    ax[0].bar(x + w / 2, ms_d, w, label="exact SDPA (D1)", color="#0b6e99")
    ax[0].set_yscale("log"); plain_log_axis(ax[0].yaxis); ax[0].set_xticks(x, cfg); ax[0].set_ylabel("ms per training step (log)")
    ax[0].set_title("(a) training step time"); ax[0].legend(loc="upper left")
    ax[1].bar(x - w / 2, mem_t, w, label="ProbSparse (θ*)", color="#5b6675")
    ax[1].bar(x + w / 2, mem_d, w, label="exact SDPA (D1)", color="#0b6e99")
    ax[1].set_yscale("log"); plain_log_axis(ax[1].yaxis); ax[1].set_xticks(x, cfg); ax[1].set_ylabel("peak training memory, MiB (log)")
    ax[1].set_title("(b) peak training memory"); ax[1].legend(loc="upper left")
    fig.suptitle(wrap("Stage-0 probe: measured on this GPU with SYNTHETIC tensors (timing/memory only, no accuracy). "
                      "At space max, batch 256: θ* out of memory, D1 7,472 MiB.", 100))
    save(fig, "fig_r03_2_probe")


def fig_r03_4_attention_illustrative() -> None:
    rng = np.random.default_rng(0)
    n = 40; d = 8
    Q = rng.normal(size=(n, d)); K = rng.normal(size=(n, d))
    S = Q @ K.T / np.sqrt(d)
    A_exact = np.exp(S - S.max(1, keepdims=True)); A_exact /= A_exact.sum(1, keepdims=True)
    u = min(5 * int(np.ceil(np.log(n))), n)  # 20 of 40 here; at theta* it is 30 of 150-240
    U = u
    idx = rng.integers(0, n, size=(n, U))
    M = np.array([S[i, idx[i]].max() - S[i, idx[i]].sum() / n for i in range(n)])
    top = np.argsort(-M)[:u]
    A_ps = np.full((n, n), 1.0 / n)
    A_ps[top] = A_exact[top]
    C = 13
    Qs = rng.normal(size=(C, d)); Ks = rng.normal(size=(C, d)); Ss = Qs @ Ks.T / np.sqrt(d)
    A_s = np.exp(Ss - Ss.max(1, keepdims=True)); A_s /= A_s.sum(1, keepdims=True)
    fig, ax = new_fig(W2, 3.0, ncols=3)
    vmax = max(A_exact.max(), A_s.max())
    for a, M_, t in [(ax[0], A_ps, f"ProbSparse (θ*):\n{u} of {n} rows attend,\n{n-u} rows = uniform mean(V)"),
                     (ax[1], A_exact, f"Exact softmax (D1):\nall {n} rows attend"),
                     (ax[2], A_s, f"Sensor tokens (D2-D4):\n{C}×{C}, one token\nper sensor")]:
        im = a.imshow(M_, cmap="viridis", vmin=0, vmax=vmax)
        a.set_title(t); a.set_xlabel("key token"); a.set_ylabel("query token")
    fig.colorbar(im, ax=ax, shrink=0.8, label="attention weight")
    fig.suptitle("ILLUSTRATIVE, SYNTHETIC: random queries/keys on a toy sequence; not a trained model", color="#a15c00")
    save(fig, "fig_r03_4_attention_illustrative")


def fig_r03_5_complexity_illustrative() -> None:
    nn_ = np.arange(2, 801)
    u_ = np.minimum(5 * np.ceil(np.log(nn_)), nn_)
    exact = nn_ ** 2
    prob = np.where(u_ >= nn_, nn_ ** 2, 2 * nn_ * u_)  # sampled scores U*n + kept rows u*n, U = u
    fig, ax = new_fig(W2, 3.3)
    ax.plot(nn_, exact, label="exact attention: n² score entries", color=BLUE)
    ax.plot(nn_, prob, label="ProbSparse: ≈ (U + u)·n, u = U = min(5⌈ln n⌉, n)", color="#5b6675", ls="--")
    for v, lab, yy in [(150, "θ* FD003/FD004\nchannel 150", 3e1), (240, "θ* FD001\nchannel 240", 4e2),
                       (700, "P2 space\nmax 700", 3e1)]:
        ax.axvline(v, color="#a15c00", ls=":", lw=1); ax.text(v + 5, yy, lab, color="#a15c00")
    ax.axvspan(2, 16, facecolor=SKY, edgecolor=BLUE, hatch="\\\\", alpha=0.25)
    ax.text(18, 2e5, "D2-D4 sensor attention:\nn = C ≤ 16 (temporal: N ≤ 17)", color=BLUE)
    ax.set_yscale("log"); plain_log_axis(ax.yaxis); ax.set_xlabel("tokens per sequence n"); ax.set_ylabel("score entries per head (log)")
    ax.set_title("ILLUSTRATIVE (analytic operation counts, not measured time)", color="#a15c00")
    ax.legend(loc="lower right")
    save(fig, "fig_r03_5_complexity_illustrative")


def fig_r03_6_patch_coverage() -> None:
    # FD002 theta*: L 50, P 15, S 15 -> code N = 3 covers cycles 1-45; D3/D4 end padding N' = 4 covers 1-50
    fig, ax = new_fig(W2, 2.6)
    L, P, S = 50, 15, 15
    ax.add_patch(plt.Rectangle((0, 2), L, 0.6, color="#d9dee5")); ax.text(-1, 2.3, "window", ha="right", va="center")
    for k in range(3):
        ax.add_patch(plt.Rectangle((k * S, 1.1), P, 0.6, fill=True, color="#5b6675", alpha=0.6 + 0.1 * k, ec="k"))
    ax.add_patch(plt.Rectangle((45, 1.1), 5, 0.6, facecolor=ORANGE, edgecolor=DARK_ORANGE, hatch="xx", alpha=0.8))
    ax.text(47.5, 1.78, "dropped", ha="center", va="bottom", color=DARK_ORANGE, fontsize=7)
    ax.text(-1, 1.4, "θ* code (N = 3)", ha="right", va="center")
    for k in range(4):
        ax.add_patch(plt.Rectangle((k * S, 0.2), P, 0.6, color="#0b6e99", alpha=0.55 + 0.1 * k, ec="k"))
    ax.add_patch(plt.Rectangle((50, 0.2), 15, 0.6, fill=False, hatch="//", ec="#0b6e99"))
    ax.text(58, 0.95, "pad: S copies of x̃_L", ha="center", va="bottom", fontsize=7)
    ax.text(-1, 0.5, "D3/D4 end padding (N' = 4)", ha="right", va="center")
    ax.set_xlim(-34, 67); ax.set_ylim(0, 2.9); ax.set_yticks([])
    ax.set_xticks([0, 15, 30, 45, 50], ["1", "16", "31", "46", "50"]); ax.set_xlabel("cycle in window")
    ax.text(50, 2.7, "newest cycle", ha="center")
    ax.set_title(wrap("FD002/FD003 θ* temporal patches (L 50, P 15, S 15): schematic of exp_A9/covered.md", 95))
    save(fig, "fig_r03_6_patch_coverage")


def fig_r04_1_chain() -> None:
    s1 = load("experiments/round_03/exp_B15/stage1_summary.json")
    subsets = s1["subsets"]  # FD001, FD003, FD004
    per = s1["per_design_subset"]
    theta_best: dict[str, float] = {}
    for p in s1["hypotheses"]["H1"]["pairs"]:
        theta_best[p["subset"]] = min(theta_best.get(p["subset"], 1e9), p["b"])
    designs = ["θ* (study 0)", "D1 EXACT", "D2 SENSOR-TOKEN", "D3 SENSOR-ALIGNED", "D4 SACA"]
    vals = {s: [theta_best[s]] + [per[d][s]["best"] for d in ["D1", "D2", "D3", "D4"]] for s in subsets}
    sd = [float(np.mean([theta_best[s] for s in subsets]))] + [s1["S_D"][d] for d in ["D1", "D2", "D3", "D4"]]

    fig, ax = new_fig(W2, 3.6, ncols=2)
    x = np.arange(len(designs))
    for s, m in zip(subsets, ["o", "s", "^"]):
        col, _, ls = SUBSET_STYLE[SUBSETS.index(s)]
        ax[0].plot(x, vals[s], marker=m, ms=3, color=col, ls=ls, label=s)
    ax[0].plot(x, sd, "k--", lw=1.4, marker="D", ms=3, label="mean over 3 subsets (S_D,\nscreening statistic)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([wrap(d, 13) for d in designs], rotation=0)
    ax[0].set_ylabel("best validation RMSE of 4 trials (cycles)")
    ax[0].set_title(wrap("(a) Best of study 0 per design (validation only)", 42))
    ax[0].legend()
    hy = s1["hypotheses"]
    labels = ["H1\nD1 − θ*", "H2\nD2 − D1", "H3\nD3 − D2", "H4\nD4 − D3"]
    med = [hy[h]["median_delta"] for h in ["H1", "H2", "H3", "H4"]]
    neg = [hy[h]["n_delta_negative"] for h in ["H1", "H2", "H3", "H4"]]
    cols = [ORANGE if m > 0 else BLUE for m in med]  # orange hatched: later design worse (delta > 0); blue: better
    bars = ax[1].bar(range(4), med, color=cols, edgecolor="#333333", linewidth=0.4)
    for bar, m in zip(bars, med):
        if m > 0:
            bar.set_hatch("//")
    for i, (m, n) in enumerate(zip(med, neg)):
        ax[1].text(i, m + (0.05 if m > 0 else -0.12), f"{m:+.2f}\n{n}/12 better", ha="center",
                   va="bottom" if m > 0 else "top")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("median paired Δ validation RMSE")
    ax[1].set_ylim(-1.9, 1.1)
    ax[1].set_title(wrap("(b) Step-wise paired deltas, 12 non-independent pairs", 42))
    fig.suptitle(wrap("Round 3 stage 1 (exp_B15): exploratory, validation only, FD001/FD003/FD004, one study per design", 100))
    save(fig, "fig_r04_1_chain")


def fig_r04_4_window_tokens() -> None:
    """Temporal patches of D4 with end padding at the frozen cells. N', N and P come from the shape-script output
    (``results/shapes/d4_shapes.json``); the orange padding bar has length N' * S - L cycles."""
    with open(SHAPES_JSON, encoding="utf-8") as f:
        recs = json.load(f)
    row_defs = [("FD001", ["FD001"]), ("FD002-FD004", ["FD002", "FD003", "FD004"])]
    fig, axes = new_fig(W2, 4.8, nrows=2, ncols=3, sharey=True)
    for r, (lab, subs) in enumerate(row_defs):
        for c, L in enumerate([30, 40, 50]):
            cell = [x for x in recs if x["L"] == L and x["subset"] in subs]
            key = {(x["P_t"], x["S_t"], x["N_no_padding"], x["N_prime_end_padding"]) for x in cell}
            assert len(key) == 1, (lab, L, key)  # same patching in every subset and seed of the row
            P, S, n_code, n_pad = next(iter(key))
            pad = n_pad * S - L
            ax = axes[r, c]
            ax.add_patch(plt.Rectangle((0, 0.6), L, 0.3, color="#cfd8e3"))
            ax.add_patch(plt.Rectangle((L, 0.6), pad, 0.3, color="#f3d9b1", hatch="//"))
            for n in range(n_pad):
                s = n * S
                ax.add_patch(plt.Rectangle((s, 0.1), P, 0.35, fill=False, ec="#0b6e99", lw=1.0))
                ax.text(s + P / 2, 0.27, f"p{n + 1}", ha="center", va="center")
            ax.set_xlim(-1, 66)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([])
            ok = "ok" if n_code >= 2 else "REJECTED"
            ax.set_title(f"{lab}, L = {L} (P = S = {P})\nP2 rule N_t = {n_code} ({ok}); D4 N' = {n_pad}\n"
                         f"padding = N'·S − L = {pad} cycles")
            ax.set_xlabel("cycle in window")
    fig.legend(handles=[Patch(color="#cfd8e3", label="data (L cycles)"),
                        Patch(facecolor="#f3d9b1", hatch="//", label="end padding, last value repeated"),
                        Patch(fill=False, ec="#0b6e99", label="patch p_k")], loc="outside lower center", ncol=3)
    fig.suptitle(wrap("D4 temporal patches with end padding (N' = ⌊(L−P)/S⌋ + 2; patch sizes and counts from the "
                      "shape script, frozen exp_B16 and exp_A12 configurations)", 100))
    save(fig, "fig_r04_4_window_tokens")


def fig_r04_5_bn_ln_illustrative() -> None:
    rng = np.random.default_rng(0)
    B, S, d = 6, 5, 8
    Z = rng.normal(0, 1, (B, S, d)) + np.linspace(-2, 2, B)[:, None, None]
    bn = (Z - Z.mean(axis=(0, 1), keepdims=True)) / Z.std(axis=(0, 1), keepdims=True)
    ln = (Z - Z.mean(axis=2, keepdims=True)) / Z.std(axis=2, keepdims=True)
    fig, ax = new_fig(W2, 3.0, ncols=3)
    for a, M, t in zip(ax, [Z, bn, ln], ["input Z\n(token means shifted\nper sample)",
                                        "BatchNorm: per\nfeature j over all (b, s)",
                                        "LayerNorm: per token\n(b, s) over d features"]):
        im = a.imshow(M.reshape(B * S, d), aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
        a.set_title(t)
        a.set_xlabel("feature j")
        a.set_ylabel("token (b, s)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(wrap("ILLUSTRATIVE, SYNTHETIC: which statistics each norm uses (random numbers, not a trained model)", 100),
                 color="#a15c00")
    save(fig, "fig_r04_5_bn_ln_illustrative")


def fig_score_function() -> None:
    """NASA Score per engine as a function of d = predicted - true RUL: the corrected constants (early /13, late /10)
    against the swapped constants of the submitted code (early /10, late /13). Computed from the formula (R0.6)."""
    from ..evaluation import A_EARLY, A_LATE

    d = np.linspace(-50, 50, 401)
    ok = np.where(d < 0, np.expm1(-d / A_EARLY), np.expm1(d / A_LATE))
    swapped = np.where(d < 0, np.expm1(-d / 10.0), np.expm1(d / 13.0))
    fig, ax = new_fig(W1, 3.1)
    ax.plot(d, ok, color=BLUE, lw=1.6, label="this work: early /13, late /10")
    ax.plot(d, swapped, color=ORANGE, lw=1.6, ls="--", label="submitted code: early /10,\nlate /13 (swapped)")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("d = predicted − true RUL (cycles)\nd < 0 early, d ≥ 0 late")
    ax.set_ylabel("score contribution of one engine")
    ax.set_ylim(0, 160)
    ax.set_title("NASA Score: computed from the\nformula (not a result)")
    ax.legend(loc="upper center")
    ax.grid(alpha=0.3)
    save(fig, "fig_score_function")


FIGURES: dict[str, Callable[[], None]] = {
    "fig_v01_1_final_rmse": fig_v01_1_final_rmse,
    "fig_v01_2_primary_pairs": fig_v01_2_primary_pairs,
    "fig_v01_3_hier_bootstrap": fig_v01_3_hier_bootstrap,
    "fig_v01_4_nosmooth_padding": fig_v01_4_nosmooth_padding,
    "fig_v01_5_size_curve": fig_v01_5_size_curve,
    "fig_v01_6_win_drop": fig_v01_6_win_drop,
    "fig_v01_7_fd004_regime": fig_v01_7_fd004_regime,
    "fig_r03_1_diagnosis": fig_r03_1_diagnosis,
    "fig_r03_2_probe": fig_r03_2_probe,
    "fig_r03_4_attention_illustrative": fig_r03_4_attention_illustrative,
    "fig_r03_5_complexity_illustrative": fig_r03_5_complexity_illustrative,
    "fig_r03_6_patch_coverage": fig_r03_6_patch_coverage,
    "fig_r04_1_chain": fig_r04_1_chain,
    "fig_r04_4_window_tokens": fig_r04_4_window_tokens,
    "fig_r04_5_bn_ln_illustrative": fig_r04_5_bn_ln_illustrative,
    "fig_score_function": fig_score_function,
}


def main(argv: list[str] | None = None) -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--only", default=None, help="one figure name")
    a = ap.parse_args(argv)
    OUT = os.path.abspath(a.out_dir)
    for name, fn in FIGURES.items():
        if a.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
