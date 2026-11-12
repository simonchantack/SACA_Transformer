"""Release-and-manuscript figures, version 02 (RULES section 32): redrawn from the same frozen data, no new numbers.

Differences from ``figures.py`` (version 01, kept unchanged): the model is labelled SACA; model names use their published
capitalisation; no file names, experiment identifiers or rule references are printed; zero-size bars get a marker;
files carry the suffix ``_v02``. Every plotted array is also recorded in ``PLOTTED`` so that a test can compare it with the
frozen source values.

Figures for the response to reviewers and the paper (R1.23): PNG at 300 dpi and PDF, one function per figure.

Data access is unchanged from version 01:
every number is read from a frozen artifact (named in each figure) or is a formula; figures marked ILLUSTRATIVE use
synthetic toy data and say so in the figure itself. Planning graphics (GPU-budget projections) are not regenerated.

Layout (fix attempt 1, R1.23): every figure is drawn at its final print size, 3.5 in (one ESWA column) or 7.2 in (two
columns), so that nothing is scaled when it is placed in the paper. All text is at least 7 pt at that size; ``save``
refuses to write a figure that is wider than 7.2 in or has any visible text below 7 pt (tick labels and offset text included;
mathtext sub/superscripts are drawn at the nominal size inside ``script_scale``, see there). The ``fig_r04_4`` padding bar is
read from the shape-script output (``results/shapes/d4_shapes.json``): its length is ``N' * S - L`` cycles.

``python -m saca_rul.reporting.figures_v02 --out-dir figures`` writes every figure; ``--only NAME`` writes one.
"""
import argparse
import contextlib
import json
import os
import re
import textwrap
from collections.abc import Callable, Iterator

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
SUFFIX = "_v02"
PLOTTED: dict[str, dict[str, object]] = {}
MODEL_NAME = {"d4": "SACA", "theta_star": "θ*", "patchtst": "PatchTST", "itransformer": "iTransformer",
              "crossformer": "Crossformer", "lstm": "LSTM", "ridge": "ridge"}
SHAPES_JSON = str(resolve_path("results/shapes/d4_shapes.json"))
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SUB = SUBSETS
W1, W2 = 3.5, 7.2  # one column, two columns (inches)
MIN_PT = 7.0
BLUE, ORANGE, SKY, GREY = "#0072B2", "#E69F00", "#56B4E9", "#666666"  # Okabe-Ito; no red/green pairs (RULES section 37)
DARK_ORANGE = "#8F5A00"  # text and hatch lines on orange fills
SUBSET_STYLE = [(BLUE, "o", "-"), (ORANGE, "s", "--"), (SKY, "^", ":"), (GREY, "D", "-.")]  # colour, marker, line style per subset
PAD_EDGE = "#9a5b00"  # darker than the padding fill, so that the hatch lines are visible
MINUS = re.compile(r"(?<![\w.)])-(?=[\d.])")  # ASCII hyphen used as a minus sign (before a number, not after a word)


@contextlib.contextmanager
def script_scale(factor: float = 1.0) -> Iterator[None]:
    """Draw mathtext sub/superscripts at ``factor`` times the base size while the block runs (matplotlib default 0.7).

    With 0.7 a 7 pt label would print its subscript at 4.9 pt, below the 7 pt floor, so the figures draw scripts at full
    size (lowered only). matplotlib has no public setting for this; the module constant ``matplotlib._mathtext.SHRINK_FACTOR``
    is set on entry and always restored in ``finally`` (nothing changes at import). The parsed-layout cache is cleared on
    entry and exit so that no layout made with the other scale is reused. The guard is the output test: every text span in
    the shipped PDFs must measure at least 7 pt (``tests/test_figures_v02.py``)."""
    import matplotlib._mathtext as mt
    from matplotlib.mathtext import MathTextParser

    def clear() -> None:
        cached = getattr(MathTextParser, "_parse_cached", None)
        if cached is not None and hasattr(cached, "cache_clear"):
            cached.cache_clear()

    old = getattr(mt, "SHRINK_FACTOR", None)
    if old is not None:
        mt.SHRINK_FACTOR = factor
    clear()
    try:
        yield
    finally:
        if old is not None:
            mt.SHRINK_FACTOR = old
        clear()


plt.rcParams.update({"mathtext.default": "regular", "mathtext.fontset": "dejavusans", "axes.unicode_minus": True,
                     "hatch.linewidth": 0.8,
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.titlesize": 8, "figure.dpi": 130, "pdf.fonttype": 42, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
})


def rec(fig: str, key: str, value: object) -> None:
    """Record a plotted array (as nested lists of floats) for the value test."""
    PLOTTED.setdefault(fig, {})[key] = np.asarray(value, dtype=float).tolist()


def load(rel):
    with open(os.path.join(WS, rel), encoding="utf-8") as f:
        return json.load(f)


def wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.fill(line, width) for line in s.split("\n"))


def new_fig(width: float, height: float, **kw):
    return plt.subplots(figsize=(width, height), layout="constrained", **kw)


def effective_pt(t: Text) -> float:
    """Nominal size of a text object. Mathtext scripts are drawn at the nominal size inside ``script_scale``; the measured
    size of every span in the written PDFs is checked by the output test, not here."""
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


def fix_minus(fig) -> None:
    """Replace an ASCII hyphen that is a minus sign (before a digit) by U+2212 in every text object of the figure."""
    for t in fig.findobj(Text):
        s = t.get_text()
        if "-" in s and "$" not in s:
            t.set_text(MINUS.sub("\u2212", s))


def save(fig, name):
    """Write ``name``_v02.png (300 dpi) and ``name``_v02.pdf into the output folder (exact figure size, no cropping)."""
    fix_minus(fig)
    with script_scale(1.0):
        check_figure(fig, name)
        os.makedirs(OUT, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"{name}{SUFFIX}.{ext}"), dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", name + SUFFIX)


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def _v4():
    v4 = load("experiments/final/combined_table_v4.json")
    rows = [(k, MODEL_NAME[k]) for k in ("d4", "theta_star", "patchtst", "itransformer", "crossformer", "lstm", "ridge")]
    return v4, rows, v4["best_baseline_by_validation"]


def fig_v01_1_final_rmse() -> None:
    v4, ROWS, BK = _v4()
    fig, axes = new_fig(W2, 4.5, nrows=2, ncols=2)
    for ax, s in zip(axes.ravel(), SUBSETS):
        for i, (key, lab) in enumerate(ROWS):
            ps = v4["rows"][key]["per_subset"][s]["per_seed"]["RMSE"]
            rec("fig_v01_1_final_rmse", f"{s}/{key}", ps)
            col = "#0b6e99" if key == "d4" else ("#a15c00" if key == BK[s] else "#777777")
            ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), ps, s=9, color=col, alpha=0.8)
            ax.hlines(np.mean(ps), i - 0.3, i + 0.3, color=col, lw=1.8)
        ax.set_xticks(range(len(ROWS)))
        ax.set_xticklabels([r[1] for r in ROWS], rotation=35, ha="right")
        ax.set_title(f"{s} (best baseline: {MODEL_NAME[BK[s]]})")
        ax.set_ylabel("test RMSE (cycles)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(wrap("Test RMSE of the final rows. Dots: split seeds 0-4; bar: mean. "
                      "Blue: SACA; orange: validation-chosen best baseline; grey: other rows.", 100))
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
        cs.append("#0072B2" if p["delta"] < 0 else "#E69F00")  # Okabe-Ito blue (SACA lower) / orange (SACA higher)
    bars = ax.bar(xs, ys, color=cs, edgecolor="#333333", linewidth=0.4)
    for bar, y in zip(bars, ys):
        if y >= 0:
            bar.set_hatch("//")  # orange bars also carry a hatch, so the sign is readable without colour
    ax.scatter(xs, ys, marker="D", s=10, color="k", zorder=4, label="value of each pair (marker shows near-zero pairs)")
    rec("fig_v01_2_primary_pairs", "delta", ys)
    rec("fig_v01_2_primary_pairs", "margin", ni["margin_rmse"])
    rec("fig_v01_2_primary_pairs", "mean_delta", prim["mean_delta"])
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(ni["margin_rmse"], color="#222222", ls="--", lw=1.2, label=f"non-inferiority margin +{ni['margin_rmse']} cycles")
    ax.axhline(prim["mean_delta"], color="#555555", ls=":", lw=1.2, label=f"mean Δ {prim['mean_delta']:+.2f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p['subset'][2:]}\ns{p['seed']}" for p in pairs])
    ax.set_xlabel("subset number (FD00x) and split seed (s)")
    ax.set_ylabel("Δ test RMSE, SACA − best baseline (cycles)")
    ax.set_title(wrap(f"Preregistered primary: {prim['n_dstar_better']}/20 pairs SACA lower; exact Wilcoxon p = "
                      f"{prim['wilcoxon_exact_two_sided_p']:.4f} → {prim['outcome']}; NI one-sided p = "
                      f"{ni['p_one_sided']:.5f}", 95))
    ax.legend(loc="lower left")
    save(fig, "fig_v01_2_primary_pairs")


def fig_v01_3_hier_bootstrap() -> None:
    hb = load("experiments/final/posthoc_R2.10/hier_bootstrap_results.json")
    COMP = [("theta_star", "θ*"), ("patchtst", "PatchTST"), ("itransformer", "iTransformer"),
            ("crossformer", "Crossformer"), ("lstm", "LSTM"), ("ridge", "ridge"), ("best_baseline", "best baseline")]
    fig, axes = new_fig(W2, 4.8, nrows=2, ncols=2, sharey=True)
    for ax, s in zip(axes.ravel(), SUBSETS):
        for i, (key, lab) in enumerate(COMP):
            r = hb["per_subset"][s][key]["RMSE"]
            lo, hi = r["ci95"]
            rec("fig_v01_3_hier_bootstrap", f"{s}/{key}", [r["delta"], lo, hi])
            excl = (hi < 0) or (lo > 0)
            border = (s == "FD001" and key == "ridge")
            # blue filled circle: CI excludes 0; orange diamond: borderline; grey open square: CI covers 0
            col, mk, mfc = (ORANGE, "D", ORANGE) if border else ((BLUE, "o", BLUE) if excl else (GREY, "s", "white"))
            ax.errorbar(r["delta"], i, xerr=[[r["delta"] - lo], [hi - r["delta"]]], fmt=mk, color=col, mfc=mfc, capsize=2, ms=3.5)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(s)
        ax.set_xlabel("Δ RMSE, SACA − row")
        ax.grid(axis="x", alpha=0.3)
        ax.set_yticks(range(len(COMP)))
        ax.set_yticklabels([c[1] for c in COMP])
        ax.tick_params(labelleft=True)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Hierarchical seeds × engines bootstrap, 95% percentile CI (conservative)")
    fig.supxlabel(wrap("Blue circle: CI excludes 0. Orange diamond: FD001 ridge, borderline. Grey open square: CI covers 0.", 105), fontsize=7)
    save(fig, "fig_v01_3_hier_bootstrap")


def fig_v01_4_nosmooth_padding() -> None:
    pc = load("experiments/final/posthoc_R0.23_nosmooth_c/padding_nosmooth_results.json")
    vb = load("experiments/final/nosmooth_b_valsim/valsim_results.json")
    fig, ax = new_fig(W2, 3.4, ncols=2)
    w = 0.38
    x = np.arange(4)
    for k, (src, title) in enumerate([("c", "(a) Test: post-hoc split of saved predictions"),
                                      ("b", "(b) Validation-only simulation with test-like cuts")]):
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
        rec("fig_v01_4_nosmooth_padding", f"{src}/padded_mean", pad)
        rec("fig_v01_4_nosmooth_padding", f"{src}/padded_ci", pad_ci)
        rec("fig_v01_4_nosmooth_padding", f"{src}/unpadded_mean", unpad)
        rec("fig_v01_4_nosmooth_padding", f"{src}/unpadded_ci", unpad_ci)
        pe = np.array([[m - c[0], c[1] - m] for m, c in zip(pad, pad_ci)]).T
        ue = np.array([[m - c[0], c[1] - m] for m, c in zip(unpad, unpad_ci)]).T
        ax[k].bar(x - w / 2, pad, w, yerr=pe, capsize=2, color=ORANGE, hatch="//", edgecolor=DARK_ORANGE, linewidth=0.5, label="end-padded windows (T < 50)")
        ax[k].bar(x + w / 2, unpad, w, yerr=ue, capsize=2, color=BLUE, label="unpadded windows")
        for i in range(4):
            top = max(unpad[i] + ue[1][i], 0.0)  # value label sits above the upper error-bar cap
            ax[k].annotate(f"{unpad[i]:+.2f}", (x[i], top), xytext=(1, 3), textcoords="offset points",
                           ha="left", va="bottom")  # starts at the left edge of the unpadded bar: clear of the padded bar
        ax[k].axhline(0, color="k", lw=0.6)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(SUBSETS)
        ax[k].set_ylabel("RMSE gap, none − EMA (cycles)\nmean, t 95% CI")
        ax[k].set_title(wrap(title, 42))
        ax[k].margins(y=0.3)
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
            rec("fig_v01_5_size_curve", f"{split}/{s}/mean", means)
            rec("fig_v01_5_size_curve", f"{split}/{s}/std", sds)
            col, _, ls = SUBSET_STYLE[SUBSETS.index(s)]
            ax[k].errorbar(fs, means, yerr=sds, marker=mk, color=col, ls=ls, capsize=2, ms=3, label=s)
            if s == "FD001":
                med = np.median(a10["per_subset"][s]["f0.25"][f"{split}_RMSE"]["per_seed"])
                rec("fig_v01_5_size_curve", f"{split}/FD001/median_f0.25", med)
                ax[k].scatter([0.25], [med], marker="x", s=30, color="k", zorder=5,
                              label="FD001 f 0.25 median\n(seed 2 failed)")
        ax[k].set_xticks(fs)
        ax[k].set_xlabel("fraction f of training engines")
        ax[k].set_ylabel(f"{'validation' if split == 'val' else 'test'} RMSE, mean ± std (5 seeds)")
        ax[k].set_title(f"({'a' if k == 0 else 'b'}) {'Validation' if split == 'val' else 'Test'} (f = 1.0: full training set)")
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
            rec("fig_v01_6_win_drop", f"{a}/{s}", [m, lo, hi])
            xpos = j + (i - 1.5) * width
            ax.errorbar(xpos, m, yerr=[[m - lo], [hi - m]], fmt=SUBSET_STYLE[i][1], capsize=2, ms=3.5,
                        color=SUBSET_STYLE[i][0], label=s if j == 0 else None)
            if a == "L30" and s != "FD001":
                ax.text(xpos, hi + 0.08, "†", ha="center")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(["L = 30", "L = 40", "tied rate 0.05", "tied rate 0.10", "tied rate 0.30"])
    ax.set_ylabel("Δ validation RMSE, arm − frozen SACA\n(paired t 95% CI)")
    ax.set_title(wrap("Validation-only sensitivity of the frozen SACA configuration (window length L; tied dropout rate). "
                      "Positive = arm worse. † = L = 30 cells that need end padding (FD002-FD004)", 95))
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
    rec("fig_v01_7_fd004_regime", "theta_star", th)
    rec("fig_v01_7_fd004_regime", "d4", d4)
    fig, ax = new_fig(W2, 3.2)
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [100 * t[0] for t in th], 0.4, yerr=[100 * t[1] for t in th], capsize=2, color="#777777", label="θ*")
    ax.bar(x + 0.2, [100 * t[0] for t in d4], 0.4, yerr=[100 * t[1] for t in d4], capsize=2, color="#0b6e99", label="SACA")
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(names[k], 14) for k in keys])
    ax.set_ylabel("share of FD004 test Score (%)\nmean ± std over 5 seeds")
    ax.set_title("FD004 Score share by operating regime (post-hoc, descriptive; θ* vs SACA only)")
    ax.legend()
    save(fig, "fig_v01_7_fd004_regime")


def fig_r03_1_diagnosis() -> None:
    # SOURCE: experiments/round_03/exp_A9/length_u.md (lazy rows / n), exp_A9/shapes.md (pooling shares)
    lazy = [210 / 240, 170 / 200, 120 / 150, 120 / 150]
    share_theta = [0.016, 0.015, 0.020, 0.096]
    share_d2 = [0.235, 0.158, 0.214, 0.516]
    rec("fig_r03_1_diagnosis", "lazy_fraction", lazy)
    rec("fig_r03_1_diagnosis", "share_theta", share_theta)
    rec("fig_r03_1_diagnosis", "share_d2", share_d2)
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
    fig.suptitle("Computed from the model code at the θ* configurations (no training)")
    save(fig, "fig_r03_1_diagnosis")


def fig_r03_2_probe() -> None:
    # SOURCE: experiments/round_03/exp_A5/plan.md section 6b (exp_A9/stage0_probe/probe_results.jsonl)
    def lab(name: str, t: int, c: int, b: int) -> str:
        return f"{name}\n{t} temporal +\n{c} channel tokens\nbatch {b}"
    cfg = [lab("θ* FD001", 4, 240, 256), lab("θ* FD002", 3, 200, 128), lab("θ* FD003", 3, 150, 128),
           lab("θ* FD004", 16, 150, 64), lab("search-space maximum", 16, 700, 128)]
    ms_t = [20.3, 48.8, 40.7, 15.9, 257.6]; ms_d = [29.4, 46.0, 32.4, 15.1, 382.9]
    mem_t = [533, 1075, 809, 436, 5424]; mem_d = [316, 802, 597, 389, 3769]
    for k_, v_ in (("ms_theta", ms_t), ("ms_d1", ms_d), ("mem_theta", mem_t), ("mem_d1", mem_d)):
        rec("fig_r03_2_probe", k_, v_)
    fig, ax = new_fig(W2, 4.6, nrows=2)
    x = np.arange(5); w = 0.38
    ax[0].bar(x - w / 2, ms_t, w, label="ProbSparse (θ*)", color="#5b6675")
    ax[0].bar(x + w / 2, ms_d, w, label="exact SDPA (D1)", color="#0b6e99")
    ax[0].set_yscale("log"); plain_log_axis(ax[0].yaxis); ax[0].set_ylim(10, 600); ax[0].set_yticks([10, 20, 50, 100, 200, 500]); ax[0].set_xticks(x, cfg); ax[0].set_ylabel("ms per training step (log)")
    ax[0].set_title("(a) training step time"); ax[0].legend(loc="upper left")
    ax[1].bar(x - w / 2, mem_t, w, label="ProbSparse (θ*)", color="#5b6675")
    ax[1].bar(x + w / 2, mem_d, w, label="exact SDPA (D1)", color="#0b6e99")
    ax[1].set_yscale("log"); plain_log_axis(ax[1].yaxis); ax[1].set_ylim(200, 12000); ax[1].set_yticks([200, 500, 1000, 2000, 5000, 10000]); ax[1].set_xticks(x, cfg); ax[1].set_ylabel("peak training memory, MiB (log)")
    ax[1].set_title("(b) peak training memory"); ax[1].legend(loc="upper left")
    fig.suptitle(wrap("Probe on an NVIDIA GeForce RTX 5070 Ti Laptop GPU with SYNTHETIC tensors (timing and memory only, no accuracy). "
                      "Search-space maximum at batch 256: θ* out of memory, D1 7,472 MiB.", 100))
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
    rec("fig_r03_4_attention_illustrative", "A_probsparse", A_ps)
    rec("fig_r03_4_attention_illustrative", "A_exact", A_exact)
    rec("fig_r03_4_attention_illustrative", "A_sensor", A_s)
    fig, ax = new_fig(W2, 3.0, ncols=3)
    vmax = max(A_exact.max(), A_s.max())
    for a, M_, t in [(ax[0], A_ps, f"ProbSparse (θ*):\n{u} of {n} rows attend,\n{n-u} rows = uniform mean(V)"),
                     (ax[1], A_exact, f"Exact softmax (D1):\nall {n} rows attend"),
                     (ax[2], A_s, f"Sensor tokens (D2, D3, SACA):\n{C}×{C}, one token\nper sensor")]:
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
    rec("fig_r03_5_complexity_illustrative", "exact", exact)
    rec("fig_r03_5_complexity_illustrative", "probsparse", prob)
    fig, ax = new_fig(W2, 3.3)
    ax.plot(nn_, exact, label="exact attention: n² score entries", color=BLUE)
    ax.plot(nn_, prob, label="ProbSparse: ≈ (U + u)·n, u = U = min(5⌈ln n⌉, n)", color="#5b6675", ls="--")
    for v, lab, yy in [(150, "θ* FD003/FD004\nchannel 150", 3e1), (240, "θ* FD001\nchannel 240", 4e2),
                       (700, "search-space\nmax 700", 3e1)]:
        ax.axvline(v, color="#a15c00", ls=":", lw=1); ax.text(v + 5, yy, lab, color="#a15c00")
    ax.axvspan(2, 16, facecolor=SKY, edgecolor=BLUE, hatch="\\\\", alpha=0.25)
    ax.text(18, 2e5, "D2, D3, SACA sensor attention:\nn = C ≤ 16 (temporal: N ≤ 17)", color=BLUE)
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
    ax.text(58, 0.95, r"pad: S copies of $\tilde{x}_L$", ha="center", va="bottom", fontsize=7)
    ax.text(-1, 0.5, "D3, SACA end padding (N' = 4)", ha="right", va="center")
    ax.set_xlim(-34, 67); ax.set_ylim(0, 2.9); ax.set_yticks([])
    ax.set_xticks([0, 15, 30, 45, 50], ["1", "16", "31", "46", "50"]); ax.set_xlabel("cycle in window")
    ax.text(50, 2.7, "newest cycle", ha="center")
    ax.set_title(wrap("FD002/FD003 θ* temporal patches (L 50, P 15, S 15): schematic", 95))
    save(fig, "fig_r03_6_patch_coverage")


def fig_r04_1_chain() -> None:
    s1 = load("experiments/round_03/exp_B15/stage1_summary.json")
    subsets = s1["subsets"]  # FD001, FD003, FD004
    per = s1["per_design_subset"]
    theta_best: dict[str, float] = {}
    for p in s1["hypotheses"]["H1"]["pairs"]:
        theta_best[p["subset"]] = min(theta_best.get(p["subset"], 1e9), p["b"])
    designs = ["θ* (study 0)", "D1 EXACT", "D2 SENSOR-TOKEN", "D3 SENSOR-ALIGNED", "SACA"]
    vals = {s: [theta_best[s]] + [per[d][s]["best"] for d in ["D1", "D2", "D3", "D4"]] for s in subsets}
    sd = [float(np.mean([theta_best[s] for s in subsets]))] + [s1["S_D"][d] for d in ["D1", "D2", "D3", "D4"]]

    rec("fig_r04_1_chain", "best_by_subset", [vals[s] for s in subsets])
    rec("fig_r04_1_chain", "S_D", sd)
    fig, ax = new_fig(W2, 3.6, ncols=2)
    x = np.arange(len(designs))
    for s, m in zip(subsets, ["o", "s", "^"]):
        col, _, ls = SUBSET_STYLE[["FD001", "FD002", "FD003", "FD004"].index(s)]
        ax[0].plot(x, vals[s], marker=m, ms=3, color=col, ls=ls, label=s)
    ax[0].plot(x, sd, "k--", lw=1.4, marker="D", ms=3, label="mean over 3 subsets ($S_D$,\nscreening statistic)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([wrap(d, 13) for d in designs], rotation=0)
    ax[0].set_ylabel("best validation RMSE of 4 trials (cycles)")
    ax[0].set_title(wrap("(a) Best of study 0 per design (validation only)", 42))
    ax[0].legend()
    hy = s1["hypotheses"]
    labels = ["H1\nD1 − θ*", "H2\nD2 − D1", "H3\nD3 − D2", "H4\nSACA − D3"]
    med = [hy[h]["median_delta"] for h in ["H1", "H2", "H3", "H4"]]
    neg = [hy[h]["n_delta_negative"] for h in ["H1", "H2", "H3", "H4"]]
    rec("fig_r04_1_chain", "median_delta", med)
    rec("fig_r04_1_chain", "n_delta_negative", neg)
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
    fig.suptitle(wrap("Design screening: exploratory, validation only, FD001/FD003/FD004, one study per design", 100))
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
            rec("fig_r04_4_window_tokens", f"{lab}/L{L}", [P, S, n_code, n_pad, pad])
            ax = axes[r, c]
            ax.add_patch(plt.Rectangle((0, 0.6), L, 0.3, color="#cfd8e3"))
            ax.add_patch(plt.Rectangle((L, 0.6), pad, 0.3, facecolor="#f3d9b1", edgecolor=PAD_EDGE, linewidth=0.6, hatch="//"))
            for n in range(n_pad):
                s = n * S
                ax.add_patch(plt.Rectangle((s, 0.1), P, 0.35, fill=False, ec="#0b6e99", lw=1.0))
                ax.text(s + P / 2, 0.27, f"$p_{{{n + 1}}}$", ha="center", va="center")
            ax.set_xlim(-1, 66)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([])
            ok = "meets $N_t$ ≥ 2" if n_code >= 2 else "fails $N_t$ ≥ 2"
            ax.set_title(f"{lab}, L = {L} (P = S = {P})\nwithout padding: $N_t$ = {n_code} ({ok})\n"
                         f"SACA: N' = {n_pad}; padding N'·S − L = {pad} cycles")
            ax.set_xlabel("cycle in window")
    fig.legend(handles=[Patch(color="#cfd8e3", label="data (L cycles)"),
                        Patch(facecolor="#f3d9b1", edgecolor=PAD_EDGE, linewidth=0.6, hatch="//", label="end padding, last value repeated"),
                        Patch(fill=False, ec="#0b6e99", label="patch $p_k$")], loc="outside lower center", ncol=3)
    fig.suptitle(wrap("SACA temporal patches with end padding (N' = ⌊(L−P)/S⌋ + 2; patch sizes and counts from the "
                      "shape computation at the frozen configurations)", 100))
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
    rec("fig_score_function", "d", d)
    rec("fig_score_function", "this_work", ok)
    rec("fig_score_function", "swapped", swapped)
    fig, ax = new_fig(W1, 3.1)
    ax.plot(d, ok, color=BLUE, lw=1.6, label="this work: early /13, late /10")
    ax.plot(d, swapped, color=ORANGE, lw=1.6, ls="--", label="swapped constants:\nearly /10, late /13")
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
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--only", default=None, help="one figure name")
    a = ap.parse_args(argv)
    OUT = os.path.abspath(a.out_dir)
    for name, fn in FIGURES.items():
        if a.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
