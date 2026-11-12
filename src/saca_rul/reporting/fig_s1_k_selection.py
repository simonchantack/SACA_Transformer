"""Supplementary Fig. S1 (R1.28): inertia and silhouette of the k-means on the operating settings, k = 2..10, FD002 and FD004.

The values are read from the saved outputs of the post-hoc k check (``experiments/round_04_pre/exp_C2/outputs/k_selection_FD00{2,4}.json``, five
split seeds each). Nothing is clustered or recomputed here, and no test data are read. k = 6 was fixed in the earlier implementation
before this check; the plot is a confirmation, not a selection step.

Style as in ``figures_v02``: 7.2 in wide, text of at least 7 pt, Okabe-Ito colours plus markers and line styles, so that the five seeds
can be told apart without colour. ``python -m saca_rul.reporting.fig_s1_k_selection --out-dir figures`` writes
``fig_s1_k_selection_v02.png`` (300 dpi) and ``.pdf``.
"""
import argparse
import json
import os

from . import figures_v02 as F

SUBSETS = ["FD002", "FD004"]
SEEDS = ["0", "1", "2", "3", "4"]
KS = list(range(2, 11))
K_USED = 6
# colour, marker, line style per split seed (Okabe-Ito: blue, orange, sky blue, grey, black)
SEED_STYLE = [(F.BLUE, "o", "-"), (F.ORANGE, "s", "--"), (F.SKY, "^", ":"), (F.GREY, "D", "-."), ("#000000", "v", (0, (5, 1, 1, 1)))]
REL = "experiments/round_04_pre/exp_C2/outputs/k_selection_{subset}.json"


def load_values(subset: str) -> dict[str, dict[str, list[float]]]:
    """Per split seed: inertia and silhouette for k = 2..10, copied from the saved output (no recomputation)."""
    d = F.load(REL.format(subset=subset))
    assert d["ks"] == KS and sorted(d["per_seed"]) == SEEDS, (subset, d["ks"], sorted(d["per_seed"]))
    return {s: {m: [float(d["per_seed"][s]["per_k"][str(k)][m]) for k in KS] for m in ("inertia", "silhouette")} for s in SEEDS}


def fig_s1_k_selection() -> None:
    fig, ax = F.new_fig(F.W2, 4.8, nrows=2, ncols=2)
    for j, subset in enumerate(SUBSETS):
        vals = load_values(subset)
        for i, metric in enumerate(("inertia", "silhouette")):
            a = ax[i, j]
            for (s, v), (col, mk, ls) in zip(vals.items(), SEED_STYLE):
                F.rec("fig_s1_k_selection", f"{subset}/{metric}/seed{s}", v[metric])
                a.plot(KS, v[metric], color=col, marker=mk, ls=ls, ms=3, lw=1.0, label=f"split seed {s}")
            a.axvline(K_USED, color="#333333", ls="--", lw=0.8, label="k = 6")
            a.set_xticks(KS)
            a.set_xlabel("number of clusters k")
            a.grid(alpha=0.3)
            if metric == "inertia":
                a.set_yscale("log")
                F.plain_log_axis(a.yaxis)
                a.set_ylim(top=3e7)
                a.set_ylabel("inertia (log scale)")
                a.set_title(f"{subset}: inertia, training-engine rows")
            else:
                a.set_ylabel("silhouette")
                a.set_title(f"{subset}: silhouette, 10,000-row sample")
    handles, labels = ax[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6)
    fig.suptitle("k-means on the three operating settings, k = 2 to 10 (five split seeds; the curves nearly coincide)")
    F.save(fig, "fig_s1_k_selection")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--out-dir", default=F.OUT)
    a = ap.parse_args(argv)
    F.OUT = os.path.abspath(a.out_dir)
    fig_s1_k_selection()
    print(json.dumps({k: len(v) for k, v in F.PLOTTED.items()}))


if __name__ == "__main__":
    main()
