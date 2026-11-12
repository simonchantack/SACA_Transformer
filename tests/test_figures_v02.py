"""Version-02 figures (RULES section 32): plotted values equal the frozen source values, and no figure text carries an
internal identifier. The value test needs the frozen workspace; the text scan needs only the shipped PDFs."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from saca_rul.config import ROOT
from saca_rul.reporting import figures_v02 as F
from saca_rul.reporting.posthoc_common import WS

WSP = Path(WS)
FORBIDDEN = re.compile(r"exp_|RULES|D4|\.json|\.csv|option [bc]")
SUBS = ["FD001", "FD002", "FD003", "FD004"]


def jl(rel: str) -> dict:
    return json.loads((WSP / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plotted(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, object]]:
    F.PLOTTED.clear()
    F.OUT = str(tmp_path_factory.mktemp("figs"))
    F.main(["--out-dir", F.OUT])
    return F.PLOTTED


def close(a: object, b: object) -> None:
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=1e-12)


@pytest.mark.frozen
def test_all_sixteen_figures_are_recorded(plotted: dict[str, dict[str, object]]) -> None:
    assert len(F.FIGURES) == 16
    assert len(list(Path(F.OUT).glob("*_v02.png"))) == 16 and len(list(Path(F.OUT).glob("*_v02.pdf"))) == 16


@pytest.mark.frozen
def test_fig1_final_rmse_equals_combined_table(plotted: dict[str, dict[str, object]]) -> None:
    v4 = jl("experiments/final/combined_table_v4.json")
    for s in SUBS:
        for k in ("d4", "theta_star", "patchtst", "itransformer", "crossformer", "lstm", "ridge"):
            close(plotted["fig_v01_1_final_rmse"][f"{s}/{k}"], v4["rows"][k]["per_subset"][s]["per_seed"]["RMSE"])


@pytest.mark.frozen
def test_fig2_pairs_equal_primary_test_and_zero_pair_is_kept(plotted: dict[str, dict[str, object]]) -> None:
    v4 = jl("experiments/final/combined_table_v4.json")
    pairs = v4["preregistered_primary_exp_B16"]["pairs"]
    d = plotted["fig_v01_2_primary_pairs"]
    close(d["delta"], [p["delta"] for p in pairs])
    close(d["margin"], v4["non_inferiority_exp_B16"]["margin_rmse"])
    close(d["mean_delta"], v4["preregistered_primary_exp_B16"]["mean_delta"])
    near_zero = [p for p in pairs if abs(p["delta"]) < 0.001]
    assert near_zero, "the near-zero pair (FD004 seed 1) must still be in the data"
    assert near_zero[0]["subset"] == "FD004" and near_zero[0]["seed"] == 1


@pytest.mark.frozen
def test_fig3_bootstrap_equals_frozen_results(plotted: dict[str, dict[str, object]]) -> None:
    hb = jl("experiments/final/posthoc_R2.10/hier_bootstrap_results.json")
    for s in SUBS:
        for k in ("theta_star", "patchtst", "itransformer", "crossformer", "lstm", "ridge", "best_baseline"):
            r = hb["per_subset"][s][k]["RMSE"]
            close(plotted["fig_v01_3_hier_bootstrap"][f"{s}/{k}"], [r["delta"], *r["ci95"]])


@pytest.mark.frozen
def test_fig4_nosmooth_equals_frozen_results(plotted: dict[str, dict[str, object]]) -> None:
    pc = jl("experiments/final/posthoc_R0.23_nosmooth_c/padding_nosmooth_results.json")
    vb = jl("experiments/final/nosmooth_b_valsim/valsim_results.json")
    d = plotted["fig_v01_4_nosmooth_padding"]
    close(d["c/padded_mean"], [pc["per_subset"][s]["B_nosmooth_gap"]["gap_none_minus_ema_RMSE_padded"]["mean"] for s in SUBS])
    close(d["c/unpadded_mean"], [pc["per_subset"][s]["B_nosmooth_gap"]["gap_none_minus_ema_RMSE_unpadded"]["mean"] for s in SUBS])
    close(d["b/padded_mean"], [vb["summary"][s]["padded"]["gap_none_minus_ema"]["mean"] for s in SUBS])
    close(d["b/unpadded_ci"], [vb["summary"][s]["unpadded"]["gap_none_minus_ema"]["t95"] for s in SUBS])


@pytest.mark.frozen
def test_fig5_and_fig6_equal_frozen_metrics(plotted: dict[str, dict[str, object]]) -> None:
    a10 = jl("experiments/round_04/exp_A10/metrics.json")
    arms = ["f0.25", "f0.5", "f0.75", "control"]
    for split in ("val", "test"):
        for s in SUBS:
            close(plotted["fig_v01_5_size_curve"][f"{split}/{s}/mean"], [a10["per_subset"][s][a][f"{split}_RMSE"]["mean"] for a in arms])
            close(plotted["fig_v01_5_size_curve"][f"{split}/{s}/std"], [a10["per_subset"][s][a][f"{split}_RMSE"]["std"] for a in arms])
    a12 = jl("experiments/round_04/exp_A12/metrics.json")
    for s in SUBS:
        diff = np.array(a12["per_subset"][s]["L30"]["val_RMSE"]["per_seed"]) - np.array(a12["per_subset"][s]["control"]["val_RMSE"]["per_seed"])
        close(plotted["fig_v01_6_win_drop"][f"L30/{s}"][0], diff.mean())


@pytest.mark.frozen
def test_fig7_regime_and_chain_equal_frozen_results(plotted: dict[str, dict[str, object]]) -> None:
    rr = jl("experiments/final/posthoc_R1.24/regime_results.json")
    keys = sorted(rr["regime_names"], key=int)
    close(plotted["fig_v01_7_fd004_regime"]["d4"], [rr["summary"]["regimes"][k]["Score_share"]["d4"] for k in keys])
    s1 = jl("experiments/round_03/exp_B15/stage1_summary.json")
    close(plotted["fig_r04_1_chain"]["median_delta"], [s1["hypotheses"][h]["median_delta"] for h in ("H1", "H2", "H3", "H4")])
    close(plotted["fig_r04_1_chain"]["S_D"][1:], [s1["S_D"][d] for d in ("D1", "D2", "D3", "D4")])


@pytest.mark.frozen
def test_hand_entered_diagnosis_and_probe_values_occur_in_their_source_tables(plotted: dict[str, dict[str, object]]) -> None:
    shapes = (WSP / "experiments/round_03/exp_A9/shapes.md").read_text(encoding="utf-8")
    for v in plotted["fig_r03_1_diagnosis"]["share_theta"] + plotted["fig_r03_1_diagnosis"]["share_d2"]:  # type: ignore[operator]
        assert f"{v:.3f}" in shapes
    plan = (WSP / "experiments/round_03/exp_A5/plan.md").read_text(encoding="utf-8")
    for key in ("ms_theta", "ms_d1"):
        for v in plotted["fig_r03_2_probe"][key]:  # type: ignore[attr-defined]
            assert f"{v:.1f}" in plan
    for key in ("mem_theta", "mem_d1"):
        for v in plotted["fig_r03_2_probe"][key]:  # type: ignore[attr-defined]
            assert f"{int(v):,}" in plan


@pytest.mark.frozen
def test_patch_counts_equal_shape_output(plotted: dict[str, dict[str, object]]) -> None:
    recs = json.loads((ROOT / "results" / "shapes" / "d4_shapes.json").read_text(encoding="utf-8"))
    for key, v in plotted["fig_r04_4_window_tokens"].items():
        lab, L = key.split("/L")
        subs = ["FD001"] if lab == "FD001" else ["FD002", "FD003", "FD004"]
        r = next(x for x in recs if x["L"] == int(L) and x["subset"] in subs)
        assert v == [r["P_t"], r["S_t"], r["N_no_padding"], r["N_prime_end_padding"], r["N_prime_end_padding"] * r["S_t"] - int(L)]  # type: ignore[comparison-overlap]


def pdf_text(p: Path) -> str:
    # PyMuPDF is a [dev] dependency: a missing module is a failure, not a skip, so the scan cannot be silently bypassed.
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # type: ignore[no-redef]
    with fitz.open(p) as doc:
        return "\n".join(page.get_text() for page in doc)


@pytest.mark.parametrize("name", sorted(F.FIGURES))
def test_shipped_v02_pdf_has_no_internal_identifiers(name: str) -> None:
    p = ROOT / "figures" / f"{name}_v02.pdf"
    text = pdf_text(p)
    assert len(text.strip()) > 20
    assert not FORBIDDEN.search(text), FORBIDDEN.search(text)


def test_v02_files_are_300_dpi_png_and_pdf() -> None:
    from PIL import Image

    pngs = sorted((ROOT / "figures").glob("*_v02.png"))
    assert len(pngs) == 17  # 16 figures plus supplementary Fig. S1
    for p in pngs:
        im = Image.open(p)
        assert abs(im.info["dpi"][0] - 300) < 0.01 and im.size[0] <= round(7.2 * 300)
        assert p.with_suffix(".pdf").stat().st_size > 1000


HYPHEN_MINUS = re.compile(r"(?<![\w.)])-(?=\d)")


@pytest.mark.parametrize("name", sorted(F.FIGURES))
def test_shipped_v02_pdf_typography(name: str) -> None:
    """True minus sign, no literal subscript underscore, no ambiguous 'tied p' (RULES 32; final_check_08 A4, A7)."""
    text = pdf_text(ROOT / "figures" / f"{name}_v02.pdf")
    assert not HYPHEN_MINUS.search(text), HYPHEN_MINUS.search(text)
    assert "_" not in text, "literal underscore: use mathtext subscripts"
    assert "tied p " not in text and "tok," not in text and "b256" not in text and not re.search(r"space max\b", text)


def test_probe_figure_names_the_gpu() -> None:
    text = pdf_text(ROOT / "figures" / "fig_r03_2_probe_v02.pdf")
    assert "RTX 5070 Ti Laptop GPU" in " ".join(text.split())


def test_padding_hatch_edge_differs_from_fill() -> None:
    """The hatch is drawn in the edge colour; it must differ from the fill or it is invisible (A4)."""
    assert F.PAD_EDGE.lower() != "#f3d9b1"


def test_script_scale_is_not_changed_by_import_and_is_restored() -> None:
    """The private matplotlib constant is 0.7 after import and after drawing (override only inside ``script_scale``)."""
    import matplotlib._mathtext as mt

    assert float(getattr(mt, "SHRINK_FACTOR", 0.7)) == 0.7
    with F.script_scale(1.0):
        assert float(getattr(mt, "SHRINK_FACTOR", 1.0)) == 1.0
    assert float(getattr(mt, "SHRINK_FACTOR", 0.7)) == 0.7


@pytest.mark.parametrize("name", sorted(F.FIGURES))
def test_shipped_v02_pdf_every_text_span_is_at_least_seven_points(name: str) -> None:
    """Output guard: the rendered size of every non-blank span, subscripts included, is at least 7.0 pt."""
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf  # type: ignore[no-redef]
    sizes = []
    with pymupdf.open(ROOT / "figures" / f"{name}_v02.pdf") as doc:
        for page in doc:
            for b in page.get_text("dict")["blocks"]:
                for ln in b.get("lines", []):
                    sizes += [sp["size"] for sp in ln["spans"] if sp["text"].strip()]
    assert sizes and min(sizes) >= F.MIN_PT - 1e-6, min(sizes)


def test_regenerated_png_is_byte_identical_to_shipped(tmp_path: Path) -> None:
    """Drawing with the scoped override reproduces a shipped figure with subscripts byte for byte."""
    import matplotlib._mathtext as mt

    old_out = F.OUT
    F.OUT = str(tmp_path)
    try:
        F.fig_r04_4_window_tokens()
    finally:
        F.OUT = old_out
    assert float(getattr(mt, "SHRINK_FACTOR", 0.7)) == 0.7
    new = (tmp_path / "fig_r04_4_window_tokens_v02.png").read_bytes()
    assert new == (ROOT / "figures" / "fig_r04_4_window_tokens_v02.png").read_bytes()
