"""Supplementary Fig. S1 (k selection) and the colour-blind-safe palette (RULES section 37)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from saca_rul.config import ROOT
from saca_rul.reporting import fig_s1_k_selection as S1
from saca_rul.reporting import figures_v02 as F
from saca_rul.reporting.posthoc_common import WS

REPORTING = Path(__file__).resolve().parents[1] / "src" / "saca_rul" / "reporting"
KS = [str(k) for k in range(2, 11)]


@pytest.mark.frozen
def test_s1_plotted_arrays_equal_saved_k_selection_outputs(tmp_path: Path) -> None:
    F.PLOTTED.clear()
    S1.main(["--out-dir", str(tmp_path)])
    d = F.PLOTTED["fig_s1_k_selection"]
    assert len(d) == 20  # 2 subsets x 2 metrics x 5 seeds
    for subset in ("FD002", "FD004"):
        src = json.loads((Path(WS) / f"experiments/round_04_pre/exp_C2/outputs/k_selection_{subset}.json").read_text(encoding="utf-8"))
        for seed in "01234":
            for m in ("inertia", "silhouette"):
                np.testing.assert_array_equal(np.asarray(d[f"{subset}/{m}/seed{seed}"], float), [src["per_seed"][seed]["per_k"][k][m] for k in KS])
    assert (tmp_path / "fig_s1_k_selection_v02.png").stat().st_size > 1000


def test_s1_shipped_files_are_300_dpi_and_have_no_internal_identifiers() -> None:
    from PIL import Image

    im = Image.open(ROOT / "figures" / "fig_s1_k_selection_v02.png")
    assert abs(im.info["dpi"][0] - 300) < 0.01 and im.size[0] <= round(7.2 * 300)
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # type: ignore[no-redef]
    with fitz.open(ROOT / "figures" / "fig_s1_k_selection_v02.pdf") as doc:
        text = "\n".join(p.get_text() for p in doc)
        sizes = [sp["size"] for p in doc for b in p.get_text("dict")["blocks"] for ln in b.get("lines", [])
                 for sp in ln["spans"] if sp["text"].strip()]
    assert not re.search(r"exp_|RULES|\.json|\.csv", text) and "_" not in text
    assert min(sizes) >= F.MIN_PT - 1e-6


@pytest.mark.parametrize("script", ["figures.py", "figures_v02.py", "fig_s1_k_selection.py"])
def test_figure_scripts_use_no_red_or_green_colours(script: str) -> None:
    """The red and green hex codes and matplotlib's default green/red cycle colours were removed (RULES section 37)."""
    text = (REPORTING / script).read_text(encoding="utf-8")
    for bad in ("#b42318", "#1d7a3a", "#d62728", "#2ca02c", "#b8860b", "#1f77b4", "#ff7f0e"):
        assert bad not in text.lower()
