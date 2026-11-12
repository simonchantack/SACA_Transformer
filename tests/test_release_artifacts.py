"""Release-artifact checks that need no workspace: shipped frozen files and hashes, figure print size, stage shapes, and
the AST equivalence of the shipped modules with the tree of the reproduction check."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from saca_rul.config import ROOT

FROZEN = ROOT / "results" / "frozen"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_frozen_tables_and_script_of_record_are_shipped_with_their_hashes() -> None:
    assert sha(FROZEN / "combined_table_v4.csv").startswith("4f4d5368")
    assert sha(FROZEN / "combined_table_v4.json").startswith("189d784f")
    assert sha(FROZEN / "combined_table_v4.md").startswith("e994beca")
    assert sha(FROZEN / "run_regime_split.py").startswith("573d5189")


def test_per_seed_metrics_files_match_their_recorded_hashes() -> None:
    lines = (FROZEN / "metrics" / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    for line in lines:
        h, name = line.split()[:2]
        assert sha(FROZEN / "metrics" / name) == h
        d = json.loads((FROZEN / "metrics" / name).read_text(encoding="utf-8"))
        assert "per_seed" in json.dumps(d)


@pytest.mark.parametrize("name", [p.stem for p in sorted((ROOT / "figures").glob("*.png"))])
def test_figure_is_print_size_and_has_a_pdf(name: str) -> None:
    im = Image.open(ROOT / "figures" / f"{name}.png")
    assert im.size[0] <= round(7.2 * 300) and abs(im.info["dpi"][0] - 300) < 0.01
    assert (ROOT / "figures" / f"{name}.pdf").stat().st_size > 1000


def test_there_are_sixteen_figures_per_version() -> None:
    """Version 01 (kept): 16 figures. Version 02 (RULES section 32, suffix _v02): 16 figures plus supplementary Fig. S1 (17)."""
    pngs = list((ROOT / "figures").glob("*.png"))
    assert len([p for p in pngs if not p.stem.endswith("_v02")]) == 16
    assert len([p for p in pngs if p.stem.endswith("_v02")]) == 17


def test_stage_shapes_and_padding_arithmetic() -> None:
    recs = json.loads((ROOT / "results" / "shapes" / "d4_shapes.json").read_text(encoding="utf-8"))
    for r in recs:
        st = {(s["stage"], s["tensor"]): s["shape"] for s in r["stages"]}
        B, C, L, S, N1, d = r["batch"], r["C"], r["L"], r["S_t"], r["N_prime_end_padding"], r["d_model"]
        assert st[("0 input", "x (B, C, L)")] == [B, C, L]
        assert [B * C, N1, r["P_t"]] in st.values()  # unfold
        assert [B * C, N1 + 1, d] in st.values()  # anchor + patches
        assert st[("6 head", "RUL prediction (B,)")] == [B]
        assert N1 * S - L >= 0
    md = (ROOT / "results" / "shapes" / "d4_shapes.md").read_text(encoding="utf-8")
    assert "Stage-by-stage tensor shapes" in md and "after S end-padding copies and unfold" in md


def test_shipped_modules_on_the_repro_path_equal_the_launch_tree(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("ast_equivalence", ROOT / "scripts" / "ast_equivalence.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    on_path = mod.repro_path_modules()
    assert len(on_path) >= 15
    for rel in sorted(on_path):
        txt = mod.LAUNCH / (rel + ".txt")
        shipped = ROOT / rel
        eq, _ = mod.compare(txt, shipped)
        if not eq and mod.sha(txt) != mod.sha(shipped):
            ok, _lines = mod.explain(rel, txt, shipped)
            assert ok, rel
