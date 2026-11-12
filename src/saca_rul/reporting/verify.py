"""Regenerate frozen tables and post-hoc analyses and compare them with the frozen outputs.

Table check (combined table v4): the CSV must hash-match the frozen file byte for byte. The JSON and Markdown files
differ from the frozen ones in exactly the provenance fields that name the generator and the test-log state at
generation time. The check substitutes the frozen values of those fields into the regenerated text and then requires
the sha256 to equal the frozen sha256. It also checks that the first 401 rows of the present test log hash to the
"at generation" value (the log has only been appended to since: rows 402-404 are logged post-hoc readings).

Post-hoc check (R2.10, R0.23 + no-smoothing (c), no-smoothing (b)): the regenerated JSON must equal the frozen JSON
except for the keys that record the logged run (``log_row``, ``inputs_sha256_file``) and the protocol text.
No test log row is written; no GPU is used.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..config import ROOT
from .posthoc_common import WS

FROZEN_FINAL = Path(WS) / "experiments" / "final"
FROZEN_SHA = {
    "combined_table_v4.csv": "4f4d5368add2dbd6737e683290744f52c00eb014da457da55045bd939c02a628",
    "combined_table_v4.json": "189d784f96d79f017cc9886367b1b5c5773e3921ecc8355a4cd9bde71d29eb55",
    "combined_table_v4.md": "e994beca237516a318a759dc1dced813e2ee2be4ddf9b8cf7480d24d440566a6",
}
LOG_AT_GENERATION_SHA = "adfb5dcec327edf118854ada3e1e8120460ccf2c6ecb244236ed19ac909c1c15"
LOG_ROWS_AT_GENERATION = 401


def _env() -> dict[str, str]:
    import os

    e = dict(os.environ)
    e.update(PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="")
    return e


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_log_prefix_check() -> dict[str, Any]:
    """The first 401 data rows (plus header) of the present test log hash to the value recorded at generation."""
    data = (Path(WS) / "experiments" / "test_eval_log.csv").read_bytes()
    ends = [i + 1 for i, b in enumerate(data) if b == 10]
    n_rows = len(ends) - 1
    prefix_ok = sha256_bytes(data[: ends[LOG_ROWS_AT_GENERATION]]) == LOG_AT_GENERATION_SHA
    return {"rows_now": n_rows, "prefix_401_rows_hash_equals_generation_value": prefix_ok}


def _frozen_provenance(frozen_text: str, key_re: str) -> str:
    m = re.search(key_re, frozen_text)
    assert m, key_re
    return m.group(0)


def verify_tables(out_dir: str | Path) -> dict[str, Any]:
    """Run the v4 generator into ``out_dir`` and compare. Returns a dict of booleans and sha256 values."""
    out = Path(out_dir)
    subprocess.run([sys.executable, "-m", "saca_rul.reporting.tables", "--version", "v4", "--out-dir", str(out)],
                   check=True, capture_output=True, env=_env())
    res: dict[str, Any] = {}
    # CSV: direct
    res["combined_table_v4.csv"] = {"sha256": sha256_bytes((out / "combined_table_v4.csv").read_bytes()),
                                    "hash_match_direct": sha256_bytes((out / "combined_table_v4.csv").read_bytes())
                                    == FROZEN_SHA["combined_table_v4.csv"]}
    # JSON and Markdown: substitute the frozen provenance values
    fj = (FROZEN_FINAL / "combined_table_v4.json").read_bytes().decode("utf-8")
    nj = (out / "combined_table_v4.json").read_bytes().decode("utf-8")
    for key in ("generator", "generator_sha256", "final_rows", "test_eval_log_sha256_at_generation"):
        frozen_line = _frozen_provenance(fj, rf'"{key}": "[^"]*"')
        nj = nj.replace(re.search(rf'"{key}": "[^"]*"', nj).group(0), frozen_line, 1)  # type: ignore[union-attr]
    res["combined_table_v4.json"] = {"hash_match_after_provenance_substitution": sha256_bytes(nj.encode("utf-8"))
                                     == FROZEN_SHA["combined_table_v4.json"]}
    fm = (FROZEN_FINAL / "combined_table_v4.md").read_bytes().decode("utf-8")
    nm = (out / "combined_table_v4.md").read_bytes().decode("utf-8")
    pat = r"test_eval_log\.csv sha256 at generation [0-9a-f]{64}"
    nm = re.sub(pat, lambda _m: _frozen_provenance(fm, pat), nm, count=1)
    res["combined_table_v4.md"] = {"hash_match_after_provenance_substitution": sha256_bytes(nm.encode("utf-8"))
                                   == FROZEN_SHA["combined_table_v4.md"]}
    res["substituted_provenance_fields"] = ["generator", "generator_sha256", "final_rows",
                                            "test_eval_log_sha256_at_generation (json and md)"]
    res["test_log_check"] = test_log_prefix_check()
    return res


POSTHOC = {
    "R2.10": ("saca_rul.reporting.posthoc_r210", "posthoc_R2.10/hier_bootstrap_results.json",
              "hier_bootstrap_results.json", ("log_row", "inputs_sha256_file")),
    "R0.23_c": ("saca_rul.reporting.posthoc_r023", "posthoc_R0.23_nosmooth_c/padding_nosmooth_results.json",
                "padding_nosmooth_results.json", ("log_row", "inputs_sha256_file")),
    "nosmooth_b": ("saca_rul.reporting.posthoc_valsim", "nosmooth_b_valsim/valsim_results.json",
                   "valsim_results.json", ("protocol",)),
}


def verify_posthoc(name: str, out_dir: str | Path) -> dict[str, Any]:
    """Regenerate one post-hoc analysis into ``out_dir`` and compare it with the frozen JSON."""
    mod, frozen_rel, fname, ignore = POSTHOC[name]
    subprocess.run([sys.executable, "-m", mod, "--out-dir", str(out_dir)], check=True, capture_output=True, env=_env())
    a = json.loads((FROZEN_FINAL / frozen_rel).read_text(encoding="utf-8"))
    b = json.loads((Path(out_dir) / fname).read_text(encoding="utf-8"))
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return {"analysis": name, "keys_differing": diff, "only_ignored_keys_differ": set(diff) <= set(ignore),
            "ignored_keys": list(ignore)}


def verify_shipped(frozen_dir: str | Path, sums_file: str | Path) -> dict[str, bool]:
    """Check files against a ``sha256sum`` list (used for the shipped R1.24 outputs)."""
    ok: dict[str, bool] = {}
    for line in Path(sums_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        p = Path(frozen_dir) / name
        ok[name] = p.exists() and sha256_bytes(p.read_bytes()) == h
    return ok
