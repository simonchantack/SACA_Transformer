"""Generate configs/*.yaml (one per reported training experiment). Run once; the YAML files are the release artifacts.

Every path is an environment-expandable string (``${NAME:-default}``) resolved against the release root at load time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "${SACA_FROZEN_ROOT:-../workspace/experiments}"
SUBS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [0, 1, 2, 3, 4]
P2 = "configs/protocols/protocol_P2.yaml"
P2R4 = "configs/protocols/protocol_P2_r4.yaml"
P2NS = "configs/protocols/protocol_P2_nosmooth.yaml"


def sel(name: str) -> str:
    return f"configs/selections/{name}_config_selection_frozen.json"


def paths(exp: str, frozen_rel: str) -> dict[str, Any]:
    return {
        "data_dir": "${SACA_DATA_DIR:-../workspace/code_base/Data}",
        "splits": "configs/data_splits.json",
        "output_dir": f"${{SACA_OUTPUT_DIR:-results/runs}}/{exp}",
        "frozen_dir": f"{FROZEN}/{frozen_rel}",
        "test_log": "${SACA_TEST_LOG:-results/test_eval_log.csv}",
    }


def write(name: str, cfg: dict[str, Any]) -> None:
    with open(ROOT / "configs" / f"{name}.yaml", "w", encoding="utf-8") as f:
        f.write(f"# {cfg['description']}\n")
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, width=120)


def base(exp: str, family: str, protocol: str, selection: str, frozen_rel: str, desc: str, concerns: list[str],
         arm: dict[str, Any] | None = None,
         test_read: str = "evaluate_test (purpose final) after selection_frozen.json") -> dict[str, Any]:
    return {"description": desc, "experiment_id": exp, "concerns": concerns, "family": family,
            "protocol": protocol, "selection": selection, "subsets": SUBS, "seeds": SEEDS, "arm": arm or {},
            "determinism": "strict (torch.use_deterministic_algorithms(True, warn_only=False), "
                           "CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic memory fill)",
            "test_read": test_read, "paths": paths(exp, frozen_rel)}


def main() -> None:
    write("exp_B6_theta_star", base(
        "exp_B6", "dualdim", P2, sel("exp_B6"), "round_02/exp_B6",
        "Corrected base model theta* (ProbSparse, parallel branches): P2-tuned configuration, 5 retrains per subset.",
        ["R0.1", "R0.2", "R0.3", "R0.4", "R2.7"]))
    pt = base("exp_B10_topup", "patchtst", P2, sel("exp_B10_topup"), "round_02/exp_B10_topup",
              "PatchTST baseline (official code, regression head). FD001, FD002, FD004: exp_B10_topup; FD003: exp_B10.",
              ["R2.2", "R1.19"])
    pt["selection_by_subset"] = {"FD003": sel("exp_B10")}
    pt["paths"]["frozen_dir_by_subset"] = {"FD003": f"{FROZEN}/round_02/exp_B10"}
    write("exp_B10_patchtst", pt)
    write("exp_B11_itransformer", base("exp_B11", "itransformer", P2, sel("exp_B11"), "round_02/exp_B11",
                                       "iTransformer baseline (official encoder, TSLib head).", ["R2.2", "R1.19"]))
    write("exp_B12_crossformer", base("exp_B12", "crossformer", P2, sel("exp_B12"), "round_02/exp_B12",
                                      "Crossformer baseline (official encoder, TSLib head).", ["R2.2", "R1.19"]))
    write("exp_B16_d4_main", base(
        "exp_B16", "r3_D4", P2, sel("exp_B16"), "round_03/exp_B16",
        "D4 SACA, the model of the paper: frozen configuration of the preregistered stage 2, 5 retrains per subset.",
        ["R1.3", "R1.31", "R0.11", "R2.16", "R1.39", "R2.17"]))
    write("exp_B17_d1_at_theta", base("exp_B17", "r3_D1", P2, sel("exp_B17"), "round_03/exp_B17",
                                      "D1 (exact attention) at the theta* configuration; validation only.",
                                      ["R1.21", "R0.19"], test_read="none (validation only)"))
    write("exp_B18_d4_sensor_order", base(
        "exp_B18", "r3_D4", P2, sel("exp_B18"), "round_03/exp_B18",
        "D4 sensor-order permutation retrains (post-result exploratory); order k on split seed k; validation only.",
        ["R2.16", "R1.39"], arm={"name": "sensor_order", "sensor_order": "configs/arms/sensor_orders.json"},
        test_read="none (validation only)"))
    write("exp_B19_theta_sensor_order", base(
        "exp_B19", "dualdim", P2, sel("exp_B19"), "round_03/exp_B19",
        "theta* sensor-order permutation retrains (post-result exploratory; round-2 determinism setting in the frozen "
        "runs); validation only.", ["R2.16", "R1.39"],
        arm={"name": "sensor_order", "sensor_order": "configs/arms/sensor_orders.json"},
        test_read="none (validation only)"))
    write("exp_V1_nosmooth", base("exp_V1_nosmooth", "r3_D4", P2NS, sel("exp_V1_nosmooth"), "round_04/exp_V1/nosmooth",
                                  "D4 without smoothing (one-factor ablation of the causal EMA).", ["R0.10"],
                                  arm={"name": "nosmooth"}, test_read="evaluate_test (purpose r4_final_ablation)"))
    write("exp_V2_lstm", base("exp_V2", "lstm", P2R4, sel("exp_V2"), "round_04/exp_V2",
                              "LSTM baseline under the same 12-trial rule.", ["R1.19"],
                              test_read="evaluate_test (purpose r4_final_baseline)"))
    write("exp_V3_ridge", base("exp_V3", "ridge", P2R4, sel("exp_V3"), "round_04/exp_V3",
                               "Ridge baseline (CPU, float64, closed form).", ["R1.19"],
                               test_read="evaluate_test (purpose r4_final_baseline)"))
    for f in ("0.25", "0.5", "0.75"):
        write(f"exp_A10_size_f{f}", base(
            f"exp_A10_f{f}", "r3_D4", P2, sel(f"exp_A10_f{f}"), f"round_04/exp_A10/f{f}",
            f"D4 training-size arm: nested engine sub-sample, fraction {f} of TR; pipeline refit on the sub-sample.",
            ["R0.18", "R1.13", "R2.13"],
            arm={"name": f"f{f}", "tr_subsample": {"file": "configs/arms/subsample_ids.json", "fraction": float(f)}},
            test_read="evaluate_test (purpose r4_final_ablation)"))
    write("exp_A11_layernorm", base("exp_A11", "r3_D4", P2, sel("exp_A11"), "round_04/exp_A11/layernorm",
                                    "D4 with LayerNorm in place of BatchNorm in every encoder layer.", ["R1.20"],
                                    arm={"name": "layernorm", "norm": "layer"},
                                    test_read="evaluate_test (purpose r4_final_ablation)"))
    write("exp_A12_L30", base(
        "exp_A12_L30", "r3_D4", P2, sel("exp_A12_L30"), "round_04/exp_A12/L30",
        "D4 window L = 30; on FD002-FD004 the P2 buildability rule (N_t >= 2) is bypassed (end-padded D4 has N' = 2); "
        "validation only.", ["R1.26"], arm={"name": "L30", "bypass_buildability": True, "bypass_L": 30},
        test_read="none (validation only)"))
    write("exp_A12_L40", base("exp_A12_L40", "r3_D4", P2, sel("exp_A12_L40"), "round_04/exp_A12/L40",
                              "D4 window L = 40; validation only.", ["R1.26"], arm={"name": "L40"},
                              test_read="none (validation only)"))
    for p in ("0.05", "0.1", "0.3"):
        write(f"exp_A13_dropout_p{p}", base(
            f"exp_A13_p{p}", "r3_D4", P2, sel(f"exp_A13_p{p}"), f"round_04/exp_A13/p{p}",
            f"D4 tied dropout p = {p} (dropout_t = dropout_c); validation only; the frozen configuration is unchanged.",
            ["R2.14", "R3.3", "R1.12"], arm={"name": f"p{p}"}, test_read="none (validation only)"))


if __name__ == "__main__":
    main()
