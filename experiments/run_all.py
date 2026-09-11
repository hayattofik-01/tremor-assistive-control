"""Run every experiment in order and assemble the main comparison table
(Section 20 of the project scope) into results/tables/main_comparison.csv."""
import importlib
import json

import pandas as pd

RESULTS_DIR = "results"


def _load(name):
    return importlib.import_module(f"experiments.{name}")


def main():
    mod01 = _load("01_core_analysis")
    mod02 = _load("02_control_comparison")
    mod03 = _load("03_mpc")
    mod04 = _load("04_system_identification")
    mod05 = _load("05_generalization")
    mod06 = _load("06_data_reduction")

    print("\n=== Experiment A: open-loop / structural analysis ===")
    summary = mod01.run()

    print("\n=== Experiment B: pole placement vs LQR ===")
    control_results, sensitivity = mod02.run()

    print("\n=== Experiment C: LQR vs constrained MPC + observer ===")
    mpc_results = mod03.run()

    print("\n=== Experiment D: NN vs kernel identification ===")
    ident_results = mod04.run()

    print("\n=== Experiment E: generalization ===")
    generalization_results = mod05.run()

    print("\n=== Experiment F: data reduction ===")
    reduction_df = mod06.run()

    rows = []
    for name, m in control_results.items():
        rows.append({
            "method": name,
            "tracking_rmse": m["tracking_rmse"],
            "residual_tremor_rms": m["residual_tremor_rms"],
            "control_effort": m["control_effort"],
            "max_abs_control": None,
            "n_violations": None,
        })
    for name in ("lqr_saturated", "mpc"):
        m = mpc_results[name]
        rows.append({
            "method": name,
            "tracking_rmse": m["tracking_rmse"],
            "residual_tremor_rms": m["residual_tremor_rms"],
            "control_effort": m["control_effort"],
            "max_abs_control": m["max_abs_control"],
            "n_violations": m["n_violations"],
        })

    main_table = pd.DataFrame(rows)
    main_table.to_csv(f"{RESULTS_DIR}/tables/main_comparison.csv", index=False)
    print("\nMain comparison table:\n", main_table)

    ident_table = pd.DataFrame(ident_results).T
    ident_table.to_csv(f"{RESULTS_DIR}/tables/identification_comparison.csv")
    print("\nIdentification comparison table:\n", ident_table)

    with open(f"{RESULTS_DIR}/tables/summary.json", "w") as f:
        json.dump({
            "structural_summary": {
                "eigenvalues": [complex(e).__repr__() for e in summary["eigenvalues"]],
                "controllability_rank": int(summary["controllability_rank"]),
                "observability_rank": int(summary["observability_rank"]),
            },
            "lqr_sensitivity": sensitivity,
            "generalization": generalization_results,
            "observer_velocity_rmse": mpc_results["observer_velocity_rmse"],
        }, f, indent=2, default=str)

    print("\nAll experiments completed. Figures in results/figures/, tables in results/tables/.")


if __name__ == "__main__":
    main()
