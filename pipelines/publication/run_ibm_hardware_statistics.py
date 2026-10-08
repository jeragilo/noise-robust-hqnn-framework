
"""
Consolidated five-seed IBM Marrakesh hardware statistical analysis.

Reads existing experimental results only.
Does not submit quantum hardware jobs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

ROOT = Path("results/publication/ibm_hardware_validation")
OUT = ROOT / "statistics"

SEEDS = [101, 202, 303, 404, 505]

KEYS = ["representation", "decoder"]

COMPARISONS = [
    (
        "R2_all_z",
        "random_forest",
        "R0_parity",
        "fixed_threshold",
    ),
    (
        "R3_zeng_xyz",
        "random_forest",
        "R0_parity",
        "fixed_threshold",
    ),
    (
        "R3_zeng_xyz",
        "random_forest",
        "R2_all_z",
        "random_forest",
    ),
]


# ============================================================
# LOAD AND VALIDATE RESULTS
# ============================================================

def load_results():

    frames = []
    reference_keys = None

    for seed in SEEDS:

        path = ROOT / f"seed{seed}_hardware_metrics.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing hardware results: {path}"
            )

        df = pd.read_csv(path)

        required_columns = {
            "seed",
            "representation",
            "decoder",
            "accuracy",
            "macro_f1",
        }

        if not required_columns.issubset(df.columns):
            raise RuntimeError(
                f"Missing columns in {path}"
            )

        if len(df) != 17:
            raise RuntimeError(
                f"Seed {seed}: expected 17 evaluations, "
                f"found {len(df)}"
            )

        if set(df["seed"]) != {seed}:
            raise RuntimeError(
                f"Incorrect seed values in {path}"
            )

        if df.duplicated(KEYS).any():
            raise RuntimeError(
                f"Duplicate configurations for seed {seed}"
            )

        if df[["accuracy", "macro_f1"]].isna().any().any():
            raise RuntimeError(
                f"Missing metrics for seed {seed}"
            )

        if not (
            df[["accuracy", "macro_f1"]]
            .apply(lambda col: col.between(0, 1))
            .all()
            .all()
        ):
            raise RuntimeError(
                f"Invalid metric range for seed {seed}"
            )

        current_keys = set(
            zip(df["representation"], df["decoder"])
        )

        if reference_keys is None:
            reference_keys = current_keys

        elif current_keys != reference_keys:
            raise RuntimeError(
                f"Configuration mismatch for seed {seed}"
            )

        frames.append(df)

        print(f"Seed {seed}: 17 evaluations verified")

    data = pd.concat(frames, ignore_index=True)

    if len(data) != 85:
        raise RuntimeError(
            "Expected 85 total model evaluations."
        )

    return data


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def compute_summary(data):

    summary = (
        data.groupby(KEYS, sort=True)
        .agg(
            n=("accuracy", "size"),
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            sd_macro_f1=("macro_f1", "std"),
        )
        .reset_index()
    )

    critical = stats.t.ppf(
        0.975,
        df=len(SEEDS) - 1,
    )

    summary["accuracy_ci_lower"] = (
        summary["mean_accuracy"]
        - critical
        * summary["sd_accuracy"]
        / np.sqrt(len(SEEDS))
    )

    summary["accuracy_ci_upper"] = (
        summary["mean_accuracy"]
        + critical
        * summary["sd_accuracy"]
        / np.sqrt(len(SEEDS))
    )

    # Keep reported accuracy intervals within valid bounds.
    summary["accuracy_ci_lower"] = (
        summary["accuracy_ci_lower"].clip(lower=0)
    )

    summary["accuracy_ci_upper"] = (
        summary["accuracy_ci_upper"].clip(upper=1)
    )

    summary["macro_f1_ci_lower"] = (
        summary["mean_macro_f1"]
        - critical
        * summary["sd_macro_f1"]
        / np.sqrt(len(SEEDS))
    ).clip(lower=0)

    summary["macro_f1_ci_upper"] = (
        summary["mean_macro_f1"]
        + critical
        * summary["sd_macro_f1"]
        / np.sqrt(len(SEEDS))
    ).clip(upper=1)

    return summary


# ============================================================
# PAIRED STATISTICAL COMPARISONS
# ============================================================

def compute_paired_comparisons(data):

    rows = []

    critical = stats.t.ppf(
        0.975,
        df=len(SEEDS) - 1,
    )

    for a_rep, a_dec, b_rep, b_dec in COMPARISONS:

        a = data[
            (data["representation"] == a_rep)
            & (data["decoder"] == a_dec)
        ].set_index("seed").loc[SEEDS]

        b = data[
            (data["representation"] == b_rep)
            & (data["decoder"] == b_dec)
        ].set_index("seed").loc[SEEDS]

        differences = (
            a["accuracy"].to_numpy()
            - b["accuracy"].to_numpy()
        )

        mean = float(np.mean(differences))

        sd = float(
            np.std(differences, ddof=1)
        )

        se = sd / np.sqrt(len(SEEDS))

        if sd > 0:

            t_stat, p_value = stats.ttest_1samp(
                differences,
                popmean=0,
            )

            ci_lower = mean - critical * se
            ci_upper = mean + critical * se

        else:

            t_stat = np.nan
            p_value = np.nan

            ci_lower = mean
            ci_upper = mean

        rows.append({
            "comparison": f"{a_rep} - {b_rep}",
            "decoder_a": a_dec,
            "decoder_b": b_dec,
            "n_seeds": len(SEEDS),
            "mean_difference": mean,
            "sd_difference": sd,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "t_statistic": t_stat,
            "p_value_unadjusted": p_value,
        })

    return pd.DataFrame(rows)


# ============================================================
# MEASUREMENT RESOURCE COMPARISON
# ============================================================

def measurement_resources():

    samples_per_seed = 100
    shots_per_basis = 1024

    rows = []

    configurations = [
        ("R0_parity", 1),
        ("R2_all_z", 1),
        ("R3_zeng_xyz", 3),
    ]

    for representation, bases in configurations:

        circuits_per_seed = (
            samples_per_seed * bases
        )

        shots_per_seed = (
            circuits_per_seed * shots_per_basis
        )

        rows.append({
            "representation": representation,
            "measurement_bases": bases,
            "circuits_per_seed": circuits_per_seed,
            "shots_per_circuit": shots_per_basis,
            "shots_per_seed": shots_per_seed,
            "shots_five_seeds": shots_per_seed * 5,
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("BEYOND PARITY | IBM HARDWARE STATISTICS")
    print("=" * 75)

    data = load_results()

    summary = compute_summary(data)

    paired = compute_paired_comparisons(data)

    resources = measurement_resources()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUT / "hardware_summary.csv",
        index=False,
    )

    paired.to_csv(
        OUT / "hardware_paired_comparisons.csv",
        index=False,
    )

    data.to_csv(
        OUT / "hardware_all_seed_results.csv",
        index=False,
    )

    resources.to_csv(
        OUT / "hardware_measurement_resources.csv",
        index=False,
    )

    print("\nFIVE-SEED SUMMARY")
    print("-" * 75)

    print(
        summary.round(4).to_string(index=False)
    )

    print("\nPAIRED COMPARISONS")
    print("-" * 75)

    print(
        paired.round(4).to_string(index=False)
    )

    print("\nMEASUREMENT RESOURCES")
    print("-" * 75)

    print(
        resources.to_string(index=False)
    )

    print("\nOUTPUT DIRECTORY:")
    print(OUT)

    print("\nANALYSIS COMPLETE")
    print("No IBM hardware jobs submitted.")


if __name__ == "__main__":
    main()

