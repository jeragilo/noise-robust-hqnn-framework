"""
Pilot statistical analysis for the HQNN readout publication study.

IMPORTANT:
This analysis uses only three matched seeds and is intended for
experimental-design guidance, not final publication inference.

It analyzes the fixed-decoder control results and reports:
- paired accuracy differences
- standard deviation of paired differences
- 95% t confidence intervals
- paired t statistics and p-values
- paired Cohen's dz
- win counts across matched seeds

Primary comparisons:
1. Rich readouts vs fixed parity
2. XYZ vs single-basis alternatives
3. Full engineered representation vs simpler representations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats


INPUT_PATH = (
    Path("results")
    / "publication"
    / "fixed_decoder_control"
    / "fixed_decoder_control.json"
)

OUTPUT_DIR = (
    Path("results")
    / "publication"
    / "pilot_statistics"
)

NOISE_LEVELS = [0.00, 0.05, 0.10]

DECODERS = [
    "logistic_regression",
    "random_forest",
]

PRIMARY_REPRESENTATIONS = [
    "R2_all_z",
    "R3_zeng_xyz",
    "R4_probabilities",
    "R5_z_zz",
    "R6_prob_z_zz",
    "R7_full",
]


def load_results() -> Dict[str, object]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing fixed-decoder results: {INPUT_PATH}"
        )

    with open(INPUT_PATH, "r") as file:
        return json.load(file)


def get_runs_for_noise(
    data: Dict[str, object],
    noise_level: float,
) -> List[Dict[str, object]]:
    return sorted(
        [
            run
            for run in data["runs"]
            if np.isclose(
                run["noise_level"],
                noise_level,
            )
        ],
        key=lambda run: run["seed"],
    )


def parity_values(
    runs: List[Dict[str, object]],
) -> np.ndarray:
    return np.asarray(
        [
            run["R0_parity"]["accuracy"]
            for run in runs
        ],
        dtype=float,
    )


def representation_values(
    runs: List[Dict[str, object]],
    decoder: str,
    representation: str,
) -> np.ndarray:
    return np.asarray(
        [
            run["decoders"][decoder][representation]["accuracy"]
            for run in runs
        ],
        dtype=float,
    )


def paired_statistics(
    reference: np.ndarray,
    comparison: np.ndarray,
) -> Dict[str, object]:
    """
    Positive difference means comparison outperforms reference.
    """

    if len(reference) != len(comparison):
        raise ValueError(
            "Paired samples must have equal length."
        )

    differences = comparison - reference

    n = len(differences)

    mean_difference = float(
        np.mean(differences)
    )

    sd_difference = float(
        np.std(
            differences,
            ddof=1,
        )
    ) if n > 1 else 0.0

    standard_error = (
        sd_difference / np.sqrt(n)
        if n > 1
        else 0.0
    )

    if n > 1:
        critical_t = float(
            stats.t.ppf(
                0.975,
                df=n - 1,
            )
        )

        ci_lower = (
            mean_difference
            - critical_t * standard_error
        )

        ci_upper = (
            mean_difference
            + critical_t * standard_error
        )
    else:
        ci_lower = mean_difference
        ci_upper = mean_difference

    if n > 1 and sd_difference > 0:
        cohen_dz = (
            mean_difference
            / sd_difference
        )
    elif mean_difference == 0:
        cohen_dz = 0.0
    else:
        cohen_dz = None

    if n > 1:
        t_result = stats.ttest_rel(
            comparison,
            reference,
        )

        t_statistic = float(
            t_result.statistic
        )

        p_value = float(
            t_result.pvalue
        )
    else:
        t_statistic = None
        p_value = None

    return {
        "n": n,
        "reference_values": reference.tolist(),
        "comparison_values": comparison.tolist(),
        "paired_differences": differences.tolist(),
        "mean_difference": mean_difference,
        "sd_difference": sd_difference,
        "ci_95": [
            float(ci_lower),
            float(ci_upper),
        ],
        "cohen_dz": (
            float(cohen_dz)
            if cohen_dz is not None
            else None
        ),
        "paired_t_statistic": t_statistic,
        "paired_t_p_value": p_value,
        "comparison_wins": int(
            np.sum(differences > 0)
        ),
        "ties": int(
            np.sum(
                np.isclose(
                    differences,
                    0.0,
                )
            )
        ),
        "reference_wins": int(
            np.sum(differences < 0)
        ),
    }


def analyze_vs_parity(
    data: Dict[str, object],
) -> List[Dict[str, object]]:
    results = []

    for noise_level in NOISE_LEVELS:
        runs = get_runs_for_noise(
            data,
            noise_level,
        )

        parity = parity_values(runs)

        for decoder in DECODERS:
            for representation in PRIMARY_REPRESENTATIONS:

                values = representation_values(
                    runs,
                    decoder,
                    representation,
                )

                results.append(
                    {
                        "noise_level": noise_level,
                        "decoder": decoder,
                        "reference": "R0_parity",
                        "comparison": representation,
                        **paired_statistics(
                            parity,
                            values,
                        ),
                    }
                )

    return results


def analyze_xyz_tradeoff(
    data: Dict[str, object],
) -> List[Dict[str, object]]:
    """
    Quantify the accuracy gain from the three-basis XYZ strategy
    relative to representations obtainable from Z-basis measurements.
    """

    results = []

    alternatives = [
        "R2_all_z",
        "R4_probabilities",
        "R5_z_zz",
        "R6_prob_z_zz",
        "R7_full",
    ]

    for noise_level in NOISE_LEVELS:
        runs = get_runs_for_noise(
            data,
            noise_level,
        )

        for decoder in DECODERS:

            xyz = representation_values(
                runs,
                decoder,
                "R3_zeng_xyz",
            )

            for alternative in alternatives:

                alt = representation_values(
                    runs,
                    decoder,
                    alternative,
                )

                results.append(
                    {
                        "noise_level": noise_level,
                        "decoder": decoder,
                        "reference": alternative,
                        "comparison": "R3_zeng_xyz",
                        **paired_statistics(
                            alt,
                            xyz,
                        ),
                    }
                )

    return results


def print_key_results(
    vs_parity: List[Dict[str, object]],
    xyz_tradeoff: List[Dict[str, object]],
) -> None:

    print("\n" + "=" * 80)
    print("PILOT PAIRED STATISTICS")
    print("=" * 80)

    print(
        "\nRandom Forest: representation gains over fixed parity"
    )

    for row in vs_parity:
        if (
            row["decoder"] == "random_forest"
            and row["comparison"]
            in {
                "R2_all_z",
                "R3_zeng_xyz",
                "R5_z_zz",
                "R6_prob_z_zz",
                "R7_full",
            }
        ):
            print(
                f"noise={row['noise_level']:.2f} | "
                f"{row['comparison']:18s} | "
                f"gain={row['mean_difference']:+.4f} | "
                f"95% CI=[{row['ci_95'][0]:+.4f}, "
                f"{row['ci_95'][1]:+.4f}] | "
                f"dz={row['cohen_dz']} | "
                f"wins={row['comparison_wins']}/{row['n']}"
            )

    print(
        "\nRandom Forest: XYZ gain over single-basis alternatives"
    )

    for row in xyz_tradeoff:
        if row["decoder"] == "random_forest":
            print(
                f"noise={row['noise_level']:.2f} | "
                f"XYZ vs {row['reference']:18s} | "
                f"gain={row['mean_difference']:+.4f} | "
                f"95% CI=[{row['ci_95'][0]:+.4f}, "
                f"{row['ci_95'][1]:+.4f}] | "
                f"wins={row['comparison_wins']}/{row['n']}"
            )

    print(
        "\nLogistic Regression: R7 vs parity"
    )

    for row in vs_parity:
        if (
            row["decoder"] == "logistic_regression"
            and row["comparison"] == "R7_full"
        ):
            print(
                f"noise={row['noise_level']:.2f} | "
                f"gain={row['mean_difference']:+.4f} | "
                f"95% CI=[{row['ci_95'][0]:+.4f}, "
                f"{row['ci_95'][1]:+.4f}] | "
                f"dz={row['cohen_dz']} | "
                f"wins={row['comparison_wins']}/{row['n']}"
            )


def run_pipeline() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_results()

    vs_parity = analyze_vs_parity(
        data
    )

    xyz_tradeoff = analyze_xyz_tradeoff(
        data
    )

    output = {
        "description": (
            "Pilot paired statistical analysis of the "
            "fixed-decoder HQNN readout experiment."
        ),
        "warning": (
            "Only three matched seeds are available. "
            "Confidence intervals, p-values, and effect sizes "
            "are exploratory and must not be treated as final "
            "publication inference."
        ),
        "comparisons_vs_parity": vs_parity,
        "xyz_measurement_tradeoff": xyz_tradeoff,
    }

    output_path = (
        OUTPUT_DIR
        / "pilot_statistics.json"
    )

    with open(
        output_path,
        "w",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print_key_results(
        vs_parity,
        xyz_tradeoff,
    )

    print("\nPilot statistical analysis complete.")
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    run_pipeline()
