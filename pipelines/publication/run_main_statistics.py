"""
Final statistical analysis for the publication-scale HQNN readout study.

This script performs statistical inference on the frozen results produced by
run_main_readout_study.py. It does NOT execute quantum circuits or modify the
locked experimental protocol.

Primary analyses:
1. Descriptive statistics
2. Rich representations vs fixed parity
3. Rich representations vs learned parity
4. Logistic regression vs random forest
5. XYZ vs single-basis alternatives
6. Clean-to-noisy robustness
7. Accuracy and macro-F1 consistency
8. Holm multiple-comparison correction
9. Paired effect sizes
10. Measurement-cost tradeoff
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# PATHS
# =============================================================================

RESULTS_DIR = (
    Path("results")
    / "publication"
    / "main_readout_study"
)

INPUT_PATH = (
    RESULTS_DIR
    / "main_readout_study_enriched.csv"
)

STATISTICS_DIR = (
    RESULTS_DIR
    / "statistics"
)

DESCRIPTIVE_PATH = (
    STATISTICS_DIR
    / "descriptive_statistics.csv"
)

PAIRED_PATH = (
    STATISTICS_DIR
    / "paired_comparisons.csv"
)

DECODER_PATH = (
    STATISTICS_DIR
    / "decoder_comparisons.csv"
)

ROBUSTNESS_PATH = (
    STATISTICS_DIR
    / "noise_robustness.csv"
)

MEASUREMENT_COST_PATH = (
    STATISTICS_DIR
    / "measurement_cost_analysis.csv"
)

SUMMARY_PATH = (
    STATISTICS_DIR
    / "main_statistics_summary.json"
)


# =============================================================================
# CONSTANTS
# =============================================================================

NOISE_LEVELS = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.20,
]

RICH_REPRESENTATIONS = [
    "R2_all_z",
    "R3_zeng_xyz",
    "R4_probabilities",
    "R5_z_zz",
    "R6_prob_z_zz",
    "R7_full",
]

SINGLE_BASIS_REPRESENTATIONS = [
    "R2_all_z",
    "R4_probabilities",
    "R5_z_zz",
    "R6_prob_z_zz",
    "R7_full",
]

LEARNED_REPRESENTATIONS = [
    "R0L_learned_parity",
    "R1_single_z",
    "R2_all_z",
    "R3_zeng_xyz",
    "R4_probabilities",
    "R5_z_zz",
    "R6_prob_z_zz",
    "R7_full",
]

METRICS = [
    "accuracy",
    "macro_f1",
]


# =============================================================================
# HELPERS
# =============================================================================

def mean_ci(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    mean = float(
        np.mean(values)
    )

    if n <= 1:
        return {
            "n": n,
            "mean": mean,
            "std": 0.0,
            "se": 0.0,
            "ci95_lower": mean,
            "ci95_upper": mean,
        }

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    se = float(
        std
        / np.sqrt(n)
    )

    critical_t = float(
        stats.t.ppf(
            0.975,
            df=n - 1,
        )
    )

    margin = (
        critical_t
        * se
    )

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_lower": float(
            mean - margin
        ),
        "ci95_upper": float(
            mean + margin
        ),
    }


def paired_effect(
    a: np.ndarray,
    b: np.ndarray,
) -> Dict[str, Any]:
    """
    Paired comparison where positive difference means A > B.
    """

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if len(a) != len(b):
        raise ValueError(
            "Paired arrays must have equal length."
        )

    differences = (
        a - b
    )

    n = len(
        differences
    )

    descriptive = mean_ci(
        differences
    )

    if n > 1:
        t_result = stats.ttest_rel(
            a,
            b,
        )

        diff_std = float(
            np.std(
                differences,
                ddof=1,
            )
        )

        if diff_std > 0:
            dz = float(
                np.mean(differences)
                / diff_std
            )
        else:
            dz = None

    else:
        t_result = None
        dz = None

    try:
        if np.allclose(
            differences,
            0.0,
        ):
            wilcoxon_statistic = 0.0
            wilcoxon_p = 1.0
        else:
            wilcoxon_result = (
                stats.wilcoxon(
                    differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            )

            wilcoxon_statistic = float(
                wilcoxon_result.statistic
            )

            wilcoxon_p = float(
                wilcoxon_result.pvalue
            )

    except ValueError:
        wilcoxon_statistic = None
        wilcoxon_p = None

    wins = int(
        np.sum(
            differences > 0
        )
    )

    ties = int(
        np.sum(
            np.isclose(
                differences,
                0.0,
            )
        )
    )

    losses = int(
        np.sum(
            differences < 0
        )
    )

    return {
        "n": n,
        "mean_difference": descriptive[
            "mean"
        ],
        "std_difference": descriptive[
            "std"
        ],
        "se_difference": descriptive[
            "se"
        ],
        "ci95_lower": descriptive[
            "ci95_lower"
        ],
        "ci95_upper": descriptive[
            "ci95_upper"
        ],
        "paired_t_statistic": (
            float(
                t_result.statistic
            )
            if t_result is not None
            else None
        ),
        "paired_t_p": (
            float(
                t_result.pvalue
            )
            if t_result is not None
            else None
        ),
        "cohens_dz": dz,
        "wilcoxon_statistic": (
            wilcoxon_statistic
        ),
        "wilcoxon_p": (
            wilcoxon_p
        ),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def holm_adjust(
    p_values: List[float],
) -> List[float]:
    """
    Holm step-down family-wise error correction.
    """

    p = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(
        p
    )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, index in enumerate(
        order
    ):
        multiplier = (
            m - rank
        )

        value = min(
            1.0,
            multiplier
            * p[index],
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted[index] = (
            running_max
        )

    return adjusted.tolist()


def paired_vectors(
    df: pd.DataFrame,
    representation_a: str,
    decoder_a: str,
    representation_b: str,
    decoder_b: str,
    noise_level: float,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    a = df[
        (
            df["representation"]
            == representation_a
        )
        & (
            df["decoder"]
            == decoder_a
        )
        & (
            np.isclose(
                df["noise_level"],
                noise_level,
            )
        )
    ][
        [
            "seed",
            metric,
        ]
    ].rename(
        columns={
            metric: "a",
        }
    )

    b = df[
        (
            df["representation"]
            == representation_b
        )
        & (
            df["decoder"]
            == decoder_b
        )
        & (
            np.isclose(
                df["noise_level"],
                noise_level,
            )
        )
    ][
        [
            "seed",
            metric,
        ]
    ].rename(
        columns={
            metric: "b",
        }
    )

    merged = a.merge(
        b,
        on="seed",
        how="inner",
        validate="one_to_one",
    ).sort_values(
        "seed"
    )

    if len(merged) != 20:
        raise RuntimeError(
            "Expected 20 paired seeds for "
            f"{representation_a}/{decoder_a} "
            f"vs {representation_b}/{decoder_b} "
            f"at noise={noise_level}, "
            f"got {len(merged)}."
        )

    return (
        merged["a"].to_numpy(
            dtype=float
        ),
        merged["b"].to_numpy(
            dtype=float
        ),
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataset(
    df: pd.DataFrame,
) -> None:
    required_columns = {
        "seed",
        "noise_type",
        "noise_level",
        "shots",
        "num_qubits",
        "architecture",
        "representation",
        "decoder",
        "measurement_bases",
        "feature_dimension",
        "accuracy",
        "macro_f1",
        "accuracy_drop_from_clean",
        "accuracy_retention",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    if len(df) != 1700:
        raise RuntimeError(
            "Expected exactly 1700 rows, "
            f"found {len(df)}."
        )

    seeds = sorted(
        df["seed"].unique()
    )

    if len(seeds) != 20:
        raise RuntimeError(
            "Expected 20 seeds, "
            f"found {len(seeds)}."
        )

    observed_noise = sorted(
        df["noise_level"].unique()
    )

    if not np.allclose(
        observed_noise,
        NOISE_LEVELS,
    ):
        raise RuntimeError(
            "Unexpected noise levels: "
            f"{observed_noise}"
        )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "seed",
                "noise_level",
                "representation",
                "decoder",
            ]
        ).sum()
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate experimental rows "
            f"detected: {duplicate_count}"
        )


# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

def build_descriptive_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    group_columns = [
        "noise_level",
        "representation",
        "decoder",
        "measurement_bases",
        "feature_dimension",
    ]

    for keys, group in df.groupby(
        group_columns,
        sort=True,
    ):
        (
            noise_level,
            representation,
            decoder,
            measurement_bases,
            feature_dimension,
        ) = keys

        for metric in METRICS:
            result = mean_ci(
                group[
                    metric
                ].to_numpy(
                    dtype=float
                )
            )

            rows.append(
                {
                    "noise_level": float(
                        noise_level
                    ),
                    "representation": (
                        representation
                    ),
                    "decoder": decoder,
                    "metric": metric,
                    "measurement_bases": int(
                        measurement_bases
                    ),
                    "feature_dimension": int(
                        feature_dimension
                    ),
                    **result,
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PAIRED REPRESENTATION COMPARISONS
# =============================================================================

def build_paired_comparisons(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    for noise_level in NOISE_LEVELS:
        for metric in METRICS:
            for decoder in [
                "logistic_regression",
                "random_forest",
            ]:
                for representation in (
                    RICH_REPRESENTATIONS
                ):
                    a, b = paired_vectors(
                        df=df,
                        representation_a=representation,
                        decoder_a=decoder,
                        representation_b="R0_parity",
                        decoder_b="fixed_threshold",
                        noise_level=noise_level,
                        metric=metric,
                    )

                    result = paired_effect(
                        a,
                        b,
                    )

                    rows.append(
                        {
                            "comparison_family": (
                                "rich_vs_fixed_parity"
                            ),
                            "noise_level": (
                                noise_level
                            ),
                            "metric": metric,
                            "decoder": decoder,
                            "representation_a": (
                                representation
                            ),
                            "representation_b": (
                                "R0_parity"
                            ),
                            **result,
                        }
                    )

            for decoder in [
                "logistic_regression",
                "random_forest",
            ]:
                for representation in (
                    RICH_REPRESENTATIONS
                ):
                    a, b = paired_vectors(
                        df=df,
                        representation_a=representation,
                        decoder_a=decoder,
                        representation_b=(
                            "R0L_learned_parity"
                        ),
                        decoder_b=decoder,
                        noise_level=noise_level,
                        metric=metric,
                    )

                    result = paired_effect(
                        a,
                        b,
                    )

                    rows.append(
                        {
                            "comparison_family": (
                                "rich_vs_learned_parity"
                            ),
                            "noise_level": (
                                noise_level
                            ),
                            "metric": metric,
                            "decoder": decoder,
                            "representation_a": (
                                representation
                            ),
                            "representation_b": (
                                "R0L_learned_parity"
                            ),
                            **result,
                        }
                    )

            for decoder in [
                "logistic_regression",
                "random_forest",
            ]:
                for alternative in (
                    SINGLE_BASIS_REPRESENTATIONS
                ):
                    a, b = paired_vectors(
                        df=df,
                        representation_a=(
                            "R3_zeng_xyz"
                        ),
                        decoder_a=decoder,
                        representation_b=alternative,
                        decoder_b=decoder,
                        noise_level=noise_level,
                        metric=metric,
                    )

                    result = paired_effect(
                        a,
                        b,
                    )

                    rows.append(
                        {
                            "comparison_family": (
                                "xyz_vs_single_basis"
                            ),
                            "noise_level": (
                                noise_level
                            ),
                            "metric": metric,
                            "decoder": decoder,
                            "representation_a": (
                                "R3_zeng_xyz"
                            ),
                            "representation_b": (
                                alternative
                            ),
                            **result,
                        }
                    )

    result_df = pd.DataFrame(
        rows
    )

    result_df[
        "paired_t_p_holm"
    ] = np.nan

    result_df[
        "wilcoxon_p_holm"
    ] = np.nan

    family_columns = [
        "comparison_family",
        "noise_level",
        "metric",
        "decoder",
    ]

    for _, index in result_df.groupby(
        family_columns
    ).groups.items():
        index = list(
            index
        )

        t_values = (
            result_df.loc[
                index,
                "paired_t_p",
            ]
            .astype(float)
            .tolist()
        )

        result_df.loc[
            index,
            "paired_t_p_holm",
        ] = holm_adjust(
            t_values
        )

        w_values = (
            result_df.loc[
                index,
                "wilcoxon_p",
            ]
            .astype(float)
            .tolist()
        )

        result_df.loc[
            index,
            "wilcoxon_p_holm",
        ] = holm_adjust(
            w_values
        )

    return result_df


# =============================================================================
# DECODER CAPACITY
# =============================================================================

def build_decoder_comparisons(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    for noise_level in NOISE_LEVELS:
        for metric in METRICS:
            for representation in (
                LEARNED_REPRESENTATIONS
            ):
                a, b = paired_vectors(
                    df=df,
                    representation_a=representation,
                    decoder_a="random_forest",
                    representation_b=representation,
                    decoder_b="logistic_regression",
                    noise_level=noise_level,
                    metric=metric,
                )

                result = paired_effect(
                    a,
                    b,
                )

                rows.append(
                    {
                        "noise_level": (
                            noise_level
                        ),
                        "metric": metric,
                        "representation": (
                            representation
                        ),
                        "decoder_a": (
                            "random_forest"
                        ),
                        "decoder_b": (
                            "logistic_regression"
                        ),
                        **result,
                    }
                )

    result_df = pd.DataFrame(
        rows
    )

    result_df[
        "paired_t_p_holm"
    ] = np.nan

    result_df[
        "wilcoxon_p_holm"
    ] = np.nan

    for _, index in result_df.groupby(
        [
            "noise_level",
            "metric",
        ]
    ).groups.items():
        index = list(
            index
       )
        t_values = (
            result_df.loc[
                index,
                "paired_t_p",
            ]
            .astype(float)
            .tolist()
        )

        result_df.loc[
            index,
            "paired_t_p_holm",
        ] = holm_adjust(
            t_values
        )

        w_values = (
            result_df.loc[
                index,
                "wilcoxon_p",
            ]
            .astype(float)
            .tolist()
        )

        result_df.loc[
            index,
            "wilcoxon_p_holm",
        ] = holm_adjust(
            w_values
        )

    return result_df


# =============================================================================
# NOISE ROBUSTNESS
# =============================================================================

def build_noise_robustness(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare every noisy condition against the clean condition using
    paired seeds.

    Positive mean_difference means clean performance is higher than
    noisy performance.
    """

    rows: List[
        Dict[str, Any]
    ] = []

    configurations = (
        df[
            [
                "representation",
                "decoder",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "representation",
                "decoder",
            ]
        )
    )

    for _, config in configurations.iterrows():
        representation = config[
            "representation"
        ]

        decoder = config[
            "decoder"
        ]

        for metric in METRICS:
            clean = df[
                (
                    df["representation"]
                    == representation
                )
                & (
                    df["decoder"]
                    == decoder
                )
                & (
                    np.isclose(
                        df["noise_level"],
                        0.0,
                    )
                )
            ][
                [
                    "seed",
                    metric,
                ]
            ].rename(
                columns={
                    metric: "clean",
                }
            )

            for noise_level in (
                NOISE_LEVELS[1:]
            ):
                noisy = df[
                    (
                        df["representation"]
                        == representation
                    )
                    & (
                        df["decoder"]
                        == decoder
                    )
                    & (
                        np.isclose(
                            df["noise_level"],
                            noise_level,
                        )
                    )
                ][
                    [
                        "seed",
                        metric,
                    ]
                ].rename(
                    columns={
                        metric: "noisy",
                    }
                )

                merged = clean.merge(
                    noisy,
                    on="seed",
                    how="inner",
                    validate="one_to_one",
                ).sort_values(
                    "seed"
                )

                if len(merged) != 20:
                    raise RuntimeError(
                        "Expected 20 paired seeds "
                        "for robustness analysis, "
                        f"got {len(merged)} for "
                        f"{representation}/{decoder} "
                        f"at noise={noise_level}."
                    )

                clean_values = (
                    merged[
                        "clean"
                    ].to_numpy(
                        dtype=float
                    )
                )

                noisy_values = (
                    merged[
                        "noisy"
                    ].to_numpy(
                        dtype=float
                    )
                )

                result = paired_effect(
                    clean_values,
                    noisy_values,
                )

                with np.errstate(
                    divide="ignore",
                    invalid="ignore",
                ):
                    retention = (
                        noisy_values
                        / clean_values
                    )

                finite_retention = (
                    retention[
                        np.isfinite(
                            retention
                        )
                    ]
                )

                retention_stats = (
                    mean_ci(
                        finite_retention
                    )
                    if len(
                        finite_retention
                    ) > 0
                    else {
                        "n": 0,
                        "mean": None,
                        "std": None,
                        "se": None,
                        "ci95_lower": None,
                        "ci95_upper": None,
                    }
                )

                rows.append(
                    {
                        "representation": (
                            representation
                        ),
                        "decoder": decoder,
                        "metric": metric,
                        "noise_level": (
                            noise_level
                        ),
                        "clean_mean": float(
                            np.mean(
                                clean_values
                            )
                        ),
                        "noisy_mean": float(
                            np.mean(
                                noisy_values
                            )
                        ),
                        "mean_drop": result[
                            "mean_difference"
                        ],
                        "drop_ci95_lower": (
                            result[
                                "ci95_lower"
                            ]
                        ),
                        "drop_ci95_upper": (
                            result[
                                "ci95_upper"
                            ]
                        ),
                        "paired_t_statistic": (
                            result[
                                "paired_t_statistic"
                            ]
                        ),
                        "paired_t_p": result[
                            "paired_t_p"
                        ],
                        "cohens_dz": result[
                            "cohens_dz"
                        ],
                        "wilcoxon_statistic": (
                            result[
                                "wilcoxon_statistic"
                            ]
                        ),
                        "wilcoxon_p": result[
                            "wilcoxon_p"
                        ],
                        "wins_clean_over_noisy": (
                            result[
                                "wins"
                            ]
                        ),
                        "ties": result[
                            "ties"
                        ],
                        "losses_clean_below_noisy": (
                            result[
                                "losses"
                            ]
                        ),
                        "mean_retention": (
                            retention_stats[
                                "mean"
                            ]
                        ),
                        "retention_ci95_lower": (
                            retention_stats[
                                "ci95_lower"
                            ]
                        ),
                        "retention_ci95_upper": (
                            retention_stats[
                                "ci95_upper"
                            ]
                        ),
                    }
                )

    result_df = pd.DataFrame(
        rows
    )

    result_df[
        "paired_t_p_holm"
    ] = np.nan

    result_df[
        "wilcoxon_p_holm"
    ] = np.nan

    for _, index in result_df.groupby(
        [
            "representation",
            "decoder",
            "metric",
        ]
    ).groups.items():
        index = list(
            index
        )

        t_values = (
            result_df.loc[
                index,
                "paired_t_p",
            ]
            .astype(float)
            .tolist()
        )

        result_df.loc[
            index,
            "paired_t_p_holm",
        ] = holm_adjust(
            t_values
        )

        w_values = (
            result_df.loc[
                index,
                "wilcoxon_p",
            ]
            .astype(float)
            .tolist()
        )

        result_df.loc[
            index,
            "wilcoxon_p_holm",
        ] = holm_adjust(
            w_values
        )

    return result_df


# =============================================================================
# MEASUREMENT-COST ANALYSIS
# =============================================================================

def build_measurement_cost_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare the three-basis XYZ representation against each
    single-basis rich alternative.

    This is a performance-cost analysis. It does not claim formal
    statistical equivalence or non-inferiority.
    """

    rows: List[
        Dict[str, Any]
    ] = []

    for noise_level in NOISE_LEVELS:
        for metric in METRICS:
            for decoder in [
                "logistic_regression",
                "random_forest",
            ]:
                xyz_subset = df[
                    (
                        df["representation"]
                        == "R3_zeng_xyz"
                    )
                    & (
                        df["decoder"]
                        == decoder
                    )
                    & (
                        np.isclose(
                            df["noise_level"],
                            noise_level,
                        )
                    )
                ]

                xyz_bases = int(
                    xyz_subset[
                        "measurement_bases"
                    ].iloc[0]
                )

                xyz_dimension = int(
                    xyz_subset[
                        "feature_dimension"
                    ].iloc[0]
                )

                for alternative in (
                    SINGLE_BASIS_REPRESENTATIONS
                ):
                    alternative_subset = df[
                        (
                            df["representation"]
                            == alternative
                        )
                        & (
                            df["decoder"]
                            == decoder
                        )
                        & (
                            np.isclose(
                                df["noise_level"],
                                noise_level,
                            )
                        )
                    ]

                    alternative_bases = int(
                        alternative_subset[
                            "measurement_bases"
                        ].iloc[0]
                    )

                    alternative_dimension = int(
                        alternative_subset[
                            "feature_dimension"
                        ].iloc[0]
                    )

                    xyz_values, alt_values = (
                        paired_vectors(
                            df=df,
                            representation_a=(
                                "R3_zeng_xyz"
                            ),
                            decoder_a=decoder,
                            representation_b=(
                                alternative
                            ),
                            decoder_b=decoder,
                            noise_level=(
                                noise_level
                            ),
                            metric=metric,
                        )
                    )

                    result = paired_effect(
                        xyz_values,
                        alt_values,
                    )

                    basis_ratio = (
                        float(
                            xyz_bases
                            / alternative_bases
                        )
                        if alternative_bases
                        > 0
                        else None
                    )

                    rows.append(
                        {
                            "noise_level": (
                                noise_level
                            ),
                            "metric": metric,
                            "decoder": decoder,
                            "xyz_representation": (
                                "R3_zeng_xyz"
                            ),
                            "alternative": (
                                alternative
                            ),
                            "xyz_bases": (
                                xyz_bases
                            ),
                            "alternative_bases": (
                                alternative_bases
                            ),
                            "basis_ratio_xyz_to_alt": (
                                basis_ratio
                            ),
                            "xyz_feature_dimension": (
                                xyz_dimension
                            ),
                            "alternative_feature_dimension": (
                                alternative_dimension
                            ),
                            "xyz_mean": float(
                                np.mean(
                                    xyz_values
                                )
                            ),
                            "alternative_mean": float(
                                np.mean(
                                    alt_values
                                )
                            ),
                            "xyz_minus_alternative": (
                                result[
                                    "mean_difference"
                                ]
                            ),
                            "difference_ci95_lower": (
                                result[
                                    "ci95_lower"
                                ]
                            ),
                            "difference_ci95_upper": (
                                result[
                                    "ci95_upper"
                                ]
                            ),
                            "paired_t_statistic": (
                                result[
                                    "paired_t_statistic"
                                ]
                            ),
                            "paired_t_p": (
                                result[
                                    "paired_t_p"
                                ]
                            ),
                            "cohens_dz": (
                                result[
                                    "cohens_dz"
                                ]
                            ),
                            "wilcoxon_statistic": (
                                result[
                                    "wilcoxon_statistic"
                                ]
                            ),
                            "wilcoxon_p": (
                                result[
                                    "wilcoxon_p"
                                ]
                            ),
                            "xyz_wins": result[
                                "wins"
                            ],
                            "ties": result[
                                "ties"
                            ],
                            "xyz_losses": result[
                                "losses"
                            ],
                        }
                    )

    result_df = pd.DataFrame(
        rows
    )

    result_df[
        "paired_t_p_holm"
    ] = np.nan

    result_df[
        "wilcoxon_p_holm"
    ] = np.nan

    for _, index in result_df.groupby(
        [
            "noise_level",
            "metric",
            "decoder",
        ]
    ).groups.items():
        index = list(
            index
        )

        t_values = (
            result_df.loc[
                index,
                "paired_t_p",
            ]
            .astype(float)
            .tolist()
        )

        result_df.loc[
            index,
            "paired_t_p_holm",
        ] = holm_adjust(
            t_values
        )

        w_values = (
            result_df.loc[
                index,
                "wilcoxon_p",
            ]
            .astype(float)
            .tolist()
        )

        result_df.loc[
            index,
            "wilcoxon_p_holm",
        ] = holm_adjust(
            w_values
        )

    return result_df


# =============================================================================
# SUMMARY EXTRACTION
# =============================================================================

def dataframe_records(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame into JSON-safe records.
    """

    clean = df.copy()

    clean = clean.replace(
        {
            np.nan: None,
            np.inf: None,
            -np.inf: None,
        }
    )

    return clean.to_dict(
        orient="records"
    )


def build_key_findings(
    descriptive: pd.DataFrame,
    paired: pd.DataFrame,
    decoder: pd.DataFrame,
    robustness: pd.DataFrame,
    measurement_cost: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Extract a compact set of numerical findings without making
    unsupported publication claims.
    """

    findings: Dict[
        str,
        Any,
    ] = {}

    clean_accuracy = descriptive[
        (
            descriptive["metric"]
            == "accuracy"
        )
        & (
            np.isclose(
                descriptive[
                    "noise_level"
                ],
                0.0,
            )
        )
    ].copy()

    clean_accuracy = (
        clean_accuracy.sort_values(
            "mean",
            ascending=False,
        )
    )

    findings[
        "clean_accuracy_ranking"
    ] = dataframe_records(
        clean_accuracy[
            [
                "representation",
                "decoder",
                "measurement_bases",
                "feature_dimension",
                "mean",
                "std",
                "ci95_lower",
                "ci95_upper",
            ]
        ]
    )

    high_noise_accuracy = descriptive[
        (
            descriptive["metric"]
            == "accuracy"
        )
        & (
            np.isclose(
                descriptive[
                    "noise_level"
                ],
                0.20,
            )
        )
    ].copy()

    high_noise_accuracy = (
        high_noise_accuracy.sort_values(
            "mean",
            ascending=False,
        )
    )

    findings[
        "noise_020_accuracy_ranking"
    ] = dataframe_records(
        high_noise_accuracy[
            [
                "representation",
                "decoder",
                "measurement_bases",
                "feature_dimension",
                "mean",
                "std",
                "ci95_lower",
                "ci95_upper",
            ]
        ]
    )

    parity_tests = paired[
        (
            paired[
                "comparison_family"
            ]
            == "rich_vs_fixed_parity"
        )
        & (
            paired["metric"]
            == "accuracy"
        )
    ].copy()

    findings[
        "rich_vs_fixed_parity_accuracy"
    ] = dataframe_records(
        parity_tests
    )

    learned_parity_tests = paired[
        (
            paired[
                "comparison_family"
            ]
            == "rich_vs_learned_parity"
        )
        & (
            paired["metric"]
            == "accuracy"
        )
    ].copy()

    findings[
        "rich_vs_learned_parity_accuracy"
    ] = dataframe_records(
        learned_parity_tests
    )

    decoder_accuracy = decoder[
        decoder["metric"]
        == "accuracy"
    ].copy()

    findings[
        "decoder_capacity_accuracy"
    ] = dataframe_records(
        decoder_accuracy
    )

    high_noise_cost = measurement_cost[
        (
            measurement_cost[
                "metric"
            ]
            == "accuracy"
        )
        & (
            np.isclose(
                measurement_cost[
                    "noise_level"
                ],
                0.20,
            )
        )
    ].copy()

    findings[
        "measurement_cost_noise_020"
    ] = dataframe_records(
        high_noise_cost
    )

    robustness_accuracy = robustness[
        robustness["metric"]
        == "accuracy"
    ].copy()

    findings[
        "noise_robustness_accuracy"
    ] = dataframe_records(
        robustness_accuracy
    )

    return findings


# =============================================================================
# CONSOLE REPORT
# =============================================================================

def print_key_report(
    descriptive: pd.DataFrame,
    paired: pd.DataFrame,
    decoder: pd.DataFrame,
    robustness: pd.DataFrame,
    measurement_cost: pd.DataFrame,
) -> None:
    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL MAIN-STUDY STATISTICAL ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        "\nRandom Forest accuracy:"
    )

    for noise_level in NOISE_LEVELS:
        subset = descriptive[
            (
                descriptive["metric"]
                == "accuracy"
            )
            & (
                descriptive["decoder"]
                == "random_forest"
            )
            & (
                np.isclose(
                    descriptive[
                        "noise_level"
                    ],
                    noise_level,
                )
            )
        ].sort_values(
            "mean",
            ascending=False,
        )

        print(
            f"\nNoise={noise_level:.2f}"
        )

        for _, row in subset.iterrows():
            print(
                f"{row['representation']:24s} | "
                f"mean={row['mean']:.4f} | "
                f"95% CI=["
                f"{row['ci95_lower']:.4f}, "
                f"{row['ci95_upper']:.4f}] | "
                f"bases={int(row['measurement_bases'])}"
            )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "XYZ vs one-basis alternatives | "
        "Random Forest | accuracy"
    )

    print(
        "-" * 80
    )

    subset = measurement_cost[
        (
            measurement_cost[
                "decoder"
            ]
            == "random_forest"
        )
        & (
            measurement_cost[
                "metric"
            ]
            == "accuracy"
        )
    ].sort_values(
        [
            "noise_level",
            "alternative",
        ]
    )

    for _, row in subset.iterrows():
        print(
            f"noise={row['noise_level']:.2f} | "
            f"XYZ vs {row['alternative']:18s} | "
            f"diff={row['xyz_minus_alternative']:+.4f} | "
            f"95% CI=["
            f"{row['difference_ci95_lower']:+.4f}, "
            f"{row['difference_ci95_upper']:+.4f}] | "
            f"Holm p={row['paired_t_p_holm']:.6g} | "
            f"bases="
            f"{int(row['xyz_bases'])}:"
            f"{int(row['alternative_bases'])}"
        )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Decoder effect | RF minus LR | accuracy"
    )

    print(
        "-" * 80
    )

    subset = decoder[
        decoder["metric"]
        == "accuracy"
    ].sort_values(
        [
            "noise_level",
            "representation",
        ]
    )

    for _, row in subset.iterrows():
                print(
            f"noise={row['noise_level']:.2f} | "
            f"{row['representation']:24s} | "
            f"RF-LR={row['mean_difference']:+.4f} | "
            f"95% CI=["
            f"{row['ci95_lower']:+.4f}, "
            f"{row['ci95_upper']:+.4f}] | "
            f"Holm p={row['paired_t_p_holm']:.6g} | "
            f"dz={row['cohens_dz']}"
        )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Rich representations vs fixed parity | "
        "accuracy"
    )

    print(
        "-" * 80
    )

    subset = paired[
        (
            paired[
                "comparison_family"
            ]
            == "rich_vs_fixed_parity"
        )
        & (
            paired["metric"]
            == "accuracy"
        )
    ].sort_values(
        [
            "noise_level",
            "decoder",
            "representation_a",
        ]
    )

    for _, row in subset.iterrows():
        print(
            f"noise={row['noise_level']:.2f} | "
            f"{row['decoder']:20s} | "
            f"{row['representation_a']:18s} | "
            f"gain={row['mean_difference']:+.4f} | "
            f"95% CI=["
            f"{row['ci95_lower']:+.4f}, "
            f"{row['ci95_upper']:+.4f}] | "
            f"Holm p={row['paired_t_p_holm']:.6g} | "
            f"wins={int(row['wins'])}/"
            f"{int(row['n'])}"
        )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Noise robustness | clean minus noisy | "
        "Random Forest accuracy"
    )

    print(
        "-" * 80
    )

    subset = robustness[
        (
            robustness["decoder"]
            == "random_forest"
        )
        & (
            robustness["metric"]
            == "accuracy"
        )
    ].sort_values(
        [
            "representation",
            "noise_level",
        ]
    )

    for _, row in subset.iterrows():
        print(
            f"{row['representation']:24s} | "
            f"noise={row['noise_level']:.2f} | "
            f"drop={row['mean_drop']:+.4f} | "
            f"95% CI=["
            f"{row['drop_ci95_lower']:+.4f}, "
            f"{row['drop_ci95_upper']:+.4f}] | "
            f"retention={row['mean_retention']:.4f} | "
            f"Holm p={row['paired_t_p_holm']:.6g}"
        )


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(
    descriptive: pd.DataFrame,
    paired: pd.DataFrame,
    decoder: pd.DataFrame,
    robustness: pd.DataFrame,
    measurement_cost: pd.DataFrame,
    summary: Dict[str, Any],
) -> None:
    STATISTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptive.to_csv(
        DESCRIPTIVE_PATH,
        index=False,
    )

    paired.to_csv(
        PAIRED_PATH,
        index=False,
    )

    decoder.to_csv(
        DECODER_PATH,
        index=False,
    )

    robustness.to_csv(
        ROBUSTNESS_PATH,
        index=False,
    )

    measurement_cost.to_csv(
        MEASUREMENT_COST_PATH,
        index=False,
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            allow_nan=False,
        )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline() -> None:
    print(
        "=" * 80
    )

    print(
        "FINAL STATISTICAL ANALYSIS | "
        "FROZEN MAIN READOUT STUDY"
    )

    print(
        "=" * 80
    )

    print(
        f"\nLoading: {INPUT_PATH}"
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Frozen main-study results were not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        "Validating frozen dataset..."
    )

    validate_dataset(
        df
    )

    seeds = sorted(
        int(seed)
        for seed in df[
            "seed"
        ].unique()
    )

    print(
        "Dataset validation passed."
    )

    print(
        f"Seeds: {len(seeds)}"
    )

    print(
        f"Noise levels: {NOISE_LEVELS}"
    )

    print(
        "No quantum circuits will be executed."
    )

    # -------------------------------------------------------------------------
    # Descriptive statistics
    # -------------------------------------------------------------------------

    print(
        "\nBuilding descriptive statistics..."
    )

    descriptive = (
        build_descriptive_statistics(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Paired representation comparisons
    # -------------------------------------------------------------------------

    print(
        "Building paired representation comparisons..."
    )

    paired = (
        build_paired_comparisons(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Decoder-capacity comparisons
    # -------------------------------------------------------------------------

    print(
        "Building decoder-capacity comparisons..."
    )

    decoder = (
        build_decoder_comparisons(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Noise robustness
    # -------------------------------------------------------------------------

    print(
        "Building clean-to-noisy robustness analysis..."
    )

    robustness = (
        build_noise_robustness(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Measurement cost
    # -------------------------------------------------------------------------

    print(
        "Building measurement-cost analysis..."
    )

    measurement_cost = (
        build_measurement_cost_analysis(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Internal consistency checks
    # -------------------------------------------------------------------------

    expected_descriptive_rows = (
        17
        * len(NOISE_LEVELS)
        * len(METRICS)
    )

    if len(
        descriptive
    ) != expected_descriptive_rows:
        raise RuntimeError(
            "Unexpected number of descriptive rows: "
            f"expected {expected_descriptive_rows}, "
            f"found {len(descriptive)}."
        )

    expected_decoder_rows = (
        len(NOISE_LEVELS)
        * len(METRICS)
        * len(LEARNED_REPRESENTATIONS)
    )

    if len(
        decoder
    ) != expected_decoder_rows:
        raise RuntimeError(
            "Unexpected number of decoder-comparison rows: "
            f"expected {expected_decoder_rows}, "
            f"found {len(decoder)}."
        )

    expected_robustness_rows = (
        17
        * (
            len(NOISE_LEVELS)
            - 1
        )
        * len(METRICS)
    )

    if len(
        robustness
    ) != expected_robustness_rows:
        raise RuntimeError(
            "Unexpected number of robustness rows: "
            f"expected {expected_robustness_rows}, "
            f"found {len(robustness)}."
        )

    expected_measurement_cost_rows = (
        len(NOISE_LEVELS)
        * len(METRICS)
        * 2
        * len(
            SINGLE_BASIS_REPRESENTATIONS
        )
    )

    if len(
        measurement_cost
    ) != expected_measurement_cost_rows:
        raise RuntimeError(
            "Unexpected number of measurement-cost rows: "
            f"expected {expected_measurement_cost_rows}, "
            f"found {len(measurement_cost)}."
        )

    # -------------------------------------------------------------------------
    # Build compact machine-readable summary
    # -------------------------------------------------------------------------

    print(
        "Extracting key numerical findings..."
    )

    key_findings = (
        build_key_findings(
            descriptive=descriptive,
            paired=paired,
            decoder=decoder,
            robustness=robustness,
            measurement_cost=measurement_cost,
        )
    )

    summary: Dict[
        str,
        Any,
    ] = {
        "analysis": (
            "final_main_readout_statistics"
        ),
        "input_file": str(
            INPUT_PATH
        ),
        "protocol_status": (
            "frozen"
        ),
        "quantum_simulation_executed": (
            False
        ),
        "n_rows": int(
            len(df)
        ),
        "n_seeds": int(
            len(seeds)
        ),
        "seeds": seeds,
        "noise_levels": [
            float(x)
            for x in NOISE_LEVELS
        ],
        "metrics": list(
            METRICS
        ),
        "multiple_comparison_method": (
            "Holm step-down correction"
        ),
        "paired_parametric_test": (
            "paired t-test"
        ),
        "paired_nonparametric_check": (
            "Wilcoxon signed-rank test"
        ),
        "paired_effect_size": (
            "Cohen's dz"
        ),
        "confidence_interval": (
            "95% Student-t confidence interval"
        ),
        "measurement_cost_note": (
            "XYZ uses three measurement bases; "
            "single-basis alternatives use one. "
            "This analysis reports performance-cost "
            "contrasts but does not claim formal "
            "equivalence or non-inferiority."
        ),
        "output_counts": {
            "descriptive_rows": int(
                len(descriptive)
            ),
            "paired_comparison_rows": int(
                len(paired)
            ),
            "decoder_comparison_rows": int(
                len(decoder)
            ),
            "robustness_rows": int(
                len(robustness)
            ),
            "measurement_cost_rows": int(
                len(
                    measurement_cost
                )
            ),
        },
        "key_findings": (
            key_findings
        ),
    }

    # -------------------------------------------------------------------------
    # Save everything before printing interpretation-oriented report
    # -------------------------------------------------------------------------

    print(
        "Saving statistical outputs..."
    )

    save_outputs(
        descriptive=descriptive,
        paired=paired,
        decoder=decoder,
        robustness=robustness,
        measurement_cost=measurement_cost,
        summary=summary,
    )

    # -------------------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------------------

    print_key_report(
        descriptive=descriptive,
        paired=paired,
        decoder=decoder,
        robustness=robustness,
        measurement_cost=measurement_cost,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL STATISTICAL ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSaved descriptive statistics:\n"
        f"{DESCRIPTIVE_PATH}"
    )

    print(
        f"\nSaved paired comparisons:\n"
        f"{PAIRED_PATH}"
    )

    print(
        f"\nSaved decoder comparisons:\n"
        f"{DECODER_PATH}"
    )

    print(
        f"\nSaved robustness analysis:\n"
        f"{ROBUSTNESS_PATH}"
    )

    print(
        f"\nSaved measurement-cost analysis:\n"
        f"{MEASUREMENT_COST_PATH}"
    )

    print(
        f"\nSaved machine-readable summary:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These statistics analyze the frozen 100-condition "
        "main experiment. No quantum measurements were rerun."
    )

    print(
        "Statistical non-significance must not be interpreted "
        "as proof of equivalence."
    )


if __name__ == "__main__":
    run_pipeline()
