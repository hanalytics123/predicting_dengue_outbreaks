"""
11_investigate_modelling_splits.py

Compare candidate chronological train / validation / test splits for the
one-week-ahead Maynas dengue case forecasting project.

This script does NOT train models or select a winning split automatically.
It provides descriptive evidence so the final temporal evaluation design can
be chosen before modelling.

Input:
    data/processed/modelling/maynas_dengue_features_1w_ahead.csv

Outputs:
    outputs/audit/modelling_split_investigation_summary.txt
    outputs/audit/modelling_split_period_statistics.csv
    outputs/audit/modelling_split_year_statistics.csv
    outputs/audit/modelling_split_month_statistics.csv
    outputs/audit/modelling_split_high_incidence_weeks.csv

Repository safety:
    Exported outputs contain repository-relative references only and never
    write the resolved local/workplace filesystem path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modelling"
    / "maynas_dengue_features_1w_ahead.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "modelling_split_investigation_summary.txt"
PERIOD_STATS_FILE = OUTPUT_DIR / "modelling_split_period_statistics.csv"
YEAR_STATS_FILE = OUTPUT_DIR / "modelling_split_year_statistics.csv"
MONTH_STATS_FILE = OUTPUT_DIR / "modelling_split_month_statistics.csv"
HIGH_WEEKS_FILE = OUTPUT_DIR / "modelling_split_high_incidence_weeks.csv"

DATE_COL = "week_start_date"
TARGET_COL = "dengue_cases"


# ---------------------------------------------------------------------------
# Candidate split definitions
#
# All boundaries are chronological and non-overlapping.
# Test is deliberately held at 2021-2023 in all three options so candidate
# validation designs can be compared without repeatedly redefining the final
# future evaluation period.
# ---------------------------------------------------------------------------

SPLITS = {
    "A_2000_2017__2018_2020__2021_2023": {
        "train": ("2000-01-01", "2017-12-31"),
        "validation": ("2018-01-01", "2020-12-31"),
        "test": ("2021-01-01", "2023-12-31"),
    },
    "B_2000_2016__2017_2020__2021_2023": {
        "train": ("2000-01-01", "2016-12-31"),
        "validation": ("2017-01-01", "2020-12-31"),
        "test": ("2021-01-01", "2023-12-31"),
    },
    "C_2000_2018__2019_2020__2021_2023": {
        "train": ("2000-01-01", "2018-12-31"),
        "validation": ("2019-01-01", "2020-12-31"),
        "test": ("2021-01-01", "2023-12-31"),
    },
}

# High-incidence thresholds are derived from the full observed target only for
# descriptive split investigation. They are NOT model features and are NOT
# outbreak labels.
PERCENTILE_LEVELS = [0.90, 0.95, 0.99]


# ---------------------------------------------------------------------------
# Load and validate
# ---------------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Feature dataset not found at the expected repository-relative location."
    )

df = pd.read_csv(INPUT_FILE)

required = [
    DATE_COL,
    TARGET_COL,
    "all_candidate_features_available",
    "core_features_available",
]

missing_required = [
    col for col in required
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        "Feature dataset is missing required column(s): "
        + ", ".join(missing_required)
    )

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    errors="raise",
)

df = (
    df
    .sort_values(DATE_COL)
    .reset_index(drop=True)
)

if df[DATE_COL].duplicated().any():
    raise ValueError(
        "Feature dataset contains duplicate week_start_date values."
    )

if not df[DATE_COL].dt.weekday.eq(6).all():
    raise ValueError(
        "Feature dataset contains non-Sunday week_start_date values."
    )

if df[TARGET_COL].isna().any():
    raise ValueError(
        "Feature dataset contains missing dengue target values."
    )


# ---------------------------------------------------------------------------
# Descriptive high-incidence thresholds
# ---------------------------------------------------------------------------

thresholds = {
    q: float(df[TARGET_COL].quantile(q))
    for q in PERCENTILE_LEVELS
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_bool(series: pd.Series) -> pd.Series:
    """Normalise booleans that may have been read from CSV as strings."""
    if pd.api.types.is_bool_dtype(series):
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )


df["all_candidate_features_available"] = parse_bool(
    df["all_candidate_features_available"]
)

df["core_features_available"] = parse_bool(
    df["core_features_available"]
)


def period_mask(
    data: pd.DataFrame,
    start: str,
    end: str,
) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    return (
        data[DATE_COL].ge(start_ts)
        & data[DATE_COL].le(end_ts)
    )


def summarise_period(
    subset: pd.DataFrame,
    split_name: str,
    period_name: str,
    start: str,
    end: str,
) -> dict:
    if subset.empty:
        raise ValueError(
            f"{split_name} / {period_name} contains no observations."
        )

    target = subset[TARGET_COL]

    row = {
        "split": split_name,
        "period": period_name,
        "defined_start": start,
        "defined_end": end,
        "first_observed_week": subset[DATE_COL].min().date(),
        "last_observed_week": subset[DATE_COL].max().date(),
        "observations": len(subset),
        "core_feature_eligible_rows": int(
            subset["core_features_available"].fillna(False).sum()
        ),
        "all_candidate_feature_eligible_rows": int(
            subset["all_candidate_features_available"].fillna(False).sum()
        ),
        "total_cases": float(target.sum()),
        "mean_cases": float(target.mean()),
        "median_cases": float(target.median()),
        "std_cases": float(target.std()),
        "min_cases": float(target.min()),
        "max_cases": float(target.max()),
        "p90_cases": float(target.quantile(0.90)),
        "p95_cases": float(target.quantile(0.95)),
        "p99_cases": float(target.quantile(0.99)),
    }

    for q, threshold in thresholds.items():
        label = int(q * 100)
        row[f"weeks_ge_fullseries_p{label}"] = int(
            (target >= threshold).sum()
        )
        row[f"share_ge_fullseries_p{label}"] = float(
            (target >= threshold).mean()
        )

    return row


# ---------------------------------------------------------------------------
# Candidate split statistics
# ---------------------------------------------------------------------------

period_rows = []
year_rows = []
month_rows = []
high_week_rows = []

for split_name, periods in SPLITS.items():

    # Confirm split ordering.
    train_end = pd.Timestamp(periods["train"][1])
    val_start = pd.Timestamp(periods["validation"][0])
    val_end = pd.Timestamp(periods["validation"][1])
    test_start = pd.Timestamp(periods["test"][0])

    if not (train_end < val_start <= val_end < test_start):
        raise ValueError(
            f"Split {split_name} is not strictly chronological."
        )

    for period_name, (start, end) in periods.items():

        subset = df.loc[
            period_mask(df, start, end)
        ].copy()

        period_rows.append(
            summarise_period(
                subset,
                split_name,
                period_name,
                start,
                end,
            )
        )

        # Year-level statistics
        yearly = (
            subset
            .assign(
                calendar_year=subset[DATE_COL].dt.year
            )
            .groupby(
                "calendar_year",
                as_index=False,
            )
            .agg(
                observations=(TARGET_COL, "size"),
                total_cases=(TARGET_COL, "sum"),
                mean_cases=(TARGET_COL, "mean"),
                median_cases=(TARGET_COL, "median"),
                max_cases=(TARGET_COL, "max"),
            )
        )

        yearly["split"] = split_name
        yearly["period"] = period_name

        year_rows.extend(
            yearly[
                [
                    "split",
                    "period",
                    "calendar_year",
                    "observations",
                    "total_cases",
                    "mean_cases",
                    "median_cases",
                    "max_cases",
                ]
            ].to_dict("records")
        )

        # Month-of-year statistics
        monthly = (
            subset
            .assign(
                calendar_month=subset[DATE_COL].dt.month
            )
            .groupby(
                "calendar_month",
                as_index=False,
            )
            .agg(
                observations=(TARGET_COL, "size"),
                total_cases=(TARGET_COL, "sum"),
                mean_cases=(TARGET_COL, "mean"),
                median_cases=(TARGET_COL, "median"),
                max_cases=(TARGET_COL, "max"),
            )
        )

        monthly["split"] = split_name
        monthly["period"] = period_name

        month_rows.extend(
            monthly[
                [
                    "split",
                    "period",
                    "calendar_month",
                    "observations",
                    "total_cases",
                    "mean_cases",
                    "median_cases",
                    "max_cases",
                ]
            ].to_dict("records")
        )

        # High-incidence weeks based on full-series 95th percentile.
        high_subset = subset.loc[
            subset[TARGET_COL] >= thresholds[0.95],
            [DATE_COL, TARGET_COL]
        ].copy()

        if not high_subset.empty:
            high_subset["split"] = split_name
            high_subset["period"] = period_name
            high_subset["fullseries_threshold"] = thresholds[0.95]

            high_week_rows.extend(
                high_subset[
                    [
                        "split",
                        "period",
                        DATE_COL,
                        TARGET_COL,
                        "fullseries_threshold",
                    ]
                ].to_dict("records")
            )


period_stats = pd.DataFrame(period_rows)
year_stats = pd.DataFrame(year_rows)
month_stats = pd.DataFrame(month_rows)
high_weeks = pd.DataFrame(high_week_rows)

period_stats.to_csv(
    PERIOD_STATS_FILE,
    index=False,
)

year_stats.to_csv(
    YEAR_STATS_FILE,
    index=False,
)

month_stats.to_csv(
    MONTH_STATS_FILE,
    index=False,
)

high_weeks.to_csv(
    HIGH_WEEKS_FILE,
    index=False,
)


# ---------------------------------------------------------------------------
# Additional whole-series context
# ---------------------------------------------------------------------------

whole_series_years = (
    df
    .assign(
        calendar_year=df[DATE_COL].dt.year
    )
    .groupby(
        "calendar_year",
        as_index=False,
    )
    .agg(
        observations=(TARGET_COL, "size"),
        total_cases=(TARGET_COL, "sum"),
        mean_cases=(TARGET_COL, "mean"),
        median_cases=(TARGET_COL, "median"),
        max_cases=(TARGET_COL, "max"),
    )
    .sort_values(
        "total_cases",
        ascending=False,
    )
)

top_years = whole_series_years.head(10)

top_weeks = (
    df[
        [
            DATE_COL,
            TARGET_COL,
        ]
    ]
    .sort_values(
        TARGET_COL,
        ascending=False,
    )
    .head(15)
)


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

summary = [
    "=" * 78,
    "MODELLING SPLIT INVESTIGATION",
    "=" * 78,
    "",
    "PURPOSE",
    "-" * 78,
    (
        "Compare candidate chronological train / validation / test designs "
        "before fitting any forecasting model."
    ),
    (
        "This is descriptive analysis only. High-incidence thresholds are used "
        "to compare target distributions across periods and do not redefine the "
        "project as an outbreak-classification task."
    ),
    "",
    "FORECASTING SCOPE",
    "-" * 78,
    "Outcome: weekly dengue case count",
    "Forecast horizon: one week ahead",
    "Evaluation design: chronological future holdout",
    "",
    "FULL-SERIES TARGET CONTEXT",
    "-" * 78,
    f"Observed weeks: {len(df):,}",
    f"First week: {df[DATE_COL].min().date()}",
    f"Last week: {df[DATE_COL].max().date()}",
    f"Mean weekly cases: {df[TARGET_COL].mean():.2f}",
    f"Median weekly cases: {df[TARGET_COL].median():.2f}",
    f"Maximum weekly cases: {df[TARGET_COL].max():.0f}",
    f"90th percentile: {thresholds[0.90]:.2f}",
    f"95th percentile: {thresholds[0.95]:.2f}",
    f"99th percentile: {thresholds[0.99]:.2f}",
    "",
    "CANDIDATE SPLITS",
    "-" * 78,
]

for split_name, periods in SPLITS.items():
    summary.extend(
        [
            split_name,
            f"  Train:      {periods['train'][0]} to {periods['train'][1]}",
            (
                f"  Validation: {periods['validation'][0]} "
                f"to {periods['validation'][1]}"
            ),
            f"  Test:       {periods['test'][0]} to {periods['test'][1]}",
            "",
        ]
    )

summary.extend(
    [
        "PERIOD STATISTICS",
        "-" * 78,
    ]
)

for split_name in SPLITS:
    summary.append("")
    summary.append(split_name)

    subset = period_stats.loc[
        period_stats["split"] == split_name
    ]

    for period_name in [
        "train",
        "validation",
        "test",
    ]:
        row = subset.loc[
            subset["period"] == period_name
        ].iloc[0]

        summary.extend(
            [
                f"  {period_name.upper()}",
                (
                    f"    Observed weeks: {int(row['observations']):,} "
                    f"(core eligible: {int(row['core_feature_eligible_rows']):,}; "
                    f"all-feature eligible: "
                    f"{int(row['all_candidate_feature_eligible_rows']):,})"
                ),
                f"    Total cases: {row['total_cases']:,.0f}",
                (
                    f"    Mean / median: "
                    f"{row['mean_cases']:.2f} / {row['median_cases']:.2f}"
                ),
                f"    Standard deviation: {row['std_cases']:.2f}",
                f"    Maximum: {row['max_cases']:.0f}",
                (
                    f"    Period p90 / p95 / p99: "
                    f"{row['p90_cases']:.2f} / "
                    f"{row['p95_cases']:.2f} / "
                    f"{row['p99_cases']:.2f}"
                ),
                (
                    f"    Weeks >= full-series p95 "
                    f"({thresholds[0.95]:.2f} cases): "
                    f"{int(row['weeks_ge_fullseries_p95']):,} "
                    f"({row['share_ge_fullseries_p95'] * 100:.2f}%)"
                ),
            ]
        )

summary.extend(
    [
        "",
        "HIGHEST-BURDEN YEARS IN THE FULL SERIES",
        "-" * 78,
    ]
)

for _, row in top_years.iterrows():
    summary.append(
        f"{int(row['calendar_year'])}: "
        f"total={row['total_cases']:,.0f}, "
        f"mean={row['mean_cases']:.2f}, "
        f"median={row['median_cases']:.2f}, "
        f"max={row['max_cases']:.0f}"
    )

summary.extend(
    [
        "",
        "TOP 15 INDIVIDUAL WEEKS",
        "-" * 78,
    ]
)

for _, row in top_weeks.iterrows():
    summary.append(
        f"{row[DATE_COL].date()}: {row[TARGET_COL]:,.0f} cases"
    )

summary.extend(
    [
        "",
        "INTERPRETATION GUIDANCE",
        "-" * 78,
        (
            "1. Prefer a split that leaves enough historical observations for "
            "training while giving validation and test periods meaningful "
            "variation in dengue burden."
        ),
        (
            "2. The final test period should remain untouched during model "
            "selection and hyperparameter tuning."
        ),
        (
            "3. Differences in MAE/RMSE can be strongly affected by epidemic "
            "peaks because the target is highly right-skewed."
        ),
        (
            "4. Feature eligibility should be considered separately from target "
            "availability. Models using fewer features may legitimately retain "
            "more observations."
        ),
        (
            "5. This script does not choose a split automatically. The final "
            "choice should be documented as a methodological decision before "
            "model fitting."
        ),
        "",
        "OUTPUTS",
        "-" * 78,
        "outputs/audit/modelling_split_investigation_summary.txt",
        "outputs/audit/modelling_split_period_statistics.csv",
        "outputs/audit/modelling_split_year_statistics.csv",
        "outputs/audit/modelling_split_month_statistics.csv",
        "outputs/audit/modelling_split_high_incidence_weeks.csv",
        "",
        "=" * 78,
        "SPLIT INVESTIGATION COMPLETE",
        "=" * 78,
    ]
)

SUMMARY_FILE.write_text(
    "\n".join(summary),
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print("=" * 78)
print("MODELLING SPLIT INVESTIGATION COMPLETE")
print("=" * 78)
print(f"Rows analysed: {len(df):,}")
print(
    "Full-series p95 threshold used descriptively: "
    f"{thresholds[0.95]:.2f} cases"
)
print()
print("Candidate split period summaries:")

for split_name in SPLITS:
    print()
    print(split_name)

    subset = period_stats.loc[
        period_stats["split"] == split_name
    ]

    for period_name in [
        "train",
        "validation",
        "test",
    ]:
        row = subset.loc[
            subset["period"] == period_name
        ].iloc[0]

        print(
            f"  {period_name:<10} "
            f"n={int(row['observations']):>4} | "
            f"mean={row['mean_cases']:>7.2f} | "
            f"median={row['median_cases']:>6.2f} | "
            f"max={row['max_cases']:>5.0f} | "
            f">=p95={int(row['weeks_ge_fullseries_p95']):>3}"
        )

print()
print("Created:")
print(" - outputs/audit/modelling_split_investigation_summary.txt")
print(" - outputs/audit/modelling_split_period_statistics.csv")
print(" - outputs/audit/modelling_split_year_statistics.csv")
print(" - outputs/audit/modelling_split_month_statistics.csv")
print(" - outputs/audit/modelling_split_high_incidence_weeks.csv")
