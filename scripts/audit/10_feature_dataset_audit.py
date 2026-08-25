"""
10_feature_dataset_audit.py

Independent audit of the one-week-ahead dengue feature dataset.

Scope
-----
Target:
    Weekly dengue case counts.

Forecast horizon:
    One week ahead.

Feature timing:
    For target week t, predictors must only use information available through
    week t-1, except calendar/seasonality features that are known in advance.

Inputs:
    data/processed/master/maynas_dengue_master_dataset.csv
    data/processed/modelling/maynas_dengue_features_1w_ahead.csv

Outputs:
    outputs/audit/feature_dataset_audit_summary.txt
    outputs/audit/feature_dataset_missingness.csv
    outputs/audit/feature_dataset_lag_validation.csv
    outputs/audit/feature_dataset_rolling_validation.csv
    outputs/audit/feature_dataset_redundancy.csv
    outputs/audit/feature_dataset_problem_rows.csv

Repository safety:
    No absolute/local filesystem paths are written into exported outputs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master"
    / "maynas_dengue_master_dataset.csv"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modelling"
    / "maynas_dengue_features_1w_ahead.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "feature_dataset_audit_summary.txt"
MISSINGNESS_FILE = OUTPUT_DIR / "feature_dataset_missingness.csv"
LAG_VALIDATION_FILE = OUTPUT_DIR / "feature_dataset_lag_validation.csv"
ROLLING_VALIDATION_FILE = OUTPUT_DIR / "feature_dataset_rolling_validation.csv"
REDUNDANCY_FILE = OUTPUT_DIR / "feature_dataset_redundancy.csv"
PROBLEM_ROWS_FILE = OUTPUT_DIR / "feature_dataset_problem_rows.csv"

DATE_COL = "week_start_date"
TARGET_COL = "dengue_cases"

KNOWN_DENGUE_GAPS = pd.DatetimeIndex(
    [
        "2000-04-30",
        "2000-06-11",
        "2000-06-25",
        "2000-07-09",
    ]
)

EXPECTED_DENGUE_LAGS = [1, 2, 4, 8, 12]
EXPECTED_ENVIRONMENTAL_LAGS = [1, 2, 4, 8, 12]

ENVIRONMENTAL_BASE_FEATURES = [
    "precip_sum_mm",
    "temperature_c_mean",
    "relative_humidity_pct_mean",
    "specific_humidity_kgkg_mean",
    "wind_speed_ms_mean",
    "surface_pressure_hpa_mean",
    "ndvi_mean_weekly",
]

TOLERANCE = 1e-9
HIGH_CORR_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required column(s): "
            + ", ".join(missing)
        )


def bool_status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def compare_numeric_series(
    actual: pd.Series,
    expected: pd.Series,
    tolerance: float = TOLERANCE,
) -> tuple[int, int, float]:
    """
    Compare two numeric series allowing aligned NaN values.

    Returns:
        compared_rows, mismatches, max_abs_difference
    """
    actual = pd.to_numeric(actual, errors="coerce")
    expected = pd.to_numeric(expected, errors="coerce")

    both_nan = actual.isna() & expected.isna()
    one_nan = actual.isna() ^ expected.isna()

    both_values = actual.notna() & expected.notna()
    diffs = (actual[both_values] - expected[both_values]).abs()

    value_mismatch = diffs > tolerance

    mismatches = int(one_nan.sum() + value_mismatch.sum())
    compared_rows = len(actual)

    max_abs_diff = float(diffs.max()) if len(diffs) else 0.0

    return compared_rows, mismatches, max_abs_diff


def is_boolean_like(series: pd.Series) -> bool:
    vals = set(series.dropna().unique().tolist())
    return vals.issubset({True, False, 0, 1})


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

master = pd.read_csv(MASTER_FILE)
features = pd.read_csv(FEATURE_FILE)

require_columns(master, [DATE_COL, TARGET_COL], "Master dataset")
require_columns(features, [DATE_COL, TARGET_COL], "Feature dataset")

master[DATE_COL] = pd.to_datetime(master[DATE_COL], errors="raise")
features[DATE_COL] = pd.to_datetime(features[DATE_COL], errors="raise")

if "forecast_origin_week_start_date" in features.columns:
    features["forecast_origin_week_start_date"] = pd.to_datetime(
        features["forecast_origin_week_start_date"],
        errors="raise",
    )

master = master.sort_values(DATE_COL).reset_index(drop=True)
features = features.sort_values(DATE_COL).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

structural_checks = []

def add_structural_check(name: str, condition: bool, detail: str) -> None:
    structural_checks.append(
        {
            "check": name,
            "status": bool_status(condition),
            "detail": detail,
        }
    )

add_structural_check(
    "Expected feature rows",
    len(features) == 1248,
    f"Observed rows: {len(features):,}",
)

add_structural_check(
    "Unique target weeks",
    not features[DATE_COL].duplicated().any(),
    f"Duplicate week_start_date values: {int(features[DATE_COL].duplicated().sum())}",
)

add_structural_check(
    "Sunday alignment",
    features[DATE_COL].dt.weekday.eq(6).all(),
    f"Non-Sunday target weeks: {int((~features[DATE_COL].dt.weekday.eq(6)).sum())}",
)

add_structural_check(
    "Chronological order",
    features[DATE_COL].is_monotonic_increasing,
    "Rows sorted ascending by week_start_date",
)

add_structural_check(
    "No missing target",
    features[TARGET_COL].notna().all(),
    f"Missing target values: {int(features[TARGET_COL].isna().sum())}",
)

add_structural_check(
    "No negative target",
    (features[TARGET_COL] >= 0).all(),
    f"Negative target values: {int((features[TARGET_COL] < 0).sum())}",
)

numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
inf_count = int(np.isinf(features[numeric_cols].to_numpy()).sum())

add_structural_check(
    "No infinite numeric values",
    inf_count == 0,
    f"Infinite numeric values: {inf_count}",
)

# Target alignment with master dataset
target_merge = features[[DATE_COL, TARGET_COL]].merge(
    master[[DATE_COL, TARGET_COL]],
    on=DATE_COL,
    how="left",
    suffixes=("_feature", "_master"),
)

target_mismatch = int(
    (
        target_merge[f"{TARGET_COL}_feature"]
        != target_merge[f"{TARGET_COL}_master"]
    ).sum()
)

add_structural_check(
    "Target values align with master dataset",
    target_mismatch == 0,
    f"Target mismatches: {target_mismatch}",
)

# Forecast origin check
if "forecast_origin_week_start_date" in features.columns:
    expected_origin = features[DATE_COL] - pd.Timedelta(days=7)
    origin_mismatches = int(
        (
            features["forecast_origin_week_start_date"]
            != expected_origin
        ).sum()
    )

    add_structural_check(
        "Forecast origin is exactly one week before target",
        origin_mismatches == 0,
        f"Forecast-origin mismatches: {origin_mismatches}",
    )

if "forecast_horizon_weeks" in features.columns:
    horizon_ok = features["forecast_horizon_weeks"].eq(1).all()

    add_structural_check(
        "Forecast horizon fixed at one week",
        horizon_ok,
        (
            "Unique values: "
            + ", ".join(
                map(
                    str,
                    sorted(
                        features["forecast_horizon_weeks"]
                        .dropna()
                        .unique()
                        .tolist()
                    ),
                )
            )
        ),
    )


# ---------------------------------------------------------------------------
# Reconstruct complete weekly calendar for independent lag checks
# ---------------------------------------------------------------------------

calendar = pd.date_range(
    start=master[DATE_COL].min(),
    end=master[DATE_COL].max(),
    freq="W-SUN",
)

master_weekly = (
    master.set_index(DATE_COL)
    .reindex(calendar)
    .rename_axis(DATE_COL)
)

observed_missing_weeks = master_weekly.index[
    master_weekly[TARGET_COL].isna()
]

unexpected_gaps = observed_missing_weeks.difference(KNOWN_DENGUE_GAPS)
missing_known_gaps = KNOWN_DENGUE_GAPS.difference(observed_missing_weeks)

add_structural_check(
    "Known dengue gaps preserved",
    len(unexpected_gaps) == 0 and len(missing_known_gaps) == 0,
    (
        f"Observed missing weeks: {len(observed_missing_weeks)}; "
        f"unexpected gaps: {len(unexpected_gaps)}; "
        f"missing expected gaps: {len(missing_known_gaps)}"
    ),
)


# ---------------------------------------------------------------------------
# Lag validation
# ---------------------------------------------------------------------------

lag_validation_rows = []

def record_lag_validation(
    feature_name: str,
    source_name: str,
    lag_weeks: int,
    group: str,
) -> None:
    if feature_name not in features.columns:
        lag_validation_rows.append(
            {
                "feature": feature_name,
                "source_variable": source_name,
                "lag_weeks": lag_weeks,
                "group": group,
                "status": "MISSING_COLUMN",
                "rows_checked": 0,
                "mismatches": np.nan,
                "max_abs_difference": np.nan,
            }
        )
        return

    expected = master_weekly[source_name].shift(lag_weeks)

    expected_for_targets = expected.reindex(
        pd.DatetimeIndex(features[DATE_COL])
    ).reset_index(drop=True)

    actual = features[feature_name].reset_index(drop=True)

    rows_checked, mismatches, max_abs_diff = compare_numeric_series(
        actual,
        expected_for_targets,
    )

    lag_validation_rows.append(
        {
            "feature": feature_name,
            "source_variable": source_name,
            "lag_weeks": lag_weeks,
            "group": group,
            "status": "PASS" if mismatches == 0 else "FAIL",
            "rows_checked": rows_checked,
            "mismatches": mismatches,
            "max_abs_difference": max_abs_diff,
        }
    )


for lag in EXPECTED_DENGUE_LAGS:
    record_lag_validation(
        feature_name=f"dengue_cases_lag{lag}",
        source_name=TARGET_COL,
        lag_weeks=lag,
        group="dengue_history",
    )

for source in ENVIRONMENTAL_BASE_FEATURES:
    require_columns(master, [source], "Master dataset")

    for lag in EXPECTED_ENVIRONMENTAL_LAGS:
        record_lag_validation(
            feature_name=f"{source}_lag{lag}",
            source_name=source,
            lag_weeks=lag,
            group="environmental",
        )

lag_validation_df = pd.DataFrame(lag_validation_rows)
lag_validation_df.to_csv(LAG_VALIDATION_FILE, index=False)

lag_failures = int(
    lag_validation_df["status"].isin(["FAIL", "MISSING_COLUMN"]).sum()
)

add_structural_check(
    "Independent lag validation",
    lag_failures == 0,
    f"Lag validation failures: {lag_failures}",
)


# ---------------------------------------------------------------------------
# Rolling feature validation
# ---------------------------------------------------------------------------

rolling_validation_rows = []

def validate_rolling(
    feature_name: str,
    source_name: str,
    window: int,
    statistic: str,
) -> None:
    if feature_name not in features.columns:
        rolling_validation_rows.append(
            {
                "feature": feature_name,
                "source_variable": source_name,
                "window_weeks": window,
                "statistic": statistic,
                "status": "MISSING_COLUMN",
                "rows_checked": 0,
                "mismatches": np.nan,
                "max_abs_difference": np.nan,
            }
        )
        return

    historical = master_weekly[source_name].shift(1)

    if statistic == "mean":
        expected = historical.rolling(
            window=window,
            min_periods=window,
        ).mean()
    elif statistic == "sum":
        expected = historical.rolling(
            window=window,
            min_periods=window,
        ).sum()
    elif statistic == "std":
        expected = historical.rolling(
            window=window,
            min_periods=window,
        ).std()
    else:
        raise ValueError(f"Unsupported rolling statistic: {statistic}")

    expected_for_targets = expected.reindex(
        pd.DatetimeIndex(features[DATE_COL])
    ).reset_index(drop=True)

    actual = features[feature_name].reset_index(drop=True)

    rows_checked, mismatches, max_abs_diff = compare_numeric_series(
        actual,
        expected_for_targets,
    )

    rolling_validation_rows.append(
        {
            "feature": feature_name,
            "source_variable": source_name,
            "window_weeks": window,
            "statistic": statistic,
            "status": "PASS" if mismatches == 0 else "FAIL",
            "rows_checked": rows_checked,
            "mismatches": mismatches,
            "max_abs_difference": max_abs_diff,
        }
    )


validate_rolling(
    "dengue_cases_prev4w_mean",
    TARGET_COL,
    4,
    "mean",
)

validate_rolling(
    "dengue_cases_prev8w_mean",
    TARGET_COL,
    8,
    "mean",
)

validate_rolling(
    "dengue_cases_prev4w_std",
    TARGET_COL,
    4,
    "std",
)

validate_rolling(
    "precip_sum_mm_prev4w_sum",
    "precip_sum_mm",
    4,
    "sum",
)

validate_rolling(
    "precip_sum_mm_prev8w_sum",
    "precip_sum_mm",
    8,
    "sum",
)

for source in [
    "temperature_c_mean",
    "relative_humidity_pct_mean",
    "specific_humidity_kgkg_mean",
    "wind_speed_ms_mean",
    "surface_pressure_hpa_mean",
    "ndvi_mean_weekly",
]:
    validate_rolling(
        f"{source}_prev4w_mean",
        source,
        4,
        "mean",
    )

rolling_validation_df = pd.DataFrame(rolling_validation_rows)
rolling_validation_df.to_csv(ROLLING_VALIDATION_FILE, index=False)

rolling_failures = int(
    rolling_validation_df["status"].isin(["FAIL", "MISSING_COLUMN"]).sum()
)

add_structural_check(
    "Independent rolling-feature validation",
    rolling_failures == 0,
    f"Rolling validation failures: {rolling_failures}",
)


# ---------------------------------------------------------------------------
# Derived feature checks
# ---------------------------------------------------------------------------

derived_checks = []

# dengue_recent_change_1v4
if "dengue_recent_change_1v4" in features.columns:
    expected_change = (
        features["dengue_cases_lag1"]
        - features["dengue_cases_lag4"]
    )

    _, mismatch_count, max_abs_diff = compare_numeric_series(
        features["dengue_recent_change_1v4"],
        expected_change,
    )

    derived_checks.append(
        {
            "check": "dengue_recent_change_1v4",
            "status": "PASS" if mismatch_count == 0 else "FAIL",
            "mismatches": mismatch_count,
            "max_abs_difference": max_abs_diff,
        }
    )

# seasonality
if {"week_sin", "week_cos"}.issubset(features.columns):
    day_of_year = features[DATE_COL].dt.dayofyear.astype(float)
    angle = 2.0 * math.pi * (day_of_year - 1.0) / 365.2425

    expected_sin = np.sin(angle)
    expected_cos = np.cos(angle)

    _, sin_mismatch, sin_max = compare_numeric_series(
        features["week_sin"],
        pd.Series(expected_sin),
    )

    _, cos_mismatch, cos_max = compare_numeric_series(
        features["week_cos"],
        pd.Series(expected_cos),
    )

    derived_checks.extend(
        [
            {
                "check": "week_sin",
                "status": "PASS" if sin_mismatch == 0 else "FAIL",
                "mismatches": sin_mismatch,
                "max_abs_difference": sin_max,
            },
            {
                "check": "week_cos",
                "status": "PASS" if cos_mismatch == 0 else "FAIL",
                "mismatches": cos_mismatch,
                "max_abs_difference": cos_max,
            },
        ]
    )

derived_failures = sum(
    row["status"] == "FAIL"
    for row in derived_checks
)

add_structural_check(
    "Derived feature validation",
    derived_failures == 0,
    f"Derived-feature failures: {derived_failures}",
)


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------

# Exact raw environmental base columns should not appear in final feature dataset.
unexpected_raw_predictors = [
    col
    for col in ENVIRONMENTAL_BASE_FEATURES
    if col in features.columns
]

add_structural_check(
    "No target-week raw environmental predictors",
    len(unexpected_raw_predictors) == 0,
    (
        "Unexpected raw columns: "
        + (
            ", ".join(unexpected_raw_predictors)
            if unexpected_raw_predictors
            else "None"
        )
    ),
)

# Master identifiers should not silently become predictor columns.
forbidden_identifier_columns = [
    "country",
    "adm1_name",
    "adm2_name",
    "case_definition",
]

present_forbidden_identifiers = [
    col
    for col in forbidden_identifier_columns
    if col in features.columns
]

add_structural_check(
    "No static master identifiers in feature table",
    len(present_forbidden_identifiers) == 0,
    (
        "Present identifiers: "
        + (
            ", ".join(present_forbidden_identifiers)
            if present_forbidden_identifiers
            else "None"
        )
    ),
)


# ---------------------------------------------------------------------------
# Missingness audit
# ---------------------------------------------------------------------------

missingness_df = (
    features.isna()
    .sum()
    .rename("missing_count")
    .to_frame()
)

missingness_df["missing_pct"] = (
    missingness_df["missing_count"] / len(features) * 100.0
)

missingness_df = (
    missingness_df
    .sort_values(
        ["missing_count"],
        ascending=False,
    )
    .reset_index()
    .rename(columns={"index": "column"})
)

missingness_df.to_csv(MISSINGNESS_FILE, index=False)

feature_metadata_cols = {
    DATE_COL,
    "forecast_origin_week_start_date",
    "forecast_horizon_weeks",
    "target_calendar_year",
    "target_calendar_month",
    "target_iso_year",
    "target_iso_week",
    TARGET_COL,
    "feature_missing_count",
    "all_candidate_features_available",
    "core_features_available",
}

qa_cols = [
    col
    for col in features.columns
    if col.startswith("qa_prev_")
]

candidate_feature_cols = [
    col
    for col in features.columns
    if col not in feature_metadata_cols
    and col not in qa_cols
]

recomputed_missing_count = features[candidate_feature_cols].isna().sum(axis=1)

if "feature_missing_count" in features.columns:
    feature_missing_mismatch = int(
        (
            pd.to_numeric(
                features["feature_missing_count"],
                errors="coerce",
            )
            != recomputed_missing_count
        ).sum()
    )

    add_structural_check(
        "feature_missing_count is accurate",
        feature_missing_mismatch == 0,
        f"Rows with incorrect feature_missing_count: {feature_missing_mismatch}",
    )

if "all_candidate_features_available" in features.columns:
    expected_flag = recomputed_missing_count.eq(0)

    actual_flag = features["all_candidate_features_available"]
    if actual_flag.dtype != bool:
        actual_flag = actual_flag.astype(str).str.lower().map(
            {"true": True, "false": False}
        )

    availability_mismatch = int(
        (actual_flag != expected_flag).sum()
    )

    add_structural_check(
        "all_candidate_features_available is accurate",
        availability_mismatch == 0,
        f"Flag mismatches: {availability_mismatch}",
    )


# ---------------------------------------------------------------------------
# Problem-row export
# ---------------------------------------------------------------------------

problem_mask = (
    recomputed_missing_count.gt(0)
)

problem_rows = features.loc[
    problem_mask,
    [
        DATE_COL,
        TARGET_COL,
        "feature_missing_count",
        "all_candidate_features_available",
        "core_features_available",
    ]
].copy()

problem_rows["missing_features"] = features.loc[
    problem_mask,
    candidate_feature_cols,
].apply(
    lambda row: "; ".join(
        row.index[row.isna()].tolist()
    ),
    axis=1,
)

problem_rows.to_csv(PROBLEM_ROWS_FILE, index=False)


# ---------------------------------------------------------------------------
# Constant / near-constant / redundancy checks
# ---------------------------------------------------------------------------

analysis_cols = [
    col
    for col in candidate_feature_cols
    if pd.api.types.is_numeric_dtype(features[col])
]

constant_features = []
near_constant_features = []

for col in analysis_cols:
    non_missing = features[col].dropna()

    if len(non_missing) == 0:
        continue

    nunique = non_missing.nunique()

    if nunique <= 1:
        constant_features.append(col)
        continue

    proportions = non_missing.value_counts(normalize=True)
    top_share = float(proportions.iloc[0])

    if top_share >= 0.995:
        near_constant_features.append(
            {
                "feature": col,
                "dominant_value_share": top_share,
                "unique_values": nunique,
            }
        )

corr_matrix = features[analysis_cols].corr()

redundancy_rows = []

for i, col_a in enumerate(analysis_cols):
    for col_b in analysis_cols[i + 1:]:
        corr = corr_matrix.loc[col_a, col_b]

        if pd.isna(corr):
            continue

        if abs(corr) >= HIGH_CORR_THRESHOLD:
            redundancy_rows.append(
                {
                    "feature_a": col_a,
                    "feature_b": col_b,
                    "pearson_r": corr,
                    "abs_pearson_r": abs(corr),
                }
            )

redundancy_df = pd.DataFrame(redundancy_rows)

if not redundancy_df.empty:
    redundancy_df = redundancy_df.sort_values(
        "abs_pearson_r",
        ascending=False,
    )

redundancy_df.to_csv(REDUNDANCY_FILE, index=False)


# ---------------------------------------------------------------------------
# QA checks
# ---------------------------------------------------------------------------

qa_summary_rows = []

for col in qa_cols:
    counts = features[col].value_counts(dropna=False)

    for value, count in counts.items():
        qa_summary_rows.append(
            {
                "qa_column": col,
                "value": value,
                "count": int(count),
            }
        )

qa_summary_df = pd.DataFrame(qa_summary_rows)


# ---------------------------------------------------------------------------
# Overall status
# ---------------------------------------------------------------------------

structural_df = pd.DataFrame(structural_checks)
overall_failures = structural_df["status"].eq("FAIL").sum()

overall_status = "PASS" if overall_failures == 0 else "FAIL"


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

summary = [
    "=" * 78,
    "ONE-WEEK-AHEAD FEATURE DATASET AUDIT",
    "=" * 78,
    "",
    "OVERALL RESULT",
    "-" * 78,
    overall_status,
    "",
    "DATASET STRUCTURE",
    "-" * 78,
    f"Rows: {len(features):,}",
    f"Columns: {features.shape[1]:,}",
    f"Candidate feature columns audited: {len(candidate_feature_cols):,}",
    f"First target week: {features[DATE_COL].min().date()}",
    f"Last target week: {features[DATE_COL].max().date()}",
    "",
    "STRUCTURAL / LEAKAGE CHECKS",
    "-" * 78,
]

for row in structural_checks:
    summary.append(
        f"{row['status']}: {row['check']} — {row['detail']}"
    )

summary.extend(
    [
        "",
        "MISSINGNESS",
        "-" * 78,
        f"Rows with at least one missing candidate feature: {int(problem_mask.sum()):,}",
        f"Rows with all candidate features available: {int((~problem_mask).sum()):,}",
        "",
        "Columns with missing values:",
    ]
)

missing_nonzero = missingness_df[
    missingness_df["missing_count"] > 0
]

if missing_nonzero.empty:
    summary.append("  None")
else:
    for _, row in missing_nonzero.iterrows():
        summary.append(
            f"  {row['column']}: "
            f"{int(row['missing_count']):,} "
            f"({row['missing_pct']:.2f}%)"
        )

summary.extend(
    [
        "",
        "LAG VALIDATION",
        "-" * 78,
        f"Lag features checked: {len(lag_validation_df):,}",
        f"Lag validation failures: {lag_failures:,}",
        "",
        "ROLLING VALIDATION",
        "-" * 78,
        f"Rolling features checked: {len(rolling_validation_df):,}",
        f"Rolling validation failures: {rolling_failures:,}",
        "",
        "CONSTANT / NEAR-CONSTANT FEATURES",
        "-" * 78,
        f"Constant candidate features: {len(constant_features):,}",
    ]
)

if constant_features:
    summary.extend(
        [f"  {col}" for col in constant_features]
    )
else:
    summary.append("  None")

summary.append(
    f"Near-constant candidate features (>=99.5% dominant value): "
    f"{len(near_constant_features):,}"
)

if near_constant_features:
    for row in near_constant_features:
        summary.append(
            f"  {row['feature']}: "
            f"{row['dominant_value_share']:.4f} dominant share"
        )
else:
    summary.append("  None")

summary.extend(
    [
        "",
        "HIGH CORRELATION / REDUNDANCY",
        "-" * 78,
        (
            f"Feature pairs with |Pearson r| >= "
            f"{HIGH_CORR_THRESHOLD:.2f}: {len(redundancy_df):,}"
        ),
    ]
)

if redundancy_df.empty:
    summary.append("  None")
else:
    for _, row in redundancy_df.head(30).iterrows():
        summary.append(
            f"  {row['feature_a']} <-> {row['feature_b']}: "
            f"r = {row['pearson_r']:.4f}"
        )

if len(redundancy_df) > 30:
    summary.append(
        f"  ... {len(redundancy_df) - 30:,} additional pair(s) "
        "written to feature_dataset_redundancy.csv"
    )

summary.extend(
    [
        "",
        "QA FIELDS",
        "-" * 78,
    ]
)

if qa_summary_df.empty:
    summary.append("No QA fields found.")
else:
    for col in qa_cols:
        summary.append(f"{col}:")
        subset = qa_summary_df[
            qa_summary_df["qa_column"] == col
        ]
        for _, row in subset.iterrows():
            summary.append(
                f"  {row['value']}: {int(row['count']):,}"
            )

summary.extend(
    [
        "",
        "AUDIT OUTPUTS",
        "-" * 78,
        "outputs/audit/feature_dataset_audit_summary.txt",
        "outputs/audit/feature_dataset_missingness.csv",
        "outputs/audit/feature_dataset_lag_validation.csv",
        "outputs/audit/feature_dataset_rolling_validation.csv",
        "outputs/audit/feature_dataset_redundancy.csv",
        "outputs/audit/feature_dataset_problem_rows.csv",
        "",
        "INTERPRETATION",
        "-" * 78,
        (
            "A PASS indicates that the engineered dataset reproduces the expected "
            "calendar-week lags and rolling calculations, aligns targets with the "
            "master dataset, and contains no detected target-week environmental "
            "leakage."
        ),
        (
            "Missing engineered values are not automatically treated as errors. "
            "They may arise from the start of the time series, the known dengue "
            "surveillance gaps, or delayed MODIS availability and should be "
            "considered explicitly when defining model-specific samples."
        ),
        "",
        "=" * 78,
        "AUDIT COMPLETE",
        "=" * 78,
    ]
)

SUMMARY_FILE.write_text(
    "\n".join(summary),
    encoding="utf-8",
)

print("=" * 78)
print("FEATURE DATASET AUDIT")
print("=" * 78)
print(f"Overall result: {overall_status}")
print(f"Rows: {len(features):,}")
print(f"Candidate features audited: {len(candidate_feature_cols):,}")
print(f"Lag validation failures: {lag_failures}")
print(f"Rolling validation failures: {rolling_failures}")
print(f"Rows with missing candidate features: {int(problem_mask.sum()):,}")
print(f"High-correlation feature pairs: {len(redundancy_df):,}")
print()
print("Created:")
print(" - outputs/audit/feature_dataset_audit_summary.txt")
print(" - outputs/audit/feature_dataset_missingness.csv")
print(" - outputs/audit/feature_dataset_lag_validation.csv")
print(" - outputs/audit/feature_dataset_rolling_validation.csv")
print(" - outputs/audit/feature_dataset_redundancy.csv")
print(" - outputs/audit/feature_dataset_problem_rows.csv")
