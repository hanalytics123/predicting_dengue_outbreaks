"""
08_engineer_features.py

Create a one-week-ahead dengue case forecasting feature dataset for Maynas,
Loreto, Peru.

ACTIVE PROJECT SCOPE
--------------------
Target:
    Weekly dengue case counts (regression / forecasting), not outbreak classes.

Forecast horizon:
    One week ahead.

Forecasting convention:
    For target week t, predictors may use information available up to the end
    of week t-1. Contemporaneous environmental values from target week t are
    therefore not used as predictors.

Weekly convention:
    Sunday-Saturday weeks identified by Sunday week_start_date.

Input:
    data/processed/master/maynas_dengue_master_dataset.csv

Outputs:
    data/processed/modelling/maynas_dengue_features_1w_ahead.csv
    outputs/summary/feature_engineering_summary.txt
    outputs/summary/feature_inventory.csv

Important:
    The master dengue series contains four known missing surveillance weeks.
    Feature engineering is performed on a complete Sunday weekly calendar
    before returning to observed dengue target weeks. This prevents a simple
    row shift from accidentally treating a two-week gap as a one-week lag.

    Environmental values are not reconstructed for missing dengue master
    weeks. Consequently, some observations immediately after surveillance
    gaps may have missing lagged predictors. These rows are retained for
    transparent downstream auditing rather than silently imputed or removed.

    Summary outputs intentionally do not include local filesystem paths.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master"
    / "maynas_dengue_master_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "modelling"
OUTPUT_FILE = OUTPUT_DIR / "maynas_dengue_features_1w_ahead.csv"

SUMMARY_DIR = PROJECT_ROOT / "outputs" / "summary"
SUMMARY_FILE = SUMMARY_DIR / "feature_engineering_summary.txt"
INVENTORY_FILE = SUMMARY_DIR / "feature_inventory.csv"

DATE_COL = "week_start_date"
TARGET_COL = "dengue_cases"

DENGUE_LAGS = [1, 2, 4, 8, 12]
ENVIRONMENTAL_LAGS = [1, 2, 4, 8, 12]

# Representative variables selected to avoid obvious duplicate measures while
# retaining each major environmental domain identified during EDA.
ENVIRONMENTAL_BASE_FEATURES = [
    "precip_sum_mm",
    "temperature_c_mean",
    "relative_humidity_pct_mean",
    "specific_humidity_kgkg_mean",
    "wind_speed_ms_mean",
    "surface_pressure_hpa_mean",
    "ndvi_mean_weekly",
]

# QA fields are retained for auditability but are not intended to be model
# predictors. They are shifted to the previous week to preserve the one-week-
# ahead forecasting convention.
QA_COLUMNS = [
    "precip_week_complete",
    "temp_humidity_week_complete",
    "wind_week_complete",
    "pressure_week_complete",
    "low_valid_pixel_coverage",
    "ndvi_available",
    "ndvi_within_age_limit",
    "ndvi_stale",
]

ROLLING_WINDOWS = [4, 8]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if expected input columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            "The master dataset is missing required column(s): "
            + ", ".join(missing)
        )


def is_sunday(series: pd.Series) -> pd.Series:
    """Return True for Sunday dates (Python weekday: Monday=0, Sunday=6)."""
    return series.dt.weekday.eq(6)


def add_lags(
    df: pd.DataFrame,
    source_col: str,
    lags: list[int],
    feature_group: str,
    inventory: list[dict],
) -> None:
    """Add calendar-week lags to an already reindexed weekly dataframe."""
    for lag in lags:
        new_col = f"{source_col}_lag{lag}"
        df[new_col] = df[source_col].shift(lag)
        inventory.append(
            {
                "feature": new_col,
                "group": feature_group,
                "source_variable": source_col,
                "transformation": f"{lag}-week lag",
                "available_for_target_week": "Yes",
            }
        )


def add_historical_rolling(
    df: pd.DataFrame,
    source_col: str,
    windows: list[int],
    statistic: str,
    feature_group: str,
    inventory: list[dict],
) -> None:
    """
    Add rolling features using only information strictly before target week.

    shift(1) is applied before the rolling calculation so target-week data can
    never enter the feature.
    """
    historical = df[source_col].shift(1)

    for window in windows:
        if statistic == "mean":
            values = historical.rolling(window=window, min_periods=window).mean()
            suffix = "mean"
        elif statistic == "sum":
            values = historical.rolling(window=window, min_periods=window).sum()
            suffix = "sum"
        elif statistic == "std":
            values = historical.rolling(window=window, min_periods=window).std()
            suffix = "std"
        else:
            raise ValueError(f"Unsupported rolling statistic: {statistic}")

        new_col = f"{source_col}_prev{window}w_{suffix}"
        df[new_col] = values

        inventory.append(
            {
                "feature": new_col,
                "group": feature_group,
                "source_variable": source_col,
                "transformation": (
                    f"{statistic} over previous {window} calendar weeks; "
                    "target week excluded"
                ),
                "available_for_target_week": "Yes",
            }
        )


# ---------------------------------------------------------------------------
# Load and validate
# ---------------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

require_columns(df, [DATE_COL, TARGET_COL] + ENVIRONMENTAL_BASE_FEATURES)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise")
df = df.sort_values(DATE_COL).reset_index(drop=True)

if df[DATE_COL].duplicated().any():
    duplicates = int(df[DATE_COL].duplicated().sum())
    raise ValueError(
        f"Master dataset contains {duplicates} duplicate {DATE_COL} value(s)."
    )

if not is_sunday(df[DATE_COL]).all():
    bad_dates = df.loc[~is_sunday(df[DATE_COL]), DATE_COL].dt.strftime("%Y-%m-%d")
    raise ValueError(
        "All week_start_date values must be Sundays. Non-Sunday value(s): "
        + ", ".join(bad_dates.head(10))
    )


# ---------------------------------------------------------------------------
# Reindex to complete weekly calendar
# ---------------------------------------------------------------------------

observed_target_dates = set(df[DATE_COL])

full_calendar = pd.date_range(
    start=df[DATE_COL].min(),
    end=df[DATE_COL].max(),
    freq="W-SUN",
)

weekly = (
    df.set_index(DATE_COL)
    .reindex(full_calendar)
    .rename_axis(DATE_COL)
    .reset_index()
)

weekly["is_observed_dengue_week"] = weekly[DATE_COL].isin(observed_target_dates)

missing_calendar_weeks = weekly.loc[
    ~weekly["is_observed_dengue_week"], DATE_COL
].copy()


# ---------------------------------------------------------------------------
# Forecast timing fields
# ---------------------------------------------------------------------------

weekly["forecast_origin_week_start_date"] = (
    weekly[DATE_COL] - pd.Timedelta(days=7)
)
weekly["forecast_horizon_weeks"] = 1

# Target-week seasonality is known in advance and is therefore legitimate.
day_of_year = weekly[DATE_COL].dt.dayofyear.astype(float)
season_angle = 2.0 * math.pi * (day_of_year - 1.0) / 365.2425

weekly["week_sin"] = np.sin(season_angle)
weekly["week_cos"] = np.cos(season_angle)

weekly["target_calendar_year"] = weekly[DATE_COL].dt.year
weekly["target_calendar_month"] = weekly[DATE_COL].dt.month
weekly["target_iso_year"] = weekly[DATE_COL].dt.isocalendar().year.astype("Int64")
weekly["target_iso_week"] = weekly[DATE_COL].dt.isocalendar().week.astype("Int64")


# ---------------------------------------------------------------------------
# Feature inventory
# ---------------------------------------------------------------------------

feature_inventory: list[dict] = [
    {
        "feature": "week_sin",
        "group": "seasonality",
        "source_variable": DATE_COL,
        "transformation": "cyclical annual sine term for target week",
        "available_for_target_week": "Yes",
    },
    {
        "feature": "week_cos",
        "group": "seasonality",
        "source_variable": DATE_COL,
        "transformation": "cyclical annual cosine term for target week",
        "available_for_target_week": "Yes",
    },
]


# ---------------------------------------------------------------------------
# Dengue history features
# ---------------------------------------------------------------------------

add_lags(
    weekly,
    source_col=TARGET_COL,
    lags=DENGUE_LAGS,
    feature_group="dengue_history",
    inventory=feature_inventory,
)

add_historical_rolling(
    weekly,
    source_col=TARGET_COL,
    windows=ROLLING_WINDOWS,
    statistic="mean",
    feature_group="dengue_history",
    inventory=feature_inventory,
)

add_historical_rolling(
    weekly,
    source_col=TARGET_COL,
    windows=[4],
    statistic="std",
    feature_group="dengue_history",
    inventory=feature_inventory,
)

weekly["dengue_recent_change_1v4"] = (
    weekly[f"{TARGET_COL}_lag1"] - weekly[f"{TARGET_COL}_lag4"]
)
feature_inventory.append(
    {
        "feature": "dengue_recent_change_1v4",
        "group": "dengue_history",
        "source_variable": TARGET_COL,
        "transformation": "1-week lag minus 4-week lag",
        "available_for_target_week": "Yes",
    }
)


# ---------------------------------------------------------------------------
# Environmental features
# ---------------------------------------------------------------------------

for source_col in ENVIRONMENTAL_BASE_FEATURES:
    add_lags(
        weekly,
        source_col=source_col,
        lags=ENVIRONMENTAL_LAGS,
        feature_group="environmental",
        inventory=feature_inventory,
    )

# Compact rolling summaries. These deliberately use previous weeks only.
add_historical_rolling(
    weekly,
    source_col="precip_sum_mm",
    windows=[4, 8],
    statistic="sum",
    feature_group="environmental",
    inventory=feature_inventory,
)

for source_col in [
    "temperature_c_mean",
    "relative_humidity_pct_mean",
    "specific_humidity_kgkg_mean",
    "wind_speed_ms_mean",
    "surface_pressure_hpa_mean",
    "ndvi_mean_weekly",
]:
    add_historical_rolling(
        weekly,
        source_col=source_col,
        windows=[4],
        statistic="mean",
        feature_group="environmental",
        inventory=feature_inventory,
    )


# ---------------------------------------------------------------------------
# Previous-week QA fields
# ---------------------------------------------------------------------------

qa_output_columns = []

for source_col in QA_COLUMNS:
    if source_col not in weekly.columns:
        continue

    new_col = f"qa_prev_{source_col}"
    weekly[new_col] = weekly[source_col].shift(1)
    qa_output_columns.append(new_col)

    feature_inventory.append(
        {
            "feature": new_col,
            "group": "qa_not_model_predictor",
            "source_variable": source_col,
            "transformation": "previous-week QA value",
            "available_for_target_week": "Yes",
        }
    )


# ---------------------------------------------------------------------------
# Return to observed dengue target weeks
# ---------------------------------------------------------------------------

feature_columns = [
    row["feature"]
    for row in feature_inventory
    if row["group"] != "qa_not_model_predictor"
]

output_columns = [
    DATE_COL,
    "forecast_origin_week_start_date",
    "forecast_horizon_weeks",
    "target_calendar_year",
    "target_calendar_month",
    "target_iso_year",
    "target_iso_week",
    TARGET_COL,
    "week_sin",
    "week_cos",
]

# Avoid duplicates while preserving order.
for col in feature_columns:
    if col not in output_columns:
        output_columns.append(col)

for col in qa_output_columns:
    if col not in output_columns:
        output_columns.append(col)

model_df = weekly.loc[weekly["is_observed_dengue_week"], output_columns].copy()

# Flags to support the next audit step. No rows are removed here.
model_df["feature_missing_count"] = model_df[feature_columns].isna().sum(axis=1)
model_df["all_candidate_features_available"] = (
    model_df["feature_missing_count"] == 0
)

# A lighter eligibility flag for a core benchmark feature set.
core_features = [
    "week_sin",
    "week_cos",
    "dengue_cases_lag1",
    "dengue_cases_lag2",
    "dengue_cases_lag4",
    "dengue_cases_prev4w_mean",
]

model_df["core_features_available"] = model_df[core_features].notna().all(axis=1)

if model_df[TARGET_COL].isna().any():
    raise ValueError(
        "Observed dengue target rows unexpectedly contain missing dengue_cases."
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

model_df.to_csv(OUTPUT_FILE, index=False)

inventory_df = pd.DataFrame(feature_inventory)
inventory_df.to_csv(INVENTORY_FILE, index=False)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

missing_feature_counts = (
    model_df[feature_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

rows_with_any_missing = int((model_df["feature_missing_count"] > 0).sum())
rows_all_features = int(model_df["all_candidate_features_available"].sum())
rows_core_features = int(model_df["core_features_available"].sum())

summary_lines = [
    "=" * 78,
    "ONE-WEEK-AHEAD DENGUE FEATURE ENGINEERING SUMMARY",
    "=" * 78,
    "",
    "ACTIVE FORECASTING SCOPE",
    "-" * 78,
    "Outcome: weekly dengue case count",
    "Forecast horizon: 1 week ahead",
    "Target week convention: Sunday-Saturday, identified by Sunday week_start_date",
    (
        "Predictor timing: target week t is predicted using information available "
        "through week t-1"
    ),
    "Target-week environmental measurements are not used as predictors.",
    "",
    "INPUT / OUTPUT STRUCTURE",
    "-" * 78,
    f"Observed dengue target rows: {len(model_df):,}",
    f"Candidate model features: {len(feature_columns):,}",
    f"Retained previous-week QA fields: {len(qa_output_columns):,}",
    f"Output columns: {model_df.shape[1]:,}",
    "",
    "CALENDAR CONTINUITY",
    "-" * 78,
    f"Full Sunday calendar weeks: {len(weekly):,}",
    f"Observed dengue weeks: {int(weekly['is_observed_dengue_week'].sum()):,}",
    f"Missing dengue surveillance weeks: {len(missing_calendar_weeks):,}",
]

if len(missing_calendar_weeks):
    summary_lines.extend(
        [
            "Missing weeks:",
            *[
                f"  {date.strftime('%Y-%m-%d')}"
                for date in missing_calendar_weeks
            ],
        ]
    )

summary_lines.extend(
    [
        "",
        "FEATURE GROUPS",
        "-" * 78,
        f"Dengue lags: {', '.join(map(str, DENGUE_LAGS))} week(s)",
        (
            "Environmental lag candidates: "
            + ", ".join(map(str, ENVIRONMENTAL_LAGS))
            + " week(s)"
        ),
        "Dengue rolling windows: previous 4 and 8 weeks",
        "Environmental rolling summaries: previous weeks only",
        "Seasonality: annual sine/cosine terms based on target week",
        "",
        "ENVIRONMENTAL BASE VARIABLES",
        "-" * 78,
        *[f"  {col}" for col in ENVIRONMENTAL_BASE_FEATURES],
        "",
        "MISSINGNESS AFTER FEATURE ENGINEERING",
        "-" * 78,
        f"Rows with at least one missing candidate feature: {rows_with_any_missing:,}",
        f"Rows with all candidate features available: {rows_all_features:,}",
        f"Rows with core benchmark features available: {rows_core_features:,}",
        "",
        "Candidate features with missing values:",
    ]
)

nonzero_missing = missing_feature_counts[missing_feature_counts > 0]

if len(nonzero_missing) == 0:
    summary_lines.append("  None")
else:
    for feature, count in nonzero_missing.items():
        pct = 100.0 * count / len(model_df)
        summary_lines.append(f"  {feature}: {count:,} ({pct:.2f}%)")

summary_lines.extend(
    [
        "",
        "MODELLING SAFEGUARDS",
        "-" * 78,
        (
            "1. Feature construction was performed on a complete weekly calendar so "
            "lags represent true calendar-week offsets."
        ),
        (
            "2. All historical rolling features are shifted before aggregation, "
            "preventing target-week leakage."
        ),
        (
            "3. Target-week seasonality is retained because the calendar date is known "
            "at forecast time."
        ),
        (
            "4. Missing features caused by the start of the series, MODIS availability, "
            "or surveillance gaps are retained for downstream audit."
        ),
        (
            "5. No imputation, feature selection, scaling, train/test splitting, or "
            "model fitting is performed in this script."
        ),
        "",
        "NEXT STEP",
        "-" * 78,
        (
            "Audit the engineered feature dataset, then define chronological training, "
            "validation and final test periods before modelling."
        ),
        "",
        "=" * 78,
        "FEATURE ENGINEERING COMPLETE",
        "=" * 78,
    ]
)

SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")

print("=" * 78)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 78)
print(f"Rows: {len(model_df):,}")
print(f"Candidate features: {len(feature_columns):,}")
print(f"Rows with all candidate features available: {rows_all_features:,}")
print(f"Rows with core features available: {rows_core_features:,}")
print()
print("Created:")
print(" - data/processed/modelling/maynas_dengue_features_1w_ahead.csv")
print(" - outputs/summary/feature_engineering_summary.txt")
print(" - outputs/summary/feature_inventory.csv")
