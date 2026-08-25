from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ndvi"
    / "maynas_mod13q1_ndvi_2000_2023.csv"
)

DENGUE_WEEKLY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dengue"
    / "maynas_dengue_weekly.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ndvi"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "audit"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NATIVE_OUTPUT = (
    PROCESSED_DIR
    / "maynas_mod13q1_ndvi_native.csv"
)

WEEKLY_OUTPUT = (
    PROCESSED_DIR
    / "maynas_mod13q1_ndvi_weekly.csv"
)

SUMMARY_OUTPUT = (
    AUDIT_DIR
    / "ndvi_processing_summary.txt"
)


# ============================================================
# PROCESSING SETTINGS
# ============================================================

# Weekly periods use Sunday as the week start.
WEEK_FREQUENCY = "W-SAT"

# Coverage established during the audit.
EXPECTED_START_DATE = pd.Timestamp("2000-02-18")
EXPECTED_END_DATE = pd.Timestamp("2023-12-19")

# Maximum acceptable age of the most recent NDVI observation
# when carrying values forward onto the weekly timeline.
#
# MOD13Q1 is nominally a 16-day composite product. A 32-day
# threshold allows one missed composite interval before a weekly
# value is considered stale.
MAX_NDVI_AGE_DAYS = 32


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 78)
print("PREPARE MAYNAS MOD13Q1 NDVI DATA")
print("=" * 78)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"NDVI input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

if df["date"].isna().any():
    raise ValueError(
        "One or more NDVI dates could not be parsed."
    )

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

print(f"Input: {INPUT_FILE}")
print(f"Rows loaded: {len(df):,}")
print(f"First date: {df['date'].min()}")
print(f"Last date:  {df['date'].max()}")


# ============================================================
# VALIDATE EXPECTED COVERAGE
# ============================================================

if df["date"].min() != EXPECTED_START_DATE:
    raise ValueError(
        "Unexpected first NDVI date: "
        f"{df['date'].min()} "
        f"(expected {EXPECTED_START_DATE})"
    )

if df["date"].max() != EXPECTED_END_DATE:
    raise ValueError(
        "Unexpected last NDVI date: "
        f"{df['date'].max()} "
        f"(expected {EXPECTED_END_DATE})"
    )

if df["date"].duplicated().any():
    raise ValueError(
        "Duplicate NDVI observation dates found."
    )


# ============================================================
# PREPARE NATIVE 16-DAY SERIES
# ============================================================

native = df.copy()

native["days_since_previous"] = (
    native["date"]
    .diff()
    .dt.days
)

# Relative valid-pixel coverage is useful as a QA indicator.
# It is calculated against the maximum observed valid-pixel
# count in the full series rather than treated as an absolute
# measure of polygon area.
max_valid_pixels = (
    native["valid_pixel_count"]
    .max()
)

native["valid_pixel_fraction_of_max"] = (
    native["valid_pixel_count"]
    / max_valid_pixels
)

# Flag unusually low-coverage composites for later inspection.
# This is descriptive only; rows are NOT removed.
coverage_q05 = (
    native["valid_pixel_count"]
    .quantile(0.05)
)

native["low_valid_pixel_coverage"] = (
    native["valid_pixel_count"]
    <= coverage_q05
)

native.to_csv(
    NATIVE_OUTPUT,
    index=False
)


# ============================================================
# CREATE WEEKLY TIMELINE
# ============================================================

# NDVI is aligned to the dengue modelling timeline.
#
# Dengue uses Sunday-Saturday epidemiological weeks identified
# by the Sunday week_start_date. Using the dengue timeline here
# ensures that NDVI can be joined directly to the target data.
#
# Weeks before the first MODIS observation remain missing.
# Later weeks use only the most recent MODIS observation
# available on or before the Saturday week end.

if not DENGUE_WEEKLY_FILE.exists():
    raise FileNotFoundError(
        f"Dengue weekly file not found:\n{DENGUE_WEEKLY_FILE}"
    )

dengue_weeks = pd.read_csv(
    DENGUE_WEEKLY_FILE,
    usecols=["week_start_date"]
)

dengue_weeks["week_start_date"] = pd.to_datetime(
    dengue_weeks["week_start_date"],
    errors="coerce"
)

if dengue_weeks["week_start_date"].isna().any():
    raise ValueError(
        "One or more dengue week_start_date values could not be parsed."
    )

# Create a COMPLETE Sunday weekly sequence between the first and
# last dengue weeks. This deliberately includes the four known
# missing dengue surveillance weeks so that NDVI processing is
# independent of gaps in the target observations.
first_week_start = dengue_weeks["week_start_date"].min()
last_week_start = dengue_weeks["week_start_date"].max()

weekly = pd.DataFrame({
    "week_start_date": pd.date_range(
        start=first_week_start,
        end=last_week_start,
        freq="W-SUN"
    )
})

weekly["week_end_date"] = (
    weekly["week_start_date"]
    + pd.to_timedelta(
        6,
        unit="D"
    )
)


# ============================================================
# ALIGN NDVI TO WEEKLY TIMELINE
# ============================================================

# Important modelling decision:
#
# Each weekly row receives the most recent MODIS observation
# available ON OR BEFORE the week end date.
#
# This avoids look-ahead leakage. We do not linearly interpolate
# using a future satellite observation.

ndvi_columns = [
    "date",
    "ndvi_mean",
    "ndvi_median",
    "ndvi_std",
    "ndvi_min",
    "ndvi_max",
    "valid_pixel_count",
    "valid_pixel_fraction_of_max",
    "low_valid_pixel_coverage",
    "modis_image_id",
]

ndvi_for_merge = (
    native[
        ndvi_columns
    ]
    .sort_values("date")
    .copy()
)

weekly = pd.merge_asof(
    weekly.sort_values(
        "week_end_date"
    ),
    ndvi_for_merge,
    left_on="week_end_date",
    right_on="date",
    direction="backward",
    allow_exact_matches=True
)

weekly = weekly.rename(
    columns={
        "date":
            "ndvi_observation_date"
    }
)


# ============================================================
# NDVI AGE / STALENESS
# ============================================================

weekly["ndvi_age_days"] = (
    weekly["week_end_date"]
    - weekly["ndvi_observation_date"]
).dt.days

weekly["ndvi_available"] = (
    weekly["ndvi_observation_date"]
    .notna()
)

weekly["ndvi_within_age_limit"] = (
    weekly["ndvi_available"]
    &
    (
        weekly["ndvi_age_days"]
        <= MAX_NDVI_AGE_DAYS
    )
)

weekly["ndvi_stale"] = (
    weekly["ndvi_available"]
    &
    ~weekly["ndvi_within_age_limit"]
)


# ============================================================
# NULL STALE VALUES
# ============================================================

# Stale NDVI values are retained in the raw alignment fields for
# auditability, but modelling feature columns are nulled when the
# most recent observation is more than MAX_NDVI_AGE_DAYS old.

feature_columns = [
    "ndvi_mean",
    "ndvi_median",
    "ndvi_std",
    "ndvi_min",
    "ndvi_max",
]

for column in feature_columns:

    modelling_column = (
        f"{column}_weekly"
    )

    weekly[
        modelling_column
    ] = weekly[
        column
    ]

    weekly.loc[
        ~weekly[
            "ndvi_within_age_limit"
        ],
        modelling_column
    ] = np.nan


# ============================================================
# FIRST-WEEK / PRE-COVERAGE HANDLING
# ============================================================

# Weeks before the first MODIS observation legitimately have
# no NDVI value. These are left missing rather than backfilled
# from a future observation.

pre_ndvi_weeks = (
    weekly[
        "ndvi_observation_date"
    ]
    .isna()
    .sum()
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

weekly = weekly[
    [
        "week_start_date",
        "week_end_date",

        "ndvi_observation_date",
        "ndvi_age_days",

        "ndvi_mean_weekly",
        "ndvi_median_weekly",
        "ndvi_std_weekly",
        "ndvi_min_weekly",
        "ndvi_max_weekly",

        "valid_pixel_count",
        "valid_pixel_fraction_of_max",
        "low_valid_pixel_coverage",

        "ndvi_available",
        "ndvi_within_age_limit",
        "ndvi_stale",

        "modis_image_id",
    ]
]

weekly.to_csv(
    WEEKLY_OUTPUT,
    index=False
)


# ============================================================
# PROCESSING SUMMARY
# ============================================================

native_intervals = (
    native[
        "days_since_previous"
    ]
    .dropna()
    .value_counts()
    .sort_index()
)

summary = []

summary.append("=" * 78)
summary.append("MAYNAS MOD13Q1 NDVI PROCESSING SUMMARY")
summary.append("=" * 78)

summary.append("")
summary.append(f"Input rows: {len(df):,}")
summary.append(f"First MODIS date: {df['date'].min()}")
summary.append(f"Last MODIS date: {df['date'].max()}")

summary.append("")
summary.append("Native interval counts:")

for interval, count in (
    native_intervals.items()
):
    summary.append(
        f"  {int(interval)} days: {count:,}"
    )

summary.append("")
summary.append(
    "Valid-pixel coverage:"
)

summary.append(
    "  Minimum valid pixels: "
    f"{native['valid_pixel_count'].min():,.0f}"
)

summary.append(
    "  Median valid pixels: "
    f"{native['valid_pixel_count'].median():,.0f}"
)

summary.append(
    "  Maximum valid pixels: "
    f"{native['valid_pixel_count'].max():,.0f}"
)

summary.append(
    "  5th percentile threshold: "
    f"{coverage_q05:,.0f}"
)

summary.append(
    "  Low-coverage composites flagged: "
    f"{native['low_valid_pixel_coverage'].sum():,}"
)

summary.append("")
summary.append(
    f"Weekly rows: {len(weekly):,}"
)

summary.append(
    "Weeks before first available MODIS observation: "
    f"{pre_ndvi_weeks:,}"
)

summary.append(
    "Weeks with NDVI available: "
    f"{weekly['ndvi_available'].sum():,}"
)

summary.append(
    "Weeks within NDVI age limit: "
    f"{weekly['ndvi_within_age_limit'].sum():,}"
)

summary.append(
    "Weeks flagged as stale: "
    f"{weekly['ndvi_stale'].sum():,}"
)

summary.append(
    "Maximum permitted NDVI age: "
    f"{MAX_NDVI_AGE_DAYS} days"
)

summary.append("")
summary.append(
    "Alignment method: each Sunday-Saturday weekly row is assigned "
    "the most recent MOD13Q1 observation available on or before "
    "the week end date."
)

summary.append(
    "No future MODIS observation is used to estimate an earlier "
    "week, avoiding look-ahead leakage."
)

summary.append(
    "NDVI values older than the configured age threshold are "
    "retained for auditability but nulled in the *_weekly "
    "modelling feature columns."
)

summary.append("")
summary.append(
    f"Native output: {NATIVE_OUTPUT}"
)

summary.append(
    f"Weekly output: {WEEKLY_OUTPUT}"
)

summary_text = "\n".join(
    summary
)

SUMMARY_OUTPUT.write_text(
    summary_text,
    encoding="utf-8"
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print()
print(
    summary_text
)

print()
print("=" * 78)
print(
    "PROCESSING COMPLETE"
)
print("=" * 78)

print(
    f"Native:  {NATIVE_OUTPUT}"
)

print(
    f"Weekly:  {WEEKLY_OUTPUT}"
)

print(
    f"Summary: {SUMMARY_OUTPUT}"
)