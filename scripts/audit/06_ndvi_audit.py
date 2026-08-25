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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ROW_AUDIT_OUTPUT = (
    OUTPUT_DIR
    / "ndvi_row_validation.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "ndvi_audit_summary.txt"
)


# ============================================================
# EXPECTED STRUCTURE
# ============================================================

EXPECTED_COLUMNS = {
    "date",
    "year",
    "month",
    "day_of_year",
    "ndvi_mean",
    "ndvi_median",
    "ndvi_std",
    "ndvi_min",
    "ndvi_max",
    "valid_pixel_count",
    "modis_image_id",
}

EXPECTED_START_DATE = pd.Timestamp("2000-02-18")
EXPECTED_END_DATE = pd.Timestamp("2023-12-19")

EXPECTED_CADENCE_DAYS = 16

NDVI_MIN_ALLOWED = -1.0
NDVI_MAX_ALLOWED = 1.0


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 78)
print("MAYNAS MOD13Q1 NDVI AUDIT")
print("=" * 78)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"NDVI input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Input: {INPUT_FILE}"
)

print(
    f"Rows loaded: {len(df):,}"
)


# ============================================================
# COLUMN VALIDATION
# ============================================================

actual_columns = set(
    df.columns
)

missing_columns = (
    EXPECTED_COLUMNS
    - actual_columns
)

unexpected_columns = (
    actual_columns
    - EXPECTED_COLUMNS
)

if missing_columns:
    raise ValueError(
        "Missing expected NDVI columns:\n"
        + "\n".join(
            f" - {column}"
            for column
            in sorted(
                missing_columns
            )
        )
    )


# ============================================================
# TYPE CONVERSION
# ============================================================

df[
    "date"
] = pd.to_datetime(
    df[
        "date"
    ],
    errors="coerce"
)

numeric_columns = [
    "year",
    "month",
    "day_of_year",
    "ndvi_mean",
    "ndvi_median",
    "ndvi_std",
    "ndvi_min",
    "ndvi_max",
    "valid_pixel_count",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# ROW-LEVEL VALIDATION
# ============================================================

audit = pd.DataFrame({
    "date":
        df["date"],

    "modis_image_id":
        df["modis_image_id"],
})


# ------------------------------------------------------------
# Basic date checks
# ------------------------------------------------------------

audit[
    "date_valid"
] = df[
    "date"
].notna()

audit[
    "year_matches_date"
] = (
    df[
        "year"
    ]
    ==
    df[
        "date"
    ].dt.year
)

audit[
    "month_matches_date"
] = (
    df[
        "month"
    ]
    ==
    df[
        "date"
    ].dt.month
)

audit[
    "day_of_year_matches_date"
] = (
    df[
        "day_of_year"
    ]
    ==
    df[
        "date"
    ].dt.dayofyear
)


# ------------------------------------------------------------
# NDVI physical range checks
# ------------------------------------------------------------

for column in [
    "ndvi_mean",
    "ndvi_median",
    "ndvi_min",
    "ndvi_max",
]:

    audit[
        f"{column}_in_range"
    ] = (
        df[column]
        .between(
            NDVI_MIN_ALLOWED,
            NDVI_MAX_ALLOWED,
            inclusive="both"
        )
    )


audit[
    "ndvi_std_non_negative"
] = (
    df[
        "ndvi_std"
    ]
    >= 0
)


# ------------------------------------------------------------
# Internal ordering checks
# ------------------------------------------------------------

audit[
    "ndvi_min_le_mean"
] = (
    df[
        "ndvi_min"
    ]
    <=
    df[
        "ndvi_mean"
    ]
)

audit[
    "ndvi_mean_le_max"
] = (
    df[
        "ndvi_mean"
    ]
    <=
    df[
        "ndvi_max"
    ]
)

audit[
    "ndvi_min_le_median"
] = (
    df[
        "ndvi_min"
    ]
    <=
    df[
        "ndvi_median"
    ]
)

audit[
    "ndvi_median_le_max"
] = (
    df[
        "ndvi_median"
    ]
    <=
    df[
        "ndvi_max"
    ]
)


# ------------------------------------------------------------
# Valid-pixel checks
# ------------------------------------------------------------

audit[
    "valid_pixel_count_positive"
] = (
    df[
        "valid_pixel_count"
    ]
    > 0
)

audit[
    "valid_pixel_count_integer_like"
] = np.isclose(
    df[
        "valid_pixel_count"
    ],
    np.round(
        df[
            "valid_pixel_count"
        ]
    ),
    equal_nan=False
)


# ------------------------------------------------------------
# Required-value completeness
# ------------------------------------------------------------

required_value_columns = [
    "date",
    "ndvi_mean",
    "ndvi_median",
    "ndvi_std",
    "ndvi_min",
    "ndvi_max",
    "valid_pixel_count",
    "modis_image_id",
]

audit[
    "required_values_complete"
] = (
    df[
        required_value_columns
    ]
    .notna()
    .all(
        axis=1
    )
)


# ============================================================
# TEMPORAL AUDIT
# ============================================================

sorted_df = (
    df
    .sort_values(
        "date"
    )
    .reset_index(
        drop=True
    )
)

sorted_df[
    "days_since_previous"
] = (
    sorted_df[
        "date"
    ]
    .diff()
    .dt.days
)

duplicate_dates = (
    sorted_df[
        "date"
    ]
    .duplicated(
        keep=False
    )
)

duplicate_image_ids = (
    sorted_df[
        "modis_image_id"
    ]
    .duplicated(
        keep=False
    )
)

date_intervals = (
    sorted_df[
        "days_since_previous"
    ]
    .dropna()
)

unexpected_intervals = (
    sorted_df[
        sorted_df[
            "days_since_previous"
        ].notna()
        &
        (
            sorted_df[
                "days_since_previous"
            ]
            != EXPECTED_CADENCE_DAYS
        )
    ][
        [
            "date",
            "days_since_previous",
            "modis_image_id",
        ]
    ]
)

first_date = (
    sorted_df[
        "date"
    ].min()
)

last_date = (
    sorted_df[
        "date"
    ].max()
)


# ============================================================
# OVERALL ROW VALIDATION
# ============================================================

validation_columns = [
    column
    for column
    in audit.columns
    if column not in {
        "date",
        "modis_image_id",
    }
]

audit[
    "overall_row_validation"
] = np.where(
    audit[
        validation_columns
    ].all(
        axis=1
    ),
    "PASS",
    "FAIL"
)

audit.to_csv(
    ROW_AUDIT_OUTPUT,
    index=False
)


# ============================================================
# BUILD SUMMARY
# ============================================================

summary = []

summary.append(
    "=" * 78
)

summary.append(
    "MAYNAS MOD13Q1 NDVI AUDIT SUMMARY"
)

summary.append(
    "=" * 78
)

summary.append("")

summary.append(
    f"Input file: {INPUT_FILE}"
)

summary.append(
    f"Rows: {len(df):,}"
)

summary.append(
    f"Columns: {len(df.columns):,}"
)

summary.append("")

summary.append(
    "COLUMN CHECK"
)

summary.append(
    "-" * 78
)

summary.append(
    f"Missing expected columns: "
    f"{len(missing_columns):,}"
)

summary.append(
    f"Unexpected columns: "
    f"{len(unexpected_columns):,}"
)

if unexpected_columns:

    for column in sorted(
        unexpected_columns
    ):

        summary.append(
            f"  - {column}"
        )


# ------------------------------------------------------------
# Date coverage
# ------------------------------------------------------------

summary.append("")
summary.append(
    "TEMPORAL COVERAGE"
)

summary.append(
    "-" * 78
)

summary.append(
    f"First date: {first_date}"
)

summary.append(
    f"Expected first date: "
    f"{EXPECTED_START_DATE}"
)

summary.append(
    f"First date matches expectation: "
    f"{first_date == EXPECTED_START_DATE}"
)

summary.append(
    f"Last date: {last_date}"
)

summary.append(
    f"Expected last date: "
    f"{EXPECTED_END_DATE}"
)

summary.append(
    f"Last date matches expectation: "
    f"{last_date == EXPECTED_END_DATE}"
)

summary.append(
    f"Duplicate dates: "
    f"{int(duplicate_dates.sum()):,}"
)

summary.append(
    f"Duplicate MODIS image IDs: "
    f"{int(duplicate_image_ids.sum()):,}"
)

summary.append("")

summary.append(
    "Observed interval counts:"
)

interval_counts = (
    date_intervals
    .value_counts()
    .sort_index()
)

for interval, count in (
    interval_counts.items()
):

    summary.append(
        f"  {int(interval)} days: "
        f"{count:,}"
    )

summary.append(
    f"Intervals not equal to "
    f"{EXPECTED_CADENCE_DAYS} days: "
    f"{len(unexpected_intervals):,}"
)

if len(
    unexpected_intervals
) > 0:

    interval_output = (
        OUTPUT_DIR
        / "ndvi_unexpected_date_intervals.csv"
    )

    unexpected_intervals.to_csv(
        interval_output,
        index=False
    )

    summary.append(
        f"Unexpected interval file: "
        f"{interval_output}"
    )


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

summary.append("")
summary.append(
    "MISSING VALUES"
)

summary.append(
    "-" * 78
)

for column in df.columns:

    missing_count = int(
        df[
            column
        ]
        .isna()
        .sum()
    )

    summary.append(
        f"  {column}: "
        f"{missing_count:,}"
    )


# ------------------------------------------------------------
# NDVI ranges
# ------------------------------------------------------------

summary.append("")
summary.append(
    "NDVI VALUE RANGES"
)

summary.append(
    "-" * 78
)

for column in [
    "ndvi_mean",
    "ndvi_median",
    "ndvi_std",
    "ndvi_min",
    "ndvi_max",
]:

    summary.append(
        f"  {column}: "
        f"min={df[column].min()}, "
        f"max={df[column].max()}"
    )


# ------------------------------------------------------------
# Valid pixel count
# ------------------------------------------------------------

summary.append("")
summary.append(
    "VALID PIXEL COUNT"
)

summary.append(
    "-" * 78
)

summary.append(
    "Minimum valid pixels: "
    f"{df['valid_pixel_count'].min():,.0f}"
)

summary.append(
    "Median valid pixels: "
    f"{df['valid_pixel_count'].median():,.0f}"
)

summary.append(
    "Maximum valid pixels: "
    f"{df['valid_pixel_count'].max():,.0f}"
)

summary.append(
    "Rows with zero/non-positive "
    "valid pixels: "
    f"{int((df['valid_pixel_count'] <= 0).sum()):,}"
)


# ------------------------------------------------------------
# Row validation
# ------------------------------------------------------------

summary.append("")
summary.append(
    "ROW-LEVEL VALIDATION"
)

summary.append(
    "-" * 78
)

validation_counts = (
    audit[
        "overall_row_validation"
    ]
    .value_counts()
)

for status, count in (
    validation_counts.items()
):

    summary.append(
        f"  {status}: "
        f"{count:,}"
    )

summary.append("")

summary.append(
    "Individual validation failures:"
)

for column in validation_columns:

    fail_count = int(
        (
            ~audit[
                column
            ]
        ).sum()
    )

    summary.append(
        f"  {column}: "
        f"{fail_count:,}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

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
    "AUDIT COMPLETE"
)
print("=" * 78)

print(
    f"\nRow validation:\n"
    f"{ROW_AUDIT_OUTPUT}"
)

print(
    f"\nSummary:\n"
    f"{SUMMARY_OUTPUT}"
)
