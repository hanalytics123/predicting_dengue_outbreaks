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
    / "processed"
    / "modelling"
    / "maynas_dengue_modelling_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "modelling_dataset_audit_summary.txt"
)

COLUMN_AUDIT_OUTPUT = (
    OUTPUT_DIR
    / "modelling_dataset_column_audit.csv"
)

MISSING_ROWS_OUTPUT = (
    OUTPUT_DIR
    / "modelling_dataset_missing_rows.csv"
)

HIGH_CORRELATION_OUTPUT = (
    OUTPUT_DIR
    / "modelling_dataset_high_correlations.csv"
)

KNOWN_DENGUE_GAPS_OUTPUT = (
    OUTPUT_DIR
    / "modelling_dataset_known_dengue_gaps.csv"
)


# ============================================================
# SETTINGS
# ============================================================

JOIN_KEY = "week_start_date"
TARGET = "dengue_cases"

EXPECTED_ROWS = 1248
EXPECTED_FIRST_WEEK = pd.Timestamp("2000-01-02")
EXPECTED_LAST_WEEK = pd.Timestamp("2023-12-24")

KNOWN_DENGUE_GAPS = pd.DatetimeIndex([
    "2000-04-30",
    "2000-06-11",
    "2000-06-25",
    "2000-07-09",
])

HIGH_CORRELATION_THRESHOLD = 0.90

# Plausibility checks are deliberately broad. They are intended
# to flag obvious processing/unit problems, not define biological
# thresholds for modelling.
RANGE_RULES = {
    "dengue_cases": (0, None),

    "precip_sum_mm": (0, None),
    "precip_mean_daily_mm": (0, None),
    "precip_max_daily_mm": (0, None),
    "precip_min_daily_mm": (0, None),
    "rain_days": (0, 7),

    "temperature_c_mean": (-20, 60),
    "temperature_c_min": (-20, 60),
    "temperature_c_max": (-20, 60),

    "dewpoint_c_mean": (-40, 50),
    "dewpoint_c_min": (-40, 50),
    "dewpoint_c_max": (-40, 50),

    "relative_humidity_pct_mean": (0, 100),
    "relative_humidity_pct_min": (0, 100),
    "relative_humidity_pct_max": (0, 100),

    "specific_humidity_kgkg_mean": (0, 0.1),
    "specific_humidity_kgkg_min": (0, 0.1),
    "specific_humidity_kgkg_max": (0, 0.1),

    "wind_speed_ms_mean": (0, 100),
    "wind_speed_ms_min": (0, 100),
    "wind_speed_ms_max": (0, 100),

    "surface_pressure_hpa_mean": (700, 1100),
    "surface_pressure_hpa_min": (700, 1100),
    "surface_pressure_hpa_max": (700, 1100),

    "ndvi_mean_weekly": (-1, 1),
    "ndvi_median_weekly": (-1, 1),
    "ndvi_min_weekly": (-1, 1),
    "ndvi_max_weekly": (-1, 1),
    "ndvi_std_weekly": (0, None),
    "ndvi_age_days": (0, None),
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 78)
print("MAYNAS MODELLING DATASET AUDIT")
print("=" * 78)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Modelling dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

if JOIN_KEY not in df.columns:
    raise ValueError(
        f"Required column '{JOIN_KEY}' not found."
    )

if TARGET not in df.columns:
    raise ValueError(
        f"Required target '{TARGET}' not found."
    )

df[JOIN_KEY] = pd.to_datetime(
    df[JOIN_KEY],
    errors="coerce"
)

date_columns = [
    column
    for column in df.columns
    if (
        column.endswith("_date")
        or column == JOIN_KEY
    )
]

for column in date_columns:
    if column == JOIN_KEY:
        continue

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

df = (
    df
    .sort_values(JOIN_KEY)
    .reset_index(drop=True)
)


# ============================================================
# BASIC STRUCTURE
# ============================================================

row_count = len(df)
column_count = len(df.columns)

duplicate_week_count = int(
    df[JOIN_KEY]
    .duplicated()
    .sum()
)

duplicate_row_count = int(
    df.duplicated()
    .sum()
)

invalid_week_count = int(
    df[JOIN_KEY]
    .isna()
    .sum()
)

first_week = df[JOIN_KEY].min()
last_week = df[JOIN_KEY].max()

all_week_starts_sunday = bool(
    (
        df[JOIN_KEY]
        .dropna()
        .dt.weekday
        == 6
    ).all()
)


# ============================================================
# DENGUE TIMELINE / KNOWN GAPS
# ============================================================

expected_calendar_weeks = pd.date_range(
    start=EXPECTED_FIRST_WEEK,
    end=EXPECTED_LAST_WEEK,
    freq="W-SUN"
)

observed_weeks = pd.DatetimeIndex(
    df[JOIN_KEY]
    .dropna()
    .unique()
)

missing_calendar_weeks = (
    expected_calendar_weeks
    .difference(
        observed_weeks
    )
)

unexpected_missing_weeks = (
    missing_calendar_weeks
    .difference(
        KNOWN_DENGUE_GAPS
    )
)

known_gaps_recovered = (
    KNOWN_DENGUE_GAPS
    .intersection(
        missing_calendar_weeks
    )
)

pd.DataFrame({
    "known_missing_dengue_week":
        KNOWN_DENGUE_GAPS,
    "absent_from_modelling_dataset":
        [
            week not in observed_weeks
            for week in KNOWN_DENGUE_GAPS
        ],
}).to_csv(
    KNOWN_DENGUE_GAPS_OUTPUT,
    index=False
)


# ============================================================
# MISSINGNESS
# ============================================================

missing_counts = (
    df
    .isna()
    .sum()
)

missing_pct = (
    missing_counts
    / len(df)
    * 100
)

rows_with_missing = (
    df[
        df.isna()
        .any(axis=1)
    ]
    .copy()
)

if not rows_with_missing.empty:
    rows_with_missing.to_csv(
        MISSING_ROWS_OUTPUT,
        index=False
    )


# ============================================================
# INFINITE VALUES
# ============================================================

numeric_columns = (
    df
    .select_dtypes(
        include=[np.number]
    )
    .columns
    .tolist()
)

infinite_counts = {}

for column in numeric_columns:
    infinite_counts[column] = int(
        np.isinf(
            df[column]
            .to_numpy(
                dtype=float,
                na_value=np.nan
            )
        ).sum()
    )


# ============================================================
# CONSTANT / NEAR-CONSTANT FIELDS
# ============================================================

unique_non_null = (
    df
    .nunique(
        dropna=True
    )
)

constant_columns = (
    unique_non_null[
        unique_non_null <= 1
    ]
    .index
    .tolist()
)

# A descriptive flag only. These are not automatically removed.
near_constant_columns = []

for column in df.columns:

    non_null = (
        df[column]
        .dropna()
    )

    if (
        len(non_null) == 0
        or column in constant_columns
    ):
        continue

    proportions = (
        non_null
        .value_counts(
            normalize=True
        )
    )

    if (
        not proportions.empty
        and proportions.iloc[0] >= 0.95
    ):
        near_constant_columns.append(
            column
        )


# ============================================================
# RANGE / PLAUSIBILITY CHECKS
# ============================================================

range_results = {}

for column, (
    lower,
    upper
) in RANGE_RULES.items():

    if column not in df.columns:
        continue

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    invalid = pd.Series(
        False,
        index=df.index
    )

    if lower is not None:
        invalid = (
            invalid
            | (values < lower)
        )

    if upper is not None:
        invalid = (
            invalid
            | (values > upper)
        )

    range_results[column] = int(
        invalid.sum()
    )


# ============================================================
# QA / COMPLETENESS FLAGS
# ============================================================

qa_columns = [
    column
    for column in df.columns
    if (
        "complete" in column.lower()
        or "missing" in column.lower()
        or "available" in column.lower()
        or "stale" in column.lower()
        or "coverage" in column.lower()
        or "observations" in column.lower()
        or "valid_pixel" in column.lower()
        or column.startswith("has_")
    )
]

qa_summary = {}

for column in qa_columns:

    series = df[column]

    if pd.api.types.is_bool_dtype(
        series
    ) or set(
        series.dropna().unique()
    ).issubset(
        {True, False}
    ):

        qa_summary[column] = {
            "true": int(
                (series == True).sum()
            ),
            "false": int(
                (series == False).sum()
            ),
            "missing": int(
                series.isna().sum()
            ),
        }


# ============================================================
# WEEK-END CONSISTENCY
# ============================================================

week_end_columns = [
    column
    for column in df.columns
    if column.endswith(
        "week_end_date"
    )
]

week_end_mismatch_counts = {}

expected_week_end = (
    df[JOIN_KEY]
    + pd.to_timedelta(
        6,
        unit="D"
    )
)

for column in week_end_columns:

    actual = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    comparable = (
        actual.notna()
        & expected_week_end.notna()
    )

    mismatch_count = int(
        (
            actual[
                comparable
            ]
            != expected_week_end[
                comparable
            ]
        ).sum()
    )

    week_end_mismatch_counts[
        column
    ] = mismatch_count


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

target = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

target_quantiles = (
    target.quantile(
        [
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)

target_stats = {
    "count":
        int(target.count()),

    "missing":
        int(target.isna().sum()),

    "zero_weeks":
        int((target == 0).sum()),

    "negative_weeks":
        int((target < 0).sum()),

    "mean":
        float(target.mean()),

    "median":
        float(target.median()),

    "std":
        float(target.std()),

    "min":
        float(target.min()),

    "max":
        float(target.max()),

    "skewness":
        float(target.skew()),
}


# ============================================================
# HIGH CORRELATIONS
# ============================================================

# Exclude obvious identifier/calendar/QA columns from this
# screening so the output focuses on candidate numeric features.
correlation_exclusions = {
    "calendar_year",
    "calendar_month",
    "iso_year",
    "iso_week",
}

correlation_columns = [
    column
    for column in numeric_columns
    if column not in correlation_exclusions
]

corr = (
    df[
        correlation_columns
    ]
    .corr()
)

high_corr_rows = []

for i, first in enumerate(
    corr.columns
):

    for second in corr.columns[
        i + 1:
    ]:

        value = corr.loc[
            first,
            second
        ]

        if (
            pd.notna(value)
            and abs(value)
            >= HIGH_CORRELATION_THRESHOLD
        ):

            high_corr_rows.append({
                "variable_1":
                    first,

                "variable_2":
                    second,

                "correlation":
                    value,

                "absolute_correlation":
                    abs(value),
            })


high_corr_df = (
    pd.DataFrame(
        high_corr_rows
    )
)

if not high_corr_df.empty:

    high_corr_df = (
        high_corr_df
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
    )

    high_corr_df.to_csv(
        HIGH_CORRELATION_OUTPUT,
        index=False
    )

else:

    pd.DataFrame(
        columns=[
            "variable_1",
            "variable_2",
            "correlation",
            "absolute_correlation",
        ]
    ).to_csv(
        HIGH_CORRELATION_OUTPUT,
        index=False
    )


# ============================================================
# COLUMN AUDIT
# ============================================================

column_rows = []

for column in df.columns:

    series = df[column]

    numeric_series = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        if column in numeric_columns
        else None
    )

    column_rows.append({
        "column":
            column,

        "dtype":
            str(
                series.dtype
            ),

        "missing_count":
            int(
                series.isna().sum()
            ),

        "missing_pct":
            float(
                series.isna().mean()
                * 100
            ),

        "unique_non_null":
            int(
                series.nunique(
                    dropna=True
                )
            ),

        "constant":
            column
            in constant_columns,

        "near_constant_95pct":
            column
            in near_constant_columns,

        "infinite_count":
            infinite_counts.get(
                column,
                0
            ),

        "range_rule_violations":
            range_results.get(
                column
            ),

        "numeric_min":
            (
                float(
                    numeric_series.min()
                )
                if (
                    numeric_series is not None
                    and numeric_series.notna().any()
                )
                else None
            ),

        "numeric_max":
            (
                float(
                    numeric_series.max()
                )
                if (
                    numeric_series is not None
                    and numeric_series.notna().any()
                )
                else None
            ),
    })


column_audit = pd.DataFrame(
    column_rows
)

column_audit.to_csv(
    COLUMN_AUDIT_OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append(
    "=" * 78
)

summary.append(
    "MAYNAS MODELLING DATASET AUDIT SUMMARY"
)

summary.append(
    "=" * 78
)

summary.append("")
summary.append(
    "DATASET STRUCTURE"
)

summary.append(
    "-" * 78
)

summary.append(
    f"Rows: {row_count:,}"
)

summary.append(
    f"Columns: {column_count:,}"
)

summary.append(
    f"Expected rows: {EXPECTED_ROWS:,}"
)

summary.append(
    f"Row count matches expectation: "
    f"{row_count == EXPECTED_ROWS}"
)

summary.append(
    f"First week: {first_week}"
)

summary.append(
    f"Expected first week: "
    f"{EXPECTED_FIRST_WEEK}"
)

summary.append(
    f"Last week: {last_week}"
)

summary.append(
    f"Expected last week: "
    f"{EXPECTED_LAST_WEEK}"
)

summary.append(
    f"All week starts Sunday: "
    f"{all_week_starts_sunday}"
)

summary.append(
    f"Duplicate weeks: "
    f"{duplicate_week_count:,}"
)

summary.append(
    f"Duplicate full rows: "
    f"{duplicate_row_count:,}"
)

summary.append(
    f"Invalid/missing week_start_date: "
    f"{invalid_week_count:,}"
)


summary.append("")
summary.append(
    "DENGUE SURVEILLANCE TIMELINE"
)

summary.append(
    "-" * 78
)

summary.append(
    f"Calendar weeks in study period: "
    f"{len(expected_calendar_weeks):,}"
)

summary.append(
    f"Observed dengue weeks: "
    f"{len(observed_weeks):,}"
)

summary.append(
    f"Missing calendar weeks: "
    f"{len(missing_calendar_weeks):,}"
)

summary.append(
    f"Known missing dengue weeks recovered: "
    f"{len(known_gaps_recovered):,} / "
    f"{len(KNOWN_DENGUE_GAPS):,}"
)

summary.append(
    f"Unexpected missing weeks: "
    f"{len(unexpected_missing_weeks):,}"
)

for week in missing_calendar_weeks:
    label = (
        "KNOWN"
        if week in KNOWN_DENGUE_GAPS
        else "UNEXPECTED"
    )

    summary.append(
        f"  {week.date()} - {label}"
    )


summary.append("")
summary.append(
    "MISSINGNESS"
)

summary.append(
    "-" * 78
)

summary.append(
    f"Rows containing >=1 missing value: "
    f"{len(rows_with_missing):,}"
)

summary.append(
    f"Columns containing missing values: "
    f"{int((missing_counts > 0).sum()):,}"
)

if (
    missing_counts > 0
).any():

    for column, count in (
        missing_counts[
            missing_counts > 0
        ]
        .sort_values(
            ascending=False
        )
        .items()
    ):

        summary.append(
            f"  {column}: "
            f"{int(count):,} "
            f"({missing_pct[column]:.2f}%)"
        )

else:

    summary.append(
        "No missing values."
    )


summary.append("")
summary.append(
    "INFINITE VALUES"
)

summary.append(
    "-" * 78
)

total_inf = sum(
    infinite_counts.values()
)

summary.append(
    f"Total infinite numeric values: "
    f"{total_inf:,}"
)

for column, count in (
    infinite_counts.items()
):

    if count > 0:

        summary.append(
            f"  {column}: "
            f"{count:,}"
        )


summary.append("")
summary.append(
    "TARGET DISTRIBUTION"
)

summary.append(
    "-" * 78
)

for key in [
    "count",
    "missing",
    "zero_weeks",
    "negative_weeks",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "skewness",
]:

    summary.append(
        f"{key}: "
        f"{target_stats[key]}"
    )

summary.append("")
summary.append(
    "Target quantiles:"
)

for quantile, value in (
    target_quantiles.items()
):

    summary.append(
        f"  {quantile:.0%}: "
        f"{value}"
    )


summary.append("")
summary.append(
    "RANGE / PLAUSIBILITY CHECKS"
)

summary.append(
    "-" * 78
)

if range_results:

    for column, count in (
        range_results.items()
    ):

        summary.append(
            f"{column}: "
            f"{count:,} violation(s)"
        )

else:

    summary.append(
        "No configured range checks applied."
    )


summary.append("")
summary.append(
    "CONSTANT FIELDS"
)

summary.append(
    "-" * 78
)

summary.append(
    f"Constant columns: "
    f"{len(constant_columns):,}"
)

for column in constant_columns:

    summary.append(
        f"  {column}"
    )

summary.append("")
summary.append(
    f"Near-constant columns (>=95% same non-null value): "
    f"{len(near_constant_columns):,}"
)

for column in near_constant_columns:

    summary.append(
        f"  {column}"
    )


summary.append("")
summary.append(
    "QA / COMPLETENESS FLAGS"
)

summary.append(
    "-" * 78
)

if qa_summary:

    for column, counts in (
        qa_summary.items()
    ):

        summary.append(
            f"{column}: "
            f"True={counts['true']:,}, "
            f"False={counts['false']:,}, "
            f"Missing={counts['missing']:,}"
        )

else:

    summary.append(
        "No Boolean QA/completeness flags identified."
    )


summary.append("")
summary.append(
    "WEEK-END CONSISTENCY"
)

summary.append(
    "-" * 78
)

if week_end_mismatch_counts:

    for column, count in (
        week_end_mismatch_counts.items()
    ):

        summary.append(
            f"{column}: "
            f"{count:,} mismatch(es)"
        )

else:

    summary.append(
        "No source week_end_date columns found."
    )


summary.append("")
summary.append(
    "HIGH CORRELATIONS"
)

summary.append(
    "-" * 78
)

summary.append(
    f"Threshold: "
    f"|r| >= {HIGH_CORRELATION_THRESHOLD}"
)

summary.append(
    f"Pairs flagged: "
    f"{len(high_corr_df):,}"
)

if not high_corr_df.empty:

    for _, row in (
        high_corr_df
        .head(30)
        .iterrows()
    ):

        summary.append(
            f"  {row['variable_1']} <-> "
            f"{row['variable_2']}: "
            f"r={row['correlation']:.4f}"
        )

    if len(
        high_corr_df
    ) > 30:

        summary.append(
            "  ... additional pairs saved to "
            "modelling_dataset_high_correlations.csv"
        )


summary.append("")
summary.append(
    "AUDIT OUTPUTS"
)

summary.append(
    "-" * 78
)

summary.append(
    f"Column audit: "
    f"{COLUMN_AUDIT_OUTPUT}"
)

summary.append(
    f"Missing rows: "
    f"{MISSING_ROWS_OUTPUT}"
)

summary.append(
    f"High correlations: "
    f"{HIGH_CORRELATION_OUTPUT}"
)

summary.append(
    f"Known dengue gaps: "
    f"{KNOWN_DENGUE_GAPS_OUTPUT}"
)


# ============================================================
# OVERALL STATUS
# ============================================================

critical_issues = []

if row_count != EXPECTED_ROWS:
    critical_issues.append(
        "unexpected row count"
    )

if duplicate_week_count > 0:
    critical_issues.append(
        "duplicate weeks"
    )

if invalid_week_count > 0:
    critical_issues.append(
        "invalid week_start_date"
    )

if not all_week_starts_sunday:
    critical_issues.append(
        "non-Sunday week starts"
    )

if len(
    unexpected_missing_weeks
) > 0:
    critical_issues.append(
        "unexpected dengue timeline gaps"
    )

if total_inf > 0:
    critical_issues.append(
        "infinite numeric values"
    )

if any(
    count > 0
    for count in range_results.values()
):
    critical_issues.append(
        "range/plausibility violations"
    )

summary.append("")
summary.append(
    "OVERALL RESULT"
)

summary.append(
    "-" * 78
)

if critical_issues:

    summary.append(
        "REVIEW REQUIRED"
    )

    for issue in critical_issues:

        summary.append(
            f"  - {issue}"
        )

else:

    summary.append(
        "PASS - no critical structural, temporal, infinite-value "
        "or configured plausibility issues detected."
    )

    summary.append(
        "Expected NDVI missingness and known dengue surveillance "
        "gaps are retained for downstream EDA/modelling decisions."
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
    "AUDIT COMPLETE"
)
print("=" * 78)

print(
    f"Summary: {SUMMARY_OUTPUT}"
)

print(
    f"Column audit: {COLUMN_AUDIT_OUTPUT}"
)

print(
    f"High correlations: {HIGH_CORRELATION_OUTPUT}"
)

if not rows_with_missing.empty:

    print(
        f"Missing rows: {MISSING_ROWS_OUTPUT}"
    )
