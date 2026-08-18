import pandas as pd
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "dengue.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FILE = PROCESSED_DIR / "maynas_dengue_weekly.csv"
VALIDATION_FILE = AUDIT_DIR / "maynas_target_validation.csv"
MISSING_WEEKS_FILE = AUDIT_DIR / "maynas_missing_weeks.csv"

print("=" * 72)
print("PREPARE MAYNAS WEEKLY DENGUE TARGET")
print("=" * 72)
print(f"Project root:     {PROJECT_ROOT}")
print(f"Raw input:        {RAW_FILE}")
print(f"Processed output: {TARGET_FILE}")

# ============================================================
# 1. LOAD RAW DATA
# ============================================================

df = pd.read_csv(RAW_FILE)

required_columns = {
    "adm_0_name",
    "adm_1_name",
    "adm_2_name",
    "S_res",
    "T_res",
    "calendar_start_date",
    "dengue_total",
    "case_definition_standardised",
    "FAO_GAUL_code",
    "UUID",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        "The raw file is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

print(f"\nRaw rows: {len(df):,}")

# ============================================================
# 2. PARSE THE WEEK START DATE
# ============================================================

df["calendar_start_date"] = pd.to_datetime(
    df["calendar_start_date"],
    errors="coerce"
)

invalid_start_dates = df["calendar_start_date"].isna().sum()

if invalid_start_dates:
    raise ValueError(
        f"{invalid_start_dates:,} rows have invalid calendar_start_date values."
    )

# ============================================================
# 3. FILTER TO THE SELECTED STUDY AREA
# ============================================================

maynas = df[
    (df["adm_0_name"].str.upper() == "PERU")
    & (df["adm_1_name"].str.upper() == "LORETO")
    & (df["adm_2_name"].str.upper() == "MAYNAS")
    & (df["S_res"] == "Admin2")
    & (df["T_res"] == "Week")
].copy()

if maynas.empty:
    raise ValueError(
        "No weekly Admin2 records were found for Maynas, Loreto, Peru."
    )

maynas = maynas.sort_values("calendar_start_date").reset_index(drop=True)

print(f"Maynas weekly rows: {len(maynas):,}")
print(
    "Observed period: "
    f"{maynas['calendar_start_date'].min().date()} to "
    f"{maynas['calendar_start_date'].max().date()}"
)

# ============================================================
# 4. VALIDATE ONE RECORD PER WEEK
# ============================================================

duplicate_weeks = maynas[
    maynas.duplicated(
        subset=["calendar_start_date"],
        keep=False
    )
].copy()

if not duplicate_weeks.empty:
    print("\nWARNING: duplicate week-start dates were found.")
    print(
        duplicate_weeks[
            [
                "calendar_start_date",
                "dengue_total",
                "UUID",
                "FAO_GAUL_code",
            ]
        ].head(20)
    )
    raise ValueError(
        "Maynas contains more than one record for at least one week. "
        "Do not aggregate automatically; investigate the duplicate records."
    )

print("\nDuplicate week-start dates: 0")

# ============================================================
# 5. VALIDATE TARGET VALUES
# ============================================================

maynas["dengue_total"] = pd.to_numeric(
    maynas["dengue_total"],
    errors="coerce"
)

missing_cases = maynas["dengue_total"].isna().sum()
negative_cases = (maynas["dengue_total"] < 0).sum()

if missing_cases:
    raise ValueError(
        f"{missing_cases:,} Maynas records have missing/non-numeric dengue_total values."
    )

if negative_cases:
    raise ValueError(
        f"{negative_cases:,} Maynas records have negative dengue_total values."
    )

non_integer_cases = maynas["dengue_total"].mod(1).ne(0).sum()

if non_integer_cases:
    raise ValueError(
        f"{non_integer_cases:,} dengue_total values are not whole numbers."
    )

maynas["dengue_total"] = maynas["dengue_total"].astype("int64")

print(f"Missing dengue totals: {missing_cases}")
print(f"Negative dengue totals: {negative_cases}")
print(f"Zero-case weeks: {(maynas['dengue_total'] == 0).sum():,}")

# ============================================================
# 6. CHECK CASE-DEFINITION CONSISTENCY
# ============================================================

case_definitions = (
    maynas["case_definition_standardised"]
    .fillna("<missing>")
    .value_counts(dropna=False)
)

print("\nCase definitions:")
print(case_definitions)

if len(case_definitions) > 1:
    print(
        "\nWARNING: More than one case definition occurs in the Maynas series. "
        "Review this before modelling."
    )

# ============================================================
# 7. CHECK WEEKLY TEMPORAL CONTINUITY
# ============================================================

observed_dates = pd.DatetimeIndex(
    maynas["calendar_start_date"]
    .drop_duplicates()
    .sort_values()
)

weekday_counts = (
    maynas["calendar_start_date"]
    .dt.day_name()
    .value_counts()
)

print("\nWeek-start weekdays:")
print(weekday_counts)

all_sundays = maynas["calendar_start_date"].dt.dayofweek.eq(6).all()

if not all_sundays:
    print(
        "\nWARNING: Not every Maynas observation begins on a Sunday. "
        "Review the temporal convention before modelling."
    )

expected_dates = pd.date_range(
    start=observed_dates.min(),
    end=observed_dates.max(),
    freq="W-SUN"
)

missing_weeks = expected_dates.difference(observed_dates)
unexpected_weeks = observed_dates.difference(expected_dates)

coverage_pct = (
    len(observed_dates) / len(expected_dates) * 100
    if len(expected_dates)
    else 0
)

print("\nWeekly continuity:")
print(f"Expected weeks:   {len(expected_dates):,}")
print(f"Observed weeks:   {len(observed_dates):,}")
print(f"Missing weeks:    {len(missing_weeks):,}")
print(f"Unexpected dates: {len(unexpected_weeks):,}")
print(f"Coverage:         {coverage_pct:.2f}%")

# ============================================================
# 8. CHECK GAPS BETWEEN CONSECUTIVE OBSERVATIONS
# ============================================================

date_diffs = maynas["calendar_start_date"].diff().dt.days

gap_distribution = (
    date_diffs.dropna()
    .value_counts()
    .sort_index()
)

print("\nGap between consecutive observations (days):")
print(gap_distribution)

irregular_gaps = maynas.loc[
    date_diffs.notna() & date_diffs.ne(7),
    ["calendar_start_date"]
].copy()

if not irregular_gaps.empty:
    irregular_gaps["previous_week"] = (
        maynas["calendar_start_date"].shift(1)
        .loc[irregular_gaps.index]
    )
    irregular_gaps["gap_days"] = (
        irregular_gaps["calendar_start_date"]
        - irregular_gaps["previous_week"]
    ).dt.days

    print("\nIrregular weekly gaps:")
    print(irregular_gaps.head(20))

# ============================================================
# 9. CREATE MODELLING TARGET TABLE
# ============================================================

target = maynas[
    [
        "calendar_start_date",
        "dengue_total",
        "case_definition_standardised",
    ]
].copy()

target = target.rename(
    columns={
        "calendar_start_date": "week_start_date",
        "dengue_total": "dengue_cases",
        "case_definition_standardised": "case_definition",
    }
)

iso_calendar = target["week_start_date"].dt.isocalendar()

target["calendar_year"] = target["week_start_date"].dt.year
target["calendar_month"] = target["week_start_date"].dt.month
target["iso_year"] = iso_calendar["year"].astype("int64")
target["iso_week"] = iso_calendar["week"].astype("int64")

target["country"] = "Peru"
target["adm1_name"] = "Loreto"
target["adm2_name"] = "Maynas"

target = target[
    [
        "week_start_date",
        "calendar_year",
        "calendar_month",
        "iso_year",
        "iso_week",
        "country",
        "adm1_name",
        "adm2_name",
        "case_definition",
        "dengue_cases",
    ]
]

if target["week_start_date"].duplicated().any():
    raise ValueError(
        "Duplicate week_start_date values remain in the processed target."
    )

# ============================================================
# 10. CREATE VALIDATION SUMMARY
# ============================================================

validation_summary = pd.DataFrame(
    {
        "metric": [
            "study_area",
            "spatial_resolution",
            "temporal_resolution",
            "first_week",
            "last_week",
            "observed_weeks",
            "expected_weeks",
            "missing_weeks",
            "coverage_pct",
            "duplicate_weeks",
            "missing_case_values",
            "negative_case_values",
            "zero_case_weeks",
            "total_cases",
            "mean_weekly_cases",
            "median_weekly_cases",
            "max_weekly_cases",
            "all_week_starts_sunday",
            "case_definitions",
        ],
        "value": [
            "Maynas, Loreto, Peru",
            "Admin2",
            "Week",
            target["week_start_date"].min().date().isoformat(),
            target["week_start_date"].max().date().isoformat(),
            len(observed_dates),
            len(expected_dates),
            len(missing_weeks),
            round(coverage_pct, 2),
            len(duplicate_weeks),
            missing_cases,
            negative_cases,
            int((target["dengue_cases"] == 0).sum()),
            int(target["dengue_cases"].sum()),
            round(target["dengue_cases"].mean(), 2),
            round(target["dengue_cases"].median(), 2),
            int(target["dengue_cases"].max()),
            all_sundays,
            " | ".join(case_definitions.index.astype(str)),
        ],
    }
)

# ============================================================
# 11. SAVE OUTPUTS
# ============================================================

target.to_csv(
    TARGET_FILE,
    index=False,
    date_format="%Y-%m-%d"
)

validation_summary.to_csv(
    VALIDATION_FILE,
    index=False
)

pd.DataFrame(
    {"missing_week_start_date": missing_weeks}
).to_csv(
    MISSING_WEEKS_FILE,
    index=False,
    date_format="%Y-%m-%d"
)

# ============================================================
# 12. FINAL REPORT
# ============================================================

print("\n" + "=" * 72)
print("TARGET PREPARATION COMPLETE")
print("=" * 72)

print("\nProcessed target:")
print(target.head())

print("\nTarget summary:")
print(target["dengue_cases"].describe())

print(f"\nSaved: {TARGET_FILE.resolve()}")
print(f"Saved: {VALIDATION_FILE.resolve()}")
print(f"Saved: {MISSING_WEEKS_FILE.resolve()}")

if len(missing_weeks) == 0:
    print(
        "\nResult: the Maynas target forms a complete weekly series "
        "across the observed period."
    )
else:
    print(
        "\nResult: missing weeks were identified. Review "
        "maynas_missing_weeks.csv before proceeding to climate-data joins."
    )