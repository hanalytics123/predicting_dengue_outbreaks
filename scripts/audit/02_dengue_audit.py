import pandas as pd
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = PROJECT_ROOT / "data" / "raw" / "dengue.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DENGUE DATA AUDIT")
print("=" * 70)
print(f"Project root: {PROJECT_ROOT}")
print(f"Input file:   {FILE_PATH}")
print(f"Output dir:   {OUTPUT_DIR.resolve()}")


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

df = pd.read_csv(FILE_PATH)

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. BASIC STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)
print(df.dtypes)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = (
    df.isna()
      .sum()
      .to_frame("missing_count")
      .assign(missing_pct=lambda x: x["missing_count"] / len(df) * 100)
      .sort_values("missing_pct", ascending=False)
)

print(missing)


# ============================================================
# 3. COUNTRY CHECK
# ============================================================

print("\n" + "=" * 70)
print("COUNTRY CHECK")
print("=" * 70)

print(df["adm_0_name"].value_counts(dropna=False))


# ============================================================
# 4. SPATIAL RESOLUTION
# ============================================================

print("\n" + "=" * 70)
print("SPATIAL RESOLUTION")
print("=" * 70)

print(df["S_res"].value_counts(dropna=False))

print("\nADM1 values:")
print(sorted(df["adm_1_name"].dropna().unique()))

print("\nNumber of unique ADM1 areas:")
print(df["adm_1_name"].nunique())

print("\nNumber of unique ADM2 areas:")
print(df["adm_2_name"].nunique())


# ============================================================
# 5. TEMPORAL RESOLUTION
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL RESOLUTION")
print("=" * 70)

print(df["T_res"].value_counts(dropna=False))


# ============================================================
# 6. PARSE DATES
# ============================================================

# Parse both date columns.
# errors="coerce" converts invalid dates to NaT so they can be audited.
df["calendar_start_date"] = pd.to_datetime(
    df["calendar_start_date"],
    errors="coerce"
)

df["calendar_end_date"] = pd.to_datetime(
    df["calendar_end_date"],
    errors="coerce"
)

print("\nOverall date range:")
print("Start:", df["calendar_start_date"].min())
print("End:  ", df["calendar_end_date"].max())

print("\nRows with invalid/missing start dates:")
print(df["calendar_start_date"].isna().sum())

print("\nRows with invalid/missing end dates:")
print(df["calendar_end_date"].isna().sum())

# Flag records where the reported end date is before the start date.
invalid_date_order = df[
    df["calendar_start_date"].notna()
    & df["calendar_end_date"].notna()
    & (df["calendar_end_date"] < df["calendar_start_date"])
].copy()

print("\nRows where end date is earlier than start date:")
print(len(invalid_date_order))


# ============================================================
# 7. CASE DEFINITIONS
# ============================================================

print("\n" + "=" * 70)
print("CASE DEFINITIONS")
print("=" * 70)

print(
    df["case_definition_standardised"]
    .value_counts(dropna=False)
)


# ============================================================
# 8. DENGUE TOTAL AUDIT
# ============================================================

print("\n" + "=" * 70)
print("DENGUE TOTAL AUDIT")
print("=" * 70)

print(df["dengue_total"].describe())

print("\nMissing dengue totals:")
print(df["dengue_total"].isna().sum())

print("\nNegative dengue totals:")
print((df["dengue_total"] < 0).sum())

print("\nZero dengue totals:")
print((df["dengue_total"] == 0).sum())


# ============================================================
# 9. EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("EXACT DUPLICATES")
print("=" * 70)

exact_duplicates = df.duplicated().sum()
print("Exact duplicate rows:", exact_duplicates)


# ============================================================
# 10. LOCATION-TIME DUPLICATES USING ORIGINAL KEY
# ============================================================

duplicate_key = [
    "FAO_GAUL_code",
    "calendar_start_date",
    "calendar_end_date",
    "T_res",
    "S_res"
]

possible_duplicates = df[
    df.duplicated(subset=duplicate_key, keep=False)
].sort_values(duplicate_key)

print("\nPotential duplicate location-period records using original key:")
print(len(possible_duplicates))

if len(possible_duplicates) > 0:
    print(
        possible_duplicates[
            duplicate_key
            + [
                "adm_1_name",
                "adm_2_name",
                "dengue_total",
                "case_definition_standardised",
                "UUID"
            ]
        ].head(30)
    )


# ============================================================
# 11. FAO GAUL CODE CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("FAO GAUL CODE CONSISTENCY")
print("=" * 70)

code_name_check = (
    df.groupby("FAO_GAUL_code")
      .agg(
          n_adm1=("adm_1_name", "nunique"),
          n_adm2=("adm_2_name", "nunique"),
          adm1_examples=(
              "adm_1_name",
              lambda x: ", ".join(
                  sorted(x.dropna().astype(str).unique())[:5]
              )
          ),
          adm2_examples=(
              "adm_2_name",
              lambda x: ", ".join(
                  sorted(x.dropna().astype(str).unique())[:5]
              )
          )
      )
      .reset_index()
)

problem_codes = code_name_check[
    (code_name_check["n_adm1"] > 1)
    | (code_name_check["n_adm2"] > 1)
]

print("Codes mapping to multiple names:")
print(problem_codes.head(30))


# ============================================================
# 12. ADM1 SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ADM1 SUMMARY")
print("=" * 70)

adm1 = df[
    (df["S_res"] == "Admin1")
    & (df["adm_1_name"].notna())
].copy()

adm1_summary = (
    adm1.groupby("adm_1_name")
        .agg(
            observations=("dengue_total", "size"),
            first_date=("calendar_start_date", "min"),
            last_date=("calendar_end_date", "max"),
            total_cases=("dengue_total", "sum"),
            mean_cases=("dengue_total", "mean"),
            median_cases=("dengue_total", "median"),
            max_cases=("dengue_total", "max"),
            zero_case_periods=("dengue_total", lambda x: (x == 0).sum()),
            missing_cases=("dengue_total", lambda x: x.isna().sum()),
            gaul_codes=("FAO_GAUL_code", "nunique"),
            temporal_resolutions=(
                "T_res",
                lambda x: ", ".join(
                    sorted(x.dropna().astype(str).unique())
                )
            )
        )
        .sort_values("observations", ascending=False)
)

adm1_summary["zero_case_pct"] = (
    adm1_summary["zero_case_periods"]
    / adm1_summary["observations"]
    * 100
)

print(adm1_summary)


# ============================================================
# 13. WEEKLY ADM1 DATA ONLY
# ============================================================

weekly_adm1 = df[
    (df["S_res"] == "Admin1")
    & (df["T_res"] == "Week")
    & (df["adm_1_name"].notna())
].copy()

print("\nWeekly ADM1 rows:", len(weekly_adm1))


# ============================================================
# 14. WEEKLY COVERAGE BY ADM1
# ============================================================

weekly_coverage = (
    weekly_adm1.groupby("adm_1_name")
               .agg(
                   observations=("calendar_start_date", "size"),
                   unique_weeks=("calendar_start_date", "nunique"),
                   first_week=("calendar_start_date", "min"),
                   last_week=("calendar_start_date", "max"),
                   total_cases=("dengue_total", "sum"),
                   mean_cases=("dengue_total", "mean"),
                   median_cases=("dengue_total", "median"),
                   max_cases=("dengue_total", "max")
               )
)

weekly_coverage["duplicate_week_rows"] = (
    weekly_coverage["observations"]
    - weekly_coverage["unique_weeks"]
)

weekly_coverage = weekly_coverage.sort_values(
    ["unique_weeks", "total_cases"],
    ascending=False
)

print("\nWeekly ADM1 coverage:")
print(weekly_coverage)


# ============================================================
# 15. IDENTIFY APPARENT MISSING WEEKS
# ============================================================

def audit_missing_weeks(group):
    """
    Compare observed start dates with a complete 7-day sequence.

    This is only a diagnostic. It assumes observations should occur
    at exact 7-day intervals from the first observed date.
    """

    dates = (
        group["calendar_start_date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(dates) < 2:
        return pd.Series({
            "expected_weeks": len(dates),
            "observed_weeks": len(dates),
            "missing_weeks": 0,
            "coverage_pct": 100.0
        })

    expected = pd.date_range(
        start=dates.min(),
        end=dates.max(),
        freq="7D"
    )

    missing_dates = expected.difference(dates)

    return pd.Series({
        "expected_weeks": len(expected),
        "observed_weeks": len(dates),
        "missing_weeks": len(missing_dates),
        "coverage_pct": len(dates) / len(expected) * 100
    })


continuity = (
    weekly_adm1.groupby("adm_1_name")
               .apply(audit_missing_weeks)
               .sort_values("coverage_pct", ascending=False)
)

print("\nWeekly continuity diagnostic:")
print(continuity)


# ============================================================
# 16. ADM2 COVERAGE WITHIN EACH ADM1
# ============================================================

adm2 = df[
    (df["S_res"] == "Admin2")
    & (df["adm_1_name"].notna())
    & (df["adm_2_name"].notna())
].copy()

adm2_summary = (
    adm2.groupby("adm_1_name")
        .agg(
            adm2_areas=("adm_2_name", "nunique"),
            observations=("dengue_total", "size"),
            first_date=("calendar_start_date", "min"),
            last_date=("calendar_end_date", "max"),
            total_cases=("dengue_total", "sum")
        )
        .sort_values("observations", ascending=False)
)

print("\nADM2 coverage within each ADM1:")
print(adm2_summary)


# ============================================================
# 17. ADM1 / ADM2 PERIOD OVERLAP
# ============================================================

adm1_periods = (
    df.loc[
        df["S_res"] == "Admin1",
        ["adm_1_name", "calendar_start_date"]
    ]
    .drop_duplicates()
)

adm2_periods = (
    df.loc[
        df["S_res"] == "Admin2",
        ["adm_1_name", "calendar_start_date"]
    ]
    .drop_duplicates()
)

overlap = adm1_periods.merge(
    adm2_periods,
    on=["adm_1_name", "calendar_start_date"],
    how="inner"
)

overlap_summary = (
    overlap.groupby("adm_1_name")
           .size()
           .sort_values(ascending=False)
           .rename("periods_with_both_adm1_and_adm2")
)

print("\nPeriods containing BOTH ADM1 and ADM2 records:")
print(overlap_summary)


# ============================================================
# 18. LORETO ADM1 CHECK
# ============================================================

loreto_adm1 = weekly_adm1[
    weekly_adm1["adm_1_name"].str.upper() == "LORETO"
].copy()

print("\n" + "=" * 70)
print("LORETO ADM1 CHECK")
print("=" * 70)

print("Rows:", len(loreto_adm1))

if len(loreto_adm1) > 0:
    print("First week:", loreto_adm1["calendar_start_date"].min())
    print("Last week: ", loreto_adm1["calendar_start_date"].max())

    print("\nDengue totals:")
    print(loreto_adm1["dengue_total"].describe())

    print("\nFirst 10 records:")
    print(
        loreto_adm1[
            [
                "calendar_start_date",
                "calendar_end_date",
                "dengue_total",
                "FAO_GAUL_code",
                "UUID"
            ]
        ]
        .sort_values("calendar_start_date")
        .head(10)
    )


# ============================================================
# 19. COMBINE KEY ADM1 QUALITY METRICS
# ============================================================

region_audit = (
    weekly_coverage
    .join(continuity, rsuffix="_continuity")
    .join(adm2_summary[["adm2_areas"]], how="left")
)

region_audit = region_audit.sort_values(
    ["coverage_pct", "unique_weeks", "total_cases"],
    ascending=[False, False, False]
)

print("\n" + "=" * 70)
print("FINAL REGION COMPARISON")
print("=" * 70)

print(region_audit)


# ============================================================
# 20. LORETO ADM2 DETAILED AUDIT
# ============================================================

print("\n" + "=" * 70)
print("LORETO ADM2 DETAILED AUDIT")
print("=" * 70)

loreto_adm2 = df[
    (df["adm_1_name"].str.upper() == "LORETO")
    & (df["S_res"] == "Admin2")
    & (df["T_res"] == "Week")
    & (df["adm_2_name"].notna())
].copy()


# ------------------------------------------------------------
# 20.1 ADM2 areas present
# ------------------------------------------------------------

print("\nLoreto ADM2 areas:")

print(
    sorted(
        loreto_adm2["adm_2_name"]
        .dropna()
        .unique()
    )
)


# ------------------------------------------------------------
# 20.2 Coverage and case volume for each ADM2
# ------------------------------------------------------------

loreto_adm2_summary = (
    loreto_adm2
    .groupby("adm_2_name")
    .agg(
        observations=("calendar_start_date", "size"),
        unique_weeks=("calendar_start_date", "nunique"),
        first_week=("calendar_start_date", "min"),
        last_week=("calendar_start_date", "max"),
        total_cases=("dengue_total", "sum"),
        mean_cases=("dengue_total", "mean"),
        median_cases=("dengue_total", "median"),
        max_cases=("dengue_total", "max"),
        zero_weeks=("dengue_total", lambda x: (x == 0).sum()),
        gaul_codes=("FAO_GAUL_code", "nunique")
    )
    .sort_values("unique_weeks", ascending=False)
)

loreto_adm2_summary["zero_pct"] = (
    loreto_adm2_summary["zero_weeks"]
    / loreto_adm2_summary["observations"]
    * 100
)

print("\nLoreto ADM2 summary:")
print(loreto_adm2_summary)


# ------------------------------------------------------------
# 20.3 True duplicate check using ADM1 + ADM2 + start date
# ------------------------------------------------------------

adm2_key = [
    "adm_1_name",
    "adm_2_name",
    "calendar_start_date"
]

true_duplicates = loreto_adm2[
    loreto_adm2.duplicated(
        subset=adm2_key,
        keep=False
    )
].sort_values(adm2_key)

print("\nPotential true Loreto ADM2 duplicates:")
print(len(true_duplicates))

if len(true_duplicates) > 0:
    print(
        true_duplicates[
            adm2_key
            + [
                "calendar_end_date",
                "dengue_total",
                "FAO_GAUL_code",
                "UUID"
            ]
        ].head(30)
    )


# ------------------------------------------------------------
# 20.4 Number of reporting ADM2 areas per week
# ------------------------------------------------------------

weekly_reporting = (
    loreto_adm2
    .groupby("calendar_start_date")
    .agg(
        reporting_adm2=("adm_2_name", "nunique"),
        weekly_cases=("dengue_total", "sum")
    )
    .sort_index()
)

print("\nReporting ADM2 areas per week:")
print(
    weekly_reporting["reporting_adm2"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 20.5 Coverage by year and ADM2
# ------------------------------------------------------------

loreto_adm2["year"] = loreto_adm2[
    "calendar_start_date"
].dt.year

coverage_by_year = (
    loreto_adm2
    .groupby(["year", "adm_2_name"])
    .agg(
        reported_weeks=("calendar_start_date", "nunique"),
        cases=("dengue_total", "sum")
    )
    .reset_index()
)

coverage_pivot = coverage_by_year.pivot(
    index="year",
    columns="adm_2_name",
    values="reported_weeks"
).fillna(0)

print("\nNumber of reported weeks by year and ADM2:")
print(coverage_pivot)


# ------------------------------------------------------------
# 20.6 Number of reporting areas per year
# ------------------------------------------------------------

areas_per_year = (
    loreto_adm2
    .groupby("year")["adm_2_name"]
    .nunique()
    .rename("reporting_adm2_areas")
)

print("\nNumber of Loreto ADM2 areas reporting each year:")
print(areas_per_year)


# ------------------------------------------------------------
# 20.7 Maynas-specific audit
# ------------------------------------------------------------

maynas = loreto_adm2[
    loreto_adm2["adm_2_name"].str.upper() == "MAYNAS"
].copy()

print("\n" + "=" * 70)
print("MAYNAS CHECK")
print("=" * 70)

print("Rows:", len(maynas))

if len(maynas) > 0:
    maynas_dates = (
        maynas["calendar_start_date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    print("Unique weeks:", len(maynas_dates))
    print("First week:  ", maynas_dates.min())
    print("Last week:   ", maynas_dates.max())

    print("\nDengue totals:")
    print(maynas["dengue_total"].describe())

    print("\nZero-case weeks:")
    print((maynas["dengue_total"] == 0).sum())

    # Check duplicate Maynas records for the same week.
    maynas_duplicates = maynas[
        maynas.duplicated(
            subset=["adm_2_name", "calendar_start_date"],
            keep=False
        )
    ].sort_values("calendar_start_date")

    print("\nPotential duplicate Maynas weeks:")
    print(len(maynas_duplicates))


# ============================================================
# 21. EXPORT ALL AUDIT TABLES
# ============================================================

missing.to_csv(
    OUTPUT_DIR / "missing_values.csv"
)

adm1_summary.to_csv(
    OUTPUT_DIR / "adm1_summary.csv"
)

weekly_coverage.to_csv(
    OUTPUT_DIR / "adm1_weekly_coverage.csv"
)

continuity.to_csv(
    OUTPUT_DIR / "adm1_weekly_continuity.csv"
)

adm2_summary.to_csv(
    OUTPUT_DIR / "adm2_summary.csv"
)

region_audit.to_csv(
    OUTPUT_DIR / "region_selection_audit.csv"
)

possible_duplicates.to_csv(
    OUTPUT_DIR / "potential_duplicate_records.csv",
    index=False
)

problem_codes.to_csv(
    OUTPUT_DIR / "gaul_code_mapping_issues.csv",
    index=False
)

invalid_date_order.to_csv(
    OUTPUT_DIR / "invalid_date_order_records.csv",
    index=False
)

loreto_adm2_summary.to_csv(
    OUTPUT_DIR / "loreto_adm2_summary.csv"
)

weekly_reporting.to_csv(
    OUTPUT_DIR / "loreto_weekly_reporting_coverage.csv"
)

coverage_pivot.to_csv(
    OUTPUT_DIR / "loreto_adm2_coverage_by_year.csv"
)

coverage_by_year.to_csv(
    OUTPUT_DIR / "loreto_adm2_coverage_long.csv",
    index=False
)

areas_per_year.to_csv(
    OUTPUT_DIR / "loreto_reporting_areas_by_year.csv"
)

true_duplicates.to_csv(
    OUTPUT_DIR / "loreto_true_duplicate_records.csv",
    index=False
)

if len(maynas) > 0:
    maynas.to_csv(
        OUTPUT_DIR / "maynas_weekly_records.csv",
        index=False
    )

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
