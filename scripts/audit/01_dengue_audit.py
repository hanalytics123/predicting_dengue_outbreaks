import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD RAW DATA
# ============================================================

FILE_PATH = "data/raw/dengue.csv"

df = pd.read_csv(FILE_PATH)

print("Shape:", df.shape)
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
# 3. CHECK COUNTRY
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


# ============================================================
# 7. CASE DEFINITION
# ============================================================

print("\n" + "=" * 70)
print("CASE DEFINITIONS")
print("=" * 70)

print(
    df["case_definition_standardised"]
    .value_counts(dropna=False)
)


# ============================================================
# 8. DENGUE CASE VALUES
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
# 9. CHECK FOR EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("EXACT DUPLICATES")
print("=" * 70)

exact_duplicates = df.duplicated().sum()
print("Exact duplicate rows:", exact_duplicates)


# ============================================================
# 10. CHECK POSSIBLE LOCATION-TIME DUPLICATES
# ============================================================

# These are more important than exact duplicate rows.
# If the same geographic unit and reporting period appears more
# than once, investigate WHY before aggregating.

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

print("\nPotential duplicate location-period records:")
print(len(possible_duplicates))

print(possible_duplicates[
    duplicate_key
    + [
        "adm_1_name",
        "adm_2_name",
        "dengue_total",
        "case_definition_standardised",
        "UUID"
    ]
].head(30))


# ============================================================
# 11. CHECK FAO GAUL CODE CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("FAO GAUL CODE CONSISTENCY")
print("=" * 70)

# Does one code map to multiple geographic names?

code_name_check = (
    df.groupby("FAO_GAUL_code")
      .agg(
          n_adm1=("adm_1_name", "nunique"),
          n_adm2=("adm_2_name", "nunique"),
          adm1_examples=("adm_1_name",
                         lambda x: ", ".join(
                             sorted(x.dropna().astype(str).unique())[:5]
                         )),
          adm2_examples=("adm_2_name",
                         lambda x: ", ".join(
                             sorted(x.dropna().astype(str).unique())[:5]
                         ))
      )
      .reset_index()
)

problem_codes = code_name_check[
    (code_name_check["n_adm1"] > 1) |
    (code_name_check["n_adm2"] > 1)
]

print("Codes mapping to multiple names:")
print(problem_codes.head(30))


# ============================================================
# 12. ADM1-LEVEL AUDIT
# ============================================================

print("\n" + "=" * 70)
print("ADM1 SUMMARY")
print("=" * 70)

adm1 = df[
    (df["S_res"] == "Admin1") &
    (df["adm_1_name"].notna())
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
            zero_case_periods=("dengue_total",
                               lambda x: (x == 0).sum()),
            missing_cases=("dengue_total",
                           lambda x: x.isna().sum()),
            gaul_codes=("FAO_GAUL_code", "nunique"),
            temporal_resolutions=("T_res",
                                  lambda x: ", ".join(
                                      sorted(x.dropna().astype(str).unique())
                                  ))
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
    (df["S_res"] == "Admin1") &
    (df["T_res"] == "Week") &
    (df["adm_1_name"].notna())
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
# 15. IDENTIFY MISSING WEEKS
# ============================================================

def audit_missing_weeks(group):
    """
    Compare observed dates with a complete 7-day sequence between
    the first and final observations.

    This assumes observations should occur every 7 days.
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

    missing = expected.difference(dates)

    return pd.Series({
        "expected_weeks": len(expected),
        "observed_weeks": len(dates),
        "missing_weeks": len(missing),
        "coverage_pct": len(dates) / len(expected) * 100
    })


continuity = (
    weekly_adm1.groupby("adm_1_name")
               .apply(audit_missing_weeks)
               .sort_values("coverage_pct", ascending=False)
)

print("\nWeekly continuity:")
print(continuity)


# ============================================================
# 16. ADM2 COVERAGE WITHIN EACH ADM1
# ============================================================

adm2 = df[
    (df["S_res"] == "Admin2") &
    (df["adm_1_name"].notna()) &
    (df["adm_2_name"].notna())
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
# 17. CHECK WHETHER ADM1 AND ADM2 COEXIST FOR SAME PERIODS
# ============================================================

# Important:
# If ADM1 totals and constituent ADM2 values exist for the same
# dates, do NOT sum both levels together.

adm1_periods = (
    df.loc[df["S_res"] == "Admin1",
           ["adm_1_name", "calendar_start_date"]]
      .drop_duplicates()
)

adm2_periods = (
    df.loc[df["S_res"] == "Admin2",
           ["adm_1_name", "calendar_start_date"]]
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
# 18. SPECIAL CHECK: LORETO
# ============================================================

loreto = weekly_adm1[
    weekly_adm1["adm_1_name"].str.upper() == "LORETO"
].copy()

print("\n" + "=" * 70)
print("LORETO CHECK")
print("=" * 70)

print("Rows:", len(loreto))

if len(loreto) > 0:
    print("First week:", loreto["calendar_start_date"].min())
    print("Last week:", loreto["calendar_start_date"].max())

    print("\nDengue totals:")
    print(loreto["dengue_total"].describe())

    print("\nFirst 10 records:")
    print(
        loreto[
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
# 20. EXPORT AUDIT TABLES
# ============================================================

OUTPUT_DIR = "outputs/audit"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

missing.to_csv(
    f"{OUTPUT_DIR}/missing_values.csv"
)

adm1_summary.to_csv(
    f"{OUTPUT_DIR}/adm1_summary.csv"
)

weekly_coverage.to_csv(
    f"{OUTPUT_DIR}/adm1_weekly_coverage.csv"
)

continuity.to_csv(
    f"{OUTPUT_DIR}/adm1_weekly_continuity.csv"
)

adm2_summary.to_csv(
    f"{OUTPUT_DIR}/adm2_summary.csv"
)

region_audit.to_csv(
    f"{OUTPUT_DIR}/region_selection_audit.csv"
)

possible_duplicates.to_csv(
    f"{OUTPUT_DIR}/potential_duplicate_records.csv",
    index=False
)

print("\nAudit complete.")
print(f"Outputs saved to: {OUTPUT_DIR}")