from pathlib import Path
import pandas as pd
import numpy as np


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
    / "summary"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "modelling_dataset_summary.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Modelling dataset not found."
    )

df = pd.read_csv(INPUT_FILE)

df["week_start_date"] = pd.to_datetime(
    df["week_start_date"],
    errors="coerce"
)

date_columns = [
    column
    for column in df.columns
    if column.endswith("_date")
]

for column in date_columns:
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )


# ============================================================
# VARIABLE GROUPS
# ============================================================

groups = {
    "Dengue target and identifiers": [
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
    ],

    "PERSIANN precipitation": [
        column
        for column in df.columns
        if (
            column.startswith("precip_")
            or column == "rain_days"
        )
    ],

    "CFS temperature / humidity": [
        column
        for column in df.columns
        if any(
            token in column
            for token in [
                "temperature_c_",
                "dewpoint_c_",
                "relative_humidity_pct_",
                "specific_humidity_kgkg_",
                "temp_humidity_",
            ]
        )
    ],

    "CFS wind": [
        column
        for column in df.columns
        if any(
            token in column
            for token in [
                "u_wind_ms_",
                "v_wind_ms_",
                "wind_speed_ms_",
                "wind_",
            ]
        )
        and not column.startswith("has_")
    ],

    "CFS pressure": [
        column
        for column in df.columns
        if (
            column.startswith("surface_pressure_")
            or column.startswith("pressure_")
        )
    ],

    "MODIS NDVI": [
        column
        for column in df.columns
        if (
            column.startswith("ndvi_")
            or column in {
                "valid_pixel_count",
                "valid_pixel_fraction_of_max",
                "low_valid_pixel_coverage",
                "modis_image_id",
            }
        )
    ],

    "Source availability flags": [
        column
        for column in df.columns
        if column.startswith("has_")
    ],
}


# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.extend([
    "=" * 78,
    "MAYNAS DENGUE MODELLING DATASET SUMMARY",
    "=" * 78,
    "",
    "PURPOSE",
    "-" * 78,
    (
        "Integrated weekly modelling dataset for predicting dengue in "
        "Maynas, Loreto, Peru using climate and environmental predictors."
    ),
    (
        "Each row represents one observed dengue surveillance week. "
        "Dengue is the master table; environmental datasets were left-joined "
        "on week_start_date."
    ),
    "",
    "TEMPORAL DEFINITION",
    "-" * 78,
    "Weekly convention: Sunday-Saturday",
    "week_start_date: Sunday identifying each modelling week",
    f"First observed dengue week: {df['week_start_date'].min().date()}",
    f"Last observed dengue week: {df['week_start_date'].max().date()}",
    f"Observed dengue weeks / modelling rows: {len(df):,}",
    (
        "Calendar weeks in full study period: 1,252. Four known dengue "
        "surveillance weeks are absent in 2000 and are not treated as "
        "zero-case observations."
    ),
    "Known missing dengue weeks:",
    "  2000-04-30",
    "  2000-06-11",
    "  2000-06-25",
    "  2000-07-09",
    "",
    "DATASET DIMENSIONS",
    "-" * 78,
    f"Rows: {len(df):,}",
    f"Columns: {len(df.columns):,}",
    f"Duplicate week_start_date values: {int(df['week_start_date'].duplicated().sum()):,}",
    f"Duplicate full rows: {int(df.duplicated().sum()):,}",
    "",
    "TARGET",
    "-" * 78,
    "Target column: dengue_cases",
    "Case definition: Probable and confirmed",
    f"Target observations: {df['dengue_cases'].count():,}",
    f"Missing target values: {df['dengue_cases'].isna().sum():,}",
    f"Zero-case weeks: {(df['dengue_cases'] == 0).sum():,}",
    f"Minimum weekly cases: {df['dengue_cases'].min():.0f}",
    f"Mean weekly cases: {df['dengue_cases'].mean():.2f}",
    f"Median weekly cases: {df['dengue_cases'].median():.2f}",
    f"Standard deviation: {df['dengue_cases'].std():.2f}",
    f"Maximum weekly cases: {df['dengue_cases'].max():.0f}",
    f"Skewness: {df['dengue_cases'].skew():.2f}",
    "",
    "Target percentiles:",
])

for q in [
    0.01, 0.05, 0.10, 0.25, 0.50,
    0.75, 0.90, 0.95, 0.99
]:
    summary.append(
        f"  {q:.0%}: {df['dengue_cases'].quantile(q):.2f}"
    )


summary.extend([
    "",
    "SOURCE DATASETS",
    "-" * 78,
    (
        "Dengue surveillance: weekly probable and confirmed cases for "
        "Maynas, Loreto, Peru."
    ),
    (
        "PERSIANN-CDR: precipitation spatially aggregated to Maynas and "
        "summarised to Sunday-Saturday weeks."
    ),
    (
        "CFSR / CFSv2: 6-hour forecast climate variables on a 0.5-degree "
        "grid, spatially aggregated using grid-cell centres within the "
        "GAUL 2024 Maynas polygon."
    ),
    (
        "MODIS MOD13Q1: vegetation/NDVI observations aligned to weekly "
        "rows using the most recent observation available on or before "
        "the week end date."
    ),
    "",
    "CFS VARIABLES",
    "-" * 78,
    "Temperature: degrees Celsius",
    "Dew point: degrees Celsius",
    "Relative humidity: percent",
    "Specific humidity: kg/kg",
    "U wind component: metres per second",
    "V wind component: metres per second",
    "Wind speed: metres per second",
    "Surface pressure: hPa",
    "",
    "WIND-SPEED DEFINITION",
    "-" * 78,
    (
        "Wind speed is the magnitude of the Maynas-level mean wind vector "
        "at each 6-hour timestamp, calculated as sqrt(u^2 + v^2), before "
        "daily/weekly temporal aggregation."
    ),
    "",
    "PRESSURE STREAM SELECTION",
    "-" * 78,
    "Selected CFS pressure stream: pgbh / pgrbh",
    "Excluded alternative stream: ipvh / ipvgrbh",
    (
        "The pgbh/pgrbh stream was retained to provide a continuous "
        "pressure-product lineage across the study period. Representative "
        "comparisons showed negligible differences in Maynas-level spatial "
        "mean pressure between the parallel streams."
    ),
    "",
    "NDVI ALIGNMENT",
    "-" * 78,
    (
        "MOD13Q1 observations are not interpolated using future satellite "
        "observations. Each weekly row receives the most recent NDVI "
        "observation available on or before the Saturday week end."
    ),
    "Maximum NDVI carry-forward age: 32 days",
    (
        "The first six modelling rows pre-date the first MODIS observation "
        "and therefore retain missing NDVI values."
    ),
    "",
    "MISSING DATA",
    "-" * 78,
])

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    summary.append("No missing values.")
else:
    for column, count in missing.items():
        summary.append(
            f"{column}: {int(count):,} "
            f"({count / len(df) * 100:.2f}%)"
        )


summary.extend([
    "",
    "DATA QUALITY / QA",
    "-" * 78,
])

qa_fields = [
    "precip_week_complete",
    "temp_humidity_week_complete",
    "wind_week_complete",
    "pressure_week_complete",
    "low_valid_pixel_coverage",
    "ndvi_available",
    "ndvi_within_age_limit",
    "ndvi_stale",
    "has_precipitation_data",
    "has_temp_humidity_data",
    "has_wind_data",
    "has_pressure_data",
    "has_ndvi_data",
]

for column in qa_fields:
    if column in df.columns:
        counts = df[column].value_counts(dropna=False)
        summary.append(f"{column}:")
        for value, count in counts.items():
            summary.append(
                f"  {value}: {count:,}"
            )


summary.extend([
    "",
    "VARIABLE INVENTORY",
    "-" * 78,
])

already_listed = set()

for group_name, columns in groups.items():
    columns = [
        column
        for column in columns
        if column in df.columns
        and column not in already_listed
    ]

    if not columns:
        continue

    summary.append("")
    summary.append(
        f"{group_name} ({len(columns)} columns):"
    )

    for column in columns:
        summary.append(
            f"  {column}"
        )
        already_listed.add(column)

remaining_columns = [
    column
    for column in df.columns
    if column not in already_listed
]

if remaining_columns:
    summary.append("")
    summary.append(
        f"Other retained fields ({len(remaining_columns)} columns):"
    )
    for column in remaining_columns:
        summary.append(
            f"  {column}"
        )


summary.extend([
    "",
    "KNOWN REDUNDANCY / HIGH CORRELATIONS FROM AUDIT",
    "-" * 78,
    "precip_days_observed <-> precip_mean_valid_grid_cells: r = 1.0000",
    "valid_pixel_count <-> valid_pixel_fraction_of_max: r = 1.0000",
    "precip_mean_valid_grid_cells <-> precip_missing_days: r = -1.0000",
    "precip_days_observed <-> precip_missing_days: r = -1.0000",
    "precip_sum_mm <-> precip_mean_daily_mm: r = 0.9997",
    "dewpoint_c_mean <-> specific_humidity_kgkg_mean: r = 0.9978",
    "dewpoint_c_max <-> specific_humidity_kgkg_max: r = 0.9977",
    "dewpoint_c_min <-> specific_humidity_kgkg_min: r = 0.9965",
    "surface_pressure_hpa_mean <-> surface_pressure_hpa_max: r = 0.9361",
    "surface_pressure_hpa_mean <-> surface_pressure_hpa_min: r = 0.9287",
    "temperature_c_max <-> relative_humidity_pct_min: r = -0.9115",
    (
        "These relationships are documented for EDA/feature-selection "
        "purposes. No variables have yet been removed on this basis."
    ),
    "",
    "CURRENT PROCESSING STATUS",
    "-" * 78,
    (
        "The integrated dataset has passed structural, temporal, missingness, "
        "infinite-value and broad plausibility checks."
    ),
    (
        "No feature selection, target transformation, lag engineering, "
        "imputation or modelling has yet been applied."
    ),
    (
        "The next analytical stage is exploratory data analysis, followed "
        "by evidence-based feature engineering and time-aware modelling."
    ),
])


summary_text = "\n".join(summary)

OUTPUT_FILE.write_text(
    summary_text,
    encoding="utf-8"
)

print(summary_text)

print()
print("=" * 78)
print("SUMMARY CREATED")
print("=" * 78)
print(
    "Output: outputs/summary/modelling_dataset_summary.txt"
)
