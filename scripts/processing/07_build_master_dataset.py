from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DENGUE_FILE = PROJECT_ROOT / "data" / "processed" / "dengue" / "maynas_dengue_weekly.csv"
PRECIP_FILE = PROJECT_ROOT / "data" / "processed" / "persiann" / "maynas_precipitation_weekly.csv"
TEMP_HUMIDITY_FILE = PROJECT_ROOT / "data" / "processed" / "cfs" / "maynas_cfs_temp_humidity_weekly.csv"
WIND_FILE = PROJECT_ROOT / "data" / "processed" / "cfs" / "maynas_cfs_wind_weekly.csv"
PRESSURE_FILE = PROJECT_ROOT / "data" / "processed" / "cfs" / "maynas_cfs_pressure_weekly.csv"
NDVI_FILE = PROJECT_ROOT / "data" / "processed" / "ndvi" / "maynas_mod13q1_ndvi_weekly.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "master"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_OUTPUT = OUTPUT_DIR / "maynas_dengue_master_dataset.csv"
SUMMARY_OUTPUT = AUDIT_DIR / "master_dataset_integration_summary.txt"
JOIN_AUDIT_OUTPUT = AUDIT_DIR / "master_dataset_join_audit.csv"

JOIN_KEY = "week_start_date"
EXPECTED_DENGUE_ROWS = 1248


def load_weekly_dataset(name, path):
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found:\n{path}")

    df = pd.read_csv(path)

    if JOIN_KEY not in df.columns:
        raise ValueError(f"{name}: required join key '{JOIN_KEY}' not found.")

    df[JOIN_KEY] = pd.to_datetime(df[JOIN_KEY], errors="coerce")

    if df[JOIN_KEY].isna().any():
        raise ValueError(
            f"{name}: one or more {JOIN_KEY} values could not be parsed."
        )

    duplicate_count = int(df[JOIN_KEY].duplicated().sum())

    if duplicate_count > 0:
        raise ValueError(
            f"{name}: found {duplicate_count:,} duplicate {JOIN_KEY} values."
        )

    return df.sort_values(JOIN_KEY).reset_index(drop=True)


def rename_environment_columns(df, source):
    if source == "precipitation":
        rename_map = {
            "calendar_days_present": "precip_calendar_days_present",
            "mean_valid_grid_cells": "precip_mean_valid_grid_cells",
            "rain_day_threshold_mm": "precip_rain_day_threshold_mm",
            "missing_precip_days": "precip_missing_days",
            "week_complete": "precip_week_complete",
        }

    elif source == "temp_humidity":
        rename_map = {
            "week_end_date": "temp_humidity_week_end_date",
            "observations_6hourly": "temp_humidity_observations_6hourly",
            "complete_week": "temp_humidity_week_complete",
        }

    elif source == "wind":
        rename_map = {
            "week_end_date": "wind_week_end_date",
            "observations_6hourly": "wind_observations_6hourly",
            "complete_week": "wind_week_complete",
        }

    elif source == "pressure":
        rename_map = {
            "week_end_date": "pressure_week_end_date",
            "observations_6hourly": "pressure_observations_6hourly",
            "complete_week": "pressure_week_complete",
        }

    elif source == "ndvi":
        rename_map = {
            "week_end_date": "ndvi_week_end_date",
        }

    else:
        rename_map = {}

    rename_map = {
        old: new
        for old, new in rename_map.items()
        if old in df.columns
    }

    return df.rename(columns=rename_map)


def source_match_stats(master_weeks, source_name, source_df):
    source_weeks = pd.DatetimeIndex(source_df[JOIN_KEY].unique())
    matched = master_weeks.intersection(source_weeks)
    missing = master_weeks.difference(source_weeks)

    return {
        "source": source_name,
        "source_rows": len(source_df),
        "matched_dengue_weeks": len(matched),
        "unmatched_dengue_weeks": len(missing),
        "first_source_week": source_df[JOIN_KEY].min(),
        "last_source_week": source_df[JOIN_KEY].max(),
        "first_unmatched_dengue_week": missing.min() if len(missing) else pd.NaT,
        "last_unmatched_dengue_week": missing.max() if len(missing) else pd.NaT,
    }


print("=" * 78)
print("BUILD MAYNAS DENGUE MASTER DATASET")
print("=" * 78)

dengue = load_weekly_dataset("dengue", DENGUE_FILE)
precipitation = rename_environment_columns(
    load_weekly_dataset("precipitation", PRECIP_FILE),
    "precipitation",
)
temp_humidity = rename_environment_columns(
    load_weekly_dataset("temp_humidity", TEMP_HUMIDITY_FILE),
    "temp_humidity",
)
wind = rename_environment_columns(
    load_weekly_dataset("wind", WIND_FILE),
    "wind",
)
pressure = rename_environment_columns(
    load_weekly_dataset("pressure", PRESSURE_FILE),
    "pressure",
)
ndvi = rename_environment_columns(
    load_weekly_dataset("ndvi", NDVI_FILE),
    "ndvi",
)

if len(dengue) != EXPECTED_DENGUE_ROWS:
    print(
        "WARNING: dengue master table contains "
        f"{len(dengue):,} rows; expected {EXPECTED_DENGUE_ROWS:,}."
    )

master_weeks = pd.DatetimeIndex(dengue[JOIN_KEY].unique())

join_audit_rows = []

for source_name, source_df in [
    ("precipitation", precipitation),
    ("temp_humidity", temp_humidity),
    ("wind", wind),
    ("pressure", pressure),
    ("ndvi", ndvi),
]:
    join_audit_rows.append(
        source_match_stats(
            master_weeks,
            source_name,
            source_df,
        )
    )

join_audit = pd.DataFrame(join_audit_rows)
join_audit.to_csv(JOIN_AUDIT_OUTPUT, index=False)

model = dengue.copy()

for source_name, source_df in [
    ("precipitation", precipitation),
    ("temp_humidity", temp_humidity),
    ("wind", wind),
    ("pressure", pressure),
    ("ndvi", ndvi),
]:
    rows_before = len(model)

    model = model.merge(
        source_df,
        on=JOIN_KEY,
        how="left",
        validate="one_to_one",
    )

    if len(model) != rows_before:
        raise ValueError(
            f"{source_name}: row count changed after left join."
        )

if len(model) != len(dengue):
    raise ValueError(
        "Integrated modelling dataset row count does not match dengue master."
    )

if model[JOIN_KEY].duplicated().any():
    raise ValueError(
        "Integrated modelling dataset contains duplicate weeks."
    )

if not model[JOIN_KEY].equals(dengue[JOIN_KEY]):
    raise ValueError(
        "Integrated modelling dataset no longer preserves dengue week order."
    )

anchors = {
    "has_precipitation_data": "precip_sum_mm",
    "has_temp_humidity_data": "temperature_c_mean",
    "has_wind_data": "wind_speed_ms_mean",
    "has_pressure_data": "surface_pressure_hpa_mean",
    "has_ndvi_data": "ndvi_mean_weekly",
}

for flag, anchor in anchors.items():
    if anchor in model.columns:
        model[flag] = model[anchor].notna()

preferred_dengue_columns = [
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

ordered_columns = [
    column
    for column in preferred_dengue_columns
    if column in model.columns
]

ordered_columns.extend(
    column
    for column in model.columns
    if column not in ordered_columns
)

model = model[ordered_columns]

model.to_csv(
    MASTER_OUTPUT,
    index=False,
)

summary = [
    "=" * 78,
    "MAYNAS DENGUE MASTER DATASET INTEGRATION SUMMARY",
    "=" * 78,
    "",
    "MASTER TABLE",
    "-" * 78,
    "Master dataset: dengue",
    f"Dengue prediction observations: {len(dengue):,}",
    f"First observed dengue week: {dengue[JOIN_KEY].min()}",
    f"Last observed dengue week: {dengue[JOIN_KEY].max()}",
    (
        "Known dengue surveillance gaps: 4 weeks in 2000; "
        "these are not manufactured as zero-case observations."
    ),
    "",
    "JOIN COVERAGE",
    "-" * 78,
]

for _, row in join_audit.iterrows():
    summary.append(
        f"{row['source']}: "
        f"{int(row['matched_dengue_weeks']):,} / {len(dengue):,} "
        f"dengue weeks matched; "
        f"{int(row['unmatched_dengue_weeks']):,} unmatched"
    )

summary.extend([
    "",
    "POST-JOIN VALIDATION",
    "-" * 78,
    f"Integrated rows: {len(model):,}",
    f"Integrated columns: {len(model.columns):,}",
    (
        "Duplicate week_start_date values: "
        f"{int(model[JOIN_KEY].duplicated().sum()):,}"
    ),
    (
        "Row count preserved from dengue master table: "
        f"{len(model) == len(dengue)}"
    ),
    "",
    "MISSING VALUES AFTER INTEGRATION",
    "-" * 78,
])

missing_counts = model.isna().sum()

missing_items = (
    missing_counts[missing_counts > 0]
    .sort_values(ascending=False)
)

if missing_items.empty:
    summary.append("No missing values.")
else:
    for column, count in missing_items.items():
        summary.append(
            f"{column}: {int(count):,}"
        )

summary.extend([
    "",
    "SOURCE AVAILABILITY FLAGS",
    "-" * 78,
])

for column in anchors:
    if column in model.columns:
        summary.append(
            f"{column}: {int(model[column].sum()):,} / "
            f"{len(model):,} rows available"
        )

summary.extend([
    "",
    "INTEGRATION METHOD",
    "-" * 78,
    (
        "Dengue is the master table because each dengue row defines "
        "a prediction observation. PERSIANN precipitation, CFS "
        "temperature/humidity, CFS wind, CFS surface pressure and "
        "MODIS NDVI are left-joined on week_start_date."
    ),
    (
        "All environmental candidate predictors and QA/completeness "
        "fields are retained at this stage. No feature selection or "
        "dropping of QA fields is performed during integration."
    ),
    "",
    f"Output dataset: {MASTER_OUTPUT}",
    f"Join audit: {JOIN_AUDIT_OUTPUT}",
])

summary_text = "\n".join(summary)
SUMMARY_OUTPUT.write_text(summary_text, encoding="utf-8")

print()
print(summary_text)

print()
print("=" * 78)
print("INTEGRATION COMPLETE")
print("=" * 78)
print(f"Master dataset: {MASTER_OUTPUT}")
print(f"Join audit: {JOIN_AUDIT_OUTPUT}")
print(f"Summary: {SUMMARY_OUTPUT}")
