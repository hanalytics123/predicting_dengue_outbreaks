import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PRECIP_DIR = PROJECT_ROOT / "data" / "raw" / "precipitation"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_OUTPUT = PROCESSED_DIR / "maynas_precipitation_daily.csv"
WEEKLY_OUTPUT = PROCESSED_DIR / "maynas_precipitation_weekly.csv"
VALIDATION_OUTPUT = AUDIT_DIR / "maynas_precipitation_validation.csv"
MISSING_DATES_OUTPUT = AUDIT_DIR / "maynas_precipitation_missing_dates.csv"
NO_VALID_CELLS_OUTPUT = AUDIT_DIR / "maynas_precipitation_no_valid_cells.csv"

ZIP_FILES = [
    RAW_PRECIP_DIR / "maynas_persiann_cdr_2000_2011.zip",
    RAW_PRECIP_DIR / "maynas_persiann_cdr_2012_2023.zip",
]

RAIN_DAY_THRESHOLD_MM = 1.0
NODATA_VALUE = -99.0
EXPECTED_NCOLS = 17
EXPECTED_NROWS = 18
EXPECTED_CELLSIZE = 0.25

DATE_PATTERN = re.compile(r"CDR_(\d{8})z\.asc$", re.IGNORECASE)

print("=" * 76)
print("PREPARE MAYNAS PERSIANN-CDR PRECIPITATION DATA - V2")
print("=" * 76)


def parse_arcgrid_text(text):
    lines = text.strip().splitlines()
    if len(lines) < 7:
        raise ValueError("ASCII grid file is too short.")

    header = {}
    for line in lines[:6]:
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed ArcGrid header line: {line!r}")
        key = parts[0].strip().lower()
        value = parts[1].strip()
        header[key] = int(float(value)) if key in {"ncols", "nrows"} else float(value)

    grid = np.loadtxt(io.StringIO("\n".join(lines[6:])), dtype=float)

    if grid.shape != (header["nrows"], header["ncols"]):
        raise ValueError(
            f"Grid shape {grid.shape} does not match "
            f"{header['nrows']} x {header['ncols']}."
        )

    return header, grid


def validate_header(header, filename):
    if header["ncols"] != EXPECTED_NCOLS:
        raise ValueError(f"{filename}: unexpected ncols.")
    if header["nrows"] != EXPECTED_NROWS:
        raise ValueError(f"{filename}: unexpected nrows.")
    if not np.isclose(header["cellsize"], EXPECTED_CELLSIZE):
        raise ValueError(f"{filename}: unexpected cellsize.")

    nodata = header.get("nodata_value")
    if nodata is None or not np.isclose(nodata, NODATA_VALUE):
        raise ValueError(f"{filename}: unexpected NODATA value.")


for zip_path in ZIP_FILES:
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing expected ZIP file:\n{zip_path}")

daily_records = []
header_reference = None

for zip_path in ZIP_FILES:
    print(f"\nReading {zip_path.name}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".asc")]
        print(f"ASCII raster files found: {len(members):,}")

        for member in members:
            filename = Path(member).name
            match = DATE_PATTERN.search(filename)
            if not match:
                raise ValueError(f"Unexpected filename: {filename}")

            date = pd.to_datetime(match.group(1), format="%Y%m%d")

            with zf.open(member) as f:
                text = f.read().decode("utf-8")

            header, grid = parse_arcgrid_text(text)
            validate_header(header, filename)

            if header_reference is None:
                header_reference = header.copy()
            else:
                for key in [
                    "ncols", "nrows", "xllcorner", "yllcorner",
                    "cellsize", "nodata_value"
                ]:
                    if not np.isclose(header_reference[key], header[key]):
                        raise ValueError(
                            f"{filename}: grid definition changed for {key}."
                        )

            valid_values = grid[~np.isclose(grid, NODATA_VALUE)]

            if valid_values.size == 0:
                daily_records.append({
                    "date": date,
                    "precip_mean_mm": np.nan,
                    "precip_min_cell_mm": np.nan,
                    "precip_max_cell_mm": np.nan,
                    "valid_grid_cells": 0,
                    "source_zip": zip_path.name,
                    "source_file": filename,
                    "source_data_available": False,
                })
                continue

            daily_records.append({
                "date": date,
                "precip_mean_mm": float(valid_values.mean()),
                "precip_min_cell_mm": float(valid_values.min()),
                "precip_max_cell_mm": float(valid_values.max()),
                "valid_grid_cells": int(valid_values.size),
                "source_zip": zip_path.name,
                "source_file": filename,
                "source_data_available": True,
            })

daily = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)

duplicate_dates = daily[daily.duplicated("date", keep=False)]
if not duplicate_dates.empty:
    raise ValueError("Duplicate daily precipitation dates were found.")

first_date = daily["date"].min()
last_date = daily["date"].max()
expected_daily_dates = pd.date_range(first_date, last_date, freq="D")
missing_file_dates = expected_daily_dates.difference(pd.DatetimeIndex(daily["date"]))

no_valid_cells = daily[~daily["source_data_available"]].copy()

negative_precip_days = int((daily["precip_mean_mm"] < 0).sum())
if negative_precip_days:
    raise ValueError("Negative precipitation values remain after masking NODATA.")

daily["rain_day"] = np.where(
    daily["precip_mean_mm"].notna(),
    (daily["precip_mean_mm"] >= RAIN_DAY_THRESHOLD_MM).astype(float),
    np.nan,
)

daily["week_start_date"] = (
    daily["date"]
    - pd.to_timedelta(daily["date"].dt.dayofweek.add(1).mod(7), unit="D")
)

weekly = (
    daily.groupby("week_start_date", as_index=False)
    .agg(
        precip_sum_mm=("precip_mean_mm", lambda x: x.sum(min_count=1)),
        precip_mean_daily_mm=("precip_mean_mm", "mean"),
        precip_max_daily_mm=("precip_mean_mm", "max"),
        precip_min_daily_mm=("precip_mean_mm", "min"),
        rain_days=("rain_day", lambda x: x.sum(min_count=1)),
        calendar_days_present=("date", "count"),
        precip_days_observed=("precip_mean_mm", "count"),
        mean_valid_grid_cells=("valid_grid_cells", "mean"),
    )
    .sort_values("week_start_date")
    .reset_index(drop=True)
)

weekly["rain_day_threshold_mm"] = RAIN_DAY_THRESHOLD_MM
weekly["missing_precip_days"] = 7 - weekly["precip_days_observed"]
weekly["week_complete"] = weekly["precip_days_observed"].eq(7)

DENGUE_TARGET = PROCESSED_DIR / "maynas_dengue_weekly.csv"
if not DENGUE_TARGET.exists():
    raise FileNotFoundError(f"Could not find dengue target:\n{DENGUE_TARGET}")

dengue = pd.read_csv(DENGUE_TARGET, parse_dates=["week_start_date"])
target_start = dengue["week_start_date"].min()
target_end = dengue["week_start_date"].max()

weekly_target_range = weekly[
    (weekly["week_start_date"] >= target_start)
    & (weekly["week_start_date"] <= target_end)
].copy()

join_check = dengue[["week_start_date"]].merge(
    weekly_target_range[["week_start_date", "precip_days_observed", "week_complete"]],
    on="week_start_date",
    how="left",
)

missing_precip_weeks = join_check[join_check["precip_days_observed"].isna()].copy()
incomplete_precip_weeks = weekly_target_range[
    ~weekly_target_range["week_complete"]
].copy()

validation = pd.DataFrame({
    "metric": [
        "source",
        "spatial_unit",
        "daily_first_date",
        "daily_last_date",
        "daily_expected_calendar_days",
        "daily_raster_files_observed",
        "daily_dates_without_raster_file",
        "daily_rasters_without_valid_cells",
        "duplicate_daily_dates",
        "grid_ncols",
        "grid_nrows",
        "grid_cellsize_degrees",
        "grid_xllcorner",
        "grid_yllcorner",
        "valid_grid_cell_count_min_positive",
        "valid_grid_cell_count_max",
        "weekly_rows_target_range",
        "incomplete_precip_weeks_target_range",
        "dengue_weeks_without_precip_row",
        "rain_day_threshold_mm",
    ],
    "value": [
        "PERSIANN-CDR",
        "Maynas, Loreto, Peru",
        first_date.date().isoformat(),
        last_date.date().isoformat(),
        len(expected_daily_dates),
        len(daily),
        len(missing_file_dates),
        len(no_valid_cells),
        len(duplicate_dates),
        int(header_reference["ncols"]),
        int(header_reference["nrows"]),
        header_reference["cellsize"],
        header_reference["xllcorner"],
        header_reference["yllcorner"],
        int(daily.loc[daily["valid_grid_cells"] > 0, "valid_grid_cells"].min()),
        int(daily["valid_grid_cells"].max()),
        len(weekly_target_range),
        len(incomplete_precip_weeks),
        len(missing_precip_weeks),
        RAIN_DAY_THRESHOLD_MM,
    ],
})

daily.to_csv(DAILY_OUTPUT, index=False, date_format="%Y-%m-%d")
weekly_target_range.to_csv(WEEKLY_OUTPUT, index=False, date_format="%Y-%m-%d")
validation.to_csv(VALIDATION_OUTPUT, index=False)

pd.DataFrame({"missing_date": missing_file_dates}).to_csv(
    MISSING_DATES_OUTPUT, index=False, date_format="%Y-%m-%d"
)

no_valid_cells.to_csv(
    NO_VALID_CELLS_OUTPUT, index=False, date_format="%Y-%m-%d"
)

print("\n" + "=" * 76)
print("PRECIPITATION PREPARATION COMPLETE")
print("=" * 76)
print(f"Daily rows: {len(daily):,}")
print(f"Dates with no raster file: {len(missing_file_dates):,}")
print(f"Rasters with no valid Maynas cells: {len(no_valid_cells):,}")
print(f"Weekly rows in dengue range: {len(weekly_target_range):,}")
print(f"Incomplete precipitation weeks: {len(incomplete_precip_weeks):,}")
print(f"Dengue weeks without precipitation row: {len(missing_precip_weeks):,}")
print(f"\nSaved: {DAILY_OUTPUT.resolve()}")
print(f"Saved: {WEEKLY_OUTPUT.resolve()}")
print(f"Saved: {VALIDATION_OUTPUT.resolve()}")
print(f"Saved: {MISSING_DATES_OUTPUT.resolve()}")
print(f"Saved: {NO_VALID_CELLS_OUTPUT.resolve()}")