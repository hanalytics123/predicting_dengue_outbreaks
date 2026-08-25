from pathlib import Path
import gzip
import tempfile
import json

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape, Point
from shapely.prepared import prep


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CFSR_DIR = PROJECT_ROOT / "data" / "raw" / "cfsr" / "temp_humidity"
CFSV2_DIR = PROJECT_ROOT / "data" / "raw" / "cfsv2" / "temp_humidity"

MAYNAS_BOUNDARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "boundaries"
    / "maynas_gaul2024.geojson"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "cfs"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SIX_HOURLY_OUTPUT = PROCESSED_DIR / "maynas_cfs_temp_humidity_6hourly.csv"
DAILY_OUTPUT = PROCESSED_DIR / "maynas_cfs_temp_humidity_daily.csv"
WEEKLY_OUTPUT = PROCESSED_DIR / "maynas_cfs_temp_humidity_weekly.csv"
SUMMARY_OUTPUT = AUDIT_DIR / "cfs_temp_humidity_processing_summary.txt"


# ============================================================
# EXPECTED STRUCTURE
# ============================================================

EXPECTED_VARIABLES = {
    "TMP_L103": "temperature_c",
    "DPT_L103": "dewpoint_c",
    "R_H_L103": "relative_humidity_pct",
    "SPF_H_L103": "specific_humidity_kgkg",
}

EXPECTED_FORECAST_HOUR = 6
EXPECTED_GRID_RESOLUTION = 0.5

# Weekly periods use Sunday as the week start, consistent with
# the precipitation and dengue weekly datasets. Pandas W-SAT
# groups Sunday-Saturday weeks and labels them by Saturday.
WEEK_FREQUENCY = "W-SAT"


# ============================================================
# HELPERS
# ============================================================

def load_boundary(path):
    if not path.exists():
        raise FileNotFoundError(f"Maynas boundary not found:\n{path}")

    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
        if len(features) != 1:
            raise ValueError(
                f"Expected exactly one Maynas feature, found {len(features)}."
            )
        geom = shape(features[0]["geometry"])
    elif geojson.get("type") == "Feature":
        geom = shape(geojson["geometry"])
    else:
        geom = shape(geojson)

    if not geom.is_valid:
        raise ValueError("Maynas boundary geometry is invalid.")

    return geom


def normalise_lon(value):
    value = float(value)
    return value - 360 if value > 180 else value


def build_grid_mask(lat_values, lon_values, boundary):
    """
    Select grid-cell centres that fall inside or on the Maynas polygon.
    The GDEX files use the same 0.5-degree grid, so this mask is built
    once and reused.
    """
    prepared = prep(boundary)

    mask = np.zeros((len(lat_values), len(lon_values)), dtype=bool)

    for i, lat in enumerate(lat_values):
        for j, lon_native in enumerate(lon_values):
            lon = normalise_lon(lon_native)
            point = Point(lon, float(lat))
            mask[i, j] = prepared.contains(point) or boundary.touches(point)

    return mask


def decompress_to_temp(filepath):
    temp_file = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    temp_path = Path(temp_file.name)

    try:
        with gzip.open(filepath, "rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)
    finally:
        temp_file.close()

    return temp_path


def validate_dataset(ds, filepath):
    missing = [v for v in EXPECTED_VARIABLES if v not in ds.variables]
    if missing:
        raise ValueError(
            f"{filepath.name}: expected variables missing: {missing}"
        )

    if "time" not in ds.coords:
        raise ValueError(f"{filepath.name}: time coordinate missing.")

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError(f"{filepath.name}: lat/lon coordinates missing.")

    if "forecast_hour" not in ds.variables:
        raise ValueError(f"{filepath.name}: forecast_hour missing.")

    forecast_hours = {
        int(v)
        for v in np.asarray(ds["forecast_hour"].values).ravel()
        if not pd.isna(v)
    }

    if forecast_hours != {EXPECTED_FORECAST_HOUR}:
        raise ValueError(
            f"{filepath.name}: unexpected forecast hours {forecast_hours}"
        )

    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)

    if len(lat) > 1:
        lat_spacing = np.unique(np.round(np.abs(np.diff(lat)), 6))
        if not np.allclose(
            lat_spacing,
            EXPECTED_GRID_RESOLUTION,
            atol=0.001
        ):
            raise ValueError(
                f"{filepath.name}: unexpected latitude spacing {lat_spacing}"
            )

    if len(lon) > 1:
        lon_spacing = np.unique(np.round(np.abs(np.diff(lon)), 6))
        if not np.allclose(
            lon_spacing,
            EXPECTED_GRID_RESOLUTION,
            atol=0.001
        ):
            raise ValueError(
                f"{filepath.name}: unexpected longitude spacing {lon_spacing}"
            )


def spatial_mean(data_array, mask):
    """
    Calculate an unweighted mean of grid-cell centres inside Maynas.

    At this small latitude range and 0.5-degree resolution, cell areas are
    very similar. The mask ensures cells outside Maynas are excluded.
    """
    values = np.asarray(data_array.values, dtype=float)

    if values.ndim != 3:
        raise ValueError(
            f"Expected time x lat x lon array; got shape {values.shape}"
        )

    masked = np.where(mask[np.newaxis, :, :], values, np.nan)

    return np.nanmean(masked, axis=(1, 2))


def process_source(directory, source, boundary, cached_grid):
    files = sorted(directory.glob("*.nc.gz"))

    if not files:
        raise FileNotFoundError(f"No .nc.gz files found in:\n{directory}")

    print()
    print("=" * 78)
    print(f"PROCESSING {source}")
    print("=" * 78)
    print(f"Files found: {len(files):,}")

    rows = []
    failed = []

    for index, filepath in enumerate(files, start=1):
        print(
            f"[{index:,}/{len(files):,}] {filepath.name}",
            end=" ... ",
            flush=True
        )

        temp_path = None

        try:
            temp_path = decompress_to_temp(filepath)

            with xr.open_dataset(
                temp_path,
                engine="netcdf4",
                decode_times=True
            ) as ds:

                validate_dataset(ds, filepath)

                lat = np.asarray(ds["lat"].values, dtype=float)
                lon = np.asarray(ds["lon"].values, dtype=float)

                grid_key = (
                    tuple(np.round(lat, 6)),
                    tuple(np.round(lon, 6))
                )

                if grid_key not in cached_grid:
                    mask = build_grid_mask(lat, lon, boundary)
                    selected_cells = int(mask.sum())

                    if selected_cells == 0:
                        raise ValueError(
                            f"{filepath.name}: no grid-cell centres fall "
                            "inside Maynas."
                        )

                    cached_grid[grid_key] = (mask, selected_cells)

                    print(
                        f"grid mask={selected_cells} cells",
                        end=" ... ",
                        flush=True
                    )

                mask, selected_cells = cached_grid[grid_key]

                times = pd.to_datetime(ds["time"].values)

                temperature_k = spatial_mean(ds["TMP_L103"], mask)
                dewpoint_k = spatial_mean(ds["DPT_L103"], mask)
                relative_humidity = spatial_mean(ds["R_H_L103"], mask)
                specific_humidity = spatial_mean(ds["SPF_H_L103"], mask)

                for i, timestamp in enumerate(times):
                    rows.append({
                        "timestamp": timestamp,
                        "source": source,
                        "temperature_c": temperature_k[i] - 273.15,
                        "dewpoint_c": dewpoint_k[i] - 273.15,
                        "relative_humidity_pct": relative_humidity[i],
                        "specific_humidity_kgkg": specific_humidity[i],
                        "maynas_grid_cells": selected_cells,
                        "source_file": filepath.name,
                    })

            print("OK")

        except Exception as error:
            print(f"FAILED: {error}")
            failed.append((filepath.name, f"{type(error).__name__}: {error}"))

        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    return pd.DataFrame(rows), failed


def check_six_hourly_completeness(df):
    timestamps = pd.DatetimeIndex(df["timestamp"].sort_values().unique())

    if timestamps.empty:
        raise ValueError("No timestamps available for completeness check.")

    expected = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq="6h"
    )

    missing = expected.difference(timestamps)
    duplicates = int(df["timestamp"].duplicated().sum())

    return expected, missing, duplicates


# ============================================================
# LOAD MAYNAS BOUNDARY
# ============================================================

print("=" * 78)
print("PREPARE CFS TEMPERATURE / HUMIDITY DATA")
print("=" * 78)

maynas = load_boundary(MAYNAS_BOUNDARY)

print(f"Boundary: {MAYNAS_BOUNDARY}")
print(f"Boundary type: {maynas.geom_type}")
print(f"Boundary bounds: {maynas.bounds}")


# ============================================================
# PROCESS CFSR + CFSV2
# ============================================================

grid_cache = {}

cfsr_df, cfsr_failed = process_source(
    CFSR_DIR,
    "CFSR",
    maynas,
    grid_cache
)

cfsv2_df, cfsv2_failed = process_source(
    CFSV2_DIR,
    "CFSv2",
    maynas,
    grid_cache
)


# ============================================================
# HARMONISE
# ============================================================

combined = pd.concat(
    [cfsr_df, cfsv2_df],
    ignore_index=True
)

combined["timestamp"] = pd.to_datetime(combined["timestamp"])

combined = combined.sort_values(
    ["timestamp", "source"]
).reset_index(drop=True)

# There should be no CFSR/CFSv2 overlap based on the audit.
duplicate_timestamp_count = int(
    combined["timestamp"].duplicated().sum()
)

if duplicate_timestamp_count > 0:
    duplicate_times = combined.loc[
        combined["timestamp"].duplicated(keep=False),
        ["timestamp", "source", "source_file"]
    ]

    duplicate_output = (
        AUDIT_DIR
        / "cfs_temp_humidity_duplicate_timestamps.csv"
    )

    duplicate_times.to_csv(
        duplicate_output,
        index=False
    )

    raise ValueError(
        f"Found {duplicate_timestamp_count:,} duplicate timestamps. "
        f"See {duplicate_output}"
    )


# ============================================================
# COMPLETE 6-HOURLY SEQUENCE CHECK
# ============================================================

expected_times, missing_times, duplicate_count = (
    check_six_hourly_completeness(combined)
)

if len(missing_times) > 0:
    missing_output = (
        AUDIT_DIR
        / "cfs_temp_humidity_missing_6hourly_timestamps.csv"
    )

    pd.DataFrame({
        "missing_timestamp": missing_times
    }).to_csv(
        missing_output,
        index=False
    )

    print()
    print(
        f"WARNING: {len(missing_times):,} expected 6-hourly "
        f"timestamps are missing."
    )
    print(f"See: {missing_output}")


# ============================================================
# SAVE HARMONISED 6-HOURLY DATA
# ============================================================

combined.to_csv(
    SIX_HOURLY_OUTPUT,
    index=False
)


# ============================================================
# DAILY FEATURES
# ============================================================

daily_source = (
    combined
    .set_index("timestamp")
    .sort_index()
)

daily = daily_source.resample("D").agg({
    "temperature_c": ["mean", "min", "max"],
    "dewpoint_c": ["mean", "min", "max"],
    "relative_humidity_pct": ["mean", "min", "max"],
    "specific_humidity_kgkg": ["mean", "min", "max"],
})

daily.columns = [
    "_".join(column)
    for column in daily.columns
]

daily["observations_6hourly"] = (
    daily_source["temperature_c"]
    .resample("D")
    .count()
)

daily["complete_day"] = (
    daily["observations_6hourly"] == 4
)

daily = daily.reset_index().rename(
    columns={"timestamp": "date"}
)

daily.to_csv(
    DAILY_OUTPUT,
    index=False
)


# ============================================================
# WEEKLY FEATURES
# ============================================================

# Use the harmonised 6-hourly series directly for weekly climate
# statistics. Weeks end Saturday, therefore week_start_date is Sunday.

weekly_source = (
    combined
    .set_index("timestamp")
    .sort_index()
)

weekly = weekly_source.resample(
    WEEK_FREQUENCY
).agg({
    "temperature_c": ["mean", "min", "max"],
    "dewpoint_c": ["mean", "min", "max"],
    "relative_humidity_pct": ["mean", "min", "max"],
    "specific_humidity_kgkg": ["mean", "min", "max"],
})

weekly.columns = [
    "_".join(column)
    for column in weekly.columns
]

weekly["observations_6hourly"] = (
    weekly_source["temperature_c"]
    .resample(WEEK_FREQUENCY)
    .count()
)

weekly["complete_week"] = (
    weekly["observations_6hourly"] == 28
)

weekly = weekly.reset_index().rename(
    columns={"timestamp": "week_end_date"}
)

weekly["week_start_date"] = (
    weekly["week_end_date"]
    - pd.to_timedelta(6, unit="D")
)

weekly = weekly[
    [
        "week_start_date",
        "week_end_date",
        *[
            column
            for column in weekly.columns
            if column not in {
                "week_start_date",
                "week_end_date"
            }
        ]
    ]
]

weekly.to_csv(
    WEEKLY_OUTPUT,
    index=False
)


# ============================================================
# PROCESSING SUMMARY
# ============================================================

summary = []

summary.append("=" * 78)
summary.append("CFS TEMPERATURE / HUMIDITY PROCESSING SUMMARY")
summary.append("=" * 78)
summary.append("")

summary.append(f"CFSR files processed: {len(list(CFSR_DIR.glob('*.nc.gz'))):,}")
summary.append(f"CFSR failed during processing: {len(cfsr_failed):,}")
summary.append(f"CFSv2 files processed: {len(list(CFSV2_DIR.glob('*.nc.gz'))):,}")
summary.append(f"CFSv2 failed during processing: {len(cfsv2_failed):,}")
summary.append("")

summary.append(f"6-hourly rows: {len(combined):,}")
summary.append(f"First timestamp: {combined['timestamp'].min()}")
summary.append(f"Last timestamp: {combined['timestamp'].max()}")
summary.append(f"Expected 6-hourly timestamps: {len(expected_times):,}")
summary.append(f"Missing 6-hourly timestamps: {len(missing_times):,}")
summary.append(f"Duplicate timestamps: {duplicate_count:,}")
summary.append("")

summary.append(
    "CFSR final timestamp: "
    f"{cfsr_df['timestamp'].max() if not cfsr_df.empty else 'N/A'}"
)
summary.append(
    "CFSv2 first timestamp: "
    f"{cfsv2_df['timestamp'].min() if not cfsv2_df.empty else 'N/A'}"
)
summary.append("")

summary.append(f"Daily rows: {len(daily):,}")
summary.append(
    f"Incomplete days: {(~daily['complete_day']).sum():,}"
)
summary.append(f"Weekly rows: {len(weekly):,}")
summary.append(
    f"Incomplete weeks: {(~weekly['complete_week']).sum():,}"
)
summary.append("")

summary.append("Units after processing:")
summary.append("  temperature_c: degrees Celsius")
summary.append("  dewpoint_c: degrees Celsius")
summary.append("  relative_humidity_pct: percent")
summary.append("  specific_humidity_kgkg: kg/kg")
summary.append("")

summary.append(
    "Spatial method: unweighted mean of 0.5-degree grid-cell centres "
    "falling inside the GAUL 2024 Maynas polygon."
)
summary.append(
    "Temporal method: native 6-hourly observations retained; daily and "
    "Monday-Sunday weekly mean/min/max features derived."
)

if cfsr_failed or cfsv2_failed:
    summary.append("")
    summary.append("PROCESSING FAILURES:")

    for filename, error in cfsr_failed:
        summary.append(f"  CFSR - {filename}: {error}")

    for filename, error in cfsv2_failed:
        summary.append(f"  CFSv2 - {filename}: {error}")

summary_text = "\n".join(summary)

SUMMARY_OUTPUT.write_text(
    summary_text,
    encoding="utf-8"
)

print()
print(summary_text)

print()
print("=" * 78)
print("PROCESSING COMPLETE")
print("=" * 78)
print(f"6-hourly: {SIX_HOURLY_OUTPUT}")
print(f"Daily:    {DAILY_OUTPUT}")
print(f"Weekly:   {WEEKLY_OUTPUT}")
print(f"Summary:  {SUMMARY_OUTPUT}")