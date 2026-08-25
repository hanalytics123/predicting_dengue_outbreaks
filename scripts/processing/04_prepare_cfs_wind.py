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

CFSR_DIR = PROJECT_ROOT / "data" / "raw" / "cfsr" / "wind"
CFSV2_DIR = PROJECT_ROOT / "data" / "raw" / "cfsv2" / "wind"

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

SIX_HOURLY_OUTPUT = PROCESSED_DIR / "maynas_cfs_wind_6hourly.csv"
DAILY_OUTPUT = PROCESSED_DIR / "maynas_cfs_wind_daily.csv"
WEEKLY_OUTPUT = PROCESSED_DIR / "maynas_cfs_wind_weekly.csv"
SUMMARY_OUTPUT = AUDIT_DIR / "cfs_wind_processing_summary.txt"


# ============================================================
# EXPECTED STRUCTURE
# ============================================================

EXPECTED_FORECAST_HOUR = 6
EXPECTED_GRID_RESOLUTION = 0.5

# Common CFS/GRIB-derived variable names for 10 m wind.
U_WIND_CANDIDATES = [
    "U_GRD_L103",
    "UGRD_L103",
]

V_WIND_CANDIDATES = [
    "V_GRD_L103",
    "VGRD_L103",
]

WEEK_FREQUENCY = "W-SAT"


# ============================================================
# HELPERS
# ============================================================

def load_boundary(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Maynas boundary not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
        if len(features) != 1:
            raise ValueError(
                "Expected exactly one Maynas feature, "
                f"found {len(features)}."
            )
        geom = shape(features[0]["geometry"])

    elif geojson.get("type") == "Feature":
        geom = shape(geojson["geometry"])

    else:
        geom = shape(geojson)

    if not geom.is_valid:
        raise ValueError(
            "Maynas boundary geometry is invalid."
        )

    return geom


def normalise_lon(value):
    value = float(value)
    return value - 360 if value > 180 else value


def build_grid_mask(
    lat_values,
    lon_values,
    boundary
):
    """
    Build a Boolean mask selecting 0.5-degree grid-cell centres
    falling inside or on the GAUL 2024 Maynas polygon.

    The same grid is reused across CFSR and CFSv2, so this mask
    is cached after first use.
    """

    prepared = prep(boundary)

    mask = np.zeros(
        (
            len(lat_values),
            len(lon_values)
        ),
        dtype=bool
    )

    for i, lat in enumerate(lat_values):

        for j, lon_native in enumerate(lon_values):

            lon = normalise_lon(
                lon_native
            )

            point = Point(
                lon,
                float(lat)
            )

            mask[i, j] = (
                prepared.contains(point)
                or boundary.touches(point)
            )

    return mask


def decompress_to_temp(filepath):
    """
    Decompress a .nc.gz file to a temporary NetCDF.
    """

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".nc",
        delete=False
    )

    temp_path = Path(
        temp_file.name
    )

    try:

        with gzip.open(
            filepath,
            "rb"
        ) as source:

            while True:

                chunk = source.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                temp_file.write(
                    chunk
                )

    finally:

        temp_file.close()

    return temp_path


def find_variable(
    ds,
    candidates,
    descriptive_terms
):
    """
    Find an expected wind variable by internal variable name first,
    then by descriptive metadata.
    """

    for candidate in candidates:

        if candidate in ds.variables:
            return candidate

    for variable_name in ds.data_vars:

        variable = ds[
            variable_name
        ]

        attrs = variable.attrs

        text = " ".join([
            str(variable_name),
            str(
                attrs.get(
                    "long_name",
                    ""
                )
            ),
            str(
                attrs.get(
                    "standard_name",
                    ""
                )
            ),
            str(
                attrs.get(
                    "description",
                    ""
                )
            ),
        ]).lower()

        if any(
            term in text
            for term
            in descriptive_terms
        ):
            return variable_name

    raise ValueError(
        "Expected wind variable could not be identified. "
        f"Candidates tried: {candidates}"
    )


def validate_dataset(
    ds,
    filepath
):
    """
    Reconfirm the core assumptions established by the wind audit.
    """

    if "time" not in ds.coords:
        raise ValueError(
            f"{filepath.name}: time coordinate missing."
        )

    if (
        "lat" not in ds.coords
        or "lon" not in ds.coords
    ):
        raise ValueError(
            f"{filepath.name}: lat/lon coordinates missing."
        )

    if "forecast_hour" not in ds.variables:
        raise ValueError(
            f"{filepath.name}: forecast_hour missing."
        )

    forecast_hours = {
        int(value)
        for value
        in np.asarray(
            ds["forecast_hour"].values
        ).ravel()
        if not pd.isna(value)
    }

    if forecast_hours != {
        EXPECTED_FORECAST_HOUR
    }:
        raise ValueError(
            f"{filepath.name}: unexpected forecast hours "
            f"{forecast_hours}"
        )

    lat = np.asarray(
        ds["lat"].values,
        dtype=float
    )

    lon = np.asarray(
        ds["lon"].values,
        dtype=float
    )

    if len(lat) > 1:

        lat_spacing = np.unique(
            np.round(
                np.abs(
                    np.diff(lat)
                ),
                6
            )
        )

        if not np.allclose(
            lat_spacing,
            EXPECTED_GRID_RESOLUTION,
            atol=0.001
        ):
            raise ValueError(
                f"{filepath.name}: unexpected latitude "
                f"spacing {lat_spacing}"
            )

    if len(lon) > 1:

        lon_spacing = np.unique(
            np.round(
                np.abs(
                    np.diff(lon)
                ),
                6
            )
        )

        if not np.allclose(
            lon_spacing,
            EXPECTED_GRID_RESOLUTION,
            atol=0.001
        ):
            raise ValueError(
                f"{filepath.name}: unexpected longitude "
                f"spacing {lon_spacing}"
            )


def spatial_mean(
    data_array,
    mask
):
    """
    Calculate an unweighted spatial mean across grid-cell
    centres falling inside Maynas.
    """

    values = np.asarray(
        data_array.values,
        dtype=float
    )

    if values.ndim != 3:

        raise ValueError(
            "Expected time x lat x lon array; "
            f"got shape {values.shape}"
        )

    masked = np.where(
        mask[
            np.newaxis,
            :,
            :
        ],
        values,
        np.nan
    )

    return np.nanmean(
        masked,
        axis=(
            1,
            2
        )
    )


def process_source(
    directory,
    source,
    boundary,
    cached_grid
):
    """
    Process every wind file for one CFS generation.
    """

    files = sorted(
        directory.glob(
            "*.nc.gz"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No .nc.gz files found in:\n"
            f"{directory}"
        )

    print()
    print("=" * 78)
    print(
        f"PROCESSING {source}"
    )
    print("=" * 78)
    print(
        f"Files found: {len(files):,}"
    )

    rows = []
    failed = []

    for index, filepath in enumerate(
        files,
        start=1
    ):

        print(
            f"[{index:,}/{len(files):,}] "
            f"{filepath.name}",
            end=" ... ",
            flush=True
        )

        temp_path = None

        try:

            temp_path = decompress_to_temp(
                filepath
            )

            with xr.open_dataset(
                temp_path,
                engine="netcdf4",
                decode_times=True
            ) as ds:

                validate_dataset(
                    ds,
                    filepath
                )

                u_name = find_variable(
                    ds,
                    U_WIND_CANDIDATES,
                    [
                        "u-component of wind",
                        "u component of wind",
                        "eastward wind",
                    ]
                )

                v_name = find_variable(
                    ds,
                    V_WIND_CANDIDATES,
                    [
                        "v-component of wind",
                        "v component of wind",
                        "northward wind",
                    ]
                )

                lat = np.asarray(
                    ds["lat"].values,
                    dtype=float
                )

                lon = np.asarray(
                    ds["lon"].values,
                    dtype=float
                )

                grid_key = (
                    tuple(
                        np.round(
                            lat,
                            6
                        )
                    ),
                    tuple(
                        np.round(
                            lon,
                            6
                        )
                    )
                )

                if grid_key not in cached_grid:

                    mask = build_grid_mask(
                        lat,
                        lon,
                        boundary
                    )

                    selected_cells = int(
                        mask.sum()
                    )

                    if selected_cells == 0:

                        raise ValueError(
                            f"{filepath.name}: no grid-cell "
                            "centres fall inside Maynas."
                        )

                    cached_grid[
                        grid_key
                    ] = (
                        mask,
                        selected_cells
                    )

                    print(
                        f"grid mask={selected_cells} cells",
                        end=" ... ",
                        flush=True
                    )

                mask, selected_cells = (
                    cached_grid[
                        grid_key
                    ]
                )

                times = pd.to_datetime(
                    ds["time"].values
                )

                u_wind = spatial_mean(
                    ds[u_name],
                    mask
                )

                v_wind = spatial_mean(
                    ds[v_name],
                    mask
                )

                # Calculate speed after spatially averaging
                # the U and V components for Maynas.
                #
                # This represents the magnitude of the regional
                # mean wind vector at each 6-hour timestamp.
                wind_speed = np.sqrt(
                    np.square(
                        u_wind
                    )
                    +
                    np.square(
                        v_wind
                    )
                )

                for i, timestamp in enumerate(
                    times
                ):

                    rows.append({
                        "timestamp":
                            timestamp,

                        "source":
                            source,

                        "u_wind_ms":
                            u_wind[i],

                        "v_wind_ms":
                            v_wind[i],

                        "wind_speed_ms":
                            wind_speed[i],

                        "maynas_grid_cells":
                            selected_cells,

                        "source_file":
                            filepath.name,
                    })

            print("OK")

        except Exception as error:

            print(
                f"FAILED: {error}"
            )

            failed.append(
                (
                    filepath.name,
                    f"{type(error).__name__}: "
                    f"{error}"
                )
            )

        finally:

            if (
                temp_path is not None
                and temp_path.exists()
            ):

                temp_path.unlink()

    return (
        pd.DataFrame(
            rows
        ),
        failed
    )


def check_six_hourly_completeness(
    df
):
    """
    Compare observed timestamps with a complete 6-hourly sequence.
    """

    timestamps = pd.DatetimeIndex(
        df["timestamp"]
        .sort_values()
        .unique()
    )

    if timestamps.empty:

        raise ValueError(
            "No timestamps available for completeness check."
        )

    expected = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq="6h"
    )

    missing = expected.difference(
        timestamps
    )

    duplicates = int(
        df[
            "timestamp"
        ]
        .duplicated()
        .sum()
    )

    return (
        expected,
        missing,
        duplicates
    )


# ============================================================
# LOAD MAYNAS BOUNDARY
# ============================================================

print("=" * 78)
print(
    "PREPARE CFS WIND DATA"
)
print("=" * 78)

maynas = load_boundary(
    MAYNAS_BOUNDARY
)

print(
    f"Boundary: {MAYNAS_BOUNDARY}"
)

print(
    f"Boundary type: "
    f"{maynas.geom_type}"
)

print(
    f"Boundary bounds: "
    f"{maynas.bounds}"
)


# ============================================================
# PROCESS CFSR + CFSV2
# ============================================================

grid_cache = {}

cfsr_df, cfsr_failed = (
    process_source(
        CFSR_DIR,
        "CFSR",
        maynas,
        grid_cache
    )
)

cfsv2_df, cfsv2_failed = (
    process_source(
        CFSV2_DIR,
        "CFSv2",
        maynas,
        grid_cache
    )
)


# ============================================================
# HARMONISE
# ============================================================

combined = pd.concat(
    [
        cfsr_df,
        cfsv2_df
    ],
    ignore_index=True
)

combined[
    "timestamp"
] = pd.to_datetime(
    combined[
        "timestamp"
    ]
)

combined = (
    combined
    .sort_values(
        [
            "timestamp",
            "source"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_timestamp_count = int(
    combined[
        "timestamp"
    ]
    .duplicated()
    .sum()
)

if duplicate_timestamp_count > 0:

    duplicate_times = (
        combined.loc[
            combined[
                "timestamp"
            ].duplicated(
                keep=False
            ),
            [
                "timestamp",
                "source",
                "source_file"
            ]
        ]
    )

    duplicate_output = (
        AUDIT_DIR
        / "cfs_wind_duplicate_timestamps.csv"
    )

    duplicate_times.to_csv(
        duplicate_output,
        index=False
    )

    raise ValueError(
        f"Found {duplicate_timestamp_count:,} "
        "duplicate timestamps. "
        f"See {duplicate_output}"
    )


# ============================================================
# 6-HOURLY COMPLETENESS CHECK
# ============================================================

(
    expected_times,
    missing_times,
    duplicate_count
) = check_six_hourly_completeness(
    combined
)

if len(
    missing_times
) > 0:

    missing_output = (
        AUDIT_DIR
        / "cfs_wind_missing_6hourly_timestamps.csv"
    )

    pd.DataFrame({
        "missing_timestamp":
            missing_times
    }).to_csv(
        missing_output,
        index=False
    )

    print()
    print(
        f"WARNING: {len(missing_times):,} "
        "expected 6-hourly timestamps "
        "are missing."
    )

    print(
        f"See: {missing_output}"
    )


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
    .set_index(
        "timestamp"
    )
    .sort_index()
)

daily = (
    daily_source
    .resample(
        "D"
    )
    .agg({
        "u_wind_ms": [
            "mean",
            "min",
            "max"
        ],
        "v_wind_ms": [
            "mean",
            "min",
            "max"
        ],
        "wind_speed_ms": [
            "mean",
            "min",
            "max"
        ],
    })
)

daily.columns = [
    "_".join(
        column
    )
    for column
    in daily.columns
]

daily[
    "observations_6hourly"
] = (
    daily_source[
        "wind_speed_ms"
    ]
    .resample(
        "D"
    )
    .count()
)

daily[
    "complete_day"
] = (
    daily[
        "observations_6hourly"
    ]
    == 4
)

daily = (
    daily
    .reset_index()
    .rename(
        columns={
            "timestamp":
                "date"
        }
    )
)

daily.to_csv(
    DAILY_OUTPUT,
    index=False
)


# ============================================================
# WEEKLY FEATURES
# ============================================================

weekly_source = (
    combined
    .set_index(
        "timestamp"
    )
    .sort_index()
)

weekly = (
    weekly_source
    .resample(
        WEEK_FREQUENCY
    )
    .agg({
        "u_wind_ms": [
            "mean",
            "min",
            "max"
        ],
        "v_wind_ms": [
            "mean",
            "min",
            "max"
        ],
        "wind_speed_ms": [
            "mean",
            "min",
            "max"
        ],
    })
)

weekly.columns = [
    "_".join(
        column
    )
    for column
    in weekly.columns
]

weekly[
    "observations_6hourly"
] = (
    weekly_source[
        "wind_speed_ms"
    ]
    .resample(
        WEEK_FREQUENCY
    )
    .count()
)

weekly[
    "complete_week"
] = (
    weekly[
        "observations_6hourly"
    ]
    == 28
)

weekly = (
    weekly
    .reset_index()
    .rename(
        columns={
            "timestamp":
                "week_end_date"
        }
    )
)

weekly[
    "week_start_date"
] = (
    weekly[
        "week_end_date"
    ]
    - pd.to_timedelta(
        6,
        unit="D"
    )
)

weekly = weekly[
    [
        "week_start_date",
        "week_end_date",
        *[
            column
            for column
            in weekly.columns
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

summary.append(
    "=" * 78
)

summary.append(
    "CFS WIND PROCESSING SUMMARY"
)

summary.append(
    "=" * 78
)

summary.append("")

summary.append(
    f"CFSR files processed: "
    f"{len(list(CFSR_DIR.glob('*.nc.gz'))):,}"
)

summary.append(
    f"CFSR failed during processing: "
    f"{len(cfsr_failed):,}"
)

summary.append(
    f"CFSv2 files processed: "
    f"{len(list(CFSV2_DIR.glob('*.nc.gz'))):,}"
)

summary.append(
    f"CFSv2 failed during processing: "
    f"{len(cfsv2_failed):,}"
)

summary.append("")

summary.append(
    f"6-hourly rows: "
    f"{len(combined):,}"
)

summary.append(
    f"First timestamp: "
    f"{combined['timestamp'].min()}"
)

summary.append(
    f"Last timestamp: "
    f"{combined['timestamp'].max()}"
)

summary.append(
    f"Expected 6-hourly timestamps: "
    f"{len(expected_times):,}"
)

summary.append(
    f"Missing 6-hourly timestamps: "
    f"{len(missing_times):,}"
)

summary.append(
    f"Duplicate timestamps: "
    f"{duplicate_count:,}"
)

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

summary.append(
    f"Daily rows: "
    f"{len(daily):,}"
)

summary.append(
    f"Incomplete days: "
    f"{(~daily['complete_day']).sum():,}"
)

summary.append(
    f"Weekly rows: "
    f"{len(weekly):,}"
)

summary.append(
    f"Incomplete weeks: "
    f"{(~weekly['complete_week']).sum():,}"
)

summary.append("")

summary.append(
    "Units after processing:"
)

summary.append(
    "  u_wind_ms: metres per second"
)

summary.append(
    "  v_wind_ms: metres per second"
)

summary.append(
    "  wind_speed_ms: metres per second"
)

summary.append("")

summary.append(
    "Spatial method: unweighted mean of 0.5-degree "
    "grid-cell centres falling inside the GAUL 2024 "
    "Maynas polygon."
)

summary.append(
    "Wind-speed method: magnitude of the Maynas-level "
    "mean U/V wind vector calculated at each 6-hour timestamp "
    "as sqrt(u^2 + v^2)."
)

summary.append(
    "Temporal method: native 6-hourly observations retained; "
    "daily and Sunday-Saturday weekly mean/min/max features derived."
)

if cfsr_failed or cfsv2_failed:

    summary.append("")
    summary.append(
        "PROCESSING FAILURES:"
    )

    for filename, error in cfsr_failed:

        summary.append(
            f"  CFSR - "
            f"{filename}: "
            f"{error}"
        )

    for filename, error in cfsv2_failed:

        summary.append(
            f"  CFSv2 - "
            f"{filename}: "
            f"{error}"
        )


summary_text = "\n".join(
    summary
)

SUMMARY_OUTPUT.write_text(
    summary_text,
    encoding="utf-8"
)

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
    f"6-hourly: "
    f"{SIX_HOURLY_OUTPUT}"
)

print(
    f"Daily:    "
    f"{DAILY_OUTPUT}"
)

print(
    f"Weekly:   "
    f"{WEEKLY_OUTPUT}"
)

print(
    f"Summary:  "
    f"{SUMMARY_OUTPUT}"
)