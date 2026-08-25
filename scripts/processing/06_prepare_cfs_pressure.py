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

CFSR_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cfsr"
    / "pressure"
)

CFSV2_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cfsv2"
    / "pressure"
)

MAYNAS_BOUNDARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "boundaries"
    / "maynas_gaul2024.geojson"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cfs"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "audit"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SIX_HOURLY_OUTPUT = (
    PROCESSED_DIR
    / "maynas_cfs_pressure_6hourly.csv"
)

DAILY_OUTPUT = (
    PROCESSED_DIR
    / "maynas_cfs_pressure_daily.csv"
)

WEEKLY_OUTPUT = (
    PROCESSED_DIR
    / "maynas_cfs_pressure_weekly.csv"
)

SUMMARY_OUTPUT = (
    AUDIT_DIR
    / "cfs_pressure_processing_summary.txt"
)

STREAM_SELECTION_OUTPUT = (
    AUDIT_DIR
    / "cfs_pressure_stream_selection.csv"
)


# ============================================================
# EXPECTED STRUCTURE
# ============================================================

EXPECTED_FORECAST_HOUR = 6
EXPECTED_GRID_RESOLUTION = 0.5

PRESSURE_VARIABLE = "PRES_L1"

WEEK_FREQUENCY = "W-SUN"


# ============================================================
# STREAM-SELECTION RULE
# ============================================================

def is_selected_pressure_stream(filename):
    """
    Retain only the pgbh / pgrbh pressure stream.

    This rule is based on the duplicate investigation:
      - CFSR pre-2010 contains only pgbh
      - from 2010 an ipvh stream also appears
      - CFSv2 contains parallel pgrbh and ipvgrbh streams
      - representative Maynas means were almost identical
      - retaining pgbh/pgrbh provides one continuous lineage

    Explicitly exclude ipvh / ipvgrbh.
    """

    name = filename.lower()

    if (
        "ipvh" in name
        or "ipvgrbh" in name
    ):
        return False

    return (
        name.startswith("pgbh")
        or ".pgrbh." in name
    )


# ============================================================
# HELPERS
# ============================================================

def load_boundary(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Maynas boundary not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        geojson = json.load(
            file
        )

    if (
        geojson.get(
            "type"
        )
        == "FeatureCollection"
    ):

        features = geojson.get(
            "features",
            []
        )

        if len(
            features
        ) != 1:

            raise ValueError(
                "Expected exactly one Maynas feature, "
                f"found {len(features)}."
            )

        geometry = shape(
            features[
                0
            ][
                "geometry"
            ]
        )

    elif (
        geojson.get(
            "type"
        )
        == "Feature"
    ):

        geometry = shape(
            geojson[
                "geometry"
            ]
        )

    else:

        geometry = shape(
            geojson
        )

    if not geometry.is_valid:

        raise ValueError(
            "Maynas boundary geometry is invalid."
        )

    return geometry


def normalise_lon(value):
    value = float(
        value
    )

    return (
        value - 360
        if value > 180
        else value
    )


def build_grid_mask(
    lat_values,
    lon_values,
    boundary
):
    """
    Select 0.5-degree grid-cell centres that fall inside
    or on the GAUL 2024 Maynas polygon.
    """

    prepared = prep(
        boundary
    )

    mask = np.zeros(
        (
            len(
                lat_values
            ),
            len(
                lon_values
            )
        ),
        dtype=bool
    )

    for i, lat in enumerate(
        lat_values
    ):

        for j, lon_native in enumerate(
            lon_values
        ):

            lon = normalise_lon(
                lon_native
            )

            point = Point(
                lon,
                float(
                    lat
                )
            )

            mask[
                i,
                j
            ] = (
                prepared.contains(
                    point
                )
                or boundary.touches(
                    point
                )
            )

    return mask


def decompress_to_temp(
    filepath
):
    """
    Decompress one .nc.gz file to a temporary .nc file.
    """

    temp_file = (
        tempfile.NamedTemporaryFile(
            suffix=".nc",
            delete=False
        )
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


def validate_dataset(
    ds,
    filepath
):
    """
    Reconfirm core pressure assumptions established by the audit.
    """

    if PRESSURE_VARIABLE not in ds.variables:

        raise ValueError(
            f"{filepath.name}: "
            f"{PRESSURE_VARIABLE} missing."
        )

    if "time" not in ds.coords:

        raise ValueError(
            f"{filepath.name}: "
            "time coordinate missing."
        )

    if (
        "lat" not in ds.coords
        or "lon" not in ds.coords
    ):

        raise ValueError(
            f"{filepath.name}: "
            "lat/lon coordinates missing."
        )

    if "forecast_hour" not in ds.variables:

        raise ValueError(
            f"{filepath.name}: "
            "forecast_hour missing."
        )

    forecast_hours = {
        int(
            value
        )
        for value
        in np.asarray(
            ds[
                "forecast_hour"
            ].values
        ).ravel()
        if not pd.isna(
            value
        )
    }

    if forecast_hours != {
        EXPECTED_FORECAST_HOUR
    }:

        raise ValueError(
            f"{filepath.name}: "
            "unexpected forecast hours "
            f"{forecast_hours}"
        )

    pressure_units = (
        ds[
            PRESSURE_VARIABLE
        ]
        .attrs
        .get(
            "units"
        )
    )

    if (
        pressure_units is not None
        and str(
            pressure_units
        ).strip().lower()
        not in {
            "pa",
            "pascal",
            "pascals"
        }
    ):

        raise ValueError(
            f"{filepath.name}: "
            "unexpected pressure units "
            f"{pressure_units}"
        )

    pressure_level = (
        ds[
            PRESSURE_VARIABLE
        ]
        .attrs
        .get(
            "level"
        )
    )

    if (
        pressure_level is not None
        and
        "ground or water surface"
        not in str(
            pressure_level
        ).lower()
    ):

        raise ValueError(
            f"{filepath.name}: "
            "unexpected pressure level "
            f"{pressure_level}"
        )

    lat = np.asarray(
        ds[
            "lat"
        ].values,
        dtype=float
    )

    lon = np.asarray(
        ds[
            "lon"
        ].values,
        dtype=float
    )

    if len(
        lat
    ) > 1:

        lat_spacing = np.unique(
            np.round(
                np.abs(
                    np.diff(
                        lat
                    )
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
                f"{filepath.name}: "
                "unexpected latitude spacing "
                f"{lat_spacing}"
            )

    if len(
        lon
    ) > 1:

        lon_spacing = np.unique(
            np.round(
                np.abs(
                    np.diff(
                        lon
                    )
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
                f"{filepath.name}: "
                "unexpected longitude spacing "
                f"{lon_spacing}"
            )


def spatial_mean(
    data_array,
    mask
):
    """
    Calculate the unweighted spatial mean across Maynas
    grid-cell centres.
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


def get_selected_files(
    directory
):
    """
    Return selected pgbh/pgrbh files and excluded alternative-stream files.
    """

    all_files = sorted(
        directory.glob(
            "*.nc.gz"
        )
    )

    selected = [
        filepath
        for filepath
        in all_files
        if is_selected_pressure_stream(
            filepath.name
        )
    ]

    excluded = [
        filepath
        for filepath
        in all_files
        if not is_selected_pressure_stream(
            filepath.name
        )
    ]

    return (
        all_files,
        selected,
        excluded
    )


def process_source(
    directory,
    source,
    boundary,
    grid_cache
):
    """
    Process the selected pressure stream for one CFS generation.
    """

    (
        all_files,
        files,
        excluded_files
    ) = get_selected_files(
        directory
    )

    if not files:

        raise FileNotFoundError(
            "No selected pgbh/pgrbh pressure "
            f"files found in:\n{directory}"
        )

    print()
    print("=" * 78)
    print(
        f"PROCESSING {source}"
    )
    print("=" * 78)

    print(
        f"All pressure files: "
        f"{len(all_files):,}"
    )

    print(
        f"Selected pgbh/pgrbh files: "
        f"{len(files):,}"
    )

    print(
        f"Excluded alternative-stream files: "
        f"{len(excluded_files):,}"
    )

    rows = []
    failed = []

    stream_rows = []

    for filepath in files:

        stream_rows.append({
            "source":
                source,

            "filename":
                filepath.name,

            "selected":
                True,

            "reason":
                "pgbh/pgrbh continuous pressure stream"
        })

    for filepath in excluded_files:

        stream_rows.append({
            "source":
                source,

            "filename":
                filepath.name,

            "selected":
                False,

            "reason":
                "excluded alternative ipvh/ipvgrbh pressure stream"
        })

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

            temp_path = (
                decompress_to_temp(
                    filepath
                )
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

                lat = np.asarray(
                    ds[
                        "lat"
                    ].values,
                    dtype=float
                )

                lon = np.asarray(
                    ds[
                        "lon"
                    ].values,
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

                if grid_key not in grid_cache:

                    mask = (
                        build_grid_mask(
                            lat,
                            lon,
                            boundary
                        )
                    )

                    selected_cells = int(
                        mask.sum()
                    )

                    if selected_cells == 0:

                        raise ValueError(
                            f"{filepath.name}: "
                            "no grid-cell centres "
                            "fall inside Maynas."
                        )

                    grid_cache[
                        grid_key
                    ] = (
                        mask,
                        selected_cells
                    )

                    print(
                        f"grid mask="
                        f"{selected_cells} cells",
                        end=" ... ",
                        flush=True
                    )

                (
                    mask,
                    selected_cells
                ) = grid_cache[
                    grid_key
                ]

                times = pd.to_datetime(
                    ds[
                        "time"
                    ].values
                )

                pressure_pa = (
                    spatial_mean(
                        ds[
                            PRESSURE_VARIABLE
                        ],
                        mask
                    )
                )

                pressure_hpa = (
                    pressure_pa
                    / 100.0
                )

                for i, timestamp in enumerate(
                    times
                ):

                    rows.append({
                        "timestamp":
                            timestamp,

                        "source":
                            source,

                        "surface_pressure_pa":
                            pressure_pa[
                                i
                            ],

                        "surface_pressure_hpa":
                            pressure_hpa[
                                i
                            ],

                        "maynas_grid_cells":
                            selected_cells,

                        "source_file":
                            filepath.name,
                    })

            print(
                "OK"
            )

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
        failed,
        pd.DataFrame(
            stream_rows
        ),
        len(
            all_files
        ),
        len(
            files
        ),
        len(
            excluded_files
        ),
    )


def check_six_hourly_completeness(
    df
):
    """
    Compare observed timestamps against a complete 6-hourly sequence.
    """

    timestamps = pd.DatetimeIndex(
        df[
            "timestamp"
        ]
        .sort_values()
        .unique()
    )

    if timestamps.empty:

        raise ValueError(
            "No timestamps available "
            "for completeness check."
        )

    expected = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq="6h"
    )

    missing = (
        expected.difference(
            timestamps
        )
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
    "PREPARE CFS SURFACE PRESSURE DATA"
)
print("=" * 78)

maynas = load_boundary(
    MAYNAS_BOUNDARY
)

print(
    f"Boundary: "
    f"{MAYNAS_BOUNDARY}"
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

(
    cfsr_df,
    cfsr_failed,
    cfsr_streams,
    cfsr_total_files,
    cfsr_selected_files,
    cfsr_excluded_files,
) = process_source(
    CFSR_DIR,
    "CFSR",
    maynas,
    grid_cache
)

(
    cfsv2_df,
    cfsv2_failed,
    cfsv2_streams,
    cfsv2_total_files,
    cfsv2_selected_files,
    cfsv2_excluded_files,
) = process_source(
    CFSV2_DIR,
    "CFSv2",
    maynas,
    grid_cache
)


# ============================================================
# SAVE STREAM-SELECTION AUDIT
# ============================================================

stream_selection = pd.concat(
    [
        cfsr_streams,
        cfsv2_streams
    ],
    ignore_index=True
)

stream_selection.to_csv(
    STREAM_SELECTION_OUTPUT,
    index=False
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
        / "cfs_pressure_selected_stream_duplicate_timestamps.csv"
    )

    duplicate_times.to_csv(
        duplicate_output,
        index=False
    )

    raise ValueError(
        "Selected pgbh/pgrbh stream still contains "
        f"{duplicate_timestamp_count:,} duplicate timestamps. "
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
        / "cfs_pressure_missing_6hourly_timestamps.csv"
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
        f"WARNING: "
        f"{len(missing_times):,} "
        "expected 6-hourly timestamps "
        "are missing."
    )

    print(
        f"See: {missing_output}"
    )


# ============================================================
# SAVE 6-HOURLY DATA
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
        "surface_pressure_hpa": [
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
        "surface_pressure_hpa"
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
        "surface_pressure_hpa": [
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
        "surface_pressure_hpa"
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
    "CFS SURFACE PRESSURE PROCESSING SUMMARY"
)

summary.append(
    "=" * 78
)

summary.append("")

summary.append(
    "STREAM SELECTION"
)

summary.append(
    "-" * 78
)

summary.append(
    "Selected stream: pgbh / pgrbh"
)

summary.append(
    "Excluded stream: ipvh / ipvgrbh"
)

summary.append(
    "Rationale: pgbh/pgrbh provides a continuous pressure "
    "file lineage across the full study period. Representative "
    "stream comparisons showed small grid-cell differences but "
    "negligible differences in Maynas-level spatial mean pressure."
)

summary.append("")

summary.append(
    f"CFSR total pressure files: "
    f"{cfsr_total_files:,}"
)

summary.append(
    f"CFSR selected pgbh files: "
    f"{cfsr_selected_files:,}"
)

summary.append(
    f"CFSR excluded alternative-stream files: "
    f"{cfsr_excluded_files:,}"
)

summary.append(
    f"CFSR failed during processing: "
    f"{len(cfsr_failed):,}"
)

summary.append("")

summary.append(
    f"CFSv2 total pressure files: "
    f"{cfsv2_total_files:,}"
)

summary.append(
    f"CFSv2 selected pgrbh files: "
    f"{cfsv2_selected_files:,}"
)

summary.append(
    f"CFSv2 excluded alternative-stream files: "
    f"{cfsv2_excluded_files:,}"
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
    "Units:"
)

summary.append(
    "  Raw/6-hourly pressure: Pa"
)

summary.append(
    "  Derived pressure: hPa"
)

summary.append(
    "  Daily/weekly modelling features: hPa"
)

summary.append("")

summary.append(
    "Spatial method: unweighted mean of 0.5-degree grid-cell "
    "centres falling inside the GAUL 2024 Maynas polygon."
)

summary.append(
    "Temporal method: native 6-hourly observations retained; "
    "daily and Monday-Sunday weekly mean/min/max pressure "
    "features derived."
)

if (
    cfsr_failed
    or cfsv2_failed
):

    summary.append("")
    summary.append(
        "PROCESSING FAILURES:"
    )

    for filename, error in (
        cfsr_failed
    ):

        summary.append(
            f"  CFSR - "
            f"{filename}: "
            f"{error}"
        )

    for filename, error in (
        cfsv2_failed
    ):

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
    f"Stream selection audit: "
    f"{STREAM_SELECTION_OUTPUT}"
)

print(
    f"Summary: "
    f"{SUMMARY_OUTPUT}"
)
