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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "cfs_pressure_stream_comparison_summary.txt"
)

DETAIL_OUTPUT = (
    OUTPUT_DIR
    / "cfs_pressure_stream_comparison_details.csv"
)

PRE2010_PATTERN_OUTPUT = (
    OUTPUT_DIR
    / "cfs_pressure_pre2010_file_patterns.csv"
)


# ============================================================
# REPRESENTATIVE COMPARISON PAIRS
# ============================================================

REPRESENTATIVE_PAIRS = [
    {
        "source": "CFSR",
        "timestamp": "2010-01-01 06:00:00",
        "file_a": "ipvh06.gdas.20100101-20100105.grb2.nc.gz",
        "file_b": "pgbh06.gdas.20100101-20100105.grb2.nc.gz",
    },
    {
        "source": "CFSR",
        "timestamp": "2010-07-02 18:00:00",
        "file_a": "ipvh06.gdas.20100701-20100705.grb2.nc.gz",
        "file_b": "pgbh06.gdas.20100701-20100705.grb2.nc.gz",
    },
    {
        "source": "CFSR",
        "timestamp": "2011-01-01 00:00:00",
        "file_a": "ipvh06.gdas.20101226-20101231.grb2.nc.gz",
        "file_b": "pgbh06.gdas.20101226-20101231.grb2.nc.gz",
    },
    {
        "source": "CFSv2",
        "timestamp": "2011-01-01 06:00:00",
        "file_a": "ipvh06.gdas.20110101-20110105.grb2.nc.gz",
        "file_b": "pgbh06.gdas.20110101-20110105.grb2.nc.gz",
    },
    {
        "source": "CFSv2",
        "timestamp": "2017-07-01 18:00:00",
        "file_a": "cdas1.20170701.ipvgrbh.grb2.nc.gz",
        "file_b": "cdas1.20170701.pgrbh.grb2.nc.gz",
    },
    {
        "source": "CFSv2",
        "timestamp": "2023-12-31 00:00:00",
        "file_a": "cdas1.20231230.ipvgrbh.grb2.nc.gz",
        "file_b": "cdas1.20231230.pgrbh.grb2.nc.gz",
    },
]


# ============================================================
# HELPERS
# ============================================================

def load_boundary(path):
    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
        if len(features) != 1:
            raise ValueError(
                f"Expected one Maynas feature, found {len(features)}."
            )
        geometry = shape(features[0]["geometry"])
    elif geojson.get("type") == "Feature":
        geometry = shape(geojson["geometry"])
    else:
        geometry = shape(geojson)

    if not geometry.is_valid:
        raise ValueError("Maynas boundary is invalid.")

    return geometry


def normalise_lon(value):
    value = float(value)
    return value - 360 if value > 180 else value


def build_grid_mask(lat_values, lon_values, boundary):
    prepared = prep(boundary)

    mask = np.zeros(
        (len(lat_values), len(lon_values)),
        dtype=bool
    )

    for i, lat in enumerate(lat_values):
        for j, lon_native in enumerate(lon_values):
            lon = normalise_lon(lon_native)
            point = Point(lon, float(lat))

            mask[i, j] = (
                prepared.contains(point)
                or boundary.touches(point)
            )

    return mask


def decompress_to_temp(filepath):
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".nc",
        delete=False
    )

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


def extract_pressure_at_timestamp(
    filepath,
    timestamp,
    boundary,
    grid_cache
):
    """
    Open one compressed NetCDF pressure file and return the
    PRES_L1 grid plus Maynas spatial mean for the requested
    timestamp.
    """

    temp_path = None

    try:
        temp_path = decompress_to_temp(filepath)

        with xr.open_dataset(
            temp_path,
            engine="netcdf4",
            decode_times=True
        ) as ds:

            if "PRES_L1" not in ds.variables:
                raise ValueError(
                    f"{filepath.name}: PRES_L1 not found."
                )

            times = pd.to_datetime(ds["time"].values)
            target = pd.Timestamp(timestamp)

            matches = np.where(times == target)[0]

            if len(matches) != 1:
                raise ValueError(
                    f"{filepath.name}: expected exactly one match for "
                    f"{target}, found {len(matches)}."
                )

            time_index = int(matches[0])

            lat = np.asarray(
                ds["lat"].values,
                dtype=float
            )

            lon = np.asarray(
                ds["lon"].values,
                dtype=float
            )

            grid_key = (
                tuple(np.round(lat, 6)),
                tuple(np.round(lon, 6))
            )

            if grid_key not in grid_cache:
                mask = build_grid_mask(
                    lat,
                    lon,
                    boundary
                )

                if int(mask.sum()) == 0:
                    raise ValueError(
                        f"{filepath.name}: no Maynas grid-cell centres."
                    )

                grid_cache[grid_key] = mask

            mask = grid_cache[grid_key]

            pressure = np.asarray(
                ds["PRES_L1"].isel(time=time_index).values,
                dtype=float
            )

            if pressure.ndim != 2:
                raise ValueError(
                    f"{filepath.name}: expected 2-D pressure grid, "
                    f"got shape {pressure.shape}."
                )

            masked_values = np.where(
                mask,
                pressure,
                np.nan
            )

            maynas_mean_pa = float(
                np.nanmean(masked_values)
            )

            attrs = dict(
                ds["PRES_L1"].attrs
            )

            return {
                "pressure_grid": pressure,
                "maynas_mean_pa": maynas_mean_pa,
                "lat": lat,
                "lon": lon,
                "units": attrs.get("units"),
                "level": attrs.get("level"),
                "long_name": attrs.get("long_name"),
                "forecast_hour": (
                    float(
                        np.asarray(
                            ds["forecast_hour"]
                            .isel(time=time_index)
                            .values
                        )
                    )
                    if "forecast_hour" in ds.variables
                    else np.nan
                ),
            }

    finally:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            temp_path.unlink()


# ============================================================
# PRE-2010 FILE-PATTERN CHECK
# ============================================================

def extract_year_from_filename(filename):
    text = str(filename)

    for token in text.split("."):
        if (
            len(token) >= 8
            and token[:8].isdigit()
            and token[:4].startswith(("19", "20"))
        ):
            return int(token[:4])

    return None


pre2010_rows = []

for filepath in sorted(
    CFSR_DIR.glob("*.nc.gz")
):
    year = extract_year_from_filename(
        filepath.name
    )

    if year is None or year >= 2010:
        continue

    if filepath.name.startswith("ipvh"):
        pattern = "ipvh"
    elif filepath.name.startswith("pgbh"):
        pattern = "pgbh"
    else:
        pattern = "other"

    pre2010_rows.append({
        "filename": filepath.name,
        "year": year,
        "pattern": pattern,
    })

pre2010_df = pd.DataFrame(
    pre2010_rows
)

if not pre2010_df.empty:
    pre2010_df.to_csv(
        PRE2010_PATTERN_OUTPUT,
        index=False
    )


# ============================================================
# RUN REPRESENTATIVE PAIR COMPARISONS
# ============================================================

print("=" * 78)
print("COMPARE CFS PRESSURE STREAMS")
print("=" * 78)

maynas = load_boundary(
    MAYNAS_BOUNDARY
)

grid_cache = {}
comparison_rows = []

for pair in REPRESENTATIVE_PAIRS:

    source = pair["source"]
    source_dir = (
        CFSR_DIR
        if source == "CFSR"
        else CFSV2_DIR
    )

    file_a = source_dir / pair["file_a"]
    file_b = source_dir / pair["file_b"]

    for path in [file_a, file_b]:
        if not path.exists():
            raise FileNotFoundError(
                f"Representative pressure file not found:\n{path}"
            )

    timestamp = pd.Timestamp(
        pair["timestamp"]
    )

    print()
    print(
        f"{source} | {timestamp}"
    )
    print(
        f"  A: {file_a.name}"
    )
    print(
        f"  B: {file_b.name}"
    )

    result_a = extract_pressure_at_timestamp(
        file_a,
        timestamp,
        maynas,
        grid_cache
    )

    result_b = extract_pressure_at_timestamp(
        file_b,
        timestamp,
        maynas,
        grid_cache
    )

    if not np.array_equal(
        result_a["lat"],
        result_b["lat"]
    ):
        raise ValueError(
            "Latitude grids differ between representative pair."
        )

    if not np.array_equal(
        result_a["lon"],
        result_b["lon"]
    ):
        raise ValueError(
            "Longitude grids differ between representative pair."
        )

    grid_a = result_a[
        "pressure_grid"
    ]

    grid_b = result_b[
        "pressure_grid"
    ]

    both_nan = (
        np.isnan(grid_a)
        & np.isnan(grid_b)
    )

    finite_both = (
        np.isfinite(grid_a)
        & np.isfinite(grid_b)
    )

    diff = np.full(
        grid_a.shape,
        np.nan,
        dtype=float
    )

    diff[
        finite_both
    ] = (
        grid_a[
            finite_both
        ]
        -
        grid_b[
            finite_both
        ]
    )

    exact_equal = np.array_equal(
        grid_a,
        grid_b,
        equal_nan=True
    )

    differing_cells = int(
        np.sum(
            finite_both
            & (
                grid_a
                != grid_b
            )
        )
    )

    max_abs_difference_pa = (
        float(
            np.nanmax(
                np.abs(diff)
            )
        )
        if np.any(
            finite_both
        )
        else np.nan
    )

    mean_abs_difference_pa = (
        float(
            np.nanmean(
                np.abs(diff)
            )
        )
        if np.any(
            finite_both
        )
        else np.nan
    )

    mean_difference_pa = (
        float(
            np.nanmean(diff)
        )
        if np.any(
            finite_both
        )
        else np.nan
    )

    maynas_mean_difference_pa = (
        result_a[
            "maynas_mean_pa"
        ]
        -
        result_b[
            "maynas_mean_pa"
        ]
    )

    comparison_rows.append({
        "source":
            source,

        "timestamp":
            timestamp,

        "file_a":
            file_a.name,

        "file_b":
            file_b.name,

        "exact_grid_equal":
            exact_equal,

        "grid_cell_count":
            grid_a.size,

        "finite_cell_count":
            int(
                np.sum(
                    finite_both
                )
            ),

        "differing_cell_count":
            differing_cells,

        "max_abs_difference_pa":
            max_abs_difference_pa,

        "mean_abs_difference_pa":
            mean_abs_difference_pa,

        "mean_difference_a_minus_b_pa":
            mean_difference_pa,

        "maynas_mean_a_pa":
            result_a[
                "maynas_mean_pa"
            ],

        "maynas_mean_b_pa":
            result_b[
                "maynas_mean_pa"
            ],

        "maynas_mean_difference_a_minus_b_pa":
            maynas_mean_difference_pa,

        "units_a":
            result_a[
                "units"
            ],

        "units_b":
            result_b[
                "units"
            ],

        "level_a":
            result_a[
                "level"
            ],

        "level_b":
            result_b[
                "level"
            ],

        "forecast_hour_a":
            result_a[
                "forecast_hour"
            ],

        "forecast_hour_b":
            result_b[
                "forecast_hour"
            ],
    })

    print(
        f"  Exact grid equality: "
        f"{exact_equal}"
    )

    print(
        f"  Differing cells: "
        f"{differing_cells:,}"
    )

    print(
        f"  Max abs difference: "
        f"{max_abs_difference_pa:.6f} Pa"
    )

    print(
        f"  Mean abs difference: "
        f"{mean_abs_difference_pa:.6f} Pa"
    )

    print(
        f"  Maynas mean A: "
        f"{result_a['maynas_mean_pa']:.6f} Pa"
    )

    print(
        f"  Maynas mean B: "
        f"{result_b['maynas_mean_pa']:.6f} Pa"
    )


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    DETAIL_OUTPUT,
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
    "CFS PRESSURE STREAM COMPARISON SUMMARY"
)

summary.append(
    "=" * 78
)

summary.append("")

summary.append(
    "PRE-2010 CFSR FILE PATTERNS"
)

summary.append(
    "-" * 78
)

if pre2010_df.empty:

    summary.append(
        "No pre-2010 CFSR pressure files identified."
    )

else:

    pattern_counts = (
        pre2010_df[
            "pattern"
        ]
        .value_counts()
    )

    for pattern, count in (
        pattern_counts.items()
    ):

        summary.append(
            f"{pattern}: {count:,} files"
        )

    summary.append("")

    summary.append(
        "Pre-2010 year coverage by pattern:"
    )

    for pattern in (
        sorted(
            pre2010_df[
                "pattern"
            ].unique()
        )
    ):

        subset = pre2010_df[
            pre2010_df[
                "pattern"
            ] == pattern
        ]

        summary.append(
            f"  {pattern}: "
            f"{subset['year'].min()} -> "
            f"{subset['year'].max()}"
        )


summary.append("")
summary.append(
    "REPRESENTATIVE STREAM COMPARISONS"
)

summary.append(
    "-" * 78
)

for _, row in (
    comparison_df.iterrows()
):

    summary.append(
        f"{row['source']} | "
        f"{row['timestamp']}"
    )

    summary.append(
        f"  A: {row['file_a']}"
    )

    summary.append(
        f"  B: {row['file_b']}"
    )

    summary.append(
        f"  Exact grid equality: "
        f"{row['exact_grid_equal']}"
    )

    summary.append(
        f"  Differing cells: "
        f"{int(row['differing_cell_count']):,} "
        f"of {int(row['finite_cell_count']):,} "
        "finite cells"
    )

    summary.append(
        f"  Max absolute difference: "
        f"{row['max_abs_difference_pa']:.6f} Pa"
    )

    summary.append(
        f"  Mean absolute difference: "
        f"{row['mean_abs_difference_pa']:.6f} Pa"
    )

    summary.append(
        f"  Maynas mean A: "
        f"{row['maynas_mean_a_pa']:.6f} Pa"
    )

    summary.append(
        f"  Maynas mean B: "
        f"{row['maynas_mean_b_pa']:.6f} Pa"
    )

    summary.append(
        "  Maynas mean A-B: "
        f"{row['maynas_mean_difference_a_minus_b_pa']:.6f} Pa"
    )

    summary.append(
        f"  Units: "
        f"{row['units_a']} / "
        f"{row['units_b']}"
    )

    summary.append(
        f"  Levels: "
        f"{row['level_a']} / "
        f"{row['level_b']}"
    )

    summary.append(
        f"  Forecast hours: "
        f"{row['forecast_hour_a']} / "
        f"{row['forecast_hour_b']}"
    )

    summary.append("")


summary.append(
    "INTERPRETATION GUIDE"
)

summary.append(
    "-" * 78
)

summary.append(
    "If all representative pairs are exactly equal, the two streams "
    "are redundant for surface pressure and one stream can be selected."
)

summary.append(
    "If values differ only trivially, prefer the stream that gives "
    "continuous coverage from 2000 through 2023, subject to metadata "
    "consistency."
)

summary.append(
    "If differences are material, do not deduplicate yet; investigate "
    "the scientific distinction between the pgrbh/pgbh and "
    "ipvgrbh/ipvh products."
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
    "COMPARISON COMPLETE"
)
print("=" * 78)

print(
    f"\nComparison details:\n"
    f"{DETAIL_OUTPUT}"
)

print(
    f"\nPre-2010 file patterns:\n"
    f"{PRE2010_PATTERN_OUTPUT}"
)

print(
    f"\nSummary:\n"
    f"{SUMMARY_OUTPUT}"
)
