from pathlib import Path
import gzip
import tempfile

import numpy as np
import pandas as pd
import xarray as xr


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FILE_INVENTORY_OUTPUT = (
    OUTPUT_DIR
    / "cfs_pressure_file_inventory.csv"
)

VARIABLE_INVENTORY_OUTPUT = (
    OUTPUT_DIR
    / "cfs_pressure_variable_inventory.csv"
)

VALIDATION_OUTPUT = (
    OUTPUT_DIR
    / "cfs_pressure_validation.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "cfs_pressure_audit_summary.txt"
)


# ============================================================
# EXPECTED DATA STRUCTURE
# ============================================================

EXPECTED_FORECAST_HOUR = 6.0
EXPECTED_GRID_RESOLUTION = 0.5

EXPECTED_LAT_MIN = -5.0
EXPECTED_LAT_MAX = 0.0

EXPECTED_LON_MIN = -76.0
EXPECTED_LON_MAX = -71.0

COORD_TOLERANCE = 0.01
GRID_TOLERANCE = 0.001


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_nc_files(directory):
    """
    Return all compressed NetCDF files in a directory.
    """
    return sorted(
        directory.glob("*.nc.gz")
    )


def normalise_longitude(value):
    """
    Convert longitude from 0-360 to -180 to 180.
    """
    value = float(value)

    if value > 180:
        return value - 360

    return value


def almost_equal(
    actual,
    expected,
    tolerance
):
    """
    Floating-point comparison.
    """
    if actual is None:
        return False

    return (
        abs(actual - expected)
        <= tolerance
    )


def unique_coordinate_spacing(values):
    """
    Calculate unique absolute spacing between adjacent
    coordinate values.
    """
    values = [
        float(value)
        for value in values
    ]

    if len(values) < 2:
        return []

    spacings = []

    for first, second in zip(
        values[:-1],
        values[1:]
    ):
        difference = abs(
            second - first
        )

        if difference > 0:
            spacings.append(
                round(
                    difference,
                    6
                )
            )

    return sorted(
        set(spacings)
    )


def classify_variable(
    variable_name,
    attrs
):
    """
    Identify surface pressure while avoiding pressure reduced
    to mean sea level.
    """

    name_upper = (
        str(variable_name)
        .upper()
    )

    text = " ".join([
        str(variable_name),
        str(attrs.get("long_name", "")),
        str(attrs.get("standard_name", "")),
        str(attrs.get("description", "")),
        str(attrs.get("level", "")),
        str(attrs.get("level_type", "")),
    ]).lower()

    # Explicitly avoid mean-sea-level pressure.
    if (
        "mean sea level" in text
        or "pressure reduced to msl" in text
        or "pressure reduced to mean sea level" in text
        or "prmsl" in name_upper.lower()
    ):
        return None

    # Common CFS/GRIB pressure names.
    if (
        name_upper.startswith("PRES")
        or name_upper.startswith("PRESS")
    ):
        return "surface_pressure"

    if (
        "surface pressure" in text
        or "pressure at ground" in text
        or "pressure at surface" in text
    ):
        return "surface_pressure"

    # Generic pressure is only classified when the level metadata
    # explicitly indicates the ground/water surface.
    if (
        "pressure" in text
        and (
            "ground or water surface" in text
            or "ground/water surface" in text
            or "surface" in text
        )
    ):
        return "surface_pressure"

    return None


def infer_surface_level(
    variable_name,
    attrs
):
    """
    Verify that the pressure variable is on the ground/water
    surface level.

    Returns:
        PASS
        FAIL
        UNKNOWN
    """

    text = " ".join([
        str(variable_name),
        str(attrs.get("long_name", "")),
        str(attrs.get("standard_name", "")),
        str(attrs.get("description", "")),
        str(attrs.get("level", "")),
        str(attrs.get("level_type", "")),
        str(attrs.get("coordinates", "")),
    ]).lower()

    pass_patterns = [
        "ground or water surface",
        "ground/water surface",
        "ground or water",
        "surface - value: 0",
        "surface value: 0",
    ]

    for pattern in pass_patterns:
        if pattern in text:
            return "PASS"

    fail_patterns = [
        "mean sea level",
        "isobaric surface",
        "height above ground",
        "2 m above ground",
        "10 m above ground",
    ]

    for pattern in fail_patterns:
        if pattern in text:
            return "FAIL"

    # L1 commonly represents the ground/water surface in GRIB,
    # but if the metadata does not state this explicitly we do
    # not claim independent verification.
    if "_l1" in text:
        return "UNKNOWN"

    return "UNKNOWN"


def decompress_to_temp(filepath):
    """
    Decompress one .nc.gz file to a temporary .nc file.
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


def inspect_file(
    filepath,
    source
):
    """
    Decompress and inspect one pressure .nc.gz file.

    Returns:
        file-level record
        variable-level records
        validation record
    """

    file_record = {
        "source": source,
        "filename": filepath.name,
        "file_size_bytes":
            filepath.stat().st_size,
        "read_success": False,
        "error": None,
    }

    validation = {
        "source": source,
        "filename": filepath.name,

        "readable": "FAIL",

        "forecast_6h": "UNKNOWN",
        "grid_0_5_degree": "UNKNOWN",
        "latitude_extent": "UNKNOWN",
        "longitude_extent": "UNKNOWN",

        "surface_pressure_present": False,
        "surface_level": "UNKNOWN",

        "overall_validation": "UNKNOWN",
        "validation_notes": "",
    }

    variable_records = []
    notes = []
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

            file_record[
                "read_success"
            ] = True

            validation[
                "readable"
            ] = "PASS"

            # =================================================
            # DIMENSIONS
            # =================================================

            file_record[
                "dimensions"
            ] = ";".join(
                f"{name}={size}"
                for name, size
                in ds.sizes.items()
            )

            # =================================================
            # TIME
            # =================================================

            if "time" in ds.coords:
                times = pd.to_datetime(
                    ds["time"].values
                )

                if len(times) > 0:
                    file_record[
                        "time_count"
                    ] = len(times)

                    file_record[
                        "first_time"
                    ] = times.min()

                    file_record[
                        "last_time"
                    ] = times.max()

                    file_record[
                        "duplicate_time_count"
                    ] = int(
                        pd.Series(times)
                        .duplicated()
                        .sum()
                    )

                    if len(times) > 1:
                        differences = (
                            pd.Series(times)
                            .sort_values()
                            .diff()
                            .dropna()
                        )

                        file_record[
                            "median_time_interval_hours"
                        ] = (
                            differences
                            .median()
                            .total_seconds()
                            / 3600
                        )

            # =================================================
            # LATITUDE
            # =================================================

            lat_name = next(
                (
                    name
                    for name
                    in ["lat", "latitude"]
                    if name in ds.coords
                ),
                None
            )

            if lat_name:
                lat_values = np.asarray(
                    ds[lat_name].values,
                    dtype=float
                )

                lat_min = float(
                    lat_values.min()
                )

                lat_max = float(
                    lat_values.max()
                )

                lat_spacing = (
                    unique_coordinate_spacing(
                        lat_values
                    )
                )

                file_record[
                    "lat_count"
                ] = len(lat_values)

                file_record[
                    "lat_min"
                ] = lat_min

                file_record[
                    "lat_max"
                ] = lat_max

                file_record[
                    "lat_resolutions"
                ] = ";".join(
                    map(
                        str,
                        lat_spacing
                    )
                )

                lat_ok = (
                    almost_equal(
                        lat_min,
                        EXPECTED_LAT_MIN,
                        COORD_TOLERANCE
                    )
                    and
                    almost_equal(
                        lat_max,
                        EXPECTED_LAT_MAX,
                        COORD_TOLERANCE
                    )
                )

                validation[
                    "latitude_extent"
                ] = (
                    "PASS"
                    if lat_ok
                    else "FAIL"
                )

                if not lat_ok:
                    notes.append(
                        "Unexpected latitude extent: "
                        f"{lat_min} to {lat_max}"
                    )

            # =================================================
            # LONGITUDE
            # =================================================

            lon_name = next(
                (
                    name
                    for name
                    in ["lon", "longitude"]
                    if name in ds.coords
                ),
                None
            )

            if lon_name:
                lon_values = np.asarray(
                    ds[lon_name].values,
                    dtype=float
                )

                lon_values_normalised = [
                    normalise_longitude(
                        value
                    )
                    for value
                    in lon_values
                ]

                lon_min = min(
                    lon_values_normalised
                )

                lon_max = max(
                    lon_values_normalised
                )

                lon_spacing = (
                    unique_coordinate_spacing(
                        lon_values
                    )
                )

                file_record[
                    "lon_count"
                ] = len(lon_values)

                file_record[
                    "lon_min"
                ] = lon_min

                file_record[
                    "lon_max"
                ] = lon_max

                file_record[
                    "lon_resolutions"
                ] = ";".join(
                    map(
                        str,
                        lon_spacing
                    )
                )

                lon_ok = (
                    almost_equal(
                        lon_min,
                        EXPECTED_LON_MIN,
                        COORD_TOLERANCE
                    )
                    and
                    almost_equal(
                        lon_max,
                        EXPECTED_LON_MAX,
                        COORD_TOLERANCE
                    )
                )

                validation[
                    "longitude_extent"
                ] = (
                    "PASS"
                    if lon_ok
                    else "FAIL"
                )

                if not lon_ok:
                    notes.append(
                        "Unexpected longitude extent: "
                        f"{lon_min} to {lon_max}"
                    )

            # =================================================
            # GRID VALIDATION
            # =================================================

            grid_checks = []

            for resolution_text in [
                file_record.get(
                    "lat_resolutions"
                ),
                file_record.get(
                    "lon_resolutions"
                ),
            ]:
                if resolution_text:
                    grid_checks.extend(
                        float(value)
                        for value
                        in resolution_text.split(";")
                        if value
                    )

            if grid_checks:
                grid_ok = all(
                    almost_equal(
                        value,
                        EXPECTED_GRID_RESOLUTION,
                        GRID_TOLERANCE
                    )
                    for value
                    in grid_checks
                )

                validation[
                    "grid_0_5_degree"
                ] = (
                    "PASS"
                    if grid_ok
                    else "FAIL"
                )

                if not grid_ok:
                    notes.append(
                        "Unexpected grid resolution: "
                        f"{grid_checks}"
                    )

            # =================================================
            # FORECAST-HOUR VALIDATION
            # =================================================

            if "forecast_hour" in ds.variables:
                forecast_values = (
                    pd.Series(
                        np.asarray(
                            ds[
                                "forecast_hour"
                            ].values
                        ).ravel()
                    )
                    .dropna()
                )

                if not forecast_values.empty:
                    forecast_hours = sorted(
                        {
                            float(value)
                            for value
                            in forecast_values
                        }
                    )

                    file_record[
                        "forecast_hours"
                    ] = ";".join(
                        str(value)
                        for value
                        in forecast_hours
                    )

                    forecast_ok = all(
                        almost_equal(
                            value,
                            EXPECTED_FORECAST_HOUR,
                            0.001
                        )
                        for value
                        in forecast_hours
                    )

                    validation[
                        "forecast_6h"
                    ] = (
                        "PASS"
                        if forecast_ok
                        else "FAIL"
                    )

                    if not forecast_ok:
                        notes.append(
                            "Unexpected forecast hour(s): "
                            f"{forecast_hours}"
                        )

            else:
                notes.append(
                    "forecast_hour variable not found"
                )

            # =================================================
            # VARIABLES
            # =================================================

            file_record[
                "data_variables"
            ] = ";".join(
                ds.data_vars.keys()
            )

            surface_level_results = []

            for variable_name in ds.data_vars:
                variable = ds[
                    variable_name
                ]

                attrs = dict(
                    variable.attrs
                )

                classification = (
                    classify_variable(
                        variable_name,
                        attrs
                    )
                )

                level_status = (
                    infer_surface_level(
                        variable_name,
                        attrs
                    )
                )

                if classification == "surface_pressure":
                    validation[
                        "surface_pressure_present"
                    ] = True

                    surface_level_results.append(
                        level_status
                    )

                variable_records.append({
                    "source":
                        source,

                    "filename":
                        filepath.name,

                    "variable_name":
                        variable_name,

                    "classification":
                        classification,

                    "dimensions":
                        ";".join(
                            variable.dims
                        ),

                    "shape":
                        "x".join(
                            map(
                                str,
                                variable.shape
                            )
                        ),

                    "dtype":
                        str(
                            variable.dtype
                        ),

                    "long_name":
                        attrs.get(
                            "long_name"
                        ),

                    "standard_name":
                        attrs.get(
                            "standard_name"
                        ),

                    "units":
                        attrs.get(
                            "units"
                        ),

                    "level":
                        attrs.get(
                            "level"
                        ),

                    "level_type":
                        attrs.get(
                            "level_type"
                        ),

                    "surface_level_validation":
                        level_status,

                    "forecast_hours":
                        file_record.get(
                            "forecast_hours"
                        ),

                    "lat_resolution":
                        file_record.get(
                            "lat_resolutions"
                        ),

                    "lon_resolution":
                        file_record.get(
                            "lon_resolutions"
                        ),

                    "first_time":
                        file_record.get(
                            "first_time"
                        ),

                    "last_time":
                        file_record.get(
                            "last_time"
                        ),
                })

            # =================================================
            # SURFACE-LEVEL VALIDATION
            # =================================================

            if surface_level_results:
                if "FAIL" in surface_level_results:
                    validation[
                        "surface_level"
                    ] = "FAIL"

                elif all(
                    result == "PASS"
                    for result
                    in surface_level_results
                ):
                    validation[
                        "surface_level"
                    ] = "PASS"

                else:
                    validation[
                        "surface_level"
                    ] = "UNKNOWN"

            else:
                notes.append(
                    "Surface pressure variable not identified"
                )

            # =================================================
            # OVERALL VALIDATION
            # =================================================

            hard_checks = [
                validation[
                    "readable"
                ],
                validation[
                    "forecast_6h"
                ],
                validation[
                    "grid_0_5_degree"
                ],
                validation[
                    "latitude_extent"
                ],
                validation[
                    "longitude_extent"
                ],
            ]

            if (
                "FAIL" in hard_checks
                or not validation[
                    "surface_pressure_present"
                ]
            ):
                validation[
                    "overall_validation"
                ] = "FAIL"

            elif all(
                result == "PASS"
                for result
                in hard_checks
            ):
                validation[
                    "overall_validation"
                ] = (
                    "PASS"
                    if validation[
                        "surface_level"
                    ] == "PASS"
                    else "PASS_WITH_LEVEL_UNVERIFIED"
                )

    except Exception as error:
        file_record[
            "error"
        ] = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        notes.append(
            file_record[
                "error"
            ]
        )

        validation[
            "overall_validation"
        ] = "FAIL"

    finally:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            temp_path.unlink()

    validation[
        "validation_notes"
    ] = " | ".join(
        notes
    )

    return (
        file_record,
        variable_records,
        validation
    )


# ============================================================
# AUDIT ONE SOURCE
# ============================================================

def audit_source(
    directory,
    source
):
    files = get_nc_files(
        directory
    )

    print()
    print("=" * 78)
    print(
        f"AUDITING {source}"
    )
    print("=" * 78)

    print(
        f"Directory: {directory}"
    )

    print(
        f".nc.gz files found: "
        f"{len(files):,}"
    )

    print()

    file_records = []
    variable_records = []
    validation_records = []

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

        (
            file_record,
            variables,
            validation
        ) = inspect_file(
            filepath,
            source
        )

        file_records.append(
            file_record
        )

        variable_records.extend(
            variables
        )

        validation_records.append(
            validation
        )

        print(
            validation[
                "overall_validation"
            ]
        )

    return (
        file_records,
        variable_records,
        validation_records
    )


# ============================================================
# CHECK DIRECTORIES
# ============================================================

for directory in [
    CFSR_DIR,
    CFSV2_DIR
]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found:\n"
            f"{directory}"
        )


# ============================================================
# RUN AUDIT
# ============================================================

print()
print("=" * 78)
print(
    "CFS SURFACE PRESSURE VALIDATION AUDIT"
)
print("=" * 78)

all_file_records = []
all_variable_records = []
all_validation_records = []

for directory, source in [
    (
        CFSR_DIR,
        "CFSR"
    ),
    (
        CFSV2_DIR,
        "CFSv2"
    ),
]:
    (
        files,
        variables,
        validations
    ) = audit_source(
        directory,
        source
    )

    all_file_records.extend(
        files
    )

    all_variable_records.extend(
        variables
    )

    all_validation_records.extend(
        validations
    )


# ============================================================
# DATAFRAMES
# ============================================================

file_df = pd.DataFrame(
    all_file_records
)

variable_df = pd.DataFrame(
    all_variable_records
)

validation_df = pd.DataFrame(
    all_validation_records
)


# ============================================================
# SAVE INVENTORIES
# ============================================================

file_df.to_csv(
    FILE_INVENTORY_OUTPUT,
    index=False
)

variable_df.to_csv(
    VARIABLE_INVENTORY_OUTPUT,
    index=False
)

validation_df.to_csv(
    VALIDATION_OUTPUT,
    index=False
)


# ============================================================
# BUILD SUMMARY
# ============================================================

summary = [
    "=" * 78,
    "CFS SURFACE PRESSURE VALIDATION SUMMARY",
    "=" * 78,
]

for source in [
    "CFSR",
    "CFSv2"
]:
    files = file_df[
        file_df["source"]
        == source
    ]

    validations = validation_df[
        validation_df["source"]
        == source
    ]

    variables = variable_df[
        variable_df["source"]
        == source
    ]

    summary.extend([
        "",
        source,
        "-" * 78,
        f"Downloaded .nc.gz files: {len(files):,}",
        f"Readable files: {files['read_success'].sum():,}",
        f"Unreadable files: {(~files['read_success']).sum():,}",
        "",
        "Overall validation:",
    ])

    for status, count in (
        validations[
            "overall_validation"
        ]
        .value_counts(
            dropna=False
        )
        .items()
    ):
        summary.append(
            f"  {status}: {count:,}"
        )

    for column, label in [
        (
            "forecast_6h",
            "6-hour forecast"
        ),
        (
            "grid_0_5_degree",
            "0.5-degree grid"
        ),
        (
            "latitude_extent",
            "Expected latitude extent"
        ),
        (
            "longitude_extent",
            "Expected longitude extent"
        ),
        (
            "surface_level",
            "Ground/water surface level"
        ),
    ]:
        summary.extend([
            "",
            f"{label}:",
        ])

        for status, count in (
            validations[
                column
            ]
            .value_counts(
                dropna=False
            )
            .items()
        ):
            summary.append(
                f"  {status}: {count:,}"
            )

    summary.extend([
        "",
        "Surface pressure variable presence:",
        "  Present: "
        f"{validations['surface_pressure_present'].sum():,}",
        "",
        "Classified variables:",
    ])

    for variable, count in (
        variables[
            "classification"
        ]
        .fillna(
            "unclassified"
        )
        .value_counts()
        .items()
    ):
        summary.append(
            f"  {variable}: {count:,}"
        )

    first_times = pd.to_datetime(
        files[
            "first_time"
        ],
        errors="coerce"
    )

    last_times = pd.to_datetime(
        files[
            "last_time"
        ],
        errors="coerce"
    )

    if first_times.notna().any():
        summary.append("")
        summary.append(
            "Earliest timestamp: "
            f"{first_times.min()}"
        )

    if last_times.notna().any():
        summary.append(
            "Latest timestamp: "
            f"{last_times.max()}"
        )


# ============================================================
# CFSR -> CFSV2 TRANSITION
# ============================================================

summary.extend([
    "",
    "=" * 78,
    "CFSR -> CFSv2 TRANSITION",
    "=" * 78,
])

cfsr_last = pd.to_datetime(
    file_df.loc[
        file_df[
            "source"
        ] == "CFSR",
        "last_time"
    ],
    errors="coerce"
).max()

cfsv2_first = pd.to_datetime(
    file_df.loc[
        file_df[
            "source"
        ] == "CFSv2",
        "first_time"
    ],
    errors="coerce"
).min()

summary.append(
    f"Latest CFSR timestamp: "
    f"{cfsr_last}"
)

summary.append(
    f"Earliest CFSv2 timestamp: "
    f"{cfsv2_first}"
)

if (
    pd.notna(cfsr_last)
    and pd.notna(cfsv2_first)
):
    summary.append(
        "CFSv2 first timestamp minus "
        "CFSR last timestamp: "
        f"{cfsv2_first - cfsr_last}"
    )


# ============================================================
# TEMPORAL COMPLETENESS
# ============================================================

summary.extend([
    "",
    "=" * 78,
    "TEMPORAL COMPLETENESS",
    "=" * 78,
])

for source in [
    "CFSR",
    "CFSv2"
]:
    source_files = file_df[
        (
            file_df[
                "source"
            ] == source
        )
        & file_df[
            "read_success"
        ]
    ].copy()

    all_times = []

    for _, row in (
        source_files.iterrows()
    ):
        first_time = pd.to_datetime(
            row.get(
                "first_time"
            ),
            errors="coerce"
        )

        last_time = pd.to_datetime(
            row.get(
                "last_time"
            ),
            errors="coerce"
        )

        time_count = row.get(
            "time_count"
        )

        if (
            pd.notna(
                first_time
            )
            and pd.notna(
                last_time
            )
            and pd.notna(
                time_count
            )
            and int(
                time_count
            ) > 0
        ):
            all_times.extend(
                pd.date_range(
                    start=first_time,
                    end=last_time,
                    periods=int(
                        time_count
                    )
                ).tolist()
            )

    if all_times:
        all_index = pd.DatetimeIndex(
            all_times
        )

        unique_index = (
            pd.DatetimeIndex(
                sorted(
                    pd.unique(
                        all_index
                    )
                )
            )
        )

        expected = pd.date_range(
            start=unique_index.min(),
            end=unique_index.max(),
            freq="6h"
        )

        missing = (
            expected.difference(
                unique_index
            )
        )

        duplicate_count = (
            len(
                all_index
            )
            - len(
                unique_index
            )
        )

        summary.extend([
            "",
            f"{source}:",
            f"  Unique timestamps: {len(unique_index):,}",
            f"  Expected timestamps: {len(expected):,}",
            f"  Missing timestamps: {len(missing):,}",
            f"  Duplicate timestamps: {duplicate_count:,}",
        ])

        if len(
            missing
        ) > 0:
            missing_output = (
                OUTPUT_DIR
                / f"{source.lower()}_pressure_missing_timestamps.csv"
            )

            pd.DataFrame({
                "missing_timestamp":
                    missing
            }).to_csv(
                missing_output,
                index=False
            )

            summary.append(
                "  Missing timestamp file: "
                f"{missing_output}"
            )


# ============================================================
# WRITE SUMMARY
# ============================================================

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
    "AUDIT COMPLETE"
)
print("=" * 78)

print(
    f"\nFile inventory:\n"
    f"{FILE_INVENTORY_OUTPUT}"
)

print(
    f"\nVariable inventory:\n"
    f"{VARIABLE_INVENTORY_OUTPUT}"
)

print(
    f"\nValidation results:\n"
    f"{VALIDATION_OUTPUT}"
)

print(
    f"\nSummary:\n"
    f"{SUMMARY_OUTPUT}"
)
