from pathlib import Path
import gzip
import tempfile
import math

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
    / "temp_humidity"
)

CFSV2_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cfsv2"
    / "temp_humidity"
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
    / "cfs_temp_humidity_file_inventory.csv"
)

VARIABLE_INVENTORY_OUTPUT = (
    OUTPUT_DIR
    / "cfs_temp_humidity_variable_inventory.csv"
)

VALIDATION_OUTPUT = (
    OUTPUT_DIR
    / "cfs_temp_humidity_validation.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "cfs_temp_humidity_audit_summary.txt"
)


# ============================================================
# EXPECTED DATA STRUCTURE
# ============================================================

EXPECTED_FORECAST_HOUR = 6.0
EXPECTED_GRID_RESOLUTION = 0.5
EXPECTED_HEIGHT_METRES = 2.0

# Expected GDEX regional extraction.
#
# Latitude:
#     North = 0
#     South = -5
#
# Longitude:
#     West = -76
#     East = -71
#
# Some files may use 0-360 longitude:
#     284 to 289
EXPECTED_LAT_MIN = -5.0
EXPECTED_LAT_MAX = 0.0

EXPECTED_LON_MIN = -76.0
EXPECTED_LON_MAX = -71.0

# Floating-point tolerance for coordinate comparisons.
COORD_TOLERANCE = 0.01

# Small tolerance for checking 0.5-degree spacing.
GRID_TOLERANCE = 0.001

# Known download failure reported by GDEX.
KNOWN_MISSING_FILES = {
    "CFSv2": {
        "cdas1.20210130.pgrbh.grb2.nc.gz"
    },
    "CFSR": set()
}


# ============================================================
# EXPECTED VARIABLES
# ============================================================

EXPECTED_CLASSIFICATIONS = {
    "temperature",
    "dewpoint_temperature",
    "relative_humidity",
    "specific_humidity"
}


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

    if len(values) < 2:
        return []

    values = [
        float(value)
        for value in values
    ]

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
    Classify a meteorological variable using both
    descriptive metadata and common CFS NetCDF names.
    """

    name_upper = (
        str(variable_name)
        .upper()
    )

    metadata_text = " ".join([
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
        )
    ]).lower()

    # --------------------------------------------------------
    # Common CFS / GRIB-derived names
    # --------------------------------------------------------

    if name_upper.startswith("DPT"):
        return "dewpoint_temperature"

    if name_upper.startswith("R_H"):
        return "relative_humidity"

    if (
        name_upper.startswith("SPF_H")
        or name_upper.startswith("SPFH")
    ):
        return "specific_humidity"

    if name_upper.startswith("TMP"):
        return "temperature"

    # --------------------------------------------------------
    # Descriptive metadata
    # --------------------------------------------------------

    if (
        "dewpoint" in metadata_text
        or "dew point" in metadata_text
    ):
        return "dewpoint_temperature"

    if (
        "relative humidity"
        in metadata_text
    ):
        return "relative_humidity"

    if (
        "specific humidity"
        in metadata_text
    ):
        return "specific_humidity"

    if "temperature" in metadata_text:
        return "temperature"

    return None


def metadata_text(
    variable_name,
    attrs
):
    """
    Combine variable metadata into one lower-case string
    for level checks.
    """

    return " ".join([
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
        str(
            attrs.get(
                "level",
                ""
            )
        ),
        str(
            attrs.get(
                "level_type",
                ""
            )
        ),
        str(
            attrs.get(
                "coordinates",
                ""
            )
        )
    ]).lower()


def infer_two_metre_level(
    variable_name,
    attrs
):
    """
    Try to verify that a variable represents 2 m above
    ground.

    Returns:
        PASS
        FAIL
        UNKNOWN

    Some CFS NetCDF conversions encode the GRIB level
    rather than explicitly writing "2 m above ground",
    so UNKNOWN is not automatically treated as failure.
    """

    text = metadata_text(
        variable_name,
        attrs
    )

    # Explicit 2 m wording.
    two_metre_patterns = [
        "2 m above ground",
        "2 meters above ground",
        "2 metre above ground",
        "2 metres above ground",
        "height above ground 2",
        "height_above_ground = 2",
        "height_above_ground=2"
    ]

    for pattern in two_metre_patterns:

        if pattern in text:
            return "PASS"

    # Explicit evidence for some other height.
    if (
        "10 m above ground" in text
        or "10 meters above ground" in text
        or "10 metres above ground" in text
    ):
        return "FAIL"

    # L103 is GRIB "specified height above ground".
    # It establishes the level type but not necessarily
    # the numeric height by itself.
    if "_l103" in text:
        return "UNKNOWN"

    return "UNKNOWN"


def inspect_file(
    filepath,
    source
):
    """
    Decompress and inspect one .nc.gz file.

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
        "error": None
    }

    validation = {
        "source": source,
        "filename": filepath.name,

        "readable": "FAIL",

        "forecast_6h": "UNKNOWN",
        "grid_0_5_degree": "UNKNOWN",
        "latitude_extent": "UNKNOWN",
        "longitude_extent": "UNKNOWN",

        "temperature_present": False,
        "dewpoint_temperature_present": False,
        "relative_humidity_present": False,
        "specific_humidity_present": False,

        "two_metre_level": "UNKNOWN",

        "overall_validation": "UNKNOWN",
        "validation_notes": ""
    }

    variable_records = []

    notes = []

    temp_path = None

    try:

        # ====================================================
        # DECOMPRESS TEMPORARILY
        # ====================================================

        with gzip.open(
            filepath,
            "rb"
        ) as gz_file:

            with tempfile.NamedTemporaryFile(
                suffix=".nc",
                delete=False
            ) as temp_file:

                temp_path = Path(
                    temp_file.name
                )

                while True:

                    chunk = gz_file.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    temp_file.write(
                        chunk
                    )


        # ====================================================
        # OPEN NETCDF
        # ====================================================

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

                    sorted_times = (
                        pd.Series(times)
                        .sort_values()
                        .reset_index(drop=True)
                    )

                    duplicate_count = int(
                        sorted_times
                        .duplicated()
                        .sum()
                    )

                    file_record[
                        "duplicate_time_count"
                    ] = duplicate_count

                    if len(
                        sorted_times
                    ) > 1:

                        differences = (
                            sorted_times
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

            lat_name = None

            for candidate in [
                "lat",
                "latitude"
            ]:

                if candidate in ds.coords:

                    lat_name = candidate
                    break


            if lat_name:

                lat_values = (
                    ds[lat_name]
                    .values
                )

                lat_values_float = [
                    float(value)
                    for value in lat_values
                ]

                lat_min = min(
                    lat_values_float
                )

                lat_max = max(
                    lat_values_float
                )

                lat_spacing = (
                    unique_coordinate_spacing(
                        lat_values_float
                    )
                )

                file_record[
                    "lat_count"
                ] = len(
                    lat_values_float
                )

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

                lat_extent_ok = (
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
                    if lat_extent_ok
                    else "FAIL"
                )

                if not lat_extent_ok:

                    notes.append(
                        "Unexpected latitude extent: "
                        f"{lat_min} to {lat_max}"
                    )


            # =================================================
            # LONGITUDE
            # =================================================

            lon_name = None

            for candidate in [
                "lon",
                "longitude"
            ]:

                if candidate in ds.coords:

                    lon_name = candidate
                    break


            if lon_name:

                lon_values = (
                    ds[lon_name]
                    .values
                )

                lon_values_native = [
                    float(value)
                    for value in lon_values
                ]

                lon_values_normalised = [
                    normalise_longitude(
                        value
                    )
                    for value
                    in lon_values_native
                ]

                lon_min = min(
                    lon_values_normalised
                )

                lon_max = max(
                    lon_values_normalised
                )

                lon_spacing = (
                    unique_coordinate_spacing(
                        lon_values_native
                    )
                )

                file_record[
                    "lon_count"
                ] = len(
                    lon_values_native
                )

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

                lon_extent_ok = (
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
                    if lon_extent_ok
                    else "FAIL"
                )

                if not lon_extent_ok:

                    notes.append(
                        "Unexpected longitude extent: "
                        f"{lon_min} to {lon_max}"
                    )


            # =================================================
            # GRID VALIDATION
            # =================================================

            lat_res = (
                file_record.get(
                    "lat_resolutions"
                )
            )

            lon_res = (
                file_record.get(
                    "lon_resolutions"
                )
            )

            grid_checks = []

            for resolution_text in [
                lat_res,
                lon_res
            ]:

                if resolution_text:

                    values = [
                        float(value)
                        for value
                        in resolution_text.split(";")
                        if value
                    ]

                    grid_checks.extend(
                        values
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

            if (
                "forecast_hour"
                in ds.variables
            ):

                forecast_values = (
                    pd.Series(
                        ds[
                            "forecast_hour"
                        ].values
                    )
                    .dropna()
                )

                if not forecast_values.empty:

                    unique_forecast_hours = sorted(
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
                        in unique_forecast_hours
                    )

                    forecast_ok = all(
                        almost_equal(
                            value,
                            EXPECTED_FORECAST_HOUR,
                            0.001
                        )
                        for value
                        in unique_forecast_hours
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
                            f"{unique_forecast_hours}"
                        )

            else:

                notes.append(
                    "forecast_hour variable not found"
                )


            # =================================================
            # DATA VARIABLES
            # =================================================

            file_record[
                "data_variables"
            ] = ";".join(
                ds.data_vars.keys()
            )

            level_results = []

            for variable_name in (
                ds.data_vars
            ):

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
                    infer_two_metre_level(
                        variable_name,
                        attrs
                    )
                )

                if (
                    classification
                    in EXPECTED_CLASSIFICATIONS
                ):

                    level_results.append(
                        level_status
                    )

                    validation[
                        f"{classification}_present"
                    ] = True


                variable_record = {

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

                    "two_metre_validation":
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
                        )
                }

                variable_records.append(
                    variable_record
                )


            # =================================================
            # LEVEL VALIDATION
            # =================================================

            if level_results:

                if "FAIL" in level_results:

                    validation[
                        "two_metre_level"
                    ] = "FAIL"

                    notes.append(
                        "At least one expected variable "
                        "appears to use a non-2 m level"
                    )

                elif all(
                    result == "PASS"
                    for result
                    in level_results
                ):

                    validation[
                        "two_metre_level"
                    ] = "PASS"

                else:

                    validation[
                        "two_metre_level"
                    ] = "UNKNOWN"

            else:

                notes.append(
                    "No expected temp/humidity "
                    "variables classified"
                )


            # =================================================
            # OVERALL VALIDATION
            # =================================================

            hard_checks = [
                validation[
                    "readable"
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
                validation[
                    "forecast_6h"
                ]
            ]

            if "FAIL" in hard_checks:

                validation[
                    "overall_validation"
                ] = "FAIL"

            elif all(
                result == "PASS"
                for result
                in hard_checks
            ):

                if (
                    validation[
                        "two_metre_level"
                    ]
                    == "PASS"
                ):

                    validation[
                        "overall_validation"
                    ] = "PASS"

                else:

                    # Structure is otherwise correct,
                    # but the height could not be
                    # verified from metadata.
                    validation[
                        "overall_validation"
                    ] = "PASS_WITH_LEVEL_UNVERIFIED"

            else:

                validation[
                    "overall_validation"
                ] = "UNKNOWN"


    except Exception as error:

        file_record[
            "error"
        ] = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        notes.append(
            file_record["error"]
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

    known_missing = (
        KNOWN_MISSING_FILES.get(
            source,
            set()
        )
    )

    if known_missing:

        print(
            "Known download failures:"
        )

        for filename in sorted(
            known_missing
        ):

            print(
                f" - {filename}"
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
    "CFS TEMPERATURE / HUMIDITY VALIDATION AUDIT"
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
    )

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
# BUILD AUDIT SUMMARY
# ============================================================

summary = []

summary.append(
    "=" * 78
)

summary.append(
    "CFS TEMPERATURE / HUMIDITY VALIDATION SUMMARY"
)

summary.append(
    "=" * 78
)


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

    summary.append("")
    summary.append(
        source
    )
    summary.append(
        "-" * 78
    )

    summary.append(
        f"Downloaded .nc.gz files: "
        f"{len(files):,}"
    )

    known_missing = (
        KNOWN_MISSING_FILES.get(
            source,
            set()
        )
    )

    summary.append(
        f"Known failed downloads: "
        f"{len(known_missing):,}"
    )

    for filename in sorted(
        known_missing
    ):

        summary.append(
            f"  - {filename}"
        )

    summary.append(
        f"Readable files: "
        f"{files['read_success'].sum():,}"
    )

    summary.append(
        f"Unreadable files: "
        f"{(~files['read_success']).sum():,}"
    )

    summary.append("")

    summary.append(
        "Overall validation:"
    )

    validation_counts = (
        validations[
            "overall_validation"
        ]
        .value_counts(
            dropna=False
        )
    )

    for status, count in (
        validation_counts.items()
    ):

        summary.append(
            f"  {status}: {count:,}"
        )


    # --------------------------------------------------------
    # Individual validation rules
    # --------------------------------------------------------

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
            "two_metre_level",
            "2 m above-ground level"
        )

    ]:

        summary.append("")
        summary.append(
            f"{label}:"
        )

        counts = (
            validations[
                column
            ]
            .value_counts(
                dropna=False
            )
        )

        for status, count in (
            counts.items()
        ):

            summary.append(
                f"  {status}: {count:,}"
            )


    # --------------------------------------------------------
    # Variable inventory
    # --------------------------------------------------------

    summary.append("")
    summary.append(
        "Classified variables:"
    )

    variable_counts = (
        variables[
            "classification"
        ]
        .fillna(
            "unclassified"
        )
        .value_counts()
    )

    for variable, count in (
        variable_counts.items()
    ):

        summary.append(
            f"  {variable}: "
            f"{count:,}"
        )


    # --------------------------------------------------------
    # Temporal coverage
    # --------------------------------------------------------

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
# TRANSITION CHECK
# ============================================================

summary.append("")
summary.append(
    "=" * 78
)

summary.append(
    "CFSR -> CFSv2 TRANSITION"
)

summary.append(
    "=" * 78
)


cfsr_last = pd.to_datetime(
    file_df.loc[
        file_df["source"]
        == "CFSR",
        "last_time"
    ],
    errors="coerce"
).max()


cfsv2_first = pd.to_datetime(
    file_df.loc[
        file_df["source"]
        == "CFSv2",
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

    transition_difference = (
        cfsv2_first
        - cfsr_last
    )

    summary.append(
        "CFSv2 first timestamp minus "
        "CFSR last timestamp: "
        f"{transition_difference}"
    )


# ============================================================
# KNOWN CFSV2 GAP
# ============================================================

summary.append("")
summary.append(
    "=" * 78
)

summary.append(
    "KNOWN ACQUISITION ISSUE"
)

summary.append(
    "=" * 78
)

summary.append(
    "The following CFSv2 source file failed "
    "during GDEX download:"
)

summary.append(
    "cdas1.20210130.pgrbh.grb2.nc.gz"
)

summary.append(
    "GDEX response: HTTP 500 Internal Server Error."
)

summary.append(
    "This file should be treated as a known raw-data "
    "acquisition gap and investigated during temporal "
    "completeness checking rather than silently imputed."
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