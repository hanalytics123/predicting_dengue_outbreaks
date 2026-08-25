from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

FILE_INVENTORY = AUDIT_DIR / "cfs_pressure_file_inventory.csv"
VARIABLE_INVENTORY = AUDIT_DIR / "cfs_pressure_variable_inventory.csv"

SUMMARY_OUTPUT = AUDIT_DIR / "cfs_pressure_duplicate_investigation_summary.txt"
OVERLAP_OUTPUT = AUDIT_DIR / "cfs_pressure_duplicate_overlap_groups.csv"
FILE_PATTERN_OUTPUT = AUDIT_DIR / "cfs_pressure_duplicate_file_patterns.csv"
VARIABLE_PATTERN_OUTPUT = AUDIT_DIR / "cfs_pressure_duplicate_variable_patterns.csv"


def filename_pattern(filename):
    """
    Replace an obvious YYYYMMDD token with <DATE> so that
    distinct source-file naming streams are easier to compare.
    """
    parts = str(filename).split(".")
    transformed = []

    for part in parts:
        if (
            len(part) == 8
            and part.isdigit()
            and part.startswith(("19", "20"))
        ):
            transformed.append("<DATE>")
        else:
            transformed.append(part)

    return ".".join(transformed)


print("=" * 78)
print("INVESTIGATE CFS PRESSURE DUPLICATES")
print("=" * 78)

for path in [FILE_INVENTORY, VARIABLE_INVENTORY]:
    if not path.exists():
        raise FileNotFoundError(f"Required audit file not found:\n{path}")

files = pd.read_csv(FILE_INVENTORY)
variables = pd.read_csv(VARIABLE_INVENTORY)

for column in ["first_time", "last_time"]:
    files[column] = pd.to_datetime(files[column], errors="coerce")
    if column in variables.columns:
        variables[column] = pd.to_datetime(
            variables[column],
            errors="coerce"
        )

files["filename_pattern"] = files["filename"].map(filename_pattern)

print(f"Pressure file inventory rows: {len(files):,}")
print(f"Pressure variable inventory rows: {len(variables):,}")


# ============================================================
# RECONSTRUCT TIMESTAMP-TO-FILE MAP
# ============================================================

timestamp_rows = []

for _, row in files.iterrows():
    first_time = row.get("first_time")
    last_time = row.get("last_time")
    time_count = row.get("time_count")

    if (
        pd.isna(first_time)
        or pd.isna(last_time)
        or pd.isna(time_count)
    ):
        continue

    time_count = int(time_count)

    if time_count <= 0:
        continue

    times = pd.date_range(
        start=first_time,
        end=last_time,
        periods=time_count
    )

    for timestamp in times:
        timestamp_rows.append({
            "source": row["source"],
            "timestamp": timestamp,
            "filename": row["filename"],
            "filename_pattern": row["filename_pattern"],
            "forecast_hours": row.get("forecast_hours"),
            "data_variables": row.get("data_variables"),
            "file_size_bytes": row.get("file_size_bytes"),
        })

timestamp_map = pd.DataFrame(timestamp_rows)

timestamp_counts = (
    timestamp_map
    .groupby(["source", "timestamp"], as_index=False)
    .size()
    .rename(columns={"size": "file_count"})
)

duplicate_timestamps = timestamp_counts[
    timestamp_counts["file_count"] > 1
].copy()

duplicate_details = (
    timestamp_map
    .merge(
        duplicate_timestamps[
            ["source", "timestamp", "file_count"]
        ],
        on=["source", "timestamp"],
        how="inner"
    )
    .sort_values(["source", "timestamp", "filename"])
)

duplicate_details.to_csv(
    OVERLAP_OUTPUT,
    index=False
)


# ============================================================
# FILE-PATTERN ANALYSIS
# ============================================================

file_pattern_summary = (
    duplicate_details
    .groupby(
        ["source", "filename_pattern"],
        dropna=False
    )
    .agg(
        duplicate_rows=("timestamp", "size"),
        unique_duplicate_timestamps=("timestamp", "nunique"),
        unique_files=("filename", "nunique"),
        first_duplicate_timestamp=("timestamp", "min"),
        last_duplicate_timestamp=("timestamp", "max"),
        median_file_size_bytes=("file_size_bytes", "median"),
    )
    .reset_index()
    .sort_values(
        ["source", "unique_duplicate_timestamps"],
        ascending=[True, False]
    )
)

file_pattern_summary.to_csv(
    FILE_PATTERN_OUTPUT,
    index=False
)


# ============================================================
# VARIABLE-PATTERN ANALYSIS
# ============================================================

pressure_variables = variables[
    variables["classification"]
    .fillna("")
    .eq("surface_pressure")
].copy()

group_columns = [
    "source",
    "variable_name",
    "units",
    "level",
    "level_type",
    "forecast_hours",
    "lat_resolution",
    "lon_resolution",
]

# Keep only grouping columns that actually exist.
group_columns = [
    column
    for column in group_columns
    if column in pressure_variables.columns
]

variable_pattern_summary = (
    pressure_variables
    .groupby(
        group_columns,
        dropna=False
    )
    .agg(
        file_count=("filename", "nunique"),
        first_timestamp=("first_time", "min"),
        last_timestamp=("last_time", "max"),
    )
    .reset_index()
    .sort_values(
        ["source", "file_count"],
        ascending=[True, False]
    )
)

variable_pattern_summary.to_csv(
    VARIABLE_PATTERN_OUTPUT,
    index=False
)


# ============================================================
# DUPLICATION BY YEAR
# ============================================================

period_rows = []

for source in ["CFSR", "CFSv2"]:
    source_dupes = duplicate_timestamps[
        duplicate_timestamps["source"] == source
    ].copy()

    if source_dupes.empty:
        continue

    source_dupes["year"] = source_dupes["timestamp"].dt.year

    by_year = (
        source_dupes
        .groupby("year", as_index=False)
        .agg(
            duplicate_timestamps=("timestamp", "nunique"),
            min_file_count=("file_count", "min"),
            max_file_count=("file_count", "max"),
        )
    )

    for _, row in by_year.iterrows():
        period_rows.append({
            "source": source,
            "year": int(row["year"]),
            "duplicate_timestamps": int(row["duplicate_timestamps"]),
            "min_file_count": int(row["min_file_count"]),
            "max_file_count": int(row["max_file_count"]),
        })

period_df = pd.DataFrame(period_rows)


# ============================================================
# REPRESENTATIVE DUPLICATE GROUPS
# ============================================================

representative_groups = []

for source in ["CFSR", "CFSv2"]:
    source_dupes = duplicate_timestamps[
        duplicate_timestamps["source"] == source
    ].sort_values("timestamp")

    if source_dupes.empty:
        continue

    positions = {
        "first": source_dupes.iloc[0]["timestamp"],
        "middle": source_dupes.iloc[len(source_dupes) // 2]["timestamp"],
        "last": source_dupes.iloc[-1]["timestamp"],
    }

    for label, timestamp in positions.items():
        subset = duplicate_details[
            (duplicate_details["source"] == source)
            & (duplicate_details["timestamp"] == timestamp)
        ]

        representative_groups.append({
            "source": source,
            "position": label,
            "timestamp": timestamp,
            "files": " | ".join(
                subset["filename"].astype(str).tolist()
            ),
            "patterns": " | ".join(
                subset["filename_pattern"].astype(str).tolist()
            ),
        })

representative_df = pd.DataFrame(representative_groups)


# ============================================================
# SUMMARY
# ============================================================

summary = [
    "=" * 78,
    "CFS PRESSURE DUPLICATE INVESTIGATION SUMMARY",
    "=" * 78,
    "",
    "SOURCE FILE COUNTS",
    "-" * 78,
]

for source, count in files["source"].value_counts().sort_index().items():
    summary.append(f"{source}: {count:,}")

summary.extend([
    "",
    "DUPLICATE TIMESTAMP COUNTS",
    "-" * 78,
])

for source in ["CFSR", "CFSv2"]:
    source_counts = timestamp_counts[
        timestamp_counts["source"] == source
    ]
    source_duplicates = duplicate_timestamps[
        duplicate_timestamps["source"] == source
    ]

    summary.append(f"{source}:")
    summary.append(
        "  Unique timestamps represented: "
        f"{len(source_counts):,}"
    )
    summary.append(
        "  Timestamps represented by >1 file: "
        f"{len(source_duplicates):,}"
    )

    if not source_duplicates.empty:
        summary.append(
            "  First duplicate timestamp: "
            f"{source_duplicates['timestamp'].min()}"
        )
        summary.append(
            "  Last duplicate timestamp: "
            f"{source_duplicates['timestamp'].max()}"
        )
        summary.append("  Files per duplicated timestamp:")

        distribution = (
            source_duplicates["file_count"]
            .value_counts()
            .sort_index()
        )

        for file_count, count in distribution.items():
            summary.append(
                f"    {int(file_count)} files: "
                f"{count:,} timestamps"
            )

summary.extend([
    "",
    "DUPLICATE FILE PATTERNS",
    "-" * 78,
])

if file_pattern_summary.empty:
    summary.append("No duplicate file patterns found.")
else:
    for source in ["CFSR", "CFSv2"]:
        subset = file_pattern_summary[
            file_pattern_summary["source"] == source
        ]

        if subset.empty:
            continue

        summary.append(source)

        for _, row in subset.iterrows():
            summary.append(
                "  "
                f"{row['filename_pattern']} | "
                f"files={int(row['unique_files']):,} | "
                f"duplicate timestamps="
                f"{int(row['unique_duplicate_timestamps']):,} | "
                f"{row['first_duplicate_timestamp']} -> "
                f"{row['last_duplicate_timestamp']}"
            )

summary.extend([
    "",
    "SURFACE PRESSURE VARIABLE PATTERNS",
    "-" * 78,
])

if variable_pattern_summary.empty:
    summary.append("No surface-pressure variable patterns found.")
else:
    for _, row in variable_pattern_summary.iterrows():
        parts = [str(row["source"])]

        for column in group_columns:
            if column == "source":
                continue
            parts.append(f"{column}={row.get(column)}")

        parts.extend([
            f"files={int(row['file_count']):,}",
            f"{row['first_timestamp']} -> {row['last_timestamp']}",
        ])

        summary.append(" | ".join(parts))

summary.extend([
    "",
    "DUPLICATION BY YEAR",
    "-" * 78,
])

if period_df.empty:
    summary.append("No duplicate periods found.")
else:
    for source in ["CFSR", "CFSv2"]:
        subset = period_df[period_df["source"] == source]

        if subset.empty:
            continue

        summary.append(source)

        for _, row in subset.iterrows():
            summary.append(
                "  "
                f"{int(row['year'])}: "
                f"{int(row['duplicate_timestamps']):,} "
                "duplicate timestamps "
                f"(files/timestamp "
                f"{int(row['min_file_count'])}-"
                f"{int(row['max_file_count'])})"
            )

summary.extend([
    "",
    "REPRESENTATIVE DUPLICATE GROUPS",
    "-" * 78,
])

if representative_df.empty:
    summary.append("No representative duplicate groups available.")
else:
    for _, row in representative_df.iterrows():
        summary.append(
            f"{row['source']} {row['position']} "
            f"{row['timestamp']}:"
        )
        summary.append(f"  {row['files']}")

summary.extend([
    "",
    "NEXT INTERPRETATION STEP",
    "-" * 78,
    "Use the file-pattern and variable-pattern outputs to determine "
    "whether duplicate timestamps represent distinct pressure streams "
    "or repeated copies of the same product. Do not drop duplicates "
    "until representative source files have been compared.",
    "If duplicate files share identical variable metadata and source "
    "pattern, compare their actual pressure values for representative "
    "timestamps before choosing a deduplication rule.",
])

summary_text = "\n".join(summary)

SUMMARY_OUTPUT.write_text(
    summary_text,
    encoding="utf-8"
)

print()
print(summary_text)

print()
print("=" * 78)
print("INVESTIGATION COMPLETE")
print("=" * 78)
print(f"\nDuplicate overlap groups:\n{OVERLAP_OUTPUT}")
print(f"\nFile-pattern summary:\n{FILE_PATTERN_OUTPUT}")
print(f"\nVariable-pattern summary:\n{VARIABLE_PATTERN_OUTPUT}")
print(f"\nSummary:\n{SUMMARY_OUTPUT}")
