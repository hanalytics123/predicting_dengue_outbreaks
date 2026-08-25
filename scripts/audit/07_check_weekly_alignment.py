from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "dengue": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "dengue"
        / "maynas_dengue_weekly.csv"
    ),
    "precipitation": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "persiann"
        / "maynas_precipitation_weekly.csv"
    ),
    "temp_humidity": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "cfs"
        / "maynas_cfs_temp_humidity_weekly.csv"
    ),
    "wind": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "cfs"
        / "maynas_cfs_wind_weekly.csv"
    ),
    "pressure": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "cfs"
        / "maynas_cfs_pressure_weekly.csv"
    ),
    "ndvi": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ndvi"
        / "maynas_mod13q1_ndvi_weekly.csv"
    ),
}

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
    / "weekly_alignment_check_summary.txt"
)

DETAIL_OUTPUT = (
    OUTPUT_DIR
    / "weekly_alignment_check_details.csv"
)


# ============================================================
# HELPERS
# ============================================================

def check_dataset(name, path):
    """
    Run a lightweight validation of one weekly dataset.
    """

    result = {
        "dataset": name,
        "path": str(path),
        "file_exists": path.exists(),
        "rows": None,
        "columns": None,
        "week_start_present": False,
        "first_week_start": None,
        "last_week_start": None,
        "all_week_starts_sunday": False,
        "non_sunday_week_count": None,
        "duplicate_week_count": None,
        "missing_week_start_count": None,
        "expected_week_sequence_count": None,
        "missing_expected_weeks": None,
        "extra_nonweekly_dates": None,
        "status": "FAIL",
        "notes": "",
    }

    if not path.exists():
        result["notes"] = "File not found."
        return result

    try:
        df = pd.read_csv(path)

        result["rows"] = len(df)
        result["columns"] = len(df.columns)

        if "week_start_date" not in df.columns:
            result["notes"] = "week_start_date column not found."
            return result

        result["week_start_present"] = True

        df["week_start_date"] = pd.to_datetime(
            df["week_start_date"],
            errors="coerce"
        )

        missing_week_start_count = int(
            df["week_start_date"]
            .isna()
            .sum()
        )

        result["missing_week_start_count"] = (
            missing_week_start_count
        )

        valid_dates = (
            df[
                "week_start_date"
            ]
            .dropna()
            .sort_values()
        )

        if valid_dates.empty:
            result["notes"] = "No valid week_start_date values."
            return result

        result["first_week_start"] = (
            valid_dates.min()
        )

        result["last_week_start"] = (
            valid_dates.max()
        )

        # pandas weekday:
        # Monday=0 ... Sunday=6
        non_sunday = (
            valid_dates.dt.weekday != 6
        )

        non_sunday_count = int(
            non_sunday.sum()
        )

        result["non_sunday_week_count"] = (
            non_sunday_count
        )

        result["all_week_starts_sunday"] = (
            non_sunday_count == 0
        )

        duplicate_week_count = int(
            valid_dates
            .duplicated()
            .sum()
        )

        result["duplicate_week_count"] = (
            duplicate_week_count
        )

        expected_weeks = pd.date_range(
            start=valid_dates.min(),
            end=valid_dates.max(),
            freq="W-SUN"
        )

        observed_unique = pd.DatetimeIndex(
            valid_dates.unique()
        )

        missing_expected = (
            expected_weeks.difference(
                observed_unique
            )
        )

        extra_nonweekly = (
            observed_unique.difference(
                expected_weeks
            )
        )

        result["expected_week_sequence_count"] = (
            len(expected_weeks)
        )

        result["missing_expected_weeks"] = (
            len(missing_expected)
        )

        result["extra_nonweekly_dates"] = (
            len(extra_nonweekly)
        )

        problems = []

        if missing_week_start_count > 0:
            problems.append(
                f"{missing_week_start_count} missing week_start_date value(s)"
            )

        if non_sunday_count > 0:
            problems.append(
                f"{non_sunday_count} week_start_date value(s) are not Sundays"
            )

        if duplicate_week_count > 0:
            problems.append(
                f"{duplicate_week_count} duplicate week_start_date value(s)"
            )

        if len(missing_expected) > 0:
            problems.append(
                f"{len(missing_expected)} missing weekly date(s) within coverage"
            )

        if len(extra_nonweekly) > 0:
            problems.append(
                f"{len(extra_nonweekly)} dates fall outside the Sunday weekly sequence"
            )

        if problems:
            result["notes"] = "; ".join(
                problems
            )
            result["status"] = "FAIL"

        else:
            result["notes"] = "Weekly alignment checks passed."
            result["status"] = "PASS"

        return result

    except Exception as error:
        result["notes"] = (
            f"{type(error).__name__}: {error}"
        )
        return result


# ============================================================
# RUN CHECKS
# ============================================================

print("=" * 78)
print("WEEKLY DATASET ALIGNMENT CHECK")
print("=" * 78)

results = []

for name, path in DATASETS.items():

    print()
    print(f"Checking {name} ...")

    result = check_dataset(
        name,
        path
    )

    results.append(
        result
    )

    print(
        f"  Status: {result['status']}"
    )

    print(
        f"  Rows: {result['rows']}"
    )

    print(
        f"  First week: {result['first_week_start']}"
    )

    print(
        f"  Last week: {result['last_week_start']}"
    )

    print(
        f"  Notes: {result['notes']}"
    )


# ============================================================
# SAVE DETAIL OUTPUT
# ============================================================

detail_df = pd.DataFrame(
    results
)

detail_df.to_csv(
    DETAIL_OUTPUT,
    index=False
)


# ============================================================
# CROSS-DATASET ALIGNMENT
# ============================================================

summary = []

summary.append(
    "=" * 78
)

summary.append(
    "WEEKLY DATASET ALIGNMENT CHECK SUMMARY"
)

summary.append(
    "=" * 78
)

summary.append("")

for result in results:

    summary.append(
        f"{result['dataset']}"
    )

    summary.append(
        "-" * 78
    )

    summary.append(
        f"Path: {result['path']}"
    )

    summary.append(
        f"Status: {result['status']}"
    )

    summary.append(
        f"Rows: {result['rows']}"
    )

    summary.append(
        f"First week_start_date: "
        f"{result['first_week_start']}"
    )

    summary.append(
        f"Last week_start_date: "
        f"{result['last_week_start']}"
    )

    summary.append(
        "All week_start_date values Sunday: "
        f"{result['all_week_starts_sunday']}"
    )

    summary.append(
        f"Non-Sunday weeks: "
        f"{result['non_sunday_week_count']}"
    )

    summary.append(
        f"Duplicate weeks: "
        f"{result['duplicate_week_count']}"
    )

    summary.append(
        f"Missing week_start_date values: "
        f"{result['missing_week_start_count']}"
    )

    summary.append(
        f"Missing expected weeks inside coverage: "
        f"{result['missing_expected_weeks']}"
    )

    summary.append(
        f"Notes: {result['notes']}"
    )

    summary.append("")


# ============================================================
# DENGUE MASTER-TIMELINE OVERLAP CHECK
# ============================================================

summary.append(
    "=" * 78
)

summary.append(
    "OVERLAP WITH DENGUE MASTER TIMELINE"
)

summary.append(
    "=" * 78
)

dengue_path = DATASETS["dengue"]

if dengue_path.exists():

    dengue = pd.read_csv(
        dengue_path,
        usecols=[
            "week_start_date"
        ]
    )

    dengue[
        "week_start_date"
    ] = pd.to_datetime(
        dengue[
            "week_start_date"
        ],
        errors="coerce"
    )

    dengue_weeks = pd.DatetimeIndex(
        dengue[
            "week_start_date"
        ]
        .dropna()
        .unique()
    )

    for name, path in DATASETS.items():

        if name == "dengue":
            continue

        if not path.exists():
            summary.append(
                f"{name}: file not found"
            )
            continue

        df = pd.read_csv(
            path,
            usecols=[
                "week_start_date"
            ]
        )

        df[
            "week_start_date"
        ] = pd.to_datetime(
            df[
                "week_start_date"
            ],
            errors="coerce"
        )

        source_weeks = pd.DatetimeIndex(
            df[
                "week_start_date"
            ]
            .dropna()
            .unique()
        )

        dengue_without_source = (
            dengue_weeks.difference(
                source_weeks
            )
        )

        source_overlap = (
            dengue_weeks.intersection(
                source_weeks
            )
        )

        summary.append(
            f"{name}:"
        )

        summary.append(
            f"  Dengue weeks matched: "
            f"{len(source_overlap):,} / "
            f"{len(dengue_weeks):,}"
        )

        summary.append(
            f"  Dengue weeks with no matching "
            f"{name} week: "
            f"{len(dengue_without_source):,}"
        )

        if len(
            dengue_without_source
        ) > 0:

            summary.append(
                f"  First unmatched dengue week: "
                f"{dengue_without_source.min()}"
            )

            summary.append(
                f"  Last unmatched dengue week: "
                f"{dengue_without_source.max()}"
            )

else:

    summary.append(
        "Dengue file not found; overlap check skipped."
    )


# ============================================================
# FINAL STATUS
# ============================================================

summary.append("")
summary.append(
    "=" * 78
)

summary.append(
    "OVERALL RESULT"
)

summary.append(
    "=" * 78
)

failed_datasets = [
    result[
        "dataset"
    ]
    for result
    in results
    if result[
        "status"
    ] != "PASS"
]

if failed_datasets:

    summary.append(
        "FAIL - one or more datasets require attention:"
    )

    for dataset in failed_datasets:

        summary.append(
            f"  - {dataset}"
        )

else:

    summary.append(
        "PASS - all weekly datasets use Sunday week_start_date "
        "values with no duplicates or internal weekly gaps."
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
    "CHECK COMPLETE"
)
print("=" * 78)

print(
    f"Details: {DETAIL_OUTPUT}"
)

print(
    f"Summary: {SUMMARY_OUTPUT}"
)
