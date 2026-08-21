import geopandas as gpd
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_BOUNDARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boundaries"
    / "gaul2024"
)

RAW_SHP = RAW_BOUNDARY_DIR / "GAUL_2024_L2.shp"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "boundaries"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "maynas_gaul2024.geojson"

print("=" * 72)
print("EXTRACT MAYNAS BOUNDARY FROM FAO GAUL 2024 LEVEL 2")
print("=" * 72)
print(f"Project root: {PROJECT_ROOT}")
print(f"Input shapefile: {RAW_SHP}")
print(f"Output GeoJSON: {OUTPUT_FILE}")

# ============================================================
# 1. CHECK INPUT EXISTS
# ============================================================

if not RAW_SHP.exists():
    raise FileNotFoundError(
        f"Could not find shapefile:\n{RAW_SHP}\n\n"
        "Expected the GAUL 2024 Level 2 shapefile to be stored in:\n"
        "data/raw/boundaries/gaul2024/"
    )

# ============================================================
# 2. LOAD GAUL 2024 LEVEL 2
# ============================================================

print("\nLoading GAUL 2024 Level 2...")
gaul = gpd.read_file(RAW_SHP)

print(f"Rows loaded: {len(gaul):,}")
print(f"CRS: {gaul.crs}")

print("\nAvailable columns:")
print(gaul.columns.tolist())

# ============================================================
# 3. VALIDATE REQUIRED FIELDS
# ============================================================

required_columns = {
    "gaul0_code",
    "gaul0_name",
    "gaul1_code",
    "gaul1_name",
    "gaul2_code",
    "gaul2_name",
    "geometry",
}

missing_columns = required_columns.difference(gaul.columns)

if missing_columns:
    raise ValueError(
        "Required GAUL fields are missing: "
        + ", ".join(sorted(missing_columns))
    )

# ============================================================
# 4. FILTER TO MAYNAS, LORETO, PERU
# ============================================================

maynas = gaul[
    (gaul["gaul0_name"].str.strip().str.casefold() == "peru")
    & (gaul["gaul1_name"].str.strip().str.casefold() == "loreto")
    & (gaul["gaul2_name"].str.strip().str.casefold() == "maynas")
].copy()

print("\nMatches found:")
print(len(maynas))

if len(maynas) == 0:
    raise ValueError(
        "No Maynas feature was found using Peru -> Loreto -> Maynas."
    )

if len(maynas) > 1:
    print(
        maynas[
            [
                "gaul0_code",
                "gaul0_name",
                "gaul1_code",
                "gaul1_name",
                "gaul2_code",
                "gaul2_name",
            ]
        ]
    )
    raise ValueError(
        "More than one Maynas feature was found. Investigate before exporting."
    )

# ============================================================
# 5. VERIFY IDENTIFIERS
# ============================================================

record = maynas.iloc[0]

expected = {
    "gaul0_code": 207,
    "gaul0_name": "Peru",
    "gaul1_code": 2215,
    "gaul1_name": "Loreto",
    "gaul2_code": 119192,
    "gaul2_name": "Maynas",
}

print("\nSelected feature:")
for field, expected_value in expected.items():
    actual_value = record[field]
    print(f"{field}: {actual_value}")

    if str(actual_value).strip().casefold() != str(expected_value).strip().casefold():
        raise ValueError(
            f"Unexpected value for {field}: "
            f"{actual_value!r} (expected {expected_value!r})"
        )

print("\nAdministrative identifiers verified.")

# ============================================================
# 6. VALIDATE GEOMETRY
# ============================================================

if maynas.geometry.isna().any():
    raise ValueError("Maynas geometry is missing.")

geom = maynas.geometry.iloc[0]

print(f"Geometry type: {geom.geom_type}")
print(f"Geometry valid: {geom.is_valid}")
print(f"Geometry empty: {geom.is_empty}")

if geom.is_empty:
    raise ValueError("Maynas geometry is empty.")

if not geom.is_valid:
    raise ValueError(
        "Maynas geometry is invalid. Do not export before investigating the geometry."
    )

# ============================================================
# 7. VALIDATE / STANDARDISE CRS
# ============================================================

if gaul.crs is None:
    raise ValueError(
        "The GAUL shapefile has no coordinate reference system."
    )

if maynas.crs.to_epsg() != 4326:
    print(f"\nReprojecting from {maynas.crs} to EPSG:4326...")
    maynas = maynas.to_crs(epsg=4326)

print(f"Export CRS: {maynas.crs}")

# ============================================================
# 8. CALCULATE BASIC SPATIAL DIAGNOSTICS
# ============================================================

minx, miny, maxx, maxy = maynas.total_bounds

print("\nBounding box:")
print(f"West:  {minx:.6f}")
print(f"South: {miny:.6f}")
print(f"East:  {maxx:.6f}")
print(f"North: {maxy:.6f}")

# Approximate area using WGS 84 / UTM zone 18S for diagnostic purposes.
maynas_metric = maynas.to_crs(epsg=32718)
area_km2 = maynas_metric.geometry.area.iloc[0] / 1_000_000

print(f"Approximate area: {area_km2:,.2f} km^2")

# ============================================================
# 9. KEEP ONLY USEFUL ATTRIBUTES
# ============================================================

columns_to_keep = [
    "iso3_code",
    "map_code",
    "gaul0_code",
    "gaul0_name",
    "gaul1_code",
    "gaul1_name",
    "gaul2_code",
    "gaul2_name",
    "continent",
    "disp_en",
    "geometry",
]

available_columns = [
    col for col in columns_to_keep
    if col in maynas.columns
]

maynas_export = maynas[available_columns].copy()

# ============================================================
# 10. EXPORT GEOJSON
# ============================================================

maynas_export.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)

print("\nGeoJSON exported successfully.")

# ============================================================
# 11. RELOAD AND VERIFY EXPORT
# ============================================================

check = gpd.read_file(OUTPUT_FILE)

if len(check) != 1:
    raise ValueError(
        f"Export verification failed: expected 1 feature, found {len(check)}."
    )

if check.geometry.isna().any() or check.geometry.iloc[0].is_empty:
    raise ValueError(
        "Export verification failed: geometry is missing or empty."
    )

if int(check.iloc[0]["gaul2_code"]) != 119192:
    raise ValueError(
        "Export verification failed: GAUL2 code does not equal 119192."
    )

print("\nExport verification passed.")
print(f"Saved: {OUTPUT_FILE.resolve()}")

# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("MAYNAS BOUNDARY EXTRACTION COMPLETE")
print("=" * 72)

print("Source: FAO GAUL 2024 Level 2")
print("Country: Peru")
print("ADM1: Loreto")
print("ADM2: Maynas")
print("GAUL2 code: 119192")
print(f"Geometry: {check.geometry.iloc[0].geom_type}")
print(f"CRS: {check.crs}")
print(f"Approximate area: {area_km2:,.2f} km^2")
print(f"Output: {OUTPUT_FILE.resolve()}")

print(
    "\nNext step: upload maynas_gaul2024.geojson to the "
    "PERSIANN User Shapefile domain and verify that the "
    "displayed polygon matches Maynas."
)