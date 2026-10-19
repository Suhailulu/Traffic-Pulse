#!/usr/bin/env python3
"""
TrafficPulse - Taxi Zone & Geospatial Inspector
Validates spatial join integrity between NYC TLC trip records (PULocationID, DOLocationID),
the official taxi zone lookup CSV, and the GeoJSON spatial boundaries.
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

DEFAULT_LOOKUP_PATH = Path("data/raw/taxi_zone_lookup.csv")
DEFAULT_GEOJSON_PATH = Path("data/raw/taxi_zones.geojson")
DEFAULT_GEO_REPORT_PATH = Path("reports/taxi_zones_spatial_report.md")

def inspect_taxi_zones(lookup_path: Path = DEFAULT_LOOKUP_PATH, geojson_path: Path = DEFAULT_GEOJSON_PATH) -> dict:
    """
    Analyzes taxi zone lookup table and GeoJSON boundaries to verify joinability.
    """
    lookup_path = Path(lookup_path)
    geojson_path = Path(geojson_path)

    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup file missing: {lookup_path}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file missing: {geojson_path}")

    # 1. Inspect Lookup CSV
    df_lookup = pd.read_csv(lookup_path)
    lookup_columns = list(df_lookup.columns)
    total_lookup_zones = len(df_lookup)
    
    # Ensure LocationID is integer
    df_lookup["LocationID"] = pd.to_numeric(df_lookup["LocationID"], errors="coerce")
    lookup_ids = set(df_lookup["LocationID"].dropna().astype(int))

    # Borough breakdown
    borough_counts = df_lookup["Borough"].value_counts().to_dict()

    # 2. Inspect GeoJSON Boundaries
    with open(geojson_path, "r", encoding="utf-8") as f:
        geo_data = json.load(f)

    features = geo_data.get("features", [])
    total_features = len(features)

    # Detect the location ID key in GeoJSON properties (case-insensitive)
    geojson_ids = set()
    sample_props = features[0].get("properties", {}) if features else {}
    id_key_candidate = None
    
    # Check lowercase normalized keys
    normalized_keys = {k.lower(): k for k in sample_props.keys()}
    for target in ["locationid", "location_id", "objectid", "zone_id", "id"]:
        if target in normalized_keys:
            id_key_candidate = normalized_keys[target]
            break

    if id_key_candidate:
        for feat in features:
            val = feat.get("properties", {}).get(id_key_candidate)
            if val is not None:
                try:
                    geojson_ids.add(int(val))
                except (ValueError, TypeError):
                    pass

    # Compare ID Sets
    in_lookup_not_in_geo = sorted(list(lookup_ids - geojson_ids))
    in_geo_not_in_lookup = sorted(list(geojson_ids - lookup_ids))

    report = {
        "lookup_path": str(lookup_path),
        "geojson_path": str(geojson_path),
        "lookup_columns": lookup_columns,
        "total_lookup_zones": total_lookup_zones,
        "total_geojson_features": total_features,
        "geojson_id_property_key": id_key_candidate,
        "borough_distribution": borough_counts,
        "lookup_id_range": [min(lookup_ids), max(lookup_ids)] if lookup_ids else [None, None],
        "geojson_id_range": [min(geojson_ids), max(geojson_ids)] if geojson_ids else [None, None],
        "unmapped_zones_in_geojson": in_lookup_not_in_geo,
        "extra_zones_in_geojson": in_geo_not_in_lookup,
        "inspected_at": datetime.now().isoformat()
    }
    return report

def generate_spatial_markdown_report(report_dict: dict, output_path: Path):
    """Outputs a markdown summary of the geospatial reconciliation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unmapped_desc = ", ".join(map(str, report_dict["unmapped_zones_in_geojson"])) or "None"
    
    lines = [
        "# TrafficPulse: Taxi Zone & Spatial Boundary Reconciliation Report",
        "",
        f"- **Lookup Table:** `{report_dict['lookup_path']}` ({report_dict['total_lookup_zones']} records)",
        f"- **GeoJSON Boundaries:** `{report_dict['geojson_path']}` ({report_dict['total_geojson_features']} polygon features)",
        f"- **Identified GeoJSON Join Key:** `{report_dict['geojson_id_property_key']}`",
        f"- **Lookup LocationID Range:** `{report_dict['lookup_id_range'][0]}` to `{report_dict['lookup_id_range'][1]}`",
        f"- **Inspection Timestamp:** `{report_dict['inspected_at']}`",
        "",
        "---",
        "",
        "## 1. Join Key Architecture",
        "",
        "| Dataset | Foreign / Primary Key | Data Type | Notes |",
        "| :--- | :--- | :--- | :--- |",
        "| **TLC Trip Records** | `PULocationID`, `DOLocationID` | `int64` / `int32` | Taxi pickup and dropoff zone references |",
        "| **Taxi Zone Lookup** | `LocationID` | `int64` | Primary key for zone names and boroughs |",
        f"| **Taxi Zone GeoJSON** | `{report_dict['geojson_id_property_key']}` | `integer` / `string` | Spatial boundary polygon mapping |",
        "",
        "---",
        "",
        "## 2. Borough Distribution (Lookup Table)",
        "",
        "| Borough | Total Designated Taxi Zones |",
        "| :--- | :---: |"
    ]

    for b, c in report_dict["borough_distribution"].items():
        lines.append(f"| **{b}** | {c} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Discrepancy Analysis & Missing Boundaries",
        f"- **Zones in Lookup without GeoJSON Polygons ({len(report_dict['unmapped_zones_in_geojson'])} zones):** `{unmapped_desc}`",
        "  - *Reasoning:* Zone 264 represents 'Unknown / Outside NYC' and Zone 265 represents 'NV (Non-NYC / Out of State)'. These do not possess physical polygons in NYC territory.",
        "- **Zones in GeoJSON without Lookup Match:** " + (f"`{report_dict['extra_zones_in_geojson']}`" if report_dict['extra_zones_in_geojson'] else "`None (100% matched)`"),
        "",
        "---",
        "",
        "## 4. Methodological & Spatial Proxy Limitations",
        "> [!IMPORTANT]",
        "> **Traffic Analysis Limitation:** Taxi zones are irregular polygonal neighborhood boundaries (derived from NYC DCP Neighborhood Tabulation Areas), **not** individual road links, street centerlines, or highway segments.",
        "> ",
        "> When computing speed and congestion metrics:",
        "> 1. A trip from `PULocationID` to `DOLocationID` represents an **aggregate OD corridor travel time**, not a single street's level-of-service.",
        "> 2. Speed reflects commercial urban trip velocity (including traffic lights, passenger boarding, and street routing).",
        "> 3. Intra-zone trips (`PULocationID == DOLocationID`) reflect local circulating traffic within the neighborhood."
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[REPORT] Spatial inspection report written to: {output_path}")

def run_spatial_inspector(lookup_path: Path = DEFAULT_LOOKUP_PATH, geojson_path: Path = DEFAULT_GEOJSON_PATH, report_path: Path = DEFAULT_GEO_REPORT_PATH):
    """Entry point for geospatial verification."""
    print("=" * 80)
    print(" TrafficPulse - Geospatial & Taxi Zone Inspector")
    print(f" Lookup Path:  {lookup_path}")
    print(f" GeoJSON Path: {geojson_path}")
    print("=" * 80)

    report = inspect_taxi_zones(lookup_path, geojson_path)
    print("\n[SPATIAL RECONCILIATION SUMMARY]")
    print(f"  - Total Lookup Records:    {report['total_lookup_zones']}")
    print(f"  - Total GeoJSON Polygons:  {report['total_geojson_features']}")
    print(f"  - GeoJSON Join Key:        {report['geojson_id_property_key']}")
    print(f"  - Non-spatial Zones (NV):  {report['unmapped_zones_in_geojson']}")

    generate_spatial_markdown_report(report, report_path)
    return report

if __name__ == "__main__":
    run_spatial_inspector()
