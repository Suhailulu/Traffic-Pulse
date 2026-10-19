#!/usr/bin/env python3
"""
TrafficPulse - Parquet Schema Inspector
Leverages PyArrow's ParquetFile interface to inspect multi-million row
Parquet files directly from file metadata without full memory loading.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq

DEFAULT_PARQUET_PATH = Path("data/raw/yellow_tripdata_2024-01.parquet")
DEFAULT_REPORT_PATH = Path("reports/dataset_metadata_report.md")

def inspect_parquet_schema(parquet_path: Path) -> dict:
    """
    Inspects a Parquet file using pyarrow.parquet.ParquetFile metadata.
    Avoids reading data pages or loading dataframes into memory.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")

    file_size_bytes = parquet_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    try:
        parquet_file = pq.ParquetFile(parquet_path)
    except Exception as e:
        raise ValueError(f"Failed to parse Parquet file metadata: {e}")

    meta = parquet_file.metadata
    schema = parquet_file.schema_arrow

    num_rows = meta.num_rows
    num_columns = meta.num_columns
    num_row_groups = meta.num_row_groups
    created_by = meta.created_by or "Unavailable"
    format_version = meta.format_version or "Unavailable"

    # Analyze column definitions & types
    columns_info = []
    field_names = []
    timestamp_fields = []
    location_fields = []
    distance_fields = []
    passenger_fields = []
    fare_fields = []
    other_fields = []

    for idx in range(num_columns):
        col_meta = meta.row_group(0).column(idx) if num_row_groups > 0 else None
        field = schema.field(idx)
        col_name = field.name
        arrow_type = str(field.type)
        physical_type = col_meta.physical_type if col_meta else "Unavailable"
        field_names.append(col_name)

        col_entry = {
            "index": idx,
            "name": col_name,
            "arrow_type": arrow_type,
            "physical_type": physical_type,
            "nullable": field.nullable
        }
        columns_info.append(col_entry)

        # Semantic categorization
        col_lower = col_name.lower()
        if "datetime" in col_lower or "time" in col_lower:
            timestamp_fields.append(col_name)
        elif "locationid" in col_lower or "pulocation" in col_lower or "dolocation" in col_lower or "zone" in col_lower:
            location_fields.append(col_name)
        elif "distance" in col_lower:
            distance_fields.append(col_name)
        elif "passenger" in col_lower:
            passenger_fields.append(col_name)
        elif "amount" in col_lower or "fare" in col_lower or "tax" in col_lower or "tip" in col_lower or "surcharge" in col_lower or "fee" in col_lower:
            fare_fields.append(col_name)
        else:
            other_fields.append(col_name)

    # Row-group summary
    row_groups_summary = []
    for rg_idx in range(num_row_groups):
        rg_meta = meta.row_group(rg_idx)
        row_groups_summary.append({
            "row_group_index": rg_idx,
            "num_rows": rg_meta.num_rows,
            "total_byte_size": rg_meta.total_byte_size,
            "total_mb": round(rg_meta.total_byte_size / (1024 * 1024), 2)
        })

    report = {
        "file_path": str(parquet_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "total_rows": num_rows,
        "num_columns": num_columns,
        "num_row_groups": num_row_groups,
        "created_by": created_by,
        "format_version": format_version,
        "fields": {
            "timestamp_fields": timestamp_fields,
            "location_fields": location_fields,
            "distance_fields": distance_fields,
            "passenger_fields": passenger_fields,
            "fare_fields": fare_fields,
            "other_fields": other_fields
        },
        "columns": columns_info,
        "row_groups": row_groups_summary,
        "inspected_at": datetime.now().isoformat()
    }
    return report

def generate_markdown_report(report_dict: dict, output_path: Path):
    """Formats inspection output into a clean GitHub Flavored Markdown document."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# TrafficPulse: NYC TLC Trip Record Dataset Schema Inspection Report",
        "",
        f"- **File Inspected:** `{report_dict['file_path']}`",
        f"- **File Size:** `{report_dict['file_size_mb']} MB` (`{report_dict['file_size_bytes']:,}` bytes)",
        f"- **Total Rows (Metadata):** `{report_dict['total_rows']:,}` rows",
        f"- **Total Columns:** `{report_dict['num_columns']}`",
        f"- **Total Row Groups:** `{report_dict['num_row_groups']}`",
        f"- **Parquet Format Version:** `{report_dict['format_version']}`",
        f"- **Generated / Written By:** `{report_dict['created_by']}`",
        f"- **Inspection Timestamp:** `{report_dict['inspected_at']}`",
        "",
        "---",
        "",
        "## 1. Domain Field Categorization",
        "",
        "| Category | Identified Columns | Description / Analytical Role |",
        "| :--- | :--- | :--- |",
        f"| **Timestamps** | `{', '.join(report_dict['fields']['timestamp_fields']) or 'None'}` | Temporal granularity for diurnal cycle, day-of-week, and rush hour extraction |",
        f"| **Spatial Zones** | `{', '.join(report_dict['fields']['location_fields']) or 'None'}` | Pickup (PULocationID) and Dropoff (DOLocationID) taxi zone identifiers |",
        f"| **Trip Distance** | `{', '.join(report_dict['fields']['distance_fields']) or 'None'}` | Core vehicle travel distance in miles for speed calculation |",
        f"| **Passenger Capacity** | `{', '.join(report_dict['fields']['passenger_fields']) or 'None'}` | Occupancy load factor |",
        f"| **Fares & Surcharges** | `{', '.join(report_dict['fields']['fare_fields']) or 'None'}` | Monetary fields (congestion surcharges, airport fees, base fare) |",
        f"| **Operational / Metadata** | `{', '.join(report_dict['fields']['other_fields']) or 'None'}` | VendorID, RatecodeID, payment_type, store_and_fwd_flag |",
        "",
        "---",
        "",
        "## 2. Complete Column Specifications",
        "",
        "| # | Column Name | Arrow Data Type | Physical Parquet Type | Nullable |",
        "| :---: | :--- | :--- | :--- | :---: |"
    ]

    for col in report_dict["columns"]:
        lines.append(
            f"| {col['index']} | **`{col['name']}`** | `{col['arrow_type']}` | `{col['physical_type']}` | {col['nullable']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Row Group Storage Layout",
        "",
        "| Row Group Index | Row Count | Uncompressed Size (MB) |",
        "| :---: | :---: | :---: |"
    ])

    for rg in report_dict["row_groups"]:
        lines.append(f"| {rg['row_group_index']} | {rg['num_rows']:,} | {rg['total_mb']} MB |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Big Data & Memory Efficiency Notes",
        "- **Zero RAM Loading:** All attributes in this report were extracted directly from the Parquet FileMetaData footer.",
        "- **Distributed Partitioning Strategy:** With PySpark / Hive, partition pruning should target `PULocationID` or date slices derived from `tpep_pickup_datetime`.",
        "- **Data Leakage Safeguard:** Fare-related fields and actual trip duration are strictly post-trip metrics and will be excluded from the predictive congestion feature set."
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[REPORT] Schema inspection report written to: {output_path}")

def run_inspector(parquet_path: Path = DEFAULT_PARQUET_PATH, report_path: Path = DEFAULT_REPORT_PATH):
    """Executes schema inspection and generates report."""
    print("=" * 80)
    print(" TrafficPulse - Parquet Schema Inspector")
    print(f" Target File: {parquet_path}")
    print("=" * 80)

    report = inspect_parquet_schema(parquet_path)
    
    print("\n[METADATA SUMMARY]")
    print(f"  - Total Rows:       {report['total_rows']:,}")
    print(f"  - Total Columns:    {report['num_columns']}")
    print(f"  - File Size:        {report['file_size_mb']} MB")
    print(f"  - Row Groups:       {report['num_row_groups']}")
    print(f"  - Timestamp Fields: {report['fields']['timestamp_fields']}")
    print(f"  - Location Fields:  {report['fields']['location_fields']}")
    print(f"  - Distance Fields:  {report['fields']['distance_fields']}")

    generate_markdown_report(report, report_path)
    return report

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PARQUET_PATH
    run_inspector(target)
