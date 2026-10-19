#!/usr/bin/env python3
"""
TrafficPulse - Data Processing & Cleaning Module
Performs memory-safe, streaming batch validation on NYC TLC Parquet records.
Computes trip duration, speed, temporal features, and congestion classifications
without loading the entire dataset into memory at once.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

DEFAULT_RAW_PARQUET = Path("data/raw/yellow_tripdata_2024-01.parquet")
DEFAULT_PROCESSED_PARQUET = Path("data/processed/cleaned_trips_2024_01.parquet")
DEFAULT_SAMPLE_PROCESSED = Path("data/processed/cleaned_trips_sample.parquet")
DEFAULT_CLEANING_REPORT = Path("reports/data_cleaning_report.md")

# Data Quality Validation Thresholds
MIN_DURATION_MINUTES = 1.0
MAX_DURATION_MINUTES = 240.0   # 4 hours
MIN_DISTANCE_MILES = 0.1
MAX_DISTANCE_MILES = 100.0
MIN_SPEED_MPH = 1.0
MAX_SPEED_MPH = 65.0           # Urban NYC legal highway limit
MIN_PASSENGERS = 1
MIN_LOCATION_ID = 1
MAX_LOCATION_ID = 265

def categorize_congestion(speed: float) -> str:
    """Categorizes traffic flow into Low, Moderate, or High congestion based on speed."""
    if speed >= 15.0:
        return "Low"
    elif speed >= 8.0:
        return "Moderate"
    else:
        return "High"

def clean_batch(df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    """
    Applies data quality rules to a single chunk/batch and tracks rejection reasons.
    """
    initial_count = len(df)
    metrics["total_raw_rows"] += initial_count

    # 1. Ensure required columns exist
    req_cols = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "trip_distance", "PULocationID", "DOLocationID"
    ]
    for c in req_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column in parquet batch: {c}")

    # Drop nulls in essential columns
    null_mask = df[req_cols].isna().any(axis=1)
    metrics["dropped_nulls"] += int(null_mask.sum())
    df = df[~null_mask].copy()

    # 2. Timestamp sanity: dropoff must be strictly after pickup
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    
    duration_sec = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
    df["trip_duration_minutes"] = duration_sec / 60.0

    invalid_duration_mask = (df["trip_duration_minutes"] < MIN_DURATION_MINUTES) | (df["trip_duration_minutes"] > MAX_DURATION_MINUTES)
    metrics["dropped_duration_outliers"] += int(invalid_duration_mask.sum())
    df = df[~invalid_duration_mask].copy()

    # 3. Distance filters
    df["trip_distance"] = pd.to_numeric(df["trip_distance"], errors="coerce")
    invalid_dist_mask = (df["trip_distance"] < MIN_DISTANCE_MILES) | (df["trip_distance"] > MAX_DISTANCE_MILES)
    metrics["dropped_distance_outliers"] += int(invalid_dist_mask.sum())
    df = df[~invalid_dist_mask].copy()

    # 4. LocationID filters
    invalid_loc_mask = (
        (df["PULocationID"] < MIN_LOCATION_ID) | (df["PULocationID"] > MAX_LOCATION_ID) |
        (df["DOLocationID"] < MIN_LOCATION_ID) | (df["DOLocationID"] > MAX_LOCATION_ID)
    )
    metrics["dropped_invalid_locations"] += int(invalid_loc_mask.sum())
    df = df[~invalid_loc_mask].copy()

    # 5. Fare & Passenger validation if present
    if "fare_amount" in df.columns:
        invalid_fare = (df["fare_amount"] <= 0)
        metrics["dropped_non_positive_fare"] += int(invalid_fare.sum())
        df = df[~invalid_fare].copy()

    if "passenger_count" in df.columns:
        # Fill missing passenger count with 1 (default taxi occupancy)
        df["passenger_count"] = df["passenger_count"].fillna(1).astype(int)
        invalid_passengers = (df["passenger_count"] < MIN_PASSENGERS)
        metrics["dropped_invalid_passengers"] += int(invalid_passengers.sum())
        df = df[~invalid_passengers].copy()

    # 6. Calculate speed (mph)
    df["speed_mph"] = df["trip_distance"] / (df["trip_duration_minutes"] / 60.0)

    # 7. Speed anomaly filter
    invalid_speed = (df["speed_mph"] < MIN_SPEED_MPH) | (df["speed_mph"] > MAX_SPEED_MPH)
    metrics["dropped_speed_outliers"] += int(invalid_speed.sum())
    df = df[~invalid_speed].copy()

    # 8. Feature engineering (Temporal & Congestion)
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour.astype("int32")
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date.astype(str)
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek.astype("int32")
    df["is_weekend"] = (df["pickup_day_of_week"] >= 5).astype("int32")

    # 9. Congestion Target Label (Low: >=15mph, Moderate: 8-15mph, High: <8mph)
    conditions = [
        df["speed_mph"] >= 15.0,
        (df["speed_mph"] >= 8.0) & (df["speed_mph"] < 15.0),
        df["speed_mph"] < 8.0
    ]
    choices = ["Low", "Moderate", "High"]
    df["congestion_level"] = "Moderate"
    for cond, choice in zip(conditions, choices):
        df.loc[cond, "congestion_level"] = choice
    df["congestion_level"] = df["congestion_level"].astype(str)

    # 10. Enforce explicit column types across all batches to ensure Parquet schema uniformity
    type_casts = {
        "VendorID": "int32",
        "PULocationID": "int32",
        "DOLocationID": "int32",
        "passenger_count": "int64",
        "payment_type": "int64",
        "trip_distance": "float64",
        "fare_amount": "float64",
        "extra": "float64",
        "mta_tax": "float64",
        "tip_amount": "float64",
        "tolls_amount": "float64",
        "improvement_surcharge": "float64",
        "total_amount": "float64",
        "congestion_surcharge": "float64",
        "Airport_fee": "float64",
        "trip_duration_minutes": "float64",
        "speed_mph": "float64",
        "RatecodeID": "float64"
    }
    for col, dtype in type_casts.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    metrics["total_cleaned_rows"] += len(df)
    return df

def run_data_cleaning(
    input_path: Path = DEFAULT_RAW_PARQUET,
    output_path: Path = DEFAULT_PROCESSED_PARQUET,
    sample_output_path: Path = DEFAULT_SAMPLE_PROCESSED,
    report_path: Path = DEFAULT_CLEANING_REPORT,
    batch_size: int = 150_000,
    sample_limit: int = 100_000
) -> dict:
    """
    Streams Parquet in batches, cleans each batch, and writes the output incrementally.
    Avoids loading full 3-million row dataset into memory.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    sample_output_path = Path(sample_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input Parquet file not found: {input_path}")

    print("=" * 80)
    print(" TrafficPulse - Streaming Batch Data Cleaning Pipeline")
    print(f" Source:      {input_path}")
    print(f" Destination: {output_path}")
    print(f" Batch Size:  {batch_size:,} records")
    print("=" * 80)

    start_time = time.time()
    metrics = {
        "total_raw_rows": 0,
        "total_cleaned_rows": 0,
        "dropped_nulls": 0,
        "dropped_duration_outliers": 0,
        "dropped_distance_outliers": 0,
        "dropped_invalid_locations": 0,
        "dropped_non_positive_fare": 0,
        "dropped_invalid_passengers": 0,
        "dropped_speed_outliers": 0,
        "batches_processed": 0
    }

    parquet_file = pq.ParquetFile(input_path)
    writer = None
    sample_rows = []
    sample_collected = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            metrics["batches_processed"] += 1
            df_batch = batch.to_pandas()
            cleaned_df = clean_batch(df_batch, metrics)

            if len(cleaned_df) > 0:
                # Convert to Arrow Table
                arrow_table = pa.Table.from_pandas(cleaned_df, preserve_index=False)
                
                # Initialize ParquetWriter on first batch
                if writer is None:
                    writer = pq.ParquetWriter(output_path, arrow_table.schema, compression="snappy")
                writer.write_table(arrow_table)

                # Collect rows for fast sample dataset
                if sample_collected < sample_limit:
                    needed = sample_limit - sample_collected
                    sample_rows.append(cleaned_df.iloc[:needed])
                    sample_collected += min(len(cleaned_df), needed)

            # Progress log
            elapsed = time.time() - start_time
            print(f"[PROGRESS] Batch {metrics['batches_processed']} | Raw: {metrics['total_raw_rows']:,} | Retained: {metrics['total_cleaned_rows']:,} ({metrics['total_cleaned_rows']/max(metrics['total_raw_rows'], 1)*100:.1f}%) | {elapsed:.1f}s")

    finally:
        if writer is not None:
            writer.close()

    # Save fast sample dataset for local analytics and dashboard
    if sample_rows:
        sample_df = pd.concat(sample_rows, ignore_index=True)
        sample_table = pa.Table.from_pandas(sample_df, preserve_index=False)
        pq.write_table(sample_table, sample_output_path, compression="snappy")
        print(f"[SAMPLE] Saved fast local analytics sample ({len(sample_df):,} rows) to: {sample_output_path}")

    duration_sec = time.time() - start_time
    metrics["processing_duration_seconds"] = round(duration_sec, 2)
    metrics["retention_rate_pct"] = round(
        (metrics["total_cleaned_rows"] / max(metrics["total_raw_rows"], 1)) * 100, 2
    )

    # Generate Markdown Audit Report
    generate_cleaning_report(metrics, input_path, output_path, report_path)
    return metrics

def generate_cleaning_report(metrics: dict, src_path: Path, dest_path: Path, report_path: Path):
    """Outputs a comprehensive audit report of data quality decisions."""
    lines = [
        "# TrafficPulse: Data Quality & Cleaning Audit Report",
        "",
        f"- **Source Parquet:** `{src_path}`",
        f"- **Cleaned Parquet:** `{dest_path}`",
        f"- **Execution Duration:** `{metrics['processing_duration_seconds']} seconds`",
        f"- **Total Batches Processed:** `{metrics['batches_processed']}`",
        f"- **Initial Raw Records:** `{metrics['total_raw_rows']:,}`",
        f"- **Retained Clean Records:** `{metrics['total_cleaned_rows']:,}`",
        f"- **Dataset Retention Rate:** **`{metrics['retention_rate_pct']}%`**",
        "",
        "---",
        "",
        "## 1. Data Cleaning Rule Violations Breakdown",
        "",
        "| Rule / Error Category | Condition | Dropped Records | % of Raw Records |",
        "| :--- | :--- | :---: | :---: |",
        f"| **Missing / Null Attributes** | Null in essential trip fields | `{metrics['dropped_nulls']:,}` | `{metrics['dropped_nulls']/metrics['total_raw_rows']*100:.2f}%` |",
        f"| **Duration Outliers** | `duration < 1 min` or `> 240 mins` | `{metrics['dropped_duration_outliers']:,}` | `{metrics['dropped_duration_outliers']/metrics['total_raw_rows']*100:.2f}%` |",
        f"| **Distance Outliers** | `distance < 0.1 mi` or `> 100 mi` | `{metrics['dropped_distance_outliers']:,}` | `{metrics['dropped_distance_outliers']/metrics['total_raw_rows']*100:.2f}%` |",
        f"| **Invalid Taxi Zones** | `PULocationID` / `DOLocationID` $\\notin [1, 265]$ | `{metrics['dropped_invalid_locations']:,}` | `{metrics['dropped_invalid_locations']/metrics['total_raw_rows']*100:.2f}%` |",
        f"| **Non-Positive Fare Amount** | `fare_amount <= 0` | `{metrics['dropped_non_positive_fare']:,}` | `{metrics['dropped_non_positive_fare']/metrics['total_raw_rows']*100:.2f}%` |",
        f"| **Invalid Passenger Counts** | `passenger_count < 1` | `{metrics['dropped_invalid_passengers']:,}` | `{metrics['dropped_invalid_passengers']/metrics['total_raw_rows']*100:.2f}%` |",
        f"| **Unrealistic Speed Anomalies** | `speed < 1 mph` or `> 65 mph` | `{metrics['dropped_speed_outliers']:,}` | `{metrics['dropped_speed_outliers']/metrics['total_raw_rows']*100:.2f}%` |",
        "",
        "---",
        "",
        "## 2. Derived Features Added",
        "- **`trip_duration_minutes`**: Floating point duration computed from dropoff minus pickup timestamp.",
        "- **`speed_mph`**: Calculated average velocity in miles per hour ($d / (t / 60)$).",
        "- **`pickup_hour`**: Integer (0 to 23) representing diurnal time slice.",
        "- **`pickup_date`**: Calendar date (YYYY-MM-DD).",
        "- **`pickup_day_of_week`**: Day index (0 = Monday, 6 = Sunday).",
        "- **`is_weekend`**: Binary flag (1 for Saturday/Sunday, 0 for weekdays).",
        "- **`congestion_level`**: Transparent target label (`Low`: $\\ge 15\\text{ mph}$, `Moderate`: $8-15\\text{ mph}$, `High`: $< 8\\text{ mph}$)."
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[AUDIT] Cleaning audit report generated at: {report_path}")

if __name__ == "__main__":
    run_data_cleaning()
