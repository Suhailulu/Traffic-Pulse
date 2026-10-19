#!/usr/bin/env python3
"""
TrafficPulse - Basic Traffic Analytics Module
Processes NYC TLC Parquet records in memory-safe PyArrow streaming batches.
Computes trip volumes by hour, peak hours, average durations, pickup zone hotspots,
and estimated speeds without loading the entire dataset into memory at once.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

DEFAULT_RAW_PARQUET = Path("data/raw/yellow_tripdata_2024-01.parquet")
DEFAULT_LOOKUP_CSV = Path("data/raw/taxi_zone_lookup.csv")
DEFAULT_HOURLY_CSV = Path("data/processed/hourly_traffic_summary.csv")
DEFAULT_ZONE_CSV = Path("data/processed/zone_traffic_summary.csv")
DEFAULT_SUMMARY_JSON = Path("reports/traffic_analysis_report.json")

# Data Quality Thresholds for Traffic Analysis
MIN_DURATION_MINUTES = 1.0
MAX_DURATION_MINUTES = 240.0   # 4 hours
MIN_DISTANCE_MILES = 0.1
MAX_DISTANCE_MILES = 100.0
MIN_SPEED_MPH = 1.0
MAX_SPEED_MPH = 65.0           # Urban legal limit for NYC arterial roads

def load_zone_lookup(lookup_path: Path = DEFAULT_LOOKUP_CSV) -> dict:
    """Loads taxi zone metadata into a dictionary mapping LocationID to (Zone, Borough)."""
    lookup_path = Path(lookup_path)
    if not lookup_path.exists():
        return {}
    df = pd.read_csv(lookup_path)
    lookup = {}
    for _, row in df.iterrows():
        try:
            loc_id = int(row["LocationID"])
            zone = str(row["Zone"]) if pd.notna(row["Zone"]) else "Unknown"
            borough = str(row["Borough"]) if pd.notna(row["Borough"]) else "Unknown"
            lookup[loc_id] = {"zone": zone, "borough": borough}
        except (ValueError, TypeError):
            continue
    return lookup

def validate_and_extract_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Validates trip records in a batch, filters out invalid observations,
    and returns the cleaned slice alongside audit counters.
    """
    stats = {
        "batch_raw_count": len(df),
        "invalid_timestamps": 0,
        "invalid_durations": 0,
        "invalid_distances": 0,
        "invalid_speeds": 0,
        "valid_count": 0
    }

    if df.empty:
        return pd.DataFrame(), stats

    # 1. Null check on critical fields
    required = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "PULocationID"]
    df = df.dropna(subset=required).copy()
    if df.empty:
        return pd.DataFrame(), stats

    # 2. Timestamp validation & duration calculation
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])

    # Dropoff must be after pickup
    valid_ts = df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"]
    stats["invalid_timestamps"] = int((~valid_ts).sum())
    df = df[valid_ts].copy()
    if df.empty:
        return pd.DataFrame(), stats

    duration_sec = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
    df["trip_duration_minutes"] = duration_sec / 60.0

    valid_dur = (df["trip_duration_minutes"] >= MIN_DURATION_MINUTES) & (df["trip_duration_minutes"] <= MAX_DURATION_MINUTES)
    stats["invalid_durations"] = int((~valid_dur).sum())
    df = df[valid_dur].copy()
    if df.empty:
        return pd.DataFrame(), stats

    # 3. Distance validation
    df["trip_distance"] = pd.to_numeric(df["trip_distance"], errors="coerce")
    valid_dist = (df["trip_distance"] >= MIN_DISTANCE_MILES) & (df["trip_distance"] <= MAX_DISTANCE_MILES)
    stats["invalid_distances"] = int((~valid_dist).sum())
    df = df[valid_dist].copy()
    if df.empty:
        return pd.DataFrame(), stats

    # 4. Estimated speed calculation (mph)
    df["speed_mph"] = df["trip_distance"] / (df["trip_duration_minutes"] / 60.0)
    valid_speed = (df["speed_mph"] >= MIN_SPEED_MPH) & (df["speed_mph"] <= MAX_SPEED_MPH)
    stats["invalid_speeds"] = int((~valid_speed).sum())
    df = df[valid_speed].copy()

    # 5. Extract pickup hour
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour.astype(int)
    df["PULocationID"] = df["PULocationID"].astype(int)

    stats["valid_count"] = len(df)
    return df, stats

def run_traffic_analysis(
    input_path: Path = DEFAULT_RAW_PARQUET,
    lookup_path: Path = DEFAULT_LOOKUP_CSV,
    hourly_output_path: Path = DEFAULT_HOURLY_CSV,
    zone_output_path: Path = DEFAULT_ZONE_CSV,
    summary_report_path: Path = DEFAULT_SUMMARY_JSON,
    batch_size: int = 200_000
) -> dict:
    """
    Executes memory-safe streaming traffic analysis across all batches.
    Aggregates hourly distributions, peak hours, durations, and zone volumes.
    """
    input_path = Path(input_path)
    hourly_output_path = Path(hourly_output_path)
    zone_output_path = Path(zone_output_path)
    summary_report_path = Path(summary_report_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input Parquet dataset not found: {input_path}")

    hourly_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_report_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" TrafficPulse - Streaming Batch Traffic Analytics Engine")
    print(f" Source Parquet: {input_path}")
    print(f" Batch Size:     {batch_size:,} records")
    print("=" * 80)

    start_time = time.time()
    zone_lookup = load_zone_lookup(lookup_path)

    # Global Accumulators
    total_raw_records = 0
    total_valid_trips = 0
    total_duration_sum = 0.0
    total_speed_sum = 0.0
    total_distance_sum = 0.0

    # Hourly aggregators: hour (0..23) -> {count, duration_sum, speed_sum, distance_sum}
    hourly_stats = {
        h: {"trip_count": 0, "duration_sum": 0.0, "speed_sum": 0.0, "distance_sum": 0.0}
        for h in range(24)
    }

    # Zone aggregators: PULocationID -> {count, duration_sum, speed_sum, distance_sum}
    zone_stats = {}

    audit_totals = {
        "invalid_timestamps": 0,
        "invalid_durations": 0,
        "invalid_distances": 0,
        "invalid_speeds": 0
    }

    parquet_file = pq.ParquetFile(input_path)
    batch_idx = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_idx += 1
        df_batch = batch.to_pandas()
        total_raw_records += len(df_batch)

        cleaned_df, stats = validate_and_extract_metrics(df_batch)
        for k in audit_totals:
            audit_totals[k] += stats[k]

        if cleaned_df.empty:
            continue

        n_valid = len(cleaned_df)
        total_valid_trips += n_valid
        total_duration_sum += float(cleaned_df["trip_duration_minutes"].sum())
        total_speed_sum += float(cleaned_df["speed_mph"].sum())
        total_distance_sum += float(cleaned_df["trip_distance"].sum())

        # Group by hour within batch
        hourly_grp = cleaned_df.groupby("pickup_hour").agg(
            trip_count=("trip_duration_minutes", "count"),
            duration_sum=("trip_duration_minutes", "sum"),
            speed_sum=("speed_mph", "sum"),
            distance_sum=("trip_distance", "sum")
        ).reset_index()

        for _, row in hourly_grp.iterrows():
            h = int(row["pickup_hour"])
            hourly_stats[h]["trip_count"] += int(row["trip_count"])
            hourly_stats[h]["duration_sum"] += float(row["duration_sum"])
            hourly_stats[h]["speed_sum"] += float(row["speed_sum"])
            hourly_stats[h]["distance_sum"] += float(row["distance_sum"])

        # Group by pickup zone within batch
        zone_grp = cleaned_df.groupby("PULocationID").agg(
            trip_count=("trip_duration_minutes", "count"),
            duration_sum=("trip_duration_minutes", "sum"),
            speed_sum=("speed_mph", "sum"),
            distance_sum=("trip_distance", "sum")
        ).reset_index()

        for _, row in zone_grp.iterrows():
            zid = int(row["PULocationID"])
            if zid not in zone_stats:
                zone_stats[zid] = {"trip_count": 0, "duration_sum": 0.0, "speed_sum": 0.0, "distance_sum": 0.0}
            zone_stats[zid]["trip_count"] += int(row["trip_count"])
            zone_stats[zid]["duration_sum"] += float(row["duration_sum"])
            zone_stats[zid]["speed_sum"] += float(row["speed_sum"])
            zone_stats[zid]["distance_sum"] += float(row["distance_sum"])

        print(f"[PROGRESS] Batch {batch_idx:02d} | Valid trips accumulated: {total_valid_trips:,} | {time.time() - start_time:.1f}s")

    elapsed_time = time.time() - start_time

    # 1. Compile Hourly Summary Table
    hourly_rows = []
    for h in range(24):
        c = hourly_stats[h]["trip_count"]
        dur = round(hourly_stats[h]["duration_sum"] / c, 2) if c > 0 else 0.0
        spd = round(hourly_stats[h]["speed_sum"] / c, 2) if c > 0 else 0.0
        dist = round(hourly_stats[h]["distance_sum"] / c, 2) if c > 0 else 0.0
        hourly_rows.append({
            "hour": h,
            "trip_count": c,
            "avg_duration_minutes": dur,
            "avg_speed_mph": spd,
            "avg_distance_miles": dist
        })
    df_hourly = pd.DataFrame(hourly_rows)
    df_hourly.to_csv(hourly_output_path, index=False)
    print(f"\n[OUTPUT] Saved hourly traffic summary to: {hourly_output_path}")

    # 2. Compile Zone Summary Table
    zone_rows = []
    for zid, zdata in zone_stats.items():
        c = zdata["trip_count"]
        dur = round(zdata["duration_sum"] / c, 2) if c > 0 else 0.0
        spd = round(zdata["speed_sum"] / c, 2) if c > 0 else 0.0
        dist = round(zdata["distance_sum"] / c, 2) if c > 0 else 0.0
        zinfo = zone_lookup.get(zid, {"zone": "Unknown", "borough": "Unknown"})
        zone_rows.append({
            "PULocationID": zid,
            "zone_name": zinfo["zone"],
            "borough": zinfo["borough"],
            "trip_count": c,
            "avg_duration_minutes": dur,
            "avg_speed_mph": spd,
            "avg_distance_miles": dist
        })
    df_zone = pd.DataFrame(zone_rows)
    df_zone = df_zone.sort_values(by="trip_count", ascending=False).reset_index(drop=True)
    df_zone.to_csv(zone_output_path, index=False)
    print(f"[OUTPUT] Saved zone traffic summary to: {zone_output_path}")

    # 3. Compute Macro Key Performance Indicators (KPIs)
    overall_avg_duration = round(total_duration_sum / total_valid_trips, 2) if total_valid_trips > 0 else 0.0
    overall_avg_speed = round(total_speed_sum / total_valid_trips, 2) if total_valid_trips > 0 else 0.0
    overall_avg_distance = round(total_distance_sum / total_valid_trips, 2) if total_valid_trips > 0 else 0.0

    peak_row = df_hourly.loc[df_hourly["trip_count"].idxmax()] if not df_hourly.empty else {}
    low_row = df_hourly.loc[df_hourly["trip_count"].idxmin()] if not df_hourly.empty else {}

    top_5_zones = df_zone.head(5).to_dict(orient="records") if not df_zone.empty else []

    # 4. Generate JSON Summary Report
    summary_report = {
        "execution_metadata": {
            "source_parquet": str(input_path),
            "executed_at": datetime.now().isoformat(),
            "processing_duration_seconds": round(elapsed_time, 2),
            "total_raw_records": total_raw_records,
            "total_valid_trips": total_valid_trips,
            "retention_rate_pct": round((total_valid_trips / max(total_raw_records, 1)) * 100, 2)
        },
        "audit_rejections": audit_totals,
        "traffic_kpis": {
            "overall_avg_duration_minutes": overall_avg_duration,
            "overall_avg_speed_mph": overall_avg_speed,
            "overall_avg_distance_miles": overall_avg_distance,
            "peak_travel_hour": int(peak_row.get("hour", 0)),
            "peak_hour_trips": int(peak_row.get("trip_count", 0)),
            "lowest_travel_hour": int(low_row.get("hour", 0)),
            "lowest_hour_trips": int(low_row.get("trip_count", 0))
        },
        "top_pickup_hotspots": top_5_zones,
        "hourly_distribution": df_hourly.to_dict(orient="records"),
        "methodology_limitations": [
            "Taxi trip duration reflects commercial point-to-point transit times rather than continuous highway lane sensor counts.",
            "Average speeds include boarding/alighting delays, traffic light dwell times, and street grid turns.",
            "Severe speed drops serve as an empirical indicator of corridor friction rather than vehicle density."
        ]
    }

    with open(summary_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[OUTPUT] Saved executive traffic analysis report to: {summary_report_path}")

    # Print summary highlights to terminal
    print("\n" + "=" * 80)
    print(" TRAFFIC ANALYSIS RESULTS SUMMARY")
    print("=" * 80)
    print(f"  - Total Raw Trips Processed:     {total_raw_records:,}")
    print(f"  - Total Valid Trips Analyzed:     {total_valid_trips:,} ({summary_report['execution_metadata']['retention_rate_pct']}%)")
    print(f"  - Peak Travel Hour:               Hour {int(peak_row['hour'])}:00 ({int(peak_row['trip_count']):,} trips)")
    print(f"  - Lowest Travel Hour:             Hour {int(low_row['hour'])}:00 ({int(low_row['trip_count']):,} trips)")
    print(f"  - Overall Average Trip Duration:  {overall_avg_duration} minutes")
    print(f"  - Overall Estimated Average Speed:{overall_avg_speed} mph")
    print(f"  - Overall Average Trip Distance:  {overall_avg_distance} miles")
    print("\n  Top 5 High-Demand Pickup Zones:")
    for idx, z in enumerate(top_5_zones, 1):
        print(f"    {idx}. Zone {z['PULocationID']}: {z['zone_name']} ({z['borough']}) - {z['trip_count']:,} trips | Avg Speed: {z['avg_speed_mph']} mph")
    print("=" * 80)

    return summary_report

if __name__ == "__main__":
    run_traffic_analysis()
