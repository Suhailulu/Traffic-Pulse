"""
TrafficPulse - Traffic Analytics Unit Tests
Tests timestamp validation, trip duration calculation, invalid distance handling,
hourly aggregation, estimated speed, and empty dataset edge cases.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from src.analytics.traffic_analysis import (
    validate_and_extract_metrics,
    run_traffic_analysis,
    load_zone_lookup
)

def make_record(
    pickup_time="2024-01-10 14:00:00",
    duration_min=20.0,
    distance_mi=4.0,
    puloc=161
):
    pickup = datetime.fromisoformat(pickup_time)
    dropoff = pickup + timedelta(minutes=duration_min)
    return {
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": dropoff,
        "trip_distance": distance_mi,
        "PULocationID": puloc
    }

# ==============================================================================
# 1. Validation & Calculation Unit Tests
# ==============================================================================

def test_valid_trip_extraction():
    """Verifies that a normal valid trip computes exact duration and speed."""
    df = pd.DataFrame([make_record(duration_min=30.0, distance_mi=6.0)])
    cleaned, stats = validate_and_extract_metrics(df)

    assert len(cleaned) == 1
    assert stats["valid_count"] == 1
    row = cleaned.iloc[0]
    assert row["trip_duration_minutes"] == 30.0
    assert row["speed_mph"] == 12.0  # (6 miles / 0.5 hours = 12 mph)
    assert row["pickup_hour"] == 14

def test_timestamp_validation_dropoff_before_pickup():
    """Verifies that reversed or zero timestamps are rejected."""
    t0 = datetime(2024, 1, 10, 10, 0, 0)
    df = pd.DataFrame([
        {
            "tpep_pickup_datetime": t0,
            "tpep_dropoff_datetime": t0 - timedelta(minutes=5),  # Negative duration
            "trip_distance": 2.0,
            "PULocationID": 161
        },
        {
            "tpep_pickup_datetime": t0,
            "tpep_dropoff_datetime": t0,  # Zero duration
            "trip_distance": 2.0,
            "PULocationID": 161
        }
    ])
    cleaned, stats = validate_and_extract_metrics(df)
    assert len(cleaned) == 0
    assert stats["invalid_timestamps"] == 2

def test_trip_duration_outliers():
    """Verifies that trips < 1 min or > 4 hours are rejected."""
    df = pd.DataFrame([
        make_record(duration_min=0.3, distance_mi=0.5),   # Too short
        make_record(duration_min=360.0, distance_mi=15.0), # Too long (6 hrs)
        make_record(duration_min=15.0, distance_mi=3.0)   # Valid (15 mins)
    ])
    cleaned, stats = validate_and_extract_metrics(df)
    assert len(cleaned) == 1
    assert stats["invalid_durations"] == 2
    assert cleaned.iloc[0]["trip_duration_minutes"] == 15.0

def test_invalid_distance_handling():
    """Verifies that 0, negative, and excessive distances (> 100 miles) are rejected."""
    df = pd.DataFrame([
        make_record(distance_mi=0.0),    # Zero distance
        make_record(distance_mi=-2.5),   # Negative distance
        make_record(distance_mi=150.0),  # Excessive distance
        make_record(distance_mi=5.0)     # Valid
    ])
    cleaned, stats = validate_and_extract_metrics(df)
    assert len(cleaned) == 1
    assert stats["invalid_distances"] == 3
    assert cleaned.iloc[0]["trip_distance"] == 5.0

def test_unrealistic_speed_outliers():
    """Verifies that speeds > 65 mph or < 1 mph are rejected."""
    df = pd.DataFrame([
        make_record(duration_min=10.0, distance_mi=15.0), # 90 mph -> dropped
        make_record(duration_min=60.0, distance_mi=0.3),  # 0.3 mph -> dropped
        make_record(duration_min=20.0, distance_mi=4.0)   # 12.0 mph -> kept
    ])
    cleaned, stats = validate_and_extract_metrics(df)
    assert len(cleaned) == 1
    assert stats["invalid_speeds"] == 2
    assert cleaned.iloc[0]["speed_mph"] == 12.0

def test_empty_dataframe_handling():
    """Verifies that empty dataframes return cleanly without raising errors."""
    empty_df = pd.DataFrame()
    cleaned, stats = validate_and_extract_metrics(empty_df)
    assert cleaned.empty
    assert stats["batch_raw_count"] == 0
    assert stats["valid_count"] == 0

# ==============================================================================
# 2. End-to-End Pipeline & Hourly Aggregation Test
# ==============================================================================

def test_hourly_and_zone_aggregation_pipeline(tmp_path):
    """Tests end-to-end aggregation, peak hour discovery, and file persistence."""
    raw_file = tmp_path / "test_raw.parquet"
    lookup_file = tmp_path / "test_lookup.csv"
    hourly_out = tmp_path / "hourly_summary.csv"
    zone_out = tmp_path / "zone_summary.csv"
    report_out = tmp_path / "report.json"

    # Create mock lookup
    lookup_file.write_text("LocationID,Borough,Zone,service_zone\n161,Manhattan,Midtown Center,Yellow Zone\n237,Manhattan,Upper East Side South,Yellow Zone\n", encoding="utf-8")

    # Generate synthetic trips across two hours (Hour 9 and Hour 18)
    records = [
        # Hour 9: 1 trip
        make_record(pickup_time="2024-01-10 09:15:00", duration_min=15.0, distance_mi=3.0, puloc=161),
        # Hour 18: 3 trips (Peak Hour)
        make_record(pickup_time="2024-01-10 18:10:00", duration_min=20.0, distance_mi=4.0, puloc=161),
        make_record(pickup_time="2024-01-10 18:25:00", duration_min=30.0, distance_mi=5.0, puloc=237),
        make_record(pickup_time="2024-01-10 18:40:00", duration_min=10.0, distance_mi=2.5, puloc=161),
        # Invalid trip (should be excluded)
        make_record(pickup_time="2024-01-10 18:00:00", duration_min=0.2, distance_mi=0.0, puloc=161)
    ]
    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, raw_file)

    report = run_traffic_analysis(
        input_path=raw_file,
        lookup_path=lookup_file,
        hourly_output_path=hourly_out,
        zone_output_path=zone_out,
        summary_report_path=report_out,
        batch_size=10
    )

    assert report["execution_metadata"]["total_valid_trips"] == 4
    assert report["traffic_kpis"]["peak_travel_hour"] == 18
    assert report["traffic_kpis"]["peak_hour_trips"] == 3
    assert hourly_out.exists()
    assert zone_out.exists()
    assert report_out.exists()

    df_h = pd.read_csv(hourly_out)
    assert len(df_h) == 24
    h18 = df_h[df_h["hour"] == 18].iloc[0]
    assert h18["trip_count"] == 3
    assert h18["avg_duration_minutes"] == 20.0  # (20 + 30 + 10) / 3 = 20.0
