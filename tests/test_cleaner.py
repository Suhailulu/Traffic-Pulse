"""
TrafficPulse - Data Cleaning & Processing Unit Tests
Tests outlier removal, duration/speed calculations, temporal feature extraction,
and congestion categorization rules.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from src.processing.cleaner import clean_batch, categorize_congestion, run_data_cleaning

def make_test_row(
    pickup_str="2024-01-15 08:30:00",
    duration_min=15.0,
    distance_mi=3.0,
    puloc=161,
    doloc=237,
    fare=14.50,
    passengers=1
):
    pickup = datetime.fromisoformat(pickup_str)
    dropoff = pickup + timedelta(minutes=duration_min)
    return {
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": dropoff,
        "trip_distance": distance_mi,
        "PULocationID": puloc,
        "DOLocationID": doloc,
        "fare_amount": fare,
        "passenger_count": passengers
    }

def test_categorize_congestion():
    assert categorize_congestion(20.0) == "Low"
    assert categorize_congestion(15.0) == "Low"
    assert categorize_congestion(12.5) == "Moderate"
    assert categorize_congestion(8.0) == "Moderate"
    assert categorize_congestion(7.9) == "High"
    assert categorize_congestion(3.0) == "High"

def test_clean_batch_valid_row():
    df = pd.DataFrame([make_test_row(duration_min=15.0, distance_mi=3.0)])
    metrics = {
        "total_raw_rows": 0, "total_cleaned_rows": 0, "dropped_nulls": 0,
        "dropped_duration_outliers": 0, "dropped_distance_outliers": 0,
        "dropped_invalid_locations": 0, "dropped_non_positive_fare": 0,
        "dropped_invalid_passengers": 0, "dropped_speed_outliers": 0
    }
    cleaned = clean_batch(df, metrics)
    assert len(cleaned) == 1
    row = cleaned.iloc[0]
    assert row["trip_duration_minutes"] == 15.0
    assert row["speed_mph"] == 12.0  # (3.0 mi / 0.25 hr = 12 mph)
    assert row["congestion_level"] == "Moderate"
    assert row["pickup_hour"] == 8
    assert row["pickup_date"] == "2024-01-15"
    assert row["is_weekend"] == 0

def test_clean_batch_duration_filter():
    # 0.5 min (too short) and 300 mins (too long)
    df = pd.DataFrame([
        make_test_row(duration_min=0.5, distance_mi=1.0),
        make_test_row(duration_min=300.0, distance_mi=10.0),
        make_test_row(duration_min=20.0, distance_mi=4.0)
    ])
    metrics = {
        "total_raw_rows": 0, "total_cleaned_rows": 0, "dropped_nulls": 0,
        "dropped_duration_outliers": 0, "dropped_distance_outliers": 0,
        "dropped_invalid_locations": 0, "dropped_non_positive_fare": 0,
        "dropped_invalid_passengers": 0, "dropped_speed_outliers": 0
    }
    cleaned = clean_batch(df, metrics)
    assert len(cleaned) == 1
    assert metrics["dropped_duration_outliers"] == 2

def test_clean_batch_speed_outlier_filter():
    # 80 mph (unrealistic urban speed) and 0.5 mph (standstill/stopped)
    df = pd.DataFrame([
        make_test_row(duration_min=10.0, distance_mi=15.0), # 90 mph -> dropped
        make_test_row(duration_min=60.0, distance_mi=0.2),  # 0.2 mph -> dropped
        make_test_row(duration_min=15.0, distance_mi=3.0)   # 12 mph -> kept
    ])
    metrics = {
        "total_raw_rows": 0, "total_cleaned_rows": 0, "dropped_nulls": 0,
        "dropped_duration_outliers": 0, "dropped_distance_outliers": 0,
        "dropped_invalid_locations": 0, "dropped_non_positive_fare": 0,
        "dropped_invalid_passengers": 0, "dropped_speed_outliers": 0
    }
    cleaned = clean_batch(df, metrics)
    assert len(cleaned) == 1
    assert metrics["dropped_speed_outliers"] == 2

def test_clean_batch_invalid_fare_and_location():
    df = pd.DataFrame([
        make_test_row(fare=-5.0),     # negative fare -> dropped
        make_test_row(puloc=300),     # invalid location -> dropped
        make_test_row(passengers=0),  # zero passengers -> dropped
        make_test_row(distance_mi=0)  # zero distance -> dropped
    ])
    metrics = {
        "total_raw_rows": 0, "total_cleaned_rows": 0, "dropped_nulls": 0,
        "dropped_duration_outliers": 0, "dropped_distance_outliers": 0,
        "dropped_invalid_locations": 0, "dropped_non_positive_fare": 0,
        "dropped_invalid_passengers": 0, "dropped_speed_outliers": 0
    }
    cleaned = clean_batch(df, metrics)
    assert len(cleaned) == 0
    assert metrics["total_cleaned_rows"] == 0

def test_run_data_cleaning_pipeline(tmp_path):
    # Test end-to-end file pipeline with mini synthetic Parquet
    raw_path = tmp_path / "raw.parquet"
    out_path = tmp_path / "cleaned.parquet"
    sample_path = tmp_path / "sample.parquet"
    report_path = tmp_path / "report.md"

    rows = [
        make_test_row(duration_min=10.0, distance_mi=2.0),
        make_test_row(duration_min=20.0, distance_mi=6.0),
        make_test_row(duration_min=0.2, distance_mi=0.0) # invalid
    ]
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, raw_path)

    res = run_data_cleaning(
        input_path=raw_path,
        output_path=out_path,
        sample_output_path=sample_path,
        report_path=report_path,
        batch_size=10
    )

    assert res["total_raw_rows"] == 3
    assert res["total_cleaned_rows"] == 2
    assert out_path.exists()
    assert sample_path.exists()
    assert report_path.exists()
