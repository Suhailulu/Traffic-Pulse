"""
TrafficPulse - Dashboard Unit Tests (Phase 5)
Tests data loading, zone lookup mapping, spatial centroid extraction,
KPI calculation consistency, filtering logic, and prediction engine integration.
"""

import pytest
import pandas as pd
from datetime import date
from pathlib import Path

from dashboard.app import (
    load_trip_dataset,
    load_zone_lookup,
    load_precomputed_summaries,
    load_geojson_and_centroids,
    load_prediction_model,
    compute_summary_kpis,
    filter_trip_data,
    find_closest_zone,
    geocode_osm_location,
    generate_direct_leaflet_html,
    NEEDED_COLUMNS
)

# ==============================================================================
# 1. Data Loader & Schema Verification Tests
# ==============================================================================

def test_load_trip_dataset_structure():
    """Verifies that loaded trip dataset adheres to expected columns and valid data types."""
    df, desc = load_trip_dataset()
    assert df is not None, "Trip dataset should be loaded"
    assert not df.empty, "Dataset should not be empty"
    for col in NEEDED_COLUMNS:
        assert col in df.columns, f"Missing expected column: {col}"

    assert "Full" in desc or "Sample" in desc
    assert len(df) > 0


def test_load_zone_lookup():
    """Verifies that zone lookup dictionary contains valid zone metadata."""
    lookup = load_zone_lookup()
    assert len(lookup) >= 260, f"Expected at least 260 zones, found {len(lookup)}"

    # Check key zone: Midtown Center (ID 161)
    assert 161 in lookup
    midtown = lookup[161]
    assert midtown["zone_name"] == "Midtown Center"
    assert midtown["borough"] == "Manhattan"
    assert "display_name" in midtown


def test_load_precomputed_summaries():
    """Verifies precomputed CSV and JSON summaries."""
    hourly_df, zone_df, report = load_precomputed_summaries()
    assert hourly_df is not None
    assert len(hourly_df) == 24
    assert set(hourly_df.columns).issuperset({"hour", "trip_count", "avg_duration_minutes", "avg_speed_mph"})

    assert zone_df is not None
    assert len(zone_df) > 200
    assert report is not None
    assert "traffic_kpis" in report


def test_load_geojson_and_centroids():
    """Verifies that GeoJSON boundaries are parsed and centroids are computed."""
    geo_data, centroids = load_geojson_and_centroids()
    assert geo_data is not None
    assert "features" in geo_data
    assert len(centroids) >= 250

    # Zone 161 should have valid NYC coordinates
    assert 161 in centroids
    lat, lon = centroids[161]
    assert 40.5 <= lat <= 41.0
    assert -74.3 <= lon <= -73.6


# ==============================================================================
# 2. KPI Computation Tests
# ==============================================================================

def test_compute_summary_kpis_with_dataframe():
    """Tests that summary KPIs match exact mathematical definitions from data."""
    lookup = {
        1: {"zone_name": "Alpha", "borough": "Manhattan"},
        2: {"zone_name": "Beta", "borough": "Queens"}
    }
    mock_df = pd.DataFrame([
        {"pickup_hour": 14, "PULocationID": 1, "trip_duration_minutes": 10.0, "speed_mph": 12.0},
        {"pickup_hour": 14, "PULocationID": 1, "trip_duration_minutes": 20.0, "speed_mph": 14.0},
        {"pickup_hour": 18, "PULocationID": 2, "trip_duration_minutes": 30.0, "speed_mph": 10.0}
    ])

    kpis = compute_summary_kpis(mock_df, lookup)
    assert kpis["total_trips"] == 3
    assert kpis["avg_duration_minutes"] == 20.0  # (10 + 20 + 30) / 3 = 20.0
    assert kpis["avg_speed_mph"] == 12.0        # (12 + 14 + 10) / 3 = 12.0
    assert kpis["peak_hour"] == 14              # Hour 14 has 2 trips vs Hour 18 with 1 trip
    assert kpis["peak_volume"] == 2
    assert kpis["busiest_zone_id"] == 1         # Zone 1 has 2 trips vs Zone 2 with 1 trip
    assert kpis["busiest_zone_name"] == "Alpha"


def test_compute_summary_kpis_empty_dataframe_fallback():
    """Verifies graceful fallback to benchmark summary when dataframe is empty."""
    lookup = load_zone_lookup()
    _, _, report = load_precomputed_summaries()
    kpis = compute_summary_kpis(pd.DataFrame(), lookup, fallback_report=report)

    assert kpis["total_trips"] > 0
    assert kpis["avg_duration_minutes"] > 0
    assert kpis["avg_speed_mph"] > 0
    assert kpis["source"] == "Precomputed Benchmark Summary"


# ==============================================================================
# 3. Filtering Logic Tests
# ==============================================================================

def test_filter_trip_data_by_hour():
    """Verifies that hour filtering correctly subsets data."""
    lookup = load_zone_lookup()
    mock_df = pd.DataFrame([
        {"pickup_date": "2024-01-10", "pickup_hour": 8, "PULocationID": 161, "is_weekend": 0},
        {"pickup_date": "2024-01-10", "pickup_hour": 14, "PULocationID": 161, "is_weekend": 0},
        {"pickup_date": "2024-01-10", "pickup_hour": 20, "PULocationID": 161, "is_weekend": 0}
    ])

    res = filter_trip_data(
        df=mock_df,
        date_range=None,
        selected_boroughs=["All Boroughs"],
        selected_zones=[],
        hour_range=(10, 18),
        day_type="All Days",
        zone_lookup=lookup
    )

    assert len(res) == 1
    assert res.iloc[0]["pickup_hour"] == 14


def test_filter_trip_data_by_day_type():
    """Verifies weekday and weekend filtering separation."""
    lookup = load_zone_lookup()
    mock_df = pd.DataFrame([
        {"pickup_date": "2024-01-10", "pickup_hour": 12, "PULocationID": 161, "is_weekend": 0},
        {"pickup_date": "2024-01-13", "pickup_hour": 12, "PULocationID": 161, "is_weekend": 1}
    ])

    res_weekday = filter_trip_data(mock_df, None, ["All Boroughs"], [], (0, 23), "Weekdays Only (Mon-Fri)", lookup)
    res_weekend = filter_trip_data(mock_df, None, ["All Boroughs"], [], (0, 23), "Weekends Only (Sat-Sun)", lookup)

    assert len(res_weekday) == 1
    assert res_weekday.iloc[0]["is_weekend"] == 0
    assert len(res_weekend) == 1
    assert res_weekend.iloc[0]["is_weekend"] == 1


def test_filter_trip_data_by_borough():
    """Verifies geographic borough filtering."""
    lookup = {
        161: {"zone_name": "Midtown Center", "borough": "Manhattan"},
        132: {"zone_name": "JFK Airport", "borough": "Queens"}
    }
    mock_df = pd.DataFrame([
        {"pickup_date": "2024-01-10", "pickup_hour": 12, "PULocationID": 161, "is_weekend": 0},
        {"pickup_date": "2024-01-10", "pickup_hour": 12, "PULocationID": 132, "is_weekend": 0}
    ])

    res_manhattan = filter_trip_data(mock_df, None, ["Manhattan"], [], (0, 23), "All Days", lookup)
    assert len(res_manhattan) == 1
    assert res_manhattan.iloc[0]["PULocationID"] == 161


# ==============================================================================
# 4. Prediction Integration Tests
# ==============================================================================

def test_prediction_model_integration():
    """Verifies that the trained ML model integrates seamlessly into dashboard."""
    predictor = load_prediction_model()
    assert predictor is not None, "Model bundle should load successfully"

    # Inference call for Midtown Center at 6 PM Wednesday
    result = predictor.predict(hour=18, zone_id=161, day_of_week=2)
    assert result["status"] == "SUCCESS"
    assert result["prediction"]["congestion_level"] in ["Low", "Moderate", "High"]
    assert 0.0 <= result["prediction"]["confidence"] <= 1.0

    probs = result["prediction"]["class_probabilities"]
    assert set(probs.keys()) == {"Low", "Moderate", "High"}
    assert pytest.approx(sum(probs.values()), abs=1e-2) == 1.0


# ==============================================================================
# 5. Leaflet & OpenStreetMap API Integration Tests
# ==============================================================================

def test_find_closest_zone_times_square():
    """Verifies that coordinates for Times Square (40.7580, -73.9855) map to Midtown / Theater District."""
    _, centroids = load_geojson_and_centroids()
    best_zid = find_closest_zone(40.7580, -73.9855, centroids)
    assert best_zid is not None
    lookup = load_zone_lookup()
    assert best_zid in [230, 161, 163]  # Times Sq / Midtown Center / Midtown North
    assert lookup[best_zid]["borough"] == "Manhattan"


def test_geocode_osm_location_empty_input():
    """Verifies that empty and whitespace queries return None gracefully without making API requests."""
    assert geocode_osm_location("") is None
    assert geocode_osm_location("   ") is None


def test_generate_direct_leaflet_html():
    """Verifies that standalone Leaflet HTML contains Leaflet CDN, OSM tiles, and marker JS."""
    top_zones = pd.DataFrame([
        {"PULocationID": 161, "trip_count": 1500, "avg_speed": 9.2, "avg_duration": 14.5}
    ])
    centroids = {161: (40.75775, -73.97735)}
    lookup = {161: {"zone_name": "Midtown Center", "borough": "Manhattan"}}

    html = generate_direct_leaflet_html(
        center=(40.7306, -73.9352),
        zoom=11,
        top_zones=top_zones,
        centroids=centroids,
        zone_lookup=lookup,
        tile_provider="OpenStreetMap (Standard)"
    )

    assert "leaflet.js" in html
    assert "leaflet.css" in html
    assert "tile.openstreetmap.org" in html
    assert "L.circleMarker" in html
    assert "Midtown Center" in html
