"""
TrafficPulse - Ingestion Unit Tests
Tests schema inspection, download error handling, empty/corrupt file detection,
and spatial zone reconciliation using synthetic PyArrow fixtures.
"""

import os
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.ingestion.schema_inspector import inspect_parquet_schema
from src.ingestion.download_dataset import download_file, compute_sha256
from src.geospatial.zone_inspector import inspect_taxi_zones

@pytest.fixture
def sample_synthetic_parquet(tmp_path):
    """Generates a small synthetic Parquet file adhering to the NYC TLC schema."""
    parquet_path = tmp_path / "synthetic_sample.parquet"
    
    schema = pa.schema([
        ("VendorID", pa.int64()),
        ("tpep_pickup_datetime", pa.timestamp("us")),
        ("tpep_dropoff_datetime", pa.timestamp("us")),
        ("passenger_count", pa.float64()),
        ("trip_distance", pa.float64()),
        ("PULocationID", pa.int64()),
        ("DOLocationID", pa.int64()),
        ("fare_amount", pa.float64()),
        ("total_amount", pa.float64()),
    ])

    data = [
        pa.array([1, 2]),
        pa.array([1704067200000000, 1704070800000000], type=pa.timestamp("us")),
        pa.array([1704068100000000, 1704072000000000], type=pa.timestamp("us")),
        pa.array([1.0, 2.0]),
        pa.array([2.5, 4.8]),
        pa.array([161, 237]),
        pa.array([236, 142]),
        pa.array([12.5, 22.0]),
        pa.array([18.3, 29.5]),
    ]
    table = pa.Table.from_arrays(data, schema=schema)
    pq.write_table(table, parquet_path)
    return parquet_path

@pytest.fixture
def sample_corrupt_file(tmp_path):
    """Generates an invalid/corrupt Parquet file."""
    corrupt_path = tmp_path / "corrupt_data.parquet"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_PARQUET_MAGIC_HEADER_123456789")
    return corrupt_path

@pytest.fixture
def sample_empty_file(tmp_path):
    """Generates an empty 0-byte file."""
    empty_path = tmp_path / "empty_file.parquet"
    empty_path.touch()
    return empty_path

# ==============================================================================
# 1. Schema Inspector Tests
# ==============================================================================

def test_inspect_valid_parquet_schema(sample_synthetic_parquet):
    """Tests metadata extraction from a valid Parquet fixture."""
    report = inspect_parquet_schema(sample_synthetic_parquet)
    
    assert report["total_rows"] == 2
    assert report["num_columns"] == 9
    assert report["num_row_groups"] == 1
    assert "tpep_pickup_datetime" in report["fields"]["timestamp_fields"]
    assert "tpep_dropoff_datetime" in report["fields"]["timestamp_fields"]
    assert "PULocationID" in report["fields"]["location_fields"]
    assert "DOLocationID" in report["fields"]["location_fields"]
    assert "trip_distance" in report["fields"]["distance_fields"]
    assert "fare_amount" in report["fields"]["fare_fields"]
    assert report["file_size_bytes"] > 0

def test_inspect_corrupt_parquet(sample_corrupt_file):
    """Verifies that corrupted files raise a ValueError."""
    with pytest.raises(ValueError, match="Failed to parse Parquet file metadata"):
        inspect_parquet_schema(sample_corrupt_file)

def test_inspect_empty_file(sample_empty_file):
    """Verifies that 0-byte files raise a ValueError."""
    with pytest.raises(ValueError):
        inspect_parquet_schema(sample_empty_file)

def test_inspect_nonexistent_file(tmp_path):
    """Verifies that missing files raise a FileNotFoundError."""
    missing_path = tmp_path / "nonexistent.parquet"
    with pytest.raises(FileNotFoundError):
        inspect_parquet_schema(missing_path)

def test_inspect_missing_domain_columns(tmp_path):
    """Tests schema inspection when key domain columns (like distance) are missing."""
    sparse_path = tmp_path / "sparse.parquet"
    table = pa.Table.from_arrays([pa.array([1, 2])], names=["id"])
    pq.write_table(table, sparse_path)

    report = inspect_parquet_schema(sparse_path)
    assert report["total_rows"] == 2
    assert len(report["fields"]["timestamp_fields"]) == 0
    assert len(report["fields"]["distance_fields"]) == 0
    assert len(report["fields"]["location_fields"]) == 0

# ==============================================================================
# 2. Downloader Tests (Mocked)
# ==============================================================================

def test_download_file_success(tmp_path):
    """Tests successful streaming download and atomic rename."""
    dest = tmp_path / "downloaded.parquet"
    mock_content = b"PAR1_MOCK_PARQUET_STREAM_DATA_12345"

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {"content-length": str(len(mock_content)), "etag": "test-etag"}
    mock_resp.iter_content = MagicMock(return_value=[mock_content])
    mock_resp.__enter__.return_value = mock_resp

    with patch("requests.get", return_value=mock_resp):
        res = download_file("http://example.com/test.parquet", dest, force=True)
        assert res["status"] == "DOWNLOADED"
        assert res["file_size_bytes"] == len(mock_content)
        assert dest.exists()
        assert not dest.with_suffix(".parquet.tmp").exists()

def test_download_file_cached(tmp_path):
    """Verifies that existing non-empty files are not downloaded again without force=True."""
    dest = tmp_path / "existing.parquet"
    dest.write_bytes(b"existing_valid_data")

    with patch("requests.get") as mock_get:
        res = download_file("http://example.com/test.parquet", dest, force=False)
        assert res["status"] == "CACHED"
        mock_get.assert_not_called()

def test_download_file_empty_failure(tmp_path):
    """Verifies that receiving an empty 0-byte stream raises an error and cleans up tmp."""
    dest = tmp_path / "empty_download.parquet"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {"content-length": "0"}
    mock_resp.iter_content = MagicMock(return_value=[])
    mock_resp.__enter__.return_value = mock_resp

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="empty"):
            download_file("http://example.com/empty.parquet", dest, force=True)
        assert not dest.exists()
        assert not dest.with_suffix(".parquet.tmp").exists()

def test_download_file_http_error(tmp_path):
    """Verifies that HTTP errors (e.g., 404 Not Found) are propagated."""
    dest = tmp_path / "error_file.parquet"
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("404 Client Error: Not Found")
    mock_resp.__enter__.return_value = mock_resp

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(Exception, match="404 Client Error"):
            download_file("http://example.com/notfound.parquet", dest, force=True)
        assert not dest.exists()

# ==============================================================================
# 3. Spatial Zone Reconciliation Tests
# ==============================================================================

def test_spatial_zone_reconciliation(tmp_path):
    """Tests cross-validation between lookup CSV and GeoJSON properties."""
    csv_path = tmp_path / "taxi_zone_lookup.csv"
    geojson_path = tmp_path / "taxi_zones.geojson"

    # Synthetic lookup CSV
    csv_content = (
        "LocationID,Borough,Zone,service_zone\n"
        "1,EWR,Newark Airport,EWR\n"
        "2,Queens,Jamaica Bay,Boro Zone\n"
        "264,Unknown,NV,Unknown\n"
    )
    csv_path.write_text(csv_content, encoding="utf-8")

    # Synthetic GeoJSON (Zone 264 has no geometry)
    geojson_dict = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"location_id": "1", "zone": "Newark Airport", "borough": "EWR"},
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
            },
            {
                "type": "Feature",
                "properties": {"location_id": "2", "zone": "Jamaica Bay", "borough": "Queens"},
                "geometry": {"type": "Polygon", "coordinates": [[[1, 1], [2, 1], [2, 2], [1, 1]]]}
            }
        ]
    }
    geojson_path.write_text(json.dumps(geojson_dict), encoding="utf-8")

    report = inspect_taxi_zones(csv_path, geojson_path)
    assert report["total_lookup_zones"] == 3
    assert report["total_geojson_features"] == 2
    assert report["geojson_id_property_key"] == "location_id"
    assert 264 in report["unmapped_zones_in_geojson"]
    assert len(report["extra_zones_in_geojson"]) == 0
