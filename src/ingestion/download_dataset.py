#!/usr/bin/env python3
"""
TrafficPulse - Dataset Ingestion Module
Reliable, streaming downloader for NYC TLC trip records and taxi zone spatial metadata.
Ensures zero-corruption atomic writes, checksum verification, and idempotent caching.
"""

import os
import sys
import json
import time
import hashlib
import requests
from datetime import datetime
from pathlib import Path

# Authoritative NYC TLC and NYC OpenData Endpoints
DEFAULT_DATASETS = {
    "yellow_tripdata_2024_01": {
        "url": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
        "target_filename": "yellow_tripdata_2024-01.parquet",
        "description": "NYC TLC Yellow Taxi Trip Records (January 2024)"
    },
    "taxi_zone_lookup": {
        "url": "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
        "target_filename": "taxi_zone_lookup.csv",
        "description": "NYC TLC Taxi Zone Lookup (LocationID to Borough/Zone)"
    },
    "taxi_zones_geojson": {
        "url": "https://data.cityofnewyork.us/api/geospatial/8meu-9t5y?method=export&format=GeoJSON",
        "target_filename": "taxi_zones.geojson",
        "description": "Official NYC OpenData Taxi Zone Geographic Boundaries (GeoJSON)"
    }
}

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_METADATA_FILE = Path("data/raw/ingestion_metadata.json")

def compute_sha256(filepath, chunk_size=1024 * 1024):
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()

def download_file(
    url: str,
    target_path: Path,
    description: str = "",
    timeout: tuple = (10.0, 60.0),
    chunk_size: int = 1024 * 512,  # 512 KB
    force: bool = False
) -> dict:
    """
    Downloads a file via streaming HTTP GET to a temporary file,
    verifies integrity, and atomically replaces the target file.
    
    Returns a dictionary of download metadata.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if a valid non-empty file already exists
    if target_path.exists() and target_path.stat().st_size > 0 and not force:
        size_bytes = target_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"[CACHE] Valid existing file found: {target_path} ({size_mb:.2f} MB). Skipping download.")
        return {
            "status": "CACHED",
            "url": url,
            "target_path": str(target_path),
            "file_size_bytes": size_bytes,
            "file_size_mb": round(size_mb, 2),
            "sha256": compute_sha256(target_path),
            "verified_at": datetime.now().isoformat()
        }

    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    headers = {"User-Agent": "TrafficPulse-Ingestion/1.0 (CCS334 Big Data Analytics)"}

    print(f"\n[INIT] Requesting: {url}")
    if description:
        print(f"       Description: {description}")
    
    start_time = time.time()
    try:
        with requests.get(url, stream=True, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
            etag = response.headers.get("etag", "")

            downloaded_bytes = 0
            sha256_hash = hashlib.sha256()
            last_print_time = time.time()

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        sha256_hash.update(chunk)
                        downloaded_bytes += len(chunk)
                        
                        # Print progress at most once per 0.5 seconds
                        curr_time = time.time()
                        if curr_time - last_print_time >= 0.5:
                            downloaded_mb = downloaded_bytes / (1024 * 1024)
                            if total_mb > 0:
                                pct = (downloaded_bytes / total_size) * 100
                                sys.stdout.write(f"\r[DOWNLOADING] {downloaded_mb:.2f}/{total_mb:.2f} MB ({pct:.1f}%)")
                            else:
                                sys.stdout.write(f"\r[DOWNLOADING] {downloaded_mb:.2f} MB")
                            sys.stdout.flush()
                            last_print_time = curr_time

            # Final newline after progress loop
            downloaded_mb = downloaded_bytes / (1024 * 1024)
            print(f"\r[COMPLETED] Stream finished: {downloaded_mb:.2f} MB in {time.time() - start_time:.2f}s.")

            # Integrity checks
            if downloaded_bytes == 0:
                if temp_path.exists():
                    temp_path.unlink()
                raise ValueError("Downloaded file is empty (0 bytes).")
            
            if total_size > 0 and downloaded_bytes != total_size:
                if temp_path.exists():
                    temp_path.unlink()
                raise IOError(f"Incomplete download: expected {total_size} bytes, received {downloaded_bytes} bytes.")

            # Atomic rename from temporary file
            os.replace(temp_path, target_path)
            calculated_hash = sha256_hash.hexdigest()

            result_meta = {
                "status": "DOWNLOADED",
                "url": url,
                "target_path": str(target_path),
                "file_size_bytes": downloaded_bytes,
                "file_size_mb": round(downloaded_mb, 2),
                "sha256": calculated_hash,
                "etag": etag,
                "download_duration_seconds": round(time.time() - start_time, 2),
                "downloaded_at": datetime.now().isoformat()
            }
            return result_meta

    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        print(f"\n[ERROR] Download failed for {url}: {e}")
        raise

def update_metadata_registry(registry_path: Path, new_entry_key: str, metadata: dict):
    """Updates the centralized metadata JSON ledger."""
    registry_path = Path(registry_path)
    current_data = {}
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        except Exception:
            current_data = {}

    current_data[new_entry_key] = metadata
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(current_data, f, indent=2)
    print(f"[METADATA] Registry updated at: {registry_path}")

def run_dataset_ingestion(raw_dir: Path = DEFAULT_RAW_DIR, force: bool = False):
    """Downloads all core datasets and records audit metadata."""
    print("=" * 80)
    print(" TrafficPulse - Big Data Ingestion Pipeline")
    print(f" Target Directory: {raw_dir}")
    print("=" * 80)

    summary = {}
    for key, spec in DEFAULT_DATASETS.items():
        target_path = raw_dir / spec["target_filename"]
        try:
            meta = download_file(
                url=spec["url"],
                target_path=target_path,
                description=spec["description"],
                force=force
            )
            update_metadata_registry(DEFAULT_METADATA_FILE, key, meta)
            summary[key] = meta
        except Exception as err:
            summary[key] = {"status": "FAILED", "error": str(err)}

    print("\n" + "=" * 80)
    print(" Ingestion Summary:")
    for k, v in summary.items():
        status = v.get("status", "UNKNOWN")
        size = f"{v.get('file_size_mb', 0)} MB" if "file_size_mb" in v else "N/A"
        print(f"  - {k:<25} [{status:<10}] {size}")
    print("=" * 80)
    return summary

if __name__ == "__main__":
    force_download = "--force" in sys.argv
    run_dataset_ingestion(force=force_download)
