# TrafficPulse: Big Data Analytics & Predictive Intelligence for Urban Traffic Congestion

**Course:** CCS334 – Big Data Analytics  
**Project Type:** College Mini Project  
**Target SDG:** UN SDG 11 – *Sustainable Cities and Communities (Target 11.2 & 11.a)*  
**Dataset:** NYC TLC Trip Record Dataset (Yellow Taxi Parquet records + Taxi Zone Spatial Data)  

---

## 1. Project Overview

**TrafficPulse** is an end-to-end Big Data analytics and predictive intelligence platform designed to study, quantify, and predict urban traffic congestion using multi-million record trip telemetry from the New York City Taxi and Limousine Commission (TLC).

Urban traffic congestion represents a major economic and environmental burden, contributing to greenhouse gas emissions, commuter delays, and fuel inefficiency. Aligned with **UN Sustainable Development Goal 11 (Sustainable Cities and Communities)**, TrafficPulse leverages distributed storage, parallel computing, and machine learning to transform raw urban mobility records into actionable congestion intelligence.

---

## 2. Dataset Sources & Official Specifications

All datasets are obtained directly from official NYC government and commission portals:

| Dataset | Source Agency | Official URL / Format | Verified Size | Record Count |
| :--- | :--- | :--- | :--- | :--- |
| **Yellow Taxi Trip Records (Jan 2024)** | NYC TLC | [CloudFront Parquet CDN](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet) | `47.65 MB` (`49,961,641` B) | **2,964,624 rows** |
| **Taxi Zone Lookup Table** | NYC TLC | [CloudFront CSV](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv) | `12.33 KB` | **265 zones** |
| **Taxi Zones Spatial Boundaries** | NYC OpenData | [NYC Open Data GeoJSON](https://data.cityofnewyork.us/api/geospatial/8meu-9t5y?method=export&format=GeoJSON) | `3.71 MB` | **263 polygons** |

---

## 3. Verified Parquet Schema (January 2024 Records)

Schema inspection performed directly from Parquet metadata footers using PyArrow (zero dataframe memory loading):

```text
Total Rows: 2,964,624 | Total Columns: 19 | Row Groups: 3 | Format Version: 2.6
Writer: parquet-cpp-arrow version 14.0.2
```

| Index | Column Name | Arrow Data Type | Physical Parquet Type | Analytical Role / Domain Meaning |
| :---: | :--- | :--- | :--- | :--- |
| 0 | `VendorID` | `int32` | `INT32` | TPEP provider (1=Creative Mobile, 2=VeriFone) |
| 1 | `tpep_pickup_datetime` | `timestamp[us]` | `INT64` | Trip start timestamp (Hour, Day, Rush hour) |
| 2 | `tpep_dropoff_datetime` | `timestamp[us]` | `INT64` | Trip end timestamp (Duration computation) |
| 3 | `passenger_count` | `int64` | `INT64` | Number of passengers in vehicle |
| 4 | `trip_distance` | `double` | `DOUBLE` | Elapsed trip distance in miles |
| 5 | `RatecodeID` | `int64` | `INT64` | Final rate code (1=Standard, 2=JFK, 3=Newark) |
| 6 | `store_and_fwd_flag` | `large_string` | `BYTE_ARRAY` | Memory store & forward flag (Y/N) |
| 7 | `PULocationID` | `int32` | `INT32` | TLC Taxi Zone in which passenger was picked up |
| 8 | `DOLocationID` | `int32` | `INT32` | TLC Taxi Zone in which passenger dropped off |
| 9 | `payment_type` | `int64` | `INT64` | Payment method (1=Credit, 2=Cash, etc.) |
| 10 | `fare_amount` | `double` | `DOUBLE` | Time-and-distance fare calculated by meter |
| 11 | `extra` | `double` | `DOUBLE` | Miscellaneous extras and surcharges ($0.50, $1.00) |
| 12 | `mta_tax` | `double` | `DOUBLE` | $0.50 MTA tax automatically triggered |
| 13 | `tip_amount` | `double` | `DOUBLE` | Tip amount (credit card tips only) |
| 14 | `tolls_amount` | `double` | `DOUBLE` | Total bridge and tunnel toll amount |
| 15 | `improvement_surcharge`| `double` | `DOUBLE` | $0.30 or $1.00 improvement surcharge |
| 16 | `total_amount` | `double` | `DOUBLE` | Total charged to passengers |
| 17 | `congestion_surcharge` | `double` | `DOUBLE` | New York State Congestion Surcharge |
| 18 | `Airport_fee` | `double` | `DOUBLE` | Airport pickup fee ($1.25 or $1.75) |

### Row Group Storage Layout
- **Row Group 0:** 1,048,576 rows (26.93 MB uncompressed)
- **Row Group 1:** 1,048,576 rows (26.66 MB uncompressed)
- **Row Group 2:** 867,472 rows (21.60 MB uncompressed)

---

## 4. Geospatial Architecture & Taxi Zone Reconciliation

### Primary & Foreign Keys
- **Trip Records:** `PULocationID` and `DOLocationID` (`int32`)
- **Taxi Zone Lookup Table:** `LocationID` (`int64`)
- **Taxi Zone GeoJSON:** `locationid` (property attribute matching `LocationID`)

### Zone Reconciliation Findings
- **265 zones** exist in the official lookup table.
- **263 polygon features** exist in the GeoJSON boundary dataset.
- **Missing GeoJSON Geometries:**
  - `LocationID 264`: `Unknown / Outside NYC` (catch-all code for missing GPS telemetry)
  - `LocationID 265`: `Outside of NYC / NV` (territory outside the 5 boroughs)
  - `LocationID 104 & 105`: Governor's Island / Ellis Island / Liberty Island (merged/non-taxicab zones)
  - `LocationID 57`: Queens (Corona boundary consolidation)

### Critical Traffic Analysis Limitations
> [!IMPORTANT]
> **Spatial Proxy Caveat:** Taxi zones represent polygonal neighborhood districts (derived from NYC Department of City Planning Neighborhood Tabulation Areas), **not** individual street centerlines or lane sensor monitors.
>
> 1. Derived vehicle speed represents an **aggregate OD corridor velocity**, which accounts for passenger boarding, traffic signals, and street network topology.
> 2. Intra-zone trips (`PULocationID == DOLocationID`) reflect neighborhood circulating mobility.
> 3. Travel time variations across identical hours serve as an empirical probe for urban traffic congestion, distinct from fixed loop-detector vehicle density.

---

## 5. Project Directory Structure

```text
TrafficPulse/
├── data/
│   ├── raw/                 # Downloaded raw datasets & metadata (git-ignored)
│   │   ├── yellow_tripdata_2024-01.parquet  (47.65 MB, 2,964,624 rows)
│   │   ├── taxi_zone_lookup.csv             (265 zones)
│   │   ├── taxi_zones.geojson               (263 boundary polygons)
│   │   └── ingestion_metadata.json          (Audit hashes & timestamps)
│   ├── processed/           # Cleaned and validated datasets (git-ignored)
│   └── sample/              # Test fixtures and micro datasets
├── src/
│   ├── ingestion/           # Data collection & schema inspection modules
│   │   ├── download_dataset.py     # Streaming atomic downloader with checksums
│   │   └── schema_inspector.py     # Metadata-only Parquet inspector
│   ├── processing/          # Cleaning, outlier rejection & PySpark transformations
│   ├── analytics/           # Congestion metrics, diurnal curves & corridor bottlenecks
│   ├── prediction/          # Leakage-free feature engineering & ML models
│   └── geospatial/          # Spatial reconciliation & boundary handling
│       └── zone_inspector.py       # Join validation and boundary cross-checker
├── dashboard/               # Streamlit pages and visual components
├── hive/                    # HiveQL external table DDL & analytical queries
├── mapreduce/               # Hadoop Streaming mapper.py & reducer.py
├── notebooks/               # Interactive experimentation and analysis notebooks
├── tests/                   # Pytest test suite for data quality & ML leakage
│   └── test_ingestion.py    # 10 comprehensive unit tests (all passing)
├── reports/                 # Verified execution outputs, evaluation metrics & logs
│   ├── dataset_metadata_report.md   # Parquet schema inspection details
│   └── taxi_zones_spatial_report.md # Spatial boundary reconciliation details
├── scripts/                 # Automation & verification scripts
│   ├── verify_env.py        # Environment inspection & diagnostic script
│   └── test_urls.py         # URL connectivity tester
├── docs/                    # Architecture diagrams & academic documentation
├── docker-compose.yml       # Hadoop HDFS, Hive, and YARN/MapReduce cluster definition
├── requirements.txt         # Minimal locked project dependencies
├── .env.example             # Configuration template
├── .gitignore               # Strict exclusion of large datasets, .venv, and models
└── README.md                # Project documentation and reproduction guide
```

---

## 6. Big Data Infrastructure Management (Hadoop HDFS, Hive, PySpark)

### Port Allocation & Service Matrix
| Service | Container Name | Host Port | Protocol / Purpose | Memory Limit |
| :--- | :--- | :--- | :--- | :--- |
| **HDFS NameNode** | `trafficpulse-namenode` | `9870` | WebHDFS REST API & Storage Web UI | `768 MB` (Heap: 512M) |
| **HDFS IPC** | `trafficpulse-namenode` | `9000` | Hadoop RPC Filesystem endpoint | - |
| **YARN ResourceManager** | `trafficpulse-namenode` | `8088` | MapReduce Application Coordinator UI | - |
| **HDFS DataNode** | `trafficpulse-datanode` | `9864` | Block Storage Worker Web UI | `768 MB` (Heap: 512M) |
| **Apache HiveServer2** | `trafficpulse-hive-server` | `10000` | Hive JDBC/ODBC & Thrift Server | `768 MB` (Heap: 512M) |
| **HiveServer2 Web UI** | `trafficpulse-hive-server` | `10002` | Hive Analytical Query Monitor UI | - |
| **Hive Metastore DB** | `trafficpulse-hive-db` | `5432` | PostgreSQL Relational Schema Catalog | `256 MB` |

### Startup & Shutdown Commands
```powershell
# 1. Start Docker Desktop application (via Windows Start Menu or Desktop shortcut)

# 2. Start HDFS Cluster (Conservative footprint: ~1.5 GB RAM)
docker compose up -d hadoop-master hadoop-worker

# 3. Start Apache Hive Warehouse (adds ~1.0 GB RAM)
docker compose up -d hive-metastore-db hive-server

# 4. Check Container Health & Status
docker compose ps

# 5. Stop Cluster and Preserve Storage Volumes
docker compose down

# 6. Stop and Wipe Clean (Remove Data Volumes)
docker compose down -v
```

### Component Validation Commands
```powershell
# Validate HDFS Web Interface and Root Directory Listing
python scripts/validate_hdfs.py

# Validate PySpark Engine & SparkSession
python scripts/validate_pyspark.py

# Validate MapReduce Streaming Pipeline (Locally or on YARN)
cat data/sample/sample_trips.csv | python mapreduce/mapper_test.py | sort | python mapreduce/reducer_test.py
```

---

## 7. How to Reproduce & Run

### Step 1: Virtual Environment Setup
```powershell
# Navigate to project directory
cd "C:\Users\suhai\OneDrive\Desktop\Traffic Pulse"

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Step 2: Environment Diagnostic Check
```powershell
python scripts\verify_env.py
```

### Step 3: Run Ingestion & Download Raw Data
```powershell
python src\ingestion\download_dataset.py
```

### Step 4: Run Parquet Schema Inspector
```powershell
python src\ingestion\schema_inspector.py
```

### Step 5: Run Geospatial Zone Inspector
```powershell
python src\geospatial\zone_inspector.py
```

### Step 6: Execute Automated Test Suite
```powershell
pytest tests\test_ingestion.py -v
```

---

## 7. Development Verification Rules
1. **Zero Fabrication**: All analytical metrics, speed statistics, and model evaluation scores reflect genuine execution outputs.
2. **Strict Leakage Prevention**: Machine learning feature matrices never include post-trip attributes (`trip_duration`, `fare_amount`, `actual_speed`).
3. **Modular Testing**: Every module is tested before proceeding to downstream components.
