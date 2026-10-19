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

## 5. Verified Traffic Analytics Results (January 2024 Records)

Aggregated from all **2,887,991 verified trips** using memory-safe PyArrow streaming batches:

- **Total Analyzed Trips:** `2,887,991` (97.42% retention)
- **Peak Travel Hour:** **Hour 18:00 (6:00 PM)** with `207,784 trips`
- **Lowest Travel Hour:** **Hour 04:00 (4:00 AM)** with `15,462 trips`
- **Overall Mean Trip Duration:** `14.99 minutes`
- **Overall Estimated Average Speed:** `11.53 mph`
- **Overall Mean Trip Distance:** `3.31 miles`

### Top 5 High-Demand Pickup Zones
| Rank | Zone ID | Zone Name | Borough | Trip Volume | Avg Speed (mph) | Avg Duration |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `161` | **Midtown Center** | Manhattan | `141,243` | `9.13 mph` | 13.92 min |
| 2 | `237` | **Upper East Side South** | Manhattan | `141,002` | `9.19 mph` | 11.03 min |
| 3 | `132` | **JFK Airport** | Queens | `140,160` | `27.27 mph` | 37.27 min |
| 4 | `236` | **Upper East Side North** | Manhattan | `134,533` | `9.91 mph` | 11.23 min |
| 5 | `162` | **Midtown East** | Manhattan | `105,113` | `9.58 mph` | 13.06 min |

### Key Diurnal Observations
- **Free-flow Velocity Window:** 04:00 – 06:00 AM exhibits the highest average urban speeds (`18.07 - 19.69 mph`).
- **Evening Peak Friction:** 03:00 – 06:00 PM exhibits the lowest urban speeds (`9.88 - 10.37 mph`) paired with peak passenger demand (> 200,000 trips/hr).

---

## 6. Analytical Output Artifacts
- **Hourly Traffic Summary:** [data/processed/hourly_traffic_summary.csv](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/data/processed/hourly_traffic_summary.csv)
- **Zone Traffic Summary:** [data/processed/zone_traffic_summary.csv](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/data/processed/zone_traffic_summary.csv)
- **Executive JSON Report:** [reports/traffic_analysis_report.json](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/reports/traffic_analysis_report.json)

---

## 7. How to Reproduce & Run

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

## 7. Machine Learning Congestion Prediction (Phase 4)

### 7.1 Objective & Approach
Phase 4 implements a functional, leak-free machine learning classification pipeline to predict urban traffic congestion levels (`Low`, `Moderate`, `High`) for any given hour, day of the week, and TLC geographic zone in New York City.

### 7.2 Target Label Definition
Congestion is derived using an empirical, domain-grounded vehicle traversal speed indicator:
- **Low Congestion (`speed >= 15.0 mph`):** Fluid, uncongested traffic flow with minimal street delay. Typical of late-night travel (02:00 - 05:00 AM) and uncongested outer-borough / airport expressway corridors (e.g., JFK Airport).
- **Moderate Congestion (`8.0 mph <= speed < 15.0 mph`):** Steady urban street traffic operating under standard signalized arterial cycles. Represents the predominant daytime traffic state across Manhattan and Brooklyn.
- **High Congestion (`speed < 8.0 mph`):** Severe corridor friction, stop-and-go congestion, and gridlock conditions. Highly concentrated in Midtown/Downtown corridors during evening rush hours (04:00 - 07:00 PM).

### 7.3 Feature Engineering & Strict Leakage Prevention
To guarantee that the model only uses information available **at the time the prediction is made**, post-trip measurements (`trip_duration`, `trip_distance`, `fare_amount`, `speed_mph`) are strictly excluded from the feature space.

The input feature vector consists of:
1. `pickup_hour` (integer 0–23): Diurnal cycle indicator.
2. `pickup_day_of_week` (integer 0–6, Mon=0, Sun=6): Weekly commute cycle.
3. `is_weekend` (binary 0/1): Weekend vs. weekday transit pattern flag.
4. `PULocationID` (integer 1–265): TLC geographic origin zone.
5. `hist_volume` (float): Historical trip volume for that zone and hour (from training data).
6. `hist_avg_duration` (float): Historical average trip duration in minutes for that zone and hour (from training data).
7. `hist_avg_speed` (float): Historical average speed in mph for that zone and hour (from training data).

### 7.4 Chronological Train/Test Split
To prevent temporal data leakage and reflect real-world deployment, records were partitioned chronologically:
- **Dataset:** 199,989 verified trip records sampled uniformly across all row groups of `data/processed/cleaned_trips_2024_01.parquet`.
- **Chronological Cutoff Timestamp:** `2024-01-24 15:36:40`
- **Training Set (75%):** `149,991 records` (January 1 to January 24, 2024)
- **Testing Set (25%):** `49,998 records` (January 24 to January 30, 2024)
- **Historical Profiling:** 2,831 unique (Zone × Hour) aggregate metric vectors were computed **strictly on the training split**. Unseen zone/hour combinations fall back gracefully to global training medians.

### 7.5 Model Performance & Evaluation Metrics
The primary Random Forest Classifier (`n_estimators=100`, `max_depth=12`, `min_samples_split=10`) was compared against a naive majority-class baseline and a regularized Logistic Regression baseline on the out-of-time chronological test set:

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Majority Class: Moderate)** | `49.10%` | `0.1637` | `0.3333` | `0.2195` | `0.3234` |
| **Logistic Regression Baseline** | `55.16%` | `0.6110` | `0.4832` | `0.4608` | `0.4745` |
| **Random Forest Classifier (TrafficPulse)** | **`59.17%`** | **`0.6414`** | **`0.5301`** | **`0.5377`** | **`0.5570`** |

*Random Forest demonstrates a **+10.07 percentage point** accuracy improvement over the naive baseline.*

#### Random Forest Detailed Classification Report (Test Set: 49,998 trips)
| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Low** (Speed >= 15 mph) | `0.6738` | `0.4560` | `0.5439` | 8,017 |
| **Moderate** (8 - 15 mph) | `0.5624` | `0.8645` | `0.6815` | 24,550 |
| **High** (Speed < 8 mph) | `0.6879` | `0.2698` | `0.3876` | 17,431 |
| **Overall Accuracy** | - | - | **`0.5917`** | **49,998** |
| **Weighted Average** | **`0.6240`** | **`0.5917`** | **`0.5570`** | **49,998** |

#### Test Set Confusion Matrix (Rows: Actual, Columns: Predicted [Low, Moderate, High])
```text
               Predicted Low   Predicted Moderate   Predicted High
Actual Low             3,656                4,094              267
Actual Moderate        1,460               21,223            1,867
Actual High              310               12,418            4,703
```

#### Feature Importance Hierarchy
```text
1. hist_avg_speed       (45.10%)  ██████████████████████
2. hist_avg_duration    (20.29%)  ██████████
3. PULocationID          (9.05%)  ████
4. pickup_hour           (8.03%)  ████
5. hist_volume           (7.80%)  ████
6. pickup_day_of_week    (6.65%)  ███
7. is_weekend            (3.09%)  █
```

### 7.6 How to Train & Run Predictions
```powershell
# 1. Train model, evaluate against baselines, and serialize to src/prediction/models/
python -m src.prediction.train_model

# 2. Run real-time congestion inference via CLI
# Example: Predict congestion for Midtown Center (Zone 161) at 6:00 PM (Hour 18) on Wednesday (Day 2)
python -m src.prediction.predict_congestion --hour 18 --zone 161 --day 2

# Example: Predict congestion for JFK Airport (Zone 132) at 3:00 AM (Hour 3)
python -m src.prediction.predict_congestion --hour 3 --zone 132 --day 2

# 3. Run prediction unit tests (19 tests)
pytest tests/test_prediction.py -v
```

### 7.7 Critical Limitations
> [!CAUTION]
> **Proxy-Based Congestion Caveat:**
> 1. **Commercial Vehicle Speed vs. Actual Road Traffic:** Yellow taxi speeds serve as an empirical proxy for street corridor delay, incorporating passenger boarding, traffic light dwells, and turning movements. They do not constitute direct physical road loop-detector or radar vehicle density measurements.
> 2. **Pre-trip Scheduling Scope:** Predictions are generated strictly from scheduled departure parameters and historical baselines. Real-time traffic anomalies (sudden collisions, emergency weather conditions, or street closures) cannot be observed at pre-trip inference time.
> 3. **Single-Month Representation:** Model training relies on January 2024 telemetry. Seasonal tourist influx, summer congestion shifts, and severe winter storm days are specific to winter conditions and require continuous multi-season retraining.

---

## 8. Verification & Test Suite Summary
TrafficPulse features an automated test suite verifying data quality, ingestion security, traffic analytics calculations, and machine learning prediction consistency:

```powershell
pytest -v
```
**Test Results:** **42 passed in 18.84s**
- `tests/test_cleaner.py`: 6 tests passing (data quality bounds, anomaly filtering, duration math)
- `tests/test_ingestion.py`: 10 tests passing (Parquet schema, checksums, spatial reconciliation)
- `tests/test_traffic_analysis.py`: 7 tests passing (hourly aggregations, speed statistics, edge cases)
- `tests/test_prediction.py`: 19 tests passing (chronological split, serialization, input validation, inference probability calibration, deterministic repeatability)

---

## 9. Development Verification Rules
1. **Zero Fabrication**: All analytical metrics, speed statistics, and model evaluation scores reflect genuine execution outputs.
2. **Strict Leakage Prevention**: Machine learning feature matrices never include post-trip attributes (`trip_duration`, `fare_amount`, `actual_speed`).
3. **Modular Testing**: Every module is tested before proceeding to downstream components.

