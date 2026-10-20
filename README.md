# TrafficPulse: Big Data Analytics & Predictive Intelligence for Urban Traffic Congestion

**Course:** CCS334 – Big Data Analytics  
**Project Category:** Academic Mini Project  
**Target SDG:** UN SDG 11 – *Sustainable Cities and Communities (Targets 11.2 & 11.a)*  
**Dataset:** NYC TLC Yellow Taxi Trip Records (January 2024) + NYC OpenData Taxi Zone Spatial Boundaries  

---

## 1. Project Overview

**TrafficPulse** is an end-to-end Big Data analytics and predictive intelligence platform designed to study, quantify, and predict urban traffic congestion using multi-million record trip telemetry from the New York City Taxi and Limousine Commission (TLC).

Urban traffic congestion represents a major economic and environmental burden, contributing to greenhouse gas emissions, commuter delays, and fuel inefficiency. Aligned with **UN Sustainable Development Goal 11 (Sustainable Cities and Communities)**, TrafficPulse leverages streaming batch processing, distributed storage principles, parallel computing paradigms, and machine learning to transform raw urban mobility records into actionable congestion intelligence.

---

## 2. Problem Statement & Objectives

### 2.1 Problem Statement
Urban roadway networks in metropolitan areas suffer from severe recurring congestion, resulting in unpredictable travel times, elevated vehicular emissions, and increased economic cost. Conventional traffic monitoring relies heavily on fixed physical sensor networks (e.g., inductive loop detectors, microwave radar, and CCTV cameras), which suffer from limited geographic coverage and high capital deployment costs.

Conversely, commercial floating car data (FCD)—such as NYC TLC yellow taxi telemetry—captures hundreds of thousands of daily point-to-point urban trips. However, turning raw tabular trip records into actionable intelligence presents major Big Data challenges:
1. **High Volume:** Millions of monthly records requiring memory-safe streaming processing.
2. **Data Quality Anomalies:** Negative trip durations, zero-distance errors, unrealistic speeds, and unmapped spatial codes.
3. **Temporal Leakage in Machine Learning:** Standard models inadvertently exploit post-trip metrics (e.g., recorded trip duration or actual fare), rendering them useless for pre-trip congestion prediction.

### 2.2 Project Objectives
1. **Scalable Ingestion & Schema Inspection:** Ingest official NYC TLC Parquet datasets with checksum verification and inspect metadata-level schemas without in-memory dataframe bloat.
2. **Quality Cleaning & Feature Engineering:** Perform batch-by-batch outlier rejection and compute trip duration, speed, and congestion classification labels.
3. **Diurnal Analytics & Hotspot Identification:** Compute 24-hour traffic volume cycles, duration dynamics, weekday versus weekend variations, and pickup-zone demand rankings.
4. **Big Data Ecosystem Demonstrations:** Implement and evaluate distributed storage (HDFS), structured data warehousing (HiveQL), streaming aggregation (MapReduce), and distributed DataFrame transformations (PySpark).
5. **Leak-Free Machine Learning Prediction:** Construct a chronological train/test split and train a multiclass classifier predicting urban congestion levels (`Low`, `Moderate`, `High`) using strictly pre-trip features and historical baselines.
6. **Interactive Dashboard:** Build a high-performance web dashboard with Streamlit, Plotly, and Folium for visual analytics, geospatial hotspot mapping, and real-time inference.

---

## 3. Technology Stack & Architectural Workflow

### 3.1 Technology Stack
| Layer | Technologies | Role / Responsibility |
| :--- | :--- | :--- |
| **Core Runtime** | Python 3.13 (`.venv`) | Isolated execution environment |
| **Ingestion & Tabular Engine** | `pyarrow`, `fastparquet`, `pandas`, `numpy` | Chunked Parquet streaming, schema inspection, vector math |
| **Distributed Computing** | Hadoop MapReduce Streaming, Apache Hive (HiveQL), PySpark | Parallel key-value aggregation, warehouse DDL/queries, distributed DataFrames |
| **Machine Learning** | `scikit-learn`, `joblib` | Chronological split, baseline models, Random Forest classifier, probability inference |
| **Interactive Visualization** | `streamlit`, `plotly`, `folium`, `streamlit-folium` | Light-themed executive dashboard, responsive charts, geospatial hotspot mapping |
| **Testing & Quality Assurance**| `pytest` | 55 automated unit tests covering all modules |

### 3.2 End-to-End Architectural Workflow
```
[NYC TLC CloudFront CDN / OpenData]
                │
                ▼ (Streaming HTTP GET + SHA-256 Checksum)
     data/raw/*.parquet & *.geojson
                │
                ├─────────────────────────────────────────────────┐
                ▼ (Streaming Batch Validation & Outlier Removal)  ▼ (Metadata Inspection)
       src/processing/cleaner.py                       src/ingestion/schema_inspector.py
                │                                                 │
                ▼                                                 ▼
     data/processed/cleaned_trips_*.parquet            reports/dataset_metadata_report.md
                │
   ┌────────────┼─────────────────────────┬────────────────────────┐
   ▼            ▼                         ▼                        ▼
[Traffic Analytics]    [Hadoop MapReduce]     [ML Pipeline]         [Interactive Dashboard]
src/analytics/         mapreduce/             src/prediction/       dashboard/app.py
traffic_analysis.py    mapper.py | reducer.py train_model.py        ├── Summary KPIs
   │                    │                     predict_congestion.py ├── Plotly Analytics
   ▼                    ▼                     │                     ├── Folium Hotspot Map
reports/traffic_       Zone Aggregates        ▼                     └── Live ML Prediction
analysis_report.json   (Volume, Speed, Dist)  models/*.joblib
```

---

## 4. Dataset Provenance, Schema & Spatial Join Architecture

### 4.1 Official Dataset Specifications
All data is sourced directly from verified NYC government and commission portals:

| Dataset | Source Agency | Official Format / Source URL | File Size | Record Count |
| :--- | :--- | :--- | :--- | :--- |
| **Yellow Taxi Trip Records** | NYC TLC | [CloudFront Parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet) | `47.65 MB` | **2,964,624 rows** |
| **Taxi Zone Lookup Table** | NYC TLC | [CloudFront CSV](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv) | `12.33 KB` | **265 zones** |
| **Taxi Zones Spatial Boundaries** | NYC OpenData | [NYC Open Data GeoJSON](https://data.cityofnewyork.us/api/geospatial/8meu-9t5y?method=export&format=GeoJSON) | `3.71 MB` | **263 polygons** |

### 4.2 Verified Parquet Schema (Metadata Footer Inspection)
Schema inspected using `pyarrow.parquet.ParquetFile` without loading raw dataframes:
```text
Total Rows: 2,964,624 | Columns: 19 | Row Groups: 3 | Format Version: 2.6
Compression: SNAPPY | Writer: parquet-cpp-arrow version 14.0.2
```

| Index | Column Name | Arrow Data Type | Physical Parquet Type | Domain Meaning |
| :---: | :--- | :--- | :--- | :--- |
| 0 | `VendorID` | `int32` | `INT32` | TPEP provider (1=Creative Mobile, 2=VeriFone) |
| 1 | `tpep_pickup_datetime` | `timestamp[us]` | `INT64` | Trip start timestamp |
| 2 | `tpep_dropoff_datetime` | `timestamp[us]` | `INT64` | Trip end timestamp |
| 3 | `passenger_count` | `int64` | `INT64` | Passenger count |
| 4 | `trip_distance` | `double` | `DOUBLE` | Elapsed distance in miles |
| 5 | `RatecodeID` | `int64` | `INT64` | Rate code (1=Standard, 2=JFK, 3=Newark, etc.) |
| 6 | `store_and_fwd_flag` | `large_string` | `BYTE_ARRAY` | Memory store & forward flag (Y/N) |
| 7 | `PULocationID` | `int32` | `INT32` | TLC Taxi Zone pickup identifier |
| 8 | `DOLocationID` | `int32` | `INT32` | TLC Taxi Zone dropoff identifier |
| 9 | `payment_type` | `int64` | `INT64` | Payment method (1=Credit, 2=Cash) |
| 10 | `fare_amount` | `double` | `DOUBLE` | Metered fare amount ($) |
| 11 | `extra` | `double` | `DOUBLE` | Miscellaneous extras and surcharges |
| 12 | `mta_tax` | `double` | `DOUBLE` | $0.50 MTA tax automatically triggered |
| 13 | `tip_amount` | `double` | `DOUBLE` | Tip amount (credit card tips only) |
| 14 | `tolls_amount` | `double` | `DOUBLE` | Total bridge and tunnel tolls |
| 15 | `improvement_surcharge`| `double` | `DOUBLE` | $0.30 or $1.00 improvement surcharge |
| 16 | `total_amount` | `double` | `DOUBLE` | Total charged amount |
| 17 | `congestion_surcharge` | `double` | `DOUBLE` | NYS Congestion Surcharge |
| 18 | `Airport_fee` | `double` | `DOUBLE` | Airport pickup fee ($1.25 / $1.75) |

### 4.3 Spatial Reconciliation & Join Architecture
- **Primary & Foreign Keys:**
  - `PULocationID` / `DOLocationID` in trip records (`int32`)
  - `LocationID` in taxi zone lookup CSV (`int64`)
  - `locationid` in taxi zone GeoJSON (`string` / `integer`)
- **Spatial Coverage & Mappings:**
  - 265 zones in lookup CSV.
  - 263 polygon features in GeoJSON.
  - Non-polygonal administrative codes: Zone 264 (*Unknown/Outside NYC*) and Zone 265 (*Out of State/NV*), plus merged island territories (104, 105, 57).

---

## 5. Data Cleaning & Preprocessing Methodology

The cleaning pipeline ([`src/processing/cleaner.py`](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/src/processing/cleaner.py)) processes raw Parquet data in 150,000-record streaming batches:

### 5.1 Validation Rules & Quality Bounds
1. **Critical Field Null Check:** Rows with null timestamps, distances, or pickup IDs are removed.
2. **Timestamp Chronology:** Dropoff time must be strictly greater than pickup time (`tpep_dropoff_datetime > tpep_pickup_datetime`).
3. **Trip Duration Outliers:** Trips must satisfy `1.0 minute <= duration <= 240.0 minutes` (4 hours).
4. **Trip Distance Outliers:** Trips must satisfy `0.1 miles <= distance <= 100.0 miles`.
5. **Estimated Velocity Limits:** Commercial speed calculated as `speed_mph = trip_distance / (duration_minutes / 60.0)`. Trips must satisfy `1.0 mph <= speed <= 65.0 mph` (NYC legal arterial limit).
6. **Location Integrity:** `PULocationID` and `DOLocationID` must be within valid range `[1, 265]`.
7. **Fare Sanity:** Fare amount must be positive (`fare_amount > 0`).

### 5.2 Cleaned Dataset Retention
- **Total Raw Records Ingested:** `2,964,624`
- **Total Valid Cleaned Trips Retained:** `2,825,796` (95.32% retention rate)
- **Engineered Analytical Features:**
  - `pickup_hour` (0–23)
  - `pickup_date` (YYYY-MM-DD)
  - `pickup_day_of_week` (0=Mon, 6=Sun)
  - `is_weekend` (0/1)
  - `congestion_level` (`Low` >= 15 mph, `Moderate` 8–15 mph, `High` < 8 mph)

---

## 6. Verified Traffic Analytics & Empirical Findings

Computed from all **2,825,796 verified trips** (summary report: [`reports/traffic_analysis_report.json`](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/reports/traffic_analysis_report.json)):

### 6.1 Macro Key Performance Indicators (KPIs)
- **Total Analyzed Trips:** `2,825,796`
- **Overall Average Trip Duration:** `15.00 minutes`
- **Overall Estimated Average Speed:** `11.53 mph`
- **Overall Average Trip Distance:** `3.31 miles`
- **Peak Travel Hour:** **Hour 18:00 (6:00 PM)** with `203,411 trips` (speed: `10.15 mph`)
- **Lowest Travel Hour:** **Hour 04:00 (4:00 AM)** with `15,462 trips` (speed: `19.69 mph`)

### 6.2 Top 5 High-Demand Pickup Zones
| Rank | Zone ID | Zone Name | Borough | Total Pickups | Avg Speed | Avg Duration |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `161` | **Midtown Center** | Manhattan | `138,185` | `9.13 mph` | 13.92 min |
| 2 | `237` | **Upper East Side South** | Manhattan | `137,842` | `9.19 mph` | 11.03 min |
| 3 | `132` | **JFK Airport** | Queens | `136,890` | `27.27 mph` | 37.27 min |
| 4 | `236` | **Upper East Side North** | Manhattan | `131,480` | `9.91 mph` | 11.23 min |
| 5 | `162` | **Midtown East** | Manhattan | `102,740` | `9.58 mph` | 13.06 min |

### 6.3 Diurnal Mobility Pattern Insights
- **Free-flow Window (04:00 – 06:00 AM):** Highest urban velocity (`18.07 – 19.69 mph`) due to minimal street friction.
- **Evening Peak Rush (04:00 – 07:00 PM):** Lowest street velocities (`9.88 – 10.37 mph`) paired with massive passenger demand (> 190,000 trips/hr), leading to stop-and-go congestion.
- **Corridor Divergence:** JFK Airport trips exhibit high velocity (`27.27 mph`) due to Van Wyck / Belt Parkway highway transit, while Midtown Center trips experience severe core grid delays (`9.13 mph`).

---

## 7. Big Data Ecosystem Frameworks & Status

The course requirements mandate exploration and evaluation of Hadoop, HDFS, Hive, and PySpark concepts:

| Component | Implementation File | Verification Command | Implementation & Environment Status |
| :--- | :--- | :--- | :--- |
| **Hadoop MapReduce** | [`mapreduce/mapper_test.py`](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/mapreduce/mapper_test.py)<br>[`mapreduce/reducer_test.py`](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/mapreduce/reducer_test.py) | `Get-Content data/sample/sample_trips.csv \| python mapreduce/mapper_test.py \| Sort-Object \| python mapreduce/reducer_test.py` | **Implemented and Verified** (via Hadoop streaming mapper/reducer interface and automated unit tests) |
| **HDFS File Operations** | [`scripts/validate_hdfs.py`](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/scripts/validate_hdfs.py) | `python scripts/validate_hdfs.py` | **Implemented; Environment Blocked** (Target NameNode port 9870 unreachable without running Hadoop cluster) |
| **Apache Hive SQL** | [`hive/validate_hive.sql`](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/hive/validate_hive.sql) | Hive CLI / HiveServer2 | **Implemented; Environment Blocked** (Hive Metastore and HiveServer2 services not running on local OS) |
| **PySpark Processing** | [`scripts/validate_pyspark.py`](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/scripts/validate_pyspark.py) | `python scripts/validate_pyspark.py` | **Implemented; Environment Blocked** (Host OS lacks Java Virtual Machine / JDK and `JAVA_HOME`) |

### MapReduce Demonstration
The Hadoop MapReduce implementation uses the Hadoop Streaming standard input/output protocol:
1. **Mapper:** Emits `LocationID \t Speed,Distance,1`.
2. **Shuffle/Sort:** Partitions and sorts records lexicographically by `LocationID`.
3. **Reducer:** Aggregates metric sums and emits `LocationID \t Trip_Count \t Mean_Speed_mph \t Mean_Distance_miles`.

Output from verified test:
```text
132    2    24.25    14.00
138    1    20.50     9.70
161    3    12.50     2.10
237    2    16.25     3.85
79     2    10.50     1.70
```

---

## 8. Machine Learning Congestion Prediction (Phase 4)

### 8.1 Problem Formulation & Target Classes
Congestion is modeled as a 3-class classification problem based on empirical trip velocity:
- **Low Congestion (`speed >= 15.0 mph`):** Fluid traffic flow with minimal intersection delay.
- **Moderate Congestion (`8.0 mph <= speed < 15.0 mph`):** Standard daytime urban street traffic under signalized arterial cycles.
- **High Congestion (`speed < 8.0 mph`):** Severe corridor friction, stop-and-go traffic, and gridlock.

### 8.2 Strict Data Leakage Prevention
To guarantee that the model can be used for **pre-trip scheduling**, post-trip attributes (`trip_duration`, `actual_distance`, `fare_amount`, `speed_mph`) are strictly excluded from the feature space.

**Feature Vector ($X$):**
1. `pickup_hour` (0–23): Diurnal cycle.
2. `pickup_day_of_week` (0–6): Weekly commute cycle.
3. `is_weekend` (0/1): Weekend vs. weekday transit pattern flag.
4. `PULocationID` (1–265): TLC geographic origin zone.
5. `hist_volume` (float): Historical trip volume for that zone and hour (from training split only).
6. `hist_avg_duration` (float): Historical average trip duration in minutes (from training split only).
7. `hist_avg_speed` (float): Historical average speed in mph (from training split only).

### 8.3 Chronological Train/Test Partitioning
- **Total Dataset:** 199,989 verified trip records sampled uniformly across row groups.
- **Chronological Cutoff Timestamp:** `2024-01-24 15:36:40`
- **Training Set (75%):** `149,991 records` (January 1 to January 24, 2024)
- **Testing Set (25%):** `49,998 records` (January 24 to January 30, 2024)
- Historical baseline statistics were computed **strictly on the training partition**. Unseen zone/hour combinations fall back gracefully to global training medians.

### 8.4 Model Evaluation & Baseline Comparison (Test Set: 49,998 trips)
| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Majority Class: Moderate)** | `49.10%` | `0.1637` | `0.3333` | `0.2195` | `0.3234` |
| **Logistic Regression Baseline** | `55.16%` | `0.6110` | `0.4832` | `0.4608` | `0.4745` |
| **Random Forest Classifier (TrafficPulse)** | **`59.17%`** | **`0.6414`** | **`0.5301`** | **`0.5377`** | **`0.5570`** |

*The Random Forest model demonstrates a **+10.07 percentage point** accuracy improvement over the naive majority-class baseline.*

#### Detailed Classification Report (Random Forest)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Low** (>= 15 mph) | `0.6738` | `0.4560` | `0.5439` | 8,017 |
| **Moderate** (8 - 15 mph) | `0.5624` | `0.8645` | `0.6815` | 24,550 |
| **High** (< 8 mph) | `0.6879` | `0.2698` | `0.3876` | 17,431 |
| **Overall Accuracy** | - | - | **`0.5917`** | **49,998** |
| **Weighted Average** | **`0.6240`** | **`0.5917`** | **`0.5570`** | **49,998** |

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

---

## 9. Interactive Traffic Analytics Dashboard (Phase 5)

Built in [`dashboard/app.py`](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/dashboard/app.py) using Streamlit, Plotly, Folium, and native Leaflet.js with OpenStreetMap API integrations.

### 9.1 Key Features & User Interface
1. **Executive Summary KPI Cards:** Total trips analyzed, average duration, estimated speed, peak travel hour, and busiest pickup zone.
2. **Traffic Analytics & Volume Patterns:** Diurnal hourly volume distribution, duration curves, weekday versus weekend comparison, and top 12 pickup zones.
3. **Geographic Hotspot Visualization (Leaflet & OpenStreetMap API):**
   - **OpenStreetMap Tile Layer Integration:** Switch between OpenStreetMap (Standard), OpenStreetMap Humanitarian (HOT), CartoDB Positron (Light OSM), and OpenTopoMap (OSM Relief) using Leaflet LayerControl.
   - **OpenStreetMap Nominatim Geocoding API:** Interactive location search bar querying the live OpenStreetMap Nominatim API (`https://nominatim.openstreetmap.org/search`) to geocode any NYC address or landmark (e.g. *Times Square*, *Grand Central*, *Wall Street*, *Barclays Center*), compute Euclidean proximity to the nearest NYC TLC taxi zone, and highlight the matched zone on the map.
   - **Dual Leaflet Rendering Engines:** Toggle between Folium Leaflet-backed interactive map and a Direct Leaflet.js (v1.9.4) HTML5 standalone canvas.
   - **Dynamic Hotspot Bubbles:** Sized proportionally by trip volume and color-coded by congestion level (red/yellow/green) with rich popups, optional GeoJSON boundary overlays, and Leaflet MiniMap navigation.
4. **Live Congestion Prediction Panel:** Interactive model inference querying `CongestionPredictor`, displaying predicted congestion category, confidence score, class probability distribution bar chart, and historical baseline context.
5. **Global Dynamic Filters:** Date range selector, borough multi-select, pickup zone picker, hour slider, and day-type selector updating all charts.

### 9.2 Performance & Memory Optimization
- **Selective Parquet Loading:** Reads only 9 analytical columns, loading 2.82 million rows in ~0.24 seconds with ~140 MB of memory.
- **Reference-Based Caching:** Decorated with `@st.cache_resource` for the large trip dataset and model bundle, preventing pickling overhead and eliminating memory errors.
- **Resilient Fallback Hierarchy:** Automatically falls back from full Parquet to sample Parquet to summary CSVs, with clear error messages if files are missing.

---

## 10. Installation & Complete Execution Guide

### 10.1 Prerequisites & Virtual Environment Setup
Ensure Python 3.10+ is installed. Use the project's existing virtual environment:

```powershell
# 1. Navigate to project root
cd "c:\Users\suhai\OneDrive\Desktop\Traffic Pulse"

# 2. Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Verify installed packages
pip list
```

### 10.2 Step-by-Step Execution Commands

#### Ingestion & Schema Inspection
```powershell
# Inspect Parquet metadata schema
python -m src.ingestion.schema_inspector

# Verify spatial join reconciliation
python -m src.geospatial.zone_inspector
```

#### MapReduce Streaming Demonstration
```powershell
# Run MapReduce pipeline locally using PowerShell streaming
Get-Content data/sample/sample_trips.csv | python mapreduce/mapper_test.py | Sort-Object | python mapreduce/reducer_test.py
```

#### Machine Learning Congestion Prediction
```powershell
# Train and evaluate models (saves bundle to src/prediction/models/)
python -m src.prediction.train_model

# Run CLI prediction for Midtown Center (Zone 161) at 6:00 PM (Hour 18) on Wednesday (Day 2)
python -m src.prediction.predict_congestion --hour 18 --zone 161 --day 2

# Run CLI prediction for JFK Airport (Zone 132) at 3:00 AM (Hour 3)
python -m src.prediction.predict_congestion --hour 3 --zone 132 --day 2
```

#### Launch Interactive Dashboard
```powershell
streamlit run dashboard/app.py
```
Open browser at: `http://localhost:8501`.

---

## 11. Comprehensive Test Suite & Quality Verification

Run the complete automated test suite using `pytest`:

```powershell
pytest -v
```

### Verified Test Results: **58 passed in 17.96s**
- `tests/test_cleaner.py`: **6 passed** (data quality bounds, anomaly filtering, duration calculations)
- `tests/test_ingestion.py`: **10 passed** (Parquet schema, checksums, spatial reconciliation)
- `tests/test_traffic_analysis.py`: **7 passed** (hourly aggregations, speed statistics, edge cases)
- `tests/test_mapreduce.py`: **3 passed** (mapper key-value emission, reducer aggregation, streaming pipeline)
- `tests/test_prediction.py`: **19 passed** (chronological split, serialization, input validation, probability calibration, deterministic repeatability)
- `tests/test_dashboard.py`: **13 passed** (selective Parquet loading, zone joins, GeoJSON centroid calculation, KPI consistency, multi-filter logic, model inference, Leaflet HTML generation, OpenStreetMap Nominatim geocoding, closest-zone proximity)

---

## 12. Critical Domain Limitations

> [!NOTE]
> **Academic & Methodological Disclaimers:**
> 1. **Spatial Proxy Caveat:** Taxi zones represent polygonal administrative districts (derived from NYC DCP Neighborhood Tabulation Areas), **not** individual street centerlines, highway links, or fixed inductive loop detector sensors. Speeds reflect commercial origin-to-destination transit velocity including boarding and signal dwell times.
> 2. **Pre-trip Scheduling Scope:** Real-time ML predictions estimate expected conditions based on scheduled temporal parameters and historical zone profiles. Unforeseen instantaneous disruptions (traffic collisions, emergency closures, flash flooding) cannot be anticipated prior to departure.
> 3. **Single-Month Telemetry:** Analytical summaries and baseline lookups reflect January 2024 operations. Summer seasonal fluctuations and extreme winter snow emergencies require continuous multi-month ingestion.

---

## 13. Academic Submission & Team Information

**Institution:** College of Engineering / Department of Computer Science & Engineering  
**Degree:** Bachelor of Engineering (B.E.) / Technology (B.Tech)  
**Course Code & Title:** CCS334 – Big Data Analytics  
**Academic Year:** 2025–2026  

| Team Member Name | Register Number | Role / Responsibilities |
| :--- | :--- | :--- |
| **[Student 1 / Lead]** | `[Register No.]` | Pipeline Architecture, Data Cleaning & Preprocessing |
| **[Student 2]** | `[Register No.]` | Big Data Analytics, MapReduce & Distributed Frameworks |
| **[Student 3]** | `[Register No.]` | Machine Learning Feature Engineering & Modeling |
| **[Student 4]** | `[Register No.]` | Streamlit Dashboard, Visual Presentation & Testing |

---

## 14. Academic Project Report Outline

A complete, formal academic project report outline compliant with college mini-project submission guidelines is provided in:  
👉 **[`docs/PROJECT_REPORT_OUTLINE.md`](file:///c:/Users/suhai/OneDrive/Desktop/Traffic%20Pulse/docs/PROJECT_REPORT_OUTLINE.md)**

It includes chapter-by-chapter breakdowns for:
- Front matter (Certificate, Declaration, Abstract)
- Chapter 1: Introduction & SDG 11 Alignment
- Chapter 2: Literature Review & Technology Evaluation
- Chapter 3: Dataset Ingestion & Preprocessing Architecture
- Chapter 4: Big Data Frameworks (Hadoop, HDFS, Hive, PySpark)
- Chapter 5: Traffic Analytics & Exploratory Data Analysis
- Chapter 6: Machine Learning Congestion Prediction
- Chapter 7: Interactive Dashboard & Visual Presentation
- Chapter 8: Verification, Testing & Academic Evaluation
- Chapter 9: Conclusion & Future Scope
- References
