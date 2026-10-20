# TrafficPulse: NYC TLC Trip Record Dataset Schema Inspection Report

- **File Inspected:** `data\raw\yellow_tripdata_2024-01.parquet`
- **File Size:** `47.65 MB` (`49,961,641` bytes)
- **Total Rows (Metadata):** `2,964,624` rows
- **Total Columns:** `19`
- **Total Row Groups:** `3`
- **Parquet Format Version:** `2.6`
- **Generated / Written By:** `parquet-cpp-arrow version 14.0.2`
- **Inspection Timestamp:** `2026-10-20T16:25:02.786400`

---

## 1. Domain Field Categorization

| Category | Identified Columns | Description / Analytical Role |
| :--- | :--- | :--- |
| **Timestamps** | `tpep_pickup_datetime, tpep_dropoff_datetime` | Temporal granularity for diurnal cycle, day-of-week, and rush hour extraction |
| **Spatial Zones** | `PULocationID, DOLocationID` | Pickup (PULocationID) and Dropoff (DOLocationID) taxi zone identifiers |
| **Trip Distance** | `trip_distance` | Core vehicle travel distance in miles for speed calculation |
| **Passenger Capacity** | `passenger_count` | Occupancy load factor |
| **Fares & Surcharges** | `fare_amount, mta_tax, tip_amount, tolls_amount, improvement_surcharge, total_amount, congestion_surcharge, Airport_fee` | Monetary fields (congestion surcharges, airport fees, base fare) |
| **Operational / Metadata** | `VendorID, RatecodeID, store_and_fwd_flag, payment_type, extra` | VendorID, RatecodeID, payment_type, store_and_fwd_flag |

---

## 2. Complete Column Specifications

| # | Column Name | Arrow Data Type | Physical Parquet Type | Nullable |
| :---: | :--- | :--- | :--- | :---: |
| 0 | **`VendorID`** | `int32` | `INT32` | True |
| 1 | **`tpep_pickup_datetime`** | `timestamp[us]` | `INT64` | True |
| 2 | **`tpep_dropoff_datetime`** | `timestamp[us]` | `INT64` | True |
| 3 | **`passenger_count`** | `int64` | `INT64` | True |
| 4 | **`trip_distance`** | `double` | `DOUBLE` | True |
| 5 | **`RatecodeID`** | `int64` | `INT64` | True |
| 6 | **`store_and_fwd_flag`** | `large_string` | `BYTE_ARRAY` | True |
| 7 | **`PULocationID`** | `int32` | `INT32` | True |
| 8 | **`DOLocationID`** | `int32` | `INT32` | True |
| 9 | **`payment_type`** | `int64` | `INT64` | True |
| 10 | **`fare_amount`** | `double` | `DOUBLE` | True |
| 11 | **`extra`** | `double` | `DOUBLE` | True |
| 12 | **`mta_tax`** | `double` | `DOUBLE` | True |
| 13 | **`tip_amount`** | `double` | `DOUBLE` | True |
| 14 | **`tolls_amount`** | `double` | `DOUBLE` | True |
| 15 | **`improvement_surcharge`** | `double` | `DOUBLE` | True |
| 16 | **`total_amount`** | `double` | `DOUBLE` | True |
| 17 | **`congestion_surcharge`** | `double` | `DOUBLE` | True |
| 18 | **`Airport_fee`** | `double` | `DOUBLE` | True |

---

## 3. Row Group Storage Layout

| Row Group Index | Row Count | Uncompressed Size (MB) |
| :---: | :---: | :---: |
| 0 | 1,048,576 | 26.93 MB |
| 1 | 1,048,576 | 26.66 MB |
| 2 | 867,472 | 21.6 MB |

---

## 4. Big Data & Memory Efficiency Notes
- **Zero RAM Loading:** All attributes in this report were extracted directly from the Parquet FileMetaData footer.
- **Distributed Partitioning Strategy:** With PySpark / Hive, partition pruning should target `PULocationID` or date slices derived from `tpep_pickup_datetime`.
- **Data Leakage Safeguard:** Fare-related fields and actual trip duration are strictly post-trip metrics and will be excluded from the predictive congestion feature set.