# TrafficPulse: Data Quality & Cleaning Audit Report

- **Source Parquet:** `data\raw\yellow_tripdata_2024-01.parquet`
- **Cleaned Parquet:** `data\processed\cleaned_trips_2024_01.parquet`
- **Execution Duration:** `8.11 seconds`
- **Total Batches Processed:** `20`
- **Initial Raw Records:** `2,964,624`
- **Retained Clean Records:** `2,825,796`
- **Dataset Retention Rate:** **`95.32%`**

---

## 1. Data Cleaning Rule Violations Breakdown

| Rule / Error Category | Condition | Dropped Records | % of Raw Records |
| :--- | :--- | :---: | :---: |
| **Missing / Null Attributes** | Null in essential trip fields | `0` | `0.00%` |
| **Duration Outliers** | `duration < 1 min` or `> 240 mins` | `36,999` | `1.25%` |
| **Distance Outliers** | `distance < 0.1 mi` or `> 100 mi` | `38,599` | `1.30%` |
| **Invalid Taxi Zones** | `PULocationID` / `DOLocationID` $\notin [1, 265]$ | `0` | `0.00%` |
| **Non-Positive Fare Amount** | `fare_amount <= 0` | `31,754` | `1.07%` |
| **Invalid Passenger Counts** | `passenger_count < 1` | `30,487` | `1.03%` |
| **Unrealistic Speed Anomalies** | `speed < 1 mph` or `> 65 mph` | `989` | `0.03%` |

---

## 2. Derived Features Added
- **`trip_duration_minutes`**: Floating point duration computed from dropoff minus pickup timestamp.
- **`speed_mph`**: Calculated average velocity in miles per hour ($d / (t / 60)$).
- **`pickup_hour`**: Integer (0 to 23) representing diurnal time slice.
- **`pickup_date`**: Calendar date (YYYY-MM-DD).
- **`pickup_day_of_week`**: Day index (0 = Monday, 6 = Sunday).
- **`is_weekend`**: Binary flag (1 for Saturday/Sunday, 0 for weekdays).
- **`congestion_level`**: Transparent target label (`Low`: $\ge 15\text{ mph}$, `Moderate`: $8-15\text{ mph}$, `High`: $< 8\text{ mph}$).