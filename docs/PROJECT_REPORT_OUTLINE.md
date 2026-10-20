# TrafficPulse: Academic Project Report Outline
**Course:** CCS334 – Big Data Analytics  
**Degree / Program:** B.E. / B.Tech Computer Science and Engineering  
**Project Title:** Big Data Analytics and Predictive Intelligence for Urban Traffic Congestion  
**Target SDG:** UN SDG 11 – Sustainable Cities and Communities (Targets 11.2 & 11.a)  

---

## Academic Report Structure

### Front Matter
1. **Title Page**
   - Project Title: *TrafficPulse: Big Data Analytics & Predictive Intelligence for Urban Traffic Congestion*
   - Course Code & Title: *CCS334 – Big Data Analytics*
   - Student Details: Name, Register Number, Branch/Section
   - Institution Name, Department, City, State, Year
2. **Bonafide Certificate**
   - Signature of Project Supervisor / Faculty Guide
   - Signature of Head of Department (HOD)
3. **Declaration**
   - Statement of original work and academic integrity
4. **Acknowledgement**
   - Faculty guide, department laboratory staff, peer contributors
5. **Abstract**
   - Context, problem addressed, Big Data methodology, ML model results, interactive visual interface, and SDG 11 impact (250–300 words).
6. **Table of Contents**
7. **List of Tables**
8. **List of Figures**
9. **List of Abbreviations & Acronyms**
   - TLC, HDFS, YARN, HiveQL, ML, RF, OD, KPI, SDG, CDN

---

### Chapter 1: Introduction
- **1.1 Background & Motivation:** Urban mobility challenges in mega-cities, vehicular idling, congestion externalities, fuel wastage, and carbon emissions.
- **1.2 Problem Statement:** Lack of scalable Big Data processing pipelines to ingest, analyze, and predict multi-million record urban street traffic flows from raw GPS and fare-meter telemetry without data leakage.
- **1.3 Objectives:**
  - Ingest and validate multi-million row NYC TLC Yellow Taxi records.
  - Implement streaming batch data cleaning and speed metric derivation.
  - Compute diurnal congestion curves, peak hour profiles, and zone hotspot rankings.
  - Demonstrate distributed Big Data processing principles (MapReduce streaming, HDFS storage, Hive schema, PySpark).
  - Train and evaluate a leak-free Machine Learning model predicting congestion states (`Low`, `Moderate`, `High`).
  - Deploy a user-facing interactive Streamlit analytics dashboard.
- **1.4 Alignment with UN SDG 11:** Target 11.2 (sustainable transport systems) and Target 11.a (regional development planning).
- **1.5 Report Organization:** Overview of subsequent chapters.

---

### Chapter 2: Literature Review & Technological Stack
- **2.1 Literature Review:**
  - Existing studies on urban traffic congestion modeling and taxi telemetry as urban probes.
  - Limitations of fixed inductive loop detectors versus floating car data (FCD).
  - Machine learning classification techniques in intelligent transportation systems (ITS).
- **2.2 Comparative Technology Evaluation:**
  - Tabular Processing: PyArrow / Pandas streaming vs. monolithic in-memory loading.
  - Distributed Computing: Hadoop MapReduce, Apache Spark, Apache Hive.
  - Machine Learning: Dummy majority-class baseline, Logistic Regression, and Random Forest Classifier.
  - User Presentation: Streamlit, Plotly Express, and Folium geospatial mapping.

---

### Chapter 3: Dataset Ingestion & Preprocessing Architecture
- **3.1 Dataset Description & Provenance:**
  - NYC TLC Yellow Taxi Trip Records (January 2024: 2,964,624 raw rows, 47.65 MB Parquet).
  - NYC TLC Taxi Zone Lookup (265 zone descriptions).
  - NYC OpenData Taxi Zone Boundaries (263 polygon features in GeoJSON).
- **3.2 Parquet Schema & Compression Layout:**
  - Row group structure, physical Arrow types, metadata footers.
- **3.3 Data Quality Audit & Cleaning Rules:**
  - Outlier bounds: duration (1–240 min), distance (0.1–100 miles), speed (1–65 mph).
  - Rejection tracking: negative durations, missing coordinates, zero-fare anomalies.
- **3.4 Cleaned Dataset Characteristics:**
  - 2,825,796 verified trips retained (retention rate: 95.32%).

---

### Chapter 4: Big Data Frameworks & Distributed Processing
- **4.1 Distributed Storage with Hadoop HDFS:**
  - Block distribution, NameNode metadata management, DataNode replication.
  - WebHDFS REST API interaction and status auditing (`scripts/validate_hdfs.py`).
- **4.2 MapReduce Aggregation Pipeline:**
  - Hadoop Streaming architecture.
  - Key-Value mapper design (`mapreduce/mapper_test.py`): emitting `(LocationID) -> (speed, distance, 1)`.
  - Shuffle and sort mechanism.
  - Reducer aggregation logic (`mapreduce/reducer_test.py`): computing mean velocity, volume, and average distance per zone.
- **4.3 Structured Querying with Apache Hive:**
  - Hive Metastore architecture and external table mapping (`hive/validate_hive.sql`).
  - HiveQL aggregation queries for borough-level mobility metrics.
- **4.4 PySpark Engine Transformations:**
  - Resilient Distributed Datasets (RDDs) vs. Spark DataFrames.
  - Transformations and actions (`scripts/validate_pyspark.py`).
  - Environment requirements (JVM bridge, memory limits).

---

### Chapter 5: Traffic Analytics & Exploratory Data Analysis
- **5.1 Diurnal Hourly Volume Cycle:**
  - Morning rush (08:00–09:00), afternoon lull, and evening peak (18:00–19:00: 203,411 trips).
  - Free-flow early morning period (04:00–05:00: lowest volume 15,462 trips, highest speed 19.69 mph).
- **5.2 Average Trip Duration Dynamics:**
  - Correlation between volume spikes and duration elongation.
- **5.3 Weekday vs. Weekend Transit Patterns:**
  - Commuter vs. leisure mobility differences.
- **5.4 Geographic Demand & Hotspot Analysis:**
  - High-demand pickup zones: Midtown Center (Zone 161), Upper East Side South (Zone 237), JFK Airport (Zone 132).
  - Urban core vs. airport expressway velocity contrasts.

---

### Chapter 6: Machine Learning Congestion Prediction
- **6.1 Problem Formulation & Target Variable:**
  - Multiclass classification: `Low` (>= 15 mph), `Moderate` (8–15 mph), `High` (< 8 mph).
- **6.2 Feature Engineering & Leakage Prevention:**
  - Strict pre-trip feature space: `pickup_hour`, `pickup_day_of_week`, `is_weekend`, `PULocationID`.
  - Historical aggregates (`hist_volume`, `hist_avg_duration`, `hist_avg_speed`) calculated exclusively from the training set.
- **6.3 Chronological Train/Test Partitioning:**
  - 75% Training (Jan 1–24: 149,991 trips) vs. 25% Testing (Jan 24–30: 49,998 trips).
- **6.4 Model Evaluation & Baseline Comparison:**
  - Dummy Baseline (Majority Class): 49.10% accuracy.
  - Logistic Regression: 55.16% accuracy.
  - Random Forest Classifier: **59.17% accuracy** (+10.07 percentage points improvement).
  - Macro Precision: 0.6414 | Weighted F1: 0.5570.
- **6.5 Feature Importance Analysis:**
  - Gini importance rankings: `hist_avg_speed` (45.10%), `hist_avg_duration` (20.29%), `PULocationID` (9.05%).

---

### Chapter 7: Interactive Dashboard & Visual Presentation
- **7.1 Dashboard Design Principles:**
  - Streamlit application framework with light aesthetic, green accents, and responsive layout.
- **7.2 Executive KPI Module:**
  - Dynamic KPI cards computing metrics from active filter slices.
- **7.3 Traffic Analytics Visualization:**
  - Interactive Plotly figures for diurnal distribution, duration curves, weekday vs. weekend, and zone rankings.
- **7.4 Geographic Folium Mapping:**
  - Centroid hotspot markers scaled by volume and colored by speed category.
  - Administrative boundary polygon overlays.
- **7.5 Real-Time Model Inference Interface:**
  - Interactive prediction inputs, probability bar charts, and historical baseline context.
- **7.6 Performance Optimization:**
  - Selective Parquet column loading (~140 MB RAM) and `@st.cache_resource` caching.

---

### Chapter 8: Verification, Testing & Academic Evaluation
- **8.1 Automated Test Suite:**
  - 55 automated unit tests across cleaner, ingestion, traffic analytics, MapReduce, prediction, and dashboard modules.
  - 100% test pass rate with pytest.
- **8.2 System Limitations & Methodological Caveats:**
  - Polygonal zone boundary proxy limitations (not individual road links).
  - Pre-trip baseline scope vs. real-time dynamic accident alerts.
  - Single-month winter telemetry representation.

---

### Chapter 9: Conclusion & Future Scope
- **9.1 Summary of Contributions:**
  - Scalable pipeline processing 2.82 million urban records.
  - Functional MapReduce and Big Data SQL architectures.
  - Validated ML congestion prediction model.
  - Production-ready interactive dashboard.
- **9.2 Future Enhancements:**
  - Ingestion of multi-year seasonal data.
  - Integration of weather telemetry (precipitation, snow).
  - Deployment on managed cloud clusters (Google Cloud Dataproc / AWS EMR).

---

### References
- New York City Taxi and Limousine Commission (TLC) Data Portal.
- United Nations Department of Economic and Social Affairs: Sustainable Development Goal 11.
- Apache Hadoop, Apache Hive, and Apache Spark Official Documentation.
- Scikit-learn Machine Learning Library Documentation.
- Streamlit and Plotly Documentation.
