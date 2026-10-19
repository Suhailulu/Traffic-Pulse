# TrafficPulse: Big Data Analytics & Predictive Intelligence for Urban Traffic Congestion

**Course:** CCS334 – Big Data Analytics  
**Project Type:** College Mini Project  
**Target SDG:** UN SDG 11 – *Sustainable Cities and Communities (Target 11.2 & 11.a)*  
**Dataset:** NYC TLC Trip Record Dataset (Yellow Taxi Parquet records + Taxi Zone GeoJSON)  

---

## 1. Project Overview

**TrafficPulse** is an end-to-end Big Data analytics and predictive intelligence platform designed to study, quantify, and predict urban traffic congestion using multi-million record trip telemetry from the New York City Taxi and Limousine Commission (TLC).

Urban traffic congestion represents a major economic and environmental burden, contributing to greenhouse gas emissions, commuter delays, and fuel inefficiency. Aligned with **UN Sustainable Development Goal 11 (Sustainable Cities and Communities)**, TrafficPulse leverages distributed storage, parallel computing, and machine learning to transform raw urban mobility records into actionable congestion intelligence.

---

## 2. Technology Stack & Big Data Ecosystem

| Layer | Technologies | Role / Justification |
| :--- | :--- | :--- |
| **Distributed Storage** | Hadoop HDFS | Reliable, distributed, fault-tolerant storage for raw and cleaned Parquet trip files. |
| **Distributed Processing** | Apache Spark (PySpark) | In-memory distributed transformations, feature extraction, and temporal aggregations. |
| **Data Warehouse & SQL** | Apache Hive | Partitioned external tables and HiveQL analytical queries for corridor bottleneck analysis. |
| **Batch Aggregation** | Hadoop MapReduce | Native streaming MapReduce (`mapper.py` / `reducer.py`) for zone-hour speed aggregation. |
| **Machine Learning** | Scikit-Learn | Congestion classification (Random Forest & Gradient Boosting) with zero data leakage. |
| **Interactive Dashboard** | Streamlit | Web-based command center for decision makers and urban planners. |
| **Visual Analytics** | Plotly & Folium | Diurnal curves, speed distributions, and interactive geospatial choropleth maps. |
| **Infrastructure** | Docker & Docker Compose | Containerized Big Data cluster ensuring reproducible HDFS and Hive execution. |

---

## 3. Project Directory Structure

```text
TrafficPulse/
├── data/
│   ├── raw/                 # Ingested raw TLC Parquet datasets (git-ignored)
│   ├── processed/           # Cleaned and validated datasets (git-ignored)
│   └── sample/              # Unit test fixtures and sample records
├── src/
│   ├── ingestion/           # Data downloaders and schema inspectors
│   ├── processing/          # PySpark and Pandas cleaning and validation pipelines
│   ├── analytics/           # Congestion index formulas and bottleneck ranking
│   ├── prediction/          # Leakage-free ML models and evaluation logic
│   └── geospatial/          # Taxi zone GeoJSON boundary mapping
├── dashboard/               # Streamlit application pages and components
├── hive/                    # HiveQL table definitions and analytical queries
├── mapreduce/               # Hadoop MapReduce mapper, reducer, and runner
├── notebooks/               # Exploratory data analysis and experimental validation
├── tests/                   # Pytest test suite for data quality and ML leakage
├── reports/                 # Execution logs, metrics tables, and evaluation outputs
├── scripts/                 # Environment validation and utility automation
├── docs/                    # Architecture diagrams and technical documentation
├── docker-compose.yml       # Hadoop HDFS, Hive, and YARN cluster definition
├── requirements.txt         # Project dependencies
├── .env.example             # Environment variable configuration template
├── .gitignore               # Excludes large binaries, virtual environments, and secrets
└── README.md                # Project documentation and reproduction guide
```

---

## 4. Environment Setup & Reproduction Guide

### Prerequisites
- Python 3.10 – 3.13
- Git 2.x+
- Docker Desktop (for containerized Hadoop HDFS, Hive, and MapReduce)

### Step 1: Clone and Set Up Virtual Environment
```bash
# Clone the repository
git clone <repository-url>
cd "Traffic Pulse"

# Create a Python virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Upgrade pip and install core dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
```bash
# Copy example configuration template
cp .env.example .env
```

### Step 3: Verify Local Environment
Run the automated environment verification script to check your Python runtime, packages, Git, and Docker status:
```bash
python scripts/verify_env.py
```

### Step 4: Big Data Cluster (Docker)
To start the containerized Hadoop HDFS, YARN, and Hive cluster:
```bash
docker compose up -d
```
- **HDFS NameNode UI:** http://localhost:9870
- **YARN ResourceManager UI:** http://localhost:8088
- **HiveServer2 Port:** `localhost:10000`

---

## 5. Development Principles & Verification Rules
1. **Zero Fabrication**: All analytical metrics, speed statistics, and model evaluation scores reflect genuine execution outputs.
2. **Strict Leakage Prevention**: Machine learning feature matrices never include post-trip attributes (`trip_duration`, `fare_amount`, `actual_speed`).
3. **Modular Testing**: Every module is tested before proceeding to downstream components.
