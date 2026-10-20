# TrafficPulse: Taxi Zone & Spatial Boundary Reconciliation Report

- **Lookup Table:** `data\raw\taxi_zone_lookup.csv` (265 records)
- **GeoJSON Boundaries:** `data\raw\taxi_zones.geojson` (263 polygon features)
- **Identified GeoJSON Join Key:** `locationid`
- **Lookup LocationID Range:** `1` to `265`
- **Inspection Timestamp:** `2026-10-20T16:25:03.754362`

---

## 1. Join Key Architecture

| Dataset | Foreign / Primary Key | Data Type | Notes |
| :--- | :--- | :--- | :--- |
| **TLC Trip Records** | `PULocationID`, `DOLocationID` | `int64` / `int32` | Taxi pickup and dropoff zone references |
| **Taxi Zone Lookup** | `LocationID` | `int64` | Primary key for zone names and boroughs |
| **Taxi Zone GeoJSON** | `locationid` | `integer` / `string` | Spatial boundary polygon mapping |

---

## 2. Borough Distribution (Lookup Table)

| Borough | Total Designated Taxi Zones |
| :--- | :---: |
| **Queens** | 69 |
| **Manhattan** | 69 |
| **Brooklyn** | 61 |
| **Bronx** | 43 |
| **Staten Island** | 20 |
| **EWR** | 1 |
| **Unknown** | 1 |

---

## 3. Discrepancy Analysis & Missing Boundaries
- **Zones in Lookup without GeoJSON Polygons (5 zones):** `57, 104, 105, 264, 265`
  - *Reasoning:* Zone 264 represents 'Unknown / Outside NYC' and Zone 265 represents 'NV (Non-NYC / Out of State)'. These do not possess physical polygons in NYC territory.
- **Zones in GeoJSON without Lookup Match:** `None (100% matched)`

---

## 4. Methodological & Spatial Proxy Limitations
> [!IMPORTANT]
> **Traffic Analysis Limitation:** Taxi zones are irregular polygonal neighborhood boundaries (derived from NYC DCP Neighborhood Tabulation Areas), **not** individual road links, street centerlines, or highway segments.
> 
> When computing speed and congestion metrics:
> 1. A trip from `PULocationID` to `DOLocationID` represents an **aggregate OD corridor travel time**, not a single street's level-of-service.
> 2. Speed reflects commercial urban trip velocity (including traffic lights, passenger boarding, and street routing).
> 3. Intra-zone trips (`PULocationID == DOLocationID`) reflect local circulating traffic within the neighborhood.