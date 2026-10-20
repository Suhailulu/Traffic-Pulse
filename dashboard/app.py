#!/usr/bin/env python3
"""
TrafficPulse - Interactive Traffic Analytics Dashboard
Course: CCS334 - Big Data Analytics
Phase 5: Interactive Traffic Analytics Dashboard

Features:
- Streamlit application with Plotly visualization engine
- Professional light theme with green accents
- Key summary metrics (total trips, avg duration, avg speed, peak hour, busiest zone)
- Diurnal hourly distributions, duration curves, weekday vs. weekend comparisons
- Geographic hotspot visualization using Folium & NYC TLC Taxi Zone boundaries
- Real-time ML Congestion Prediction using trained Random Forest model
- Caching and memory-safe selective loading
- Comprehensive graceful fallback handling for missing datasets and models
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MiniMap

# Support running from root or dashboard directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt import of streamlit_folium with fallback
try:
    from streamlit_folium import st_folium
    HAS_STREAMLIT_FOLIUM = True
except ImportError:
    HAS_STREAMLIT_FOLIUM = False

# ==============================================================================
# Configuration & Constants
# ==============================================================================
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "src" / "prediction" / "models"

FULL_PARQUET = PROCESSED_DIR / "cleaned_trips_2024_01.parquet"
SAMPLE_PARQUET = PROCESSED_DIR / "cleaned_trips_sample.parquet"
HOURLY_CSV = PROCESSED_DIR / "hourly_traffic_summary.csv"
ZONE_CSV = PROCESSED_DIR / "zone_traffic_summary.csv"
LOOKUP_CSV = RAW_DIR / "taxi_zone_lookup.csv"
GEOJSON_FILE = RAW_DIR / "taxi_zones.geojson"
SUMMARY_JSON = REPORTS_DIR / "traffic_analysis_report.json"
MODEL_BUNDLE = MODELS_DIR / "congestion_rf_model.joblib"
METRICS_JSON = MODELS_DIR / "model_metrics.json"

NEEDED_COLUMNS = [
    "pickup_date",
    "PULocationID",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "trip_duration_minutes",
    "speed_mph",
    "trip_distance",
    "congestion_level"
]

COLOR_PRIMARY = "#198754"       # Bootstrap green / emerald
COLOR_DARK_GREEN = "#0F5132"    # Deep forest green
COLOR_LIGHT_GREEN = "#D1E7DD"   # Subtle light green tint
COLOR_BG_CARD = "#FFFFFF"       # Clean white card background
COLOR_BORDER = "#E9ECEF"        # Light gray border
COLOR_TEXT_DARK = "#212529"     # High-contrast charcoal text
COLOR_TEXT_MUTED = "#6C757D"    # Subtle secondary text

CONGESTION_COLORS = {
    "Low": "#198754",       # Emerald Green (Fluid >= 15 mph)
    "Moderate": "#F59E0B",  # Amber / Warm Orange (8 - 15 mph)
    "High": "#DC3545"       # Red (Severe Congestion < 8 mph)
}

# ==============================================================================
# Data Loading & Caching Layer
# ==============================================================================

@st.cache_resource(show_spinner="Loading TrafficPulse trip dataset...")
def load_trip_dataset() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads trip dataset. Tries full cleaned parquet first, then sample parquet.
    Returns (DataFrame, source_description).
    """
    candidates = [
        (FULL_PARQUET, "Full Processed Dataset (2.82M trips)"),
        (SAMPLE_PARQUET, "Sample Processed Dataset (100k trips)")
    ]

    for path, desc in candidates:
        if path.exists():
            try:
                df = pd.read_parquet(path, columns=NEEDED_COLUMNS)
                # Filter out anomalous years (e.g. 2002, 2009) to keep pure Jan 2024
                df = df[df["pickup_date"].astype(str).str.startswith("2024-01")].copy()
                df["pickup_hour"] = df["pickup_hour"].astype(int)
                df["PULocationID"] = df["PULocationID"].astype(int)
                return df, desc
            except Exception as err:
                print(f"Warning: Failed reading {path}: {err}", file=sys.stderr)
                continue

    return None, "No parquet trip dataset found"


@st.cache_data
def load_zone_lookup() -> Dict[int, Dict[str, Any]]:
    """Loads taxi zone metadata into a dictionary keyed by LocationID."""
    if not LOOKUP_CSV.exists():
        return {}
    try:
        df = pd.read_csv(LOOKUP_CSV)
        lookup = {}
        for _, row in df.iterrows():
            try:
                zid = int(row["LocationID"])
                zone = str(row["Zone"]) if pd.notna(row["Zone"]) else f"Zone {zid}"
                borough = str(row["Borough"]) if pd.notna(row["Borough"]) else "Unknown"
                service_zone = str(row["service_zone"]) if pd.notna(row["service_zone"]) else "Unknown"
                lookup[zid] = {
                    "LocationID": zid,
                    "zone_name": zone,
                    "borough": borough,
                    "service_zone": service_zone,
                    "display_name": f"{zone} ({borough}) [ID: {zid}]"
                }
            except (ValueError, TypeError):
                continue
        return lookup
    except Exception:
        return {}


@st.cache_data
def load_precomputed_summaries() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict]]:
    """Loads pre-aggregated hourly and zone CSV tables and report JSON."""
    hourly_df = None
    zone_df = None
    report_dict = None

    if HOURLY_CSV.exists():
        try:
            hourly_df = pd.read_csv(HOURLY_CSV)
        except Exception:
            pass

    if ZONE_CSV.exists():
        try:
            zone_df = pd.read_csv(ZONE_CSV)
        except Exception:
            pass

    if SUMMARY_JSON.exists():
        try:
            with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
                report_dict = json.load(f)
        except Exception:
            pass

    return hourly_df, zone_df, report_dict


@st.cache_data
def load_geojson_and_centroids() -> Tuple[Optional[dict], Dict[int, Tuple[float, float]]]:
    """
    Loads NYC Taxi Zones GeoJSON and computes fast centroid coordinates for hotspots.
    """
    if not GEOJSON_FILE.exists():
        return None, {}

    try:
        with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
            geo_data = json.load(f)

        centroids = {}
        for feat in geo_data.get("features", []):
            props = feat.get("properties", {})
            zid_val = props.get("locationid")
            if zid_val is None:
                continue
            try:
                zid = int(zid_val)
            except (ValueError, TypeError):
                continue

            geom = feat.get("geometry", {})
            gtype = geom.get("type")
            coords = geom.get("coordinates", [])
            pts = []

            if gtype == "Polygon":
                for ring in coords:
                    for pt in ring:
                        pts.append(pt)
            elif gtype == "MultiPolygon":
                for poly in coords:
                    for ring in poly:
                        for pt in ring:
                            pts.append(pt)

            if pts:
                avg_lon = sum(p[0] for p in pts) / len(pts)
                avg_lat = sum(p[1] for p in pts) / len(pts)
                centroids[zid] = (round(avg_lat, 5), round(avg_lon, 5))

        return geo_data, centroids
    except Exception as err:
        print(f"Error loading GeoJSON: {err}", file=sys.stderr)
        return None, {}


@st.cache_resource
def load_prediction_model():
    """Loads trained CongestionPredictor model bundle."""
    if not MODEL_BUNDLE.exists():
        return None
    try:
        from src.prediction.predict_congestion import CongestionPredictor
        predictor = CongestionPredictor(model_path=MODEL_BUNDLE, zone_lookup_path=LOOKUP_CSV)
        return predictor
    except Exception as err:
        print(f"Error initializing predictor: {err}", file=sys.stderr)
        return None


@st.cache_data
def load_model_metrics() -> Optional[dict]:
    """Loads serialized model metrics JSON."""
    if not METRICS_JSON.exists():
        return None
    try:
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ==============================================================================
# Helper Functions: Metric Computations
# ==============================================================================

def compute_summary_kpis(
    df: Optional[pd.DataFrame],
    zone_lookup: Dict[int, Dict[str, Any]],
    fallback_report: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Computes key performance indicators with consistent domain definitions:
    - Total trips analyzed
    - Average trip duration
    - Estimated average speed
    - Peak travel hour
    - Busiest pickup zone
    """
    if df is not None and not df.empty:
        total_trips = len(df)
        avg_duration = round(float(df["trip_duration_minutes"].mean()), 1)
        avg_speed = round(float(df["speed_mph"].mean()), 1)

        # Peak Hour
        hour_counts = df["pickup_hour"].value_counts()
        peak_hour = int(hour_counts.idxmax())
        peak_volume = int(hour_counts.max())

        # Busiest Zone
        zone_counts = df["PULocationID"].value_counts()
        busiest_id = int(zone_counts.idxmax())
        busiest_vol = int(zone_counts.max())
        zone_info = zone_lookup.get(busiest_id, {})
        busiest_name = zone_info.get("zone_name", f"Zone {busiest_id}")
        busiest_borough = zone_info.get("borough", "Unknown")

        return {
            "total_trips": total_trips,
            "avg_duration_minutes": avg_duration,
            "avg_speed_mph": avg_speed,
            "peak_hour": peak_hour,
            "peak_hour_formatted": f"{peak_hour:02d}:00 - {peak_hour+1:02d}:00",
            "peak_volume": peak_volume,
            "busiest_zone_id": busiest_id,
            "busiest_zone_name": busiest_name,
            "busiest_zone_borough": busiest_borough,
            "busiest_zone_volume": busiest_vol,
            "source": "Active Filtered Data"
        }

    # Fallback to precomputed JSON report if available
    if fallback_report and "traffic_kpis" in fallback_report:
        kpis = fallback_report["traffic_kpis"]
        top_hotspots = fallback_report.get("top_pickup_hotspots", [])
        top_zone = top_hotspots[0] if top_hotspots else {}

        return {
            "total_trips": fallback_report.get("execution_metadata", {}).get("total_valid_trips", 0),
            "avg_duration_minutes": kpis.get("overall_avg_duration_minutes", 0.0),
            "avg_speed_mph": kpis.get("overall_avg_speed_mph", 0.0),
            "peak_hour": kpis.get("peak_travel_hour", 18),
            "peak_hour_formatted": f"{kpis.get('peak_travel_hour', 18):02d}:00",
            "peak_volume": kpis.get("peak_hour_trips", 0),
            "busiest_zone_id": top_zone.get("PULocationID", 0),
            "busiest_zone_name": top_zone.get("zone_name", "Unknown Zone"),
            "busiest_zone_borough": top_zone.get("borough", "Unknown"),
            "busiest_zone_volume": top_zone.get("trip_count", 0),
            "source": "Precomputed Benchmark Summary"
        }

    return {
        "total_trips": 0,
        "avg_duration_minutes": 0.0,
        "avg_speed_mph": 0.0,
        "peak_hour": 0,
        "peak_hour_formatted": "N/A",
        "peak_volume": 0,
        "busiest_zone_id": 0,
        "busiest_zone_name": "No data available",
        "busiest_zone_borough": "N/A",
        "busiest_zone_volume": 0,
        "source": "Unavailable"
    }


def filter_trip_data(
    df: pd.DataFrame,
    date_range: Optional[Tuple[date, date]],
    selected_boroughs: List[str],
    selected_zones: List[int],
    hour_range: Tuple[int, int],
    day_type: str,
    zone_lookup: Dict[int, Dict[str, Any]]
) -> pd.DataFrame:
    """Applies user selections to trip dataframe."""
    if df is None or df.empty:
        return pd.DataFrame()

    filtered = df.copy()

    # 1. Date Range Filter
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start_str = date_range[0].strftime("%Y-%m-%d")
        end_str = date_range[1].strftime("%Y-%m-%d")
        filtered = filtered[(filtered["pickup_date"] >= start_str) & (filtered["pickup_date"] <= end_str)]

    # 2. Hour Range Filter
    filtered = filtered[(filtered["pickup_hour"] >= hour_range[0]) & (filtered["pickup_hour"] <= hour_range[1])]

    # 3. Day Type Filter
    if day_type == "Weekdays Only (Mon-Fri)":
        filtered = filtered[filtered["is_weekend"] == 0]
    elif day_type == "Weekends Only (Sat-Sun)":
        filtered = filtered[filtered["is_weekend"] == 1]

    # 4. Borough Filter
    if selected_boroughs and "All Boroughs" not in selected_boroughs:
        valid_zone_ids = {
            zid for zid, meta in zone_lookup.items()
            if meta.get("borough") in selected_boroughs
        }
        filtered = filtered[filtered["PULocationID"].isin(valid_zone_ids)]

    # 5. Specific Zone Filter
    if selected_zones:
        filtered = filtered[filtered["PULocationID"].isin(selected_zones)]

    return filtered


# ==============================================================================
# OpenStreetMap & Leaflet.js API Integration Helpers
# ==============================================================================

def find_closest_zone(lat: float, lon: float, centroids: Dict[int, Tuple[float, float]]) -> Optional[int]:
    """Finds the closest NYC TLC taxi zone ID to a given (lat, lon) coordinate."""
    if not centroids:
        return None
    best_zid = None
    min_dist_sq = float("inf")
    for zid, (zlat, zlon) in centroids.items():
        dist_sq = (lat - zlat) ** 2 + (lon - zlon) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_zid = zid
    return best_zid


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_osm_location(query: str) -> Optional[Dict[str, Any]]:
    """
    Queries the OpenStreetMap Nominatim Geocoding API for an NYC address/landmark.
    Returns dictionary with lat, lon, display_name, or None on failure.
    """
    if not query or not query.strip():
        return None
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "TrafficPulse-Academic-Project/1.0 (CCS334 Big Data Analytics)"}
    params = {
        "q": f"{query.strip()}, New York",
        "format": "json",
        "limit": 1,
        "addressdetails": 1
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200 and resp.json():
            item = resp.json()[0]
            return {
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "display_name": item.get("display_name", query),
                "type": item.get("type", "location")
            }
    except Exception as err:
        print(f"OSM Nominatim lookup error: {err}", file=sys.stderr)
    return None


def generate_direct_leaflet_html(
    center: Tuple[float, float],
    zoom: int,
    top_zones: pd.DataFrame,
    centroids: Dict[int, Tuple[float, float]],
    zone_lookup: Dict[int, Dict[str, Any]],
    tile_provider: str = "OpenStreetMap (Standard)",
    searched_loc: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generates standalone HTML with pure Leaflet.js (v1.9.4) and OpenStreetMap API tiles.
    """
    tile_urls = {
        "OpenStreetMap (Standard)": ("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"),
        "OpenStreetMap Humanitarian (HOT)": ("https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png", "&copy; OpenStreetMap contributors, Tiles style by Humanitarian OpenStreetMap Team"),
        "CartoDB Positron (Light OSM)": ("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png", "&copy; OpenStreetMap contributors &copy; CARTO"),
        "OpenTopoMap (OSM Relief)": ("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", "Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap")
    }

    tile_url, tile_attr = tile_urls.get(tile_provider, tile_urls["OpenStreetMap (Standard)"])

    markers_js = []
    max_trips = max(top_zones["trip_count"].max(), 1) if not top_zones.empty else 1

    for _, row in top_zones.iterrows():
        zid = int(row["PULocationID"])
        if zid not in centroids:
            continue
        lat, lon = centroids[zid]
        trips = int(row["trip_count"])
        speed = round(float(row["avg_speed"]), 1)
        dur = round(float(row["avg_duration"]), 1)
        zinfo = zone_lookup.get(zid, {})
        zname = zinfo.get("zone_name", f"Zone {zid}").replace("'", "\\'")
        borough = zinfo.get("borough", "Unknown").replace("'", "\\'")

        radius = round(6 + 16 * np.sqrt(trips / max_trips), 1)

        if speed < 8.0:
            color = "#DC3545"
            state = "High Congestion"
        elif speed < 15.0:
            color = "#F59E0B"
            state = "Moderate Congestion"
        else:
            color = "#198754"
            state = "Fluid Traffic"

        popup_content = (
            f"<div style='font-family:sans-serif;font-size:12px;min-width:180px;'>"
            f"<h4 style='margin:0 0 4px 0;color:#0F5132;'>{zname}</h4>"
            f"<div style='color:#6C757D;margin-bottom:6px;'>{borough} (ID: {zid})</div>"
            f"<b>Trips:</b> {trips:,}<br>"
            f"<b>Avg Speed:</b> {speed} mph<br>"
            f"<b>Avg Duration:</b> {dur} mins<br>"
            f"<b>State:</b> <span style='color:{color};font-weight:bold;'>{state}</span>"
            f"</div>"
        )

        markers_js.append(
            f"L.circleMarker([{lat}, {lon}], {{"
            f"  radius: {radius},"
            f"  fillColor: '{color}',"
            f"  color: '{color}',"
            f"  weight: 1.5,"
            f"  opacity: 1,"
            f"  fillOpacity: 0.65"
            f"}}).bindPopup(\"{popup_content}\").bindTooltip('{zname}: {trips:,} trips | {speed} mph').addTo(map);"
        )

    search_marker_js = ""
    if searched_loc:
        slat = searched_loc["lat"]
        slon = searched_loc["lon"]
        sname = searched_loc["display_name"].replace("'", "\\'")
        szid = searched_loc.get("closest_zone_id", "N/A")
        szname = searched_loc.get("closest_zone_name", "Unknown").replace("'", "\\'")

        search_popup = (
            f"<div style='font-family:sans-serif;font-size:12px;min-width:200px;'>"
            f"<h4 style='margin:0 0 4px 0;color:#0F5132;'>📍 OSM Geocoded Point</h4>"
            f"<div style='margin-bottom:6px;'>{sname}</div>"
            f"<b>Matched Taxi Zone:</b> {szname} (ID: {szid})<br>"
            f"<b>Coordinates:</b> {slat:.4f}, {slon:.4f}"
            f"</div>"
        )
        search_marker_js = (
            f"var searchIcon = L.divIcon({{"
            f"  html: '<div style=\"background-color:#0F5132;color:white;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 10px rgba(15,81,50,0.8);font-size:16px;\">★</div>',"
            f"  className: 'osm-custom-pin',"
            f"  iconSize: [28, 28],"
            f"  iconAnchor: [14, 14]"
            f"}});"
            f"var sMarker = L.marker([{slat}, {slon}], {{icon: searchIcon}}).bindPopup(\"{search_popup}\").addTo(map);"
            f"sMarker.openPopup();"
            f"map.setView([{slat}, {slon}], 13);"
        )

    markers_str = "\n      ".join(markers_js)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin: 0; padding: 0; background: #F8F9FA; }}
    #leaflet-map {{ height: 520px; width: 100%; border-radius: 8px; border: 1px solid #E9ECEF; }}
    .leaflet-popup-content-wrapper {{ border-radius: 6px; box-shadow: 0 3px 14px rgba(0,0,0,0.2); }}
  </style>
</head>
<body>
  <div id="leaflet-map"></div>
  <script>
    var map = L.map('leaflet-map').setView([{center[0]}, {center[1]}], {zoom});
    
    L.tileLayer('{tile_url}', {{
      maxZoom: 19,
      attribution: "{tile_attr}"
    }}).addTo(map);

    L.control.scale({{ metric: true, imperial: true }}).addTo(map);

    // Hotspot circle markers
    {markers_str}

    // Geocoded search marker if present
    {search_marker_js}
  </script>
</body>
</html>
"""
    return html


# ==============================================================================
# UI View: Custom CSS & Styling (Light Theme + Professional Green Accent)
# ==============================================================================

def apply_custom_styles():
    """Injects refined CSS to guarantee a light theme with clean green accents."""
    st.markdown("""
        <style>
        /* Overall Page Canvas */
        .stApp {
            background-color: #F8F9FA;
            color: #212529;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        /* Top Brand Header */
        .brand-header {
            background: linear-gradient(135deg, #0F5132 0%, #198754 100%);
            padding: 24px 28px;
            border-radius: 10px;
            color: #FFFFFF;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(15, 81, 50, 0.12);
        }
        .brand-header h1 {
            color: #FFFFFF !important;
            margin: 0 0 6px 0;
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .brand-header p {
            color: #D1E7DD;
            margin: 0;
            font-size: 14px;
        }

        /* KPI Metric Cards */
        .metric-card {
            background-color: #FFFFFF;
            border: 1px solid #E9ECEF;
            border-left: 5px solid #198754;
            border-radius: 8px;
            padding: 16px 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            margin-bottom: 12px;
            transition: none;
        }
        .metric-title {
            font-size: 13px;
            font-weight: 600;
            color: #6C757D;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #212529;
            line-height: 1.1;
        }
        .metric-subtitle {
            font-size: 12px;
            color: #198754;
            margin-top: 6px;
            font-weight: 500;
        }

        /* Information / Disclaimer Box */
        .info-callout {
            background-color: #F4FBF7;
            border: 1px solid #C3E6CB;
            border-left: 4px solid #198754;
            border-radius: 6px;
            padding: 12px 16px;
            font-size: 13px;
            color: #155724;
            margin-bottom: 16px;
        }

        .warning-callout {
            background-color: #FFF9DB;
            border: 1px solid #FFE066;
            border-left: 4px solid #F59E0B;
            border-radius: 6px;
            padding: 12px 16px;
            font-size: 13px;
            color: #7A5B04;
            margin-bottom: 16px;
        }

        /* Section Titles */
        .section-header {
            font-size: 18px;
            font-weight: 700;
            color: #0F5132;
            margin: 20px 0 12px 0;
            padding-bottom: 6px;
            border-bottom: 2px solid #E9ECEF;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E9ECEF;
        }

        /* Streamlit Button styling */
        .stButton>button {
            background-color: #198754;
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #0F5132;
            color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, subtitle: str = ""):
    """Renders a structured KPI card using styled HTML."""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# View 1: Executive Overview & Summary Metrics
# ==============================================================================

def render_overview_section(
    df_filtered: pd.DataFrame,
    kpis: Dict[str, Any],
    hourly_df: Optional[pd.DataFrame],
    zone_lookup: Dict[int, Dict[str, Any]]
):
    """Renders Summary KPI cards and core traffic analytics charts."""
    st.markdown('<div class="section-header">1. Executive Summary KPIs</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Total Trips", f"{kpis['total_trips']:,}", f"Source: {kpis['source']}")
    with col2:
        render_kpi_card("Avg Trip Duration", f"{kpis['avg_duration_minutes']} min", "Point-to-point elapsed")
    with col3:
        render_kpi_card("Estimated Speed", f"{kpis['avg_speed_mph']} mph", "NYC street grid velocity")
    with col4:
        render_kpi_card("Peak Travel Hour", f"{kpis['peak_hour']:02d}:00", f"{kpis['peak_volume']:,} trips recorded")
    with col5:
        render_kpi_card("Busiest Zone", kpis["busiest_zone_name"][:18], f"{kpis['busiest_zone_volume']:,} trips ({kpis['busiest_zone_borough']})")

    st.markdown('<div class="section-header">2. Traffic Analytics & Volume Patterns</div>', unsafe_allow_html=True)

    # 1. Hourly Trip Volume Chart & 2. Average Trip Duration by Hour
    row1_col1, row1_col2 = st.columns(2)

    if df_filtered is not None and not df_filtered.empty:
        # Group from filtered data
        hourly_stats = df_filtered.groupby("pickup_hour").agg(
            trip_count=("trip_duration_minutes", "count"),
            avg_duration=("trip_duration_minutes", "mean"),
            avg_speed=("speed_mph", "mean")
        ).reindex(range(24), fill_value=0).reset_index()
    elif hourly_df is not None and not hourly_df.empty:
        hourly_stats = hourly_df.rename(columns={"hour": "pickup_hour", "avg_duration_minutes": "avg_duration", "avg_speed_mph": "avg_speed"})
    else:
        hourly_stats = pd.DataFrame()

    with row1_col1:
        if not hourly_stats.empty:
            peak_h = kpis["peak_hour"]
            # Color peak hour darker
            colors = [COLOR_DARK_GREEN if h == peak_h else COLOR_PRIMARY for h in hourly_stats["pickup_hour"]]

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=hourly_stats["pickup_hour"],
                y=hourly_stats["trip_count"],
                marker=dict(color=colors, line=dict(color="#0F5132", width=1)),
                hovertemplate="<b>Hour %{x}:00</b><br>Trips: %{y:,}<extra></extra>"
            ))
            fig_vol.update_layout(
                title=dict(text="Hourly Trip Volume Distribution (24-Hour Cycle)", font=dict(size=14, color=COLOR_TEXT_DARK)),
                xaxis=dict(title="Pickup Hour of Day", tickmode="linear", tick0=0, dtick=2, gridcolor="#F1F3F5"),
                yaxis=dict(title="Number of Trips", gridcolor="#F1F3F5"),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=40, r=20, t=40, b=40),
                height=340
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Hourly volume data is unavailable.")

    with row1_col2:
        if not hourly_stats.empty:
            fig_dur = go.Figure()
            fig_dur.add_trace(go.Scatter(
                x=hourly_stats["pickup_hour"],
                y=hourly_stats["avg_duration"].round(1),
                mode="lines+markers",
                line=dict(color=COLOR_PRIMARY, width=3),
                marker=dict(size=6, color=COLOR_DARK_GREEN),
                fill="tozeroy",
                fillcolor="rgba(25, 135, 84, 0.08)",
                hovertemplate="<b>Hour %{x}:00</b><br>Avg Duration: %{y:.1f} mins<extra></extra>"
            ))
            fig_dur.update_layout(
                title=dict(text="Average Trip Duration by Hour", font=dict(size=14, color=COLOR_TEXT_DARK)),
                xaxis=dict(title="Pickup Hour of Day", tickmode="linear", tick0=0, dtick=2, gridcolor="#F1F3F5"),
                yaxis=dict(title="Duration (Minutes)", gridcolor="#F1F3F5"),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=40, r=20, t=40, b=40),
                height=340
            )
            st.plotly_chart(fig_dur, use_container_width=True)
        else:
            st.info("Hourly duration data is unavailable.")

    # 3. Weekday vs. Weekend Comparison & 4. Pickup-Zone Trip Volume
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        if df_filtered is not None and not df_filtered.empty and "is_weekend" in df_filtered.columns:
            w_comp = df_filtered.groupby(["pickup_hour", "is_weekend"]).agg(
                trip_count=("trip_duration_minutes", "count"),
                avg_speed=("speed_mph", "mean")
            ).reset_index()
            w_comp["Type"] = w_comp["is_weekend"].map({0: "Weekday (Mon-Fri)", 1: "Weekend (Sat-Sun)"})

            fig_comp = px.bar(
                w_comp,
                x="pickup_hour",
                y="trip_count",
                color="Type",
                barmode="group",
                color_discrete_map={"Weekday (Mon-Fri)": COLOR_PRIMARY, "Weekend (Sat-Sun)": "#6C757D"},
                labels={"pickup_hour": "Pickup Hour", "trip_count": "Trip Count", "Type": "Period"},
                title="Weekday vs. Weekend Trip Volume Comparison"
            )
            fig_comp.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                xaxis=dict(tickmode="linear", tick0=0, dtick=2, gridcolor="#F1F3F5"),
                yaxis=dict(gridcolor="#F1F3F5"),
                margin=dict(l=40, r=20, t=40, b=40),
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Weekday/Weekend granular comparison requires the parquet dataset.")

    with row2_col2:
        # Top Pickup Zones
        if df_filtered is not None and not df_filtered.empty:
            zone_vols = df_filtered.groupby("PULocationID").agg(
                trip_count=("trip_duration_minutes", "count"),
                avg_speed=("speed_mph", "mean")
            ).reset_index().sort_values(by="trip_count", ascending=False).head(12)

            zone_vols["zone_label"] = zone_vols["PULocationID"].apply(
                lambda zid: f"{zone_lookup.get(zid, {}).get('zone_name', f'Zone {zid}')} ({zone_lookup.get(zid, {}).get('borough', 'NYC')[:4]})"
            )

            fig_zone = px.bar(
                zone_vols.sort_values(by="trip_count", ascending=True),
                x="trip_count",
                y="zone_label",
                orientation="h",
                color="avg_speed",
                color_continuous_scale="Viridis",
                labels={"trip_count": "Trip Volume", "zone_label": "Pickup Zone", "avg_speed": "Avg Speed (mph)"},
                title="Top 12 High-Demand Pickup Zones"
            )
            fig_zone.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                xaxis=dict(gridcolor="#F1F3F5"),
                yaxis=dict(gridcolor="#F1F3F5"),
                margin=dict(l=40, r=20, t=40, b=40),
                height=340
            )
            st.plotly_chart(fig_zone, use_container_width=True)
        else:
            st.info("Zone demand rankings require active data.")


# ==============================================================================
# View 2: Geographic Visualization (Folium Map)
# ==============================================================================

def render_geographic_section(
    df_filtered: Optional[pd.DataFrame],
    geo_data: Optional[dict],
    centroids: Dict[int, Tuple[float, float]],
    zone_lookup: Dict[int, Dict[str, Any]],
    zone_summary_df: Optional[pd.DataFrame]
):
    """
    Renders interactive Leaflet.js maps with OpenStreetMap API tile layers,
    OpenStreetMap Nominatim geocoding search, and TLC taxi zone hotspot overlays.
    """
    st.markdown('<div class="section-header">Geographic Hotspots & Spatial Distribution (Leaflet & OpenStreetMap API)</div>', unsafe_allow_html=True)

    # Required Methodology Disclaimer
    st.markdown("""
        <div class="info-callout">
            <strong>Methodology & Spatial Boundary Limitation:</strong>
            Taxi zones represent irregular administrative polygons (derived from NYC DCP Neighborhood Tabulation Areas),
            <strong>not</strong> individual road links, street centerlines, or highway lane segments.
            Speeds reflect aggregate commercial origin-to-destination transit velocity including boarding and signal dwell times.
        </div>
    """, unsafe_allow_html=True)

    # Check if geographic boundaries are available
    if not geo_data or not centroids:
        st.markdown("""
            <div class="warning-callout">
                <strong>Geographic Boundary Notice:</strong>
                The official NYC Taxi Zone GeoJSON boundary file (<code>data/raw/taxi_zones.geojson</code>) is currently unavailable.
                Map rendering has been gracefully disabled to prevent inaccurate or fabricated spatial approximations.
            </div>
        """, unsafe_allow_html=True)
        return

    # Aggregate zone-level indicators
    if df_filtered is not None and not df_filtered.empty:
        zone_agg = df_filtered.groupby("PULocationID").agg(
            trip_count=("trip_duration_minutes", "count"),
            avg_duration=("trip_duration_minutes", "mean"),
            avg_speed=("speed_mph", "mean")
        ).reset_index()
    elif zone_summary_df is not None and not zone_summary_df.empty:
        zone_agg = zone_summary_df.rename(columns={
            "avg_duration_minutes": "avg_duration",
            "avg_speed_mph": "avg_speed"
        })
    else:
        st.warning("No zone summary data available to populate the map.")
        return

    # --------------------------------------------------------------------------
    # 1. OpenStreetMap Nominatim Geocoding API Search Bar
    # --------------------------------------------------------------------------
    with st.expander("🔍 Search NYC Location via OpenStreetMap (OSM) Nominatim API", expanded=False):
        st.caption("Enter an NYC address, landmark, or street name. TrafficPulse queries the OpenStreetMap Nominatim API, locates coordinates, and links to the closest NYC TLC Taxi Zone.")
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            osm_query = st.text_input(
                "Search Location / Landmark",
                value=st.session_state.get("osm_search_text", ""),
                placeholder="e.g. Times Square, Grand Central, Wall Street, JFK Airport, Barclays Center",
                label_visibility="collapsed"
            )
        with search_col2:
            search_clicked = st.button("Geocode via OSM", use_container_width=True)

        if search_clicked and osm_query.strip():
            with st.spinner("Querying OpenStreetMap Nominatim API..."):
                geocoded = geocode_osm_location(osm_query)
                if geocoded:
                    best_zid = find_closest_zone(geocoded["lat"], geocoded["lon"], centroids)
                    geocoded["closest_zone_id"] = best_zid
                    zinfo = zone_lookup.get(best_zid, {}) if best_zid else {}
                    geocoded["closest_zone_name"] = zinfo.get("zone_name", "Unknown Zone")
                    geocoded["closest_zone_borough"] = zinfo.get("borough", "Unknown")
                    st.session_state["osm_searched_loc"] = geocoded
                    st.session_state["osm_search_text"] = osm_query
                    st.success(f"📍 **Found via OpenStreetMap:** {geocoded['display_name'][:80]}... → **Matched Taxi Zone:** {geocoded['closest_zone_name']} (ID: {best_zid}, {geocoded['closest_zone_borough']})")
                else:
                    st.warning("No matching NYC location found via OpenStreetMap Nominatim. Try a different query (e.g. 'Times Square, New York').")
        elif "osm_searched_loc" in st.session_state and st.session_state["osm_searched_loc"]:
            geocoded = st.session_state["osm_searched_loc"]
            st.info(f"📍 Active OSM Pin: **{geocoded.get('display_name', '')[:70]}...** (Matched Zone: {geocoded.get('closest_zone_name')} [ID: {geocoded.get('closest_zone_id')}])")

    searched_loc = st.session_state.get("osm_searched_loc", None)

    # --------------------------------------------------------------------------
    # 2. Map Configuration & Canvas
    # --------------------------------------------------------------------------
    col_map_opts, col_map_disp = st.columns([1, 4])

    with col_map_opts:
        st.markdown("**Leaflet & OSM Layer Controls**")
        map_engine = st.radio(
            "Rendering Engine",
            ["Leaflet (Folium-powered)", "Direct Leaflet.js HTML5"],
            index=0,
            help="Choose between Folium Leaflet wrapper or native standalone Leaflet 1.9.4 HTML5 canvas."
        )

        tile_choice = st.selectbox(
            "OpenStreetMap Base Tile Layer",
            [
                "OpenStreetMap (Standard)",
                "OpenStreetMap Humanitarian (HOT)",
                "CartoDB Positron (Light OSM)",
                "OpenTopoMap (OSM Relief)"
            ],
            index=0
        )

        show_hotspots = st.checkbox("Show Hotspot Bubbles", value=True)
        show_boundaries = st.checkbox("Show Zone Boundaries", value=False)
        show_minimap = st.checkbox("Enable Leaflet MiniMap", value=True)
        top_n_zones = st.slider("Top N Zones to Plot", min_value=10, max_value=100, value=40, step=10)

        st.markdown("""
            <div style="font-size: 12px; margin-top: 15px; color: #495057;">
                <strong>Hotspot Color Key:</strong><br>
                <span style="color:#DC3545;">●</span> High Congestion (&lt; 8 mph)<br>
                <span style="color:#F59E0B;">●</span> Moderate (8 - 15 mph)<br>
                <span style="color:#198754;">●</span> Fluid Flow (&gt;= 15 mph)
            </div>
            <div style="font-size: 11px; margin-top: 12px; color: #6C757D;">
                Map data &copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors via Leaflet.js.
            </div>
        """, unsafe_allow_html=True)

    with col_map_disp:
        # Determine center coordinates
        if searched_loc:
            map_center = [searched_loc["lat"], searched_loc["lon"]]
            map_zoom = 13
        else:
            map_center = [40.730610, -73.935242]
            map_zoom = 11

        top_zones = zone_agg.sort_values(by="trip_count", ascending=False).head(top_n_zones)

        # Mode A: Direct Leaflet.js HTML5 Canvas
        if map_engine == "Direct Leaflet.js HTML5":
            leaflet_html = generate_direct_leaflet_html(
                center=map_center,
                zoom=map_zoom,
                top_zones=top_zones,
                centroids=centroids,
                zone_lookup=zone_lookup,
                tile_provider=tile_choice,
                searched_loc=searched_loc
            )
            components.html(leaflet_html, height=530)

        # Mode B: Folium Leaflet-powered Engine
        else:
            # Configure tile layer according to user selection
            if tile_choice == "OpenStreetMap (Standard)":
                tiles_param = "OpenStreetMap"
                attr_param = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            elif tile_choice == "OpenStreetMap Humanitarian (HOT)":
                tiles_param = "https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png"
                attr_param = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by Humanitarian OpenStreetMap Team'
            elif tile_choice == "OpenTopoMap (OSM Relief)":
                tiles_param = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                attr_param = 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, SRTM | Map style: &copy; OpenTopoMap'
            else:
                tiles_param = "cartodbpositron"
                attr_param = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'

            m = folium.Map(
                location=map_center,
                zoom_start=map_zoom,
                tiles=tiles_param,
                attr=attr_param if "http" in tiles_param else None
            )

            # Add alternate OpenStreetMap tile layers for user toggling via Leaflet LayerControl
            if tile_choice != "OpenStreetMap (Standard)":
                folium.TileLayer("OpenStreetMap", name="OpenStreetMap Standard").add_to(m)
            if tile_choice != "CartoDB Positron (Light OSM)":
                folium.TileLayer("cartodbpositron", name="CartoDB Positron (Light)").add_to(m)

            # Optional Leaflet MiniMap plugin
            if show_minimap:
                MiniMap(tile_layer="OpenStreetMap", position="bottomright", width=120, height=100).add_to(m)

            # 1. Optional GeoJSON boundary outlines
            if show_boundaries and geo_data:
                folium.GeoJson(
                    geo_data,
                    name="Taxi Zone Boundaries",
                    style_function=lambda x: {
                        "fillColor": "#198754",
                        "color": "#0F5132",
                        "weight": 0.8,
                        "fillOpacity": 0.08
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=["zone", "borough", "locationid"],
                        aliases=["Zone:", "Borough:", "Location ID:"]
                    )
                ).add_to(m)

            # 2. Zone Hotspots (Centroid Circle Markers)
            if show_hotspots:
                max_trips = max(top_zones["trip_count"].max(), 1)

                for _, row in top_zones.iterrows():
                    zid = int(row["PULocationID"])
                    if zid not in centroids:
                        continue

                    lat, lon = centroids[zid]
                    trips = int(row["trip_count"])
                    speed = round(float(row["avg_speed"]), 1)
                    dur = round(float(row["avg_duration"]), 1)
                    zinfo = zone_lookup.get(zid, {})
                    zname = zinfo.get("zone_name", f"Zone {zid}")
                    borough = zinfo.get("borough", "Unknown")

                    # Scale circle radius between 6 and 22 pixels
                    radius = 6 + 16 * np.sqrt(trips / max_trips)

                    # Determine color by average speed
                    if speed < 8.0:
                        marker_color = "#DC3545"
                        level_label = "High Congestion"
                    elif speed < 15.0:
                        marker_color = "#F59E0B"
                        level_label = "Moderate Congestion"
                    else:
                        marker_color = "#198754"
                        level_label = "Fluid Traffic"

                    popup_html = f"""
                        <div style="font-family: sans-serif; font-size: 12px; width: 200px;">
                            <h4 style="margin: 0 0 4px 0; color: #0F5132;">{zname}</h4>
                            <div style="color: #6C757D; margin-bottom: 6px;">{borough} (ID {zid})</div>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr><td><strong>Trips:</strong></td><td align="right">{trips:,}</td></tr>
                                <tr><td><strong>Avg Speed:</strong></td><td align="right">{speed} mph</td></tr>
                                <tr><td><strong>Avg Duration:</strong></td><td align="right">{dur} mins</td></tr>
                                <tr><td><strong>State:</strong></td><td align="right" style="color:{marker_color}; font-weight:bold;">{level_label}</td></tr>
                            </table>
                        </div>
                    """

                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=radius,
                        popup=folium.Popup(popup_html, max_width=240),
                        tooltip=f"{zname} ({borough}): {trips:,} trips | {speed} mph",
                        color=marker_color,
                        fill=True,
                        fill_color=marker_color,
                        fill_opacity=0.65,
                        weight=1.5
                    ).add_to(m)

            # 3. Add Geocoded Search Pin Marker if location was queried
            if searched_loc:
                search_pin_html = f"""
                    <div style="font-family: sans-serif; font-size: 12px; width: 220px;">
                        <h4 style="margin: 0 0 4px 0; color: #0F5132;">📍 OSM Geocoded Point</h4>
                        <div style="margin-bottom: 6px; font-size: 11px;">{searched_loc.get('display_name', '')[:100]}</div>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr><td><strong>Closest Zone:</strong></td><td align="right"><b>{searched_loc.get('closest_zone_name')}</b></td></tr>
                            <tr><td><strong>Zone ID:</strong></td><td align="right">{searched_loc.get('closest_zone_id')} ({searched_loc.get('closest_zone_borough')})</td></tr>
                            <tr><td><strong>Coordinates:</strong></td><td align="right">{searched_loc['lat']:.4f}, {searched_loc['lon']:.4f}</td></tr>
                        </table>
                    </div>
                """
                folium.Marker(
                    location=[searched_loc["lat"], searched_loc["lon"]],
                    popup=folium.Popup(search_pin_html, max_width=260),
                    tooltip=f"OSM Search: {searched_loc.get('display_name', '')[:40]}",
                    icon=folium.Icon(color="green", icon="star", prefix="fa")
                ).add_to(m)

            # Add Leaflet Layer Control
            folium.LayerControl(position="topright", collapsed=True).add_to(m)

            # Render Map via streamlit_folium or components.html
            if HAS_STREAMLIT_FOLIUM:
                st_folium(m, width=None, height=520, returned_objects=[])
            else:
                components.html(m._repr_html_(), height=520)


# ==============================================================================
# View 3: Congestion Prediction Engine (Phase 4 Model Integration)
# ==============================================================================

def render_prediction_section(
    predictor,
    zone_lookup: Dict[int, Dict[str, Any]],
    model_metrics: Optional[dict]
):
    """Renders real-time interactive ML congestion inference panel."""
    st.markdown('<div class="section-header">Live Machine Learning Congestion Predictor</div>', unsafe_allow_html=True)

    if predictor is None:
        st.markdown("""
            <div class="warning-callout">
                <strong>Prediction Engine Unavailable:</strong>
                Trained model bundle not found at <code>src/prediction/models/congestion_rf_model.joblib</code>.
                Predictions cannot be computed until the model training pipeline has completed via:
                <br><code>python -m src.prediction.train_model</code>.
                <br><em>No synthetic or fabricated predictions will be generated.</em>
            </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("""
        <div class="info-callout">
            <strong>Model Architecture & Scope:</strong>
            Powered by a serialized <strong>Random Forest Classifier</strong> trained strictly on pre-trip features (temporal cycle and historical zone baselines).
            Predictions classify expected commercial transit conditions: <strong>Low</strong> (&ge; 15 mph), <strong>Moderate</strong> (8 - 15 mph), or <strong>High</strong> (&lt; 8 mph).
        </div>
    """, unsafe_allow_html=True)

    # Prediction Parameter Form
    pred_col1, pred_col2 = st.columns([1, 1])

    with pred_col1:
        st.markdown("##### Query Parameters")

        # Zone selection sorted alphabetically
        zone_options = sorted(
            [meta["display_name"] for meta in zone_lookup.values() if meta["borough"] != "Unknown"]
        )
        # Default to Midtown Center (ID 161)
        default_index = 0
        for i, opt in enumerate(zone_options):
            if "[ID: 161]" in opt:
                default_index = i
                break

        selected_zone_str = st.selectbox("Pickup Taxi Zone", zone_options, index=default_index)
        # Extract zone ID from label "[ID: XXX]"
        selected_zone_id = int(selected_zone_str.split("[ID: ")[1].replace("]", ""))

        # Hour selection
        selected_hour = st.slider(
            "Scheduled Pickup Hour",
            min_value=0,
            max_value=23,
            value=18,
            format="%02d:00"
        )
        st.caption(f"Formatted: {selected_hour:02d}:00 ({'Morning' if selected_hour < 12 else 'Afternoon/Evening'})")

        # Day of week selection
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_day_name = st.selectbox("Day of Week", day_names, index=2)  # Default Wednesday
        selected_day_idx = day_names.index(selected_day_name)

        run_pred = st.button("Run Real-time Congestion Prediction", use_container_width=True)

    with pred_col2:
        st.markdown("##### Inference Output & Analysis")

        try:
            # Perform prediction using actual saved model
            result = predictor.predict(
                hour=selected_hour,
                zone_id=selected_zone_id,
                day_of_week=selected_day_idx
            )

            pred_data = result["prediction"]
            level = pred_data["congestion_level"]
            confidence = pred_data["confidence"]
            probs = pred_data["class_probabilities"]
            hist_ctx = result["historical_context"]

            level_color = CONGESTION_COLORS.get(level, "#198754")

            # Main Result Card
            st.markdown(f"""
                <div style="background-color: #FFFFFF; border: 2px solid {level_color}; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
                    <div style="font-size: 13px; color: #6C757D; text-transform: uppercase; font-weight: 600;">Predicted Congestion State</div>
                    <div style="font-size: 32px; font-weight: 800; color: {level_color}; margin: 4px 0;">{level.upper()} CONGESTION</div>
                    <div style="font-size: 14px; color: #212529;"><strong>Confidence:</strong> {confidence * 100:.1f}%</div>
                    <div style="font-size: 13px; color: #495057; margin-top: 6px;">{pred_data['description']}</div>
                </div>
            """, unsafe_allow_html=True)

            # Class Probability Bar Chart
            df_prob = pd.DataFrame([
                {"Class": cls, "Probability": prob * 100, "Color": CONGESTION_COLORS.get(cls, "#198754")}
                for cls, prob in probs.items()
            ])

            fig_p = px.bar(
                df_prob,
                x="Probability",
                y="Class",
                orientation="h",
                text=df_prob["Probability"].apply(lambda p: f"{p:.1f}%"),
                color="Class",
                color_discrete_map=CONGESTION_COLORS,
                labels={"Probability": "Model Probability (%)", "Class": "Congestion Class"}
            )
            fig_p.update_layout(
                title=dict(text="Class Probability Distribution", font=dict(size=13, color=COLOR_TEXT_DARK)),
                xaxis=dict(range=[0, 100], gridcolor="#F1F3F5"),
                yaxis=dict(gridcolor="#F1F3F5"),
                showlegend=False,
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=20, r=20, t=35, b=20),
                height=180
            )
            st.plotly_chart(fig_p, use_container_width=True)

            # Historical baseline metrics
            st.markdown("###### Historical Baseline Context for Zone & Hour")
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                st.metric("Historical Speed", f"{hist_ctx['historical_avg_speed_mph']:.1f} mph")
            with bcol2:
                st.metric("Historical Duration", f"{hist_ctx['historical_avg_duration_minutes']:.1f} min")
            with bcol3:
                st.metric("Hourly Volume", f"{hist_ctx['historical_volume_hourly']:.0f} trips")

            if hist_ctx.get("fallback_defaults_applied"):
                st.caption("ℹ️ Unobserved zone/hour combination: global training baseline applied.")

        except Exception as e:
            st.error(f"Prediction execution failed: {e}")

    # Model Performance Summary Accordion
    if model_metrics:
        with st.expander("View Model Architecture & Validation Metrics (Test Set Evaluation)"):
            m_col1, m_col2, m_col3 = st.columns(3)
            rf_metrics = model_metrics.get("random_forest", {})
            baseline = model_metrics.get("baseline_dummy", {})

            with m_col1:
                st.metric("Random Forest Accuracy", f"{rf_metrics.get('accuracy', 0)*100:.2f}%", f"+{(rf_metrics.get('accuracy', 0) - baseline.get('accuracy', 0))*100:.1f}% vs baseline")
            with m_col2:
                st.metric("Macro Precision", f"{rf_metrics.get('precision_macro', 0):.4f}")
            with m_col3:
                st.metric("Macro F1-Score", f"{rf_metrics.get('f1_macro', 0):.4f}")

            # Feature Importance
            feat_imp = model_metrics.get("feature_importances", {})
            if feat_imp:
                df_imp = pd.DataFrame([
                    {"Feature": f, "Importance": val * 100}
                    for f, val in feat_imp.items()
                ]).sort_values(by="Importance", ascending=True)

                fig_imp = px.bar(
                    df_imp,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color_discrete_sequence=[COLOR_PRIMARY],
                    labels={"Importance": "Gini Importance (%)", "Feature": "Predictive Feature"},
                    title="Feature Importance Hierarchy (Random Forest)"
                )
                fig_imp.update_layout(
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    margin=dict(l=20, r=20, t=35, b=20),
                    height=240
                )
                st.plotly_chart(fig_imp, use_container_width=True)


# ==============================================================================
# Main Application Entry Point
# ==============================================================================

def main():
    # Page Setup
    st.set_page_config(
        page_title="TrafficPulse - Traffic Analytics Dashboard",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject Light Theme CSS
    apply_custom_styles()

    # Load Data Resources
    df_raw, dataset_source = load_trip_dataset()
    zone_lookup = load_zone_lookup()
    hourly_df, zone_summary_df, report_dict = load_precomputed_summaries()
    geo_data, centroids = load_geojson_and_centroids()
    predictor = load_prediction_model()
    model_metrics = load_model_metrics()

    # Brand Header Banner
    st.markdown("""
        <div class="brand-header">
            <h1>🚦 TrafficPulse — Big Data Urban Traffic Analytics</h1>
            <p>Course: CCS334 Big Data Analytics | NYC TLC Trip Record Dataset (January 2024)</p>
        </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Sidebar: Navigation & Interactive Filters
    # --------------------------------------------------------------------------
    st.sidebar.markdown("### 🧭 Dashboard Navigation")
    nav_option = st.sidebar.radio(
        "Select Panel",
        ["📊 Executive Overview & Analytics", "🗺️ Geographic Hotspots", "🤖 Congestion Prediction Panel"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Global Filters")

    # Available dates
    if df_raw is not None and not df_raw.empty:
        available_dates = sorted(df_raw["pickup_date"].unique())
        min_date = datetime.strptime(available_dates[0], "%Y-%m-%d").date()
        max_date = datetime.strptime(available_dates[-1], "%Y-%m-%d").date()
    else:
        min_date = date(2024, 1, 1)
        max_date = date(2024, 1, 31)

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Borough Filter
    all_boroughs = sorted(list({
        meta["borough"] for meta in zone_lookup.values()
        if meta["borough"] not in ["Unknown", "EWR"]
    }))
    selected_boroughs = st.sidebar.multiselect(
        "Borough Filter",
        ["All Boroughs"] + all_boroughs,
        default=["All Boroughs"]
    )

    # Dynamic Zone Selection based on Borough Filter
    if "All Boroughs" in selected_boroughs:
        eligible_zones = zone_lookup
    else:
        eligible_zones = {
            zid: meta for zid, meta in zone_lookup.items()
            if meta["borough"] in selected_boroughs
        }

    zone_choices = {
        meta["display_name"]: zid
        for zid, meta in sorted(eligible_zones.items(), key=lambda x: x[1]["zone_name"])
    }

    selected_zone_names = st.sidebar.multiselect(
        "Specific Pickup Zones (Optional)",
        list(zone_choices.keys()),
        default=[]
    )
    selected_zone_ids = [zone_choices[name] for name in selected_zone_names]

    # Hour Slider
    hour_range = st.sidebar.slider(
        "Hour Range (24h)",
        min_value=0,
        max_value=23,
        value=(0, 23),
        format="%02d:00"
    )

    # Day Type Radio
    day_type = st.sidebar.radio(
        "Day of Week Type",
        ["All Days", "Weekdays Only (Mon-Fri)", "Weekends Only (Sat-Sun)"]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Loaded Data Source:**<br>{dataset_source}", unsafe_allow_html=True)
    if predictor:
        st.sidebar.caption("✅ Prediction Model: **Online**")
    else:
        st.sidebar.caption("⚠️ Prediction Model: **Offline**")

    # --------------------------------------------------------------------------
    # Apply Filters to Trip Data
    # --------------------------------------------------------------------------
    if df_raw is not None and not df_raw.empty:
        df_filtered = filter_trip_data(
            df=df_raw,
            date_range=date_range if isinstance(date_range, tuple) and len(date_range) == 2 else None,
            selected_boroughs=selected_boroughs,
            selected_zones=selected_zone_ids,
            hour_range=hour_range,
            day_type=day_type,
            zone_lookup=zone_lookup
        )
    else:
        df_filtered = pd.DataFrame()

    # Compute KPIs
    kpis = compute_summary_kpis(df_filtered, zone_lookup, report_dict)

    # --------------------------------------------------------------------------
    # Render Selected Panel
    # --------------------------------------------------------------------------
    if "Executive Overview" in nav_option:
        render_overview_section(df_filtered, kpis, hourly_df, zone_lookup)
    elif "Geographic Hotspots" in nav_option:
        render_geographic_section(df_filtered, geo_data, centroids, zone_lookup, zone_summary_df)
    elif "Prediction Panel" in nav_option:
        render_prediction_section(predictor, zone_lookup, model_metrics)


if __name__ == "__main__":
    main()
