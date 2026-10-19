#!/usr/bin/env python3
"""
TrafficPulse - Real-time Traffic Congestion Inference Engine
Phase 4: Predicts urban traffic congestion level ('Low', 'Moderate', 'High')
for a user-specified hour, geographic zone, and day of week.

Handles:
- Input validation (valid hour 0-23, valid zone 1-265, valid day 0-6).
- Graceful handling of missing or unseen historical feature combinations.
- Real-time probability estimation across all congestion levels.
- Contextual enrichment with zone names, boroughs, and historical baseline metrics.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path("src/prediction/models/congestion_rf_model.joblib")
DEFAULT_LOOKUP_CSV = Path("data/raw/taxi_zone_lookup.csv")

class CongestionPredictor:
    """
    Inference wrapper for the TrafficPulse Random Forest congestion classification model.
    """
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        zone_lookup_path: Path = DEFAULT_LOOKUP_CSV
    ):
        self.model_path = Path(model_path)
        self.zone_lookup_path = Path(zone_lookup_path)
        self.model_bundle = None
        self.model = None
        self.feature_columns = []
        self.target_classes = []
        self.historical_lookup = {}
        self.global_defaults = {}
        self.zone_metadata = {}

        self._load_model()
        self._load_zone_metadata()

    def _load_model(self):
        """Loads serialized model bundle containing weights, lookups, and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model bundle not found at: {self.model_path}. "
                "Please run `python -m src.prediction.train_model` first."
            )
        self.model_bundle = joblib.load(self.model_path)
        self.model = self.model_bundle["model"]
        self.feature_columns = self.model_bundle["feature_columns"]
        self.target_classes = self.model_bundle["target_classes"]
        self.historical_lookup = self.model_bundle["historical_lookup"]
        self.global_defaults = self.model_bundle["global_defaults"]

    def _load_zone_metadata(self):
        """Loads human-readable zone names and borough associations."""
        if not self.zone_lookup_path.exists():
            return
        df = pd.read_csv(self.zone_lookup_path)
        for _, row in df.iterrows():
            try:
                loc_id = int(row["LocationID"])
                self.zone_metadata[loc_id] = {
                    "zone_name": str(row["Zone"]) if pd.notna(row["Zone"]) else "Unknown Zone",
                    "borough": str(row["Borough"]) if pd.notna(row["Borough"]) else "Unknown Borough"
                }
            except (ValueError, TypeError):
                continue

    def validate_inputs(
        self,
        hour: int,
        zone_id: int,
        day_of_week: int
    ) -> None:
        """Validates input types and ranges, raising informative ValueErrors on invalid values."""
        if not isinstance(hour, (int, np.integer)):
            raise TypeError(f"Hour must be an integer, got {type(hour).__name__}")
        if not (0 <= hour <= 23):
            raise ValueError(f"Invalid hour: {hour}. Must be an integer between 0 and 23.")

        if not isinstance(zone_id, (int, np.integer)):
            raise TypeError(f"Zone ID must be an integer, got {type(zone_id).__name__}")
        if not (1 <= zone_id <= 265):
            raise ValueError(f"Invalid zone ID: {zone_id}. NYC taxi zones must be between 1 and 265.")

        if not isinstance(day_of_week, (int, np.integer)):
            raise TypeError(f"Day of week must be an integer, got {type(day_of_week).__name__}")
        if not (0 <= day_of_week <= 6):
            raise ValueError(f"Invalid day_of_week: {day_of_week}. Must be 0 (Monday) through 6 (Sunday).")

    def predict(
        self,
        hour: int,
        zone_id: int,
        day_of_week: int = 1,
        is_weekend: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes real-time inference for a given zone, hour, and day of week.
        """
        # 1. Input Validation
        self.validate_inputs(hour, zone_id, day_of_week)

        if is_weekend is None:
            is_weekend = 1 if day_of_week >= 5 else 0

        # 2. Retrieve Historical Features with Graceful Fallback
        lookup_key = f"{int(zone_id)}_{int(hour)}"
        fallback_applied = False
        if lookup_key in self.historical_lookup:
            hist_data = self.historical_lookup[lookup_key]
        else:
            hist_data = self.global_defaults
            fallback_applied = True

        hist_volume = float(hist_data.get("hist_volume", self.global_defaults["hist_volume"]))
        hist_avg_duration = float(hist_data.get("hist_avg_duration", self.global_defaults["hist_avg_duration"]))
        hist_avg_speed = float(hist_data.get("hist_avg_speed", self.global_defaults["hist_avg_speed"]))

        # 3. Assemble Feature Vector
        feature_dict = {
            "pickup_hour": hour,
            "pickup_day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "PULocationID": zone_id,
            "hist_volume": hist_volume,
            "hist_avg_duration": hist_avg_duration,
            "hist_avg_speed": hist_avg_speed
        }
        X = pd.DataFrame([feature_dict])[self.feature_columns]

        # 4. Model Prediction
        pred_label = self.model.predict(X)[0]
        pred_probs = self.model.predict_proba(X)[0]
        prob_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.model.classes_, pred_probs)
        }

        # 5. Enrich Contextual Response
        zone_info = self.zone_metadata.get(zone_id, {
            "zone_name": f"Zone {zone_id}",
            "borough": "Unknown"
        })

        # Human-readable interpretation of congestion level
        descriptions = {
            "Low": "Fluid traffic flow (estimated speeds >= 15 mph). Minimal intersection delay.",
            "Moderate": "Steady urban traffic flow (estimated speeds 8 - 15 mph). Normal signal delays.",
            "High": "Severe corridor friction / stop-and-go congestion (estimated speeds < 8 mph)."
        }

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        return {
            "status": "SUCCESS",
            "prediction": {
                "congestion_level": pred_label,
                "confidence": round(float(np.max(pred_probs)), 4),
                "description": descriptions.get(pred_label, "Urban traffic state"),
                "class_probabilities": prob_dict
            },
            "query_parameters": {
                "pickup_hour": hour,
                "pickup_time_formatted": f"{hour:02d}:00",
                "day_of_week": day_of_week,
                "day_name": day_names[day_of_week],
                "is_weekend": bool(is_weekend),
                "zone_id": zone_id,
                "zone_name": zone_info["zone_name"],
                "borough": zone_info["borough"]
            },
            "historical_context": {
                "historical_volume_hourly": hist_volume,
                "historical_avg_duration_minutes": hist_avg_duration,
                "historical_avg_speed_mph": hist_avg_speed,
                "fallback_defaults_applied": fallback_applied
            },
            "model_metadata": {
                "model_type": "RandomForestClassifier",
                "trained_at": self.model_bundle.get("trained_at", "Unknown"),
                "cutoff_timestamp": self.model_bundle.get("cutoff_timestamp", "Unknown")
            }
        }

def predict_congestion(
    hour: int,
    zone_id: int,
    day_of_week: int = 1,
    is_weekend: Optional[int] = None,
    model_path: Path = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
    """
    Convenience function to instantiate predictor and run inference in a single call.
    """
    predictor = CongestionPredictor(model_path=model_path)
    return predictor.predict(hour=hour, zone_id=zone_id, day_of_week=day_of_week, is_weekend=is_weekend)

def main():
    parser = argparse.ArgumentParser(description="TrafficPulse Congestion Predictor (Phase 4)")
    parser.add_argument("--hour", type=int, default=18, help="Pickup hour (0-23)")
    parser.add_argument("--zone", type=int, default=161, help="NYC Taxi Zone ID (1-265, e.g. 161 for Midtown Center)")
    parser.add_argument("--day", type=int, default=2, help="Day of week (0=Mon, 1=Tue, ..., 6=Sun)")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to trained model bundle")

    args = parser.parse_args()

    try:
        result = predict_congestion(
            hour=args.hour,
            zone_id=args.zone,
            day_of_week=args.day,
            model_path=Path(args.model_path)
        )
        print("=" * 70)
        print(" TrafficPulse - Traffic Congestion Inference")
        print("=" * 70)
        print(f" Target Zone:       {result['query_parameters']['zone_name']} (ID {result['query_parameters']['zone_id']}, {result['query_parameters']['borough']})")
        print(f" Target Time:       {result['query_parameters']['day_name']} at {result['query_parameters']['pickup_time_formatted']}")
        print(f" Predicted State:   [{result['prediction']['congestion_level'].upper()}] (Confidence: {result['prediction']['confidence'] * 100:.1f}%)")
        print(f" Description:       {result['prediction']['description']}")
        print("\n Class Probabilities:")
        for cls, prob in result["prediction"]["class_probabilities"].items():
            bar = "#" * int(prob * 30)
            print(f"   - {cls:<10}: {prob * 100:>5.1f}% | {bar}")
        print("\n Historical Baseline for Zone/Hour:")
        print(f"   - Average Speed:    {result['historical_context']['historical_avg_speed_mph']:.2f} mph")
        print(f"   - Average Duration: {result['historical_context']['historical_avg_duration_minutes']:.2f} min")
        print(f"   - Typical Volume:   {result['historical_context']['historical_volume_hourly']:.0f} trips")
        print("=" * 70)
    except Exception as e:
        print(f"Inference error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
