"""
TrafficPulse - Prediction Unit Tests (Phase 4)
Tests model training, serialization, inference, input validation,
missing-feature fallback handling, and feature consistency.
"""

import pytest
import numpy as np
from pathlib import Path
import joblib

from src.prediction.train_model import (
    load_processed_dataset,
    chronological_split,
    build_historical_features,
    train_and_evaluate_models,
    FEATURE_COLUMNS,
    TARGET_CLASSES,
    DEFAULT_MODEL_DIR,
    MODEL_FILENAME,
    METRICS_FILENAME
)
from src.prediction.predict_congestion import (
    CongestionPredictor,
    predict_congestion
)

MODEL_BUNDLE_PATH = DEFAULT_MODEL_DIR / MODEL_FILENAME

# ==============================================================================
# 1. Model Serialization & Bundle Loading Tests
# ==============================================================================

def test_model_bundle_exists():
    """Verifies that trained model bundle and metrics JSON exist and are valid."""
    assert MODEL_BUNDLE_PATH.exists(), f"Model bundle {MODEL_BUNDLE_PATH} does not exist"
    metrics_path = DEFAULT_MODEL_DIR / METRICS_FILENAME
    assert metrics_path.exists(), f"Metrics file {metrics_path} does not exist"

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    assert "model" in bundle
    assert "dummy_model" in bundle
    assert "feature_columns" in bundle
    assert "historical_lookup" in bundle
    assert "global_defaults" in bundle
    assert bundle["feature_columns"] == FEATURE_COLUMNS

def test_predictor_initialization():
    """Verifies that CongestionPredictor initializes and loads internal structures."""
    predictor = CongestionPredictor(model_path=MODEL_BUNDLE_PATH)
    assert predictor.model is not None
    assert len(predictor.feature_columns) == len(FEATURE_COLUMNS)
    assert set(predictor.target_classes) == set(TARGET_CLASSES)
    assert len(predictor.historical_lookup) > 0
    assert "hist_volume" in predictor.global_defaults

# ==============================================================================
# 2. Prediction Output & Probability Structure Tests
# ==============================================================================

def test_valid_prediction_structure():
    """Verifies prediction response keys, valid class labels, and probability axioms."""
    result = predict_congestion(hour=18, zone_id=161, day_of_week=2, model_path=MODEL_BUNDLE_PATH)

    assert result["status"] == "SUCCESS"
    assert "prediction" in result
    assert "query_parameters" in result
    assert "historical_context" in result

    pred = result["prediction"]
    assert pred["congestion_level"] in TARGET_CLASSES
    assert 0.0 <= pred["confidence"] <= 1.0

    probs = pred["class_probabilities"]
    for cls in TARGET_CLASSES:
        assert cls in probs
        assert 0.0 <= probs[cls] <= 1.0

    # Probabilities must sum to 1.0 (accounting for float rounding)
    total_prob = sum(probs.values())
    assert pytest.approx(total_prob, abs=1e-2) == 1.0

def test_peak_vs_off_peak_divergence():
    """
    Verifies domain behavior: Midtown Center at 6 PM should exhibit significantly
    higher congestion probability than JFK Airport at 3 AM.
    """
    midtown_rush = predict_congestion(hour=18, zone_id=161, day_of_week=2, model_path=MODEL_BUNDLE_PATH)
    jfk_night = predict_congestion(hour=3, zone_id=132, day_of_week=2, model_path=MODEL_BUNDLE_PATH)

    rush_high_prob = midtown_rush["prediction"]["class_probabilities"]["High"]
    night_high_prob = jfk_night["prediction"]["class_probabilities"]["High"]

    assert rush_high_prob > night_high_prob
    assert jfk_night["prediction"]["congestion_level"] == "Low"

# ==============================================================================
# 3. Input Validation & Error Handling Tests
# ==============================================================================

@pytest.mark.parametrize("invalid_hour", [-1, 24, 100])
def test_invalid_hour_handling(invalid_hour):
    """Verifies ValueError on out-of-range hours."""
    with pytest.raises(ValueError, match="Invalid hour"):
        predict_congestion(hour=invalid_hour, zone_id=161, model_path=MODEL_BUNDLE_PATH)

@pytest.mark.parametrize("invalid_zone", [0, -5, 266, 999])
def test_invalid_zone_handling(invalid_zone):
    """Verifies ValueError on out-of-range taxi zones."""
    with pytest.raises(ValueError, match="Invalid zone ID"):
        predict_congestion(hour=12, zone_id=invalid_zone, model_path=MODEL_BUNDLE_PATH)

@pytest.mark.parametrize("invalid_day", [-1, 7, 10])
def test_invalid_day_handling(invalid_day):
    """Verifies ValueError on out-of-range day of week."""
    with pytest.raises(ValueError, match="Invalid day_of_week"):
        predict_congestion(hour=12, zone_id=161, day_of_week=invalid_day, model_path=MODEL_BUNDLE_PATH)

def test_type_error_on_non_integer_inputs():
    """Verifies TypeError when passing invalid types."""
    with pytest.raises(TypeError):
        predict_congestion(hour="noon", zone_id=161, model_path=MODEL_BUNDLE_PATH)
    with pytest.raises(TypeError):
        predict_congestion(hour=12, zone_id="Midtown", model_path=MODEL_BUNDLE_PATH)

# ==============================================================================
# 4. Missing Features & Graceful Fallback Handling Tests
# ==============================================================================

def test_missing_feature_graceful_fallback():
    """
    Tests an unobserved or rare zone/hour combination.
    Verifies that global training medians are safely used without crashing.
    """
    predictor = CongestionPredictor(model_path=MODEL_BUNDLE_PATH)
    # Zone 200 at 4 AM might have 0 trips in the historical lookup
    result = predictor.predict(hour=4, zone_id=200, day_of_week=0)

    assert result["status"] == "SUCCESS"
    assert result["prediction"]["congestion_level"] in TARGET_CLASSES
    # Historical context values must be non-null and positive
    assert result["historical_context"]["historical_avg_speed_mph"] > 0
    assert result["historical_context"]["historical_avg_duration_minutes"] > 0

# ==============================================================================
# 5. Feature Consistency & Determinism Tests
# ==============================================================================

def test_feature_consistency_between_training_and_inference():
    """
    Verifies that columns constructed during inference match exactly the feature
    column names and ordering expected by the trained model.
    """
    predictor = CongestionPredictor(model_path=MODEL_BUNDLE_PATH)
    assert list(predictor.model.feature_names_in_) == FEATURE_COLUMNS

def test_deterministic_predictions():
    """Verifies identical input parameters yield identical class probabilities."""
    res1 = predict_congestion(hour=14, zone_id=237, day_of_week=1, model_path=MODEL_BUNDLE_PATH)
    res2 = predict_congestion(hour=14, zone_id=237, day_of_week=1, model_path=MODEL_BUNDLE_PATH)

    assert res1["prediction"]["congestion_level"] == res2["prediction"]["congestion_level"]
    assert res1["prediction"]["confidence"] == res2["prediction"]["confidence"]
    assert res1["prediction"]["class_probabilities"] == res2["prediction"]["class_probabilities"]

# ==============================================================================
# 6. Chronological Split Logic Test
# ==============================================================================

def test_chronological_split_strict_ordering():
    """Verifies that chronological split partitions cleanly without temporal overlap."""
    import pandas as pd
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "tpep_pickup_datetime": dates,
        "speed_mph": np.random.uniform(5, 25, 100),
        "congestion_level": ["Moderate"] * 100
    })
    train_df, test_df, cutoff = chronological_split(df, train_ratio=0.8)

    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_df["tpep_pickup_datetime"].max() < test_df["tpep_pickup_datetime"].min()
    assert cutoff == test_df["tpep_pickup_datetime"].min()
