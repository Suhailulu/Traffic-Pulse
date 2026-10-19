#!/usr/bin/env python3
"""
TrafficPulse - Machine Learning Congestion Prediction Module
Phase 4: Trains baseline and Random Forest classifiers to predict urban traffic
congestion levels ('Low', 'Moderate', 'High') from historical NYC Yellow Taxi trip patterns.

Features:
- Chronological train/test split to prevent temporal leakage.
- Pre-trip feature engineering (hour, day of week, weekend indicator, zone).
- Historical aggregate metrics (volume, average duration, average speed) computed
  strictly from training data to avoid target leakage.
- Comparison against Dummy (most-frequent) and Logistic Regression baselines.
- Model serialization to src/prediction/models/.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DEFAULT_PROCESSED_PARQUET = Path("data/processed/cleaned_trips_2024_01.parquet")
DEFAULT_SAMPLE_PARQUET = Path("data/processed/cleaned_trips_sample.parquet")
DEFAULT_LOOKUP_CSV = Path("data/raw/taxi_zone_lookup.csv")
DEFAULT_MODEL_DIR = Path("src/prediction/models")
MODEL_FILENAME = "congestion_rf_model.joblib"
METRICS_FILENAME = "model_metrics.json"

TARGET_CLASSES = ["Low", "Moderate", "High"]
FEATURE_COLUMNS = [
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "PULocationID",
    "hist_volume",
    "hist_avg_duration",
    "hist_avg_speed"
]

def load_processed_dataset(
    data_path: Path = DEFAULT_PROCESSED_PARQUET,
    sample_rows_per_group: int = 10000,
    max_total_rows: int = None
) -> pd.DataFrame:
    """
    Loads trip records efficiently from the processed Parquet dataset.
    Reads across all row groups to ensure comprehensive temporal representation across January 2024.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        # Fall back to sample parquet if full parquet is absent
        if DEFAULT_SAMPLE_PARQUET.exists():
            data_path = DEFAULT_SAMPLE_PARQUET
        else:
            raise FileNotFoundError(f"Processed dataset not found at {data_path} or {DEFAULT_SAMPLE_PARQUET}")

    req_cols = [
        "tpep_pickup_datetime",
        "pickup_date",
        "pickup_hour",
        "pickup_day_of_week",
        "is_weekend",
        "PULocationID",
        "trip_duration_minutes",
        "speed_mph",
        "congestion_level"
    ]

    pf = pq.ParquetFile(data_path)
    num_rg = pf.num_row_groups

    dfs = []
    total_loaded = 0

    for i in range(num_rg):
        if sample_rows_per_group and sample_rows_per_group > 0:
            rg = pf.read_row_group(i, columns=req_cols)
            slice_size = min(sample_rows_per_group, len(rg))
            batch_df = rg.slice(0, slice_size).to_pandas()
        else:
            batch_df = pf.read_row_group(i, columns=req_cols).to_pandas()

        dfs.append(batch_df)
        total_loaded += len(batch_df)
        if max_total_rows and total_loaded >= max_total_rows:
            break

    df = pd.concat(dfs, ignore_index=True)

    # Filter out stray clock-drift records; restrict strictly to January 2024
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    valid_dates = (df["pickup_date"] >= "2024-01-01") & (df["pickup_date"] <= "2024-01-31")
    df = df[valid_dates].copy()

    # Drop any records with missing essential features or target
    df = df.dropna(subset=req_cols).copy()
    df = df[df["congestion_level"].isin(TARGET_CLASSES)].copy()

    # Sort chronologically to preserve real temporal order
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    return df

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.75
) -> tuple[pd.DataFrame, pd.DataFrame, datetime]:
    """
    Performs a strict chronological train/test split to prevent target leakage.
    Trips prior to the cutoff timestamp form the training partition; subsequent trips form the test set.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame")

    cutoff_idx = int(len(df) * train_ratio)
    cutoff_datetime = df.iloc[cutoff_idx]["tpep_pickup_datetime"]

    train_df = df[df["tpep_pickup_datetime"] < cutoff_datetime].copy().reset_index(drop=True)
    test_df = df[df["tpep_pickup_datetime"] >= cutoff_datetime].copy().reset_index(drop=True)

    return train_df, test_df, cutoff_datetime

def build_historical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Computes historical traffic profile features (volume, avg duration, avg speed)
    STRICTLY from the training partition to prevent future information leakage.
    Returns enriched train/test sets, the lookup dictionary, and global defaults.
    """
    # Group by Pickup Zone and Pickup Hour
    grouped = train_df.groupby(["PULocationID", "pickup_hour"]).agg(
        hist_volume=("speed_mph", "count"),
        hist_avg_duration=("trip_duration_minutes", "mean"),
        hist_avg_speed=("speed_mph", "mean")
    ).reset_index()

    # Compute global fallback defaults from overall training set
    global_defaults = {
        "hist_volume": float(train_df["PULocationID"].value_counts().median()),
        "hist_avg_duration": float(train_df["trip_duration_minutes"].mean()),
        "hist_avg_speed": float(train_df["speed_mph"].mean())
    }

    # Build fast in-memory dictionary for live inference: (zone_id, hour) -> metrics dict
    lookup_dict = {}
    for _, row in grouped.iterrows():
        key = f"{int(row['PULocationID'])}_{int(row['pickup_hour'])}"
        lookup_dict[key] = {
            "hist_volume": float(row["hist_volume"]),
            "hist_avg_duration": round(float(row["hist_avg_duration"]), 2),
            "hist_avg_speed": round(float(row["hist_avg_speed"]), 2)
        }

    # Merge features into training partition
    train_enriched = train_df.merge(grouped, on=["PULocationID", "pickup_hour"], how="left")
    train_enriched = train_enriched.fillna(global_defaults)

    test_enriched = None
    if test_df is not None and not test_df.empty:
        test_enriched = test_df.merge(grouped, on=["PULocationID", "pickup_hour"], how="left")
        test_enriched = test_enriched.fillna(global_defaults)

    return train_enriched, test_enriched, lookup_dict, global_defaults

def train_and_evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: Path = DEFAULT_MODEL_DIR
) -> dict:
    """
    Trains baseline models (Dummy majority class, Logistic Regression) and
    the primary Random Forest classifier. Evaluates performance on the test set.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["congestion_level"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["congestion_level"]

    # 1. Baseline Model 1: Majority Class Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)

    # 2. Baseline Model 2: Logistic Regression
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # 3. Primary Model: Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    t_start = time.time()
    rf.fit(X_train, y_train)
    fit_duration = time.time() - t_start
    rf_preds = rf.predict(X_test)

    # Evaluation Metrics Helper
    def compute_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
        return {
            "accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
            "precision_macro": float(round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4)),
            "precision_weighted": float(round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4)),
            "recall_macro": float(round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4)),
            "recall_weighted": float(round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4)),
            "f1_macro": float(round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)),
            "f1_weighted": float(round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)),
            "confusion_matrix": {
                "labels": TARGET_CLASSES,
                "matrix": cm.tolist()
            }
        }

    metrics_dummy = compute_metrics(y_test, dummy_preds)
    metrics_lr = compute_metrics(y_test, lr_preds)
    metrics_rf = compute_metrics(y_test, rf_preds)

    # Detailed per-class report for Random Forest
    rf_report = classification_report(y_test, rf_preds, labels=TARGET_CLASSES, output_dict=True, zero_division=0)

    # Feature Importance
    feature_importances = {
        feat: float(round(imp, 4))
        for feat, imp in sorted(zip(FEATURE_COLUMNS, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    }

    # Consolidated Metrics Dict
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "feature_columns": FEATURE_COLUMNS,
            "target_classes": TARGET_CLASSES,
            "training_duration_seconds": round(fit_duration, 2)
        },
        "target_distribution": {
            "train": train_df["congestion_level"].value_counts().to_dict(),
            "test": test_df["congestion_level"].value_counts().to_dict()
        },
        "baseline_dummy": metrics_dummy,
        "baseline_logistic_regression": metrics_lr,
        "random_forest": metrics_rf,
        "random_forest_classification_report": rf_report,
        "feature_importances": feature_importances
    }

    return results, rf, dummy

def run_model_training(
    data_path: Path = DEFAULT_PROCESSED_PARQUET,
    model_dir: Path = DEFAULT_MODEL_DIR,
    sample_rows_per_group: int = 10000,
    max_total_rows: int = None
) -> dict:
    """
    Orchestrates data loading, chronological splitting, feature engineering,
    model training, evaluation, and serialization.
    """
    print("=" * 80)
    print(" TrafficPulse - Traffic Congestion Model Training (Phase 4)")
    print(" Course: CCS334 Big Data Analytics | Urban Congestion Classification")
    print("=" * 80)

    # 1. Load Data
    print(f"\n[1] Loading processed trip records from: {data_path}...")
    df = load_processed_dataset(
        data_path=data_path,
        sample_rows_per_group=sample_rows_per_group,
        max_total_rows=max_total_rows
    )
    print(f"    Loaded {len(df):,} valid records covering {df['pickup_date'].min()} to {df['pickup_date'].max()}.")

    # 2. Chronological Split
    print("\n[2] Performing chronological train/test split (75% / 25%)...")
    train_df, test_df, cutoff_ts = chronological_split(df, train_ratio=0.75)
    print(f"    Split Cutoff Timestamp: {cutoff_ts}")
    print(f"    Training Partition:     {len(train_df):,} records ({train_df['pickup_date'].min()} to {train_df['pickup_date'].max()})")
    print(f"    Testing Partition:      {len(test_df):,} records ({test_df['pickup_date'].min()} to {test_df['pickup_date'].max()})")

    # 3. Feature Engineering (Strictly from training data)
    print("\n[3] Engineering historical profile features (preventing target leakage)...")
    train_enriched, test_enriched, lookup_dict, global_defaults = build_historical_features(train_df, test_df)
    print(f"    Generated {len(lookup_dict):,} (PULocationID x Hour) historical feature profiles.")

    # 4. Train and Evaluate Models
    print("\n[4] Training Baseline Classifiers and Random Forest Model...")
    metrics, rf_model, dummy_model = train_and_evaluate_models(train_enriched, test_enriched, model_dir=model_dir)

    print("\n[5] Model Evaluation Summary:")
    print(f"    - Dummy Baseline Accuracy:       {metrics['baseline_dummy']['accuracy'] * 100:.2f}%")
    print(f"    - Logistic Regression Accuracy: {metrics['baseline_logistic_regression']['accuracy'] * 100:.2f}%")
    print(f"    - Random Forest Accuracy:        {metrics['random_forest']['accuracy'] * 100:.2f}%")
    print(f"    - Random Forest Weighted F1:     {metrics['random_forest']['f1_weighted']:.4f}")
    print(f"    - Random Forest Macro F1:        {metrics['random_forest']['f1_macro']:.4f}")
    print("\n    Feature Importances:")
    for feat, imp in metrics["feature_importances"].items():
        print(f"      * {feat:<20}: {imp:.4f}")

    # 5. Serialize Model Bundle & Metrics
    print(f"\n[6] Serializing model bundle to: {model_dir}...")
    model_bundle = {
        "model": rf_model,
        "dummy_model": dummy_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_classes": TARGET_CLASSES,
        "historical_lookup": lookup_dict,
        "global_defaults": global_defaults,
        "cutoff_timestamp": str(cutoff_ts),
        "trained_at": datetime.now().isoformat()
    }

    bundle_path = model_dir / MODEL_FILENAME
    metrics_path = model_dir / METRICS_FILENAME

    joblib.dump(model_bundle, bundle_path, compress=3)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"    Successfully saved model:   {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KB)")
    print(f"    Successfully saved metrics: {metrics_path}")
    print("\n" + "=" * 80)
    print(" Model training and serialization complete.")
    print("=" * 80)

    return metrics

if __name__ == "__main__":
    run_model_training()
