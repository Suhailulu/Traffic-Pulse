"""
TrafficPulse - Prediction Package
Phase 4 Machine Learning Congestion Classification
"""

def get_predictor(*args, **kwargs):
    from src.prediction.predict_congestion import CongestionPredictor
    return CongestionPredictor(*args, **kwargs)

def predict_congestion(*args, **kwargs):
    from src.prediction.predict_congestion import predict_congestion as _predict
    return _predict(*args, **kwargs)

def run_model_training(*args, **kwargs):
    from src.prediction.train_model import run_model_training as _train
    return _train(*args, **kwargs)

__all__ = [
    "get_predictor",
    "predict_congestion",
    "run_model_training"
]
