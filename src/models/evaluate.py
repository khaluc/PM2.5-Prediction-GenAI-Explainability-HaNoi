"""Regression metrics for multi-horizon PM2.5 forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_mape(y_true, y_pred, epsilon: float = 1.0) -> float:
    """MAPE percentage with a denominator floor for near-zero PM2.5 values."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(actual), epsilon)
    return float(np.mean(np.abs(actual - predicted) / denominator) * 100)


def evaluate_forecast(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE, R² and safe MAPE for one horizon."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "MAPE": safe_mape(actual, predicted),
    }


def evaluate_multi_horizon(
    y_true,
    y_pred,
    *,
    model_name: str,
    horizons: list[int],
    split_name: str,
) -> pd.DataFrame:
    """Return one metric row per model and forecast horizon."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if actual.ndim == 1:
        actual = actual.reshape(-1, 1)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Prediction shape {predicted.shape} does not match target {actual.shape}"
        )
    if actual.shape[1] != len(horizons):
        raise ValueError("Number of target columns must equal number of horizons")

    rows = []
    for index, horizon in enumerate(horizons):
        metrics = evaluate_forecast(actual[:, index], predicted[:, index])
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "horizon_hours": int(horizon),
                "samples": len(actual),
                **metrics,
            }
        )
    return pd.DataFrame(rows)
