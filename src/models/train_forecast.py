"""Forecast model factories and sequence preparation utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class SequenceData:
    x: np.ndarray
    y_scaled: np.ndarray
    y_original: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    effective_features: list[str]


def station_feature_names(station_categories: list[str]) -> list[str]:
    return [f"station__{station}" for station in station_categories]


def prepare_tree_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    station_categories: list[str],
    station_column: str = "station_id",
) -> tuple[np.ndarray, list[str]]:
    """Build a float32 matrix with deterministic station one-hot columns."""
    numeric = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    station_values = frame[station_column].astype(str).to_numpy()
    station_matrix = np.column_stack(
        [(station_values == station).astype(np.float32) for station in station_categories]
    )
    return (
        np.hstack([numeric, station_matrix]),
        feature_columns + station_feature_names(station_categories),
    )


def fit_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    random_state: int,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 80)),
        max_depth=params.get("max_depth", 20),
        min_samples_leaf=int(params.get("min_samples_leaf", 3)),
        max_features=params.get("max_features", 0.5),
        max_samples=params.get("max_samples", 0.7),
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    return model.fit(x_train, y_train)


def fit_xgboost_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    random_state: int,
):
    from xgboost import XGBRegressor

    models = []
    for index in range(y_train.shape[1]):
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(params.get("n_estimators", 350)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 8)),
            min_child_weight=float(params.get("min_child_weight", 3)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            tree_method="hist",
            eval_metric="mae",
            n_jobs=6,
            random_state=random_state + index,
        )
        models.append(model.fit(x_train, y_train[:, index]))
    return models


def fit_lightgbm_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    random_state: int,
):
    from lightgbm import LGBMRegressor

    models = []
    for index in range(y_train.shape[1]):
        model = LGBMRegressor(
            objective="regression",
            n_estimators=int(params.get("n_estimators", 350)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 63)),
            min_child_samples=int(params.get("min_child_samples", 30)),
            subsample=float(params.get("subsample", 0.8)),
            subsample_freq=1,
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            n_jobs=6,
            random_state=random_state + index,
            verbosity=-1,
        )
        models.append(model.fit(x_train, y_train[:, index]))
    return models


def predict_horizon_models(models, x: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict(x) for model in models])


def _with_station_columns(
    frame: pd.DataFrame,
    feature_columns: list[str],
    station_categories: list[str],
    station_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    work = frame.copy()
    station_lookup = {
        station: index for index, station in enumerate(station_categories)
    }
    denominator = max(len(station_categories) - 1, 1)
    station_codes = work[station_column].astype(str).map(station_lookup)
    if station_codes.isna().any():
        unknown = sorted(work.loc[station_codes.isna(), station_column].unique())
        raise ValueError(f"Unknown station categories: {unknown}")
    station_feature = "station__code"
    work[station_feature] = (station_codes / denominator).astype("float32")
    return work, feature_columns + [station_feature]


def build_lstm_sequences(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    station_categories: list[str],
    sequence_length: int,
    station_column: str = "station_id",
    timestamp_column: str = "timestamp",
    feature_scaler: StandardScaler | None = None,
    target_scaler: StandardScaler | None = None,
    fit_scalers: bool = False,
    stride: int = 1,
) -> SequenceData:
    """Create station-local sequences and break windows across time gaps."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    work, effective_features = _with_station_columns(
        frame, feature_columns, station_categories, station_column
    )
    work[timestamp_column] = pd.to_datetime(
        work[timestamp_column], errors="raise", utc=True
    )
    work = work.sort_values([station_column, timestamp_column]).reset_index(drop=True)

    if fit_scalers:
        feature_scaler = StandardScaler().fit(work[effective_features])
        target_scaler = StandardScaler().fit(
            work[target_columns].to_numpy(dtype=np.float32)
        )
    if feature_scaler is None or target_scaler is None:
        raise ValueError("Fitted feature and target scalers are required")

    x_parts: list[np.ndarray] = []
    y_scaled_parts: list[np.ndarray] = []
    y_original_parts: list[np.ndarray] = []
    for _, station in work.groupby(station_column, sort=False):
        station = station.sort_values(timestamp_column).copy()
        segment_ids = station[timestamp_column].diff().ne(pd.Timedelta(hours=1)).cumsum()
        for _, segment in station.groupby(segment_ids, sort=False):
            if len(segment) < sequence_length:
                continue
            x_values = feature_scaler.transform(segment[effective_features]).astype(
                np.float32
            )
            y_original = segment[target_columns].to_numpy(dtype=np.float32)
            y_scaled = target_scaler.transform(y_original).astype(np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(
                x_values, window_shape=sequence_length, axis=0
            ).transpose(0, 2, 1)
            x_parts.append(np.ascontiguousarray(windows[::stride]))
            y_original_parts.append(y_original[sequence_length - 1 :: stride])
            y_scaled_parts.append(y_scaled[sequence_length - 1 :: stride])

    if not x_parts:
        raise ValueError("No continuous segment is long enough for LSTM sequences")
    return SequenceData(
        x=np.concatenate(x_parts),
        y_scaled=np.concatenate(y_scaled_parts),
        y_original=np.concatenate(y_original_parts),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        effective_features=effective_features,
    )


def fit_lstm_model(
    train: SequenceData,
    validation: SequenceData,
    params: dict[str, Any],
    random_state: int,
):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    model = _create_lstm_model(train, params, random_state)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(params.get("patience", 2)),
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        train.x,
        train.y_scaled,
        validation_data=(validation.x, validation.y_scaled),
        epochs=int(params.get("epochs", 6)),
        batch_size=int(params.get("batch_size", 512)),
        callbacks=callbacks,
        verbose=2,
        shuffle=False,
    )
    return model, history


def _create_lstm_model(
    train: SequenceData,
    params: dict[str, Any],
    random_state: int,
):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(random_state)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=train.x.shape[1:]),
            tf.keras.layers.LSTM(
                int(params.get("units", 48)), return_sequences=False
            ),
            tf.keras.layers.Dropout(float(params.get("dropout", 0.2))),
            tf.keras.layers.Dense(
                int(params.get("dense_units", 32)), activation="relu"
            ),
            tf.keras.layers.Dense(train.y_scaled.shape[1]),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def fit_lstm_fixed_epochs(
    train: SequenceData,
    params: dict[str, Any],
    random_state: int,
    epochs: int,
):
    """Retrain LSTM on train+validation without looking at the test set."""
    model = _create_lstm_model(train, params, random_state)
    history = model.fit(
        train.x,
        train.y_scaled,
        epochs=max(int(epochs), 1),
        batch_size=int(params.get("batch_size", 512)),
        verbose=2,
        shuffle=False,
    )
    return model, history


def predict_lstm(model, sequences: SequenceData) -> np.ndarray:
    predicted_scaled = model.predict(sequences.x, verbose=0)
    return sequences.target_scaler.inverse_transform(predicted_scaled)
