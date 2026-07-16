"""Train, compare and select PM2.5 forecasting models."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import evaluate_multi_horizon  # noqa: E402
from src.models.train_forecast import (  # noqa: E402
    build_lstm_sequences,
    fit_lightgbm_models,
    fit_lstm_fixed_epochs,
    fit_lstm_model,
    fit_random_forest,
    fit_xgboost_models,
    predict_horizon_models,
    predict_lstm,
    prepare_tree_matrix,
)

LOGGER = logging.getLogger("forecast_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def _load_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    LOGGER.info("Reading %s", path)
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    if frame.empty:
        raise ValueError(f"Empty model split: {path}")
    return frame


def evaluation_rows(
    frame: pd.DataFrame,
    *,
    sequence_length: int,
    station_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """Return rows for which a complete station-local LSTM window exists."""
    work = frame.copy()
    work[timestamp_column] = pd.to_datetime(
        work[timestamp_column], errors="raise", utc=True
    )
    work = work.sort_values([station_column, timestamp_column]).reset_index(drop=True)
    gap = work.groupby(station_column, sort=False)[timestamp_column].diff().ne(
        pd.Timedelta(hours=1)
    )
    work["_segment"] = gap.groupby(work[station_column], sort=False).cumsum()
    position = work.groupby([station_column, "_segment"], sort=False).cumcount()
    return work.loc[position >= sequence_length - 1].drop(columns="_segment")


def _history_dict(history) -> dict[str, list[float]]:
    return {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "tensorflow",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _save_metrics(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    LOGGER.info("Metrics: %s", output)


def _model_payload(
    *,
    model_name: str,
    horizons: list[int],
    feature_columns: list[str],
    effective_feature_columns: list[str],
    station_categories: list[str],
    station_column: str,
    timestamp_column: str,
    model: Any = None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model": model,
        "horizons": horizons,
        "target_columns": [f"target_pm25_t_plus_{hour}h" for hour in horizons],
        "feature_columns": feature_columns,
        "effective_feature_columns": effective_feature_columns,
        "station_categories": station_categories,
        "station_column": station_column,
        "timestamp_column": timestamp_column,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started_at = time.perf_counter()

    with Path(args.config).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    forecast = config["forecast"]
    metadata = json.loads(Path(forecast["feature_metadata"]).read_text(encoding="utf-8"))

    horizons = [int(value) for value in forecast.get("horizons", [1, 3, 6])]
    target_columns = list(metadata["target_columns"])
    expected_targets = [f"target_pm25_t_plus_{hour}h" for hour in horizons]
    if target_columns != expected_targets:
        raise ValueError(
            f"Target metadata {target_columns} does not match horizons {horizons}"
        )
    feature_columns = list(metadata["feature_columns"])
    station_column = str(forecast.get("station_column", "station_id"))
    timestamp_column = str(forecast.get("timestamp_column", "timestamp"))
    required_columns = list(
        dict.fromkeys(
            [timestamp_column, station_column, *feature_columns, *target_columns]
        )
    )
    split_dir = Path(forecast["split_dir"])
    train = _load_frame(split_dir / "train.csv.gz", required_columns)
    validation = _load_frame(split_dir / "validation.csv.gz", required_columns)
    test = _load_frame(split_dir / "test.csv.gz", required_columns)
    station_categories = sorted(train[station_column].astype(str).unique().tolist())
    if set(validation[station_column].astype(str).unique()) - set(station_categories):
        raise ValueError("Validation contains stations not present in train")
    if set(test[station_column].astype(str).unique()) - set(station_categories):
        raise ValueError("Test contains stations not present in train")

    lstm_config = dict(forecast["lstm"])
    lstm_features = list(lstm_config["features"])
    missing_lstm = sorted(set(lstm_features) - set(feature_columns))
    if missing_lstm:
        raise ValueError(f"Unknown LSTM features: {missing_lstm}")
    sequence_length = int(lstm_config.get("sequence_length", 24))
    training_stride = int(lstm_config.get("training_stride", 2))
    random_state = int(forecast.get("random_state", 42))

    common_validation = evaluation_rows(
        validation,
        sequence_length=sequence_length,
        station_column=station_column,
        timestamp_column=timestamp_column,
    )
    common_test = evaluation_rows(
        test,
        sequence_length=sequence_length,
        station_column=station_column,
        timestamp_column=timestamp_column,
    )
    LOGGER.info(
        "Rows - train: %s, validation: %s (%s comparable), test: %s (%s comparable)",
        f"{len(train):,}",
        f"{len(validation):,}",
        f"{len(common_validation):,}",
        f"{len(test):,}",
        f"{len(common_test):,}",
    )

    y_train = train[target_columns].to_numpy(dtype=np.float32)
    y_validation = common_validation[target_columns].to_numpy(dtype=np.float32)
    validation_metrics: list[pd.DataFrame] = []
    durations: dict[str, float] = {}

    baseline_started = time.perf_counter()
    baseline_validation = np.repeat(
        common_validation[["pm25"]].to_numpy(dtype=np.float32),
        len(horizons),
        axis=1,
    )
    validation_metrics.append(
        evaluate_multi_horizon(
            y_validation,
            baseline_validation,
            model_name="Baseline",
            horizons=horizons,
            split_name="validation",
        )
    )
    durations["Baseline"] = time.perf_counter() - baseline_started

    LOGGER.info("Preparing tree-model matrices")
    x_train, effective_tree_features = prepare_tree_matrix(
        train, feature_columns, station_categories, station_column
    )
    x_validation, _ = prepare_tree_matrix(
        common_validation, feature_columns, station_categories, station_column
    )

    LOGGER.info("Training Random Forest")
    model_started = time.perf_counter()
    random_forest = fit_random_forest(
        x_train, y_train, forecast["random_forest"], random_state
    )
    random_forest_prediction = random_forest.predict(x_validation)
    durations["RandomForest"] = time.perf_counter() - model_started
    validation_metrics.append(
        evaluate_multi_horizon(
            y_validation,
            random_forest_prediction,
            model_name="RandomForest",
            horizons=horizons,
            split_name="validation",
        )
    )
    del random_forest, random_forest_prediction
    gc.collect()

    LOGGER.info("Training XGBoost")
    model_started = time.perf_counter()
    xgboost_models = fit_xgboost_models(
        x_train, y_train, forecast["xgboost"], random_state
    )
    xgboost_prediction = predict_horizon_models(xgboost_models, x_validation)
    durations["XGBoost"] = time.perf_counter() - model_started
    validation_metrics.append(
        evaluate_multi_horizon(
            y_validation,
            xgboost_prediction,
            model_name="XGBoost",
            horizons=horizons,
            split_name="validation",
        )
    )
    del xgboost_models, xgboost_prediction
    gc.collect()

    LOGGER.info("Training LightGBM")
    model_started = time.perf_counter()
    lightgbm_models = fit_lightgbm_models(
        x_train, y_train, forecast["lightgbm"], random_state
    )
    lightgbm_prediction = predict_horizon_models(lightgbm_models, x_validation)
    durations["LightGBM"] = time.perf_counter() - model_started
    validation_metrics.append(
        evaluate_multi_horizon(
            y_validation,
            lightgbm_prediction,
            model_name="LightGBM",
            horizons=horizons,
            split_name="validation",
        )
    )
    del lightgbm_models, lightgbm_prediction, x_train, x_validation
    gc.collect()

    LOGGER.info("Preparing LSTM sequences")
    lstm_train = build_lstm_sequences(
        train,
        feature_columns=lstm_features,
        target_columns=target_columns,
        station_categories=station_categories,
        sequence_length=sequence_length,
        station_column=station_column,
        timestamp_column=timestamp_column,
        fit_scalers=True,
        stride=training_stride,
    )
    lstm_validation = build_lstm_sequences(
        validation,
        feature_columns=lstm_features,
        target_columns=target_columns,
        station_categories=station_categories,
        sequence_length=sequence_length,
        station_column=station_column,
        timestamp_column=timestamp_column,
        feature_scaler=lstm_train.feature_scaler,
        target_scaler=lstm_train.target_scaler,
        stride=1,
    )
    if not np.allclose(lstm_validation.y_original, y_validation, equal_nan=True):
        raise ValueError("LSTM validation rows are not aligned with common validation")

    LOGGER.info(
        "Training LSTM with %s train and %s validation sequences",
        f"{len(lstm_train.x):,}",
        f"{len(lstm_validation.x):,}",
    )
    model_started = time.perf_counter()
    lstm_model, lstm_history = fit_lstm_model(
        lstm_train, lstm_validation, lstm_config, random_state
    )
    lstm_prediction = predict_lstm(lstm_model, lstm_validation)
    durations["LSTM"] = time.perf_counter() - model_started
    validation_metrics.append(
        evaluate_multi_horizon(
            y_validation,
            lstm_prediction,
            model_name="LSTM",
            horizons=horizons,
            split_name="validation",
        )
    )
    initial_lstm_history = _history_dict(lstm_history)
    best_lstm_epoch = int(np.argmin(initial_lstm_history["val_loss"]) + 1)
    lstm_train_sequences = len(lstm_train.x)
    lstm_validation_sequences = len(lstm_validation.x)
    del lstm_model, lstm_history, lstm_prediction, lstm_train, lstm_validation
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except ImportError:
        pass
    gc.collect()

    validation_table = pd.concat(validation_metrics, ignore_index=True)
    selection_metric = str(forecast.get("selection_metric", "MAE")).upper()
    ranking = (
        validation_table.groupby("model", as_index=False)[
            ["MAE", "RMSE", "R2", "MAPE"]
        ]
        .mean()
        .rename(columns={column: f"mean_{column}" for column in ["MAE", "RMSE", "R2", "MAPE"]})
    )
    selection_column = f"mean_{selection_metric}"
    if selection_column not in ranking.columns:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    ascending = selection_metric != "R2"
    ranking = ranking.sort_values(selection_column, ascending=ascending).reset_index(drop=True)
    selected_model = str(ranking.iloc[0]["model"])
    LOGGER.info("Selected model by mean %s: %s", selection_metric, selected_model)

    validation_output = Path(forecast["validation_metrics_output"])
    _save_metrics(validation_table, validation_output)

    LOGGER.info("Retraining %s on train + validation", selected_model)
    final_fit_started = time.perf_counter()
    combined_train = pd.concat([train, validation], ignore_index=True)
    artifact_dir = Path(forecast["artifacts_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib_output = artifact_dir / "pm25_forecast.joblib"
    keras_output: Path | None = None

    if selected_model == "Baseline":
        final_prediction = np.repeat(
            common_test[["pm25"]].to_numpy(dtype=np.float32),
            len(horizons),
            axis=1,
        )
        payload = _model_payload(
            model_name=selected_model,
            horizons=horizons,
            feature_columns=["pm25"],
            effective_feature_columns=["pm25"],
            station_categories=station_categories,
            station_column=station_column,
            timestamp_column=timestamp_column,
        )
        payload["strategy"] = "persistence"
        joblib.dump(payload, joblib_output)
    elif selected_model in {"RandomForest", "XGBoost", "LightGBM"}:
        x_combined, effective_tree_features = prepare_tree_matrix(
            combined_train, feature_columns, station_categories, station_column
        )
        x_test, _ = prepare_tree_matrix(
            common_test, feature_columns, station_categories, station_column
        )
        y_combined = combined_train[target_columns].to_numpy(dtype=np.float32)
        if selected_model == "RandomForest":
            final_model = fit_random_forest(
                x_combined, y_combined, forecast["random_forest"], random_state
            )
            final_prediction = final_model.predict(x_test)
        elif selected_model == "XGBoost":
            final_model = fit_xgboost_models(
                x_combined, y_combined, forecast["xgboost"], random_state
            )
            final_prediction = predict_horizon_models(final_model, x_test)
        else:
            final_model = fit_lightgbm_models(
                x_combined, y_combined, forecast["lightgbm"], random_state
            )
            final_prediction = predict_horizon_models(final_model, x_test)
        payload = _model_payload(
            model_name=selected_model,
            model=final_model,
            horizons=horizons,
            feature_columns=feature_columns,
            effective_feature_columns=effective_tree_features,
            station_categories=station_categories,
            station_column=station_column,
            timestamp_column=timestamp_column,
        )
        joblib.dump(payload, joblib_output, compress=3)
        del x_combined, x_test, y_combined, final_model
    else:
        final_lstm_train = build_lstm_sequences(
            combined_train,
            feature_columns=lstm_features,
            target_columns=target_columns,
            station_categories=station_categories,
            sequence_length=sequence_length,
            station_column=station_column,
            timestamp_column=timestamp_column,
            fit_scalers=True,
            stride=training_stride,
        )
        final_lstm_test = build_lstm_sequences(
            test,
            feature_columns=lstm_features,
            target_columns=target_columns,
            station_categories=station_categories,
            sequence_length=sequence_length,
            station_column=station_column,
            timestamp_column=timestamp_column,
            feature_scaler=final_lstm_train.feature_scaler,
            target_scaler=final_lstm_train.target_scaler,
            stride=1,
        )
        if not np.allclose(final_lstm_test.y_original, common_test[target_columns]):
            raise ValueError("LSTM test rows are not aligned with common test")
        final_model, _ = fit_lstm_fixed_epochs(
            final_lstm_train,
            lstm_config,
            random_state,
            best_lstm_epoch,
        )
        final_prediction = predict_lstm(final_model, final_lstm_test)
        keras_output = artifact_dir / "pm25_lstm.keras"
        final_model.save(keras_output)
        payload = _model_payload(
            model_name=selected_model,
            horizons=horizons,
            feature_columns=lstm_features,
            effective_feature_columns=final_lstm_train.effective_features,
            station_categories=station_categories,
            station_column=station_column,
            timestamp_column=timestamp_column,
        )
        payload.update(
            {
                "keras_model_path": str(keras_output),
                "feature_scaler": final_lstm_train.feature_scaler,
                "target_scaler": final_lstm_train.target_scaler,
                "sequence_length": sequence_length,
            }
        )
        joblib.dump(payload, joblib_output, compress=3)

    final_fit_seconds = time.perf_counter() - final_fit_started
    y_test = common_test[target_columns].to_numpy(dtype=np.float32)
    test_metrics = evaluate_multi_horizon(
        y_test,
        final_prediction,
        model_name=selected_model,
        horizons=horizons,
        split_name="test",
    )
    test_output = Path(forecast["test_metrics_output"])
    _save_metrics(test_metrics, test_output)

    predictions_output = Path(
        forecast.get(
            "test_predictions_output", "artifacts/forecast_test_predictions.csv.gz"
        )
    )
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions = common_test[[timestamp_column, station_column]].copy()
    for index, horizon in enumerate(horizons):
        predictions[f"actual_pm25_t_plus_{horizon}h"] = y_test[:, index]
        predictions[f"predicted_pm25_t_plus_{horizon}h"] = final_prediction[:, index]
    predictions.to_csv(predictions_output, index=False, compression="gzip")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "selection": f"lowest mean validation {selection_metric} across horizons",
            "test_usage": "test evaluated once after model selection",
            "baseline": "persistence: future PM2.5 equals current PM2.5",
            "lstm_sequence_length_hours": sequence_length,
            "lstm_training_stride": training_stride,
            "mape_denominator_floor": 1.0,
        },
        "rows": {
            "train": len(train),
            "validation": len(validation),
            "validation_comparable": len(common_validation),
            "test": len(test),
            "test_comparable": len(common_test),
            "lstm_train_sequences": lstm_train_sequences,
            "lstm_validation_sequences": lstm_validation_sequences,
        },
        "horizons": horizons,
        "features": {
            "tree_feature_count": len(effective_tree_features),
            "lstm_features": lstm_features + ["station__code"],
        },
        "validation_ranking": ranking.to_dict(orient="records"),
        "selected_model": selected_model,
        "test_metrics": test_metrics.to_dict(orient="records"),
        "duration_seconds": {
            "validation_models": durations,
            "final_fit": final_fit_seconds,
            "total": time.perf_counter() - started_at,
        },
        "lstm": {
            "best_epoch": best_lstm_epoch,
            "initial_history": initial_lstm_history,
        },
        "artifacts": {
            "model_metadata": str(joblib_output),
            "keras_model": str(keras_output) if keras_output else None,
            "validation_metrics": str(validation_output),
            "test_metrics": str(test_output),
            "test_predictions": str(predictions_output),
        },
        "package_versions": _package_versions(),
    }
    report_output = Path(forecast["training_report_output"])
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Model artifact: %s", joblib_output)
    LOGGER.info("Training report: %s", report_output)
    LOGGER.info("Test metrics:\n%s", test_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
