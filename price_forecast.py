"""Pronosticos multi-horizonte con intervalos cuantiles conformalizados.

El intervalo representa el precio de cierre al final de cada horizonte, no el
maximo/minimo intraperiodo. La cobertura y direccion se miden en un bloque de
test cronologico que no participa en el entrenamiento ni en la calibracion.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import config

logger = logging.getLogger(__name__)

FORECAST_HORIZONS_MINUTES = {
    "15 min": 15,
    "30 min": 30,
    "1 hora": 60,
    "1 día": 1440,
}
QUANTILES = np.array([0.10, 0.50, 0.90])


def timeframe_to_minutes(timeframe: str) -> int:
    value = int(timeframe[:-1])
    unit = timeframe[-1].lower()
    multipliers = {"m": 1, "h": 60, "d": 1440}
    if value <= 0 or unit not in multipliers:
        raise ValueError(f"Temporalidad no soportada: {timeframe}")
    return value * multipliers[unit]


def horizon_bars(timeframe: str) -> dict[str, int | None]:
    timeframe_minutes = timeframe_to_minutes(timeframe)
    return {
        label: (
            minutes // timeframe_minutes
            if minutes >= timeframe_minutes and minutes % timeframe_minutes == 0
            else None
        )
        for label, minutes in FORECAST_HORIZONS_MINUTES.items()
    }


def conformal_adjustment(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float = 0.80,
) -> float:
    """Calcula el ensanchamiento CQR usando exclusivamente calibracion."""
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage debe estar entre 0 y 1")
    if len(y_true) == 0:
        raise ValueError("La muestra de calibracion no puede estar vacia")
    scores = np.maximum(lower - y_true, y_true - upper)
    quantile_level = min(1.0, math.ceil((len(scores) + 1) * target_coverage) / len(scores))
    try:
        adjustment = float(np.quantile(scores, quantile_level, method="higher"))
    except TypeError:  # compatibilidad con numpy anterior
        adjustment = float(np.quantile(scores, quantile_level, interpolation="higher"))
    # No estrechar intervalos cuantiles que ya superan la cobertura objetivo.
    return max(0.0, adjustment)


def _sorted_quantiles(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 3)
    if predictions.shape[1] != 3:
        raise ValueError("El modelo debe producir tres cuantiles")
    # XGBoost advierte que puede existir quantile crossing.
    return np.sort(predictions, axis=1)


def _new_quantile_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILES,
        n_estimators=180,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=config.MODEL_RANDOM_STATE,
        n_jobs=-1,
    )


def _evaluation_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    adjustment: float,
) -> dict:
    quantiles = _sorted_quantiles(predictions)
    lower = quantiles[:, 0] - adjustment
    median = quantiles[:, 1]
    upper = quantiles[:, 2] + adjustment
    coverage = float(((y_true >= lower) & (y_true <= upper)).mean())
    direction_accuracy = float((np.sign(median) == np.sign(y_true)).mean())
    positive_rate = float((y_true > 0).mean())
    direction_baseline = max(positive_rate, 1.0 - positive_rate)
    median_error_pct = float(np.median(np.abs(np.exp(median - y_true) - 1.0)) * 100.0)
    interval_width_pct = float(np.mean(np.exp(upper) - np.exp(lower)) * 100.0)
    return {
        "coverage": coverage,
        "direction_accuracy": direction_accuracy,
        "direction_baseline": direction_baseline,
        "median_abs_error_pct": median_error_pct,
        "mean_interval_width_pct": interval_width_pct,
        "test_samples": len(y_true),
    }


def train_price_forecasters(
    df: pd.DataFrame,
    feature_cols: list[str],
    timeframe: str,
    target_coverage: float = 0.80,
    max_training_rows: int = 180_000,
) -> dict:
    """Entrena un modelo por horizonte con train/calibracion/test cronologicos."""
    bundles: dict[str, dict] = {}
    bars_by_horizon = horizon_bars(timeframe)

    for label, minutes in FORECAST_HORIZONS_MINUTES.items():
        bars = bars_by_horizon[label]
        if bars is None:
            bundles[label] = {
                "available": False,
                "reason": f"El horizonte es menor o incompatible con velas {timeframe}.",
                "minutes": minutes,
            }
            continue

        working = df[["timestamp", "close", *feature_cols]].copy()
        working["future_log_return"] = np.log(
            working["close"].shift(-bars) / working["close"]
        )
        working = working.replace([np.inf, -np.inf], np.nan).dropna()
        if len(working) > max_training_rows:
            working = working.iloc[-max_training_rows:].copy()
        if len(working) < 5_000:
            bundles[label] = {
                "available": False,
                "reason": "Se necesitan al menos 5,000 observaciones limpias.",
                "minutes": minutes,
            }
            continue

        n_rows = len(working)
        train_end = int(n_rows * 0.70)
        calibration_end = int(n_rows * 0.80)
        # Purga igual al horizonte para impedir que targets de un bloque usen
        # precios pertenecientes al siguiente bloque.
        train_stop = max(1, train_end - bars)
        calibration_stop = max(train_end + 1, calibration_end - bars)

        X_train = working[feature_cols].iloc[:train_stop]
        y_train = working["future_log_return"].iloc[:train_stop].to_numpy()
        X_calibration = working[feature_cols].iloc[train_end:calibration_stop]
        y_calibration = working["future_log_return"].iloc[train_end:calibration_stop].to_numpy()
        X_test = working[feature_cols].iloc[calibration_end:]
        y_test = working["future_log_return"].iloc[calibration_end:].to_numpy()

        model = _new_quantile_model()
        logger.info("Entrenando pronostico %s con %d filas", label, len(X_train))
        model.fit(X_train, y_train, verbose=False)

        calibration_predictions = _sorted_quantiles(model.predict(X_calibration))
        adjustment = conformal_adjustment(
            y_calibration,
            calibration_predictions[:, 0],
            calibration_predictions[:, 2],
            target_coverage,
        )
        test_predictions = model.predict(X_test)
        metrics = _evaluation_metrics(y_test, test_predictions, adjustment)
        calibration_residuals = (
            y_calibration - calibration_predictions[:, 1]
        ).astype(np.float32)

        bundles[label] = {
            "available": True,
            "minutes": minutes,
            "bars": bars,
            "model": model,
            "adjustment": adjustment,
            "calibration_residuals": calibration_residuals,
            "metrics": metrics,
            "test_start": working["timestamp"].iloc[calibration_end],
            "test_end": working["timestamp"].iloc[-1],
            "target_coverage": target_coverage,
        }
    return bundles


def predict_price_ranges(
    bundles: dict,
    latest_features: pd.DataFrame,
    current_price: float,
) -> list[dict]:
    if current_price <= 0:
        raise ValueError("current_price debe ser positivo")
    rows = []
    for label, minutes in FORECAST_HORIZONS_MINUTES.items():
        bundle = bundles.get(label, {"available": False, "reason": "Sin modelo"})
        if not bundle.get("available"):
            rows.append({
                "horizon": label,
                "available": False,
                "reason": bundle.get("reason", "No disponible"),
            })
            continue

        quantiles = _sorted_quantiles(bundle["model"].predict(latest_features))[0]
        lower_return = float(quantiles[0] - bundle["adjustment"])
        median_return = float(quantiles[1])
        upper_return = float(quantiles[2] + bundle["adjustment"])
        residuals = np.asarray(bundle["calibration_residuals"])
        probability_up = float((residuals > -median_return).mean())
        metrics = bundle["metrics"]
        rows.append({
            "horizon": label,
            "available": True,
            "lower_price": current_price * math.exp(lower_return),
            "median_price": current_price * math.exp(median_return),
            "upper_price": current_price * math.exp(upper_return),
            "expected_change_pct": (math.exp(median_return) - 1.0) * 100.0,
            "probability_up": probability_up,
            "coverage": metrics["coverage"],
            "direction_accuracy": metrics["direction_accuracy"],
            "direction_baseline": metrics["direction_baseline"],
            "median_abs_error_pct": metrics["median_abs_error_pct"],
            "mean_interval_width_pct": metrics["mean_interval_width_pct"],
            "test_samples": metrics["test_samples"],
            "target_coverage": bundle["target_coverage"],
        })
    return rows
