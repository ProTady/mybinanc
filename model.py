"""Entrenamiento y evaluacion temporal del modelo de senales de trading."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config

logger = logging.getLogger(__name__)

CLASS_LABELS = np.array([-1, 0, 1], dtype=int)
CLASS_NAMES = ["short", "flat", "long"]


def split_time_series_data(
    df: pd.DataFrame, feature_cols: list[str], test_size: float = 0.2
) -> tuple:
    """Divide cronologicamente, dejando el bloque mas reciente como test final."""
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1")
    if len(df) < 2:
        raise ValueError("No hay suficientes datos para dividir")

    split_idx = int(len(df) * (1 - test_size))
    X = df[feature_cols]
    y = df["target"]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_train, df_test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    logger.info(
        "Train: %d (%s a %s) | Test: %d (%s a %s)",
        len(X_train), df_train["timestamp"].min(), df_train["timestamp"].max(),
        len(X_test), df_test["timestamp"].min(), df_test["timestamp"].max(),
    )
    return X_train, X_test, y_train, y_test, df_train, df_test


def _encode_target(y: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=int)
    if not np.isin(values, CLASS_LABELS).all():
        raise ValueError("El target solo puede contener -1 (short), 0 (flat) y 1 (long)")
    return values + 1


def _new_model(n_estimators: int = 300, early_stopping_rounds: int | None = None) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.MODEL_RANDOM_STATE,
        eval_metric="mlogloss",
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1,
    )


def probabilities_to_signals(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Convierte probabilidades [short, flat, long] en -1/0/1 con abstencion."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Se esperaban probabilidades con forma (n, 3)")
    best_class = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    signals = CLASS_LABELS[best_class]
    # Flat siempre es una decision valida. Las direcciones requieren confianza.
    directional = signals != 0
    signals[directional & (confidence < threshold)] = 0
    return signals.astype(int)


def optimize_signal_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    risk_reward_ratio: float = config.RISK_REWARD_RATIO,
) -> tuple[float, dict]:
    """Elige el umbral usando solo predicciones out-of-fold temporales.

    El score es una utilidad conservadora por observacion: una direccion correcta
    gana R, la opuesta pierde 1 y operar una etiqueta flat pierde 0.5. No operar
    aporta cero, por lo que el optimizador equilibra calidad y cobertura.
    """
    y_true = np.asarray(y_true, dtype=int)
    thresholds = np.linspace(
        config.SIGNAL_THRESHOLD_MIN,
        config.SIGNAL_THRESHOLD_MAX,
        config.SIGNAL_THRESHOLD_STEPS,
    )
    best = None
    min_signals = max(30, int(len(y_true) * 0.002))

    for threshold in thresholds:
        signals = probabilities_to_signals(probabilities, float(threshold))
        active = signals != 0
        count = int(active.sum())
        if count < min_signals:
            continue
        utility = np.zeros(len(y_true), dtype=float)
        utility[active & (signals == y_true)] = risk_reward_ratio
        utility[active & (y_true == 0)] = -0.5
        utility[active & (y_true != 0) & (signals != y_true)] = -1.0
        score = float(utility.mean())
        precision = float((signals[active] == y_true[active]).mean())
        candidate = (score, precision, -threshold, float(threshold), count)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    if best is None:
        return 1.0, {
            "utility": 0.0,
            "directional_precision": 0.0,
            "coverage": 0.0,
            "signals": 0,
            "enabled": False,
        }

    score, precision, _, threshold, count = best
    if score <= 0:
        # Si ninguna configuracion aporta utilidad positiva en train, operar en
        # test seria convertir el holdout en un experimento. El modelo se apaga.
        return 1.0, {
            "utility": score,
            "directional_precision": precision,
            "coverage": 0.0,
            "signals": 0,
            "enabled": False,
        }
    return threshold, {
        "utility": score,
        "directional_precision": precision,
        "coverage": count / len(y_true),
        "signals": count,
        "enabled": True,
    }


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    gap: int = config.TARGET_HORIZON_BARS,
) -> XGBClassifier:
    """Entrena XGBoost multiclase y aprende el umbral con walk-forward purgado."""
    if len(X_train) <= n_splits + gap:
        raise ValueError("No hay suficientes filas para la validacion temporal")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    oof_indices: list[np.ndarray] = []
    oof_probabilities: list[np.ndarray] = []
    fold_metrics = []
    best_iterations = []
    encoded = _encode_target(y_train)

    logger.info("Validacion walk-forward con %d folds y gap=%d", n_splits, gap)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        model = _new_model(early_stopping_rounds=30)
        weights = compute_sample_weight("balanced", encoded[train_idx])
        model.fit(
            X_train.iloc[train_idx], encoded[train_idx],
            sample_weight=weights,
            eval_set=[(X_train.iloc[val_idx], encoded[val_idx])],
            verbose=False,
        )
        probabilities = model.predict_proba(X_train.iloc[val_idx])
        raw_predictions = CLASS_LABELS[probabilities.argmax(axis=1)]
        balanced_acc = balanced_accuracy_score(y_train.iloc[val_idx], raw_predictions)
        macro_f1 = f1_score(y_train.iloc[val_idx], raw_predictions, average="macro", zero_division=0)
        fold_metrics.append({"balanced_accuracy": balanced_acc, "macro_f1": macro_f1})
        oof_indices.append(val_idx)
        oof_probabilities.append(probabilities)
        best_iterations.append(int(getattr(model, "best_iteration", 299)) + 1)
        logger.info("Fold %d/%d - balanced_acc=%.4f macro_f1=%.4f", fold, n_splits, balanced_acc, macro_f1)

    all_indices = np.concatenate(oof_indices)
    all_probabilities = np.vstack(oof_probabilities)
    threshold, threshold_metrics = optimize_signal_threshold(
        y_train.iloc[all_indices].to_numpy(), all_probabilities
    )
    final_estimators = max(50, int(np.median(best_iterations)))
    final_model = _new_model(n_estimators=final_estimators)
    final_weights = compute_sample_weight("balanced", encoded)
    final_model.fit(X_train, encoded, sample_weight=final_weights, verbose=False)

    # Metadatos de entrenamiento disponibles para consola y dashboard.
    final_model.signal_threshold_ = threshold
    final_model.threshold_metrics_ = threshold_metrics
    final_model.cv_metrics_ = fold_metrics
    final_model.class_names_ = CLASS_NAMES
    logger.info("Umbral OOF=%.3f | cobertura=%.2f%% | precision direccional=%.2f%%", threshold,
                threshold_metrics["coverage"] * 100, threshold_metrics["directional_precision"] * 100)
    return final_model


def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evalua una sola vez sobre el test final usando el umbral aprendido en train."""
    probabilities = model.predict_proba(X_test)
    threshold = float(getattr(model, "signal_threshold_", 0.5))
    predictions = probabilities_to_signals(probabilities, threshold)
    active = predictions != 0
    directional_precision = (
        float((predictions[active] == y_test.to_numpy()[active]).mean()) if active.any() else 0.0
    )
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, labels=CLASS_LABELS, average="macro", zero_division=0),
        "recall": recall_score(y_test, predictions, labels=CLASS_LABELS, average="macro", zero_division=0),
        "f1_score": f1_score(y_test, predictions, labels=CLASS_LABELS, average="macro", zero_division=0),
        "directional_precision": directional_precision,
        "signal_coverage": float(active.mean()),
        "signal_threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CLASS_LABELS),
        "predictions": predictions,
        "probabilities": probabilities,
    }
    logger.info(
        "Test: accuracy=%.4f balanced_acc=%.4f macro_f1=%.4f dir_precision=%.4f cobertura=%.2f%%",
        metrics["accuracy"], metrics["balanced_accuracy"], metrics["f1_score"],
        directional_precision, metrics["signal_coverage"] * 100,
    )
    return metrics


def get_feature_importances(model: XGBClassifier, feature_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
