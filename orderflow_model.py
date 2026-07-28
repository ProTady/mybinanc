"""Modelo explicable de microestructura agrupada en velas de 5 minutos."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config
from whale_trades import BINANCE_KLINES_URL, normalize_binance_symbol

ORDERFLOW_TIMEFRAME = "5m"
ORDERFLOW_INTERVAL_MS = 5 * 60 * 1000


def init_orderflow_db(db_file: str = config.DB_FILE) -> None:
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orderflow_klines (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                close_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                base_volume REAL NOT NULL,
                quote_volume REAL NOT NULL,
                trade_count INTEGER NOT NULL,
                taker_buy_base REAL NOT NULL,
                taker_buy_quote REAL NOT NULL,
                PRIMARY KEY(symbol, timeframe, open_time)
            )
        """)


def _fetch_orderflow_klines(
    symbol: str,
    start_time: int,
    end_time: int,
    limit: int = 1000,
    timeout: float = 15.0,
) -> list[list]:
    params = {
        "symbol": normalize_binance_symbol(symbol),
        "interval": ORDERFLOW_TIMEFRAME,
        "startTime": int(start_time),
        "endTime": int(end_time),
        "limit": min(1000, int(limit)),
    }
    request = Request(
        f"{BINANCE_KLINES_URL}?{urlencode(params)}",
        headers={"User-Agent": "binapredic/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if isinstance(payload, dict) and "code" in payload:
        raise RuntimeError(f"Binance API {payload.get('code')}: {payload.get('msg')}")
    if not isinstance(payload, list):
        raise RuntimeError("Respuesta inesperada de Binance para klines")
    return payload


def _save_orderflow_klines(
    symbol: str, rows: list[list], db_file: str = config.DB_FILE
) -> int:
    if not rows:
        return 0
    normalized = normalize_binance_symbol(symbol)
    records = [(
        normalized,
        ORDERFLOW_TIMEFRAME,
        int(row[0]),
        int(row[6]),
        float(row[1]),
        float(row[2]),
        float(row[3]),
        float(row[4]),
        float(row[5]),
        float(row[7]),
        int(row[8]),
        float(row[9]),
        float(row[10]),
    ) for row in rows]
    init_orderflow_db(db_file)
    with sqlite3.connect(db_file) as conn:
        before = conn.total_changes
        conn.executemany("""
            INSERT INTO orderflow_klines
            (symbol, timeframe, open_time, close_time, open, high, low, close,
             base_volume, quote_volume, trade_count, taker_buy_base,
             taker_buy_quote)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe, open_time) DO UPDATE SET
                close_time=excluded.close_time,
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                base_volume=excluded.base_volume,
                quote_volume=excluded.quote_volume,
                trade_count=excluded.trade_count,
                taker_buy_base=excluded.taker_buy_base,
                taker_buy_quote=excluded.taker_buy_quote
        """, records)
        changed = conn.total_changes - before
    return changed


def sync_orderflow_klines(
    symbol: str,
    db_file: str = config.DB_FILE,
    initial_lookback_days: int = 30,
) -> dict:
    """Descarga 30 dias inicialmente y despues solo velas nuevas/corregidas."""
    init_orderflow_db(db_file)
    normalized = normalize_binance_symbol(symbol)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    with sqlite3.connect(db_file) as conn:
        row = conn.execute(
            "SELECT MAX(open_time) FROM orderflow_klines "
            "WHERE symbol=? AND timeframe=?",
            (normalized, ORDERFLOW_TIMEFRAME),
        ).fetchone()
    latest = int(row[0]) if row and row[0] is not None else None
    if latest is None:
        next_start = now_ms - initial_lookback_days * 24 * 3600 * 1000
    else:
        # Reconsultar la ultima vela permite corregir datos parciales heredados.
        next_start = latest

    inserted_or_updated = 0
    requests_made = 0
    while next_start < now_ms:
        page = _fetch_orderflow_klines(normalized, next_start, now_ms)
        requests_made += 1
        if not page:
            break
        closed = [row for row in page if int(row[6]) < now_ms]
        inserted_or_updated += _save_orderflow_klines(normalized, closed, db_file)
        page_last = int(page[-1][0])
        if len(page) < 1000 or page_last < next_start:
            break
        next_start = page_last + ORDERFLOW_INTERVAL_MS

    return {
        "rows_changed": inserted_or_updated,
        "requests": requests_made,
    }


def load_orderflow_klines(
    symbol: str,
    db_file: str = config.DB_FILE,
    lookback_days: int = 30,
) -> pd.DataFrame:
    init_orderflow_db(db_file)
    normalized = normalize_binance_symbol(symbol)
    cutoff = int(
        (datetime.now(timezone.utc).timestamp() - lookback_days * 86400) * 1000
    )
    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query("""
            SELECT open_time, close_time, open, high, low, close, base_volume,
                   quote_volume, trade_count, taker_buy_base, taker_buy_quote
            FROM orderflow_klines
            WHERE symbol=? AND timeframe=? AND open_time>=?
            ORDER BY open_time
        """, conn, params=(normalized, ORDERFLOW_TIMEFRAME, cutoff))
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def build_orderflow_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy().sort_values("open_time").reset_index(drop=True)
    safe_volume = data["base_volume"].replace(0, np.nan)
    safe_quote = data["quote_volume"].replace(0, np.nan)
    safe_open = data["open"].replace(0, np.nan)
    safe_trades = data["trade_count"].replace(0, np.nan)

    data["taker_sell_base"] = data["base_volume"] - data["taker_buy_base"]
    data["taker_sell_quote"] = data["quote_volume"] - data["taker_buy_quote"]
    data["buy_ratio_base"] = data["taker_buy_base"] / safe_volume
    data["buy_ratio_quote"] = data["taker_buy_quote"] / safe_quote
    data["flow_imbalance"] = (
        data["taker_buy_base"] - data["taker_sell_base"]
    ) / safe_volume
    data["signed_quote_flow"] = (
        data["taker_buy_quote"] - data["taker_sell_quote"]
    ) / safe_quote
    data["avg_trade_btc"] = data["base_volume"] / safe_trades
    data["avg_trade_usdt"] = data["quote_volume"] / safe_trades
    data["return_5m"] = data["close"].pct_change()
    data["body_pct"] = (data["close"] - data["open"]) / safe_open
    data["range_pct"] = (data["high"] - data["low"]) / safe_open
    data["upper_wick_pct"] = (
        data["high"] - data[["open", "close"]].max(axis=1)
    ) / safe_open
    data["lower_wick_pct"] = (
        data[["open", "close"]].min(axis=1) - data["low"]
    ) / safe_open
    data["vwap_distance"] = (
        data["close"] - data["quote_volume"] / safe_volume
    ) / data["close"].replace(0, np.nan)

    volume_mean = data["base_volume"].rolling(48).mean().shift(1)
    volume_std = data["base_volume"].rolling(48).std().shift(1)
    count_mean = data["trade_count"].rolling(48).mean().shift(1)
    count_std = data["trade_count"].rolling(48).std().shift(1)
    data["volume_zscore"] = (data["base_volume"] - volume_mean) / (volume_std + 1e-9)
    data["trade_count_zscore"] = (
        data["trade_count"] - count_mean
    ) / (count_std + 1e-9)
    data["imbalance_ma_3"] = data["flow_imbalance"].rolling(3).mean()
    data["imbalance_ma_12"] = data["flow_imbalance"].rolling(12).mean()
    data["volume_ratio_12"] = data["base_volume"] / (
        data["base_volume"].rolling(12).mean() + 1e-9
    )

    for lag in (1, 2, 3):
        data[f"imbalance_lag_{lag}"] = data["flow_imbalance"].shift(lag)
        data[f"return_lag_{lag}"] = data["return_5m"].shift(lag)

    minutes = data["timestamp"].dt.hour * 60 + data["timestamp"].dt.minute
    data["time_sin"] = np.sin(2 * np.pi * minutes / 1440)
    data["time_cos"] = np.cos(2 * np.pi * minutes / 1440)

    next_close = data["close"].shift(-1)
    data["next_return"] = next_close / data["close"] - 1.0
    data["target"] = np.where(next_close.notna(), (next_close > data["close"]).astype(int), np.nan)
    data["candle_direction"] = np.where(
        data["close"] > data["open"], "positive",
        np.where(data["close"] < data["open"], "negative", "doji"),
    )
    data["flow_regime"] = pd.cut(
        data["flow_imbalance"],
        bins=[-np.inf, -0.20, -0.05, 0.05, 0.20, np.inf],
        labels=[
            "venta fuerte", "venta moderada", "neutral",
            "compra moderada", "compra fuerte",
        ],
    )

    feature_cols = [
        "buy_ratio_base", "buy_ratio_quote", "flow_imbalance",
        "signed_quote_flow", "avg_trade_btc", "avg_trade_usdt",
        "return_5m", "body_pct", "range_pct", "upper_wick_pct",
        "lower_wick_pct", "vwap_distance", "volume_zscore",
        "trade_count_zscore", "imbalance_ma_3", "imbalance_ma_12",
        "volume_ratio_12", "imbalance_lag_1", "imbalance_lag_2",
        "imbalance_lag_3", "return_lag_1", "return_lag_2",
        "return_lag_3", "time_sin", "time_cos",
    ]
    return data, feature_cols


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best = (0.0, 0.5)
    for threshold in np.linspace(0.40, 0.60, 41):
        predictions = (probabilities >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, predictions)
        if score > best[0]:
            best = (score, float(threshold))
    return best[1]


def train_orderflow_model(
    featured: pd.DataFrame,
    feature_cols: list[str],
    minimum_rows: int = 2_000,
) -> dict:
    clean = featured.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*feature_cols, "target", "next_return"]
    )
    if len(clean) < minimum_rows:
        return {
            "available": False,
            "reason": f"Se necesitan {minimum_rows:,} velas; disponibles: {len(clean):,}.",
            "rows": len(clean),
        }

    n_rows = len(clean)
    train_end = int(n_rows * 0.70)
    calibration_end = int(n_rows * 0.80)
    X_train = clean[feature_cols].iloc[:train_end]
    y_train = clean["target"].iloc[:train_end].astype(int)
    X_calibration = clean[feature_cols].iloc[train_end:calibration_end]
    y_calibration = clean["target"].iloc[train_end:calibration_end].astype(int)
    X_test = clean[feature_cols].iloc[calibration_end:]
    y_test = clean["target"].iloc[calibration_end:].astype(int)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.MODEL_RANDOM_STATE,
        n_jobs=-1,
    )
    weights = compute_sample_weight("balanced", y_train)
    model.fit(X_train, y_train, sample_weight=weights, verbose=False)
    calibration_probabilities = model.predict_proba(X_calibration)[:, 1]
    threshold = _select_threshold(
        y_calibration.to_numpy(), calibration_probabilities
    )
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    positive_rate = float(y_test.mean())
    metrics = {
        "accuracy": accuracy_score(y_test, test_predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_predictions),
        "precision_up": precision_score(y_test, test_predictions, zero_division=0),
        "recall_up": recall_score(y_test, test_predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_probabilities),
        "baseline_accuracy": max(positive_rate, 1.0 - positive_rate),
        "test_rows": len(y_test),
        "test_start": clean["timestamp"].iloc[calibration_end],
        "test_end": clean["timestamp"].iloc[-1],
    }
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return {
        "available": True,
        "model": model,
        "threshold": threshold,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "importances": importances,
        "rows": len(clean),
    }


def predict_next_orderflow_candle(
    bundle: dict,
    featured: pd.DataFrame,
) -> dict:
    if not bundle.get("available"):
        return bundle
    latest = featured.replace([np.inf, -np.inf], np.nan).dropna(
        subset=bundle["feature_cols"]
    ).iloc[-1]
    # A row selected from a mixed-type DataFrame inherits ``object`` dtype.
    # Force numeric input so XGBoost receives the exact schema used in training.
    X_latest = latest[bundle["feature_cols"]].astype(float).to_frame().T
    probability_up = float(bundle["model"].predict_proba(X_latest)[0, 1])
    prediction = "positive" if probability_up >= bundle["threshold"] else "negative"
    buy_base = float(latest["taker_buy_base"])
    sell_base = float(latest["taker_sell_base"])
    imbalance = float(latest["flow_imbalance"])

    if latest["candle_direction"] == "positive" and imbalance > 0.05:
        explanation = "Vela positiva confirmada por compras agresoras."
    elif latest["candle_direction"] == "negative" and imbalance < -0.05:
        explanation = "Vela negativa confirmada por ventas agresoras."
    elif latest["candle_direction"] == "positive" and imbalance < -0.05:
        explanation = "Absorción compradora: la vela subió pese a ventas agresoras dominantes."
    elif latest["candle_direction"] == "negative" and imbalance > 0.05:
        explanation = "Absorción vendedora: la vela cayó pese a compras agresoras dominantes."
    else:
        explanation = "Flujo equilibrado; el color tuvo baja confirmación por volumen taker."

    return {
        "available": True,
        "timestamp": latest["timestamp"],
        "open": float(latest["open"]),
        "close": float(latest["close"]),
        "candle_direction": latest["candle_direction"],
        "buy_base": buy_base,
        "sell_base": sell_base,
        "delta_base": buy_base - sell_base,
        "buy_ratio": float(latest["buy_ratio_base"]),
        "trade_count": int(latest["trade_count"]),
        "flow_regime": str(latest["flow_regime"]),
        "explanation": explanation,
        "probability_up": probability_up,
        "probability_down": 1.0 - probability_up,
        "prediction": prediction,
        "threshold": bundle["threshold"],
        "metrics": bundle["metrics"],
    }


def build_orderflow_pattern_stats(featured: pd.DataFrame) -> pd.DataFrame:
    clean = featured.dropna(subset=["flow_regime", "target", "next_return"])
    if clean.empty:
        return pd.DataFrame()
    return (
        clean.groupby(["candle_direction", "flow_regime"], observed=True)
        .agg(
            candles=("target", "size"),
            next_positive_rate=("target", "mean"),
            next_mean_return=("next_return", "mean"),
            mean_imbalance=("flow_imbalance", "mean"),
            mean_volume=("base_volume", "mean"),
        )
        .reset_index()
    )


def load_top_trades_for_candle(
    symbol: str,
    candle_open_time: pd.Timestamp,
    db_file: str = config.DB_FILE,
    limit: int = 20,
) -> pd.DataFrame:
    start_ms = int(candle_open_time.timestamp() * 1000)
    end_ms = start_ms + ORDERFLOW_INTERVAL_MS
    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query("""
            SELECT timestamp, price, quantity, quote_value, side
            FROM aggregate_trades
            WHERE symbol=? AND timestamp>=? AND timestamp<?
            ORDER BY quote_value DESC
            LIMIT ?
        """, conn, params=(
            normalize_binance_symbol(symbol), start_ms, end_ms, limit
        ))
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
