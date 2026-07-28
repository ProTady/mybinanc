"""Flujo publico de operaciones agregadas grandes de Binance Spot.

No identifica personas ni wallets. El lado representa al agresor (taker): si el
comprador es maker, el agresor fue una venta; en caso contrario fue una compra.
"""

from __future__ import annotations

import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import config

BINANCE_MARKET_DATA_URL = "https://data-api.binance.vision/api/v3/aggTrades"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


def normalize_binance_symbol(symbol: str) -> str:
    """Convierte BTC/USDT a BTCUSDT y rechaza valores no aptos para Spot."""
    normalized = symbol.split(":", 1)[0].replace("/", "").replace("-", "").upper().strip()
    if not normalized or not normalized.isalnum():
        raise ValueError("Simbolo invalido para Binance Spot")
    return normalized


def init_agg_trades_db(db_file: str = config.DB_FILE) -> None:
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS aggregate_trades (
                symbol TEXT NOT NULL,
                agg_trade_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                quote_value REAL NOT NULL,
                side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
                buyer_is_maker INTEGER NOT NULL,
                first_trade_id INTEGER,
                last_trade_id INTEGER,
                PRIMARY KEY (symbol, agg_trade_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_aggregate_trades_symbol_time
            ON aggregate_trades(symbol, timestamp)
        """)


def fetch_aggregate_trades(
    symbol: str,
    limit: int = 1000,
    from_id: int | None = None,
    timeout: float = 10.0,
) -> list[dict]:
    """Descarga trades agregados publicos; no requiere API key."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit debe estar entre 1 y 1000")
    params: dict[str, int | str] = {
        "symbol": normalize_binance_symbol(symbol),
        "limit": limit,
    }
    if from_id is not None:
        params["fromId"] = int(from_id)
    request = Request(
        f"{BINANCE_MARKET_DATA_URL}?{urlencode(params)}",
        headers={"User-Agent": "binapredic/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if isinstance(payload, dict) and "code" in payload:
        raise RuntimeError(f"Binance API {payload.get('code')}: {payload.get('msg')}")
    if not isinstance(payload, list):
        raise RuntimeError("Respuesta inesperada de Binance")
    return payload


def summarize_taker_volume_klines(
    klines: list[list],
    periods_hours: tuple[int, ...] = (24, 72),
    now_ms: int | None = None,
) -> dict[int, dict]:
    """Resume volumen base segun el lado taker de velas Binance.

    En la respuesta kline, indice 5 es volumen base total e indice 9 es
    volumen base de compras taker. El resto corresponde a ventas taker.
    """
    now_ms = now_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    summaries: dict[int, dict] = {}
    for hours in periods_hours:
        cutoff = now_ms - int(hours * 3600 * 1000)
        selected = [row for row in klines if cutoff <= int(row[0]) <= now_ms]
        total_base = sum(float(row[5]) for row in selected)
        taker_buy_base = sum(float(row[9]) for row in selected)
        taker_sell_base = max(0.0, total_base - taker_buy_base)
        summaries[int(hours)] = {
            "buy_base": taker_buy_base,
            "sell_base": taker_sell_base,
            "total_base": total_base,
            "net_base": taker_buy_base - taker_sell_base,
            "buy_pct": taker_buy_base / total_base * 100.0 if total_base else 0.0,
            "candles": len(selected),
        }
    return summaries


def fetch_taker_volume_summaries(
    symbol: str,
    periods_hours: tuple[int, ...] = (24, 72),
    interval: str = "5m",
    timeout: float = 10.0,
) -> dict[int, dict]:
    """Obtiene totales completos aproximados a velas de 5 minutos en una llamada."""
    if not periods_hours or max(periods_hours) <= 0:
        raise ValueError("periods_hours debe contener periodos positivos")
    interval_minutes = {"1m": 1, "5m": 5, "15m": 15}.get(interval)
    if interval_minutes is None:
        raise ValueError("Intervalo no soportado para el resumen")
    required = math.ceil(max(periods_hours) * 60 / interval_minutes) + 2
    if required > 1000:
        raise ValueError("El periodo excede el limite de una llamada de Binance")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - int(max(periods_hours) * 3600 * 1000) - interval_minutes * 60_000
    params = {
        "symbol": normalize_binance_symbol(symbol),
        "interval": interval,
        "startTime": start_ms,
        "endTime": now_ms,
        "limit": required,
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
    return summarize_taker_volume_klines(payload, periods_hours, now_ms)


def calculate_flow_pressure(
    volume_summaries: dict[int, dict],
    large_buy_quote: float = 0.0,
    large_sell_quote: float = 0.0,
) -> dict:
    """Combina flujo total 24/72h y operaciones grandes recientes en [-1, 1]."""
    def normalized(summary: dict) -> float:
        total = float(summary.get("total_base", 0.0))
        return float(summary.get("net_base", 0.0)) / total if total > 0 else 0.0

    imbalance_24h = normalized(volume_summaries.get(24, {}))
    imbalance_72h = normalized(volume_summaries.get(72, {}))
    large_total = float(large_buy_quote + large_sell_quote)
    large_imbalance = (
        float(large_buy_quote - large_sell_quote) / large_total if large_total > 0 else 0.0
    )

    total_24_component = float(np.tanh(5.0 * imbalance_24h))
    total_72_component = float(np.tanh(5.0 * imbalance_72h))
    if large_total > 0:
        score = 0.55 * large_imbalance + 0.30 * total_24_component + 0.15 * total_72_component
    else:
        score = 0.65 * total_24_component + 0.35 * total_72_component
    score = float(np.clip(score, -1.0, 1.0))

    if score >= 0.35:
        label = "compradora fuerte"
    elif score >= 0.10:
        label = "compradora moderada"
    elif score <= -0.35:
        label = "vendedora fuerte"
    elif score <= -0.10:
        label = "vendedora moderada"
    else:
        label = "neutral"
    return {
        "score": score,
        "label": label,
        "imbalance_24h": imbalance_24h,
        "imbalance_72h": imbalance_72h,
        "large_imbalance": large_imbalance,
    }


def calculate_flow_adjusted_levels(
    current_price: float,
    atr: float,
    flow_score: float,
    atr_multiplier: float = 2.0,
    risk_reward_ratio: float = 1.5,
    fee: float = 0.001,
    slippage: float = 0.0002,
) -> dict[str, dict]:
    """Genera zonas long/short; el flujo modifica cuanto retroceso esperar."""
    if current_price <= 0 or atr <= 0:
        raise ValueError("current_price y atr deben ser positivos")
    score = float(np.clip(flow_score, -1.0, 1.0))
    # Flujo comprador permite una entrada long mas cercana; flujo vendedor
    # exige mayor descuento. Para short se aplica el razonamiento simetrico.
    long_wait_atr = float(np.clip(0.40 - 0.25 * score, 0.15, 0.65))
    short_wait_atr = float(np.clip(0.40 + 0.25 * score, 0.15, 0.65))
    long_entry = current_price - atr * long_wait_atr
    short_entry = current_price + atr * short_wait_atr
    stop_distance = atr * atr_multiplier
    long_cost_buffer = long_entry * 2.0 * (fee + slippage)
    short_cost_buffer = short_entry * 2.0 * (fee + slippage)
    return {
        "long": {
            "entry": long_entry,
            "stop": long_entry - stop_distance,
            "target": long_entry + stop_distance * risk_reward_ratio + long_cost_buffer,
            "wait_atr": long_wait_atr,
        },
        "short": {
            "entry": short_entry,
            "stop": short_entry + stop_distance,
            "target": short_entry - stop_distance * risk_reward_ratio - short_cost_buffer,
            "wait_atr": short_wait_atr,
        },
    }


def parse_aggregate_trade(raw: dict, symbol: str) -> dict:
    price = float(raw["p"])
    quantity = float(raw["q"])
    buyer_is_maker = bool(raw["m"])
    return {
        "symbol": normalize_binance_symbol(symbol),
        "agg_trade_id": int(raw["a"]),
        "timestamp": int(raw["T"]),
        "price": price,
        "quantity": quantity,
        "quote_value": price * quantity,
        # Buyer maker => el taker vendio. Buyer taker => compra agresora.
        "side": "sell" if buyer_is_maker else "buy",
        "buyer_is_maker": int(buyer_is_maker),
        "first_trade_id": int(raw.get("f", raw["a"])),
        "last_trade_id": int(raw.get("l", raw["a"])),
    }


def save_aggregate_trades(
    raw_trades: list[dict], symbol: str, db_file: str = config.DB_FILE
) -> int:
    if not raw_trades:
        return 0
    init_agg_trades_db(db_file)
    parsed = [parse_aggregate_trade(item, symbol) for item in raw_trades]
    rows = [tuple(item.values()) for item in parsed]
    with sqlite3.connect(db_file) as conn:
        before = conn.total_changes
        conn.executemany("""
            INSERT OR IGNORE INTO aggregate_trades
            (symbol, agg_trade_id, timestamp, price, quantity, quote_value,
             side, buyer_is_maker, first_trade_id, last_trade_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        inserted = conn.total_changes - before
    return inserted


def _latest_stored_id(symbol: str, db_file: str) -> int | None:
    init_agg_trades_db(db_file)
    normalized = normalize_binance_symbol(symbol)
    with sqlite3.connect(db_file) as conn:
        row = conn.execute(
            "SELECT MAX(agg_trade_id) FROM aggregate_trades WHERE symbol = ?",
            (normalized,),
        ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def sync_aggregate_trades(
    symbol: str,
    db_file: str = config.DB_FILE,
    initial_trades: int = 10_000,
    max_incremental_pages: int = 5,
    resync_recent_trades: int = 5_000,
) -> dict:
    """Sincroniza trades y salta al presente si el atraso supera la capacidad.

    Intentar recuperar cada aggTrade despues de muchas horas desconectado puede
    dejar al monitor permanentemente atrasado. En ese caso se conserva la base
    existente, se informa la brecha y se descarga una ventana reciente completa.
    """
    normalized = normalize_binance_symbol(symbol)
    latest = _latest_stored_id(normalized, db_file)
    inserted = 0
    requests_made = 1
    skipped = 0
    resynced = False

    # Consultar siempre la punta remota permite saber inmediatamente si la base
    # local sigue al dia; consultar solo fromId ocultaba atrasos de muchas horas.
    recent = fetch_aggregate_trades(normalized, limit=1000)
    if not recent:
        return {
            "inserted": 0,
            "requests": requests_made,
            "latest_id": latest,
            "remote_latest_id": None,
            "gap_before": 0,
            "skipped": 0,
            "resynced": False,
        }

    newest_id = int(recent[-1]["a"])
    gap_before = max(0, newest_id - latest) if latest is not None else 0

    if latest is None:
        first_id = max(0, newest_id - max(1000, initial_trades) + 1)
        resynced = True
    elif gap_before <= 0:
        # Guardar la pagina reciente tambien repara cualquier hueco pequeno.
        inserted += save_aggregate_trades(recent, normalized, db_file)
        first_id = None
    elif gap_before <= 1000:
        # La pagina reciente contiene todos los IDs nuevos.
        inserted += save_aggregate_trades(recent, normalized, db_file)
        first_id = None
    elif gap_before <= max_incremental_pages * 1000:
        # Brecha recuperable: no se omite ningun trade.
        next_id = latest + 1
        for _ in range(max_incremental_pages):
            page = fetch_aggregate_trades(normalized, limit=1000, from_id=next_id)
            requests_made += 1
            if not page:
                break
            inserted += save_aggregate_trades(page, normalized, db_file)
            page_last = int(page[-1]["a"])
            if page_last >= newest_id or len(page) < 1000 or page_last < next_id:
                break
            next_id = page_last + 1
        first_id = None
    else:
        # Brecha demasiado grande: priorizar datos actuales. Los agregados
        # completos de 24/72h se calculan aparte con klines y no se ven afectados.
        first_id = max(0, newest_id - max(1000, resync_recent_trades) + 1)
        skipped = max(0, first_id - latest - 1)
        resynced = True

    if first_id is not None:
        max_pages = math.ceil((newest_id - first_id + 1) / 1000)
        page_starts = [first_id + page * 1000 for page in range(max_pages)]
        with ThreadPoolExecutor(max_workers=min(5, max_pages)) as executor:
            pages = list(executor.map(
                lambda start: fetch_aggregate_trades(normalized, limit=1000, from_id=start),
                page_starts,
            ))
        requests_made += len(pages)
        for page in pages:
            if not page:
                continue
            inserted += save_aggregate_trades(page, normalized, db_file)

    return {
        "inserted": inserted,
        "requests": requests_made,
        "latest_id": _latest_stored_id(normalized, db_file),
        "remote_latest_id": newest_id,
        "gap_before": gap_before,
        "skipped": skipped,
        "resynced": resynced,
    }


def load_aggregate_trades(
    symbol: str,
    lookback_hours: float = 24.0,
    db_file: str = config.DB_FILE,
    max_rows: int = 100_000,
) -> pd.DataFrame:
    init_agg_trades_db(db_file)
    normalized = normalize_binance_symbol(symbol)
    cutoff_ms = int((datetime.now(timezone.utc).timestamp() - lookback_hours * 3600) * 1000)
    query = """
        SELECT agg_trade_id, timestamp, price, quantity, quote_value, side
        FROM aggregate_trades
        WHERE symbol = ? AND timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT ?
    """
    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query(query, conn, params=(normalized, cutoff_ms, max_rows))
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def classify_large_trades(
    trades: pd.DataFrame,
    min_quote_value: float = 50_000.0,
    percentile: float = 99.5,
) -> tuple[pd.DataFrame, float]:
    """Clasifica grandes operaciones con un piso y percentil dinamico."""
    if not 0 <= percentile <= 100:
        raise ValueError("percentile debe estar entre 0 y 100")
    if trades.empty:
        return trades.copy(), float(min_quote_value)

    dynamic_threshold = float(np.percentile(trades["quote_value"], percentile))
    threshold = max(float(min_quote_value), dynamic_threshold)
    ranked = trades.copy()
    ranked["intensity_percentile"] = ranked["quote_value"].rank(pct=True) * 100.0
    large = ranked[ranked["quote_value"] >= threshold].copy()
    large["seconds_since_previous"] = large["timestamp"].diff().dt.total_seconds()
    large["seconds_since_same_side"] = (
        large.groupby("side")["timestamp"].diff().dt.total_seconds()
    )
    return large.reset_index(drop=True), threshold


def analyze_large_trade_price_impact(
    trades: pd.DataFrame,
    min_quantity: float,
    horizons_minutes: tuple[int, ...] = (1, 5, 15, 30),
    max_lookup_delay_seconds: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mide el cambio posterior a cada aggTrade que supera una cantidad base.

    Busca el primer precio disponible a partir de cada horizonte. Si existe una
    discontinuidad de datos mayor al margen permitido, deja el resultado vacío
    para no confundir una brecha de conexión con impacto de mercado.
    """
    if min_quantity <= 0:
        raise ValueError("min_quantity debe ser positivo")
    required = {"agg_trade_id", "timestamp", "price", "quantity", "side"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    ordered = trades.sort_values(["timestamp", "agg_trade_id"]).reset_index(drop=True)
    events = ordered[ordered["quantity"] >= min_quantity].copy()
    if events.empty:
        return pd.DataFrame(), events

    trade_times = ordered["timestamp"].astype("int64").to_numpy()
    trade_prices = ordered["price"].to_numpy(dtype=float)
    event_times = events["timestamp"].astype("int64").to_numpy()
    event_prices = events["price"].to_numpy(dtype=float)
    tolerance_ns = int(max_lookup_delay_seconds * 1_000_000_000)

    for minutes in horizons_minutes:
        target_times = event_times + int(minutes * 60 * 1_000_000_000)
        future_indices = np.searchsorted(trade_times, target_times, side="left")
        valid = future_indices < len(trade_times)
        safe_indices = np.minimum(future_indices, len(trade_times) - 1)
        lookup_delay = trade_times[safe_indices] - target_times
        valid &= lookup_delay >= 0
        valid &= lookup_delay <= tolerance_ns
        returns = np.full(len(events), np.nan)
        returns[valid] = (
            trade_prices[safe_indices[valid]] / event_prices[valid] - 1.0
        ) * 100.0
        events[f"return_{minutes}m_pct"] = returns

    summaries = []
    for side in ("buy", "sell"):
        side_events = events[events["side"] == side]
        for minutes in horizons_minutes:
            values = side_events[f"return_{minutes}m_pct"].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            summaries.append({
                "side": side,
                "horizon_minutes": minutes,
                "samples": len(values),
                "mean_change_pct": float(np.mean(values)),
                "median_change_pct": float(np.median(values)),
                "p25_change_pct": float(np.percentile(values, 25)),
                "p75_change_pct": float(np.percentile(values, 75)),
                "probability_up": float((values > 0).mean()),
                "probability_down": float((values < 0).mean()),
                "adverse_probability": float(
                    (values < 0).mean() if side == "buy" else (values > 0).mean()
                ),
            })
    return pd.DataFrame(summaries), events.reset_index(drop=True)


def find_new_quantity_alerts(
    trades: pd.DataFrame,
    min_quantity: float,
    last_seen_id: int | None,
) -> pd.DataFrame:
    """Retorna solamente eventos nuevos que ameritan una alerta."""
    if trades.empty:
        return trades.copy()
    alerts = trades[trades["quantity"] >= min_quantity].copy()
    if last_seen_id is not None:
        alerts = alerts[alerts["agg_trade_id"] > last_seen_id]
    return alerts.sort_values("agg_trade_id").reset_index(drop=True)


def seconds_since_last_event(
    large_trades: pd.DataFrame, side: str, now: pd.Timestamp | None = None
) -> float | None:
    events = large_trades[large_trades["side"] == side]
    if events.empty:
        return None
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    last = events["timestamp"].max()
    return max(0.0, float((now - last).total_seconds()))


def predict_next_large_buy(
    large_trades: pd.DataFrame,
    now: pd.Timestamp | None = None,
    horizons_minutes: tuple[int, ...] = (5, 15, 30),
    minimum_events: int = 12,
) -> dict:
    """Estimador empirico de llegada; no predice identidad ni garantiza eventos."""
    buys = large_trades[large_trades["side"] == "buy"].sort_values("timestamp")
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    intervals = buys["timestamp"].diff().dt.total_seconds().dropna()
    if len(buys) < minimum_events or len(intervals) < minimum_events - 1:
        return {
            "status": "insufficient_data",
            "events": len(buys),
            "minimum_events": minimum_events,
            "probabilities": {minutes: None for minutes in horizons_minutes},
            "eta_range_minutes": None,
            "confidence": "insuficiente",
        }

    # Tasa Poisson suavizada con un prior conservador de cinco eventos en 75
    # minutos. Evita probabilidades extremas cuando apenas existe historial.
    median_interval = max(1.0, float(intervals.median()))
    recent_median = max(1.0, float(intervals.tail(min(10, len(intervals))).median()))
    acceleration = float(np.clip(median_interval / recent_median, 0.75, 1.33))

    recent_cutoff = now - pd.Timedelta(minutes=30)
    recent = large_trades[large_trades["timestamp"] >= recent_cutoff]
    buys_recent = int((recent["side"] == "buy").sum())
    sells_recent = int((recent["side"] == "sell").sum())
    imbalance = (buys_recent - sells_recent) / max(1, buys_recent + sells_recent)
    imbalance_factor = float(np.clip(1.0 + 0.15 * imbalance, 0.85, 1.15))

    observed_span = max(1.0, float((buys["timestamp"].iloc[-1] - buys["timestamp"].iloc[0]).total_seconds()))
    prior_events = 5.0
    prior_exposure_seconds = 75.0 * 60.0
    base_rate = (len(intervals) + prior_events) / (observed_span + prior_exposure_seconds)
    rate_per_second = base_rate * acceleration * imbalance_factor
    probabilities = {
        minutes: float(1.0 - np.exp(-rate_per_second * minutes * 60.0))
        for minutes in horizons_minutes
    }
    eta_range = (
        -math.log(0.75) / rate_per_second / 60.0,
        -math.log(0.25) / rate_per_second / 60.0,
    )
    return {
        "status": "ok",
        "events": len(buys),
        "probabilities": probabilities,
        "eta_range_minutes": eta_range,
        "median_interval_minutes": median_interval / 60.0,
        "acceleration": acceleration,
        "recent_imbalance": imbalance,
        "confidence": "moderada" if len(buys) >= 50 else "baja",
    }


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "Sin eventos"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes} min {secs} s"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"
