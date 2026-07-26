"""Backtest causal con abstencion, costes netos y valoracion mark-to-market."""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def run_backtest(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    initial_capital: float = 10000.0,
    risk_per_trade: float = 0.01,
    atr_multiplier: float = 2.0,
    risk_reward_ratio: float = 1.5,
    trade_direction: str = "both",
    fee: float = 0.001,
    slippage: float = config.SLIPPAGE,
    max_holding_bars: int = config.MAX_HOLDING_BARS,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Simula senales -1=short, 0=flat y 1=long sobre velas OHLC.

    La senal de la vela t entra en la apertura t+1. El TP incorpora un buffer
    aproximado de costes, igual al usado al construir el target.
    """
    if trade_direction not in {"long", "short", "both"}:
        raise ValueError("trade_direction debe ser 'long', 'short' o 'both'")
    if max_holding_bars < 1:
        raise ValueError("max_holding_bars debe ser mayor o igual a 1")
    if len(df_test) != len(predictions):
        raise ValueError("df_test y predictions deben tener la misma longitud")
    if len(df_test) == 0:
        raise ValueError("df_test no puede estar vacio")

    df = df_test.copy().reset_index(drop=True)
    df["prediction"] = np.asarray(predictions, dtype=int)
    if not np.isin(df["prediction"], [-1, 0, 1]).all():
        raise ValueError("Las senales validas son -1 (short), 0 (flat) y 1 (long)")

    capital = float(initial_capital)
    position = None
    trades_history: list[dict] = []
    equity_curve: list[dict] = []
    active_bars = 0

    def close_position(exit_price: float, exit_time, reason: str) -> None:
        nonlocal capital, position
        direction = position["type"]
        entry_price = position["entry_price"]
        size = position["size"]
        gross_pnl = (
            (exit_price - entry_price) * size
            if direction == "long"
            else (entry_price - exit_price) * size
        )
        fees_paid = (entry_price + exit_price) * size * fee
        slippage_paid = (entry_price + exit_price) * size * slippage
        net_pnl = gross_pnl - fees_paid - slippage_paid
        capital += net_pnl
        trades_history.append({
            "type": direction,
            "result": "win" if net_pnl > 0 else "loss",
            "exit_reason": reason,
            "entry_time": position["entry_time"],
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "pnl": gross_pnl,
            "fees": fees_paid,
            "slippage": slippage_paid,
            "net_pnl": net_pnl,
            "capital_after": capital,
            "risk": position["risk"],
            "holding_bars": position["holding_bars"],
        })
        position = None

    for i in range(len(df)):
        bar = df.iloc[i]

        # Una posicion abierta en t+1 empieza a evaluarse con el OHLC completo
        # de esa misma vela.
        if position is not None and i >= position["entry_index"]:
            position["holding_bars"] += 1
            active_bars += 1
            high, low, open_price = float(bar["high"]), float(bar["low"]), float(bar["open"])
            stop, take = position["stop_loss"], position["take_profit"]

            if position["type"] == "long":
                stopped = low <= stop
                take_hit = high >= take
                if stopped:  # tambien resuelve conservadoramente una vela que toca ambos
                    close_position(min(stop, open_price), bar["timestamp"], "stop_loss")
                elif take_hit:
                    close_position(take, bar["timestamp"], "take_profit")
            else:
                stopped = high >= stop
                take_hit = low <= take
                if stopped:
                    close_position(max(stop, open_price), bar["timestamp"], "stop_loss")
                elif take_hit:
                    close_position(take, bar["timestamp"], "take_profit")

            if position is not None and position["holding_bars"] >= max_holding_bars:
                close_position(float(bar["close"]), bar["timestamp"], "time_exit")

        # Toda posicion pendiente se liquida en el ultimo cierre.
        if i == len(df) - 1 and position is not None:
            close_position(float(bar["close"]), bar["timestamp"], "end_of_test")

        # Equity mark-to-market: incluye PnL no realizado y coste estimado de salida.
        equity = capital
        if position is not None:
            close_price = float(bar["close"])
            unrealized = (
                (close_price - position["entry_price"]) * position["size"]
                if position["type"] == "long"
                else (position["entry_price"] - close_price) * position["size"]
            )
            estimated_costs = (
                (position["entry_price"] + close_price)
                * position["size"] * (fee + slippage)
            )
            equity = capital + unrealized - estimated_costs

        equity_curve.append({
            "timestamp": bar["timestamp"],
            "capital": equity,
            "price": float(bar["close"]),
        })

        # La senal conocida al cierre actual entra en la proxima apertura.
        if position is None and i < len(df) - 1 and capital > 0:
            signal = int(bar["prediction"])
            allowed = (
                (signal == 1 and trade_direction in {"long", "both"})
                or (signal == -1 and trade_direction in {"short", "both"})
            )
            if allowed:
                next_bar = df.iloc[i + 1]
                entry_price = float(next_bar["open"])
                sl_distance = float(bar["atr"]) * atr_multiplier
                if not np.isfinite(sl_distance) or sl_distance <= 0 or sl_distance >= entry_price:
                    continue
                cost_buffer = entry_price * 2.0 * (fee + slippage)
                tp_distance = sl_distance * risk_reward_ratio + cost_buffer
                risk_amount = capital * risk_per_trade
                size = risk_amount / sl_distance
                max_size = capital / (entry_price * (1.0 + fee + slippage))
                size = min(size, max_size)
                if size <= 0:
                    continue

                direction = "long" if signal == 1 else "short"
                position = {
                    "type": direction,
                    "entry_price": entry_price,
                    "entry_time": next_bar["timestamp"],
                    "entry_index": i + 1,
                    "stop_loss": entry_price - sl_distance if signal == 1 else entry_price + sl_distance,
                    "take_profit": entry_price + tp_distance if signal == 1 else entry_price - tp_distance,
                    "size": size,
                    "risk": sl_distance * size,
                    "holding_bars": 0,
                }

    df_equity = pd.DataFrame(equity_curve)
    metrics = calculate_backtest_metrics(df_equity, trades_history, initial_capital)
    metrics["exposure_pct"] = active_bars / len(df) * 100.0
    return df_equity, trades_history, metrics


def calculate_backtest_metrics(
    df_equity: pd.DataFrame, trades: list[dict], initial_capital: float
) -> dict:
    """Calcula estadisticas usando PnL neto, no el PnL bruto previo a costes."""
    final_capital = float(df_equity.iloc[-1]["capital"])
    total_return = (final_capital / initial_capital - 1.0) * 100.0
    price_start, price_end = df_equity.iloc[0]["price"], df_equity.iloc[-1]["price"]
    hold_return = (price_end / price_start - 1.0) * 100.0

    capitals = df_equity["capital"].to_numpy(dtype=float)
    peaks = np.maximum.accumulate(capitals)
    drawdowns = np.divide(peaks - capitals, peaks, out=np.zeros_like(peaks), where=peaks != 0)
    max_drawdown = float(np.max(drawdowns) * 100.0)

    net_pnls = np.array([t.get("net_pnl", t["pnl"] - t.get("fees", 0.0)) for t in trades])
    winning = int((net_pnls > 0).sum()) if len(net_pnls) else 0
    losing = int(len(net_pnls) - winning)
    gross_profits = float(net_pnls[net_pnls > 0].sum()) if len(net_pnls) else 0.0
    gross_losses = float(abs(net_pnls[net_pnls < 0].sum())) if len(net_pnls) else 0.0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else (float("inf") if gross_profits > 0 else 0.0)

    return {
        "total_return_pct": total_return,
        "hold_return_pct": hold_return,
        "win_rate_pct": winning / len(trades) * 100.0 if trades else 0.0,
        "max_drawdown_pct": max_drawdown,
        "total_trades": len(trades),
        "winning_trades": winning,
        "losing_trades": losing,
        "profit_factor": profit_factor,
        "expectancy": float(net_pnls.mean()) if len(net_pnls) else 0.0,
        "total_costs": float(sum(t.get("fees", 0.0) + t.get("slippage", 0.0) for t in trades)),
        "final_capital": final_capital,
    }
