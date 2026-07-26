import unittest

import numpy as np
import pandas as pd

from backtest import calculate_backtest_metrics, run_backtest
from features import add_target_and_clean
from model import optimize_signal_threshold, probabilities_to_signals


class TargetTests(unittest.TestCase):
    def test_triple_barrier_labels_long_and_drops_unknown_tail(self):
        timestamps = pd.date_range("2026-01-01", periods=6, freq="5min")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": [100, 100, 101, 101, 101, 101],
            "high": [100.5, 102, 101.5, 101.5, 101.5, 101.5],
            "low": [99.5, 99.5, 100.5, 100.5, 100.5, 100.5],
            "close": [100, 101.5, 101, 101, 101, 101],
            "atr": [1, 1, 1, 1, 1, 1],
        })
        labeled = add_target_and_clean(
            df, horizon_bars=2, atr_multiplier=1, risk_reward_ratio=1,
            fee=0, slippage=0,
        )
        self.assertEqual(int(labeled.iloc[0]["target"]), 1)
        self.assertEqual(len(labeled), len(df) - 2)
        self.assertEqual(labeled["timestamp"].max(), timestamps[-3])


class SignalTests(unittest.TestCase):
    def test_confidence_threshold_abstains(self):
        probabilities = np.array([[0.40, 0.20, 0.40], [0.10, 0.80, 0.10]])
        np.testing.assert_array_equal(
            probabilities_to_signals(probabilities, 0.60), np.array([0, 0])
        )

    def test_unprofitable_validation_disables_trading(self):
        y = np.zeros(100, dtype=int)
        probabilities = np.tile([0.8, 0.1, 0.1], (100, 1))
        threshold, diagnostics = optimize_signal_threshold(y, probabilities)
        self.assertEqual(threshold, 1.0)
        self.assertFalse(diagnostics["enabled"])


class BacktestTests(unittest.TestCase):
    def test_flat_signals_create_no_trades(self):
        timestamps = pd.date_range("2026-01-01", periods=4, freq="5min")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": [100] * 4,
            "high": [101] * 4,
            "low": [99] * 4,
            "close": [100] * 4,
            "atr": [1] * 4,
        })
        equity, trades, metrics = run_backtest(df, np.zeros(4, dtype=int))
        self.assertFalse(trades)
        self.assertEqual(metrics["total_return_pct"], 0.0)
        self.assertEqual(float(equity.iloc[-1]["capital"]), 10000.0)

    def test_profit_factor_uses_net_pnl(self):
        equity = pd.DataFrame({"capital": [100, 99], "price": [100, 100]})
        trades = [
            {"pnl": 10, "fees": 12, "net_pnl": -2},
            {"pnl": 5, "fees": 0, "net_pnl": 5},
        ]
        metrics = calculate_backtest_metrics(equity, trades, 100)
        self.assertEqual(metrics["profit_factor"], 2.5)


if __name__ == "__main__":
    unittest.main()
