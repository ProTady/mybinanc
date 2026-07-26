import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from whale_trades import (
    classify_large_trades,
    calculate_flow_adjusted_levels,
    calculate_flow_pressure,
    load_aggregate_trades,
    parse_aggregate_trade,
    predict_next_large_buy,
    save_aggregate_trades,
    summarize_taker_volume_klines,
)


class AggregateTradeTests(unittest.TestCase):
    def test_aggressor_side_is_derived_from_buyer_maker(self):
        base = {"a": 1, "p": "100", "q": "2", "f": 10, "l": 10, "T": 1}
        sell = parse_aggregate_trade({**base, "m": True}, "BTC/USDT")
        buy = parse_aggregate_trade({**base, "a": 2, "m": False}, "BTC/USDT")
        self.assertEqual(sell["side"], "sell")
        self.assertEqual(buy["side"], "buy")
        self.assertEqual(buy["quote_value"], 200.0)

    def test_persistence_and_loading(self):
        handle, path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        raw = [
            {"a": 1, "p": "100", "q": "2", "f": 1, "l": 1, "T": now_ms, "m": False},
            {"a": 2, "p": "101", "q": "3", "f": 2, "l": 2, "T": now_ms + 1, "m": True},
        ]
        try:
            self.assertEqual(save_aggregate_trades(raw, "BTC/USDT", path), 2)
            self.assertEqual(save_aggregate_trades(raw, "BTC/USDT", path), 0)
            loaded = load_aggregate_trades("BTC/USDT", 1, path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(set(loaded["side"]), {"buy", "sell"})
        finally:
            os.remove(path)

    def test_dynamic_threshold_uses_floor(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC"),
            "quote_value": [100, 200, 300, 400],
            "side": ["buy", "sell", "buy", "sell"],
            "price": [10] * 4,
            "quantity": [10, 20, 30, 40],
        })
        large, threshold = classify_large_trades(frame, min_quote_value=350, percentile=50)
        self.assertEqual(threshold, 350)
        self.assertEqual(len(large), 1)

    def test_predictor_is_probabilistic_and_monotonic(self):
        now = pd.Timestamp("2026-01-01 01:00:00", tz="UTC")
        times = pd.date_range(end=now - pd.Timedelta(minutes=1), periods=30, freq="2min", tz="UTC")
        frame = pd.DataFrame({
            "timestamp": times,
            "side": ["buy" if i % 2 == 0 else "sell" for i in range(30)],
            "quote_value": np.full(30, 100_000.0),
        })
        result = predict_next_large_buy(frame, now=now, minimum_events=12)
        probabilities = result["probabilities"]
        self.assertEqual(result["status"], "ok")
        self.assertLess(probabilities[5], probabilities[15])
        self.assertLess(probabilities[15], probabilities[30])
        self.assertTrue(all(0 <= value <= 1 for value in probabilities.values()))

    def test_taker_volume_summary_separates_buy_and_sell(self):
        now_ms = 1_000_000_000
        # Campos relevantes: [0]=apertura, [5]=volumen base, [9]=compra taker.
        row_recent = [now_ms - 60_000, "0", "0", "0", "0", "10", 0, 0, 0, "6"]
        row_48h = [now_ms - 48 * 3600_000, "0", "0", "0", "0", "20", 0, 0, 0, "5"]
        summaries = summarize_taker_volume_klines([row_recent, row_48h], (24, 72), now_ms)
        self.assertEqual(summaries[24]["buy_base"], 6)
        self.assertEqual(summaries[24]["sell_base"], 4)
        self.assertEqual(summaries[72]["buy_base"], 11)
        self.assertEqual(summaries[72]["sell_base"], 19)

    def test_flow_pressure_and_levels_are_directionally_symmetric(self):
        summaries = {
            24: {"total_base": 100, "net_base": 20},
            72: {"total_base": 300, "net_base": 30},
        }
        pressure = calculate_flow_pressure(summaries, large_buy_quote=200, large_sell_quote=100)
        self.assertGreater(pressure["score"], 0)
        levels = calculate_flow_adjusted_levels(100, 2, pressure["score"], fee=0, slippage=0)
        self.assertLess(levels["long"]["wait_atr"], levels["short"]["wait_atr"])
        self.assertLess(levels["long"]["entry"], 100)
        self.assertGreater(levels["short"]["entry"], 100)
        self.assertGreater(levels["long"]["target"], levels["long"]["entry"])
        self.assertLess(levels["short"]["target"], levels["short"]["entry"])


if __name__ == "__main__":
    unittest.main()
