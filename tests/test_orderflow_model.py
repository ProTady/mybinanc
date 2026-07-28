import unittest

import numpy as np
import pandas as pd

from orderflow_model import (
    build_orderflow_features,
    build_orderflow_pattern_stats,
    predict_next_orderflow_candle,
    train_orderflow_model,
)


def synthetic_klines(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    timestamp = pd.date_range(
        "2026-01-01", periods=rows, freq="5min", tz="UTC"
    )
    flow = np.sin(np.arange(rows) / 7.0) * 0.25
    returns = np.roll(flow, 1) * 0.0005 + rng.normal(0, 0.0004, rows)
    close = 60_000 * np.cumprod(1 + returns)
    open_price = np.r_[close[0], close[:-1]]
    volume = rng.uniform(10, 80, rows)
    taker_buy = volume * (flow + 1) / 2
    high = np.maximum(open_price, close) * 1.0005
    low = np.minimum(open_price, close) * 0.9995
    quote_volume = volume * (open_price + close) / 2
    return pd.DataFrame({
        "open_time": (timestamp.astype("int64") // 1_000_000).astype(int),
        "close_time": (
            timestamp.astype("int64") // 1_000_000 + 299_999
        ).astype(int),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "base_volume": volume,
        "quote_volume": quote_volume,
        "trade_count": rng.integers(500, 2500, rows),
        "taker_buy_base": taker_buy,
        "taker_buy_quote": quote_volume * (flow + 1) / 2,
        "timestamp": timestamp,
    })


class OrderflowModelTests(unittest.TestCase):
    def test_features_separate_aggressive_buy_and_sell_volume(self):
        featured, columns = build_orderflow_features(synthetic_klines())
        row = featured.iloc[100]
        self.assertAlmostEqual(
            row["taker_buy_base"] + row["taker_sell_base"],
            row["base_volume"],
        )
        self.assertAlmostEqual(
            row["flow_imbalance"],
            2 * row["buy_ratio_base"] - 1,
        )
        self.assertIn("flow_imbalance", columns)
        self.assertTrue(np.isnan(featured.iloc[-1]["target"]))

    def test_pattern_table_groups_candle_and_flow_regime(self):
        featured, _ = build_orderflow_features(synthetic_klines())
        patterns = build_orderflow_pattern_stats(featured)
        self.assertFalse(patterns.empty)
        self.assertIn("next_positive_rate", patterns.columns)
        self.assertTrue(patterns["next_positive_rate"].between(0, 1).all())
        self.assertEqual(int(patterns["candles"].sum()), len(featured) - 1)

    def test_latest_mixed_row_is_numeric_for_xgboost_prediction(self):
        featured, columns = build_orderflow_features(synthetic_klines())
        bundle = train_orderflow_model(featured, columns, minimum_rows=200)
        prediction = predict_next_orderflow_candle(bundle, featured)
        self.assertTrue(prediction["available"])
        self.assertGreaterEqual(prediction["probability_up"], 0)
        self.assertLessEqual(prediction["probability_up"], 1)
        self.assertAlmostEqual(
            prediction["buy_base"] + prediction["sell_base"],
            synthetic_klines().iloc[-1]["base_volume"],
        )


if __name__ == "__main__":
    unittest.main()
