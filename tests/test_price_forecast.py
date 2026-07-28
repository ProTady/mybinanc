import unittest

import numpy as np
import pandas as pd

from price_forecast import (
    conformal_adjustment,
    horizon_bars,
    predict_price_ranges,
    timeframe_to_minutes,
)


class FakeQuantileModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, features):
        return np.tile(self.values, (len(features), 1))


class PriceForecastTests(unittest.TestCase):
    def test_timeframe_and_horizon_mapping(self):
        self.assertEqual(timeframe_to_minutes("5m"), 5)
        self.assertEqual(timeframe_to_minutes("1h"), 60)
        mapping = horizon_bars("5m")
        self.assertEqual(mapping["15 min"], 3)
        self.assertEqual(mapping["1 día"], 288)
        unavailable = horizon_bars("1h")
        self.assertIsNone(unavailable["15 min"])
        self.assertEqual(unavailable["1 día"], 24)

    def test_conformal_adjustment_expands_undercovered_interval(self):
        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        lower = np.full(5, -0.25)
        upper = np.full(5, 0.25)
        adjustment = conformal_adjustment(y, lower, upper, target_coverage=0.8)
        self.assertGreater(adjustment, 0)

    def test_price_ranges_are_ordered_and_probability_is_bounded(self):
        bundle = {
            "15 min": {
                "available": True,
                "model": FakeQuantileModel([-0.02, 0.01, 0.03]),
                "adjustment": 0.005,
                "calibration_residuals": np.array([-0.02, 0.0, 0.02]),
                "metrics": {
                    "coverage": 0.82,
                    "direction_accuracy": 0.56,
                    "direction_baseline": 0.51,
                    "median_abs_error_pct": 0.4,
                    "mean_interval_width_pct": 2.0,
                    "test_samples": 100,
                },
                "target_coverage": 0.8,
            }
        }
        rows = predict_price_ranges(bundle, pd.DataFrame({"x": [1]}), 100.0)
        first = rows[0]
        self.assertLess(first["lower_price"], first["median_price"])
        self.assertLess(first["median_price"], first["upper_price"])
        self.assertTrue(0 <= first["probability_up"] <= 1)
        self.assertEqual(first["coverage"], 0.82)


if __name__ == "__main__":
    unittest.main()
