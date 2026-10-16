from __future__ import annotations

import unittest

from quant.strategy.basic_strategies import basic_strategy_prediction, resolve_strategy_params


class BasicStrategiesTests(unittest.TestCase):
  def test_buy_and_hold_is_constant_positive_signal(self) -> None:
    prediction = basic_strategy_prediction("buy_and_hold_long", {}, {"signal_bps": 20.0})
    self.assertAlmostEqual(prediction, 0.002)

  def test_momentum_uses_five_bar_return(self) -> None:
    row = {"close_return_5_bar": "0.0012"}
    prediction = basic_strategy_prediction("momentum_5_bar", row)
    self.assertGreater(prediction, 0.0)

  def test_mean_reversion_flips_sign_of_one_bar_return(self) -> None:
    row = {"close_return_1_bar": "0.0008"}
    prediction = basic_strategy_prediction("mean_reversion_1_bar", row)
    self.assertLess(prediction, 0.0)

  def test_imbalance_strategies_respect_dead_zone(self) -> None:
    row = {
      "trades_window_imbalance": "0.05",
      "top_window_size_imbalance_mean": "-0.04",
    }
    self.assertEqual(basic_strategy_prediction("trade_imbalance_follow", row), 0.0)
    self.assertEqual(basic_strategy_prediction("order_book_imbalance_follow", row), 0.0)

  def test_vwap_reversion_goes_long_when_price_is_below_vwap(self) -> None:
    row = {
      "close": "100",
      "trades_window_vwap": "100.4",
    }
    prediction = basic_strategy_prediction("vwap_reversion", row)
    self.assertGreater(prediction, 0.0)

  def test_resolve_strategy_params_applies_overrides(self) -> None:
    params = resolve_strategy_params("momentum_5_bar", {"max_signal_bps": 10.0})
    self.assertEqual(params["max_signal_bps"], 10.0)
    self.assertEqual(params["source_column"], "close_return_5_bar")


if __name__ == "__main__":
  unittest.main()
