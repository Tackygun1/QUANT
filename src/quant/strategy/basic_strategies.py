from __future__ import annotations

from typing import Any


BASIC_STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
  "buy_and_hold_long": {
    "description": "Always long with a constant bullish signal.",
    "default_params": {
      "signal_bps": 25.0,
    },
    "builder": "_buy_and_hold_long_prediction",
  },
  "momentum_5_bar": {
    "description": "Follow the sign of the recent 5-bar return once it clears a dead zone.",
    "default_params": {
      "source_column": "close_return_5_bar",
      "activation_threshold_bps": 3.0,
      "prediction_scale": 1.25,
      "max_signal_bps": 25.0,
    },
    "builder": "_scaled_column_prediction",
  },
  "mean_reversion_1_bar": {
    "description": "Fade the most recent 1-bar return once it stretches beyond a dead zone.",
    "default_params": {
      "source_column": "close_return_1_bar",
      "activation_threshold_bps": 2.0,
      "prediction_scale": -1.5,
      "max_signal_bps": 25.0,
    },
    "builder": "_scaled_column_prediction",
  },
  "trade_imbalance_follow": {
    "description": "Follow trade-side imbalance when one side dominates the candle.",
    "default_params": {
      "source_column": "trades_window_imbalance",
      "activation_threshold": 0.1,
      "signal_scale_bps": 15.0,
      "max_signal_bps": 25.0,
    },
    "builder": "_imbalance_follow_prediction",
  },
  "order_book_imbalance_follow": {
    "description": "Follow top-of-book size imbalance when the book leans strongly to one side.",
    "default_params": {
      "source_column": "top_window_size_imbalance_mean",
      "activation_threshold": 0.08,
      "signal_scale_bps": 12.0,
      "max_signal_bps": 25.0,
    },
    "builder": "_imbalance_follow_prediction",
  },
  "vwap_reversion": {
    "description": "Fade price moves away from intrabar VWAP.",
    "default_params": {
      "price_column": "close",
      "vwap_column": "trades_window_vwap",
      "activation_threshold_bps": 2.0,
      "prediction_scale": 1.0,
      "max_signal_bps": 25.0,
    },
    "builder": "_vwap_reversion_prediction",
  },
}


def available_basic_strategies() -> tuple[str, ...]:
  return tuple(BASIC_STRATEGY_REGISTRY)


def resolve_strategy_params(
  strategy_name: str,
  params: dict[str, Any] | None = None,
) -> dict[str, Any]:
  spec = _strategy_spec(strategy_name)
  resolved = dict(spec["default_params"])
  if params:
    resolved.update(params)
  return resolved


def basic_strategy_prediction(
  strategy_name: str,
  row: dict[str, str],
  params: dict[str, Any] | None = None,
) -> float:
  spec = _strategy_spec(strategy_name)
  resolved_params = resolve_strategy_params(strategy_name, params)
  builder_name = spec["builder"]
  return globals()[builder_name](row, resolved_params)


def _strategy_spec(strategy_name: str) -> dict[str, Any]:
  if strategy_name not in BASIC_STRATEGY_REGISTRY:
    raise ValueError(f"Unknown basic strategy: {strategy_name}")
  return BASIC_STRATEGY_REGISTRY[strategy_name]


def _buy_and_hold_long_prediction(_row: dict[str, str], params: dict[str, Any]) -> float:
  return float(params["signal_bps"]) / 10_000


def _scaled_column_prediction(row: dict[str, str], params: dict[str, Any]) -> float:
  source_value = _row_float(row, str(params["source_column"]))
  threshold_rate = float(params["activation_threshold_bps"]) / 10_000
  if abs(source_value) < threshold_rate:
    return 0.0
  return _clip_signal(
    source_value * float(params["prediction_scale"]),
    max_signal_bps=float(params["max_signal_bps"]),
  )


def _imbalance_follow_prediction(row: dict[str, str], params: dict[str, Any]) -> float:
  source_value = _row_float(row, str(params["source_column"]))
  if abs(source_value) < float(params["activation_threshold"]):
    return 0.0
  signal_rate = source_value * float(params["signal_scale_bps"]) / 10_000
  return _clip_signal(signal_rate, max_signal_bps=float(params["max_signal_bps"]))


def _vwap_reversion_prediction(row: dict[str, str], params: dict[str, Any]) -> float:
  price = _row_float(row, str(params["price_column"]))
  vwap = _row_float(row, str(params["vwap_column"]))
  if price <= 0 or vwap <= 0:
    return 0.0
  deviation = (vwap - price) / price
  threshold_rate = float(params["activation_threshold_bps"]) / 10_000
  if abs(deviation) < threshold_rate:
    return 0.0
  return _clip_signal(
    deviation * float(params["prediction_scale"]),
    max_signal_bps=float(params["max_signal_bps"]),
  )


def _clip_signal(value: float, *, max_signal_bps: float) -> float:
  max_signal_rate = max_signal_bps / 10_000
  return max(-max_signal_rate, min(max_signal_rate, value))


def _row_float(row: dict[str, str], column: str) -> float:
  value = row.get(column, "")
  if value == "":
    return 0.0
  return float(value)
