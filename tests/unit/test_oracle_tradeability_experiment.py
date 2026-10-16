from __future__ import annotations

from quant.research.oracle_profit_bound import OracleConstraintProfile
from quant.research.oracle_tradeability_experiment import (
  load_tradeability_prediction_lookup_from_csv,
  relabel_tradeability_rows,
)


def test_relabel_tradeability_rows_replaces_target_with_oracle_activity() -> None:
  rows = [
    {
      "feature_date": "2026-01-01",
      "candle_start": "2026-01-01T00:00:00Z",
      "tradeable_label_1_bar": "0",
      "forward_return_1_bar": "0.03",
    },
    {
      "feature_date": "2026-01-01",
      "candle_start": "2026-01-01T00:05:00Z",
      "tradeable_label_1_bar": "0",
      "forward_return_1_bar": "-0.01",
    },
    {
      "feature_date": "2026-01-01",
      "candle_start": "2026-01-01T00:10:00Z",
      "tradeable_label_1_bar": "0",
      "forward_return_1_bar": "-0.01",
    },
  ]
  output_rows, stats = relabel_tradeability_rows(
    rows=rows,
    oracle_profile=OracleConstraintProfile(
      label="hold2",
      min_hold_bars=2,
      max_position_change=2.0,
      max_trades_per_day=0,
    ),
    execution_cost_rate=0.001,
    label_mode="active_position",
  )
  assert [row["tradeable_label_1_bar"] for row in output_rows] == ["1", "1", "1"]
  assert [row["oracle_position_label_1_bar"] for row in output_rows] == ["1", "1", "-1"]
  assert [row["oracle_tradeability_active_label_1_bar"] for row in output_rows] == ["1", "1", "1"]
  assert [row["oracle_tradeability_entry_label_1_bar"] for row in output_rows] == ["1", "0", "1"]
  assert output_rows[0]["legacy_tradeable_label_1_bar"] == "0"
  assert stats["positive_labels"] == 3


def test_relabel_tradeability_rows_supports_entry_action_labels() -> None:
  rows = [
    {
      "feature_date": "2026-01-01",
      "candle_start": "2026-01-01T00:00:00Z",
      "tradeable_label_1_bar": "0",
      "forward_return_1_bar": "0.03",
    },
    {
      "feature_date": "2026-01-01",
      "candle_start": "2026-01-01T00:05:00Z",
      "tradeable_label_1_bar": "0",
      "forward_return_1_bar": "-0.01",
    },
    {
      "feature_date": "2026-01-01",
      "candle_start": "2026-01-01T00:10:00Z",
      "tradeable_label_1_bar": "0",
      "forward_return_1_bar": "-0.01",
    },
  ]
  output_rows, stats = relabel_tradeability_rows(
    rows=rows,
    oracle_profile=OracleConstraintProfile(
      label="hold2",
      min_hold_bars=2,
      max_position_change=2.0,
      max_trades_per_day=0,
    ),
    execution_cost_rate=0.001,
    label_mode="entry_action",
  )
  assert [row["tradeable_label_1_bar"] for row in output_rows] == ["1", "0", "1"]
  assert [row["oracle_tradeability_active_label_1_bar"] for row in output_rows] == ["1", "1", "1"]
  assert [row["oracle_tradeability_entry_label_1_bar"] for row in output_rows] == ["1", "0", "1"]
  assert stats["positive_labels"] == 2
  assert stats["active_periods"] == 3
  assert stats["trades"] == 2


def test_load_tradeability_prediction_lookup_from_csv(tmp_path) -> None:
  path = tmp_path / "predictions.csv"
  path.write_text(
    "provider,symbol,candle_start,target,prediction,prediction_probability_tradeable,prediction_confidence\n"
    "coinbase,BTC-USD,2026-01-01T00:00:00Z,1,1,0.8,0.4\n",
    encoding="utf-8",
  )
  lookup = load_tradeability_prediction_lookup_from_csv(path)
  assert lookup["2026-01-01T00:00:00Z"]["prediction_probability_tradeable"] == "0.8"
