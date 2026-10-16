from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from quant.research.overnight_training import (
  EvolutionSearchConfig,
  OvernightDataConfig,
  OvernightTrainingConfig,
  candidate_fitness,
  evaluate_candidate,
  mutate_candidate_params,
  mutate_candidate_execution_params,
  random_candidate_params,
  random_candidate_execution_params,
  resolve_candidate_execution_sizing,
)
from quant.risk.sizing import PositionSizingConfig


class OvernightTrainingCandidateTests(unittest.TestCase):
  def test_random_hybrid_candidate_params_have_nested_spaces(self) -> None:
    params = random_candidate_params("lightgbm_hybrid_return_meta", __import__("random").Random(7))

    self.assertIn("transformer", params)
    self.assertIn("meta", params)
    self.assertIn("lookback", params["transformer"])
    self.assertEqual(params["transformer"]["backbone"], "timemixer")
    self.assertIn("scales", params["transformer"])
    self.assertIn("n_estimators", params["meta"])
    self.assertIn("sample_weight_scheme", params["meta"])

  def test_mutated_hybrid_candidate_params_keep_expected_shape(self) -> None:
    parent = {
      "transformer": {
        "lookback": 32,
        "backbone": "timemixer",
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "scales": (1, 2, 4),
        "mixer_layers": 2,
        "dropout": 0.1,
        "batch_size": 16,
        "epochs": 3,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "device": "cpu",
      },
      "meta": {
        "learning_rate": 0.05,
        "n_estimators": 150,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "sample_weight_scheme": "hmm_regime_target_balance",
        "regime_change_boost": 1.25,
        "regime_change_threshold": 0.35,
        "random_state": 7,
      },
    }

    mutated = mutate_candidate_params(
      model_name="lightgbm_hybrid_return_meta",
      parent_params=parent,
      rng=__import__("random").Random(11),
      mutation_power=0.35,
    )

    self.assertEqual(mutated["transformer"]["device"], "cpu")
    self.assertEqual(mutated["transformer"]["backbone"], "timemixer")
    self.assertEqual(mutated["meta"]["random_state"], 7)
    self.assertIn(mutated["transformer"]["lookback"], (24, 32, 48, 64))
    self.assertIn(tuple(mutated["transformer"]["scales"]), ((1, 2, 4), (1, 2, 4, 8), (1, 3, 6)))
    self.assertIn(mutated["meta"]["num_leaves"], (15, 31, 63))
    self.assertIn(mutated["meta"]["sample_weight_scheme"], ("hmm_regime_balance", "hmm_regime_target_balance"))

  def test_random_tabular_execution_candidate_params_have_expected_keys(self) -> None:
    params = random_candidate_execution_params("histgb_return", __import__("random").Random(5))

    self.assertEqual(
      set(params),
      {
        "hold_signal_threshold_bps",
        "flip_signal_threshold_bps",
        "min_hold_bars",
        "min_rebalance_abs_delta",
      },
    )

  def test_random_hybrid_execution_candidate_params_have_expected_keys(self) -> None:
    params = random_candidate_execution_params("lightgbm_hybrid_return_meta", __import__("random").Random(5))

    self.assertEqual(
      set(params),
      {
        "signal_threshold_bps",
        "signal_scale_bps",
        "hold_signal_threshold_bps",
        "flip_signal_threshold_bps",
        "min_hold_bars",
        "min_rebalance_abs_delta",
      },
    )

  def test_random_hybrid_direction_execution_candidate_params_trade_on_margin(self) -> None:
    params = random_candidate_execution_params("lightgbm_hybrid_direction_meta", __import__("random").Random(5))

    self.assertEqual(
      set(params),
      {
        "classification_margin_threshold",
        "classification_margin_scale",
        "classification_hold_margin_threshold",
        "classification_flip_margin_threshold",
        "class_change_persistence_bars",
        "min_hold_bars",
        "min_rebalance_abs_delta",
      },
    )

  def test_mutated_execution_candidate_params_keep_expected_shape(self) -> None:
    mutated = mutate_candidate_execution_params(
      model_name="histgb_return",
      parent_params={
        "hold_signal_threshold_bps": 2.0,
        "flip_signal_threshold_bps": 10.0,
        "min_hold_bars": 3,
        "min_rebalance_abs_delta": 0.1,
      },
      rng=__import__("random").Random(13),
    )

    self.assertIn(mutated["hold_signal_threshold_bps"], (0.0, 1.0, 2.0, 3.0, 5.0))
    self.assertIn(mutated["flip_signal_threshold_bps"], (0.0, 5.0, 10.0, 15.0))
    self.assertIn(mutated["min_hold_bars"], (0, 1, 3, 5))
    self.assertIn(mutated["min_rebalance_abs_delta"], (0.0, 0.05, 0.1, 0.15, 0.2))

  def test_mutated_hybrid_execution_candidate_params_keep_expected_shape(self) -> None:
    mutated = mutate_candidate_execution_params(
      model_name="lightgbm_hybrid_return_meta",
      parent_params={
        "signal_threshold_bps": 0.5,
        "signal_scale_bps": 5.0,
        "hold_signal_threshold_bps": 1.0,
        "flip_signal_threshold_bps": 5.0,
        "min_hold_bars": 1,
        "min_rebalance_abs_delta": 0.05,
      },
      rng=__import__("random").Random(17),
    )

    self.assertIn(mutated["signal_threshold_bps"], (0.0, 0.25, 0.5, 1.0))
    self.assertIn(mutated["signal_scale_bps"], (2.5, 5.0, 7.5, 10.0))
    self.assertIn(mutated["hold_signal_threshold_bps"], (0.0, 0.5, 1.0, 2.0, 3.0))
    self.assertIn(mutated["flip_signal_threshold_bps"], (0.0, 2.5, 5.0, 10.0))
    self.assertIn(mutated["min_hold_bars"], (0, 1, 2, 3))
    self.assertIn(mutated["min_rebalance_abs_delta"], (0.0, 0.01, 0.025, 0.05, 0.1))

  def test_resolves_candidate_execution_sizing_overrides(self) -> None:
    sizing = resolve_candidate_execution_sizing(
      base_sizing=PositionSizingConfig(signal_threshold_bps=0.5),
      execution_params={
        "signal_scale_bps": 7.5,
        "min_rebalance_abs_delta": 0.1,
        "min_hold_bars": 3,
      },
    )

    self.assertEqual(sizing.signal_threshold_bps, 0.5)
    self.assertEqual(sizing.signal_scale_bps, 7.5)
    self.assertEqual(sizing.min_rebalance_abs_delta, 0.1)
    self.assertEqual(sizing.min_hold_bars, 3)

  def test_hybrid_fitness_penalizes_inactive_candidates(self) -> None:
    active = candidate_fitness(
      model_name="lightgbm_hybrid_return_meta",
      search_config=EvolutionSearchConfig(hybrid_min_trade_count=25, hybrid_inactivity_penalty=0.01),
      backtest_summary=self._backtest_summary(total_net_return=0.002, max_drawdown=0.001, trades=25),
    )
    inactive = candidate_fitness(
      model_name="lightgbm_hybrid_return_meta",
      search_config=EvolutionSearchConfig(hybrid_min_trade_count=25, hybrid_inactivity_penalty=0.01),
      backtest_summary=self._backtest_summary(total_net_return=0.002, max_drawdown=0.001, trades=0),
    )

    self.assertGreater(active, inactive)

  @staticmethod
  def _backtest_summary(*, total_net_return: float, max_drawdown: float, trades: int):
    class _Summary:
      def __init__(self) -> None:
        self.total_net_return = total_net_return
        self.max_drawdown = max_drawdown
        self.trades = trades
        self.active_periods = 0
        self.joined_rows = 0
        self.sharpe_like = 0.0
        self.status = "completed"
        self.output_dir = None

    return _Summary()


@unittest.skipUnless(
  importlib.util.find_spec("torch") and importlib.util.find_spec("lightgbm"),
  "torch and lightgbm are required for overnight hybrid smoke tests",
)
class OvernightHybridSmokeTests(unittest.TestCase):
  def test_evaluates_hybrid_candidate_end_to_end(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      config = OvernightTrainingConfig(
        workspace_root=root / "workspace",
        data=OvernightDataConfig(provider="coinbase", product="BTC-USD"),
        search=EvolutionSearchConfig(
          minimum_rows=8,
          validation_fraction=0.25,
          hybrid_recent_rows=48,
          hybrid_horizons=(1, 3),
          hybrid_quantiles=(0.1, 0.5, 0.9),
          hybrid_meta_validation_fraction=0.25,
          hybrid_initial_train_rows=12,
          hybrid_validation_rows=6,
          hybrid_step_rows=6,
          hybrid_embargo_rows=1,
          hybrid_purge_rows=1,
          model_cycle=("lightgbm_hybrid_return_meta",),
        ),
      )

      candidate = {
        "candidate_id": "00001_test",
        "model_name": "lightgbm_hybrid_return_meta",
        "params": {
          "transformer": {
            "lookback": 4,
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "device": "cpu",
          },
          "meta": {
            "learning_rate": 0.05,
            "n_estimators": 50,
            "num_leaves": 15,
            "min_child_samples": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.001,
            "reg_lambda": 0.001,
            "random_state": 7,
          },
        },
        "execution_params": {
          "hold_signal_threshold_bps": 2.0,
          "flip_signal_threshold_bps": 10.0,
          "min_hold_bars": 3,
          "min_rebalance_abs_delta": 0.1,
        },
        "origin": "test",
        "parent_id": "",
        "started_at": "2026-04-13T00:00:00Z",
      }

      result = evaluate_candidate(
        config=config,
        rows=self._build_rows(56),
        candidate=candidate,
      )

      artifact_dir = (
        config.workspace_root
        / "models"
        / "evolution"
        / "lightgbm_hybrid_return_meta"
        / "00001_test"
      )
      self.assertEqual(result["status"], "completed")
      self.assertTrue((artifact_dir / "model.pkl").exists())
      self.assertTrue((artifact_dir / "hybrid_transformer" / "model.pt").exists())
      self.assertTrue((artifact_dir / "validation_predictions.csv").exists())
      self.assertTrue((artifact_dir / "pipeline" / "features" / "training_datasets" / "market_hybrid_stacked").exists())
      self.assertIn("backtest", result)
      self.assertEqual(result["execution_params"]["min_rebalance_abs_delta"], 0.1)
      self.assertEqual(result["execution_sizing"]["min_hold_bars"], 3)

      metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
      self.assertEqual(metadata["extra"]["execution_params"]["min_rebalance_abs_delta"], 0.1)
      self.assertEqual(metadata["extra"]["execution_sizing"]["min_hold_bars"], 3)

  def test_evaluates_hybrid_direction_candidate_end_to_end(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      config = OvernightTrainingConfig(
        workspace_root=root / "workspace",
        data=OvernightDataConfig(provider="coinbase", product="BTC-USD"),
        search=EvolutionSearchConfig(
          minimum_rows=8,
          validation_fraction=0.25,
          hybrid_recent_rows=48,
          hybrid_horizons=(1, 3),
          hybrid_quantiles=(0.1, 0.5, 0.9),
          hybrid_meta_validation_fraction=0.25,
          hybrid_initial_train_rows=12,
          hybrid_validation_rows=6,
          hybrid_step_rows=6,
          hybrid_embargo_rows=1,
          hybrid_purge_rows=1,
          model_cycle=("lightgbm_hybrid_direction_meta",),
        ),
      )

      candidate = {
        "candidate_id": "00002_test",
        "model_name": "lightgbm_hybrid_direction_meta",
        "params": {
          "transformer": {
            "lookback": 4,
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "device": "cpu",
          },
          "meta": {
            "learning_rate": 0.05,
            "n_estimators": 50,
            "num_leaves": 15,
            "min_child_samples": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.001,
            "reg_lambda": 0.001,
            "random_state": 7,
          },
        },
        "execution_params": {
          "signal_threshold_bps": 0.25,
          "signal_scale_bps": 5.0,
          "hold_signal_threshold_bps": 1.0,
          "flip_signal_threshold_bps": 5.0,
          "min_hold_bars": 1,
          "min_rebalance_abs_delta": 0.05,
        },
        "origin": "test",
        "parent_id": "",
        "started_at": "2026-04-13T00:00:00Z",
      }

      result = evaluate_candidate(
        config=config,
        rows=self._build_rows(56),
        candidate=candidate,
      )

      artifact_dir = (
        config.workspace_root
        / "models"
        / "evolution"
        / "lightgbm_hybrid_direction_meta"
        / "00002_test"
      )
      self.assertEqual(result["status"], "completed")
      self.assertTrue((artifact_dir / "model.pkl").exists())
      self.assertIn("backtest", result)
      self.assertEqual(result["task_type"], "classification")
      self.assertEqual(result["execution_sizing"]["signal_threshold_bps"], 0.25)

  @staticmethod
  def _build_rows(length: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(length):
      minute = index * 5
      hour = minute // 60
      minute_of_hour = minute % 60
      close = 100.0 + index * 0.25 + math.sin(index / 4.0)
      rows.append(
        {
          "provider": "coinbase",
          "symbol": "BTC-USD",
          "feature_date": "2026-04-12",
          "interval": "5m",
          "candle_start": f"2026-04-12T{hour:02d}:{minute_of_hour:02d}:00.000Z",
          "candle_end": f"2026-04-12T{hour:02d}:{(minute_of_hour + 5) % 60:02d}:00.000Z",
          "is_closed": "true",
          "open": str(close - 0.2),
          "high": str(close + 0.4),
          "low": str(close - 0.6),
          "close": str(close),
          "volume": str(5.0 + (index % 7)),
          "trades_window_count": str(50 + index),
          "trade_imbalance": str(math.sin(index / 5.0)),
          "rolling_return_std_5_bar": str(0.0008 + (index % 4) * 0.0001),
          "forward_return_1_bar": str(0.0012 * math.sin(index / 6.0)),
          "forward_return_3_bar": str(0.0018 * math.cos(index / 7.0)),
          "direction_label_1_bar": str(1 if index % 3 == 0 else (-1 if index % 3 == 1 else 0)),
          "tradeable_label_1_bar": str(1 if index % 4 in (0, 1) else 0),
          "has_any_forward_label": "true",
          "has_1_bar_label": "true",
        }
      )
    return rows


if __name__ == "__main__":
  unittest.main()
