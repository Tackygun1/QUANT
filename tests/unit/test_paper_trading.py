from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from quant.execution.paper_trading import (
  PaperTradingConfig,
  STACKED_SUPPORTING_ARTIFACT_SPECS,
  align_to_interval,
  create_paper_ledger,
  create_paper_signal,
  load_hybrid_transformer_artifact,
  predict_live_artifact_outputs,
  required_history_bars_for_live_scoring,
  resolve_artifact_search_root,
  resolve_saved_artifact_path,
  resolve_artifact_dir,
  state_file_path,
  write_state,
)
from quant.models.baselines import RidgeReturnRegressor
from quant.models.base import TrainingArtifactMetadata
from quant.risk.sizing import PositionSizingConfig


class PaperTradingTests(unittest.TestCase):
  def test_resolves_latest_artifact_run(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      older = root / "20260413T010000Z"
      newer = root / "20260413T020000Z"
      older.mkdir(parents=True)
      newer.mkdir(parents=True)
      (older / "metadata.json").write_text("{}", encoding="utf-8")
      (newer / "metadata.json").write_text("{}", encoding="utf-8")

      resolved = resolve_artifact_dir(root)

      self.assertEqual(resolved, newer)

  def test_creates_signal_and_state_from_feature_row(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      artifact_dir = root / "artifact"
      artifact_dir.mkdir(parents=True)

      model = RidgeReturnRegressor(alpha=1.0)
      model.fit(
        [
          [100.0, 120.0, 90.0, 110.0, 0.002],
          [110.0, 130.0, 105.0, 125.0, 0.002],
          [125.0, 140.0, 120.0, 135.0, 0.002],
        ],
        [0.0002, 0.0007, 0.0012],
      )
      model.save(artifact_dir / "model.pkl")
      TrainingArtifactMetadata(
        model_name="ridge_return",
        task_type="regression",
        target_column="forward_return_1_bar",
        provider="coinbase",
        symbol="BTC-USD",
        feature_columns=("open", "high", "low", "close", "rolling_return_std_5_bar"),
        train_rows=3,
        validation_rows=1,
      ).write_json(artifact_dir / "metadata.json")

      row = {
        "provider": "coinbase",
        "symbol": "BTC-USD",
        "candle_start": "2026-04-13T03:00:00.000Z",
        "candle_end": "2026-04-13T03:05:00.000Z",
        "open": "136.0",
        "high": "142.0",
        "low": "134.0",
        "close": "141.0",
        "rolling_return_std_5_bar": "0.001",
      }
      metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

      signal_payload, order_payload = create_paper_signal(
        row=row,
        feature_rows=[row],
        metadata=metadata,
        model=RidgeReturnRegressor.load(artifact_dir / "model.pkl"),
        artifact_dir=artifact_dir,
        config=PaperTradingConfig(artifact_dir=artifact_dir, product="BTC-USD"),
        generated_at=datetime(2026, 4, 13, 3, 5, tzinfo=UTC),
        previous_position=0.0,
        previous_state={},
        feature_path=root / "features.csv",
        runtime_root=root / "runtime",
      )

      self.assertEqual(signal_payload["model_name"], "ridge_return")
      self.assertEqual(signal_payload["candle_start"], row["candle_start"])
      self.assertGreater(signal_payload["target_position"], 0.0)
      self.assertEqual(signal_payload["active_position_bars"], 1)
      self.assertIsNotNone(signal_payload["stop_loss_price"])
      self.assertIsNotNone(signal_payload["take_profit_price"])
      self.assertGreater(signal_payload["take_profit_price"], signal_payload["entry_price"])
      self.assertIsNotNone(order_payload)
      self.assertEqual(order_payload["order_side"], "BUY")

      ledger_payload = create_paper_ledger(
        row=row,
        signal_payload=signal_payload,
        previous_state={},
      )
      self.assertLess(ledger_payload["equity"], 1.0)

      state_path = state_file_path(root / "paper", metadata)
      write_state(state_path, signal_payload, ledger_payload)
      state = json.loads(state_path.read_text(encoding="utf-8"))
      self.assertEqual(state["model_name"], "ridge_return")
      self.assertEqual(state["target_position"], signal_payload["target_position"])
      self.assertEqual(state["active_position_bars"], 1)
      self.assertEqual(state["equity"], ledger_payload["equity"])
      self.assertIn("take_profit_price", state)

  def test_creates_meta_signal_with_supporting_predictions(self) -> None:
    row = {
      "provider": "coinbase",
      "symbol": "BTC-USD",
      "candle_start": "2026-04-13T03:20:00.000Z",
      "candle_end": "2026-04-13T03:25:00.000Z",
      "open": "71050.0",
      "close": "71030.0",
      "rolling_return_std_5_bar": "0.001",
    }
    metadata = {
      "model_name": "lightgbm_return_meta",
      "task_type": "regression",
      "target_column": "forward_return_1_bar",
      "provider": "coinbase",
      "symbol": "BTC-USD",
      "feature_columns": (
        "open",
        "close",
        "rolling_return_std_5_bar",
        "oof_ridge_return_prediction",
      ),
    }

    class FakeModel:
      def predict(self, features: list[list[float]]) -> list[float]:
        return [sum(features[0])]

    with mock.patch(
      "quant.execution.paper_trading.resolve_live_feature_values",
      return_value=(
        {
          **row,
          "oof_ridge_return_prediction": "0.002",
        },
        {
          "base_predictions": {"oof_ridge_return_prediction": 0.002},
          "supporting_artifacts": {"oof_ridge_return_prediction": "/tmp/ridge"},
        },
      ),
    ):
      signal_payload, order_payload = create_paper_signal(
        row=row,
        feature_rows=[row],
        metadata=metadata,
        model=FakeModel(),
        artifact_dir=Path("/tmp/meta"),
        config=PaperTradingConfig(artifact_dir=Path("/tmp/meta"), product="BTC-USD"),
        generated_at=datetime(2026, 4, 13, 3, 25, tzinfo=UTC),
        previous_position=0.0,
        previous_state={},
        feature_path=Path("/tmp/features.csv"),
        runtime_root=Path("/tmp/runtime"),
      )

    self.assertAlmostEqual(signal_payload["prediction"], 142080.003)
    self.assertIn("base_predictions", signal_payload)
    self.assertEqual(signal_payload["base_predictions"]["oof_ridge_return_prediction"], 0.002)
    self.assertEqual(order_payload["order_side"], "BUY")

  def test_creates_signal_with_secondary_tradeability_artifact_outputs(self) -> None:
    row = {
      "provider": "coinbase",
      "symbol": "BTC-USD",
      "candle_start": "2026-04-13T03:20:00.000Z",
      "candle_end": "2026-04-13T03:25:00.000Z",
      "close": "71030.0",
      "rolling_return_std_5_bar": "0.001",
    }
    metadata = {
      "model_name": "lightgbm_hybrid_direction_meta",
      "task_type": "classification",
      "target_column": "direction_label_1_bar",
      "provider": "coinbase",
      "symbol": "BTC-USD",
      "feature_columns": (
        "close",
        "rolling_return_std_5_bar",
      ),
      "extra": {
        "primary_prediction_column": "prediction_margin",
      },
    }

    class FakeDirectionModel:
      classes_ = [-1, 0, 1]

      def predict(self, features: list[list[float]]) -> list[int]:
        return [1]

      def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
        return [[0.05, 0.05, 0.9]]

    class FakeTradeabilityModel:
      classes_ = [0, 1]

      def predict(self, features: list[list[float]]) -> list[int]:
        return [0]

      def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
        return [[0.7, 0.3]]

    tradeability_artifact = type(
      "TradeabilityArtifact",
      (),
      {
        "artifact_dir": Path("/tmp/tradeability"),
        "metadata": {
          "model_name": "lightgbm_hybrid_tradeability_meta",
          "task_type": "classification",
          "target_column": "tradeable_label_1_bar",
          "provider": "coinbase",
          "symbol": "BTC-USD",
          "feature_columns": ("close", "rolling_return_std_5_bar"),
        },
        "model": FakeTradeabilityModel(),
      },
    )()

    with mock.patch(
      "quant.execution.paper_trading.resolve_live_feature_values",
      side_effect=[
        (dict(row), {}),
        (dict(row), {}),
      ],
    ):
      signal_payload, order_payload = create_paper_signal(
        row=row,
        feature_rows=[row],
        metadata=metadata,
        model=FakeDirectionModel(),
        artifact_dir=Path("/tmp/meta"),
        config=PaperTradingConfig(
          artifact_dir=Path("/tmp/meta"),
          product="BTC-USD",
          sizing=PositionSizingConfig(
            max_abs_position=1.0,
            max_position_change=1.0,
            classification_fixed_target_position=1.0,
            classification_margin_threshold=0.1,
            tradeability_gate_enabled=True,
            tradeability_gate_threshold=0.55,
          ),
        ),
        generated_at=datetime(2026, 4, 13, 3, 25, tzinfo=UTC),
        previous_position=0.0,
        previous_state={},
        feature_path=Path("/tmp/features.csv"),
        runtime_root=Path("/tmp/runtime"),
        tradeability_artifact=tradeability_artifact,
      )

    self.assertIsNone(order_payload)
    self.assertEqual(signal_payload["target_position"], 0.0)
    self.assertIn("tradeability_gate_entry", signal_payload["constraint_flags"])
    self.assertEqual(signal_payload["tradeability_artifact_dir"], "/tmp/tradeability")
    self.assertEqual(signal_payload["tradeability_prediction_outputs"]["prediction_probability_tradeable"], 0.3)

  def test_preserves_existing_entry_and_trade_plan_while_holding(self) -> None:
    row = {
      "provider": "coinbase",
      "symbol": "BTC-USD",
      "candle_start": "2026-04-13T03:30:00.000Z",
      "candle_end": "2026-04-13T03:35:00.000Z",
      "open": "104.0",
      "high": "106.0",
      "low": "103.0",
      "close": "105.0",
      "rolling_return_std_5_bar": "0.001",
    }
    metadata = {
      "model_name": "ridge_return",
      "task_type": "regression",
      "target_column": "forward_return_1_bar",
      "provider": "coinbase",
      "symbol": "BTC-USD",
      "feature_columns": ("close", "rolling_return_std_5_bar"),
    }

    class FakeModel:
      def predict(self, features: list[list[float]]) -> list[float]:
        return [0.01]

    previous_state = {
      "candle_start": "2026-04-13T03:25:00.000Z",
      "target_position": 1.0,
      "active_position_bars": 3,
      "entry_price": 100.0,
      "max_favorable_return": 0.03,
      "max_adverse_return": -0.01,
      "stop_loss_price": 98.0,
      "take_profit_price": 104.0,
      "take_profit_distance_rate": 0.04,
      "planned_reward_risk_ratio": 2.0,
      "equity": 1.0,
      "peak_equity": 1.0,
      "session_start_equity": 1.0,
      "session_date": "2026-04-13",
    }

    signal_payload, order_payload = create_paper_signal(
      row=row,
      feature_rows=[row],
      metadata=metadata,
      model=FakeModel(),
      artifact_dir=Path("/tmp/artifact"),
      config=PaperTradingConfig(artifact_dir=Path("/tmp/artifact"), product="BTC-USD"),
      generated_at=datetime(2026, 4, 13, 3, 35, tzinfo=UTC),
      previous_position=1.0,
      previous_state=previous_state,
      feature_path=Path("/tmp/features.csv"),
      runtime_root=Path("/tmp/runtime"),
    )

    self.assertIsNone(order_payload)
    self.assertEqual(signal_payload["target_position"], 1.0)
    self.assertEqual(signal_payload["entry_price"], 100.0)
    self.assertEqual(signal_payload["stop_loss_price"], 98.0)
    self.assertEqual(signal_payload["take_profit_price"], 104.0)
    self.assertAlmostEqual(signal_payload["open_return"], 0.05, places=9)
    self.assertGreaterEqual(signal_payload["max_favorable_return"], 0.06)

  def test_predicts_trend_state_probability_outputs(self) -> None:
    class FakeTrendModel:
      classes_ = [-1, 0, 1]

      def predict(self, features: list[list[float]]) -> list[int]:
        return [1]

      def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
        return [[0.2, 0.3, 0.5]]

    outputs = predict_live_artifact_outputs(
      model=FakeTrendModel(),
      metadata={
        "model_name": "histgb_trend_state",
        "task_type": "classification",
        "target_column": "future_trend_state_label_24_bar",
        "feature_columns": ("close", "volume"),
      },
      row={"close": "100", "volume": "10"},
      feature_rows=[],
    )

    self.assertEqual(outputs["prediction"], 1.0)
    self.assertEqual(outputs["prediction_probability_downtrend"], 0.2)
    self.assertEqual(outputs["prediction_probability_chop"], 0.3)
    self.assertEqual(outputs["prediction_probability_uptrend"], 0.5)
    self.assertAlmostEqual(outputs["prediction_trend_state_margin"], 0.3)

  def test_predicts_tradeability_probability_outputs(self) -> None:
    class FakeTradeabilityModel:
      classes_ = [0, 1]

      def predict(self, features: list[list[float]]) -> list[int]:
        return [1]

      def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
        return [[0.35, 0.65]]

    outputs = predict_live_artifact_outputs(
      model=FakeTradeabilityModel(),
      metadata={
        "model_name": "lightgbm_hybrid_tradeability_meta",
        "task_type": "classification",
        "target_column": "tradeable_label_1_bar",
        "feature_columns": ("close", "volume"),
      },
      row={"close": "100", "volume": "10"},
      feature_rows=[],
    )

    self.assertEqual(outputs["prediction"], 1.0)
    self.assertEqual(outputs["prediction_probability_not_tradeable"], 0.35)
    self.assertEqual(outputs["prediction_probability_tradeable"], 0.65)
    self.assertAlmostEqual(outputs["prediction_tradeability_margin"], 0.3)

  def test_probability_outputs_follow_probability_argmax_when_hard_prediction_differs(self) -> None:
    class FakeTrendModel:
      classes_ = [-1, 0, 1]

      def predict(self, features: list[list[float]]) -> list[int]:
        return [-1]

      def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
        return [[0.1, 0.2, 0.7]]

    outputs = predict_live_artifact_outputs(
      model=FakeTrendModel(),
      metadata={
        "model_name": "histgb_trend_state",
        "task_type": "classification",
        "target_column": "future_trend_state_label_24_bar",
        "feature_columns": ("close", "volume"),
      },
      row={"close": "100", "volume": "10"},
      feature_rows=[],
    )

    self.assertEqual(outputs["prediction"], 1.0)
    self.assertAlmostEqual(outputs["prediction_trend_state_margin"], 0.6)

  def test_predicts_triple_barrier_probability_outputs(self) -> None:
    class FakeTripleBarrierModel:
      classes_ = [-1, 0, 1]

      def predict(self, features: list[list[float]]) -> list[int]:
        return [1]

      def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
        return [[0.15, 0.25, 0.6]]

    outputs = predict_live_artifact_outputs(
      model=FakeTripleBarrierModel(),
      metadata={
        "model_name": "histgb_triple_barrier_fast",
        "task_type": "classification",
        "target_column": "triple_barrier_label_6_bar",
        "feature_columns": ("close", "volume"),
      },
      row={"close": "100", "volume": "10"},
      feature_rows=[],
    )

    self.assertEqual(outputs["prediction"], 1.0)
    self.assertEqual(outputs["prediction_probability_stop"], 0.15)
    self.assertEqual(outputs["prediction_probability_no_hit"], 0.25)
    self.assertEqual(outputs["prediction_probability_profit"], 0.6)
    self.assertAlmostEqual(outputs["prediction_triple_barrier_margin"], 0.45)

  def test_registers_fast_and_slow_trend_supporting_artifacts(self) -> None:
    self.assertIn("oof_fast_trend_state_probability_uptrend", STACKED_SUPPORTING_ARTIFACT_SPECS)
    self.assertIn("oof_slow_trend_state_probability_uptrend", STACKED_SUPPORTING_ARTIFACT_SPECS)
    self.assertIn("oof_fast_triple_barrier_probability_profit", STACKED_SUPPORTING_ARTIFACT_SPECS)
    self.assertEqual(
      STACKED_SUPPORTING_ARTIFACT_SPECS["oof_fast_trend_state_probability_uptrend"].model_path_name,
      "histgb_trend_state_fast",
    )
    self.assertEqual(
      STACKED_SUPPORTING_ARTIFACT_SPECS["oof_slow_trend_state_probability_uptrend"].model_path_name,
      "histgb_trend_state",
    )
    self.assertEqual(
      STACKED_SUPPORTING_ARTIFACT_SPECS["oof_fast_triple_barrier_probability_profit"].source_column,
      "prediction_probability_profit",
    )

  def test_marks_take_profit_hit_in_ledger(self) -> None:
    row = {
      "provider": "coinbase",
      "symbol": "BTC-USD",
      "candle_start": "2026-04-13T03:25:00.000Z",
      "candle_end": "2026-04-13T03:30:00.000Z",
      "open": "100.0",
      "high": "102.5",
      "low": "99.5",
      "close": "101.0",
      "close_return_1_bar": "0.01",
    }
    signal_payload = {
      "generated_at": "2026-04-13T03:30:00Z",
      "provider": "coinbase",
      "symbol": "BTC-USD",
      "model_name": "histgb_return",
      "task_type": "regression",
      "candle_start": "2026-04-13T03:25:00.000Z",
      "candle_end": "2026-04-13T03:30:00.000Z",
      "prediction": 0.001,
      "target_position": 0.0,
      "position_delta": -0.4,
      "turnover": 0.4,
      "transaction_cost": 0.0002,
    }
    previous_state = {
      "candle_start": "2026-04-13T03:20:00.000Z",
      "target_position": 0.4,
      "equity": 1.0,
      "peak_equity": 1.0,
      "session_start_equity": 1.0,
      "session_date": "2026-04-13",
      "entry_price": 100.0,
      "stop_loss_price": 99.0,
      "take_profit_price": 102.0,
    }

    ledger_payload = create_paper_ledger(
      row=row,
      signal_payload=signal_payload,
      previous_state=previous_state,
    )

    self.assertEqual(ledger_payload["exit_reason"], "take_profit")
    self.assertTrue(ledger_payload["take_profit_hit"])
    self.assertAlmostEqual(ledger_payload["exit_price"], 102.0)

  def test_resolves_models_root_from_artifact_path(self) -> None:
    artifact_dir = Path("/tmp/data/trade_demo/models/ensembles/lightgbm_return_meta/coinbase/BTC-USD/20260413T030739Z")
    resolved = resolve_artifact_search_root(artifact_dir, None)
    self.assertEqual(resolved, Path("/tmp/data/trade_demo/models"))

  def test_resolves_saved_artifact_path_from_artifact_local_and_repo_relative_reference(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      artifact_dir = root / "experiments" / "overnight" / "coinbase" / "BTC-USD" / "run" / "models" / "evolution" / "lightgbm_hybrid_direction_meta" / "00108"
      artifact_dir.mkdir(parents=True)
      local_hybrid = artifact_dir / "hybrid_transformer"
      local_hybrid.mkdir()
      repo_relative_hybrid = root / "experiments" / "overnight" / "coinbase" / "BTC-USD" / "run" / "models" / "hybrid_transformers" / "quantile_forecaster" / "coinbase" / "BTC-USD" / "20260416T000000Z"
      repo_relative_hybrid.mkdir(parents=True)

      resolved_local = resolve_saved_artifact_path(artifact_dir, "hybrid_transformer")
      resolved_repo_relative = resolve_saved_artifact_path(
        artifact_dir,
        "experiments/overnight/coinbase/BTC-USD/run/models/hybrid_transformers/quantile_forecaster/coinbase/BTC-USD/20260416T000000Z",
      )

      self.assertEqual(resolved_local, local_hybrid.resolve())
      self.assertEqual(resolved_repo_relative, repo_relative_hybrid.resolve())

  def test_loads_hybrid_transformer_from_configured_metadata_path(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      artifact_dir = root / "meta" / "20260608T000000Z"
      artifact_dir.mkdir(parents=True)
      configured_hybrid = root / "support" / "hybrid" / "20260607T195947Z"
      configured_hybrid.mkdir(parents=True)
      (configured_hybrid / "metadata.json").write_text(
        json.dumps(
          {
            "model_name": "hybrid_quantile_transformer",
            "task_type": "regression",
            "target_column": "forward_return_1_bar",
            "provider": "coinbase",
            "symbol": "BTC-USD",
            "feature_columns": ["close"],
            "train_rows": 1,
            "validation_rows": 1,
          }
        ),
        encoding="utf-8",
      )
      (configured_hybrid / "model.pt").write_bytes(b"placeholder")

      sentinel_model = object()
      with mock.patch(
        "quant.execution.paper_trading.HybridQuantileTransformer.load",
        return_value=sentinel_model,
      ):
        loaded = load_hybrid_transformer_artifact(
          artifact_dir=artifact_dir,
          metadata={
            "provider": "coinbase",
            "symbol": "BTC-USD",
            "extra": {
              "hybrid_transformer_artifact_dir": str(configured_hybrid),
            },
          },
          artifact_search_root=None,
        )

      self.assertEqual(loaded.artifact_dir, configured_hybrid.resolve())
      self.assertIs(loaded.model, sentinel_model)

  def test_loads_hybrid_transformer_from_walk_forward_hint_when_configured_path_is_not_deployable(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      artifact_dir = root / "meta" / "20260608T000000Z"
      artifact_dir.mkdir(parents=True)
      walk_forward_hint = (
        root
        / "models"
        / "walk_forward"
        / "hybrid_transformers"
        / "quantile_forecaster"
        / "coinbase"
        / "BTC-USD"
        / "20260607T214628Z"
      )
      walk_forward_hint.mkdir(parents=True)
      deployable_hybrid = (
        root
        / "models"
        / "hybrid_transformers"
        / "quantile_forecaster"
        / "coinbase"
        / "BTC-USD"
        / "20260622T010000Z"
      )
      deployable_hybrid.mkdir(parents=True)
      (deployable_hybrid / "metadata.json").write_text(
        json.dumps(
          {
            "model_name": "hybrid_quantile_transformer",
            "task_type": "regression",
            "target_column": "forward_return_1_bar",
            "provider": "coinbase",
            "symbol": "BTC-USD",
            "feature_columns": ["close"],
            "train_rows": 1,
            "validation_rows": 1,
          }
        ),
        encoding="utf-8",
      )
      (deployable_hybrid / "model.pt").write_bytes(b"placeholder")

      sentinel_model = object()
      with mock.patch(
        "quant.execution.paper_trading.HybridQuantileTransformer.load",
        return_value=sentinel_model,
      ):
        loaded = load_hybrid_transformer_artifact(
          artifact_dir=artifact_dir,
          metadata={
            "provider": "coinbase",
            "symbol": "BTC-USD",
            "extra": {
              "hybrid_transformer_artifact_dir": str(walk_forward_hint),
            },
          },
          artifact_search_root=None,
        )

      self.assertEqual(loaded.artifact_dir, deployable_hybrid)
      self.assertIs(loaded.model, sentinel_model)

  def test_required_history_bars_uses_supporting_transformer_lookback(self) -> None:
    class FakeModel:
      pass

    class FakeTransformer:
      def __init__(self, lookback: int) -> None:
        self.config = type("Config", (), {"lookback": lookback})()

    bars = required_history_bars_for_live_scoring(
      metadata={
        "model_name": "lightgbm_hybrid_direction_meta",
        "feature_columns": ("oof_hybrid_transformer_return_q50_1_bar",),
      },
      model=FakeModel(),
      supporting_artifacts={
        "__hybrid_transformer__": type(
          "Loaded",
          (),
          {"model": FakeTransformer(64)},
        )(),
      },
    )

    self.assertEqual(bars, 64)

  def test_aligns_time_to_interval(self) -> None:
    aligned = align_to_interval(datetime(2026, 4, 13, 3, 8, 59, tzinfo=UTC), "5m")
    self.assertEqual(aligned, datetime(2026, 4, 13, 3, 5, tzinfo=UTC))


if __name__ == "__main__":
  unittest.main()
