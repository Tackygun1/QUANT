from __future__ import annotations

import math

from quant.research.oracle_profit_bound import (
  OracleConstraintProfile,
  compute_constrained_cost_aware_oracle_positions,
  compute_cost_aware_oracle_positions,
  simulate_position_path,
)


def test_cost_aware_oracle_flips_when_edge_exceeds_cost() -> None:
  positions = compute_cost_aware_oracle_positions(
    realized_returns=(0.01, -0.01),
    execution_cost_rate=0.001,
  )
  assert positions == (1, -1)


def test_cost_aware_oracle_stays_flat_when_flip_is_not_worth_cost() -> None:
  positions = compute_cost_aware_oracle_positions(
    realized_returns=(0.0004, -0.0004),
    execution_cost_rate=0.001,
  )
  assert positions == (0, 0)


def test_simulate_position_path_compounds_net_returns_and_costs() -> None:
  rows, summary = simulate_position_path(
    candle_starts=("2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"),
    realized_returns=(0.01, 0.02),
    positions=(1, 1),
    execution_cost_rate=0.001,
    label="oracle",
  )

  expected_first = 1.0 * (1.0 + 0.01 - 0.001)
  expected_final = expected_first * (1.0 + 0.02)
  assert len(rows) == 2
  assert math.isclose(summary.final_equity, expected_final)
  assert math.isclose(summary.total_net_return, expected_final - 1.0)
  assert summary.trades == 1
  assert summary.active_periods == 2
  assert math.isclose(summary.total_transaction_cost, 0.001)


def test_constrained_oracle_blocks_direct_full_reversal_when_max_position_change_is_one() -> None:
  positions = compute_constrained_cost_aware_oracle_positions(
    candle_starts=(
      "2026-01-01T00:00:00Z",
      "2026-01-01T00:05:00Z",
      "2026-01-01T00:10:00Z",
    ),
    realized_returns=(0.10, -0.05, -0.05),
    execution_cost_rate=0.001,
    profile=OracleConstraintProfile(
      label="constrained",
      max_position_change=1.0,
    ),
  )
  assert positions == (1, 0, -1)


def test_constrained_oracle_respects_min_hold_bars() -> None:
  positions = compute_constrained_cost_aware_oracle_positions(
    candle_starts=(
      "2026-01-01T00:00:00Z",
      "2026-01-01T00:05:00Z",
      "2026-01-01T00:10:00Z",
    ),
    realized_returns=(0.03, -0.01, -0.01),
    execution_cost_rate=0.001,
    profile=OracleConstraintProfile(
      label="constrained",
      min_hold_bars=2,
      max_position_change=2.0,
    ),
  )
  assert positions == (1, 1, -1)


def test_constrained_oracle_respects_max_trades_per_day() -> None:
  positions = compute_constrained_cost_aware_oracle_positions(
    candle_starts=(
      "2026-01-01T00:00:00Z",
      "2026-01-01T00:05:00Z",
      "2026-01-01T00:10:00Z",
    ),
    realized_returns=(0.01, -0.05, 0.03),
    execution_cost_rate=0.001,
    profile=OracleConstraintProfile(
      label="constrained",
      max_trades_per_day=1,
      max_position_change=2.0,
    ),
  )
  assert positions == (0, 0, 1)


def test_constrained_oracle_resets_trade_budget_on_new_day() -> None:
  positions = compute_constrained_cost_aware_oracle_positions(
    candle_starts=(
      "2026-01-01T23:50:00Z",
      "2026-01-01T23:55:00Z",
      "2026-01-02T00:00:00Z",
    ),
    realized_returns=(-0.05, 0.01, 0.03),
    execution_cost_rate=0.001,
    profile=OracleConstraintProfile(
      label="constrained",
      max_trades_per_day=1,
      max_position_change=2.0,
    ),
  )
  assert positions == (-1, -1, 1)
