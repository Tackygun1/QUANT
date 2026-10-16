from __future__ import annotations

import csv
import json
import math
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from quant.models.base import utc_run_slug


@dataclass(frozen=True, slots=True)
class OracleProfitBoundConfig:
  backtest_summary_path: Path
  output_root: Path = Path("reports/oracle_profit_bounds")
  constrained_profiles: tuple["OracleConstraintProfile", ...] = field(default_factory=lambda: DEFAULT_CONSTRAINED_PROFILES)


@dataclass(frozen=True, slots=True)
class OracleConstraintProfile:
  label: str
  min_hold_bars: int = 0
  max_position_change: float = 2.0
  max_trades_per_day: int = 0


@dataclass(frozen=True, slots=True)
class OracleConstraintResult:
  profile: OracleConstraintProfile
  oracle: OraclePathSummary
  model_capture_ratio: float


@dataclass(frozen=True, slots=True)
class OraclePathSummary:
  label: str
  total_net_return: float
  total_turnover: float
  total_transaction_cost: float
  trades: int
  active_periods: int
  max_drawdown: float
  final_equity: float


@dataclass(frozen=True, slots=True)
class OracleProfitBoundSummary:
  status: str
  output_dir: Path
  backtest_summary_path: Path
  backtest_output_dir: Path
  provider: str
  symbol: str
  model_name: str
  periods: int
  evaluation_start_time: str
  evaluation_end_time: str
  fee_bps: float
  slippage_bps: float
  execution_cost_rate: float
  model: OraclePathSummary
  frictionless_oracle: OraclePathSummary
  cost_aware_oracle: OraclePathSummary
  greedy_cost_aware_oracle: OraclePathSummary
  constrained_oracles: tuple[OracleConstraintResult, ...]
  model_capture_of_cost_aware_oracle: float
  model_capture_of_frictionless_oracle: float


def run_oracle_profit_bound(config: OracleProfitBoundConfig) -> OracleProfitBoundSummary:
  backtest_summary = _load_json(config.backtest_summary_path)
  backtest_output_dir = config.backtest_summary_path.parent
  periods_path = backtest_output_dir / "periods.csv"
  period_rows = _load_csv(periods_path)
  execution_cost_rate = (
    float(backtest_summary.get("fee_bps", 0.0))
    + float(backtest_summary.get("slippage_bps", 0.0))
  ) / 10_000.0

  realized_returns = [float(row["realized_return"]) for row in period_rows]
  candle_starts = [str(row["candle_start"]) for row in period_rows]
  output_dir = (
    config.output_root
    / str(backtest_summary["model_name"])
    / str(backtest_summary["provider"])
    / str(backtest_summary["symbol"])
    / utc_run_slug()
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  if not realized_returns:
    summary = OracleProfitBoundSummary(
      status="skipped",
      output_dir=output_dir,
      backtest_summary_path=config.backtest_summary_path,
      backtest_output_dir=backtest_output_dir,
      provider=str(backtest_summary.get("provider", "")),
      symbol=str(backtest_summary.get("symbol", "")),
      model_name=str(backtest_summary.get("model_name", "")),
      periods=0,
      evaluation_start_time="",
      evaluation_end_time="",
      fee_bps=float(backtest_summary.get("fee_bps", 0.0)),
      slippage_bps=float(backtest_summary.get("slippage_bps", 0.0)),
      execution_cost_rate=execution_cost_rate,
      model=_summary_from_backtest(backtest_summary),
      frictionless_oracle=OraclePathSummary("frictionless_oracle", 0.0, 0.0, 0.0, 0, 0, 0.0, 1.0),
      cost_aware_oracle=OraclePathSummary("cost_aware_oracle", 0.0, 0.0, 0.0, 0, 0, 0.0, 1.0),
      greedy_cost_aware_oracle=OraclePathSummary("greedy_cost_aware_oracle", 0.0, 0.0, 0.0, 0, 0, 0.0, 1.0),
      constrained_oracles=(),
      model_capture_of_cost_aware_oracle=0.0,
      model_capture_of_frictionless_oracle=0.0,
    )
    _write_summary(output_dir / "summary.json", summary)
    _write_csv(output_dir / "oracle_periods.csv", [], ORACLE_PERIOD_FIELDS)
    return summary

  frictionless_positions = tuple(_sign(value) for value in realized_returns)
  greedy_positions = tuple(_sign(value) for value in realized_returns)
  cost_aware_positions = compute_cost_aware_oracle_positions(
    realized_returns=tuple(realized_returns),
    execution_cost_rate=execution_cost_rate,
  )

  frictionless_periods, frictionless_summary = simulate_position_path(
    candle_starts=tuple(candle_starts),
    realized_returns=tuple(realized_returns),
    positions=frictionless_positions,
    execution_cost_rate=0.0,
    label="frictionless_oracle",
  )
  greedy_periods, greedy_summary = simulate_position_path(
    candle_starts=tuple(candle_starts),
    realized_returns=tuple(realized_returns),
    positions=greedy_positions,
    execution_cost_rate=execution_cost_rate,
    label="greedy_cost_aware_oracle",
  )
  cost_aware_periods, cost_aware_summary = simulate_position_path(
    candle_starts=tuple(candle_starts),
    realized_returns=tuple(realized_returns),
    positions=cost_aware_positions,
    execution_cost_rate=execution_cost_rate,
    label="cost_aware_oracle",
  )
  constrained_rows: list[dict[str, Any]] = []
  constrained_results: list[OracleConstraintResult] = []

  model_summary = _summary_from_backtest(backtest_summary)
  for profile in config.constrained_profiles:
    constrained_positions = compute_constrained_cost_aware_oracle_positions(
      candle_starts=tuple(candle_starts),
      realized_returns=tuple(realized_returns),
      execution_cost_rate=execution_cost_rate,
      profile=profile,
    )
    profile_rows, profile_summary = simulate_position_path(
      candle_starts=tuple(candle_starts),
      realized_returns=tuple(realized_returns),
      positions=constrained_positions,
      execution_cost_rate=execution_cost_rate,
      label=profile.label,
    )
    constrained_rows.extend(profile_rows)
    constrained_results.append(
      OracleConstraintResult(
        profile=profile,
        oracle=profile_summary,
        model_capture_ratio=_capture_ratio(model_summary.total_net_return, profile_summary.total_net_return),
      )
    )

  summary = OracleProfitBoundSummary(
    status="completed",
    output_dir=output_dir,
    backtest_summary_path=config.backtest_summary_path,
    backtest_output_dir=backtest_output_dir,
    provider=str(backtest_summary["provider"]),
    symbol=str(backtest_summary["symbol"]),
    model_name=str(backtest_summary["model_name"]),
    periods=len(realized_returns),
    evaluation_start_time=candle_starts[0],
    evaluation_end_time=candle_starts[-1],
    fee_bps=float(backtest_summary["fee_bps"]),
    slippage_bps=float(backtest_summary["slippage_bps"]),
    execution_cost_rate=execution_cost_rate,
    model=model_summary,
    frictionless_oracle=frictionless_summary,
    cost_aware_oracle=cost_aware_summary,
    greedy_cost_aware_oracle=greedy_summary,
    constrained_oracles=tuple(constrained_results),
    model_capture_of_cost_aware_oracle=_capture_ratio(
      model_summary.total_net_return,
      cost_aware_summary.total_net_return,
    ),
    model_capture_of_frictionless_oracle=_capture_ratio(
      model_summary.total_net_return,
      frictionless_summary.total_net_return,
    ),
  )
  _write_summary(output_dir / "summary.json", summary)
  _write_csv(
    output_dir / "oracle_periods.csv",
    frictionless_periods + greedy_periods + cost_aware_periods + constrained_rows,
    ORACLE_PERIOD_FIELDS,
  )
  return summary


def compute_cost_aware_oracle_positions(
  *,
  realized_returns: tuple[float, ...],
  execution_cost_rate: float,
) -> tuple[int, ...]:
  states = (-1, 0, 1)
  scores: dict[int, float] = {-1: float("-inf"), 0: 0.0, 1: float("-inf")}
  backpointers: list[dict[int, int | None]] = []

  for realized_return in realized_returns:
    step_pointers: dict[int, int | None] = {}
    next_scores: dict[int, float] = {}
    for state in states:
      best_previous_state: int | None = None
      best_score = float("-inf")
      for previous_state, previous_score in scores.items():
        if not math.isfinite(previous_score):
          continue
        turnover = abs(state - previous_state)
        step_factor = 1.0 + state * realized_return - turnover * execution_cost_rate
        if step_factor <= 0:
          continue
        candidate_score = previous_score + math.log(step_factor)
        if candidate_score > best_score:
          best_score = candidate_score
          best_previous_state = previous_state
      next_scores[state] = best_score
      step_pointers[state] = best_previous_state
    scores = next_scores
    backpointers.append(step_pointers)

  final_state = max(states, key=lambda state: scores[state])
  positions = [final_state]
  for step_index in range(len(backpointers) - 1, 0, -1):
    previous_state = backpointers[step_index][positions[-1]]
    if previous_state is None:
      raise ValueError("Oracle reconstruction failed due to missing backpointer.")
    positions.append(previous_state)
  positions.reverse()
  return tuple(positions)


def compute_constrained_cost_aware_oracle_positions(
  *,
  candle_starts: tuple[str, ...],
  realized_returns: tuple[float, ...],
  execution_cost_rate: float,
  profile: OracleConstraintProfile,
) -> tuple[int, ...]:
  if len(candle_starts) != len(realized_returns):
    raise ValueError("Constrained oracle inputs must have the same length.")
  positions = (-1, 0, 1)
  capped_hold_bars = max(0, profile.min_hold_bars)
  hold_state_cap = max(1, capped_hold_bars)
  capped_trade_count = max(0, profile.max_trades_per_day)
  state_space: list[tuple[int, int, int]] = []
  state_to_index: dict[tuple[int, int, int], int] = {}
  for position in positions:
    hold_values = (0,) if position == 0 else range(1, hold_state_cap + 1)
    for hold_bars in hold_values:
      for trade_count in range(capped_trade_count + 1 or 1):
        state = (position, hold_bars, trade_count)
        state_to_index[state] = len(state_space)
        state_space.append(state)
  start_index = state_to_index[(0, 0, 0)]
  state_count = len(state_space)
  active_scores = [float("-inf")] * state_count
  active_scores[start_index] = 0.0
  backpointers: list[array] = []

  for index, (candle_start, realized_return) in enumerate(zip(candle_starts, realized_returns, strict=True)):
    current_session_date = candle_start[:10]
    previous_session_date = candle_starts[index - 1][:10] if index > 0 else current_session_date
    session_reset = current_session_date != previous_session_date
    next_scores = [float("-inf")] * state_count
    step_pointers = array("h", [-1]) * state_count
    for previous_index, previous_score in enumerate(active_scores):
      if not math.isfinite(previous_score):
        continue
      previous_position, previous_hold_bars, previous_trade_count = state_space[previous_index]
      effective_previous_trade_count = 0 if session_reset else previous_trade_count
      for target_position in positions:
        if not _is_transition_allowed(
          previous_position=previous_position,
          previous_hold_bars=previous_hold_bars,
          target_position=target_position,
          profile=profile,
        ):
          continue
        turnover = abs(target_position - previous_position)
        trade_count = effective_previous_trade_count + (1 if turnover > 0 else 0)
        if capped_trade_count > 0 and trade_count > capped_trade_count:
          continue
        step_factor = 1.0 + target_position * realized_return - turnover * execution_cost_rate
        if step_factor <= 0:
          continue
        hold_bars = _next_hold_bars(
          previous_position=previous_position,
          previous_hold_bars=previous_hold_bars,
          target_position=target_position,
        )
        if target_position != 0:
          hold_bars = min(hold_bars, hold_state_cap)
        next_index = state_to_index[(target_position, hold_bars, trade_count if capped_trade_count > 0 else 0)]
        candidate_score = previous_score + math.log(step_factor)
        if candidate_score > next_scores[next_index]:
          next_scores[next_index] = candidate_score
          step_pointers[next_index] = previous_index
    if not any(math.isfinite(score) for score in next_scores):
      raise ValueError(f"No feasible constrained oracle state at index {index} for profile {profile.label}.")
    active_scores = next_scores
    backpointers.append(step_pointers)

  final_index = max(range(state_count), key=lambda idx: active_scores[idx])
  reconstructed_positions = [state_space[final_index][0]]
  current_index = final_index
  for step_index in range(len(backpointers) - 1, 0, -1):
    previous_index = backpointers[step_index][current_index]
    if previous_index < 0:
      raise ValueError("Constrained oracle reconstruction failed due to missing backpointer.")
    current_index = int(previous_index)
    reconstructed_positions.append(state_space[current_index][0])
  reconstructed_positions.reverse()
  return tuple(reconstructed_positions)


def simulate_position_path(
  *,
  candle_starts: tuple[str, ...],
  realized_returns: tuple[float, ...],
  positions: tuple[int, ...],
  execution_cost_rate: float,
  label: str,
) -> tuple[list[dict[str, Any]], OraclePathSummary]:
  if len(candle_starts) != len(realized_returns) or len(realized_returns) != len(positions):
    raise ValueError("Oracle period inputs must have the same length.")

  rows: list[dict[str, Any]] = []
  equity = 1.0
  peak_equity = 1.0
  max_drawdown = 0.0
  previous_position = 0
  trades = 0
  active_periods = 0
  total_turnover = 0.0
  total_cost = 0.0

  for candle_start, realized_return, position in zip(candle_starts, realized_returns, positions, strict=True):
    turnover = abs(position - previous_position)
    transaction_cost = turnover * execution_cost_rate
    gross_return = position * realized_return
    net_return = gross_return - transaction_cost
    equity *= 1.0 + net_return
    peak_equity = max(peak_equity, equity)
    if peak_equity > 0:
      max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
    if turnover > 0:
      trades += 1
    if position != 0:
      active_periods += 1
    total_turnover += turnover
    total_cost += transaction_cost
    rows.append(
      {
        "label": label,
        "candle_start": candle_start,
        "position": position,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "gross_return": gross_return,
        "net_return": net_return,
        "equity": equity,
      }
    )
    previous_position = position

  summary = OraclePathSummary(
    label=label,
    total_net_return=equity - 1.0,
    total_turnover=total_turnover,
    total_transaction_cost=total_cost,
    trades=trades,
    active_periods=active_periods,
    max_drawdown=max_drawdown,
    final_equity=equity,
  )
  return rows, summary


ORACLE_PERIOD_FIELDS = [
  "label",
  "candle_start",
  "position",
  "turnover",
  "transaction_cost",
  "gross_return",
  "net_return",
  "equity",
]


DEFAULT_CONSTRAINED_PROFILES = (
  OracleConstraintProfile(
    label="constrained_hold2_no_full_reversal",
    min_hold_bars=2,
    max_position_change=1.0,
    max_trades_per_day=0,
  ),
  OracleConstraintProfile(
    label="constrained_hold2_no_full_reversal_max40_trades_per_day",
    min_hold_bars=2,
    max_position_change=1.0,
    max_trades_per_day=40,
  ),
  OracleConstraintProfile(
    label="constrained_hold6_no_full_reversal_max40_trades_per_day",
    min_hold_bars=6,
    max_position_change=1.0,
    max_trades_per_day=40,
  ),
  OracleConstraintProfile(
    label="constrained_hold6_no_full_reversal_max20_trades_per_day",
    min_hold_bars=6,
    max_position_change=1.0,
    max_trades_per_day=20,
  ),
)


def _summary_from_backtest(payload: dict[str, Any]) -> OraclePathSummary:
  return OraclePathSummary(
    label="model",
    total_net_return=float(payload.get("total_net_return", 0.0)),
    total_turnover=float(payload.get("total_turnover", 0.0)),
    total_transaction_cost=float(payload.get("total_transaction_cost", 0.0)),
    trades=int(payload.get("trades", 0)),
    active_periods=int(payload.get("active_periods", 0)),
    max_drawdown=float(payload.get("max_drawdown", 0.0)),
    final_equity=1.0 + float(payload.get("total_net_return", 0.0)),
  )


def _capture_ratio(model_return: float, oracle_return: float) -> float:
  if oracle_return == 0:
    return 0.0
  return model_return / oracle_return


def _load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def _write_summary(path: Path, summary: OracleProfitBoundSummary) -> None:
  payload = {
    "status": summary.status,
    "output_dir": str(summary.output_dir),
    "backtest_summary_path": str(summary.backtest_summary_path),
    "backtest_output_dir": str(summary.backtest_output_dir),
    "provider": summary.provider,
    "symbol": summary.symbol,
    "model_name": summary.model_name,
    "periods": summary.periods,
    "evaluation_start_time": summary.evaluation_start_time,
    "evaluation_end_time": summary.evaluation_end_time,
    "fee_bps": summary.fee_bps,
    "slippage_bps": summary.slippage_bps,
    "execution_cost_rate": summary.execution_cost_rate,
    "model": asdict(summary.model),
    "frictionless_oracle": asdict(summary.frictionless_oracle),
    "cost_aware_oracle": asdict(summary.cost_aware_oracle),
    "greedy_cost_aware_oracle": asdict(summary.greedy_cost_aware_oracle),
    "constrained_oracles": [
      {
        "profile": asdict(result.profile),
        "oracle": asdict(result.oracle),
        "model_capture_ratio": result.model_capture_ratio,
      }
      for result in summary.constrained_oracles
    ],
    "model_capture_of_cost_aware_oracle": summary.model_capture_of_cost_aware_oracle,
    "model_capture_of_frictionless_oracle": summary.model_capture_of_frictionless_oracle,
  }
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})


def _sign(value: float) -> int:
  if value > 0:
    return 1
  if value < 0:
    return -1
  return 0


def _is_transition_allowed(
  *,
  previous_position: int,
  previous_hold_bars: int,
  target_position: int,
  profile: OracleConstraintProfile,
) -> bool:
  if abs(target_position - previous_position) > profile.max_position_change:
    return False
  if previous_position != target_position and previous_position != 0 and previous_hold_bars < max(0, profile.min_hold_bars):
    return False
  return True


def _next_hold_bars(
  *,
  previous_position: int,
  previous_hold_bars: int,
  target_position: int,
) -> int:
  if target_position == 0:
    return 0
  if target_position == previous_position:
    return previous_hold_bars + 1
  return 1
