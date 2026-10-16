from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class PositionSizingConfig:
  volatility_column: str = "rolling_return_std_5_bar"
  volatility_target: float = 0.0015
  volatility_floor: float = 0.0005
  max_abs_position: float = 1.0
  signal_threshold_bps: float = 2.0
  signal_scale_bps: float = 10.0
  max_trade_risk_fraction: float = 0.01
  stop_loss_sigma: float = 2.0
  stop_loss_floor_bps: float = 100.0
  max_position_change: float = 0.5
  max_daily_loss_fraction: float = 0.02
  max_portfolio_drawdown_fraction: float = 0.1
  hold_signal_threshold_bps: float = 0.0
  flip_signal_threshold_bps: float = 0.0
  classification_margin_threshold: float = 0.15
  classification_margin_scale: float = 0.35
  classification_fixed_target_position: float = 0.0
  classification_hold_margin_threshold: float = 0.1
  classification_flip_margin_threshold: float = 0.25
  classification_stale_exit_margin_threshold: float = 0.0
  tradeability_gate_enabled: bool = False
  tradeability_gate_probability_column: str = "tradeability_prediction_probability_tradeable"
  tradeability_gate_confidence_column: str = "tradeability_prediction_confidence"
  tradeability_gate_threshold: float = 0.55
  tradeability_gate_min_confidence: float = 0.0
  forecast_gate_enabled: bool = False
  forecast_gate_fast_column: str = "oof_hybrid_transformer_return_q50_6_bar"
  forecast_gate_slow_column: str = "oof_hybrid_transformer_return_q50_24_bar"
  forecast_gate_fast_threshold_bps: float = 3.0
  forecast_gate_slow_threshold_bps: float = 5.0
  forecast_gate_require_alignment: bool = True
  regime_chop_gate_enabled: bool = False
  regime_chop_probability_column: str = "regime_hmm_p_chop"
  regime_chop_threshold: float = 0.55
  regime_chop_entry_threshold_scale: float = 1.5
  regime_chop_flip_threshold_scale: float = 1.5
  regime_chop_max_position_scale: float = 0.5
  target_position_smoothing: float = 0.0
  hold_flat_until_time_stop: bool = False
  class_change_persistence_bars: int = 0
  min_hold_bars: int = 0
  max_hold_bars: int = 0
  stale_exit_bars: int = 0
  stale_exit_signal_threshold_bps: float = 0.0
  hold_until_stop_or_take_profit: bool = False
  profit_take_exit_multiple: float = 0.0
  trailing_profit_activation_multiple: float = 0.0
  trailing_profit_retrace_multiple: float = 0.0
  min_rebalance_abs_delta: float = 0.0
  trend_filter_enabled: bool = False
  trend_filter_column: str = "trend_return_24_bar"
  trend_filter_higher_timeframe_column: str = "trend_return_96_bar"
  trend_filter_threshold_bps: float = 20.0
  trend_filter_countertrend_scale: float = 0.0
  trend_filter_require_alignment: bool = True
  trend_filter_regime_bias_bps: float = 0.0
  trend_filter_exit_enabled: bool = False
  trend_state_filter_enabled: bool = False
  trend_state_uptrend_probability_column: str = "oof_trend_state_probability_uptrend"
  trend_state_downtrend_probability_column: str = "oof_trend_state_probability_downtrend"
  trend_state_chop_probability_column: str = "oof_trend_state_probability_chop"
  trend_state_confidence_column: str = "oof_trend_state_confidence"
  trend_state_gate_threshold: float = 0.55
  trend_state_countertrend_scale: float = 0.0
  trend_state_chop_threshold: float = 0.55
  trend_state_chop_scale: float = 0.0
  trend_state_min_confidence: float = 0.0
  trend_state_regime_bias_bps: float = 0.0
  trend_state_exit_enabled: bool = False
  trend_state_exit_gate_threshold: float = 0.7
  trend_state_chop_exit_threshold: float = 0.8


@dataclass(frozen=True, slots=True)
class RiskState:
  equity: float = 1.0
  peak_equity: float = 1.0
  session_start_equity: float = 1.0


@dataclass(frozen=True, slots=True)
class PositionState:
  entry_price: float | None = None
  mark_price: float | None = None
  open_return: float = 0.0
  max_favorable_return: float = 0.0
  max_adverse_return: float = 0.0


@dataclass(frozen=True, slots=True)
class PositionDecision:
  signal: float
  direction: int
  confidence: float
  volatility: float
  volatility_scale: float
  target_position: float
  raw_target_position: float = 0.0
  stop_distance_rate: float = 0.0
  max_position_by_risk: float = 0.0
  session_drawdown: float = 0.0
  portfolio_drawdown: float = 0.0
  pending_direction: int = 0
  pending_direction_bars: int = 0
  constraint_flags: tuple[str, ...] = ()


def target_position_from_prediction(
  *,
  task_type: str,
  prediction: float,
  row: dict[str, str],
  sizing: PositionSizingConfig,
  execution_cost_rate: float,
  previous_position: float = 0.0,
  active_position_bars: int = 0,
  pending_direction: int = 0,
  pending_direction_bars: int = 0,
  risk_state: RiskState | None = None,
  position_state: PositionState | None = None,
) -> PositionDecision:
  prediction, overlay_flags = apply_prediction_overlays(
    prediction=prediction,
    row=row,
    sizing=sizing,
  )
  volatility = max(_row_float(row, sizing.volatility_column, sizing.volatility_floor), sizing.volatility_floor)
  volatility_scale = min(sizing.max_abs_position, sizing.volatility_target / volatility)

  if task_type == "regression":
    decision = _regression_decision(
      prediction=prediction,
      volatility=volatility,
      volatility_scale=volatility_scale,
      sizing=sizing,
      execution_cost_rate=execution_cost_rate,
      constraint_flags=overlay_flags,
    )
  else:
    decision = _classification_decision(
      prediction=prediction,
      volatility=volatility,
      volatility_scale=volatility_scale,
      sizing=sizing,
      constraint_flags=overlay_flags,
    )
  decision = replace(
    decision,
    stop_distance_rate=_stop_distance_rate(volatility=volatility, sizing=sizing),
  )
  decision = apply_position_persistence_overlays(
    decision=decision,
    task_type=task_type,
    prediction=prediction,
    row=row,
    previous_position=previous_position,
    active_position_bars=active_position_bars,
    pending_direction=pending_direction,
    pending_direction_bars=pending_direction_bars,
    sizing=sizing,
    execution_cost_rate=execution_cost_rate,
    position_state=position_state,
  )
  return apply_position_constraints(
    decision=decision,
    sizing=sizing,
    previous_position=previous_position,
    risk_state=risk_state or RiskState(),
  )


def _regression_decision(
  *,
  prediction: float,
  volatility: float,
  volatility_scale: float,
  sizing: PositionSizingConfig,
  execution_cost_rate: float,
  constraint_flags: tuple[str, ...] = (),
) -> PositionDecision:
  threshold_rate = sizing.signal_threshold_bps / 10_000
  scale_rate = max(sizing.signal_scale_bps / 10_000, 1e-9)
  required_edge = threshold_rate + execution_cost_rate
  signal = abs(prediction)
  direction = _sign(prediction)

  if direction == 0 or signal <= required_edge:
    return PositionDecision(
      signal=signal,
      direction=0,
      confidence=0.0,
      volatility=volatility,
      volatility_scale=volatility_scale,
      target_position=0.0,
      raw_target_position=0.0,
      constraint_flags=constraint_flags,
    )

  confidence = min(1.0, (signal - required_edge) / scale_rate)
  raw_target_position = direction * min(sizing.max_abs_position, confidence * volatility_scale)
  return PositionDecision(
    signal=signal,
    direction=direction,
    confidence=confidence,
    volatility=volatility,
    volatility_scale=volatility_scale,
    target_position=raw_target_position,
    raw_target_position=raw_target_position,
    constraint_flags=constraint_flags,
  )


def _classification_decision(
  *,
  prediction: float,
  volatility: float,
  volatility_scale: float,
  sizing: PositionSizingConfig,
  constraint_flags: tuple[str, ...] = (),
) -> PositionDecision:
  direction = _sign(prediction)
  signal = min(1.0, abs(float(prediction)))
  required_edge = max(0.0, sizing.classification_margin_threshold)
  scale = max(1e-9, sizing.classification_margin_scale)
  if direction == 0 or signal <= required_edge:
    return PositionDecision(
      signal=signal,
      direction=0,
      confidence=0.0,
      volatility=volatility,
      volatility_scale=volatility_scale,
      target_position=0.0,
      raw_target_position=0.0,
      constraint_flags=constraint_flags,
    )

  confidence = min(1.0, (signal - required_edge) / scale)
  fixed_target_position = min(
    sizing.max_abs_position,
    max(0.0, sizing.classification_fixed_target_position),
  )
  if fixed_target_position > 0:
    raw_target_position = direction * fixed_target_position
  else:
    raw_target_position = direction * min(sizing.max_abs_position, confidence * volatility_scale)
  return PositionDecision(
    signal=signal,
    direction=direction,
    confidence=confidence,
    volatility=volatility,
    volatility_scale=volatility_scale,
    target_position=raw_target_position,
    raw_target_position=raw_target_position,
    constraint_flags=constraint_flags,
  )


def apply_position_constraints(
  *,
  decision: PositionDecision,
  sizing: PositionSizingConfig,
  previous_position: float,
  risk_state: RiskState,
) -> PositionDecision:
  target_position = float(decision.target_position)
  constraint_flags: list[str] = list(decision.constraint_flags)
  stop_distance_rate = max(decision.stop_distance_rate, _stop_distance_rate(volatility=decision.volatility, sizing=sizing))
  max_position_by_risk = sizing.max_abs_position
  if sizing.max_trade_risk_fraction > 0:
    max_position_by_risk = min(
      sizing.max_abs_position,
      sizing.max_trade_risk_fraction / stop_distance_rate,
    )
    if abs(target_position) > max_position_by_risk:
      target_position = _sign(target_position) * max_position_by_risk
      constraint_flags.append("risk_budget_cap")

  max_position_change = max(0.0, sizing.max_position_change)
  if max_position_change > 0:
    lower_bound = previous_position - max_position_change
    upper_bound = previous_position + max_position_change
    adjusted_target = min(upper_bound, max(lower_bound, target_position))
    if adjusted_target != target_position:
      target_position = adjusted_target
      constraint_flags.append("turnover_cap")

  session_drawdown = _drawdown_fraction(
    peak_equity=risk_state.session_start_equity,
    equity=risk_state.equity,
  )
  portfolio_drawdown = _drawdown_fraction(
    peak_equity=risk_state.peak_equity,
    equity=risk_state.equity,
  )
  if (
    sizing.max_daily_loss_fraction > 0
    and session_drawdown >= sizing.max_daily_loss_fraction
  ):
    target_position = 0.0
    constraint_flags.append("daily_loss_halt")

  if (
    sizing.max_portfolio_drawdown_fraction > 0
    and portfolio_drawdown >= sizing.max_portfolio_drawdown_fraction
  ):
    target_position = 0.0
    constraint_flags.append("drawdown_halt")

  target_position = min(sizing.max_abs_position, max(-sizing.max_abs_position, target_position))
  return PositionDecision(
    signal=decision.signal,
    direction=_sign(target_position),
    confidence=decision.confidence if target_position != 0 else 0.0,
    volatility=decision.volatility,
    volatility_scale=decision.volatility_scale,
    target_position=target_position,
    raw_target_position=decision.raw_target_position,
    stop_distance_rate=stop_distance_rate,
    max_position_by_risk=max_position_by_risk,
    session_drawdown=session_drawdown,
    portfolio_drawdown=portfolio_drawdown,
    pending_direction=decision.pending_direction,
    pending_direction_bars=decision.pending_direction_bars,
    constraint_flags=tuple(dict.fromkeys(constraint_flags)),
  )


def apply_prediction_overlays(
  *,
  prediction: float,
  row: dict[str, str],
  sizing: PositionSizingConfig,
) -> tuple[float, tuple[str, ...]]:
  adjusted_prediction = prediction
  flags: list[str] = []

  adjusted_prediction, trend_flags = _apply_trend_filter_overlay(
    prediction=adjusted_prediction,
    row=row,
    sizing=sizing,
  )
  flags.extend(trend_flags)
  adjusted_prediction, trend_state_flags = _apply_trend_state_overlay(
    prediction=adjusted_prediction,
    row=row,
    sizing=sizing,
  )
  flags.extend(trend_state_flags)
  return adjusted_prediction, tuple(dict.fromkeys(flags))


def _apply_trend_filter_overlay(
  *,
  prediction: float,
  row: dict[str, str],
  sizing: PositionSizingConfig,
) -> tuple[float, tuple[str, ...]]:
  if not sizing.trend_filter_enabled:
    return prediction, ()

  trend_direction = _trend_filter_direction(row, sizing)
  prediction_direction = _sign(prediction)
  adjusted_prediction = prediction
  flags: list[str] = []

  if trend_direction != 0 and prediction_direction != 0 and prediction_direction != trend_direction:
    adjusted_prediction = prediction * sizing.trend_filter_countertrend_scale
    flags.append("trend_filter")

  bias_rate = sizing.trend_filter_regime_bias_bps / 10_000
  if trend_direction != 0 and bias_rate != 0:
    adjusted_prediction += trend_direction * bias_rate
    flags.append("trend_regime_bias")

  return adjusted_prediction, tuple(flags)


def _apply_trend_state_overlay(
  *,
  prediction: float,
  row: dict[str, str],
  sizing: PositionSizingConfig,
) -> tuple[float, tuple[str, ...]]:
  if not sizing.trend_state_filter_enabled:
    return prediction, ()

  trend_state_direction = _trend_state_filter_direction(row, sizing)
  prediction_direction = _sign(prediction)
  adjusted_prediction = prediction
  flags: list[str] = []

  if trend_state_direction != 0 and prediction_direction != 0 and prediction_direction != trend_state_direction:
    adjusted_prediction = prediction * sizing.trend_state_countertrend_scale
    flags.append("trend_state_filter")

  chop_probability = _row_optional_float(row, sizing.trend_state_chop_probability_column)
  chop_threshold = min(1.0, max(0.0, sizing.trend_state_chop_threshold))
  if chop_probability is not None and chop_probability >= chop_threshold and _sign(adjusted_prediction) != 0:
    adjusted_prediction *= sizing.trend_state_chop_scale
    flags.append("trend_state_chop")

  bias_rate = sizing.trend_state_regime_bias_bps / 10_000
  if trend_state_direction != 0 and bias_rate != 0:
    adjusted_prediction += trend_state_direction * bias_rate
    flags.append("trend_state_regime_bias")

  return adjusted_prediction, tuple(flags)


def apply_position_persistence_overlays(
  *,
  decision: PositionDecision,
  task_type: str,
  prediction: float,
  row: dict[str, str],
  previous_position: float,
  active_position_bars: int,
  pending_direction: int,
  pending_direction_bars: int,
  sizing: PositionSizingConfig,
  execution_cost_rate: float,
  position_state: PositionState | None = None,
) -> PositionDecision:
  previous_direction = _sign(previous_position)
  target_direction = _sign(decision.target_position)
  prediction_direction = _sign(prediction)
  signal = abs(prediction)
  if task_type == "classification":
    hold_threshold = max(0.0, sizing.classification_hold_margin_threshold)
    flip_threshold = max(
      max(0.0, sizing.classification_flip_margin_threshold),
      max(0.0, sizing.classification_margin_threshold),
    )
  else:
    hold_threshold = max(sizing.hold_signal_threshold_bps / 10_000, 0.0) + execution_cost_rate
    flip_threshold = max(sizing.flip_signal_threshold_bps / 10_000, 0.0) + execution_cost_rate
  target_position = decision.target_position
  flags = list(decision.constraint_flags)
  next_pending_direction = 0
  next_pending_direction_bars = 0

  if (
    previous_direction != 0
    and target_direction == 0
    and prediction_direction == previous_direction
    and signal >= hold_threshold
  ):
    target_position = previous_position
    flags.append("hold_hysteresis")
  elif (
    sizing.hold_flat_until_time_stop
    and previous_direction != 0
    and target_direction == 0
  ):
    target_position = previous_position
    flags.append("time_stop_hold")

  if previous_direction != 0 and target_direction != 0 and target_direction != previous_direction:
    if signal < flip_threshold:
      if signal >= hold_threshold:
        target_position = 0.0
        flags.append("flip_exit_hysteresis")
      else:
        target_position = previous_position
        flags.append("flip_hysteresis")
    else:
      required_persistence = max(0, sizing.class_change_persistence_bars)
      if required_persistence > 0:
        next_pending_direction = target_direction
        next_pending_direction_bars = (
          max(0, pending_direction_bars) + 1
          if pending_direction == target_direction
          else 1
        )
        if next_pending_direction_bars < required_persistence:
          target_position = previous_position
          flags.append("class_change_persistence")
        else:
          next_pending_direction = 0
          next_pending_direction_bars = 0

  target_position, forecast_flags = apply_forecast_gate_overlay(
    row=row,
    previous_position=previous_position,
    target_position=target_position,
    sizing=sizing,
  )
  flags.extend(forecast_flags)

  target_position, regime_flags = apply_regime_chop_overlay(
    row=row,
    previous_position=previous_position,
    target_position=target_position,
    signal=signal,
    sizing=sizing,
  )
  flags.extend(regime_flags)

  target_position, tradeability_flags = apply_tradeability_gate_overlay(
    row=row,
    previous_position=previous_position,
    target_position=target_position,
    sizing=sizing,
  )
  flags.extend(tradeability_flags)

  target_position, smoothing_flags = apply_target_smoothing_overlay(
    previous_position=previous_position,
    target_position=target_position,
    sizing=sizing,
  )
  flags.extend(smoothing_flags)

  if (
    previous_direction != 0
    and active_position_bars > 0
    and active_position_bars < max(0, sizing.min_hold_bars)
    and _sign(target_position) != previous_direction
  ):
    target_position = previous_position
    flags.append("min_hold")

  if (
    previous_direction != 0
    and _sign(target_position) == previous_direction
    and abs(target_position - previous_position) < max(0.0, sizing.min_rebalance_abs_delta)
  ):
    target_position = previous_position
    flags.append("min_rebalance")

  if sizing.hold_until_stop_or_take_profit:
    target_position, exit_flags = apply_hold_until_stop_or_take_profit_overlay(
      target_position=target_position,
      task_type=task_type,
      signal=signal,
      row=row,
      previous_position=previous_position,
      active_position_bars=active_position_bars,
      sizing=sizing,
      execution_cost_rate=execution_cost_rate,
      stop_distance_rate=decision.stop_distance_rate,
      position_state=position_state,
    )
    if previous_direction != 0 and active_position_bars > 0:
      next_pending_direction = 0
      next_pending_direction_bars = 0
  else:
    target_position, exit_flags = apply_duration_exit_overlays(
      target_position=target_position,
      task_type=task_type,
      signal=signal,
      row=row,
      previous_position=previous_position,
      active_position_bars=active_position_bars,
      sizing=sizing,
      execution_cost_rate=execution_cost_rate,
      stop_distance_rate=decision.stop_distance_rate,
      position_state=position_state,
    )
  flags.extend(exit_flags)

  return PositionDecision(
    signal=decision.signal,
    direction=_sign(target_position),
    confidence=decision.confidence if target_position != 0 else 0.0,
    volatility=decision.volatility,
    volatility_scale=decision.volatility_scale,
    target_position=target_position,
    raw_target_position=decision.raw_target_position,
    stop_distance_rate=decision.stop_distance_rate,
    max_position_by_risk=decision.max_position_by_risk,
    session_drawdown=decision.session_drawdown,
    portfolio_drawdown=decision.portfolio_drawdown,
    pending_direction=next_pending_direction,
    pending_direction_bars=next_pending_direction_bars,
    constraint_flags=tuple(dict.fromkeys(flags)),
  )


def apply_forecast_gate_overlay(
  *,
  row: dict[str, str],
  previous_position: float,
  target_position: float,
  sizing: PositionSizingConfig,
) -> tuple[float, tuple[str, ...]]:
  if not sizing.forecast_gate_enabled:
    return target_position, ()

  previous_direction = _sign(previous_position)
  target_direction = _sign(target_position)
  if target_direction == 0:
    return target_position, ()

  fast_threshold = max(0.0, sizing.forecast_gate_fast_threshold_bps) / 10_000
  slow_threshold = max(0.0, sizing.forecast_gate_slow_threshold_bps) / 10_000
  fast_direction = _threshold_sign(
    _row_optional_float(row, sizing.forecast_gate_fast_column),
    fast_threshold,
  )
  slow_direction = _threshold_sign(
    _row_optional_float(row, sizing.forecast_gate_slow_column),
    slow_threshold,
  )
  if sizing.forecast_gate_require_alignment:
    allowed = fast_direction == target_direction and slow_direction == target_direction
  else:
    allowed = fast_direction == target_direction or slow_direction == target_direction
  if allowed:
    return target_position, ()

  if previous_direction == 0:
    return 0.0, ("forecast_gate_entry",)
  if target_direction != previous_direction:
    return 0.0, ("forecast_gate_reverse",)
  if abs(target_position) > abs(previous_position):
    return previous_position, ("forecast_gate_scale_in",)
  return target_position, ()


def apply_tradeability_gate_overlay(
  *,
  row: dict[str, str],
  previous_position: float,
  target_position: float,
  sizing: PositionSizingConfig,
) -> tuple[float, tuple[str, ...]]:
  if not sizing.tradeability_gate_enabled:
    return target_position, ()

  probability = _row_optional_float(row, sizing.tradeability_gate_probability_column)
  if probability is None:
    return target_position, ()

  confidence = _row_optional_float(row, sizing.tradeability_gate_confidence_column)
  min_confidence = min(1.0, max(0.0, sizing.tradeability_gate_min_confidence))
  if confidence is not None and confidence < min_confidence:
    return target_position, ()

  threshold = min(1.0, max(0.0, sizing.tradeability_gate_threshold))
  if probability >= threshold:
    return target_position, ()

  previous_direction = _sign(previous_position)
  target_direction = _sign(target_position)
  if previous_direction == 0 and target_direction != 0:
    return 0.0, ("tradeability_gate_entry",)
  if (
    previous_direction != 0
    and target_direction != 0
    and target_direction != previous_direction
  ):
    return 0.0, ("tradeability_gate_reverse",)
  if (
    previous_direction != 0
    and target_direction == previous_direction
    and abs(target_position) > abs(previous_position)
  ):
    return previous_position, ("tradeability_gate_scale_in",)
  return target_position, ()


def apply_regime_chop_overlay(
  *,
  row: dict[str, str],
  previous_position: float,
  target_position: float,
  signal: float,
  sizing: PositionSizingConfig,
) -> tuple[float, tuple[str, ...]]:
  if not sizing.regime_chop_gate_enabled:
    return target_position, ()

  chop_probability = _row_optional_float(row, sizing.regime_chop_probability_column)
  threshold = min(1.0, max(0.0, sizing.regime_chop_threshold))
  if chop_probability is None or chop_probability < threshold:
    return target_position, ()

  previous_direction = _sign(previous_position)
  target_direction = _sign(target_position)
  flags: list[str] = []

  if previous_direction == 0 and target_direction != 0:
    entry_threshold = max(0.0, sizing.classification_margin_threshold) * max(1.0, sizing.regime_chop_entry_threshold_scale)
    if signal < entry_threshold:
      return 0.0, ("regime_chop_entry",)

  if previous_direction != 0 and target_direction != 0 and target_direction != previous_direction:
    flip_threshold = max(0.0, sizing.classification_flip_margin_threshold) * max(1.0, sizing.regime_chop_flip_threshold_scale)
    if signal < flip_threshold:
      return 0.0, ("regime_chop_reverse",)

  max_position_scale = min(1.0, max(0.0, sizing.regime_chop_max_position_scale))
  if max_position_scale > 0 and target_direction != 0:
    capped_abs_position = min(abs(target_position), sizing.max_abs_position * max_position_scale)
    if capped_abs_position < abs(target_position):
      target_position = target_direction * capped_abs_position
      flags.append("regime_chop_size_cap")
  return target_position, tuple(flags)


def apply_target_smoothing_overlay(
  *,
  previous_position: float,
  target_position: float,
  sizing: PositionSizingConfig,
) -> tuple[float, tuple[str, ...]]:
  smoothing = min(1.0, max(0.0, sizing.target_position_smoothing))
  if smoothing <= 0.0 or smoothing >= 1.0 or target_position == previous_position:
    return target_position, ()
  smoothed_target = previous_position + (target_position - previous_position) * smoothing
  return smoothed_target, ("target_smoothing",)


def apply_hold_until_stop_or_take_profit_overlay(
  *,
  target_position: float,
  task_type: str,
  signal: float,
  row: dict[str, str],
  previous_position: float,
  active_position_bars: int,
  sizing: PositionSizingConfig,
  execution_cost_rate: float,
  stop_distance_rate: float,
  position_state: PositionState | None = None,
) -> tuple[float, tuple[str, ...]]:
  previous_direction = _sign(previous_position)
  if previous_direction == 0 or active_position_bars <= 0:
    return target_position, ()

  held_target_position, exit_flags = apply_hold_release_overlays(
    task_type=task_type,
    signal=signal,
    row=row,
    previous_position=previous_position,
    active_position_bars=active_position_bars,
    sizing=sizing,
    execution_cost_rate=execution_cost_rate,
    stop_distance_rate=stop_distance_rate,
    position_state=position_state,
  )
  if exit_flags:
    return held_target_position, exit_flags
  if target_position != previous_position:
    return previous_position, ("hold_until_exit",)
  return previous_position, ()


def apply_hold_release_overlays(
  *,
  task_type: str,
  signal: float,
  row: dict[str, str],
  previous_position: float,
  active_position_bars: int,
  sizing: PositionSizingConfig,
  execution_cost_rate: float,
  stop_distance_rate: float,
  position_state: PositionState | None = None,
) -> tuple[float, tuple[str, ...]]:
  if sizing.max_hold_bars > 0 and active_position_bars >= max(0, sizing.max_hold_bars):
    return 0.0, ("max_hold_exit",)

  price_exit_target, price_exit_flags = apply_price_target_exit_overlays(
    row=row,
    previous_position=previous_position,
    active_position_bars=active_position_bars,
    sizing=sizing,
    stop_distance_rate=stop_distance_rate,
    position_state=position_state,
  )
  if price_exit_flags:
    return price_exit_target, price_exit_flags

  if _should_trailing_profit_exit(
    position_state=position_state,
    stop_distance_rate=stop_distance_rate,
    sizing=sizing,
  ):
    return 0.0, ("trailing_profit_exit",)

  regime_exit_target, regime_exit_flags = apply_regime_exit_overlays(
    row=row,
    previous_position=previous_position,
    sizing=sizing,
  )
  if regime_exit_flags:
    return regime_exit_target, regime_exit_flags

  if sizing.stale_exit_bars > 0 and active_position_bars >= max(0, sizing.stale_exit_bars):
    stale_threshold = _stale_exit_threshold(
      task_type=task_type,
      sizing=sizing,
      execution_cost_rate=execution_cost_rate,
    )
    if stale_threshold > 0 and signal < stale_threshold:
      return 0.0, ("stale_exit",)

  return previous_position, ()


def apply_duration_exit_overlays(
  *,
  target_position: float,
  task_type: str,
  signal: float,
  row: dict[str, str],
  previous_position: float,
  active_position_bars: int,
  sizing: PositionSizingConfig,
  execution_cost_rate: float,
  stop_distance_rate: float,
  position_state: PositionState | None = None,
) -> tuple[float, tuple[str, ...]]:
  previous_direction = _sign(previous_position)
  target_direction = _sign(target_position)
  if previous_direction == 0 or target_direction != previous_direction or active_position_bars <= 0:
    return target_position, ()

  if sizing.max_hold_bars > 0 and active_position_bars >= max(0, sizing.max_hold_bars):
    return 0.0, ("max_hold_exit",)

  price_exit_target, price_exit_flags = apply_price_target_exit_overlays(
    row=row,
    previous_position=previous_position,
    active_position_bars=active_position_bars,
    sizing=sizing,
    stop_distance_rate=stop_distance_rate,
    position_state=position_state,
  )
  if price_exit_flags:
    return price_exit_target, price_exit_flags

  if _should_trailing_profit_exit(
    position_state=position_state,
    stop_distance_rate=stop_distance_rate,
    sizing=sizing,
  ):
    return 0.0, ("trailing_profit_exit",)

  if sizing.stale_exit_bars > 0 and active_position_bars >= max(0, sizing.stale_exit_bars):
    stale_threshold = _stale_exit_threshold(
      task_type=task_type,
      sizing=sizing,
      execution_cost_rate=execution_cost_rate,
    )
    if stale_threshold > 0 and signal < stale_threshold:
      return 0.0, ("stale_exit",)

    regime_exit_target, regime_exit_flags = apply_regime_exit_overlays(
      row=row,
      previous_position=previous_position,
      sizing=sizing,
    )
    if regime_exit_flags:
      return regime_exit_target, regime_exit_flags

  return target_position, ()


def apply_price_target_exit_overlays(
  *,
  row: dict[str, str],
  previous_position: float,
  active_position_bars: int,
  sizing: PositionSizingConfig,
  stop_distance_rate: float,
  position_state: PositionState | None = None,
) -> tuple[float, tuple[str, ...]]:
  previous_direction = _sign(previous_position)
  if previous_direction == 0 or active_position_bars <= 0:
    return previous_position, ()

  if _should_stop_loss_exit(
    row=row,
    previous_position=previous_position,
    position_state=position_state,
    stop_distance_rate=stop_distance_rate,
  ):
    return 0.0, ("stop_loss_exit",)

  if _should_take_profit_exit(
    row=row,
    previous_position=previous_position,
    position_state=position_state,
    stop_distance_rate=stop_distance_rate,
    sizing=sizing,
  ):
    return 0.0, ("profit_take_exit",)

  return previous_position, ()


def apply_regime_exit_overlays(
  *,
  row: dict[str, str],
  previous_position: float,
  sizing: PositionSizingConfig,
) -> tuple[float, tuple[str, ...]]:
  previous_direction = _sign(previous_position)
  if previous_direction == 0:
    return previous_position, ()

  flags: list[str] = []
  if sizing.trend_filter_exit_enabled and _trend_filter_direction(row, sizing) == -previous_direction:
    flags.append("trend_filter_exit")

  if sizing.trend_state_exit_enabled:
    if _trend_state_exit_direction(row, sizing) == -previous_direction:
      flags.append("trend_state_exit")
    chop_probability = _row_optional_float(row, sizing.trend_state_chop_probability_column)
    if (
      chop_probability is not None
      and chop_probability >= min(1.0, max(0.0, sizing.trend_state_chop_exit_threshold))
    ):
      flags.append("trend_state_chop_exit")

  if flags:
    return 0.0, tuple(dict.fromkeys(flags))
  return previous_position, ()


def _stale_exit_threshold(
  *,
  task_type: str,
  sizing: PositionSizingConfig,
  execution_cost_rate: float,
) -> float:
  if task_type == "classification":
    return max(0.0, sizing.classification_stale_exit_margin_threshold)
  return max(0.0, sizing.stale_exit_signal_threshold_bps / 10_000) + execution_cost_rate


def _should_stop_loss_exit(
  *,
  row: dict[str, str],
  previous_position: float,
  position_state: PositionState | None,
  stop_distance_rate: float,
) -> bool:
  previous_direction = _sign(previous_position)
  stop_loss_price = _position_target_price(
    direction=previous_direction,
    entry_price=position_state.entry_price if position_state is not None else None,
    distance_rate=stop_distance_rate,
    favorable=False,
  )
  if stop_loss_price is None:
    return False
  if previous_direction > 0:
    adverse_price = _row_optional_float(row, "low")
    if adverse_price is None:
      adverse_price = position_state.mark_price if position_state is not None else None
    if adverse_price is not None:
      return adverse_price <= stop_loss_price
    return position_state is not None and position_state.max_adverse_return <= -stop_distance_rate

  adverse_price = _row_optional_float(row, "high")
  if adverse_price is None:
    adverse_price = position_state.mark_price if position_state is not None else None
  if adverse_price is not None:
    return adverse_price >= stop_loss_price
  return position_state is not None and position_state.max_adverse_return <= -stop_distance_rate


def _should_take_profit_exit(
  *,
  row: dict[str, str],
  previous_position: float,
  position_state: PositionState | None,
  stop_distance_rate: float,
  sizing: PositionSizingConfig,
) -> bool:
  previous_direction = _sign(previous_position)
  take_profit_multiple = max(0.0, sizing.profit_take_exit_multiple)
  if take_profit_multiple <= 0:
    return False
  take_profit_price = _position_target_price(
    direction=previous_direction,
    entry_price=position_state.entry_price if position_state is not None else None,
    distance_rate=take_profit_multiple * stop_distance_rate,
    favorable=True,
  )
  if take_profit_price is None:
    return False
  if previous_direction > 0:
    favorable_price = _row_optional_float(row, "high")
    if favorable_price is None:
      favorable_price = position_state.mark_price if position_state is not None else None
    if favorable_price is not None:
      return favorable_price >= take_profit_price
    return position_state is not None and position_state.open_return >= take_profit_multiple * stop_distance_rate

  favorable_price = _row_optional_float(row, "low")
  if favorable_price is None:
    favorable_price = position_state.mark_price if position_state is not None else None
  if favorable_price is not None:
    return favorable_price <= take_profit_price
  return position_state is not None and position_state.open_return >= take_profit_multiple * stop_distance_rate


def _position_target_price(
  *,
  direction: int,
  entry_price: float | None,
  distance_rate: float,
  favorable: bool,
) -> float | None:
  if direction == 0 or entry_price is None or entry_price <= 0 or distance_rate <= 0:
    return None
  if favorable:
    if direction > 0:
      return entry_price * (1.0 + distance_rate)
    return entry_price * (1.0 - distance_rate)
  if direction > 0:
    return entry_price * (1.0 - distance_rate)
  return entry_price * (1.0 + distance_rate)

def _should_trailing_profit_exit(
  *,
  position_state: PositionState | None,
  stop_distance_rate: float,
  sizing: PositionSizingConfig,
) -> bool:
  if position_state is None or stop_distance_rate <= 0:
    return False
  activation_multiple = max(0.0, sizing.trailing_profit_activation_multiple)
  retrace_multiple = max(0.0, sizing.trailing_profit_retrace_multiple)
  if activation_multiple <= 0 or retrace_multiple <= 0:
    return False
  if position_state.max_favorable_return < activation_multiple * stop_distance_rate:
    return False
  giveback = max(0.0, position_state.max_favorable_return - position_state.open_return)
  return giveback >= retrace_multiple * stop_distance_rate


def _trend_filter_direction(row: dict[str, str], sizing: PositionSizingConfig) -> int:
  threshold = sizing.trend_filter_threshold_bps / 10_000
  primary_direction = _threshold_sign(_row_optional_float(row, sizing.trend_filter_column), threshold)
  higher_direction = _threshold_sign(
    _row_optional_float(row, sizing.trend_filter_higher_timeframe_column),
    threshold,
  )

  if sizing.trend_filter_require_alignment:
    if primary_direction != 0 and primary_direction == higher_direction:
      return primary_direction
    return 0

  if primary_direction != 0 and higher_direction != 0 and primary_direction != higher_direction:
    return 0
  return primary_direction or higher_direction


def _trend_state_filter_direction(row: dict[str, str], sizing: PositionSizingConfig) -> int:
  uptrend_probability = _row_optional_float(row, sizing.trend_state_uptrend_probability_column)
  downtrend_probability = _row_optional_float(row, sizing.trend_state_downtrend_probability_column)
  if uptrend_probability is None and downtrend_probability is None:
    return 0

  confidence = _row_optional_float(row, sizing.trend_state_confidence_column)
  min_confidence = min(1.0, max(0.0, sizing.trend_state_min_confidence))
  if confidence is not None and confidence < min_confidence:
    return 0

  uptrend_probability = max(0.0, uptrend_probability or 0.0)
  downtrend_probability = max(0.0, downtrend_probability or 0.0)
  gate_threshold = min(1.0, max(0.0, sizing.trend_state_gate_threshold))
  strongest_probability = max(uptrend_probability, downtrend_probability)
  if strongest_probability < gate_threshold:
    return 0
  if uptrend_probability > downtrend_probability:
    return 1
  if downtrend_probability > uptrend_probability:
    return -1
  return 0


def _trend_state_exit_direction(row: dict[str, str], sizing: PositionSizingConfig) -> int:
  uptrend_probability = _row_optional_float(row, sizing.trend_state_uptrend_probability_column)
  downtrend_probability = _row_optional_float(row, sizing.trend_state_downtrend_probability_column)
  if uptrend_probability is None and downtrend_probability is None:
    return 0

  uptrend_probability = max(0.0, uptrend_probability or 0.0)
  downtrend_probability = max(0.0, downtrend_probability or 0.0)
  gate_threshold = min(1.0, max(0.0, sizing.trend_state_exit_gate_threshold))
  strongest_probability = max(uptrend_probability, downtrend_probability)
  if strongest_probability < gate_threshold:
    return 0
  if uptrend_probability > downtrend_probability:
    return 1
  if downtrend_probability > uptrend_probability:
    return -1
  return 0


def mark_position_state(
  *,
  row: dict[str, str],
  previous_position: float,
  entry_price: float | None,
  max_favorable_return: float = 0.0,
  max_adverse_return: float = 0.0,
) -> PositionState:
  previous_direction = _sign(previous_position)
  if previous_direction == 0 or entry_price is None or entry_price <= 0:
    return PositionState()

  mark_price = _row_optional_float(row, "close")
  favorable_price = _row_optional_float(row, "high")
  adverse_price = _row_optional_float(row, "low")
  if previous_direction < 0:
    favorable_price, adverse_price = adverse_price, favorable_price

  open_return = _position_return(previous_direction, entry_price, mark_price)
  current_favorable_return = max(0.0, _position_return(previous_direction, entry_price, favorable_price))
  current_adverse_return = min(0.0, _position_return(previous_direction, entry_price, adverse_price))
  return PositionState(
    entry_price=entry_price,
    mark_price=mark_price,
    open_return=open_return,
    max_favorable_return=max(max(0.0, max_favorable_return), current_favorable_return),
    max_adverse_return=min(min(0.0, max_adverse_return), current_adverse_return),
  )


def next_position_state(
  *,
  row: dict[str, str],
  previous_position: float,
  target_position: float,
  previous_state: PositionState | None = None,
) -> PositionState:
  previous_state = previous_state or PositionState()
  target_direction = _sign(target_position)
  previous_direction = _sign(previous_position)
  if target_direction == 0:
    return PositionState()

  entry_price = previous_state.entry_price
  current_price = _row_optional_float(row, "close")
  if current_price is None or current_price <= 0:
    return PositionState(entry_price=entry_price)

  if previous_direction == 0 or previous_direction != target_direction or entry_price is None or entry_price <= 0:
    return mark_position_state(
      row=row,
      previous_position=target_position,
      entry_price=current_price,
    )

  if abs(target_position) > abs(previous_position):
    added_position = abs(target_position) - abs(previous_position)
    blended_entry_price = (
      (abs(previous_position) * entry_price) + (added_position * current_price)
    ) / max(abs(target_position), 1e-9)
    return mark_position_state(
      row=row,
      previous_position=target_position,
      entry_price=blended_entry_price,
    )

  return mark_position_state(
    row=row,
    previous_position=target_position,
    entry_price=entry_price,
    max_favorable_return=previous_state.max_favorable_return,
    max_adverse_return=previous_state.max_adverse_return,
  )


def _row_float(row: dict[str, str], column: str, default: float) -> float:
  value = row.get(column, "")
  if value == "":
    return default
  return float(value)


def _row_optional_float(row: dict[str, str], column: str) -> float | None:
  value = row.get(column, "")
  if value == "":
    return None
  return float(value)


def _sign(value: float) -> int:
  if value > 0:
    return 1
  if value < 0:
    return -1
  return 0


def _threshold_sign(value: float | None, threshold: float) -> int:
  if value is None:
    return 0
  if value >= threshold:
    return 1
  if value <= -threshold:
    return -1
  return 0


def _position_return(direction: int, entry_price: float, mark_price: float | None) -> float:
  if direction == 0 or entry_price <= 0 or mark_price is None or mark_price <= 0:
    return 0.0
  if direction > 0:
    return (mark_price / entry_price) - 1.0
  return (entry_price / mark_price) - 1.0


def _stop_distance_rate(*, volatility: float, sizing: PositionSizingConfig) -> float:
  return max(
    sizing.stop_loss_floor_bps / 10_000,
    sizing.stop_loss_sigma * max(volatility, sizing.volatility_floor),
    1e-9,
  )


def next_active_position_bars(
  *,
  previous_position: float,
  previous_active_position_bars: int,
  target_position: float,
  same_candle: bool = False,
) -> int:
  target_direction = _sign(target_position)
  previous_direction = _sign(previous_position)
  if target_direction == 0:
    return 0
  if previous_direction == target_direction and previous_direction != 0:
    return max(1, previous_active_position_bars) if same_candle else max(1, previous_active_position_bars) + 1
  return 1


def _drawdown_fraction(*, peak_equity: float, equity: float) -> float:
  if peak_equity <= 0:
    return 0.0
  return max(0.0, (peak_equity - equity) / peak_equity)
