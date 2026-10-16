from __future__ import annotations

import math
from dataclasses import dataclass, field


REGIME_STATE_NAMES = ("bear", "chop", "bull")
REGIME_STATE_INDEX = {name: index for index, name in enumerate(REGIME_STATE_NAMES)}


@dataclass(frozen=True, slots=True)
class RegimeObservation:
  short_strength: float
  long_strength: float
  momentum_strength: float
  volatility_regime: float

  @property
  def change_signal(self) -> float:
    return self.long_strength


@dataclass(frozen=True, slots=True)
class RegimeFeatureSnapshot:
  hmm_p_bear: float | None = None
  hmm_p_chop: float | None = None
  hmm_p_bull: float | None = None
  hmm_confidence: float | None = None
  hmm_entropy: float | None = None
  hmm_trend_score: float | None = None
  hmm_state_index: int | None = None
  hmm_transition_risk: float | None = None
  hmm_expected_duration: float | None = None
  bocpd_change_prob: float | None = None
  bocpd_expected_run_length: float | None = None
  bocpd_stability_score: float | None = None
  bocpd_change_flag: int | None = None


def build_regime_observation(context_row: dict[str, str]) -> RegimeObservation | None:
  short_return = _float_or_none(context_row.get("trend_return_6_bar", ""))
  medium_return = _float_or_none(context_row.get("trend_return_24_bar", ""))
  if medium_return is None:
    medium_return = short_return

  ema_distance = _first_nonempty_float(
    context_row,
    "trend_close_vs_ema_6_pct",
    "trend_close_vs_ema_24_pct",
    "structure_close_vs_session_vwap_pct",
  )
  volatility = _first_nonempty_float(
    context_row,
    "trend_realized_volatility_20_bar",
    "rolling_return_std_5_bar",
    "trend_realized_volatility_48_bar",
  )

  if short_return is None or medium_return is None or ema_distance is None or volatility is None:
    return None

  volatility_scale = max(abs(volatility), 5e-4)
  return RegimeObservation(
    short_strength=short_return / volatility_scale,
    long_strength=medium_return / volatility_scale,
    momentum_strength=ema_distance / volatility_scale,
    volatility_regime=math.log(max(volatility_scale, 1e-6) / 1e-3),
  )


@dataclass(slots=True)
class OnlineRegimeFeatureModel:
  max_run_length: int = 96
  hazard: float = 1.0 / 48.0
  transition_prior_diagonal: float = 8.0
  transition_prior_off_diagonal: float = 1.0
  emission_counts: list[int] = field(default_factory=lambda: [0, 0, 0])
  emission_sums: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0, 0.0] for _ in REGIME_STATE_NAMES])
  emission_sumsq: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0, 0.0] for _ in REGIME_STATE_NAMES])
  transition_counts: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0] for _ in REGIME_STATE_NAMES])
  posterior: list[float] = field(default_factory=lambda: [1.0 / len(REGIME_STATE_NAMES) for _ in REGIME_STATE_NAMES])
  previous_bucket_index: int | None = None
  run_length_probs: list[float] = field(default_factory=lambda: [1.0])
  run_means: list[float] = field(default_factory=lambda: [0.0])
  run_variances: list[float] = field(default_factory=lambda: [1.0])
  run_counts: list[int] = field(default_factory=lambda: [0])

  def update(self, observation: RegimeObservation | None) -> RegimeFeatureSnapshot:
    if observation is None:
      return RegimeFeatureSnapshot()

    transition_matrix = self._transition_matrix()
    state_means, state_variances = self._emission_parameters()
    likelihoods = [
      _gaussian_diag_pdf(
        values=(
          observation.short_strength,
          observation.long_strength,
          observation.momentum_strength,
          observation.volatility_regime,
        ),
        means=state_means[state_index],
        variances=state_variances[state_index],
      )
      for state_index in range(len(REGIME_STATE_NAMES))
    ]
    predicted = [
      sum(self.posterior[prev_state] * transition_matrix[prev_state][state_index] for prev_state in range(len(REGIME_STATE_NAMES)))
      for state_index in range(len(REGIME_STATE_NAMES))
    ]
    posterior = _normalize([
      predicted[state_index] * likelihoods[state_index]
      for state_index in range(len(REGIME_STATE_NAMES))
    ])

    bocpd_change_prob, bocpd_expected_run_length = self._update_bocpd(observation.change_signal)

    state_index = max(range(len(posterior)), key=lambda index: posterior[index])
    confidence = max(posterior)
    entropy = _entropy(posterior)
    transition_risk = 1.0 - sum(
      posterior[state] * transition_matrix[state][state]
      for state in range(len(REGIME_STATE_NAMES))
    )
    expected_duration = sum(
      posterior[state] * _expected_geometric_duration(transition_matrix[state][state])
      for state in range(len(REGIME_STATE_NAMES))
    )
    snapshot = RegimeFeatureSnapshot(
      hmm_p_bear=posterior[REGIME_STATE_INDEX["bear"]],
      hmm_p_chop=posterior[REGIME_STATE_INDEX["chop"]],
      hmm_p_bull=posterior[REGIME_STATE_INDEX["bull"]],
      hmm_confidence=confidence,
      hmm_entropy=entropy,
      hmm_trend_score=posterior[REGIME_STATE_INDEX["bull"]] - posterior[REGIME_STATE_INDEX["bear"]],
      hmm_state_index=state_index,
      hmm_transition_risk=transition_risk,
      hmm_expected_duration=expected_duration,
      bocpd_change_prob=bocpd_change_prob,
      bocpd_expected_run_length=bocpd_expected_run_length,
      bocpd_stability_score=(1.0 - bocpd_change_prob),
      bocpd_change_flag=1 if bocpd_change_prob >= 0.35 else 0,
    )

    bucket_index = _provisional_bucket(observation)
    self._update_emissions(bucket_index, observation)
    if self.previous_bucket_index is not None:
      self.transition_counts[self.previous_bucket_index][bucket_index] += 1.0
    self.previous_bucket_index = bucket_index
    self.posterior = posterior
    return snapshot

  def _transition_matrix(self) -> list[list[float]]:
    matrix: list[list[float]] = []
    for state_index, counts in enumerate(self.transition_counts):
      row = []
      for next_state_index in range(len(REGIME_STATE_NAMES)):
        prior = (
          self.transition_prior_diagonal
          if next_state_index == state_index
          else self.transition_prior_off_diagonal
        )
        row.append(counts[next_state_index] + prior)
      total = sum(row)
      matrix.append([value / total for value in row])
    return matrix

  def _emission_parameters(self) -> tuple[list[list[float]], list[list[float]]]:
    prior_means = {
      "bear": (-0.75, -1.25, -0.75, 0.15),
      "chop": (0.0, 0.0, 0.0, -0.15),
      "bull": (0.75, 1.25, 0.75, 0.05),
    }
    prior_variances = {
      "bear": (1.0, 1.5, 1.0, 0.4),
      "chop": (0.8, 0.8, 0.8, 0.3),
      "bull": (1.0, 1.5, 1.0, 0.4),
    }
    means: list[list[float]] = []
    variances: list[list[float]] = []
    for state_name in REGIME_STATE_NAMES:
      state_index = REGIME_STATE_INDEX[state_name]
      count = self.emission_counts[state_index]
      if count <= 1:
        means.append(list(prior_means[state_name]))
        variances.append(list(prior_variances[state_name]))
        continue

      state_means: list[float] = []
      state_vars: list[float] = []
      for feature_index in range(4):
        mean = self.emission_sums[state_index][feature_index] / count
        variance = (
          self.emission_sumsq[state_index][feature_index] / count
          - mean * mean
        )
        blended_mean = 0.35 * prior_means[state_name][feature_index] + 0.65 * mean
        blended_var = 0.35 * prior_variances[state_name][feature_index] + 0.65 * max(variance, 0.15)
        state_means.append(blended_mean)
        state_vars.append(max(blended_var, 0.1))
      means.append(state_means)
      variances.append(state_vars)
    return means, variances

  def _update_emissions(self, bucket_index: int, observation: RegimeObservation) -> None:
    values = (
      observation.short_strength,
      observation.long_strength,
      observation.momentum_strength,
      observation.volatility_regime,
    )
    self.emission_counts[bucket_index] += 1
    for feature_index, value in enumerate(values):
      self.emission_sums[bucket_index][feature_index] += value
      self.emission_sumsq[bucket_index][feature_index] += value * value

  def _update_bocpd(self, value: float) -> tuple[float, float]:
    prior_predictive = _gaussian_pdf(value, 0.0, 1.25)
    growth_probs: list[float] = []
    for probability, mean, variance in zip(self.run_length_probs, self.run_means, self.run_variances):
      predictive = _gaussian_pdf(value, mean, math.sqrt(max(variance, 0.2) + 0.25))
      growth_probs.append((1.0 - self.hazard) * probability * predictive)

    cp_prob = self.hazard * prior_predictive * sum(self.run_length_probs)
    updated_probs = [cp_prob, *growth_probs]
    updated_probs = updated_probs[: self.max_run_length + 1]
    updated_probs = _normalize(updated_probs)

    new_means = [value]
    new_variances = [1.0]
    new_counts = [1]
    for mean, variance, count in zip(self.run_means, self.run_variances, self.run_counts):
      updated_mean, updated_variance, updated_count = _update_running_gaussian(mean, variance, count, value)
      new_means.append(updated_mean)
      new_variances.append(updated_variance)
      new_counts.append(updated_count)
    self.run_length_probs = updated_probs
    self.run_means = new_means[: len(updated_probs)]
    self.run_variances = new_variances[: len(updated_probs)]
    self.run_counts = new_counts[: len(updated_probs)]

    expected_run_length = sum(
      run_length * probability
      for run_length, probability in enumerate(self.run_length_probs)
    )
    return self.run_length_probs[0], expected_run_length


def _provisional_bucket(observation: RegimeObservation) -> int:
  if observation.long_strength >= 1.0 and observation.momentum_strength >= 0.25:
    return REGIME_STATE_INDEX["bull"]
  if observation.long_strength <= -1.0 and observation.momentum_strength <= -0.25:
    return REGIME_STATE_INDEX["bear"]
  return REGIME_STATE_INDEX["chop"]


def _gaussian_diag_pdf(
  *,
  values: tuple[float, ...],
  means: list[float],
  variances: list[float],
) -> float:
  log_probability = 0.0
  for value, mean, variance in zip(values, means, variances):
    stable_variance = max(variance, 1e-3)
    log_probability += -0.5 * (
      math.log(2.0 * math.pi * stable_variance)
      + ((value - mean) ** 2) / stable_variance
    )
  return math.exp(log_probability)


def _gaussian_pdf(value: float, mean: float, std: float) -> float:
  stable_std = max(abs(std), 1e-3)
  variance = stable_std * stable_std
  return math.exp(
    -0.5 * (
      math.log(2.0 * math.pi * variance)
      + ((value - mean) ** 2) / variance
    )
  )


def _normalize(values: list[float]) -> list[float]:
  total = sum(values)
  if total <= 0:
    return [1.0 / len(values) for _ in values]
  return [value / total for value in values]


def _entropy(probabilities: list[float]) -> float:
  if not probabilities:
    return 0.0
  max_entropy = math.log(len(probabilities))
  if max_entropy <= 0:
    return 0.0
  entropy = -sum(
    probability * math.log(probability)
    for probability in probabilities
    if probability > 0
  )
  return entropy / max_entropy


def _expected_geometric_duration(self_transition_probability: float) -> float:
  bounded = min(max(self_transition_probability, 1e-6), 0.999)
  return 1.0 / (1.0 - bounded)


def _update_running_gaussian(mean: float, variance: float, count: int, value: float) -> tuple[float, float, int]:
  if count <= 0:
    return value, 1.0, 1
  new_count = count + 1
  delta = value - mean
  new_mean = mean + delta / new_count
  m2 = max(variance, 0.0) * count
  new_m2 = m2 + delta * (value - new_mean)
  new_variance = max(new_m2 / new_count, 0.15)
  return new_mean, new_variance, new_count


def _float_or_none(value: str | None) -> float | None:
  if value in (None, ""):
    return None
  return float(value)


def _first_nonempty_float(row: dict[str, str], *columns: str) -> float | None:
  for column in columns:
    value = _float_or_none(row.get(column))
    if value is not None:
      return value
  return None
