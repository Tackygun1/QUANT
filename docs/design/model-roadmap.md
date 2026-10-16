# Model Roadmap

## Goal

Build a small, auditable model stack for a Python-first crypto trading bot without jumping straight into an overfit research mess.

The system should separate:

- forecasting,
- trade selection,
- position sizing,
- and risk enforcement.

Do not start by building ten independent models. Start with a narrow stack that earns the right to become more complex.

## Recommended Model Stack

### 1. Non-ML baselines

Build these first so every ML model has a real benchmark:

- buy-and-hold baseline,
- flat or no-trade baseline,
- simple momentum rule,
- simple mean-reversion rule,
- spread and fee-aware threshold rule.

If a learned model cannot beat these after fees and slippage, it is not useful.

### 2. Tabular baseline models

These should be the first learned models in production-like research.

Recommended first models:

- linear regression or ridge for multi-horizon return prediction,
- logistic regression for direction or trade/no-trade classification,
- gradient-boosted trees for nonlinear tabular signals.

Why start here:

- they train quickly,
- they are easier to debug,
- feature importance is easier to inspect,
- and they establish whether the feature set has signal before sequence models are introduced.

### 3. Sequence model

Add one Transformer-style forecaster only after tabular baselines are stable.

Recommended role:

- predict multi-horizon returns or return buckets from sequential trade, candle, and order-book features,
- emit both point forecasts and uncertainty-like outputs when practical.

The Transformer should not place trades directly. Its job is to produce forecasts that become inputs to the final decision layer.

### 4. Meta-model / combiner

Use a tree ensemble as the final decision model that combines:

- engineered tabular features,
- Transformer outputs,
- market state features,
- and execution cost context.

Recommended responsibilities:

- decide trade direction,
- decide whether expected edge exceeds transaction costs,
- provide a confidence score for position sizing.

This is the most practical hybrid design for the project.

### 5. Position-sizing model

Do not begin with a fully learned sizing model.

Start with a rule-based sizing policy driven by:

- forecast strength,
- forecast confidence,
- volatility,
- current exposure,
- and hard risk caps.

Later, if justified, replace or augment this with a small model that predicts expected payoff conditional on taking the trade.

### 6. Regime model

Treat this as optional and later-stage.

Useful examples:

- volatility regime classifier,
- trend/chop classifier,
- liquidity regime detector.

The regime model should usually act as a feature source or model router, not as the trading policy by itself.

## What Not To Build First

Avoid these until the simpler stack is working:

- reinforcement learning policies,
- end-to-end deep models that directly output orders,
- a separate model for every symbol and every horizon before shared baselines exist,
- complex online learning without stable offline evaluation,
- a learned risk engine.

These approaches add variance, debugging difficulty, and false discovery risk too early.

## Recommended Build Order

### Phase 1. Single-venue, single-frequency baselines

Start with:

- one exchange,
- one or two symbols,
- and one bar interval plus a small set of microstructure aggregates.

Suggested first target:

- BTC-USD and optionally ETH-USD,
- 1-minute bars,
- rolling trade-flow and top-of-book features.

Deliverables:

- deterministic baselines,
- linear/logistic baselines,
- one boosted-tree model,
- fee-aware backtest.

### Phase 2. Better labels and validation

Add:

- multi-horizon return labels,
- event-based or threshold labels for tradeability,
- walk-forward validation,
- purged or embargoed CV when horizons overlap.

Deliverables:

- reliable offline scorecards,
- model comparison tables,
- leakage checks.

### Phase 3. Transformer forecaster

Add one sequence model trained on rolling windows of normalized features.

Deliverables:

- forecast outputs saved alongside tabular features,
- comparison against the boosted-tree baseline,
- ablation showing whether the sequence model adds signal.

### Phase 4. Hybrid decision layer

Train the meta-model on:

- tabular features,
- Transformer predictions,
- volatility and liquidity context,
- cost estimates.

Deliverables:

- trade/no-trade classifier,
- confidence score,
- rule-based position sizing driven by the model output.

### Phase 5. Regime awareness and portfolio extensions

Only after the single-asset setup is stable:

- add regime features,
- add model routing by regime,
- expand to more symbols,
- add portfolio constraints.

## Labeling Strategy

Use more than one label family. One label is rarely enough for a trading system.

Recommended initial labels:

- future log return at several horizons,
- direction label with a dead zone around zero,
- tradeability label: whether move magnitude exceeds estimated fees and slippage,
- future trend-state labels: uptrend, downtrend, or chop from rolling future log-price slope and path efficiency,
- triple-barrier labels: which profit/stop/no-hit outcome occurs first within the horizon,
- realized volatility label for risk scaling.

The most important label is usually not “will price go up,” but “is there enough edge after costs to trade.”

## Feature Strategy

Start with features that can be computed reliably from low-cost data:

- returns over several windows,
- rolling volatility,
- volume imbalance,
- signed trade flow,
- spread,
- top-of-book imbalance,
- candle range and momentum features,
- time-of-day or session context,
- recent fill or exposure state for execution-aware models.

Do not begin with raw full-depth order book tensors unless the ingestion and storage pipeline is already stable.

## Model Interfaces

Every model should expose the same high-level contract:

- `fit(dataset, config)`,
- `predict(dataset_slice)`,
- `predict_proba(dataset_slice)` when applicable,
- `save(path)`,
- `load(path)`.

Every training run should also emit:

- feature schema,
- label schema,
- train/validation/test windows,
- metrics,
- artifact path,
- model version identifier.

## Evaluation Standard

Evaluate models in layers:

- forecast metrics: MAE, RMSE, directional accuracy, calibration,
- decision metrics: precision, recall, hit rate, average trade expectancy,
- strategy metrics: Sharpe-like metrics, drawdown, turnover, fee drag, exposure,
- robustness metrics: walk-forward stability, sensitivity to costs, sensitivity to thresholds.

The acceptance bar is not “best validation loss.” The acceptance bar is stable out-of-sample performance after realistic costs.

## Practical Package Plan

The code should be organized around interfaces, not libraries.

Recommended package responsibilities:

- `src/quant/models/trees`: LightGBM, XGBoost, CatBoost wrappers,
- `src/quant/models/transformers`: sequence forecasters,
- `src/quant/models/ensembles`: meta-models and prediction combiners,
- `src/quant/training`: training loops, artifact writing, and experiment metadata,
- `src/quant/validation`: walk-forward splits, purging, embargo, and metrics.

Keep data transforms and feature generation out of model classes.

## Concrete First Implementation

If starting immediately, build this exact first stack:

1. Ridge regression for 5-minute forward return.
2. Logistic regression for up/down/no-trade classification.
3. LightGBM model on tabular features.
4. A rule-based position sizer with volatility targeting and max exposure caps.
5. A fee-aware backtest and walk-forward evaluator.

Only after that:

6. A Transformer forecaster for multi-horizon returns.
7. A LightGBM meta-model that combines tabular features and Transformer outputs.

### Trend-recognition extension

The next trend-focused layer is now a dedicated `histgb_trend_state` model trained on `future_trend_state_label_24_bar`.

Use it as a feature source:

- run tabular walk-forward validation for `histgb_trend_state`,
- stack `prediction`, `prediction_trend_state_margin`, `prediction_probability_downtrend`, `prediction_probability_chop`, `prediction_probability_uptrend`, `prediction_probability_max`, and `prediction_confidence`,
- train the hybrid direction meta-model with `baseline_core` plus `trend_state_oof`.

This keeps future trend labels out of the meta-model while giving the decision layer leak-safe trend-state probabilities.

### Executable trend stack

The next iteration should separate fast execution timing from slow regime context instead of forcing a single 24-bar trend model to do both.

- train `histgb_trend_state_fast` on `future_trend_state_label_6_bar` for short-horizon trade timing,
- keep `histgb_trend_state` on `future_trend_state_label_24_bar` as the slower regime prior,
- train `histgb_triple_barrier_fast` on `triple_barrier_label_6_bar` so after-cost barrier outcomes become a second fast execution signal,
- stack the three leak-safe OOF streams into `market_executable_trend_stacked`,
- compare a `direction_label_1_bar` meta-target against a `triple_barrier_label_6_bar` meta-target before promoting anything.

## Decision Rule

The best way to build the models for this project is:

- one venue first,
- one dataset definition first,
- one evaluation framework first,
- one strong tree baseline first,
- one sequence model second,
- one hybrid meta-model third.

That is enough complexity to be useful without making the project unmanageable.
