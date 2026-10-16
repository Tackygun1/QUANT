# Better Trend Recognition Models and Approaches

## Why This Note Exists

The current promoted model stack is profitable in some untouched windows, but its repeated weakest point has been `trend_recognition`, with `timing_churn` second. In practice that means:

- it is late to recognize bullish or bearish regime shifts,
- it can stay structurally biased the wrong way for too long,
- and it often makes too many small reallocations around uncertain trend state.

This note reviews model families that are better aligned to that problem than "just train a bigger forecaster."

## Main Conclusion

The best next move is not a deeper generic Transformer.

The best next move is a combination of:

1. explicit regime-state modeling,
2. a stronger multi-scale sequence backbone,
3. and regime-aware training and evaluation.

My ranking for this repo, based on the papers below and the current BTC `5m` hybrid-direction stack, is:

1. `HMM` or `HSMM` plus `BOCPD` regime features on top of the current hybrid direction model,
2. replace or supplement the current sequence forecaster with `TimeMixer` or `PatchTST`,
3. train with regime-aware objectives instead of pure ERM on overlapping windows.

That ranking is an inference from the literature plus the repo's measured failure mode. It is not a direct claim from any one paper.

## What Trend Recognition Actually Needs

For this project, "better trend recognition" should mean better performance on four separate tasks:

- detect whether the market is in `bull`, `bear`, or `chop`,
- estimate whether that state is likely to persist,
- detect when the current state is breaking,
- and decide when there is no trend edge worth trading after costs.

That is a different problem than pure one-step return regression. A model can be decent at next-bar direction and still be bad at regime recognition.

## Best Model Families

### 1. Explicit Regime Models

#### HMM

Hidden Markov Models are still one of the cleanest tools for regime detection. They are useful because they produce:

- filtered regime probabilities,
- transition probabilities,
- and a causal estimate of the current hidden state.

That is a good fit for a trading system that needs to distinguish trend, reversal, and chop states before making trade decisions. A finance-specific example shows HMM-based regime switching can improve allocation decisions relative to static models.

Best use in this repo:

- not as the final trader,
- but as a feature layer or routing layer for the existing LightGBM meta-model.

Recommended outputs:

- `p_bull`,
- `p_bear`,
- `p_chop`,
- regime entropy,
- probability of state transition next bar.

#### HSMM

The weakness of a plain HMM is that state durations are implicitly geometric. That is often unrealistic for market trends, because bull or bear phases can persist longer than a geometric dwell-time assumption suggests.

HSMMs fix that by modeling state duration explicitly. That matters for this repo because one of the current problems is not only recognizing a trend, but recognizing how long it is likely to keep running.

Best use in this repo:

- classify persistent trend states,
- estimate remaining trend duration,
- reduce premature flipping around regime boundaries.

My recommendation:

- if we add one classical regime model, prefer `HSMM` over plain `HMM` once the basic HMM baseline is in place.

#### BOCPD

Bayesian Online Changepoint Detection is not a trend model by itself. It is a state-transition detector. That makes it valuable here because the current model struggles most at regime breaks.

BOCPD gives:

- posterior changepoint probability,
- run-length distribution,
- and causal online change detection.

Best use in this repo:

- detect "trend just broke" events,
- suppress new entries immediately after uncertain regime transitions,
- and provide a run-length feature to the meta-model.

My recommendation:

- BOCPD should be added even if we do not adopt a full HMM or HSMM first.

### 2. Multi-Scale Neural Backbones

#### TimeMixer

TimeMixer is the most directly aligned neural architecture I found for this exact weakness. Its core premise is that time series carry different information at different sampling scales, and that fine-scale and coarse-scale components should be mixed differently. The paper explicitly separates microscopic and macroscopic information and uses decomposition plus multiscale mixing.

That matches the repo's problem almost perfectly:

- microstructure and short candles drive entry timing,
- coarse context drives trend state,
- and the current hybrid model appears weak at combining those scales.

My recommendation:

- `TimeMixer` is the best candidate to replace the current sequence forecaster if the goal is better trend recognition rather than just another forecaster.

#### PatchTST

PatchTST is also a strong fit. Its patching design lets the model look further back efficiently, which is useful for medium-horizon trend formation. The channel-independent design is also attractive for mixed feature groups like returns, volume, imbalance, and regime features.

Why it is useful here:

- it can see longer context without exploding attention cost,
- it is a stronger default than a generic time-series Transformer,
- and it is easier to adopt than a more exotic architecture.

My recommendation:

- `PatchTST` is the safest near-term neural replacement.
- If we want the lowest integration risk, start with PatchTST.
- If we want the architecture most tailored to multiscale trend recognition, start with TimeMixer.

#### TimesNet

TimesNet converts 1D temporal variation into 2D representations based on multiple periods. That is attractive for crypto because BTC has intraday periodicity, session structure, and repeating volatility patterns.

Why it may help:

- trend recognition is often confounded by periodic intraday effects,
- and TimesNet is specifically built around multi-period structure.

My recommendation:

- use `TimesNet` if we conclude the main missed trends are really missed periodic structures,
- otherwise rank it behind TimeMixer and PatchTST for this repo.

#### TFT

Temporal Fusion Transformer is still valuable because it combines local processing, long-range attention, and explicit feature selection and gating. It is not the newest model on this list, but it remains useful if interpretability matters.

Why it may help:

- variable selection could tell us whether the model is actually using trend features,
- gating can suppress noisy inputs in chop,
- and multi-horizon structure fits the current stack.

My recommendation:

- `TFT` is a good diagnostic model,
- but not my first pick for fixing the current weakness.

#### TiDE and N-BEATS

These are important because they are fast, strong, and simpler than many attention models.

- `TiDE` is attractive when speed and stability matter more than architecture novelty.
- `N-BEATS` is attractive when we want explicit trend and seasonality-style decomposition from a dense model.

My recommendation:

- use these as strong baselines in trend-recognition experiments,
- especially if larger Transformer-style models keep overfitting or under-trading.

### 3. OOD and Regime-Robust Training

The current project already has evidence that search-slice winners can fail on untouched holdouts. That makes distribution shift part of the problem, not just model capacity.

FOIL is relevant here because it frames time-series forecasting as an out-of-distribution problem and uses invariant learning across inferred environments. That idea is a good fit for this repo even if we do not implement FOIL exactly as published.

Best use in this repo:

- treat different date ranges or volatility regimes as environments,
- force the model to rely on features that remain predictive across regimes,
- and stop rewarding shortcuts that only work in one training window.

My recommendation:

- add regime-aware or environment-aware training after the next regime-model experiment,
- not before.

## Best Approach for This Repo

### Best Immediate Fix

Add an explicit regime layer before changing the whole neural stack.

Concrete first experiment:

1. Fit a `3`-state `HMM` on close returns, realized volatility, and maybe session VWAP distance.
2. Add `BOCPD` changepoint probability and run-length features.
3. Feed those features into the existing `lightgbm_hybrid_direction_meta`.
4. Re-run untouched holdouts and weakness diagnostics.

Why this is first:

- it directly targets the measured failure mode,
- it is cheap to train,
- and it will tell us quickly whether the problem is lack of regime representation or lack of sequence capacity.

### Best Medium-Term Upgrade

Replace the current sequence model with a multiscale backbone.

My ranking:

1. `TimeMixer`
2. `PatchTST`
3. `TimesNet`
4. `TFT`

That ranking is my inference from the papers plus the current repo behavior.

Why:

- TimeMixer is the most directly trend-aware by construction.
- PatchTST is the safest engineering step.
- TimesNet is compelling if periodicity is the main missed signal.
- TFT is strongest when interpretability is the main need.

### Best Long-Term Research Direction

Move from pure direction prediction to joint regime-direction modeling.

That means training a model to output both:

- regime state or regime probabilities,
- and conditional trade direction or edge.

A practical version for this repo would be:

- Head 1: `bull / bear / chop`
- Head 2: `trade_long / trade_short / no_trade`
- Head 3: expected regime persistence or transition probability

That is a better formulation for trend recognition than a single scalar return prediction.

## What I Would Not Do Next

I would not start with:

- a larger generic Transformer,
- more evolutionary search on the same architecture,
- end-to-end RL,
- or a separate deep model that tries to detect support and resistance directly as a first supervised target.

The current evidence says the bottleneck is regime representation and robustness, not a lack of search depth.

## Recommended Experiment Order

### Experiment A. Regime Features on the Existing Hybrid Stack

- Add `HMM` regime probabilities.
- Add `BOCPD` changepoint probability and run length.
- Re-run the current hybrid direction model without changing the execution layer.

Success criterion:

- better uptrend alignment,
- less one-sided structural bias,
- and improved untouched-window PnL without increasing churn.

### Experiment B. HMM vs HSMM

- Compare `HMM` and `HSMM` on the same features.
- Focus on regime-duration realism and trend persistence detection.

Success criterion:

- fewer false flips,
- better hold quality,
- and better trend-segment accuracy, not just bar accuracy.

### Experiment C. Replace the Sequence Forecaster

- Train `TimeMixer` and `PatchTST` on the same multi-horizon labels.
- Keep the meta-model and execution policy fixed.

Success criterion:

- better trend-recognition diagnostics,
- not just better RMSE.

### Experiment D. Regime-Aware Training

- Define temporal environments by date window, volatility bucket, or inferred regime.
- Apply a worst-group or invariant-style objective on top of the best candidate from Experiment C.

Success criterion:

- smaller gap between search-slice performance and untouched holdout performance.

## Metrics That Matter for Trend Recognition

Do not judge the next experiment mainly by RMSE.

Use:

- uptrend alignment rate,
- downtrend alignment rate,
- regime transition accuracy,
- average hold return by regime,
- cost share of gross return,
- and bias balance between long and short exposure.

Trend recognition is a regime problem first and a point-forecast problem second.

## Bottom Line

If I were choosing one next implementation for this repo, it would be:

1. `BOCPD` plus `HMM` regime features into the current hybrid direction model,
2. then `TimeMixer` as the next sequence backbone,
3. then regime-aware training.

That path is the best tradeoff between research value, implementation cost, and fit to the repo's actual failure mode.

## Sources

- PatchTST, OpenReview: https://openreview.net/forum?id=Jbdc0vTOcol
- TimeMixer, arXiv: https://arxiv.org/abs/2405.14616
- TimesNet, OpenReview: https://openreview.net/forum?id=ju_Uqw384Oq
- Temporal Fusion Transformer, Google Research: https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/
- TiDE, Google Research: https://research.google/pubs/long-horizon-forecasting-with-tide-time-series-dense-encoder/
- Bayesian Online Changepoint Detection, Adams and MacKay: https://lips.cs.princeton.edu/bibliography/adams2007changepoint/
- Regime-Switching Factor Investing with Hidden Markov Models: https://www.mdpi.com/1911-8074/13/12/311
- Stylized facts of financial time series and hidden semi-Markov models: https://doi.org/10.1016/j.csda.2006.07.021
- FOIL, ICML 2024: https://arxiv.org/abs/2406.09130
