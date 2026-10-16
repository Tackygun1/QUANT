# Run Historical Replay

## Purpose

Replay saved supervised market rows through a trained model artifact at accelerated speed so the local viewer shows candles forming, paper entries, planned stops, take-profit levels, and realized exits.

## Inputs

- A saved model artifact
- A `training_datasets/market_supervised` source root
- A replay config like [historical_replay.example.json](/home/gofish/QUANT/configs/strategies/historical_replay.example.json)

## Outputs

```text
<replay_root>/features/training_datasets/market_supervised/...
<replay_root>/paper/signals/...
<replay_root>/paper/orders/...
<replay_root>/paper/ledger/...
<replay_root>/paper/state/...
<replay_root>/replay_summary.json
```

## Example

```bash
python3 scripts/trading/run_historical_replay.py \
  --config configs/strategies/historical_replay.example.json
```

Then serve the viewer against the replay workspace:

```bash
python3 scripts/data/run_market_viewer.py \
  --data-root data/replay/demo/features \
  --paper-root data/replay/demo/paper \
  --port 8770
```

## Notes

- This is a visualization tool, not a backtest replacement.
- The demo config intentionally uses faster replay and tighter TP/SL settings so trade actions are visible in the UI.
- Meta-model artifacts are supported as long as their supporting artifacts are reachable from the configured `artifact_search_root`.
