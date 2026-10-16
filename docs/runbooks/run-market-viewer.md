# Run Market Viewer

## Purpose

Serve a local browser UI that plots candlesticks and volume from the joined supervised dataset, plus live paper-trading entries, stops, take-profit levels, and exit hits.

## Example

```bash
python3 scripts/data/run_market_viewer.py --data-root data/features
```

Then open:

```text
http://127.0.0.1:8765
```

## Notes

- The viewer expects `training_datasets/market_supervised` under the given data root.
- If `data/paper` is available, the viewer overlays paper signals, planned stops and take-profits, realized exits, equity, and position history.
- The browser UI supports auto-refresh for live monitoring.
- It uses Plotly from a CDN in the browser.
- The plotted rows come from the joined supervised dataset, so labels and candle data stay aligned.
