from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.monitoring.market_viewer import ViewerConfig, run_market_viewer


def main() -> int:
  args = _build_parser().parse_args()
  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
  run_market_viewer(
    ViewerConfig(
      data_root=args.data_root,
      paper_root=args.paper_root,
      host=args.host,
      port=args.port,
    )
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run a local UI for market candles and volume.")
  parser.add_argument("--data-root", type=Path, default=Path("data/features"))
  parser.add_argument("--paper-root", type=Path, default=Path("data/paper"))
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--log-level", default="INFO")
  return parser
