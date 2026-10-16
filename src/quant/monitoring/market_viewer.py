from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from quant.execution.historical_replay import (
  load_replay_control,
  replay_control_path,
  replay_status_path,
  save_replay_control,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ViewerConfig:
  data_root: Path = Path("data/features")
  paper_root: Path = Path("data/paper")
  host: str = "127.0.0.1"
  port: int = 8765


def run_market_viewer(config: ViewerConfig) -> None:
  html_path = Path(__file__).with_name("static") / "market_viewer.html"
  html = html_path.read_text(encoding="utf-8")
  series_index = build_series_index(config.data_root)

  class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
      parsed = urlparse(self.path)
      if parsed.path == "/":
        self._send_html(html)
        return
      if parsed.path == "/api/series":
        self._send_json({"series": series_index})
        return
      if parsed.path == "/api/data":
        params = parse_qs(parsed.query)
        provider = _first_param(params, "provider")
        symbol = _first_param(params, "symbol")
        feature_date = _first_param(params, "date")
        if not provider or not symbol or not feature_date:
          self._send_json({"error": "provider, symbol, and date are required"}, status=HTTPStatus.BAD_REQUEST)
          return
        payload = load_series_payload(config.data_root, config.paper_root, provider, symbol, feature_date)
        self._send_json(payload)
        return
      if parsed.path == "/api/replay-control":
        self._send_json(load_replay_runtime_payload(config.data_root))
        return

      self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
      parsed = urlparse(self.path)
      if parsed.path != "/api/replay-control":
        self.send_error(HTTPStatus.NOT_FOUND)
        return

      replay_root = resolve_replay_root(config.data_root)
      if replay_root is None:
        self._send_json({"available": False, "error": "Replay controls are unavailable for this data root."}, status=HTTPStatus.NOT_FOUND)
        return

      content_length = int(self.headers.get("Content-Length", "0"))
      raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
      try:
        payload = json.loads(raw_body.decode("utf-8"))
      except json.JSONDecodeError:
        self._send_json({"error": "Invalid JSON payload."}, status=HTTPStatus.BAD_REQUEST)
        return

      control_path = replay_control_path(replay_root)
      current = load_replay_control(control_path, default_delay=0.15)
      updated = save_replay_control(
        control_path,
        step_delay_seconds=float(payload.get("step_delay_seconds", current.get("step_delay_seconds", 0.15))),
        paused=bool(payload.get("paused", current.get("paused", False))),
      )
      runtime_payload = load_replay_runtime_payload(config.data_root)
      runtime_payload["control"] = updated
      self._send_json(runtime_payload)

    def log_message(self, format: str, *args: object) -> None:
      LOGGER.info("%s - %s", self.address_string(), format % args)

    def _send_html(self, html_text: str) -> None:
      encoded = html_text.encode("utf-8")
      self.send_response(HTTPStatus.OK)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(encoded)))
      self.end_headers()
      self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
      encoded = json.dumps(payload).encode("utf-8")
      self.send_response(status)
      self.send_header("Content-Type", "application/json; charset=utf-8")
      self.send_header("Content-Length", str(len(encoded)))
      self.end_headers()
      self.wfile.write(encoded)

  server = ThreadingHTTPServer((config.host, config.port), Handler)
  LOGGER.info("Market viewer running at http://%s:%s", config.host, config.port)
  LOGGER.info("Using data root %s", config.data_root)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    LOGGER.info("Shutting down market viewer.")
  finally:
    server.server_close()


def build_series_index(data_root: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  dataset_root = data_root / "training_datasets" / "market_supervised"
  if not dataset_root.exists():
    return rows

  for path in sorted(dataset_root.rglob("*.csv")):
    parts = path.relative_to(dataset_root).parts
    if len(parts) != 5:
      continue
    provider, symbol, year, month, filename = parts
    feature_date = f"{year}-{month}-{Path(filename).stem}"
    rows.append(
      {
        "provider": provider,
        "symbol": symbol,
        "date": feature_date,
        "path": str(path),
      }
    )
  return rows


def load_series_payload(data_root: Path, paper_root: Path, provider: str, symbol: str, feature_date: str) -> dict[str, Any]:
  year, month, day = feature_date.split("-")
  dataset_path = (
    data_root
    / "training_datasets"
    / "market_supervised"
    / provider
    / symbol
    / year
    / month
    / f"{day}.csv"
  )
  if not dataset_path.exists():
    return {"provider": provider, "symbol": symbol, "date": feature_date, "rows": [], "summary": {}, "paper": {}}

  with dataset_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

  candles = [
    {
      "time": row["candle_start"],
      "open": _to_float(row.get("open")),
      "high": _to_float(row.get("high")),
      "low": _to_float(row.get("low")),
      "close": _to_float(row.get("close")),
      "volume": _to_float(row.get("volume")),
      "trade_count": _to_float(row.get("trades_window_count")),
      "mid_price": _to_float(row.get("top_window_mid_last")),
      "spread": _to_float(row.get("top_window_spread_mean")),
      "spread_bps": _to_float(row.get("top_window_spread_mean_bps")),
      "forward_return_1_bar_bps": _to_float(row.get("forward_return_1_bar_bps")),
      "direction_label_1_bar": row.get("direction_label_1_bar", ""),
      "tradeable_label_1_bar": row.get("tradeable_label_1_bar", ""),
      "has_any_forward_label": row.get("has_any_forward_label", ""),
    }
    for row in rows
  ]

  close_values = [value for value in (_to_float(row.get("close")) for row in rows) if value is not None]
  volume_values = [value for value in (_to_float(row.get("volume")) for row in rows) if value is not None]
  labeled_rows = sum(1 for row in rows if row.get("has_any_forward_label") == "true")

  summary = {
    "row_count": len(rows),
    "labeled_row_count": labeled_rows,
    "first_time": candles[0]["time"] if candles else "",
    "last_time": candles[-1]["time"] if candles else "",
    "last_close": close_values[-1] if close_values else None,
    "high_max": max((candle["high"] for candle in candles if candle["high"] is not None), default=None),
    "low_min": min((candle["low"] for candle in candles if candle["low"] is not None), default=None),
    "total_volume": sum(volume_values) if volume_values else None,
  }
  paper_payload = load_paper_payload(paper_root, provider, symbol, feature_date)
  replay_payload = load_replay_runtime_payload(data_root)
  return {
    "provider": provider,
    "symbol": symbol,
    "date": feature_date,
    "rows": candles,
    "summary": summary,
    "paper": paper_payload,
    "replay": replay_payload,
  }


def load_paper_payload(paper_root: Path, provider: str, symbol: str, feature_date: str) -> dict[str, Any]:
  if not paper_root.exists():
    return {
      "signals": [],
      "orders": [],
      "ledger": [],
      "state": {},
      "summary": {},
    }

  signals = _load_jsonl_day_rows(paper_root / "signals", provider, symbol, feature_date)
  orders = _load_jsonl_day_rows(paper_root / "orders", provider, symbol, feature_date)
  ledger = _load_jsonl_day_rows(paper_root / "ledger", provider, symbol, feature_date)
  state = _load_latest_state(paper_root / "state" / provider / symbol)
  latest_ledger = ledger[-1] if ledger else {}
  latest_signal = signals[-1] if signals else {}
  take_profit_hits = sum(1 for row in ledger if str(row.get("exit_reason", "")).startswith("take_profit"))
  stop_loss_hits = sum(1 for row in ledger if str(row.get("exit_reason", "")).startswith("stop_loss"))
  active_trade_signals = sum(1 for row in signals if row.get("trade_plan_status") == "active")

  summary = {
    "signal_count": len(signals),
    "order_count": len(orders),
    "ledger_count": len(ledger),
    "take_profit_hit_count": take_profit_hits,
    "stop_loss_hit_count": stop_loss_hits,
    "active_trade_signal_count": active_trade_signals,
    "latest_model_name": latest_signal.get("model_name") or latest_ledger.get("model_name") or "",
    "latest_prediction": _to_float(_first_non_empty(latest_signal.get("prediction"), latest_ledger.get("prediction"))),
    "latest_target_position": _to_float(_first_non_empty(latest_signal.get("target_position"), state.get("target_position"))),
    "latest_equity": _to_float(_first_non_empty(latest_ledger.get("equity"), state.get("equity"))),
    "latest_net_return": _to_float(latest_ledger.get("net_return")),
    "latest_take_profit_price": _to_float(_first_non_empty(latest_signal.get("take_profit_price"), state.get("take_profit_price"))),
    "latest_stop_loss_price": _to_float(_first_non_empty(latest_signal.get("stop_loss_price"), state.get("stop_loss_price"))),
    "latest_exit_reason": _first_non_empty(latest_ledger.get("exit_reason"), state.get("exit_reason")) or "",
  }
  return {
    "signals": signals,
    "orders": orders,
    "ledger": ledger,
    "state": state,
    "summary": summary,
  }


def load_replay_runtime_payload(data_root: Path) -> dict[str, Any]:
  replay_root = resolve_replay_root(data_root)
  if replay_root is None:
    return {"available": False, "control": {}, "status": {}}

  control_path = replay_control_path(replay_root)
  status_path = replay_status_path(replay_root)
  control = load_replay_control(control_path, default_delay=0.15) if control_path.exists() else {}
  status = _load_json_file(status_path) if status_path.exists() else {}
  return {
    "available": bool(control or status),
    "control": control,
    "status": status,
    "replay_root": str(replay_root),
  }


def resolve_replay_root(data_root: Path) -> Path | None:
  for candidate in (data_root, data_root.parent):
    control_path = replay_control_path(candidate)
    status_path = replay_status_path(candidate)
    if control_path.exists() or status_path.exists():
      return candidate
  return None


def _to_float(value: str | None) -> float | None:
  if value in (None, ""):
    return None
  try:
    return float(value)
  except ValueError:
    return None


def _first_param(params: dict[str, list[str]], key: str) -> str:
  values = params.get(key, [])
  return values[0] if values else ""


def _load_jsonl_day_rows(root: Path, provider: str, symbol: str, feature_date: str) -> list[dict[str, Any]]:
  year, month, day = feature_date.split("-")
  series_root = root / provider / symbol / year / month / day
  if not series_root.exists():
    return []

  rows: list[dict[str, Any]] = []
  for path in sorted(series_root.glob("*.jsonl")):
    with path.open("r", encoding="utf-8") as handle:
      for line in handle:
        line = line.strip()
        if not line:
          continue
        rows.append(json.loads(line))
  rows.sort(key=lambda row: str(row.get("generated_at", "")))
  return rows


def _load_latest_state(root: Path) -> dict[str, Any]:
  if not root.exists():
    return {}

  states = sorted(root.glob("*.json"))
  if not states:
    return {}
  return json.loads(states[-1].read_text(encoding="utf-8"))


def _load_json_file(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _first_non_empty(*values: Any) -> str | None:
  for value in values:
    if value not in (None, ""):
      return str(value)
  return None
