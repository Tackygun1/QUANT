from __future__ import annotations

import unittest

from quant.utils.config import config_value


class ConfigValueTests(unittest.TestCase):
  def test_cli_zero_values_override_payload_defaults(self) -> None:
    payload = {
      "fee_bps": 5.0,
      "embargo_rows": 8,
      "enabled": True,
    }

    self.assertEqual(config_value(0.0, payload, "fee_bps", 1.0), 0.0)
    self.assertEqual(config_value(0, payload, "embargo_rows", 1), 0)
    self.assertFalse(config_value(False, payload, "enabled", True))

  def test_payload_empty_values_override_defaults(self) -> None:
    payload = {
      "providers": [],
      "dataset_name": "",
    }

    self.assertEqual(config_value(None, payload, "providers", ("coinbase",)), [])
    self.assertEqual(config_value(None, payload, "dataset_name", "market_supervised"), "")


if __name__ == "__main__":
  unittest.main()
