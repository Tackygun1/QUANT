from __future__ import annotations

import unittest

from quant.validation.leakage import effective_purge_rows, infer_target_horizon_rows
from quant.validation.walk_forward import expanding_window_splits


class ValidationLeakageTests(unittest.TestCase):
  def test_infers_largest_target_horizon(self) -> None:
    self.assertEqual(
      infer_target_horizon_rows(
        (
          "forward_return_1_bar",
          "future_trend_state_label_24_bar",
          "triple_barrier_label_6_bar",
        )
      ),
      24,
    )

  def test_effective_purge_rows_never_undershoots_target_horizon(self) -> None:
    self.assertEqual(effective_purge_rows(None, "forward_return_96_bar"), 96)
    self.assertEqual(effective_purge_rows(8, "forward_return_96_bar"), 96)
    self.assertEqual(effective_purge_rows(128, "forward_return_96_bar"), 128)

  def test_expanding_split_supports_purged_training_tail(self) -> None:
    splits = expanding_window_splits(
      12,
      initial_train_rows=5,
      validation_rows=2,
      step_rows=2,
      purge_rows=2,
    )

    self.assertEqual(splits[0], ([0, 1, 2], [5, 6]))


if __name__ == "__main__":
  unittest.main()
