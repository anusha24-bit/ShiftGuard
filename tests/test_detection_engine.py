from __future__ import annotations

import unittest
import pandas as pd

from src.detection.engine import filter_shifts_to_prediction_window


class TestDetectionEngine(unittest.TestCase):
    def test_filter_shifts_to_prediction_window_keeps_only_prediction_period(self) -> None:
        shifts = [
            {'datetime_utc': '2020-12-31T20:00:00', 'type': 'scheduled'},
            {'datetime_utc': '2021-01-02T00:00:00', 'type': 'unexpected'},
            {'datetime_utc': '2021-02-15T00:00:00', 'type': 'scheduled'},
        ]
        prediction_window = (pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-31'))

        filtered = filter_shifts_to_prediction_window(shifts, prediction_window)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['type'], 'unexpected')


if __name__ == '__main__':
    unittest.main()
