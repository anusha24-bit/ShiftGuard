from __future__ import annotations

import unittest
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.retraining import selective


class TestSelectiveRetraining(unittest.TestCase):
    def test_reject_all_decisions_returns_no_shift_events(self) -> None:
        workspace_tmp = Path.cwd() / 'tests_tmp'
        workspace_tmp.mkdir(exist_ok=True)
        tmpdir = workspace_tmp / f'retraining_{uuid.uuid4().hex}'
        tmpdir.mkdir()
        try:
            base = Path(tmpdir)
            detection_dir = base / 'detection'
            decisions_dir = base / 'decisions'
            attribution_dir = base / 'attribution'
            detection_dir.mkdir()
            decisions_dir.mkdir()
            attribution_dir.mkdir()

            pd.DataFrame(
                [{'datetime_utc': '2021-06-01T00:00:00', 'type': 'scheduled', 'severity': 4}]
            ).to_csv(detection_dir / 'EURUSD_shifts.csv', index=False)

            pd.DataFrame(
                [{'datetime_utc': '2021-06-01T00:00:00', 'decision': 'reject', 'notes': 'false alarm'}]
            ).to_csv(decisions_dir / 'EURUSD_decisions.csv', index=False)

            with patch.object(selective, 'DETECTION_DIR', str(detection_dir)), \
                 patch.object(selective, 'DECISIONS_DIR', str(decisions_dir)), \
                 patch.object(selective, 'ATTRIBUTION_DIR', str(attribution_dir)):
                events = selective.get_shift_events('EURUSD')

            self.assertTrue(events.empty)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
