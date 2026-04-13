from __future__ import annotations

import unittest
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import run_pipeline


class TestRunPipeline(unittest.TestCase):
    def test_has_review_decisions_requires_selected_pair_approvals(self) -> None:
        workspace_tmp = Path.cwd() / 'tests_tmp'
        workspace_tmp.mkdir(exist_ok=True)
        tmpdir = workspace_tmp / f'pipeline_{uuid.uuid4().hex}'
        tmpdir.mkdir()
        try:
            decisions_dir = Path(tmpdir)
            pd.DataFrame(
                [{'datetime_utc': '2021-06-01T00:00:00', 'decision': 'auto_confirm'}]
            ).to_csv(decisions_dir / 'EURUSD_decisions.csv', index=False)

            with patch.object(run_pipeline, 'DECISIONS_DIR', str(decisions_dir)):
                self.assertTrue(run_pipeline.has_review_decisions(['EURUSD']))
                self.assertFalse(run_pipeline.has_review_decisions(['EURUSD', 'GBPJPY']))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
