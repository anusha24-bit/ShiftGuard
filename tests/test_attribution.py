from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.attribution import shap_analysis


class TestAttribution(unittest.TestCase):
    def test_group_attribution_returns_valid_percentages(self) -> None:
        shap_values = np.array([[1.0, 2.0, 0.5], [1.5, 2.5, 0.5]])
        feature_names = ['ema_feature', 'macro_feature', 'sess_london']
        groups = {'technical': ['ema_feature'], 'macro': ['macro_feature']}

        with patch.object(shap_analysis, 'FEATURE_GROUPS', groups):
            result = shap_analysis.compute_group_attribution(shap_values, feature_names)

        self.assertAlmostEqual(sum(result.values()), 100.0, places=1)
        self.assertEqual(next(iter(result)), 'macro')


if __name__ == '__main__':
    unittest.main()
