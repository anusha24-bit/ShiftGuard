"""
Performance Drift Monitor
Monitors XGBoost's prediction error stream using DDM (Drift Detection Method).
Catches when the model's accuracy degrades — validates the detection engine.
"""
import numpy as np
import pandas as pd


class DDM:
    """
    Drift Detection Method (Gama et al., 2004).
    Monitors error rate stream. Flags warning and drift levels
    based on standard deviation thresholds.
    """
    def __init__(self, warning_level=2.0, drift_level=3.0, min_samples=30):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        self.reset()

    def reset(self):
        self.n = 0
        self.p = 0  # error rate
        self.s = 0  # std dev
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.in_warning = False

    def update(self, error_value):
        """
        Update with a new error value (0 or 1 for classification, or continuous error).
        Returns: (is_drift, is_warning)
        """
        self.n += 1
        # Online mean and std
        self.p += (error_value - self.p) / self.n
        self.s = np.sqrt(self.p * (1 - self.p) / self.n) if self.n > 1 else 0

        if self.n < self.min_samples:
            return False, False

        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s

        is_warning = (self.p + self.s) >= (self.p_min + self.warning_level * self.s_min)
        is_drift = (self.p + self.s) >= (self.p_min + self.drift_level * self.s_min)

        if is_drift:
            self.reset()

        return is_drift, is_warning


def detect_performance_drift(predictions_df, window=30, drift_threshold=3.0):
    """
    Monitor model performance using DDM on directional errors.

    Args:
        predictions_df: DataFrame with columns [datetime_utc, actual, predicted]
        window: Rolling window for error tracking
        drift_threshold: DDM drift level (std devs above minimum)

    Returns:
        List of detected performance drift events
    """
    df = predictions_df.copy()
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

    # Directional error: 1 if wrong direction, 0 if correct
    df['dir_error'] = (np.sign(df['actual']) != np.sign(df['predicted'])).astype(int)

    # Absolute error
    df['abs_error'] = np.abs(df['actual'] - df['predicted'])

    # Rolling error rate
    df['rolling_error_rate'] = df['dir_error'].rolling(window).mean()
    df['rolling_mae'] = df['abs_error'].rolling(window).mean()

    # Run DDM
    ddm = DDM(warning_level=2.0, drift_level=drift_threshold)
    drifts = []
    warnings = []

    for i, row in df.iterrows():
        is_drift, is_warning = ddm.update(row['dir_error'])

        if is_drift:
            drifts.append({
                'datetime_utc': str(row['datetime_utc']),
                'type': 'performance_drift',
                'rolling_error_rate': round(row['rolling_error_rate'], 4) if pd.notna(row['rolling_error_rate']) else None,
                'rolling_mae': round(row['rolling_mae'], 6) if pd.notna(row['rolling_mae']) else None,
                'severity': 3,  # performance drifts are medium severity
            })
        elif is_warning:
            warnings.append({
                'datetime_utc': str(row['datetime_utc']),
                'type': 'performance_warning',
                'rolling_error_rate': round(row['rolling_error_rate'], 4) if pd.notna(row['rolling_error_rate']) else None,
            })

    return drifts, warnings
