"""
Unexpected Shift Detector
Detects distribution shifts on non-event days using ADWIN.
These are shifts with no calendar match — flash crashes, geopolitical shocks, regime changes.
"""
import numpy as np
import pandas as pd
from river.drift import ADWIN


def detect_unexpected_shifts(df, feature_cols, calendar_dates=None, delta=0.002):
    """
    Run ADWIN on key feature streams to detect unexpected shifts.

    ADWIN maintains a variable-length window and detects when the mean
    of the window has statistically changed.

    Args:
        df: Feature dataframe with datetime_utc column
        feature_cols: Feature columns to monitor
        calendar_dates: Set of dates (str YYYY-MM-DD) with scheduled events — shifts
                        on these dates are classified as scheduled, not unexpected
        delta: ADWIN sensitivity (lower = more sensitive, default 0.002)

    Returns:
        List of detected shift dicts
    """
    df = df.copy()
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    df['date_str'] = df['datetime_utc'].dt.date.astype(str)

    if calendar_dates is None:
        calendar_dates = set()

    # Select key features to monitor (not all — too noisy)
    # Focus on volatility and return features that react fastest to shifts
    monitor_features = [c for c in [
        'log_return', 'atr_14', 'rolling_std_5', 'vol_ratio',
        'bb_width', 'gk_vol', 'vix_change', 'hl_range',
    ] if c in feature_cols]

    if not monitor_features:
        monitor_features = feature_cols[:8]  # fallback

    shifts = []
    seen_dates = set()  # deduplicate shifts on same date

    for feat in monitor_features:
        adwin = ADWIN(delta=delta)
        values = df[feat].fillna(0).values

        for i in range(len(values)):
            adwin.update(values[i])
            in_drift = adwin.drift_detected

            if in_drift:
                dt = df['datetime_utc'].iloc[i]
                date_str = df['date_str'].iloc[i]

                # Skip if this is a scheduled event date
                if date_str in calendar_dates:
                    continue

                # Deduplicate: one shift per date per feature
                key = f"{date_str}_{feat}"
                if key in seen_dates:
                    continue
                seen_dates.add(key)

                shifts.append({
                    'datetime_utc': str(dt),
                    'date': date_str,
                    'type': 'unexpected',
                    'trigger_feature': feat,
                    'trigger_value': round(float(values[i]), 6),
                    'adwin_width': adwin.width,
                })

    # Aggregate: group by date, count how many features triggered
    if not shifts:
        return []

    shifts_df = pd.DataFrame(shifts)
    aggregated = []

    for date, group in shifts_df.groupby('date'):
        n_features = len(group)
        severity = min(5, n_features)  # more features triggered = higher severity
        trigger_features = group['trigger_feature'].tolist()

        aggregated.append({
            'datetime_utc': group['datetime_utc'].iloc[0],
            'type': 'unexpected',
            'severity': severity,
            'n_features_triggered': n_features,
            'trigger_features': '; '.join(trigger_features),
        })

    return aggregated
