"""
Quick comparison: Monthly (180-bar) vs Daily (6-bar) step size.
Runs EURUSD only with both settings and compares overall + April 1-10 results.
"""
import sys, os
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.features.regime import compute_regime_features, compute_adaptive_regime_labels
from src.features.technical import compute_technical_features
from src.features.volatility import compute_volatility_features
from src.features.macro import compute_macro_features
from src.features.sentiment import compute_sentiment_features

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

META_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction',
             'volume', 'month', 'regime_label', 'target_regime', 'regime_changed',
             'target_transition_6', 'target_transition_12', 'target_transition_18',
             'bars_since_regime_change']

REGIME_PARAMS = {
    'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 300,
    'reg_alpha': 0.1, 'reg_lambda': 3.0, 'tree_method': 'hist',
    'random_state': 42, 'verbosity': 0, 'objective': 'multi:softprob',
    'num_class': 5, 'eval_metric': 'mlogloss',
}

TRAIN_BARS = 180 * 6
CONFIDENCE_THRESHOLD = 0.55


def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def create_5class_regime(df):
    ret_20 = df['close'].pct_change(20)
    trend_up = ret_20 > 0.005
    trend_down = ret_20 < -0.005
    high_vol = df['atr_pct_short'] > 60
    low_vol = df['atr_pct_short'] <= 60
    labels = np.full(len(df), 2)
    labels[trend_up & low_vol] = 0
    labels[trend_up & high_vol] = 1
    labels[trend_down & low_vol] = 3
    labels[trend_down & high_vol] = 4
    df['market_state'] = labels
    df['target_market_state'] = df['market_state'].shift(-1).ffill().astype(int)
    return df


def run_walkforward(df, feature_cols, step_bars, cooldown_bars, vol_threshold, label):
    """Run walk-forward with given step size. Returns records list + retrain count."""
    base_feature_cols = [c for c in feature_cols if c not in [
        'atr_pct_short', 'atr_pct_long', 'vol_ratio_5_60', 'vol_ratio_5_20',
        'vol_compressed', 'compression_duration', 'range_contraction',
        'hurst_exponent', 'consecutive_dir_bars', 'vol_divergence',
        'event_vol_interaction', 'range_expansion', 'abs_gap', 'gap_expansion',
        'market_state', 'target_market_state', 'target_dir',
    ]]

    cursor = TRAIN_BARS
    total = len(df)
    train_start = max(0, cursor - TRAIN_BARS)

    X_init = df.iloc[train_start:cursor][feature_cols].values
    y_regime_init = df.iloc[train_start:cursor]['target_market_state'].values

    # ML baseline (static)
    X_init_base = df.iloc[train_start:cursor][base_feature_cols].values
    dir_model = xgb.XGBClassifier(
        learning_rate=0.01, max_depth=5, n_estimators=500,
        reg_alpha=0.1, reg_lambda=5.0, tree_method='hist',
        random_state=42, verbosity=0, eval_metric='logloss',
    )
    dir_model.fit(X_init_base, df.iloc[train_start:cursor]['target_dir'].values, verbose=False)

    # Regime model
    unique_classes = np.unique(y_regime_init)
    X_regime = X_init.copy()
    y_regime = y_regime_init.copy()
    for cls in range(5):
        if cls not in unique_classes:
            idx = np.random.randint(0, len(X_regime))
            X_regime = np.vstack([X_regime, X_regime[idx:idx+1]])
            y_regime = np.append(y_regime, cls)
    regime_model = xgb.XGBClassifier(**REGIME_PARAMS)
    regime_model.fit(X_regime, y_regime, verbose=False)

    retrain_count = 0
    bars_since_retrain = cooldown_bars
    all_records = []
    step = 0
    total_steps = (total - TRAIN_BARS) // step_bars

    while cursor + step_bars <= total:
        step_end = cursor + step_bars
        chunk = df.iloc[cursor:step_end]
        X_chunk = chunk[feature_cols].values
        actual_returns = chunk['target_return'].values

        # ML
        ml_pred = dir_model.predict(chunk[base_feature_cols].values)
        ml_signals = np.where(ml_pred == 1, 1, -1)

        # ShiftGuard
        regime_probs = regime_model.predict_proba(X_chunk)
        regime_pred = regime_model.predict(X_chunk)
        regime_conf = regime_probs.max(axis=1)
        sg_signals = np.zeros(len(X_chunk))
        for i in range(len(X_chunk)):
            if regime_conf[i] < CONFIDENCE_THRESHOLD:
                continue
            if chunk['bars_since_regime_change'].iloc[i] < 3:
                continue
            if regime_pred[i] in [0, 1]:
                sg_signals[i] = 1
            elif regime_pred[i] in [3, 4]:
                sg_signals[i] = -1

        for i in range(len(actual_returns)):
            all_records.append({
                'datetime_utc': chunk['datetime_utc'].iloc[i],
                'actual_return': actual_returns[i],
                'ml_signal': ml_signals[i],
                'sg_signal': sg_signals[i],
                'regime': regime_pred[i],
                'regime_confidence': regime_conf[i],
            })

        # Retrain triggers
        should_retrain = False
        if chunk['regime_changed'].sum() > 0:
            should_retrain = True
        if 'vol_ratio' in chunk.columns:
            if (chunk['vol_ratio'] > 2.5).sum() >= vol_threshold:
                should_retrain = True
        if len(all_records) >= 60:
            recent = all_records[-60:]
            recent_sg = [(r['sg_signal'] * r['actual_return']) > 0
                         for r in recent if r['sg_signal'] != 0]
            if len(recent_sg) >= 10 and sum(recent_sg) / len(recent_sg) < 0.45:
                should_retrain = True
        if 'event_surprise' in chunk.columns:
            big = (chunk['event_surprise'].abs() > chunk['event_surprise'].abs().quantile(0.95)
                   if len(chunk) > 5 else False)
            if isinstance(big, pd.Series) and big.sum() > 0:
                should_retrain = True

        if should_retrain and bars_since_retrain >= cooldown_bars:
            ts = max(0, step_end - TRAIN_BARS)
            X_r = df.iloc[ts:step_end][feature_cols].values
            y_r = df.iloc[ts:step_end]['target_market_state'].values
            for cls in range(5):
                if cls not in np.unique(y_r):
                    idx = np.random.randint(0, len(X_r))
                    X_r = np.vstack([X_r, X_r[idx:idx+1]])
                    y_r = np.append(y_r, cls)
            regime_model = xgb.XGBClassifier(**REGIME_PARAMS)
            regime_model.fit(X_r, y_r, verbose=False)
            retrain_count += 1
            bars_since_retrain = 0

        bars_since_retrain += step_bars
        step += 1
        cursor = step_end

        if step % 500 == 0:
            print(f"    [{label}] step {step}/{total_steps}")

    return all_records, retrain_count


def evaluate(records, label):
    df = pd.DataFrame(records)
    results = {}
    for name, col in [('ML', 'ml_signal'), ('ShiftGuard', 'sg_signal')]:
        sig = df[col].values
        ret = df['actual_return'].values
        mask = sig != 0
        n = mask.sum()
        if n == 0:
            results[name] = {'wr': 0, 'n': 0, 'pf': 0, 'pnl': 0}
            continue
        pnl = sig[mask] * ret[mask]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        wr = len(wins) / n * 100
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99
        results[name] = {
            'wr': round(wr, 1),
            'n': int(n),
            'trade_pct': round(n / len(df) * 100, 1),
            'pf': round(min(float(pf), 99), 2),
            'pnl': round(float(pnl.sum() * 100), 3),
        }
    # April subset
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    apr = df[(df['datetime_utc'] >= '2026-04-01') & (df['datetime_utc'] < '2026-04-11')]
    apr_results = {}
    for name, col in [('ML', 'ml_signal'), ('ShiftGuard', 'sg_signal')]:
        sig = apr[col].values
        ret = apr['actual_return'].values
        mask = sig != 0
        n = mask.sum()
        if n == 0:
            apr_results[name] = {'wr': 0, 'n': 0, 'pf': 0, 'pnl': 0}
            continue
        pnl = sig[mask] * ret[mask]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        wr = len(wins) / n * 100 if n > 0 else 0
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99
        apr_results[name] = {
            'wr': round(wr, 1),
            'n': int(n),
            'pf': round(min(float(pf), 99), 2),
            'pnl': round(float(pnl.sum() * 100), 3),
        }
    return results, apr_results


if __name__ == '__main__':
    print("Building EURUSD dataset...")
    path = os.path.join(DATA_DIR, 'price', 'EURUSD_4h.csv')
    df = pd.read_csv(path, parse_dates=['datetime_utc'])
    df = df.sort_values('datetime_utc').reset_index(drop=True)
    df = compute_technical_features(df)
    df = compute_volatility_features(df)
    df = compute_macro_features(df, 'EURUSD', DATA_DIR)
    df = compute_sentiment_features(df, 'EURUSD', DATA_DIR)
    df = compute_regime_features(df)
    df = compute_adaptive_regime_labels(df)
    df = create_5class_regime(df)
    df['target_return'] = np.log(df['close'].shift(-1) / df['close'])
    df['target_dir'] = (df['target_return'] > 0).astype(int)
    df = df.iloc[260:].reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)
    feature_cols = get_feature_cols(df)
    print(f"Rows: {len(df)}, Features: {len(feature_cols)}\n")

    # --- Monthly ---
    print("Running MONTHLY cadence (step=180, no cooldown needed)...")
    monthly_recs, monthly_rt = run_walkforward(df, feature_cols, step_bars=180,
                                                cooldown_bars=0, vol_threshold=3, label='Monthly')
    monthly_all, monthly_apr = evaluate(monthly_recs, 'Monthly')

    # --- Daily ---
    print("\nRunning DAILY cadence (step=6, cooldown=30)...")
    daily_recs, daily_rt = run_walkforward(df, feature_cols, step_bars=6,
                                            cooldown_bars=30, vol_threshold=1, label='Daily')
    daily_all, daily_apr = evaluate(daily_recs, 'Daily')

    # --- Compare ---
    print(f"\n{'='*70}")
    print("EURUSD — MONTHLY vs DAILY Retraining Cadence")
    print(f"{'='*70}")

    print(f"\n  OVERALL (full dataset)")
    print(f"  {'Metric':<20}{'Monthly':>15}{'Daily':>15}{'Delta':>12}")
    print(f"  {'-'*60}")
    for name in ['ShiftGuard']:
        m = monthly_all[name]
        d = daily_all[name]
        print(f"  {'Win Rate':<20}{m['wr']:>14.1f}%{d['wr']:>14.1f}%{d['wr']-m['wr']:>+11.1f}%")
        print(f"  {'Trades':<20}{m['n']:>15}{d['n']:>15}{d['n']-m['n']:>+12}")
        print(f"  {'Trade %':<20}{m['trade_pct']:>14.1f}%{d['trade_pct']:>14.1f}%{d['trade_pct']-m['trade_pct']:>+11.1f}%")
        print(f"  {'Profit Factor':<20}{m['pf']:>15.2f}{d['pf']:>15.2f}{d['pf']-m['pf']:>+12.2f}")
        print(f"  {'Total PnL':<20}{m['pnl']:>+14.3f}%{d['pnl']:>+14.3f}%{d['pnl']-m['pnl']:>+11.3f}%")
    print(f"  {'Retrains':<20}{monthly_rt:>15}{daily_rt:>15}{daily_rt-monthly_rt:>+12}")

    print(f"\n  APRIL 1-10, 2026")
    print(f"  {'Metric':<20}{'Monthly':>15}{'Daily':>15}{'Delta':>12}")
    print(f"  {'-'*60}")
    for name in ['ShiftGuard']:
        m = monthly_apr[name]
        d = daily_apr[name]
        print(f"  {'Win Rate':<20}{m['wr']:>14.1f}%{d['wr']:>14.1f}%{d['wr']-m['wr']:>+11.1f}%")
        print(f"  {'Trades':<20}{m['n']:>15}{d['n']:>15}{d['n']-m['n']:>+12}")
        print(f"  {'Profit Factor':<20}{m['pf']:>15.2f}{d['pf']:>15.2f}{d['pf']-m['pf']:>+12.2f}")
        print(f"  {'Total PnL':<20}{m['pnl']:>+14.3f}%{d['pnl']:>+14.3f}%{d['pnl']-m['pnl']:>+11.3f}%")
