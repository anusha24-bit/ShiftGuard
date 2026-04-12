"""
Monthly vs Daily retraining cadence — Realistic P&L comparison.
Runs walk-forward for all 3 pairs at both cadences, then shows
realistic P&L with costs/leverage/taxes across 1m, 3m, 6m, 1y, 2y, 5y.

Usage:
    python src/models/cadence_pnl_compare.py
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

# Realistic cost model
CAPITAL_PER_PAIR = 10000
LEVERAGE = {'EURUSD': 20, 'GBPJPY': 10, 'XAUUSD': 10}
SPREAD = {'EURUSD': 0.00010, 'GBPJPY': 0.00020, 'XAUUSD': 0.00030}
COMMISSION = 0.00003
SLIPPAGE = 0.00002
SWAP = 0.000005
STOP_LOSS = 0.01
TAX_RATE = 0.30

HORIZONS = {
    '1 Month':  ('2026-03-01', '2026-04-01'),
    '3 Months': ('2026-01-01', '2026-04-01'),
    '6 Months': ('2025-10-01', '2026-04-01'),
    '1 Year':   ('2025-04-01', '2026-04-01'),
    '2 Years':  ('2024-04-01', '2026-04-01'),
    '5 Years':  ('2021-04-01', '2026-04-01'),
}

PAIRS = ['EURUSD', 'GBPJPY', 'XAUUSD']


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


def build_dataset(pair):
    path = os.path.join(DATA_DIR, 'price', f'{pair}_4h.csv')
    df = pd.read_csv(path, parse_dates=['datetime_utc'])
    df = df.sort_values('datetime_utc').reset_index(drop=True)
    df = compute_technical_features(df)
    df = compute_volatility_features(df)
    df = compute_macro_features(df, pair, DATA_DIR)
    df = compute_sentiment_features(df, pair, DATA_DIR)
    df = compute_regime_features(df)
    df = compute_adaptive_regime_labels(df)
    df = create_5class_regime(df)
    df['target_return'] = np.log(df['close'].shift(-1) / df['close'])
    df['target_dir'] = (df['target_return'] > 0).astype(int)
    df = df.iloc[260:].reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def run_walkforward(df, feature_cols, step_bars, cooldown_bars, vol_threshold, label):
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

    # Static ML baseline
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

        # Technical
        tech_signals = []
        for _, row in chunk.iterrows():
            sig = 0
            rsi = row.get('rsi_14', 50)
            macd = row.get('macd_hist', 0)
            if pd.isna(rsi) or pd.isna(macd):
                tech_signals.append(0)
                continue
            if rsi < 30:
                sig = 1
            elif rsi > 70:
                sig = -1
            elif macd > 0:
                sig = 1
            elif macd < 0:
                sig = -1
            tech_signals.append(sig)
        tech_signals = np.array(tech_signals)

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
                'tech_signal': tech_signals[i],
                'ml_signal': ml_signals[i],
                'sg_signal': sg_signals[i],
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
            print(f"      [{label}] step {step}/{total_steps}")

    return pd.DataFrame(all_records), retrain_count


def compute_realistic_pnl(trades_by_pair, pair, col):
    """Compute realistic P&L for a strategy column on a pair's trades."""
    lev = LEVERAGE[pair]
    sp = SPREAD[pair]
    s = trades_by_pair[col].values
    r = trades_by_pair['actual_return'].values
    tm = s != 0
    nt = int(tm.sum())
    if nt == 0:
        return {'trades': 0, 'wins': 0, 'after_tax': 0, 'stopped': 0}

    raw = s[tm] * r[tm]
    lev_pnl = raw * lev
    sl_lev = STOP_LOSS * lev
    capped = np.where(lev_pnl < -sl_lev, -sl_lev, lev_pnl)

    cost_per = (sp + COMMISSION + SLIPPAGE + SWAP) * lev
    net = capped - cost_per

    gross = CAPITAL_PER_PAIR * capped.sum()
    costs = CAPITAL_PER_PAIR * cost_per * nt
    pre_tax = gross - costs
    tax = max(0, pre_tax * TAX_RATE)
    after_tax = pre_tax - tax

    return {
        'trades': nt,
        'wins': int((net > 0).sum()),
        'after_tax': after_tax,
        'stopped': int((lev_pnl < -sl_lev).sum()),
    }


if __name__ == '__main__':
    # Step 1: Build datasets
    datasets = {}
    for pair in PAIRS:
        print(f"Building {pair} dataset...")
        datasets[pair] = build_dataset(pair)

    # Step 2: Run both cadences for all pairs
    cadences = {
        'Monthly': {'step': 180, 'cooldown': 0, 'vol_thresh': 3},
        'Daily':   {'step': 6,   'cooldown': 30, 'vol_thresh': 1},
    }

    # {cadence: {pair: (df_trades, retrain_count)}}
    results = {}
    for cname, cparams in cadences.items():
        results[cname] = {}
        for pair in PAIRS:
            df = datasets[pair]
            feature_cols = get_feature_cols(df)
            print(f"\n  Running {pair} — {cname} (step={cparams['step']})...")
            trades_df, rt = run_walkforward(
                df, feature_cols,
                step_bars=cparams['step'],
                cooldown_bars=cparams['cooldown'],
                vol_threshold=cparams['vol_thresh'],
                label=f'{pair}/{cname}',
            )
            trades_df['datetime_utc'] = pd.to_datetime(trades_df['datetime_utc'])
            results[cname][pair] = (trades_df, rt)
            print(f"    {pair}/{cname}: {len(trades_df)} bars, {rt} retrains")

    # Step 3: Compare across horizons
    portfolio_capital = CAPITAL_PER_PAIR * len(PAIRS)  # $30K total

    print(f"\n{'='*90}")
    print(f"  MONTHLY vs DAILY RETRAINING — REALISTIC P&L COMPARISON")
    print(f"  ($10,000 per pair = ${portfolio_capital:,} portfolio | costs + leverage + 30% tax)")
    print(f"{'='*90}")

    strategies = [
        ('Technical', 'tech_signal'),
        ('ML Direction', 'ml_signal'),
        ('ShiftGuard', 'sg_signal'),
    ]

    for hname, (start, end) in HORIZONS.items():
        print(f"\n{'='*90}")
        print(f"  {hname} ({start} → {end})")
        print(f"{'='*90}")

        print(f"\n  {'Strategy':<16}{'Cadence':<10}{'Trades':>8}{'Win%':>8}"
              f"{'After Tax':>14}{'Balance':>14}{'Result':>10}")
        print(f"  {'-'*78}")

        for sname, col in strategies:
            for cname in ['Monthly', 'Daily']:
                total_trades = 0
                total_wins = 0
                total_after_tax = 0
                total_stopped = 0

                for pair in PAIRS:
                    tdf, _ = results[cname][pair]
                    mask = (tdf['datetime_utc'] >= start) & (tdf['datetime_utc'] < end)
                    chunk = tdf[mask]
                    if len(chunk) == 0:
                        continue
                    r = compute_realistic_pnl(chunk, pair, col)
                    total_trades += r['trades']
                    total_wins += r['wins']
                    total_after_tax += r['after_tax']
                    total_stopped += r['stopped']

                wr = total_wins / total_trades * 100 if total_trades > 0 else 0
                balance = portfolio_capital + total_after_tax

                if balance <= 0:
                    status = 'BLOWN'
                elif total_after_tax > 0:
                    status = 'PROFIT'
                else:
                    status = 'LOSS'

                print(f"  {sname:<16}{cname:<10}{total_trades:>8}{wr:>7.1f}%"
                      f"  ${total_after_tax:>+12,.0f}  ${balance:>12,.0f}  {status:>8}")

            # blank line between strategies
            if sname != 'ShiftGuard':
                print()

    # Retrain count summary
    print(f"\n{'='*90}")
    print(f"  RETRAIN COUNTS")
    print(f"{'='*90}")
    print(f"  {'Pair':<10}{'Monthly':>10}{'Daily':>10}{'Delta':>10}")
    print(f"  {'-'*38}")
    for pair in PAIRS:
        m_rt = results['Monthly'][pair][1]
        d_rt = results['Daily'][pair][1]
        print(f"  {pair:<10}{m_rt:>10}{d_rt:>10}{d_rt - m_rt:>+10}")
