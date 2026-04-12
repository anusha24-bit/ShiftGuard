"""
Two-Phase Split:
  Phase 1: Train 2015-2018, Test 2019-2020
  Phase 2: Retrain 2021-2024, Test 2025+

Compares:
  - Static: Train Phase 1 only, never retrain, test on both phases
  - ShiftGuard: Train Phase 1, retrain at Phase 2, test on both phases
  - ML Direction: Same splits, no regime filtering

Usage:
    python src/models/two_phase_split.py
"""
import sys, os, json
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.features.technical import compute_technical_features
from src.features.volatility import compute_volatility_features
from src.features.macro import compute_macro_features
from src.features.sentiment import compute_sentiment_features
from src.features.regime import compute_regime_features, compute_adaptive_regime_labels

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'two_phase')
os.makedirs(RESULTS_DIR, exist_ok=True)

META_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction',
             'volume', 'month', 'regime_label', 'target_regime', 'regime_changed',
             'target_transition_6', 'target_transition_12', 'target_transition_18',
             'bars_since_regime_change', 'market_state', 'target_market_state', 'target_dir']

REGIME_PARAMS = {
    'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 300,
    'reg_alpha': 0.1, 'reg_lambda': 3.0, 'tree_method': 'hist',
    'random_state': 42, 'verbosity': 0, 'objective': 'multi:softprob',
    'num_class': 5, 'eval_metric': 'mlogloss',
}

ML_PARAMS = {
    'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500,
    'reg_alpha': 0.1, 'reg_lambda': 5.0, 'tree_method': 'hist',
    'random_state': 42, 'verbosity': 0, 'eval_metric': 'logloss',
}

capital = 10000
CONF = 0.55


def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def create_5class_regime(df):
    ret_20 = df['close'].pct_change(20)
    trend_up = ret_20 > 0.005
    trend_down = ret_20 < -0.005
    high_vol = df['atr_pct_short'] > 60
    labels = np.full(len(df), 2)
    labels[trend_up & ~high_vol] = 0
    labels[trend_up & high_vol] = 1
    labels[trend_down & ~high_vol] = 3
    labels[trend_down & high_vol] = 4
    df['market_state'] = labels
    df['target_market_state'] = df['market_state'].shift(-1).ffill().astype(int)
    return df


def ensure_classes(X, y, n=5):
    for cls in range(n):
        if cls not in np.unique(y):
            idx = np.random.randint(0, len(X))
            X = np.vstack([X, X[idx:idx+1]])
            y = np.append(y, cls)
    return X, y


def get_signals_tech(test_df):
    signals = []
    for _, row in test_df.iterrows():
        rsi = row.get('rsi_14', 50)
        macd = row.get('macd_hist', 0)
        if pd.isna(rsi) or pd.isna(macd):
            signals.append(0); continue
        if rsi < 30: signals.append(1)
        elif rsi > 70: signals.append(-1)
        elif macd > 0: signals.append(1)
        elif macd < 0: signals.append(-1)
        else: signals.append(0)
    return np.array(signals)


def get_signals_ml(ml_model, test_df, base_cols):
    pred = ml_model.predict(test_df[base_cols].values)
    return np.where(pred == 1, 1, -1)


def get_signals_sg(regime_model, test_df, feature_cols):
    X = test_df[feature_cols].values
    probs = regime_model.predict_proba(X)
    pred = regime_model.predict(X)
    conf = probs.max(axis=1)
    signals = np.zeros(len(X))
    for i in range(len(X)):
        if conf[i] < CONF: continue
        if test_df['bars_since_regime_change'].iloc[i] < 3: continue
        if pred[i] in [0, 1]: signals[i] = 1
        elif pred[i] in [3, 4]: signals[i] = -1
    return signals


def evaluate(signals, returns):
    tm = signals != 0
    nt = tm.sum()
    if nt == 0:
        return {'trades': 0, 'win_rate': 0, 'profit': 0, 'balance': capital, 'pf': 0, 'return_pct': 0}
    pnl = signals[tm] * returns[tm]
    w = pnl[pnl > 0]
    l = pnl[pnl < 0]
    wr = len(w) / nt * 100
    pr = capital * pnl.sum()
    pf = w.sum() / abs(l.sum()) if l.sum() != 0 else 99
    return {
        'trades': int(nt), 'win_rate': round(wr, 1),
        'profit': round(pr, 0), 'balance': round(capital + pr, 0),
        'pf': round(min(pf, 99), 2), 'return_pct': round(pnl.sum() * 100, 1),
    }


def run_pair(pair_name):
    print(f"\n{'='*80}")
    print(f"  {pair_name} — Two-Phase Experiment")
    print(f"{'='*80}")

    path = os.path.join(DATA_DIR, 'price', f'{pair_name}_4h.csv')
    df = pd.read_csv(path, parse_dates=['datetime_utc'])
    df = df.sort_values('datetime_utc').reset_index(drop=True)

    df = compute_technical_features(df)
    df = compute_volatility_features(df)
    df = compute_macro_features(df, pair_name, DATA_DIR)
    df = compute_sentiment_features(df, pair_name, DATA_DIR)
    df = compute_regime_features(df)
    df = compute_adaptive_regime_labels(df)
    df = create_5class_regime(df)
    df['target_return'] = np.log(df['close'].shift(-1) / df['close'])
    df['target_dir'] = (df['target_return'] > 0).astype(int)
    df = df.iloc[260:].reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    base_cols = [c for c in feature_cols if c not in [
        'atr_pct_short', 'atr_pct_long', 'vol_ratio_5_60', 'vol_ratio_5_20',
        'vol_compressed', 'compression_duration', 'range_contraction',
        'hurst_exponent', 'consecutive_dir_bars', 'vol_divergence',
        'event_vol_interaction', 'range_expansion', 'abs_gap', 'gap_expansion',
        'market_state', 'target_market_state', 'target_dir',
    ]]

    # Splits
    train1 = df[(df['datetime_utc'] >= '2015-01-01') & (df['datetime_utc'] < '2019-01-01')]
    test1 = df[(df['datetime_utc'] >= '2019-01-01') & (df['datetime_utc'] < '2021-01-01')]
    train2 = df[(df['datetime_utc'] >= '2021-01-01') & (df['datetime_utc'] < '2025-01-01')]
    test2 = df[df['datetime_utc'] >= '2025-01-01']

    print(f"  Phase 1 Train: {len(train1)} | Test: {len(test1)}")
    print(f"  Phase 2 Train: {len(train2)} | Test: {len(test2)}")

    # ---- PHASE 1: Train 2015-2018, Test 2019-2020 ----
    ml1 = xgb.XGBClassifier(**ML_PARAMS)
    ml1.fit(train1[base_cols].values, train1['target_dir'].values, verbose=False)

    X_reg1, y_reg1 = ensure_classes(train1[feature_cols].values, train1['target_market_state'].values)
    regime1 = xgb.XGBClassifier(**REGIME_PARAMS)
    regime1.fit(X_reg1, y_reg1, verbose=False)

    # ---- PHASE 2: Retrain 2021-2024 ----
    ml2 = xgb.XGBClassifier(**ML_PARAMS)
    ml2.fit(train2[base_cols].values, train2['target_dir'].values, verbose=False)

    X_reg2, y_reg2 = ensure_classes(train2[feature_cols].values, train2['target_market_state'].values)
    regime2 = xgb.XGBClassifier(**REGIME_PARAMS)
    regime2.fit(X_reg2, y_reg2, verbose=False)

    results = {}

    for phase_name, test_df, ml_static, ml_retrained, reg_static, reg_retrained in [
        ('Phase 1 (2019-2020)', test1, ml1, ml1, regime1, regime1),
        ('Phase 2 (2025+)', test2, ml1, ml2, regime1, regime2),
    ]:
        y_test = test_df['target_return'].values
        print(f"\n  --- {phase_name} ({len(test_df)} bars) ---")
        print(f"  {'Strategy':<35} | {'Trades':>7} | {'Win%':>6} | {'Profit':>10} | {'Balance':>10} | {'Return':>8} | {'PF':>5}")
        print(f"  {'-'*90}")

        phase_results = {}

        # Technical
        tech = get_signals_tech(test_df)
        r = evaluate(tech, y_test)
        phase_results['Technical'] = r
        print(f"  {'Technical (RSI/MACD)':<35} | {r['trades']:>7,} | {r['win_rate']:>5.1f}% | ${r['profit']:>+9,.0f} | ${r['balance']:>9,.0f} | {r['return_pct']:>+7.1f}% | {r['pf']:>5.2f}")

        # ML Static (trained Phase 1 only)
        ml_s = get_signals_ml(ml_static, test_df, base_cols)
        r = evaluate(ml_s, y_test)
        phase_results['ML Static'] = r
        print(f"  {'ML Direction (static, Phase 1)':<35} | {r['trades']:>7,} | {r['win_rate']:>5.1f}% | ${r['profit']:>+9,.0f} | ${r['balance']:>9,.0f} | {r['return_pct']:>+7.1f}% | {r['pf']:>5.2f}")

        # ML Retrained
        ml_r = get_signals_ml(ml_retrained, test_df, base_cols)
        r = evaluate(ml_r, y_test)
        phase_results['ML Retrained'] = r
        print(f"  {'ML Direction (retrained)':<35} | {r['trades']:>7,} | {r['win_rate']:>5.1f}% | ${r['profit']:>+9,.0f} | ${r['balance']:>9,.0f} | {r['return_pct']:>+7.1f}% | {r['pf']:>5.2f}")

        # SG Static (regime from Phase 1 only)
        sg_s = get_signals_sg(reg_static, test_df, feature_cols)
        r = evaluate(sg_s, y_test)
        phase_results['SG Static'] = r
        print(f"  {'ShiftGuard (static, Phase 1)':<35} | {r['trades']:>7,} | {r['win_rate']:>5.1f}% | ${r['profit']:>+9,.0f} | ${r['balance']:>9,.0f} | {r['return_pct']:>+7.1f}% | {r['pf']:>5.2f}")

        # SG Retrained
        sg_r = get_signals_sg(reg_retrained, test_df, feature_cols)
        r = evaluate(sg_r, y_test)
        phase_results['SG Retrained'] = r
        print(f"  {'ShiftGuard (retrained)':<35} | {r['trades']:>7,} | {r['win_rate']:>5.1f}% | ${r['profit']:>+9,.0f} | ${r['balance']:>9,.0f} | {r['return_pct']:>+7.1f}% | {r['pf']:>5.2f}")

        results[phase_name] = phase_results

    with open(os.path.join(RESULTS_DIR, f'{pair_name}_two_phase.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    all_results = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        results = run_pair(pair)
        all_results[pair] = results

    # Combined Phase 2 summary
    print(f"\n\n{'='*80}")
    print(f"  COMBINED Phase 2 (2025+) — $10K per pair = $30K")
    print(f"{'='*80}")
    print(f"  {'Strategy':<35} | {'Total Profit':>12} | {'Balance':>10} | {'Return':>8}")
    print(f"  {'-'*70}")

    for sname in ['Technical', 'ML Static', 'ML Retrained', 'SG Static', 'SG Retrained']:
        total = sum(all_results[p]['Phase 2 (2025+)'][sname]['profit'] for p in pairs)
        bal = 30000 + total
        ret = total / 30000 * 100
        print(f"  {sname:<35} | ${total:>+11,.0f} | ${bal:>9,.0f} | {ret:>+7.1f}%")

    with open(os.path.join(RESULTS_DIR, 'two_phase_overall.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: results/two_phase/")
