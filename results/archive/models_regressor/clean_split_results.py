"""
Clean Split Results
Train: 2022-2024 (3 years)
Test:  2025-2026 (final validation, completely unseen)

All models trained on same data, tested on same data.

Usage:
    python src/models/clean_split_results.py
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'clean_split')
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
CONFIDENCE_THRESHOLD = 0.55


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


def ensure_all_classes(X, y, n=5):
    unique = np.unique(y)
    for cls in range(n):
        if cls not in unique:
            idx = np.random.randint(0, len(X))
            X = np.vstack([X, X[idx:idx+1]])
            y = np.append(y, cls)
    return X, y


def evaluate(signals, returns, name):
    tm = signals != 0
    nt = tm.sum()
    if nt == 0:
        return {'name': name, 'trades': 0, 'win_rate': 0, 'profit': 0, 'balance': capital, 'pf': 0}
    pnl = signals[tm] * returns[tm]
    w = pnl[pnl > 0]
    l = pnl[pnl < 0]
    wr = len(w) / nt * 100
    pr = capital * pnl.sum()
    pf = w.sum() / abs(l.sum()) if l.sum() != 0 else 99
    return {
        'name': name, 'trades': int(nt), 'win_rate': round(wr, 1),
        'profit': round(pr, 0), 'balance': round(capital + pr, 0),
        'pf': round(min(pf, 99), 2), 'return_pct': round(pnl.sum() * 100, 1),
    }


def run_pair(pair_name):
    print(f"\n{'='*70}")
    print(f"  {pair_name} — Train: 2022-2024 | Test: 2025-2026")
    print(f"{'='*70}")

    # Build dataset
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

    # Split
    train = df[(df['datetime_utc'] >= '2022-01-01') & (df['datetime_utc'] < '2025-01-01')]
    test = df[df['datetime_utc'] >= '2025-01-01']

    print(f"  Train: {len(train)} bars (2022-2024)")
    print(f"  Test:  {len(test)} bars (2025-2026)")

    X_train = train[feature_cols].values
    y_dir_train = train['target_dir'].values
    y_regime_train = train['target_market_state'].values

    X_test = test[feature_cols].values
    y_test = test['target_return'].values

    # Base features (no regime) for ML direction
    base_cols = [c for c in feature_cols if c not in [
        'atr_pct_short', 'atr_pct_long', 'vol_ratio_5_60', 'vol_ratio_5_20',
        'vol_compressed', 'compression_duration', 'range_contraction',
        'hurst_exponent', 'consecutive_dir_bars', 'vol_divergence',
        'event_vol_interaction', 'range_expansion', 'abs_gap', 'gap_expansion',
        'market_state', 'target_market_state', 'target_dir',
    ]]

    # --- Train all models ---

    # 1. ML Direction (XGBClassifier on base features)
    ml_model = xgb.XGBClassifier(**ML_PARAMS)
    ml_model.fit(train[base_cols].values, y_dir_train, verbose=False)

    # 2. Regime Classifier (5-class)
    X_reg, y_reg = ensure_all_classes(X_train, y_regime_train)
    regime_model = xgb.XGBClassifier(**REGIME_PARAMS)
    regime_model.fit(X_reg, y_reg, verbose=False)

    # --- Test all strategies ---

    # Technical
    tech_signals = []
    for _, row in test.iterrows():
        rsi = row.get('rsi_14', 50)
        macd = row.get('macd_hist', 0)
        if pd.isna(rsi) or pd.isna(macd):
            tech_signals.append(0)
            continue
        if rsi < 30: tech_signals.append(1)
        elif rsi > 70: tech_signals.append(-1)
        elif macd > 0: tech_signals.append(1)
        elif macd < 0: tech_signals.append(-1)
        else: tech_signals.append(0)
    tech_signals = np.array(tech_signals)

    # ML Direction
    ml_pred = ml_model.predict(test[base_cols].values)
    ml_signals = np.where(ml_pred == 1, 1, -1)

    # ShiftGuard
    regime_probs = regime_model.predict_proba(X_test)
    regime_pred = regime_model.predict(X_test)
    regime_conf = regime_probs.max(axis=1)

    sg_signals = np.zeros(len(X_test))
    for i in range(len(X_test)):
        if regime_conf[i] < CONFIDENCE_THRESHOLD:
            continue
        if test['bars_since_regime_change'].iloc[i] < 3:
            continue
        state = regime_pred[i]
        if state in [0, 1]: sg_signals[i] = 1
        elif state in [3, 4]: sg_signals[i] = -1

    # --- Evaluate ---
    print(f"\n  {'Strategy':<28} | {'Trades':>7} | {'Win%':>6} | {'Profit':>10} | {'Balance':>10} | {'Return':>8} | {'PF':>5}")
    print(f"  {'-'*85}")

    results = {}
    for sname, signals in [('Technical (RSI/MACD)', tech_signals),
                            ('ML Direction (XGBoost)', ml_signals),
                            ('ShiftGuard (Regime)', sg_signals)]:
        r = evaluate(signals, y_test, sname)
        results[sname] = r
        profit_str = f"${r['profit']:>+9,.0f}"
        bal_str = f"${r['balance']:>9,.0f}"
        print(f"  {sname:<28} | {r['trades']:>7,} | {r['win_rate']:>5.1f}% | {profit_str} | {bal_str} | {r.get('return_pct',0):>+7.1f}% | {r['pf']:>5.2f}")

    # Save
    with open(os.path.join(RESULTS_DIR, f'{pair_name}_clean_split.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    all_results = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        results = run_pair(pair)
        all_results[pair] = results

    # Combined
    print(f"\n\n{'='*70}")
    print(f"  COMBINED ($10K per pair = $30K) — Test: 2025-2026")
    print(f"{'='*70}")
    print(f"  {'Strategy':<28} | {'Total Profit':>12} | {'Balance':>10} | {'Return':>8}")
    print(f"  {'-'*65}")

    for sname in ['Technical (RSI/MACD)', 'ML Direction (XGBoost)', 'ShiftGuard (Regime)']:
        total_profit = sum(all_results[p][sname]['profit'] for p in all_results)
        balance = 30000 + total_profit
        ret = total_profit / 30000 * 100
        print(f"  {sname:<28} | ${total_profit:>+11,.0f} | ${balance:>9,.0f} | {ret:>+7.1f}%")

    with open(os.path.join(RESULTS_DIR, 'clean_split_overall.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: results/clean_split/")
