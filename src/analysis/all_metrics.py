"""Compare tracked baseline outputs, ML direction, and ShiftGuard trading metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PAIRS = ['EURUSD', 'GBPJPY', 'XAUUSD']
CAPITAL = 10000
STOP_LOSS = {'EURUSD': 0.005, 'GBPJPY': 0.01, 'XAUUSD': 0.01}
TRADE_COST = {'EURUSD': 0.00015, 'GBPJPY': 0.00025, 'XAUUSD': 0.00035}
LEVERAGE = 20
TAX_RATE = 0.30
BASE_DIR = Path(__file__).resolve().parents[2] / 'results'

HORIZONS = {
    '1 Month': ('2026-03-01', '2026-04-01'),
    '3 Months': ('2026-01-01', '2026-04-01'),
    '6 Months': ('2025-10-01', '2026-04-01'),
    '1 Year': ('2025-04-01', '2026-04-01'),
    '2 Years': ('2024-04-01', '2026-04-01'),
    '5 Years': ('2021-04-01', '2026-04-01'),
}


def apply_stop_loss(signal: float, actual_return: float, stop_loss: float) -> float:
    pnl = signal * actual_return
    return -stop_loss if pnl < -stop_loss else pnl


def compute_trading_metrics(signals: np.ndarray, returns: np.ndarray, stop_loss: float, cost: float) -> dict[str, float]:
    trade_mask = signals != 0
    trades = int(trade_mask.sum())
    if trades == 0:
        return {'trades': 0, 'win_rate': 0, 'profit': 0, 'balance': CAPITAL, 'pf': 0}

    pnl = np.array([(apply_stop_loss(s, r, stop_loss) - cost) * LEVERAGE for s, r in zip(signals[trade_mask], returns[trade_mask])])
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    total_profit = max(CAPITAL * pnl.sum(), -CAPITAL)
    tax = total_profit * TAX_RATE if total_profit > 0 else 0
    after_tax = total_profit - tax
    balance = CAPITAL + after_tax
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99
    return {
        'trades': trades,
        'win_rate': round(len(wins) / trades * 100, 1),
        'profit': round(after_tax, 0),
        'balance': round(balance, 0),
        'tax': round(tax, 0),
        'pf': round(min(profit_factor, 99), 2),
    }


def compute_ml_metrics(actual_dir: np.ndarray, pred_dir: np.ndarray) -> dict[str, float]:
    return {
        'accuracy': round(accuracy_score(actual_dir, pred_dir) * 100, 1),
        'f1': round(f1_score(actual_dir, pred_dir, zero_division=0), 3),
        'precision': round(precision_score(actual_dir, pred_dir, zero_division=0), 3),
        'recall': round(recall_score(actual_dir, pred_dir, zero_division=0), 3),
    }


def main() -> None:
    winrate_data = {}
    for pair in PAIRS:
        df = pd.read_csv(BASE_DIR / 'winrate' / f'{pair}_winrate_trades.csv')
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        winrate_data[pair] = df

    baseline_data: dict[str, dict[str, pd.DataFrame | None]] = {}
    for baseline_name, baseline_dir in [('B1 LSTM', 'baseline1/baseline1_lstm'), ('B2 BiLSTM', 'baseline2/baseline2_bilstm'), ('B3 Stacked', 'baseline3/baseline3_stacked')]:
        baseline_data[baseline_name] = {}
        for pair in PAIRS:
            path = BASE_DIR / f'{baseline_dir}_{pair}.csv'
            if not path.exists():
                baseline_data[baseline_name][pair] = None
                continue
            df = pd.read_csv(path)
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            df['actual_dir'] = (df['actual'] > 0).astype(int)
            df['pred_dir'] = (df['predicted'] > 0).astype(int)
            df['signal'] = np.where(df['predicted'] > 0, 1, -1)
            baseline_data[baseline_name][pair] = df

    for pair in PAIRS:
        print(f"\n{'='*120}")
        print(f"  {pair} - $10K, 1:20 leverage, costs, 30% tax, SL={'0.5%' if pair=='EURUSD' else '1%'}")
        print(f"{'='*120}")
        header = f"{'Horizon':<10} | {'Model':<14} | {'Acc':>5} | {'F1':>5} | {'WinR':>5} | {'Trades':>6} | {'PF':>5} | {'Profit':>12} | {'Balance':>10}"
        print(header)
        print('-' * len(header))
        stop_loss = STOP_LOSS[pair]

        for hname, (start, end) in HORIZONS.items():
            models: list[tuple[str, dict[str, float] | None, dict[str, float] | None]] = []
            for baseline_name in ['B1 LSTM', 'B2 BiLSTM', 'B3 Stacked']:
                bdf = baseline_data[baseline_name][pair]
                if bdf is None:
                    models.append((baseline_name, None, None))
                    continue
                chunk = bdf[(bdf['datetime_utc'] >= start) & (bdf['datetime_utc'] < end)]
                if len(chunk) < 5:
                    models.append((baseline_name, None, None))
                    continue
                ml_metrics = compute_ml_metrics(chunk['actual_dir'].values, chunk['pred_dir'].values)
                trade_metrics = compute_trading_metrics(chunk['signal'].values, chunk['actual'].values, stop_loss, TRADE_COST[pair])
                models.append((baseline_name, ml_metrics, trade_metrics))

            chunk = winrate_data[pair][(winrate_data[pair]['datetime_utc'] >= start) & (winrate_data[pair]['datetime_utc'] < end)]
            if len(chunk) >= 5:
                for strategy_name, col in [('ML Direction', 'ml_signal'), ('ShiftGuard', 'sg_signal')]:
                    signals = chunk[col].values
                    returns = chunk['actual_return'].values
                    trade_metrics = compute_trading_metrics(signals, returns, stop_loss, TRADE_COST[pair])
                    traded = signals != 0
                    if traded.sum() > 0:
                        ml_metrics = compute_ml_metrics((returns[traded] > 0).astype(int), (signals[traded] > 0).astype(int))
                    else:
                        ml_metrics = {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}
                    models.append((strategy_name, ml_metrics, trade_metrics))

            first = True
            for model_name, ml_metrics, trade_metrics in models:
                horizon_label = hname if first else ""
                first = False
                if ml_metrics is None or trade_metrics is None:
                    print(f"{horizon_label:<10} | {model_name:<14} | {'-':>5} | {'-':>5} | {'-':>5} | {'-':>6} | {'-':>5} | {'-':>12} | {'-':>10}")
                    continue
                profit_str = f"${trade_metrics['profit']:>+11,.0f}" if trade_metrics['profit'] != 0 else f"{'$0':>12}"
                balance_str = f"${trade_metrics['balance']:>9,.0f}"
                print(f"{horizon_label:<10} | {model_name:<14} | {ml_metrics['accuracy']:>5.1f} | {ml_metrics['f1']:>5.3f} | {trade_metrics['win_rate']:>5.1f} | {trade_metrics['trades']:>6} | {trade_metrics['pf']:>5.2f} | {profit_str} | {balance_str}")
            print('-' * len(header))


if __name__ == '__main__':
    main()
