"""Summarize pipeline retraining outputs as P&L-style tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = ['EURUSD', 'GBPJPY', 'XAUUSD']
CAPITAL = 10000
STOP_LOSS = {'EURUSD': 0.005, 'GBPJPY': 0.01, 'XAUUSD': 0.01}
TRADE_COST = {'EURUSD': 0.00015, 'GBPJPY': 0.00025, 'XAUUSD': 0.00035}
LEVERAGE = 20
TAX_RATE = 0.30
BASE_DIR = Path(__file__).resolve().parents[2] / 'results' / 'retraining'

HORIZONS = {
    '1 Month': ('2026-02-01', '2026-03-18'),
    '3 Months': ('2025-12-18', '2026-03-18'),
    '6 Months': ('2025-09-18', '2026-03-18'),
    '1 Year': ('2025-03-18', '2026-03-18'),
    '2 Years': ('2024-03-18', '2026-03-18'),
    '5 Years': ('2021-03-18', '2026-03-18'),
}

STRATEGIES = {
    'No Retrain': 'pred_none',
    'Blind Monthly': 'pred_blind',
    'ShiftGuard': 'pred_shap',
    'Oracle': 'pred_oracle',
}


def compute(signals: np.ndarray, returns: np.ndarray, stop_loss: float, cost: float) -> dict[str, float]:
    trade_mask = signals != 0
    trades = int(trade_mask.sum())
    if trades == 0:
        return {'trades': 0, 'win_rate': 0, 'profit': 0, 'balance': CAPITAL, 'pf': 0}

    pnl = []
    for signal, actual_return in zip(signals[trade_mask], returns[trade_mask]):
        value = signal * actual_return
        if value < -stop_loss:
            value = -stop_loss
        value = (value - cost) * LEVERAGE
        pnl.append(value)
    pnl_arr = np.array(pnl)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    total_profit = max(CAPITAL * pnl_arr.sum(), -CAPITAL)
    tax = total_profit * TAX_RATE if total_profit > 0 else 0
    after_tax = total_profit - tax
    balance = CAPITAL + after_tax
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99
    return {
        'trades': trades,
        'win_rate': round(len(wins) / trades * 100, 1),
        'profit': round(after_tax, 0),
        'balance': round(balance, 0),
        'pf': round(min(profit_factor, 99), 2),
    }


def main() -> None:
    for pair in PAIRS:
        path = BASE_DIR / f'{pair}_walkforward_bars.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

        print(f"\n{'='*110}")
        print(f"  {pair} - PIPELINE RESULTS - $10K, 1:20 leverage, costs, 30% tax, SL={'0.5%' if pair=='EURUSD' else '1%'}")
        print(f"{'='*110}")
        header = f"{'Horizon':<10} | {'Strategy':<16} | {'WinR':>5} | {'Trades':>6} | {'PF':>5} | {'Profit':>12} | {'Balance':>10}"
        print(header)
        print('-' * len(header))

        for hname, (start, end) in HORIZONS.items():
            chunk = df[(df['datetime_utc'] >= start) & (df['datetime_utc'] <= end)]
            if len(chunk) < 5:
                continue

            first = True
            for strategy_name, col in STRATEGIES.items():
                preds = chunk[col].values
                actuals = chunk['actual'].values
                signals = np.where(preds > 0, 1, -1)
                metrics = compute(signals, actuals, STOP_LOSS[pair], TRADE_COST[pair])
                horizon_label = hname if first else ""
                first = False
                profit_str = f"${metrics['profit']:>+11,.0f}"
                balance_str = f"${metrics['balance']:>9,.0f}"
                print(f"{horizon_label:<10} | {strategy_name:<16} | {metrics['win_rate']:>5.1f} | {metrics['trades']:>6} | {metrics['pf']:>5.2f} | {profit_str} | {balance_str}")
            print('-' * len(header))


if __name__ == '__main__':
    main()
