"""P&L with 0.5% stop loss from existing trade CSVs. No model runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = ['EURUSD', 'GBPJPY', 'XAUUSD']
BASE_DIR = Path(__file__).resolve().parents[2] / 'results' / 'winrate'
CAPITAL = 10000
STOP_LOSS = 0.005
SPREAD = {'EURUSD': 0.00010, 'GBPJPY': 0.00020, 'XAUUSD': 0.00030}
COMMISSION = 0.00003
SLIPPAGE = 0.00002
SWAP = 0.000005
TAX_RATE = 0.30

HORIZONS = {
    '1 Month': ('2026-03-01', '2026-04-01'),
    '3 Months': ('2026-01-01', '2026-04-01'),
    '6 Months': ('2025-10-01', '2026-04-01'),
    '1 Year': ('2025-04-01', '2026-04-01'),
    '2 Years': ('2024-04-01', '2026-04-01'),
    '5 Years': ('2021-04-01', '2026-04-01'),
}


def main() -> None:
    data = {
        pair: pd.read_csv(BASE_DIR / f'{pair}_winrate_trades.csv', parse_dates=['datetime_utc'])
        for pair in PAIRS
    }
    strategies = [('Technical', 'tech_signal'), ('ML Direction', 'ml_signal'), ('ShiftGuard', 'sg_signal')]

    for lev_label, lev_val in [('NO LEVERAGE (1:1)', 1), ('WITH LEVERAGE (1:20)', 20)]:
        print()
        print('=' * 90)
        print(f'  {lev_label} | $10,000/pair | 0.5% SL | costs + 30% tax')
        print('=' * 90)
        for hname, (start, end) in HORIZONS.items():
            print()
            print(f'  {hname} ({start} to {end})')
            header = '  {:<16}{:>8}{:>8}{:>14}{:>14}{:>10}'.format(
                'Strategy', 'Trades', 'Win%', 'After Tax', 'Balance', 'Result'
            )
            print(header)
            print('  ' + '-' * 68)
            for sname, col in strategies:
                total_trades = 0
                total_wins = 0
                total_after_tax = 0.0
                for pair in PAIRS:
                    chunk = data[pair][
                        (data[pair]['datetime_utc'] >= start) & (data[pair]['datetime_utc'] < end)
                    ]
                    if chunk.empty:
                        continue
                    signals = chunk[col].values
                    returns = chunk['actual_return'].values
                    trade_mask = signals != 0
                    trades = int(trade_mask.sum())
                    if trades == 0:
                        continue
                    raw = signals[trade_mask] * returns[trade_mask]
                    leveraged = raw * lev_val
                    stop_val = STOP_LOSS * lev_val
                    capped = np.where(leveraged < -stop_val, -stop_val, leveraged)
                    cost_per_trade = (SPREAD[pair] + COMMISSION + SLIPPAGE + SWAP) * lev_val
                    net = capped - cost_per_trade
                    gross = CAPITAL * capped.sum()
                    costs = CAPITAL * cost_per_trade * trades
                    pre_tax = gross - costs
                    tax = max(0, pre_tax * TAX_RATE)
                    after_tax = pre_tax - tax
                    total_trades += trades
                    total_wins += int((net > 0).sum())
                    total_after_tax += after_tax

                win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
                balance = 30000 + total_after_tax
                status = 'BLOWN' if balance <= 0 else ('PROFIT' if total_after_tax > 0 else 'LOSS')
                row = '  {:<16}{:>8}{:>7.1f}%  ${:>+12,.0f}  ${:>12,.0f}  {:>8}'.format(
                    sname, total_trades, win_rate, total_after_tax, balance, status
                )
                print(row)


if __name__ == '__main__':
    main()
