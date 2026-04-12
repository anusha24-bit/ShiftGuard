"""
April 1-10 2026 — Bar-by-bar prediction report from walk-forward results.
Reads the already-generated winrate trade CSVs and zooms into the April window.
"""
import os, json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'winrate')

REGIME_NAMES = {
    0: 'Trend-Up/LowVol',
    1: 'Trend-Up/HighVol',
    2: 'Ranging',
    3: 'Trend-Down/LowVol',
    4: 'Trend-Down/HighVol',
}

SIGNAL_MAP = {1: 'LONG', -1: 'SHORT', 0: 'SIT-OUT'}

START = '2026-04-01'
END   = '2026-04-11'  # exclusive


def report_pair(pair):
    path = os.path.join(RESULTS_DIR, f'{pair}_winrate_trades.csv')
    df = pd.read_csv(path, parse_dates=['datetime_utc'])
    apr = df[(df['datetime_utc'] >= START) & (df['datetime_utc'] < END)].copy()

    if len(apr) == 0:
        print(f"  No data for {pair} in April 1-10 2026")
        return None

    apr['date'] = apr['datetime_utc'].dt.strftime('%Y-%m-%d')
    apr['time'] = apr['datetime_utc'].dt.strftime('%H:%M')
    apr['regime_name'] = apr['regime'].map(REGIME_NAMES)
    apr['sg_action'] = apr['sg_signal'].map(SIGNAL_MAP)
    apr['ml_action'] = apr['ml_signal'].map(SIGNAL_MAP)
    apr['tech_action'] = apr['tech_signal'].map(SIGNAL_MAP)

    # PnL per bar
    apr['tech_pnl'] = apr['tech_signal'] * apr['actual_return']
    apr['ml_pnl']   = apr['ml_signal']   * apr['actual_return']
    apr['sg_pnl']   = apr['sg_signal']   * apr['actual_return']

    apr['tech_win'] = apr['tech_pnl'] > 0
    apr['ml_win']   = apr['ml_pnl']   > 0
    apr['sg_win']   = apr['sg_pnl']   > 0

    print(f"\n{'='*90}")
    print(f"  {pair} — April 1-10, 2026  ({len(apr)} bars)")
    print(f"{'='*90}")

    # Bar-by-bar table
    print(f"\n  {'Date':<12}{'Time':<7}{'Return':>9}  {'Regime':<22}{'Conf':>6}  "
          f"{'Tech':>7}{'ML':>8}{'SG':>10}  {'SG PnL':>9}")
    print("  " + "-" * 86)

    for _, r in apr.iterrows():
        ret_str = f"{r['actual_return']*100:+.3f}%"
        conf_str = f"{r['regime_confidence']:.2f}"
        sg_pnl_str = f"{r['sg_pnl']*100:+.4f}%" if r['sg_signal'] != 0 else "—"
        print(f"  {r['date']:<12}{r['time']:<7}{ret_str:>9}  {r['regime_name']:<22}{conf_str:>6}  "
              f"{r['tech_action']:>7}{r['ml_action']:>8}{r['sg_action']:>10}  {sg_pnl_str:>9}")

    # Summary stats
    print(f"\n  --- Summary ---")

    for label, sig_col, pnl_col, win_col in [
        ('Technical', 'tech_signal', 'tech_pnl', 'tech_win'),
        ('ML Direction', 'ml_signal', 'ml_pnl', 'ml_win'),
        ('ShiftGuard', 'sg_signal', 'sg_pnl', 'sg_win'),
    ]:
        traded = apr[apr[sig_col] != 0]
        n = len(traded)
        if n == 0:
            print(f"  {label:<15} No trades")
            continue
        wins = traded[win_col].sum()
        wr = wins / n * 100
        total_pnl = traded[pnl_col].sum() * 100
        gross_win = traded.loc[traded[pnl_col] > 0, pnl_col].sum()
        gross_loss = abs(traded.loc[traded[pnl_col] < 0, pnl_col].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
        print(f"  {label:<15} Trades: {n:>3}  Win Rate: {wr:>5.1f}%  "
              f"Total PnL: {total_pnl:>+7.3f}%  Profit Factor: {min(pf,99):>.2f}")

    return apr


if __name__ == '__main__':
    print("=" * 90)
    print("  SHIFTGUARD — April 1-10, 2026 Prediction Report")
    print("  (from daily-cadence walk-forward with multi-trigger retraining)")
    print("=" * 90)

    all_apr = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        apr_df = report_pair(pair)
        if apr_df is not None:
            all_apr[pair] = apr_df

    # Cross-pair summary
    if all_apr:
        print(f"\n{'='*90}")
        print("  CROSS-PAIR APRIL 1-10 SUMMARY")
        print(f"{'='*90}")
        print(f"\n  {'Pair':<10}{'Tech WR':<10}{'ML WR':<10}{'SG WR':<10}"
              f"{'SG Trades':<12}{'SG PnL':>10}")
        print("  " + "-" * 58)
        for pair, apr in all_apr.items():
            # Tech
            t_traded = apr[apr['tech_signal'] != 0]
            t_wr = t_traded['tech_win'].mean() * 100 if len(t_traded) > 0 else 0
            # ML
            m_traded = apr[apr['ml_signal'] != 0]
            m_wr = m_traded['ml_win'].mean() * 100 if len(m_traded) > 0 else 0
            # SG
            s_traded = apr[apr['sg_signal'] != 0]
            s_wr = s_traded['sg_win'].mean() * 100 if len(s_traded) > 0 else 0
            s_n = len(s_traded)
            s_pnl = s_traded['sg_pnl'].sum() * 100 if len(s_traded) > 0 else 0
            print(f"  {pair:<10}{t_wr:<10.1f}{m_wr:<10.1f}{s_wr:<10.1f}"
                  f"{s_n:<12}{s_pnl:>+9.3f}%")
