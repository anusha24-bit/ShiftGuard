"""Run a walk-forward robustness experiment for the canonical monitored model."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2] / '.mpl-cache'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.main_xgboost import evaluate, get_feature_cols

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
PREDICTIONS_DIR = PROJECT_ROOT / 'results' / 'predictions'
WALKFORWARD_DIR = PROJECT_ROOT / 'results' / 'walkforward'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
WALKFORWARD_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EVAL_START = pd.Timestamp('2021-01-01')
DEFAULT_CHUNK_SIZE = 180


def load_holdout_summary() -> dict[str, dict[str, object]]:
    path = PREDICTIONS_DIR / 'xgboost_summary.json'
    with open(path) as f:
        return json.load(f)


def canonical_model_params(best_params: dict[str, object]) -> dict[str, object]:
    params = dict(best_params)
    params.setdefault('tree_method', 'hist')
    params.setdefault('random_state', 42)
    params.setdefault('verbosity', 0)
    return params


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in df.itertuples(index=False):
        lines.append('| ' + ' | '.join(str(value) for value in row) + ' |')
    return '\n'.join(lines)


def run_pair_walkforward(
    pair_name: str,
    pair_summary: dict[str, object],
    chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(PROCESSED_DIR / f'{pair_name}_features.csv')
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], errors='coerce')
    df = df.dropna(subset=['datetime_utc']).sort_values('datetime_utc').reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    eval_mask = df['datetime_utc'] >= EVAL_START
    if not eval_mask.any():
        raise ValueError(f'{pair_name}: no rows found on or after {EVAL_START.date()}')

    start_idx = int(np.flatnonzero(eval_mask.values)[0])
    params = canonical_model_params(pair_summary['best_params'])  # type: ignore[index]

    prediction_chunks: list[pd.DataFrame] = []
    window_rows: list[dict[str, object]] = []
    cursor = start_idx
    window_id = 0

    while cursor < len(df):
        chunk_end = min(cursor + chunk_size, len(df))
        train_df = df.iloc[:cursor]
        test_df = df.iloc[cursor:chunk_end]
        if train_df.empty or test_df.empty:
            break

        model = xgb.XGBRegressor(**params)
        model.fit(
            train_df[feature_cols].values,
            train_df['target_return'].values,
            verbose=False,
        )
        preds = model.predict(test_df[feature_cols].values)
        metrics = evaluate(test_df['target_return'].values, preds)
        window_id += 1

        pred_df = pd.DataFrame({
            'pair': pair_name,
            'window_id': window_id,
            'datetime_utc': test_df['datetime_utc'].values,
            'actual': test_df['target_return'].values,
            'predicted': preds,
            'abs_error': np.abs(test_df['target_return'].values - preds),
        })
        prediction_chunks.append(pred_df)

        window_rows.append({
            'pair': pair_name,
            'window_id': window_id,
            'train_end': str(train_df['datetime_utc'].iloc[-1].date()),
            'test_start': str(test_df['datetime_utc'].iloc[0].date()),
            'test_end': str(test_df['datetime_utc'].iloc[-1].date()),
            'train_rows': int(len(train_df)),
            'test_rows': int(len(test_df)),
            **metrics,
        })
        cursor = chunk_end

    pred_df = pd.concat(prediction_chunks, ignore_index=True)
    window_df = pd.DataFrame(window_rows)
    overall_metrics = evaluate(pred_df['actual'].values, pred_df['predicted'].values)

    summary = {
        'pair': pair_name,
        'chunk_size_bars': chunk_size,
        'evaluation_start': str(EVAL_START.date()),
        'n_refits': int(len(window_df)),
        'feature_count': int(len(feature_cols)),
        'overall': overall_metrics,
    }

    pred_df.to_csv(WALKFORWARD_DIR / f'{pair_name}_walkforward_predictions.csv', index=False)
    window_df.to_csv(WALKFORWARD_DIR / f'{pair_name}_walkforward_windows.csv', index=False)
    return pred_df, window_df, summary


def build_comparison_frame(
    holdout_summary: dict[str, dict[str, object]],
    walkforward_summary: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair, pair_holdout in holdout_summary.items():
        pair_walk = walkforward_summary[pair]['overall']
        holdout_test = pair_holdout['test']  # type: ignore[index]
        rows.append({
            'pair': pair,
            'holdout_mae': round(float(holdout_test['mae']), 6),
            'walkforward_mae': round(float(pair_walk['mae']), 6),
            'mae_delta': round(float(pair_walk['mae']) - float(holdout_test['mae']), 6),
            'holdout_dir_acc': round(float(holdout_test['dir_acc']) * 100, 2),
            'walkforward_dir_acc': round(float(pair_walk['dir_acc']) * 100, 2),
            'dir_acc_delta_pp': round((float(pair_walk['dir_acc']) - float(holdout_test['dir_acc'])) * 100, 2),
            'holdout_f1': round(float(holdout_test['f1']), 4),
            'walkforward_f1': round(float(pair_walk['f1']), 4),
            'f1_delta': round(float(pair_walk['f1']) - float(holdout_test['f1']), 4),
            'refits': int(walkforward_summary[pair]['n_refits']),
        })
    comparison_df = pd.DataFrame(rows).sort_values('pair').reset_index(drop=True)
    comparison_df.to_csv(WALKFORWARD_DIR / 'holdout_vs_walkforward.csv', index=False)
    return comparison_df


def plot_metric_comparison(comparison_df: pd.DataFrame) -> None:
    pairs = comparison_df['pair'].tolist()
    x = np.arange(len(pairs))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(x - width / 2, comparison_df['holdout_mae'], width, label='Static Holdout')
    axes[0].bar(x + width / 2, comparison_df['walkforward_mae'], width, label='Walk-Forward')
    axes[0].set_title('MAE: Static Holdout vs Walk-Forward')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pairs)
    axes[0].set_ylabel('MAE')
    axes[0].grid(axis='y', alpha=0.2)
    axes[0].legend()

    axes[1].bar(x - width / 2, comparison_df['holdout_dir_acc'], width, label='Static Holdout')
    axes[1].bar(x + width / 2, comparison_df['walkforward_dir_acc'], width, label='Walk-Forward')
    axes[1].set_title('Directional Accuracy: Static Holdout vs Walk-Forward')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pairs)
    axes[1].set_ylabel('Directional Accuracy (%)')
    axes[1].grid(axis='y', alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'walkforward_metric_comparison.png', dpi=180)
    plt.close(fig)


def plot_rolling_error_comparison(pairs: list[str]) -> None:
    fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 4 * len(pairs)), sharex=False)
    if len(pairs) == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        static_df = pd.read_csv(PREDICTIONS_DIR / f'xgboost_{pair}_predictions.csv')
        walk_df = pd.read_csv(WALKFORWARD_DIR / f'{pair}_walkforward_predictions.csv')
        static_df['datetime_utc'] = pd.to_datetime(static_df['datetime_utc'], errors='coerce')
        walk_df['datetime_utc'] = pd.to_datetime(walk_df['datetime_utc'], errors='coerce')

        static_df = static_df.dropna(subset=['datetime_utc']).sort_values('datetime_utc').reset_index(drop=True)
        walk_df = walk_df.dropna(subset=['datetime_utc']).sort_values('datetime_utc').reset_index(drop=True)

        static_roll = pd.Series(np.abs(static_df['actual'].values - static_df['predicted'].values)).rolling(90).mean()
        walk_roll = pd.Series(np.abs(walk_df['actual'].values - walk_df['predicted'].values)).rolling(90).mean()

        ax.plot(static_df['datetime_utc'], static_roll, label='Static Holdout', alpha=0.9)
        ax.plot(walk_df['datetime_utc'], walk_roll, label='Walk-Forward', alpha=0.9)
        ax.set_title(f'90-Bar Rolling MAE - {pair}')
        ax.set_ylabel('Rolling MAE')
        ax.grid(alpha=0.2)
        ax.legend()

    axes[-1].set_xlabel('Datetime (UTC)')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'walkforward_rolling_mae.png', dpi=180)
    plt.close(fig)


def write_markdown_report(
    comparison_df: pd.DataFrame,
    walkforward_summary: dict[str, dict[str, object]],
    chunk_size: int,
) -> None:
    mae_better = int((comparison_df['mae_delta'] < 0).sum())
    acc_better = int((comparison_df['dir_acc_delta_pp'] > 0).sum())
    best_mae_pair = comparison_df.loc[comparison_df['mae_delta'].idxmin()]
    worst_mae_pair = comparison_df.loc[comparison_df['mae_delta'].idxmax()]

    report_table = comparison_df.copy()
    report_table['holdout_mae'] = report_table['holdout_mae'].map(lambda x: f'{x:.6f}')
    report_table['walkforward_mae'] = report_table['walkforward_mae'].map(lambda x: f'{x:.6f}')
    report_table['mae_delta'] = report_table['mae_delta'].map(lambda x: f'{x:+.6f}')
    report_table['holdout_dir_acc'] = report_table['holdout_dir_acc'].map(lambda x: f'{x:.2f}%')
    report_table['walkforward_dir_acc'] = report_table['walkforward_dir_acc'].map(lambda x: f'{x:.2f}%')
    report_table['dir_acc_delta_pp'] = report_table['dir_acc_delta_pp'].map(lambda x: f'{x:+.2f} pp')

    lines = [
        '# Walk-Forward Validation Report',
        '',
        '## Setup',
        f'- Canonical monitored model: `XGBRegressor` with frozen pair-specific params from `results/predictions/xgboost_summary.json`.',
        f'- Validation scheme: expanding-window walk-forward refit every `{chunk_size}` bars (about 30 trading days on 4H data).',
        f'- Evaluation window: `{EVAL_START.date()}` onward.',
        '- Comparison baseline: current single-fit chronological holdout metrics already tracked in `results/predictions/xgboost_summary.json`.',
        '',
        '## Headline',
        f'- Walk-forward improved MAE on `{mae_better}/3` pairs and directional accuracy on `{acc_better}/3` pairs.',
        f"- Strongest MAE gain: `{best_mae_pair['pair']}` ({best_mae_pair['mae_delta']:+.6f}).",
        f"- Largest MAE regression: `{worst_mae_pair['pair']}` ({worst_mae_pair['mae_delta']:+.6f}).",
        '',
        '## Pair Comparison',
        '',
        to_markdown_table(report_table[['pair', 'holdout_mae', 'walkforward_mae', 'mae_delta', 'holdout_dir_acc', 'walkforward_dir_acc', 'dir_acc_delta_pp', 'refits']]),
        '',
        '## Report-Ready Paragraph',
        (
            'We ran an additional robustness experiment using expanding-window walk-forward validation on the canonical '
            'ShiftGuard `XGBRegressor`. Starting from the 2021 evaluation boundary, the model was refit before each '
            f'{chunk_size}-bar block using only data available up to that point, and the combined walk-forward predictions '
            'were compared against the current single-fit holdout benchmark. The results show that walk-forward updating '
            f'improves MAE on {mae_better} of the 3 pairs and improves directional accuracy on {acc_better} of the 3 pairs. '
            f"The largest MAE gain appears on {best_mae_pair['pair']}, while {worst_mae_pair['pair']} is the clearest case "
            'where periodic refitting does not help. Overall, the experiment suggests that the monitored model is reasonably '
            'stable under stricter time-aware evaluation, but the value of rolling retraining remains pair dependent rather '
            'than universally beneficial.'
        ),
        '',
        '## Slide Bullets',
        '- Added a new walk-forward robustness experiment on the canonical monitored model.',
        f'- Expanding-window refit every `{chunk_size}` bars using only past data.',
        f'- MAE improved on `{mae_better}/3` pairs; directional accuracy improved on `{acc_better}/3` pairs.',
        f"- Best walk-forward improvement: `{best_mae_pair['pair']}`.",
        '- Takeaway: stricter temporal evaluation supports the current pipeline, but rolling updates are still market-specific rather than universally helpful.',
        '',
        '## Output Files',
        '- `results/walkforward/*_walkforward_predictions.csv`',
        '- `results/walkforward/*_walkforward_windows.csv`',
        '- `results/walkforward/holdout_vs_walkforward.csv`',
        '- `results/figures/walkforward_metric_comparison.png`',
        '- `results/figures/walkforward_rolling_mae.png`',
    ]

    report_path = WALKFORWARD_DIR / 'WALKFORWARD_REPORT.md'
    report_path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', nargs='+', default=['EURUSD', 'GBPJPY', 'XAUUSD'])
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    holdout_summary = load_holdout_summary()
    walkforward_summary: dict[str, dict[str, object]] = {}
    all_windows: list[pd.DataFrame] = []

    for pair in args.pairs:
        if pair not in holdout_summary:
            raise KeyError(f'{pair}: not found in results/predictions/xgboost_summary.json')
        print(f'Running walk-forward validation for {pair}...')
        _, window_df, pair_summary = run_pair_walkforward(pair, holdout_summary[pair], args.chunk_size)
        walkforward_summary[pair] = pair_summary
        all_windows.append(window_df)

    windows_df = pd.concat(all_windows, ignore_index=True)
    windows_df.to_csv(WALKFORWARD_DIR / 'walkforward_windows_all_pairs.csv', index=False)

    with open(WALKFORWARD_DIR / 'walkforward_summary.json', 'w') as f:
        json.dump(walkforward_summary, f, indent=2)

    comparison_df = build_comparison_frame(holdout_summary, walkforward_summary)
    plot_metric_comparison(comparison_df)
    plot_rolling_error_comparison(args.pairs)
    write_markdown_report(comparison_df, walkforward_summary, args.chunk_size)

    print(f"Saved walk-forward outputs to {WALKFORWARD_DIR}")
    print(f"Saved walk-forward figures to {FIGURES_DIR}")


if __name__ == '__main__':
    main()
