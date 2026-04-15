# Walk-Forward Validation Report

## Setup
- Canonical monitored model: `XGBRegressor` with frozen pair-specific params from `results/predictions/xgboost_summary.json`.
- Validation scheme: expanding-window walk-forward refit every `180` bars (about 30 trading days on 4H data).
- Evaluation window: `2021-01-01` onward.
- Comparison baseline: current single-fit chronological holdout metrics already tracked in `results/predictions/xgboost_summary.json`.

## Headline
- Walk-forward improved MAE on `2/3` pairs and directional accuracy on `3/3` pairs.
- Strongest MAE gain: `XAUUSD` (-0.000851).
- Largest MAE regression: `GBPJPY` (+0.000015).

## Pair Comparison

| pair | holdout_mae | walkforward_mae | mae_delta | holdout_dir_acc | walkforward_dir_acc | dir_acc_delta_pp | refits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EURUSD | 0.001141 | 0.001122 | -0.000019 | 59.34% | 60.87% | +1.53 pp | 48 |
| GBPJPY | 0.001621 | 0.001636 | +0.000015 | 52.45% | 53.10% | +0.65 pp | 48 |
| XAUUSD | 0.003483 | 0.002632 | -0.000851 | 51.34% | 55.84% | +4.50 pp | 47 |

## Report-Ready Paragraph
We ran an additional robustness experiment using expanding-window walk-forward validation on the canonical ShiftGuard `XGBRegressor`. Starting from the 2021 evaluation boundary, the model was refit before each 180-bar block using only data available up to that point, and the combined walk-forward predictions were compared against the current single-fit holdout benchmark. The results show that walk-forward updating improves MAE on 2 of the 3 pairs and improves directional accuracy on 3 of the 3 pairs. The largest MAE gain appears on XAUUSD, while GBPJPY is the clearest case where periodic refitting does not help. Overall, the experiment suggests that the monitored model is reasonably stable under stricter time-aware evaluation, but the value of rolling retraining remains pair dependent rather than universally beneficial.

## Slide Bullets
- Added a new walk-forward robustness experiment on the canonical monitored model.
- Expanding-window refit every `180` bars using only past data.
- MAE improved on `2/3` pairs; directional accuracy improved on `3/3` pairs.
- Best walk-forward improvement: `XAUUSD`.
- Takeaway: stricter temporal evaluation supports the current pipeline, but rolling updates are still market-specific rather than universally helpful.

## Output Files
- `results/walkforward/*_walkforward_predictions.csv`
- `results/walkforward/*_walkforward_windows.csv`
- `results/walkforward/holdout_vs_walkforward.csv`
- `results/figures/walkforward_metric_comparison.png`
- `results/figures/walkforward_rolling_mae.png`
