# ShiftGuard

**A Human-in-the-Loop Framework for Distribution Shift Detection, Attribution, and Adaptive Retraining in Non-Stationary Forex Markets**

CS 6140 — Machine Learning | Prof. Smruthi Mukund | Northeastern University, Silicon Valley

**Team**: Sohan Mahesh · Anusha Ravi Kumar · Dishaben Manubhai Patel

---

## Overview

ShiftGuard is a market regime awareness system that detects distribution shifts in forex markets, explains what caused them via SHAP attribution, and adaptively retrains to maintain performance. It doesn't predict price direction — it **classifies market regimes and only trades when conditions are favorable**, sitting out during uncertain transitions.

Core result: **Technical indicators achieve 50% win rate (coin flip). ShiftGuard achieves 58-59% win rate with 2x profit factor — and is the only strategy that survives real-world trading costs.**

---

## Key Results

### Win Rate (4H bars, walk-forward validated, p < 0.000001)

| Strategy | Win Rate | Trades | Profit Factor |
|---|---|---|---|
| Technical (RSI/MACD) | 50.1% | 50,040 | 1.01 |
| ML Direction (XGBoost, static) | 53.0% | 50,040 | 1.12 |
| **ShiftGuard (Regime-Filtered)** | **58.7%** | **18,663** | **2.05** |

### Realistic P&L ($30K portfolio, leverage, spread, commission, slippage, swap, 30% tax)

| Horizon | Technical | ML Direction | ShiftGuard |
|---|---|---|---|
| 1 Month | -$11,010 (LOSS) | +$5,130 | **+$6,287** |
| 3 Months | +$2,524 | +$2,708 | **+$46,605** |
| 6 Months | -$19,023 (LOSS) | -$17,401 (LOSS) | **+$64,922** |
| 1 Year | -$85,825 (BLOWN) | -$30,076 (BLOWN) | **+$115,103** |

**Technical and ML blow up after trading costs. ShiftGuard is the only profitable strategy because it takes 63% fewer trades, paying 63% less in friction.**

---

## How It Works

```
Features → Regime Classifier → "Trending Up / Down / Ranging?"
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Trending Up     Ranging          Trending Down
              → GO LONG       → SIT OUT        → GO SHORT
              (high conf)     (avoid noise)    (high conf)
                    │                               │
                    └───────────┬───────────────────┘
                                ▼
                    Detection Engine monitors for regime change
                                │
                    SHAP explains: "Sentiment drove this shift"
                                │
                    Retrain regime classifier on new data
                                │
                    Resume trading with updated model
```

---

## Pipeline (7 Phases)

1. **Data Ingestion** — 4H OHLCV (Dukascopy), economic calendar, macro (FRED), sentiment (VIX/DXY)
2. **Feature Engineering** — 85 features across 5 groups: Technical, Volatility, Macro, Sentiment, Regime
3. **Baseline Models** — LSTM, BiLSTM+Attention, Stacked Ensemble (all predict direction ~50-55%)
4. **Main Model** — XGBoost 5-class regime classifier with adaptive retraining
5. **Shift Detection** — Dual-mode: KS/MMD for scheduled events + ADWIN for unexpected shifts
6. **SHAP Attribution** — TreeSHAP traces each shift to feature group (Technical vs Sentiment vs Macro)
7. **Human-in-the-Loop** — Streamlit dashboard for shift review + selective retraining

---

## Data

- **3 Currency Pairs**: EUR/USD, GBP/JPY, XAU/USD
- **11 Years**: January 2015 — April 2026
- **~18,000 4H bars per pair**
- **Sources**: Dukascopy (OHLCV), FRED (macro), Investing.com (calendar), yfinance (sentiment)
- **16 data files** across 5 categories: price, calendar, macro, sentiment, events

---

## Model Lineup

| Role | Model | Features | Purpose |
|---|---|---|---|
| Baseline 1 | LSTM (unidirectional) | Technical only | Simplest baseline |
| Baseline 2 | BiLSTM + Custom Attention | All features | Medium complexity |
| Baseline 3 | Stacked: RF + BiLSTM → LightGBM meta | All features | Maximum complexity |
| **Main** | **XGBoost 5-class Regime Classifier** | **All 5 groups (85 features)** | **Regime-aware trade filtering** |

---

## Why ShiftGuard Works

1. **Fewer trades = less friction**: 18,663 trades vs 50,040. Each trade costs spread + commission + slippage. Trading less saves $960K in costs over 10 years.
2. **Higher conviction**: Only trades during clear trending regimes (confidence > 55%). Avoids ranging/choppy markets where losses pile up.
3. **Adaptive retraining**: Regime classifier retrains 93 times at detected regime changes. Stale models degrade; ShiftGuard adapts.
4. **Self-normalizing feature awareness**: SHAP identifies that technical features self-correct after shifts, while macro/sentiment features don't. ShiftGuard retrains only when non-technical shifts occur.

---

## Statistical Significance

All results are statistically significant (paired t-test):

| Pair | t-statistic | p-value | 95% CI (PnL difference) |
|---|---|---|---|
| EUR/USD | 8.756 | < 0.000001 | [0.00032, 0.00050] |
| GBP/JPY | 8.343 | < 0.000001 | [0.00041, 0.00066] |
| XAU/USD | 9.702 | < 0.000001 | [0.00058, 0.00088] |

All confidence intervals exclude zero. Results validated across 11 years with no overfitting (positive edge every single year 2015-2026).

---

## Project Structure

```
ShiftGuard/
├── src/
│   ├── features/          # 5 feature groups (technical, volatility, macro, sentiment, regime)
│   ├── models/            # Baselines + main model + experiments
│   ├── detection/         # Dual-mode shift detection engine
│   ├── attribution/       # SHAP analysis
│   ├── retraining/        # Selective retraining strategies
│   └── dashboard/         # Streamlit HITL dashboard
├── data/
│   ├── raw/               # 16 source CSVs
│   └── processed/         # Merged feature matrices
├── results/
│   ├── predictions/       # Model predictions
│   ├── detection/         # Shift detection output
│   ├── attribution/       # SHAP results
│   ├── winrate/           # Win rate experiment (primary results)
│   ├── figures/           # Equity curves, bar charts, stat tests
│   └── regime/            # Regime classification results
└── requirements.txt
```

---

## References

1. Gama, J. et al. (2014). A survey on concept drift adaptation. ACM Computing Surveys.
2. Lu, J. et al. (2019). Learning under concept drift: A review. IEEE TKDE.
3. Ganin, Y. et al. (2016). Domain-adversarial training of neural networks. JMLR.
4. Monarch, R. (2021). Human-in-the-Loop Machine Learning. Manning.
5. Amershi, S. et al. (2014). Power to the people. AI Magazine.
6. Lundberg, S. & Lee, S. (2017). A unified approach to interpreting model predictions. NeurIPS.
7. Tsay, R.S. (2010). Analysis of Financial Time Series. 3rd Edition. Wiley.
