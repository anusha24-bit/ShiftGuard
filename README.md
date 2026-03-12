# ShiftGuard

**A Human-in-the-Loop Framework for Distribution Shift Detection, Attribution, and Adaptive Retraining in Non-Stationary Forex Markets**

---

## Overview

ShiftGuard is a machine learning pipeline that detects, explains, and adapts to distribution shifts in forex markets. It combines three capabilities that have traditionally been developed in isolation:

1. **Event-Aware Shift Detection** — A dual-mode detection engine that separates scheduled event-driven shifts (e.g., CPI reports, Non-Farm Payroll) from unexpected anomalies (e.g., geopolitical crises).
2. **Interpretable Attribution** — A SHAP-based attribution layer that traces each detected shift back to the specific feature groups that changed.
3. **Human-in-the-Loop Oversight** — A Streamlit dashboard where users can review alerts, confirm or reject detected shifts, and trigger selective model retraining.

## Why Forex?

Unlike stock markets, forex operates continuously across global sessions, is driven by sovereign-level forces like central bank policy and geopolitical events, and experiences shifts at every scale — from catastrophic disruptions like COVID-19 and Brexit, to routine changes caused by interest rate differentials, news flow, and scheduled macro releases. These shifts differ in timing, magnitude, and feature signatures, yet almost no existing system treats them differently.

---

## Project Structure

```
ShiftGuard/
├── README.md
├── PROJECT_ABSTRACT.md
├── requirements.txt
├── data/
│   ├── raw/                  # Raw forex OHLCV and macro data
│   └── processed/            # Feature-engineered datasets
├── src/
│   ├── data/
│   │   ├── fetch_data.py     # Data collection scripts (forex pairs, macro indicators)
│   │   └── preprocess.py     # Feature engineering and windowing
│   ├── detection/
│   │   ├── statistical.py    # KS, MMD, ADWIN, DDM shift detectors
│   │   └── event_aware.py    # Scheduled-event detection mode
│   ├── attribution/
│   │   └── shap_explain.py   # SHAP-based feature attribution for detected shifts
│   ├── retraining/
│   │   └── adaptive.py       # Selective retraining logic
│   └── utils/
│       └── helpers.py        # Shared utilities
├── dashboard/
│   └── app.py                # Streamlit human-in-the-loop dashboard
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis
└── tests/
    └── ...                   # Unit and integration tests
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/anusha24-bit/ShiftGuard.git
   cd ShiftGuard
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   # venv\Scripts\activate          # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Data Setup

Forex and macroeconomic data can be fetched using the data collection scripts:

```bash
python src/data/fetch_data.py
```

> **Note:** You may need API keys for certain data providers. See the script header for configuration details.

---

## Usage

### Run the Detection Pipeline

```bash
python src/detection/statistical.py --pair EUR/USD --window 30
```

### Run the Attribution Analysis

```bash
python src/attribution/shap_explain.py --shift-id <SHIFT_ID>
```

### Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard allows you to:
- View detected distribution shifts in real time
- Inspect SHAP-based feature attributions for each shift
- Confirm or reject shift alerts
- Trigger selective model retraining on approved shifts

---

## Key Technologies

- **Shift Detection:** ADWIN, DDM, Kolmogorov-Smirnov (KS), Maximum Mean Discrepancy (MMD)
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Dashboard:** Streamlit
- **ML Framework:** scikit-learn / XGBoost (base models)
- **Data:** Forex OHLCV, macroeconomic indicators (CPI, NFP, interest rates)

---

## Team

| Name | Email |
|------|-------|
| Sohan Mahesh | mahesh.so@northeastern.edu |
| Anusha Ravi Kumar | ravikumar.anu@northeastern.edu |
| Dishaben Manubhai Patel | patel.dishabe@northeastern.edu |

---

## References

- Gama, J., et al. (2014). A survey on concept drift adaptation.
- Lu, J., et al. (2019). Learning under concept drift: A review.
- Ganin, Y., et al. (2016). Domain-adversarial training of neural networks.
- Monarch, R. (2021). *Human-in-the-Loop Machine Learning*.
- Amershi, S., et al. (2014). Power to the people: The role of humans in interactive machine learning.

---

## License

This project is developed as part of coursework at Northeastern University.
