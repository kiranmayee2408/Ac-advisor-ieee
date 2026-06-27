# AC Advisor — IEEE-Published ML System for Automobile A/C Energy Prediction

[![IEEE Published](https://img.shields.io/badge/IEEE-Published_2026-blue?style=flat&logo=ieee)](https://ieeexplore.ieee.org/document/11393777)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-red?style=flat&logo=streamlit)](https://ac-advisor-app-by-csun-hemanth-kiranmayee.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **IEEE Publication:** *AC Advisor: A Comparative Evaluation of ML and State Space Models for Automobile Air Conditioning Energy Prediction* — [Read on IEEE Xplore →](https://ieeexplore.ieee.org/document/11393777)

A production ML system that predicts automobile A/C energy consumption from OBD-II telemetry data, with SHAP explainability and an interactive Streamlit "What-If" coach.

---

## Live Demo

**[→ Try the live Streamlit app](https://ac-advisor-app-by-csun-hemanth-kiranmayee.streamlit.app)**

Upload or simulate a drive, adjust A/C parameters, and get real-time energy predictions with SHAP feature explanations.

---

## Research Summary

This project was accepted and published at IEEE in 2026. The study addresses a gap in automotive A/C energy modelling: most prior work uses physics-based simulations that don't generalize across vehicle types and real-world drive conditions.

**Dataset:** 37,000+ OBD-II telemetry records collected across urban, highway, and mixed driving cycles.

**Key Finding:** CatBoost with hyperparameter tuning achieved the best balance of accuracy and inference speed for production deployment (MAPE < 10%), while the Transformer model showed competitive accuracy but impractical latency for real-time use.

---

## Model Comparison Results

| Model | MAPE (%) | RMSE | R² | Inference Time |
|---|---|---|---|---|
| **CatBoost (tuned)** | **7.4%** | 0.31 | **0.94** | 2ms |
| XGBoost | 8.1% | 0.34 | 0.93 | 1ms |
| Random Forest | 9.2% | 0.39 | 0.91 | 4ms |
| Linear Regression | 14.7% | 0.61 | 0.79 | <1ms |
| LSTM | 8.8% | 0.37 | 0.92 | 18ms |
| Transformer | 7.9% | 0.33 | 0.93 | 47ms |

**Winner: CatBoost** — MAPE < 10% threshold met, best R², practical inference time for real-time edge deployment.

---

## Architecture

```
OBD-II Telemetry (37K+ records)
        │
        ▼
Feature Engineering (features.py)
  ├── Vehicle speed, RPM, throttle position
  ├── Ambient temp, cabin setpoint, compressor load
  ├── Rolling window aggregates (5s, 30s, 60s)
  └── Derived: ΔT (cabin vs ambient), load factor
        │
        ▼
Model Training & Comparison
  ├── CatBoost  ← production model
  ├── XGBoost
  ├── Random Forest
  ├── Linear Regression (baseline)
  ├── LSTM
  └── Transformer
        │
        ▼
SHAP Explainability
  ├── Global feature importance
  ├── Per-prediction SHAP waterfall
  └── Interaction effects (speed × ΔT)
        │
        ▼
Streamlit App (What-If Coach)
  ├── Replay historical drives
  ├── Simulate A/C setting changes
  └── Real-time SHAP explanations
```

---

## SHAP Feature Importance

Top features driving A/C energy consumption (from SHAP analysis across 37K records):

| Rank | Feature | Contribution |
|---|---|---|
| 1 | ΔT (cabin setpoint − ambient) | 34% |
| 2 | Vehicle speed (rolling 30s avg) | 21% |
| 3 | Compressor duty cycle | 17% |
| 4 | Engine RPM | 11% |
| 5 | Solar load proxy | 9% |
| 6 | Cabin volume (vehicle class) | 8% |

**Key insight:** The temperature differential (ΔT) is by far the dominant driver. Reducing setpoint by 2°C during highway driving reduces predicted energy draw by ~18%.

---

## Quickstart

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate ac-advisor
streamlit run app.py
```

### Option B — Docker

```bash
docker build -t ac-advisor .
docker run --rm -p 8501:8501 ac-advisor
```

### Option C — pip

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
Ac-advisor-ieee/
├── app.py                    # Streamlit UI — replay, what-if, SHAP coach
├── ac_advisor/
│   ├── features.py           # Feature engineering — single source of truth
│   ├── predictor.py          # Model loading, predict_now(), predict_sim()
│   ├── comfort.py            # ΔT comfort bands and scoring
│   └── coach.py              # Nearest-context recall and action logging
├── models/
│   └── catboost_model.cbm    # Trained CatBoost model
├── data/
│   └── EnergyPredictionDataset_ReadyForModel.csv
├── assets/                   # SHAP plots, architecture diagrams
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

**Why CatBoost over XGBoost?** CatBoost handles categorical features (vehicle class, drive mode) natively without manual encoding, and showed better calibration on the tail of the distribution — exactly where dangerous high-load predictions occur.

**Why heuristic features over raw signals?** Raw OBD-II signals are noisy at 10Hz. Rolling window aggregates (5s, 30s, 60s) capture the thermal inertia of the A/C system and reduce prediction variance by ~30%.

**Why Streamlit over FastAPI for the demo?** The target users (automotive engineers, fleet operators) need interactive what-if exploration, not programmatic API access. Streamlit enables zero-install deployment for non-technical stakeholders.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{lokam2026acadvisor,
  title     = {AC Advisor: A Comparative Evaluation of ML and State Space Models 
               for Automobile Air Conditioning Energy Prediction},
  author    = {Lokam, Kiranmayee and Tulabandula, Hemanth Kumar},
  booktitle = {IEEE},
  year      = {2026},
  url       = {https://ieeexplore.ieee.org/document/11393777}
}
```
