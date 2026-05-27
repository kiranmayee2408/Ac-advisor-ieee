# AC Advisor — Automobile A/C Energy Prediction

**IEEE-published** ML system for real-time automobile air conditioning energy prediction, with SHAP explainability, a live Streamlit app, and a comparative benchmark of 6 ML/DL architectures on 37,000+ real OBD-II records.

[![IEEE](https://img.shields.io/badge/IEEE-Published_2026-blue?logo=ieee)](https://ieeexplore.ieee.org/document/11393777)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red?logo=streamlit)](https://ac-advisor-app-by-csun-hemanth-kiranmayee.streamlit.app)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](.)
[![License](https://img.shields.io/badge/license-MIT-green)](.)

> 🚀 **Live App:** https://ac-advisor-app-by-csun-hemanth-kiranmayee.streamlit.app
>
> 📄 **IEEE Paper:** https://ieeexplore.ieee.org/document/11393777

---

## The Problem

Automobile air conditioning accounts for **5–25% of total vehicle fuel consumption** depending on ambient temperature and driving behavior. Most existing solutions use simplified physics models that ignore real driving dynamics — they can't predict A/C load accurately across varied conditions.

We asked: *can a data-driven ML model trained on real OBD-II telemetry predict A/C energy consumption accurately enough to be useful in production?*

---

## What We Built

An end-to-end ML system trained on **37,000+ real OBD-II telemetry records** collected across **1,000+ miles of driving data**, with:

- Comparative benchmark of **6 ML/DL architectures**
- **SHAP-based explainability** — shows which features drive each prediction
- **Live Streamlit app** for real-time scenario prediction
- **Docker deployment** for reproducibility
- **FastAPI inference endpoint** for programmatic access

---

## Model Benchmark Results

| Model | MAPE | R² | Inference Latency |
|---|---|---|---|
| **CatBoost** ✅ | **< 10%** | **0.94** | ~3ms |
| XGBoost | 11.2% | 0.92 | ~4ms |
| Random Forest | 13.4% | 0.89 | ~12ms |
| Transformer (custom) | 10.8% | 0.91 | ~28ms |
| LSTM | 14.1% | 0.88 | ~18ms |
| Linear Baseline | 22.3% | 0.71 | <1ms |

**CatBoost selected for production:** best MAPE with lowest inference latency among non-linear models. The Transformer matched it on accuracy but at 9x the latency — not viable for real-time inference.

---

## System Architecture

```
OBD-II Telemetry (37K+ records, 1000+ miles)
         │
         ▼
┌─────────────────────────┐
│  Data Pipeline          │  Bronze/Silver/Gold on AWS
│  (AWS S3 + Glue)        │  Schema validation, deduplication
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Feature Engineering    │  Thermal load index
│                         │  RPM efficiency ratio
│                         │  Driving mode classification
│                         │  Ambient × AC interaction terms
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Model Training         │  6 architectures benchmarked
│  + Evaluation           │  MLflow experiment tracking
│                         │  Cross-validation, MAPE/R² metrics
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Streamlit App          │  Real-time scenario prediction
│  + SHAP Explainability  │  Feature importance per prediction
│  + FastAPI Endpoint     │  Programmatic inference access
└─────────────────────────┘
```

---

## Key Features

### SHAP Explainability
Every prediction includes a breakdown of which features drove the result:

```
Prediction: 1.87 kW

SHAP contributions:
  ambient_temp_c   +0.62 kW  ████████████  (hot day = high load)
  ac_on            +0.41 kW  ████████      (AC state)
  speed_mph        +0.28 kW  █████         (aerodynamic heat)
  throttle_pct     +0.19 kW  ████          (engine load)
  engine_rpm       -0.07 kW  ██            (efficient RPM range)
  base value        0.44 kW
```

This explainability layer is what separates this from a black-box model — drivers and fleet managers can see *why* the A/C is consuming energy and act on it.

### Live Scenario Prediction
The Streamlit app lets you input any driving scenario and get an instant prediction:

- Try: *65 mph, 38°C ambient, A/C on* → predicted ~2.1 kW, cost ~$0.25/hr
- Try: *city driving at 25 mph, 28°C* → predicted ~1.4 kW, fuel penalty ~7%

---

## Feature Engineering

| Feature | Description | SHAP Importance |
|---|---|---|
| `ambient_temp_c` | Outside temperature — drives cabin thermal load | #1 |
| `ac_on` | Binary A/C state | #2 |
| `speed_mph` | Aerodynamic heat gain at speed | #3 |
| `throttle_pct` | Engine load proxy | #4 |
| `engine_rpm` | Compressor drive speed | #5 |
| `thermal_load_index` | Derived: temp × ac_on interaction | #6 |
| `rpm_efficiency` | Derived: rpm / max(speed, 1) | #7 |

---

## Dataset

- **37,000+ OBD-II records** from real driving sessions
- **1,000+ miles** of urban, suburban, and highway driving
- **Ambient temperature range:** 18°C – 44°C
- **Recording rate:** 1 Hz
- **Vehicles:** Multiple sedan models (2018–2022)

---

## Quickstart

```bash
git clone https://github.com/kiranmayee2408/Ac-advisor-ieee.git
cd Ac-advisor-ieee
pip install -r requirements.txt

# Run the Streamlit app locally
streamlit run app.py
```

### Docker
```bash
docker build -t ac-advisor .
docker run -p 8501:8501 ac-advisor
```

---

## Project Structure

```
Ac-advisor-ieee/
├── ac_advisor/          # Core ML pipeline: preprocessing, training, evaluation
├── data/                # OBD-II telemetry dataset
├── models/              # Trained model artifacts (CatBoost + others)
├── assets/              # SHAP plots, benchmark charts, figures
├── catboost_info/       # CatBoost training logs and metadata
├── app.py               # Streamlit app — live prediction interface
├── Dockerfile           # Container for reproducible deployment
└── requirements.txt
```

---

## Authors

Built as part of graduate research at **California State University, Northridge**.

📄 **Full paper:** [AC Advisor: A Comparative Evaluation of ML and State Space Models for Automobile Air Conditioning Energy Prediction](https://ieeexplore.ieee.org/document/11393777) — IEEE, 2026

---

## License

MIT
