# Delivery Time Analysis & Prediction System

## Overview
This project implements an end-to-end **delivery time prediction pipeline** using real-world logistics data.  
The system covers **data preprocessing, feature engineering, model training, evaluation, selection, and production-ready inference**.

The primary objective is to predict **delivery duration (in minutes)** based on distance, traffic conditions, temporal factors, and weather.

---

## Problem Statement
Accurate delivery time estimation is critical for:
- Customer experience optimization
- Operational planning
- Logistics and fleet efficiency

Delivery time is influenced by **non-linear and interacting factors**, making this a regression problem well-suited to ensemble learning methods.

---

## Dataset
Primary dataset used:
- **Amazon Delivery Dataset (~43,000 rows)**

The dataset contains:
- Store and drop-off coordinates
- Order timestamps
- Traffic conditions
- Weather categories
- Actual delivery time (target variable)

---

## Feature Engineering

The following features were engineered and used for modeling:

| Feature | Description |
|------|------------|
| `distance_km` | Haversine distance between store and drop location |
| `hour` | Hour of day extracted from order timestamp |
| `day_of_week` | Integer-encoded weekday (0–6) |
| `is_weekend` | Binary weekend indicator |
| `traffic_level` | Ordinal traffic encoding (Low → Jam) |
| `Weather` | One-hot encoded categorical feature |

All transformations applied during training are **reused during inference** to ensure consistency.

---

## Modeling Strategy

Multiple regression models were trained and evaluated:

- Linear Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- **XGBoost (Selected Best Model)**

### Evaluation Metrics
Models were compared using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The best model was selected based on **lowest test MAE**, prioritizing real-world error minimization.

---

## Results Summary

| Model | MAE | RMSE | R² |
|-----|-----|------|----|
| Linear Regression | High | High | Low |
| KNN | Moderate | Moderate | Low |
| Random Forest | Overfitting observed | Unstable | Low |
| **XGBoost** | **Best** | **Best** | **0.33** |

**XGBoost demonstrated the strongest generalization performance**, capturing non-linear interactions between traffic, time, and distance.

---

## Project Structure

```
Delivery_ETA
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── model.py
│   └── predict.py
│   └── preprocess.py
│   └── analysis.py
│   └── eda_summary.py
├── models/
│   ├── amazon_best_model.pkl
│   └── Weather_encoder.pkl
├── results/
│   ├── figures/
│   └── tables/
├── README.md
└── requirements.txt
```

---


## Inference & Prediction

CLI Execution
```bash
python src/predict.py
```

Programmatic Usage
```bash
from src.predict import predict_delivery_time

predict_delivery_time(
    distance_km=12.5,
    hour=16,
    day_of_week=4,
    weather="Sunny",
    traffic_level="High",
    is_weekend=1
)

```
The inference pipeline automatically:

- Computes distance if provided longitude and latitude only

- Normalizes categorical inputs

- Applies the same encoders used during training

---

## Key Insights

- Traffic level is the strongest predictor of delivery time.

- Temporal features (hour and weekday) show consistent influence.

- Distance alone is insufficient, confirming the need for contextual features.

- Ensemble models significantly outperform linear baselines.

- Artifacts & Reproducibility

- Trained model and encoders are serialized using joblib

- All plots and tables are stored under results/

- Inference logic is fully decoupled from training logic

## Future Improvements

- Residual and error distribution analysis

- Integration of real-time traffic data

- API deployment (FastAPI)

- Continuous retraining with additional datasets
