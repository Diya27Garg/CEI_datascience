## WEEK 2
# Tesla Deliveries ML Pipeline (2015–2025)

End-to-end ML pipeline on Tesla delivery and production data.

## Dataset
https://www.kaggle.com/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025

## Pipeline
Preprocessing → EDA → Feature Engineering → Regression Modeling → 
Hyperparameter Tuning → Time Series Forecasting → Segmented Analysis

## Key findings
- Caught and resolved two data leakage cases
- RandomForest best model: Test RMSE 314, R² 0.99
- ADF test confirmed stationarity (p=0.0001), d=0 used in ARIMA
- Segmented analysis confirmed uniform feature importance across all vehicle models
