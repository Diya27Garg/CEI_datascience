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



# Week 3 — Customer Intelligence System
### Country Socio-Economic Analysis using Machine Learning

## Dataset
Kaggle — Unsupervised Learning on Country Data
https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data
167 countries × 9 features (GDP, child mortality, life expectancy, income, etc.)

## What We Did
- Exploratory Data Analysis — distributions, heatmap, GDP comparisons
- Data cleaning, median imputation and StandardScaler
- Elbow Method to confirm k=3
- K-Means clustering — grouped countries into 3 development tiers
- DBSCAN as comparison model
- PCA for 2D visualization
- Random Forest + XGBoost on cluster labels
- Feature importance analysis
- 
## Results

| Cluster | Countries | Avg GDP | Child Mortality | Life Expectancy |
|---|---|---|---|---|
|  Underdeveloped | 47 | $1,922 | 92.96 | 59 years |
|  Developing | 84 | $6,486 | 21.93 | 73 years |
|  Developed | 36 | $42,494 | 5.00 | 80 years |

- Random Forest : 100% accuracy
- XGBoost : 94.12% accuracy
- Silhouette Score : 0.2833

## Key Finding
Child mortality (0.258) is a stronger development predictor than 
GDP (0.198) — healthcare outcomes matter more than economic output.
