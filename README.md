# M5 Retail Sales Forecasting

A machine learning project for forecasting Walmart retail sales using the Kaggle M5 Forecasting competition dataset.

## Project Overview

The goal is to forecast daily sales for thousands of retail product-store combinations over a 28-day horizon.

This project explores time-series feature engineering, LightGBM modeling, validation, and recursive multi-step forecasting.

## Methodology

- **Model:** LightGBM
- **Feature engineering:** Calendar-based features, historical sales features, and other time-related predictors
- **Forecasting strategy:** Recursive 28-day prediction
- **Evaluation:** Kaggle competition leaderboard and local validation metrics

## Workflow

1. Load and inspect the competition datasets.
2. Process sales history, calendar information, and selling prices.
3. Construct features for the forecasting model.
4. Train the LightGBM model.
5. Generate daily forecasts for the 28-day prediction horizon.
6. Format predictions into the Kaggle submission template.
7. Submit predictions and review leaderboard performance.

## Results

Kaggle leaderboard results from the submitted prediction file:

| Metric | Score |
|---|---:|
| Public Score | 2.87251 |
| Private Score | 3.03081 |

These are the scores reported for the submitted solution. They are not local MAE or RMSE measurements.

## Repository Structure

```text
m5-retail-sales-forecasting/
├── README.md
└── notebook2c59687c3b.ipynb
```

## Tools

- Python
- Pandas
- NumPy
- LightGBM
- Kaggle Notebooks

## Limitations and Future Improvements

- Explore additional lag and rolling-window features.
- Improve validation for time-dependent data.
- Investigate performance on intermittent and zero-sales products.
- Compare alternative forecasting strategies and feature sets.

## Disclaimer

This repository documents an educational machine learning project using the Kaggle M5 Forecasting competition dataset. The reported leaderboard scores correspond to the submitted solution and may not be reproducible without the same data processing and modeling configuration.
