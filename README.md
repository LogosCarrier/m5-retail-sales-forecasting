# M5 Retail Sales Forecasting

A LightGBM-based retail demand forecasting pipeline for the [Kaggle M5 Forecasting — Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy).

This project formulates daily unit-sales prediction as a supervised learning problem and implements feature engineering, sampled-series model training, recursive multi-step inference, and submission validation.

## 1. Problem Definition

The objective is to predict daily sales for Walmart product–store combinations over a 28-day horizon.

The M5 dataset contains hierarchical retail sales across products, departments, categories, stores, and states. Demand patterns vary substantially across series, including products with frequent zero-sales observations.

The competition evaluates forecasts using **Weighted Root Mean Squared Scaled Error (WRMSSE)**.

## 2. Modeling Pipeline

### Feature Engineering

The pipeline constructs 36 candidate features from sales history, calendar information, and selling prices. Constant features are removed before model fitting, leaving 33 features in the recorded training run.

Feature groups include:

| Feature group | Examples |
|---|---|
| Calendar | Weekday, month, year, day of month, weekend indicator |
| Events | Event indicator and a local event-window indicator |
| Historical lags | 28-, 56-, 84-, and 364-day lags |
| Rolling statistics | Historical means, standard deviations, and zero-sales ratios over 7, 28, 56, and 364 days |
| Demand behavior | Non-zero mean sales and short-to-long-term mean ratios |
| Price and promotions | Selling price, log-transformed price, missing-price indicator, and state-specific SNAP features |

Historical features are calculated from the available sales window for each training or forecast date. The model uses the same selected feature list during inference.

### Training Strategy

The recorded configuration uses a reproducible random sample of 5,000 product–store series.

| Parameter | Value |
|---|---:|
| Model | `LGBMRegressor` |
| Objective | Regression |
| Candidate features | 36 |
| Features retained | 33 |
| Training series | 5,000 |
| Number of estimators | 1,600 |
| Learning rate | 0.02 |
| Number of leaves | 31 |
| Subsample parameter | 0.8 |
| Column-sampling parameter | 0.8 |
| Random seed | 42 |

Training examples are constructed at regularly spaced historical target dates, with target dates ending before the final 28-day validation window.

### Recursive Forecasting

The forecasting function generates predictions one day at a time.

1. Construct features from the currently available history.
2. Predict the next day's sales.
3. Clip negative predictions to zero.
4. Append the predicted values to the historical window.
5. Repeat until all 28 forecast days are generated.

This procedure is applied to both the validation horizon and the final evaluation horizon.

### Submission Validation

The pipeline aligns predictions with the competition template using series identifiers and constructs the required `F1`–`F28` columns.

Before submission, it checks:

- Expected submission shape: `60,980 × 29`
- Column names and order against the original template
- Identifier order against the original template
- Absence of missing forecast values
- Non-negative forecast values

## 3. Competition Results

| Evaluation | Score |
|---|---:|
| Public leaderboard | 2.87251 |
| Private leaderboard | 3.03081 |

These are the reported scores for the submitted solution, not locally calculated MAE or RMSE values.

Leaderboard results are provided as an outcome of this particular run. They do not, by themselves, establish performance against a baseline or demonstrate that each feature improves forecasting accuracy.

## 4. Repository Contents

```text
m5-retail-sales-forecasting/
├── README.md
└── notebook2c59687c3b.ipynb
```

The notebook contains the implementation of the training and inference pipeline. The trained model, selected feature list, feature-importance table, run summary, and generated submission are produced as separate artifacts by the notebook.

## 5. Reproducibility

The implementation was developed in Kaggle Notebooks and expects the M5 competition files to be available in the Kaggle input directory.

Required Python packages include:

- NumPy
- Pandas
- LightGBM
- Joblib

The notebook uses a fixed random seed for the training-series sample and records the selected feature list alongside the serialized model.

To reproduce the run, configure the competition dataset, execute the notebook cells in order, and verify the generated submission before submitting it to Kaggle.

## 6. Limitations and Future Work

Several aspects warrant further evaluation:

- **Sampling coverage:** Training is performed on 5,000 sampled series rather than all 30,490 series.
- **Baseline comparison:** The recorded leaderboard scores need to be complemented by controlled comparisons against seasonal-naive and simpler feature-based baselines.
- **Feature ablation:** The contribution of the added lag, rolling-statistic, calendar, and price features should be measured independently.
- **Demand heterogeneity:** Performance should be evaluated separately for intermittent-demand and zero-sales-heavy series.
- **Validation methodology:** The forecasting procedure and feature construction should be tested under a consistent time-based validation protocol.

## 7. Project Scope

This repository documents an applied machine learning project focused on retail demand forecasting, feature engineering, recursive inference, and competition submission integrity. The current implementation is an experimental forecasting pipeline rather than a production-ready demand planning system.
