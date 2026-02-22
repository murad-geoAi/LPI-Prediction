# Liquefaction Potential Index (LPI) Regression

## Methods and Results (PCA + GroupKFold)

**Generated:** February 2026 (pipeline run: 2026-02-23)  
**Dataset:** `combined_6.5_7_7.5 (1).csv` (fallback: `combined_6.5_7_7.5.csv`)  
**Code:** `lpi_regression_pipeline.py`  
**Outputs:** `results/`

---

## 1. Overview

This work evaluates supervised regression models for predicting the **Liquefaction Potential Index (LPI)** from depth-resolved borehole geotechnical parameters. Because the predictor space is highly collinear (multiple depths; repeated scenario variables), we use **principal component analysis (PCA)** within each model pipeline.

The dataset contains **three scenario rows per borehole** (Mw = 6.5, 7.0, 7.5). To prevent **data leakage**, cross-validation is performed using **grouped folds by BoreholeID**, ensuring no borehole contributes to both training and validation data.

---

## 2. Data

### 2.1 Structure

- **Raw rows:** 1,659  
- **Raw boreholes:** 553 (each borehole appears in exactly 3 scenario rows)  
- **Deduplication:** exact duplicates in *feature space* removed (to avoid duplicates spanning folds)  
- **Rows after deduplication:** 1,452  
- **Boreholes after deduplication:** 484 (still 3 scenario rows per borehole)

### 2.2 Features and target

- **Target:** `lpi` (Liquefaction Potential Index)
- **Excluded from predictors (target-derived):** `lpi_component_3/6/9/12/15/18`
- **Spatial metadata (not used as predictors):** `Latitude`, `Longitude`, `BoreholeID`
- **Predictors (24 columns):**
  - `FC_3 … FC_18` (fines content, %) at 6 depths
  - `N_3 … N_18` (SPT N-value) at 6 depths
  - `M_3 … M_18` (Mw) repeated across depths (identical within each row)
  - `amax_3 … amax_18` (PGA, g) repeated across depths (identical within each row)

### 2.3 Target summary (after deduplication; n = 1,452)

| Statistic | Value |
| --- | ---: |
| Mean | 33.47 |
| Std | 17.00 |
| Min | 0.00 |
| 25th percentile | 20.72 |
| Median | 36.47 |
| 75th percentile | 47.40 |
| Max | 60.96 |

---

## 3. Methods

### 3.1 Leakage controls

1. **Group-wise CV:** `GroupKFold(n_splits=10)` with groups = `BoreholeID` (all three scenario rows per borehole remain in the same fold).
2. **Fold-contained preprocessing:** scaling and PCA are fit on training folds only via an `sklearn.Pipeline`.
3. **PCA component selection without peeking:** models use `PCA(n_components=0.95)`, so the number of components is determined from training data within each fold (variance threshold = 95%).

### 3.2 Pipeline

Per fold, each model uses the same preprocessing:

```
StandardScaler → PCA(variance threshold = 0.95) → Regressor
```

### 3.3 Candidate models

- Linear Regression
- Ridge (α = 1.0)
- Lasso (α = 0.1)
- ElasticNet (α = 0.1, l1_ratio = 0.5)
- Random Forest (n_estimators = 200)
- Gradient Boosting (n_estimators = 200)
- SVR (RBF; C = 10, ε = 0.5)
- XGBoost (n_estimators = 200) if available

### 3.4 Metrics

- **R²:** \( R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} \)
- **RMSE:** \( \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2} \)
- **MAE:** \( \frac{1}{n}\sum_i |y_i - \hat{y}_i| \)

All summary results are reported as **mean ± standard deviation** over the 10 grouped folds. For the best model, **out-of-fold (OOF)** predictions are aggregated across folds to compute a single held-out estimate.

---

## 4. Results

### 4.1 PCA summary (reporting fit on full deduplicated dataset)

The 95% variance criterion yields **6 principal components**. The explained variance ratio for PCs 1–6 is:

| PC | Variance explained (%) |
| --- | ---: |
| PC1 | 50.00 |
| PC2 | 29.50 |
| PC3 | 7.26 |
| PC4 | 4.35 |
| PC5 | 3.04 |
| PC6 | 1.67 |

### 4.2 Cross-validation performance (GroupKFold, k = 10)

| Model | R² (mean ± std) | RMSE (mean ± std) | MAE (mean ± std) | Train R² (mean ± std) |
| --- | --- | --- | --- | --- |
| **SVR (RBF)** | **0.9946 ± 0.0010** | **1.240 ± 0.123** | **0.856 ± 0.044** | 0.9962 ± 0.0001 |
| XGBoost | 0.9904 ± 0.0027 | 1.635 ± 0.212 | 1.130 ± 0.127 | 0.9980 ± 0.0007 |
| Gradient Boosting | 0.9900 ± 0.0026 | 1.677 ± 0.200 | 1.178 ± 0.114 | 0.9977 ± 0.0001 |
| Random Forest | 0.9896 ± 0.0032 | 1.697 ± 0.252 | 1.161 ± 0.140 | 0.9987 ± 0.0000 |
| Ridge (α = 1.0) | 0.9559 ± 0.0224 | 3.472 ± 0.868 | 2.364 ± 0.197 | 0.9584 ± 0.0022 |
| Linear Regression | 0.9559 ± 0.0224 | 3.472 ± 0.869 | 2.364 ± 0.197 | 0.9584 ± 0.0022 |
| ElasticNet | 0.9559 ± 0.0211 | 3.485 ± 0.825 | 2.405 ± 0.195 | 0.9582 ± 0.0022 |
| Lasso (α = 0.1) | 0.9558 ± 0.0221 | 3.479 ± 0.857 | 2.392 ± 0.197 | 0.9582 ± 0.0022 |

### 4.3 Best model (SVR RBF): OOF metrics

| Metric | Value |
| --- | ---: |
| OOF R² | 0.9946 |
| OOF RMSE | 1.246 |
| OOF MAE | 0.857 |

### 4.4 Model differences (Friedman test)

The per-fold R² distributions differ significantly across models:

- **Friedman χ²:** 59.30  
- **p-value:** 2.08e-10

---

## 5. Figures and artifacts (journal-ready exports)

All figures are saved as **300 DPI PNG** and **vector PDF**:

- `results/01_eda_overview.(png|pdf)` — LPI distribution, Q–Q plot, boxplot by Mw
- `results/02_correlation_heatmap.(png|pdf)` — correlation matrix (features + target)
- `results/03_pca_variance.(png|pdf)` — scree + cumulative explained variance
- `results/04_pca_loadings.(png|pdf)` — PCA loading heatmap (PC1–PC6)
- `results/05_model_comparison.(png|pdf)` — model comparison (R² / RMSE / MAE)
- `results/06_fold_r2_boxplots.(png|pdf)` — per-fold R² distributions
- `results/07_train_vs_test_r2.(png|pdf)` — train vs test R² (overfitting check)
- `results/08_best_model_residuals.(png|pdf)` — OOF diagnostics: actual vs predicted, residuals, Q–Q
- `results/09_pca_importance.(png|pdf)` — PC–LPI correlations and PC variance
- `results/10_feature_importance.(png|pdf)` — PCA-weighted feature importance
- `results/11_spatial_lpi_map.(png|pdf)` — spatial maps (actual / predicted / residual)

Tabular artifacts:

- `results/cv_results_summary.csv`
- `results/cv_results_numeric.csv`
- `results/pca_component_correlations.csv`
- `results/feature_importance_pca.csv`
- `results/spatial_predictions.csv` (includes `BoreholeID`, scenario `M`, and `amax`)
- `results/results.json`

---

## 6. Notes and limitations

- **Scenario duplication is handled explicitly.** GroupKFold prevents the three scenario rows of each borehole from leaking across folds.
- **Repeated columns (`M_d`, `amax_d`) are identical within each row.** PCA/feature-importance plots distribute importance across these duplicated columns; for interpretability, it is reasonable to collapse them to single variables (Mw, PGA) in a follow-up analysis.
- **Model selection bias:** the “best model” is selected by cross-validation mean R²; a fully unbiased estimate of the model-selection procedure would require nested CV.

---

## 7. Reproducibility

```bash
python lpi_regression_pipeline.py
```

Dependencies (typical): `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, and optionally `xgboost`.
