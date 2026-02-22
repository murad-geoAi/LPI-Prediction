"""
LPI Regression Pipeline: GroupKFold Cross-Validation with PCA + Multiple Models
Dataset: combined_6.5_7_7.5 (1).csv (or combined_6.5_7_7.5.csv)
Target: lpi (Liquefaction Potential Index)
"""

# ============================================================
# SECTION 0: IMPORTS & CONFIGURATION
# ============================================================
import os
from pathlib import Path
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
np.random.seed(42)

# Reproducible paths (relative to this file)
BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Prefer a clean filename if present, otherwise fall back to the "(1)" file.
_csv_candidates = [
    BASE_DIR / "combined_6.5_7_7.5.csv",
    BASE_DIR / "combined_6.5_7_7.5 (1).csv",
]
CSV_PATH = next((p for p in _csv_candidates if p.exists()), None)
if CSV_PATH is None:
    raise FileNotFoundError(
        "Could not find dataset CSV. Expected one of: "
        + ", ".join(str(p) for p in _csv_candidates)
    )

# Journal-style plotting defaults (kept lightweight and portable)
sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
plt.rcParams.update(
    {
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    }
)


def savefig(fig, stem: str, dpi: int = 300):
    """Save each figure as high-res PNG + vector PDF."""
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

print("=" * 60)
print("LPI Regression Pipeline — GroupKFold + PCA + Models")
print("=" * 60)

# ============================================================
# SECTION 1: DATA LOADING & EXPLORATION
# ============================================================
print("\n[1] Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"  Dataset shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Define feature groups
META_COLS = ["Latitude", "Longitude", "BoreholeID"]
TARGET_COL = "lpi"
LPI_COMP_COLS = [c for c in df.columns if c.startswith("lpi_component")]
FEATURE_COLS = [
    c for c in df.columns if c not in META_COLS + [TARGET_COL] + LPI_COMP_COLS
]

print(f"\n  Meta cols ({len(META_COLS)}): {META_COLS}")
print(f"  Feature cols ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"  LPI component cols ({len(LPI_COMP_COLS)}): {LPI_COMP_COLS}")
print(f"  Target: {TARGET_COL}")

# Basic statistics
print("\n--- Target (LPI) Statistics ---")
print(df[TARGET_COL].describe().to_string())
print(f"\n  Missing values in features: {df[FEATURE_COLS].isnull().sum().sum()}")
print(f"  Missing values in target  : {df[TARGET_COL].isnull().sum()}")

# Duplicate rows
n_dup = df.duplicated(subset=FEATURE_COLS).sum()
print(f"  Duplicate rows (by features): {n_dup}")
df = df.drop_duplicates(subset=FEATURE_COLS).copy()
print(f"  Shape after dedup: {df.shape}")
print("\n--- Target (LPI) Statistics (after dedup) ---")
print(df[TARGET_COL].describe().to_string())

# ---- EDA Plots ----
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# LPI distribution
axes[0].hist(
    df[TARGET_COL], bins=40, color="steelblue", edgecolor="white", linewidth=0.5
)
axes[0].set_title("LPI Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("LPI Value")
axes[0].set_ylabel("Frequency")
axes[0].axvline(
    df[TARGET_COL].mean(),
    color="red",
    linestyle="--",
    label=f"Mean={df[TARGET_COL].mean():.2f}",
)
axes[0].legend()

# Q-Q plot of LPI
(osm, osr), (slope, intercept, r) = stats.probplot(df[TARGET_COL], dist="norm")
axes[1].scatter(osm, osr, s=8, alpha=0.5, color="steelblue")
axes[1].plot(osm, slope * np.array(osm) + intercept, "r-", lw=2)
axes[1].set_title("Q-Q Plot of LPI", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Theoretical Quantiles")
axes[1].set_ylabel("Sample Quantiles")

# LPI vs spatial box (by unique M values as earthquake magnitude proxy)
if "M_3" in df.columns:
    mag_order = sorted(df["M_3"].unique())
    mag_data = [df[df["M_3"] == m][TARGET_COL].values for m in mag_order]
    axes[2].boxplot(mag_data, labels=[str(m) for m in mag_order], patch_artist=True)
    axes[2].set_title(
        "LPI by Earthquake Magnitude (M_3)", fontsize=13, fontweight="bold"
    )
    axes[2].set_xlabel("Magnitude")
    axes[2].set_ylabel("LPI")
else:
    axes[2].axis("off")

plt.tight_layout()
savefig(fig, "01_eda_overview")
print("  Saved: 01_eda_overview.(png|pdf)")

# Correlation heatmap of features
corr = df[FEATURE_COLS + [TARGET_COL]].corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(
    corr,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    annot=False,
    ax=ax,
    linewidths=0.3,
    linecolor="white",
)
ax.set_title("Feature Correlation Matrix (incl. LPI)", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig(fig, "02_correlation_heatmap")
print("  Saved: 02_correlation_heatmap.(png|pdf)")

# ============================================================
# SECTION 2: PREPROCESSING
# ============================================================
print("\n[2] Preprocessing...")
X = df[FEATURE_COLS].values
y = df[TARGET_COL].values
lat = df["Latitude"].values
lon = df["Longitude"].values
groups_raw = df["BoreholeID"].values
groups, _ = pd.factorize(groups_raw, sort=False)

print(f"  X shape: {X.shape}, y shape: {y.shape}")
print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")

# Feature variance analysis before scaling
variances = pd.Series(X.var(axis=0), index=FEATURE_COLS).sort_values(ascending=False)
print("\n  Feature variances (top 10):")
print(variances.head(10).to_string())

# ============================================================
# SECTION 3: PCA ANALYSIS
# ============================================================
print("\n[3] PCA Analysis (on standardized features)...")
PCA_VARIANCE_THRESHOLD = 0.95
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(X)

pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

# Number of components for 95% variance
n_comp_95 = np.argmax(cumulative >= 0.95) + 1
n_comp_90 = np.argmax(cumulative >= 0.90) + 1
n_comp_85 = np.argmax(cumulative >= 0.85) + 1
print(f"  Components for 85% variance: {n_comp_85}")
print(f"  Components for 90% variance: {n_comp_90}")
print(f"  Components for 95% variance: {n_comp_95}")

print("  Note: Model evaluation fits PCA inside each CV fold to avoid leakage.")

# Fit threshold-based PCA on full data for reporting/interpretability figures only.
pca_reporting = PCA(
    n_components=PCA_VARIANCE_THRESHOLD, svd_solver="full", random_state=42
)
X_pca_reporting = pca_reporting.fit_transform(X_scaled)
N_PCA_REPORTING = int(pca_reporting.n_components_)
print(
    f"  PCA(variance>={PCA_VARIANCE_THRESHOLD:.2f}) selects {N_PCA_REPORTING} components on full data"
)

# PCA scree plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(range(1, len(explained) + 1), explained * 100, color="steelblue", alpha=0.7)
axes[0].plot(range(1, len(explained) + 1), explained * 100, "ro-", markersize=4)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance (%)")
axes[0].set_title("Scree Plot", fontsize=13, fontweight="bold")
axes[0].set_xlim(0.5, min(25, len(explained)) + 0.5)

axes[1].plot(range(1, len(cumulative) + 1), cumulative * 100, "b-o", markersize=4)
axes[1].axhline(85, color="orange", linestyle="--", label="85%")
axes[1].axhline(90, color="red", linestyle="--", label="90%")
axes[1].axhline(95, color="green", linestyle="--", label="95%")
axes[1].axvline(n_comp_95, color="green", linestyle=":", alpha=0.7)
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance (%)")
axes[1].set_title("Cumulative Explained Variance", fontsize=13, fontweight="bold")
axes[1].legend()
axes[1].set_xlim(0.5, min(25, len(cumulative)) + 0.5)

plt.tight_layout()
savefig(fig, "03_pca_variance")
print("  Saved: 03_pca_variance.(png|pdf)")

# PCA loadings heatmap (reporting PCA fit on full data)
pca_model = pca_reporting
loadings = pd.DataFrame(
    pca_model.components_.T,
    index=FEATURE_COLS,
    columns=[f"PC{i+1}" for i in range(N_PCA_REPORTING)],
)

fig, ax = plt.subplots(figsize=(max(10, N_PCA_REPORTING), 8))
_annot_kws = (
    {"annot": True, "fmt": ".2f"} if N_PCA_REPORTING <= 10 else {"annot": False}
)
sns.heatmap(
    loadings,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax,
    linewidths=0.3,
    **_annot_kws,
)
ax.set_title(
    f"PCA Loadings (First {N_PCA_REPORTING} Components)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
savefig(fig, "04_pca_loadings")
print("  Saved: 04_pca_loadings.(png|pdf)")

# ============================================================
# SECTION 4: K-FOLD CROSS-VALIDATION SETUP
# ============================================================
print("\n[4] Setting up GroupKFold cross-validation...")
N_FOLDS = 10
cv_splitter = GroupKFold(n_splits=N_FOLDS)
print(f"  CV = GroupKFold, K = {N_FOLDS} folds (grouped by BoreholeID)")


# Define pipelines: StandardScaler → PCA → Model
def make_pipeline(model):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=PCA_VARIANCE_THRESHOLD,
                    svd_solver="full",
                    random_state=42,
                ),
            ),
            ("model", model),
        ]
    )


# ============================================================
# SECTION 5: MODELS
# ============================================================
print("\n[5] Defining regression models...")

MODELS = {
    "Linear Regression": make_pipeline(LinearRegression()),
    "Ridge (a=1.0)": make_pipeline(Ridge(alpha=1.0, random_state=42)),
    "Lasso (a=0.1)": make_pipeline(Lasso(alpha=0.1, random_state=42, max_iter=5000)),
    "ElasticNet": make_pipeline(
        ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000)
    ),
    "Random Forest": make_pipeline(
        RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )
    ),
    "Gradient Boosting": make_pipeline(
        GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        )
    ),
    "SVR (RBF)": make_pipeline(SVR(kernel="rbf", C=10.0, epsilon=0.5, gamma="scale")),
}

if HAS_XGB:
    MODELS["XGBoost"] = make_pipeline(
        XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    )

print(f"  Models defined: {list(MODELS.keys())}")

# ============================================================
# SECTION 6: K-FOLD TRAINING & EVALUATION
# ============================================================
print("\n[6] Running GroupKFold cross-validation...")
SCORING = {
    "r2": "r2",
    "neg_mse": "neg_mean_squared_error",
    "neg_mae": "neg_mean_absolute_error",
}

cv_results = {}
fold_predictions = {}  # per-fold predictions for residual analysis

for name, pipe in MODELS.items():
    print(f"  Training {name}...", end=" ", flush=True)

    # cross_validate for multiple metrics
    cv = cross_validate(
        pipe,
        X,
        y,
        groups=groups,
        cv=cv_splitter,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=-1,
    )

    r2_scores = cv["test_r2"]
    mse_scores = -cv["test_neg_mse"]
    mae_scores = -cv["test_neg_mae"]
    rmse_scores = np.sqrt(mse_scores)

    cv_results[name] = {
        "R2_mean": r2_scores.mean(),
        "R2_std": r2_scores.std(),
        "RMSE_mean": rmse_scores.mean(),
        "RMSE_std": rmse_scores.std(),
        "MAE_mean": mae_scores.mean(),
        "MAE_std": mae_scores.std(),
        "MSE_mean": mse_scores.mean(),
        "MSE_std": mse_scores.std(),
        "train_R2_mean": cv["train_r2"].mean(),
        "train_R2_std": cv["train_r2"].std(),
        "r2_scores_all": r2_scores.tolist(),
        "rmse_scores_all": rmse_scores.tolist(),
    }

    print(
        f"R²={r2_scores.mean():.4f} ± {r2_scores.std():.4f}  "
        f"RMSE={rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}"
    )

# Summary table
results_df = pd.DataFrame(
    {
        name: {
            "R² (mean ± std)": f"{v['R2_mean']:.4f} ± {v['R2_std']:.4f}",
            "RMSE (mean ± std)": f"{v['RMSE_mean']:.3f} ± {v['RMSE_std']:.3f}",
            "MAE (mean ± std)": f"{v['MAE_mean']:.3f} ± {v['MAE_std']:.3f}",
            "Train R²": f"{v['train_R2_mean']:.4f} ± {v['train_R2_std']:.4f}",
        }
        for name, v in cv_results.items()
    }
).T

print("\n--- Cross-Validation Results Summary ---")
print(results_df.to_string())
results_df.to_csv(OUT / "cv_results_summary.csv")
print(f"  Saved: cv_results_summary.csv")

# Also save full numeric results
numeric_df = pd.DataFrame(
    {
        name: {
            "R2_mean": v["R2_mean"],
            "R2_std": v["R2_std"],
            "RMSE_mean": v["RMSE_mean"],
            "RMSE_std": v["RMSE_std"],
            "MAE_mean": v["MAE_mean"],
            "MAE_std": v["MAE_std"],
            "MSE_mean": v["MSE_mean"],
            "MSE_std": v["MSE_std"],
            "Train_R2_mean": v["train_R2_mean"],
            "Train_R2_std": v["train_R2_std"],
        }
        for name, v in cv_results.items()
    }
).T
numeric_df.to_csv(OUT / "cv_results_numeric.csv")

# ============================================================
# SECTION 7: VISUALIZATIONS — MODEL COMPARISON
# ============================================================
print("\n[7] Generating comparison plots...")
model_names = list(cv_results.keys())
r2_means = [cv_results[n]["R2_mean"] for n in model_names]
r2_stds = [cv_results[n]["R2_std"] for n in model_names]
rmse_means = [cv_results[n]["RMSE_mean"] for n in model_names]
rmse_stds = [cv_results[n]["RMSE_std"] for n in model_names]
mae_means = [cv_results[n]["MAE_mean"] for n in model_names]
mae_stds = [cv_results[n]["MAE_std"] for n in model_names]

palette = plt.cm.get_cmap("tab10", len(model_names))
colors = [palette(i) for i in range(len(model_names))]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))


def bar_plot(ax, means, stds, ylabel, title, best_high=True):
    bars = ax.bar(
        range(len(model_names)),
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    best_idx = np.argmax(means) if best_high else np.argmin(means)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)


bar_plot(
    axes[0], r2_means, r2_stds, "R² Score", "R² Score (10-Fold CV)", best_high=True
)
bar_plot(axes[1], rmse_means, rmse_stds, "RMSE", "RMSE (10-Fold CV)", best_high=False)
bar_plot(axes[2], mae_means, mae_stds, "MAE", "MAE (10-Fold CV)", best_high=False)

plt.suptitle(
    "Model Comparison — GroupKFold(PCA + Model)",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()
savefig(fig, "05_model_comparison")
print("  Saved: 05_model_comparison.(png|pdf)")

# Box plots of per-fold R² scores
fig, ax = plt.subplots(figsize=(12, 5))
fold_r2_data = [cv_results[n]["r2_scores_all"] for n in model_names]
bp = ax.boxplot(
    fold_r2_data,
    patch_artist=True,
    labels=model_names,
    medianprops=dict(color="red", linewidth=2),
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=9)
ax.set_title(
    "Per-Fold R² Score Distribution (10-Fold CV)", fontsize=13, fontweight="bold"
)
ax.set_ylabel("R² Score")
plt.tight_layout()
savefig(fig, "06_fold_r2_boxplots")
print("  Saved: 06_fold_r2_boxplots.(png|pdf)")

# Train vs Test R² (overfitting check)
train_r2s = [cv_results[n]["train_R2_mean"] for n in model_names]
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(model_names))
w = 0.35
ax.bar(
    x - w / 2,
    train_r2s,
    width=w,
    label="Train R²",
    color="steelblue",
    alpha=0.8,
    edgecolor="k",
)
ax.bar(
    x + w / 2,
    r2_means,
    width=w,
    label="Test R²",
    color="coral",
    alpha=0.8,
    edgecolor="k",
)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=9)
ax.set_title("Train vs Test R² — Overfitting Check", fontsize=13, fontweight="bold")
ax.set_ylabel("R² Score")
ax.legend()
plt.tight_layout()
savefig(fig, "07_train_vs_test_r2")
print("  Saved: 07_train_vs_test_r2.(png|pdf)")

# ============================================================
# SECTION 8: BEST MODEL — RESIDUALS & PREDICTION ANALYSIS
# ============================================================
print("\n[8] Best model analysis...")
best_name = model_names[np.argmax(r2_means)]
print(f"  Best model: {best_name}  (R²={max(r2_means):.4f})")

best_pipe = MODELS[best_name]

# Collect OOF (out-of-fold) predictions
oof_pred = np.zeros(len(y))
oof_true = np.zeros(len(y))
for fold_i, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y, groups=groups)):
    best_pipe.fit(X[train_idx], y[train_idx])
    oof_pred[val_idx] = best_pipe.predict(X[val_idx])
    oof_true[val_idx] = y[val_idx]

residuals = oof_true - oof_pred
final_r2 = r2_score(oof_true, oof_pred)
final_rmse = np.sqrt(mean_squared_error(oof_true, oof_pred))
final_mae = mean_absolute_error(oof_true, oof_pred)
print(f"  OOF metrics: R²={final_r2:.4f}, RMSE={final_rmse:.3f}, MAE={final_mae:.3f}")

# Residual plots
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Predicted vs Actual
axes[0, 0].scatter(oof_true, oof_pred, s=15, alpha=0.4, color="steelblue")
lo, hi = min(oof_true.min(), oof_pred.min()), max(oof_true.max(), oof_pred.max())
axes[0, 0].plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect fit")
axes[0, 0].set_xlabel("Actual LPI")
axes[0, 0].set_ylabel("Predicted LPI")
axes[0, 0].set_title(
    f"Actual vs Predicted — {best_name}\nR²={final_r2:.4f}",
    fontsize=12,
    fontweight="bold",
)
axes[0, 0].legend()

# Residuals vs Predicted
axes[0, 1].scatter(oof_pred, residuals, s=15, alpha=0.4, color="coral")
axes[0, 1].axhline(0, color="k", linestyle="--", lw=1.5)
axes[0, 1].set_xlabel("Predicted LPI")
axes[0, 1].set_ylabel("Residuals (Actual - Predicted)")
axes[0, 1].set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")

# Residual histogram
axes[1, 0].hist(
    residuals, bins=40, color="mediumseagreen", edgecolor="white", linewidth=0.5
)
axes[1, 0].axvline(0, color="red", linestyle="--", lw=1.5)
axes[1, 0].set_xlabel("Residual")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Residual Distribution", fontsize=12, fontweight="bold")

# Q-Q of residuals
(osm2, osr2), (slope2, intercept2, r2) = stats.probplot(residuals, dist="norm")
axes[1, 1].scatter(osm2, osr2, s=10, alpha=0.5, color="steelblue")
axes[1, 1].plot(osm2, slope2 * np.array(osm2) + intercept2, "r-", lw=2)
axes[1, 1].set_title("Q-Q Plot of Residuals", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Theoretical Quantiles")
axes[1, 1].set_ylabel("Residual Quantiles")

plt.suptitle(f"Best Model Analysis: {best_name}", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig(fig, "08_best_model_residuals")
print("  Saved: 08_best_model_residuals.(png|pdf)")

# ============================================================
# SECTION 9: PCA COMPONENT IMPORTANCE
# ============================================================
print("\n[9] PCA component analysis for best model...")

# Refit PCA to get loadings
scaler_final = StandardScaler()
pca_final = PCA(
    n_components=PCA_VARIANCE_THRESHOLD, svd_solver="full", random_state=42
)
X_scaled_final = scaler_final.fit_transform(X)
X_pca = pca_final.fit_transform(X_scaled_final)
N_PCA_FINAL = int(pca_final.n_components_)

# Correlate PCA components with LPI
pc_corr = []
for i in range(N_PCA_FINAL):
    r, p = stats.pearsonr(X_pca[:, i], y)
    pc_corr.append(
        {
            "PC": f"PC{i+1}",
            "Pearson_r": r,
            "p_value": p,
            "Variance_explained": pca_final.explained_variance_ratio_[i] * 100,
        }
    )
pc_corr_df = pd.DataFrame(pc_corr)
print(pc_corr_df.to_string())
pc_corr_df.to_csv(OUT / "pca_component_correlations.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_corr = ["coral" if r < 0 else "steelblue" for r in pc_corr_df["Pearson_r"]]
axes[0].bar(
    pc_corr_df["PC"],
    pc_corr_df["Pearson_r"],
    color=colors_corr,
    edgecolor="k",
    linewidth=0.5,
)
axes[0].axhline(0, color="k", lw=1)
axes[0].set_title("PC Correlation with LPI Target", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Pearson r")
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(
    pc_corr_df["PC"],
    pc_corr_df["Variance_explained"],
    color="mediumseagreen",
    edgecolor="k",
    linewidth=0.5,
)
axes[1].set_title("Variance Explained by Each PC", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Principal Component")
axes[1].set_ylabel("Variance Explained (%)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
savefig(fig, "09_pca_importance")
print("  Saved: 09_pca_importance.(png|pdf)")

# ============================================================
# SECTION 10: ORIGINAL FEATURE IMPORTANCE (via PCA loadings)
# ============================================================
print("\n[10] Original feature importance reconstruction...")
# Absolute contribution of each original feature: sum of |loading * variance_explained|
loadings_arr = pca_final.components_  # shape (N_PCA_FINAL, n_features)
var_exp = pca_final.explained_variance_ratio_  # shape (N_PCA_FINAL,)
feature_importance = np.sum(np.abs(loadings_arr) * var_exp[:, np.newaxis], axis=0)
feat_imp_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": feature_importance})
feat_imp_df = feat_imp_df.sort_values("Importance", ascending=False)
print(feat_imp_df.to_string())
feat_imp_df.to_csv(OUT / "feature_importance_pca.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors_fi = plt.cm.get_cmap("viridis", len(feat_imp_df))
bar_colors = [colors_fi(i) for i in range(len(feat_imp_df))]
ax.barh(
    feat_imp_df["Feature"][::-1],
    feat_imp_df["Importance"][::-1],
    color=bar_colors[::-1],
    edgecolor="k",
    linewidth=0.3,
)
ax.set_xlabel("PCA-Weighted Importance Score")
ax.set_title(
    "Original Feature Importance\n(via PCA Loadings × Variance Explained)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
savefig(fig, "10_feature_importance")
print("  Saved: 10_feature_importance.(png|pdf)")

# ============================================================
# SECTION 11: SPATIAL ANALYSIS — PREDICTED LPI MAP
# ============================================================
print("\n[11] Spatial analysis of OOF predictions...")
spatial_df = pd.DataFrame(
    {
        "BoreholeID": groups_raw,
        "M": df["M_3"].values if "M_3" in df.columns else np.nan,
        "amax": df["amax_3"].values if "amax_3" in df.columns else np.nan,
        "Latitude": lat,
        "Longitude": lon,
        "LPI_actual": oof_true,
        "LPI_predicted": oof_pred,
        "Residual": residuals,
        "AbsResidual": np.abs(residuals),
    }
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sc0 = axes[0].scatter(
    spatial_df["Longitude"],
    spatial_df["Latitude"],
    c=spatial_df["LPI_actual"],
    cmap="RdYlGn_r",
    s=20,
    alpha=0.7,
)
plt.colorbar(sc0, ax=axes[0], label="LPI")
axes[0].set_title("Actual LPI", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

sc1 = axes[1].scatter(
    spatial_df["Longitude"],
    spatial_df["Latitude"],
    c=spatial_df["LPI_predicted"],
    cmap="RdYlGn_r",
    s=20,
    alpha=0.7,
)
plt.colorbar(sc1, ax=axes[1], label="LPI")
axes[1].set_title(f"Predicted LPI ({best_name})", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Longitude")

sc2 = axes[2].scatter(
    spatial_df["Longitude"],
    spatial_df["Latitude"],
    c=spatial_df["Residual"],
    cmap="RdBu",
    s=20,
    alpha=0.7,
)
plt.colorbar(sc2, ax=axes[2], label="Residual")
axes[2].set_title("Residuals (Actual - Predicted)", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Longitude")

plt.tight_layout()
savefig(fig, "11_spatial_lpi_map")
spatial_df.to_csv(OUT / "spatial_predictions.csv", index=False)
print("  Saved: 11_spatial_lpi_map.(png|pdf), spatial_predictions.csv")

# ============================================================
# SECTION 12: STATISTICAL TESTS ON CV RESULTS
# ============================================================
print("\n[12] Statistical significance testing...")
# Friedman test across models on per-fold R² scores
all_fold_r2 = [cv_results[n]["r2_scores_all"] for n in model_names]
if len(model_names) >= 3:
    stat, pval = stats.friedmanchisquare(*all_fold_r2)
    print(f"  Friedman test: chi2={stat:.4f}, p={pval:.6f}")
    friedman_sig = pval < 0.05
    print(f"  Significant difference between models: {friedman_sig}")
else:
    stat, pval = None, None
    print("  Need >=3 models for Friedman test")

# ============================================================
# SECTION 13: SAVE FINAL RESULTS JSON
# ============================================================
print("\n[13] Saving results JSON...")
results_json = {
    "dataset": {
        "path": str(CSV_PATH),
        "n_samples": len(df),
        "n_features": len(FEATURE_COLS),
        "n_groups": int(pd.Series(groups).nunique()),
        "group_col": "BoreholeID",
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "target_stats": {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        },
    },
    "pca": {
        "variance_threshold": float(PCA_VARIANCE_THRESHOLD),
        "n_comp_85": int(n_comp_85),
        "n_comp_90": int(n_comp_90),
        "n_comp_95": int(n_comp_95),
        "n_components_full_data_at_threshold": int(N_PCA_REPORTING),
        "explained_variance_per_pc": [float(v) for v in explained[:N_PCA_REPORTING]],
        "cumulative_at_n": float(cumulative[N_PCA_REPORTING - 1]),
    },
    "cv": {
        "type": "GroupKFold",
        "n_splits": int(N_FOLDS),
        "group_col": "BoreholeID",
    },
    "model_results": {
        name: {
            "R2_mean": float(v["R2_mean"]),
            "R2_std": float(v["R2_std"]),
            "RMSE_mean": float(v["RMSE_mean"]),
            "RMSE_std": float(v["RMSE_std"]),
            "MAE_mean": float(v["MAE_mean"]),
            "MAE_std": float(v["MAE_std"]),
            "MSE_mean": float(v["MSE_mean"]),
            "MSE_std": float(v["MSE_std"]),
            "Train_R2_mean": float(v["train_R2_mean"]),
            "Train_R2_std": float(v["train_R2_std"]),
            "r2_scores_all": [float(x) for x in v["r2_scores_all"]],
            "rmse_scores_all": [float(x) for x in v["rmse_scores_all"]],
        }
        for name, v in cv_results.items()
    },
    "best_model": {
        "name": best_name,
        "OOF_R2": float(final_r2),
        "OOF_RMSE": float(final_rmse),
        "OOF_MAE": float(final_mae),
    },
    "friedman_test": {
        "statistic": float(stat) if stat else None,
        "p_value": float(pval) if pval else None,
        "significant": bool(friedman_sig) if pval else None,
    },
}

with open(OUT / "results.json", "w") as f:
    json.dump(results_json, f, indent=2)
print("  Saved: results.json")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print(f"  Best model : {best_name}")
print(f"  All results: {OUT}")
print("=" * 60)
