"""
Train and evaluate the 30-day readmission risk models.

A few things this pipeline is deliberate about, because getting them wrong
is the easiest way to produce a healthcare model that looks great and is
useless in production:

1. **Patient-level train/test split.** The same patient can appear in this
   dataset multiple times (69,987 unique patients across 99,340 encounters
   after cleaning). A random row-level split would let a patient's other
   encounters leak into both train and test, inflating apparent
   performance. We split on `patient_nbr` instead (GroupShuffleSplit) so no
   patient's encounters appear in both sets, and `_naive_split_comparison`
   below quantifies exactly what that leakage would have been worth.
2. **Race is excluded from the model features.** It's present in the raw
   data and is genuinely associated with outcomes in this dataset, as it is
   in most US healthcare data, reflecting structural inequities rather than
   biology. Training a clinical risk score on race directly risks baking
   discriminatory proxies into a tool meant to allocate care management
   resources. It's kept in the data for a fairness check (do error rates
   hold up across race groups?) but dropped from `X`.
3. **Calibration gets checked, not assumed.** A ranking metric like AUC
   says nothing about whether a predicted 20% risk actually corresponds to
   a 20% observed readmission rate. Since the payer-economics numbers in
   the README rely on the predicted probabilities meaning what they say,
   `_plot_calibration` checks that directly.

Run with:  python -m src.train
"""
from __future__ import annotations

import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_prep import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    prepare,
)

MODELS_DIR = "models"
FIG_DIR = "reports/figures"
RANDOM_STATE = 42

# Race is intentionally excluded from the feature set, see module docstring.
# It's still used later for a fairness/error-rate check.
FAIRNESS_COL = "race"


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = prepare()

    model_features = [f for f in MODEL_FEATURES if f != FAIRNESS_COL]
    cat_features = [f for f in CATEGORICAL_FEATURES if f != FAIRNESS_COL]

    X = df[model_features]
    y = df[TARGET]
    groups = df["patient_nbr"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    race_test = df["race"].iloc[test_idx]

    assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx])), (
        "patient leakage between train/test split"
    )

    # --- Logistic regression baseline (one-hot + scaling, in a Pipeline so
    # the same transform is trivially reusable at inference time) ---------
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )
    logreg_pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    logreg_pipe.fit(X_train, y_train)
    logreg_proba = logreg_pipe.predict_proba(X_test)[:, 1]

    # --- XGBoost with native categorical support --------------------------
    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()

    pos = y_train.sum()
    neg = len(y_train) - pos
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="auc",
        enable_categorical=True,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train_xgb, y_train)
    xgb_proba = xgb.predict_proba(X_test_xgb)[:, 1]

    # --- Metrics -----------------------------------------------------
    metrics = {}
    for name, proba in [("logistic_regression", logreg_proba), ("xgboost", xgb_proba)]:
        auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        preds_at_50 = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_at_50).ravel()
        metrics[name] = {
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix_at_0.5": {
                "true_negative": int(tn), "false_positive": int(fp),
                "false_negative": int(fn), "true_positive": int(tp),
            },
        }

    # Top-decile capture: care-management teams can't call every discharged
    # patient. What share of actual 30-day readmissions do we catch if we
    # only have capacity to intervene on the riskiest 10-20% of discharges?
    for pct, label in [(0.1, "top_decile_capture_rate"), (0.2, "top_quintile_capture_rate")]:
        order = np.argsort(-xgb_proba)
        n = int(len(order) * pct)
        captured = y_test.values[order[:n]].sum()
        metrics["xgboost"][label] = round(float(captured / y_test.sum()), 4)

    # --- Fairness check: does error rate hold up across race groups? -----
    fairness = {}
    for group, sub in pd.DataFrame({"race": race_test, "y": y_test.values, "proba": xgb_proba}).groupby("race", observed=True):
        if len(sub) < 50:
            continue
        preds = (sub["proba"] >= 0.5).astype(int)
        fairness[group] = {
            "n": int(len(sub)),
            "actual_readmit_rate": round(float(sub["y"].mean()), 4),
            "predicted_positive_rate": round(float(preds.mean()), 4),
            "recall": round(float((preds[sub["y"] == 1] == 1).mean()) if (sub["y"] == 1).any() else float("nan"), 4),
            "roc_auc": round(float(roc_auc_score(sub["y"], sub["proba"])) if sub["y"].nunique() > 1 else float("nan"), 4),
        }
    metrics["fairness_by_race_group"] = fairness

    # --- Why the patient-level split matters, shown rather than asserted --
    # Fit the identical XGBoost architecture on a naive row-level split that
    # ignores patient_nbr entirely. Because ~30% of patients here have more
    # than one encounter, a naive split lets some of a patient's encounters
    # land in train while others from the same patient land in test, so the
    # model can partly learn "this patient's baseline risk" from train and
    # then get credit for "predicting" it in test. That is leakage, and the
    # gap between these numbers and the patient-level numbers above is what
    # it's worth in this dataset.
    metrics["naive_split_comparison"] = _naive_split_comparison(df, model_features, cat_features)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save each categorical column's training-time categories alongside the
    # model. A single-row inference DataFrame (like the one the Streamlit
    # app builds) can't reliably infer a matching category dtype on its own,
    # especially when a field's value is missing: pandas falls back to an
    # empty float64 category index in that case, which XGBoost's native
    # categorical support rejects outright. Rebuilding each column with
    # pd.CategoricalDtype(categories=...) at inference time avoids that.
    cat_categories = {c: X[c].cat.categories.tolist() for c in cat_features}

    joblib.dump({"pipeline": logreg_pipe, "features": model_features},
                os.path.join(MODELS_DIR, "logreg.joblib"))
    joblib.dump(
        {
            "model": xgb,
            "features": model_features,
            "cat_features": cat_features,
            "cat_categories": cat_categories,
        },
        os.path.join(MODELS_DIR, "xgb.joblib"),
    )

    print(json.dumps(metrics, indent=2))

    _plot_roc_pr(y_test, {"Logistic Regression": logreg_proba, "XGBoost": xgb_proba})
    _plot_readmission_by_prior_visits(df)
    _plot_calibration(y_test.values, xgb_proba)
    _plot_shap(xgb, X_test_xgb, cat_features)

    print(f"\nSaved models to {MODELS_DIR}/, figures to {FIG_DIR}/")


def _naive_split_comparison(df, model_features, cat_features):
    """Fit the same XGBoost architecture on a plain stratified row-level
    split (no grouping on patient_nbr) and return its test-set metrics
    alongside the fraction of test patients who also show up in that naive
    split's training set. This is the comparison that makes the leakage the
    patient-level split avoids concrete instead of just asserted.
    """
    X = df[model_features]
    y = df[TARGET]
    groups = df["patient_nbr"]

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    overlap = set(groups_train) & set(groups_test)
    pct_overlap = len(overlap) / groups_test.nunique()

    pos = y_train.sum()
    neg = len(y_train) - pos
    xgb_naive = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="auc",
        enable_categorical=True,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_naive.fit(X_train, y_train)
    proba = xgb_naive.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "pct_test_patients_also_seen_in_train": round(float(pct_overlap), 4),
        "note": "Same model, same features, a random row-level split instead of "
        "GroupShuffleSplit on patient_nbr. Compare against the patient-level "
        "xgboost numbers above to see how much of that performance was "
        "coming from repeat patients leaking across the split.",
    }


def _plot_calibration(y_test, proba):
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(mean_pred, frac_pos, marker="o", label="XGBoost")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted risk (bin)")
    ax.set_ylabel("Observed readmission rate (bin)")
    ax.set_title("Calibration Curve")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "calibration.png"), dpi=150)
    plt.close(fig)


def _plot_roc_pr(y_test, proba_dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, proba):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "roc_pr_curves.png"), dpi=150)
    plt.close(fig)


def _plot_readmission_by_prior_visits(df):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bucket = pd.cut(df["total_prior_visits"], [-1, 0, 1, 2, 3, 5, 100],
                     labels=["0", "1", "2", "3", "4-5", "6+"])
    rate = df.groupby(bucket, observed=True)[TARGET].mean()
    rate.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_xlabel("Prior outpatient + emergency + inpatient visits")
    ax.set_ylabel("30-day readmission rate")
    ax.set_title("Readmission Rate by Prior Utilization")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "readmission_by_prior_visits.png"), dpi=150)
    plt.close(fig)


def _plot_shap(xgb, X_test, cat_features):
    import shap

    sample = X_test.sample(min(2000, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(sample)
    fig = plt.figure(figsize=(7.5, 6))
    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
