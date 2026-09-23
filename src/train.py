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
   `_plot_calibration` checks that directly. That check is also why
   neither model uses class weighting: reweighting the positive class
   inflated every predicted probability (mean predicted risk ~0.43 against
   an actual rate of ~0.11) while barely changing the ranking, so it was
   dropped.
4. **The ML models have to beat the clinical baseline.** Hospitals already
   flag readmission risk with the LACE index (src/lace.py), so it's scored
   on the same test set and compared head to head, including a decision
   curve and a bootstrap CI on the AUC difference.
5. **Uncertainty is reported, not just point estimates.** Headline metrics
   come with 95% CIs from a bootstrap that resamples patients rather than
   encounters (src/evaluation.py).

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
from src.evaluation import (
    age_band,
    bootstrap_metrics,
    calibration_summary,
    capture_rate,
    decision_curve,
    subgroup_report,
    top_k_mask,
)
from src.lace import lace_components
from src.threshold_optimization import (
    DEFAULT_COST_PER_OUTREACH,
    DEFAULT_COST_PER_READMISSION,
    DEFAULT_EFFECTIVENESS,
    compute_cost_curve,
    find_optimal_threshold,
)

MODELS_DIR = "models"
FIG_DIR = "reports/figures"
RANDOM_STATE = 42

# Race is intentionally excluded from the feature set, see module docstring.
# It's still used later for a fairness/error-rate check.
FAIRNESS_COL = "race"

N_BOOTSTRAP = 1000

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "lace": "LACE index",
}
# Same color per model in every figure.
MODEL_COLORS = {
    "XGBoost": "#4C72B0",
    "Logistic Regression": "#DD8452",
    "LACE index": "#55A868",
}


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
    test_df = df.iloc[test_idx]

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
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    logreg_pipe.fit(X_train, y_train)
    logreg_proba = logreg_pipe.predict_proba(X_test)[:, 1]

    # --- XGBoost with native categorical support --------------------------
    xgb = _make_xgb()
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    # --- LACE index, the clinical baseline --------------------------------
    # LACE is a 0-19 point score, not a probability. To put it on the same
    # footing for calibration and decision curves, map score -> risk with a
    # one-feature logistic regression fit on the training set only.
    lace = lace_components(df)
    lace_to_risk = LogisticRegression(penalty=None, max_iter=1000)
    lace_to_risk.fit(lace[["lace_score"]].iloc[train_idx], y_train)
    lace_proba = lace_to_risk.predict_proba(lace[["lace_score"]].iloc[test_idx])[:, 1]

    probas = {"logistic_regression": logreg_proba, "xgboost": xgb_proba, "lace": lace_proba}
    y_arr = y_test.values

    # --- Metrics -----------------------------------------------------
    metrics = {
        "test_set": {
            "encounters": int(len(y_test)),
            "patients": int(test_df["patient_nbr"].nunique()),
            "readmissions_30d": int(y_test.sum()),
            "readmission_rate": round(float(y_test.mean()), 4),
        }
    }

    # Care-management teams can't call every discharged patient. The
    # operating point used throughout is "flag the riskiest 20%", so the
    # confusion matrix and subgroup checks below are all at that cutoff
    # rather than an arbitrary 0.5.
    top20_cutoff = float(np.quantile(xgb_proba, 0.80))
    for name, proba in probas.items():
        flagged = top_k_mask(proba, 0.20)
        tn, fp, fn, tp = confusion_matrix(y_arr, flagged.astype(int)).ravel()
        metrics[name] = {
            "roc_auc": round(roc_auc_score(y_arr, proba), 4),
            "pr_auc": round(average_precision_score(y_arr, proba), 4),
            "top_decile_capture_rate": round(capture_rate(y_arr, proba, 0.10), 4),
            "top_quintile_capture_rate": round(capture_rate(y_arr, proba, 0.20), 4),
            "calibration": calibration_summary(y_arr, proba),
            "confusion_matrix_top_20pct": {
                "true_negative": int(tn), "false_positive": int(fp),
                "false_negative": int(fn), "true_positive": int(tp),
            },
        }
    metrics["xgboost"]["top_20pct_risk_cutoff"] = round(top20_cutoff, 4)
    metrics["lace"]["raw_score_roc_auc"] = round(
        roc_auc_score(y_arr, lace["lace_score"].iloc[test_idx]), 4
    )
    metrics["lace"]["mean_score"] = round(float(lace["lace_score"].iloc[test_idx].mean()), 2)

    # --- 95% CIs, resampling patients rather than encounters -------------
    metrics["confidence_intervals"] = bootstrap_metrics(
        y_arr, probas, test_df["patient_nbr"].values,
        differences=[("xgboost", "lace"), ("xgboost", "logistic_regression")],
        n_boot=N_BOOTSTRAP, seed=RANDOM_STATE,
    )
    metrics["confidence_intervals"]["method"] = (
        f"{N_BOOTSTRAP} bootstrap resamples of test-set patients (all of a "
        "patient's encounters move together), 2.5th/97.5th percentiles."
    )

    # --- Decision curve analysis -----------------------------------------
    dca = decision_curve(y_arr, probas)
    breakeven = DEFAULT_COST_PER_OUTREACH / (DEFAULT_COST_PER_READMISSION * DEFAULT_EFFECTIVENESS)
    metrics["decision_curve"] = {
        "net_benefit_at_selected_thresholds": {
            f"{t:.2f}": {
                col: round(float(dca.loc[np.isclose(dca["threshold"], t), col].iloc[0]), 5)
                for col in ["xgboost", "logistic_regression", "lace", "treat_all"]
            }
            for t in [0.06, 0.10, 0.15, 0.20, 0.30]
        },
        "cost_breakeven_threshold": round(breakeven, 4),
        "note": "Net benefit = TP/n - FP/n * pt/(1-pt), in units of true "
        "positives per discharge. The cost break-even threshold is "
        "outreach cost / (readmission cost * assumed effectiveness) under "
        "the base-case assumptions in src/threshold_optimization.py.",
    }

    # --- Subgroup checks: race, age, sex -----------------------------------
    # Race isn't a model feature (module docstring), which is exactly why it
    # needs checking here: leaving it out doesn't guarantee the model treats
    # groups the same. All groups get the same top-20% score cutoff.
    subgroup_cols = {
        "race": test_df["race"],
        "age_band": age_band(test_df["age_numeric"]),
        "gender": test_df["gender"],
    }
    metrics["subgroup_performance"] = {
        col: subgroup_report(y_arr, xgb_proba, values, flag_threshold=top20_cutoff)
        for col, values in subgroup_cols.items()
    }

    # --- Why the patient-level split matters, shown rather than asserted --
    # Fit the identical XGBoost architecture on a naive row-level split that
    # ignores patient_nbr entirely. Because ~30% of patients here have more
    # than one encounter, a naive split lets some of a patient's encounters
    # land in train while others from the same patient land in test, so the
    # model can partly learn "this patient's baseline risk" from train and
    # then get credit for "predicting" it in test. That is leakage, and the
    # gap between these numbers and the patient-level numbers above is what
    # it's worth in this dataset.
    metrics["naive_split_comparison"] = _naive_split_comparison(df, model_features)

    # --- Cost-sensitive threshold optimization ----------------------------
    # Persist the held-out predictions so the Streamlit app's ROI simulator
    # can recompute this live without needing the raw CSV.
    pd.DataFrame({"y": y_arr, "proba": xgb_proba}).to_parquet(
        os.path.join(MODELS_DIR, "test_predictions.parquet")
    )
    metrics["threshold_optimization"] = _threshold_optimization_summary(y_arr, xgb_proba)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

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

    _plot_roc_pr(y_test, {MODEL_LABELS[k]: v for k, v in probas.items()})
    _plot_readmission_by_prior_visits(df)
    _plot_calibration(y_arr, {MODEL_LABELS[k]: v for k, v in probas.items()})
    _plot_decision_curve(dca, breakeven)
    _plot_shap(xgb, X_test, cat_features)
    _plot_cost_threshold_curve(y_arr, xgb_proba)

    print(f"\nSaved models to {MODELS_DIR}/, figures to {FIG_DIR}/")


def _make_xgb() -> XGBClassifier:
    # No scale_pos_weight: see point 3 of the module docstring.
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        enable_categorical=True,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _naive_split_comparison(df, model_features):
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

    xgb_naive = _make_xgb()
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


def _plot_calibration(y_test, proba_dict):
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, proba in proba_dict.items():
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name, color=MODEL_COLORS[name])
    upper = 0.35
    ax.plot([0, upper], [0, upper], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Mean predicted risk (decile bin)")
    ax.set_ylabel("Observed readmission rate (bin)")
    ax.set_title("Calibration Curve")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "calibration.png"), dpi=150)
    plt.close(fig)


def _plot_decision_curve(dca, breakeven):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key in ["xgboost", "logistic_regression", "lace"]:
        ax.plot(dca["threshold"], dca[key], label=MODEL_LABELS[key],
                color=MODEL_COLORS[MODEL_LABELS[key]])
    ax.plot(dca["threshold"], dca["treat_all"], color="gray", lw=1, label="Contact everyone")
    ax.axhline(0, color="black", lw=1, label="Contact no one")
    ax.axvline(breakeven, color="#d73027", linestyle="--", alpha=0.7,
               label=f"Cost break-even ({breakeven:.2f})")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(-0.01, dca["treat_all"].max() + 0.02)
    ax.set_xlabel("Threshold probability for outreach")
    ax.set_ylabel("Net benefit (readmissions caught per discharge)")
    ax.set_title("Decision Curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "decision_curve.png"), dpi=150)
    plt.close(fig)


def _plot_roc_pr(y_test, proba_dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        color = MODEL_COLORS[name]
        axes[0].plot(fpr, tpr, color=color, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(rec, prec, color=color, label=f"{name} (AP={average_precision_score(y_test, proba):.3f})")
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


def _threshold_optimization_summary(y_test, xgb_proba):
    """Base-case optimal threshold plus a small sensitivity table across
    assumed outreach effectiveness. See src/threshold_optimization.py for
    why these are scenario inputs rather than validated figures.
    """
    base_curve = compute_cost_curve(y_test, xgb_proba)
    best = find_optimal_threshold(base_curve)

    sensitivity = {}
    for effectiveness in [0.10, 0.15, 0.20]:
        curve = compute_cost_curve(y_test, xgb_proba, effectiveness=effectiveness)
        row = find_optimal_threshold(curve)
        sensitivity[f"{effectiveness:.0%}_effectiveness"] = {
            "optimal_threshold": row["threshold"],
            "flagged_pct": row["flagged_pct"],
            "net_savings": float(row["net_savings"]),
        }

    return {
        "assumptions": {
            "cost_per_outreach": DEFAULT_COST_PER_OUTREACH,
            "cost_per_readmission": DEFAULT_COST_PER_READMISSION,
            "base_case_effectiveness": DEFAULT_EFFECTIVENESS,
            "note": "Scenario inputs, not validated figures for this program. "
            "A real ROI estimate needs a program-effectiveness study.",
        },
        "base_case_optimal_threshold": best["threshold"],
        "base_case_flagged_pct": best["flagged_pct"],
        "base_case_net_savings": float(best["net_savings"]),
        "sensitivity_by_effectiveness": sensitivity,
    }


def _plot_cost_threshold_curve(y_test, xgb_proba):
    curve = compute_cost_curve(y_test, xgb_proba)
    best = find_optimal_threshold(curve)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(curve["threshold"], curve["net_savings"] / 1e6, color=MODEL_COLORS["XGBoost"])
    ax.axvline(best["threshold"], color="#d73027", linestyle="--", alpha=0.7,
               label=f"Optimal threshold ({best['threshold']:.2f})")
    ax.set_xlabel("Risk threshold for flagging outreach")
    ax.set_ylabel("Net savings vs. no program ($M)")
    ax.set_title("Cost-Sensitive Threshold Optimization")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cost_threshold_curve.png"), dpi=150)
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
