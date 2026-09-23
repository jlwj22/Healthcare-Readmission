"""
Evaluation helpers that go beyond a single AUC number.

- **Decision curve analysis** (Vickers & Elkin, 2006): net benefit of acting
  on a model at each threshold probability, compared against the two
  default policies of contacting everyone or no one. It's the standard way
  clinical prediction papers show whether a model is worth using, not just
  whether it discriminates.
- **Patient-clustered bootstrap confidence intervals.** The test set still
  has repeat encounters from the same patient, so resampling encounters
  independently would understate the uncertainty. Patients are resampled
  instead, taking all of their encounters with them.
- **Calibration summary**: Brier score, observed/expected ratio, and
  calibration slope.
- **Subgroup report**: discrimination, calibration, and outreach flag rates
  broken out by a grouping column (race, age band, sex).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# --- Decision curve analysis -----------------------------------------------

def net_benefit(y: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    """Net benefit of flagging everyone with predicted risk >= threshold.

    NB = TP/n - FP/n * (pt / (1 - pt)). The odds pt/(1-pt) is how many false
    positives a decision maker is willing to accept per true positive, which
    is exactly what choosing a threshold implies.
    """
    y = np.asarray(y)
    flagged = np.asarray(proba) >= threshold
    n = len(y)
    tp = (flagged & (y == 1)).sum()
    fp = (flagged & (y == 0)).sum()
    return tp / n - fp / n * (threshold / (1 - threshold))


def decision_curve(
    y: np.ndarray,
    proba_dict: dict[str, np.ndarray],
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Net benefit per model across thresholds, plus treat-all / treat-none."""
    y = np.asarray(y)
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.51, 0.01), 2)
    prevalence = y.mean()

    rows = []
    for t in thresholds:
        row = {
            "threshold": float(t),
            "treat_all": prevalence - (1 - prevalence) * (t / (1 - t)),
            "treat_none": 0.0,
        }
        for name, proba in proba_dict.items():
            row[name] = net_benefit(y, proba, t)
        rows.append(row)
    return pd.DataFrame(rows)


# --- Metrics ---------------------------------------------------------------

def top_k_mask(proba: np.ndarray, pct: float) -> np.ndarray:
    """Boolean mask for exactly the riskiest `pct` of rows.

    Rank-based rather than a score cutoff, so a coarse score like LACE (lots
    of tied values) still flags exactly `pct` of discharges instead of
    everyone tied at the cutoff.
    """
    proba = np.asarray(proba)
    order = np.argsort(-proba, kind="stable")
    mask = np.zeros(len(proba), dtype=bool)
    mask[order[: int(len(proba) * pct)]] = True
    return mask


def capture_rate(y: np.ndarray, proba: np.ndarray, pct: float) -> float:
    """Share of all positives found in the top `pct` of predicted risk."""
    y = np.asarray(y)
    return float(y[top_k_mask(proba, pct)].sum() / y.sum())


def calibration_slope(y: np.ndarray, proba: np.ndarray) -> float:
    """Slope from regressing the outcome on logit(predicted risk).

    1.0 is ideal. Below 1 means predictions are too extreme (overfit),
    above 1 means they're too timid.
    """
    p = np.clip(np.asarray(proba), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit, y)
    return float(lr.coef_[0, 0])


def calibration_summary(y: np.ndarray, proba: np.ndarray) -> dict:
    y = np.asarray(y)
    proba = np.asarray(proba)
    return {
        "brier_score": round(float(brier_score_loss(y, proba)), 4),
        "mean_predicted_risk": round(float(proba.mean()), 4),
        "observed_rate": round(float(y.mean()), 4),
        "observed_to_expected": round(float(y.mean() / proba.mean()), 3),
        "calibration_slope": round(calibration_slope(y, proba), 3),
    }


DEFAULT_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "brier_score": brier_score_loss,
    "top_decile_capture_rate": lambda y, p: capture_rate(y, p, 0.10),
    "top_quintile_capture_rate": lambda y, p: capture_rate(y, p, 0.20),
}


# --- Patient-clustered bootstrap -------------------------------------------

def _cluster_indices(groups: np.ndarray) -> list[np.ndarray]:
    codes, _ = pd.factorize(pd.Series(groups))
    order = np.argsort(codes, kind="stable")
    boundaries = np.flatnonzero(np.diff(codes[order])) + 1
    return np.split(order, boundaries)


def bootstrap_metrics(
    y: np.ndarray,
    proba_dict: dict[str, np.ndarray],
    groups: np.ndarray,
    metrics: dict[str, Callable] | None = None,
    differences: list[tuple[str, str]] | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Point estimates and 95% percentile CIs, resampling patients.

    `differences` lists (model_a, model_b) pairs to also report
    metric(a) - metric(b) for, computed on the same resample each time so
    the CI accounts for the two models being scored on the same patients.
    """
    y = np.asarray(y)
    metrics = metrics or DEFAULT_METRICS
    differences = differences or []
    clusters = _cluster_indices(np.asarray(groups))
    rng = np.random.default_rng(seed)

    samples = {name: {m: [] for m in metrics} for name in proba_dict}
    diff_samples = {f"{a}_minus_{b}": {m: [] for m in metrics} for a, b in differences}

    for _ in range(n_boot):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([clusters[i] for i in picked])
        yb = y[idx]
        scores = {
            name: {m: fn(yb, proba[idx]) for m, fn in metrics.items()}
            for name, proba in proba_dict.items()
        }
        for name in proba_dict:
            for m in metrics:
                samples[name][m].append(scores[name][m])
        for a, b in differences:
            for m in metrics:
                diff_samples[f"{a}_minus_{b}"][m].append(scores[a][m] - scores[b][m])

    def summarize(point, draws):
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return {"estimate": round(float(point), 5), "ci_low": round(float(lo), 5), "ci_high": round(float(hi), 5)}

    out = {}
    for name, proba in proba_dict.items():
        out[name] = {
            m: summarize(fn(y, proba), samples[name][m]) for m, fn in metrics.items()
        }
    for a, b in differences:
        key = f"{a}_minus_{b}"
        out[key] = {
            m: summarize(fn(y, proba_dict[a]) - fn(y, proba_dict[b]), diff_samples[key][m])
            for m, fn in metrics.items()
        }
    return out


# --- Subgroups -------------------------------------------------------------

def subgroup_report(
    y: np.ndarray,
    proba: np.ndarray,
    group: pd.Series,
    flag_threshold: float,
    min_size: int = 200,
) -> dict:
    """Per-group discrimination, calibration, and outreach flag rates.

    `flag_threshold` should be one fixed score cutoff for everyone (e.g. the
    top-20% cutoff on the whole test set), so the question is whether the
    same policy treats groups differently, not whether each group gets its
    own quota.
    """
    # .values rather than np.asarray keeps an ordered categorical (age bands)
    # in its own order instead of sorting labels alphabetically.
    frame = pd.DataFrame({
        "y": np.asarray(y),
        "proba": np.asarray(proba),
        "group": group.values if isinstance(group, pd.Series) else group,
    })
    report = {}
    for name, sub in frame.groupby("group", observed=True):
        if len(sub) < min_size:
            continue
        flagged = sub["proba"] >= flag_threshold
        positives = sub["y"] == 1
        report[str(name)] = {
            "n": int(len(sub)),
            "observed_rate": round(float(sub["y"].mean()), 4),
            "mean_predicted_risk": round(float(sub["proba"].mean()), 4),
            "observed_to_expected": round(float(sub["y"].mean() / sub["proba"].mean()), 3),
            "roc_auc": round(float(roc_auc_score(sub["y"], sub["proba"])), 4) if sub["y"].nunique() > 1 else None,
            "flagged_rate": round(float(flagged.mean()), 4),
            "recall_when_flagged": round(float(flagged[positives].mean()), 4) if positives.any() else None,
            "false_positive_rate": round(float(flagged[~positives].mean()), 4),
        }
    return report


def age_band(age_numeric: pd.Series) -> pd.Series:
    return pd.cut(
        age_numeric, [0, 50, 60, 70, 80, 100],
        labels=["Under 50", "50-59", "60-69", "70-79", "80+"], right=False,
    )
