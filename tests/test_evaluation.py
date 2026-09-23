import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    _cluster_indices,
    bootstrap_metrics,
    calibration_summary,
    capture_rate,
    decision_curve,
    net_benefit,
    subgroup_report,
)


def test_net_benefit_of_a_perfect_model_is_prevalence():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    assert net_benefit(y, y.astype(float), 0.2) == pytest.approx(0.2)


def test_net_benefit_penalizes_false_positives_by_threshold_odds():
    # Flag everyone: 1 TP, 3 FP out of 4, at pt=0.25 -> odds 1/3.
    y = np.array([1, 0, 0, 0])
    nb = net_benefit(y, np.ones(4), 0.25)
    assert nb == pytest.approx(1 / 4 - 3 / 4 * (1 / 3))


def test_decision_curve_treat_all_matches_flag_everyone():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    curve = decision_curve(y, {"everyone": np.ones(500)}, thresholds=np.array([0.1, 0.3]))
    assert np.allclose(curve["everyone"], curve["treat_all"])
    assert (curve["treat_none"] == 0).all()


def test_capture_rate():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    proba = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])
    assert capture_rate(y, proba, 0.2) == pytest.approx(2 / 3)


def test_calibration_summary_observed_to_expected():
    y = np.array([1, 0, 0, 0])
    summary = calibration_summary(y, np.full(4, 0.5))
    assert summary["observed_to_expected"] == pytest.approx(0.5)
    assert summary["brier_score"] == pytest.approx(0.25)


def test_cluster_indices_keep_each_patient_together():
    groups = np.array([7, 3, 7, 9, 3, 7])
    clusters = _cluster_indices(groups)
    assert sorted(len(c) for c in clusters) == [1, 2, 3]
    for c in clusters:
        assert len(set(groups[c])) == 1


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    n = 600
    y = rng.integers(0, 2, n)
    good = np.clip(y * 0.6 + rng.normal(0.2, 0.2, n), 0, 1)
    noise = rng.uniform(0, 1, n)
    groups = np.repeat(np.arange(n // 2), 2)
    out = bootstrap_metrics(
        y, {"good": good, "noise": noise}, groups,
        differences=[("good", "noise")], n_boot=100,
    )
    auc = out["good"]["roc_auc"]
    assert auc["ci_low"] <= auc["estimate"] <= auc["ci_high"]
    assert out["good_minus_noise"]["roc_auc"]["ci_low"] > 0


def test_subgroup_report_uses_one_shared_cutoff():
    y = np.array([1, 0, 1, 0] * 100)
    proba = np.array([0.9, 0.1, 0.4, 0.3] * 100)
    group = pd.Series(["a", "a", "b", "b"] * 100)
    report = subgroup_report(y, proba, group, flag_threshold=0.5, min_size=10)
    assert report["a"]["flagged_rate"] == 0.5
    assert report["b"]["flagged_rate"] == 0.0
    assert report["a"]["recall_when_flagged"] == 1.0
