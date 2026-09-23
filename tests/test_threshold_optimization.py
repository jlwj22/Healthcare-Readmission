import numpy as np

from src.threshold_optimization import compute_cost_curve, find_optimal_threshold


def test_net_savings_zero_at_max_threshold():
    # At threshold 1.0 nothing gets flagged, so the program does nothing:
    # program cost equals the baseline cost of every actual readmission.
    y = np.array([1, 0, 1, 0, 0])
    proba = np.array([0.9, 0.2, 0.6, 0.1, 0.4])
    curve = compute_cost_curve(y, proba, thresholds=np.array([0.99]))
    assert curve.iloc[0]["net_savings"] == 0.0
    assert curve.iloc[0]["flagged_pct"] == 0.0


def test_flagging_a_true_positive_saves_money_when_effective():
    # One real readmission, model flags it above threshold: outreach costs
    # $150, and effectiveness > 0 means some of the $17,700 is avoided, so
    # net savings should be positive and less than the full readmission cost.
    y = np.array([1, 0, 0])
    proba = np.array([0.8, 0.1, 0.1])
    curve = compute_cost_curve(
        y, proba, cost_per_outreach=150, cost_per_readmission=17_700,
        effectiveness=0.15, thresholds=np.array([0.5]),
    )
    row = curve.iloc[0]
    assert row["true_positive"] == 1
    assert row["false_positive"] == 0
    expected_savings = 17_700 * 0.15 - 150
    assert row["net_savings"] == round(expected_savings, 2)


def test_no_effectiveness_means_outreach_is_pure_cost():
    # If outreach has zero assumed effectiveness, flagging a true positive
    # only adds outreach cost with no offsetting benefit, so net savings
    # should be negative by exactly the outreach cost.
    y = np.array([1, 0])
    proba = np.array([0.9, 0.1])
    curve = compute_cost_curve(
        y, proba, cost_per_outreach=150, cost_per_readmission=17_700,
        effectiveness=0.0, thresholds=np.array([0.5]),
    )
    assert curve.iloc[0]["net_savings"] == -150.0


def test_find_optimal_threshold_picks_max_net_savings():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    proba = np.array([0.95, 0.1, 0.85, 0.2, 0.15, 0.9, 0.05, 0.3, 0.25, 0.4])
    curve = compute_cost_curve(y, proba)
    best = find_optimal_threshold(curve)
    assert best["net_savings"] == curve["net_savings"].max()
