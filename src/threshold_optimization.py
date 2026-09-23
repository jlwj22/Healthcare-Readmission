"""
Cost-sensitive threshold optimization for outreach targeting.

This turns the payer-cost framing already used in the README (AHRQ's
$17,700 average cost of an adult 30-day readmission) into a decision-support
tool: given an assumed cost per outreach contact and an assumed relative
risk reduction from that outreach, what risk threshold maximizes net
savings, and how much does that answer move if those assumptions change?

The effectiveness and outreach-cost numbers are scenario inputs, not
validated figures for this specific program. Estimating a real number would
require a program-effectiveness study (e.g. an RCT or a pre/post analysis
of an actual outreach program), which this project doesn't have. Treat the
output as a sensitivity analysis, not a claimed ROI.

Shared by src/train.py (to generate the README figure/metrics) and app.py
(to power a live "what if" slider), so the cost math lives in one place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COST_PER_OUTREACH = 150.0
DEFAULT_COST_PER_READMISSION = 17_700.0
DEFAULT_EFFECTIVENESS = 0.15


def compute_cost_curve(
    y: np.ndarray,
    proba: np.ndarray,
    cost_per_outreach: float = DEFAULT_COST_PER_OUTREACH,
    cost_per_readmission: float = DEFAULT_COST_PER_READMISSION,
    effectiveness: float = DEFAULT_EFFECTIVENESS,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Net savings vs. a do-nothing baseline, at each candidate threshold.

    Baseline: no program, every actual readmission costs `cost_per_readmission`.
    Program at threshold t: outreach cost on everyone flagged (TP+FP), full
    readmission cost on everyone missed (FN), and reduced readmission cost
    on true positives caught (they still readmit at rate `1 - effectiveness`).
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    baseline_cost = y.sum() * cost_per_readmission

    rows = []
    for t in thresholds:
        flagged = proba >= t
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        tn = int((~flagged & (y == 0)).sum())

        program_cost = (
            (tp + fp) * cost_per_outreach
            + fn * cost_per_readmission
            + tp * (1 - effectiveness) * cost_per_readmission
        )
        net_savings = baseline_cost - program_cost

        rows.append({
            "threshold": round(float(t), 4),
            "flagged_pct": round((tp + fp) / len(y), 4),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "net_savings": round(float(net_savings), 2),
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(curve_df: pd.DataFrame) -> pd.Series:
    """Row of `curve_df` with the highest net savings."""
    return curve_df.loc[curve_df["net_savings"].idxmax()]
