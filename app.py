"""
Readmission Risk & Care Management Targeting Dashboard
---------------------------------------------------------
Interactive Streamlit app built the way a health plan's care-management
analytics team would use it: enter a recently discharged member's
encounter profile, get a predicted 30-day readmission risk, a risk tier
for outreach targeting, and the top factors behind that prediction (SHAP).

Run with: streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.data_prep import AGE_MIDPOINTS
from src.threshold_optimization import (
    DEFAULT_COST_PER_OUTREACH,
    DEFAULT_COST_PER_READMISSION,
    DEFAULT_EFFECTIVENESS,
    compute_cost_curve,
    find_optimal_threshold,
)

st.set_page_config(page_title="Readmission Risk & Care Targeting", page_icon="\U0001F3E5", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load("models/xgb.joblib")
    return bundle["model"], bundle["features"], bundle["cat_features"], bundle["cat_categories"]


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


model, features, cat_features, cat_categories = load_model()
explainer = load_explainer(model)

st.title("\U0001F3E5 Readmission Risk & Care Management Targeting")
st.caption(
    "Trained on the UCI **Diabetes 130-US Hospitals (1999-2008)** dataset (~99k inpatient "
    "encounters) with an XGBoost classifier. Enter a discharge profile to estimate 30-day "
    "readmission risk and see what's driving it. Built the way a health plan's "
    "care-management analytics team would use it to prioritize post-discharge outreach, "
    "not for clinical diagnosis."
)

with st.sidebar:
    st.header("Member encounter profile")
    age_label = st.selectbox("Age group", list(AGE_MIDPOINTS.keys()), index=6)
    time_in_hospital = st.slider("Days in hospital (this stay)", 1, 14, 4)
    num_lab_procedures = st.slider("Number of lab procedures", 0, 120, 40)
    num_procedures = st.slider("Number of procedures", 0, 6, 1)
    num_medications = st.slider("Number of medications", 0, 80, 15)
    number_diagnoses = st.slider("Number of diagnoses", 1, 16, 7)
    st.divider()
    st.caption("Prior utilization (strongest signal in the model)")
    number_outpatient = st.slider("Prior outpatient visits", 0, 20, 0)
    number_emergency = st.slider("Prior emergency visits", 0, 20, 0)
    number_inpatient = st.slider("Prior inpatient visits", 0, 20, 0)
    st.divider()
    discharge_disposition_id = st.selectbox(
        "Discharge disposition",
        [("1", "Discharged to home"), ("6", "Home with home health service"),
         ("3", "Transferred to SNF"), ("22", "Transferred to rehab facility")],
        format_func=lambda x: x[1],
    )[0]
    admission_source_id = st.selectbox(
        "Admission source",
        [("7", "Emergency Room"), ("1", "Physician Referral"), ("4", "Transfer from a hospital")],
        format_func=lambda x: x[1],
    )[0]
    admission_type_id = st.selectbox(
        "Admission type", [("1", "Emergency"), ("2", "Urgent"), ("3", "Elective")],
        format_func=lambda x: x[1],
    )[0]
    diag_1_category = st.selectbox(
        "Primary diagnosis category",
        ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Genitourinary",
         "Injury", "Musculoskeletal", "Neoplasms", "Other"],
    )
    change = st.selectbox("Medication changed this visit?", ["No", "Ch"], format_func=lambda x: "Yes" if x == "Ch" else "No")
    diabetes_med = st.selectbox("On a diabetes medication?", ["Yes", "No"])
    insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
    gender = st.selectbox("Gender", ["Female", "Male"])

# Default every categorical (medication) feature to "No", the overwhelming
# majority class for the ~19 medication columns not exposed as sliders,
# and every numeric feature to 0, then override with the fields above.
row = {f: ("No" if f in cat_features else 0) for f in features}
row.update({
    "age_numeric": AGE_MIDPOINTS[age_label],
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "number_diagnoses": number_diagnoses,
    "total_prior_visits": number_outpatient + number_emergency + number_inpatient,
    "num_medication_changes": 1 if insulin in ("Up", "Down") else 0,
    "gender": gender,
    "admission_type_id": admission_type_id,
    "discharge_disposition_id": discharge_disposition_id,
    "admission_source_id": admission_source_id,
    "medical_specialty_grouped": "Missing",
    "diag_1_category": diag_1_category,
    "diag_2_category": "Missing",
    "diag_3_category": "Missing",
    # These two aren't asked about in the sidebar, so default to "not
    # measured", which is what roughly 95% of encounters in the training
    # data actually are for these two labs. That's a real missing value,
    # not the literal string "None": the raw dataset happens to spell "not
    # measured" as the text "None", but pandas' own CSV parsing already
    # reads that as NaN during training, so the categories learned for
    # these two columns don't include a "None" category at all.
    "max_glu_serum": np.nan,
    "A1Cresult": np.nan,
    "change": change,
    "diabetesMed": diabetes_med,
    "insulin": insulin,
})

X = pd.DataFrame([row])[features]
for c in cat_features:
    # Cast against the exact categories the model was trained on rather
    # than letting a single-row DataFrame infer its own category dtype.
    # A one-row column has no way to know what the full category set
    # looks like, and when every value in it happens to be missing,
    # pandas falls back to an empty float64 category index, which
    # XGBoost's categorical support rejects outright.
    X[c] = X[c].astype(pd.CategoricalDtype(categories=cat_categories[c]))

risk = float(model.predict_proba(X)[0, 1])

# Tier cutoffs are on the calibrated probability scale. 15% is roughly where
# the top 20% of test-set discharges starts (models/metrics.json,
# "top_20pct_risk_cutoff"), so High and Very High together are the
# "riskiest fifth" a capacity-limited outreach team would work first.
if risk < 0.08:
    tier, color = "Low", "#1a9850"
    action = "Routine discharge follow-up. Outreach capacity is better spent on higher tiers first."
elif risk < 0.15:
    tier, color = "Moderate", "#fee08b"
    action = ("Worth contacting if the program has capacity beyond the riskiest fifth of "
              "discharges, e.g. an automated check-in rather than a nurse call.")
elif risk < 0.25:
    tier, color = "High", "#fc8d59"
    action = ("In the riskiest fifth of discharges. A post-discharge call, medication "
              "reconciliation, or early follow-up visit is the standard low-cost intervention here.")
else:
    tier, color = "Very High", "#d73027"
    action = ("Top of the outreach list. At AHRQ's cited average of $17,700 per adult 30-day "
              "readmission, this is where limited care-management capacity goes first.")

col1, col2, col3 = st.columns(3)
col1.metric("Predicted 30-day readmission risk", f"{risk:.1%}")
col2.metric("Risk tier", tier)
col3.metric("Member population base rate", "11.4%", help="Overall 30-day readmission rate across the cleaned dataset.")

st.markdown(
    f"<div style='padding:0.75rem;border-radius:8px;background:{color}22;border:1px solid {color};'>"
    f"<b>Care management framing:</b> this member falls in the <b>{tier}</b> risk tier. "
    f"{action}</div>",
    unsafe_allow_html=True,
)

st.subheader("What's driving this prediction?")
shap_values = explainer(X)
contribs = (
    pd.DataFrame({"feature": features, "shap_value": shap_values.values[0]})
    .sort_values("shap_value", key=abs, ascending=False)
    .head(8)
)
contribs["direction"] = np.where(contribs["shap_value"] > 0, "Increases risk", "Decreases risk")
st.bar_chart(contribs.set_index("feature")["shap_value"])
st.dataframe(contribs[["feature", "shap_value", "direction"]], width="stretch", hide_index=True)

st.subheader("Outreach cost-benefit simulator")
st.caption(
    "How many discharges is it worth calling, given what an outreach contact costs "
    "and how effective you assume it is at preventing a readmission? Recomputed live "
    "on the held-out test set (2,789 actual 30-day readmissions among 24,757 discharges). "
    "These cost/effectiveness numbers are scenario inputs, not a validated ROI: a real "
    "estimate needs a program-effectiveness study, not a model."
)

test_predictions = pd.read_parquet("models/test_predictions.parquet")

sim_col1, sim_col2 = st.columns(2)
cost_per_outreach = sim_col1.slider(
    "Cost per outreach contact ($)", 25, 500, int(DEFAULT_COST_PER_OUTREACH), step=25
)
effectiveness_pct = sim_col2.slider(
    "Assumed relative risk reduction from outreach (%)", 0, 40,
    int(DEFAULT_EFFECTIVENESS * 100),
)

curve = compute_cost_curve(
    test_predictions["y"].values,
    test_predictions["proba"].values,
    cost_per_outreach=cost_per_outreach,
    cost_per_readmission=DEFAULT_COST_PER_READMISSION,
    effectiveness=effectiveness_pct / 100,
)
best = find_optimal_threshold(curve)

roi_col1, roi_col2, roi_col3 = st.columns(3)
roi_col1.metric("Optimal risk threshold", f"{best['threshold']:.2f}")
roi_col2.metric("Discharges flagged for outreach", f"{best['flagged_pct']:.1%}")
roi_col3.metric("Net savings vs. no program", f"${best['net_savings']:,.0f}")

st.line_chart(curve.set_index("threshold")["net_savings"])

with st.expander("About this model"):
    st.markdown(
        """
        - **Data:** [Diabetes 130-US Hospitals, 1999-2008](https://doi.org/10.24432/C5230J) (UCI ML Repository), ~99,340 encounters after excluding deaths/hospice discharges.
        - **Model:** XGBoost, native categorical handling, patient-level (not row-level) train/test split to prevent leakage across a patient's multiple encounters.
        - **Test-set performance:** ROC-AUC 0.677 (95% CI 0.665-0.689, see `models/metrics.json`), modest discrimination, which is the honest result for this task. 30-day readmission is a genuinely hard prediction problem, and this is in line with published results on this dataset. The model still captures ~40% of actual readmissions in the top-risk 20% of discharges, which is the number that matters for a targeting use case.
        - **Versus the clinical standard:** the LACE index hospitals already use scores ROC-AUC 0.571 on the same test set and catches 28% of readmissions in its top 20%, so the model adds about 0.11 of AUC and 13 points of capture over it.
        - **Calibration:** mean predicted risk is 11.2% against an observed 11.3%, with a calibration slope of 0.95 (`reports/figures/calibration.png`). That's what makes the percentages above usable as probabilities, and it's why the model is trained without class weighting.
        - **Why the split matters:** the identical model fit on a naive row-level split (allowing the same patient in both train and test) scores about 0.3 points of ROC-AUC higher, a modest but real inflation from patient leakage that the patient-level split in this project avoids. Full comparison in the README.
        - **In payer terms:** at AHRQ's cited $17,700/readmission average, the top-risk 20% of discharges concentrates roughly $19.9M of the $49.4M in readmission cost exposure present in this test set alone. See the README for the full breakdown.
        - **Fairness note:** race is excluded from the model's features by design; error rates were checked across race subgroups post-hoc (`models/metrics.json`).
        - This is a portfolio/demo project on public research data, not a validated clinical or actuarial tool.
        """
    )
