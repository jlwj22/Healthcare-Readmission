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

st.set_page_config(page_title="Readmission Risk & Care Targeting", page_icon="\U0001F3E5", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load("models/xgb.joblib")
    return bundle["model"], bundle["features"], bundle["cat_features"]


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


model, features, cat_features = load_model()
explainer = load_explainer(model)

st.title("\U0001F3E5 Readmission Risk & Care Management Targeting")
st.caption(
    "Trained on the UCI **Diabetes 130-US Hospitals (1999-2008)** dataset (~99k inpatient "
    "encounters) with an XGBoost classifier. Enter a discharge profile to estimate 30-day "
    "readmission risk and see what's driving it -- built the way a health plan's "
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

# Default every categorical (medication) feature to "No" -- the overwhelming
# majority class for the ~19 medication columns not exposed as sliders --
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
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "change": change,
    "diabetesMed": diabetes_med,
    "insulin": insulin,
})

X = pd.DataFrame([row])[features]
for c in cat_features:
    X[c] = X[c].astype("category")

risk = float(model.predict_proba(X)[0, 1])

if risk < 0.08:
    tier, color = "Low", "#1a9850"
elif risk < 0.15:
    tier, color = "Moderate", "#fee08b"
elif risk < 0.25:
    tier, color = "High", "#fc8d59"
else:
    tier, color = "Very High", "#d73027"

col1, col2, col3 = st.columns(3)
col1.metric("Predicted 30-day readmission risk", f"{risk:.1%}")
col2.metric("Risk tier", tier)
col3.metric("Member population base rate", "11.4%", help="Overall 30-day readmission rate across the cleaned dataset.")

st.markdown(
    f"<div style='padding:0.75rem;border-radius:8px;background:{color}22;border:1px solid {color};'>"
    f"<b>Care management framing:</b> this member falls in the <b>{tier}</b> risk tier. "
    "At AHRQ's cited average of $17,700 per adult 30-day readmission, outreach capacity is "
    "worth spending on this tier first -- a post-discharge call, medication reconciliation, or "
    "early follow-up visit is a standard, low-cost intervention a payer care-management program "
    "uses this kind of score to prioritize.</div>",
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
st.dataframe(contribs[["feature", "shap_value", "direction"]], use_container_width=True, hide_index=True)

with st.expander("About this model"):
    st.markdown(
        """
        - **Data:** [Diabetes 130-US Hospitals, 1999-2008](https://doi.org/10.24432/C5230J) (UCI ML Repository), ~99,340 encounters after excluding deaths/hospice discharges.
        - **Model:** XGBoost, native categorical handling, patient-level (not row-level) train/test split to prevent leakage across a patient's multiple encounters.
        - **Test-set performance:** ROC-AUC ≈ 0.68 (see `models/metrics.json`) -- modest discrimination, which is the honest result for this task: 30-day readmission is a genuinely hard prediction problem, and this is in line with published results on this dataset. The model still captures ~40% of actual readmissions in the top-risk 20% of discharges, which is the number that matters for a targeting use case.
        - **In payer terms:** at AHRQ's cited $17,700/readmission average, the top-risk 20% of discharges concentrates roughly $19.9M of the $49.4M in readmission cost exposure present in this test set alone -- see the README for the full breakdown.
        - **Fairness note:** race is excluded from the model's features by design; error rates were checked across race subgroups post-hoc (`models/metrics.json`).
        - This is a portfolio/demo project on public research data, not a validated clinical or actuarial tool.
        """
    )
