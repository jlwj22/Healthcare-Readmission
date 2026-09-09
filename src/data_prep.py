"""
Data cleaning and feature engineering for the diabetic hospital-readmission
dataset.

Source: Diabetes 130-US hospitals for years 1999-2008 (Clore, Cios,
DeShazo & Strack, 2014; UCI ML Repository #296). 101,766 inpatient
encounters for diabetic patients across 130 US hospitals.

Target: whether the encounter was followed by an inpatient readmission
within 30 days (`readmitted == "<30"`), the outcome CMS's Hospital
Readmissions Reduction Program (HRRP) actually penalizes hospitals on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/diabetic_data.csv"
PROCESSED_PATH = "data/processed/readmission_clean.parquet"

TARGET = "readmitted_lt_30"

# Discharge dispositions meaning the patient died or entered hospice care --
# these encounters cannot be meaningfully "readmitted" and are excluded
# rather than treated as negative (non-readmitted) examples, following the
# original dataset documentation.
EXPIRED_OR_HOSPICE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

# Columns with no signal (near-constant) or that are pure identifiers/too
# sparse to be useful (weight is 97% missing).
DROP_COLS = ["weight", "payer_code", "examide", "citoglipton"]

MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

# ICD-9 -> broad clinical category, following the grouping used in the
# original Strack et al. (2014) analysis of this dataset.
def _icd9_to_category(code: str) -> str:
    if code is None or code == "?" or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        val = float(code)
    except ValueError:
        return "Other"
    if val == 250 or (250 <= val < 251):
        return "Diabetes"
    if (390 <= val <= 459) or val == 785:
        return "Circulatory"
    if (460 <= val <= 519) or val == 786:
        return "Respiratory"
    if (520 <= val <= 579) or val == 787:
        return "Digestive"
    if 800 <= val <= 999:
        return "Injury"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if (580 <= val <= 629) or val == 788:
        return "Genitourinary"
    if 140 <= val <= 239:
        return "Neoplasms"
    return "Other"


AGE_MIDPOINTS = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}

CATEGORICAL_FEATURES = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty_grouped",
    "diag_1_category", "diag_2_category", "diag_3_category",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
] + MEDICATION_COLS

NUMERIC_FEATURES = [
    "age_numeric", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "total_prior_visits",
    "num_medication_changes",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["?"], low_memory=False)
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Row-level exclusions -------------------------------------------
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE_DISPOSITIONS)]
    df = df[df["gender"] != "Unknown/Invalid"]

    # --- Target -----------------------------------------------------
    df[TARGET] = (df["readmitted"] == "<30").astype(int)

    # --- Drop low-value columns ------------------------------------------
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # --- Categorical cleanup ----------------------------------------
    df["race"] = df["race"].fillna("Unknown")

    top_specialties = df["medical_specialty"].value_counts().head(10).index
    df["medical_specialty_grouped"] = np.where(
        df["medical_specialty"].isna(), "Missing",
        np.where(df["medical_specialty"].isin(top_specialties), df["medical_specialty"], "Other"),
    )

    df["diag_1_category"] = df["diag_1"].apply(_icd9_to_category)
    df["diag_2_category"] = df["diag_2"].apply(_icd9_to_category)
    df["diag_3_category"] = df["diag_3"].apply(_icd9_to_category)

    # id-type columns are codes, not ordinal magnitudes, so treat as strings
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[col] = df[col].astype(str)

    # --- Numeric / ordinal engineering -----------------------------------
    df["age_numeric"] = df["age"].map(AGE_MIDPOINTS)
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["num_medication_changes"] = (
        df[MEDICATION_COLS].isin(["Up", "Down"]).sum(axis=1)
    )

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    return df


def prepare(raw_path: str = RAW_PATH, save_path: str | None = PROCESSED_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = clean_and_engineer(df)
    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path)
    return df


if __name__ == "__main__":
    out = prepare()
    print(f"Prepared {len(out):,} rows (from raw {len(load_raw()):,}), {out.shape[1]} columns -> {PROCESSED_PATH}")
    print(f"30-day readmission rate: {out[TARGET].mean():.3%}")
    print(f"Unique patients: {out['patient_nbr'].nunique():,}")
