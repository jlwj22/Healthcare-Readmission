import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    EXPIRED_OR_HOSPICE_DISPOSITIONS,
    MODEL_FEATURES,
    TARGET,
    _icd9_to_category,
    clean_and_engineer,
)


@pytest.fixture
def raw_sample():
    base = {
        "encounter_id": 1, "patient_nbr": 100, "race": "Caucasian", "gender": "Female",
        "age": "[60-70)", "weight": np.nan, "admission_type_id": 1,
        "discharge_disposition_id": 1, "admission_source_id": 7, "time_in_hospital": 3,
        "payer_code": np.nan, "medical_specialty": np.nan, "num_lab_procedures": 40,
        "num_procedures": 1, "num_medications": 12, "number_outpatient": 0,
        "number_emergency": 0, "number_inpatient": 1, "diag_1": "250.01",
        "diag_2": "401", "diag_3": "?", "number_diagnoses": 5,
        "max_glu_serum": "None", "A1Cresult": "None", "change": "No",
        "diabetesMed": "Yes", "readmitted": "NO",
    }
    for med in [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
        "examide", "citoglipton",
    ]:
        base[med] = "No"

    rows = []
    for i in range(3):
        r = dict(base)
        r["patient_nbr"] = 100 + i
        r["encounter_id"] = i + 1
        rows.append(r)
    # one row that should be excluded: expired
    expired = dict(base)
    expired["encounter_id"] = 99
    expired["patient_nbr"] = 999
    expired["discharge_disposition_id"] = 11
    rows.append(expired)
    # one row with an unknown/invalid gender that should be excluded
    unknown_gender = dict(base)
    unknown_gender["encounter_id"] = 98
    unknown_gender["patient_nbr"] = 998
    unknown_gender["gender"] = "Unknown/Invalid"
    rows.append(unknown_gender)
    # one readmitted-within-30-days row
    readmit = dict(base)
    readmit["encounter_id"] = 2
    readmit["patient_nbr"] = 101
    readmit["readmitted"] = "<30"
    rows.append(readmit)

    return pd.DataFrame(rows)


def test_excludes_expired_and_hospice(raw_sample):
    out = clean_and_engineer(raw_sample)
    assert out["discharge_disposition_id"].astype(str).isin(
        [str(d) for d in EXPIRED_OR_HOSPICE_DISPOSITIONS]
    ).sum() == 0


def test_excludes_unknown_gender(raw_sample):
    out = clean_and_engineer(raw_sample)
    assert (out["gender"] == "Unknown/Invalid").sum() == 0


def test_target_matches_readmitted_lt_30(raw_sample):
    out = clean_and_engineer(raw_sample)
    readmitted_rows = out[out[TARGET] == 1]
    assert len(readmitted_rows) == 1


def test_model_features_present(raw_sample):
    out = clean_and_engineer(raw_sample)
    for col in MODEL_FEATURES:
        assert col in out.columns, f"missing engineered column: {col}"


def test_no_missing_in_model_features(raw_sample):
    out = clean_and_engineer(raw_sample)
    assert out[MODEL_FEATURES].isna().sum().sum() == 0


@pytest.mark.parametrize(
    "code,expected",
    [
        ("250.01", "Diabetes"),
        ("401", "Circulatory"),
        ("486", "Respiratory"),
        ("V27", "Other"),
        ("E888", "Other"),
        (None, "Missing"),
        ("?", "Missing"),
    ],
)
def test_icd9_grouping(code, expected):
    assert _icd9_to_category(code) == expected


def test_total_prior_visits_sums_correctly(raw_sample):
    out = clean_and_engineer(raw_sample)
    row = out.iloc[0]
    assert row["total_prior_visits"] == (
        row["number_outpatient"] + row["number_emergency"] + row["number_inpatient"]
    )
