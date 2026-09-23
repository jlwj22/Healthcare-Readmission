import pandas as pd
import pytest

from src.lace import (
    charlson_index,
    comorbidity_points,
    ed_visit_points,
    is_emergent,
    lace_components,
    length_of_stay_points,
)


@pytest.mark.parametrize("days,points", [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (6, 4), (7, 5), (13, 5), (14, 7), (20, 7),
])
def test_length_of_stay_points(days, points):
    assert length_of_stay_points(days) == points


@pytest.mark.parametrize("charlson,points", [(0, 0), (1, 1), (3, 3), (4, 5), (9, 5)])
def test_comorbidity_points_cap_at_five(charlson, points):
    assert comorbidity_points(charlson) == points


@pytest.mark.parametrize("visits,points", [(0, 0), (2, 2), (4, 4), (11, 4)])
def test_ed_visit_points_cap_at_four(visits, points):
    assert ed_visit_points(visits) == points


def test_emergent_via_admission_type_or_er_source():
    assert is_emergent("1", "1")      # emergency admission type
    assert is_emergent("3", "7")      # elective type, but came through the ER
    assert not is_emergent("3", "1")  # elective, physician referral


def test_charlson_single_conditions():
    assert charlson_index(["428"]) == 1          # heart failure
    assert charlson_index(["250.01"]) == 1       # diabetes, uncomplicated
    assert charlson_index(["250.4"]) == 2        # diabetes with renal complication
    assert charlson_index(["585"]) == 2          # chronic kidney disease
    assert charlson_index(["197"]) == 6          # metastatic
    assert charlson_index(["V42.0"]) == 2        # kidney transplant status
    assert charlson_index(["276", "V27", None, "?"]) == 0


def test_charlson_handles_dropped_leading_zeros():
    # The dataset writes ICD-9 042 (HIV) as "42".
    assert charlson_index(["42"]) == 6


def test_charlson_hierarchy_counts_only_severe_form():
    assert charlson_index(["250.01", "250.6"]) == 2   # not 1 + 2
    assert charlson_index(["571", "572.3"]) == 3      # mild + severe liver -> severe only
    assert charlson_index(["153", "197"]) == 6        # malignancy + metastatic -> metastatic


def test_charlson_sums_distinct_conditions_once():
    assert charlson_index(["428", "428.0", "410"]) == 2  # CHF counted once, plus MI


def test_lace_components_total():
    df = pd.DataFrame({
        "time_in_hospital": [5, 1],
        "admission_type_id": ["1", "3"],
        "admission_source_id": ["7", "1"],
        "diag_1": ["428", "V27"],
        "diag_2": ["585", None],
        "diag_3": ["250.01", None],
        "number_emergency": [2, 0],
    })
    out = lace_components(df)
    # L=4, A=3, Charlson=1+2+1=4 -> C=5, E=2
    assert out.loc[0, "lace_score"] == 4 + 3 + 5 + 2
    assert out.loc[1, "lace_score"] == 1
