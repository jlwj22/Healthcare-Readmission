"""
LACE index: the clinical rule-of-thumb baseline for 30-day readmission.

LACE (van Walraven et al., CMAJ 2010) is the score many hospitals and care
management programs already use to flag readmission risk. It adds up four
components:

    L  Length of stay (days)                0-7 points
    A  Acuity: emergent admission           0 or 3 points
    C  Comorbidity: Charlson index          0-5 points
    E  Emergency department visits          0-4 points

for a total of 0-19. Any ML model that wants to replace it has to actually
beat it, so src/train.py scores every test encounter with LACE and reports
it alongside the ML models.

Two adaptations to this dataset, both of which work against LACE a little:

- **Charlson comes from only three diagnosis codes.** The dataset records a
  primary, secondary, and tertiary ICD-9 code per encounter, not the full
  problem list, so comorbidities coded further down are invisible here.
- **ED visits cover the prior year, not the prior 6 months.** That's how
  `number_emergency` is defined in the dataset documentation.

Charlson conditions and weights follow the Quan et al. (Med Care 2005)
enhanced ICD-9-CM coding with the original Charlson weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (condition, weight, numeric ICD-9 ranges as [low, high), V-code prefixes)
# Ranges are numeric because this dataset drops leading zeros from ICD-9
# codes (e.g. HIV's 042 shows up as "42"), so string prefixes would miss them.
CHARLSON_CONDITIONS = [
    ("myocardial_infarction", 1, [(410, 411), (412, 413)], []),
    ("congestive_heart_failure", 1, [
        (398.91, 398.92), (402.01, 402.02), (402.11, 402.12), (402.91, 402.92),
        (404.01, 404.02), (404.03, 404.04), (404.11, 404.12), (404.13, 404.14),
        (404.91, 404.92), (404.93, 404.94), (425.4, 426), (428, 429),
    ], []),
    ("peripheral_vascular", 1, [
        (93.0, 93.1), (437.3, 437.4), (440, 442), (443.1, 444), (447.1, 447.2),
        (557.1, 557.2), (557.9, 558),
    ], ["V43.4"]),
    ("cerebrovascular", 1, [(362.34, 362.35), (430, 439)], []),
    ("dementia", 1, [(290, 291), (294.1, 294.2), (331.2, 331.3)], []),
    ("chronic_pulmonary", 1, [
        (416.8, 417), (490, 506), (506.4, 506.5), (508.1, 508.2), (508.8, 508.9),
    ], []),
    ("rheumatic", 1, [
        (446.5, 446.6), (710.0, 710.5), (714.0, 714.3), (714.8, 714.9), (725, 726),
    ], []),
    ("peptic_ulcer", 1, [(531, 535)], []),
    ("mild_liver", 1, [
        (70.22, 70.24), (70.32, 70.34), (70.44, 70.45), (70.54, 70.55),
        (70.6, 70.7), (70.9, 71), (570, 572), (573.3, 573.5), (573.8, 574),
    ], ["V42.7"]),
    ("diabetes_uncomplicated", 1, [(250.0, 250.4), (250.8, 251)], []),
    ("diabetes_complicated", 2, [(250.4, 250.8)], []),
    ("hemiplegia_paraplegia", 2, [
        (334.1, 334.2), (342, 344), (344.0, 344.7), (344.9, 345),
    ], []),
    ("renal", 2, [
        (403.01, 403.02), (403.11, 403.12), (403.91, 403.92),
        (404.02, 404.04), (404.12, 404.14), (404.92, 404.94),
        (582, 583), (583.0, 583.8), (585, 587), (588.0, 588.1),
    ], ["V42.0", "V45.1", "V56"]),
    ("malignancy", 2, [(140, 173), (174, 195.9), (200, 209), (238.6, 238.7)], []),
    ("moderate_severe_liver", 3, [(456.0, 456.3), (572.2, 572.9)], []),
    ("metastatic_tumor", 6, [(196, 200)], []),
    ("hiv_aids", 6, [(42, 45)], []),
]

# When a patient has both the mild and severe form of a condition, only the
# severe one counts (standard Charlson hierarchy).
_HIERARCHY = [
    ("diabetes_complicated", "diabetes_uncomplicated"),
    ("moderate_severe_liver", "mild_liver"),
    ("metastatic_tumor", "malignancy"),
]

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]


def _code_conditions(code) -> set[str]:
    """Charlson conditions matched by a single ICD-9 code string."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return set()
    code = str(code).strip()
    if not code or code == "?":
        return set()

    matched = set()
    if code[0] in "VE":
        for name, _, _, v_prefixes in CHARLSON_CONDITIONS:
            if any(code.startswith(p) for p in v_prefixes):
                matched.add(name)
        return matched

    try:
        val = round(float(code), 2)
    except ValueError:
        return matched
    for name, _, ranges, _ in CHARLSON_CONDITIONS:
        if any(lo <= val < hi for lo, hi in ranges):
            matched.add(name)
    return matched


_WEIGHTS = {name: weight for name, weight, _, _ in CHARLSON_CONDITIONS}


def charlson_index(codes) -> int:
    """Charlson comorbidity index from an iterable of ICD-9 codes."""
    conditions = set()
    for code in codes:
        conditions |= _code_conditions(code)
    for severe, mild in _HIERARCHY:
        if severe in conditions:
            conditions.discard(mild)
    return sum(_WEIGHTS[c] for c in conditions)


def length_of_stay_points(days: int) -> int:
    if days < 1:
        return 0
    if days <= 3:
        return int(days)
    if days <= 6:
        return 4
    if days <= 13:
        return 5
    return 7


def comorbidity_points(charlson: int) -> int:
    return 5 if charlson >= 4 else int(charlson)


def ed_visit_points(visits: int) -> int:
    return min(int(visits), 4)


def is_emergent(admission_type_id, admission_source_id) -> bool:
    """Admitted as an emergency, or came in through the emergency room."""
    return str(admission_type_id) == "1" or str(admission_source_id) == "7"


def lace_components(df: pd.DataFrame) -> pd.DataFrame:
    """L, A, C, E points and the total LACE score for each encounter."""
    charlson = df[DIAG_COLS].apply(lambda r: charlson_index(r.values), axis=1)
    out = pd.DataFrame(index=df.index)
    out["L"] = df["time_in_hospital"].map(length_of_stay_points)
    out["A"] = [
        3 if is_emergent(t, s) else 0
        for t, s in zip(df["admission_type_id"], df["admission_source_id"])
    ]
    out["charlson_index"] = charlson
    out["C"] = charlson.map(comorbidity_points)
    out["E"] = df["number_emergency"].map(ed_visit_points)
    out["lace_score"] = out[["L", "A", "C", "E"]].sum(axis=1)
    return out


def lace_score(df: pd.DataFrame) -> pd.Series:
    return lace_components(df)["lace_score"]
