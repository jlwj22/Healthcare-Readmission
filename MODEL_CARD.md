# Model Card: 30-Day Readmission Risk (XGBoost)

A short, structured summary of what this model is, what it's for, how well
it works, and where it shouldn't be trusted. Format loosely follows
Mitchell et al., "Model Cards for Model Reporting" (2019). Every number
here comes from `models/metrics.json`, which `python -m src.train`
regenerates.

## Model details

| | |
|---|---|
| **Model** | XGBoost gradient-boosted trees, binary classifier, native categorical handling |
| **Hyperparameters** | 400 trees, max depth 5, learning rate 0.05, 80% row and column subsampling, no class weighting |
| **Output** | Probability of an unplanned inpatient readmission within 30 days of discharge |
| **Inputs** | 44 encounter-level features: demographics (age, sex), admission and discharge codes, length of stay, lab/procedure/medication counts, prior-year outpatient/ED/inpatient visits, grouped ICD-9 diagnosis categories, HbA1c and glucose results, and 21 diabetes medication indicators |
| **Excluded on purpose** | Race (used only for subgroup evaluation), patient ID, payer code |
| **Comparators** | Logistic regression on the same features; the LACE index (`src/lace.py`) |
| **Code** | `src/train.py` (training and evaluation), `app.py` (demo) |
| **Artifact** | `models/xgb.joblib` |

## Intended use

- **Primary use:** ranking recently discharged members of a health plan so
  a care-management team can prioritize post-discharge outreach (a call,
  medication reconciliation, an early follow-up visit) when it can't reach
  everyone.
- **Users:** population-health and care-management analysts, not
  clinicians at the point of care.
- **Also reasonable:** using the calibrated probabilities to estimate
  expected readmission counts or cost exposure for a population, and to
  run what-if analyses on outreach thresholds.

## Out of scope

- Clinical diagnosis or treatment decisions for an individual patient.
- Denying, limiting, or pricing coverage or services. A risk score built to
  target *extra help* should not be repurposed to ration care.
- Any population unlike the training data (non-diabetic patients,
  pediatrics, non-US settings) without re-validation.
- Live deployment on current data without prospective validation. The
  training data is from 1999-2008.

## Training and evaluation data

[Diabetes 130-US Hospitals for Years 1999-2008](https://doi.org/10.24432/C5230J)
(UCI ML Repository, CC BY 4.0). 101,766 inpatient encounters of diabetic
patients from 130 US hospitals; 99,340 encounters from 69,987 patients
after excluding deaths, hospice discharges, and invalid records. The 30-day
readmission rate is 11.4%.

Split 75/25 **by patient** (`GroupShuffleSplit` on `patient_nbr`), so no
patient appears in both sets. Test set: 24,757 encounters, 17,497 patients,
2,789 readmissions (11.3%).

## Performance

95% CIs from 1,000 bootstrap resamples of test-set patients.

| Metric | XGBoost | Logistic regression | LACE index |
|---|---|---|---|
| ROC-AUC | 0.677 [0.665, 0.689] | 0.667 [0.656, 0.679] | 0.571 [0.558, 0.583] |
| PR-AUC | 0.235 [0.216, 0.255] | 0.220 [0.201, 0.240] | 0.138 [0.130, 0.146] |
| Readmissions caught, top 10% | 25.4% [23.6, 27.0] | 23.2% [21.6, 24.8] | 14.4% [13.3, 15.9] |
| Readmissions caught, top 20% | 40.4% [38.5, 42.3] | 38.6% [36.9, 40.2] | 27.7% [26.0, 29.4] |
| Brier score | 0.095 | 0.096 | 0.099 |

- **vs. LACE:** +0.106 [0.091, 0.120] ROC-AUC, +12.7 [10.3, 15.1] points
  of top-20% capture.
- **vs. logistic regression:** +0.010 [0.004, 0.016] ROC-AUC.
- **Leakage check:** the same model on a naive row-level split scores
  0.681 ROC-AUC, about 0.004 higher, with 36% of its test patients also in
  training. The patient-level numbers above are the ones to trust.

## Calibration

| | |
|---|---|
| Mean predicted risk | 11.2% |
| Observed rate | 11.3% |
| Observed / expected | 1.00 |
| Calibration slope | 0.95 (1.0 is ideal) |

An earlier version trained with `scale_pos_weight` had the same ranking but
a mean predicted risk of 43%; class weighting was removed for that reason.
See `reports/figures/calibration.png`.

## Decision analysis

- **Decision curve:** XGBoost has the highest net benefit of any policy
  (contact everyone, contact no one, logistic regression, LACE) at nearly
  every threshold from 0.03 to 0.45; logistic regression edges it by a
  negligible margin at 0.08 (`reports/figures/decision_curve.png`).
- **Cost scenario:** at an assumed $150 per outreach contact, $17,700 per
  readmission (AHRQ HCUP), and 15% assumed effectiveness, outreach breaks
  even at 5.6% risk, and the net-savings-maximizing threshold is 0.05. The
  cost and effectiveness figures are scenario inputs, not measured program
  results.

## Subgroup evaluation

All groups use one shared cutoff (the top 20% of risk overall).

- **Race** (not a model input): calibration within about one percentage
  point for African American, Caucasian, and Hispanic members. Recall
  ranges 40-43% for groups with recorded race. Members with race
  unrecorded have lower recall (29%, n=561, noisy).
- **Age:** calibration holds across bands, but ROC-AUC falls from about
  0.72 under 60 to 0.62 at 80+. The model ranks the oldest members least
  well.
- **Sex:** similar calibration; recall 42% (female) vs. 39% (male).

Full table in the README and `models/metrics.json` → `subgroup_performance`.

## Limitations

- **Old data.** 1999-2008 care patterns, coding, and costs; the HRRP
  program started after this period and changed hospital discharge
  behavior.
- **Diabetic inpatients only**, from hospitals that participated in one
  research database.
- **Encounter snapshot, not claims history.** No pharmacy fills,
  post-discharge follow-up, social determinants, or cost data, which is a
  big part of why discrimination tops out near 0.68.
- **Only three diagnosis codes per encounter**, which limits comorbidity
  detection for both the model and LACE.
- **No temporal or external validation.** The dataset has no dates or
  hospital identifiers, so the test set is a random patient-level holdout
  from the same period and hospitals.

## Ethical considerations

- Race is excluded as an input, but other features (prior utilization,
  discharge disposition) can act as proxies for access to care, so subgroup
  checks need to run on every retrain, not just once.
- Prior-utilization features reward members who already use the health
  system. Members with poor access to care may look lower-risk than they
  are, the failure mode documented in Obermeyer et al. (Science, 2019).
- Outputs are a prioritization aid for adding outreach, and a human care
  manager should make the final call.
