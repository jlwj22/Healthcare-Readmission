# 30-Day Readmission Risk: Health Plan Care Management Model

A full-pipeline risk model built the way a health plan's population-health
or care-management analytics team would build it. It predicts which
recently discharged diabetic members are likely to be readmitted within 30
days, with a patient-level (leakage-safe) evaluation, a calibration check,
a fairness audit, SHAP explainability, and an interactive Streamlit demo for
a care-management targeting workflow.

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-portfolio%20project-lightgrey)

**[Live demo →](#running-the-demo)** &nbsp;|&nbsp; **[Notebook →](notebooks/01_eda_and_modeling.ipynb)**

![App demo](reports/screenshots/app_demo.png)

![Readmission by prior utilization](reports/figures/readmission_by_prior_visits.png)

## The business problem

Unplanned readmissions are one of the most expensive, most
preventable-adjacent categories of medical spend a health plan carries, and
one of the few a payer can actually act on before the cost hits, just by
knowing who to reach first. A care-management team can't call every
recently discharged member, so this model scores each discharge so that
limited outreach capacity (a call, medication reconciliation, an early
follow-up visit) goes to the members most likely to bounce back within 30
days. It's the same underlying signal CMS's Hospital Readmissions Reduction
Program (HRRP) financially penalizes hospitals on, which is exactly why
payers and provider systems both build this kind of model, just from
opposite sides of the same incentive.

## Dataset

[**Diabetes 130-US Hospitals for Years 1999-2008**](https://doi.org/10.24432/C5230J)
(Clore, Cios, DeShazo & Strack, 2014; UCI ML Repository, CC BY 4.0),
101,766 inpatient encounters for diabetic patients across 130 US hospitals.
After excluding encounters that ended in death or hospice discharge (which
can't meaningfully be "readmitted") and 3 rows with an unrecorded gender,
that leaves **99,340 encounters, 69,987 unique patients**. Target:
readmission within 30 days, an **11.4%** base rate after cleaning.

## A few choices this project is deliberate about

Most portfolio versions of this dataset get at least one of these wrong, so
they're worth calling out directly instead of just doing them quietly.

1. **Patient-level train/test split.** About 30% of patients in this
   dataset have more than one encounter. A random row-level split lets the
   same patient's encounters leak across train and test, which inflates
   apparent accuracy. This pipeline splits on `patient_nbr`
   (`GroupShuffleSplit`) instead and asserts the split is leakage-free
   before training even starts. See the split comparison below for what
   that actually buys you.
2. **Race is excluded from the model's features.** It's genuinely
   predictive in this data, as it is in most US healthcare data, which
   reflects structural inequities rather than biology. Training a clinical
   risk score that allocates care-management resources directly on race
   risks baking that discrimination into the tool, so it's kept out of `X`
   and instead used for a post-hoc fairness check (below).
3. **The model reports calibrated probabilities, not just a ranking.** A
   payer pricing outreach against expected cost needs a number close to the
   true probability, not just a score that ranks members correctly. See the
   calibration check below.

## Approach

1. **Clean and engineer** (`src/data_prep.py`): drop near-empty columns
   (weight is 97% missing), impute and flag the rest, collapse ICD-9
   diagnosis codes into 9 clinical categories (circulatory, diabetes,
   respiratory, and so on, following Strack et al.'s original grouping),
   and engineer prior-utilization and medication-change features.
2. **Model, champion vs. challenger** (`src/train.py`): class-weighted
   logistic regression baseline vs. XGBoost with native categorical
   support, evaluated on the patient-level held-out split.
3. **Evaluate for the actual use case**: ROC-AUC and PR-AUC, plus
   top-decile and top-quintile capture rate, meaning what share of real
   readmissions get caught if a care team can only follow up with the
   riskiest 10-20% of discharges.
4. **Check that the probabilities themselves are trustworthy**: a
   calibration curve, since a payer pricing outreach or cost exposure off
   this model needs the predicted probability to actually mean what it
   says.
5. **Check fairness before anything else**: error rates and
   predicted-positive rates broken out by race subgroup, even though race
   isn't a model input.
6. **Explain every score**: SHAP values, surfaced per-patient in the demo
   app.

## Results

Evaluated on a held-out, patient-level 25% test split (24,757 encounters,
no patient overlap with train):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression (baseline) | 0.668 | 0.220 |
| **XGBoost** | **0.675** | **0.232** |

An AUC around 0.68 is the honest result here, not a shortcoming to explain
away. Predicting a 30-day readmission from encounter data alone is a
genuinely hard problem. It depends heavily on factors this dataset simply
doesn't capture: housing stability, caregiver support, medication adherence
after discharge. Published academic and applied analyses of this exact
dataset land in the same 0.65-0.70 range, so a model claiming 0.90+ here
would be the signal something leaked, not that the model is better.

What matters for the actual use case is ranking, not raw discrimination.

- **Flagging the riskiest 10% of discharges catches 25% of all 30-day readmissions.**
- **Flagging the riskiest 20% catches 40%.**

That's the number a care-management team sizing a follow-up-call program
would actually use.

![ROC / PR curves](reports/figures/roc_pr_curves.png)

### Is the model calibrated?

Ranking members correctly isn't the whole job here. If a payer wants to use
predicted risk to estimate expected cost exposure (the section below does
exactly that), the predicted probabilities need to line up with observed
outcomes, not just put the right members in the right order.

![Calibration curve](reports/figures/calibration.png)

The model is reasonably well calibrated across most of the risk range,
tracking close to the diagonal, with some noise at the highest-risk bins
where there are fewer observations to average over. This wasn't guaranteed
by training XGBoost with a standard log-loss objective and class weighting,
so it's worth checking rather than assuming.

### Why the patient-level split actually matters

It's one thing to say a naive split leaks and another to show what that
leakage is worth. `src/train.py` also fits the identical XGBoost
architecture on a plain stratified row-level split that ignores
`patient_nbr` entirely, so the same patient can and does land in both train
and test.

| Split | ROC-AUC | PR-AUC | Test patients also seen in train |
|---|---|---|---|
| Patient-level (`GroupShuffleSplit`, used everywhere else in this project) | 0.675 | 0.232 | 0% (by construction) |
| Naive row-level (`train_test_split`, stratified on outcome only) | 0.678 | 0.235 | 35.8% |

The gap here is real but smaller than the leakage horror stories you'll
sometimes see, which is itself an interesting result. With `patient_nbr`
excluded from the feature set, the model can't directly memorize "this ID
tends to be high risk." What leaks instead is subtler: two encounters from
the same patient often share very similar demographic and clinical feature
values, so seeing one of them in training gives the model a small edge on
the other one at test time, even without ever seeing the patient ID
directly. About a third of a point of AUC doesn't sound dramatic, but on a
model this close to its ceiling, and multiplied across every metric and
report a team might build on top of it, it's exactly the kind of quiet
optimism that a good validation process is supposed to catch before it
reaches production. It's also a more honest finding than claiming the naive
split was wildly inflated: the actual answer is "modest, measurable, and
worth guarding against anyway."

SHAP shows prior inpatient visits, discharge disposition, and primary
diagnosis category dominate the prediction, consistent with both clinical
intuition and the EDA.

![SHAP summary](reports/figures/shap_summary.png)

### What that's worth in payer terms

AHRQ's Healthcare Cost and Utilization Project puts the **average cost of an
adult 30-day readmission at $17,700** (2020 Nationwide Readmissions
Database; [HCUP Statistical Brief #307](https://hcup-us.ahrq.gov/reports/statbriefs/sb307-readmissions-2020.jsp)).
Applying that to this test set alone (24,757 discharges, 2,789 actual
30-day readmissions) reframes the capture-rate numbers above as cost
concentration, not just model accuracy:

| Outreach targets | Discharges reviewed | Readmissions captured | Cost exposure captured |
|---|---|---|---|
| Top 10% by risk score | 2,476 | 705 | ≈ $12.5M |
| Top 20% by risk score | 4,951 | 1,125 | ≈ $19.9M |
| All 2,789 readmissions | 24,757 | 2,789 | ≈ $49.4M |

In other words, a care-management program that can only reach a fifth of
discharged members still gets in front of 40% of the total cost exposure,
which is the pitch a payer-side data science team actually has to make to
get outreach headcount funded. This is the cost exposure the flagged
population represents, not a claim about dollars an intervention would
save. That requires an actual program-effectiveness study, not a model.

### Fairness check

Race is not a model feature, but error rates were checked across race
subgroups post-hoc (`models/metrics.json`). Recall and predicted-positive
rate move together reasonably closely across groups, with more noise in the
smaller ones as you'd expect from sample size alone, and per-group AUC
stays in a similar band to the overall model. This isn't a substitute for a
full audit before any real deployment, but it's the kind of check that
should run before a model like this goes anywhere near a live workflow.

Full narrative, additional EDA, and every chart above (regenerated from
scratch, not screenshots) live in
[`notebooks/01_eda_and_modeling.ipynb`](notebooks/01_eda_and_modeling.ipynb).

## Running the demo

```bash
git clone <this-repo>
cd healthcare-readmission
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce data cleaning, training, evaluation, and all figures:
python -m src.train

# Launch the interactive risk dashboard:
streamlit run app.py
```

The app takes a discharge profile (prior utilization, length of stay,
diagnosis category, medications, and so on) and returns a 30-day
readmission risk estimate, a risk tier for care-management targeting, and
the top SHAP factors behind that specific prediction.

## Project structure

```
├── app.py                     # Streamlit demo
├── src/
│   ├── data_prep.py           # cleaning + feature engineering (shared by training & app)
│   └── train.py                # patient-level split, trains both models, evaluates, fairness check
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── data/reference/
│   └── ids_mapping.csv        # admission/discharge code -> description lookup
├── models/                    # trained model artifacts + metrics.json (generated)
├── reports/figures/           # evaluation charts (generated)
├── tests/                     # unit tests for the data pipeline
└── requirements.txt
```

## Notes and limitations

This is a portfolio project on public research data (1999-2008, 130 US
hospitals), not a validated clinical or actuarial tool. A real deployment
on a health plan's book of business would need prospective validation on
current data (care patterns and cost trends have shifted substantially
since 2008), a full fairness and bias audit beyond the subgroup check
above, integration with data this dataset doesn't have (claims history and
cost data instead of a single-encounter snapshot, social determinants of
health, pharmacy fill data, post-discharge follow-up records), and sign-off
from clinical, actuarial, and compliance review before it touches a live
care-management or utilization-management workflow.

## License

MIT, see [LICENSE](LICENSE). Dataset used under its original CC BY 4.0
license; see [`data/reference/ids_mapping.csv`](data/reference/ids_mapping.csv)
for the source code mappings and the dataset DOI above for full attribution.
