# TCGA-LUAD Survival-Aware GAM Meta-Learner

Survival analysis pipeline on TCGA lung adenocarcinoma (LUAD) data. Four base
survival models are stacked with a GAM (spline) meta-learner, using nested
cross-validation to prevent leakage.

## Pipeline

Everything is in a single script: [`notebooks/GAM1.3.py`](notebooks/GAM1.3.py).

- **Base learners:** Random Survival Forest, Gradient Boosting Survival, XGBoost (Cox), DeepSurv
- **Meta-learner:** Cox GAM with per-model B-spline smooths (Survival-Aware GAM)
- **Evaluation:** nested CV, held-out test C-index, bootstrap CIs, Kaplan-Meier risk stratification, feature-selection and model-contribution analysis, plus targeted/pairwise ablations

## Data

Not included in the repo (patient data, git-ignored). Download the clinical data
for **"Lung Adenocarcinoma (TCGA, PanCancer Atlas)"** from
[cBioPortal](https://www.cbioportal.org/) and place it at:

```
data/luad_tcga_pan_can_atlas_2018_clinical_data.csv
```

## Setup

Use **Python 3.11** (3.14 has no prebuilt wheels for `scikit-survival`/`ecos`
and will try to compile from source).

```bash
py -3.11 -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
```

**Windows note:** if `torch` fails to install with a "filename too long"
error, create the venv at a short path outside OneDrive (e.g. `C:\tcga-venv`)
— deep nested paths exceed the 260-char limit.

## Run

```bash
python notebooks/GAM1.3.py
```

Runs on CPU by default (uses CUDA if available). Outputs — metrics CSVs, the
text report, and plots — are written to `results/`.

## External validation

External validation against independent GEO cohorts (e.g. GSE31210, GSE68465)
is planned and will reuse GAM1.3's preprocessing and trained meta-learner.
