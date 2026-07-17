"""
External Validation of the GAM1.3 Ensemble — TCGA-LUAD -> OncoSG
================================================================
Trains the GAM1.3 stacking ensemble (RSF, GBS, XGBoost-Cox, DeepSurv -> GAM
meta-learner) on ALL TCGA-LUAD patients, then evaluates it on an independent
external cohort: Lung Adenocarcinoma (OncoSG, Nat Genet 2020), n=305.

Because the two cohorts share only a subset of features, the model is trained
on the SIX features common to both, so the exact same model transfers:
    Age, Sex, Stage (coarsened to I/II/III/IV), TMB, Mutation Count,
    Fraction Genome Altered
(Hypoxia scores, aneuploidy, MSI, etc. exist only in TCGA and are excluded —
imputing them as constants for the external cohort would not be real validation.)

Reuses the corrected methodology from GAM1.3:
  - XGBoost censoring via label sign (no weight=event)
  - Early stopping on a nested split, never on the prediction target
  - Leakage-free: all preprocessing/splines fit on TCGA only

Outputs -> results/
"""

from pathlib import Path
import random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from patsy import dmatrix, build_design_matrices
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TCGA_CSV   = DATA_DIR / "luad_tcga_pan_can_atlas_2018_clinical_data.csv"
ONCOSG_CSV = DATA_DIR / "luad_oncosg_2020_clinical_data.csv"

print("=" * 78)
print("EXTERNAL VALIDATION — GAM1.3 ENSEMBLE:  TCGA-LUAD (train)  ->  OncoSG (test)")
print("=" * 78)
print(f"Device: {device}\n")

CAT_FEATS = ["SEX", "STAGE"]
# Mutation count dropped (Spearman 0.98 with TMB in TCGA) to keep the feature
# set fair and non-redundant across all models.
NUM_FEATS = ["AGE", "TMB", "FGA"]
COMMON_FEATS = CAT_FEATS + NUM_FEATS

# ============================================================================
# DATA HARMONIZATION
# ============================================================================

def coarsen_stage(s):
    """Map any AJCC stage string to coarse I/II/III/IV (or NaN)."""
    s = str(s).upper().replace("STAGE", "").strip()
    if s in ("", "NAN", "NA"):
        return np.nan
    if s.startswith("IV") or s == "4":
        return "IV"
    if s.startswith("III"):
        return "III"
    if s.startswith("II"):
        return "II"
    if s.startswith("I") or s == "1":
        return "I"
    return np.nan

def harmonize(df, cols):
    """Build the common-feature frame + survival targets from a source frame.
    `cols` maps harmonized name -> source column name."""
    out = pd.DataFrame(index=df.index)
    out["AGE"]       = pd.to_numeric(df[cols["AGE"]], errors="coerce")
    out["SEX"]       = df[cols["SEX"]].astype(str).str.strip().str.capitalize()
    out["STAGE"]     = df[cols["STAGE"]].map(coarsen_stage)
    out["TMB"]       = pd.to_numeric(df[cols["TMB"]], errors="coerce")
    out["MUT_COUNT"] = pd.to_numeric(df[cols["MUT_COUNT"]], errors="coerce")
    out["FGA"]       = pd.to_numeric(df[cols["FGA"]], errors="coerce")
    out["OS_time"]   = pd.to_numeric(df[cols["OS_time"]], errors="coerce")
    status           = df[cols["OS_status"]].astype(str)
    out["OS_event"]  = status.str.strip().str.startswith("1").astype(int)  # '1:DECEASED'
    out = out.dropna(subset=["OS_time", "OS_event"])
    out = out[out["OS_time"] > 0]
    # Categorical missing -> explicit 'Unknown' so OHE can encode it
    for c in CAT_FEATS:
        out[c] = out[c].where(out[c].notna(), "Unknown").replace({"": "Unknown"})
    return out.reset_index(drop=True)

print("[1] Loading and harmonizing cohorts...")
tcga_raw = pd.read_csv(TCGA_CSV)
onco_raw = pd.read_csv(ONCOSG_CSV)

tcga = harmonize(tcga_raw, {
    "AGE": "Diagnosis Age", "SEX": "Sex",
    "STAGE": "Neoplasm Disease Stage American Joint Committee on Cancer Code",
    "TMB": "TMB (nonsynonymous)", "MUT_COUNT": "Mutation Count",
    "FGA": "Fraction Genome Altered",
    "OS_time": "Overall Survival (Months)", "OS_status": "Overall Survival Status",
})
onco = harmonize(onco_raw, {
    "AGE": "AGE", "SEX": "SEX", "STAGE": "STAGE",
    "TMB": "TMB_NONSYNONYMOUS", "MUT_COUNT": "MUTATION_COUNT",
    "FGA": "FRACTION_GENOME_ALTERED",
    "OS_time": "OS_MONTHS", "OS_status": "OS_STATUS",
})

print(f"  TCGA   (train): n={len(tcga):3d}  events={tcga['OS_event'].sum():3d} "
      f"({tcga['OS_event'].mean()*100:.1f}%)")
print(f"  OncoSG (test):  n={len(onco):3d}  events={onco['OS_event'].sum():3d} "
      f"({onco['OS_event'].mean()*100:.1f}%)")
print(f"  Common features: {COMMON_FEATS}")
print(f"  Stage dist  TCGA: {tcga['STAGE'].value_counts().to_dict()}")
print(f"  Stage dist  Onco: {onco['STAGE'].value_counts().to_dict()}")

# ============================================================================
# PREPROCESSING (FIT ON TCGA ONLY)
# ============================================================================

print("\n[2] Preprocessing (fit on TCGA only)...")
pre = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_FEATS),
    ("num", SimpleImputer(strategy="median"), NUM_FEATS),
], remainder="drop")

Xtr_p = pre.fit_transform(tcga[COMMON_FEATS])
Xex_p = pre.transform(onco[COMMON_FEATS])
scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr_p)
Xex = scaler.transform(Xex_p)
feat_names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
print(f"  Encoded feature dim: {Xtr.shape[1]}  ({feat_names})")

y_tcga = Surv.from_arrays(event=tcga["OS_event"].values.astype(bool),
                          time=tcga["OS_time"].values)
y_onco_ev = onco["OS_event"].values.astype(bool)
y_onco_ti = onco["OS_time"].values

# ============================================================================
# BASE LEARNERS  (OOF on TCGA for meta-learner + final models -> OncoSG)
# ============================================================================

class DeepSurv(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def cox_ph_loss(risk, times, events):
    order = torch.argsort(-times)
    r, e = risk[order], events[order]
    log_cum = torch.logcumsumexp(r, dim=0)
    return -(e * (r - log_cum)).sum() / (e.sum() + 1e-8)

def train_deepsurv(Xt, yt, Xes, yes, n):
    net = DeepSurv(n).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tt = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    Xt_t, Xes_t = tt(Xt), tt(Xes)
    yt_ti, yt_ev = tt([e["time"] for e in yt]), tt([e["event"] for e in yt])
    yes_ti, yes_ev = tt([e["time"] for e in yes]), tt([e["event"] for e in yes])
    best, wait, state = np.inf, 0, None
    for _ in range(400):
        net.train(); opt.zero_grad()
        cox_ph_loss(net(Xt_t), yt_ti, yt_ev).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = cox_ph_loss(net(Xes_t), yes_ti, yes_ev).item()
        if vl < best - 1e-6:
            best, wait, state = vl, 0, {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= 25:
                break
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    return net

def ds_predict(net, X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

XGB_P = dict(objective="survival:cox", eval_metric="cox-nloglik", eta=0.05,
             max_depth=3, subsample=0.8, colsample_bytree=0.8, seed=SEED, verbosity=0)

def xgb_label(y):
    return [e["time"] if e["event"] else -e["time"] for e in y]

def fit_xgb(Xt, yt, Xes, yes):
    dtr = xgb.DMatrix(Xt, label=xgb_label(yt))
    des = xgb.DMatrix(Xes, label=xgb_label(yes))
    m = xgb.train(XGB_P, dtr, num_boost_round=2000, evals=[(des, "v")],
                  early_stopping_rounds=100, verbose_eval=False)
    it = (m.best_iteration + 1) if hasattr(m, "best_iteration") else m.num_boosted_rounds()
    return m, it

print("\n[3] Base learners: 5-fold OOF on TCGA + final models...")
META = ["RSF", "GBS", "XGB", "DeepSurv"]
n_tr = len(Xtr)
oof = np.zeros((n_tr, 4))
kf = KFold(5, shuffle=True, random_state=SEED)

for k, (tr_i, vl_i) in enumerate(kf.split(Xtr)):
    Xt, Xv = Xtr[tr_i], Xtr[vl_i]
    yt, yv = y_tcga[tr_i], y_tcga[vl_i]
    # nested split for early stopping (never touches Xv)
    es_tr, es_vl = train_test_split(np.arange(len(Xt)), test_size=0.2,
                                    random_state=SEED, stratify=yt["event"])
    Xes, yes = Xt[es_vl], yt[es_vl]

    rsf = RandomSurvivalForest(n_estimators=500, max_features="sqrt",
                               min_samples_leaf=3, random_state=SEED, n_jobs=-1).fit(Xt, yt)
    oof[vl_i, 0] = rsf.predict(Xv)
    gbs = GradientBoostingSurvivalAnalysis(n_estimators=500, learning_rate=0.05,
                                           max_depth=3, random_state=SEED).fit(Xt, yt)
    oof[vl_i, 1] = gbs.predict(Xv)
    xm, it = fit_xgb(Xt[es_tr], yt[es_tr], Xes, yes)
    oof[vl_i, 2] = xm.predict(xgb.DMatrix(Xv), iteration_range=(0, it))
    net = train_deepsurv(Xt[es_tr], yt[es_tr], Xes, yes, Xt.shape[1])
    oof[vl_i, 3] = ds_predict(net, Xv)
    print(f"  fold {k+1}/5 done")

# TCGA internal OOF C-index per base model (sanity)
print("  TCGA OOF C-index:  " + "  ".join(
    f"{m}={concordance_index_censored(y_tcga['event'], y_tcga['time'], oof[:, i])[0]:.3f}"
    for i, m in enumerate(META)))

# Final base learners on ALL TCGA -> apply to OncoSG
es_tr, es_vl = train_test_split(np.arange(n_tr), test_size=0.2,
                                random_state=SEED, stratify=y_tcga["event"])
Xes_f, yes_f = Xtr[es_vl], y_tcga[es_vl]

rsf_f = RandomSurvivalForest(n_estimators=500, max_features="sqrt",
                             min_samples_leaf=3, random_state=SEED, n_jobs=-1).fit(Xtr, y_tcga)
gbs_f = GradientBoostingSurvivalAnalysis(n_estimators=500, learning_rate=0.05,
                                         max_depth=3, random_state=SEED).fit(Xtr, y_tcga)
xm_f, it_f = fit_xgb(Xtr[es_tr], y_tcga[es_tr], Xes_f, yes_f)
net_f = train_deepsurv(Xtr[es_tr], y_tcga[es_tr], Xes_f, yes_f, Xtr.shape[1])

ext_pred = np.column_stack([
    rsf_f.predict(Xex),
    gbs_f.predict(Xex),
    xm_f.predict(xgb.DMatrix(Xex), iteration_range=(0, it_f)),
    ds_predict(net_f, Xex),
])

# ============================================================================
# GAM META-LEARNER  (spline stack, fit on TCGA OOF)
# ============================================================================

print("\n[4] GAM meta-learner...")
meta_tr = pd.DataFrame(oof, columns=META)
meta_ex = pd.DataFrame(ext_pred, columns=META)
for f in META:                                  # clip external to TCGA range
    lo, hi = meta_tr[f].min(), meta_tr[f].max()
    meta_ex[f] = meta_ex[f].clip(lo, hi)

design_infos = {}
def build_spline_matrix(frame, fit):
    parts = []
    for f in META:
        if fit:
            sp = dmatrix(f"bs({f}, df=4, degree=3, include_intercept=False)",
                         frame, return_type="dataframe")
            design_infos[f] = sp.design_info
        else:
            sp = pd.DataFrame(build_design_matrices([design_infos[f]], frame)[0],
                              index=frame.index)
        sp.columns = [f"{f}_s{i}" for i in range(sp.shape[1])]
        parts.append(sp)
    return pd.concat(parts, axis=1)

sp_tr = build_spline_matrix(meta_tr, fit=True)
sp_ex = build_spline_matrix(meta_ex, fit=False)

y_tr_s = np.array(list(zip(y_tcga["event"], y_tcga["time"])),
                  dtype=[("event", bool), ("time", float)])

# choose GAM alpha by 5-fold CV on TCGA OOF meta-features (no external leakage)
best_alpha_gam, best_c = 0.01, -1
for a in [0.001, 0.005, 0.01, 0.05, 0.1]:
    cs = []
    for tr_i, vl_i in KFold(5, shuffle=True, random_state=SEED).split(sp_tr):
        g = CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=0.9, max_iter=100000)
        g.fit(sp_tr.values[tr_i], y_tr_s[tr_i])
        cs.append(concordance_index_censored(
            y_tr_s["event"][vl_i], y_tr_s["time"][vl_i],
            g.predict(sp_tr.values[vl_i]))[0])
    if np.mean(cs) > best_c:
        best_c, best_alpha_gam = np.mean(cs), a
print(f"  best GAM alpha (TCGA CV): {best_alpha_gam} (CV C-index {best_c:.3f})")

gam = CoxnetSurvivalAnalysis(alphas=[best_alpha_gam], l1_ratio=0.9, max_iter=100000)
gam.fit(sp_tr.values, y_tr_s)
ext_risk_gam = gam.predict(sp_ex.values)

# ============================================================================
# EVALUATION ON OncoSG
# ============================================================================

print("\n" + "=" * 78)
print("EXTERNAL RESULTS ON OncoSG")
print("=" * 78)

def boot_ci(ev, ti, risk, n=1000):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(ev), len(ev), replace=True)
        if ev[idx].sum() == 0:
            continue
        vals.append(concordance_index_censored(ev[idx], ti[idx], risk[idx])[0])
    return np.percentile(vals, [2.5, 97.5])

rows = []
for i, m in enumerate(META):
    c = concordance_index_censored(y_onco_ev, y_onco_ti, ext_pred[:, i])[0]
    lo, hi = boot_ci(y_onco_ev, y_onco_ti, ext_pred[:, i])
    rows.append((m, c, lo, hi))
c_gam = concordance_index_censored(y_onco_ev, y_onco_ti, ext_risk_gam)[0]
lo_g, hi_g = boot_ci(y_onco_ev, y_onco_ti, ext_risk_gam)
rows.append(("GAM Ensemble", c_gam, lo_g, hi_g))

print(f"\n{'Model':<16} {'C-index':>8}   95% CI")
print("-" * 46)
for m, c, lo, hi in rows:
    print(f"{m:<16} {c:>8.4f}   [{lo:.4f}, {hi:.4f}]")

# KM stratification on OncoSG by GAM risk tertiles
risk_grp = pd.qcut(ext_risk_gam, 3, labels=["Low", "Medium", "High"])
lr = logrank_test(y_onco_ti[risk_grp == "Low"], y_onco_ti[risk_grp == "High"],
                  y_onco_ev[risk_grp == "Low"], y_onco_ev[risk_grp == "High"])
mv = multivariate_logrank_test(y_onco_ti, risk_grp, y_onco_ev)
print(f"\nKM risk stratification (GAM tertiles):")
print(f"  Low vs High log-rank p = {lr.p_value:.4e}")
print(f"  Multivariate log-rank p = {mv.p_value:.4e}")

fig, ax = plt.subplots(figsize=(9, 6))
kmf = KaplanMeierFitter()
for grp, color in zip(["Low", "Medium", "High"], ["#2E7D32", "#F57C00", "#C62828"]):
    mask = risk_grp == grp
    kmf.fit(y_onco_ti[mask], y_onco_ev[mask], label=f"{grp} risk (n={mask.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2.5)
ax.set_xlabel("Time (Months)"); ax.set_ylabel("Overall Survival Probability")
ax.set_title(f"External Validation on OncoSG — GAM Risk Tertiles\n"
             f"Low vs High log-rank p = {lr.p_value:.4f}")
ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "external_km_oncosg.png", dpi=300, bbox_inches="tight")
plt.close()

# Save summary
summary = pd.DataFrame(rows, columns=["Model", "C_index", "CI_low", "CI_high"])
summary.to_csv(OUTPUT_DIR / "external_validation_oncosg.csv", index=False)
with open(OUTPUT_DIR / "external_validation_oncosg.txt", "w", encoding="utf-8") as f:
    f.write("EXTERNAL VALIDATION — TCGA-LUAD (train) -> OncoSG (test)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Train (TCGA):  n={len(tcga)}, events={tcga['OS_event'].sum()}\n")
    f.write(f"Test (OncoSG): n={len(onco)}, events={onco['OS_event'].sum()}\n")
    f.write(f"Common features: {COMMON_FEATS}\n\n")
    f.write(summary.to_string(index=False))
    f.write(f"\n\nKM (GAM tertiles): Low vs High p={lr.p_value:.4e}, "
            f"multivariate p={mv.p_value:.4e}\n")

print(f"\n✓ Saved: results/external_validation_oncosg.txt / .csv")
print(f"✓ Saved: results/external_km_oncosg.png")
print("\n" + "=" * 78)
print("EXTERNAL VALIDATION COMPLETE")
print("=" * 78)
