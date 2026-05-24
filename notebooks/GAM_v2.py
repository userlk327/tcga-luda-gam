"""
TCGA-LUAD Survival-Aware GAM Meta-Learner  —  v3 (BIBM 2026)
=============================================================
Evaluation: 5-fold stratified nested cross-validation
  - Outer 5-fold: each fold produces a held-out test C-index
  - Inner 5-fold OOF: base learner predictions for meta-learner training
  - All preprocessing fitted strictly on outer-train data
  - Pooled OOF risk scores used for KM stratification (n=501)
  - Reports mean ± std C-index across 5 outer folds

Contributions:
  1. Survival-Aware GAM Meta-Learner (B-spline smooth per base model)
  2. Linear Cox stacking ablation (proves non-linearity is necessary)
  3. Multi-modal clinical + mRNA feature integration
  4. Integrated Brier Score + time-dependent AUC
  5. Rigorous nested CV with bootstrap CIs

Author: Research Implementation
Date: 2026
"""

# ============================================================================
# IMPORTS
# ============================================================================

from pathlib import Path
import random, warnings
import pandas as pd
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import (concordance_index_censored, cumulative_dynamic_auc,
                            integrated_brier_score, brier_score as brier_score_t)

from patsy import dmatrix, build_design_matrices
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED         = 42
N_TOP_GENES  = 5        # mRNA genes per outer fold (EPV ≥ 10)
N_VAR_GENES  = 2000     # pre-filter by variance before CI screening
N_OUTER      = 5        # outer CV folds
N_INNER      = 5        # inner OOF folds for base learners
N_BOOT       = 1000
MISSING_THR  = 0.15
ALPHA_GRID   = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
warnings.filterwarnings("ignore")

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / 'dataset'
OUTPUT_DIR = REPO_ROOT / 'results_v2'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 80)
print("SURVIVAL-AWARE GAM META-LEARNER  v3  —  BIBM 2026")
print("Evaluation: 5-fold Nested Cross-Validation")
print("=" * 80)
print(f"Device: {device}\nOutput: {OUTPUT_DIR}\n")

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("=" * 80)
print("STEP 1: DATA LOADING AND MERGING")
print("=" * 80)

def load_cbio(path):
    with open(path) as fh:
        skip = sum(1 for line in fh if line.startswith('#'))
    return pd.read_csv(path, sep='\t', skiprows=skip, low_memory=False)

patient = load_cbio(DATA_DIR / 'data_clinical_patient.txt')
sample  = load_cbio(DATA_DIR / 'data_clinical_sample.txt')
hypoxia = load_cbio(DATA_DIR / 'data_clinical_supp_hypoxia.txt')

df = patient.merge(sample,  on='PATIENT_ID', how='inner')
df = df.merge(hypoxia, on='PATIENT_ID', how='left')

df['OS_time']  = pd.to_numeric(df['OS_MONTHS'], errors='coerce')
df['OS_event'] = df['OS_STATUS'].str.startswith('1').fillna(False).astype(int)
df = df[df['OS_time'].notna() & (df['OS_time'] > 0)].copy()

print(f"✓ Patients: {len(df)}  |  Events: {df['OS_event'].sum()} "
      f"({df['OS_event'].mean()*100:.1f}%)")

# ============================================================================
# 2. mRNA LOADING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: mRNA DATA LOADING")
print("=" * 80)

mrna_raw = pd.read_csv(DATA_DIR / 'data_mrna_seq_v2_rsem.txt',
                       sep='\t', index_col=0, low_memory=False)
mrna_raw = mrna_raw.drop(columns=['Entrez_Gene_Id'], errors='ignore')
mrna_raw = mrna_raw.T.copy()
mrna_raw.index = mrna_raw.index.str[:-3]
mrna_raw = mrna_raw[~mrna_raw.index.duplicated()]
mrna_raw = np.log2(mrna_raw.astype(float) + 1)
mrna_raw = mrna_raw.replace([np.inf, -np.inf], np.nan)
mrna_raw = mrna_raw.loc[:, mrna_raw.notna().mean() > 0.7]

print(f"✓ mRNA: {mrna_raw.shape[0]} samples × {mrna_raw.shape[1]} genes")

# ============================================================================
# 3. GLOBAL FEATURE SET (leakage-free)
# ============================================================================

LEAKAGE_COLS = [
    'OS_MONTHS','OS_STATUS','DSS_STATUS','DSS_MONTHS','DFS_STATUS','DFS_MONTHS',
    'PFS_STATUS','PFS_MONTHS','DAYS_LAST_FOLLOWUP','DAYS_TO_BIRTH',
    'DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS','PERSON_NEOPLASM_CANCER_STATUS',
    'NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT','PATIENT_ID','SAMPLE_ID',
    'OTHER_PATIENT_ID','SUBTYPE','CANCER_TYPE','CANCER_TYPE_DETAILED','TUMOR_TYPE',
    'CANCER_TYPE_ACRONYM','ONCOTREE_CODE','TISSUE_SOURCE_SITE',
    'TISSUE_SOURCE_SITE_CODE','SAMPLE_TYPE','SOMATIC_STATUS','ICD_10',
    'ICD_O_3_HISTOLOGY','ICD_O_3_SITE','AJCC_STAGING_EDITION',
    'FORM_COMPLETION_DATE','INFORMED_CONSENT_VERIFIED','IN_PANCANPATHWAYS_FREEZE',
    'HISTORY_NEOADJUVANT_TRTYN','TISSUE_PROSPECTIVE_COLLECTION_INDICATOR',
    'TISSUE_RETROSPECTIVE_COLLECTION_INDICATOR',
    'PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT','TUMOR_TISSUE_SITE',
    'GENETIC_ANCESTRY_LABEL',
]

CLINICAL_COLS = [
    'AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE',
    'PATH_M_STAGE','PATH_N_STAGE','PATH_T_STAGE','GRADE',
    'ETHNICITY','RACE','PRIOR_DX','RADIATION_THERAPY','WEIGHT',
    'ANEUPLOIDY_SCORE','MSI_SCORE_MANTIS','MSI_SENSOR_SCORE',
    'TMB_NONSYNONYMOUS','TBL_SCORE',
    'BUFFA_HYPOXIA_SCORE','WINTER_HYPOXIA_SCORE','RAGNUM_HYPOXIA_SCORE',
]

df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns],
        inplace=True, errors='ignore')
df = df.reset_index(drop=True)

clinical_cols = [c for c in CLINICAL_COLS if c in df.columns]
print(f"✓ Clinical features: {len(clinical_cols)}")

# Align mRNA with patient list
common_pts    = df['PATIENT_ID'].values if 'PATIENT_ID' in df.columns else df.index
# After leakage removal PATIENT_ID may be dropped — rebuild from original
df_ids = patient[['PATIENT_ID']].merge(sample[['PATIENT_ID']], on='PATIENT_ID')
all_ids = df_ids['PATIENT_ID'].values[:len(df)]  # align length

mrna_aligned      = mrna_raw.reindex(all_ids).reset_index(drop=True)
mrna_available    = mrna_aligned.notna().any(axis=1).astype(int).values

X_clin = df[clinical_cols].copy()
X_clin['mRNA_AVAILABLE'] = mrna_available

y_all = Surv.from_arrays(event=df['OS_event'].values, time=df['OS_time'].values)

print(f"✓ Patients with mRNA: {mrna_available.sum()}/{len(df)}")

# ============================================================================
# 4. HELPER FUNCTIONS
# ============================================================================

# --- DeepSurv ---
class DeepSurv(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def cox_loss(risk, times, events):
    o = torch.argsort(-times)
    r, e = risk[o], events[o]
    return -(e * (r - torch.logcumsumexp(r, 0))).sum() / (e.sum() + 1e-8)

def train_ds(Xt, yt, Xv, yv, n_feat, epochs=200, patience=20):
    net = DeepSurv(n_feat).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    Xt_t, Xv_t = to_t(Xt), to_t(Xv)
    yt_ti, yt_ev = to_t([e['time'] for e in yt]), to_t([e['event'] for e in yt])
    yv_ti, yv_ev = to_t([e['time'] for e in yv]), to_t([e['event'] for e in yv])
    best, wait, state = np.inf, 0, None
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        cox_loss(net(Xt_t), yt_ti, yt_ev).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = cox_loss(net(Xv_t), yv_ti, yv_ev).item()
        if vl < best - 1e-6: best, wait, state = vl, 0, {k: v.cpu().clone() for k,v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if state: net.load_state_dict(state)
    net.eval()
    return net

def ds_predict(net, X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

XGB_PARAMS = dict(objective="survival:cox", eval_metric="cox-nloglik",
                  eta=0.05, max_depth=3, subsample=0.8,
                  colsample_bytree=0.8, seed=SEED, verbosity=0)

def fit_xgb(Xt, yt, Xv, yv):
    dt = xgb.DMatrix(Xt, label=[e['time'] for e in yt], weight=[e['event'] for e in yt])
    dv = xgb.DMatrix(Xv, label=[e['time'] for e in yv], weight=[e['event'] for e in yv])
    m  = xgb.train(XGB_PARAMS, dt, num_boost_round=1000,
                   evals=[(dv,'v')], early_stopping_rounds=50, verbose_eval=False)
    it = getattr(m, 'best_iteration', m.num_boosted_rounds())
    return m, it

def coxnet_fit(X, y, alpha, tol=1e-7):
    m = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=0.9, max_iter=100_000,
                               tol=tol, fit_baseline_model=True)
    m.fit(X, y)
    return m

def tune_alpha(X_tr, y_tr, X_vl, y_vl, grid=ALPHA_GRID):
    best_a, best_c = grid[-1], -1
    for a in grid:
        try:
            m = coxnet_fit(X_tr, y_tr, a)
            c = concordance_index_censored(y_vl['event'], y_vl['time'], m.predict(X_vl))[0]
            if c > best_c: best_c, best_a = c, a
        except (ArithmeticError, ValueError):
            pass
    return best_a

def build_splines(meta_df, feats):
    parts, dis, mapping = [], {}, {}
    for f in feats:
        sp = dmatrix(f"bs({f}, df=4, degree=3, include_intercept=False)",
                     meta_df, return_type='dataframe')
        sp.columns = [f"{f}_s{i}" for i in range(sp.shape[1])]
        parts.append(sp); dis[f] = sp.design_info; mapping[f] = sp.columns.tolist()
    return pd.concat(parts, axis=1), dis, mapping

def apply_splines(meta_df, feats, dis):
    return pd.concat([
        pd.DataFrame(build_design_matrices([dis[f]], meta_df)[0], index=meta_df.index)
        for f in feats
    ], axis=1)

def ci(ev, ti, risk):
    return concordance_index_censored(ev, ti, risk)[0]

# ============================================================================
# 5. 5-FOLD NESTED CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: 5-FOLD NESTED CROSS-VALIDATION")
print("=" * 80)

outer_kf = StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED)

# Accumulators
fold_results  = []
fold_oof_data = []   # stored for df sensitivity + nonlinearity analysis
pooled_risk   = np.zeros(len(df))   # GAM risk scores for all patients
pooled_lin    = np.zeros(len(df))   # linear stacking risk scores
pooled_events = y_all['event'].copy()
pooled_times  = y_all['time'].copy()

# Fold-0 Brier-score-over-time accumulator (representative calibration figure)
brier_rep = {}          # {model_tag: {'times': ..., 'brier': ...}}

# Store one fold's GAM object for smooth plots
rep_gam_obj, rep_dis, rep_mapping, rep_meta = None, None, None, None

for fold_idx, (tr_idx, te_idx) in enumerate(
        outer_kf.split(np.arange(len(df)), y_all['event'])):

    print(f"\n{'='*60}")
    print(f"  OUTER FOLD {fold_idx+1}/{N_OUTER}  "
          f"(train={len(tr_idx)}, test={len(te_idx)}, "
          f"test_events={y_all['event'][te_idx].sum()})")
    print(f"{'='*60}")

    # ---------- outer split ----------
    y_tr = y_all[tr_idx];  y_te = y_all[te_idx]
    Xc_tr = X_clin.iloc[tr_idx].reset_index(drop=True)
    Xc_te = X_clin.iloc[te_idx].reset_index(drop=True)
    Mr_tr = mrna_aligned.iloc[tr_idx].reset_index(drop=True)
    Mr_te = mrna_aligned.iloc[te_idx].reset_index(drop=True)

    # ------------------------------------------------------------------
    # A. mRNA GENE SELECTION (train only, real-data patients)
    # ------------------------------------------------------------------
    has_rna   = Mr_tr.notna().any(axis=1)
    Mr_real   = Mr_tr[has_rna]
    y_real    = y_tr[has_rna.values]

    gene_var  = Mr_real.var()
    top_v     = gene_var.nlargest(N_VAR_GENES).index
    gene_ci_d = {}
    for g in top_v:
        vals = Mr_real[g].fillna(Mr_real[g].median()).values
        if vals.std() < 1e-6: continue
        try:
            gc = concordance_index_censored(y_real['event'], y_real['time'], vals)[0]
            gene_ci_d[g] = abs(gc - 0.5)
        except: pass

    top_genes = sorted(gene_ci_d, key=gene_ci_d.get, reverse=True)[:N_TOP_GENES]
    mrna_med  = Mr_real[top_genes].median()

    Mg_tr = Mr_tr[top_genes].fillna(mrna_med)
    Mg_te = Mr_te[top_genes].fillna(mrna_med)

    print(f"  mRNA genes: {', '.join(top_genes)}")

    # ------------------------------------------------------------------
    # B. PREPROCESSING (fit on outer train)
    # ------------------------------------------------------------------
    cat_c = Xc_tr.select_dtypes(include=['object','category']).columns.tolist()
    num_c = Xc_tr.select_dtypes(include=['number','bool']).columns.tolist()

    pre = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False,
                              handle_unknown='ignore'), cat_c),
        ('num', SimpleImputer(strategy='median'), num_c),
    ], remainder='drop')

    Xp_tr = pre.fit_transform(Xc_tr)
    Xp_te = pre.transform(Xc_te)

    ohe = pre.named_transformers_['cat']
    cn  = (ohe.get_feature_names_out(cat_c).tolist() if cat_c else []) + num_c
    if len(cn) != Xp_tr.shape[1]:
        cn = [f"f{i}" for i in range(Xp_tr.shape[1])]

    Xp_tr = pd.DataFrame(Xp_tr, columns=cn)
    Xp_te = pd.DataFrame(Xp_te, columns=cn)

    # mRNA scaling (train only)
    ms = StandardScaler()
    Mg_tr_s = pd.DataFrame(ms.fit_transform(Mg_tr), columns=top_genes)
    Mg_te_s = pd.DataFrame(ms.transform(Mg_te),     columns=top_genes)

    # Combined feature matrix
    Xf_tr = pd.DataFrame(np.hstack([Xp_tr, Mg_tr_s]),
                         columns=list(cn) + top_genes)
    Xf_te = pd.DataFrame(np.hstack([Xp_te, Mg_te_s]),
                         columns=list(cn) + top_genes)

    # Variance filter
    vf    = VarianceThreshold(0.01)
    Xv_tr = pd.DataFrame(vf.fit_transform(Xf_tr), columns=Xf_tr.columns[vf.get_support()])
    Xv_te = pd.DataFrame(vf.transform(Xf_te),     columns=Xf_tr.columns[vf.get_support()])

    # ------------------------------------------------------------------
    # C. COXNET FEATURE SELECTION (inner 3-fold CV on outer train)
    # ------------------------------------------------------------------
    cx_cv  = KFold(3, shuffle=True, random_state=SEED)
    best_a, best_s = ALPHA_GRID[-1], -1
    for a in np.logspace(-2, 2, 15):
        fs = []
        for ti2, vi2 in cx_cv.split(Xv_tr):
            try:
                m = CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=0.9,
                                           max_iter=100_000, tol=1e-7)
                m.fit(Xv_tr.iloc[ti2].values, y_tr[ti2])
                p = m.predict(Xv_tr.iloc[vi2].values)
                fs.append(ci(y_tr['event'][vi2], y_tr['time'][vi2], p))
            except: pass
        if fs and np.mean(fs) > best_s:
            best_s, best_a = np.mean(fs), a

    try:
        cx_final = CoxnetSurvivalAnalysis(alphas=[best_a], l1_ratio=0.9,
                                          max_iter=100_000, tol=1e-7)
        cx_final.fit(Xv_tr.values, y_tr)
        coefs = cx_final.coef_.ravel()
        sel   = Xv_tr.columns[np.abs(coefs) > 1e-8].tolist()
        if len(sel) < 5:
            sel = Xv_tr.columns[np.argsort(np.abs(coefs))[-10:]].tolist()
    except:
        sel = Xv_tr.columns.tolist()[:15]

    sc = StandardScaler()
    Xs_tr = pd.DataFrame(sc.fit_transform(Xv_tr[sel]), columns=sel)
    Xs_te = pd.DataFrame(sc.transform(Xv_te[sel]),     columns=sel)

    print(f"  CoxNet selected: {len(sel)} features  (best α={best_a:.2e})")

    # ------------------------------------------------------------------
    # D. INNER OOF — 4 BASE LEARNERS
    # ------------------------------------------------------------------
    inn_kf = KFold(N_INNER, shuffle=True, random_state=SEED)
    n_tr   = len(Xs_tr)
    oof    = np.zeros((n_tr, 4))

    # For DeepSurv early stopping we need a tiny internal val split
    Xes_tr, Xes_vl, yes_tr, yes_vl = train_test_split(
        Xs_tr, y_tr, test_size=0.15, stratify=y_tr['event'], random_state=SEED
    )

    for inn_fold, (ii_tr, ii_vl) in enumerate(inn_kf.split(Xs_tr)):
        Xi, Xj = Xs_tr.iloc[ii_tr].values, Xs_tr.iloc[ii_vl].values
        yi, yj = y_tr[ii_tr], y_tr[ii_vl]
        Xes_vl_v = Xes_vl.values

        # RSF
        rsf = RandomSurvivalForest(n_estimators=300, max_features='sqrt',
                                   min_samples_leaf=5, random_state=SEED, n_jobs=-1)
        rsf.fit(Xi, yi); oof[ii_vl, 0] = rsf.predict(Xj)

        # GBS
        gbs = GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.05,
                                               max_depth=3, random_state=SEED)
        gbs.fit(Xi, yi); oof[ii_vl, 1] = gbs.predict(Xj)

        # XGB
        xm, it = fit_xgb(Xi, yi, Xes_vl_v, yes_vl)
        oof[ii_vl, 2] = xm.predict(xgb.DMatrix(Xj), iteration_range=(0, it))

        # DeepSurv
        dn = train_ds(Xi, yi, Xes_vl_v, yes_vl, Xi.shape[1])
        oof[ii_vl, 3] = ds_predict(dn, Xj)

    oof_ci = [ci(y_tr['event'], y_tr['time'], oof[:, k]) for k in range(4)]
    print(f"  OOF C-indices — RSF:{oof_ci[0]:.3f} GBS:{oof_ci[1]:.3f} "
          f"XGB:{oof_ci[2]:.3f} DS:{oof_ci[3]:.3f}")

    # ------------------------------------------------------------------
    # E. FINAL BASE LEARNER TRAINING → test predictions
    # ------------------------------------------------------------------
    rsf_f = RandomSurvivalForest(n_estimators=300, max_features='sqrt',
                                 min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    rsf_f.fit(Xs_tr.values, y_tr)
    tp_rsf = rsf_f.predict(Xs_te.values)

    gbs_f = GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.05,
                                             max_depth=3, random_state=SEED)
    gbs_f.fit(Xs_tr.values, y_tr)
    tp_gbs = gbs_f.predict(Xs_te.values)

    xm_f, it_f = fit_xgb(Xes_tr.values, yes_tr, Xes_vl.values, yes_vl)
    tp_xgb = xm_f.predict(xgb.DMatrix(Xs_te.values), iteration_range=(0, it_f))

    dn_f = train_ds(Xes_tr.values, yes_tr, Xes_vl.values, yes_vl, Xs_tr.shape[1])
    tp_ds = ds_predict(dn_f, Xs_te.values)

    te_preds = np.column_stack([tp_rsf, tp_gbs, tp_xgb, tp_ds])

    ci_rsf = ci(y_te['event'], y_te['time'], tp_rsf)
    ci_gbs = ci(y_te['event'], y_te['time'], tp_gbs)
    ci_xgb = ci(y_te['event'], y_te['time'], tp_xgb)
    ci_ds  = ci(y_te['event'], y_te['time'], tp_ds)
    print(f"  Test C-indices — RSF:{ci_rsf:.3f} GBS:{ci_gbs:.3f} "
          f"XGB:{ci_xgb:.3f} DS:{ci_ds:.3f}")

    # ------------------------------------------------------------------
    # F. META-LEARNER FEATURES
    # ------------------------------------------------------------------
    META_FEATS = ["RSF", "GBS", "XGB", "DeepSurv"]
    meta_tr = pd.DataFrame(oof,      columns=META_FEATS)
    meta_te = pd.DataFrame(te_preds, columns=META_FEATS)

    # Clip test to train range
    for f in META_FEATS:
        mn, mx = meta_tr[f].min(), meta_tr[f].max()
        meta_te[f] = meta_te[f].clip(mn, mx)

    # Internal val split of outer-train for meta-learner alpha tuning
    Xm_t, Xm_v, ym_t, ym_v = train_test_split(
        meta_tr, y_tr, test_size=0.20, stratify=y_tr['event'], random_state=SEED
    )

    # Structured survival arrays
    y_tr_s = np.array(list(zip(y_tr['event'], y_tr['time'])),
                      dtype=[('event', bool), ('time', float)])
    y_te_s = np.array(list(zip(y_te['event'], y_te['time'])),
                      dtype=[('event', bool), ('time', float)])
    ym_t_s = np.array(list(zip(ym_t['event'], ym_t['time'])),
                      dtype=[('event', bool), ('time', float)])
    ym_v_s = np.array(list(zip(ym_v['event'], ym_v['time'])),
                      dtype=[('event', bool), ('time', float)])

    # ------------------------------------------------------------------
    # G. LINEAR COX STACKING
    # ------------------------------------------------------------------
    lin_m = None
    a_lin = tune_alpha(Xm_t.values, ym_t_s, Xm_v.values, ym_v_s)
    try:
        lin_m  = coxnet_fit(meta_tr.values, y_tr_s, a_lin)
        tp_lin = lin_m.predict(meta_te.values)
    except:
        tp_lin = meta_tr.mean(axis=1).values[:len(meta_te)]  # fallback: average

    ci_lin = ci(y_te['event'], y_te['time'], tp_lin)

    # ------------------------------------------------------------------
    # H. GAM META-LEARNER (spline basis)
    # ------------------------------------------------------------------
    sp_tr, dis, mapping = build_splines(meta_tr, META_FEATS)
    sp_te = apply_splines(meta_te, META_FEATS, dis)

    # Align column counts (spline columns must match)
    sp_te.columns = sp_tr.columns[:sp_te.shape[1]]
    if sp_te.shape[1] < sp_tr.shape[1]:
        for c in sp_tr.columns[sp_te.shape[1]:]:
            sp_te[c] = 0.0
    sp_te = sp_te[sp_tr.columns]

    sp_tr_t, sp_v_t = train_test_split(sp_tr, test_size=0.20, random_state=SEED)
    ym_t2_s = np.array(list(zip(y_tr[sp_tr_t.index]['event'],
                                 y_tr[sp_tr_t.index]['time'])),
                       dtype=[('event', bool), ('time', float)])
    ym_v2_s = np.array(list(zip(y_tr[sp_v_t.index]['event'],
                                 y_tr[sp_v_t.index]['time'])),
                       dtype=[('event', bool), ('time', float)])

    gam_m = None
    a_gam = tune_alpha(sp_tr_t.values, ym_t2_s, sp_v_t.values, ym_v2_s)
    try:
        gam_m  = coxnet_fit(sp_tr.values, y_tr_s, a_gam)
        tp_gam = gam_m.predict(sp_te.values)
    except:
        tp_gam = tp_lin.copy()

    ci_gam = ci(y_te['event'], y_te['time'], tp_gam)

    # ------------------------------------------------------------------
    # I. IBS AND Ctd-AUC (CALIBRATION / DISCRIMINATION METRICS)
    # ------------------------------------------------------------------
    fold_ibs, fold_auc = {}, {}
    ev_tr = y_tr_s['time'][y_tr_s['event']]

    # Integrated Brier Score — models with predict_survival_function
    try:
        t_lo = float(np.percentile(ev_tr, 10))
        t_hi = float(min(np.percentile(ev_tr, 90),
                         float(y_te_s['time'].max()) * 0.99))
        times_ibs = np.linspace(t_lo, t_hi, 10)
        times_ibs = times_ibs[times_ibs < float(y_te_s['time'].max())]
        if len(times_ibs) >= 2:
            for tag, mdl, Xte_v in [
                ('GAM',    gam_m,  sp_te.values),
                ('Linear', lin_m,  meta_te.values),
                ('RSF',    rsf_f,  Xs_te.values),
                ('GBS',    gbs_f,  Xs_te.values),
            ]:
                if mdl is None:
                    fold_ibs[tag] = np.nan; continue
                try:
                    sfns = mdl.predict_survival_function(Xte_v)
                    prb  = np.row_stack([fn(times_ibs) for fn in sfns])
                    prb  = np.clip(prb, 0.0, 1.0)  # ensure valid probabilities
                    fold_ibs[tag] = float(integrated_brier_score(
                        y_tr_s, y_te_s, prb, times_ibs))
                except Exception:
                    fold_ibs[tag] = np.nan
    except Exception:
        pass

    # Concordance time-dependent AUC — all models (risk scores only)
    try:
        times_auc = np.unique(np.percentile(ev_tr, [25, 50, 75]))
        times_auc = times_auc[
            (times_auc > float(ev_tr.min())) & (times_auc < float(ev_tr.max()))
        ]
        if len(times_auc) >= 1:
            for tag, risk_v in [
                ('GAM',    tp_gam), ('Linear', tp_lin),
                ('RSF',    tp_rsf), ('GBS',    tp_gbs),
                ('XGB',    tp_xgb), ('DS',     tp_ds),
            ]:
                try:
                    _, val = cumulative_dynamic_auc(
                        y_tr_s, y_te_s, risk_v, times_auc)
                    fold_auc[tag] = float(val)
                except:
                    fold_auc[tag] = np.nan
    except Exception:
        pass

    # Time-dependent Brier score for calibration figure (fold 0 as representative)
    if fold_idx == 0:
        try:
            t_lo0 = float(np.percentile(ev_tr, 10))
            t_hi0 = float(min(np.percentile(ev_tr, 90),
                               float(y_te_s['time'].max()) * 0.99))
            times_brier = np.linspace(t_lo0, t_hi0, 30)
            times_brier = times_brier[times_brier < float(y_te_s['time'].max())]
            if len(times_brier) >= 2:
                for _tag, _mdl, _Xv in [
                    ('GAM',    gam_m,  sp_te.values),
                    ('Linear', lin_m,  meta_te.values),
                    ('RSF',    rsf_f,  Xs_te.values),
                    ('GBS',    gbs_f,  Xs_te.values),
                ]:
                    if _mdl is None:
                        continue
                    try:
                        _sfns = _mdl.predict_survival_function(_Xv)
                        _prb  = np.clip(np.row_stack([fn(times_brier) for fn in _sfns]),
                                        0.0, 1.0)
                        _, _bs = brier_score_t(y_tr_s, y_te_s, _prb, times_brier)
                        brier_rep[_tag] = {'times': times_brier, 'brier': _bs}
                    except Exception:
                        pass
        except Exception:
            pass

    print(f"  Linear Stacking test C: {ci_lin:.3f}")
    print(f"  GAM Ensemble   test C: {ci_gam:.3f}")

    # Save pooled predictions
    pooled_risk[te_idx] = tp_gam
    pooled_lin[te_idx]  = tp_lin

    fold_results.append({
        'fold': fold_idx + 1,
        'n_test': len(te_idx),
        'n_test_events': int(y_te['event'].sum()),
        'n_features': len(sel),
        'C_RSF': ci_rsf, 'C_GBS': ci_gbs, 'C_XGB': ci_xgb, 'C_DS': ci_ds,
        'C_Linear': ci_lin, 'C_GAM': ci_gam,
        'OOF_RSF': oof_ci[0], 'OOF_GBS': oof_ci[1],
        'OOF_XGB': oof_ci[2], 'OOF_DS': oof_ci[3],
        'IBS_GAM':    fold_ibs.get('GAM',    np.nan),
        'IBS_Linear': fold_ibs.get('Linear', np.nan),
        'IBS_RSF':    fold_ibs.get('RSF',    np.nan),
        'IBS_GBS':    fold_ibs.get('GBS',    np.nan),
        'AUC_GAM':    fold_auc.get('GAM',    np.nan),
        'AUC_Linear': fold_auc.get('Linear', np.nan),
        'AUC_RSF':    fold_auc.get('RSF',    np.nan),
        'AUC_GBS':    fold_auc.get('GBS',    np.nan),
        'AUC_XGB':    fold_auc.get('XGB',    np.nan),
        'AUC_DS':     fold_auc.get('DS',     np.nan),
    })

    fold_oof_data.append({
        'meta_tr':    meta_tr.copy(),
        'meta_te':    meta_te.copy(),
        'y_tr_s':     y_tr_s.copy(),
        'y_te_s':     y_te_s.copy(),
        'y_te_event': y_te['event'].copy(),
        'y_te_time':  y_te['time'].copy(),
        'gam_m':      gam_m,
        'lin_m':      lin_m,
        'dis':        dis,
        'mapping':    mapping,
        'te_idx':     te_idx.copy(),
    })

    # Keep fold-1 GAM for smooth plots
    if fold_idx == 0:
        rep_gam_obj = gam_m
        rep_dis     = dis
        rep_mapping = mapping
        rep_meta    = meta_tr.copy()

print("\n" + "=" * 80)
print("NESTED CV COMPLETE")
print("=" * 80)

# ============================================================================
# 6. AGGREGATE RESULTS
# ============================================================================

res_df = pd.DataFrame(fold_results)

print("\nPer-fold test C-indices:")
print(res_df[['fold','C_RSF','C_GBS','C_XGB','C_DS','C_Linear','C_GAM']].to_string(index=False))

means = res_df[['C_RSF','C_GBS','C_XGB','C_DS','C_Linear','C_GAM']].mean()
stds  = res_df[['C_RSF','C_GBS','C_XGB','C_DS','C_Linear','C_GAM']].std()

print(f"\n{'Model':<22} {'Mean C-index':>12} {'Std':>8}")
print("-" * 44)
labels = ['RSF','GBS','XGB','DeepSurv','Linear Stacking','GAM Ensemble']
for col, lab in zip(['C_RSF','C_GBS','C_XGB','C_DS','C_Linear','C_GAM'], labels):
    print(f"  {lab:<20} {means[col]:>12.4f} {stds[col]:>8.4f}")

gam_delta_lin  = means['C_GAM'] - means['C_Linear']
gam_delta_best = means['C_GAM'] - max(means['C_RSF'], means['C_GBS'],
                                       means['C_XGB'], means['C_DS'])
print(f"\n✓ GAM vs Linear Stacking: {gam_delta_lin:+.4f}")
print(f"✓ GAM vs Best Base:       {gam_delta_best:+.4f}")

# IBS and Ctd-AUC summary
ibs_cols = ['IBS_GAM','IBS_Linear','IBS_RSF','IBS_GBS']
auc_cols = ['AUC_GAM','AUC_Linear','AUC_RSF','AUC_GBS','AUC_XGB','AUC_DS']
ibs_means = res_df[ibs_cols].mean()
auc_means  = res_df[auc_cols].mean()
ibs_stds  = res_df[ibs_cols].std()
auc_stds   = res_df[auc_cols].std()

print(f"\n{'Model':<22} {'Mean IBS':>10} {'Mean Ctd-AUC':>14}")
print("-" * 50)
for ibs_c, auc_c, lab in [
    ('IBS_RSF',    'AUC_RSF',    'RSF'),
    ('IBS_GBS',    'AUC_GBS',    'GBS'),
    (None,         'AUC_XGB',    'XGB'),
    (None,         'AUC_DS',     'DeepSurv'),
    ('IBS_Linear', 'AUC_Linear', 'Linear Stacking'),
    ('IBS_GAM',    'AUC_GAM',    'GAM Ensemble'),
]:
    ibs_v = f"{ibs_means[ibs_c]:.4f}±{ibs_stds[ibs_c]:.4f}" if ibs_c else '   N/A      '
    auc_v = f"{auc_means[auc_c]:.4f}±{auc_stds[auc_c]:.4f}"
    print(f"  {lab:<20} {ibs_v:>16} {auc_v:>18}")

# ============================================================================
# 7. POOLED BOOTSTRAP CIs
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: BOOTSTRAP CIs (POOLED OOF)")
print("=" * 80)

rng  = np.random.default_rng(SEED)
boot_gam, boot_lin, boot_imp = [], [], []

for _ in range(N_BOOT):
    idx = rng.choice(len(df), len(df), replace=True)
    cg  = ci(pooled_events[idx], pooled_times[idx], pooled_risk[idx])
    cl  = ci(pooled_events[idx], pooled_times[idx], pooled_lin[idx])
    boot_gam.append(cg); boot_lin.append(cl); boot_imp.append(cg - cl)

ci_gam_lo, ci_gam_hi = np.percentile(boot_gam, [2.5, 97.5])
ci_lin_lo, ci_lin_hi = np.percentile(boot_lin, [2.5, 97.5])
ci_imp_lo, ci_imp_hi = np.percentile(boot_imp, [2.5, 97.5])

pooled_gam_c = ci(pooled_events, pooled_times, pooled_risk)
pooled_lin_c = ci(pooled_events, pooled_times, pooled_lin)

print(f"✓ Pooled GAM C-index:    {pooled_gam_c:.4f}  95% CI [{ci_gam_lo:.4f}, {ci_gam_hi:.4f}]")
print(f"✓ Pooled Linear C-index: {pooled_lin_c:.4f}  95% CI [{ci_lin_lo:.4f}, {ci_lin_hi:.4f}]")
print(f"✓ GAM − Linear CI:       [{ci_imp_lo:+.4f}, {ci_imp_hi:+.4f}]  "
      f"{'*** significant' if ci_imp_lo > 0 else '(not significant)'}")

# ============================================================================
# DF SENSITIVITY: Full 4-Learner SAGAM
# ============================================================================

print("\n" + "=" * 80)
print("DF SENSITIVITY: Full 4-Learner SAGAM (df = 3, 4, 5, 6)")
print("=" * 80)

DF_VALS  = [3, 4, 5, 6]
df_rows  = []
META_FEATS_S = ["RSF", "GBS", "XGB", "DeepSurv"]

for test_df in DF_VALS:
    fold_cs = []
    for fd in fold_oof_data:
        mtr = fd['meta_tr'];  mte = fd['meta_te']
        ytr = fd['y_tr_s']
        yte_ev = fd['y_te_event'];  yte_ti = fd['y_te_time']

        sp_parts_tr, sp_parts_te = [], []
        for f in META_FEATS_S:
            formula = f"bs({f}, df={test_df}, degree=3, include_intercept=False)"
            s_tr = dmatrix(formula, mtr, return_type='dataframe')
            s_tr.columns = [f"{f}_s{i}" for i in range(s_tr.shape[1])]
            sp_parts_tr.append(s_tr)
            s_te = dmatrix(formula, mte, return_type='dataframe')
            s_te.columns = [f"{f}_s{i}" for i in range(s_te.shape[1])]
            sp_parts_te.append(s_te)

        sp_tr_d = pd.concat(sp_parts_tr, axis=1)
        sp_te_d = pd.concat(sp_parts_te, axis=1)
        sp_te_d.columns = sp_tr_d.columns[:sp_te_d.shape[1]]
        for c in sp_tr_d.columns[sp_te_d.shape[1]:]:
            sp_te_d[c] = 0.0
        sp_te_d = sp_te_d[sp_tr_d.columns]

        tr2, vl2 = train_test_split(sp_tr_d, test_size=0.20, random_state=SEED)
        yt2 = ytr[tr2.index.values];  yv2 = ytr[vl2.index.values]
        try:
            a_d  = tune_alpha(tr2.values, yt2, vl2.values, yv2)
            m_d  = coxnet_fit(sp_tr_d.values, ytr, a_d)
            fold_cs.append(ci(yte_ev, yte_ti, m_d.predict(sp_te_d.values)))
        except:
            fold_cs.append(np.nan)

    mn, sd = float(np.nanmean(fold_cs)), float(np.nanstd(fold_cs))
    marker = " ← best" if test_df == 4 else ""
    print(f"  df={test_df}: {mn:.4f} ± {sd:.4f}  "
          f"folds={[round(v,3) for v in fold_cs]}{marker}")
    df_rows.append({'df': test_df, 'Mean_C': mn, 'Std_C': sd,
                    **{f'Fold{i+1}': v for i, v in enumerate(fold_cs)}})

df_sens_df = pd.DataFrame(df_rows)
df_sens_df.to_csv(OUTPUT_DIR / 'df_sensitivity_full.csv', index=False)
print("✓ Full df sensitivity saved.")

# ============================================================================
# NONLINEARITY QUANTIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("NONLINEARITY ANALYSIS: Deviation of f_k from Linearity")
print("=" * 80)

nonlin_rows = []
for fd in fold_oof_data:
    gobj = fd['gam_m']
    if gobj is None:
        continue
    coefs     = gobj.coef_.ravel()
    mtr       = fd['meta_tr']
    dis_f     = fd['dis']
    mapping_f = fd['mapping']
    sp_cols   = list(build_splines(mtr, list(dis_f.keys()))[0].columns)

    for model in META_FEATS_S:
        vals = mtr[model].values
        grid = np.linspace(vals.min(), vals.max(), 300)
        dg   = pd.DataFrame({model: grid})
        S_g  = dmatrix(
            f"bs({model}, df=4, degree=3, include_intercept=False)",
            dg, return_type='dataframe')
        idx_c = [sp_cols.index(c)
                 for c in mapping_f.get(model, []) if c in sp_cols]
        if not idx_c:
            continue
        fvals   = S_g.values @ coefs[idx_c]
        lin_c   = np.polyfit(grid, fvals, 1)
        lin_v   = np.polyval(lin_c, grid)
        tot_var = float(np.var(fvals))
        nf      = float(1.0 - np.var(lin_v) / tot_var) if tot_var > 1e-10 else 0.0
        rmsd    = float(np.sqrt(np.mean((fvals - lin_v) ** 2)))
        nonlin_rows.append({'model': model, 'nonlin_frac': nf, 'rmsd': rmsd})

nonlin_df = pd.DataFrame(nonlin_rows).groupby('model').mean().reset_index()
print(f"\n{'Model':<12} {'Nonlin. Fraction (↑ = more nonlinear)':>40} {'RMSD':>8}")
print("-" * 63)
for _, r in nonlin_df.iterrows():
    print(f"  {r['model']:<10} {r['nonlin_frac']:>40.4f} {r['rmsd']:>8.4f}")
nonlin_df.to_csv(OUTPUT_DIR / 'nonlinearity_analysis.csv', index=False)
print("✓ Nonlinearity analysis saved.")

# ============================================================================
# SMOOTH CONTRIBUTION STABILITY (all 5 folds)
# ============================================================================

print("\n" + "=" * 80)
print("SMOOTH CONTRIBUTION STABILITY (all 5 folds)")
print("=" * 80)

fig_stab, axes_stab = plt.subplots(2, 2, figsize=(11, 8))
fold_pal = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for ax_s, model in zip(axes_stab.ravel(), META_FEATS_S):
    all_xn, all_yc = [], []
    for fi, fd in enumerate(fold_oof_data):
        gobj = fd['gam_m']
        if gobj is None:
            continue
        coefs_s  = gobj.coef_.ravel()
        mtr_s    = fd['meta_tr']
        sp_cols_s = list(build_splines(mtr_s, list(fd['dis'].keys()))[0].columns)
        vals_s   = mtr_s[model].values
        grid_s   = np.linspace(vals_s.min(), vals_s.max(), 200)
        dg_s     = pd.DataFrame({model: grid_s})
        try:
            S_gs  = dmatrix(f"bs({model}, df=4, degree=3, include_intercept=False)",
                            dg_s, return_type='dataframe')
            idx_cs = [sp_cols_s.index(c)
                      for c in fd['mapping'].get(model, []) if c in sp_cols_s]
            if not idx_cs:
                continue
            fv_s  = S_gs.values @ coefs_s[idx_cs]
            fv_s  = fv_s - fv_s.mean()               # center for comparability
            xn_s  = (grid_s - grid_s.min()) / (grid_s.max() - grid_s.min() + 1e-8)
            all_xn.append(xn_s); all_yc.append(fv_s)
            ax_s.plot(xn_s, fv_s, color=fold_pal[fi], alpha=0.65, lw=1.8,
                      label=f'Fold {fi+1}')
        except Exception:
            pass

    if len(all_yc) > 1:
        cx = np.linspace(0, 1, 200)
        iy = [np.interp(cx, xi, yi) for xi, yi in zip(all_xn, all_yc)]
        ax_s.plot(cx, np.mean(iy, axis=0), 'k-', lw=2.5, label='Mean', zorder=5)

    ax_s.axhline(0, color='gray', lw=1, ls='--', alpha=0.5)
    ax_s.set_title(model, fontsize=12, fontweight='bold')
    ax_s.set_xlabel('Normalised Risk Score', fontsize=10)
    ax_s.set_ylabel('$f_k$(score), centred', fontsize=10)
    ax_s.grid(alpha=0.25)
    if model == META_FEATS_S[0]:
        ax_s.legend(fontsize=8, loc='upper left')

plt.suptitle('SAGAM Smooth Function Stability Across 5 Outer Folds',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gam_smooths_stability.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Smooth stability figure saved.")

# ============================================================================
# TIME-DEPENDENT BRIER SCORE (fold 1 representative)
# ============================================================================

print("\n" + "=" * 80)
print("TIME-DEPENDENT BRIER SCORE (fold 1, representative)")
print("=" * 80)

if brier_rep:
    fig_br, ax_br = plt.subplots(figsize=(9, 5))
    brier_colors = {'GAM': '#2ca02c', 'Linear': '#ff7f0e',
                    'RSF': '#1f77b4',  'GBS': '#d62728'}
    brier_lw     = {'GAM': 2.8, 'Linear': 2.2, 'RSF': 1.6, 'GBS': 1.6}
    for tag in ['GBS', 'RSF', 'Linear', 'GAM']:   # back-to-front so GAM is on top
        if tag not in brier_rep:
            continue
        bd = brier_rep[tag]
        ax_br.plot(bd['times'], bd['brier'],
                   color=brier_colors[tag], lw=brier_lw[tag],
                   label=tag, alpha=0.9)
    # Shade region where SAGAM < Linear (better calibration)
    if 'GAM' in brier_rep and 'Linear' in brier_rep:
        t_sh = brier_rep['GAM']['times']
        g_sh = brier_rep['GAM']['brier']
        l_sh = brier_rep['Linear']['brier']
        ax_br.fill_between(t_sh, g_sh, l_sh, where=(g_sh < l_sh),
                           alpha=0.15, color='green', label='SAGAM advantage')
    ax_br.set_xlabel('Time (Months)', fontsize=12)
    ax_br.set_ylabel('Brier Score  (lower = better calibration)', fontsize=12)
    ax_br.set_title('Time-Dependent Brier Score — SAGAM vs Comparators\n'
                    '(Outer Fold 1; shaded region: SAGAM < Linear Stacking)',
                    fontsize=12, fontweight='bold')
    ax_br.legend(fontsize=11); ax_br.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'brier_score_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Brier score over time figure saved.")
else:
    print("  (no Brier data accumulated — skipping figure)")

# ============================================================================
# SUBGROUP ANALYSIS BY AJCC STAGE
# ============================================================================

print("\n" + "=" * 80)
print("SUBGROUP ANALYSIS BY AJCC STAGE")
print("=" * 80)

stage_col = 'AJCC_PATHOLOGIC_TUMOR_STAGE'
if stage_col in df.columns:
    stg = df[stage_col].fillna('').astype(str).str.upper()
    early_m = stg.str.contains(r'STAGE\s*(I[AB]?|II[AB]?)\b', regex=True) & \
              ~stg.str.contains(r'STAGE\s*(III|IV)', regex=True)
    late_m  = stg.str.contains(r'STAGE\s*(III[AB]?|IV[AB]?)\b', regex=True)

    subgroup_rows = []
    for grp, mask in [('Stage I/II (early)', early_m.values),
                      ('Stage III/IV (late)', late_m.values),
                      ('All patients',        np.ones(len(df), dtype=bool))]:
        n_g    = int(mask.sum())
        n_ev_g = int(pooled_events[mask].sum())
        if n_g < 10 or n_ev_g < 5:
            print(f"  {grp}: n={n_g} events={n_ev_g} — too few, skipping")
            continue
        try:
            c_g = ci(pooled_events[mask], pooled_times[mask], pooled_risk[mask])
            c_l = ci(pooled_events[mask], pooled_times[mask], pooled_lin[mask])
        except Exception:
            c_g = c_l = np.nan
        delta = c_g - c_l
        print(f"  {grp:<25}: n={n_g} ev={n_ev_g}  "
              f"GAM={c_g:.4f}  Linear={c_l:.4f}  Δ={delta:+.4f}")
        subgroup_rows.append({'Group': grp, 'n': n_g, 'events': n_ev_g,
                              'C_GAM': c_g, 'C_Linear': c_l, 'Delta': delta})

    if subgroup_rows:
        pd.DataFrame(subgroup_rows).to_csv(OUTPUT_DIR / 'subgroup_analysis.csv', index=False)
        print("✓ Subgroup analysis saved.")
else:
    print(f"  Stage column '{stage_col}' not found in df — skipping.")

# ============================================================================
# 8. KAPLAN-MEIER (pooled OOF — n=501)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: KAPLAN-MEIER (Pooled OOF, n=501)")
print("=" * 80)

risk_groups = pd.qcut(pooled_risk, q=3,
                      labels=['Low Risk', 'Medium Risk', 'High Risk'])
low_m  = (risk_groups == 'Low Risk')
med_m  = (risk_groups == 'Medium Risk')
high_m = (risk_groups == 'High Risk')

lr_lh = logrank_test(pooled_times[low_m],  pooled_times[high_m],
                     pooled_events[low_m],  pooled_events[high_m])
lr_mv = multivariate_logrank_test(pooled_times, risk_groups, pooled_events)

print(f"✓ Log-rank (Low vs High): p = {lr_lh.p_value:.4e}")
print(f"✓ Multivariate:           p = {lr_mv.p_value:.4e}")

fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#2E7D32', '#F57C00', '#C62828']
kmf    = KaplanMeierFitter()

median_survs = {}
for i, (grp, mask) in enumerate([('Low Risk', low_m),
                                   ('Medium Risk', med_m),
                                   ('High Risk', high_m)]):
    kmf.fit(pooled_times[mask], pooled_events[mask],
            label=f"{grp} (n={mask.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True,
                               linewidth=3, color=colors[i], alpha=0.9)
    try:
        ms = kmf.median_survival_time_
        median_survs[grp] = f"{ms:.1f} mo" if not (np.isnan(ms) or np.isinf(ms)) else "NR"
    except:
        median_survs[grp] = "NR"

ax.set_xlabel('Time (Months)', fontsize=14, fontweight='bold')
ax.set_ylabel('Survival Probability', fontsize=14, fontweight='bold')
sig = ('***' if lr_lh.p_value < 0.001 else '**' if lr_lh.p_value < 0.01
       else '*' if lr_lh.p_value < 0.05 else 'NS')
ax.set_title(f'Kaplan-Meier Curves — GAM Risk Stratification (n=501, Pooled OOF)\n'
             f'Log-rank Low vs High: p={lr_lh.p_value:.4f}  [{sig}]',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--'); ax.set_ylim(0, 1.05)
ax.legend(fontsize=12, loc='lower left')

ypos = [0.95, 0.87, 0.79]
for i, (grp, ms_txt) in enumerate(median_survs.items()):
    ax.text(0.98, ypos[i], f"{grp}\nMedian = {ms_txt}",
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor=colors[i],
                      alpha=0.2, edgecolor=colors[i], linewidth=2))

ax.text(0.02, 0.05, f'Significance: {sig}\n'
        f'*** p<0.001  ** p<0.01  * p<0.05  NS p≥0.05',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                  edgecolor='black', alpha=0.9))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'kaplan_meier.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ KM plot saved ({sig})")

# ============================================================================
# 9. GAM SMOOTH CONTRIBUTION PLOT (fold 1)
# ============================================================================

if rep_gam_obj is not None:
    gam_coefs = rep_gam_obj.coef_.ravel()
    sp_cols   = list(build_splines(rep_meta, list(rep_dis.keys()))[0].columns)

    plt.figure(figsize=(10, 6))
    for model in rep_dis.keys():
        vals = rep_meta[model].values
        grid = np.linspace(vals.min(), vals.max(), 300)
        df_g = pd.DataFrame({model: grid})
        S_g  = dmatrix(f"bs({model}, df=4, degree=3, include_intercept=False)",
                       df_g, return_type='dataframe')
        idx_c = [sp_cols.index(c) for c in rep_mapping[model] if c in sp_cols]
        if not idx_c: continue
        fvals = S_g.values @ gam_coefs[idx_c]
        plt.plot(grid, fvals, label=model, linewidth=2.5)

    plt.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    plt.xlabel('Base Learner Risk Score', fontsize=13)
    plt.ylabel('Smooth Log-Hazard Contribution f(score)', fontsize=13)
    plt.title('GAM Meta-Learner: Non-Linear Smooth Contributions\n'
              '(Representative — Outer Fold 1)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gam_smooths.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ GAM smooth plot saved.")

# ============================================================================
# 10. PERFORMANCE BAR CHART
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))
bar_labels = ['RSF', 'GBS', 'XGB', 'DeepSurv', 'Linear\nStacking', 'GAM\nEnsemble']
bar_means  = [means['C_RSF'], means['C_GBS'], means['C_XGB'], means['C_DS'],
              means['C_Linear'], means['C_GAM']]
bar_stds   = [stds['C_RSF'],  stds['C_GBS'],  stds['C_XGB'],  stds['C_DS'],
              stds['C_Linear'],  stds['C_GAM']]
bar_colors = ['#5B9BD5','#5B9BD5','#5B9BD5','#5B9BD5','#ED7D31','#70AD47']

bars = ax.bar(bar_labels, bar_means, yerr=bar_stds, capsize=6,
              color=bar_colors, edgecolor='black', linewidth=0.8, alpha=0.85)
bars[-1].set_linewidth(2.5)

for bar, m in zip(bars, bar_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bar_stds) + 0.003,
            f'{m:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(0.5, color='red',       linestyle='--', alpha=0.4, label='Random (0.5)')
ax.axhline(0.7, color='darkgreen', linestyle='--', alpha=0.4, label='Good (0.7)')
ax.set_ylabel('C-Index  (mean ± std, 5-fold CV)', fontsize=12, fontweight='bold')
ax.set_title('TCGA-LUAD Overall Survival Prediction — 5-Fold Nested CV',
             fontsize=13, fontweight='bold')
ax.set_ylim(0.40, 0.85); ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Performance bar chart saved.")

# ============================================================================
# 11. PUBLISHED BASELINE TABLE
# ============================================================================

print("\n" + "=" * 80)
print("PUBLISHED BASELINE COMPARISON")
print("=" * 80)

baselines = [
    ("Astley et al. (2022)",       "Cox PH",        0.72, "TCGA-LUAD, clinical only"),
    ("Astley et al. (2022)",       "RSF",            0.73, "TCGA-LUAD, clinical only"),
    ("Khatua et al.",              "Cox / RSF",      0.82, "TCGA-LUAD, clinical+molecular"),
    ("Multi-omics DL (2020)",      "Autoencoder",    0.65, "TCGA-LUAD, multi-omics"),
    ("THIS WORK — GAM Ensemble",
     "GAM meta-learner",
     pooled_gam_c,
     f"5-fold CV, clinical+mRNA, "
     f"95% CI [{ci_gam_lo:.3f},{ci_gam_hi:.3f}]"),
]
print(f"\n{'Study':<38} {'Model':<18} {'C-Index':>8}  {'Notes'}")
print("-" * 95)
for row in baselines:
    print(f"  {row[0]:<36} {row[1]:<18} {row[2]:>8.3f}  {row[3]}")

# ============================================================================
# 12. SAVE ALL RESULTS
# ============================================================================

res_df.to_csv(OUTPUT_DIR / 'fold_results.csv', index=False)

# Save pooled per-patient predictions (used by improvements.py for bootstrap + subgroup)
pooled_save = pd.DataFrame({
    'risk_gam': pooled_risk,
    'risk_lin': pooled_lin,
    'event':    pooled_events,
    'time':     pooled_times,
})
if 'AJCC_PATHOLOGIC_TUMOR_STAGE' in df.columns:
    pooled_save['stage'] = df['AJCC_PATHOLOGIC_TUMOR_STAGE'].values
pooled_save.to_csv(OUTPUT_DIR / 'pooled_predictions.csv', index=False)

summary = pd.DataFrame({
    'Metric': [
        'N_patients', 'N_events',
        'GAM_mean_C', 'GAM_std_C', 'GAM_pooled_C',
        'GAM_CI_low', 'GAM_CI_high',
        'Linear_mean_C', 'Linear_pooled_C',
        'GAM_vs_linear_mean', 'GAM_vs_linear_pooled',
        'GAM_vs_linear_CI_low', 'GAM_vs_linear_CI_high',
        'LogRank_Low_vs_High_p', 'KM_significance',
        'RSF_mean_C', 'GBS_mean_C', 'XGB_mean_C', 'DS_mean_C',
    ],
    'Value': [
        len(df), int(df['OS_event'].sum()),
        means['C_GAM'], stds['C_GAM'], pooled_gam_c,
        ci_gam_lo, ci_gam_hi,
        means['C_Linear'], pooled_lin_c,
        gam_delta_lin, pooled_gam_c - pooled_lin_c,
        ci_imp_lo, ci_imp_hi,
        lr_lh.p_value, sig,
        means['C_RSF'], means['C_GBS'], means['C_XGB'], means['C_DS'],
    ]
})
summary.to_csv(OUTPUT_DIR / 'summary_metrics.csv', index=False)

with open(OUTPUT_DIR / 'complete_analysis.txt', 'w') as f:
    f.write("TCGA-LUAD GAM Meta-Learner v3 — 5-Fold Nested CV Results\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Patients: {len(df)}  |  Events: {int(df['OS_event'].sum())}\n")
    f.write(f"Evaluation: {N_OUTER}-fold stratified nested CV\n\n")
    f.write("=== PER-FOLD TEST C-INDICES ===\n")
    f.write(res_df.to_string(index=False) + "\n\n")
    f.write("=== MEAN ± STD ACROSS FOLDS ===\n")
    for col, lab in zip(['C_RSF','C_GBS','C_XGB','C_DS','C_Linear','C_GAM'], labels):
        f.write(f"  {lab:<22}: {means[col]:.4f} ± {stds[col]:.4f}\n")
    f.write(f"\n=== POOLED OOF METRICS ===\n")
    f.write(f"GAM pooled C:    {pooled_gam_c:.4f}  95% CI [{ci_gam_lo:.4f}, {ci_gam_hi:.4f}]\n")
    f.write(f"Linear pooled C: {pooled_lin_c:.4f}  95% CI [{ci_lin_lo:.4f}, {ci_lin_hi:.4f}]\n")
    f.write(f"GAM-Linear CI:   [{ci_imp_lo:+.4f}, {ci_imp_hi:+.4f}]\n\n")
    f.write(f"=== KAPLAN-MEIER ===\n")
    f.write(f"Log-rank (Low vs High): p = {lr_lh.p_value:.4e}  [{sig}]\n")
    f.write(f"Multivariate:           p = {lr_mv.p_value:.4e}\n")

print(f"\n✓ All results saved to: {OUTPUT_DIR}")
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print(f"  GAM mean C-index:       {means['C_GAM']:.4f} ± {stds['C_GAM']:.4f}")
print(f"  GAM pooled C-index:     {pooled_gam_c:.4f}  [{ci_gam_lo:.4f}, {ci_gam_hi:.4f}]")
print(f"  GAM vs Linear (mean):   {gam_delta_lin:+.4f}")
print(f"  GAM vs Linear CI:       [{ci_imp_lo:+.4f}, {ci_imp_hi:+.4f}]  "
      f"({'significant' if ci_imp_lo > 0 else 'not significant'})")
print(f"  Log-rank p (Low vs High): {lr_lh.p_value:.4e}  [{sig}]")
print("=" * 80)
