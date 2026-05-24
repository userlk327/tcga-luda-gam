"""
Multi-Cohort External Validation — LUAD GEO Cohorts
====================================================
Trains SAGAM on all 501 TCGA-LUAD patients then evaluates on:
  Primary:   GSE31210  (Takeuchi et al., GPL570, 226 stage I/II pure LUAD)
  Secondary: GSE68465  (Director's Challenge, GPL96, mixed stage LUAD)

For each cohort the full pipeline runs independently (different gene overlaps
may differ between cohorts — each uses the best available gene set).
"""

from pathlib import Path
import re, warnings, random, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import GEOparse

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

import torch, torch.nn as nn, torch.optim as optim
import xgboost as xgb
from patsy import dmatrix, build_design_matrices
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / 'dataset'
OUTPUT_DIR = REPO_ROOT / 'results_v2'
GEO_DIR    = REPO_ROOT / 'geo_cache'
OUTPUT_DIR.mkdir(exist_ok=True); GEO_DIR.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_GEOS = ['GSE31210', 'GSE68465']

print("=" * 70)
print("MULTI-COHORT EXTERNAL VALIDATION — LUAD")
print(f"Cohorts: {', '.join(TARGET_GEOS)}")
print("=" * 70)

# ================================================================
# HELPERS (shared with external_validation.py)
# ================================================================

class DeepSurv(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n,64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32,1))
    def forward(self, x): return self.net(x).squeeze(-1)

def cox_loss(risk, times, events):
    o = torch.argsort(-times); r, e = risk[o], events[o]
    return -(e*(r - torch.logcumsumexp(r, 0))).sum() / (e.sum() + 1e-8)

def train_ds(Xt, yt, Xv, yv, n, epochs=200, patience=20):
    net = DeepSurv(n).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    Xt_t, Xv_t = to_t(Xt), to_t(Xv)
    yt_ti = to_t([e['time'] for e in yt]); yt_ev = to_t([e['event'] for e in yt])
    yv_ti = to_t([e['time'] for e in yv]); yv_ev = to_t([e['event'] for e in yv])
    best, wait, state = np.inf, 0, None
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        cox_loss(net(Xt_t), yt_ti, yt_ev).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = cox_loss(net(Xv_t), yv_ti, yv_ev).item()
        if vl < best - 1e-6: best, wait, state = vl, 0, {k:v.cpu().clone() for k,v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if state: net.load_state_dict(state)
    net.eval(); return net

def ds_pred(net, X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

XGB_P = dict(objective="survival:cox", eval_metric="cox-nloglik",
             eta=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
             seed=SEED, verbosity=0)

def ci_score(ev, ti, risk):
    return concordance_index_censored(ev.astype(bool), ti, risk)[0]

def coxnet_fit(X, y, a):
    m = CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=0.9, max_iter=100_000,
                               tol=1e-7, fit_baseline_model=True)
    m.fit(X, y); return m

ALPHA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

def tune_alpha(Xtr, ytr, Xvl, yvl):
    ba, bc = ALPHA_GRID[-1], -1
    for a in ALPHA_GRID:
        try:
            m = coxnet_fit(Xtr, ytr, a)
            c = ci_score(yvl['event'], yvl['time'], m.predict(Xvl))
            if c > bc: bc, ba = c, a
        except: pass
    return ba

def build_splines(meta_df, feats):
    parts, dis_list, mapping = [], [], {}
    for f in feats:
        sp = dmatrix(f"bs({f}, df=4, degree=3, include_intercept=False)",
                     meta_df, return_type='dataframe')
        sp.columns = [f"{f}_s{i}" for i in range(sp.shape[1])]
        parts.append(sp); dis_list.append(sp.design_info); mapping[f] = sp.columns.tolist()
    return pd.concat(parts, axis=1), dis_list, mapping

def get_gene_symbol_col(pt):
    for c in ['Gene Symbol','Gene.Symbol','GENE_SYMBOL','Symbol','gene_symbol']:
        if c in pt.columns: return c
    for c in pt.columns:
        if 'symbol' in c.lower() or ('gene' in c.lower() and c.lower() != 'gene_id'):
            return c
    return None

# NOTE: deliberately exclude 'status' and 'contact' — they are substrings of
# legitimate survival columns (vital_status, months_to_last_contact_or_death).
SKIP_META = ['submission_date','last_update_date','contact_email','contact_name',
             'contact_phone','protocol','hyb_protocol','scan_protocol',
             'supplementary','data_processing','label_protocol','extract_protocol',
             'platform_id','series_id','data_row_count','channel_count',
             'taxid_ch1','organism_ch1','molecule_ch1','geo_accession',
             'description']

TIME_KW = ['overall survival','os month','os_month','survival month','days to death',
           'days_to_death','os_days','time to death','months survived','survival time',
           'follow-up','follow up','last follow','last contact','last alive',
           'disease free','relapse free','time (month','os time','days to last',
           'months to last','total surv','months to last contact','last contact or death',
           'mths to last','time to last contact','months_to_last']
EVENT_KW = ['vital status','vital_status','os_status','os status',
            'survival status','patient status','alive or dead']

def _to_months(series):
    raw = pd.to_numeric(series, errors='coerce')
    if pd.notna(raw.median()) and raw.median() > 200: raw = raw / 30.44
    return raw

def _to_event(series):
    sl = series.astype(str).str.strip().str.lower()
    num = pd.to_numeric(sl, errors='coerce')
    if num.isin([0.0, 1.0]).mean() > 0.8: return num
    ev = pd.Series(np.nan, index=series.index)
    ev[sl.str.contains(r'\bdead\b|\bdeceased\b|\bdied\b|\bdeath\b', na=False, regex=True)] = 1.0
    ev[sl.str.contains(r'\balive\b|\bliving\b|\bcensored\b', na=False, regex=True)] = 0.0
    return ev

def _skip(col):
    cl = col.lower()
    return any(s in cl for s in SKIP_META)

def parse_survival(pheno):
    pheno = pheno.copy()
    os_time  = pd.Series(np.nan, index=pheno.index)
    os_event = pd.Series(np.nan, index=pheno.index)

    for col in pheno.columns:
        if _skip(col): continue
        vals = pheno[col].astype(str)
        kv = vals.str.extract(r'^([^:]+):\s*(.+)$')
        if not kv[0].notna().any(): continue
        kl = kv.loc[kv[0].notna(), 0].str.lower().str.strip()
        vl = kv.loc[kv[0].notna(), 1].str.strip()
        if os_time.isna().all():
            t_m = kl.str.contains('|'.join(re.escape(k) for k in TIME_KW), na=False)
            if t_m.any():
                raw = _to_months(vl[t_m])
                os_time[t_m[t_m].index] = raw.values
        if os_event.isna().all():
            e_m = kl.str.contains('|'.join(re.escape(k) for k in EVENT_KW), na=False)
            if e_m.any():
                ev = _to_event(vl[e_m])
                os_event[e_m[e_m].index] = ev.values

    def _ck(col):
        cl = col.lower()
        return re.sub(r'^characteristics_ch\d+\.\d+\.?', '', cl).replace('_',' ').strip()

    if os_time.isna().all() or os_event.isna().all():
        for col in pheno.columns:
            if _skip(col): continue
            ck = _ck(col)
            if os_time.isna().all() and any(k in ck for k in TIME_KW):
                raw = _to_months(pheno[col])
                if raw.notna().sum() > max(5, len(pheno)*0.25): os_time = raw
            if os_event.isna().all() and any(k in ck for k in EVENT_KW):
                ev = _to_event(pheno[col])
                if ev.notna().sum() > max(5, len(pheno)*0.25): os_event = ev

    # Pass 3: merge multi-column death events (GSE31210 style)
    death_ev_cols  = [c for c in pheno.columns if not _skip(c)
                      and re.search(r'\bdeath\b', _ck(c))
                      and 'days' not in _ck(c) and 'month' not in _ck(c)]
    death_day_cols = [c for c in pheno.columns if not _skip(c)
                      and 'days before death' in _ck(c)]
    if death_ev_cols and os_event.isna().mean() > 0.3:
        merged = pd.Series(np.nan, index=pheno.index)
        for c in death_ev_cols:
            ev = _to_event(pheno[c]); fill = merged.isna() & ev.notna(); merged[fill] = ev[fill]
        if merged.notna().sum() > 10 and merged.sum() > 0: os_event = merged
    if death_day_cols and os_time.isna().mean() > 0.3:
        merged = pd.Series(np.nan, index=pheno.index)
        for c in death_day_cols:
            raw = _to_months(pheno[c]); fill = merged.isna() & raw.notna(); merged[fill] = raw[fill]
        if merged.notna().sum() > 10: os_time = merged

    return os_time, os_event

def load_geo(geo_id):
    print(f"  Loading {geo_id}...")
    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(GEO_DIR), silent=True)
    except Exception as e:
        print(f"  Download failed: {e}"); return None
    try:
        expr = gse.pivot_samples('VALUE').apply(pd.to_numeric, errors='coerce')
        plat_id = list(gse.gpls.keys())[0]
        plat    = gse.gpls[plat_id].table
        sym_col = get_gene_symbol_col(plat)
        if sym_col is None: print("  No gene symbol column."); return None
        annot = plat[['ID', sym_col]].copy()
        annot.columns = ['ID','GeneSymbol']
        annot = annot.dropna(subset=['GeneSymbol'])
        annot = annot[annot['GeneSymbol'].str.strip() != '']
        annot['GeneSymbol'] = annot['GeneSymbol'].str.split('///').str[0].str.strip()
        annot = annot.drop_duplicates('GeneSymbol').set_index('ID')
        expr.index = expr.index.astype(str)
        shared = expr.index.intersection(annot.index)
        if len(shared) == 0: print("  No probe matches."); return None
        expr2 = expr.loc[shared].copy()
        expr2.index = annot.loc[shared, 'GeneSymbol']
        expr2 = expr2[~expr2.index.duplicated(keep='first')]
        expr2 = np.log2(expr2.clip(lower=0) + 1)
        expr_t = expr2.T
        print(f"  Expression: {expr_t.shape[0]} samples × {expr_t.shape[1]} genes")
    except Exception as e:
        print(f"  Expression failed: {e}"); return None
    try:
        pheno = gse.phenotype_data
        os_time, os_event = parse_survival(pheno)
        print(f"  Survival: time non-null={os_time.notna().sum()}, "
              f"event non-null={os_event.notna().sum()}")
    except Exception as e:
        print(f"  Phenotype failed: {e}"); return None
    if os_event.notna().any() and os_event.sum() == 0:
        print(f"  Zero events — skipping {geo_id}."); return None
    common = expr_t.index.intersection(os_time.index)
    if len(common) == 0: print("  No sample overlap."); return None
    expr_t = expr_t.loc[common]; os_time = os_time.loc[common]; os_event = os_event.loc[common]
    valid = os_time.notna() & (os_time > 0) & os_event.notna()
    n_v = valid.sum()
    print(f"  Valid patients: {n_v}")
    if n_v < 30: print(f"  Too few ({n_v})."); return None
    return {'expr_t': expr_t[valid], 'os_time': os_time[valid], 'os_event': os_event[valid],
            'geo_id': geo_id, 'n': int(n_v), 'events': int(os_event[valid].sum()),
            'gse': gse}

# ================================================================
# 1. LOAD TCGA
# ================================================================

print("\n[1] Loading TCGA-LUAD...")

def load_cbio(path):
    with open(path) as fh:
        skip = sum(1 for line in fh if line.startswith('#'))
    return pd.read_csv(path, sep='\t', skiprows=skip, low_memory=False)

patient = load_cbio(DATA_DIR / 'data_clinical_patient.txt')
sample  = load_cbio(DATA_DIR / 'data_clinical_sample.txt')
df_tcga = patient.merge(sample, on='PATIENT_ID', how='inner')
df_tcga['OS_time']  = pd.to_numeric(df_tcga['OS_MONTHS'], errors='coerce')
df_tcga['OS_event'] = df_tcga['OS_STATUS'].str.startswith('1').fillna(False).astype(int)
df_tcga = df_tcga[df_tcga['OS_time'].notna() & (df_tcga['OS_time'] > 0)].copy().reset_index(drop=True)
print(f"  TCGA: n={len(df_tcga)}, events={df_tcga['OS_event'].sum()}")

mrna_raw = pd.read_csv(DATA_DIR/'data_mrna_seq_v2_rsem.txt', sep='\t', index_col=0, low_memory=False)
mrna_raw = mrna_raw.drop(columns=['Entrez_Gene_Id'], errors='ignore').T.copy()
mrna_raw.index = mrna_raw.index.str[:-3]
mrna_raw = mrna_raw[~mrna_raw.index.duplicated()]
mrna_raw = np.log2(mrna_raw.astype(float) + 1).replace([np.inf,-np.inf], np.nan)
mrna_raw = mrna_raw.loc[:, mrna_raw.notna().mean() > 0.7]

ids_tcga = df_tcga['PATIENT_ID'].values
mrna_tcga = mrna_raw.reindex(ids_tcga).reset_index(drop=True)
y_tcga = Surv.from_arrays(event=df_tcga['OS_event'].values, time=df_tcga['OS_time'].values)

has_rna  = mrna_tcga.notna().any(axis=1)
mrna_real = mrna_tcga[has_rna]; y_real = y_tcga[has_rna.values]
gene_var  = mrna_real.var()
top_v     = gene_var.nlargest(2000).index
gene_ci_d = {}
for g in top_v:
    vals = mrna_real[g].fillna(mrna_real[g].median()).values
    if vals.std() < 1e-6: continue
    try: gene_ci_d[g] = abs(ci_score(y_real['event'], y_real['time'], vals) - 0.5)
    except: pass
TOP_GENES = sorted(gene_ci_d, key=gene_ci_d.get, reverse=True)[:10]
mrna_med  = mrna_real[TOP_GENES].median()
print(f"  Top TCGA genes: {', '.join(TOP_GENES)}")

# ================================================================
# 2. DOWNLOAD BOTH COHORTS
# ================================================================

print("\n[2] Loading GEO cohorts...")
geo_datasets = {}
for geo_id in TARGET_GEOS:
    data = load_geo(geo_id)
    if data is not None:
        geo_datasets[geo_id] = data
    else:
        print(f"  WARNING: {geo_id} unavailable.")

if not geo_datasets:
    print("No external datasets available."); sys.exit(1)

# ================================================================
# 3–7. PER-COHORT PIPELINE
# ================================================================

TRANSFER_CLIN = ['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE',
                 'PATH_M_STAGE','PATH_N_STAGE','PATH_T_STAGE']
META_FEATS = ["RSF","GBS","XGB","DS"]

all_results = {}

for GEO_ID, geo_data in geo_datasets.items():
    print(f"\n{'='*70}")
    print(f"PIPELINE: {GEO_ID}  (n={geo_data['n']}, events={geo_data['events']})")
    print(f"{'='*70}")

    ext_expr   = geo_data['expr_t']
    ext_time   = geo_data['os_time']
    ext_event  = geo_data['os_event']
    n_ext      = geo_data['n']
    ev_ext     = geo_data['events']

    # --- Gene matching ---
    common_genes = [g for g in TOP_GENES if g in ext_expr.columns]
    if len(common_genes) < 2:
        top100 = sorted(gene_ci_d, key=gene_ci_d.get, reverse=True)[:100]
        common_genes = [g for g in top100 if g in ext_expr.columns][:10]
    USE_GENES = common_genes
    print(f"  Gene overlap: {len(USE_GENES)}/{len(TOP_GENES)}  → {USE_GENES}")

    # --- Clinical features ---
    clin_cols = [c for c in TRANSFER_CLIN if c in df_tcga.columns]
    Xc_tcga   = df_tcga[clin_cols].copy()
    cat_c = Xc_tcga.select_dtypes(include=['object','category']).columns.tolist()
    num_c = Xc_tcga.select_dtypes(include=['number','bool']).columns.tolist()
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_c),
        ('num', SimpleImputer(strategy='median'), num_c),
    ], remainder='drop')
    Xp_tcga = pre.fit_transform(Xc_tcga)
    ohe = pre.named_transformers_['cat']
    cn  = (ohe.get_feature_names_out(cat_c).tolist() if cat_c else []) + num_c
    if len(cn) != Xp_tcga.shape[1]: cn = [f"f{i}" for i in range(Xp_tcga.shape[1])]

    # External clinical (parse from phenotype)
    Xc_ext = pd.DataFrame(np.nan, index=ext_expr.index, columns=clin_cols)
    pheno_ext = geo_data['gse'].phenotype_data
    for col in pheno_ext.columns:
        coll = col.lower(); vals = pheno_ext[col].astype(str)
        kv = vals.str.extract(r'^([^:]+):\s*(.+)$')
        ks = kv[0].str.lower().str.strip() if kv[0].notna().any() else pd.Series('', index=pheno_ext.index)
        vs = kv[1].str.strip() if kv[1].notna().any() else vals
        am = ks.str.contains('age', na=False)
        sm = ks.str.contains('gender|sex', na=False)
        gm = ks.str.contains('stage', na=False)
        if am.any() and 'AGE' in Xc_ext.columns:
            Xc_ext.loc[pheno_ext.index.intersection(ext_expr.index), 'AGE'] = \
                pd.to_numeric(vs[am], errors='coerce').reindex(ext_expr.index).values
        if sm.any() and 'SEX' in Xc_ext.columns:
            Xc_ext.loc[pheno_ext.index.intersection(ext_expr.index), 'SEX'] = \
                vs[sm].reindex(ext_expr.index).values
        if gm.any() and 'AJCC_PATHOLOGIC_TUMOR_STAGE' in Xc_ext.columns:
            def map_s(s):
                s = str(s).upper()
                if 'IV' in s or '4' in s: return 'STAGE IV'
                elif 'III' in s or '3' in s: return 'STAGE III'
                elif 'II' in s or '2' in s: return 'STAGE II'
                else: return 'STAGE I'
            Xc_ext.loc[pheno_ext.index.intersection(ext_expr.index),
                       'AJCC_PATHOLOGIC_TUMOR_STAGE'] = \
                vs[gm].reindex(ext_expr.index).map(map_s).values
    for c in cat_c:
        if c in Xc_ext.columns:
            Xc_ext[c] = Xc_ext[c].fillna(Xc_tcga[c].mode()[0] if not Xc_tcga[c].mode().empty else 'Unknown')
    for c in num_c:
        if c in Xc_ext.columns:
            Xc_ext[c] = pd.to_numeric(Xc_ext[c], errors='coerce').fillna(
                pd.to_numeric(Xc_tcga[c], errors='coerce').median())
    Xp_ext = pre.transform(Xc_ext)

    # mRNA
    if USE_GENES:
        mrna_sc = StandardScaler()
        mrna_sc.fit(mrna_real[USE_GENES].fillna(mrna_med[USE_GENES]))
        tcga_mrna_s = mrna_sc.transform(mrna_tcga[USE_GENES].fillna(mrna_med[USE_GENES]))
        ext_mrna_s  = mrna_sc.transform(
            ext_expr[USE_GENES].fillna(mrna_real[USE_GENES].median()))
        X_tcga = np.hstack([Xp_tcga, tcga_mrna_s])
        X_ext  = np.hstack([Xp_ext,  ext_mrna_s])
        feat_names = cn + USE_GENES
    else:
        X_tcga = Xp_tcga; X_ext = Xp_ext; feat_names = cn

    X_tcga = pd.DataFrame(X_tcga, columns=feat_names)
    X_ext  = pd.DataFrame(X_ext,  columns=feat_names, index=ext_expr.index)
    print(f"  Features: {X_tcga.shape[1]}")

    # Structured survival
    y_s = np.array(list(zip(df_tcga['OS_event'].values, df_tcga['OS_time'].values)),
                   dtype=[('event',bool),('time',float)])
    y_ext_s = np.array(list(zip(ext_event.values, ext_time.values)),
                       dtype=[('event',bool),('time',float)])

    # Early-stopping split
    X_tr2, X_es, y_tr2, y_es = train_test_split(
        X_tcga, y_s, test_size=0.15, stratify=y_s['event'], random_state=SEED)

    # CoxNet feature selection
    best_cx, ba = -1, 0.1
    for a in np.logspace(-2, 2, 12):
        fss = []
        for ti, vi in KFold(3, shuffle=True, random_state=SEED).split(X_tcga):
            try:
                m = CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=0.9, max_iter=100_000, tol=1e-7)
                m.fit(X_tcga.iloc[ti].values, y_s[ti])
                fss.append(ci_score(y_s['event'][vi], y_s['time'][vi],
                                    m.predict(X_tcga.iloc[vi].values)))
            except: pass
        if fss and np.mean(fss) > best_cx: best_cx, ba = np.mean(fss), a
    cx_f = CoxnetSurvivalAnalysis(alphas=[ba], l1_ratio=0.9, max_iter=100_000, tol=1e-7)
    cx_f.fit(X_tcga.values, y_s)
    coefs = cx_f.coef_.ravel()
    sel   = X_tcga.columns[np.abs(coefs) > 1e-8].tolist()
    if len(sel) < 5: sel = X_tcga.columns[np.argsort(np.abs(coefs))[-10:]].tolist()
    print(f"  Selected features: {len(sel)}")

    sc = StandardScaler()
    Xs     = pd.DataFrame(sc.fit_transform(X_tcga[sel]), columns=sel)
    Xe     = pd.DataFrame(sc.transform(X_ext[sel]),  columns=sel, index=ext_expr.index)
    Xtr2s  = pd.DataFrame(sc.transform(X_tr2[sel]),  columns=sel)
    Xess   = pd.DataFrame(sc.transform(X_es[sel]),   columns=sel)

    # Inner 5-fold OOF
    kf  = KFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(Xs), 4))
    for tr_i, vl_i in kf.split(Xs):
        Xi, Xj = Xs.iloc[tr_i].values, Xs.iloc[vl_i].values
        yi, yj = y_s[tr_i], y_s[vl_i]
        rsf = RandomSurvivalForest(n_estimators=300, max_features='sqrt',
                                   min_samples_leaf=5, random_state=SEED, n_jobs=-1)
        rsf.fit(Xi, yi); oof[vl_i, 0] = rsf.predict(Xj)
        gbs = GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.05,
                                               max_depth=3, random_state=SEED)
        gbs.fit(Xi, yi); oof[vl_i, 1] = gbs.predict(Xj)
        dt = xgb.DMatrix(Xi, label=[e['time'] for e in yi], weight=[e['event'] for e in yi])
        dv = xgb.DMatrix(Xess.values, label=[e['time'] for e in y_es], weight=[e['event'] for e in y_es])
        xm = xgb.train(XGB_P, dt, num_boost_round=500, evals=[(dv,'v')],
                       early_stopping_rounds=30, verbose_eval=False)
        it = getattr(xm, 'best_iteration', xm.num_boosted_rounds())
        oof[vl_i, 2] = xm.predict(xgb.DMatrix(Xj), iteration_range=(0, it))
        dn = train_ds(Xi, yi, Xess.values, y_es, Xi.shape[1])
        oof[vl_i, 3] = ds_pred(dn, Xj)

    # Final models → external predictions
    rsf_f = RandomSurvivalForest(n_estimators=300, max_features='sqrt',
                                  min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    rsf_f.fit(Xs.values, y_s); ext_rsf = rsf_f.predict(Xe.values)
    gbs_f = GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.05,
                                              max_depth=3, random_state=SEED)
    gbs_f.fit(Xs.values, y_s); ext_gbs = gbs_f.predict(Xe.values)
    dt_f = xgb.DMatrix(Xtr2s.values, label=[e['time'] for e in y_tr2], weight=[e['event'] for e in y_tr2])
    dv_f = xgb.DMatrix(Xess.values,  label=[e['time'] for e in y_es],  weight=[e['event'] for e in y_es])
    xm_f = xgb.train(XGB_P, dt_f, num_boost_round=500, evals=[(dv_f,'v')],
                     early_stopping_rounds=30, verbose_eval=False)
    it_f = getattr(xm_f, 'best_iteration', xm_f.num_boosted_rounds())
    ext_xgb = xm_f.predict(xgb.DMatrix(Xe.values), iteration_range=(0, it_f))
    dn_f = train_ds(Xtr2s.values, y_tr2, Xess.values, y_es, Xs.shape[1])
    ext_ds = ds_pred(dn_f, Xe.values)

    ext_preds = np.column_stack([ext_rsf, ext_gbs, ext_xgb, ext_ds])
    meta_tcga = pd.DataFrame(oof,       columns=META_FEATS)
    meta_ext  = pd.DataFrame(ext_preds, columns=META_FEATS, index=ext_expr.index)
    for f in META_FEATS:
        mn, mx = meta_tcga[f].min(), meta_tcga[f].max()
        meta_ext[f] = meta_ext[f].clip(mn, mx)

    sp_tr, dis_list, mapping = build_splines(meta_tcga, META_FEATS)
    sp_ext_parts = []
    for i, f in enumerate(META_FEATS):
        S = build_design_matrices([dis_list[i]], meta_ext)[0]
        sp_ext_parts.append(pd.DataFrame(S, index=meta_ext.index))
    sp_ext = pd.concat(sp_ext_parts, axis=1)
    sp_ext.columns = sp_tr.columns

    sp_tv, sp_vl = train_test_split(sp_tr, test_size=0.2, random_state=SEED)
    ym_tv = np.array(list(zip(y_s[sp_tv.index]['event'], y_s[sp_tv.index]['time'])),
                     dtype=[('event',bool),('time',float)])
    ym_vl = np.array(list(zip(y_s[sp_vl.index]['event'], y_s[sp_vl.index]['time'])),
                     dtype=[('event',bool),('time',float)])

    a_gam = tune_alpha(sp_tv.values, ym_tv, sp_vl.values, ym_vl)
    gam_f = coxnet_fit(sp_tr.values, y_s, a_gam)
    ext_risk = gam_f.predict(sp_ext.values)

    a_lin = tune_alpha(meta_tcga.loc[sp_tv.index].values, ym_tv,
                       meta_tcga.loc[sp_vl.index].values, ym_vl)
    lin_f = coxnet_fit(meta_tcga.values, y_s, a_lin)
    ext_lin = lin_f.predict(meta_ext.values)

    # Evaluate
    results = {
        'RSF':             ci_score(y_ext_s['event'], y_ext_s['time'], ext_rsf),
        'GBS':             ci_score(y_ext_s['event'], y_ext_s['time'], ext_gbs),
        'XGBoost-Cox':     ci_score(y_ext_s['event'], y_ext_s['time'], ext_xgb),
        'DeepSurv':        ci_score(y_ext_s['event'], y_ext_s['time'], ext_ds),
        'Linear Stacking': ci_score(y_ext_s['event'], y_ext_s['time'], ext_lin),
        'SAGAM (ours)':    ci_score(y_ext_s['event'], y_ext_s['time'], ext_risk),
    }
    print(f"\n  Results on {GEO_ID}:")
    for k, v in results.items(): print(f"    {k:<20}: {v:.4f}")

    # Bootstrap CI
    rng = np.random.default_rng(SEED)
    boot_g, boot_l = [], []
    for _ in range(1000):
        idx = rng.choice(len(y_ext_s), len(y_ext_s), replace=True)
        boot_g.append(ci_score(y_ext_s['event'][idx], y_ext_s['time'][idx], ext_risk[idx]))
        boot_l.append(ci_score(y_ext_s['event'][idx], y_ext_s['time'][idx], ext_lin[idx]))
    ci_glo, ci_ghi = np.percentile(boot_g, [2.5, 97.5])
    ci_llo, ci_lhi = np.percentile(boot_l, [2.5, 97.5])
    delta = results['SAGAM (ours)'] - results['Linear Stacking']
    print(f"\n  SAGAM CI:  [{ci_glo:.4f}, {ci_ghi:.4f}]")
    print(f"  Linear CI: [{ci_llo:.4f}, {ci_lhi:.4f}]")
    print(f"  SAGAM - Linear: {delta:+.4f}")

    # KM figure
    risk_grp = pd.qcut(ext_risk, q=3, labels=['Low','Medium','High'])
    lr_lh = logrank_test(
        y_ext_s['time'][risk_grp=='Low'],  y_ext_s['time'][risk_grp=='High'],
        y_ext_s['event'][risk_grp=='Low'], y_ext_s['event'][risk_grp=='High'])
    sig = ('***' if lr_lh.p_value < 0.001 else '**' if lr_lh.p_value < 0.01
           else '*' if lr_lh.p_value < 0.05 else 'NS')
    print(f"  Log-rank (Low vs High): p={lr_lh.p_value:.4e} [{sig}]")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#2E7D32', '#F57C00', '#C62828']
    kmf = KaplanMeierFitter()
    for i, (grp, msk) in enumerate([('Low', risk_grp=='Low'),
                                     ('Medium', risk_grp=='Medium'),
                                     ('High', risk_grp=='High')]):
        kmf.fit(y_ext_s['time'][msk], y_ext_s['event'][msk],
                label=f"{grp} Risk (n={msk.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2.5,
                                   color=colors[i], alpha=0.9)
    ax.set_xlabel('Time (Months)', fontsize=13)
    ax.set_ylabel('Survival Probability', fontsize=13)
    ax.set_title(f'External Validation — {GEO_ID} (n={n_ext}, events={ev_ext})\n'
                 f'SAGAM C={results["SAGAM (ours)"]:.3f} [{ci_glo:.3f},{ci_ghi:.3f}]  '
                 f'  Log-rank p={lr_lh.p_value:.4f} [{sig}]',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--'); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc='lower left')
    plt.tight_layout()
    km_path = OUTPUT_DIR / f'ext_km_{GEO_ID.lower()}.png'
    plt.savefig(km_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  KM figure → {km_path.name}")

    # Save per-cohort text
    txt_path = OUTPUT_DIR / f'ext_val_{GEO_ID.lower()}.txt'
    with open(txt_path, 'w') as f:
        f.write(f"EXTERNAL VALIDATION — {GEO_ID}\n{'='*50}\n")
        f.write(f"n={n_ext}  events={ev_ext}  genes={', '.join(USE_GENES)}\n\n")
        for k, v in results.items(): f.write(f"  {k:<20}: {v:.4f}\n")
        f.write(f"\nSAGAM 95% CI:   [{ci_glo:.4f}, {ci_ghi:.4f}]\n")
        f.write(f"Linear 95% CI:  [{ci_llo:.4f}, {ci_lhi:.4f}]\n")
        f.write(f"SAGAM - Linear: {delta:+.4f}\n")
        f.write(f"\nLog-rank Low vs High: p={lr_lh.p_value:.4e} [{sig}]\n")

    all_results[GEO_ID] = {
        'n': n_ext, 'events': ev_ext,
        'C_SAGAM': results['SAGAM (ours)'],
        'C_Linear': results['Linear Stacking'],
        'C_RSF': results['RSF'], 'C_GBS': results['GBS'],
        'C_XGB': results['XGBoost-Cox'], 'C_DS': results['DeepSurv'],
        'Delta_SL': delta,
        'SAGAM_CI_lo': ci_glo, 'SAGAM_CI_hi': ci_ghi,
        'Linear_CI_lo': ci_llo, 'Linear_CI_hi': ci_lhi,
        'KM_p': lr_lh.p_value, 'KM_sig': sig,
        'genes': USE_GENES,
    }

# ================================================================
# SUMMARY TABLE
# ================================================================

print("\n" + "=" * 70)
print("MULTI-COHORT SUMMARY")
print("=" * 70)
print(f"\n{'Cohort':<12} {'n':>5} {'ev':>5} {'genes':>6} "
      f"{'C_SAGAM':>8} {'C_Lin':>8} {'Δ(S-L)':>8} {'KM-p':>10}")
print("-" * 68)
for gid, r in all_results.items():
    print(f"  {gid:<10} {r['n']:>5} {r['events']:>5} {len(r['genes']):>6} "
          f"{r['C_SAGAM']:>8.4f} {r['C_Linear']:>8.4f} "
          f"{r['Delta_SL']:>+8.4f} {r['KM_p']:>10.4e}")

summary_df = pd.DataFrame(all_results).T
summary_df.to_csv(OUTPUT_DIR / 'ext_val_multi_summary.csv')
print(f"\n✓ Summary → ext_val_multi_summary.csv")
print("=" * 70)
print("DONE")
print("=" * 70)
