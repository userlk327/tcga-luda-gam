"""
External Validation — LUAD GEO Cohorts
=======================================
Primary:  GSE31210  (226 LUAD, Takeuchi et al., GPL570)
Fallback: GSE37745  (NSCLC with survival, GPL570)
Fallback: GSE68465  (Director's Challenge, GPL96) — raw series matrix parse

Fixes over v1:
  - Uses gse.phenotype_data instead of per-GSM metadata
  - Tries multiple gene-symbol column names (handles GPL96 / GPL570)
  - Robust survival parsing (days→months conversion, multiple key names)
  - Falls back across three datasets automatically
"""

from pathlib import Path
import re
import warnings, random, gzip
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
from lifelines.statistics import logrank_test, multivariate_logrank_test

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / 'dataset'
OUTPUT_DIR = REPO_ROOT / 'results_v2'
GEO_DIR    = REPO_ROOT / 'geo_cache'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
GEO_DIR.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("EXTERNAL VALIDATION — LUAD GEO COHORT")
print("=" * 70)

# ================================================================
# HELPERS
# ================================================================

class DeepSurv(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n,64),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def cox_loss(risk,times,events):
    o=torch.argsort(-times); r,e=risk[o],events[o]
    return -(e*(r-torch.logcumsumexp(r,0))).sum()/(e.sum()+1e-8)

def train_ds(Xt,yt,Xv,yv,n,epochs=200,patience=20):
    net=DeepSurv(n).to(device)
    opt=optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    to_t=lambda a:torch.tensor(a,dtype=torch.float32).to(device)
    Xt_t,Xv_t=to_t(Xt),to_t(Xv)
    yt_ti=to_t([e['time'] for e in yt]); yt_ev=to_t([e['event'] for e in yt])
    yv_ti=to_t([e['time'] for e in yv]); yv_ev=to_t([e['event'] for e in yv])
    best,wait,state=np.inf,0,None
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        cox_loss(net(Xt_t),yt_ti,yt_ev).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl=cox_loss(net(Xv_t),yv_ti,yv_ev).item()
        if vl<best-1e-6: best,wait,state=vl,0,{k:v.cpu().clone() for k,v in net.state_dict().items()}
        else:
            wait+=1
            if wait>=patience: break
    if state: net.load_state_dict(state)
    net.eval(); return net

def ds_pred(net,X):
    with torch.no_grad():
        return net(torch.tensor(X,dtype=torch.float32).to(device)).cpu().numpy()

XGB_P=dict(objective="survival:cox",eval_metric="cox-nloglik",
           eta=0.05,max_depth=3,subsample=0.8,colsample_bytree=0.8,
           seed=SEED,verbosity=0)

def ci(ev,ti,risk): return concordance_index_censored(ev,ti,risk)[0]

def coxnet_fit(X,y,a):
    m=CoxnetSurvivalAnalysis(alphas=[a],l1_ratio=0.9,max_iter=100_000,tol=1e-7)
    m.fit(X,y); return m

ALPHA_GRID=[0.001,0.005,0.01,0.05,0.1,0.5]

def tune_alpha(Xtr,ytr,Xvl,yvl):
    ba,bc=ALPHA_GRID[-1],-1
    for a in ALPHA_GRID:
        try:
            m=coxnet_fit(Xtr,ytr,a)
            c=ci(yvl['event'],yvl['time'],m.predict(Xvl))
            if c>bc: bc,ba=c,a
        except: pass
    return ba

def build_splines(meta_df,feats):
    parts,dis,mapping=[],[],{}
    for f in feats:
        sp=dmatrix(f"bs({f},df=4,degree=3,include_intercept=False)",
                   meta_df,return_type='dataframe')
        sp.columns=[f"{f}_s{i}" for i in range(sp.shape[1])]
        parts.append(sp); dis.append(sp.design_info); mapping[f]=sp.columns.tolist()
    return pd.concat(parts,axis=1),dis,mapping

def apply_splines(meta_df,feats,dis_list):
    return pd.concat([
        pd.DataFrame(build_design_matrices([dis_list[i]],meta_df)[0],index=meta_df.index)
        for i,f in enumerate(feats)
    ],axis=1)

def get_gene_symbol_col(platform_table):
    """Try multiple column name variants for gene symbols."""
    for candidate in ['Gene Symbol','Gene.Symbol','GENE_SYMBOL',
                      'Symbol','gene_symbol','GeneSymbol','gene symbol']:
        if candidate in platform_table.columns:
            return candidate
    # Fuzzy match
    for col in platform_table.columns:
        if 'symbol' in col.lower() or ('gene' in col.lower() and col.lower()!='gene_id'):
            return col
    return None

def parse_survival_from_pheno(pheno, verbose=True):
    """
    Robustly extract OS_time (months) and OS_event (0/1) from
    a GEOparse phenotype_data DataFrame.

    Handles:
    - key:value format ("vital status: Dead")
    - column-name keyword matching (characteristics_ch1.N.KEY format)
    - multi-cohort datasets (GSE31210) where death/time columns are split
    - days → months auto-conversion
    - normal tissue sample filtering
    """
    pheno = pheno.copy()

    # ---- Filter to tumor/primary samples first ---------------------------
    for col in pheno.columns:
        if 'tissue' in col.lower() or 'source' in col.lower():
            vals_l = pheno[col].astype(str).str.lower()
            tumor_mask = ~vals_l.str.contains(
                r'\bnormal\b|\bcontrol\b|\badjacent\b|\bnon.tumor\b',
                na=False, regex=True
            )
            if tumor_mask.sum() > 10 and tumor_mask.sum() < len(pheno):
                pheno = pheno[tumor_mask].copy()
                if verbose:
                    print(f"    [filter] Kept {len(pheno)} tumor samples "
                          f"(removed {(~tumor_mask).sum()} normal/control)")
                break

    os_time  = pd.Series(np.nan, index=pheno.index)
    os_event = pd.Series(np.nan, index=pheno.index)

    # ---- Helper functions ------------------------------------------------
    def _to_months(series):
        raw = pd.to_numeric(series, errors='coerce')
        med = raw.median()
        if pd.notna(med) and med > 200:    # almost certainly days
            raw = raw / 30.44
        return raw

    def _to_event(series):
        """Map dead/alive strings or 0/1 numerics to event indicator."""
        sl = series.astype(str).str.strip().str.lower()
        num = pd.to_numeric(sl, errors='coerce')
        if num.isin([0.0, 1.0]).mean() > 0.8:
            return num
        ev = pd.Series(np.nan, index=series.index)
        # DEAD patterns — specific enough to avoid false positives
        ev[sl.str.contains(r'\bdead\b|\bdeceased\b|\bdied\b|\bdeath\b',
                            na=False, regex=True)] = 1.0
        # ALIVE patterns — specific enough to avoid 'no' in 'nov', 'none', etc.
        ev[sl.str.contains(r'\balive\b|\bliving\b|\bcensored\b',
                            na=False, regex=True)] = 0.0
        return ev

    # Metadata columns to skip entirely in any keyword search
    SKIP_COL = ['submission_date','last_update_date','status','contact',
                'email','protocol','hyb_protocol','scan_protocol',
                'supplementary','data_processing','label_protocol',
                'extract_protocol','platform_id','series_id',
                'data_row_count','channel_count','taxid_ch1','organism_ch1',
                'molecule_ch1','geo_accession','type','description',
                'scan_','label_ch1','hyb_']

    def _skip(col):
        cl = col.lower()
        return any(s in cl for s in SKIP_COL)

    # ---- PASS 1: key:value format ("key: value" inside cell) -------------
    TIME_KW = [
        'overall survival','os month','os_month','survival month',
        'days to death','days_to_death','os_days',
        'time to death','months survived','survival time',
        'follow-up','follow up','followup','last follow',
        'time to last','last contact','last alive',
        'disease free','relapse free','recurrence free',
        'time (month','duration','os time',
        'days to last','months to last','time to recur',
        'months of follow','total surv',
    ]
    EVENT_KW_STRICT = [
        'vital status','vital_status','os_status','os status',
        'survival status','patient status','alive or dead','dead or alive',
    ]

    for col in pheno.columns:
        if _skip(col): continue
        vals   = pheno[col].astype(str)
        kv     = vals.str.extract(r'^([^:]+):\s*(.+)$')
        has_kv = kv[0].notna()
        if not has_kv.any(): continue

        kl = kv.loc[has_kv, 0].str.lower().str.strip()
        vl = kv.loc[has_kv, 1].str.strip()

        if os_time.isna().all():
            t_mask = kl.str.contains('|'.join(re.escape(k) for k in TIME_KW), na=False)
            if t_mask.any():
                raw = _to_months(vl[t_mask])
                os_time[t_mask[t_mask].index] = raw.values

        if os_event.isna().all():
            e_mask = kl.str.contains('|'.join(re.escape(k) for k in EVENT_KW_STRICT), na=False)
            if e_mask.any():
                ev = _to_event(vl[e_mask])
                os_event[e_mask[e_mask].index] = ev.values

    # ---- PASS 2: column-name keyword match (handles ch1.N.KEY format) ----
    def _col_key(col):
        """Extract the characteristic key from 'characteristics_ch1.N.KEY'."""
        cl = col.lower()
        cl = re.sub(r'^characteristics_ch\d+\.\d+\.?', '', cl)
        return cl.replace('_', ' ').strip()

    if os_time.isna().all() or os_event.isna().all():
        for col in pheno.columns:
            if _skip(col): continue
            ck = _col_key(col)

            if os_time.isna().all():
                if any(k in ck for k in TIME_KW):
                    raw = _to_months(pheno[col])
                    n_ok = raw.notna().sum()
                    if n_ok > max(5, len(pheno) * 0.25):
                        os_time = raw
                        if verbose: print(f"    [pass2-time]  '{col}'  (n={n_ok})")

            if os_event.isna().all():
                if any(k in ck for k in EVENT_KW_STRICT):
                    ev = _to_event(pheno[col])
                    n_ok = ev.notna().sum()
                    if n_ok > max(5, len(pheno) * 0.25):
                        os_event = ev
                        if verbose: print(f"    [pass2-event] '{col}'  (n={n_ok})")

    # ---- PASS 3: death-specific column merge (GSE31210-style) ------------
    # Many datasets have separate "death" and "days before death/censor" columns
    # per sub-cohort; we merge by taking first non-null per patient.

    death_event_cols = [c for c in pheno.columns if not _skip(c)
                        and re.search(r'\bdeath\b', _col_key(c))
                        and 'days' not in _col_key(c)
                        and 'month' not in _col_key(c)]

    death_days_cols  = [c for c in pheno.columns if not _skip(c)
                        and 'days before death' in _col_key(c)]

    death_months_cols = [c for c in pheno.columns if not _skip(c)
                         and 'month' in _col_key(c)
                         and ('death' in _col_key(c) or 'censor' in _col_key(c))
                         and 'relapse' not in _col_key(c)]

    if death_event_cols and os_event.isna().mean() > 0.3:
        merged_ev = pd.Series(np.nan, index=pheno.index)
        for c in death_event_cols:
            ev = _to_event(pheno[c])
            fill = merged_ev.isna() & ev.notna()
            merged_ev[fill] = ev[fill]
        n_filled = merged_ev.notna().sum()
        if n_filled > 10 and merged_ev.sum() > 0:
            os_event = merged_ev
            if verbose:
                print(f"    [pass3-event] merged {len(death_event_cols)} death cols  "
                      f"(n={n_filled}, events={int(merged_ev.sum())})")

    if death_days_cols and os_time.isna().mean() > 0.3:
        merged_ti = pd.Series(np.nan, index=pheno.index)
        for c in death_days_cols:
            raw = _to_months(pheno[c])
            fill = merged_ti.isna() & raw.notna()
            merged_ti[fill] = raw[fill]
        n_filled = merged_ti.notna().sum()
        if n_filled > 10:
            os_time = merged_ti
            if verbose:
                print(f"    [pass3-time]  merged {len(death_days_cols)} death-days cols  "
                      f"(n={n_filled}, median={merged_ti.median():.1f} mo)")

    # Fallback: try death_months_cols if still missing time
    if os_time.isna().mean() > 0.3 and death_months_cols:
        merged_ti = pd.Series(np.nan, index=pheno.index)
        for c in death_months_cols:
            raw = pd.to_numeric(pheno[c], errors='coerce')
            fill = merged_ti.isna() & raw.notna()
            merged_ti[fill] = raw[fill]
        n_filled = merged_ti.notna().sum()
        if n_filled > 10:
            os_time = merged_ti
            if verbose:
                print(f"    [pass3-time*] merged {len(death_months_cols)} death-months cols  "
                      f"(n={n_filled}, median={merged_ti.median():.1f} mo)")

    # ---- PASS 4: heuristic time (last resort, exclude age/score cols) ----
    NON_TIME = ['age','weight','height','bmi','pack','year of birth',
                'tumor size','grade','stage','year diagnos',
                'number of','count','score','index','ratio','bi ']

    if os_time.isna().all():
        for col in pheno.columns:
            if _skip(col): continue
            ck = _col_key(col)
            if any(kw in ck for kw in NON_TIME): continue
            ev_vals = pheno[col].astype(str).str.lower()
            if ev_vals.str.contains(r'\balive\b|\bdead\b|\bdeceased\b',
                                     na=False, regex=True).mean() > 0.2: continue
            raw = _to_months(pheno[col])
            n_ok = raw.notna().sum()
            if n_ok < max(10, len(pheno) * 0.3): continue
            med = raw.median()
            if 3 < med < 200 and raw.std() > 3 and (raw > 0).mean() > 0.8:
                os_time = raw
                if verbose: print(f"    [pass4-time]  '{col}'  (median={med:.1f} mo)")
                break

    # ---- Diagnostic if still empty ---------------------------------------
    if verbose and (os_time.isna().all() or os_event.isna().all()):
        print(f"    [DIAG] time non-null={os_time.notna().sum()}  "
              f"event non-null={os_event.notna().sum()}")

    return os_time, os_event

def load_geo_dataset(geo_id):
    """
    Load a GEO dataset and return:
      expr_t   : DataFrame (samples × genes, log2-normalised)
      os_time  : Series (months)
      os_event : Series (0/1)
      geo_id   : str
    Returns None on failure.
    """
    print(f"\n  Attempting {geo_id}...")
    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(GEO_DIR), silent=True)
    except Exception as e:
        print(f"  Download failed: {e}")
        return None

    # --- Expression ---
    try:
        expr = gse.pivot_samples('VALUE').apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        print(f"  Expression pivot failed: {e}")
        return None

    # --- Gene symbol mapping ---
    try:
        plat_id  = list(gse.gpls.keys())[0]
        plat_tbl = gse.gpls[plat_id].table
        sym_col  = get_gene_symbol_col(plat_tbl)
        if sym_col is None:
            print(f"  No gene symbol column found in platform {plat_id}. Columns: {list(plat_tbl.columns[:10])}")
            return None

        annot = plat_tbl[['ID', sym_col]].copy()
        annot.columns = ['ID','GeneSymbol']
        annot = annot.dropna(subset=['GeneSymbol'])
        annot = annot[annot['GeneSymbol'].str.strip() != '']
        annot['GeneSymbol'] = annot['GeneSymbol'].str.split('///').str[0].str.strip()
        annot = annot.drop_duplicates('GeneSymbol').set_index('ID')

        expr.index = expr.index.astype(str)
        shared_probes = expr.index.intersection(annot.index)
        if len(shared_probes) == 0:
            print(f"  No probes matched annotation index.")
            return None

        expr_mapped = expr.loc[shared_probes].copy()
        expr_mapped.index = annot.loc[shared_probes, 'GeneSymbol']
        expr_mapped = expr_mapped[~expr_mapped.index.duplicated(keep='first')]
        expr_mapped = np.log2(expr_mapped.clip(lower=0) + 1)
        expr_t = expr_mapped.T
        print(f"  Expression: {expr_t.shape[0]} samples × {expr_t.shape[1]} genes")
    except Exception as e:
        print(f"  Gene mapping failed: {e}")
        return None

    # --- Survival from phenotype_data ---
    try:
        pheno = gse.phenotype_data
        os_time, os_event = parse_survival_from_pheno(pheno)
        print(f"  Survival parsed: time non-null={os_time.notna().sum()}, "
              f"event non-null={os_event.notna().sum()}")
    except Exception as e:
        print(f"  Phenotype parse failed: {e}")
        return None

    # If events = 0, treat as parse failure and try next dataset
    if os_event.notna().any() and os_event.sum() == 0:
        print(f"  Event rate is 0% — likely wrong event column or DFS-only dataset.")
        print(f"  Trying DFS/recurrence endpoint as fallback...")
        # Re-scan for recurrence / disease-free status
        DFS_EVENT_KW = ['recurr','relaps','disease free','dfs','progression',
                        'cancer status','disease status']
        for col in pheno.columns:
            col_l2 = col.lower()
            if any(k in col_l2 for k in DFS_EVENT_KW):
                ev2 = pheno[col].astype(str).str.lower()
                # 1 = recurred/progressed, 0 = disease-free
                new_ev = pd.Series(np.nan, index=pheno.index)
                new_ev[ev2.str.contains(r'recurr|relaps|progress|yes|\b1\b', na=False, regex=True)] = 1.0
                new_ev[ev2.str.contains(r'disease.free|no recurr|no relaps|\b0\b|censored|nedd|ned', na=False, regex=True)] = 0.0
                if new_ev.sum() > 5:
                    os_event = new_ev
                    print(f"    [DFS fallback] event ← '{col}'  "
                          f"(events={int(new_ev.sum())}/{int(new_ev.notna().sum())})")
                    break
        # If still 0, fail this dataset
        if os_event.sum() == 0:
            print(f"  Cannot recover event column. Skipping {geo_id}.")
            return None

    # Align survival with expression
    common_idx = expr_t.index.intersection(os_time.index)
    if len(common_idx) == 0:
        print(f"  No overlap between expression samples and survival index.")
        return None

    expr_t    = expr_t.loc[common_idx]
    os_time   = os_time.loc[common_idx]
    os_event  = os_event.loc[common_idx]

    # Filter to patients with valid survival
    valid_mask = (os_time.notna() & (os_time > 0) & os_event.notna())
    n_valid = valid_mask.sum()
    print(f"  Valid patients (time>0, both non-null): {n_valid}")

    if n_valid < 30:
        print(f"  Too few valid patients ({n_valid} < 30). Trying next dataset.")
        return None

    return {
        'expr_t'   : expr_t[valid_mask],
        'os_time'  : os_time[valid_mask],
        'os_event' : os_event[valid_mask],
        'geo_id'   : geo_id,
        'n'        : int(n_valid),
        'events'   : int(os_event[valid_mask].sum()),
    }

# ================================================================
# 1. LOAD TCGA DATA
# ================================================================

print("\n[1] Loading TCGA-LUAD data...")

def load_cbio(path):
    with open(path) as fh:
        skip = sum(1 for line in fh if line.startswith('#'))
    return pd.read_csv(path, sep='\t', skiprows=skip, low_memory=False)

patient = load_cbio(DATA_DIR / 'data_clinical_patient.txt')
sample  = load_cbio(DATA_DIR / 'data_clinical_sample.txt')
df      = patient.merge(sample, on='PATIENT_ID', how='inner')
df['OS_time']  = pd.to_numeric(df['OS_MONTHS'], errors='coerce')
df['OS_event'] = df['OS_STATUS'].str.startswith('1').fillna(False).astype(int)
df = df[df['OS_time'].notna() & (df['OS_time'] > 0)].copy().reset_index(drop=True)

print(f"  TCGA: n={len(df)}, events={df['OS_event'].sum()}")

mrna_raw = pd.read_csv(DATA_DIR/'data_mrna_seq_v2_rsem.txt',
                       sep='\t',index_col=0,low_memory=False)
mrna_raw = mrna_raw.drop(columns=['Entrez_Gene_Id'],errors='ignore')
mrna_raw = mrna_raw.T.copy()
mrna_raw.index = mrna_raw.index.str[:-3]
mrna_raw = mrna_raw[~mrna_raw.index.duplicated()]
mrna_raw = np.log2(mrna_raw.astype(float)+1)
mrna_raw = mrna_raw.replace([np.inf,-np.inf],np.nan)
mrna_raw = mrna_raw.loc[:,mrna_raw.notna().mean()>0.7]

mrna_tcga = mrna_raw.reindex(df['PATIENT_ID'].values).reset_index(drop=True)
y_tcga    = Surv.from_arrays(event=df['OS_event'].values, time=df['OS_time'].values)

# Top prognostic genes on full TCGA (train set for external validation)
has_rna  = mrna_tcga.notna().any(axis=1)
mrna_real = mrna_tcga[has_rna]
y_real    = y_tcga[has_rna.values]

gene_var  = mrna_real.var()
top_v     = gene_var.nlargest(2000).index
gene_ci_d = {}
for g in top_v:
    vals = mrna_real[g].fillna(mrna_real[g].median()).values
    if vals.std() < 1e-6: continue
    try:
        gc = ci(y_real['event'], y_real['time'], vals)
        gene_ci_d[g] = abs(gc - 0.5)
    except: pass

TOP_GENES = sorted(gene_ci_d, key=gene_ci_d.get, reverse=True)[:10]
mrna_med  = mrna_real[TOP_GENES].median()
print(f"  TCGA top genes: {', '.join(TOP_GENES)}")

# ================================================================
# 2. DOWNLOAD EXTERNAL DATASET (try in order)
# ================================================================

print("\n[2] Loading external GEO dataset...")

geo_data = None
for geo_id in ['GSE31210', 'GSE37745', 'GSE68465', 'GSE50081']:
    geo_data = load_geo_dataset(geo_id)
    if geo_data is not None:
        break

if geo_data is None:
    print("\n  All GEO downloads failed or returned too few patients.")
    print("  Generating synthetic validation report for paper structure.")
    # Create a minimal placeholder so the paper can still be written
    with open(OUTPUT_DIR / 'external_validation.txt', 'w') as f:
        f.write("External validation pending — GEO survival metadata not parseable automatically.\n")
        f.write("Manual download of GSE31210 clinical supplement required.\n")
        f.write("See notebooks/external_validation.py for instructions.\n")
    import sys; sys.exit(0)

ext_expr   = geo_data['expr_t']
ext_time   = geo_data['os_time']
ext_event  = geo_data['os_event']
GEO_ID     = geo_data['geo_id']
n_ext      = geo_data['n']
ev_ext     = geo_data['events']

print(f"\n  Using {GEO_ID}: n={n_ext}, events={ev_ext} ({ev_ext/n_ext*100:.1f}%)")

# ================================================================
# 3. MATCH mRNA GENES
# ================================================================

print("\n[3] Matching mRNA genes...")

common_genes = [g for g in TOP_GENES if g in ext_expr.columns]
if len(common_genes) < 2:
    # Expand search to top-100 TCGA genes
    top100 = sorted(gene_ci_d, key=gene_ci_d.get, reverse=True)[:100]
    common_genes = [g for g in top100 if g in ext_expr.columns][:10]

print(f"  Overlapping genes: {len(common_genes)}/{len(TOP_GENES)}  →  {common_genes}")

USE_GENES = common_genes

# ================================================================
# 4. BUILD FEATURE MATRICES
# ================================================================

print("\n[4] Building feature matrices...")

TRANSFER_CLIN = ['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE',
                 'PATH_M_STAGE','PATH_N_STAGE','PATH_T_STAGE']
clin_cols = [c for c in TRANSFER_CLIN if c in df.columns]

Xc_tcga = df[clin_cols].copy()
cat_c = Xc_tcga.select_dtypes(include=['object','category']).columns.tolist()
num_c = Xc_tcga.select_dtypes(include=['number','bool']).columns.tolist()

pre = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'), cat_c),
    ('num', SimpleImputer(strategy='median'), num_c),
], remainder='drop')

Xp_tcga = pre.fit_transform(Xc_tcga)
ohe = pre.named_transformers_['cat']
cn  = (ohe.get_feature_names_out(cat_c).tolist() if cat_c else []) + num_c
if len(cn) != Xp_tcga.shape[1]:
    cn = [f"f{i}" for i in range(Xp_tcga.shape[1])]

# External clinical (use available fields; fill unknown with train mode)
Xc_ext = pd.DataFrame(np.nan, index=ext_expr.index, columns=clin_cols)
pheno_ext = GEOparse.get_GEO(geo=GEO_ID, destdir=str(GEO_DIR), silent=True).phenotype_data

for col in pheno_ext.columns:
    coll = col.lower()
    vals = pheno_ext[col].astype(str)
    kv   = vals.str.extract(r'^([^:]+):\s*(.+)$')
    key_series = kv[0].str.lower().str.strip() if kv[0].notna().any() else pd.Series('', index=pheno_ext.index)
    val_series = kv[1].str.strip() if kv[1].notna().any() else vals

    age_mask  = key_series.str.contains('age',na=False)
    sex_mask  = key_series.str.contains('gender|sex',na=False)
    stg_mask  = key_series.str.contains('stage',na=False)

    if age_mask.any() and 'AGE' in Xc_ext.columns:
        Xc_ext.loc[pheno_ext.index.intersection(ext_expr.index),'AGE'] = \
            pd.to_numeric(val_series[age_mask], errors='coerce').reindex(ext_expr.index).values
    if sex_mask.any() and 'SEX' in Xc_ext.columns:
        Xc_ext.loc[pheno_ext.index.intersection(ext_expr.index),'SEX'] = \
            val_series[sex_mask].reindex(ext_expr.index).values
    if stg_mask.any() and 'AJCC_PATHOLOGIC_TUMOR_STAGE' in Xc_ext.columns:
        def map_stg(s):
            s = str(s).upper()
            if 'IV' in s or '4' in s: return 'STAGE IV'
            elif 'III' in s or '3' in s: return 'STAGE III'
            elif 'II' in s or '2' in s: return 'STAGE II'
            else: return 'STAGE I'
        Xc_ext.loc[pheno_ext.index.intersection(ext_expr.index),'AJCC_PATHOLOGIC_TUMOR_STAGE'] = \
            val_series[stg_mask].reindex(ext_expr.index).map(map_stg).values

# Fill remaining with train-set mode (categorical) or median (numeric)
for c in cat_c:
    if c in Xc_ext.columns:
        mode_val = Xc_tcga[c].mode()[0] if not Xc_tcga[c].mode().empty else 'Unknown'
        Xc_ext[c] = Xc_ext[c].fillna(mode_val)
for c in num_c:
    if c in Xc_ext.columns:
        Xc_ext[c] = pd.to_numeric(Xc_ext[c], errors='coerce').fillna(
            pd.to_numeric(Xc_tcga[c], errors='coerce').median())

Xp_ext = pre.transform(Xc_ext)

# mRNA features
if USE_GENES:
    ext_mrna_raw  = ext_expr[USE_GENES].copy()
    tcga_mrna_sel = mrna_tcga[USE_GENES].fillna(mrna_med[USE_GENES])
    mrna_sc = StandardScaler()
    mrna_sc.fit(mrna_real[USE_GENES].fillna(mrna_med[USE_GENES]))
    tcga_mrna_s = mrna_sc.transform(tcga_mrna_sel)
    ext_mrna_s  = mrna_sc.transform(ext_mrna_raw.fillna(
        mrna_real[USE_GENES].median()))
    X_tcga = np.hstack([Xp_tcga, tcga_mrna_s])
    X_ext  = np.hstack([Xp_ext,  ext_mrna_s])
    feat_names = cn + USE_GENES
else:
    X_tcga = Xp_tcga
    X_ext  = Xp_ext
    feat_names = cn

X_tcga = pd.DataFrame(X_tcga, columns=feat_names)
X_ext  = pd.DataFrame(X_ext,  columns=feat_names, index=ext_expr.index)
print(f"  Features: {X_tcga.shape[1]}")

# ================================================================
# 5. TRAIN SAGAM ON ALL TCGA
# ================================================================

print("\n[5] Training SAGAM on all 501 TCGA patients...")

y_s = np.array(list(zip(df['OS_event'].values, df['OS_time'].values)),
               dtype=[('event',bool),('time',float)])
y_ext_s = np.array(list(zip(ext_event.values, ext_time.values)),
                   dtype=[('event',bool),('time',float)])

# Early-stopping split (internal, no test leakage)
X_tr2,X_es,y_tr2,y_es = train_test_split(
    X_tcga, y_s, test_size=0.15, stratify=y_s['event'], random_state=SEED)

# CoxNet feature selection
cx_cv = KFold(3, shuffle=True, random_state=SEED)
ba,best_cx = 0.1,-1
for a in np.logspace(-2,2,12):
    fss=[]
    for ti,vi in cx_cv.split(X_tcga):
        try:
            m=CoxnetSurvivalAnalysis(alphas=[a],l1_ratio=0.9,max_iter=100_000,tol=1e-7)
            m.fit(X_tcga.iloc[ti].values, y_s[ti])
            fss.append(ci(y_s['event'][vi],y_s['time'][vi],
                          m.predict(X_tcga.iloc[vi].values)))
        except: pass
    if fss and np.mean(fss)>best_cx: best_cx,ba=np.mean(fss),a

cx_f = CoxnetSurvivalAnalysis(alphas=[ba],l1_ratio=0.9,max_iter=100_000,tol=1e-7)
cx_f.fit(X_tcga.values, y_s)
coefs = cx_f.coef_.ravel()
sel   = X_tcga.columns[np.abs(coefs)>1e-8].tolist()
if len(sel)<5: sel=X_tcga.columns[np.argsort(np.abs(coefs))[-10:]].tolist()
print(f"  CoxNet selected: {len(sel)} features")

sc = StandardScaler()
Xs = pd.DataFrame(sc.fit_transform(X_tcga[sel]), columns=sel)
Xe = pd.DataFrame(sc.transform(X_ext[sel]),  columns=sel, index=ext_expr.index)
Xtr2s = pd.DataFrame(sc.transform(X_tr2[sel]), columns=sel)
Xess  = pd.DataFrame(sc.transform(X_es[sel]),  columns=sel)

# Inner 5-fold OOF
META_FEATS = ["RSF","GBS","XGB","DS"]
kf  = KFold(5, shuffle=True, random_state=SEED)
oof = np.zeros((len(Xs),4))

for tr_i,vl_i in kf.split(Xs):
    Xi,Xj = Xs.iloc[tr_i].values, Xs.iloc[vl_i].values
    yi,yj = y_s[tr_i], y_s[vl_i]

    rsf=RandomSurvivalForest(n_estimators=300,max_features='sqrt',min_samples_leaf=5,random_state=SEED,n_jobs=-1)
    rsf.fit(Xi,yi); oof[vl_i,0]=rsf.predict(Xj)

    gbs=GradientBoostingSurvivalAnalysis(n_estimators=300,learning_rate=0.05,max_depth=3,random_state=SEED)
    gbs.fit(Xi,yi); oof[vl_i,1]=gbs.predict(Xj)

    dt=xgb.DMatrix(Xi,label=[e['time'] for e in yi],weight=[e['event'] for e in yi])
    dv=xgb.DMatrix(Xess.values,label=[e['time'] for e in y_es],weight=[e['event'] for e in y_es])
    xm=xgb.train(XGB_P,dt,num_boost_round=500,evals=[(dv,'v')],
                 early_stopping_rounds=30,verbose_eval=False)
    it=getattr(xm,'best_iteration',xm.num_boosted_rounds())
    oof[vl_i,2]=xm.predict(xgb.DMatrix(Xj),iteration_range=(0,it))

    dn=train_ds(Xi,yi,Xess.values,y_es,Xi.shape[1])
    oof[vl_i,3]=ds_pred(dn,Xj)

# Final models → external predictions
rsf_f=RandomSurvivalForest(n_estimators=300,max_features='sqrt',min_samples_leaf=5,random_state=SEED,n_jobs=-1)
rsf_f.fit(Xs.values,y_s); ext_rsf=rsf_f.predict(Xe.values)

gbs_f=GradientBoostingSurvivalAnalysis(n_estimators=300,learning_rate=0.05,max_depth=3,random_state=SEED)
gbs_f.fit(Xs.values,y_s); ext_gbs=gbs_f.predict(Xe.values)

dt_f=xgb.DMatrix(Xtr2s.values,label=[e['time'] for e in y_tr2],weight=[e['event'] for e in y_tr2])
dv_f=xgb.DMatrix(Xess.values, label=[e['time'] for e in y_es], weight=[e['event'] for e in y_es])
xm_f=xgb.train(XGB_P,dt_f,num_boost_round=500,evals=[(dv_f,'v')],
               early_stopping_rounds=30,verbose_eval=False)
it_f=getattr(xm_f,'best_iteration',xm_f.num_boosted_rounds())
ext_xgb=xm_f.predict(xgb.DMatrix(Xe.values),iteration_range=(0,it_f))

dn_f=train_ds(Xtr2s.values,y_tr2,Xess.values,y_es,Xs.shape[1])
ext_ds=ds_pred(dn_f,Xe.values)

ext_preds=np.column_stack([ext_rsf,ext_gbs,ext_xgb,ext_ds])

# Build meta-features
meta_tcga=pd.DataFrame(oof,columns=META_FEATS)
meta_ext=pd.DataFrame(ext_preds,columns=META_FEATS,index=ext_expr.index)
for f in META_FEATS:
    mn,mx=meta_tcga[f].min(),meta_tcga[f].max()
    meta_ext[f]=meta_ext[f].clip(mn,mx)

sp_tr,dis_list,mapping=build_splines(meta_tcga,META_FEATS)
sp_ext_parts=[]
for i,f in enumerate(META_FEATS):
    S=build_design_matrices([dis_list[i]],meta_ext)[0]
    sp_ext_parts.append(pd.DataFrame(S,index=meta_ext.index))
sp_ext=pd.concat(sp_ext_parts,axis=1)
sp_ext.columns=sp_tr.columns

# GAM alpha tuning
sp_tv,sp_vl=train_test_split(sp_tr,test_size=0.2,random_state=SEED)
ym_tv_s=np.array(list(zip(y_s[sp_tv.index]['event'],y_s[sp_tv.index]['time'])),
                 dtype=[('event',bool),('time',float)])
ym_vl_s=np.array(list(zip(y_s[sp_vl.index]['event'],y_s[sp_vl.index]['time'])),
                 dtype=[('event',bool),('time',float)])

a_gam=tune_alpha(sp_tv.values,ym_tv_s,sp_vl.values,ym_vl_s)
gam_f=coxnet_fit(sp_tr.values,y_s,a_gam)
ext_risk=gam_f.predict(sp_ext.values)

# Linear stacking
a_lin=tune_alpha(meta_tcga.loc[sp_tv.index].values,ym_tv_s,
                 meta_tcga.loc[sp_vl.index].values,ym_vl_s)
lin_f=coxnet_fit(meta_tcga.values,y_s,a_lin)
ext_lin=lin_f.predict(meta_ext.values)

# ================================================================
# 6. EVALUATE
# ================================================================

print("\n[6] External validation results...")

results = {
    'RSF':             ci(y_ext_s['event'],y_ext_s['time'],ext_rsf),
    'GBS':             ci(y_ext_s['event'],y_ext_s['time'],ext_gbs),
    'XGBoost-Cox':     ci(y_ext_s['event'],y_ext_s['time'],ext_xgb),
    'DeepSurv':        ci(y_ext_s['event'],y_ext_s['time'],ext_ds),
    'Linear Stacking': ci(y_ext_s['event'],y_ext_s['time'],ext_lin),
    'SAGAM (ours)':    ci(y_ext_s['event'],y_ext_s['time'],ext_risk),
}
for k,v in results.items(): print(f"  {k:<20}: {v:.4f}")

# Bootstrap
rng=np.random.default_rng(SEED)
boot_gam,boot_lin=[],[]
for _ in range(1000):
    idx=rng.choice(len(y_ext_s),len(y_ext_s),replace=True)
    boot_gam.append(ci(y_ext_s['event'][idx],y_ext_s['time'][idx],ext_risk[idx]))
    boot_lin.append(ci(y_ext_s['event'][idx],y_ext_s['time'][idx],ext_lin[idx]))

ci_glo,ci_ghi=np.percentile(boot_gam,[2.5,97.5])
ci_llo,ci_lhi=np.percentile(boot_lin,[2.5,97.5])
print(f"\n  SAGAM 95% CI:  [{ci_glo:.4f}, {ci_ghi:.4f}]")
print(f"  Linear 95% CI: [{ci_llo:.4f}, {ci_lhi:.4f}]")

# KM
risk_grp=pd.qcut(ext_risk,q=3,labels=['Low','Medium','High'])
lr_lh=logrank_test(y_ext_s['time'][risk_grp=='Low'],y_ext_s['time'][risk_grp=='High'],
                   y_ext_s['event'][risk_grp=='Low'],y_ext_s['event'][risk_grp=='High'])
lr_mv=multivariate_logrank_test(y_ext_s['time'],risk_grp,y_ext_s['event'])
sig=('***' if lr_lh.p_value<0.001 else '**' if lr_lh.p_value<0.01
     else '*' if lr_lh.p_value<0.05 else 'NS')

print(f"\n  Log-rank Low vs High: p={lr_lh.p_value:.4e} [{sig}]")
print(f"  Multivariate:          p={lr_mv.p_value:.4e}")

fig,ax=plt.subplots(figsize=(10,7))
colors=['#2E7D32','#F57C00','#C62828']
kmf=KaplanMeierFitter()
for i,(grp,mask) in enumerate([('Low',risk_grp=='Low'),
                                ('Medium',risk_grp=='Medium'),
                                ('High',risk_grp=='High')]):
    kmf.fit(y_ext_s['time'][mask],y_ext_s['event'][mask],
            label=f"{grp} Risk (n={mask.sum()})")
    kmf.plot_survival_function(ax=ax,ci_show=True,linewidth=2.5,color=colors[i],alpha=0.9)

ax.set_xlabel('Time (Months)',fontsize=13,fontweight='bold')
ax.set_ylabel('Survival Probability',fontsize=13,fontweight='bold')
ax.set_title(f'External Validation — {GEO_ID} (LUAD, n={n_ext})\n'
             f'SAGAM C-index: {results["SAGAM (ours)"]:.3f} [{ci_glo:.3f},{ci_ghi:.3f}]  |  '
             f'Log-rank p={lr_lh.p_value:.4f} [{sig}]',
             fontsize=12,fontweight='bold')
ax.grid(True,alpha=0.3,linestyle='--'); ax.set_ylim(0,1.05)
ax.legend(fontsize=11,loc='lower left')
ax.text(0.02,0.05,f'Significance: {sig}',transform=ax.transAxes,fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle='round',facecolor='lightyellow',edgecolor='black',alpha=0.9))
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'ext_kaplan_meier.png',dpi=300,bbox_inches='tight')
plt.close()

# Save
with open(OUTPUT_DIR/'external_validation.txt','w') as f:
    f.write(f"EXTERNAL VALIDATION — {GEO_ID}\n")
    f.write("="*50+"\n")
    f.write(f"n={n_ext}  events={ev_ext} ({ev_ext/n_ext*100:.1f}%)\n")
    f.write(f"Genes used: {', '.join(USE_GENES)}\n\n")
    for k,v in results.items(): f.write(f"  {k:<20}: {v:.4f}\n")
    f.write(f"\nSAGAM 95% CI:  [{ci_glo:.4f}, {ci_ghi:.4f}]\n")
    f.write(f"Linear 95% CI: [{ci_llo:.4f}, {ci_lhi:.4f}]\n")
    f.write(f"\nLog-rank Low vs High: p={lr_lh.p_value:.4e} [{sig}]\n")
    f.write(f"Multivariate:          p={lr_mv.p_value:.4e}\n")

pd.DataFrame({'boot_gam':boot_gam,'boot_lin':boot_lin}).to_csv(
    OUTPUT_DIR/'ext_bootstrap.csv',index=False)

print("\n"+"="*70)
print("DONE")
print("="*70)
print(f"  SAGAM C-index ({GEO_ID}): {results['SAGAM (ours)']:.4f}  [{ci_glo:.4f},{ci_ghi:.4f}]")
print(f"  KM p-value:               {lr_lh.p_value:.4e}  [{sig}]")
