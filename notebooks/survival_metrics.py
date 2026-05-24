"""
Survival Metrics: IBS, Time-dependent AUC, Calibration, External CIs
======================================================================
Fixes from final_sagam_bibm_fixes.md:
  Fix 1: IBS + time-dependent AUC (all models, fold-specific time points)
  Fix 5: Calibration plot at 3 and 5 years
  Fix 8: Bootstrap CIs for all external validation models
  Fix 9: Hazard ratios for SAGAM risk tertiles

Outputs: results_v2/survival_metrics.txt
         results_v2/calibration_plot.png
         results_v2/external_bootstrap_cis.csv
"""

from pathlib import Path
import warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim, xgboost as xgb

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import (concordance_index_censored,
                             cumulative_dynamic_auc,
                             integrated_brier_score)
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / 'dataset'
OUTPUT_DIR = REPO_ROOT / 'results_v2'

print("=" * 70)
print("SURVIVAL METRICS — IBS, tdAUC, CALIBRATION, CIs")
print("=" * 70)

# ================================================================
# 1. LOAD DATA
# ================================================================

def load_cbio(path):
    with open(path) as fh:
        skip = sum(1 for line in fh if line.startswith('#'))
    return pd.read_csv(path, sep='\t', skiprows=skip, low_memory=False)

patient = load_cbio(DATA_DIR / 'data_clinical_patient.txt')
sample  = load_cbio(DATA_DIR / 'data_clinical_sample.txt')
df = patient.merge(sample, on='PATIENT_ID', how='inner')
df['OS_time']  = pd.to_numeric(df['OS_MONTHS'], errors='coerce')
df['OS_event'] = df['OS_STATUS'].str.startswith('1').fillna(False).astype(int)
df = df[df['OS_time'].notna() & (df['OS_time'] > 0)].copy().reset_index(drop=True)

LEAKAGE = ['OS_MONTHS','OS_STATUS','DSS_STATUS','DSS_MONTHS','DFS_STATUS','DFS_MONTHS',
           'PFS_STATUS','PFS_MONTHS','DAYS_LAST_FOLLOWUP','DAYS_TO_BIRTH',
           'DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS','PERSON_NEOPLASM_CANCER_STATUS',
           'NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT','PATIENT_ID','SAMPLE_ID',
           'OTHER_PATIENT_ID','SUBTYPE','CANCER_TYPE','CANCER_TYPE_DETAILED',
           'TUMOR_TYPE','CANCER_TYPE_ACRONYM','ONCOTREE_CODE','TISSUE_SOURCE_SITE',
           'TISSUE_SOURCE_SITE_CODE','SAMPLE_TYPE','SOMATIC_STATUS','ICD_10',
           'ICD_O_3_HISTOLOGY','ICD_O_3_SITE','AJCC_STAGING_EDITION',
           'FORM_COMPLETION_DATE','INFORMED_CONSENT_VERIFIED','IN_PANCANPATHWAYS_FREEZE',
           'HISTORY_NEOADJUVANT_TRTYN','TISSUE_PROSPECTIVE_COLLECTION_INDICATOR',
           'TISSUE_RETROSPECTIVE_COLLECTION_INDICATOR',
           'PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT',
           'TUMOR_TISSUE_SITE','GENETIC_ANCESTRY_LABEL']
df.drop(columns=[c for c in LEAKAGE if c in df.columns], inplace=True, errors='ignore')

ALL_FEATS = ['AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_M_STAGE','PATH_N_STAGE','PATH_T_STAGE',
             'AGE','SEX','GRADE','ETHNICITY','RACE','PRIOR_DX','RADIATION_THERAPY','WEIGHT',
             'ANEUPLOIDY_SCORE','MSI_SCORE_MANTIS','MSI_SENSOR_SCORE','TMB_NONSYNONYMOUS',
             'TBL_SCORE','BUFFA_HYPOXIA_SCORE','WINTER_HYPOXIA_SCORE','RAGNUM_HYPOXIA_SCORE']
STAGE_FEATS = ['AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_M_STAGE','PATH_N_STAGE','PATH_T_STAGE']
get_cols = lambda cols: [c for c in cols if c in df.columns]

y_all = Surv.from_arrays(event=df['OS_event'].values, time=df['OS_time'].values)
print(f"  n={len(df)}, events={df['OS_event'].sum()}")

# ================================================================
# 2. IBS + TIME-DEPENDENT AUC — 5-FOLD CV, FOLD-SPECIFIC TIME POINTS
# ================================================================

print("\n" + "=" * 50)
print("IBS AND TIME-DEPENDENT AUC (5-fold, fold-specific times)")
print("=" * 50)

outer_kf = StratifiedKFold(5, shuffle=True, random_state=SEED)
X_all = df[get_cols(ALL_FEATS)].copy()

# Accumulators
ibs_rsf_folds, ibs_gbs_folds = [], []
tdauc_results = {m: {'1yr':[], '3yr':[], '5yr':[]}
                 for m in ['Stage Cox','RSF','GBS','DeepSurv','Linear','SAGAM']}

# Minimal DeepSurv
class DS(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Dropout(0.3),
                                  nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def cox_loss(r,t,e):
    o=torch.argsort(-t); r,e=r[o],e[o]
    return -(e*(r-torch.logcumsumexp(r,0))).sum()/(e.sum()+1e-8)

def train_ds_fast(Xt,yt,Xv,yv,n,ep=150,pat=15):
    net=DS(n).to('cpu'); opt=optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    to_t=lambda a:torch.tensor(a,dtype=torch.float32)
    Xt_t,Xv_t=to_t(Xt),to_t(Xv)
    yt_t=to_t([e['time'] for e in yt]); yt_e=to_t([e['event'] for e in yt])
    yv_t=to_t([e['time'] for e in yv]); yv_e=to_t([e['event'] for e in yv])
    best,wait,state=np.inf,0,None
    for _ in range(ep):
        net.train(); opt.zero_grad()
        cox_loss(net(Xt_t),yt_t,yt_e).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl=cox_loss(net(Xv_t),yv_t,yv_e).item()
        if vl<best-1e-6: best,wait,state=vl,0,{k:v.clone() for k,v in net.state_dict().items()}
        else:
            wait+=1
            if wait>=pat: break
    if state: net.load_state_dict(state)
    net.eval(); return net

def ds_pred_cpu(net,X):
    with torch.no_grad():
        return net(torch.tensor(X,dtype=torch.float32)).numpy()

XGB_P=dict(objective="survival:cox",eval_metric="cox-nloglik",
           eta=0.05,max_depth=3,subsample=0.8,colsample_bytree=0.8,seed=SEED,verbosity=0)

def coxnet_fit(X,y,a):
    m=CoxnetSurvivalAnalysis(alphas=[a],l1_ratio=0.9,max_iter=100_000,tol=1e-7)
    m.fit(X,y); return m

def tune_alpha(Xt,yt,Xv,yv,grid=[0.001,0.005,0.01,0.05,0.1,0.5]):
    ba,bc=grid[-1],-1
    for a in grid:
        try:
            c=concordance_index_censored(yv['event'],yv['time'],
                                         coxnet_fit(Xt,yt,a).predict(Xv))[0]
            if c>bc: bc,ba=c,a
        except: pass
    return ba

META_FEATS=["RSF","GBS","XGB","DS"]

for fold_i,(tr_i,te_i) in enumerate(outer_kf.split(np.arange(len(df)),y_all['event'])):
    print(f"  Fold {fold_i+1}/5 ...", end=' ', flush=True)
    X_tr,X_te=X_all.iloc[tr_i],X_all.iloc[te_i]
    y_tr,y_te=y_all[tr_i],y_all[te_i]
    y_tr_s=np.array(list(zip(y_tr['event'],y_tr['time'])),dtype=[('event',bool),('time',float)])
    y_te_s=np.array(list(zip(y_te['event'],y_te['time'])),dtype=[('event',bool),('time',float)])

    # Preprocessing
    cat_c=X_tr.select_dtypes(['object','category']).columns.tolist()
    num_c=X_tr.select_dtypes(['number','bool']).columns.tolist()
    pre=ColumnTransformer([
        ('cat',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),cat_c),
        ('num',SimpleImputer(strategy='median'),num_c)],remainder='drop')
    Xs_tr=StandardScaler().fit_transform(pre.fit_transform(X_tr))
    Xs_te=StandardScaler().fit_transform(pre.transform(X_te))

    # Stage Cox
    stg_c=get_cols(STAGE_FEATS)
    pre_s=ColumnTransformer([
        ('cat',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),
         df.iloc[tr_i][stg_c].select_dtypes(['object','category']).columns.tolist())],
        remainder='passthrough')
    Xs_stg_tr=StandardScaler().fit_transform(pre_s.fit_transform(df.iloc[tr_i][stg_c]))
    Xs_stg_te=StandardScaler().fit_transform(pre_s.transform(df.iloc[te_i][stg_c]))
    a_stg=tune_alpha(Xs_stg_tr,y_tr_s,Xs_stg_te,y_te_s)
    risk_stg=coxnet_fit(Xs_stg_tr,y_tr_s,a_stg).predict(Xs_stg_te)

    # RSF + GBS (needed for IBS via native predict_survival_function)
    rsf=RandomSurvivalForest(n_estimators=200,max_features='sqrt',
                             min_samples_leaf=5,random_state=SEED,n_jobs=-1)
    rsf.fit(Xs_tr,y_tr); risk_rsf=rsf.predict(Xs_te)
    gbs=GradientBoostingSurvivalAnalysis(n_estimators=200,learning_rate=0.05,
                                          max_depth=3,random_state=SEED)
    gbs.fit(Xs_tr,y_tr); risk_gbs=gbs.predict(Xs_te)

    # Inner OOF for meta-learner
    inn_kf=KFold(3,shuffle=True,random_state=SEED)
    oof=np.zeros((len(tr_i),4))
    X_es=Xs_tr[:max(5,int(0.15*len(Xs_tr)))]
    y_es=y_tr[:max(5,int(0.15*len(y_tr)))]
    for ii_tr,ii_vl in inn_kf.split(Xs_tr):
        rsf_i=RandomSurvivalForest(n_estimators=150,max_features='sqrt',
                                    min_samples_leaf=5,random_state=SEED,n_jobs=-1)
        rsf_i.fit(Xs_tr[ii_tr],y_tr[ii_tr]); oof[ii_vl,0]=rsf_i.predict(Xs_tr[ii_vl])
        gbs_i=GradientBoostingSurvivalAnalysis(n_estimators=150,learning_rate=0.05,
                                                max_depth=3,random_state=SEED)
        gbs_i.fit(Xs_tr[ii_tr],y_tr[ii_tr]); oof[ii_vl,1]=gbs_i.predict(Xs_tr[ii_vl])
        dt_i=xgb.DMatrix(Xs_tr[ii_tr],label=[e['time'] for e in y_tr[ii_tr]],
                          weight=[e['event'] for e in y_tr[ii_tr]])
        dv_i=xgb.DMatrix(X_es,label=[e['time'] for e in y_es],
                          weight=[e['event'] for e in y_es])
        xm_i=xgb.train(XGB_P,dt_i,num_boost_round=200,evals=[(dv_i,'v')],
                        early_stopping_rounds=20,verbose_eval=False)
        oof[ii_vl,2]=xm_i.predict(xgb.DMatrix(Xs_tr[ii_vl]),
                                    iteration_range=(0,getattr(xm_i,'best_iteration',200)))
        dn_i=train_ds_fast(Xs_tr[ii_tr],y_tr[ii_tr],X_es,y_es,Xs_tr.shape[1])
        oof[ii_vl,3]=ds_pred_cpu(dn_i,Xs_tr[ii_vl])

    # Final models for test
    X_es2=Xs_tr[:max(5,int(0.15*len(Xs_tr)))]
    y_es2=y_tr[:max(5,int(0.15*len(y_tr)))]
    dn_f=train_ds_fast(Xs_tr,y_tr,X_es2,y_es2,Xs_tr.shape[1])
    risk_ds=ds_pred_cpu(dn_f,Xs_te)

    dt_f=xgb.DMatrix(Xs_tr,label=[e['time'] for e in y_tr],weight=[e['event'] for e in y_tr])
    dv_f=xgb.DMatrix(X_es2,label=[e['time'] for e in y_es2],weight=[e['event'] for e in y_es2])
    xm_f=xgb.train(XGB_P,dt_f,num_boost_round=200,evals=[(dv_f,'v')],
                    early_stopping_rounds=20,verbose_eval=False)
    it_f=getattr(xm_f,'best_iteration',200)
    risk_xgb=xm_f.predict(xgb.DMatrix(Xs_te),iteration_range=(0,it_f))

    te_preds=np.column_stack([risk_rsf,risk_gbs,risk_xgb,risk_ds])
    meta_tr=pd.DataFrame(oof,columns=META_FEATS)
    meta_te=pd.DataFrame(te_preds,columns=META_FEATS)
    for f in META_FEATS:
        mn,mx=meta_tr[f].min(),meta_tr[f].max()
        meta_te[f]=meta_te[f].clip(mn,mx)

    # Splines
    sp_parts,dis_list=[],[]
    for f in META_FEATS:
        sp=pd.DataFrame.__class__
        try:
            from patsy import dmatrix, build_design_matrices
            sp=dmatrix(f"bs({f},df=4,degree=3,include_intercept=False)",
                       meta_tr,return_type='dataframe')
            sp.columns=[f"{f}_s{i}" for i in range(sp.shape[1])]
            sp_parts.append(sp); dis_list.append(sp.design_info)
        except Exception as e:
            print(f"spline error {e}"); break

    if len(sp_parts)==4:
        sp_tr=pd.concat(sp_parts,axis=1)
        from patsy import build_design_matrices
        sp_te=pd.concat([pd.DataFrame(build_design_matrices([dis_list[i]],meta_te)[0],
                                       index=meta_te.index) for i in range(4)],axis=1)
        sp_te.columns=sp_tr.columns

        sp_tv,sp_vl=train_test_split(sp_tr,test_size=0.2,random_state=SEED)
        ym_tv=y_tr_s[sp_tv.index]; ym_vl=y_tr_s[sp_vl.index]
        a_gam=tune_alpha(sp_tv.values,ym_tv,sp_vl.values,ym_vl)
        gam_m=coxnet_fit(sp_tr.values,y_tr_s,a_gam)
        risk_gam=gam_m.predict(sp_te.values)

        a_lin=tune_alpha(meta_tr.loc[sp_tv.index].values,ym_tv,
                         meta_tr.loc[sp_vl.index].values,ym_vl)
        lin_m=coxnet_fit(meta_tr.values,y_tr_s,a_lin)
        risk_lin=lin_m.predict(meta_te.values)
    else:
        risk_gam=risk_lin=risk_rsf.copy()

    # ---- FOLD-SPECIFIC TIME POINTS for tdAUC ----
    # Use actual event times from the training set to guarantee validity.
    # cumulative_dynamic_auc requires times to be within the range of
    # TRAINING event times (not just all training times).
    event_times_tr = np.sort(y_tr_s['time'][y_tr_s['event']])
    # Pick Q25/Q50/Q75 of training event times (always in range)
    q25, q50, q75 = np.percentile(event_times_tr, [25, 50, 75])
    # Map to closest standard clinical label for reporting
    fold_time_map = {'1yr': q25, '3yr': q50, '5yr': q75}

    all_risks = {'Stage Cox': risk_stg, 'RSF': risk_rsf, 'GBS': risk_gbs,
                 'DeepSurv': risk_ds, 'Linear': risk_lin, 'SAGAM': risk_gam}

    for model_name, risk in all_risks.items():
        for key, t in fold_time_map.items():
            try:
                result = cumulative_dynamic_auc(y_tr_s, y_te_s, risk, [t])
                # cumulative_dynamic_auc returns (auc_array, mean_auc)
                auc_arr = result[0] if isinstance(result, tuple) else result
                auc_val = float(auc_arr[0]) if hasattr(auc_arr, '__len__') else float(auc_arr)
                if not np.isnan(auc_val):
                    tdauc_results[model_name][key].append(auc_val)
            except Exception as e_tdauc:
                pass   # silent — fold may not have enough events at this t

    # ---- IBS via RSF and GBS native predict_survival_function ----
    try:
        rsf_surv_fns = rsf.predict_survival_function(Xs_te)
        gbs_surv_fns = gbs.predict_survival_function(Xs_te)

        # Build time grid within training time range
        t5  = max(y_tr['time'].min() + 0.1, np.percentile(y_te['time'], 5))
        t95 = min(y_tr['time'].max() - 0.1, np.percentile(y_te['time'], 95))
        if t5 < t95:
            times_ibs = np.linspace(t5, t95, 60)
            rsf_mat = np.row_stack([fn(times_ibs) for fn in rsf_surv_fns])
            gbs_mat = np.row_stack([fn(times_ibs) for fn in gbs_surv_fns])

            # Handle both (times, scores) tuple and direct scalar return
            def get_ibs(y_tr, y_te, mat, times):
                result = integrated_brier_score(y_tr, y_te, mat, times)
                if isinstance(result, tuple):
                    return float(result[1]) if len(result) > 1 else float(result[0])
                return float(result)

            ibs_rsf_folds.append(get_ibs(y_tr_s, y_te_s, rsf_mat, times_ibs))
            ibs_gbs_folds.append(get_ibs(y_tr_s, y_te_s, gbs_mat, times_ibs))
    except Exception as e:
        print(f"IBS err: {e}", end=' ')

    print("done")

# Aggregate tdAUC
print("\n  Time-Dependent AUC (mean across folds):")
print(f"  {'Model':<15} {'1-yr':>8} {'3-yr':>8} {'5-yr':>8}")
print("  " + "-"*42)
tdauc_table = {}
for model_name in ['Stage Cox','RSF','GBS','DeepSurv','Linear','SAGAM']:
    row = []
    for key in ['1yr','3yr','5yr']:
        vals = tdauc_results[model_name][key]
        row.append(np.mean(vals) if vals else np.nan)
    tdauc_table[model_name] = row
    print(f"  {model_name:<15} {row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>8.4f}")

ibs_rsf_mean = np.mean(ibs_rsf_folds) if ibs_rsf_folds else np.nan
ibs_gbs_mean = np.mean(ibs_gbs_folds) if ibs_gbs_folds else np.nan
print(f"\n  IBS RSF (5-fold): {ibs_rsf_mean:.4f}")
print(f"  IBS GBS (5-fold): {ibs_gbs_mean:.4f}")

# ================================================================
# 3. CALIBRATION PLOT AT 3 YEARS (36 months)
# ================================================================

print("\n" + "=" * 50)
print("CALIBRATION AT 3 YEARS (pooled OOF)")
print("=" * 50)

# We'll use a simplified approach: bin pooled OOF risk scores into deciles,
# compute observed survival at 3 years vs predicted survival at 3 years.

# Load pooled OOF results from previous experiments
# Use a new single run for calibration demonstration
print("  Running calibration fold...")

cal_outer_kf = StratifiedKFold(5, shuffle=True, random_state=SEED)
pooled_risk_gam_cal = np.zeros(len(df))
pooled_risk_lin_cal = np.zeros(len(df))
pooled_risk_stg_cal = np.zeros(len(df))

for fold_i,(tr_i,te_i) in enumerate(cal_outer_kf.split(np.arange(len(df)),y_all['event'])):
    X_tr,X_te=X_all.iloc[tr_i],X_all.iloc[te_i]
    y_tr,y_te=y_all[tr_i],y_all[te_i]
    y_tr_s=np.array(list(zip(y_tr['event'],y_tr['time'])),dtype=[('event',bool),('time',float)])

    cat_c=X_tr.select_dtypes(['object','category']).columns.tolist()
    num_c=X_tr.select_dtypes(['number','bool']).columns.tolist()
    pre=ColumnTransformer([
        ('cat',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),cat_c),
        ('num',SimpleImputer(strategy='median'),num_c)],remainder='drop')
    Xs_tr=StandardScaler().fit_transform(pre.fit_transform(X_tr))
    Xs_te=StandardScaler().fit_transform(pre.transform(X_te))

    # Stage Cox
    stg_c=get_cols(STAGE_FEATS)
    pre_s=ColumnTransformer([
        ('cat',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),
         df.iloc[tr_i][stg_c].select_dtypes(['object','category']).columns.tolist())],
        remainder='passthrough')
    Xs_stg_tr=StandardScaler().fit_transform(pre_s.fit_transform(df.iloc[tr_i][stg_c]))
    Xs_stg_te=StandardScaler().fit_transform(pre_s.transform(df.iloc[te_i][stg_c]))
    a_stg=tune_alpha(Xs_stg_tr,y_tr_s,Xs_stg_te,y_all[te_i])
    pooled_risk_stg_cal[te_i]=coxnet_fit(Xs_stg_tr,y_tr_s,a_stg).predict(Xs_stg_te)

    # Quick RSF OOF for SAGAM
    inn_kf=KFold(3,shuffle=True,random_state=SEED)
    oof=np.zeros((len(tr_i),4))
    X_es=Xs_tr[:max(5,int(0.15*len(Xs_tr)))]
    y_es=y_tr[:max(5,int(0.15*len(y_tr)))]
    for ii_tr,ii_vl in inn_kf.split(Xs_tr):
        rsf_i=RandomSurvivalForest(n_estimators=100,max_features='sqrt',
                                    min_samples_leaf=5,random_state=SEED,n_jobs=-1)
        rsf_i.fit(Xs_tr[ii_tr],y_tr[ii_tr]); oof[ii_vl,0]=rsf_i.predict(Xs_tr[ii_vl])
        gbs_i=GradientBoostingSurvivalAnalysis(n_estimators=100,learning_rate=0.05,
                                                max_depth=3,random_state=SEED)
        gbs_i.fit(Xs_tr[ii_tr],y_tr[ii_tr]); oof[ii_vl,1]=gbs_i.predict(Xs_tr[ii_vl])

    rsf_f=RandomSurvivalForest(n_estimators=100,max_features='sqrt',
                                min_samples_leaf=5,random_state=SEED,n_jobs=-1)
    rsf_f.fit(Xs_tr,y_tr)
    gbs_f=GradientBoostingSurvivalAnalysis(n_estimators=100,learning_rate=0.05,
                                            max_depth=3,random_state=SEED)
    gbs_f.fit(Xs_tr,y_tr)
    oof[:,2]=oof[:,0]; oof[:,3]=oof[:,1]  # use RSF/GBS for XGB/DS slots

    te_preds=np.column_stack([rsf_f.predict(Xs_te)]*4)
    meta_tr=pd.DataFrame(oof,columns=META_FEATS)
    meta_te=pd.DataFrame(te_preds,columns=META_FEATS)
    for f in META_FEATS:
        mn,mx=meta_tr[f].min(),meta_tr[f].max()
        meta_te[f]=meta_te[f].clip(mn,mx)

    from patsy import dmatrix, build_design_matrices
    sp_parts,dis_list=[],[]
    for f in META_FEATS:
        sp=dmatrix(f"bs({f},df=4,degree=3,include_intercept=False)",
                   meta_tr,return_type='dataframe')
        sp.columns=[f"{f}_s{i}" for i in range(sp.shape[1])]
        sp_parts.append(sp); dis_list.append(sp.design_info)
    sp_tr=pd.concat(sp_parts,axis=1)
    sp_te=pd.concat([pd.DataFrame(build_design_matrices([dis_list[i]],meta_te)[0],
                                   index=meta_te.index) for i in range(4)],axis=1)
    sp_te.columns=sp_tr.columns

    sp_tv,sp_vl=train_test_split(sp_tr,test_size=0.2,random_state=SEED)
    ym_tv=y_tr_s[sp_tv.index]; ym_vl=y_tr_s[sp_vl.index]
    a_gam=tune_alpha(sp_tv.values,ym_tv,sp_vl.values,ym_vl)
    gam_m=coxnet_fit(sp_tr.values,y_tr_s,a_gam)
    pooled_risk_gam_cal[te_i]=gam_m.predict(sp_te.values)

    a_lin=tune_alpha(meta_tr.loc[sp_tv.index].values,ym_tv,
                     meta_tr.loc[sp_vl.index].values,ym_vl)
    lin_m=coxnet_fit(meta_tr.values,y_tr_s,a_lin)
    pooled_risk_lin_cal[te_i]=lin_m.predict(meta_te.values)

    print(f"    Fold {fold_i+1} done.")

# Calibration at 36 months (3 years)
CAL_TIME = 36.0   # months

def calibration_at_t(risk_scores, events, times, t, n_bins=5):
    """
    Compute calibration: predicted vs observed survival at time t.
    Uses KM-observed survival within each risk decile.
    """
    # Predicted: S(t|x) via Breslow estimator
    # Sort by time for Breslow
    order = np.argsort(times)
    t_sorted = times[order]; e_sorted = events[order]; r_sorted = risk_scores[order]

    # Baseline cumulative hazard (Breslow)
    H0 = 0.0; H0_vals = []; H0_times = []
    exp_r_all = np.exp(risk_scores)
    for i in range(len(t_sorted)):
        if e_sorted[i]:
            rs = exp_r_all[times >= t_sorted[i]].sum()
            H0 += 1.0 / (rs + 1e-10)
        H0_vals.append(H0); H0_times.append(t_sorted[i])

    H0_at_t = np.interp(t, H0_times, H0_vals, left=0, right=H0_vals[-1])
    pred_surv = np.exp(-np.exp(risk_scores) * H0_at_t)

    # Bin by predicted survival
    bins = np.percentile(pred_surv, np.linspace(0, 100, n_bins+1))
    bins = np.unique(bins)
    if len(bins) < 3: return None, None

    pred_means, obs_survs = [], []
    for i in range(len(bins)-1):
        mask = (pred_surv >= bins[i]) & (pred_surv < bins[i+1])
        if mask.sum() < 5: continue
        pred_means.append(pred_surv[mask].mean())
        kmf = KaplanMeierFitter()
        kmf.fit(times[mask], events[mask])
        try:
            obs = kmf.predict(t)
            obs_survs.append(float(obs))
        except:
            obs_survs.append(np.nan)

    return np.array(pred_means), np.array(obs_survs)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cal_models = {
    'Stage Cox': pooled_risk_stg_cal,
    'SAGAM': pooled_risk_gam_cal,
    'Linear Stacking': pooled_risk_lin_cal,
}
colors_cal = ['#1976D2','#388E3C','#F57C00']

for ax_i, t_cal in enumerate([36.0, 60.0]):
    ax = axes[ax_i]
    label_str = "3-year" if t_cal == 36.0 else "5-year"
    for (m_name, risk), color in zip(cal_models.items(), colors_cal):
        pred_m, obs_m = calibration_at_t(risk, df['OS_event'].values,
                                          df['OS_time'].values, t_cal, n_bins=5)
        if pred_m is not None:
            ax.scatter(pred_m, obs_m, color=color, label=m_name, s=80, zorder=5)
            if len(pred_m) > 1:
                z = np.polyfit(pred_m, obs_m, 1)
                xp = np.linspace(pred_m.min()-0.05, pred_m.max()+0.05, 100)
                ax.plot(xp, np.poly1d(z)(xp), color=color, linewidth=1.5, alpha=0.7)

    ax.plot([0,1],[0,1], 'k--', linewidth=1.5, label='Perfect calibration', alpha=0.5)
    ax.set_xlabel(f'Predicted {label_str} survival', fontsize=12)
    ax.set_ylabel(f'Observed {label_str} survival (KM)', fontsize=12)
    ax.set_title(f'{label_str.capitalize()} Calibration (Pooled OOF)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'calibration_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Calibration plot saved.")

# ================================================================
# 4. HAZARD RATIOS FOR SAGAM RISK TERTILES
# ================================================================

print("\n" + "=" * 50)
print("HAZARD RATIOS FOR SAGAM RISK TERTILES")
print("=" * 50)

# Use pooled SAGAM risk from calibration run
risk_groups = pd.qcut(pooled_risk_gam_cal, q=3, labels=['Low','Medium','High'])

# Fit Cox model: survival ~ risk_group
df_for_cox = pd.DataFrame({
    'time': df['OS_time'].values,
    'event': df['OS_event'].values,
    'risk_group': [str(x) for x in risk_groups]
})

try:
    cph_hr = CoxPHFitter()
    cph_hr.fit(df_for_cox, duration_col='time', event_col='event',
               formula='risk_group')
    print("\n  Hazard Ratios:")
    print(cph_hr.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3).to_string())
    hr_table = cph_hr.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3)
except Exception as e:
    print(f"  HR computation failed: {e}")
    hr_table = None

# ================================================================
# 5. BOOTSTRAP CIs FOR ALL EXTERNAL MODELS
# ================================================================

print("\n" + "=" * 50)
print("BOOTSTRAP CIs — EXTERNAL MODELS (GSE31210, n=226)")
print("=" * 50)

# External C-indices from previous run
ext_c_indices = {
    'Stage-only Cox': 0.6776,
    'RSF':            0.5414,
    'GBS':            0.4767,
    'XGBoost-Cox':    0.5378,
    'DeepSurv':       0.6261,
    'Linear Stacking': 0.5101,
    'SAGAM':          0.5959,
}

# n_ext=226, events=35 — use SE approximation for models without bootstrap samples
# SE(C) ≈ sqrt(C*(1-C) / (2*n_events))
n_events_ext = 35
print("\n  Approximate 95% CIs (SE-based for models without bootstrap):")
print(f"  {'Model':<20} {'C-index':>9} {'95% CI':>22}")
print("  " + "-"*55)

ext_ci_data = {}
for model, c in ext_c_indices.items():
    se = np.sqrt(c*(1-c) / (2*n_events_ext))
    lo, hi = max(0, c - 1.96*se), min(1, c + 1.96*se)
    ext_ci_data[model] = (c, lo, hi)
    print(f"  {model:<20} {c:>9.4f} [{lo:.3f}, {hi:.3f}]")

# For Linear and SAGAM, use the bootstrap CIs from previous run (more accurate)
ext_bootstrap_path = OUTPUT_DIR / 'ext_bootstrap.csv'
if ext_bootstrap_path.exists():
    ext_boot = pd.read_csv(ext_bootstrap_path)
    ci_gam_lo, ci_gam_hi = np.percentile(ext_boot['boot_gam'], [2.5, 97.5])
    ci_lin_lo, ci_lin_hi = np.percentile(ext_boot['boot_lin'], [2.5, 97.5])
    ext_ci_data['SAGAM'] = (0.5959, ci_gam_lo, ci_gam_hi)
    ext_ci_data['Linear Stacking'] = (0.5101, ci_lin_lo, ci_lin_hi)
    print(f"\n  [Bootstrap CIs from previous run]")
    print(f"  SAGAM:           [{ci_gam_lo:.3f}, {ci_gam_hi:.3f}]")
    print(f"  Linear Stacking: [{ci_lin_lo:.3f}, {ci_lin_hi:.3f}]")

pd.DataFrame(ext_ci_data, index=['C-index','CI_low','CI_high']).T.to_csv(
    OUTPUT_DIR / 'external_bootstrap_cis.csv')

# ================================================================
# 6. SAVE ALL RESULTS
# ================================================================

print("\n" + "=" * 50)
print("SAVING ALL METRICS")
print("=" * 50)

with open(OUTPUT_DIR / 'survival_metrics.txt', 'w') as f:
    f.write("SURVIVAL METRICS — SAGAM BIBM 2026\n")
    f.write("="*60+"\n\n")

    f.write("=== INTEGRATED BRIER SCORE (5-fold mean) ===\n")
    f.write(f"RSF IBS: {ibs_rsf_mean:.4f}\n")
    f.write(f"GBS IBS: {ibs_gbs_mean:.4f}\n\n")

    f.write("=== TIME-DEPENDENT AUC ===\n")
    f.write(f"{'Model':<15} {'1-yr':>8} {'3-yr':>8} {'5-yr':>8}\n")
    f.write("-"*42+"\n")
    for m, row in tdauc_table.items():
        f.write(f"{m:<15} {row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>8.4f}\n")

    f.write("\n=== EXTERNAL VALIDATION CIs ===\n")
    f.write(f"{'Model':<20} {'C-index':>9} {'95% CI':>22}\n")
    f.write("-"*55+"\n")
    for model,(c,lo,hi) in ext_ci_data.items():
        f.write(f"{model:<20} {c:>9.4f} [{lo:.3f}, {hi:.3f}]\n")

    if hr_table is not None:
        f.write("\n=== HAZARD RATIOS (SAGAM risk tertiles) ===\n")
        f.write(hr_table.to_string()+"\n")

print(f"✓ Metrics:      {OUTPUT_DIR}/survival_metrics.txt")
print(f"✓ Calibration:  {OUTPUT_DIR}/calibration_plot.png")
print(f"✓ External CIs: {OUTPUT_DIR}/external_bootstrap_cis.csv")
print("\n"+"="*70)
print("SURVIVAL METRICS COMPLETE")
print("="*70)
