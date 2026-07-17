"""
Rigor checks for reviewer critique:
 #1 paired-bootstrap CIs on the Table II deltas (5-feat vs 6-feat, per model)
 #5 paired-bootstrap CIs on GAM vs best ML (5-feat)
 #2 clinical-only (age,sex,stage) and stage-only external baselines
 #4 case-mix: stage distribution & event rate by stage, both cohorts
 #10 DeepSurv external seed variance (5 seeds)
Produces per-patient external risk vectors, then paired bootstraps.
"""
import warnings, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from patsy import dmatrix, build_design_matrices
warnings.filterwarnings("ignore"); SEED=42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED); dev=torch.device("cpu")
from pathlib import Path
D=lambda f: str(Path(__file__).resolve().parent.parent / "data" / f)

def coarsen(s):
    s=str(s).upper().replace("STAGE","").strip()
    if s in("","NAN","NA"):return np.nan
    if s.startswith("IV")or s=="4":return "IV"
    if s.startswith("III"):return "III"
    if s.startswith("II"):return "II"
    if s.startswith("I")or s=="1":return "I"
    return np.nan
def harm(df,c):
    o=pd.DataFrame(index=df.index)
    o["AGE"]=pd.to_numeric(df[c["AGE"]],errors="coerce")
    o["SEX"]=df[c["SEX"]].astype(str).str.strip().str.capitalize()
    o["STAGE"]=df[c["STAGE"]].map(coarsen)
    o["TMB"]=pd.to_numeric(df[c["TMB"]],errors="coerce")
    o["MUT_COUNT"]=pd.to_numeric(df[c["MUT_COUNT"]],errors="coerce")
    o["FGA"]=pd.to_numeric(df[c["FGA"]],errors="coerce")
    o["OS_time"]=pd.to_numeric(df[c["OS_time"]],errors="coerce")
    o["OS_event"]=df[c["OS_status"]].astype(str).str.strip().str.startswith("1").astype(int)
    o=o.dropna(subset=["OS_time","OS_event"]); o=o[o["OS_time"]>0]
    for cc in ["SEX","STAGE"]:o[cc]=o[cc].where(o[cc].notna(),"Unknown").replace({"":"Unknown"})
    return o.reset_index(drop=True)
tcga=harm(pd.read_csv(D("luad_tcga_pan_can_atlas_2018_clinical_data.csv")),
  {"AGE":"Diagnosis Age","SEX":"Sex","STAGE":"Neoplasm Disease Stage American Joint Committee on Cancer Code",
   "TMB":"TMB (nonsynonymous)","MUT_COUNT":"Mutation Count","FGA":"Fraction Genome Altered",
   "OS_time":"Overall Survival (Months)","OS_status":"Overall Survival Status"})
onco=harm(pd.read_csv(D("luad_oncosg_2020_clinical_data.csv")),
  {"AGE":"AGE","SEX":"SEX","STAGE":"STAGE","TMB":"TMB_NONSYNONYMOUS","MUT_COUNT":"MUTATION_COUNT",
   "FGA":"FRACTION_GENOME_ALTERED","OS_time":"OS_MONTHS","OS_status":"OS_STATUS"})
ytr=Surv.from_arrays(event=tcga["OS_event"].astype(bool),time=tcga["OS_time"].values)
ev,ti=onco["OS_event"].values.astype(bool),onco["OS_time"].values

class DS(nn.Module):
    def __init__(s,n):
        super().__init__();s.net=nn.Sequential(nn.Linear(n,128),nn.ReLU(),nn.Dropout(.3),
            nn.Linear(128,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,1))
    def forward(s,x):return s.net(x).squeeze(-1)
def closs(r,t,e):
    o=torch.argsort(-t);r,e=r[o],e[o];return -(e*(r-torch.logcumsumexp(r,0))).sum()/(e.sum()+1e-8)
def train_ds(Xt,yt,Xe,ye,n,seed=SEED):
    torch.manual_seed(seed)
    net=DS(n).to(dev);opt=optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    tt=lambda a:torch.tensor(a,dtype=torch.float32).to(dev);Xtt,Xet=tt(Xt),tt(Xe)
    yt_t,yt_e=tt([e["time"]for e in yt]),tt([e["event"]for e in yt]);ye_t,ye_e=tt([e["time"]for e in ye]),tt([e["event"]for e in ye])
    best,wait,st=np.inf,0,None
    for _ in range(400):
        net.train();opt.zero_grad();closs(net(Xtt),yt_t,yt_e).backward();opt.step();net.eval()
        with torch.no_grad():vl=closs(net(Xet),ye_t,ye_e).item()
        if vl<best-1e-6:best,wait,st=vl,0,{k:v.cpu().clone()for k,v in net.state_dict().items()}
        else:
            wait+=1
            if wait>=25:break
    if st:net.load_state_dict(st)
    net.eval();return net
def dsp(net,X):
    with torch.no_grad():return net(torch.tensor(X,dtype=torch.float32).to(dev)).cpu().numpy()
XGBP=dict(objective="survival:cox",eval_metric="cox-nloglik",eta=.05,max_depth=3,subsample=.8,colsample_bytree=.8,seed=SEED,verbosity=0)
def xlab(y):return [e["time"] if e["event"] else -e["time"] for e in y]

def prep(NUM):
    CAT=["SEX","STAGE"]
    ohe=OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore").fit(tcga[CAT])
    imp=SimpleImputer(strategy="median").fit(tcga[NUM]) if NUM else None
    def build(df):
        parts=[ohe.transform(df[CAT])]
        if NUM: parts.append(imp.transform(df[NUM]))
        return np.hstack(parts)
    Xt=build(tcga); Xe=build(onco)
    mu,sd=Xt.mean(0),Xt.std(0)+1e-8
    return (Xt-mu)/sd,(Xe-mu)/sd

def base_preds(NUM):
    Xt,Xe=prep(NUM)
    es_tr,es_vl=train_test_split(np.arange(len(Xt)),test_size=.2,random_state=SEED,stratify=ytr["event"])
    rsf=RandomSurvivalForest(n_estimators=500,max_features="sqrt",min_samples_leaf=3,random_state=SEED,n_jobs=-1).fit(Xt,ytr)
    gbs=GradientBoostingSurvivalAnalysis(n_estimators=500,learning_rate=.05,max_depth=3,random_state=SEED).fit(Xt,ytr)
    xm=xgb.train(XGBP,xgb.DMatrix(Xt[es_tr],label=xlab(ytr[es_tr])),num_boost_round=2000,
        evals=[(xgb.DMatrix(Xt[es_vl],label=xlab(ytr[es_vl])),"v")],early_stopping_rounds=100,verbose_eval=False)
    it=(xm.best_iteration+1) if hasattr(xm,"best_iteration") else xm.num_boosted_rounds()
    net=train_ds(Xt[es_tr],ytr[es_tr],Xt[es_vl],ytr[es_vl],Xt.shape[1])
    return {"RSF":rsf.predict(Xe),"GBS":gbs.predict(Xe),
            "XGB":xm.predict(xgb.DMatrix(Xe),iteration_range=(0,it)),"DeepSurv":dsp(net,Xe)}, (Xt,Xe,es_vl)

def gam_pred(CONT,LOG):
    CAT=["SEX","STAGE"]
    ohe=OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore").fit(tcga[CAT])
    dinfo={}
    def design(df,fit):
        parts=[]
        for f in CONT:
            v=pd.to_numeric(df[f],errors="coerce")
            if f in LOG: v=np.log1p(v)
            v=v.fillna(v.median() if v.notna().any() else 0)
            if fit:
                mu,sd=v.mean(),v.std()+1e-8; dinfo[f+"_ms"]=(mu,sd); z=(v-mu)/sd
                sp=dmatrix("bs(x, df=4, degree=3, include_intercept=False) - 1",{"x":z.values},return_type="dataframe")
                dinfo[f]=(sp.design_info,float(z.min()),float(z.max()))
            else:
                mu,sd=dinfo[f+"_ms"]; z=((v-mu)/sd)
                di,zmn,zmx=dinfo[f]; z=z.clip(zmn,zmx); sp=pd.DataFrame(build_design_matrices([di],{"x":z.values})[0])
            sp.columns=[f"{f}_s{i}"for i in range(sp.shape[1])];sp.index=df.index;parts.append(sp)
        parts.append(pd.DataFrame(ohe.transform(df[CAT]),index=df.index))
        return pd.concat(parts,axis=1) if parts else pd.DataFrame(index=df.index)
    Gt=design(tcga,True).values; Ge=design(onco,False).values
    yts=np.array(list(zip(tcga["OS_event"].astype(bool),tcga["OS_time"])),dtype=[("event",bool),("time",float)])
    m=CoxnetSurvivalAnalysis(alphas=[0.001],l1_ratio=0.5,max_iter=200000).fit(Gt,yts)
    return m.predict(Ge)

print("Training models (6-feat, 5-feat, clinical, stage-only)...",flush=True)
b6,_=base_preds(["AGE","TMB","MUT_COUNT","FGA"]); g6=gam_pred(["AGE","TMB","MUT_COUNT","FGA"],{"TMB","MUT_COUNT"})
b5,_=base_preds(["AGE","TMB","FGA"]);            g5=gam_pred(["AGE","TMB","FGA"],{"TMB"})
g_clin=gam_pred(["AGE"],set())        # age spline + stage + sex
g_stage=gam_pred([],set())            # stage + sex only (no continuous)
def C(risk):return concordance_index_censored(ev,ti,risk)[0]

def boot_diff(rA,rB,n=2000):
    rng=np.random.default_rng(SEED);d=[]
    for _ in range(n):
        idx=rng.choice(len(ev),len(ev),replace=True)
        if ev[idx].sum()==0:continue
        d.append(concordance_index_censored(ev[idx],ti[idx],rA[idx])[0]-concordance_index_censored(ev[idx],ti[idx],rB[idx])[0])
    d=np.array(d);return d.mean(),np.percentile(d,[2.5,97.5]),float((d>0).mean())

print("\n#1 PAIRED-BOOTSTRAP on Table II deltas (C 5feat - C 6feat):")
for k in ["RSF","GBS","XGB","DeepSurv"]:
    m,ci,p=boot_diff(b5[k],b6[k]); print(f"  {k:<10} delta={m:+.4f}  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  P(>0)={p:.3f}")
m,ci,p=boot_diff(g5,g6); print(f"  {'GAM':<10} delta={m:+.4f}  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  P(>0)={p:.3f}")

print("\n#5 PAIRED-BOOTSTRAP GAM vs best ML (5-feat, C_model - C_GAM):")
for k in ["XGB","DeepSurv","RSF"]:
    m,ci,p=boot_diff(b5[k],g5); print(f"  {k:<10}-GAM  diff={m:+.4f}  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  P(model>GAM)={p:.3f}")

print("\n#2 CLINICAL/STAGE BASELINES (external C):")
print(f"  Stage+Sex only          : {C(g_stage):.4f}")
print(f"  Clinical (age+stage+sex): {C(g_clin):.4f}")
print(f"  + TMB,FGA (5-feat GAM)  : {C(g5):.4f}")
print(f"  Best ML (XGB 5-feat)    : {C(b5['XGB']):.4f}")

print("\n#4 CASE-MIX (stage distribution & event rate):")
for name,df in [("TCGA",tcga),("OncoSG",onco)]:
    vc=df["STAGE"].value_counts(normalize=True).reindex(["I","II","III","IV"]).round(3)
    er={s:round(df[df.STAGE==s]["OS_event"].mean(),3) for s in ["I","II","III","IV"]}
    print(f"  {name}: stage%={vc.to_dict()}  event-rate-by-stage={er}")

print("\n#HEADLINE PAIRED-BOOTSTRAP: ML/feature vs baselines (external, C_A - C_B):")
for name, rA, rB in [
    ("XGB  - Stage+Sex", b5['XGB'], g_stage),
    ("XGB  - Clinical ", b5['XGB'], g_clin),
    ("Clinical - Stage+Sex (age)", g_clin, g_stage),
    ("FullGAM - Clinical (genomics)", g5, g_clin)]:
    m,ci,p=boot_diff(rA,rB)
    print(f"  {name:<30} diff={m:+.4f}  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  P(A>B)={p:.3f}")

print("\n#CASE-MIX DEMONSTRATION: external C reweighted to TCGA's stage mix:")
tcga_prop=tcga['STAGE'].value_counts(normalize=True)
onco_prop=onco['STAGE'].value_counts(normalize=True)
w=onco['STAGE'].map(lambda s: tcga_prop.get(s,0)/onco_prop.get(s,1e-9)).values
w=np.nan_to_num(w); w=w/w.sum()
def reweighted_C(risk,n=2000):
    rng=np.random.default_rng(SEED); cs=[]
    for _ in range(n):
        idx=rng.choice(len(ev),len(ev),replace=True,p=w)
        if ev[idx].sum()==0: continue
        cs.append(concordance_index_censored(ev[idx],ti[idx],risk[idx])[0])
    return np.mean(cs),np.percentile(cs,[2.5,97.5])
for name,risk in [("Stage+Sex",g_stage),("Full GAM",g5),("XGB",b5['XGB'])]:
    m1,ci1=reweighted_C(risk)
    print(f"  {name:<10} external C={C(risk):.4f}  -> reweighted to TCGA stage mix={m1:.4f} [{ci1[0]:.4f},{ci1[1]:.4f}]")

print("\n#10 DeepSurv external C across 5 seeds (5-feat):")
Xt5,Xe5=prep(["AGE","TMB","FGA"])
es_tr,es_vl=train_test_split(np.arange(len(Xt5)),test_size=.2,random_state=SEED,stratify=ytr["event"])
cs=[]
for sd_ in [1,2,3,4,5]:
    net=train_ds(Xt5[es_tr],ytr[es_tr],Xt5[es_vl],ytr[es_vl],Xt5.shape[1],seed=sd_); cs.append(C(dsp(net,Xe5)))
print(f"  seeds={[round(c,4) for c in cs]}  mean={np.mean(cs):.4f}  sd={np.std(cs):.4f}")
