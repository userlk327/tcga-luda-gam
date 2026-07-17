"""
Leakage-free internal benchmark on the 6 common features (addresses reviewer
critiques #1 and #4): all preprocessing fit INSIDE each outer fold; the stacking
ensemble is given an internal cross-validated C-index on the SAME 6 features so
it can be compared apples-to-apples with its external number (0.595).
"""
import warnings, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim, xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from patsy import dmatrix, build_design_matrices
warnings.filterwarnings("ignore")
SEED=42; np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
dev=torch.device("cpu")
from pathlib import Path
DATA=lambda f: str(Path(__file__).resolve().parent.parent / "data" / f)

CAT=["SEX","STAGE"]; NUM=["AGE","TMB","FGA"]; COMMON=CAT+NUM  # mutation count dropped (0.98 corr with TMB)
def coarsen(s):
    s=str(s).upper().replace("STAGE","").strip()
    if s in("","NAN","NA"):return np.nan
    if s.startswith("IV")or s=="4":return "IV"
    if s.startswith("III"):return "III"
    if s.startswith("II"):return "II"
    if s.startswith("I")or s=="1":return "I"
    return np.nan
def harmonize(df,c):
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
    for cc in CAT: o[cc]=o[cc].where(o[cc].notna(),"Unknown").replace({"":"Unknown"})
    return o.reset_index(drop=True)
tcga=harmonize(pd.read_csv(DATA("luad_tcga_pan_can_atlas_2018_clinical_data.csv")),
    {"AGE":"Diagnosis Age","SEX":"Sex","STAGE":"Neoplasm Disease Stage American Joint Committee on Cancer Code",
     "TMB":"TMB (nonsynonymous)","MUT_COUNT":"Mutation Count","FGA":"Fraction Genome Altered",
     "OS_time":"Overall Survival (Months)","OS_status":"Overall Survival Status"})
y=Surv.from_arrays(event=tcga["OS_event"].astype(bool),time=tcga["OS_time"].values)

class DS(nn.Module):
    def __init__(s,n):
        super().__init__(); s.net=nn.Sequential(nn.Linear(n,128),nn.ReLU(),nn.Dropout(.3),
            nn.Linear(128,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,1))
    def forward(s,x):return s.net(x).squeeze(-1)
def coxloss(r,t,e):
    o=torch.argsort(-t);r,e=r[o],e[o];return -(e*(r-torch.logcumsumexp(r,0))).sum()/(e.sum()+1e-8)
def train_ds(Xt,yt,Xe,ye,n):
    net=DS(n).to(dev);opt=optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    tt=lambda a:torch.tensor(a,dtype=torch.float32).to(dev)
    Xtt,Xet=tt(Xt),tt(Xe)
    yt_t,yt_e=tt([e["time"]for e in yt]),tt([e["event"]for e in yt])
    ye_t,ye_e=tt([e["time"]for e in ye]),tt([e["event"]for e in ye])
    best,wait,st=np.inf,0,None
    for _ in range(400):
        net.train();opt.zero_grad();coxloss(net(Xtt),yt_t,yt_e).backward();opt.step()
        net.eval()
        with torch.no_grad():vl=coxloss(net(Xet),ye_t,ye_e).item()
        if vl<best-1e-6:best,wait,st=vl,0,{k:v.cpu().clone()for k,v in net.state_dict().items()}
        else:
            wait+=1
            if wait>=25:break
    if st:net.load_state_dict(st)
    net.eval();return net
def dsp(net,X):
    with torch.no_grad():return net(torch.tensor(X,dtype=torch.float32).to(dev)).cpu().numpy()
XGBP=dict(objective="survival:cox",eval_metric="cox-nloglik",eta=.05,max_depth=3,
    subsample=.8,colsample_bytree=.8,seed=SEED,verbosity=0)
def xlab(y):return [e["time"] if e["event"] else -e["time"] for e in y]
def fitxgb(Xt,yt,Xe,ye):
    m=xgb.train(XGBP,xgb.DMatrix(Xt,label=xlab(yt)),num_boost_round=2000,
        evals=[(xgb.DMatrix(Xe,label=xlab(ye)),"v")],early_stopping_rounds=100,verbose_eval=False)
    it=(m.best_iteration+1) if hasattr(m,"best_iteration") else m.num_boosted_rounds()
    return m,it

def preprocess_fit(train_df):
    pre=ColumnTransformer([("cat",OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore"),CAT),
        ("num",SimpleImputer(strategy="median"),NUM)],remainder="drop")
    Xp=pre.fit_transform(train_df[COMMON]); sc=StandardScaler().fit(Xp)
    return pre,sc
def preprocess_apply(pre,sc,df): return sc.transform(pre.transform(df[COMMON]))

META=["RSF","GBS","XGB","DS"]
def base_fit_predict(Xtr,ytr,Xte):
    es_tr,es_vl=train_test_split(np.arange(len(Xtr)),test_size=.2,random_state=SEED,stratify=ytr["event"])
    Xes,yes=Xtr[es_vl],ytr[es_vl]
    rsf=RandomSurvivalForest(n_estimators=500,max_features="sqrt",min_samples_leaf=3,random_state=SEED,n_jobs=-1).fit(Xtr,ytr)
    gbs=GradientBoostingSurvivalAnalysis(n_estimators=500,learning_rate=.05,max_depth=3,random_state=SEED).fit(Xtr,ytr)
    xm,it=fitxgb(Xtr[es_tr],ytr[es_tr],Xes,yes)
    net=train_ds(Xtr[es_tr],ytr[es_tr],Xes,yes,Xtr.shape[1])
    return np.column_stack([rsf.predict(Xte),gbs.predict(Xte),
        xm.predict(xgb.DMatrix(Xte),iteration_range=(0,it)),dsp(net,Xte)])

n=len(tcga); pooled=np.zeros((n,5))  # RSF,GBS,XGB,DS,Ensemble
outer=KFold(5,shuffle=True,random_state=SEED)
for k,(tr,te) in enumerate(outer.split(tcga)):
    print(f"outer fold {k+1}/5",flush=True)
    tr_df,te_df=tcga.iloc[tr].reset_index(drop=True),tcga.iloc[te].reset_index(drop=True)
    ytr,yte=y[tr],y[te]
    pre,sc=preprocess_fit(tr_df)                         # fit preprocessing on TRAIN fold only
    Xtr,Xte=preprocess_apply(pre,sc,tr_df),preprocess_apply(pre,sc,te_df)
    # base learners: final on outer-train -> predict outer-test
    te_base=base_fit_predict(Xtr,ytr,Xte)
    pooled[te,0:4]=te_base
    # ensemble: inner 3-fold OOF meta on outer-train, GAM meta, predict outer-test
    inner=KFold(3,shuffle=True,random_state=SEED); oof=np.zeros((len(tr),4))
    for it_,iv_ in inner.split(Xtr):
        oof[iv_]=base_fit_predict(Xtr[it_],ytr[it_],Xtr[iv_])
    mtr=pd.DataFrame(oof,columns=META); mte=pd.DataFrame(te_base,columns=META)
    for f in META:
        lo,hi=mtr[f].min(),mtr[f].max(); mte[f]=mte[f].clip(lo,hi)
    dinfo={}; parts=[]
    for f in META:
        sp=dmatrix(f"bs({f}, df=4, degree=3, include_intercept=False)",mtr,return_type="dataframe")
        dinfo[f]=sp.design_info; sp.columns=[f"{f}_s{i}"for i in range(sp.shape[1])]; parts.append(sp)
    sptr=pd.concat(parts,axis=1)
    spte=pd.concat([pd.DataFrame(build_design_matrices([dinfo[f]],mte)[0]) for f in META],axis=1)
    yts=np.array(list(zip(ytr["event"],ytr["time"])),dtype=[("event",bool),("time",float)])
    ba,bc=0.01,-1
    for a in [0.001,0.005,0.01,0.05,0.1]:
        cs=[]
        for t2,v2 in KFold(5,shuffle=True,random_state=SEED).split(sptr):
            g=CoxnetSurvivalAnalysis(alphas=[a],l1_ratio=.9,max_iter=100000).fit(sptr.values[t2],yts[t2])
            cs.append(concordance_index_censored(yts["event"][v2],yts["time"][v2],g.predict(sptr.values[v2]))[0])
        if np.mean(cs)>bc:bc,ba=np.mean(cs),a
    gam=CoxnetSurvivalAnalysis(alphas=[ba],l1_ratio=.9,max_iter=100000).fit(sptr.values,yts)
    pooled[te,4]=gam.predict(spte.values)

print("\n=== LEAKAGE-FREE INTERNAL C-INDEX (6 features, pooled 5-fold) ===")
for i,m in enumerate(["RSF","GBS","XGB","DeepSurv","GAM Ensemble"]):
    c=concordance_index_censored(y["event"],y["time"],pooled[:,i])[0]
    print(f"  {m:<14} {c:.4f}")
