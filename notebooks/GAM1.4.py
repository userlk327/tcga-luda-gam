"""
GAM1.4 — Feature-Level Survival GAM (spline-Cox)  +  External Validation
=========================================================================
A methodological pivot from GAM1.3. Instead of a GAM *meta-learner* stacked on
base-model risk scores (which was flat internally and degraded externally),
GAM1.4 fits a GAM *directly on the raw clinical/genomic features*:

    log-hazard = sum_k f_k(x_k) + linear(stage, sex)

where f_k are cubic B-spline smooths on the continuous features
(Age, TMB, Mutation Count, Fraction Genome Altered). TMB and Mutation Count
are log1p-transformed first (heavy right skew). This is a single, transparent
elastic-net Cox model — no stacking.

Rationale:
  - Nonlinearity is far more likely at the feature level (U-shaped age risk,
    saturating TMB effect) than on already-monotone base-model scores, so the
    'GAM' is justified.
  - It operates on features that harmonize across cohorts, so it should
    transfer better than a meta-stack of shifting base-model outputs.

Evaluated the SAME way as external_validation.py, on the six features common to
both cohorts, so numbers are directly comparable:
  - Internal:  TCGA train/test split (test C-index + bootstrap CI + KM)
  - External:  refit on ALL TCGA -> OncoSG (C-index + CI + KM)
  - Interpretability: per-feature smooth-shape plots vs a linear reference

Outputs -> results/
"""

from pathlib import Path
import random, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from patsy import dmatrix, build_design_matrices
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); random.seed(SEED)

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TCGA_CSV   = DATA_DIR / "luad_tcga_pan_can_atlas_2018_clinical_data.csv"
ONCOSG_CSV = DATA_DIR / "luad_oncosg_2020_clinical_data.csv"

# Mutation count is dropped: it is near-collinear with TMB (Spearman 0.98 in
# TCGA), so keeping both makes the individual smooths uninterpretable. TMB (the
# standardized, per-megabase metric) is retained.
CONT_FEATS = ["AGE", "TMB", "FGA"]                  # spline-smoothed
LOG_FEATS  = {"TMB"}                                 # log1p before splining
CAT_FEATS  = ["STAGE", "SEX"]                        # linear (one-hot)
DF_SPLINE  = 4
L1_RATIO   = 0.5                                     # elastic net (let smooths survive)
ALPHA_GRID = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

print("=" * 78)
print("GAM1.4 — FEATURE-LEVEL SPLINE-COX  +  EXTERNAL VALIDATION")
print("=" * 78)

# ============================================================================
# DATA HARMONIZATION  (identical mapping to external_validation.py)
# ============================================================================

def coarsen_stage(s):
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
    out = pd.DataFrame(index=df.index)
    out["AGE"]       = pd.to_numeric(df[cols["AGE"]], errors="coerce")
    out["SEX"]       = df[cols["SEX"]].astype(str).str.strip().str.capitalize()
    out["STAGE"]     = df[cols["STAGE"]].map(coarsen_stage)
    out["TMB"]       = pd.to_numeric(df[cols["TMB"]], errors="coerce")
    out["MUT_COUNT"] = pd.to_numeric(df[cols["MUT_COUNT"]], errors="coerce")
    out["FGA"]       = pd.to_numeric(df[cols["FGA"]], errors="coerce")
    out["OS_time"]   = pd.to_numeric(df[cols["OS_time"]], errors="coerce")
    out["OS_event"]  = df[cols["OS_status"]].astype(str).str.strip().str.startswith("1").astype(int)
    out = out.dropna(subset=["OS_time", "OS_event"])
    out = out[out["OS_time"] > 0]
    for c in CAT_FEATS:
        out[c] = out[c].where(out[c].notna(), "Unknown").replace({"": "Unknown"})
    return out.reset_index(drop=True)

print("\n[1] Loading and harmonizing cohorts...")
tcga = harmonize(pd.read_csv(TCGA_CSV), {
    "AGE": "Diagnosis Age", "SEX": "Sex",
    "STAGE": "Neoplasm Disease Stage American Joint Committee on Cancer Code",
    "TMB": "TMB (nonsynonymous)", "MUT_COUNT": "Mutation Count",
    "FGA": "Fraction Genome Altered",
    "OS_time": "Overall Survival (Months)", "OS_status": "Overall Survival Status",
})
onco = harmonize(pd.read_csv(ONCOSG_CSV), {
    "AGE": "AGE", "SEX": "SEX", "STAGE": "STAGE",
    "TMB": "TMB_NONSYNONYMOUS", "MUT_COUNT": "MUTATION_COUNT",
    "FGA": "FRACTION_GENOME_ALTERED",
    "OS_time": "OS_MONTHS", "OS_status": "OS_STATUS",
})
print(f"  TCGA   (train): n={len(tcga)}  events={tcga['OS_event'].sum()}")
print(f"  OncoSG (test):  n={len(onco)}  events={onco['OS_event'].sum()}")

# ============================================================================
# DESIGN MATRIX:  spline smooths on continuous + one-hot on categorical
# ============================================================================

def surv(frame):
    return np.array(list(zip(frame["OS_event"].astype(bool), frame["OS_time"])),
                    dtype=[("event", bool), ("time", float)])

def fit_cont_params(frame):
    """Per-feature transform params (log flag, median, mean, std, z-range) from train."""
    p = {}
    for f in CONT_FEATS:
        v = frame[f].astype(float)
        if f in LOG_FEATS:
            v = np.log1p(v)
        med = v.median()
        v = v.fillna(med)
        mean, std = float(v.mean()), float(v.std()) + 1e-8
        z = (v - mean) / std
        p[f] = {"log": f in LOG_FEATS, "median": med, "mean": mean, "std": std,
                "zmin": float(z.min()), "zmax": float(z.max())}   # spline knot range
    return p

def apply_cont(frame, params):
    Z = pd.DataFrame(index=frame.index)
    for f in CONT_FEATS:
        v = frame[f].astype(float)
        if params[f]["log"]:
            v = np.log1p(v)
        v = v.fillna(params[f]["median"])
        z = (v - params[f]["mean"]) / params[f]["std"]
        # clip external points into the train knot range so splines don't extrapolate
        Z[f] = z.clip(params[f]["zmin"], params[f]["zmax"])
    return Z

class GamDesign:
    """Builds the spline+categorical design matrix; fit on train, reused elsewhere."""
    def __init__(self):
        self.cont_params = None
        self.spline_info = {}
        self.ohe = None
        self.columns = None

    def fit(self, frame):
        self.cont_params = fit_cont_params(frame)
        Z = apply_cont(frame, self.cont_params)
        parts = []
        for f in CONT_FEATS:
            # '- 1' suppresses patsy's per-term intercept (avoids a duplicate
            # all-ones column per feature, which makes the design rank-deficient)
            sp = dmatrix(f"bs(x, df={DF_SPLINE}, degree=3, include_intercept=False) - 1",
                         {"x": Z[f].values}, return_type="dataframe")
            self.spline_info[f] = sp.design_info
            sp.columns = [f"{f}_s{i}" for i in range(sp.shape[1])]
            sp.index = frame.index
            parts.append(sp)
        self.ohe = OneHotEncoder(drop="first", sparse_output=False,
                                 handle_unknown="ignore").fit(frame[CAT_FEATS])
        cat = pd.DataFrame(self.ohe.transform(frame[CAT_FEATS]),
                           columns=list(self.ohe.get_feature_names_out(CAT_FEATS)),
                           index=frame.index)
        parts.append(cat)
        design = pd.concat(parts, axis=1)
        self.columns = design.columns.tolist()
        return design

    def transform(self, frame):
        Z = apply_cont(frame, self.cont_params)
        parts = []
        for f in CONT_FEATS:
            arr = build_design_matrices([self.spline_info[f]], {"x": Z[f].values})[0]
            sp = pd.DataFrame(arr, columns=[f"{f}_s{i}" for i in range(arr.shape[1])],
                              index=frame.index)
            parts.append(sp)
        cat = pd.DataFrame(self.ohe.transform(frame[CAT_FEATS]),
                           columns=list(self.ohe.get_feature_names_out(CAT_FEATS)),
                           index=frame.index)
        parts.append(cat)
        return pd.concat(parts, axis=1)[self.columns]

def tune_alpha(Xtr, ytr):
    """5-fold CV over ALPHA_GRID; return alpha with best mean C-index."""
    best_a, best_c = ALPHA_GRID[0], -1
    for a in ALPHA_GRID:
        cs = []
        for tr_i, vl_i in KFold(5, shuffle=True, random_state=SEED).split(Xtr):
            try:
                m = CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=L1_RATIO, max_iter=200000)
                m.fit(Xtr[tr_i], ytr[tr_i])
                cs.append(concordance_index_censored(
                    ytr["event"][vl_i], ytr["time"][vl_i], m.predict(Xtr[vl_i]))[0])
            except Exception:
                pass
        if cs and np.mean(cs) > best_c:
            best_c, best_a = np.mean(cs), a
    return best_a, best_c

def boot_ci(ev, ti, risk, n=1000):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(ev), len(ev), replace=True)
        if ev[idx].sum() == 0:
            continue
        vals.append(concordance_index_censored(ev[idx], ti[idx], risk[idx])[0])
    return np.percentile(vals, [2.5, 97.5])

def km_stratify(risk, ev, ti, label, fname):
    grp = pd.qcut(risk, 3, labels=["Low", "Medium", "High"])
    lr = logrank_test(ti[grp == "Low"], ti[grp == "High"], ev[grp == "Low"], ev[grp == "High"])
    mv = multivariate_logrank_test(ti, grp, ev)
    fig, ax = plt.subplots(figsize=(9, 6))
    kmf = KaplanMeierFitter()
    for g, color in zip(["Low", "Medium", "High"], ["#2E7D32", "#F57C00", "#C62828"]):
        mask = grp == g
        kmf.fit(ti[mask], ev[mask], label=f"{g} risk (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2.5)
    ax.set_xlabel("Time (Months)"); ax.set_ylabel("Overall Survival Probability")
    ax.set_title(f"GAM1.4 — {label}\nLow vs High log-rank p = {lr.p_value:.4f}")
    ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches="tight")
    plt.close()
    return lr.p_value, mv.p_value

# ============================================================================
# EXTERNAL VALIDATION  (fit on ALL TCGA -> OncoSG)  [built first: also used for smooths]
# ============================================================================

print("\n[2] External validation on OncoSG (fit on all TCGA)...")
design_full = GamDesign()
X_full = design_full.fit(tcga).values
y_full = surv(tcga)
X_ext = design_full.transform(onco).values
y_ext_ev = onco["OS_event"].values.astype(bool)
y_ext_ti = onco["OS_time"].values

alpha_ext, cv_c_ext = tune_alpha(X_full, y_full)
model_ext = CoxnetSurvivalAnalysis(alphas=[alpha_ext], l1_ratio=L1_RATIO, max_iter=200000)
model_ext.fit(X_full, y_full)
risk_ext = model_ext.predict(X_ext)
c_ext = concordance_index_censored(y_ext_ev, y_ext_ti, risk_ext)[0]
lo_ext, hi_ext = boot_ci(y_ext_ev, y_ext_ti, risk_ext)
print(f"  best alpha={alpha_ext} (TCGA CV C-index {cv_c_ext:.3f})")
print(f"  OncoSG external C-index: {c_ext:.4f}  [{lo_ext:.4f}, {hi_ext:.4f}]")
p_lh_ext, p_mv_ext = km_stratify(risk_ext, y_ext_ev, y_ext_ti,
                                 "External (OncoSG) risk tertiles", "gam14_km_oncosg.png")
print(f"  OncoSG KM: Low vs High p={p_lh_ext:.4e}, multivariate p={p_mv_ext:.4e}")

# ============================================================================
# SMOOTH-SHAPE PLOTS  (interpretability — does nonlinearity exist?)
# ============================================================================

print("\n[3] Smooth-shape plots (per-feature f_k, model fit on all TCGA)...")
coefs = model_ext.coef_.ravel()
design = design_full
col_index = {c: i for i, c in enumerate(design.columns)}
orig_range = {f: (tcga[f].astype(float).quantile(0.02), tcga[f].astype(float).quantile(0.98))
              for f in CONT_FEATS}

fig, axes = plt.subplots(1, len(CONT_FEATS), figsize=(5 * len(CONT_FEATS), 4.5))
axes = np.atleast_1d(axes).ravel()
titles = {"AGE": "Age (years)", "TMB": "TMB (nonsynonymous)",
          "MUT_COUNT": "Mutation Count", "FGA": "Fraction Genome Altered"}
for ax, f in zip(axes, CONT_FEATS):
    lo, hi = orig_range[f]
    grid = np.linspace(lo, hi, 300)
    gv = np.log1p(grid) if design.cont_params[f]["log"] else grid.astype(float)
    z = (gv - design.cont_params[f]["mean"]) / design.cont_params[f]["std"]
    z = np.clip(z, design.cont_params[f]["zmin"], design.cont_params[f]["zmax"])
    basis = build_design_matrices([design.spline_info[f]], {"x": z})[0]
    idx = [col_index[f"{f}_s{i}"] for i in range(basis.shape[1])]
    fvals = basis @ coefs[idx]
    fvals = fvals - fvals.mean()                      # center for readability
    ax.plot(grid, fvals, color="#1565C0", linewidth=2.5, label="GAM smooth $f_k$")
    # linear reference (OLS slope through the smooth)
    b = np.polyfit(grid, fvals, 1)
    ax.plot(grid, np.polyval(b, grid), color="gray", ls="--", lw=1.5,
            alpha=0.7, label="linear reference")
    ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    rug = tcga[f].astype(float).clip(lo, hi).values[::4]
    ax.scatter(rug, np.full(len(rug), fvals.min() - 0.05 * (fvals.max() - fvals.min())),
               marker="|", color="#1565C0", alpha=0.25, s=20)
    ax.set_xlabel(titles[f]); ax.set_ylabel("Contribution to log-hazard")
    ax.set_title(f"Smooth: {titles[f]}", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.suptitle("GAM1.4 — Per-Feature Smooth Contributions (TCGA train)\n"
             "Curvature away from the dashed line = nonlinear prognostic effect",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gam14_smooths.png", dpi=300, bbox_inches="tight")
plt.close()
print("  saved: results/gam14_smooths.png")

# ============================================================================
# INTERNAL EVALUATION ON TCGA  (nested 5-fold CV — stable, leakage-free)
# ============================================================================

print("\n[4] Internal evaluation on TCGA (nested 5-fold CV)...")
y_all = surv(tcga)
oof_risk = np.zeros(len(tcga))
for k, (tr_i, te_i) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(tcga)):
    d = GamDesign()
    Xtr = d.fit(tcga.iloc[tr_i].reset_index(drop=True)).values
    Xte = d.transform(tcga.iloc[te_i].reset_index(drop=True)).values
    ytr = y_all[tr_i]
    a, _ = tune_alpha(Xtr, ytr)                              # inner CV tunes alpha
    m = CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=L1_RATIO, max_iter=200000).fit(Xtr, ytr)
    oof_risk[te_i] = m.predict(Xte)
    print(f"  fold {k+1}/5 done (alpha={a})")

c_int = concordance_index_censored(y_all["event"], y_all["time"], oof_risk)[0]
lo_int, hi_int = boot_ci(y_all["event"], y_all["time"], oof_risk)
print(f"  TCGA pooled OOF C-index: {c_int:.4f}  [{lo_int:.4f}, {hi_int:.4f}]")
p_lh_int, p_mv_int = km_stratify(oof_risk, y_all["event"], y_all["time"],
                                 "Internal (TCGA pooled OOF) risk tertiles", "gam14_km_tcga.png")
print(f"  TCGA KM: Low vs High p={p_lh_int:.4e}, multivariate p={p_mv_int:.4e}")

# ============================================================================
# LIKELIHOOD-RATIO NONLINEARITY TEST  (unpenalized Cox: spline vs linear)
# ============================================================================

print("\n[5] Likelihood-ratio nonlinearity test (Cox partial likelihood, all TCGA)...")
from sksurv.linear_model import CoxPHSurvivalAnalysis
from scipy import stats

Xfull_df = design_full.transform(tcga)                    # spline + categorical
Zlin = apply_cont(tcga, design_full.cont_params)          # 1 linear col per continuous
spline_cols = {f: [c for c in design_full.columns if c.startswith(f + "_s")] for f in CONT_FEATS}
cat_cols = [c for c in design_full.columns
            if not any(c.startswith(f + "_s") for f in CONT_FEATS)]

def make_design(spline_set):
    # For spline features drop one basis column (identifiability: B-spline bases
    # sum to 1, so the full block is rank-deficient for an unpenalized Cox fit).
    parts = []
    for f in CONT_FEATS:
        parts.append(Xfull_df[spline_cols[f][1:]] if f in spline_set else Zlin[[f]])
    parts.append(Xfull_df[cat_cols])
    return pd.concat(parts, axis=1)

def breslow_loglik(eta, time, event):
    """Cox partial log-likelihood (Breslow ties) at a given linear predictor eta."""
    order = np.argsort(time)
    eta_s, time_s, ev_s = eta[order], time[order], event[order].astype(bool)
    m = eta_s.max()
    csum = np.cumsum(np.exp(eta_s - m)[::-1])[::-1]        # csum[i] = sum_{k>=i} exp(eta-m)
    ll = 0.0
    for t in np.unique(time_s[ev_s]):
        D = (time_s == t) & ev_s
        first = np.searchsorted(time_s, t, side="left")
        ll += eta_s[D].sum() - D.sum() * (m + np.log(csum[first]))
    return ll

def cox_ll(design_df):
    """Fit Cox (unpenalized; tiny ridge fallback) and return its partial log-lik + #params."""
    X = design_df.values.astype(float)
    for a in (0.0, 1e-4, 1e-2):
        try:
            model = CoxPHSurvivalAnalysis(alpha=a).fit(X, y_full)
            eta = X @ model.coef_
            return breslow_loglik(eta, tcga["OS_time"].values,
                                  tcga["OS_event"].values), X.shape[1], a
        except Exception:
            continue
    return np.nan, design_df.shape[1], None

ll_lin,  k_lin,  a_lin  = cox_ll(make_design(set()))                 # all linear
ll_full, k_full, a_full = cox_ll(make_design(set(CONT_FEATS)))       # all spline
lr_overall = 2 * (ll_full - ll_lin)
df_overall = k_full - k_lin
p_overall = stats.chi2.sf(lr_overall, df_overall)
print(f"  Overall spline vs linear: LR={lr_overall:.2f}, df={df_overall}, p={p_overall:.3e}")

# Per-feature: spline for THAT feature only vs fully linear
perfeat = {}
for f in CONT_FEATS:
    ll_f, k_f, _ = cox_ll(make_design({f}))
    lr = 2 * (ll_f - ll_lin)
    df_f = k_f - k_lin
    perfeat[f] = (lr, df_f, stats.chi2.sf(lr, df_f))
    print(f"  {f:<10} nonlinearity: LR={lr:6.2f}, df={df_f}, p={perfeat[f][2]:.3e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("GAM1.4 SUMMARY")
print("=" * 78)
print(f"{'Setting':<24}{'C-index':>10}   95% CI")
print("-" * 54)
print(f"{'TCGA internal (CV)':<24}{c_int:>10.4f}   [{lo_int:.4f}, {hi_int:.4f}]")
print(f"{'OncoSG external':<24}{c_ext:>10.4f}   [{lo_ext:.4f}, {hi_ext:.4f}]")
print(f"\nKM stratification (Low vs High):")
print(f"  TCGA internal:  p = {p_lh_int:.4e}")
print(f"  OncoSG external: p = {p_lh_ext:.4e}")

with open(OUTPUT_DIR / "gam14_results.txt", "w", encoding="utf-8") as f:
    f.write("GAM1.4 — Feature-Level Spline-Cox\n" + "=" * 50 + "\n\n")
    f.write(f"Continuous (spline df={DF_SPLINE}): {CONT_FEATS}  (log1p: {sorted(LOG_FEATS)})\n")
    f.write(f"Categorical (linear): {CAT_FEATS}\n")
    f.write(f"Elastic net l1_ratio={L1_RATIO}\n\n")
    f.write(f"TCGA internal (nested 5-fold CV) C-index: {c_int:.4f} "
            f"[{lo_int:.4f}, {hi_int:.4f}]\n")
    f.write(f"OncoSG external C-index:                  {c_ext:.4f} "
            f"[{lo_ext:.4f}, {hi_ext:.4f}] (alpha={alpha_ext})\n\n")
    f.write(f"KM Low vs High:  TCGA p={p_lh_int:.4e}  |  OncoSG p={p_lh_ext:.4e}\n\n")
    f.write("Likelihood-ratio nonlinearity test (unpenalized Cox, all TCGA):\n")
    f.write(f"  Overall spline vs linear: LR={lr_overall:.2f}, df={df_overall}, p={p_overall:.3e}\n")
    for f_ in CONT_FEATS:
        lr, dff, pv = perfeat[f_]
        f.write(f"  {f_:<10} LR={lr:6.2f}, df={dff}, p={pv:.3e}\n")

print("\n✓ Saved: results/gam14_results.txt")
print("✓ Saved: results/gam14_smooths.png, gam14_km_tcga.png, gam14_km_oncosg.png")
print("=" * 78)
