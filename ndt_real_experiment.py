"""ndt_real_experiment.py -- REAL-DATA instantiation of the trust-gate case study
on the public Colosseum O-RAN COMMAG dataset (Bonati et al. 2021).

No counterfactuals are fabricated: we gate OBSERVED (state, config, outcome) records.
The twin predicts a QoS outcome (downlink buffer occupancy = congestion) from the
state; twin optimism arises GENUINELY from distribution shift -- the twin is trained
on one scheduling policy and evaluated on another (nothing is injected). Reality is the
measured buffer. TFS blocks when the conformal bound on the twin's error, fit on
HELD-OUT REAL residuals in the record's channel-quality region, exceeds a harm budget.

Outputs: real_results_table.md and a coverage number. Honest: whatever the data shows.
"""
import glob, os, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

SEEDS       = range(10)
SLA_FLOOR   = 20000.0          # bytes: buffer above this = SLA violation (congestion)
ALPHA       = 0.10
N_CQI_BINS  = 4
TRAIN_POLICY, TEST_POLICY = 0, 2   # twin trained on one policy, tested on the shifted one
FEATURES    = ["dl_cqi", "sum_granted_prbs", "sum_requested_prbs", "slice_prb"]
OUTCOME     = "dl_buffer [bytes]"

def load():
    fs = glob.glob(os.path.join(os.path.dirname(__file__), "colosseum_data", "*_metrics.csv"))
    dfs = []
    for f in fs:
        d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]; dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    keep = FEATURES + [OUTCOME, "scheduling_policy"]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[keep].dropna()
    df["sum_requested_prbs"] = df["sum_requested_prbs"].clip(lower=0)   # data has stray negatives
    df["y"] = np.log1p(df[OUTCOME].clip(lower=0))                        # log-buffer, stable target
    return df

FLOOR_LOG = np.log1p(SLA_FLOOR)

def run(seed, region_mode="cqi"):
    rng = np.random.default_rng(seed)
    df = load()
    tr  = df[df.scheduling_policy == TRAIN_POLICY]                       # twin training regime
    sh  = df[df.scheduling_policy == TEST_POLICY].sample(frac=1.0, random_state=seed)  # shifted regime
    m = len(sh) // 2
    fit, test = sh.iloc[:m], sh.iloc[m:]                                 # conformal-fit / evaluation (both REAL, shifted)

    twin = GradientBoostingRegressor(n_estimators=250, max_depth=3, random_state=seed)
    twin.fit(tr[FEATURES].values, tr["y"].values)

    pf = twin.predict(fit[FEATURES].values); rf = np.abs(pf - fit["y"].values)
    Xt = test[FEATURES].values
    y_pred = twin.predict(Xt); y_real = test["y"].values

    # region key: 'cqi' = channel-quality bins; 'predmag' = twin predicted-magnitude bins;
    # 'learned' = bins of a fitted error-predictor (state -> expected |twin error|), the open-problem region.
    if region_mode == "cqi":
        kf, kt = fit["dl_cqi"].values, test["dl_cqi"].values
    elif region_mode == "predmag":
        kf, kt = pf, y_pred
    else:  # learned error-predictor region (trained on FIT residuals, applied out-of-sample to TEST)
        ep = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed)
        ep.fit(fit[FEATURES].values, np.abs(pf - fit["y"].values))
        kf, kt = ep.predict(fit[FEATURES].values), ep.predict(Xt)
    edges = np.quantile(kf, np.linspace(0, 1, N_CQI_BINS + 1)); edges[0], edges[-1] = -np.inf, np.inf
    def region(x): return np.clip(np.digitize(x, edges[1:-1]), 0, N_CQI_BINS - 1)
    reg_f = region(kf)

    # conformal bound per region from HELD-OUT REAL residuals (twin vs real), NOT twin vs twin
    B = {}
    for b in range(N_CQI_BINS):
        e = rf[reg_f == b]
        B[b] = float(np.quantile(e, 1 - ALPHA)) if len(e) >= 10 else float(np.quantile(rf, 1 - ALPHA))

    reg_t = region(kt)
    Bt = np.array([B[b] for b in reg_t])

    valid = y_pred <= FLOOR_LOG           # twin says SLA ok (C1)
    harmful = y_real > FLOOR_LOG          # real record actually violates SLA
    h = float(np.median(list(B.values()))) # harm budget = typical regional bound (operating point)
    tfs_ok = Bt <= h

    def metrics(commit):
        commit = np.asarray(commit, bool); blocked = ~commit
        safe = ~harmful
        catch = float(np.mean(blocked[harmful])) if harmful.any() else np.nan
        fa    = float(np.mean(blocked[safe]))    if safe.any() else np.nan
        slav  = float(np.mean(harmful[commit]))  if commit.any() else 0.0
        return catch, fa, slav, float(np.mean(commit))

    c1 = valid
    c2 = valid & tfs_ok
    rate = float(np.mean(c1 & ~c2)) / max(float(np.mean(c1)), 1e-9)     # extra blocking C2 adds over C1
    c3 = c1 & (rng.random(len(test)) >= rate)
    emp_cov = float(np.mean(np.abs(y_pred - y_real) <= Bt))            # empirical coverage of the bound
    # localizability diagnostic: can the regional bound Bt separate harmful from safe records?
    from sklearn.metrics import roc_auc_score
    loc_auc = float(roc_auc_score(harmful, Bt)) if (harmful.any() and (~harmful).any() and np.ptp(Bt) > 0) else 0.5
    return dict(C0=metrics(np.ones(len(test), bool)), C1=metrics(c1), C2=metrics(c2), C3=metrics(c3),
                cov=emp_cov, base=float(np.mean(harmful)), opt=float(np.mean(valid & harmful)), loc_auc=loc_auc)

if __name__ == "__main__":
    labels = {"C0":"C0 (commit all)","C1":"C1 (validity only)","C2":"C2 (validity + TFS)","C3":"C3 (random block)"}
    out_md = ["Colosseum O-RAN COMMAG data (real); twin trained on scheduling_policy %d, evaluated on %d; 10 seeds."%(TRAIN_POLICY,TEST_POLICY)]
    for mode, name in [("cqi","CQI-region (primary)"), ("predmag","predicted-magnitude region (pre-specified variant)"), ("learned","LEARNED error-predictor region")]:
        R = [run(s, region_mode=mode) for s in SEEDS]
        def agg(cond,k,R=R):
            v=np.array([r[cond][k] for r in R]); return v.mean(),1.96*v.std(ddof=1)/np.sqrt(len(v))
        auc=np.mean([r['loc_auc'] for r in R]); base=np.mean([r['base'] for r in R]); cov=np.mean([r['cov'] for r in R])
        hdr=f"\n### {name}\nbase harmful={base:.3f} | conformal coverage={cov:.3f} (nom {1-ALPHA:.2f}) | LOCALIZABILITY AUC={auc:.3f}"
        print(hdr); out_md.append(hdr)
        tbl=["\n| Condition | Catch | False-alarm | SLA-violation | Commit |","|---|---|---|---|---|"]
        for c in ["C0","C1","C2","C3"]:
            (ca,cae),(fa,fae),(sl,sle),(cm,_)=[agg(c,i) for i in range(4)]
            tbl.append(f"| {labels[c]} | {ca:.3f} ± {cae:.3f} | {fa:.3f} ± {fae:.3f} | {sl:.3f} ± {sle:.3f} | {cm:.3f} |")
        print("\n".join(tbl)); out_md.append("\n".join(tbl))
    open("real_results_table.md","w").write("\n".join(out_md)+"\n")
    print("\nsaved real_results_table.md")
