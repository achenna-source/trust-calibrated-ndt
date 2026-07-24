"""coverage_check.py — empirical vs nominal conformal coverage of the twin-error bound.
Answers the reviewer's requirement: does the split-conformal bound B_alpha achieve its
nominal (1-alpha) coverage on held-out real data? Reports per-regime coverage + a plot.
"""
import numpy as np, pandas as pd
import ndt_trust_experiment as e

def coverage(seed=0):
    rng = np.random.default_rng(seed)
    df = e.load_kpm(rng)
    thr = float(np.quantile(df["load"], e.SURGE_QUANTILE))
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n=len(df); a=int(0.5*n); b=int(0.75*n)
    cal, fit, test = df.iloc[:a], df.iloc[a:b], df.iloc[b:]     # twin-train / conformal-fit / coverage-test
    twin = e.fit_twin(cal)

    def resid(split):
        X=split[["load","prb","sinr_db"]].values
        real=e.reality_latency(split["load"].values, split["prb"].values, split["sinr_db"].values, thr)
        return np.abs(twin.predict(X)-real), split["load"].values

    r_fit, load_fit = resid(fit)
    r_te,  load_te  = resid(test)
    levels = np.array([0.50,0.60,0.70,0.80,0.85,0.90,0.95])
    rows=[]
    for L in levels:
        for name, mfit, mte in [("normal", load_fit<=thr, load_te<=thr),
                                ("surge",  load_fit>thr,  load_te>thr)]:
            if mfit.sum()<10 or mte.sum()<10: continue
            B = np.quantile(r_fit[mfit], L)               # conformal bound at level L on FIT split
            emp = float(np.mean(r_te[mte] <= B))          # empirical coverage on held-out TEST split
            rows.append(dict(level=float(L), regime=name, empirical=emp, n=int(mte.sum())))
    return pd.DataFrame(rows)

if __name__=="__main__":
    # average empirical coverage over seeds
    dfs=[coverage(s) for s in range(10)]
    allc=pd.concat(dfs)
    g=allc.groupby(["regime","level"])["empirical"].agg(["mean","std"]).reset_index()
    print("=== Empirical vs nominal coverage (10 seeds) ===")
    for _,r in g.iterrows():
        print(f"  {r['regime']:7s} nominal={r['level']:.2f}  empirical={r['mean']:.3f} +/- {1.96*r['std']/np.sqrt(10):.3f}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(4.6,4.2))
        plt.plot([0.5,1],[0.5,1],"k--",lw=1,label="ideal (empirical = nominal)")
        for name,mk in [("normal","o"),("surge","s")]:
            sub=g[g.regime==name].sort_values("level")
            plt.plot(sub["level"],sub["mean"],mk+"-",label=name)
        plt.xlabel("nominal coverage 1-alpha"); plt.ylabel("empirical coverage")
        plt.legend(); plt.grid(alpha=.3); plt.title("Conformal bound coverage")
        plt.tight_layout(); plt.savefig("coverage.png",dpi=140); print("saved coverage.png")
    except Exception as ex: print("plot skip:",ex)
