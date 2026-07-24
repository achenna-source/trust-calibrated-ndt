"""sensitivity_sweep.py — operating characteristic of the trust gate across scenario strength.
Varies the twin-optimism magnitude (DIVERGENCE_GAIN; 0 = a FAITHFUL twin) at fixed harm
budget, and reports C1/C2/C3 catch, C2 false-alarm, and live SLA-violation, averaged over
seeds. The GAIN=0 row is the NULL regime: a faithful twin, where TFS should correctly stay
silent (C2 == C1, near-zero false alarm) -- the control that answers "you engineered optimism"."""
import numpy as np, pandas as pd
import ndt_trust_experiment as e
e.MODEL_NAME = "__force_fallback__"      # rule controller, no torch

def metric(df, cond, col): return float(df.loc[cond, col])

def sweep(gains, seeds=range(8), n=400):
    rows=[]
    for g in gains:
        e.DIVERGENCE_GAIN = float(g)
        accs={k:[] for k in ["C0_sla","C1_catch","C2_catch","C2_fa","C2_sla","C3_catch","C3_sla"]}
        for s in seeds:
            o=e.run(seed=s, n_actions=n)
            accs["C0_sla"].append(metric(o,"C0 (commit all)","sla_violation"))
            accs["C1_catch"].append(metric(o,"C1 (validity only)","catch_rate"))
            accs["C2_catch"].append(metric(o,"C2 (validity + TFS)","catch_rate"))
            accs["C2_fa"].append(metric(o,"C2 (validity + TFS)","false_alarm"))
            accs["C2_sla"].append(metric(o,"C2 (validity + TFS)","sla_violation"))
            accs["C3_catch"].append(metric(o,"C3 (random block ctrl)","catch_rate"))
            accs["C3_sla"].append(metric(o,"C3 (random block ctrl)","sla_violation"))
        row={"gain":g}; row.update({k:float(np.nanmean(v)) for k,v in accs.items()})
        rows.append(row)
    return pd.DataFrame(rows)

if __name__=="__main__":
    df=sweep([0,15,30,45,60,90,120])
    pd.set_option("display.width",160)
    print("\n=== Sensitivity sweep over twin-optimism magnitude (h=%.0f, 8 seeds) ===" % e.HARM_BUDGET_H)
    print(df.round(3).to_string(index=False))
    print("\nGAIN=0 is the NULL regime (faithful twin): expect C2 catch ~ C1 catch and C2 false-alarm ~ 0.")
    df.to_csv("sensitivity.csv", index=False)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(8.4,3.6))
        ax[0].plot(df.gain,df.C0_sla,"k--o",label="C0/C1 (no trust gate)")
        ax[0].plot(df.gain,df.C2_sla,"g-s",label="C2 (validity+TFS)")
        ax[0].plot(df.gain,df.C3_sla,"r-^",label="C3 (random block)")
        ax[0].set_xlabel("twin-optimism magnitude"); ax[0].set_ylabel("live SLA-violation"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].plot(df.gain,df.C2_catch,"g-s",label="C2 catch")
        ax[1].plot(df.gain,df.C3_catch,"r-^",label="C3 catch")
        ax[1].plot(df.gain,df.C2_fa,"b-o",label="C2 false-alarm")
        ax[1].set_xlabel("twin-optimism magnitude"); ax[1].set_ylabel("rate"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig("sensitivity.png",dpi=140); print("saved sensitivity.png")
    except Exception as ex: print("plot skip:",ex)
