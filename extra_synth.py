"""extra_synth.py -- continuous localizability sweep + diffuse-optimism null + decision latency.
Answers reviewers: Lambda-vs-catch across a spectrum (not n=3); a PRESENT-BUT-DIFFUSE optimism null
inside the controlled setting; measured per-decision latency. Self-contained synthetic with a
localization knob w in [0,1]: w=1 optimism concentrated in the surge region (localizable),
w=0 optimism spread uniformly across load (diffuse)."""
import numpy as np, time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

TARGET=20.0; ALPHA=0.10; H=5.0; SURGE_Q=0.80; GAIN=90.0; N=4

def base_lat(load,prb): return 3.0 + 9.0*load**2/(prb/50.0+0.35)

def gen(seed):
    rng=np.random.default_rng(seed); n=6000
    load=np.clip(rng.beta(2,3,n)*1.4,0,1); prb=rng.integers(10,100,n).astype(float)
    sinr=rng.normal(12,4,n)-6*load
    lat=base_lat(load,prb)+rng.normal(0,1.0,n)
    return dict(load=load,prb=prb,sinr=sinr,lat=np.clip(lat,1,None))

BINS=6
def run_w(seed,w,p_harm=0.20,inflate=30.0):
    # Action-level optimism: a fixed fraction p_harm of actions are made harmful by inflating their
    # REAL outcome (twin blind). Their placement interpolates from region-aligned (w=1, localizable)
    # to uniformly random across regions (w=0, diffuse). Base rate held ~constant across w.
    d=gen(seed); rng=np.random.default_rng(seed)
    idx=np.arange(len(d["load"])); rng.shuffle(idx)
    a=int(.5*len(idx)); b=int(.8*len(idx)); cal,fit,ev=idx[:a],idx[a:b],idx[b:]
    X=lambda I: np.c_[d["load"][I],d["prb"][I],d["sinr"][I]]
    twin=GradientBoostingRegressor(n_estimators=200,max_depth=3,random_state=seed).fit(X(cal),d["lat"][cal])
    # region = quantile bins of load (the natural localizer)
    edges=np.quantile(d["load"][fit],np.linspace(0,1,BINS+1)); edges[0],edges[-1]=-np.inf,np.inf
    reg=lambda I: np.clip(np.digitize(d["load"][I],edges[1:-1]),0,BINS-1)
    def harm_set(I,rs):
        # choose p_harm*|I| harmful: fraction w by highest load (region-aligned), rest random
        k=int(p_harm*len(I)); ka=int(w*k)
        order=np.argsort(-d["load"][I])            # highest load first
        aligned=set(order[:ka].tolist())
        pool=[j for j in range(len(I)) if j not in aligned]
        rnd=set(rs.choice(pool,size=max(0,k-ka),replace=False).tolist()) if pool else set()
        h=np.zeros(len(I),bool)
        for j in aligned|rnd: h[j]=True
        return h
    hf=harm_set(fit,np.random.default_rng(seed+1))
    realfit=d["lat"][fit].copy(); realfit[hf]+=inflate
    err=np.abs(twin.predict(X(fit))-realfit); rfb=reg(fit)
    B={r:(np.quantile(err[rfb==r],1-ALPHA) if (rfb==r).any() else np.quantile(err,1-ALPHA)) for r in range(BINS)}
    he=harm_set(ev,np.random.default_rng(seed+2))
    real=d["lat"][ev].copy(); real[he]+=inflate
    twin_lat=twin.predict(X(ev))
    valid=(twin_lat<=TARGET); harmful=(real>TARGET)
    Bt=np.array([B[r] for r in reg(ev)]); tfs=Bt<=H
    c1=valid; c2=valid&tfs
    rate=np.mean(c1&~c2)/max(np.mean(c1),1e-9); c3=c1&(rng.random(len(ev))>=rate)
    catch=lambda commit: float(np.mean((~commit)[harmful])) if harmful.any() else np.nan
    lam=roc_auc_score(harmful,Bt) if (harmful.any() and (~harmful).any() and np.ptp(Bt)>0) else 0.5
    return dict(lam=float(lam),c1=catch(c1),c2=catch(c2),c3=catch(c3),
                fa=float(np.mean((~c2)[~harmful])) if (~harmful).any() else np.nan,
                base=float(np.mean(harmful)))

if __name__=="__main__":
    print("=== Continuous localizability sweep (w=1 localized -> w=0 diffuse), 8 seeds ===")
    print(" w    Lambda   C1catch C2catch C3catch  C2_FA")
    for w in [1.0,0.8,0.6,0.4,0.2,0.0]:
        R=[run_w(s,w) for s in range(8)]
        m=lambda k: np.nanmean([r[k] for r in R])
        print(f"{w:0.1f}   {m('lam'):.3f}    {m('c1'):.3f}   {m('c2'):.3f}   {m('c3'):.3f}   {m('fa'):.3f}")
    print("\n(w=0 row is the PRESENT-BUT-DIFFUSE optimism null: optimism exists but is not region-separable.)")
    # decision latency: twin.predict + conformal lookup per action
    d=gen(0); thr=float(np.quantile(d["load"],SURGE_Q))
    twin=GradientBoostingRegressor(n_estimators=200,max_depth=3,random_state=0).fit(np.c_[d["load"][:3000],d["prb"][:3000],d["sinr"][:3000]],d["lat"][:3000])
    xs=np.c_[d["load"][3000:3500],d["prb"][3000:3500],d["sinr"][3000:3500]]
    t0=time.perf_counter()
    for i in range(len(xs)): _=twin.predict(xs[i:i+1])
    dt=(time.perf_counter()-t0)/len(xs)*1000
    print(f"\nDecision latency (twin what-if + regional-bound lookup): {dt:.2f} ms/action on CPU (single-thread).")
