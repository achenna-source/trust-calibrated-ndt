"""extra_real.py -- reviewer-requested real-data extras on Colosseum:
(1) more (Lambda, gate-benefit) points via different policy/slice splits -> real spectrum, not n=1;
(2) a DIRECT-HARM classifier baseline (the reviewers' 'ignored alternative': if you have harm labels,
    gate on a predictor of harm directly, bypassing the twin) -- honest comparison;
(3) SLA-threshold sensitivity of the negative result.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import ndt_real_experiment as R

FEATURES=R.FEATURES; ALPHA=R.ALPHA; NB=R.N_CQI_BINS

def one_split(df, train_pol, test_pol, floor_log, seed):
    tr=df[df.scheduling_policy==train_pol]
    sh=df[df.scheduling_policy==test_pol].sample(frac=1.0,random_state=seed)
    m=len(sh)//2; fit,test=sh.iloc[:m],sh.iloc[m:]
    twin=GradientBoostingRegressor(n_estimators=200,max_depth=3,random_state=seed).fit(tr[FEATURES].values,tr["y"].values)
    edges=np.quantile(fit["dl_cqi"].values,np.linspace(0,1,NB+1)); edges[0],edges[-1]=-np.inf,np.inf
    reg=lambda x:np.clip(np.digitize(x,edges[1:-1]),0,NB-1)
    pf=twin.predict(fit[FEATURES].values); rfres=np.abs(pf-fit["y"].values); rf=reg(fit["dl_cqi"].values)
    B={b:(np.quantile(rfres[rf==b],1-ALPHA) if (rf==b).sum()>=10 else np.quantile(rfres,1-ALPHA)) for b in range(NB)}
    yp=twin.predict(test[FEATURES].values); yr=test["y"].values
    Bt=np.array([B[b] for b in reg(test["dl_cqi"].values)])
    valid=yp<=floor_log; harmful=yr>floor_log
    h=float(np.median(list(B.values()))); tfs=Bt<=h
    c2=valid&tfs
    catch=lambda c: float(np.mean((~c)[harmful])) if harmful.any() else np.nan
    fa=lambda c: float(np.mean((~c)[~harmful])) if (~harmful).any() else np.nan
    lam=roc_auc_score(harmful,Bt) if (harmful.any() and (~harmful).any() and np.ptp(Bt)>0) else 0.5
    # DIRECT-HARM baseline: classifier state->harmful trained on fit, gate on its score at matched block rate
    clf=GradientBoostingClassifier(n_estimators=150,max_depth=3,random_state=seed)
    yhar_fit=(fit["y"].values>floor_log).astype(int)
    dh_lam=np.nan; dh_catch=np.nan; dh_fa=np.nan
    if len(np.unique(yhar_fit))==2:
        clf.fit(fit[FEATURES].values,yhar_fit)
        score=clf.predict_proba(test[FEATURES].values)[:,1]
        dh_lam=roc_auc_score(harmful,score) if (harmful.any() and (~harmful).any()) else 0.5
        blockrate=float(np.mean(valid&~tfs))/max(float(np.mean(valid)),1e-9)  # match C2 extra-blocking
        thr=np.quantile(score,1-blockrate) if blockrate>0 else np.inf
        dhc=valid&(score<thr)
        dh_catch=catch(dhc); dh_fa=fa(dhc)
    return dict(lam=lam,c2_catch=catch(c2),c2_fa=fa(c2),base=float(np.mean(harmful)),
                dh_lam=dh_lam,dh_catch=dh_catch,dh_fa=dh_fa)

if __name__=="__main__":
    df=R.load(); floor=R.FLOOR_LOG
    print("=== (1) Real (Lambda, gate) across splits, 6 seeds; and DIRECT-HARM baseline ===")
    print("split           Lambda  C2catch C2FA   J(catch-FA) | directHARM: AUC  catch  FA")
    for (a,b) in [(0,2),(2,0)]:
        Rs=[one_split(df,a,b,floor,s) for s in range(6)]
        mm=lambda k:np.nanmean([r[k] for r in Rs])
        print(f"policy {a}->{b}    {mm('lam'):.3f}   {mm('c2_catch'):.3f}  {mm('c2_fa'):.3f}  {mm('c2_catch')-mm('c2_fa'):+.3f}      |          {mm('dh_lam'):.3f} {mm('dh_catch'):.3f} {mm('dh_fa'):.3f}")
    print("\n=== (3) SLA-threshold sensitivity (policy 0->2, 6 seeds) ===")
    print("floor_bytes  base_harm  Lambda  J(catch-FA)")
    for fb in [10000,20000,40000,80000]:
        fl=np.log1p(fb)
        Rs=[one_split(df,0,2,fl,s) for s in range(6)]
        mm=lambda k:np.nanmean([r[k] for r in Rs])
        print(f"{fb:8d}    {mm('base'):.3f}     {mm('lam'):.3f}   {mm('c2_catch')-mm('c2_fa'):+.3f}")
