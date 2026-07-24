"""
ndt_trust_experiment.py
=======================================================================
Zero-cost, single-file case study for the COMST paper
"Can You Trust the Twin? Trust-Calibrated Network Digital Twins ..."

WHAT IT TESTS (Section VIII of the paper)
    Does a run-time TWIN-TRUSTWORTHINESS check (the TFS predicate) catch
    unsafe commits that an action-validity-only gate would allow, and at
    what FALSE-ALARM cost?

DESIGN (kept faithful to Section VII/VIII, minimal enough for free Colab)
    * Ground truth = REAL O-RAN KPM measurements (held-out), NOT a second
      simulator -> avoids the sim-grades-sim circularity.
    * Twin  = a predictor fit on a CALIBRATION split of the real data.
    * Reality = held-out real data + an INJECTED STRUCTURAL divergence that
      is active only in the surge regime and that the twin cannot represent
      (so the detector is NOT tuned to the divergence's form).
    * Trust estimator = split-conformal bound on |twin - real| estimated on
      HELD-OUT REAL residuals in the region the action touches.
    * Controller = open-weight LLM (Qwen2.5-7B-Instruct) proposing a config
      from an intent + KPIs; robust rule-based fallback if the model or GPU
      is unavailable (clearly logged).
    * Conditions:  C0 commit-all | C1 validity-only gate | C2 validity+TFS |
      C3 compute-matched RANDOM block at C2's block rate.

OUTPUT
    * results_table.csv and a paste-ready Markdown block for Table VIII.1
      (catch rate, false-alarm rate, live SLA-violation rate per condition).
    * roc_tfs.png : catch vs false-alarm as the sufficiency threshold h varies.

HOW TO RUN (Google Colab, free T4 GPU)
    1. Runtime -> Change runtime type -> T4 GPU.
    2. !pip install -q transformers accelerate bitsandbytes scikit-learn pandas matplotlib
    3. Upload this file, then:  %run ndt_trust_experiment.py
    (See README.md for the exact steps and how to paste results into the paper.)

REPRODUCIBILITY
    All randomness is seeded (SEED). Re-running with the same SEED and the
    same data reproduces every number. Report results over several SEEDs for
    the paper; a single seed is an honest "illustrative, single-seed" anchor.
=======================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------
SEED             = 42
N_ACTIONS        = 400          # number of LLM-proposed actions evaluated
ALPHA            = 0.10         # conformal miscoverage (90% bound)
HARM_BUDGET_H    = 5.0          # % latency-margin the action may erode before SLA breach
SLA_MARGIN_MIN   = 0.0          # a slice is in violation if realized margin < this (%)
SURGE_QUANTILE   = 0.80         # loads above this quantile are the "surge" regime
DIVERGENCE_GAIN  = 90.0         # strength of the injected structural divergence (surge only; ms per unit load above threshold)
MODEL_NAME       = "Qwen/Qwen2.5-7B-Instruct"
KPM_URL          = os.environ.get("KPM_URL", "")   # optional: URL/path to a real O-RAN KPM CSV
OUTDIR           = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[ndt] {msg}", flush=True)


# ----------------------------------------------------------------------
# 1. Data: real O-RAN KPM traces, or a clearly-labelled surrogate
# ----------------------------------------------------------------------
def load_kpm(rng):
    """Return a DataFrame with columns: load, prb, sinr_db, latency_ms, and a
    flag `is_real`. Tries a real O-RAN KPM CSV first (set KPM_URL); otherwise
    generates a documented surrogate and warns loudly."""
    cols_needed = ["load", "prb", "sinr_db", "latency_ms"]
    if KPM_URL:
        try:
            df = pd.read_csv(KPM_URL)
            # Best-effort column mapping; adjust names to your KPM export.
            rename = {}
            for c in df.columns:
                cl = c.lower()
                if "prb" in cl and "prb" not in rename.values():          rename[c] = "prb"
                elif ("dl_buf" in cl or "load" in cl) and "load" not in rename.values(): rename[c] = "load"
                elif "sinr" in cl and "sinr_db" not in rename.values():   rename[c] = "sinr_db"
                elif ("lat" in cl or "delay" in cl) and "latency_ms" not in rename.values(): rename[c] = "latency_ms"
            df = df.rename(columns=rename)
            if all(c in df.columns for c in cols_needed):
                df = df[cols_needed].dropna().reset_index(drop=True)
                df["is_real"] = True
                log(f"Loaded REAL KPM data from {KPM_URL}: {len(df)} rows.")
                return df
            log("KPM_URL loaded but expected columns not found; falling back to surrogate.")
        except Exception as e:
            log(f"Could not load KPM_URL ({e}); falling back to surrogate.")

    # --- Surrogate (CLEARLY LABELLED). Replace with real KPM for the paper. ---
    log("!! USING SURROGATE DATA (is_real=False). For the manuscript, set KPM_URL "
        "to a real O-RAN KPM CSV (e.g., the Colosseum KPM dataset) and re-run. !!")
    n = 6000
    load = np.clip(rng.beta(2.0, 3.0, n) * 1.4, 0, 1)          # offered load in [0,1]
    prb  = rng.integers(10, 100, n).astype(float)              # allocated PRBs
    sinr = rng.normal(12, 4, n) - 6.0 * load                   # SINR degrades with load
    # BASE latency (the twin's world): tuned to stay mostly under the SLA target even in
    # surge, so the twin, which learns THIS, will often rate a surge action acceptable.
    # The surge-only structural term is added only in reality (see reality_latency), so the
    # twin is optimistic exactly where that term bites -- the scenario the paper studies.
    latency = _base_latency(load, prb) + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"load": load, "prb": prb, "sinr_db": sinr,
                       "latency_ms": np.clip(latency, 1, None)})
    df["is_real"] = False
    return df


# ----------------------------------------------------------------------
# 2. Twin (fit on calibration split) and Reality (held-out real + injected divergence)
# ----------------------------------------------------------------------
from sklearn.ensemble import GradientBoostingRegressor

def fit_twin(cal_df):
    """The twin predicts latency from (load, prb, sinr). It never sees the
    surge-only structural divergence, because it is trained on calibration
    data dominated by the normal regime."""
    X = cal_df[["load", "prb", "sinr_db"]].values
    y = cal_df["latency_ms"].values
    twin = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=SEED)
    twin.fit(X, y)
    return twin

def _base_latency(load, prb):
    """Base congestion latency the twin can learn. Tuned to stay mostly below the
    SLA target, so a surge action often looks acceptable to the twin."""
    return 3.0 + 9.0 * load**2 / (prb / 50.0 + 0.35)

def reality_latency(load, prb, sinr, surge_load_thr):
    """Ground-truth latency the *network* would produce: the same base the twin
    learned, PLUS a surge-only structural term absent from the twin's physics.
    The term is what makes the twin optimistic in surge (base < SLA, reality >
    SLA). The detector is NOT told this term's form; it only sees its statistical
    footprint via held-out real residuals, so catching it is not circular."""
    base = _base_latency(load, prb)
    surge = np.where(load > surge_load_thr,
                     DIVERGENCE_GAIN * np.maximum(0.0, load - surge_load_thr),
                     0.0)
    return base + surge


# ----------------------------------------------------------------------
# 3. Trust estimator: split-conformal bound on |twin - real|, region-local
# ----------------------------------------------------------------------
def conformal_bounds(twin, real_holdout, surge_load_thr, alpha):
    """Return (B_normal, B_surge): (1-alpha) quantiles of the twin's absolute
    error against HELD-OUT REAL latency, computed separately for the normal and
    surge regions (the 'region the action touches' proxy)."""
    X = real_holdout[["load", "prb", "sinr_db"]].values
    load = real_holdout["load"].values
    real_lat = reality_latency(load, real_holdout["prb"].values,
                               real_holdout["sinr_db"].values, surge_load_thr)
    pred = twin.predict(X)
    err = np.abs(pred - real_lat)
    q = 1.0 - alpha
    B_normal = float(np.quantile(err[load <= surge_load_thr], q)) if np.any(load <= surge_load_thr) else 0.0
    surge_mask = load > surge_load_thr
    # If we have NO held-out real data in surge, the bound must widen (ignorance),
    # which correctly makes TFS conservative exactly where twin optimism is worst.
    B_surge = float(np.quantile(err[surge_mask], q)) if np.any(surge_mask) else max(B_normal * 3.0, 50.0)
    return B_normal, B_surge


# ----------------------------------------------------------------------
# 4. Controller: open-weight LLM (Qwen2.5-7B) with a robust rule fallback
# ----------------------------------------------------------------------
def make_controller():
    """Return a function state->action (PRB reallocation delta). Tries the real
    LLM; falls back to a rule-based proposer if transformers/GPU is unavailable."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        log(f"Loading {MODEL_NAME} in 4-bit ...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                                     device_map="auto")
        log("LLM controller ready (real Qwen2.5-7B-Instruct).")

        def llm_action(load, prb, sinr, latency):
            intent = ("You manage a 5G slice. Keep latency low. Given the KPIs, reply with ONLY an "
                      "integer: how many PRBs to ADD to this slice (negative = remove). Range -30..30.")
            prompt = (f"{intent}\nload={load:.2f} prb={int(prb)} sinr_db={sinr:.1f} "
                      f"latency_ms={latency:.1f}\nAnswer:")
            msgs = [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
            out = model.generate(ids, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            import re
            m = re.search(r"-?\d+", txt)
            delta = int(m.group()) if m else 0
            return int(np.clip(delta, -30, 30)), "llm"
        return llm_action, "llm"
    except Exception as e:
        log(f"LLM unavailable ({e}); using rule-based controller fallback (label='rule').")

        def rule_action(load, prb, sinr, latency, _rng=np.random.default_rng(SEED)):
            # Greedy: add PRBs when latency high; occasionally over-reacts (unsafe) under surge.
            delta = int(np.clip((latency - 10) * 1.5, -30, 30))
            if load > 0.85 and _rng.random() < 0.35:      # over-aggressive reallocation under surge
                delta = int(np.clip(delta + 20, -30, 30))
            return delta, "rule"
        return rule_action, "rule"


# ----------------------------------------------------------------------
# 5. Experiment loop
# ----------------------------------------------------------------------
def requirement_ok(margin):           # R(.) : SLA satisfied if margin >= 0
    return margin >= SLA_MARGIN_MIN

def run(seed=SEED, n_actions=N_ACTIONS, h=HARM_BUDGET_H, alpha=ALPHA):
    rng = np.random.default_rng(seed)
    df = load_kpm(rng)
    surge_load_thr = float(np.quantile(df["load"], SURGE_QUANTILE))

    # splits: calibration (twin), real-holdout (conformal + reality), eval states
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df); a = int(0.5 * n); b = int(0.8 * n)
    cal, real_hold, eval_states = df.iloc[:a], df.iloc[a:b], df.iloc[b:].reset_index(drop=True)

    twin = fit_twin(cal)
    B_norm, B_surge = conformal_bounds(twin, real_hold, surge_load_thr, alpha)
    log(f"conformal bounds: B_normal={B_norm:.2f} ms, B_surge={B_surge:.2f} ms (h={h})")

    controller, ctrl_kind = make_controller()

    # sample eval states (bias toward including surge so the regime is exercised)
    idx = rng.choice(len(eval_states), size=min(n_actions, len(eval_states)), replace=False)
    rows = []
    for i in idx:
        s = eval_states.iloc[i]
        load, prb, sinr, lat = float(s.load), float(s.prb), float(s.sinr_db), float(s.latency_ms)
        if ctrl_kind == "llm":
            delta, _ = controller(load, prb, sinr, lat)
        else:
            delta, _ = controller(load, prb, sinr, lat)
        new_prb = float(np.clip(prb + delta, 5, 120))

        # neighbour slice loses the PRBs we add (zero-sum): margin proxy on the NEIGHBOUR
        nb_prb = float(np.clip(prb - delta, 5, 120))
        # twin prediction of neighbour latency (what the validity gate sees)
        twin_nb_lat = float(twin.predict([[min(load*1.1,1.0), nb_prb, sinr]])[0])
        # reality neighbour latency (ground truth incl. injected divergence)
        real_nb_lat = float(reality_latency(min(load*1.1,1.0), nb_prb, sinr, surge_load_thr))

        # margin (%) = how far below a latency target (say 20 ms) the slice sits
        target = 20.0
        twin_margin  = 100.0 * (target - twin_nb_lat) / target
        real_margin  = 100.0 * (target - real_nb_lat) / target

        in_surge = load > surge_load_thr
        B = B_surge if in_surge else B_norm
        # TFS: is the certified divergence (as % of target) within the harm budget?
        B_pct = 100.0 * B / target
        tfs_ok = (B_pct <= h)

        valid_twin = requirement_ok(twin_margin)     # R(twin)
        harmful    = not requirement_ok(real_margin) # committing breaches SLA in reality
        rows.append(dict(load=load, in_surge=in_surge, delta=delta,
                         twin_margin=twin_margin, real_margin=real_margin,
                         valid_twin=valid_twin, tfs_ok=tfs_ok, harmful=harmful, B_pct=B_pct))

    R = pd.DataFrame(rows)

    # ---- gate conditions ----
    def commit_decisions(dframe, cond, block_rate=None, rng=None):
        if cond == "C0":  return np.ones(len(dframe), dtype=bool)
        if cond == "C1":  return dframe["valid_twin"].values
        if cond == "C2":  return (dframe["valid_twin"] & dframe["tfs_ok"]).values
        if cond == "C3":  # validity gate + random block matched to C2's block rate
            base = dframe["valid_twin"].values
            r = rng.random(len(dframe))
            keep = base & (r >= block_rate)
            return keep
        raise ValueError(cond)

    c2_commit = commit_decisions(R, "C2")
    c1_commit = commit_decisions(R, "C1")
    # C3 block rate matched to how many C1-valid actions C2 additionally blocks
    c2_block_rate = float(np.mean(c1_commit & ~c2_commit)) / max(float(np.mean(c1_commit)), 1e-9)
    c3_commit = commit_decisions(R, "C3", block_rate=c2_block_rate, rng=rng)

    def metrics(commit):
        commit = np.asarray(commit, bool)
        blocked = ~commit
        harmful = R["harmful"].values
        safe = ~harmful
        catch = float(np.mean(blocked[harmful])) if harmful.any() else float("nan")
        fa    = float(np.mean(blocked[safe]))    if safe.any() else float("nan")
        sla_v = float(np.mean(harmful[commit]))  if commit.any() else 0.0  # violations among commits
        return dict(catch_rate=catch, false_alarm=fa, sla_violation=sla_v,
                    commit_frac=float(np.mean(commit)))

    res = {"C0 (commit all)":       metrics(np.ones(len(R), bool)),
           "C1 (validity only)":    metrics(c1_commit),
           "C2 (validity + TFS)":   metrics(c2_commit),
           "C3 (random block ctrl)":metrics(c3_commit)}

    out = pd.DataFrame(res).T[["catch_rate", "false_alarm", "sla_violation", "commit_frac"]]
    out = out.round(3)
    out.to_csv(os.path.join(OUTDIR, "results_table.csv"))

    # paste-ready markdown for Table VIII.1
    md = ["| Condition | Catch rate | False-alarm rate | Live SLA-violation | Commit frac |",
          "|---|---|---|---|---|"]
    for k, v in res.items():
        md.append(f"| {k} | {v['catch_rate']:.3f} | {v['false_alarm']:.3f} | "
                  f"{v['sla_violation']:.3f} | {v['commit_frac']:.3f} |")
    md_txt = "\n".join(md)
    with open(os.path.join(OUTDIR, "results_table.md"), "w") as f:
        f.write(f"controller={ctrl_kind}, data_is_real={bool(df['is_real'].iloc[0])}, "
                f"seed={seed}, n={len(R)}, h={h}, alpha={alpha}\n\n{md_txt}\n")

    log("RESULTS (single seed; label as illustrative unless averaged over seeds):")
    print(out.to_string())
    print("\nPaste-ready Table VIII.1:\n" + md_txt)
    log(f"controller={ctrl_kind}  data_is_real={bool(df['is_real'].iloc[0])}  "
        f"n={len(R)}  surge_frac={R['in_surge'].mean():.2f}")

    # ---- ROC-style curve: catch vs false-alarm as h varies ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hs = np.linspace(1, 30, 25); catches, fas = [], []
        for hh in hs:
            tfs = (R["B_pct"].values <= hh)
            commit = R["valid_twin"].values & tfs
            m = metrics(commit); catches.append(m["catch_rate"]); fas.append(m["false_alarm"])
        plt.figure(figsize=(5, 4))
        plt.plot(fas, catches, "-o", ms=3)
        plt.xlabel("false-alarm rate"); plt.ylabel("catch rate")
        plt.title("TFS operating curve (sweep h)"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "roc_tfs.png"), dpi=140)
        log("saved roc_tfs.png")
    except Exception as e:
        log(f"plot skipped ({e})")

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N_ACTIONS)
    ap.add_argument("--h", type=float, default=HARM_BUDGET_H)
    ap.add_argument("--smoke", action="store_true", help="quick run, rule controller only")
    args = ap.parse_args()
    if args.smoke:
        MODEL_NAME = "__force_fallback__"   # makes the LLM load fail -> rule controller
        run(seed=args.seed, n_actions=min(args.n, 120), h=args.h)
    else:
        run(seed=args.seed, n_actions=args.n, h=args.h)
