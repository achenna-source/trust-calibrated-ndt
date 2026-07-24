# Pre-registration — alternative region function for the real-data trust gate

Written BEFORE running the variant (the CQI-region result is already known: TFS ~= random).
Purpose: test whether the negative real-data result is robust to the choice of gating region,
or an artefact of the CQI partition. Report the outcome WHATEVER it is; no selection of the
better of several — this is ONE pre-specified alternative with an a-priori rationale.

## Alternative region function (pre-specified)
`region = quantile bins of the twin's PREDICTED outcome magnitude (predicted log-buffer)`, N=4 bins.

**Rationale (a-priori):** twin optimism is under-prediction of congestion; it should concentrate
where the twin predicts LOW buffer (says "fine") while extrapolating out of its training regime.
Binning by predicted magnitude is the textbook "where is the model extrapolating" localizer, and
is the natural region function if CQI does not align with the twin's error.

## Success criterion (pre-specified)
The variant "rescues" a positive result iff C2 catch exceeds C3 catch by >= 10 percentage points
at comparable false-alarm. Otherwise the negative is deemed ROBUST to region choice.

## Localizability diagnostic (pre-specified, reported for every setup)
`localizability AUC` = ROC-AUC of using each record's regional conformal bound B(region) to predict
the binary label "harmful". AUC ~ 0.5 => the regional bound does NOT separate harmful from safe
(diffuse error, TFS degrades to random); AUC high => error is localizable, TFS is expected to help.
This is the operator-facing preflight the paper proposes: compute it on held-out data before trusting the gate.
We report it for (i) synthetic surge-region, (ii) real CQI-region, (iii) real predicted-magnitude region.
