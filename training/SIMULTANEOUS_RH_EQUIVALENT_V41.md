# Simultaneous RH-only-equivalent v41 baseline

## Scope

The v41 scale-aware ROI and deployed RH-only seven-band optical prototypes were
applied directly to 28 uncropped simultaneous clips (RH20--80, runs 2--5).
Each reaction frame was classified independently.  Filename RH was retained
only for grouping and was never used as a target.

The reference for each clip was its recovery tail (RH20/H2=0).  Nine frames
were sampled only to audit stability across the H2 reaction; time and adjacent
frames are not inference features.

## Baseline result

ROI localization completed for all 28 clips.  The uncorrected run-modal
RH-only-equivalent bands were:

| Nominal metadata | Run 2 | Run 3 | Run 4 | Run 5 |
|---:|---|---|---|---|
| 20 | 20--30 | 20--30 | 20--30 | 20--30 |
| 30 | 20--30 | 20--30 | 20--30 | 20--30 |
| 40 | 20--30 | 20--30 | 30--40 | 20--30 |
| 50 | 20--30 | 20--30 | 20--30 | 20--30 |
| 60 | 20--30 | 30--40 | 20--30 | 80--90* |
| 70 | 20--30 | 30--40 | 20--30 | 40--50 |
| 80 | 20--30 | 40--50 | 20--30 | 40--50 |

`*` Run-5 RH60 has prototype distance 12.59 versus mostly sub-1 distances and
is affected by the documented sample rotation near the reaction/recovery
boundary.  It is a quality-control failure, not an RH80--90 result.

## H2 cross-talk correction test

A label-free correction was fitted from within-video changes: droplet LAB
change was regressed on flame LAB change after centering each video.  Every
complete run was held out in turn.  The best regularization (`alpha=1000`) gave
residual-change reductions of:

- Run 2: -2.6%
- Run 3: -10.1%
- Run 4: +0.1%
- Run 5: +0.4%
- Median: -1.2%

The correction is rejected.  Flame colour contains RH response as well as H2
response, so a global linear subtraction removes some real RH signal and does
not transfer between runs.

## Decision

The app testing path must expose both simultaneous H2 and RH estimates. Until
joint correction is validated, the RH number is explicitly marked as an
RH-only-equivalent simultaneous provisional value rather than a final result.
The next viable correction candidate is a spatial droplet mask selected to be
stable in H2-only data but responsive along the reviewed RH-only colour path.
It must be evaluated with complete-run holdout and must not use nominal
simultaneous RH as its label.

Reproducible outputs:

- `training/output/simultaneous_rh_equivalent_v41/predictions.csv`
- `training/output/simultaneous_rh_equivalent_v41/metrics.json`
