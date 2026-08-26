# H2 consensus four-band audit

## Question

Can the current videos support an optical H2 model with the ranges
`0--1 / 2 / 3 / 4%`, if 4% is redefined as the common stable maximum response
and the unusually strong tail of `H2_only_test_2` is removed?

This audit does not export or deploy an application model.

## Data policy

Eight independent cropped runs were evaluated at 2 frames/s (933 labelled
frames).  Every fold held out one complete video.

| Source | 0--1 | 2 | 3 | 4 | Notes |
|---|---|---|---|---|---|
| `H2_only_test_2_more_cropped` | yes | yes | yes | 29--31 s | frames after 31 s excluded |
| `H2_only_test_3_more_cropped` | yes | yes | yes | stable late response | old nominal 4 label not used |
| `H2_only_test_cropped` | yes | yes | yes | stable late response | cross-lighting run |
| `RH20_2_x2_cropped` | yes | yes | no | no | weak run |
| `RH20_3_x2_cropped` | yes | yes | yes | no | weak 2--3 response |
| `RH20_4_cropped` | yes | yes | yes | late endpoint | common maximum |
| `RH20_5_x2_cropped` | yes | yes | yes | late endpoint | strongest RH20 run |
| `RH20_cropped` | yes | yes | yes | no | weaker normal-speed run |

Only clear endpoint neighbourhoods and ramp intervals were retained. Recovery
ramps were excluded; their final endpoints alone supplied 0--1 anchors. The
labels are reference-anchored optical bands, not new independent gas-meter
measurements.

## Held-out result

Seven model families were compared under the same complete-video holdout. The
best video-macro result was shrinkage LDA.

| Metric | Result |
|---|---:|
| Frame-weighted exact accuracy | 0.668 |
| Video-macro exact accuracy | 0.676 |
| Mean absolute error | 0.467 percentage-point labels |
| 0--1 recall | 0.837 |
| 2 recall | 0.624 |
| 3 recall | 0.361 |
| 4 recall | 0.720 |
| Actual 3 predicted as 4 | 0.089 |

Confusion matrix (rows=reference, columns=prediction):

| | 0--1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 0--1 | 226 | 43 | 0 | 1 |
| 2 | 23 | 128 | 45 | 9 |
| 3 | 1 | 92 | 61 | 15 |
| 4 | 0 | 46 | 35 | 208 |

The four-band definition avoids widespread 3-to-4 over-promotion, but it does
not meet the requested 0.85 recall per class. The central problem remains the
2/3 boundary: 92 of 169 reference-3 frames are called 2. This is evidence that
merging 0 and 1 and renaming the common maximum cannot by itself make the
intermediate optical states reproducible across runs.

## Decision

Do not deploy this model yet. Keep the four-band formulation as the preferred
output structure, because it is more honest than the previous five exact
classes, but first replace broad time windows with per-run flame trajectory
landmarks. The next analysis should align each run by its calibration-to-maximum
colour path and audit the candidate 2/3 transition frames visually. Only after
that review should the 2/3 boundary be retrained and compared against this
0.676 video-macro baseline.

Artifacts:

- `training/output/h2_consensus_four_band_v1/metrics.json`
- `training/output/h2_consensus_four_band_v1/consensus_four_band_confusion.png`
- `training/output/h2_consensus_four_band_v1/labelled_fixed_mask_rows.npz`
