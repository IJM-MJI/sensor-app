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
| Frame-weighted exact accuracy | 0.680 |
| Video-macro exact accuracy | 0.693 |
| Mean absolute error | 0.444 percentage-point labels |
| 0--1 recall | 0.852 |
| 2 recall | 0.663 |
| 3 recall | 0.373 |
| 4 recall | 0.709 |
| Actual 3 predicted as 4 | 0.041 |

Confusion matrix (rows=reference, columns=prediction):

| | 0--1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 0--1 | 230 | 39 | 0 | 1 |
| 2 | 12 | 136 | 54 | 3 |
| 3 | 5 | 94 | 63 | 7 |
| 4 | 0 | 48 | 36 | 205 |

The final comparison uses all eleven features common to every fixed flame mask:
mean/median Lab and five within-flame chroma percentiles.  The percentiles
improved the video-macro score from 0.676 to 0.693 and reduced 3-to-4 leakage
from 0.089 to 0.041.  The four-band definition avoids widespread 3-to-4 over-promotion, but it does
not meet the requested 0.85 recall per class. The central problem remains the
2/3 boundary: 94 of 169 reference-3 frames are called 2. This is evidence that
merging 0 and 1 and renaming the common maximum cannot by itself make the
intermediate optical states reproducible across runs.

## Decision

Do not deploy this model yet. Keep the four-band formulation as the preferred
output structure, because it is more honest than the previous five exact
classes. Per-run landmark trimming and plateau reassignment were also tested
and rejected (see `H2_CONSENSUS_LANDMARK_AUDIT.md`). The next model should use
the absolute calibration-frame flame/background descriptors as domain context,
in addition to the current response deltas, and must beat this 0.693
video-macro baseline without reducing 2% or 4% recall.

Artifacts:

- `training/output/h2_consensus_four_band_v1/metrics.json`
- `training/output/h2_consensus_four_band_v1/consensus_four_band_confusion.png`
- `training/output/h2_consensus_four_band_v1/labelled_fixed_mask_rows.npz`
