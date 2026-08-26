# H2 confirmed three-band audit

## Corrected visual constraint

The two user-reviewed photographs are not two neighboring low-H2 states.

- Photograph 1: **Initial**
- Photograph 2: **H2 2-3%**

The paired flame-color shift is small (approximately `Delta L*=-1`,
`Delta a*=-2`, `Delta b*=+1`). Therefore, a small move toward green can already
represent H2 2-3%; a large visible color change must not be required by the
state gate.

## Re-evaluation

Existing concentration references were regrouped into three operational bands:

- `0-1%`
- `2-3%`
- `4%`

Validation held out one complete video at a time. The selected shrinkage-LDA
model used 933 rows and obtained:

| Metric | Result |
|---|---:|
| Frame accuracy | 83.49% |
| Video-macro accuracy | 85.32% |
| Band MAE | 0.166 bands |
| Recall, 0-1% | 85.19% |
| Recall, 2-3% | 92.25% |
| Recall, 4% | 70.59% |

The `0-1%` and `2-3%` bands now meet the 0.85 recall target. The `4%` band does
not, so this candidate is **not deployed to the app**.

## Why the 4% band fails

All 85 reference-4 frames predicted as 2-3% came from only three runs:

| Run | Reference 4 predicted as 2-3 |
|---|---:|
| `test_3` | 51 |
| `run4` | 18 |
| `run5_x2` | 16 |

These are precisely runs whose physical arrival at 4% was previously
questioned. The error concentration therefore points to an unreliable 4%
reference definition, rather than a general inability to distinguish the two
confirmed photographs.

## Next decision

Before another model change, rebuild the 4% reference pool using only frames
that have independent evidence of a full endpoint. Do not improve the reported
score by relabeling ambiguous endpoints from appearance alone. If no trustworthy
4% run exists, the scientifically honest application output is `0-1% / 2-3% /
upper endpoint not calibrated`, until a verified 4% run is recorded.

Artifacts:

- `training/output/h2_confirmed_three_band_v1/metrics.json`
- `training/output/h2_confirmed_three_band_v1/three_band_confusion.png`
