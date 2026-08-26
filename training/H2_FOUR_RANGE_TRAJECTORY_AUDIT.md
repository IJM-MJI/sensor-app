# H2 four-range trajectory matching audit

## Goal

Rebuild `0 / 1-2 / 2-3 / 4%` reference frames from optical order within each
video, rather than accepting every nominal timeline interval as an equivalent
cross-video state.

## Method

- A complete video is excluded from every training and tuning operation.
- The other videos produce single-frame range probabilities.
- Probabilities are monotonically ordered through each reaction trajectory.
- Only frames close to a stable optical range landmark are retained.
- Every run/range block is temporally capped and equally weighted.
- The final student still requires only calibration plus one measurement image.

Three teacher pools were compared:

- `all`: includes the additional angle-80 and weak H2-only candidates;
- `official`: the eight runs in the existing four-range evaluation;
- `trusted90`: the official 90-degree pool without difficult `run2` teacher
  frames. `run2` remains fully present as a held-out evaluation video.

## Results

| Teacher pool | Accuracy | Video macro | Recall 0 | Recall 1-2 | Recall 2-3 | Recall 4 |
|---|---:|---:|---:|---:|---:|---:|
| All | 64.35% | 66.33% | 91.00% | 59.02% | 47.34% | 68.86% |
| Official | 61.73% | 59.60% | 89.00% | 49.76% | 55.03% | 64.71% |
| Trusted 90 | 63.43% | 62.95% | 92.00% | 51.22% | 52.07% | 68.86% |

The recall-balanced selection is `trusted90_logistic_tol0.40_cap8`. Compared
with the previous non-trajectory recall-balanced candidate, its limiting
recall improved from 45.56% to 51.22%, and 2-3% recall improved from 45.56% to
52.07%.

## Decision

Trajectory ordering and run balancing are useful, but the result is not ready
for application deployment. The principal remaining confusion is adjacent and
ordered: `1-2 <-> 2-3` and `2-3 <-> 4`. This indicates different response-scale
families between videos, rather than random state detection failure.

## Next step

Fit a calibration-selected family model. Cluster runs using only information
available at runtime from the calibration image, then train a separate optical
range boundary within each family. Family selection and concentration must be
evaluated together while holding out a complete video. If calibration cannot
reliably select the correct family, the four-range model cannot be justified
from the current videos and new verified endpoints will be required.

Artifacts:

- `training/h2_four_range_trajectory_matching.py`
- `training/output/h2_four_range_trajectory_v1/metrics.json`
- `training/output/h2_four_range_trajectory_v1/trajectory_matching_confusion.png`
