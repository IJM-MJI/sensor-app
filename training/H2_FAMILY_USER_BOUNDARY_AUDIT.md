# H2 family boundary refinement from user review

## Applied review

- Kept as 2-3: `test` 30-35 s, `test_3` 88 s, `run3` 55-59 s,
  and `run4` 84-90 s.
- Mapped `test_3` 24-28 s to 1-2.
- Mapped `run4` 65-78 s to 2-3 while preserving 55-64.5 s as its lower
  landmark.

## Results

| Metric | A before | A reviewed | B before | B reviewed |
|---|---:|---:|---:|---:|
| Accuracy | 82.69% | 86.82% | 69.10% | 73.60% |
| Recall 0 | 97.22% | 97.22% | 97.06% | 97.06% |
| Recall 1-2 | 87.88% | 92.00% | 62.34% | 72.00% |
| Recall 2-3 | 74.43% | 79.52% | 62.69% | 65.96% |
| Recall 4 | 96.97% | 98.48% | n/a | n/a |

The review improves both families without changing correctly judged points.
Family A now exceeds 0.85 for overall accuracy and three of four recalls. Its
remaining limiting range is 2-3.

## Asymmetric threshold audit

An A-family bias toward 2-3 was swept under the constraint that recalls for 0,
1-2, and 4 remain at least 0.85. The best feasible bias was `+0.625`:

- recall 0: 97.22%
- recall 1-2: 85.33%
- recall 2-3: 81.90%
- recall 4: 98.48%

It does not reach the all-range 0.85 target, so the threshold override is not
deployed.

## Decision and next step

Keep the reviewed labels and the unbiased stable-landmark candidate. Do not
export either family model yet. The next automatic experiment should add
spatial flame-colour distribution features: separate upper/lower and
left/right `a*b*` summaries inside the fixed flame mask. The current global
medians can hide a partial yellow-to-green change and may be the reason the
confirmed 2-3 frames still look like 1-2 numerically.

Artifacts:

- `training/h2_family_user_boundary_refinement.py`
- `training/h2_family_a_bias_audit.py`
- `training/output/h2_family_landmarks_user_v2/metrics.json`
- `training/output/h2_family_landmarks_user_v2/user_boundary_confusions.png`
- `training/output/h2_family_a_bias_v1/metrics.json`
- `training/output/h2_family_a_bias_v1/bias_tradeoff.png`
