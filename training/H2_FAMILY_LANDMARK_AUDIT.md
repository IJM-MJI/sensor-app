# H2 family-specific landmark refinement

## Method

Within user-confirmed families A and B, training frames were restricted to
stable optical landmarks near each run/range center. Every run/range block was
temporally capped, while every labelled frame in the held-out video remained
in evaluation. Feature families included flame `a*`, `a*b*`, six Lab summaries,
and all eleven Lab/chroma summaries.

## Family A

The selected candidate uses flame `a*b*`, shrinkage LDA, 90% central landmark
retention, and at most six frames per run/range.

| Metric | Before | Refined |
|---|---:|---:|
| Accuracy | 75.71% | 82.69% |
| Video-macro accuracy | 68.36% | 82.62% |
| Recall 0 | 94.44% | 97.22% |
| Recall 1-2 | 37.88% | 87.88% |
| Recall 2-3 | 77.17% | 74.43% |
| Recall 4 | 98.48% | 96.97% |

The major residual is 56 reference `2-3%` frames predicted as `1-2%`: 21 from
`test`, 32 from `test_3`, and 3 from `test_2`.

## Family B

The selected candidate uses six Lab summaries, shrinkage LDA, 50% central
landmark retention, and at most six frames per run/range.

| Metric | Before | Refined |
|---|---:|---:|
| Accuracy | 62.36% | 69.10% |
| Recall 0 | 100.00% | 97.06% |
| Recall 1-2 | 41.56% | 62.34% |
| Recall 2-3 | 67.16% | 62.69% |

The two videos have a shifted boundary: all 20 late `run3` 2-3 frames are
predicted as 1-2, while 29 `run4` 1-2 frames are predicted as 2-3.

## Decision

The stable-landmark strategy is retained as the best family-A candidate, but
it is not deployed because A's 2-3 recall and every B response recall remain
below 0.85. Only four independent A runs and two B runs are available, so the
large configuration comparison is an audit result, not a final unbiased model
selection.

## Review needed

The boundary atlas contains representative remaining errors. User review of
the shown state, not a blanket timeline shift, will decide whether to relabel
specific optical intervals or retain them as genuinely difficult examples.

Artifacts:

- `training/h2_family_landmark_refinement.py`
- `training/make_h2_family_boundary_review.py`
- `training/output/h2_family_landmarks_v1/metrics.json`
- `training/output/h2_family_landmarks_v1/family_landmark_confusions.png`
- `training/output/h2_family_landmarks_v1/boundary_review_atlas.jpg`
