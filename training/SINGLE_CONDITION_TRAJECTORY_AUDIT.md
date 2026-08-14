# Single-condition optical trajectory audit

Date: 2026-08-14

## Purpose and protocol

This audit checks whether the same supplied H2/RH endpoint has the same optical
colour in a completely different run. It is intentionally independent of the
trained classifier. Only frames within 0.55 s of the user-supplied ramp endpoint
are used; rounded ramp-interior frames are never treated as exact stages.

For each run and endpoint, the median calibrated LAB delta is measured in the
flame mask (H2) or droplet mask (RH). It is compared with leave-one-run-out stage
consensus colours after robust channel scaling. `optical-nearest` is a review
hint, not an automatic replacement label.

## Result

| Task | Endpoint run/stages | Same-stage optical matches | Matching rate |
|---|---:|---:|---:|
| H2-only | 25 | 8 | 32.0% |
| H2O-only | 34 | 9 | 26.5% |

These are cross-run trajectory agreement rates, not application accuracy. The
low values explain why an exact concentration model can fit 5-second blocks from
one run but fails on a held-out recording.

Within each run, response magnitude still usually increases with concentration:

- H2 progress correlation: daylight-5 0.60, indoor-4 0.90, test-2 0.90,
  test-3 0.90, test-indoor 0.40.
- RH progress correlation: daylight recovery 0.89, indoor long 0.94,
  indoor fast 0.54, response-3 0.61, response-6 0.61.

This supports the timeline direction for most recordings. The main failure is
not a single global timeline shift: the optical path changes by run, and the
fast response recordings contain non-monotonic intermediate endpoints.

## Highest-priority review findings

- `H2_only_test_3`: after all five H2 runs use the same crop domain, its
  within-run progression rises to 0.90. Cross-run colours still overlap other
  stages, but the earlier apparent endpoint reversal was partly a domain-scale
  artifact rather than a reason to relabel the timeline.
- `H2_only_4`: endpoints H2 2--4% do progress within the run, but their absolute
  path lies near other runs' low-stage colours.
- `H2_only_5`: a matching `_cropped` source was supplied after the first audit.
  Cross-run endpoint agreement increased from 7/25 to 8/25 and its within-run
  progress correlation is now 0.60. It remains a valuable daylight holdout.
- `H2O_only_2_extract` (indoor fast): RH40 responds strongly, the intermediate
  RH50--80 endpoints compress, and RH90 moves again. Exact 10% stages are not a
  single monotonic colour path in this run.
- `H2O_only_extract_3min` + `extract_extra` (indoor long): overall progression is
  strong (0.94), making this the best RH trajectory anchor despite cross-run
  offsets.
- Response-3/6: the rapid RH endpoints give mixed progress (0.61 each), so they
  are valuable for state/shape coverage but should receive lower weight for
  exact 10% concentration boundaries.

## Generated review artifacts

The local output directory is `training/output/single_condition_trajectory_audit`:

- `single_condition_trajectory_audit.png/.pdf/.svg`: publication-ready run paths.
- `h2_trajectory_review.jpg`, `rh_trajectory_review.jpg`: endpoint frames whose
  nominal and optical-nearest stages disagree.
- `trajectory_audit.csv`: one row per run/stage with video and timestamp.
- `trajectory_audit.json`: summary, mismatches and within-run progress metrics.

No labels or deployed models are changed by this audit.

## Next modelling action

Use the reliable long/ordered runs as exact-stage anchors and treat fast/mixed
runs as ordered or interval-censored supervision rather than exact class labels.
For H2, verify the high-priority test-3 endpoint and preserve daylight-5 as a
domain holdout. Retrain H2 first, then RH, and accept a model only if whole-run
held-out exact accuracy and stage-balanced accuracy improve together.
