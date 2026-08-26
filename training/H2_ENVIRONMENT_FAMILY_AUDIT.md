# H2 user-confirmed environment-family audit

## Families

- A: `test_2`, `test_3`, `test`, `run2`
- B: `run3`, `run4`
- C: `run5_normal`, `run5_x2`

`run2` was tested both as training data and as evaluation-only weak-response
data. Family C contains one physical run at two playback speeds, so its result
is a consistency diagnostic and not independent accuracy.

## Family A

Using the former global 4% labels produced only 32.56% accuracy because the
`test_3` upper tail did not match the other runs. Based on the prior user review
that `test_3` reaches the 2-3 response family, its former upper tail was mapped
to `2-3%`, while `test` and `test_2` retained the 4% candidate.

| Metric | Corrected A |
|---|---:|
| Complete-run held-out accuracy | 75.71% |
| Video-macro accuracy | 68.36% |
| Recall 0 | 94.44% |
| Recall 1-2 | 37.88% |
| Recall 2-3 | 77.17% |
| Recall 4 | 98.48% |

Retaining weak `run2` in training gave the best minimum recall. Its own held-out
accuracy remained only 30%, confirming that it is useful as a weak-response
anchor but not representative of the family as a whole.

## Family B

Only the shared `0 / 1-2 / 2-3%` ranges can be independently validated because
`run3` has no verified 4% reference.

| Metric | B |
|---|---:|
| Accuracy | 62.36% |
| Recall 0 | 100.00% |
| Recall 1-2 | 41.56% |
| Recall 2-3 | 67.16% |

## Family C

The two files are the same physical run at different playback speeds. Their
shared-range consistency was 83.97%, with recalls 100.00%, 98.39%, and 62.50%
for `0`, `1-2`, and `2-3`. This is evidence that the extraction is repeatable,
but it is not independent validation.

## Decision

Environment families are useful and should be retained. They expose that the
main remaining error is now the adjacent `1-2 <-> 2-3` boundary within each
family, rather than initial-state detection. No family model is exported to the
app yet.

## Next step

Rebuild the `1-2` and `2-3` optical landmarks separately inside A and B. Keep
`run2` as a weak A anchor with limited weight, and treat C as one run until an
independent C-environment recording exists. After the concentration models are
stable, validate calibration-image family selection and concentration jointly.

Artifacts:

- `training/h2_environment_family_analysis.py`
- `training/output/h2_environment_families_v1/metrics.json`
- `training/output/h2_environment_families_v1/family_confusions.png`
