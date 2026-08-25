# H2 calibration-circle lock audit

## Motivation

The browser locks the ROI geometry from the calibration photograph.  The
previous training cache periodically re-detected the circle and, in the
daylight `H2_only_test_3` run, changed its radius from 53 to 78 pixels.  This
allowed ROI motion and exposure changes to supervise the concentration model
even though those changes are absent after browser calibration.

## Data and protocol

- H2-only cropped videos: `test`, `test_2`, `test_3`, run 4, and run 5.
- The circle selected near 2 s was reused for every frame in each run.
- Evaluation remained complete-video held-out; frames from a held-out video
  were never used to fit that fold.
- Run 4 and run 5 were also audited separately because the user judged their
  maximum physical response to be only about 2--3%, not a trustworthy 4%.

## Results

| Configuration | Exact | Video-macro exact | Within one stage | MAE (stages) |
|---|---:|---:|---:|---:|
| Deployed moving-ROI training | 0.445 | 0.440 | 0.962 | 0.596 |
| Calibration-locked ROI, all five nominal timelines | 0.384 | 0.404 | 0.757 | 0.982 |
| Calibration-locked ROI, reliable exact runs only | 0.553 | 0.433 | 0.786 | 0.840 |

Reliable-run per-video exact accuracy with the calibration lock was:

| Held-out run | Exact | Within one stage | MAE |
|---|---:|---:|---:|
| `H2_only_test` | 0.413 | 0.875 | 0.713 |
| `H2_only_test_2` | 0.167 | 0.333 | 2.206 |
| `H2_only_test_3` | 0.720 | 0.914 | 0.414 |

## Decision

Do not replace the deployed global model.  Geometry locking materially helps
the daylight `test_3` domain, but a single locked model does not transfer back
to `test_2`.  The next experiment must separate indoor and daylight reference
profiles (or obtain another trustworthy daylight H2 run) and evaluate the
profile selector independently.  A model fitted and tested only on `test_3`
may be useful as an explicitly labelled reference-matching mode, but it is not
an independent held-out validation.
