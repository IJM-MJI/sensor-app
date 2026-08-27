# H2 environment-routed concentration model v1

## Decision

The automatic A/B environment router remains the first stage. It now selects the
H2 concentration estimator as follows:

- Environment A: keep the existing app H2 model. A newly trained dedicated
  family model reached only 66.5% held-out accuracy, below the existing model's
  88.2%, so it was rejected.
- Environment B: use a calibration-locked flame mask and a local-substrate
  colour reference, then classify `0`, `1–2`, or `2–3% H2` with the exported
  linear model. There is no validated 4% class for environment B.

## Data used

Environment B uses two physically independent runs:

| Validation run | Source | Speed used | Optical labels |
|---|---|---:|---|
| run3 | `1_90_RH20_3_x2_cropped` | x2 | initial/0, 1–2, 2–3 |
| run4 | `1_90_RH20_4` | normal | initial/0, 1–2, 2–3 |

The normal and x2 copies of the same run are never treated as independent
validation samples. The normal-speed run4 was selected because its flame colour
change is visually clearer; its confirmed boundaries were converted only when
an x2 copy was inspected.

## Held-out result

Each result below trains on one whole run and tests on the other.

| Method | Exact accuracy |
|---|---:|
| Existing app model on B-domain frames | 42.9% |
| Dynamic-ROI dedicated B model | 48.1% |
| Fixed flame mask + local substrate (deployed) | **78.1%** |

Deployed-model recalls are:

| Reference band | Recall |
|---|---:|
| 0% | 93.5% |
| 1–2% | 70.0% |
| 2–3% | 80.0% |

This is a substantial improvement but is still below the 0.85 target, so the
app labels the routed result experimental. Independent environment-B runs are
needed before treating it as a paper-grade quantitative model.

## App smoke-test points

These points verify that browser extraction matches offline extraction. They do
not replace held-out validation because the same runs contributed to training.

- `1_90_RH20_3_x2_cropped`: calibrate near 1 s; check 2 s (0), 45–50 s
  (1–2), and 57–59 s (2–3).
- `1_90_RH20_4` normal speed: calibrate near 2 s; check 5 s (0), 40 s
  (1–2), and 55 s or 84 s (2–3).

After this smoke test, the next modelling task is simultaneous-response
quantitation: preserve the four-state classifier, estimate H2 from the flame,
and estimate RH from the water-drop ROI using the H2O-only calibration basis.
