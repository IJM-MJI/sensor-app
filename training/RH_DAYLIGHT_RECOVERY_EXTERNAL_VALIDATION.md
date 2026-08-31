# Daylight RH recovery external validation

## Protocol

`1_90_H2O_only.MOV` is an RH90→20 descending recovery run recorded under a
different daylight condition. The frozen Place-2 seven-band endpoint model was
evaluated without changing its prototypes or thresholds.

- calibration: 38.0 s, RH20 recovery tail
- evaluation: 0.5 s before each RH90/80/70/60/50/40/30 segment boundary
- comparison: single frame versus the same endpoint's immediately preceding
  three-frame median
- confusion matrix: reference-row normalized 0–1

The calibration frame's raw `top_a` is approximately 118.8, so v39's
response3/response6 stable-profile router (minimum 127.5) does not activate.
This is therefore an external environment test of the generic frozen endpoint
model, not a validation of either response-specific profile.

## Result

| Method | n | Exact | Balanced | Within one | MAE |
|---|---:|---:|---:|---:|---:|
| Single | 7 | 0.286 | 0.286 | 0.857 | 8.57 %RH |
| Latest trailing 3 | 7 | 0.286 | 0.286 | 0.857 | 8.57 %RH |

Both methods produced the same seven predictions. Correct ranges were 20–30
and 60–70. The other five points were predominantly one adjacent range lower.
The three-frame median therefore did not address the failure.

## Decision

The 0.85 exact/balanced/every-range target fails. No app threshold is changed.
The pattern is consistent with direction/environment shift: the model was built
from response runs, while this video is a descending recovery under daylight.
It should not be repaired by selecting label-specific timestamps after seeing
the predictions.

The next useful external test is a newly recorded **rising** RH20→90 run with
the same seven stage holds and a separate RH20 calibration. That separates
environment transfer from recovery hysteresis.
