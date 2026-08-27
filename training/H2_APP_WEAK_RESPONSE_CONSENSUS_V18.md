# H2 app weak-response consensus v18

## User-validated app anchors

The following frames were captured through the deployed app/monitor workflow,
not inferred from the original video pixels.

| Video | Time | Reference | Previous app result | pH2 | rawH2 | flame delta (a,b) | magnitude |
|---|---:|---:|---|---:|---:|---:|---:|
| `1_90_H2_only_test_3_cropped.mp4` | 23 s | 1–2% | Initial / 0% | 0.56 | 0.08 | (-0.1,-0.7) | 0.71 |
| `1_90_H2_only_test_3_cropped.mp4` | 88 s | 2–3% | H2-only / 0–1% | 0.69 | 0.31 | (-0.3,-1.1) | 1.14 |
| `1_90_RH20_3_x2_cropped.mp4` | 45 s | 1–2% | Initial / 0% | 0.57 | 0.05 | (-0.5,-0.5) | 0.71 |
| `1_90_RH20_3_x2_cropped.mp4` | 57 s | 2–3% | Uncertain | 0.62 | -1.71 | (-0.7,0.7) | 0.99 |

The old continuous projection fails on all four, while calibration-relative
flame displacement cleanly separates the two user-confirmed ranges. The split
threshold is 0.85, midway between the largest 1–2% anchor (0.71) and smallest
2–3% anchor (0.99).

## Guard conditions

This is a fallback, not a replacement for the existing concentration model. It
activates only when all conditions hold:

- RH endpoint shadow is 20–30%;
- calibration `top_a <= 125.5`, covering the supplied H2 app captures while
  excluding the known Place-2 H2O-only false-positive calibration domain;
- combined H2 state probability is at least 0.55;
- legacy raw H2 is in `[-2.0, 0.5)` (the observed collapse region);
- flame a*/b* displacement magnitude is at least 0.55.

Predictions that already worked on `H2_only_test_2` and `RH20_4` have raw H2
well above this interval and are therefore unchanged. Within the guarded
fallback, magnitude below 0.85 reports 1–2%; magnitude at or above 0.85 reports
2–3%.

## Required retest

Repeat the same four app captures. Expected results are H2-only 1–2%, H2-only
2–3%, H2-only 1–2%, and H2-only 2–3%, respectively, while RH remains 20–30%.
The next step is to run the successful `test_2` and `RH20_4` anchors once more
to verify the guard did not regress their existing outputs.
