# Place-2 profile-specific stable-window RH model

## Objective

Train separate RH models for `response3` and `response6` using the stable
optical centres found in the interval-selection audit. The current app takes
one photo immediately, so deployment is decided from single-frame validation,
not from a one-second validation average.

Validation frames were required to be at least 0.75 s from the selected
training centre. Eight previously user-reviewed points remained:

- response3: 1.0, 9.0, 26.5 and 30.0 s
- response6: 14.5, 15.5, 19.0 and 23.0 s

This is same-run validation and is not equivalent to a new independent run.

## Results

| Candidate | Input | Exact | Within one range | MAE (%RH) |
|---|---|---:|---:|---:|
| profile 1-NN | single frame | **0.750** | 0.875 | 6.25 |
| profile polyline | single frame | **0.750** | 0.875 | 6.25 |
| profile ridge | single frame | 0.250 | 0.750 | 10.00 |
| profile 1-NN | ±0.5 s median | 1.000 | 1.000 | 0.00 |
| profile polyline | ±0.5 s median | 1.000 | 1.000 | 0.00 |
| profile ridge | ±0.5 s median | 0.250 | 0.875 | 8.75 |

The one-second median demonstrates that the stable prototypes themselves are
consistent. It cannot be used as the deployment score because the app was
explicitly changed to immediate single-photo capture.

## Single-frame errors

| Run | Time | Reference | Prediction |
|---|---:|---|---|
| response3 | 9.0 s | 40–50 | 50–60 |
| response3 | 26.5 s | 60–70 | 20–30 |

All four tested response6 points were correct. The response3 26.5 s failure is
not a middle-boundary error: one frame moves from a high-RH state back to the
low endpoint in calibrated LAB space. This is consistent with the previously
observed 26.5→27 s optical jump and indicates a frame/ROI/exposure outlier.

## Decision

Do not deploy this candidate. Its single-frame exact accuracy is 0.750, but
the frozen outer-band accuracy is only 0.800, below the required 0.85. The
current app is left unchanged.

The next candidate should preserve immediate capture while using a very short
three-frame burst (approximately 0.1–0.2 s total) or an equivalent frame
outlier check. That is materially different from the rejected 3–5 second
accumulation and directly targets the response3 single-frame jump.

