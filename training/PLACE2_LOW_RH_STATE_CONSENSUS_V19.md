# Place-2 low-RH state consensus v19

## App-domain failure

Two adjacent frames from `1_90_H2O_only_3(response).mp4`, calibrated at 0.5 s,
were falsely classified as simultaneous:

| Time | Optical RH shadow | pH2 | pRH | H2 expert | raw H2 | Previous state |
|---:|---:|---:|---:|---:|---:|---|
| 2.5 s | 20–30% | 0.50 | 0.51 | 0.21 | 1.19 | simultaneous |
| 3.0 s | 40–50% transition | 0.49 | 0.51 | 0.25 | 1.22 | simultaneous |

The four-state probabilities are nearly tied, while the independent flame
expert contains little H2 evidence. The 3 s frame is an RH ramp boundary and is
not used as an exact concentration anchor, but it is still H2O-only rather than
simultaneous.

## Guard

The state is changed from simultaneous to H2O-only only when all of the
following hold:

- calibration `top_a >= 127` (Place-2 H2O-only app domain);
- RH shadow is 20–30 or 40–50%;
- combined RH probability is at least the combined H2 probability;
- independent H2 expert probability is below 0.35;
- legacy raw H2 is in `[0.5, 1.5)`.

All validated H2-only app captures have calibration `top_a <= 125.5`, so this
rule cannot alter their state or concentration. Higher RH response3/response6
frames already classify as H2O-only and do not enter the guard.

## Retest

- 2.5 s should report H2O-only, H2 0%, RH 20–30%.
- 3.0 s should report H2O-only and H2 0%; its RH output is a transition
  diagnostic and may be 40–50%.
- Response6 13, 17, and 19 s should remain 40–50, 60–70, and 80–90%.
