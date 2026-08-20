# H2 verified partial-response correction

## Human decisions

- `1_90_H2_only_test.mp4` at 2 s is a valid H2 0% calibration anchor.
- The late response in `1_90_H2_only_4.mp4` did not reach 4%; treat it as
  interval-censored 2--3% response.
- The late response in `1_90_H2_only_5.mp4` did not reach 4%; treat it as
  interval-censored 2--3% response, optically closer to 3% than run 4.

Run 4 rows after the confirmed 2% endpoint at 30 s and run 5 rows after the
confirmed 2% endpoint at 13 s are removed from exact validation. They retain a
low-weight capped 2--3% pseudo-ramp for domain coverage. Weak rows from the
held-out video are excluded from its training fold.

## A/B results on 574 verified exact frames

| Policy | Exact | Balanced | Within ±1 | MAE | Recall 0/1/2/3/4 |
|---|---:|---:|---:|---:|---|
| Corrected, partial weight .10 | .470 | .478 | .909 | .622 | .592/.457/.500/.718/.123 |
| Corrected, partial weight .002 | **.528** | **.525** | **.929** | **.565** | .658/.424/.392/.316/.832 |
| Corrected, near-zero partial | .517 | .514 | .929 | .575 | — |

The old model scores .559 exact on the same retained frames, but that is not an
admissible corrected-label result because its training folds still used the now
rejected run 4/5 4% labels. Leakage-safe ensembles of corrected-label models did
not exceed the .002 model.

## Cropped run 5 decision

The new cropped video is useful for visual/mask review, but a full cropped-v4
feature cache is not compatible with the present RH20 auxiliary feature domain:
the cropped baseline gave exact .325 and 4% recall 0. Do not deploy it as a
drop-in training replacement. A future cropped-only model requires all auxiliary
runs to be re-extracted under the same geometry.

## Decision

Keep the current app model unchanged. Preserve the verified partial-response
profile as the scientifically correct reference policy. The next H2 task is to
recover 2%/3% separation using only verified runs, rather than restoring the
incorrect 4% labels.
