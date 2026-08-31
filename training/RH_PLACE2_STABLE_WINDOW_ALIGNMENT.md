# Place-2 RH stable-window interval validation

## Purpose

This test checks whether the weak seven-band result was mainly caused by
choosing the wrong instant inside each response interval. Both recordings are
Place 2:

- `1_90_H2O_only_3(response).mp4` (`response3`)
- `1_90_H2O_only_6(response).mp4` (`response6`)

One complete recording is always excluded from training. The model is trained
on the other recording and then evaluated on the excluded recording.

## Compared extraction methods

| Method | Definition |
|---|---|
| single anchor | One frame at the original guide time |
| anchor window | Median LAB colour over guide time ±0.5 s |
| stable window | Lowest-variance 1 s window inside the predeclared ordered interval |

Stable-window selection uses only local LAB variation and distance from the
guide time. It does not choose a frame because it produces the desired class.

## Results

| Method | Exact accuracy | Balanced accuracy | Within one range | MAE (%RH) |
|---|---:|---:|---:|---:|
| single anchor | 0.500 | 0.500 | 0.857 | 6.43 |
| anchor window | 0.500 | 0.500 | 0.929 | 5.71 |
| stable window | **0.786** | **0.786** | **1.000** | **2.14** |

Stable-window complete-run folds:

- train `response6`, hold out `response3`: **0.857**
- train `response3`, hold out `response6`: **0.714**

This confirms that interval/frame selection was a material error source. A
median at the old time was insufficient; moving to a stable optical endpoint
was responsible for the improvement.

## Stable optical times selected automatically

| RH range | response3 | response6 |
|---|---:|---:|
| 20–30 | 2.25 s | 9.00 s |
| 30–40 | 4.25 s | 11.50 s |
| 40–50 | 10.00 s | 13.00 s |
| 50–60 | 18.00 s | 16.50 s |
| 60–70 | 27.25 s | 17.75 s |
| 70–80 | 31.50 s | 19.75 s |
| 80–90 | 37.50 s | 25.00 s |

The selected values are window centres. Each feature is the median from ±0.5
s, not a single frame.

## Remaining errors

The stable-window confusion matrix shows three remaining errors out of 14:

- held-out `response3` 40–50 is predicted as 50–60
- held-out `response6` 40–50 is predicted as 30–40
- held-out `response6` 50–60 is predicted as 60–70

All errors are adjacent ranges. The unresolved boundary is therefore
concentrated around 40–60%, rather than being a general failure across the
whole RH ramp.

## Decision

Do not replace the current app model yet. Exact accuracy 0.786 and the
`response6` fold accuracy 0.714 do not meet the predeclared 0.85 deployment
rule, and 40–50 recall is still zero in the combined two-run matrix.

The next useful step is to fit the current profile-specific model with these
stable-window centres, then test it on frames not used for window selection.
Only the middle 40–60 boundaries need adjustment; the low and high ranges
should remain frozen to avoid regressing the bands that already transfer.

