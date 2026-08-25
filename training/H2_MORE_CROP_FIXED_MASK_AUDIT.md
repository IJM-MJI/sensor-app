# H2 tight-crop fixed-mask audit

## Question

Do the user-supplied `more_cropped` videos remove enough ROI contamination for
the nominal H2 stages to transfer between independent runs?

## Data and extraction

- `1_90_H2_only_test_2_more_cropped.mp4`
- `1_90_H2_only_test_3_more_cropped.mp4`
- full reaction/hold videos sampled at 2 Hz;
- calibration frame at 2 s;
- the editor's black side bars removed once per run;
- flame and inert droplet pixel masks selected on the calibration frame and
  reused without motion for every later frame;
- flame LAB changes were measured relative to calibration and also relative to
  the droplet control.

The resulting mask review is
`training/output/h2_more_crop_fixed_mask_v2/fixed_shape_masks.jpg`.  It confirms
that the final red mask covers the flame and the blue mask covers the two
droplets, without the upper cable, gray patch, white patch, or chamber rim.

## Complete-video held-out result

Each fold trained on one complete video and tested the other.  The best tested
candidate was gradient boosting.

| Metric | Result |
|---|---:|
| Exact five-stage accuracy | 0.456 |
| Within one stage | 0.562 |
| MAE | 1.547 stages |
| `0–1 / 2–3 / 4` range accuracy | 0.501 |
| Held-out `test_2` exact | 0.805 |
| Held-out `test_3` exact | 0.111 |

The large asymmetry is not caused by a bad mask.  A model trained on the weaker
`test_3` trajectory can recognize much of the stronger `test_2` trajectory,
but a model trained on `test_2` cannot map the nominal late `test_3` labels to
the same optical states.

## Optical comparison against the test_2 reference

Ignoring the `test_3` timeline and matching only its calibrated flame colour to
the five `test_2` reference stages gives:

| `test_3` time | Nominal timeline stage | Closest `test_2` optical stage |
|---:|---:|---:|
| 2 s | 0% | 0% |
| 10 s | 1% | 1% |
| 20 s | 2% | 2% |
| 28 s | 3% | 2% |
| 40 s | about 3% | 2% |
| 60 s | about 3% | 3% |
| 90 s | between 3 and 4% | 2% |
| 120 s | between 3 and 4% | 3% |
| 150 s | nearly 4% | 3% |

The calibrated trajectory plot is
`training/output/h2_more_crop_fixed_mask_v2/calibrated_flame_trajectory.png`.
`test_2` reaches approximately `Δa* = -5` to `-6`, whereas `test_3` settles
mostly near `Δa* = -2` to `-3`.  Thus the nominal `test_3` 4% interval does not
show the same optical response as the trustworthy `test_2` 4% reference.

## Decision

Do not deploy a five-class model trained from the two nominal timelines.  The
tight crops successfully fix the ROI, but they also demonstrate a run-response
mismatch that image preprocessing cannot erase.

The defensible next training experiment is:

1. keep `test_2` as the strong 0/1/2/3/4 reference;
2. use `test_3` only as weak optical augmentation for 0%, 1%, 2%, and the later
   3%-equivalent frames; exclude its nominal 4% frames from exact supervision;
3. implement a calibration-locked flame pixel mask in the browser-compatible
   extractor, then evaluate the new training policy on a third complete H2 run;
4. deploy only if the third-run held-out result improves without reducing the
   verified 3% recall.

Reproduce with:

```powershell
.\.venv\Scripts\python.exe training\h2_more_crop_fixed_mask_analysis.py `
  --video-root 'C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1' `
  --output training\output\h2_more_crop_fixed_mask_v2 --sample-hz 2
```
