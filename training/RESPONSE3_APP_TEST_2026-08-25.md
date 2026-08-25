# Response-3 app test audit (2026-08-25)

The seven supplied screenshots were interpreted in order as 2, 3, 5, 7, 11,
25, and 28 seconds. A 38-second screenshot was not supplied.

| Time | Expected RH range | Four-state output | RH output | Quantitation usable? |
|---:|---|---|---|---|
| 2 s | 20--30 | No Sensor | none | no |
| 3 s | 20--30 | simultaneous | hidden/pending | no |
| 5 s | 40--50 | simultaneous | hidden/pending | no |
| 7 s | 40--50 | H2-only | legacy forced 20--30 | no |
| 11 s | 60--70 | H2-only | legacy forced 20--30 | no |
| 25 s | 60--70 | uncertain | uncertain | no |
| 28 s | 80--90 | H2O-only | 80--90, `rhD=7.98` | diagnostic only |

Only the 28-second frame reached the experimental RH model through the state
gate, so these screenshots do not measure the RH range model's accuracy. The
dominant failures are upstream four-state classification and unstable circle
selection. The detected circle visibly changes in centre/radius between
frames; the 7-second frame in particular uses a much smaller circle than the
early frames. The high `rhD` at 28 seconds also indicates a substantial
app/monitor-to-training feature-domain difference even though its nearest
range happens to be correct.

The first two screenshots do not visibly show the calibrated status bar,
whereas screenshots from 5 seconds onward do. The repeat test must confirm that
calibration was performed at 0.5 seconds before loading the 2-second frame and
remained active throughout the run.

App diagnostic v2 therefore:

- anchors Hough-circle selection to the normalized calibration circle;
- evaluates the endpoint RH model in shadow mode regardless of the selected
  four-state class or uncertainty;
- keeps the production state gate unchanged;
- shows shadow range, nearest-prototype distance, and LAB vector as
  `rhShadow`, `d`, and `v`.

The next repeat needs 0.5-second calibration followed by all eight endpoints:
2, 3, 5, 7, 11, 25, 28, and 38 seconds. The shadow output is the RH-model result;
the main state label remains the independent four-state result.

## Diagnostic-v2 repeat and root cause

The repeat supplied all eight endpoints after calibration. It proved that the
calibration anchor itself was wrong: every selected circle after the first
failure was a reflection on the lower-left metal rim, not the chamber window.

| Time | selected circle `(x,y,r)` | shadow RH | valid? |
|---:|---|---|---|
| 2 s | no circle | none | no |
| 3 s | `(63,349,62)` | 80--90 | no |
| 5 s | `(87,357,103)` | 20--30 | no |
| 7 s | `(64,303,71)` | 80--90 | no |
| 11 s | `(105,331,88)` | 20--30 | no |
| 25 s | `(73,373,52)` | 60--70 | no |
| 28 s | `(73,325,57)` | 40--50 | no |
| 38 s | `(58,343,63)` | 20--30 | no |

No RH accuracy conclusion can be drawn from these outputs because the RH
feature pixels came entirely from metal. The changing state labels are also
invalid for the same reason.

Response3 ROI fix v3 expands the maximum Hough radius for tightly cropped
frames and rejects candidates outside a central-chamber geometry envelope
before either scoring or calibration anchoring. The supplied original video
was audited at 0.5, 2, 3, 5, 7, 11, 25, 28, and 38 seconds. All nine selected
circles stayed on the chamber window: `x=125--136`, `y=188--205`, and
`r=71--75` after 480-pixel resizing. The reproducible audit is
`training/audit_response3_circle_roi.py`; its CSV and review sheet are written
under `training/output/response3_roi_audit/`.

For the next app repeat, first inspect the calibration overlay. Its red/yellow
circle must enclose the central chamber window. If it is on metal, stop and
send only that calibration screenshot; do not continue the endpoint sequence.

## ROI-fix-v3 repeat

With the lower-left reflection rejected, all eight requested endpoint images
used a central chamber ROI. In endpoint order (2, 3, 5, 7, 11, 25, 28, 38 s),
the shadow ranges were 40--50, 20--30, 60--70, 60--70, 60--70, 80--90,
80--90, and 80--90. Against the supplied range targets this is 4/8 exact;
the correct endpoints were 3, 11, 28, and 38 s.

The remaining test exposed a geometry-domain mismatch rather than a justified
new concentration boundary. The training cache uses one locked circle for all
endpoints in a video. The app instead re-detected the circle on every image;
the screenshots show selected `y=172--231` and `r=228--265`. That changes the
normalized tight-droplet pixels between otherwise aligned images. Version v4
therefore reuses the normalized calibration circle for the entire aligned run
and adds an inner-aperture radius prior during the initial calibration. RH
prototypes are deliberately unchanged until the fixed-geometry repeat shows
which errors remain.

## Fixed-geometry app-domain follow-up

The fixed ROI improved response3 from 4/8 to 6/8. Additional frames showed:

| Time | observed relative LAB vector | shadow range |
|---:|---|---|
| 1.5 s | `(-4.4, 1, 0)` | 40--50 (wrong) |
| 2.5 s | `(-2.8, 1, 0)` | 40--50 (wrong) |
| 6.0 s | `(-5.7, 3, 3)` | 40--50 (correct) |
| 6.5 s | `(-4.8, 3, 2)` | 40--50 (correct) |

The early errors contain lightness change but essentially no chromatic change.
Reducing the global L* weight was rejected because a 0.5 weight changed the
otherwise-correct response6 18 s result from 60--70 to 80--90. Version v5
instead applies a narrow brightness-only guard: when both absolute relative
`a*` and `b*` are at most 1.25, the range is 20--30. None of the observed
middle/high app vectors or middle/high stored prototypes meet this condition.
It corrects response3 1.5, 2, and 2.5 s while preserving the tested response3
and response6 middle/high predictions.

The concentration guard alone would leave the independent four-state forest's
low-confidence label as `Uncertain / Retake`. Version v6 also recovers
`Initial / Low Response` when the guard is active and combined H2 probability
is below 0.25. This does not post-process confident four-state predictions and
cannot suppress an H2/simultaneous candidate above that threshold.
