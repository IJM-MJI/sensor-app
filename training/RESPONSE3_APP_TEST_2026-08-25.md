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
