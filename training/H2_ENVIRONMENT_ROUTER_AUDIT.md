# H2 environment-router audit

## Goal

Select the A, B, or C H2 concentration model from the calibration image before
quantifying concentration. Reaction frames and future endpoints are not used.

## Results

| Feature domain | Accuracy | A recall | B recall | Decision |
|---|---:|---:|---:|---|
| Offline fixed flame mask | 93.8% | 96.9% | 87.5% | Valid research result |
| Current app baseline means | 83.3% | 75.0% | 100.0% | Do not deploy |
| Current app-mask distributions | 64.6% | 50.0% | 93.8% | Reject |

The independent offline A/B result exceeds the 0.85 target. The technical
A/B/C result is 98.4%, but C has only one physical recording and its
normal/x2 copies cannot establish independent C accuracy.

The current app-domain router misses the weak `run2` A environment. Adding
distribution statistics to the app's broad flame ROI makes transfer worse,
showing that the fixed offline flame mask and the deployed ROI are not yet
equivalent. Deploying this router now could send a valid image to the wrong
concentration model, so no application behaviour was changed.

## Next implementation

Port the calibration-locked fixed flame mask used by the 93.8% router into the
browser extraction path. Re-extract app-domain calibration features with that
same mask and repeat complete-video-held-out A/B validation. Add C only as an
experimental route until a second independent family-C recording exists.
