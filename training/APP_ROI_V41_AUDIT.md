# App ROI v41 cropped/uncropped audit

## Failure reproduced

The v40 selector treated every frame with aspect ratio below `1.35` as a tight
crop. The new uncropped saved frames have ratios `1.25--1.32`, so a partial arc
formed by the plate and lower plumbing was converted into a forced aperture of
about `r=147` at the app's 480 px processing scale. This included the plumbing
and produced false simultaneous classifications.

## v41 rule

- Accept a compact optical aperture only when `r / short-side <= 0.22`, its
  centre is above `0.55 * height`, and warm pixels occur in both the flame and
  droplet halves.
- Use the tight-crop outer-ring conversion only when no such compact aperture
  exists.
- A low-confidence/tied simultaneous vote now requires confirmation from both
  the independent H2 expert and the H2 concentration projection. Without that
  confirmation, dominant RH evidence is retained or the frame is rejected.

## Browser regression

The production OpenCV.js path was exercised in a local browser using the exact
saved files supplied in `Downloads/app_test`.

| Set | Calibration ROI at 480 px | Locked measurements | Simultaneous outputs |
|---|---|---:|---:|
| uncropped | `(229,110,r=63)`, compact aperture | 6/6 | 0/6 |
| cropped | `(240,213,r=164)`, tight-crop aperture | 8/8 | 0/8 |

The uncropped measurement radii remained `62--66` after normalized calibration
locking. The cropped measurement radii remained `160--176`. The cropped 27 s
frame that previously produced a weak simultaneous result (`pState=0.38`,
`gap=0.02`) was blocked and reported as low response instead.

This audit establishes ROI geometry and state-gate behaviour only. It does not
claim that seven-band RH concentration accuracy has reached the 0.85 target.
