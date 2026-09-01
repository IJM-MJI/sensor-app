# v41 single-frame colour-path feature audit

## Scope

This audit keeps the seven user-reviewed optical stages and uses one frame per
measurement. Commanded RH timing and adjacent-frame medians are excluded.

The three cropped recordings were decoded in their visible orientation and
centre-cropped to reproduce the near-square saved PNG input. The mirrored v41
selector then locked nearly identical chamber geometry:

| Profile | v41 ROI at 480 px |
|---|---|
| response3 | `(234, 247, r=184)` |
| response6 | `(234, 247, r=185)` |
| daylight | `(241, 246, r=185)` |

This removes ROI scale and position as the explanation for the remaining
colour-path mismatch.

## Features compared

- tight droplet minus nearby substrate LAB;
- flame LAB and a/b;
- droplet a/b;
- droplet minus flame a/b;
- registered droplet minus flame a/b;
- normalized hue direction `(a-128)/chroma`, `(b-128)/chroma`;
- combinations of flame, droplet and tight-mask features.

All features are relative to each recording's reviewed 20–30 calibration
frame. Both the current 40–50 anchors and the proposed response3 7 s /
response6 13 s anchors were tested.

## Result

No shared feature family transfers the seven stages between response3 and
response6. With the corrected v41 geometry, the best candidates still produce
only about one correct stage out of seven in both complete-run directions.

On the independently reviewed daylight frames, the best proposed-timing
candidate reached only `0.375` exact accuracy. The old tight-droplet feature
reached `0.25` or less. The timing shift changes some local distances but does
not align the three recording environments.

The failure persists after ROI, neutral-background correction, calibration
subtraction, within-frame flame references, and hue-direction normalization.
It is therefore not defensible to remove the current profile distinction and
choose one global nearest colour prototype.

## Interpretation

The visible sensor stages can follow the same qualitative colour order while
their numerical LAB trajectory is translated, scaled, and locally warped by
lighting, exposure, recording, and sensor orientation. One dry calibration
point removes translation but cannot infer every scale/warp parameter.

The absolute `top_a` threshold is still too brittle: the same profile can move
outside its threshold after crop/environment changes. The replacement should
retain environment-specific optical paths but select them from a multichannel
calibration signature, then reject calibrations that are far from every known
profile.

## Decision

1. Do not deploy a shared response3/response6/daylight 1-NN.
2. Do not deploy the 7 s / 13 s timing change by itself.
3. Keep single-frame inference and seven 10%-wide stages.
4. Train one single-frame colour-path profile per verified environment.
5. Replace the one-dimensional `top_a` route with a calibrated multichannel
   profile router and an explicit out-of-profile retake condition.

Reproduce with:

```powershell
.\.venv\Scripts\python.exe -X utf8 training\rh_v41_color_path_feature_ab.py
```
