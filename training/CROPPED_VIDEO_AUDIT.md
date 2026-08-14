# Cropped recording audit

Date: 2026-08-14

## Sources

The `_cropped` recordings in `recordings/1` were evaluated with the original
timelines and logical run IDs. The crops are visually usable: they retain the
whole chamber/sample, and the response-3/response-6 rotations put the flame
above the droplet as required by the semantic masks.

The raw recordings stay external to Git. The local QA montage is generated at
`training/output/cropped_video_qc_montage.jpg`.

## Concentration A/B result

All values below are calibration-aware held-out-video results. No evaluated
cropped model was deployed to the browser application.

| Dataset | H2 exact | H2 within ±1 stage | RH exact | RH within one stage |
|---|---:|---:|---:|---:|
| Original | 46.1% | 87.5% | 36.9% | 66.1% |
| Cropped replaces original | 40.0% | 85.8% | 27.8% | 52.8% |
| Original + cropped augmentation | 33.2% | 78.4% | 28.3% | 51.6% |

Within-run block scores increased for the replacement experiment (H2 exact
46.8% to 52.8%; RH exact 52.1% to 56.2%), while unseen-video scores decreased.
This is evidence that the current extractor is learning run/crop geometry in
addition to sensor colour. It is not evidence that the crops are poor.

## Geometry finding

The Hough detector alternates between the chamber aperture and smaller internal
circles in some cropped frames. A second experiment that always selected the
largest circle also reduced held-out accuracy, because the actual chamber centre
drifts in handheld recordings and the largest edge is not always the same
physical boundary.

The cropped files should therefore be retained as geometry-normalisation and
robustness data, but not simply mixed into concentration fitting. The next model
revision must first map every frame into one canonical sample coordinate system
using stable sample/chamber landmarks. Concentration features can then be
re-extracted from the flame mask for H2 and the droplet mask for RH and evaluated
again with whole-run held-out validation.
