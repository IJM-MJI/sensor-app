# Endpoint mask review guide

## How to read the sheets

Each example is shown twice:

- `RAW` (left): the decoded video frame.
- `MASK` (right): the exact pixels sent to the colour feature extractor.
- Red/orange pixels: H2 flame measurement.
- Blue/cyan pixels: RH droplet measurement.
- Yellow rectangle: allowed flame search zone.
- Cyan rectangle: allowed droplet search zone.
- Green circle: detected chamber coordinate system.
- `timeline`: supplied endpoint concentration.
- `other-run nearest`: concentration whose median colour in the other runs is
  closest. This is a review hint, not a corrected label.

A good mask colours the printed flame or droplet while leaving most of the gray
card untouched. If red/blue fills card background, tubing, shadows or a reference
piece, the disagreement is an extraction error and must not be fixed by changing
the timeline.

## Current finding

The H2 red mask generally follows the flame, including the newly supplied
`1_90_H2_only_5_cropped.mp4`. The RH blue mask frequently includes substantial
card background because the current extractor always keeps roughly the most
colour-distant 35% of each search zone. This is now the primary RH feature bug.

The RH20 simultaneous review confirms that these runs are useful H2-only weak
supervision. Runs 2/4/5 mostly cover the flame; run 3 also captures a diagonal
dark obstruction in the flame zone, so it should be down-weighted or revisited
after the shape-mask refinement.

Local review files:

- `training/output/endpoint_mask_review/h2_endpoint_mask_review.jpg`
- `training/output/endpoint_mask_review/rh_endpoint_mask_review.jpg`
- `training/output/endpoint_mask_review/rh20_h2_weak_mask_review.jpg`

## Quantitative A/B

Adding `H2_only_5_cropped` improved the centred-crop H2 evaluation from 40.4% to
43.3% exact, from 92.0% to 95.5% within one stage, and MAE from 0.68 to 0.61%p.

Adding the four RH20 reaction runs as training-only weak supervision (endpoints
weighted strongly, ramp interior at 0.10) produced 46.2% exact, 44.6%
stage-balanced, 97.5% within one stage, and 0.56%p MAE on the unchanged five
H2-only held-out runs. The original deployed-data result was 46.1%, 48.8%,
87.5%, and 0.67%p respectively. Gross errors improve, but exact/stage-balanced
accuracy do not improve together, so the browser model remains unchanged.

The next experiment must tighten/segment the printed shapes and then repeat this
same whole-run held-out A/B before any model deployment.
