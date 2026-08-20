# H2 run-normalization and frame-quality A/B

## Purpose

Test app-computable normalization from the calibration image and suppress only
isolated frame-level optical outliers. All five H2-only recordings remain in
leave-one-video-out evaluation.

The baseline-colour projection is useful physically: the median 4% flame L
delta changes sign between runs, but its projection onto the printed flame's
initial LAB vector largely restores a common response direction. It uses only
the calibration image and current image, so it is deployable in the browser.

## Video-held-out results

| Model | Exact | Balanced | Within ±1 | MAE | Recall 0/1/2/3/4 |
|---|---:|---:|---:|---:|---|
| Reviewed selective baseline | **0.521** | **0.528** | **0.948** | **0.532** | .513/.489/.673/.534/.430 |
| Local-path outlier weighting | 0.513 | 0.521 | 0.946 | 0.546 | .513/.478/.660/.516/.437 |
| Baseline projection | 0.425 | 0.471 | 0.912 | 0.667 | .500/.543/.535/.306/.469 |
| Projection + outlier weighting | 0.448 | 0.472 | 0.918 | 0.637 | .500/.511/.478/.378/.495 |

The projection improves isolated 1% and 4% recall but damages the middle
stages. Leakage-safe pair/rule/confidence ensembles with the baseline did not
improve the baseline: the best selective rule reverted to the baseline result.
The earlier confidence hybrid remains the best overall result (balanced .549,
within-one-stage .974, MAE .508).

## Interpretation and next action

The main error is not caused by a few motion/outlier frames or a simple
illumination sign flip. Nominally equal endpoint labels correspond to different
stable optical states across runs. Do not deploy the projection experiment.

Next, make a compact 0%/4% endpoint review atlas for every H2-only run with raw
frame, registered flame mask, calibrated LAB deltas, current prediction, and
timeline reference. Human review should mark each endpoint as valid, partial
response, or invalid. Those decisions can supervise a frame-level endpoint
mixture without removing an entire illumination domain.

