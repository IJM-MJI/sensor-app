# H2 yellow-to-green optical-coordinate audit

## User-photo colour check

The flame was measured inside a manually bounded flame region, retaining
moderately saturated sensing pixels.

| Photo | Median RGB | HSV hue | OpenCV Lab | Visual interpretation |
|---|---|---:|---|---|
| 1 | 146, 144, 103 | 58° | 150, 123, 149 | user-confirmed Initial |
| 2 | 143, 143, 98 | 58° | 149, 121, 150 | user-confirmed H2 2-3% |

OpenCV Lab a*=123 and 121 correspond to centred a* values near -5 and -7.
The second flame is therefore measurably farther in the green direction, but
the shift is small. The earlier visual estimate of 1-2% for photograph 2 was
incorrect. The user-confirmed assignment shows that a small green shift can
already mean H2 2-3%; a dramatic visible change must not be required.

In the current `test_2` trajectory, the approximate photo-pair change in a*
lies between the learned optical-2 and optical-3 anchors. This supports using
the corrected pair as an external coarse-state constraint, while still not
claiming an exact 2% versus 3% assignment from the photographs alone.

## Model tested

The run-balanced trajectory labels were transformed into a shared optical
coordinate containing:

- projection along the learned optical-2 to optical-3 yellow-to-green vector;
- orthogonal colour residual;
- calibration-relative response magnitude and direction cosine;
- distances to equal-run optical-2 and optical-3 centroids;
- separate mean, median, and chroma-distribution summaries.

The direction and centroids were refitted without each held-out video. Runtime
would still require only one calibration image and one measurement image.

## Findings

The learned direction was stable across held-out folds. Its largest common
component was decreasing Lab a*, confirming the expected yellow-to-green H2
response. Optical-2/3 centroid separation was also substantial in the training
coordinates.

Nevertheless, all six combinations of trajectory certainty (0.75/0.85) and
per-run cap (6/12/20) rejected every proposed override. Final metrics therefore
remain identical to the baseline:

| Metric | Result |
|---|---:|
| Exact accuracy | 0.6753 |
| Video-macro accuracy | 0.6837 |
| Recall 2% | 0.6735 |
| Recall 3% | 0.3789 |
| MAE | 0.4457 |

## Decision

Do not deploy the coordinate specialist. The experiment confirms that the
chemical colour direction is detectable and agrees with the supplied photos,
but the current independent runs do not provide a stable shared boundary
between optical 2% and 3%.

## Next step

The next automatic analysis should stop forcing the weak and strong run
families onto one exact 2/3 boundary. Fit a hierarchical model that first
estimates each sample's response-strength family from its calibration and
low-response geometry, then uses a family-specific optical coordinate. Validate
family selection and concentration together with the same complete-video
holdout. The application remains single-shot and requires no user decision.
