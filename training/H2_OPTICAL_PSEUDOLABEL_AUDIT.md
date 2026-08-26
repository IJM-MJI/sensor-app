# H2 2–3% optical-state audit

## Objective

Separate H2 2% and 3% using the observed flame colour state rather than assuming
that every nominal timeline endpoint reached the requested concentration.  The
fixed reference is `1_90_H2_only_test_2_cropped`: 20–22 s for optical 2% and
29–31 s for optical 3%.

## Data used

- H2-only: `test_2`, `test_3`, `test`, run 4, and run 5 cropped videos.
- Simultaneous RH20: runs 2, 3, 4, 5 x2, and 5 normal cropped videos.
- Angle augmentation: `1_80_2.MOV`, rotated to flame-up and compared only in
  its RH20 reaction segment.
- RH30 was screened separately as an optional source. It was not admitted to
  the training set.

Timelines bound the reaction periods only. Within those periods, frames are
assigned to the nearest fixed test-2 optical anchor only when both distance and
2-vs-3 margin checks pass. The feature vector uses the fixed flame mask and 11
baseline-subtracted Lab/chroma distribution measurements.

## Selected optical states

| Source | Optical 2% | Optical 3% | Interpretation |
|---|---:|---:|---|
| test_2 | 16 | 3 | anchor run |
| test_3 | 43 | 76 | main independent 3% source |
| test | 42 | 3 | mostly test-2-like 2% |
| H2-only run 4 | 73 | 1 | mostly reaches optical 2% |
| H2-only run 5 | 123 | 0 | mostly reaches optical 2% |
| RH20 run 2 | 9 | 0 | weak 2% augmentation |
| RH20 run 3 | 3 | 2 | small mixed contribution |
| RH20 run 4 | 10 | 3 | small mixed contribution |
| RH20 run 5 x2 | 1 | 10 | useful 3% contribution |
| RH20 run 5 normal | 5 | 2 | small mixed contribution |
| angle-80 run 2 | 16 | 0 | useful viewpoint augmentation for 2% |

This supports the visual assessment that H2-only runs 4 and 5 did not reach the
same optical state as test-2 3–4%; they are predominantly test-2-like 2%.

## Validation

The 99.76% pseudo-label reference match is only a self-consistency check because
the same optical rule creates those labels. It is **not** an accuracy estimate.

The non-circular comparison is against untouched, broad 2/3 timeline intervals:

| Training pool | Exact match | Video-macro match | Recall 2% | Recall 3% |
|---|---:|---:|---:|---:|
| 90-degree only | 45.66% | 44.62% | 36.73% | 56.52% |
| + angle-80 RH20 | 49.02% | 48.85% | 43.37% | 55.90% |

Angle-80 improves overall and 2% matching while nearly preserving 3% recall, so
it is acceptable as a 2% viewpoint augmentation. The remaining low agreement
also confirms that nominal timeline labels and optical states frequently differ;
it does not establish which source is correct without an independent reference.

## RH30 decision

Late reaction frames from four RH30 runs were screened with the same anchors.
Thirteen frames passed: all thirteen were optical 2%, and none were optical 3%.
RH30 therefore does not address the missing 3% diversity and introduces an RH
simultaneous-response confound. It is excluded from H2 quantification training.

## Deployment decision

Do not replace the current application model yet. The useful additions are
unbalanced: RH20 and angle-80 strengthen 2%, but independent 3% evidence still
comes mainly from test-3 and RH20 run-5 x2. Deploying now risks improving 2%
while reducing 3% recall.

## Next step

Build a constrained 2-vs-3 specialist using only the trusted optical candidates,
then test it video-held-out with a rule that may override the existing four-band
model only when 3% recall is not reduced. This is an entirely automatic test;
no user review or manual runtime decision is required.
