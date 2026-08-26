# Automatic H2 optical 2-vs-3 specialist audit

## Purpose

Test whether the trusted optical candidates from H2-only, RH20, and the
angle-80 RH20 run can safely correct 2%/3% predictions from the four-band H2
model. No user review or manual runtime decision is part of this pipeline.

## Protocol

- Four-band baseline: shrinkage LDA, complete-video held out.
- Specialist training: 441 confident optical candidates (341 optical 2%, 100
  optical 3%).
- Candidate sources: all available H2-only runs, five RH20 runs, and the
  flame-up-aligned angle-80 RH20 run.
- RH30 remains excluded because its late stable screening produced 13 optical
  2% candidates and no optical 3% candidates.
- `test_2` defines the optical anchors, so specialist overrides are not scored
  on that run.
- For each other held-out video, model family and asymmetric thresholds are
  selected using only the remaining evaluation videos.
- Logistic, shrinkage LDA, mean consensus, and strict two-model consensus were
  tested.

An override is acceptable only if it improves 2% recall while preserving 0–1%,
3%, and 4% recall, exact accuracy, video-macro accuracy, and MAE. It must also
avoid reducing exact accuracy or 3% recall on any individual meta-training
video.

## Results

| Metric | Four-band baseline | Nested specialist |
|---|---:|---:|
| Exact accuracy | 0.6753 | 0.6753 |
| Video-macro accuracy | 0.6837 | 0.6837 |
| MAE, percentage points | **0.4457** | 0.4491 |
| Recall 0–1% | 0.8395 | 0.8395 |
| Recall 2% | 0.6735 | 0.6735 |
| Recall 3% | 0.3789 | 0.3789 |
| Recall 4% | 0.7042 | 0.7042 |

Only one held-out fold selected a rule. It changed three baseline 3%
predictions to 2%; all three belonged to true 4% rows. Accuracy was unchanged,
but their numerical error increased, causing the MAE regression.

A diagnostic grid using all evaluation labels found **no** threshold/model
combination that satisfied every safety constraint. This confirms that the
failure is not merely an unlucky nested threshold choice.

## Decision

Do not deploy the specialist. RH20 and angle-80 data provide useful optical 2%
variation but do not supply enough independent 3% variation. A 2/3 correction
trained on this pool cannot improve 2% without either moving true 2% frames in
the wrong direction or increasing error on 4% frames.

## Next automatic step

Replace frame-only 2/3 correction with a trajectory model that uses the
calibration-relative direction and magnitude of flame colour change across
consecutive frames. The evaluation remains fully automatic and complete-video
held out. The application model should change only if that model improves 2%
and 3% while preserving the endpoint bands and overall error.
