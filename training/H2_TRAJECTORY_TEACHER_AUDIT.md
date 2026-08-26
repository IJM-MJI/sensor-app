# H2 trajectory-teacher audit

## Runtime design

The video trajectory is used only to clean training labels. The application
student remains a single-frame model: one calibration image plus one immediate
measurement image. There is no multi-second accumulation and no user judgement.

## Method

For every H2-only, RH20, and angle-80 RH20 reaction run, a leave-one-run-out
frame model produces an optical 3% probability. Isotonic regression enforces the
known non-decreasing H2 reaction direction within that run. Only low/high ends
of the smoothed probability become student-training examples. Four confidence
cutoffs (`0.65`, `0.75`, `0.85`, `0.90`) were tested automatically.

The student may change only baseline 2%/3% predictions. Model and thresholds
are selected inside each complete-video-held-out fold. Deployment requires 2%
improvement while preserving 0–1%, 3%, 4%, exact accuracy, video-macro accuracy,
MAE, and per-video 3%/accuracy.

## Results

| Variant | Teacher rows (2/3) | Exact | Video macro | R2 | R3 | MAE | Safe |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | – | 0.6753 | 0.6837 | 0.6735 | 0.3789 | 0.4457 | – |
| 0.65 | 328/75 | 0.6742 | 0.6828 | **0.7602** | 0.2671 | 0.4615 | no |
| 0.75 | 317/73 | 0.6867 | 0.6921 | 0.7398 | 0.3602 | **0.4423** | no |
| 0.85 | 289/60 | **0.6878** | **0.6952** | 0.7398 | 0.3665 | **0.4412** | no |
| 0.90 | 246/57 | 0.6855 | 0.6912 | 0.7500 | 0.3416 | 0.4480 | no |

The temporal teacher clearly makes the optical 2% region easier, but every
nested held-out variant reduces 3% recall. The 0.85 variant is closest to the
desired balance: exact accuracy, macro accuracy, 2% recall, and MAE improve, but
3% recall falls from 0.3789 to 0.3665. It is therefore not deployable under the
3%-preservation constraint.

An all-label diagnostic grid contains safe fixed thresholds, but those
thresholds were selected using the evaluation labels and are not an unbiased
deployment result. They are evidence that the trajectory teacher is useful,
not permission to update the application.

## Interpretation

The monotonic plots confirm three distinct run families:

- angle-80, H2-only run 4, and H2-only run 5 remain mostly in the low optical
  response region and reinforce 2%;
- RH20 run 5 and test-3 contain sustained high optical response and supply most
  3% examples;
- several runs cross the boundary briefly or noisily, so a fold trained without
  one of the few high-response runs shifts the 2/3 boundary toward 2%.

The remaining limitation is independent high-response/3% run diversity, not
the absence of a temporal smoothing algorithm.

## Decision and next step

Keep the production application unchanged. Next, apply run-balanced training:
cap the many correlated 2% frames per video, give equal weight to each source
run, and enforce a minimum contribution from the independent 3% runs. Evaluate
the 0.75/0.85 teacher labels with the same nested video-held-out constraints.
This remains fully automatic and single-frame at runtime.
