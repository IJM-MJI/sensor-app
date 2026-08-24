# RH rising-Reaction consensus A/B (2026-08-24)

## Scope

This experiment starts the RH-only quantitative refinement without changing any
user-supplied endpoint label. It uses the registered droplet ROI and only the
four rising H2O-only Reaction groups:

- `rh-indoor-fast`: `1_90_H2O_only_2_extract`
- `rh-indoor-long`: `1_90_H2O_only_extract_3min` plus its extra continuation
- `rh-response-3`: `1_90_H2O_only_3(response)`
- `rh-response-6`: `1_90_H2O_only_6(response)`

The daylight 90-to-20% Recovery recording is excluded from this A/B because its
descending hysteresis is not interchangeable with rising Reaction. It remains
available for a separate recovery model/audit.

Each held-out run supplies only its known 20--30% Initial calibration anchor.
The model never receives that run's middle/high endpoint labels during fitting
or hyperparameter selection.

## Models

1. **One-anchor logistic:** application-compatible Initial-centred endpoint
   classifier.
2. **Consensus prototype:** builds group-balanced level prototypes from the
   remaining runs. Feature subset and median/nearest/two-reference consensus
   are selected inside the outer leave-one-run-out split.

## Held-out results

| Metric | One-anchor logistic | Consensus prototype |
|---|---:|---:|
| exact accuracy | .710 | **.742** |
| stage-balanced accuracy | .484 | **.493** |
| within one stage | .855 | **.887** |
| MAE | 5.81%RH | **4.68%RH** |

Per-stage recall (20--30/40/50/60/70/80/90):

- one-anchor logistic: `.906/.250/.000/.250/.818/.500/.667`
- consensus prototype: `.969/.500/.250/.000/.818/.250/.667`

The consensus improves 40% and 50%, but loses the sole correct 60% endpoint and
reduces 80% recall. It therefore does not satisfy the per-stage preservation
rule and is not deployed.

## Run-level diagnosis

- Indoor-fast: consensus recognizes 40% and 50%, then compresses 60--90% to
  approximately 50%.
- Indoor-long: 40--60% remain close to the Initial-domain prototype despite the
  user's confirmed visible colour change; this indicates a cross-run colour
  coordinate problem rather than absence of response.
- Response-3: 40% is recovered, 50% maps to 60%, and 60% maps back to 40%, so the
  middle path is not monotonic in the current compact features.
- Response-6: 40/50/60% map to 60/60/70%; the short endpoint series is advanced
  relative to other Reaction runs.

The endpoint predictions often have very small prototype margins, confirming
that 40--60% regions overlap after Initial-only centring. A single low anchor is
not enough to place every run on a common absolute RH colour coordinate.

## Decision and next evidence

Keep the current RH app model unchanged. Before changing labels, visually audit
the four 60% Reaction endpoints together. The important question is whether the
nominal 60% frame in each run is a valid full-response endpoint or still an
optical transition. If all are valid, the next model must learn run/domain
normalization rather than timeline correction. If a response endpoint is
transitional, it should become interval supervision rather than an exact label.

Artifacts:

- `output/rh_consensus_endpoint_v1/metrics.json`
- `output/rh_consensus_endpoint_v1/predictions.csv`
- `output/rh_consensus_endpoint_v1/rh_consensus_endpoint_validation.png`
- `output/middle_endpoint_review/rh_middle_endpoint_review.jpg`

## Place-1 versus place-2 60% candidate audit

The user identified indoor-fast/indoor-long as place 1 and response-3/response-6
as place 2, then requested an optical comparison using response-3 at about 5 s
and response-6 at about 13 s. These requested frames were compared with the two
place-1 nominal 60% endpoints after each run's 20--30% Initial centring. No
timeline label was changed.

| Place-2 run | Requested frame | Requested distance | Nominal 60% distance |
|---|---:|---:|---:|
| response-3 | 5.07 s | **.589** | 1.116 at 10.93 s |
| response-6 | 13.07 s | **.812** | 1.855 at 16.00 s |

Values above use registered-droplet LAB; LAB plus the flame reference and the
full compact feature set give the same requested-is-closer conclusion. The
requested times are approximately the supplied **40% endpoints**, not alternate
60% timestamps. Therefore this result must not relabel them as 60%. It shows
that a place-2 40% state can occupy the place-1 60% optical region even after
Initial subtraction. The main RH bottleneck is location/domain-dependent colour
response, not a single shared timing correction.

The next defensible A/B is a location-aware mixture: infer the acquisition
domain from the calibration frame, then use separate place-1 and place-2 RH
prototypes. It must be evaluated by holding out a complete run within each
place. If it fails, the present videos do not contain enough independent runs
per place for absolute RH quantitation.

Additional artifacts:

- `output/rh_location_60_candidates_v1/metrics.json`
- `output/rh_location_60_candidates_v1/comparison.csv`
- `output/rh_location_60_candidates_v1/rh_location_60_candidate_atlas.jpg`
