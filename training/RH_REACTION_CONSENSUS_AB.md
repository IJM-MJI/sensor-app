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

## Calibration-selected place mixture result

The proposed place-1/place-2 split was evaluated by hiding a complete run. The
held-out run's 20--30% calibration features voted for a place; the RH expert
then used only the remaining run(s) assigned to that place. A true-place oracle
was also evaluated to separate domain-selection error from concentration error.

Calibration place selection was correct for both place-1 runs but classified
both place-2 response runs as place 1: **2/4 = 50% domain accuracy**. The
predeclared registered-droplet LAB expert obtained:

| Middle-endpoint model | Exact | Balanced | MAE | Recall 40/50/60 |
|---|---:|---:|---:|---|
| Current global one-anchor | .167 | .167 | 16.25%RH | .25/.00/.25 |
| True-place oracle | .333 | .333 | 8.33%RH | .25/.50/.25 |
| Calibration-selected place | .250 | .250 | 10.00%RH | .25/.25/.25 |
| Global forced-middle control | **.500** | **.500** | **6.67%RH** | **.25/.50/.75** |

The global forced-middle control uses the same complete-run holdout and the same
Initial centring but does not split prototypes by place. It preserves 40%
recall and improves both 50% and 60%. Therefore the gain comes from a dedicated
40/50/60 expert, not from place separation. Even correct place information is
weaker than pooling independent runs, so a place mixture is not deployed.

The forced-middle score is not yet a full application score: it assumes the
sample is already known to be within 40--60%. The next leakage-safe stage must
first classify a coarse band (`20--30`, `40--60`, `70--90`) on every held-out
endpoint and invoke the middle expert only when that gate predicts `40--60`.

Artifacts:

- `output/rh_location_mixture_v1/metrics.json`
- `output/rh_location_mixture_v1/predictions.csv`
- `output/rh_location_mixture_v1/rh_location_mixture_validation.png`

## Verified place ceiling and coarse-to-middle hierarchy

The user confirmed that place 1 normally reaches only about 70--80% RH even
when its supplied timeline names a 90% setpoint, whereas place 2 reaches 90%.
Accordingly, the place-1 `rh-indoor-fast` nominal-90 endpoint is now interval
supervision (`70--80%`) and is excluded from exact-stage scoring. Place-2 90%
endpoints remain exact. This is a reference correction, not a model-generated
relabel.

A complete-run-held-out hierarchy then classified every endpoint into
`20--30`, `40--60`, or `70--90`; the dedicated global middle expert was invoked
only when the gate predicted `40--60`.

| Model | Exact | Balanced | Within one stage | MAE |
|---|---:|---:|---:|---:|
| Fine baseline | .689 | .456 | .869 | 5.82%RH |
| Gate + middle expert | .689 | .487 | .852 | 5.82%RH |

The coarse gate obtained .770 exact band accuracy, but its recalls were
`.969/.333/.706` for low/middle/high. The hierarchy raised 60% recall from .25
to .50, but left 50% recall at zero, lowered low-stage recall, and reduced the
within-one-stage score. It therefore fails the deployment criteria and does
not replace the app model.

The failure is systematic across acquisition domains: most place-1 40--60%
endpoints are routed to the low band, while later place-2 middle endpoints are
routed to the high band. Automatic place selection and even the true-place
oracle already failed to solve this in the preceding test. The existing videos
therefore do not support a defensible automatic location-specific correction.

Artifacts:

- `output/rh_coarse_middle_hierarchy_v1/metrics.json`
- `output/rh_coarse_middle_hierarchy_v1/predictions.csv`
- `output/rh_coarse_middle_hierarchy_v1/rh_coarse_middle_hierarchy.png`

## Human-guided yellow-to-purple colour path

The user's visual hypothesis was tested directly on 31 rising-Reaction
endpoints: low RH is yellow, followed by orange, scarlet, and purple as RH
increases. The analysis reconstructed each run's calibrated droplet colour and
used LAB direction, circular hue change, chroma, warm/purple pixel fractions,
and cached pixel-colour quantiles. Place-1 nominal 90 remained excluded as
verified 70--80% interval evidence. Every score held out a complete run.

The hypothesis is supported as a within-run trajectory. Background-corrected
hue moved in one direction with RH at 100%, 80%, 100%, and 100% of consecutive
stage transitions in indoor-fast, indoor-long, response-3, and response-6.
However, the magnitude was not transferable. At 80%, hue change was about
-20/-11 degrees in the two place-1 runs and -47/-69 degrees in the two place-2
runs. Even within the same place the scale differed substantially.

Ordered threshold models improved markedly over a single straight-line ridge,
but the best exact-endpoint candidate still reached only .355 exact/.307
balanced accuracy (delta LAB), while the absolute-colour-plus-path candidate
reached .290 exact/.321 balanced. No 40--70% candidate approached the required
.85 per-stage recall. Cached raw-colour channel quantiles were also too coarse:
they summarize each channel separately and cannot preserve the paired pixel hue
distribution that a person sees.

Decision: retain the colour-path finding as a strong feature-design clue but do
not deploy these models. A future extraction pass should calculate paired-pixel
hue histograms inside the registered droplet mask (yellow/orange/scarlet/purple
fractions) rather than trying to reconstruct literal colour from independent
channel quantiles. That A/B must still use complete-run holdout.

Artifacts:

- `output/rh_human_color_path_v1/metrics.json`
- `output/rh_human_color_path_v1/predictions.csv`
- `output/rh_human_color_path_v1/rh_human_color_path_validation.png`
