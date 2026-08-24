# Dual-sensor application progress report — 2026-08-20

## Target

Build a single-photo application that first classifies `initial/none`, `H2 only`,
`RH only`, or `simultaneous`, then reports H2 and RH concentration ranges from
the flame and droplet shapes. RH in simultaneous exposure remains anchored to
the H2O-only droplet response; the simultaneous interference correction is a
separate final stage.

## Completed

1. Consolidated supplied H2-only, H2O-only, and simultaneous timelines. Ramp
   intervals are interpreted as concentrations reached at interval ends.
2. Added chamber detection, quarter-turn orientation locking, flame/droplet
   regions, calibration-image subtraction, patch/illumination balancing, and
   single-photo inference. Recovery does not supervise H2 response quantitation.
3. Trained and deployed a direct four-class state model. State is not derived
   from the numeric H2/RH outputs. Raw video-held-out exact accuracy is .608;
   stable-segment aggregation is .759. The app rejects low-confidence frames.
4. Built ordered H2 and RH concentration validation with video-held-out and
   five-second-block protocols, concentration confusion matrices, and browser
   model export.
5. Added reviewed RH20 simultaneous ramps as weak H2 supervision only for the
   user-approved 0%, 1%, and 3% stages. The current best leakage-safe H2
   confidence hybrid has exact .519, balanced .549, within-one-stage .974, and
   MAE .508 stage. Recall for H2 0--4% is .513/.543/.748/.422/.516.
6. Tested whole-run removal, endpoint removal, local optical-outlier weighting,
   calibration-colour projection, and leakage-safe ensembles. None improved the
   current H2 hybrid, so the weaker candidates were not deployed.
7. RH-only quantitation currently has video-held-out exact .285 and MAE 14.8% RH;
   within-run five-second blocks reach exact .636 and MAE 5.6% RH. The gap shows
   that cross-run/domain generalization, not within-video separation, is the
   limiting RH problem.

## Latest endpoint decision atlas

`training/output/h2_endpoint_decision_atlas_v1/h2_endpoint_decision_atlas.jpg`
contains raw chamber crops, the registered flame region, actual selected pixels,
calibrated LAB deltas, current hybrid prediction, and within-run optical
stability for all five H2-only 0%/4% anchors.

Automatic diagnostic result (not a replacement for human validity review):

- 6/10 endpoints: prediction/reference consistent.
- 3/10 endpoints: adjacent-stage ambiguity (`H2_only_test` at its actual 2 s
  calibration point, plus run 5 at 0% and 4%).
- 1/10 endpoint: stable but two stages wrong (`H2_only_4`, 4%).

The atlas also exposes flame-mask contamination by a white patch or chamber
hardware in several frames. This can corrupt concentration features even when
the timeline label is correct and must be separated from label-validity review.

## Current boundary of completion

- Four-state classification: implemented and deployed, but simultaneous recall
  remains the weakest state and the requested .85 per-state target is not met.
- H2-only concentration: complete experimental pipeline and usable adjacent-band
  behavior, but exact-stage accuracy is not final.
- RH-only concentration: stages are separable within a run, but unseen-run
  accuracy is not final.
- Simultaneous concentration: state classification exists; quantitative RH
  interference correction is still pending.
- Paper figures: validation plots/confusion matrices exist; final figures must
  wait for the selected H2/RH/simultaneous models.

## Verified H2 reference correction

Human review confirmed `H2_only_test` at 2 s as H2 0%. Run 4 and run 5 did not
reach 4%; their late response is now treated as weak 2--3% interval evidence,
not exact validation. With leakage-safe folds and weight .002, the corrected
model obtains exact .528, balanced .525, within-one-stage .929, and 0%/4%
recalls .658/.832. It is scientifically cleaner but still weak at 2%/3%, so it
has not replaced the app model.

## H2 2/3% quality and RH20 boundary test

The low-quality run-4/run-5 hypothesis and reviewed RH20 stage-2/stage-3
alternatives were tested in `H2_23_QUALITY_RH20_AB.md`. Run 5's late partial
frame was objectively the weakest (Laplacian variance 47 versus 143--531 in the
verified runs). Display enhancement makes the colour easier to inspect but does
not recover missing physical response and is not used for inference.

A leakage-safe, verified-only 2-vs-3 expert reached .815 exact and .774 balanced
accuracy. Adding RH20 2/3 weak samples reduced those values to .740/.690, so
they were rejected. Confidence-gating the expert with the corrected global
model kept exact accuracy at .528, raised balanced accuracy from .525 to .545,
within-one-stage from .929 to .944, and lowered MAE from .565 to .542, but 3%
recall fell from .316 to .243. It is therefore retained as an experiment and
not deployed yet.

The app now includes an optional display-only colour enhancement switch for a
captured frame. The ML path still consumes the untouched canvas pixels.

The subsequent asymmetric-threshold test preserved 3% recall in its diagnostic
fixed-gate result (.316) and raised 2% recall from .392 to .432, but the three
new correct 2% frames all came from run 4. When run 4 was genuinely held out,
the nested selector found no feasible gate. The honest aggregate result did not
improve 2% recall, so the gate was not deployed. Details are appended to
`H2_23_QUALITY_RH20_AB.md`.

## RH rising-Reaction consensus checkpoint

The first RH-only follow-up now separates rising Reaction from the single
daylight Recovery run and compares an Initial-centred logistic model with a
leave-one-run-out consensus prototype. Consensus improves exact accuracy from
.710 to .742, stage balance from .484 to .493, within-one-stage from .855 to
.887, and MAE from 5.81 to 4.68%RH. It raises 40/50% recall to .50/.25, but 60%
recall falls from .25 to zero and 80% falls from .50 to .25. It is not deployed.
No endpoint label was changed. See `RH_REACTION_CONSENSUS_AB.md`.

The follow-up place audit compared the user-requested response-3 5 s and
response-6 13 s frames with place-1 60% endpoints. Both requested frames are
substantially closer than their own nominal 60% endpoints across all tested
registered-droplet feature sets. Because 5/13 s are the supplied place-2 40%
endpoints, this is evidence of a place-dependent optical coordinate shift, not
a reason to relabel them 60%. This motivated a direct location-mixture A/B.

The location-mixture A/B then showed that Calibration cannot select the place
reliably: both place-1 runs were correct, but both place-2 runs were assigned to
place 1 (50% domain accuracy). The automatic place model scored .250 exact on
40/50/60 endpoints and even the true-place oracle scored .333. A no-place,
forced-middle prototype scored .500 exact with recalls .25/.50/.75 and MAE
6.67%RH. Thus pooling runs is better than splitting locations; the place model
is rejected. The next candidate is a coarse-band gate plus the global middle
expert.

The place ceiling has now been corrected: place 1's nominal 90% endpoint is
treated as 70--80% interval evidence, while place 2 retains exact 90% labels.
The resulting coarse-band gate reached .770 accuracy, but only .333 recall for
the crucial 40--60% band. Routing its middle predictions to the dedicated
expert left exact accuracy unchanged at .689, improved balanced accuracy from
.456 to .487, reduced within-one-stage accuracy from .869 to .852, and did not
recover 50%. The hierarchy is scientifically cleaner but is not deployable.
See `RH_REACTION_CONSENSUS_AB.md` and
`output/rh_coarse_middle_hierarchy_v1/`.

The user-observed yellow -> orange -> scarlet -> purple RH path was then tested
on exact Reaction endpoints. Initial-centred background-corrected hue followed
the correct direction for 80--100% of stage transitions in every run, so the
visual observation is present in the extracted sensor signal. The amount of
hue travel is strongly run/location dependent, however: place-1 80% endpoints
moved about 11--20 degrees, compared with 47--69 degrees in place 2. Ordered
models did not transfer to a hidden run (best exact .367, balanced .321 across
the tested candidates after removing an unavailable endpoint), and none passed
the .85 per-stage rule. The app remains
unchanged. The next existing-video feature experiment is paired-pixel hue-bin
extraction from the registered droplet mask; independent LAB channel quantiles
cannot represent the literal colour family reliably.

Paired-pixel hue extraction was subsequently run on the registered droplet
mask. On the corrected 30-endpoint held-out set, named colour fractions raised
exact accuracy from .333 to .433, balanced accuracy from .304 to .357,
within-one-stage accuracy from .633 to .767, and lowered MAE from 13.33 to
11.83%RH. It recovered 50/80 endpoints but left 40/60 at zero and reduced 90
from 1.00 to .50. The candidate is rejected, the app is unchanged, and the H2
yellow-to-light-green extension is intentionally not started. The completed
colour-family atlas shows a broad-yellow quantization problem in place 1 and
board/hardware mask contamination in place 2. It also removed an invalid 180 s
substitute for the unavailable 189 s indoor-long endpoint; extra 9 s remains
the valid 70% evidence.

## Next actions

1. Keep the place-1 nominal-90 reference as a 70--80% interval and place-2 90%
   as exact; do not merge these into one exact 90% class.
2. Record at least one new independent rising-Reaction run per place, preferably
   two, with stable holds and chamber-sensor readings at 40/50/60%. Include
   confirmed 70/80% in place 1 and 70/80/90% in place 2.
3. Repeat complete-run-held-out coarse and exact-stage validation. Require at
   least .85 recall in every deployed band/stage, not frame-random accuracy.
4. Until that evidence exists, keep exact RH quantitation experimental and show
   an uncertainty/range result rather than silently applying this hierarchy.
5. Tighten the response-run droplet mask against board edges/hardware and add
   relative within-yellow hue/chroma/intensity features for place 1, then repeat
   the same corrected 30-endpoint A/B.
6. Apply the same paired-pixel idea to H2 yellow-to-light-green only if the RH
   correction improves exact and balanced accuracy without lowering any stage.
7. Fit simultaneous H2 and H2O-only-referenced RH interference correction only
   after the RH-only reference model passes its held-out criterion.
8. Revisit the H2 2/3 gate only after an independent high-quality run supplies
   enough 2%/3% frames to test the .95 rule without using run 4 for selection.
9. Perform independent photo/app validation and then freeze publication figures.

## Inner-ROI plus within-yellow A/B

The two follow-up corrections were tested together on the same 30 valid RH-only
rising endpoints.  The registered droplet template was contracted to an inner
core, the large droplet and satellite were measured separately, and hue,
chroma, and lightness were expressed relative to each run's 20--30% baseline.
The large-droplet-only candidate improved exact accuracy from .433 to .467 and
balanced accuracy from .357 to .429.  It restored 90% recall to 1.00 and raised
80% recall to .50, but 40% and 60% remained zero and 70% fell from .25 to zero.
It therefore fails the every-stage and .85 criteria and is not deployed.

Adding the small satellite reduced exact/balanced accuracy to .300/.250.  Its
weak colour response and frequent low pixel count dilute the useful large-drop
signal, so it must remain an audit/consistency region rather than a primary RH
concentration feature.

The endpoint audit explains the remaining 40/50 ambiguity in response-6: main
droplet hue was 78.25/77.62 degrees and median lightness was identical at the
two endpoints.  Response-3 likewise differed by only .82 degrees, while its
chroma change had the opposite direction from response-6.  A single-frame,
cross-run 40-vs-50 decision is therefore not identifiable reliably from these
endpoints.  The next existing-video experiment should aggregate several late
interval frames and report a 40--50 band unless an independently held-out run
shows a reproducible within-yellow separation.
