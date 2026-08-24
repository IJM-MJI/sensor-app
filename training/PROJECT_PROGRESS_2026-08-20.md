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

## Next actions

1. Human-audit the four rising-Reaction 60% endpoints in the aligned RH sheet;
   distinguish valid full response from a transition endpoint.
2. If valid, test a domain-normalized middle-stage expert; if transitional,
   downgrade only the reviewed response endpoint to interval supervision.
3. Fit simultaneous H2 and H2O-only-referenced RH interference correction.
4. Revisit the H2 2/3 gate only after an independent high-quality run supplies
   enough 2%/3% frames to test the .95 rule without using run 4 for selection.
5. Perform independent photo/app validation and then freeze publication figures.
