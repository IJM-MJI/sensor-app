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

## Next actions

1. Human-mark the ten atlas endpoints as `VALID`, `PARTIAL RESPONSE`, or
   `INVALID REFERENCE`.
2. Tighten the flame component mask to remove patch/hardware pixels and rerun a
   held-out mask A/B without changing labels.
3. Apply human endpoint weights and the improved mask together; deploy only if
   H2 video-held-out metrics improve.
4. Repeat the endpoint/mask audit for RH-only, prioritizing 40--60% RH.
5. Fit simultaneous H2 and H2O-only-referenced RH interference correction.
6. Perform independent photo/app validation and then freeze publication figures.
