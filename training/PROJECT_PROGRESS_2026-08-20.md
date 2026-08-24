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

### Place-1 indoor-long interpretation audit

Per the user's visual interpretation, the indoor-long run was re-read as an
ordered within-yellow path rather than global colour families.  Large-droplet
hue at 20--30/40/50/60% was 102.54/99.77/97.84/95.95 degrees: a clean,
approximately linear move from olive-yellow toward ochre/orange.  A
leave-one-concentration-out linear interpolation within this run gave 25.45,
39.73, 50.04, and 60.18%RH (MAE .23%RH).  Descriptive hue midpoints are
101.15, 98.80, and 96.90 degrees.

This is strong evidence that 40/50 is optically separable in indoor-long, but
it is not an independent-run accuracy estimate.  Median lightness did not stay
strictly ordered (50% was slightly lighter than 40%), so hue shift must be the
primary feature and darkness only a low-weight confirmation.  These local
thresholds must not be deployed globally until a second place-1 long-format
run or late-interval pseudo-replicates validate them.

The requested late-interval audit then sampled five frames per block. One
second blocks reached .880 exact/.875 balanced with recalls
.90/1.00/.80/.80. Half-second blocks appeared better at .950/.938 with
recalls 1.00/1.00/.75/1.00, but this is not a valid monotonic colour model:
the RH40 and RH50 median hues reversed (100.32 vs 100.74 degrees) and their
ranges overlapped. Quarter-second blocks fell to .700/.625 and failed RH50
entirely. Only the single exact endpoint samples retained the clean
olive-yellow-to-orange order; neighbouring frames do not reproduce it
reliably.

Therefore the local thresholds remain undeployed. The evidence supports an
experimental 40--50% band from a still photo, not robust exact 40/50 output.
The generated frame sheet isolates RH40 at 44.75/45.00 s and RH50 from
88.25--90.00 s for human adjudication before changing endpoint supervision.

Endpoint-after sampling did not resolve the physical-order problem. Half-second
blocks over the first 1.5 s scored .960 exact/.950 balanced, and quarter-second
blocks over the first second scored 1.000/1.000, but both learned a reversed
RH40/RH50 relation: median hue was 100.14 degrees at 40% and 100.78 degrees at
50%, with overlapping ranges. Starting at +.75 s still gave 99.96/100.68
degrees and .880 exact. A single frame near +.95 s briefly restored the expected
ordering, but the surrounding frames did not. Lightness was also reversed
(50% lighter than 40%). These high within-run scores therefore reflect a
repeatable run-specific signature, not the requested olive-to-orange physical
trajectory, and remain undeployed.

The enlarged raw/patch-balanced review confirms that the ROI follows the main
droplet. The next scientifically defensible existing-data output is a 40--50%
band. Exact 40/50 requires another independent place-1 rising run or a supplied
human decision that a specific endpoint frame, rather than its surrounding
frames, is the valid optical reference.

### Cross-run RH40/50 spatial audit

The 40/50 question was then tested across all four rising RH-only runs, rather
than inferred from indoor-long alone. Three runs showed the expected endpoint
hue decrease from 40 to 50% (indoor-long -1.93, response-3 -.82, response-6
-.62 degrees); indoor-fast reversed by +1.69 degrees. A complete-run-held-out
binary classifier using whole-droplet relative colour scored .500 with
40/50 recalls .75/.25. Splitting the main droplet into core/rim/top/bottom/
left/right improved exact/balanced accuracy to .625, but 50% recall remained
.25. Excluding the known poor-ROI indoor-fast run reduced both candidates to
.500 with 40/50 recalls .67/.33, so FAST alone does not explain the failure.

The cross-run zoom sheet shows that long, response-3, and response-6 do contain
a visible 40-to-50 darkening/orange tendency, but its magnitude and whole-frame
exposure shift differ substantially by run. Current per-run endpoint evidence
is insufficient to learn a transferable exact 40/50 boundary. The spatial
candidate is not deployed; 40--50 remains the honest cross-run output band.

The follow-up illumination-control A/B subtracted both nearby-substrate change
and the nominally invariant H2=0 flame change from the large-droplet LAB
response. Nearby-substrate correction scored .250 exact/balanced; flame and
dual controls scored .500/.500. Adding dual controls to the spatial model left
the previous .625/.625 result and 40/50 recalls 1.00/.25 unchanged. The control
regions had ample pixels, so this is not a missing-mask failure. Predictions
still separated mainly by run/location, showing that a simple global exposure
offset is not the dominant source of non-transferability. No corrected model
is deployed.

### Four-range RH held-out result

Exact endpoints were merged into 20--30/40--50/60--70/80--90% ranges and all
whole, spatial, and illumination-control candidates were evaluated with a
complete run held out. The best dual-control model reached .600 exact, .604
balanced, and .967 within one adjacent range. Per-range recalls were
1.00/.375/.375/.667. The high adjacent-range score confirms that most errors
are one-band scale shifts between runs, but neither the middle ranges nor the
.85 deployment rule passed. The four-range model is therefore not deployed.

### Place-2 response3/response6 pairwise ranges

Place 1 was intentionally deferred because FAST and long differ in lighting,
ROI quality, and response scale. For place 2, response3 was trained and
response6 held out, then the direction was reversed, using a fixed regularized
model rather than test-fold tuning. Nearby-substrate correction reached .875
exact/.875 balanced in both directions and 1.00 within one band. Combined
recalls for 20--30/40--50/60--70/80--90 were 1.00/.75/.75/1.00. The two errors
were response3 40% predicted 20--30 and response6 70% predicted 80--90.

This is promising evidence that a place-2 profile is viable, but it does not
pass the predeclared .85 recall in every band. Feature selection also cannot be
independently confirmed with only these two runs, so it remains experimental
and is not yet deployed to the app. A third independent place-2 rising run is
the clean validation needed before freezing this profile.

The two place-2 errors were then audited over +/-1 s at 4 Hz. Response3 stayed
in the 20--30 prediction from nominal RH35 through RH44.3, including its 40%
endpoint. Response6 stayed predominantly in 80--90 from nominal RH65.3 through
RH74.7; only one early frame briefly predicted 60--70. Confidence was low
(roughly .27--.35), but moving either endpoint by several frames did not restore
the correct range. The enlarged droplets and substrate-corrected LAB traces are
continuous across the supplied endpoints. Therefore neither error is caused by
a single bad boundary frame, and the timelines are unchanged. The remaining
failure is a run-to-run response-scale mismatch, not an endpoint annotation
mistake.

### Place-2 run-local timeline-warp sensitivity

The full response3 and response6 rising trajectories were predicted by a model
trained on the opposite run.  The inferred 40--50/60--70/80--90 transitions
did not share one constant delay.  Response3 crossed them at nominal RH
45.7/52.5/75.6%, while response6 crossed them at 39.3/60.0/66.7%.  The implied
boundary shifts span -8.3 to +10.7% RH and change sign within each run.  A
single global time shift is therefore rejected; run-to-run response amplitude
and local kinetics both matter.

A diagnostic label-sensitivity test then applied the user's directional
hypothesis. Moving response3's 40% endpoint from 5.0 to 6.13 s reduced pairwise
exact/balanced accuracy from .875/.875 to .8125/.8125 and reduced the minimum
band recall from .75 to .50.  Moving response6's 70% endpoint from 18.0 to
16.67 s improved exact/balanced accuracy to .9375/.9375 and corrected the
60--70 recall to 1.00, although 40--50 recall remained .75. Applying both
shifts returned accuracy to .875 and minimum recall to .50.  Consequently the
response6 high-range adjustment is a useful run-local calibration candidate,
but the response3 shift is rejected. These values were chosen using the same
two trajectories and are not independent validation, so neither change is
deployed to the app.

### Response3 endpoint-model adjustment

The response3 40% time was scanned across the single-frame optical 40--50%
window (6.40--7.47 s), while retaining the response6 70% correction at 16.67 s.
Times from 6.67 s onward made the held-out response3 endpoints perfect, but
reduced the opposite response6 fold to .75. No response3 time shift improved
both transfer directions, so its supplied 5 s endpoint was restored.

Nine fixed classifier families were then tested without changing response3's
timeline. Background-substrate-controlled 1-nearest-neighbour was the only
candidate to classify all 16 endpoints correctly when each complete run was
held out in turn: exact/balanced accuracy and every range recall were 1.00.
Most alternative models reached .9375 exact but retained a .75 minimum recall,
showing that response3 is separable as a local colour pattern even though a
single linear boundary misses it.

The 1-NN endpoint result is sensitive to sampling position. Testing the last
0.27 s before each endpoint scored .906 exact/.906 balanced with combined
recalls 1.00/.875/.875/.875, but the response3-only fold was .8125. Extending
to the last .53 s reduced the minimum recall to .727. Adding these late-ramp
frames to training made transfer worse rather than better, because even this
short interval contains appreciable ramp progression. The defensible current
candidate is therefore an endpoint/full-response range model, not a general
ramp-frame model. It remains undeployed until it is tested on an independent
place-2 run or in a clearly marked experimental app profile.

### Experimental app profile

The endpoint candidate was exported as `sensor-rh-place2-model.js` and wired
into the app only after the four-state classifier selects H2O-only. The app now
extracts the same tight registered main-droplet median and nearby-substrate
median used by the offline model, subtracts their RH20 calibration difference,
standardizes the resulting LAB vector, and applies the 16-prototype 1-NN
profile. Existing H2 quantitation, simultaneous-number suppression, and state
classification are unchanged. The UI labels this output `Place-2 endpoint
experimental` and includes the nearest-prototype distance as `rhD`.

JavaScript syntax, all 16 exported prototype labels, asset loading, and the
visible app version were verified locally. The automated browser had no camera,
so real/monitor-captured endpoint validation remains the next required step.
The exact calibration frames, endpoints, expected ranges, repetitions, and
acceptance rule are recorded in `PLACE2_RH_APP_TEST.md`.
