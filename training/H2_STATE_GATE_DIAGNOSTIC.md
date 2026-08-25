# H2 state-gate diagnostic v8

The four-state model's combined H2 probability (`pH2`) is not sufficient to
recover the early response3 frames: it was 0.38 at 1.5 s and 0.61 at 2 s even
though both frames were RH-only low-response conditions.  Raising the existing
`pH2 < 0.25` recovery threshold would therefore risk hiding real H2 responses.

Version v8 leaves the selected state unchanged and exposes two independent
flame-centred measurements on every calibrated frame:

- `pH2x`: probability from the deployed binary H2-presence forest;
- `rawH2`: continuous output of the deployed H2 concentration model, including
  when the four-state result is `Uncertain / Retake`.

The app must not convert an uncertain frame to Initial until the following
saved-video comparison shows a conservative separation rule.

## Required saved-frame comparison

Use `Load saved frame`, not a photograph of a monitor.

1. `1_90_H2_only_test_2`: calibrate at 2 s, then test 13, 21, 30, and 51 s
   (nominal H2 endpoints 1, 2, 3, and 4%).
2. `1_90_H2O_only_3(response)`: use the established low-RH calibration and
   retest 1.5, 2.0, and 2.5 s.

For every frame retain the state plus `pH2`, `pH2x`, `rawH2`, `dFl`, and `dDr`.
The state gate may be adopted only if all three RH-only frames are rejected
without changing any accepted H2 endpoint to Initial.  A threshold selected
from these same frames remains an app-domain diagnostic and requires a later
untouched-run check before it is reported as validation accuracy.

## Offline safety observation

Applying `rawH2 <= 0` to every low-confidence held-out frame would recover many
RH-only cases, but it would also convert one nominal H2 4% frame from
`H2_only_test_2` at 88 s.  The candidate is therefore not deployed without the
additional RH low-chroma condition and the app-domain endpoint comparison.

## Low-H2 state-label A/B

The app-domain test showed that the nominal 1% endpoint had `rawH2=0.25` but
was labelled Initial.  Inspection of the state-training policy found that H2
1--2% ramp frames had no `h2_present` target; only the earlier stable 3--4%
anchors supervised H2 presence.

A leave-one-experiment-group-out A/B added H2-only rows only after their
continuous nominal ramp reached 1%.  Relative to the same 320-tree model:

| Policy | Accuracy | Balanced accuracy | H2-only recall | RH-only accuracy |
|---|---:|---:|---:|---:|
| Stable 3--4% anchors | 71.9% | 70.1% | 73.5% | 79.1% |
| Add verified >=1% ramp endpoints | 73.1% | 70.6% | 77.5% | 80.0% |

The >=1% policy improves H2-only recall without reducing the RH-only result and
is therefore used for the next exported four-state model.  The RH-only
low-response post-processing threshold remains disabled until this retrained
model is repeated on the supplied app frames.

## v9 app repeat and v10 guard

The repeated H2 endpoints produced `rawH2=0.25, 1.47, 3.44, 5.25`.  The 2--4%
states remained H2-only, while the nominal 1% frame remained optically
indistinguishable from Initial.  It must not be promoted solely from its
timeline label.

The response3 RH-only frames produced `rawH2=0.04, -2.18, 0.42`.  The first was
Uncertain, the second became a low-confidence false H2-only, and the third was
correctly H2O-only.  Version v10 recovers Initial only when all of these hold:

- the direct RH endpoint vector has no chromatic change;
- `-3.0 <= rawH2 <= 0.10`;
- the state is Uncertain, or H2-only with maximum state probability below 0.45.

The lower raw bound preserves the previously identified `rawH2=-7.06` gross
H2 extrapolation as Uncertain instead of silently converting it to Initial.
The rule does not affect the supplied H2 endpoint values or the response3
2.5 s H2O-only result.

## v10 repeat and v11 low-H2 consensus

The RH-only correction behaved as intended: response3 1.5 and 2 s became
Initial with `lowGuard=1`, while 2.5 s remained H2O-only.  On the H2 run, 15 s
still had weak optical evidence (`rawH2=0.78`, `pH2x=0.28`), 17 s had a
consistent flame response (`rawH2=1.61`, `pH2x=0.61`, combined `pH2=0.47`),
and 19 s was already H2-only (`rawH2=1.78`).

Version v11 promotes a weak Initial result to H2-only only when:

- maximum four-state probability is below 0.50;
- the RH endpoint vector remains low-chroma;
- combined four-state H2 probability is at least 0.40;
- independent H2 probability is at least 0.55;
- `rawH2 >= 1.50`.

The rule changes the supplied 17 s app-domain case.  In the 881-frame grouped
held-out audit it promoted no Initial, H2O-only, or H2-only frame; consequently
it introduces no measured false promotion but also cannot be claimed as a
held-out accuracy improvement.  It remains an app-domain correction pending a
repeat on an untouched H2-only run.
