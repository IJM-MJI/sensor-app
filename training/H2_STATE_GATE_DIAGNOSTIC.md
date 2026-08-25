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
