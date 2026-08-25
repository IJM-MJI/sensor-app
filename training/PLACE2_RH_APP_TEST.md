# Place-2 experimental RH endpoint app test

This profile is restricted to H2O-only, full-response photographs. It reports
four ranges: 20--30, 40--50, 60--70, and 80--90% RH.

## Test procedure

1. Display one video with the flame above the droplet and keep the monitor and
   camera position fixed.
2. Calibrate once at the listed low-RH frame.
3. Move to each endpoint, pause, and take one photo immediately. Do not use a
   five-second accumulation.
4. Record the state result, RH range, and `rhD` value shown in diagnostics.
5. Repeat each endpoint twice without recalibrating. Recalibrate before
   switching to the other video.

## Response 3

Video: `1_90_H2O_only_3(response).mp4`

- Calibration: 0.5 s (RH20 condition)
- 2 s, 3 s: expected 20--30% RH
- 5 s, 6.5 s: expected 40--50% RH
- 11 s, 25 s: expected 60--70% RH
- 28 s, 38 s: expected 80--90% RH

## Response 6

Video: `1_90_H2O_only_6(response).mp4`

- Calibration: 2 s (RH20 condition)
- 7 s, 10 s: expected 20--30% RH
- 13 s, 14 s: expected 40--50% RH
- 17 s, 18 s: expected 60--70% RH
- 19 s, 32 s: expected 80--90% RH

The app-domain sweep placed the response-6 60--70 optical band at 17--18 s and
the 80--90 transition at 19 s, so app validation no longer uses 16.67 s. The
current prototype file retains its 16.67 s provenance until an independent
physical-sample run authorizes retraining. Response-3 7 s is the first frame
of the following ramp; 6.5 s is used as the transition-safe 40--50 range anchor.

## Acceptance rule

- The four-state classifier must first return `H2O-only Response`.
- At least 28 of 32 repeated endpoint photographs must match their RH range.
- Every range must have at least 7 correct results out of 8.
- A result obtained from a ramp-transition frame is diagnostic only and is not
  counted against the endpoint model.
