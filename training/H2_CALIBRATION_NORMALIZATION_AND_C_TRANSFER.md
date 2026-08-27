# H2 calibration normalization and family-C transfer audit

## Deployable normalization

Only information available to the application was used: the calibration image
and the current image. No reaction endpoint or future frame was used.

For family B, dividing colour change by the initial flame-to-substrate contrast
changed accuracy from 80.7% to 79.8% and minimum recall from 77.4% to 76.5%, so
it is not accepted as a replacement. It did, however, recover 10 of 11 weak
`run3` `2-3` frames that the raw model missed, while increasing errors in
`run4`. This indicates a useful environment-specific correction rather than a
universal normalization.

## Family-C transfer

`run5_normal` was used as the only independent physical family-C run.
`run5_x2` was excluded from independent validation because it is the same
recording at a different playback speed.

| Three-run model | Accuracy | 0 recall | 1-2 recall | 2-3 recall |
|---|---:|---:|---:|---:|
| Raw colour change | 70.7% | 100.0% | 65.8% | 65.0% |
| Calibration-normalized | 63.1% | 100.0% | 60.8% | 53.1% |

Mixing family C with B therefore harms transfer and is rejected.

## Duplicate consistency check

Training on one of `run5_normal` / `run5_x2` and testing on the other gives
86.5% agreement, with recalls of 100.0% (`0`), 80.6% (`1-2`), and 85.9%
(`2-3`). This is not independent validation, but it shows that family-C labels
are internally coherent. The transfer failure is caused by environmental
appearance differences rather than playback speed alone.

## Decision and next step

Keep A, B, and C as separate concentration-model environments. Train an
environment router from calibration-frame geometry and initial flame contrast,
then run the corresponding concentration model. The router must be validated by
holding out complete videos; family C must count as one physical run.
