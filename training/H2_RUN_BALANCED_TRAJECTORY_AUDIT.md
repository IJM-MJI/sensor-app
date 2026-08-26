# H2 run-balanced trajectory audit

## What “same number of similar frames” means

For every video and optical class, temporally redundant frames are evenly
subsampled to a maximum of 6, 12, or 20. A run with fewer trustworthy frames is
not synthetically duplicated. Instead, sample weights make every contributing
`(run, class)` block carry equal total influence, and make optical 2% and 3%
carry equal total class influence.

This is stricter than merely counting frames: 120 nearly identical frames from
one long video cannot outweigh six independent frames from another video.

## Data and validation

- trajectory-teacher confidence: 0.75 and 0.85;
- H2-only, RH20, and angle-80 RH20 sources;
- RH30 excluded because it supplied no optical 3% candidates;
- caps: 6, 12, and 20 frames per run/class;
- weighted logistic student;
- complete-video-held-out nested gate selection;
- runtime remains one calibration image plus one measurement image.

## Result

All six balancing variants declined every safe 2%/3% override. Their final
predictions therefore match the four-band baseline:

| Metric | Result |
|---|---:|
| Exact accuracy | 0.6753 |
| Video-macro accuracy | 0.6837 |
| Recall 0–1% | 0.8395 |
| Recall 2% | 0.6735 |
| Recall 3% | 0.3789 |
| Recall 4% | 0.7042 |
| MAE | 0.4457 |

For confidence 0.85 with caps 12/20, one held-out fold proposed three 3-to-2
changes. Those rows were true 4% frames, so MAE increased to 0.4491 and the
candidate was rejected.

At cap 6, the retained examples illustrate the actual limitation. Optical 2%
has six examples in most long runs, while optical 3% is absent in angle-80,
H2-only runs 4/5, and RH20 run 2, and has only one or two examples in several
other runs. Weighting can remove 2% dominance, but cannot create missing 3%
optical domains.

## Decision

Keep the production model unchanged. Run balancing is retained as the correct
training policy, but it does not by itself improve a held-out video.

## Next step

Normalise each run into a calibration-relative optical coordinate before
classification. Instead of comparing absolute Lab differences across lighting
domains, represent the flame response by its direction along the yellow-to-
green trajectory and its magnitude relative to that run's initial dispersion.
Then repeat the same run-balanced, complete-video-held-out test. This remains
automatic and single-frame at application runtime.
