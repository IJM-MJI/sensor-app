# Place-1 external validation of the Place-2 RH model

## Protocol

The deployed Place-2 seven-range optical model was frozen and applied without
retraining to two independent Place-1 H2O-only rising runs:

- `1_90_H2O_only_2_extract.mp4` (fast)
- `1_90_H2O_only_extract_3min.mp4` plus
  `1_90_H2O_only_extract_extra.mp4` (one continuous long run)

Each run used its own initial low-RH frame as calibration. The scored samples
were the confirmed ramp endpoints from 20--30 through 70--80. The 80--90 band
was excluded because Place 1 did not reliably reach RH90.

## Result

Frozen Place-2 model:

- Exact accuracy: **0.250** (3/12)
- Balanced accuracy over observed ranges: **0.250**
- Within one adjacent range: **0.417**
- Mean absolute error: **15.83 percentage points RH**

Place-1 candidate trained on one run and tested on the other complete run:

- Exact accuracy: **0.333** (4/12)
- Balanced accuracy over observed ranges: **0.333**
- Within one adjacent range: **0.583**
- Mean absolute error: **11.67 percentage points RH**
- Deployment gate: **failed** for both models (`exact >= 0.85` and every
  observed recall `>= 0.85` were required)

The row-normalized confusion matrix is generated in
`training/output/rh_place1_external_validation_v1` with every cell expressed
on a 0--1 scale.

## Interpretation

The Place-2 model systematically underestimates Place-1 intermediate and high
responses. This is not evidence that the droplet has no usable trajectory:
the Euclidean calibrated colour-change magnitude remains strongly ordered with
RH in both Place-1 runs (Spearman 0.886 fast and 0.943 long). Instead, the
response scale differs substantially between the two places and even between
the fast and long Place-1 recordings.

Therefore the Place-2 thresholds must not be widened to absorb Place 1. The
first environment-specific Place-1 candidate was tested by holding out a
complete run, but improved exact accuracy only from 0.250 to 0.333. The app
must keep Place 1 outside the quantitative scope rather than presenting an
unsupported number. A deployable Place-1 profile needs at least one additional
independent rising run under each lighting style, or a more stable acquisition
condition that reduces the fast/long scale difference.
