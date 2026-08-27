# H2 automatic environment router v2

## Decision

The app may automatically route calibration frames between H2 environment A and B.
The accepted router uses the full calibration-frame HSV distribution. It does not ask
the user to choose an environment and it does not use the later response frame.

The calibration-locked flame mask alone was rejected. Its complete-video-held-out
recall was 0.75 for A and 0.50 for B, below the predeclared 0.85 minimum.

## Accepted validation

- Protocol: eight initial/calibration frames per run, with each complete video held
  out in turn.
- Inputs: 10th, 25th, 50th, 75th and 90th percentiles, mean and standard deviation
  of all three HSV channels inside the central 84% of the app-sized frame.
- Model: standardized RBF SVM.
- A recall: 32/32 = 1.000.
- B recall: 16/16 = 1.000.
- Overall: 48/48 = 1.000.
- Full-fit JavaScript export check: 1.000 training-domain agreement; minimum absolute
  decision margin 0.927.

## Data assignment

| Environment | Independent videos used |
|---|---|
| A | `H2_only_test_2`, `H2_only_test_3`, `H2_only_test`, `RH20_run2_x2` |
| B | `RH20_run3_x2`, `RH20_run4_x2` |
| C | Not deployed: only one physical run (`run5 normal/x2` are duplicates) |

Only initial low-response intervals were used. Response concentration labels and the
user-reviewed 1–2/2–3 boundaries were not used to choose the environment.

## App integration

`sensor-h2-environment-router.js` contains the generated scaler and SVM. During
calibration, `index.html` extracts the same HSV summary, selects A or B, and locks the
result until Reset/Re-calibrate. The selected environment and decision score are shown
in the diagnostic text. This change does not yet replace the single H2 concentration
model; it supplies the validated routing prerequisite for the next model export.

## Remaining work

1. Train/export app-domain A- and B-specific H2 concentration models using the agreed
   ranges `0`, `1–2`, `2–3`, `4` (B currently has no independently demonstrated 4%).
2. Evaluate the complete chain: calibration routing plus the selected family model,
   holding out a complete video at both stages.
3. Accept deployment only if routed end-to-end accuracy improves without reducing the
   existing 2–3 recall. Keep C experimental until another independent C run exists.
