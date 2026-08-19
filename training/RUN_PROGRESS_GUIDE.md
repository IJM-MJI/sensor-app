# Run-normalized concentration progress analysis

## Question

Can run-to-run colour differences be removed well enough to approach 0.85
recall at every H2/RH stage? `run_progress_analysis.py` compares two calibration
schemes under a complete independent-run holdout.

## Data used

| Task | Independent groups | Feature rows | Exact evaluation | Interval rows | Direct signal |
|---|---:|---:|---:|---:|---|
| H2-only | 5 | 1,514 | 77 | 932 | registered flame LAB/chroma distribution |
| H2O-only | 5 | 1,154 | 84 | 958 | calibration-registered droplet LAB |

H2 uses `H2_only_test`, `test_2`, `test_3`, `H2_only_4`, and `H2_only_5`.
RH uses daylight recovery, indoor fast, indoor long (the 3-minute and extra
clips are one group), response-3, and response-6. Simultaneous clips and H2
recovery do not enter concentration training or evaluation.

- Exact endpoint/hold rows directly train and score the exact-stage candidate.
- Ramp interval rows train only the censored ordinal candidate through one-sided
  threshold constraints. They are never assigned a made-up midpoint label.
- The selected one-anchor models happened to be the exact-stage candidates for
  both tasks; the interval candidates remain in `metrics.json` and were rejected
  by nested held-out selection.
- All feature scaling, model choice, C, and interval weight are selected inside
  the outer training runs. No measurement frame from the held-out run trains the
  model.

## Calibration schemes

**Initial anchor only (application-compatible):** subtract the held-out run's
known H2 0% / RH20–30 calibration colour, then retain the complete multichannel
shape colour vector.

**Initial + high anchor (diagnostic):** additionally use the held-out run's known
highest endpoint and project colour onto the low-to-high axis. This requires an
extra known-concentration calibration and is not the current application flow.

## Result

| Task/model | Exact | Stage-balanced | Within one stage | MAE |
|---|---:|---:|---:|---:|
| H2 endpoint baseline | 62.3% | 52.4% | 93.5% | 0.44%p |
| H2 Initial anchor | 71.4% | 50.4% | 87.0% | 0.45%p |
| H2 Initial + high | 64.9% | 42.4% | 89.6% | 0.55%p |
| RH endpoint exact | 69.0% | 41.4% | 81.0% | 6.55%p |
| RH Initial anchor | 67.9% | 46.6% | 85.7% | 6.07%p |
| RH Initial + high | 44.0% | 20.4% | 64.3% | 12.32%p |

Initial centering is useful, especially for RH stage balance and gross error.
The high anchor is not sufficient and often hurts: the trajectory figure shows
that intermediate colours do not lie on a common straight low-to-high path.
For example, H2 daylight-5 has a very large 1% projection followed by a lower
2% projection, and several RH runs flatten or reverse around 40–60%.

Per-stage recall with the application-compatible Initial anchor is:

- H2: `0=0.90, 1=0.60, 2=0.20, 3=0.20, 4=0.62`
- RH: `20–30=0.88, 40=0.40, 50=0.20, 60=0.20, 70=0.75, 80=0.20, 90=0.64`

Therefore the current data do not support a defensible 0.85 recall claim for
every stage. A second high reference photograph alone does not solve the issue.
The next informative data are stable middle-stage holds from independent runs,
especially H2 2/3% and RH 40/50/60/80%, with several seconds after the chamber
has reached each target.

## Visual outputs

- `training/output/run_progress_v3/run_progress_validation.png/.pdf/.svg`
- `training/output/run_progress_v3/run_progress_trajectories.png/.pdf/.svg`
- `training/output/run_progress_v3/metrics.json`
- `training/output/run_progress_v3/predictions.csv`
- `training/output/run_progress_v3/normalized_trajectories.csv`

```powershell
.venv\Scripts\python training\run_progress_analysis.py `
  --cache training\cache\v7-verified-orientation-recovery-tail\features_registered_drop_v2.csv `
  --output training\output\run_progress_v3
```

