# Endpoint-exact and ramp-interval concentration labels

## Label policy

The supplied H2 and RH timelines specify the concentration reached at each
interval end. `training/endpoint_interval_analysis.py` therefore rebuilds the
quantitative dataset as follows:

- the single sampled frame nearest each stated endpoint is an exact concentration
  (0.55 s is only the decoding/search tolerance, not an exact-label window);
- a repeated-value segment is an exact hold;
- a ramp interior stores only its lower and upper possible stage;
- H2 recovery is excluded rather than being relabelled as immediate 0%;
- RH 20% and 30% map to the single application class `20–30%`;
- long baseline, hold, and ramp segments are capped per run/stage so video
  duration cannot dominate training or evaluation;
- all hyperparameters are selected inside the outer held-out-run split.

The generated `endpoint_interval_dataset.csv` is the auditable label table. An
empty `exact` field means the row is interval-censored and may train ordering
constraints but must not be scored as an exact concentration.

## Registered-ROI result

Evaluation uses the calibration-registered RH droplet cache and holds a complete
independent run out at a time. The old linear-ramp model is rescored on the exact
same endpoint frames.

| Task/model | Exact | Stage-balanced | Within one stage | MAE |
|---|---:|---:|---:|---:|
| H2 old linear-ramp labels | 62.3% | 52.4% | 93.5% | 0.44%p |
| H2 endpoint exact multinomial | 64.9% | 46.6% | 92.2% | 0.47%p |
| H2 interval-censored ordinal | 77.9% | 50.3% | 94.8% | 0.29%p |
| RH old linear-ramp labels | 33.3% | 26.2% | 70.2% | 12.38%p |
| RH endpoint exact multinomial | 69.0% | 41.4% | 81.0% | 6.55%p |
| RH interval-censored ordinal | 57.1% | 27.8% | 79.8% | 9.58%p |

The strict endpoint protocol contains 77 H2 and 84 RH evaluation frames. Only
one frame per run is available for most middle-stage endpoints, so each of those
recalls is based on roughly five independent examples. RH improves substantially,
confirming that treating ramp interiors as exact linear concentrations was a
major label error. Exact accuracy alone is inflated by the genuine low/high
holds; stage-balanced accuracy remains the model-selection criterion.

Generated local artifacts:

- `training/output/endpoint_interval_registered_v3/endpoint_interval_dataset.csv`
- `training/output/endpoint_interval_registered_v3/metrics.json`
- `training/output/endpoint_interval_registered_v3/predictions.csv`
- `training/output/endpoint_interval_registered_v3/endpoint_interval_validation.png/.pdf/.svg`

## Reproduce

```powershell
.venv\Scripts\python training\endpoint_interval_analysis.py `
  --cache training\cache\v7-verified-orientation-recovery-tail\features_registered_drop_v2.csv `
  --output training\output\endpoint_interval_registered_v3
```

