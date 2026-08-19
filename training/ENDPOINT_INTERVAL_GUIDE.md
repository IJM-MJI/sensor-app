# Endpoint-exact and ramp-interval concentration labels

## Label policy

The supplied H2 and RH timelines specify the concentration reached at each
interval end. `training/endpoint_interval_analysis.py` therefore rebuilds the
quantitative dataset as follows:

- the final 0.55 s before each stated endpoint is an exact concentration;
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
| H2 old linear-ramp labels | 55.8% | 47.2% | 92.6% | 0.52%p |
| H2 endpoint exact multinomial | 58.9% | 47.8% | 88.4% | 0.58%p |
| H2 interval-censored ordinal | 66.3% | 46.8% | 86.3% | 0.51%p |
| RH old linear-ramp labels | 30.2% | 25.8% | 68.1% | 12.63%p |
| RH endpoint exact multinomial | 56.9% | 40.2% | 75.9% | 8.53%p |
| RH interval-censored ordinal | 50.9% | 32.1% | 65.5% | 12.28%p |

The H2 endpoint policy improves exact accuracy only slightly when sparse stages
are weighted equally. RH improves substantially, confirming that treating ramp
interiors as exact linear concentrations was a major RH label error. However,
the middle-stage recalls remain low: H2 2/3% are 20/30%, and RH 50/60% are 0%
on unseen runs. The endpoint relabelling is correct, but run-to-run colour-path
normalization is still required before replacing the browser model.

Generated local artifacts:

- `training/output/endpoint_interval_registered_v2/endpoint_interval_dataset.csv`
- `training/output/endpoint_interval_registered_v2/metrics.json`
- `training/output/endpoint_interval_registered_v2/predictions.csv`
- `training/output/endpoint_interval_registered_v2/endpoint_interval_validation.png/.pdf/.svg`

## Reproduce

```powershell
.venv\Scripts\python training\endpoint_interval_analysis.py `
  --cache training\cache\v7-verified-orientation-recovery-tail\features_registered_drop_v2.csv `
  --output training\output\endpoint_interval_registered_v2
```

