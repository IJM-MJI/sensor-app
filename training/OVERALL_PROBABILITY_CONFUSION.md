# Overall probability confusion matrices

All displayed cells are row-normalized probabilities in the interval 0--1:

`P(predicted class | reference class)`

Every reference row therefore sums to 1.00. Raw observation counts remain in
the JSON and CSV outputs for reproducibility.

Four matrices are reported separately because they have different classes and
validation protocols:

1. direct four-state classification;
2. H2 concentration, environment A;
3. H2 concentration, environment B;
4. RH range quantitation.

Recent app-domain single-frame guards are not inserted into the held-out
matrices. They are smoke-test evidence, not independent held-out predictions.
Likewise, the post-reviewed transition-safe RH matrix is excluded from the
primary figure; nominal endpoints remain the defensible RH result.

Reproduce with:

```powershell
.venv\Scripts\python.exe training\overall_probability_confusion.py
```

Outputs are written to `training/output/overall_probability_confusion_v1/`.

## Current results

| Task | Exact accuracy | Balanced accuracy | Minimum class recall | Samples |
|---|---:|---:|---:|---:|
| Four-state | 0.730 | 0.704 | 0.427 | 881 |
| H2 environment A | 0.878 | 0.787 | 0.462 | 393 |
| H2 environment B | 0.781 | 0.812 | 0.700 | 201 |
| RH range | 0.875 | 0.875 | 0.750 | 16 |

### Four-state

| Reference / prediction | Initial | H2 only | RH only | Simultaneous |
|---|---:|---:|---:|---:|
| Initial | 0.792 | 0.056 | 0.090 | 0.062 |
| H2 only | 0.112 | 0.763 | 0.056 | 0.068 |
| RH only | 0.017 | 0.000 | 0.834 | 0.148 |
| Simultaneous | 0.287 | 0.189 | 0.098 | 0.427 |

### H2 environment A

| Reference / prediction | 0 | 1-2 | 2-3 | 4 |
|---|---:|---:|---:|---:|
| 0 | 0.462 | 0.256 | 0.051 | 0.231 |
| 1-2 | 0.000 | 0.769 | 0.231 | 0.000 |
| 2-3 | 0.000 | 0.000 | 0.976 | 0.024 |
| 4 | 0.000 | 0.000 | 0.061 | 0.939 |

### H2 environment B

| Reference / prediction | 0 | 1-2 | 2-3 |
|---|---:|---:|---:|
| 0 | 0.935 | 0.065 | 0.000 |
| 1-2 | 0.000 | 0.700 | 0.300 |
| 2-3 | 0.000 | 0.200 | 0.800 |

### RH range

| Reference / prediction | 20-30 | 40-50 | 60-70 | 80-90 |
|---|---:|---:|---:|---:|
| 20-30 | 1.000 | 0.000 | 0.000 | 0.000 |
| 40-50 | 0.000 | 0.750 | 0.250 | 0.000 |
| 60-70 | 0.000 | 0.250 | 0.750 | 0.000 |
| 80-90 | 0.000 | 0.000 | 0.000 | 1.000 |

The principal weaknesses are simultaneous-state recall (0.427), environment-A
H2 0 recall (0.462), and separation of adjacent middle RH ranges (0.750 each).
The RH result is based on only 16 audited endpoints and therefore has much
wider uncertainty than the frame-level H2 and state evaluations.

The environment-A result corrects the x2 timing of `RH20_2_x2`: its initial
0% interval is 0--5 s, not 10--14 s. Treating this visibly mismatched run as a
separate robustness set raises primary-set exact accuracy from 0.878 to 0.932
and H2 0 recall from 0.462 to 0.565, but exclusion alone does not solve 0%
recognition. A calibration-relative zero gate is still required.
