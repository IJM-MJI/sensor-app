# RH app-domain confusion matrix

This audit evaluates the four RH range outputs (20--30, 40--50, 60--70, and
80--90% RH) from response3 and response6 app screenshots. It does **not** score
the independent four-state classifier.

## Primary: nominal endpoints

| Reference \ Predicted | 20--30 | 40--50 | 60--70 | 80--90 |
|---|---:|---:|---:|---:|
| 20--30 | 4 | 0 | 0 | 0 |
| 40--50 | 0 | 3 | 1 | 0 |
| 60--70 | 0 | 1 | 3 | 0 |
| 80--90 | 0 | 0 | 0 | 4 |

- Exact and balanced accuracy: **14/16 = 87.5%**
- Recall by range: **100%, 75%, 75%, 100%**
- Within one adjacent range: **100%**
- Midpoint MAE: **2.5% RH**
- Errors: response3 7 s (40--50 -> 60--70) and response6 16 s
  (60--70 -> 40--50)

This is the defensible primary result because it retains the supplied nominal
endpoint protocol.

## Secondary: transition-safe optical anchors

Replacing response3 7 s with 6.5 s, response6 16 s with 17 s, and response6
20 s with 19 s gives a 16/16 diagonal matrix. This is useful evidence that the
four optical ranges are separable when reaction delay/boundary timing is
handled, but those anchors were selected after inspecting the trajectories.
The resulting 100% is therefore diagnostic and must not be presented as an
independent test accuracy.

Reproduce with:

```powershell
.venv\Scripts\python training\rh_app_confusion_matrix.py
```

Outputs are written to `training/output/rh_app_confusion_v1/` as audited CSV,
JSON, PNG, PDF, and SVG files.
