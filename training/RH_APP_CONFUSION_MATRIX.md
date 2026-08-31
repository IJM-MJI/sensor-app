# Place-2 RH app-domain confusion matrix

The current audit uses all seven displayed RH ranges: 20–30, 30–40, 40–50,
50–60, 60–70, 70–80, and 80–90% RH.

## Confirmed result

- Response3 before the v37 ordinal correction: **2/7 (28.6%)**
- Response3 after v37: **7/7**
- Final explicitly confirmed app anchors: **13/13**
- H2O-only state at all final anchors: correct

## Frozen-v37 unused-time validation

- New points: **9/14 (64.3%)**
- All 27 app observations: **22/27 (81.5%)**
- Uncertain rate in the new points: **0/14**
- All five errors were in an adjacent RH range
- New-point within-one-range accuracy: **14/14**

The plotted confusion matrices are row-normalized and display values from 0 to
1, as requested. The primary current estimate is the 14-point unused-time
result; the 27-point total includes tuned anchors and is therefore optimistic.

The 13/13 result is a **tuned deployment-anchor confirmation**. Response3 and
response6 supplied the prototypes and threshold evidence, so this value is not
independent accuracy and must not be presented as held-out performance.

## Next validation gate

Use time blocks that were not used to choose thresholds, or preferably record a
new independent Place-2 H2O-only run. Report that result as the main confusion
matrix; the current figure is suitable only as a calibration/deployment
consistency figure.

Reproduce with:

```powershell
.venv\Scripts\python training\rh_app_confusion_matrix.py
```

Outputs are written to `training/output/rh_app_confusion_v2/` as audited CSV,
JSON, PNG, PDF, and SVG files.
