# User-reviewed optical-stage single-frame A/B

## Label policy

This audit uses the sensor's visually reviewed optical response stages as the
reference. The commanded chamber timeline is not used as concentration
supervision, and every measurement is one decoded frame.

The independent `1_90_H2O_only` daylight recording was reviewed as:

| Time | Optical stage |
|---:|---|
| 4 s | 30–40 |
| 7, 9, 13 s | 40–50 |
| 17 s | 50–60 |
| 21 s | 60–70 |
| 25, 27 s | 70–80 |
| calibration | 20–30 |

The `response3` and `response6` representatives were visually confirmed to
show the same stage at their corresponding reviewed times. H2O-only and
H2-only are treated as following the same colour path; illumination and
environment remain nuisance variables that must be normalized.

## Disputed 40–50 anchor

Two one-frame anchor policies were compared while leaving every other band
unchanged:

- current: response3 10.0 s, response6 14.5 s;
- proposed: response3 7.0 s, response6 13.0 s.

On one representative frame per band in each response run, standardized 1-NN
complete-run holdout changed as follows:

| Policy | Exact | Balanced | Within one | MAE (%RH) | response3 fold | response6 fold |
|---|---:|---:|---:|---:|---:|---:|
| current | 0.500 | 0.500 | 0.857 | 6.43 | 0.571 | 0.429 |
| proposed | **0.571** | **0.571** | **0.929** | **5.00** | 0.571 | **0.571** |

The standardized distance between the 40–50 and 50–60 representatives was:

| Policy | response3 | response6 |
|---|---:|---:|
| current | **0.804** | 0.535 |
| proposed | 0.680 | **0.822** |

The proposed times improve response6 separation and the aggregate held-out
score, but reduce response3 separation. More importantly, 50–60 recall remains
zero for the 1-NN cross-run comparison under both policies. A timing-only app
change is therefore not sufficient and is not deployed from this audit.

## Daylight diagnostic and extractor mismatch

The legacy offline `registered_drop_v2` feature path predicted none of the
eight newly reviewed daylight frames correctly under either timing policy. Its
predictions ran from 80–90 at 4–7 s down to 20–30 at 21–27 s, opposite the
reviewed optical-stage order.

This is not accepted as an app accuracy result. The legacy offline extractor
does not reproduce the scale-aware v41 browser ROI path on these frames. The
conflict between the visible review and extracted trajectory establishes that
the next comparison must use the exact v41 browser features before changing
the concentration anchors.

## Decision

1. Keep seven 10%-wide optical stages.
2. Keep single-frame inference; do not require adjacent-frame median input.
3. Preserve both current and proposed 40–50 times as A/B candidates.
4. Do not change the deployed anchor from timing evidence alone.
5. Replace the absolute `top_a` profile gate and validate a common,
   calibration-relative colour-path coordinate using exact v41 features.

Reproduce the audit with:

```powershell
.\.venv\Scripts\python.exe -X utf8 training\rh_user_optical_single_frame_ab.py
```
