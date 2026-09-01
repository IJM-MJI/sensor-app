# App ROI v40 crop/uncropped audit

## Failure reproduced

The v39 calibration detector selected the flame-centred circle in
`app_test/1_90_H2O_only_calibrate.png`. At the app's 480 px processing scale,
that false candidate was approximately `(244,149,r=102)`: its compact warm
colour coverage was `0.485`, so it outscored the full chamber aperture. The
resulting normalized circle was then correctly—but undesirably—locked for all
subsequent saved frames.

## v40 selection policy

1. A candidate receives additional support only when warm sensor pixels occur
   in both its upper flame half and lower droplet half.
2. A near-square tight chamber crop must contain a large central metal-ring
   candidate. Its noisy outer-ring geometry is regularized to the optical
   aperture at approximately `(0.50w, 0.52h, 0.40 × short-side)`.
3. Calibration geometry remains locked for aligned frames.
4. If a later image's aspect ratio differs by more than about 8%, the locked
   circle is not reused and the ROI is detected again for the new crop.

## Geometry audit

`audit_app_roi_v40.py` mirrors the browser selector and writes the candidate
sheet and CSV to `training/output/app_roi_v40_audit/`.

- `app_test` tight-crop flow: calibration 1 + measurement 8 = **9/9 pass**
- video calibration frames: cropped/uncropped daylight, response3, response6
  = **6/6 pass**
- total geometry cases: **15/15 pass**

At 480 px processing scale the tight calibration selected `(240,213,r=164)`.
The eight differently sized still images then reused its normalized geometry,
producing aperture radii `160–176 px` rather than the old flame-only radius.

The six direct-video selections were:

| Input | Crop | Selected circle |
|---|---|---|
| daylight RH20 calibration | cropped | `(241,133,r=87)` |
| daylight RH20 calibration | uncropped | `(321,77,r=57)` |
| response3 calibration | cropped | `(229,85,r=88)` |
| response3 calibration | uncropped | `(135,202,r=75)` |
| response6 calibration | cropped | `(245,103,r=84)` |
| response6 calibration | uncropped | `(169,289,r=60)` |

The local browser loaded the v40 app with no JavaScript console errors. This
audit establishes ROI geometry robustness, not RH/H2 concentration accuracy;
the latter must still be evaluated with the existing held-out protocols.

## Reproduce

```powershell
.venv\Scripts\python training\audit_app_roi_v40.py
```
