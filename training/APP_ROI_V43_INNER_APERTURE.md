# App ROI v43 inner-aperture lock

## User test evidence

The v42 cover-aligned overlay exposed a second, real geometry problem.  The
Run-3 test sequence alternated between an inner optical-aperture calibration
radius of 41 px and an outer-flange radius of 58--59 px.  Both circles were
concentric, but the larger radius changed every normalized sensing mask and
made state/concentration results incomparable across RH clips.

## Selection change

For wide uncropped frames:

- compact inner candidates are limited to radius ratio <=0.20 and receive a
  radius prior centred at 0.17 of the short side;
- concentric wide outer candidates at ratio 0.20--0.28 are converted to the
  optical aperture with the measured 0.72 radius ratio;
- implausible lower-frame compact candidates are excluded above normalized
  y=0.38.

Tight crop handling is unchanged.

## Reproducible Run-3 check

Using the user-test calibration times, the selected `(cx, cy, radius)` values
are now:

| Clip | Calibration time | v43 circle |
|---|---:|---|
| RH20 run 3 | 95 s | (311, 61, 43) |
| RH40 run 3 | 133 s | (310, 63, 42) |
| RH60 run 3 | 145 s | (310, 62, 41) |
| RH80 run 3 | 163 s | (310, 62, 42) |

The four radii previously spanned 41--59 px.  They now span 41--43 px with a
stable centre.  The RH-only-equivalent audit still gives the expected Run-3
modal progression 20--30, 20--30, 30--40, 40--50, so the geometry change does
not manufacture a nominal-RH target.
