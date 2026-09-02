# App ROI v42 display alignment

## Observed failure

On uncropped 16:9 simultaneous frames, the preview element uses
`object-fit: cover` inside a square viewport.  The image is scaled by its height
and cropped on both horizontal sides.  The debug canvas was instead stretched
from the full source dimensions to the square.  Consequently, a correct source
ROI appeared shifted left and horizontally compressed on screen.

This was a display-coordinate defect.  It did not by itself prove that feature
extraction used the wrong source pixels.

## v42 correction

The debug and manual ROI overlays now share one cover transform:

1. `scale = max(displayWidth/sourceWidth, displayHeight/sourceHeight)`
2. center the scaled source with signed x/y offsets
3. map source circle centre and radius through that scale and offset

Manual taps use the inverse transform before storing normalized source
coordinates.  This also fixes manual ROI placement on wide saved frames.

For a representative 480x270 source shown in a square viewport, the old debug
overlay used independent x/y stretching.  v42 uses a single cover scale and a
negative horizontal offset, matching the visible cropped image.

## Retest requirement

After deployment, recalibrate because v41 may have locked an ROI selected while
the misleading overlay was visible.  Confirm that both circles surround the
optical aperture and card.  If the circles align but the state remains H2-only,
that is a state/quantitation error to diagnose separately rather than an ROI
display error.
