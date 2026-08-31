# Place-2 RH immediate micro-burst validation

## Purpose

The stable profile model was internally consistent with a one-second median,
but its single-photo accuracy was limited by isolated frame outliers. This test
compares one frame with three and five immediately adjacent video frames.

Both source videos are 30 fps:

- three frames span approximately 66.7 ms
- five frames span approximately 133.3 ms

The same eight non-neighbour validation points from the profile-model audit
were used. Stable training prototypes remain one-second medians; only the
measurement input changes.

## Results

| Input | Exact accuracy | Outer-band accuracy | Within one range | MAE (%RH) |
|---|---:|---:|---:|---:|
| single frame | 0.750 | 0.800 | 0.875 | 6.25 |
| 3-frame median | **1.000** | **1.000** | **1.000** | **0.00** |
| 5-frame median | **1.000** | **1.000** | **1.000** | **0.00** |

The three-frame median recovers both former response3 errors, including the
26.5 s frame that previously jumped from 60–70% back to 20–30%. Five frames
provide no additional benefit on this validation set.

## App implementation

The camera shutter now captures three newly presented video frames and writes
their per-pixel RGB median to the existing analysis canvas. At 30 fps this
takes about 67 ms. `requestVideoFrameCallback` is used when available so the
three samples are distinct camera frames; the fallback waits 34 ms between
frames.

This applies to live camera calibration and measurement. `Load saved frame`
still evaluates the selected image once because a single image contains no
adjacent frames.

## Limitations and next gate

The offline validation uses adjacent frames from monitor/video recordings, not
a live handheld phone burst. The next gate is therefore an app test using the
camera shutter on response3 and response6 playback. The result must confirm
that the ~67 ms burst does not cause visible ROI displacement and preserves
the H2-only results before it is considered final.

