# H2 spatial flame audit

## Question

Can calibration-relative colour from the top, bottom, left, and right portions of
the fixed flame mask improve the user-reviewed H2 bands (`0`, `1-2`, `2-3`, `4`)?

## Data and protocol

- Family A: `test_2`, `test_3`, `test`, `run2`
- Family B: `run3`, `run4`
- User-reviewed corrections are applied before evaluation.
- Every reported sample is predicted by a model that did not train on any frame
  from that sample's video (complete-video-held-out validation).
- Features are differences from each video's calibration frame. Each fixed flame
  mask is summarized globally and in top/bottom/left/right regions using Lab mean,
  median, and quartiles.

## Results

| Family / model | Accuracy | 0 recall | 1-2 recall | 2-3 recall | 4 recall | Decision |
|---|---:|---:|---:|---:|---:|---|
| A existing reviewed model | 86.8% | 97.2% | 92.0% | 79.5% | 98.5% | Retain |
| A spatial candidate | 84.8% | 83.3% | 58.7% | 90.5% | 97.0% | Reject |
| B existing reviewed model | 73.6% | 97.1% | 72.0% | 66.0% | n/a | Baseline |
| B spatial-only candidate | 77.0% | 94.1% | 62.0% | 78.7% | n/a | Reject alone |
| B conservative spatial gate | 74.7% | 97.1% | 76.0% | 66.0% | n/a | Retain as candidate |

The family-B gate only changes an existing `2-3` prediction to `1-2` when the
spatial model supports `1-2` with probability at least 0.50. It changed two
held-out frames; both corrections were right. The reverse promotion is disabled
because it reduced `1-2` recall.

## Conclusion

Spatial subdivision is not a general replacement for the current colour model.
It is useful as a conservative family-B correction, but family B still does not
meet the target of 0.85 recall for every band. The limiting evidence is the
cross-run mismatch: when `run3` is held out, its reviewed `2-3` frames are still
predicted as `1-2`. More threshold tuning on these same two runs would overfit.

## Next experiment

Use family C (`run5_normal` and `run5_x2`) only as one physical-run consistency
check, not as two independent held-out runs. Then test whether its trajectory can
anchor the 1-2/2-3 direction in B without training and testing on duplicate frames.
