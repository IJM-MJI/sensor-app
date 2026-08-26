# H2 four-range audit

## Operational output definition

The requested H2 display ranges are:

- `0%`
- `1-2%`
- `2-3%`
- `4%`

The shared value 2 is a reporting boundary, not a duplicated training label.
Internally, former optical-2 anchors map to `1-2%` and former optical-3 anchors
map to `2-3%`.

The exact-zero pool was cleaned before evaluation. Only narrow initial and
full-recovery endpoints were retained; early ramp frames formerly included in
the `0-1%` class were removed.

## Complete-video held-out result

The most recall-balanced candidate was an RBF SVM:

| Metric | Result |
|---|---:|
| Rows | 763 |
| Frame accuracy | 59.90% |
| Video-macro accuracy | 60.31% |
| Recall, 0% | 96.00% |
| Recall, 1-2% | 62.93% |
| Recall, 2-3% | 45.56% |
| Recall, 4% | 53.63% |

Shrinkage LDA had higher overall accuracy (67.76%) but only 37.28% recall for
`2-3%`. Neither candidate is suitable for deployment.

## Hierarchical check

A two-stage structure was also tested:

1. classify exact zero versus any H2 response;
2. for response frames only, classify `1-2 / 2-3 / 4`.

This did not improve the limiting response-range recall. Exact zero is already
well separated; the remaining problem is inconsistent cross-video placement of
the three response anchors.

## Decision and next step

Keep the four requested display ranges, but do not export this model to the app.
The next analysis must rebuild response anchors by calibrated optical order
within each run, then match equivalent response strength across runs. In
particular, questionable 4% tails must not define the 4% class merely because
of their timeline label.

Artifacts:

- `training/h2_four_range_analysis.py`
- `training/output/h2_four_range_v1/metrics.json`
- `training/output/h2_four_range_v1/four_range_confusion.png`
