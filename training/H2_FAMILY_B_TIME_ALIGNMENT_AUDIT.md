# H2 family-B time-alignment audit

## Hypothesis

`run4` responds faster/more strongly than `run3`, so identical nominal ramp
timelines should not be forced to share the same optical transition time.

## User constraints

- `run3` 55 s and 59 s remain `2-3`.
- `run4` 65 s, 84 s, and 90 s remain `2-3`.
- Candidate boundaries were restricted to 50-55 s for `run3` and 55-65 s for
  `run4`.
- Evaluation holds out each complete video.

## Result

| Labelling | Accuracy | 0 recall | 1-2 recall | 2-3 recall | Minimum recall |
|---|---:|---:|---:|---:|---:|
| Previous: run3 50 s, run4 65 s | 73.6% | 97.1% | 72.0% | 66.0% | 66.0% |
| Best sweep: run3 55 s, run4 56 s | 92.1% | 97.1% | 92.7% | 90.3% | 90.3% |

The direction agrees with the hypothesis: delay the weaker `run3` boundary and
advance the stronger `run4` boundary. However, the 56 s `run4` boundary leaves
only two held-out `run4` frames in the `1-2` reference class. Therefore 92.1%
is a promising provisional result, not yet an application-ready estimate.

## Decision

Do not deploy this boundary until the 55-60 s `run4` frames are optically
reviewed. If they are confirmed as `2-3`, use the aligned labels and validate
against family C as one physical run. If they remain `1-2`, reject the optimum
and use the best boundary consistent with that review.
