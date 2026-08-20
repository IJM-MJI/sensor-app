# H2 endpoint training A/B

This experiment keeps all five H2-only recordings in held-out evaluation and
changes training weights only. It therefore tests generalization rather than
inflating accuracy by dropping difficult validation recordings.

| Training policy | Exact | Balanced | Within ±1 | MAE |
|---|---:|---:|---:|---:|
| Reviewed selective 0/1/3 baseline | 0.521 | 0.528 | 0.948 | 0.532 |
| Confidence hybrid baseline | 0.519 | **0.549** | **0.974** | **0.508** |
| Suppress run 5 endpoints + run 4 4% | 0.477 | 0.468 | 0.938 | 0.587 |
| Suppress all run 5 + run 4 4% | 0.451 | 0.436 | 0.925 | 0.626 |

Per-stage held-out recall for the endpoint-only suppression was 0.408, 0.435,
0.610, 0.534, and 0.354 for H2 0--4%, respectively. Suppressing the whole of
run 5 reduced it further to 0.382, 0.337, 0.616, 0.492, and 0.354.

Conclusion: the endpoint recordings are inconsistent, but simply excluding
their labels removes useful illumination/domain coverage. Keep the current
confidence hybrid as the best evaluated model. The next experiment should use
run-conditional normalization or an endpoint-quality mixture, retaining each
run as a domain while rejecting only frame-level optical outliers.

