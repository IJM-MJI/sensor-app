# H2 family-B confirmed alignment

The user confirmed `run4` 50 s, 55 s, and 56 s as `2-3`. The final audit uses:

- `run3`: `1-2` through 54.5 s; `2-3` from 55 s.
- `run4`: newly extracted 30-49.5 s as `1-2`; `2-3` from 50 s.
- Complete-video-held-out validation.
- Balanced stable-frame selection per run and class.

## Result

| Version | Accuracy | 0 recall | 1-2 recall | 2-3 recall | Minimum recall |
|---|---:|---:|---:|---:|---:|
| Previous boundary | 73.6% | 97.1% | 72.0% | 66.0% | 66.0% |
| Confirmed boundary with added ramp frames | 80.7% | 97.1% | 78.5% | 77.4% | 77.4% |

The earlier provisional 92.1% score is rejected because advancing the `run4`
boundary without extracting earlier `1-2` frames left only two held-out `1-2`
samples. Restoring balanced support gives the defensible 80.7% result above.

The remaining failure is concentrated in the weak `run3` 55-59 s interval:
when the model trains only on `run4`, all eleven of these confirmed `2-3`
samples are predicted as `1-2`. Further time-label movement would contradict
the reviewed frames. The next valid test is response-amplitude normalization,
followed by family-C consistency validation.
