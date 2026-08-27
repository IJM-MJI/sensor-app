# Place-2 middle-RH recovery v20

## Validated app frames

The v19 retest produced the following results:

| Video | Time | Result | Assessment |
|---|---:|---|---|
| `1_90_H2O_only_3(response).mp4` | 2.5 s | Initial, H2 0%, RH 20-30% | acceptable low optical response |
| `1_90_H2O_only_3(response).mp4` | 3.0 s | H2O-only, H2 0%, RH 40-50% | correct transition result |
| `1_90_H2O_only_3(response).mp4` | 5.0 s | Uncertain | false rejection |
| `1_90_H2O_only_6(response).mp4` | 13.0 s | H2O-only, H2 0%, RH 40-50% | correct independent run |

The 5 s frame is the only remaining error. It has `H2 expert=0.04`,
`raw H2=0.04`, and calibration-relative flame shift `0.79`. This combination
contains essentially no H2 evidence but is optically separated from the 2 and
2.5 s low-response frames.

## Narrow recovery guard

An undecided frame is recovered as H2O-only 40-50% only when all conditions
hold:

- calibration `top_a >= 127`;
- the endpoint model still reports 20-30%;
- H2 expert probability is below 0.10;
- raw H2 lies in `[-0.2, 0.5)`;
- calibration-relative flame colour shift is at least 0.55.

The 2.5 s frame has H2 expert 0.20 and flame shift 0.20, so it remains the
honest Initial / Low Response result. The 3 s frame is already accepted by the
v19 rule. All validated H2-only frames have `top_a <= 125.5` and cannot enter
this guard.

## Required retest

- response3: 2.5, 3, and 5 s;
- response6: 13 s;
- H2 regression anchors: `H2_only_test_2` 30 s and `RH20_4` 79 s.
