# H2 other-run reference audit

## Reference policy

`1_90_H2_only_test_2_more_cropped.mp4` is the strong 0--4% optical reference.
The user and fixed-mask audit agree that `test_3` reaches only a 3%-equivalent
response, despite its nominal controller ramp toward 4%.

The following additional cropped H2-only videos were compared after locking the
flame pixels at the 2 s calibration frame:

- `1_90_H2_only_test_cropped.mp4`
- `1_90_H2_only_4_cropped.mp4`
- `1_90_H2_only_5_cropped.mp4`

Each video used a separate tight card crop and fixed flame mask.  The final mask
review is `training/output/h2_other_run_reference_v2/all_run_fixed_masks.jpg`.

## Test_2-equivalent optical stages

| Run | Selected time | Closest optical stage | Interpretation |
|---|---:|---:|---|
| `test` | 30 s | 2% | clear intermediate response |
| `test` | 100 s | 3% | late response is closer to test_2 3% than 4% |
| run 4 | 90--122 s | 2% | agrees with the earlier 2--3% judgement |
| run 5 | 21--130 s | 2% | agrees with the earlier 2--3% judgement |
| `test_3` | 60--150 s | 3% | does not reach the test_2 4% appearance |

These are optical-equivalent stages, not independent gas-meter measurements.
Cross-lighting distance is larger for `test` than for the tight test_2/test_3
pair, so its exact late label remains lower-confidence.

## Relabelled held-out evaluation

The quantitative analysis was rerun with:

- `test_2` retained as a complete 0--4% reference;
- `test_3` capped at 3%;
- run 4 and run 5 excluded from exact-stage validation;
- no browser model deployment.

| Metric | Result |
|---|---:|
| Complete-video held-out exact | 0.737 |
| Video-macro exact | 0.633 |
| Within one stage | 0.953 |
| MAE | 0.321 stages |
| Stage-balanced accuracy | 0.450 |
| 4% recall | 0/31 = 0.000 |

Correcting `test_3` therefore raises aggregate accuracy substantially, but the
result is not deployable as a complete five-stage model.  Almost all of the
gain comes from no longer teaching the long, weak `test_3` tail as 4%.  When a
true 4% run is held out, the remaining runs predict it as 3%, confirming that
the current data contain only one consistently strong test_2-like 4% response.

## Decision and next experiment

Use `test_3`, run 4, run 5, and the RH20 runs as weak 0--3% augmentation.  Keep
test_2 as the only strong 4% anchor and mark the app's 4% output experimental.
This can improve the 0--3% boundaries, but an independently validated 4% recall
near 0.85 cannot be demonstrated until another run reaches the same optical 4%
state as test_2.

Before deployment, implement the same calibration-locked flame-pixel mask in
the browser extractor and verify a new candidate without sacrificing 3% recall.

## RH20 run 3 maximum check

`1_90_RH20_3_x2_cropped.mp4` was checked separately using its stated 0--60 s
reaction and 60--97 s recovery.  Because reaction begins at the first frame,
the fixed flame mask and colour baseline were taken at 0 s.  The 50--60 s late
reaction remained closest to the test_2 2% optical centroid; interpolation of
the dominant green-channel change places it roughly between 2% and 3%.

| Optical reference/state | Calibration-relative flame Δa* |
|---|---:|
| test_2 2% | -0.94 |
| test_2 3% | -3.97 |
| test_2 4% | -5.15 |
| RH20 run 3, 50--60 s | -1.58 to -2.22 |

The run therefore does **not** supply a test_2-equivalent 4% example.  It is
useful as weak 2--3% augmentation and as a complete reaction/recovery state run,
but it must not supervise the exact 4% class.

## All cropped RH20 runs

The remaining cropped RH20 videos were then audited with the same fixed-flame
method.  Reaction labels use the final ten seconds before the supplied reaction
boundary.  Recovery validation uses only the final two seconds because the
supplied recovery endpoint, rather than its start, is the fully recovered state.

| Run | Reaction window | Late median Δa* | test_2-equivalent stage | Training use |
|---|---:|---:|---:|---|
| run 2 x2 | 50--60 s | -1.51 | 2% | weak 1--2% / ordering |
| run 3 x2 | 50--60 s | -2.09 | 2% (toward 3%) | weak 2--3% |
| run 4 | 110--120 s | -2.88 | 3% | useful weak 3% |
| run 5 x2 | 44--54 s | -3.46 | 3% | strongest additional weak 3% |
| run 5 normal | 80--90 s | -2.42 | 2%, occasionally 3% | weak 2--3% / ordering |

No RH20 run reaches the test_2 4% reference (`Δa* = -5.15`).  Run 5 x2 is the
closest but remains around the test_2 3% reference (`Δa* = -3.97`).

Run 5 x2 and its normal-speed recording return to the test_2-equivalent 0%
state at the recovery endpoint.  Runs 2, 3, and 4 retain a small positive a*
offset and match the 0--1% optical band, which is suitable illumination-domain
augmentation but should not redefine the 1% reaction boundary.

Selective use for the next model is therefore:

- strong exact 4%: test_2 only;
- weak 3%: run 4 and run 5 x2;
- weak 2--3% ordering: run 3 and run 5 normal;
- weak 1--2% ordering: run 2;
- strong experimental-condition 0% anchors: final recovery endpoints of all
  runs, with the optical 0--1% band retained for display uncertainty.
