# H2 consensus landmark refinement audit

## Work completed

The eight-run four-band dataset was plotted as calibration-relative flame Δa*
versus time. Representative reference-3 frames predicted as 2 or 3 were also
extracted into a visual review montage.

Training-only optical trimming retained the central 100%, 85%, 70%, or 55% of
each run/class cloud. Held-out rows were never trimmed. A second experiment
moved run4's apparent early plateau from 3 to 4 and removed test3's lighter
late exposure toggles.

## Findings

- run4's nominal 3 interval and final maximum have almost the same flame colour;
- test3's late response alternates between two exposure states inside one label;
- run5_x2 has a clearer 3-to-4 separation than run4;
- the same reference-3 appearance is predicted as 2 in multiple independent
  runs, not only in isolated blurred frames.

| Variant | Video-macro exact | 2 recall | 3 recall | 4 recall | Decision |
|---|---:|---:|---:|---:|---|
| 6-feature broad baseline | 0.676 | 0.624 | 0.361 | 0.720 | superseded |
| 6-feature stable 70% training | 0.685 | 0.532 | 0.408 | 0.747 | reject: sacrifices 2 |
| forced run4/test3 landmark relabel | 0.684 | 0.673 | 0.133 | 0.677 | reject |
| 11-feature broad windows | **0.693** | **0.663** | **0.373** | **0.709** | retain as audit baseline |

The forced plateau correction demonstrates an identifiability problem: run4's
newly labelled 4 frames look like class 3 in other runs. A single calibrated
response delta cannot recover a different label when the optical appearances
overlap. It would be misleading to deploy the 68.5% trimmed result merely
because its aggregate score is higher.

## Next experiment

Add calibration-frame context to every sample: absolute flame Lab/chroma,
substrate/background Lab, and flame-minus-background descriptors. These values
are available after the app's existing calibration photo and can condition the
concentration model on lighting/location without requiring patches in the
measurement photo. Evaluate them with the same complete-video holdout and
retain the model only if it improves the 0.693 video-macro baseline while
protecting both 2% and 4% recall.

Artifacts:

- `training/output/h2_consensus_landmark_v2/run_aligned_flame_a_trajectory.png`
- `training/output/h2_consensus_landmark_v2/three_percent_boundary_review.jpg`
- `training/output/h2_consensus_landmark_v2/metrics.json`
