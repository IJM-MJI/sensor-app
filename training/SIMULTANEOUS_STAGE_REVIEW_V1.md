# Simultaneous optical-stage review v1

## Purpose

The app already recognizes the simultaneous state, but its H2 and RH numbers
remain hidden because the single-condition regressions do not transfer through
cross-interference.  Existing recordings are sufficient for a first correction
pass; no new capture is required.  What is missing is optical-stage review of
the simultaneous reactions.

## Review artifact

Run:

```powershell
.\.venv\Scripts\python.exe -X utf8 training\make_simultaneous_stage_review.py `
  --video-root "C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"
```

The command creates four run-wise sheets under
`training/output/simultaneous_stage_review/`.  Every sheet has seven rows
(nominal RH20--80) and nine search frames across the known reaction window.
The column percentage is only a search position and is never used as an H2
label.  `candidate_index.csv` contains the exact video times, the uncorrected
H2-only regression estimate, and its nearest 0--4% stage.  That estimate is a
comparison aid, not simultaneous ground truth.

Run 5 analysis opens the RH-labelled files `1_90_RH20_5_x2.mp4` through
`1_90_RH80_5_x2.mp4` directly.  They were derived from `1_90_5.MOV`, but the
full source is not used in place of the clips.  The `_x2` suffix denotes
playback/extraction speed; the times printed on the sheet are clip-local times.

The similarly named files without a repeat suffix (`1_90_RH20.mp4` through
`1_90_RH90.mp4`) are not an independent run 1.  Duration and frame comparison
show that they are 1x-speed duplicates of run 5, so including both versions
would leak identical frames across training and validation.

For each row, identify the frames whose **flame** matches the reviewed H2-only
0%, 1%, 2%, 3%, and 4% optical stages.  The same frame's **droplet** is then
assigned one of the seven reviewed H2O-only optical RH bands.  Nominal chamber
RH remains metadata and does not replace that droplet review.

All review images are decoded from the uncropped RH-labelled videos.  Chamber
and card localization must therefore use the same scale-aware ROI principle as
the deployed v41 app; crop coordinates are not training inputs.

## Training gate

After review, train a two-output correction with run-wise holdout:

1. H2 stage from flame features conditioned on droplet features.
2. Seven-band RH stage from droplet features conditioned on flame features.
3. Hold out an entire simultaneous run at a time.
4. Keep the app's simultaneous numbers hidden unless overall exact accuracy and
   every reported band's recall are at least 0.85.  Adjacent-band accuracy is
   diagnostic only and cannot satisfy the gate.

This preserves single-frame inference and avoids treating elapsed time or the
nominal simultaneous RH setpoint as optical ground truth.

## H2-only transfer probe

Two H2-only comparisons were run on the Run-5 candidates:

- Direct application of the existing H2-only regression is invalid.  The first
  simultaneous frames already read roughly 1--2%, and several rows decrease to
  0--1% as the reaction advances.
- Per-video endpoint normalization followed by nearest-stage matching to
  `1_90_H2_only_5.mp4` is also invalid.  Its predicted stages jump backward and
  skip levels.  Even the H2-only reference projection is slightly non-monotonic:
  the 3% anchor projects to 1.09 while the 4% endpoint is 1.00.

The reproducible output is
`training/output/simultaneous_stage_review/h2_only_stage_probe.csv`.  These
negative results show that the old whole-flame LAB summary cannot determine
simultaneous H2 time anchors.  Human-reviewed flame-stage anchors or a more
spatially selective flame feature are required before quantitative deployment.
