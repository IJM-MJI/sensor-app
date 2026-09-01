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
label.  `candidate_index.csv` contains the exact video times.

For each row, identify the frames whose **flame** matches the reviewed H2-only
0%, 1%, 2%, 3%, and 4% optical stages.  The same frame's **droplet** is then
assigned one of the seven reviewed H2O-only optical RH bands.  Nominal chamber
RH remains metadata and does not replace that droplet review.

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
