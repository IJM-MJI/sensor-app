# Simultaneous quantitation gate v1

## Reproducible transfer result

The four-state model may identify a simultaneous response, but that does not
make either single-condition concentration model quantitatively transferable.
The analyses were rerun from the current verified cache before deployment.

For late-reaction frames, the H2-only model produced the following medians:

| Nominal simultaneous RH metadata | Median raw H2 estimate |
|---:|---:|
| 20 | 3.04% |
| 30 | 2.38% |
| 40 | 1.94% |
| 50 | 2.18% |
| 60 | 1.41% |
| 70 | 1.79% |
| 80 | 1.59% |

The late reaction is the part expected to contain the strongest H2 response,
yet the transferred estimate drops as the simultaneous humidity condition
changes. This demonstrates RH interference in the flame quantitation path.
There are no supplied within-reaction H2 stage timestamps for the simultaneous
clips, so the intermediate `1–2` and `2–3` ranges cannot be validated honestly.

The direct H2O-only RH transfer is also unresolved. Its level medians are
compressed to approximately 78.5--80.3% across nominal simultaneous RH20--80.
The attempted H2 interference subtraction made the held-out H2-only droplet
residual 9.3% worse. Rank ordering improved, but nominal simultaneous RH is
diagnostic metadata rather than optical RH ground truth and is not an accuracy
measurement.

## App policy

- `H2_only`: report the environment-routed experimental H2 range.
- `H2O_only`: report the H2O-only experimental RH range.
- `simultaneous`: report the four-state condition, but hide both concentration
  numbers until their cross-interference corrections are validated.
- RH90 simultaneous remains outside the intended quantitative scope; the target
  range remains RH20--80.

This prevents a state-classification success from being presented as an
unsupported quantitative result. Diagnostic raw values remain available to the
developer trace for model development.

### Confirmed simultaneous RH target (2026-09-02)

The RH number embedded in `1_90_RHxx_<run>` is the simultaneous experiment
setting, not the displayed concentration target.  Because H2 and humid flow are
supplied together, the optical RH response can be lower than the corresponding
RH-only experiment.  The deployed target is therefore the **RH-only-equivalent
optical band**:

- Compare the simultaneous droplet response with the reviewed RH-only colour
  path and report one of 20--30, 30--40, ..., 80--90% RH.
- Keep the filename RH only as grouping/diagnostic metadata.  It must not
  supervise the reported RH class.
- Condition the droplet estimate on flame features to remove H2 cross-talk.
- Use a single frame.  Elapsed time and neighbouring-frame voting are not model
  inputs.
- Use the uncropped RH-labelled files directly.  The deployed scale-aware ROI
  is responsible for locating the chamber and sensing card.

Thus a nominal simultaneous `RH70` frame may correctly display, for example,
`50--60% RH (RH-only equivalent)` when its corrected droplet colour matches that
RH-only stage.

## Next modelling requirement

The next model must estimate the two cross-interference terms jointly:

1. H2 estimate from the flame after conditioning on the registered droplet
   response.
2. RH-only-equivalent seven-band RH from the droplet after conditioning on the
   registered flame response.  Nominal simultaneous RH is not the label.

For a genuine concentration confusion matrix, the simultaneous reaction needs
H2 stage anchors within the reaction interval (at least the video times at which
0, 1, 2, 3, and 4% are reached). Without those anchors, existing videos support
state classification and endpoint diagnostics, but not intermediate H2
concentration truth.

`make_simultaneous_anchor_template.py` generates
`simultaneous_stage_anchors.csv` for all 28 RH20--80 clips in runs 2--5. Known
reaction start/end times prefill the 0%/4% columns; 1%/2%/3% stay intentionally
blank until the experiment log or human video review supplies them. They must
not be filled by linear interpolation, because response lag is one of the
effects this evaluation is meant to measure.
