# Training timelines

All `1_90_RHXX_N` clips begin at the Reaction start for that RH. Times below are
the source-recording times; clip-local time is obtained by subtracting the
Reaction start and dividing by two only when the filename contains `_x2`.

## Labelling rules

- Filename convention: `1_<view angle>_<recording index>`; for example,
  `1_90_2` is the second recording at 90 degrees and `1_70_3` is the third
  recording at 70 degrees.
- Viewing angle and in-plane sample orientation are separate metadata. The
  70/80-degree run-2 recordings below show the droplet on the left and flame on
  the right, so their shape ROI is rotated 90 degrees before feature extraction.
- Recordings on/after 2026-08-07 have the flame at the top. Earlier recordings
  are not assigned an orientation from the `90` filename. The older H2-only and
  H2O-only files listed here were visually verified with the flame at the top.
  Older full simultaneous files `1_90_4/5` have the flame on the right, while
  the derived `1_90_RHxx_2/3/4/5` state-training clips were visually verified
  with the flame at the top. Full-video legacy features are therefore excluded
  until separately re-extracted with fixed ROIs and orientations.

- `H2_only`: flame/H2 reference data.
- `H2O_only`: droplet/RH reference data.
- Neither suffix: simultaneous condition.
- Reaction: stated RH setpoint with an H2 0 to 4% ramp.
- Recovery: RH20 and H2 0%, used as the baseline/recovery condition.
- Simultaneous RH setpoints are metadata only. They do not supervise the optical
  RH model because reduced simultaneous flow makes setpoint RH90 resemble roughly
  RH-only 60--80.
- Simultaneous state classification is evaluated only through nominal RH80.
  Nominal RH90 reaction frames are tagged `simultaneous_rh90_saturated` and
  excluded from H2/state supervision because droplet saturation obscures the
  flame response. Their recovery tails may still provide RH20/H2 0 baselines.
- The quantitative RH model is trained only from `H2O_only` droplet colours.

## Run 5

| RH | Reaction | Recovery | Clip file |
|---:|---|---|---|
| 20 | 0:00-1:48 | 1:48-3:34 | `1_90_RH20_5_x2.mp4` |
| 30 | 3:34-5:16 | 5:16-7:21 | `1_90_RH30_5_x2.mp4` |
| 40 | 7:21-9:42 | 9:42-11:35 | `1_90_RH40_5_x2.mp4` |
| 50 | 11:35-14:24 | 14:24-17:13 | `1_90_RH50_5_x2.mp4` |
| 60 | 17:13-19:41 | 19:41-21:04 | `1_90_RH60_5_x2.mp4` (sample rotates near boundary) |
| 70 | 21:04-24:36 | 24:36-26:20 | `1_90_RH70_5_x2.mp4` |
| 80 | 26:20-28:48 | 28:48-30:33 | `1_90_RH80_5_x2.mp4` |
| 90 | 30:33-34:04 | 34:04-35:07 | `1_90_RH90_5_x2.mp4` |

## Run 4

| RH | Reaction | Recovery | Speed |
|---:|---|---|---:|
| 20 | 0:00-2:00 | 2:00-3:00 | 2x |
| 30 | 3:00-5:00 | 5:00-6:00 | 1x |
| 40 | 6:00-7:30 | 7:30-8:28 | 1x |
| 50 | 8:28-10:30 | 10:30-11:00 | 1x |
| 60 | 11:00-13:30 | 13:30-14:29 | 2x |
| 70 | 14:29-16:34 | 16:34-17:36 | 1x |
| 80 | 17:36-19:52 | 19:52-20:53 | 2x |
| 90 | 20:53-23:00 | 23:00-24:01 | 1x |

## Run 2

| RH | Reaction | Recovery |
|---:|---|---|
| 20 | 0:20-2:20 | 2:20-3:10 |
| 30 | 3:10-5:13 | 5:13-6:05 |
| 40 | 6:05-8:06 | 8:06-9:04 |
| 50 | 9:04-11:07 | 11:07-12:07 |
| 60 | 12:07-14:11 | 14:11-15:15 |
| 70 | 15:15-17:22 | 17:22-18:37 |
| 80 | 18:37-20:43 | 20:43-21:50 |
| 90 | 21:50-24:18 | 24:18-25:18 |

The RH60 source timestamp was corrected from 12:04 to 12:07. The existing RH60
clip is about three seconds longer than the corrected Reaction + Recovery span,
so its first three seconds are treated as pre-reaction baseline footage.

## Run 3

| RH | Reaction | Recovery |
|---:|---|---|
| 20 | 0:00-2:00 | 2:00-3:15 |
| 30 | 3:20-5:10 | 5:10-6:30 |
| 40 | 6:30-7:30 | 7:30-8:45 |
| 50 | 8:45-10:00 | 10:00-11:00 |
| 60 | 11:00-12:10 | 12:10-13:27 |
| 70 | 13:27-14:50 | 14:50-15:45 |
| 80 | 15:45-17:10 | 17:10-18:30 |
| 90 | 18:30-20:30 | 20:30-21:22 |

The 3:15-3:20 gap is excluded from learning.

## Full simultaneous recordings at other viewing angles

### `1_80_2.MOV` (80 degrees, recording 2)

| RH | Reaction | Recovery |
|---:|---|---|
| 20 | 0:08-2:18 | 2:18-2:53 |
| 30 | 2:53-5:08 | 5:08-6:04 |
| 40 | 6:04-8:14 | 8:14-9:08 |
| 50 | 9:08-11:10 | 11:10-12:00 |
| 60 | 12:00-14:14 | 14:14-15:16 |
| 70 | 15:16-17:18 | 17:18-18:01 |
| 80 | 18:01-20:27 | 20:27-21:19 |
| 90 | 21:19-23:04 | 23:04-24:01 |

### `1_70_2.MOV` (70 degrees, recording 2)

| RH | Reaction | Recovery |
|---:|---|---|
| 20 | 0:08-2:08 | 2:08-3:05 |
| 30 | 3:05-4:22 | 4:22-6:10 |
| 40 | 6:10-7:42 | 7:42-9:00 |
| 50 | 9:00-11:03 | 11:03-11:49 |
| 60 | 11:49-13:53 | 14:14-14:54 |
| 70 | 14:54-16:42 | 16:42-18:15 |
| 80 | 18:15-20:49 | 20:49-22:06 |
| 90 | 22:06-24:25 | 24:25-25:43 |

The 13:53-14:14 interval in `1_70_2.MOV` is unlabelled and excluded.

## H2-only concentration references

| Video | 0% | 1% | 2% | 3% | 4% | Recovery |
|---|---|---|---|---|---|---|
| `1_90_H2_only_test.mp4` | before recording | 0-15 s | 15-25 s | 25-30 s | 30 s-end | - |
| `1_90_H2_only_test_2.mp4` | 0-4 s | 4-13 s | 13-21 s | 21-30 s | 30 s-end | - |
| `1_90_H2_only_test_3.MOV` | 0-3 s | 3-10 s | 10-20 s | 20-28 s | 28-152 s | - |
| `1_90_H2_only_4.mp4` | 0-5 s | 5-13 s | 13-30 s | 30-109 s | 109-122 s | 122 s-end |
| `1_90_H2_only_5.mp4` | 0-5 s | 5-8 s | 8-13 s | 13-21 s | 21-130 s | 130 s-end |

`1_90_H2_only_test.mp4` has no timed 0% segment inside the recording, so it
uses the indoor 0% baseline from `1_90_H2_only_test_2.mp4`. Concentrations 1%
and 2% are treated as an uncertain transition band; binary H2 training uses
only 0% as negative and 3--4% as positive.

## H2O-only humidity references

| Video | Lighting | Timeline |
|---|---|---|
| `1_90_H2O_only_2_extract.mp4` | Indoor | 20% 3-6 s; 30% 6-9 s; 40% 9-15 s; 50% 15-25 s; 60% 25-35 s; 70% 35-45 s; 80% 45-72 s; 90% 72-140 s |
| `1_90_H2O_only.MOV` | Daylight | 90% 0-8 s; 80% 8-11 s; 70% 11-13 s; 60% 13-15 s; 50% 15-20 s; 40% 20-23 s; 30% 23-30 s; 20% 30-39 s |
| `1_90_H2O_only_extract_3min.mp4` + `extract_extra.mp4` | Indoor | source timeline: 20% 0-14 s; 30% 14-25 s; 40% 25-45 s; 50% 45-90 s; 60% 90-120 s; 70% 120-189 s; 80% 189-267 s; no 90% |

The full `1_90_H2O_only_2.mp4` and its 140-second extract are not counted as
independent recordings. The extract is used because it contains exactly the
labelled section. The two long-extract files are also one recording group.

## H2O-only response recordings with confirmed timelines

| Video | Decoded-pixel alignment | Confirmed RH timeline | Excluded tail |
|---|---|---|---|
| `1_90_H2O_only_6(response).mp4` | flame right; rotate 1 quarter-turn | 20% 0-7 s; 30% 7-10 s; 40% 10-13 s; 50% 13-14 s; 60% 14-16 s; 70% 16-18 s; 80% 18-20 s; 90% 20-32 s | after 32 s (>90/range exceeded) |
| `1_90_H2O_only_3(response).mp4` | flame left; rotate 3 quarter-turns | 20% 0-2 s; 30% 2-3 s; 40% 3-5 s; 50% 5-7 s; 60% 7-11 s; 70% 11-25 s; 80% 25-28 s; 90% 28-38 s | after 38 s (>90/range exceeded) |

Container rotation metadata explains why the display orientation and decoded
pixel orientation differ. Both videos use fixed, visually verified chamber ROIs
and 4 Hz sampling because several RH steps are only one to three seconds long.
Only the confirmed intervals above supervise state and concentration models;
the range-exceeded tails are excluded. Earlier automatically inferred weak
labels for these recordings are disabled and are not duplicated in training.
