# Place-2 seven-band complete-run-held-out validation

## Question

Can one shared seven-range optical model train on all of response3 and predict
response6, and then train on response6 and predict response3?

Both `1_90_H2O_only_3(response).mp4` and
`1_90_H2O_only_6(response).mp4` are Place-2 recordings. Each fold excludes one
complete video, including all neighbouring frames from that video.

## Data

The evaluation uses 27 user-confirmed frames:

- response3: 14 frames, two times per RH range
- response6: 13 frames, one to three times per RH range

Features are calibrated large-droplet colour minus nearby substrate colour in
LAB. Three pre-declared candidates were compared: standardized 1-NN, ordinal
ridge and multinomial logistic regression.

## Results

| Candidate | Exact | Balanced | Adjacent | MAE (%RH) |
|---|---:|---:|---:|---:|
| standardized 1-NN | **0.444** | **0.448** | 0.889 | 7.41 |
| ordinal ridge | 0.407 | 0.417 | **0.926** | **6.67** |
| multinomial logistic | 0.407 | 0.362 | 0.815 | 8.89 |

Selected diagnostic candidate: standardized 1-NN.

Its complete-run folds were:

- train response6, hold out response3: exact **0.429**
- train response3, hold out response6: exact **0.462**

The candidate fails the deployment rule requiring overall exact accuracy,
every range recall and both complete-run folds to reach 0.85. It also performs
worse than the frozen v37 app's unused-time estimate of 0.643.

## Decision

Do not replace the app's separated response3/response6 profiles with a shared
seven-band model. The existing four broad-band complete-run audit reaches
0.875, so the optical response transfers at coarse resolution, but the two
runs do not share stable 10%-wide middle-range boundaries.

The next defensible improvement requires either a third independent Place-2
rising run to learn run-to-run variation, or wider output intervals when the
single-frame evidence cannot resolve neighbouring middle ranges. Reusing more
neighbouring frames from these same two videos would increase sample count but
would not add an independent experiment.
