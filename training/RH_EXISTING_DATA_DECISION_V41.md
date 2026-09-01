# RH existing-data decision after v41 ROI validation

## Scope

새 촬영 없이 사용할 수 있는 RH 자료를 run 단위로 분리해 검토했다. 농도 결과는
기존 endpoint/run-holdout 산출물을 사용하고, ROI 및 false simultaneous 여부는 v41
브라우저 회귀 결과를 사용한다. cropped 영상은 같은 원본의 파생본이므로 독립 표본으로
중복 계산하지 않는다.

## Geometry and state gate

| Test | Result |
|---|---:|
| cropped ROI lock | 8/8 |
| uncropped ROI lock | 6/6 |
| false simultaneous after v41 guard | 0 |

## Seven-band concentration

`response3 ↔ response6` complete-run holdout의 공유 7구간 모델 결과:

| Metric | Score |
|---|---:|
| exact accuracy | 0.444 |
| balanced accuracy | 0.448 |
| within one adjacent range | 0.889 |
| MAE | 7.41 %RH |
| response3 held out exact | 0.429 |
| response6 held out exact | 0.462 |

사전 고정한 9개 시간/3-frame 정책 중 최선인 `latest trailing3`도 exact와 balanced가
각각 `0.643`, adjacent가 `0.929`, MAE가 `4.29 %RH`였다. 따라서 프레임 시점만
변경해서 7구간 0.85를 만드는 방법은 현재 자료에서 지지되지 않는다.

선택된 1-NN 공유 모델의 행 정규화 confusion matrix(0–1):

| True / Pred. | 20–30 | 30–40 | 40–50 | 50–60 | 60–70 | 70–80 | 80–90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20–30 | 0.750 | 0.250 | 0 | 0 | 0 | 0 | 0 |
| 30–40 | 0.667 | 0.333 | 0 | 0 | 0 | 0 | 0 |
| 40–50 | 0 | 0 | 0.333 | 0.667 | 0 | 0 | 0 |
| 50–60 | 0 | 0.200 | 0.600 | 0.200 | 0 | 0 | 0 |
| 60–70 | 0.250 | 0 | 0.250 | 0.250 | 0.250 | 0 | 0 |
| 70–80 | 0 | 0 | 0 | 0 | 0 | 0.667 | 0.333 |
| 80–90 | 0 | 0 | 0 | 0 | 0 | 0.400 | 0.600 |

## Existing coarse-range evidence

기존 16개 audited endpoint의 4구간 결과는 exact와 balanced가 `0.875`였지만
40–50과 60–70 recall이 각각 `0.750`이라 모든 대각값 0.85 조건은 실패한다.

| True / Pred. | 20–30 | 40–50 | 60–70 | 80–90 |
|---|---:|---:|---:|---:|
| 20–30 | 1.000 | 0 | 0 | 0 |
| 40–50 | 0 | 0.750 | 0.250 | 0 |
| 60–70 | 0 | 0.250 | 0.750 | 0 |
| 80–90 | 0 | 0 | 0 | 1.000 |

관측된 오류는 모두 두 중간 범위 사이에서만 발생했다. 두 범위를 합치면 다음 3구간
행렬이 된다. 이는 새 모델의 독립 성능이 아니라 기존 16 endpoint를 더 넓게 표현했을 때의
파생 결과다.

| True / Pred. | Low 20–30 | Middle 40–70 | High 80–90 |
|---|---:|---:|---:|
| Low 20–30 | 1.000 | 0 | 0 |
| Middle 40–70 | 0 | 1.000 | 0 |
| High 80–90 | 0 | 0 | 1.000 |

## Recovery external check

daylight 90→20 recovery는 frozen reaction 모델에서 exact/balanced `0.286`, adjacent
`0.857`, MAE `8.57 %RH`였다. 따라서 상승 reaction과 하강 recovery는 하나의
정량 모델로 합치지 않는다.

## Decision

새 촬영이 불가능한 동안 선택 가능한 정직한 경로는 다음 둘이다.

1. 앱의 7구간 표시를 유지하되 `experimental`로 남기고 0.85를 주장하지 않는다.
2. 기본 출력을 `20–30`, `40–70`, `80–90`의 3개 validated broad range로 바꾸고,
   기존 7구간은 보조 experimental estimate로만 표시한다.

현재 자료에서 모든 행 recall 0.85 이상을 만족시키려면 두 번째 경로가 유일하다.
다만 3구간의 1.0은 16 endpoint에서 파생된 값이므로 새 환경 일반화 성능으로 해석하지
않는다.

