# Cross-run optical concentration atlas

## 질문

한 기준 영상의 H2/RH 색 상태와 다른 영상의 유사 프레임을 타임라인 없이 연결할 수 있는지
시험했다. 후보 영상의 마지막 상태를 최고 농도로 강제하지 않으므로 실제 가스 부족 또는
불완전 반응 run은 낮은 reference-equivalent 농도에서 멈출 수 있다.

## 기준과 데이터 격리

- H2 기준: `1_90_H2_only_test_2_cropped`의 0→4% Reaction
- RH 20–80 기준: `1_90_H2O_only_extract_3min/extra_cropped`
- RH 90 보완: `H2O_only_3/6(response)`의 90% 구간
- H2는 불꽃, RH는 등록된 물방울 특징만 사용
- 모든 candidate의 타임라인은 매칭 중 사용하지 않음
- candidate 타임라인은 매칭 완료 후 평가에만 공개
- simultaneous와 H2 recovery는 제외

각 영상은 자체 Initial 대비 LAB·상대 형상색·chroma 분포로 표현한다. candidate 프레임과
기준 궤적의 거리를 계산하고, 농도가 감소하지 않는 경로를 동적 계획법으로 선택한다.
RH daylight recovery만 역방향 경로를 허용한다. 경로의 끝은 4% 또는 90%로 고정하지 않는다.

## 발견된 광학적 최대값

| H2 candidate run | test_2-equivalent 최대값 |
|---|---:|
| `h2-daylight-5` | 3.05% |
| `h2-indoor-4` | 1.85% |
| `h2-test-3` | 1.05% |
| `h2-test-indoor` | 3.05% |

이 결과는 일부 run이 기준 영상의 4% 색까지 도달하지 않는다는 가정을 허용한 결과다.
다만 이것만으로 실제 가스 부족과 촬영/run별 색 좌표 이동을 구분할 수는 없다.

## 타임라인 공개 후 평가

| 평가 | Exact | Stage-balanced | ±1단계 | MAE |
|---|---:|---:|---:|---:|
| H2 전체 candidate ramp | 11.7% | 27.9% | 53.0% | 1.51%p |
| H2 endpoint만 | 45.6% | - | 80.7% | 0.75%p |
| RH 전체 candidate ramp | 17.3% | 12.6% | 40.6% | 27.67%RH |
| RH endpoint만 | 24.2% | - | 50.0% | 16.83%RH |

단일 reference의 절대 색 궤적은 run 간 이동을 흡수하지 못한다. 특히
`rh-indoor-fast`는 20→90% 타임라인 전체가 기준 atlas의 20→24% 범위에 압축됐다.
이는 사용자가 확인한 실제 물방울 변화가 없다는 의미가 아니라, 그 변화 방향과 크기가
`rh-indoor-long`의 좌표와 다르다는 의미다.

## 자동 학습 여부

상호 최근접이고 단계 거리 여유가 있는 프레임만 고신뢰 pseudo-label 후보로 제한했다.

| 축 | 후보 수 | coverage | 타임라인 공개 후 exact | ±1단계 |
|---|---:|---:|---:|---:|
| H2 | 6 | 0.7% | 50.0% | 100% |
| RH | 11 | 1.8% | 36.4% | 54.5% |

정확 라벨로 자동 추가하기에는 수와 정밀도가 모두 부족하다. 따라서 이 pseudo-label은
현재 모델 학습이나 앱에 배포하지 않는다. 결과는 잘못된 자동 라벨을 방지하는 audit로
보존한다.

## 다음 단계

한 영상의 절대 좌표 대신 여러 독립 run이 동의하는 **consensus atlas**가 필요하다.

1. 제공된 endpoint 라벨은 유지하되 각 run을 차례로 완전히 제외한다.
2. 나머지 run들의 단계별 중심과 허용 분산을 만들어 하나의 점이 아닌 색 영역으로 표현한다.
3. 제외한 run은 Initial 한 점만 보정하고 각 단계 영역과 비교한다.
4. 최소 두 reference run이 같은 단계에 동의할 때만 pseudo-label 후보로 만든다.
5. H2 1/2/3%, RH 40/50/60% recall과 stage-balanced accuracy가 함께 개선될 때만 배포한다.

재현 코드는 `cross_run_atlas_analysis.py`, 결과는
`output/cross_run_atlas_v1/{metrics.json,matches.csv,reference_atlas.csv}`와 H2/RH PNG에 있다.

## Daylight recovery RH 기준 A/B

사용자 요청에 따라 `1_90_H2O_only_cropped.mp4`의 90→20% recovery 전체를 단일 RH
기준으로 다시 평가했다. 논리 영상명 `1_90_H2O_only.MOV`의 타임라인을 유지하면서 실제
프레임은 `_cropped.mp4`를 우선 사용한다.

| RH 기준 | 전체 exact | Stage-balanced | Endpoint exact | Endpoint ±1단계 | Endpoint MAE |
|---|---:|---:|---:|---:|---:|
| Indoor long 20–80 + response 90 | 17.3% | 12.6% | 24.2% | 50.0% | 16.83%RH |
| Daylight recovery 90–20 | 15.3% | 17.0% | 38.7% | 54.6% | 17.31%RH |

Daylight 기준은 endpoint exact와 stage-balanced를 높였고 특히 `rh-response-6`에는
exact 61.2%, MAE 5.99%RH로 잘 맞았다. 반면 `rh-indoor-fast`는 여전히 20–30%에
압축되고 indoor long도 최대 59.5%에 머문다. 따라서 daylight recovery는 더 나은
reference 구성요소지만 모든 Reaction run의 단일 기준으로 곧바로 배포할 수는 없다.
다음 consensus atlas에서는 상승/하강 방향을 별도 prototype으로 유지한다.

결과 경로는 `output/cross_run_atlas_daylight_rh_v1`이다.
