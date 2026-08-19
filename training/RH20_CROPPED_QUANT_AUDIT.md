# RH20 cropped H2 quantitation audit

## 데이터

다음 네 독립 crop run을 2 Hz로 다시 추출했다.

- `1_90_RH20_2_x2_cropped.mp4`
- `1_90_RH20_3_x2_cropped.mp4`
- `1_90_RH20_4_cropped.mp4` (normal speed)
- `1_90_RH20_5_x2_cropped.mp4`

총 944프레임이며 불꽃이 위가 되도록 정렬했다. `1_90_RH20_cropped.mp4`는 run 5 x2와
대응 프레임 상관 0.9999인 같은 원본의 normal-speed 사본이므로 중복 사용하지 않았다.

## 타임라인 없는 사전 매칭

기존 Reaction 시간을 숨기고 불꽃 변화의 자동 peak와 `H2_only_test_2` optical atlas만으로
분석했을 때 매칭된 최대값은 run 2/3/4/5에서 각각 3.95/3.25/2.75/1.0%였다. 944프레임
중 상호 최근접 고신뢰 학습 후보는 8개뿐이었고, run 2와 run 5의 자동 peak도 불안정했다.
따라서 이 결과만으로 exact 0--4% 라벨을 만들지 않는다.

## 기존에 알려진 RH20 Reaction을 사용한 A/B

이전에 제공된 Reaction 경계는 사용하되 내부 1/2/3%는 낮은 가중치의 ramp 순서로만
사용했다. 원본과 crop은 동시에 넣지 않고 완전히 교체했다. 검증은 기존 H2-only 다섯
run만 사용했다.

모든 crop 끝을 4%로 강제하면 H2 4% recall이 0%가 되어 실패했다. 실제 H2가 부족할 수
있다는 조건을 반영해 각 crop의 끝을 위 optical-equivalent 최대값으로 제한하고 endpoint도
약한 가중치로 바꾼 결과는 다음과 같다.

| 조건 | Exact | Stage-balanced | ±1단계 | MAE | H2 1–4 exact | H2 1–4 balanced |
|---|---:|---:|---:|---:|---:|---:|
| 기존 RH20 원본 weak ramp | 43.3% | 50.0% | 91.1% | 0.668%p | 38.7% | 37.9% |
| Crop + unforced optical max | 48.0% | 48.6% | 96.8% | 0.554%p | 48.5% | 50.2% |

| H2 단계 | 기존 recall | Crop optical recall |
|---:|---:|---:|
| 0% | 98.7% | 42.1% |
| 1% | 21.7% | 38.0% |
| 2% | 49.1% | 75.5% |
| 3% | 30.1% | 41.7% |
| 4% | 50.5% | 45.8% |

Crop optical 후보는 반응 단계 1--4%의 exact/balanced와 전체 exact, ±1단계, MAE를
크게 개선하지만 0% 경계를 악화시킨다. 앱은 상태 모델이 Initial이면 H2 0%로 고정하므로
이 후보는 **H2-response 전용 농도 모델**로 유망하다. 그러나 실제 상태 게이트와 결합한
end-to-end 검증 전에는 배포하지 않는다. 원본 0% anchor를 단순 추가한 hybrid는 exact
38.1%, balanced 46.8%로 악화되어 채택하지 않았다.

## 다음 단계

현재 앱의 4상태 held-out 예측을 concentration 예측 앞에 실제로 적용한다. Initial은 0%로
고정하고 H2-only/simultaneous로 판정된 프레임에만 crop optical 농도 모델을 통과시킨다.
end-to-end H2 0--4% confusion matrix와 coverage가 기존 앱보다 좋아질 때만 모델을 내보낸다.

