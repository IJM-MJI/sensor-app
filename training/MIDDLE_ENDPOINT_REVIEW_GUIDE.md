# Middle endpoint visual review

## 목적

H2 2/3%와 RH 40/50/60/80%의 엄격한 ramp endpoint를 독립 run끼리 같은 방향과
크기로 비교한다. 타임라인을 자동 수정하는 자료가 아니라, 라벨 오류·센서 반응 지연·
run별 광학 경로 차이·마스크 오류를 사람이 구분하기 위한 QA 자료다.

## 사용 데이터와 처리 흐름

```text
사용자 제공 타임라인
        │  각 구간 끝에 가장 가까운 프레임 1개만 exact endpoint
        ▼
endpoint_interval_registered_v3/endpoint_interval_dataset.csv
        │  logical video/time으로 캐시와 크롭 원본 연결
        ├── features_registered_drop_v2.csv ── 원/회전/등록 물방울 마스크
        └── *_cropped.mp4 우선 사용 ───────── 실제 endpoint 영상 프레임
        │
        ├── run_progress_v3/predictions.csv ── Initial 한 점 보정 held-out 예측
        └── normalized_trajectories.csv ────── two-anchor 경로 진행률(진단 전용)
        ▼
불꽃이 위가 되도록 회전 + 동일 원 크기 정렬 + 실제 측정 픽셀 표시
```

| 대상 | exact endpoint 수 | 비교 run | 모델 입력 영역 |
|---|---:|---:|---|
| H2 2%, 3% | 단계별 5 | 5 | 불꽃 형상 |
| RH 40%, 50%, 60%, 80% | 단계별 5 | 5 | 등록된 주 물방울 형상 |

Reaction/H2O response는 구간 끝에 가장 가까운 단 한 프레임만 exact로 사용한다.
H2 recovery는 포함하지 않는다. RH daylight recovery는 사용자가 제공한 하강 endpoint를
그대로 사용한다. RH 정답은 H2O-only 영상만 사용하며 simultaneous 영상은 들어가지 않는다.

## 그림 읽는 법

- 각 셀 왼쪽은 정렬된 원본 endpoint, 오른쪽은 실제 모델 측정 픽셀이다.
- H2는 주황색 불꽃 픽셀, RH는 청록색 물방울 픽셀이다.
- 초록 테두리는 held-out 예측 exact, 주황은 인접 한 단계, 빨강은 두 단계 이상 차이다.
- `held-out prediction`은 해당 run 전체를 학습에서 뺀 one-anchor 모델 결과다.
- `two-anchor path progress`는 같은 run의 저·고 endpoint 사이에서 본 상대 위치이며 앱에
  쓸 결과가 아니라 색 경로가 직선적인지 확인하는 진단값이다.

산출물:

- `output/middle_endpoint_review/h2_middle_endpoint_review.jpg`
- `output/middle_endpoint_review/rh_middle_endpoint_review.jpg`
- `output/middle_endpoint_review/middle_endpoint_review.csv`

## 결과 요약

| 단계 | exact run | 인접 1단계 | 2단계 이상 오차 |
|---|---:|---:|---:|
| H2 2% | 1/5 | 3/5 | 1/5 |
| H2 3% | 1/5 | 2/5 | 2/5 |
| RH 40% | 2/5 | 0/5 | 3/5 |
| RH 50% | 1/5 | 2/5 | 2/5 |
| RH 60% | 1/5 | 1/5 | 3/5 |
| RH 80% | 1/5 | 3/5 | 1/5 |

이번 시트의 측정 영역은 도형을 제대로 덮고 있다. 따라서 이 endpoint들의 주된 병목은
마스크 잘림이 아니다. 다음 세 패턴이 보인다.

1. **H2는 하나의 공통 지연으로 정렬되지 않는다.** `h2-indoor-4`의 2% endpoint는 0%로,
   `h2-test-3`와 `h2-test-indoor`의 3% endpoint는 1%로 보이지만, `h2-daylight-5`의 2%는
   오히려 3%로 예측된다. 영상별 지연 또는 서로 다른 색 경로가 함께 존재한다.
2. **`rh-indoor-fast`는 중·고 농도 색 변화가 압축된다.** 40%는 맞지만 50/60/80%가 각각
   40/40/40%로 예측되며 신뢰도도 0.90 이상이다. 원 검출을 크롭으로 안정화한 뒤에도
   남는 현상이어서 정확 단계 학습에서 가장 먼저 제외/저가중치 A/B할 후보이다.
3. **짧은 response run은 endpoint가 매우 빠르다.** `rh-response-3/6`의 40--60%는 다른
   run과 크게 어긋나며 two-anchor 진행률도 낮거나 음수다. 제공된 시간표가 맞더라도
   구간 끝 한 프레임이 광학적 안정 농도를 대표하지 못할 가능성이 크다.

RH daylight recovery의 50/60%가 각각 70/80%로 보이는 것은 하강 반응 지연과 일치한다.
따라서 상승 Reaction과 하강 Recovery를 같은 exact 농도 분류기에 합치는 것은 보류한다.

## 다음 판정과 재학습 규칙

사용자 검토 결과 `H2_only_4` 30초, `H2_only_test_3` 28초,
`H2_only_test` 30초는 각각 full 2/3/3%가 맞고, `H2O_only_2_extract`도 실제
물방울 색이 충분히 변한다. 따라서 이 라벨들은 유지하며 자동 이동하거나 제외하지 않는다.

남은 검토 질문은 짧은 response run의 각 endpoint가 광학적으로 안정된 상태인지 여부다.
여기서 안정 상태란 시간표가 틀렸다는 뜻이 아니라, 1--3초짜리 ramp 끝의 한 프레임이
다른 느린 run의 같은 농도 색까지 충분히 도달했는지를 뜻한다.

검토 시 참고한 항목은 다음과 같다.

- `1_90_H2_only_4` 30초 full 2%: 사용자 확인 완료
- `1_90_H2_only_test_3` 28초와 `1_90_H2_only_test` 30초 full 3%: 사용자 확인 완료
- `1_90_H2O_only_2_extract` 물방울 변화: 사용자 확인 완료
- `H2O_only_3(response)`와 `H2O_only_6(response)`의 빠른 40--60% endpoint가 안정 상태인지

판정 뒤에는 두 가지 A/B만 수행한다.

1. 타임라인이 맞지만 아직 과도 상태이면 exact가 아니라 ramp interval로 낮춰 학습한다.
2. run 전체의 색 변화가 압축됐거나 광학 경로가 다르면 exact 학습에서 제외/저가중치하고,
   다른 run held-out 성능이 실제로 올라가는지 확인한다.

라벨을 변경한 뒤에도 독립 run held-out stage-balanced accuracy가 함께 오르지 않으면 배포하지
않는다. 목표 0.85는 같은 영상의 프레임을 나눠 얻는 점수가 아니라, 보지 않은 독립 run에서
달성해야 한다.
