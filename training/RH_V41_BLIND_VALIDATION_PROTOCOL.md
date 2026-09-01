# RH v41 blind validation protocol

## 목적

v41의 ROI 및 H2O-only 상태 판정을 동결한 상태에서 RH 7구간 농도 성능이
독립 영상에서도 0.85 이상인지 검증한다. 기존 `1_90_H2O_only.MOV`는
90→20% recovery 영상이므로 모델 수정용으로 다시 사용하지 않고 외부검증 자료로만
남긴다.

## 촬영 구성

- 조건: H2 0%, 각 출력 구간의 중앙값 RH 25→35→45→55→65→75→85% 상승
- 독립 run: 최소 3개
- 각 RH 단계 유지시간: 20초
- 조명, 카메라 거리와 각도는 실제 사용 범위 안에서 run마다 다시 설치
- 각 run 시작 시 RH 20%, H2 0% 프레임으로 별도 보정
- 원본 영상 하나를 보관하고 cropped 영상은 같은 원본에서 파생

| RH 단계 | 영상 구간 | 평가 burst |
|---:|---:|---:|
| 25% (20–30) | 0–20 s | 13–19 s, 1초 간격 7개 |
| 35% (30–40) | 20–40 s | 33–39 s, 1초 간격 7개 |
| 45% (40–50) | 40–60 s | 53–59 s, 1초 간격 7개 |
| 55% (50–60) | 60–80 s | 73–79 s, 1초 간격 7개 |
| 65% (60–70) | 80–100 s | 93–99 s, 1초 간격 7개 |
| 75% (70–80) | 100–120 s | 113–119 s, 1초 간격 7개 |
| 85% (80–90) | 120–140 s | 133–139 s, 1초 간격 7개 |

각 burst는 앱과 동일한 연속 3프레임 median을 사용한다. 시간을 결과를 본 뒤
변경하지 않는다.

## 데이터 분리

1. Run 1–2: 모델 및 threshold 개발용
2. Run 3: 수정 중 열어보지 않는 최종 blind validation용
3. Cropped/uncropped 쌍: 정확도 표본을 두 배로 세지 않고 ROI 불변성 검사에만 사용

## 평가 항목

- H2O-only 상태 recall
- RH 7구간 exact accuracy
- stage-balanced accuracy
- 각 RH 구간 recall
- ±1구간 이내 정확도
- MAE (%RH)
- cropped/uncropped 동일 시점 예측 일치율
- 행 정규화 confusion matrix (각 행 합계 1.0)

## 통과 기준

- exact accuracy ≥ 0.85
- stage-balanced accuracy ≥ 0.85
- 모든 RH 구간 recall ≥ 0.85
- H2O-only 상태 recall ≥ 0.85
- cropped/uncropped ROI geometry pass = 1.0
- false simultaneous rate = 0

위 조건 중 하나라도 실패하면 blind run의 특정 프레임에 맞춘 예외 규칙을 추가하지
않는다. 실패 구간과 환경을 새 학습 run에 포함한 뒤 새로운 전체 run으로 다시 검증한다.

## 진행 순서

1. v41 ROI/state 코드를 고정한다.
2. Run 1을 촬영해 안정화 시간과 고정 추출 규칙을 확인한다.
3. 같은 규칙으로 Run 2를 촬영하고 모델을 한 번만 조정한다.
4. Run 3 전체를 blind evaluation한다.
5. 0–1 confusion matrix와 구간별 recall을 보고 통과 여부를 결정한다.
