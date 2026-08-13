# Dual Sensor Camera Classifier

카메라 영상에서 센서의 불꽃 영역과 물방울 영역 색 변화를 읽어 다음 네 상태를 분류하는 실험용 웹 앱입니다.

- 반응 없음
- H2 반응
- 높은 습도 반응
- H2 + 습도 동시 반응

## 사용 방법

1. 센서를 `RH 20%, H2 0%` 회복 조건에 둡니다.
2. 앱에서 **Calibrate**를 눌러 기준 영상을 촬영합니다.
3. 불꽃이 위, 물방울이 아래가 되도록 센서를 정렬하고 **Capture**로 상태를 판별합니다.
4. 센서·조명·카메라 위치가 바뀌면 다시 보정합니다.

H2 또는 RH 확률이 판정 임계값의 ±0.10 안에 있으면 앱은 상태를 강제로 확정하지 않고 `Uncertain / Retake`를 표시합니다.

원형 창 안에서 불꽃이 위쪽, 물방울이 아래쪽에 보이도록 촬영해야 합니다. 모델은 보정 프레임과 현재 프레임의 LAB 색 차이를 사용하므로 조명과 ROI 크기를 일정하게 유지하는 것이 중요합니다.

## 학습 데이터와 라벨 규칙

주 학습 원본은 `recordings/1`의 2026-08-07 이후 영상이며, 이전 영상은 ROI와 방향을 직접 확인한 경우에만 사용합니다. 상세 타임라인은 [`training/TIMELINES.md`](training/TIMELINES.md)에 기록했습니다.

파일명은 `1_<촬영 각도>_<촬영 회차>` 규칙입니다. 촬영 각도와 영상 안에서 센서 도형이 돌아간 방향은 별도로 관리하며, 회전된 영상은 불꽃/물방울 ROI를 정렬한 후 특징을 계산합니다. 컨테이너 회전 메타데이터 때문에 재생 화면과 OpenCV 픽셀 방향이 다를 수 있어 디코딩된 대표 프레임으로 방향을 확인합니다. 타임라인이 없는 영상의 자동 RH 추정치는 농도 정답으로 사용하지 않으며, 영상 내부 변화가 크고 안정적인 프레임만 상태 학습 보강에 사용할 수 있습니다.

- `H2_only`: H2 반응 학습용
- `H2O_only`: 습도/물방울 색 학습용
- 접미사가 없는 영상: H2와 RH 동시 주입
- Reaction: 해당 RH 설정값에서 H2가 0→4%로 증가
- Recovery: RH 20%, H2 0%

동시 주입 영상의 RH 설정값은 광학적 RH 정답으로 사용하지 않습니다. 현재 장치에서는 동시 주입 시 물 유량이 감소하므로 RH 90 설정도 RH-only 60~80과 비슷하게 보일 수 있습니다. 높은 습도 모델은 `H2O_only` 물방울 색만으로 학습했습니다.

## 검증 결과

프레임을 무작위로 섞지 않고 영상 단위로 분리한 교차 검증 결과입니다.

| 모델 | Balanced accuracy | ROC AUC | 용도 |
|---|---:|---:|---|
| H2 반응 감지 | 81.2% | 0.902 | 방향 정렬된 불꽃, simultaneous RH90 포화 구간 제외 |
| RH 70% 이상 감지 | 80.2% | 0.887 | 5개 독립 RH-only run의 물방울 색만 사용 |

상태 모델은 불꽃·물방울·전체 고채도 영역의 LAB 변화량을 함께 사용해 Initial, H2-only, H2O-only, simultaneous를 직접 분류합니다. RH 농도 모델은 H2O-only 물방울 색만 사용하며 simultaneous 영상의 RH 설정값은 정량 학습 정답에 들어가지 않습니다. 흰색·회색 보정 패치로 LAB 색 편향을 먼저 보정하고 고정된 불꽃/물방울 형태 영역에서만 특징을 계산합니다. 카메라 활성화 후 0.5초만 준비하고, 촬영 버튼을 누르는 순간 한 프레임을 고정해 분석하므로 손으로 들고 있는 동안의 이동 평균으로 ROI가 흐려지지 않습니다.

현재 방향과 타임라인을 확인한 43개 분할/단독 영상에서 4,373개 특징 프레임을 관리합니다. H2-only와 일반 H2O-only 영상은 최소 2 Hz, 짧은 response 영상 두 개는 확정 범위까지 4 Hz로 추출합니다. 상태 검증에서는 고밀도 정량 프레임이 특정 영상에 과도한 가중치를 주지 않도록 일반 영상을 4초 간격으로 다시 샘플링하며, simultaneous RH90 reaction은 포화 범위로 제외합니다. 최종 상태 검증 표본은 16개 독립 실험 그룹의 741프레임입니다. 한 원본 실험에서 RH별로 잘라낸 클립은 같은 그룹으로 묶어 통째로 홀드아웃하므로 촬영 조건이 학습과 검증에 동시에 들어가지 않습니다. 타임라인이 없거나 ROI·방향이 확인되지 않은 기존 긴 영상은 예측 검토용으로만 남기고 학습·검증 수치에서 제외했습니다.

단일 축 검출기의 독립 영상 홀드아웃 성능은 H2 반응 balanced accuracy 81.2%, AUC 0.902, RH 70% 이상 반응 balanced accuracy 80.2%, AUC 0.887입니다. H2 정량은 불꽃 영역을, RH 정량은 H2O-only 물방울 영역을 사용합니다. 두 검출기와 4상태 직접 모델 모두 새로운 센서·조명·카메라에 대한 외부 검증은 아직 필요합니다.

검증된 방향표로 정렬한 `1_70_2.MOV`와 `1_80_2.MOV`를 포함하고 Recovery 마지막 안정 구간만 Initial로 사용합니다. `1_90_H2O_only_3(response)`와 `1_90_H2O_only_6(response)`는 제공된 정식 타임라인으로 상태·정량 검증에 포함하며, 각각 38초와 32초 이후의 범위 초과 구간은 제외합니다. simultaneous 상태 목표는 명목 RH30--80이며 RH90 reaction은 `saturated/out-of-scope`로 제외합니다. 앱에 배포한 320-tree 직접 4상태 모델의 프레임 정확도는 71.0%, balanced accuracy는 70.1%, 안정 구간 정확도는 80.0%, simultaneous 재현율은 42.7%입니다. 최고 상태 확률 0.35 이상만 표시하면 coverage 92.6%, 판정 정확도 73.8%이며 나머지는 `Uncertain / Retake`로 처리합니다. 타임라인과 영상 반응이 충돌하는 구간은 `training/output/simultaneous_review`에 run별 판별 시트로 보존합니다.

앱은 촬영 순간의 단일 프레임을 분석하고, 분류가 확실하며 Initial 보정을 마친 경우에만 검증된 강한 반응 범위의 숫자를 표시합니다. H2 숫자는 불꽃 LAB 특징, RH 숫자는 H2O-only 물방울 LAB 특징으로만 학습한 보정식을 사용합니다. simultaneous 영상에 단독 RH식을 그대로 적용하면 명목 RH20--80이 약 75--83%로 뭉쳐 낮은 RH를 크게 과대평가했습니다. 따라서 간섭 보정 전까지 simultaneous의 RH 숫자는 앱에서 숨기고 상태만 표시하며, H2 숫자는 검증 범위 안에서만 표시합니다. 단독 조건 숫자 표시 범위는 H2 3--4%(독립 영상 MAE 0.44%p)와 RH 70--90%(MAE 8.04%p)입니다. 농도 변경 후 5초 미만 프레임은 정량 학습·검증에서 제외했습니다. 전체 0--4% 및 RH 20--90% 모델은 농도별 오차가 불균형하여 범위 밖 숫자를 표시하지 않습니다. 이 값은 연구용 광학 추정치이며 인증된 정량 계측값이 아닙니다.

H2-only 영상에서 불꽃 변화로 예측한 물방울 간섭을 simultaneous 물방울 특징에서 빼는 보정도 검증했습니다. 명목 단계와 optical-equivalent RH의 순서 상관은 0.929에서 0.964로 개선됐지만 결과 범위가 여전히 약 77--83%로 압축됩니다. 명목 RH는 실제 광학 RH 정답이 아니므로 이 결과만으로 농도 정확도를 주장하거나 앱 숫자를 표시하지 않습니다.

H2 ramp 타임라인은 명목 농도로 보존합니다. H2-only 영상은 제공된 단계값을 그대로 사용하고, Recovery 시작부터 H2 0%로 수정했습니다. 각 영상의 색 반응을 다른 영상으로 학습한 색 예측값과 비교해 반응 지연을 별도로 추정하되, 30초와 60초 탐색에서 최적값이 유지되고 오차 개선이 충분한 경우에만 적용합니다. 타임라인이 없거나 ROI 방향이 검증되지 않은 기존 캐시는 지연 추정과 정량 검증에서 제외합니다. 상세 지연값과 탈락 사유는 `training/output/h2_lag_report.json`에 기록됩니다.

최신 정량 검증은 4,373개 특징 프레임을 사용합니다. 단독 조건은 2 Hz, 짧은 response 영상은 4 Hz로 추출했고 상태 정확도는 일반 영상을 4초 간격으로 다시 샘플링해 실험별 가중치를 유지합니다. 앱의 정적 Ridge 정량 범위는 H2 3--4%(LOVO MAE 0.44%p), H2O-only RH 70--90%(LOVO MAE 8.04%p)입니다. 5초 시간 모델은 RH 70--90%에서 MAE 4.89%p를 보였지만 앱에 아직 배포하지 않았습니다. 전체 범위는 농도별 오차가 불균형하여 범위 밖 숫자를 표시하지 않습니다.

### 전체 농도 단계 검증

`training/ordinal_concentration_analysis.py`는 H2-only의 불꽃과 H2O-only의 물방울을 중심으로 농도 증가에 따른 보정 LAB 색 궤적을 비교합니다. 상대 형상 색은 프레임 공통 조명 변화를 제거하는 내부 기준으로만 사용합니다. H2 recovery 중간 프레임은 명목 H2가 0%여도 광학 반응이 남아 있으므로 0% 학습에서 제외하고, 실제 initial과 recovery 마지막만 사용합니다. 짧은 단계는 농도 변경 후 1초부터 포함하고 희소 단계가 모델 선택에서 동일한 비중을 갖도록 stage-balanced accuracy를 사용합니다.

두 검증을 함께 생성합니다. 영상 하나를 통째로 제외하면 H2 정확도 53.9%(stage-balanced 34.8%, MAE 0.78%p), RH 정확도 34.4%(stage-balanced 30.1%, MAE 14.96%p)입니다. 동일 run의 다른 5초 시간 블록으로 검증하면 H2 정확도 71.8%(stage-balanced 50.6%, MAE 0.51%p), RH 정확도 63.4%(stage-balanced 55.9%, MAE 6.60%p)입니다. 후자는 사용 시 calibration으로 촬영 run 차이를 일부 알고 있는 상황에 가까우나 완전히 새로운 run에 대한 성능은 아닙니다. 따라서 전체 범위 숫자는 아직 앱에 배포하지 않았습니다.

Calibration 프레임에서 불꽃/물방울 픽셀을 완전히 고정하는 방식도 별도로 시험했습니다. RH 70--90% 회귀 오차는 4.97%p로 감소했지만 4상태 balanced accuracy가 70.1%에서 60.1%로 낮아지고 전체 농도 단계 오차가 증가해 채택하지 않았습니다. 다음 추출기는 원 위치를 유지하되 촬영 중의 미세 회전·이동을 먼저 정합한 뒤 고정 마스크를 적용해야 합니다.

## 주의 사항

이 앱은 연구용 프로토타입이며 안전 경보기나 정량 계측기가 아닙니다. 특히 H2 감지는 제한된 단일 센서·촬영 환경에서 학습되어 다른 센서, 조명, 카메라에는 일반화가 검증되지 않았습니다. 수소 안전 판단에는 반드시 인증된 전용 검지기를 사용하세요.

## 재학습

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-training.txt
.venv\Scripts\python training\align_h2_lag.py `
  --input training\cache\v7-verified-orientation-recovery-tail\legacy_continuous.csv `
  --output training\cache\v7-verified-orientation-recovery-tail\legacy_continuous_lag_corrected.csv
.venv\Scripts\python training\train_models.py --reuse-cache
.venv\Scripts\python training\analyze_unlabeled_rh.py `
  --video-root "C:\path\to\recordings\1"
.venv\Scripts\python training\state_condition_analysis.py
.venv\Scripts\python training\quantitative_analysis.py
.venv\Scripts\python training\ordinal_concentration_analysis.py
.venv\Scripts\python training\simultaneous_interference_analysis.py
```

지연 상세와 신뢰 여부는 `training/output/h2_lag_report.json`에 저장됩니다. 학습 산출물은 `training/output/metrics.json`, `training/output/models.json`, `sensor-model.js`, `sensor-state-model.js`에 저장됩니다. 영상과 캐시는 용량 때문에 저장소에서 제외합니다.

정량 모델 후보와 영상 단위 오차는 `training/quantitative_analysis.py`로 비교합니다. 논문용 검증 Figure는 `training/output/quantitative/quantitative_validation`의 PNG(600 dpi), PDF, SVG로 생성되며, 동일 Figure의 점과 전체 예측값을 CSV로 함께 저장합니다.

전체 단계별 색 궤적과 confusion matrix는 `training/output/ordinal_concentration/ordinal_concentration_validation`의 PNG(500 dpi), PDF, SVG로 생성되고 프레임별 예측은 같은 폴더의 `predictions.csv`에 저장됩니다.
