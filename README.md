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

상태 모델은 불꽃·물방울·전체 고채도 영역의 LAB 변화량을 함께 사용해 Initial, H2-only, H2O-only, simultaneous를 직접 분류합니다. RH 농도 모델은 H2O-only 물방울 색만 사용하며 simultaneous 영상의 RH 설정값은 정량 학습 정답에 들어가지 않습니다. 흰색·회색 패치는 학습 영상 촬영 당시 조명과 카메라를 안정화하기 위한 참고물일 뿐 모델 입력으로 사용하지 않으며, 실제 특징은 센서 내부의 중성 배경을 기준으로 보정한 고정 불꽃/물방울 형태 영역에서 계산합니다. 카메라 활성화 후 0.5초만 준비하고, 촬영 버튼을 누르는 순간 한 프레임을 고정해 분석하므로 손으로 들고 있는 동안의 이동 평균으로 ROI가 흐려지지 않습니다.

현재 방향과 타임라인을 확인한 43개 분할/단독 영상에서 4,373개 특징 프레임을 관리합니다. H2-only와 일반 H2O-only 영상은 최소 2 Hz, 짧은 response 영상 두 개는 확정 범위까지 4 Hz로 추출합니다. 상태 검증에서는 고밀도 정량 프레임이 특정 영상에 과도한 가중치를 주지 않도록 일반 영상을 4초 간격으로 다시 샘플링하며, simultaneous RH90 reaction은 포화 범위로 제외합니다. 최종 상태 검증 표본은 16개 독립 실험 그룹의 741프레임입니다. 한 원본 실험에서 RH별로 잘라낸 클립은 같은 그룹으로 묶어 통째로 홀드아웃하므로 촬영 조건이 학습과 검증에 동시에 들어가지 않습니다. 타임라인이 없거나 ROI·방향이 확인되지 않은 기존 긴 영상은 예측 검토용으로만 남기고 학습·검증 수치에서 제외했습니다.

단일 축 검출기의 독립 영상 홀드아웃 성능은 H2 반응 balanced accuracy 81.2%, AUC 0.902, RH 70% 이상 반응 balanced accuracy 80.2%, AUC 0.887입니다. H2 정량은 불꽃 영역을, RH 정량은 H2O-only 물방울 영역을 사용합니다. 두 검출기와 4상태 직접 모델 모두 새로운 센서·조명·카메라에 대한 외부 검증은 아직 필요합니다.

검증된 방향표로 정렬한 `1_70_2.MOV`와 `1_80_2.MOV`를 포함하고 Recovery 마지막 안정 구간만 Initial로 사용합니다. `1_90_H2O_only_3(response)`와 `1_90_H2O_only_6(response)`는 제공된 정식 타임라인으로 상태·정량 검증에 포함하며, 각각 38초와 32초 이후의 범위 초과 구간은 제외합니다. simultaneous 상태 목표는 명목 RH30--80이며 RH90 reaction은 `saturated/out-of-scope`로 제외합니다. 앱에 배포한 320-tree 직접 4상태 모델의 프레임 정확도는 71.0%, balanced accuracy는 70.1%, 안정 구간 정확도는 80.0%, simultaneous 재현율은 42.7%입니다. 최고 상태 확률 0.35 이상만 표시하면 coverage 92.6%, 판정 정확도 73.8%이며 나머지는 `Uncertain / Retake`로 처리합니다. 타임라인과 영상 반응이 충돌하는 구간은 `training/output/simultaneous_review`에 run별 판별 시트로 보존합니다.

앱은 촬영 순간의 단일 프레임을 분석하며 Initial 보정을 마친 뒤 H2 Reaction `0/1/2/3/4%`와 H2O-only RH `20–30/40/50/60/70/80/90%` 전체 단계를 실험적으로 표시합니다. 회귀값이 단계 경계에 가까우면 한 숫자를 강제하지 않고 인접 범위로 표시하며, 상태 분류가 불확실하거나 학습 범위를 벗어나면 `Uncertain / Retake`를 표시합니다. H2 숫자는 불꽃 LAB·chroma 분포 특징, RH 숫자는 H2O-only 물방울 LAB 특징으로만 학습합니다. simultaneous 영상에 단독 RH식을 그대로 적용하면 낮은 RH를 크게 과대평가하므로 간섭 보정 전까지 simultaneous RH 숫자는 계속 숨깁니다. 이 값은 연구용 광학 추정치이며 인증된 정량 계측값이 아닙니다.

H2-only 영상에서 불꽃 변화로 예측한 물방울 간섭을 simultaneous 물방울 특징에서 빼는 보정도 검증했습니다. 명목 단계와 optical-equivalent RH의 순서 상관은 0.929에서 0.964로 개선됐지만 결과 범위가 여전히 약 77--83%로 압축됩니다. 명목 RH는 실제 광학 RH 정답이 아니므로 이 결과만으로 농도 정확도를 주장하거나 앱 숫자를 표시하지 않습니다.

H2-only 타임라인의 각 구간 끝은 해당 농도에 도달한 시점입니다. 구간 시작부터 끝까지 농도를 선형 보간하며 Recovery도 시작 즉시 0%로 바꾸지 않고 마지막에 0%가 되도록 하강시킵니다. 이 명목 ramp와 영상의 색 반응 사이에 추가 지연이 있는지는 영상별로 별도 검증합니다. 타임라인이 없거나 ROI 방향이 검증되지 않은 기존 캐시는 지연 추정과 정량 검증에서 제외합니다.

최신 정량 검증은 4,373개 특징 프레임을 사용합니다. 단독 조건은 2 Hz, 짧은 response 영상은 4 Hz로 추출했고 상태 정확도는 일반 영상을 4초 간격으로 다시 샘플링해 실험별 가중치를 유지합니다. 전체 단계 모델은 아래의 독립 영상 홀드아웃 결과를 근거로 앱에 실험 배포하며, 정확 단계의 일반화 성능이 아직 충분하지 않기 때문에 단계 경계에서는 인접 범위를 표시합니다.

### 전체 농도 단계 검증

`training/ordinal_concentration_analysis.py`는 H2-only의 불꽃과 H2O-only의 물방울을 중심으로 농도 증가에 따른 보정 LAB 색 궤적을 비교합니다. 상대 형상 색은 프레임 공통 조명 변화를 제거하는 내부 기준으로만 사용합니다. H2와 RH 모두 ramp endpoint 사이의 연속 농도를 계산한 뒤 가장 가까운 출력 단계로 confusion matrix를 만들며, Recovery도 점진적 하강 타깃으로 포함합니다. 초기 상태와 반응 농도를 분리하는 2단계 후보도 같은 held-out 조건에서 비교하며, 희소 단계가 모델 선택에서 동일한 비중을 갖도록 stage-balanced accuracy를 사용합니다.

RH는 20%와 30%를 하나의 `20–30%` 구간으로 합치고 40–90%는 10% 단위로 유지합니다. 모든 video-held-out 평가는 테스트 영상의 0% calibration만 허용하고 그 영상의 나머지 프레임 전체를 학습에서 제외하는 `calibration-aware video-held-out`입니다. H2 정량 평가는 Reaction만 사용하며 정확도 46.1%(stage-balanced 48.8%, ±1% 이내 87.5%, MAE 0.67%p)입니다. RH endpoint ramp와 최종 RH20 calibration을 적용한 정확도는 36.9%(stage-balanced 35.0%, MAE 12.27%p), 동일 run 5초 블록은 정확도 52.1%(stage-balanced 52.1%, MAE 6.24%p)입니다. 선택된 전체 단계 모델은 `sensor-concentration-model.js`로 내보내며 Python 원본과 내보낸 트리/선형식의 예측 오차가 1e-9 이하인지 생성 시 자동 검사합니다.

H2의 긴 4% hold가 전체 점수를 부풀리는지 확인하기 위해 Reaction과 Recovery를 별도로 평가합니다. 회전 정렬된 불꽃 마스크에서 chroma 10/25/50/75/90 백분위수를 추가한 Reaction 전용 calibration-aware 모델은 정확도 46.1%, stage-balanced 48.8%, ±1% 이내 87.5%, MAE 0.67%p입니다. H2 1%와 2% 재현율은 각각 44.6%, 65.4%이며, 동일 run 5초 블록은 정확도 46.8%, MAE 0.57%p입니다. 모든 분포 통계를 넣으면 영상 질감에 과적합했기 때문에 held-out으로 확인된 chroma 백분위수만 유지합니다. Recovery는 독립 run이 두 개뿐이고 여전히 정량 성능이 부족해 모델에 합치지 않습니다. 단일 사진 앱에는 검증된 Reaction 모델만 후보로 두고, Recovery는 별도 상태 또는 불확실 결과로 취급합니다.

`training/h2_endpoint_range_analysis.py`는 구간 마지막 1초를 정확한 endpoint, 구간 내부를 농도 범위로 취급하고 4% hold와 baseline의 총 가중치를 제한합니다. 동일한 96개 endpoint 프레임에서 현재 Reaction 모델은 정확도 63.5%, stage-balanced 61.2%, ±1% 이내 94.8%, MAE 0.42%p였습니다. 범위 가중치와 규제를 outer test 영상 밖에서만 선택한 nested 범위 모델은 정확도 53.1%, MAE 0.54%p로 더 낮아 최종 모델로 채택하지 않습니다. Endpoint 평가는 별도의 엄격한 audit로 유지하고, recovery는 두 방식 모두 정량 학습에서 제외합니다.

`training/inspect_reference_patches.py`는 회전 정규화 뒤 위-오른쪽과 아래-왼쪽의 흰색/회색 패치를 직접 검출해 시각 QA를 생성합니다. 이 패치는 학습 영상 촬영 안정화를 위한 것이며 실제 앱 촬영에는 없으므로 배포 입력이나 모델 특징으로 요구하지 않습니다. 패치 색을 고정 LAB 값으로 강제하는 보정은 전체 ramp 점수를 소폭 높였지만 4% endpoint를 크게 낮췄고, 패치 제거 또는 두 패치 중심 미세 정렬도 held-out 성능을 낮췄습니다. 따라서 관련 실험은 audit 코드로만 보존하며 배포 특징 추출기는 기존 중성 픽셀 보정과 원형 ROI/quarter-turn 정렬을 유지합니다.

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

전체 단계별 색 궤적과 confusion matrix는 `training/output/ordinal_concentration/ordinal_concentration_validation`, H2 phase별 결과는 `h2_phase_validation`의 PNG(500 dpi), PDF, SVG로 생성되고 프레임별 예측은 같은 폴더의 `predictions.csv`에 저장됩니다.
H2 0/1/2%의 명목 단계 마지막 프레임을 같은 크기로 정렬한 판독용 montage는 `training/make_h2_low_stage_montage.py`로 생성합니다.
