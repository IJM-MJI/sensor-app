# Place-2 RH seven-band timing-policy audit

## 목표

프레임 추출 시점을 바꾸는 것만으로 7구간 RH exact accuracy를 0.85 이상으로
올릴 수 있는지 확인했다. 정답을 본 뒤 개별 프레임을 고르는 누수를 피하기 위해 다음
9개 규칙을 미리 고정하고 `response3 ↔ response6` complete-run holdout으로 평가했다.

- 구간별 후보 중 `earliest`, `middle`, `latest`
- `single`, 직전 3프레임 median(`trailing3`), 중앙 3프레임 median(`centered3`)

## 결과

| 정책 | exact | balanced | 인접 한 구간 이내 |
|---|---:|---:|---:|
| earliest single | 0.571 | 0.571 | 0.857 |
| earliest trailing3 | 0.500 | 0.500 | 0.929 |
| earliest centered3 | 0.571 | 0.571 | 0.857 |
| middle single | 0.429 | 0.429 | 0.857 |
| middle trailing3 | 0.500 | 0.500 | 0.929 |
| middle centered3 | 0.357 | 0.357 | 0.929 |
| latest single | 0.500 | 0.500 | 0.857 |
| **latest trailing3** | **0.643** | **0.643** | **0.929** |
| latest centered3 | 0.500 | 0.500 | 0.929 |

최선인 `latest_trailing3`의 held-out exact는 response3 `0.714`, response6
`0.571`이다. 구간 recall은 20–30 `0.50`, 30–40 `1.00`, 40–50 `0.50`,
50–60 `0.00`, 60–70 `1.00`, 70–80 `0.50`, 80–90 `1.00`이다.

## 판정

프레임 시간을 늦추고 직전 3프레임을 쓰면 MAE는 `4.29 %RH`, 인접 한 구간 이내는
`0.929`로 개선된다. 그러나 exact/balanced 및 모든 구간 recall `0.85` 조건은
통과하지 못한다. 따라서 이 정책을 앱에 배포하지 않는다.

기존 full-trajectory 감사에서도 두 run의 광학 경계 이동은 -8.3에서 +10.7 %RH로
부호가 달라 하나의 공통 시간 지연으로 설명되지 않았다. 남은 병목은 단일 불량 프레임보다
run별 색 경로와 반응 크기 차이다. 시간을 정답별로 개별 선택하면 같은 데이터에서는 높은
수치를 만들 수 있지만 독립 영상 정확도로 인정할 수 없다.

0.85 목표를 정직하게 시험하려면 프로필별 새 run이 필요하다. 첫 run에서 안정화 지연과
추출 규칙을 한 번 고정하고, 두 번째 run 전체에서 7구간을 평가해야 한다. 현재 자료에서는
`latest_trailing3`을 새 run의 사전 고정 후보로만 사용한다.
