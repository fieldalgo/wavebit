# 유한 접촉 셀 안의 수동 금속

작성: 2026-09-13.

대상:
- `src/wavebit/rcwa/_engine/conforming/material_sector.py`
- `src/wavebit/rcwa/_engine/conforming/cartesian_pair_opening.py`
- `src/wavebit/rcwa/_engine/conforming/cartesian_pair_gap.py`
- `src/wavebit/rcwa/_engine/conforming/cartesian_pair_gap_autodiff.py`
- `tests/rcwa/test_nvf_pair_material_sector.py`
- `studies/rcwa/nvf_pair_material_sector/nvf_pair_material_sector.py`
- `studies/rcwa/nvf_pair_material_sector/records/nvf_pair_material_sector.log`

실험적인 유한 쌍 계열은 이제 명시적으로 주어진 복소 반평면 하나가 약형식 방정식을 보증할 때
실수부가 음수인 수동 재료를 받아들입니다.
이것은 기존의 정확한 접촉/열림 대수를 금속으로 확장하며, NumPy 해석 미분과 JAX 틈 미분도
포함합니다. 하지만 일반적인 거리 혼합 법선장을 고치거나 유한 셀의 광학 정확도를 인정하는 것은
**아닙니다**.
아래 광학 비교는 이 성긴 셀들을 그 대체물로 쓰는 것을 기각합니다.

## 물리 방정식은 그대로 두는 보증서

원래의 스칼라 약형식에 대해 `z=exp(-i*theta)` 로 두고 다음을 요구합니다.

```text
Im(epsilon_j) >= 0,
Re(z*epsilon_j) > 0 for every material, including background.
```

상수가 아닌 임의의 유한 퍼텐셜 u에 대해 회전한 에너지의 실수부는
`sum_j Re(z*epsilon_j) integral_j |grad u|^2` 이며, 상수 게이지를 고정하면 엄밀히 양수입니다.
따라서 회전하지 않은 실수 유전율이 양수일 필요 없이 강압성(coercivity)을 얻습니다. 접촉 대각
블록과 붕괴하는 차이 대각 블록도 같은 성질을 물려받습니다. 틈이 양수일 때는 올바르게 다시
스케일한 합동 변환이 물리 에너지입니다. 대칭이 아닌 원래 틈 행렬의 에르미트 부분은 여전히 그
에너지를 검사하는 올바른 방법이 아닙니다.

위상은 **허용 판정과 진단에만** 들어갑니다. 실제 약형식 행렬, 우변, 필드, 에너지 계수, 종방향
유전율과 미분은 주어진 물리 재료를 그대로 씁니다. 손실을 더하거나, 재료를 잘라 내거나, 재료를
켤레로 바꾸거나, 기하를 스냅하거나, 광학 매개변수를 맞추는 일은 없습니다. 유효한 보증 각도
사이에서 바꾸어도 NumPy 계수, 닫힘(closure)과 해석 미분은 비트 단위로 같습니다.

셀의 전기장 변수 변환을 `B` 라고 하면, 물리 에너지는 `epsilon B^H B` 의 푸리에 압축입니다.
압축된 B의 역행렬로 합동 변환해도 음이 아닌 물리적 소산은 보존됩니다. 양수인 **회전한** 저장
에너지도 보존됩니다. 더 정확히 말하면, 직교 푸리에 사영은
`integral |B u|^2 >= |Fourier(B) u|^2` 를 줍니다. 따라서 압축된 변환이 가역이면 회전한 닫힘은
`min_j Re(z*epsilon_j) I` 로, 그 소산 부분은 `min_j Im(epsilon_j) I` 로 아래에서 유계입니다.
독립 테스트가 두 하한을 모두 확인합니다. 회전하지 않은 저장 에너지가 음수이면 그대로 보고하며,
양수로 다시 이름 붙이거나 광학 이득과 혼동하지 않습니다.

## 명시적 사용법과 실패 시 동작

`CartesianPairOpeningCell`, `CartesianPairGapCell`, `CartesianPairGapEvaluator` 는 라디안 단위의
정적 생성자 키워드 `coercivity_angle` 를 받습니다. 기본값 0은 실수부가 양수인 수동 영역을
유지합니다. `[0, pi]` 안의 값으로 다른 수동 섹터를 보증할 수 있습니다. 예를 들면 다음과 같습니다.

```python
import numpy as np
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_gap import CartesianPairGapCell

geometry = dict(basis=np.diag([700.0, 700.0]), widths=(300.0, 300.0), height=300.0)
plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=1)
cell = CartesianPairGapCell(plan, order=4, coercivity_angle=1.53)
slots, derivative, metadata, _ = cell.closure(
    gap=0.0, epsilon=(12.25, -10.0 + 1j), background=1.0, **geometry
)
assert metadata["minimum_rotated_storage"] > 0
assert metadata["minimum_dissipation"] > -1e-9
```

같은 생성자 키워드가 [틈 미분 가이드](../../guide/rcwa/cartesian-pair-gap-autodiff.md)의 JAX 값
평가기에도 적용됩니다. 각도는 추적(tracing) 전에 고정됩니다. 재료 값은 계속 추적되는 값으로
남으며, 평가할 때마다 보증 조건을 검사합니다. 스윕이 선택한 반평면을 벗어나면 각도를 자동으로
바꾸지 않고 거부합니다. 이번 증분은 틈 접선을 인정합니다. 임의의 기하/재료 접선과 고계 미분은
이번에 새로 한 독립 점검의 범위 밖입니다.

각도가 0이 아니면 정규화한 재료 사영이 `64 * float64.eps` 보다 커야 합니다. 이것은 반올림 오차를
위한 허용 여유이며, 방정식 안의 스무딩 매개변수가 아닙니다. 이 여유 덕분에 `theta=pi/2` 에서
삼각함수 반올림 오차 때문에 부호가 반대인 무손실 재료가 허용되는 일을 막습니다. 이 방법은 양수와
음수가 섞인 무손실 스펙트럼을 허용하지 않습니다. 균일한 음수 매질을 포함해, 모두 음수인 무손실
재료는 `theta=pi` 로 보증할 수 있습니다. 이득, 0인 재료, 유한하지 않은 값과 유효하지 않은 각도는
거부합니다. 유효하지 않은 재료 값은 즉시 실행에서는 바로 예외를 내고, JIT 아래에서는 안전한 중간
계산 뒤에 계수/슬롯을 모두 NaN으로 만듭니다. 원래의 잔차와 역행렬 허용 오차는 바뀌지 않았습니다.

메타데이터는 `minimum_storage` 와 기존의 회전하지 않은 블록 값을 보존하고,
`minimum_rotated_storage`,
`contact_minimum_rotated_eigenvalue`,
`difference_minimum_rotated_eigenvalue`, `coercivity_angle` 를 더합니다.
일반 `CartesianRectanglePairNVF` 어댑터, 그 재료 영역, 일반 NVF 디스패치와 기하 허용 판정은 이
확장의 범위 밖에 남습니다.

## 유한 계열의 독립 검증

새 테스트 40개에는 정규화한 틈 `0, 1/64, 1/16` 에서 선언된 차트 네 개 위의 금속 사례 12개가
들어 있으며, 세분 수준 0/1, 서로 다른 기울어진 기하 두 개, 부호가 있는 기저/서로 다른 차수 대조를
포함합니다. 이 테스트들은 후보인 다시 스케일한 약형식 행렬이 아니라, 원래의 독립적인 유리
절점/Laurent 방정식과 움직이는 심플렉스 Duffy 적분을 다시 씁니다. q16/q24 기준은 비교 전에
서로 일치해야 합니다. 계수와 닫힘 값은 `3e-10` 절대 기준을 유지하고, 셀 정규화 길이를 곱한 틈
미분은 `3e-9` 를 유지합니다.
약형식 잔차는 `1e-10` 을 유지합니다. 별도의 끝점 구현과 실제 JIT 틈 접선은 접촉에서 그 기준과
일치합니다.

추가 테스트는 손실이 있거나 없는 균일한 음수 매질, 스칼라 하한, 보증 각도 불변성, 이득/경계
기각과 유효하지 않은 JIT 출력을 다룹니다. 기존 끝점/틈/자동 미분 테스트 347개도 통과합니다. 외부
코드, 의존성이나 기준 배열은 추가하지 않았습니다. 이 확장은 저장소의 기존 NumPy/JAX 구현과 해석적
약형식 방정식을 씁니다.

## 광학 비교와 결정

정사각형 셀은 700이고, 중심이 x 오프셋 -150/+150에 있는 300 정사각형 상자 두 개, 층 두께 300,
파장 900, 위/아래 epsilon 1/2.25, theta .2, phi 0입니다. 틈이 양수이면 두 직사각형이 대칭적으로
바깥쪽으로 움직입니다.
유한 셀은 `coercivity_angle=1.53` 을 쓰고, 표본화한 법선 격자 없이 바뀌지 않은 조화 에너지 닫힘을
씁니다. 카르테시안 Li는 같은 물리적 칠하기를 독립적으로 적분하고 같은 Maxwell 솔버를 씁니다.

재료 `12.25 | -10+i` 의 접촉에서:

| 차수 | 셀 세분 | p T | Li p T | s T | Li s T |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | 0 | .429988475 | .483946395 | .471708640 | .480212354 |
| 4 | 1 | .248701662 | .483946395 | .367536644 | .480212354 |
| 8 | 0 | .303281789 | .480976241 | .417832833 | .402781608 |
| 8 | 1 | .405985788 | .480976241 | .356434074 | .402781608 |

이 셀들은 광학적으로 수렴하지 않았습니다. 셀을 세분하면 유한 차수 Li와의 거리가 오히려 나빠질 수
있으며, Li 자체도 차수에 따른 변화가 큽니다.
기록된 행렬 일치와 틈 미분 일치로는 어느 쪽의 연속체 광학 값도 정확한 기준으로 확립할 수 없습니다.

기존의 유한 구성은 이미 같은 재료 사이의 거리장 첨점(cusp)을 없앱니다. 재료가 모두 4이고 차수
4/세분 1일 때 p
`T(1e-6)-T(0)=1.60e-11` 입니다. 셀에서 분해한 2차 접촉 기울기는 길이 단위당 약 `1.60e-5` 인
반면, 같은 차수의 Li는 `6.57e-6` 을 줍니다. s 기울기는 `-5.83e-5` 대 `-1.58e-4` 입니다. 따라서
접촉 잔차가 작다는 것만으로는 이번에도 물리적 미분을 보증하지 못합니다. 이제 금속 접촉 값도
유한하고 미분 가능하지만, 그 광학 기울기는 세분 수준 사이에서, 그리고 Li와 비교해서 크게 다르며
때로는 부호까지 다릅니다.

결과 로그에는 두 편광, 모든 틈, 2차 한쪽 할선 창 두 개, 에너지/약형식 진단과 소스 해시가 남아
있습니다. `1e-9` 값은 전파 반올림 오차 수준에 이를 수 있으며, 보고한 기울기를 정의하지 않습니다.
이 구현은 이후 작업을 위해, 조건부로 인정한 금속 지원 유한 접촉 계열을 제공합니다. 위 차이를
해소하는 셀 수렴과 광학 수렴 결과가 나올 때까지 기본 통합에서는 제외합니다.

```sh
PYTHONPATH=. python -m pytest -q tests/rcwa/test_nvf_pair_material_sector.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python studies/rcwa/nvf_pair_material_sector/nvf_pair_material_sector.py --orders 4 8
```


이후의 [세분 확장](nvf-pair-refinement.md)은 수준 2/3을 허용하고, 값/JAX 약형식 블록을 더 적은
저장 공간으로 축약합니다. 그 공간에서도 독립적인 유한 미분은 통과하지만, 확장한 금속 광학 비교는
여전히 수렴을 확립하지 못합니다. 위의 0/1 표는 원래의 재료 허용 근거로 남습니다.
