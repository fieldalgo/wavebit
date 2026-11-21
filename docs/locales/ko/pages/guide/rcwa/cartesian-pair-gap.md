# 접촉과 양의 틈을 가로지르는 유한 쌍

실험적이며 NumPy 전용인 `CartesianPairGapCell`은 정확한 접촉을 포함해, 명시적으로 준 음이 아닌 물리적 틈에서
유한 쌍 하나를 평가합니다. 계수 또는 선택된 에너지 closure를 그 해석적 틈 미분과 함께 반환합니다.
`wavebit.rcwa._engine.conforming.cartesian_pair_gap`에서 가져오고, `wavebit.rcwa.experimental`의 기존
`CartesianPairOpeningPlan`을 주세요.

```python
import numpy as np
from wavebit.rcwa._engine.conforming.cartesian_pair_gap import CartesianPairGapCell
from wavebit.rcwa.experimental import CartesianPairOpeningPlan

geometry = dict(
    basis=np.array([[700.0, 315.0], [0.0, 490.0]]),
    widths=(140.0, 193.0),
    height=210.0,
    offset=(49.0, -21.0),
)
plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=1)
cell = CartesianPairGapCell(plan, order=(1, 2))
settings = dict(geometry, epsilon=(4.0 + 0.8j, 9.0 + 0.3j), background=1.0 + 0.1j)
length = np.sqrt(np.linalg.det(geometry["basis"]))

for gap in (0.0, length / 64):
    coefficients, derivative, metadata, details = cell.coefficients(gap=gap, **settings)
    slots, slot_derivative, closure_metadata, _ = cell.closure(gap=gap, **settings)
    assert coefficients.shape == derivative.shape == (9, 9, 5)
    assert all(slot.shape == (1, 15, 15) for slot in slots)
    assert all(slot.shape == (1, 15, 15) for slot in slot_derivative)
    assert metadata["derivative_units"] == "per caller length unit"
```

## 물리적 계열

주어지는 모든 길이는 나노미터처럼 호출자의 단위 하나를 사용합니다. `basis`는 열이 격자 벡터이고 행렬식이
유한한 양수인 유한한 실수 2×2(2-by-2) 행렬입니다. `widths=(left_width, right_width)`는 양수이며, 수평
직사각형들은 양수 `height` 하나를 함께 씁니다. `epsilon`은 대응하는 왼쪽/오른쪽 쌍입니다. 직사각형은
물리적인 카르테시안 모서리를 유지합니다.

`center = basis @ (.5, .5) + offset`이라 합시다. 두 직사각형 중심의 y 좌표는 이 값과 같고, x 좌표는
`center.x - (right_width + gap)/2`와
`center.x + (left_width + gap)/2`입니다. 틈을 바꾸면 바깥 셀은 고정되고, 왼쪽 직사각형은 틈 변화량의
절반만큼 왼쪽으로, 오른쪽 직사각형은 절반만큼 오른쪽으로 움직입니다. `offset`은 적분 셀 전체를 평행
이동합니다.

`gap`은 0 이상인 유한한 실수 스칼라입니다. 0에서 `derivative`는 오른쪽 미분입니다. 양의 틈에서는 다른
모든 설정을 고정한 국소 미분입니다. 둘 다 **호출자 길이 단위당** 값으로 반환됩니다.
이 명시적 계열은 [`CartesianRectanglePairNVF`](cartesian-rectangle-pair-nvf.md)의 접촉을 보존하는 JAX
계약을 바꾸지 않습니다. 이 인터페이스에는
`xp`, JAX 또는 자동 미분 모드가 없습니다.

## plan과 허용 조건

`CartesianPairOpeningPlan.from_geometry`로 접촉 상태에서 구체적인 plan을 만드세요.
plan은 순환 차트 네 개 중 하나를 고르며, `shift=0`, `1`, `2` 또는 `3`은 특정 차트를 요청합니다.
`refinement`의 기본값은 0이며 0부터 3까지의 정수를 받습니다.
[세분화 노트](../../validation/rcwa/nvf-pair-refinement.md)에는 더 세밀한 공간의 독립적인 검사, 자원
비용과 광학적 한계가 기록되어 있습니다. plan은 이 불변 선택만 저장하고, 셀은 고정된 위상을 재사용합니다.
모든 물리 기하와 재료는 셀을 평가할 때 줍니다.

매 평가마다 접촉 차트의 유효성, 엄격한 범위 포함, 요청한 틈에서의 양의 정칙 삼각형과 붕괴 삼각형의
양의 열림 기울기를 검사합니다. 차트가 유효하지 않게 되면 명시적으로 다시 계획해야 합니다.
평행사변형 안에 들어가는 틈이라도 삼각형 가드를 통과하지 못할 수 있습니다.
기하적으로 유효한 구간이라고 해서 균일한 조건수나 정확도가 보장되지는 않습니다.

기본적으로 `epsilon`과 `background`는 유한하고 엄격히 양수인 실수부와 음이 아닌 허수부를 가져야 합니다.
생성자에 명시적인 `coercivity_angle`을 주면 공통의 증명된 반평면에 있는 수동 금속도 허용합니다.
조건과 광학적 한계는 [재료 섹터](../../validation/rcwa/nvf-pair-material-sector.md)를 참고하세요.
복소 기하는 허수부가 0이어도 거부합니다. 음수, 불리언, 복소수 또는 유한하지 않은 틈 값, 유효하지 않은
기하/재료, 지원하지 않는 차수와 추가 키워드는 오류를 냅니다.
서로 다른 높이, 수직 이동, 회전한 직사각형, 겹침, 여러 쌍과 텐서 재료는 이 계열의 범위 밖입니다.

## 반환되는 계수와 진단

`order`의 기본값은 1이며, 0을 포함해 음이 아닌 정수 또는 `(x, y)` 쌍을 받습니다. 아홉 개의 필드는
`Bxx, Bxy, Byx, Byy, Gxx, Gxy, Gyy,
epsilon, Gyx`입니다. 모양은 `(9, 4*y+1, 4*x+1)`이며, y와 x 레이블은
필요한 계수 차이를 모두 포괄합니다. 위상은 물리적 셀 면적으로 정규화한
`exp(-2*pi*1j * [m,n] @ inv(basis) @ position)`입니다.

두 메서드 모두 `(value, derivative, metadata, details)`를 반환합니다. closure의 값과 미분은 층 축이 하나인
다섯 슬롯 `ConvMatrices` 튜플입니다. 역행렬을 구하기 전에 요청한 차이를 선택합니다. closure는 현재
설정에서 계수를 다시 계산하며, 캐시된 값으로 덮어쓰는 것은 받지 않습니다.

메타데이터에는 물리적 틈, `normalized_gap=gap/sqrt(det(basis))`, 면적, 정규화 길이, 차수, 차트, 개수,
원래 잔차와 사영된 잔차, 조건수가 기록됩니다. closure는 저장 에너지, 부호 있는 소산과 역행렬 잔차 검사도
보고합니다. details는 정규화된 행렬과 그 미분, 정규화된 꼭짓점, 필드와 푸리에 가중치를 노출합니다.
접미사 `_ds`는 정규화된 틈에 대한 미분을 뜻합니다. 이 진단 배열은 물리적 길이 단위를 쓰지 않습니다.
반환되는 계수와 closure의 미분은 물리적 길이 단위를 씁니다.

[방법 노트](../../validation/rcwa/nvf-cartesian-pair-gap.md)는 정확한 유한 재축척과 그 독립적인 적격성
확인을 설명합니다. 이 노트가 연속체나 패턴 층의 광학적 정확도를 보증하지는 않습니다. 별도의
[끝점 응답](cartesian-pair-opening.md)은 독립적으로 적격성을 확인한 기준으로 계속 사용할 수 있습니다.

## Fixed reference transmission potentials

`CartesianPairTransmissionGalerkinGapCell`은 전체 텐서와 그 해석적 틈/재료 JVP를 갖춘 별도의 실험적
NumPy 계열입니다. 대칭 카르테시안 쌍 메시 위의 shift 0 plan이 필요합니다.
`closure`는 `(value, tangent, metadata)`를 반환합니다. 두 행렬 집합 모두 `rcwa.operators`가 쓰는 좌표계인
셀 중심을 기준으로 하며, `offset`은 그 공개 원점에 대한 쌍의 위치를 정합니다. 위에서 설명한 비공개 구성은
기존 차트를 그대로 유지합니다.
셀과 열림 plan은 아래처럼 둘 다 `wavebit.rcwa.experimental`에서 사용할 수 있습니다.
[NVF와 보조 솔버 선택](nvf-method-selection.md)에서 이 구성을 고르거나 끄는 방법을 보여 줍니다.

```python
from wavebit.rcwa.experimental import (
    CartesianPairOpeningPlan, CartesianPairTransmissionGalerkinGapCell,
)

cell = CartesianPairTransmissionGalerkinGapCell(
    CartesianPairOpeningPlan(0), orders=(1, 1), degree=3,
    reference=(12.25, -10.0+1.0j, 1.0), branches=2,
    quadrature=80, angular_quadrature=80, coercivity_angle=1.53,
)
value, tangent, metadata = cell.closure(
    gap=0.0, basis=np.diag([700.0, 700.0]),
    widths=(300.0, 300.0), height=300.0,
    epsilon=(12.25, -10.0+1.0j), background=1.0,
    gap_tangent=1.0,
)
```

`reference=(left, right, background)`는 필수이며 보조 투과 계수를 고정합니다. 각 값은 유한하고 0이 아니어야
하며, 그로부터 나오는 정수가 아닌 모드는 유한한 에너지를 가져야 합니다. 바깥 모드는
`Re(lambda) > 0`을, 접합 모드는 열림 모서리 에너지가 적분 가능하도록 `Re(lambda) > 1/2`를 요구합니다.
허용되지 않는 기준 모음은 통째로 거부합니다. 물리 재료는 각
`closure` 호출에 속합니다. 물리 재료는 기준과 달라도 되며 `epsilon_tangent`와 `background_tangent`를 통해
변할 수 있습니다. 물리 재료를 바꿔도 기준 기저를 다시 맞추지는 않습니다. 그 JVP 동안 기하는 고정됩니다.

`branches`는 1부터 4까지의 정수이며 기본값은 2입니다. 정수가 아닌 기준 모드가 있는 모든 꼭짓점에서
차례로 커지는 근 `lambda`, `2-lambda`, `2+lambda`, `4-lambda`를 추가합니다. 실수부와 허수부는 따로
유지됩니다. 공통 다항식 차수는 3부터 8까지의 정수이며 기본값은 3입니다. 기존의 밀집 자원 한계
`degree * 2**plan.refinement <= 16`이 적용됩니다. 차수 5부터 8까지는 접촉에서 완전한 보상(compensated)
공통 푸리에 방정식과 자유 푸리에 방정식을 사용하며, 독립적인 물리 에너지 검사까지 같은 정밀도를
유지합니다. 차수 3과 4는 기존 평가 경로를 유지합니다. 양의 틈에서는 같은 선택된 퍼텐셜 공간과 그
완전한 물리적 변분을 사용합니다.

차수 5부터 8까지에서 `cutoff_order=1` 또는 `2`는 선택한 기준 모드에 `h*(1-h)**m`을 곱한 항을 그 차수까지
차례로 모두 추가합니다. 기본값 `0`은 원래의 꼭짓점 모자(corner-hat) 승수 `h`를 유지합니다.
이렇게 중첩된 공간과 차수 7/8의 모든 공간은 최종 2워드 조립 전에 완전한 필드를 3워드 산술로
결합합니다. 기준 계수와 물리 계수는 여전히 위에서 설명한 서로 다른 역할을 합니다.

반지름 방향 구적법의 기본값은 80이며 유효 차수는 최소 `20*max(degree, 4)`입니다(차수 5부터 8까지
각각 100, 120, 140, 160). `angular_quadrature=None`은 두 방향 모두에 그 유효 차수를 사용합니다.
각도 방향 차수를 명시하면 최소 `max(12, 2*max(order)+4)`여야 합니다.

메타데이터는 `closure='fixed_transmission_pair_galerkin'`을 밝히고, 기준 계수, 가지 개수와 실수 방향의
개수를 기록합니다.
[방법과 측정 노트](../../validation/rcwa/nvf-pair-transmission-potentials.md)는 유한한 물리 검사와 아직
해결되지 않은 광학적 수렴을 구분합니다.
