# NVF와 보조 솔버 선택

행렬 구성 방식을 명시적으로 선택하세요. 일반 NVF는 실험적 투과 셀을 호출하지 않으며,
보조 계산이 실패해도 다른 구성 방식으로 대체(fallback)하지 않습니다. 보조 셀은 다른
구성(constitutive) 행렬을 제공합니다. 일반 NVF 계산에 덧붙이는 보정이 아닙니다.

| 선택 | 선택 방법 | 범위 |
| --- | --- | --- |
| 일반 NVF, 보조 솔버 끔 | `rcwa.run(problem, numerics=rcwa.Numerics(factorization='nvf'))` | 기존 래스터 또는 지원되는 형상 NVF 경로와 그 선택된 사영자(projector). |
| NVF 끔, 명시적 카르테시안 Li | `rcwa.run(problem, numerics=rcwa.Numerics(factorization='li'))` | 기존 래스터 지원 또는 허용된 축 정렬 직사각형 형상. 사용자가 고르는 것이며, NVF의 자동 대체가 결코 아닙니다. |
| 실험적 보조 구성 | `rcwa.experimental.CartesianPairTransmissionGalerkinGapCell`을 만들고, 그 행렬을 `rcwa.operators.assemble`이 만든 연산자 대신 넣은 뒤 `rcwa.operators.solve`를 호출합니다 | 수평으로 놓인 높이가 같은 직사각형 쌍 하나에 대한 NumPy 유한 셀로, 해석적인 틈/재료 JVP를 갖습니다. |

세 번째 선택은 실험적입니다. 내부 물리 검사는 모두 유지하지만,
광학적 수렴과 실용적인 실행 시간은 아직 해결되지 않았습니다. 지원하는
기하와 미분은 일반 NVF보다 좁습니다. 선택한 차수나 메시가 정확도를
보증하지는 않습니다.
[측정과 적격성 노트](../../validation/rcwa/nvf-pair-transmission-potentials.md)를 참고하세요.

## 보조 솔버의 명시적 호출

```python
import numpy as np
from wavebit import Material, Units, rcwa
from wavebit.rcwa import experimental, operators

basis = np.diag([700., 700.])
orders = (1, 1)
cell = experimental.CartesianPairTransmissionGalerkinGapCell(
    experimental.CartesianPairOpeningPlan(0, refinement=0), orders,
    reference=(12.25, -10.+1j, 1.), degree=3, branches=2,
    quadrature=80, angular_quadrature=48, coercivity_angle=1.53,
)
matrices, gap_jvp, diagnostics = cell.closure(
    gap=0., basis=basis, widths=(300., 300.), height=300.,
    epsilon=(12.25, -10.+1j), background=1., gap_tangent=1.,
)
problem = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(
        lattice=basis, top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        # A background-only layer fixes the thickness; the matrices carry the pair.
        layers=(rcwa.Layer(thickness=300., background=Material(epsilon=1.)),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(
        side='top', polarization='s', theta=.2, phi=0.),)))
assembled = operators.assemble(problem, numerics=rcwa.Numerics(orders=orders),
                               backend='numpy').replace(epsilon=(matrices,))
answer = operators.solve(problem, assembled)
```

이 예제의 낮은 푸리에 차수와 다항식 차수는 호출 방법을 보여 줄 뿐,
수렴한 광학 결과가 아닙니다. `gap_jvp`는 구성 행렬의 미분이며,
투과나 반사의 미분이 아닙니다. `operators.solve`는 주어진 행렬을 그대로
사용합니다. 조립에 쓰인 `rcwa.Numerics`의 인수분해 설정은 그 행렬을 다시
만들거나 끄지 않습니다. 보조 구성을 끄려면 쌍을 층 영역으로 선언하고,
대신 `rcwa.run`과 명시적인 인수분해로 문제를 실행하세요.

이 셀은 양의 대각 격자, 엄격하게 셀 안에 들어 있는 높이가 같은
수평 직사각형, 음이 아닌 틈과 허용된 수동 스칼라 재료를 받습니다.
높이가 다른 경우, 임의의 형상, 겹침, 여러 쌍과 텐서 재료는
이 보조 계열의 범위 밖입니다. 셀 구성은 NumPy 전용이며, 이 계열에는
JAX 추적이나 일반적인 형상 그래디언트 인터페이스가 없습니다.
[쌍 가이드](cartesian-pair-gap.md#fixed-reference-transmission-potentials)는
기준 재료, 지원되는 JVP 방향, 차수, 구적법과 자원 한계를 정의합니다.
명시적으로 선택하면 그 실행 시 검사도 계속 활성화됩니다.

## 같은 예제에서 전환하기

실행 가능한 [선택 예제](../../../examples/rcwa/nvf_method_selection.py)는
명시적인 방법 인수를 제공하며, 계산 전에 선택된 방법을 출력합니다.

```shell
python examples/rcwa/nvf_method_selection.py --method nvf
python examples/rcwa/nvf_method_selection.py --method li
python examples/rcwa/nvf_method_selection.py --method auxiliary --degree 3
```

이는 예제 스크립트의 옵션이며 `rcwa.Numerics`의 새 키워드가 아닙니다.
스크립트는 현재의 물리 Problem API를 사용하고, 보조 행렬에는 명시적인 `rcwa.operators`를 사용합니다.
스크립트의 기본값은 일반 원형 NVF입니다. 차수와 메시 세분화 인수는
보조 선택에서만 받으며, 무시될 설정이나 알 수 없는 방법은 거부합니다.
세 실행 모두 같은 물리적 쌍과 광학 설정을 사용합니다. 출력되는 경과
시간에는 구성과 선택된 경로의 내부 검사가 포함됩니다. 낮은 차수의 실행 한 번은
조건을 맞춘 정확도 벤치마크도, 접촉 광학에 대한 증명서도 아닙니다.
