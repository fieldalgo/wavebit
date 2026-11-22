# 사각 격자가 아닌 격자 위의 두 카르테시안 직사각형

`rcwa.experimental.CartesianRectanglePairNVF`는 패턴 층마다 수평으로 놓인 높이가 같은 두 물리 직사각형에
대한 유한 NVF 에너지 행렬을 만듭니다. 필수인 고정 plan 위에서 양의 틈 또는 전체 높이에 걸친 정확한
접촉을 지원합니다. 사각이 아닌 격자 위에서도 직사각형은 카르테시안 모서리를 유지합니다. 이 명시적인
실험적 빌더는 `rcwa.operators.solve`에 쓸 `ConvMatrices`를 제공하며, 광학적 수렴을 보증하거나 다른 기하
빌더를 바꾸지 않습니다.

```python
import numpy as np
from wavebit import Material, Units, rcwa
from wavebit.rcwa import experimental, operators

lattice = np.array([[700., 315.], [0., 490.]])
structure = [dict(background=1., shapes=[
    dict(lx=140., ly=210., cx=-56., cy=-21., material=4.),
    dict(lx=193., ly=210., cx=127.5, cy=-21., material=9.),
])]
builder = experimental.CartesianRectanglePairNVF((1, 2), refinement=1)
plan = builder.plan(structure, lattice)
conv = builder.conv_matrices(structure, lattice, plan=plan)
numerics = rcwa.Numerics(orders=builder.orders)
transmitted = []
for polarization in ('p', 's'):
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(
            lattice=lattice, top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
            # A background-only layer fixes the thickness; conv carries the pair.
            layers=(rcwa.Layer(thickness=300., background=Material(epsilon=1.)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(
            side='top', polarization=polarization, theta=.2, phi=0.),)))
    assembled = operators.assemble(problem, numerics=numerics).replace(epsilon=(conv,))
    transmitted.append(operators.solve(problem, assembled).T.sum())
print(*transmitted)  # p, then s
```

예제의 틈은 17단위이고 경계 상자 offset은 `(49, -21)`입니다. 모든 길이, 파장과 두께는 같은 단위를
써야 합니다. 이를 사용하는 문제에서 원래 격자와 같은 푸리에 차수를 선언하세요.
[`ConformingNVF`](conforming-nvf.md)에서 설명한 것처럼, 문제에서 조립된 층 연산자는 빌더의 행렬로
대체됩니다. 보조 메시 세분화와 광학 푸리에 차수는 서로 다른 해상도입니다. 현재 한계는
[유한 셀 대조 검사](../../validation/rcwa/nvf-cartesian-pair.md)에 설명되어 있습니다.

## 받는 입력

| 입력 | 계약 |
| --- | --- |
| `orders` | 음이 아닌 정수로 된 명시적인 `(Mu, Mv)` 쌍으로, 두 격자 벡터 방향의 반폭이며 기본값은 `(1, 1)`입니다. 정수 하나만 주면 거부합니다(`experimental.orders`). 한 축 또는 두 축 모두 0이어도 됩니다. 닫힘(closure) 전에 원래의 차이 모드 집합을 정확히 선택합니다. |
| `refinement` | 0부터 3까지의 정수이며 기본값은 0. 떨어진 쌍의 자유 퍼텐셜 미지수는 8/35/143/575개, 접촉 쌍은 6/27/111/447개입니다. [세분화 제어와 광학적 한계](../../validation/rcwa/nvf-pair-refinement.md)를 참고하세요. |
| `lattice` | `Lattice` 또는 열이 물리 기저 벡터이고 행렬식이 유한한 양수인 유한한 실수 2×2(2-by-2) 행렬. 격자 축소는 수행하지 않습니다. |
| `structure` | `background`와 `shapes`만 담은 딕셔너리의 비어 있지 않은 리스트 또는 튜플. 배경의 기본값은 진공입니다. |
| 균일한 층 | `shapes`를 생략하거나 빈 리스트 또는 튜플을 주세요. 이 층을 구성할 때도 plan이 필요합니다. |
| 패턴 층 | `rectangle` 형상 정확히 두 개. `rectangle`은 기본 유형이기도 합니다. `square`를 포함한 다른 이름은 거부합니다. |
| 길이 | 양의 물리 길이 `lx`와 `ly`. 각각 없으면 따로 `l`로 대체됩니다. 명시적인 `lx`/`ly`가 우선합니다. 두 높이는 정확히 같아야 합니다. |
| 자세 | `angle`은 0이어야 합니다. 물리 카르테시안 `cx`와 `cy`의 기본값은 0입니다. 두 `cy` 값은 정확히 같아야 하고, 왼쪽과 오른쪽 중심은 서로 달라야 합니다. |
| 재료 | 스칼라 물리 유전율 또는 `Material(epsilon=...)`. 유한하고 실수부가 양수, 허수부가 음이 아니어야 합니다. 선언된 투자율, 텐서와 이득은 거부합니다. |

계획 단계는 입력 시퀀스를 바꾸지 않고 물리적인 왼쪽 직사각형과 오른쪽 직사각형을 식별합니다. 폭
`wL`, `wR`와 중심 `xL < xR`에 대해 다음을 사용합니다.

```text
gap = (xR - xL) - (wL + wR) / 2
offset_x = (xL + xR) / 2 + (wR - wL) / 4
offset_y = common cy
```

경계 직사각형은 그 쌍을 중심으로 한 기본 셀 안에 엄격하게 들어가야 하며, 선택된 차트에서 양의 유한한
삼각형을 가져야 합니다. 셀 전체가 쌍의 offset과 함께 평행 이동하므로, 포함 검사를 바꾸지 않고 호출자의
셀 경계를 넘을 수 있습니다. 반환되는 행렬은 `rcwa.operators`가 쓰는 좌표계인 셀 중심을 기준으로 하며,
offset은 쌍의 공개 중심입니다.
반환된 행렬에 위상, 재료 정규화 또는 회전을 추가로 적용하지 마세요.

**계산된** 틈은 엄격히 양수이거나 정확히 0이어야 합니다. 접촉으로 스냅하는 일은 없습니다. 모서리를
공유하려고 했더라도 부동소수점 중심 계산 때문에 아주 작은 양의 틈이 남으면 떨어진 쌍으로 취급하고,
음의 틈은 거부합니다. 예제의 단위에서 정확한 접촉은 폭 `(140, 193)`과 중심 `(-47.5, 119)`로 지정할 수
있습니다. 서로 다른 높이, 서로 다른 수직 중심, 0이 아닌 각도, 겹침, 부분 모서리/꼭짓점 접촉, 추가 형상과
알 수 없는 키는 거부합니다. 일반적인 쌍 배치를 모두 다룬다고 주장하지 않습니다.

## 계획과 실제 값에 대한 미분

JAX 추적 전에 구체적인 입력으로 `plan(structure, lattice)`를 호출하세요. 불변 plan은 층 종류, 물리적인
왼쪽에서 오른쪽 순서로 정렬한 원래 입력 인덱스, 접촉 상태, 순환 메시 선택과 세분화를 저장합니다. 실제
기하나 재료 값은 저장하지 않습니다. 모든 `conv_matrices(..., plan=plan)` 호출에 plan을 명시적으로 주세요.
입력 순서, 층 종류, 접촉 상태 또는 세분화를 바꾸면 그에 맞는 새 plan이 필요합니다. 재실행은 미분되는
계산 안에서 형상을 정렬하거나 계획기를 실행하지 않습니다.

NumPy 또는 JAX x64에서는 격자 항목, 폭, 공통 높이, 공통 수직 중심, 쌍 offset과 물리 재료가 실제(live)
값으로 남습니다. 양의 틈 쌍은 틈이 양수로 유지되는 동안 틈 변화도 허용합니다. 접촉에서의 미분은 공유
모서리를 보존해야 합니다. 공유 높이 매개변수 하나와 공유 수직 중심 매개변수 하나를 쓰고, 접촉에서
`d(xR - xL) = (dwL + dwR) / 2`를 만족하는 중심 접선을 사용하세요. 높이 하나, 수직 중심 하나 또는 접촉 틈만
움직이는 임의의 접선은 primal 점이 유효하더라도 이 제약된 계열의 범위 밖입니다. 반올림이나 스냅 연산을
통한 미분은 제공하지 않습니다.

즉시 실행에서 유효하지 않은 입력은 오류를 냅니다. 추적되는 수치 값이 유효하지 않으면 해당 층의 행렬이
NaN으로 오염되며, 유효하지 않은 격자는 균일한 층을 포함해 모든 층을 오염시킵니다. JIT 안에서의 계획은
거부합니다. 포함 관계, 양의 삼각형, 원래의 약형 잔차와 유한 저장/소산 검사는 계속 활성화됩니다. 재실행
메시가 양이라고 해서 순환 계획 선택을 가로지르는 연속성이 증명되지는 않습니다. 이것은 유한 공간에서의
미분이며, 접촉 열림 극한이나 패턴 층의 광학적 정확도 보장이 아닙니다.
