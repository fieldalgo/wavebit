# 아핀 프로파일로 만드는 conforming NVF

`rcwa.experimental.AffineProfileNVF`는 분수 좌표 띠(strip) 프로파일의 명시적인 아핀 상(image)에 대한
에너지 행렬을 만듭니다. 격자 벡터는 실제로 직교하지 않아도 됩니다. 이 빌더는 `rcwa.operators.solve`에
쓸 `ConvMatrices`를 반환하며, 이를 사용하는 문제와 같은 격자와 유지 푸리에 차수를 사용합니다.

```python
import numpy as np
from wavebit import Material, Units, rcwa
from wavebit.rcwa import experimental, operators

lattice = np.array([[700., 315.], [0., 490.]])
structure = [dict(background=1., profile=dict(
    width=.5,
    height=1.-17./700.,
    seams=[31./700.],
    values=[4., 9.],
    offset=(0., 0.),
))]
builder = experimental.AffineProfileNVF(
    (2, 1), rows=2, body_degree=2, gap_degree=2, grade_y=False,
)
conv = builder.conv_matrices(structure, lattice)
numerics = rcwa.Numerics(orders=builder.orders)
transmitted = []
for polarization in ('p', 's'):
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(
            lattice=lattice, top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
            # A background-only layer fixes the thickness; conv carries the profile.
            layers=(rcwa.Layer(thickness=300., background=Material(epsilon=1.)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(
            side='top', polarization=polarization, theta=.2, phi=0.),)))
    assembled = operators.assemble(problem, numerics=numerics).replace(epsilon=(conv,))
    transmitted.append(operators.solve(problem, assembled).T.sum())
print(*transmitted)  # p, then s
```

빌더의 행렬은 [`ConformingNVF`](conforming-nvf.md)에서 설명한 것처럼 조립된 층 연산자를 대체합니다.
각 편광은 따로 조립합니다.

이 예제는 유한 모델을 지정합니다. 보조 셀 해상도와 광학 푸리에 차수는 예시일 뿐이며, 계산하려는
관측값에 맞춰 각각 세분화해야 합니다.
[독립적인 광학 비교](../../validation/rcwa/nvf-affine-convergence.md)에 남아 있는 정확도 한계가
기록되어 있습니다.

## 좌표와 입력

주어진 격자를 열 벡터로 이루어진 A로 씁니다. 중심 기준 분수 좌표가 `(u, v)`인 점의 공개 카르테시안
위치는 셀 중심에서 잰 `A @ [u, v] + offset`입니다. 반환되는 행렬도 `rcwa.operators`가 쓰는 좌표계인
이 중심을 사용합니다. 행렬에 반 셀 위상을 더하거나 공개 필드 표본 좌표에 반 셀 이동을 더하지 마세요.
프로파일은 `-width/2 < u < width/2`와 `-height/2 < v < height/2`를 차지합니다.
재료 구간은 u가 증가하는 방향으로 이어지며 `seams`로 나뉩니다.
주기 셀의 나머지 부분은 층의 배경으로 채워집니다.

| 입력 | 계약 |
| --- | --- |
| `orders` | 음이 아닌 정수로 된 명시적인 `(Mu, Mv)` 쌍이며 기본값은 `(3, 3)`입니다. 정수 하나만 주면 거부합니다(`experimental.orders`). 행렬 closure 전에 요청한 모드를 정확히 선택하며, 어느 축에서든 0차도 포함됩니다. |
| `lattice` | `Lattice` 또는 행렬식이 양수인 유한한 실수 2×2(2-by-2) 배열. 벡터와 `offset`은 사용하는 문제의 길이 단위를 씁니다. 정수 기저 축소는 수행하지 않습니다. |
| `structure` | `background`와 `profile`을 가진 층 딕셔너리의 비어 있지 않은 시퀀스. 프로파일이 없거나 `None`이면 균일한 층입니다. 배경의 기본값은 1입니다. |
| `width` | 필수. 1보다 엄격히 작은 양의 분수 폭. |
| `height` | 필수. 최대 1인 양의 분수 폭. 높이 1은 주기 경계를 가로지르는 정확한 접촉입니다. 그 너머로는 기존의 접촉 반올림 허용치만 받습니다. |
| `seams` | 필수. 폭 범위 안에 있는 분수 u 위치들의 정렬된 1차원 시퀀스. 재료가 하나이면 빈 시퀀스를 쓰세요. 겹치는 이음매는 폭이 0인 구간과 그 열림 미분을 유지합니다. |
| `values` | 필수. 물리적 스칼라 재료의 시퀀스로, 이음매 개수보다 하나 많습니다. 각 값은 유전율이거나 `Material(epsilon=...)`일 수 있습니다. |
| `offset` | 선택. 셀 중심으로부터의 실수 카르테시안 변위이며 기본값은 `(0., 0.)`입니다. 평행 이동에는 제한이 없습니다. |
| 재료 | 유한한 스칼라, 비자성, 허수부가 음이 아닌 양의 실수 유전율. 배경과 각 구간의 손실 비율은 서로 달라도 됩니다. |

폭, 높이와 이음매는 분수 값이고, offset은 물리적 카르테시안 변위입니다. 예제에서 분수 틈 `17/700`은
격자의 길이 단위로 법선 방향 물리 폭 11.9에 해당합니다. 일반적으로 그 폭은
`(1-height) / norm(inv(A).T @ [0, 1])`입니다.
기준 직사각형은 전단(shear)을 받으면 물리적인 평행사변형이 됩니다.
일반적인 물리 직사각형 딕셔너리에는 [`ConformingNVF`](conforming-nvf.md)와 그 지원 격자 계약을
사용하세요. 아핀 프로파일 빌더는 일반 형상 딕셔너리를 분수 프로파일로 해석하지 않고, 알 수 없는 키를
거부합니다.

## 미분과 행렬 사용

빌더는 `jax.jit` 바깥에서 만드세요. 그러면 기하, offset, 격자와 재료 값을 JAX x64와 함께
`builder.conv_matrices(structure, lattice, xp=jax.numpy)`에 실제(live) 값으로 넘길 수 있습니다.
명시적으로 정렬된 구간에는 정렬 계획이 필요 없습니다. 컴파일된 호출 안에서는 층 개수와 프로파일 배열
모양을 고정하세요. 즉시 실행에서 유효하지 않은 기하는 오류를 내고, 추적되는 수치 값이 유효하지 않으면
NaN 행렬이 나옵니다. NumPy 구성도 64비트 정밀도를 사용합니다. 내보낸 행렬은 `backend='torch'`로 조립한
연산자에 쓰도록 Torch 텐서로 변환할 수 있습니다.

반환되는 푸리에 행렬은 물리적 offset을 포함해 셀 중심을 원점으로 사용합니다.
셀은 이미 실험실 좌표계로 필드를 반환합니다. 어댑터는 행렬을 닫기 전에 에너지 계수와 종방향 재료
계수에 물리적 배경을 되돌려 넣습니다. 무손실 저장과 수동 손실은 내보내는 행렬을 바꾸지 않고 검사합니다.

정확한 적층 접촉 값과 유한 셀 열림 미분에는
[독립적인 대조 검사](../../validation/rcwa/nvf-affine-cells.md)가 있습니다. 이 검사들은 연속체 열림 미분의
수렴이나 광학적 정확도를 보증하지 않습니다. 사각이 아닌 격자 위의 임의의 물리 직사각형, 곡선, 겹침,
텐서 또는 자성 재료, 이득, 실수부가 양수가 아닌 유전율은 이 어댑터의 계약 범위 밖입니다.
