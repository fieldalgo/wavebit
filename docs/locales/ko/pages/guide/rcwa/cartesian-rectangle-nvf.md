# 사각 격자가 아닌 격자 위의 카르테시안 직사각형 NVF

`rcwa.experimental.CartesianRectangleNVF`는 패턴 층마다 서로 떨어진 물리 직사각형 하나에 대한 유한
NVF 에너지 행렬을 만듭니다. 사각이 아닌 격자 위에서도 직사각형 모서리는 물리적인 길이와 각도를
유지합니다. 이 실험적 빌더는 `rcwa.operators.solve`에 쓸 `ConvMatrices`를 반환하며, 유한한 보조 해상도가
광학적 수렴을 보증하지는 않습니다.

```python
import numpy as np
from wavebit import Material, Units, rcwa
from wavebit.rcwa import experimental, operators

lattice = np.array([[700., 315.], [0., 490.]])
structure = [dict(background=1., shapes=[dict(
    type='rectangle', lx=350., ly=300., angle=3.45,
    cx=49., cy=-21., material=4.,
)])]
builder = experimental.CartesianRectangleNVF((1, 2), refinement=1)
plan = builder.plan(structure, lattice)
conv = builder.conv_matrices(structure, lattice, plan=plan)
numerics = rcwa.Numerics(orders=builder.orders)
transmitted = []
for polarization in ('p', 's'):
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(
            lattice=lattice, top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
            # A background-only layer fixes the thickness; conv carries the rectangle.
            layers=(rcwa.Layer(thickness=300., background=Material(epsilon=1.)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(
            side='top', polarization=polarization, theta=.2, phi=0.),)))
    assembled = operators.assemble(problem, numerics=numerics).replace(epsilon=(conv,))
    transmitted.append(operators.solve(problem, assembled).T.sum())
print(*transmitted)  # p, then s
```

빌더와 이를 사용하는 문제에서 같은 원래 격자와 푸리에 차수를 사용하세요. [`ConformingNVF`](conforming-nvf.md)에서
설명한 것처럼, 문제에서 조립된 층 연산자는 빌더의 행렬로 대체됩니다.
이 예제는 행렬 사용법을 보여 줍니다. 계산하려는 양에 맞춰 보조 셀과 광학 푸리에 차수를 모두
세분화하세요. 지금까지 검증된 내용은 [유한 셀 대조 검사](../../validation/rcwa/nvf-cartesian-cell.md)에
설명되어 있습니다.

## 받는 입력

| 입력 | 계약 |
| --- | --- |
| `orders` | 음이 아닌 정수로 된 명시적인 `(Mu, Mv)` 쌍으로, 두 격자 벡터 방향의 반폭이며 기본값은 `(1, 1)`입니다. 정수 하나만 주면 거부합니다(`experimental.orders`). 어느 축이든 0을 허용합니다. closure 전에 원래의 차이 모드 집합을 정확히 선택합니다. |
| `refinement` | 0, 1 또는 2이며 기본값은 0. 각각 자유 스칼라 퍼텐셜 미지수 4, 19, 79개를 줍니다. 광학 푸리에 차수와는 독립적입니다. |
| `lattice` | `Lattice` 또는 행렬식이 유한한 양수이고 열이 물리 기저 벡터인 유한한 실수 2×2(2-by-2) 배열. 격자 축소는 수행하지 않습니다. |
| `structure` | `background`와 `shapes`만 담은 층 딕셔너리의 비어 있지 않은 리스트 또는 튜플. 배경의 기본값은 진공입니다. |
| 균일한 층 | `shapes`를 생략하거나 빈 리스트 또는 튜플을 주세요. 이 층의 행렬을 만들 때도 plan은 필요합니다. |
| 패턴 층 | 유형이 `rectangle`인 형상 정확히 하나. 이 유형은 기본 유형이기도 합니다. `square`를 포함한 다른 유형 표기는 거부합니다. |
| 직사각형 길이 | 양의 물리 길이 `lx`와 `ly`. 각각 자신의 키가 없으면 따로 `l`로 대체됩니다. 명시적인 `lx`나 `ly`가 우선합니다. |
| 자세 | 실험실 좌표계 라디안 단위의 `angle`(기본값 0). 호출자의 셀 중심으로부터의 물리 카르테시안 offset `cx`와 `cy`(각각 기본값 0). |
| 재료 | 물리 스칼라 유전율 또는 `Material(epsilon=...)`. 배경과 포함체는 유한하고 비자성이며 실수부가 양수, 허수부가 음이 아니어야 합니다. 손실 비율은 서로 독립적이어도 됩니다. |

직사각형은 offset과 함께 평행 이동하는 기본 셀의 중심에 놓입니다. 직사각형은 자신을 중심으로 한 그
셀 안에 엄격하게 들어가야 하고, 선택된 메시에서 양의 유한한 삼각형을 가져야 합니다. offset으로 주기
구조 전체를 자유롭게 평행 이동할 수 있습니다. 평행 이동으로 호출자가 고른 셀 모서리를 넘어도 허용되는
직사각형 치수가 커지지는 않습니다.
반환되는 행렬은 `rcwa.operators`가 쓰는 좌표계인 셀 중심을 기준으로 하며, offset은 직사각형의 공개
중심입니다.
반환된 행렬에 위상, 재료 축척 또는 카르테시안 회전을 추가로 적용하지 마세요.

알 수 없는 키, 추가 형상, 지원하지 않는 형상 종류, 텐서나 이득 재료와 선언된 투자율은 거부합니다.
접촉, 겹치는 주기 이미지와 여러 직사각형을 위한 일반 메시는 이 빌더의 계약 범위 밖입니다. 기존의
[`ConformingNVF`](conforming-nvf.md)와
[`AffineProfileNVF`](affine-profile-nvf.md)는 각자의 기하 계약을 유지하며, 이 명시적 빌더가 그 허용
규칙을 바꾸지 않습니다.

## 계획과 실제 값에 대한 미분

빌더를 만들고 JAX 추적 전에 구체적인 기하로 `plan(structure, lattice)`를 호출하세요. 불변 plan은 각
층이 균일한지 패턴 층인지와, 패턴 층의 순환 꼭짓점 선택을 저장합니다. 모든
`conv_matrices(..., plan=plan)` 호출에 plan을 명시적으로 주세요. 층 개수나 균일/패턴 상태가 바뀌면 새
plan이 필요합니다.

재실행(replay)은 격자, 직사각형 길이, 자세와 재료 값을 다시 읽습니다. 이 값들은 JAX x64에서
`conv_matrices(..., plan=plan, xp=jax.numpy)`를 쓰면 `jax.jit` 아래에서도 실제(live) 값으로 남습니다.
NumPy 구성도 64비트 정밀도를 요구합니다. 빌더는 미분되는 계산 안에서 기하 계획기를 실행하지 않습니다.
미분은 고정된 유한 공간에 대한 것이며, 꼭짓점 선택 사이의 전환은 포함하지 않습니다.

직사각형 각도에 pi를 더하는 것처럼 매개변수 표현을 바꿀 때는 명시적으로 다시 계획하세요. 새 plan은 같은
물리 직사각형을 다른 원시 꼭짓점 순서로 표현할 수 있습니다. 재실행은 포함 관계와 삼각형 유효성을
검사하지만, 메시를 양으로 유지하는 모든 표현 변화를 감지하지는 못합니다. 계획이 갈리는 지점(tie)을
가로지르는 연속성이나 미분은 주장하지 않습니다.

즉시 실행에서 유효하지 않은 입력은 오류를 냅니다. 추적되는 수치 값이 유효하지 않으면 해당 층의 행렬에
NaN이 생기며, 유효하지 않은 격자는 균일한 층을 포함해 모든 층에 영향을 줍니다. 독립적인 약형 시스템과
미분 대조 검사는 이 유한 구성의 적격성을 확인할 뿐, 연속체 접촉 동작이나 Maxwell 투과 정확도를
확인하지는 않습니다.
