# 직사각형 층으로 만드는 conforming NVF

`rcwa.experimental.ConformingNVF`는 재료에 따라 달라지는 conforming 셀을 구성하고, 그 에너지 닫힘(closure)을
`ConvMatrices`로 내보냅니다. `rcwa.operators.solve`는 조립된 연산자 대신 이 행렬을 사용합니다. 이 빌더는
일반적인 `background` / `shapes` 딕셔너리를 받습니다. 설치된 Wavebit 패키지만 있으면 충분하며, 연구용
스크립트는 실행 시 의존성이 아닙니다.

각 패턴 층은 직교 축소 격자의 어느 한 축을 따라 늘어선, 서로 겹치지 않는 직사각형의 시퀀스입니다.
직사각형들은 그 축에 수직인 방향의 범위와 중심을 공유합니다. 공유 모서리 양쪽의 재료는 서로 달라도 되며,
늘어선(tiling) 방향은 층마다 달라도 됩니다. 내부 틈에는 층의 배경 재료가 들어가며, 이 틈은 JAX 재실행
중에 열리거나 닫힐 수 있습니다.

```python
import numpy as np
from wavebit import Material, Units, rcwa
from wavebit.rcwa import experimental, operators

lattice = np.diag([700., 430.])
# Layers are listed from the bottom to the top.
structure = [
                dict(background=1.7, shapes=[
        dict(type='rectangle', lx=671., ly=155., cy=-97.5, material=4.),
        dict(type='rectangle', lx=671., ly=155., cy=97.5, material=9.),
    ]),
                dict(background=2.3),
                dict(background=1.7, shapes=[
        dict(type='rectangle', lx=186., ly=401., cx=-82., material=4.),
        dict(type='rectangle', lx=124., ly=401., cx=113., material=9.),
    ]),
            ]
builder = experimental.ConformingNVF((2, 1), rows=4, body_degree=2, gap_degree=4)
conv = builder.conv_matrices(structure, lattice)
numerics = rcwa.Numerics(orders=builder.orders)
transmitted = []
for polarization in ('p', 's'):
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(
            lattice=lattice, top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
            # Background-only layers fix the thicknesses; conv carries the patterns.
            layers=tuple(rcwa.Layer(thickness=t, background=Material(epsilon=layer['background']))
                         for t, layer in zip((150., 60., 150.), structure))),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(
            side='top', polarization=polarization, theta=.2, phi=.37),)))
    assembled = operators.assemble(problem, numerics=numerics).replace(epsilon=(conv,))
    transmitted.append(operators.solve(problem, assembled).T.sum())
print(*transmitted)  # p, then s
```

`operators.assemble`은 물리 적층, 파장과 수치 설정을 고정합니다. 그 `epsilon`에는 파장마다 다섯 슬롯
튜플이 하나씩 들어 있습니다. 그 튜플을 빌더의 행렬로 바꾸고 `operators.solve`를 호출하면 선언한 적층을
그 행렬로 풉니다. 편광 문자열은 문제에서 선언한 배치의 일부이므로, 각 편광은 따로 조립합니다.

이것은 유한 모델 예제일 뿐, 정확도를 위한 처방이 아닙니다. 셀 해상도와 유지 푸리에 차수는 관심 있는
관측값에 맞춰 각각 따로 세분화해야 합니다. 마지막 차수 증가에 따른 작은 변화만으로는 광학 오차의 상한이
정해지지 않습니다. [에너지와 수렴 연구](../../validation/rcwa/nvf-gap-convergence.md)에서 이 차이를
설명합니다.

## 입력과 지원하는 기하

| 입력 | 계약 |
| --- | --- |
| `orders` | 음이 아닌 정수로 된 명시적인 `(Mu, Mv)` 쌍으로 기본값은 `(3, 3)`이며, 이를 사용하는 `rcwa.Numerics(orders=...)`와 일치해야 합니다. 정수 하나만 주면 거부합니다(`experimental.orders`). 에너지 행렬을 닫기 전에 이 차수 집합으로 사영합니다. |
| `lattice` | `wavebit.Lattice` 또는 열이 기저 벡터인 2×2(2-by-2) 배열. 정수 유니모듈러 변환으로 직교 기저로 축소되는 오른손 기저를 지원합니다. 축소된 두 주기는 양수이며 서로 독립적으로 달라질 수 있습니다. |
| `structure` | 각각 `background`와/또는 `shapes`를 가진 층 딕셔너리의 비어 있지 않은 리스트 또는 튜플. 형상이 없는 층은 균일한 층입니다. |
| `cx`, `cy` | 원래 주어진 셀 중심으로부터의 카르테시안 offset이며 단위는 격자의 길이 단위입니다. 기본값은 0입니다. 주기적 평행 이동에는 제한이 없지만, 좌표는 따로따로 감싸진(wrapped) 조각이 아니라 한 주기보다 짧은 범위의 정렬된 프로파일 하나를 기술해야 합니다. |
| `lx`, `ly` | 직사각형 자신의 축을 따른 양의 물리 길이. 하나의 tiling에 속한 직사각형은 수직 방향 범위와 중심이 같습니다. tiling 축을 따른 범위는 그 주기보다 엄격히 작아야 하며, 수직 방향 범위는 정확한 접촉을 포함해 다른 축의 주기까지 닿을 수 있습니다. |
| `angle` | 실험실 x축으로부터의 라디안이며 기본값은 0입니다. 직사각형 모서리는 직교 축소 격자 축에 정렬됩니다. pi/2를 더하고 `lx`와 `ly`를 맞바꾸면 같은 직사각형을 기술합니다. 고정된 격자 안의 임의 각도는 지원하지 않습니다. |
| 재료 | 실수부가 양수이고 허수부가 음이 아닌 유한한 스칼라 유전율로, 숫자 또는 `Material(epsilon=...)`. 모든 형상과 배경은 각자 독립적인 손실 비율을 가질 수 있습니다. `mu=1`을 포함해 `mu`를 선언하면 지원하지 않습니다. |

형상 리스트는 어떤 순서로 주어도 되며, 계획 단계에서 선택한 tiling 축을 따라 정렬합니다. 서로 다른 재료
사이의 공유 경계는 실제(live) 값으로 남습니다.
직사각형을 같은 재료의 이웃들로 나눠도 계면 제약이 추가되지 않습니다.
균일한 층과 패턴 층은 `rcwa.Stack.layers`와 같은 아래에서 위로의 순서로 하나의 적층에 함께 있을 수
있습니다.

곡선, 겹침, 서로 다른 수직 방향 범위나 중심, 축소 격자에 정렬되지 않은 모서리, 실제로 직교하지 않는 점
격자, 텐서, 이득과 실수부가 양수가 아닌 유전율은 거부합니다.
이 빌더는 지원하지 않는 입력에 대해 다른 NVF나 Li 인수분해를 대신 선택하지 않습니다. 기존의 원형 형상
NVF는 일반적인 `rcwa.Problem` 생애주기를 통해 계속 사용할 수 있습니다.

직교하지 않는 격자 위의 분수 프로파일에 대한 명시적인 아핀 상에는
[`AffineProfileNVF`](affine-profile-nvf.md)를 사용하세요. 그 빌더는 분수 폭, 높이와 재료 구간을 받으며,
여기서 설명한 물리 직사각형 입력을 다르게 해석하지 않습니다.

## 동등한 격자 기저

비스듬한 기저도 직교 점 격자를 기술할 수 있습니다. 예를 들어
`a1=(700,0)`과 `a2=(2100,430)`은 `(700,0)`과 `(0,430)`과 같은 평행 이동을 생성합니다. 계획 단계는
`lattice @ T`가 직교가 되게 하는 정수 행렬 T를 기록합니다. 직사각형의 길이와 각도는 물리적 의미를
유지하며, 어댑터는 이를 평행사변형으로 바꾸지 않습니다.

```python
import numpy as np
from wavebit.rcwa import experimental

lattice = np.array([[700., 2100.], [0., 430.]])
structure = [dict(background=1.7, shapes=[
    dict(type='rectangle', lx=350., ly=401., material=4.+.4j),
])]
builder = experimental.ConformingNVF((2, 1), rows=2, body_degree=2, gap_degree=2)
plan = builder.plan(structure, lattice)
conv = builder.conv_matrices(structure, lattice, plan=plan)
print(plan.reduction, conv.epx.shape)
```

이를 사용하는 문제는 여전히 원래 `lattice`와 같은 차수를 선언합니다. 이것이 문제의 물리적 파수 벡터를
정합니다. 어댑터는 에너지 닫힘 전에 바로 그 모드 집합의 변환된 버전을 선택하므로, 출력 행렬의 차원은
커지지 않습니다. 정수 전단이 크면 필요한 푸리에 계수를 계산하는 비용이 커질 수 있습니다.

중심은 여전히 원래 주어진 셀 중심으로부터의 offset입니다. 형상의 절대 위치를 고정한 채 격자 A에서 다른
기저 Anew로 바꿀 때는 모든 `(cx,cy)` offset에 `sum(A-Anew, axis=1)/2`를 더하세요.
[기저 변경 유도](../../validation/rcwa/nvf-lattice-rebasing.md)에서 위상과 모드 대응을 설명합니다.
`(700,0),(50,430)`과 같은 실제 전단은 여전히 지원하지 않습니다.

## 수동 복소 재료

모든 형상과 배경은 각자의 수동 복소 유전율을 받습니다. 예를 들어 재료 `4+.4j`와 `9+1.8j`는 무손실 배경
`1.7`이나 흡수하는 배경 `1.7+.2j`를 함께 쓸 수 있습니다. JAX 재실행 중에는 실수부와 허수부 모두 실제
값으로 남습니다. 이득과 양수가 아닌 실수부는 거부합니다.

셀은 복소 시행(trial) 필드와 켤레 시험(test) 필드를 사용합니다. 두 가로 방향 교차항을 모두 유지하며,
전체 복소 재료 에너지를 엔진의 구성 행렬로 사영합니다.
[복소 재료 유도](../../validation/rcwa/nvf-complex-material.md)에서 이 구성과 그 독립적인 대조 검사를
설명합니다. [이전 유도](../../validation/rcwa/nvf-common-loss.md)에서 설명한 공통 인자 경우는 여전히
유용한 정확한 대조 기준이지만, 더 이상 입력 제한은 아닙니다.

아래 JAX 예제처럼 각 재료의 허수부를 독립적으로 변화시키세요. 허수부가 0일 때 수동 손실 미분은 양의
손실 쪽에서의 한쪽 미분입니다. 추적되는 재료가 유효하지 않으면 NaN 행렬이 나오고, 즉시 실행 호출은 오류를
냅니다. 메시는 재료 대비의 크기와 위상을 따르며, 손실이 바뀌어도 같은 위상(topology)을 유지합니다.

## 해상도 제어

생성자 키워드는 고정된 셀 근사를 설정하며, JAX 트레이스 동안 고정된 채로 남습니다.

| 키워드 | 기본값 | 의미 |
| --- | --- | --- |
| `rows` | `4` | 비균일화와 다항식 보강 전에, 선택한 tiling 축에 수직인 방향의 초기 분할 수. 최소 2입니다. |
| `body_degree` | `2` | 몸체 퍼텐셜 차수로, 1부터 4까지입니다. |
| `gap_degree` | `4` | 틈 퍼텐셜 차수로, 1부터 6까지입니다. |
| `flux_basis` | `'local'` | 국소 적분 플럭스 기저. 수치 비교용으로 `'global'`도 사용할 수 있습니다. |
| `edge_fraction` | `1e-5` | 가장 작은 기준 모서리 구간으로, 0과 `.001` 사이(양 끝 제외)여야 합니다. 해상도 설정이며, 틈 차단값이나 수렴 허용 오차가 아닙니다. |
| `grade_y` | `True` | tiling 축에 수직인 몸체 구간에 고정된 모서리 비균일화를 추가합니다. |
| `x_subdivisions`, `y_subdivisions` | `1` | 각 기준 구간을 tiling 축을 따라, 그리고 수직 방향으로 나누는 양의 정수 분할 수입니다. |
| `grading_strength` | `8.` | 재료 도약 비균일화 세기로, 0부터 32까지입니다. |
| `grading_radius` | `.01` | 정규화된 비균일화 반지름: 국소 플럭스에서는 `.0001`부터 `.1`까지, 전역 플럭스에서는 `.005`부터 `.1`까지. |
| `quadrature` | `12` | 기본 구적법 설정으로 최소 2입니다. 다항식 사영은 해석적 모멘트를 사용합니다. |

재료 비균일화는 부호 있는 로그 유전율 도약을 제곱해 더하기 전에 스무딩합니다. 그래서 닫히는 구간의 두 경계는
합쳐지거나 상쇄되며, 사라진 배경 구간의 특징이 메시에 남지 않습니다. 이는 이전의 독립 도약 밀도를
대체하며, 비균일화를 켜면 유한 해상도 값이 바뀝니다.
[떨어진 직사각형 유도](../../validation/rcwa/nvf-separated-rectangles.md)에서 공식과 그 독립적인 대조
검사를 설명합니다. 세기 0은 고정된 메시를 유지합니다. 어떤 비균일화 설정도 광학적 정확도를 보증하지 않습니다.

계수 필드와 에너지는 함께 솔버의 카르테시안 좌표계로 회전되고 푸리에 원점으로 평행 이동됩니다. 각 축은
따로 정규화되며, 물리적 종횡비는 셀 방정식과 에너지 적분에 들어갑니다. 물리적 주기를 700단위로 제한하지
않습니다. [직사각형 셀 유도](../../validation/rcwa/nvf-rectangular-cells.md)에서 정사각형 셀 계수만
늘리면 왜 다른 결과가 나오는지 설명합니다.
두 번째 축 방향 tiling에서는 셀이 축소된 오른손 좌표계 `(a2, -a1)`과 역수 종횡비를 사용합니다. 각 층은
원래 솔버의 공통 카르테시안 좌표계와 푸리에 차수 배치로 내보냅니다. 반환되는 행렬은 `rcwa.operators`가
쓰는 좌표계인 셀 중심을 기준으로 하므로, 위상을 더하지 마세요.
[층 축 구성](../../validation/rcwa/nvf-layer-axes.md)에서 이 대응을 설명합니다.

## JAX 계획과 재실행

NumPy와 x64를 켠 JAX로 셀을 구성하며, 출력은 complex128입니다. 반환된 행렬은 `backend='torch'`로
조립한 연산자에 쓰도록 Torch 텐서로 변환할 수도 있지만, 셀 빌더는 Torch 기하 미분을 제공하지 않습니다.
JAX 미분은 셀 구성과 에너지 닫힘에 관한 것입니다. 이를 사용하는 고유값 솔버는 자신의
[자동 미분 동작](../common/autodiff.md)을 그대로 유지합니다.

`jit` 전에 구체적인 기하로 불변 plan을 만드세요. plan은 각 층의 입력 인덱스 순서와 tiling 축, 그리고
정수 격자 축소를 저장합니다. 많은 단일 직사각형의 경우처럼 두 축이 모두 유효하면 계획 단계는 첫 번째
축소 축을 우선합니다.
좌표, 치수, 재료, 배경, 두 격자 주기와 공통 회전은 재실행에서 실제 값으로 남습니다. 기록된 축소는 실제
격자를 직교로 유지해야 합니다. 독립적인 연속 전단은 이 지원 계열을 벗어나므로, 즉시 실행에서는 거부되고
추적 중에는 NaN을 만듭니다.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from wavebit.rcwa import experimental

lattice = np.eye(2)*700.
builder = experimental.ConformingNVF((1, 1), rows=2, body_degree=2, gap_degree=2)

def layers(gap, epsilon, loss):
    return [dict(background=1., shapes=[
        dict(type='rectangle', lx=350., ly=700.-gap, material=epsilon+1j*loss),
    ])]

plan = builder.plan(layers(17., 4., .1), lattice)

@jax.jit
def material_matrix(gap, epsilon, loss):
    return builder.conv_matrices(
        layers(gap, epsilon, loss), jnp.asarray(lattice), plan=plan, xp=jnp,
    ).epx

gap_derivative, loss_derivative = jax.jacfwd(material_matrix, argnums=(0, 2))(17., 4., .1)
print(gap_derivative.shape, loss_derivative.shape)
```

재실행 동안 형상과 층의 개수와 순서를 고정하세요. 직사각형이 겹치지 않는 한, 인접한 모서리는 독립적으로
움직여 내부 배경 구간을 열거나 닫을 수 있습니다. 나중에 두 축이 모두 유효해지더라도 재실행은 선택한 축을
유지하며, 미분 중에 유한 근사를 바꾸지 않습니다.
그 개수나 순서를 바꿀 때, 또는 선택한 축이 유효하지 않게 될 때는 다시 계획하세요.
유효하지 않은 구체적 기하는 오류를 내고, 추적되는 기하나 재료가 유효하지 않으면 그 층에 NaN 행렬이
나옵니다. 수직 방향 주기 전체에 걸친 접촉에서 틈 미분은 양의 틈 쪽에서의 한쪽 극한입니다. 내부 배경
구간이 닫힐 때도 마찬가지입니다. 틈을 0으로 스냅하지 않습니다. 정규화된 접촉에서 기계 엡실론의 32배
이내에서는, 격자 노름이나 범위 뺄셈에서 생기는 음의 반올림 오차도 같은 해석적 방정식으로 다룹니다.
이렇게 하면 접촉 미분이 실제 값으로 유지됩니다. 범위를 그보다 크게 넘으면 거부합니다. 이 허용치는
부동소수점 기하 허용 오차이지, 광학적 정확도 기준이 아닙니다.

같은 격자와 차수로 문제를 조립하세요. 균일한 층이나 영역 층으로 된 적층은 `(u, 0)`, `(0, v)` 또는
`(0, 0)`에서도 2D 연산자를 선택합니다. 한 행짜리 표본 `epsilon` 층으로 된 적층은 1D 계산으로
보내집니다. 이 경우 면내 결합을 모두 유지하려면 `rcwa.Numerics(formulation='first_order')`를 설정하세요.

[통합 기록](../../validation/rcwa/nvf-energy-engine.md)에서 독립적인 대조 검사와 남아 있는 기하 및 정확도
한계를 확인할 수 있습니다.
