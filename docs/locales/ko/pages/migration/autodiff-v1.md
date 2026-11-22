# 계산 미분하기

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 필요합니다.
> 현재 설치에는 이 네임스페이스가 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행되지 않았습니다.


:::{note}
이 페이지는 예전 인터페이스의 전체 호환성 참조입니다. `wavebit.compat.legacy`를 명시적으로 쓰세요. 새 애플리케이션은 [현재 생애주기](../guide/common/lifecycle.md)와 [현재 API 참조](../reference/api.md)로 시작하세요. 여기 인용한 수치 근거는 원래의 범위를 그대로 유지합니다.
:::

그래디언트는 기하에서 회절 효율까지 처음부터 끝까지 흐릅니다. `jax.grad`와 PyTorch
autograd는 사용자에게 아무것도 요구하지 않고, `jax.jit`과 `vmap`에는 객체 하나가 더 필요합니다.

## 공개 API만으로 그래디언트가 동작합니다

```python
import jax, jax.numpy as jnp

solver = wavebit.rcwa(backend='jax', structure=structure, ...)
grad = jax.grad(lambda u: jnp.sum(solver.solve(structure=u).T))(structure)
```

PyTorch에서는 즉시 실행으로 다음과 같습니다.

```python
u = torch.tensor(raster, dtype=torch.float64, requires_grad=True)
solver = wavebit.rcwa(backend='torch', structure=u, ...)
solver.solve().T.sum().backward()          # u.grad
```

둘은 일치합니다. 빠른 시작 격자에서 둘 다 노름이 `0.0802694256415…`인 그래디언트를 보고하며,
약 1e-15까지 맞습니다.

JAX의 `solver.grad(parameter_names, forward, loss_fn)`은 표 재료와 기본 편광을 포함해 이름 붙은
매개변수 지정과 선언을 복원합니다. 사용자 정의 직접 지정과 형상 레시피나 여기(excitation)
딕셔너리의 편집은 임시 상태를 씁니다. forward가 계산에 넘기는 구체적인 조건 키워드는 평소의
효과를 유지하지만, 이름 붙은 그래디언트 매개변수는 그래도 복원됩니다. forward나 손실 함수가
예외를 내더라도 임시 계산 상태를 해제합니다. 이전 필드는 저장한 결과를 쓰거나, 솔버를 통해
필드를 읽기 전에 `solve()`를 다시 호출하세요. 손실 함수 자체는 여전히 현재 필드를 재구성할 수
있습니다.
임시로 바꾼 재료 표를 복원하면서 호출이 구체적인 파장을 유지하면, 표는 그 파장에서 다시
계산됩니다. 그 파장이 복원된 표의 범위 밖이면 호출은 예외를 내고 호출 전의 장면을 복원합니다.

기존 솔버를 붙잡는 원시 JAX 변환에서는 반환된 결과로부터 재구성하거나
(`solver.solve(...).reconstruct_field(...)`), `solve_field` 또는 `solve_field_eager`를 쓰세요.
고수준 계산은 전체 연산이 반환될 때 추적된 실행 상태를 해제합니다. 이전에 저장한 결과는 자신의
필드를 유지합니다. 추적 안에서만 쓰이는 솔버와 `grad`/`fit` 도우미 범위는 재구성이나 손실 계산이
끝날 때까지 legacy 솔버의 필드 접근을 유지합니다. 저수준 `solve_conv` API는 `solver.scattering`을
포함해 호출자가 소유한 상태를 유지합니다. 그 가변 저수준 상태를 읽을 때는 변환되는 함수 안에서
솔버를 만들거나 도우미 범위를 쓰세요.

재사용하는 소스 솔버도 결과가 필드를 소유한다는 같은 규칙을 따릅니다. 리스트나 부가
딕셔너리에 중첩된 `PlaneWave` 진폭을 포함해 추적된 소스 키워드는 임시 지정이며, 계산이 예외를
내더라도 복원됩니다. 구체적인 소스 키워드는 평소의 효과를 유지합니다.

미분 가능한 것은 래스터만이 아닙니다. **`Material`** 객체는 자식이 `(epsilon, mu)`인 고정된
pytree이므로 `jax.grad`가 그 안까지 닿습니다. 그리고 **`Lattice`** 객체는 각도가 아니라 기저를
저장하므로, 미분하는 함수 안에서 만들면 셀 자체의 모양이 설계 변수가 됩니다. 비대각 성분은
전단이며, `period`를 어떻게 써도 이를 표현할 수 없습니다.

```python
def loss(angle):
    A = wavebit.Lattice.oblique(400., 400., angle)
    solver = wavebit.rcwa(backend='jax', lattice=A, structure=structure, ...)
    return jnp.sum(solver.solve().T)

jax.grad(loss)(jnp.asarray(np.radians(105.)))
```

여기서 두 모델링 경로는 같은 질문을 미분하지 않습니다. 래스터는 격자 좌표로 저장되므로 그 기저
그래디언트는 셀을 기울이고 **패턴도 함께 전단합니다**. 반면 매개변수 형상의 절대 데카르트
길이는 셀만 기울입니다. 평범한 장면에서 두 값은 3.6x 차이가 납니다.
[`../oblique-lattice.md`](../theory/rcwa/oblique-lattice.md)의 "Differentiating the cell" 절을 보세요.

## Constraining several layer thicknesses

`layer_thicknesses(logits, total, minimum=0.)`는 층마다 하나의 실수 설계 가중치를 합이 고정된
양의 물리적 두께로 사상합니다. 그 출력을 `Stack.from_structure`나 솔버의 평범한 `thickness`
인수에 넘기세요. 같은 함수가 NumPy, JAX, Torch 배열에서 동작하며, FDTD 전용 형상 객체를 새로
도입하지 않습니다.

```python
from wavebit.compat.legacy import layer_thicknesses, Stack

# Use float64 design variables for FDTD geometry and gradient comparisons.
weights = jnp.array([-.2, .3, 0.], dtype=jnp.float64)
lengths = layer_thicknesses(weights, total=.6, minimum=[.04, .03, .04])
scene = Stack.from_structure(structure, lengths, top=1., bot=4., period=.05)
```

`total`은 양의 정적 호스트 스칼라여야 합니다. `minimum`은 음이 아닌 정적 호스트 스칼라이거나
층마다 하나씩인 값이며, 그 합은 `total`보다 엄격히 작아야 합니다. 둘 다 백엔드 텐서를 받지 않고,
둘 다 미분되지 않습니다.
1차원 `logits` 배열이 배열 프레임워크를 정하며, 그 부동소수점 dtype과 장치를 보존합니다. 평범한
숫자 리스트는 NumPy를 쓰고, 정수 가중치는 float64로 승격됩니다.

사상은 `minimum + (total-sum(minimum))*softmax(logits)`입니다. 모든 가중치에 같은 상수를 더해도
어떤 길이도 바뀌지 않으므로, 마지막 가중치를 0으로 고정하면 그 중복된 최적화 방향이 없어집니다.
가중치가 크게 벌어지면 변환이 포화되어 그래디언트가 작아질 수 있습니다. 양의 최솟값은 포화된
층이 사라지지 않게 합니다. 최솟값이 0이면 길이가 0으로 언더플로할 때 조용히 층을 없애는 대신
예외를 냅니다. 그 밖의 유한하지 않은 구체적 가중치도 예외를 내고, 잘못된 JAX 추적 가중치는 NaN
길이를 반환합니다.

FDTD에는 기본적으로 같은 총합을 갖는 정적 `mesh_thickness`가 필요합니다. 명시적인
[`variable_depth=True`](../guide/fdtd/time-domain.md#changing-the-total-stack-depth)는 총합이 바뀌는
독립적인 물리 길이를 허용하며, 이 총합 고정 사상은 선택한 총합을 계속 일정하게 유지합니다. 계면
평균 폭이 h일 때, 첫 층과 마지막 층의 최솟값이 적어도 h/2이면 내부 지지 영역이 바깥 면 안쪽에
머뭅니다. [다층 예제](../../examples/fdtd/fdtd_multilayer_optimization.py)는 코팅 두께 세 개와 유전율
세 개를 함께 바꾼 뒤, 최종 설계를 더 촘촘한 메시와 독립적인 광학 계산으로 확인합니다.

## `jax.jit`에 무언가가 더 필요한 이유

래스터 파이프라인의 한 단계는 값을 계산하는 대신 구조적인 무언가를 결정합니다. 재료 계면이 어느
쪽을 향하는지입니다. 이 단계는 값에 따라 분기하므로, 분기할 값을 갖지 않는 jit 추적자(tracer)를
통과하지 못합니다.

**grad** 추적자는 구체적인 값을 가지고 있으며, 그래서 위의 코드에는 아무것도 필요하지 않습니다.
결정들은 그래디언트를 구하는 바로 그 지점에서 값을 하나씩 읽으며, 거기서 내린 결정은 낡을 수가
없습니다.

`freeze_geometry()`는 구체적인 구조에 대해 그 결정들을 한 번 내리고 데이터로 돌려줍니다.

```python
geometry = solver.freeze_geometry()          # a StaticGeometry
step = jax.jit(lambda u: jnp.sum(solver.solve(structure=u, geometry=geometry).T))
```

고정은 `jax.jit`과 `vmap` 아래에서만 필요합니다. PyTorch에는 고정할 기록이 없으므로 낡을 기록도
없습니다.

## JIT 전에 `fit`이 각 설계를 준비하게 하기

고정은 그것을 만든 구조를 기술합니다. 상 계면을 보존하는 재료 스윕은 고정을 재사용할 수 있지만,
계면이 움직이거나 상이 합쳐지면 새 고정이 필요할 수 있습니다.
값을 읽을 수 있는 `solve`는 더 이상 맞지 않는 기록으로 답하는 대신
`StaleGeometryError`를 냅니다. **`jax.jit` 아래에서는 비교할 대상이 없어서 그 검사를 실행할 수
없습니다**. 그래서 JAX의 `solver.fit`은 매 반복마다 JIT 밖에서 현재 기하를 준비한 뒤, 그것을
컴파일된 값/그래디언트 단계와 최적화 단계에 명시적으로 넘깁니다. `forward`로는 `solver.solve`,
`solver.solve_field`, `solver.solve_field_eager` 또는 이 메서드 중 하나의 `functools.partial`을
넘기세요. partial에 고정된 키워드 조건은 준비에 포함되며, partial 안의 예전 `geometry=`는 매
단계 교체됩니다.

기존의 속성 이름 API에는 수동 고정이 필요 없습니다.

```python
import optax

best = solver.fit(['structure', 'thickness'], solver.solve,
                  lambda result: -result.T.sum().real,
                  optax.adam(1e-2), iteration=20)
```

형상 최적화에서는 초기 수치 설계 매개변수와, 그것을 솔버 조건으로 사상하는 순수하고 미분 가능한
`parameterize` 함수를 넘기세요. 형상 유형 문자열과 그 밖의 선언은 미분 가능한 매개변수 트리가
아니라 그 함수 안에 두어야 합니다.

```python
def design(width):
    return dict(structure=[dict(background=1., shapes=[
        dict(type='rectangle', cx=0., cy=0., lx=width, ly=190.,
             angle=0., material=4.)])])

initial = dict(width=jnp.asarray(230., dtype=jnp.float64))
solver = wavebit.rcwa(backend='jax', period=(700., 700.), thickness=(150.,),
                     wavelength=850., truncation_order=(2, 2), **design(**initial))
best = solver.fit(initial, solver.solve, lambda result: -result.T.sum().real,
                  optax.adam(1.), iteration=20, parameterize=design)
```

`fit`은 최종 설계 매개변수를 반환하고 그것을 사상한 조건을 솔버에 남깁니다. 기하를 준비하기 위해
구체적인 매개변수로 `design`을 계산하고, 형상 미분을 유지하기 위해 미분되는 계산 안에서 한 번 더
계산합니다. 기록은 내용으로 비교됩니다. 같은 기록은 컴파일된 단계를 재사용하고, 바뀐 기록은 다시
컴파일해야 합니다. 배열을 담은 래스터/텐서 기록의 경우 `fit`은 모든 필드를 직전 기록과 비교하며,
배열의 모양, dtype, 값이 모두 정확히 일치할 때만 그 객체를 유지합니다.
준비에는 여전히 비용이 듭니다. 기록을 고정한다고 연속적인 설계 값이 고정되지는 않습니다.
형상 레시피, 셀 기저, 그린 셀 선언, 주변 `Material` 값에 있는 NumPy 배열은 이 임시 계산을 위해
복사됩니다. 따라서 임시 조건을 통한 제자리 편집은 원래 설계를 바꾸지 않습니다.

컴파일된 각 forward와 손실은 임시 솔버 상태에서 실행됩니다. 그 직접 지정과 형상 레시피나 여기
딕셔너리의 편집은 미분이 끝나기 전에, 실패한 경우에도 복원되며, 최종적으로 사상된 조건은 최적화
후에 적용됩니다.

칠한 형상 경로는 "기하를 움직인다"를 더 좁게 해석합니다. 그 기록이 좌표가 아니라 조합 정보를
담기 때문입니다. 형상이 미끄러지는 것은 비트 단위로 똑같이 재생되며, *어느 형상이 어디를
덮는지*가 바뀔 때만 기록이 낡습니다. 어떤 움직임이 그런지는
[`../boundary-modeling.md`](../guide/rcwa/boundary-modeling.md)를 보세요.

고정은 기하가 통과한 검사도 기록합니다. 거부될 층에서 만든 기록은 그 거부를 재생하며, 살아 있는
장면이 지나갈 수 없는 입구를 고정된 장면이 지나가게 두지 않습니다.

임의의 `forward`는 다른 장면을 만들거나 여러 계산을 수행할 수 있습니다. `fit`은 클로저에서 그
장면을 추론할 수 없으므로 명시적인 준비 계약이 필요합니다.
`prepare_geometry(**conditions)`가 매 단계 JIT 밖에서 실행되고, 그 반환값이
`forward(geometry=..., **conditions)`로 전달됩니다. 예를 들어 이 솔버를 감싸는 함수라면 다음과
같습니다.

```python
def forward(geometry, **conditions):
    return solver.solve(geometry=geometry, **conditions)

best = solver.fit(['thickness'], forward, lambda result: -result.T.sum().real,
                  optax.adam(1.), iteration=20,
                  prepare_geometry=solver.freeze_geometry)
```

감싸는 함수는 초기 고정을 붙잡는 대신 넘겨받은 기하를 써야 합니다.
`forward`가 `functools.partial`이면 준비 함수도 그 고정된 키워드 기본값을 받으며, 현재 조건이
우선하고 `geometry`는 제외됩니다. 내장 솔버 forward의 경우 솔버 조건만 준비 함수에 전달되고,
표본화 크기와 `auto_detune` 같은 호출 옵션은 forward에 남습니다. 키워드나 위치로 묶인 기하는
현재 기록으로 교체됩니다.

`prepare_geometry`가 없으면 불투명한 forward는 최적화 전에 거부됩니다. 설계 변수의 매핑을 받는
사용자 정의 forward는 그 변수를 솔버 속성으로 지정하지 않고 반환하며, 그 해석은 forward가
책임집니다. `parameterize`를 주면 준비와 forward 호출 둘 다보다 먼저 실행됩니다.

적분 래스터 경로에서는 값이 같은 픽셀도 서로 독립적인 설계 변수로 남습니다. 푸리에 구성은
미분과 JAX 추적 중에 모든 픽셀 가장자리를 유지하는데, 같은 값 블록마다 대표 하나만 고르면 다른
픽셀의 미분이 버려지기 때문입니다. 구체적인 계산에서는 여전히 같은 값 블록을 압축합니다. 따라서
추적된 적분 구성은 압축된 즉시 실행 계산보다 비용이 더 들 수 있으며, 특히 반복된 값이 많은 큰
래스터에서 그렇습니다.

루프를 직접 작성한다면 같은 순서를 따르세요. **현재** 설계 조건을 `freeze_geometry`에 적용하고,
결과 기록을 컴파일된 계산의 정적 인수로 넘긴 뒤, 매개변수를 갱신합니다. 준비는 JIT 밖에 두세요.
이렇게 하면 이산 계획이 갱신되지만, 접촉이나 겹침 전이가 전이 지점 자체에서 미분 가능해지지는
않습니다.

서로 떨어진 텐서 원과 타원은 계면 계획용 다각형이 아니라 해석적 경계를 통해 법선장에 들어가므로,
다각형 꼭짓점이었던 점을 지나는 법선장 표본 때문에 답이 튀지 않습니다. 각 표본은 어떤 격자에서든
곡선의 가장 가까운 주기 사본의 경계까지의 거리와 그 사본의 레벨셋 방향을 읽습니다. 레벨셋 방향은
형상의 음함수의 그래디언트로, 다각형 선분의 벡터처럼 곡선에서 표본 쪽을 향하며 곡선 위에서는
바깥쪽 법선입니다. 고정된 기하는 각 곡선을 어떤 사본들에 대해 탐색하는지 기록합니다. 곡선이 자기
중심의 Wigner-Seitz 셀 안에 있는 동안은 자기 셀의 이웃이고, 셀을 벗어나면 더 넓은 고리입니다.
새 계획이 기록에 없는 사본을 탐색하게 되면, 예를 들어 곡선이 셀 밖으로 커져 그 이웃보다 넓은
고리가 필요해지면, 즉시 실행 재생은 이름을 밝혀 거부합니다. 셀 안으로 다시 줄어든 곡선은 기록된
고리가 이웃을 담고 있으므로 새 계산과 똑같이 재생됩니다. JIT 아래에서는 곡선이 셀을 벗어나기
전에 다시 고정하세요.

기하 계획이 유효한 동안에도 표본화된 NVF 확장에는 매끄럽지 않은 점이 있을 수 있습니다. 곡선의 두
주기 사본 사이의 동률, 두 형상의 계면이 같은 거리에 있는 곳에서 대비 가중 혼합의 전환, 그리고
대각선을 가로지르며 가장 가까운 면이 바뀌는 직사각형의 모서리 같은 진짜 모서리입니다. 이 중
하나를 가로질러 옮겨진 법선장 표본은 답을 계단처럼 바꿉니다. 칠한 겹침은 다각형 계면 근사를
유지합니다. 자동 미분은 전환점에서 선택된 가지의 미분을 보고합니다. 그래디언트가 유한하다는 것만으로
양쪽 미분이 성립하지는 않으며, 간격이 격자 간격보다 작은 중심 차분은 한 매끄러운 조각 위에 머물러서
자동 미분을 검증하지 못한 채 일치합니다.

이 계단은 구간에 걸친 그래디언트도 치우치게 하며, `shape_nvf_grid_size`를 키워도 그 치우침이
확실히 사라지지 않습니다. 격자를 세분하면 계단 하나하나는 작아지지만 더 많은 표본이 가로지르기
때문입니다. 타원의 방향은 중심축(medial axis), 즉 두 끝의 곡률 중심 사이에 놓인 장축 구간을
가로질러 전환이 없도록 선택되며, 다각형 경로의 극한인 최근접점 벡터에는 이 전환이 있습니다. 하나뿐인
특이점인 타원의 중심은 계단을 만들지 않지만, 중심 옆을 거리 `d`로 지나가는 표본은 반축이
`a > b`일 때 단축을 따라 약 `d (b/a)**2`(장축을 따라서는 `d (a/b)**2`)의 움직임 동안 방향이 반
바퀴 돌고, 거기서 그래디언트는 대략 `1/d`처럼 커집니다. 지나가는 경로가 중심에 가까워질수록 한없이
커지므로, 그런 경로 옆에서 내딛는 최적화 단계는 얼마든지 커질 수 있습니다.

따라서 NVF 격자를 세분하는 것으로는 그런 그래디언트를 검증할 수 없습니다. 손실 자체와 대조해
검증하세요. 격자 간격 몇 개에 걸친 구간에서, 손실과 같은 매개변수 값으로, 그리고 타원 중심을 지나는
표본이 방향을 돌리는 구간보다 더 촘촘하게 표본화하여 그래디언트를 적분하고, 그 적분을 구간에 걸친
손실의 변화와 비교하세요.
[텐서 형상 적분](../guide/rcwa/tensor-shape-integration.md)에 방향 규칙과 이 구간의 바탕이 된 측정이
있습니다.

## 금속에서의 역규칙 하한

`1/eps`의 부호가 바뀌는 층, 즉 유전체 옆에 금속이 있는 층에서 NVF 마무리 단계는 `<1/eps>`를
역행렬로 만들기 전에 `1/(2 max|eps|)`보다 작은 모든 특이값을 그 하한까지 올립니다.
[금속에서의 역규칙](../theory/rcwa/nvf-metal-inverse-rule.md)을 보세요. 수동 유전체로 된 층은 이
하한에 닿지 않지만, 이득과 손실을 섞은 유전체 층은 닿을 수 있습니다. 텐서 재료가 있는 장면에는
하한이 적용되지 않습니다.

1차 미분은 정확합니다. 하한을 적용한 행렬 `U max(S, f) V^H`는 SVD를 미분하지 않고 닫힌 형태로
미분되므로, 균일한 층의 축퇴된 특이값이나 두 재료 층의 두 묶음이 있어도 미분이 유한합니다. 하한
자체가 유전율에 의존하는 부분도 포함됩니다. 금 원판과 금 구멍에서 지름과 `Re eps`, `Im eps`에
대한 미분은 JAX와 PyTorch 모두에서, 수직 입사와 비스듬한 입사 모두에서 중심 차분과 1e-8까지
일치합니다.

순방향 모드도 같은 미분을 얻습니다. `jax.jvp`와 `jax.jacfwd`, PyTorch의
`torch.autograd.forward_ad`입니다. `torch.func.vmap`을 통한 `torch.autograd`(파장이나 설계의 배치를
`backward()`나 `forward_ad`로 미분하는 경우)도 하한이 걸리는 곳에서 정확하며, 각 레인에 하한이
걸리든 걸리지 않든 마찬가지입니다. 하한이 걸리지 않는 레인은 비트 단위로 `<1/eps>` 자체를 얻습니다.
`torch.func`의 grad 및 jvp 변환(`grad`, `vjp`, `jvp`, `jacrev`, `jacfwd`, `hessian`, 단독으로
또는 `vmap`과 합성)은 하한이 걸리고 그 입력이 하한을 적용한 행렬에 닿는 곳에서, 모드 규칙에서와
마찬가지로 이름을 밝혀 `NotImplementedError`를 냅니다. 그런 곳에서는 `torch.autograd`를 쓰세요.
`solve()` 전체를 통과하는 순방향 모드도 동작합니다.
[Forward mode](#forward-mode)를 보세요.

1차 미분만 지원합니다. 닫힌 형태는 `<1/eps>`의 변화에 대해 선형이므로, 이를 통한 2차 미분에는
하한의 곡률이 빠집니다. 그래서 하한이 걸리는 층을 통과하는 2차 미분은 모드 연산자에서처럼 예외를
냅니다. `create_graph=True`와 두 혼합 차수(역방향 패스를 통과해 운반되는 `forward_ad` 접선, 또는
그것을 통과하는 역방향 패스) 모두에 대해 JAX에서는 `NotImplementedError`, PyTorch에서는
`RuntimeError`입니다. 수동 유전체 층에서는 하한이 항등 사상이므로 `eps_conv_matrices()`의 헤시안이
정확하게 유지됩니다. `jax.jit`과 `jax.checkpoint` 아래에서는 추적을 만들 때 하한이 걸리는지 알 수
없으므로, 거기서는 어떤 NVF 층을 통과하는 2차 미분도 거부합니다.

하한에는 꺾임도 있습니다. 설계가 움직이면 특이값이 하한을 가로지를 수 있습니다. 손실이 있는
금속에서는 거기서 응답이 연속으로 유지되지만 그 미분은 계단처럼 바뀝니다. 그 모드가 하한에
붙잡혀 있다가 설계를 따라가게 되기 때문입니다. 계단은 몇 배일 수 있으며(121개 고조파에서 금
원판의 지름 미분이 7x), 잘린 집합은 지름이 몇 나노미터 바뀔 때마다 달라집니다. 정확히 하한에 있는
값은 잘리지 않은 것으로 치지만, 교차 지점 자체에서는 계산된 특이값이 어느 쪽에 놓일지를 반올림이
정하므로 어느 쪽의 한쪽 기울기든 나올 수 있습니다. 위의 NVF 계단과 마찬가지로, 그런 교차를
가로질러 얻은 그래디언트는 한 점이 아니라 구간에 걸친 손실과 대조해 확인하세요.

무손실 금속에서는 더 나쁩니다. 거기서는 응답 자체가 튑니다. 그때 `<1/eps>`는 에르미트 행렬이고,
잘린 고유값의 부호가 바뀌는 곳에서 하한이 그것을 `-f`에서 `+f`로 옮깁니다. 그 원판을 무손실로
만들면 `Re eps = -27.857`에서 `R`이 0.137만큼 떨어집니다.
자동 미분은 한쪽의 기울기를 반환하며 그 계단을 보고할 수 없습니다. 금속에 조금이라도 손실이
있으면 그 도약은 가파르지만 연속적인 꺾임이 됩니다.

## 미분 중의 수치 재시도

레일리 특이점에서는 기존의 `auto_detune` 재시도가 필요할 수 있습니다. 일반적인 순방향 계산에서 그
기본 상대 간격은 여전히 `1e-12`입니다. 자동 미분 중에는 모드 미분의 상쇄 오차를 줄이기 위해 기본
재시도가 `max(1e-8, 64 * machine_epsilon)`을 씁니다. 이것은 이미 수치 재시도가 필요한 계산만
바꿉니다. 명시적인 수치 `auto_detune` 간격은 주어진 그대로 쓰이고, `False`는 재시도를 끕니다. 경고는
실제로 쓴 파장이나 각도를 보고합니다.
이것은 그 근처 문제에서의 미분이며, 물리적인 회절 문턱에서의 미분을 정의하지 않습니다. 추상적인
JAX 컴파일은 즉시 실행 재시도 정책을 실행할 수 없습니다.

## 반복되는 광학 모드

재료에 대한 1차 미분은 각 층의 모드 연산자의 전체 행렬 변분을 씁니다. 이렇게 하면 균일하거나
대칭인 설계처럼 모드가 고유값을 공유할 때도 편광 결합과 간섭의 미분이 유지됩니다. 순방향 재료를
바꾸거나 고유값을 섭동하지 않습니다. 유지되는 필드도 효율 계산과 같은 변분을 씁니다.

이 모드 규칙은 1차 미분을 지원합니다. 모드 연산자의 2차 미분은 지원하지 않으며, 재료 헤시안은
부분적인 결과를 반환하는 대신 예외를 냅니다. PyTorch는 비선형 재료 매개변수화를 거치는 경우를
포함해 역방향 패스가 모드 연산자에 닿을 때 `create_graph=True`를 거부합니다.
미분에는 또한 역행렬이 있는 모드 기저와 분리된 순방향·역방향 부분 공간이 필요합니다. 모드 척도의
64 machine eps 이내에서 반대 방향 모드가 충돌하면 즉시 실행 미분에서는 거부하고, 추상적인 JAX
컴파일에서는 그 값과 그래디언트가 유한하지 않게 됩니다. 0/반대 방향 제곱근 모드도 마찬가지로 그
미분 가능한 가지의 범위 밖입니다. 레일리 특이점에는 평소의 수치 재시도 정책이 그대로 적용됩니다.

컴파일된 재료 또는 필드 그래디언트는 조밀한 모드 변분 행렬을 운반하므로, 즉시 실행 순방향 계산보다
더 많은 작업과 메모리가 필요할 수 있습니다. 1차 미분은 1차 최적화기와 함께 쓰고, 최적화하는 물리
문제에 대해 푸리에 수렴과 기하 분해능 수렴을 여전히 확인하세요.

## Forward mode

`jax.jvp`, `jax.jacfwd`, `torch.autograd.forward_ad`는 `solve()`, `result.reconstruct_field`, 산란
행렬, `solve_field`를 통해 역방향 모드와 같은 1차 미분을 줍니다. 원판의 지름에 대해서는 다음과
같습니다.

```python
def transmission(width, backend):
    disc = dict(type='circle', lx=width, material=12.25)
    return wavebit.rcwa(backend=backend, structure=[dict(background=1.0, shapes=[disc])],
                        ...).solve().T

primal, tangent = jax.jvp(lambda w: transmission(w, 'jax'), (jnp.asarray(w0),), (jnp.asarray(1.0),))
```

PyTorch에서는 다음과 같습니다.

```python
from torch.autograd import forward_ad
with forward_ad.dual_level():
    width = forward_ad.make_dual(torch.tensor(w0, dtype=torch.float64),
                                 torch.tensor(1.0, dtype=torch.float64))
    tangent = forward_ad.unpack_dual(transmission(width, 'torch')).tangent
```

순방향 패스는 입력 방향 하나를, 역방향 패스는 출력 하나를 운반하므로, 순방향 모드는 적은 수의 설계
매개변수에 대해 많은 출력을 구할 때 적합합니다. 한 기하 매개변수에 대한 모든 차수의 효율, 필드
셀, 또는 산란 행렬이 그런 예입니다.

모드 연산자의 규칙은 모드 기저 `V`를 고정한 채 연산자의 변화에 대해 선형인 `V^-1 dA V`이므로 두
모드 모두에 쓰입니다. 순방향 모드는 이를 적힌 그대로 운반하고, 역방향 모드는 그 전치를 운반합니다.
[`tests/rcwa/test_forward_mode.py`](https://github.com/fieldalgo/wavebit/blob/main/tests/rcwa/test_forward_mode.py)는
`R`, `T`, 진폭, 필드의 순방향 접선을 역방향 모드와 1e-11까지, 중심 차분과는 간격으로 제한되는
정확도까지 대조합니다.
기하, 재료, 입사, 셀 자체에 대해, 다음 경로에서 그렇게 합니다.

- 1D TE와 TM, 그리고 conical;
- Li, Laurent, NVF 인수분해를 쓰는 2D와 z 결합 정식화;
- 직사각형, 육각형, 비스듬한 셀 위의 형상. 형상 두 개로 된 층, 그리고 맨 배열이나 중첩 리스트로
  주어진 셀을 포함합니다;
- NVF 래스터의 주기와 셀 각도(`'distance'` 법선장을 통해서; `'spectral'`과 `'smooth'`도
  2026-11-10부터 셀 미분을 운반하며, 고정된 기록은 필드를 보관하는 대신 이를 재생합니다.
  [`../oblique-lattice.md`](../theory/rcwa/oblique-lattice.md)를 보세요);
- NVF 하한이 걸리는 금속, 자성 재료와 텐서 재료;
- `excitation`, 산란 행렬과 `solve_field`;
- 레일리 점에서의 `auto_detune` 재시도.

중심 차분이 미분을 판정할 수 없는 곳에서는 어느 모드도 판정할 수 없습니다. 위의 NVF 계단은 순방향
모드에도 그대로 적용됩니다.

2차 미분은 두 미분 모두에서 모드 연산자가 움직이는 곳이면 어디서든 이름을 밝혀 거부합니다. 그곳이
바로 1차 규칙을 다시 미분하면 부분적인 헤시안이 반환될 곳입니다.

- JAX에서는 연산자가 바깥 변환의 접선을 운반하게 되면 `jax.hessian`, `jax.jacfwd(jax.jacrev(f))`,
  `jax.grad`의 `jax.jvp` 등이 `NotImplementedError`를 냅니다. `jax.jit`과 `jax.checkpoint` 아래에서도
  마찬가지입니다.
- PyTorch에서는 규칙을 통과하는 `create_graph=True`와, `forward_ad` 듀얼로 만든 연산자를 통과하는
  역방향 패스에 대해 `RuntimeError`가 발생합니다. 두 혼합 차수 모두 그 지점에서 규칙에 닿습니다.

PyTorch의 신호는 듀얼 자체입니다. 그래서 듀얼 수준이 열려 있는 동안 만든 그래프를 통과하는 1차
역방향 패스도, 그 수준이 닫힌 뒤라도 거부됩니다. 역방향 패스는 듀얼 수준 밖에서 수행하세요.

한 매개변수가 어떤 연산자도 움직이지 않는 2차 미분은 규칙을 많아야 한 번 미분하므로 거부되지
않습니다. 두께가 그런 매개변수이며, 단독이든 유전율과 함께든 마찬가지입니다. 다음 2차 미분은
정확합니다.

- `jax.hessian`과 혼합 미분의 모든 JAX 차수를 통한 경우;
- `create_graph=True`를 통한 경우. 혼합 미분이라면 두께를 먼저 취하세요. 유전율을 먼저 취하면 역방향
  패스가 grad 모드를 켠 채 규칙을 실행하므로 거부됩니다.

PyTorch에서는 다른 모든 경로도 정확합니다. 역방향 패스와 결합한 `forward_ad`를 어느 순서로든 쓰는
경우, 그리고 어떤 연산자도 움직이지 않는 매개변수에 대해 `torch.func`의 `jvp(jvp)`, `jacfwd(jacfwd)`,
`grad(jvp)`와 그보다 깊은 모든 중첩, `vmap(jacfwd)`와 `vmap(hessian)`을 grad 모드를 켜든 끄든 쓰는
경우입니다(움직이는 연산자에 대한 `torch.func`는 거부합니다. 아래를 보세요). 여기에는 우회책이
필요합니다. PyTorch 자체의 `torch.linalg.solve` 미분(torch 2.13)은 두 계열에서 오류 없이 틀립니다.

- 순방향 모드 규칙을 다시 미분하는 경우. 그 규칙과 grad 모드를 끈 역방향 패스가 PyTorch가 미분하지
  않는 LU 인수를 재사용하기 때문입니다(pytorch#196695; pytorch#192667이 순방향 규칙을 고치며, 2.15
  릴리스 후보에 들어 있습니다). grad 모드를 끈다는 것은 `create_graph=False`만이 아닙니다.
  `torch.func.vjp`, `jacrev`, `hessian`은 호출된 grad 모드에서 역방향 패스를 실행하므로,
  `torch.no_grad()`, `torch.set_grad_enabled(False)`, `torch.inference_mode()` 아래에서는 이들도 인수를
  재사용합니다;
- 두 `vmap` 수준 아래의 순방향 모드 규칙. 1차 미분인 `vmap(jacfwd)`와 `vmap(hessian)`이 해당하며,
  거기서 `lu_solve`가 인수는 한 수준으로, 우변은 다른 수준으로 배치 처리해 틀린 레인을 반환하기
  때문입니다. `torch.no_grad()` 아래 `vmap(jacrev)`의 역방향 패스도 그렇게 실행합니다.

적층의 모든 선형 풀이는 `backend.linear_solve`를 거칩니다. PyTorch의 규칙이 정확한 곳에서는 바이트
단위로 `torch.linalg.solve` 자체입니다. 즉시 실행 풀이, 1차 미분(두 `vmap` 수준 아래의 순방향 모드
수준, 곧 `vmap(jacfwd)`와, PyTorch가 거기서 옳기는 하지만 `vmap(vmap(jvp))`, 그리고
`torch.no_grad()` 아래의 `vmap(jacrev)`, `vmap(grad)`, `vmap(vjp)`는 제외), 그리고 grad 모드를 켠
상태에서 역방향 패스가 다시 풀이하는 `hessian`, `jvp(grad)`, `create_graph=True`가 그렇습니다. 그 밖의
곳에서는 PyTorch가 모든 차수에서 정확하게 미분하는 규칙을 가진 LU 인수로 풉니다
(`tests/common/test_linear_solve_derivatives.py`). 순방향 모드 수준이 다른 수준 안에 있는 곳, grad
모드를 끈 채 들어간 `torch.func` grad 수준이 다른 수준 안에 있는 곳(`torch.no_grad()` 아래의
`hessian`), `torch.func.grad`, `vjp`, `jacrev` 안에서 `forward_ad` 듀얼을 만드는 곳, 그리고 두 `vmap`
수준 아래입니다. 다른 입력이 grad를 요구하는 동안 취한 `forward_ad` 접선도 역방향 패스가 뒤따를 수
있으므로 그 경로를 따르며, 마지막 몇 자리에서 움직입니다(상대 8.7e-15). `torch.no_grad()` 아래의
`jvp(grad)`, `vmap(grad)`, `vmap(vjp)`, 그리고 어느 grad 모드에서든 `vmap(vmap(jvp))`도 마찬가지로,
어느 쪽이든 정확하지만 마지막 몇 자리에서 움직입니다. `torch.func.grad`는 언제나 다시 풀이하고, 자기
`vmap` 안에서 호출된 `vjp_fn`은 두 번째 수준을 보지 못하며, 두 `vmap` 수준이 모두 행렬을 배치 처리하는
`jvp`는 인수와 우변을 똑같이 배치 처리하지만, 이 수준들 중 어느 것도 풀이 지점에서 `jacrev`나
`vmap(jacfwd)`의 수준과 구별할 수 없기 때문입니다.

`torch.func` grad 수준은 들어갈 때의 grad 모드, 곧 `jacrev`와 `hessian`이 그 역방향 패스를 실행하는
모드로 읽힙니다. `torch.func.vjp`에서 얻은 `vjp_fn`을 다른 모드에서, 즉 `vjp` 자체는 그렇지 않았는데
`torch.no_grad()` 아래나 `create_graph=False`로 호출하면, 풀이 시점에는 아직 실행되지 않은 상태입니다.
바깥의 순방향 모드 수준이 행렬을 움직여 그 역방향 패스를 미분하게 되는 곳, 또는 grad 수준 뒤에
추가된 `vmap`이 그 수준이 들어갈 때 있던 `vmap` 옆에서 그것을 배치 처리하는 곳에서는, 역방향 패스가
호출을 명시하는 `RuntimeError`를 냅니다. 그 밖의 곳에서는 정확합니다. `vjp_fn`은 `vjp`에 들어갈 때의
grad 모드에서 호출하세요.

2026-10-12까지는 다음이 틀린 값으로 돌아왔습니다.

- 1D TM 격자에서 필드 판독값의 두께 2차 미분을 `jvp(jvp)`나 역방향 패스와 결합한 `forward_ad`로
  구하면 1.1001006774e-04 대신 1.1801616522e-04가 나왔습니다;
- 2D 래스터의 두께 2차 미분을 `jvp(jvp)`, `jacfwd(jacfwd)`, `grad(jvp)` 또는 `forward_ad` 위의
  역방향 패스로 구하면 ETM에서 812.595272 대신 760.746214가 나왔고(`torch.func.hessian`은
  812.595272), 필드 값이 있는 SMM에서는 725.018988이 나왔습니다. 역방향 패스 위의 `forward_ad`로
  구하면 그 래스터의 다른 판독값에 대해 11.18 대신 1.284가 나왔습니다;
- `torch.no_grad()` 아래의 `torch.func.hessian`이나 `jvp(jacrev)`, 그리고 그 안에서 취한
  `forward_ad` 접선의 `torch.func.grad`나 `jacrev`는 같은 틀린 값을 냈습니다. 1D TM 격자에서는
  1.1801616522e-04, 2D 래스터에서는 760.746214였습니다;
- `d2/(d eps d thickness)`를 어느 혼합 차수로 구하든 0.002997 대신 0.004266과 0.004953이
  나왔습니다;
- 1D TM 격자의 세 두께 레인에 대한 `vmap(jacfwd)`는 두 번째 레인에서 0.01147 대신 197.0을, 2D
  래스터에서는 NaN을 냈습니다. `torch.no_grad()` 아래의 `vmap(jacrev)`는 거기서 0.011471 대신
  196.96을, 래스터에서는 NaN을 냈습니다.

`torch.func`의 grad 및 jvp 변환(`grad`, `vjp`, `jvp`, `jacrev`, `jacfwd`, `hessian`, 그리고 각각을
`vmap`과 합성한 것)은 변환이 모드 연산자를 미분하는 곳, 즉 입력이 모드 연산자에 닿는 곳이면 어디서든
`torch.func`를 명시하는 `NotImplementedError`를 냅니다. 규칙의 1차 보호 장치는 그런 수준을 2차 미분과
구별할 수 없습니다. `torch.func.grad` 아래에서는 그 역방향 패스가 grad 모드를 켠 채 실행되는데, 이는
`create_graph=True`와 똑같아 보이고, `torch.func.jvp` 아래에서는 `jvp(jvp)`가 부분적인 헤시안을
반환하는 것을 막을 수단이 없습니다. 그런 곳에서는 `torch.autograd`를 쓰세요.

입력이 어떤 연산자도 움직이지 않는 변환, 예를 들어 두께에 대한 변환은 정확합니다. `nn.Parameter`
유전율처럼 `requires_grad`로 붙잡힌 텐서가 연산자에 들어가는 경우에도 그렇습니다.
`torch.func.grad`, `jvp`, `jacrev`, `jacfwd`와 그것들을 `vmap`과 합성한 것은 즉시 실행 미분을
반환합니다. 그중 무엇을 쓴 뒤든, 붙잡힌 텐서로의 이후 `backward()`도 즉시 실행 혼합 미분을 반환하며,
`torch.func.grad` 밖의 `forward_ad` 듀얼도 마찬가지입니다. `torch.func.jvp`, `jacfwd`, `jvp(vmap)`
뒤의 그 역방향 패스는 순방향 위의 역방향으로, 위에서 말한 `torch.linalg.solve` 결함의 경우입니다.
2026-10-12까지 그것은 1D TM 격자의 필드 판독값에서 상대 0.12, 금 원판에서 0.018만큼 오류 없이
어긋났습니다. 입구 검사는 각 수준 자체의 플래그를 읽습니다. grad 변환 안에서 계산된 값은 입력에 의존하지
않더라도 그 수준의 래퍼를 지니므로, 래퍼의 유형만 물으면 이것들을 모두 거부했을 것입니다. 중첩된
grad 및 jvp 변환(`hessian`, `grad(grad)`, `jvp(jvp)`, `jvp(jvp(grad))`, `vmap(hessian)`, 그리고
두께에 대한 `torch.func.grad` 위에 유전율에 대한 `jacfwd`를 얹은 것 같은 혼합 편미분)은 연산자가
미분을 전혀 운반하지 않는 동안 모든 차수에서, grad 모드를 켜든 끄든 두께에 대해 정확합니다.
연산자가 미분을 운반하는 곳에서는 안쪽 수준을 유형으로 세고 풀이를 거부합니다.
2026-10-11까지는 혼합 편미분이 1D TM 격자에서 상대 0.15만큼 오류 없이 어긋났습니다. 다른 grad나
jvp 수준 아래의 접선을 읽을 수 없었기 때문입니다.

`torch.func.vmap`을 통한 `torch.autograd`는 지원되며 정확합니다. 배치 풀이의 `backward()`나
`torch.autograd.grad`, 중첩된 `vmap`, 배치된 파장 옆에서 `requires_grad`로 붙잡힌 매개변수,
`vmap`을 통한 `forward_ad`는 각각 레인마다 즉시 실행 autograd의 미분을 반환합니다
(`tests/rcwa/test_torch_vmap_autograd.py`; conical x 차단점에서는
`tests/rcwa/test_conical_cutoff.py`). 2026-10-11까지 이들은 고유 미분이 빠진 미분을 오류 없이
반환했습니다. `vmap`이 풀이에 넘기는 래퍼의 `requires_grad`가 False로 읽혀서 모드 규칙이 전혀
실행되지 않았기 때문입니다. conical 격자에서 `dR/d(wavelength)`는 즉시 실행 autograd와 중심 차분이
-0.004225로 읽는 곳에서 -0.003033으로 읽혔습니다. 806661e4에서 34개 장면을 열 가지 변환으로 조사한
결과 340개 칸 중 60개가 틀렸습니다. 그때는 `vmap`을 통한 `create_graph=True`가 부분적인 헤시안을
반환했지만, 지금은 즉시 실행에서처럼 거부합니다.

## 추적할 수 없는 것

| | 이유 |
| --- | --- |
| legacy `solver.reconstruct_field`에서 `thickness`에 대한 **명시적 `sample_z`** | 이 솔버 메서드는 구체적인 두께로 구간을 나누므로 추적된 두께는 거부하며, 개수는 계속 추적할 수 있습니다. 결과가 소유하는 `result.reconstruct_field`는 각 점이 자기 층 안에 머무는 동안 명시적 높이와 두께 미분을 지원합니다. [필드](../guide/rcwa/fields.md#differentiating-internal-fields)를 보세요 |
| `torch.compile` | 지원하지만 결코 추적되지 않습니다. 모든 공개 wavebit 호출 객체는 `torch.compiler.disable` 입구이므로, wavebit은 컴파일된 호출자 안에서 즉시 실행되고 그 값과 그래디언트는 `torch.func`와 이중 역방향 주변에서도 즉시 실행 torch의 것입니다. 컴파일되는 것은 wavebit 출력에 대한 호출자 자신의 산술뿐이며, 컴파일해도 wavebit 부분은 빨라지지 않습니다. `fullgraph=True`와 `torch.export`(strict와 non-strict)는 이름을 밝혀 거부합니다. 2026-10까지는 wavebit을 그래프 밖에 두는 장치가 없어서, 기본 모드가 오류 없이 틀린 값 *과* 그래디언트를 반환했습니다. 즉시 실행의 -0.895 대신 1D TE 손실 -4.43, 0.272 대신 그 그래디언트 -0.118이었는데, Inductor(torch 2.13)가 `array_api_compat`의 복소 `eye` 대각을 그것을 읽는 커널 뒤에 썼기 때문입니다([`../torch-compile.md`](../development/torch-compile.md) §6) |
| `torch.jit.trace` | 해결된 결과의 판독값과 `Lattice`를 포함해 모든 공개 wavebit 호출에서 이름을 밝혀 거부합니다(`NotImplementedError`, "wavebit cannot be traced by torch.jit.trace"). 이것은 값에 의존하는 모든 결정을 상수로 기록하고 `grad_fn`이 없는 출력을 반환합니다. 2026-10까지는 계산이 66개의 `TracerWarning`과 함께 추적되었고, 추적된 계산과 다른 항을 섞은 손실은 즉시 실행 torch가 0.2721483665790으로 읽는 p = 0에서 그래디언트를 오류 없이 0.0으로 읽었습니다. 해결된 결과의 `reconstruct_field`는 한 층의 식을 다른 층에서 재생했고(0.00344 대신 d/dz -0.00406), 추적된 길이의 `Lattice.hexagonal`은 상수가 되었습니다. 즉시 실행 torch나 `torch.compile`을 쓰세요 |
| **`solve()`를 통한 `torch.func` grad 및 jvp 변환** (`torch.func.grad`, `vjp`, `jvp`, `jacrev`, `jacfwd`, `hessian`, 단독으로 또는 `vmap`과 합성) | 변환의 입력이 모드 연산자에 닿는 곳에서 이름을 밝혀 거부합니다. 그 1차 규칙의 보호 장치가 그런 수준을 2차 미분과 구별할 수 없기 때문입니다. 역방향 모드와 `forward_ad`가 모두 동작하고 `torch.func.vmap`을 통해서도 동작하는 `torch.autograd`를 쓰세요([Forward mode](#forward-mode)). 두께처럼 어떤 연산자에도 닿지 않는 매개변수는 `vmap(jacfwd)`를 포함해 그 각각에서 grad 모드를 켜든 끄든, 연산자에 붙잡힌 `requires_grad` 텐서가 있어도 정확합니다. 중첩된 grad 및 jvp 변환(`jvp(jvp)`, `hessian`, `vmap(hessian)`, 3차 이상)은 연산자가 미분을 운반하지 않는 동안 거기서 정확하며, 미분을 운반하면 거부됩니다. `torch.func.vjp`에 들어갈 때와 다른 grad 모드에서 호출한 `vjp_fn`은 그 역방향 패스가 틀릴 곳에서 이름을 밝혀 거부됩니다. 바깥의 순방향 모드 수준이 행렬을 움직이는 경우, 또는 수준 뒤에 추가된 `vmap`이 그 수준이 들어갈 때 있던 것 옆에 있는 경우입니다 |
| **층의 모드 연산자가 두 미분 모두에서 움직이는 2차 미분** (`jax.hessian`, `jacfwd(jacrev)`, `jvp(grad)`, `create_graph=True`, PyTorch의 혼합 차수) | 이름을 밝혀 거부합니다. 모드 규칙은 1차입니다. 두께처럼 어떤 연산자도 움직이지 않는 매개변수 하나는 거부되지 않습니다([Forward mode](#forward-mode)) |

통과하기 때문에 짚어 둘 만한 함정이 하나 있습니다. **솔버는 추적 밖에서 만드세요.** 추적되는
함수 *안에서* 솔버를 만드는 jit 테스트는 솔버를 pytree로 평탄화하지 않으므로, 어떤 키워드가 jit된
모든 계산에서 조용히 빠지는데도 그 키워드의 이름을 딴 테스트는 통과할 수 있습니다.

## 어떤 백엔드를 쓸지, 그리고 jit을 쓸지

취향의 문제가 아니며, 답은 모델링 경로마다 다릅니다. 형상 레시피에서는 jit한 JAX 단계가 모든
크기에서 이기지만, `truncation_order ~ 5` 아래의 래스터에서는 즉시 실행 PyTorch가 확실히 이깁니다.
형상 레시피에서 어느 쪽을 고르기 전에, 둘보다 영향이 큰 `shape_nvf_grid_size`를 먼저 보세요. 표는
[`backends-v1.md`](backends-v1.md)에, 측정은
[`../backend-choice.md`](../validation/rcwa/backend-choice.md)에 있습니다. 둘 다 이 호환성
인터페이스를 기술합니다. 현재 생애주기는 `jax.jit` 아래에서 추적된 형상을 거부하고, 형상은 즉시
실행 `jax.grad` 아래에서만 미분합니다(
[미분 가이드](../guide/common/autodiff.md#rcwa-which-inputs-differentiate-on-which-backend)를 보세요).

튜토리얼 [`02-jax`](https://github.com/fieldalgo/wavebit/blob/main/tutorials/rcwa/02-autograd-and-optimization-jax.ipynb)와
[`02-pytorch`](https://github.com/fieldalgo/wavebit/blob/main/tutorials/rcwa/02-autograd-and-optimization-pytorch.ipynb)는
각 백엔드에서 전체 최적화를 실행합니다.
