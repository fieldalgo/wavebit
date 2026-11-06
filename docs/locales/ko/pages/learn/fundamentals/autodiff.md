# 17. 자동미분과 최적화: 어느 방향으로 바꾸면 좋을까

:::{note}
다운로드 가능한 교육 스크립트는 현재 물리 Problem/Job API를 사용합니다. 일부 설명 코드와 그림에는 이전 API 표기가 남아 있으므로 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 함께 참고하세요. 과거 측정 파일과 원본 출처는 보존하며, 이전한 스크립트의 검증과 구분합니다.
:::

순방향 계산은 설계 변수에서 결과로 갑니다. 미분은 “설계 변수를 아주 조금 바꾸면 목표량이 얼마나 변하는가”를 계산합니다. 자동미분은 현재 실행한 **유한 차수 수치 프로그램의 미분**입니다. 물리 해가 충분히 수렴했는지는 별도의 질문입니다.

```{figure} ../../figures/learn/learn/16-autodiff.svg
:alt: 두께에서 행렬과 광학 응답을 거쳐 목표함수로 이어지는 미분 경로
:figclass: learning-figure

그림 17-1. 두께에서 행렬과 광학 응답을 거쳐 목표함수로 이어지는 미분 경로
```

균일한 `n=2` 박막 두께를 `d=900q` nm로 정하고 반사율을 줄여봅니다. `q`를 쓰면 이 예제의 설계 변수가 무차원입니다. 반환되는 기울기는 `dR/dq`이며 `dR/dd`와 수치가 다릅니다. 이 예제에서는 `dR/dd=(dR/dq)/900`입니다.

실행 파일: {download}`05_jax_gradient.py <../../../examples/rcwa/learning/learn/05_jax_gradient.py>`

```{literalinclude} ../../../examples/rcwa/learning/learn/05_jax_gradient.py
:language: python
:linenos:
```

| q = 0.1에서의 실제 계산 | 값 |
| --- | --- |
| R | 0.193241233492 |
| JAX dR/dq | 1.051632496046 |
| PyTorch dR/dq | 1.051632496046 |
| 중앙차분 dR/dq | 1.051632489893 |
| JAX와 중앙차분의 절대 차이 | 6.153e-09 |

중앙차분과 자동미분의 작은 차이는 **이 균일 박막의 이 점**에서 구현한 미분을 확인한 것입니다. 모든 형상·모든 변수의 미분을 보증하지 않습니다. 서로 다른 간격의 중앙차분을 살펴보고 수치 정밀도와 차수 변화도 함께 확인하는 습관이 필요합니다.

```{figure} ../../figures/learn/learn/17-optimization.svg
:alt: 교재 실행에서 실제로 얻은 반사율 감소와 두께 하한 도달
:figclass: learning-figure

그림 17-2. 교재 실행에서 실제로 얻은 반사율 감소와 두께 하한 도달
```

예제의 반복은 `q≥0.03` 제약의 하한에 도달합니다. 이후 값이 같은 것은 계속 개선되고 있다는 뜻이 아닙니다. 전역 최적해라는 주장도 아닙니다. 실제 설계에서는 제조 가능한 크기, 최소 간격, 재료 제약과 여러 파장·각도의 성능을 목표함수에 반영해야 합니다.

PyTorch로 같은 점의 기울기를 계산한 파일은 {download}`06_torch_gradient.py <../../../examples/rcwa/learning/learn/06_torch_gradient.py>`입니다. `.backward()` 뒤 `q.grad`를 읽으며, 반복 최적화에서는 기울기 누적을 의도에 맞게 초기화해야 합니다. 이 교재의 JAX 예제는 eager 실행이며 JIT 사용 예제로 오해하지 않습니다.

## JIT에서 구체적인 값으로 준비한 계산이 필요한 이유

일부 기하학 단계는 경계의 연결·포함 관계처럼 수치값을 보고 분기합니다. JIT 추적 중의 값만으로는 그 결정을 항상 할 수 없습니다. 추적 함수 바깥에서 구체적인 문제로 `rcwa.prepare(...)`를 호출하면 그 결정을 한 번 내리고, `prepared.run(problem)`이 추적 값을 그 결정에 따라 계산합니다.

```python
# JIT 준비의 개념 골격: 추적 바깥에서 구체적인 값으로 준비
prepared = rcwa.prepare(problem, numerics=numerics, backend="jax")
# JIT 함수 안에서는 갱신한 문제를 실행한다: prepared.run(updated_problem)
```

이 코드는 완전한 JIT 최적화 예제가 아니라 준비 단계의 의미를 보여줍니다. 재료값만 바뀌는 고정 형상은 준비한 계산을 재사용할 수 있습니다. 형상·접촉·덮임 관계가 바뀌면 다시 준비해야 합니다. 도형 경로가 기록하는 조합 관계와 래스터 경로가 기록하는 정보가 다르므로 모든 움직임을 같은 기준으로 처리하지 않습니다. `jax.jit` 안에서 추적한 도형은 `rcwa.Numerics(topology_check=...)`가 검사할 수 있는 경우에만 받습니다. [검사하는 도형 JIT](../../guide/rcwa/guarded-shape-jit.md), [물리 문제 미분](../../guide/common/autodiff.md), [NVF 교재](../nvf/index.md)를 참고합니다.
