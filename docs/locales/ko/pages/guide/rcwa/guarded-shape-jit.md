# JAX 변환 안에서 형상 위상 검사하기

준비된 형상 계산은 경계끼리 교차하는지, 어느 영역이 다른 영역을 덮는지,
법선장에 어떤 주기 이미지가 필요한지 등의 이산적인 결정을 기록합니다.
연속 매개변수를 움직이면 이 결정이 무효가 될 수 있습니다. 오래된 결정으로
계산한 결과가 에너지를 보존하더라도 재사용이 유효하다는 뜻은 아닙니다.

기본값 `Numerics(topology_check='device')`는 고정 셀의 Cartesian Li 직사각형 검사를
허용합니다. 다른 형상의 구체적인 위상 검사는 일반적인 장치 배열 연산으로
실행할 수 없으므로 JIT나 vmap 아래에서는 tracing 전에 거절합니다.
Eager `jax.grad`는 구체적인 primal을 검사할 수 있습니다.

**스칼라 재료 영역층과 평면파 하나**를 사용하는 JAX 계산에서 기존의 구체적인
검사를 CPU에서 실행하려면 `Numerics(topology_check='host')`를 사용하세요.
원, 타원, 직사각형과 지원되는 다각형 painting을 다루며, 선택한 엔진이 traced
계산을 지원하는 경우 셀과 방향 매개변수도 포함합니다. 텐서, 래스터, 결맞은
다중 소스의 변환 지원을 추가하지 않으며, 지원되는 projector와 형상 조합도
바꾸지 않습니다.

```python
import jax
jax.config.update('jax_enable_x64', True)
from wavebit import Material, Units, rcwa

def design(radius):
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=180., regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=radius), material=Material(4.)),)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization='p'),)))

numerics = rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=64, topology_check='host')
prepared = rcwa.prepare(design(120.), numerics=numerics, backend='jax')
objective = lambda radius: prepared.run(design(radius)).T.sum()
value, derivative = jax.jit(jax.value_and_grad(objective))(121.)
```

순수 JAX 콜백은 형상 값을 CPU로 보내 유효성 불리언을 반환합니다.
새 형상이나 광학 계산 결과를 반환하지 않습니다. 수치 계산과 미분 가능한 입력은
JAX 그래프를 유지합니다. vmap 아래에서 검사는 형상마다 순차 실행하므로,
특히 복잡한 painting에서 처리량을 제한할 수 있습니다. 이는 명시적인 호스트
전송 정책이며 완전히 장치 안에서 실행된다는 주장이나 백엔드 속도 측정이 아닙니다.
CPU JAX 실행은 회귀 테스트가 다루며, 가속기 성능은 측정하지 않았습니다.
호스트 정책의 원과 기본 정책의 Cartesian Li 직사각형에 대해 jitted step을
CPU에서 측정한 결과는 미분 가이드의 [백엔드 속도](../common/autodiff.md#backend-speed)에 있습니다.

호스트 검사가 현재 형상을 거절하면 `result.validity.checks['shape_topology']`가
False입니다. 회절 파워와 진폭, 유지된 장, 경계 장, 산란행렬이 NaN이 됩니다.
연결된 순방향·역방향 미분에도 NaN이 전파되므로, 거절된 계산이 기울기 0인
최적점처럼 보이지 않습니다. 미분 대상 입력과 독립인 출력의 미분은 구조적으로
0일 수 있습니다. Jacobian의 유한성을 실행 허용 여부의 판정으로 쓰지 마세요.

RCWA의 `result.validity.checks['numerical_finite']`는 내부 계산이 유한하지 않거나
**실행 허용 검사 중 하나라도 실패하면** False입니다. `shape_topology` 등 다른
검사로 실패 사유를 확인하세요. True라도 수렴이나 모든 선택적 관측량의 정의를
보증하지 않습니다. 예를 들어 법선 입사 플럭스가 없는 소스의 효율은 정의되지
않을 수 있습니다. 최적화에서는 실패한 검사를 확인해 스텝을 거절한 뒤에야 다음
스텝을 수용하거나 불확도를 추정해야 합니다.

무효한 점 뒤에도 유효한 점은 올바르게 계산되며, 마지막 호출의 형상을 캐시하지
않습니다. Eager 실행의 무효한 재사용은 여전히 `IncompatibleProblemError`를 내고,
`prepared.validate(concrete_problem)`은 eager 검사 보고서를 반환합니다.
위상이 바뀌면 구체적인 입력으로 다시 준비하세요. 위상 변화에 걸친 매끄러운
미분은 보장하지 않습니다.

이 정책은 재사용을 검사하며 광학적 수렴을 판단하지 않습니다. 특히 painting의
유한한 법선장 격자와 다각형 근사는 기존 미분 한계를 그대로 가집니다.
새로 준비한 계산의 유한차분과 미분을 비교하고 수치 해상도를 독립적으로 높이세요.

겹친 다각형 근사 곡선에서 스칼라 painting 법선장은 경계 선적분을 사용하므로 사라지는 클리핑 구간의 기여가 연속적으로 줄어듭니다. 위상이 바뀌면 다시 준비하세요. [클리핑 계획 연속성 검증](../../validation/rcwa/normal-field-known-limits.md)은 재료 라벨 수정, 선적분과 적용 범위를 설명합니다. 물리적 접촉, 다각형 근사, 유한한 법선 격자는 여전히 독립적인 수렴 및 미분 검증이 필요합니다.

## 최적화 중 자동 재준비

기하 처리 계획이 바뀌어도 최적화를 계속하려면
`rcwa.compile(build, observe, numerics=...)`를 사용하세요.
`build(parameters)`는 `Problem`을, `observe(result)`는 계산할 스칼라나 배열을
반환합니다. 제어 객체는 매번 구체적인 후보를 호스트에서 검사합니다. 기존 계획이
맞으면 재사용하고, 맞지 않으면 **바로 그 새 후보**로 다시 준비하고 컴파일하여
그 점의 값과 미분을 반환합니다. 이전 매개변수로 되돌리거나 최적화기에 위상 예외의
복구 처리를 요구하지 않습니다.

```python
evaluate = rcwa.compile(design, lambda result: result.T.sum(), numerics=numerics)
value = evaluate(121.)
value, gradient = evaluate.value_and_grad(121.)
value, jacobian = evaluate.value_and_jacobian(121.)
```

Python 최적화 루프나 SciPy에서 직접 호출하세요. 내부에서 JAX JIT를 관리하므로
제어 객체를 다시 `jax.jit`, `jax.grad`, `jax.vmap`으로 감싸지 않습니다.
실수 스칼라 목적함수에는 `value_and_grad`, 배열 관측량에는
`value_and_jacobian` 또는 `jacobian`을 쓰세요. 매개변수는 구체적이고 유한한
실수 부동소수점 배열 또는 스칼라여야 합니다. `build`와 `observe`는 순수 함수이며
JAX tracing이 가능해야 합니다. 모델 구조와 출력 shape를 유지하고, 바뀌는 모든
입력은 외부의 가변 상태가 아닌 매개변수로 전달하세요.

이 진입점은 명시적으로 `topology_check='host'`를 사용하며 평면파 하나와 스칼라
재료 지원 범위를 따릅니다. 다른 수치 설정은 유지합니다. 준비 하나와 각 값·미분
메서드의 컴파일 함수를 보관하므로, 위상이 바뀌면 컴파일 비용이 다시 들 수 있습니다.
`evaluate.preparation_count`는 성공적으로 적용한 준비 횟수입니다. 실패한 후보는
마지막 정상 계획을 덮어쓰지 않습니다. 예전 위상으로 돌아갈 때도 재준비할 수
있으며, 과거의 기하 계획이나 계산 답을 계속 쌓아두지 않습니다.

기하 재사용 검사 실패만 재준비합니다. 잘못된 치수, 비유한 입력, 미지원 요청,
선택한 관측량이나 미분의 비유한 결과는 명시적인 오류로 남습니다. 호스트 사전
검사를 통과한 이전 계획이 컴파일된 검사에서 거절되면 같은 매개변수로 한 번
재준비해 재시도합니다. 새 계획도 실패하면 무한 재시도하지 않고 오류를 알립니다.

[자동 재준비 예제](../../../examples/rcwa/automatic_repreparation.py)는 수동 재시도
처리 없이 SciPy의 잔차·Jacobian 호출로 사각형의 위치를 맞추며, 겹침에서 분리로
넘어갑니다. CPU JAX를 회귀 테스트로 검증하고 GPU 실행은 측정하지 않았습니다.
재준비는 지원되는 새 형상을 계산할 수 있게 하지만, 정확한 위상 경계에서의 매끄러운
미분, 최적화기의 수렴, 법선장 격자의 수렴을 보장하지 않습니다. 저수준
`Prepared.run`은 위에서 설명한 고정 계획 계약을 유지합니다.

실행 가능한 예제는 [직사각형 접촉](../../../examples/rcwa/nvf_contact.py),
[접하는 타원](../../../examples/rcwa/nvf_curved_contact.py),
[겹치는 곡선](../../../examples/rcwa/nvf_curved_painting.py)을 다룹니다.
원래 형상, 재료, Fourier 차수, projector, 독립 검사를 유지합니다.
