# 물리 문제를 미분하기

변경할 수 없는(immutable) 선언 안에 살아 있는(live) JAX 또는 Torch 값을 넣고, 갱신한 문제 전체를 준비된 실행마다 넘기세요. 선언의 `.replace(...)` 연산은 이전 결과나 준비된 계약을 바꾸지 않고 새 입력을 만듭니다.

예를 들어 목적 함수는 재료나 층 두께를 바꾸고 `prepared.run(updated_problem)`에서 스칼라를 반환할 수 있습니다. 준비는 미분하는 함수 바깥에서, 호환되는 구체적인 값으로 하세요. 푸리에 차수, 배열 모양, 영역 위상(topology), 계산 격자 구조, 모니터 배치는 구조적 선택이며, 이를 바꾸려면 새로 준비해야 합니다.

백엔드의 변환 지원과 물리적 미분 가능성은 별개의 문제입니다. 고정된 관측점을 지나는 형상 경계는 어떤 재료가 표본으로 잡히는지를 바꿀 수 있습니다. FDTD는 이산 계산 격자와 소스/모니터 스텐실도 고정하므로, 미분은 그 이산화된 문제에 대한 것입니다. 검사가 추적한 값에 의존할 때는 유효성 정보를 확인하고, 최적화에 쓰기 전에 독립적인 섭동으로 그래디언트를 검증하세요.

JAX 배정밀도는 명시적인 x64 설정이 필요합니다. [백엔드](backends.md)를 참조하세요. 확립된 커널 수준 유도, 변환 검사, 유한 차분 비교, 이전 예제는 [호환 자동 미분 설명서](../../migration/autodiff-v1.md)와 [FDTD 미분 연구 가이드](../fdtd/fd-differentiation.md)를 참조하세요. 그 근거는 명시된 형상, 백엔드, 이산화 범위 안에서만 유효합니다.

## RCWA: which inputs differentiate on which backend

`rcwa.prepare(...)`와 `prepared.run(problem)`을 거칠 때, 백엔드마다 허용하는 살아 있는 입력이 다릅니다.

| 살아 있는 입력 | PyTorch (즉시 autograd) | JAX (`jax.grad`, `jax.jit`) |
| --- | --- | --- |
| 반공간 매질을 포함한 재료 계수 | 지원 | 지원 |
| 파장 | 지원 | 지원 |
| 층 두께 | 지원 | 지원 |
| 래스터 유전율 값 | 지원 | 지원. [고정한 래스터 준비](#a-held-raster-preparation-under-jit) 참조 |
| Cartesian Li(`factorization='li'`) 아래 고정 셀·회전각 0인 직사각형: 크기와 중심 | 지원 | `jax.jit`, `jax.jit(jax.grad(...))`, `jax.vmap` 아래에서 지원. 보호된 모서리 순서 |
| `Circle.radius`, `Rectangle.size`, `Rectangle.angle`, `Polygon.vertices` 같은 그 밖의 영역 형상 | 지원 | 즉시 `jax.grad` 아래에서 지원. 허용된 스칼라 형상에 대한 JIT/vmap은 명시적인 `Numerics(topology_check='host')`가 필요하며, 기본 장치 정책은 `prepared.host_topology`로 거부 |
| 셀, `Stack.period` | 지원 | 영역 형상과 같음. `Stack.lattice`도 마찬가지 |
| 존스 편광 쌍, `PlaneWave(polarization=(s, p))` | 지원 | 지원 |

Cartesian Li 실행 시점 보호 장치를 추가하기 전에, 지원 항목들을 예제 하나에서 중심 차분과 비교했으며 상대 일치는 4e-10에서 2e-7 사이였습니다. 확인한 PyTorch 형상 항목은 원 반지름(4.6e-9), 직사각형 폭(1.8e-8), 직사각형 각도(1.3e-7)였고, 래스터 항목은 `('sampled', 'li')`를 썼습니다. `jax.grad` 아래에서 형상 미분은 PyTorch의 미분과 상대적으로 4e-14에서 3e-13 사이로 일치하고(원 반지름, 직사각형의 폭, 중심, 각도, 그리고 주기 길이), 중심 차분과는 반지름에서 8e-10, 폭에서 4e-8, 각도에서 8e-8까지 일치합니다. 다각형 꼭짓점은 `factorization='laurent'` 아래에서 PyTorch와 1.4e-13까지 일치합니다. 주기나 존스 쌍은 `(px, 600.)`처럼 그래디언트를 추적하는 텐서를 담은 튜플일 수도 있고, `torch.stack([px, py])`처럼 텐서 하나일 수도 있습니다. 두 표기는 같은 PyTorch 미분을 줍니다(중심 차분 일치는 주기 길이에서 4.1e-10, 존스 성분에서 1.5e-8). 두 표기는 서로 다른 배치이므로, 실행할 때 쓸 표기로 준비하세요.

고정 셀 Cartesian Li 직사각형에는 배열 값으로 동작하는 모서리 순서 보호 장치가 있습니다. 그 크기와 중심은 닫히는 틈을 포함해 JIT와 vmap 아래에서 움직일 수 있습니다. 모서리가 교차하거나, 직사각형이 붕괴하거나, 셀을 벗어나면 `result.validity.checks['rectilinear_topology']`가 거짓이 되고, 그러면 전력, 진폭, 보관한 필드, 산란이 NaN이 됩니다. 접촉점에서 미분은 준비된 순서가 나타내는 쪽을 따릅니다. [실행 가능한 예제](../../../examples/rcwa/rectilinear_contact.py)를 참조하세요. 그 밖의 형상은 여전히 아래에 설명하는 구체적인 값 검사가 필요합니다.

`Polygon.vertices`는 `(N, 2)` 텐서로도, 스칼라 쌍 `N`개로도 미분됩니다. 두 표기는 서로 다른 배치이므로, 실행할 때 쓸 표기로 준비하세요. `factorization='laurent'` 아래에서 고정 위상 미분은 해석적 다각형 푸리에 변환을 따릅니다. 기본 스칼라 칠하기 NVF는 2026-11-13부터 경계 선적분을 써 왔습니다. 사라지는 구간은 기여를 잃고, 직선 분할은 가산적입니다. 그 미분은 국소 차분뿐 아니라 유한한 매개변수 구간에 걸쳐서도 검증하세요. 문서화된 겹치는 곡선 구간은 0.180%와 0.116%까지 맞아떨어지지만, 작은 다각형 면 기울기 꺾임은 남아 있습니다. 측정 범위는 [다각형 영역](../rcwa/boundary-modeling.md#polygon-regions)과 [자르기 계획 연속성](../../validation/rcwa/normal-field-known-limits.md)을 참조하세요. 법선 계산 격자와 푸리에 차수는 각각 독립적으로 세분화하세요.

### 즉시 `jax.grad`는 PyTorch의 가지를 따릅니다

`jit`나 `vmap` 없이 쓰면 `jax.grad`나 `jax.jacfwd` 트레이서는 PyTorch 텐서처럼 값을 지니며, 생애주기는 즉시 판정에 숫자가 필요한 곳마다 `jax.lax.stop_gradient`를 거쳐 그 값을 읽습니다. 따라서 즉시 `jax.grad` 아래에서 `rcwa.prepare`, `rcwa.check`, `Prepared.validate`, `rcwa.run`은 미분하는 지점에서 동작하고, 결맞는 소스, 적응형 `detuning`, 서로 다른 위상이 256개를 넘는 래스터도 PyTorch에서처럼 미분됩니다. 검사가 실패하면 PyTorch가 일으키는 오류를 그대로 일으킵니다. 음수 두께는 코드 `layer.thickness`인 `ValidationError`, 움직인 래스터 분할은 `prepared.raster_partition`, 유한하지 않은 관측값은 코드 `rcwa.nonfinite`인 `NumericalError`입니다. `jax.jit`나 `jax.vmap` 아래에서는 읽을 값이 없으므로 같은 경우들이 계속 거부됩니다. `rcwa.coherent_transform`, `prepared.detuning_transform`, `rcwa.raster_partition_transform`, `prepare.concrete`, `check.concrete`, `validate.concrete`입니다.

각각 예제 하나에서, 영역 유전율, 층 두께, 반지름에 대한 평면파 두 개짜리 미분, 레일리 점에서의 디튜닝(detuned) 미분, 그리고 위상 400개짜리 래스터 두 개의 미분은 두 백엔드 모두에서 중심 차분과 2e-7 이내로 일치합니다([`tests/api/test_eager_grad_parity.py`](../../../tests/api/test_eager_grad_parity.py)). JAX와 PyTorch는 영역 유전율을 빼면 반올림 오차 수준까지 일치합니다. 영역 유전율에서는 PyTorch의 미분이 Richardson 외삽값에서 1.1e-8(평면파 하나)에서 5.6e-8(두 개) 떨어져 있고, JAX의 미분은 2e-10 이내에 있습니다.

### 두께 미분과 반지름 미분

이 예제는 기본 장치 정책을 씁니다. 준비 하나가 목적 함수의 모든 실행을 담당합니다. JAX에서 층 두께는 `jax.jit` 아래에서도 미분되고, 반지름은 즉시 `jax.grad` 아래에서 미분되며 `jax.jit` 아래에서는 거부됩니다. PyTorch에서는 반지름도 미분됩니다. 작은 푸리에 차수와 64점 NVF 계산 격자는 빠른 시연용이며, 수렴한 설계가 아닙니다. 파장은 셀의 레일리 이상에서 벗어나 있습니다. 기판의 1차 차수가 스치는 0.9 µm(0.6 µm에 기판 굴절률 1.5를 곱한 값)에서는 파장 미분이 두 백엔드 모두에서 NaN입니다.

```python
import jax
jax.config.update('jax_enable_x64', True)
import torch
from wavebit import Material, Units, UnsupportedFeatureError, rcwa

def problem(radius, thickness):
    return rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(
            period=(.6, .6), bottom=Material(epsilon=2.25),
            layers=(rcwa.Layer(thickness=thickness, regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=radius), material=Material(epsilon=12.25)),)),)),
        excitation=rcwa.Excitation(wavelength=.85, sources=(rcwa.PlaneWave(),)))

numerics = rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=64)

on_jax = rcwa.prepare(problem(.12, .18), numerics=numerics, backend='jax')
T00 = lambda t: on_jax.run(problem(.12, t)).transmission.order(0, 0).efficiency[0]
print(jax.jit(jax.grad(T00))(.18))           # -21.87787...
T00_r = lambda r: on_jax.run(problem(r, .18)).transmission.order(0, 0).efficiency[0]
print(jax.grad(T00_r)(.12))                  # -67.08197...
try:
    jax.jit(jax.grad(T00_r))(.12)
except UnsupportedFeatureError as error:
    print(error.code)                        # prepared.host_topology

on_torch = rcwa.prepare(problem(.12, .18), numerics=numerics, backend='torch')
radius = torch.tensor(.12, dtype=torch.float64, requires_grad=True)
on_torch.run(problem(radius, .18)).transmission.order(0, 0).efficiency[0].backward()
print(float(radius.grad))                    # -67.08197...
```

두 반지름 미분은 서로, 그리고 NumPy에서의 중심 차분(단계 `1e-6`)과 2e-8 이내로 일치합니다.

### Why JAX-traced geometry is refused under jit

미분은 두 백엔드에서 같은 계산입니다. 두 백엔드가 모두 허용하는 입력에서는 반올림 오차 수준까지 일치합니다. 층 두께 미분 하나는 PyTorch에서 7.695586811926e-4, `jax.jit` 아래에서 7.695586811925e-4로 측정되었습니다. 두 백엔드는 그 계산 전에 실행되는 검사 하나에서만 다릅니다.

보호되는 Cartesian Li 부분집합 바깥에서는 준비 단계가 형상의 조합 구조, 즉 어떤 영역이 겹치고 어떤 모서리가 교차하는지를 기록합니다. 꼭짓점 좌표는 살아 있는 값으로 남아 그래디언트를 운반합니다. 실행마다 새 형상이 여전히 그 기록에 맞는지 검사하며, 이 검사는 구체적인 값에 따라 분기합니다. PyTorch 텐서는 값을 지니므로 즉시 실행마다 검사가 이루어지고, 위상 경계를 넘는 단계는 코드 `prepared.topology`인 `IncompatibleProblemError`를 일으킵니다. 그 형상은 다시 준비하세요.

두 JAX 변환은 여기서 서로 다릅니다. `jax.jit` 아래에서 트레이서는 모양과 dtype은 있지만 값이 없으므로 검사를 실행할 수 없습니다. 검사를 건너뛰면 형상이 더 이상 맞지 않는 기록을 재생해 오류 없이 틀린 답을 반환할 수 있습니다. `jit` 없는 `jax.grad` 아래에서는 트레이서의 원시값(primal)이 구체적입니다. 실행마다 `jax.lax.stop_gradient`를 거쳐 그 값을 읽고, PyTorch 텐서의 값을 검사하듯 그 사본에서 위상을 검사합니다. 풀이는 추적한 값과 그 접선을 그대로 유지합니다. 위상 경계를 넘는 grad 단계는 PyTorch에서처럼 `prepared.topology`를 일으킵니다. 생애주기는 그 읽기를 시도해 보는 방식으로 두 경우를 구별합니다. `jit` 안의 `jax.grad`를 포함해 `jax.jit`나 `jax.vmap` 아래에서는 읽기가 실패하고, 추적을 만들 때 형상을 `prepared.host_topology`로 거부합니다. Cartesian Li 부분집합은 그 검사를 배열 연산으로 제공합니다. 일반 NVF와 칠하기 곡선 위상은 기본 장치 정책에서 계속 거부됩니다. 명시적인 `Numerics(topology_check='host')`는 대신 공유 구체적 값 검사를 JAX CPU 콜백으로 실행하고, 잘못된 재사용에서는 NaN 출력과 함께 `shape_topology=False`를 반환합니다. vmap 아래에서 검사는 순차적으로 실행되며, 수치 입력과 미분은 JAX 그래프를 유지합니다. 범위와 비용은 [JAX 변환 안에서 형상 위상 검사하기](../rcwa/guarded-shape-jit.md)를 참조하세요.

이는 결함이 아니라 생애주기가 명시한 계약입니다. [통합 API 설계](../../design/unified-api.md#7-prepared-reuse-and-differentiation)는 제한 없는 형상 미분을 약속하지 않으며, 호스트에서만 할 수 있는 위상 판정을 검사할 수 없을 때는 추적을 만드는 시점에 거부하도록 요구합니다. 재료, 파장, 층 두께 미분은 `jax.jit` 아래에서도 JAX 그래프를 유지합니다.

[접촉](../../../examples/rcwa/nvf_contact.py), [접하는 타원](../../../examples/rcwa/nvf_curved_contact.py), [칠한 곡선](../../../examples/rcwa/nvf_curved_painting.py)에 대한 현재 예제는 명시적 호스트 정책을 씁니다. [호환 자동 미분 설명서](../../migration/autodiff-v1.md)는 과거 재현을 위해 이전의 검사 없는 추적 재생 작업 흐름을 보존합니다. 그 성능 측정은 콜백 정책의 속도를 입증하지 않습니다.

### jit 아래 사각이 아닌 셀에서의 동점 표본

*2026-11-11 측정: CPU, x64를 켠 JAX 0.12.0.1, NumPy 2.5.4.*

육각형이나 그 밖의 사각이 아닌 셀에서 형상 NVF 법선장은 각 표본을 형상 중심의 가장 가까운 주기 이미지로 옮깁니다. 일부 표본은 두 이미지에서 같은 거리인 Wigner–Seitz 모서리 위에 정확히 놓입니다. 그곳에서는 어느 이미지든 유효하며, 접기(fold)는 다른 후보가 엄격히 더 가깝지 않으면 첫 번째 후보를 유지합니다. NumPy, PyTorch, 즉시 JAX는 표본을 같은 비트로 놓고 같은 선택을 합니다. `jax.jit` 아래에서는 XLA가 거리 제곱 산술을 융합해 동점 후보 하나를 1 ulp 다르게 반올림할 수 있고, 그러면 접기가 다른 이미지를 유지합니다. 중심에 형상이 있는 육각형 셀에서 이런 일은 `shape_nvf_grid_size=21`에서 441개 표본 중 1개, 65에서 4,225개 중 2개, 기본 계산 격자(1024)에서 1,048,576개 중 8개에서 일어났습니다. 그 표본에서 법선이 뒤집히므로, jit한 실행은 아주 조금 다른 층을 풉니다.

단위 육각형 셀 안의 원(반지름 0.28)과 기울어진 타원에서, 차수 (3, 3), s 편광 평면파 하나로 측정했습니다. 표는 준비 하나를 거친 `jax.jit`와 즉시 JAX를 비교합니다.

| 양 | 계산 격자 21 | 계산 격자 65 | 기본 계산 격자 |
| --- | --- | --- | --- |
| 영역의 `d(sum T)/d(eps)`, 상대값 (원, 타원) | 2.6e-6, 3.4e-6 | 5.5e-6, 2.4e-6 | 9.2e-8, 4.1e-8 |
| `sum T`, 절댓값 (원, 타원) | 1.4e-7, 1.0e-8 | 2.2e-7, 8.9e-8 | 3.0e-9, 6.8e-10 |
| `topology_check='host'` 아래 `d(sum T)/d(radius)`, 상대값 (원) | 6.0e-5 | 6.5e-5 | 5.9e-8 |

미분 차이는 계산 격자 자체의 이산화 오차보다 작습니다. 계산 격자 21과 65에서는 그 계산 격자에서 기본 계산 격자로 갈 때의 미분 변화보다 1.2–315배 작습니다. 기본 계산 격자에서 유전율 미분의 차이는 1024점에서 2048점으로 갈 때의 변화보다 4.6배(원), 15배(타원) 작습니다. `sum T`가 늘 그 오차보다 작지는 않습니다. 계산 격자 65에서 원의 jit 차이(2.2e-7)는 계산 격자 65에서 기본 계산 격자로 갈 때의 변화(1.4e-7)보다 큽니다. 기본 계산 격자에서 동점에 표본을 두지 않는 정사각형 셀과 비스듬한 셀(`Lattice.oblique(1., 1.1, 1.3)`)에서는 같은 비교가 4e-14와 6e-14를 줍니다.

이 현상은 사각이 아닌 셀(직사각형 셀은 후보 없이 접습니다)에서, 표본을 셀 모서리에 두는 계산 격자에서, 그리고 `jax.jit(jax.grad(...))`를 포함한 `jax.jit` 아래에서만 일어날 수 있습니다. 즉시 JAX는 NumPy(여기서 `sum T`는 5.6e-15까지)와도, PyTorch와도 일치합니다([`tests/api/test_eager_grad_parity.py`](../../../tests/api/test_eager_grad_parity.py)가 셀 모서리에 표본이 있는 육각형 계산 격자를 확인합니다). `jax.jit` 아래에서 추적한 반지름이나 중심은 기본 `topology_check='device'`가 거부합니다(`prepared.host_topology`). 이를 허용하는 `topology_check='host'` 아래에서는 마지막 행이 보여 주듯 형상 미분이 같은 동점을 지닙니다. 동점 규칙은 의도적으로 유지합니다. jit 아래에서 비트 단위로 재현되는 규칙은 동점 표본에서 NumPy와 PyTorch 값을 바꿀 것이기 때문입니다. jit한 결과가 이보다 더 가깝게 즉시 JAX와 맞아야 한다면, 즉시 실행하거나 계산 격자를 수렴시키세요.

### A held raster preparation under jit

래스터 값은 살아 있는 값으로 남지만, 일부 푸리에 경로는 준비할 때 어떤 픽셀이 같은 위상을 공유하는지 기록합니다. `jax.jit` 아래에서 위상 분할이 준비된 것과 다른 추적 패턴을 받은 실행은 NaN 출력을 반환하고, `result.validity.checks['raster_partition']`가 거짓이 됩니다. 낡은 답을 반환하지는 않습니다. PyTorch에서는 같은 실행이 `IncompatibleProblemError`를 일으킵니다. 패턴이 추적의 상수이고 파장, 각도 또는 두께만 추적하는 경우에는 즉시 실행과 같은 방식으로 검사합니다. 기록하는 경로에서 패턴이 움직이면 즉시 실행과 같은 오류인 코드 `prepared.topology`의 `IncompatibleProblemError`를 일으킵니다.

| `fourier_coefficients`, `factorization` | 고정한 준비 하나를 거친 움직인 패턴 |
| --- | --- |
| `'sampled'`, `'li'` | 올바름. 아무것도 기록하지 않음 |
| `'sampled'`, `'nvf'` | 추적: jit 아래 NaN. 추적의 상수, 그리고 PyTorch: 오류 발생 |
| `'integrated'`, `'li'` | 추적: jit 아래 NaN. 추적의 상수, 그리고 PyTorch: 오류 발생 |

jit 아래에서 패턴이 매 단계 움직인다면 `('sampled', 'li')`를 쓰거나, 구체적인 패턴마다 변환 바깥에서 준비하세요. 기록하는 경로는 jit 아래에서 서로 다른 위상 256개까지만 분할을 보호합니다. 그보다 많으면 추적한 래스터를 `rcwa.raster_partition_transform`으로 거부하며, 즉시 `jax.grad`와 PyTorch는 그 값으로 검사합니다.

### 셀 미분 아래 래스터의 법선장

`factorization='nvf'` 아래에서 래스터의 법선장은 셀 모양에 의존하므로, `Stack.period`나 `Stack.lattice`에 대한 미분은 법선장 자체의 변화를 포함합니다. `rcwa.prepare`는 법선장과, 그것을 만들 때의 선택(예를 들어 어떤 경계면 픽셀이 가장 가까운지)을 기록합니다. `prepared.run`은 기록된 법선장을 쓰고 그 미분을 미분하는 셀에서 다시 만들므로, 미분은 실행이 반환하는 답의 미분입니다.

| `Numerics(raster_normal_method=...)` | `prepared.run`을 거친 셀 미분 |
| --- | --- |
| `'distance'`(기본값), `'spectral'`, `'smooth'` | PyTorch와 즉시 `jax.grad` 아래에서 지원 |
| `'contour'` | 미분을 취할 때 `ValidationError` 코드 `rcwa.invalid`로 거부. 셀을 미분하지 않는 실행은 영향 없음 |

2026-11-10까지는 기록된 법선장을 고정해 두었으므로, 준비를 거친 셀 미분에서 법선장의 몫이 빠졌습니다. 기울어진 셀의 8 x 8 래스터에서 미분의 106%까지, `period` 미분의 27%까지였습니다. 래스터 값, 두께 또는 파장에 대한 미분은 법선장과 무관했으므로 바뀌지 않았습니다. `'contour'`는 다른 셀에서 법선장을 다시 만들 수 없으므로, 고정하는 대신 거부합니다. 측정은 [NVF 법선 방법](../../theory/rcwa/nvf-normal-methods.md#the-cell-derivative-through-a-record)을 참조하세요.

### Backend speed

[백엔드 선택](../../validation/rcwa/backend-choice.md)의 측정은, 형상 맞춤에 jit한 JAX를 권장하는 내용을 포함해, 이전 솔버 인터페이스(`freeze_geometry()`와 `solve(geometry=...)`. 2026-10-15에 `wavebit.compat`와 함께 퇴역했으며 과거 체크아웃에서 실행됩니다)를 거쳐 얻은 것입니다. 이 생애주기나 그 Cartesian Li 실행 시점 보호 장치, 호스트 콜백 정책의 측정이 아닙니다. `prepared.run`을 거친 백엔드 속도는 아래에서 측정합니다.

`prepared.run`을 거친 미분 한 단계 -- 실행과 투과 합의 그래디언트 -- 는 CPU 하나(Apple M4, 2026-10-16, `99aedad4`의 `main`, PyTorch 2.14.1, JAX 0.11.0)에서 다음 시간이 걸렸습니다. 각 칸은 PyTorch / 즉시 `jax.grad` / `jax.jit(jax.grad)`입니다. 원(600 nm 셀 안의 반지름 160 nm, 256점 계산 격자의 NVF)은 `jax.jit` 아래에서 이를 허용하는 정책인 `Numerics(topology_check='host')`를 쓰고, 260 x 200 nm 직사각형은 기본 보호 장치가 허용하는 Cartesian Li를 쓰며, 두께는 원 문제의 두께이고 기본 수치 설정을 씁니다.

| `orders` | 원 반지름 | 직사각형 폭 | 층 두께 |
| --- | --- | --- | --- |
| `(3, 3)` | 0.036 / 0.17 / 0.023 s | 0.027 / 0.16 / 0.019 s | 0.025 / 0.12 / 0.018 s |
| `(5, 5)` | 0.20 / 0.28 / 0.11 s | 0.19 / 0.25 / 0.10 s | 0.17 / 0.21 / 0.089 s |
| `(7, 7)` | 0.81 / 0.81 / 0.49 s | 0.74 / 0.75 / 0.49 s | 0.68 / 0.63 / 0.40 s |
| `(9, 9)` | 2.65 / 2.65 / 1.93 s | 2.78 / 2.56 / 1.90 s | 2.39 / 2.26 / 1.68 s |

각 항목은 첫 호출 뒤 다섯 번 호출의 중앙값이며, 구성마다 새 프로세스에서, 다른 작업과 공유하는 기계에서 측정했습니다(시작 시 부하 평균 4.1, 끝날 때 8.8). 가장 빠른 호출은 중앙값의 9% 이내였습니다. 즉시 `jax.grad`는 프로세스의 첫 호출에서 만나는 연산마다 컴파일하느라 9.6에서 16 s를 더 쓰고, `jax.jit(jax.grad)`는 1.4에서 4.5 s 만에 컴파일했으며, PyTorch의 첫 호출은 중앙값보다 많아야 0.2 s 더 걸렸습니다. 따라서 jit한 단계는 측정한 모든 차수에서, 두께와 마찬가지로 형상에서도 가장 빨랐습니다. PyTorch의 1.4에서 1.9배입니다. `jit` 없이 `jax.grad`는 낮은 차수에서 PyTorch보다 느리고(반지름의 경우 `(3, 3)`에서 4.7배), `(7, 7)`부터는 비슷합니다. 그날 아침 더 부하가 큰 기계(부하 평균 약 9)에서 PyTorch 2.13.0으로 한 이전 실행은 더 느린 PyTorch 단계 -- `(9, 9)`에서 반지름에 대해 2.65 s 대비 6.4 s -- 를 측정했으며, 버전과 부하의 영향을 분리하려고 반복하지는 않았습니다. 그 기록은 연구의 이력에 있습니다. 스크립트와 그 기록은 [`studies/rcwa/lifecycle_backend_speed/`](https://github.com/fieldalgo/wavebit/tree/main/studies/rcwa/lifecycle_backend_speed)에 있습니다.
