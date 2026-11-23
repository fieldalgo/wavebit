# 일반 FDTD 엔진 미분하기

:::{note}
이 페이지는 이전 엔진 선언과 측정으로 정한 그 허용 규칙을 보존합니다. 기둥 장면은 이제 [현재 API로 작성한 완전한 예제](substrate-gradients.md)가 있으며, 이 예제는 아래쪽 열린 면을 관통해 이어지는 스무딩 기판을 명시적으로 사용합니다. 현재 모듈도 같은 엔진과 검사를 실행합니다. [In the current module](#in-the-current-module)에서 아래 선언이 현재 모듈의 무엇에 대응하는지 정리합니다.
:::

:::{admonition} 이전 인터페이스
:class: earlier-interface

2026-10-15에 `wavebit.compat`와 함께 퇴역한 이전 인터페이스입니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`wavebit.compat.legacy.fd`는 NumPy, JAX, PyTorch에서 실행되었고, 뒤의 두 백엔드에서는 실행을
미분할 수 있었습니다. 재료 값, 물체의 모양과 위치, 소스의 진폭, 위상, 지연과 대역, 그리고 Bloch 위상이
모두 `jax.grad`나 Torch autograd의 잎(leaf)이 될 수 있습니다. 이 페이지는 그렇게 설정하는 방법, 무엇을
추적(trace)할 수 있는지, 그리고 추적한 값이 실행이 보증할 수 있는 영역을 벗어나면 실행이 무엇을 하는지
설명합니다. 그 근거가 되는 측정은
[`fd-subpixel-occupancy.md`](../../validation/fdtd/fd-subpixel-occupancy.md) 끝부분의 C6 절들이며, 그중 마지막 절은
"Traced material values beside traced geometry"입니다.
:::

## 단 하나의 규칙: 계획은 구체적인 값으로, 실행은 추적한 값으로

:::{admonition} 이전 인터페이스
:class: earlier-interface

2026-10-15에 `wavebit.compat`와 함께 퇴역한 이전 인터페이스입니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

실행은 `fd.plan`(정적이며 호스트에서 동작합니다. 계산 격자, 경계, 모든 거부)에 이어 `fd.lower`와
시간 진행(그래디언트가 흘러가는 배열)으로 이루어집니다. 계획은 **한 번, 일반 숫자로** 만드세요 --
`plan = fd.plan(scene)` -- 그리고 추적하는 모든 실행에 `fd.run(traced_scene, 'torch', plan=plan)`
또는 `'jax'`처럼 넘기세요.

계획은 숫자로만 확인할 수 있는 것 -- 스무딩 연산자의 안정성, 기둥 아래 접합부의 배치, 흡수층이 받아들일
수 있는 차수 -- 을 계획을 만든 값에서 보증합니다. 추적하는 실행은 그 계획을 재사용하고, 살아 있는 값을
읽으며, 실행 시점에 움직일 수 있는 값을 검사합니다([When the result is NaN](#when-the-result-is-nan)
참조). JAX에서는 계획이 컴파일된 시간 진행 프로그램을 재사용할 수 있게 해 주는 것이기도 합니다. 값을 몇
개 시도하든 계획 하나에 컴파일은 한 번입니다.
:::

### In the current module

`wavebit.fdtd`는 같은 규칙을 자체 이름으로 유지합니다. 계획은 일반 숫자로 만든
`prepared = fdtd.prepare(problem, numerics=numerics, backend='torch')`(또는 `'jax'`)이고, 그래디언트가
흘러가는 추적 실행은 `prepared.run(traced_problem)`입니다. 물체가 예약하는 셀은 `range=` 대신 구조의
이름을 키로 하는 `fdtd.Numerics(geometry_ranges={'pillar': RANGE})`로 지정합니다. 재료 잎은
`Material(epsilon=...)`이고, 투과한 0차는 `result['T'].efficiency[0, 0, 0]`이며, 실행 시점 검사가 거부한
실행은 `result.validity.checks['runtime_stability']`가 거짓입니다. 추적한 `Bloch` 위상이나 `InvariantAxis`의
`k`는 아직 준비된 계산에서 살아 있는 값이 아닙니다. `prepared.run`은 이를 정적 필드로 보고 거부합니다.

## 기둥의 반지름, 높이, 굴절률을 함께

:::{admonition} 이전 인터페이스
:class: earlier-interface

2026-10-15에 `wavebit.compat`와 함께 퇴역한 이전 인터페이스입니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

메타표면 셀 하나입니다. 유리 기판(아래쪽 PML을 관통하는 스무딩 `fd.Box`)과 그 위에 맞닿아 선 원기둥
기둥이 있고, 수직 입사 평면파가 비춥니다. 기둥의 반지름, 높이, 유전율이 설계 변수이고, 목적 함수는 투과한
0차입니다. 어느 백엔드에서든 장면과 그 계획은 다음과 같습니다.

```python
# Historical engine declaration; see substrate-gradients.md for the physical API.
from wavebit.compat.legacy import fd

PLANE = -0.2375                                   # the substrate's top face
GLASS = fd.Material(eps=2.1)
RANGE = ((-.145, .145), (-.145, .145), (PLANE - .001, PLANE + .401))


def cell(radius, height, eps):
    pw = fd.PlaneWave('pw', z=.45, direction='-z', polarization='s',
                      band=fd.Band(wavelength=1., df=.15))
    pillar = fd.Cylinder(radius=radius, height=height, axis='z',
                         center=(.011, -.007, PLANE + height / 2), range=RANGE)
    return fd.Scene(
        domain=fd.Domain(size=(.6, .6, 2.1), center=(0., 0., .05)),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=(
            fd.PML(cells=8, wavelength=1., background=GLASS, allow_thin=True),
            fd.PML(cells=8, wavelength=1., allow_thin=True))),
        resolution=fd.Resolution(dx=.05), run=fd.Run(steps=600, courant=.5),
        sources=(pw,),
        structures=(fd.Structure(fd.Box(size=(10., 10., 2.), center=(0., 0., PLANE - 1.)), GLASS),
                    fd.Structure(pillar, fd.Material(eps=eps))),
        monitors=(fd.Diffraction('T', z=-.55, side='transmit', wavelengths=(1.,), orders=0),))


plan = fd.plan(cell(.12, .3, 2.25))
```

PyTorch에서는 다음과 같습니다.

```python
import torch                                      # earlier interface, as above

leaves = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (.12, .3, 2.25)]
result = fd.run(cell(*leaves), 'torch', plan=plan)
T = result['T'].orders.incident.T[0, 0, 0]
T.backward()
print(float(T), [float(v.grad) for v in leaves])
```

JAX에서는 다음과 같습니다. 배열을 만들기 전에 64비트 숫자를 켜세요.

```python
import jax                                        # earlier interface, as above
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


def transmitted(v):
    result = fd.run(cell(v[0], v[1], v[2]), 'jax', plan=plan)
    return result['T'].orders.incident.T[0, 0, 0]


print(jax.grad(transmitted)(jnp.array([.12, .3, 2.25])))
```

둘 다 같은 그래디언트를 출력합니다(반지름 `0.1933`, 높이 `-0.0818`, eps `0.00625`).

이 장면에서 모양을 추적할 수 있게 만드는 것은 다음 세 가지입니다.

* **`range=`** 인자는 스윕 어디에서든 기둥이 차지할 수 있는 셀을 예약합니다. 실행에 묻는 모든 크기와 위치는
  물체 전체를 그 범위 안에 두어야 합니다.
* **높이가 중심과 함께 움직입니다.** 중심이 `PLANE + height / 2`이므로 밑면이 기판 면 위에 머뭅니다.
  밑면이 면을 벗어난 기둥은 계획의 접합부가 상정한 기둥이 아닙니다.
* **기판의 모양은 구체적인 값으로 둡니다.** 기판의 유전율도 추적할 수 있습니다 -- 첫 번째 구조에 잎을
  넣은 `fd.Material(eps=...)`를 넘기세요 -- 기판이 아래쪽 PML을 채우더라도 그렇습니다. 너무 가파르게
  빠져나가는 차수를 PML이 흡수하도록 요구받지 않게 하는 규칙은 실행 시점에 유지됩니다(아래 참조).
:::

## 해석적 응답에서 비활성 Lorentz 항

:::{admonition} 이전 인터페이스
:class: earlier-interface

2026-10-15에 `wavebit.compat`와 함께 퇴역한 이전 인터페이스입니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

유한한 입력에서 `fd.Lorentz(..., strength=0).eps_term(omega)`는 감쇠 없는 공진점에서도 0을 기여합니다.
`fd.eps_at`은 극점을 합할 때 이 규약을 이어받습니다. 정확히 비활성인 진동자는 감수율을 기여하지 않으므로,
분모를 계산하기 전에 그렇지 않으면 부정형이 될 `0 / 0`을 제거합니다. 이 지름길은 구체적이고 정확히 0인
세기를 요구합니다. 0이 아닌 작은 세기를 없애는 허용 오차가 아닙니다. 현재 `wavebit.materials.Lorentz.epsilon_term`은 SI 비율과 세기로 같은 비활성 극점 규약을 구현합니다. 잘못된 추적 매개변수는 여전히 NaN을 만듭니다.

미분하는 세기는 값이 0이어도 비활성이 아닐 수 있습니다. 공진에서 떨어진 곳에서
`D = omega_0**2 - omega**2 - 1j*gamma*omega`에 대해 그 미분은 `d(eps_term)/d(strength) = 1/D`입니다.
추적한 세기에서는 순방향 모드 접선(tangent)을 포함해 이 의존성을 유지합니다. 값이 0이라는 것만으로 그
의존성을 끊어서는 안 됩니다. 반대로 구체적인 비활성 세기에서는 유한한 주파수, 공진 주파수, 감쇠에 대한
미분이 계속 0입니다. 활성이거나 세기를 추적하는 진동자의 감쇠 없는 공진은 여전히 특이점이며, 이 규약은
그곳에서 유한한 정규화나 미분을 제공하지 않습니다.

이것은 해석적 재료 응답입니다. 시간 영역의 극점 점화식, 그 안정성 정책, 그리고 계획된 `fd.run`이
허용하는 재료·형상 미분은 바뀌지 않습니다.
:::

## 추적할 수 있는 값

:::{admonition} 이전 인터페이스
:class: earlier-interface

2026-10-15에 `wavebit.compat`와 함께 퇴역한 이전 인터페이스입니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

| 잎 | 위치 | 참고 |
| --- | --- | --- |
| 일반 `fd.Material`의 `eps`, `mu` | 모든 구조, 배경 | 각각 스칼라 하나입니다(`eps=(x, y, z)`가 아니라 `eps=x`). Diffraction 모니터가 있는 장면에서 PlaneWave의 축을 따라 놓인 안정화 PML 층(`PML(stabilize=...)`)을 채우는 재료는 안 됩니다. 추적한 형상이 있거나 Sphere, Cylinder, Prism 또는 회전한 Box 옆에서는: PML이 배경에 맞춰져 있거나 장면에 PlaneWave 또는 TFSF 소스나 `FarField(propagation='yee')`가 있으면 배경은 안 됩니다. 맞닿은 접합부의 매질 -- 기둥, 그 면, 배경 -- 은 추적하든 하지 않든 `mu`를 지정하지 않습니다(전기적 접합부만 측정했습니다). |
| `fd.Conductor`의 `sigma`, `fd.Dispersive`의 극점 매개변수 | 구조 | 추적한 형상이 있거나 Sphere, Cylinder, Prism 또는 회전한 Box 옆에서 `fd.run`으로 실행하면 증명서가 없으므로 거부합니다. |
| `radius`, `center` | `fd.Sphere(range=...)` | |
| `radius`, `height`, `center` | `fd.Cylinder(range=...)` | `axis`는 정적으로 둡니다. |
| `size`, `center` | `fd.Box(range=...)` | `angle`과 `axis`는 정적으로 둡니다. |
| `vertices`, `height`, `center` | `fd.Prism(range=...)` | 꼭짓점 개수, `angle`, `axis`는 정적으로 둡니다. |
| 진폭, 위상, 지연, `Band(wavelength=, df=)` | 소스 | 추적한 대역은 계획의 파장당 셀 수 검사를 건너뜁니다. |
| Bloch 위상 또는 `k` | `fd.Bloch`, `fd.Axis1D` | |

범위를 지정한 여러 물체를 한 장면에서 추적할 수 있습니다. 단, 각 물체의 예약 범위를 보간 셀 하나만큼
넓혔을 때 서로 셀을 공유하지 않아야 합니다. 범위가 없는 구조는 추적하는 모든 물체의 넓힌 범위에서 셀
하나만큼 떨어져 있어야 하며, 추적하는 기둥이 서 있는 기판만은 예외입니다. 범위를 지정한 물체의 재료는
등방성이고 손실이 없어야 합니다(`fd.Material`, `eps`와 `mu` 각각 숫자 하나). 짝이 되는 재료 -- 박막,
블록, 구체적인 기둥이 서 있는 면 -- 도 위 규칙 안에서 추적할 수 있습니다.

**구체적인** 형상을 가진 스무딩 구조도 같은 규칙으로 `eps`나 `mu`를 추적할 수 있습니다. 구체적인 값으로
계획하고 `plan=`을 넘기세요. Sphere, Cylinder, Prism 또는 회전한 Box -- 스무딩이 필드 성분을 결합하는
물체 -- 에서는 계획의 안정성 증명서에 숫자가 필요하므로 `plan=None`을 거부합니다. 회전하지 않은 Box와
셀 스냅 `Slab`, `Region`은 늘 그래 왔듯이 추적한 값을 계획합니다.
:::

## When the result is NaN

:::{admonition} 이전 인터페이스
:class: earlier-interface

2026-10-15에 `wavebit.compat`와 함께 퇴역한 이전 인터페이스입니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

추적한 값은 계획이었다면 거부했을 값에 도달할 수 있습니다. 그때 실행은 예외를 일으키지 않고 --
`jax.grad` 아래에서는 예외를 일으킬 대상이 없습니다 -- `nan` 필드, 결과, 그래디언트를 반환하며,
`result.diagnostics['stable']`은 `False`가 됩니다. 실행 시점 검사는 다음과 같습니다.

| 검사 | 실패 조건 | 구체적인 값 경로의 거부 |
| --- | --- | --- |
| 범위 | 추적한 물체가 `range=`를 벗어나거나 폭이 셀 네 개 미만이 됨 | "cells across" |
| 면 위의 밑면 | 추적한 기둥의 밑면이 기판 면을 벗어남(반올림 오차 이상) | 맞닿지 않음 |
| 양수성 | 추적한 `eps`나 `mu`가 양수가 아님(또는 `nan`) | `Material(eps=...)` |
| 시간 간격 | 가장 작은 굴절률 때문에 `dt`가 안정성 한계를 넘음 | "stability bound" |
| 들어 올린 연산자 | 스무딩 연산자의 별(star) 하나가 그 들어 올림에서 양의 정부호가 아님 | 구성 관계 증명서 |
| 분해능 | 추적한 매질이 대역의 최단 파장이나 어떤 `Diffraction` 파장에서 파장당 셀 8개 아래로 떨어짐. 보고 파장에서는 표피 깊이당 셀이 하나 미만인 금속도 거부 | "cells per wavelength" 또는 "skin depth" |
| 스치는 차수 | PlaneWave의 축을 따라 PML 층을 채우는 추적 매질 때문에 `Diffraction` 모니터가 요청한 차수가 층이 받아들이지 않는 출사각으로 빠져나감 | "Grazing order refused" |
| 추적한 `k`에서의 스침 | 추적한 Bloch `k`나 위상, 또는 추적한 Axis1D `k` 때문에 `Diffraction` 차수나 PlaneWave 자체의 입사가 순수 늘림(pure-stretch) PML을 층이 받아들이지 않는 출사각으로 빠져나감 | "Grazing order refused" |

계획을 거쳐 낮춘(lower) 추적 횡방향 `k`가 실행의 `k`입니다. PlaneWave의 열은 면의 살아 있는 값으로
구동되므로, 계획 하나로 `k`를 스윕하면 입사도 스윕됩니다. 이때 스침 규칙은 그 살아 있는 `k`에서 셉니다.
안정화 PML 면 옆에서는 그 규칙이 호스트에서 면의 정확한 반사도 읽으므로, 추적한 `k`를 이름을 밝혀
거부합니다.

스침 검사는 판정 대상 PML 층 안의 추적한 일반 재료마다 실행되며, 셀 스냅 Slab과 스무딩 물체가 없는
장면도 포함합니다. 고정된 PML 기준 매질은 늘림 분포를 고정할 뿐, 나가는 차수를 판정하는 데 쓰는 물리적
굴절률을 고정하지 않습니다. 분해능 검사는 추적한 모든 재료에 대해 실행되며, 허용된 구체적 형상 위의 도체·극점
매개변수도 포함합니다. 이 검사는 즉시 실행(eager) 검사와 같은 복소 굴절률을 읽습니다. `abs(eps(w) * mu)`가
가장 큰 축을 고른 다음 수동(passive) 제곱근을 취합니다. 분산은 `eps_inf`를 굴절률로 쓰지 않고 해당
파장에서 계산합니다. 전도도는 기존 즉시 실행 전도성 박막 범위에서만 기여하며, 이것이 미분할 수 있는 도체 형상의
범위를 넓히지는 않습니다. 금속 같은 매질(`Re(n) < abs(Im(n))`)은 셀 8개 규칙 대신 보고 파장에서의 표피
깊이를 씁니다. 소스가 없으면 그 파장은 `Resolution(wavelength=)`이나 처음 계획된 PML 파장에서 옵니다. 둘 다
없으면 즉시 실행 검사와 실행 시점 검사 모두 판정할 파장이 없습니다. 보고 파장 검사는 소스가 없는 균질 진공에
대한 즉시 실행 검사의 면제도 그대로 유지합니다. Diffraction 파장에는 여전히 별도의 셀 8개 검사가 있습니다.

일반 `fd.Material`의 경우, 이 검사는 구체적 형상을 가진 Slab, Region 또는 회전하지 않은 Box 위의 추적한
`eps`도 포함합니다. 이런 경우 계획은 셀을 셀 수 없었다고 경고만 할 수 있습니다. `dx = 0.05`와 위의 대역
(최단 파장 0.642)에서 하한은 `eps * mu`를 `(0.642 / (8 * 0.05))**2 = 2.58`까지 허용합니다. 기둥 굴절률이
1.6을 넘으면 더 촘촘한 계산 격자가 필요합니다. 같은 셀(주기 0.6, 셀 8개짜리 PML)에서 기판의 1차 회절
차수는 `eps`가 `(0.95 / 0.6)**2 = 2.507`을 넘는 순간 파장 0.95에서 전파하기 시작합니다. 이는 수평선 바로
위이며, 어떤 PML도 흡수하지 못합니다. 거기서부터 실행은 `nan`이 되며, 구체적인 장면이 거부되는 것과
같습니다. 최적화기의 한 단계가 `nan`을 반환하면 한 걸음 물러나 변수에 한계를 두세요. 실행 보고
(`result.report.warnings`)가 추적한 값이 따라야 하는 검사의 이름을 알려 줍니다.
:::

## 거부하는 것과 그 이유

:::{admonition} 이전 인터페이스
:class: earlier-interface

2026-10-15에 `wavebit.compat`와 함께 퇴역한 이전 인터페이스입니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

다음은 어떤 실행 시점 검사로도 대신할 수 없으므로 장면을 실행할 때 이름을 밝혀 거부합니다. 안정화 PML
행은 스무딩 물체가 없어도 적용됩니다. 그 정확한 반사 조사(survey)에는 실행 시점 미분 검사가 없습니다.
나머지 행은 추적한 형상이 있거나 Sphere, Cylinder, Prism 또는 회전한 Box 옆에서 적용됩니다
(구체적 형상을 직접 `fd.lower`하면 전도도 행을 건너뛰고, 시드는 `mu`를 어떻게 지정했는지 비교하는 데만 읽습니다).

| 거부 대상 | 이유 | 대처 |
| --- | --- | --- |
| 축별로 준 추적 `eps`나 `mu` | 스무딩 연산자는 등방성 숫자 하나에 맞춰 만들어짐 | 스칼라 하나를 추적 |
| PML이 맞춰진(자체 `background=`가 없는) 추적 배경, 또는 PlaneWave나 TFSF 소스, 격자 `FarField`가 있는 추적 배경 | PML 정합, 평면파 소스의 횡방향 파수 벡터, 스침 조사, 원거리장 사상이 배경으로부터 한 번 만들어짐 | PlaneWave, TFSF 또는 격자 `FarField` 옆에서는 배경을 구체적인 값으로 두고 대신 구조의 재료를 미분. 어떤 PML 설정으로도 풀리지 않음. 정합된 PML만 있는 경우에는 모든 면에 구체적인 기준을 두면 풀림(`background=`; 공개 API에서는 모든 Open 면마다 항목을 둔 `PML(reference_media=...)`) |
| 스침 규칙이 판정하는 안정화 PML 층을 채우는 추적 재료 | 안정화 면에서 규칙은 면의 정확한 반사를 호스트에서만 읽음 | 구체적인 값으로 두거나, 면을 순수 늘림으로 |
| `Symmetry` 평면 위 스무딩 영역 옆의 추적 재료 | 거울 검사가 계획의 대비를 읽는데, 대비가 0일 수 있음 | 영역을 거울에서 셀 하나 떨어뜨림 |
| `orders=None`인 `Diffraction` 평면의 추적 매질 | 추적한 굴절률로는 전파하는 차수를 셀 수 없음 | `orders=`를 지정 |
| 기둥이 서 있는 면이 형상을 추적하는 물체인 경우 | 접합부는 서 있는 물체를 통해서만 미분됨 | 면의 형상을 구체적인 값으로 |
| 추적한 전도도나 극점 매개변수 | 증명서 없음 | 구체적인 값으로 |
| 그 재료 없이 만든 `plan=`에서의 추적 값, 또는 `mu`를 다르게 지정한 경우 | 계획이 다른 매질에 대해 보증됨 | 같은 장면을 구체적인 값으로 계획 |
:::

## 미분은 얼마나 정확한가

:::{admonition} 이전 인터페이스
:class: earlier-interface

2026-10-15에 `wavebit.compat`와 함께 퇴역한 이전 인터페이스입니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

미분은 이산 솔버의 미분 그 자체입니다. 위 셀의 탐침(probe) 버전
(`studies/fdtd/traced_material/fdtd_traced_material_probe.py`: 파장 세 개, T와 R, 중심의 x를 네 번째
잎으로 추가)에서 JAX와 Torch는 `3e-15`까지 일치하고, 둘 다 구체적인 NumPy 솔버의 중심 차분 -- 섭동한
형상마다 따로 계획 -- 과 단계 `3e-6`에서 모양에 대해 `9e-8`, 기둥 유전율에 대해 `4e-9`, 기판 유전율에
대해 `1e-9`까지 일치합니다. 주의할 점이 두 가지 있습니다.

* **이산 솔버는 조각별로 매끄럽습니다.** 몇몇 고립된 형상에서는 voxel의 분류가 바뀌고(원기둥에서 가장
  가까운 면이 벽에서 뚜껑으로 바뀌거나, 뚜껑이 계산 격자 평면 위에 정확히 놓이는 경우), 값이 `T`의
  약 `1e-5`만큼 뜁니다. 그래디언트는 양쪽 모두에서 옳지만, 그 도약에 걸친 유한 차분은 옳지 않습니다.
  원기둥 뚜껑 테두리 바로 바깥에서도 그래디언트가 틀리지 않으면서 가파릅니다. 테두리의 모멘트는 접평면의
  모멘트이고, 그 법선은 테두리까지의 거리 `delta` 안에서 90도 돕니다. 따라서 기울기는 `1/delta`로
  커집니다. 테두리 voxel 하나(`dx = 0.25`)에 대해 `0.33 / delta`로 측정되었고, `delta = 1e-5`에서 3e4,
  몇 ulp 이내에서는 1e15를 넘습니다. JAX와 Torch는 단계 `delta / 100`인 중심 차분과 일치합니다.
  동점 처리 규칙이나 몇 ulp짜리 혼합으로는 이를 없앨 수 없습니다. 둘 다 자기 띠 바깥의 값을 그대로 두고,
  그 값들이 그곳의 기울기를 정하기 때문입니다. 정확한 테두리 모멘트만이 이를 없앨 수 있습니다.
  `tests/fdtd/test_fd_shapes.py::test_the_cap_rims_steep_gradient_is_the_planes_own_derivative`가
  기울기, 동점에서의 유한한 값, 그리고 동점이 뚜껑 법선으로 가는 것을 고정합니다.
* **미분은 수렴에 대한 진술이 아닙니다.** 위 셀은 빠르게 돌리려고 얇은 PML과 짧은 실행을 쓰며, 그 경우
  손실 없는 셀의 `T + R`은 1에서 약 `5e-4` 벗어납니다. 셀 24개짜리 PML, `.05` 대역, 6000단계에서는 1과
  `1.1e-6`까지 맞고, 미분은 기둥 유전율에 대해 `dT`의 `5e-5`, 반지름에 대해 `6e-5`까지 `dT + dR = 0`을
  유지합니다. 기판 유전율에서는 `dT + dR`이 약 `5e-5`(작은 `dT`의 0.2%)입니다. 두꺼운 기판의 광경로가
  바뀌면서 잔차 `T + R - 1` 자체가 `1e-6` 수준에서 출렁이고, 이것이 그 기울기입니다. 최적점을 믿기 전에
  자신의 관측값을 세분화하며 확인하세요.
:::

## 어떤 백엔드를 쓸까

:::{admonition} 이전 인터페이스
:class: earlier-interface

2026-10-15에 `wavebit.compat`와 함께 퇴역한 이전 인터페이스입니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

JAX는 계획마다 실행을 한 번 컴파일하며(탐침 셀에서 CPU 스레드 하나로 약 50 s), 그 뒤로는 반복이
빠릅니다. Torch는 즉시 실행되며(같은 출력 두 개짜리 야코비안에 약 7 s), 평가 횟수가 적을 때 알맞습니다.
NumPy는 미분할 수 없지만, 구체적인 NumPy 실행은 그래디언트를 비교할 기준입니다. 기둥의 접합부 배치는
형상에 속하므로, 섭동한 장면마다 따로 계획하세요.

집중 회귀 테스트는 `python -m pytest tests/fdtd/test_material_resolution_guard.py`입니다.
이 테스트는 소스가 없는 장면에서 일반, 전도성, Debye, Drude 재료의 즉시 실행 검사 문턱을 사이에 두고
양쪽을 확인하며(공진점의 구체적인 비활성 Lorentz 극점 포함), 허용된 낮추기 계수와 그 미분을 구체적인
유한 차분과 비교하고, JAX jit, Torch, 규칙 파장, PML 대체 경로를 확인합니다. 이 보호 장치는 자동 미분에
대해 상수이며, 허용된 계수를 바꾸지 않습니다.

보호 장치 회귀 테스트는 경계값을 포함하는 문턱과 그에 가장 가까운 표현 가능한 이웃 값에서의 즉시 실행
판정도 비교합니다. 보고 파장과 같거나, 더 짧거나, 더 긴 Diffraction 파장을 가진 균질 진공, 이방성
절댓값 굴절률 선택과 첫 성분 동점, 공진점에서 구체적인 비활성 Lorentz 세기 대 전달된(carried) 0 세기,
그리고 공진에서 떨어진 곳의 활성 Lorentz 계수 미분을 다룹니다. 대체 경로 테스트는 (가장 짧은 것이 아닌)
처음 계획된 PML 파장, PML보다 Resolution이 우선하는 것, 둘보다 소스 대역이 우선하는 것, 그리고 기존의
0, 유한 또는 NaN 보호 값과의 곱셈을 유지하는지 확인합니다. 이 검사들은 물리적 문턱을 바꾸지 않습니다.


추적한 재료 값은 생성자의 유한 값 요구를 그대로 따르며, Lorentz 세기가 구체적으로 0일 때도 전도도와
모든 극점 매개변수가 이에 포함됩니다. 양의 무한대는 허용되는 양수 매개변수가 아닙니다. NaN, 음수, 금지된
0과 마찬가지로 재료 보호 장치를 오염시킵니다. 전도도, 감쇠, 진동자 세기는 여전히 0을 허용하고, 유전율,
투자율, 주파수, Debye 이완 시간은 엄격히 양수여야 합니다. 잘라내기나 물리적 상한은 도입하지 않습니다.
유한한 극단 입력은 이 범위 입구를 통과합니다. 이것만으로 극단적인 유한 규모에서의 CFL 한계, 분해능, 점화식
산술이 보증되지는 않으며, 비활성 해석적 응답이 잘못된 매개변수 선언을 면제해 주지도 않습니다.
`tests/fdtd/test_material_finite_guard.py`가 허용된 계수 미분을 포함해 이를 다룹니다.

매우 큰 유한 Drude/Lorentz 감쇠에서는 `gamma / dt`가 넘치기(overflow) 전에 ADE 계수가 대수적으로
다시 스케일한 분모를 씁니다. 분자와 분모에 `dt`를 곱하면 `D = 1/dt + gamma/2`가 되고, 계수는
`(2/dt - omega_0**2*dt)/D`, `(1/dt - gamma/2)/D`, `strength*dt/D`입니다.
구현은 역수가 정규수로 남도록 2의 정확한 거듭제곱으로도 다시 스케일합니다.
Lorentz 광학 응답은 `gamma*omega` 곱이 넘칠 경우 그 곱을 만들기 전에 분모를 감쇠로 나눕니다. Float64
Drude 감수율은 원래 응답이 유한하더라도, 주파수 자동 미분이 쓰는 역분모 곱이 비정규수(subnormal) 범위에
들어가기 전에 더 일찍 다시 스케일합니다. 보통 크기의 입력은 원래 산술을 유지합니다.
JAX 최적화 장벽(optimization barrier)이 XLA 플래그를 바꾸지 않고 이 중간 스케일링을 유지합니다.
어떤 재료 매개변수도 잘라내지 않고, 감쇠에 물리적 상한도 도입하지 않습니다.
이 재스케일링이 유한한 입력의 모든 극단 조합에서 미분을 보증하지는 않습니다.

이 2차 극점들에서, 가장 작은 정규 float64 값보다 작은 0이 아닌 ADE 구동항은 비정규수를 0으로 내리는
지원 JAX CPU 경로에서 보존할 수 없습니다. 그런 구동항이나 유한하지 않은 ADE 계수는 즉시 실행에서
`PoleArithmeticRefused`로 거부하며, 추적한 값이 그 경계를 넘으면 기존 실행 시점 보호 장치를 오염시킵니다.
이 기계 표현 가능성 입구는 NumPy, Torch와 공유하므로, 활성 진동자가 한 백엔드에서만 조용히 사라질 수
없습니다. 이것은 광학 수렴 허용 오차가 아닙니다. 구체적인 0 세기는 여전히 허용되며, 그 해석적 세기 접선도
유지됩니다. 추적한 매개변수가 구체적인 피연산자에서 생기는 호스트 산술 예외를 억제하지는 않습니다. 예를 들어
추적한 Lorentz 감쇠 옆의 표현할 수 없는 구체적 `omega_0**2`는 공개 생애주기에서 여전히
`PoleArithmeticRefused`로 보고됩니다. 살아 있는 매개변수에 의존하는 표현 가능성은 Torch를 포함해 계속
실행 시점 무효화를 씁니다. 계획 단계는 이를 보증하려고 매개변수를 떼어 내지(detach) 않습니다. JAX의 0
세기 검사는 0과 비교하기 전에 두 정수 크기 워드를 접습니다. 이렇게 해서 관찰한 ARM CPU 컴파일 회귀에서
0이 아님 분류가 보존됩니다. 다른 플랫폼과 컴파일러 버전은 각각 검증이 필요합니다.
[산술 회귀 기록](../../validation/fdtd/fd-pole-arithmetic.md)을 참조하세요. 모든 float64 계산과 마찬가지로,
미분 자체가 자료형의 표현 가능 범위를 벗어나면 보증하지 않습니다. 최종 해석적 미분이 표현 가능하더라도 중간
미분 연산이 아래로 넘칠(underflow) 수 있습니다.
다음 정규화 Drude 예제는 미분 목적 함수를 스케일하지 않고도 작은 허수부 응답과 그 주파수 미분을 모두
유지합니다.

```python
import torch
from wavebit.materials._time_domain import Drude  # the solvers' private model

omega = torch.tensor(1., dtype=torch.float64, requires_grad=True)
chi = Drude(omega_p=1e50, gamma=1e308).eps_term(omega).imag
frequency_derivative, = torch.autograd.grad(chi, omega)
print(chi.detach().item(), frequency_derivative.item())  # 1e-208, -1e-208
```

이 예제는 공유 공개 재료 선언의 SI 비율이 아니라, 솔버의 비공개 정규화 단위 모델과 그 비율을 씁니다.
Float64 공개 `wavebit.materials.Drude.epsilon_term`은 SI 주파수로 같은 평가기를 씁니다. 공개 SI 평가와
정규화 FDTD 비율로의 변환은 따로 테스트합니다. 정규화 예제의 정확한 `gamma=1e308`은 사용 가능한 단위
선택으로 유한한 SI 감마를 변환해서는 얻을 수 없습니다. float32 주파수나 극점 매개변수가 들어간 공개 호출은
혼합 자료형 승격을 포함해 이전의 산술과 자료형을 유지합니다. 이 수정은 극단 값 지원을 float32나 혼합
정밀도로 넓히지 않습니다.

`tests/materials/test_drude_frequency_scaling.py`는 스케일하지 않은 허수부 응답과 순방향/역방향 주파수
미분을 독립적인 100자리 Decimal 공식과 비교하며, 보통 입력과 두 산술 전환점의 float64 이웃 값을 포함합니다.
공개 SI 평가, FDTD 단위 변환, 한 파장에서의 재료 평가, 공개 float32 호환성도 확인합니다. Lorentz 응답,
0 세기 처리, ADE 계수는 이 Drude 응답 수정으로 바뀌지 않습니다. 이 검사들이 극단적인 유한 주파수, 극점
매개변수, 미분 목적 함수의 모든 조합을 보증하지는 않습니다. 유한한 응답과 표현 가능한 최종 미분이 표현
가능한 중간 산술을 보장하지 않기 때문입니다. 그런 영역에서는 최적화에 필요한 미분을 독립적인 해석적 식이나
표현 가능한 유한 차분과 비교해 확인하세요.

산술 검사는 계획이 선택한 점화식을 따릅니다. 공간 박막 항은 자체 계수와 CFL 검사를 쓰며, 쓰지 않는
스칼라 극점 점화식을 계산하지 않습니다. 특히 구체적인 비활성 하이브리드 박막 슬롯은 쓰지 않는 제곱이나
역수를 만들지 않으므로 극단적인 유한 공진 주파수나 이완 시간을 가질 수 있습니다. 그 슬롯은 할당된 채로
남고, 같은 박막의 활성 슬롯은 기존 표현 가능성·안정성 검사를 그대로 거칩니다. 이것이 CFL 한계에 대한
일반 단일 극점 Lorentz 공진의 기여를 없애지는 않으며, 전달된 0 세기나 0이 아닌 비정규수를 비활성 슬롯으로
분류하지도 않습니다. 유한성/부호 선언 검사는 모든 매개변수에 계속 적용됩니다.
`tests/fdtd/test_selected_pole_arithmetic.py`는 공개 준비 단계, 활성 계열 옆의 비활성 슬롯, 활성 극단
거부, 그리고 부호 있는 두 0에서의 일반 살아 있는 세기 접선을 다룹니다.

`tests/fdtd/test_large_damping.py`는 가지 이웃 값과 0인 세기·0이 아닌 세기를 독립적인 Decimal 산술과 비교하고,
유한한 광학 감수율을 확인하며, 작은 물리적 `check`/`prepare`/살아 있는 실행 장면을 비교하고, 구동항 표현
가능성 입구의 양쪽과 허용된 값/세기 미분을 시험합니다.

전달된 광학 주파수는 감쇠 문턱에 백엔드 스칼라 피연산자를 씁니다.
비활성 일반 응답은 0이 아닌 감쇠를 유지하므로, 선택되지 않은 Lorentz 공진이 선택된 응답의 주파수 미분을
NaN으로 오염시킬 수 없습니다.
`tests/fdtd/test_damping_response_review.py`는 구체적인 값/전달된 값의 일치와 큰 감쇠 공진에서의 해석적
주파수 접선을 확인합니다.
:::
