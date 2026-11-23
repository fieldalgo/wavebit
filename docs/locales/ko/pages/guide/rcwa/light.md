# RCWA 조명 선언하기

`rcwa.Excitation(wavelength=..., sources=(...))`는 진공 파장과, 면·방향·편광·진폭을 가지는 소스를 서로 분리합니다. 소스의 각도는 모두 라디안 단위입니다. 소스 하나는 `theta/phi`나, 문제 길이 단위의 역수를 단위로 하는 `kx/ky` 중 하나만 쓰며, 둘을 함께 쓰지 않습니다.

```python
from wavebit import rcwa
excitation = rcwa.Excitation(
    wavelength=(.85, .9, .95),
    sources=(rcwa.PlaneWave(side="top", polarization="s", amplitude=1.,
                            theta=.2, phi=0.),))
```

평면파 진폭의 단위는 V/m입니다. 편광은 `"s"`, `"p"`, 또는 전기장 크기가 1이 되도록 정규화되는 복소 `(s,p)` Jones 쌍이며, 쌍의 상대 위상은 그대로 유지됩니다. 나가는 진폭비는 정규화된 이 쌍에 대해 선형입니다. `(a_s, a_p)`에 대한 `amplitude_ratio_s`, `amplitude_ratio_p`, `amplitude_ratio_xyz`는 `"s"`에 대한 값에 `a_s/n`을 곱한 것과 `"p"`에 대한 값에 `a_p/n`을 곱한 것의 합이고(`n = sqrt(|a_s|**2 + |a_p|**2)`), 소스의 `amplitude`는 이 값들을 바꾸지 않습니다. 그 진폭이 차수들의 `reference['incident_field']`입니다. `field_s`, `field_p`, `field_xyz`는 진폭비에 이 진폭을 곱한 값, 곧 V/m 단위의 나가는 필드이므로 진폭에 비례하고, `power`는 진폭의 제곱에 비례합니다. 따라서 두 번 계산하면 한 방향과 한 파장에서 모든 편광의 진폭비를 얻습니다. 결합할 때는 진폭비를, 또는 같은 소스 `amplitude`를 공유하는 계산들의 `field_*`를 결합하고, `R`, `T`, `power`는 결합하지 마세요. 이들은 크기의 제곱이므로 더하면 두 상태 사이의 교차항이 빠집니다. `IncidentOrders`는 결맞는 입사 회절 차수를 지정합니다. `GaussianBeam`은 각 스펙트럼을 기술합니다. `ElectricDipole(position=..., moment=...)`는 C·m 단위의 물리적 전기 쌍극자 모멘트를 지정합니다. 이 소스 계열들은 지원 범위와 정규화 요건이 서로 다르며, 요청한 조합은 `check`가 검증합니다.

각 생성자는 [레퍼런스](../../reference/rcwa/index.md)에 정리되어 있습니다. 아래 절들은 방향, 편광, 입사 면을 자세히 설명합니다.

:::{note}
아래의 두 기능은 아직 현재 API에 없으므로, 해당 대목은 이전
`wavebit.compat.legacy` 인터페이스를 쓰며 그렇다고 밝혀 둡니다. `psi`/`chi` 편광 좌표와,
한 번의 풀이로 얻는 s/p 응답 `s_inc`/`p_inc`입니다. 세 번째 기능이었던 순수 1D `tetm` 선언은 이제
단어 `'s'`와 `'p'`가 맡습니다(아래 참고).
:::

각 소스의 입사 방향은 각도 또는 횡방향 파수 벡터 성분으로 지정합니다.
편광, 비추는 회절 차수, 입사 면은 별도의 설정입니다.
기본값은 [RCWA 레퍼런스](../../reference/rcwa/index.md)를 참고하세요.

## Two ways to specify direction

| 입력 | 의미 | 파장 스윕 중 고정되는 것 |
| --- | --- | --- |
| `theta`, `phi` | 입사광이 진행하는 방향의 극각과 방위각(라디안) | 각도. k는 파장과 입사 재료의 분산에 따라 바뀝니다 |
| `kx`, `ky` | 실수인 카르테시안 횡방향 각파수 | 물리적 kx와 ky. 각도가 바뀔 수 있습니다 |

`kx`와 `ky`의 단위는 길이의 역수이며, `Units(length='nm')`에서는 nm**-1입니다. 이 값은
**k0로 나눈 값이 아니며**, 역격자 차수 번호도 아닙니다. 생략한 성분은 0입니다.
`None`이 아닌 `theta`나 `phi`와 함께 어느 한 성분이라도 주면, 각도가 0이더라도 오류가 발생합니다.
두 형식 모두 주지 않으면 기본값은 방위각 0의 수직 입사입니다.

예를 들어 다음 두 입력은 epsilon=2.25, mu=1(n=1.5)인 무손실 매질에서
같은 파를 지정합니다.

```python
import numpy as np
from wavebit import Material, Units, rcwa

def problem(source):
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(
            period=(700., 700.), top=Material(epsilon=2.25), bottom=Material(epsilon=1.),
            layers=(rcwa.Layer(thickness=100., background=Material(epsilon=4.)),)),
        excitation=rcwa.Excitation(wavelength=500., sources=(source,)))

numerics = rcwa.Numerics(orders=(0, 0))
by_angle = rcwa.run(problem(rcwa.PlaneWave(polarization='s', theta=np.pi/6, phi=0.)),
                    numerics=numerics)
k0 = 2*np.pi/500.
by_k = rcwa.run(problem(rcwa.PlaneWave(polarization='s', kx=0.75*k0, ky=0.)),
                numerics=numerics)
np.testing.assert_allclose(by_angle.T, by_k.T, atol=1e-12)
```

두 형식 모두 실제 `Stack.top`/`Stack.bottom` 재료가 필요합니다. epsilon과 mu가
법선 방향 파수 벡터, 전기장과 자기장의 관계, 입사 전력을 정하기 때문입니다.
k를 직접 주는 방식은 흡수하는 반공간에서도 쓸 수 있습니다. 이때 k는 그 반공간 안의 파를
지정하며, 그 위에 다른 매질이 더 있다고 가정하지 않습니다.

두 성분은 입사 면이 어느 쪽이든, 사각이 아닌 격자(비스듬한 격자)에서도 고정된 물리적 x/y 축을
기준으로 합니다. 솔버는 각 회절 차수에 역격자 오프셋을 더합니다.
`result.reflection.k_parallel`과 `result.transmission.k_parallel`은 그
**차수별 성분을 길이의 역수 단위로**(k0로 나누지 않은 값으로) 마지막
`(kx, ky)` 축에 담아 알려 줍니다.

직접 주는 성분은 유한한 실수 스칼라여야 합니다. 이렇게 하면 표면을 따라 세기가 균일한 단일 파가
정해집니다. 전력이 0인 소멸파 입사는 정규화된 회절 효율을 정의할 수 없으므로 거부합니다.
복소 성분도 거부하며, 흡수 매질에서도 마찬가지입니다. 흡수 매질에서는 실수 `theta`가 입력이고,
실수 각도는 `kx`를 실수로 유지합니다. 복소 횡방향 파수 벡터는
복소 `theta`/`phi`를 통해서만 줄 수 있습니다. 이 경우 진폭비와 필드는 반환되지만, `R`, `T`와 포트 전력은 NaN이고
`result.reflection.metadata['power_defined']`는 false입니다. 거기서는 균일 세기 전력
정규화가 성립하지 않기 때문입니다([실수 각도 입사](../../reference/common/conventions.md#real-angle-incidence)
참고). 이전 인터페이스는 대신 형식적인 `R`과 `T`를 반환했습니다. 무손실 반공간에서 오는
소멸파 입사 `theta = pi/2 + 1j*alpha`는 `kx`를 실수로 유지하지만, 그 입사파는 전력을 나르지 않습니다. 그 `R`,
`T`, `incident_power`, `power_entering`, `absorption`은 NaN이고
`metadata['efficiency_defined']`는 false입니다. 반대편까지 터널링한 빛은 실제 전력을 나르므로
`power_defined`는 true로 남고 W 단위의 포트 전력이 반환됩니다(플래그는
[실수 각도 입사](../../reference/common/conventions.md#real-angle-incidence)
참고). 진폭비와 필드도 반환됩니다. 이전 인터페이스의 `R`, `T`, `power_entering`도
이 경우 NaN입니다. 직접 주는 성분에는
각도가 없으므로, 음굴절률 매질을 포함한 어떤 매질에서도 주어진 그대로 받아들입니다.
`kx=ky=0`이면 s/p 기저는 방위각 0을 씁니다(s는 y 방향, p는 xz 평면). 필드의 방향은
`polarization`으로 정하세요.
이전 인터페이스의 순수 1D `tetm` 입력에 해당하는 경우에는 `ky`를 생략하고(0이 됩니다),
conical 입사에는 `polarization`을 쓰세요.

`Job` 직렬화와 JAX/PyTorch 미분도 같은 입력을 씁니다. 소스를 각도에서 직접 k로 바꾸려면
같은 교체에서 `theta`와 `phi`를 함께 비우세요:
`source.replace(theta=None, phi=None, kx=..., ky=...)`. 되돌릴 때는 두 k 성분을 모두
비웁니다. 새 k는 새로 계산할 새 문제이며, 앞선 결과는 자신의 필드를 그대로 유지합니다.

## What `theta` and `phi` mean

`theta`와 `phi`는 **입사광이 들어오는 방향**입니다. 고정된 x/y 축에서, 빛을 받는 반공간의
안쪽 법선으로부터 잰 극각과 `+x`에서 `+y` 쪽으로 잰 방위각입니다. 이 각도는 선택한 반공간인
`Stack.top` 또는 `Stack.bottom` 자체에서 잽니다. 별도의 공기층을 가정하지 않습니다.
실수 각도는 표면을 따라 균일한 입사 세기를 지정합니다.

이로써 고정되는 것은 면내 파수 벡터입니다.

```text
k_par / k0 = s * N * sin(theta) * (cos(phi), sin(phi))
```

`N`은 빛을 받는 매질의 위상 굴절률(무손실이면 `|n|`)이고, `s`는 그 굴절률
`n = sqrt(eps)*sqrt(mu)`의 부호입니다. 다음 세 경우가 모든 매질을 포괄합니다.

* **무손실 양굴절률 매질**(`s = +1`, `N = n`): 익숙한
  `kx = k0*n*sin(theta)*cos(phi)`, `ky = k0*n*sin(theta)*sin(phi)`입니다. 위상과 전력이 함께
  진행하므로 `theta`는 두 방향을 모두 뜻합니다.
* **음굴절률 매질**(`s = -1`: `Re(eps) < 0`이고 `Re(mu) < 0`인 이중 음성 매질, 또는
  수동 자성 금속처럼 위상 속도가 음인 다른 매질). 위상이 전력과 반대로 진행하므로, 같은 빛이라도
  파수 벡터는 반대쪽을 향합니다:
  `kx = -k0*|n|*sin(theta)*cos(phi)`. 무손실이면 이는 부호를 가진 `n < 0`에 대해 여전히
  `kx = k0*n*sin(theta)*cos(phi)`이며, 음의 굴절률에 대한 스넬 법칙입니다.
  `n = -sqrt(2)`에서 20도로 들어온 빛은 유리 안으로 -18.81도로 들어가며, 들어온 쪽과 같은
  법선 쪽에 놓입니다. 이것이 음굴절입니다.
* 부호와 상관없는 **흡수 매질**: `N`은 복소수 `eps*mu` 전체로부터 정해지고, 법선 방향
  파수 벡터는 복소수이며, 입사파는 구조로 가는 동안 약해집니다. 이때 `theta`와
  `phi`는 `s * Re(k)`의 각도이고, 포인팅 벡터는 손실과 편광으로 정해지는 만큼 이 각도에서
  벗어납니다. `eps = 2+0.5i`에서 45.8도로 입사한 p광은 1.7도 벗어나고, 금속 안의 p광은
  법선의 반대쪽으로 벗어납니다(표는
  [음굴절률 매질](../../theory/rcwa/negative-index-media.md) §7에 있습니다). 흡수 매질에서의 실수 각도 입사에는
  `abs(theta) < pi/2`가 필요합니다.

어떤 매질이든 실수 `theta`는 `[-pi/2, pi/2]` 안에 있어야 합니다. `check`와 `run`은 어느 쪽
끝이든 넘어선 값을(`float32(pi/2)`를 넘는 경우이므로 단정밀도 스치듯 입사 각도는 통과합니다)
`source.direction`으로 거부합니다. 법선의 반대쪽에서 구조를 비추려면 `-theta`나
`phi + pi`를 쓰세요. 복소 `theta`에는 이런 제한이 없습니다.

```python
import numpy as np
from wavebit import Material, Units, rcwa

def lit_from(top):
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(
            period=(1200., 1200.), top=top, bottom=Material(epsilon=2.25),
            layers=(rcwa.Layer(thickness=100., background=Material(epsilon=2.25)),)),
        excitation=rcwa.Excitation(wavelength=1000., sources=(
            rcwa.PlaneWave(theta=np.deg2rad(30.), phi=0.),)))

numerics = rcwa.Numerics(orders=(0, 0))
result = rcwa.run(lit_from(Material(epsilon=4. + 3j)), numerics=numerics)
kx, ky = result.reflection.k_parallel[0, 0, 0]   # wavelength, order_v, order_u
# With lengths in nm: kx ≈ 0.00677139 nm^-1, ky = 0 nm^-1.
```

여기서 30도는 **흡수하는 입사 반공간 안에서의** 각도입니다. 음굴절률 매질에서 비추어도
같은 스크립트가 그대로 실행되며, `kx`의 부호만 달라집니다.

```python
nim = Material(epsilon=-3 + 0.1j, mu=-0.5 + 0.05j)      # n_top = -1.2254+0.0816j
result = rcwa.run(lit_from(nim), numerics=numerics)
result.reflection.k_parallel[0, 0, 0, 0]  # negative: the light travels along +x, its phase along -x
```

정확히 `theta = 0`일 때 s/p 좌표계는 `theta -> 0+`에서 `k_par`가 다가가는 방향을 택합니다.
곧 `phi`, 음굴절률 쪽에서는 `phi + pi`이므로, 어느 쪽에서든 입사 필드는 수직 입사를 지나며
연속입니다. [유도와 복소 각도
규약](../../reference/common/conventions.md#real-angle-incidence)은 입력의 정확한 의미를,
[음굴절률 매질](../../theory/rcwa/negative-index-media.md)은 이중 음성 매질에 관한 모든 것을 설명합니다.

전력은 유한 적층과의 경계면에서의 입사파를 기준으로 합니다.
`result.outputs['power_entering']`은 입사파와 반사파의 간섭을 포함합니다.
`result.outputs['power_entering'] - result.T.sum(axis=(1, 2))`와 같은 `result.absorption`은
모든 유한 층에서의 흡수이며, 두 반공간에서의 흡수는 포함하지 않습니다. W 단위의 포트
전력은 파마다 따로 매긴 값이어서 그 간섭을 빼므로, 입사 와트에서 반사 와트와 투과 와트를 뺀
수지는 간섭을 놓칩니다. 소멸파 입사에서는 이 간섭이 반대편에 닿는 모든 것을 나릅니다.

## The polarization comes in three kinds

현재 `PlaneWave`는 `polarization='s'`, `'p'` 또는 `(a_s, a_p)`를 받습니다. 쌍은 아래의 세 번째
종류이고, 이제 단어가 첫 번째 종류의 일을 합니다. 단어는 편광이 고정되어 있음을 선언합니다(아래
"The words are a declaration" 절 참고). 두 번째 종류인 좌표는 현재 API에 아직 없는
이전 인터페이스의 형식입니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

처음 두 행은 이전 인터페이스의 선언입니다(2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다). 현재 `PlaneWave`는 세 번째만 받습니다.

| | 쓰는 법 | 무엇인가 |
| --- | --- | --- |
| **선언** | `tetm='te'` / `'tm'` | 이 문제에는 편광의 자유도도, 방위각의 자유도도 없다는 뜻입니다. 한 행짜리 래스터 구조에만 쓸 수 있으며(형상 구조는 y 방향으로 불변이더라도 거부합니다. 그 경우에는 `phi=0`에서 `polarization=(1, 0)` / `(0, 1)`을 쓰세요), 이 선언이 순수 TE/TM 정식화를 가능하게 합니다. 이 정식화는 conical 정식화보다 2배에서 4배 빠르며, conical 정식화는 `phi=0`에서의 `polarization=(1, 0)` / `(0, 1)`과 같은 진폭, 필드, 효율을 1e-13까지 일치하게 줍니다(2026-09-27부터. 그 전에는 입사 차수 기준으로 `kx = 0` 반대편에 있는 차수들의 진폭 부호가 달랐습니다) |
| **좌표** | `psi`, `chi` | 상태를 나타내는 두 각도입니다. `psi`는 타원의 기울기, `chi`는 타원율입니다. `chi=0`은 선편광이고(정확히 `psi` 표기와 같습니다), `chi=+/-pi/4`는 원편광입니다. 스윕과 최적화에 쓰는 형식입니다 |
| **진폭** | `polarization=(a_s, a_p)` | 결과가 돌아오는 것과 같은 (s, p) 기저의 Jones 쌍 그 자체입니다. 전체 위상을 담는 유일한 형식이며, 크기는 정규화로 지워지므로 `(3, 4j)`와 `(0.6, 0.8j)`는 같은 입사입니다 |
:::

```python
rcwa.PlaneWave(polarization=(1, -1j))           # the pair, straight from another code
```

:::{admonition} 이전 인터페이스
:class: earlier-interface

2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 이 코드는 과거 체크아웃에서만 실행됩니다.

```python
# Earlier interface: the current API has no psi/chi declaration yet, and tetm's pure path
# is taken by polarization='s' / 'p'.
import numpy as np
from wavebit.compat import legacy as wavebit

wavebit.rcwa(tetm='te', ...)                    # fast 1D path; psi reads back as None
wavebit.rcwa(psi=0.4, ...)                      # linear, at an angle
wavebit.rcwa(psi=0.4, chi=np.pi / 4, ...)       # circular
```
:::

### The words are a declaration, the pair is a value

`polarization='s'` 또는 `'p'`는 편광이 고정되어 있다는 뜻입니다. 누구도 스윕하거나 미분하는
매개변수가 아닙니다. 순수 1D TE/TM 정식화에 필요한 것이 바로 이것이므로, 2026-11-03부터 단어는
같은 빛에 대해 같은 답을 주는 곳이면 어디서든 이 정식화를 택합니다. 모든 층이 한 행짜리 래스터(모양이 `(1, Nx)`인
`Layer(epsilon=...)`)이고, 셀이 축에 정렬되어 있으며, `phi`와 `ky`가 없거나 아무것도 추적하거나
미분하지 않는 0인 경우입니다. `'s'`는 TE, `'p'`는 TM입니다. 그 날짜 전에는 모든 한 행짜리 래스터가
conical 정식화로 풀렸습니다.

그 밖의 모든 경우에 단어는 자신이 가리키는 쌍 `(1, 0)` 또는 `(0, 1)`과 정확히 똑같이 풀리며,
거부되는 것은 없습니다.

- 홈이 y 방향으로 나 있지 않은 기울어진 셀은 2D 경로로 풉니다.
- 0이 아닌 `phi`나 `ky`, 또는 추적되거나 미분되는 값(`jax.jit`, `jax.grad`,
  `vmap`, `requires_grad` 텐서)은 conical 경로로 풉니다. 방위각과 그 미분을 담는 경로입니다.
- `Numerics(formulation='first_order')`는 자신이 작성된 2D 배치로 풉니다.
- `Outputs(scattering=True)`는 conical로 남습니다. 요청한 행렬이 두 편광의 포트를 모두
  유지하는데(`2N x 2N`, 채널 `'sp'`), 순수 경로의 행렬은 한 편광의 포트만 가지기 때문입니다.

Jones 쌍은 값이므로, `(1, 0)`을 포함해 항상 conical 경로를 유지합니다. 스윕하거나
미분할 수 있기 때문입니다. 두 정식화는 반올림 오차 수준에서 일치하며,
`tests/api/test_unified_rcwa_pure_1d.py`의 모든 경우에서 진폭, 효율, 전력, 필드, 경계 필드가
상대 2e-13 이하로 같고, JAX와 PyTorch 그래디언트도 마찬가지입니다. 순수 정식화가
더 빠르고 더 작습니다. 아래 "비용"을 참고하세요. `check(...).resolved['solve_path']`는 계산이 택할
경로를, `result.diagnostics['solve_path']`는 각 파장이 실제로 택한 경로를 알려 줍니다.
값은 `'1d_te'`, `'1d_tm'`, `'1d_conical'` 또는 `'2d'`입니다. 앞의 것은 `check`나 `prepare`에
주어진 구체적인 문제에서 읽으므로, 예를 들어 `jax.jit` 안에서 `phi`를 추적하는 준비된 계산은
계획으로 `'1d_te'`를 보고하고, 실제로 택한 경로인 `'1d_conical'`을 기록합니다.

#### 비용

2026-11-03에 공개 `rcwa.run`으로 측정했습니다. 10코어 노트북의 NumPy에서,
1000 nm 셀 안에 유전율 12.25인 560 nm 막대를 공기 중에 두고, 두께 200 nm, 파장 1363 nm,
수직 입사, `fourier_coefficients='integrated'`인 50픽셀 래스터로 `'p'`와
`(0, 1)`을 비교했습니다. 시간은 각각 7회, 3회, 1회 실행 중 최솟값입니다. 메모리는 계산 중
프로세스가 기준선 대비 차지한 최대 사용량입니다.

| `orders` | `'p'` (순수 TM) | `(0, 1)` (conical) | 시간 비 | 최대 메모리, 순수 / conical | `sum(R)` 차이 |
| --- | --- | --- | --- | --- | --- |
| `(50, 0)` | 0.017 s | 0.045 s | 2.7x | 1.8 / 26 MB | 5e-13 |
| `(200, 0)` | 0.29 s | 1.22 s | 4.1x | 31 / 251 MB | 8e-12 |
| `(800, 0)` | 10.8 s | 57.8 s | 5.3x | 0.39 / 5.9 GB | 4e-10 |

conical 배치는 한 변이 순수 배치의 두 배인 행렬을 풀므로, 시간과 메모리의 격차는 절단이
커질수록 벌어지며, 두 답 사이의 마지막 자릿수 차이도 함께 커집니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

`tetm`, `psi`/`chi`, `polarization`은 같은 것을 세 가지로 표기한 것이 아니라 서로 다른 세 *종류*입니다.
`tetm`을 다른 둘 중 하나와 함께 쓰면 오류가 나고, `psi`를 `polarization`과 함께 써도 마찬가지입니다.
무엇을 쓸지는 대개 하려는 일로 정해집니다. `(psi, chi)`는 중복이 없는 두 실수이므로
편광 상태에 대한 그래디언트는 평범한 2D 문제가 됩니다. 반면 쌍은 네 실수 중 두 개를
어떤 전력도 바꿀 수 없는 방향에 씁니다.
:::

:::{admonition} 이전 인터페이스
:class: earlier-interface

**이전 인터페이스에서는 편광 스윕에 풀이가 한 번이면 됩니다.** 그 인터페이스에서
`answers='amplitude'` 이상으로 한 모든 풀이는 같은 인수분해에서 얻은 순수 s와 순수 p에 대한
응답인 `result.s_inc`와 `result.p_inc`도 반환합니다. 어떤 쌍 `(a_s, a_p)`에 대해서든 진폭 `r_s`, `r_p`, `t_s`, `t_p`, `r_xyz`,
`t_xyz`는 `s_inc` 배열에 `a_s/n`을 곱한 것과 `p_inc` 배열에 `a_p/n`을 곱한 것의 합과 같습니다.
여기서 `n = sqrt(|a_s|**2 + |a_p|**2)`이고, 쌍의 전체 위상은 유지됩니다. 이 관계는 두
정식화 모두에서, ETM과 SMM에서, 어느 쪽 면에서든, `excitation=`에서도 성립합니다.
결합할 때는 `R`과 `T`가 아니라 진폭을 결합하세요. 효율은 크기의 제곱이므로 두 변형의
효율을 더하면 교차항이 빠집니다. `tetm=`에서와 `answers='efficiency'`에서는 `s_inc`와
`p_inc`가 `None`입니다. 현재 `Result`는 아직 이 둘을 반환하지 않습니다. `'s'`와 `'p'`를 한 번씩
계산하고 그 `amplitude_ratio_s`, `amplitude_ratio_p`, `amplitude_ratio_xyz`를 같은 방식으로 결합하세요. 두 계산은
`s_inc`와 `p_inc`를 5e-15까지 재현합니다.
:::

`IncidentOrders`에 지정한 모든 차수는 입사 매질에서 전파해야 합니다. 라이트 라인(light line)을
넘어선 차수는 구조를 구동하면서도, 모든 효율의 분모인 입사 플럭스에는 아무것도 보태지 않으므로
수동 적층에서 `R + T`가 1을 넘게 됩니다. 그런 차수를 요청하면 거부합니다. 이 때문에 결합할 수 있는
빔의 수도 대략 `2 * period / wavelength`로 제한됩니다.

`IncidentOrders`의 가중치는 *그 차수 자신의* `s_hat` 또는 `p_hat`에 곱해집니다.
`phi = 0`인 1D 격자에서 TE의 경우 이는 `sign(kx) * y_hat`이므로, `kx < 0`인 입사 차수는
`-y` 방향으로 편광되어 들어옵니다. [`../conventions.md`](../../reference/common/conventions.md)를 참고하세요. 가중치는
빛을 받는 면의 `x = y = 0`, 곧 셀 중심에서의 그 빔의 진폭입니다. 이전
인터페이스의 `excitation=`이 받는 것과 같은 가중치이며, 같은 빛입니다.

규범적 서술은 [`../conventions.md`](../../reference/common/conventions.md)를,
입사 차수는 [`../multimode-incidence.md`](multimode-incidence.md)를 참고하세요. 튜토리얼
[`06`](https://github.com/fieldalgo/wavebit/blob/main/tutorials/rcwa/06-polarization-pair.ipynb)과
[`07`](https://github.com/fieldalgo/wavebit/blob/main/tutorials/rcwa/07-multimode-incidence.ipynb)은
이전 인터페이스를 씁니다.

## 빛이 들어오는 면

`PlaneWave(side='bottom')`은 기판을 통해 비춥니다. **장면의 이름표는 움직이지 않습니다**.
`Stack.top`은 여전히 상부 매질이고, `Stack.bottom`은 여전히 기판이며, `Stack.layers[0]`은
여전히 `bottom`에 맞닿은 층입니다. 달라지는 것은 결과가 입사를 기준으로 한다는 점(`result.R`은
기판 쪽으로 되돌아온 빛입니다)과 `theta` / `phi`를 입사 매질에서 잰다는 점입니다.
따라서 다른 솔버와 비교할 때는 수치가 맞아떨어지기 전에 각도를 그 매질 기준으로 변환해야 합니다.

`side='top'`은 `-z` 방향으로, `'bottom'`은 `+z` 방향으로 진행합니다.
`theta`는 선택한 면의 안쪽 법선에서 잰 각도입니다. 층 순서는 항상
아래에서 위로이며, `phi`는 어느 면에서든 같은 고정 x/y 축을 씁니다.

## 양쪽 면에서, 또는 서로 다른 방향으로 들어오는 빔

소스 튜플에 면마다 `PlaneWave`를 하나씩 넣으세요: `rcwa.PlaneWave(side='top', ...)`와
`rcwa.PlaneWave(side='bottom', ...)`. 각 PlaneWave는 자신의 방향, 편광, 복소 진폭을
따로 지정할 수 있습니다. 계산은 전력을 평가하기 전에 복소 경계 필드를 결합하고, 각 포트에서
들어오고 나가는 전력을 `result.outputs['top']`과 `result.outputs['bottom']`으로 반환합니다.
이런 계산에는 `Outputs(diffraction=False)`가 필요합니다.
예제, 계산 흐름도, 위상 기준, 서로 다른 Bloch 섹터에 필요한 평균 영역은
[결맞는 조명](coherent-illumination.md)을 참고하세요.
