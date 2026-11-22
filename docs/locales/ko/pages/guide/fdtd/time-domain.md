# 시간 영역 계산

:::{note}
네 가지 예제는 최상위 `wavebit.fdtd` 장면 API를 사용합니다. 처음 두 예제와,
[공통 장면 어댑터](#common-scene-adapter) 아래의 재료 미분 예제, 그리고 [소스 배치](#source-position-independent-of-padding) 예제입니다.
나머지는 과거의 주기 팩토리 `wavebit.compat.legacy.fdtd` 또는 그 native 격자인
`fdtd_grid`용으로 작성되었으며, 이들은 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 그 코드는 과거 체크아웃에서만 실행되며,
각 예제는 *이전 인터페이스* 블록 안에 있습니다. 그 기능은 새 이름으로 현재도 제공됩니다.
[`wavebit.fdtd.periodic`](periodic-problems.md)은 래스터 층과 형상 타입 층을 받고, 시간에 따른 전체 필드를
기록하며(`Snapshots`, 점 탐침은 `Result.sample`), 비균일(`algorithm='reference'`, `grading=`) 메시와
층에 맞춘 메시를 만들고, 각도·형상·두께·반공간 입력을 미분하며, Jones 쌍을 받고,
펄스·추출·차단·`DecayCheck` 제어를 제공합니다. 모드 피팅은 `wavebit.observations.fit_modes`이고,
셀별 격자는 [`wavebit.fdtd.native`](native-grids.md)입니다. 필드 그리기와 애니메이션에는 현재
공개 대응 기능이 없습니다. 과거 블록과 함께 인용한 수치는 그 인터페이스에서 측정했으며 다시
실행하지 않았습니다. 대부분의 과거 블록은 첫 주석에 현재 표기를 적어 둡니다.
:::

`wavebit.fdtd`는 두 번째 솔버입니다. 평면파가 비추는 주기 적층도 다른 문제와 똑같은 `fdtd.Problem`입니다.
x와 y 면은 `Periodic()`(비스듬한 파동이라면 `Bloch.from_source`가 작성하는 `Bloch` 면),
z 면은 `Open()`이고 그 바깥에 PML을 덧붙이며, 층은 구조물로, 소스는 `PlaneWave` 하나로,
양쪽에는 `DiffractionMonitor`를 하나씩 둡니다. 각 모니터는 RCWA가 반환하는 것과 같은
차수별 복소 s/p 진폭비와 효율을 반환하며, 평면파 자체의 측정된 입사 필드로 정규화합니다.
각 요소는 [FDTD 솔버가 문제를 모델링하는 방식](model.md)에서 설명합니다.

과거의 주기 팩토리 `wavebit.fdtd(stack=..., incidence=...)`는 과거 RCWA 팩토리와 같은 `Stack`과
`Incidence`를 읽고 같은 `Result` 컨테이너를 반환했습니다. 이 팩토리는 2026-10-15에 `wavebit.compat`이
퇴역할 때까지 `wavebit.compat.legacy.fdtd(...)`로 유지되었습니다. `backend` 인수가 없으면
시간 기록과 비균일 메시를 포함한 레거시 NumPy 솔버를 실행했고, `backend="numpy"`, `"jax"` 또는
`"torch"`를 명시하면 공유 Yee 엔진을 선택하고 재료 자동 미분을 지원했습니다.
지원하는 입력은 [공통 장면 어댑터](#common-scene-adapter)를, 같은 장면에서 두 경로를 비교한 결과는
[경로 일치](../../validation/fdtd/fdtd-route-parity.md)를 보세요.

이 페이지는 주기 적층을 다룹니다. 과거의 `wavebit.compat.legacy.fd`와 그
명시적 `fd.Scene` 도메인, 소스, 경계, 이름 붙은 모니터는
[일반 FDTD 이전 페이지](../../migration/general-fdtd-v1.md)를 보세요. 장면 타입과 규약은
서로 별개입니다.

둘 다 **연구용 알파**입니다. 기하, 입사, 재료의 새로운 조합을 사용하기 전에
[지원 범위와 검증 근거](fdtd-alpha.md)를 읽으세요.

## Shared periodic conventions

현재 모듈에서 두 솔버는 모두 길이를 문제의 `Units`로, 각도를 라디안으로 받으며
+z가 위쪽입니다. FDTD에서 층은 셀 스냅된 `SnappedSlab(z=(lo, hi))`이거나 스무딩된
`Box`입니다. 반공간은 `Open()` 면까지 닿아 그 면의 PML 안으로 이어지는 `SnappedSlab`이며,
장면의 `background`는 어떤 구조물도 차지하지 않은 모든 셀을 채웁니다. `direction='-z'`(기본값)인
`PlaneWave`는 적층을 위에서, `'+z'`는 아래에서 비추며, `polarization`은 `'s'` 또는 `'p'`입니다.
`DiffractionMonitor`의 `amplitude_ratio_s`와 `amplitude_ratio_p`는 각 출사 차수 고유의 s/p 좌표계를
사용하고 셀 중심을 측면 위상 원점으로 삼습니다. `efficiency`는 전력 비율이며, `reference_plane`과 `incident_plane`은
위상이 기준으로 삼는 면을 지정합니다(기본값은 모니터 자신의 면). 진폭비에는 단위가
없습니다. 진폭비를 나누는 측정 입사 스펙트럼 `reference['incident_spectrum']`
(V*s/m)도 같은 셀 중심을 기준으로 하므로, 펄스의 사용 가능 대역
(`metadata['source_usable']`) 안에서는 `raw=True` 모니터의 `spectrum_s`와 `spectrum_p`가
진폭비에 이 스펙트럼을 곱한 값과 반올림 오차 범위에서 같습니다. 그 대역 밖에서는 진폭비가 NaN이고 원시 스펙트럼은
유한한 값으로 남습니다. 펄스에는 V/m 단위의 단일 입사 필드가 없으므로 여기서 차수의 `field_s`, `field_p`,
`field_xyz`는 None입니다. 필드는 E가 V/m, H가 A/m 단위입니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

과거 팩토리들은 서로 다릅니다. `wavebit.compat.legacy.rcwa(...)`는 솔버를 반환하고 그 솔버의
`.solve()`가 결과를 반환하지만, `wavebit.compat.legacy.fdtd(...)`는 결과를 직접 반환합니다.
일반 `fd` 장면 API와 native 격자 `fdtd_grid` API에는 각자의 입력 규약이 있습니다.
과거의 두 주기 경로에서는 다음과 같습니다.

- 층은 **아래에서 위로** 나열합니다. 아래 경계면은 `z=0`, 위는
  `z=sum(thickness)`이며 +z는 위쪽을 가리킵니다. `top`과 `bot`은 항상
  물리적인 위쪽과 아래쪽 반공간을 가리킵니다. 두 주기 엔진 모두 공개 어댑터를 통해
  `incidence_side='top'`과 `'bottom'`을 지원합니다.
- 각도는 라디안입니다. 두 엔진 모두 실수 입사를 요구합니다. 백엔드를 생략한
  엔진은 부호 있는 스칼라 `theta`를 받지만, 명시적 백엔드는
  `0 <= theta < pi/2`를 요구합니다. 명시적 경로에서는 방위각 `phi`로
  횡방향을 선택합니다. 직교 좌표 `kx`/`ky` 입사는 거부합니다.
- `result.incident`는 요청한 편광 상태입니다. TE/TM 선언은
  `'te'`, `'tm'` 또는 `wavebit.compat.legacy.Tetm` 멤버를 받으며, 문자열은 대소문자를 구분하지 않습니다.
  복소 `r_s`, `r_p`, `t_s`, `t_p`는 순수 TE/TM을 포함해 각 출사 차수 고유의 s/p 좌표계를 사용하고,
  셀 중심을 측면 위상 원점으로 삼습니다.
  반사는 빛이 들어오는 면을, 투과는 반대쪽 면을
  기준으로 합니다. R/T는 전력 비율이며, `power_entering`과 `absorption`은
  이 엔진들이 지원하는 실수 양의 입사 반공간을 사용합니다.
- 직교 좌표 차수 진폭 `r_xyz`/`t_xyz`와 RCWA 재구성 상태는
  주기 FDTD가 제공하지 않습니다. 기록된 필드나 주파수 모니터를 사용하세요.

백엔드 이름은 `'numpy'`, `'jax'`, `'torch'` 또는 해당하는
`wavebit.Backend` 멤버이며, 정수 선택자는 거부합니다. `backend`를 생략하면
레거시 NumPy 엔진을 선택합니다. 이는 `'numpy'`를 명시적으로 선택하는 것과 다릅니다.

결과의 산란 진폭은 두 엔진에서 같은 정규화를 공유하지만,
원시 진단 필드는 그렇지 않습니다.

| 진단 | 백엔드 생략 | 명시적 백엔드 |
| --- | --- | --- |
| 주파수 모니터 필드 | 소프트 소스 펄스 스펙트럼으로 나눔. 입사 정규화에는 짝이 맞는 빈 기준 계산을 사용 | 각 파장에서 빛이 들어오는 면의 단위 입사 E로 보정 |
| 시간 기록 | 요청한 Jones 상태를 포함해 전파된 소스 파형 | 일정한 중심 주파수 보정을 적용한 인과적 반송파 |
| 필드 영역 | `region`으로 구분하는 TFSF 전체/산란 영역 | 전류 시트 양쪽 모두 전체 필드 |

H는 두 주기 경로 모두에서 `Z0 H`로 저장되며, `power_z`는 공개 +z 방향을 따라
부호를 가집니다. 위에서 들어와 투과한 전력은 음수, 아래에서 들어오면 양수입니다.
회절 T는 음이 아닌 출사 전력 비율로 남습니다. 각 필드의
native `t/z/y/x` 축과 공개 `source_z`, `layer_planes`, `z_nodes`,
`pml_bounds` 메타데이터를 읽으세요. 레거시 진단 `grid`는 내부 수치 격자이며
공개 좌표 맵이 아닙니다. y 방향으로 불변인 필드는 `y=0`에서 샘플링하며,
사용하지 않는 두 번째 주기를 바꿔도 위상이 변하지 않습니다.

이전 세부 사항과 정확한 통합 범위는
[솔버 통합 체크포인트](../../development/solver-integration.md)를 보세요.
:::

## 같은 장면, 두 가지 방식

```python
import numpy as np
from wavebit import Material, Units, fdtd, rcwa

units = Units(length='nm')
air, glass, silicon = Material(epsilon=1.), Material(epsilon=1.5**2), Material(epsilon=3.5**2)

# A 700 nm cell centred on x = 0, lit from above at 900 nm; a 300 nm ridge fills x = 0..350 nm.
harmonic = rcwa.run(rcwa.Problem(
    units=units,
    stack=rcwa.Stack(period=(700., 700.), top=air, bottom=glass, layers=(rcwa.Layer(
        thickness=300., background=air, regions=(rcwa.Region(
            geometry=rcwa.Rectangle(size=(350., 700.), center=(175., 0.)), material=silicon),)),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization='s'),))),
    numerics=rcwa.Numerics(orders=(40, 0)))

# The same cell, one cell deep in y; the substrate continues into the bottom PML.
problem = fdtd.Problem(
    units=units,
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(700., 10., 1200.), center=(0., 0., 150.)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(
            fdtd.Structure('substrate', shape=fdtd.SnappedSlab(z=(-450., 0.)),
                           material=glass),
            fdtd.Structure('ridge',
                           shape=fdtd.SnappedBox(size=(350., 700., 300.), center=(175., 0., 150.)),
                           material=silicon))),
    sources=(fdtd.PlaneWave('light', plane=fdtd.Plane(z=650.), polarization='s',
                            pulse=fdtd.GaussianPulse(wavelength=900., frequency_sigma_hz=2e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=500.), wavelengths=(900.,),
                                source='light', wave='reflected', orders=(1, 0),
                                reference_plane=300., incident_plane=300.),
        fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=-300.), wavelengths=(900.,),
                                source='light', wave='transmitted', orders=(1, 0),
                                reference_plane=0., incident_plane=300.))))
time_domain = fdtd.run(problem, numerics=fdtd.Numerics(
    grid=fdtd.UniformGrid(spacing=10.),                                  # 10 nm cells
    duration=fdtd.Duration(periods=80, source='light')))

reflected, transmitted = time_domain['r'], time_domain['t']
for m in (-1, 0, 1):
    a, b = harmonic.transmission.order(m).efficiency[0], transmitted.order(m).efficiency[0]
    print(f'T[{m:+d}]   rcwa {a:.6f}   fdtd {b:.6f}   gap {abs(a - b):.1e}')
print(f'R + T    rcwa {harmonic.R.sum() + harmonic.T.sum():.6f}   '
      f'fdtd {reflected.efficiency.sum() + transmitted.efficiency.sum():.6f}')
print(f'T[+1] - T[-1] = '
      f'{abs(transmitted.order(1).efficiency[0] - transmitted.order(-1).efficiency[0]):.1e}')
```

```
T[-1]   rcwa 0.221306   fdtd 0.218569   gap 2.7e-03
T[+0]   rcwa 0.352483   fdtd 0.350980   gap 1.5e-03
T[+1]   rcwa 0.221306   fdtd 0.218569   gap 2.7e-03
R + T    rcwa 1.000000   fdtd 1.000014
T[+1] - T[-1] = 2.8e-17
```

두 문제는 장면이 아니라 `Material`과 `Units` 객체를 공유합니다. RCWA는 층으로 이루어진
[`Stack`](../rcwa/scene.md)을, FDTD는 구조물로 이루어진 도메인을 읽습니다. 여기서는 모든 면이
셀 경계에 놓이므로 `fdtd.check`는 스냅된 면을 보고하지 않습니다. 각 `DiffractionMonitor`는
RCWA의 `result.transmission`과 `result.reflection`이 반환하는 것, 즉 `order(m).efficiency`와
앞쪽에 파장 축이 있는 복소 `amplitude_ratio_s`, `amplitude_ratio_p`를 반환하며, 위상은 RCWA와 마찬가지로
`reference_plane`과 `incident_plane`이 지정한 면으로 옮겨집니다. FDTD 수치는
2026-10-15에 현재 모듈로 다시 생성했습니다. 과거 팩토리는 같은 10 nm 격자에서 2.0e-3의
차이를 출력했습니다. 이 차이는 격자에서 생깁니다. 2차 정확도이므로 5 nm 셀에서는
가장 큰 차이가 7.0e-4로 줄어듭니다(과거 경로는 5 nm에서 ~7e-4, 2.5 nm에서 ~1.9e-4를 측정했습니다).

**`R + T`가 정확히 1이 아닌 것은 결함이 아니라 방법의 특성입니다.** Yee 격자는 다른 모든 양을
계산하는 것과 같은 차수인 `O(dz^2)`까지 에너지를 보존하므로, 손실 없는 장면도 1에서
조금 벗어난 값을 돌려줍니다. 대조를 위해 보면 위의 RCWA 행은 1e-15의 몇 배 이내로 1입니다.
여기에는 의도적으로 이에 대한 일반 가드가 없습니다. 예전에 하나 있었지만 평범한
시간 영역 계산 243건 중 85건에서 발동해 제거했습니다([`result-guards.md`](../../reference/common/result-guards.md)).

**`T[+1] == T[-1]`은 수직 입사에서 이 장면의 대칭성 검사입니다.** 대칭적인
이산화는 거친 격자에서도 이 등식을 유지할 수 있으므로, 어느 효율의 오차도
제한하지 않습니다. 정확도를 평가하려면 격자와 실행 시간을 세밀하게 늘려 보세요.

## 둘 중 어느 것을 쓸까

같은 정확도에서 RCWA는 단일 파장을 112배에서 2850배 빠르게 계산하므로,
문제는 언제나 몇 개의 파장이 필요한가뿐입니다. 설정이 아니라 정확도를 맞춘 비교,
즉 같은 최악의 차수별 오차에 도달하는 가장 거친 셀과 가장 저렴한 절단(`rcwa.Numerics(orders=...)`;
이 표를 측정한 과거 인터페이스에서는 `truncation_order`)으로 비교하면 손익분기점은 다음과 같습니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스로 측정했습니다. 그 `truncation_order`는 현재 `rcwa.Numerics(orders=...)`의 각 주기 축 방향 반폭 `N`입니다.

| 장면 | FDTD 1회 실행 | 이에 맞먹는 가장 저렴한 RCWA | 손익분기 파장 수 |
| --- | --- | --- | --- |
| 손실 없는 유전체 격자, 20 nm 셀 | 0.5 s | `truncation_order` 5, 0.51 ms | ~1000 |
| 같은 격자, 10 nm 셀 | 2.0 s | `truncation_order` 10, 0.71 ms | ~2850 |
| 60 nm Drude 금속 격자, 10 nm 셀 | 2.4 s | `truncation_order` 60, 21.4 ms | ~112 |
:::

RCWA 쪽이 어려워질수록 이 수치는 시간 영역 쪽으로 기울고, 격자를 세밀하게 할수록
반대쪽으로 기웁니다. 따라서 **한 번의 실행으로 대역을 얻고 싶고 주파수 영역 계산이
비쌀 때**(금속, 많은 층, 높은 절단 차수) 시간 영역을 쓰세요. 또한 RCWA가 전혀
표현할 수 없는 두 가지, 즉 필드가 도착하는 과정의 기록과, 파장마다 복소수 하나가 아니라
극점 모델로 주어진 재료에도 시간 영역을 쓰세요.

조화 필드(`rcwa.Outputs(fields=True)`와 `result.field(...)`), 기울어진 `lattice`,
타원 영역(현재 FDTD 모듈에는 `Cylinder`와 `Prism` 물체는 있지만 타원은 없습니다. 과거 명시적 경로의
[면적 레시피](#rotated-rectangle-area-integration)는 층마다 원이나 타원을 하나씩 받습니다), 소멸파(복소 각도) 입사에는
`rcwa`를 쓰세요. 현재
FDTD 모듈은 JAX와 PyTorch에서 재료 값, 소스 매개변수, 범위가 지정된
스무딩 물체의 기하를 미분합니다({ref}`실행 미분하기 <fdtd-model-differentiating>`).
과거의 명시적 백엔드 경로는 여기에 아래에서 설명하는 각도, 형상, 두께 미분을 더합니다.
현재 `PlaneWave`는 `'s'` 또는 `'p'`를 받습니다. 타원 편광 회절은
과거 팩토리에서 `psi`와 `chi`, 또는 `polarization=(a_s, a_p)`로 사용할 수 있습니다. 예를 들어 `(1, 1j)`는
s/p 좌표계에서의 원편광 상태입니다. 이 쌍은 전체 위상을 유지한 채 정규화됩니다.
시간 기록도 추가 직교(quadrature) 소스 실행을 통해 복소 계수를 지원합니다.
[편광 세부 사항](fdtd.md#elliptical-and-circular-incidence)을 보세요.

## 시간 영역 실행이 추가하는 키워드

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 키워드와 기본값은 현재 `fdtd` 모듈이 아니라 과거 팩토리의 것입니다(대체 항목은 다음 표에 있습니다).

과거 팩토리에서 장면 키워드는 과거 RCWA 팩토리의 키워드와 같습니다. `structure`,
`thickness`, `top`, `bot`, `period`, `mu`, `wavelength`, `theta`, `phi`, `tetm`, `psi`입니다.
wavebit의 다른 모든 곳과 마찬가지로 **각도는 라디안**입니다.
아래 표는 **백엔드를 생략한 NumPy 경로**의 기본값입니다. 명시적 백엔드의
기본값은 [공통 장면 어댑터](#common-scene-adapter)에 있습니다. 이 기본값들은 측정으로 정했으며,
그 측정은 [`fdtd.md`](fdtd.md)에 있습니다.

| 키워드 | 기본값 | 용도 |
| --- | --- | --- |
| `resolution` | **장면에서 가장 조밀한 재료 안에서** 파장당 20개 셀 | 단위 길이당 셀 수이므로, 길이 단위가 nm이면 `resolution=0.1`은 10 nm 셀입니다. 비용을 좌우하는 손잡이입니다. 두 배로 늘리면 공칭 셀 크기가 절반이 되고, 완전한 3D에서는 작업량이 대략 16배(셀 8배, 단계 2배)가 될 수 있습니다. 수렴 속도는 기하에 따라 다르며, 날카로운 모서리는 측정된 세분화 연구가 필요합니다. 극점 모델은 셀 크기를 정하는 기준 굴절률을 높이므로, Drude 금속은 따로 요청하지 않아도 세밀한 격자를 받습니다 |
| `n_periods` | `80` | 실행할 광학 주기 수입니다. 누적 DFT가 의미를 갖기 전에 펄스가 들어와서 산란되고 빠져나가야 합니다. 40에서는 투과 모니터에 아직 최대값의 61%가 남아 있어 답이 0.09만큼 틀리고, 80에서는 잔차가 5e-6이며 답이 더 이상 움직이지 않습니다. 48 주기 미만의 실행은 소스가 방출을 끝내지 않았으므로 바로 거부합니다. 단지 안정되기에 너무 짧은 실행은 `UnsettledFieldWarning` 경고를 내고 `residual`을 보고합니다. 비스듬하거나 스치듯에 가까운 장면은 수직 입사 장면의 두 배가 필요한 경우가 많습니다 |
| `n_pml` | `12` | 흡수층 깊이이며 길이가 아니라 **셀** 단위입니다. 올바른 단계형 CPML은 dz에 무관하므로 셀이 알맞은 단위입니다. 12에서 층은 1.1e-5를 반사하고, 20이면 1.5e-6이 됩니다. 2 미만은 거부합니다. 그 이상의 얇은 층은 받아들이지만 안전하지 않습니다. 평범한 슬래브에서 2개 셀은 RCWA가 0.885를 주는 곳에서 경고 없이 T = 5.03을 반환했습니다. 바꾸려면 두 배로 늘려 답이 유지되는지 확인하세요 |
| `pml_reflection` | `1e-8` | 흡수층의 강도이며, 단계형 감쇠가 목표로 하는 공칭 왕복 진폭으로 나타냅니다. `n_pml`은 그 깊이입니다. 두 흡수층과 입사파 자체의 흡수층에 모두 적용합니다. [CPML strength](#cpml-strength)를 보세요 |
| `courant` | `0.99` | 안정성 한계에 대한 비율입니다. 한계에는 `n_min`이 들어 있으므로 진공보다 빠른 재료(`eps < 1`)도 발산하지 않고 안정합니다. 실행이 발산할 때만 낮추세요 |
| `subpixel` | `True` | 각 Yee 성분 고유의 셀에 대해 재료를 평균합니다. 그 성분이 가리키는 축 방향으로는 조화 평균, 나머지 방향으로는 산술 평균입니다. 셀을 가로지르는 경계에서 1차가 아니라 2차 정확도를 얻게 해 주는 것이 이것입니다. 5 nm 격자에서 점 샘플링의 3.4e-3에 비해 2.7e-4입니다 |
| `grading` | 꺼짐 | 기하 비율(예: `1.06`)로, 구조물 안에서는 세밀한 셀을 유지하고 각 반공간 쪽으로 갈수록 거칠게 합니다. `x`는 균일하게 유지됩니다. 차수 추출이 그 축에 대한 FFT이기 때문입니다. 기본값이 꺼짐인 것은 조심성이 아니라 측정 결과입니다. 같은 정확도에서 금속 격자에서는 1.6배 빠르지만 유전체 격자에서는 오히려 *손해*입니다 |
| `dispersion`, `poles` | 꺼짐 | 주파수 의존성이며, 반드시 쌍으로 줍니다. 어떤 셀이 분산성인지 표시하는 같은 모양의 정수 래스터(`0` = 없음)와, 그 밖의 각 id에 대한 `{id: [(omega0, gamma, strength), ...]}`입니다. `omega0 = 0`이면 Drude, 그 밖에는 Lorentz입니다 |
| `record_field` | 꺼짐 | 전기장을 N 단계마다 진단에 보관하며, 측정한 축도 함께 보관합니다. 꺼져 있으면 비용이 없습니다. 버퍼를 할당하는 것은 이 키워드뿐입니다 |
| `pad_scale` | `1.0` | 구조물과 TFSF 면, 모니터, PML 각각 사이의 여유 거리에 곱하는 배수입니다. `1.0`은 해당 반공간에서 반 파장입니다 |
| `truncation_order` | 모든 전파하는 차수 | 보고할 회절 차수의 수이며, 대역에서 **가장 짧은** 파장에 대해 계산합니다 |
| `return_diagnostics` | `False` | `result` 대신 `(result, diagnostics)`를 반환합니다 |

`wavelength`는 숫자뿐 아니라 **시퀀스**도 받으며, 여기서 의미가 넓어지는 키워드는
이것 하나입니다. 한 번의 실행에서 대역이 나오고, 항목마다 `Result`가 하나씩 나옵니다.
:::

현재 모듈에는 이런 종류의 기본값이 없습니다. 격자와 실행 시간은 항상
선언하며, `check`는 {ref}`정확도 규칙 <fdtd-model-accuracy-rules>`에 따라 선언된 장면을
받아들이거나 거부합니다. 각 키워드가 어디로 옮겨졌는지, 무엇이
다른지는 다음과 같습니다.

| 과거 키워드 | 현재 모듈 | 그곳의 기본값과 규칙 |
| --- | --- | --- |
| `resolution` | `fdtd.UniformGrid(spacing=dx)`, 모든 축에 하나의 정육면체 간격 | 없음. `check`는 어떤 재료에서든 파장당 8개 셀 미만을 거부하고 12 미만이면 경고합니다 |
| `n_periods` | `fdtd.Duration(periods=N, source=name)`, 또는 `seconds=`나 `steps=` | 없음. 최소값이 없으며, 끝날 때 모니터 영역에 아직 필드가 남아 있으면 `UnsettledFieldWarning` |
| `n_pml` | `Numerics(absorber=...)` 안의 `fdtd.PML(cells=12)` | 12. `allow_thin=True`가 아니면 더 적은 값은 거부합니다 |
| `pml_reflection` | `fdtd.PML(reflection=1e-8)` | 같은 기본값 |
| `courant` | `fdtd.Numerics(courant=0.5)` | 3차원 진공 한계 `dx / (c√3)`의 0.5 |
| `subpixel` | 형상이 결정합니다. `SnappedSlab`과 `SnappedBox`는 셀 스냅, `Box`, `Sphere`, `Cylinder`, `Prism`은 스무딩 | — |
| `grading` | 사용할 수 없음. 격자는 균일합니다 | — |
| `dispersion`, `poles` | `DispersiveMaterial(epsilon_inf, poles=(Drude(...), Lorentz(...), Debye(...)))`, 비율은 SI 단위 | — |
| `record_field` | `fdtd.TimeSeriesMonitor(name, at=..., component=...)`: 한 Yee 점의 한 성분 | — |
| `pad_scale` | `Domain` 크기. 각 PML은 그 바깥에 덧붙입니다 | 구조물 자신의 매질에서 반 파장보다 가까우면 `AbsorberProximityWarning` |
| `truncation_order` | `fdtd.DiffractionMonitor(orders=(Mu, Mv))`, 명시적인 쌍. 정수 하나만 주면 거부합니다(`monitor.orders`) | 모든 전파하는 차수 |
| `return_diagnostics` | `result.validity`, `result.diagnostics`, `result.provenance`, 항상 존재 | — |
| `wavelength` (대역) | 각 주파수 영역 모니터의 `wavelengths` | — |

현재 모듈은 또한 기본 흡수층이 감당할 수 없는 스치듯 각도로 PML을 통과해 나가는 회절 차수를
거부하며, 그 차수를 받아들일 수 있는 더 깊은 `PML`과 더 긴 `Duration`을 알려 줍니다.
다음 예제가 이 규칙에 부딪힙니다.

## 분산, 그리고 한 번의 실행에서 얻는 대역

시간 영역 계산은 복소 유전율을 받을 수 없습니다. 복소수 하나는 한 주파수에서의 재료를
기술할 뿐이지만, 펄스는 대역 전체를 실어 나릅니다. 주파수 의존성은 그 본래의 형태인
극점 모델로 주며, 대역 안의 각 파장은 그 모델이 *그 파장에서* 갖는 유전율을 봅니다.
손익분기 표가 측정하는 것이 바로 이 조합입니다. 한 번의 실행과, 파장마다 서로 다른 `eps`를
받는 파장별 RCWA 계산을 비교합니다.

```python
import numpy as np
from wavebit import DispersiveMaterial, Drude, Material, Units, fdtd, rcwa
from wavebit.units import C0

units = Units(length='nm')
# Gold-ish Drude, one pole, with SI rates: omega_p = 2 pi c / 136 nm, gamma = 2 pi c / 17 um.
gold = DispersiveMaterial(epsilon_inf=1., poles=(
    Drude(omega_p=2 * np.pi * C0 / 136e-9, gamma=2 * np.pi * C0 / 17e-6),))
air, glass = Material(epsilon=1.), Material(epsilon=1.5**2)
band = (820., 900., 980.)

# A 60 nm gold strip over x = -350..0 nm of a 700 nm cell, on glass, lit from above.
problem = fdtd.Problem(
    units=units,
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(700., 10., 1050.), center=(0., 0., 75.)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(
            fdtd.Structure('substrate', shape=fdtd.SnappedSlab(z=(-450., 0.)),
                           material=glass),
            fdtd.Structure('strip',
                           shape=fdtd.SnappedBox(size=(350., 700., 60.), center=(-175., 0., 30.)),
                           material=gold))),
    sources=(fdtd.PlaneWave('light', plane=fdtd.Plane(z=500.),
                            pulse=fdtd.GaussianPulse(wavelength=900., frequency_sigma_hz=2e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=350.), wavelengths=band,
                                source='light', wave='reflected'),
        fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=-300.), wavelengths=band,
                                source='light', wave='transmitted'))))
# The -1 and +1 orders leave the glass at cos(theta) = 0.36 at 980 nm, a grazing exit the
# default PML refuses; the refusal asks for this deeper absorber and at least 160 periods.
spectrum = fdtd.run(problem, numerics=fdtd.Numerics(                      # one run
    grid=fdtd.UniformGrid(spacing=10.), duration=fdtd.Duration(periods=160, source='light'),
    absorber=fdtd.PML(cells=24, reflection=6e-10)))

reference = rcwa.run(rcwa.Problem(                                        # one solve per wavelength
    units=units,
    stack=rcwa.Stack(period=(700., 700.), top=air, bottom=glass, layers=(rcwa.Layer(
        thickness=60., background=air, regions=(rcwa.Region(
            geometry=rcwa.Rectangle(size=(350., 700.), center=(-175., 0.)), material=gold),)),)),
    excitation=rcwa.Excitation(wavelength=np.array(band),
                               sources=(rcwa.PlaneWave(polarization='s'),))),
    numerics=rcwa.Numerics(orders=(60, 0)))

eps = gold.at(np.array(band), units=units).epsilon   # the closed form the stepping discretises
R = spectrum['r'].efficiency.sum(axis=(1, 2))
T = spectrum['t'].efficiency.sum(axis=(1, 2))
for i, wl in enumerate(band):
    print(f'{wl:.0f} nm  eps {eps[i].real:8.3f}{eps[i].imag:+.3f}j   '
          f"T0 fdtd {spectrum['t'].order(0).efficiency[i]:.4f} "
          f'rcwa {reference.transmission.order(0).efficiency[i]:.4f}   '
          f'absorbed fdtd {1 - R[i] - T[i]:.4f} '
          f'rcwa {1 - reference.R[i].sum() - reference.T[i].sum():.4f}')
```

```
820 nm  eps  -35.269+1.749j   T0 fdtd 0.4073 rcwa 0.4049   absorbed fdtd 0.0107 rcwa 0.0111
900 nm  eps  -42.671+2.312j   T0 fdtd 0.4090 rcwa 0.4060   absorbed fdtd 0.0126 rcwa 0.0131
980 nm  eps  -50.753+2.983j   T0 fdtd 0.4206 rcwa 0.4172   absorbed fdtd 0.0149 rcwa 0.0154
```

FDTD 열은 2026-10-15에 현재 모듈로 다시 생성했습니다(RCWA 열은 1e-12까지
변하지 않았습니다). 과거 팩토리는 100 주기와 자체 12셀 흡수층에서
T0 0.4036, 0.4039, 0.4144와 흡수 0.0112, 0.0135, 0.0162를 출력했습니다. 셀 스냅된 띠는
`AbsorberProximityWarning`도 일으킵니다. 이 규칙은 반 파장 여유 거리를 금속 굴절률의 실수부로
정하는데, 그 값이 작아서 마이크로미터 단위의 거리를 요구합니다. 여기의 450 nm와 540 nm 간격은
과거 경로가 유리(300 nm)와 공기(450 nm)에서 패딩으로 둔 반 파장 이상입니다.

대역은 근사가 아니라 항등식입니다. 각 파장은 한 번의 실행에 대한 각자의 푸리에
변환이며, 같은 펄스로 900 nm만 요청한 실행도 1e-14까지 같은
효율을 반환합니다. 과거 팩토리는 요청한 범위로 펄스 크기를 정했고, 거기서는
단일 파장 실행 세 번이 대역 결과와 3e-3까지 일치했습니다. 이는 두 답 사이의 차이가 아니라 두 소스
*대역폭* 사이의 차이였습니다. 여기서는 펄스를 선언합니다. 그 `frequency_sigma_hz`가 대역을 정하고,
모든 진폭을 측정된 입사 스펙트럼으로 나누며, 펄스가 `cutoff` 아래로 떨어진 파장은
`result.validity.checks['source_reference']`가 거짓인 채로 NaN을 돌려줍니다. 이는 과거
경로가 고정된 펄스 폭으로 한때 측정했던 현상에 대한 방어입니다. 최대값의 2.4e-7인 입사 스펙트럼이
실제 `R`이 0.04인 장면에서 `R = 105`를 반환했습니다.

**비스듬한 입사에서 대역은 물리적 횡방향 `(kx, ky)`를 고정합니다.**
따라서 극각은 파장에 따라 달라집니다. 900 nm에서 중심 각도가 30도라면 860 nm에서는 약 28.5도,
940 nm에서는 31.5도입니다. 현재 모듈에서는
`Bloch.from_source`의 `Bloch` 면이 이를 펄스의 중심 파장에서 고정하며,
각 `DiffractionMonitor` 결과는 극각을 라디안 단위로
`metadata['theta_radians']`에 보고합니다. 과거 팩토리에서는 `diagnostics['angles']`가
극각을 라디안으로, `azimuth`가 phi를 보고합니다. 백엔드를 생략한
경로는 `angles`의 도 단위 호환 보기인 `angles_deg`도 유지합니다.
수직 입사는 영향을 받지 않습니다.
`sin 0`은 모든 파장에서 0이므로, 각도를 고정한 대역은 수직 입사에서만 얻을 수 있습니다.

## 필드가 도착하는 모습 보기

조화 솔버는 말할 수 없고 이 솔버만 말할 수 있는 것이 하나 있습니다. 조화 계산은 `e^{-iwt}`를
가정하고, 따라서 단일 주파수가 문제 전체를 기술한다고 가정한 상태에서 필드가 *무엇인지*에
답합니다. 빛이 도착하는 그림은 없습니다. 빛이 언제나 이미 와 있었다고 가정하기 때문입니다.
`record_field=N`은 전기장을 N 단계마다 보관하며, **고유한 좌표축도 함께** 보관합니다.
좌표축을 직접 다시 만드는 사람은 세 가지 점에서 틀립니다. 비균일 메시에는 하나의 `dz`가 없고,
면 내 좌표는 셀 중심에서 측정하며, `Ey`는 픽셀 중심이 아니라 x 노드에 놓입니다.

현재 모듈은 한 성분의 시계열을 가장 가까운 Yee 점에서 보관하는
`fdtd.TimeSeriesMonitor(name, at=..., component=...)`(시간은 초, 필드는 V/m 또는 A/m)와, 최종 E와
H를 보관하는 `fdtd.Outputs(final_fields=True)`를 제공하며, 각각 샘플링한 좌표를 함께 줍니다.
최상위 장면은 시간에 따른 전체 도메인을 기록하지 않습니다. `wavebit.fdtd.periodic`은 기록합니다
(`Outputs(snapshots=Snapshots(...))`). 이 예제와 아래의 그리기는 과거
팩토리를 사용합니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic Outputs(snapshots=Snapshots(...)).
import numpy as np
from wavebit.compat import legacy as wavebit
from wavebit.compat.legacy import Stack, Layer, Incidence

eps_hi = wavebit.eps_from_refractive_index(3.5)
scene = Stack(layers=[Layer(thickness=300.,
                            structure=np.array([[1., 1., eps_hi, eps_hi]]))],
              top=1., bot=wavebit.eps_from_refractive_index(1.5), period=(700.,))
light = Incidence(wavelength=900., theta=0., phi=0., tetm='te')

result, diagnostics = wavebit.fdtd(stack=scene, incidence=light, resolution=0.1,
                                   record_field=100, return_diagnostics=True)

print('residual', f"{diagnostics['residual']:.1e}")
print('component', diagnostics['field_e_component'])
print('field_e  ', diagnostics['field_e'].shape, '(snapshots, nz, ny, nx)')
print('field_x  ', f"{diagnostics['field_x'][0]:.1f} .. {diagnostics['field_x'][-1]:.1f}")
print('field_z  ', f"{diagnostics['field_z'][0]:.1f} .. {diagnostics['field_z'][-1]:.1f}")
```

```
residual 4.6e-06
component ey
field_e   (103, 154, 1, 70) (snapshots, nz, ny, nx)
field_x   -350.0 .. 340.0
field_z   -420.0 .. 1110.0
```

`field_e`는 모든 실행에서 `(snapshots, nz, ny, nx)`이므로, 교차 격자는 모양이 다른 배열이 아니라
`ny`가 더 큰 배열입니다. `field_e_component`는 어떤 성분을 담고 있는지 알려 줍니다.
TE 실행과 conical 입사의 두 갈래 모두에서는 `ey`, TM에서는 `ex`입니다. 두 성분은
같은 위치에 있지 않으므로 이 구분이 중요합니다. Yee 격자는 `Ey`를 x 노드에, `Ex`를 반 칸 위치에 엇갈려 배치합니다.
따라서 `field_x`와 `field_y`는 *그* 성분의 축이며, conical 실행은
`field_ex`와 그 고유의 축 쌍을 함께 담습니다.

축을 읽지 않고 다시 만들면 세 가지 점에서, 그것도 모두 조용히 틀립니다. 메시가 비균일이면 `field_z`는
`arange(nz) * dz`가 아니고, 이 라이브러리의 모든 면 내 좌표는 셀 중심에서 측정하므로
`field_x`는 0에서 시작하지 않으며, `wavebit.compat.legacy.meshgrid_xy`가 반환하는 픽셀
중심은 `Ey`가 있는 곳에서 반 칸 떨어져 있습니다. 세 가지 모두의 근거가 되는 측정은
[`fdtd.md`](fdtd.md) §8h에 있습니다.

같은 기록을 선택적 Matplotlib(`pip install 'wavebit[plot]'`)으로
그리거나 애니메이션으로 만들 수 있습니다.

```python
# Earlier interface; plot_fdtd_field and animate_fdtd_field have no current public equivalent.
# Use record_field=10 in the solve above to resolve the optical oscillation.
fig, ax, mesh = wavebit.plot_fdtd_field(diagnostics, frame=len(diagnostics['field_t']) // 2)
animation = wavebit.animate_fdtd_field(diagnostics, interval=50)
# Keep animation alive, then use plt.show() or display(HTML(animation.to_jshtml())).
# animation.save('field.gif', writer='pillow')
```

두 함수 모두 가장 가까운 기록된 `y` 좌표(기본값 0)에서 x-z 단면을 보여 주며,
제목에 실제로 선택한 행을 표시합니다. `component='ex'`는 기록되어 있으면 Ex를 선택하며,
그 고유의 엇갈린 좌표를 사용합니다. `part='real'`은 기본적으로 부호 있는 E를 보여 주며, `'imag'`와 `'abs'`도
사용할 수 있습니다. E의 절댓값은 세기가 아니라 필드 진폭입니다. 색 척도는 선택한
시간 이력 전체에 걸쳐 고정됩니다. 비균일화한 z 좌표는 비균일 메시로 그리고,
PML 셀은 음영으로 표시하며, TFSF 선은 기록이 입사 쪽의 산란 필드에서
적층 쪽의 전체 필드로 바뀌는 곳을 표시합니다. 위에서 입사하면
위쪽이 산란 필드, 아래쪽이 전체 필드이고, 아래에서 입사하면 그 영역이 뒤바뀝니다.
층 경계면을 표시하지만, 면 내 기하를 암시하지는 않습니다.

시간은 솔버의 `c=1` 단위이며, 길이와 같은 단위를 씁니다. 재생 `interval`은
스냅숏당 밀리초이며 시뮬레이션 시간을 바꾸지 않습니다. 샘플링이 부족한 기록은
경고를 냅니다. 애니메이션을 느리게 해도 빠진 광학 주기를 되살릴 수 없으므로 계산에서
`record_field`를 줄이세요. `record_field`는 양의 정수여야 합니다.

`psi`를 사용하면 기록에는 두 소스 편광을 요청한 대로 결맞게 결합한 결과가
Ey와 Ex 모두 포함됩니다. 진폭은 원시 소스 척도를 유지하며, 정규화된
모니터 계수가 아닙니다. 안정화 진단은 두 기저 실행 중 더 큰
잔차를 보고합니다. 기록은 여전히 기본적으로 꺼져 있습니다.
:::

## 더 읽을거리

[`fdtd.md`](fdtd.md)는 결정 문서입니다. 각 기본값을 무엇에 대해 측정했는지,
거부 목록 전체와 각 거부가 무엇을 보호하는지, 그리고 조화 솔버는 말할 수 없고
시간 영역 솔버만 말할 수 있는 두 가지 질문을 담고 있습니다. [`result-guards.md`](../../reference/common/result-guards.md)는
계산이 자기 답에 대해 무엇을 검사하는지, 그리고 왜 일반적인 에너지 경고가 없는지 설명합니다.
[FDTD 솔버가 문제를 모델링하는 방식](model.md)은 현재 모듈의 격자, 구조물,
경계, 소스, 모니터와 실행이 통과해야 하는 정확도 규칙을 다룹니다.


:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

전기장과 자기장의 모든 성분을 기록하려면 `record_components='all'`을 추가하세요.
`diag['fields']['hz']['values']`와 그 `t/z/y/x` 축을 읽거나, 그리기·애니메이션 도우미에
`component='hz'`를 넘기세요. H는 `Z0 H`로 정규화되며, 그 타임스탬프는 E와
반 시간 간격만큼 다릅니다. 엇갈린 필드를 결합하기 전에 [성분 배치](fdtd.md#recording-all-six-field-components)를
보세요.


`wavebit.compat.legacy.fdtd_poynting(diag, z=50., x=0., y=0.)`는 여섯 필드를 모두 공통 Yee 위치와
시각으로 보간한 뒤, 한 점의 순간 에너지 흐름을 `S[:, 0:3]`에 반환합니다.
`sample_fdtd_fields`는 보간한 E/H를 직접 반환합니다. 둘 다
`record_components='all'`이 필요합니다. 단위, 샘플링 한계, 전체 플럭스와 산란 플럭스의 구분은
[점 탐침](fdtd.md#point-probes-and-instantaneous-energy-flow)을 보세요.
:::


## 스트리밍 주파수 평면

현재 모듈에서 `fdtd.FieldMonitor(name, wavelengths=..., plane=fdtd.Plane(z=...))`는
여섯 성분의 원시 푸리에 페이저를 각자의 Yee 점에서 반환하고(V·s/m와
A·s/m), `fdtd.FluxMonitor`는 한 평면을 지나는 알짜 스펙트럼 에너지(J/Hz)를 반환합니다. 둘 다
소스 스펙트럼으로 나누지 않습니다. 이 절의 나머지는 과거 팩토리를 설명합니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`wavebit.compat.legacy.fdtd`에 `field_monitors={'near': z_near, 'transmitted': z_out}`와
`return_diagnostics=True`를 함께 넘기세요. `record_field`는 필요 없습니다.
`diag['monitors']`의 각 항목에는 복소 `fields` 배열 여섯 개 `(wavelength, y, x)`,
`x`/`y` 좌표, `flux_density` `(wavelength, y, x, 3)`, `power_z`
`(wavelength,)`가 들어 있습니다. 공간 보간은 측면 가장자리의 Bloch 위상을 포함해
native Yee 성분을 같은 위치로 모으며, 푸리에 적분은 각 성분의 native
시각을 사용합니다. 복소 Jones 계수는 전력을 계산하기 전에 필드를 결합합니다.

이 절은 **백엔드를 생략한 경우**의 모니터 정규화를 설명합니다. 정규화 기준은
소프트 소스 펄스 스펙트럼이며 **입사 전력이 아닙니다**. 명시적 백엔드의 보정은
[공유 백엔드 기록과 모니터](#shared-backend-time-records-and-user-monitors)에서 설명합니다.
H를 Z0 H로 저장하므로 플럭스는 `0.5*real(E cross conj(H))`이며, `power_z`는
그 z 성분을 주기 셀에 대해, y 불변 장면이라면 y 깊이 단위당
적분합니다. 투과를 정규화하려면 짝이 맞는 빈 장면을 실행하세요. 산란 반사
평면에서는 그 전력의 부호를 바꾸고 빈 장면의 입사 전력으로 나누세요.
전체 필드 평면에는 입사파와 반사파의 간섭이 들어 있으므로, 반사를
분리해 측정한 것이 아닙니다. `region`은 TFSF
경계의 어느 쪽인지 나타냅니다. 보간이 그 경계에 닿는 평면은 거부합니다.
이 국소 보간 플럭스는 메시를 세분화하면 수렴하며, 유한한 해상도에서
모드 회절 전력과 정확히 같을 필요는 없습니다.

세 파장 슬래브 검사는 `examples/fdtd/fdtd_frequency_planes.py`를 보세요.
저장 공간은 성분 여섯 개 × 파장 수 × 평면 면적에 비례하며,
시간 간격 수에 따라 늘어나지 않습니다. 이것은 스트리밍 푸리에 측정이며,
전체 부피의 조화 필드 재구성이 아닙니다.
:::

## NumPy, JAX, Torch가 공유하는 native 격자

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`wavebit.compat.legacy.fdtd_grid`는 NumPy, JAX, Torch가 하나의 Array API 계산
코어를 공유하는 직교 좌표 Yee 인터페이스입니다. `backend` 선택자는 RCWA와 마찬가지로
`'numpy'`, `'jax'`, `'torch'` 또는 `wavebit.Backend` 멤버를 받으며, 정수는 거부합니다.
native 격자 좌표는 주기 Stack 어댑터를 거치지 않습니다.
native `(z,y,x)` 유전율 또는 `(3,z,y,x)` 대각 계수와,
`(3,z,y,x)` 초기 E/H 및 전류 소스 배열을 받습니다.
`spacing=(dz,dy,dx)`, `dt`, `steps`를 c=1 단위로 명시하세요. E는 시각 0에서,
H는 `-dt/2`에서 시작합니다. 스칼라 또는 브로드캐스트 가능한 양의 투자율과 음이 아닌
전기 전도도를 지원합니다. 계수는 각 성분의 native Yee 위치에 두어야 하며,
이 함수는 경계면을 래스터화하거나 평균하지 않습니다.

```python
# Earlier interface; current: wavebit.fdtd.native (Medium of per-cell arrays).
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

source = jnp.zeros((3, 32, 2, 2)).at[1, 8].set(1.)
pulse = jnp.sin(jnp.arange(80) * .3)

def objective(epsilon):
    out = wavebit.fdtd_grid(
        epsilon, spacing=(1., 1., 1.), dt=.2, steps=80,
        source=source, pulse=pulse, wavelengths=[4.], monitor_z=[20],
        backend='jax', checkpoint_steps=16)
    return out['power_z'][0, 0]

value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.ones((32, 2, 2)))
```

native 격자 스펙트럼은 장면 솔버의 소스 정규화된 평면과 달리
**정규화하지 않은 시간 적분**입니다. 전기·자기 스펙트럼의 모양은
`(wavelength, plane, component, y, x)`, 플럭스 밀도의 모양은
`(wavelength, plane, y, x, 3)`, 전력의 모양은 `(wavelength, plane)`입니다.
소스 또는 입사 전력 정규화에는 짝이 맞는 기준 시뮬레이션을 사용하세요.

기본적으로 모든 축은 주기 경계입니다. `absorber_cells=N`은 z 양 끝에 단계형 전기·자기 감쇠를
추가합니다. 이것은 CPML이 아니며 폭·반사에 대한 수렴 검사가 필요합니다.
`pml_cells`는 추가로 균질한 두 z 끝에 합성곱 PML을 제공합니다.
아래의 재료·경계 절을 보세요. 이 엔진은 아직 곡선 메시나
기울어진 메시를 받지 않습니다. 정적 비균일 z 폭과 장면 비균일화는 이제 지원합니다.
아래의 공통 장면 어댑터가 Stack/Incidence 입력을 변환합니다. 레거시 장면 솔버는
NumPy에서 그 기능을 유지하며, 그 인터페이스에서 JAX 백엔드를 선택하는 것은
여전히 지원하지 않습니다.

JAX는 컴파일된 scan을 사용하고 역방향 미분 중에 블록을 다시 계산합니다.
Torch는 비재진입(non-reentrant) 블록 체크포인팅과 함께 즉시 실행 autograd를 사용하며,
`torch.compile` 아래에서는 컴파일된 그래프 밖에서 즉시 실행됩니다(
[`torch.compile`](../../development/torch-compile.md) 참고).
`checkpoint_steps`는 저장하는 블록 경계와 재계산 및 블록 내 저장 공간 사이의
균형을 조절합니다. 시간 이력은 `record_field=N`으로 선택할 수 있으며, 역방향 모드
메모리와 요청한 스냅숏 저장 공간은 여전히 시뮬레이션 길이에 따라 늘어납니다. 모양, 간격, 단계 수, 모니터 구성은
정적입니다. 재료, 소스, 펄스, 초기 필드는 미분 가능합니다.
최적화 전체에서 유효한 고정 dt를 고르세요. 구체값의 잘못된 재료나
CFL 위반은 예외를 일으키고, 추적되는 계수는 결과에 `stable`을 노출하므로 호출자가
제약해야 합니다. 최적화기는 양의 epsilon/mu와
음이 아닌 전도도를 유지해야 합니다. 배열을 만들기 전에 JAX x64(`jax.config.update('jax_enable_x64', True)`)를
켜세요. 즉시 실행 계산은 스스로 이를 켜지만, 꺼진 상태에서
`jax.jit`, `jax.grad`, `jax.vmap` 안에서 도달한 계산은 `RuntimeError`를 일으킵니다. 변환이
이미 입력을 32비트로 추적했기 때문입니다.

`device='cpu'` 또는 `device='gpu'`로 장치를 선택하며, Torch는 `'cuda'`와
장치 객체도 받습니다. device를 생략하면 Torch는 텐서 epsilon의 장치를 따르고,
그렇지 않으면 CPU를 사용합니다. 사용할 수 없는 하드웨어는 예외를 일으킵니다.
CPU 일치, 이산 고유 모드 전파, 재료·펄스 미분, 블록
체크포인트 동등성을 테스트합니다. 외부 CPU 검사는 추가로 슬래브의
복소 필드·전력을 Meep과, native 3D 갱신·재료 그래디언트를
FDTDX와 비교하며, 고정된 외부 출력이 회귀 테스트를 뒷받침합니다.
[검증 범위](fdtd-alpha.md)를 보세요. 이 과거 비교는 CPU를 사용했습니다.
현재 공개 native API에 대한 실제 Torch CUDA 근거는
[native CUDA 검증](../../validation/fdtd/native-cuda.md)에 따로 기록되어 있습니다.
`examples/fdtd/fdtd_grid_gradient.py`는 슬래브 투과를 미분하고,
`studies/fdtd/grid_benchmark/fdtd_grid_benchmark.py`는 컴파일 시간을 예열된 실행과 분리해 보고합니다.
그 측정 결과와 속도 향상이 호스트 덕분인 이유는
[설계 노트](fdtd.md#streaming-fields-and-compiled-native-grids)에 있습니다.
:::


## native 분산 재료와 CPML

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`fdtd_grid(..., poles=[(omega0, gamma, strength), ...])`는 중심 ADE 점화식으로 전기
분극을 전진시킵니다. 각 매개변수는 native 전기 성분 위치에서
`(3,z,y,x)`로 브로드캐스트됩니다. 재료 법칙은
`epsilon(omega) = epsilon_infinity + sum(strength / (omega0**2 - omega**2 - 1j*gamma*omega))`입니다.
`epsilon` 인수는 epsilon-infinity이며, `omega0=0`이면 Drude입니다. 주파수와
감쇠는 c=1 시간 단위의 **각**주파수입니다. 매개변수는 유한하고,
음이 아니며, 실수여야 합니다. epsilon-infinity와 mu는 양수여야 합니다. 극점 주파수와
세기는 보수적인 CFL 한계를 더 엄격하게 만듭니다. 추적되는 잘못된 매개변수는
`stable=False`를 반환하므로 최적화 매개변수화에서 배제해야 합니다.

두 초기 분극 수준은 모두 0입니다. 최종 `polarization`과
`previous_polarization`은 E 시각과 그 한 단계 이전의 성분 배열 튜플입니다.
따라서 활성 극점마다 E/H 외에 부피 배열 두 개를 보관합니다.
이 API는 아직 재시작 상태나 자기·비대각 분산을 받지 않습니다.
경계면 래스터화는 없으므로 native 계수 가중치를 명시적으로 주세요.
접선 방향 평면 경계면에서는 반 셀 세기 가중치가 해석적 슬래브 해에
수렴합니다. 이것은 일반적인 텐서 경계면 평균 규칙이 아닙니다.

`pml_cells=N, pml_wavelength=lambda_center`는 z 양 끝에 합성곱 PML을 추가하며,
Wavebit의 기존 계수 정식화를 native E/H 위치에서 평가해 사용합니다.
끝 층과 **인접한 내부 셀 하나**는
`pml_background=((epsilon_top, mu_top), (epsilon_bottom, mu_bottom))` 값
(기본값: 진공)
(epsilon=mu=1), 0의 전기 전도도, 0의 진동자 세기와 일치해야 합니다. x/y는
주기 경계로 남습니다. 이것은 `absorber_cells`와 함께 쓸 수 없으며, 후자는
더 단순한 단계형 감쇠 선택지로 남아 있습니다. 물리적 소스와 측정 평면은
내부에 두세요. CPML 기억 배열 네 개를 보관하며, 각각 공간
부피 모양입니다. 기하와 중심 파장은 정적이며, 양의 실수
`pml_background` 쌍은 그래디언트를 유지할 수 있습니다. CPML 늘림은 그 굴절률을
따르므로, 외부 재료 미분에는 흡수 갱신이 포함됩니다.
끝 계수는 여전히 주어진 배경과 정확히 일치해야 합니다.

검증에는 변위 보존, NumPy/JAX 일치, 세 극점 매개변수 모두에 대한 유한
차분, 체크포인트 일치, 수직·비스듬한 펄스 제거가 포함됩니다. 외부 CPU 실행은
유전체, Lorentz, Drude, 2극점 매질에 대해 복소 슬래브 필드와 전력을
Meep과 비교하고, 3D 혼합 극점 갱신·세기 그래디언트를 FDTDX와 비교합니다.
직접 비교를 위해 Meep이 문서화한 분산의 점 샘플링 방식을 명시적으로 맞추며, 별도의
가중 실행으로 Airy 해에 대한 수렴을 검사합니다. 체크아웃의
`tests/references/README.md`에 있는 외부 기준 기록을 보세요.

PML 두께와 실행 시간은 수렴 매개변수입니다. 0에 가까운 주파수 성분을 가진
넓은 펄스는 천천히 감쇠하는 CPML 꼬리를 남길 수 있으며, 단순히
폭을 늘리는 것으로는 시간 수렴을 입증할 수 없습니다. 기준 펄스는
DC가 0인 대칭 펄스이며, 테스트는 실행 시간도 늘리고 PML 폭도 두 배로 합니다.
이것으로 임의 대역이나 스치듯 각도의 정확도가 확립되지는 않습니다.
`examples/fdtd/fdtd_dispersive_gradient.py`는 CPML을 켠 상태에서 Lorentz 진동자 세기를
미분합니다. 위의 과거 외부 비교는 CPU
근거로 남아 있으며, 현재 API에는 별도의 [범위가 정해진 CUDA 필드·미분 검사](../../validation/fdtd/native-cuda.md)가 있습니다.
:::

### CPML strength

현재 모듈에서 강도는 `fdtd.PML(reflection=1e-8)`, 깊이는
`fdtd.PML(cells=12)`이며, 둘 다 `Numerics(absorber=...)`를 통해 모든 `Open()` 면에 한 번에 설정합니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스에서는 `fdtd_grid`, `wavebit.compat.legacy.fdtd(..., backend=...)`,
`backend`를 생략한 `wavebit.compat.legacy.fdtd`가 모두 `pml_reflection=1e-8`을 받습니다. 이 값은
0과 1 사이(양 끝 제외)의 유한한 실수 정적 스칼라여야 하며, 불리언, 복소수, 차원이 있는
배열은 거부합니다. 같은 설정이 아래 입사를 포함해 양 끝에 적용되고,
장면의 보정·구조 실행 네 번 모두에 적용됩니다.
`backend`를 생략하면 입사파를 나르는 1차원 격자의 흡수층도 단계형으로 만듭니다.
이 값은 native 결과나 장면 진단에 `pml_reflection`으로 기록됩니다.
native `pml_cells=0`이면 효과가 없습니다.

이것은 **공칭 수직 입사 왕복 진폭 목표**이며, 주파수 이동을 적용하기 전의
연속 프로파일에 대해 정의됩니다. 전력 목표도 아니고 측정된 반사의 상한도
아닙니다. 값을 작게 하면 다항식의 차수, kappa, 주파수 이동, 셀 폭,
PML 깊이를 바꾸지 않은 채 전도도를 키웁니다. `pml_reflection=1e-40`은 `1e-8`의 다섯 배 전도도를 주며,
40자리 정확도를 뜻하지 않습니다. 기본 프로파일은 바뀌지 않습니다.

강도와 독립적으로 두께를 바꾸려면 `n_pml`(장면) 또는 `pml_cells`(native 격자)를 사용하세요.
충분히 분해되지 않은 격자에서는 흡수를 강하게 하면 반사가 늘 수 있습니다.
강도, 물리적 두께, 해상도를 비교하고, 경계를 바꾼 뒤에는 시간 감쇠를
다시 확인하세요. 방향성 장면에서는 되돌아오는 진폭과
보정된 산란 꼬리가 경계 오차와 시간 오차를 구분하는 데 도움이 됩니다.
[레일리 근처 강도 연구](../../validation/fdtd/fdtd-pml-strength.md)에 구체적인
비교와 남은 한계가 기록되어 있습니다.
:::


## native Bloch 위상과 입사 그래디언트

현재 모듈에서 `Bloch(phase=...)` 또는 `Bloch(k=...)` 면 쌍은 같은 규약을 사용하며
실행을 복소수로 만들고, `Bloch.from_source`는 비스듬한
`PlaneWave`에 필요한 쌍을 작성합니다. 이 절의 나머지는 과거 native 격자 `fdtd_grid`를 설명합니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`bloch_phase=(phase_z, phase_y, phase_x)`는 각 시뮬레이션 주기를 가로지르는 실수 위상 전진을
라디안으로 지정합니다. `F(r+L_axis)=exp(1j*phase_axis)*F(r)`입니다. 이는 물리 벡터 성분 순서가 아니라
`spacing`의 `(z,y,x)` 순서를 따릅니다. 위상이 0이면
주기 경계로 돌아갑니다. 갱신과 모니터 보간 모두 감싸는 가장자리에서
위상을 적용합니다. z CPML에는 `phase_z=0`이 필요하며, 횡방향 위상은
CPML 및 분산 재료와 함께 계속 사용할 수 있습니다.

위상은 JAX로 미분할 수 있습니다. 평면 소스에서는 경계 위상과 함께 소스의 공간 위상도
바꾸세요. `phase_x=kx*nx*dx`, `source_y(x)=exp(1j*kx*x)`이며,
진공에서 `kx=2*pi*sin(theta)/wavelength`입니다. 복소 시간 반송파
`envelope*exp(-1j*omega*t)`는 의도한 전파 주파수를 선택합니다. 0이 아닌 고정 Bloch 파수 벡터에서
실수 사인파는 두 시간 주파수를 모두 포함합니다.
엔진은 주어진 복소 필드를 전진시킬 뿐, 직교 계산을 자동으로 추가하거나
입사 전력을 정규화하지 않습니다. 투과 비율을 구하려면 같은 소스와 Bloch 위상을 쓰는
짝이 맞는 빈 기준 계산을 사용하세요.

테스트는 세 축 모두에서 임의 위상을 갖는 정확한 이산 Maxwell 모드, 감싸인 평면에서의
모니터 보간, 유한 차분과 비교한 위상 미분, FDTDX 복소 갱신·재료 그래디언트를
검사합니다. 고정 필드 경계 문제의 미분이 자동으로 입사각 미분이
되지는 않습니다. 소스·초기 필드 의존성을 미분하는 함수 안에 포함해야 합니다.
`examples/fdtd/fdtd_bloch_gradient.py`는 진공 끝 CPML과 함께 소스와 경계를 같이
바꾸며 스펙트럼 평면 전력의 미분을 보고합니다.
:::


## 백엔드 구조

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

native 솔버는 이제 RCWA와 같은 `array_api_compat` 네임스페이스와
`wavebit._common.backend.as_array` 변환을 사용합니다. Maxwell 회전(curl), CPML, ADE, Bloch
감싸기, 푸리에 평면 누적은
`src/wavebit/fdtd/_periodic/grid.py`에 구현이 하나만 있습니다. 배열 생성·장치 선택과 시간 루프 실행은
`src/wavebit/fdtd/_periodic/runtime.py`에 있습니다. NumPy 루프, JAX scan, 또는 Torch 체크포인트
루프입니다. 물리 코드는 JAX나 Torch를 import하지 않으며 백엔드별
갱신 분기가 없습니다. 선택적 프레임워크는 해당 백엔드를 선택했을 때만 불러옵니다.

`examples/fdtd/fdtd_backends.py`는 CPML/Lorentz 목적 함수 하나를 세 백엔드 모두로
평가하고, 두 자동 미분을 NumPy 유한 차분과 비교합니다.
Torch 테스트는 또한 살아 있는 입력 그래프를 보존하고, 체크포인트 블록 크기를 비교하며,
극점, 재료, 소스, 초기 필드, Bloch 위상 미분을 실행해 봅니다. 기존
FDTDX 비분산, 2극점, 복소 Bloch 골든 값은 JAX뿐 아니라 Torch로도
검사합니다. 이 값들은 고정된 기준값이며, 테스트를 실행할 때마다 외부 도구를 새로 실행하는 것이
아닙니다. 정밀도 검사에는 명시적인 배정밀도 입력 텐서를 사용하세요.

이로써 native 격자의 **계산 백엔드**가 통합됩니다. `wavebit.compat.legacy.fdtd`
Stack/Incidence 어댑터는 `backend`를 명시적으로 주면 이 엔진을 선택합니다.
그 정적 래스터화, 전류 시트 기준 보정, 모드 Result
추출은 아래에서 설명합니다. 기존 native `fdtd_grid` 호출은 계속 호환됩니다.
:::


## Common scene adapter

현재 모듈은 JAX와 PyTorch 백엔드에서 살아 있는 재료 값을 미분합니다. 여기서는
실행 전체와 `DiffractionMonitor` 정규화를 거쳐 슬래브의 유전율을 미분합니다.

```python
import torch
from wavebit import Material, Units, fdtd

value = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
problem = fdtd.Problem(
    units=Units(length='um'),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4, .05, 2.), center=(0., 0., .15)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(fdtd.Structure('slab', shape=fdtd.SnappedSlab(z=(0., .3)),
                                   material=Material(epsilon=value)),)),
    sources=(fdtd.PlaneWave('light', plane=fdtd.Plane(z=.9),
                            pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=2e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=-.5), wavelengths=(1.,), source='light',
                                wave='transmitted', orders=(0, 0)),)))
result = fdtd.run(problem, backend='torch', numerics=fdtd.Numerics(
    grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(periods=30, source='light')))
derivative, = torch.autograd.grad(result['t'].efficiency.sum(), value)
```

여기서 `T`는 0.9849, 그 미분은 유전율 단위당 0.0470이며, 닫힌 형태의 해는 0.9837과 0.0435입니다
(현재 모듈, 2026-10-15. 과거 어댑터는 자체 격자에서 0.9866과 0.0405를 주었습니다).
JAX에서는 구체값으로 한 번 준비하고, 추적되는 값을 문제에 넣은 채
`prepared.run(problem)`을 미분하세요. 여기서 두 백엔드는
1e-15까지 일치합니다. `Open()` 면에 닿는 재료에는 선언이 하나 더 필요합니다. 그 PML은
구체값으로 한 번 만들어지므로, 반공간은
`PML(reference_media=...)`가 그 면의 프로파일을 고정할 때만 미분됩니다
([반공간 재료 미분](#half-space-material-derivatives) 참고).
무엇이 살아 있는지는 {ref}`실행 미분하기 <fdtd-model-differentiating>`에 나열되어 있습니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

이 절의 나머지는 과거 어댑터, 즉 `backend`를 명시한 `wavebit.compat.legacy.fdtd(...)`를
설명하며, 아래 예제 대부분은 여전히 이를 사용합니다.

이 어댑터는 NumPy, JAX, Torch에서 실행됩니다. JAX 결과는
등록된 pytree이므로, 완전한 `Result`를 반환하는 함수를 jit할 수 있습니다.
입사 `theta`/`phi`, 래스터 epsilon, 래스터 mu, 전기 Drude/Lorentz 매개변수, Jones 진폭은
백엔드 그래프를 유지합니다. 래스터 셀 경계는 정적입니다. 조화 평균은
각 필드 성분에 수직한 방향으로, 산술 평균은 접선 방향으로 적용합니다.
횡방향 평면 안에서는 epsilon과 mu 모두에 대해 수직 방향 조화 평균을
접선 방향 산술 평균보다 먼저 수행합니다. 이 순서 덕분에
x와 y를 바꾸면 같은 이산화 재료가 회전합니다. 이것은 래스터 모서리에서 여전히 대각,
분리형 근사이며, 완전한 텐서 균질화가 아닙니다.
극점 마스크는 수직 성분을 포함해 산술 채움 비율을 사용합니다. 이는
분산 경계면에 대한 근사이며, 일반적인 텐서 균질화기가 아닙니다.

지원하는 장면 입력에는 직사각형 1차원 래스터와 교차 래스터,
공유 RCWA 규약을 사용하는 직사각형/정사각형/원/타원 레시피(회전과 곡선 형상은
선택형 면적 적분으로 지원),
양의 실수 자성 반공간, 수직·비스듬한·conical 입사, Jones
편광, 전기 극점 레이블, 스칼라 또는 벡터 파장이 포함됩니다. 출력에는
`incident`, `s_inc`, `p_inc`, 회절 차수 메타데이터, 복소 r/t,
전력 R/T가 들어 있습니다. 선언된 TE/TM과 결합 Jones 출력 모두 각 출사
차수 고유의 s/p 좌표계를 사용하며, 진폭은 셀 중심 기준입니다. 반사는
빛이 들어오는 경계면을, 투과는 반대쪽 경계면을 기준으로 합니다.
이 경로에서는 `r_xyz`/`t_xyz`와 RCWA 필드 재구성을 사용할 수 없습니다.

전류 기저 시트 두 개와 균일한 입사 매질 기준 실행 두 번으로
입사 편광을 보정하고 반사 평면에서 입사 필드를 뺍니다.
네 번의 실행은 모두 같은 갱신 코어를 공유합니다. 이것은 기준 정규화이며,
TFSF 소스가 아닙니다. 추출은 Yee 평면 보간을 보정하고 이산
균질 전파를 사용해 위상 기준을 경계면으로 옮깁니다. 비스듬한
입사에서는 횡방향 파수 벡터가 파장 대역 전체에서 고정되며,
`return_diagnostics=True`는 그에 해당하는 각도를 포함합니다.

메시, 주기, 파장은 정적입니다. 직사각형 중심과 크기는 그 메시에서
바꿀 수 있습니다. [직사각형 미분](#transverse-rectangle-shape-derivatives)을 보세요. 내부 층
경계는 `mesh_thickness`로 전체 깊이를 고정한 채 바꿀 수 있으며, 선택형
`variable_depth=True`는 그 합계도 바꿉니다. [층 미분](#internal-layer-thickness-derivatives)과
[전체 깊이 미분](#changing-the-total-stack-depth)을 보세요.
반공간 epsilon/mu는 아래에서 설명하듯 명시적인 `mesh_indices`와 `truncation_order`로
그래디언트를 유지할 수 있습니다.
재료 최적화가 격자 크기를 바꾸지 못하도록 명시적인 `resolution`이 필요합니다.
여기서의 기본값은 `courant=.8`, `n_periods=40`, `n_pml=12`입니다. 최소 재료 값이나 극점 주파수가
요구하면 더 작은 Courant 비율을 고르세요.
시간 간격 안정성 한계는 native 코어가 검사합니다. 구체값의 잘못된
입력은 예외를 일으키고, 추적되는 잘못된 JAX 실행은 NaN 진폭과 효율을 반환하며,
선택적 진단에 `stable=False`가 들어갑니다. 새로운 구조, 특히 공진마다 해상도,
PML 폭, 패딩, 실행 시간을 세분화해 보세요.
정확한 스치듯 차수와 격자의 샘플링 한계를 넘는 차수는 거부합니다.
강한 소멸파 진폭을 먼 평면으로 외삽하는 것은 조건이 나쁩니다.

명시적 백엔드 어댑터는 현재 기울어진 격자나
임의 곡선 형상 미분을 받지 않으며, 지원하지 않는
명시적 백엔드 키워드는 오류를 일으킵니다. 암묵적인 백엔드 대체는 없습니다.
이 과거 장면 측정은 CPU를 사용했습니다. 현재 주기 API에는
[CUDA 슬래브 검사 두 개](../../validation/fdtd/native-cuda.md)가 있지만,
여기서 설명하는 더 넓은 장면군을 보증하지는 않습니다. 현재 일반
API에는 [별도의 범위가 정해진 CUDA 경로](../../validation/fdtd/general-cuda.md)가 있습니다.

테스트는 세분화에 따른 복소 슬래브 진폭을 Airy와, 자성
conical Jones 진폭과 비대칭 회절을 RCWA와, 파장 대역
투과를 저장된 Meep 공개 API 데이터와, JAX/Torch 재료 미분을
유한 차분과 비교합니다. 교차 분산 장면은 추가로 Torch/
NumPy 일치와 극점 세기 미분을 검사합니다. 이 장면 검사들은
native 코어의 Meep/FDTDX 필드·미분 기준을 보완합니다.
:::


### Transverse rectangle shape derivatives

현재 모듈은 `Numerics(geometry_ranges=...)` 상자를 예약한 스무딩 물체의 기하를 미분하지만,
`Box` 각도, 반공간 `SnappedSlab` 위에 바로 붙어 서 있거나 `Periodic()` 면 근처에 있는 스무딩 물체,
한 셀 안의 스무딩 표면 두 개는 미분하지 않으며, 타원은 없습니다. 이 절과 다음 네 절의 예제는 각각 이 한계 중 하나에 걸리므로
과거의 명시적 백엔드 경로를 사용합니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

명시적 백엔드는 직사각형과 정사각형에 대해 RCWA와 같은 층 딕셔너리와 `Material` 객체를
받습니다. 기본값 `shape_integration='separable'`은
축에 정렬된 형상을 받으며, 선택형 면적 적분은 회전을 지원합니다. 중심은
셀 중심에서 측정하고, 크기는 `period`의 단위로 나타낸 절대 길이입니다.
나중 형상이 주기 이미지를 포함해 앞선 형상 위에 칠해집니다. 좌표,
크기, 스칼라 양의 실수 epsilon/mu는 NumPy, JAX, Torch 값일 수 있습니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic (Rectangle regions on a Material substrate).
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def reflection(width):
    scene = wavebit.Stack.from_structure([
        dict(background=1., shapes=[
            dict(type='rectangle', lx=width, ly=.05, cx=.083, material=2.25)])
    ], [.2], top=1., bot=1.21, period=[1.3, .05])
    result = wavebit.fdtd(scene, wavebit.Incidence(wavelength=1.), backend='jax',
                          resolution=20, n_pml=20, n_periods=65, pulse_width=3.,
                          truncation_order=(1, 0), shape_smoothing=[.05, 0.],
                          align_layers=True, diffraction_method='directional')
    return result.s_inc.R.sum()

value, slope = jax.jit(jax.value_and_grad(reflection))(jnp.array(.43))
```

이 레시피에는 유한한 양의 직사각형 주기 두 개와
`subpixel=True`가 필요합니다. 분리형 직사각형에서는 각 크기가 양수이고
그 주기보다 크지 않아야 합니다.
분리형 `type` 값은 `rectangle`과 `square`이며, `l`과 생략된
중심은 공유 RCWA에서와 같은 의미를 유지합니다. 기본 분리형 방법은
정적인 `angle=0`을 요구합니다. 원과 [타원](#elliptical-shape-derivatives)에는 면적 방법이 필요합니다. 자성 매질에는 각 형상의 `Material(epsilon,
mu=...)`를 사용하세요. 별도의 래스터 `mu`, 분산 레이블, 극점은
이 레시피 경로에서 받지 않습니다. 래스터 입력은 기존의
분산 지원을 유지합니다. 살아 있는 층 두께를 함께 쓰려면 평소와 같은
`mesh_thickness`와, 선택한 고정 합계 또는 가변 깊이 제약도 필요합니다.

형상 개수, 타입, 칠하는 순서, 주기, Yee 메시는 정적으로 남습니다.
모든 형상이 정적인 `ly=period[1]`을 갖는 분리형 레시피는 한 행 격자
지름길을 사용합니다. 현재 전체 주기와 같은 값을 가진 텐서를 포함해 살아 있는 `ly`는
y 방향 전체 샘플링을 유지합니다. 기존의 배열 phi 규칙도 y 방향 전체 샘플링을 유지합니다.
지원하지 않는 입력은 예외를 일으키고, 추적되는 잘못된 기하는 NaN 산란과
진단의 `stable=False`를 만듭니다. 지원하지 않는 형상이나 각도를 조용히 버리지 않습니다.

기본값 `shape_smoothing=0`은 날카로운 주기 재료 영역을 각
Yee 셀에 대해 적분합니다. 미분은 구간별로 매끄러우며, 형상의 면이 셀 면을 지날 때
점프할 수 있습니다. 선택적인 정적 음이 아닌 스칼라 또는 `(x,y)` 쌍은 셀 적분 전에
칠한 층 전체의 공통 평행 이동을 평균합니다.
폭은 물리 길이 단위를 사용하며, 각 주기를 넘어서는 안 되고,
각각 독립적으로 0일 수 있습니다. 양의 폭은 고정된 칠하기 위상 구조 안에서 셀을 지날 때
C1 셀 비율을 줍니다. 또한 샘플링되는 재료 프로파일도 바꾸므로,
격자를 세분화할 때 폭을 줄이고 의도한 날카로운 기하를 따로 검사하세요.

한 형상이 다른 형상을 덮기 시작하는 접촉에서는 양의 평균을 사용하더라도
물리적인 한쪽 미분을 가질 수 있습니다. 이것은 서로 관계없는 투영 모서리가
같은 좌표를 공유하는 경우와 다르며, 그런 경우는 올바른 미분을 유지합니다.
크기 한계에서의 미분, 위상 구조 변화, 일반적인 2차 미분은
보장하지 않습니다. 수직 방향 조화 평균과 접선 방향 산술 평균은
모서리에 대한 일반적인 유효 텐서가 아니라 기존의 분리형 모서리 근사를
유지합니다.

진단은 분리형 방법에 대해 `geometry_kind`, `shape_integration`, `shape_smoothing`, 그리고 셀 중심 x/y 좌표의
살아 있는 `shape_edges`를 보고합니다. 이 모서리는 재료 분할을 기술하며,
반환되는 Yee `shape`, `spacing`, 샘플링 좌표는 고정됩니다.
[측정된 세분화와 독립 기준](../../validation/fdtd/fdtd-rectangle-gradients.md)을 보세요.
:::

### Rotated rectangle area integration

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

명시적 백엔드에서 `shape_integration='area'`를 설정하면 직사각형의 `angle`을
중심, 크기, 스칼라 재료와 함께 라디안 단위로 미분합니다.
같은 공유 RCWA 레시피를 사용합니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic (Rectangle angle, differentiable).
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def reflection(angle):
    stack = wavebit.Stack.from_structure([
        dict(background=1., shapes=[
            dict(lx=.3, ly=.18, angle=angle, material=2.25)])],
        [.2], top=1., bot=1.21, period=[.6, .6])
    result = wavebit.fdtd(stack, wavebit.Incidence(wavelength=1.),
                          backend='jax', resolution=16, n_pml=16, n_periods=60,
                          pulse_width=2.5, truncation_order=0, align_layers=True,
                          shape_integration='area', diffraction_method='directional')
    return result.s_inc.R.sum()

value, slope = jax.jit(jax.value_and_grad(reflection))(jnp.asarray(.37))
```

면적 경로는 나중 형상이 위에 칠해지는 여러 직사각형/정사각형,
또는 층마다 원/디스크/타원 하나를 받습니다. 빈 형상 목록은 균일한 층을 나타냅니다. 주기는 정적인 양수
두 개여야 합니다. 크기는 양수여야 합니다. 직사각형에서는 투영 범위
`abs(cos(angle))*lx + abs(sin(angle))*ly`와
`abs(sin(angle))*lx + abs(cos(angle))*ly`가 각 주기를 넘어서는 안 됩니다.
중심은 주기적으로 감쌉니다. `subpixel=True`가 필요합니다. epsilon/mu에는 형상의 `Material`을 사용하세요. 레시피 분산
레이블과 별도의 래스터 mu는 여전히 지원하지 않습니다. 회전과 기하는
JAX/Torch에서 살아 있을 수 있으며, 주기, 메시, 형상 개수는 정적으로 남습니다.

살아 있는 각도가 0이거나 직사각형이 현재 주기를 채우는 경우를 포함해
y 주기 전체 메시를 항상 유지합니다. 기하는 각 Yee 셀에 대해
정확하게 적분하며, 횡방향 재료 계수는 산술 면적 평균을 사용합니다.
이는 각도가 0일 때도 분리형 방법과 다릅니다.
완전한 회전 유효 텐서가 아니며, 경계면에서 2차 정확도를
보장하지 않습니다. 층 샘플링은 기존의 z 평균 규칙을 유지하며,
살아 있는 층 두께에는 `mesh_thickness`와 선택한 깊이 계약이 필요합니다.

기본값 `shape_smoothing=0`에서는 일치하는 평행한 셀 모서리와 형상 모서리가
서로 다른 한쪽 미분을 가질 수 있습니다. 횡방향 평행 이동을 평균하려면
위 계산에 `shape_smoothing=.05` 또는 정적인 `(x, y)` 쌍을 추가하세요. 각
폭은 유한하고 음이 아니며 그 주기보다 크지 않아야 합니다. 0인
성분은 그 축을 날카롭게 유지합니다. 양의 폭은 샘플링 격자나 구적 차수 설정 없이
정확한 쌍선형 다각형 모멘트를 사용합니다. 이렇게 하면 평균한 축에서
셀을 지날 때의 미분이 매끄러워지며, 일반적인 2차 미분은
제공하지 않습니다. 폭은 미분하는 동안 정적으로 유지됩니다.

평균은 재료 분포를 바꿉니다. 직사각형을 균일한 평행 이동 상자와
합성곱한 것을 나타냅니다. 이 점진 목표를 세분화할 때는 물리적 폭을 고정하거나,
날카로운 목표를 향해 폭을 줄일 때 생기는 추가 편향을 명시적으로 연구하세요.
주기 적분한 재료 면적은 보존됩니다. 회전 유효 텐서를 도입하지는 않습니다.
독립적인 기하 기준과 짝이 맞는 점진 프로파일 기준은
[평균 측정](../../validation/fdtd/fdtd-rotation-smoothing.md)을 보세요.

결과를 설계 그래디언트로 사용할 때는 격자 세분화와 독립적인 물리 기준을 확인하세요.
[회전 측정](../../validation/fdtd/fdtd-rotation-gradients.md)에
독립적인 면적 검사, 완전한 그래프 검사, 광학 검사가 기록되어 있습니다.
:::



### Multiple rotated rectangles

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

한 층에 여러 직사각형이나 정사각형을 두려면 같은 `shapes` 목록을
`shape_integration='area'`와 함께 사용하세요. 각 직사각형은 저마다 살아 있는
위치, 두 크기, 각도, 양의 스칼라 `Material(epsilon, mu=...)`를 가질 수 있습니다.
주기 셀 가장자리를 통해 겹치는 경우를 포함해 **나중 형상이 앞선 형상을 덮어씁니다**.
완전히 가려진 형상은 경계가 드러나기 전까지 재료나 기하에 대한
민감도가 없습니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic (overlapping Rectangle regions).
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def reflection(angle):
    stack = wavebit.Stack.from_structure([
        dict(background=1., shapes=[
            dict(lx=.17, ly=.1, cx=-.035, cy=-.012, angle=angle, material=1.8),
            dict(lx=.15, ly=.12, cx=.025, cy=.014, angle=-.27, material=2.25)])
    ], [.2], top=1., bot=1.21, period=[.4, .3])
    result = wavebit.fdtd(stack, wavebit.Incidence(wavelength=1.), backend='jax',
                          resolution=15, n_pml=15, n_periods=60, pulse_width=2.5,
                          truncation_order=0, shape_integration='area',
                          shape_smoothing=.035, diffraction_method='directional')
    return jnp.real(result.s_inc.r_s[0, 0])

value, slope = jax.jit(jax.value_and_grad(reflection))(jnp.asarray(.31))
```

형상 개수, 형상 타입, 칠하는 순서는 정적으로 남습니다. 위치와
각도는 한 JAX 추적 안에서 겹침 구성을 넘나들 수 있으며, 호스트의
잘라내기 계획은 필요하지 않습니다. 이것은 **구간별 1차 미분**을 유지합니다.
실제 형상 접촉과 일치하는 경계는 양의 `shape_smoothing`을 사용하더라도 서로 다른 한쪽
형상 미분을 가질 수 있습니다. 스무딩은 이미 칠한 재료 패턴 전체를
평행 이동하는 것이며, 두 형상 사이의 상대적인 접촉을 부드럽게 하지 않습니다.

각 직사각형의 투영 범위는 주기 안에 들어가야 합니다. 원/디스크/타원을
포함하는 모든 층은 여전히 그 형상 하나만 담아야 하며, 곡선
형상을 다른 면적 형상과 섞는 것은 거부합니다. 층 두께, 입사, 스칼라
재료 제약은 바뀌지 않습니다.

경계 교차점이 각 쪽의 재료를 선택합니다. 정확한 직선
셀 적분은 보조 래스터나 쌍별 면적 빼기 없이 칠한 재료를 복원합니다.
평행 이동 평균은 분할된 선분에 3차 피적분 함수를 남기며, 이를 가우스 노드 두 개로
산술적으로 정확히 평가합니다. 반올림 오차는
여전히 부동소수점 정확도를 제한합니다. 작업량과 저장 공간은
경계 사건의 수에 따라 늘어납니다. 형상 개수에 엄격한 한계가 없다는 것이 큰 배치에 대한
성능을 보장하는 것은 아닙니다.

[칠한 형상 측정](../../validation/fdtd/fdtd-painted-shapes.md)은 기하 검사와
완전한 그래프 검사를 점진 광학 세분화와 분리합니다. 횡방향 재료
계수는 여전히 산술 평균이며, 이 확장은 완전한
경계면 유효 텐서를 도입하지 않습니다.
:::

### Circular shape derivatives

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`shape_integration='area'`와 명시적 백엔드와 함께 `type='circle'`(별칭 `'disk'`)을
사용하세요. 공유 RCWA 규약은 `lx` 또는 `l`을 반지름이 아니라 **지름**으로 사용하며,
`ly`는 거부합니다. 기본 중심은 주기 중심을 기준으로 `(cx, cy)=(0, 0)`입니다.
중심은 주기적으로 감쌉니다. 유한한 `angle`은 원에 아무 효과가
없으며, 그 미분은 0입니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic (Circle regions on a Material substrate).
import jax
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def transmission(diameter):
    stack = wavebit.Stack.from_structure([
        dict(background=1., shapes=[dict(type='circle', lx=diameter,
                                         cx=.071, cy=-.047, material=2.25)])
    ], [.2], top=1., bot=1.21, period=[.6, .6])
    result = wavebit.fdtd(stack, wavebit.Incidence(wavelength=1.), backend='jax',
                          resolution=20, n_pml=20, n_periods=60, pulse_width=2.5,
                          truncation_order=0, shape_integration='area',
                          shape_smoothing=.05, diffraction_method='directional')
    return result.s_inc.t_s[0, 0]

t = jax.jit(transmission)(.3)
dt_ddiameter = jax.jit(jax.jacfwd(transmission))(.3)
```

지름은 양수여야 하며 직사각형 주기 **어느 쪽**보다도 크지 않아야 합니다.
층마다 형상 하나라는 제한은 원에도 적용되므로, 원과 직사각형은
서로 다른 층에 놓을 수는 있지만 한 면적 층에 함께 칠할 수는 없습니다.
두 주기, 형상 타입·개수, 스무딩 폭, 메시는 정적으로 남습니다.
`subpixel=True`가 필요합니다. 중심, 지름, 양의 스칼라 형상
`Material` epsilon/mu는 아래 입사와 기존 메시 계약 아래의 살아 있는 층 두께를 포함해
JAX/Torch 그래디언트를 유지합니다.

`shape_smoothing=0`에서 디스크/셀 면적은 해석적 원형 활꼴 적분을 사용합니다.
경계를 다각형이나 보조 래스터로 근사하지 않습니다. 접하는
접촉에서 1차 미분은 유한하지만, 2차 미분은 특이할 수 있습니다.
양의 스무딩은 직사각형과 같은 물리적 상자 평행 이동의 의미를
가집니다. 디스크 적분은 정확한 평평한 띠와, 각도 방향 호 띠에 대한 고정된 16점
가우스 규칙을 사용하며, 그 피적분 함수는 낮은 차수의
삼각 다항식입니다. 사용자가 설정하는 구적 차수는 없습니다. 이렇게 하면
커널 조각이 매우 좁을 때 전역 모멘트 사이의 상쇄를 피할 수 있습니다.
폭이 0, 전체 주기, 셀 폭인 경우를 지원합니다.

횡방향 재료 계수는 여전히 산술 면적 평균을 사용합니다. 이것은
완전한 유효 텐서가 아니므로, 정확한 셀 기하가 날카로운 경계면의
광학 그래디언트 정확도를 확립하지는 않습니다. 해상도와 짝이 맞는
물리 기준을 확인하세요. [디스크 측정](../../validation/fdtd/fdtd-circle-gradients.md)은
기하, 완전한 그래프 미분, 프로파일 편향, 독립 광학 검사를 분리합니다.
:::

### Elliptical shape derivatives

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

명시적 공통 경로는 `shape_integration='area'`와 함께
`type='ellipse'`를 받습니다. `lx`와 `ly`는 회전 전 **두 주축의 전체 길이**이며,
주기와 같은 단위입니다. `l`은
두 길이에 공통으로 쓰는 대체값입니다. `angle`은 타원을 라디안 단위로 반시계 방향으로
회전하며, 현재 값이 0이어도 살아 있을 수 있습니다. 중심은
주기 중심 기준의 공통 규약을 따르며 주기적으로 감쌉니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic Ellipse.
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def transmission(parameters):
    angle, width, height = parameters
    stack = wavebit.Stack.from_structure([
        dict(background=1., shapes=[dict(type='ellipse', angle=angle,
             lx=width, ly=height, cx=.071, cy=-.047, material=2.25)])
    ], [.2], top=1., bot=1.21, period=[.6, .6])
    result = wavebit.fdtd(stack, wavebit.Incidence(wavelength=1.), backend='jax',
                          resolution=20, n_pml=20, n_periods=60, pulse_width=2.5,
                          truncation_order=0, shape_integration='area',
                          shape_smoothing=.05, diffraction_method='directional')
    return result.s_inc.t_s[0, 0]

parameters = jnp.array([.37, .34, .2])
t = jax.jit(transmission)(parameters)
dt_dshape = jax.jit(jax.jacfwd(transmission))(parameters)
```

각 축 길이는 양수여야 합니다. 투영된 전체 폭
`sqrt((lx*cos(angle))**2 + (ly*sin(angle))**2)`와
`sqrt((lx*sin(angle))**2 + (ly*cos(angle))**2)`는 x와 y 주기 안에 들어가야 합니다.
따라서 회전한 타원이 들어맞는다면 주축이 회전 전의 주기를 넘을 수도
있습니다. 형상 개수·타입, 두 직사각형 주기, 메시,
스무딩 폭은 정적으로 남습니다. 타원 층에는 형상 하나와 `subpixel=True`가 필요합니다. 중심, 두 길이, 각도, 양의 스칼라 형상
`Material` epsilon/mu는 JAX/Torch 그래디언트를 유지합니다. 층 두께와 아래
입사 계약도 적용됩니다.

날카로운 타원 셀과 평균한 타원 셀 모두 고정된 16점 가우스 규칙으로
각도 방향 띠 적분을 사용합니다. 직교 좌표 셀 면이 해석적
타원과 교차하며, 그 경계는 다각형으로 만들거나 보조 래스터에서 샘플링하지 않습니다.
부동소수점 반올림 단위 몇 개 이내의 투영 접점은 조건이 나쁜 역사인 미분을
피하기 위해 극한 끝점을 사용합니다. 일반적인
2차 미분은 제공하지 않습니다.

`shape_smoothing`은 **실험실 x/y 축**을 따라 평행 이동을 평균합니다.
타원을 회전해도 이 평균 상자는 회전하지 않습니다. 0, 좁은 폭, 셀 전체,
주기 전체 폭은 원에서와 같은 의미를 가집니다. 횡방향
재료 계수는 산술 평균이며, 완전한 곡면 경계면
유효 텐서가 아닙니다. 실제로 최적화하는 재료 프로파일에 대해 독립 기준과 해상도 수렴을
확인하세요. [타원 측정](../../validation/fdtd/fdtd-ellipse-gradients.md)은
적분 정확도, 완전한 그래프, 점진 광학 세분화를 분리합니다.
:::

### Incidence-angle derivatives

현재 모듈에서 `PlaneWave`의 `theta`와 `phi`는 정적입니다. 이 값들이 격자와
시간 루프를 정하며, 추적되는 값은 거부합니다. 이 예제들은 과거 경로를 사용합니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

명시적 `backend`를 사용하면 `theta`와 `phi`는 실수 스칼라 NumPy, JAX 또는
Torch 배열일 수 있습니다. 둘 중 하나라도 배열이면, 상수 0차원 배열을 포함해
항상 명시적인 `truncation_order`를 주세요. 이렇게 하면 각도가 바뀌는 동안
출력 모양이 고정됩니다. Python 또는 NumPy 스칼라 숫자는 계속
자동 차수 선택을 지원합니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic (theta differentiates).
import torch
from wavebit.compat import legacy as wavebit

theta = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
result = wavebit.fdtd(
    structure=[[[2.25]]], thickness=[.2], top=1., bot=1., period=.2,
    wavelength=1., theta=theta, tetm='tm', backend='torch',
    resolution=20, n_periods=40, truncation_order=0)
dT_dtheta, = torch.autograd.grad(result.incident.T.sum(), theta)
```

미분은 라디안당 값이며, 기하, 셀 크기, 시간 간격, 갱신
횟수, 소스 위치, PML, 회절 창을 고정한 상태의 값입니다. `Incidence(theta=theta, phi=phi)`도
같은 그래프를 유지합니다. 경로에는 소스 시트 위상, Bloch 경계,
공간 푸리에 추출, 보간, 경계면까지의 전파,
출사 s/p 좌표계, 전력 인자, 두 입사 기준 실행이 모두 포함됩니다. 또한
정규화된 주파수 모니터, 잘라낸 시간 기록, 접두 구간
수렴 진단도 포함됩니다. 재료와 전기 극점 매개변수를 두 각도와 함께
미분할 수 있으며, 독립 횡방향 해상도,
정적 z 비균일화, 층 정렬, 소스 노치, 아래 입사도 계속
사용할 수 있습니다.

방위각 미분에는 `polarization=(a_s, a_p)` 또는 `psi`/`chi`를 사용하세요. 배열
`phi`는 값이 0인 배열을 포함해 `tetm`과 함께 쓸 수 없습니다.
스칼라 TE/TM 규약은 x-z 평면을 고정하므로 변하는
방위각을 따라갈 수 없습니다. 순수 s와 p는 각각 `polarization=(1., 0.)`과 `(0., 1.)`입니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic PlaneWave(theta=, phi=, polarization=(a_s, a_p)).
import torch
from wavebit.compat import legacy as wavebit

angles = torch.tensor([.3, .4], dtype=torch.float64, requires_grad=True)
result = wavebit.fdtd(
    structure=[[[1.3, 2.2], [1.8, 1.1]]], thickness=[.2],
    top=1., bot=1., period=[.3, .4], wavelength=1.,
    theta=angles[0], phi=angles[1], polarization=(1., .6j),
    backend='torch', resolution=20, n_periods=40, truncation_order=0)
dT_dangles, = torch.autograd.grad(result.incident.T.sum(), angles)
```

배열 방위각은 0일 때를 포함해 항상 표현된 y 주기 전체를 사용합니다.
선언된 `Ly`에 대해 `Ny = max(2, ceil(Ly Ry))`입니다. 일반 스칼라 `phi=0`을 쓰는
한 행 래스터는 기존의 y 두 셀 지름길을 유지합니다. 따라서
그 스칼라를 배열로 바꾸면 격자와 시간 간격이 한 번 바뀔 수 있으며, 이후
배열 값을 바꿀 때는 고정된 채로 유지됩니다. 유한 차분 비교에는
일관되게 배열을 사용하세요. NumPy 벡터 인덱싱은 스칼라 숫자를 만들므로,
배열 각도 계산과 비교할 때는 그런 phi를 `np.asarray(phi)`로 감싸세요.
x 주기 하나만 있는 경우에도 보조
y 주기 `2/Ry`(두 셀)는 그대로 있습니다. 기존의 y 불변 모니터·기록 단면과
y 단위 길이당 전력 규약은 바뀌지 않습니다.

여러 파장에는 횡방향 Bloch 파수 벡터가 하나만 있으며, 요청한
파장 범위의 중간점 `lambda_c`로 정합니다. 정적인
입사 반공간에서

`theta(lambda) = asin((lambda/lambda_c) * sin(theta))`입니다.

따라서 각도 미분은 이 곡선을 따라 광대역 실험 전체를 바꿉니다.
각 파장에서의 별도 고정 각도 미분이 아닙니다.
진단은 파장에 따라 달라지는 극각에 대해 `angles`에 백엔드 배열을,
phi에 대해 스칼라 `azimuth`를 반환합니다. phi를 바꾸면 공유
횡방향 벡터가 회전하며, 그 크기와 극각은 고정됩니다.
고정 파수 벡터 규약은
[Meep 공개 FAQ](https://meep.readthedocs.io/en/latest/FAQ/)에도 설명되어 있습니다.

theta는 유한한 실수로 `0 <= theta < pi/2`를 만족해야 하며, phi는 음수를 포함해
라디안 단위의 어떤 유한한 실수 값도 받습니다. 입사파는 대역 전체에서
전파해야 합니다. 정확한 물리적 스치듯 입사, 보간 나이퀴스트
영점, 방향성 방법의 이산 특이점은 여전히 지원하지 않습니다.
구체값의 잘못된 각도는 시간 전진 전에 예외를 일으키고, JAX JIT 아래의
추상적인 잘못된 각도는 `stable=False`와 NaN 산란, 정규화된 모니터, 시간
기록을 만듭니다. 불리언 전파 마스크는 분류일 뿐, 미분 가능한
관측값이 아닙니다. 수직 입사에서 theta 미분은 고정된 phi 반직선을 따르고,
phi 미분은 지정된 s/p 편광 좌표계를 회전합니다.
횡방향 좌표계는 원점에서 유일한 임의 방향을 갖지 않습니다.
문턱이나 수명이 긴 공진에서는 민감도를 해석하기 전에 공간과
시간 수렴을 확립하세요. 검사한 사례와 한계는
[theta](../../validation/fdtd/fdtd-scene-angles.md)와 [방위각](../../validation/fdtd/fdtd-scene-azimuth.md) 검증
노트를 보세요.
:::


### Independent transverse resolution

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

명시적 백엔드 어댑터는 단위 길이당 셀 수로 `transverse_resolution=R` 또는
`transverse_resolution=(Rx, Ry)`를 받습니다. 이 양의 유한한
정적 값은 x/y 샘플링을 정하고, `resolution`은 계속 z 메시,
층 정렬, PML 셀 두께를 정합니다. `None`이면 두 축 모두 `resolution`을 사용합니다.
쌍은 **x, y 순서**이며, native `spacing`과 격자 `shape`는 z, y, x
순서로 남습니다. 재료와 극점 그래디언트는 평소의 의미를 유지하며, 메시 크기와
이 해상도 제어는 미분할 수 없습니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic Numerics(transverse_spacing=...).
from wavebit.compat import legacy as wavebit

result, diagnostic = wavebit.fdtd(
    structure=[[[2.25, 1.]]], thickness=[.2], top=1., bot=1., period=.8,
    wavelength=1., tetm='te', backend='numpy', resolution=40,
    transverse_resolution=(80, 40), truncation_order=1,
    return_diagnostics=True)
print(diagnostic['transverse_resolution'])  # (80.0, 40.0), requested x/y targets
print(diagnostic['spacing'])                # actual dz, dy, dx
```

표현된 각 횡방향 주기는 목표 셀 수를 올림해 최소 두 개의 셀로 나뉩니다.
`Nx = max(2, ceil(Lx Rx))`이며, 완전히 표현된 y 축도 마찬가지입니다.
일반 스칼라 `phi=0`의 한 행 래스터는 y 셀 두 개를 유지합니다. y
주기를 주면 그 폭은 `Ly/2`이며, `Ry`를 바꿔도 이 축약된 축은 더 나뉘지 않습니다.
배열 phi는 0일 때도 전체 주기 샘플링을 사용합니다. x 주기가 하나뿐이면
보조 y 주기는 `2/Ry`이므로
두 셀의 폭은 `1/Ry`이며, y 불변 구조에 대한 conical 조명도 마찬가지입니다.
진단은 목표값이 모든 축에서 똑같이 달성되었다고 가정하지 않고
실제 간격을 보여 줍니다.

횡방향 샘플링을 바꿔도 z 노드, 층 면, 선택한
소스 시트, PML 면, 회절 평면은 유지됩니다. CFL 시간 간격은
**실제 간격 세 개 모두**로 다시 계산하므로, z 셀 수가 그대로여도
갱신 횟수가 늘 수 있습니다. 수치 주파수로 지정한 소스 영점은
다른 이산 차단에 맞춰 자동으로 다시 조정되지 않습니다.
나이퀴스트와 스치듯 검사는 새 메시를 사용하며, 여전히 분해되지 않은 유지
차수를 거부합니다. 전류 기저·기준 실행 네 번 모두 같은 메시를 사용합니다.

독립 세분화는 한 방향이 정확도를 제한할 때 유용하지만,
작업량이 줄거나 오차가 단조롭게 작아지는 것을 보장하지는 않습니다.
[횡방향 세분화 연구](../../validation/fdtd/fdtd-transverse-resolution.md)는
조화 격자 정확도를 유한 시간·유한 경계 차이와 분리합니다.
:::


### Source position independent of padding

현재 모듈에서 입사 시트는 문제가 놓은 위치에 있습니다.
`PlaneWave(plane=fdtd.Plane(z=...))`는 그 좌표에 가장 가까운 노드 평면에 시트를 놓고, 각
`DiffractionMonitor(plane=...)`는 측정 평면을 놓으며, `Domain`이 패딩을 정하므로, 셋 중
어느 것도 다른 것을 움직이지 않습니다. 두께 0.213 µm인 막과, 그 윗면에서 1 µm 위에 있는 시트의 예입니다.

```python
from wavebit import Material, Units, fdtd

thickness, distance = .213, 1.                     # um; the sheet 1 um above the top face
problem = fdtd.Problem(
    units=Units(length='um'),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 2.), center=(0., 0., .5)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        # A smoothed Box keeps both faces where they are declared; a SnappedSlab would snap them.
        structures=(fdtd.Structure('film', shape=fdtd.Box(size=(.2, .2, thickness),
                                                          center=(0., 0., thickness / 2)),
                                   material=Material(epsilon=2.25)),)),
    sources=(fdtd.PlaneWave('light', plane=fdtd.Plane(z=thickness + distance),
                            pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=2.25e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=thickness + .2), wavelengths=(1.,),
                                source='light', wave='reflected',
                                reference_plane=thickness, incident_plane=thickness),
        fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=-.2), wavelengths=(1.,),
                                source='light', wave='transmitted',
                                reference_plane=0., incident_plane=thickness))))
result = fdtd.run(problem, numerics=fdtd.Numerics(
    grid=fdtd.UniformGrid(spacing=.025), duration=fdtd.Duration(periods=50, source='light')))
print(f"{problem.sources[0].plane.position:.4f}")     # requested sheet position
print(f"{result.source_references['light']['plane_E']:.4f}")  # the node plane E sits on
```

```
1.2130
1.2250
```

이 값들은 현재 모듈(2026-10-15)에서 얻었습니다. `frequency_sigma_hz=2.25e13`은
과거의 `pulse_width=3.`에 해당하며([Gaussian source width](#gaussian-source-width) 참고), 막의
`T`는 0.8759로, 과거 경로의 0.8771, 닫힌 형태의 0.8752와 비교됩니다. Box는
과거 경로의 `align_layers=True`가 노드에 놓았던 0.213 µm 면을 유지합니다. Box는
스무딩 물체가 모두 일으키는 `SubpixelAccuracyWarning`을 일으킵니다.

과거 명시적 백엔드 경로는 같은 배치를 적층 면으로부터의 거리로
표기합니다. `source_distance`는 선택적으로 입사 전류 시트를
입사 경계면에서 양의 정적 거리만큼 떨어진 곳에 길이 단위로 놓습니다. `None`이면
원래 소스 위치를 유지합니다. 메시, PML, 층 경계, 추출 평면을 바꾸지 않고
가장 가까운 z 노드를 선택합니다. 위
입사에서 목표는 `z=total_thickness+source_distance`이고, 아래 입사에서는
`z=-source_distance`입니다. `variable_depth=True`에서는 이 배치에 공칭
메시 깊이를 사용하며, 물리적 면이 움직여도 소스는 고정됩니다.
`align_layers=True`와 함께 `source_distance=1., diffraction_distance=.2`를 쓰면 과거
경로는 위 막에 대해 시트를 1.213의 노드에 놓았습니다.

선택한 노드는 입사 반공간 안에 있어야 하며, PML 안쪽 면에서 최소 두 셀을 지나고,
반사 평면보다 최소 세 셀 간격 앞에 있어야 합니다. 잘못되었거나, 유한하지 않거나, 양수가 아니거나, 배치할 수 없는 거리는
시간 전진 전에 예외를 일으킵니다. 사용자 주파수 모니터는 여전히 전류
시트를 가로질러 보간할 수 없습니다. 비균일 메시에서 거리는 셀 수가 아니라 물리 거리이며,
보고되는 노드는 국소 스냅 거리만큼 요청과 다를 수 있습니다.

이로써 서로 다른 두 실험이 가능해집니다. `source_distance`와 `diffraction_distance`를
모두 고정한 채 `pad_scale`을 바꾸거나, 패딩을 고정한 채
소스를 옮기는 것입니다. 전자는 경계 여유 거리를, 후자는
조명 경로를 바꿉니다. 비균일 메시를 바꾸면 가장 가까운 노드가 달라질 수 있으므로
실제 좌표를 확인하세요. 각 파장은 여전히 자체 균일 매질
기준 보정을 사용합니다. 소스 배치는 유한 시간 과도 현상과
경계 반사를 바꾸므로, 보정된 진폭이 같으려면 모든 격자에서 정확히 불변이라고
가정하는 대신 수렴이 필요합니다.

이 옵션은 공유 NumPy/JAX/Torch 장면 엔진, 아래 입사,
비균일화, 소스 노치, 기록, 재료 그래디언트를 지원합니다. 거리 자체는
정적이며 형상 미분이 없습니다. 레거시 장면 경로에는 이에 해당하는
키워드가 없습니다. 소스를 옮기는 것과 경계를 옮기는 것을 구분해야 하는
차단 근처 대조 실험은 [소스 여유 거리 측정](../../validation/fdtd/fdtd-source-clearance.md)을
보세요.


### Diffraction planes and cutoff diagnostics

현재 모듈에서 각 측정 평면은 문제가 선언한 `DiffractionMonitor(plane=...)`이며,
`reference_plane`과 `incident_plane`이 그 위상 기준을 옮깁니다(위
예제처럼). 각 결과는 모든 차수의 Yee `k_normal`(포트의 법선 축 방향
파수 벡터 성분)과 `k_parallel`을 담습니다.
차단 분류는 없으며, `wavebit.fdtd.periodic`은 차수마다 하나씩
`result.diagnostics['diffraction']`(`propagating`, `cutoff_wavelength`)에 보고합니다. 아래 예제는
과거 경로를 사용합니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`diffraction_distance`는 반사 및 투과 기준 경계면에서 추출 평면까지의 거리를
제어합니다. 두 갈래 모두에 쓰는 양의 정적 스칼라, 또는 길이 단위의
`(reflection, transmission)` 쌍을 받습니다. 기본값
`None`은 원래 배치를 유지합니다. 각 목표는 가장 가까운
전기장 z 노드에 스냅되며, 이 설정은 메시, 소스, PML 배치를 바꾸지 않습니다.
반사 평면은 전류 시트와 적층 사이에 있어야 하고, 투과
평면은 PML보다 앞에 있어야 합니다. 어느 평면도 혼합 경계면 셀에 닿아서는 안 됩니다.
배치할 수 없는 요청은 시뮬레이션을 조용히 확장하지 않고 예외를 일으킵니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic result.diagnostics['diffraction'].
import numpy as np
from wavebit.compat import legacy as wavebit

result, diag = wavebit.fdtd(
    structure=np.full((1, 1, 1), 2.25), thickness=[.213],
    top=1., bot=1., period=.1, wavelength=1., tetm='tm', theta=.25,
    backend='numpy', resolution=80, n_pml=24, align_layers=True,
    diffraction_distance=(.2, .3), return_diagnostics=True,
)
print(diag['diffraction']['reflection_monitor_z'])
print(diag['diffraction']['cutoff_resolved'])
```

`diag['diffraction']`은 아래 입사를 포함해 실제로 놓인 `reflection_monitor_z`와
`transmission_monitor_z`를 원래 장면 좌표로 보고합니다.
이 쌍은 항상 입사 기준입니다. 복소 r/t는 이 측정 평면이 아니라
경계면을 기준으로 남습니다. 평면을 옮기면 유한 격자의 기준 보정이
달라질 수 있고, 따라서 r/t뿐 아니라 보정된 필드도 달라질 수 있습니다.
공간, PML, 실행 시간 세분화 아래에서 위치를 비교하세요.

진단에는 또한 `(order_y, order_x)` 배열 `order_x`, `order_y`,
`kx`, `ky`가 들어 있습니다. 뒤의 둘은 대역 전체에서 고정된 횡방향 파수 벡터이며 단위는 길이당 라디안입니다.
그 `reflection`과 `transmission` 매핑은 각
손실 없는 반공간을 다음 배열로 기술합니다. 그 모양은 고정되어 있으며, 수치
값은 살아 있는 각도·재료를 따르고, 불리언 마스크는
현재 매개변수를 분류합니다.

| 키 | 모양 | 의미 |
| --- | --- | --- |
| `q` | `(wavelength, order_y, order_x)` | 출사 방향의 정규화된 종방향 파수 벡터입니다. 전파하면 양의 실수, 감쇠하면 양의 허수입니다. 아래 입사에서 부호 있는 직교 좌표 kz가 아닙니다. |
| `propagating` | 같음 | 연속체의 열린 채널 마스크입니다. 닫힌 채널의 출사 R/T 전력은 정확히 0입니다. |
| `yee_cutoff_parameter` | 같음 | 아래에서 정의하는 이산 하한 차단 매개변수입니다. |
| `cutoff_resolved` | 같음 | 연속체와 이산 하한 차단의 부호가 일치하는지 여부입니다. |
| `p_projection_gain` | 같음 | `n/abs(q)`, 즉 접선 E에서 p 진폭으로의 투영 이득입니다. 값이 크면 추출 오차가 증폭됩니다. |
| `cutoff_wavelength` | `(order_y, order_x)` | 고정된 횡방향 k에 대해 `2*pi*n/hypot(kx,ky)`, 횡방향 k가 0이면 무한대입니다. |

여기서 `n=sqrt(epsilon*mu)`, `k0=2*pi/wavelength`이며, Yee 차단 매개변수는 다음과 같습니다.

```text
(2/k0)^2 * ((n*sin(k0*dt/2)/dt)^2
            - (sin(kx*dx/2)/dx)^2 - (sin(ky*dy/2)/dy)^2).
```

이 값의 양의 부호는 종방향 통과 대역의 아래쪽 경계일 뿐입니다. 종방향 나이퀴스트 한계,
메시 반사, PML 오차를 배제하지 않습니다.
바깥쪽 스칼라 `diag['diffraction']['cutoff_resolved']`는 요청한 모든 파장에서 모든
유지 차수가 두 반공간 모두에서 일치할 때만 참입니다. 이 값은
시간 수렴 플래그와 독립적이며 정확도 증명서가 아닙니다.
반올림 오차 크기만큼 떨어진 경우를 포함해 정확한 스치듯 입사는 시간 전진 전에
거부합니다. 차수를 조용히 버리지 않습니다. 거친 격자에서 잘못 열린
채널과, 차단 분류가 올바른데도 분해되지 않은 TM 응답의 예는
[레일리 근처 측정](../../validation/fdtd/fdtd-rayleigh.md)을 보세요.
:::


### Directional diffraction extraction

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

기본값 `diffraction_method='electric'`은 기존 추출 방식을 유지합니다.
명시적 백엔드에서 `diffraction_method='directional'`은 E와 H를 모두 사용해
각 측정 평면에서 적층을 향하는 전파와 적층에서 멀어지는 전파를 분리합니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic Numerics(extraction_method='directional').
import numpy as np
from wavebit.compat import legacy as wavebit

result, diag = wavebit.fdtd(
    structure=np.full((1, 1, 1), 2.25), thickness=[.2],
    top=1., bot=1., period=.1, wavelength=1., tetm='tm',
    backend='numpy', resolution=40, n_pml=12,
    diffraction_method='directional', diffraction_distance=.2,
    return_diagnostics=True, convergence_window=100,
)
print(diag['diffraction']['incoming']['p_inc']['t_p'])
```

이 옵션은 전기·자기 공간 보간과 각각의 DFT 타임스탬프를 포함한
**이산 Yee 어드미턴스**를 사용합니다. conical
입사, 자성 반공간, 결합된 차수 창, 소멸파 가지를 지원합니다.
각 평면은 바로 위와 아래의 z 셀 폭이 같아야 하며,
비균일화 전이 구간 안의 평면은 거부합니다. 다른 곳의 비균일화와 정렬된
층 내부는 계속 사용할 수 있지만, 그 전파 오차는 여전히
세분화가 필요합니다. 이산 차단과 종방향·시간 나이퀴스트 특이점은
시간 전진 전에 거부합니다. 물리적 스치듯 입사 거부도 적용됩니다.

반사는 실제 필드의 후방 성분에서 균일 매질
기준의 후방 성분을 뺀 값을 사용합니다. 투과는 실제 필드의 전방 성분을 사용합니다. 입사 정규화는
**전방** 기준 필드를 사용합니다. 네 번의 native 계산, 재료 그래디언트,
경계면 기준 평면은 그대로입니다. 시간 기록은 반송파에서의 전방
기준 보정을 사용합니다. 보정된 수렴 진단은
`dft_subsample`이 1을 넘는 경우를 포함해 전체 DFT와 접두 구간 DFT에 같은 방향성 추출을 사용합니다.

`diag['diffraction']['method']`는 방법을 식별합니다. 방향성 추출에서
`diag['diffraction']['incoming']`에는 `incident`, `s_inc`, `p_inc` 매핑이 있으며,
각각 모양이 `(wavelength, order_y, order_x)`인 복소 `r_s`, `r_p`, `t_s`, `t_p` 배열을
담습니다.

| 키 | 의미 |
| --- | --- |
| `r_s`, `r_p` | 반사 평면에서 실제 필드의 전방 성분에서 전방 기준을 뺀 값이며, 입사 경계면을 기준으로 합니다. 미리 정한 기준 조명은 제외합니다. |
| `t_s`, `t_p` | 투과 평면에서 실제 필드의 후방 성분이며, 출사 경계면을 기준으로 합니다. |

접두어는 측정 **갈래**를 가리키며, 이 성분들은 해당 출사 r/t와
반대 방향으로 진행합니다. p 부호는 각 파동 고유의 전파 방향을 따릅니다.
아래 입사에서도 갈래는 입사 기준으로 남습니다. 값은 단위 입사 E 진폭이며, 반사 전력 비율이 아닙니다.
소멸파 항목을 전력으로 해석해서는 안 됩니다.

방향 분리는 들어오는 파동이 출사 계수에 직접 더해지는 것을 막습니다.
그러나 되돌아오는 파동이 조명과 구조 응답을 바꾸는 효과를 되돌릴 수는
없습니다. 유한 DFT에는 필드가 아직 진동하는 동안 끝점 항도 있으며,
겉보기 입사 성분에는 그 오차가 포함될 수 있습니다.
진폭 꼬리, 실행 시간, 격자, 패딩, PML을 각각 독립적으로 비교하세요. 정확한 모드 검증과
여전히 시간 수렴에 실패하는 레일리 근처 예는
[방향성 측정](../../validation/fdtd/fdtd-directional.md)을 보세요.
:::


### Gaussian source width

현재 모듈에서 모든 소스는 선언된 `GaussianPulse(wavelength,
frequency_sigma_hz)`를 가지며, 이는 진폭 스펙트럼의 표준편차(Hz)입니다. 아래의 과거
`pulse_width=w`는 `w`를 길이로 할 때 `frequency_sigma_hz = c / (sqrt(2) * pi * w)`에 해당합니다.
`w = 3` µm이면 2.25e13 Hz입니다. 사용 가능 대역은 그 스펙트럼이 펄스의
`cutoff`(기본값 1e-3)까지 떨어지는 곳에서 끝나며, 그 밖의 `DiffractionMonitor` 파장은 NaN을 돌려줍니다.
이 절의 나머지는 과거 경로의 `pulse_width`를 설명합니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

명시적 백엔드 장면 어댑터는 c=1 시간 단위(길이와 같은 수치 단위)의 양의 정적 스칼라
`pulse_width`를 받습니다. 이는 가우시안
포락선의 1/e **진폭 반폭**이며, 주파수 대역폭이나 FWHM이 아닙니다.
생략하거나 `None`을 넘기면 중심 진공 파장
`centre = (min(wavelengths)+max(wavelengths))/2`를 사용하며, 원래 펄스를 유지합니다.

샘플 시각 `t=n*dt`에서 실제 실행과 기준 실행 모두 다음을 사용합니다.

```text
pulse(t) = exp(-((t-5*pulse_width)/pulse_width)^2) * exp(-2*pi*i*t/centre).
```

폭이 클수록 주파수 대역이 좁아집니다. 이렇게 하면 비스듬한 입사에서 천천히 감쇠하는
스치는 파동을 줄일 수 있지만, 중심에서 먼 파장에서는 소스가 약해지기도
합니다. 폭을 늘릴 때는 전체 실행을 더 길게 하세요.

```python
# Earlier interface; current: wavebit.fdtd.periodic diagnostics['pulse'] and Numerics(decay_check=DecayCheck(...)).
import numpy as np
from wavebit.compat import legacy as wavebit

result, diag = wavebit.fdtd(
    structure=np.full((1, 1, 1), 2.25), thickness=[.2],
    top=1., bot=1., period=.1, wavelength=[.9, 1., 1.1],
    theta=.45, tetm='te', backend='numpy', resolution=40,
    pulse_width=3., n_periods=80, truncation_order=0,
    return_diagnostics=True, convergence_window=200,
)
print(diag['pulse']['relative_spectrum'])
print(diag['convergence']['state_decayed'])
```

`diag['pulse']`는 `width`, `peak_time=5*width`, `frequency=1/centre`와, 요청한 파장 순서의
`relative_spectrum`을 제공합니다. 후자는 잘리지 않은 이상적인
가우시안의 반송파 대비 진폭 스펙트럼입니다.
`exp(-(pi*width*(1/wavelength-1/centre))^2)`입니다. 이것은 소스 포락선을 기술할 뿐,
측정된 입사 플럭스나 보정 오차 한계가 아닙니다.

한 시간 간격보다 짧은 폭과, 요청한 주파수를 공칭 최대값의 기계 엡실론 100배
아래로 떨어뜨리는 폭은 거부합니다. 이 수치 꼬리
방어는 백엔드 소스 dtype을 따르며, 통과한 값이 약하게 여기된 주파수에서의
정확도를 증명하지는 않습니다. 이 방어는 폭을 생략했을 때도 적용됩니다.
실패하면 `pulse_width`를 줄이거나 요청한
파장 구간을 좁히세요. 최소 실행 시간에는 펄스 폭 열 개와
장면 왕복 여유가 포함됩니다. `n_periods`는 여전히 소스 최대점 이후의 시간이 아니라
0부터의 전체 시간을 중심 파장 광학 주기로 잽니다.

이 설정은 아래 입사, 비균일화, 기록, DFT 부분 샘플링,
재료 자동 미분과 함께 작동합니다. 폭 자체는 정적입니다. 레거시 장면 경로에는
`pulse_width` 키워드가 없으며, native 격자 API는 명시적인 펄스 배열을 받습니다.
소스 꼬리, 분산 경계면, 패턴이 있는 금속 모서리의 효과를 따로 본 결과는
[비스듬한 펄스 측정](../../validation/fdtd/fdtd-oblique-pulse.md)을 보세요.
:::


### Suppressing selected source frequencies

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

명시적 백엔드 장면 어댑터는 `pulse_notches`도 받습니다. 역 c=1 시간 단위의
양의 정적 주파수, 또는 주파수 한두 개로 된 벡터이며,
**파장이나 각주파수가 아닙니다**. `None`은 원래
가우시안을 유지합니다. 각 주파수는 시간 격자의 나이퀴스트 주파수보다 낮아야 합니다.
예를 들어 파장 0.9, 1.0, 1.1을 측정하면서 주파수 0.8의 원치 않는 진동을
약하게 하려면 다음과 같이 합니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic Pulse(notch_frequencies_hz=...).
import numpy as np
from wavebit.compat import legacy as wavebit

result, diag = wavebit.fdtd(
    structure=np.full((1, 1, 1), 2.25), thickness=[.2],
    top=1., bot=1., period=.1, wavelength=[.9, 1., 1.1],
    backend='numpy', resolution=40, truncation_order=0,
    pulse_width=2., pulse_notches=[.8], n_periods=45,
    return_diagnostics=True,
)
print(diag['pulse']['relative_spectrum'])
```

각 노치 주파수 `fn`에서 성형은 다음과 같은 인과적 두 샘플 필터를 적용합니다.

```text
p_new[n] = p[n] - exp(-2*pi*i*fn*dt)*p[n-1],
```

첫 샘플 이전은 0으로 확장합니다. 완전한 유한 합성곱은
부동소수점 반올림 오차 범위에서 `fn`에 스펙트럼 영점을 가집니다. 각 노치는
필터링 전에 마지막 샘플 하나를 예약하며, 실행 시간 여유에는
그 샘플이 포함됩니다. 성형된 소스는 샘플링된 최대 진폭이 1이 되도록 정규화됩니다.
같은 주파수 두 개는 이중 영점을 만듭니다. 그보다 높은 중복도는 거부합니다.
반복된 차분이 가우시안의 유한한 시작·끝 가장자리와
부동소수점 오차를 증폭하기 때문입니다. 영점 두 개만으로도 대역 밖 여기가 늘 수 있으므로,
영점이 많을수록 낫다고 가정하지 말고 실행 시간과 소스 폭을 비교하세요.
실제·기준 실행 네 번 모두
같은 성형 펄스를 사용하며, 요청한 각 파장은 자체
입사 필드 보정을 유지합니다. 계산 후 필드 데이터를 필터링하지는 않습니다.

성형된 소스에서 `diag['pulse']['spectrum_kind']`는 `'sampled_l1'`입니다.
이때 `relative_spectrum`은 요청한 각 주파수에서의 **실제 유한 샘플** 푸리에 크기를
`sum(abs(pulse))`로 나눈 값입니다. 따라서 0과 1 사이에 있으며,
반송파에 대한 비율이 아닙니다. `carrier_fraction`은 반송파에서
같은 정규화를 사용하고, `notches`는 적용한 주파수를 보고하며,
`sampled_peak_time`은 실제 성형 펄스의 최대점 위치를 알려 줍니다. 기존의
`width`, `peak_time`, `frequency`는 여전히 바탕이 되는 가우시안을 기술합니다.
노치가 없으면 `spectrum_kind='ideal_gaussian'`이며 위의 원래 해석적
`relative_spectrum`이 유지됩니다.

요청한 주파수에 놓인 노치는 거부합니다. 시간 필드를 기록할 때는
반송파가 그 기록을 정규화하므로 반송파도 계속 여기되어 있어야 합니다. 시간 기록이
없으면 요청하지 않은 반송파는 억제해도 됩니다. 필요한 샘플링 스펙트럼이
소스 L1 한계의 엡실론 100배 이하이면 백엔드의 소스 정밀도를 기준으로
거부합니다. 이것은 수치적 하한일 뿐입니다. 남아 있는 약한
여기는 여전히 유한 시간, 경계, 반올림 오차를 증폭할 수 있습니다.

이 제어는 아래 입사, 비균일화, 방향성 추출,
기록, 재료·극점 그래디언트를 지원합니다. 노치 주파수와 소스 폭은
정적이며, 레거시 장면 경로에는 `pulse_notches` 옵션이 없습니다. 임의의 파형이나 소스 그래디언트에는
native 격자의 명시적 `pulse` 배열을 사용하세요.
스펙트럼 영점은 여기를 바꿀 뿐, 재료나 열린 경계 모델을 바꾸지 않으며,
수렴이 짧아진다고 보장하지 않습니다. 회절 문턱 근처에서 사용하기 전에
[차단 소스 측정](../../validation/fdtd/fdtd-pulse-notches.md)을 보세요.
:::


### Shared-backend time records and user monitors

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

```python
# Earlier interface; current whole-domain records: wavebit.fdtd.periodic Snapshots.
import torch
from wavebit.compat import legacy as wavebit
from wavebit.compat.legacy import Stack, Incidence

value = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
scene = Stack.from_structure(value.reshape(1, 1, 1), [.3], top=1., bot=1., period=.4)
result, diag = wavebit.fdtd(
    scene, Incidence(wavelength=1., tetm='te'), backend='torch',
    resolution=20, n_periods=30, truncation_order=0,
    record_field=5, record_components='all',
    field_monitors={'inside': .15, 'after': .6},
    return_diagnostics=True,
)
trajectory = diag['fields']['ey']      # values(time,z,y,x), t/z/y/x axes
plane = diag['monitors']['after']     # six fields(wavelength,y,x), flux, power
```

두 옵션 모두 이제 공통 NumPy/JAX/Torch 스테퍼를 사용합니다. 기록은
`0,N,2N,...` 단계 뒤에 샘플링하며, E의 시각은 `(step+1)*dt`, H는 `(step+.5)*dt`입니다. 각 성분은
고유한 native Yee 좌표를 유지합니다. y 불변 래스터는 크기 1인
y 축을 반환합니다. 저수준 `fdtd_grid`는 같은 기록 옵션을 받으며
native 격자 전체를 유지합니다. 기록을 켜면 기본적으로 여섯 성분을 모두
기록합니다. 저장 공간을 줄이려면 성분 하나나 성분 시퀀스를 선택하세요. JAX는 기록
구간 단위로 scan하고, Torch는 그 안에서 체크포인트합니다. 샘플링하지 않은 전진 필드는
기록에 추가하지 않습니다. 역전파와 스냅숏 메모리는 여전히 크기를 가늠해야 합니다.

사용자 모니터 좌표는 아래 경계면이 `z=0`이고 +z가 위쪽입니다. 모니터는
코어의 같은 위치에 모은 노드 평면을 z 방향으로 선형 보간하며, PML 밖에 있어야 하고
전류 시트에서 떨어져 있어야 합니다. 필드는 요청한 각 파장에서 독립적으로,
빛이 들어오는 경계면의 단위 입사 전기장 진폭으로 보정합니다.
`power_z`는 주기 평균한 정규화 `E x H* / 2`를 적분하며, y 불변
출력은 y 단위 길이당 값입니다. 공간 보간은 유한 격자 오차를 일으키므로,
플럭스를 모드 회절 효율과 비교할 때는 격자를 세분화하세요.

시간 기록은 펄스의 반송파 주파수 스펙트럼을 곱한 **일정한 중심 주파수 보정**을
사용합니다. 기록은 실제 물리 필드가 `Re(values)`인 인과적 복소 반송파로
남습니다. 이로써 입사 Jones 상태가 반송파 파장에서 고정됩니다.
그 파장에서 벗어나면 고정된 전류 시트가 방출하는
편광이 독립적으로 정규화된 주파수 영역
모니터 상태와 같을 필요가 없습니다. 파장에 따라 달라지는 보정을 시간
샘플에 직접 적용하면 이 파형이 보존되지 않습니다. 반송파에서는 기록의 오프라인 DFT를
소스 스펙트럼으로 나눈 값이 스트리밍 정규화 모니터와 일치합니다.

전류 시트는 양방향으로 복사합니다. 기록과 사용자 모니터에는
레거시 TFSF 산란 필드 영역과 달리 **양쪽 모두 전체 필드**가 들어 있습니다.
회절 반사는 여전히 별도의 균일 기준 실행을 뺍니다.
진단은 정규화, 시간 표현, 소스 좌표,
층 경계면, PML 경계를 명시합니다. 기존의 NumPy 그리기, 애니메이션, 점
탐침 도우미는 이 장면 기록을 받으며 전류 시트에 올바른 레이블을 붙입니다.
이 시각화·탐침 도우미는 NumPy 후처리로 남으므로, 사용하기 전에
JAX/Torch 진단을 명시적으로 변환·분리하세요. native 기록과 모니터
배열 자체는 재료와 편광 그래디언트를 보존합니다. JIT 함수는
선택한 배열 출력을 반환할 수 있으며, 설명용 진단 문자열은
JAX 배열 출력이 아닙니다.

기록 테스트는 독립적인 정확한 이산 Yee 고유 모드를 전파하고,
스트리밍 DFT와 오프라인 DFT를 비교하며, 모든 기록 구간의 꼬리를 검사하고,
세 백엔드를 비교하며, 이력과 모니터 전력을 미분하고, 원편광
직교 조합을 보존하며, 그리기·탐침 호환성을 실행해 봅니다. 기록을 켜도
회절 결과는 바뀌지 않습니다.
:::


### Half-space material derivatives

반공간은 `Open()` 면에 닿으며, 그곳의 PML은 구체값으로 한 번 만들어집니다.
현재 모듈은 `fdtd.PML(reference_media={face: Material(...)})`가 추적되는 매질이 닿는
모든 면의 프로파일을 고정하면 이 미분을 계산합니다. 그 항목이 없으면, 그곳의 추적되는
유전율이나 투자율은 시간 전진 전에 `ValidationError`("PML on 'z-' is matched to a
background whose permittivity or permeability is traced")를 일으킵니다. 안정화된 PML(`stabilizing_loss`)을
채우는 추적 매질은 그 항목이 있어도 거부합니다.
전체 예제는 [스무딩 기판 가이드](substrate-gradients.md)에 있습니다. main에서 측정한 결과
(2026-11-09, 수직 입사, `dx = 0.05`, PML 셀 10개), 투과 (0, 0) 효율의
미분은 두 z 면을 고정한 상태에서 JAX와 PyTorch 모두 중심 차분과 일치합니다. `SnappedSlab`
기판은 6e-9, 이어진 `Box` 기판은 2e-8, 빛이 들어오는 쪽의
`Scene(background=)`는 3e-8 이내입니다. 아래 예제는 과거 경로를 사용합니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

명시적 백엔드에서 `top`과 `bot`은 양의 실수 스칼라
유전율, 또는 양의 실수 스칼라 계수를 가진 `Material(epsilon, mu)`를 받습니다.
배열 계수는 `mesh_indices=(n_top, n_bottom)`과 `truncation_order`를
모두 명시하면 그래프를 유지합니다.
`mesh_indices`에는 **굴절률**이 들어가지만, 재료는 계속
**유전율과 투자율**을 받습니다. 최적화나 유한 차분 스윕 동안
이 공칭 쌍을 일정하게 유지하세요.
NumPy 벡터 인덱싱은 배열 대신 호스트 스칼라를 반환할 수 있습니다. 그래도
교란한 모든 평가에 같은 명시적 메시 굴절률과 차수 창을
주세요. 교란할 때마다 공칭 쌍을 다시 계산하면 비교하는
실험이 바뀝니다.

```python
# Earlier interface; the current route is fdtd.PML(reference_media=...), described above.
import jax
import jax.numpy as jnp
from wavebit.compat import legacy as wavebit

jax.config.update('jax_enable_x64', True)

def transmission(p):
    result = wavebit.fdtd(
        structure=jnp.full((1, 1, 1), 2.25), thickness=[.2],
        top=wavebit.Material(p[0], p[1]), bot=wavebit.Material(p[2], p[3]),
        period=.2, wavelength=1., backend='jax', resolution=20,
        mesh_indices=(1.1, 1.4), truncation_order=0,
        align_layers=True, n_periods=40,
    )
    return result.s_inc.T.sum()

value, gradient = jax.jit(jax.value_and_grad(transmission))(
    jnp.array([1.2, 1.1, 1.6, 1.15]))
```

공칭 굴절률은 패딩 길이와 비균일 z 셀 크기를 정합니다.
실제 epsilon/mu는 래스터화된 경계 셀, Maxwell 갱신,
CPML 늘림, Bloch 위상, 방향성 투영, 경계면 위상 기준,
입사 필드·전력 정규화를 결정합니다. 메시, 소스와 모니터 위치,
시간 간격, 갱신 횟수는 고정됩니다. 같은 계약이 NumPy
유한 차분과 Torch autograd에도 적용됩니다. 일반 호스트 계수는
`mesh_indices`를 생략하면 이전의 자동 메시 선택을 유지합니다.

theta/phi를 고정했을 때 **입사** 매질을 바꾸면 중심 파장의 입사각이
고정되도록 횡방향 파수 벡터가 바뀝니다. 대역 전체가
그 파수 벡터를 공유합니다. 이것은 kx/ky를 고정한 미분이 아닙니다.
각도, 래스터 재료, 전기 극점은 반공간 계수 네 개와 함께, 그리고
`mesh_thickness`를 사용하는 합계 일정 층 두께와 함께 동시에 바꿀 수 있습니다.
주기, 파장, 메시 기하는 정적으로 남으며,
손실·분산 반공간은 지원하지 않습니다.

이 쌍은 아래 조명에서도 항상 원래의 **위, 아래**를 가리킵니다.
진단은 정적인 `mesh_indices`와 살아 있는 `background`를 반환하며, 후자는 원래의 위/아래 순서로 된
`(epsilon, mu)`의 `(2,2)` 배열입니다. 실제 굴절률을 늘리면 더 긴 펄스 실행이 필요할 수 있지만,
실행을 조용히 늘리지는 않습니다.
구체값의 잘못된 재료, 부족한 실행 시간, CFL 위반, 특이
채널은 예외를 일으킵니다. JAX JIT 아래에서는 `stable`을 무효로 하고 산란과
정규화된 필드를 NaN으로 가립니다. 최적화기는 유효한 영역 안에 머물러야 합니다.

공칭 굴절률이 넓은 재료 범위에서 물리적 메시나 PML 정확도를
보장하지는 않습니다. 관련 설계점에서 해상도, 패딩, 물리적 PML 두께, 실행 시간을
확인하세요. [검증 노트](../../validation/fdtd/fdtd-halfspace-gradients.md)는
유한 격자 미분의 정확성을 연속체 및 외부 솔버와의
일치와 분리합니다.
:::

### Internal layer thickness derivatives

현재 모듈에서 `SnappedSlab`의 범위는 정적이므로 추적되는 `z`는 거부하며, 범위가 지정된
미분 가능한 `Box` 층은 반공간 `SnappedSlab`이나 다른 셀 스냅 면 위에 바로 붙어 설 수
없습니다. 이 절과 다음 절의 예제는 과거 경로를 사용합니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

정적인 `mesh_thickness`를 넘겨 z 메시를 고르고, 백엔드 `thickness`
값으로 내부 경계면을 움직이세요. 실제 길이는 양수로 유지되어야 하고,
그 합은 공칭 전체 깊이와 같아야 합니다. 이렇게 하면 바깥쪽 기준 평면 두 개,
소스, 모니터, PML, 기록 좌표, 시간 간격이 모두
고정됩니다. 이 표기는 RCWA 및 타입이 있는 `Stack` 객체와 공유합니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic (thickness differentiates).
import jax
import jax.numpy as jnp
from wavebit.compat.legacy import Stack, Incidence, fdtd

jax.config.update('jax_enable_x64', True)

def reflection(d):
    scene = Stack.from_structure(
        jnp.array([[[2.25]], [[1.5]]]), [.4-d, d],
        top=1., bot=2.25, period=.05,
    )
    light = Incidence(wavelength=1., polarization=(1., 0.))
    result = fdtd(
        scene, light, backend='jax', resolution=40,
        mesh_thickness=[.2, .2], align_layers=True, truncation_order=0,
        n_pml=40, n_periods=60, pulse_width=3.,
        diffraction_method='directional',
    )
    return result.s_inc.R.sum()

value, derivative = jax.jit(jax.value_and_grad(reflection))(.173)
```

여기서 첫 번째(아래) 층은 기판 스페이서입니다. 그 길이가 바뀌어도
아래 출력 면은 z=0, 빛이 들어오는 위쪽 면은 z=0.4에 유지됩니다. 층이 더 많으면 양의 설계
비율을 고정된 합계로 정규화하세요. 그래디언트는 합계가 일정한 경로를 따라 정의되며,
전체 적층 깊이의 독립적인 변화를 나타내지 않습니다.
공유 함수 [`wavebit.layer_thicknesses`](../../reference/common/index.md)는
지정한 최소 길이를 강제하면서 이 정규화를
수행합니다. [다층 최적화 예제](../../../examples/fdtd/fdtd_multilayer_optimization.py)는
독립적인 두께 좌표 세 개와 유전율 세 개를 결합하며,
그 [검증 보고서](../../validation/fdtd/fdtd-multilayer-optimization.md)에는 조밀한
독립 스펙트럼과 세분화한 그래디언트가 들어 있습니다.

`subpixel=True`가 필요합니다. 셀 겹침 비율은 epsilon/mu 평균과
전기 극점 마스크를 통해 두께 그래프를 유지합니다. 값은 연속이지만,
경계가 셀 면을 지날 때 미분에 꺾임이 생길 수 있습니다. 그 교차점의
하위 그래디언트를 유일한 물리적 형상 미분으로 사용하지 마세요.
연속적인 1차 미분이 필요하면 아래에서 설명하는 선택적 `layer_smoothing`을 설정하세요.
그래디언트 정확도를 평가할 때는 격자를 세분화하고 여러 경계면 위치를 샘플링하세요.
[검증 노트](../../validation/fdtd/fdtd-layer-gradients.md)는 AD/유한
차분, Airy 수렴, 독립적인 공개 API Meep 교란을 분리합니다.
[최적화 예제](../../../examples/fdtd/fdtd_layer_optimization.py)에는
백트래킹과 두 배 해상도에서의 최종 검사가 들어 있습니다.

JAX/Torch 두께 배열에는 명시적인 `mesh_thickness`가 필요합니다. 메시를 고르기 위해
즉시 실행 텐서를 분리하지 않습니다. NumPy 유한 차분도 같은 옵션을 사용합니다.
`align_layers=True`는 공칭 면을 맞추며, 실제 경계면은
고정된 셀 사이를 움직입니다. 이 옵션은 비균일화, 아래 조명,
잘라낸 기록, 필드 모니터와도 함께 작동합니다. 진단은 원래 순서의 정적인
`mesh_thickness`와 살아 있는 `layer_planes`를 반환합니다. 잘못된 구체값은 예외를 일으키며, JIT에서의
무효는 산란과 정규화된 필드를 가리고 `stable=False`를 설정합니다.
:::

### Changing the total stack depth

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

층 길이를 독립적으로 최적화하려면 명시적 백엔드, `mesh_thickness`,
`subpixel=True`와 함께 `variable_depth=True`를 설정하세요. 모든 물리적 길이는
양수로 유지되어야 합니다. 그 합은 정적인 공칭 합과 달라도 됩니다.
아래 면은 z=0에 머물고 위 면은 z=sum(thickness)에 있습니다.
빛이 들어오는 면이 입구, 반대쪽 면이 출구이며, 복소
진폭은 RCWA와 같은 규약으로 그 물리적 면을 기준으로 합니다. 재료 비율과 기준 평면 전파 위상 모두
JAX/Torch 미분을 유지합니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic Numerics(variable_depth=True).
import jax
import numpy as np
from wavebit.compat.legacy import fdtd

jax.config.update('jax_enable_x64', True)

def transmission(d):
    result = fdtd(
        structure=np.array([[[2.25]]]), thickness=[d],
        mesh_thickness=[.2], variable_depth=True, layer_smoothing=.025,
        top=1., bot=1.21, period=.05, wavelength=1.,
        backend='jax', resolution=40, n_pml=40, n_periods=60,
        pulse_width=3., truncation_order=0, diffraction_method='directional',
    )
    return result.s_inc.t_s.sum()

amplitude, derivative = jax.jit(
    lambda d: (transmission(d), jax.jacfwd(transmission)(d))
)(.173)
```

`variable_depth`는 정적인 불리언이며 기본값은 False입니다. 공칭 메시 길이는
여전히 노드, 소스 시트, 추출 평면, PML, 시간 간격,
기록 샘플을 결정합니다. 이 실험실 좌표는 위 입사와 아래 입사 모두에서
고정됩니다. `source_distance`와 `diffraction_distance`는 공칭 면으로부터의 거리를
선택하며, 최적화되는 경계를 따라 소스나 검출기를 움직이지 않습니다.
정적인 사용자 모니터와 기록 범위는 원래 좌표를 유지합니다.
반환되는 필드는 물리적 입사 면의 단위 입사 E로 정규화된 채 남으므로,
면이 움직이면 정규화 위상이 바뀔 수 있습니다.

모든 물리적 경계면과 그 평균 지지 영역은
고정된 회절 스텐실에서 떨어져 있어야 합니다. 바깥쪽 각 스텐실에는
인접 셀 폭 하나만큼의 보수적인 여유 거리가 있습니다. 의도한 설계 범위가 그 경계를 넘는다면
공칭 패딩을 늘리거나 공칭 길이를 조정하세요. 즉시 실행의 잘못된 입력은 예외를 일으키고,
추적되는 잘못된 입력은 `stable=False`와 NaN 산란·정규화 필드를 줍니다.
진단은 원래 좌표로 `variable_depth`, 정적인 `mesh_thickness`, 살아 있는 `layer_planes`,
살아 있는 `reflection_z`/`transmission_z`를 보여 줍니다.

양의 `layer_smoothing`은 움직이는 바깥쪽 면도 평균하며,
z=0의 고정된 물리적 면은 날카롭게 남습니다. 첫 번째와 마지막 층 길이는
층이 하나뿐인 적층을 포함해 최소 그 폭의 절반이어야 합니다. 물리적 점진
전이 구간은 출구 기준 평면을 반 폭만큼 넘어 이어집니다. 기준은
그 중심까지의 평면파 외삽이며, 그곳의 재료가 출사 반공간과 같다는
주장이 아닙니다. 스무딩이 0이면 날카로운 셀 겹침을 유지합니다.

맞춘 메시나 비균일 메시에서 셀 안의 면까지의 전파는 그
셀의 길이당 이산 위상을 사용합니다. 노드에서는 오른쪽 셀이
한쪽 위상 미분을 정의합니다. 따라서 인접 셀 폭이 다르면 재료 비율을 스무딩하더라도
위상 기울기에 꺾임이 생길 수 있습니다. 물리적 정확도는
메시 세분화로 평가하세요.
[움직이는 깊이 측정](../../validation/fdtd/fdtd-variable-depth.md)을 보세요.
:::

### Continuous first derivatives at internal interfaces

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

명시적 백엔드와 `mesh_thickness`와 함께 `layer_smoothing=h`를 설정하면 재료를
샘플링하기 전에 각 내부 경계면 위치를 `[-h/2, h/2]`에 걸쳐 균일하게 평균합니다.
`h`는 `thickness`와 같은 물리 길이 단위의 정적이고 유한하며 음이 아닌 실수
스칼라입니다. 기본값 `0.`은 정확한 셀 겹침을 유지합니다.
양의 폭에는 `subpixel=True`가 필요하며, 백엔드 텐서는 즉시 실행 스칼라
텐서라도 이 폭을 정할 수 없습니다. 균일 메시에서 `h=1/resolution`은
셀 하나만큼 평균합니다. 맞춘 메시나 비균일 메시에서는 물리적 폭을 고를 때
실제 셀 폭을 사용하세요.

기본적으로 적층의 바깥쪽 두 면은 고정되고 날카롭게 남습니다. 모든 내부 경계면은
바깥쪽 면에서 최소 `h/2` 떨어져 있어야 하며, 내부 평균 영역끼리는 서로
겹쳐도 됩니다. `[d, D-d]`의 경우 양의 길이에 더해
`h/2 <= d <= D-h/2`를 뜻합니다. 잘못된 구체값 장면은 예외를 일으키고, 잘못된 JAX 추적 장면은
다른 기하 실패와 같은 `stable=False`와 NaN 가림을 사용합니다.
여유 거리 검사는 전체 깊이 검사와 같이 D의 기계 엡실론 32배를 허용하므로,
면에 닿는 경우도 부동소수점 반사 후에 유효하게 남습니다.
층이 하나인 적층에는 내부 경계면이 없으므로, 위에서 설명한 대로
`variable_depth=True`가 움직이는 출구를 평균하지 않는 한 이 옵션은 물리적 효과가
없습니다. 진단에는 정적인 `layer_smoothing` 값이 포함됩니다.

셀 비율은 움직이는 경계가 셀 면을 지나는 곳을 포함해 연속적인 1차 미분을
가집니다. 이것은 내부 경계면 샘플링에 대한 C1 보장이며, 불리언 유효성 판정,
가능한 모든 손실 함수, 고차 미분에 대한 보장이 아닙니다. epsilon/mu와 전기 극점 채움은
기존의 평균 규칙과 JAX/Torch 그래프를 유지합니다.

양의 폭은 재료 프로파일을 바꿉니다. 한 번의 최적화 안에서는 폭을 고정하고,
날카로운 경계면 구조가 의도한 물리 구조라면 메시와 함께 폭을 줄이고 최종
설계를 따로 검사하세요. [스무딩 측정](../../validation/fdtd/fdtd-layer-smoothing.md)은 이 편향을
유한 격자 오차와 구분합니다. 백트래킹 최적화 뒤에 스무딩한 경우와 날카로운 경우 모두
더 세밀한 격자에서 검사하려면 코팅 예제를 `--smoothing-cells 1`로 실행하세요.
:::

### 공통 엔진의 아래 조명

현재 모듈에서 아래에서 오는 빛은 적층 아래에 놓은 `PlaneWave(direction='+z')`이며,
`wave='reflected'` 모니터는 그 아래에, `wave='transmitted'` 모니터는 적층 위에
둡니다. 장면 자체는 바뀌지 않습니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

과거 팩토리에서는 `Incidence(..., incidence_side='bottom')`를 설정하거나 `incidence_side='bottom'`을 넘기세요.
두 주기 경로 모두 이를 지원하며, 이 절은 명시적 엔진을 자세히 다룹니다. 구조, 위/아래 재료, 층 순서는
원래 장면 그대로 남습니다. 각도는 RCWA와 마찬가지로 입사
매질의 안쪽 법선을 기준으로 잽니다. R은 그 매질로 반사되는 전력이고,
T는 반대쪽으로 빠져나갑니다. 복소 r은 아래
경계면을, t는 위 경계면을 기준으로 합니다. `incident`, `s_inc`, `p_inc`와 차수 레이블은
입사 기준 규약을 유지합니다.

공개 어댑터는 아래쪽과 위쪽 반공간을 적절한
내부 엔진 포트에 연결합니다. 조명 방향과 무관하게 모든 층 배열은 아래에서 위로 남고, 모든 공개
좌표는 +z가 위쪽입니다. 내부 계산이 반사된 좌표를 사용할 때 E는 극성 벡터,
H는 축성 벡터이며, 어댑터는 필드를 반환하기 전에 둘 다 변환합니다. 사용자 평면 `power_z`는 부호 있는
실험실 플럭스입니다. 아래에서 들어와 투과하면 보통 `power_z`가 **양수**이고,
위에서 들어와 투과하면 음수입니다. 회절 T는 어느 경우든 양수입니다.

이 경로는 JAX/Torch 재료 그래디언트를 포함해 위 입사와 같은 래스터, 양의 자성 매질, 전기 극점,
Jones 상태, 기록, 모니터 옵션을 지원합니다. 백엔드를 생략한 NumPy 경로도
공개 어댑터를 통해 아래 입사를 지원하며, 자체 소스 정규화를 사용하고
자동 미분은 없습니다. 그 비공개 시간 전진 커널은 단방향으로 남습니다.
상반성 비교에는 격자 세분화가 필요합니다. 경계면이 격자 노드에 맞지 않는 적층을
뒤집으면 샘플링된 경계면이 바뀌기 때문입니다. 비대칭 손실 2층 테스트에서
파수 벡터를 맞춘 투과 차이는
해상도 40에서 0.0083, 해상도 80에서 0.0018로 줄어듭니다.

검증에는 RCWA와 비교한 복소 자성 다층 진폭,
교차 분산 층 뒤집기, 물리적 필드와 플럭스 부호, 입력
불변성, 타입이 있는 장면 그래디언트, 실제로 실행한 Meep 1.34.0 기준이 포함됩니다. Meep
기준은 소스를 원래 적층 아래에 물리적으로 놓으며, 시뮬레이션을
대칭 반전하지 않습니다. 해상도 80에서 세 파장의 r/t는 복소 진폭으로 6e-4 이내로
일치합니다. Meep의 실행 시간을 30에서 40으로 늘리면 기준값이
7e-15 미만으로 바뀝니다. 문서화된 Meep API만 사용했으며, 그 구현은
읽지 않았습니다. `tests/references/interface/validate_scene_bottom_meep.py`와 고정된 기준값
`tests/references/golden_scene_bottom_meep.json`을 보세요.
:::


### 정적 비균일 z 메시

현재 모듈의 격자는 균일한 정육면체 격자입니다. 이 절과 다음 절은
과거 엔진을 설명합니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

공유 엔진은 이제 z 인덱스마다 셀 폭 하나를 갖는 정적인 양의 벡터 `z_spacing`을
받습니다. 이 값은 `spacing[0]`을 대체하며, x와 y는 균일하게 남습니다.
마지막 폭은 주기를 닫는 셀이므로, z 주기 전체는 모든 폭의
합입니다. native E 노드는 0에서 시작해 누적 폭을 따르며,
z 반 칸 성분은 각 셀 중점에 놓입니다. 반환되는 `z_nodes`, `z_spacing`과
성분별 기록 축은 실제 기하를 기술합니다.

자기장 갱신은 인접한 E 노드 사이 거리를, 전기장 갱신은
인접한 H 중점 사이 거리를 사용합니다. CFL 한계는 가장 작은
셀 폭을 사용합니다. 평면 보간 가중치는 물리 거리에 따라 달라집니다. CPML과
선택적 정합 전도도 흡수층은 각 끝의 물리적 폭을 사용하므로,
셀 수가 같다고 흡수층 두께가 같은 것은 아닙니다. 재료, 극점,
필드, 소스는 NumPy/JAX/Torch 동작과 그래디언트를 유지합니다. 메시 자체는
정적이며 미분 가능한 설계 변수가 아닙니다.

`Stack` 장면에서는 `grading=True`(비율 1.06) 또는 1과 2 사이의 비율을 넘기세요.

```python
# Earlier interface; current: wavebit.fdtd.periodic Numerics(algorithm='reference', grading=...).
import numpy as np
from wavebit.compat import legacy as wavebit
from wavebit.compat.legacy import Stack, Incidence

scene = Stack.from_structure(np.full((1, 1, 1), 2.25), [.3],
                             top=1., bot=1., period=.1)
result, diag = wavebit.fdtd(
    scene, Incidence(wavelength=1., tetm='te'), backend='jax',
    resolution=40, grading=1.06, n_periods=30,
    truncation_order=0, return_diagnostics=True,
)
print(diag['shape'], diag['z_spacing'].min(), diag['z_spacing'].max())
```

기본값 `align_layers=False`에서 구조는 z 방향으로 균일하게 남고,
바깥쪽 두 경계면은 정확히 노드 위에 있습니다. 내부 층 경계면은 셀을 가로지를 수 있습니다.
반공간 패딩은 점진적으로 커져 최대 코어 폭의 두 배까지 되며, 매질 안의 가장 짧은 파장당
20개 셀이라는 상한으로 제한됩니다. 요청한 메시가 이미 그 상한보다
거칠면 비균일화가 메시를 더 거칠게 하지 않습니다. 이 기하 정책은
최적화되는 래스터나 반공간 값이 아니라, 해상도와 정적인 공칭 `mesh_indices`(생략하면 실제 호스트 반공간
굴절률)에 따라 달라집니다.
기존 `graded_cells`가 기하급수적 경사를 제공합니다. E 노드 재료 평균은
실제 비대칭 쌍대 셀 범위를 사용하고, 반 노드 값은 원시 셀을 사용합니다.

소스와 모니터 인덱스는 실제로 만들어진 메시에서 선택합니다. 패딩에
소스, 모니터, PML이 들어가지 않으면 `pad_scale`을 늘리세요. CPML은 여전히 끝마다
`n_pml`개 셀을 세며, 물리적 두께는 메시에서 측정합니다. 위상
추출은 각 모니터와 그 기준 경계면 사이에서 국소 이산 균질 분산을
적분합니다. 비균일화는 메시 반사를 일으키며, 이 위상 보정은 그것을 제거하지 않습니다.
새로운 구조에서는 해상도, 비균일화, 패딩, 실행 시간을 세분화해 보세요.
해상도 40의 슬래브 예제는 200개 대신 112개 z 셀을 사용합니다. 이것은 셀 수 절감이며,
보편적인 속도 향상을 보장하지 않습니다.

기록, 사용자 평면, 아래 입사, 재료 미분은 모두 장면 비균일화와
함께 작동합니다. native 격자 테스트는 독립적인 일반화
Yee 고유값 문제와 비교하고, 물리적으로 가중한 보간, 균일 격자 일치,
CPML/ADE 민감도와 흡수를 검사합니다. 장면 테스트는 Airy 세분화,
RCWA와 비교한 자성 conical 진폭, 기존의 독립 Meep 아래
조명 데이터, 비균일 메시의 기록과 모니터를 거친 JAX/Torch 미분을 검사합니다.
Meep 기준은 균일 격자를 사용하므로, 이 비교는 동일한 이산화가 아니라
세분화에 따른 광학적 일치를 확립합니다. Meep
구현은 읽지 않았습니다.
:::

### Aligning every layer boundary

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

명시적 백엔드 장면 API는 `grading`과 독립적으로 `align_layers=True`를
받습니다. 두께 `d`인 층에는 폭이 `d / ceil(d * resolution)`인 같은 크기의
z 셀 `ceil(d * resolution)`개를 사용합니다. 모든 층은 물리적
두께를 유지하고, 모든 경계는 z 노드 위에 놓입니다. 구조 안의 셀 폭은
최대 `1/resolution`이며, x/y 샘플링은 바뀌지 않습니다. 기본값은 `False`입니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic Numerics(align_layers=True).
import numpy as np
from wavebit.compat import legacy as wavebit

result, diag = wavebit.fdtd(
    structure=np.array([[[2.25]], [[1.5]]]), thickness=[.113, .213],
    top=1., bot=1., period=.1, wavelength=1., tetm='tm', theta=.45,
    backend='numpy', resolution=40, align_layers=True,
    n_periods=50, pulse_width=3., truncation_order=0,
    return_diagnostics=True,
)
print(diag['align_layers'], diag['layer_planes'])
```

비균일화가 없으면 패딩은 폭 `1/resolution`을 유지하므로, 층 면에서 셀 폭이
갑자기 바뀔 수 있습니다. 비균일화를 쓰면 각 반공간 경사는 인접한
층의 셀 폭에서 시작하며, 위에서 설명한 것과 같은 거칠게 하기 정책을 따릅니다. 비균일화는
서로 다른 층 사이의 급변을 매끄럽게 하지 않습니다. 재료 겹침 가중치, 시간 간격,
CPML, 위상 추출, 필드 좌표는 모두 실제로 만들어진 메시를 사용합니다.
`pml_bounds`는 바깥 경계에서 주기를 닫는 마지막 셀을 포함하므로, 보고되는 각
흡수층 범위에는 `n_pml`개 셀이 모두 들어 있습니다.

얇은 층은 층이 셀 하나에 딱 맞는 경우를 포함해 전역 시간 간격을 훨씬 작게 만들 수 있습니다.
규모를 키우기 전에 `diag['dt']`, `diag['steps']`, `diag['shape']`,
`diag['z_nodes']`, `diag['z_spacing']`을 확인하세요. 맞춘 층이
반드시 잘 분해된 것은 아닙니다. 메시 선택은 정적으로 남습니다. 두께
미분에는 위에서 설명한 대로 `mesh_thickness`를 주세요. 그러면 정렬은
공칭 면에 맞춥니다. 이것은 적응형 세분화가 아닙니다. 이 옵션에는 불리언과
명시적 백엔드 선택이 필요하며, 백엔드를 생략한 레거시 경로는 이를 받지 않습니다.

기록, 물리적 기록 범위, 주파수 평면, 아래 입사,
NumPy/JAX/Torch 재료·극점 그래디언트는 평소의 의미를 유지합니다.
아래 입사 좌표는 원래 적층 좌표계에 남습니다. 정렬은 수직 Ez 셀 안에서
서로 다른 층이 섞이는 것을 피하지만, 분산
평균 모델을 바꾸거나 측면 래스터 모서리를 정렬하지는 않습니다.
[다층 측정](../../validation/fdtd/fdtd-layer-alignment.md)은 복소 TE/TM
세분화를 검사하고, 이 평면 개선을 분해되지 않은 금속 모서리와 구분합니다.
:::

### Pulse-decay diagnostics in the common engine

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`fdtd_grid(..., convergence_window=N, convergence_tol=1e-4)`는 시간 루프를 바꾸거나 일찍 멈추지 않고
마지막 `N` 단계를 검사합니다. `N`은 `steps`보다 작은 양의
정수여야 하며, 상대 진폭 허용 오차는 0과 1 사이(양 끝 제외)여야
합니다. 기본값 `convergence_window=None`은 추가
축약 연산과 스펙트럼 누적기를 끕니다. 명시적 백엔드 장면 API는
`return_diagnostics=True`와 함께 같은 옵션을 받습니다.

native 결과의 `convergence` 매핑에는 다음이 들어 있습니다.

| 필드 | 의미 |
| --- | --- |
| `field_decay` | 마지막 창에서의 최대 필드 노름을 초기 필드를 포함한 전체 실행에서의 최대값으로 나눈 값 |
| `signal_present` | 유한하고 0이 아닌 최대 필드 노름이 관측되었는지 여부. 0인 시뮬레이션은 수렴의 근거가 아닙니다 |
| `source_inactive` | 마지막 창의 펄스 최대값이 전체 실행 최대값의 `convergence_tol`배 이하이거나, 전류 패턴이 0인지 여부 |
| `field_decayed` | 유효한 입력/CFL, 0이 아닌 신호, 비활성 소스, `field_decay <= convergence_tol`을 모두 만족하는지 여부 |
| `spectral_tail` | 각 파장과 모니터 평면에 대해, 마지막 창의 DFT 기여의 E/H 결합 노름을 전체 실행 DFT 노름으로 나눈 값 |
| `spectral_signal_present` | 그 스펙트럼 비율의 분모가 유한하고 0이 아닌지 여부 |
| `window_steps`, `tolerance` | 요청한 설정 |

정의되지 않는 비율은 NaN입니다. 필드 노름은 흡수층을 포함해 각 성분의 native 제어 부피로
가중한 `abs(E)**2 + abs(Z0 H)**2` 합의 제곱근입니다.
E와 H는 반 시간 간격의 시간 차이를 유지합니다. 이것은
필드 감쇠 지표이며, 특히 분산 매질에서는 **저장된 전자기 에너지가 아닙니다**.
ADE와 PML 보조 상태는 포함하지 않습니다. 지속되는
무손실 필드와 활성 연속파 소스는 이 펄스 기준을 통과해서는
안 됩니다. 관련 진동과 전파를 포함할 만큼 긴 창을 고르고,
더 긴 실행으로 반복하세요. 조용한 창이 지연된
재방출이나 나중의 소스 펄스를 배제하지는 못합니다.

장면 진단에는 `convergence['bases']`가 들어 있으며, 소스 기저마다 매핑이 하나씩 있고
`actual`과 `reference` native 진단이 따로 있습니다. 그 스펙트럼은
보정하지 않은 기저 스펙트럼이며, native 실행의 파장·모니터 순서를 따릅니다(시간 필드를 기록할 때는
추가 반송파 주파수 포함). 장면 수준의
`field_decayed`는 네 번의 실행이 모두 통과해야 참입니다. 이것은 r/t 정확도,
공간 해상도, PML 수렴, 수명이 긴 재료 진동의 부재를
증명하지 않습니다. `stable`은 계속 수렴이 아니라 재료/CFL 유효성을 뜻합니다.

진단 배열은 백엔드 그래디언트를 유지하지만, 최대값과 불리언 판정은
모든 곳에서 매끄럽지는 않습니다. 고정된 단계 수는 기존의 미분
계약을 유지합니다. 펄스 감쇠와 검출기 수렴의 구분은
[FDTDX의 종료 조건 설계](https://github.com/ymahlau/fdtdx/blob/main/src/fdtdx/fdtd/stop_conditions.py)를 참고했습니다.
이 구현은 그 소스 코드를 복사하지 않습니다.

native 수렴 매핑은 `electric_spectrum_before_window`와
`magnetic_spectrum_before_window`도 반환하며, 모양과 단위는 전체
스펙트럼과 같습니다. 이것은 마지막 창의 적분을 빼서 얻은 접두 구간 적분이며,
극히 약한 접두 구간은 상대 정밀도를 잃을 수 있습니다.

장면 API에서 `convergence['amplitude_change']`는 요청한 편광에 대해
`r_s`, `r_p`, `t_s`, `t_p`를 **부호 있는 복소 차이(전체 − 접두 구간)** 로 매핑하며,
모양은 `(wavelength, order_y, order_x)`이고 스칼라 입력에도 길이 1인
파장 축이 있습니다. 두 창 모두 각자의 입사 기준
보정과 같은 경계면 기준 평면을 사용합니다. `comparison_time`은
접두 구간의 끝 시각입니다. `calibration_valid`는 파장별 불리언입니다. 특이하거나,
유한하지 않거나, 수치적으로 사라지는 접두 구간 기준은 NaN 변화를 냅니다.

`amplitudes_settled`는 유한하고 유효한 보정, 네 실행 모두의 비활성 소스,
유효한 재료/CFL 입력, 그리고 요청한 모든 진폭 변화의 절댓값이
`convergence_tol` 이하일 것을 요구합니다. 이는 `field_decayed`와 별개입니다.
약한 회절 차수의 상대 정확도나 앞으로의 안정화를
증명하지 않습니다. 공간 오차가 남아 있는 상태에서의 시간 안정화 측정 예는
[펄스 안정화와 공간 정확도](../../validation/fdtd/fdtd-pulse-convergence.md)를 보세요.

같은 수렴 매핑에는 이제 `result.s_inc`와
`result.p_inc`로 반환되는 두 단위 전기장 입사 기저에 해당하는 `s_inc`와 `p_inc`도 들어 있습니다.
각각에는 위의 지정 편광 양과 같은 단위, 모양, 접두 구간 보정, 유효성
규칙을 따르는 `amplitude_change`, `calibration_valid`,
`amplitudes_settled`가 있습니다. 이것은 보정된 s/p
열이며, `convergence['bases']`에 저장된 원시 전류 기저가 아닙니다. 지정한 Jones 벡터와는
독립적입니다. 기존의 최상위 `amplitude_change`와
`amplitudes_settled`는 지정 편광의 의미를 유지합니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic Numerics(decay_check=DecayCheck(...)).
c = diagnostic['convergence']
print(c['s_inc']['amplitudes_settled'], c['p_inc']['amplitudes_settled'])
print(c['scattering_settled'])
```

`scattering_change_bound`는 `r_s`, `r_p`, `t_s`, `t_p` 각각을,
**임의의 단위 Jones 입력**에 대해 가능한 전체 − 접두 구간 변화의 최대 절댓값으로 매핑하며,
파장과 출사 차수마다 따로 계산합니다. 두 입사 열에서의 변화가 `ds`와
`dp`인 출력 행의 한계는 `sqrt(abs(ds)**2 + abs(dp)**2)`입니다.
이 노름은 열 사이의 보강 간섭을 고려합니다. 각 열을 허용 오차와만
비교해서는 이 한계를 얻을 수 없습니다. 하나의 Jones
벡터가 모든 출력 행의 한계를 동시에 달성할 필요는 없습니다.

`scattering_settled`는 이 한계가 모두 `convergence_tol` 이하이고,
보정된 두 열이 모두 유효하며, 소스가 비활성이고, 재료/CFL 입력이 유효할 것을 요구합니다.
`state_decayed`는 요구하지 않으며, 연속체 해에 대한
오차 한계가 아닙니다. 잘못된 접두 구간은 NaN 한계와 거짓 플래그를 줍니다.
배열은 재료 그래디언트를 유지하며, 행 노름이 정확히 0이면 보고되는
한계는 0인 하위 그래디언트를 사용합니다. 플래그는 불리언 판정이며, 매끄러운 목적 함수가 아닙니다.

이 필드들은 모두 기존 `convergence_window`로 켜지며, 네 번의 장면 실행과
접두 구간 스펙트럼을 재사용하므로 추가 시간 전진이 없습니다.
자동 종료는 여전히 없습니다. [손실 ENZ 측정](../../validation/fdtd/fdtd-enz.md)은
s 열이 p 열보다 훨씬 먼저 안정되고, 상당한 시간 이산화 오차에도
결국 둘 다 통과하는 사례를 보여 줍니다.
:::

### 주파수 모니터 부분 샘플링

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`dft_subsample=K`는 `fdtd_grid`와 명시적 백엔드 장면 API에서 사용할 수 있습니다.
기본값은 `1`입니다. 단계 수보다 크지 않은 양의 정수여야 합니다.
`K > 1`이면 시간 기록에 쓰는 추가 반송파를 포함해 요청한 모든 파장이
샘플링 나이퀴스트 주파수보다 엄격히 낮아야 합니다.
이 검사로 전체 과도 신호가 대역 제한되어 있음이 확립되지는 않습니다.

엔진은 `0,K,2K,...`번째 갱신 뒤에 DFT 평면을 평가하며, 기존의 E/H
반 시간 간격 타임스탬프를 사용합니다. 한 샘플은 `min(K, steps-n)`번의 갱신을 대표하며, 마지막
부분 블록은 그에 맞게 가중됩니다. 이것은 **복조된** 필드에 대한 왼쪽 샘플 구적입니다.
앨리어싱 방지 필터는 적용하지 않습니다. 광대역 펄스, 빠르게 변하는 필드, 공진 재료에
사용하기 전에 `K=1`과 비교하세요.
간격을 자동으로 선택하는 기능은 없습니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic Numerics(dft_subsample=...).
result, diagnostic = wavebit.fdtd(
    stack, incidence, backend='numpy', resolution=40,
    dft_subsample=2, return_diagnostics=True)
print(diagnostic['dft_samples'])
```

모든 Maxwell/ADE/CPML 갱신은 여전히 실행됩니다. 샘플링하지 않는 단계는 NumPy/Torch에서는 Python 분기로,
JAX에서는 `lax.cond`로 모니터 보간과 푸리에 누적을 건너뜁니다. 누적기 저장 공간은 바뀌지 않습니다. native 최종 필드와
원시 시간 기록은 바뀌지 않으며, 보정된 장면 기록은 기준 정규화의 작은
변화를 물려받을 수 있습니다. 결과·장면 진단은
`dft_subsample`과 `dft_samples`(스펙트럼이 없는 native 실행에서는 0)를 보고합니다.

`convergence_window`를 쓰면 접두 구간 경계를 가로지르는 샘플 블록은
구적 가중치를 접두 구간과 꼬리로 나눕니다. 따라서 접두 구간 스펙트럼은
같은 간격과 시작 샘플을 쓰는 더 짧은 실행과 일치합니다. 필드 감쇠 노름은
여전히 갱신마다 평가합니다. 부분 샘플링한 스펙트럼이 안정된 것처럼 보여도
앨리어싱이 없음을 증명하지 않습니다. [샘플링 측정](../../validation/fdtd/fdtd-dft-sampling.md)을 보세요.
:::

### native 격자의 한 영역 기록하기

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`fdtd_grid(..., record_field=N, record_region=(z, y, x))`는 요청한 각 스냅숏에서 선택한
영역만 저장합니다. 각 항목은 정수 인덱스이거나 양의 간격을 가진 슬라이스이며,
음수 인덱스와 슬라이스 경계는 Python 인덱싱을 따릅니다.
정수 선택은 길이 1인 축을 유지합니다. 빈 영역, 역순 슬라이스,
범위를 벗어난 정수 인덱스, 정수가 아닌 항목은 거부합니다. 영역은
정적이며 `record_field`가 필요하고, native 격자 API에서 사용할 수 있습니다.

```python
# Earlier interface; current: wavebit.fdtd.native Snapshots(slices=...).
out = wavebit.fdtd_grid(
    epsilon, spacing=(dz, dy, dx), dt=dt, steps=steps,
    record_field=10, record_components=('ey', 'hx'),
    record_region=(slice(4, 20, 2), 0, slice(None)))
plane_history = out['fields']['ey']  # values: (time, 8, 1, nx)
```

선택은 NumPy, JAX, Torch 모두에서 스냅숏을 쌓기 전에 일어납니다. 보관하는
필드 이력 배열을 줄이지만, Maxwell 격자, 체크포인트
상태, 최종 E/H 배열, 주파수 모니터 누적기는 줄이지 않습니다. 원시 값, 타임스탬프,
좌표 벡터는 전체 도메인 기록에서 같은 영역을 선택한 것과 일치합니다.
Yee 엇갈림 배치가 보존되므로, 같은 셀 인덱스라도 필드 성분에 따라 물리
좌표가 다를 수 있습니다. z 간격이 비균일하면 공칭 dz로 좌표를 다시 만들지 말고
반환된 성분 좌표를 사용하세요.

예를 들어 `(12,4,6)` 격자에서 `(3,1,2)`개 셀을 선택하면 같은 성분과 스냅숏 수에서
필드 이력 값을 48배 적게 저장합니다. 이것은 배열
저장 비율이며, 최대 프로세스 메모리나 속도에 대한 주장이 아닙니다. 공간 부분 샘플링은
필터링이나 보간 없이 노드를 선택합니다. 영역 좌표는
미분할 수 없으며, 기록된 필드 값은 재료·소스 그래디언트를 유지합니다.
:::

### Stack 장면의 물리적 기록 범위

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

명시적 백엔드 `fdtd` 장면 API는 `x`, `y`, `z` 중 어느 것에서든 `(lower, upper)` 물리 좌표로 가는
비어 있지 않은 매핑 `record_bounds`를 받습니다. 이를 쓰려면
`record_field`와 `return_diagnostics=True`가 필요합니다. 범위는 유한한 실수여야 하며
`lower <= upper`를 만족해야 합니다. 지정하지 않은 축은 전체 범위를 유지합니다.

```python
# Earlier interface; current: wavebit.fdtd.periodic Snapshots(bounds=...).
result, diagnostic = wavebit.fdtd(
    stack, incidence, backend='jax', resolution=40,
    record_field=10, record_components=('ey', 'hx'),
    record_bounds={'z': (-0.2, 0.5), 'x': (-0.1, 0.1)},
    return_diagnostics=True)
```

반환되는 필드 기록과 같은 좌표를 사용하세요. x/y는 주기 셀의
중심을 원점으로 하고, z=0은 아래 경계면이며 양의 z는 위쪽입니다.
이 좌표는 아래 입사에서도 고정됩니다. y 불변 장면은
y=0만 노출합니다. 범위는 좌표 크기에 비례하는 기계 엡실론 8배의
부동소수점 허용 오차로 native 샘플 좌표를 양 끝 포함으로 선택합니다. 두 범위가 같으면
기존 좌표 평면 하나를 선택하며, 보간을 요청하는 것이 아닙니다. 시뮬레이션
밖으로 뻗은 범위는 시뮬레이션 샘플과의 교집합을 선택합니다.

요청한 각 성분은 모든 축에서 최소 하나의 샘플을 가져야 하며, 그렇지 않으면
더 넓은 범위나 더 세밀한 해상도를 제안하는 오류를 일으킵니다.
성분이 엇갈려 배치되므로 반환되는 좌표 배열과 모양이 다를 수 있습니다.
필드를 결합하거나 보간할 때는 각 성분 고유의 좌표를 사용하세요.
편의용 별칭 `field_e`/`field_z`는 여전히 기본 기록
성분을 따릅니다.

엔진은 선택한 성분 샘플을 덮는 가장 작은 공통 native 상자를 저장한 뒤,
엇갈린 개별 경계를 잘라냅니다. 공간 선택은
스냅숏 누적 전에 일어나며, 전체 도메인을 덮는 범위는 여전히 그 도메인을 기록합니다. 전체 시뮬레이션, 주파수 모니터, 재료
그래디언트는 원래 동작을 유지합니다. 범위는 정적 기하이며
최적화 매개변수가 아닙니다. `fdtd_grid`에서 격자 인덱스로 직접 선택하려면
대신 `record_region`을 사용하세요.
:::

### Fitting frequencies and decay rates from a time trace

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`wavebit.compat.legacy.fit_fdtd_modes(t, signal, modes=...)`는 소스가 무시할 수 있을 만큼 작아진 뒤
한 점·한 성분을 분석합니다. 현재 대응 기능은 `wavebit.observations.fit_modes`입니다. 이것은
균일하게 샘플링한 어떤 시계열에도 쓸 수 있는 NumPy 후처리이므로, 현재 `fdtd.TimeSeriesMonitor`의 시계열
(`result[name].times_seconds`(초), `.values`)에도 쓸 수 있습니다. 이때 주파수와 감쇠율은
Hz와 1/s 단위로 나옵니다. 기록된 필드에서 시계열을 선택하세요.

```python
# Earlier interface; current: wavebit.observations.fit_modes on wavebit.fdtd.periodic Snapshots.
field = diagnostics['fields']['ey']
signal = np.asarray(field['values'])[:, 0, 0, 0]
fit = wavebit.fit_fdtd_modes(
    np.asarray(field['t']), signal, modes=2, start=60., stop=120.,
)
print(fit['frequency'], fit['decay_rate'], fit['relative_residual'])
```

이 예제는 기록 구간에 60부터 120까지가 포함되어 있고 그 구간에서
소스가 무시할 만하다고 가정합니다. 인덱스는 그 성분 고유의
엇갈린 좌표를 사용해 고르세요. 분석은 NumPy CPU 배열을 사용하므로, JAX/Torch 데이터는
명시적으로 호스트로 옮기고, 그래디언트를 지닌 Torch 기록은 분리하세요. 이 분석은
시뮬레이션을 바꾸지 않으며 피팅을 거친 그래디언트도 제공하지 않습니다.

모델은 `amplitude * exp((-decay_rate-2j*pi*frequency)*(t-time_origin))`의 합입니다.
진폭은 선택한 첫 샘플을 기준으로 합니다. `frequency`는 c=1
시간 단위당 주기 수이며, 양의 `decay_rate`는 에너지 감쇠가 아니라 진폭 감쇠를 나타냅니다.
`quality_factor`는 `pi*abs(frequency)/decay_rate`입니다. 양수가 아닌 감쇠는
NaN Q를 만들며, 확인할 수 있도록 남겨 둡니다. 피팅된 감쇠가 매우 작으면 반올림 오차일 수 있으므로,
큰 Q에는 독립적인 검사가 필요합니다. 실수 코사인에는 켤레
주파수가 둘 다 필요하므로, 감쇠 코사인 하나에 모드 두 개를 사용하세요. 복소 기록은
켤레 쌍을 포함할 필요가 없습니다. 기본값은 지수 함수 하나를 요청합니다.

선택한 샘플은 균일한 간격이어야 합니다. 주파수는 부호 있는
나이퀴스트 가지를 사용하며, 이 함수는 앨리어싱된 기록을 필터링하거나 복구하지 않습니다. 최소
`2*modes+1`개 샘플이 필요합니다. 선택적 `pencil`은 Hankel 행렬의
열 개수를 `modes`와 `samples-modes` 사이에서 고릅니다. 기본값은
`min(samples//2, max(128, 2*modes))`이며, 저장 공간은
`samples*pencil`에 비례합니다. `rcond=1e-10`은 상대 특잇값 하한입니다. 분해되지 않은
모드 수와 진폭 기저는 거부합니다. 이 하한은 물리적 잡음
추정치나 모드 주파수 불확도가 아닙니다.

결과에는 주파수 순으로 정렬된 모드, 정규화된 `singular_values`,
`basis_condition`, 선택한 `t`, 재구성한 `fitted` 시계열이 들어 있습니다.
`relative_residual`은 이 구간에서만의 피팅 오차입니다. 공진을 식별하기 전에
서로 다른 시작·끝 시각, 모드 수, pencil 폭을 비교하세요.
대수적으로 감소하는 꼬리는 잔차가 작으면서도 주파수와 감쇠율이 바뀌는
여러 지수 함수로 근사될 수 있습니다. 이것은 Q를 증명하지도 않고,
DFT를 무한 시간으로 자동 외삽하는 것을 정당화하지도 않습니다.
[링다운 측정](../../validation/fdtd/fdtd-ringdown.md)은 회절
문턱 근처에서 이 한계를 보여 주고, 정확한 모드와 외부 Meep 기록에 대해 피팅기를 검증합니다.
:::

### 남은 재료 진동 확인하기

:::{admonition} 이전 인터페이스
:class: earlier-interface

과거 인터페이스이며 2026-10-15에 `wavebit.compat`과 함께 퇴역했습니다. 아래의 이름과 기본값은 현재 API가 아니며, 그 코드는 과거 체크아웃에서만 실행됩니다.

`convergence_window`를 켜면 native 수렴 매핑은 모든 ADE 극점의
활동도도 입력 극점 순서대로 보고합니다. 각 E 성분 제어
부피에서 다음을 평가합니다.

$$
A_j=\sum_{\text{cells, components}} \Delta V\,
\frac{\omega_{0,j}^2 |P_j|^2+|(P_j-P_{j,\mathrm{previous}})/\Delta t|^2}{s_j},
$$

여기서 $s_j$는 극점 세기이며, 세기가 0인 셀은 0을 기여합니다.
이 양의 진동자 활동도는 분극과 그 이산
속도를 결합합니다. 둘은 서로 다른 시간 수준에서 평가되므로, 보존되는
이산 에너지나 완전한 전자기·PML 에너지 측정이 아닙니다. Drude
극점에서는 omega0=0이므로, 정적인 분극 오프셋은 그 속도가
사라지면 활동도가 없습니다.

| 필드 | 의미 |
| --- | --- |
| `material_activity_peak` | 전체 실행에서 각 극점의 최대 활동도 |
| `material_activity_tail` | 요청한 마지막 창에서 각 극점의 최대 활동도 |
| `material_signal_present` | 각 극점의 활동도 최대값이 유한하고 양수인지 여부 |
| `material_decay` | 여기된 각 극점에 대한 꼬리/최대값의 제곱근. 여기되지 않은 극점은 NaN |
| `material_decayed` | 재료/CFL 입력이 유효하고 활동도가 유한하며, 여기된 모든 극점이 `material_decay <= convergence_tol`인지 여부 |
| `state_decayed` | `field_decayed`와 `material_decayed`가 모두 참인지 여부 |

극점이 없으면 극점별 배열은 비어 있고 `material_decayed=True`입니다. 여기되지 않은
극점은 재료 플래그를 막지 않습니다. 여기된 각 극점은 자신의 최대값을 사용하므로,
약하게 여기된 느린 진동자도 이 보수적인 플래그를 거짓으로 유지할 수 있습니다.
결합 플래그는 여전히 0이 아닌 관측 필드와 비활성 소스를 요구합니다.
CPML 보조 기억을 검사하거나, 공간 정확도를 확립하거나,
앞으로의 동작을 예측하지는 않습니다.

장면 진단은 두 소스 기저에 대한 실제 실행과 기준 실행 모두를 합산한
`material_decayed`와 `state_decayed`를 보여 주며, 극점별 값은
`convergence['bases']`에 남습니다. 기존의 `field_decayed`와 `amplitudes_settled`는
각자의 별도 의미를 유지합니다. 모든 진단은 선택형이고, 고정 단계이며, 백엔드
native로 남습니다. 활동도 축약 연산은 극점 수와
격자 셀 수에 비례하는 작업을 추가하며, 시간 루프 동안 극점마다 스칼라 두 개만 보관합니다. 비율과
최대값은 미분 가능한 곳에서 그래디언트를 유지합니다.

보정된 진폭이 재료 운동이 감쇠하기 전에 안정되는 사례와,
격자 오차 및 재료 매개변수 민감도에 대한 별도 검사는
[저손실 혼합 금속 측정](../../validation/fdtd/fdtd-mixed-metal.md)을 보세요.
:::
