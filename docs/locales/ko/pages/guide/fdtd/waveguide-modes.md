# 도파관 모드

`fdtd.solve_modes`는 **실행이 쓰는 그 격자 위에서** 단면의 도파 모드를 찾습니다.
`fdtd.prepare`와 똑같이 문제를 준비하고, 시간 전진이 면 위에서 곱하는 유전율과
투자율을 읽어, 이산 갱신식 자체의 고유값 문제를 풉니다. 그래서 돌려준 모드는
연속계가 아니라 실행의 모드입니다. 실행 안에서 출발시키면 반올림 수준까지 자기
자신으로 남고, 연속계 모드와의 차이는 격자의 이산화 오차입니다.

```python
from wavebit import Material, Units, fdtd

problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
    domain=fdtd.Domain(size=(3., 3., 0.4)),
    boundaries=fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.Periodic()),
    structures=(fdtd.Structure('core', shape=fdtd.SnappedBox(size=(1., .5, 10.)),
                               material=Material(epsilon=4.)),)))
numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=1))
modes = fdtd.solve_modes(problem, numerics=numerics, plane=fdtd.Plane(z=0.),
                         wavelength=1.0, count=2)
modes[0].effective_index        # 격자 자체의 분산으로 계산한 beta c / omega
modes[0].E['Ex'], modes[0].H['Hy']   # 각 성분의 Yee 점에서 V/m, A/m
modes.axes, modes.coordinates['Ex']  # ('y', 'x')와 그 점들의 좌표
```

## 입력

- `problem`, `numerics`: 실행과 같은 선언. 장면과 격자만 쓰고 광원·측정기는 무시합니다.
  시간 간격은 중요합니다. leapfrog의 `Omega = (2/dt) sin(omega dt/2)`가 `omega`를
  대신합니다.
- `plane`: 전파 축에 수직인 `fdtd.Plane`. 위치는 가장 가까운 노드 면으로 맞춰지며, 그
  면은 축을 따라 매질이 균일한 물리 영역 안에 있어야 합니다(양쪽 한 층씩 비교).
  가로 범위를 주지 않으면 창은 물리 단면 전체입니다. 범위(`fdtd.Plane(z=0., x=(-1., 1.))`)는
  노드 면으로 맞춰지고 그 자리를 벽으로 닫습니다.
- `wavelength`: `Problem.units` 단위의 진공 파장.
- `count`: 유효 굴절률이 큰 순서로 몇 개의 모드를 받을지. 갇힌 모드가 요청보다 적으면
  찾은 개수를 알려 주며 거부합니다(`mode.count`).
- `backend`: 돌려줄 배열의 종류 `'numpy'`, `'jax'`, `'torch'`. 고유값 계산은 백엔드와
  상관없이 호스트(SciPy)에서 한 번 실행하므로 세 경우 모두 비트까지 같은 값을 돌려줍니다.

## 출력

`WaveguideModes`는 모드들(`modes[0]`이 굴절률이 가장 큼), 면의 `axis`와 맞춰진
`position`, `wavelength`, 저장 순서로 된 두 배열 축 `axes`(z 면이면 `('y', 'x')`),
각 성분의 `coordinates`를 담습니다. 각 `WaveguideMode`에는 다음이 있습니다.

- `effective_index`: `beta c / omega`. `beta`는 격자 자체의 분산
  `q = (2/dx) sin(beta dx/2)`에서 구하며, `beta_per_meter`는 rad/m 단위의 `beta`입니다.
- `E`, `H`: 성분 자기 Yee 점 위의 V/m, A/m 복소 배열. 가로 성분은 실수, 세로 성분은
  허수입니다. `exp(-i omega t)` 실행의 페이저이며, 각 성분은 자기 시각과 자기 위치
  기준입니다. E는 노드 면, 축을 따라 반 칸 떨어진 H 성분은 그 반 칸 면 기준입니다.
- `polarization`: 가로 전기 에너지 중 각 가로 축 방향의 몫(x 방향 편광 모드면
  `{'x': 0.98, 'y': 0.02}`).
- `power_watts`: 1. `FluxMonitor`가 그 모드에 대해 잴 플럭스(노드 면의 E와 두 H 반 칸
  면의 평균)를 `+axis` 방향 1 W로 정규화합니다(`InvariantAxis` 하나면 W/m, 둘이면 W/m²).

모드들은 창 위에서 켤레 없는 교차곱으로 직교합니다. 굴절률이 다른 두 모드는 격자
위에서 정확히 `sum (E_u H_v - E_v H_u) dA = 0`입니다. 겹친 쌍(정사각 도파관의 두 편광)은
이미 직교화해서 돌려줍니다.

## 가로 면

| 면 | 처리 |
| --- | --- |
| `PEC`, `PMC` | 커널 자체의 벽, 커널이 두는 자리(PMC 벽은 아래쪽 면에서 반 칸 바깥). |
| `Periodic` | 창이 축 전체를 덮으면 주기 조건. |
| `Open` | 물리 영역 가장자리에서 한 칸 안쪽의 벽(창의 가장자리 노드가 벽이며, 창 위의 플럭스 면은 흡수층을 읽으면 안 됨). 가장 바깥 칸에서의 가로 E가 최고값의 1e-3 이하인 모드만 받습니다. 누설 모드, 차단에 가까운 모드, 잘린 상자의 모드는 받지 않습니다. |
| `InvariantAxis` | `k = 0`일 때 1차원(평판) 극한. |
| `Bloch`, `MirrorSymmetry`, `k != 0`인 `InvariantAxis` | 거부(`mode.boundary`). |

## 거부하는 것

각 거부는 코드를 지닌 구조화된 오류입니다.

- `mode.material`: 창 안의 전도성·분산 매질, 또는 장면 어딘가의 결합 평활(constitutive)
  재료나 `discretization='stencil'`. 칸 맞춤 물체(`SnappedBox`, `SnappedSlab`)와 돌리지
  않은 평활 `Box` 면은 대각 매질로 내려가므로 받습니다. `Cylinder`, `Prism`, `Sphere`,
  돌린 `Box`는 성분을 결합하므로 받지 않습니다.
- `mode.nonuniform`: 면에서 축을 따라 매질이 바뀜.
- `mode.traced`: 재료·파장·범위 지정 기하 안의 JAX tracer나 기울기를 지닌 Torch 텐서.
  모드 계산은 미분하지 않으니 구체값으로 푸세요.
- `mode.outside_domain`, `mode.plane`: 물리 영역 밖의 면, `InvariantAxis`를 가로지르는 면,
  비었거나 밖으로 나간 범위.
- `mode.count`, `mode.convergence`: 요청보다 적은 갇힌 모드, 잔차가 1e-8을 넘는 고유쌍.

## 실행에서 모드 재기: `ModeMonitor`

```python
outputs = fdtd.Outputs(monitors=(
    fdtd.ModeMonitor('out', plane=fdtd.Plane(z=3.), wavelengths=(1.5, 1.55), modes=2),))
spectrum = fdtd.run(problem.replace(outputs=outputs), numerics=numerics)['out']
spectrum.forward_energy, spectrum.backward_energy   # (파장, 모드), J/Hz
spectrum.forward, spectrum.backward                  # 복소 진폭, sqrt(J/Hz)
spectrum.effective_index, spectrum.modes[0][0]       # 사영에 쓴 모드
```

측정기는 각 파장에서 `solve_modes`와 똑같이 면의 모드를 풀고, 실행의 푸리에 변환 장을 그
모드들에 사영합니다. 노드 면의 E와 두 H 반 칸 면의 평균을 짝짓는데, `FluxMonitor`와 같은
방식입니다. 모드들이 격자 위에서 정확히 직교하므로, 사영은 다른 모드(소멸파 모드 포함)의
섞임 없이 각 모드와 각 방향을 나눕니다. `forward_energy`와 `backward_energy`는 면의 `+`·`-`
법선 쪽으로 나르는 양측 펄스 스펙트럼 에너지이며 단위는 `FluxMonitor.spectral_energy`와
같습니다. 사영한 모드들만으로 이루어진 장이라면 모드에 대한
`sum(forward_energy - backward_energy)`가 플럭스입니다. 투과율은 기준 실행의 같은 양(모드
광원이 있으면 그 주입 스펙트럼)으로 나누어 얻습니다.

면은 도파관의 균일한 구간을 가로질러야 하며, 창·면·거부는 `solve_modes`와 같습니다. 면에
닿는 구조는 구체값이어야 합니다. 다른 구조는 추적된 값을 지닐 수 있으므로, 예를 들어 광원과
측정기 사이의 산란체를 통해 에너지를 미분할 수 있습니다(중앙 차분과의 비교는
[검증 기록](../../validation/fdtd/fd-waveguide-modes.md)). `Open` 면을 가로지르면 창은 측정기와
`solve_modes` 모두 물리 가장자리에서 한 칸 안쪽에서 멈추고, 측정기는 흡수층 칸을 읽지 않습니다.

## 정확도

등록한 검사와 측정 결과는 [검증 기록](../../validation/fdtd/fd-waveguide-modes.md)에
있습니다. 요약하면, 균일 매질은 자기 굴절률과 Yee 분산의 닫힌 식을 1e-12까지 주고,
속이 빈 금속 도파관은 이산 닫힌 식을 1e-12 정도까지 줍니다. 대칭 평판은 정확한 TE·TM
분산에 2차로 수렴하고, 출판된 유전체 모서리 기준값에도 2차로 다가가며, 풀어낸 모드를
실행에서 출발시키면 반올림 수준까지 자기 자신으로 남습니다. 유한 격자에서의 유효
굴절률은 그 격자의 값이므로, 다른 FDTD 결과처럼 간격을 줄여 수렴을 확인하세요.
