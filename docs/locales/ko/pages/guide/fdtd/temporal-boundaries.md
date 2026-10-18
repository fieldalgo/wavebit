# 시간 경계: 실행 도중 바뀌는 유전율

시간 경계는 공간 경계면을 시간 축으로 옮긴 것입니다. 어떤 영역의 유전율이 한 순간에
`epsilon_before`에서 `epsilon_after`로 바뀌면, 파동은 파수를 유지한 채 주파수가 바뀌고
전진파와 후진파로 나뉩니다. RCWA를 포함해 한 주파수만 다루는 방법으로는 이 현상을
표현할 수 없습니다. `fdtd.SwitchedMaterial`이 이런 매질을 선언합니다.

```python
import numpy as np
from wavebit import Units, fdtd
from wavebit.units import C0

medium = fdtd.SwitchedMaterial(epsilon_before=1.0, epsilon_after=4.0,
                               switch_time=0.3e-6 / C0)        # 초 단위
problem = fdtd.Problem(
    units=Units(length='um'),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.01, .01, 1.)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
        structures=(fdtd.Structure('switched', fdtd.Slab(z=(-.5, .5)), medium),)),
    initial=fdtd.InitialField(E=(None, lambda x, y, z: np.cos(2 * np.pi * z), None)),
    outputs=fdtd.Outputs(monitors=(fdtd.Record('p', at=(0., 0., 0.), component='Ey'),)))
result = fdtd.run(problem, numerics=fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.01),
                                                  duration=fdtd.Duration(steps=430)))
```

## 무엇이 연속인가

갱신식은 전기 변위로 씁니다. `dD/dt = curl H`, `D = epsilon(t) E`입니다. 유전율이 바뀌는
순간에 `D`와 자기장은 연속이고 `E`는 연속이 아닙니다. 진폭 `E0`, 굴절률 `n1`, `n2`인
전진파에 두 조건을 맞추면 다음이 나옵니다
([Morgenthaler 1958](https://doi.org/10.1109/TMTT.1958.1124533);
[Xiao, Maywar, Agrawal 2014](https://doi.org/10.1364/OL.39.000574)).

```text
E_forward  / E0 = n1 (n1 + n2) / (2 n2^2)
E_backward / E0 = n1 (n1 - n2) / (2 n2^2)
omega_after     = omega_before * n1 / n2
```

`epsilon 1 -> 4`에서는 `0.375`와 `-0.125`이고 주파수는 절반이 됩니다.

같은 변화를 `epsilon(t) dE/dt = curl H`로 쓰면 `D` 대신 `E`가 연속이 됩니다. 변화 시점
밖에서는 맞는 식이지만 그 시점에서는 틀린 식입니다. 주파수와 속도는 여전히 맞게 나오지만,
두 진폭은 해상도와 관계없이 위 값의 정확히 두 배가 됩니다. 회귀 시험은 이 변이가 계속
실패하는지 확인합니다.

## 지원 범위

재료는 손실이 없고 등방이며, 유전율이 한 번만 바뀝니다. 두 유전율은 JAX와 PyTorch에서
추적·미분할 수 있습니다. 변화 시각은 정적인 초 단위 값입니다. 변화는 `switch_time`과 같거나
그 뒤의 첫 E 시간 단계에서 일어나므로, `switch_time`은 실행 시작에서 적어도 두 시간 단계
뒤여야 합니다.

다음 경우에는 보정이 정확하지 않으므로 구조를 거절합니다.

| 거절 | 이유 |
|---|---|
| 평활화된 도형(`Box`, `Sphere`, `Cylinder`, `Prism`) | 경계 셀에 두 매질이 섞이는데 한쪽만 바뀝니다. `fdtd.Slab`이나 `fdtd.Region`을 쓰세요 |
| 다른 구조와 겹침 | 다른 구조가 칠한 셀까지 바뀝니다 |
| 바뀌는 `background`, 면으로의 `continuation` | 한 매질에 맞춘 흡수층은 다른 매질에 맞지 않습니다 |
| 흡수층에 닿음 | 흡수층이 보정의 기준인 갱신식을 감쇠시킵니다 |
| 영역 안의 광원 | 주입하는 장이 고정된 한 매질 기준으로 만들어집니다 |
| 영역 안의 `Diffraction`·`FarField` 판독 | 결과를 고정된 한 매질의 임피던스로 나눕니다 |
| `discretisation='stencil'` | 보정의 기준인 계수를 바꿉니다 |

`Record`, `Flux`처럼 영역 밖에서 읽는 판독은 영향을 받지 않습니다.

## 검증

`tests/fdtd/test_fd_switched.py`가 솔버를 위 해석해와 비교합니다.

| 확인 항목 | 결과 |
|---|---|
| 전진·후진 진폭, `eps 1 -> 4`, 주기 경계 직선 위 파속, 960/1920/3840 셀 | 오차 4.9e-5, 1.2e-5, 3.1e-6; 차수 2.00 |
| 공개 API의 정상파 진폭, 간격 0.02/0.01/0.005 µm | 오차 2.0e-4, 5.2e-5, 1.3e-5; 차수 1.9-2.0 |
| `eps 1 -> 4` 뒤의 주기 | 1.5e-3 이내로 2 |
| `E` 연속 변이 | 두 진폭 모두 해석해의 2.000배 |
| NumPy 대비 JAX·PyTorch | JAX는 최댓값의 1.4e-14, PyTorch는 비트 단위로 같음 |
| JAX·PyTorch의 `d/d epsilon_after`, `d/d epsilon_before` | 중심 차분과 상대 1e-6 이내로 일치 |

## 한계

- 내부 에너지 진단은 이 영역을 줄곧 `epsilon_before`로 가중합니다. 변화 뒤에는 그 값이 그
  영역의 장 에너지가 아닙니다. 시간 경계는 실제로도 에너지를 바꾸므로, 에너지 판독으로는 이
  기능을 검증할 수 없습니다.
- 구조마다 변화는 한 번입니다. 연속 변조(광자 시간 결정)와 상승 시간이 있는 변화는 아직
  제공하지 않습니다.
- 이 기능은 Meep·FDTDx와 비교하지 않았습니다.
