# 일반 FDTD 엔진으로 계산하는 주기 메타표면

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 필요합니다.
> 이 네임스페이스는 현재 설치본에 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행하지 않았습니다.


:::{note}
이 페이지의 FDTD 예제는 과거 측정에 쓴 정규화 선언을 `wavebit.compat.legacy.fd`를 거쳐 그대로 보존합니다. RCWA 비교는 현재 `wavebit.rcwa`를 씁니다. 현재 FDTD API는 이제 명시적인 `Structure.continuation`으로 고정 Box 기판을 지원합니다. 그 물리 단위와 선언 제약은 [스무딩 기판](substrate-gradients.md)을 참조하세요. 이 페이지의 측정은 그 API로 다시 실행하지 않았습니다.
:::

`fd`는 평면파가 어떤 각도로든 비추는 주기 배열의 셀 하나를 풀고, 거기서 나가는 빛을 회절 차수로 읽습니다.
각 차수의 고유한 s/p 좌표계에서의 복소 진폭과 효율을, 이전 `wavebit.compat.legacy.rcwa`가 반환하는 것과 같은
`Result` 배치로 돌려줍니다. 이 페이지는 그런 셀을 만들고, 주기와 각도를 고르는 방법, 차수를 읽는 방법,
그리고 RCWA와 비교하는 방법을 설명합니다. 그 근거가 되는 측정은
[`fdtd-metasurface-validation.md`](../../validation/fdtd/fdtd-metasurface-validation.md)에 있습니다.

## 격자 셀

유리 위에 TiO2 같은 기둥을 0.6 파장 간격으로 두고 30도로 비춥니다. 이를 투과하는 빛은 `(0, 0)`과 `(-1, 0)`
두 차수로 나가며, 두 번째 차수의 몫은 대역에 걸쳐 줄어듭니다(입사 s 전력의 0.95에서 60%, 1.0에서 21%,
1.05에서 13%).

```python
# Earlier normalized declarations retained with the measurements on this page.
from wavebit.compat.legacy import fd

N = 25                                    # cells per vacuum wavelength
DX = 1.0 / N
GLASS = fd.Material(eps=2.13)
H = 0.625                                 # pillar height; the substrate's face is z = 0

pw = fd.PlaneWave('pw', z=H + 0.6, direction='-z', theta=30.0, phi=0.0,
                  polarization='s', band=fd.Band(wavelength=1.0, df=0.06))
substrate = fd.Structure(fd.Box(size=(10.0, 10.0, 3.0), center=(0.0, 0.0, -1.5)), GLASS)
pillar = fd.Structure(fd.Cylinder(radius=0.15, height=H, axis='z', center=(0.0, 0.0, H / 2)),
                      fd.Material(eps=5.8))
wavelengths = (0.95, 1.0, 1.05)
Z_LO, Z_HI = -0.8 - 24 * DX, H + 1.4 + 24 * DX   # 0.3 past each monitor, then the absorber
scene = fd.Scene(
    domain=fd.Domain(size=(0.6, 0.6, Z_HI - Z_LO), center=(0.0, 0.0, (Z_HI + Z_LO) / 2)),
    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=(
        fd.PML(cells=24, reflection=1e-12, background=GLASS),
        fd.PML(cells=24, reflection=1e-12))),
    resolution=fd.Resolution(dx=DX), run=fd.Run(periods=120),
    structures=(substrate, pillar), sources=(pw,),
    monitors=(fd.Diffraction('r', z=H + 1.1, side='reflect', wavelengths=wavelengths,
                             reference_plane=H, incident_plane=H),
              fd.Diffraction('t', z=-0.5, side='transmit', wavelengths=wavelengths,
                             reference_plane=0.0, incident_plane=H)))

res = fd.run(scene, 'numpy')
t = res['t'].orders
print(res['r'].theta_at)                  # each wavelength's own angle, degrees
print(t.order(-1, 0).T, t.order(0, 0).T)  # efficiencies per wavelength
```

각 부분이 하는 일은 다음과 같습니다.

* **셀 하나, 소스에서 가져온 Bloch 면.** 영역은 x와 y 방향으로 한 주기이며, `fd.Bloch.from_source(pw)`가
  그 면들에 평면파 자체의 위상을 줍니다.
* **기판은 아래쪽 흡수층을 관통하며**, 그 흡수층에는 무엇을 흡수하는지 알려 줍니다:
  `fd.PML(..., background=GLASS)`. 기판은 셀보다 넓은 스무딩 `fd.Box`이므로 `z = 0` 아래의 모든 층을 정확히
  채웁니다.
* **기둥은 기판 위에 맞닿아 섭니다.** 기둥의 밑면은 기판의 면이며, 둘을 함께 담은 셀은 기본값인
  `Structure(junction='split')`로 처리합니다([`fdtd-differentiation.md`](fdtd-differentiation.md)와
  `fd.check(scene)` 참조).
* **두 모니터는 각각 균일한 매질 하나 안에 놓이고** -- 소스 위의 공기, 기둥 아래의 유리 -- RCWA와 같은 곳을
  진폭 기준으로 삼습니다. `r`은 기둥의 윗면, `t`는 기판의 면이며, 둘 다 위쪽의 입사파(`incident_plane=H`)를
  기준으로 합니다.
* **흡수층은 기본값이 아니라 목표 1e-12의 셀 24개입니다.** 비스듬한 차수는 `cos theta`가 0.55 근처에서
  나가고, 흡수층은 대략 목표값의 `0.919 cos theta` 제곱을 되돌려 보내기 때문입니다.

## 주기와 각도 고르기

`fd`의 평면파는 대역 전체에서 횡방향 파수 벡터를 고정합니다. 지정한 각도는 대역의 중심 파장에서 정확하고,
다른 파장은 각자의 각도를 가집니다(`res['r'].theta_at`). 그러면 차수 `(m, n)`의 횡방향 파수 벡터는

```
q = (sin(theta) cos(phi) + m / Px,  sin(theta) sin(phi) + n / Py)
```

이며(중심 파장의 진공 파수 단위), 파장 `lambda`에서 `lambda |q| < n_med`이면 굴절률 `n_med`인 매질에서
전파합니다. `lambda |q| = n_med` 근처에서는 두 가지가 잘못되며, 계획 단계는 그럴듯한 숫자를 반환하는 대신 두
경우 모두 거부하거나 경고합니다.

* 표면에 가깝게 나가는 차수(작은 `cos theta`)는 잘 흡수되지 않습니다 -- 기본 흡수층에서 `cos theta = 0.41`
  아래이면 계획이 거부합니다.
* 대역 안에서 차단점을 넘는 차수(레일리 이상)는 실행이 자리 잡는 데 오래 걸리게 하고, 답을 파장에 대해
  날카롭게 만듭니다.

긴 실행 전에, 대역 전체에서 모든 차수가 분명히 안에 있거나 분명히 밖에 있는 주기와 각도를 1분쯤 찾아볼
가치가 있습니다.

```python
import math

def layout(Px, Py, theta, phi, n_sub, wavelengths=(0.95, 1.0, 1.05), reach=4):
    s = math.sin(math.radians(theta))
    q0 = (s * math.cos(math.radians(phi)), s * math.sin(math.radians(phi)))
    for side, n_med in (('r', 1.0), ('t', n_sub)):
        for m in range(-reach, reach + 1):
            for n in range(-reach, reach + 1):
                u = math.hypot(q0[0] + m / Px, q0[1] + n / Py)
                ratios = [lam * u / n_med for lam in wavelengths]
                if min(ratios) < 1:
                    print(side, (m, n), 'propagates' if max(ratios) < 1 else 'CROSSES CUTOFF',
                          [round(r, 3) for r in ratios])

layout(0.6, 0.6, 30.0, 0.0, math.sqrt(2.13))
```

비율이 약 0.87보다 크거나(`cos theta`가 0.5 미만) 1보다 몇 퍼센트 이내로 큰 곳이 문제가 되는 곳입니다.

## 차수 읽기

`res['r'].orders`와 `res['t'].orders`는 `wavebit.observations.diffraction.Result`입니다.

* `.order(m, n)`은 한 차수의 숫자들을 주며, 각각은 모니터의 파장들에 대한 배열입니다. 효율 `R` / `T`와
  그 s/p 성분, 그리고 그 차수 고유의 s/p 좌표계에서 RCWA의 p 부호 규약을 따르는 진폭 `r_s`, `r_p`(반사)
  또는 `t_s`, `t_p`(투과)입니다.
* `.R`, `.T`와 진폭은 모양이 `(wavelengths, n_v, n_u)`인 전체 배열이기도 하며 -- `v`(y) 차수 번호가
  먼저 -- `.orders_v`와 `.orders_u`로 표지가 붙습니다. 위치로 인덱싱하지 말고 `.order(m, n)`을 쓰세요.
* 기본적으로 모니터는 자신이 놓인 매질에서 그 파장 중 어느 하나에서라도 전파하는 모든 차수를 유지합니다.
  `orders=`로 그 창을 넓히거나 좁힙니다.
* 모니터에 `polarizations='both'`를 주면 다른 편광으로 장면을 한 번 더 실행하고 `.incident` 옆에 `.s_inc`와
  `.p_inc`를 채웁니다.

현재 `fdtd.DiffractionMonitor`는 같은 선택을 `DiffractionOrders`로 반환합니다. `.order(m, n)`은 파장들에 대해
`efficiency`, `amplitude_ratio_s`, `amplitude_ratio_p`(측정한 입사 필드에 대한 비, 단위 없음)를 주고, 전체 배열은
같은 `(wavelengths, n_v, n_u)` 모양이며, 각 파장의 각도는 `metadata['theta_radians']`입니다. 아직
`polarizations='both'`는 없습니다.

## RCWA와 비교하기

같은 셀을 파장마다 그 파장 고유의 각도에서 `wavebit.rcwa`로 푸세요.

```python
import math
import numpy as np
from wavebit import Material, Units, rcwa

theta_at = np.asarray(res['r'].theta_at)
for i, lam in enumerate(wavelengths):
    problem = rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(
            period=(0.6, 0.6), top=Material(epsilon=1.0), bottom=Material(epsilon=2.13),
            layers=(rcwa.Layer(thickness=H, background=Material(epsilon=1.0), regions=(
                rcwa.Region(geometry=rcwa.Circle(radius=0.15), material=Material(epsilon=5.8)),)),)),
        excitation=rcwa.Excitation(wavelength=lam, sources=(rcwa.PlaneWave(
            side='top', polarization='s', theta=math.radians(theta_at[i]), phi=0.0),)))
    ref = rcwa.run(problem, numerics=rcwa.Numerics(orders=(15, 15)))
    print(lam, t.order(-1, 0).t_s[i], complex(ref.transmission.order(-1, 0).amplitude_ratio_s[0]))
```

`polarization='s'`는 s 입사, `'p'`는 p 입사이며, 복소 `(a_s, a_p)` 쌍은 둘의 임의 혼합입니다.

**위상에는 기둥의 위치가 중요합니다.** 두 엔진 모두 셀 중심에서 차수의 진폭을 표시하고, RCWA는 형상도
거기서부터 배치하지만, `fd`는 기둥을 `center=`가 말하는 곳에 둡니다. 기둥이 중심에서 `(xc, yc)`에 있고 RCWA
형상이 `(0, 0)`에 있으면, 차수 `(m, n)`의 진폭은 `exp(-2 pi i (m xc / Px + n yc / Py))`만큼 다릅니다. 효율과
0차는 영향을 받지 않습니다. RCWA 원에 같은 `center=(xc, yc)`를 주거나, 비교하기 전에 FDTD 진폭에
`exp(+2 pi i (m xc / Px + n yc / Py))`를 곱하세요.

**RCWA에도 오차가 있습니다.** 비교하는 숫자의 움직임이 판단하려는 차이보다 훨씬 작아질 때까지
`Numerics(orders=...)`를 올리세요. 고굴절률 직사각형은 RCWA의 형상 경로에서 느리게 수렴합니다. 픽셀
모서리가 직사각형의 모서리와 일치하는 래스터(`rcwa.Layer(epsilon=...)`)에서
`Numerics(fourier_coefficients='integrated', factorization='li')`를 쓰면 더 빨리 수렴하며, 좁은 변에도 긴
변만큼 많은 차수가 필요합니다.

## 계산 격자는 얼마나 촘촘해야 하는가

C7 캠페인은 셀 세 개를 각각 세 가지 분해능에서 RCWA와 비교해 측정했습니다. 숫자는 모든 전파 차수, 두 출사
편광, 두 입사 편광, 세 파장에 걸친 최악의 오차입니다 -- 전형적인 셀이 아니라 가장 나쁜 셀입니다.

| 기둥 | 파장당 셀 수 | 최악 진폭 오차 | 최악 효율 오차 |
| --- | --- | --- | --- |
| 유리 위 epsilon-4 원기둥 | 20 / 40 / 80 | 2.9e-2 / 7.4e-3 / 2.1e-3 | 1.7e-2 / 4.6e-3 / 1.4e-3 |
| 유리 위 epsilon-5.8 원기둥 | 25 / 50 / 100 | 7.4e-2 / 2.0e-2 / 6.1e-3 | 1.8e-2 / 4.5e-3 / 1.1e-3 |
| 유리 위 epsilon-12 상자, conical | 40 / 80 / 120 | 2.2e-2 / 7.1e-3 / 3.6e-3 | 1.2e-2 / 4.1e-3 / 2.2e-3 |

"파장당 셀 수"는 진공 파장 기준입니다. 계획 단계는 가장 밀한 재료 *안에서* 파장당 셀이 8개보다 적으면
거부하고(따라서 이 대역에서 epsilon 5.8은 25, epsilon 12는 40이 하한입니다), 12개 미만이면 경고합니다. 두
원기둥의 가장 큰 오차는 2차로 줄어듭니다 -- 분해능을 두 배로 할 때마다 약 4분의 1이 됩니다. epsilon-12
상자의 오차는 1.5에서 1.75차로 줄어들어, 두 배로 할 때마다 약 3분의 1이 됩니다(그 이유에 대해 알려진 것과
알려지지 않은 것은 [`fdtd-metasurface-validation.md`](../../validation/fdtd/fdtd-metasurface-validation.md)에
있습니다). 이 느린 수렴률은 상자가 유리와 접촉하는 데서 오는 것이 아닙니다. 같은 상자를 기판에서 0.1 띄워도
같은 수렴률로 수렴하며, 오차는 1.3배(진폭), 1.6에서 1.7배(효율) 큽니다. 상자가 유리 위에 서 있을 때 접촉이
자체 오차를 더하는지는 알 수 없습니다. 상자를 띄우면 밑면 모서리도 공기 중에 놓이기 때문입니다. 기둥 아래에
틈을 남겨도 수렴 차수를 되찾을 수는 없습니다. 2D 능선으로 실행하면 상자의 xz 단면은 파장당 셀 320개까지
2차로 수렴합니다. yz 단면은 640개까지 개별 진폭의 수렴이 더 느립니다. 파장 0.95에서 s-s 투과 진폭은 마지막
두 배 단계에서 기준값 기반 수렴 차수가 약 1.52이고, 160/320/640에서 기준값 없는 자기 수렴 차수가 약
1.59입니다. 사전 등록한 시간·흡수층 검사가 그 낮은 차수를 분해합니다. 능선의 높이를 줄여 강한 표본 반사
정점을 측정 대역 밖으로 옮겨도 낮은 차수가 남습니다. 높이를 바꾸면 모서리 필드도 바뀌므로, 이 대조 실험들은
메커니즘을 밝히지 못하며 그것이 3D 상자의 성긴 계산 격자 거동을 일으킨다는 것도 보여 주지 못합니다. 이
확장은 새로운 3D 수렴 사다리를 추가하지 않습니다. 기존의 40셀 M3 회귀 행은 느린 테스트가 다시 실행합니다.

이후의 3D 비교는 상자의 높이를 두 배로 하고, 별도의 실행 길이 보정 뒤에 공통 40/60/80 사다리를 씁니다.
시간·흡수층 대조를 두었을 때, 원래 높이는 자기 수렴 차수가 낮은 칸 22개, 2차 칸 23개, 혼합 칸 아홉 개이고,
높은 상자는 낮은 칸 11개, 2차 칸 20개, 혼합 칸 23개입니다. 변화는 양방향으로 일어나므로, 높이가 균일한
2차 수렴을 되돌려 주지는 않습니다. 이는 유한 계산 격자에서의 기준값 없는 분류입니다. 상한이 있는 RCWA
사다리는 사전 등록한 퍼짐 규칙 아래에서 높은 상자의 어떤 칸에 대해서도 최종 직접 오차 차수를 분해할 수
없습니다. 낮은 칸이 적다고 해서 더 정확한 설계가 입증되거나 느린 수렴의 원인이 밝혀지지는 않습니다. C7은
여전히 통과하지 못했습니다. 108칸 전체 비교는 검증 문서에 있습니다.

다른 FDTD 코드 둘도 epsilon-12 상자에서 더 낫지 않습니다. 같은 계산 격자에서 실행하고 같은 방식으로
채점하면, Meep 1.34.0의 최악 오차는 모든 단계에서 Wavebit의 1.4에서 2.1배이고, FDTDx 0.6.3의 오차는(파장
하나에서) 5.6에서 18배입니다. 어느 코드도 2차로 수렴하지 않습니다. 이것이 장면에 한계가 있다는 증거는
아닙니다. Meep은 자체 오차 하한에서 평평해지고, FDTDx는 계단 근사 재료를 씁니다. 그래서 느린 수렴 차수는
이런 장면에서 Wavebit의 알려진 한계로 남겨 둡니다. 자신의 장면에서 수렴을 확인하세요.
검증 문서의 "Cross-tool comparison"을 참조하세요.

실제적인 결과는 세 가지입니다.

* **모든 차수에서 퍼센트 수준을 원한다면, 이 굴절률들에서는 파장당 셀 40개가 대략 적당하며**, 성긴 실행도
  첫 확인으로는 괜찮습니다. 하한에서 최악의 차수는 몇 퍼센트 벗어나고, 대부분은 훨씬 덜 벗어납니다.
* **관심 있는 파장과 관측값을 여러 분해능에서 확인하세요.** 많은 칸에 걸친 최댓값은 더 느리게 수렴하는 작은
  성분을 가릴 수 있습니다. 오차가 단일 거듭제곱 `C n^(-p)`를 따르고, 파장당 셀 수를 `n`에서 `2n`으로 바꿀 때
  숫자가 `d`만큼 움직인다면, 더 촘촘한 쪽의 오차는 `d/(2^p - 1)`입니다. 2차에서는 `d/3`, 1차에서는 `d`입니다.
  2차 Richardson 추정 `X(2n) + (X(2n) - X(n))/3`을 쓰기 전에, 적어도 세 가지 분해능으로 수렴 차수를 추정하고
  그 안정성을 확인하세요. 능선 확장에는 단일 거듭제곱 모델이 표본 단계를 설명하지 못하는 상쇄가 들어 있습니다.
* **실행 길이는 메시 세분화와 따로 확인하세요.** M3의 높이를 0.60으로 두 배로 하면 상당한 과도 성분이
  남습니다. 셀 40개에서 120 주기와 180 주기 모두 1e-5 모니터 감쇠 기준을 통과하지 못합니다. 240 주기에서는
  감쇠 검사를 통과하지만, 대역 전체의 가장 큰 `|R + T - 1|`은 여전히 약 0.5%입니다. 그 경고 기준은 스펙트럼
  관측값의 오차 한계가 아닙니다. 더 긴 실행과 더 두꺼운 흡수층을 같은 시간에서 비교하고, 그 변화가 해석하려는
  계산 격자 변화에 비해 작을 것을 요구하세요. 시간 보정은 위의 검증 문서에 기록되어 있으며, 그 자체가 공간
  수렴 결과는 아닙니다.

실행 시간이 고정되어 있으면 실행 비용은 분해능의 네제곱으로 늘어납니다(3차원과 시간 간격). 오래 울리는
형상은 더 많은 주기가 필요할 수도 있습니다. JAX를 쓴 12스레드 CPU에서 80셀 epsilon-4 셀은 20분, 120셀
epsilon-12 셀은 84분이 걸렸습니다. 셀 20에서 40개에서는 실행이 몇 초에서 몇 분 걸립니다.
