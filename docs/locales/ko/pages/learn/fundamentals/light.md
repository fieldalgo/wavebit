# 08. 어떤 빛을 넣는가: 방향과 편광

> **과거 API 재현:** 이 페이지에서 `wavebit.compat`를 쓰는 코드·설명은 현재 설치의 API가 아닙니다.
> 해당 이름은 [퇴역 전 소스 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)에 있습니다.
> 측정 재현에는 본문에 적힌 원래 커밋·환경을 우선 사용하세요. 원본 측정값은 갱신하지 않았습니다.


:::{note}
다운로드 가능한 교육 스크립트는 현재 물리 Problem/Job API를 사용합니다. 일부 설명 코드와 그림에는 이전 API 표기가 남아 있으므로 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 함께 참고하세요. 과거 측정 파일과 원본 출처는 보존하며, 이전한 스크립트의 검증과 구분합니다.
:::

입사광 하나를 지정할 때는 **파장, 빛이 들어오는 쪽, 진행 방향, 편광**을 정합니다. 이 장에서는 한 입사파를 위나 아래에서 넣는 방법과 손실 매질 안에서의 파수를 설명합니다.

여러 입사파의 크기·위상을 각각 정하거나 양쪽에서 동시에 빛을 넣는 방법은 [9장](multiple-beams.md)에서 이어집니다. 계산 후의 전력·효율은 [13장](power-and-efficiency.md), 전기장·자기장 조회는 [14장](fields.md)에서 설명합니다.

| 정하려는 내용 | 입력과 설명 |
| --- | --- |
| 파장 | `wavelength`: 모든 길이와 같은 단위를 사용하며, 한 계산의 모든 입사파가 공유합니다. |
| 진행 방향 | {ref}`theta·phi 또는 kx·ky <incident-direction-inputs>` |
| 편광 | {ref}`polarization 또는 psi·chi <incident-polarization>` |
| 한쪽에서 입사 | {ref}`incidence_side로 위·아래 선택 <incident-from-bottom>` |

손실이 있는 바깥 매질을 사용할 때의 파수와 감쇠는 {ref}`이 장 마지막 절 <absorbing-incidence>`에서 따로 설명합니다.

(incident-direction-inputs)=
## 입사광의 진행 방향을 입력하는 두 가지 방법

`theta`는 입사 매질 안의 **위상 진행 방향**과 표면에서 구조 안쪽을 향하는 법선(위쪽 입사: `−z`, 아래쪽 입사: `+z`) 사이의 각도이고, `phi`는 방위각입니다. 위상 진행 방향은 파봉처럼 같은 위상의 점이 이동하는 방향을 뜻합니다. 위상이 에너지 흐름을 거슬러 가는 음굴절률 입사 매질에서는 `theta`가 대신 빛이 구조로 들어오는 방향을 가리킵니다(2026-10-05부터, [음굴절률 매질](../../theory/rcwa/negative-index-media.md) 7절). 각도 단위는 라디안입니다. 전기장이 진동하는 방향인 편광은 다음 절에서 따로 정합니다.

이 방향 입력은 표면에서 세기가 균일한 단일 입사파를 지정합니다. 흡수성 입사 매질에서는 위상 진행 방향과 전력 흐름 방향이 다를 수 있으며, 그때의 파수는 {ref}`손실 매질 설명 <absorbing-incidence>`에서 다룹니다.

아래 그림은 무손실 입사 매질의 예입니다. s 편광은 입사면에 수직, p 편광은 입사면 안에 있습니다. 수직 입사에서는 진행 방향만으로 입사면을 하나로 정할 수 없으므로 선택한 방위각·기저 규약도 함께 봐야 합니다.

```{figure} ../../figures/learn/learn/08-light-azimuth.svg
:name: light-figure-08-light-azimuth
:alt: 위쪽 입사 theta 40도, phi 30도의 입체도. theta는 수직선과 빛의 경로 사이에, phi는 xy 표면의 +x축과 파수의 xy 성분 사이에 그려져 있다. s는 입사면에 수직이고 p는 입사면 안에서 k에 수직이다.
:figclass: learning-figure

그림 8-1. 같은 입체도에서 입사각 θ, 방위각 φ와 s·p 방향을 함께 읽습니다.
```

**회색 면은 시료의 윗표면(xy), 파란 반투명 판은 가상의 입사면입니다.** 입사면은 빛의 진행 방향 k와 표면에 수직인 선을 함께 포함합니다. 이 예의 φ=30°에서는 xz 평면을 z축 주위로 30° 돌린 면입니다. 파란 화살표를 따라 빛이 시료로 내려옵니다.

같은 빛의 경로 위에서 **주황색 s는 판 밖으로, 초록색 p는 판 안에서** 뻗습니다. 둘 다 k에 수직인 전기장의 기저 방향입니다. `(1, 0)`은 s 방향, `(0, 1)`은 p 방향으로 전기장이 진동한다는 뜻이지 빛이 그 방향으로 이동한다는 뜻이 아닙니다. 입체 그림을 평면으로 투영했기 때문에 화면에서는 세 방향의 직각이 그대로 보이지 않습니다.

**φ는 xy 표면의 +x축에서 입사 파수 k의 xy 성분까지 재는 각도**입니다. 그림에서 파란 점선 화살표가 k의 xy 성분이고, 주황색 호가 φ입니다. +z 쪽에서 내려다볼 때 +x에서 +y로 도는 반시계 방향이 양의 방향입니다. θ는 빛의 기울기를, φ는 표면 위에서 어느 쪽으로 진행하는지를 정합니다.

그림 아래의 도형 단면과 경계 법선은 [11장의 NVF 설명](nvf.md)에서 다룹니다.


진행 방향은 각도로 지정하거나, 표면과 평행한 파수 성분으로 지정합니다.

| 입력 | 직접 정하는 값 | 파장을 바꿀 때 |
| --- | --- | --- |
| `theta`, `phi` | 입사 매질 안의 위상 진행각·방위각, 라디안 | 각도를 유지하고 파수를 다시 구함 |
| `kx`, `ky` | 입사파의 실제 x·y 방향 파수 | 입력한 파수를 유지하므로 위상 진행각은 달라질 수 있음 |

**두 방법 중 하나를 사용합니다.** `kx`·`ky`의 단위는 길이의 역수입니다. 파장을 nm로 입력한다면 파수는 nm⁻¹이며, `k0`로 나눈 값이 아닙니다. `kx`와 `ky` 중 하나만 쓰면 다른 성분은 0입니다. 방향을 모두 생략하면 수직 입사입니다.

예를 들어 입사 매질의 `ε=2.25`, `μ=1`이면 굴절률은 1.5입니다. 이 매질 안에서 위상각 30°, 방위각 0°인 빛은 `kx=0.75 k0`, `ky=0`이므로 다음 두 입력은 같은 빛을 지정합니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

scene = dict(
    top=2.25, bot=1.,
    incidence_side="top",                 # 위쪽 매질에서 아래로 입사 (기본값)
    wavelength=500., period=[700., 700.],   # 길이 단위: nm
    structure=np.full((1, 2, 2), 4.), thickness=[100.],
    truncation_order=0, polarization=(1., 0.),
)
by_angle = wavebit.rcwa(**scene, theta=np.pi/6, phi=0.)

k0 = 2*np.pi/scene["wavelength"]           # 진공 파수, nm⁻¹
by_k = wavebit.rcwa(**scene, kx=0.75*k0, ky=0.)

angle_result = by_angle.solve()
k_result = by_k.solve()
np.testing.assert_allclose(angle_result.T, k_result.T, atol=1e-12)
```

`kx`, `ky`를 직접 넣어도 **입사 매질의 ε·μ는 필요합니다.** 같은 횡파수라도 재료가 달라지면 수직 방향 파수와 전기장·자기장의 관계, 입사 전력이 달라지기 때문입니다. 손실 입사 매질에서도 직접 파수를 지정할 수 있으며, 별도의 공기층을 가정하지 않습니다.

입력값은 **실제 xyz 좌표계의 실수 파수**입니다. 사선 격자의 `u`, `v` 차수 번호와는 다르며, 회절 차수에 따른 역격자 파수는 솔버가 더합니다.

`incidence_side`는 빛을 구조의 어느 쪽에서 비추는지 정합니다. 위 예제의 `incidence_side="top"`은 위쪽 매질 `top`에서 구조를 향해 **아래로(−z)** 입사한다는 뜻이며, 생략해도 기본값은 `"top"`입니다. `incidence_side="bottom"`으로 지정하면 아래쪽 매질 `bot`에서 구조를 향해 **위로(+z)** 입사합니다. 아래쪽 입사에서도 직접 지정한 `kx`, `ky`의 부호는 그대로 사용합니다.

직접 입력한 값은 유한한 실수여야 합니다. 입사 전력이 0인 소멸파는 회절 효율의 기준으로 사용할 수 없어 거부합니다. 복소수도 거부하며, 이때 오류 메시지가 대신 쓸 입력을 안내합니다. 이유와 복소 파수가 필요할 때의 방법은 {ref}`복소 횡파수가 필요할 때 <complex-transverse-wavevector>`에서 설명합니다. `kx=ky=0`이면 빛이 표면에 수직으로 들어오는 **수직 입사**입니다. 이때는 진행 방향만으로 입사면을 하나로 정할 수 없으므로, 방위각 0을 기준으로 **s를 y 방향, p를 x 방향**으로 정합니다. 전기장의 방향은 `polarization`으로 지정합니다.

TE·TM을 분리하는 1D 계산에서는 `ky`가 0이어야 합니다. 조건과 입력 방법은 {ref}`편광 절 <d-te-tm>`에서 설명합니다.

`theta`·`phi`와 `kx`·`ky`를 함께 넣으면 오류입니다. 솔버의 입력 방식을 바꿀 때에는 먼저 반대쪽 두 입력을 `None`으로 비운 뒤 새 값을 지정합니다. 파장·각도·파수를 여러 값으로 바꾸어 계산하는 방법은 {ref}`18장 <jobs-and-sweeps>`에서 설명합니다.

`order_kxy()`는 **계산된 차수별 파수를 `k0`로 나누어** 반환하는 조회 함수입니다. 직접 입력과 단위가 다르므로, 반환값에 `2π/λ`를 곱해야 실제 파수가 됩니다.

(incident-polarization)=
## 편광을 지정하는 방법

**편광은 한 위치에서 전기장의 방향과 크기가 시간에 따라 어떻게 변하는지를 뜻합니다.** 전기장의 끝점이 직선을 왕복하면 선편광, 원을 그리면 원편광, 타원을 그리면 타원편광입니다. 빛이 진행하는 방향은 앞에서 정한 `theta`·`phi` 또는 `kx`·`ky`로 유지하고, 편광 설정으로 전기장의 진동을 정합니다.

Wavebit에서는 전기장을 **s 성분과 p 성분**으로 나누어 입력합니다. s는 입사면에 수직인 방향이고, p는 입사면 안에서 빛의 진행 방향에 수직인 방향입니다. 아래의 편광 궤적과 각도 설명은 **무손실 등방성 입사 매질**을 기준으로 합니다. 그림에서는 s와 p를 가로·세로축으로 펼쳐 그립니다.

```{figure} ../../figures/learn/learn/08-polarization-states.svg
:name: light-figure-08-polarization-states
:alt: 한 위치의 전기장 끝점 궤적을 s 가로축과 p 세로축으로 비교한다. 같은 위상의 (1, 1)은 직선, 크기가 같고 위상차가 90도인 (1, 1j)는 원, 크기 비가 2 대 1이고 위상차가 90도인 (2, 1j)는 타원을 만든다. 주황색 화살표는 시작 시각의 전기장이다.
:figclass: learning-figure

그림 8-2. 초록색은 한 주기 동안 전기장 끝점이 그리는 궤적이고, 주황색은 그중 한 시각의 전기장입니다. 빛이 공간을 이동하는 경로를 그린 것이 아닙니다. 세 입력 모두 진폭 쌍을 정규화한 뒤 같은 눈금으로 그렸습니다.
```

(s-p)=
### s와 p의 크기·위상을 직접 입력하기

`polarization=(a_s, a_p)`에서 **첫 번째 값은 s 성분, 두 번째 값은 p 성분의 전기장 진폭**입니다. 이 두 복소수를 묶은 것을 **Jones 벡터**라고 합니다. 각 값의 절댓값은 진동의 크기를, 두 값의 위상차는 두 성분이 진동하는 시간 차이를 정합니다.

| 만들고 싶은 편광 | 입력 | 두 성분의 관계 |
| --- | --- | --- |
| s 선편광 | `polarization=(1., 0.)` | s 성분만 진동 |
| p 선편광 | `polarization=(0., 1.)` | p 성분만 진동 |
| s와 p 사이 45° 선편광 | `polarization=(1., 1.)` | 크기가 같고 위상도 같음 |
| 반대 대각선의 선편광 | `polarization=(1., -1.)` | 크기가 같고 위상차가 180° |
| 원편광 | `polarization=(1., 1j)` | 크기가 같고 위상차가 90° |
| 반대로 회전하는 원편광 | `polarization=(1., -1j)` | 크기가 같고 위상차가 −90° |
| s 쪽으로 긴 타원편광 | `polarization=(2., 1j)` | s의 진폭이 p의 2배이고 위상차가 90° |

Python의 **`1j`는 허수 단위 i**입니다. `a_s=1`, `a_p=1j`를 넣으면 한 위치에서 s 성분은 `cos(ωt)`, p 성분은 `sin(ωt)`에 비례해 진동합니다. 여기서 ω는 빛의 각주파수, t는 시간입니다. s가 최대일 때 p는 0이고, 1/4주기 뒤에는 s가 0, p가 최대가 됩니다. 두 진동을 더하면 전기장 끝점이 원을 그립니다. Wavebit의 시간 규약은 `exp(−iωt)`입니다.

**두 성분을 모두 넣는다고 반드시 타원편광이 되는 것은 아닙니다.** 위상차가 0° 또는 180°이면 전기장은 한 직선 위에서 왕복합니다. 원편광에는 같은 크기와 ±90° 위상차가 모두 필요합니다. 원편광의 좌·우 명칭은 관찰 방향에 따라 혼동하기 쉬우므로, 여기서는 입력 쌍과 그림의 시간 진행 화살표로 회전 방향을 표시합니다.

입력한 쌍은 `sqrt(abs(a_s)**2 + abs(a_p)**2)`로 나누어 정규화됩니다. 따라서 `(1, 1j)`와 `(10, 10j)`는 같은 입사 상태가 되며, `(1, 1j)`를 직접 `sqrt(2)`로 나누어 넣을 필요는 없습니다. `(0, 0)`은 전기장이 없는 입력이므로 사용할 수 없습니다. **이 정규화는 두 계수의 크기를 맞추는 것이며, 입사 전력을 1 W로 지정한다는 뜻은 아닙니다.** 전력은 매질과 입사 방향까지 반영하여 계산합니다.

다음은 공기에서 격자로 원편광을 입사시키는 실행 예제입니다. 길이 단위는 nm이고, `structure`의 숫자는 상대 유전율입니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

scene = dict(
    backend='numpy',
    top=1., bot=2.25,
    incidence_side="top",
    wavelength=900., period=[700., 700.],
    theta=np.deg2rad(30.), phi=0.,
    structure=np.array([[[1., 1., 12.25, 12.25]]]),
    thickness=[300.],
    truncation_order=dict(x=4, y=0),
)

solver = wavebit.rcwa(**scene, polarization=(1., 1j))
result = solver.solve()
print("반사율:", result.R.sum())
print("투과율:", result.T.sum())
```

`polarization`만 표의 다른 쌍으로 바꾸면 같은 구조·입사 방향에서 편광에 따른 응답을 비교할 수 있습니다. s로 입사하더라도 구조가 두 편광 성분을 결합하면 반사파나 투과파에는 p 성분이 생길 수 있습니다. 입력 편광은 입사파의 상태를 정합니다.

(psi-chi)=
### 각도로 입력하기: psi와 chi

편광의 방향이나 타원 모양을 각도로 지정하려면 `psi`, `chi`를 사용합니다. **두 각도 모두 실수이며, 단위는 라디안**입니다.

- **ψ (`psi`)**: p 방향에서 s 방향 쪽으로 잰 편광 타원의 긴 축 각도입니다. 선편광에서는 전기장이 왕복하는 직선의 각도입니다.
- **χ (`chi`)**: 타원의 납작한 정도를 정하는 각도입니다. `abs(tan(chi))`는 짧은 반지름을 긴 반지름으로 나눈 비입니다. −45°부터 +45° 범위로 나타내며, 부호를 바꾸면 전기장이 도는 방향이 반대가 됩니다.

```{figure} ../../figures/learn/learn/08-polarization-angles.svg
:name: light-figure-08-polarization-angles
:alt: p를 세로축, s를 가로축으로 그린 편광 타원. psi는 p축에서 긴 축까지 재는 각도다. chi는 반지름의 비로 정해지며 0도는 선, 절댓값 45도는 원, 그 사이는 타원이다.
:figclass: learning-figure

그림 8-3. ψ는 타원의 긴 축 방향, χ는 두 반지름의 비와 회전 방향을 정합니다. 원에서는 긴 축과 짧은 축이 같아 ψ를 바꾸어도 편광 타원의 모양과 회전 방향은 같습니다.
```

| 편광 | 각도 입력 | 대응하는 진폭 쌍 |
| --- | --- | --- |
| p 선편광 | `psi=0., chi=0.` | `(0, 1)` |
| s 선편광 | `psi=np.pi/2, chi=0.` | `(1, 0)` |
| 45° 선편광 | `psi=np.pi/4, chi=0.` | `(1, 1)` |
| 원편광 | `psi=0., chi=-np.pi/4` | `(1, 1j)`와 같은 편광 |
| 반대 회전의 원편광 | `psi=0., chi=np.pi/4` | `(1, -1j)`와 같은 편광 |

`chi`를 생략하면 0, 즉 선편광입니다. 편광 입력을 모두 생략하면 s 선편광을 사용합니다. 원편광 행의 두 표현은 공통 위상까지 같다는 뜻은 아닙니다. 두 성분에 같은 위상을 더해도 편광 타원과 반사율·투과율은 같지만, 복소 전기장 진폭의 위상은 달라질 수 있습니다.

위 코드의 `scene`을 그대로 사용하여 p에서 s 쪽으로 30° 기울어진 선편광을 넣으려면 다음처럼 씁니다.

```python
solver = wavebit.rcwa(**scene, psi=np.deg2rad(30.), chi=0.)
result = solver.solve()
```

이 선편광은 `polarization=(0.5, np.sqrt(3)/2)`로도 지정할 수 있습니다. **각도 입력과 진폭 쌍 중 하나만 사용합니다.** `psi`·`chi`와 `polarization`을 한 호출에 함께 넣으면 오류입니다.

````{dropdown} 수식으로 보기: psi·chi에서 두 진폭을 구하기
Wavebit는 다음 식으로 두 전기장 계수를 구합니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>a<sub>s</sub> = sin ψ cos χ + i cos ψ sin χ</p>
<p>a<sub>p</sub> = cos ψ cos χ − i sin ψ sin χ</p>
</div>
```

`chi=0`이면 `a_s=sin(psi)`, `a_p=cos(psi)`입니다. 이 때문에 `psi=0`이 p, `psi=π/2`가 s입니다. 이 두 계수는 처음부터 절댓값 제곱의 합이 1이므로 따로 정규화하지 않습니다.
````

```{raw} html
<section class="exercise polarization-exercise" id="polarization-explorer" aria-labelledby="pol-title">
<h3 id="pol-title">직접 바꿔보기 · psi와 chi로 편광 정하기</h3>
<p class="concept">무손실 등방성 입사 매질의 예입니다. 한 위치에서 전기장 끝점이 그리는 궤적을 s·p 축으로 봅니다.</p>
<div class="controls">
<label for="pol-psi">긴 축 방향 ψ <output id="pol-psi-out" for="pol-psi">30°</output>
<input id="pol-psi" type="range" min="0" max="180" step="1" value="30">
<span class="control-hint">p 방향에서 s 방향 쪽으로 잽니다.</span></label>
<label for="pol-chi">타원 모양 χ <output id="pol-chi-out" for="pol-chi">20°</output>
<input id="pol-chi" type="range" min="-45" max="45" step="1" value="20">
<span class="control-hint">0°: 선 · ±45°: 원 · 부호: 회전 방향</span></label>
</div>
<svg viewBox="0 0 400 320" role="img" aria-label="psi와 chi로 지정한 전기장 끝점의 편광 궤적">
<defs>
<marker id="pol-motion-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#117c77"/></marker>
<marker id="pol-field-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#b56626"/></marker>
</defs>
<line x1="45" y1="150" x2="355" y2="150" stroke="#abbec5"/>
<line x1="200" y1="15" x2="200" y2="280" stroke="#abbec5"/>
<text x="358" y="155">s</text><text x="210" y="28">p</text>
<polyline id="pol-curve" points="" fill="none" stroke="#117c77" stroke-width="3"/>
<polyline id="pol-motion" points="" fill="none" stroke="#117c77" stroke-width="3" marker-end="url(#pol-motion-arrow)"/>
<line id="pol-vector" x1="200" y1="150" x2="200" y2="150" stroke="#b56626" stroke-width="3" marker-end="url(#pol-field-arrow)"/>
<circle cx="200" cy="150" r="3" fill="#354b60"/>
<text x="30" y="303">초록: 한 주기 궤적 · 주황: t = 0의 전기장</text>
</svg>
<div class="readout" id="pol-readout" aria-live="polite" aria-atomic="true">
<strong id="pol-state">타원편광</strong>
<p id="pol-shape-note">짧은 반지름 / 긴 반지름 = 0.364</p>
<p>각도로 입력<code id="pol-angle-code">psi=np.deg2rad(30.), chi=np.deg2rad(20.)</code></p>
<p>진폭 쌍으로 입력 (반올림)<code id="pol-pair-code">polarization=(0.4698+0.2962j, 0.8138-0.1710j)</code></p>
</div>
<p class="explain">두 코드 중 한 가지를 rcwa()의 편광 입력으로 사용합니다. 슬라이더는 도(°) 단위이며, 표시한 코드는 np.deg2rad()로 라디안으로 바꿉니다.</p>
</section>
```

먼저 `chi=0°`에서 `psi`를 바꿔 선편광의 방향을 돌려 보세요. 이어서 `chi`를 ±45° 쪽으로 움직이면 직선이 타원을 거쳐 원이 됩니다. `chi`의 부호를 바꾸면 작은 초록 화살표의 진행 방향이 반대로 바뀝니다. 원편광에서는 `psi`를 바꾸어도 원의 모양은 같고, 주황색으로 표시한 시작 시각의 전기장 방향이 달라집니다.

(d-te-tm)=
### 1D 구조에서 TE 또는 TM만 계산하기

**y 방향으로 구조가 변하지 않고 입사파도 xz 평면 안에서 진행하며, 재료가 TE와 TM을 결합하지 않는 경우**에는 두 편광을 분리해서 계산할 수 있습니다. 이때 `tetm="te"`는 전기장이 y 방향인 TE, `tetm="tm"`은 자기장이 y 방향이고 전기장은 xz 평면에 있는 TM을 선택합니다. 이 조건에서 TE는 s, TM은 p에 대응합니다.

위 `scene`은 y 방향으로 일정한 등방성 격자이고 `phi=0`이므로 다음처럼 따로 계산할 수 있습니다.

```python
te_result = wavebit.rcwa(**scene, tetm="te").solve()
tm_result = wavebit.rcwa(**scene, tetm="tm").solve()
```

`tetm`에서는 `ky`가 0이어야 하며, 생략하면 0입니다. `ky=0.`을 직접 적어도 됩니다. 0이 아닌 `ky`와 함께 지정하면 오류가 발생합니다. `tetm`은 TE 또는 TM 하나만 푸는 계산 방식을 선택하므로 `psi`, `chi`, `polarization`과 함께 사용하지 않습니다. 2D 패턴이나 `ky`가 0이 아닌 입사 조건에서는 `polarization` 또는 `psi`·`chi`로 입사 편광을 정합니다. 일반 계산에서 s 입사만 지정하려면 `polarization=(1, 0)`을 쓰면 됩니다.

(incident-from-bottom)=
## 위쪽과 아래쪽 중 입사 쪽을 정하기

한쪽에서 입사하는 계산은 `incidence_side`로 빛이 들어오는 쪽을 정합니다. 기본값은 `"top"`입니다.

| 설정 | 입사 매질과 진행 방향 | 반사율 `R` | 투과율 `T` |
| --- | --- | --- | --- |
| `"top"` | 위쪽 `top`에서 아래로(−z) | 위쪽으로 돌아가는 빛 | 아래쪽으로 나가는 빛 |
| `"bottom"` | 아래쪽 `bot`에서 위로(+z) | 아래쪽으로 돌아가는 빛 | 위쪽으로 나가는 빛 |

`incidence_side="bottom"`은 아래쪽 반무한 매질 `bot`에서 위쪽(+z)으로 빛을 넣는 설정입니다. 입사파는 바닥 `z=0`에서 맨 아래층인 `structure[0]`을 먼저 만납니다. `structure`와 `thickness`는 아래층부터 위층 순서로 입력하며, `top`은 위쪽 매질, `bot`은 아래쪽 매질입니다.

반사율 `R`은 아래쪽 `bot`으로 돌아가는 전력의 비율이고, 투과율 `T`는 위쪽 `top`으로 나가는 전력의 비율입니다. 두 값 모두 아래쪽에서 들어온 입사 전력을 기준으로 합니다.

(incidence-side-angles)=
(thetaphipsichi)=
### 위·아래 입사에 공통인 각도와 편광

**`theta`·`phi`·`psi`·`chi`는 위·아래 입사에서 같은 규칙으로 지정합니다.** 입사 쪽은 `incidence_side`로 정하고, 각도는 다음과 같이 읽습니다. 아래 설명은 무손실 등방성 입사 매질을 기준으로 합니다.

| 입력 | 의미 |
| --- | --- |
| `theta` | 빛의 진행 방향과 표면 안쪽을 향하는 법선 사이의 각도. 위쪽 입사는 −z, 아래쪽 입사는 +z가 기준이며, 양쪽 모두 0°는 수직 입사입니다. |
| `phi` | 빛의 진행 벡터를 xy 평면에 투영했을 때, +x에서 +y 쪽으로 잰 방위각. 양쪽 모두 고정된 x·y축을 기준으로 합니다. |
| `psi` | 해당 입사파의 p 방향에서 s 방향 쪽으로 잰 편광 타원의 긴 축 각도. |
| `chi` | 편광 타원의 모양과 s/p 성분의 상대 위상을 정하는 각도. 양쪽 모두 같은 진폭 변환식을 사용합니다. |

예를 들어 `theta=30°`, `phi=0`이면 **양쪽 모두 −x 쪽에서 와서 x가 증가하는 방향으로 진행합니다.** 표면의 입사점을 기준으로 경로를 비교하면 다음과 같습니다.

| 입사 쪽 | 빛이 오는 쪽 | 표면으로 향하는 진행 방향 |
| --- | --- | --- |
| 아래쪽: `bottom` | −x 쪽 아래 | +x·+z 방향으로 올라옴 |
| 위쪽: `top` | −x 쪽 위 | +x·−z 방향으로 내려옴 |

양쪽 모두 진행 벡터의 x 성분은 양수이고, z 성분의 부호가 다릅니다. 위쪽 입사 그림의 선이 −x 쪽 위로 뻗어 있는 것은 빛이 오는 쪽을 보여 주며, 화살표는 그 선을 따라 표면으로 향합니다.

편광은 각 입사파의 s·p 방향을 기준으로 지정합니다. 선편광인 `chi=0`에서 `psi=0`은 p 편광, `psi=90°`는 s 편광, `psi=45°`는 같은 크기·같은 위상의 s와 p 성분을 갖는 선편광입니다. `abs(chi)=45°`이면 원편광입니다. 위·아래 입사 모두 이 규칙으로 원하는 편광을 직접 지정합니다.

(light-detail-id11)=
### 예제: 유리에서 위로 입사하는 선편광

아래 예제는 유리 `bot=2.25`에서 빛이 두 층을 향해 올라오는 설정입니다. `theta=30°`, `phi=0`이므로 −x 쪽 아래에서 표면으로 진행하며, 유리 안의 진행 방향은 +z 법선과 30°를 이룹니다. 편광은 `psi=45°`, `chi=0°`인 선편광입니다. 코드에서는 각도를 라디안으로 입력합니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

solver = wavebit.rcwa(
    backend='numpy',
    top=1., bot=2.25,                       # 위: 공기, 아래: 유리
    incidence_side="bottom",               # 유리에서 위로 입사
    wavelength=600., period=[1500., 1500.], # 길이 단위: nm
    structure=np.array([
        [[4.]],                            # 맨 아래층: z=0~100 nm
        [[9.]],                            # 그 위층: z=100~300 nm
    ]),
    thickness=[100., 200.],
    theta=np.deg2rad(30.), phi=0.,          # kx > 0, kz > 0
    psi=np.deg2rad(45.), chi=0.,            # 같은 크기·같은 위상의 s와 p
    truncation_order=0,
)
result = solver.solve()
print("유리 쪽으로 돌아가는 반사율:", result.R.sum())
print("공기 쪽으로 나가는 투과율:", result.T.sum())
```

각도로 입력한 빛의 파수는 입사 매질의 ε·μ와 파장으로 구합니다. 이 예제에서는 아래쪽 유리의 재료값을 사용합니다. 위·아래 입사 매질이 다르면 같은 각도와 파장에서도 파수의 크기는 달라질 수 있습니다.

(absorbing-incidence)=
## 손실이 있는 입사·투과 매질의 파수

구조 위·아래의 반무한 매질 중 빛이 들어오는 쪽을 **입사 매질**, 구조를 통과한 빛이 나가는 쪽을 **투과 매질**이라고 합니다. 두 매질 모두 빛을 흡수하는 손실 매질일 수 있습니다.

이 절에서는 두 매질이 등방성이고, 표면과 평행한 파수 `kx`, `ky`가 실수인 경우를 설명합니다. 파수를 계산할 때는 각 매질의 **유전율 ε와 투자율 μ의 실수부와 허수부를 모두 사용합니다.** 손실 매질에서는 표면에 수직인 파수 `kz`가 복소수가 되며, 그 허수부가 표면에 수직인 방향의 감쇠를 나타냅니다.

| 입사 설정 | 입사 매질 | 투과 매질 |
| --- | --- | --- |
| `incidence_side="top"` | 위쪽 `top` | 아래쪽 `bot` |
| `incidence_side="bottom"` | 아래쪽 `bot` | 위쪽 `top` |

**입사파의 `kx`, `ky`를 먼저 정하고, 양쪽 매질에서 각각 `kz`를 구합니다.** 각도로 입력하면 입사 매질의 ε·μ와 `theta`, `phi`로 입사파의 횡파수를 정합니다. 직접 입력하면 주어진 `kx`, `ky`를 사용합니다. 투과파의 방향은 이 횡파수와 투과 매질의 ε·μ에서 결정됩니다.

(light-detail-id4)=
### 입사 매질에 손실이 있을 때

여기서 정하는 입사 조건은 **표면의 어느 위치에서나 입사파의 세기가 같도록 비추는 것**입니다. 손실이 있는 입사 매질에서도 이 조건을 유지하기 위해 `kx`, `ky`는 실수로 정합니다. 표면에 수직인 성분 `kz`에는 허수부가 있을 수 있으므로, 파수 벡터 **k 전체는 복소수**입니다.

`Re`는 실수부, `Im`은 허수부를 뜻합니다. 벡터의 세 성분에서 실수부끼리, 허수부끼리 모으면 다음과 같습니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p><b>k</b> = Re(<b>k</b>) + i Im(<b>k</b>)</p>
<p>Re(<b>k</b>) = (k<sub>x</sub>, k<sub>y</sub>, Re(k<sub>z</sub>))</p>
<p>Im(<b>k</b>) = (0, 0, Im(k<sub>z</sub>))</p>
</div>
```

- **실수부 Re(k)는 위상이 진행하는 방향을 정합니다.** 같은 위상의 점들을 이은 면을 파면이라고 하며, Re(k)는 파면에 수직입니다. `kx`나 `ky`가 0이 아니면 위상은 표면에 비스듬한 방향으로 진행합니다.
- **허수부 Im(k)는 진폭이 위치에 따라 감쇠하는 방향과 정도를 정합니다.** 이 조건에서는 x·y 성분이 0이므로 같은 높이에서 옆으로 이동해도 진폭은 같습니다. z 성분의 허수부에 의해 구조를 향해 진행할수록 진폭과 세기가 줄어듭니다.

따라서 **위상은 비스듬히 진행하면서, 세기는 높이에 따라서만 달라질 수 있습니다.** 아래 두 패널은 그 한 입사파의 세기와 위상을 나누어 보여줍니다.

```{figure} ../../figures/learn/learn/08-absorbing-wave.svg
:name: light-figure-08-absorbing-wave
:alt: 같은 입사파의 세기와 위상을 나누어 보여주는 두 패널. 왼쪽은 같은 높이에서 세기가 같고 아래로 갈수록 약해지는 모습을 색 농도로 나타낸다. 오른쪽은 기울어진 파면과 그에 수직인 위상 진행 방향 Re(k)를 나타낸다. theta는 이 방향과 아래쪽 법선 사이의 각이다.
:figclass: learning-figure

그림 8-4. 왼쪽의 A와 B는 같은 세기이고, 더 낮은 C와 D도 서로 같지만 A·B보다 약합니다. 오른쪽의 가는 선은 파면, 굵은 화살표는 위상 진행 방향 Re(k)입니다.
```

**같은 세기라는 말은 같은 순간의 전기장 값도 같다는 뜻은 아닙니다.** 같은 높이의 A와 B에서도 위상이 다르면 전기장의 진동 시점은 다를 수 있습니다. 여기서 세기는 전기장과 자기장으로 구한 전력 흐름의 시간 평균을 뜻합니다.

그림의 `theta`는 **Re(k)와 구조 안쪽 법선 사이의 각도**입니다. 위쪽 입사에서는 아래쪽 법선인 −z를 기준으로 잽니다. 전력 흐름의 방향은 전기장과 자기장으로 구하며, 손실 매질에서는 Re(k)의 방향과 다를 수 있습니다.

**Wavebit의 `kx`, `ky` 직접 입력은 실수만 지원합니다.** 허수부가 0이 아닌 값을 넣으면 오류가 발생하고, 오류 메시지가 흡수성 입사에는 실수 `theta`를 쓰라고 안내합니다. 위의 설명과 그림은 표면에서 입사파의 세기가 균일한 이 입력 조건에 해당합니다. 복소 파수가 꼭 필요한 경우는 {ref}`아래 <complex-transverse-wavevector>`에서 설명합니다.

:::{dropdown} 수식으로 보기: 위상과 감쇠를 나누어 읽기
위치 벡터를 **r = (x, y, z)**, 시간을 t, 각주파수를 ω라고 하면, 평면파의 공간·시간 변화를 나타내는 인자는 다음처럼 나뉩니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>exp[i(<b>k</b> · <b>r</b> − ωt)]</p>
<p>= exp[−Im(<b>k</b>) · <b>r</b>] × exp[i(Re(<b>k</b>) · <b>r</b> − ωt)]</p>
</div>
```

첫째 인자는 진폭의 크기를, 둘째 인자는 진동의 위상을 정합니다. `kx`, `ky`가 실수이면 각 인자에 들어가는 위치 의존성은 다음과 같습니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>진폭 인자 = exp[−Im(k<sub>z</sub>) z]</p>
<p>위상 = k<sub>x</sub>x + k<sub>y</sub>y + Re(k<sub>z</sub>) z − ωt</p>
</div>
```

**진폭에는 z만 들어가지만, 위상에는 x·y·z가 모두 들어갈 수 있습니다.** 이것이 같은 높이에서는 세기가 일정하면서 위상은 비스듬히 진행할 수 있는 이유입니다.

그림에서 **입사파는 아래로 내려갈수록 약해집니다.** 아래쪽으로 잰 수직 이동 거리를 `d > 0`, 수직 방향의 진폭 감쇠계수를 `κ > 0`이라고 쓰면 다음과 같습니다. κ의 단위는 길이의 역수입니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>이동 후 진폭 = 이동 전 진폭 × exp(−κd)</p>
</div>
```

`exp(−κd)`는 1보다 작으므로 진폭이 줄어듭니다. 더 깊이 내려가 d가 커질수록 더 약해집니다.

이를 Wavebit의 z 좌표로 쓰면 부호가 붙습니다. **+z가 위쪽이므로 아래로 d만큼 이동할 때 좌표 변화는 `Δz = −d`입니다.** 감쇠 방향도 아래쪽이므로 `Im(kz) = −κ`입니다. 위의 진폭 인자를 두 위치에서 비교하면 같은 결과가 나옵니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>이동 후 진폭 / 이동 전 진폭</p>
<p>= exp[−Im(k<sub>z</sub>) Δz] = exp[−(−κ)(−d)] = exp(−κd)</p>
</div>
```

여기서는 **입사파만의 진폭 변화**를 설명했습니다. 반사파가 함께 있을 때의 전체 전기장은 입사파와 반사파의 전기장을 더해 구합니다.
:::

반사·투과 효율의 기준은 **입사 쪽 경계면에 도달한 입사파만의 전력**입니다. 이 전력을 1로 정규화해 결과를 읽는 방법은 {ref}`13장 <half-space-power>`에서 설명합니다.

무손실이고 굴절률 n이 양수인 등방성 입사 매질에서는 `theta`, `phi`로부터 다음과 같이 횡파수를 구합니다. 여기서 n은 입사 매질의 굴절률이고, k₀ = 2π/λ는 진공 파수입니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>k<sub>x</sub> = k<sub>0</sub> n sin(θ) cos(φ)</p>
<p>k<sub>y</sub> = k<sub>0</sub> n sin(θ) sin(φ)</p>
</div>
```

손실이 있는 입사 매질에서는 **입력한 `theta`, `phi`가 나타내는 위상 진행 방향을 유지하면서, 같은 높이에서 입사파의 세기가 일정하도록** 파수를 구합니다. 이를 위해 `kx`, `ky`는 실수로 정하고, `kz`는 복소수가 될 수 있도록 계산합니다. 세 성분은 그 매질의 상대 유전율 ε와 상대 투자율 μ로 정해지는 다음 관계도 만족해야 합니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>k<sub>x</sub>² + k<sub>y</sub>² + k<sub>z</sub>² = k<sub>0</sub>² ε<sub>r</sub> μ<sub>r</sub></p>
</div>
```

무손실 매질과 흡수성 매질 모두, **반사·투과 효율을 계산할 때 `theta`는 −90°보다 크고 90°보다 작은 각도**로 지정합니다. 코드에서는 라디안을 사용하므로 `abs(theta) < π/2`입니다. 0°는 표면에 수직인 위상 진행 방향이고, 절댓값이 90°에 가까울수록 표면과 나란해집니다.

정확히 ±90°를 제외하는 이유는 두 경우에 차이가 있습니다.

- **무손실 매질:** 파수는 유한하지만 `kz=0`이 되어 빛이 표면과 나란히 진행합니다. 표면을 통과하는 입사 전력이 0이므로, 이 전력을 분모로 하는 반사·투과 효율을 정의할 수 없습니다.
- **흡수성 매질:** 위상이 표면과 나란히 진행하는 θ=±90°에서는 `Re(kz)=0`입니다. 그러나 여기서 다루는 흡수성 입사 조건에서는 `Re(kz)`와 `Im(kz)`의 곱이 0이 아니어야 합니다. 따라서 두 조건을 동시에 만족할 수 없습니다. 아래에서 파수 관계식의 허수부를 비교하면 그 이유를 확인할 수 있습니다.

:::{dropdown} 계산식: 복소 재료값에서 실수 kx·ky를 구하는 방법
출발점은 위에 쓴 매질의 파수 관계식입니다. `kx`, `ky`가 실수이면 두 성분의 제곱에는 허수부가 없고, `kz²`의 허수부는 `2 Re(kz) Im(kz)`입니다. 따라서 양변의 허수부를 비교하면:

```{raw} html
<div class="power-equations" style="text-align:center">
<p>k<sub>x</sub>² + k<sub>y</sub>² + k<sub>z</sub>² = k<sub>0</sub>² ε<sub>r</sub> μ<sub>r</sub></p>
<p>2 Re(k<sub>z</sub>) Im(k<sub>z</sub>) = k<sub>0</sub>² Im(ε<sub>r</sub> μ<sub>r</sub>)</p>
</div>
```

**ε와 μ의 곱에 허수부가 있으면 오른쪽은 0이 아닙니다.** 그런데 θ=±90°에서는 위상 진행 방향에 수직 성분이 없으므로 `Re(kz)=0`입니다. `Im(kz)`가 유한하다면 왼쪽은 0이 되어 위 식을 만족할 수 없습니다. 이것이 표면에서 세기가 균일한 흡수성 입사파에 정확히 ±90°를 지정할 수 없는 이유입니다.

입사 매질의 재료값과 각도에서 파수를 구하기 위해, 먼저 **상대 유전율과 상대 투자율의 곱을 실수부 a와 허수부 b로 나눕니다.**

```{raw} html
<div class="power-equations" style="text-align:center">
<p>ε<sub>r</sub> μ<sub>r</sub> = a + ib</p>
<p>a = Re(ε<sub>r</sub> μ<sub>r</sub>), &nbsp; b = Im(ε<sub>r</sub> μ<sub>r</sub>)</p>
</div>
```

예를 들어 `ε=4+3j`, `μ=1`이면 `a=4`, `b=3`입니다. 진공 파장 λ에서 진공 파수는 `k0=2π/λ`입니다.

다음으로 **위상 진행을 나타내는 실수 벡터의 길이를 진공 파수로 나눈 값**을 N이라고 둡니다. `kx`, `ky`는 이미 실수이므로 그대로 쓰고, 복소수인 `kz`에서는 실수부만 사용합니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>N = √[k<sub>x</sub>² + k<sub>y</sub>² + (Re(k<sub>z</sub>))²] / k<sub>0</sub></p>
</div>
```

**N은 단위가 없는 실수**입니다. 예를 들어 N=2이면, 위상 진행 방향에서 한 주기가 반복되는 거리는 진공 파장의 절반인 λ/2입니다. 감쇠를 나타내는 `Im(kz)`는 이 벡터의 길이에 포함하지 않습니다.

N을 재료값과 위상각 θ로 계산한 뒤, 입사 방향 θ·φ에 따라 `kx`, `ky`를 구합니다.

```text
N² = (a + sqrt(a² + b² / cos²(theta))) / 2

kx = k0 s N sin(theta) cos(phi)
ky = k0 s N sin(theta) sin(phi)
```

`s`는 입사 매질의 부호 있는 굴절률 `n = √ε·√μ`의 실수부 부호입니다. 보통 매질과 비자성 매질은 모두 `s = +1`이라 위 식은 예전과 같습니다. 음굴절률(음위상속도) 매질에서만 `s = −1`입니다. 그런 매질에서는 위상이 빛을 거슬러 가므로, θ가 언제나 **빛(에너지)이 구조로 들어오는 방향**을 가리키도록 부호를 뒤집습니다(2026-10-05부터, [음굴절률 매질](../../theory/rcwa/negative-index-media.md) 7절).

이 식은 `kx² + ky² + kz² = k0² εr μr`를 만족하면서 `kx`, `ky`가 실수가 되도록 구한 것입니다. `N`은 손실 매질에서 각도에도 의존하며, 재료의 복소 굴절률 `n` 자체와 구별합니다. `kz`의 방향은 위쪽 입사면 −z, 아래쪽 입사면 +z를 향하도록 선택합니다.

무손실 양의 굴절률 매질에서는 `b=0`, `a=n²>0`이므로 `N=n`입니다. 수직 입사에서는 두 횡파수가 모두 0입니다. 자세한 정의와 복소 각도 입력의 별도 의미는 [좌표·부호 규약](../../reference/common/conventions.md#real-angle-incidence)을 참고합니다.
:::

(absorbing-incidence-example)=
#### 예시: 손실 입사 매질 안의 위상각 30°

아래 예제의 `theta=30°`는 `top=4+3j`인 **그 매질 안에서의 위상각**입니다. 단위는 nm로 통일했습니다. 유한 층과 `bot`은 같은 무손실 재료로 두고, 입사파의 횡파수를 확인합니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

solver = wavebit.rcwa(
    top=4. + 3j, bot=2.25,
    incidence_side="top",
    wavelength=1000., period=[1200., 1200.],
    theta=np.deg2rad(30.), phi=0.,
    polarization=(1., 0.),                         # s 입사
    structure=np.full((1, 2, 2), 2.25),
    thickness=[100.],
    truncation_order=0,                           # 균일층이므로 0차만 필요
)
result = solver.solve()

kx_over_k0, ky_over_k0 = solver.order_kxy()
k0 = 2 * np.pi / solver.wavelength
print(k0 * kx_over_k0, k0 * ky_over_k0)             # 파수 단위: nm⁻¹
```

`order_kxy()`가 반환하는 값은 `k0`로 나눈 차수별 파수입니다. 여기서는 0차만 사용하므로 각각 원소가 하나이며, `k0`를 곱하면 `kx ≈ 0.00677139 nm⁻¹`, `ky = 0 nm⁻¹`입니다. **이 30°는 선택한 손실 입사 매질 안의 위상각**입니다. 계산한 전력 결과를 읽는 방법은 {ref}`13장 <half-space-power>`에서 설명합니다.

**무손실 매질의 식을 그대로 쓸 수 없는 이유**도 이 예제로 확인할 수 있습니다. `top=4+3j`의 복소 굴절률은 `n = 2.1213 + 0.7071i`입니다. 위의 무손실 식 `kx = k0 n sin(θ)`에 넣으면 `kx/k0 = 1.0607 + 0.3536i`인 복소수가 되어, 실수여야 하는 `kx`로 쓸 수 없습니다. 그래서 30°를 위상 진행 방향 `Re(k)`의 각도로 읽고, 다음 조건을 사용합니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>tan(θ) = k<sub>x</sub> / |Re(k<sub>z</sub>)|</p>
</div>
```

그런데 이 조건으로 `kx`를 구하려면 `Re(kz)`가 필요하고, `kz`는 파수 관계식 `kz² = k0² εr μr − kx²`에서 구하므로 다시 `kx`가 필요합니다. **두 조건이 서로 물려 있으므로 함께 풀어야 합니다.** `kx`를 바꿔 가며 두 조건을 확인하면 다음과 같습니다.

| `kx/k0` | 파수 관계식으로 구한 `Re(kz)`의 크기 / `k0` | 위상 진행 방향의 각도 |
| ---: | ---: | ---: |
| 1.0 | 1.9030 | 27.72° |
| 1.1 | 1.8556 | 30.66° |
| 1.077700 | 1.866632 | 30.00° |

위 계산식의 `N² = (a + sqrt(a² + b² / cos²(theta))) / 2`는 이 시행착오를 한 번에 푸는 닫힌 식입니다. `a=4`, `b=3`, θ=30°를 넣으면 `N = 2.155400`이고, `kx/k0 = N sin(30°) = 1.077700`, `Re(kz)`의 크기는 `N cos(30°) = 1.866632`로 표의 마지막 줄과 같습니다. 참고로 `n`의 실수부만 써서 `kx/k0 = Re(n) sin(30°) = 1.0607`로 두면 위상 진행 방향이 29.50°가 되어 30°에서 조금 어긋납니다.

(complex-transverse-wavevector)=
#### 복소 횡파수가 필요할 때: 복소 `theta`

`kx`, `ky`를 직접 입력할 때 복소수를 넣으면 거부합니다. 직접 입력은 **표면을 따라 세기가 균일한 조명**을 뜻하는데, 횡파수가 복소수이면 세기가 표면을 따라 변하기 때문입니다. 흡수성 입사 매질에서도 마찬가지이며, 손실은 `kz`의 허수부로 들어갑니다. 흡수성 입사는 위 예시처럼 **실수 `theta`로 지정하면** Wavebit이 실수 `kx`, `ky`를 구합니다.

거부할 때의 오류 메시지는 이 내용을 그대로 안내합니다.

- 직접 입력한 `kx`, `ky`는 균일한 조명을 뜻하므로 실수여야 합니다.
- 흡수성 입사에는 실수 `theta`, `phi`를 씁니다.
- 복소 횡파수는 복소 `theta`, `phi`로만 지정할 수 있으며, 그 역변환식을 함께 알려 줍니다.

복소 횡파수를 일부러 써야 한다면 복소 `theta`, `phi`를 넣습니다. 이때의 각도는 위상 진행 방향이 아니라 **형식적인 매개변수**이며, 입사 매질의 굴절률 n을 복소수 그대로 써서 다음과 같이 정합니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>k<sub>x</sub> / k<sub>0</sub> = n sin(θ) cos(φ), &nbsp; k<sub>y</sub> / k<sub>0</sub> = n sin(θ) sin(φ)</p>
</div>
```

원하는 `kx`에서 거꾸로 구하려면, `ky = 0`일 때 `theta = arcsin(kx / (k0 n))`, `phi = 0`으로 둡니다.

이 경로로 횡파수에 허수부가 생기면 결과를 읽는 방법이 달라집니다.

- **세기가 표면을 따라 변합니다.** `top=4+3j`, 진공 파장 1,000 nm에서 `kx/k0 = n sin(30°) = 1.0607 + 0.3536i`이면, 한 주기 1200 nm를 지나는 동안 세기가 약 1/207로 줄어듭니다.
- **`R`, `T`는 입사 전력에 대한 비가 아니라 형식적인 비율입니다.** 무손실 구조에서도 합이 1을 넘을 수 있습니다. 13장 예제 구조에서 `top`만 무손실 `4`로 바꾸고 `theta = 30° + 0.2i`를 넣으면 `T = 1.379`입니다.
- **`power_entering`과 `absorption`은 `NaN`으로 표시합니다.** 이유는 {ref}`13장 <half-space-power>`에서 설명합니다.
- **풀이마다 `ComplexIncidenceWarning` 경고를 한 번 냅니다.** 경고 분류가 따로 있으므로, 의도한 계산이라면 `warnings.filterwarnings('ignore', category=wavebit.ComplexIncidenceWarning)`로 이 경고만 끌 수 있습니다.
- **다른 솔버와 답이 다를 수 있습니다.** 위아래 반공간의 각 차수에서 `kz`의 두 근 중 무엇을 고를지가 규약이 되어, 같은 입력에서도 Wavebit, fmmax, RETICOLO가 서로 다른 반사계수를 낼 수 있습니다. 규칙마다 무엇이 다른지는 {ref}`부록 A <complex-kx-roots>`에서 무손실 매질로 설명하고, 흡수 입사 매질에서 실수 `theta`와 복소 `theta`가 왜 다른 `kx`를 주는지와 흡수·이득 반공간에서 달라지는 점은 {ref}`부록 B <complex-kx-lossy-gain>`에서 설명합니다.

소멸파 입사처럼 `theta = π/2 + iα` 꼴의 복소 각도는 `sin(θ) = cosh(α)`가 실수입니다. 입사 매질이 무손실이면 `kx/k0 = n cosh(α)`도 실수로 남으므로 `ComplexIncidenceWarning`이 나오지 않습니다. 이때 입사파 자체가 소멸파여서 표면 쪽으로 전력을 나르지 않으므로, 반사 전력과 투과 전력을 나눌 기준이 없습니다. 그래서 `R`, `T`, `power_entering`, `absorption`은 경고 없이 `NaN`이고, `r_s`, `t_s` 같은 진폭은 평소대로 반환됩니다. 입사 매질이 흡수성이면 n이 복소수라 `kx`도 복소수가 되고 경고가 나옵니다.

(absorbing-transmission)=
### 투과 매질에 손실이 있을 때

투과 매질에 들어가도 **같은 회절 차수의 `kx`, `ky`는 그대로 이어집니다.** 경계면에서 전기장과 자기장을 연결하려면 표면을 따라 변하는 위상이 서로 맞아야 하기 때문입니다. 0차의 횡파수는 입사파와 같고, 다른 차수에는 [6장](lattice.md)에서 설명한 역격자 파수가 더해집니다. 이렇게 정해진 차수별 횡파수는 입사 쪽과 투과 쪽에서 같습니다.

각 차수의 투과파는 다음 관계를 만족합니다. 아래첨자 t는 투과 매질을 뜻하며, ε와 μ는 상대 유전율·상대 투자율입니다. 식의 `kx`, `ky`는 그 차수의 실제 파수입니다.

```{raw} html
<div class="power-equations">
<p>k<sub>z,t</sub>² = k<sub>0</sub>² ε<sub>r,t</sub> μ<sub>r,t</sub> − k<sub>x</sub>² − k<sub>y</sub>²</p>
<p>k<sub>0</sub> = 2π / λ</p>
</div>
```

`kz`에는 두 부호의 해가 있습니다. 흡수성 투과 매질에서는 **구조에서 멀어지는 쪽으로 감쇠하는 해**를 선택합니다. Wavebit의 +z는 위쪽이므로, 아래쪽 `bot`으로 나가는 투과파는 z가 작아지는 쪽으로, 위쪽 `top`으로 나가는 투과파는 z가 커지는 쪽으로 감쇠합니다.

따라서 투과 매질에 손실이 있어도 실수인 `kx`, `ky`를 사용할 수 있습니다. 손실은 복소 `kz`의 허수부에 따른 감쇠로 나타납니다. **`theta`, `phi`는 입사파를 지정하는 값이며, 투과파의 각도를 따로 입력하는 옵션이 아닙니다.** 투과파의 위상 방향은 `Re(k)`에서 읽으며, 전력 흐름 방향과는 다를 수 있습니다.

(light-detail-id7)=
#### 예시: 공기에서 손실 매질로 투과하기

공기 `top=1`에서 위상각 30°로 입사하고, 아래쪽에는 `bot=4+3j`인 손실 매질을 둡니다. 두 반무한 매질의 μ는 1입니다. 유한 층도 공기로 두어 입사 쪽과 재료가 같게 했습니다. 파장과 모든 길이의 단위는 nm입니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

solver = wavebit.rcwa(
    top=1., bot=4. + 3j,
    incidence_side="top",
    wavelength=1000., period=[1200., 1200.],
    theta=np.deg2rad(30.), phi=0.,
    polarization=(1., 0.),                         # s 입사
    structure=np.ones((1, 2, 2)),                  # 공기인 유한 층
    thickness=[100.],
    truncation_order=0,                           # 균일층이므로 0차만 필요
)
result = solver.solve()

kx_over_k0, ky_over_k0 = solver.order_kxy()
k0 = 2 * np.pi / solver.wavelength
kx = k0 * kx_over_k0[0]
ky = k0 * ky_over_k0[0]

# 이 예제는 μ=1. 아래쪽으로 나가며 감쇠하는 kz를 선택합니다.
kz_transmitted = -np.sqrt(k0**2 * (4. + 3j) - kx**2 - ky**2)
print(kx, ky, kz_transmitted)
print(result.R.sum(), result.T.sum())
```

| 계산값 | 결과 |
| --- | --- |
| 입사파와 0차 투과파의 `kx` | 약 0.00314159 nm⁻¹ |
| 입사파와 0차 투과파의 `ky` | 0 nm⁻¹ |
| 투과파의 `kz` | 약 −0.01299294 − 0.00455768i nm⁻¹ |
| 투과파의 위상 방향과 아래쪽 법선(−z) 사이 각도 | 약 13.59° |
| 경계면 반사율 `R.sum()` | 약 0.215748 |
| 경계면 투과율 `T.sum()` | 약 0.784252 |

`kz`의 실수부가 음수인 것은 위상이 아래쪽으로 진행한다는 뜻입니다. 허수부도 음수이지만, 아래로 갈수록 z가 작아지므로 `exp(i kz z)`의 크기는 줄어듭니다. **음의 허수부를 이 예제에서 증폭으로 읽으면 안 됩니다.**

`T`는 유한 구조의 아랫면 `z=0`을 넘어 손실 매질로 들어가는 전력의 비입니다. 그 뒤에도 투과파는 계속 감쇠합니다. 이 단일 투과파가 경계면에서 수직으로 `d`만큼 더 깊이 들어가면, 표면에 수직인 전력 흐름의 크기는 경계면 값의 `exp(−2 |Im(kz)| d)`배가 됩니다. 이 예제에서 `d=100 nm`이면 약 0.4019배입니다.

입사 매질과 유한 층이 모두 무손실이므로 이 예제의 `R+T`는 1입니다. 반무한 투과 매질 안에서 이후에 흡수되는 전력은 유한 층의 `result.absorption`에 포함되지 않습니다. 전력의 기준면과 흡수 범위는 {ref}`13장 <half-space-power>`에서 설명합니다. 입사·투과 매질이 모두 손실성이어도, 입사 조건으로 횡파수를 정하고 각 매질의 ε·μ로 수직 파수를 구하는 순서는 같습니다.

균일한 다층막에서 횡파수의 연속성과 흡수성 외부 매질의 해 선택은 [Byrnes의 다층막 해설 §2·§5](https://arxiv.org/html/1603.02720v5#S2)에서도 확인할 수 있습니다. 그 문헌의 z축 방향은 이 예제와 다르므로 파수의 부호는 각 좌표 규약에 맞춰 읽습니다.
