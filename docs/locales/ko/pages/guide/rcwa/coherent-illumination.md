# 결맞는 조명과 내부 광원

:::{note}
아래의 모든 예제는 현재 `rcwa` 생애주기를 씁니다. 계산 그룹의 수는
`result.diagnostics['source_group_counts']`이며, 파장마다 하나의 개수입니다. 한 가지 양은
이전 `wavebit.compat.legacy` 인터페이스(2026-10-15에 퇴역)만 보고했고, 현재 결과는
아직 이를 담지 않습니다. 들어오는 파와 나가는 파의 간섭을 포함한 유한 영역의 순 플럭스
(`net_into_stack`)입니다. 해당하는 곳은 예제에 밝혀 두었습니다.
:::

평면파마다 방향, 편광 또는 위상이 따로 필요하거나, 빛이 두 반공간에서 모두 들어올 때는
`rcwa.Excitation(sources=...)`에 소스를 여러 개 넣으세요. 모든 빔은 여기의 진공 파장을
공유합니다. 계산은 선형 재료를 가정합니다.

```python
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length='nm'),                       # all lengths in nm
    stack=rcwa.Stack(
        period=(900., 850.),
        top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=100., background=Material(epsilon=4.+.6j)),)),
    excitation=rcwa.Excitation(
        wavelength=600.,
        sources=(
            rcwa.PlaneWave(side='top', polarization='s', kx=.002, ky=0.),  # nm^-1
            rcwa.PlaneWave(side='bottom', polarization='s', kx=.002, ky=0.,
                           amplitude=np.exp(1j*np.deg2rad(90.))))),
    outputs=rcwa.Outputs(diffraction=False, fields=True))
result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0)))  # a uniform slab
top, bottom = result.outputs['top'], result.outputs['bottom']
incident = top.incoming + bottom.incoming
output_top, output_bottom = top.outgoing / incident, bottom.outgoing / incident
print(output_top, output_bottom, result.absorption)
# [0.73987951] [0.05936313] [0.20075736]
```

`Stack.top`과 `Stack.bottom`은 여전히 층 적층 위와 아래의 재료를 기술합니다.
각 소스는 `side='top'` 또는 `'bottom'`으로 자신이 들어오는 면을 지정합니다. 층 순서는
`z=0`에서 시작해 아래에서 위로 그대로입니다. 소스가 둘 이상이거나 단일 `PlaneWave`가 아닌
소스가 하나라도 있는 계산은 차수별 `R`/`T` 대신 경계 전력과 요청한 필드를 보고하므로,
`Outputs(diffraction=False)`가 필요합니다. `result.outputs['top']`과 `result.outputs['bottom']`은
`incoming`과 `outgoing` 전력을 가진 `BoundaryPower` 기록이며, 출력 비율은 그 둘의
비입니다.

## 각 PlaneWave가 지정하는 것

| PlaneWave 인수 | 의미 |
| --- | --- |
| `side` | `'top'`(기본값) 또는 `'bottom'`: 빔이 들어오는 반공간입니다. |
| `theta`, `phi` | 라디안 단위의 방향 각도이며, 그 빔의 입사 쪽 재료를 기준으로 평가합니다. |
| `kx`, `ky` | 실수인 카르테시안 횡방향 파수 벡터 성분이며, 문제의 `Units`에서 길이의 역수 단위입니다. 같은 PlaneWave에서 각도와 함께 쓰지 마세요. |
| `polarization` | `'s'`, `'p'`, 또는 유클리드 길이로 정규화되는 전기장 s/p 계수 `(a_s, a_p)`입니다. 기본값은 `'s'`입니다. |
| `amplitude` | 그 빔의 정규화된 s/p 쌍에 곱하는 V/m 단위의 복소 배율입니다. `0.5*np.exp(1j*np.deg2rad(45.))`는 크기 0.5, 위상 +45°를 지정합니다. |

이전 인터페이스의 `psi`, `chi` 편광 각도는 `PlaneWave`의 필드가 아닙니다. 대신
Jones 쌍을 주세요.

각도는 적층 안쪽을 향하는 법선을 기준으로 합니다. 위에서는 −z, 아래에서는 +z입니다.
방위각은 두 면 모두에서 같은 xyz 좌표계로 횡방향을 정합니다.
방위각이 없는 수직 입사는 φ=0의 s/p 기저를 씁니다. s/p의 상대 부호는
Wavebit의 전기장 진폭 규약을 따르며, 수직 입사에서 그 기저 벡터는
두 입사 면 모두에서 s=+y, p=−x입니다.

방향이 없는 빔은 수직 입사입니다. 공유되는 기본값은 없습니다. 각 `PlaneWave`는
자신의 방향과 편광을 직접 밝히며, 그래서 예제에서 `kx`를 반복합니다. 한 면의 여러 평면파는
같은 `side`를 가진 여러 소스입니다.

PlaneWave 진폭의 기준은 **그 빔 쪽 바깥 경계면의 x=y=0**입니다.
top은 `z=sum(thickness)`, bottom은 `z=0`입니다. 두 빔의 위상차는 이 두
기준점 사이의 위상차입니다. 어느 기준면이든 평행 이동하려면 전파
위상 인자가 필요합니다. 두 매질, 각도, 편광의 어드미턴스가 서로 다를 수 있으므로,
진폭의 크기가 같다고 해서 일반적으로 법선 방향 전력이 같은 것은 아닙니다.

### 바깥 매질이 달라도 kx와 ky는 공통

예제에서 위쪽 굴절률은 1이고 아래쪽 굴절률은 1.5입니다. 같은
`kx=0.002 nm^-1`이 위쪽에서는 약 11.01°, 아래쪽에서는 7.31°의 각도를 줍니다.
두 빔 모두 같은 회절 격자 간격 위의 유효한 입력이며, 법선 방향 파수와
전력 가중치는 각자의 재료로부터 계산합니다.

기준점에서 아래쪽 빔의 위상을 바꾸면 다음과 같습니다.

| 아래쪽 위상 | `output_top` | `output_bottom` | `absorption` |
| --- | --- | --- | --- |
| 0° | 0.481181 | 0.242506 | 0.276313 |
| 90° | 0.739880 | 0.059363 | 0.200757 |
| 180° | 0.393406 | 0.411418 | 0.195176 |

재료와 두 개별 입사 전력은 바뀌지 않습니다. 간섭이
나가는 필드와 흡수 슬래브 안의 필드를 바꿉니다.

## 같은 면에서 들어오는 여러 빔

여러 `PlaneWave` 소스에 같은 `side`를 주세요. 입사 빔이 없는 반대쪽 면은 생략합니다. 그래도
두 바깥 재료는 여전히 `Stack.top`과 `Stack.bottom`에 둡니다. 모든 빔은
`wavelength`를 공유하지만, 방향, 편광, 진폭의 크기와 위상은 각각 따로 가질 수
있습니다.

다음 완전한 예제는 균일한 흡수 슬래브를 위에서
두 s 편광 빔으로 비춥니다. 하나는 수직이고, 다른 하나는 +x 쪽으로 20° 기울어져 있습니다.

```python
import numpy as np
from wavebit import Material, Units, rcwa

beams = (
    rcwa.PlaneWave(
        side='top', theta=0., phi=0., polarization='s', amplitude=1.,
    ),
    rcwa.PlaneWave(
        side='top', theta=np.deg2rad(20.), phi=0., polarization='s',
        amplitude=0.5 * np.exp(1j * np.deg2rad(90.)),
    ),
)
same_side = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(
        period=(900., 850.), top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=100., background=Material(epsilon=4.+.6j)),)),
    excitation=rcwa.Excitation(wavelength=600., sources=beams),
    outputs=rcwa.Outputs(diffraction=False, power_region='infinite'))
numerics = rcwa.Numerics(orders=(0, 0))  # uniform slab
same_side_result = rcwa.run(same_side, numerics=numerics)
top, bottom = same_side_result.outputs['top'], same_side_result.outputs['bottom']
print(top.outgoing / top.incoming)     # total reflectance
print(bottom.outgoing / top.incoming)  # total transmittance
print(same_side_result.absorption)
```

`beams[0]`과 `beams[1]`의 첨자는 튜플 위치이지 회절 차수가 아닙니다.
입력을 더 지정하려면 `PlaneWave` 객체를 더 추가하세요. 아래에서만 비추는 경우에는
`output_bottom`이 반사율이고 `output_top`이 투과율입니다.

두 번째 빔은 **위쪽 경계면의 x=y=0**에서 첫 번째 빔에 비해 전기장 진폭의 크기가 절반이고
위상이 +90°입니다. 횡방향 파수 벡터가 서로 다르므로
이 위상차는 위치에 따라 달라지며, 각 위치에서는 시간에 대해 고정되어 있습니다.
`polarization`은 한 빔 안의 s/p 성분 비를 정하고,
`amplitude`는 정규화된 쌍 전체에 곱해집니다. 그 입력을 p 편광으로 만들려면 두 번째 빔의
`polarization`만 `'p'`로 바꾸세요.

진폭의 크기는 전력 비율이 아닙니다. 이 무손실 등방성 입사
매질에서 각 빔의 안쪽 법선 방향 플럭스는 `abs(amplitude)**2 *
cos(theta)`에 비례합니다. 따라서 두 번째 빔은 첫 번째 빔의 법선 방향 입사 플럭스의 약 `0.25*cos(20°) = 0.235`배를
나릅니다.

이 두 방향은 두 개의 계산 그룹을 이루므로, 명시적인 `power_region`이
필요합니다. 국소 필드는 결맞은 상태로 남지만, 교차항은 무한 면적 평균에서
사라집니다. 유한한 직사각형에서의 간섭을 포함하려면, 그 직사각형을 `power_region`으로 두고
문제를 다시 계산하세요.

```python
window = same_side.replace(outputs=same_side.outputs.replace(
    power_region=((-450., 450.), (-425., 425.)),  # x and y bounds, nm
))
window_power = rcwa.run(window, numerics=numerics).outputs
print(window_power['top'].outgoing / window_power['top'].incoming,
      window_power['bottom'].outgoing / window_power['top'].incoming)
```

이 비율들은 같은 직사각형에서 평균한 입사파만의 플럭스를 분모로 씁니다.
유한 창의 전력과 흡수는 아래의 관측 영역 절을 참고하세요.

### PlaneWave 목록과 입사 차수

방향과 편광을 독립적으로 고를 때는 소스를 따로 쓰세요. 기준 방향에서 정수배의 역격자
벡터만큼 떨어진 방향들을 모두 같은 s/p 계수로 비출 때는 `IncidentOrders` 안의 `weights`를
쓰세요.

```python
order_problem = same_side.replace(
    excitation=same_side.excitation.replace(sources=(
        rcwa.IncidentOrders(
            side='top', theta=0., phi=0., polarization='s', amplitude=1.,
            weights=(((0, 0), 1.), ((1, 0), 0.5 * np.exp(1j * np.deg2rad(90.))))),
    )),
    outputs=rcwa.Outputs(diffraction=False, power_region='cell'),
)
order_result = rcwa.run(order_problem, numerics=rcwa.Numerics(orders=(1, 0)))
```

여기서 이름표 `(m, 0)`은 x 차수입니다: `kx_m = kx_reference + m*2*pi/Px`이고 `ky=0`입니다.
공기 중에서 파장 600 nm, 주기 900 nm일 때 +1 입력의 각도는 약 41.81°이며,
앞의 목록에 있던 20° 각도가 아닙니다. 2D에서 이름표 `(m, n)`은 첫 번째와
두 번째 역격자 벡터에 곱해집니다. 전체 전기장 s/p 진폭은
`amplitude * weight * normalized_polarization`이며, 각 차수의 가중치는 `weights`에서 옵니다.
차수마다 다른 편광을 주려면 평면파를 따로 지정하세요. 차수
이름표는 `Numerics(orders=...)` 안에 들어가야 하며, 각 입력은 안쪽 법선 방향으로 양의 전력을 날라야 합니다.

## 독립적인 각도와 계산 흐름

두 면에 서로 독립적인 각도를 지정할 수 있습니다.

```python
two_angles = problem.replace(
    excitation=rcwa.Excitation(wavelength=600., sources=(
        rcwa.PlaneWave(side='top', theta=np.deg2rad(20.), phi=0.),
        rcwa.PlaneWave(
            side='bottom', theta=np.deg2rad(35.), phi=0., amplitude=0.7j,
        ),
    )),
    outputs=rcwa.Outputs(diffraction=False, fields=True, power_region='infinite'),
)
result = rcwa.run(two_angles, numerics=rcwa.Numerics(orders=(0, 0)))
assert result.diagnostics['source_group_counts'] == (2,)   # two calculation groups
```

솔버는 먼저 각 빔의 물리적 횡방향 파수 벡터를 평가합니다. 두 입력의 차가 정수
m,n과 역격자 벡터 b1,b2에 대해 `m*b1+n*b2`와 같으면 두 입력은 회절 격자 간격을 공유할 수 있습니다.
이런 입력들이 **Bloch 섹터**, 곧
계산 그룹을 이룹니다. 이 관계를 만족하지 않는 입력에는 별도의 RCWA
행렬이 필요합니다. `result.diagnostics['source_group_counts']`는 그 수를 파장마다
하나씩 보고합니다.

![PlaneWave 방향을 그룹으로 묶고, 산란 행렬이 복소 경계 진폭을 만들며, 합한 E/H 필드가 관측 영역의 전력을 정합니다.](../../_static/coherent-illumination.svg)

각 그룹 안에서 솔버는 2포트 산란 행렬 하나를 만듭니다. 유지하는
직사각형 차수 창은 그 그룹의 모든 입력 주변에 `Numerics(orders=...)`만큼의 이웃 차수를
포함합니다. `IncidentOrders.weights` 이름표는 선언한 차수 범위 안에 있어야 하며,
따로 준 PlaneWave 방향들은 합친 창을 더 넓힐 수 있습니다. 따라서 서로 멀리 떨어진
입력은 단일 수직 입사 풀이보다 비용이 더 들 수 있습니다.

나가는 두 진폭에는 양쪽 면의 기여가 모두 들어 있습니다.

```{math}
\begin{pmatrix}b_\mathrm{bottom}\\ b_\mathrm{top}\end{pmatrix}
=
\begin{pmatrix}S_{11}&S_{12}\\S_{21}&S_{22}\end{pmatrix}
\begin{pmatrix}a_\mathrm{bottom}\\ a_\mathrm{top}\end{pmatrix}.
```

여기서 a는 이 계산의 경계 기저에서 들어오는 전기장 s/p 진폭이고, b는 나가는 진폭입니다.
서로 다른 그룹의 물리적 카르테시안 필드는
각자의 공간 위상을 붙여 재구성한 뒤 더합니다.

```{math}
\mathbf E=\sum_j\mathbf E_j,\qquad
\mathbf H=\sum_j\mathbf H_j,\qquad
\langle\mathbf S\rangle=\tfrac12\operatorname{Re}(\mathbf E\times\mathbf H^*).
```

이렇게 하면 결맞는 교차항이 유지됩니다. 풀이마다 효율을 계산해
더하면 그 항들이 버려집니다. 출력 전력에는 경계 필드만
필요하며, 층 전체의 필드를 복원하는 것은 선택 사항입니다.

## 전력을 평균할 면적 고르기

그룹이 여럿이면 `Outputs(power_region=...)`을 명시적으로 지정하세요. 그 중첩은 일반적으로
구조의 한 셀마다 반복되지 않으므로, 평균 면적이 질문의 일부가 됩니다.

| `power_region` | 평균 |
| --- | --- |
| `"cell"` | 중심에 놓인 격자 평행사변형입니다. 그룹이 하나일 때만 기본값입니다. |
| `"infinite"` | 무한한 횡단면에 대한 평균입니다. 서로 다른 횡방향 파수 벡터 사이의 교차 기여는 이 평균에서 0입니다. |
| `((xmin, xmax), (ymin, ymax))` | 문제의 길이 단위로 준 유한한 카르테시안 직사각형이며, 모든 교차항을 포함합니다. |

적분에는 푸리에 지수 함수의 해석적 적분을 씁니다. 무한 면적
평균은 결맞는 빔이라도 일부 간섭 기여를 **공간적으로** 없앱니다. 이는 국소 필드를
바꾸거나 빔이 결맞지 않다고 선언하는 것이 아닙니다. 유한
직사각형에서는 무한 면적 평균에서 사라지는 간섭이 보일 수 있습니다.

## 결과 읽기

결과는 출력을 **나가는 면**으로 이름 짓습니다. 위로 나가는 빛에는
위쪽 입력의 반사와 아래쪽 입력의 투과가 함께 들어 있을 수 있으므로,
반사나 투과라는 하나의 이름을 붙일 수 없습니다.

| 결과 | 의미 |
| --- | --- |
| `outputs['top'].incoming`, `outputs['bottom'].incoming` | 들어오는 파만의 전력이며, 적층 안쪽 방향이 양수입니다. |
| `outputs['top'].outgoing`, `outputs['bottom'].outgoing` | 나가는 파만의 전력이며, 적층에서 멀어지는 방향이 양수입니다. |
| `output_top` | 위로 나가는 전력을 두 면의 입사파만의 전력 합으로 나눈 값이며, 첫 번째 예제처럼 계산합니다. |
| `output_bottom` | 그에 대응하는 아래로 나가는 비율입니다. |
| `absorption` | 적층으로 들어가는 순 전력을 입사 전력으로 나눈 값이며, 셀에서 평균한 하나의 주기 그룹이나 무한 평면 평균에 대해 정의됩니다. 유한 직사각형이나 여러 그룹의 셀 평균에서 이를 읽으면 `UnsupportedFeatureError`가 발생합니다. |

이전 인터페이스는 들어오는 파와 나가는 파의 간섭을 포함해, 아래쪽 전체 z 플럭스에서 위쪽 전체 z 플럭스를
뺀 `power.net_into_stack`도 보고합니다. 현재 결과는 보고하지 않습니다.

전력 값은 SI 단위입니다. 진폭이 V/m이면 `incoming`과 `outgoing`은 셀이나 직사각형을
지나는 와트(`units='W'`)이고, `'infinite'`에서는 W/m² 단위의 평균
(`units='W/m^2'`)입니다. 출력 비율에서는 단위가 약분됩니다.

바깥 매질이 무손실이고 주기 평균이나 무한 평면 평균이면
`absorption = 1 - output_top - output_bottom`입니다. 바깥 매질이 흡수하면
`absorption`을 쓰세요. 순 플럭스에 들어오는 파와 나가는 파의 간섭도 들어 있을 수 있기 때문입니다.
유한한 관측 직사각형은 옆면을 통해 전력을 주고받을 수 있으므로,
두 수평 경계의 플럭스만으로는 전체 흡수를 정하지 않습니다.
유한 직사각형을 지나는 나가는 파만의 플럭스는 국소적으로 음수가 될 수도 있습니다.
이런 영역 비율은 확률처럼 범위가 제한된 효율이 아닙니다.

```python
regional = rcwa.run(two_angles.replace(outputs=two_angles.outputs.replace(
    power_region=((-400., 400.), (-300., 300.)))), numerics=rcwa.Numerics(orders=(0, 0)))
top, bottom = regional.outputs['top'], regional.outputs['bottom']
print(top.outgoing / (top.incoming + bottom.incoming))
# net_into_stack is reported only by the earlier interface's result.boundary_power(region).
```

## 경계 필드와 내부 필드

필드는 각 위치에서 모든 입력 빔의 결맞는 합을 포함한 복소 전기장 성분과 자기장 성분을
줍니다. 전력 비율이 아닙니다.
경계 필드와 내부 필드에는 모두 `Outputs(fields=True)`가 필요하며, 이 설정은 나중의 표본 추출에
필요한 모드 데이터를 보관합니다. 공간 격자를 미리 계산하지는 않습니다.

이 설정이 없어도 결과는 선언한 `power_region`에서 평가한 전력을 담고 있습니다. 다만
복소 경계 필드는 보관하지 않으며, 현재 결과에는 다른 영역에서 그 필드를 적분하는 메서드가
없습니다. 위에서처럼 새 영역은 새 계산입니다.

### 경계 필드

다음 예제는 위의 100 nm 슬래브를 이어서 씁니다.

```python
x = np.linspace(-300., 300., 61)
y = np.array([0.])

incoming = result.boundary_fields(side='top', x=x, y=y, part='incoming')
outgoing = result.boundary_fields(side='top', x=x, y=y, part='outgoing')
above = result.boundary_fields(side='top', x=x, y=y, part='total')

print(above.values.shape)  # (1, 1, 61, 6); above = incoming + outgoing
```

| 인수 | 의미 |
| --- | --- |
| `side` | 필수: `"top"` 또는 `"bottom"`. 해당 반공간 쪽에서 다가간 그 바깥 경계면의 필드를 반환합니다. |
| `x`, `y` | 필수: 양의 정수 표본 개수, 또는 1차원 물리 좌표 배열. `[0.]`은 셀 중심입니다. |
| `part="incoming"` | 선택한 면에서 적층으로 들어오는 파만. |
| `part="outgoing"` | 그 면을 통해 적층을 떠나는 모든 파. 양면 조명에서는 그 면 입력의 반사와 반대쪽 면 입력의 투과가 합쳐집니다. |
| `part="total"` | 들어오는 필드와 나가는 필드의 합이며, 기본값입니다. |

이 슬래브에서 `"top"`은 위쪽 반공간에서 z=100 nm를, `"bottom"`은
아래쪽 반공간에서 z=0을 표본합니다. 이 메서드는 반공간 안의 임의 높이를
받지 않습니다.

### 내부 필드

위의 문제는 이미 `Outputs(fields=True)`를 요청합니다. 이 설정 없이 계산한 결과에서는
표본을 읽을 수 없으므로, 내부 표본을 읽기 전에 이 설정을 넣어 문제를 다시 계산하세요.

```python
inside = result.field(x=x, y=y, z=np.array([0., 50., 100.]))

print(inside.values.shape)  # (1, 3, 1, 61, 6)
middle = inside.values[0, 1]  # z=50 nm plane, shape (1, 61, 6)
```

| 인수 | 의미 |
| --- | --- |
| `x`, `y` | 필수: 양의 정수 표본 개수, 또는 1차원 좌표 배열. |
| `z` | 필수: 층마다의 양의 정수 개수, 또는 적층 바닥에서 위쪽으로 잰 `[0, sum(thickness)]` 범위의 절대 높이 배열. |

내부 필드는 두 진행 방향과 모든 빔의 기여를 포함합니다.
`part` 인수는 없습니다. 층 두께가 100과 200이면 층들은
z=0–100과 z=100–300을 차지하며, z가 층마다 다시 시작하지 않습니다. z=100 같은 내부 경계면
점은 경계면 위의 층을 씁니다. 바깥쪽 두
끝점에서는 가장 아래 층과 가장 위 층의 값을 반환합니다.

따라서 `boundary_fields(side='top', ...)`와 `field(..., z=[sum(thickness)])`는 같은
경계면에 서로 다른 재료 쪽에서 다가갑니다. 법선 방향 E/H 성분은 불연속일 수 있고, 접선 방향 Ex, Ey,
Hx, Hy는 유지한 푸리에 표현에서 연속입니다.

### 좌표, 배열 축과 성분

두 메서드는 [공통 필드 표본 규칙](fields.md#coordinates)을 공유합니다.
정수 개수는 셀을 표본하며, z 개수는 층마다 따로 표본합니다.
`[0.]`이나 `np.linspace(...)` 같은 명시적 좌표는 문제의 길이 단위와
셀 중심 기준의 카르테시안 x,y를 쓰며, 사각이 아닌 격자에서도 마찬가지입니다. x와 y 좌표의
모든 조합을 평가하고, 내부 표본은 주어진 각 z에서 그 평면을
요청한 순서대로 반복합니다.

| 메서드 | `.values`의 모양 |
| --- | --- |
| `boundary_fields` | `(n_wavelengths, Ny, Nx, 6)` |
| `field` | `(n_wavelengths, Nz, Ny, Nx, 6)` |

두 메서드 모두 공개 xyz 좌표계의 복소 `(Ex, Ey, Ez, Hx, Hy, Hz)`를 반환하며, E는 V/m,
H는 A/m 단위입니다. Wavebit은 처음부터 정규화된 자기장
`H_tilde = Z0 H`로 방정식을 세우며(진공 임피던스 Z0 ≈ 376.73 Ω), 자기장 성분을
반환하기 전에 Z0로 나눕니다. 이전 인터페이스는 `Z0 H`를 반환했습니다.

```python
from wavebit.units import Z0

Ey = inside.values[0, 1, 0, :, 1]  # z=50 nm, y=0, all requested x positions
Ey_amplitude = np.abs(Ey)
Ey_phase_deg = np.angle(Ey, deg=True)
E_squared = np.sum(np.abs(inside.values[..., :3])**2, axis=-1)  # (1, 3, 1, 61)
H = inside.values[..., 3:]  # A/m, shape (1, 3, 1, 61, 3)
H_normalized = Z0 * H  # the normalized Z0 H the earlier interface returned
```

`Ey.real`은 t=0에서의 순간 필드입니다. 다른 시각에서는
`Re[Ey exp(-i omega t)]`를 쓰세요. `E_squared`는 전기장 크기의 제곱입니다.
전력 플럭스에는 `Re(E × H*)/2`를 통해 E와 H가 모두 필요합니다.

필드는 면적 평균하거나 입사 전력으로 나눈 값이 아닙니다. 모든
입력 빔의 진폭을 두 배로 하면 이 필드도 두 배가 됩니다. `power_region`만 바꾸면
고정된 위치의 필드는 바뀌지 않습니다. 페이저 해석은 [필드와 회절 관측값](fields.md)을
참고하세요. 단일 평면파의 결과에도 같은 표본 메서드가 있습니다.

### 자기장 정규화와 물리 단위

솔버 내부에서 `H_tilde = Z0 H`를 쓰면 전기장과 자기장이 같은 단위
스케일에 놓입니다. 공개 결과는 이를 되돌립니다. 입력 전기장 진폭이 V/m이면 반환되는
자기장 성분은 A/m입니다. 예를 들어 입력 진폭 1 V/m에서 1인 정규화된 자기장 성분은
약 0.00265 A/m로 반환됩니다. 입력 진폭이 상대값이면 E와 H가 모두 절대값이 되기 전에
물리적인 전기장 기준이 필요합니다.

양의 상수 Z0로 나누면 크기와 단위는 바뀌지만
공간 패턴과 위상은 보존됩니다. 시간 평균 전력 플럭스는 다음과 같습니다.

```{math}
\langle\mathbf S\rangle
=\frac12\operatorname{Re}(\mathbf E\times\mathbf H^*)
=\frac{1}{2Z_0}\operatorname{Re}
  (\mathbf E\times\widetilde{\mathbf H}^{\,*}).
```

E가 V/m이고 H가 A/m이면 이 값은 W/m²입니다. `BoundaryPower` 값에는 이미
`1/Z0` 인자와 셀 또는 직사각형의 면적이 들어 있습니다. 와트이며, 무한 평면 평균에서는
W/m²입니다. 이 인자는 전력 비에서 약분되므로, 반환된 R/T나
`output_top`/`output_bottom`을 Z0로 나누지 마세요.

다른 RCWA 패키지들은 이전 인터페이스처럼 정규화된 단위로 자기장을 반환하며,
흔히 이름을 그냥 `H`라고 씁니다. 그들과 비교하려면 현재 H에 Z0를 곱하세요.

| 패키지 | 자기장 규약 |
| --- | --- |
| <span style="white-space:nowrap">Meent</span> | 정규화된 스케일은 [재료 행렬](https://github.com/kc-ml2/meent/blob/main/meent/on_numpy/emsolver/transfer_method.py)과 [필드 재구성](https://github.com/kc-ml2/meent/blob/main/meent/on_numpy/emsolver/field_distribution.py)에서 나옵니다. 반환되는 자기장 성분은 A/m로 변환되지 않습니다. |
| <span style="white-space:nowrap">S4</span> | [원 논문 §3.1](https://web.stanford.edu/group/fan/publication/Liu_ComputerPhysicsCommunications_183_2233_2012.pdf)은 SI 자기장에서 `Z0 H_SI`로의 변환을 명시적으로 정의합니다. |
| <span style="white-space:nowrap">FMMAX</span> | [문서화된 규약](https://facebookresearch.github.io/fmmax/#fmm-conventions)은 진공 유전율, 투자율, c를 1로 두므로 같은 상대 E/H 단위 스케일을 줍니다. |
| <span style="white-space:nowrap">RETICOLO</span> | [설명서 p. 4](https://arxiv.org/pdf/2101.00901#page=4)는 Maxwell 방정식에서 진공 유전율, 투자율, c를 1로 둡니다. `res3()`가 반환하는 H 성분은 그 단위계를 씁니다. |

패키지 사이에서 복소 필드 값을 비교하려면 입력
진폭 스케일, 좌표축, 위상 규약도 맞춰야 합니다.

## 실행 범위와 검증

소스가 둘 이상이거나 단일 `PlaneWave`가 아닌 소스가 하나라도 있는 계산은
카르테시안 **SMM 기반 1차 정식화**를 씁니다. NumPy, JAX, PyTorch를 즉시 실행에서
지원합니다. PlaneWave 그룹 묶기에는 구체적인 방향이 필요하므로, `jax.jit`나
`jax.vmap` 아래에서 추적된 입력은 거부합니다(코드 `rcwa.coherent_transform`). 즉시 실행의 `jax.grad`는
PyTorch처럼 그 값을 읽어 미분합니다.
`torch.compile` 아래에서는, 모든 공개 wavebit 호출 가능 객체가 그렇듯 컴파일된 호출자 안에서
즉시 실행됩니다([`torch-compile.md`](../../development/torch-compile.md) §6). 영역 유전율, 층 두께,
반지름에 대한 즉시 실행 미분은 두 평면파에 대해 두 백엔드 모두에서 중심
차분과 비교했습니다(`tests/api/test_eager_grad_parity.py`). 위쪽 면 `GaussianBeam`(위쪽 면에서의
위상이 적층 높이에 따라 움직입니다)의 층 두께 미분, `ElectricDipole`이 있는 계산(호스트 층과 그 위 층)의
층 두께 미분, 변화시킨 층 위의 내부 필드의 층 두께 미분, 그리고
`GaussianBeam` 편광에 대한 미분도 비교했습니다(`tests/rcwa/test_source_derivatives.py`).
`ElectricDipole`의 `position` 모든 성분, `moment`의 실수부와 허수부,
`GaussianBeam`의 `focus` 모든 성분에 대한 미분은, 패턴 층을 가진 2층 적층에서
`bloch_k`가 있거나 없는 주기 쌍극자, 고립 쌍극자, 어느 쪽 면에서든 비추는 빔에 대해
`jax.grad`, `jax.jacfwd`, PyTorch에서 필드와 전력을 대상으로 중심
차분과 비교했습니다(`tests/rcwa/test_source_placement_gradients.py`). 복소
`moment` 잎과 실수 결과 L에 대해 PyTorch의 그래디언트는 `dL/dRe(p) + 1j*dL/dIm(p)`이고,
JAX의 그래디언트는 그 켤레입니다. 쌍극자의 호스트 층은 z의 값으로 정하며
미분할 수 없습니다. z에 대한 미분은 그 층 안에서의 미분이고, 경계면 위의 쌍극자는
미분 여부와 상관없이 거부하며(한쪽 미분은 반환하지 않습니다), 한 평면에 놓인 두 쌍극자는
어느 쪽의 높이에 대한 미분도 거부합니다. `GaussianBeam`은 구체적인 방향들로 각 스펙트럼을 만들므로,
그 방향(`theta`, `phi`, `kx`, `ky`), 그 `waist`(그 방향들을 스케일합니다), 파장,
빛을 받는 반공간의 재료에 대한 미분은 부분적인 값을 반환하는 대신 거부합니다. Bloch 섹터를 고르는
쌍극자의 `bloch_k`에 대한 미분도 마찬가지로 거부합니다(코드 `rcwa.unsupported`).
그 밖의 입력과 소스는 비교하지 않았습니다. 이 경로는 단일 평면파의 `R`, `T`, 차수별
진폭이나 산란 행렬을 노출하지 않습니다. `Outputs(diffraction=True)`는 거부하고(코드
`rcwa.coherent_diffraction`), `Outputs(scattering=True)`도 거부합니다(코드
`rcwa.coherent_scattering`). 준비된 문제는 셀, 파장, 소스
방향과 표본, 반공간 응답을 고정하며, 그중 하나를 바꾸려면 새 `prepare`가 필요합니다
(코드 `prepared.source_grouping`). 진폭, 편광, 쌍극자 위치와 모멘트,
빔 초점, 층 재료는 계속 바꿀 수 있습니다.

외부 평면파는 파장이 하나이고, kx/ky가 실수이며, 입사 법선 방향 전력이 양수입니다.
바깥 반공간은 균일하고 등방성으로 남습니다. 수동적인 흡수 반공간과 자성
반공간은 지원합니다. 이 기능은 이득 매질의 가지 처방을
확립하지 않습니다. 층 재료에는 Li/NVF와 이방성 제한이 그대로 적용됩니다.
이 경로는 레일리 특이점에서 파장을 디튜닝하지 않으며,
`Numerics(detuning=...)`를 거부합니다(코드 `rcwa.coherent_detuning`). 맞춤이 특이해지면
가까운 공통 매개변수 점을 골라 검증하세요.

이전 인터페이스의 차수 키 맵 `excitation={0: 1., 1: .5j}`는 일반적인
단면 솔버를 실행합니다. 그 복소 가중치는 `IncidentOrders.weights`와 똑같이 위에서 밝힌
중심 기준을 쓰므로, 두 형식은 같은 가중치를 받고 같은 빛을
비춥니다(2026-09-27부터. 그 전에는 일반 가중치의
기준이 셀 모서리였습니다). 그 입구는 차수별 `R`,
`T`와 진폭도 답하는데, 현재 `IncidentOrders` 계산은 아직 그렇지 않습니다. 같은 면에서 같은 가중치일 때
그 필드와 경계파는 이 경로의 것과 같습니다.

검증은 두 편광에 대한 독립적인 4방정식 슬래브 맞춤,
서로 다른 자성 반공간과 흡수 반공간, 위상에 따른 슬래브 흡수,
과거의 단면 필드와 효율 극한, 회전한 텐서 층, 사각이 아닌 패턴 셀에서의 결맞는
필드 중첩, 역격자 차수 그룹 묶기,
유한 면적 적분, 그리고 지수 함수가 커지지 않는 두꺼운 흡수층을 다룹니다.
[테스트](../../../tests/rcwa/test_coherent_illumination.py)와
[실행 가능한 예제](../../../examples/rcwa/coherent_illumination.py)를 참고하세요. 예제는
현재 `rcwa` 생애주기를 쓰며 그룹 수를 `source_group_counts`에서 읽습니다.

## 소스 객체

`rcwa.Excitation(sources=...)`은 비어 있지 않은 소스 튜플을 받습니다. `PlaneWave`는
정확히 하나의 입사 평면파를 나타냅니다. `IncidentOrders`는 역격자 차수 가중치로
여러 평면파를 나타냅니다. `GaussianBeam`은 국소화된
외부 입사 필드를 기술하고, `ElectricDipole`은 주어진 내부 전류를 기술합니다.
한 계산의 모든 소스는 파장을 공유하며 결맞게 결합됩니다.

```python
sources = (
    rcwa.PlaneWave(side='top', kx=.002, ky=0., amplitude=1.),
    rcwa.PlaneWave(side='bottom', kx=.002, ky=0., amplitude=.5j),
)
# Pass rcwa.Excitation(wavelength=600., sources=sources) to rcwa.Problem.
```

`side`의 기본값은 `'top'`입니다. `ElectricDipole`에는 면이 없습니다. 그 `position`이
쌍극자를 층 안에 놓습니다.

## 가우시안 각 스펙트럼

`GaussianBeam`은 `exp(-waist**2 * (ku**2 + kv**2)/4)`로 가중한 전파하는 각 스펙트럼을
씁니다. 여기서 ku와 kv는 중심
진행 방향에 수직입니다. 각 전기장 벡터는 자신의 파수 벡터에 수직으로
투영되며, 그 자기장은 Maxwell 방정식에서 나옵니다.
전파할 수 없거나 적층에서 멀어지는 방향을 향하는 성분은 제외합니다.
이렇게 하면 벡터 빔이 정의되며, 각도 퍼짐이 크면 포락선이 정확히 스칼라
가우시안일 필요는 없습니다.

```python
gaussian = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(
        period=(900., 850.), top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=100., background=Material(epsilon=4.)),)),
    excitation=rcwa.Excitation(wavelength=600., sources=(
        rcwa.GaussianBeam(
            side='top', theta=0., phi=0., polarization='s',
            waist=1500., focus=(0., 0., 100.), sampling=(9, 9),
        ),
    )),
    outputs=rcwa.Outputs(diffraction=False,
                         power_region=((-3000., 3000.), (-3000., 3000.))),
)
gaussian_result = rcwa.run(gaussian, numerics=rcwa.Numerics(orders=(0, 0)))
```

`waist`는 스칼라 포락선의 1/e 진폭 반지름입니다. 양의 실수, 또는 그런 값 하나를 담은
0차원 배열이나 텐서입니다. `focus`는 문제의 길이 단위로 준 전역 xyz 좌표이고,
`amplitude`는 구조가 없을 때 초점에서의 전기장 벡터 노름과 공통 위상입니다. 입사 매질은 등방성이고
무손실이며 양의 epsilon과 mu를 가져야 합니다. 음굴절률 쪽에서 쏘는 빔은
거부합니다. 그 각 스펙트럼, 횡파 조건, 초점 위상은 위상과 전력이 함께 진행하는 파를
위해 작성되었기 때문입니다. 음굴절률 쪽에서 오는 `PlaneWave`는 지원하며,
이때 `theta`와 `phi`는 단일 평면파에서와 같은 뜻, 곧 빛의 방향입니다
([`negative-index-media.md`](../../theory/rcwa/negative-index-media.md) §7). 따라서 그런 빔은 그 쪽에서 비추는
`PlaneWave` 소스들로 만들 수 있습니다. `sampling`은 중점 격자의
크기를 줍니다. `angular_spectrum_extent`(기본값 4)는 빔의 각 횡방향에서
+/-angular_spectrum_extent/waist 범위를 덮습니다. 두 매개변수 모두 수렴을 확인하세요. 유한한 k 간격은
공간 복제를 만들므로, 표본 수를 늘려 복제를 관측 영역에서 멀리 보내세요.

유한한 직사각형 `power_region`이 필요합니다. 출력 비율은 같은 직사각형에서 모은
전력끼리 비교합니다. 전체 빔 전력에 다가가려면 모으는 면적을 넓히세요.
이들은 평면파 세기의 결맞지 않은 평균이 아니라 결맞는 필드입니다. 각 성분에는
스칼라 근축 H 근사를 쓰지 않습니다.

## 층 안의 전기 쌍극자

`ElectricDipole`은 C m 단위의 복소 전기 모멘트 p를 지정하며, 자기 일관적으로 유도되는
입자 분극률이 아닙니다. exp(-i omega t) 규약에서 그
전류는 J=-i omega p delta(r-r0)입니다. 모든 위치 좌표는 전역 xyz이며,
z=0은 첫 번째 층의 바닥이고 +z는 위쪽을 향합니다.

```python
dipole = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(
        period=(900., 850.), top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=300., background=Material(epsilon=1.)),)),
    excitation=rcwa.Excitation(wavelength=600., sources=(
        rcwa.ElectricDipole(
            position=(0., 0., 150.), moment=(0., 1e-29, 0.),
            repetition='periodic', bloch_k=(0., 0.),
        ),
    )),
    outputs=rcwa.Outputs(diffraction=False, fields=True, power_region='cell'),
)
dipole_result = rcwa.run(dipole, numerics=rcwa.Numerics(orders=(1, 1)))
emitted_up = dipole_result.outputs['top'].outgoing  # W through one cell
print(emitted_up / (900e-9 * 850e-9))  # mean W/m^2
```

`Units`는 길이 단위당 미터 수를 정하며, SI 모멘트에는 이것이 필요합니다. 전기장은
V/m, 자기장은 A/m입니다. 외부 파와 결맞게 섞을 때는 외부 파의
전기장 진폭도 V/m로 주세요. `BoundaryPower`는 셀이나
직사각형을 지나는 와트를 보고합니다. 평균 W/m^2를 얻으려면 그 면적(m^2)으로 나누세요.

호스트 층은 epsilon과 mu 모두에 대해 영역이 없거나, 모든 공간 칸에서 재료 값이
같은 래스터여야 합니다. 균일한 텐서
재료는 지원하며, 다른 층에는 패턴이 있어도 됩니다. 소스는
층 안쪽에 엄격하게 있어야 합니다. 솔버는 그 높이에서 층을 나누고, 적분한
전류 불연속을 넣은 뒤, 아핀 산란 사상(산란 행렬과 소스가 만든 나가는 진폭)을
연쇄합니다. 여기에는 방출의 다중 반사가 포함되며, 여러 내부 소스와 외부 소스를 함께
쓸 수 있습니다. 특이한 소스 평면 자체에서의 필드는
거부합니다. 그 위나 아래에서 표본하세요.

`repetition="periodic"`은 모든 격자 셀에 위상
exp(i bloch_k dot R)을 가진 사본을 놓습니다. bloch_k를 생략하면 0입니다. 모든
역격자 벡터 G에 대해 exp(i G dot R) = 1이므로, bloch_k와 bloch_k + G는 같은 배열을 기술합니다. bloch_k는
역격자 벡터만큼 줄여 원점 중심의 역격자 셀 안으로 옮기며(줄인 각 좌표는
[-1/2, 1/2)에 있습니다. 직사각형 격자에서는 제1 브릴루앙 영역입니다), 쌍극자의
유지 차수는 그곳을 중심으로 합니다. 외부 파와 Bloch 섹터를 공유하는 쌍극자는, 다른 파가
그러하듯 자신의 중심 차수를 유지하도록 그 섹터의 차수 창을 넓힙니다. `repetition="isolated"`는
대신 기본값이 `sampling=(5,5)`인 브릴루앙 영역 중점 구적법을 씁니다. 이 경우에는
bloch_k를 주지 마세요. 유한한 관측 직사각형이 필요합니다. 유한
구적법은 더 큰 슈퍼셀을 반복한 것과 같습니다. 격자와
푸리에 차수의 수렴을 확인하고, 복제가 관심 영역 밖에 있도록 하세요. 도파 모드나
레일리 특이점에는 여전히 잘 정의된 극한 처방이 필요합니다.
소스 인터페이스는 파장을 조용히 바꾸거나 손실을 더하지 않습니다.

내부 소스가 하나라도 있으면 출력 비율은 정의되지 않으며, `result.absorption`을 읽으면
`UnsupportedFeatureError`가 발생합니다. 정규화하지 않은 들어오는/나가는 경계 전력은 계속
쓸 수 있습니다(이전 인터페이스는 `net_into_stack`도 보고합니다). 경계의 순 플럭스에는
소스의 공급과 재료 손실이 함께 들어 있으며, 재료 흡수만을 뜻하지 않습니다. 소스가 한 일,
정칙화한 자기장(self-field), Purcell 인자는 반환하지 않습니다. 모든 형식화된 소스는 즉시 실행
1차 SMM을 씁니다. `jit`와 자동 미분은 이 API의 지원 범위 밖입니다.

검증에는 세 가지 전기 모멘트 방향 모두에 대한 균질 다이애드 Green 함수 스펙트럼,
SI 길이 스케일링, 접선 방향 전류 불연속,
내부/외부 중첩, Bloch 구적법과 명시적 슈퍼셀의 비교,
그리고 NumPy, JAX, PyTorch에서의 가우시안 초점 프로파일과 진공 투과가 포함됩니다.
[소스 테스트](../../../tests/rcwa/test_sources.py)를 참고하세요.
