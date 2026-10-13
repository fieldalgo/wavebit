# 14. 위치별 전기장과 자기장 구하기

:::{note}
이 교재의 실행 예제는 원래 수치 기록을 재현할 수 있도록 `wavebit.compat.legacy` 인터페이스를 명시적으로 사용합니다. 새 작업은 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 참고하세요. 기존 측정값은 이번 문서 이행에서 다시 계산하지 않았습니다.
:::

반사율·투과율은 빛이 전달하는 전력의 비율입니다. 층 안에서 전기장과 자기장이 어떻게 분포하는지 보려면, 계산이 끝난 뒤 원하는 위치에서 **필드**를 읽습니다. 필드를 읽는다는 것은 그 위치의 전기장·자기장 성분을 구한다는 뜻입니다.

일반 입사, 여러 평면파, 가우시안 빔, 다이폴 소스 모두 `solve()`가 반환한 **결과 객체**에서 필드를 읽습니다. 이 장에서는 계산 준비, 위치 지정, 반환 배열, 경계면의 장을 차례로 설명한 뒤 복소수의 의미와 자기장 단위를 다룹니다.

(field-preparation)=
## 계산 결과에서 필드 읽기

층 내부의 필드가 필요하면 `answers="field"`로 계산합니다.

```python
result = solver.solve(answers="field")
cell = result.reconstruct_field(sample_x=41, sample_y=21, sample_z=20)
```

`result`는 이 계산에 사용한 파수·두께와 층별 계산 상태를 보관합니다. 읽을 위치만 바꿀 때는 다시 풀 필요가 없습니다. 같은 결과에서 여러 위치의 필드를 반복해서 읽을 수 있고, `solver`의 설정을 바꾸거나 다시 계산해도 앞서 얻은 `result`는 처음 계산한 조건의 필드를 반환합니다.

`answers`에 따라 계산 후 읽을 수 있는 정보는 다음과 같습니다. 이 규칙은 입사파와 소스를 지정하는 방식에 관계없이 같습니다.

| `answers` 설정 | 전력 결과 | 경계면 필드 | 층 내부 필드 |
| --- | --- | --- | --- |
| `"efficiency"` | 보관 | — | — |
| `"amplitude"` — 기본값 | 보관 | 보관 | — |
| `"field"` | 보관 | 보관 | 보관 |

경계면의 전기장·자기장은 `result.boundary_fields()`로 읽으며, 계산할 때 `answers="amplitude"` 또는 `"field"`를 지정해야 합니다. 층 내부의 전기장·자기장을 `result.reconstruct_field()`로 읽으려면 `answers="field"`가 필요합니다.

`answers="efficiency"`에서는 전력을 계산하는 데 사용한 복소 장을 결과에 보관하지 않습니다. 필드가 필요해졌다면 필요한 `answers` 설정으로 다시 `solve()`합니다.

소스를 지정한 결과에서 `result.boundary_power(region)`으로 관측 영역을 바꾸어 전력을 다시 구할 때도 `"amplitude"` 이상이 필요합니다. 처음 지정한 영역에서 이미 구한 전력은 모든 설정에서 `result.power`로 읽을 수 있습니다.

(ordinary-field-sampling)=
## 층 내부의 필드 구하기

### 실행 예제

두께 120 nm인 균일한 층에 TE 편광으로 입사하는 예제입니다. `solve()`로 먼저 계산한 다음, 층 바닥부터 윗면까지 25개 높이에서 필드를 읽습니다. 이 예제의 길이 단위는 nm입니다.

실행 파일: {download}`03_fields.py <../../../examples/rcwa/learning/learn/03_fields.py>`

```{literalinclude} ../../../examples/rcwa/learning/learn/03_fields.py
:language: python
:start-at: import numpy as np
:end-at: Ey = np.asarray(cell[:, 0, 0, 1])
```

`cell`의 모양은 `(25, 1, 1, 6)`입니다. 마지막 축의 두 번째 성분이 `Ey`이므로, `cell[:, 0, 0, 1]`은 25개 높이에서 읽은 전기장 y 성분입니다. 균일한 층에 수직 입사하므로 이 예제의 장은 x·y 위치에 따라 달라지지 않습니다.

```{figure} ../../figures/learn/learn/12-field-data.svg
:alt: 두께 120 nm인 균일한 층에서 높이에 따른 Ey의 실수부, 허수부와 크기의 제곱
:figclass: learning-figure

그림 14-1. 예제에서 구한 전기장 y 성분. 복소 진폭의 실수부·허수부와 크기의 제곱을 각각 표시합니다.
```

(field-sample-coordinates)=
### 읽을 위치 지정하기

`sample_x`, `sample_y`, `sample_z`에는 **점의 개수**를 정수로 넣거나, **좌표 자체**를 1차원 배열로 넣습니다. `reconstruct_field()`에서는 생략한 항목이 각각 `20`입니다.

| 입력 | 의미 | 예 |
| --- | --- | --- |
| `sample_x` | 셀의 x 방향 점 개수, 또는 x 좌표 배열 | `41`, `[-100., 0., 100.]` |
| `sample_y` | 셀의 y 방향 점 개수, 또는 y 좌표 배열 | `21`, `[0.]` |
| `sample_z` | 층마다 배치할 점 개수, 또는 층 전체를 기준으로 한 z 좌표 배열 | `20`, `[0., 60., 120.]` |

위 예제에서 셀 중심의 세 높이만 읽으려면 다음처럼 씁니다.

```python
three_heights = result.reconstruct_field(
    sample_x=[0.], sample_y=[0.], sample_z=[0., 60., 120.],
)
print(three_heights.shape)  # (3, 1, 1, 6)
```

`sample_z=20`은 **각 층에 20개씩** 점을 배치합니다. 두 층이면 z 방향의 결과가 총 40개입니다. 반면 `sample_z=[20.]`은 높이 20 nm에서 한 번 읽는다는 뜻입니다.

- x·y의 원점은 셀 중심입니다. 직사각형 셀에서 개수를 지정하면 각 방향의 셀 양 끝을 포함해 균일하게 배치합니다. 개수가 1이면 음의 방향 끝점이므로, 중심의 한 점을 읽으려면 `[0.]`을 씁니다.
- z의 원점은 맨 아래층의 바닥이며, 위쪽으로 갈수록 증가합니다. 좌표 배열은 `0 ≤ z ≤ sum(thickness)` 범위에서 지정합니다.
- 좌표의 길이 단위는 `wavelength`, `period`, `thickness`와 같습니다. 예제처럼 이 값들을 nm로 넣었다면 좌표도 nm입니다.

x·y 좌표를 배열로 지정하면 모든 x·y 조합을 읽고, 이를 각 z 높이에서 반복합니다. 입력한 좌표의 순서는 그대로 유지됩니다. y에 따라 변하지 않는 순수 1D 장도 요청한 y 위치 수만큼 반환됩니다.

:::{dropdown} 층 경계와 사선 격자에서의 좌표

**층 경계:** 정확히 내부 경계면에 놓인 z 좌표는 위쪽 층의 값으로 읽습니다. `z=0`과 `z=sum(thickness)`에서는 각각 맨 아래층과 맨 위층의 안쪽 값을 읽습니다.

점 개수로 `sample_z`를 지정하면 각 층을 따로 나눕니다. 개수가 2 이상이면 각 층의 양 끝을 포함하므로, 내부 경계면은 아래층 쪽 값과 위층 쪽 값이 한 번씩 들어갑니다. 법선 성분 `Ez`, `Hz`는 재료 경계에서 달라질 수 있습니다. `sample_z=1`은 각 층의 아래쪽 면을 읽습니다.

**사선 격자:** 정수 `sample_x`, `sample_y`는 두 격자 벡터를 따라 점을 배치하며, 배열의 수평 축은 `(v, u)`에 대응합니다. 두 항목을 모두 좌표 배열로 넣으면 직교 좌표 x·y의 모든 조합에서 읽습니다. 사선 격자에서는 한쪽만 개수, 다른 쪽은 좌표 배열로 섞어 지정하지 않습니다.

**다이폴:** 점 다이폴이 놓인 높이에서는 장이 특이하므로, 그 높이를 제외하고 위·아래 위치를 지정합니다. 점 개수 방식이 소스 높이를 포함하는 경우에도 다른 표본 위치를 선택해야 합니다.
:::

(fields-detail-id26)=
### 반환 배열에서 성분 고르기

모든 결과의 층 내부 필드는 `(nz, ny, nx, 6)` 모양입니다. 앞의 세 축으로 위치를 고르고, 마지막 축으로 성분을 고릅니다.

```{figure} ../../figures/learn/learn/13-field-layout.svg
:alt: 공통 필드 배열은 z, y, x, 성분 순서이며 마지막 축에는 Ex Ey Ez Z0Hx Z0Hy Z0Hz가 들어 있다
:figclass: learning-figure

그림 14-2. 결과 객체에서 읽는 필드 배열의 축과 여섯 성분. 자기장은 Z₀가 곱해진 정규화 값입니다.
```

| 인덱스 | `0` | `1` | `2` | `3` | `4` | `5` |
| --- | --- | --- | --- | --- | --- | --- |
| 성분 | E<sub>x</sub> | E<sub>y</sub> | E<sub>z</sub> | Z<sub>0</sub>H<sub>x</sub> | Z<sub>0</sub>H<sub>y</sub> | Z<sub>0</sub>H<sub>z</sub> |

```python
E = cell[..., :3]           # 전기장 세 성분
H_normalized = cell[..., 3:] # 정규화된 자기장 세 성분
Ey = cell[..., 1]           # 모든 위치의 전기장 y 성분
```

TE/TM에서도 여섯 성분의 위치는 같습니다. 순수 TE에서는 `Ex`, `Ez`, `Hy`가 0이고, 순수 TM에서는 `Ey`, `Hx`, `Hz`가 0입니다.

일반 결합 계산에서 함께 구한 순수 s·p 편광 응답은 `incidence="s"` 또는 `"p"`로 하나씩 읽습니다. 기본값 `"incident"`는 지정한 입사 조건의 결과입니다.

```python
# s·p 응답을 함께 계산한 결과에서 사용
s_field = result.reconstruct_field(incidence="s")
p_field = result.reconstruct_field(incidence="p")
```

이 선택은 서로 다른 입사파의 기여를 분리하는 기능이 아닙니다. 순수 TE/TM과 여러 소스를 합친 결과에서는 `incidence="incident"`만 사용합니다.

:::{dropdown} 두께를 자동미분하며 필드를 계산할 때

두께를 바꿀 때마다 해당 두께로 `solve()`하고, 반환된 결과에서 필드를 읽습니다.

```python
def field_for_thickness(d):
    result = solver.solve(thickness=d, answers="field")
    return result.reconstruct_field(
        sample_x=[0.], sample_y=[0.], sample_z=20,
    )
```

일반 입사 결과의 공통 필드 함수는 JAX·PyTorch에서 두께에 대한 자동미분을 지원합니다. `sample_z=20`은 각 층 안의 상대 위치에 점을 배치하므로, 두께가 바뀌면 표본의 실제 높이도 함께 바뀝니다.

`sample_z=[50., 100.]`처럼 좌표를 넣으면 고정된 높이에서의 변화를 계산합니다. 두께를 바꾸는 동안 표본이 같은 층 안에 머무르는 구간에서 미분할 수 있습니다. 경계가 표본을 통과하면 해당 위치의 재료와 장의 법선 성분이 바뀔 수 있으므로, 그 지점에서 매끄러운 미분을 기대할 수는 없습니다. JAX `jit`에서는 좌표가 층 밖으로 나가지 않도록 입력 범위도 유지해야 합니다.

`PlaneWave`·다이폴 등의 소스 계산은 소스 분류와 층 분할에서 구체적인 수치값을 사용합니다. 위 설명은 일반 입사 계산의 자동미분에 해당합니다.
:::

(illumination-fields)=
(fields-detail-id23)=
## 경계면의 입사파·출력파·전체 장 읽기

경계면은 `result.boundary_fields(...)`로 읽습니다. 단일 입사에도, 여러 소스를 합친 결과에도 같은 함수를 사용합니다. 아래는 위·아래에서 s 편광을 넣는 예제이며, 아래 입사파의 상대 위상은 90°입니다.

```python
field_solver = wavebit.rcwa(
    wavelength=600., period=(900., 850.),  # 길이 단위: nm
    top=1., bot=2.25,
    structure=[dict(background=4.+0.6j, shapes=[])],
    thickness=[100.], truncation_order=0,
    kx=0.002, ky=0.,                       # 단위: nm⁻¹
    excitation={
        "top": wavebit.PlaneWave(polarization=(1., 0.)),
        "bottom": wavebit.PlaneWave(
            polarization=(1., 0.),
            amplitude=np.exp(1j * np.deg2rad(90.)),
        ),
    },
)
field_result = field_solver.solve(answers="field")
x = np.linspace(-300., 300., 61)
y = np.array([0.])
```

(fields-detail-id24)=
### 읽을 파 선택하기

위 경계면은 `z=100 nm`, 아래 경계면은 `z=0`입니다. 다음 코드는 위 경계면의 `y=0`인 선에서 필드를 읽습니다.

```python
incoming = field_result.boundary_fields("top", x, y, part="incoming")
outgoing = field_result.boundary_fields("top", x, y, part="outgoing")
boundary = field_result.boundary_fields("top", x, y, part="total")
print(boundary.shape)  # (1, 61, 6)
# 같은 위치에서 boundary = incoming + outgoing
```

| 입력 | 의미 |
| --- | --- |
| `side` | `"top"` 또는 `"bottom"`. 읽을 경계면을 지정합니다. |
| `sample_x`, `sample_y` | 점 개수 또는 좌표 배열. 생략하면 각각 `[0.]`으로 셀 중심을 읽습니다. |
| `part="incoming"` | 선택한 쪽에서 층을 향해 들어오는 파 |
| `part="outgoing"` | 층에서 선택한 쪽으로 나가는 파 |
| `part="total"` — 기본값 | 들어오는 파와 나가는 파를 합친 전체 장 |

반환 모양은 `(ny, nx, 6)`이며, 성분 순서는 층 내부와 같습니다. 여러 소스의 기여는 복소수로 합쳐집니다. 양면 입사에서 위쪽의 `outgoing`에는 위 입사파의 반사 성분과 아래 입사파의 투과 성분이 함께 들어 있습니다.

(fields-detail-id25)=
### 경계면의 바깥쪽과 안쪽

`boundary_fields()`는 경계면의 **반무한 매질 쪽 값**을 반환합니다. 층 안쪽 값은 `reconstruct_field()`로 읽습니다.

```python
inside = field_result.reconstruct_field(x, y, sample_z=[0., 50., 100.])
print(inside.shape)  # (3, 1, 61, 6)
middle = inside[1]   # z=50 nm인 선의 필드
```

| 호출 | 읽는 쪽 |
| --- | --- |
| `field_result.boundary_fields("top", x, y)` | 위 경계면의 `top` 반무한 매질 쪽 |
| `field_result.reconstruct_field(x, y, sample_z=[100.])` | 위 경계면의 100 nm 층 안쪽 |

자유 표면 전하·전류가 없는 경계에서 접선 성분 `Ex`, `Ey`, `Hx`, `Hy`는 양쪽에서 이어집니다. 법선 성분 `Ez`, `Hz`는 재료에 따라 달라질 수 있습니다. 층 내부 함수는 모든 소스가 만드는 전체 장을 반환하므로 `part`를 지정하지 않습니다.

위치별 필드는 `power_region`으로 면적 평균하거나 입사 전력으로 나눈 값이 아닙니다. 소스의 진폭과 위상을 반영한 각 위치의 복소 전기장·자기장입니다.

(fields-detail-id2)=
(complex-field-time)=
## 복소수에서 진폭·위상·순간 전기장 읽기

위 `inside` 배열에서 중간 높이의 전기장 y 성분을 꺼내 보겠습니다.

```python
Ey = inside[1, 0, :, 1]              # z=50 nm, y=0인 선의 Ey
Ey_amplitude = np.abs(Ey)           # 진폭의 크기
Ey_phase_deg = np.angle(Ey, deg=True) # 위상, 단위: 도
```

한 위치에서 반환된 값이 {math}`E_y=Ae^{i\varphi}`라면, {math}`A=|E_y|`는 진폭의 크기이고 {math}`\varphi=\arg(E_y)`는 위상입니다. 두 정보를 한 복소수로 표현한 {math}`E_y`를 **복소 진폭**, 또는 페이저(phasor)라고 부릅니다.

실제 시간에 따라 진동하는 전기장 성분 {math}`e_y(t)`는 다음과 같습니다. {math}`\omega`는 빛의 각주파수, {math}`t`는 시간이며, 여기서는 한 위치를 고정해 놓고 봅니다.

```{math}
e_y(t)
=\operatorname{Re}\!\left[E_y e^{-i\omega t}\right]
=A\cos(\varphi-\omega t).
```

따라서 `Ey.real`은 `t=0`에서의 순간 전기장이고, `np.abs(Ey)`는 진동의 크기입니다. 시간 위상 `omega_t = ωt`를 라디안으로 주면 다른 순간의 값도 구할 수 있습니다.

```python
omega_t = np.pi / 2
Ey_instant = np.real(Ey * np.exp(-1j * omega_t))
```

예를 들어 {math}`E_y=e^{i\pi/3}`이면 진폭의 크기는 1, 위상은 60°입니다. 순간 전기장은 {math}`\cos(\pi/3-\omega t)`이므로 `t=0`에서 0.5이고, 시간이 지나면서 −1과 1 사이를 진동합니다.

```{raw} html
<section class="exercise" aria-labelledby="phasor-title"><h3 id="phasor-title">직접 바꿔보기 · 복소수 하나에서 실제 진동 읽기</h3><p class="concept">설명용 단일 성분 E=exp(iπ/3), 크기 1. 위에서 계산한 박막장의 값과는 다른 개념 예제입니다.</p><div class="controls"><label for="phasor-time">시간 위상 ωt <output id="phasor-time-out" for="phasor-time">0°</output><input id="phasor-time" type="range" min="0" max="360" value="0"></label></div><svg viewBox="0 0 400 260" role="img" aria-label="복소평면에서 시간에 따라 회전하는 phasor와 실수축 투영"><circle cx="155" cy="125" r="94" fill="none" stroke="#cad7df"/><line x1="30" y1="125" x2="293" y2="125" stroke="#abbec5"/><line x1="155" y1="15" x2="155" y2="240" stroke="#abbec5"/><text x="299" y="132">실수</text><text x="163" y="21">허수</text><line id="phasor-vector" x1="155" y1="125" x2="202" y2="44" stroke="#285ab0" stroke-width="3"/><line id="phasor-drop" x1="202" y1="44" x2="202" y2="125" stroke="#b56626" stroke-dasharray="4 4"/><circle id="phasor-point" cx="202" cy="125" r="6" fill="#b56626"/><text x="42" y="258">주황 점: 실제 순간 값인 실수부</text></svg><div class="readout" id="phasor-readout" aria-live="polite">ωt=0일 때 실수부는 0.5, |E|²=1입니다.</div></section>
```

전기장 세 성분의 크기 제곱을 합한 {math}`|\mathbf E|^2`는 다음처럼 읽습니다.

```python
E_squared = np.sum(np.abs(inside[..., :3])**2, axis=-1)
print(E_squared.shape)  # (3, 1, 61)
```

이 값은 각 위치에서 전기장이 얼마나 큰지 비교할 때 사용합니다. 전력이 흐르는 방향과 크기는 자기장도 함께 사용한 포인팅 벡터로 구합니다.

(magnetic-field-units)=
## 자기장의 정규화와 물리 단위

Wavebit은 Maxwell 방정식을 계산할 때부터 자기장 {math}`\mathbf H` 대신 **정규화된 자기장** {math}`\widetilde{\mathbf H}=Z_0\mathbf H`를 변수로 사용합니다. {math}`Z_0\approx376.73\,\Omega`는 진공 임피던스입니다. 필드 배열에도 이 정규화된 값이 들어 있으므로, 자기장 자체가 필요하면 {math}`Z_0`로 나눕니다.

```{math}
\mathbf H=\frac{\widetilde{\mathbf H}}{Z_0}.
```

위의 6성분 배열에서는 마지막 세 성분이 정규화된 자기장입니다.

```python
E = inside[..., :3]
H_normalized = inside[..., 3:]
Z0 = 376.730313668
H = H_normalized / Z0
```

:::{dropdown} 순수 TM 입력의 진폭 기준

`tetm="tm"`의 순수 1D 계산은 입사파의 정규화된 자기장을 기준으로 진폭을 지정합니다. 반면 `polarization=(a_s, a_p)`와 `PlaneWave`는 전기장 기준입니다. 공통 필드 함수는 이 입력 기준을 그대로 유지하며, 반환한 장을 입사 전력으로 나누거나 자동으로 같은 크기에 맞추지 않습니다.
:::

입력 전기장 진폭 `1`을 `1 V/m`로 해석했다면 `E`의 단위는 V/m, 변환한 `H`의 단위는 A/m입니다. 예를 들어 자기장 성분이 `1`로 반환되었다면, 변환한 값은 약 `0.00265 A/m`입니다.

입력 진폭을 상대값으로만 정했다면 실제 단위를 부여할 기준도 정해야 합니다. 예를 들어 계산의 전기장 `1`을 `10 V/m`에 대응시키려면, 위 `E`와 `H`에 모두 10을 곱합니다. {math}`Z_0`로 나누는 것은 일정한 크기 변환이므로, 자기장의 공간 분포 모양과 위상은 바뀌지 않습니다.

### 전력 밀도를 구할 때

시간평균 포인팅 벡터는 전기장과 자기장의 복소 진폭으로 구합니다. 별표 {math}`*`는 복소켤레입니다.

```{math}
\langle\mathbf S\rangle
=\frac12\operatorname{Re}(\mathbf E\times\mathbf H^*)
=\frac{1}{2Z_0}\operatorname{Re}
  (\mathbf E\times\widetilde{\mathbf H}^{\,*}).
```

위에서 변환한 `H`를 사용하면 다음과 같습니다.

```python
S = 0.5 * np.real(np.cross(E, np.conj(H)))
Sz = S[..., 2]
```

`E`를 V/m, `H`를 A/m로 맞추었다면 `S`는 W/m² 단위입니다. `Sz > 0`이면 위쪽인 +z로, `Sz < 0`이면 아래쪽인 −z로 전력이 흐릅니다. 이 값에서 반사율·투과율과 순유입 전력을 구하는 과정은 [13장](power-and-efficiency.md)에서 설명합니다.

반사율·투과율처럼 전력의 비율을 구할 때는 분자와 분모의 `1/Z0`가 상쇄됩니다. 따라서 반환된 `R`, `T`, `output_top`, `output_bottom`에 자기장 단위 변환을 다시 적용하지 않습니다.

:::{dropdown} PlaneWave 결과의 result.power를 W 단위로 환산하기

`result.power.incoming_top`, `incoming_bottom`, `outgoing_top`, `outgoing_bottom`은 `power_region`에서 평균한 각 입사·출력파의 법선 전력 흐름입니다. 각 파가 들어오거나 나가는 방향을 양의 방향으로 정하며, 공통 인자 `1/Z0`를 생략한 계산 단위로 반환됩니다.

`result.power`에서 선택한 값을 {math}`P_{\mathrm{calc}}`라 쓰겠습니다. 계산의 전기장 진폭 `1`을 실제 {math}`E_{\mathrm{scale}}\,\mathrm{V/m}`에 대응시키고, 물리적인 관측 면적을 {math}`A_{\mathrm{obs}}\,\mathrm{m}^2`라 하면 총 전력은 다음과 같습니다.

```{math}
P\;[\mathrm W]
=P_{\mathrm{calc}}\,\frac{E_{\mathrm{scale}}^2}{Z_0}\,A_{\mathrm{obs}}.
```

예를 들어 관측 영역을 직사각형으로 지정했다면, 그 직사각형의 실제 면적을 m²로 바꾸어 사용합니다. 무한 영역의 평균값만으로는 유한한 총 전력이 정해지지 않으므로, 총 전력을 말하려면 면적도 정해야 합니다.
:::

:::{dropdown} 다른 RCWA 툴의 자기장 반환 규약

| 툴 | 자기장 값의 의미와 근거 |
| --- | --- |
| <span style="white-space:nowrap">Meent</span> | 필드 코드에서 `Hx`, `Hy`, `Hz`로 부르는 값도 정규화된 자기장에 해당합니다. [필드 복원 코드](https://github.com/kc-ml2/meent/blob/main/meent/on_numpy/emsolver/field_distribution.py)와 [행렬 구성 코드](https://github.com/kc-ml2/meent/blob/main/meent/on_numpy/emsolver/transfer_method.py)의 식에서 이 스케일을 확인할 수 있습니다. |
| <span style="white-space:nowrap">S4</span> | 반환하는 `H`는 Z₀H<sub>SI</sub>에 해당합니다. [공식 논문 §3.1](https://web.stanford.edu/group/fan/publication/Liu_ComputerPhysicsCommunications_183_2233_2012.pdf)에 이 단위 변환이 명시되어 있습니다. |
| <span style="white-space:nowrap">FMMAX</span> | [공식 규약](https://facebookresearch.github.io/fmmax/#fmm-conventions)에서 진공의 ε₀·μ₀·c를 모두 1로 둡니다. 이 단위계의 반환 자기장은 같은 정규화 기준으로 해석합니다. |
| <span style="white-space:nowrap">RETICOLO</span> | [공식 설명서 4쪽](https://arxiv.org/pdf/2101.00901#page=4)에서 ε₀·μ₀·c를 1로 둔 Maxwell 방정식을 사용합니다. `res3()`의 `H` 성분도 이 단위계를 따릅니다. |

따라서 반환값 이름이 `H`라고 되어 있어도 바로 A/m 단위라고 해석하지 않습니다. 툴 사이의 복소 필드를 비교할 때는 자기장 정규화뿐 아니라 입사 진폭의 기준, 좌표축과 위상 규약도 맞춥니다.
:::
