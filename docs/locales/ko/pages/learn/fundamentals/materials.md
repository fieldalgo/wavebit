# 07. 재료: 유전율, 흡수, 파장에 따른 변화

:::{note}
다운로드 가능한 교육 스크립트는 현재 물리 Problem/Job API를 사용합니다. 일부 설명 코드와 그림에는 이전 API 표기가 남아 있으므로 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 함께 참고하세요. 과거 측정 파일과 원본 출처는 보존하며, 이전한 스크립트의 검증과 구분합니다.
:::

**계산에 넣는 재료값은 상대 유전율 εᵣ입니다.** 아래첨자 r은 상대(relative)를 뜻하며, 진공 유전율 ε<sub>0</sub>에 대한 비 εᵣ = ε/ε<sub>0</sub>를 나타냅니다.

상수로 볼 재료는 `12.25`처럼 숫자로 바로 넣고, 파장에 따라 값이 달라지는 재료는 분산 정보를 가진 객체를 한 번 지정합니다. 솔버는 현재 파장에 해당하는 값을 평가해 계산합니다. 굴절률을 별도의 계산 입력으로 관리할 필요는 없습니다.

```{figure} ../../figures/learn/learn/07-material.svg
:alt: 재료를 한 번 지정하면 파장마다 상대 유전율을 평가해 RCWA 계산에 전달한다
:figclass: learning-figure

그림 7-1. 재료를 한 번 지정하면 파장마다 상대 유전율을 평가해 RCWA 계산에 전달한다
```

(material-inputs)=
## 재료 입력 방법을 먼저 고르기

재료를 고를 때는 **방향에 따라 달라지는가(등방성·이방성)**와 **파장에 따라 달라지는가(상수·분산)**를 따로 봅니다. 이방성 재료도 파장에 따라 변할 수 있습니다. 아래 표는 물리적 분류와 현재 Wavebit의 입력 경로를 연결합니다.

| 가진 정보 | 입력 예 | 계산에 쓰이는 값 |
| --- | --- | --- |
| 상수 등방성 유전율 | `12.25` | 모든 방향에 같은 εᵣ |
| 흡수를 포함한 상수 유전율 | `4. + 0.1j` | 복소 εᵣ |
| 상수 ε와 μ | `wavebit.Material(epsilon=4., mu=1.2)` | εᵣ와 μᵣ를 함께 지정 |
| 번들 재료표 | `wavebit.materials.load('Si')` | 계산 파장에서 자동 평가한 εᵣ |
| 직접 가진 파장별 ε·μ | `wavebit.TabulatedMaterial(...)` | 사용자 표에서 보간한 스칼라 또는 텐서 εᵣ·μᵣ |
| 외부 n+ik 표 | `wavebit.TabulatedMaterial.from_refractive_index(..., mu=...)` | 굴절률과 선택적 투자율로 유전율 계산; mu 생략 시 비자성 |
| 이방성 유전율 텐서 | `Material(epsilon=[2.25, 4., 2.25])` 또는 3×3 배열 | 방향별 응답과 성분 간 결합; 아래 별도 예제 참고 |

상수와 사용자 표는 등방성·이방성 모두 표현할 수 있습니다. 번들 n·k 자료는 비자성 등방성 재료로 평가합니다. `from_refractive_index`는 등방성 재료용이며, `mu`를 주면 자성 재료도 변환합니다. 래스터에는 평가된 숫자 배열을 넣습니다. 도형의 `material`과 층의 `background`에는 스칼라 또는 지원되는 텐서 재료·분산 객체를 할당합니다. `top`·`bot` 반무한 매질은 스칼라 재료만 받으며, 실수와 복소수 모두 사용할 수 있습니다. 입력 방법은 아래 {ref}`반무한 매질 설명 <material-half-spaces>`을 참고합니다.

## 실리콘을 한 번 지정하고 파장만 바꾸기

```python
silicon = wavebit.materials.load('Si')
# 파장을 여기서 고정하지 않는다. silicon에는 실리콘의 분산 표가 남는다.
```

이 객체를 도형의 `material`, 층의 `background`, 또는 위·아래 매질인 `top`, `bot`에 넣습니다. 아래 예제에서는 **같은 원에 같은 silicon 객체를 붙여 두고, 파장만 바꿉니다.** 파장마다 도형을 다시 만들거나 유전율을 직접 교체하지 않습니다.

실행 파일: {download}`09_material_sweep.py <../../../examples/rcwa/learning/learn/09_material_sweep.py>`

```{literalinclude} ../../../examples/rcwa/learning/learn/09_material_sweep.py
:language: python
:linenos:
```

| 파장 (nm) | 원의 상대 유전율 εᵣ | 전체 R | 전체 T |
| --- | --- | --- | --- |
| 800.0 | 13.634191 + 0.047966j | 0.057296958 | 0.940747954 |
| 900.0 | 13.240457 + 0.054569j | 0.050530329 | 0.945208958 |
| 1000.0 | 12.903107 + 0.029383j | 0.015798325 | 0.983445526 |

표에서 파장뿐 아니라 원 안의 유전율도 함께 바뀌는 것을 확인할 수 있습니다. 이 예제의 차수 `(1, 1)`과 NVF 표본 수 64는 사용 흐름을 짧게 보여주기 위한 설정이며, 광학 수렴을 입증하는 설정이 아닙니다.

## 상수 재료와 분산 재료는 무엇이 다른가

**`TabulatedMaterial`과 `Material`은 모두 재료를 나타내는 객체입니다.** `TabulatedMaterial`은 파장별 데이터를 가지고 있고, `Material`은 유전율과 투자율 값을 담고 있습니다. 따라서 계산 파장을 바꾸면 `TabulatedMaterial`은 그 파장에 해당하는 값을 구하고, `Material`은 저장된 값을 그대로 사용합니다.

| 지정 방법 | 값의 종류 | 파장을 바꿨을 때 |
| --- | --- | --- |
| 숫자 `12.25` | 숫자 | 유전율 12.25를 유지; 객체 생성 불필요 |
| `wavebit.Material(epsilon=4., mu=1.2)` | `Material` | 유전율 4와 투자율 1.2를 함께 유지 |
| `wavebit.materials.load('Si')` | `TabulatedMaterial` | 실리콘 표에서 새 파장에 해당하는 값을 구함 |
| `silicon.at(900.)` | `MaterialResponse` | 900 nm에서 구한 유전율과 투자율을 유지 |

`materials.load('Si')`는 실리콘의 **파장별 재료 데이터**를 불러옵니다. 이 재료를 도형이나 층에 할당하면 계산 파장에 맞는 재료값을 자동으로 사용합니다.

`silicon.at(900.)`은 **900 nm에서의 유전율과 투자율을 구해 `MaterialResponse` 객체에 담으며**, 이 객체는 `Material`처럼 할당합니다. 이 결과를 도형이나 층에 할당하면, 계산 파장을 바꿔도 900 nm에서 구한 유전율과 투자율을 그대로 사용합니다.

### 두 객체 모두 같은 자리에 재료로 할당합니다

도형의 `material`, 층의 `background`, 위·아래 매질의 `top`, `bot`에는 두 종류의 객체를 모두 넣을 수 있습니다. 아래 예제는 같은 실리콘을 층의 재료로 지정하되, 파장에 따라 재료값을 바꿀지 고정할지를 선택합니다.

```python
silicon = wavebit.materials.load('Si')       # TabulatedMaterial: 파장별 데이터
silicon_900 = silicon.at(900.)              # MaterialResponse: 900 nm의 ε·μ 값

# 계산 파장에 맞춰 실리콘의 재료값을 구하는 층
layer = dict(background=silicon, shapes=[])

# 계산 파장이 달라져도 900 nm의 재료값을 유지하는 층
fixed_layer = dict(background=silicon_900, shapes=[])
```

계산 파장이 1,000 nm라면 `layer`는 1,000 nm의 실리콘 값을, `fixed_layer`는 900 nm에서 구한 값을 사용합니다. **파장 스윕에서 실리콘의 파장별 특성을 반영하려면 `silicon`을 그대로 할당하면 됩니다.** 솔버가 파장에 맞는 값을 구하므로 `.at()`을 직접 호출할 필요가 없습니다.

`.at(900.)`을 호출해도 원래 `silicon`의 파장별 데이터는 그대로 남습니다. `silicon_900`은 별도로 만든 고정값 재료입니다.

## 어떤 재료가 있고, 길이 단위는 어떻게 고를까

```python
wavebit.materials.available()
# ('filmetrics/Al2O3', 'filmetrics/Si', 'filmetrics/Si3N4', 'filmetrics/SiO2')
silicon = wavebit.materials.load('filmetrics/Si')  # 출처까지 지정
print(silicon.wavelength_range)                  # (190.7, 2500.0), 기본 nm
silicon_um = wavebit.materials.load('Si', wavelength_unit='um')
value = silicon_um.at(0.9)                       # 0.9 μm = 900 nm
```

`Si`처럼 짧게 부를 수 있고 대소문자는 구별하지 않습니다. 출처가 여러 개라 이름이 모호하면 하나를 임의로 선택하지 않습니다. `filmetrics/Si`처럼 출처를 포함해서 고릅니다. **같은 화학식이라도 제작 조건에 따라 광학 특성이 다를 수 있으므로 자료 선택은 물리 모델의 일부입니다.** 필요한 파일 하나만 읽고 불변 객체를 재사용하며, 전체 재료표를 사용자가 먼저 읽을 필요가 없습니다.

기본 길이 단위는 `nm`입니다. `wavelength_unit='um'`은 표의 파장 축을 μm로 바꿉니다. 그러면 계산 파장뿐 아니라 주기·도형 크기·층 두께도 모두 μm로 입력합니다. `nm`, `um`, `mm`, `m`을 지원하며, 이미 만든 도형의 크기를 몰래 바꾸지는 않습니다. 원래 포함돼 있던 `p_Si.mat`은 단위·원출처 정보가 파일에 없어 이 목록에 넣지 않았고, 2026-10-14에 패키지에서 제거했습니다.

`solver.structure`, `solver.top`, `solver.bot`에서 읽는 재료는 현재 파장에서 평가한 값입니다. 새 재료나 도형을 넣을 때는 `structure`나 `top`, `bot`을 다시 할당합니다. 읽어 온 평가용 딕셔너리의 일부만 직접 고치면 원래 분산 정의를 바꾼 것이 아니므로 이런 편집은 피합니다.

## 내가 가진 유전율 표를 쓰려면

```python
material = wavebit.TabulatedMaterial(
    wavelengths=[800., 900., 1000.],
    epsilon=[4.+0.1j, 4.2+0.12j, 4.5+0.15j],
)
value = material.at(850.)
# value.epsilon = 4.1 + 0.11j
```

양수이고 증가하는 파장 표와, 파장마다 하나의 복소 유전율을 지정합니다. 최소 두 표본이 필요하며 표본 사이의 복소 유전율은 선형 보간합니다. 필요한 경우 같은 길이의 `mu` 표도 지정할 수 있습니다. 표는 복사해서 보관하므로 원래 배열을 나중에 수정해도 재료 정의가 몰래 바뀌지 않습니다. 파장마다 스칼라, 대각 성분 세 개, 또는 3×3 텐서를 넣을 수 있습니다. 한 물리량의 모든 표본은 같은 모양이어야 하며 텐서는 성분별로 보간합니다.

## 표본 파장과 계산 파장은 다르다

`wavelengths=[800., 900., 1000.]`은 **재료 데이터가 주어진 표본 지점**입니다. 실제 계산은 이 범위 안의 임의의 파장을 사용할 수 있습니다.

```python
value = material.at(850.)              # 800과 900 사이의 ε를 선형 보간
result = solver.solve(wavelength=873.2) # 할당된 분산 재료도 이 파장에서 평가
```

ε와 μ 표를 함께 넣으면 두 값 모두 보간합니다. 아래 자성 예제에서는 850 nm에서 ε=4.1, μ=1.15입니다. 표본 파장을 하나씩 추가하지 않아도 됩니다. 다만 표의 범위가 800–1000 nm라면 1100 nm에서는 오류가 나므로 더 넓은 파장을 포함한 자료가 필요합니다. `materials.load('Si')`의 기본 자료는 190.7–2500 nm를 포함합니다. 범위 안에서 보간할 수 있다는 것과 표가 실제 재료를 충분히 정확하게 나타낸다는 것은 별개이므로, 급격한 분산 변화가 있는 구간은 표본 간격과 원자료도 확인합니다.

## 투자율 μ 데이터도 있으면 함께 넣는다

```python
magnetic = wavebit.TabulatedMaterial(
    wavelengths=[800., 900., 1000.],
    epsilon=[4., 4.2, 4.5],
    mu=[1.1, 1.2, 1.3],
)
value = magnetic.at(850.)
# value.epsilon = 4.1, value.mu = 1.15
```

이 객체를 재료로 할당하면 파장마다 ε와 μ가 함께 바뀝니다. **현재 번들 표는 n·k만 담고 있어 μ를 읽어 오는 자료가 아닙니다.** 비자성 재료로 가정해 μᵣ=1을 사용합니다. 일반 등방성 매질에서 `n²=εᵣ μᵣ`이므로 n·k만으로 ε와 μ를 각각 알아낼 수 없습니다. 자성 재료는 측정되거나 모델로 정해진 두 값을 위처럼 함께 입력해야 합니다.

## 굴절률은 외부 자료를 가져올 때만 변환하면 된다

외부 광학 자료는 흔히 `n, k` 표로 제공됩니다. `materials.load`로 선택한 재료는 이 표의 n과 k를 보간한 뒤, 비자성 조건 μᵣ=1에서 `(n+ik)²`로 유전율을 계산합니다. 이 변환은 내부에서 처리됩니다. 직접 가진 유전율 표를 쓰면 그 변환이 필요 없습니다. 두 방식은 보간하는 양도 다르므로, 굴절률을 먼저 보간해 제곱하는 것과 유전율을 바로 보간하는 것이 표 사이에서 항상 같지는 않습니다.

직접 가진 n+ik 자료는 `TabulatedMaterial.from_refractive_index(wavelengths, index)`로 명시적으로 가져옵니다. 유전율로부터 굴절률 그래프가 필요하면 비자성 조건에서 `wavebit.refractive_index_from_eps(silicon.at(wavelengths).epsilon)`을 쓸 수 있습니다. `.at`에는 파장 배열도 넣을 수 있으며, 출력 유전율도 같은 모양의 배열입니다. 별도의 굴절률 검색 함수나 재료 이름의 숨은 접미사는 사용하지 않습니다. Wavebit의 시간 규약은 `exp(−iωt)`이며, 보통의 수동 매질은 스칼라 유전율의 양의 허수부로 손실을 표현합니다.


## 자성 재료의 굴절률을 유전율로 바꾸기

등방성 재료의 복소 굴절률을 `n_complex`라고 하면 `n_complex² = εᵣ μᵣ`입니다. 따라서 **자성 재료는 굴절률을 제곱한 뒤 상대 투자율로 나눕니다.** 굴절률과 투자율은 같은 파장의 값이어야 합니다.

```text
εᵣ = n_complex² / μᵣ
```

`eps_from_refractive_index`에 `mu`를 함께 주면 이 변환을 수행합니다. 예를 들어 굴절률이 2이고 상대 투자율이 1.25이면 상대 유전율은 3.2입니다. 변환에 사용한 투자율을 `Material`에도 함께 넣습니다.

```python
n_complex = 2.0
mu_r = 1.25
magnetic = wavebit.Material(
    epsilon=wavebit.eps_from_refractive_index(n_complex, mu=mu_r),
    mu=mu_r,
)
# magnetic.epsilon = 3.2, magnetic.mu = 1.25
```

파장별 굴절률 표에도 투자율을 함께 지정할 수 있습니다.

```python
magnetic = wavebit.TabulatedMaterial.from_refractive_index(
    wavelengths=[800., 1000.],
    index=[2., 3.],
    mu=[1., 2.],
)
value = magnetic.at(900.)
# 굴절률 2.5, 상대 투자율 1.5로 보간
# value.epsilon = 2.5**2 / 1.5, value.mu = 1.5
```

`index`에는 `2.+0.1j`처럼 흡수를 포함한 복소 굴절률도 넣을 수 있습니다. `mu`는 위처럼 **같은 표본 파장에서의 값 배열**이나, `mu=1.25`처럼 **표 범위에서 일정한 상수**로 지정합니다. `mu`를 생략하면 기존처럼 비자성 조건 `μᵣ=1`을 사용합니다.

이 모델은 **굴절률과 투자율을 각각 보간한 다음 εᵣ=n_complex²/μᵣ를 계산**합니다. 먼저 표본마다 유전율로 변환하고 그 유전율을 보간하는 방법과는 중간 파장의 값이 다를 수 있습니다.

굴절률만으로는 유전율과 투자율을 따로 알아낼 수 없습니다. 투자율은 별도의 측정값이나 모델이 필요합니다. 변환에 쓰는 투자율은 유한하고 0이 아니어야 합니다. 보간 도중 투자율이 0이 되면 일반 실행에서는 오류가 나고, JAX JIT에서는 해당 유전율이 NaN이 됩니다. **이 변환은 등방성 재료용입니다.** 이방성 재료는 `epsilon`과 `mu` 텐서를 직접 지정합니다.

## 파장 스윕과 미분에서 기억할 것

표와 구조의 모든 길이는 같은 단위를 써야 합니다. 모든 분산 객체는 표 범위 밖의 파장을 일반 실행에서 오류로 거절합니다. JAX JIT 안에서 범위를 벗어난 원소는 NaN이 됩니다. 끝값으로 몰래 고정하거나 표 밖으로 외삽하지 않으므로, 파장 스윕 범위는 `wavelength_range` 안에 둡니다.

도형·층 배경·위아래 매질에 `TabulatedMaterial`을 할당했다면, `solver.solve(wavelength=...)`로 파장을 바꿀 때 그 파장의 재료값을 자동으로 구합니다.

**숫자로 만든 래스터 배열은 이미 평가된 유전율 분포입니다.** 픽셀별 재료 이름이나 분산 정의를 보관하지 않으므로, 래스터 스윕은 `.at(wavelength)`로 재료를 평가하고 배열을 갱신해야 합니다. `solve_conv`에 직접 넣는 행렬도 사용자가 해당 파장에 맞게 다시 만들어야 합니다.

JAX·PyTorch에서는 파장 미분에 재료값의 변화도 포함됩니다. 이것은 표의 구간별 선형 보간에 대한 미분이며 실제 물질의 분산 모델을 피팅했다는 뜻은 아닙니다. 표본 지점의 미분은 오른쪽 구간을, 마지막 지점은 왼쪽 구간을 사용합니다. 표본 값 자체는 고정 자료이며 미분할 설계 변수로 취급하지 않습니다. RCWA의 분산 표가 그대로 FDTD의 시간 영역 재료 모델이 되는 것도 아닙니다.

(material-half-spaces)=
## top과 bot: 구조 밖의 반무한 매질

두께를 지정해 쌓은 층들을 기준으로, `top`은 맨 위층 위의 공간을 채우는 매질이고 `bot`은 맨 아래층 아래의 공간을 채우는 매질입니다. 예를 들어 유리 기판 위의 박막을 공기 중에서 계산한다면 `top`은 공기, `bot`은 유리로 지정할 수 있습니다.

`top`은 위쪽으로, `bot`은 아래쪽으로 무한히 이어진다고 가정합니다. 이런 영역을 **반무한 매질**이라고 하며, 두께(`thickness`)를 지정하지 않습니다.

Wavebit의 `top`·`bot`에는 등방성 재료를 지정합니다. 스칼라 유전율이나 스칼라 ε·μ를 가진 `Material`, 분산 재료 객체를 넣습니다. 이방성 반무한 매질은 지원하지 않습니다. 이방성 재료는 두께가 있는 층의 `background`나 도형의 `material`에 할당할 수 있습니다.

```python
top = 1.                                           # 비자성 공기
bot = 4. + 0.3j                                    # 흡수성 비자성 매질
magnetic = wavebit.Material(epsilon=4.+0.3j, mu=1.2) # ε와 μ를 함께 지정
```

어느 쪽에서 비추는지는 `incidence_side`가 정합니다. 기본값 `"top"`이면 `top`이 입사 매질이고, `"bottom"`이면 `bot`이 입사 매질입니다. **실수 `theta`, `phi`는 선택한 입사 매질 안에서의 위상 진행 방향**을 지정합니다. `theta`는 구조 안쪽 법선에서 잰 각도, `phi`는 고정된 xy 좌표계에서의 방위각이며 단위는 라디안입니다.

방향을 직접 파수 `kx`, `ky`로 지정하는 방법은 {ref}`8장의 두 입력 방식 <incident-direction-inputs>`에서 설명합니다. 이때도 `top`·`bot`의 ε·μ는 필요합니다. 손실이 있는 입사·투과 매질의 파수 계산과 예제는 {ref}`8장 <absorbing-incidence>`에서 설명합니다. 반사율·투과율의 기준면과 유한 층의 흡수는 {ref}`13장 <half-space-power>`을 참고합니다.

(material-anisotropy)=
## 등방성과 이방성: 방향별로 다른 재료 응답

재료의 한 지점에서 전기장의 방향을 바꿔 본다고 생각해 봅시다. **어느 방향에서나 같은 유전율로 설명되는 재료가 등방성 재료**입니다. **방향을 구분해서 유전율을 나타내야 하는 재료가 이방성 재료**입니다. 여기서 방향은 전기장과 재료의 관계를 설명하는 기준입니다.

### 숫자 하나와 세 개의 값

등방성 재료는 상대 유전율 한 값으로 지정합니다. 이방성 재료의 기준 방향이 x·y·z축과 나란하다면, 각 방향의 값을 세 개로 지정할 수 있습니다.

```python
from wavebit.compat import legacy as wavebit

isotropic = 2.25
crystal = wavebit.Material(epsilon=[2.25, 4., 2.25])
```

| 재료 | x 방향 값 | y 방향 값 | z 방향 값 |
| --- | --- | --- | --- |
| `isotropic` | 2.25 | 2.25 | 2.25 |
| `crystal` | 2.25 | 4.0 | 2.25 |

`crystal`에서는 y 방향의 전기장에 대한 응답이 x·z 방향과 다릅니다. 세 값을 넣는 입력은 **대각 성분을 x, y, z 순서로 지정하는 방법**이며, 아래에서 설명할 나머지 행렬 성분은 0으로 둡니다.

### 유전율을 3×3 행렬로 쓰는 이유

전기장 **E**와 전기 변위 **D**의 관계를 다음과 같이 씁니다. 전기 변위는 전기장에 재료의 분극 반응을 포함해 나타내는 양이고, ε<sub>0</sub>는 진공의 유전율입니다.

```{raw} html
<div class="power-equations">
<p><b>D</b> = ε<sub>0</sub> ε<sub>r</sub> <b>E</b></p>
</div>
```

등방성 재료에서는 εᵣ가 숫자 하나입니다. 일반적인 이방성 재료에서는 전기장의 x·y·z 성분이 전기 변위의 각 성분에 어떻게 기여하는지 구분해야 하므로, εᵣ를 3×3 행렬로 표현합니다. 이 재료량을 **유전율 텐서**라고 부릅니다. 아래 ε<sub>ij</sub>는 상대 유전율 텐서의 각 성분이며, 두 글자 ij는 아래첨자입니다.

```{raw} html
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" style="font-size:1.2rem" aria-label="상대 유전율의 3행 3열 텐서">
<mrow><msub><mi>ε</mi><mi>r</mi></msub><mo>=</mo><mo>[</mo><mtable>
<mtr><mtd><msub><mi>ε</mi><mrow><mi>x</mi><mi>x</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>x</mi><mi>y</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>x</mi><mi>z</mi></mrow></msub></mtd></mtr>
<mtr><mtd><msub><mi>ε</mi><mrow><mi>y</mi><mi>x</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>y</mi><mi>y</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>y</mi><mi>z</mi></mrow></msub></mtd></mtr>
<mtr><mtd><msub><mi>ε</mi><mrow><mi>z</mi><mi>x</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>z</mi><mi>y</mi></mrow></msub></mtd><mtd><msub><mi>ε</mi><mrow><mi>z</mi><mi>z</mi></mrow></msub></mtd></mtr>
</mtable><mo>]</mo></mrow>
</math>
```

**행은 D의 방향, 열은 E의 방향**입니다. 첫째 행을 풀어 쓰면 다음과 같습니다.

```{raw} html
<div class="power-equations">
<p>D<sub>x</sub> = ε<sub>0</sub> (ε<sub>xx</sub>E<sub>x</sub> + ε<sub>xy</sub>E<sub>y</sub> + ε<sub>xz</sub>E<sub>z</sub>)</p>
</div>
```

| 성분 | 뜻 |
| --- | --- |
| ε<sub>xx</sub> | x 방향 전기장 E<sub>x</sub>가 x 방향 전기 변위 D<sub>x</sub>에 기여 |
| ε<sub>xy</sub> | y 방향 전기장 E<sub>y</sub>가 x 방향 전기 변위 D<sub>x</sub>에 기여 |
| ε<sub>xz</sub> | z 방향 전기장 E<sub>z</sub>가 x 방향 전기 변위 D<sub>x</sub>에 기여 |

행과 열이 같은 ε<sub>xx</sub>·ε<sub>yy</sub>·ε<sub>zz</sub>를 **대각 성분**, 나머지를 **비대각 성분**이라고 합니다. 예를 들어 ε<sub>xy</sub>가 0이 아니면, y 방향 전기장만 있어도 x 방향 전기 변위가 생길 수 있습니다.

전체 행렬을 알고 있다면 행·열을 x, y, z 순서로 입력합니다. 모든 성분은 Wavebit의 x·y·z축을 기준으로 표현합니다.

```python
material = wavebit.Material(epsilon=[
    [3.0, 0.2, 0.0],   # D의 x 성분에 연결되는 행
    [0.2, 4.0, 0.0],   # D의 y 성분에 연결되는 행
    [0.0, 0.0, 2.5],   # D의 z 성분에 연결되는 행
])
```

이 예제는 ε<sub>xy</sub> = ε<sub>yx</sub> = 0.2인 재료입니다. Wavebit는 입력 행렬을 그대로 사용하며, 서로 다른 두 성분을 임의로 같게 만들지 않습니다. 성분에는 복소수도 넣을 수 있습니다.

상대 투자율 μᵣ도 같은 형식으로 지정합니다. `mu`를 생략하면 비자성 조건인 μᵣ = 1을 사용합니다.

```python
magnetic = wavebit.Material(
    epsilon=[2., 3., 4.],
    mu=[1.1, 1.2, 1.3],
)
```

(material-rotation)=
### 재료를 회전하면 행렬이 어떻게 바뀌는가

먼저 **공간에 고정된 계산 좌표축**(x·y·z)과 **재료와 함께 회전하는 주축**(x′·y′·z′)을 구분합니다. 여기서 주축은 앞의 `crystal`처럼 유전율을 대각 행렬로 표현할 수 있는 재료의 기준 방향입니다. 처음에는 두 축을 나란히 놓았지만, 재료만 돌리면 계산 좌표축은 그대로 있고 재료의 주축은 함께 돌아갑니다.

#### 고정된 계산축과 회전한 재료축

| 표기 | 무엇을 나타내는가 | 재료만 회전하면 |
| --- | --- | --- |
| x·y·z | 전기장·전기 변위와 재료 행렬의 성분을 표시하는 고정된 계산 좌표축 | 그대로 유지 |
| x′·y′·z′ | 재료에 붙어 있는 주축. ′는 재료축을 구분하는 표시 | 재료와 함께 회전 |

재료를 z축 주위로 +45° 회전한 상태를 xy 평면에서 보면 다음과 같습니다. +z 방향에서 원점을 내려다볼 때 양의 회전은 반시계 방향입니다.

```{figure} ../../figures/learn/learn/07-material-axes.svg
:alt: 고정된 계산축 x는 오른쪽, y는 위쪽이다. 재료를 45도 회전하면 재료축 x′는 오른쪽 위를 향하고 유전율은 2.25, y′는 왼쪽 위를 향하고 유전율은 4이다. 계산축은 그대로 남는다.
:figclass: learning-figure

그림 7-2. 회색은 고정된 계산축, 파란색은 +45° 회전한 재료축입니다. z와 z′는 화면 밖을 향하며 서로 일치합니다.
```

#### 어떤 값이 그대로이고, 어떤 값이 바뀌는가

상대 유전율이 `[2.25, 4., 2.25]`인 재료를 회전한다고 해 봅시다.

| 재료축 | 회전 전 방향과 유전율 | +45° 회전 후 방향과 유전율 |
| --- | --- | --- |
| 첫째 주축 | → x축 방향, 2.25 | ↗ x′축 방향, 2.25 |
| 둘째 주축 | ↑ y축 방향, 4.0 | ↖ y′축 방향, 4.0 |
| 셋째 주축 | +z 방향, 2.25 | +z′ 방향, 2.25 |

**값 2.25와 4가 적용되는 방향이 재료와 함께 돌아간 것입니다.** 회전한 재료축 x′·y′·z′에서 표현하면 유전율 행렬은 여전히 대각 성분이 `[2.25, 4., 2.25]`입니다. 반면 고정된 x·y·z축으로 같은 재료를 표현하면 비대각 성분이 생깁니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.25])
rotated = crystal.rotated(np.deg2rad(45.), axis="z")
```

Q를 z축 주위의 +45° 회전행렬이라고 하면, `.rotated()`는 다음 곱셈을 수행합니다.

```{raw} html
<div class="power-equations">
<p>ε<sub>회전 후</sub> = Q ε<sub>회전 전</sub> Q<sup>T</sup></p>
</div>
```

**아래 행렬은 앞의 Q와 뒤의 Qᵀ를 모두 곱한 최종 결과입니다.** `rotated.epsilon`에는 이 값이 저장되며, 각 성분은 고정된 x·y·z축 기준입니다.

```text
[ 3.125  -0.875   0.   ]
[-0.875   3.125   0.   ]
[ 0.      0.      2.25 ]
```

따라서 회전 후의 ε<sub>xx</sub>와 ε<sub>yy</sub>는 3.125이고, ε<sub>xy</sub>와 ε<sub>yx</sub>는 −0.875입니다. 이 값들은 위 표에서 회전한 재료축을 따라 읽은 2.25와 4를, 고정된 계산축으로 표현한 결과입니다.

벡터 하나의 방향을 회전할 때는 QE처럼 회전행렬을 앞에서 곱합니다. 유전율 텐서는 **전기장 E를 전기 변위 D로 연결하는 관계**이므로, 재료 회전은 입력 E 쪽의 Qᵀ와 출력 D 쪽의 Q를 함께 반영한 Q ε<sub>회전 전</sub> Qᵀ로 계산합니다. 위에서 완성한 행렬을 전기장에 적용할 때는 Q나 Qᵀ를 다시 곱하지 않습니다.

#### 회전한 재료의 유전율 텐서를 계산하는 과정

재료를 +45° 회전한 뒤 전기장 E를 가했을 때의 반응은 다음과 같은 등가 계산으로 구할 수 있습니다. **회전 전 재료 텐서를 그대로 두고, 전기장 성분을 −45° 회전해 적용한 다음, 계산된 전기 변위 성분을 +45° 회전합니다.**

두 재료 행렬은 회전 전후를 구분해 표기합니다. **둘 다 공간에 고정된 x·y·z축을 기준으로 표현한 상대 유전율 행렬입니다.**

| 표기 | 뜻 | 이 예제의 값 |
| --- | --- | --- |
| ε<sub>회전 전</sub> | 재료를 돌리기 전의 텐서 | 대각 성분 2.25·4·2.25, 비대각 성분 0 |
| ε<sub>회전 후</sub> | 재료를 +45° 돌린 뒤의 텐서 | 위에서 구한 `rotated.epsilon` 행렬 |

Q는 +45° 회전행렬입니다. Q<sup>T</sup>는 Q의 전치행렬이며, 역회전인 −45°를 나타냅니다. E′는 E의 성분을 −45° 돌린 중간값이고, D′는 이 E′에 회전 전 재료 텐서를 적용해 구한 전기 변위입니다.

| 단계 | 계산 | 뜻 |
| --- | --- | --- |
| ① 전기장 성분을 −45° 회전 | E′ = Q<sup>T</sup>E | 원래 재료에 적용할 중간 전기장 성분을 구함 |
| ② 회전 전 유전율 텐서 적용 | D′ = ε<sub>0</sub> ε<sub>회전 전</sub> E′ | 대각 성분 2.25·4·2.25를 각각 적용하고 ε<sub>0</sub>를 곱함 |
| ③ 계산된 전기 변위 성분을 +45° 회전 | D = QD′ | 회전한 재료에 원래 전기장 E를 가했을 때의 전기 변위를 구함 |

회전 전 텐서가 각 방향의 성분에 서로 다른 유전율을 적용하므로, 중간의 역회전과 마지막 정회전을 모두 포함해야 합니다. 위의 세 식을 합치면 다음 관계를 얻습니다.

```{raw} html
<div class="power-equations">
<p><b>D</b> = ε<sub>0</sub> Q ε<sub>회전 전</sub> Q<sup>T</sup> <b>E</b></p>
<p>ε<sub>회전 후</sub> = Q ε<sub>회전 전</sub> Q<sup>T</sup></p>
</div>
```

이 관계는 어떤 전기장 E에도 성립합니다. 따라서 `.rotated()`는 재료와 회전 각도만으로 ε<sub>회전 후</sub> = Q ε<sub>회전 전</sub> Q<sup>T</sup>를 미리 계산해 둡니다. 이후 솔버는 고정된 xyz 기준의 전기장 E에 회전 후 텐서를 바로 적용합니다.

#### 직접 계산하기: 오른쪽을 향하는 전기장

같은 +45° 회전 예제를 xy 평면에서 계산해 봅시다. 전기장은 E = (1, 0) V/m로 두고, z 성분은 0입니다. 아래 행렬은 3차원 회전행렬 중 xy 부분입니다.

```{raw} html
<div class="rotation-matrices">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" style="font-size:1.15rem" aria-label="45도 회전행렬 Q는 루트 2분의 1 곱하기 행렬 1, 마이너스 1, 1, 1">
<mrow><mi>Q</mi><mo>=</mo><mfrac><mn>1</mn><msqrt><mn>2</mn></msqrt></mfrac><mo>[</mo><mtable rowspacing="0.35em" columnspacing="0.8em"><mtr><mtd><mn>1</mn></mtd><mtd><mo>−</mo><mn>1</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow>
</math>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" style="font-size:1.15rem" aria-label="역회전행렬 Q의 전치는 루트 2분의 1 곱하기 행렬 1, 1, 마이너스 1, 1">
<mrow><msup><mi>Q</mi><mi>T</mi></msup><mo>=</mo><mfrac><mn>1</mn><msqrt><mn>2</mn></msqrt></mfrac><mo>[</mo><mtable rowspacing="0.35em" columnspacing="0.8em"><mtr><mtd><mn>1</mn></mtd><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mo>−</mo><mn>1</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow>
</math>
</div>
```

각 벡터를 두 성분의 순서쌍으로 표시하면 다음과 같습니다. 공통 인수 ε<sub>0</sub>는 밖에 두고 D/ε<sub>0</sub>를 계산하므로, 표의 수치 단위는 모두 V/m입니다.

| 단계 | 수행한 계산 | x·y 성분 |
| --- | --- | --- |
| 입력 E | 오른쪽을 향하는 전기장 | (1, 0) |
| E′ = Q<sup>T</sup>E | 전기장 성분을 −45° 회전 | (1/√2, −1/√2) |
| D′/ε<sub>0</sub> = ε<sub>회전 전</sub>E′ | 회전 전 텐서 적용 | (2.25/√2, −4/√2) |
| D/ε<sub>0</sub> = Q(D′/ε<sub>0</sub>) | 계산된 전기 변위 성분을 +45° 회전 | (3.125, −0.875) |

중간 전기장 E′는 ↘ 방향입니다. 회전 전 텐서를 적용하면 x 성분에는 2.25, y 성분에는 4가 곱해집니다. 그 결과를 +45° 돌리면, +45° 회전한 재료에 원래의 → 전기장 E를 가했을 때의 전기 변위 D를 얻습니다.

마지막 단계의 두 성분을 풀어 쓰면:

```{raw} html
<div class="power-equations">
<p>D<sub>x</sub>/ε<sub>0</sub> = (2.25 + 4)/2 = 3.125 V/m</p>
<p>D<sub>y</sub>/ε<sub>0</sub> = (2.25 − 4)/2 = −0.875 V/m</p>
</div>
```

앞서 구한 `rotated.epsilon`에 E = (1, 0, 0)를 바로 곱해도 D/ε<sub>0</sub> = (3.125, −0.875, 0)으로 같습니다.

#### 등방성 재료는 왜 회전해도 같은가

등방성 재료의 상대 유전율이 ε<sub>s</sub>라면 행렬은 ε<sub>s</sub>I입니다. I는 대각 성분이 모두 1이고 나머지가 0인 **항등행렬**입니다.

```{raw} html
<div class="power-equations">
<p>Q (ε<sub>s</sub>I) Q<sup>T</sup> = ε<sub>s</sub> QQ<sup>T</sup> = ε<sub>s</sub>I</p>
</div>
```

모든 성분에 같은 값을 곱하므로 −45°와 +45° 회전이 서로 상쇄됩니다. 이방성 예제에서는 두 성분에 2.25와 4라는 서로 다른 값을 곱하기 때문에, 회전한 재료를 고정된 계산축에서 나타내는 행렬이 달라집니다.

#### Wavebit에서 회전을 지정할 때

`.rotated(angle, axis=...)`는 회전한 재료를 새 객체로 반환합니다. 각도는 라디안이며 회전 방향은 오른손 규칙입니다. `axis`에는 `"x"`, `"y"`, `"z"`를 넣습니다. 지정된 ε와 μ를 함께 회전하고, 여러 번 호출하면 호출한 순서대로 회전합니다. 도형의 위치·모양과 격자는 그대로 유지됩니다.

x축이나 y축 주위로 재료를 기울이면 ε<sub>xz</sub>·ε<sub>yz</sub>·ε<sub>zx</sub>·ε<sub>zy</sub> 같은 성분이 생길 수 있습니다. 전체 3×3 행렬을 직접 입력하거나 `.rotated(...)`로 만든 재료를 같은 방식으로 할당합니다.

### 한 층 전체에 이방성 재료를 넣기

이방성 재료도 층의 `background`나 도형의 `material`에 넣습니다. 먼저 층 전체를 같은 재료로 채우려면 `background`를 지정하고 `shapes=[]`로 둡니다.

아래는 공기 사이에 두께 220 nm인 이방성 박막을 놓고, 위에서 수직으로 빛을 비추는 예제입니다. 길이 단위는 nm입니다.

<!-- uniform-anisotropic-example:start -->
```python
from wavebit.compat import legacy as wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.25])
solver = wavebit.rcwa(
    wavelength=600., thickness=[220.],
    period=(600., 600.), top=1., bot=1.,
    structure=[{"background": crystal, "shapes": []}],
    theta=0., phi=0., truncation_order=(0, 0),
)
result = solver.solve()
print("y 편광 반사율:", float(result.s_inc.R.sum()))
print("x 편광 반사율:", float(result.p_inc.R.sum()))
```
<!-- uniform-anisotropic-example:end -->

| 입사 전기장 방향 | 해당 방향의 상대 유전율 | 반사율 |
| --- | --- | --- |
| y 방향 | 4.0 | 0.357473 |
| x 방향 | 2.25 | 0.016308 |

이 예제의 수직 입사·`phi=0` 설정에서는 s 편광의 전기장이 y축과, p 편광의 전기장이 x축과 나란합니다. 방향별 유전율이 다르므로 반사율도 다릅니다. xy 평면에서 재료가 일정한 박막이어서 Fourier 차수는 0으로 충분합니다. `period`는 계산 셀 크기입니다.

### 도형 안에 이방성 재료를 넣기

아래는 공기 속 원기둥에 회전한 이방성 재료를 할당하는 예제입니다. 원기둥 안은 어느 위치에서나 같은 재료이고, 원기둥 밖은 공기입니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.25])
rotated = crystal.rotated(np.deg2rad(45.), axis="z")
solver = wavebit.rcwa(
    wavelength=700., period=(600., 600.), thickness=[170.],
    top=1., bot=1.,
    structure=[{"background": 1., "shapes": [
        {"type": "circle", "lx": 270., "material": rotated},
    ]}],
    truncation_order=(3, 3), shape_nvf_grid_size=128,
)
result = solver.solve()
solver.plot_structure(component="xx")
```

지름은 270 nm, 높이는 층 두께와 같은 170 nm입니다. `plot_structure(component="xx")`는 ε<sub>xx</sub>의 공간 분포를 보여줍니다. `component="xy"`를 지정하면 ε<sub>xy</sub>를 볼 수 있습니다. 위의 Fourier 차수와 NVF 표본 수는 예제를 실행하기 위한 값이며, 필요한 정확도는 [19장의 수렴 확인](convergence.md)으로 판단합니다.

**이방성은 한 위치에서의 방향별 성질이고, 재료 분포는 위치에 따른 차이입니다.** 이 둘은 별도로 정합니다.

| 구조 | 방향에 따른 성질 | xy 평면의 재료 분포 |
| --- | --- | --- |
| 같은 이방성 재료로 채운 박막 | 이방성 | 어디서나 같음 |
| 공기 속 등방성 원기둥 | 각 재료는 등방성 | 원기둥 안과 밖이 다름 |
| 공기 속 이방성 원기둥 | 원기둥 재료는 이방성 | 원기둥 안과 밖이 다름 |

### 래스터 배열로 넣기

래스터는 각 공간 칸에 재료값을 저장합니다. 전체 유전율 행렬을 저장할 때 `structure`의 배열 모양은 `(층, 행, 열, 3, 3)`입니다. 마지막 두 축은 **각 칸의 유전율 행렬의 행과 열**입니다. 투자율은 별도의 `mu` 배열로 넣습니다.

아래는 한 층의 2×4개 칸을 모두 같은 재료로 채운 배열입니다.

```python
import numpy as np

cell_tensor = np.diag([2.25, 4., 2.25])
structure = np.broadcast_to(cell_tensor, (1, 2, 4, 3, 3)).copy()
```

박막 예제의 `structure` 자리에 이 배열을 넣으면 같은 균일층을 표현합니다. 칸마다 다른 재료를 넣으려면 해당 칸의 행렬을 바꿉니다. 예를 들어 `structure[0, 0, 0] = np.eye(3)`은 첫 층의 0번 행·0번 열을 상대 유전율 1인 등방성 재료로 바꿉니다. 투자율 배열도 같은 공간 격자를 사용하며, 각 칸에 스칼라 또는 3×3 행렬을 저장할 수 있습니다.

### 파장별 이방성 재료를 넣기

`TabulatedMaterial`에 파장마다 세 대각 성분 또는 3×3 행렬을 넣으면, 계산 파장에 맞춰 각 성분을 보간합니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

crystal_table = wavebit.TabulatedMaterial(
    wavelengths=[500., 700.],
    epsilon=[[2., 3., 2.5], [3., 4., 2.9]],
    mu=[[1.1, 1.2, 1.3], [1.2, 1.3, 1.4]],
)
value = crystal_table.at(600.)
# epsilon의 대각 성분: [2.5, 3.5, 2.7]
# mu의 대각 성분: [1.15, 1.25, 1.35]

rotated_table = crystal_table.rotated(np.deg2rad(30.), axis="z")
```

`rotated_table`을 도형이나 층 배경에 한 번 할당하면, 파장을 바꿀 때마다 그 파장의 회전된 재료값을 사용합니다. 숫자 래스터를 사용할 때는 파장별로 재료값을 구해 배열을 갱신합니다.

### 계산 옵션과 지원 범위

위 예제처럼 재료를 할당하고 `solve()`를 호출하면 Wavebit가 필요한 계산 행렬을 구성합니다. 구조 표현에 따라 선택할 수 있는 방식은 다음과 같습니다.

| 구조 | 계산 방식 |
| --- | --- |
| xy 평면에서 재료가 일정한 층 | 균일층 계산; 전체 3×3 ε·μ 지원 |
| x·y축과 나란한 직교 셀의 래스터 | NVF 또는 Li |
| 같은 직교 셀 안에서 변이 x·y축과 나란한 사각형 도형 | NVF 또는 Li |
| 원·타원·회전한 도형, 또는 사선 격자 | NVF |
| 위·아래 반무한 매질 `top`·`bot` | 등방성 스칼라 ε·μ 지원; 이방성 텐서는 미지원 |

이방성 재료가 포함된 패턴의 기본 설정은 `factorization="nvf"`, `nvf_projector="linear"`입니다. `linear`는 NVF의 투영행렬을 만드는 방식이며, 입사광의 편광은 [8장](light.md)의 별도 입력으로 정합니다. Li를 선택하려면 위 표의 조건에 맞는 구조에서 `factorization="li"`를 지정합니다.

ε나 μ에 xz·yz·zx·zy 성분이 있으면 그 성분까지 함께 다루는 1차 정식화(`first_order`, 옛 이름 4ff)가 자동으로 선택됩니다. 층 연결에는 기본값인 `layer_algorithm="ETM"`과 `layer_algorithm="SMM"`을 모두 사용할 수 있습니다. 두 방식 모두 반사·투과와 구조 내부의 전기장·자기장을 계산합니다. 연결 방식과 산란행렬 사용법은 [15장](layer-algorithms.md)에서 설명합니다. [전체 텐서 예제](../../guide/rcwa/full-tensor-patterns.md)에서 기울어진 재료를 사각형 도형에 넣는 실행 코드를 볼 수 있습니다.

Li·NVF와 Fourier 계수 계산 방법은 [10장](fourier-orders.md), NVF의 법선장과 투영행렬은 [11장](nvf.md)에서 설명합니다. 이방성 NVF 도형의 주기 복제본끼리 겹치거나 셀 경계 너머에서 도형들이 서로 덮는 경우에는 `fourier_coefficients="sampled"`를 사용합니다. `"integrated"`의 형상 조건은 [도형 적분 안내](../../guide/rcwa/tensor-shape-integration.md)에 정리되어 있습니다.

유한한 차수의 NVF에서는 손실 없는 재료라도 R+T가 정확히 1이 아닐 수 있습니다. 계산 정확도는 [19장](convergence.md)에 따라 관심 결과의 수렴을 확인해 판단합니다.

```{dropdown} 자동미분을 사용할 때의 조건
NumPy·JAX·PyTorch가 같은 재료 입력을 받습니다. 자동미분의 기본 사용법은 [17장](autodiff.md)을 참고합니다.

- NVF의 도형 적분 경로는 재료값과 도형의 위치·크기·회전 각도, 격자 미분을 지원합니다. JAX 추적 전에 `freeze_geometry()`로 도형들의 교차·접촉 관계를 기록하고 `solve(geometry=...)`에 전달합니다. 겹침이나 주기 접촉 관계가 달라지면 추적 밖에서 다시 기록합니다. 법선장이 여러 경계로부터 같은 거리에 놓이는 지점 근처에서는 미분값의 격자 수렴도 확인합니다.
- NVF의 `"sampled"` 도형과 텐서 래스터는 고정된 경계에서 재료값 미분을 지원합니다. 이 경로의 도형·격자 미분은 지원하지 않습니다.
- Li 텐서 래스터는 재료값과 직교 격자 주기를 미분할 수 있습니다. Li 사각형 도형은 경계 순서가 유지되는 범위에서 재료값·위치·크기·주기를 미분할 수 있습니다. 차수·법선장 격자·계수 계산 방식을 바꾸면 기록한 계산 준비 정보도 갱신합니다.
- 분산 표의 표본 값과 표 전체를 회전하는 각도는 고정 자료입니다. 재료 회전 각도를 설계 변수로 미분하려면 `crystal_table.at(wavelength).rotated(angle)`처럼 현재 파장의 값을 구한 뒤 회전합니다. `.at()`에 파장 배열을 넣어 얻은 값은 스펙트럼 확인용이며, 한 재료 영역에는 재료 객체나 한 파장에서 구한 값을 할당합니다.
- 미분이나 JAX 추적 중에는 현재 0인 비대각 성분도 유지합니다. 따라서 현재 값에서 xz·yz·zx·zy 성분이 0이어도 1차 경로로 계산할 수 있습니다.
```
