# 재료와 파장 스윕

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 필요합니다.
> 현재 설치에는 이 네임스페이스가 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행되지 않았습니다.


:::{note}
이 페이지는 예전 인터페이스의 전체 호환성 참조입니다. `wavebit.compat.legacy`를 명시적으로 쓰세요. 새 애플리케이션은 [현재 생애주기](../guide/common/lifecycle.md)와 [현재 API 참조](../reference/api.md)로 시작하세요. 여기 인용한 수치 근거는 원래의 범위를 그대로 유지합니다.
:::

영역에 재료를 한 번 지정하세요. Wavebit은 RCWA 계산 파장마다 그 재료의 상대 유전율과
선택적인 상대 투자율을 계산합니다.

## 재료 선택하기

```python
from wavebit.compat import legacy as wavebit

silicon = wavebit.materials.catalog.load('Si')
solver = wavebit.rcwa(
    wavelength=900., period=(600., 600.), thickness=[120.],
    structure=[dict(background=silicon, shapes=[])],
    truncation_order=(0, 0), polarization=(1., 0.), factorization='li',
)
for wavelength in (800., 900., 1000.):
    result = solver.solve(wavelength=wavelength)
    print(wavelength, result.R.sum(), result.T.sum())
```

`materials.load`는 언제나 변경할 수 없는 `TabulatedMaterial`을 반환합니다. 파장을 받지
않습니다. 선택한 데이터 파일만 읽어 들이며, 같은 선택을 반복하면 캐시를 씁니다. 표 읽기는
수치 백엔드나 정밀도에 의존하지 않습니다.

```python
wavebit.materials.catalog.available()             # source-qualified catalog identifiers
silicon = wavebit.materials.catalog.load('Green2008-Li1980/Si')
value = silicon.at(900.)                   # numerical Material at one wavelength
spectrum = silicon.at([800., 900., 1000.]) # Material containing epsilon arrays
print(value.epsilon, silicon.wavelength_range)
```

이름은 대소문자를 구분하지 않습니다. `Si` 같은 짧은 이름은 모호하지 않을 때 동작하고,
한정된 이름은 특정 데이터셋을 식별합니다. 카탈로그에는
`Green2008-Li1980/Si`, `Luke2015/Si3N4`, `Malitson1962/Al2O3`, `Malitson1965/SiO2`가 있으며,
각각 수치의 출처인 논문의 이름을 땄습니다(2026-10-19). 예전의
`filmetrics/...` 표에는 기록된 출처나 이용 조건이 없어서 제거되었습니다. 그 이름을 읽으면
대체 이름을 나열하는 오류가 납니다.
화학식이 재료의 모든 증착 조건이나 형태를 지정하지는 않습니다.
시료에 맞는 데이터를 고르세요. 파장 단위와 데이터 출처를 모두 선언하지 않았던 예전
`p_Si.mat` 자산은 2026-10-14에 패키지에서 제거되었습니다.

## 장면의 길이 단위 고르기

기본값은 **nm**입니다. 마이크로미터로 표현한 장면에서는 다음과 같이 합니다.

```python
silicon = wavebit.materials.catalog.load('Si', wavelength_unit='um')
value = silicon.at(0.9)
# Use wavelength=0.9, period=0.6, thickness=[0.12], etc. in that scene.
```

`wavelength_unit`은 `nm`, `um`, `mm`, `m`을 받습니다. 이 값은 표의 파장 축을 받는 장면의
단위로 변환합니다. 기하나 솔버 파장은 결코 바꾸지 않습니다. 장면의 수치 입력에는 단위가
없으므로 모든 길이가 서로 맞아야 합니다. 사용자 정의 표도 마찬가지로 장면과 같은 길이
단위를 씁니다.

## 정의와 계산된 값

| 입력 | 계산 파장이 바뀔 때의 의미 |
| --- | --- |
| `Material(12.25)` 또는 `12.25` | 상수 상대 유전율 |
| `materials.load('Si')` | 유지된 재료 정의를 다시 계산 |
| `silicon.at(900.)` | 900 nm에서 계산한 고정된 수치 `Material` |
| `TabulatedMaterial(wavelengths, epsilon)` | 사용자 정의 유전율 표를 다시 계산 |

`Material(epsilon, mu=None)`은 수치 값을 담는 컨테이너입니다. `mu=None`은 비자성 경로,
즉 물리적으로 상대 투자율이 1인 경로를 선택합니다. `mu=1`을 명시하면 자성 경로가
선택됩니다. `Material.at(wavelength)`는 같은 상수 객체를 반환하며, 상수를 파장 배열로
브로드캐스트하지 않습니다.

`Material(-2 + 0.1j, mu=-1 + 0.1j)`처럼 유전율과 투자율의 실수부가 모두 음수인 `Material`은
이중 음성 매질, 즉 음굴절률 매질입니다. 빛이 들어오든 아니든 어느 쪽 반무한 공간으로도,
어느 층으로도 쓸 수 있지만, 전파하는 능동(이득) 매질은 거부합니다.
그 굴절률은 `solver.n_top` / `n_bot`이 보고하듯 `Re(n) < 0`인 `sqrt(eps)*sqrt(mu)`이며,
그런 쪽에서의 `theta`도 여전히 빛이 진행하는 방향을 나타냅니다. [음굴절률
매질](../theory/rcwa/negative-index-media.md)이 무엇이 바뀌는지와 그 이유를 설명합니다.

모델은 형상의 `material`, 층의 `background`, `top` / `bot` 슬롯에서 동작합니다.
`solver.top`, `solver.bot`, `solver.structure`는 현재 파장에서 계산된 수치 값을 보여
줍니다. 영역을 바꾸려면 `top`, `bot` 또는 새 `structure`를 지정하세요. 계산된 그
딕셔너리를 제자리에서 편집하지 마세요.

래스터 `structure`와 `mu` 배열은 수치 필드입니다. 픽셀마다 재료 정체성을 유지하지
않습니다. 래스터 스윕에서는 `.at(wavelength)`로 재료를 계산하고 파장마다 배열을 다시
만드세요. 마찬가지로 `solve_conv`는 이미 만들어진 행렬을 쓰며 대신 다시 만들어 주지
않습니다.

## Anisotropic materials

스칼라, 대각, 전체 텐서 데이터에 같은 `Material` 지정을 쓰세요.
텐서 성분은 Wavebit의 x, y, z 축을 가리킵니다. 구성 관계식은
`D = epsilon_0 * epsilon_r @ E`입니다.

이방성은 한 점에서 서로 다른 필드 방향에 대한 재료의 응답을 기술합니다. 공간 변화는
별개의 속성입니다. 균일한 이방성 층은 xy 평면 어디서나 같은 재료 텐서를 가집니다. 공기
속의 이방성 원기둥은 이방성 재료를 담은 패턴 구조인데, 원기둥과 그 주변 사이에서 재료가
바뀌기 때문입니다. 아래 예제는 이 두 경우를 이 순서로 보여 줍니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.25])
rotated = crystal.rotated(np.deg2rad(30.), axis='z')
# Equivalent full tensor spelling, with physical signs:
full = wavebit.Material(epsilon=[[2.25, 0., 0.],
                                [0., 4., 0.],
                                [0., 0., 2.25]])
solver = wavebit.rcwa(
    wavelength=600., period=(600., 600.), thickness=[220.],
    structure=[{'background': crystal, 'shapes': []}],
    truncation_order=(0, 0), phi=0.,
)
result = solver.solve()
print(float(result.s_inc.R.sum()))  # 0.3574726625
print(float(result.p_inc.R.sum()))  # 0.0163080252
```

세 값은 xx, yy, zz 대각 성분입니다. 3×3 배열에는 독립적인 비대각 성분을 넣을 수 있으며,
자동으로 대칭화되지 않습니다.
`.rotated(angle, axis='x'|'y'|'z')`는 라디안 단위의 능동적인 오른손 회전
`R @ tensor @ R.T`를 epsilon과 선택적인 mu 모두에 적용합니다. 결과는 새 재료이며, 형상과
격자는 회전하지 않습니다. 여러 번 호출하면 적은 순서대로 회전이 적용됩니다. 스칼라는
스칼라로 남습니다. 이 메서드는 살아 있는 JAX/PyTorch 재료 미분과 각도 미분을 보존합니다.

패턴에도 같은 지정을 씁니다.

```python
solver = wavebit.rcwa(
    wavelength=700., period=(600., 600.), thickness=[170.],
    structure=[{'background': 1., 'shapes': [
        {'type': 'circle', 'lx': 270., 'material': rotated},
    ]}],
    truncation_order=(3, 3), shape_nvf_grid_size=128,
)
result = solver.solve()
solver.plot_structure(component='xx')
```

같은 형상 `material` 슬롯과 층 `background` 슬롯은 스칼라 재료와 텐서 재료를 모두 받습니다.
이 예제는 지름 270 nm, 높이 170 nm인 이방성 원기둥을 상대 유전율이 1인 배경에 놓습니다.
재료 축을 회전해도 원기둥의 모양은 바뀌지 않습니다. `component='xx'`는 상대 유전율의 xx
성분을 표시하며, 다른 성분을 보려면 다른 component를 쓰세요.

`shape_nvf_grid_size=128`은 NVF 표본화를 조절합니다. 이 격자 크기와 푸리에 차수는 시작
값이지 수렴 결과가 아닙니다. 계수 구성과 표본화는
[수치 설정 고르기](../guide/rcwa/numerics.md)를 보세요.

텐서 래스터는 공간 픽셀마다 3×3 재료 행렬을 저장합니다. 그 모양은
**`(layers, Ny, Nx, 3, 3)`**(마지막 두 축에 물리적 구성 성분)입니다. `mu`는 또 다른
텐서 래스터일 수도 있고 같은 공간 격자 위의 스칼라 필드일 수도 있습니다. 마찬가지로 스칼라
epsilon 필드를 텐서 mu와 짝지을 수도 있습니다. 래스터 배열은 재료 모델이 아니라 계산된
값이며, 층은 여전히 아래에서 위 순서입니다.

| 재료와 기하 | 지원하는 경로 |
| --- | --- |
| 균일한 대각 또는 전체 epsilon/mu 텐서 | `Material`을 지정; `solve()`가 정확한 균일 블록을 만듦 |
| xz/yz/zx/zy 결합이 있는 균일한 epsilon 또는 mu | 형상 없는 배경 또는 상수 텐서 래스터; `formulation='auto'`가 `'first_order'`를 선택 |
| 전체 epsilon 및/또는 mu 패턴 | 기본값은 linear NVF; 래스터와 축 정렬 직사각형 칠하기에는 Cartesian Li |
| 텐서 파장 표 | 파장마다 대각 또는 3×3 텐서를 가진 `TabulatedMaterial` |
| 명시적 mu를 포함해 z 결합이 있는 패턴 | 세 성분 전체 인수분해와 1차 정식화; ETM 또는 SMM 층 알고리즘 |
| 이방성 `top` 또는 `bot` 반무한 공간 | 지원하지 않음; 반무한 공간은 스칼라로 유지 |

이방성 재료를 담은 패턴 구조의 기본값은
`factorization='nvf'`와 `nvf_projector='linear'`입니다.
텐서 입력에는 이 값들이 기본으로 선택됩니다. NVF는 재료 계면의 법선을 써서 재료–필드 곱을
구성합니다. `linear`는 사영 연산자 구성을 고르는 것이지 입사 편광을 고르는 것이 아닙니다.
명시적인 `factorization='li'`는 축 정렬 직사각형 셀 안의 텐서 래스터와 축 정렬 직사각형
칠하기를 지원하며, `li_nesting='fixed'`만 허용합니다. 두 규칙 모두 epsilon과 mu의 아홉 성분을
모두 유지합니다. `factorization='laurent'`는 균일한 텐서를 포함해 모든 텐서 재료에서
거부합니다. 텐서 Laurent 구성은 작성되어 있지 않고, 균일한 층은 어떤 규칙에서도 같으므로
그 경우에는 키워드를 지정하지 마세요. z 결합이 있으면 1차 솔버가 자동으로 선택됩니다. 예제와
검증은 [전체 텐서 패턴](../guide/rcwa/full-tensor-patterns.md)을 보세요.
[텐서 Li 구성과 사용법](../theory/rcwa/tensor-li-factorization.md)도 보세요.
NVF 아래의 텐서 형상에서는 `fourier_coefficients='sampled'`가, 적분 경로가 현재 거부하는
주기 칠하기도 지원합니다. 자기 자신의 상(image)과 겹치거나 셀 가장자리를 통해 서로 겹치는
형상입니다. 이 한계와 수치 구성은
[텐서 형상 적분](../guide/rcwa/tensor-shape-integration.md)을 보세요. 호환되지 않는 명시적
옵션은 오류를 냅니다.
균일한 이방성 층은 `shapes=[]`인 층의 `background`로 지정할 수 있습니다. 래스터로 표현하려면
모든 공간 픽셀에 같은 재료 행렬을 저장하세요. 0이 아닌 xz/yz/zx/zy 성분을 가진 균일한 층은
형상 없는 배경이나 1×1 공간 래스터로 지원됩니다.

`plot_structure`와 `slice_structure`에는 `'xx'`나 `'xy'` 같은 텐서 `component`가 필요합니다.
이들은 이방성 매질에 하나의 굴절률을 부여하거나 대역 제한된 텐서 근사를 표시하지 않습니다.

형상 패턴이나 래스터 NVF를 JAX로 추적하려면 구체적인 데이터로 `freeze_geometry()`를 한 번
호출하고 그 기록을 `solve(geometry=...)`로 재사용하세요. 적분 NVF 형상은 저장된 교차 및 주기
계면 위상이 유효한 동안 재료, 위치, 크기, 각도, 격자 미분을 지원합니다. 즉시 실행 재생은 그
위상을 검사하지만 JIT은 다시 계획할 수 없으므로, 접촉이나 겹침이 바뀌는 지점을 지나거나
타원이 자기 중심의 Wigner-Seitz 셀 밖으로 커질 때는 추적 밖에서 다시 고정하세요. 표본화된
법선 확장은 전환되는 곳에서 여전히 계단처럼 변합니다. 곡선의 주기 사본 사이의 동률, 대비
가중 혼합에서 형상 사이, 그리고 모서리에서입니다. 이 계단은 격자를 세분해도 확실히 사라지지
않는 크기만큼 그래디언트를 치우치게 합니다. 그래디언트를 설계 매개변수의 한 구간에 걸친 손실과
대조해 확인하세요. [자동 미분](../guide/common/autodiff.md)을 보세요.

명시적인 표본화 형상 경로와 NVF 아래의 텐서 래스터는 고정된 기하를 유지합니다.
재료 미분은 지원하고, 형상 미분과 격자 미분은 거부합니다.
기하, 법선 방법, 재료 계면이 바뀌면 새 기록이 필요합니다. 두 형상 경로 어느 쪽이든 푸리에
차수, 형상 격자, 계수 모드를 바꿔도 새 기록이 필요합니다. 텐서 Li 래스터에는 고정된 법선이
필요 없으며, 재료 미분과 직사각형 주기 미분을 지원합니다. Li 직사각형은 고정된 가장자리
순서로 재료, 위치, 크기, 주기 미분을 지원하며, 방향은 축 정렬로 유지됩니다. 표 표본과 표
회전은 고정된 데이터이므로, 각도를 최적화할 때는 `.at(wavelength).rotated(angle)`을 쓰세요.
균일한 재료/각도, 두께, 파장 미분은 선택한 솔버의 미분 지원 범위 안에서 계속 쓸 수 있습니다.
추적된 전체 텐서는 현재 0인 결합의 미분을 포함해 아홉 슬롯을 모두 유지합니다.
[전체 텐서 패턴](../guide/rcwa/full-tensor-patterns.md)을 보세요.

자신의 행렬을 넘기는 호출자는 여전히 `solve_conv`를 쓸 수 있습니다. 그
**비대각 슬롯에는 물리 성분에 음수를 곱한 값이 들어갑니다**. `Material`과 래스터는 언제나
물리 성분 자체를 받습니다. 바탕의 계산은
[슬롯 규약](../theory/rcwa/off-diagonal-slot-convention.md),
[텐서 인수분해](../theory/rcwa/patterned-anisotropic-factorization.md),
[1차 정식화](../theory/rcwa/first-order-formulation.md)를 보세요.

## 측정했다면 투자율도 넣기

표에는 epsilon과 mu를 모두 담을 수 있습니다.

```python
magnetic = wavebit.TabulatedMaterial(
    wavelengths=[800., 900., 1000.],
    epsilon=[4., 4.2, 4.5],
    mu=[1.1, 1.2, 1.3],
)
value = magnetic.at(850.)  # epsilon=4.1, mu=1.15
```

두 양 모두 파장 스윕 동안 다시 계산되고 저장된 Job에 유지됩니다. 동봉된 카탈로그에는
**n과 k만** 있으므로 비자성 재료(`mu=None`, 물리적으로 mu_r=1)를 가정합니다. 굴절률
데이터만으로는 둘을 따로 얻을 수 없고 곱 `epsilon * mu`만 얻습니다. 같은 파장에서 등방성
굴절률과 상대 투자율을 모두 알고 있다면 명시적으로 변환하세요.

```python
n_complex, mu_r = 2.+0.1j, 1.25
material = wavebit.Material(
    epsilon=wavebit.eps_from_refractive_index(n_complex, mu=mu_r),
    mu=mu_r,
)
```

`eps_from_refractive_index(n, mu=...)`는 `n**2 / mu`를 계산하며, `mu`를 생략하면 비자성
`n**2` 변환을 유지합니다. 이것은 스칼라 등방성 관계이며 배치에는 원소별로 적용됩니다.
굴절률에서 이방성 텐서를 복원하지는 않습니다.
투자율은 유한하고 0이 아니어야 합니다. 잘못된 값은 즉시 실행에서 오류를 내거나, 추적된 JAX 배열에서는
잘못된 원소가 NaN이 됩니다.

파장에 따라 변하는 굴절률 데이터에는 같은 격자 위의 투자율을 함께 주세요.

```python
magnetic = wavebit.TabulatedMaterial.from_refractive_index(
    wavelengths=[800., 1000.],
    index=[2., 3.],
    mu=[1., 2.],
)
value = magnetic.at(900.)  # index=2.5, mu=1.5, epsilon=2.5**2/1.5
# mu=1.25 is also accepted for constant permeability over the table range.
```

굴절률과 투자율은 따로 보간되고, 그다음 계산된 유전율이 `index**2 / mu`가 됩니다. 끝점을
먼저 제곱하고 나눈 뒤 epsilon을 보간하면 표본 사이에서 다른 재료가 정의됩니다. 주어진
투자율은 계산된 `Material`과 저장된 Job에 유지됩니다.
보간이 투자율 0에 이르면 변환이 즉시 실행에서 오류를 내고, JAX JIT 아래에서는 해당 epsilon 원소가
NaN이 됩니다. 스칼라 굴절률로는 mu가 0일 때의 epsilon을 정할 수 없으므로, 이 경우에는 독립적인
구성 데이터가 필요합니다.

## 자신의 데이터 가져오기

```python
material = wavebit.TabulatedMaterial(
    wavelengths=[800., 900., 1000.],
    epsilon=[4.+0.1j, 4.2+0.12j, 4.5+0.15j],
)
print(material.at(850.).epsilon)   # 4.1 + 0.11j
```

표본은 둘 이상의 양수이며 엄격히 증가하는 파장에서 주어진 유한한 스칼라, 대각 성분 세 개,
또는 3×3 텐서여야 합니다. 한 양의 모든 표본은 같은 모양이어야 합니다. 복소 유전율은 성분별로
보간됩니다. 선택적인 `mu`는 길이가 같은 표이며 같은 방식으로 보간됩니다. 입력 배열은 변경할
수 없는 튜플로 복사됩니다. 예를 들면 다음과 같습니다.

```python
crystal = wavebit.TabulatedMaterial(
    wavelengths=[500., 700.],
    epsilon=[[2., 3., 2.5], [3., 4., 2.9]],
    mu=[[1.1, 1.2, 1.3], [1.2, 1.3, 1.4]],
).rotated(0.3, axis='z')
# Assign crystal to a background or shape; it is evaluated at each solve wavelength.
```

`from_refractive_index`는 등방성 스칼라 가져오기이며, 선택적으로 스칼라 mu를 포함합니다.
이 경로에서 텐서 투자율은 거부합니다. 일반 텐서는 하나의 굴절률로 지정할 수 없습니다. 텐서
데이터에서 `.at(wavelengths).epsilon`은 파장 배열 모양 뒤에 `(3,)` 또는 `(3, 3)`을 덧붙입니다.
그런 스펙트럼은 분석용입니다. 영역에는 표 자체를 지정하거나 한 파장에서 계산한 값을 지정하세요.

굴절률로 발표된 외부 데이터는 명시적으로 가져오세요.

```python
material = wavebit.TabulatedMaterial.from_refractive_index(
    wavelengths=[800., 1000.], index=[2.+0.1j, 3.+0.3j],
)
```

이 비자성 모델은 `(n + 1j*k)**2`를 계산하기 전에 n과 k를 보간합니다.
동봉된 카탈로그도 이 규약을 따릅니다. epsilon을 직접 보간하면 표본 사이에서 다른 모델이
됩니다. `mu=...`를 주면 `from_refractive_index`는 투자율도 보간하고 제곱한 굴절률을 그 값으로
나눕니다. 솔버 입력은 여전히 epsilon과 mu입니다.
유도된 굴절률 그래프에는 `refractive_index_from_eps(value.epsilon)`을 쓸 수 있으며, 별도의
굴절률 조회나 이름 접미사는 필요 없습니다. 흡수를 없애면 물리 모델이 바뀌므로, 주는 데이터에
명시적으로 드러나야 합니다.

## 표본 파장 사이에서 계산하기

`wavelengths` 생성자 인수는 재료 표본이 알려진 위치를 나타냅니다. 허용되는 계산 파장을
열거하는 것이 아닙니다. 800, 900, 1000에서의 표는 850이나 873.2에서도 계산할 수 있으며,
epsilon과 선택적인 mu는 인접한 표본 사이에서 보간됩니다. 800–1000 밖에서 계산하려면 그 범위를
덮는 표가 필요합니다. 동봉된 Si 표는 250–14000 nm를 덮습니다. 보간으로는 원본 데이터에 없는
스펙트럼 특징을 복원할 수 없으므로, 충분한 표본 간격을 고르세요.

## 유효 범위, 미분, 저장된 Job

범위를 벗어나거나 유한하지 않은 계산은 즉시 실행에서 `ValueError`를 냅니다. JAX JIT 아래에서는
잘못된 원소가 NaN이 되고, 같은 배열의 유효한 원소는 유효한 값으로 남습니다.
어떤 계산도 조용히 끝점에 고정하거나 외삽하지 않습니다.

NumPy, JAX, PyTorch는 같은 보간을 공유합니다. `.at`은 파장 배열의 모양과 백엔드를
보존합니다. JAX와 PyTorch는 계산 중 재료에 미치는 영향까지 포함해 파장에 대해 미분합니다.
표 표본은 정적이며 최적화 변수가 아닙니다. 내부 매듭점에서 미분은 오른쪽 구간을 쓰고,
마지막 끝점은 마지막 구간을 씁니다. 이것은 보간의 미분이지 맞춘 물리적 분산 법칙의 미분이
아닙니다. 평소의 솔버 제약과 고정 기하 요구 사항은 그대로 적용됩니다.

`JobSpec.to_json()`은 재료의 표를 장면의 길이 단위로 `dispersion` 객체 안에 포함합니다.
그것을 다시 구성하는 데 카탈로그나 원본 파일이 필요 없습니다.
표 값과 보간하는 양은 `spec.key()`에 반영됩니다.
`JobSpec.expand()`와 `wavebit.run(point)`는 각 스윕 점을 계산합니다. 이 기능은 스키마 9에서
도입되었습니다. 현재 스키마는 **10**이며, 층 순서를 아래에서 위로, z를 위쪽이 양수인 높이로도
바꿉니다. 예전 문서는 [규약](../reference/common/conventions.md)의 좌표 마이그레이션을 따라야
하며, 버전 번호만 바꾸는 것으로는 충분하지 않습니다.

이 표들은 RCWA를 위한 주파수 영역 구성 데이터를 제공합니다. FDTD를 위한 인과적인 시간 영역
구현은 제공하지 않으며, FDTD에는 별도의 모델이 필요합니다.
