# 11. NVF는 전체 계산의 어디에 들어갈까

:::{note}
이 페이지의 본문과 코드는 현재 `rcwa` API를 씁니다. 그림에는 이전 생성자의 이름(`bot`, `structure`,
원의 지름 `lx`)이 남아 있을 수 있습니다. 현재 표기는 [기본 약속](conventions.md)을 참고하세요.
:::

NVF는 **재료 경계조건을 반영해 재료와 장의 곱을 Fourier 행렬로 구성하는 방법**입니다. 이렇게 만든 행렬로 각 층의 고유모드를 구하고, 층 사이의 경계조건을 연결해 반사·투과와 전기장·자기장을 계산합니다. 이 장에서는 NVF가 필요한 이유부터 방향 정보의 표현, 옵션 설정, 계산 순서까지 살펴봅니다.

## 재료 경계에서 연속인 성분

재료 경계면에 **나란한 방향을 접선 방향**, **수직인 방향을 법선 방향**이라고 합니다. 경계 양쪽의 재료를 1과 2라고 할 때, 자유 표면 전하가 없는 일반 유전체 경계에서는 다음 두 조건을 동시에 만족합니다. 여기서 **연속**이란 경계 바로 양쪽에서 값이 같다는 뜻입니다.

- **접선 방향의 전기장 {math}`E`가 연속입니다:** {math}`E_{1,\parallel}=E_{2,\parallel}`.
- **법선 방향의 전속밀도 {math}`D`가 연속입니다:** {math}`D_{1,\perp}=D_{2,\perp}`.

등방성 재료에서는 {math}`D=\varepsilon_0\varepsilon_r E`입니다. 따라서 유전율이 다른 두 재료의 경계에서는 **접선 전기장은 같아도 접선 전속밀도는 달라질 수 있고, 법선 전속밀도는 같아도 법선 전기장은 달라질 수 있습니다.** 예를 들어 상대유전율이 2인 재료와 4인 재료가 맞닿아 있다면, 법선 성분은 {math}`2E_{1,\perp}=4E_{2,\perp}`를 만족하므로 두 번째 재료의 법선 전기장은 첫 번째의 절반입니다.

NVF는 경계의 법선 방향을 이용해 접선·법선 성분을 구분하고, 위의 연속 조건에 맞는 Fourier 곱셈 규칙을 적용합니다. 특히 곡선 경계에서는 접선·법선 방향이 위치에 따라 달라지므로, 각 위치의 방향 정보가 필요합니다.

```{figure} ../../figures/learn/learn/10-nvf.svg
:alt: 래스터·도형에서 NVF 행렬을 거쳐 공통 솔버로 합류하는 위치
:figclass: learning-figure

그림 11-1. 래스터·도형에서 NVF 행렬을 거쳐 공통 솔버로 합류하는 위치
```

## 경계 방향을 행렬에 반영하기

경계의 법선 `n`은 기하학에서 정해집니다. 셀 전체에 연장한 보조장 `J`로 투영행렬 `P`를 만들고, 이를 재료 구성 행렬 조립에 사용합니다. 단위 실수 벡터라면 `P=nnᵀ`, 적절히 정규화한 복소 보조벡터라면 `P=JJᴴ` 형태입니다. `P`는 벡터에 작용할 수 있는 행렬이므로 법선 쪽 성분을 선택하는 연산에 들어갑니다. 경계 바깥의 연장과 복소 구성까지 실제 기하학적 법선 하나라고 해석하지 않습니다.

## NVF 계산 옵션

| 옵션 | 무엇을 선택하나 |
| --- | --- |
| `raster_normal_method` | 래스터에서 보조 방향장을 만드는 방법: distance / contour / spectral / smooth |
| `shape_nvf_grid_size` | 도형 NVF 계산용 표본 수, 축당 개수. 128이면 128×128 |
| `nvf_projector` | NVF 투영 구성: linear / circular |

도형에 `fourier_coefficients='integrated'`를 사용하면, **재료의 Fourier 계수는 도형 영역을 적분해 구합니다.** `shape_nvf_grid_size`는 **경계의 법선 방향을 반영하는 NVF 보조장과 투영행렬을 계산할 격자 크기**를 정합니다.

예를 들어 원형 도형에 `shape_nvf_grid_size=128`을 지정하면, 재료의 Fourier 계수는 원의 위치·반지름으로 정의한 원 영역을 적분해 구하고, NVF 계산에는 128×128 격자를 사용합니다. 이를 256으로 늘리면 NVF 계산의 격자가 256×256으로 촘촘해집니다. 원 영역을 적분해서 구한 재료의 Fourier 계수는 그대로지만, NVF 계산이 달라져 최종 반사율·투과율은 달라질 수 있습니다.

`linear/circular`는 여기서는 **입사광 편광이 아닙니다.** 원편광 입사에 linear NVF를 사용할 수 있고 선편광 입사에 circular NVF를 사용할 수 있습니다. circular 구성은 장 연장의 방향이 불명확한 곳을 다루는 다른 선택이지만, 언제나 더 정확하거나 더 빠르다는 뜻은 아닙니다. 래스터와 도형의 구성 세부도 같지 않습니다.

기본 도형 NVF 격자는 `max(1024, 4*Nx+1, 4*Ny+1)`입니다. 계산을 빨리 하려고 줄일 때는 관심 출력과 기울기 모두의 수렴을 확인해야 합니다.

## NVF를 선택하고 계산하기

`rcwa.Problem`에 구조와 빛을 적고 `rcwa.Numerics`에서 `factorization="nvf"`를 지정합니다. `rcwa.run(problem, numerics=numerics)`이 NVF 행렬 구성부터 반사·투과 계산까지 실행합니다.

다음은 반지름 160 nm(지름 320 nm)인 원을 포함한 두께 180 nm의 층을 계산하는 예입니다. 길이 단위는 nm이며, `Material(epsilon=4.)`와 `bottom=Material(epsilon=2.25)`는 각각 도형과 아래 매질의 상대유전율입니다.

```python
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length="nm"),
    stack=rcwa.Stack(
        period=(700., 700.),
        top=Material(epsilon=1.), bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(
            thickness=180., background=Material(epsilon=1.),
            regions=(rcwa.Region(geometry=rcwa.Circle(radius=160.),
                                 material=Material(epsilon=4.)),)),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(
        rcwa.PlaneWave(side="top", theta=0.2, phi=0.3, polarization=(0., 1.)),)))
numerics = rcwa.Numerics(
    orders=(2, 2),
    fourier_coefficients="integrated",
    factorization="nvf",
    nvf_projector="linear",
    shape_nvf_grid_size=128,
)
result = rcwa.run(problem, numerics=numerics)
print("전체 반사율:", result.R.sum())
print("전체 투과율:", result.T.sum())
```

도형 입력에서는 `integrated`와 `nvf`가 기본값이지만, 위 예제에는 어떤 방법을 사용하는지 보이도록 적었습니다. `polarization=(0., 1.)`은 p 편광 입사입니다. `result.R`과 `result.T`는 회절차수별 반사율·투과율 배열이며, 각 배열을 합하면 전체 반사율·투과율을 얻습니다.

## 계산 결과의 수렴 확인하기

수렴 확인은 **계산 설정을 더 정밀하게 바꾸어도 관심 결과가 충분히 안정적인지 확인하는 과정**입니다. 위 예제의 Fourier 차수와 NVF 격자 크기는 실행을 위한 시작값입니다. 필요한 정확도에 따라 다음 순서로 비교합니다.

1. `orders`를 고정하고 `shape_nvf_grid_size`를 128, 256, 512처럼 늘려 관심 반사율·투과율의 변화를 확인합니다.
2. NVF 격자 크기를 고정하고 `orders`를 `(2, 2)`, `(3, 3)`, `(4, 4)`처럼 늘려 비교합니다.
3. 두 설정을 함께 더 높여도 결과가 필요한 허용 오차 안에 있는지 확인합니다. 미분값을 사용하는 최적화라면 미분값도 함께 비교합니다.

위 예제는 모든 재료가 무손실이므로 전체 반사율과 투과율의 합이 1에 가까운지도 확인할 수 있습니다. 다만 에너지 보존을 만족해도 차수나 격자 크기에 따라 개별 반사율·투과율은 달라질 수 있으므로, 수렴 비교도 필요합니다.

법선장과 투영행렬을 더 자세히 공부하려면 [NVF 원리·구현·사용법 교재](../nvf/index.md)를 참고합니다. 설정을 바꾸며 결과를 비교한 예는 [NVF 수렴 확인 예제](../nvf/convergence.md)에 있습니다.
