# 10. 정확도와 비용을 결정하는 Fourier 차수

> **과거 API 재현:** 이 페이지에서 `wavebit.compat`를 쓰는 코드·설명은 현재 설치의 API가 아닙니다.
> 해당 이름은 [퇴역 전 소스 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)에 있습니다.
> 측정 재현에는 본문에 적힌 원래 커밋·환경을 우선 사용하세요. 원본 측정값은 갱신하지 않았습니다.


:::{note}
다운로드 가능한 교육 스크립트는 현재 물리 Problem/Job API를 사용합니다. 일부 설명 코드와 그림에는 이전 API 표기가 남아 있으므로 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 함께 참고하세요. 과거 측정 파일과 원본 출처는 보존하며, 이전한 스크립트의 검증과 구분합니다.
:::

Fourier 차수는 계산에 남길 공간 주파수 성분의 범위를 정합니다. 먼저 차수와 행렬 크기의 관계를 살펴보고, 형상 해상도·계수 계산 방법·Li와 NVF가 각각 무엇을 바꾸는지 구분합니다.

## Fourier 차수와 행렬 크기

`truncation_order`에 지정하는 x·y 방향의 최대 차수를 각각 {math}`N_x`, {math}`N_y`라고 씁니다. 기본 0차 단일 입사에서는 {math}`m=-N_x,\ldots,N_x`, {math}`n=-N_y,\ldots,N_y`인 성분을 계산에 남깁니다. 두 차수의 조합으로 만들어지는 **전체 Fourier 성분 수**를 {math}`H`라고 하면 다음과 같습니다.

```{math}
H=(2N_x+1)(2N_y+1).
```

예를 들어 `truncation_order=dict(x=2, y=1)`이면 {math}`N_x=2`, {math}`N_y=1`입니다. x 방향 차수는 −2, −1, 0, 1, 2로 **5개**, y 방향 차수는 −1, 0, 1로 **3개**이므로 전체 성분 수는 {math}`H=5\times3=15`입니다.

다중 입사 차수 설정은 실제 차수 창을 확장하거나 옮길 수 있으므로 그때는 `result.orders_u`, `orders_v`를 직접 읽습니다.

```{figure} ../../figures/learn/learn/10-orders.svg
:alt: Fourier 차수 격자와 결과 배열의 행·열 대응
:figclass: learning-figure

그림 10-1. Fourier 차수 격자와 결과 배열의 행·열 대응
```

```{raw} html
<section class="exercise" aria-labelledby="orders-title"><h3 id="orders-title">직접 바꿔보기 · 차수를 올리면 몇 개를 남길까?</h3><p class="concept">개념 조작 · 기본 0차 단일 입사 · 실제 RCWA 계산이나 시간 측정은 아닙니다.</p><div class="controls"><label for="order-x">x 방향 최대 차수 N<sub>x</sub> <output id="order-x-out" for="order-x">2</output><input id="order-x" type="range" min="0" max="10" value="2"></label><label for="order-y">y 방향 최대 차수 N<sub>y</sub> <output id="order-y-out" for="order-y">1</output><input id="order-y" type="range" min="0" max="10" value="1"></label></div><svg viewBox="0 0 400 245" role="img" aria-label="남긴 Fourier 차수의 개념 격자. 주황은 0차."><g id="order-cells"></g><text x="125" y="241">주황: (m, n) = (0, 0)</text></svg><div class="readout" id="order-readout" aria-live="polite">전체 성분 수 H = 15개 · 결과 배열 크기 (행, 열) = (3, 5) · H² = 225</div><noscript><p>JavaScript를 켜면 차수 격자를 조작할 수 있습니다.</p></noscript></section>
```

두 방향을 모두 늘리면 차수 수가 곱으로 증가합니다. 많은 핵심 행렬의 저장량은 대략 {math}`H^2`, 조밀 고유값 계산의 연산량은 대략 {math}`H^3` 규모입니다. 실제 시간은 모드 정식화, 층 수, factorization, 캐시, 백엔드에 따라 달라집니다. 위 조작의 H²는 행렬 크기에 대한 설명이며 실제 시간 벤치마크가 아닙니다.

## 모양을 표현하는 것과 장을 표현하는 것

| 설정 | 바꾸는 것 |
| --- | --- |
| 재료 래스터의 행·열 수 | 형상 및 재료 분포를 표현하는 칸 |
| `truncation_order` | 해와 연산자에 남기는 Fourier 차수 |
| `fourier_coefficients` | 주어진 재료 분포에서 Fourier 계수를 얻는 방법 |
| `factorization` | 불연속 재료와 장의 곱을 유한 행렬로 만드는 규칙 |
| `shape_nvf_grid_size` | 도형 NVF 투영장을 표본화하는 격자 |
| 장 재구성의 `sample_x/y/z` | 이미 계산한 해를 출력할 위치 |

래스터는 한 층을 행·열의 격자로 나누고 각 칸에 재료값을 넣은 배열입니다. 등방성 재료의 유전율과 투자율은 각각 `2.25` 같은 **스칼라값 하나**로 지정할 수 있습니다. 이방성 재료를 표현하려면 각 칸의 유전율 또는 투자율을 **3×3 행렬**로 지정합니다. 행렬의 성분과 입력 방법은 {ref}`7장 <material-anisotropy>`에서 설명합니다.

**3×3 행렬은 재료값을 표현하는 형식입니다.** 등방성 유전율도 대각 성분이 모두 같은 행렬 {math}`\varepsilon_r I`로 표현할 수 있습니다. 아래 표에서는 각 칸에 스칼라값을 넣었는지, 3×3 행렬을 넣었는지에 따라 입력을 구분합니다.

래스터의 기본값인 `fourier_coefficients='sampled'`는 배열의 값에 이산 Fourier 변환을 적용해 계수를 구합니다. `'integrated'`는 각 격자 칸을 재료값이 일정한 영역으로 보고 적분합니다. 어느 방법을 쓰더라도 입력한 픽셀 분포를 계산 대상으로 삼습니다.

도형 입력은 등방성·이방성 재료 모두 `fourier_coefficients='integrated'`가 기본값입니다. 도형 영역을 적분해 Fourier 계수를 구하며, 겹치지 않는 원·타원·사각형에는 해석식을 사용합니다. 도형이 겹치면 각 재료가 남는 영역을 구분해 적분하고, 곡선이 포함된 겹침 영역에는 다각형 근사를 사용합니다.

| 입력 형식 | Fourier 계수 기본값 | factorization 기본값 |
| --- | --- | --- |
| 래스터 — 각 칸에 스칼라 재료값 | `sampled` | `li` |
| 래스터 — 각 칸에 3×3 재료 행렬 | `sampled` | `nvf` |
| 도형 — 스칼라 또는 3×3 재료 행렬 | `integrated` | `nvf` |

**각 칸에 3×3 재료 행렬을 넣은 래스터**의 패턴을 NVF로 계산할 때는 `fourier_coefficients='sampled'`를 사용합니다. Li로 계산할 때는 `'sampled'`와 `'integrated'`를 모두 지원합니다. 스칼라값을 넣은 래스터는 Li와 NVF 모두 두 계수 계산 방법을 지원합니다.

계수 계산 방법과 factorization은 서로 다른 단계입니다. 도형 영역의 적분에 해석식을 사용해도 NVF의 법선장과 보조함수는 격자에서 계산하므로, `shape_nvf_grid_size`에 대한 수렴 확인은 별도로 필요합니다. 사선 격자와 이방성 재료의 지원 범위는 아래에서 설명합니다.

## Li와 NVF는 어디에서 달라질까

전체 계산에서 두 방식이 쓰이는 위치는 {ref}`2장의 계산 흐름 그림 <factorization-flow>`을 참고합니다. 여기서는 행렬 구성 방식과 사용 조건을 비교합니다.

`li`는 격자 u·v축에 따른 직접 규칙·역규칙을 적용합니다. 현재 Wavebit의 Li 경로는 **u·v축이 실제 x·y축과 나란한 직교 격자**에서 등방성 재료와 전체 3×3 ε·μ를 가진 이방성 재료를 지원합니다. 입력 형상은 래스터 또는 축에 정렬된 사각형 도형입니다.

`nvf`는 경계의 법선 방향을 이용해 행렬을 구성합니다. 이 과정에서는 대각 성분과 비대각 성분을 함께 구성하며, 이미 완성된 Li 행렬에 몇 개의 항만 덧붙이는 흐름으로 읽지 않습니다. 법선장과 투영행렬의 의미는 [11장](nvf.md)에서 설명합니다.

**사선 격자에서도 이방성 재료를 NVF로 계산할 수 있습니다.** 이방성 재료가 포함된 패턴 구조를 NVF로 계산할 때는 `nvf_projector="linear"`를 사용합니다. Li와 NVF 모두 xz·yz·zx·zy 결합을 포함한 재료 행렬을 구성하며, 이때 계산은 세 방향을 함께 다루는 1차 정식화(`first_order`, 옛 이름 4ff)로 넘어갑니다. NVF의 Fourier 차수가 너무 낮으면 모드의 진행 방향을 구분하기 어려울 수 있습니다. 이때는 오류가 발생하며, JAX 추적 중에는 NaN으로 표시합니다. 차수와 법선장 해상도를 높여 결과의 수렴을 확인합니다. 자세한 재료 지원 범위는 [7장](materials.md)를 참고합니다.

다음은 이방성 재료의 주축을 0.3 rad 회전하고, 사각형 도형을 Li 방식으로 계산하는 예입니다. 재료의 주축만 회전하며 사각형의 변은 x·y축에 정렬되어 있습니다.

```python
from wavebit.compat import legacy as wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.8]).rotated(0.3)
solver = wavebit.rcwa(
    wavelength=730., period=(600., 600.), thickness=[173.],
    structure=[{"background": 1., "shapes": [
        {"type": "rectangle", "lx": 240., "ly": 300., "material": crystal},
    ]}],
    factorization="li", truncation_order=dict(x=3, y=3),
)
result = solver.solve()
```

`factorization="li"`를 선택하면 NVF 전용 옵션인 `nvf_projector`, `raster_normal_method`, `shape_nvf_grid_size`는 지정하지 않습니다. 위 차수는 사용 예시의 시작값이며, 관심 결과의 차수 수렴을 확인합니다.
