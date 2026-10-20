# 07. 첫 계산: 원형 구조의 NVF 반사·투과

:::{note}
다운로드 가능한 교육 스크립트는 현재 물리 Problem/Job API를 사용합니다. 일부 설명 코드와 그림에는 이전 API 표기가 남아 있으므로 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 함께 참고하세요. 과거 측정 파일과 원본 출처는 보존하며, 이전한 스크립트의 검증과 구분합니다.
:::

예제는 Wavebit의 현재 소스 checkout을 설치한 환경에서 실행합니다. 이 기준 버전의 설치 안내는 소스에서 `python -m pip install .`을 사용하는 방식입니다. Wavebit는 PyPI에도 배포되어 있습니다(`pip install wavebit`). 릴리스는 지금 읽는 checkout과 다를 수 있으니 `wavebit.__version__`을 비교하세요. Python과 의존성 조건은 현재 checkout의 [설치 안내](../../getting-started/installation.md)를 따릅니다.

```bash
# 먼저 Wavebit 소스 폴더에서 설치
python -m pip install .

# 그다음 같은 폴더에서 실행
python examples/rcwa/learning/nvf/01_shape_nvf.py
```

아래는 {download}`예제 1 전체 파일 <../../../examples/rcwa/learning/nvf/01_shape_nvf.py>`과 같은 코드입니다. 길이는 모두 nm로 해석하며, 함수 이름 `problem`, `run`와 `summarize`는 설명을 위해 이 예제에서 정의한 함수입니다.

```{literalinclude} ../../../examples/rcwa/learning/nvf/01_shape_nvf.py
:language: python
:linenos:
```

## 반드시 확인할 입력 규약

**재료 숫자는 굴절률 n이 아니라 상대유전율 ε입니다.** 예제의 `Material(epsilon=4.)`는 비자성 재료에서 굴절률 n=2에 해당합니다. `bottom=Material(epsilon=2.25)`는 n=1.5입니다. 굴절률을 가지고 있다면 `wavebit.eps_from_refractive_index(n)`으로 변환할 수 있습니다. 시간 의존성은 `exp(−iωt)`이므로 수동 손실 재료는 통상 `Im(ε)>0` 규약입니다.

`Circle(radius=160.)`은 **반지름** 160 nm를 지정합니다. `center=(0., 0.)`은 셀 중심이며 `Stack.layers`는 바닥부터 위로 나열합니다. `top`과 `bottom`은 그 위아래의 반무한 매질입니다. `wavelength`, `period`, `thickness`, 도형 크기는 같은 길이 단위를 쓰고, 각도는 radian입니다.

`theta=.2`, `phi=.3`은 입사 방향입니다. `PlaneWave(polarization="p")`와 `"s"`를 각각 계산하여 두 순수 편광을 명시적으로 비교합니다. 각 결과의 `R`, `T`는 해당 입사의 전력 효율입니다.

## 실제 실행 결과

기준 코드의 NumPy CPU, double precision으로 실행한 값입니다. 전체 출력과 환경은 {download}`measurements.json <../../validation/learning-records/nvf-measurements.json>`에 있습니다. 자릿수가 많다고 물리적 정확도가 그만큼 확보된 것은 아닙니다.

| 입사 | 전체 R | 전체 T | 에너지 잔차 abs(R+T−1) |
| --- | --- | --- | --- |
| p | 0.002869503084 | 0.997130496916 | 1.44×10⁻¹⁵ |
| s | 0.005397410043 | 0.994602589957 | 8.88×10⁻¹⁶ |

`R.sum()`과 `T.sum()`은 모든 회절차수의 전력 효율을 합한 값입니다. 전기장 진폭을 단순히 제곱하거나 더한 값이 아닙니다. 무손실 예제이므로 R+T≈1을 확인할 수 있지만, 이것만으로 Fourier 차수의 정확도를 확정하지는 않습니다.

## x/y 차수와 배열 순서

```{figure} ../../figures/learn/nvf/09-orders.svg
:alt: 차수는 x 먼저, 결과 배열은 y 행과 x 열
:figclass: learning-figure

그림 7-1. 차수는 x 먼저, 결과 배열은 y 행과 x 열
```

`Numerics(orders=(Mu, Mv))`에서 두 격자 벡터 방향(이 직사각형 셀에서는 x와 y)의 반폭은 서로 독립이며 항상 쌍으로 씁니다. 현재 결과의 축은 `(파장, v, u)`이므로 `(2,1)`이면 한 파장의 shape는 `(1,3,5)`입니다. `result.transmission.orders_u`, `.orders_v`로 순서를 확인하고 `result.transmission.order(m,n).efficiency`로 특정 차수의 투과 효율을 읽습니다. 모든 유지 모드가 멀리까지 전파하는 것은 아니며, 감쇠하는 모드도 근접장과 경계조건 표현에 필요합니다.

```{raw} html
<section class="exercise" aria-labelledby="orders-title"><h3 id="orders-title">직접 바꿔 보기 · 내가 유지하는 Fourier 모드는?</h3><p class="explain">선택된 정수 모드만 보여 줍니다. 점은 편광이 아니라 (m,n) Fourier 라벨입니다.</p><div class="controls"><label for="nx">Nx <output id="nx-value" for="nx">2</output><input id="nx" type="range" min="0" max="4" step="1" value="2"></label><label for="ny">Ny <output id="ny-value" for="ny">1</output><input id="ny" type="range" min="0" max="4" step="1" value="1"></label></div><svg class="mini-svg" viewBox="0 0 400 340" role="img" aria-label="선택된 Fourier 모드의 격자"><rect x="32" y="20" width="312" height="280" fill="white" stroke="#b3cbc2"/><g id="mode-dots"></g><text x="370" y="172" font-size="16" fill="#182b3e">m</text><text x="189" y="14" font-size="16" fill="#182b3e">n</text><text x="187" y="325" font-size="14" fill="#526779">0</text></svg><div class="readout" id="orders-result" aria-live="polite"></div><noscript><p class="no-js">본문의 식에서 (2,1)의 15개 모드를 확인할 수 있습니다.</p></noscript></section>
```
