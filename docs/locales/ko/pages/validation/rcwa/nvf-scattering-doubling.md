# NVF 연구 모델을 위한 고유벡터 없는 전파

`studies/rcwa/nvf_scattering_doubling/nvf_scattering_doubling.py`는 짧은 슬라이스 Maxwell 전달과 그 뒤를
잇는 산란 행렬 배가(doubling)를 구현합니다. 기록된 회전 직사각형 재료 미분 픽스처(fixture)에서, 이
방법은 당시 존재하던 정규화 고유벡터 미분의 약 6 ppm s 편광 편향을 없앱니다. 첫 번째 독립 NumPy 할선과의
새 차이는 약 0.028 ppm입니다. 원래의 2단계 미분 기준은 허용 오차를 완화하지 않고 통과합니다.

이 구현은 구성 행렬과 함께 직접 사용하거나, 다항식 몸체 연구 모델의 명시적 대안인
`ScatteringDoublingGapModel`을 통해 사용할 수 있습니다. 셀 방정식, 실행 중(live) 재료 단계화와 NVF Gram
닫힘은 바뀌지 않았습니다. 제품 NVF 디스패치와 그 기존 고유값 솔버도 바뀌지 않았습니다. 이것은 새 경로의
전파 미분 문제를 고치는 것이며, 예전 푸리에 보조 필드에 따로 있는 s 접촉 오차를 고치지는 않습니다.

**제품 미분 갱신(2026-09-13).** 이제 모드 미분은 고유값 간격을 정규화하고, Lorentzian 분모 가드를
`(matrix_dimension * machine_epsilon)**2`로 설정합니다. 예전의 `1e-12` 상대 분모와 고정 절대 하한은 수치적으로
구별되는 모드를 감쇠시켰습니다. 칠한 층의 폭 미분에서 예전 가드는 `-0.71919175`를, 수정한 가드는
`-0.71978335`를 주며, 후자는 독립 NumPy 할선과 일치합니다. `A@A`를 eig 기반으로 표현한 것도 가깝지만 구별되는
모드에 대해 연산자 규모 16자릿수(sixteen decades)에 걸쳐 JAX와 Torch에서 직접 곱의 비대각 미분과 일치합니다.
조화 에너지 모델의 예전 s 편광 예상 실패(expected failure)는 이제 바뀌지 않은 2단계 기준을 통과하며 일반 회귀
테스트가 되었습니다. 아래 측정은 이전 비교를 기술합니다. 산란 배가는 여전히 고유벡터 미분 자체를 피합니다.
유한한 역수 가드는 반복되거나 결함 있는 고유값에서 일반적으로 정확한 미분이 아닙니다.

## 모드 미분 없는 전파

기존의 규모 조정된 전기/자기 좌표에서, 균질 층의 생성자를 1차 Maxwell 인자 L과 R로 씁니다.

```text
F(tau) = exp(-tau [[0, L], [R, 0]])
tau = (2 pi / wavelength) thickness / 2^doublings
P = tau^2 L R
Q = tau^2 R L
```

짧은 슬라이스에서는 짝수 급수와 홀수 급수를 계산합니다.

```text
C(P) = sum P^k / (2k)!
S(P) = sum P^k / (2k+1)!
F00 = C(P)       F01 = -tau L S(Q)
F10 = -tau R S(P)  F11 = C(Q).
```

LR과 RL 모두, 차수 k^2의 결과를 되찾기 위해 차수 k^4의 항을 빼는 일을 피하도록 전개한 식을 사용합니다.
테스트는 이 식들을 복소 에르미트 교차 블록을 가진 1차 곱과 비교합니다. 이 방법은 행렬 제곱근을 만들지 않고,
모드 가지를 고르지 않으며, 고유값이나 고유값 간격으로 나누지 않습니다. 따라서 0인 층 전파 고유값도 같은
전해석(entire) 급수로 계산됩니다.

기준 어드미턴스는 `Y = i [[0,-I],[I,0]]`이며 `Y^2=I`입니다. `E=a+b`, `H=Y(a-b)`로 쓰면 플럭스가
`|a|^2-|b|^2`에 비례하는 기준 포트를 얻습니다. 짧은 전달만 산란 좌표로 변환한 다음, 선형 풀이를 사용해 이를
자기 자신과 `doublings`번 연결합니다. 증가하는 소멸파 전달 인자는 층 전체 두께에 걸쳐 결코 만들어지지
않습니다. 마지막으로 이 기준 포트를 물리적 외부 p/s 파동에 맞추고 출사 플럭스를 계산합니다. 외부 정규화는
기존 연구 계측 도구와 같습니다.

NumPy는 SciPy 풀이를 사용하고, JAX는 일반 행렬 곱과 풀이를 float64로 미분합니다. 이 경로에는 사용자 정의
고유벡터 미분, 고정된 재료 필드, 멈춘 기하 미분이 없습니다.

## 수치 검사와 한계

유지하는 Taylor 항이 K개일 때, alpha를 P와 Q의 무한대 노름의 최댓값이라고 합니다. 꼬리 추정은 첫 번째로
생략된 짝수 항과 비율 `alpha / ((2K+3)(2K+4))`의 기하 급수 상한을 사용하며, 인자
`1 + |tau| max(||L||,||R||)`를 통해 홀수 블록도 포괄합니다.

비율은 1보다 작아야 하고, 추정된 국소 꼬리에 슬라이스 수를 곱한 값은 1e-13보다 작아야 합니다. 분해되지
않은 슬라이스는 NumPy에서 예외를 일으키고 JIT에서는 NaN을 냅니다. 배가 횟수나 항 수를 늘리는 것은 명시적으로
해야 합니다. 이것은 보수적인 국소 급수 검사이며, 임의로 조건이 나쁜 경계 정합을 거치는 전역 오차 상한이
증명된 것은 아닙니다. 별도의 슬라이스 세분화와 독립적인 광학 비교가 여전히 필요합니다.

현재의 무손실 계측 도구는 유한한 T/R과 1e-9 미만의 전력 불균형도 요구합니다. 이 가드로 광학적 절단 정확도를
보증할 수는 없습니다. 흡수 재료는 이 무손실 전력 계약의 범위 밖입니다. 픽스처는 고정된 주기, 파장, 외부
매질을 유지합니다. 두께와 실행 중(live) 극각/방위각 입사각은 `doubling_powers`나 모델의 투과 메서드에 넘길 수
있습니다. [자세 확장](nvf-posed-gap.md)은 수직 입사를 지나는 편광 좌표계와 남아 있는 외부 스침 제외를
문서화합니다. `doubling_response`는 복소 출사 진폭도 반환합니다. 행렬 수준 함수에서 차수는 스칼라이거나
방향성을 가질 수 있지만, 상속된 셀 어댑터는 현재 스칼라 차수만 받습니다.

독립 균일 슬랩 대조는 두께 30000 nm와 방향성 차수 `(2,3)`까지 확장됩니다. 이후의
[거울 섹터 통합](nvf-mirror-doubling.md)은 NumPy와 JAX를 모두 지원하고, 차수 24까지 값을 비교하며, 배가
방법 안에서 더 낮은 순방향/미분 비용을 측정합니다. 공유 커널은 균질 층의 z 대칭도 사용해 중복되는 산란
블록을 피합니다. 이것은 모드 계측 도구의 차수 40에서 성능 검증을 받지 않았고, 다층 제품 기하로
일반화되지 않았으며, 바깥 광학 수렴을 선언하는 데 사용되지도 않았습니다.

## 미분 측정과 물리적 대조

원래의 재료 편향 픽스처는 차수 2, 푸리에 보조 차수 7, 직사각형 폭 440 nm, 높이 270 nm, 각도 .31 라디안,
중심 `(17,-21)` nm, 상대 유전율 6.25를 사용합니다. 바로 그 보조 모델을 재사용하면 전파의 변화만 분리할 수
있습니다.

| 양 | 새 미분 | 독립 모드 할선, 간격 1e-5 |
| --- | ---: | ---: |
| dT_p / d epsilon | 0.00846134966851 | 0.00846134614041 |
| dT_s / d epsilon | 0.00619272391621 | 0.00619272374491 |

이 간격에서 s 상대 차이는 약 0.028 ppm입니다. 할선에는 반올림 오차가 남아 있으므로, 이것은 절대 미분 오차의
증명서가 아닙니다. 폭, 높이, 각도, 재료 미분은 두 원래 간격 크기 모두에서 원래의 `atol=2e-9, rtol=3e-6`
비교를 통과합니다. 예전 정규화 경로의 예상 실패는 위의 제품 미분 갱신으로 원래 기준을 통과하게 될 때까지
유지했습니다.

추가 테스트는 현재의 다항식 몸체 NVF 어댑터를 차수 2, 행 4, 몸체 차수 4, 틈 차수 6, 가장자리 비율 1e-5로
검사합니다.

- JIT 틈, 움직이는 실제 이음매, 두 재료 미분을 독립 모드 NumPy 할선과 비교하며, 같은 재료와 대비 0인 경우를
  포함합니다.
- 단일 재료 JIT 접촉 미분 두 가지를, 따로 인수분해한 데카르트 단측 접촉 극한과 비교하며 기존의 nm당 1e-7
  기준을 유지합니다. 혼합 재료 접촉에는 [국소 플럭스 연구](nvf-local-flux.md)에 기록된 대로 추가 셀 세분화가
  필요합니다.
- 평행 이동한 분할/병합 직사각형의 동등성, 그리고 평행 이동 미분이 0인지 확인합니다.
- 정확히 스치는 층을 포함해, 균일 모드와 반복 모드의 재료 미분을 독립 슬랩 공식과 비교합니다.

유전율 12.25인 접촉 대조에서 새 JIT 틈 미분은 p와 s에 대해 nm당 0.0009240734238과 0.0233077786485입니다.
따로 인수분해하고 2단계 외삽한 기준은 0.0009240733023과 0.0233077783351을 줍니다. nm당 1.21e-10과 3.13e-10의
절대 불일치는 명시한 이 저차수 픽스처에서 바뀌지 않은 1e-7 기준보다 작습니다.

정확한 층 스침에서는 필드 값을 독립적인 Jordan 극한 2×2 전달과 비교합니다. 그 재료 미분에 대해 간격 1e-5의
일반 Airy 할선은 눈에 보이는 O(h^2) 오차를 가집니다. 간격 1e-5, 5e-6, 2.5e-6에서 p 값은 2.1351583380,
2.1351547468, 2.1351538493입니다. 연속한 두 쌍으로부터의 Richardson 외삽은 바뀌지 않은 허용 오차 안에서 JIT
값 2.1351535499와 일치합니다. 테스트는 미분을 비교하기 전에 두 외삽 기준이 서로 일치하는지 검사합니다.

추가 대조는 틈 0, 17, 400 nm에서 혼합 재료 p/s 전력을 이전 모드 풀이와 비교하면서, 슬라이스 수를 독립적으로
바꿉니다. 이 대조는 전파만 분리합니다. 셀 수렴 연구나 바깥 푸리에 수렴 연구를 대신하지 않습니다.

## 사용과 재현

```python
from studies.rcwa.nvf_scattering_doubling.nvf_scattering_doubling import ScatteringDoublingGapModel

model = ScatteringDoublingGapModel(
    2, rows=4, body_degree=4, gap_degree=6, edge_fraction=1e-5,
)
result = model.transmission(
    17., dict(seams=[31.], values=[4., 9.]), polarization='both',
)
# result['T'] and result['R'] are vectors in p, s order.
# Pass xp=jax.numpy for live JIT derivatives, with JAX x64 enabled.
```

상속된 `shape_transmission` 인터페이스는 같은 방식으로 순서를 정하고 평행 이동한 수평 직사각형 타일링을
지원합니다. 일반적인 CSG 어댑터가 되지는 않습니다. `polarization='p'` 또는 `'s'`는 예전 연구 인터페이스와
마찬가지로 스칼라를 반환합니다. `doubling_powers(order, slots, xp)`를 직접 사용하면 무손실 계약 아래에서
호환되는 임의의 유한 구성 행렬을 받습니다.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_scattering_doubling.nvf_scattering_doubling --jax
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_scattering_doubling.nvf_scattering_doubling --mirror --jax
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_scattering_doubling.nvf_scattering_doubling --gap 0 --seams --values 12.25 --jax
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_scattering_doubling.py
```

전체 행렬 CLI는 셀 잔차, 에르미트성, 국소 Taylor 꼬리 추정, 전력 균형, 그리고 선택적으로 JIT 틈/이음매/재료
Jacobian을 보고합니다. `--mirror`를 쓰면 같은 내부 가드가 적용되며, 반환되는 진단에는 셀 정보와 전력 균형이
담깁니다. 고차수 거울 CLI의 `--progress` 출력은 섹터별 Taylor 상한도 보여 줍니다. 원시 출력은
`studies/rcwa/nvf_scattering_doubling/records/nvf_scattering_doubling_material.log`와
`nvf_scattering_doubling_gap.log`에 있습니다.

2026-09-08 검증: **18개 통과**. 별도의 연구 구현을 근거로 제품의 실패를 통과로 바꿔 표기하지 않습니다.
