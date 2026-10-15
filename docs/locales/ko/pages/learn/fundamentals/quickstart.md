# 04. 설치하고 첫 계산을 끝까지 읽어보기

:::{note}
다운로드 가능한 교육 스크립트는 현재 물리 Problem/Job API를 사용합니다. 일부 설명 코드와 그림에는 이전 API 표기가 남아 있으므로 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 함께 참고하세요. 과거 측정 파일과 원본 출처는 보존하며, 이전한 스크립트의 검증과 구분합니다.
:::

현재 소스의 Python 요구사항은 **3.12 이상**입니다. 아래는 Wavebit 소스 폴더에서 설치하는 방법입니다. 공개된 알파 버전은 PyPI에서도 설치할 수 있습니다(`pip install wavebit`). [설치 안내](../../getting-started/installation.md)를 참고하세요. 사용하지 않는 백엔드를 모두 설치할 필요는 없습니다.

```sh
python -m pip install .
# JAX 예제를 사용할 환경에서는
python -m pip install '.[jax]'
# PyTorch 예제를 사용할 환경에서는
python -m pip install '.[pytorch]'
```

이 책을 실행 확인할 때는 기존 환경을 사용했습니다. NumPy 2.5.1, SciPy 1.18.0, JAX 0.11.0, PyTorch 2.13.0입니다. 환경과 각 실행 기록은 {download}`measurements.json <../../validation/learning-records/learn-measurements.json>`에 들어 있습니다. 새 외부 코드나 의존성을 이 교재에 추가하지 않았습니다.

```{figure} ../../figures/learn/learn/04-first-grating.svg
:alt: 첫 예제의 700 nm 주기, 300 nm 높이의 반반 격자와 입사광
:figclass: learning-figure

그림 4-1. 첫 예제의 700 nm 주기, 300 nm 높이의 반반 격자와 입사광
```

공기와 `n=3.5` 재료가 가로로 반씩 차지하는 격자입니다. y 방향으로는 변하지 않습니다. 아래는 `n=1.5` 유리, 위는 공기입니다. 한 줄의 4픽셀은 왼쪽 두 칸과 오른쪽 두 칸이라는 **정확한 반반 형상**을 나타냅니다. 이 단순한 형상에서도 장을 표현하는 Fourier 차수는 별도로 필요합니다.

실행 파일: {download}`01_first_grating.py <../../../examples/rcwa/learning/learn/01_first_grating.py>`

그림과 코드 모두 **상대 유전율을 그대로** 표시합니다. 공기는 `1.`, 패턴 재료는 `12.25`, 아래쪽 유리 매질은 `2.25`입니다.

```{literalinclude} ../../../examples/rcwa/learning/learn/01_first_grating.py
:language: python
:linenos:
```

| 실제 실행값 | 값 |
| --- | --- |
| 전체 R | 0.205331628258 |
| 전체 T | 0.794668371742 |
| 0차 T | 0.353101123079 |
| 결과 배열 크기 (`result.T.shape`) | (1, 1, 21) |

입사 에너지를 1로 보았을 때 약 20.53%가 반사되고 79.47%가 투과합니다. 0차 투과는 약 35.31%이며, 나머지 투과는 다른 전파 가능한 차수로 나갑니다. **전체 투과와 0차 투과는 다릅니다.** 이 예제의 `R+T≈1`은 무손실 에너지 균형을 확인하지만, 이것만으로 Fourier 차수가 충분하다는 결론은 낼 수 없습니다.
