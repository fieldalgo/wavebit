# 20. 코드 지도와 다음 공부

:::{note}
다운로드 가능한 교육 스크립트는 현재 물리 Problem/Job API를 사용합니다. 일부 설명 코드와 그림에는 이전 API 표기가 남아 있으므로 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 함께 참고하세요. 과거 측정 파일과 원본 출처는 보존하며, 이전한 스크립트의 검증과 구분합니다.
:::

현재 폴더 구조를 공부할 때는 파일 크기 순서보다 **데이터가 흘러가는 순서**로 읽는 것이 이해하기 쉽습니다. 이 지도는 FDTD 합류 전 작업본에 대한 설명이며 최종 통합 구조의 설계안은 아닙니다.

```{figure} ../../figures/learn/learn/20-code-map.svg
:alt: 공개 API에서 형상·행렬·모드·층 연결·결과로 이어지는 소스 코드 지도
:figclass: learning-figure

그림 20-1. 공개 API에서 형상·행렬·모드·층 연결·결과로 이어지는 소스 코드 지도
```

| 먼저 읽을 곳 | 이해할 질문 |
| --- | --- |
| [src/wavebit/__init__.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/__init__.py), [src/wavebit/compat/legacy.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/compat/legacy.py) | 공개 API는 무엇이며 어떤 백엔드를 만드는가 |
| [src/wavebit/_common/periodic.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/_common/periodic.py), [materials/](https://github.com/fieldalgo/wavebit/tree/develop/src/wavebit/materials/), [compat/_jobs_v1.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/compat/_jobs_v1.py) | 장면·재료·빛·계산 명세를 어떻게 표현하는가 |
| [src/wavebit/lattice.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/lattice.py), [geometry.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/geometry.py) | 셀 좌표와 고정할 기하학 정보는 무엇인가 |
| [src/wavebit/rcwa/_engine/parametric.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/parametric.py), [boundary.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/boundary.py) | 도형 Fourier 계수와 겹침 경계를 어떻게 처리하는가 |
| [src/wavebit/rcwa/_engine/factorization.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/factorization.py), [normal_methods.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/normal_methods.py) | 재료와 장의 곱이 어떤 행렬로 조립되는가 |
| [src/wavebit/rcwa/_engine/engine.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/engine.py) | 입력에서 실제 계산 경로·상태를 어떻게 연결하는가 |
| [src/wavebit/rcwa/_engine/solve_1d.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/solve_1d.py), [solve_2d.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/solve_2d.py), [solve_2d_first_order.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/solve_2d_first_order.py) | 모드 방정식과 층 계산은 무엇인가 |
| [src/wavebit/rcwa/_engine/scattering.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/scattering.py), [fields.py](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/fields.py) | 산란행렬과 공간 장을 어떻게 얻는가 |
| [src/wavebit/rcwa/_backends/](https://github.com/fieldalgo/wavebit/tree/develop/src/wavebit/rcwa/_backends/) | NumPy/JAX/PyTorch가 공통 물리에 어떤 계산 도구를 제공하는가 |
| [src/wavebit/rcwa/_engine/conforming/](https://github.com/fieldalgo/wavebit/tree/develop/src/wavebit/rcwa/_engine/conforming/) | 별도 NVF 구성·보조 계열은 일반 solve와 어떻게 구분되는가 |

모든 `_core` 파일을 사용자가 직접 import할 필요는 없습니다. 일반 프로그램은 공개 `wavebit` API로 작성하고, 내부를 이해할 때 이 지도를 사용합니다. 예전 `on_numpy`, `on_jax`, `on_torch`는 현재 소스 어댑터 폴더명이 아닙니다.

## 고급 기능을 찾아가는 입구

| 공개 API 묶음 | 어떤 단계인가 | 이어 읽을 문서 |
| --- | --- | --- |
| `ConvMatrices`, `solve_conv` | 사용자가 준비한 구성 행렬로 모드·층 계산 진행 | [수치 설정과 energy closure](../../guide/rcwa/numerics.md) |
| `ConformingNVF` | 지원되는 사각형 배열의 별도 구성 행렬 생성 | [직접 사용 안내](../../guide/rcwa/conforming-nvf.md) |
| `AffineProfileNVF` | 별도의 affine profile 구성 계열 | [범위와 예제](../../guide/rcwa/affine-profile-nvf.md) |
| `CartesianRectangleNVF`, `CartesianRectanglePairNVF` | 한 사각형·사각형 쌍에 특화된 구성 계열 | [한 사각형](../../guide/rcwa/cartesian-rectangle-nvf.md), [사각형 쌍](../../guide/rcwa/cartesian-rectangle-pair-nvf.md) |
| `CartesianPairOpeningPlan`, `CartesianPairTransmissionGalerkinGapCell` | 간격과 접촉을 다루는 별도 실험 구성 | [gap 계열의 범위](../../guide/rcwa/cartesian-pair-gap.md) |
| `ScatteringMatrix`, `Port` | SMM의 양쪽 입·출력 관계와 연결 | [이전 인터페이스 API](../../migration/legacy-api.md#results-and-state) |
| `Formulation` | 2차 정식화(`second_order`, 옛 이름 2ff)와 1차 정식화(`first_order`, 옛 이름 4ff) 중 어느 쪽으로 풀지 고르고 확인; 일반 생성자 인자가 아닌 속성 | [1차 정식화](../../theory/rcwa/first-order-formulation.md) |
| Bessel J1 구현 선택 함수 | 특수함수 구현의 고급 진단 | [Bessel J1](../../validation/rcwa/bessel-j1.md) |

이 묶음들은 모두 같은 지원 범위를 갖지 않습니다. 각 안내의 형상·재료·백엔드 제약과 미해결 범위를 먼저 읽고, 일반 `solve()`가 내부에서 자동으로 선택한다고 가정하지 않습니다.

## 알아두면 문서가 쉬워지는 단어

| 단어 | 이 책에서의 뜻 |
| --- | --- |
| 장면(scene) | 구조·재료·격자·빛을 묶은 물리 문제 |
| 차수(order) | 반복 구조의 공간 주파수 번호 m,n |
| 모드(mode) | 한 층의 결합 방정식이 허용하는 전파·감쇠 상태 |
| convolution matrix | Fourier 계수 사이의 곱셈을 표현한 행렬 |
| factorization | 불연속 함수의 곱을 유한 행렬에서 다루는 규칙 |
| phasor | 시간 진동의 크기와 위상을 담은 복소 진폭 |
| 포트(port) | 바깥 매질과 파동 기저를 포함한 입·출력 접속 정보 |
| closure | 선택한 근사 공간에서 구성 관계를 닫아 행렬로 만드는 단계 |
| 수렴(convergence) | 해상도·차수 변화에 관심 출력이 충분히 안정되는지 확인하는 과정 |
| 자동미분(autodiff) | 실행한 연산들의 미분을 연결해 입력 민감도를 구하는 방법 |

## 직접 답해보기

1. 첫 예제에서 T00보다 전체 T가 큰 이유는? **±1차 투과 전력도 있기 때문**입니다. 합산할 출력이 설계 목표와 맞는지 확인합니다.
2. 원 지름을 바꾸는 것과 Fourier 차수를 올리는 것은 같은가? **전자는 물리 설계, 후자는 그 설계의 수치 근사**를 바꿉니다.
3. `sample_x`를 두 배로 늘리면 RCWA가 더 높은 Fourier 차수를 푸는가? **아닙니다.** 같은 해를 더 많은 위치에서 재구성합니다.
4. 원편광 빛을 쓰려면 `nvf_projector='circular'`로 설정해야 하는가? **아닙니다.** 입사 편광은 `polarization` 또는 `psi/chi`로 정합니다.
5. JAX와 중앙차분이 일치하면 접촉 형상의 물리 해도 수렴했는가? **그 결론은 나오지 않습니다.** 같은 이산 계산의 미분 확인과 물리 수렴은 다릅니다.

## 이 책의 근거와 남겨 둔 범위

규약은 [conventions.md](../../reference/common/conventions.md), 사용자 인자는 [keywords.md](../../reference/keywords.md), 기능 제약은 현재 코드와 매니페스트를 기준으로 확인했습니다. 기존 문서에서 일부 표현이 현재 구현과 다를 때에는 실제 구현과 공식 규약을 우선했습니다. 예를 들어 현재 ETM의 scattering 접근은 예외이며, 첫 래스터 행은 셀의 −y 쪽입니다.

[빌드 기준과 출처](../../development/learning-provenance.md), {download}`실행 결과 <../../validation/learning-records/learn-measurements.json>`, [검토 범위](../../development/learning-provenance.md)를 함께 보관합니다. 이번 작업은 교재의 9개 짧은 예제를 확인한 것이며, 중단한 전체 회귀·설치 검사를 재개하지 않았습니다. FDTD 통합, 새로운 전체 폴더 구조 평가, 공개 릴리스 검증은 이 교재가 대신하지 않습니다.

Wavebit 코드의 라이선스와 고지사항은 저장소의 [LICENSE](https://github.com/fieldalgo/wavebit/blob/develop/LICENSE), [LICENSE.meent](https://github.com/fieldalgo/wavebit/blob/develop/LICENSE.meent), [LICENSE.fmmax](https://github.com/fieldalgo/wavebit/blob/develop/LICENSE.fmmax)와 관련 고지를 확인합니다. 이 교재의 도식과 설명은 새로 작성했고 외부 라이브러리 소스를 옮겨 넣지 않았습니다. 외부 프로그램의 결과·표·소스를 재사용할 때에는 각각의 출처와 이용 조건을 별도로 확인해야 합니다.
