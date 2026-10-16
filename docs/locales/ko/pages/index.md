# Wavebit 문서

**버전:** {sub-ref}`release`

**바로가기:** [소스 저장소](https://github.com/fieldalgo/wavebit) |
[문제 제보](https://github.com/fieldalgo/wavebit/issues) |
[기여 안내](https://github.com/fieldalgo/wavebit/blob/main/CONTRIBUTING.md) |
[FieldAlgo](https://fieldalgo.com)

Wavebit은 RCWA와 FDTD로 전자기파를 계산하고, 지원되는 계산 경로에서는 JAX와 PyTorch를 이용한 자동미분을 제공합니다. 두 솔버는 물리적 재료 정의, 명시적인 단위, 변경 불가능한 문제 선언과 실행 절차를 공유합니다. 각 방법의 수치 근사와 검증된 기능 범위는 서로 다릅니다.

주기적인 적층 구조라면 [첫 RCWA 계산](getting-started/rcwa.md)부터, 공간 영역에 소스를 넣고 관측값을 기록하려면 [첫 FDTD 펄스 계산](getting-started/fdtd.md)부터 시작하세요. [학습 교재](learn/index.md)는 기초 물리를 설명하고, [이행 안내](migration/index.md)는 기존 과학 계산을 재현하는 호환 인터페이스를 구분해 안내합니다.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 시작하기
:link: getting-started/index
:link-type: doc

설치하고 솔버를 선택해 첫 계산을 실행합니다.
:::

:::{grid-item-card} 학습 교재
:link: learn/index
:link-type: doc

순서대로 개념을 익히고 실행 예제를 바꿔 봅니다.
:::

:::{grid-item-card} 사용 안내
:link: guide/index
:link-type: doc

재료, 형상, 입사 조건과 수치 설정을 정의합니다.
:::

:::{grid-item-card} API 참조
:link: reference/index
:link-type: doc

함수 인자, 허용값과 결과의 의미를 확인합니다.
:::

:::{grid-item-card} 이론과 검증
:link: theory/index
:link-type: doc

유도 과정, 측정 비교와 적용 한계를 살펴봅니다.
:::

:::{grid-item-card} 개발 안내
:link: development/index
:link-type: doc

소스를 수정하고 작업 사본을 디버깅하며 문서를 빌드합니다.
:::

::::

## 먼저 확인할 일곱 가지 규약

[공통 규약](reference/common/conventions.md)과 [현재 실행 절차](guide/common/lifecycle.md)에서 각 입력의 물리적 의미와 현재 API·호환 인터페이스의 차이를 설명합니다.

| 규약 | 의미 |
| --- | --- |
| **재료값은 상대 유전율입니다** | 굴절률을 알고 있다면 `eps_from_refractive_index`로 변환합니다. |
| **페이저는 `e^{-iωt}`를 사용합니다** | 유전율의 양의 허수부는 수동적인 전기적 손실을 뜻합니다. |
| **적층 순서는 아래에서 위입니다** | z의 원점은 맨 아래 경계면이며 위쪽으로 증가합니다. |
| **전력과 진폭을 구분합니다** | 회절 효율은 전력의 비율이고, 복소 진폭은 위상도 포함합니다. |
| **회절 배열의 첫 축은 파장입니다** | 그다음은 v 차수, u 차수이며, 차수 쌍을 적을 때는 `(u,v)` 순서입니다. |
| **주기 셀의 원점은 중심입니다** | 형상 좌표는 셀 중심을 기준으로 하며 물리적 길이 단위를 사용합니다. |
| **단위를 명시합니다** | 길이는 `Units`를 따르고 시간은 초, 변화율은 SI, 각도는 라디안, E는 V/m, H는 A/m입니다. |

## 이어서 공부하기

[기초 교재](learn/fundamentals/index.md)는 형상, 재료, 빛, 회절, 전력, 필드와 자동미분을 차례로 다룹니다. [NVF 교재](learn/nvf/index.md)는 Fourier 곱과 재료 경계면을 설명합니다. [FDTD 교재](learn/fdtd/index.md)는 소스, 흡수층, 관측값과 해상도 검증을 공개된 검증 자료에 연결합니다.

본문과 함께 실행할 수 있는 [학습 예제](learn/examples.md), 솔버별 예제와 [노트북](https://github.com/fieldalgo/wavebit/tree/main/tutorials)을 제공합니다. 기존 교재는 호환 인터페이스를 사용하는 부분을 명시하며, 당시의 수치 기록은 다시 계산하지 않고 보존했습니다.
