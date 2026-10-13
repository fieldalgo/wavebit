# 06. 입력 두 종류와 설정의 역할

:::{note}
이 교재의 실행 예제는 원래 수치 기록을 재현할 수 있도록 `wavebit.compat.legacy` 인터페이스를 명시적으로 사용합니다. 새 작업은 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 참고하세요. 기존 측정값은 이번 문서 이행에서 다시 계산하지 않았습니다.
:::

Wavebit는 재료 구조를 크게 두 방식으로 받습니다. **raster·shape는 도형의 입력 방식이고, NVF는 경계 정보를 재료 행렬에 반영하는 계산 방식입니다.** 같은 원을 표현하더라도 픽셀 이미지와 연속 도형은 실제 입력 기하가 서로 다릅니다.

```{figure} ../../figures/learn/nvf/07-inputs.svg
:alt: 픽셀로 만든 원과 circle 도형
:figclass: learning-figure

그림 6-1. 픽셀로 만든 원과 circle 도형
```

| 항목 | raster: 픽셀 배열 | shape: 도형 사전 |
| --- | --- | --- |
| 입력 | `(층 수, y 픽셀 수, x 픽셀 수)` 배열 | 층마다 `background`, `shapes` |
| 기본 Fourier 방식 | `sampled` | `integrated` |
| 기본 factorization | `li` | `nvf` |
| NVF 켜기 | `factorization='nvf'` 명시 | 기본값이며 명시해도 됨 |
| 경계 방향의 출발점 | 픽셀 재료 패턴에서 추정 | 도형 경계와 기하에서 구성 |
| 방향장 생성법 `raster_normal_method` | NVF에서 사용 | 지정하면 거부 |
| 투영 방식 `nvf_projector` | NVF에서 linear / circular | NVF에서 linear / circular |
| 투영장 샘플 수 `shape_nvf_grid_size` | 지정하면 거부 | NVF에서 사용 |

## 이름보다 역할부터 읽기

`raster_normal_method`와 `shape_nvf_grid_size`는 이름이 비슷하지만 같은 경로에서 함께 조절하는 설정이 아닙니다. 이 교재에서는 적용 대상까지 붙여 읽습니다. 아래의 한국어 이름은 설명용이며, Python 코드에는 **현재 지원되는 키워드**를 그대로 사용합니다.

| 설명용 이름 | 실제 API 키워드 | 답하는 질문 | 적용 대상 |
| --- | --- | --- | --- |
| 래스터 방향장 생성법 | `raster_normal_method` | 픽셀에서 방향을 어떻게 추정하고 연장할까? | raster NVF |
| 쉐입 투영장 샘플 수 | `shape_nvf_grid_size` | 보조벡터·P를 몇 개 점에서 평가할까? | shape NVF |
| NVF 투영 방식 | `nvf_projector` | 어떤 구성으로 성분 분해 기준 P를 만들까? | raster / shape NVF |
| 유지할 Fourier 차수 | `truncation_order` | 광학 계산에서 몇 개 Fourier 모드를 남길까? | 두 입력 모두 |

**현재 API는 새 이름을 사용합니다.** 예전 `normal_method` → `raster_normal_method`, `normal_grid` → `shape_nvf_grid_size`, `projector` → `nvf_projector`로 바뀌었습니다. 이전 이름은 별칭으로 받지 않습니다. `shape_nvf_grid`는 검토했던 후보일 뿐 실제 키워드가 아닙니다. 옵션 값과 수치 알고리즘은 같습니다. 현재 JobSpec 스키마 버전은 **10**입니다. Python 생성자는 현재 버전을 넣습니다. 이전 저장 파일은 층 목록과 대응하는 두께·재료 배열을 함께 역순으로 옮겨야 합니다. 장 표본의 깊이 `d`는 높이 `h_total − d`로 바꾸고, 기존 z 결합 텐서는 해당 좌표 변환도 적용합니다. 여기서 `h_total = sum(thickness)`는 전체 구조 높이입니다. 버전 숫자만 바꾸면 같은 시료를 보존할 수 없습니다. [좌표 이전 규칙](../../reference/common/conventions.md)을 확인하세요.

## 계수를 얻는 방식과 곱 규칙은 서로 독립

`fourier_coefficients`는 재료의 Fourier 계수를 얻는 방법이고, `factorization`은 불연속 곱을 처리하는 방법입니다. raster에서는 sampled/integrated와 li/nvf의 네 조합이 가능합니다. integrated raster는 각 픽셀 사각형의 적분을 반영하지만, 원래 픽셀 배열의 계단형 기하를 매끈한 원으로 바꾸지는 않습니다.

shape 경로는 단순한 원·타원·회전 사각형의 재료 계수를 해석적 적분으로 만들 수 있습니다. 다만 모든 도형 조합을 항상 닫힌 식으로 처리하는 것은 아닙니다. 겹치는 곡선의 painting 등에는 다각형 근사와 별도의 기하 정밀도 조절이 들어갑니다. NVF의 투영 계수도 별도의 수치 적분이 필요하므로, “재료 계수의 해석적 적분”이 “광학 해의 정확한 닫힌 해”를 뜻하지 않습니다.

```{figure} ../../figures/learn/nvf/08-settings.svg
:alt: 생성 방법, 투영 방식, 샘플 수와 Fourier 차수는 서로 다른 설정이다
:figclass: learning-figure

그림 6-2. 생성 방법, 투영 방식, 샘플 수와 Fourier 차수는 서로 다른 설정이다
```

## 래스터 방향장 생성법: raster_normal_method

래스터에는 원이나 사각형의 수학적 경계가 직접 들어 있지 않습니다. 재료값이 놓인 픽셀 배열에서 경계 방향을 추정하고, 내부까지 연장한 방향장을 만들어야 합니다. `raster_normal_method`는 이 생성 방법을 선택합니다. 기본값은 `distance`입니다.

| 방법 | 방향장을 얻는 출발점 | 알아둘 점 |
| --- | --- | --- |
| `distance` | 경계까지의 거리 변화 | 기본값. 픽셀 격자 때문에 방향 오차가 생길 수 있음 |
| `contour` | 픽셀에서 복원한 등고선의 기하 | 선택 의존성 contourpy가 필요함 |
| `spectral` | 평활화한 패턴의 Fourier 미분 | 날카로운 경계와 동률 처리의 영향을 확인 |
| `smooth` | 부드럽게 만든 패턴의 공간 미분 | 이 교재의 2상 raster 예제에서 사용 |

그 다음 `nvf_projector`가 이 정보를 이용해 linear 또는 circular 투영을 만듭니다. **네 가지 raster_normal_method와 두 가지 nvf_projector는 별개의 선택입니다.** shape는 도형의 기하에서 자체 방향장을 구성하므로 raster용 raster_normal_method를 지정하지 않습니다.

기준 구현의 contour/spectral/smooth는 두 재료의 패턴에 맞춘 경로입니다. 세 값 이상 또는 회색조 패턴은 distance 경로로 위임됩니다. 요청한 이름만 보고 내부에서 항상 같은 방식이 실행된다고 가정하면 안 됩니다. 또한 이 방향장 구성은 JAX/PyTorch를 골라도 기하 준비 단계에서 NumPy/SciPy로 수행되는 부분이 있습니다. 자세한 근거는 [`normal_methods.py`](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/normal_methods.py)에 있습니다.

## 쉐입 투영장 샘플 수: shape_nvf_grid_size

shape는 도형의 경계 기하로 보조장과 투영장을 구성할 수 있습니다. 하지만 P의 Fourier 계수를 수치적으로 구하려면 **한 주기 영역의 여러 위치에서 P를 평가**해야 합니다. `shape_nvf_grid_size`는 이 평가 격자의 한 축당 점 수입니다.

| 설정 | 한 주기 영역에서 평가하는 점 | 바뀌는 것 |
| --- | --- | --- |
| `shape_nvf_grid_size=128` | 128×128 = 16,384개 | 보조장·투영장 샘플링 해상도 |
| `shape_nvf_grid_size=256` | 256×256 = 65,536개 | 같은 장을 4배 많은 점에서 평가 |

도형을 128×128개의 재료 픽셀로 바꾸는 설정이 아닙니다. 도형과 Fourier 차수를 유지한 채 **보조장 J와 P를 읽는 간격**을 바꾸는 것입니다. 두 배로 늘리면 샘플 수는 네 배가 되지만, 전체 실행 시간이 반드시 네 배가 된다는 뜻은 아닙니다.

기본값은 `max(1024, 4Nx+1, 4Ny+1)`입니다. 너무 작은 명시적 격자는 유지할 Fourier 차수에 필요한 최소 조건을 만족하지 못해 거부될 수 있습니다. 촘촘한 격자는 투영 계수의 샘플링 오차를 줄이는 데 쓰며, 값과 기울기의 수렴은 각각 확인해야 합니다.

raster에서는 원래 이미지 해상도를 사용하고 필요한 경우 aliasing 하한까지 올립니다. 사용자가 넣은 shape_nvf_grid_size는 그 경로를 조절하지 않으므로 현재 API는 명시적인 지정을 거부합니다. raster의 기하 정밀도를 높이려면 원본 도형을 더 정밀한 픽셀로 다시 표현해야 하며, 단순 픽셀 복제로 잃어버린 경계 정보가 복원되지는 않습니다.

**shape_nvf_grid_size는 “몇 개 점에서 읽느냐”, truncation_order는 “몇 개 Fourier 성분을 남기느냐”입니다.** 하나를 높인다고 다른 하나의 부족함까지 해결되지는 않습니다. 9장에서 두 설정을 따로 바꾸며 수렴을 비교합니다.

이 책의 shape 예제는 빠른 학습을 위해 `shape_nvf_grid_size=128`을 명시합니다. **기본값을 128로 낮추라는 권장이 아닙니다.** 실제 사용에서는 아래의 수렴 절차로 선택하세요.
