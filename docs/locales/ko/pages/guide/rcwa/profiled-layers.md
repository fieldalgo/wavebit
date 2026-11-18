# 프로파일 층: 측벽각, 라운딩, 자동 z 슬라이스

`rcwa.ProfiledLayer`는 높이에 따라 단면이 바뀌는 층입니다. 측벽각과 상단 라운딩, footing이 있는 라인이나 위로 갈수록 윤곽이 줄어드는 기둥이 여기에 해당하고, 슬라이스는 Wavebit이 직접 자릅니다. 사용자는 형상 파라미터만 선언합니다. 준비 단계가 슬라이스를 정하고, 실행 결과는 사용한 슬라이스 수와 `z` 방향 계단의 추정 오차를 알려 줍니다. 모든 파라미터는 PyTorch와 JAX에서 미분됩니다.

```python
import math
from wavebit import Material, Units, rcwa

silicon = Material(epsilon=15.2 + 0.16j)
line = rcwa.ProfiledLayer(thickness=0.1, regions=(rcwa.Region(
    geometry=rcwa.Trapezoid(cd=0.045, cd_at='middle', swa=math.radians(86.),
                            top_rounding=0.005, footing=0.003),
    material=silicon),))
problem = rcwa.Problem(
    units=Units(length='um'),
    stack=rcwa.Stack(period=(0.12, 0.12), bottom=silicon, layers=(line,)),
    excitation=rcwa.Excitation(wavelength=0.5, sources=(rcwa.PlaneWave(theta=1.13, polarization='p'),)))
numerics = rcwa.Numerics(orders=(12, 0), factorization='li', slicing=rcwa.Slicing(count='auto', tolerance=1e-4))
result = rcwa.run(problem, numerics=numerics)
report = result.diagnostics['z_slicing'][0]
print(report['count'], report['staircase_error'])
```

[`examples/rcwa/tapered_line_swa.py`](../../../examples/rcwa/tapered_line_swa.py)는 테이퍼진 실리콘 라인의 반사 스펙트럼을 측벽각에 따라 훑습니다.

## 형상

`ProfiledLayer`는 `rcwa.Layer`처럼 `thickness`, `background`, `regions`를 받고, 영역은 나중 것이 앞의 것 위에 칠해집니다. 어느 영역이든 아래 세 형상 가운데 하나를 가질 수 있습니다. 이 형상들의 윤곽은 높이 `z`에 따라 바뀌며, `z`는 층의 아래 면에서 위로 문제의 길이 단위로 잽니다. 보통 형상(`Rectangle`, `Circle` 등)도 함께 둘 수 있고, 이들은 모든 높이에서 같습니다.

| 형상 | 나타내는 것 | 각 슬라이스는 |
| --- | --- | --- |
| `rcwa.Trapezoid(cd, swa, top_rounding, footing, center, cd_at)` | `y` 방향으로 변하지 않고 셀을 `y` 방향으로 가로지르는 라인. 이 `x`-`z` 단면을 가짐 | 슬라이스 폭의 `Rectangle`. 닫힌 형식이라 가로 계단이 없음 |
| `rcwa.Loft(bottom, top)` | 꼭짓점이 `bottom`(`z = 0`)에서 `top`(`z = thickness`)으로 순서대로 짝지어 선형으로 움직이는 윤곽 | 칠하기 경로의 `Polygon` |
| `rcwa.Profile(section)` | 임의의 윤곽. `section(z)`가 꼭짓점을 돌려주며, 모든 높이에서 개수가 같아야 함 | 칠하기 경로의 `Polygon` |

슬라이스는 내부 구현입니다. 결과와 층 인덱스는 공개된 층 하나를 기준으로 하며, `ProfiledLayer`와 보통 `Layer`는 자유롭게 섞어 쌓을 수 있습니다. `result.field(z=...)`는 평소처럼 스택 안의 높이를 받습니다. 높이 대신 정수 개수를 주면 층을 샘플링하듯 슬라이스마다 샘플링하므로, 슬라이스 사이의 면이 `z`에 나타납니다. 두 층의 라인이 옆으로 어긋난 경우(overlay 오차)는 각 층에 따로 `center`를 주면 됩니다. 가로 위치는 모든 층에서 셀 중심 기준 절대 좌표입니다.

## 측벽각과 라운딩

![Trapezoid 단면](../../figures/rcwa/profiled-trapezoid.svg)

| 양 | 규약 |
| --- | --- |
| `swa` | 바닥면과 측벽이 **재료 안쪽에서** 이루는 각. 단위는 `Rectangle.angle`과 같은 **라디안**: `math.radians(86.)` |
| `swa = pi/2` | 수직 측벽 |
| `swa < pi/2` | 위로 갈수록 좁아짐. 보통의 테이퍼 |
| `swa > pi/2` | 위로 갈수록 넓어짐. undercut(re-entrant) 측벽 |
| `swa=(left, right)` | 두 측벽에 각각 다른 각. 숫자 하나는 양쪽 모두를 뜻함 |
| `cd` | 라운딩 전 직선 측벽에서, `cd_at`이 가리키는 높이의 폭 |
| `cd_at` | 층의 `'bottom'`(기본), `'middle'`, `'top'`. 피팅에서는 각도에 따라 폭이 가장 덜 움직이는 `'middle'`을 흔히 씀 |
| `center` | 그 폭의 중점의 `x`, 셀 중심 기준 |
| `top_rounding` | 위쪽 두 모서리를 대신하는 볼록한 원호의 반지름. 재료를 뺌 |
| `footing` | 측벽과 바닥 사이를 채우는 오목한 fillet의 반지름. 재료를 더함 |

각 원호는 이어 주는 두 면 모두에 접합니다. 그래서 어느 높이에서든 폭은 모든 파라미터에 대해 연속 미분 가능하고, 슬라이스가 직선 측벽에서 원호로 넘어가는 자리에서도 미분이 튀지 않습니다. 내부에서 측벽은 `(z - z_cd)·tan(pi/2 - swa)`만큼 옮겨지며, 이 값은 `swa = pi/2`에서 정확히 0입니다. 그래서 라운딩 없는 수직 `Trapezoid`는 `Rectangle(size=(cd, Ly))` 층과 비트 단위까지 같은 구조입니다.

층에 맞지 않는 형상은 문제를 준비할 때 `shape.profile`로 거부합니다. 층 안에서 측벽끼리 만나는 경우, 라운딩이 위쪽 폭보다 넓은 경우, 라운딩과 footing을 합친 높이가 측벽보다 높은 경우, 라인이 자기 주기 이미지에 닿는 경우가 여기에 해당합니다. `Trapezoid`는 셀이 `Stack(period=...)`로 주어져야 합니다. 그렇지 않으면 `shape.profile_cell`로 거부합니다.

## How the layer is sliced

`rcwa.Numerics(slicing=rcwa.Slicing(...))`가 스택의 모든 프로파일 층에 적용됩니다.

| 필드 | 기본값 | 뜻 |
| --- | --- | --- |
| `count` | `'auto'` | 프로파일 층 하나의 슬라이스 수, 또는 `'auto'`. `'auto'`는 `min_count`에서 시작해, 추정 계단 오차가 `tolerance` 이하가 되거나 `max_count`에 이를 때까지 두 배씩 늘림 |
| `rule` | `'arclength'` | 슬라이스 경계를 두는 곳. `x`-`z` 평면에서 측벽 호 길이를 같은 간격으로 나누거나, `'uniform'`이면 두께를 같게 나눔 |
| `tolerance` | `1e-4` | 더 잘게 잘라서 앞으로 생길 효율 변화의 최댓값으로 허용하는 추정치 |
| `min_count`, `max_count` | `4`, `64` | `'auto'`가 시도하는 첫 개수와 가장 큰 개수 |
| `estimate` | `'prepare'` | 오차를 언제 추정하는가. `'prepare'`, `'run'`(매 실행마다 슬라이스 절반으로 한 번 더 풀어서, 비용 약 1.5배), `None` |

각 슬라이스는 자기 **중점** 높이의 단면을 씁니다. 매끄러운 프로파일에서 중점 규칙의 계단 오차는 `N**-2`로 줄어듭니다. 오차 추정은 이것을 가정하며, 측정에서도 그렇게 나왔습니다(아래). 둥근 모서리에서는 원호가 수평으로 꺾이는 곳에서 매끄러움이 깨집니다. 균일하게 자르면 차수가 1.4 쪽으로 떨어지고, 기본값처럼 같은 호 길이로 자르면 2를 유지합니다. 윤곽이 가장 빨리 움직이는 곳에 슬라이스가 몰리기 때문입니다.

![균일 슬라이스와 호 길이 슬라이스](../../figures/rcwa/profiled-slicing.svg)

**경계는 문제를 준비할 때 층 두께에 대한 비율로 고정됩니다.** 실행은 두께와 슬라이스 폭만 움직입니다. 그래서 측벽각, 라운딩, 높이에 대한 미분은 고정된 계단 하나의 미분이고, 모든 파라미터에 대해 매끄럽습니다. 다른 설계에서 `rcwa.run`을 새로 부르면 그 설계에 맞는 경계를 다시 계획합니다. 여러 설계를 같은 계단에서 비교하거나 유한차분을 구하려면 `rcwa.prepare` 하나로 실행하세요.

### 슬라이스 수 고르기

- **설계 하나나 짧은 sweep이라면 `'auto'`에 맡기세요.** 스택을 4, 8, 16, ... 슬라이스로 풀면서, [Richardson](https://doi.org/10.1098/rsta.1911.0009) 추정 `max|R_N - R_N/2| / (2**p - 1)`이 `tolerance`보다 작아질 때까지 갑니다. `p`는 마지막 세 개수에서 관측한 차수이며 2를 넘지 않게 씁니다. 준비 비용은 이 풀이들을 합한 만큼입니다.
- **피팅이나 최적화에는 `count`를 고정하세요.** 대표 설계 하나에서 `'auto'`를 한 번 돌려 `report['count']`를 읽고, `Slicing(count=N)`으로 준비합니다. 그러면 반복마다 계단 하나만 풀고, 미분도 그 계단의 미분이 됩니다.
- **설계가 개수를 정한 곳에서 멀리 움직인다면 `estimate='run'`을 쓰세요.** 추정치가 설계를 따라 갱신됩니다.
- **`'uniform'`은 다른 코드의 균일 계단과 비교할 때 씁니다.** 둥근 프로파일에서는 수렴이 더 느립니다.

`max_count`에 먼저 닿으면 마지막 개수로 풀고, `rcwa.diagnostics.SlicingWarning`을 내며, `result.validity.checks`에 `z_slicing_converged`를 거짓으로 남깁니다.

### 실행 결과가 알려 주는 것

`result.diagnostics['z_slicing']`에는 공개 층마다 항목이 하나씩 있습니다. 보통 층은 `None`이고, 프로파일 층은 다음 매핑입니다.

| 키 | 값 |
| --- | --- |
| `count` | 이번 실행에서 푼 슬라이스 수. 층을 통째로 풀었으면 1(아래 참고) |
| `planned_count` | 준비 단계가 고른 개수 |
| `rule`, `fractions` | 규칙과 슬라이스 경계. 경계는 두께에 대한 비율 |
| `merged` | 층을 한 층으로 풀었는지 |
| `staircase_error` | 효율의 계단 오차 추정치. 모든 프로파일 층을 통째로 풀었으면 `0.0` |
| `observed_order`, `order_used` | 마지막 세 개수에서 관측한 수렴 차수와 추정에 쓴 차수 |
| `levels`, `converged`, `tolerance` | `'auto'`가 푼 개수들과 `tolerance`를 만족했는지 |
| `source` | 추정이 어디서 왔는가: `'prepare'`, `'run'`, `'merged'` |

모든 프로파일 층이 함께 잘게 나뉘므로 추정치는 스택 전체에 대한 값입니다. 프로파일 층마다 같은 숫자가 들어갑니다.

### 높이에 따라 변하지 않는 층

라운딩 없는 수직 `Trapezoid`나 두 윤곽이 같은 `Loft`처럼 모든 슬라이스가 같다면, 그 층은 보통 층 하나로 풀립니다. 결과는 직접 그렇게 쓴 층과 비트 단위까지 같습니다. 다만 형상 값 중 미분을 가진 것이 없을 때만 그렇습니다. 측벽각이 JAX 추적자이거나 `requires_grad`인 텐서라면 층을 슬라이스한 채로 둡니다. 그래야 `swa = pi/2`에서의 미분이 그 각이 만들 테이퍼를 봅니다. 이때 값은 한 층으로 푼 결과와 반올림 오차 수준으로 같습니다(1e-12를 시험함).

## 왜 래스터가 아니라 영역 경로인가

모든 슬라이스는 영역으로 만들어집니다. 라인의 슬라이스는 닫힌 형식 경로의 `Rectangle`이고, `Loft`와 `Profile`의 슬라이스는 [경계 모델링](boundary-modeling.md)의 칠하기 경로에 놓이는 `Polygon`입니다. 샘플링(래스터) 층은 폭을 픽셀 단위로 양자화합니다. 그러면 측벽각이 바뀔 때 폭이 계단처럼 움직이고, 반사율도 각에 대해 계단이 되어 계단 사이에서는 미분이 0이 됩니다. 그래서 래스터 층이 있는 스택에 `ProfiledLayer`를 넣으면 `layer.profile_raster`로 거부하고, 보통 `Layer`에 프로파일 형상을 넣으면 `layer.profile_geometry`로 거부합니다.

## 인수분해와 비용

`Trapezoid`의 슬라이스는 높이 전체를 차지하는 사각형이라 경계면이 모두 `x`에 수직입니다. 여기에는 `factorization='li'`가 정확하며, CPU 코어 하나에서 차수 31개일 때 슬라이스당 약 10 ms가 듭니다. 기본값인 NVF도 같은 답(1e-14 이내)에 이르지만 슬라이스마다 법선장을 만들어 각각 약 1 s가 걸립니다. 라인에는 `'li'`를 쓰세요. `Loft`와 `Profile`은 `Polygon`이 받는 인수분해를 받으며, `'li'`는 받지 않습니다.

슬라이스 하나는 풀어야 할 층 하나입니다. JAX의 즉시 실행은 층당 약 4 s가 들고, `jax.jit` 컴파일 시간은 보통 층을 쌓은 스택처럼 슬라이스 수에 따라 늘어납니다. JAX로 개발하는 동안에는 개수를 작게 고정해 두세요.

## 미분

프로파일 형상의 모든 숫자는 영역 형상과 같은 방식으로 미분됩니다([미분 가이드](../common/autodiff.md)). `cd`, `swa`, `top_rounding`, `footing`, `center`와 층 `thickness`가 PyTorch와 즉시 실행 `jax.grad`에서 미분됩니다. `Trapezoid`의 슬라이스는 축에 나란한 사각형이므로 `factorization='li'`에서는 `jax.jit`으로도 돌아갑니다. `Loft`의 꼭짓점은 `Polygon`의 꼭짓점처럼 미분됩니다. 둥근 프로파일 하나에서 이 파라미터 각각에 대한 자동 미분이, 같은 준비로 구한 중앙 차분과 상대 4e-9 이내로 맞았습니다([측정](../../validation/rcwa/profiled-layers.md)).

`Profile`의 `section`은 슬라이스의 중점 높이를 받으며, 두께가 추적된 값이면 이 높이도 추적된 값입니다. 그러니 실행할 백엔드의 연산으로 작성하세요. 호출 가능한 객체는 이식 가능한 형태가 없으므로 `Profile`이 든 문제는 job으로 저장할 수 없습니다. `Trapezoid`와 `Loft`가 든 문제는 저장할 수 있습니다.

## 여기에 없는 것

- 높이에 따라 바뀌는 재료(점진 굴절률, 유효 매질 거칠기, 계면 층)는 이번 `ProfiledLayer`에 포함되지 않습니다.
- 연속 `z` 정식화는 없습니다. 층은 항상 계단이며, 그 오차를 보고합니다.
- 3D CAD, STEP, STL, GDS 입력은 읽지 않습니다.

## 참고 문헌

- L. F. Richardson, *The approximate arithmetical solution by finite differences of physical
  problems involving differential equations*, Phil. Trans. R. Soc. A **210**, 307-357 (1911).
  [doi:10.1098/rsta.1911.0009](https://doi.org/10.1098/rsta.1911.0009). 오차 추정.
- E. Popov, M. Nevière, B. Gralak and G. Tayeb, *Staircase approximation validity for
  arbitrary-shaped gratings*, J. Opt. Soc. Am. A **19**, 33-42 (2002).
  [doi:10.1364/JOSAA.19.000033](https://doi.org/10.1364/JOSAA.19.000033). 계단 근사가 어떻게 수렴하고 어디서 수렴하지 않는가.
- J. P. Hugonin and P. Lalanne, *RETICOLO software for grating analysis*,
  [arXiv:2101.00901](https://arxiv.org/abs/2101.00901). 슬라이스를 견준 외부 기준.
