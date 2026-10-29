# 03. 가장 먼저 지켜야 할 약속: 좌표, 단위, 재료

:::{note}
이 장의 그림에는 이전 생성자의 이름이 남아 있습니다: `bot`, `structure[i]`, `thickness=[d0, d1]`, 원의 지름 `lx`.
현재 API에서는 각각 `Stack(bottom=...)`, 아래층부터 층마다 `Layer(thickness=...)` 하나를 담는 `Stack(layers=(...))`,
`Circle(radius=...)`입니다. 본문과 코드는 현재 이름을 씁니다.
:::

Wavebit는 숫자만 보고 nm인지 µm인지 알아내지 않습니다. `Units(length="nm")`로 **하나의 길이 단위**를 정하고 `wavelength`, `period`, `thickness`, 도형 위치와 크기에 일관되게 씁니다. 이 책의 실행 예제는 nm를 사용합니다. 각도는 라디안이므로 30도는 `np.deg2rad(30.)`입니다.

## 먼저 전체 구조를 본다

아래 그림을 **반복되는 한 셀 → 쌓인 층 → 들어오는 빛** 순서로 읽어 보세요. 하나의 셀 안에 재료 패턴을 정하면 같은 패턴이 x·y 방향으로 반복됩니다. 그런 층들을 z 방향으로 쌓고, 빛이 들어올 방향과 파장을 지정합니다.

```{figure} ../../figures/learn/learn/03-stack-overview.svg
:alt: 두 층으로 이루어진 주기 구조의 3차원 개념도. 노란 테두리 안의 한 셀이 Px와 Py 간격으로 반복되고, structure 0과 1은 두께 d0와 d1로 바닥부터 위로 쌓인다. 위쪽 매질에서 들어오는 빛의 진행 방향 k와 입사각 theta, 방위각 phi를 표시한다.
:figclass: learning-figure

그림 3-1. **위에서 비스듬히 입사하는 예시**입니다. 옆으로는 주기 `period`, 위로는 층 두께 `thickness`, 위에서는 입사광을 읽습니다. 옅은 입사면은 빛의 진행 방향과 수직선을 포함하는 **가상의 면**입니다. 실제 재료 층이 아닙니다.
```

노란 테두리는 도형 하나의 경계가 아니라 **배경까지 포함한 한 셀**입니다. `Px`, `Py`는 도형의 폭이 아니라 셀의 반복 간격입니다. 그림에는 3×3 셀만 보이지만 계산에서는 평면 방향으로 계속 반복되는 구조를 다룹니다. 청록색은 반복 도형을, 회색 면은 층 구획을 보여 주며 특정 유전율 값에 대응하지 않습니다.

`layers[0]`은 가장 아래층이고 `layers[1]`은 그 위층입니다. 각 `rcwa.Layer`가 자기 `thickness`를 가지며, 여기서는 `d0`와 `d1`입니다. 바닥을 `z=0`으로 두면 첫째 층은 `0≤z≤d0`, 둘째 층은 `d0≤z≤d0+d1`을 차지합니다. 층 두께의 합인 전체 구조 높이를 `h_total`로 쓰면 윗면은 `z=h_total`입니다. 그림의 두 층에서는 `h_total=d0+d1`입니다.

위쪽 입사에서는 빛이 `−z`로 내려오며 `theta`는 안쪽 법선 `−z`에서 잽니다. 아래쪽 입사에서는 `+z`로 올라오며 `theta`도 그쪽 안쪽 법선 `+z`에서 잽니다. `phi`는 두 경우 모두 xy 성분을 `+x`에서 `+y` 쪽으로 잰 방위각입니다. 입사면을 바꿔도 층 목록과 좌표는 그대로입니다. [방향과 편광은 8장](light.md)에서 자세히 설명합니다.

그림의 두 층 구조는 입력 항목을 연결하기 위한 개념도입니다. 아래 실행 예제는 **패턴 층 하나**를 갖는 원기둥 구조입니다.

## 두 입력 방식에 공통인 좌표와 층 순서

**좌표축·셀 중심 원점·층 순서는 래스터와 도형(쉐입) 방식에 공통입니다.** 좌표축은 오른손 규칙 `x̂ × ŷ = ẑ`을 따릅니다. **`+z`는 위쪽이고, 구조 바닥이 `z=0`입니다.** 입체 그림의 `+x`는 화면 오른쪽 아래, `+y`는 표면을 따라 오른쪽 위, `+z`는 수직 위를 향합니다.

아래 왼쪽 xy 좌표도는 `+x`가 오른쪽, `+y`가 위쪽이며 `+z`는 화면 밖(`⊙`)입니다. 이제 위쪽에서 구조를 내려다보는 방향과도 맞습니다. 오른쪽은 같은 구조를 옆에서 본 층 순서입니다.

```{figure} ../../figures/learn/learn/03-coordinates.svg
:alt: 왼쪽은 x가 오른쪽, y가 위쪽, z가 화면 밖인 오른손 xy 좌표도이다. 래스터 방식의 4행 4열 배열을 단색 격자로 표시하며 재료값은 생략했다. 오른쪽 회색 구획은 재료 무늬를 생략한 두 층의 순서와 두께를 표시한다.
:figclass: learning-figure

그림 3-2. 왼쪽은 한 층의 xy 좌표, 오른쪽은 z 방향 층 순서이며 **래스터와 도형 방식에 공통**입니다. 왼쪽에 겹쳐 그린 격자와 행·열 번호는 **래스터 배열의 예시**입니다. `[층, 행 y, 열 x]`도 래스터 배열의 표기입니다. 도형 방식은 같은 xy 좌표에서 중심·크기·재료를 지정합니다.
```

`Stack.bottom`은 바닥 아래, `Stack.top`은 전체 구조 위의 반무한 매질입니다. 둘은 층 목록에 포함하지 않습니다.

| 입력·배열 | 의미 |
| --- | --- |
| `Layer(epsilon=array[y, x])` | 래스터: 행, 열. 사선 격자에서는 v, u |
| `period=(Px, Py)` | 직접 쓰는 쌍은 x 먼저 |
| `Numerics(orders=(Mu, Mv))` | 남길 차수의 격자 벡터 방향 반폭도 u 먼저(직사각형 셀에서는 x 먼저). 항상 쌍을 명시 |
| `result.T[row, column]` | 결과 배열은 v 방향 차수가 행, u 방향 차수가 열 |
| `layers=(Layer(thickness=d0, ...), Layer(thickness=d1, ...))` | 바닥부터 위로 나열한 층 |
| `top`, `bottom` | 위·아래의 반무한 매질; 입사 방향을 바꿔도 이름 유지 |

xy 셀의 중심은 `(0, 0)`입니다. 직교 셀의 래스터 **0번 행은 −y 쪽**이며, 첫 픽셀 중심은 `−Py/2 + Py/(2·행 수)`에 있습니다. x도 같은 방식입니다. `+z`는 바닥에서 위로 향하고, `layers`의 번호도 그 순서로 증가합니다. 오래된 일부 설명에 나오는 ‘첫 행은 y=0’은 현재 규약이 아닙니다.

### 코드로 쌓아 보기

```python
from wavebit import Material, rcwa

stack = rcwa.Stack(
    bottom=Material(epsilon=2.25),       # 기판: z < 0
    layers=(
        rcwa.Layer(thickness=100., background=Material(epsilon=4.)),   # z: 0 → 100
        rcwa.Layer(thickness=200., background=Material(epsilon=9.)),   # z: 100 → 300
        rcwa.Layer(thickness=50., background=Material(epsilon=2.)),    # z: 300 → 350
    ),
    top=Material(epsilon=1.),            # 위쪽 공기: z > 350
    period=(600., 600.),
)
light = rcwa.PlaneWave(side="top", polarization="s")
```

아래에서 비추려면 `side="bottom"`만 바꿉니다. 층을 뒤집지 않습니다. `R`은 입사한 쪽으로 돌아가는 빛, `T`는 반대쪽으로 나간 빛입니다.

## 재료 입력 방법 찾아보기

이 장에서는 **상수 재료를 어느 영역에 넣는지**부터 배웁니다. 입력 방법 전체와 이방성은 [7장: 재료](materials.md)에서 이어집니다.

| 표현하려는 재료 | 입력 방법 | 자세한 설명 |
| --- | --- | --- |
| 일정한 등방성 유전율 | `Material(epsilon=12.25)`; 흡수가 있으면 복소수 값 | 아래 상수 재료 예제와 {ref}`7장 <material-inputs>` |
| 유전율과 투자율을 함께 지정 | `Material(epsilon=4., mu=1.2)` | 아래 ε·μ 예제와 [7장](materials.md) |
| 재료표에서 실리콘 등을 선택 | `materials.load('Si')` | [7장: 재료 선택과 파장 스윕](materials.md) |
| 직접 가진 파장별 ε·μ 자료 | `TabulatedMaterial(...)` | [7장: 사용자 재료표](materials.md) |
| 방향에 따라 반응이 다른 이방성 재료 | `Material(epsilon=[εx, εy, εz])` 또는 3×3 텐서 | {ref}`7장: 등방성과 이방성 <material-anisotropy>` |

**이방성도 `Material`을 층이나 도형에 할당합니다.** Wavebit가 텐서용 계산 행렬을 구성합니다. 이방성 도형은 재료 경계를 적분하는 방식이 기본값이고, 법선장은 여전히 수치적으로 계산합니다. 층과 도형에는 전체 3×3 재료 텐서를 넣을 수 있습니다. 위·아래 반무한 매질에는 등방성 재료를 지정합니다. 예제와 지원 범위는 [7장](materials.md)에 정리했습니다.

## 상수 유전율은 `Material`로 넣는다

`Material(epsilon=...)`에 넣는 수는 **상대 유전율 εr**입니다. 유전율을 12.25로 정한 예시 재료라면 다음이면 충분합니다.

```python
from wavebit import Material

silicon_like = Material(epsilon=12.25)
```

이것을 영역의 `material`, 층의 `background`, 위·아래 매질인 `top`, `bottom`에 넣습니다. **모든 재료 자리는 `Material`을 받으며, `12.25` 같은 맨 숫자는 거부합니다**(코드 `material.type`). 숫자만으로는 유전율인지 굴절률인지 알 수 없기 때문입니다. 위 변수는 실제 실리콘 데이터를 불러온 것이 아니라 일정한 유전율을 가정한 예시입니다.

### 예시: 공기 속 원기둥을 기판 위에 놓기

아래 코드는 그대로 실행할 수 있습니다. 600×600 nm 셀마다 지름 240 nm, 높이 180 nm인 원기둥이 반복됩니다. 원기둥의 상대 유전율은 12.25, 주변 공기는 1, 아래 기판은 2.25로 정했습니다.

```{figure} ../../figures/learn/learn/03-cylinder-materials.svg
:alt: 기판 위 한 셀의 원기둥을 보여 주는 입체도와 평면도. 셀은 600×600 nm, 원기둥은 지름 240 nm와 높이 180 nm이며 셀 중심에 있다. z=0 아래는 유전율 2.25인 반무한 기판 bot, z=180 위는 유전율 1인 공기 top이다. 패턴 층 안에서 원기둥은 material 12.25, 그 바깥 공기는 background 1이다. 위에서 파장 900 nm인 빛이 수직 입사한다.
:figclass: learning-figure

그림 3-3. **코드와 같은 한 셀**을 그렸습니다. 점선 상자는 패턴 층의 범위이며 물체의 벽이 아닙니다. 원기둥은 x·y 방향으로 600 nm마다 반복됩니다. 기판은 아래로 계속 이어지며, 그림의 기판 두께는 표시를 위한 생략입니다. 이 예제의 모든 매질은 μr=1입니다.
```


<!-- material-assignment-example:start -->
```python
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length="nm"),            # 모든 길이는 nm
    stack=rcwa.Stack(
        period=(600., 600.),             # x·y 방향 반복 간격
        top=Material(epsilon=1.),        # 위쪽 공기: 상대 유전율 1
        bottom=Material(epsilon=2.25),   # 아래쪽 기판: 상대 유전율 2.25
        layers=(rcwa.Layer(
            thickness=180.,              # 패턴 층의 두께
            background=Material(epsilon=1.),   # 패턴 층에서 도형 바깥의 공기
            regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=120.),   # 셀 중심에 놓은 반지름 120 nm 원
                material=Material(epsilon=12.25)),)),)),   # 원 내부: 상대 유전율 12.25
    excitation=rcwa.Excitation(
        wavelength=900.,
        sources=(rcwa.PlaneWave(side="top", theta=0., polarization="s"),)),   # 위에서 수직 입사
)

result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 1)))   # 빠르게 실행해 보는 낮은 차수
print(f"전체 반사율: {float(result.R.sum()):.6f}")
print(f"전체 투과율: {float(result.T.sum()):.6f}")
```
<!-- material-assignment-example:end -->

위 코드를 실행한 출력입니다.

```text
전체 반사율: 0.044164
전체 투과율: 0.955836
```

반사율 약 **4.42%**, 투과율 약 **95.58%**입니다. 두 값은 입사 에너지에 대한 비율이며, 합은 반올림 오차 범위에서 1입니다. **`orders=(1, 1)`로 계산한 실행 예시 값이며, 수렴을 확인한 최종 예측값은 아닙니다.**

**재료를 넣는 곳은 네 군데입니다.** `top`은 패턴 위 공간, `bottom`은 패턴 아래 기판, `background`는 패턴 층에서 도형 바깥 공간, `material`은 도형 내부입니다. 위쪽 공기와 층의 배경에 같은 값 1을 썼지만 서로 다른 영역입니다. 원기둥 재료를 바꾸려면 영역에 넣는 `Material`만 바꾸면 됩니다.

`rcwa.Circle(radius=120.)`은 기본으로 셀 중심(`center=(0., 0.)`)에 놓이며, `radius`는 **반지름**입니다. 원기둥의 지름은 240 nm입니다. 모든 길이에 nm를 사용했으므로 `wavelength`, `period`, `thickness`, `radius`를 같은 기준으로 읽습니다. 출력은 모든 회절차수를 합한 반사율과 투과율입니다.

여기의 낮은 Fourier 차수는 입력과 실행 흐름을 빠르게 확인하기 위한 설정입니다. 정확도는 [10장의 차수 설명](fourier-orders.md)과 [19장의 수렴 확인](convergence.md)으로 이어집니다. 여러 재료 입력 방법은 [7장](materials.md)에서 비교합니다.

### ε와 μ를 함께 지정할 때

상대 유전율 εr과 상대 투자율 μr을 함께 정하려면 `Material` 객체에 두 값을 넣습니다. 아래는 **같은 원기둥 구조에서 원기둥 재료만 εr=4, μr=1.2로 바꾼** 독립 실행 예제입니다. 실제 재료 데이터를 불러온 것이 아니라 두 값을 직접 지정한 가상 재료입니다.

<!-- magnetic-material-example:start -->
```python
from wavebit import Material, Units, rcwa

pillar_material = Material(epsilon=4., mu=1.2)

problem = rcwa.Problem(
    units=Units(length="nm"),            # 모든 길이는 nm
    stack=rcwa.Stack(
        period=(600., 600.),             # x·y 방향 반복 간격
        top=Material(epsilon=1.),        # 위쪽 공기: 상대 유전율 1
        bottom=Material(epsilon=2.25),   # 아래쪽 기판: 상대 유전율 2.25
        layers=(rcwa.Layer(
            thickness=180.,              # 패턴 층의 두께
            background=Material(epsilon=1.),   # 패턴 층에서 도형 바깥의 공기
            regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=120.),   # 셀 중심에 놓은 반지름 120 nm 원
                material=pillar_material),)),)),   # 이 원기둥에 εr과 μr을 함께 지정
    excitation=rcwa.Excitation(
        wavelength=900.,
        sources=(rcwa.PlaneWave(side="top", theta=0., polarization="s"),)),   # 위에서 수직 입사
)

result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 1)))   # 빠르게 실행해 보는 낮은 차수
print(f"전체 반사율: {float(result.R.sum()):.6f}")
print(f"전체 투과율: {float(result.T.sum()):.6f}")
```
<!-- magnetic-material-example:end -->

위 코드를 실행한 출력입니다.

```text
전체 반사율: 0.007886
전체 투과율: 0.992114
```

이 출력도 낮은 차수 `(1, 1)`에서 확인한 실행 예시입니다.

`pillar_material`을 도형의 `material`에 넣으면 **εr=4와 μr=1.2가 그 도형에 함께 적용**됩니다. 같은 방법으로 `top`, `bottom`, `background`에도 `Material` 객체를 넣을 수 있습니다. μ를 생략한 `Material(epsilon=4.)`는 비자성 재료인 μr=1입니다. 위의 비자성 원기둥과는 재료가 달라 계산 결과도 달라집니다.

## 외부 자료가 굴절률로 주어졌을 때

굴절률과 유전율은 다른 양입니다. 비자성 재료의 외부 자료가 `n=3.5`로 주어졌다면 `εr=n²=12.25`로 변환합니다. 코드에서는 `Material(epsilon=3.5**2)` 또는 `Material(epsilon=12.25)`로 씁니다. 이미 유전율을 알고 있다면 변환하지 않습니다. 일반적인 등방성 관계는 `n²=εr μr`이므로 자성 재료에 `n²`만 유전율로 넣으면 안 됩니다.

실제 실리콘의 파장별 데이터를 쓰려면 `wavebit.materials.load('Si')`로 재료를 선택합니다. 상수·분산 표·굴절률 자료의 구분은 [재료 장](materials.md)에서 설명합니다.
