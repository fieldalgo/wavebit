# 키워드 안내

:::{note}
이 페이지는 `wavebit.rcwa`의 현재 RCWA 선언과 그것을 실행하는
`wavebit.Execution`의 필드를 나열합니다. 예전에는 이전
`rcwa(...)` 생성자의 키워드를 나열했습니다. [Earlier spellings](#earlier-spellings)가 그 각각을
현재의 위치에 대응시키고, 위치가 없는 것들을 밝힙니다. [이전 API 레퍼런스](../migration/legacy-api.md)에
완전한 생성자가 보존되어 있습니다.
:::

각 선언이 받는 모든 필드, 그 필드가 받아들이는 값, 기본값, 거부하는 값을 정리합니다.
답하는 질문별로 묶었습니다 -- 문제와 그 출력, 적층, 빛,
수치 설정, 실행 위치 -- 그리고 선언마다 절이 하나씩 있습니다.
시그니처와 docstring은 [RCWA API 레퍼런스](rcwa/index.md)에 있습니다.

FDTD 모듈에는 자체 선언(`fdtd.Problem`, `fdtd.Scene`, `fdtd.Numerics` 등)이
있으며, 이는 [FDTD API 레퍼런스](fdtd/index.md)와
[FDTD 솔버가 문제를 모델링하는 방식](../guide/fdtd/model.md)에 있습니다.

이 페이지는 찾아보기용입니다. 선택지가 무엇을 *뜻하는지*와 그 사이에서 어떻게 고르는지는 사용자
가이드가 주제별로 하나씩 다룹니다: [빛](../guide/rcwa/light.md),
[장면](../guide/rcwa/scene.md), [수치 설정](../guide/rcwa/numerics.md),
[근접장](../guide/rcwa/fields.md), [백엔드](../guide/common/backends.md).

몇 가지 규칙은 전체 인터페이스에 적용되므로 한 번 적어 둘 만합니다.

* **알 수 없는 키워드는 삼켜지지 않고 거부됩니다.** 모든 `rcwa` 선언은 키워드 전용이며
  가장 가까운 실제 필드를 알려 줍니다. `rcwa.Stack(bot=...)`은 `bottom`을 제안하는 `TypeError`를,
  `rcwa.Numerics(truncation_order=(2, 2))`는 `orders`를 나열하는 같은 예외를 일으킵니다. 이전
  생성자의 키워드도 같은 방식으로 거부되며, 별칭으로 읽히지 않습니다. `Units`와
  `Execution`은 알 수 없는 키워드를 Python 자체의 `TypeError`로 거부합니다.
* **선언은 불변입니다.** `.replace(...)`는 새 선언을 만들고 그 검증을 다시
  실행하며, `prepared.run(problem)`은 완전한 새 문제를 받습니다.
* **거부는 코드와 발견 시점으로 이름 붙습니다.** `stack.period` 같은 코드는
  발생한 `ValidationError` 또는 `UnsupportedFeatureError`의 `.code`이며, 모든 코드는
  [Refusal codes](#refusal-codes)에 나열된 `area.item` 형식 하나를 가집니다. *생성 시*는
  선언 자체의 생성자를 뜻합니다. *check 시*는 `rcwa.check`가 이를 보고하고
  `rcwa.prepare`와 `rcwa.run`이 예외를 일으킨다는 뜻이며, *prepare 시*와 *run 시*는 그보다 앞선 어떤 단계도
  이를 보지 못한다는 뜻입니다.
* **모든 재료는 재료 모델입니다** -- `Material`, `ConductiveMaterial`,
  `DispersiveMaterial` 또는 `TabulatedMaterial` -- 절대 맨 숫자가 아닙니다. 그 `epsilon`은
  굴절률이 아니라 상대 유전율이며(`eps_from_refractive_index`가 변환합니다), 맨
  `3.5`는 어느 쪽으로도 읽히지 않고 check 시 `material.type`으로 거부됩니다.
* **길이는 하나의 단위를 공유합니다.** 문제의 `Units(length=...)`이며, 주기, 두께, 형상
  치수, 소스 위치, 진공 파장이 여기에 해당합니다. `kx`, `ky`, `bloch_k`는 그
  역단위입니다. 각도는 라디안입니다.

## 문제

### `rcwa.Problem`

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `stack` | 필수 | `rcwa.Stack` | 그 밖의 모든 것: `problem.type`, 생성 시 |
| `excitation` | 필수 | `rcwa.Excitation` | 그 밖의 모든 것: `problem.type` |
| `units` | `Units()`, 미터 | `wavebit.Units`. `Units(length=...)`는 `'m'`, `'mm'`, `'um'` 또는 `'nm'`을 받습니다 | `'nm'` 같은 맨 문자열: `problem.type`. 다른 모든 단위 이름: `units.length` |
| `outputs` | `Outputs()` | `rcwa.Outputs` | 그 밖의 모든 것: `problem.type` |

### `rcwa.Outputs`

실행이 무엇을 보존하는지 정합니다. 필드 재구성과 산란 행렬은 필요한 상태를
층별로 보존해야 하므로 명시적으로 켜야 합니다. [필드와 회절 관측값](../guide/rcwa/fields.md)에서 설명합니다.

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `diffraction` | `True` | `bool`: 차수별 효율, s/p 진폭비, 출사 필드(`field_*`)와 양쪽 포트의 전력(`result.R`, `result.T`, `result.reflection`, `result.transmission`). 단일 `PlaneWave`만 이를 가집니다 | `bool`이 아닌 값(`1`): `outputs.boolean`, 생성 시. 다른 소스 옆에서, 또는 두 번째 소스 옆에서 `True`: `rcwa.coherent_diffraction`, check 시 |
| `fields` | `False` | `bool`: `result.field(x=..., y=..., z=...)`와 `result.boundary_fields(...)`가 읽는 상태를 보존 | `bool`이 아닌 값: `outputs.boolean` |
| `scattering` | `False` | `bool`: 적층의 산란 행렬을 파장마다 하나씩 `result.scattering`으로 보존 | `bool`이 아닌 값: `outputs.boolean`. `algorithm='etm'` 아래에서 `True`: `rcwa.scattering_algorithm`. 결맞는 소스와 함께 `True`: `rcwa.coherent_scattering`, 둘 다 check 시 |
| `power_region` | `None`, `'cell'`로 읽음 | `'cell'`, `'infinite'`, 또는 길이 단위의 `((xmin, xmax), (ymin, ymax))`: 결맞는 소스 실행이 경계 전력을 평균하는 영역. 단일 `PlaneWave`는 이를 읽지 않습니다 | 단일 `PlaneWave`가 아닌 소스 옆에서, 모두 `outputs.power_region`, check 시: 다른 문자열. `min < max`인 실수 `(min, max)` 쌍 두 개가 아닌 값(뒤집혔거나 폭이 0, 다른 모양, 맨 숫자). 소스들이 둘 이상의 Bloch 섹터에 걸칠 때의 `None`. 그리고 `GaussianBeam`이나 고립된 `ElectricDipole`에서는 유한 직사각형이 아닌 `None` 또는 모든 문자열 |

## 적층

[RCWA 적층 선언하기](../guide/rcwa/scene.md)에서 설명합니다.

### `rcwa.Stack`

`period`와 `lattice`는 하나의 셀을 쓰는 두 표기입니다. 둘 다 지정하면 거부되고, 어느 쪽도 지정하지 않으면
단위 정사각형이 됩니다.

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `layers` | `()` | `rcwa.Layer`의 튜플, **아래에서 위로**. 아래 계면이 `z = 0`입니다. 빈 적층은 계면 하나입니다 | `Layer`가 아닌 모든 것: `stack.layers`, 생성 시. 영역 층 옆의 래스터 층: `rcwa.mixed_layer_encoding`, check 시 |
| `top` | `Material(epsilon=1.)` | 위쪽 반공간의 재료 모델. 수동 이중 음수 매질(`Re(eps) < 0`, `Re(mu) < 0`)은 빛이 들어오는 쪽이든 아니든 어느 쪽에서도 계산됩니다. [음의 굴절률 매질](../theory/rcwa/negative-index-media.md)을 보세요 | 맨 숫자: `material.type`. 전파하는 능동 이중 음수 매질(`Re(eps*mu) > 0`이고, `Im(eps) < 0` 또는 `Im(mu) < 0`): `rcwa.invalid`, 둘 다 check 시 |
| `bottom` | `Material(epsilon=1.)` | 아래쪽 반공간, 같은 방식으로 읽음 | `top`과 같음 |
| `period` | `lattice`가 주어지지 않으면 `(1., 1.)` | `(Lx, Ly)`: 양수이고 유한한 실수 길이 두 개. 셀의 축 정렬 표기 | `lattice` 옆에서: `stack.cell_ambiguous`, 생성 시. 스칼라, `dict` 또는 양수가 아닌 길이: `stack.period`, check 시 |
| `lattice` | `None` | `(2, 2)` 배열의 열로 둔 두 기본 벡터, 또는 `wavebit.Lattice`. 직사각형이 아닌 셀의 유일한 표기. 음의 변을 가진 대각 기저는 쓰인 그대로 읽힙니다. 래스터와 차수 이름은 뒤집힌 축을 따라 진행합니다([장면 가이드](../guide/rcwa/scene.md)) | `period` 옆에서: `stack.cell_ambiguous` |

영역 기하와 `period`는 PyTorch와 즉시 실행 `jax.grad`에서 미분됩니다.
`jax.jit`나 `jax.vmap` 아래에서는 셀과 모든 영역 기하가 `prepared.host_topology`로 거부됩니다
([자동 미분](../guide/common/autodiff.md#rcwa-which-inputs-differentiate-on-which-backend)).

### `rcwa.Layer`

층은 배경 위에 영역으로 칠하거나 래스터로 표본화하며, 둘을 함께 쓰지는 않습니다.

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `thickness` | 필수 | 유한하고 음수가 아닌 실수 길이 | 음수, 복소수 또는 유한하지 않은 값: `layer.thickness`, check 시 |
| `background` | 영역 층에서 `Material(epsilon=1.)` | 어떤 영역도 칠하지 않은 곳의 재료 모델 | `epsilon` 옆에서: `layer.ambiguous`, 생성 시. 맨 숫자: `material.type`, check 시 |
| `regions` | `()` | `rcwa.Region`의 튜플. 뒤의 영역이 앞의 영역 위에 칠해지며, 어느 길이 단위에서도 같습니다 | `Region`이 아닌 모든 것: `layer.regions`. `epsilon` 옆에서: `layer.ambiguous`, 둘 다 생성 시. 경계 계획기가 하나를 다른 하나 위에 칠할 수 없는 두 영역(x 축에 대해 1 rad와 2.3 rad, 또는 그로부터 약 2e-5 rad 이내의 각도를 이루는 변들을 따라 동시에 외곽선이 맞닿는 경우. 맞닿으면서 다른 곳에서 크기의 약 1e-11 이내로 다가오거나, 크기의 약 1e-12보다 얇은 띠만큼 겹치는 경우. 또는 셀에서 너무 멀리 쓰여 좌표가 계획기의 미세 이동을 분해할 수 없는 경우): `rcwa.unsupported`, prepare 시 |
| `epsilon` | `None` | 모양이 `(Ny, Nx)`인 표본화된 상대 유전율 래스터, 복소수 허용. 행 하나는 1D 격자이며, `(Ny, Nx, 3, 3)`은 픽셀마다 텐서를 담습니다. 한 적층의 모든 래스터 층은 같은 모양입니다 | 배열 안의 `Material` 객체, 또는 모양이 다른 래스터 층: `rcwa.invalid`, check 시. 그 밖의 모양(1D, 3D, 빈 배열 또는 0차원 배열, 또는 `(Ny, Nx, 2, 2)`): 기대한 모양과 실제 모양을 담은 `layer.raster_shape`, check 시. NaN 또는 무한대 표본: `problem.nonfinite`, check 시. 어느 픽셀에서든 0인 `eps_zz`, 또는 역수가 실행 정밀도에서 오버플로하는 값(배경이나 영역 재료에도 해당): `rcwa.zero_permittivity`, check 시 |
| `mu` | `None` | `epsilon` 옆의 상대 투자율 래스터, 같은 모양. 영역 층에서 투자율은 각 `Material`에 속합니다 | `epsilon` 없이: `layer.mu_without_raster`, 생성 시. `(Ny, Nx)` 또는 `(Ny, Nx, 3, 3)`이 아닌 모양, 또는 `epsilon`과 다른 `(Ny, Nx)`(어느 쪽이든 픽셀마다 텐서를 담을 수 있음): `stack.layers[i].mu`에서의 `layer.raster_shape`, check 시 |

### `rcwa.Region`

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `geometry` | 필수 | `rcwa.Rectangle`, `rcwa.Circle`, `rcwa.Ellipse` 또는 `rcwa.Polygon` | 레시피 `dict`를 포함한 그 밖의 모든 것: `shape.type`, 생성 시 |
| `material` | 필수 | 스칼라 또는 텐서 재료 모델 | 맨 숫자: `material.type`, check 시 |

### 기하 형상

길이는 문제의 단위로 쓴 절대값이며, `center`와 `vertices`는 셀
중심에서 잽니다. 주기보다 커서 자신의 주기적 이미지와 부딪치는 형상은
prepare 시 `rcwa.invalid`로 거부됩니다.

| 선언 | 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- | --- |
| `rcwa.Rectangle` | `size` | 필수 | 전체 폭 `(wx, wy)`, 실수 | 성분이 두 개가 아님: `shape.dimension`, 생성 시. 복소수 또는 문자열 항목: `shape.value`, 생성 시 |
| | `center` | `(0., 0.)` | `(x, y)` | 성분이 두 개가 아님: `shape.dimension`. 복소수 또는 문자열 항목: `shape.value` |
| | `angle` | `0.` | 반시계 방향 회전, 라디안 | 복소수 또는 문자열: `shape.value` |
| `rcwa.Circle` | `radius` | 필수 | 길이 | 복소수 또는 문자열: `shape.value` |
| | `center` | `(0., 0.)` | `(x, y)` | 성분이 두 개가 아님: `shape.dimension`. 복소수 또는 문자열 항목: `shape.value` |
| `rcwa.Ellipse` | `radii` | 필수 | 주 반축 `(rx, ry)` | 성분이 두 개가 아님: `shape.dimension`. 복소수 또는 문자열 항목: `shape.value` |
| | `center`, `angle` | `(0., 0.)`, `0.` | `Rectangle`과 같음 | `Rectangle`과 같음 |
| `rcwa.Polygon` | `vertices` | 필수 | 경계를 따라 순서대로 놓인 `N >= 3`개의 쌍 `(x, y)`, 또는 `(N, 2)` 배열. 감는 방향은 어느 쪽이든 가능 | 셋 미만: `shape.vertices`. 단순하지 않은 외곽선(길이 0인 변, 서로 교차하거나 맞닿는 두 변): `shape.outline`, 둘 다 생성 시. `factorization='li'` 또는 `nvf_projector='circular'` 아래에서, 또는 텐서 재료와 함께: `rcwa.unsupported`, check 시 |

`factorization='li'`는 축 정렬 직사각형만 만듭니다. 그 아래에서 다른 영역은 모두
check 시 `rcwa.unsupported`로 거부됩니다.

## 빛

[빛을 기술하기](../guide/rcwa/light.md)와
[결맞는 조명과 내부 광원](../guide/rcwa/coherent-illumination.md)에서 설명합니다.

### `rcwa.Excitation`

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `wavelength` | 필수 | 진공 파장: 양수이고 유한한 실수 스칼라, 또는 그런 값들의 비어 있지 않은 1차원 배열이나 튜플. 어느 쪽이든 결과는 맨 앞에 파장 축을 유지합니다 | 복소수 또는 양수가 아님: `wavelength.value`. 2차원: `wavelength.shape`. 비어 있음: `wavelength.empty`, 모두 check 시 |
| `sources` | 필수 | `PlaneWave`, `IncidentOrders`, `GaussianBeam`, `ElectricDipole`의 비어 있지 않은 튜플. `PlaneWave` 하나는 회절 문제이고, 그 밖의 것은 경계 전력과 요청한 필드를 보고하는 결맞는 소스 실행입니다 | 비어 있음, 또는 그 밖의 모든 것(방향을 키로 쓰는 `dict` 포함): `excitation.sources`, 생성 시 |

결맞는 소스 실행은 `Outputs(diffraction=False)`, `algorithm='smm'`을 쓰고, `detuning`과
`scattering`이 없으며, PyTorch나 즉시 실행 `jax.grad`가 읽을 수 있는 값을 받습니다. `jax.jit`나
`jax.vmap` 아래에서 추적된 입력은 `rcwa.coherent_transform`으로 거부됩니다.

### `rcwa.PlaneWave`

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `side` | `'top'` | `'top'` 또는 `'bottom'`: 빛이 들어오는 반공간. 장면 레이블은 결코 움직이지 않습니다. 결과는 입사 기준이며, `theta`와 `phi`는 입사 매질에서 잽니다 | 그 밖의 모든 것(`'bot'`): `source.side`, 생성 시 |
| `polarization` | `'s'` | `'s'`, `'p'`, 또는 복소 존스 쌍 `(a_s, a_p)`. 쌍은 단위 길이로 정규화되며 전체 위상을 유지합니다 | 다른 문자열(`'te'`): `source.polarization`, 생성 시. 0이거나, 유한하지 않거나, 성분이 두 개가 아닌 쌍: `source.polarization_value`, check 시 |
| `amplitude` | `1.` | V/m 단위의 입사 전기장 진폭으로, 차수의 `reference['incident_field']`로 공개됩니다. `field_*`와 다른 필드는 이에 비례하고 `power`는 그 제곱에 비례하며, `amplitude_ratio_*`와 `efficiency`는 변하지 않습니다 | 0, 유한하지 않은 값, 또는 배열: `source.amplitude`, check 시 |
| `theta` | `None`, 수직 입사 | 안쪽 법선에서 잰 입사광 방향의 극각, 라디안. 굴절률 부호와 관계없이 무손실 매질에서 빛의 전력이 진행하는 방향입니다(음의 굴절률 매질에서는 이때 면내 파동 벡터가 반대 방향을 가리킵니다). 0이 아닌 허수부는 형식적인 복소수 각도 매개변수화를 선택합니다. [Real-angle incidence](common/conventions.md#real-angle-incidence)와 [음의 굴절률 매질](../theory/rcwa/negative-index-media.md)을 보세요 | `kx` 또는 `ky` 옆에서: `source.direction_ambiguous`, 생성 시. 배열이나 시퀀스(소스당 각도 하나): `source.direction`, check 시. 단독이든 배열 안이든, 실수부든 허수부든 NaN 또는 무한대: `excitation.sources[i].theta`에서의 `problem.nonfinite`, check 시 |
| `phi` | `None`, `0.`으로 읽음 | 입사광 방향의 방위각, 라디안 | `kx` 또는 `ky` 옆에서: `source.direction_ambiguous`. 배열이나 시퀀스: `source.direction`, check 시. NaN 또는 무한대: `excitation.sources[i].phi`에서의 `problem.nonfinite`, check 시 |
| `kx`, `ky` | `None` | 역길이 단위의 가로 각파수로, `theta`/`phi`의 대안입니다. `2π/λ`로 나누지 않은 물리값이며, 생략한 성분은 0입니다 | `theta` 또는 `phi` 옆에서: `source.direction_ambiguous`. NaN 또는 무한대: `excitation.sources[i].kx`(`.ky`)에서의 `problem.nonfinite`, check 시 |

### `rcwa.IncidentOrders`

하나의 기준 방향 주위에서 여러 회절 차수를 결맞게 조명합니다. 결맞는 소스
실행이므로 차수별 효율이 아니라 경계 전력을 보고합니다.

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `weights` | 필수 | `(((m, n), weight), ...)`: 정수 차수 쌍과 각각의 복소 가중치. 가중치는 빛이 들어오는 경계의 `x = y = 0`에서의 상대 전기장 진폭입니다. 1D 차수 `m`은 `(m, 0)`으로 씁니다 | `dict`, 빈 튜플, 또는 Python 정수 두 개가 아닌 레이블: `source.weights`(맨 정수 레이블은 평범한 `TypeError`를 일으킵니다). 반복된 레이블: `source.weights_duplicate`, 모두 생성 시. `Numerics(orders=...)` 밖의 차수, 또는 소멸하거나 스치는 차수: `rcwa.invalid`, run 시 |
| `side`, `polarization`, `amplitude`, `theta`, `phi`, `kx`, `ky` | `PlaneWave`와 같음 | 차수를 조명하는 기준 평면파 | `PlaneWave`와 같음 |

### `rcwa.GaussianBeam`

전파하는 가우시안 각 스펙트럼입니다. 유한 직사각형
`Outputs(power_region=((xmin, xmax), (ymin, ymax)))`가 필요하며, 없으면 `check`가 이를
`outputs.power_region`으로 거부합니다.

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `waist` | 필수 | 포락선의 1/e 진폭 반지름, 양의 길이 | 양수가 아님: `rcwa.invalid`, check 시 |
| `focus` | `(0., 0., 0.)` | 초점의 `(x, y, z)` | 성분이 세 개가 아님: `source.dimension`, 생성 시 |
| `sampling` | `(9, 9)` | 중심 방향에 수직인 두 축의 중점 구적점 개수 | 성분이 두 개가 아님: `source.dimension` |
| `angular_spectrum_extent` | `4.` | 그 구적의 반폭, `1/waist` 단위 | 양수이고 유한한 실수가 아님: `source.angular_spectrum_extent`, check 시 |
| `side`, `polarization`, `amplitude`, `theta`, `phi`, `kx`, `ky` | `PlaneWave`와 같음 | 중심 평면파. `amplitude`는 초점에서의 필드입니다 | `PlaneWave`와 같음 |

### `rcwa.ElectricDipole`

균일한 층 안에 지정된 쌍극자로, `exp(-iωt)`를 씁니다. 고립된 쌍극자에는 가우시안 빔과
마찬가지로 유한 직사각형 `power_region`이 필요합니다.

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `position` | 필수 | 균일한 층 안의 `(x, y, z)`: 영역이 없거나, 한 재료로 된 래스터 | 성분이 세 개가 아님: `source.dimension`, 생성 시. 균일하지 않은 호스트 층(영역, 또는 값이 둘 이상인 래스터): `excitation.sources[i]`에서의 `source.host_layer`, check 시 |
| `moment` | 필수 | `(px, py, pz)`, 복소수, C·m 단위 | 성분이 세 개가 아님: `source.dimension` |
| `repetition` | `'periodic'` | `'periodic'`, 셀마다 쌍극자 하나. 또는 `'isolated'`, 브릴루앙 영역 구적으로 표현한 쌍극자 하나 | 그 밖의 모든 것: `source.repetition`, 생성 시 |
| `bloch_k` | `None`, 0 | periodic 전용: `(kx, ky)`, 복제본 사이의 위상. `bloch_k + G`는 같은 배열이므로 역격자 벡터로 원점 중심 셀 안으로 환산됩니다 | `repetition='isolated'` 옆에서: `rcwa.invalid`, check 시 |
| `sampling` | `(5, 5)` | isolated 전용: 구적 격자. 유한한 격자는 더 큰 반복 슈퍼셀이므로 수렴시켜야 합니다 | |

## The numerics

[수치 설정 고르기](../guide/rcwa/numerics.md)에서 설명합니다.

### `rcwa.Numerics`

`None` 기본값은 문제마다 결정되며, `rcwa.check(...).resolved`가 그 결정 결과를 보고합니다.
아무것도 읽지 않는 곳에 설정한 옵션은 무시되지 않고 check 시 `rcwa.unsupported`로
거부됩니다.

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `orders` | `(0, 0)` | 유지하는 푸리에 창 `(Mu, Mv)`, 즉 두 격자 벡터 방향의 반폭이며 u가 먼저입니다: u 방향으로 `2*Mu + 1`개, v 방향으로 `2*Mv + 1`개의 차수. 음수가 아닌 정수(`int` 또는 NumPy 정수) 두 개로 된 명시적 튜플이나 리스트 | 그 밖의 모든 것 -- 맨 정수, `bool`, `dict`, 집합, range, 이터레이터 포함: `numerics.orders`, 생성 시 |
| `algorithm` | `'smm'` | 소문자로 `'smm'`(산란 행렬) 또는 `'etm'`(향상된 투과 행렬). 모드와 답은 같으며, 산란 행렬을 만들고 결맞는 소스를 실행하는 것은 `'smm'`뿐입니다 | `'ETM'`을 포함한 그 밖의 모든 것: `numerics.choice`, 생성 시. 면내 유전율이 모든 곳에서 실행의 기계 엡실론의 `sqrt`보다 작은 층에서, y 파동 벡터 성분이 있거나 평면파가 아닌 소스와 함께 쓴 `'smm'`: `rcwa.near_zero_permittivity_smm`, check 시([이유](../guide/rcwa/numerics.md#how-the-coefficients-are-obtained-and-how-the-products-are-closed)) |
| `fourier_coefficients` | `None`: 래스터에서는 `'sampled'`, 영역에서는 `'integrated'` | `'sampled'`(픽셀 격자에서 구한 급수) 또는 `'integrated'`(래스터 직사각형의 닫힌 형태 변환) | 그 밖의 모든 것: `numerics.choice`. 영역 층에서 `'sampled'`: `rcwa.unsupported`, check 시 |
| `factorization` | `None`: 래스터에서는 `'li'`, 영역에서는 `'nvf'` | `'li'`(기본 `li_nesting`에서 축이 y와 x에 정렬된 채 v, 그다음 u를 따라 적용하는 연속 규칙), `'nvf'`(법선 벡터장, 재료 계면을 따라 인수분해) 또는 `'laurent'`(모든 곱에 대한 직접 규칙 -- 명시적으로 켜야 하며, 이를 쓰는 솔버와 비교하기 위한 것). 텐서 지원은 [재료 가이드](../migration/materials-v1.md#anisotropic-materials)에 설명되어 있습니다 | 그 밖의 모든 것: `numerics.choice`. 축 정렬 직사각형이 아닌 영역에서 `'li'`, 텐서 재료에서 `'laurent'`: `rcwa.unsupported`, check 시 |
| `formulation` | `'auto'` | `'auto'`는 문제를 읽습니다. z 행이나 열이 있는 재료는 `'first_order'`를, 그 밖의 모든 것은 `'second_order'`를 취합니다. 필요하지 않은 문제에 `'first_order'`를 강제하는 것이 둘을 교차 검증하는 방법입니다 | 그 밖의 모든 것(`'2ff'`): `numerics.choice`. z 결합 텐서에서 `'second_order'`: `rcwa.invalid`, run 시. 그 행을 버리면 기계 정밀도까지 에너지를 보존하면서 다른 문제에 답하기 때문입니다 |
| `raster_normal_method` | `None`: `'distance'` | 래스터 NVF 전용: `'distance'`, `'contour'`, `'spectral'` 또는 `'smooth'` | 그 밖의 모든 것: `numerics.choice`. 영역에서 또는 `'li'` 아래에서 설정: `rcwa.unsupported`, check 시. 선택적 contourpy가 설치되지 않은 상태의 `'contour'`: `numerics.unavailable`, check 시 |
| `nvf_projector` | `None`: `'linear'` | 래스터 및 형상 NVF: `'linear'` 또는 `'circular'`. 입사 편광과 무관합니다 | 그 밖의 모든 것: `numerics.choice`. `'nvf'` 없이 설정, 또는 다각형에서 `'circular'`: `rcwa.unsupported`, check 시 |
| `shape_nvf_grid_size` | `None`: `max(1024, 4*Mu + 1, 4*Mv + 1)`. `resolved`는 래스터에서와 `'li'` 또는 `'laurent'` 아래에서는 `None`을, 그 밖에는 기본값이든 명시값이든 격자를 보고합니다. 층에 영역이 없는 적층도 포함됩니다. 이 적층은 `'nvf'`로 결정되고 격자를 보고하지만, 표본화할 계면이 없으므로 답은 격자에 의존하지 않습니다 | 형상 NVF 전용: 셀 축당 투영기 표본점 개수, 양의 `int`. `128`은 128×128 격자를 뜻합니다. 영역이 없는 층들의 적층에서도 받아들이며, 그곳에서는 답을 바꾸지 않습니다 | 양의 `int`가 아님: `numerics.grid`, 생성 시. 영역 층이 있는 적층에서 스칼라든 텐서 재료든 `max(4*Mu + 1, 4*Mv + 1)`보다 작은 값: 실행은 되지만, `check`와 `prepare` 보고에 `numerics.grid` 경고가, `prepare`와 `run`에서 `rcwa.diagnostics.ShapeGridWarning`이 나옵니다(측정된 오차는 그 클래스를 보세요. 격자는 정확도만 제어하며, `max(4*Mu + 1, 4*Mv + 1)`은 정확도 경계가 아닙니다). 영역이 없는 층들의 적층은 격자를 읽지 않으며 경고하지 않습니다. 래스터에서 또는 `'li'`나 `'laurent'` 아래에서 설정: `rcwa.unsupported`, check 시 |
| `li_nesting` | `None`: `'fixed'` | Li 전용: Li 규칙을 중첩하는 순서. `'fixed'`(모든 슬롯에서 `v`가 `u` 안쪽) 또는 `'symmetric'`(각 슬롯의 역규칙 축이 안쪽 -- x와 y를 바꾸는 것에 대해 공변). 래스터와 축 정렬 직사각형에서 읽습니다 | 그 밖의 모든 것: `numerics.choice`. `'nvf'`나 `'laurent'` 아래에서 설정, 또는 텐서 재료에서 `'symmetric'`(텐서 Li는 `'fixed'`만 만듭니다): `rcwa.unsupported`, check 시 |
| `shape_normal_method` | `None`: `'per_shape'` | 선형 투영기를 쓰는 `'nvf'` 아래의 영역 전용: 영역 층이 어떤 법선장을 만드는지. `'per_shape'`는 서로 겹치지 않는 직사각형, 원, 타원으로 된 층을 형상별로 닫힌 형태로 만들고, 칠하기가 해소하는 층(주기의 `1e-6`보다 많이 겹치는 영역, 또는 다각형이 있는 층)은 칠하기가 남긴 계면으로부터 만듭니다. `'interfaces'`는 모든 영역 층을 그 계면으로부터 만들므로, 타일과 같은 구조의 칠하기가 같은 답을 주고 영역이 겹치기 시작하는 곳에서 계단이 생기지 않습니다. 실험적: 이를 쓰는 실행은 `ExperimentalWarning`을 냅니다. 두 영역이 접촉하는 곳에서 계단이 생기며, 원이나 타원으로 된 층은 훨씬 더 많은 비용이 듭니다. [경계 모델링](../guide/rcwa/boundary-modeling.md#one-normal-field-for-every-layer-of-regions-experimental)을 보세요 | 그 밖의 모든 것: `numerics.choice`, 생성 시. 래스터에서, `'li'`나 `'laurent'` 아래에서, `nvf_projector='circular'`와 함께, 또는 텐서 재료와 함께 설정: `rcwa.unsupported`, check 시 |
| `detuning` | `None`: 요청한 점 그대로 | 양의 상대 간격. 요청한 점이 수치적으로 특이한 곳(라이트 라인 위의 차수)에서는 그만큼 떨어진 곳에서 계산을 다시 시도하며 -- 먼저 파장, 그다음 각도 -- `result.diagnostics['effective_wavelengths']`가 실제로 계산한 값을 기록합니다 | `0`, `True`, 또는 음수이거나 유한하지 않은 값: `numerics.detuning`, 생성 시. `jax.jit`나 `jax.vmap` 아래에서(즉시 실행 `jax.grad`는 PyTorch처럼 이를 실행합니다): `prepared.detuning_transform`. 결맞는 소스와 함께: `rcwa.coherent_detuning`, check 시 |
| `half_space_merge_tolerance` | `0.0`: 비트 단위로 동일한 재료만 | 층의 재료가 옆 반공간에 병합되려면 얼마나 가까워야 하는지를 나타내는 `[0, 1)` 범위의 상대 허용 오차. `0`은 모든 유전율과 투자율 값이 반공간과 비트 단위로 같은 층만, 그리고 층으로 계산하면 O(1)만큼 틀릴 수 있는 복소수 면내 파동 벡터에서, 또는 층을 계산할 수 없는 2D 경로에서 그 반공간을 정확히 스치는 경우에만 병합합니다. 다른 실수 각도는 비트를 유지합니다. 양수 값은 그 안에 드는 재료를 모든 각도에서 반공간의 재료 **로 간주**하며, 답은 그렇게 맞춘 장면의 답입니다. `result.diagnostics['half_space_merge']`가 해당 층을 알려 줍니다. [connecting algorithms §10](../theory/rcwa/connecting-algorithms.md#10-etms-labels-at-a-complex-in-plane-wavevector)을 보세요 | 음수, 1 미만이 아님, 또는 유한하지 않음: `numerics.half_space_merge_tolerance`, 생성 시. 결맞는 소스와 함께 양수: `rcwa.coherent_half_space_merge`, check 시. `IncidentOrders`와 함께, `jax.jit`나 `jax.vmap`이 추적하는 재료와 함께, 또는 재료 미분과 함께 양수: run 시 거부. 병합은 구체적인 재료로부터 결정되기 때문입니다(추적된 각도만 있는 것은 괜찮으며, jit된 함수 안에서 상수로 쓴 재료도 괜찮습니다) |
| `exceptional_points` | `'rebuild'` | `'rebuild'`: 예외점과 그 근처의 층을 정칙 기저로 다시 만들며, 그럴 수 없는 곳에서는 경고하거나(즉시 실행) NaN으로 답합니다(추적). `'off'`: 모든 경로(2차 및 1차 2D, 1D conical, 결맞는 소스)에서 즉시 실행이든 추적이든 검출, 재구성, 선별을 건너뛰는 고급 선택입니다 -- 주로 `jax.jit` 아래에서 더 빠르지만, 예외점과 그 근처에서는 경고 없이 답이 틀릴 수 있습니다. 예외점에서 떨어진 곳에서는 답이 반올림 수준까지 일치합니다. {ref}`예외점 §7 <ep-handling-off>`를 보세요 | 그 밖의 모든 것: `numerics.choice`, 생성 시 |
| `wall_resolution` | `None`: 셀 자체의 좌표 | 축 정렬 직사각형, 또는 원 하나나 축 정렬 타원 하나의 벽에서 계산 좌표의 기울기로, `(0, 1]` 범위의 숫자입니다. `0.02`(권장)는 모든 벽에서 좌표선을 50배 더 촘촘하게 놓습니다. 원이나 타원은 그에 맞춘 좌표에서 계산됩니다. 이를 쓰면 소멸 차수의 진폭은 0으로 보고되며, 경계 필드와 산란 행렬은 사용할 수 없습니다. [늘린 좌표로 벽 분해하기](../guide/rcwa/wall-resolution.md)를 보세요 | `(0, 1]` 밖의 값: `numerics.wall_resolution`, 생성 시. 2차 정식화의 `'smm'`에서 `factorization='li'` 아래, 등방성 비자성 재료, 무손실 유전체 반공간, 평면파 하나인 조건에서 축 정렬 직사각형, 또는 원 하나나 축 정렬 타원 하나(모든 패턴 층에서 같고 셀 가장자리까지 여유가 있어야 함) 이외의 모든 것: `rcwa.unsupported`, check 시. `jax.jit`, `jax.vmap`, 그래디언트 |

### NVF names changed on 2026-09-21

| 이전 키워드 | 현재 키워드 | 이유 |
| --- | --- | --- |
| `normal_method` | `raster_normal_method` | 래스터 전용 필드 구성임을 드러냅니다. |
| `normal_grid` | `shape_nvf_grid_size` | 재료 래스터나 푸리에 차수가 아니라 형상 NVF와 표본 개수임을 드러냅니다. |
| `projector` | `nvf_projector` | 입사 편광과 구별되는 NVF 투영임을 드러냅니다. |

현재 이름은 위의 `rcwa.Numerics` 필드이며, 이전 인터페이스의 `rcwa(...)`,
솔버 속성, `solve(...)` 같은 호출별 설정도 이 이름을 받았습니다. 이전 이름은
거부되며 별칭이 아닙니다. 기본값, 허용되는 옵션 값, 수치 알고리즘은
바뀌지 않았습니다. `linear` / `circular`는 여전히 투영 선택지이며,
`real` / `complex`로 이름이 바뀌지 않았습니다. 선형 형상 구성도 재료에 따라 달라지는 복소수
성분을 담을 수 있기 때문입니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

이전 `JobSpec`(`wavebit.compat.legacy`, 2026-10-15에 퇴역)은 스키마 버전 **10**을 썼습니다. 버전 7
문서라면 `numerics`에 이 키들이 있을 때 이름을 바꾸고 옵션 값은 그대로 두세요. 또한
버전 10으로 설정하기 전에 `stack.layers`를 뒤집고 [규약](common/conventions.md)의 좌표 마이그레이션을
검토하세요. 기능 매니페스트는 버전 **10**부터 새 옵션 키를 썼습니다.
이전 JobSpec 버전은 각자의 재료 및 좌표 마이그레이션도 필요합니다.
:::

## 실행 위치

[백엔드, 장치와 정밀도](../guide/common/backends.md)에서 설명합니다.

### `wavebit.Execution`

`rcwa.check`, `rcwa.prepare`, `rcwa.run`은 이 세 필드를 키워드로 받습니다. `Execution`은
`wavebit.Job`이 이들을 담는 방식입니다. 만들 때가 아니라 실행이 이를 사용할 때 검증됩니다.

| 필드 | 기본값 | 받는 값 | 거부하는 값 |
| --- | --- | --- | --- |
| `backend` | `'numpy'` | 정확히 `'numpy'`, `'jax'` 또는 `'torch'`, 또는 `wavebit.Backend` 멤버 | 그 밖의 모든 것(`'JAX'`, `1`): `execution.backend`. 설치되지 않은 extra: `execution.backend_unavailable`. `rcwa`에서 `'numpy'`나 `'jax'` 아래에, 문제의 어딘가에 PyTorch가 미분하는 PyTorch 텐서(`requires_grad`, 또는 순방향 모드나 `torch.func` 접선)가 있는 경우: 필드 경로와 함께 `execution.backend_autodiff`, check 시(grad가 필요 없는 텐서는 숫자로 읽힙니다) |
| `device` | `'cpu'` | 문자열: `'cpu'`. PyTorch에서는 `'cpu:0'`(유일한 CPU 장치. N > 0인 `'cpu:N'`은 존재하지 않는 장치입니다), `'cuda'` 또는 `'cuda:N'`도 가능. JAX에서는 `'gpu'` 또는 `'tpu'`도 가능하며, 선택적으로 `:N`을 붙입니다 | 문자열이 아님(`0`): `execution.device`. CPU가 아닌 NumPy, PyTorch의 `'mps'` 또는 JAX의 `'cuda'`: `execution.device`. 존재하지 않는 장치: `execution.device_unavailable` |
| `precision` | `'double'` | `'double'`(complex128) 또는 `'single'`(complex64) | 그 밖의 모든 것: `execution.precision`. `jax.config.update("jax_enable_x64", True)` 이전의 JAX `'double'`: `execution.jax_x64` |

## 두 축 순서 규칙과, 둘이 만나는 곳


**모든 배열 축은 `v`가 먼저입니다. 손으로 쓰는 모든 쌍은 `u`(`x`)가 먼저입니다.** 어느 쪽에도
예외가 없습니다. 둘은 손으로 쓰는 쌍 `(Mu, Mv)`인 `Numerics(orders=...)`에서 만납니다.

```python
numerics = rcwa.Numerics(orders=(4, 1))        # (Mu, Mv): u first
result.R.shape                                  # (1, 3, 9) -- wavelength, then v, then u;
                                                # the leading 4 sized the TRAILING axis
result.reflection.orders_u, result.reflection.orders_v  # the order each column and each row is
```

`result.R.shape[1:] == (len(orders_v), len(orders_u))`이므로, 둘을 잘못된 방향으로 짝지으면
그럴듯한 숫자가 아니라 길이 불일치가 납니다. `u`는 셀의 첫 번째 기저 벡터를 따라,
`v`는 두 번째를 따라 진행합니다 — 직사각형 셀에서는 x와 y입니다. `result.reflection.order(m, n)`은
차수로 직접 인덱싱하므로 이 중 아무것도 필요하지 않습니다. `orders`는 쌍만 받습니다. 이전
`truncation_order`의 이름 붙은 `dict(u=..., v=...)` 표기에는 현재 대응하는 형태가 없습니다.

같은 규칙이 모든 공개 `orders=`에 적용됩니다: `rcwa.Numerics`와 `fdtd.periodic.Numerics`
(`numerics.orders`), `fdtd.DiffractionMonitor`(`monitor.orders`, 생성 시), 그리고
`rcwa.experimental` 빌더(`experimental.orders`). 각각은 NumPy 정수를 포함해 음수가 아닌 정수의 명시적 쌍을 받고,
맨 정수와 `bool`을 거부합니다. 맨 `M`은 FDTD 모니터와 실험적 빌더에는
`(M, M)`을, `fdtd.periodic`에는 `(M, 0)`을 뜻했으므로, 의도한 쌍을 쓰세요(1차원 주기는
`(M, 0)`을 받습니다). `None`은, 필드의 기본값이 그것인 곳(`fdtd.DiffractionMonitor`, `fdtd.periodic.Numerics`)에서
그 필드의 기본 창을 유지합니다.

## 이름 있는 상수


`wavebit.Backend`는 현재 모듈이 내보내는 유일한 이름 있는 상수입니다. 그 멤버는 자신의
문자열과 같으며(`Backend.TORCH == 'torch'`), 백엔드를 받는 곳이면 어디서든 받아들여집니다.

```python
from wavebit import Backend, rcwa

result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 2), factorization='nvf'),
                  backend=Backend.TORCH)
```

`wavebit.Job` 안에는 문자열을 쓰세요. `Job.to_json()`과 `Job.key`는
`Execution(backend=Backend.TORCH)`를 "Backend is not portable job data"와 함께 거부합니다. `rcwa.Numerics`
필드는 문자열 리터럴을 받습니다. 이전 인터페이스의 다른 상수는 2026-10-15에 퇴역한
`wavebit.compat.legacy`에서만 import할 수 있었습니다. `Answers`와 `Tetm`은 현재 필드가 없는 선택지를 가리킵니다. 나머지
여섯 -- `FourierCoefficients`, `Factorization`, `NormalMethod`, `Projector`, `Nesting`,
`Formulation` -- 은 자신의 문자열과 같으므로 `rcwa.Numerics`는 이를 받지만, `Job`은
같은 방식으로 거부합니다.

## Refusal codes

모든 구조화된 실패 -- `ValidationError`, `UnsupportedFeatureError`,
`IncompatibleProblemError` 또는 `NumericalError`, 그리고 `check` 보고가 담는 `Issue` -- 는
하나의 형식 `area.item`의 `.code`를 가집니다. 점으로 구분된 둘 이상의 소문자 부분으로, 각 부분은 snake-case 단어
(`[a-z][a-z0-9_]*`)입니다. area는 거부의 `.path`가 가리키는 선언이나 하위 시스템
(`numerics`, `source`, `monitor`, `execution`, `prepared`, ...)이며, item은 그것의 무엇이
잘못되었는지를 말합니다. 코드는 정확히 일치시켜 비교합니다. `.path`는 위치를, 메시지는 바뀔 수 있는
말로 이유를 알려 줍니다.

각 솔버 모듈에는 특정 코드가 기술하지 않는 엔진 거부를 위한 포괄 코드 쌍이 있으며,
엔진의 메시지를 유지합니다: `rcwa.invalid` / `rcwa.unsupported`, `fdtd.invalid` /
`fdtd.unsupported`, `native.invalid` / `native.unsupported`(`fdtd.native`),
`periodic.invalid` / `periodic.unsupported`(`fdtd.periodic`). `.unsupported` 코드는
`UnsupportedFeatureError`(엔진의 `NotImplementedError`)이고, `.invalid` 코드는
`ValidationError`입니다. `check`는 `prepare`, `run`, `Prepared.run`이 일으키는 것과 같은 코드를 보고합니다.
예외는 경계 계획기가 해소할 수 없는 칠하기입니다. `check`는 칠하기를 계획하지 않습니다(위의
`regions`를 보세요).
코드 없이 생성된 실패는 그 클래스의 기본값을 취합니다: `input.invalid`
(`ValidationError`), `feature.unsupported`(`UnsupportedFeatureError`), `problem.incompatible`
(`IncompatibleProblemError`) 또는 `numerical.failure`(`NumericalError`). Wavebit의 어떤 거부도
이 기본값에 의존하지 않습니다.

`prepared.structure`는 모든 솔버에서, 고정 레이아웃이 준비 단계에서 고정한 레이아웃과
다른 문제에 대한 유일한 코드입니다. 그 `.path`는 바뀐 선언을 가리킵니다
(`scene.domain`, `sources.<name>`, `initial.E`, `stack.layers[1].regions`, `units.length`, ...).

두 회절 기준 코드는 물리량마다 이름 하나를 지킵니다
([진폭비, 필드와 스펙트럼](../migration/public-names.md#amplitude-ratios-fields-and-spectra)).
`observation.reference_key`는 이름이 바뀐 키(`incident_amplitude`, `incident_amplitude_units`,
`amplitude_units`)를 아직 담고 있는 `reference`를 거부하며, 그 메시지는
새 이름을 알려 줍니다. `observation.amplitude_units`는 키가 허용하는 것이 아닌 단위 레이블
(`amplitude_ratio_units` `'1'`, `spectrum_units` `'V*s/m'`)과, 옆에 레이블 없이 주어진 `incident_field` 또는
`incident_spectrum`(`incident_field_units` `'V/m'`,
`incident_spectrum_units` `'V*s/m'`)을 거부합니다.

| 영역 | 코드 | 일으키는 곳 |
| --- | --- | --- |
| `boundaries` | `boundaries.face`(생성자가 거부하는 `MirrorSymmetry`나 `InvariantAxis`도 해당) | `fdtd` |
| `check`, `validate`, `prepare` | `check.concrete`, `validate.concrete`, `prepare.concrete` | `rcwa` |
| `decay_check` | `decay_check.threshold`, `decay_check.window_seconds` | `fdtd.native`, `fdtd.periodic` |
| `dispersion` | `dispersion.poles` | `fdtd.periodic` |
| `excitation` | `excitation.sources` | `rcwa`, `fdtd.periodic` |
| `execution` | `execution.backend`, `execution.backend_unavailable`, `execution.device`, `execution.device_unavailable`, `execution.precision`, `execution.jax_x64`, `execution.backend_autodiff`(`rcwa`). `execution.cuda_scope`(검증된 FDTD CUDA 경로 밖) | 모든 솔버. `fdtd` |
| `experimental` | `experimental.orders` | `rcwa.experimental` |
| `fdtd` | `fdtd.invalid`(생성자가 거부하는 `Domain`, `Plane` 또는 `TimeSeriesMonitor` 이름이나 성분도 해당), `fdtd.unsupported`, `fdtd.padding_continuation`, `fdtd.torch_transform`(`torch.func.vmap`, `jacfwd` 또는 `hessian` 아래의 Torch 실행, 또는 `jacrev` 아래의 체크포인트 실행) | `fdtd` |
| `field` | `field.coordinates`, `field.side`, `field.part`(`result.boundary_fields`), `field.reconstruction` | `rcwa` |
| `geometry` | `geometry.cell_snapped`(경고) | `fdtd` |
| `initial` | `initial.type` | `fdtd` |
| `job`, `record` | `job.encoding`(생성자가 거부하는 저장된 선언이나 재료 표도 해당), `job.schema`, `record.format` | `wavebit.Job`, 실행 기록 |
| `layer` | `layer.ambiguous`, `layer.mu_without_raster`, `layer.raster_shape`, `layer.regions`, `layer.thickness` | `rcwa` |
| `material` | `material.type`. `material.scalar`, `material.shape`, `material.poles`, `material.wavelength_range`, `material.unknown_field`(`material.scalar`, `material.shape`, `material.poles`는 생성 시. [재료 가이드](../guide/common/materials.md)를 보세요). `material.engine_model`, `material.causal_model`, `material.pole_type`, `material.fdtd`(재료를 시간 영역으로 변환할 때) | `rcwa`, `fdtd.periodic`. `wavebit.materials`. `fdtd` |
| `medium` | `medium.poles` | `fdtd.native` |
| `migration` | `migration.shape`, `migration.shape_fields`, `migration.layer_fields`, `migration.layer_mu`, `migration.solver` | `wavebit.migration` |
| `monitor` | `monitor.type`, `monitor.geometry`, `monitor.plane`, `monitor.plane_span`, `monitor.extent`, `monitor.outside_domain`, `monitor.source`, `monitor.orders`(`monitor.geometry`는 점이 아닌 `TimeSeriesMonitor(at=...)`에도 해당) | `fdtd` |
| `native` | `native.invalid`, `native.unsupported` | `fdtd.native` |
| `numerics` | `numerics.type`, `numerics.choice`, `numerics.orders`, `numerics.grid`, `numerics.detuning`, `numerics.half_space_merge_tolerance`, `numerics.unavailable`(선택적 의존성이 설치되지 않은 옵션) | `rcwa`(`numerics.type`, `numerics.choice`, `numerics.orders`는 FDTD에서도) |
| `numerics` (FDTD) | `numerics.absorber`(매핑이 아닌 `PML(reference_media=...)`도 해당. 거부되는 `ConductiveAbsorber(cells=...)`는 `numerics.absorber.cells`에서. 그리고 `numerics.absorber`에서, 지지 영역이 결합이 허용하는 범위 밖의 `ConductiveAbsorber` 안까지 닿는 스무딩된 구조는 `check`, `prepare`, `run`, `Prepared.validate`에서 똑같이 거부), `numerics.absorber_reflection`(경고), `numerics.checkpoint_steps`, `numerics.checkpoint_scope`, `numerics.discretization`, `numerics.duration`, `numerics.geometry_ranges`, `numerics.junction`, `numerics.memory_bound`, `numerics.stopping`. `numerics.time_step_seconds`. `numerics.spacing`, `numerics.duration_seconds`, `numerics.transverse_spacing`, `numerics.reference_scope` | `fdtd`. `fdtd.native`. `fdtd.periodic` |
| `observation` | `observation.empty`, `observation.order_labels`, `observation.shape`, `observation.wavelength_axis`. `observation.amplitude_units`, `observation.reference_key` | `wavebit.DiffractionOrders`, `wavebit.RawDiffractionSpectrum`. 같은 것과 그 단일 차수 기록 `wavebit.observations.DiffractionOrder` 및 `RawDiffractionOrder` |
| `operators` | `operators.type`, `operators.shape`, `operators.slots`, `operators.spectral_shape`, `operators.source_groups`, `operators.detuning`, `operators.execution`, `operators.physical_contract`, `operators.physical_transform` | `rcwa.operators` |
| `outputs` | `outputs.boolean`, `outputs.unavailable`, `outputs.power_region`(`'cell'`, `'infinite'` 또는 양의 폭을 가진 직사각형이 아닌 결맞는 소스 `power_region`. 여러 Bloch 섹터에 걸친 `None`. 유한 직사각형 `power_region`이 없는 `GaussianBeam`이나 고립된 `ElectricDipole`) | `rcwa` |
| `periodic` | `periodic.invalid`, `periodic.unsupported` | `fdtd.periodic` |
| `prepared` | `prepared.structure`. `prepared.static_input`, `prepared.dependencies`, `prepared.pml_wavelength`. `prepared.topology`, `prepared.host_topology`, `prepared.host_topology_scope`, `prepared.source_grouping`, `prepared.detuning_transform`, `prepared.shape_topology`, `prepared.rectilinear_topology`, 그리고 런타임 가드 `prepared.finite_inputs`(NaN이나 무한대인 실행 시점 값. 값을 읽을 수 있으면 그 경로가 필드를 가리킵니다), `prepared.positive_wavelength`, `prepared.nonnegative_thickness`, `prepared.nonzero_polarization`, `prepared.raster_partition` | 모든 솔버. `fdtd`. `rcwa` |
| `problem` | `problem.type`, `problem.leaf`, `problem.nonfinite`(NaN이나 무한대인 재료 값, 래스터 표본 또는 그 밖의 수치 입력으로, 그것을 담은 필드의 경로와 함께 `check`와 `prepare`에서 거부됩니다. `fdtd`는 구조, 소스 또는 모니터를 이름으로 가리킵니다: `scene.structures.core.material.epsilon`, `sources.pw.pulse.wavelength`. NaN이나 무한대인 `CurrentDipole` 중심, 소스 평면 위치 또는 `TimeSeriesMonitor` 점은 `source.outside_domain`이나 `monitor.outside_domain`이 아니라, 도메인 밖의 유한한 점이 거부되는 경로 `sources.d.center`, `outputs.probe`에서의 `problem.nonfinite`입니다. NaN이나 무한대인 FDTD `PlaneWave`의 `theta` 또는 `phi`는, `phi`가 여전히 편광 좌표계를 정하는 수직 입사에서도, `sources.pw.phi`에서의 `problem.nonfinite`입니다. `PlaneWave`, `IncidentOrders` 또는 `GaussianBeam`의 NaN이나 무한대인 RCWA `theta` 또는 `phi`는 스칼라든 배열이든 `source.direction`이 아니라 `excitation.sources[0].phi`에서의 `problem.nonfinite`입니다. NaN이나 무한대인 RCWA `kx` 또는 `ky`는 `excitation.sources[0].kx`에서의 `problem.nonfinite`입니다) | 모든 솔버. `rcwa`, `fdtd`, `fdtd.periodic` |
| `pulse` | `pulse.frequency_sigma_hz` | `fdtd.periodic` |
| `rcwa` | `rcwa.invalid`, `rcwa.unsupported`, `rcwa.singular`, `rcwa.nonfinite`, `rcwa.host_geometry`, `rcwa.mixed_layer_encoding`, `rcwa.raster_partition_transform`, `rcwa.scattering_algorithm`, `rcwa.coherent_detuning`, `rcwa.coherent_diffraction`, `rcwa.coherent_half_space_merge`, `rcwa.coherent_scattering`, `rcwa.coherent_transform` | `rcwa` |
| `scene` | `scene.type`, `scene.structure_name`, `scene.structure_name_duplicate` | `fdtd` |
| `shape` | `shape.type`, `shape.dimension`, `shape.value`, `shape.outline`, `shape.vertices` | `rcwa`, `fdtd`(`shape.type`. 생성자가 거부하는 `SnappedSlab` 구간이나 `SnappedBox` 크기에는 `shape.dimension`) |
| `snapshots` | `snapshots.every`, `snapshots.components`, `snapshots.slices` | `fdtd.native`, `fdtd.periodic` |
| `source` | `source.type`, `source.name`, `source.name_duplicate`, `source.pulse`, `source.plane`, `source.plane_span`, `source.box`, `source.outside_domain`, `source.waveform_range`. `source.amplitude`, `source.direction`, `source.polarization_value`. `source.side`, `source.repetition`, `source.dimension`, `source.direction_ambiguous`, `source.polarization`, `source.polarization_shape`, `source.weights`, `source.weights_duplicate`, `source.angular_spectrum_extent`, `source.host_layer`(한 재료로 되어 있지 않은 층 안의 `ElectricDipole`) | `fdtd`. `rcwa`와 `fdtd.periodic`. `rcwa` |
| `stack` | `stack.layers`, `stack.period`, `stack.cell_ambiguous`, `stack.mixed_layer_encoding` | `rcwa`, `fdtd.periodic` |
| `structure` | `structure.component`(스칼라 층에서는 `None`, 텐서 층에서는 `'xy'` 같은 xyz 쌍 하나), `structure.concrete`, `structure.layer`, `structure.nonfinite`, `structure.quantity`, `structure.samples`, `structure.wavelength` | `rcwa.sample_structure` |
| `units` | `units.type`, `units.length` | 모든 솔버 |
| `wavelength` | `wavelength.value`, `wavelength.empty`, `wavelength.shape` | `rcwa`, `fdtd.periodic`, `wavebit.materials`(`TabulatedMaterial`의 `wavelengths`, 생성 시) |

## Earlier spellings

이전 `rcwa(...)` 생성자의 각 키워드와, 실행에서 그 자리를 대신한 솔버 속성 및 호출별
설정을, 지금 그것을 담는 선언 필드와 대조합니다. 장면, 빛, 수치 설정 행은
한 문제를 두 방식으로 계산해 확인했습니다. 효율, 진폭, 산란 행렬이 반올림 오차(3e-13 이하)까지
일치합니다. [이전 API 레퍼런스](../migration/legacy-api.md)가 생성자 자체를 문서화합니다.

| 이전 키워드 | 현재 위치 | 참고 |
| --- | --- | --- |
| `backend` | `Execution(backend=...)`, 또는 `rcwa.check` / `prepare` / `run`의 `backend=` | 같은 세 이름 |
| `device` | `Execution(device=...)` | 문자열만 가능. 정수 `0` / `1`과 백엔드 자체의 장치 객체는 거부됩니다 |
| `complex_dtype` | `Execution(precision=...)` | `'double'` / `'single'`. `0` / `1`과 각 네임스페이스의 dtype은 거부됩니다 |
| `wavelength` | `Excitation(wavelength=...)` | 이제 스펙트럼일 수 있습니다 |
| `period` | `Stack(period=...)` | 쌍만 가능. `dict(x=..., y=...)`는 거부됩니다 |
| `lattice` | `Stack(lattice=...)` | |
| `structure`, 래스터 | `Layer(epsilon=...)`, 층마다 `(Ny, Nx)` 래스터 하나 | |
| `structure`, 형상 레시피 목록 | `Layer(background=..., regions=(Region(geometry=..., material=...), ...))` | `cx`, `cy` → `center=(cx, cy)`. `'rectangle'` → `Rectangle(size=(lx, ly), angle=...)`, `'circle'` → `Circle(radius=lx/2)`, `'ellipse'` → `Ellipse(radii=(lx/2, ly/2), angle=...)`, `'polygon'` → `Polygon(vertices=...)` |
| `thickness` | `Layer(thickness=...)`, 층마다 하나 | |
| `mu` | `Layer(mu=...)`, `epsilon` 옆에 | |
| `top` | `Stack(top=...)` | 재료 모델. 맨 유전율 약식은 거부됩니다 |
| `bot` | `Stack(bottom=...)` | `top`과 같음 |
| `incidence_side` | `PlaneWave(side=...)` | |
| `theta`, `phi` | `PlaneWave(theta=..., phi=...)` | |
| `kx`, `ky` | `PlaneWave(kx=..., ky=...)` | |
| `polarization` | `PlaneWave(polarization=...)` | 편광이 고정되어 있음을 선언하는 `'s'`와 `'p'`도 가능(`tetm` 행 참고) |
| `psi`, `chi` | **없음** | 같은 빛은 존스 쌍 `polarization=(sin ψ cos χ + i cos ψ sin χ, cos ψ cos χ − i sin ψ sin χ)`(`chi = 0`이면 `(sin ψ, cos ψ)`)이며, 반올림 오차까지 일치합니다. 쌍의 중복된 크기와 위상이 없는 두 각도 좌표계 자체에는 필드가 없습니다 |
| `tetm` | `PlaneWave(polarization='s')` / `'p'` | 단어가 곧 선언입니다. `'s'`는 `'te'`, `'p'`는 `'tm'`입니다. 축 정렬 셀의 한 행짜리 래스터에서, `phi`와 `ky`가 없거나 추적되지도 미분되지도 않는 0이면 실행은 순수 TE/TM 경로를 타며, `report.resolved['solve_path']`와 `result.diagnostics['solve_path']`가 `'1d_te'` / `'1d_tm'`이라고 알려 줍니다. `tetm`과 달리 이 단어는 아무것도 거부하지 않습니다. 기울어진 셀, 0이 아니거나 추적되는 `phi` 또는 `ky`, `formulation='first_order'`, `Outputs(scattering=True)`는 그 단어가 가리키는 존스 쌍으로 conical 또는 2D 경로에서 계산됩니다. 존스 쌍은 항상 쌍으로 계산됩니다. 2026-11-03부터입니다. 그 전에는 모든 한 행짜리 래스터가 conical 경로에서 계산되었습니다. 두 경로는 반올림 오차까지 일치합니다(`tests/api/test_unified_rcwa_pure_1d.py` 전체에서 상대 2e-13) |
| `excitation`, 차수 맵 | `IncidentOrders(weights=(((m, n), weight), ...))` | 1D의 `{0: 1.0}`은 `((0, 0), 1.0)`이 됩니다. 결맞는 소스 실행입니다. 총 경계 전력은 이전의 차수별 합과 일치하지만, **차수별 효율에는 현재 대응하는 형태가 없습니다**(`rcwa.coherent_diffraction`) |
| `excitation`, 방향을 키로 쓰는 맵 또는 소스 목록 | `Excitation(sources=(...))` | |
| `stack=`, `incidence=` | `rcwa.Stack`, `rcwa.PlaneWave` | |
| `truncation_order` | `Numerics(orders=...)` | 음수가 아닌 정수의 쌍 `(Mu, Mv)`만 가능. 맨 정수와 `dict` 표기는 거부됩니다 |
| `layer_algorithm` | `Numerics(algorithm=...)` | 소문자. **기본값이 ETM에서 `'smm'`으로 바뀌었습니다** |
| `fourier_coefficients`, `factorization`, `raster_normal_method`, `nvf_projector`, `li_nesting`, `shape_nvf_grid_size`, `shape_normal_method` | `Numerics(...)`, 같은 이름 | 같은 값과 기본값 |
| `formulation`, 솔버 속성 | `Numerics(formulation=...)` | |
| `solve(auto_detune=...)` | `Numerics(detuning=...)` | **기본값이 바뀌었습니다.** 이전 `solve()`는 상대 간격 `1e-12`로 다시 시도했지만, `Numerics()`는 요청한 점을 계산합니다. `detuning=1e-12`는 이전의 기본 간격을 요청합니다 |
| `answers` | `Outputs(fields=...)` | `'field'`는 `fields=True`이고, `'amplitude'`와 `'efficiency'`는 둘 다 `fields=False`입니다. 어느 쪽이든 결과는 진폭비를 담습니다 |
| `solver.scattering` | `Outputs(scattering=True)`, 그다음 `result.scattering` | 파장마다 행렬 하나 |
| `power_region` | `Outputs(power_region=...)` | |
| `length_unit` | `Problem(units=Units(length=...))` | 자유로운 인자가 아니라 `'m'`, `'mm'`, `'um'`, `'nm'` 중 하나 |

두 이전 키워드에는 현재 위치가 없습니다. `psi`와 `chi`이며, 그 빛은 해당 행의 존스 쌍입니다.
차수 맵 `excitation`에는 위치가 있지만, 차수별 효율은 없습니다. 이들을 가졌던 이전
인터페이스 `wavebit.compat`는 2026-10-15에 퇴역했습니다. 그것은 [이전 API 레퍼런스](../migration/legacy-api.md)가
지목하는 과거 체크아웃에만 남아 있으므로, 문제를 실행하는 곳이 아니라 읽어 보는
기록입니다.
