# 데이터로서의 실행

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 있어야 동작합니다.
> 현재 설치본에는 이 네임스페이스가 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행하지 않았습니다.


:::{note}
이 페이지는 예전 인터페이스의 전체 호환성 레퍼런스입니다. `wavebit.compat.legacy`를 명시적으로 쓰세요. 새 애플리케이션은 [현재 생애주기](../guide/common/lifecycle.md)와 [현재 API 레퍼런스](../reference/api.md)로 시작하세요. 여기에 인용한 수치 근거는 원래의 적용 범위를 그대로 유지합니다.
:::

`wavebit.rcwa()`는 배열, 열거형(enum), 부동소수점 값 같은 Python 값으로 *솔버 객체*를 만듭니다.
스크립트에는 알맞은 형태이지만, 프로세스나 큐, 네트워크를 건너가야 하는 경우에는 맞지 않는
형태입니다. `JobSpec`은 그 반대쪽 형태입니다.

```
JobSpec (plain data)  --->  run(spec)  --->  Result
```

JSON으로 왕복 변환할 수 있고 실행을 완전히 결정하므로, 솔버 객체로는 얻을 수 없는 세 가지를
줍니다.

* **계약.** 스크립트, 분산 작업자(worker), HTTP 클라이언트 등 모든 호출자가 하나의 스키마를
  공유하므로 서로 어긋날 수 없습니다.
* **재현성.** `JobSpec.key()`는 정규화된 spec을 해시하므로, 같은 실행을 다시 풀지 않고도 같은
  실행임을 식별할 수 있습니다.
* **안전성.** 선언적인 spec을 받는 것은 임의의 Python 코드를 받는 것이 아닙니다. spec을 받는
  서비스는 호출자의 코드를 실행하지 않습니다.

## 구성

네 부분으로 이루어집니다. 물리적 장면은 [`scene.md`](../guide/rcwa/scene.md)가 설명하는 것과 같은
`Stack` / `Incidence` 객체를 다시 씁니다. `numerics`는 장면에 속하지 않는 이산화 선택을 담고,
`sweep`은 무엇이 변하는지 선언합니다.

| 필드 | 담는 것 |
| --- | --- |
| `stack` | `layers`(각각 `Layer(thickness, structure, mu)`), `top`, `bot`, `period`, `lattice` |
| `incidence` | `wavelength`, `theta`, `phi`, `kx`, `ky`, `tetm`, `psi`, `chi`, `incidence_side`, `polarization` |
| `numerics` | `backend`, `truncation_order`, `fourier_coefficients`, `factorization`, `raster_normal_method`, `nvf_projector`, `li_nesting`, `complex_dtype`, `layer_algorithm`, `shape_nvf_grid_size`, `answers` |
| `sweep` | `axes` — 해석 후의 `{field: [values]}` |

`solver`(`'rcwa'`)와 `version`이 나머지를 채웁니다.

현재 스키마 버전은 **10**입니다. 버전 7은 `numerics` 안에서 `normal_method`, `normal_grid`,
`projector`를 썼습니다. 이 키들을 각각 `raster_normal_method`, `shape_nvf_grid_size`,
`nvf_projector`로 바꾸고, 아래에 설명한, 층을 아래에서 위로 쌓는 좌표로의 마이그레이션을 적용한 다음 `version='10'`으로 설정하세요.
NVF 옵션 값 자체는 바뀌지 않았습니다.

**spec은 RCWA 실행을 기술하며, 그것만 기술합니다.** `wavebit.fdtd()`
([`time-domain.md`](../guide/fdtd/time-domain.md))는 두 번째 솔버이지만 두 번째 `solver` 값은 아닙니다. 위
`numerics`의 모든 필드는 조화(harmonic) 계산의 용어입니다. 즉 절단 차수, 인수분해, 법선장 같은
것들입니다. 반면 시간 영역 실행은 격자, 지속 시간, 흡수층으로 기술됩니다. 이 스키마에는 그런 것을
담는 자리가 없으므로, `solver='fdtd'`는 왕복 변환은 되지만 실행하면 실패할 spec으로 받아들이지 않고
이름을 들어 거부합니다.

생성자와 다른 점 두 가지는 의도한 것입니다. `numerics.complex_dtype`은 dtype 객체가 아니라
**비트 폭**(`128` / `64`)이므로 spec이 백엔드 중립을 유지합니다. 그리고 `answers`는 빛이 아니라
작업(job)을 기술하므로 `incidence`가 아니라 `numerics`에 있습니다.

열거형 타입의 `numerics` 필드(`factorization`, `li_nesting` 등)는 `Numerics(li_nesting='symmetric')`처럼
열거형 멤버나 그 이름을 대소문자 구분 없이 받습니다. `backend`는 `wavebit.rcwa`와 똑같이 열거형
멤버나 그 이름을 받습니다. 즉 `'numpy'`, `'jax'`, `'torch'`이며 정수 코드는 받지 않습니다.
생성 시점에 `Numerics.from_dict`와 똑같이 정규화하고, 나중의 대입(`numerics.li_nesting = 'symmetric'`)도
마찬가지로 정규화합니다. 따라서 Python에서 만든 spec, 만든 뒤에 편집한 같은 spec, JSON에서 파싱한
같은 spec은 서로 같고 하나의 `key()`를 공유합니다. 멤버가 아닌 이름은 생성이나 대입 시점에 JSON
입구가 내는 것과 같은 메시지로 거부되며, 거부된 대입은 필드를 원래대로 둡니다.

## 하나 실행하기

```python
import numpy as np
from wavebit.compat.legacy import JobSpec, Numerics, Sweep, Layer, Stack, Incidence, run

eps_hi = 12.25
spec = JobSpec(
    stack=Stack(layers=[Layer(thickness=300.,
                              structure=np.array([[1.0, 1.0, eps_hi, eps_hi]]))],
                top=1.0, bot=2.25, period=(700., 700.)),
    incidence=Incidence(wavelength=900., theta=0., phi=0., tetm='te'),
    numerics=Numerics(truncation_order=(10, 0)),
    sweep=Sweep.from_dict({'wavelength': {'linspace': [800., 1000., 3]}}),
)

for point in spec.expand():
    result = run(point)
    print(point.incidence.wavelength, result.R.sum(), result.T.sum())
```

```
800.0   0.460323   0.539677
900.0   0.205332   0.794668
1000.0  0.171514   0.828486
```

래스터 층은 반드시 **배열**이어야 합니다. `Layer.structure`의 중첩 `list`는 해석적 표기로 읽혀
설명과 함께 거부됩니다. 올바른 거부이지만, 격자를 의도했다면 뜻밖의 거부입니다.

## 스윕

스윕은 spec *안에* 선언하므로, "파장 501개"는 거의 똑같은 객체 501개가 아니라 작은 객체 하나로
전달됩니다. `expand()`는 이를 다시 격자점마다 스윕 없는 spec 하나로 펼칩니다.

```python
Sweep.from_dict({
    'wavelength': {'linspace': [800., 1000., 3]},   # start, stop, num
    'theta':      [0.0, 0.1, 0.2],                  # a bare list is shorthand for {'values': [...]}
})
```

스윕할 수 있는 필드는 `wavelength`, `theta`, `phi`, `psi`, `chi`, `kx`, `ky`의 일곱 가지이며, 그 밖의 것은
이름을 들어 거부합니다. 여러 축은 완전한 **데카르트 곱**을 이룹니다. 매개변수 스윕이란 원래 그런
뜻이므로 zip이 아니라 외적(outer product)입니다. `len(spec)`이 점의 개수입니다.

이 점들을 프로세스나 노드에 나누어 보내는 것은 평범한 일입니다. 서로 독립적인 풀이이기 때문이며, 세
백엔드 모두에서 똑같이 동작합니다.

## 직렬화

```python
text = spec.to_json(indent=2)
back = JobSpec.from_json(text)
assert back.key() == spec.key()          # normalised hash, so a round trip is identifiable
```

`key()`는 텍스트가 아니라 정규화된 spec을 해시하므로 서식은 값을 바꾸지 않습니다. 그래서 캐시 키로
쓸 수 있습니다.

## 문서에 반드시 적어야 하는 것과 거부되는 것

스키마에는 버전이 있으며, 관문은 일부러 엄격합니다.

| 적은 내용 | 결과 |
| --- | --- |
| `version` 없음 | `SpecError` — 파싱된 문서는 자신이 어떤 스키마로 쓰였는지 밝혀야 합니다. 버전이 없을 때 "현재"로 간주하면, 낡았을 가능성이 가장 큰 바로 그 문서들에 대해 관문이 아무 일도 하지 않게 됩니다 |
| `version: '2'` | `SpecError` — 이 빌드는 버전 `'10'`을 이해합니다 |
| `outputs: [...]` | `SpecError` — 버전 4에서 제거되었으며, 알 수 없는 필드가 아니라 이름을 들어 거부합니다 |
| 알 수 없는 필드 | `SpecError` — 가장 가까운 필드를 알려 주고(`thet` -> `theta`) 허용되는 것을 나열합니다 |
| `stack.n_top: 1.5` | 가장 가까운 필드를 알려 주지 *않는* `SpecError`: 굴절률(`n_top`, `ntop`, `n_sub`, `n1`, `RI_top`, `*_index`, `refractive*`)이나 투자율(`mu_*`)처럼 보이는 이름은 다른 물리량을 담는 자리로 안내하지 않습니다. `top`은 `1.5`를 유전율로 받아 풀어 버리기 때문입니다. 대신 재료 자리는 상대 유전율(eps = n**2)이나 `Material`이라고 알려 줍니다. `n`은 단독으로 쓰였거나, 번호가 붙었거나, 매질을 가리키는 이름(`top`, `sub`, `bg`, ...)이나 굴절률의 일부(`eff`, `real`) 옆에 있을 때만 굴절률로 읽습니다. `n_orders`나 `n_theta`는 개수이며 가장 가까운 필드 안내를 유지합니다 |
| 문자열이 아닌 키 | `SpecError` — 필드 이름은 문자열입니다 |

Python에서 spec을 만드는 것은 격식 없는 경로입니다. `JobSpec(...)`은 그것을 생성한 빌드의 버전을
가집니다. 이 클래스들로 생성한 spec은 낡았을 수가 없기 때문입니다. *파싱된* 문서만 자신을 밝혀야
합니다.

객체는 생성 후에도 같은 어휘를 지켜야 합니다. 필드가 아닌 이름에 대입하면(`spec.incidence.thet = 0.2`,
`spec.numerics.nesting = 'symmetric'`) 솔버가 `solver.thicknes = ...`에 대해 하는 것과 똑같이, 같은 근접
일치 힌트와 함께 `AttributeError`를 냅니다. 물리량을 넘나드는 추측을 거부하는 것도 같습니다
(`stack.n_top = 1.5`는 `top`으로 안내되지 않습니다). 그런 값을 저장하면 아무 신호 없이 묻힙니다. 실행은
자신이 아는 필드만 읽으므로, 바꾸려던 값이 아니라 원래 값으로 답을 내게 됩니다. 밑줄로 시작하는
이름은 자유롭게 써도 되며, 필드 자체는 전과 같이 대입됩니다.

버전은 키가 바뀔 때만이 아니라 값의 의미가 바뀔 때마다 올라갑니다. 버전 3은 재료 규약입니다. 키가
바뀌었지만(`eps`에서 `structure`로, `n_top` / `n_bot`에서 `top` / `bot`으로) **물리량**도 바뀌었습니다.
키 이름만 바꾸고 숫자는 그대로 둔 버전 2 문서는 다른 구조를 기술하며, 그 구조도 풀리고 에너지도
보존합니다. 그래서 추측으로 마이그레이션하지 않고 거부합니다.

`Material`에는 JSON 형식(`{"epsilon": e}`, 투자율을 지정했다면 `"mu"` 추가)이 있으므로 왕복 변환을
견딥니다. 맨 숫자는 맨 숫자로 남습니다. 다른 표기로 바꾸면 `key()`가 달라지기 때문입니다.

`TabulatedMaterial`은 전체 표와 함께 `dispersion` 객체에 저장됩니다.
파장 스윕은 점마다 이를 다시 평가하며, 숫자로 준 `Material` 값은 상수로
남습니다. 표는 `key()`에 포함됩니다. [재료](../guide/common/materials.md)를 보세요.

## 소비자에게 무엇이 가능한지 알리기

spec은 무엇을 실행할지 말하고, **capability manifest**는 무엇을 요청할 수 있는지 말합니다. 이는
생성된 JSON 문서로, 어떤 경로가 어떤 옵션을 구현하는지, 비스듬한 셀이 무엇을 거부하는지, 어떤 재료
경우를 풀 수 없는지를 다룹니다.

```shell
python -m wavebit.manifest > wavebit-manifest.json
```

이 문서는 `material_quantity`를 기계가 읽을 수 있는 키로 담고 있으므로, 생성기는 재료가 어떤
물리량으로 표현되는지 산문을 파싱하지 않고 *검사*할 수 있습니다. 그리고 그 답이 바뀔 때마다 문서의
버전 번호가 바뀝니다. 그래서 옛 답만 아는 소비자는 문법적으로 맞고 에너지도 보존하지만 틀린 장면을
내보내는 대신 거부할 수 있습니다.

**능력과 제약만 담습니다** — manifest는 어떤 것도 추천하지 않습니다.
[`../capability-manifest.md`](../reference/common/capability-manifest.md)를 보세요.

버전 **10**은 아래에서 위로 쌓는 `stack.layers`와, 아래 경계면에서 위쪽을 양으로 하는 높이를 씁니다.
버전을 바꾸기 전에 예전 층 목록을 뒤집고 z/텐서 규약을 검토하세요.
[좌표 마이그레이션](../reference/common/conventions.md#updating-a-pre-release-downward-positive-scene)을 보세요.
