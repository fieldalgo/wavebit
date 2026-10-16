# FDTD 솔버가 문제를 모델링하는 방식

> **과거 API 재현:** 이 페이지에서 `wavebit.compat`를 쓰는 코드·설명은 현재 설치의 API가 아닙니다.
> 해당 이름은 [퇴역 전 소스 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)에 있습니다.
> 측정 재현에는 본문에 적힌 원래 커밋·환경을 우선 사용하세요. 원본 측정값은 갱신하지 않았습니다.


`wavebit.fdtd`는 균일한 Yee 격자 위에서 시간 영역 맥스웰 방정식을 풉니다. 이 페이지는 선언한
문제가 결과가 되기까지 어떤 일이 일어나는지 설명합니다. 도메인이 어떻게 셀로 바뀌는지, 각
구조물이 그 셀에 어떻게 그려지는지, 경계·소스·모니터가 격자 위에서 무엇인지, 실행 전에 어떤
정확도 규칙이 강제되는지, 그리고 솔버가 아직 하지 않는 것이 무엇인지입니다. `check` 리포트를
읽거나 결과를 믿어도 되는지 판단할 때의 배경 지식입니다. 처음 실행해 본다면
[실행 가능한 펄스 예제](../../getting-started/fdtd.md)부터 보세요. 선언 API는
[도메인, 소스, 관측](general-fdtd.md)과 [레퍼런스](../../reference/fdtd/index.md)에 있습니다.

## 한눈에 보기

| 구성 요소 | 격자 위에서의 의미 |
| --- | --- |
| `UniformGrid(spacing=dx)` | 세 축 모두 같은 크기 `dx`의 정육면체 셀 |
| `Domain(size, center)` | 물리 영역. 중심을 기준으로 셀 단위로 **올림**됩니다 |
| `Slab`, `Region` | **셀 스냅** 구조물: 셀을 통째로 채우고, 면은 가장 가까운 셀 경계로 옮겨집니다 |
| `Box`, `Sphere`, `Cylinder`, `Prism` | **스무딩** 바디: 경계에 걸친 셀은 바디가 채운 비율만큼 섞은 재료를 받습니다 |
| `Open()` 면 | 물리 영역 **바깥**에 덧붙는 PML. 그 면에 닿는 매질에 맞춰집니다 |
| `Periodic()`, `Bloch(...)` | 셀이 반복됩니다. `Bloch`는 위상을 더하고 실행을 복소수로 만듭니다 |
| `PEC()`, `PMC()`, `Symmetry(parity)`, `Axis1D(k, k_max)` | 도체 벽, 거울면, 정확히 미분되는 불변 축 |
| `Numerics(courant=0.5)` | 시간 간격 `dt = courant · dx / (c·√3)` |
| `Record` | 한 성분의, 가장 가까운 Yee 점에서의 시계열 |
| `Field`, `Flux`, `Diffraction`, `FarField` | 실행 중에 누적되는 장의 푸리에 변환 |

실행은 CPU에서 배정밀도로, NumPy·JAX·PyTorch 백엔드 중 하나로 돌아갑니다. JAX와 PyTorch
실행은 미분할 수 있습니다. [실행 미분](#실행-미분)을 보세요.

## 단위와 규약

| 물리량 | 단위 |
| --- | --- |
| 길이, 위치, 진공 파장 | 문제의 `Units(length="m" \| "mm" \| "um" \| "nm")` |
| 시간, 펄스 지연, `Duration(seconds=...)` | 초 |
| `GaussianPulse(frequency_sigma_hz=...)` | Hz |
| 각도: `PlaneWave(theta, phi)`, `Box`/`Prism(angle)` | 라디안 |
| 전기장(`Record`, 최종 장), 평면파·TFSF 진폭 | V/m |
| 자기장 | A/m |
| `Field` 페이저 | V·s/m, A·s/m (장 시계열의 푸리에 변환) |
| `Flux` | J/Hz, 양측(two-sided) 스펙트럼 에너지 밀도 (`Axis1D` 면이 하나면 J/(Hz·m)) |
| `ConductiveMaterial(conductivity=...)` | S/m |
| `Dipole(current_moment=...)` | A·m (전기), V·m (등가 자기) |

시간 규약은 `exp(-iωt)`입니다. `+x` 방향으로 진행하는 파동은 `exp(+ikx)`로 변하고, `Bloch`
위상 `φ`는 `x + L`에서의 장이 `x`에서의 장의 `exp(+iφ)`배라는 뜻입니다. 스펙트럼 모니터는
유한한 실행 시간 전체에 걸쳐 창 함수 없이 `∫ f(t) exp(+i2πft) dt`로 변환합니다. 엔진 내부는
정규화 단위(`c = 1`, H는 진공 임피던스로 스케일)로 계산하고, 모든 공개 입력과 출력은 모듈
경계에서 변환됩니다.

## 격자

**모든 축에 같은 간격.** `UniformGrid(spacing=dx)`는 정육면체 셀을 만듭니다. x 방향 셀 `i`는
`[x_lo + i·dx, x_lo + (i+1)·dx)`를 덮고, y와 z도 같습니다. 불균일 격자나 축별 간격은
없습니다.

**도메인은 중심을 기준으로 셀 단위로 올림됩니다.** 크기 `s`인 도메인은 (반올림 오차를 빼면)
`ceil(s / dx)`개의 셀을 받고, 아래쪽 모서리는 `center - n·dx/2`입니다. `check`는 요청한 크기,
실제로 쓰인 크기(`resolved["physical_domain"]`), 셀 개수(`resolved["grid_shape"]`)를
보고합니다.

**여섯 장 성분은 엇갈린 점에 놓입니다(Yee 셀).** 한 셀 안에서 `(x, y, z)` 방향 반 셀 단위
오프셋은 다음과 같습니다.

| 성분 | 오프셋 | 성분 | 오프셋 |
| --- | --- | --- | --- |
| Ex | (1, 0, 0) | Hx | (0, 1, 1) |
| Ey | (0, 1, 0) | Hy | (1, 0, 1) |
| Ez | (0, 0, 1) | Hz | (1, 1, 0) |

즉 전기장 성분은 자기 축 방향으로 반 셀, 자기장 성분은 나머지 두 축 방향으로 반 셀씩 떨어져
있습니다. 위치를 다시 읽을 때 이것이 중요합니다. `Record`는 그 성분의 가장 가까운 점에서
샘플링하고 요청한 좌표와 실제 샘플 좌표를 함께 돌려주며, `Field`는 각 성분을 그 성분의 점에서
돌려줍니다. 보간은 하지 않습니다.

**흡수층은 바깥에 덧붙습니다.** 각 `Open()` 면은 물리 도메인 바깥에 `PML(cells=12)`개(기본값)의
셀을 받습니다. 그래서 PML 셀을 늘려도 선언한 영역이 움직이거나 줄어들지 않습니다
(`resolved["padding_cells"]`, `resolved["allocated_domain"]`).

## 시간 진행

E와 H는 번갈아 갱신됩니다(leapfrog). E는 정수 스텝에, H는 반 스텝 앞에 있습니다. 시간 간격은
`dt = courant · dx / (c·√3)`이고 기본값은 `Numerics(courant=0.5)`입니다. `courant`는 진공에서의
3차원 안정 한계인 `(0, 1)` 안에 있어야 합니다. 재료가 이 한계를 좁힐 수 있습니다. `ε·μ < 1`인
매질이나 분산 극점은 안정한 최대 스텝을 낮추고, `dt`가 그 한계를 넘는 장면은 메시지에 한계값을
담아 거절됩니다(그때는 `courant`를 낮추세요). `check`는 `resolved["dt_seconds"]`와
`resolved["steps"]`를 보고합니다.

`Duration`은 `steps`, `seconds`, 또는 `source="이름"`과 함께 쓰는 `periods`(이름 붙은 소스의
중심 파장 주기 수) 중 정확히 하나를 받습니다. 공개 모듈에는 자동 정지가 없습니다. 실행은 선언한
시간만큼 이어지고, 끝날 때까지 장이 감쇠하지 않았으면 경고가 알려 줍니다([경고](#경고) 참고).

## 구조물: 셀 스냅과 스무딩

장면은 배경 재료와, 이름이 붙은 구조물 `Structure(name, shape, material)`의 순서 있는 튜플로
이루어집니다. **구조물은 순서대로 칠해집니다. 두 구조물이 겹치면 나중 것이 이깁니다.** 각
도형은 두 가지 방식 중 하나로 격자에 그려지며, 이 차이가 장면에서 가장 중요한 정확도
결정입니다.

### 셀 스냅 도형: `Slab`과 `Region`

- `Slab(z=(lo, hi))`는 층입니다. 한 축만 범위가 있고 나머지 두 축은 도메인을 가득 채웁니다.
- `Region(size=(sx, sy, sz), center=(cx, cy, cz))`는 축에 나란한 블록입니다.

둘 다 **셀을 통째로** 채웁니다. 각 면은 가장 가까운 셀 경계 `i = round((x - x_lo) / dx)`로
옮겨지고, 옮겨진 면 사이의 모든 셀이 여섯 성분 모두에 대해 그 재료를 온전히 가집니다. 그래서
면은 최대 반 셀까지 움직이고, 이는 `dx`에 대한 1차 오차입니다. 같은 셀로 반올림되는 두 구간은
똑같은 결과를 냅니다. `dx = 0.025` µm에서 0.317 µm 필름과 0.325 µm 필름은 비트 단위로 같게
실행됩니다.

```
셀 경계 (dx = 0.025):          0.275    0.300    0.325    0.350
                                 |--------|--------|--------|
요청한 경계면 0.317 ----------------------------^  실제로는 0.325
```

`check`는 옮겨진 면마다 코드 `geometry.cell_snapped`의 경고 issue를 보고합니다. `expected`에
요청한 구간, `actual`에 실제로 쓰인 셀 경계가 들어 있고, `report.ok`는 참으로 남습니다. 층
두께가 셀 수의 정수배가 되도록 `dx`를 고르고, `dx`를 줄여 가며 수렴을 확인하세요.

셀 경계 위에 있는 스냅 면도 모든 성분에 대해 정확하지는 않습니다. 면이 놓인 노드 평면 위의
접선 방향 전기장은 그 주변 셀 전체에 걸쳐 재료를 받기 때문에, 그 장에게는 `Slab`이 통째로 반
셀 아래에 있는 것처럼 보입니다([smoothed occupancy의 G5](../../validation/fdtd/fd-subpixel-occupancy.md)).
층 전체가 움직이면 선언한 평면에서 잰 `r`, `t`의 위상이 바뀌지만 효율은 바뀌지 않습니다. 다른
솔버와 위상까지 맞춰야 한다면, 스무딩된 `Box` 층은 면을 선언한 위치에 둡니다.

스무딩 바디는 못 하고 셀 스냅 도형만 할 수 있는 것:

- **흡수 면 안으로 이어지기.** `Open()` 면의 축을 가로지르는 `Slab`(기판, 상부 매질)은 PML을
  통과해 이어지고, 그 PML은 슬랩의 재료에 맞춰집니다.
- **지원되는 모든 재료 담기.** `ConductiveMaterial`과 `DispersiveMaterial`도 됩니다.
- **주기 전체에 걸치기.** `Periodic()` 축 전체에 걸친 `Region`은 셀에 맞춰 잘리며, 라멜라
  격자의 이빨을 이렇게 씁니다. 주기를 다 덮지 않으면서 주기 이음매를 가로지르는 `Region`은
  반대편 이미지가 빠지므로 거절됩니다.

기하는 정적입니다. 크기·중심·구간은 미분되지 않습니다(재료 값은 미분됩니다).

### 스무딩 바디: `Box`, `Sphere`, `Cylinder`, `Prism`

- `Box(size, center, angle=0, axis="z")` — 직육면체 블록. `axis`를 축으로 `angle` 라디안만큼
  반시계 방향으로 돌아갑니다.
- `Sphere(radius, center)`.
- `Cylinder(radius, height, axis="z", center)`.
- `Prism(vertices, height, axis="z", center, angle=0)` — `axis` 방향으로 밀어 올린 다각형.
  `vertices`는 축에 수직인 평면에서 `center`로부터의 `(u, v)` 오프셋이며 반시계 방향입니다.
  축이 `"z"`면 `(x, y)`, `"x"`면 `(y, z)`, `"y"`면 `(z, x)`입니다. 오목한 외곽선도 됩니다.
  시계 방향이거나 스스로 교차하는 외곽선은 거절됩니다.

스무딩 바디는 스냅되지 않습니다. 엔진은 여섯 성분 각각의 Yee 점에서 그 성분의 셀을 바디가
채우는 비율과 그곳의 표면 방향을 계산하고, Farjadpour 등(*Opt. Lett.* **31**, 2972, 2006)과
Kottke, Farjadpour, Johnson(*Phys. Rev. E* **77**, 036611, 2008)의 비등방 유효 매질 규칙으로
바디의 재료를 그 셀에 이미 있던 재료와 섞습니다. 그래서 셀 경계 사이에 있는 면도 선언한
위치에 머뭅니다. 위의 0.317과 0.325 필름을 `Box`로 쓰면 서로 다른 답이 나옵니다.

이 규칙에는 한계가 있고, 모두 `check`가 강제합니다.

| 규칙 | 이유 |
| --- | --- |
| 바디가 경계 짓는 가장 작은 폭이 4셀 이상이어야 합니다. 4~8셀이면 경고 | 그보다 적은 셀에서는 섞인 셀이 어디에 떨어지느냐가 바디보다 답을 더 좌우합니다 |
| 재료: 유전율과 투자율이 스칼라인 `Material`. 축별 값은 안 됩니다 | 섞는 규칙은 성분별로 적용되고 등방 바디를 가정합니다 |
| `ConductiveMaterial`, `DispersiveMaterial`은 안 됩니다. 셀 전체에 걸친 평평한 필름만 예외 | 손실·분산 재료의 섞기는 그 필름 범위에서만 검증됐습니다 |
| `Open()` 면에 닿으면 안 됩니다 | 흡수층 안으로의 연장이 정의되지 않습니다. 그곳에는 `Slab`을 쓰세요 |
| `Periodic()` 면에 닿으면 안 됩니다. 단 주기 전체를 덮으면서 양쪽 면을 한 셀 **넘게** 지나가면 됩니다 | 그렇지 않으면 이음매에서 일부가 사라집니다 |
| 두 스무딩 표면이 같은 셀을 지나가면 안 됩니다 | 섞는 규칙은 셀당 매질 둘, 표면 하나를 기술합니다. 한 셀 이상 떨어뜨리거나 한 바디로 쓰세요 |
| 스무딩 표면이 셀 스냅 면이 자르는 셀을 함께 자르면 안 됩니다 | 그 셀에 매질이 셋 생깁니다(아래 참고) |

스무딩 바디는 셀 스냅 구조물 **안**에 놓이거나 다른 스무딩 바디 위에 설 수 있습니다. 회전하지
않은 스무딩 `Box`의 면 위에 딱 붙어 선 `Cylinder`나 회전하지 않은 `Box`는
`Numerics(junction="split")`(기본값)이 처리합니다(다른 선택지는 `"staircase"`와 `"refuse"`).

전역 `Numerics.junction` 설정은 기판 continuation을 적용한 뒤 엔진이 판별한 각 지원
flush 접합의 standing body(기둥)에만 전달됩니다. 기판과 관계없는 구조는 나열 순서와
무관하게 기존 규칙을 유지합니다. `"staircase"`는 접합 점의 소유 매질을 선택하고,
`"refuse"`는 실제 공유 계면을 거부합니다. 잘못된 값이나 지원 flush 접합 없이 지정한
비기본 설정은 `numerics.junction` 오류입니다. 기본 `"split"`은 접합이 없어도 사용할
수 있으며, 준비된 traced 접합 형상은 기존처럼 `"split"`만 지원합니다. 준비된 계산을
재사용해도 설정은 유지됩니다. traced standing body의 `"staircase"`는
`numerics.junction`에서 명시적으로 거부하며, 관계없는 traced body는 정적 staircase
접합을 바꾸지 않습니다. `"refuse"`는 준비 단계에서 접합을 거부하므로 split으로
대체된 계산을 준비할 수 없습니다.

잘못된 selector 값은 공개 `Numerics(junction=...)` 또는
`Numerics(discretisation=...)` 표기와 지원하는 값으로 설명합니다. `check`는
`invalid_input` issue를 반환하고 `prepare`는 같은 issue를 가진 `ValidationError`를
발생시킵니다. `path`는 해당 설정을 가리키며 `actual`은 입력한 값을 보존합니다.

`check(...).resolved["numerical_policies"]`와 `prepare(...).report.resolved`는 요청한
설정과 구조별 실제 적용을 함께 기록합니다. `discretisation` 행의 `stencil`은 실제
준비된 엔진 연산자, `voxel`은 기본 내장 스무딩 body를 뜻합니다. `rule=None`과
`whole_cell_fill`은 Slab/Region의 whole-cell fill 유지, `custom_shape_protocol`은
custom Shape 자체의 fill 의미를 나타냅니다. 공개 내장 `Slab`과 `Region`의 정확한
클래스를 구분하며, 사용자 하위 클래스는 check·prepare·validate 및 run의 provenance에서
custom protocol 분류를 유지합니다. `junction`은 기록된 standing body에만
규칙을 표시하고, `junction_face`와 `no_resolved_junction`을 구분하며 각 접합의 구조
이름과 축을 나열합니다. 접합 없는 기본 split 요청은 적용 없음으로 표시됩니다.
잘못되거나 지원되지 않는 요청은 적용 성공 행 대신 오류를 반환합니다.
`Prepared.validate(problem)`는 허용된 concrete 재사용의 보고서를 갱신합니다.
구조별 대상 변경이나 잘못된 ranged 접촉은 거부하고 이전 적용 행을 반환하지 않습니다.
원래 준비 보고서의 정책 기록은 변경되지 않는 snapshot입니다.

`Result.provenance["numerical_policies"]`는 해당 run의 scene과 실행 plan에서 만든 같은
정책 형식입니다. runtime guard가 필드를 무효화해도 적용된 연산자 정책을 기록하며,
그 자체가 수치값의 허용 판정은 아닙니다. 보고서의 `capabilities["runtime_validation"]`는
입력을 구체적으로 읽을 수 있으면 `"concrete_inputs"`, traced 입력을 host에서 읽지 못하면
`"deferred"`입니다. deferred validate 보고서의 성공은 구조와 정책의 확인이며 실제 수치
판정은 run의 `validity.checks["runtime_stability"]`를 확인해야 합니다. 구체적인 거부값은
issues를 반환하고 resolved 정책 행을 제공하지 않습니다. validate/run은 이전 보고서와
Result의 정책 기록을 변경하지 않습니다.

**기판 위 기둥.** 마지막 규칙 때문에 스무딩 기둥은 `Slab` 기판 위에 딱 붙여 세울 수 없습니다.
선택지는 다음과 같습니다.

1. `Open()` 면에 닿지 않는 유한한 스무딩 `Box` 기판 위에 기둥을 딱 붙여 세우기(위의 junction
   규칙);
2. `Slab` 기판 위의 셀 스냅 `Region` 기둥(직사각형, 면은 스냅됨);
3. `Slab` 면에서 반 셀 이상 띄운 스무딩 기둥.

어느 것도 "흡수층까지 이어지는 반공간 기판 위에 붙어 선 스무딩 기둥"은 아닙니다. 그 장면은
아직 공개 모듈로 쓸 수 없습니다.

**스무딩의 정확도.** 기본 규칙 `Numerics(discretisation="voxel")`은 통제된 유리 구 연구에서
2.02차로 수렴했지만, 곡면 정확도 관문을 일반적으로 통과하지는 못했습니다. 스무딩 바디가 있는
모든 실행이 이를 알리는 `SubpixelAccuracyWarning`을 냅니다. 대안인 `discretisation="stencil"`은
구에 대해 그 관문을 통과했지만(`ε = 12` 구에서 2.24차) `Sphere`에만 적용되고, 일반 숫자만 받으며
(미분 불가), `PlaneWave`나 `TFSF` 소스가 필요합니다. 측정은
[Mie 검증](../../validation/fdtd/fd-mie-validation.md)과
[smoothed occupancy](../../validation/fdtd/fd-subpixel-occupancy.md)에 있습니다.

장면 전체의 `Numerics(discretisation="stencil")` 설정은 스무딩 본체에만 전달됩니다.
`Slab`과 `Region`은 whole-cell fill을 유지합니다. `Box`, `Cylinder`, `Prism` 등
지원하지 않는 형상과 적용할 `Sphere`가 없는 장면은 `numerics.discretisation`에서
거부합니다. 이 설정은 정확한 내장 `Sphere`, `Slab`, `Region` 타입만 받습니다.
사용자 Shape와 subclass는 fill 프로토콜을 바꿀 수 있어 거부합니다.
stencil의 지원 범위는 그대로입니다. 진공 속 구 하나와 미분하지 않는
구체적인 수치만 지원하며, slab이나 region을 추가한 다중 구조물 장면은 지원하지 않습니다.

### Stencil 준비 메모리

`Numerics(discretisation="stencil", memory_bound=2 * 1024**3, ...)`는 호스트
준비 과정의 예상 메모리 한도를 바이트로 정합니다(이 예에서는 2 GiB).
`memory_bound`를 생략하거나 `None`을 쓰면 엔진 기본값 6 GiB를 유지합니다.
양의 유한 실수만 허용하며, 불리언·0·음수·비유한 값은 `numerics.memory_bound`에서
거절합니다. `discretisation="stencil"`과 적용 가능한 `Sphere`가 없으면 설정을
읽는 준비 과정도 없으므로 거절합니다.

기존 예측값을 한도와 비교하는 검사는 희소 행렬 조립과 풀이 **이전**에 실행됩니다.
이는 예상치이며 프로세스 전체 메모리의 상한이나 메모리 부족 방지 보장이 아닙니다.
`UniformGrid(spacing=...)`의 격자를 굵게 하거나 구를 줄이거나, 사용 가능한 자원을
확인해 한도를 명시적으로 조절하세요. 한도를 올려도 stencil의 물리적 지원 범위는
넓어지지 않습니다. 현재 연산자는 진공 속 구 하나를 요구합니다. 구조물이 여러 개면
`Numerics(discretisation="voxel")`을 쓰거나 지원되는 구 하나의 장면으로 줄이세요.
수치 정책은 `Structure`가 아니라 `Numerics`에서 정합니다.

## 재료

| 모델 | 용도 |
| --- | --- |
| `Material(epsilon, mu=None)` | 실수, 양수, 비분산. 스칼라 또는 축별 값 `(x, y, z)` |
| `ConductiveMaterial(epsilon, conductivity, mu=None)` | S/m 단위의 일정한 전도도 |
| `DispersiveMaterial(epsilon_inf, poles, mu=None)` | SI 비율로 준 수동(passive) `Drude`, `Lorentz`, `Debye` 극점 |

다음은 각각 메시지와 함께 거절됩니다. 복소 유전율 상수(한 진동수에서만 맞습니다. 극점 모델을
쓰세요), 3×3 전체 텐서, 표로 주거나 이미 평가된 재료. 배경 재료는 어떤 구조물도 차지하지 않은
모든 셀을 채우고, 닿는 모든 `Open()` 면에서 PML 안으로 이어집니다.

## 경계

모든 면을 선언해야 하며 기본값은 없습니다. `Boundaries(x=..., y=..., z=...)`는 축마다 면
종류 하나 또는 `(low, high)` 쌍을 받습니다.

| 면 | 의미 |
| --- | --- |
| `Open()` | 바깥으로 나가는 면: 물리 도메인 바깥의 PML이 그 면에 닿는 균일 매질에 맞춰집니다. 균일 매질만 닿을 수 있습니다. 배경, 또는 그 축을 가로지르는 `Slab`입니다. 유한한 바디가 닿거나, 직교하는 `Open()` 면을 가로지르는 `Slab`은 거절됩니다. |
| `Periodic()` | 두 면이 위상 없이 이어집니다. 주기 셀에서의 수직 입사. |
| `Bloch(phase=φ)` 또는 `Bloch(k=k)` | 두 면이 위상을 두고 이어집니다. 실행이 복소수가 되어 메모리와 계산이 두 배입니다. `Bloch.from_source(source, units=...)`가 비스듬한 `PlaneWave`에 필요한 면을 만들어 줍니다. |
| `PEC()` / `PMC()` | 벽에서 접선 E / 접선 H가 0입니다. 벽 너머로 뻗은 구조물은 잘리고 `GeometryClippedWarning`이 납니다. |
| `Symmetry(parity=±1)` | 거울면. `+1`은 접선 E가 면에 대해 짝수(PMC와 비슷한 면), `-1`은 홀수입니다. |
| `Axis1D(k=..., k_max=...)` | 미분이 정확히 `ik`인 한 셀짜리 불변 축. 2차원 원뿔 격자의 균일한 축을, 한 셀 차분의 인위적 분산 없이 표현합니다. |

`Numerics(absorber=PML(...))`가 모든 `Open()` 면의 흡수층을 정합니다. `cells=12`(그보다 적으면
`allow_thin=True` 없이는 거절), `reflection=1e-8`, `wavelength`(생략하면 소스들이 공유하는 중심
파장), 그리고 면 옆의 속박 모드가 자라지 않게 막는 추가 손실인 `stabilize`입니다(아래
`GrowingFieldWarning` 참고).

### Grazing 거절 뒤 설정 바꾸기

`check`, `prepare`, `run`은 가장 심하게 거절된 입사파 또는 회절 차수의 면·파장·반사 기준을
설명하고, 함께 거절된 나머지 항목도 나열합니다. 공개 설정 위치는 `Numerics.absorber`입니다.
수치 설정은 **모든 Open 면에 공통으로** 적용되며, 각 면의 물리 매질 또는 고정 기준 매질은
유지됩니다.

메시지에 `updated = numerics.replace(...)` 후보가 나오면 기존 `numerics` 값에 적용하세요.
안쪽의 `absorber.replace(...)`는 `wavelength`, `reference_media`, `allow_thin`을 보존합니다.
실행 시간을 바꿔야 할 때는 실제 소스 이름을 넣은 `fdtd.Duration(periods=..., source=...)`를
제시합니다. 이미 더 강한 반사 목표·더 깊은 층·충분히 긴 시간은 유지합니다. 흡수층을 깊게
하면 물리 영역 **바깥**의 패딩이 자동으로 늘어나므로 `problem.scene.domain`은 그대로 두세요.
셀과 주기 수가 늘면 메모리와 시간이 더 듭니다.

후보 하나를 적용한 뒤 `fdtd.check(problem, numerics=updated)`로 다시 검사하세요. 한 면의
후보가 모든 면·차수·매질·파장의 통과를 보장하지는 않습니다. 거절된 면들의 기준 매질이
서로 다르면 구체적인 수치 조합을 제안하지 않고 공통 흡수층 설정 변경과 전체 재검사를
안내합니다. `stabilize`를 줄이면 구속 모드가 다시 성장할 수 있으므로 감쇠도 확인하세요.
측정된 grazing 한계 아래에서는 흡수층이나 시간 변경으로 통과시킬 수 없으며 주기·각도·파장을
바꿔야 합니다. 오류 안내는 반사 허용오차나 검증된 지원 범위를 완화하지 않습니다.

## 소스

모든 소스는 `GaussianPulse(wavelength, frequency_sigma_hz, cutoff=1e-3)`를 가집니다. 중심 진동수
`f0 = c / wavelength`의 사인파에 가우시안 포락선을 씌운 것으로, **진폭 스펙트럼**의 표준편차가
`frequency_sigma_hz`입니다. 펄스는 최댓값의 `cutoff`에서 시작하고, 스펙트럼은 `exp(-k²/2) =
cutoff`인 `k`에 대해 `f0 ± k·σ`에서 `cutoff`로 떨어집니다. 그 대역의 단파장 끝이 해상도 규칙이
읽는 파장입니다(다음 절). `amplitude`, `phase`, `delay`(초)는 파형을 이동하고 스케일합니다.

`Plane`은 `fdtd.Plane(z=.3)`처럼 법선 축 하나를 스칼라로 지정합니다.
`Field`와 `Flux`는 `fdtd.Plane(z=.3, x=(-.05, .05))` 같은 유한 가로 범위도
지원합니다. `PlaneWave`와 `Diffraction`은 가로 셀 전체를 사용하므로 두 가로
범위를 모두 생략해야 합니다. 셀 경계와 일치하더라도 명시한 범위는 지원하지
않으며, `sources.<name>.plane` 또는 `outputs.<name>.plane` 경로의 구조화된
`plane_span` 오류가 됩니다. `check`는 이 오류를 보고하고, `prepare`와 `run`은
`UnsupportedFeatureError`를 발생시킵니다.

| 소스 | 하는 일 |
| --- | --- |
| `PlaneWave(name, plane, pulse, direction="-z", theta=0, phi=0, polarization="s")` | `plane`에 가장 가까운 노드 평면의 한 방향 시트로, `direction` 쪽으로만 내보냅니다. `theta`는 진행 축으로부터의 극각, `phi`는 첫 번째 가로 축(z 방향 파동이면 x)으로부터의 방위각입니다. `"s"`는 E가 입사면에 수직, `"p"`는 입사면 안입니다. 가로 면은 `Periodic()`(수직 입사)이거나 `Bloch.from_source`가 만든 `Bloch` 면이어야 합니다. 가로 파수 벡터는 펄스의 중심 파장에서 고정되므로, 비스듬한 광대역 펄스는 일정한 각도의 조명이 아닙니다. |
| `Dipole(name, center, pulse, component="Ey", kind="electric", current_moment=1)` | 점 **전류**(A·m), 또는 `kind="magnetic"`이면 그 자기 등가물(V·m)입니다. 매끄러운 스탬프(기본 `"bspline"` 또는 `"tent"`, 둘 다 반폭 두 셀)에 퍼뜨려, 중심이 어디에 있든 모멘트와 무게중심이 정확합니다. C·m 단위의 쌍극자 모멘트가 아닙니다. |
| `TFSF(name, box, pulse, direction="-z", polarization="x")` | `Region` 상자 안에만 있는 평면파(안은 전체장, 밖은 산란장)로, 유한한 산란체용입니다. 축 방향만 되고, 그 축의 두 면은 `Open()`이어야 합니다. |
| `InitialField(E, H, amplitude=1)` | 실행이 시작하는 장: `t = 0`의 E와 `t = -dt/2`의 H를 배열로, 또는 각 성분의 Yee 점에서 평가되는 호출 가능 객체로 줍니다. |

`PlaneWave.plane`에는 `fdtd.Plane(z=.5)`처럼 `fdtd.Plane`을 넣어야 합니다.
스칼라 `.5`만으로는 축을 정할 수 없습니다. 잘못된 형식이면 `check`는
`sources.<name>.plane` 경로에 오류를 보고하고, `prepare`와 `run`은 좌표를 읽기 전에
같은 경로의 `ValidationError`를 냅니다. 올바른 Plane도 물리 영역 안에 있어야 합니다.

## 출력

모니터는 `Outputs(monitors=...)`에 이름으로 선언하고 `result[name]`으로 읽습니다. 주파수 영역
모니터는 나열한 파장에서 실행 중에 변환을 누적합니다.

유한 monitor 범위에서 `Field`는 요청한 가로 구간 안의 각 성분 Yee 점을 끝점까지
포함해 반환합니다. 배열 축은 주파수, z, y, x 순서이며 좌표 배열은 실제 샘플을
나타냅니다. `Flux`는 가로 끝점을 가장 가까운 노드에 맞추고 그 사이 표면을
적분합니다. 셀 중심 샘플과 양 끝 노드의 절반 가중치를 사용하므로, 비정렬 범위의
실제 적분 면적은 요청한 기하학적 면적과 다를 수 있습니다. 보간이나 가장자리
셀의 부분 면적 적분은 하지 않습니다. 유한 Flux 구간은 적어도 한 셀을 포함하고
마지막 저장 노드 이내에서 끝나야 합니다(높은 쪽 domain 면 자체는 저장하지
않습니다). Field 구간은 요청한 모든 성분의 점을 포함해야 합니다.
두 monitor 모두 물리 domain 밖의 요청 가로 구간을 자르거나 격자에 맞추기 전에
`outputs.<name>.plane.<axis>` 경로의 `monitor_extent` 오류로 거부합니다.
역순 또는 폭이 0인 구간은 `Plane`을 만들 때 거부합니다.

| 모니터 | 돌려주는 것 |
| --- | --- |
| `Record(name, at, component)` | 한 성분의, 가장 가까운 Yee 점에서의 시계열. V/m 또는 A/m, 시간 축은 초 |
| `Field(name, wavelengths, plane=... 또는 box=..., components=...)` | 각 성분의 점에서의 푸리에 페이저(V·s/m, A·s/m) |
| `Flux(name, wavelengths, plane=... 또는 box=...)` | 평면을 (`+` 법선 방향으로) 지나거나 닫힌 상자에서 나가는 포인팅 에너지의 진동수당 양: 와트가 아니라 J/Hz 단위의 펄스 에너지 밀도. 닫힌 상자는 각 구조물을 통째로 감싸거나 통째로 빼야 합니다. |
| `Diffraction(name, plane, wavelengths, source, side, orders=None)` | 주기 셀의 한쪽(`"reflect"` 또는 `"transmit"`)으로 나가는 차수들. 각 차수의 s/p 좌표계에서의 복소 진폭과 효율이며, 이름 붙은 `PlaneWave`가 직접 잰 입사장으로 정규화됩니다. `PlaneWave`가 정확히 하나여야 합니다. `reference_plane`과 `incident_plane`은 위상 기준면을 옮기고, `raw=True`는 정규화하지 않은 스펙트럼을 남깁니다. 펄스의 사용 가능한 대역 밖 파장은 NaN으로 돌려주며, 그곳에서 `validity.checks["source_reference"]`가 거짓입니다. |
| `FarField(name, wavelengths, box, propagation="continuum")` | 열린 장면의 복사: 모든 것을 감싸는 닫힌 상자를 무한원으로 변환합니다. `intensity(theta, phi)`는 라디안을 받습니다. 주기 셀은 거절됩니다(주기 셀의 원거리장은 회절 차수입니다). `propagation="yee"`는 격자 자체의 분산을 씁니다. |
| `Outputs(final_fields=True)` | 실행 끝의 E, H 배열과 각 성분의 좌표 |

(fdtd-model-accuracy-rules)=
## 실행 전에 강제되는 정확도 규칙

`check`, `prepare`, `run`이 같은 관문을 공유합니다. 거절 메시지는 바꿔야 할 설정을 알려
줍니다.

- **모든 재료에서 파장당 8셀.** 소스 대역의 최단 파장에서(그리고 모든 `Diffraction` 파장에서),
  각 매질은 *그 매질 안에서* 파장당 8셀 이상, 즉 `λ / (n·dx) ≥ 8`로 샘플링되어야 합니다. 8
  미만이면 거절되고, 8~12이면 `CoarseMeshWarning`이 납니다. 굴절률 `n`인 구조물은 진공보다
  `n`배 촘촘한 셀이 필요합니다.
- **스무딩 바디는 4셀 이상의 폭**, 위 표와 같습니다.
- **시간 간격**은 장면 재료의 안정 한계 안에 있어야 합니다.
- **PML**은 `allow_thin=True`가 아니면 12셀 이상이어야 합니다.
- **배치.** 소스와 모니터는 물리 도메인 안에 있어야 하고, 닫힌 `Flux` 상자와 `FarField` 상자는
  위 표의 설명대로 구조물과 떨어져 있어야 합니다.

이 규칙을 통과하는 것은 입장 검사이지 수렴 결과가 아닙니다. 필요한 양이 더 이상 변하지 않을
때까지 `dx`와 실행 시간을 늘려 가며 답을 다듬어야 합니다.

## 경고

경고는 표기가 아니라 답에 관한 것입니다. 실행은 숫자를 내지만, 그 숫자는 아마 기대보다 나쁩니다.
`check`는 계획 단계의 경고를 `report.capabilities["warnings"]`에 나열하고, `run`은 이를
`wavebit.fdtd.diagnostics`의 파이썬 경고로 냅니다.

| 경고 | 의미 |
| --- | --- |
| `CoarseMeshWarning` | 매질 안에서 파장당 8~12셀, 또는 폭이 4~8셀인 스무딩 바디 |
| `SubpixelAccuracyWarning` | 장면에 스무딩 바디가 있습니다. 기본 규칙의 정확도 관문은 아직 완료되지 않았습니다 |
| `AbsorberProximityWarning` | 구조물이 흡수 면에서 (그 매질 안의) 반 파장 이내에 있습니다. 그 면의 PML은 구조물의 근접장이 아니라 그 면에 닿는 매질에 맞춰져 있습니다. 기판처럼 흡수층 안으로 이어지는 `Slab`에는 나오지 않습니다. 그 PML이 그 층에 맞춰지기 때문입니다. |
| `GeometryClippedWarning` | 구조물이 `PEC`나 `PMC` 벽 너머로 뻗어 그 부분을 잃습니다 |
| `UnsettledFieldWarning` | 모니터 영역에 실행 끝까지 장이 남아 있었습니다. 그 스펙트럼에는 잘림이 섞여 있습니다. 더 오래 실행하세요 |
| `GrowingFieldWarning` | 모든 소스가 끝난 뒤에도 장이 자랍니다. 더 오래 실행하면 더 나빠집니다. 대개 PML 옆의 속박 모드이니, 메시지가 알려 주는 값으로 `PML(stabilize=...)`를 정하세요 |
| `PMLStabilityWarning` | 안정화된 PML이 그런 성장을 막는 데 필요하다고 측정된 것보다 적은 손실을 요청합니다 |
| `RadiationApproximationWarning` | `FarField(propagation="continuum")`은 이산 플럭스와 정확히 맞지 않을 수 있습니다 |

## 수명주기와 리포트

`check(problem, numerics=...)`는 장을 한 스텝도 진행하지 않고 검증하고 추정합니다. 그
`CheckReport`에는 `issues`(오류, 그리고 `geometry.cell_snapped` 같은 경고. `report.ok`는 오류가
있을 때만 거짓), `resolved`(격자 모양, 간격, 도메인, 패딩, `dt_seconds`, 스텝 수),
`capabilities`(백엔드, 경고), `estimates`(주요 장과 재료 배열의 메모리)가 들어 있습니다.

`prepare(problem, numerics=...)`는 계산 구조를 고정하고, `prepared.run(problem)`은 완전한 문제를
그 구조에 대해 실행합니다. 재료 값, 소스의 진폭·위상·지연·펄스 파라미터, 초기장 진폭, 예약된
범위가 있는 바디의 기하는 **살아 있는(live)** 입력이라 실행마다 바꿀 수 있습니다(단, 고정된
계획 안에서만입니다. `Duration(periods=...)`의 스텝 수를 바꾸는 펄스 파장은 거절됩니다). 나머지
— 단위, 도메인, 경계, 도형, 모니터 배치, 재료 모델의 종류 — 는 고정되고, 바꾸면
`IncompatibleProblemError`가 납니다. `run`은 `prepare` 후 한 번의 `run`입니다.

`Result`는 `validity.checks`를 가집니다. `admitted`, `runtime_stability`(런타임 가드가 추적된 값을
거절하면 거짓, 아래 참고), 그리고 `Diffraction` 모니터별 `source_reference`입니다.
`result.provenance`는 백엔드, 단위, `dt_seconds`, 실제 스텝 수, 모든 도메인을 기록합니다.

`admitted`는 정적 선언의 허용 여부이며, 실패한 `runtime_stability` 판정을 덮어쓰지
않습니다. 미분하는 실행에서 `diagnostics['grazing']`는 host에서 지수를 조사할 수 없는
traced 물리 매질을 `skipped`로 표시합니다. 준비할 때 사용한 구체적인 재료 seed의
허용 행을 실제 실행 재료의 허용 판정으로 보고하지 않습니다. 실제 수치값의 유효성은
runtime guard가 판정합니다. 고정된 PML `reference_media` 항목은 수치적인 stretch를
고정할 뿐, 물리 매질의 grazing 허용 판정을 대신하지 않습니다.

설계 파장과 다른 `Diffraction` 파장에서는 출사 코사인 요구값이 1 이상일 수 있습니다.
준비할 때 해당 차수가 소멸파인 재료도 이에 해당합니다. 요구값이 1보다 크면 전파하는
차수는 허용되지 않습니다. 살아 있는 재료가 해당 차수를 전파하게 만들면 런타임 가드는
구체적인 값에 대한 `check`와 같이 거절합니다. 요구값이 1이거나 그 바로 아래일 때도
구체적인 검사와 같은 반올림된 코사인으로 판정하며, 요구값을 잘라내거나 grazing
허용오차를 완화하지 않습니다.
명시적인 아주 작은 `Axis1D(k=...)`에서는 구간의 하한이 0으로 반올림될 수 있습니다.
0은 열린 경계의 표식으로만 취급하며, 양수 재료 지수의 eager 연산과 모든 수용 기준은 그대로입니다.

각 `prepared.run(problem)`은 새로운 필드 상태에서 시작하고 자체 검사 결과를
반환합니다. 가드가 무효로 판정한 값은 NaN과 실패한 runtime validity를 반환하며,
host에서만 검사할 수 있는 지원되지 않는 trace는 명시적으로 예외를 발생시킵니다.
어느 경우도 이후의 유효한 실행을 무효로 만들거나 이전 결과를 변경하지 않습니다.
다시 실행하려면 유효한 전체 problem을 전달하세요.

(fdtd-model-differentiating)=
## 실행 미분

`backend="jax"`나 `backend="torch"`로 준비한 실행은 살아 있는 입력에 대해 미분할 수 있습니다.
재료 값, 소스의 진폭·위상·지연·펄스, 초기장 진폭입니다. JAX는 배열을 만들기 전에
`jax.config.update("jax_enable_x64", True)`가 필요합니다.

float64 Drude 감수율은 큰 감쇠에서 식을 재스케일링하여, 공개 SI 모델과 정규화된
FDTD 모델에서 시험한 주파수별 값과 미분을 보존합니다. float32 입력을 포함하는
공개 호출은 기존 산술과 dtype을 유지하며, float32나 혼합 정밀도의 극단값 지원을
추가하지 않습니다. Lorentz 응답과 ADE 계수는 바뀌지 않으며, 모든 유한 극단값과
목적함수의 미분을 보장하지 않습니다. 정규화된 재현 예제, SI·단위 변환의 범위와
중간 연산 underflow 제한은 [영문 미분 가이드](fd-differentiation.md)를 참고하세요.

기하는 예약된 지지 영역을 준 스무딩 바디만 미분됩니다.
`Numerics(geometry_ranges={"name": ((x0, x1), (y0, y1), (z0, z1))})`로 주며, 바디가 취하는 모든
값에서 바디 전체가 그 상자 안에 있어야 하고, 상자는 도메인 안에서 흡수층과 떨어져 있어야 하고,
재료는 등방·무손실 `Material`이어야 하며, 미분하는 바디들의 예약 상자가 겹치면 안 되고, 장면에
초기장이 있으면 안 됩니다. 셀 스냅 도형의 기하는 결코 미분되지 않습니다.

추적(trace)된 계산이 만드는 값은 실행 전에 검사할 수 없으므로, 엔진이 실행 중에 검사합니다.
안정 한계를 넘는 시간 간격, 양수가 아닌 유전율, 파장당 8셀 미만으로 샘플링되는 매질, 상자를
벗어나는 범위 지정 바디는 그럴듯한 숫자를 돌려주는 대신 실행의 장을 NaN으로 만들고
`validity.checks["runtime_stability"]`를 거짓으로 둡니다. 이산 솔버의 미분은 수렴에 대한 진술이
아닙니다. 유한 차분과 비교하고 순방향 계산도 함께 다듬으세요. [영문 미분 가이드](fd-differentiation.md)가
같은 엔진의 호환(compat) 표기로 측정을 다룹니다.

## 예제: 기판 위 코팅을 닫힌 꼴과 비교

여기서는 모든 경계면이 셀 경계 위에 있어서 스냅되는 면이 없습니다. 기판은 아래쪽 PML 안으로
이어지고, 그 PML은 기판에 맞춰집니다.

```python
import numpy as np
from wavebit import Material, Units, fdtd

dx = .02                                       # um; 0.1 um 코팅 = 5셀
wavelengths = (.7, .8, .9)
problem = fdtd.Problem(
    units=Units(length="um"),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(
            fdtd.Structure("substrate", fdtd.Slab(z=(-.8, 0.)), Material(epsilon=2.25)),
            fdtd.Structure("coating", fdtd.Slab(z=(0., .1)), Material(epsilon=6.25)),
        )),
    sources=(fdtd.PlaneWave("light", plane=fdtd.Plane(z=.5),
                            pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.Diffraction("r", plane=fdtd.Plane(z=.3), wavelengths=wavelengths,
                         source="light", side="reflect"),
        fdtd.Diffraction("t", plane=fdtd.Plane(z=-.3), wavelengths=wavelengths,
                         source="light", side="transmit"),
    )),
)
numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=dx),
                         duration=fdtd.Duration(periods=60, source="light"))

report = fdtd.check(problem, numerics=numerics)
assert report.ok and not report.issues         # 스냅된 면 없음
print(report.resolved["grid_shape"], report.resolved["steps"])

result = fdtd.run(problem, numerics=numerics)
R = result["r"].efficiency[:, 0, 0]
T = result["t"].efficiency[:, 0, 0]


def airy(wavelength, n1=2.5, n2=1.5, d=.1):  # 수직 입사, 공기 / 코팅 / 기판
    r01, r12 = (1 - n1) / (1 + n1), (n1 - n2) / (n1 + n2)
    t01, t12 = 2 / (1 + n1), 2 * n1 / (n1 + n2)
    phase = np.exp(2j * np.pi * n1 * d / wavelength)
    denominator = 1 + r01 * r12 * phase**2
    return (abs((r01 + r12 * phase**2) / denominator)**2,
            n2 * abs(t01 * t12 * phase / denominator)**2)


for w, r, t in zip(wavelengths, R, T):
    print(w, r, t, airy(w))
```

2026-10-15 측정: 각 파장에서 R + T는 5e-5 이내로 1이고, 0.8 µm에서 R은 0.3504, 닫힌 꼴은
0.3420입니다. `dx`를 0.01 µm로 반으로 줄이면 0.3440이 되어, 오차가 8.4e-3에서 2.1e-3으로 4.1배
줄어듭니다. 스냅되는 면이 없을 때 기대하는 2차 수렴입니다. 위상은 사정이 다릅니다. 반사 모니터에
`reference_plane=.1, incident_plane=.1`을 주면, 선언한 위쪽 면에서의 `arg r`이 0.8 µm에서 닫힌
꼴과 0.163 rad, `dx = 0.01`에서 0.080 rad 차이 납니다. 대략 `k0·dx`로 1차이며, 스냅된 층이 접선
방향 장에게 반 셀 아래에 있기 때문입니다([셀 스냅 도형](#셀-스냅-도형-slab과-region) 참고).
코팅을 0.105 µm로 쓰면 같은 장면이 `geometry.cell_snapped` 경고를 내고 0.1 µm로 실행됩니다.

## 일반 장면 API의 범위

- 일반 장면 실행은 CPU·배정밀도·균일한 정육면체 격자를 사용합니다.
- 셀별 배열과 비균일 z 격자는 `fdtd.native`로 선언합니다. 일반 장면의 CSG 합집합·교집합·차집합과 기울어진 밀어내기 축은 지원하지 않습니다.
- 일반 장면의 가우시안 빔·모드 소스·모드 모니터는 아직 구현하지 않았습니다.
- 자동 정지는 `fdtd.Decay`, 단계형 전도도 흡수층은 `fdtd.Absorber`로 제공합니다. [정지와 흡수층](stopping-and-absorbers.md)을 참고하세요.
- `fdtd.Shape`는 사용자 정의 고정 도형을 받으며 prepared 재사용 시 선언 상태를 검사합니다. 도형의 셀 support는 Open padding에 닿지 않아야 합니다.
- 손실·분산 스무딩 바디는 평평한 필름으로 제한됩니다.

## 더 읽을거리

- [도메인, 소스, 관측](general-fdtd.md) — 짧은 예제가 있는 선언 API.
- [Smoothed occupancy](../../validation/fdtd/fd-subpixel-occupancy.md)와 [Mie 검증](../../validation/fdtd/fd-mie-validation.md)
  — 스냅된 슬랩 면의 반 셀 위치를 포함해, 스무딩을 어떻게 측정하는지.
- [RCWA와 비교한 주기 메타표면](../../validation/fdtd/fd-metasurface-validation.md) — RCWA와 무엇이
  맞고 어떤 수렴 차수로 맞는지.
- [Yee 격자와 일관된 복사](../../validation/fdtd/fd-lattice-radiation.md) — 원거리장 변환.
- 필름 연구(예: [Drude 또는 Lorentz 필름 하나](../../validation/fdtd/fd-single-pole-film.md)) — 손실·분산
  층의 제한된 범위.
- [FDTD 배우기](../../learn/fdtd/index.md) — 믿을 수 있는 결과를 세우는 순서.
