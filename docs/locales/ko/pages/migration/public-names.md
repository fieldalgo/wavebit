# 공개 이름과 결과 키 (2026-10-19, 2026-10-20)

Wavebit은 아직 알파 단계이므로 general-FDTD 선언의 이름을 한 번에 바꿨습니다. 이제 모든
클래스는 그것이 무엇인지로, 모든 모니터는 무엇을 기록하는지로 이름이 붙고, 단위가 분명하지 않은
키워드에는 단위가 붙습니다. 옛 이름은 별칭으로 남기지 않고 제거했습니다. 옛 속성(`fdtd.Record`)이나
키워드(`discretisation=`)를 쓰면 바로 실패하고 새 이름을 알려 줍니다(import 형태는
[옛 철자를 쓰면](#옛-철자를-쓰면) 참고). 수치 계산은 바뀌지 않았습니다. 낮춘(lowered) 장면, 정적 계획,
모든 출력이 같습니다(`tests/fdtd/test_public_names.py`가 이름을 바꾸기 전에 측정한 여덟 장면의
계획 골격을 고정합니다).

설계와 그 이유는 [이슈 #57](https://github.com/fieldalgo/wavebit/issues/57)에 있습니다. 이
페이지는 `wavebit.fdtd`와, 같은 날의 두 번째 변경으로 이름이 바뀐 명시적 격자 모듈 `fdtd.native`,
`fdtd.periodic`을 다룹니다([native 격자와 주기 적층의 이름](#native-grid-and-periodic-stack-names)).
다음 날의 세 번째 변경으로 이름이 바뀐 RCWA, 관측 객체, 공통 모듈도 다룹니다
([공통·관측·RCWA 이름](#shared-observation-and-rcwa-names)). 같은 날의 네 번째 변경으로 바뀐, 모든
솔버의 결과와 보고서가 쓰는 키도 다룹니다([결과와 보고 키](#result-and-report-keys)).

## 클래스

| 이전 | 이후 | 이유 |
| --- | --- | --- |
| `fdtd.Record` | `fdtd.TimeSeriesMonitor` | 한 점의 한 성분을 매 스텝 기록합니다 |
| `fdtd.Field` | `fdtd.FieldMonitor` | 필드가 아니라 모니터입니다 |
| `fdtd.Flux` | `fdtd.FluxMonitor` | 선속이 아니라 모니터입니다 |
| `fdtd.Diffraction` | `fdtd.DiffractionMonitor` | 회절 차수를 기록하는 모니터입니다 |
| `fdtd.FarField` | `fdtd.FarFieldMonitor` | 원거리장으로 투영하는 모니터입니다 |
| `fdtd.Slab` | `fdtd.SnappedSlab` | Yee 셀을 통째로 채웁니다. `Box` 등 다른 물체는 매끄럽게 처리됩니다 |
| `fdtd.Region` | `fdtd.SnappedBox` | 셀에 맞춘 상자로, `SnappedSlab`의 짝입니다 |
| `fdtd.Dipole` | `fdtd.CurrentDipole` | C·m 단위의 쌍극자 모멘트가 아니라 A·m 단위의 전류 모멘트를 구동합니다 |
| `fdtd.Axis1D` | `fdtd.InvariantAxis` | 구조가 변하지 않는 한 셀짜리 축입니다 |
| `fdtd.Symmetry` | `fdtd.MirrorSymmetry` | `parity=+1` 또는 `-1`인 거울면입니다 |
| `fdtd.Absorber` | `fdtd.ConductiveAbsorber` | `PML`과 구분되는, 등급을 둔 전도성 층입니다 |
| `fdtd.Decay` | `fdtd.DecayStop` | 정지 규칙이며 `Numerics(stopping=...)`으로 씁니다 |

`InvariantAxis`에는 `algebraic` 필드가 없습니다. 옛 `Axis1D(algebraic=True)`는 받아들이는 값이
그것 하나뿐이었습니다.

## 키워드

| 이전 | 이후 |
| --- | --- |
| `Numerics(discretisation=...)` | `Numerics(discretization=...)` |
| `Numerics(decay=fdtd.Decay(threshold=..., window_seconds=...))` | `Numerics(stopping=fdtd.DecayStop(threshold=..., window_seconds=...))` |
| `PML(stabilize=...)` | `PML(stabilizing_loss=...)` |
| `PlaneWave`, `Dipole`, `TFSF` `(delay=...)` | `PlaneWave`, `CurrentDipole`, `TFSF` `(delay_seconds=...)` |
| `Diffraction(side='reflect' or 'transmit')` | `DiffractionMonitor(wave='reflected' or 'transmitted')` |
| `Bloch.from_source(..., axis1d=...)` | `Bloch.from_source(..., invariant_axes=...)` |

값은 그대로입니다. `delay`는 원래 초 단위였고, `stabilizing_loss`는 `stabilize`가 받던 수를
그대로 받습니다.

## 이름 다음은 키워드

이름이 있는 선언은 그 이름만 위치 인수로 받고, 나머지는 모두 키워드입니다. `Structure`, 다섯
모니터, 세 소스가 여기에 해당합니다.

```python
# Before
fdtd.Structure('film', fdtd.Slab(z=(-0.2, 0.0)), glass)
fdtd.Record('probe', (0.0, 0.0, 0.1))

# After
fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(-0.2, 0.0)), material=glass)
fdtd.TimeSeriesMonitor('probe', at=(0.0, 0.0, 0.1))
```

다른 선언은 원래 모두 키워드 전용이었습니다. 철자가 틀린 키워드는 가장 가까운 올바른 키워드를
알려 주며 거부됩니다.

## 이름마다 공개 경로 하나

모든 선언은 `wavebit.fdtd`에서 가져옵니다. 구현 모듈 `wavebit.fdtd.problem`, `geometry`,
`boundaries`, `sources`, `monitors`, `numerics`, `result`, `results`는 더 이상 공개되지 않습니다.
`fdtd.monitors` 같은 속성 접근은 실패하며 `wavebit.fdtd`를 쓰라고 알려 줍니다. 공개 하위 모듈은
`fdtd.native`, `fdtd.periodic`, `fdtd.diagnostics`이고, `dir(fdtd)`에는 공개 이름만 나옵니다.

## 옛 철자를 쓰면

```text
>>> fdtd.Record
AttributeError: wavebit.fdtd.Record is now wavebit.fdtd.TimeSeriesMonitor. The old name was removed, not aliased; see the public-name migration guide, docs/migration/public-names.md.
>>> fdtd.Numerics(grid=grid, duration=duration, discretisation='voxel')
TypeError: Numerics(discretisation=...) is now Numerics(discretization=...). The old keyword was removed, not aliased; see the public-name migration guide, docs/migration/public-names.md.
```

새 이름을 알려 주는 것은 속성 접근(`fdtd.Record`, `fdtd.monitors`)입니다.
`from wavebit.fdtd import Record`는 Python 자체의 `ImportError: cannot import name 'Record'`를,
`import wavebit.fdtd.monitors`는 `ModuleNotFoundError`를 냅니다. 이 두 형태에서는 Python이 모듈의
메시지를 버리기 때문이며, NumPy에서 지운 이름도 마찬가지입니다. 새 이름은 위의 표에 있습니다.

## 보고서 키와 이슈 경로

같은 이름 변경이 검사·준비·결과가 돌려주는 데이터에도 적용됩니다.

| 이전 | 이후 |
| --- | --- |
| `report.resolved['numerical_policies']['discretisation']`, `Result.provenance`의 같은 키 | `...['numerical_policies']['discretization']` |
| 이슈 경로 `numerics.discretisation` | `numerics.discretization` |
| 이슈 경로 `numerics.decay`, `numerics.decay.window_seconds` | `numerics.stopping`, `numerics.stopping.window_seconds` |

`Prepared.contract`는 살아 있는 소스 잎을 `amplitude/phase/delay_seconds`로 적습니다.

## Job 파일과 실행 기록

저장된 `Job`은 각 선언을 공개 이름으로 기록하므로 두 솔버 모두 스키마가 `wavebit.job/v2`가
되었습니다. 실행 설정은 하나뿐인 공개 경로 `wavebit.Execution`으로 태그됩니다(이전에는
`wavebit._common.execution.Execution`). 어느 솔버의 `wavebit.job/v1` 파일이든 더 이상 없는 이름으로
읽는 대신 코드 `job.schema`로 거부됩니다. 현재 선언으로 Job을 다시 만들어 저장하십시오. 스키마와
실행 태그가 키 해시에 들어가므로 RCWA job을 포함한 모든 job 키가 바뀝니다. 실행 기록도 같은 이유로
`wavebit.run/v2`이며, `wavebit.run/v1` 기록은 코드 `record.format`으로 거부됩니다. 이 변경으로
기능 매니페스트는 버전 23이 되었고, [아래](#native-grid-and-periodic-stack-names)의 native·periodic
이름 변경으로 버전 24가 되었습니다.

## 전체 예제

`examples/fdtd/pulse_slab.py`는 새 이름으로 막(film)에 펄스를 통과시킵니다.
[시작하기](../getting-started/fdtd.md)의 첫 계산도 같은 내용을 단계별로 보여 줍니다.

## Native-grid and periodic-stack names

`fdtd.native`(이미 Yee 위치에 표본화한 모델)와 `fdtd.periodic`(주기 적층)도 같은 날 두 번째
변경에서 같은 방식으로 한 번에 이름을 바꿨습니다. 별칭은 없으며, 옛 철자를 쓰면 실패하고 새
이름을 알려 줍니다. 두 모듈은 각자 자기 클래스를 가지므로 아래 표는 모든 이름을 모듈과 함께
적습니다. 몇몇 예제는 periodic 모듈을 `from wavebit.fdtd import periodic as fdtd`로 가져옵니다.
그 예제에서 `fdtd.Snapshots`는 general-FDTD 모니터가 아니라 `periodic.Snapshots`입니다.

### native·periodic 클래스

| 이전 | 이후 | 이유 |
| --- | --- | --- |
| `native.Record` | `native.Snapshots` | 격자의 장 스냅샷을 `every` 단계마다 보존합니다. 한 점을 기록하는 것은 `fdtd.TimeSeriesMonitor`입니다 |
| `periodic.Record` | `periodic.Snapshots` | 같은 일을 하며, 선택적인 물리 범위 안에서 보존합니다 |
| `native.Initial` | `native.InitialField` | 시작할 때의 장으로, `fdtd.InitialField`가 이미 쓰는 이름입니다 |
| `native.Tail` | `native.DecayCheck` | 마지막 창의 감쇠 진단으로, `fdtd.DecayStop`과 달리 실행을 멈추지 않습니다 |

`periodic.DecayCheck`는 새로 생겼습니다. 주기 적층에 쓰는 같은 선언이며, 다음 표에 있는
`periodic.Numerics`의 키워드 두 개를 대신합니다.

### native·periodic 키워드

| 이전 | 이후 |
| --- | --- |
| `native.Outputs(record=...)`, `periodic.Outputs(record=...)` | `native.Outputs(snapshots=...)`, `periodic.Outputs(snapshots=...)` |
| `native.Record(bounds=...)` | `native.Snapshots(slices=...)` |
| `native.Record` 또는 `periodic.Record`의 `components=('ex', 'hy')` | `native.Snapshots` 또는 `periodic.Snapshots`의 `components=('Ex', 'Hy')` |
| `native.Spectrum(planes=...)` | `native.Spectrum(z_indices=...)` |
| `native.Numerics(tail=native.Tail(window_seconds=w, tolerance=t))` | `native.Numerics(decay_check=native.DecayCheck(window_seconds=w, threshold=t))` |
| `periodic.Numerics(tail_steps=N, tail_tolerance=t)` | `periodic.Numerics(decay_check=periodic.DecayCheck(window_seconds=N*dt_seconds, threshold=t))` |
| `periodic.RasterDispersion(models=...)` | `periodic.RasterDispersion(poles=...)` |
| `periodic.Pulse(width_seconds=W)` | `periodic.Pulse(frequency_sigma_hz=1/(np.sqrt(2)*np.pi*W))` |

`slices=`는 native에만 있습니다. 선택자가 정수 격자 인덱스, 즉 `(z, y, x)` 각 축마다 인덱스 하나
또는 `(start, stop[, step])` 튜플이라는 것을 이름이 드러냅니다. `periodic.Snapshots`는 범위가
물리량이므로 `bounds=`를 유지합니다. `'x'`, `'y'`, `'z'`를 문제의 길이 단위로 적은 `(low, high)`
쌍에 대응시키는 mapping입니다.

성분은 Wavebit의 다른 모든 출력처럼 `'Ex'`부터 `'Hz'`까지로 적습니다. 소문자 `'ex'`는 거부되며,
메시지가 `'Ex'`를 알려 줍니다. `z_indices=`는 `planes=`에 들어 있던 값이 위치가 아니라 정수 z 노드
인덱스라는 것을 말해 줍니다. `poles=`에는 `DispersiveMaterial(poles=...)`이 받는 것과 같은
`materials.Drude`, `materials.Lorentz` 객체를 넣습니다. `threshold=`는 옛 `tolerance=`(native) 또는
`tail_tolerance=`(periodic)이며, 같은 수를 받고 기본값도 `1e-4`로 같습니다.

`Pulse(width_seconds=W)`는 포락선 `exp(-((t - t0)/W)**2)`의 시간 폭 `W`였습니다.
`frequency_sigma_hz`는 같은 가우시안을 `fdtd.GaussianPulse(frequency_sigma_hz=...)`처럼 진폭
스펙트럼의 표준편차(Hz)로 나타냅니다. 그래서 `fdtd`와 `fdtd.periodic`이 펄스를 같은 방식으로
기술합니다. 이 포락선에서는 sigma = 1/(sqrt(2)·pi·W)이므로, 옛 폭을 옮길 때는 이 식을 그대로
적으십시오. 펄스가 길수록 sigma가 작으므로, 폭을 줄이라고 하던 거부 메시지는 이제
`frequency_sigma_hz`를 키우라고 안내합니다. 기본 펄스는 그대로입니다. 1/e 반폭이 중심 파장의 한
주기인 포락선이며, sigma = f0/(sqrt(2)·pi)입니다.

### native·periodic 결과

| 이전 | 이후 |
| --- | --- |
| `native.Result.fields`, `periodic.Result.fields` | `native.Result.snapshots`, `periodic.Result.snapshots` |
| 키 `'ex'`부터 `'hz'`까지, 각 `TimeSeries`의 `components == ('ex',)` | 키 `'Ex'`부터 `'Hz'`까지, `components == ('Ex',)` |
| `native.Result.polarization` | `native.Result.polarization_density` |
| `native.Result.previous_polarization` | `native.Result.previous_polarization_density` |

`polarization_density`에는 `polarization`에 있던 값, 즉 극점마다의 재료 분극 밀도 P(C/m²)가
들어 있습니다. 마지막 단계의 값이며, `previous_polarization_density`는 그보다 한 단계 전의
값입니다. 이름은 이것이 파동의 편광(polarization)이 아니라 밀도임을 말해 줍니다. 옛 속성을 읽으면
`AttributeError`가 나며 새 이름을 알려 줍니다.

### 감쇠 창

periodic의 창은 단계 수 `tail_steps=N`이었습니다. 이제는 native에서 이미 그랬듯 시간 길이
`DecayCheck(window_seconds=...)`이며, 두 모듈 모두 같은 규칙으로 정수 개의 시간 단계에 맞춥니다.
규칙은 올림이지만, 몫이 어떤 정수보다 마지막 자리 몇 단위(ulp) 이내로만 크면 그 정수로 셉니다.
따라서 `N` 시간 단계로 적은 창은 `N` 단계를 덮습니다. `tail_steps=N`을 옮기려면 check에서 시간
간격을 얻습니다.

```python
numerics = periodic.Numerics(spacing=1/30, duration_seconds=90*units.time_unit_seconds)
dt_seconds = periodic.check(problem, numerics=numerics).resolved['time_step_seconds']
numerics = periodic.Numerics(spacing=1/30, duration_seconds=90*units.time_unit_seconds,
                             decay_check=periodic.DecayCheck(window_seconds=40*dt_seconds))
```

native의 규칙은 원래도 올림이었고, 이 허용 범위만 새로 생겼습니다. `N*time_step_seconds`로
적었지만 반올림 오차로 몫이 `N`보다 조금 커진 창은 전에는 `N + 1` 단계를 덮었고, 이제는 `N`
단계를 덮습니다. 이 차이는 `validity.convergence`의 감쇠 진단만 바꾸며 장이나 스펙트럼은 바꾸지
않습니다. 실행 길이와 같거나 더 긴 창은 거부되며, 거부 메시지는 `DecayCheck(window_seconds=...)`를
알려 줍니다.

### native·periodic 거부 메시지

```text
>>> native.Tail
AttributeError: wavebit.fdtd.native.Tail is now wavebit.fdtd.native.DecayCheck; given as Numerics(decay_check=...), with tolerance= now threshold=. The old name was removed, not aliased; see the public-name migration guide, docs/migration/public-names.md.
>>> periodic.Pulse(width_seconds=3e-15)
TypeError: Pulse(width_seconds=...) is now Pulse(frequency_sigma_hz=...); frequency_sigma_hz = 1/(sqrt(2)*pi*width_seconds), the spectral standard deviation of the same Gaussian, as in fdtd.GaussianPulse(frequency_sigma_hz=...). The old keyword was removed, not aliased; see the public-name migration guide, docs/migration/public-names.md.
```

import 형태는 [위](#옛-철자를-쓰면)와 같이 동작합니다. `from wavebit.fdtd.native import Record`는
Python 자체의 `ImportError`를 냅니다. 이슈 경로는 키워드를 따라 바뀝니다. `outputs.record`는
`outputs.snapshots`(native는 `outputs.snapshots.slices`, periodic은 `outputs.snapshots.bounds`),
`numerics.tail`(native)과 `numerics.tail_steps`(periodic)는 `numerics.decay_check`,
`dispersion.models`는 `dispersion.poles`입니다. 엔진 안에서 나는 거부 메시지는 엔진 자신의
키워드를 말했지만, 이제는 호출자가 적은 선언을 말합니다. `record_region`은
`Snapshots(slices=...)`, `monitor_z`는 `Spectrum(z_indices=...)`로 바뀌었고, periodic에서는
`record_bounds`, `pulse_width`, `n_periods`, `mesh_indices`, `truncation_order`가 각각
`Snapshots(bounds=...)`, `Pulse(frequency_sigma_hz=...)`, `Numerics(duration_seconds=...)`,
`Numerics(nominal_indices=...)`, `Numerics(orders=...)`로 바뀌었습니다. 타입 거부 메시지는 모듈을
밝힙니다. `expected Pulse`가 아니라 `expected periodic.Pulse`입니다.

### native·periodic Job 파일

저장된 Job은 general-FDTD 변경이 도입한 스키마 `wavebit.job/v2`를 그대로 씁니다. 두 변경 사이에
옛 native 또는 periodic 이름(`wavebit.fdtd.native.Record` 태그, 또는 `tail_steps`,
`width_seconds` 같은 필드)으로 저장한 v2 파일은 더 이상 없는 이름으로 읽는 대신, 알 수 없는
타입이나 필드를 밝히며 코드 `job.encoding`으로 거부됩니다. 현재 선언으로 Job을 다시 만들어
저장하십시오. 키가 해시하는 태그와 필드가 바뀌었으므로 모든 native·periodic job의 키가 바뀝니다
(모든 native job에는 `InitialField`와 `Outputs`가, 모든 periodic job에는 `Pulse`와 `Numerics`가
들어 있습니다). RCWA와 general-FDTD job의 키는 이 변경으로 바뀌지 않습니다. 기능 매니페스트는
버전 24입니다.

### native·periodic 수치

엔진은 바뀌지 않았고 같은 입력을 받습니다. 옛 철자와 새 철자를 um 단위로 나란히 실행했을 때
native 사례(E, H, 분극 밀도, 스냅샷, 스펙트럼, 7단계 감쇠 창)와 periodic 사례(R, T, r_s, 평면
페이저, 스냅샷, 펄스 폭, 40단계 감쇠 창)의 결과가 비트 단위로 같았습니다. 다를 수 있는 것은 두
가지입니다. 첫째, periodic 펄스 폭은 sigma에서 다시 계산하므로 sigma를 옛 폭의 식
`frequency_sigma_hz=1/(np.sqrt(2)*np.pi*W)`로 적었을 때에만 옛 폭과 맞고, 그때도 마지막 한두
비트 이내로 맞습니다. 위 사례에서는 정확히 같았고, 무작위 폭 10⁵개에서는 차이가 최대 마지막
자리 2단위(ulp)였습니다. 반올림한 십진수(um 시간 단위 세 개인 폭 `3*units.time_unit_seconds`에 대해
`2.25e13`)를 쓰면 폭이 조금 달라지고 수치도 조금 달라집니다. 둘째는 [감쇠 창](#감쇠-창)에서
설명한 native 반올림 허용 범위로, 감쇠 진단만 바꿉니다.

`examples/fdtd/fdtd_record.py`는 periodic 스냅샷을 얻고, `examples/fdtd/fdtd_grid_gradient.py`는
`Spectrum(z_indices=...)`에서 native 푸리에 적분을 미분합니다. 두 예제 모두 새 이름을 씁니다.

## Shared, observation and RCWA names

세 번째 일괄 변경(2026-10-20)은 두 솔버가 함께 쓰는 것, 결과가 돌려주는 객체, RCWA를 다룹니다.
이전과 같이 별칭은 없습니다. 옛 이름·키워드·속성을 쓰면 실패하고 새 이름을 알려 주며, 수치
계산은 바뀌지 않았습니다.

### 공통 모듈

| 이전 | 이후 | 이유 |
| --- | --- | --- |
| `Units.length_metres` | `Units.length_unit_meters` | 길이 단위 하나가 몇 미터인지; "meter" 철자를 하나로 통일 |
| `Units.time_seconds` | `Units.time_unit_seconds` | 내부 시간 단위 하나가 몇 초인지이며, 초 단위 시각이 아님 |
| `Units.to_metres(x)` | `x * units.length_unit_meters` | 곱셈 하나 |
| `Units.time_to_internal`, `time_from_internal`, `rate_to_internal`, `conductivity_to_internal` | 비공개 | 솔버가 자기 경계에서 하는 변환 |
| `Lattice.oblique(lx, ly, angle)` | `Lattice.oblique(length_u, length_v, angle)` | x·y 방향이 아닌 두 격자 벡터의 길이 |
| `is_axis_aligned`, `jacobian`, `order_kxy`, `periods_of`, `reduced_basis`, `reduction_transform`, `sample_grid`가 든 `wavebit.lattice.__all__` | `DrawnOn`, `Lattice`, `as_lattice`, `meshgrid_xy` | 나머지는 솔버를 위해 원시 기저를 다루는 도우미 |
| `Material.is_anisotropic` | `Material.is_tensor` | epsilon이나 mu를 주값 세 개나 3x3 텐서로 주었는지를 말하며, 등방 대각 텐서도 여기에 해당 |
| `TabulatedMaterial.interpolation` | `TabulatedMaterial.interpolated_quantity` | 보간 방법이 아니라 보간하는 양(`'epsilon'` 또는 `'index'`) |
| `Execution.scope()` | 비공개 | `run`, `check`, `prepare`가 직접 엶 |
| `wavebit.materials.time_domain`, `wavebit.materials.tabulated` | 비공개 | 그 안의 클래스는 솔버의 정규화 단위 모델이며, 공개 재료는 `wavebit.materials`에 있음 |
| `NumericalError`가 `ValidationError`(그리고 `ValueError`)였음 | `NumericalError`는 `WavebitError`이며 `ValidationError`가 아님 | 풀이 실패는 잘못된 입력이 아님. `WavebitError`가 모든 Wavebit 오류의 새 뿌리 |

`except ValueError`는 더 이상 `NumericalError`를 잡지 않습니다. `wavebit.NumericalError`나
`wavebit.WavebitError`를 잡으십시오. `ValidationError`는 여전히 `ValueError`입니다.

### 관측 객체

| 이전 | 이후 | 이유 |
| --- | --- | --- |
| `wavebit.SpectralFieldSamples` | `wavebit.RawFieldSpectrum` | `RawDiffractionSpectrum`처럼 가공하지 않은 푸리에 필드 |
| `TimeSeries.times`, `TimeSeries.time` | `TimeSeries.times_seconds` | 단위가 붙은 이름 하나 |
| `RawFieldSpectrum.frequencies`, `fdtd.SpectralFlux.frequencies` | `.frequencies_hz` | 단위를 붙임 |
| `RawFieldSpectrum.spectrum`, `RawDiffractionSpectrum.spectrum`, `SpectralFlux.metadata`의 키 `spectrum` | `.sidedness` | 스펙트럼이 아니라 `'two-sided'`를 담음 |
| `fdtd.SpectralFlux.energy_density` | `.spectral_energy` | 공간 밀도가 아니라 평면을 지나는 주파수당 에너지 |
| `fdtd.Result.sources` | `fdtd.Result.source_references` | 소스가 아니라 기록된 소스 기준값 |
| `DiffractionOrders.cartesian`, `DiffractionOrder.cartesian` | `.amplitude_xyz` | x, y, z 진폭 성분 |
| `DiffractionOrders.kz`, `DiffractionOrder.kz` | `.k_normal` | 포트 법선 방향 파수 성분(z 법선 포트이면 kz). 데카르트 부호를 그대로 쓰므로 −z 포트로 나가는 차수는 `Re(k_normal) < 0` |
| `DiffractionOrders.at_wavelength(i)` | `.at_wavelength_index(i)` | 파장이 아니라 인덱스를 받음 |
| `wavebit.records.RecordedFields` | `wavebit.records.RecordedObject` | 필드만이 아니라 기록된 어떤 객체든 담음 |
| `periodic.Result.sample(times=...)` | `sample(times_seconds=...)` | 단위를 붙임 |

생성자도 옛 키워드를 같은 방식으로 거부합니다(`TimeSeries(times=...)`는 `times_seconds=`를
알려 줍니다).

### RCWA

| 이전 | 이후 | 이유 |
| --- | --- | --- |
| `rcwa.OrderExcitation(orders=(((m, n), w), ...))` | `rcwa.IncidentOrders(weights=(((m, n), w), ...))` | 입사 차수와 각 차수의 복소 가중치 |
| `rcwa.GaussianBeam(spectral_extent=...)` | `rcwa.GaussianBeam(angular_spectrum_extent=...)` | 빔 각 스펙트럼의 범위(1/waist 단위) |
| `rcwa.sample_structure(n=...)` | `rcwa.sample_structure(samples=...)` | 표본 수 |
| `rcwa.operators.Port`의 필드 `kx`, `ky`, `index` | `kx_over_k0`, `ky_over_k0`, `refractive_index` | 정규화한 파수 성분과 포트 매질의 굴절률 |
| `rcwa.experimental` 빌더의 `truncation_order=` / `order=` | `orders=` | `rcwa.Numerics(orders=...)`와 같은 키워드 |
| `rcwa.BoundaryPower.integration_measure` | `rcwa.BoundaryPower.power_region` | 값은 그대로이며, 이름이 `Outputs(power_region=...)`와 맞음 |
| `rcwa.Result.diagnostics['source_groups']` | `diagnostics['source_group_counts']` | 결맞음 소스 묶음의 개수 |
| `wavebit.rcwa.layer_thicknesses` | `wavebit.layer_thicknesses` | FDTD도 씀 |
| `wavebit.rcwa.plotting` | 비공개 | 공개 결과가 아니라 엔진 솔버를 받음 |

풀이가 낼 수 있는 RCWA 경고는 공개 경로 하나, `wavebit.rcwa.diagnostics`에 있습니다:
`RayleighPointWarning`, `AutoDetuneWarning`, `ExceptionalPointWarning`,
`ComplexIncidenceWarning`, `GainCutoffWarning`, `SvdConvergenceWarning`.

`Port`는 named tuple이라 속성만 새 이름을 알려 줄 수 있습니다. `port.kx`는 `kx_over_k0`를
알려 주며 실패하지만, 알 수 없는 필드에 대해 `Port(kx=...)`는 파이썬 자체의 `TypeError`를,
`port._replace(kx=...)`는 파이썬 자체의 `ValueError`를 냅니다. 새 이름은 위 표에 있습니다.

`rcwa.check`는 이제 양의 유한 실수가 아닌 `GaussianBeam(angular_spectrum_extent=...)`를 코드
`source.angular_spectrum_extent`로 거부합니다. 같은 값은 전에도 check에서 거부됐지만, 코드
`rcwa.invalid`와 엔진의 표현으로 거부됐습니다. `IncidentOrders` 거부 코드는 `source.weights`입니다(전에는
`source.orders`).

`rcwa.check`와 `rcwa.run`의 거부 메시지는 엔진 키워드를 쓰던 곳에서 이제 공개 설정
(`Numerics(orders=...)`, `Numerics(detuning=...)`, `precision=`,
`GaussianBeam(angular_spectrum_extent=...)`)을 말합니다. RCWA 경고 문구는 아직 엔진 키워드를
씁니다. 이는 이후 변경에서 다룹니다.

### 공통·관측·RCWA Job 파일과 실행 기록

저장된 Job은 스키마 `wavebit.job/v2`를 그대로 씁니다. 옛 RCWA 이름(`wavebit.rcwa.OrderExcitation`
태그, 또는 `spectral_extent` 필드)이 든 v2 파일은 코드 `job.encoding`으로 거부됩니다. 현재
선언으로 Job을 다시 만들어 저장하십시오. `IncidentOrders`나 `GaussianBeam`이 든 RCWA job의 키가
바뀌고, 다른 job의 키는 바뀌지 않습니다. `TabulatedMaterial`은 여전히 필드 `interpolation`으로
저장되므로 이를 담은 job의 키도 그대로입니다.

실행 기록은 `wavebit.run/v2` 그대로입니다. 이제부터 저장하는 기록에는 새 필드 이름
(`times_seconds`, `frequencies_hz`, `sidedness`, `amplitude_xyz`, `k_normal`, `spectral_energy`,
`power_region`, `Port` 필드, `extra['source_references']`, 진단 키 `source_group_counts`)이
들어갑니다. 이전에 저장한 기록은 저장 당시의 이름을 유지합니다. `load_run`은 클래스를 다시 만들지
않고 저장된 필드를 `RecordedObject` 매핑으로 돌려주기 때문입니다. 다만 기록은 그 job을 읽을 수
있을 때만 읽힙니다. `OrderExcitation`이나 `GaussianBeam`이 든 job의 기록은 그 job 파일처럼 코드
`job.encoding`으로 거부됩니다. 기능 매니페스트는
버전 25입니다.

### 공통·관측·RCWA 수치

솔버는 같은 입력을 받습니다. FDTD 계획 골격은 바뀌지 않았습니다. 비공개가 된 materials 모듈은
여전히 옛 이름으로 기록됩니다. 기준 골든과 솔버 허용 오차는 하나도 바뀌지 않았습니다.

## Result and report keys

네 번째 변경(2026-10-20)은 모든 솔버가 보고하는 같은 사실에 키 하나씩을 줍니다. 같은 검사, 시간
간격, 기준 값, 오류가 `rcwa`, `fdtd`, `fdtd.native`, `fdtd.periodic`에서 같은 이름을 갖습니다. 키는
별칭 없이 이름만 바뀌므로 옛 키를 읽으면 `KeyError`가 납니다. 수치 계산은 바뀌지 않았습니다.

### 검증, 시간 간격, 감쇠

| 이전 | 이후 |
| --- | --- |
| native `validity.checks['stable']`, `['finite_fields']` | `['runtime_stability']`, `['numerical_finite']` |
| periodic `validity.checks['stable']`, `['finite']`(R만) | `['runtime_stability']`, `['numerical_finite']`(R과 T) |
| general FDTD: 유한성 검사 없음 | 모든 모니터 출력과 최종 필드에 대한 `validity.checks['numerical_finite']` |
| `fdtd` `report.resolved['dt_seconds']`, `provenance['dt_seconds']`, `contract['resolved_dt_seconds']` | `'time_step_seconds'`, `'time_step_seconds'`, `'resolved_time_step_seconds'` |
| native `diagnostics['tail']`; `validity.convergence`가 그 보고 자체 | `diagnostics['decay']`; `validity.convergence == {'decay': 보고 또는 None}` |
| periodic `diagnostics['convergence']`; `validity.convergence`가 그 보고에 `residual`을 더한 것 | `diagnostics['decay']`; `validity.convergence == {'decay': 보고 또는 None, 'residual': ...}` |
| 감쇠 보고 안의 `tolerance`, periodic `units` 하위 매핑 | `threshold`(`DecayCheck(threshold=)`와 같음), `<양>_units` 키 |
| general FDTD의 개수 `invariant_axes`; periodic `diagnostics['invariant_y']` | 축 이름 튜플 `invariant_axes`, 없으면 `()` |

`fdtd.periodic`의 `algorithm='reference'`는 안정성을 판단하지 않으므로
`validity.checks['runtime_stability']`가 `None`(판단 안 함)입니다. 전에는 기본값으로 `True`가
나왔습니다.

### provenance

모든 솔버의 `provenance`에 `solver`, `execution`, `numerics`, `units`, `coordinate_frame`,
`time_convention`(`'exp(-i*omega*t)'`, 철자 하나)이 있습니다. general FDTD에는 평면 키 `backend`,
`device`, `precision`이 더 이상 없습니다. `provenance['execution'].device`를 읽으십시오.
`provenance['units']`는 `Units` 선언입니다. native와 periodic은 `time_step_seconds`도 보고합니다.
periodic의 `coordinate_convention`은 `coordinate_frame`입니다. rcwa의 `code_version`은 설치된
패키지 버전입니다(전에는 고정값 `'0.1.0'`).

### 회절 기준 값

`DiffractionOrders.reference`(와 `RawDiffractionSpectrum.reference`)는 모든 생산자에서 핵심 키가
같습니다.

| 키 | 뜻 | 이전 |
| --- | --- | --- |
| `port` | 차수가 나가는 면: rcwa와 `fdtd.periodic`은 `'top'`/`'bottom'`, `fdtd`는 `'z+'` 같은 면 | fdtd `port`는 `'reflect'`/`'transmit'`; periodic `side` |
| `wave` | `'reflected'` 또는 `'transmitted'` | 새로 추가 |
| `plane` | 법선 방향의 포트 평면 위치(길이 단위) | rcwa와 periodic `z` |
| `length_units` | `plane`과 좌표의 길이 단위 | `length_unit`; periodic `coordinates` |
| `lateral_origin` | 좌표 | periodic `'cell_centre'`(이제 `(0., 0.)`) |
| `incident_amplitude`, `incident_amplitude_units` | 입사 진폭과 그 단위 | rcwa `incident_electric_amplitude_v_per_m` |
| `cell_measure`, `cell_measure_units` | 주기 셀의 측도: `'m^2'`, 또는 불변 축마다 단위 길이당 `'m'`/`'1'` | `cell_area_m2`; 숫자 `integration_measure(_units)` |
| `invariant_axes` | 축 이름 | 개수 |

rcwa에는 `incident_power`와 `incident_power_units`도 있습니다(전에는 `incident_power_w`). 기준 값의
단위 라벨(`amplitude_units`, 전력이 있으면 `power_units`, 파수가 있으면 `wavevector_units`)은 모든
생산자에서 `reference`에 있습니다. 무차원 진폭의 라벨은 `'1'`입니다. rcwa metadata의 문자열 `integration_measure`
(`'periodic cell'`)는 없어졌습니다.

### 단위와 표본 시각

- 단위 라벨 키는 `_units`로 끝납니다. `length_unit` → `length_units`, `time_unit` → `time_units`,
  native `E_unit`/`H_unit`/`polarization_unit`/`spectral_power_unit` → `*_units`, periodic 평면
  metadata `power_unit` → `power_units`. 단위마다 ASCII 철자 하나: `'V*s/m'`, `'A*s/m'`,
  `'C/m^2'`, `'W*s^2'`.
- `FieldSamples.units`(와 `TimeSeries`, `RawFieldSpectrum`)는 어디서나 한 모양의 매핑입니다. 있는
  성분마다(`'Ex'`~`'Hz'`) 하나씩과 `'coordinates'`. 전에는 fdtd `TimeSeries.units`가 문자열, native
  스냅샷이 `{'field', 'coordinates', 'time'}`, 최종 필드가 `{'E', 'H'}`였습니다.
- E와 H 표본 시각: fdtd `final_fields` metadata `E_time`/`H_time`과 native
  `diagnostics['time_seconds']`/`['magnetic_time_seconds']` → `E_time_seconds`/`H_time_seconds`.
- `fdtd.Result.source_references` 항목: `time` → `times_seconds`, `magnetic_time` →
  `magnetic_times_seconds`, `field_time` → `field_times_seconds`, `incident` → `incident_E`,
  `plane` → `plane_E`(`incident_H`, `plane_H` 옆), `time_unit`/`length_unit` →
  `time_units`/`length_units`; `field_units`는 항상 매핑.
- `diagnostics['grazing']` 행은 공개 이름 `polarization(s)`, `stabilizing_loss`,
  `invariant_axes`(축 이름)를 씁니다. 전에는 `polarisation(s)`, `stabilize`, `axis1d`였습니다.
- 방출 값은 미국 철자를 씁니다. 기능 매니페스트의 `'cell_center'`와 `omitted_center`,
  `rcwa.operators` metadata의 `'cell center (0,0)'`, rcwa provenance의 `'explicit cell-centered
  operators'`.

### 오류 코드

모든 공개 거부 코드는 `area.item`(소문자, 점 구분) 형태이며 [keywords](../reference/keywords.md)에
목록이 있습니다. 이 형태가 아니던 코드는 이름이 바뀌었습니다. 눈에 띄는 것:

| 이전 | 이후 |
| --- | --- |
| `invalid_input`(기본값, 약 100곳) | 구체적 코드(`monitor.plane`, `numerics.duration`, `source.name_duplicate` 등); 기본값은 이제 `input.invalid` |
| fdtd `prepared_structure`; native와 periodic `invalid_input` | 모든 솔버에서 `prepared.structure`, 바뀐 선언의 경로와 함께(`scene.domain`, `stack.layers` 등) |
| fdtd `prepared_dependencies`, `static_input`, `cuda_scope` | `prepared.dependencies`, `prepared.static_input`, `execution.cuda_scope` |
| fdtd `checkpoint_steps`, `checkpoint_scope`, `fdtd_input` | `numerics.checkpoint_steps`, `numerics.checkpoint_scope`, `fdtd.invalid` |
| fdtd `plane_span`, `monitor_extent`, `source_waveform_range`, `padding_continuation` | `source.plane_span` 또는 `monitor.plane_span`, `monitor.extent`, `source.waveform_range`, `fdtd.padding_continuation` |
| `backend`, `device`, `precision`, `device_unavailable`, `jax_x64_disabled` | `execution.backend`, `execution.device`, `execution.precision`, `execution.device_unavailable`, `execution.jax_x64` |
| `unknown_length_unit`, `units_required` | `units.length`, `units.type` |
| `material_scalar`, `material_shape`, `material_poles`, `material_wavelength`, `fdtd_material` | `material.scalar`, `material.shape`, `material.poles`, `material.wavelength_range`, `material.fdtd` |
| rcwa `output.unavailable` | `outputs.unavailable` |

`fdtd.native`와 `fdtd.periodic`의 `prepare`와 `run`은 엔진 거부가 맨 `ValueError`로 새어 나가던 곳에서
이제 코드가 붙은 `ValidationError`(`native.invalid`, `periodic.invalid` 또는 구체적 코드)를 냅니다.
`check`는 전부터 같은 실패를 보고했습니다.

### `orders`는 명시적 쌍

`orders=`는 모든 공개 호출(`rcwa.Numerics`, `fdtd.periodic.Numerics`, `fdtd.DiffractionMonitor`,
`rcwa.experimental` 빌더)에서 두 격자 벡터 방향의 반폭 쌍 `(Mu, Mv)`입니다. 맨 정수는 코드
`numerics.orders`, `monitor.orders`, `experimental.orders`로 거부되며, 메시지가 써야 할 쌍을
알려 줍니다. 전에는 `fdtd.DiffractionMonitor`와 실험 빌더에서 `(M, M)`, `fdtd.periodic.Numerics`에서
`(M, 0)`을 뜻했고, `rcwa.Numerics`는 맨 `TypeError`로 멈췄습니다. `orders=0`은 `orders=(0, 0)`으로
쓰십시오. NumPy 정수는 받고 `True`/`False`는 받지 않습니다. 실험 빌더의 기본값은 `(3, 3)`과
`(1, 1)`이고, 전송 gap cell의 `closure()` metadata 키 `order`는 `orders`입니다.

### 결과 키 Job 파일, 실행 기록, 수치

Job 필드 이름은 바뀌지 않으므로 job 키와 `wavebit.job/v2`는 그대로입니다. 예외가 하나 있습니다.
`orders`를 맨 정수로 저장한 job(`fdtd.DiffractionMonitor`, `fdtd.periodic.Numerics`)은 선언 자체와
마찬가지로 코드 `job.encoding`으로 거부됩니다. 원래 뜻한 쌍(모니터는 `(M, M)`, periodic은 `(M, 0)`)으로
쓰십시오. 실행 기록은 `wavebit.run/v2` 그대로입니다. 기록은 결과의 매핑을 그대로 저장하므로 이전에
저장한 기록은 옛 키를 유지하고 지금 저장하는 기록에는 새 키가 들어갑니다. 기록은 그 job을 읽을 수
있을 때만 읽힙니다. 기능 매니페스트는 버전 26입니다. 수치, 계획
골격, 기준 골든은 하나도 바뀌지 않았습니다.
