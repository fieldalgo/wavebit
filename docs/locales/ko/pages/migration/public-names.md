# `wavebit.fdtd`의 공개 이름 (2026-10-19)

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
RCWA와 공통 네임스페이스는 이후 변경에서 이어지며 이 페이지에 추가됩니다.

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
numerics = periodic.Numerics(spacing=1/30, duration_seconds=90*units.time_seconds)
dt_seconds = periodic.check(problem, numerics=numerics).resolved['time_step_seconds']
numerics = periodic.Numerics(spacing=1/30, duration_seconds=90*units.time_seconds,
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
자리 2단위(ulp)였습니다. 반올림한 십진수(um 시간 단위 세 개인 폭 `3*units.time_seconds`에 대해
`2.25e13`)를 쓰면 폭이 조금 달라지고 수치도 조금 달라집니다. 둘째는 [감쇠 창](#감쇠-창)에서
설명한 native 반올림 허용 범위로, 감쇠 진단만 바꿉니다.

`examples/fdtd/fdtd_record.py`는 periodic 스냅샷을 얻고, `examples/fdtd/fdtd_grid_gradient.py`는
`Spectrum(z_indices=...)`에서 native 푸리에 적분을 미분합니다. 두 예제 모두 새 이름을 씁니다.
