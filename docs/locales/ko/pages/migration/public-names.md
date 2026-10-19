# `wavebit.fdtd`의 공개 이름 (2026-10-19)

Wavebit은 아직 알파 단계이므로 general-FDTD 선언의 이름을 한 번에 바꿨습니다. 이제 모든
클래스는 그것이 무엇인지로, 모든 모니터는 무엇을 기록하는지로 이름이 붙고, 단위가 분명하지 않은
키워드에는 단위가 붙습니다. 옛 이름은 별칭으로 남기지 않고 제거했습니다. 옛 속성(`fdtd.Record`)이나
키워드(`discretisation=`)를 쓰면 바로 실패하고 새 이름을 알려 줍니다(import 형태는
[옛 철자를 쓰면](#옛-철자를-쓰면) 참고). 수치 계산은 바뀌지 않았습니다. 낮춘(lowered) 장면, 정적 계획,
모든 출력이 같습니다(`tests/fdtd/test_public_names.py`가 이름을 바꾸기 전에 측정한 여덟 장면의
계획 골격을 고정합니다).

설계와 그 이유는 [이슈 #57](https://github.com/fieldalgo/wavebit/issues/57)에 있습니다. 이
페이지는 `wavebit.fdtd`를 다룹니다. `fdtd.native`, `fdtd.periodic`, RCWA와 공통 네임스페이스는
이후 변경에서 이어지며 이 페이지에 추가됩니다.

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
`wavebit.run/v2`이며, `wavebit.run/v1` 기록은 코드 `record.format`으로 거부됩니다. 기능 매니페스트는
버전 23입니다.

## 전체 예제

`examples/fdtd/pulse_slab.py`는 새 이름으로 막(film)에 펄스를 통과시킵니다.
[시작하기](../getting-started/fdtd.md)의 첫 계산도 같은 내용을 단계별로 보여 줍니다.
