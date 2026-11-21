# FDTD API

[실행 가능한 첫 계산](../../getting-started/fdtd.md)과 [공통 생애주기](../../guide/common/lifecycle.md)를 보세요.
2026-10-19에 바뀐 이름은 [공개 이름 마이그레이션 가이드](../../migration/public-names.md)에 있습니다.

## 생애주기

`Problem`을 선언하고, 검사하고, 한 번 준비한 뒤 실행합니다.

```{eval-rst}
.. autofunction:: wavebit.fdtd.check

.. autofunction:: wavebit.fdtd.prepare

.. autofunction:: wavebit.fdtd.run

.. autoclass:: wavebit.fdtd.Problem
   :members:

.. autoclass:: wavebit.fdtd.Prepared
   :members:
```

## 장면과 형상

물리 도메인과 그 안의 구조물입니다. `Sphere`, `Box`, `Cylinder`, `Prism`은 스무딩 바디이고, `SnappedSlab`과 `SnappedBox`는 Yee 셀을 통째로 채웁니다.

```{eval-rst}
.. autoclass:: wavebit.fdtd.Scene
   :members:

.. autoclass:: wavebit.fdtd.Domain
   :members:

.. autoclass:: wavebit.fdtd.Structure
   :members:

.. autoclass:: wavebit.fdtd.Sphere
   :members:

.. autoclass:: wavebit.fdtd.Box
   :members:

.. autoclass:: wavebit.fdtd.Cylinder
   :members:

.. autoclass:: wavebit.fdtd.Prism
   :members:

.. autoclass:: wavebit.fdtd.SnappedSlab
   :members:

.. autoclass:: wavebit.fdtd.SnappedBox
   :members:

.. autoclass:: wavebit.fdtd.Plane
   :members:

.. autoclass:: wavebit.fdtd.Shape
   :members:
```

## 경계

면마다 물리 조건 하나를 둡니다. `Open`을 구현하는 흡수층은 수치적 선택입니다.

```{eval-rst}
.. autoclass:: wavebit.fdtd.Boundaries
   :members:

.. autoclass:: wavebit.fdtd.Open
   :members:

.. autoclass:: wavebit.fdtd.Periodic
   :members:

.. autoclass:: wavebit.fdtd.Bloch
   :members:

.. autoclass:: wavebit.fdtd.PEC
   :members:

.. autoclass:: wavebit.fdtd.PMC
   :members:

.. autoclass:: wavebit.fdtd.MirrorSymmetry
   :members:

.. autoclass:: wavebit.fdtd.InvariantAxis
   :members:
```

## 소스

라디안, 초, SI 단위 진폭을 씁니다.

```{eval-rst}
.. autoclass:: wavebit.fdtd.GaussianPulse
   :members:

.. autoclass:: wavebit.fdtd.PlaneWave
   :members:

.. autoclass:: wavebit.fdtd.CurrentDipole
   :members:

.. autoclass:: wavebit.fdtd.TFSF
   :members:

.. autoclass:: wavebit.fdtd.InitialField
   :members:
```

## 모니터

각 모니터의 이름은 그것이 기록하는 대상을 따릅니다.

```{eval-rst}
.. autoclass:: wavebit.fdtd.Outputs
   :members:

.. autoclass:: wavebit.fdtd.TimeSeriesMonitor
   :members:

.. autoclass:: wavebit.fdtd.FieldMonitor
   :members:

.. autoclass:: wavebit.fdtd.FluxMonitor
   :members:

.. autoclass:: wavebit.fdtd.DiffractionMonitor
   :members:

.. autoclass:: wavebit.fdtd.FarFieldMonitor
   :members:
```

## 수치 설정

격자, 실행 시간, 흡수층, 종료 규칙: 물리 문제를 바꾸지 않는 선택입니다.

```{eval-rst}
.. autoclass:: wavebit.fdtd.Numerics
   :members:

.. autoclass:: wavebit.fdtd.UniformGrid
   :members:

.. autoclass:: wavebit.fdtd.Duration
   :members:

.. autoclass:: wavebit.fdtd.PML
   :members:

.. autoclass:: wavebit.fdtd.ConductiveAbsorber
   :members:

.. autoclass:: wavebit.fdtd.DecayStop
   :members:
```

## 결과

```{eval-rst}
.. autoclass:: wavebit.fdtd.Result
   :members:

.. autoclass:: wavebit.fdtd.SpectralFlux
   :members:

.. autoclass:: wavebit.fdtd.FarFieldSpectrum
   :members:
```

## 도파관 모드

균일한 단면을 가로지르는 평면 위에서 구한 이산 Yee 계의 모드입니다.
[도파관 모드 가이드](../../guide/fdtd/waveguide-modes.md)를 보세요.

```{eval-rst}
.. autofunction:: wavebit.fdtd.solve_modes

.. autoclass:: wavebit.fdtd.WaveguideModes
   :members:

.. autoclass:: wavebit.fdtd.WaveguideMode
   :members:

.. autoclass:: wavebit.fdtd.ModeMonitor
   :members:

.. autoclass:: wavebit.fdtd.ModeSpectrum
   :members:

.. autoclass:: wavebit.fdtd.ModeSource
   :members:

.. autoclass:: wavebit.fdtd.GaussianBeam
   :members:
```
## Native 격자와 주기 적층 정식화

[native 격자의 단위와 관측](../../guide/fdtd/native-grids.md)과
[주기 적층의 단위와 관측](../../guide/fdtd/periodic-problems.md)을 보세요. 이 선언들도
2026-10-19에 이름이 바뀌었습니다(`native.Record`와 `periodic.Record`는 이제 `Snapshots`,
`native.Tail`은 `native.DecayCheck` 등).
[마이그레이션 가이드](../../migration/public-names.md#native-grid-and-periodic-stack-names)를 보세요.

```{eval-rst}
.. automodule:: wavebit.fdtd.native
   :members:

.. automodule:: wavebit.fdtd.periodic
   :members: Problem, RasterDispersion, Pulse, Outputs, Snapshots, Numerics, DecayCheck, Prepared, Result, check, prepare, run

.. automodule:: wavebit.observations.modes
   :members:
```
