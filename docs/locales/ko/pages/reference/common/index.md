# 공통 API

물리 재료와 단위는 두 솔버가 함께 씁니다. 공통 선언이 어느 솔버의 지원 수치 범위를 넓혀 주지는 않으며, 그 범위는 `check`가 명시적으로 보고합니다.

```{eval-rst}
.. autoclass:: wavebit.Material
   :members:

.. autoclass:: wavebit.MaterialResponse
   :members:

.. autoclass:: wavebit.ConductiveMaterial
   :members:

.. autoclass:: wavebit.DispersiveMaterial
   :members:

.. autoclass:: wavebit.Drude
   :members:

.. autoclass:: wavebit.Lorentz
   :members:

.. autoclass:: wavebit.Debye
   :members:

.. autoclass:: wavebit.TabulatedMaterial
   :members:

.. autofunction:: wavebit.eps_from_refractive_index

.. autofunction:: wavebit.refractive_index_from_eps

.. autofunction:: wavebit.layer_thicknesses

.. autoclass:: wavebit.Units
   :members:

.. autoclass:: wavebit.Lattice
   :members:

.. autoclass:: wavebit.Execution
   :members:

.. autoclass:: wavebit.Backend
   :members:

.. autoclass:: wavebit.CheckReport
   :members:

.. autoclass:: wavebit.Job
   :members:

.. autofunction:: wavebit.run

.. autoclass:: wavebit.DiffractionOrders
   :members:
   :inherited-members:

.. autoclass:: wavebit.observations.DiffractionOrder
   :members:
   :inherited-members:

.. autoclass:: wavebit.FieldSamples
   :members:

.. autoclass:: wavebit.RawFieldSpectrum
   :members:

.. autoclass:: wavebit.RawDiffractionSpectrum
   :members:
   :inherited-members:

.. autoclass:: wavebit.observations.RawDiffractionOrder
   :members:

.. autoclass:: wavebit.TimeSeries
   :members:

.. autoclass:: wavebit.Validity
   :members:

.. autoclass:: wavebit.WavebitError
   :members:

.. autoclass:: wavebit.ValidationError
   :members:

.. autoclass:: wavebit.UnsupportedFeatureError
   :members:

.. autoclass:: wavebit.IncompatibleProblemError
   :members:

.. autoclass:: wavebit.NumericalError
   :members:

.. autoclass:: wavebit.Issue
   :members:

```

## 측정 실행 기록

데이터만 읽는 불러오기 계약, 명시적 필드 표본화, 소스 출처 범위는
[연구 계산 저장과 결과 확인](../../guide/run-records.md)을 보세요.

```{eval-rst}
.. autofunction:: wavebit.records.run_recorded

.. autofunction:: wavebit.records.save_run

.. autofunction:: wavebit.records.load_run

.. autoclass:: wavebit.records.RunRecord
   :members:

.. autoclass:: wavebit.records.RecordedObject
   :members:
```
