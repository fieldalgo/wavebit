# API 레퍼런스

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 있어야 동작합니다.
> 현재 설치본에는 이 네임스페이스가 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행하지 않았습니다.


:::{note}
이 페이지는 예전 계산을 해석하는 데 쓰는 정규화된 내부 엔진 레퍼런스입니다. 여기에 나오는 비공개 경로는 구현 세부 사항입니다. `wavebit.compat`은 제거되었으며, 예전 공개 표기는 아래에 링크한 고정된 과거 체크아웃에서만 쓸 수 있습니다. 새 애플리케이션은 [현재 생애주기](../guide/common/lifecycle.md)와 [현재 API 레퍼런스](../reference/api.md)로 시작하세요. 여기에 인용한 수치 근거는 원래의 적용 범위를 그대로 유지합니다.
:::

수치 근거를 확인하는 기여자를 위해, 남아 있는 엔진 docstring에서 생성한 페이지입니다.
애플리케이션 코드에는 물리 단위의 공개 인터페이스를 쓰세요. 과거 공개 API 문서는 소스 리비전에
보존되어 있으며, 또 하나의 런타임 네임스페이스로 다시 내보내지 않습니다.

## 솔버 만들기

```{eval-rst}
.. autofunction:: wavebit.rcwa._factory.rcwa

```

`fdtd()`는 또 하나의 입구입니다. 이 함수는 솔버가 아니라 답을 돌려줍니다. 인스턴스마다 유지할
상태가 없기 때문입니다. 캐시할 고유값 분해도, 고정된 형상도, 그래디언트도 없습니다.

```{eval-rst}
.. autofunction:: wavebit.fdtd._periodic.api.fdtd

```


## 솔버 객체

`rcwa()`는 백엔드에 따라 클래스가 정해지는 솔버를 돌려줍니다. 다음은 그 모든 솔버가 갖는 메서드와
속성입니다.

```{eval-rst}
.. autoclass:: wavebit.rcwa._engine.engine.CoreFactorizationMixin
   :no-index:
   :members: solve, reconstruct_field, freeze_geometry, solve_conv, eps_conv_matrices, mu_conv_matrices, gain_cutoff_band, factorization, fourier_coefficients, raster_normal_method, nvf_projector, li_nesting, half_space_merge_tolerance, exceptional_points, shape_nvf_grid_size


.. autoclass:: wavebit.rcwa._engine.engine.CoreSolveMixin
   :no-index:
   :members: answers, theta, kx, ky, excitation, scattering, truncation_order, period, lattice


```

## 장면 기술하기

```{eval-rst}
.. autoclass:: wavebit.materials.response.Material
   :no-index:
   :members:

.. autofunction:: wavebit.materials.response.eps_from_refractive_index

.. autofunction:: wavebit.materials.response.refractive_index_from_eps

.. autofunction:: wavebit._common.periodic.layer_thicknesses

.. autofunction:: wavebit.fdtd._periodic.legacy.permittivity_at

.. autoclass:: wavebit.lattice.Lattice
   :no-index:
   :members:

.. autofunction:: wavebit.lattice.meshgrid_xy

.. autoclass:: wavebit._common.periodic.Layer
   :no-index:
   :members:

.. autoclass:: wavebit._common.periodic.Stack
   :no-index:
   :members:

.. autoclass:: wavebit._common.periodic.Incidence
   :no-index:
   :members:

```


## 데이터로서의 실행

```{eval-rst}
.. autoclass:: wavebit._common._jobs_v1.JobSpec
   :no-index:
   :members:

.. autoclass:: wavebit._common._jobs_v1.Numerics
   :no-index:
   :members:

.. autoclass:: wavebit._common._jobs_v1.Sweep
   :no-index:
   :members:

.. autofunction:: wavebit._common._jobs_v1.run

.. autoclass:: wavebit._common._jobs_v1.SpecError
   :no-index:
   :members:

```


## 실험적 보조 NVF

이 명시적 NumPy 인터페이스는 일반 NVF 경로 선택을 바꾸지 않으며
광학적 수렴을 보증하지도 않습니다. [선택 가이드](../guide/rcwa/nvf-method-selection.md)를 보세요.

```{eval-rst}
.. autoclass:: wavebit.rcwa._engine.conforming.cartesian_pair_opening.CartesianPairOpeningPlan
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin.CartesianPairTransmissionGalerkinGapCell
   :no-index:
   :members: closure
   :inherited-members:

```

## Results and state

```{eval-rst}
.. autoclass:: wavebit.rcwa._engine.scattering.ScatteringMatrix
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.scattering.Port
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.geometry.StaticGeometry
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.geometry.StaleGeometryError
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.factorization.ConvMatrices
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.anisotropy.TensorConvMatrices
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.conforming.ConformingNVF
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.conforming.affine_profile.AffineProfileNVF
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.conforming.cartesian_rectangle.CartesianRectangleNVF
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.conforming.cartesian_pair.CartesianRectanglePairNVF
   :no-index:
   :members:

```


## 재료 데이터

지정과 평가는 [재료와 파장 스윕](../guide/common/materials.md)을 보세요.

```{eval-rst}
.. autoclass:: wavebit.materials.TabulatedMaterial
   :no-index:
   :members:

.. autofunction:: wavebit.materials.catalog.load

.. autofunction:: wavebit.materials.catalog.available

```


## 과도 응답 점 프로브

```{eval-rst}
.. autofunction:: wavebit.fdtd._periodic.probes.sample_fdtd_fields

.. autofunction:: wavebit.fdtd._periodic.probes.fdtd_poynting

.. autofunction:: wavebit.fdtd._periodic.probes.fit_fdtd_modes
```

## 그림 그리기

```{eval-rst}
.. autofunction:: wavebit.rcwa._plotting.plot_structure

.. autofunction:: wavebit.rcwa._plotting.slice_structure

.. autofunction:: wavebit.fdtd._periodic.plotting.plot_fdtd_field

.. autofunction:: wavebit.fdtd._periodic.plotting.animate_fdtd_field

```


## 정밀도와 경고를 내는 것

```{eval-rst}
.. autofunction:: wavebit._common.backend._enable_x64


.. autoclass:: wavebit._common.backend.PrecisionLossWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.UnresolvedBranchWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.RayleighPointWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.AutoDetuneWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.AxisNameWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.ComplexIncidenceWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.GainCutoffWarning
   :no-index:
   :members:

```

## 면별로 분해한 결맞은 조명

물리 규약은 [가이드와 흐름도](../guide/rcwa/coherent-illumination.md)를 보세요.

```{eval-rst}
.. autoclass:: wavebit.rcwa._engine.sources.Source
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.sources.PlaneWave
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.sources.OrderExcitation
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.sources.GaussianBeam
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.sources.ElectricDipole
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.illumination.CoherentSolver
   :no-index:
   :members: solve, reconstruct_field

.. autoclass:: wavebit.rcwa._engine.illumination.CoherentResult
   :no-index:
   :members: boundary_fields, boundary_power, reconstruct_field, output_top, output_bottom, absorption

.. autoclass:: wavebit.rcwa._engine.illumination.BoundaryPower
   :no-index:
   :members:
```

## Native 격자 시간 영역

```{eval-rst}
.. autofunction:: wavebit.fdtd._periodic.grid.fdtd_grid
```
