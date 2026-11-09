# 물리 단위 주기 적층 FDTD

`from wavebit.fdtd import periodic`은 불변 객체 `Problem`, `Numerics`와 `check`, `prepare`,
`run`을 제공합니다. 이식 가능한 job은 `solver='fdtd.periodic'`을 씁니다. 2026-10-19에 바뀐
이름은 [공개 이름 이전 가이드](../../migration/public-names.md#native-grid-and-periodic-stack-names)에
있습니다.
Stack, Layer, Region, 형식이 있는 도형, Excitation, PlaneWave는 RCWA와 같은 물리 선언입니다.
층은 아래에서 위로 쌓이고, z=0은 아래 경계면, x=y=0은 셀 중심입니다.

```python
from wavebit import Units
from wavebit.fdtd import periodic as fdtd
import numpy as np

units = Units(length='um')
problem = fdtd.Problem(
    units=units,
    stack=fdtd.Stack(period=(.2,), layers=(
        fdtd.Layer(thickness=.1, epsilon=np.full((1,2), 2.25)),)),
    excitation=fdtd.Excitation(wavelength=(.85,.9,.95),
                               sources=(fdtd.PlaneWave(),)))
numerics = fdtd.Numerics(spacing=1/30,
                         duration_seconds=90*units.time_unit_seconds)
result = fdtd.run(problem, numerics=numerics)
print(result.R.sum(axis=(-2,-1)), result.T.sum(axis=(-2,-1)))
```

공간량은 선언한 길이 단위를 씁니다. 시간과 재료의 rate는 SI 단위입니다.
`duration_seconds`는 완전한 시간 단계 수로 올림합니다. `Pulse(frequency_sigma_hz=...)`와
`notch_frequencies_hz`는 물리 가우시안 광원을 조절합니다. sigma는 `fdtd.GaussianPulse`와 같이
진폭 스펙트럼의 표준편차(Hz)이며, 기본 포락선의 1/e 반폭은 중심 파장에서 한 주기입니다
(sigma = f0/(sqrt(2)·pi)). 한 대역은 중심 파장에서 정한
횡방향 Bloch 파수벡터를 공유하므로, 대역 안의 입사각은 일반적으로 서로 다릅니다.
`diagnostics['angles']`가 이를 라디안으로 보고합니다. Theta/phi는 지원하고, 데카르트 kx/ky
입사는 거부합니다.

`Numerics(orders=(Mu, Mv))`는 회절 차수 창으로, 두 격자 벡터 방향의 반폭입니다. 행이 하나인
래스터에서는 `v != 0` 차수가 모두 항등적으로 0이므로 창이 `(M, 0)`입니다. 그런 셀에서 0이 아닌
`Mv`를 돌려줄 수 없는 경우에는 `check`, `prepare`, `run`에서 `orders=(M, 0)`을 알려 주는
메시지로 거절합니다. `algorithm='reference'`는 항상 거절하고(그런 셀의 창에 `v` 축이 없음),
`algorithm='shared'`는 1차원 `period=(P,)`이거나 평면 내 입사(`phi`가 None 또는 0)일 때
거절합니다. 이때 격자는 해상도와 관계없이 y 셀을 두 개만 둡니다. 같은 격자에서 RCWA도 같은 창을
거절합니다. 2차원 주기에서 0이 아니거나 배열인 방위각이면 shared 격자가 y를 분해해 `v` 차수를
돌려주며, 이 차수들은 `(M, 0)` 답 옆에서 반올림 오차 수준의 0입니다.

이전 결과를 재현할 때는 수치 알고리즘을 명시적으로 고르세요.

| 알고리즘 | 실행 방식과 목적 |
| --- | --- |
| `shared` (기본값) | NumPy/JAX/Torch 공통 Yee 엔진. 고정 명목 메시와 재료·도형·각도 미분 지원 |
| `reference` | 이전 주기 계산, 비균일(graded) 메시, 과도 기록을 유지하는 독립 NumPy 구현 |

백엔드로 NumPy를 골라도 알고리즘은 바뀌지 않습니다. reference 경로는 shared 전용 설정을
조용히 버리지 않고 거부합니다. 그 check 리포트는 격자와 재료를 검증한다고 밝히며, 모니터
추출은 실행할 때 검사합니다. shared check는 시간을 진행하지 않고 전체 격자 승인 과정을
실행합니다. 두 경로 모두 현재 배정밀도가 필요합니다. 배열을 만들기 전에 JAX x64를 설정하세요.

`spacing`은 명목 z 격자를 정하고, `transverse_spacing`은 선택적으로 x/y를 따로 정합니다.
`nominal_layer_thickness`는 층 두께가 호출마다 바뀌어도(live) 격자를 고정합니다.
`variable_depth=True`는 실험실 좌표를 고정하고 호출마다 바뀌는 경계면 위상을 쓰면서 전체 깊이
변화를 허용합니다. `nominal_indices`는 반공간 재료가 바뀌는 동안 패딩과 비균일 간격(grading)을 고정합니다.
준비된(prepared) 실행은 매번 같은 선언 배치와, 호출마다 바뀌는 입력 전체를 요구합니다. 양수인
경계면·도형 smoothing은 수치 평균 폭이며, 그 편향은 격자 오차와 별도로 수렴시켜야 합니다.
기존 shared 엔진의 도형 제약은 그대로 적용됩니다. separable 사각형 또는 면적 적분 사각형,
개별 원·타원이며, 지원하는 칠하기·연결 구조(topology) 제약이 따릅니다. 측정된 정확도 한계는 엔진 연구
가이드를 보세요. 형식이 있는 선언을 고른다고 그 범위가 넓어지지는 않습니다.

Region/포트 재료는 비분산 `Material` 선언입니다. 래스터 epsilon은 epsilon-infinity이고,
`RasterDispersion(labels=..., poles={label: (Drude(...), Lorentz(...))})`가 물리적 전기 극점을
제공합니다. 레이블 0은 비분산입니다. 지원하지 않는 인과 모델이나 영역 분산 요청은 거부합니다.

반사·투과 `DiffractionOrders`에는 항상 맨 앞에 파장 축이 있습니다. 복소수 `amplitude_ratio_s`와
`amplitude_ratio_p`는 입사 전기장에 대한 비율(ratio)이라서 단위가 없습니다
(`reference['amplitude_ratio_units']`는 `'1'`). 물리 경계면과 셀 중심에서 각 차수의 s/p 좌표계로
기준을 잡습니다. 이 비율은 프레넬 반사계수처럼 셀의 성질이라서, 광원 진폭을 키워도 바뀌지 않습니다.
`wavebit.rcwa`도 같습니다. 광원 진폭은 `reference['incident_field']`(V/m, 단위 라벨은
`incident_field_units`)에 있고, 광원 진폭에 따라 바뀌는 값에는 단위가 붙습니다.
- `field_s`와 `field_p`는 이 비율에 `reference['incident_field']`를 곱한 값, 곧 차수마다 나가는
  전기장(V/m)입니다. 광원 진폭에 비례하고, 위상도 따라갑니다. 이 차수들에는 직교 좌표 성분인
  `amplitude_ratio_xyz`가 없으므로 `field_xyz`는 None입니다.
- `power`는 차수마다 나가는 파워이고, `reference['incident_power']`는 셀 하나를 지나는 입사 파워입니다.
  교차 셀이면 W, y 불변이면 y 방향 단위 길이당 W/m입니다(`power_units`, `incident_power_units`).
  둘 다 광원 진폭의 제곱에 비례합니다.
- 아래의 평면 값과 스냅숏은 광원 진폭 자체에 비례하고, 위상도 따라갑니다.
  `algorithm='reference'`에서는 평면 값이 soft-source 스펙트럼 보정(광원 펄스의 스펙트럼으로 나누는
  보정)을 그대로 쓰므로, 나가는 차수의 전기장을 V/m로 읽으려면 그곳에서는 `field_s`와 `field_p`를
  씁니다.

효율은 입사파만의 파워로 정규화하므로 `power = efficiency * incident_power`입니다. 파장 범위의
k_par(경계면에 나란한 파수)는 중심 파장에서 정하므로, 파장마다 셀에 들어오는 각도가 조금씩
다릅니다(`metadata['theta_radians']`). 입사 파워도 파장마다 그 각도로 계산합니다. 입사파가 나아가지
못하는 파장에서는 `incident_power`, `power`, `efficiency`가 NaN이고, 그 파장의
`metadata['power_defined']`와 `metadata['efficiency_defined']`는 False입니다. reference 스테핑은
그런 파장까지 계산하고, shared 스테핑은 그런 파장 범위를 거부합니다. 각 `reference`에는 `port`('top' 또는 'bottom'), `wave`('reflected' 또는
'transmitted'), `plane`(경계면 z, `length_units` 단위), `lateral_origin` `(0., 0.)`,
`amplitude_ratio_units`('1'), `incident_field`와 `incident_field_units`('V/m'), 그리고 `cell_measure`와
`cell_measure_units`(교차 셀이면 'm^2', y 불변이면 y 방향 단위 길이당 'm'이며 이때 `invariant_axes`는
`('y',)`)가 들어 있습니다. `R`과 `T`는 이 효율을 편하게 보는 뷰입니다. 사용 가능한 s/p 기저 관측값은
`result.basis`에 따로 보관합니다. 그 진폭도 비율이며, 각 기저의 `field_s`와 `field_p`는 선언한
진폭으로 순수 s 편광 또는 순수 p 편광이 입사했을 때 나올 전기장입니다.

`Outputs(planes={'name': z})`는 보정된 phasor를 스트리밍합니다. E는 V/m, H는 A/m이며,
평면 메타데이터에 보정 방식이 기록됩니다. shared 스테핑은 경계면의 단위 입사 E로 보정하고,
reference 스테핑은 soft-source 스펙트럼 보정을 유지합니다. `plane_power_z`에는 물리 적분
측도가 포함됩니다. 교차 셀이면 W, y 불변 문제이면 **W/m**입니다. 단위는 각 평면의
`metadata['power_units']`에 `cell_measure`, `cell_measure_units`와 함께 명시됩니다. 양의 파워는 위쪽 방향입니다. 샘플을 별도 기준 실행으로 나누는 비교가 적절한 경우가 많습니다.
[주파수 평면 예제](../../../examples/fdtd/fdtd_frequency_planes.py)를 보세요.

`Outputs(snapshots=Snapshots(every=..., components=..., bounds=...))`는 성분 `'Ex'`부터
`'Hz'`까지의 원래 Yee 스냅숏을 각 성분의 좌표와 E/H 반단계 타임스탬프(초)와 함께 보관합니다.
`result.snapshots`는 성분을 키로 씁니다. `bounds`는 `'x'`, `'y'`, `'z'`를 길이 단위의 물리
`(low, high)` 쌍에 대응시키며 shared 스테핑이 필요합니다. shared 스냅숏은 인과적 파형을 유지하기
위해 일정한 반송파 보정을 쓰며, 주파수별 평면 보정과 다릅니다.
`result.sample(x=..., y=..., z=..., times_seconds=...)`는 여섯 성분 전부를 그 SI 시각에서
오프라인 보간하므로 기본 `components`가 필요합니다. 광원 불연속을 가로지르는 보간은 거부하며
AD는 유지하지 않습니다. 순간 Poynting 벡터는 phasor 인자 1/2 없이 `cross(E.real, H.real)`입니다.
[스냅숏](../../../examples/fdtd/fdtd_record.py) 예제와
[Poynting](../../../examples/fdtd/fdtd_poynting.py) 예제를 보세요.

`Numerics(decay_check=DecayCheck(window_seconds=..., threshold=...))`는 조기 종료 없이 마지막
창에서 감쇠와 산란 변화를 진단합니다. 창은 초 단위이며 `native.DecayCheck`와 같은 규칙으로 완전한
시간 단계 수로 올림합니다. `check(...).resolved['time_step_seconds']`에서 얻은 `time_step_seconds`로
`40*time_step_seconds`를 주면 40단계를 덮습니다. 보고(`validity.convergence['decay']`와
`diagnostics['decay']`, `DecayCheck`가 없으면 `None`)에는 기저별 변화와 산란 한계가 SI 시간, 주어진
`threshold`, `_units`로 끝나는 키의 명시적 단위 라벨과 함께 남습니다. 이 검사로 격자, smoothing,
PML 수렴이 확인되지는 않습니다.

`validity.checks`에는 `runtime_stability`와 `numerical_finite`가 있습니다. `numerical_finite`는
엔진이 계산한 R과 T를 읽습니다. 공개 결과에서 값을 NaN으로 가리기 전의 값이며, 이 값이 유한한지를
알려 줍니다. 입사파가 나아가지 못하는 파장에서는 공개되는 `R`과 `T`가 일부러 NaN이고, 차수의
`metadata['efficiency_defined']`가 그 파장에서 False입니다. 그런 파장을 찾을 때는
`numerical_finite`가 아니라 이 표시를 보세요. reference 스테핑은 안정성을 판단하지 않으므로
`algorithm='reference'`에서는 `runtime_stability`가 참이 아니라 `None`(판단 안 함)입니다. `diagnostics['invariant_axes']`는 y 불변 문제에서 `('y',)`,
그 밖에는 `()`입니다. `result.provenance`에는 `solver`, `execution`, `numerics`, `units`,
`coordinate_frame`, `time_convention`, `time_step_seconds`가 기록됩니다. `observations.fit_modes`는 오프라인 ringdown
피팅에 SI 타임스탬프를 받고 Hz와 1/s 단위 감쇠율을 반환합니다. 창과 모델 차수 수렴은 별도
검사로 남습니다.

## CPU에서 외부 `jax.jit` 아래의 메모리

Wavebit에는 이 설정을 위한 옵션이 없습니다. JAX 컴파일러 설정이고, 영향을 받는 프로그램은
사용자 자신의 외부 `jax.jit`이므로 그곳에서 고릅니다.

한 층에 `shape_integration='area'` 사각형이 두 개 이상 있을 때 해당합니다. 사각형을 하나라도
회전하면 `'separable'`은 쓸 수 없으므로 `shape_integration='area'`가 필요합니다. 이 경우 순방향 모드
미분(`jax.jvp`, `jax.jacfwd`)을 CPU에서 컴파일하면, 같은 미분을 즉시 실행할 때보다 몇 배의 메모리가
필요할 수 있습니다. XLA:CPU 기본 스케줄러가 마지막 도형 칠하기 중간값이 만들어질 때까지 어느 것도
해제하지 않아 모두가 같은 순간에 살아 있기 때문입니다. 메모리 최적화 스케줄러는 같은 값을 훨씬 작은 작업 메모리로 계산합니다.

```python
options = {'xla_cpu_scheduler_type': 'CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED'}
jacobian = jax.jit(jax.jacfwd(loss), compiler_options=options)
```

프로세스의 모든 jit에 적용하려면 JAX가 백엔드를 초기화하기 전에 `XLA_FLAGS`에
`--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED`를 추가하세요. 이미 있는 플래그는
그대로 둡니다.

회전 사각형이 하나뿐인 층과 `'separable'` 경로는 영향을 받지 않습니다. 역방향 모드(`jax.grad`)는
측정한 한 경우에서 영향이 훨씬 작았습니다. 측정 결과는
[면적 적분 도형 FDTD 미분의 외부 jit 메모리](../../validation/fdtd/fdtd-jit-painting-memory.md)에 있습니다.
