# 물리 단위의 native Yee 격자

물리 모델을 이미 Yee 위치에 표본화했다면 `from wavebit.fdtd import native`를
사용합니다. `native.Problem`은 `Medium`, `Current`, `InitialField`, Bloch 위상,
관측을 `native.Numerics`와 분리합니다. `check`, `prepare`, `run`과 이식 가능한
`Job(solver='fdtd.native', ...)`을 지원합니다. 2026-10-19에 바뀐 이름은
[공개 이름 이행 안내](../../migration/public-names.md#native-grid-and-periodic-stack-names)에
정리되어 있습니다.

공간 배열 순서는 `(z,y,x)`이고 벡터에는 맨 앞에 Cartesian `(x,y,z)` 성분
축이 있습니다. `Grid.spacing`은 `Units.length` 단위의 `(dz,dy,dx)`입니다.
선택적인 `z_spacing`에는 주기적으로 닫히는 셀을 포함한 모든 양수 z 셀 폭을
넣습니다. native 좌표는 영에서 시작합니다. 격자 기하는 AD에서도 고정이며
각 E/H 성분은 고유한 Yee 위치를 가집니다.

| 입력 | 의미 |
| --- | --- |
| `Medium.epsilon`, `mu` | 양수인 상대 epsilon-infinity와 투자율; 스칼라 또는 대각 native 계수 |
| `Medium.conductivity` | 전기 전도도, S/m |
| `Oscillator(omega_0, gamma, strength)` | SI 각 공진·감쇠율과 1/s² 단위의 세기; 배열은 native E에 브로드캐스트 가능 |
| `Current.density`, `samples` | A/m² 단위 전류 밀도에 무차원 표본을 곱함; 표본 n은 `(n+1)*dt`에 끝나는 E 갱신을 구동 |
| `InitialField.E`, `InitialField.H` | t=0의 E(V/m), t=-dt/2의 H(A/m) |
| `bloch_phase` | `(z,y,x)` 순서의 실수 라디안: F(r+L)=exp(i phase) F(r) |
| `time_step_seconds`, `steps` | 명시적인 SI 시간 간격과 고정 단계 수 |

`native.PML`은 이미 입력 배열에 포함된 양쪽 z 끝의 셀을 차지하며 배열을
확장하지 않습니다. 양쪽 `(epsilon,mu)` 기준 매질은 각 끝과 내부 한 셀까지
일치해야 합니다. 그곳에 활성 극이나 전도도가 없어야 하며, PML 사용 시
z Bloch 위상은 영입니다. `fdtd.ConductiveAbsorber`는 유한한 감쇠 바닥을 갖는 기존의
정합된 점진적 전도도 층을 선택합니다. 어느 흡수층도 반사 오차를 보장하지
않으므로 폭, 세기, 해상도를 검사해야 합니다.

`Outputs(snapshots=Snapshots(every=..., components=..., slices=...))`는 선택한
성분(`'Ex'`부터 `'Hz'`까지)의 장 스냅샷을 `every` 단계마다 보존합니다. Wavebit의
모든 출력이 같은 철자를 쓰며, 소문자 `'ex'`는 거부합니다. `slices`로는 정수
인덱스로 격자의 일부를 고를 수 있습니다. `(z,y,x)` 각 축마다 인덱스 하나 또는
`(start, stop[, step])` 튜플을 줍니다. 길이 1인 축도 그대로 두며, 계산하는 영역을
잘라 내지는 않습니다. `result.snapshots`는 각 성분 이름을 물리 좌표와 SI 시각을 가진
`TimeSeries`에 대응시킵니다. H는 E보다 반 시간 단계 뒤에 있습니다. `result.E`와
`result.H`는 최종 native 배열입니다. `result.polarization_density`는 극점마다의 재료
분극 밀도 P(C/m²)이고, `previous_polarization_density`는 한 단계 전의 같은 값, 즉
이전 ADE 상태입니다.

`Spectrum(wavelengths=..., z_indices=...)`은 정수 z 노드 인덱스를 사용해
공간적으로 같은 위치의 푸리에 장을 누적합니다. `result.spectrum`의 축은
`(wavelength,plane,y,x,component)`, E 단위는 V s/m, H 단위는 A s/m입니다.
변환은 exp(+i 2 pi f t)를 쓰는 원시 적분이며 입사장 보정을 하지 않습니다.
따라서 `result.spectral_power_z`는 물리적 횡방향 셀 면적을 포함한 W s²
단위입니다. 단색파 파워나 펄스 에너지 측정량이 아닙니다. `dft_subsample`은
관측 계산만 줄이며 안티앨리어싱 필터는 없습니다.

`Numerics(decay_check=DecayCheck(window_seconds=..., threshold=...))`는 마지막
구간을 진단하며 조기 종료하지 않습니다. 창은 `periodic.DecayCheck`와 같은 규칙으로
정수 개의 시간 단계로 올림하며(N 시간 단계로 적은 창은 N 단계를 덮습니다), 실행
길이보다 짧아야 합니다. 장 감쇠, ADE 활동, 푸리에 꼬리 비율은 별개입니다.
ADE 진단은 저장 에너지가 아닌 상대 노름이고 CPML 기억은 평가하지 않습니다.
보고서는 `validity.convergence['decay']`와 `diagnostics['decay']`이며(`DecayCheck`가
없으면 `None`), 그 `threshold`는 지정한 값이고 단위 표기는 `_units`로 끝나는 키입니다.
`validity.checks`는 이 수렴 진단과 별개로 수치 판정 `runtime_stability`와
`numerical_finite`(E와 H가 유한한지)를 담습니다. `result.provenance`는 `solver`,
`execution`, `numerics`, `units`, `coordinate_frame`, `time_convention`,
`time_step_seconds`를 기록합니다.

`check`는 실행의 입력 허용 검사와 예상 배열 할당을 수행하지만 장을 진행하지
않습니다. prepared 실행은 수치 설정·배치·관측을 고정하면서 매 호출의
계수·광원·초기값·Bloch 위상을 새로 받습니다. 구체적인 잘못된 계수는 예외를
발생시키고 추적 중인 잘못된 계수는 유효하지 않은 수치 결과를 만듭니다.
NumPy, JAX, Torch는 같은 엔진을 사용합니다. 현재 실행에는 배정밀도가
필요합니다. JAX 배열을 만들기 전에 x64를 설정하세요. JAX jit와 Torch eager
AD는 지원하며 Torch 컴파일은 검증된 실행 모드가 아닙니다.

Torch CUDA는 `run` 또는 `prepare`에 `backend='torch', device='cuda:0'`를
전달합니다. 입력은 실행 장치로 변환되며 기울기는 원래의 리프 텐서에 연결됩니다.
[CUDA 검증 기록](../../validation/fdtd/native-cuda.md)에 독립 환경 설치,
검증한 수치·미분 범위, 동기화한 비용 측정이 있습니다. 일반 장면 CUDA는
[별도의 제한된 범위](../../validation/fdtd/general-cuda.md)가 있으며,
어느 기록도 모든 주기 구조 설정을 검증하지는 않습니다.

실행 가능한 [재료 미분](../../../examples/fdtd/fdtd_grid_gradient.py),
[Bloch 미분](../../../examples/fdtd/fdtd_bloch_gradient.py),
[Lorentz 미분](../../../examples/fdtd/fdtd_dispersive_gradient.py),
[세 백엔드 비교](../../../examples/fdtd/fdtd_backends.py)를 참고하세요.
예제의 명시적 목적함수 스케일링은 SI 푸리에 파워 목적함수를 무차원으로
만들며 솔버의 물리 출력은 바꾸지 않습니다.
