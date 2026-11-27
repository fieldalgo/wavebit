# 출사 필드 전파

`wavebit.postprocess.propagation`은 솔버가 내보낸 출사 복소 전기 필드를 균일한 외부 매질을
지나 계속 전파합니다. 구조 솔버를 다시 실행하지는 않습니다.

```python
from wavebit import Units
from wavebit.postprocess import propagation

view = propagation.propagate(
    result.transmission,
    target=propagation.Plane(x=[0., .1], y=[0., .1], distance=10.,
                             units=Units(length="um")),
    method="orders", chunk_size=256,
)
E = view.field.values  # (wavelength, y, x, xyz), V/m
signal = abs(E)**2     # component-wise V^2/m^2, not power flux
```

입력은 기존의 `observations.DiffractionOrders`이며, Cartesian CW 진폭, 부호가 있는 법선 및
횡방향 파수 벡터, 기준면, 횡방향 원점, 외향 법선, 단위와 continuum/시간 규약 메타데이터를
가집니다. RCWA 반사 및 투과 결과가 이 계약을 충족합니다. 효율만 있는 결과와 원시 펄스
스펙트럼은 빠진 복소 위상이나 CW 보정을 제공할 수 없습니다.

목표면의 x/y 축은 비어 있지 않은 임의의 유한 실수 벡터일 수 있으며, FFT 빈이거나 등간격일
필요가 없습니다. 거리는 0 이상이고 포트에서 바깥쪽으로 잽니다. 따라서 아래쪽 포트에서 전파하면
공개 z가 감소합니다. 반환되는 `FieldSamples`의 좌표와 파장은 목표면의 길이 단위를 씁니다.
소스 기준과 메타데이터(전력을 구할 수 있는지 여부 포함)는 diagnostics에 보존됩니다. 유효하지
않은 소스 진폭은 NaN으로 전파되며, 결코 0으로 바뀌지 않습니다.

합성은 각 벡터 진폭에
`exp(i*(kx*(x-x0) + ky*(y-y0) + kz*delta_z))`를 곱해 더합니다. 법선 파수 벡터는 이미 공개
Cartesian 부호를 지니며, `delta_z = outward_normal_z * distance`입니다. 소멸 차수는 그 방향으로
감쇠합니다. 이 경로는 무한 주기 해에 유한 개구도, 가장자리 산란도 더하지 않습니다.

## 지원 범위와 거부

이 첫 경로는 z 법선 포트, 실수 횡방향 파수 벡터, 그리고 균일하고 등방이며 비자성인 양의
무손실 외부 매질을 지원합니다. `method="orders"`는 연속체 회절차수를 받습니다. Python 결과
클래스가 일치한다는 사실만으로는 호환성이 성립하지 않으며, 이 경로에서 불완전한 메타데이터와
Yee 분산은 거부됩니다. 보정된 평면장은 아래 ASM 경로에서 연속체 근사를 명시적으로 받아들여 사용합니다.

직접 회절차수 경로는 NumPy, JAX, PyTorch CPU 즉시 실행 계산을 지원합니다. 진폭, 거리와 상류 RCWA 설계 그래프는
미분 가능한 상태로 유지됩니다. 추상 JAX JIT 검증은 명시적으로 거부하며, GPU와 배치 변환은
검증하지 않았습니다. `chunk_size`는 순방향 관측점 블록 하나하나의 크기를 제한하는 양의
정수이며, 전체 출력이나 자동 미분 저장 공간을 제한하지는 않습니다.

실패는 `propagation.input_contract`, `.normalization`, `.convention`, `.direction`,
`.sampling` 또는 `.method`와 함께 `ValidationError`/`UnsupportedFeatureError`를 사용합니다.
전력 플럭스는 자기 필드가 필요하므로 계산하지 않습니다.

`tests/postprocess/test_propagation.py`는 결맞는 간섭, 출사 부호, 소멸파 감쇠, 단위, 기준 원점,
NaN, RCWA 경계 필드, 백엔드 일치, 그리고 진폭/거리/상류 설계 미분을 독립적인 해석식이나
중앙 차분과 비교해 검사합니다.

## 샘플링된 평면의 각 스펙트럼 전파(ASM)

`method="asm"`은 균일한 Cartesian 평면에서 샘플링된 복소 전기장 성분 하나를 받습니다.
균일 매질의 스칼라 Helmholtz 방정식에 따른 단방향 전파이며, 구조 산란을 풀거나 편광과 자기장을
복원하지 않습니다. 외부 굴절률은 양의 실수여야 합니다. 좌표, 진공 파장, 거리는 명시한 길이
단위를 공유하며 시간 규약은 `exp(-i*omega*t)`입니다. 실행 예제는 이 페이지의 영어 원문을 따릅니다.

`PhasorPlane`에 `values`(y,x 복소 배열), `x`, `y`, `z`, `wavelength`,
`units`, `medium_n`, `component`, `direction`, `boundary`를 지정합니다.
`propagate(source, target=..., method="asm", numerics=ASM())`의 반환 필드는
(1,y,x,1) 모양이며 V/m입니다. 목표면은 같은 x/y 격자를 유지해야 하며 보간과 기울어진 면은
거부됩니다. 거리는 0 이상입니다. `direction="-z"`이면 공개 z가 감소하지만 소멸파는 감쇠합니다.

FFT 전달함수는
`exp(i*sqrt((2*pi*n/wavelength)**2-kx**2-ky**2)*distance)`이며 공통 출사 제곱근
규약을 사용합니다. 차단점에서도 순방향 값은 정의되지만 파장/굴절률 미분은 특이해집니다.
그 미분을 사용할 때 `diagnostics["parameter_gradient_defined"]`를 확인해야 합니다.

횡방향 경계는 직접 선택합니다.

- `boundary="periodic"`은 영 채우기를 하지 않습니다. 좌표 길이 단위당 라디안으로 주는
  `carrier_kxy=(kx,ky)`는 Bloch 반송파입니다. FFT는 이를 제거한 포락선에 적용되며
  유한 소자의 가장자리를 추가하지 않습니다.
- `boundary="isolated"`는 입력 창 밖의 장을 0으로 봅니다. 기본 `ASM(padding=None)`은
  각 비단일 축 양쪽에 ceil(N/2) 셀을 붙입니다. `padding=(pad_y,pad_x)`로 수렴성을 확인합니다.
- 표본이 하나인 축은 불변 방향이며 영 채우기를 할 수 없습니다. 기본 패딩이나 작은 격자 간격만으로
  수렴을 보장하지 않습니다. 입력 창과 패딩을 늘리고 간격을 줄여 확인해야 합니다. 유한 FFT 창에서는
  회절된 빛이 반대쪽에서 다시 들어오는 래핑이 생길 수 있습니다.

## FDTD 평면장 연결

`fdtd.periodic.Result.planes[name]`은 소스 정규화를 마친 collocated phasor입니다.
`from_phasor(..., component="Ey", wavelength_index=0, medium_n=1.,
direction="-z", boundary="periodic", assume_outgoing=True)`로 성분과 파장을 선택한 뒤
`plan_asm(source)(distance)`를 호출합니다.

사용자가 해당 평면이 균일한 외부 매질에 있고 선택한 출사파만 포함하는지 확인해야 합니다.
전기장만으로 입사파와 반사파를 분리할 수 없습니다. `assume_outgoing=True`는 그 확인을 기록할 뿐
분해를 수행하지 않습니다. 소스와 계면을 벗어난 투과측 외부 평면을 사용하십시오. 원시 펄스 Fourier
스펙트럼, 시계열, 보정 출처가 없는 장, 물리 전기장 단위가 아닌 입력은 거부합니다.
새 펄스 정규화를 추정하지 않습니다.

이 어댑터는 평면 이후에 연속체 분산을 적용합니다. FDTD 보정 출처를 보존하지만 Yee 수치분산을
보정하거나 이산 시간영역 해를 정확히 연장하지는 않습니다. 필요한 정확도에 맞게 원래 FDTD 격자를
세분화해야 합니다. diagnostics의 `continuum_approximation=True`로 이를 표시합니다.
`abs(E)**2`는 V²/m²이며 전력 플럭스가 아닙니다.

## 반복 실행, JIT, 장치

`prepared=plan_asm(source)`를 즉시 실행으로 만든 뒤
`prepared(distance, values=..., wavelength=..., medium_n=...)`를 호출합니다.
values는 준비한 모양, 백엔드, dtype, 장치와 일치해야 합니다. 격자, 반송파, 패딩을 바꾸면
계획을 새로 만들어야 하며 이들은 해당 계획의 동적 설계 변수가 아닙니다.

NumPy, JAX, PyTorch에서 복소 연산과 즉시 실행 미분을 유지합니다. JAX 배열로 만든 계획은
`jax.jit` 안에서 실행할 수 있습니다. 배정밀도가 필요하면 배열 생성 전에 JAX의 64비트 모드를
켜십시오. 즉시 실행에서 잘못된 스칼라 입력은 예외를 냅니다. JIT의 동적 입력이 잘못되면
NaN 필드와 `diagnostics["input_valid"]=False`를 반환합니다. 비유한 필드도 NaN과 false를
반환합니다. 컴파일된 작업에서는 이 플래그를 반드시 확인하십시오.

ASM 커널은 소스 장치를 유지하며 조건부 CUDA 순방향/미분 검사를 포함합니다. CPU CI 통과는
GPU 검증이나 성능 향상을 의미하지 않습니다. Torch 컴파일은 공통 정책에 따라 이 진입점을 즉시
실행합니다. 직접 회절차수 경로의 기존 CPU/JIT 제한은 유지됩니다.

`tests/postprocess/test_asm.py`는 해석적 평면파와 소멸파, 거리 0, 단위 변환, Gaussian
근축 강도와 격자/패딩 수렴, 실제 보정된 FDTD 평면, 백엔드 일치, JIT 입력 검증 및 중앙 차분
대비 자동 미분을 확인합니다. complex128의 평면파/감쇠/백엔드 비교 허용치는 1e-10,
Gaussian 근축 강도는 1e-3, 정규화 미분 오차는 1e-6입니다.
