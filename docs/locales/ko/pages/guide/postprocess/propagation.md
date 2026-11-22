# 출사 회절차수 전파

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
무손실 외부 매질을 지원합니다. `method="orders"`만 구현되어 있습니다. 샘플링된 평면의 ASM과
FDTD 생산자 정규화/분산 어댑터는 별도의 향후 작업입니다. Python 결과 클래스가 일치한다는
사실만으로는 호환성이 성립하지 않습니다. 불완전한 메타데이터와 Yee 분산은 거부됩니다.

NumPy, JAX, PyTorch CPU 즉시 실행 계산을 지원합니다. 진폭, 거리와 상류 RCWA 설계 그래프는
미분 가능한 상태로 유지됩니다. 추상 JAX JIT 검증은 명시적으로 거부하며, GPU와 배치 변환은
검증하지 않았습니다. `chunk_size`는 순방향 관측점 블록 하나하나의 크기를 제한하는 양의
정수이며, 전체 출력이나 자동 미분 저장 공간을 제한하지는 않습니다.

실패는 `propagation.input_contract`, `.normalization`, `.convention`, `.direction`,
`.sampling` 또는 `.method`와 함께 `ValidationError`/`UnsupportedFeatureError`를 사용합니다.
전력 플럭스는 자기 필드가 필요하므로 계산하지 않습니다.

`tests/postprocess/test_propagation.py`는 결맞는 간섭, 출사 부호, 소멸파 감쇠, 단위, 기준 원점,
NaN, RCWA 경계 필드, 백엔드 일치, 그리고 진폭/거리/상류 설계 미분을 독립적인 해석식이나
중앙 차분과 비교해 검사합니다.
