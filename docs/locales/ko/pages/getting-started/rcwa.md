# 첫 RCWA 계산

유전체 원기둥이 주기적으로 배열된 층에서 세 파장의 반사율과 투과율을 계산합니다. 기본 NumPy 백엔드로 실행하며, 자동미분용 패키지는 필요하지 않습니다.

## 1. 구조를 정의하고 실행하기

길이 단위는 마이크로미터입니다. 원기둥 층 위에는 공기, 아래에는 유전체 기판을 둡니다. `Problem`에는 물리 구조와 광원을, `Numerics`에는 푸리에 근사 설정을 지정합니다. 아래의 낮은 차수와 64포인트 NVF 격자는 빠른 실행을 위한 값이며, **수렴을 확인한 설계 설정이 아닙니다**.

```python
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length='um'),
    stack=rcwa.Stack(
        period=(.6, .6),
        top=Material(epsilon=1.),
        bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(
            thickness=.18,
            background=Material(epsilon=1.),
            regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=.12),
                material=Material(epsilon=12.25)),)),)),
    excitation=rcwa.Excitation(
        wavelength=np.array([.85, .9, .95]),
        sources=(rcwa.PlaneWave(side='top', polarization='s', amplitude=1.),)),
    outputs=rcwa.Outputs(fields=True))

# These small orders make this a quick demonstration, not a convergence claim.
numerics = rcwa.Numerics(orders=(1, 1), factorization='nvf', shape_nvf_grid_size=64)
report = rcwa.check(problem, numerics=numerics)
report.raise_if_invalid()
prepared = rcwa.prepare(problem, numerics=numerics)
result = prepared.run(problem)
```

## 2. 반사율과 투과율 읽기

회절 차수별 값을 더하면 각 파장에서 입사 전력 대비 전체 반사율과 투과율을 얻습니다.

```python
print("R:", result.R.sum(axis=(1, 2)))
print("T:", result.T.sum(axis=(1, 2)))
print("R + T:", (result.R + result.T).sum(axis=(1, 2)))
```

각 배열의 세 값은 순서대로 0.85, 0.90, 0.95 µm에 해당합니다. 손실이 없는 이 구조에서는 R + T가 1에 가까워야 합니다. 에너지 보존만으로 수렴을 보장할 수는 없으므로, 푸리에 차수를 높여 결과가 얼마나 달라지는지도 확인하세요.

특정 회절 차수나 파장만 선택할 수도 있습니다.

```python
print(result.R.shape)  # (3, 3, 3): wavelength, order_v, order_u
print(result.transmission.order(0, 0).efficiency)
print(result.transmission.at_wavelength_index(1).wavelengths)
```

`R`과 `T`의 축은 `(파장, v 방향 차수, u 방향 차수)`입니다. 파장이 하나여도 파장 축은 유지됩니다. (0, 0)은 회절되지 않은 0차 성분입니다.

## 3. 선택: 구조 내부의 필드 읽기

앞에서 `outputs=rcwa.Outputs(fields=True)`로 필드를 저장하도록 설정했습니다. 다음 좌표에서 필드를 샘플링합니다.

```python
samples = result.field(x=np.array([0., .1]), y=np.array([0.]), z=np.array([.09]))
print(samples.values.shape)  # (3, 1, 1, 2, 6)
print(samples.units)         # E: V/m, H: A/m, coordinates: um
```

결과 축은 `(파장, z, y, x, 성분)`이며, 성분 순서는 `(Ex, Ey, Ez, Hx, Hy, Hz)`입니다. 전기장은 V/m, 자기장은 A/m, 좌표는 문제에서 정한 길이 단위를 사용합니다.

## 다음으로 배우기

층은 **아래에서 위로** 나열합니다. x/y의 원점은 셀 중심이며, z는 유한한 층 구조의 바닥에서 위쪽으로 증가합니다. [좌표 규약](../learn/fundamentals/conventions.md), [재료 정의](../guide/common/materials.md), [RCWA 기초 과정](../learn/fundamentals/index.md)으로 이어가세요.

### 준비한 계산 재사용하기

호환되는 변경은 수정한 문제 전체를 준비된 계산에 전달하면 됩니다. 아래에서는 파장 개수를 유지하면서 값만 바꿉니다. 이전 결과는 원래 문제에 대응하며, 구조가 바뀌면 다시 준비해야 합니다. 자세한 조건은 [계산 흐름과 재사용](../guide/common/lifecycle.md)을 참고하세요.

```python
new_problem = problem.replace(excitation=problem.excitation.replace(
    wavelength=np.array([.86, .91, .96])))
new_result = prepared.run(new_problem)
# result and its retained fields still belong to the first calculation.
print(new_result.T.sum(axis=(1, 2)))
```

### 작업 정의 저장하기

직렬화가 필요할 때는 `Job`에 문제·수치 설정·실행 백엔드를 함께 담을 수 있습니다.

```python
from wavebit import Job, Execution

job = Job(solver='rcwa', problem=problem, numerics=numerics,
          execution=Execution(backend='numpy'))
assert Job.from_json(job.to_json()).key == job.key
```

다음에는 [박막 최적화 예제](https://github.com/fieldalgo/wavebit/blob/main/examples/rcwa/film_optimization.py)나 [RCWA/FDTD 비교 예제](https://github.com/fieldalgo/wavebit/blob/main/examples/cross_solver/film_and_grating.py)를 살펴보세요. 각 예제의 검증 범위는 명시된 조건에 한정됩니다.
