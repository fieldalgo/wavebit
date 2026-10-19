# 첫 FDTD 계산

유전체 박막에 펄스를 입사시키고, 박막 아래 한 지점에서 전기장 Ey의 시간 변화를 기록합니다. 기본 NumPy 백엔드로 실행할 수 있습니다.

## 1. 영역을 정의하고 실행하기

`Problem`에는 계산 영역·구조·광원·관측을, `Numerics`에는 격자·계산 시간·흡수층을 지정합니다. 길이 단위는 마이크로미터입니다. x/y 방향은 주기 경계, z 방향은 열린 경계이며, PML 흡수층은 물리 영역 **바깥에** 추가됩니다.

이 예제는 빠르게 실행해 보는 용도입니다. 보정된 스펙트럼이나 수렴 검증 결과를 제공하는 예제는 아닙니다.

```python
from wavebit import Material, Units, fdtd

problem = fdtd.Problem(
    units=Units(length="um"),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(fdtd.Structure("film", shape=fdtd.SnappedSlab(z=(-.1, .1)),
                                   material=Material(epsilon=2.25)),),
    ),
    sources=(fdtd.PlaneWave("illumination", plane=fdtd.Plane(z=.5),
        pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.TimeSeriesMonitor("probe", at=(0., 0., -.3), component="Ey"),)),
)
numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.025),
    duration=fdtd.Duration(periods=20, source="illumination"), absorber=fdtd.PML(cells=12))
report = fdtd.check(problem, numerics=numerics)
assert report.ok
result = fdtd.run(problem, numerics=numerics)
```

## 2. 기록한 신호 읽기

```python
print("samples:", len(result["probe"].time), "time unit: s; field unit:", result["probe"].units)
```

`result["probe"].time`은 초 단위의 시간 배열입니다. 전기장 Ey의 단위는 V/m이며, 자기장을 기록하면 A/m를 사용합니다. `TimeSeriesMonitor`는 선택한 성분의 가장 가까운 Yee 격자점에서 샘플링합니다. 임의 좌표의 보간값은 아닙니다.

## 선택: 준비한 계산 재사용하기

반복 실행에는 `prepared = fdtd.prepare(problem, numerics=numerics)`와 `prepared.run(problem)`을 사용할 수 있습니다. 매번 현재 문제 전체를 전달합니다. 격자 구조·계산 시간 구조·모니터 배치 등 준비 조건을 바꾸면 다시 준비해야 하며, 호환되지 않는 재사용은 오류로 알려줍니다.

## 다음으로 배우기

[FDTD 가이드](../guide/fdtd/general-fdtd.md)에서 단위와 관측 기능을, [문제 모델링](../guide/fdtd/model.md)에서 격자와 구조 표현을 확인하세요. [FDTD 학습 과정](../learn/fdtd/index.md)에서는 격자·시간을 바꾸며 수렴과 물리적 타당성을 확인합니다.

[분산 재료 필드 예제](https://github.com/fieldalgo/wavebit/blob/main/examples/fdtd/dispersive_fields.py)는 Lorentz 재료, 시간 신호, 스펙트럼 E/H와 별도의 기준 계산을 다룹니다. [박막·격자 비교 예제](https://github.com/fieldalgo/wavebit/blob/main/examples/cross_solver/film_and_grating.py)는 RCWA와 FDTD의 기준면을 맞추고 공간·시간 해상도에 따른 회절 차수의 복소 응답을 비교합니다. 특정 예제의 일치가 다른 구조나 광원에서도 수렴을 보장하지는 않습니다.
