# 첫 FDTD 계산

FDTD 문제는 물리 영역, 그 안의 구조, 소스와 요청한 관측을 기술합니다. 격자, 계산 시간, CFL 계수와 흡수층은 `fdtd.Numerics`에 속합니다. 두 솔버는 공유 `Material` 정의와 명시적인 `Units`를 사용합니다.

이 작은 펄스 계산은 유전체 박막 아래에서 Ey를 기록합니다. 작은 횡방향 영역은 x와 y로 주기적이며, z 면은 열려 있습니다. PML은 물리 영역 **바깥에** 추가됩니다. 이 예제는 빠르게 실행해 보는 용도이며, 보정된 스펙트럼이나 세분화 연구가 아닙니다.

## 1. 영역을 정의하고 실행하기

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
print("samples:", len(result["probe"].times_seconds), "time unit: s; units:", result["probe"].units)
```

모니터의 시간 축은 초 단위이고, Ey는 V/m 단위입니다. H 성분은 A/m 단위입니다. `units`는 기록한 성분을 그 단위에, `'coordinates'`를 문제의 길이 단위에 대응시킵니다. `TimeSeriesMonitor`는 선택한 성분의 가장 가까운 Yee 점에서 샘플링합니다. 임의 위치에서의 공간 보간이 아닙니다.

## 선택: 준비한 계산 재사용하기

호환되는 실행을 반복할 때는 `prepared = fdtd.prepare(problem, numerics=numerics)`와 `prepared.run(problem)`을 사용하세요. 각 실행은 현재 문제 전체를 받습니다. 격자 위상 구조, 계산 시간 구조, 모니터 배치나 그 밖의 준비된 제약을 바꾸면 새로 준비해야 하며, 거부된 재사용은 구조화된 비호환 오류를 냅니다.

단위, 지원하는 관측과 검증 한계는 [FDTD 가이드](../guide/fdtd/general-fdtd.md)를, 격자, 셀 스냅 및 스무딩 구조와 실행이 통과해야 하는 정확도 규칙은 [솔버가 문제를 모델링하는 방식](../guide/fdtd/model.md)을, 세분화와 물리적 검증은 [FDTD 학습 과정](../learn/fdtd/index.md)을 보세요.

## 분산과 보정된 관측으로 이어가기

[분산 필드 예제](https://github.com/fieldalgo/wavebit/blob/main/examples/fdtd/dispersive_fields.py)는
물리적인 Lorentz 재료, 시간 신호 모니터, 스펙트럼 E/H와 별도의 기준 실행을 추가합니다.
원시 푸리에 필드 단위를 소스로 정규화한 필드와 구별해 유지하며, 실행 중 소스 진폭 재사용을
검사합니다. 그림 그리기는 선택 사항입니다. 이어서 [박막과 격자 비교](https://github.com/fieldalgo/wavebit/blob/main/examples/cross_solver/film_and_grating.py)는
RCWA와 FDTD의 기준면을 맞추고, 공간 및 시간 창 세분화에 따라 0이 아닌 복소 회절 차수를
비교합니다. 이렇게 선언된 경우에서의 일치가 다른 재료, 기하나 펄스에 대한 수렴을 보장하지는
않습니다.
