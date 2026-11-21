# FDTD 도메인, 소스와 관측


공개 시간 영역 모듈로는 `from wavebit import fdtd`를 사용하세요. `fdtd.Problem`은 물리적 `Scene`, 이름 붙은 소스, 요청한 `Outputs`와 명시적인 `Units`를 하나로 묶습니다. 격자 간격, 지속 시간, CFL과 PML은 `fdtd.Numerics`에 속합니다.

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
print("samples:", len(result["probe"].times_seconds), "time unit: s; units:", result["probe"].units)
```

## 물리 좌표와 단위

도메인 좌표와 소스 파장은 문제에서 선언한 길이 단위를 사용하며, 각도는 라디안입니다. 도메인의 크기와 중심은 물리 영역을 나타냅니다. PML은 그 바깥에 붙는 수치적 확장이므로, PML 셀을 추가해도 그 영역이 줄어들거나 다시 정의되지 않습니다. `Open`은 열린 물리 경계를 뜻하고, `PML`은 수치적 흡수층을 따로 제어합니다.

시간과 펄스 지연(`delay_seconds`)의 단위는 초입니다. `GaussianPulse.frequency_sigma_hz`의 단위는 Hz입니다. 평면파 진폭은 V/m 단위의 전기장입니다. `fdtd.CurrentDipole(current_moment=...)`은 **전류 모멘트**를 지정합니다. 전기 전류이면 A·m, 그에 대응하는 자기 등가량이면 V·m입니다. 이는 전기 쌍극자 모멘트를 C·m로 쓰는 RCWA `ElectricDipole`과 다르므로, 두 쪽 사이에 같은 수치 진폭을 그대로 옮기지 마세요.

공유 재료의 비율(rate)은 SI 양입니다. FDTD 어댑터는 이를 내부 정규화 단위계로 변환하고, 공개 관측값을 다시 물리 단위로 변환합니다. 지원하지 않는 텐서, 손실, 소스, 경계 또는 관측 조합은 구조화된 오류를 냅니다.

## 요청하는 관측

이름 붙은 모니터가 무엇을 누적할지 결정합니다. `TimeSeriesMonitor`는 초와 필드 단위를 갖는 `TimeSeries`를 반환하며, E는 V/m, H는 A/m입니다. 모든 필드 관측의 `units`는 들어 있는 각 성분(`'Ex'`..`'Hz'`)을 그 단위에, `'coordinates'`를 문제의 길이 단위에 대응시킵니다. 전기 표본과 자기 표본은 Yee 시간 엇갈림을 그대로 유지합니다. 스펙트럼 `FieldMonitor`, `FluxMonitor`, `DiffractionMonitor`와 `FarFieldMonitor`는 정규화와 유효성 요건이 서로 다릅니다. [레퍼런스](../../reference/fdtd/index.md)를 참고하고 반환된 메타데이터를 확인하세요. 원시 회절 스펙트럼만으로 정규화된 R/T나 완전한 입사 전력이 자동으로 보증되지는 않습니다.

해석된 설정과 제한을 살펴보려면 `check`를, 재사용할 수 있는 이산 계약을 세우려면 `prepare`를, 완전한 실제 입력마다 `prepared.run(problem)`을 사용하세요. `check`는 `report.ok`를 참으로 두는 경고도 나열합니다. 계면이 셀 경계 사이에 놓이는 셀 스냅 `SnappedSlab`이나 `SnappedBox`는 `geometry.cell_snapped`로 보고되며, 요청한 범위는 `expected`에, 실제로 놓이는 셀 경계는 `actual`에 담깁니다. 스냅과 스무딩 대안은 [FDTD 솔버가 문제를 모델링하는 방식](model.md)에서 설명합니다. 이전 결과는 자신의 관측값을 그대로 유지합니다. 이산 소스/모니터 배치, 격자 위상(topology) 또는 호환되지 않는 시간 진행을 바꾸려면 새로 준비해야 합니다.

## 검증과 이전 인터페이스

[FDTD 학습 과정](../../learn/fdtd/index.md)은 소스 정규화, 흡수층 동작,
세분화, 재료와 그래디언트 검사를 연결합니다. 해당 정식화에는 물리 단위의 [주기 적층](periodic-problems.md)
및 [native 격자](native-grids.md) 인터페이스를 사용하세요.
[이전 일반 엔진 설명서](../../migration/general-fdtd-v1.md)는 정규화된
구현 세부 사항과 과거에 검증된 제한을 기록합니다.
