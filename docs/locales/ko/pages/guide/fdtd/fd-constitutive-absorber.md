# 정합 흡수층 안의 결합 유전체 형상

> **과거 API:** 이 페이지에서 `wavebit.compat`를 쓰는 코드 조각과 설명은 과거 체크아웃이 필요합니다.
> 이 네임스페이스는 현재 설치본에는 없습니다. 퇴역 이전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행되지 않았습니다.


:::{note}
이 경로는 아직 현재 `wavebit.fdtd`에 없습니다. 현재 모듈의 흡수 면은 `Open()`이고, `fdtd.PML`이나 `fdtd.ConductiveAbsorber`를 `Numerics(absorber=...)`로 지정하며, 유한 물체는 그 면에 닿을 수 없습니다. 그래서 이 페이지의 예제는 이전 인터페이스인 `wavebit.compat.legacy.fd`를 사용합니다.
:::

이 페이지는 `Absorber` 안으로 이어지는 유전체 압출(extrusion)에 대해 구현된, 범위가 제한된
`wavebit.compat.legacy.fd` 경로를 설명합니다. 그 대수, 대상 테스트, 고정된 물리 측정과 완료된
전체/느린 회귀 테스트가 비트 단위로 일치하는 동결(freezer) 비교와 함께 기록되어 있습니다. 측정된 유한
도메인 간 차이는 여전히 상당하므로, 이 경로는 임의의 도파관에 대해 검증된 저반사 종단이 아닙니다.

`Absorber(cells=...)`는 PEC 벽을 뒤에 둔 단계형 정합 손실입니다. 이는 `PML`과 다릅니다. 결합된
스무딩 유전체는 유한 보간 링크가 정확히 같은 감쇠율끼리만 연결할 때에만 이 흡수층에 들어갈 수 있습니다.
예를 들어 z 방향으로 압출한 원기둥은 가로 xy 링크만 가지므로 z 흡수층을 지나 이어질 수 있습니다.
서로 다른 감쇠율을 잇는 0이 아닌 보간 링크는 거부하며, 층 안에서 끝나는 일반적인 계면도 여기에 해당합니다.

| 요구 사항 | 계약 |
| --- | --- |
| 재료 | 양의 등방성 무손실 `Material`. 호스트를 포함해 모든 곳에서 `mu=1`. |
| 기하 | 형상 범위가 없는 구체적인 voxel 기하. 일반적인 지지 영역, 점유율, 벽 검사는 그대로 적용됩니다. 형상만으로는 정확한 링크 증명을 우회하지 못합니다. |
| 면 | 흡수 축 하나. 나머지 면은 PEC, PMC, Periodic 또는 Bloch. 이 첫 경로에서는 PML, Symmetry, Axis1D를 쓸 수 없습니다. |
| 소스 | 전기 Dipole 또는 소스 없음. 자기 소스, 초기 필드, PlaneWave, TFSF는 쓸 수 없습니다. |
| 실행 | CPU에서 NumPy, JAX 또는 Torch. 고정된 스텝 수, 시간 또는 주기 수. 자동 Decay, 그래디언트, 이를 감싸는 변환된 계산은 쓸 수 없습니다. |
| 준비된 plan | 재료, 기하, 경계 또는 정적 소스/메모리 할당이 바뀌면 다시 만드세요. |

이 간결한 API 예제는 실행된 원기둥 보정 기하와 여기를 사용하며, Record 모니터 하나를 둡니다. 이 축약한
코드 조각 자체는 따로 실행하지 않았습니다. 측정 도구는 여기에 더해 양쪽, 자기 성분과 DFT 평면도
기록합니다. 예제가 추론된 파장에 의존하지 않도록 대역과 기하를 명시했습니다.

```python
# Earlier interface: Absorber is not in the current wavebit.fdtd yet.
from wavebit.compat.legacy import fd

center = (.015, .0075, 0.)
scene = fd.Scene(
    domain=fd.Domain(size=(.8, .8, 3.2)),
    resolution=fd.Resolution(dx=.05),
    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.Absorber(cells=8)),
    structures=(fd.Structure(
        fd.Cylinder(radius=.16, height=3.4, axis='z', center=center),
        fd.Material(eps=2.25),
    ),),
    sources=(fd.Dipole(
        'drive', center=center, component='Ex',
        band=fd.Band(wavelength=1., df=.15, cutoff=1e-3),
    ),),
    monitors=(fd.Record('out', at=(.015, .0075, .4), component='Ex'),),
    run=fd.Run(time=16., courant=.5),
)
prepared = fd.plan(scene)
report = fd.check(scene)
result = fd.run(scene, plan=prepared, backend='numpy', device='cpu')
trace = result['out']
```

길이 단위는 임의이지만 도메인, 파장과 시간이 함께 씁니다(`c=1`). 흡수층 셀 여덟 개는 선언한 z 범위의
양 끝을 차지하며, 도메인 바깥에 추가되지 않습니다. 원기둥은 층을 지나는 동안 단면이 균일하도록 두 벽
너머까지 뻗어 있습니다. 재료 안에서의 상대 파장과 가로 메시는 여전히 충분한 해상도가 필요하며, 계획이
성공했다고 해서 필드 정확도가 확립되지는 않습니다. 이 예제의 기하와 해상도에서는 거친 메시 경고와
서브픽셀 정확도 경고가 나오는 것이 정상입니다.

고정된 60회 계산 연구는 유한 도메인과 지속 시간을 비교했습니다. 소스 편광 방향의 전체 전기장 시계열은
짧은 도메인과 긴 도메인 사이에서 약 8.8–9.2% 달랐고, 짝이 되는 자기장 성분은 약 16–22% 달랐습니다.
개별 스펙트럼과 후반 구간의 차이는 더 클 수 있습니다. 연구의 DFT 평면을 쓴 모든 실행은 더 긴 시간의
대조 실행까지 포함해 안정되지 않았습니다. 이것들은 민감도 측정이며, 반사 계수나 연속체 오차의 상한이
아닙니다. 더 긴 유한 도메인이 정확한 기준을 제공하지는 않습니다. 전체
[측정과 경고](../../validation/fdtd/fd-constitutive-absorber.md)를 참고하세요.

이 새 경로에서 에너지 진단은 흡수층 전체에 걸쳐 자기 손실 보정을 적용한 양의 엇갈린 저장 에너지를
읽습니다. 연속한 자기장끼리의 곱만 쓰는 이전 방식은 감쇠가 있는 셀에서 그 저장 에너지가 아닙니다.
소스가 없는 허용된 이산 시스템은 CFL 한계 안에서 보정된 저장 에너지를 소산합니다. 구동되는 가우시안
펄스는 꼬리를 통해 여전히 일을 합니다.

결합된 지지 영역이 흡수층에서 떨어져 있는 장면은 기존의 관측자와 필드 경로를 그대로 유지합니다.
유전율이 같은 곡면 물체도 잠재적 지지 영역이 흡수층과 겹치면 새 플럭스 항을 확보합니다. 그 필드는
균질한 대조 장면과 비교할 수 있지만, 두 관측자의 에너지 값은 일치하지 않아도 됩니다.

이 경로는 CPML 안의 결합 재료나 일반적인 손실/자성 스무딩을 허용하지 않습니다. 나머지 계약은
[유도와 범위를 정한 측정](../../validation/fdtd/fd-constitutive-absorber.md)과
[일반 FDTD API](../../migration/general-fdtd-v1.md)를 참고하세요.
