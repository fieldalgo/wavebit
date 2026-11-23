# 일반 FDTD: `wavebit.fd`

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 필요합니다.
> 현재 설치에는 이 네임스페이스가 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행되지 않았습니다.


:::{note}
이 페이지는 예전 인터페이스의 전체 호환성 참조입니다. `wavebit.compat.legacy`를 명시적으로 쓰세요. 새 애플리케이션은 [현재 생애주기](../guide/common/lifecycle.md)와 [현재 API 참조](../reference/api.md)로 시작하세요. 여기 인용한 수치 근거는 원래의 범위를 그대로 유지합니다.
:::

`from wavebit.compat.legacy import fd`는 일반 전자기 시간 영역 엔진을 선택합니다.
명시적인 소스, 경계, 재료, 모니터를 갖춘 균일한 Yee 격자입니다.
자체 `fd.Scene` API를 통해 유한 영역과 주기 셀을 지원합니다.
API와 검증된 범위는 아직 개발 중입니다.

Wavebit에는 현재 서로 다른 두 FDTD 진입점이 있습니다.

| 진입점 | 입력과 결과 | 설명서 |
| --- | --- | --- |
| `wavebit.fdtd(...)` | RCWA와 공유하는 주기 `Stack`/`Incidence` 인터페이스이며, 산란 `Result`를 반환합니다. 기본 경로는 legacy NumPy이고, 백엔드를 명시하면 공통 NumPy/JAX/PyTorch 경로가 선택됩니다. | [주기 적층 시간 영역 솔버](../guide/fdtd/time-domain.md) |
| `from wavebit.compat.legacy import fd` 다음의 `fd.run(scene, ...)` | 공간 영역과 이름 붙은 모니터를 가진 `fd.Scene`이며, 모니터 이름으로 색인하는 `fd.Results`를 반환합니다. | 이 페이지 |

이 장면 클래스와 재료 클래스는 별개의 API입니다. 예를 들어 `fd.Material`은
실수 시간 영역 재료 모델을 쓰지만, 루트의 `wavebit.Material`은 단일 주파수의
복소 유전율을 담을 수 있습니다. 두 진입점 사이에서 키워드나 각도 규약을 옮겨
쓰지 마세요.

## 선언, 확인, 실행

| 선언 | 계약 |
| --- | --- |
| `Domain(size=..., center=...)` | 세 값 묶음은 `(x, y, z)`입니다. 격자는 중심을 고정한 채 요청한 크기를 정수 개의 셀로 올림합니다. |
| `Boundaries(x=..., y=..., z=...)` | 모든 면을 명시해야 합니다. 면 유형 하나는 한 축의 양 끝에 모두 적용되고, `(low, high)` 쌍을 주면 양 끝을 따로 정합니다. |
| `Resolution(dx=...)` | 모든 축에 하나의 균일한 셀 크기를 씁니다. 대신 `cells_per_wavelength=`를 `wavelength=`와 함께 줄 수도 있습니다. |
| `Run(...)` | `steps=`, `time=`, `periods=` 중 정확히 하나를 줍니다. 주기 수를 쓰려면 공유되는 구체적인 소스 대역 중심이 필요합니다. `courant`의 기본값은 0.5입니다. |
| `background=`, `structures=`, `sources=`, `monitors=` | 기본값은 진공과 빈 튜플입니다. 구조물은 형상과 재료를 짝짓고, 소스와 모니터의 이름은 그 결과를 식별합니다. |

이 작은 PEC 공진기는 전기 펄스를 기록합니다. 일부러 반사하는 벽을 두었으므로
소스가 끝난 뒤에도 필드가 남을 수 있습니다. 이것은 시계열 예제이며, 열린 공간
산란이나 수렴 계산이 아닙니다.

```python
from wavebit.compat.legacy import fd

scene = fd.Scene(
    domain=fd.Domain(size=(1.0, 1.0, 1.0)),
    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
    resolution=fd.Resolution(dx=0.05),
    run=fd.Run(time=8.0),
    background=fd.Material(eps=1.0),
    sources=(fd.Dipole(
        'pulse', center=(0.0, 0.0, 0.0), component='Ey',
        band=fd.Band(wavelength=1.0, df=0.15),
    ),),
    monitors=(fd.Record('probe', at=(0.2, 0.0, 0.0), component='Ey'),),
)

plan = fd.plan(scene)
report = fd.check(scene)  # Prints a report; does not advance the fields.
result = fd.run(scene, backend='numpy', plan=plan)
probe = result['probe']

print(probe.t.shape, probe.values.shape)
print(result.final.E.shape, result.final.H.shape)
print(result.report.bytes_per_state)
```

점 탐침 클래스는 `Record`입니다. 이 클래스는 지정한 성분의 가장 가까운 Yee 점을
읽으며, 요청한 정확한 위치에서 보간한 필드를 읽지 않습니다.
`probe.t`에는 실제 표본 시각이 들어 있습니다. 전기장 성분은 정수 시간 간격에서,
자기장 성분은 반 시간 간격 이전에 기록됩니다. `probe.values`는 실행한 백엔드에
있고, 시간 좌표는 NumPy 배열입니다. 최종 `E`와 `H` 배열도 같은 반 시간 간격
오프셋을 유지합니다.

`fd.plan(scene)`은 정적 격자를 만들고 선언된 제약을 검사합니다.
`fd.check(scene)`은 독립적으로 장면을 계획하고 NumPy로 낮추어 `Report`를 출력하고
반환합니다. 시간 적분은 아닙니다. 계획을 재사용하려면 구체적인 계획을
`fd.run(..., plan=plan)`에 넘기세요. 구체적인 장면을 바꾸면 새 계획을 만드세요.
미분하는 실행은 [일반 엔진 미분하기](../guide/fdtd/fdtd-differentiation.md)의 더 좁은
계획 재사용 규칙을 따릅니다. 완료된 결과에는
`result.plan`, `result.report`, `result.sources`, `result.diagnostics`도 담겨 있습니다.

## 선택적 조기 종료

초기 필드나 Dipole 소스를 쓰는 구체적인 CPU 과도 응답 계산에는, 기존의 상한이 있는
실행에 명시적인 진폭 임계값과 시간 창을 더하세요.

```python
scene = scene.replace(run=scene.run.replace(
    decay=fd.Decay(threshold=1e-5, window=1.0),
))
result = fd.run(scene, backend='numpy')
print(result.diagnostics['stopping']['reason'])  # 'decay' or 'cap'
```

이 값들은 표기를 보여 주기 위한 것입니다. 연구하는 과도 응답에 맞게 고르세요. 두
인수 모두 필수입니다. `window`는 `Run(time=...)`과 같은 시간 단위를 쓰고,
`threshold`는 0과 1 사이(양 끝 제외)의 유한한 진폭 비율입니다. 기존의 `steps`,
`time`, `periods` 한도 중 정확히 하나를 엄격한 상한으로 유지하세요. 마지막의 불완전한
창은 종료 조건을 만족시킬 수 없습니다.

전체 필드 배열 각각과 시간에 따라 변하는 재료/흡수층 배열 각각은, 완전한 창 하나
내내 자신의 최댓값의 요청한 비율 아래에 머물러야 합니다.
창은 모든 Dipole의 하강하는 가우시안 포락선이 Band 차단값과 이 임계값 둘 다보다
낮아진 뒤에만 자격을 얻습니다. 파형은 바뀌지 않습니다.
정적 잔류나 지속되는 진동은 상한에 도달할 수 있으며, `reason='cap'`은 수렴을 주장하지
않습니다. 작은 진폭이 스펙트럼 오차 한계는 아닙니다.

`result.plan.steps`, `result.report.steps`, 이력, DFT 끝점 보정은 실제로 실행된 앞부분을
기술합니다. 종료 진단에는 `cap_steps`, 판정 기준, 그룹별 진폭도 남습니다. 반환된 짧은
계획은 원래의 상한이 있는 장면에 재사용할 수 없으니, 원래의 `fd.plan(scene)`을
보관하세요. 이력은 처음에 상한만큼 할당되고 상태 축약은 매 시간 간격마다 작업을
더하므로, 조기 종료가 반드시 더 빠르지는 않습니다.

PlaneWave/TFSF 소스, 자동 미분, 바깥을 감싸는 JAX 변환, CPU가 아닌 실행에서는 이
옵션을 거부합니다. 일반적인 고정 길이 실행은 기존 계약을 유지합니다. [종료 측정과
한계](../validation/fdtd/fdtd-decay-stopping.md)를 보세요.

## 단위, 좌표, 백엔드

| 양 | 일반 엔진의 규약 |
| --- | --- |
| 길이 | 영역, 기하, `dx`, 진공 파장에 하나의 단위를 고르세요. |
| 시간과 주파수 | `c = eps0 = mu0 = 1`: `time`과 `dt`는 길이이고, `f = 1/wavelength`이며, 각주파수는 `2*pi*f`입니다. 정규화된 시간 1은 사용자의 길이 단위를 물리적 광속으로 나눈 값에 해당합니다. |
| 재료 | `eps`와 `mu`는 상대값입니다. 전도도 `sigma`의 단위는 길이의 역수이고, 극점 주파수와 감쇠율은 각주파수를 씁니다. 복소 상수는 거부하므로 `Conductor`나 `Dispersive`를 쓰세요. |
| 각도 | `PlaneWave(theta=..., phi=...)`와 형상 회전 각도는 도 단위입니다. Bloch 위상과 소스 파형 위상은 라디안입니다. |
| 배열 | 필드의 모양은 `(3, nz, ny, nx)`이고, 성분 순서는 x/y/z입니다. 공개 좌표 세 값 묶음은 계속 `(x, y, z)`입니다. 자기장은 `Z0 * H`로 저장됩니다. |
| 시간 간격 크기 | `dt = courant * dx / sqrt(3)`. 재료와 기하에 따른 안정성 검사가 이 선택을 거부할 수 있으며, 이 진공 공식만으로 재료가 있는 장면이 보증되지는 않습니다. |

`fd.run`은 `backend='numpy'`, `'jax'`, `'torch'`(또는 `wavebit.Backend`)를 받습니다.
기본값은 NumPy입니다. 세 CPU 백엔드는 테스트되었습니다. JAX나 Torch를 고를 때는
배치를 명시적으로 유지하도록 `device='cpu'`를 요청하세요. 이 CPU 기록으로 GPU의
정확성과 성능이 입증되지는 않습니다. 계획은 `float64`를 쓰고, Bloch나 `Axis1D` 면에는
`complex128`을 씁니다. JAX와 Torch는 문서화된
[미분 범위](../guide/fdtd/fdtd-differentiation.md)를 지원합니다. 미분 가능한 백엔드를 고른다고
모든 기하나 재료 매개변수가 허용되는 것은 아닙니다.

리포트의 `bytes_per_state`는 주 격자 E/H 배열, CPML 이력, 재료 메모리의 논리적 데이터
크기이며, 예약된 비활성 극점 슬롯과 혼합 필름의 보조 배열을 포함합니다. 소스 이력과
보조 소스 열, 관측기와 에너지 진단, 실행 중 임시 배열, 체크포인트, 할당기/Python
오버헤드는 제외합니다. 프로세스나 장치의 최대 메모리가 아닙니다. 계수와 주파수 영역
모니터 누적기는 리포트에 별도 항목이 있으며, 그 항목들을 더해도 모든 실행 비용이
포함되지는 않습니다.

## 지원하는 조합과 그 한계

| 용도 | 현재 계약과 더 읽을거리 |
| --- | --- |
| 소스와 관측값 | `Dipole`, `PlaneWave`, 축 방향 `TFSF`; 점 `Record`, 스펙트럼 `Field`/`Flux`, 주기 `Diffraction`, 열린 영역 `FarField`. 정규화와 경계 제한은 [메타표면](../guide/fdtd/fdtd-metasurface.md)과 [복사](../guide/fdtd/fdtd-radiation.md)를 보세요. |
| 경계 | `PEC`, `PMC`, `Periodic`, `Bloch`, `Axis1D`, `Symmetry`, `PML`, `Absorber`이며, 재료 정합, 소스 이격, 기하 검사를 따릅니다. 흡수층은 영역 안의 셀을 차지합니다. |
| 흡수층을 관통하는 결합 유전체 기하 | 이산 결합이 감쇠 프로파일과 교환 가능할 때, 유계인 무손실 압출 형상이 정합된 `Absorber` 축 하나를 관통해 뻗을 수 있습니다. 구체적인 CPU 실행은 전기 Dipole과 고정 길이를 쓰며, PML과 자동 종료는 이 경로의 범위 밖입니다. [계약](../guide/fdtd/fdtd-constitutive-absorber.md)을 보세요. |
| 셀 스냅 재료 영역 | `Slab`과 `Region`; 실수 스칼라/대각 `Material`, `Conductor`를 통한 전도도, `Dispersive`를 통한 수동 Drude/Lorentz/Debye 모델이며, 계획 검사를 따릅니다. `fd.eps_at`은 극점 모델을 진공 파장에서 계산합니다. |
| 스무딩 유전체 형상 | `Sphere`, `Box`, `Cylinder`, `Prism`은 유계인 등방성 무손실 재료, 이격, 경계 계약을 가지며, 특정한 맞닿은 접합만 허용됩니다. 면을 가로질러 균일하거나(그 면을 통해 빠져나가는 기판) 면이 `Symmetry` 평면인 경우가 아니면, 스무딩 형상은 영역 안에 있어야 하고 표면이 각 높은 쪽(`+`) 벽에서 적어도 반 셀 떨어져 있어야 합니다. 거부 메시지가 둘 중 무엇을 어겼는지 알려 줍니다. [스무딩 점유율](../validation/fdtd/fdtd-subpixel-occupancy.md)을 보세요. `discretisation='stencil'`에는 별도의 [고립된 구 계약](../design/fdtd-symmetric-stencil.md)이 있습니다. |
| 스무딩 손실/분산 필름 | 평범한 등방성 비자성 호스트 안의, 회전하지 않고 범위를 지정하지 않은 구체적인 `Box` 하나이며, 정확히 두 축에 걸쳐 필요한 돌출을 갖고 유한한 두께 방향으로 흡수층에서 떨어져 있어야 합니다. 도체, 유한 Debye 합, 유한 Drude/Lorentz 합, 혼합 계열에는 각자의 제한이 있습니다: [전도성](../validation/fdtd/fdtd-conductive-film.md), [Debye](../validation/fdtd/fdtd-debye-sum-film.md), [진동자](../validation/fdtd/fdtd-oscillator-sum-film.md), [혼합 계열](../validation/fdtd/fdtd-hybrid-film.md). |
| 분리된 전도성 필름 | 회전하지 않고 범위를 지정하지 않은 구체적인 voxel `Box` 구조물 둘 이상이며, 모두 `Conductor` 재료(`sigma=0` 포함)를 쓰고, 하나의 평범한 양의 등방성 무손실 비자성 `Material` 호스트 안에 있습니다. 모든 필름은 법선 축을 공유하고, 패딩을 더한 격자 지지 영역은 쌍마다 서로 겹치지 않습니다. 초기 필드, 추적된 기하/재료/대역 값, 겹치는 지지 영역, 선언된 극점 계열을 섞는 적층은 거부합니다. [전도성 적층](../validation/fdtd/fdtd-conductive-stack.md)을 보세요. |
| 맞닿은 전도성 필름 | 면이 정확히 일치하고 각각 두께가 적어도 한 셀인 그런 필름 정확히 둘이, 하나의 수동 합집합 메모리를 공유합니다. 접촉 허용 오차는 없으며, 세 재료가 만나는 셀, 손실 호스트, 초기 필드, 추적은 거부합니다. 파동 정확도는 받아들여지지 않았습니다([#51](https://github.com/fieldalgo/wavebit/issues/51)). [맞닿은 필름](../validation/fdtd/fdtd-conductive-contact.md)을 보세요. |
| 분리된 Debye 필름 | 서로 겹치지 않는 평행한 voxel Box 필름 둘 이상이며, 각각 수동 Debye 극점 하나, 둘 또는 유한한 합을 갖고 같은 평범한 무손실 비자성 호스트 안에 있습니다. 선언된 슬롯과 단일 항 축약은 유지됩니다. 고정 길이만 지원하며, 초기 필드, 추적된 입력, 겹치는 패딩 지지 영역, 혼합 계열 적층은 거부합니다. [Debye 적층](../validation/fdtd/fdtd-debye-stack.md)을 보세요. |
| 분리된 단일 진동자 필름 | 평행한 voxel `Box` 필름 둘 이상이며, 각각 정확히 하나의 수동 `Drude` 또는 `Lorentz` 극점을 갖고 같은 평범한 무손실 등방성 비자성 호스트 안에 있습니다. 필름마다 다른 진동자 종류를 쓸 수 있습니다. 지지 영역은 서로 겹치지 않아야 하고, 실행 길이는 고정입니다. 초기 필드, 미분, 필름 안의 합, Debye 구성원, `Decay`는 거부합니다. [유계 계약과 검증 상태](../validation/fdtd/fdtd-oscillator-stack.md)를 보세요. |

전도성 적층은 평범한 `Scene(structures=(...))` 튜플로 선언합니다. 별도의 적층 생성자는
없습니다. 각 필름은 같은 두 가로 축에 걸쳐야 하고 양쪽에 적어도 한 셀의 돌출이 있어야
합니다. 법선 방향 두께는 적어도 `dx`이어야 하고, 패딩을 더한 지지 영역은 법선 방향 영역
벽과 모든 흡수층에서 떨어져 있어야 합니다.

이격은 고정된 물리적 틈이 아니라 실제로 패딩을 더한 반열린 격자 지지 영역에서
검사합니다. 거부 메시지는 인접한 구성원과 그 지지 구간을 알려 줍니다. 호스트 틈을
넓히거나 `dx`를 줄이세요. 전도도가 0인 구성원도 선언된 적층과 그 메모리 할당의 일부로
남습니다. 구성원, 선언 순서, 호스트 중 하나라도 바꾸면 새 계획이 필요합니다.

단일 진동자 적층도 같은 `Scene(structures=(...))` 표기를 쓰며, 구성원마다
`Dispersive(..., poles=(pole,))` 선언 하나를 둡니다. 세기가 0인 Lorentz 구성원도
메모리 슬롯을 유지하고, 보수적인 시간 간격 검사에서 유한한 명목 주파수 제곱을 그대로
씁니다. 단일 필름에서 거부되는 극단적인 비활성 주파수는 여기서도 거부됩니다. 경계 선언이나
전체 소스 할당을 바꿔도 새 계획이 필요합니다. 벌크 스펙트럼 검사는 모든 구성원의 내부
Lorentz 공명을 포함하지만, 최종 관측값 오차의 한계는 아닙니다.

필름 지원은 임의의 곡면 손실 물체, 분산 호스트, 손실 접합, 또는 그 기하/재료/대역
미분으로 확장되지 않습니다. 일반 CSG, 래스터 기하, 기울어진 압출 축, 비균일 격자,
`GaussianBeam`, `ModeSource`, `ModeMonitor`는 이 엔진에 구현되어 있지 않습니다. 내보낸
자리 표시자 이름도 생성을 거부할 수 있습니다.

계획이 성공하거나 실행이 안정적이라고 정확도가 보증되지는 않습니다. 필요한 양에 맞춰
공간, 실행 길이, 흡수 경계를 각각 독립적으로 세분하고, 경고를 읽으세요.
[C7 메타표면 비교](../validation/fdtd/fdtd-metasurface-validation.md)는 M3의 느린 수렴에 대한
원인 미해결 문제를 포함해 여전히 통과하지 못했습니다.
유계 필름 작업이 일반 C8 스무딩을 완성하지는 않으며, 그 검증 기록에는 혼합된 사례와
잡음으로 판정되지 않은 사례가 남아 있습니다. 벌크 메시 척도에 대해 유도한 규칙이 모든
계면에 대한 수렴 정리는 아닙니다.
