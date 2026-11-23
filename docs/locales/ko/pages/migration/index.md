# 물리 단위 API로 옮기기

:::{note}
일반 FDTD 선언은 2026-10-19에 이름이 바뀌었고(`fdtd.Record`는 이제
`fdtd.TimeSeriesMonitor`, `fdtd.Region`은 이제 `fdtd.SnappedBox` 등), `fdtd.native`와
`fdtd.periodic`의 선언도 같은 날 바뀌었습니다(`native.Record`와 `periodic.Record`는 이제
`Snapshots`, `native.Tail`은 이제 `native.DecayCheck` 등). RCWA, 관측 객체, 공통 모듈은
2026-10-20에 뒤따랐고(`rcwa.OrderExcitation(orders=)`는 이제 `rcwa.IncidentOrders(weights=)`,
`TimeSeries.times`는 이제 `times_seconds` 등), 결과 키와 리포트 키도 같은 날 바뀌었습니다
(`validity.checks['stable']`는 이제 `'runtime_stability'`, `orders=`는 쌍만 받음 등). 회절 진폭은
2026-11-08에 뒤따랐습니다(`amplitude_s`는 이제 `amplitude_ratio_s`이고 그 옆에 V/m 단위의 새
`field_s`가 있음; `reference['incident_amplitude']`는 이제 `incident_field` 또는
`incident_spectrum`; `fdtd.periodic`의 진폭비는 더 이상 소스 진폭에 비례하지 않음). 예전 표기와 새
표기 전체는 [공개 이름](public-names.md)을 보세요.
:::

`wavebit.compat`은 제거되었습니다. 계산은 물리 단위의 Problem/Numerics 선언, 명시적 실행,
`check` → `prepare` → `prepared.run(problem)`을 씁니다. 바탕의 수치 엔진과 그 과학적 회귀 테스트는
그대로 남아 있으며, 이를 대신할 가변(mutable) 파사드는 내보내지 않습니다.

| 예전 작업 흐름 | 현재 인터페이스 |
| --- | --- |
| 가변 `rcwa(...)` | `rcwa.Problem`, `rcwa.Numerics`, `rcwa.check/prepare/run` |
| 일반 `fd` scene/plan/run | `fdtd.Problem`, 물리 단위 `fdtd.Scene`, 별도의 `fdtd.Numerics` |
| 주기 `fdtd(...)` | [`fdtd.periodic`](../guide/fdtd/periodic-problems.md), 명시적 `shared` 또는 `reference` 알고리즘 |
| `fdtd_grid(...)` | [`fdtd.native`](../guide/fdtd/native-grids.md), SI 전류/전도도/극점, 필드, 스냅숏과 푸리에 적분 |
| `fit_fdtd_modes` | `observations.fit_modes`, 초·Hz와 1/s 단위의 감쇠율 |
| 사용자 정의 일반 FDTD 형상 | `fdtd.Shape` 정적 형상 프로토콜 |
| `solver.slice_structure` | `rcwa.sample_structure(problem, quantity='epsilon' or 'mu')` |
| 실험적 NVF와 산란 연산자 | `rcwa.experimental`과 `rcwa.operators` |
| `JobSpec` v10 | `migration.migrate_legacy_jobs(document, units=Units(...))` → 현재 `Job` 객체의 튜플 |
| capability manifest v21 | v22: `current_lifecycle`이 공개 색인이고, 자세한 `solvers`/결과 표는 내부의 정규화된 엔진을 명시적으로 기술합니다. v23 (2026-10-19): [공개 이름](public-names.md)과 `wavebit.job/v2`. v24 (2026-10-19): [native·periodic 이름](public-names.md#native-grid-and-periodic-stack-names). v25 (2026-10-20): [공통·관측·RCWA 이름](public-names.md#shared-observation-and-rcwa-names). v26 (2026-10-20): [결과 키와 리포트 키](public-names.md#result-and-report-keys). v27 (2026-11-08): [진폭비, 필드와 스펙트럼](public-names.md#amplitude-ratios-fields-and-spectra) |
| import 시점의 JAX 정밀도 변경 | 배열을 만들기 전에 `jax.config.update('jax_enable_x64', True)`를 설정 |

공간 길이는 `Units`를 쓰고, 시간·전도도·재료 비율은 SI를 씁니다. H는 A/m입니다. 층 적층은 여전히
아래에서 위로 쌓고, 각도는 라디안입니다. 복소 필드를 비교할 때는 선언한 수치 알고리즘, 셀 원점,
위상 기준, 소스 정규화, 적분 측도를 유지하세요. 솔버끼리는 진폭비, 효율, 전력, 필드를 비교하고,
한 솔버의 `incident_field`(V/m)를 다른 솔버의 `incident_spectrum`(V*s/m)과 비교하지 마세요.
native 푸리에 필드는 시간 적분이지 보정된 페이저가 아닙니다. 주기 평면 전력은 y 불변 형상에서는
W/m, 교차 셀에서는 W입니다.

## 기존 작업

유지되는 예제는 모두 물리 단위 API를 씁니다. 내부 수치 테스트와 연구 도구는 자신이 측정하는
구현을 정확히 import하며, 이런 내부 경로에는 공개 안정성 약속이 없습니다. 그 독립 기준, 허용
오차, 과학적 단언은 그대로 남아 있습니다.

JIT 아래에서 형상을 바꿀 때는 지원되는 장치 측 보호 장치나 명시적
[`topology_check='host'`](../guide/rcwa/guarded-shape-jit.md)를 씁니다. 수치적 위상 검사는 계속
켜져 있습니다. 일반 FDTD는 [종료 규칙과 흡수층](../guide/fdtd/stopping-and-absorbers.md)과 명시적
[기판 연속](../guide/fdtd/fdtd-differentiation.md)을 제공합니다. 사용자 정의 `Shape` 메서드는 격자
길이를 Problem의 단위로 받으며, 선언된 인스턴스 속성은 준비 결과를 재사용할 수 있도록 모두
정적입니다. 사용자 정의 지지 영역은 Open 패딩에서 떨어져 있어야 하고, 사용자 정의 객체는 이식 가능한
Job 태그가 아닙니다. 구현되지 않았던 모드 소스/모드 모니터/가우시안 빔 자리 표시자는 동작하는 일반
FDTD 솔버였던 적이 없습니다.

## Job 파일

```python
from wavebit import Job, Units, run
from wavebit.migration import migrate_legacy_jobs

# old_document is a v10 JSON string or mapping. Supply its actual original length unit.
jobs = migrate_legacy_jobs(old_document, units=Units(length='nm'))
for job in jobs:
    portable = Job.from_json(job.to_json())
    result = run(portable)
```

변환은 모든 스윕 점을 원래 순서대로 펼치고, 수치 선택, 재료 표, 편광을 보존합니다. 솔버를
실행하거나 원본 문서를 덮어쓰지 않습니다. 지원하지 않는 필드는 명시적으로 실패합니다. 하위 기록을
교체하기 전에 변환된 관측값을 원래 실행과 비교하세요. 다른 과거 스키마 버전은 원래의 버전 관문을
유지합니다. JSON 버전 번호를 바꾸는 것은 마이그레이션이 아닙니다.

## 과거 결과 재현

예전 [번호 붙은 튜토리얼](https://github.com/fieldalgo/wavebit/blob/main/tutorials/README.md#historical-tutorials)
열 개는 과거의 교육 기록이며 아래의 고정 소스가 필요합니다. 계산 셀과 출력은 바뀌지 않았고,
튜토리얼 색인이 각 주제를 현재의 물리 단위 예제와 가이드에 연결합니다. 두 개의 `00` 생애주기
노트북은 현재 패키지로 실행됩니다.

고정된 측정은 원래의 바이트, 커밋, 의존성 출처를 유지합니다. 다시 실행할 때는 기록된 체크아웃과
환경을 쓰세요. 일부러 마지막 compat 표기를 쓰는 스크립트에 대한 퇴역 직전 기준은
[3c8a50f8a6506240169538957e99648a4b18ca3a](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
그 리비전으로 별도의 체크아웃을 만들고, 현재 패키지 설치를 암묵적인 과거 환경으로 취급하지 마세요.
이 편의용 고정 리비전보다 더 이른 측정 커밋이 우선합니다.

예전 [API](legacy-api.md), [일반 FDTD](general-fdtd-v1.md), [재료](materials-v1.md),
[Job](jobs-v1.md), [백엔드](backends-v1.md), [자동 미분](autodiff-v1.md) 기록은 옛 규약을 설명합니다.
그 안의 과거 코드 조각은 현재 import할 수 있는 이름을 기술하지 않습니다. 문서 위치 이동과 사용자
코드 마이그레이션은 새로운 과학적 측정과 별개입니다.
