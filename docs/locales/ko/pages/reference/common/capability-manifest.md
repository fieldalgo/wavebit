# Capability manifest — 솔버가 스스로에 대해 말하는 것과 그 경계선

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 필요합니다.
> 이 네임스페이스는 현재 설치본에 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행하지 않았습니다.


작성: 2026-08-11.

현재 버전 **27**(2026-11-08)은 다섯 번째 공개 이름 변경을 기록합니다. 회절 결과에서 이름 하나는 물리량
하나를 담습니다. 진폭은 모든 솔버에서 무차원인 `amplitude_ratio_s`,
`amplitude_ratio_p`, `amplitude_ratio_xyz`이며(`fdtd.periodic`의 진폭은 더 이상 소스 진폭을 담지
않습니다), 그 옆에 연속파 구동에서 V/m 단위인 `field_s`, `field_p`, `field_xyz`가 있습니다.
`reference['incident_amplitude']`는 `incident_field`(V/m) 또는 `incident_spectrum`(V*s/m)이 되었고,
`amplitude_units`는 `amplitude_ratio_units`가 되었습니다. `RawDiffractionSpectrum`은 `DiffractionOrders`의
형제 클래스로, 그 배열은 `spectrum_s`, `spectrum_p`, `spectrum_xyz`입니다. `fdtd.periodic` 차수에는
`power`와 `incident_power`가 생겼습니다([마이그레이션 안내](../../migration/public-names.md#amplitude-ratios-fields-and-spectra) 참고).
manifest 키는 바뀌지 않고, `current_lifecycle.job.schema`는 `wavebit.job/v2`, 실행 기록은
`wavebit.run/v2`로 그대로입니다. 그러나 이것들은 소비자가 결과를 읽을 때 쓰는 이름이며, 옛 이름은 별칭 없이
사라졌습니다. v26을 기대하는 소비자는 이 버전 변경을 인식해야 합니다.

갱신 2026-10-20 (나중): manifest 버전 **26**은 네 번째 공개 이름 변경을 기록합니다. 결과와 보고서의 키가
솔버 전체에서 하나의 어휘가 되었습니다(`validity.checks['runtime_stability']`,
`time_step_seconds`, 하나로 통일한 `DiffractionOrders.reference` 키 집합, `area.item` 오류 코드, 명시적인
쌍인 `orders=`;
[마이그레이션 안내](../../migration/public-names.md#result-and-report-keys) 참고). 내보내는 manifest
값은 미국식 철자를 씁니다. `shape_coordinates.origin`과 `lattice.fourier_origin`은
`'cell_center'`이고, 키 `omitted_centre`는 `omitted_center`입니다. `current_lifecycle.job.schema`는
`wavebit.job/v2`로 그대로입니다. v25를 기대하는 소비자는 이 버전 변경을 인식해야 합니다.

갱신 2026-10-20: manifest 버전 **25**는 세 번째 공개 이름 변경을 기록합니다. RCWA, 관측 객체와 공유 모듈도
자기가 무엇인지 말하게 되었습니다(`rcwa.OrderExcitation(orders=)`는
`rcwa.IncidentOrders(weights=)`, `TimeSeries.times`는 `times_seconds` 등;
[마이그레이션 안내](../../migration/public-names.md#shared-observation-and-rcwa-names) 참고).
manifest 키는 바뀌지 않으며 `current_lifecycle.job.schema`는 `wavebit.job/v2`로 그대로입니다. v24를
기대하는 소비자는 이 버전 변경을 인식해야 합니다.

갱신 2026-10-19 (나중): manifest 버전 **24**는 두 번째 공개 이름 변경을 기록합니다.
`fdtd.native`와 `fdtd.periodic` 선언도 자기가 무엇인지 말하게 되었습니다(두 곳 모두 `Record`는
`Snapshots`, native `Tail`은 `DecayCheck` 등;
[마이그레이션 안내](../../migration/public-names.md#native-grid-and-periodic-stack-names) 참고).
manifest 키는 바뀌지 않으며 `current_lifecycle.job.schema`는 `wavebit.job/v2`로 그대로입니다. 그러나
이것들은 소비자가 호출하는 이름이며, 옛 이름은 별칭 없이 사라졌습니다. v23을 기대하는 소비자는 이 버전
변경을 인식해야 합니다.

갱신 2026-10-19: manifest 버전 **23**은 공개 이름 변경을 기록합니다. 일반 FDTD 선언이 자기가 무엇인지
말하게 되었고(모니터 이름이 `Monitor`로 끝나는 것 등;
[공개 이름 마이그레이션 안내](../../migration/public-names.md) 참고),
`current_lifecycle.job.schema`는 `wavebit.job/v2`입니다. 옛 이름은 별칭 없이 사라졌습니다.
v22를 기대하는 소비자는 이 버전 변경을 인식해야 합니다.

갱신 2026-10-16: manifest 버전 **22**는 호환성 진입점을 퇴역시킵니다. `current_lifecycle.solvers`는
`rcwa`, `fdtd`, `fdtd.native`, `fdtd.periodic`을 색인하고, `current_lifecycle.job.migration`은 명시적
변환기를 지명합니다. `scope`는 그 공개 물리 계약과 남아 있는 내부 수치 표를 구별합니다. `solvers`는 이제
애플리케이션 구성용이 아니라 과학적 진단용으로 정확한 비공개 엔진 함수를 가리킵니다. v21을 기대하는
소비자는 이 버전 변경을 인식해야 합니다. 아래의 날짜별 항목은 과거 기록입니다.

갱신 2026-10-14 (최신): manifest 버전 **21**은 이름을 바꾼 굴절률 변환을 공개합니다.
`material_quantity.conversions`의 `eps_from_refractive_index`와 `refractive_index_from_eps`,
`material_dispersion`의 `from_refractive_index`입니다 -- 소비자가 호출하는 이름이며, 옛 이름은 별칭 없이
사라졌습니다.

갱신 2026-10-13: manifest 버전 **20**은 RCWA 계약을 일반 FDTD와 주기 FDTD 기능과 통합합니다. 세부적인
`paths`, `results`, 거부 표와 `solvers` 항목은 호환성 인터페이스를 설명하고(`solvers.rcwa.entry_point`는
`wavebit.compat.legacy.rcwa`, `solvers.fdtd.entry_point`는 `wavebit.compat.legacy.fdtd`), 새
`current_lifecycle` 키는 현재의 `wavebit.rcwa` / `wavebit.fdtd` 모듈과 그
`check` / `prepare` / `run`, `wavebit.CheckReport`, `wavebit.Job` 스키마를 지명합니다. 요약에서 말하는
FDTD 버전(10에서 12; 그 브랜치는 19까지 갔습니다)은 아래의 RCWA `10`에서 `12`와 별도로 FDTD 브랜치에서
센 것입니다.

갱신 2026-09-28: 공개 합성곱 행렬과 `solve_conv` 입력은 셀 중심을 씁니다(`amplitude-sp-frame.md` §8).
`lattice.fourier_origin = 'cell_centre'`와 `lattice.fourier_origin_expression`은 추가 키이므로 manifest
버전은 바뀌지 않습니다. 비공개 빌더와 일반 `solve()`는 산술과 출력을 그대로 유지합니다. 손으로 만든 패턴
있는 모서리 기준 입력은 다시 기술해야 하지만, manifest는 그 입력 경로를 생성기에 노출한 적이 없습니다.

갱신 2026-09-27: 모든 차수별 양의 가로 방향 위상 기준이 셀 중심 `x = y = 0`으로 옮겨졌습니다
(`amplitude-sp-frame.md` §7). 버전은 올리지 않았습니다. `results.amplitudes`에 이를 알리는
`phase_reference` 키가 생겼고 -- 모르는 키를 무시하는 소비자는 무시하는 추가 키입니다 --
`cartesian.meaning`의 문구를 바꾸었습니다. 이름이 바뀌거나 모양이 바뀐 키는 없으며, 물리가 바뀐 유일한
입력인 `excitation`은 여기서 입력으로 공개되지 않으므로 어떤 생성기의 장면도 의미가 바뀌지 않습니다. 진폭을
캐시해 둔 소비자가 마주치는 것은 `m + n`이 홀수인 모든 차수의 부호뿐이며, 단일 파동의 효율은 바이트까지
그대로입니다.

갱신 2026-09-27 (나중): 차수별 직교 좌표 진폭은 `li_nesting`처럼 버전을 올리지 않고 공개됩니다.
`results.amplitudes.cartesian`은 `r_xyz` / `t_xyz`, 그 `(n_orders_v, n_orders_u, 3)` 배치와 의미를
지명합니다. 이것을 별도의 키 아래 둔 것은, 모든 `amplitudes.fields` 항목이 `(n_v, n_u)`라고 가정하는
소비자가 길이 3인 축을 만나지 않게 하려는 것입니다. `per_order_frame`은 이제 그 좌표계가 순수 1D 경로를
포함한 모든 경로의 것이라는 점과, 그것이 `ky = 0` 행에서 무엇을 뜻하는지 말합니다 -- 이미 모든 경로라고
주장하고 있었으므로 문구를 바로잡은 것입니다.

갱신 2026-09-27: `factorization='laurent'`는 `li_nesting`처럼 버전을 올리지 않고 공개됩니다. 이 값은
`enums.factorization`과 두 경로의 `factorization` 목록에 들어가고(매개변수 경로의 목록은 이제 기본값이
먼저인 두 항목짜리입니다), 새 `tensor_materials` 키는 텐서 재료가 거부하는 것 -- `'laurent'` -- 을
지명합니다. 래스터는 그것을 받지만 텐서로 된 래스터는 받지 않으므로, 경로별 목록으로는 이를 말할 수
없습니다. `lattice.factorizations_refused_on_oblique`는 `['li']`로 그대로입니다. Laurent 규칙은 어떤 축도
지명하지 않습니다.

갱신 2026-09-25: `li_nesting` 축(Li 규칙, `'fixed'` 또는 `'symmetric'`)은 `nvf_projector`처럼 버전을
올리지 않고 공개됩니다. 그 이유와 함께 `paths[raster].li_nesting`, `enums.li_nesting`, 그리고 매개변수
경로에서 조건부 `factorization='li'`에 딸린 `conditional_options` 항목 두 개입니다.

갱신 2026-09-24: manifest 버전 **12**는 지정 입사 결과를 `incident`라고 이름 붙이고, `result.r_s`,
`result.r_p`, `result.t_s`, `result.t_p`를 통한 직접 진폭 접근을 공개합니다. 필드 재구성 선택자도
`'incident'`입니다.

갱신 2026-09-21: manifest 버전 **11**은 자동 재료 분산 평가와 수치 축 `raster_normal_method`,
`nvf_projector`를 기록합니다. JobSpec 버전 **10**은 +z가 위쪽인 아래에서 위로의 층 순서를 쓰고, 재료 표를
유지하며, 형상 표본 수를 `shape_nvf_grid_size`라고 부릅니다. 옛 옵션 키를 쓰는 소비자는
[키워드 마이그레이션](../keywords.md#nvf-names-changed-on-2026-09-21)을 따라 갱신하세요.

대상:
- `src/wavebit/manifest.py` — 문서 자체
- `tests/api/test_manifest.py` — 문서를 참으로 유지하는 것
- `src/wavebit/rcwa/_engine/engine.py` — manifest가 읽는 거부 표와, manifest가 설명하는 타입 분기를 하는
  `structure` 설정자. 백엔드별 사본이 `_core`로 합쳐지기 전에는 두 항목이었고, 두 번째 항목은 이제 그
  설정자를 갖지 않는 NumPy 백엔드를 가리켰습니다.

---

## 요약

Wavebit에는 서로 독립인 수치 선택이 다섯 가지 있습니다 — 모델링 경로, 푸리에 급수, 인수분해 규칙, 백엔드,
격자 표기입니다. 이를 다스리는 규칙은 아홉 개의 문서와 `src/wavebit/rcwa/_engine/engine.py`의 거부 표 두
개에 흩어져 있습니다. 이 프로세스 밖의 프로그램들, 즉 문서, 설정 UI, 조건을 추천하는 어시스턴트에게는 그
규칙이 필요합니다. 지금까지 그것을 얻는 유일한 방법은 소스를 읽고 찾은 것을 다시 옮겨 적는 것이었고,
그렇게 해서 소비자는 1년 전에 지워진 플래그를 사용자에게 자신 있게 안내하게 됩니다.

`wavebit.manifest`는 그 대신 규칙을 하나의 JSON 문서로 공개합니다. 이 문서는 코드에서 생성되고 테스트
모음이 코드와 대조해 검사합니다.

버전 10부터 manifest는 FDTD의 사용 가능한 실행 경로를 기본값과 따로 설명합니다.
`solvers.fdtd.backends`는 NumPy/JAX/Torch를 나열하고, `differentiable`은 미분 가능한 경로가 있음을
알립니다. `routes`를 확인하세요. `backend`를 생략하면 `legacy_numpy`가 선택되는데, 이 경로는 자동 미분이
없고 위쪽 입사만 지원합니다. 백엔드를 명시하면 `common`이 선택되며, JAX/Torch는 고정 메시의 각도, 래스터,
극점, 반공간 매개변수를 미분합니다. 버전 11은 총합이 일정한 층 두께, 그 명목상의 `mesh_thickness` 요구 사항,
그리고 움직이는 내부 경계와 정적 메시 기하의 구별도 지명합니다.
버전 12는 선택적인 C1 내부 계면 평균과 그 정적 `layer_smoothing` 폭, 필요한 메시 입력과 외부 여유를
추가합니다.
배열 각도와 반공간에는 각각 이름 붙인 정적 입력 요구 사항이 있습니다. native 격자 API는 이 장면 항목이
설명하지 않습니다.

common 경로는 보장되는 순수 입사 결과 열과 CPU 검증을 나열하며, GPU 검증은 주장하지 않습니다. 보편적인
`refused` 항목은 두 장면 경로에 모두 적용됩니다. Jones/chi 편광은 지원되고 아래쪽 입사는 common 경로가
지원하므로, 둘 다 보편적인 거부가 아닙니다. 런타임 테스트는 공개된 common 백엔드 각각, 아래쪽 Jones 장면,
반공간/두께 그래디언트를 실행하고, 생략된 항목을 선언된 요구 사항과 대조해 검사합니다.

```bash
python -m wavebit.manifest > wavebit-manifest.json
```

**결정: 기능과 제약만 담습니다.** manifest는 각 경로가 무엇을 구현하는지, 무엇이 왜 거부되는지, 그리고
`structure`의 어떤 표기가 어떤 경로를 선택하는지 말합니다. 아무것도 추천하지 않습니다.

## 경계선과 그것이 거기 있는 이유

| 포함 | 제외 |
| --- | --- |
| 각 경로가 구현하는 것 | 어느 경로를 쓸지 |
| 무엇이 거부되고 왜인지 | "`truncation_order≈5` 아래에서는 PyTorch" |
| `structure`가 경로를 고르는 방식 | 실험을 기술하는 어휘 |
| 어떤 재료 사례를 풀 수 없는지 | 추출 스키마, 시스템 프롬프트 |

기준은 그 사실을 코드와 대조해 확인할 수 있느냐입니다. `_FACTORIZATION_IMPLEMENTED_BY`는 확인할 수
있습니다 — 매개변수 솔버를 만들고 `li`를 요청해서 거부하는지 보면 됩니다. *"`truncation_order≈5` 아래에서는
PyTorch가 이긴다"*는 확인할 수 없습니다. 이것은 한 CPU 호스트에서 어떤 작업 부하를 잰 측정이며, 솔버는
그에 대해 아무 의견이 없습니다. 그렇게 확인할 수 없는 사실은 모듈에 속하지 않습니다. 그것을 공개하면
manifest의 보증이 보증하지 않는 대상까지 넓어지기 때문입니다.

이 경계선은 만드는 도중에 나온 범위 문제도 정리합니다. *"내 기둥은 둥글고 그 지름을 최적화하고 싶다"*를
조건 집합으로 옮겨 주는 조언자는 쓸모 있지만, 그것은 솔버의 성질이 아니라 사용자의 문제에 대한 조언이며,
추천 엔진을 키우기 시작한 라이브러리는 더 이상 라이브러리가 아닙니다. 그런 조언자는 이 저장소 밖에 있으면서
manifest를 소비합니다. 여기 남는 것은 이 저장소만 알 수 있는 절반입니다.

### 나중에 받아들인 하나: 결과 계약 (2026-08-12)

첫 판은 입력과 제약만 담았고, "솔버가 무엇을 돌려주는가"에도 선을 그었습니다. 그 절반은 첫 릴리스를 위해
결과 표면의 이름을 바꿀 때 받아들였습니다. 경계선을 이루는 바로 그 기준, *그 사실을 코드와 대조해 확인할
수 있는가?*를 통과하기 때문입니다. `results` 절의 모든 주장 — 필드 이름, 전력 대 진폭 종류, 차수 배치,
접근자, JobSpec 기본값 — 은 `tests/api/test_manifest.py`가 **실제 결과 객체**와 대조하므로, manifest를
빠뜨린 이름 변경은 테스트 모음을 실패시킵니다. 여전히 제외되는 것은 그대로입니다. 추천, 그리고 사용자의
실험을 기술하는 어휘입니다. 이 절은 계약에 *대한* 데이터이며, 계약 자체는 규범으로 남는
[`conventions.md`](conventions.md)입니다.

## 테스트가 거울 테스트가 아닌 이유

`src/wavebit/manifest.py`는 가져올 때 `src/wavebit/rcwa/_engine/engine.py`의 표를 읽으므로, 둘이
일치한다고 단언하는 것은 변수가 자기 자신과 같다고 단언하는 셈입니다. `tests/api/test_manifest.py`는 대신
manifest를 **런타임 동작**과 대조합니다.

| 테스트 | 실패하려면 무엇이 깨져야 하는가 |
| --- | --- |
| 각 표기가 주장하는 모델링 경로를 선택함 | `structure` 설정자의 타입 분기가 바뀌거나, 표기를 설명하는 문장이 실제로 받는 것과 어긋남 |
| manifest가 내주지 않는 옵션은 거부됨 | 경로가 manifest에서 표현할 수 없다고 한 것을 더 이상 거부하지 않음 |
| manifest가 내주는 옵션은 실제로 빌드됨 | 경로가 자신이 거부하는 것을 구현한다고 설명됨 |
| 기울어진 셀의 거부가 유지됨 | `_reject_unsupported_lattice`가 느슨해지거나 엄격해짐 |
| 인용한 모든 소스가 존재함 | 주장이 기대던 파일의 이름이 바뀌거나 삭제됨 |

가운데 두 개가 중요한 쌍입니다. 거부 표가 그 가운데에 놓인 고리를 닫아 주므로, manifest는 표 하나가
*말하는* 것이 아니라 솔버가 *하는* 것에 묶입니다.

## 두 번 기술되는 제약

`lattice.refused_on_oblique`는 기울어진 셀에 따르는 두 거부를 문장으로 설명합니다. 같은 둘이
`paths_refused_on_oblique`와 `factorizations_refused_on_oblique`에도 이름으로 나열되는데, 무엇을 추천할지
정하는 소비자에게는 소속 검사가 필요하고 문장은 소속 검사가 아니기 때문입니다. 2026-08-26부터는 거부가
하나뿐입니다. 거기서 거부되던 유일한 경로인 block 경로가 제거되었으므로 `paths_refused_on_oblique`는 비어
있고, 문장은 `factorization='li'`만 설명합니다.

세 번째 목록 `normal_methods_refused_on_oblique`는 **비어 있으며**, 지우지 않고 남겨 둡니다 — 2026-08-21에
네 가지 법선장 구성이 모두 셀 계량을 갖추기 전까지는 항목이 세 개 있었습니다. 이 키는 소비자가 이미 묻는
소속 검사이며, 빈 답도 답입니다. 이 키를 없애면 "이 축에서는 아무것도 거부되지 않는다"가 "이 소비자는 이
축보다 오래된 문서와 이야기하고 있다"로 바뀝니다.

한 사실의 두 표기는 서로 어긋나기 마련이고, 소비자가 따라 행동하는 것은 기계가 읽는 표기입니다 — 따라서
어긋나면 문서는 맞게 읽히면서 동작은 틀리게 됩니다. 그것을 막는 것이
`test_the_named_and_described_skewed_refusals_agree`입니다.

## 기계가 읽을 수 있어야 했던 사실 하나 (2026-08-20)

`material_quantity`는 재료 숫자가 *무엇인지* 말합니다. `quantity: 'permittivity'`, 상대값, 복소수, 손실은
`Im(eps) > 0`, 그것을 적는 타입 `Material(epsilon, mu=None)`, 스칼라 약식, 두 변환 보조 함수, 그리고 모든 옛
표기에서 대체 표기로 가는 `retired` 맵입니다.

이것이 `summary` 안의 문장이 아니라 dict인 데는 이유가 하나 있습니다. 답이 **바뀌었기** 때문입니다. 이
날짜까지 모든 재료 칸은 굴절률을 받았습니다. 옛 답을 배운 채 그에 맞춰 장면을 계속 생성하는 소비자는 여전히
*합법적인* 숫자를 계속 내보냅니다 — `3.5`는 멀쩡한 유전율입니다 — 그래서 아무것도 예외를 일으키지 않고,
아무것도 경고하지 않으며, 에너지도 보존됩니다. 문장은 테스트할 수 없지만 소속은 테스트할 수 있으므로,
소비자는 키에 대해 단언하고 다른 구조를 조용히 푸는 대신 실행을 거부할 수 있습니다.

`retired`는 솔버 자신의 거부를 생성하는 바로 그 표(`_core/domain.RETIRED_MATERIAL_NAMES`)에서 *도출되며*,
그 옆에 다시 적은 것이 아닙니다. 그렇기 때문에 약속이 사본이 아니라 목록 자체에 대해 참으로 유지됩니다.
위의 "두 번 기술되는 제약"과 같은 논리를, 두 번째 표기가 어긋날 기회를 갖기 전에 적용한 것입니다.
2026-08-26부터는 `retired` 맵이 없습니다. 퇴역한 표기는 이름으로 거부되지 않고 삭제되었으며, 표도 함께
사라졌고, 키는 버전을 올리지 않고 제거되었습니다.

`MANIFEST_VERSION`은 이 일로 `4` -> `5`가 되었고, 버전을 올린 것 자체가 요점입니다. 이름 변경만으로는
버전을 올리지 않았을 것입니다. 모르는 키를 읽는 소비자는 모르는 것을 무시하고 큰 탈 없이 동작하기 때문입니다. 알아볼 수 있는
모양 아래에서 *물리량*이 바뀌면 그렇지 않습니다.

`6`(2026-08-20)은 같은 규칙을 한 단계 아래에 적용한 것이지 예외가 아닙니다. 릴리스 전 이름 변경 묶음은
`results.incidences.fields`의 값을 바꾸었습니다(`('res', 'res_te_inc',
'res_tm_inc')` to `('given', 's_inc', 'p_inc')`). 이것들은 소비자가 건너뛸 수 있는 키 이름이
아닙니다 — 소비자가 결과 객체를 색인할 때 쓰는 속성 이름이며, 겉보기에는 전혀 바뀌지 않은 키 아래에
공개됩니다. 내용은 모양을 유지한 채 더 이상 해석되지 않게 되었고, 이는 `material_quantity` 실패의 축소판입니다.

## 셋을 공개한 뒤 두 경로로 (2026-09-03)

`paths`에는 `raster`, `smooth`, `boundary`의 세 항목이 있었는데, 뒤의 둘은 한 경로를 두 번 공개한
것이었습니다. 둘의 기계가 읽는 키는 모두 같았고, 같은 `structure` 표기를 받으며, 둘 사이의 선택은 형상의
좌표로 **층마다** 이루어지므로 한 구조가 동시에 둘 다를 차지할 수 있습니다. 형제 항목 두 개로는 이를 말할 수
없고, 테스트 모음도 차이를 볼 수 없었습니다. 둘을 구별하던 테스트 두 개는 모두 `modeling_type`을
비교했는데, 그 값이 둘 다 `2`였습니다.

이제 이들은 `overlap` 속성을 가진 `parametric`이며, `raster`는 그대로입니다. 이 속성도 여기의 다른 모든
주장처럼 런타임과 대조됩니다. 겹치는 표기가 같은 경로를 선택하는지, 섞인 구조가 한 층은 풀고 다른 층은
풀지 않는지, 그리고 `boundary` 항목이 담던 옵션 묶음이 겹치는 층에서도 여전히 실행되는지입니다.

`smooth`는 `NormalMethod.SMOOTH`와도 겹쳤는데, 공개된 JSON이 이를 분명히 보여 주었습니다.
`paths[0].raster_normal_method`(**raster**의 옵션)에는 `'smooth'`가 들어 있었고, `paths[1].path`는
`'smooth'` *그 자체*였습니다. 겹치는 경로에서 `raster_normal_method='smooth'`를 요청한 호출자는 "smooth
형상 경로는 기하에서 직접 법선장을 만든다"는 말을 들었습니다 — 한 문장 안에서 한 단어가 두 가지를 뜻했고,
호출자가 있지도 않은 경로를 가리켰습니다.

**버전 올림, 그리고 어떤 논거가 그것을 지탱하는가.** `MANIFEST_VERSION`은 `8` -> `9`가 되었지만, 소비자가
분기에 쓰는 값이 바뀌었기 때문은 아닙니다. `path`는 이름표일 뿐 솔버에 되먹이는 값이 아니며 — `path=`
키워드는 없습니다 — 따라서 더 이상 일치하지 않는 분기는 엉뚱한 장면을 푸는 대신 다음으로 넘어갑니다. 이는
"모르는 키를 읽는 소비자는 큰 탈 없이 동작한다"는 경우이며, 그것만으로는 버전을 올릴 이유가 되지
않습니다.

버전을 올리게 만든 것은 `('smooth', 'boundary')`를 담고 있던 `shape_coordinates.applies_to`입니다. 이것은
**이 이름들을 키로 하는 소속 검사**이고, 그것이 가르는 것은 좌표 관례입니다. 자기 소스에 `'smooth'`를
하드코딩한 소비자는 이제 `False`를 받고, 절대 길이와 셀 중심 원점이 적용되지 않는다고 결론 내려 비율이나
모서리 기준 중심을 내보냅니다. 그렇게 만든 장면은 풀리고 에너지도 보존합니다 — 정확히 `'8'`의 실패이며,
`'8'`이 다루던 바로 그 관례에서 일어납니다. 교훈은 일반화됩니다. 경로 이름이 *문서의 다른 곳에서* 키로
쓰이면, 그 이름을 바꾸는 것은 단순한 이름 변경이 아닙니다.

## 거부는 로드맵이 아니라 사실입니다

`materials[].supported`는 솔버 자신의 답입니다. `false`는 오늘 실행할 수 없다는 뜻이고, `why`는 무엇이
바뀌어야 하는지 말합니다 — 패턴 있는 이방성 층을 위한 새 인수분해, `mu != 1`을 위한 두 번째 자리,
`eps_xz`를 위한 1차 솔버 같은 것입니다. 이를 공개하는 것은 의도된 일입니다. "지원하지 않음"과 "철자를
틀렸음"을 구별할 수 없는 소비자는 사용자에게 자기 입력을 디버그하라고 보낼 것이기 때문입니다.

지켜볼 항목은 `z_coupled`이며, 이미 한 번 뒤집힌 적이 있습니다. 예전에는 존재하지만 닿을 수 없는 작업을
설명했습니다 — 1차 연산자는 작성되고 고정되었지만 거기로 분기하는 것이 없었습니다. 지금은
`supported: true`로 읽히고, 그 `why`에는 함께 따르는 두 한계(`layer_algorithm`은 `ETM`이어야 하고, `mu`는
없음)가 담겨 있습니다 — 그리고 그 뒤로도 한 번 더 고쳐졌는데, 필드 재구성이 들어오면서 한계 중 하나가 더
이상 아니게 되었을 때입니다. 2026-09-24부터는 어느 한계도 지명하지 않습니다. ETM과 SMM은 둘 다 공개
선택지이고, `mu`도 다루며, `patterned_z_coupled`를 포함한 모든 `materials[]` 항목이 `supported: true`로
읽힙니다. 이런 항목들이 갖추려는 모양이 바로 이것입니다. 기능이 들어오는 날 조용히 낡아 버리는 `false`가
아니라, 가장자리를 밝혀 둔 `true`입니다.
