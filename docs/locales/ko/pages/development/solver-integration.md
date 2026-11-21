# RCWA·FDTD 호환 통합 중간 점검

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 필요합니다.
> 이 네임스페이스는 현재 설치본에 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행하지 않았습니다.


작성: 2026-10-13.
대상: `src/wavebit/fdtd/_periodic/api.py`, `src/wavebit/fdtd/_periodic/adapter.py`, `src/wavebit/fdtd/_periodic/scene.py`, `src/wavebit/fdtd/_periodic/legacy.py`, `tests/cross_solver/test_solver_integration.py`, `studies/_support/_fdtd_legacy.py`

이 중간 점검은 develop 커밋 `db37dbb8722b8993c2486c6af4f491e3b1b8bc13`의 RCWA 구현과
fdtd-general 커밋 `d1119445bb2839dd03d2af0b769a2797d0be2d13`의 FDTD 구현을 합칩니다. 패키지는 이제
`src/wavebit/` 아래에 있습니다. 주기 FDTD 어댑터는 자신의 수치 커널을 현재 RCWA의 장면, 좌표, 산란
규약으로 옮겨 줍니다.

이 페이지는 새 공개 생애주기 이전에 정한 주기 구조 호환 경계를 기록합니다. 유지된 진입점은
`wavebit.compat.legacy.rcwa(...)`, `wavebit.compat.legacy.fdtd(...)`, `wavebit.compat.legacy.fd`입니다.
아래 절의 이름과 단위는 그 호환 인터페이스를 기술합니다. 코드 조각에서 `wavebit`을 짧은 별칭으로
쓰면 `from wavebit.compat import legacy as wavebit`을 뜻합니다.

새 `wavebit.rcwa`와 `wavebit.fdtd` 모듈은 이제 공통 물리 재료와 check/prepare/run 생애주기를
제공합니다. [현재 API 안내](../getting-started/index.md)와
[구현 중간 점검](../design/unified-api.md#13-implementation-checkpoint-and-acceptance-evidence)부터
보세요. 특히 아래에 기술한 호환 필드 규약은 새 API의 SI 자기장 규약이 아닙니다. 유지된 주기 구조와
일반 FDTD의 호환 재료 선언도 서로 다릅니다.

## 호환 계약

| 계약 | 호환 주기 구조의 동작 |
| --- | --- |
| 진입점 | `rcwa(...)`는 솔버를 돌려주고 `.solve()`가 그 결과를 돌려줍니다. `fdtd(...)`는 결과를 바로 돌려줍니다. |
| 백엔드 | `'numpy'`, `'jax'`, `'torch'` 또는 `wavebit.Backend` 멤버. 정수는 거부합니다. FDTD 백엔드를 생략하면 legacy NumPy 커널을, 명시적 `'numpy'`는 공통 장면 엔진을 선택합니다. |
| 장면 | 공통 `Stack`과 `Incidence`는 각 솔버가 지원하는 부분 집합 안에서 동작합니다. 층 순서로 인덱싱한 장면 배열은 모두 아래에서 위로입니다. |
| 좌표 | 아래 계면 `z=0`, 위 `z=sum(thickness)`, +z는 위쪽. 입사 쪽이 바뀌어도 물리적 `top`/`bot` 매질과 층 순서는 고정입니다. |
| 입사 | 두 주기 경로 모두 위와 아래 입사를 지원합니다. 각도는 라디안입니다. legacy theta는 부호를 가질 수 있고, 명시적 백엔드는 `0 <= theta < pi/2`를 요구합니다. 카르테시안 `kx`/`ky` 입사는 주기 FDTD에서 명시적으로 지원하지 않습니다. |
| 편광 | TE/TM 문자열은 대소문자를 구분하지 않으며, 엔진 자체의 `Tetm` 열거형(`wavebit._common.enums.Tetm`; 공개 `wavebit.Tetm`은 없음) 멤버도 받습니다. 복소 Jones 쌍은 정규화 뒤에도 전체 위상을 보존합니다. |
| 산란 | `result.incident`는 요청한 상태입니다. 경로가 계산하는 경우 단위 s/p 열을 제공합니다. 순수 legacy TE/TM은 `s_inc`/`p_inc`를 비워 둘 수 있습니다. 진폭은 셀 중심 기준이고 각 차수 자신의 s/p 좌표계를 씁니다. 반사는 빛이 들어오는 면, 투과는 반대 면을 가리킵니다. |
| 결과의 한계 | 주기 FDTD는 R/T, 복소 s/p 진폭, 차수 라벨을 제공하지만, 카르테시안 `r_xyz`/`t_xyz`나 RCWA 재구성 상태는 없습니다. 바깥 매질이 양의 실수이면 `power_entering=1-sum(R)`과, 나가는 투과로부터 적층 전체의 흡수를 얻을 수 있습니다. |
| 진단용 각도 | `angles`는 파장에 따라 달라지는 극각(라디안)이고, `azimuth`는 phi입니다. legacy 호환 키 `angles_deg`는 같은 극각을 도 단위로 담습니다. 광대역 실행은 물리적 kx와 ky를 고정합니다. |

native 호환 `fdtd_grid` 좌표와 일반 `fd.Scene` 입력은 이 주기 어댑터를 거치지 않습니다. 주기
구조 예제를 다시 쓰기 전에 각자의 가이드를 읽으세요. 지원되는 주기 구조의 재료, 형상, 실행,
미분 옵션은 [시간 영역 가이드](../guide/fdtd/time-domain.md)에 정해져 있습니다.

## 필드 진단값은 서로 바꿔 쓸 수 있는 보정 데이터가 아니다

두 주기 경로 모두 공개 좌표로 필드를 돌려주고 H를 `Z0 H`로 저장합니다. legacy 주파수 모니터는
소프트 소스 펄스 스펙트럼으로 나누고, 명시적 백엔드 모니터는 각 파장을 빛이 들어오는 계면에서
단위 입사 E로 보정합니다. 반면 명시적 시간 기록은 중심 주파수의 상수 보정을 써서 인과적인
파형을 보존합니다. 서로 다른 경로의 절대 필드를 비교하기 전에 각 정규화 라벨을 읽고 그에 맞는
기준을 쓰세요.

`power_z`는 부호가 있는 실험실 좌표 플럭스입니다. 위쪽 반공간에서 오는 투과는 보통 z 플럭스가
음수이고, 아래에서 오는 투과는 양수입니다. 회절 T는 어느 쪽에서든 나가는 전력의 비율입니다. 성분
기록은 native Yee 좌표와 엇갈린 E/H 시각을 유지하므로, 각자의 `t/z/y/x` 축을 쓰세요. y 불변 기록은
`y=0`을 나타내며, 쓰이지 않는 두 번째 주기에 따른 인위적인 위상이 없습니다.

공개 메타데이터 `source_z`, `z_nodes`, `layer_planes`, `pml_bounds`를 쓰세요. 명시적 백엔드에서
`mesh_indices`와 `background`는 물리적인 위-다음-아래 반공간 순서를 따르고, `background`의 각 행은
`(epsilon, mu)`를 담습니다. 반면 층 순서로 인덱싱한 배열은 아래에서 위로입니다. legacy
`diagnostics['grid']`는 원래의 좌표 해석을 가진 내부 수치 격자 객체로 남아 있으며, 공개 필드 좌표
지도가 아닙니다. 공개 표본화·그리기 도우미는 변환된 메타데이터를 읽습니다.

## 과거 주기 구조 호출 코드 옮기기

1. 퇴역한 `result.given` 접근은 `result.incident`로, 정수 백엔드 선택자는 이름으로 바꾸세요. FDTD
   백엔드를 생략한 경우와 명시적 `'numpy'`의 구분은 유지하세요.
2. 예전의 위에서 아래 규약으로 작성한 장면이라면 층 축을 두께, 투자율, 분산 라벨, 그 밖의 층
   순서 데이터와 함께 뒤집으세요. 물리적 `top`과 `bot` 매질 이름은 유지하세요. 이미 현재 RCWA용으로
   작성한 장면은 뒤집지 마세요.
3. 예전의 아래쪽 방향 z 좌표는 `z_new = total_thickness - z_old`로 바꾸세요. 모니터 위치, 기록 범위,
   필드 그림을 함께 고치세요. 돌려받는 카르테시안 벡터에도 그에 맞는 극성 벡터 E와 축성 벡터 H의 반사가
   필요하며, 새 계산에서는 공개 어댑터가 이를 수행합니다.
4. R/T만이 아니라 복소 회절 진폭을 다시 확인하세요. 단일 입력 차수 진폭의 기준을 모서리에서 중심으로
   옮기면 패리티 `(-1)**(m+n)`가 생기고, 예전 고정 좌표계의 TE/TM 진폭과 음의 기울기 입사 상태는 기저
   부호 변경이 더 필요할 수 있습니다. 저장된 필드를 다시 재생하는 것은 모든 기록에 차수 패리티를 곱하는
   것과 같지 않습니다.
5. `src/`로 옮긴 뒤 체크아웃을 다시 설치하고 `wavebit.__file__`을 확인하세요.
   [소스 체크아웃과 디버깅](source-checkout.md)을 보세요.

보관된 수치 실험은 측정한 입력, 좌표계, 결과를 그대로 유지합니다. `studies/_support/_fdtd_legacy.py`는
그 실험들을 다시 재생하는 명시적 경계이며, 새 응용을 위한 공개 API가 아니고 새 공개 규약을 검증하지도
않습니다. 측정 기록 안의 과거 이름이나 부호는 새 검증인 것처럼 조용히 고쳐 쓰지 말고, 과거의 것임을
밝혀야 합니다.

## 회귀 검사

`tests/cross_solver/test_solver_integration.py`는 독립적인 관측량으로 공개 경계를 검사합니다.

| 검사 | 구별하는 것 |
| --- | --- |
| 비대칭 2층 적층, 서로 다른 바깥 매질, 양쪽 입사와 TE/TM | 대칭 진공 장면이라면 감춰질 층 순서나 포트 뒤바뀜 |
| 서로 다른 자성 바깥 매질, conical/교차 경로와 두 포트 | 올바른 kz/mu 어드미턴스를 쓴 전기장 진폭–전력 변환, Fresnel 전력과 대조 |
| 0이 아닌 회절 차수 | 전력에는 보이지 않는 복소 위상과 출사 기저 변화 |
| 부호 있는 legacy 기울기와 섞인 Jones 상태 | conical 입사를 포함한 입사 좌표계의 부호 |
| 진공 기록과 여섯 성분 해석적 3D 필드 | 공개 z 방향, E/H 반사, 가로 방향 위상 |
| y 불변 기록과 주파수 평면 | 쓰이지 않는 y 주기에 대한 의존 |
| 스칼라 TM과 그에 해당하는 Jones p 상태 | 필드 정규화와 그에 대응하는 모니터 플럭스 |
| 형식 있는 선언, 기본값, 지원하지 않는 카르테시안 입사 | 공통 입력 계약과 명시적 기능 경계 |
| JIT 재료 민감도와 유한 차분 | 어댑터를 지나도 살아 있는 JAX 재료 그래프의 보존 |

경계를 바꾼 뒤에는 이 검사들과 capability manifest를 실행하세요.

```sh
python -m pytest tests/cross_solver/test_solver_integration.py tests/api/test_manifest.py -q
```

이 묶음은 통합 규약을 시험합니다. 새로운 물리 영역에 대한 해상도, 실행 시간, PML, 외부 기준 연구를
대신하지 않습니다. 방법의 근거와 한계는 [연구용 알파 범위](../guide/fdtd/fdtd-alpha.md)에 있습니다.
