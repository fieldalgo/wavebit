# 현재 구현 구조와 수정 지점

현재의 불변 `Problem` → `check` → `prepare` → `run` API를 따라가며,
수정할 코드와 먼저 실행할 회귀 검사를 찾는 안내입니다.
[소스 체크아웃과 디버깅](source-checkout.md)에서 지금 수정하는 체크아웃을
인터프리터가 실제로 불러오는지 먼저 확인하세요.

공개 모듈은 `wavebit.rcwa`와 `wavebit.fdtd`입니다. 두 모듈 모두 입력과 결과를
변환하는 계층 뒤에서 기존 수치 구현을 재사용합니다. 내부 엔진을 재사용한다고 해서
그 엔진의 가변 객체나 정규화된 단위가 새 공개 API의 계약이 되는 것은 아닙니다.
가져온 [교재의 코드 지도](../learn/fundamentals/code-map.md)와
[주기 구조 통합 기록](solver-integration.md)은 호환 API의 흐름을 설명합니다.
새 응용 코드는 현재 공개 모듈을 사용합니다.

## 공통 선언과 단위

| 책임 | 먼저 볼 파일 | 유지해야 할 경계 |
| --- | --- | --- |
| 물리적 재료 응답 | [materials/models.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/materials/models.py), [materials/response.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/materials/response.py) | 선언된 물리 모델을 평가합니다. 상대 유전율을 굴절률로 해석하지 않습니다. |
| 기하 단위와 SI 변환 | [units.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/units.py), [materials/_lower.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/materials/_lower.py) | 길이는 `Units`를 따릅니다. 초, Hz/rad/s, S/m는 FDTD 내부 표현으로 내릴 때 한 번만 변환합니다. 표 데이터를 인과적 시간 영역 모델로 자동 피팅하지 않습니다. |
| 불변 값 교체, 실행 설정과 오류 | [_common/declarations.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/_common/declarations.py), [_common/execution.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/_common/execution.py), [errors.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/errors.py) | 물리 입력, 수치 설정, 실행 설정을 분리합니다. 모델을 조용히 바꾸는 대신 구조화된 오류를 유지합니다. |
| 이름 있는 관측량과 유효성 | [observations/contracts.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/observations/contracts.py) | 스펙트럼 축, 단위, 보정 기준, 실패한 유효성 검사를 보존합니다. |
| 재현 가능한 입력 선언 | [jobs.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/jobs.py) | Job은 지원하는 물리 입력과 실행 정책을 직렬화합니다. 솔버의 마지막 가변 상태를 저장하는 객체가 아닙니다. |

실행 가능한 Job과 계산 기록의 책임 경계는 다릅니다.
[records.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/records.py)는 실행,
아카이브 완료 처리와 읽기를 관리하고,
[_common/record_io.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/_common/record_io.py)는
수치 관측량을 인코딩하며,
[_common/provenance.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/_common/provenance.py)는
설치된 패키지의 식별 정보를 수집합니다. [계산 기록 가이드](../guide/run-records.md)를 참고하세요.
기록을 읽으면 데이터와 메타데이터가 복원되며, 솔버를 만들거나 보관된 소스를 실행하지 않습니다.

## RCWA: 물리 문제에서 스펙트럼 결과까지

단일 평면파를 기준으로 다음 순서대로 읽으세요. 결맞는 빔과 내부 소스는
아래의 소스 조합 경로로 분기합니다.

| 단계 | 구현 | 확인할 내용 |
| --- | --- | --- |
| 선언 | [rcwa/problem.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/problem.py), [rcwa/numerics.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/numerics.py) | 아래에서 위로 쌓는 층, 입사 조건, 요청한 장·산란 결과, 수치 설정을 확인합니다. |
| 검사와 준비 | [rcwa/_lifecycle.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_lifecycle.py) | `_resolve`, `prepare`, `Prepared._solvers`가 입력을 검사하고 기하를 고정하며 호환되지 않는 재사용을 거부합니다. `check`는 산란 계산을 실행하지 않습니다. |
| 실행 시점의 값 변환 | [rcwa/_adapt.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_adapt.py) | `build_solver`가 각 파장에서 재료를 평가하고 현재 배열을 내부 팩터리에 전달합니다. 미분 연결을 끊는 호스트 배열 변환을 피합니다. |
| 행렬 구성과 계산 | [rcwa/_engine/engine.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/engine.py), [parametric.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/parametric.py), [factorization.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/factorization.py) | 입력 분기, 고정한 기하의 검사, 재료 행렬과 계산 순서를 관리합니다. 입구에 예외 처리를 하나 더 넣기 전에 선택된 기하·인수분해 경로를 따라가세요. |
| 층 내부 전파와 층 연결 | [solve_1d.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/solve_1d.py), [solve_2d.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/solve_2d.py), [scattering.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/scattering.py) | 모드 연산, 전파와 계면 연결입니다. 이 부분을 바꾸면 백엔드 간 일치뿐 아니라 독립적인 물리 검증이 필요합니다. |
| 결맞는 소스 조합 | [illumination.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/illumination.py) | 소스 그룹 구성과 경계 전력을 확인합니다. 현재 eager 실행 경로이며, 일반적인 결맞는 장에 단일 입사 포트의 R/T를 임의로 부여하지 않습니다. |
| 결과 구성과 장 표본화 | [rcwa/results.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/results.py), [fields.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/fields.py) | `pack`이 파장 축과 각 결과가 소유하는 장 복원 상태를 보존합니다. 공개 E/H 단위는 V/m와 A/m입니다. |

준비한 RCWA 객체는 구조 제약과 고정된 기하를 보유합니다. 다음 호출의 입력을
직전 문제로 대체해 기억하지 않습니다. `Prepared.run(problem)`에서 `_solvers`,
선택된 엔진 계산, `pack`까지 따라가세요. 후속 계산이 이전 결과의 장을 바꾸면
안 됩니다. 지원하는 실행 시점 값과 고정 값의 경계는
[수명 주기](../guide/common/lifecycle.md), 변환 지원과 한계는
[미분](../guide/common/autodiff.md)에 설명되어 있습니다.

## FDTD: 물리 영역에서 시간 적분과 관측까지

| 단계 | 구현 | 확인할 내용 |
| --- | --- | --- |
| 선언 | [fdtd/_problem.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_problem.py), [fdtd/_numerics.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_numerics.py) | 물리 영역, 구조, 소스, 이름 있는 관측량을 선언합니다. 격자, 실행 시간과 PML은 수치 설정입니다. |
| 검사와 준비 | [fdtd/_api.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_api.py) | 공개 `check`, `prepare`, `Prepared._lower`를 확인합니다. 재사용은 선언 타입뿐 아니라 실제로 결정된 격자와 시간 간격 구조에도 의존합니다. |
| 물리 입력의 내부 표현 변환 | [fdtd/_lower_public.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_lower_public.py) | `lower_problem`이 단위를 변환하고 수치적 패딩을 물리 영역 바깥에 둡니다. 전류 모멘트, 펄스 폭, 소스 위치의 물리적 의미를 유지해야 합니다. |
| 수치 데이터 준비 | [fdtd/_prepare/plan.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_prepare/plan.py), [fdtd/_prepare/__init__.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_prepare/__init__.py) | 고정된 계획과 실행 시점의 계수를 분리합니다. 재료 지원 검사, 점유율, 소스와 모니터는 이 디렉터리의 개별 모듈이 담당합니다. |
| 장의 시간 적분 | [fdtd/_execution.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_execution.py), [fdtd/_runtime.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_runtime.py), [fdtd/_kernel/step.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_kernel/step.py) | 실행 계층이 계수를 만들고 선택한 런타임으로 Yee 상태를 갱신합니다. 서로 엇갈린 장의 시간 위치, 극점·PML 메모리, 관측 순서를 보존합니다. |
| 결과 조합과 변환 | [fdtd/_assemble.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_assemble.py), [fdtd/_result.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/fdtd/_result.py) | `adapt_result`가 초, 물리 좌표, SI 장 단위로 결과를 제공합니다. 원시 스펙트럼 진폭을 효율로 쓰려면 선언된 기준 계산으로 정규화해야 합니다. |

현재 공개 FDTD 경로는 CPU와 배정밀도만 허용합니다. 검사 성공은 입력 허용 여부와
추정치를 알려줄 뿐, 장을 시간 적분하거나 공간·시간·흡수층 수렴을 입증하지 않습니다.
[FDTD 가이드](../guide/fdtd/general-fdtd.md)에 지원하는 소스, 재료, 관측량을 설명합니다.
`_periodic` 아래의 이전 주기 FDTD 어댑터에는 별도의 좌표·정규화 계약이 있습니다.
두 경로가 모두 회절 결과를 낸다는 이유만으로 새 물리 영역 API의 변경을 그쪽에
연결하지 마세요.

## 변경 내용에 따라 첫 회귀 검사 선택하기

다음 검사는 출발점입니다. 작은 테스트 파일 하나가 영향받는 모든 물리 영역을
포괄한다는 뜻은 아닙니다. 수정 중에는 좁은 사례를 먼저 확인한 뒤 관련 파일 전체와
적절한 검증 범위를 실행하세요. 파일을 직접 지정하면 그 파일의 slow·research
사례도 실행하므로 긴 계산을 시작하기 전에 범위를 확인해야 합니다.

| 변경 내용 | 먼저 실행할 검사 | 구분해 주는 문제 |
| --- | --- | --- |
| 재료 응답 또는 SI 변환 | [test_common_material_models.py](https://github.com/fieldalgo/wavebit/blob/main/tests/materials/test_common_material_models.py), [test_lifecycle.py](https://github.com/fieldalgo/wavebit/blob/main/tests/fdtd/test_lifecycle.py) | 다른 기하 단위로 표현한 동일 문제, SI 전도도·극점, 지원하는 인과적 재료와 미분 연결을 확인합니다. |
| RCWA 준비, 실행 시점 값 또는 결과 소유권 | [test_unified_rcwa.py](https://github.com/fieldalgo/wavebit/blob/main/tests/api/test_unified_rcwa.py) | 재사용과 새 준비의 일치, 독립 Fresnel 장·전력, 변환 중 잘못된 입력, 이전 결과의 독립성을 검사합니다. |
| 결맞는 소스와 절대 전력 | [test_unified_rcwa_sources.py](https://github.com/fieldalgo/wavebit/blob/main/tests/api/test_unified_rcwa_sources.py) | SI 소스 세기, 유한한 관측 영역과 독립적으로 평가한 장을 확인합니다. |
| FDTD 영역, 소스·모니터 또는 재사용 | [test_lifecycle.py](https://github.com/fieldalgo/wavebit/blob/main/tests/fdtd/test_lifecycle.py) | 물리 영역 바깥 PML, 고정된 실험실 좌표, 소스·재료 값 재사용, 실제 실행 시간 제약을 확인합니다. |
| RCWA 모드 또는 장 복원 연산 | [test_smm_field_state.py](https://github.com/fieldalgo/wavebit/blob/main/tests/rcwa/test_smm_field_state.py), [test_lossy_media.py](https://github.com/fieldalgo/wavebit/blob/main/tests/rcwa/test_lossy_media.py) | 솔버 golden을 다시 만드는 대신 독립적인 내부 장과 계면 응답으로 확인합니다. |
| FDTD 도파관 모드 | [test_fd_waveguide_modes.py](https://github.com/fieldalgo/wavebit/blob/main/tests/fdtd/test_fd_waveguide_modes.py) | 이산 Yee 계의 모드를 닫힌 식, 정확한 평판 분산, 출판된 모서리 기준값과 비교하고, 풀어낸 모드를 시간 전진 자체로 진행시켜 확인합니다. |
| FDTD 시간 적분 또는 구성식 갱신 | [test_fd_stage3_backends.py](https://github.com/fieldalgo/wavebit/blob/main/tests/fdtd/test_fd_stage3_backends.py), [test_fd_constitutive_absorber.py](https://github.com/fieldalgo/wavebit/blob/main/tests/fdtd/test_fd_constitutive_absorber.py) | 관측 처리만 바꿀 때 원시 장이 그대로인지, 이산 에너지·일 관계와 백엔드 결과가 일치하는지 검사합니다. |
| 솔버 간 위상, 포트 또는 회절 정규화 | [test_public_lifecycle_diffraction.py](https://github.com/fieldalgo/wavebit/blob/main/tests/cross_solver/test_public_lifecycle_diffraction.py) | 현재 공개 API, 비대칭 구조와 복소수 비영차 회절을 확인합니다. 총 전력 일치만으로는 위상 오류가 드러나지 않을 수 있습니다. |
| 결과 기록과 소스 식별 | [test_run_records.py](https://github.com/fieldalgo/wavebit/blob/main/tests/api/test_run_records.py) | 배열·단위의 정확한 보존, 손상되거나 미완료인 아카이브 거부, 명시적인 입력 연결, 계산에 쓴 백엔드 없이 읽기를 확인합니다. |
| Job 직렬화 | [test_unified_jobs.py](https://github.com/fieldalgo/wavebit/blob/main/tests/api/test_unified_jobs.py) | 물리 선언, 단위, dtype·shape와 지원하지 않는 직렬화 입력을 확인합니다. |
| 거부 코드 또는 `orders=` 창 | [test_error_codes.py](https://github.com/fieldalgo/wavebit/blob/main/tests/api/test_error_codes.py) | 모든 공개 거부의 단일 `area.item` 코드, 바뀐 선언을 가리키는 `prepared.structure`, check와 같은 코드로 발생하는 엔진 거부, 모든 곳에서 명시하는 `(Mu, Mv)` 쌍을 확인합니다. |

예를 들어 다음 두 검사는 준비 작업이 시간 적분·산란 계산과 분리되어 있는지 확인합니다.

```sh
python -m pytest tests/api/test_unified_rcwa.py::test_check_does_not_run_scattering -q
python -m pytest tests/fdtd/test_lifecycle.py::test_check_never_advances_fields -q
```

기록을 남기는 quick/full/research 실행과 엄격한 이어 실행은
[검증 도구](verification.md)를 사용하세요. 기본·slow·research 선택 규칙은
[테스트 가이드](https://github.com/fieldalgo/wavebit/blob/main/tests/README.md)를 참고하세요.
수치 변경으로 기준 결과가 달라지면 허용 오차를 바꾸거나 근거 데이터를 다시 만들기 전에
규약, 연산, 수렴의 차이를 먼저 진단하세요. 과거 기록은 그대로 보존하고
새 계산은 새 기록으로 남깁니다.
