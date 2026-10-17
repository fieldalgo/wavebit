# 범위를 한정한 일반 장면 CUDA 실행

Written: 2026-10-18

Targets:

- `src/wavebit/fdtd/_api.py`
- `src/wavebit/fdtd/sources.py`
- `src/wavebit/fdtd/numerics.py`
- `src/wavebit/fdtd/_runtime.py`
- `src/wavebit/fdtd/_kernel/observe.py`
- `tests/fdtd/test_general_cuda.py`
- `tests/fdtd/test_general_cuda_observations.py`
- `tests/fdtd/test_general_cuda_checkpoint.py`
- `tests/fdtd/test_general_cuda_airy.py`
- `tests/fdtd/test_general_cuda_higher_order.py`
- `tests/fdtd/test_general_cuda_dipole.py`
- `tests/fdtd/test_general_cuda_dipole_conductive.py`
- `tests/fdtd/test_general_cuda_dipole_radiation.py`
- `tests/fdtd/test_fd_stage3_backends.py`
- `tests/api/test_fdtd_checkpoint_jobs.py`
- `studies/fdtd/general_cuda/checkpoint_cost.py`
- `studies/fdtd/general_cuda/monitor_cost.py`

일반 `wavebit.fdtd` API는 제한된 Torch CUDA 배정밀도 실행을 허용합니다.
계획 수립은 호스트에서 수행하고 기존 물리적 입력 검사를 유지하며, 시간
진행과 미분 가능한 관측은 CUDA에서 실행합니다. 일반 장면·관측 로컬
초기 검증 35사례는 CPU에서 가능한 입력 검사와 실제 CUDA 실행을
포함합니다. 모든 일반 FDTD 기능이나 연속체 수렴을 검증한 것은 아닙니다.
아래 체크포인트 연구는 전도체 한 사례의 비용을 측정하며 일반 장면 전체의
성능을 대표하지 않습니다.

## 허용 입력

`check`, `prepare`, `run`에 `backend='torch', device='cuda:0'`를 전달합니다.
prepared 실행은 장치를 유지하며 재사용할 때마다 새 입력을 검사합니다.
입력은 CPU에 있어도 되며 미분 가능한 장치 변환은 원래 리프와의 기울기
연결을 유지합니다. 다른 비CPU 백엔드와 단정밀도는 계속 거부합니다.

| 부분 | CUDA 허용 범위 |
| --- | --- |
| 격자와 실행 길이 | 기존 균일 정육면체 격자; 고정 실행 길이; `discretisation='voxel'`; `geometry_ranges` 없음 |
| 경계와 padding | `Periodic`, `Open` 면; `PML` padding |
| 재료 | 기존 비분산 `Material`, `ConductiveMaterial`과 해당 재료 검사 |
| 구조 | 셀에 스냅하는 `Slab`만 허용 |
| 광원 | `PlaneWave`, 고정 중심의 전기·자기 `Dipole`(`bspline` 또는 `tent` stamp), 초기 장 |
| 관측 | `Record`, `Field`, `Flux`, 최종 장 |

초기 E/H 패턴은 고정된 호스트 선언입니다. NumPy 배열, `requires_grad`가
없는 CPU 텐서, 호스트 callable을 사용합니다. CUDA 텐서 패턴과 미분 가능한
텐서 패턴은 허용하지 않습니다. `InitialField.amplitude`는 live 입력이며
CPU에서 CUDA로 변환해도 기울기가 연결됩니다. prepared 재사용에도 이 구분이
적용되며 CUDA prepared 계약은 범위 지정 기하를 제공한다고 표시하지 않습니다.

`Dipole.current_moment`는 전기 광원의 A·m 또는 등가 자기 광원의 V·m 단위
전류 모멘트입니다. 세기 파라미터는 `current_moment`이며,
`PlaneWave.amplitude`는 V/m 단위의 전기장 진폭입니다. 전류 모멘트·phase·
delay는 live 미분 그래프를 유지합니다. Dipole 중심과 stamp는 고정값이며
바꾸려면 다시 준비해야 합니다. 위치 미분, 주기 경계를 가로지르는 지지 영역,
PML 안의 지지 영역은 기존 광원 검사에서 계속 거부합니다.

`Decay`, 다른 흡수층인 `Absorber`, 범위 지정·스무딩 기하, 분산 재료,
Bloch·벽·대칭·불변축 경계, TFSF 광원, 회절·원거리장 관측은 이
CUDA 범위 밖입니다. 지원하지 않는 조합은 계획 수립 전에 해당 공개 필드
경로의 `cuda_scope` 이슈를 반환합니다. 입력 허용이 표본화, CFL, 재료,
padding 등 과학적 검사를 면제하지 않습니다. `check`나 `prepare` 통과는
장을 시간 진행했다는 뜻이 아닙니다.

## 근거와 한계

위 35사례는 기존 일반 장면·관측 파일의 합계입니다. 체크포인트 21사례와
Job 네 사례를 포함한 통합 실행은 60사례가 모두 통과했습니다. 별도의
기본 CPU/JAX 수명주기·통합 Job·마이그레이션 회귀 검사는 53사례가 통과했습니다.

| 검사 | 독립 기준 또는 장치 비교 |
| --- | --- |
| 주기 상자의 균일 전도체, 20단계 | E와 시간 기록의 정확한 SI 전도도 갱신식, 변하지 않는 H; 상대 `2e-14` |
| 전도도와 초기 진폭 미분 | 갱신식의 닫힌형 미분과 상대 `2e-12`; CPU와 CUDA 파라미터 리프 모두 검사 |
| CPML, 평면파, 장 기록·푸리에 장·플럭스를 포함한 슬래브 | 기존 최대값 정규화 무한대 노름 기준 `5e-13`으로 NumPy CPU 비교; prepared 재사용, 바뀐 입력, 이전 결과 보존 |
| 슬래브 유전율과 광원 미분 | NumPy 재료 중앙차분과 상대 `1e-6`, 절대 `1e-8`; 정확한 광원 이차 항등식과 상대 `2e-12` |
| 161단계 Record, Field, Flux 목적함수 | CPU/CUDA 값과 상대 `5e-13`; 광원 이차 항등식과 `2e-11`; 두 간격의 영이 아닌 재료 중앙차분과 `2e-6` |
| x·y·z 법선 슬래브, 열린 3D 펄스, 전도도가 있거나 없는 대각 epsilon·mu | 최대값 정규화 `5e-13`으로 CPU 장 비교; 방향 오차가 상쇄로 숨지 않도록 영이 아닌 상자 플럭스 여섯 면을 개별 비교 |
| 서로 다른 장치의 재료·광원 리프 | Epsilon/mu/conductivity 값과 미분은 CPU와 일치; CPU phase와 CUDA delay 미분은 두 간격의 영이 아닌 NumPy 중앙차분과 상대 `2e-6`으로 일치; 리프 장치 소유권 유지 |
| 고정 초기 패턴과 정밀도 | prepared CPU 텐서 패턴을 수정하면 재사용 무효; 미분 가능한 패턴 거부; float32 호스트 패턴은 정밀도 경고를 유지하며 실행 장은 float64, 입력은 보존하고 진폭 기울기는 원래 float32 CPU 리프로 반환 |
| 입력과 재사용 제어 | 지원 밖 범위는 계획 전에 거부; 준비는 장을 진행하지 않음; 호환되지 않거나 잘못된 재사용 거부; 잘못된 미분 가능 재료는 실패한 validity와 NaN을 유지 |

관측 목적함수는 고정된 SI 스케일 계수를 사용합니다. 실행 장의 노름으로
정규화하거나 장에 의존하는 값을 미분 그래프에서 분리하지 않습니다.
동일한 이산 문제와 그 미분을 검증한 것으로, CPU 일치가 독립 광학 해는
아닙니다. 격자·유한 실행 길이·정밀도 경고는 그대로 표시하며 허용오차나
경고 정책을 완화하지 않았습니다. 축 순열, 열린 3D
펄스, 대각 재료 검사는 유한한 장치 검증 사례를 늘린 것이며 허용하는
기능 종류나 연속체 정확도 주장을 확장하지 않습니다.

[Native CUDA 기록](native-cuda.md)에 측정 Windows/Python/Torch/GPU 환경과
독립 환경 설치가 있습니다. 그 성능 표는 native 엔진의 결과이며 이 일반
장면 실행의 성능이 아닙니다. 일반 실행의 기본값은 기존의 단순 시간 스캔입니다.
아래 메모리 결과는 어떤 입력을 미분하는지에 따라 달라집니다. 보편적인
메모리 절감, 컴파일, 장시간 안정성, 여러 GPU를 검증했다고 주장하지 않습니다.

## 고정 중심 Dipole 근거

이후 Dipole 추가 검증은 별도 실행에서 CPU 42사례와 실제 CUDA 46사례가
통과했으며 기존 CUDA 거부 대조군 아홉 사례도 통과했습니다. 앞의 60사례
실행과 별개의 기록입니다. 고정 실행 길이·배정밀도·voxel·재료·Slab·경계·
관측기의 기존 제한은 그대로 적용합니다.

| 검사 파일 | CPU / CUDA 통과 | 근거 |
| --- | ---: | --- |
| `test_general_cuda_dipole.py` | 23 / 27 | 전기·자기, 세 축, 두 stamp: 한 단계 SI 모멘트 상대 `1e-13`, 독립적으로 재구성한 Yee 무게중심 절대 `1e-13` um. 짧은 슬래브/PML 장·광원 이력·관측은 NumPy와 최대값 정규화 `5e-13`으로 일치하며 고정값·지지 영역 거부와 재사용을 검사. |
| `test_general_cuda_dipole_conductive.py` | 12 / 12 | 전기 Dipole: 세 축 × 두 stamp × 스칼라/대각 epsilon·mu, 스칼라 전도도 600 또는 2400 S/m. 독립식 `E1 = -dt*J/(epsilon0*epsilon + sigma*dt/2)`으로 같은 기준의 모멘트·무게중심을 복원하고 반 단계 SI 광원 일을 검사. |
| `test_general_cuda_dipole_radiation.py` | 7 / 7 | 진공 Gaussian bspline 광원: Larmor 에너지 세분화, 전기·자기 쌍대성, um/nm 기하 단위를 바꿔도 같은 SI 결과. |

기본 미분 사례는 CPU 전류 모멘트·delay와 CUDA phase·재료 리프를
사용합니다. 무차원 섭동 좌표에서 두 간격의 NumPy 중앙차분이 상대 `1e-7`
및 절대 `1e-10*loss`로 일치하며, 전류 모멘트의 이차 항등식은 상대
`1e-12`로 일치합니다. 41단계 체크포인트 사례는 블록 7·6단계 tail을
사용하고 광원 장 이력의 미분을 유지하며 새 실행·prepared 재사용에서 사용자
입력과 이전 이력을 보존합니다. 얇은 PML과 미완료 푸리에 시간창 경고는
의도한 것이며 짧은 장치 비교·미분 사례가 복사 정확도를 입증하지 않습니다.

별도 복사 사례는 폭 0.6파장의 진공 상자, 12셀 PML, 광원 8주기와 기본
bspline stamp를 사용합니다. 에너지는 광원의 delivered-work 이력을 시간
적분한 값이며 독립 SI Gaussian 전류 Larmor 값
`2.5026540826870543e-24` J와 비교합니다. 실제 CUDA 결과는 다음과 같습니다.

| 파장당 셀 수 | 전달 에너지 (J) | 상대 Larmor 오차 |
| --- | ---: | ---: |
| 14 | 2.3846422619180062e-24 | -0.0471546673531249 |
| 20 | 2.444240894665441e-24 | -0.02334049616593281 |
| 28 | 2.472762333282175e-24 | -0.01194401959570257 |

적합 차수는 `1.9810176952811915`, 2차 Richardson 상대 오차는
`-7.26898350460655e-5`였습니다. 기존 기준은 차수 1.9~2.1,
Richardson 절댓값 `1e-3` 미만, 거친 격자 오차 -0.06~-0.03입니다.
열린 상자의 자기/전기 에너지 비에서 1을 뺀 값은
`-2.7458726875040362e-5`로 기존 `1e-4` 기준을 통과했으며 주기 격자
쌍대성은 별도로 `1e-13`에서 검사합니다. 격자 경고는 그대로 표시합니다.
광원 일로 명시한 진공 bspline 복사 사례를 검증한 것이며 별도 원거리장이나
닫힌 표면 플럭스 정확도, tent·비진공 복사 정확도, 위치 미분 또는 임의의
광원 변환을 검증하지 않습니다.

## 공개 슬래브의 독립 검사

`test_general_cuda_airy.py`는 공개 `PlaneWave`, `Field`, `Flux` 인터페이스를
통한 독립 광학 검사입니다. CPU 세 사례와 실제 CUDA 세 사례로 총 여섯 사례가
모두 통과했습니다. 별도 실행이므로 위 60사례에는 포함하지 않습니다. 독립
광학 근거의 범위는 아래 평면 사례로 한정됩니다.
사용하지 않는 DFT 위상 생략과 no-grad 런타임 분기 이후에도 일반 CUDA 광학 세 사례와
별도 periodic 위·아래 입사 슬래브 두 사례가 모두 다시 통과했습니다.
기존 격자 경고는 그대로 표시했습니다.

사례는 진공 속 두께 0.2 um, epsilon 2.25 박막이며 횡방향 주기 경계,
열린 z 경계와 24셀 PML을 사용합니다. 파장 다섯 개는 0.80, 0.90, 1.00,
1.15, 1.30 um입니다. 슬래브·빈 영역 실행은 같은 광원과 관측점을 사용하며
제곱 전에 복소 입사 장을 빼고, 측정한 기준 장으로 광원 스펙트럼을 상쇄합니다.
기준 진폭은 대역 최대값의 0.1보다 커야 합니다. 이산 전파 위상으로 r/t의
기준면을 슬래브 면으로 이동합니다.

복소 r과 t는 기존 독립 이산 Yee Airy 방정식에 대해 최대 절대 `2e-6`,
상대 `1e-5` 오차 안에 있어야 합니다. Flux로 구한 R/T는 장 진폭의 제곱과
절대 `2e-5`로 일치해야 하며 앞·뒤 기준 플럭스에도 같은 기준을 적용합니다.
진공 박막 대조군은 반사가 정확히 0이고 이산 투과 위상은 상대 `1e-12`로
일치해야 합니다. 이 CUDA 경로에 맞춰 새로 정한 허용오차가 아닌 기존 기준입니다.

간격 0.05와 0.025 um 각각에서 푸리에 시간창을 광원 24주기에서 32주기로
늘려도 r/t 변화가 같은 이산 허용오차보다 작아야 합니다. 간격을 절반으로
줄이면 반 셀 보정한 연속체 반사 오차가 3.2~5배 감소해야 하며 거친 격자의
오차는 0.03보다 커야 합니다. 기존 계단 근사·분산 보정을 지우지 않고
검사합니다. 수직 입사의 무손실 박막 하나에 대한 검사이며 일반 기하,
분산, 장시간 안정성이나 광학 미분을 검증하지 않습니다.

## 선택적인 역방향 체크포인트

`Numerics(checkpoint_steps=K)`는 고정 실행 길이의 Torch CUDA에서 양의
정적 정수 블록 길이를 선택합니다. 기본값 `None`은 기존 스캔을 유지합니다.
명시적 체크포인트는 CPU 백엔드, 다른 장치, `Decay`에서 거부합니다. 잘못된
값, 불리언(불리언 텐서 포함), 미분·변환을 거치는 블록 길이는 계획 전에
거부하며 기존 CUDA 기능 범위도 그대로 적용합니다.

역방향 미분 중 블록을 재계산하는 정책이며 메모리 상한은 아닙니다. 관측과
광원 이력은 계속 보존합니다. 블록이 일반 장면의 전체 실행보다 길면 일반
tail만 남습니다. 한 블록으로 처리하는 native 실행기와 이 점이 다릅니다.
스케줄을 바꾸려면 새 numerics로 다시 준비해야 합니다.
`report.resolved['checkpoint_steps']`,
`prepared.contract['resolved_checkpoint_steps']`,
`result.provenance['checkpoint_steps']`에서 선택값을 확인할 수 있습니다.

체크포인트 파일은 실제 CUDA 네 사례를 포함한 21사례가 통과했습니다.
17단계 전도체의 블록 1·64는 독립 SI 갱신식과 전도도·초기 진폭 미분을
보존합니다. 163단계 슬래브의 블록 11은 9단계 tail을 남기며 최종 장,
Record, Field, Flux 목적함수와 재료·광원 미분이 기존 스캔과 일치합니다
(값 상대 `5e-13`, 기울기 상대 `1e-12`). 여러 backward 조회가 그래프를
올바르게 유지하고, 독립적인 리프로 두 번 prepared 재사용해도 이전 결과를
보존합니다. 광원 이차 항등식은 별도로 `2e-11`에서 검사하며, 잘못된 live
재료는 실패한 validity와 NaN을 유지합니다.

입력 검사에는 불리언 Torch 텐서를 블록 길이로 거부하는 회귀 세 사례가
포함됩니다.

Job 네 사례는 고정된 기존 기본 키와 명시적 스케줄의 저장·복원을 검사합니다.
`checkpoint_steps=None`은 v1 Job envelope에서 생략하며 명시적 정수는
직렬화합니다. 선택적 계약과 제한된 미분 일치를 검증한 것으로, 일반 장면의
메모리 측정은 이 합격 검사와 별개입니다.

## 범위를 한정한 2차 미분

`test_general_cuda_higher_order.py`의 실제 CUDA 두 사례가 배정밀도
17단계 균일 주기 전도체에서 통과했습니다. 단순 스캔과 블록 3·2단계 tail을
검사합니다. 목적함수는 최종 Ey 제곱의 평균 `J = A**2 * q**34`이며
`q = (1 - sigma*f)/(1 + sigma*f)`, `f = dt/(2*epsilon0*epsilon)`입니다.
스칼라 유리 갱신식을 독립적으로 미분한 값·기울기·진폭/전도도 순서의 전체
Hessian을 상대 `2e-12`, 절대 0으로 비교하며 혼합 미분의 양쪽 순서를
포함합니다. 실행의 dt와 공통 epsilon0 상수를 사용하므로 미분의 기준이지
시간 간격이나 단위 상수의 독립 검증은 아닙니다.

진폭 리프는 CPU, 전도도 리프는 CUDA에서 시작합니다. Eager
`torch.autograd.grad(..., create_graph=True)` 후 double backward의 미분은
원래 리프 장치로 돌아갑니다. 바꾼 리프로 재사용한 결과는 새 준비와 일치하며
첫 결과는 그대로 유지됩니다. 두 사례는 앞의 1차 미분 검사 합계와 별개입니다.
임의의 기하, CPML, 다른 광원·재료 종류의 Hessian이나 `torch.func` 변환을
검증하지 않습니다.

## 체크포인트 비용 실측

prepared 전도체 사례는 native 기록의 환경, Torch CPU 디스패치 한 스레드,
CUDA float64 장, 간격 0.05 um, epsilon 2.25, 전도도 600 S/m,
초기 E = (0, 1, 0), 진폭 1.2를 사용합니다. 모든 면은 주기 경계입니다.
163단계 동안 한 지점의 Ey 기록과 최종 장을 보존합니다. 목적함수는 최종
전기장의 제곱합입니다. 진폭만 미분하는 그래프와 진폭·전도도를 함께
미분하는 그래프를 비교하며 live 스칼라 리프는 CUDA에 있습니다.

각 설정은 한 번 예열한 뒤 동기화한 반복 세 번을 측정합니다. 시간에는
live 리프 생성, prepared 실행, 목적함수와 역방향 미분을 포함하고 준비,
호스트 스칼라 복사, JSON 쓰기는 제외합니다. 메모리는 예열·가비지 수집·
할당자 캐시 해제 후 세 반복의 최대값입니다. 모든 행의 시작 할당량은
0이었습니다. allocated·reserved 최대값은 Torch 할당자 수치이며 드라이버·
화면 메모리를 제외합니다. CPU 메모리는 측정하지 않았습니다. 준비를 포함하며
물리 문제도 다른 native 표와 이 시간을 직접 비교할 수는 없습니다.

측정한 16설정 모두 완료했고 모든 반복에서 해당 검사를 통과했습니다.
독립 SI 갱신식은 `q = (1 - sigma*f) / (1 + sigma*f)`,
`f = dt / (2*epsilon0*epsilon)`, 목적함수는
`J = size**3 * amplitude**2 * q**(2*steps)`입니다. `2*J/amplitude`와
`dq/dsigma = -2*f/(1 + sigma*f)**2`를 사용한 해석적 전도도 미분을
상대 `2e-12`, 절대 0으로 검사합니다. 단순 스캔과의 일치와 별개로 두 미분
그래프를 모두 독립적으로 검증합니다.

시간은 초 단위 중앙값이며 메모리는 allocated/reserved MiB(2**20바이트)입니다.
`None`은 단순 스캔, 블록 8은 20블록과 3단계 tail, 블록 32는 5블록과
3단계 tail이며 블록 200은 일반 163단계 tail만 남깁니다.

| 격자 | 블록 | 진폭 시간 | 진폭 메모리 | 진폭 + 전도도 시간 | 진폭 + 전도도 메모리 |
| --- | --- | ---: | ---: | ---: | ---: |
| 16³ | None | 0.487020 | 1.037 / 2 | 0.518653 | 31.321 / 32 |
| 16³ | 8 | 0.708443 | 5.015 / 6 | 0.737885 | 6.207 / 8 |
| 16³ | 32 | 0.673048 | 2.174 / 4 | 0.697212 | 7.865 / 10 |
| 16³ | 200 | 0.519046 | 1.037 / 2 | 0.499353 | 31.321 / 32 |
| 32³ | None | 0.499816 | 8.255 / 12 | 0.537487 | 250.509 / 334 |
| 32³ | 8 | 0.667726 | 39.796 / 54 | 0.728847 | 49.301 / 66 |
| 32³ | 32 | 0.645770 | 17.268 / 24 | 0.661251 | 62.771 / 84 |
| 32³ | 200 | 0.474110 | 8.255 / 12 | 0.514292 | 250.509 / 334 |

32³ 재료 미분 그래프에서 블록 8은 할당 최대값을 5.08배 줄였지만 중앙값
시간을 0.537487초에서 0.728847초로 늘렸습니다. 같은 물리 문제에서 진폭만
미분하면 할당 최대값이 8.255 MiB에서 39.796 MiB로 늘었습니다. 어떤 리프가
미분을 요구하는지에 따라 저장·재계산의 비용 관계가 달라집니다. 이 사례가
보편적인 최적 블록을 정하거나 다른 장·관측·광원 이력의 절감을 보장하지 않습니다.

표는 두 로컬 JSON 실행을 옮긴 것이며 원시 머신 기록은 이 페이지에 배포하지
않습니다. 진폭 전용 실행의 측정기 SHA-256은
`ffb6cad9975326b2bcd1ddd240dd4351df4ea074cbce429675ff747c1d2a2263`,
전도도 실행은 재료 미분 검사를 추가한
`997bf8c7e4de01e6dca1e2469d84a6305aa6e660a32ce1a21388c65a47496848`입니다.

## 요청한 DFT 위상만 계산

관측 호출은 요청하지 않은 장의 위상을 생략합니다. Ey 전용 관측기는
사용하지 않는 H 위상을 계산하지 않습니다. 같은 장의 성분끼리 위상을
공유하던 기존 동작은 유지합니다. 전기장은 `(n+1)*dt`, 자기장은
`(n+0.5)*dt` 시각을 유지합니다. 위상 점화식 근사나 시간 단계 사이의
캐시는 없습니다. 변경 후 CPU 검사 37사례가 통과했으며, 새 독립 오프라인
구적·위상 민감 미분 21사례와 고정 관측 기준값·기존 시간 위치 변이 검사를
포함합니다. 전기장 전용·자기장 전용·혼합 성분의 실수·복소 이력을 NumPy,
JAX, Torch에서 검사합니다.

제한된 `monitor_cost.py` 실험은 같은 RTX 4060 Ti 환경, CPU 디스패치 한
스레드, 4×4×84 격자, 128단계, 12셀 PML, 세 파장의 Ey Field 두 개와
Flux 두 개를 사용했습니다. 두 실행 모두 `torch.no_grad()`와 단순 스캔을
사용했습니다. 예열 한 번 후 동기화한 prepared 실행 다섯 번으로 아래 시간을
측정하고, **별도의 계측 실행 한 번**으로 연산 횟수를 셌습니다. 시간에는 live
검사, lowering, 결과 변환을 포함하며 최초 준비, 결과 검사, 호스트 복사,
프로파일링은 제외합니다.

| 항목 | 미사용 위상 생략 전 | 미사용 위상 생략 후 |
| --- | ---: | ---: |
| 중앙값 초 [최소–최대] | 0.591719 [0.564726–0.605297] | 0.557161 [0.537056–0.638640] |
| 프로파일 `aten::exp` 호출 | 1025 | 769 |
| 프로파일 `aten::mul` 호출 | 5888 | 5120 |
| 최대 allocated / reserved 바이트 | 420864 / 2097152 | 420864 / 2097152 |

지수 256회·곱셈 768회 감소는 128단계에서 단계마다 쓰지 않는 H 위상
두 개를 생략한 결과와 일치합니다. 모든 배치·dtype·유한값·실행 validity 검사를 통과했습니다.
기록한 장 최대값과 플럭스 요약은 전후 정확히 같았지만 전체 장의 동일성
검사는 아닙니다. 짧은 시간창에는 약한 초기 펄스만 있으므로 독립 광학 정확도
근거가 되지 않습니다. 시작 할당량은 0이며 최대값은 드라이버·화면·CPU
메모리를 제외합니다. 프로파일러의 shape·메모리 기록은 비용과 텐서 수명을
바꾸며 포괄적인 CPU·장치 시간을 더해서는 안 됩니다.

시간 범위가 겹칩니다. 관측한 중앙값으로 통계적으로 유의하거나 보편적인
속도 향상을 입증하지 않으며 최대 할당량도 줄지 않았습니다. 표는 로컬
기록을 옮긴 것이며 원시 머신 trace를 배포하지 않습니다. 기록한
`src/wavebit/fdtd/_kernel/observe.py` SHA-256은 미사용 위상 생략 전
`23e5a6878f8ee14457c1cbcec422fba4f3b197ed9bb55ef4175b1b12f8988a83`,
생략 후
`709828e0ffa334200d51ab100476f403baeaf05bd2ecb71c9b0c9f0d80cb9f01`입니다.
이 실험은 no-grad 체크포인트 디스패치 변경 이전에 수행했습니다.

## 체크포인트 정책이 있는 순방향 전용 실행

`torch.no_grad()` 또는 `torch.inference_mode()`로 Torch 기울기 기록을
끄면 스케줄이 있는 스캔도 일반 순방향 경로를 사용합니다. 역방향 재계산에
사용하는 블록 복사와 체크포인트 래퍼를 생략합니다. 요청한 정책은 보고서,
prepared 계약과 결과 provenance에 유지하며 같은 prepared 객체를 기울기
기록이 켜진 상태에서 실행하면 다시 사용합니다. 입력 허용 범위를 바꾸거나
순방향 결과를 나중에 미분 가능한 결과로 만드는 기능은 아닙니다.

이후 실행은 확장한 체크포인트 파일 23사례와 관측 미분 두 사례로 총
25사례가 통과했습니다. 앞의 60사례 합계를 수정한 것이 아닌 별도 결과입니다.
새 컨텍스트 두 사례는 163단계·블록 11에서 최종 E/H, Record, 푸리에 E/H,
Flux, 광원 이력을 두 기울기 비활성 컨텍스트의 단순 스캔과 비교합니다.
CPU 진폭과 CUDA 재료 리프는 사용자가 소유하며 변하지 않습니다. 이후 새
리프로 미분 가능한 재사용을 하여 목적함수 네 개, 보존한 backward 그래프,
광원 이차 항등식, 기울기의 장치 소유권, 이전 결과의 불변성을 검사합니다.
값 상대 `5e-13`, 미분 `1e-12`, 광원 항등식 `2e-11` 기준을 유지합니다.
기존 JAX/Torch 체크포인트 스펙트럼 미분 네 사례도 런타임 변경 후 통과했습니다.

별도 CUDA 실험은 요청한 위상만 계산하는 동작을 고정하고 같은 짧은 관측 사례를
16×16×84 격자, 128단계, 블록 8, `torch.no_grad()`로 실행했습니다.
한 번 예열한 뒤 위에서 명시한 시간 경계로 동기화한 비계측 반복 다섯 번을
측정했습니다. 두 버전 모두 모든 수치 검사를 통과했으며 기록한 장 최대값·
플럭스 요약은 정확히 같았습니다.

| 항목 | 순방향 경로 분기 전 | 순방향 경로 분기 후 |
| --- | ---: | ---: |
| 중앙값 초 [최소–최대] | 0.584975 [0.553745–0.626643] | 0.548744 [0.532387–0.580045] |
| 최대 allocated 바이트 | 7394304 | 5763072 |
| 최대 reserved 바이트 | 8388608 | 8388608 |

시간 범위가 겹치므로 중앙값만으로 일반적이거나 통계적으로 유의한 속도
향상을 입증하지 않습니다. 이 순방향 전용 실측 사례의 최대 allocated는
감소했지만 reserved는 같았습니다. 시작 allocated·reserved는 모두 0이며
드라이버·화면·CPU 메모리를 제외한다는 제한도 같습니다. 짧은 시간창의 요약
일치는 전체 장 비교나 광학 검증이 아닙니다. 로컬 기록의
`src/wavebit/fdtd/_runtime.py` SHA-256은 분기 전
`048e38423a374b7880e84f6abb8fef7e5cc2a077a7bce739e4b911bc6e57dcfa`,
분기 후
`6a3a0849bb5527e109b6c7c91d02e9c668c955a6f8a784bab5841f1f1467d11d`이며
원시 머신 기록은 여기서 배포하지 않습니다.

## 재현

[Native 환경 설정](native-cuda.md#환경과-재현)에 따라
독립 CUDA 환경과 import되는 소스 체크아웃을 확인합니다.

```sh
python -m pytest tests/fdtd/test_general_cuda.py tests/fdtd/test_general_cuda_observations.py tests/fdtd/test_general_cuda_checkpoint.py tests/api/test_fdtd_checkpoint_jobs.py -q
python -m pytest tests/fdtd/test_general_cuda_airy.py --run-slow -q
python -m pytest tests/fdtd/test_general_cuda_higher_order.py -q
python -m pytest tests/fdtd/test_general_cuda_dipole.py tests/fdtd/test_general_cuda_dipole_conductive.py -q
python -m pytest tests/fdtd/test_general_cuda_dipole_radiation.py --run-slow -q
python studies/fdtd/general_cuda/checkpoint_cost.py --sizes 16 32 --steps 163 --blocks 0 8 32 200 --repeat 3 --output artifacts/general-checkpoint.json
python studies/fdtd/general_cuda/monitor_cost.py --steps 128 --transverse-cells 4 --repeat 5 --device cuda:0 --profile --output-dir artifacts/general-monitor-profile
python studies/fdtd/general_cuda/monitor_cost.py --steps 128 --transverse-cells 16 --checkpoint-steps 8 --repeat 5 --device cuda:0 --output-dir artifacts/general-monitor-no-grad
```

연구의 `--blocks 0`은 공개 `checkpoint_steps=None`으로 변환합니다. 공개
블록 길이로 0은 유효하지 않습니다. 기본값은 두 미분 모드를 모두 측정하며
`--parameters amplitude` 또는 `--parameters material`로 하나를 고릅니다.
기존 출력 경로는 거부하고 시도와 실패한 검사를 JSON에 보존합니다.
저장소의 연구 안내는 `studies/fdtd/general_cuda/README.md`입니다.

필요한 백엔드나 장치가 없으면 CUDA 수치 검사를 건너뜁니다. 하드웨어가
필요 없는 범위 검사는 통과할 수 있습니다. CPU 전용 실행은 통과·skip
수를 구별해 보고하세요. 이 로컬 근거는 검토한 변경의 필수 CI를 대신하거나
릴리스 통합을 입증하지 않습니다.
