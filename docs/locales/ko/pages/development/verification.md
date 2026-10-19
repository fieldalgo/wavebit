# 검증 기록과 안전한 이어 실행

## 저장소 검증 정책

PR은 draft로 엽니다. draft PR에는 CI 실행기를 배정하지 않고 기능 브랜치
push도 검증을 실행하지 않습니다. 적절한 로컬 또는 원격 환경에서 관련 검사를
통과하고 검토한 뒤 Ready로 전환합니다. workflow는 main, Ready PR 이벤트,
명시적 수동 dispatch에서 실행합니다. 자동 Rebase and merge는 모든 필수 검사를
유지합니다. main은 고정 workflow/ref 대기열(`queue: max`, 취소 없음)을 사용하고,
다른 ref는 `queue: single`로 이전 실행을 대체합니다. 이 Actions 대기열은
PR 병합 큐와 별개입니다.

전체 Linux 수용에는 default/slow 범위, 예제, 문서 검증이 포함됩니다.
독립 배포 검증표는 sdist와 wheel을 만들고 소스 트리 밖에 설치하여
Python 3.12의 Linux·macOS·Windows에서 공개 NumPy RCWA/FDTD 사용, 번들 데이터와
plotting을 확인합니다. 플랫폼 smoke 통과가 그 OS에서 모든 선택적 backend나
전체 수치 시험을 통과했다는 뜻은 아닙니다. 검증이나 릴리스 준비를 보고하기 전에
정확 커밋의 검사와 artifact를 확인합니다. [기여 안내](https://github.com/fieldalgo/wavebit/blob/main/CONTRIBUTING.md)와
개발 문서가 공통 정책의 원본입니다. AGENTS.md는 짧은 기술 안내이며 CLAUDE.md는
이를 불러옵니다. 개인 선호·계정 설정·비공개 경로·세션 이력은 공개 저장소 밖에서 관리합니다.

main 대상 draft PR에서 관련 검사와 검토를 마친 뒤 필수 CI와 GitHub Rebase and merge를
거칩니다. 충돌할 때만 수동 rebase와 재실행이 필요합니다. 공개 이력을 보존하며 main 직접
push·공유 브랜치 강제 push·반영 시 squash·통합 merge 커밋을 사용하지 않습니다.
결과 main SHA와 main 검증을 보존하고, 실패한 main 변경은 릴리스 전에 PR로 수정하거나
되돌립니다. 릴리스 태그에는 정확한 결과 커밋의 검증 성공이 필요합니다.

경로·임시 디렉터리·subprocess는 이식 가능한 API와 명시적 인코딩을 사용합니다.
수치 기준을 낮추거나 설명 없는 skip으로 실패를 감추지 않고 OS별 동작을 확인합니다.
기본 언어는 영어이며 한국어 문서를 영어판과 함께 갱신합니다. 실행 예제와 코드는 영어로 유지합니다.

개발에 사용하는 Python 인터프리터로 저장소 도구를 실행합니다.

```shell
python tools/run_verification.py --profile quick
python tools/run_verification.py --profile full --workers 4
python tools/run_verification.py --profile research --workers 2
```

`quick`은 설치 메타데이터·데이터, 공개 API, 재료 단위, import, Job 직렬화 왕복,
Fresnel 응답, FDTD 갱신을 확인하는 작은 검사 묶음입니다. 실행 확인용이며
물리적 수렴을 입증하지 않습니다. `full`은 **research를 제외한 모든 테스트를
slow 테스트까지 포함해** 실행합니다. `research`는 research로 표시된 테스트만
실행하며 그 안의 slow 사례도 포함합니다. 이 도구는 명시적인 marker 선택을
사용합니다. 일반적인 `pytest tests`는 기존 기본 정책대로 research를 선택에서
제외하고 slow 테스트를 건너뜁니다.

병렬 worker는 선택 사항이며 기본은 순차 실행입니다. `--workers N`에는
pytest-xdist가 필요합니다. 도구는 자체 기록기와 xdist를 명시적으로 불러오고,
환경에 설치된 무관한 pytest 플러그인을 자동으로 불러오지 않습니다. 저장소의
기존 conftest hook과 pytest 설정은 계속 적용됩니다. 지정하지 않은 BLAS 스레드
수는 1로 설정합니다.

선택 사항인 `--slow`는 research 선택 규칙을 유지하면서 profile의 범위를 좁힙니다.
기본값 `include`는 profile의 기존 선택을 유지하고, `exclude`는 slow 사례를
제외하며, `only`는 해당 범위의 slow 사례만 선택합니다. `quick --slow only`는
거부합니다. 따라서 명시적으로 좁히지 않으면 `full`은 여전히 slow 사례를 포함합니다.

`--timeout SECONDS`는 양수인 테스트별 시간 제한을 설정하며 pytest-timeout이
필요합니다. 요청하면 도구가 플러그인을 명시적으로 불러오고, 설치되어 있지 않으면
실행을 거부합니다. `--timeout-method`는 기본값 `thread` 또는 `signal`을 받습니다.
별도의 순차 mutation 단계에도 같은 제한을 적용합니다. 옵션을 생략하면 새로운
시간 제한을 추가하지 않습니다. `--verbose`는 자세한 테스트 진행을 출력하고,
`--live-output`은 단계 로그에 저장하면서 출력을 실시간으로 보여줍니다.
수집 단계는 조용히 파일로만 기록하며 실시간·상세 출력은 실제 테스트 실행 단계에
적용합니다. thread timeout은 최종 JUnit XML이나 session summary가 완성되기 전에
pytest를 종료할 수 있습니다. 이미 flush한 JSONL 이벤트는 남기며, 중단된 노드는
미완료로 유지하고 0이 아닌 종료 상태를 반환합니다.
이 옵션은 검증용 하위 프로세스에 적용되며 일반 솔버 실행은 바꾸지 않습니다.

실행할 때마다 `artifacts/verification/` 아래 새 디렉터리가 생깁니다.
`--output`으로 다른 새 디렉터리를 지정할 수 있지만 이미 존재하는 디렉터리는
거부합니다. 테스트 시작 전에 저장 위치를 출력합니다. 다음 파일을 확인하세요.

- `inputs.json`: 저장소 소스, 테스트, fixture와 그 밖의 기록 대상 입력의 해시,
  Python과 설치 배포판 버전, 선택한 설정, `XLA_FLAGS`를 포함해 수치 실행에
  관련된 제한된 환경 변수 목록입니다. 임의의 환경 변수, 자격 증명, 패키지 인덱스 URL은 기록하지 않습니다.
- `collected-nodeids.json`: 선택한 profile의 정확한 테스트 목록과 순서입니다.
- `receipt.json`: 노드별 결과, 재사용한 노드, 무결성 검사, 단계별 종료 코드와
  경과 시간입니다. 통과, 실패, setup/teardown 오류, 건너뜀, 예상 실패,
  예상 밖 통과, 미완료를 서로 구분합니다.
- 각 단계 디렉터리: pytest 출력, JUnit XML, 순서가 있는 JSONL 단계 이벤트,
  worker별 수집 목록과 실제로 import한 패키지 위치를 보관합니다.

수집 오류, 중복 node ID, worker별 수집 목록 불일치, 입력 변경은 오류입니다.
성공한 빈 실행으로 처리하지 않습니다. 테스트 본문이 통과했더라도 teardown이
끝나야 재사용할 수 있습니다. 중단된 실행은 완료된 이벤트를 남기고 0이 아닌
종료 상태를 반환하며, 보고되지 않은 노드를 통과했다고 처리하지 않습니다.

## main 검증 대기열

각 검증 워크플로의 `main` 실행은 workflow/ref별로 고정된 동시성 그룹을 사용하며
`queue: max`, `cancel-in-progress: false`로 설정됩니다. 같은 그룹에서는 한 번에
하나씩 실행하고, 새 push가 기존 대기를 교체하지 않도록 최대 100개 pending 실행을
보존합니다. 큐가 가득 차면 추가 실행은 취소될 수 있습니다. 처리 순서는 큐 진입
순서에 따르며 커밋 또는 push 순서를 보장하지 않습니다.

다른 ref는 `queue: single`을 사용하며 같은 ref의 새 실행이 이전 실행을 대체합니다.
서로 다른 검증 워크플로와 레포는 실행기 용량 내에서 병렬 실행할 수 있습니다.
이 설정이 슬롯을 예약하거나 PR의 실행 우선순위를 보장하지는 않습니다. 기존 실행은
시작할 때의 워크플로 설정을 유지합니다. 테스트 선택, 수치 기준, 필수 검사와 릴리스
검증은 바뀌지 않으며, 릴리스에는 정확한 대상 커밋의 검증 성공이 필요합니다.
이는 CI 실행 대기열이며 PR 병합 대기열과는 다릅니다.

## CI는 두 실행 범위를 shard로 나눠 실행

[테스트 workflow](https://github.com/fieldalgo/wavebit/blob/main/.github/workflows/tests.yml)는
research를 제외하는 두 범위 모두에 검증 도구를 사용하며, 각 범위를
`--shard INDEX/COUNT`로 여러 runner에 나눕니다. 일반 범위는 16개, slow 범위는 6개입니다.

CI는 검증 결과와 그 안의 pytest 임시 자료를 checkout 바깥의 `RUNNER_TEMP`에
저장하여 study fixture의 원자료 경로 분리 조건을 지킵니다. GitHub가 이 변수를
제공하며, 로컬에서 재현할 때는 먼저 외부 작업 디렉터리로 설정해야 합니다.
artifact 업로드도 같은 결과 경로를 사용합니다.

```shell
python tools/run_verification.py --profile full --slow exclude --shard 1/16 --workers 2 \
  --shard-timings tools/ci_shard_timings.json \
  --timeout 1200 --timeout-method thread --verbose --live-output \
  --output "$RUNNER_TEMP/wavebit-verification/default"
python tools/run_verification.py --profile full --slow only --shard 1/6 --workers 2 \
  --shard-timings tools/ci_shard_timings.json \
  --timeout 2400 --timeout-method thread --verbose --live-output \
  --output "$RUNNER_TEMP/wavebit-verification/slow"
```

shard는 profile 안의 선택이며 다른 profile이 아닙니다. 모든 shard가 범위 전체를
수집해 `collected-nodeids-all.json`에 기록한 뒤, 테스트 파일 단위로 결정적으로
배정합니다. 예상 시간이 긴 파일부터 가장 가벼운 shard에 넣으므로 파일이 갈리지 않고, 같은
수집 목록과 시간표는 항상 같은 shard를 만듭니다. shard job들이 끝나면
`tools/verify_shards.py --count N`이 그 기록을 읽어, 모든 shard가 한 번씩 있고
같은 목록을 수집했고 그 목록을 빈틈없이 나눴고 각각 종료 코드 0으로 끝났을 때만
통과합니다. 이 job이 범위를 검증한 것으로 만들며, runner가 죽은 shard는 사라지는
대신 이름이 남는 실패가 됩니다.

shard는 시간보다 runner 메모리 때문입니다. 일반 범위를 worker 2개 한 번에 돌리면
사용 가능한 7.9 GB를 넘어 매번 50~61 %에서 종료되었습니다. slow 범위도 3개 shard일 때
run 39098232669의 두 시도 모두 shard 3의 runner가 약 95 %에서 종료되어 6개로 나눕니다. 테스트별 제한은
1,200초(일반)·2,400초(slow) 그대로이고, job 제한은 shard당 120분입니다. 실측 시간을 배분에 사용하지만
실제 경과 시간은 수집, 컴파일, worker 스케줄링에도 영향을 받습니다. 각 shard 안에서 소스를 수정하는 사례는 공유 체크아웃의 병렬 실행
대신 도구가 만든 별도 스냅샷에서 순차 실행합니다.

GitHub artifact `verification-default-N`과 `verification-slow-N`에는 shard별 JSON 결과
기록과 입력 목록, JSONL 이벤트, JUnit XML, 로그를 보관합니다. artifact마다 저장소를
복제해 올리지 않도록 쓰기 가능한 소스 스냅샷은 업로드에서 제외합니다.
이 artifact는 검증 기록이며 전체 소스 스냅샷이 아닙니다.

## 동일한 실행 이어가기

```shell
python tools/run_verification.py --profile full --workers 4 --resume RUN_DIRECTORY
```

profile, slow 필터, shard, 경로, worker·timeout 설정, 출력 방식, 체크아웃 입력과 환경을
그대로 유지해야 합니다.
도구는 이를 다시 fingerprint로 계산하고 선택 범위 전체를 다시 수집합니다.
새 테스트, 사라진 테스트, 순서가 바뀐 테스트를 포함해 fingerprint나 수집한
node ID가 하나라도 바뀌면 재사용을 거부합니다. 기본적으로 실패와 미완료
테스트는 다시 실행하고, 완료된 통과·건너뜀·예상 실패와 non-strict 예상 밖 통과는 원래 기록 위치와 함께
재사용합니다. `XLA_FLAGS`의 값 변경뿐 아니라 새 설정이나 제거도 환경
fingerprint를 바꾸므로 재사용을 거부합니다. 이 경우 새 실행을 시작하세요.
예상 밖 통과는 계속 별도로 보고하며, strict XPASS는 실패이므로
다시 실행합니다. 이전 실행을 덮어쓰지 않고 새 기록에 합친 결과를 남깁니다.
이것은 로컬 재현성의 근거이며 원격 서비스, 하드웨어나 기록하지 않은 외부 상태가
변하지 않는다는 보장은 아닙니다.

`--path`를 여러 번 써서 저장소의 특정 경로나 node ID로 범위를 좁힐 수 있습니다.
`--collect-only`는 실행하지 않고 선택 목록만 기록합니다. 어느 옵션도 profile의
research/slow marker 표현식의 의미를 바꾸지 않습니다.

## 소스를 잠시 바꾸는 테스트

알려진 boundary·Gamma mutation 테스트와 `source_mutation`으로 표시한 테스트는
일반 실행에서 분리됩니다. 기록된 입력의 별도 복사본에서 순차 실행합니다.
기록기는 실행 전 바이트 일치를 검사하고, 그 복사본에서 패키지가 import되는지,
실행 후 복사본이 복원되었는지, 원래 체크아웃이 그대로인지 확인합니다.
패키지 소스를 쓰는 새 테스트에는 이 marker를 붙이고 assert가 실패해도
변경을 복원해야 합니다. 공유 체크아웃의 소스를 병렬 worker로 수정하지 마세요.

quick profile은 별도의 배포판 검증을 대체하지 않습니다.

```shell
python tools/run_distribution_smoke.py
```

이 도구는 wheel과 source distribution을 만들고 체크아웃 밖의 새 설치 환경에서
패키지 데이터와 선택적인 plotting 기능까지 확인합니다.

시간표는 `tools/ci_shard_timings.json`에 고정하며, 파일별 setup/call/teardown 합계를
현재 사례 수에 맞춰 추정합니다. 새 파일은 해당 범위의 사례당 평균 비용(최소 1초)을
사용합니다. 시간표는 resume fingerprint와 shard 간 일치 검사에 포함됩니다.
시간표 없이 실행하면 기존 사례 수 기준 배분을 사용합니다.
갱신은 성공한 실행의 전체 artifact를 내려받은 뒤 아래 명령으로 수행하고 변경을 검토해 커밋합니다.

```shell
python tools/update_shard_timings.py artifacts/ci-timing-baseline --source-run https://github.com/fieldalgo/wavebit/actions/runs/RUN_ID --output tools/ci_shard_timings.json
```

## Runner diagnostics

수동 workflow 입력 `slow_runner_diagnostics`로 실행하면 slow shard만 돌며, 각 shard에서
Linux 전용 보조 프로세스가 시스템 메모리/swap, 메모리 pressure, 프로세스 RSS,
cgroup v2 제한/이벤트를 5초마다 실시간 CI 로그와 `runner-diagnostics-slow-N`에
기록합니다. 명령행과 환경 변수는 수집하지 않습니다.
비밀번호 없는 sudo와 tracefs가 지원되면 별도 tracing instance에서 커널 신호 발생의
발신 task/PID 및 수신 대상, OOM 희생 프로세스도 기록합니다. 동작에 영향을 주지 않는
SIGCONT 자체 신호로 기록 경로를 확인합니다. 권한이나 기능이 없으면 명시적으로
기록하며, 이벤트가 없다고 신호나 OOM이 없었다고 단정하지 않습니다. SIGKILL이나
호스트 유실 시 마지막 샘플과 업로드가 남지 않을 수 있습니다.

이 실행은 다른 job과 일반 범위 집계를 생략하지만, `slow tests complete`는 slow shard들이
범위 전체를 이루는지 그대로 확인합니다. `diagnostics run, not a CI gate` job이 실행을
일부러 실패로 끝내므로, 릴리스 커밋의 성공한 `tests` 실행을 받아들이는 릴리스 검증이
이 실행을 통과로 세지 않습니다. 입력을 생략하거나 false로 두면 일반 CI gate가 유지되고
보조 프로세스는 돌지 않습니다. 수치 시험, worker 수와 시간 제한은 그대로입니다. 종료 원인을
시험/자원 제한/수동 취소로 판단하기 전에 메모리 counter 및 신호 시각을 시험 진행 로그와
대조해야 합니다.
