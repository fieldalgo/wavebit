# 검증 기록과 안전한 이어 실행

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

## main 검증의 동시 실행

테스트 워크플로의 각 `main` 실행은 `github.run_id`로 구분되는 고유한 동시성
그룹을 사용합니다. 따라서 새 main 커밋은 실행 중인 검증을 취소하거나 대기 중인
main 실행을 교체하지 않고 이전 커밋과 함께 검증할 수 있습니다. 실제로 동시에
실행할 수 있는 작업 수에는 GitHub 실행기 용량 제한이 계속 적용됩니다.
PR은 ref별로 같은 그룹을 사용하므로 같은 PR의 새 변경이 이전 검증을 대체합니다.
테스트 선택, 수치 수용 기준과 필수 검사는 바뀌지 않습니다. 이 정책은 이 버전의
워크플로를 사용하는 실행에 적용되며, 이전 버전으로 이미 대기 중인 실행을
변경하지 않습니다.

## CI는 두 실행 범위를 shard로 나눠 실행

[테스트 workflow](https://github.com/fieldalgo/wavebit/blob/main/.github/workflows/tests.yml)는
research를 제외하는 두 범위 모두에 검증 도구를 사용하며, 각 범위를
`--shard INDEX/COUNT`로 여러 runner에 나눕니다. 일반 범위는 6개, slow 범위는 2개입니다.

```shell
python tools/run_verification.py --profile full --slow exclude --shard 1/6 --workers 2 \
  --timeout 1200 --timeout-method thread --verbose --live-output \
  --output artifacts/verification/default
python tools/run_verification.py --profile full --slow only --shard 1/2 --workers 2 \
  --timeout 2400 --timeout-method thread --verbose --live-output \
  --output artifacts/verification/slow
```

shard는 profile 안의 선택이며 다른 profile이 아닙니다. 모든 shard가 범위 전체를
수집해 `collected-nodeids-all.json`에 기록한 뒤, 테스트 파일 단위로 결정적으로
배정합니다. 큰 파일부터 가장 가벼운 shard에 넣으므로 파일이 갈리지 않고, 같은
수집 목록은 항상 같은 shard를 만듭니다. shard job들이 끝나면
`tools/verify_shards.py --count N`이 그 기록을 읽어, 모든 shard가 한 번씩 있고
같은 목록을 수집했고 그 목록을 빈틈없이 나눴고 각각 종료 코드 0으로 끝났을 때만
통과합니다. 이 job이 범위를 검증한 것으로 만들며, runner가 죽은 shard는 사라지는
대신 이름이 남는 실패가 됩니다.

shard는 시간보다 runner 메모리 때문입니다. 일반 범위를 worker 2개 한 번에 돌리면
사용 가능한 7.9 GB를 넘어 매번 50~61 %에서 종료되었습니다. 테스트별 제한은
1,200초(일반)·2,400초(slow) 그대로이고, job 제한은 shard당 120분입니다(정상 길이는
일반 약 50분, slow 약 40분). 각 shard 안에서 소스를 수정하는 사례는 공유 체크아웃의 병렬 실행
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
