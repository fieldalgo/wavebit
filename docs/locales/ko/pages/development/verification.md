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

실행할 때마다 `artifacts/verification/` 아래 새 디렉터리가 생깁니다.
`--output`으로 다른 새 디렉터리를 지정할 수 있지만 이미 존재하는 디렉터리는
거부합니다. 테스트 시작 전에 저장 위치를 출력합니다. 다음 파일을 확인하세요.

- `inputs.json`: 저장소 소스, 테스트, fixture와 그 밖의 기록 대상 입력의 해시,
  Python과 설치 배포판 버전, 선택한 설정, 수치 실행에 관련된 제한된 환경 변수
  목록입니다. 임의의 환경 변수, 자격 증명, 패키지 인덱스 URL은 기록하지 않습니다.
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

## 동일한 실행 이어가기

```shell
python tools/run_verification.py --profile full --workers 4 --resume RUN_DIRECTORY
```

profile, 경로, worker 설정, 체크아웃 입력과 환경을 그대로 유지해야 합니다.
도구는 이를 다시 fingerprint로 계산하고 선택 범위 전체를 다시 수집합니다.
새 테스트, 사라진 테스트, 순서가 바뀐 테스트를 포함해 fingerprint나 수집한
node ID가 하나라도 바뀌면 재사용을 거부합니다. 기본적으로 실패와 미완료
테스트는 다시 실행하고, 완료된 통과·건너뜀·예상 실패와 non-strict 예상 밖 통과는 원래 기록 위치와 함께
재사용합니다. 예상 밖 통과는 계속 별도로 보고하며, strict XPASS는 실패이므로
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
