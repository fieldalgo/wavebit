# 재현 가능한 작업(job)

`Job`은 솔버 이름, 변경할 수 없는 물리 문제, 수치 설정과 실행 정책을 기록합니다. 변경 가능한 솔버의 마지막 상태를 덤프한 것이 아니라, 실행할 수 있는 선언입니다.

```python
from wavebit import Job, Execution

job = Job(solver="rcwa", problem=problem, numerics=numerics,
          execution=Execution(backend="numpy", device="cpu", precision="double"))
encoded = job.to_json()
restored = Job.from_json(encoded)
assert restored.key == job.key
```

[첫 RCWA 계산](../../getting-started/rcwa.md)에 실행할 수 있는 완전한 예제가 있습니다. `wavebit.run(job)`은 Job 하나를 그 Job의 실행 설정으로 실행합니다. 스윕은 먼저 별도의 Job들로 펼쳐집니다. 직렬화는 지원하는 물리 선언과 단위를 보존하며, 지원하지 않는 값은 명시적으로 실패합니다. 과거의 실험 기록은 원래 측정 그대로 남으며, 선언을 변환한다고 해서 다시 생성되지 않습니다.

`JobSpec`, 변경 가능한 솔버 스윕과 이전의 평평한 수치 용어는 [호환 작업(job) 설명서](../../migration/jobs-v1.md)에 정리되어 있습니다. 새 코드는 `Job`과 솔버별 `Numerics`를 사용합니다.

선언뿐 아니라 측정한 출력까지 보존하려면 [실행 기록](../run-records.md)을 사용하세요.
실행 기록은 배열, 단위, 진단 정보와 설치된 소스의 출처 정보를 Job과 함께 저장합니다.
기록을 불러오면 저장된 데이터를 살펴볼 뿐이며, 그 Job을 다시 실행하는 것은 별도의 명시적인 작업입니다.
