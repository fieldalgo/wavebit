# 연구 계산 저장과 결과 확인

`wavebit.records`는 이식 가능한 `Job`, 계산 결과와 그 컴퓨터에서 사용한 소스를
저장합니다. 동료는 계산에 사용한 백엔드를 설치하거나 솔버를 다시 실행하지 않고,
기본 Wavebit 설치만으로 수치를 확인할 수 있습니다.

## 계산과 함께 기록하기

```python
from wavebit import Job, Material, Units, rcwa
from wavebit.records import run_recorded, load_run

problem = rcwa.Problem(
    units=Units(length="um"),
    stack=rcwa.Stack(period=(.8, .8), layers=(
        rcwa.Layer(thickness=.2, background=Material(2.25)),
    )),
    excitation=rcwa.Excitation(wavelength=(.6, .8), sources=(
        rcwa.PlaneWave(polarization="s"),
    )),
    outputs=rcwa.Outputs(fields=True),
)
job = Job(solver="rcwa", problem=problem, numerics=rcwa.Numerics())
record = run_recorded(
    job, "artifacts/runs/film-001",
    sample=lambda result: {
        "inside_film": result.field(x=(0., .1), y=(0.,), z=(.1,)),
    },
)

loaded = load_run(record.path)
transmission = loaded.outputs["transmitted"]["efficiency"]
field = loaded.samples["inside_film"]
print(transmission, field["units"], loaded.manifest["source"]["identity"])
```

저장 대상은 새 경로여야 합니다. 이미 있는 파일, 디렉터리와 심볼릭 링크는 거부합니다.
실행마다 별도 디렉터리를 사용하세요. 실행 가능한
[`record_run.py`](https://github.com/fieldalgo/wavebit/blob/main/examples/common/record_run.py)
예제는 저장·복원한 배열이 정확히 일치하는지 검사합니다.
`--output artifacts/runs/film-001`을 지정하면 기록을 남깁니다. 기본 실행은
간단한 확인을 위해 임시 디렉터리를 사용합니다.

`run_recorded`는 Job을 한 번 실행한 다음, 지정했다면 `sample(result)` 함수를
호출합니다. 실행 전후의 이식 가능한 입력, 설치된 소스 목록과 기록 대상 환경을
비교하고, 바뀌었다면 저장을 거부합니다. 소스 목록은 디스크의 파일을 설명하며,
오래 실행 중인 인터프리터가 이전 편집 내용을 다시 불러왔는지까지 입증하지는
못합니다. 정확한 실험 기록이 필요하면 소스 수정 후 인터프리터를 재시작하세요.

## 이미 계산한 결과 저장하기

```python
from wavebit import run
from wavebit.records import save_run

result = run(job)
record = save_run(
    "artifacts/runs/film-002", job=job, result=result,
    samples={"inside_film": result.field(x=(0.,), y=(0.,), z=(.1,))},
)
```

Job과 Result가 같은 솔버를 사용하는지 검사합니다. 이미 계산된 결과에 정확한
입력을 연결할 책임은 호출자에게 있습니다. manifest는 이 연결 방식을 명시하며,
소스와 환경은 저장 시점에 관측합니다. 실행 과정까지 기록에 포함하려면
`run_recorded`를 사용하세요.

FDTD도 같은 방식으로 `Job(solver="fdtd", problem=..., numerics=...)`를 만듭니다.
시간 기록, Fourier 장, 회절, flux, 소스 보정 시계열과 요청한 최종 장의
수치 및 물리 메타데이터를 보존합니다.

## 보존하는 내용

| 데이터 | 저장 표현 |
| --- | --- |
| 문제, 재료, 단위, 수치 설정과 실행 설정 | 버전이 있는 `wavebit.job/v1` JSON과 정규화된 Job key |
| 실제로 평가한 출력과 명시적 표본 | 수치 NPZ 배열과 JSON의 이름 있는 필드 |
| 배열 dtype, shape, byte order | 불러올 때 배열 payload와 대조 |
| 좌표, 성분명, 단위, 기준면, 유효성과 진단 정보 | 값과 함께 저장하는 이름 있는 필드 |
| RCWA 파장과 요청한 산란 행렬 | `record.extra` |
| FDTD 소스 보정과 요청한 최종 장 | `record.extra` |
| Python, 플랫폼과 관련 설치 패키지 버전 | manifest 환경 메타데이터 |
| 설치된 Wavebit Python, type stub과 런타임 재료 데이터 | 파일별 해시, 결합된 소스 식별값, 기본 설정에서 `source.zip` |
| 패키지가 체크아웃에 속할 때의 Git 정보 | commit, 패키지의 변경 상태와 HEAD 대비 추적된 패키지 diff |

`include_source=False`는 소스 ZIP을 생략하되 소스 목록과 식별값은 남깁니다.
Git patch는 추적된 패키지 파일을 대상으로 하며, 추적되지 않은 Python 파일은
소스 ZIP에 포함합니다. 무관한 저장소 파일, 문서, 환경 변수와 자격 증명은
수집하지 않습니다. 스냅샷 대상은 설치된 패키지입니다. 저장소 전체 체크아웃이나
설치 가능한 wheel을 만드는 기능은 아닙니다.

모든 payload에 SHA-256 체크섬이 있습니다. manifest를 마지막에 기록하여 완료를
표시합니다. 쓰기 작업이 중단되면 불완전한 디렉터리가 남을 수 있으며, `load_run`은
이를 거부합니다. 체크섬은 손상이나 데이터 불일치를 탐지하며, 작성자를 증명하는
서명은 아닙니다.

## 결과 읽기와 다시 계산하기

`load_run(path)`는 manifest, Job 식별값, 소스 목록과 payload를 검사합니다.
보관된 Python을 실행하거나, 솔버를 호출하거나, 기록에 적힌 타입 이름을 import하거나,
pickle/object 배열을 읽지 않습니다. 읽어 온 수치 배열은 독립적으로 소유하는
읽기 전용 NumPy 배열입니다. dataclass 관측량은 `RecordedFields` mapping으로
바뀝니다. 실행 중인 관측 객체의 메서드 대신 `observation["values"]`처럼 접근하세요.
`type_name`에는 원래 타입을 설명용 문자열로 남깁니다.

이 형식은 계산으로 얻은 데이터를 보관합니다. 미분 그래프, 준비한 솔버 상태나
지연 평가기를 보존하지 않습니다. RCWA 장 복원과 FDTD 원거리 장 평가는
`record.deferred`에 나타납니다. 필요한 값을 명시적으로 평가한 뒤 `samples`로
전달하세요. 원거리 장 표본에는 값과 함께 `theta`, `phi`, 각도 단위,
반환된 장 또는 세기 단위를 포함하세요.
JAX `jit`/`grad`나 Torch 컴파일 밖에서 구체적인 값을 저장합니다. 장치 배열은
미분 연결을 떼고 호스트로 복사합니다. 이식 가능한 Job 자체에는 백엔드 버퍼 대신
명시적인 호스트 배열이 필요합니다.

다시 계산하는 일은 별도의 명시적인 단계입니다.

```python
from wavebit import run
repeated_result = run(loaded.job)
```

요청한 백엔드, 정밀도와 소스 버전을 갖춘 환경에서 실행하세요. 예를 들어 JAX
배정밀도 Job은 배열 생성 전에 x64를 활성화해야 합니다. 기록은 입력과 출처를
제공하며, 의존성 설치 도구나 하드웨어·버전이 달라도 바이트 단위로 재현된다는
보장은 아닙니다. 연구하는 물리 문제에 맞춰 계산 출력과 수렴을 비교하세요.

저장소의 테스트 실행 기록은 목적이 다릅니다. quick/full/research profile과
중단 후 이어 실행은 [검증 기록](../development/verification.md)을 참고하세요.
