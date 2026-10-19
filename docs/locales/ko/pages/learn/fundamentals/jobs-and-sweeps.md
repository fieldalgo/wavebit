(jobs-and-sweeps)=
# 18. 계산 하나에서 실험으로: Job과 sweep

현재 `Job`은 solver 이름, 물리 `Problem`, `Numerics`, `Execution`을 묶습니다.
선언을 만들거나 직렬화하는 것만으로 계산하지 않습니다. `wavebit.run(job)`은
현재 Python 환경에서 한 작업을 실행하며, 클라우드 제출이나 자동 병렬화 기능은 아닙니다.

| 선언 | 저장하는 내용 |
| --- | --- |
| `Problem` | 단위, 재료, 기하, 입사파와 요청 출력 |
| `Numerics` | 이산화와 알고리즘 선택 |
| `Execution` | backend, device, precision; 기본 NumPy/CPU/double |
| `Job` | 세 선언과 solver 식별자 |

## 실리콘 층을 세 파장에서 계산하기

[7장](materials.md)의 파장 의존 재료를 층의 background에 넣습니다. 균일한 층이므로
차수 0이면 충분합니다. 불변 선언의 `.replace(...)`로 파장마다 Job을 명시적으로 만듭니다.

```python
from wavebit import Job, Material, Units, materials, rcwa, run

problem = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(period=(600., 600.), bottom=Material(2.25), layers=(
        rcwa.Layer(thickness=100., background=materials.load('Si')),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(
        rcwa.PlaneWave(polarization='s'),)))
job = Job(solver='rcwa', problem=problem, numerics=rcwa.Numerics(orders=(0, 0)))
points = [job.replace(problem=problem.replace(
    excitation=problem.excitation.replace(wavelength=wavelength)))
    for wavelength in (800., 900., 1000.)]
for point in points:
    result = run(point)
    print(point.problem.excitation.wavelength, result.R.sum(), result.T.sum())
```

각 Job은 자기 파장과 동일한 실리콘 표를 보존하며, solver는 해당 파장에서 표를 평가합니다.
수치 raster에 이미 넣은 유전율 배열은 고정값입니다. 파장을 바꾸어도 그 배열이
자동으로 파장 의존 재료 모델이 되지는 않습니다.

(jobs-json)=
## 한 계산점을 저장하고 복원하기

```python
from pathlib import Path

path = Path('silicon-point.json')
path.write_text(points[0].to_json(indent=2), encoding='utf-8')
restored = Job.from_json(path.read_text(encoding='utf-8'))
assert restored.key == points[0].key
result = run(restored)
```

`to_json()`은 명시적 단위와 재료 선언을 포함하는 현재 `wavebit.job/v2` 형식을 저장합니다.
번들 재료표의 데이터도 포함하므로 복원한 Job은 같은 표를 평가합니다. `.key`는 정규화한
입력을 식별하는 속성입니다. 결과 cache나 수치 재현성 증명은 아니므로 패키지·의존성
버전과 실행 환경도 기록하세요.

[공개 이름이 바뀌기](../../migration/public-names.md) 전에 저장한 `wavebit.job/v1` 파일은
변환하지 않고 `job.schema` 코드로 거부합니다. 현재 이름으로 문제를 다시 만들어 저장하세요.

이전 `JobSpec` JSON은 다른 형식이며 자동 변환하지 않습니다. [이전 Job 안내](../../migration/jobs-v1.md)로
원래 단위·편광·sweep을 해석한 다음 현재 선언을 구성해야 합니다.

## 파장과 각도를 함께 바꾸기

세 파장과 네 각도의 열두 조합은 `itertools.product`로 만듭니다. 아래 loop는 명시적인 순차 실행입니다.

```python
from itertools import product

for wavelength, theta in product((800., 900., 1000.), (0., .1, .2, .3)):
    source = problem.excitation.sources[0].replace(theta=theta)
    updated = problem.replace(excitation=problem.excitation.replace(
        wavelength=wavelength, sources=(source,)))
    result = run(job.replace(problem=updated))
```

각도는 radian입니다. 파장을 바꿀 때 입사각을 고정하는 것과 물리 파수를 고정하는 것은
다르므로 [입사파 안내](../../guide/rcwa/light.md)를 참고하세요. 현재 `Job`에는 `expand()`가
없습니다. Python/NumPy로 개별 계산점을 만들거나 지원되는 문제에서는 파장 벡터를 사용하세요.

## 직렬화와 sweep 실행 파일

기존 링크를 위해 {download}`04_jobspec.py <../../../examples/rcwa/learning/learn/04_jobspec.py>`라는
파일명은 유지합니다. 구현은 현재 Job을 저장·복원하고, 세 파장을 명시적으로 펼친 뒤
각 점의 무손실 에너지 균형을 확인합니다.

```{literalinclude} ../../../examples/rcwa/learning/learn/04_jobspec.py
:language: python
:linenos:
```

원래 교재는 같은 raster에서 다음 합계를 기록했습니다. 과거 측정 파일과 원본 출처는
보존하며, 그 형식은 현재 Job schema와 구분합니다.

| 파장 (nm) | R | T |
| --- | --- | --- |
| 800.0 | 0.460322555 | 0.539677445 |
| 900.0 | 0.205331628 | 0.794668372 |
| 1000.0 | 0.171513741 | 0.828486259 |

[Capability manifest](../../reference/common/capability-manifest.md)는 Job과 별개이며
자신의 인터페이스 버전을 기록합니다.
