(jobs-and-sweeps)=
# 18. 한 번의 계산을 실험으로 확장하기: JobSpec과 스윕

:::{note}
이 교재의 실행 예제는 원래 수치 기록을 재현할 수 있도록 `wavebit.compat.legacy` 인터페이스를 명시적으로 사용합니다. 새 작업은 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 참고하세요. 기존 측정값은 이번 문서 이행에서 다시 계산하지 않았습니다.
:::

`wavebit.rcwa(...)`로 솔버를 만들고 `solver.solve()`를 호출하면 Python 코드에서 바로 계산할 수 있습니다. **`JobSpec`은 계산 설정을 저장하거나 여러 조건의 계산을 정리할 때 선택해서 사용하는 객체**입니다. 구조·재료·빛·계산 옵션을 한곳에 묶어 두며, 객체를 만드는 것만으로 계산이 시작되지는 않습니다.

## JobSpec, expand(), run()은 무엇인가

세 가지 모두 Wavebit에서 제공합니다. 아래에서 `spec`은 사용자가 만든 `JobSpec` 객체를 담는 변수 이름입니다.

| 이름 | 종류 | 하는 일 |
| --- | --- | --- |
| `wavebit.JobSpec(...)` | 클래스 | 계산할 구조와 조건을 하나의 객체로 묶음 |
| `spec.expand()` | `JobSpec`의 메서드 | 스윕을 파장별·각도별 계산 설정 목록으로 나눔. 계산은 실행하지 않음 |
| `wavebit.run(point)` | 함수 | 계산 설정 하나를 받아 실제로 계산하고 `Result`를 돌려줌 |

**로컬 Python 환경에서도 사용할 수 있습니다.** 설정을 서버로 전달하는 프로그램을 만들 때도 활용할 수 있지만, 이 API 자체가 클라우드로 작업을 전송하거나 여러 계산을 자동으로 병렬 실행하지는 않습니다. `run()`은 호출한 환경에서 선택한 백엔드로 계산합니다.

## 계산 설정에는 무엇을 담는가

`JobSpec`은 아래 객체들을 묶습니다. 여기서 스윕은 **파장이나 입사각 등의 값을 바꾸며 계산을 반복하는 것**을 뜻합니다.

| 객체 | 담는 내용 | `JobSpec`에서의 위치 |
| --- | --- | --- |
| `Layer` | 한 층의 두께와 재료 분포 | `Stack.layers`의 원소 |
| `Stack` | 층 목록, 위·아래 매질, 격자 | `stack` |
| `Incidence` | 파장, 입사 방향, 편광, 입사 쪽 | `incidence` |
| `Numerics` | 백엔드, Fourier 차수, factorization 등 계산 옵션 | `numerics` |
| `Sweep` | 어떤 입력을 어떤 값들로 바꿀지 | `sweep` |

```{figure} ../../figures/learn/learn/18-jobspec.svg
:alt: JobSpec을 구성하는 구조·입사 조건·계산 옵션·스윕, 그리고 여러 파장과 각도의 모든 조합
:figclass: learning-figure

그림 18-1. 구조와 계산 옵션을 유지하면서 파장·각도의 모든 조합을 계산 설정으로 만듭니다.
```

## 예시: 실리콘 층을 세 파장에서 계산하기

[7장](materials.md)의 파장별 재료 객체를 층의 `background`에 넣습니다. `shapes=[]`이므로 층 전체가 실리콘인 균일층이며, Fourier 차수는 0으로 충분합니다. 길이 단위는 nm입니다.

```python
from wavebit.compat import legacy as wavebit

silicon = wavebit.materials.load('Si')
spec = wavebit.JobSpec(
    stack=wavebit.Stack(
        layers=[wavebit.Layer(
            thickness=100.,
            structure=dict(background=silicon, shapes=[]),
        )],
        top=1., bot=2.25, period=(600., 600.),
    ),
    incidence=wavebit.Incidence(
        wavelength=900., theta=0., polarization=(1., 0.),
    ),
    numerics=wavebit.Numerics(truncation_order=(0, 0)),
    sweep=wavebit.Sweep.from_dict({'wavelength': [800., 900., 1000.]}),
)
```

`incidence.wavelength=900.`은 기본 파장이고, 이 예제에서는 `sweep`에 적힌 세 값으로 각각 바꿔 계산합니다. 여기까지는 설정만 만들었습니다.

### expand(): 계산 설정을 세 개로 나누기

```python
points = spec.expand()
print([point.incidence.wavelength for point in points])
# [800.0, 900.0, 1000.0]
```

`points`의 원소는 파장 숫자가 아니라 **각 파장이 지정된 `JobSpec` 객체**입니다. 구조·재료·계산 옵션은 유지되고, 파장만 각각 800·900·1,000 nm로 달라집니다. 스윕이 없는 설정에서 `expand()`를 호출하면 원래 설정 하나가 들어 있는 목록을 얻습니다.

### run(): 각 설정으로 실제 계산하기

```python
for point in points:
    result = wavebit.run(point)
    print(point.incidence.wavelength, result.R.sum(), result.T.sum())
```

이 반복문은 세 계산을 차례로 실행합니다. 800 nm 계산에는 실리콘 표에서 구한 800 nm의 재료값을, 900 nm 계산에는 900 nm의 값을 사용합니다. 이것이 **계산 파장에 맞게 재료값을 자동으로 구한다**는 뜻입니다. 결과를 읽는 방법은 `solver.solve()`의 결과와 같으며 [12장](results.md)을 참고합니다.

`run()`은 한 조건의 설정을 받습니다. `spec`처럼 스윕이 들어 있는 설정을 바로 `wavebit.run(spec)`에 넣으면 오류가 나므로, 위와 같이 `expand()`로 나눈 뒤 실행합니다.

(jobs-json)=
## JSON에 저장하고 다시 불러오기

**JSON은 설정을 텍스트로 표현하는 형식**입니다. `to_json()`은 JSON 문자열을 만들고, `from_json()`은 그 문자열에서 `JobSpec`을 복원합니다. 파일에 쓰고 읽는 것은 아래의 `Path` 코드가 담당합니다.

```python
from pathlib import Path

path = Path('silicon-sweep.json')
path.write_text(spec.to_json(indent=2), encoding='utf-8')

restored = wavebit.JobSpec.from_json(path.read_text(encoding='utf-8'))
for point in restored.expand():
    result = wavebit.run(point)
    print(point.incidence.wavelength, result.R.sum(), result.T.sum())
```

이 예제에서는 **실리콘 이름만 저장하는 것이 아니라, 불러온 파장별 재료표도 JSON에 함께 저장합니다.** 복원한 설정은 그 표를 사용해 각 파장의 재료값을 구합니다. 굴절률에서 만든 재료는 굴절률 표와 선택적으로 지정한 투자율 표를, 유전율에서 만든 재료는 유전율 표와 선택적으로 지정한 투자율 표를 보존합니다. 이방성 재료의 텐서 성분도 보존합니다.

반면 `silicon.at(900.)`으로 만든 `Material`을 할당했다면 JSON에도 900 nm에서 구한 고정값이 들어갑니다. 복원한 뒤 파장을 바꿔도 그 고정값을 사용한다는 점은 [7장의 두 재료 객체 설명](materials.md)과 같습니다.

`spec.key()`는 입력 설정을 비교하기 위한 식별값입니다. 저장 전후 `spec.key() == restored.key()`이면 정규화한 입력 설정이 같습니다. 계산 결과를 재현하려면 이 설정과 함께 Wavebit·의존성 버전, 백엔드와 정밀도도 기록합니다. JSON에는 파일 형식을 구분하는 `version`이 포함되며 현재 값은 10입니다. 이전 형식 파일을 사용할 때의 변환 규칙은 [영문 설정 안내](../../migration/jobs-v1.md#what-a-document-must-say-and-what-is-refused)를 참고합니다.

## 여러 입력을 함께 바꾸기

`Sweep`으로 바꿀 수 있는 입력은 다음과 같습니다.

| 입력 | 바꾸는 값 |
| --- | --- |
| `wavelength` | 진공 파장 |
| `theta`, `phi` | 입사 위상각과 방위각 |
| `kx`, `ky` | 직접 지정하는 x·y 방향 파수 |
| `psi`, `chi` | 편광 각도 |

파장 3개와 각도 4개를 지정하면 같은 순번끼리 짝짓지 않고 **모든 조합인 12개의 계산 설정**을 만듭니다. 균등한 간격은 `{'linspace': [시작값, 끝값, 개수]}`로 지정할 수 있습니다. 아래 예제 파일에 이 표기를 사용했습니다.

`theta`·`phi`와 `kx`·`ky`는 {ref}`8장 <incident-direction-inputs>`의 두 입력 방식이므로 함께 지정하지 않습니다. 직접 파수를 스윕할 때는 기본 `Incidence`도 `Incidence(kx=..., ky=...)`로 만듭니다. 파장 스윕에서 각도를 고정하는 것과 실제 파수를 고정하는 것은 서로 다른 조건입니다.

형상 변경이나 숫자 래스터의 재료 분산은 별도의 Python 반복문에서 구조를 갱신합니다. 숫자 래스터는 재료표를 보관하지 않으므로, 파장만 바꾼다고 배열의 유전율이 바뀌지는 않습니다. 분산 재료 객체를 도형·배경·위아래 매질에 할당했을 때는 솔버가 파장별 값을 구합니다.

## 실행 파일: 래스터 구조의 저장·복원과 파장 스윕

아래 파일은 ε=1과 ε=12.25로 이루어진 래스터 구조를 JSON으로 저장·복원한 뒤 세 파장에서 계산합니다. 이 예제의 재료값은 고정되어 있으므로 파장 변화에 따른 구조의 응답을 비교합니다.

실행 파일: {download}`04_jobspec.py <../../../examples/rcwa/learning/learn/04_jobspec.py>`

```{literalinclude} ../../../examples/rcwa/learning/learn/04_jobspec.py
:language: python
:linenos:
```

| 파장 (nm) | R | T |
| --- | --- | --- |
| 800.0 | 0.460322555 | 0.539677445 |
| 900.0 | 0.205331628 | 0.794668372 |
| 1000.0 | 0.171513741 | 0.828486259 |

전체 인자와 JSON 형식은 [영문 사용 안내](../../guide/common/specs.md), 클래스와 메서드 목록은 [API 문서](../../migration/legacy-api.md#a-run-as-data)에서 확인할 수 있습니다. 외부 프로그램이 사용할 기능 목록은 [기능 매니페스트 안내](../../reference/common/capability-manifest.md)에 따로 설명합니다.
