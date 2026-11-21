# 문제 정의, 검사, 준비와 결과

공개 이름 `wavebit.rcwa`와 `wavebit.fdtd`는 솔버 모듈입니다. 각 모듈은 같은 세 가지 연산을 제공합니다.

```python
report = rcwa.check(problem, numerics=numerics,
                    backend="numpy", device="cpu", precision="double")
report.raise_if_invalid()
prepared = rcwa.prepare(problem, numerics=numerics,
                        backend="numpy", device="cpu", precision="double")
result = prepared.run(problem)
# Equivalent one-shot entry point:
result = rcwa.run(problem, numerics=numerics,
                  backend="numpy", device="cpu", precision="double")
```

시간 영역 생애주기에는 `rcwa`를 `fdtd`로 바꾸세요. `Problem`에는 물리적 입력과 요청한 출력이 담깁니다. `Numerics`에는 근사와 이산화가 담깁니다. 실행 선택은 백엔드, 장치와 정밀도를 고릅니다. 어느 것도 직전 계산의 변경 가능한 상태에 숨어 있지 않습니다.

## 문제는 모든 실행의 입력입니다

선언은 불변이며 `.replace(...)`를 제공합니다. 새 파장, 재료 매개변수, 형상 치수나 여기(excitation)에는 새 문제 값을 만드세요. 준비된 객체는 교체한 값을 다음 입력으로 기억하지 않습니다. `prepared.run(new_problem)`에는 항상 완전한 현재 문제가 필요합니다. 결과는 이후에 다른 실행을 하더라도 자신의 관측에 필요한 상태를 유지합니다.

준비 단계는 솔버에 필요한 구조적 선택을 고정합니다. 배열의 모양과 개수, 기하의 위상(topology)과 유형, 이산화, 소스/모니터 배치, 요청한 출력 구조가 여기에 해당합니다. 재사용은 준비된 계약 안에서만 허용됩니다. 모양이 같은 파장 벡터는 RCWA의 실제(live) 입력이지만, 파장 개수를 바꾸면 그 계약이 바뀝니다. FDTD 재사용은 해석된 격자와 시간 진행 구조도 검사합니다. 일반적인 즉시(eager) 평가를 지원한다고 해서 임의의 추적(traced) 값도 지원한다고 추론하지 마세요. 기본 장치 위상 정책에서는, 고정 셀 카르테시안 Li 직사각형의 크기와 중심을 제외하면, `jax.jit`나 `jax.vmap` 아래에서 추적되는 RCWA 영역 기하나 셀은 트레이스를 만들 때 거부됩니다. 위상 검사에 값이 필요하기 때문입니다. 즉시 실행되는 `jax.grad` 아래에서는 PyTorch 텐서를 읽을 때처럼 검사가 tracer의 primal 값을 읽습니다. 명시적인 `Numerics(topology_check='host')`는 실행 시 유효성 검사와 NaN 마스킹을 갖춘 CPU 콜백을 통해 스칼라 영역 기하를 허용합니다. [JAX 변환 안에서 형상 위상 검사하기](../rcwa/guarded-shape-jit.md)를 참고하세요. [자동 미분](autodiff.md#rcwa-which-inputs-differentiate-on-which-backend)도 참고하세요.

## 검증은 명시적입니다

`check`는 구조화된 문제점, 해석된 설정, 기능과 추정치를 담은 `CheckReport`를 반환합니다. FDTD 필드를 진행시키지는 않습니다. `raise_if_invalid()`는 첫 번째 오류를 그것이 거부된 클래스, 즉 같은 입력에 대해 `prepare`나 `run`이 내는 클래스로 발생시킵니다. 바뀐 `prepared.structure`에는 `IncompatibleProblemError`, 지원하지 않는 요청에는 `UnsupportedFeatureError`, 그 밖에는 `ValidationError`입니다. `raise_if_invalid(error_type)`는 대신 모든 문제점을 `error_type`으로 발생시킵니다. 지원하지 않는 물리적 또는 수치적 조합은 다른 모델을 조용히 고르는 대신 지원하지 않음으로 보고합니다.

구체적인 값에 대한 검사는 실행 전에 유효하지 않은 값을 보고할 수 있습니다. 변환 안에서는 값에 의존하는 일부 조건이 Python 예외를 발생시킬 수 없습니다. 공개 결과는 그런 검사에 대한 유효성 정보를 노출하므로, 결과를 쓰기 전에 확인하세요. 트레이스가 성공했다는 것은 변환 호환성을 확립할 뿐 물리적 수렴을 확립하지는 않습니다.

## 단위와 실행

모든 기하와 진공 파장은 `Units(length="m" | "mm" | "um" | "nm")`를 사용합니다. 시간은 초, 일반 주파수는 Hz, 각속도 비율은 rad/s, 전기 전도율은 S/m, 각도는 라디안, E는 V/m, H는 A/m입니다. 내부의 정규화된 FDTD 좌표와 자기장은 공개 경계에서 변환됩니다.

백엔드는 `"numpy"`, `"jax"`, `"torch"`이며, 정밀도는 `"single"` 또는 `"double"`입니다. 필요하면 장치를 명시적으로 고르세요. Wavebit을 가져와도(import) 모든 선택적 백엔드가 로드되거나 전역 정밀도가 바뀌지 않습니다. JAX 배정밀도를 쓰려면 배열을 만들기 전에 x64를 켜세요. [백엔드, 장치와 정밀도](backends.md)를 참고하세요.

## 실행 가능한 예제와 호환성

[RCWA](../../getting-started/rcwa.md)나 [FDTD](../../getting-started/fdtd.md)부터 시작하세요. 실행 가능한 교육용 스크립트 15개는 현재 API를 사용합니다. 일부 설명용 코드 조각과 보관된 실험 스크립트는 이전 인터페이스와 수치 기록을 보존합니다. [마이그레이션 가이드](../../migration/index.md)는 그런 호출을 현재 모듈과 구별하며, [키워드 레퍼런스](../../reference/keywords.md#earlier-spellings)는 이전 `rcwa(...)` 생성자의 각 키워드를 지금 그 값을 담는 선언 필드에 대응시키고, 아직 대응 필드가 없는 몇 가지를 밝힙니다. 허용된 스칼라 형상의 JIT 미분은 장치 가드가 없을 때 명시적인 호스트 정책을 사용합니다. 즉시 실행되는 `jax.grad`도 생애주기를 통해 실행됩니다. [자동 미분](autodiff.md#why-jax-traced-geometry-is-refused-under-jit)을 참고하세요.
