# 16. NumPy, JAX, PyTorch: 같은 물리와 다른 계산 도구

:::{note}
이 교재의 실행 예제는 원래 수치 기록을 재현할 수 있도록 `wavebit.compat.legacy` 인터페이스를 명시적으로 사용합니다. 새 작업은 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 참고하세요. 기존 측정값은 이번 문서 이행에서 다시 계산하지 않았습니다.
:::

백엔드는 배열·선형대수·자동미분을 제공하는 계산 도구입니다. 백엔드를 바꾸는 것은 RCWA를 다른 물리 이론으로 바꾸는 일이 아닙니다. 현재 기본 솔버들은 공통 코어를 사용하고, 백엔드별 어댑터가 배열과 고유값 계산 등의 차이를 처리합니다.

```{figure} ../../figures/learn/learn/15-backends.svg
:alt: 공통 물리 코어와 세 배열 백엔드, 선택적 자동미분의 관계
:figclass: learning-figure

그림 16-1. 공통 물리 코어와 세 배열 백엔드, 선택적 자동미분의 관계
```

| 선택 | 주된 용도 | 주의할 점 |
| --- | --- | --- |
| `backend='numpy'` / `Backend.NUMPY` | 첫 계산, 결과 확인, 일반 NumPy 작업 | 자동미분 백엔드가 아님 |
| `backend='jax'` / `Backend.JAX` | JAX 자동미분, 지원 경로의 JIT·배치화 | 64비트 설정과 정적 형상 정보 |
| `backend='torch'` / `Backend.TORCH` | PyTorch eager 자동미분 | `torch.compile` 안에서도 Wavebit은 즉시 실행(eager)되므로 컴파일로 얻는 이득이 없음. `fullgraph=True`, `torch.export`, `torch.jit.trace`는 거부됨 |

기본값은 `'numpy'`입니다. 이름은 각 라이브러리를 import하는 이름 그대로이고, `Backend`는 같은 세 값을 가진 문자열 열거형이라 `Backend.JAX == 'jax'`입니다. 2026-10-11까지는 정수 `0`/`1`/`2`로 골랐지만, 지금은 정수(그리고 `'JAX'`, `'pytorch'` 같은 다른 철자)를 받을 수 있는 이름 목록과 함께 거부합니다.

실험용 보조 솔버까지 세 백엔드에 같은 기능이 있다는 뜻은 아닙니다. GPU 사용 여부도 설치된 백엔드와 연산 지원에 달려 있습니다. 작은 문제에서는 전송과 준비 비용이 커질 수 있으므로 ‘GPU이므로 빠르다’고 결론내리지 않습니다. 첫 JIT 컴파일 시간과 반복 실행 시간은 따로 측정합니다.

## 정밀도와 실행 도구를 혼동하지 않기

솔버의 `complex_dtype=0`은 배정밀도, `1`은 단정밀도입니다. dtype 객체도 받을 수 있습니다. 반면 저장용 `Numerics.complex_dtype`는 비트 수인 **128 또는 64**를 사용합니다. 둘 다 숫자라고 같은 값으로 복사하지 않습니다.

여기서 `wavebit`는 `from wavebit.compat import legacy as wavebit`로 가져온 별칭이며,
`enable_x64()`는 이 호환 API의 함수입니다. 새 API의 `import wavebit`만으로는 JAX의
x64 설정이 바뀌지 않습니다. 이 교재의 JAX 예제는 **배열을 만들기 전에**
`wavebit.enable_x64()`를 호출합니다. 이미 낮은 정밀도로 만든 값을 뒤늦게 높은 dtype으로 바꿔도 잃어버린 자릿수는 복원되지 않습니다. 이 설정은 JAX 프로세스 전체의 64비트 허용에 영향을 줍니다. 고대비·공진·감쇠 모드 문제에서는 낮은 정밀도를 선택할 때 결과와 기울기를 따로 확인합니다.
