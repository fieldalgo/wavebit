# 16. NumPy, JAX, PyTorch: 같은 물리와 다른 계산 도구

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

`precision='double'`(complex128, 기본값) 또는 `'single'`(complex64)이 연산 정밀도를 정합니다. `rcwa.run`이나 `rcwa.prepare`에 넘기거나 `wavebit.Execution`에 넣습니다.

wavebit를 가져오는 것만으로는 JAX의 x64 설정이 바뀌지 않습니다. 이 교재의 JAX 예제는
**배열을 만들기 전에** `jax.config.update("jax_enable_x64", True)`를 호출합니다. 이 설정 없이
JAX를 `precision='double'`로 실행하면 거부합니다(`execution.jax_x64`). 이미 낮은 정밀도로 만든 값을 뒤늦게 높은 dtype으로 바꿔도 잃어버린 자릿수는 복원되지 않습니다. 이 설정은 JAX 프로세스 전체의 64비트 허용에 영향을 줍니다. 고대비·공진·감쇠 모드 문제에서는 낮은 정밀도를 선택할 때 결과와 기울기를 따로 확인합니다.
