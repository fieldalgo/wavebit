# 면적 적분 도형 FDTD 미분의 외부 jit 메모리

Written: 2026-10-19.

Targets: `src/wavebit/fdtd/_periodic/scene.py`, `src/wavebit/fdtd/_periodic/runtime.py`,
`studies/cross_solver/painted_shapes/fdtd_painted_shapes.py`, `tests/fdtd/test_fdtd_painted_optics.py`.

**요약.** 추가 메모리는 XLA:CPU가 컴파일된 프로그램의 실행 순서를 정하는 방식에서 생깁니다.
시간 루프나 프로그램 크기는 원인이 아닙니다. 한 층에 면적 적분 사각형이 두 개 이상 있으면
`_painted_cell_materials`가 `[edges, images, ny, nx, segments, gauss]` 모양의 float64 중간값을
많이 만듭니다. 이 중간값들의 축약(reduction)은 YNNPACK 퓨전이고, 퓨전의 입력은 전체 버퍼로
존재해야 합니다. jaxlib 0.11.0에서 XLA:CPU의 기본 스케줄러는 동시성 최적화 스케줄러입니다.
이 스케줄러가 정하는 순서로는 마지막 중간값이 만들어질 때까지 어느 것도 해제되지 않아서,
모두가 같은 순간에 살아 있습니다. 순방향 모드 미분은 그 수를 몇 배로 늘립니다. 해상도 6에서 primal은
297개, JVP 한 열은 1,071개입니다. 즉시 실행은 연산을 하나씩 실행하므로 칠하기 한 번 분량
이상의 작업 메모리를 들고 있지 않습니다.

`xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED`를 쓰면 해상도 20의
`jit(value, jacfwd)`가 요청하는 XLA temp 할당이 17.5배 작아집니다(11.67 GB → 0.67 GB).
실행 최대 메모리는 2.44 GB였습니다. 기본 스케줄러에서는 해상도 20 테스트의 docstring에 13.6 GB로
기록되어 있습니다. 야코비안은 즉시 실행 결과와 1.1e-15 이내로 일치합니다.

공개 `wavebit.fdtd.periodic.run(..., backend='jax')` 사용자는 한 층에 `shape_integration='area'`
사각형이 두 개 이상 있을 때 영향을 받습니다. 회전 사각형이 하나뿐인 층과 `'separable'` 경로는
영향을 받지 않습니다. 역방향 모드는 한 해상도에서만 측정했고, 영향이 훨씬 작습니다. jax 0.12.0에서도 효과가 있지만,
그 비율은 0.11.0의 약 4분의 1입니다.

## 구성

2026-10-19, `3f2ad83`에서 측정했습니다. Linux x86-64 클라우드 컨테이너(Intel Xeon @ 2.80 GHz,
vCPU 4개, 15 GiB), Python 3.13.16, NumPy 2.5.4, JAX/jaxlib 0.11.0이며 전부 float64입니다.
작업은 `p = [.37, .055, .22]`, 기본 지속 시간 60의 `fdtd_painted_shapes.response(p, resolution)`입니다.
각 행은 별도 프로세스이고 프로세스는 한 번에 하나씩 실행했습니다.
회전 사각형 하나 행과 역방향 모드 행(컴파일만)은 같은 날 나중에, CPU가 Intel Xeon @ 2.10 GHz로
표시되는 새 컨테이너(vCPU 4개, 15 GiB, 같은 Python·NumPy·JAX, `3f2ad83` 이후 바뀌지 않은 소스)에서
추가했습니다. 아래에서 인용하는 slow 테스트는 `tests/fdtd/test_fdtd_painted_optics.py`의 두 개입니다.
해상도 20 테스트는 `test_painted_eight_complex_channels_and_three_shape_derivatives_at_resolution_20`,
해상도 6 테스트는 `test_painted_jacobian_under_jit_equals_the_eager_one_at_resolution_6`입니다.

- **최대**는 `ru_maxrss`입니다.
- **Temp**는 `jax.jit(g).lower(p).compile().memory_analysis().temp_size_in_bytes`입니다.
- 메모리 최적화 행은
  `XLA_FLAGS=--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED`만 설정합니다.
- **jac**은 연구가 쓰고 해상도 6 테스트가 실행하는 형태인 `jit(lambda p: (f(p), jax.jacfwd(f)(p)))`입니다.
- **jvp**는 각도 방향의 순방향 미분 한 열입니다.

실행 시간은 한 번 호출한 값입니다. 프로세스별 측정 로그와 XLA 덤프는 커밋하지 않았습니다.
이를 만드는 호출은 마지막 절 "재현"에 있습니다.

## Temp 크기는 프로그램이 아니라 스케줄을 따름

| 해상도 | 경우 | 즉시 실행 최대 | jit 기본: temp / 최대 | jit 메모리 최적화: temp / 최대 |
|---:|---|---:|---:|---:|
| 6 | primal | 1.31 GB | 0.28 / 1.03 GB | 0.015 / 0.80 GB |
| 6 | jvp | 1.42 GB | 1.03 / 2.40 GB | 0.042 / 1.49 GB |
| 6 | jac | 2.37 GB | 1.33 / 2.93 GB | 0.076 / 1.72 GB |
| 8 | primal | 1.31 GB | 0.43 / 1.20 GB | 0.024 / 0.82 GB |
| 8 | jvp | 1.47 GB | 1.61 / 2.92 GB | 0.065 / 1.43 GB |
| 8 | jac | 2.41 GB | 2.08 / 3.76 GB | 0.117 / 1.82 GB |
| 10 | primal | 1.32 GB | 0.62 / 1.37 GB | 0.035 / 0.81 GB |
| 10 | jvp | 1.42 GB | 2.32 / 3.62 GB | 0.092 / 1.47 GB |
| 10 | jac | 2.49 GB | 3.00 / 4.68 GB | 0.166 / 1.90 GB |
| 20 | jac | 2.67 GB | 11.67 GB / 실행 안 함 | 0.67 / 2.44 GB |

해상도 6 JVP의 두 스케줄된 모듈은 같은 명령을 순서만 다르게 담고 있습니다. 두 경우 모두 entry 명령 2,938개,
큰 값 1,071개, YNN 퓨전 216개입니다. 그래서 컴파일 비용은 두 스케줄러에서 같습니다.
프로세스는 lowering 후 0.3–0.4 GB, 미분 프로그램을 컴파일한 뒤 1.3–1.6 GB를 차지하므로
컴파일이 1.0–1.2 GB를 더합니다. 각 jit 최대값은 대략 컴파일 후 값에 temp를 더한 값입니다.
temp는 대략 `nx·ny`에 비례해 커집니다. 해상도 6의 셀은 4×4이고, 해상도 20의 temp는
해상도 6의 8.8배입니다.

해상도 20의 기본 temp와 컴파일 후 값을 더하면 약 13.3 GB입니다. 이는 해상도 20 테스트의 docstring이
Linux CPU, jax 0.11.0에서 이 경우에 기록한 13.6 GB, 그리고 CI 러너에서 기록한 13.3 GB와
맞습니다. 해상도 20 기본 프로그램은 컴파일만 하고 실행하지 않았습니다. 13.3 GB는 이 15 GiB
기계에 여유를 너무 적게 남기기 때문입니다. 같은 docstring이 해상도 30에 대해 기록한 22.9 GB
단일 할당은 이 temp 버퍼와 부합합니다.

값은 스케줄러에 의존하지 않습니다. 해상도 6–10에서 기본과 메모리 최적화 실행의 출력 합은
비트 단위로 같고, 이 합은 즉시 실행과 3.4e-15 이내로 일치합니다. 해상도 20에서는 야코비안을
즉시 실행 결과와 원소별로 비교했고 차이는 1.1e-15입니다.

미분 프로그램의 실행 시간은 메모리 최적화 스케줄러에서 같거나 짧았습니다.

- jvp: 해상도 6, 8, 10에서 5.1 대 5.5 s, 5.6 대 7.5 s, 7.5 대 15.2 s.
- jac: 7.7 대 7.9 s, 10.8 대 13.1 s, 16.0 대 16.8 s.

primal은 해상도 8과 10에서 더 느렸습니다(2.3 대 1.7 s, 2.9 대 2.6 s). 메모리 최적화 해상도
20 jac은 92 s가 걸렸고, 즉시 실행은 248 s였습니다.

CI 설정 `--xla_cpu_multi_thread_eigen=false`를 더해도 temp는 바뀌지 않습니다(해상도 6 jvp에서
1,031,495,480바이트). `CPU_SCHEDULER_TYPE_CONCURRENCY_OPTIMIZED`는 기본값의 temp를 바이트까지
재현하므로, 이 jaxlib에서 기본값은 동시성 최적화 스케줄러입니다.

같은 해상도 6 JVP를 2026-10-20에 2.10 GHz 컨테이너에서 `main`의 `b7463c7`로 다시 컴파일했습니다.
jax/jaxlib 0.12.0에서는 기본 스케줄러 785 MB, 메모리 최적화 197 MB를 요청하며,
`CPU_SCHEDULER_TYPE_CONCURRENCY_OPTIMIZED`는 이번에도 기본값과 바이트까지 같습니다. 같은 소스를
jax 0.11.0으로 컴파일하면 여전히 1031 MB와 42 MB이므로, 차이는 소스가 아니라 새 XLA에서
옵니다. 0.12.0에서는 효과가 더 작지만(24배 대신 약 4배) 여전히 있습니다. 이 문서의 나머지는
0.11.0에서만 측정했습니다.

## 버퍼의 정체

해상도 6에 대한 XLA 버퍼 할당 보고서(`--xla_dump_to`, `*-memory-usage-report.txt`)의
할당을, 각 할당이 처음 나열하는 모양별로 묶은 결과입니다.

```text
primal: 239.2 MiB   295 x f64[41,9,4,4,9,2]     jvp: 867.5 MiB  1070 x f64[41,9,4,4,9,2]
         17.4 MiB    43 x f64[41,9,4,4,9]             82.3 MiB   203 x f64[41,9,4,4,9]
```

이 모양은 `[변 조각, 3×3 주기 이미지, ny, nx, 절단 구간, 가우스 노드 2개]`입니다.
`_painted_cell_materials`의 `cdf`·`weight` 창 루프와 그 뒤의 합, 즉 `scene.py` 615–630행에서
나옵니다. 이 값들의 HLO 스택 프레임은 해당 행을 가리킵니다. 연산 이름은
`jvp(jit(clip))/select_n`, `min`과 `max`, `jvp(jit(_where))/select_n`, `mul`, `add`, `div`입니다.

스케줄된 entry 계산은 JVP에서 이 모양의 값 1,071개(primal은 297개)를 정의합니다. 이 값들은
각자 자기 할당을 가집니다. JVP 할당 1개와 primal 할당 2개는 `c128[3,42,4,4]` 값을 먼저 나열하므로,
위 블록은 이를 그 모양으로 세어 한두 개 적게 보여 줍니다. 프로그램의
YNN 퓨전 216개 중 96개가 이 값들을 읽습니다. 전형적인 하나는 다음과 같습니다.

```text
%ynn_fusion.40 = f64[4,4,2] fusion(<40 operands>), calls=%fused_computation.40,
    backend_config={"fusion_config":{"kind":"__ynn_fusion"}}
  fused_computation.40: 40 parameters (32 of painting size), 30 multiply, 13 add,
    reduce -> f64[41,9,4,4,9], reduce -> f64[41,9,4,4],
    ROOT reduce -> f64[4,4,2], dimensions={0,1}
```

이 퓨전에는 629행의 가우스 노드·구간 합과 630행의 `sum(value*jump, axis=(0, 1))`가 들어
있습니다. JVP에서는 곱의 미분 법칙이 이를 여러 항으로 나눕니다. 축약은 마지막 큰 값이 정의되기
전에 시작합니다. 96개 퓨전 중 48개가 그보다 먼저 실행됩니다. 하지만 앞선 퓨전들이 읽는 96개를
포함한 1,071개 모두를 나머지 48개 퓨전도 읽습니다. 이 퓨전들은 마지막 큰 값이 정의된 뒤
(2,938개 명령 중 2,757–2,851번째)에 실행되므로, 1,071개가 모두 존재하기 전에는 어느 것도
마지막 사용에 이르지 않습니다. 스케줄된
entry 계산에 대한 생존 분석으로 동시에 살아 있는 수를 세었습니다.

| 스케줄 (jvp, 해상도 6) | 큰 값 | 동시 생존 | Temp |
|---|---:|---:|---:|
| 기본 (동시성 최적화) | 1,071 | 1,071 | 1031 MB |
| 메모리 최적화 | 1,071 | 37 | 42 MB |

primal도 같은 구조이며 큰 값 297개가 모두 동시에 살아 있습니다(276 MB). YNN 퓨전을 끄면
(`--xla_cpu_experimental_ynn_fusion_type=`) 해상도 6에서 jvp 107 MB, jac 433 MB입니다.
이 설정은 temp 크기만 측정했습니다.

## 배제한 원인

- **시간 루프의 접선 carry.** 프로그램에는 primal과 JVP 모두 `while` 루프 13개가 있고 반복
  횟수가 32, 7, 2로 알려져 있어 펼쳐지지 않았습니다. 가장 큰 루프 carry는 0.45 MB(접선을 포함한
  `c128[3,42,4,4]` 장)입니다. checkpoint된 scan은 메모리가 쓰이는 곳이 아닙니다.
- **캡처된 상수.** JVP 모듈의 HLO 상수는 모두 합쳐 0.26 MB입니다.
- **주된 비용으로서의 컴파일.** `3f2ad83`에서 해상도 6 테스트의 docstring은 jit 비용을 "mostly
  compilation"이라고 했고, 지금은 이 문서를 인용합니다. 컴파일은 1.0–1.2 GB를 더하고 해상도에 따라 거의 바뀌지 않습니다.
  커지는 부분은 실행 temp입니다. 여기서는 이 temp 때문에 미분 경우의 jit 최대값이 해상도 6–10에서
  즉시 실행 최대값의 1.2–2.5배, 해상도 20에서는 약 3.8배가 됩니다(해상도 20 테스트 docstring의
  13.6 GB 대 3.6 GB).
- **즉시 실행 scan 프로그램 캐시.** 외부 jit 아래에서는 모든 leaf가 tracer이므로 일반
  `lax.scan`으로 되돌아갑니다. 즉시 실행 `jacfwd`도 클로저 값이 tracer이므로 같은 경로를 탑니다.
  여기서 측정한 캐시(`_eager_scan`)는 이후 `JaxTimeLoop._run`으로 바뀌었고, 같은 규칙을
  따릅니다. tracer가 하나라도 있으면 일반 `lax.scan`을 실행합니다. jax 0.11.0과 0.12.0에서 값,
  `grad`, `jacfwd`를 외부 `jit`으로 감싼 최적화 프로그램을 교체 전 프로그램과 비교했고 명령이
  같았습니다.

해상도 6 docstring은 지속 시간 40에서 해상도 6, 8, 10에 대해 5.3, 6.2, 7.2 GB를 기록합니다.
이 값은 즉시 실행 야코비안을 jit보다 먼저 실행하는 프로세스에서 나온 것입니다. 여기서 jit만
실행하는 프로세스는 지속 시간 60에서 2.93, 3.76, 4.68 GB를 측정합니다. 즉시 실행 최대값의 차이
(해상도 20에서 여기서는 2.67 GB, docstring은 3.6 GB)는 조사하지 않았습니다.

## 공개 사용자

공개 periodic API도 같은 칠하기 경로에 도달합니다. 측정한 경우는 0.2 µm 층 하나에 있는
연구의 `Rectangle` 영역 두 개의 회전각에 대한 순방향 미분입니다. 설정은 2026-10-19부터 쓰는 공개 이름으로
적으면 다음과 같습니다(`3f2ad83`의 실행에서는 펄스를 `Pulse(width_seconds=2.5*units.time_seconds)`,
시간 단위를 `units.time_seconds`로 썼습니다.
[공개 이름 이전 가이드](../../migration/public-names.md#native-grid-and-periodic-stack-names) 참고).

- `Stack(period=(.6, .6), bottom=Material(epsilon=1.21))`, 1 µm `PlaneWave`,
  `Pulse(frequency_sigma_hz=1/(np.sqrt(2)*np.pi*2.5*units.time_unit_seconds))`.
- `Numerics(spacing=1/6, duration_seconds=60*units.time_unit_seconds, pml_cells=6, orders=0,
  align_layers=True, shape_integration='area', shape_smoothing=.05,
  extraction_method='directional')`.
- `jax.jit(jax.jvp(...))` 아래의 `run(..., backend='jax')`.

연구와 조금 다른 프로그램이 컴파일됩니다. 기본 스케줄러의 temp는 연구의 jvp와 같습니다(해상도 6에서
1031.5 MB, 10에서 2321.5 MB). 메모리 최적화 temp는 더 커서 해상도 6에서 42.5 MB 대신 201.5 MB,
10에서 92.1 MB 대신 453.2 MB이고, 감소 폭은 약 24배가 아니라 약 5배입니다.

| 공개 경로, 해상도 6 | Temp | 최대 |
|---|---:|---:|
| 회전 사각형 두 개, `area`, jit 기본 | 1031.5 MB | 2.35 GB |
| 회전 사각형 두 개, `area`, jit 메모리 최적화 | 201.5 MB | 1.57 GB |
| 회전 사각형 두 개, `area`, 즉시 실행 | — | 1.58 GB |
| 회전 사각형 하나, `area`, jit 기본 | 10.1 MB | (컴파일만) |
| 회전 사각형 하나, `area`, jit 메모리 최적화 | 21.7 MB | (컴파일만) |
| 회전 없는 사각형 두 개, `separable`, 폭 미분, jit 기본 | 0.8 MB | 0.46 GB |

해상도 10에서 사각형 두 개의 공개 temp는 기본 스케줄러 2321.5 MB, 메모리 최적화 453.2 MB입니다.

`_shape_partition`은 면적 도형이 두 개 이상인 층을 `_painted_cell_materials`로 보내며, 이
도형들은 사각형이어야 합니다. 사각형·원·타원 하나만 있는 층은 `_shape_cell_fractions`로 가고,
표의 "회전 사각형 하나" 행은 그 경로가 영향을 받지 않음을 보여 줍니다. 원과 타원은 측정하지 않았습니다.
`separable`은 회전을 거부하므로, 한 층에서 사각형 두 개 이상을 회전하거나 그 각도로 미분하는
사용자는 CPU에서 외부 jit을 쓸 때마다 영향을 받는 경로에 있습니다.

역방향 모드는 해상도 6에서 한 번, 컴파일만 해서 측정했습니다. 연구의
`jax.jit(jax.grad(lambda p: jnp.sum(jnp.abs(response(p, 6))**2)))`에서 temp는 기본 스케줄러 361 MB,
메모리 최적화 203 MB입니다. 즉시 실행 최대값은 측정하지 않았습니다. GPU는 측정하지 않았으며
XLA:GPU는 다른 스케줄러를 씁니다.

## 수정 후보

| 수정 | 범위 | 예상 효과 |
|---|---|---|
| `jax.jit(g, compiler_options={'xla_cpu_scheduler_type': 'CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED'})` | 해상도 20 테스트처럼 jit 호출 하나 | 측정: 해상도 6에서 jvp temp 1031 → 42 MB로 플래그와 같음. 해상도 20에서도 외부 jit을 다시 쓸 수 있을 것으로 예상(플래그로 2.44 GB 측정) |
| `XLA_FLAGS`에 `--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED` 추가(기존 플래그 유지) | 기존 Eigen 플래그처럼 프로세스 또는 CI job | job의 모든 jit에 프로그램별로 같은 효과. 여기서 미분 실행 시간은 같거나 짧았으나 primal은 해상도 8과 10에서 최대 38% 느렸고, 다른 프로그램은 측정하지 않음 |
| `_painted_cell_materials`에서 칠하기의 동시 생존 집합을 제한: Yee 오프셋 네 개나 변 그룹을 백엔드 루프로 순차 실행하거나, `[E,9,ny,nx,9,2]`를 만들지 않고 `cdf`/`weight` 창 곱을 축약 계산에 바로 합쳐 넣기 | 라이브러리, 모든 스케줄러와 사용자 | 어떤 호출자에서도 temp가 메모리 최적화 수치에 가까워질 것으로 예상. 구현·측정하지 않음. 백엔드 중립을 유지하고 값을 바꾸지 않아야 함 |
| YNN 퓨전 끄기 | 프로세스 | 일부: 해상도 6에서 jac temp 1333 → 433 MB. 실험적 플래그이며 권장하지 않음 |
| 사용자용 옵션 문서화 | 공개 사용자 | 완료: [주기 적층 FDTD 가이드](../../guide/fdtd/periodic-problems.md)의 "CPU에서 외부 `jax.jit` 아래의 메모리" 절. 코드 변경 없음. 사용자가 자신의 jit에서 선택 |

## 재현

`.[jax]`를 설치한 소스 checkout, `3f2ad83` 이후에서 실행합니다.

```python
import jax, numpy as np
jax.config.update('jax_enable_x64', True)
from studies.cross_solver.painted_shapes.fdtd_painted_shapes import response

p, resolution = np.array([.37, .055, .22]), 6
f = lambda p: response(p, resolution)
g = lambda p: jax.jvp(f, (p,), (np.array([1., 0., 0.]),))[1]   # or (f(p), jax.jacfwd(f)(p))
compiled = jax.jit(g).lower(p).compile()
print(compiled.memory_analysis().temp_size_in_bytes)
```

`XLA_FLAGS` 없이 한 번, `XLA_FLAGS=--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED`로
한 번 실행합니다. `--xla_dump_to=<dir> --xla_dump_hlo_as_text`를 더하면 버퍼 보고서와 스케줄된
모듈을 얻습니다. 최대 상주 메모리는 `jax.block_until_ready(compiled(p))` 뒤의
`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`입니다.
