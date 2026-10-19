# 면적 적분 도형 FDTD 미분의 외부 jit 메모리

작성일: 2026-10-19.

대상: `src/wavebit/fdtd/_periodic/scene.py`, `src/wavebit/fdtd/_periodic/runtime.py`,
`studies/cross_solver/painted_shapes/fdtd_painted_shapes.py`, `tests/fdtd/test_fdtd_painted_optics.py`.

**요약.** 추가 메모리는 XLA:CPU가 컴파일된 프로그램의 실행 순서를 정하는 방식에서 생깁니다.
시간 루프나 프로그램 크기가 원인이 아닙니다. `_painted_cell_materials`의 면적 적분 도형
칠하기는 `[edges, images, ny, nx, segments, gauss]` 모양의 float64 중간값을 많이 만들고,
각 중간값은 XLA:CPU가 YNNPACK 퓨전으로 묶는 축약 연산으로 들어갑니다. 퓨전의 입력은
퓨전이 실행되기 전에 전체 버퍼로 존재해야 합니다. jaxlib 0.11.0의 XLA:CPU 기본 스케줄러는
동시성 최적화 스케줄러이며, 프로그램의 모든 칠하기 중간값을 먼저 계산하고 축약을 마지막에
실행합니다. 그래서 이 중간값이 모두 같은 순간에 살아 있습니다. 전진 모드 미분은 이런 중간값의
수를 늘립니다. 해상도 6에서 primal은 297개, JVP 한 열은 1,071개입니다. 즉시 실행은 연산을
하나씩 보내므로 칠하기 한 번 분량 이상의 작업 메모리를 들고 있지 않습니다.
`xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED`를 쓰면 같은 프로그램의
해상도 20 `jit(value, jacfwd)` XLA 임시 할당이 17.5배 작아집니다(11.67 GB → 0.67 GB).
최대 상주 메모리는 2.44 GB로, 기본 스케줄러에서 기록된 13.68 GB와 비교됩니다. 야코비안은
즉시 실행 결과와 1.1e-15 이내로 일치합니다. `shape_integration='area'`로 공개
`wavebit.fdtd.periodic.run(..., backend='jax')`를 쓰는 사용자도 똑같이 영향을 받으며,
기본값 `'separable'` 경로는 영향을 받지 않습니다.

## 구성

2026-10-19, `3f2ad83`에서 측정했습니다. Linux x86-64 클라우드 컨테이너(Intel Xeon @ 2.80 GHz,
vCPU 4개, 15 GiB), Python 3.13.16, NumPy 2.5.4, JAX/jaxlib 0.11.0이며 전부 float64입니다.
작업은 `p = [.37, .055, .22]`, 기본 지속 시간 60의 `fdtd_painted_shapes.response(p, resolution)`입니다.
각 행은 별도 프로세스이고 프로세스는 한 번에 하나씩 실행했습니다. 최대값은 `ru_maxrss`입니다.
*Temp*는 `jax.jit(g).lower(p).compile().memory_analysis().temp_size_in_bytes`이며,
XLA:CPU는 실행마다 이를 버퍼 하나로 할당합니다. 메모리 최적화 행은
`XLA_FLAGS=--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED`만 설정합니다.
`jac`은 연구와 slow 테스트가 쓴 형태인 `jit(lambda p: (f(p), jax.jacfwd(f)(p)))`이고,
`jvp`는 각도 방향의 전진 미분 한 열입니다.

## 임시 메모리 크기는 프로그램이 아니라 스케줄을 따름

| 해상도 | 경우 | 즉시 실행 최대 | jit 기본: temp / 최대 | jit 메모리 최적화: temp / 최대 |
|---:|---|---:|---:|---:|
| 6 | primal | 1.31 GB | 0.28 / 1.03 GB | 0.015 / 0.80 GB |
| 6 | jvp | 1.42 GB | 1.03 / 2.40 GB | 0.042 / 1.49 GB |
| 6 | jac | 2.37 GB | 1.33 / 2.93 GB | 0.076 / 1.72 GB |
| 8 | jvp | 1.47 GB | 1.61 / 2.92 GB | 0.065 / 1.43 GB |
| 8 | jac | 2.41 GB | 2.08 / 3.76 GB | 0.117 / 1.82 GB |
| 10 | jvp | 1.42 GB | 2.32 / 3.62 GB | 0.092 / 1.47 GB |
| 10 | jac | 2.49 GB | 3.00 / 4.68 GB | 0.166 / 1.90 GB |
| 20 | jac | 2.67 GB | 11.67 GB / 실행 안 함 | 0.67 / 2.44 GB |

두 스케줄러는 같은 HLO 모듈을 컴파일하므로 실행 전 컴파일 최대값이 같습니다(미분 프로그램에서
1.3–1.6 GB). 각 jit 최대값은 대략 그 컴파일 최대값에 temp를 더한 값입니다. temp는 대략
`nx·ny`에 비례해 커집니다(해상도 6에서 4×4 셀, 해상도 20의 temp는 해상도 6의 8.8배).
해상도 20의 기본 temp와 컴파일 최대값의 합 약 13.3 GB는 이 문서 이전에 기록된 13.68 GB를
설명합니다. 해상도 20 기본 프로그램은 이 기계에 들어가지 않으므로 컴파일만 하고 실행하지
않았습니다. slow 테스트 docstring이 해상도 30에 대해 기록한 22.9 GB 단일 할당은 이 temp
버퍼와 부합합니다. 값은 스케줄러에 의존하지 않습니다. 해상도 6–10에서 기본과 메모리 최적화
실행의 출력 합은 마지막 자리까지 같고, 모든 jit 결과는 즉시 실행과 약 1e-15 이내로 일치합니다.
실행 시간은 메모리 최적화 스케줄러에서 같거나 짧았습니다(jvp 해상도 6, 8, 10에서 5.1 대 5.5 s,
5.6 대 7.5 s, 7.5 대 15.2 s, 해상도 20 jac은 92 s).

CI 설정 `--xla_cpu_multi_thread_eigen=false`를 더해도 temp는 바뀌지 않습니다(해상도 6 jvp에서
1,031,495,480바이트). `CPU_SCHEDULER_TYPE_CONCURRENCY_OPTIMIZED`는 기본값의 temp를 바이트까지
재현하므로, 이 jaxlib에서 기본값은 동시성 최적화 스케줄러입니다.

## 버퍼의 정체

해상도 6의 JVP 한 열에 대한 XLA 버퍼 할당 보고서
(`--xla_dump_to`, `*-memory-usage-report.txt`)를 모양별로 묶은 결과입니다.

```text
primal: 239.2 MiB   295 x f64[41,9,4,4,9,2]     jvp: 867.5 MiB  1070 x f64[41,9,4,4,9,2]
         17.4 MiB    43 x f64[41,9,4,4,9]             82.3 MiB   203 x f64[41,9,4,4,9]
```

이 모양은 `_painted_cell_materials`(`scene.py` 617–629행, `cdf`/`weight` 창 루프)의
`[변 조각, 3×3 주기 이미지, ny, nx, 절단 구간, 가우스 노드 2개]`입니다. 이 버퍼의 HLO
스택 프레임은 해당 행을 가리키고, 연산 이름은 `jvp(jit(clip))/select_n`, `min`, `mul`,
`add`입니다. 소비자는 216개의 YNN 퓨전입니다. 전형적인 하나는 다음과 같습니다.

```text
%ynn_fusion.40 = f64[4,4,2] fusion(<40 operands>), calls=%fused_computation.40,
    backend_config={"fusion_config":{"kind":"__ynn_fusion"}}
  fused_computation.40: 39 parameters (31 x f64[41,9,4,4,9,2]), 30 multiply, 13 add,
    ROOT reduce_sum = f64[4,4,2] reduce(...), dimensions={0,1}
```

이것은 `total + sum(value*jump, axis=(0, 1))`입니다. JVP에서는 곱의 미분 법칙이 이를 여러
항으로 나눕니다. 1,071개의 큰 값은 모두 그것을 쓰는 첫 축약이 실행되기 전에 정의됩니다.
스케줄된 entry 계산에 대한 생존 분석으로 동시에 살아 있는 수를 세었습니다.

| 스케줄 (jvp, 해상도 6) | 큰 버퍼 | 동시 생존 | Temp |
|---|---:|---:|---:|
| 기본 (동시성 최적화) | 1,071 | 1,071 | 1031 MB |
| 메모리 최적화 | 1,071 | 37 | 42 MB |

primal도 같은 구조이며 큰 버퍼 297개가 모두 동시에 살아 있습니다(276 MB).
YNN 퓨전을 끄면(`--xla_cpu_experimental_ynn_fusion_type=`) 생산자가 일반 축약으로 퓨전될 수
있어 일부만 개선됩니다. 해상도 6에서 jvp 107 MB, jac 433 MB입니다.

## 배제한 원인

- **시간 루프의 접선 carry.** 프로그램에는 primal과 JVP 모두 반복 횟수가 알려진(32, 7, 2)
  `while` 루프 13개가 있으며, 펼쳐지지 않았습니다. 가장 큰 루프 carry는 0.45 MB
  (접선을 포함한 `c128[3,42,4,4]` 장)입니다. checkpoint된 scan은 메모리가 쓰이는 곳이 아닙니다.
- **캡처된 상수.** JVP 모듈의 HLO 상수는 모두 합쳐 0.26 MB입니다.
- **주된 비용으로서의 컴파일.** 컴파일러는 상주 메모리 1.3–1.6 GB를 차지하며 해상도에 거의
  의존하지 않습니다. 이것이 "해상도에 따라 거의 줄지 않는" 부분입니다. 커지는 부분, 즉 미분
  경우를 즉시 실행의 4–5배로 만드는 부분은 실행 temp입니다.
- **`_eager_scan`.** 외부 jit 아래에서는 모든 leaf가 tracer이므로 일반 `lax.scan`으로
  되돌아갑니다. 즉시 실행 `jacfwd`도 클로저 값이 tracer이므로 같은 경로를 탑니다.

slow 테스트 docstring은 지속 시간 40에서 해상도 6, 8, 10에 대해 5.3, 6.2, 7.2 GB를 기록합니다.
이 값은 즉시 실행 야코비안을 jit보다 먼저 실행하는 프로세스에서 나온 것입니다. 여기서
jit만 실행하는 프로세스는 지속 시간 60에서 2.93, 3.76, 4.68 GB를 측정합니다. 즉시 실행
최대값의 차이(해상도 20에서 여기서는 2.67 GB, 기록은 3.57 GB)는 조사하지 않았습니다.

## 공개 사용자

공개 periodic API에서도 같은 프로그램이 생깁니다. `Rectangle` 영역 두 개의 회전각에 대한
전진 미분을 `fdtd.periodic.Numerics(spacing=1/6, shape_integration='area',
shape_smoothing=.05, ...)`와 `run(..., backend='jax')`로 `jax.jit(jax.jvp(...))` 아래에서
계산한 결과입니다.

| 공개 경로, 해상도 6 | Temp | 최대 |
|---|---:|---:|
| `area`, jit 기본 | 1031.5 MB | 2.35 GB |
| `area`, jit 메모리 최적화 | 201.5 MB | 1.57 GB |
| `area`, 즉시 실행 | — | 1.58 GB |
| `separable` (회전 없음, 폭 미분), jit 기본 | 0.8 MB | 0.46 GB |

해상도 10에서 공개 `area` temp는 기본 스케줄러 2321.5 MB, 메모리 최적화 453.2 MB입니다.
`separable`은 회전한 사각형을 거부하므로, CPU에서 외부 jit 아래 사각형을 회전하거나 그 각도로
미분하는 모든 공개 사용자는 영향 경로에 있습니다. GPU는 측정하지 않았으며 XLA:GPU는 다른
스케줄러를 씁니다.

## 수정 후보

| 수정 | 범위 | 예상 효과 |
|---|---|---|
| `jax.jit(g, compiler_options={'xla_cpu_scheduler_type': 'CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED'})` | slow 테스트처럼 jit 호출 하나 | 측정: 해상도 6에서 jvp temp 1031 → 42 MB로 플래그 설정과 같음. 해상도 20에서 외부 jit을 다시 허용할 수 있음(2.44 GB 측정) |
| `XLA_FLAGS=--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED` | 기존 Eigen 플래그처럼 프로세스 또는 CI job | job의 모든 jit에 프로그램별로 같은 효과. 이 프로그램들에서 실행 시간은 같거나 짧았으나 다른 프로그램에 대한 영향은 측정하지 않음 |
| `_painted_cell_materials`에서 칠하기의 동시 생존 집합을 제한: Yee 오프셋 네 개나 변 그룹을 백엔드 루프로 순차 실행하거나, `[E,9,ny,nx,9,2]`를 만들지 않고 `cdf`/`weight` 창 곱을 축약 안에서 수축 | 라이브러리, 모든 스케줄러와 사용자 | 어떤 호출자에서도 temp가 메모리 최적화 수치에 가까워질 것으로 예상. 구현·측정하지 않음. 백엔드 중립을 유지하고 값을 바꾸지 않아야 함 |
| YNN 퓨전 끄기 | 프로세스 | 일부: 해상도 6에서 jac temp 1333 → 433 MB. 실험적 플래그이며 권장하지 않음 |
| 사용자용 옵션 문서화 | 공개 사용자 | 완료: [주기 적층 FDTD 가이드](../../guide/fdtd/periodic-problems.md)의 "Memory under an outer `jax.jit` on CPU". 코드 변경 없음. 사용자가 자신의 jit에서 선택 |
