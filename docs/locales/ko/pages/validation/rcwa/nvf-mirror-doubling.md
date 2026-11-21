# NVF 거울 섹터의 미분 가능한 산란 배가

[거울 닫힘](nvf-mirror-convergence.md)과 [고유벡터 없는 전파기](nvf-scattering-doubling.md)가 이제 NumPy와
JAX에서 함께 동작합니다. 유한 NVF 필드와 Gram 닫힘은 바뀌지 않았습니다. 새 경로는 정확히 스치듯 진행하는 층
모드를 처리하고, 접촉 미분과 재료 미분 검사를 유지하며, 같은 배가 계산의 비용을 줄입니다.

기록한 CPU 고정 예제에서 차수 16 NumPy 계산은 패리티 축소 없이 47.35초에서, 축소를 쓰면 12.90초로 줄어듭니다.
차수 8에서 JIT 값과 재료 스케일 미분 하나를 함께 구하는 계산은 예열 후 중앙값 1.922초에서 0.513초로
줄어듭니다. 이것은 배가 방법 안에서 측정한 감소입니다. 시험한 NumPy 순방향 풀이에서는 기존 모드 기반 거울
솔버가 여전히 더 빠르며, 계속 기본값으로 남습니다. 운영 NVF 경로 지정은 바뀌지 않았습니다.

## 서로 다른 두 가지 대칭 축소

원래의 y 반사 제한에서 전기 성분은 공간 `[Ex_a, Ey_b]`에, 자기 성분은 `[Hx_b, Hy_a]`에 있습니다. y 차수 0은
짝수 공간에만 나타나므로 두 성분 공간의 차원이 다릅니다. 새 코드는 L, R, LR, RL을 만들 때 이 구분을
유지합니다. 두 곱 모두 4차 파수 항의 상쇄를 피하도록 전개한 식을 씁니다. 시험은 홀수 공간이 비어 있는 경우를
포함해, 네 인수 모두를 축소하지 않은 연산자의 명시적 정규직교 투영과 비교합니다.

자기 성분을 다음 식으로 전기 순서로 바꿉니다.

```text
Hc = [-i Hy_a, i Hx_b].
```

이것은 유니터리 순열이자 위상 변화입니다. 이 변환은 표준 어드미턴스를 항등 행렬로, 플럭스를 `Re(E.H Hc)`로
보냅니다. R, L, RL에도 같은 좌표 변환을 적용해야 합니다. 전기 LR 곱은 이미 올바른 공간에 있습니다. 이 순열에는
조밀한 기저 변환 곱셈이 필요 없습니다.

두 번째 축소는 z 방향으로 균일한 층에서 나옵니다. 그 Maxwell 생성자는 `(E,Hc)`에서 비대각이며, z와 Hc의 부호를
함께 뒤집어도 불변입니다. 따라서 동일한 표준 기준 포트를 쓰면 그 산란 행렬은 같은 반사 블록 r과 같은 투과 블록
t를 가집니다. 이것은 공간 대칭이며, 행렬 인수들이 교환된다는 가정이 아닙니다. 이 대칭은 서로 다른 물리적
외부 매질을 맞추기 전에 적용됩니다.

공유 배가 커널은 이제 r과 t만 저장하고 계산합니다. 한 단계는 다음과 같습니다.

```text
C = solve(I - r r, t)
r_new = r + (t r) C
t_new = t C.
```

처음 짧은 조각 풀이는 반사와 투과 우변에 대해 하나의 인수분해를 공유합니다. 테일러 급수, 보수적인 국소 꼬리
기준, 무손실 전력 기준은 이전 허용 오차를 유지합니다. 독립적인 전체 행렬 지수가, 교환하지 않는 복소 구성
행렬과 세 가지 조각 수에 대해 **네 개의 복소 산란 블록 전부**를 위상까지 포함해 검사합니다. 서로 다른 층으로
이루어진 임의의 적층에는 이 축소를 가정하지 않으며, 일반적인 네 블록 `star` 함수는 따로 남아 있습니다.

## 섹터 차원에서의 외부 정합

물리적인 위쪽과 아래쪽 어드미턴스를 같은 플럭스 좌표로 옮깁니다. 각 포트에 대해 `P=(I+Y)/2`, `M=(I-Y)/2`로
정의합니다. NumPy 경로는 이 행렬들을 희소하게 유지합니다. 정합 계산은 아래쪽 포트를 먼저 소거합니다.

```text
O = solve(P_bottom - r M_bottom, t)
R_effective = r + t M_bottom O
reflected = solve(P_top - R_effective M_top,
                  (R_effective P_top - M_top) incoming)
transmitted = O (P_top incoming + M_top reflected).
```

각 조밀 풀이는 단일 섹터 차원 H에 머물러, 2H 경계 행렬을 피합니다. JAX는 외부 사상에 조밀 상수를 쓰고 일반적인
행렬 미분을 씁니다. 살아 있는 필드 변환 역행렬, 에너지 Gram 행렬, 재료 역행렬, 전파와 정합은 모두 미분하는
계산 안에 남아 있습니다.

## 범위와 검사

계수 필드는 매개변수가 움직이는 동안 y 반사 대칭을 유지해야 합니다. 원래의 반사 허용 오차 1e-10, 실수 필드
켤레성 허용 오차 1e-9, 에르미트성 허용 오차 1e-9, 무손실 전력 허용 오차 1e-9를 유지합니다. NumPy는 잘못된
입력을 거부합니다. JIT는 비대칭이거나 실수가 아닌 필드를 대칭화하지 않고 NaN을 반환합니다. JAX는 float64를
요구합니다. 이것으로 이전에 거부한 비구조 FEM 투영이 허용되지는 않습니다.

배가를 명시적으로 선택하면 정확히 스치듯 진행하는 **층** 모드가 이제 동작합니다. 정확히 스치듯 진행하는 외부
차수는 여전히 현재 포트 정합 계약의 범위 밖입니다. 일반 기하, 0이 아닌 y Bloch 입사, 흡수하는 층과 운영 통합은
이 구현이 제공하지 않습니다. 직사각형 셀 어댑터의 일반 기하 한계도 그대로 남아 있습니다.

시험은 독립적으로 인수분해한 카르테시안 극한에 대한 두 JIT 접촉 미분, 최종 7559개 미지수 셀 접촉 기준, 두
간격 크기에서 모드 기반 NumPy 할선과 비교한 살아 있는 틈/이음매/재료 미분, 같은 재료, 대비 0, 정확한 층
스침 미분을 다룹니다. 또한 방향별 NVF/카르테시안 계산과 반폭 0 NVF/카르테시안 계산을 축소하지 않은 대응
계산과 비교합니다. 기존 운영 경로의 예상 실패(expected failure)는 하나도 제거하지 않았습니다.

## 측정

저장한 셀은 앞선 차수 40 연구에서 쓴 것과 같은 혼합 직사각형입니다: 틈 17 nm, 폭 350 nm, 이음매 31 nm,
상대 유전율 4/9, 셀 미지수 7559개, 그리고 바뀌지 않은 광학 고정 예제입니다. 시간 비교는 유한 광학 닫힘과
전파를 포함하지만, 저장한 셀 필드를 재사용합니다.

| 차수 16 경로 | 초 | p T | s T |
| --- | ---: | ---: | ---: |
| 모드 기반 거울 | 8.559 | 0.907166395579 | 0.941482300436 |
| 배가 거울 | 12.898 | 0.907166395579 | 0.941482300434 |
| 배가 전체 | 47.348 | 0.907166395579 | 0.941482300433 |

측정한 배가 비율은 3.67입니다. 이 시간 측정은 보편적인 속도 향상이나, 빠른 순방향 스윕에서 배가가 모드 기반
솔버를 대체한다는 주장이 아닙니다.

차수 24에서 배가 6회와 테일러 항 12개는 p/s T로 0.906794640678716 / 0.941569618540630을 줍니다. 이 값들은
앞선 모드 기반 결과와 약 +1.36e-12 / -1.93e-13 차이가 납니다. 이것은 그 유지 차수에서 전파기를 검사하는
것이며, 알려진 바깥 절단 오차를 해결하지는 않습니다.

테일러 항 12개를 유지한 채 배가를 6회에서 7회로 늘리면 0.906794640677999 / 0.941569618539877을 얻습니다.
p/s 차이는 -7.17e-13 / -7.53e-13입니다. 두 조각 수에 걸친 가장 큰 전력 불균형은 1.76e-12입니다. 따라서 새로
측정한 가장 높은 차수에서 국소 급수 추정에 명시적인 조각 세분 검사가 함께 붙습니다.

JIT 비교는 **배경을 포함한 모든 층 유전율**을 균일하게 스케일할 때의 p/s 전력과 그 미분을 계산합니다. 재료를
균일하게 스케일하면 보조 셀 퍼텐셜이 바뀌지 않으므로, B 행은 고정되고 Gram 행과 재료 행만 스케일됩니다. 이렇게
하면 광학 닫힘/전파 비용만 분리되고 새 셀 풀이는 제외됩니다. 컴파일 뒤에 예열된 표본 7개를 측정했으며, 값과
미분 제어가 서로 일치합니다.

| 차수 | 전체 중앙값, 초 | 거울 중앙값, 초 | 비율 | 전체 컴파일 + 첫 실행, 초 | 거울 컴파일 + 첫 실행, 초 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.05045 | 0.02191 | 2.30 | 1.180 | 2.080 |
| 8 | 1.92245 | 0.51270 | 3.75 | 3.001 | 4.873 |

차수 8에서 전체/거울 값의 가장 큰 차이는 약 2.0e-13이고, 미분의 가장 큰 차이는 약 1.3e-12입니다. 두 미분
모두 바뀌지 않은 독립 모드 할선 기준도 만족합니다. 스칼라 하나에 대한 이 순방향 모드 벤치마크는 새로 푸는 셀이나
큰 기하 야코비안을 포함한 종단 간 최적화 속도를 추정하지 않습니다. 별도의 시험이 살아 있는 직사각형 셀을
역방향 모드로 미분합니다.

## 재현

앞선 거울 연구의 cap-80 투영을 쓰거나, 그 연구에 기록된 명령으로 만드세요. 순방향 선택은 명시적입니다.

생성되는 투영 파일을 위해 `NVF_CACHE`를 이미 있는 임시 작업 디렉터리로 설정하세요.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_mirror_convergence.nvf_mirror_convergence --projection "$NVF_CACHE/wavebit-mirror-cap80.npz" --orders 16 24 --propagation doubling --progress
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_mirror_convergence.nvf_mirror_convergence --projection "$NVF_CACHE/wavebit-mirror-cap80.npz" --orders 24 --propagation doubling --doublings 7 --progress
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_scattering_doubling.nvf_scattering_doubling --mirror --jax
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_scattering_doubling.nvf_scattering_doubling --mirror --jax --gap 0
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_mirror_doubling.nvf_mirror_doubling --projection "$NVF_CACHE/wavebit-mirror-cap80.npz" --orders 4 8 --repeats 7
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_scattering_doubling.py tests/rcwa/test_nvf_mirror_convergence.py tests/rcwa/test_nvf_mirror_cartesian.py tests/rcwa/test_nvf_mirror_doubling.py
```

JIT를 직접 쓰려면 `PolynomialBodyGapModel.coefficients(..., xp=jax.numpy)`로 살아 있는 계수를 얻어
`mirror_doubling_powers(coefficients, order, jax.numpy)`에 넘기세요. 이 함수는 p/s T, R과 전력 균형 진단값을
반환합니다. 운영 백엔드 전환은 필요 없습니다. 별도의 카르테시안 CLI도 `--propagation doubling`을 받으며, 자체의
독립적인 재료 인수분해를 유지합니다.

원시 기록은 `studies/rcwa/nvf_mirror_doubling/records/nvf_mirror_doubling.log`,
`nvf_mirror_doubling_timing.log`, `nvf_mirror_doubling_jax.log`, `nvf_mirror_doubling_gap.log`입니다. 마지막
기록은 혼합 재료 접촉에서 새 CLI의 살아 있는 JIT 기하/재료 미분도 검증합니다.

2026-09-09 검증: 위의 시험 파일 네 개에서 **54개 통과**, 여기에는 새 결합 경로 시험 13개와 앞선 제어 41개가
포함됩니다. Ruff 검사와 공백 검사도 통과합니다.
