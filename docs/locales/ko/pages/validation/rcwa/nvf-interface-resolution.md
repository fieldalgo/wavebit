# 미분 가능한 메시로 실제 재료 계면 분해하기

이 페이지는 리비전 `532489b`(`908f60b`로 반영)까지 쓰인 독립 도약 밀도를 그 과거의 측정과 재현 명령까지 포함해
기록합니다. 현재의 [분리 직사각형 구성](nvf-separated-rectangles.md)은 겹치는 계면을 합치는 부호 있는 로그 도약
밀도를 씁니다. 그 유한 셀 결과는 아래 표와 다릅니다. 이전 CDF는 직접 비교를 위해 공유 재료 등급화 모듈에
`independent_graded_fractions`로 남아 있습니다. 과거의 셀 전체를 재현하려면 이전 리비전이 필요합니다.

`studies/rcwa/nvf_interface_resolution/nvf_interface_resolution.py`는 [연속 플럭스 직사각형 모델](nvf-continuous-flux.md)에
선택해서 켜는, 재료로 정의되는 메시를 더합니다. 이 메시는 미지수의 개수와 연결 구조를 고정한 채, 내부 x 노드를
실제 유전율 도약 근처에 모읍니다. 아래의 혼합 직사각형에서 상대 정적 에너지 포함 구간(enclosure)은 같은 미지수
개수에서 약 64% 좁아집니다. 독립적으로 피팅한 보조장에 대한 차수 16 s 투과율의 남은 차이는 미지수 5183개에서
약 26% 줄어듭니다.

이것은 직사각형 연구 확장입니다. 차수 20 비교에는 여전히 바깥 푸리에 절단 오차가 보입니다. 일반 기하,
독립적으로 수렴한 광학 투과율, 운영 통합은 아직 열려 있습니다.

이후의 [방향별 수렴 연구](nvf-directional-convergence.md)는 x와 y 푸리에 차수를 따로 바꿉니다. 그 연구는 더 효과적인
셀 배분도 찾아냅니다. 몸체 y 세분은 앞의 x4 모델보다 적은 미지수로 표에 실린 차수 16 s 차이를 약 60% 더
줄입니다.

## 재료 도약으로 정해지는 메시

직사각형 외곽을 r in [0,1]로 정규화합니다. 실제이거나 중복된 내부 이음매 a_j에 대해 다음을 정의합니다.

```
w_j = strength/pi * ((epsilon[j+1]-epsilon[j]) /
                      (epsilon[j+1]+epsilon[j]))^2
C(r) = r + sum_j w_j * (atan((r-a_j)/radius) - atan(-a_j/radius))
C'(r) = 1 + sum_j w_j*radius / ((r-a_j)^2 + radius^2)
```

고정된 원래 비율 q_i마다 새 노드 r_i는 `C(r_i) = q_i*C(1)`을 풉니다. 지원하는 양의 재료에서는 밀도가 엄격히
양수입니다. 따라서 이 사상은 노드를 만들거나 지우지 않고 순서와 끝점을 보존합니다. 외곽 안의 노드만 움직이고,
기존의 외부 메시는 그대로 남습니다. 어떤 이음매도 노드로 승격되지 않으며, 이음매가 메시를 지나갈 때 정렬이나
위상 전환도 일어나지 않습니다.

재료가 같은 이음매는 대수적으로 정확히 0을 기여합니다. 그런 이음매를 더해도, 다른 실제 도약이 있을 때를 포함해
메시와 필드 계수가 바뀌지 않습니다. 구현은 기여를 순서대로 누적하므로, 0인 항을 더해도 0이 아닌 항의 부동소수점
묶음이 보존됩니다. 재료가 같은지에 따라 분기하지 않습니다. 재료는 메시와 바탕 PDE 양쪽에서 살아 있으므로, 대비가
0일 때 메시 미분이 0이어도 물리적인 재료 미분은 버려지지 않습니다.

대비를 제곱하므로 정확한 산술에서는 메시 밀도가 유전율의 역수에 대해서도 불변입니다. 주(primal) 계산과 보완
계산은 같은 해상도 설정을 씁니다. 계산한 두 에너지는 별개의 변분 상한/하한 제어로 남으며, 어느 쪽도 광학 오차
한계가 아닙니다.

이것은 이전의 고정 메시 구성을 확장합니다. 노드 위치는 이제 물리적인 재료 함수에 의존할 수 있지만, 임의의
기본 도형 분할로는 바꿀 수 없습니다. 연속 몸체 플럭스와 재료 도약 적분은 유지됩니다. 이 모델 안에서는
카르테시안 Li 계산을 쓰지 않습니다.

## 근 풀이와 미분

이분법 18단계로 각 근의 구간을 좁힌 뒤, 일반적인 뉴턴 단계를 4번 수행합니다. NumPy와 JAX는 같은 순방향 방정식을
계산합니다. 미분은 재료와 이음매 의존성을 포함해 뉴턴 갱신을 그대로 통과합니다. 기하 그래디언트를 멈추거나
사용자 정의 역방향 보정을 넣지 않습니다. 근 잔차는 2e-12 미만이어야 하고 노드는 엄격한 순서를 유지해야 합니다.
잘못된 NumPy 메시는 예외를 일으키고, 잘못된 추적 갱신은 기존의 비유한 결과 보호 장치로 거부됩니다.

`grading_strength=8`과 `grading_radius=.01`은 명시적인 연구 설정입니다. 반지름은 외곽 폭에 대한 비율이므로 여기서는
3.5 nm입니다. 이 값들은 광학 목표에 피팅하지 않고 정적 포함 구간을 써서 골랐으며, 여러 구조에 걸쳐 최적이라고
주장하지 않습니다. 원래의 전역 플럭스 기저에서 지원하는 설정 범위는 강도 0부터 32까지, 반지름 .005부터 .1까지입니다.
이후의 [국소화 기저](nvf-local-flux.md)는 같은 시험 공간과 바뀌지 않은 잔차/물리 기준으로 .0001까지의 반지름을
지원합니다. 이것은 혼합 재료 접촉을 더 세밀하게 분해할 때 마주친 조건수 문제를 다루며, 그 더 세밀한 결과가
거친 설정을 보증하지는 않습니다. 강도 0은 원래의 고정 메시 모델을 정확히 재현합니다. 원래의
`ContinuousFluxGapModel`은 제어용으로 바뀌지 않은 채 남아 있습니다.

독립적인 메시 검사는 SciPy의 구간 Brent 솔버를 씁니다. JIT 메시 야코비안은 대비 0을 포함해, 독립적으로 푼 그 근들의
할선과 일치합니다. 전체 모델은 공유하는 독립 닫힌 막대 정적 검사와 광학 검사, 두 물리 접촉 미분, 움직이는 실제
이음매 미분, 평행 이동한 분할 불변성, 살아 있는 재료 미분, 잘못된 입력 검사도 통과합니다. 중복 분할을 끼워 넣은
여러 실제 도약은 틈 0, 17, 400 nm에서 비트 단위로 동일한 계수를 만듭니다.

## 같은 차원에서의 정적 비교

고정 예제는 주기 700 nm, 폭 350 nm, 틈 17 nm, 이음매 31 nm, 배경 1 안의 재료 4/9를 가집니다. 표의 모든 연속 플럭스
모델은 기본 몸체 행 8, 몸체 y 세분 2, 틈 다항식 차수 4, 가장자리 비율 1e-4를 씁니다.

| x 세분 | 스칼라 미지수 | 고정 메시 상대 포함 구간 | 재료 등급화 상대 포함 구간 |
| ---: | ---: | ---: | ---: |
| 2 | 2591 | 0.001024637861 | 0.000367876905 |
| 4 | 5183 | 0.000771780974 | 0.000280984807 |

두 경우 모두 약 64% 줄어듭니다. x2에서 새 대각 한계는 다음과 같습니다.

```
1.670761658413817 <= Uxx <= 1.670793592106591
3.183817928828364 <= Uyy <= 3.184989181912976
```

[독립적으로 피팅한 FEM 셀](nvf-gap-convergence.md#independent-static-reference)은 상대 폭 7.93665e-7로 훨씬 좁은 포함
구간을 가집니다. 그 대각 구간은 두 후보 구간 안에 모두 들어갑니다. 따라서 개선한 메시는 보조장 불확실성을 줄이지만
없애지는 못했습니다. 기록한 실행에서 등급화한 주/역수 잔차는 1.5e-12 미만입니다.

## 광학 비교와 남은 바깥 오차

파장 900 nm, 두께 300 nm, 입사각 .2 라디안, 입사 유전율 1, 기판 유전율 2.25를 씁니다. 바깥 차수 16에서:

| 보조장 / 인수분해 | 스칼라 셀 미지수 | p T | s T |
| --- | ---: | ---: | ---: |
| 피팅한 FEM 필드, 에너지 닫힘 | 43649 | 0.907165915611 | 0.941480649651 |
| 고정 메시, x2 | 2591 | 0.907171160845 | 0.941544331905 |
| 고정 메시, x4 | 5183 | 0.907170049872 | 0.941540519833 |
| 재료 등급화 메시, x2 | 2591 | 0.907169465216 | 0.941527092395 |
| 재료 등급화 메시, x4 | 5183 | 0.907168707460 | 0.941525111539 |
| 독립 카르테시안 구간 | — | 0.907193571579 | 0.941468107350 |

x4에서 피팅한 셀 투영에 대한 s 차이는 5.9870e-5에서 4.4462e-5로, 약 26% 줄어듭니다. p 차이도 4.1343e-6에서
2.7918e-6으로 줄어듭니다. 등급화한 x2 모델조차 셀 미지수를 절반만 쓰면서 두 편광 모두에서 고정 x4 모델보다 더
가깝습니다. 이것은 같은 차수의 보조장 제어에 대한 차이이며, 보증된 투과율 오차가 아닙니다. 모든 경로는 같은
Maxwell 산란 계산기를 씁니다.

독립 제어도 바깥 차수 20까지 확장했습니다.

| 바깥 차수 | 피팅 셀 p T | 피팅 셀 s T | 카르테시안 p T | 카르테시안 s T |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.907165915611 | 0.941480649651 | 0.907193571579 | 0.941468107350 |
| 20 | 0.906915509832 | 0.941563020477 | 0.906934177474 | 0.941561099135 |

차수 20에서 둘의 차이는 p에서 1.87e-5, s에서 1.92e-6입니다. 그러나 피팅 셀 투과율은 차수 16에서 20으로 가며
여전히 2.50e-4 / 8.24e-5만큼 바뀝니다. 한 차수에서 정식화끼리 일치한다고 해서 수렴이 확립되지는 않습니다. cap-40
FEM 투영은 구적점을 24개에서 28개로 늘릴 때 최대 2.98e-15만큼 바뀌므로, 이 투영 검사는 바깥 절단과 분리됩니다.
이 측정들에서 외삽한 광학 기준값은 받아들이지 않습니다.

## 재현과 검증

저장소 루트에서 실행하세요. 선택 사항인 FEM 내보내기에는 NGSolve가 필요합니다.

생성되는 투영 파일을 위해 `NVF_CACHE`를 이미 있는 임시 작업 디렉터리로 설정하세요.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_interface_resolution.nvf_interface_resolution --order 2 --gaps 0 17 --seams 0 31 --grading-strength 8 --grading-radius .01
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 16 --meshes 2:8:2 4:8:2 --gap-degrees 4 --grading 8:.01
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 16 --meshes 2:8:2 --gap-degrees 4
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_reference.nvf_gap_reference --cap 40 --h .04 --degree 8 --quadrature 24 --output "$NVF_CACHE/wavebit-gap-reference-cap40.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 20 --meshes --projection "$NVF_CACHE/wavebit-gap-reference-cap40.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_interface_resolution.py tests/rcwa/test_nvf_gap_convergence.py tests/rcwa/test_nvf_gap_reference.py tests/rcwa/test_nvf_continuous_flux.py tests/rcwa/test_nvf_harmonic_coordinates.py tests/rcwa/test_nvf_partitioned_gap_energy.py tests/rcwa/test_nvf_thin_gap_energy.py tests/rcwa/test_nvf_cell_bounds.py tests/rcwa/test_nvf_cell_corrector.py tests/rcwa/test_nvf_harmonic_energy.py tests/rcwa/test_nvf_harmonic_energy_jax.py tests/rcwa/test_resonant_nvf_probe.py tests/rcwa/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

원시 측정은 `studies/rcwa/nvf_interface_resolution/records/nvf_interface_resolution.log`,
`nvf_interface_resolution_jax.log`, `nvf_gap_reference_cap40.log`, `nvf_gap_order20.log`에 있습니다. 이전의 고정 메시와
피팅 셀 차수 16 측정은 [앞선 연구의 로그](nvf-gap-convergence.md#reproduction-and-validation)에 남아 있습니다.

2026-09-08 검증: 결합 선택에서 **168개 통과, 8개 건너뜀, 5개 xfail**입니다. 여기에는 등급화 모델을 위한 새 전용
시험 7개와 새 공유 매개변수화 25개가 포함됩니다. 건너뛴 8개는 선택 사항인 FEM 환경이 필요하며, 그 시험들은 앞선
기준 작업에서 통과했습니다. 엄격한 예상 실패 5개는 여전히 운영 모델이나 이전 모델에 붙어 있습니다. 기존 물리
허용 오차, 운영 경로 지정, 고유값 미분 정규화는 하나도 바꾸지 않았습니다.
