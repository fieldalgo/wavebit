# 직사각형 NVF 연구 모델의 다항식 몸체 필드

`studies/rcwa/nvf_polynomial_body/nvf_polynomial_body.py` 는 직사각형 틈 모델의 몸체 안에 더 높은
차수의 conforming 퍼텐셜을 구현합니다. 스칼라 미지수가 똑같이 5039개일 때,
4차 몸체는 앞선 세분 선형 몸체보다 상대 정적 구간(enclosure)을 약 37% 좁힙니다. 바깥 차수
16에서, 독립적으로 맞춘 보조 필드와 비교한 s 투과 차이는 약 82% 줄어듭니다. 3차 대안은 미지수
3887개를 쓰며 이 비교도 개선합니다.

이것은 구현하고 시험한 **연구 모델** 이며, 제품 `wavebit.rcwa` NVF 디스패치의 갱신이 아닙니다.
일반 기하와 바깥 광학 수렴은 아직 열린 문제입니다. 맞춘 셀 비교는 독립적인 3D Maxwell 오차
증명서가 아닙니다.

## 트레이스를 보존하면서 순방향 필드 풍부하게 하기

앞선 [연속 플럭스 모델](nvf-continuous-flux.md)은 정규화한 y에서 주기적 P1 함수로 몸체 퍼텐셜을
표현합니다. 각 몸체 구간에서 r이 0부터 1까지 움직인다고 합시다. 두 끝점 함수를 유지하고 다음을
더합니다.

```
phi_left(r) = 1-r,     phi_right(r) = r
beta_k(r) = integral_0^r P_k(2*u-1) du,     k = 1,...,body_degree-1
beta_k(0) = beta_k(1) = 0
```

P_k는 Legendre 다항식입니다. 각 구간은 자기만의 내부 버블 함수를 가지며, 이웃한 구간은 여전히
끝점 미지수를 공유합니다. 퍼텐셜은 모든 몸체 계면에서 연속이고 주기성을 유지합니다. 더한 함수들은
두 틈 면에서 0이 되므로, 기존 틈 트레이스와 접촉 결합에 새로운 제약, 차단이나 위상 구조 변경이
필요하지 않습니다.

몸체 구간이 E개이고 차수가 p이면 주기적 몸체 함수는 E*p개입니다.
정규화한 몸체 좌표를 [-1/2,1/2] 안의 t로 쓰면, 같은 시험 퍼텐셜은 이제 다음과 같습니다.

```
chi_body(x,t) = v(x,t) - g*t*a(x)
v(x,t) = sum_A Phi_A(t)*v_A(x)
physical y = (1-g)*t
partial_y chi_body = (partial_t v - g*a)/(1-g)
```

아핀 x 들어 올림, 재료로 적분한 x 플럭스 기저와 유한 틈 버블 함수는 앞선 정식화 그대로입니다.
몸체 강성, 질량, 하중과 경사 결합은 풍부하게 한 Phi_A로 조립합니다. 기존 블록 셀 풀이는 이
행렬들을 쓰며, 같은 해석적 틈 0 행 스케일링과 실행 중 대각 평형화를 적용합니다. 수용 잔차는
여전히 원래 방정식의 잔차입니다.

국소 미분 적분은 Legendre 직교성을 씁니다.

```
integral_0^1 beta_k'(r)*beta_l'(r) dr = delta_kl/(2*k+1)
integral_0^1 phi_left/right'(r)*beta_k'(r) dr = 0
```

이 0인 결합을 정확히 0으로 유지하면 단항식 상쇄 오차를 아주 작은 접촉 구간 폭으로 나누는 일을
피할 수 있습니다. 나머지 국소 질량과 하중 항은 그에 대응하는 해석적 Legendre 적분을 씁니다.
이것은 같은 변분 방정식의 수치 조립이며, 정칙화나 맞춘 광학 항은 없습니다.

`body_degree=1` 은 앞선 `MaterialGradedGapModel` 계수를 정확히 재현합니다. 선택적으로 쓰는 이
클래스는 기본 차수 2이며 차수 1부터 4까지 지원합니다. 수렴 측정 도구는 기본값으로 차수 1을
유지합니다. x 방향 비균일화는 여전히 물리 재료 함수가 정의합니다. 몸체 버블 함수는 재료 동일성
분기를 새로 만들지 않으며, 기본 분할이 메시에 영향을 주게 하지도 않습니다.

## 실제 에너지 적분하기

각 몸체 구간 안에서 퍼텐셜은 r에 대한 p차 다항식이고, 그 y 미분은 p-1차입니다. 사영은 물리
기저 B와 그 에너지 `B.H epsilon B` 를 모두 적분하며, y 방향으로는 2*p차까지의 정확한 푸리에
모멘트를, x 방향으로는 기존 재료 도약 모멘트를 씁니다. 최종 계수 배열 여덟 개가 같은 유지 에너지
닫힘에 들어갑니다. 이제 몸체 사영과 틈 사영이 별도의 메서드이므로, 원래 모델은 선형 몸체 구현을
정확한 대조군으로 유지할 수 있습니다.

독립 테스트는 다항식 조립 공식을 쓰지 않고 Legendre 함수 값과 Gauss 구적법으로 국소 기저를
평가합니다. 두 번째 텐서 구적법은 물리 x/y 점에서 전체 필드와 가중 에너지를 계산합니다.
두 점검 모두 차수 2, 3, 4를 다룹니다. 중첩된 차수 세분이 원시/상보 에너지 경계를 좁히고 B의 평균
1을 보존하는지도 확인합니다.

2차 모델은 물려받은 중복 분할, 평행 이동한 형상, 독립 접촉, 닫힌 적층판, 재료와 실제 이음선
테스트를 통과합니다. 추가한 4차 JIT 점검은 틈, 이음선과 두 재료 미분을 다룹니다. 최종 4차
해상도는 조건수 경고를 오류로 격상한 상태에서 독립적인 p/s 접촉 기준도 통과합니다. 이 물리
허용 오차들은 하나도 완화하지 않았습니다.

최종 4차 설정에서 차수 7, epsilon 12.25 접촉 점검은 간격 1e-4 nm와 5e-5 nm에서 외삽한 독립
카르테시안 할선과 비교해 nm당 p/s 미분 차이 -2.55e-11 / 2.01e-8을 줍니다. 둘 다 기존 1e-7
기준을 충족합니다. 3차 대안도 이 독립 접촉 점검을 통과하며, 원시 출력에 그 값과 실제 셀 잔차가
남아 있습니다.

## 같은 차원과 더 작은 차원의 비교

시험 구성은 주기 700 nm, 폭 350 nm, 틈 17 nm, 이음선 31 nm, 배경 1 안의 재료 4/9입니다. 아래
모든 구성은 기본 행 8, x 세분 2, 틈 차수 6, 가장자리 비율 1e-5, 비균일화 세기 8, 반지름 .01을
씁니다.

| 몸체 차수 | 몸체 y 세분 | 스칼라 미지수 | 상대 정적 구간 |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 5039 | 0.000182637470 |
| 2 | 1 | 2735 | 0.000193621778 |
| 2 | 2 | 5039 | 0.000126463083 |
| 3 | 1 | 3887 | 0.000124629888 |
| 4 | 1 | 5039 | 0.000115356596 |

같은 차원에서 4차는 구간을 약 37% 좁힙니다. 3차는 앞선 미지수 5039개 모델보다 미지수를 약 23%
적게 쓰면서도 구간을 좁힙니다. 이것은 이 시험 구성에서 측정한 배분 선택이며, 보편적인 최적성
주장이 아닙니다.

4차 대각 구간은 다음과 같습니다.

```
1.670771622439439 <= Uxx <= 1.670783983361078
3.184220811164662 <= Uyy <= 3.184588132038305
```

기록된 원시/역수 잔차는 2.5e-12보다 작습니다. 독립적으로 맞춘 FEM 구간은 표의 모든 후보 구간 안에
들어갑니다. 정적 셀 불확실성은 남아 있으며, 이것은 광학 투과의 상한이 아닙니다.

파장 900 nm, 두께 300 nm, 입사각 .2 라디안, 입사 유전율 1, 기판 유전율 2.25를 씁니다. 바깥 차수
16에서:

| 보조 필드 | p T | s T | 앞선 FEM 대조군과의 s 차이 |
| --- | ---: | ---: | ---: |
| 앞선 맞춘 FEM, h=.04 | 0.907165915611 | 0.941480649651 | — |
| 앞선 1차 몸체, 미지수 5039개 | 0.907167633867 | 0.941498495500 | 1.78458e-5 |
| 3차 몸체, 미지수 3887개 | 0.907167186862 | 0.941485573526 | 4.92387e-6 |
| 4차 몸체, 미지수 5039개 | 0.907166994307 | 0.941483923649 | 3.27400e-6 |

4차는 s 차이를 약 82% 줄이고, p 차이를 1.71826e-6에서 1.07870e-6으로 줄입니다. 3차는 더 적은
미지수를 쓰면서 s 차이를 약 72% 줄입니다.

바깥 차수 24에서 4차는 p/s T로
0.906794978336 / 0.941570869123을 줍니다. 앞선 맞춘 셀 사영은
0.906794061748 / 0.941568404449를 줍니다. 해당 차이는
9.16588e-7 / 2.46467e-6입니다. 이 차수에서 앞선 선형 몸체 모델의 s 차이는 1.67667e-5였으므로,
개선이 더 높은 차수에서도 유지됩니다.

[방향별 비교](nvf-directional-convergence.md)는 여전히 적용됩니다.
바깥 x/y 절단 오차는 일부 상쇄될 수 있으며, 두 차수를 모두 16에서 24로 늘려도 투과가 여전히 크게
바뀝니다. 이 구현은 셀 근사 오차를 줄일 뿐, 바깥 절단 오차를 없애지는 않습니다.

## 독립 대조군 세분하기

후보와 대조군 사이의 차이가 이제 더 작아졌으므로, 독립적인 2D FEM 기준도 차수 8에서 h=.04에서
.03 마이크로미터로 세분했습니다.
스칼라 미지수는 43649개에서 59969개로 늘고, 상대 정적 구간은 7.93665e-7에서 4.62166e-7로
좁아집니다. 이것은 후보의 텐서곱 공간과 별개인, 맞춘 삼각형 메시입니다.

새 메시에서 cap-48 구적법을 24에서 28로 바꾸면 계수 배열은 최대 2.67e-15만큼 바뀝니다. 바깥 차수
16에서 세분한 p/s T 값은 0.907165907146 / 0.941480627757입니다. 앞선 기준과의 변화는 약
8.47e-9 / 2.19e-8에 불과하며, 위에서 보고한 후보 차이보다 훨씬 작습니다. 따라서 개선은 이 독립적인
셀 세분 아래에서도 유지됩니다. 그렇다고 공유된 Maxwell 측정 도구가 독립적인 광학 기준이 되는 것은
아닙니다.

세분한 차수 24 p/s T 값은 0.906794053264 / 0.941568382861입니다. 앞선 기준과의 변화는 약
8.48e-9 / 2.16e-8입니다. 새 4차 후보는 이 더 세밀한 대조군과 9.25e-7 / 2.49e-6만큼 달라, 더 높은
광학 차수에서도 같은 결론을 확인합니다.

## 재현과 검증

저장소 루트에서 실행하세요. 별도의 기준 내보내기에는 NGSolve가 필요하고, JIT 검증에는 JAX x64가
필요합니다. 선택적 클래스는
`studies.rcwa.nvf_polynomial_body.nvf_polynomial_body.PolynomialBodyGapModel` 로 가져올 수
있습니다.

생성되는 사영 파일을 위해 `NVF_CACHE` 를 이미 있는 임시 디렉터리로 설정하세요.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 16 24 --meshes 2:8:1 --gap-degrees 6 --body-degree 4 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 0 --meshes 2:8:1 2:8:2 --gap-degrees 6 --body-degree 2 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_polynomial_body.nvf_polynomial_body --order 2 --gaps 0 17 --seams 0 --body-degree 4 --gap-degree 6 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_reference.nvf_gap_reference --cap 48 --h .03 --degree 8 --quadrature 28 --output "$NVF_CACHE/wavebit-gap-reference-h003-cap48.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 16 24 --meshes --projection "$NVF_CACHE/wavebit-gap-reference-h003-cap48.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_polynomial_body.py tests/rcwa/test_nvf_directional_convergence.py tests/rcwa/test_nvf_interface_resolution.py tests/rcwa/test_nvf_gap_convergence.py tests/rcwa/test_nvf_gap_reference.py tests/rcwa/test_nvf_continuous_flux.py tests/rcwa/test_nvf_harmonic_coordinates.py tests/rcwa/test_nvf_partitioned_gap_energy.py tests/rcwa/test_nvf_thin_gap_energy.py tests/rcwa/test_nvf_cell_bounds.py tests/rcwa/test_nvf_cell_corrector.py tests/rcwa/test_nvf_harmonic_energy.py tests/rcwa/test_nvf_harmonic_energy_jax.py tests/rcwa/test_resonant_nvf_probe.py tests/rcwa/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

원시 출력은 `studies/rcwa/nvf_polynomial_body/records/nvf_polynomial_body_static.log`,
`nvf_polynomial_body_powers.log`, `nvf_polynomial_body_jax.log`,
`nvf_polynomial_body_contact.log`, `nvf_polynomial_body_reference.log`,
`nvf_polynomial_body_reference_powers.log` 에 있습니다.

2026-09-08 검증: 위의 전체 선택에서 **213 passed, 8 skipped, 5 xfailed** 입니다. 여기에는 다항식
전용 테스트 10개, 공유 모델 매개변수화 25개, 최종 해상도 접촉 테스트 1개가 새로 더해졌습니다.
건너뛴 8개는 선택적 FEM 의존성이 필요합니다. 위의 별도 FEM 세분과 구적법 점검은 NGSolve 환경에서
실행했습니다. 엄격한 예상 실패 5개는 여전히 제품 모델이나 예전 모델에 한정됩니다. Ruff, 공백과
문서 링크 점검은 통과합니다. 제품 디스패치, 물리 허용 오차나 고유값 미분 정칙화는 바꾸지
않았습니다.

## 이후의 x 해상도 비교

[다항식 플럭스 후속 연구](nvf-polynomial-flux.md)는 더 높은 차수의 연속 x 플럭스를 구현하고
시험합니다. 스칼라 차원이 같을 때, 시험한 공간들은 원래의 선형 플럭스를 세분한 경우보다 정적 구간이
넓습니다. 그래서 새 공간은 이 모델의 기본값을 바꾸지 않고 실험적 옵션으로 남겨 둡니다.

몸체 차수 4를 유지하고 x 세분을 2에서 3으로 늘리면 미지수가 5039개 대신 7559개가 됩니다. 이렇게
하면 정적 구간이 57.7% 더 좁아지고, 더 세밀한 맞춘 셀 필드와 비교한 차수 16 p/s 차이가
55.1% / 49.3% 줄어듭니다. 그 대가로 스칼라 미지수가 50% 더 듭니다. 비용이 더 낮은 몸체 차수 3
배분은 정적 구간을 좁히지만 s 비교는 개선하지 않으므로, 정적 정확도만으로는 광학 해상도를 고를 수
없습니다. 이것들은 연구용 해상도 선택이며, 바깥 광학 수렴은 여전히 해결되지 않았습니다.
