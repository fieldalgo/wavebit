# 유한 틈을 분해하고 수렴 오차 분리하기

연속 플럭스 연구는 이제 계층적 퍼텐셜 버블로 유한 틈 안의 변화를 분해합니다.
`ContinuousFluxGapModel`의 기본값은 `gap_degree=4`이며, `gap_degree=1`은 이전의 선형 틈 모델을
재현합니다. 독립적인 접촉 미분, 실수 이음매 미분, 재료 분할 검사는 계속 통과합니다. 명시적인
중첩 몸체 세분화도 `y_subdivisions`로 쓸 수 있습니다.

새 수렴 도구는 정적 셀 근사를 바깥 푸리에 절단과 분리하고, 둘 다 독립 대조군과 비교합니다.
직사각형 보조 필드에는 이제 독립적으로 맞춘 2D FEM 기준이 있으며, 그 상대 정적 포함 구간은
8e-7보다 작습니다. 광학 투과율은 바깥 차수 16에서도 여전히 변합니다. 이 작업은 수렴한 3D
광학 답을 확립하지 않고, 기울어진 고정 장치를 풀지 않으며, 모델을 제품 NVF에 통합하지도
않습니다.

이어진 [재료로 정의한 메시 연구](nvf-interface-resolution.md)는 같은 차원에서 정적 포함 구간을
약 64% 줄이고, 맞춘 셀 대조군에 대한 차수 16의 s 차이를 좁힙니다. 그 독립 비교는 바깥 차수
20까지 이어지며, 거기서도 투과율은 여전히 변합니다. 아래 표는 그 이전의 고정 메시 측정을 그대로
남겨 둔 것입니다.

더 새로운 [방향별 비교](nvf-directional-convergence.md)는 정사각 차수뿐 아니라 `--orders NX:NY`도
받습니다. 이 비교는 각 푸리에 방향의 광학적 효과를 분리하고, 보조 셀의 x/y 미지수를 배분하는
여러 방식을 비교합니다.

## 접촉 극한을 유지하는 틈 내부 필드

정규화한 틈 좌표를 [-1/2,1/2] 안의 s라고 쓰겠습니다. 이전의 틈 퍼텐셜은 `v(x,-1/2) + g*s*a(x)`였고,
여기서 g는 틈을 주기로 나눈 값입니다. 이를 다음과 같이 확장합니다.

```
chi_gap = v(x,-1/2) + g*s*a(x) + g*sum_k beta_k(s)*b_k(x)
beta_k'(s) = P_k(2*s),       k = 1, ..., gap_degree-1
beta_k(-1/2) = beta_k(1/2) = 0
```

P_k는 Legendre 다항식입니다. 버블은 틈 내부만 바꾸며, 두 면에서의 퍼텐셜은 여전히 몸체와
일치합니다. x 아핀 리프트와 주기 x 기저는 [적분한 연속 플럭스 모델](nvf-continuous-flux.md)의
것과 같습니다.

기존 경사 보강과 마찬가지로, 각 버블 시험 방정식을 해석적으로 g로 나눕니다. M과 K를 가중치 없는
x 질량 행렬과 강성 행렬이라고 하겠습니다. 스케일한 버블 블록은 다음과 같습니다.

```
A_bb/g = D kron M + g^2*S kron K
D_kl = integral beta_k'(s)*beta_l'(s) ds = delta_kl/(2*k+1)
S_kl = integral beta_k(s)*beta_l(s) ds
```

행 스케일 이전의 v-b 결합은 `g^2*integral(beta)`에 비례하고, a-b 결합은 `g^3*integral(s*beta)`에
비례합니다. 스케일한 버블 방정식의 x 강제항은 `-g*integral(beta)*integral(L*psi')`이며, 여기서 L은
아핀 리프트 미분입니다. 버블의 법선 미분을 적분하면 0이 되므로 y 강제항은 사라집니다.

g=0에서 버블 블록은 양의 정부호이고 우변은 0입니다. 따라서 그곳에서 b=0이며 원래의 1차 틈
에너지가 유지됩니다. 작은 틈 차단값, 0으로 나누는 수치 연산, 미리 정한 광학 그래디언트는
없습니다. 보통 DOF와 버블 DOF는 같은 잔차/정의역 보호 장치를 유지합니다.

틈 퍼텐셜은 s에 대한 다항식입니다. 그 곱은 `2*gap_degree`차까지의 정확한 푸리에 모멘트를 쓰며,
NumPy/JAX가 같은 공식을 공유합니다. 차수 2, 4, 6은 전체 필드와 에너지의 텐서 Gauss 적분과
독립적으로 비교해 검사합니다. 기본값은 차수 4이고, 명시적으로 지원하는 범위는 1부터 6까지입니다.
몸체 y 격자는 원래의 비균일 구간마다 세분할 수 있으며, 이는 내부뿐 아니라 틈 근처 영역도
세밀하게 합니다. 두 세분화 모두 고정된 x 위상 구조와 같은 재료 사이의 상쇄를 보존합니다.

## Independent static reference

`studies/rcwa/nvf_gap_reference/nvf_gap_reference.py`는 모든 재료 영역에 맞춘 별도의 주기 삼각형
메시를 만들고, NGSolve로 원시 및 상반 스칼라 셀 문제를 풉니다. 재료 면적, 두 주기 식별, 평균 필드,
실제 잔차, 푸리에 투영 구적을 검사합니다. 그 결과는 보조 필드 기준이며, 독립적인 Maxwell 풀이가
아닙니다. 자원을 많이 쓰는 3D 기울어진 FEM 실행은 다시 시작하지 않았습니다.

고정 장치는 주기 700 nm, 폭 350 nm, 틈 17 nm, 이음매 31 nm이고, 배경 1 안에 재료 4/9가 있습니다.
기준 세분화 결과는 다음과 같습니다.

| 최대 메시 크기, 마이크로미터 | 다항식 차수 | 스칼라 미지수 | 상대 정적 포함 구간 |
| ---: | ---: | ---: | ---: |
| 0.06 | 6 | 15481 | 3.22659e-6 |
| 0.04 | 8 | 43649 | 7.93665e-7 |

더 세밀한 수준에서 대각 한계는 다음과 같습니다.

```
1.670777582297659 <= Uxx <= 1.670777804152657
3.184544887602813 <= Uyy <= 3.184547414999149
```

이것은 부동소수점 변분 한계이며, 구간 산술 증명서가 아닙니다. 세밀한 투영 구적을 20점에서
24점으로 늘리면 cap-32 계수 배열이 최대 1.69e-15 바뀝니다. 거친 수준의 16점에서 20점으로의 구적
변화는 6.14e-9였으며, 메시 수렴과 분리해 두었습니다. 닫힌 맞춤 적층판은 해석적 에너지 텐서와
단위 평균 필드를 복원합니다. 불완전한 재료 대응과 잘못된 기하는 거부합니다.

연속 플럭스 모델 자체의 원시/상반 풀이는 각 세분화에서 얻는 개선을 따로 보여 줍니다.

| x 세분 | 기본 몸체 행 | 몸체 y 세분 | 틈 차수 | 상대 정적 포함 구간 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 8 | 1 | 1 | 0.001954201692 |
| 2 | 8 | 1 | 4 | 0.001541216595 |
| 4 | 8 | 1 | 1 | 0.001705394515 |
| 4 | 8 | 1 | 4 | 0.001291912228 |
| 4 | 8 | 2 | 4 | 0.000771780974 |

마지막 모델에는 스칼라 미지수가 5183개 있습니다. 그 셀 근사는 여전히 맞춘 FEM 기준보다 덜
정확하며, 대수 잔차가 작다고 그 차이가 없어지지는 않습니다. 이음매 미분 관문을 통과했다고 해서
셀 근사가 수렴했다는 뜻도 아닙니다.

## 고정된 바깥 차수와 늘어나는 바깥 차수에서의 광학 비교

`studies/rcwa/nvf_gap_convergence/nvf_gap_convergence.py`는 요청한 가장 큰 푸리에 상한에서 한 번
투영한 뒤, 그 계수로 더 작은 유지 계를 각각 만듭니다. 더 낮은 상한에서 직접 계산해 이 재사용을
검사합니다. 별도의 카르테시안 도구는 폭이 0인 틈을 포함해 재료 구간을 정확히 적분합니다.
선택적 FEM 내보내기는 기하와 재료가 일치할 때만 받습니다. 세 경로 모두 같은 Maxwell 산란 도구를
공유하므로, 이 비교는 인수분해/셀 효과만 분리합니다. 서로 독립인 Maxwell 솔버 세 개가 아닙니다.

같은 고정 장치의 바깥 차수 16에서:

| 인수분해 / 보조 필드 | p T | s T |
| --- | ---: | ---: |
| 정확한 카르테시안 구간 | 0.907193571579 | 0.941468107350 |
| 맞춘 FEM 필드, 같은 에너지 닫힘 | 0.907165915611 | 0.941480649651 |
| 연속 플럭스, x4, y1, 선형 틈 | 0.907203629098 | 0.941560169727 |
| 연속 플럭스, x4, y1, 차수 4 틈 | 0.907170121909 | 0.941553172975 |
| 연속 플럭스, x4, y2, 차수 4 틈 | 0.907170049872 | 0.941540519833 |

차수 4 틈은 같은 몸체 메시에서, 독립적으로 맞춘 셀 투영에 대한 p 차이를 3.77e-5에서 4.21e-6으로
줄입니다. s 차이는 그보다 덜 줄어들며, 표에 적은 몸체 세분화 뒤에도 5.99e-5로 남습니다. 이것은
유한 차수 투영 셀 대조군과의 차이이며, 물리 투과율의 증명된 오차가 아닙니다.

맞춘 FEM 필드를 고정해도 바깥 세분화의 효과는 여전히 보입니다.

| 바깥 차수 | p T | s T |
| ---: | ---: | ---: |
| 12 | 0.907643855516 | 0.941181620193 |
| 16 | 0.907165915611 | 0.941480649651 |

약 4.78e-4 / 2.99e-4의 변화는 차수 16의 여러 정식화 사이에 남은 차이보다 큽니다. 따라서 정적
셀만 개선해서는 광학 수렴을 확립할 수 없습니다. 이 스윕에서 수치적 평탄 구간이나 외삽한 광학
기준값은 받아들이지 않습니다.

이제 두 입사 편광이 층 고유값 풀이 하나를 공유합니다. 틈 다항식 투영도 y 푸리에 모멘트를 곱하기
전에 x 기저를 먼저 축약해, 필드나 에너지를 바꾸지 않고 반복 축약을 피합니다. 독립적인 판(slab)
공식, 전체 틈 구적, JIT/유한 차분 검사로 이런 계산 변경을 검증합니다.

## Reproduction and validation

저장소 루트에서 실행하세요. 처음 명령들은 NumPy/SciPy가 필요하고, JIT 검사에는 추가로 JAX x64가
필요합니다. 맞춘 기준은 NGSolve를 설치한 별도의 환경을 씁니다.

생성한 투영 파일을 둘 기존 임시 디렉터리를 `NVF_CACHE`로 지정하세요.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_continuous_flux.nvf_continuous_flux --order 2 --gaps 0 17 --seams 0 --gap-degree 4
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 4 8 12 16 --meshes 2:8 4:8 --gap-degrees 1 4
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 16 --meshes 4:8:2 --gap-degrees 4
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_reference.nvf_gap_reference --cap 32 --h .04 --degree 8 --quadrature 20 --output "$NVF_CACHE/wavebit-gap-reference.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 4 8 12 16 --meshes --projection "$NVF_CACHE/wavebit-gap-reference.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_gap_convergence.py tests/rcwa/test_nvf_gap_reference.py tests/rcwa/test_nvf_continuous_flux.py tests/rcwa/test_nvf_harmonic_coordinates.py tests/rcwa/test_nvf_partitioned_gap_energy.py tests/rcwa/test_nvf_thin_gap_energy.py tests/rcwa/test_nvf_cell_bounds.py tests/rcwa/test_nvf_cell_corrector.py tests/rcwa/test_nvf_harmonic_energy.py tests/rcwa/test_nvf_harmonic_energy_jax.py tests/rcwa/test_resonant_nvf_probe.py tests/rcwa/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

원시 측정은 `studies/rcwa/nvf_gap_convergence/records/nvf_gap_convergence.log`와 같은 디렉터리의
`nvf_gap_body_refinement.log`, `nvf_gap_refined_powers.log`,
`nvf_gap_reference.log`, `nvf_gap_reference_powers.log`,
`nvf_gap_enriched_jax.log`에 있습니다.

2026-09-08 검증: 전체 선택에서 **136 passed, 8 skipped, 5 xfailed** 입니다. 선택적 FEM 시험 8개도
자체 환경에서 통과합니다(기존 셀 시험 5개와 새 프로파일 시험 3개). 기존의 예상 실패는 원래의
제품/이전 연구 한계에 한정된 채로 남아 있습니다. 완화한 물리 허용 오차는 없습니다. 제품 디스패치와
고유값 미분 정칙화는 바뀌지 않았습니다.
