# 움직이는 재료 이음매를 지나는 연속 몸체 플럭스

`studies/rcwa/nvf_continuous_flux/nvf_continuous_flux.py`는 국소 조화 좌표 모델이 남긴, 시험한 양의 틈
이음매 미분 불일치를 해소합니다. 이 모델의 주기 퍼텐셜은 연속 플럭스 모자 함수를 유전율의 역수에
대해 적분해 얻습니다. 이렇게 하면 요소 안에서 물리적 필드 도약이 보존되고, 인공 메시 노드에서
몸체 변위가 연속으로 유지됩니다. 기존 분할, 접촉, 닫힌 적층판, 재료 미분 관문은 계속 통과합니다.
앞선 모델들과 그 엄격한 실패 테스트는 대조군으로 계속 남아 있습니다.

이것은 여전히 직사각형 연구 모델입니다. 이 계면 검사들을 통과했다고 해서 일반 구조에 대한 수렴한
광학 투과율이 입증되거나, 이 방법이 제품 NVF에 통합되는 것은 아닙니다. 기울어진/휜 기하, 독립적인
광학 수렴, 제품 백엔드 통합은 아직 열린 과제입니다.

이후의 [유한 틈 수렴 작업](nvf-gap-convergence.md)은 내부 틈 버블과 중첩된 몸체 세분화를 추가합니다.
현재 기본값은 `gap_degree=4`이고, 아래의 초기 측정은 `gap_degree=1`을 사용했습니다.
둘 다 여기서 설명하는 연속 x 플럭스를 공유합니다. 독립 정적 FEM 기준과 광학 차수 비교는 후속
기록에 있습니다.

추가된 [재료로 정의한 메시 확장](nvf-interface-resolution.md)은 분할 불변성과 변수로 둔
기하/재료 미분을 유지하면서, 같은 개수의 x 노드를 실제 재료 도약 주변에 집중시킵니다. 이것은
선택해서 쓰는 하위 클래스이며, 원래 메시는 아래에서 쓰는 대조군으로 남습니다.

[방향별 후속 기록](nvf-directional-convergence.md)은 추가로, 보강된 셀의 선형계를 변수로 둔 대각
좌표 변환으로 평형화합니다. 같은 방정식을 풀고 원래 잔차를 검사하므로, 세분화한 접촉 모서리 설정이
조건수 경고 없이 통과할 수 있습니다.

선택해서 쓰는 [다항식 몸체 모델](nvf-polynomial-body.md)은 이제 틈에서의 트레이스를 보존하면서
주기 y 퍼텐셜을 내부 버블로 보강합니다. 몸체 차수 1은 이 모델을 정확히 재현하며, 차수 2부터 4까지는
독립적인 적분 검사와 계면 검사를 가집니다. 아래의 x 플럭스 구성은 두 구현이 공유합니다.

## 연속 플럭스 공간 적분하기

f_j를 외곽(envelope) 메시 위의 일반 주기 P1 모자 함수라 하고, 다음을 정의합니다.

```
rho(x) = 1/epsilon(x)
A_j = integral_period rho(x)*f_j(x) dx
R = sum_j A_j = integral_period rho(x) dx
F_j(x) = integral_{-1/2}^x rho(u)*f_j(u) du
ell(x) = sum_j F_j(x)/R
psi_j(x) = F_j(x)/A_j - ell(x)
```

각 psi_j는 주기적입니다. 관계식 `sum_j A_j*psi_j = 0` 덕분에 고정된 위상 인덱스에서 모자 함수 하나를
생략할 수 있고, 추가 상수 기저 함수가 게이지를 펼칩니다. x 시험 공간은 앞선 모델과 같은 스칼라
차원을 가집니다. 메시와 생략한 인덱스는 재료 조각의 개수나 같음 여부에 의존하지 않습니다. 모든
재료는 양의 실수여야 합니다.

x 퍼텐셜에는 아핀 리프트 ell을, y 퍼텐셜에는 y를 사용하고, 주기 보정은 psi 기저로 전개합니다. 그 x
미분은 다음을 만족합니다.

```
ell' = rho/R
psi_j' = rho*(f_j/A_j - 1/R)
epsilon*ell' = 1/R
epsilon*psi_j' = f_j/A_j - 1/R
```

마지막 두 식은 실제 재료 이음매 두 곳과 인공 요소 경계를 포함해 재료 몸체 안에서 x에 대해
연속입니다. 앞선 국소 조화 기저는 각 요소 안에서 플럭스가 상수였지만 요소 사이의 플럭스 도약을
허용했습니다. 그 인공 도약을 없애면, 메시를 움직이거나 보고하는 그래디언트를 순방향 풀이와 따로
바꾸지 않고도 시험한 이음매 통과 민감도 결함이 사라집니다.

몸체/틈 퍼텐셜 구성, 해석적으로 척도를 맞춘 틈 시험 방정식, 평균장 에너지 닫힘은
[conforming 틈 연구](nvf-thin-gap-energy.md)에서 물려받습니다.
[국소 조화 좌표 모델](nvf-harmonic-coordinates.md)에서와 마찬가지로, 실제 아핀 리프트는 몸체/틈
강제항과 투영 에너지 모두에 포함되어야 합니다. 같은 x 기저를 틈 안으로 확장하며, 그곳에서의 유한
근사 오차는 여전히 세분화의 대상입니다.

## 정확한 모멘트와 재료 상쇄

요소 하나와 재료 조각 하나 안에서 F_j는 물리 국소 x에 대해 이차식입니다. 기저 곱의 차수는 최대
4입니다. 새 푸리에 모멘트 도우미는 0 주파수에서 떨어진 곳에서는 중심 부분 적분을, 0 근처에서는
수렴 급수를 사용합니다. 폭이 0인 교집합도 JAX에서 미분 가능하게 유지됩니다. 독립적인 80점 가우스
규칙이 음수/0 근처 주파수와 폭이 0인 구간을 포함해 차수 0부터 4까지를 검사합니다. 그 JIT 끝점
미분은 따로 검사합니다.

요소 합계와 누적 적분은 재료 역수의 도약으로부터 조립합니다. 그런 다음 가중 기저 다항식과 리프트
다항식을 접미 구간 위의 부호 있는 변화량으로 적분합니다. 같은 재료 사이의 이음매는 동등성 조건식이나
독립 재료 매개변수 제거 없이도 정확히 0을 기여합니다. 비균일 프로필 안의 중복 이음매를 포함해, 병합한
투영 배열과 분할한 투영 배열은 시험한 NumPy 호출에서 비트 단위로 동일하게 유지됩니다. 독립적인
48점 구간별 구적이 완전한 가중 질량, 강성, 하중, 미분, 리프트 묶음을 절대 허용 오차 2e-12로
검사합니다.

공유 형상 테스트는 주기 경계를 지나는 평행 이동과, JIT 아래에서 변수로 둔 외곽, 틈, 재료, 이음매
매개변수를 다룹니다. 잘못된 일반 프로필은 오류를 발생시키고, 잘못된 추적(traced) 갱신은 유한하지
않은 데이터를 반환합니다. 원래 연구의 잔차 보호 기준은 계속 활성 상태입니다. 제품 솔버, 카르테시안
Li 분기, 고유값 정칙화는 바꾸지 않았습니다.

## 계면 측정

폭 350 nm, 주기 700 nm, 틈 17 nm, 이음매 0, 재료 4/9, 외부 차수 2, s 편광, `rows=4`,
`edge_fraction=1e-4`에서 1e-4 nm 단계의 이음매 할선은 다음과 같습니다.

| 모델 | 왼쪽 미분, /nm | 오른쪽 미분, /nm | 절대 차이 |
| --- | ---: | ---: | ---: |
| 국소 조화 좌표 | -0.001382052641 | -0.001383810998 | 1.75836e-6 |
| 적분한 연속 플럭스 | -0.001387386046 | -0.001387388302 | 2.25597e-9 |

기존 1e-7 연속성 관문은 이제 새 모델에서 통과합니다. 예전 국소 모델에서는 여전히 엄격한 예상
실패로 남습니다. 새 p/s 검사는 메시 노드에서 JIT 이음매 미분을 두 한쪽 방향 NumPy 할선과도
비교합니다. 외부 차수 3의 기본 CLI 저장 로그는 같은 s 편광 틈·이음매 조건에서 불일치가 약
1.72e-6(국소)에서 8.13e-9(적분)로 줄었음을 독립적으로 보여 줍니다. 이 유한 차분 차이에는 일반적인 곡률과 부동소수점 오차가 포함되며, 정확히 0이라고
주장하지는 않습니다.

닫힌 혼합 막대의 정적 에너지와 이음매/재료/폭 미분은 맞춰지지 않은(unfitted) 계면을 포함해 독립적인
해석적 일치를 유지합니다. 닫힌 구성 행렬과 p/s 광학 이음매/재료 미분은 같은 독립 카르테시안 대조를
통과합니다. 외부 차수 7, 균일 재료 12.25, 폭 350 nm, `rows=8`, `edge_fraction=1e-5`에서 접촉
결과는 다음과 같습니다.

| 편광 | 닫힌 T | JIT dT/dgap, /nm |
| --- | ---: | ---: |
| p | 0.626061531896529 | 0.001891546100057 |
| s | 0.878242289574428 | -0.000778519822153 |

둘 다 독립적으로 외삽한 카르테시안 양의 틈 할선에 대해 앞선 1e-7 절대 미분 관문을 통과합니다. 인공
이음매 미분은 사라지고 두 재료 미분은 모두 변수로 남습니다. 카르테시안 Li는 시험 도구일 뿐이며, 이
모델이 호출하는 것이 아닙니다.

## 셀 세분화는 광학 수렴과 별개입니다

재료 4/9, 틈 17 nm, 외부 차수 2에서 31 nm에 실제 이음매가 있을 때:

| x 세분 | 스칼라 미지수 | 정적 Uxx | 정적 Uyy | p T | s T |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 467 | 1.671011843868 | 3.187630224161 | 0.939108136894 | 0.906437006083 |
| 2 | 935 | 1.671001225919 | 3.186369469373 | 0.939108105679 | 0.906236254717 |
| 4 | 1871 | 1.670998589002 | 3.186197687187 | 0.939107991586 | 0.906224852820 |

고정된 재료에서 적분 공간들은 이 x 세분화 아래에서 중첩됩니다. 정적 주(primal) 에너지는 예상대로
감소합니다. 그 변화와 투과율에 남은 변화는 추가 수렴 작업이 필요합니다. 작은 선형 잔차(여기서는
2e-13 미만), 단위 평균 B, 에르미트성, 무손실 에너지 보존은 광학 오차 상한을 제공하지 않습니다. 앞선
연구들에서 설명한 기울어진 기준과 외부 푸리에 차수의 불확실성은 그대로 남아 있습니다.

## 재현과 검증

NumPy, SciPy, JAX를 설치한 상태로 저장소 루트에서 실행하세요.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_continuous_flux.nvf_continuous_flux --gap-degree 1
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_continuous_flux.nvf_continuous_flux --static --gap-degree 1
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_continuous_flux.nvf_continuous_flux --order 2 --gaps 17 --seams 0 --gap-degree 1
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_continuous_flux.nvf_continuous_flux --order 7 --gaps 0 --seams 31 --values 12.25 12.25 --rows 8 --edge-fraction 0.00001 --gap-degree 1
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_continuous_flux.py tests/rcwa/test_nvf_harmonic_coordinates.py tests/rcwa/test_nvf_partitioned_gap_energy.py tests/rcwa/test_nvf_thin_gap_energy.py tests/rcwa/test_nvf_cell_bounds.py tests/rcwa/test_nvf_cell_corrector.py tests/rcwa/test_nvf_harmonic_energy.py tests/rcwa/test_nvf_harmonic_energy_jax.py tests/rcwa/test_resonant_nvf_probe.py tests/rcwa/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

`ContinuousFluxGapModel`은 스칼라 프로필과 평행 이동한 수평 형상 목록 API를 물려받습니다. CLI는
국소 조화 연구의 보고 형식을 공유하고, JAX x64를 켜며, NumPy/JAX 값, 이음매 할선, 변수 미분을
기록합니다. 측정한 광학, 정적, 세분화, 접촉 데이터는
`studies/rcwa/nvf_continuous_flux/records/nvf_continuous_flux.log`에 있습니다.

2026-09-08 초기 검증: 위의 전체 선택에서 **122개 통과, 5개 건너뜀, 5개 xfailed** 결과가
나왔습니다. 새 모델은 공유 매개변수화 검사 23개와 전용 검사 다섯 개를 통과합니다. 앞선 모든 엄격한
예상 실패는 원래 모델에만 해당하며, 기존 물리 허용 오차나 분할 허용 오차는 하나도 완화하지
않았습니다. 건너뛴 항목은 선택 사항인 NGSolve 대조입니다.
