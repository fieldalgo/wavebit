# 다항식 x 플럭스와 셀 해상도 배분 시험

x 보강(enrichment)은 구현되었고 독립적으로 검사했지만, **앞선 연구 설정을 대체하지는 않습니다**.
같은 스칼라 차원에서, 시험한 2차와 3차 플럭스 공간은 원래의 연속 선형 플럭스를 세분화한 경우보다
정적 포함 구간이 더 넓습니다. 원래의 x 메시와 다항식 y 몸체를 더 균형 있게 배분하면 정확도와
비용 사이에서 쓸 만한 절충을 얻습니다.
이것은 여전히 `studies/`의 연구이며, 제품 `wavebit.rcwa`를 바꾸지 않습니다.

## 구현한 공간

`PolynomialFluxGapModel`은 `PolynomialBodyGapModel`을 확장합니다. 각 x 구간은 기존의 주기적
선형 끝점 모자 함수(hat)에 더해 다음 내부 Bernstein 플럭스를 가집니다.

```text
b_k(r) = binomial(p,k) r^k (1-r)^(p-k),  k=1,...,p-1.
```

이 플럭스 함수들은 내부 트레이스가 0이므로 인위적인 마디를 지나며 연속입니다. 또한 자기 구간
안에서 엄밀하게 양수입니다.
버블(bubble)을 포함한 모든 플럭스 함수 f_j에 대해 다음을 정의합니다.

```text
rho = 1 / epsilon
A_j = integral(rho f_j dx)
F_j(x) = integral from -1/2 to x of rho f_j du
R = sum of A_j over the original nodal hats only
L(x) = sum of F_j(x) over those nodal hats / R
psi_j(x) = F_j(x)/A_j - L(x).
```

합이 1이 되는 것은 원래의 마디 모자 함수뿐입니다. R에 버블 적분을 더하면 물리적인 아핀 리프트가
바뀝니다. 지원하는 양의 실수 재료에서는 모든 A_j가 양수이므로, 부호가 바뀌는 버블 적분을 분모로
쓰는 일은 없습니다. 이전과 같은 고정 마디 인덱스 하나를 빼고, 상수 퍼텐셜을 앞에 붙입니다.
구간이 E개이면 퍼텐셜 공간은 E*p개의 함수를 가집니다.
적분한 퍼텐셜은 주기적이고 연속이며, epsilon*psi'_j는 인위적인 마디에서 연속입니다. 그 물리
필드의 도약은 각 구간 안의 재료 계면을 따라갑니다.

코드는 재료 도약 접미 합을 정확히 적분하며, 퍼텐셜 차수는 p+1, 에너지 모멘트는 차수 2(p+1)까지
씁니다. 같은 재료 사이의 도약은 동등성 판정 없이 상쇄됩니다. 기하, 재료 값, 비균일화, 정규화와
균형 잡힌 셀 풀이는 JAX에서 살아 있는 값으로 남습니다. 플럭스 차수 1부터 3까지를 지원하며, 차수
1은 이전 구현에 비트 단위로 똑같이 위임합니다.

물리 x 메시는 여전히 `fractions`와 `per_region`에서 나옵니다. 그 구간 개수는 보강된 퍼텐셜 함수의
개수를 세는 `nx`와 별개입니다.
`ContinuousFluxGapModel._x_integrals`는 그 조립을 기존 y 몸체 및 틈 풀이와 분리합니다. 접촉 배율,
틈 버블과 광학 닫힘은 바뀌지 않았습니다.

## 정적 비교

픽스처는 같은 주기 700 nm, 포락 350 nm, 틈 17 nm, 31 nm의 실제 이음매, 배경 1 안의 상대 유전율
4/9입니다. 아래 모든 행은 기본 몸체 행 8, y 세분 1, 틈 차수 6, 가장자리 비율 1e-5, 재료 비균일화
세기/반지름 8/.01을 씁니다. 포함 구간은 독립적인 원(primal) 셀 풀이와 역재료(reciprocal-material)
셀 풀이를 씁니다.

| 플럭스 차수 | x 세분 | 몸체 차수 | 스칼라 미지수 | 상대 정적 포함 구간 |
| --- | --- | --- | --- | --- |
| 1 | 2 | 4 | 5039 | 1.153565960e-4 |
| 2 | 1 | 4 | 5039 | 1.866865042e-4 |
| 1 | 3 | 2 | 4103 | 1.271253268e-4 |
| 3 | 1 | 2 | 4103 | 1.568834636e-4 |
| 1 | 4 | 2 | 5471 | 1.072995465e-4 |
| 2 | 2 | 2 | 5471 | 1.134143150e-4 |
| 1 | 3 | 3 | 5831 | 5.810270957e-5 |
| 3 | 1 | 3 | 5831 | 8.791814062e-5 |
| 1 | 3 | 4 | 7559 | 4.880540352e-5 |
| 3 | 1 | 4 | 7559 | 7.863237192e-5 |

메시를 고정하면 차수가 높은 플럭스가 포함 구간을 좁히는데, 이는 중첩된 conforming 공간에서
예상되는 결과입니다. 그렇다고 효율이 확립되지는 않습니다. 같은 차원에서의 직접 비교는 모두 구간을
더 많이 쓴 원래 플럭스 쪽이 유리합니다. 어떤 연구 기본값이나 제품 기본값도 새 공간으로 바꾸지
않았습니다.

선형 플럭스를 유지하고 x 세분 3/몸체 차수 3을 쓰면, 앞선 5039 미지수 설정의 포함 구간이
**49.6%** 좁아지며, 미지수는 **15.7% 더 많이** 씁니다. 같은 x 해상도에서 몸체 차수 4를 쓰면
**57.7%** 좁아지며, 미지수는 **50.0% 더 많이** 씁니다. 이것은 차원 비교이지 실행 시간 단축이
아닙니다. 밀집 풀이 비용과 측정한 시간이 차원에 선형으로 비례할 필요는 없습니다.
측정한 원/역 잔차는 모두 2.3e-11보다 작습니다.

[몸체 연구](nvf-polynomial-body.md)의 독립적인 h=.03 마이크로미터, 차수 8 맞춤 FEM 셀은 훨씬
좁은 정적 포함 구간을 줍니다. 그 완전한 2×2 구간은 위의 열 개 구간 모두 안에 들어 있습니다. 정적
포함 구간은 유효 셀 에너지의 범위를 정할 뿐, 광학 투과율의 범위를 정하지 않습니다.

## 고정된 바깥 차수에서의 광학 비교

바깥 차수 16에서, 앞선 몸체 연구와 같은 층과 입사 조건은 다음 투과율을 줍니다. 독립적인 맞춤 셀
필드는 h=.03 마이크로미터, 차수 8, 스칼라 미지수 59969개를 씁니다. 모든 행은 같은 Maxwell 산란
계산 도구를 씁니다.

| 플럭스 차수 / x 세분 / 몸체 차수 | 스칼라 미지수 | p T | s T |
| --- | --- | --- | --- |
| 1 / 2 / 4, 앞선 설정 | 5039 | 0.907166994307 | 0.941483923649 |
| 3 / 1 / 3 | 5831 | 0.907166549923 | 0.941483968917 |
| 1 / 3 / 3 | 5831 | 0.907166604781 | 0.941484000955 |
| 1 / 3 / 4 | 7559 | 0.907166395579 | 0.941482300435 |
| 맞춤 FEM 보조 필드 | 59969 | 0.907165907146 | 0.941480627757 |

미지수 5831개의 대안들은 앞선 설정에 비해 p 불일치는 줄이지만 s 불일치는 조금 늘립니다. 특히
정적 포함 구간을 절반으로 줄여도 광학 불일치는 절반으로 줄어들지 **않습니다**. 미지수 5831개인 두
후보 사이의 작은 차이만으로는 어느 플럭스 공간에도 광학적 이점이 있다고 할 수 없습니다.

미지수 7559개의 선형 플럭스/몸체 차수 4 설정은 p 불일치를 1.08716e-6에서 4.88433e-7로, s 불일치를
3.29589e-6에서 1.67268e-6으로 줄입니다. 각각 약 **55.1% / 49.3%** 감소이며, 대가로 **스칼라
미지수를 50.0% 더 많이** 씁니다.
새 광학 실행 세 개 모두에서 무손실 전력 잔차는 2.4e-12보다 작습니다.
이것은 더 높은 해상도의 연구 선택지이지, 같은 비용에서 얻는 이득이 아닙니다.
이전에 관찰한 바깥 차수 오차나 상쇄 문제를 해결하지도 않습니다.

## 재현과 검증

두 CLI 제어 `--body-degree`와 `--flux-degree`는 서로 독립입니다. 수렴 구동기의 기본값은 몸체/플럭스
차수 1이므로 기존 동작을 유지합니다. `--grading` 없이 보강만 켜면 고정 메시를 그대로 씁니다. 다항식
플럭스 전용 CLI는 모델 생성자와 맞추어 몸체/플럭스 차수 2와 재료에 따라 비균일화한 메시를 기본값으로 씁니다.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 0 --meshes 1:8 --gap-degrees 6 --body-degree 4 --flux-degree 2 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 16 --meshes 1:8 --gap-degrees 6 --body-degree 3 --flux-degree 3 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 16 --meshes 3:8 --gap-degrees 6 --body-degree 3 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_gap_convergence.nvf_gap_convergence --orders 16 --meshes 3:8 --gap-degrees 6 --body-degree 4 --grading 8:.01 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_polynomial_flux.nvf_polynomial_flux --order 2 --gaps 0 17 --seams 0 --body-degree 2 --flux-degree 3 --gap-degree 4 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_polynomial_flux.nvf_polynomial_flux --static --seams 0 31 --rows 4 --body-degree 2 --flux-degree 3 --edge-fraction 1e-5
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_polynomial_flux.py tests/rcwa/test_nvf_polynomial_body.py tests/rcwa/test_nvf_directional_convergence.py tests/rcwa/test_nvf_interface_resolution.py tests/rcwa/test_nvf_gap_convergence.py tests/rcwa/test_nvf_gap_reference.py tests/rcwa/test_nvf_continuous_flux.py tests/rcwa/test_nvf_harmonic_coordinates.py tests/rcwa/test_nvf_partitioned_gap_energy.py tests/rcwa/test_nvf_thin_gap_energy.py tests/rcwa/test_nvf_cell_bounds.py tests/rcwa/test_nvf_cell_corrector.py tests/rcwa/test_nvf_harmonic_energy.py tests/rcwa/test_nvf_harmonic_energy_jax.py tests/rcwa/test_resonant_nvf_probe.py tests/rcwa/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle
```

원시 측정값은
[`nvf_polynomial_flux_static.log`](../../../studies/rcwa/nvf_polynomial_flux/records/nvf_polynomial_flux_static.log),
[`nvf_polynomial_flux_powers.log`](../../../studies/rcwa/nvf_polynomial_flux/records/nvf_polynomial_flux_powers.log),
[`nvf_polynomial_flux_jax_static.log`](../../../studies/rcwa/nvf_polynomial_flux/records/nvf_polynomial_flux_jax_static.log)에
있습니다.
저장소 밖에 쓰는 투영 캐시는 지워도 됩니다. 위 명령은 그 파일에 의존하지 않고 같은 물리 비교를
다시 만들어 냅니다.

전용 테스트는 중첩된 Gauss 규칙으로 물리 플럭스를 독립적으로 적분한 뒤, 가중/비가중 기저, 필드,
에너지 묶음을 하나하나 비교합니다.
또한 중첩된 원/쌍대 범위, 단위 평균, 중복된 실수 재료 분할, 차수 1의 비트 동일성, 그리고 지원하는
두 보강 차수와 두 편광 모두에 대해 살아 있는 JIT 틈/이음매/재료/폭 미분을 검사합니다. 구동기는
명시적으로 동일한 물리 설정과 메시 설정에 대해 검사합니다.

2026-09-08 검증: 대상 검사와 보완 회귀 선택은 **서로 다른 통과 테스트 253개, 건너뜀 8개, 엄격한
예상 실패 5개** 를 포함합니다. 이는 앞선 몸체 연구에 전용 테스트 11개, 공유 모델 매개변수화 25개,
최종 해상도 접촉 구성 네 개를 더한 것입니다.
최종 해상도 접촉 검사는 `LinAlgWarning`을 오류로 승격하고, nm당 1e-7인 독립적인 p/s 한쪽 미분
허용 오차를 유지합니다. 여기에는 미지수 5831개와 7559개인 원래 플럭스 설정과, 시험한 2/3차 플럭스
대안이 모두 포함됩니다. 건너뛴 테스트는 선택적 FEM 의존성이 필요합니다. 예상 실패 다섯 개는
여전히 제품 모델이나 오래된 모델에 한정됩니다.

3차 전용 CLI는 여기에 더해, 이음매 0 nm과 31 nm에서 독립적인 닫힌 적층(closed-laminate) 에너지와
9e-16까지, JIT 이음매 미분과 4e-19까지 일치합니다.
Ruff, 공백 검사, 로컬 문서 링크와 JSON 결과 로그가 통과합니다.
기존의 물리 허용 오차나 고유값 미분 정규화는 하나도 바꾸지 않았습니다.

제품 디스패치, 해결되지 않은 바깥 푸리에 수열, 기울어진/곡면 기하 지원은 이번 변경의 범위
밖입니다. 하나의 고정된 바깥 차수에서 맞춤 보조 필드에 더 가까워졌다는 것은 독립적인 Maxwell
검증이 아니며, 바깥 광학 급수가 수렴했다는 증거도 아닙니다.
