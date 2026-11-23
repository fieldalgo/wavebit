# 셀 오차를 광학·접촉 오차와 분리하기

조화 에너지 후보는 이제 정적 보조 셀 문제에 대해 서로 독립적인 두 가지 이산화, 즉 푸리에 Galerkin과
conforming 2차원 FEM을 가집니다. 이 조사는 **셀** 불확실성을 크게 줄이는 동시에, 새롭고 심각한
**s 편광 접촉 미분 실패**도 찾아냅니다. 이 후보가 제품 NVF의 자격을 얻게 하지는 않습니다.

이후의 [conforming 얇은 틈 정식화](nvf-thin-gap-energy.md)는 이제 두 편광 모두에서 직사각형 접촉 미분
검사를 통과합니다. 그것은 다른 셀 공간으로 누락된 움직이는 층 항을 다룹니다. 여기 기록한 고정 기하 FEM
내보내기와 유한 푸리에 실패는 바뀌지 않았습니다.

도구는 `studies/rcwa/nvf_cell_bounds/nvf_cell_bounds.py`와
`studies/rcwa/nvf_cell_corrector/nvf_cell_corrector.py`입니다. 후자는 선택적 NGSolve를 써서 푸리에 계수를
내보내고, 전자는 같은 에너지 닫힘과 Wavebit 산란 루틴을 그 내보낸 결과에 적용합니다. 이 보조 계산에는
3차원 Maxwell FEM 세분화가 필요하지 않습니다.

## 정적 셀에 대한 측정 가능한 포함 구간

실수 양의 스칼라 재료에서, 평균이 I인 주기적 무회전 필드 B는 주 에너지 텐서
`U = average(B.T epsilon B)`를 줍니다. 역수 재료 `rho=1/epsilon`에 대해 유사한 셀 문제를 풀어 에너지 텐서 V를
얻습니다. 그 무회전 시험 필드를 90도 회전하면 허용 가능한 무발산 플럭스가 됩니다. 회전 R을 쓰면 상보 한계는
`L = R inv(V) R.T`입니다. 정확한 정적 균질화 텐서는 양의 준정부호 순서에서 L과 U 사이에 놓입니다.

이 주/상보 구성과 정확 적분 푸리에 Galerkin 틀은 [Vondřejc, 2014](https://arxiv.org/abs/1412.2033)에
설명되어 있습니다. 보고한 포함 구간은 부동소수점 산술로 계산한 것이며, 구간 산술 증명서가 아닙니다. 더
중요하게는, 이것은 **광학 T에 대한 포함 구간이 아니며**, 유한 차수 구성 행렬의 모든 성분에 대한 한계도 아닙니다.
이것은 보조 필드를 그 작은 선형 잔차와 분리해 진단합니다.

아래의 상대 폭은 기울어진 접촉 픽스처에 대한 `(U-L, L)`의 가장 큰 일반화 고유값입니다.

| 셀 이산화 | 상대 에너지 포함 구간 폭 |
| --- | ---: |
| Fourier H=31 | 0.05491477823 |
| Fourier H=63 | 0.02625956204 |
| Fourier H=127 | 0.01286227328 |
| Fourier H=255 | 0.00638098156 |
| Fourier H=511 | 0.00318067546 |
| FEM h=0.06, degree=6, vertex refinement=20 | 0.00004083213 |
| FEM h=0.06, degree=8, vertex refinement=80 | 0.00000291024 |

따라서 마지막 FEM 포함 구간은 H=511 푸리에 포함 구간보다 약 1,100배 좁습니다. 이것은 **정적 오차 진단**에서의
개선이며, 광학 정확도나 속도가 1,100배라는 주장이 아닙니다. 가장 높은 FEM 셀 계산은 삼각형 1,396개와,
평균 0 구속을 포함해 스칼라 문제당 미지수 44,673개를 가집니다. 이는 앞선 3D Maxwell 풀이보다 훨씬 작습니다.

FEM 기하는 잘라서 합친 주기적 직사각형, 두 가로축의 명시적 식별, 재료 면적 검사를 씁니다. 2차원에서는 OCC
면 이름이 재료 레이블을 제공합니다. 부피 재료 태그를 쓰자 초기 대조 구성의 레이블이 `default`로 남았고, 면적
보호 장치가 그 시도를 거부했습니다. 구현된 버전은 면 이름을 쓰며, 균일 대조 구성과 막대 대조 구성을 모두
통과합니다. NGSolve는 [주기적 스칼라 공간](https://ngsolve.org/ngsolve/docs/i-tutorials/unit-2.12-periodicity/periodicity.html)을
문서화하고 있습니다.

## 유전율 래스터 없는 푸리에 사영

각 conforming 삼각형 위에서 Duffy 사상 Gauss 구적법이 B의 네 성분, 대칭 에너지 필드의 세 성분, epsilon을
푸리에 파동에 대해 적분합니다. 이 여덟 계수 배열이 Q, G, 그리고 바뀌지 않은 에너지 닫힘의 종방향 자리를
채웁니다. 점 묶음 처리로 임시 메모리를 제한합니다.

변조하지 않은 에너지 적분은 최종 내보내기에서 별도의 NGSolve 적분과 1.83e-13까지 일치합니다. 그 대조만으로는
푸리에 진동을 분해하지 못합니다. 계수 상한 44에서 Duffy 좌표당 구적점을 20개에서 24개로 늘리면 일부 계수가
4.38e-9만큼 바뀝니다. 24개에서 28개로 늘리면 3.37e-13만큼 바뀝니다. 28개 점에서 사영한 epsilon 계수는
해석적 회전 직사각형 변환과 1.78e-15까지 일치합니다.

degree-8/refinement-80 셀 설정과 구적 28에서 기울어진 구성의 광학 결과는 대략 다음과 같습니다.

| Maxwell 바깥 차수 | T |
| --- | ---: |
| 6 | 0.855485746686 |
| 10 | 0.857473156196 |
| 14 | 0.860355871708 |
| 18 | 0.861820093322 |
| 22 | 0.862617122442 |

바깥 차수 6/10/14에서 셀을 degree 6/refinement 20에서 degree 8/refinement 80으로 바꾸면 T가 약 1.4e-5만큼
바뀝니다. 18에서 22로 가는 남은 바깥 차수 변화는 0.000797029로 훨씬 큽니다. n=22 값은 앞선 H=511 푸리에
보정기 값과 0.00140120, 예비 3D FEM 부피 값과 0.00343878만큼 다릅니다. 유한 바깥 수열도, 그 독립 3D 기준도
수렴했다고 선언하지 않습니다.

축에 정렬된 닫힌 막대에서 이 conforming 필드 에너지 닫힘은 n=6에서 T=0.627676664411을 주고, 해석적 역 규칙
구성 행렬을 재현합니다. 이로써 이 대조 구성에서 푸리에 보정기의 유한 H 닫힌 값 편향이 사라집니다. 세 시험
구성 모두 같은 셀 풀이와 에너지 구성을 쓰며, 역 규칙은 직사각형 전용 분기가 아니라 독립적인 검사입니다.

## s 편광 접촉 실패가 수용 판단을 바꿉니다

이전의 약 0.54% 접촉 미분 불일치는 전기장이 사라지는 수평 틈에 접선 방향인 **p 편광**에서만 측정한 것입니다.
그 진술을 s 편광으로 일반화해서는 안 됩니다.

n=7에서 s 편광 닫힌 막대의 T는 현재 푸리에 후보와 독립적인 적분 카르테시안 Li 도구 모두에서 0.878242289574입니다.
그럼에도 다음과 같습니다.

| 접촉 미분 측정 | dT/dgap |
| --- | ---: |
| 후보 JAX, H=31 | -0.00006355713415 |
| 후보 JAX, H=127 | -0.00006355713415 |
| 후보 JAX, H=255 | -0.00006355713415 |
| 독립 Li 양의 할선, 간격 1e-4 | -0.00077830402567 |
| 독립 Li 양의 할선, 간격 1e-5 | -0.00077854585001 |

후보의 미분은 닫힌 값과 자신의 유한 차분에 일치하는데도 약 12.25배 너무 작습니다. H를 세분해도 정확한 접촉에서
이 실패는 사라지지 않습니다. 이것은 엄격한 예상 실패 연구 관문으로 유지되며, 제품의 원래 split/merge 예상
실패도 그대로 남습니다.

이 비율에는 물리적 설명이 있습니다. 호스트 안의 얇은 평면 틈에서 접선 방향 전기장은 연속이지만,
법선 방향에서는 전기장이 아니라 **변위**가 연속입니다. 호스트와 틈의 유전율을 `eh`와 `eg`라 하면, 틈 0에서 직렬 적층판을 미분하면
국소 법선 구성 변화 `eh*(eg-eh)/eg`를 얻는 반면, 매끈한 유한 필드 재료 섭동은 `eg-eh`만 줍니다. 여기서
`eh/eg=12.25`입니다. 줄어드는 틈은 O(gap) 영역 위에서 O(1) 법선장 변화를 실어 나르며, 그 제곱 필드 기여는 틈에
대해 1차로 남습니다. 고정된 매끈한 푸리에 기저는 기저를 세분하기 전에 틈 미분을 취해서는 그 기여를 잡아낼 수
없습니다. 이것은 교환되지 않는 극한이며, CG 정지 허용 오차 문제가 아닙니다.

보정은 얇은 층 필드나 동등한 계면 기여를 통해 순방향 정식화에 들어가야 합니다. 보고된 그래디언트에 12.25를
곱해서는 순방향 모델도, 그 자동 미분 일관성도 고쳐지지 않습니다. 새 FEM 도구는 고정 기하 필드를 제공할 뿐
움직이는 접촉 미분은 구현하지 **않으므로**, 개선된 셀 수렴이 이 관문을 닫지 못합니다.

이와 별개로 훨씬 작은 미분 한계도 있습니다. 평행 이동한 440x270/epsilon-6.25 시험에서 n=2/H=7일 때, Wavebit의
기존 정규화 고유값 그래디언트는 s 편광 dT/depsilon=0.00619276168을 주는데 할선은 0.0061927238을 주어 약
6 ppm의 편향이 있습니다. native 비축퇴 고유값 미분은 그 편향을 없애고 같은 엄격한 시험을 통과합니다. 두 결과 모두
보존하며, 제품 고유값 정규화와 시험 허용 오차는 바꾸지 않았습니다. 이 수치 문제는 약 92%의 접촉 미분 오차를
설명하지 않습니다.

이후의 [산란 배가 구현](nvf-scattering-doubling.md)은 고유벡터 미분을 전혀 쓰지 않고 같은 미분 관문을
통과합니다. 또한 정확한 층 스침을 처리하며, conforming 다항식 몸체 셀과 짝지으면 두 독립 접촉 검사를 모두
유지합니다. 이것은 선택해서 쓰는 연구 경로이며, 위에 기록한 옛 푸리에 접촉 실패와 제품 정규화는 바뀌지
않았습니다.

## 재현

NVF 작업 트리에서 푸리에 셀 한계에는 일반 Wavebit 환경을, conforming 내보내기에는 선택적 NGSolve 환경을 쓰세요.

생성되는 사영 파일을 위해 `NVF_CACHE`를 기존 임시 디렉터리로 설정하세요.

`NVF_FEM_PYTHON`을 선택적 NGSolve 의존성이 있는 Python 실행 파일로 설정하세요.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m studies.rcwa.nvf_cell_bounds.nvf_cell_bounds --inners 31 63 127
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 "$NVF_FEM_PYTHON" -m studies.rcwa.nvf_cell_corrector.nvf_cell_corrector --cap 44 --degree 8 --quadrature 28 --refine 80 --output "$NVF_CACHE/nvf_cell.npz"
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m studies.rcwa.nvf_cell_bounds.nvf_cell_bounds --projection "$NVF_CACHE/nvf_cell.npz" --orders 6 10 14 18 22
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m studies.rcwa.nvf_harmonic_energy_jax.nvf_harmonic_energy_jax --inners 31 127 255 --polarizations s
PYTHONPATH=$PWD python -m pytest -q tests/rcwa/test_nvf_cell_bounds.py tests/rcwa/test_nvf_harmonic_energy.py tests/rcwa/test_nvf_harmonic_energy_jax.py
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" -m unittest discover -s tests -p test_nvf_cell_corrector.py
```

선택한 출력은 `studies/rcwa/nvf_cell_bounds/records/nvf_cell_bounds.log`,
`studies/rcwa/nvf_cell_corrector/records/nvf_cell_projection.log`,
`studies/rcwa/nvf_harmonic_energy/records/nvf_harmonic_energy_s.log`에 있습니다. H=255/511 포함 구간 행과
초기 FEM 세분화 비교는 앞선 로컬 시제품에서 나온 것이며, 최종 degree-8/quadrature-28 사영은 패키지로 묶인 도구를
씁니다.
