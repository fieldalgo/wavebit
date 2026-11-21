# conforming NVF 모델의 강체 격자 자세와 변수로 둔 입사

`studies/rcwa/nvf_posed_gap/nvf_posed_gap.py`는 다항식 본체 NVF 연구 모델을 실험실 좌표의
평행 이동, 패턴을 **정사각 격자와 함께** 돌리는 강체 회전, 그리고 변수로 둔 극각/방위각 입사로
확장합니다. 복소 반사/투과 진폭과 p/s 전력은 같은 셀 방정식, Gram 닫힘(closure), 고유벡터를 쓰지 않는
산란 전파를 사용합니다. 고정된 격자 안에서 개별 기본 도형을 돌리거나 제품 디스패치를 바꾸지는
않습니다.

## 좌표, 위상과 편광

국소 기하는 여전히 순서가 정해진 재료 이음매를 가진 수평 직사각형 외곽입니다. 폭, 틈과 이음매
좌표의 단위는 nm이고, 모든 각도는 라디안입니다. 격자 회전을 alpha라 하면 국소 입사 방위각은
`lab_azimuth - alpha`입니다. 실험실 좌표의 평행 이동 d는 `R(alpha).T @ d`로 바뀐 뒤,
모든 필드/에너지 푸리에 계수에 다음을 곱합니다.

```text
exp(-2 pi i / period * (m d_local_x + n d_local_y)).
```

중앙 입사파에 대해서는 같은 위상이 출사 회절 진폭에도 나타납니다. 전력은 평행 이동에 대해
불변이고, 진폭의 미분은 `-i G_lab_x A`와 `-i G_lab_y A`입니다. 테스트는 소멸파 차수를 포함한
모든 복소 차수를 이 해석적 항등식과 비교합니다. 또 다른 테스트는 텐서 성분, 푸리에 인덱스와
방향별 차수를 4분의 1 회전만큼 독립적으로 돌린 뒤 복소 진폭을 비교합니다.

`doubling_response`는 변수 `theta`와 `azimuth`를 받아 `T`, `R`, `reflected`, `transmitted`,
`energy_error`, `taylor_tail_bound`를 반환합니다. 진폭 행은 모든 s 차수 다음에 모든 p 차수가
오고, 열은 p 입사 다음에 s 입사입니다. 편광 성분은 회전한 국소 좌표계에서 각 출사 차수를 기준으로
합니다. 실험실 직교 좌표의 벡터 필드를 만들 때는 이 좌표계를 변환하세요. `doubling_powers`는 이전의
튜플 API를 그대로 유지합니다.

중앙 입사 편광 좌표계는 지정한 방위각을 그대로 쓰므로 수직 입사에서도 미분 가능합니다.
theta는 부호가 있으며 `abs(theta) < pi/2`입니다. 테스트는 수직 입사, 부호가 있는 비스듬한 입사,
0이 아닌 방위각, 그리고 정확히 스치듯 입사하는 **층**을 독립적인 균일 슬랩 공식으로 다룹니다.
정확히 스치듯 진행하는 **외부 회절 차수**는 여전히 제외합니다. 그 플럭스 정규화가 특이하기
때문입니다. NumPy는 예외를 내고 JIT는 NaN을 반환합니다. 주기, 파장, 외부 매질과 무손실 계약은
고정이며, 기존의 선택적 두께 설정은 유지됩니다.

## 변수 매개변수 검사

차수 2, 틈 17 nm, 폭 350 nm, 이음매 31 nm, 재료 4/9, 중심 `(17,-21)` nm, 격자 회전 .31,
theta .2, 실험실 방위각 .47에서:

| 양 | p | s |
| --- | ---: | ---: |
| T | 0.943312026142 | 0.904149819534 |
| R | 0.056687973858 | 0.095850180466 |
| dT / d 격자 회전 | -0.0451392740874 | 0.0211718902307 |
| dT / d theta | -2.05364981620 | 0.962735701302 |

이 값은 기본 행 4, 본체 차수 4, 틈 차수 6, x 세분 1, 가장자리 비율 1e-5와 grading 8/.01의
원래 전역 플럭스 기저를 씁니다. 전력 불균형은 5e-14 미만입니다. 10개 매개변수 전체에 대한 JIT
야코비안은 틈, 폭, 두 평행 이동, 회전, 두 입사각, 이음매, 두 유전율을 다룹니다. 이 야코비안은
바꾸지 않은 `atol=2e-9, rtol=3e-6` 기준으로 독립적인 두 모드 NumPy 할선(secant)과 일치합니다.
평행 이동에 대한 전력 미분은 0이며, 회전과 실험실 방위각에 대한 미분은 부호가 반대입니다.

이 비교들은 명시한 유한 모델의 미분을 검증할 뿐, 셀이나 외부 푸리에 수렴을 확립하지는 않습니다.
모드 비교는 구성(constitutive) 필드는 재사용하지만 전파는 독립적입니다. 균일 슬랩 검사와 공변성
검사는 별도의 물리적 대조군을 제공합니다.

## 혼합 재료 접촉과 안정적인 세분

단일 재료의 비스듬한 접촉 테스트는 nm당 1e-7이라는 독립적인 직교 좌표 접촉 관문을 그대로
유지합니다. 이 검사를 재료 4/9로 넓히자 실제 내부 이음매 근처에서 더 큰 불일치가 드러났습니다.
이 불일치는 방위각 0에서도 나타나므로 강체 자세가 원인은 아닙니다. 외부 차수를 2에서 7로 올려도
사라지지 않으며, 보조 셀을 세분하면 줄어듭니다.

[국소화한 플럭스 기저](nvf-local-flux.md)는 시험 공간을 정확히 그대로 유지하면서 더 날카로운
재료 grading을 수치적으로 쓸 수 있게 해 줍니다. 국소 방위각 .37의 혼합 재료 fixture에 대해,
아래의 차수 2 비교는 기본 행 8, 본체 차수 4, x 세분 3, 틈 차수 6, 가장자리 비율 1e-5를 씁니다.
기울기는 틈 0, 1e-4, 5e-5 nm와 한쪽 방향 외삽으로 구합니다.

| 재료 grading 반경 / 폭 | 직교 좌표 대비 p 기울기 차이 | 직교 좌표 대비 s 기울기 차이 |
| ---: | ---: | ---: |
| .01 | 3.791e-7 | 2.609e-6 |
| .001 | 4.179e-8 | 2.866e-7 |
| .0001 | 4.959e-9 | 2.915e-8 |

마지막 행은 두 편광 모두에서 바꾸지 않은 접촉 기준을 통과합니다. 같은 x 세분을 쓰는 더 작은
기본 행 4/본체 차수 1 셀로 JIT 접촉 미분도 국소 방위각 0과 .37에서 통과합니다. 이 작은 셀은
이 접촉 대조군에는 충분하지만, 양의 틈에서의 셀 수렴을 대신하지는 않습니다. 직교 좌표 결과는
NVF 계산에 전혀 들어가지 않으며, 비교에 맞추려고 재료 미분을 고정하거나 조정하지 않았습니다.

기본값 .01 grading은 여전히 거친 연구 설정입니다. 더 촘촘한 격자에서의 통과 결과가 그 혼합 재료
접촉 기울기를 보증하지는 않습니다. 기하, 대비, 푸리에 차수 전반에 걸친 일반적인 혼합 재료 접촉도
보증되지 않습니다. `--contact-check`는 후보와 독립적인 유한 차수 대조군을 보고하며, 둘의 일치를
일부러 수렴한 광학 결과라고 표시하지 않습니다.

## 사용과 재현

```python
from studies.rcwa.nvf_posed_gap.nvf_posed_gap import PosedGapModel

model = PosedGapModel(
    2, rows=4, body_degree=4, gap_degree=6, edge_fraction=1e-5,
)
response = model.response(
    17., dict(seams=[31.], values=[4., 9.]),
    center=(17., -21.), rotation=.31, theta=.2, azimuth=.47,
)
# response['T'] and ['R']: vectors in p, s order.
# model.transmission(..., polarization='p'/'s'/'both') selects powers only.
# Pass xp=jax.numpy with x64 enabled for live derivatives.
```

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_posed_gap.nvf_posed_gap --jax
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_posed_gap.nvf_posed_gap --rows 8 --subdivisions 3 --flux-basis local --grading-radius .0001 --azimuth .68 --contact-check
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.rcwa.nvf_posed_gap.nvf_posed_gap --gap 0 --body-degree 1 --subdivisions 3 --flux-basis local --grading-radius .0001 --azimuth .68 --jax --contact-check
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/rcwa/test_nvf_posed_gap.py tests/rcwa/test_nvf_local_flux.py
```

측정값은 `studies/rcwa/nvf_posed_gap/records/nvf_posed_gap.log`,
`nvf_posed_contact_refinement.log`, `nvf_posed_contact_local_flux.log`에 있습니다.
자세 테스트 묶음은 14개, 국소화 플럭스 테스트 묶음은 7개이며, 여기에는 두 번째 혼합 재료 접촉
프로파일도 포함됩니다. 2026-09-09에 자세, 국소/전역 플럭스, 재료 grading, 다항식 공간, 산란,
거울, 틈 수렴과 이전 조화 에너지 테스트 묶음 전체를 검증한 결과는 **148개 통과,
2개 strict xfail** 입니다. 예상된 두 실패는 바꾸지 않은 이전 푸리에 필드의 s 접촉 한계와 정칙화한
모드 재료 미분이며, 다른 이름으로 바꾸어 표시하지 않았습니다. 이 실행은
`studies/rcwa/nvf_posed_gap/records/nvf_pose_local_flux_validation.log`에 보존되어 있습니다.

제품 통합,
고정된 셀 안에서의 독립적인 회전, 임의의 경계면, 그리고 이전의 고차 광학 수렴 관문은 아직 열려
있습니다.
