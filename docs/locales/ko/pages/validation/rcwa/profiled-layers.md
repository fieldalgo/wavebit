# 프로파일 층: 슬라이스를 무엇과 견주었는가

작성: 2026-11-18. 대상: `src/wavebit/rcwa/_profile.py`, `src/wavebit/rcwa/problem.py`,
`src/wavebit/rcwa/numerics.py`, `src/wavebit/materials/models.py`, `tests/rcwa/test_profiled_layer.py`,
`tests/rcwa/test_graded_materials.py`, `tests/materials/test_bruggeman.py`,
`tests/references/test_reticolo_taper_golden.py`. 점진 재료와 Bruggeman 매질은 2026-11-24에 더했습니다.

**요약.** [`ProfiledLayer`](../../guide/rcwa/profiled-layers.md)는 보통 층을 계단처럼 쌓아 풉니다. 측정 결과는 다음과 같습니다.

- **같은 계단:** RETICOLO V10과 3.5e-14 이내로 같습니다.
- **자동 개수:** RETICOLO로 외삽한 프로파일과 7.3e-5 이내로 맞습니다. 허용오차는 1e-4입니다.
- **수렴 차수:** 계단은 2차로 수렴합니다. 직선 테이퍼에서 2.000, 둥근 프로파일에서는 호 길이 슬라이스로 1.89~2.07입니다. 균일 슬라이스로는 1.35~1.48에 그칩니다.
- **미분:** 자동 미분이 중앙 차분과 4e-9 이내로 맞습니다.
- **수직 측벽:** Rectangle 층과 비트 단위까지 같습니다.
- **높이에 따라 바뀌는 재료:** 선형 점진 막은 슬라이스 512개에서 정확한 Airy 해와 3.3e-7 차이이고 차수 2.0001로 수렴합니다. Bruggeman 거칠기 층은 박막 공식과 3e-16 이내로 같습니다.

기준은 측정 전에 정했으며, 어느 것도 완화하지 않았습니다.

측정 환경은 Linux CPU 하나(x86-64)입니다. Python 3.13.17, NumPy 2.5.5, JAX 0.12.0.1, PyTorch 2.15.0+cpu, Octave 8.4.0, RETICOLO V10(`V10_2025`)을 썼습니다.

## 미리 정한 기준과 결과

| | 기준 | 결과 | 시험 |
| --- | --- | --- | --- |
| (a) | 라운딩 없는 수직 `Trapezoid`는 `Rectangle(size=(cd, Ly))` 층과, 두 윤곽이 같은 `Loft`는 `Polygon` 층과 비트 단위로 같음. 미분을 가진 각을 `pi/2`로 주면 1e-12 이내 | 비트 단위로 같음(모든 `cd_at`, 중심에서 벗어난 라인, Li와 기본 NVF, 각 하나와 각 쌍). 추적된 값: 1e-12 미만 | `test_profiled_layer.py` |
| (b) | 직선 테이퍼에서 관측 차수가 TE와 TM 모두 [1.8, 2.2] 안. 둥근 프로파일을 호 길이로 자르면 1.8 이상 | TE 2.000, TM 1.999. 둥근 프로파일은 가장 촘촘한 단계에서 1.89~2.07 | `test_profiled_layer.py`(둥근 경우는 slow 단계) |
| (c) | 같은 준비에서 자동 미분과 중앙 차분이 상대 1e-6 이내 | `cd` 2.4e-10, `swa` 8.7e-10, `top_rounding` 4.4e-10, `footing` 3.8e-9, `thickness` 2.5e-10(PyTorch). JAX의 `swa`와 `thickness`도 기준 이내. 거칠기 두께 3.7e-10, 부피 분율 2.1e-9, 계면 두께 6.4e-10(PyTorch). JAX의 분율도 기준 이내 | `test_profiled_layer.py`, `test_graded_materials.py`(JAX는 slow 단계) |
| (d) | 같은 균일 계단에서 RETICOLO와 1e-10 이내. 자동 개수가 RETICOLO 외삽 극한과 허용오차 이내 | 16개 경우 중 최악 3.5e-14. 허용오차 1e-4에 대해 2.9e-5~7.3e-5 | `test_reticolo_taper_golden.py` |
| (e) | 미세 분할 극한에서 점진 막과 유효 매질 막이 닫힌 형식과 1e-6 이내 | 선형 점진 막 대 Airy 해: 슬라이스 512개에서 3.3e-7(수직), 3.8e-7(30도). 거칠기 층 대 박막 공식 3e-16. 점진 계면 대 부분층 20 000개 5.7e-8 | `test_graded_materials.py` |
| (f) | NumPy, PyTorch, JAX가 1e-12 이내 | 1e-12 이내 | `test_profiled_layer.py`(JAX는 slow 단계) |
| (g) | `ProfiledLayer` 옆의 래스터 층을 `run`과 `check` 모두에서 `layer.profile_raster`로 거부 | 래스터 층의 경로와 함께 거부함 | `test_profiled_layer.py` |

## 계단의 수렴

관측 차수 `log2(|R_N - R_N/2| / |R_2N - R_N|)`를 개수를 두 배로 늘릴 때마다 잽니다(모든 반사·투과 효율에서의 최댓값). 대상은 `tests/references/_reticolo_taper_cases.py`의 두 라인입니다. 실리콘 비슷한 `eps = 15.2 + 0.16j` 라인이 용융 실리카 위에 있고, 주기 0.5 um, 파장 0.6 um, 차수 31개, Li 인수분해를 씁니다. `taper`는 80도 직선 테이퍼입니다. `rounded`는 두 측벽이 82도와 86도이고, 상단 라운딩 0.03 um와 footing 0.02 um가 있습니다. 열은 N = 128, 256, 512에서 끝나는 세 단계 창입니다.

| 프로파일 | 규칙 | 빛 | 128 | 256 | 512 |
| --- | --- | --- | --- | --- | --- |
| taper | 균일 또는 호 길이 | TE, 0도 | 1.996 | 1.999 | 2.000 |
| taper | 균일 또는 호 길이 | TM, 0도 | 1.989 | 1.997 | 1.999 |
| rounded | 균일 | TE, 0도 | 1.791 | 1.590 | 1.479 |
| rounded | 균일 | TE, 20도 | 1.453 | 1.365 | 1.389 |
| rounded | 균일 | TM, 0도 | 1.191 | 1.284 | 1.354 |
| rounded | 균일 | TM, 20도 | 1.256 | 1.351 | 1.408 |
| rounded | 호 길이 | TE, 0도 | 1.900 | 2.115 | 2.066 |
| rounded | 호 길이 | TE, 20도 | 1.787 | 1.852 | 1.995 |
| rounded | 호 길이 | TM, 0도 | 1.629 | 1.826 | 1.890 |
| rounded | 호 길이 | TM, 20도 | 1.724 | 1.864 | 1.925 |

직선 테이퍼의 호 길이는 높이에 비례하므로 두 규칙이 같은 곳을 자릅니다. 원호는 수평인 윗면과 만나는 곳에서 폭이 거리의 제곱근으로 변하고, 균일 슬라이스는 바로 여기서 차수를 잃습니다. 관측값이 1.4 쪽으로 내려가는데, 극한에서 차수 1.5라는 것과 맞습니다. 같은 호 길이 간격으로 자르면 윤곽이 꺾이는 곳에 슬라이스가 몰리고, 차수가 2 쪽으로 돌아옵니다. 계단 근사의 수렴이 프로파일에 따라 어떻게 달라지는지는 Popov, Nevière, Gralak, Tayeb(J. Opt. Soc. Am. A **19**, 33 (2002))가 설명합니다. 여기의 측정은 이 솔버로 이 두 유전체 라인에서 잰 것입니다.

자동 개수가 멈추는 기준인 Richardson 추정은 실제 오차를 잘 따라갑니다. 예를 들어 taper, TE, 수직 입사에서 솔버가 고른 개수의 답은 RETICOLO 외삽 극한과 3.79e-5 떨어져 있었고, 추정치는 3.83e-5였습니다. 여덟 경우 모두 추정치와 실제 거리가 15% 이내로 맞습니다.

## 외부 기준

`tests/references/gen_goldens_reticolo_taper.py`는 각 슬라이스를 자기 기하로 만듭니다. `_reticolo_taper_cases.py`에서 라운딩의 중심은 반지름만큼 옮긴 두 직선의 교점으로 구하며, 솔버의 접선 길이 공식을 쓰지 않습니다. 그리고 슬라이스마다 정확한 1D 텍스처를 RETICOLO에 넘기는데, RETICOLO의 `+z`는 스택 바깥을 향합니다. 생성기는 제품 import를 막은 채로 돌렸습니다. 이 기하와 솔버의 슬라이스는 프로파일마다 높이 2001곳에서 5.6e-17 이내로 같습니다. RETICOLO의 64, 128, 256 균일 슬라이스는 관측 차수(최대 2)로 외삽합니다. 값과 출처는 `tests/references/golden_reticolo_taper.npz`와 그 `.reference.json`에 있습니다.

## 높이에 따라 바뀌는 재료

솔버를 부르지 않는 닫힌 형식 세 가지와 견줍니다(`tests/rcwa/test_graded_materials.py`).

- 유전율이 선형으로 올라가는 막: 용융 실리카 위, 공기 아래에서 0.3 um 동안 `eps = 2.25 -> 4.0 + 0.1j`, TE, 파장 0.6 um입니다. `eps(z)`가 선형이면 필드 방정식 `E'' + k0^2 (eps(z) - sin^2(theta) eps_c) E = 0`이 Airy 방정식이라 반사율을 정확히 구할 수 있습니다. 솔버의 균일 슬라이스 128, 256, 512개는 수직 입사에서 각각 5.3e-6, 1.3e-6, 3.3e-7 떨어져 있고(30도에서는 512개에서 3.8e-7), 관측 차수는 2.0001입니다. 이들의 Richardson 외삽은 6e-12 차이입니다.
- Bruggeman 거칠기 층: 같은 재료 80 nm 위에 실리콘 비슷한 `eps = 15.2 + 0.16j`와 빈 공간을 분율 0.3, 0.5로 섞은 4 nm 층입니다. 유효 유전율을 `numpy.roots`로 따로 구한 특성 행렬 박막 공식(Born and Wolf, 1.6절)과 3e-16 차이입니다.
- 실리콘이 용융 실리카에 선형으로 섞이는 10 nm 계면 층(`rcwa.Graded.interface`), 20도 입사: 부분층마다 Bruggeman 근을 따로 구한 부분층 20 000개의 특성 행렬과, 슬라이스 512개에서 5.7e-8 차이입니다.

Bruggeman 모델 자체(`tests/materials/test_bruggeman.py`)는 혼합 방정식을 1e-14까지 만족하고, `numpy.roots`와 값의 1e-13 이내로 같습니다. 두 상을 바꿔도(`f <-> 1 - f`) 같은 값을 주고, `f = 0`과 `f = 1`에서 구성 재료를 돌려주며, Drude 구성 재료를 파장마다 평가합니다. 분율에 대한 PyTorch와 JAX 미분은 중앙 차분과 1e-8 이내로 맞습니다.

## 비용

슬라이스 하나, 파장 하나, 차수 31개, CPU 코어 하나 기준입니다. `factorization='li'`로 약 10 ms, 기본 NVF로 약 1.1 s가 걸리며, NVF 시간은 거의 다 형상 법선장(`parametric._field_tiled`) 계산입니다. 라인의 슬라이스에서 둘의 답은 1e-14 이내로 같습니다.

JAX의 즉시 실행은 층당 약 4 s가 걸리고, `jax.jit` 컴파일 시간은 슬라이스 수에 따라 늘어납니다. 손으로 쓴 층에서도 똑같으므로(층 네 개에서 16.0 s 대 16.0 s) 슬라이스 탓이 아닙니다. 슬라이스의 고유값 문제는 다른 모든 스택처럼 층마다 하나씩 풉니다. 이를 묶어서 푸는 것은 별도 변경입니다.

## 여기서 확인하지 않은 것

- 비스듬한 방위각(원뿔 입사)의 라인과 2차원 `Loft` 기둥을 외부 솔버와 견주지 않았습니다. `Loft`와 `Profile`의 슬라이스는 `Polygon`이며, `Polygon` 자체의 외부 확인은 [경계 모델링](../../guide/rcwa/boundary-modeling.md#polygon-regions)에 있습니다.
- 금속은 확인하지 않았습니다. 금속에서는 TM 계단이 느리게 수렴하거나 수렴하지 않을 수 있습니다(Popov 등).
- GPU 시간과 메모리는 재지 않았습니다.
