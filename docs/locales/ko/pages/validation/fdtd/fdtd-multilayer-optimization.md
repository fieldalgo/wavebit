# 제약이 있는 다층 최적화

작성일: 2026-09-09

대상:
- `src/wavebit/_common/periodic.py`
- `src/wavebit/rcwa/_engine/factorization.py`
- `examples/fdtd/fdtd_multilayer_optimization.py`
- `studies/fdtd/multilayer_optimization/fdtd_multilayer_optimization.py`
- `tests/common/test_layer_thicknesses.py`
- `tests/fdtd/test_fdtd_multilayer_optimization.py`
- `tests/rcwa/test_factorization_core.py`
- `tests/references/test_meep_multilayer.py`
- `tests/references/interface/meep_multilayer.py`

요약: `wavebit.layer_thicknesses`는 일반 NumPy/JAX/Torch 설계 가중치를 전체 합이 고정되고 선택적으로
개별 최솟값을 가지는 양의 길이로 바꿉니다. 같은 배열이 RCWA와 FDTD에 함께 들어갑니다. 변수 6개짜리 유전체
코팅 예제는 81개 샘플 파장에 걸친 독립적인 날카로운 적층의 평균 반사율을 5.1047%에서 0.023705%로
줄였습니다. 더 세밀한 FDTD 메시와 독립적으로 측정한 Meep 미분이 이 특정 설계를 뒷받침합니다. 최적화기는
30회 반복 한도에서 멈췄습니다. 이것은 개선된 설계이지 수렴 보증서가 아닙니다.

## 공유 기하 좌표

층 N개, 정적 전체 두께 D, 최소 길이 m, 실수 가중치 u에 대해 다음과 같이 정의합니다.

```text
w_i = exp(u_i - max(u)) / sum_j exp(u_j - max(u))
d_i = m_i + (D - sum(m)) w_i
∂d_i/∂u_j = (D - sum(m)) w_i (δ_ij - w_j).
```

가중치는 무차원입니다. 공통 이동은 중복된 자유도이므로, 예제는 마지막 가중치를 0으로 고정합니다. 출력은
일반 두께 배열이며, 기존 `Stack`과 두 솔버가 모두 받아들입니다. 별도의 FDTD 기하 표현은 없습니다. 전체
두께와 최솟값은 정적 호스트 제약이며, 가중치만 미분합니다.
[입력 계약](../../migration/autodiff-v1.md#constraining-several-layer-thicknesses)을 참고하세요.

해석적 Jacobian은 세 백엔드에 대한 독립 테스트를 제공하며, 열 합이 0이라는 점과 공통 이동의 영공간 방향도
검사합니다. 양의 최솟값은 지수 함수가 포화될 때 층이 사라지는 것을 막지만, 작은 그래디언트까지 막지는
못합니다. 길이가 0으로 언더플로하는 경우는 거부합니다. 제약은 부동소수점 산술로 계산하므로, 물리적 FDTD
비교에는 float64를 사용하세요. 잘못된 추적(traced) JAX 입력은 조용히 잘라낸 기하 대신 NaN 길이를 만듭니다.

유형 지정 RCWA 검사는 `jax.jit`와 Torch를 거쳐 길이 미분을 보존하는지 확인하고, 두 유한 차분 간격과
비교합니다. 이 검사로 균일한 1픽셀 래스터에서 JAX CPU FFT 레이아웃 실패가 드러났습니다. 샘플 하나짜리 DFT는
항등 변환이므로, 이제 공통 푸리에 생성기는 크기 1인 축에서 복소수 승격과 함께 그 항등 변환을 사용합니다.
독립적인 정확한 픽셀 적분이 컴파일된 JAX 실행을 포함해 1×1, 1×4, 4×1 래스터에서 세 백엔드를 모두
검사합니다.

## 변수 6개 코팅 예제

여기 기록된 층 배열은 과거의 위에서 아래 순서를 따릅니다. 즉 코팅이 먼저, 기판 스페이서가 마지막입니다.
현재 공개 `Stack` 배열은 아래에서 위 순서를 사용하므로, 그 경계에서 유전율 배열과 두께 배열을 모두 뒤집어야
합니다. 예제는 원래 설계 매개변수 순서를 유지하고 이 변환을 명시적으로 수행합니다.

장면은 유전체 코팅 3개와 그 아래의 기판 스페이서로 이루어집니다. 위쪽 epsilon은 1, 기판 epsilon은 4이고,
전체에서 mu=1입니다. 전체 깊이는 0.8이고 모든 층의 최소 길이는 0.04입니다. 모든 길이와 파장은 같은 임의의
물리 단위를 사용합니다.

자유 두께 가중치 3개와 실수 코팅 유전율 3개가 설계 벡터를 이룹니다. 가중치 범위는 [-4,4]이고 epsilon
범위는 [1.1,4]입니다. 스페이서 epsilon은 4로 유지되며, 그 가변 길이 덕분에 두 외부 평면이 고정됩니다. 초기
길이는 [0.08,0.10,0.12,0.50]이고 코팅 epsilon은 [2.7,1.6,3.2]입니다. 재료는 연속 최적화 변수이며, 실제로
구할 수 있거나 제작 가능한 재료의 이산 목록이 아닙니다.

목적 함수는 수직 입사와 세 파장 [0.8,1.0,1.2]에서의 s 편광 평균 반사율입니다. FDTD 설정은 해상도 40, 공칭
길이 [0.2,0.2,0.2,0.2], 맞춘 셀, 방향성 추출, CPML 1 단위, 펄스 폭 3, 지속 시간 65, 0차 추출을 사용합니다.
내부 평균은 물리 폭 h=1/resolution을 가지며, 각 최적화 동안 전체 두께, 소스, 메시, 출력 평면은 고정됩니다.
가이드는 [경계면 평균](fdtd-layer-smoothing.md)이 샘플링된 물리 분포를 바꾸면서도 연속적인 1차 미분을
주는 이유를 설명합니다.

예제는 목적 함수와 JAX 역방향 모드 그래디언트를 모두 SciPy의
[L-BFGS-B 인터페이스](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)에 넘기며,
`ftol=1e-12`, `gtol=1e-7`, 반복 예산 30회를 사용합니다. 반환된 상태와 메시지를 보고합니다. 이 실행은 SciPy가
보고한 목적 함수 평가 36회에 더해, 기록된 이력을 위한 콜백 평가를 사용했습니다. 실행은
`STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT`와 `optimizer_success=False`로 멈췄습니다.

| 양 | 초기 | 30회 반복 후 |
| --- | ---: | ---: |
| 학습 평균 R, 해상도 40 | 0.0522222 | 0.000224965 |
| 독립 날카로운 적층 평균 R, 파장 81개 | 0.0510468 | 0.000237054 |
| 독립 날카로운 적층 최대 R, 파장 81개 | 0.0912548 | 0.000540387 |

최종 물리 길이는 약 [0.0793044472,0.1541410422,0.1383468378,0.4282076728]이고, 코팅 epsilon은
[1.2308075548,1.5211291788,2.8040978636]입니다. 보존된 기준 보관 파일에는 전체 정밀도의 설계 벡터와 그로부터
복원한 기하가 저장되어 있습니다.

| 파장 | 학습 R, 격자 40, h=0.025 | 격자 80, h=0.0125 | 격자 80, 날카로움 | 날카로운 연속체 |
| --- | ---: | ---: | ---: | ---: |
| 0.8 | 0.0000444441 | 0.0000341833 | 0.0000366224 | 0.0000310290 |
| 1.0 | 0.000421000 | 0.000433841 | 0.000429257 | 0.000437442 |
| 1.2 | 0.000209450 | 0.000215167 | 0.000219167 | 0.000217510 |

81점 검사는 [0.8,1.2]를 간격 0.005로 샘플링하며, 학습에 없던 파장을 포함합니다. 이것으로 연속 대역의
최댓값이나 전역 최적이 증명되지는 않습니다. 이 실행은 arm64 CPU에서 Python 3.12.9, NumPy 2.5.1, SciPy 1.18.0,
JAX 0.11.0, Torch 2.13.0과 float64/complex128을 사용해 측정했습니다. GPU 실행은 이 측정에 포함되지
않았습니다.

## 독립 연속체 검사와 미분 검사

예제는 마지막 날카로운 적층 검사에 Fresnel 다중 반사 점화식을 사용합니다. 연구 스크립트는 따로 접선 E/H 쌍을
2×2 특성 행렬로 각 층을 통과시켜 전파합니다. 두 정식화는 서로, 그리고 RCWA와 반올림 수준까지 일치하고,
에너지를 보존하며, 균일한 스페이서를 나누어도 진폭을 보존합니다. 둘 다 FDTD 샘플링 가중치나 그 그래디언트를
사용하지 않습니다.

양의 h에 대해 연구 스크립트는 경사 끝점 사이의 구간마다 균질 슬라이스 128개를 사용해 선형 접선 epsilon
분포를 적분하며, 이웃한 경사가 겹치는 것을 허용합니다. 슬라이스 256개 계산으로 이 기준 자체의 수렴을
검사합니다. 전부 균일한 극한은 알려진 Fresnel 반사와 투과 위상을 가집니다. 이 기준들에서 유전율은 양의
실수이고 등방성이므로, 이 측정은 일반적인 이방성 또는 분산 설계 결과를 확립하지 않습니다.

아래 각 행은 세 반사율 모두와 그 전체 3×6 Jacobian을 비교합니다. 보고한 오차는 최대 절대 오차입니다.
가중치 미분과 상대 유전율 미분은 위에서 정의한 좌표에 대한 것이며, 단위 두께당 민감도가 아닙니다. 날카로운
기준은 중앙 간격 1e-5와 5e-6을 사용하며, 둘 사이의 최대 불일치는 1.39e-11입니다. 단계화 기준에서
슬라이스를 두 배로 늘렸을 때 미분의 최대 변화는 5.68e-7입니다. 둘 다 측정된 FDTD 미분 오차보다 작습니다.

| 설계 | 격자 | 날카로운 기준 대비 R 오차 | 날카로운 기준 대비 Jacobian 오차 | 단계화 기준 대비 R 오차 | 단계화 기준 대비 Jacobian 오차 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 초기 | 20 | 0.0104958 | 0.0203475 | 0.00691242 | 0.0164606 |
| 초기 | 40 | 0.00235935 | 0.00381376 | 0.00144959 | 0.00236870 |
| 초기 | 80 | 0.000626139 | 0.00127131 | 0.000398677 | 0.000986717 |
| 최적화 | 20 | 0.0000854816 | 0.00260481 | 0.0000834278 | 0.00296265 |
| 최적화 | 40 | 0.0000164420 | 0.000723824 | 0.0000154745 | 0.000715861 |
| 최적화 | 80 | 0.00000360105 | 0.000162521 | 0.00000384282 | 0.000182350 |

여기서 h는 격자 간격과 함께 줄어듭니다. 대신 h를 고정하면 단계화 재료 쪽으로 수렴하므로, 세분화만으로는
날카로운 적층의 미분을 확립할 수 없습니다. 격자 80에서도 최적화된 단계화 연속체와 날카로운 연속체의
Jacobian은 최대 0.000117274까지 차이가 납니다. 연구 스크립트는 이 분포 편향을 수치 이산화 오차와 따로
기록합니다.

별도의 JAX 및 Torch 테스트는 아래쪽 입사, 0이 아닌 theta/phi, 복소 Jones 조명을 사용합니다. 결합 변수 6개
모두 0이 아닌 미분을 가지며, 두 유한 차분 간격이 AD와 일치합니다. 이 변수들을 섭동해도 격자 좌표, 소스/PML
위치, 시간 간격, 단계 수, 공칭 기하는 유지됩니다. 이 그래프 검사는 수직 입사 물리 기준을 보완합니다.

## Meep 비교와 기준의 불확도

Meep 1.34.0이 최종 설계와 중앙 섭동 6개 모두를 독립적으로 계산했습니다. 기준은 문서화된 전역
[재료 함수](https://meep.readthedocs.io/en/latest/Python_User_Interface/#medium)와
[재료 평균](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/)을 각 방향으로 횡방향 셀 하나를 가진 3D
주기 셀에서 사용합니다. 격자 80, h=0.0125, 길이 10 셀, z PML 1 단위, Courant 0.4를 명시합니다. 횡방향 전체를
덮는 Ex 소스는 z=-3에 있고, DFT 평면은 -1과 1.4에 있습니다. 가우시안 창을 씌운 펄스는 폭 1.2, 최고점 시각 6,
종료 시각 12를 가집니다. 독립적으로 측정한 균일 위쪽 기준과 이산 Yee 전파 위상을 사용해 r을 z=0에, t를
z=0.8에 놓습니다. 기준 메타데이터에 이 설정이 저장되어 있습니다.

| 검사 | 최대 절대 변화 또는 차이 |
| --- | ---: |
| Meep 지속 시간 50 대 70, 복소 r/t | 5.29e-17 |
| Meep 지속 시간 50 대 70, 복소 Jacobian | 6.25e-14 |
| Meep 섭동 간격 1e-3 대 5e-4 | 0.000120765 |
| Meep 평균 허용 오차 1e-8 대 1e-9, 복소 r/t | 4.18e-11 |
| Meep 평균 허용 오차 1e-8 대 1e-9, 복소 Jacobian | 2.72e-7 |
| Wavebit 대 Meep, 복소 r/t | 0.00513490 |
| Wavebit 대 Meep, 복소 Jacobian | 0.00239551 |
| Wavebit 대 단계화 연속체, 복소 r/t | 0.00947685 |
| Meep 대 단계화 연속체, 복소 r/t | 0.0102665 |
| Wavebit 대 단계화 연속체, 복소 Jacobian | 0.00471004 |
| Meep 대 단계화 연속체, 복소 Jacobian | 0.00453652 |

복소 진폭 오차는 위상을 포함하며 위의 반사율 오차와는 다릅니다. 두 유한 격자 솔버는 각각 독립 연속체 분포와도
따로 검사합니다. 서로 일치하는 것만으로는 충분하지 않습니다. 커밋된 보관 파일에는 두 평균 허용 오차, 두 지속
시간, 두 섭동 크기가 모두 들어 있습니다. 일반 CI는 Meep을 설치하지 않고 이 파일을 읽습니다. Meep의 문서와
블랙박스 공개 API만 사용했습니다. Meep 구현이나 제외된 Meep 유래 FDTDX 투영 소스는 열거나 복사하거나
Wavebit에 링크하지 않았습니다.

## 재현

Wavebit의 JAX 추가 구성(extra)과 SciPy가 있는 환경에서 저장소 루트에서 실행합니다.

```shell
python -m examples.fdtd.fdtd_multilayer_optimization > artifacts/design.json
python -m studies.fdtd.multilayer_optimization.fdtd_multilayer_optimization --design artifacts/design.json --output artifacts/multilayer-check.json
pytest -q tests/common/test_layer_thicknesses.py tests/fdtd/test_fdtd_multilayer_optimization.py tests/references/test_meep_multilayer.py
```

새 최적화기 궤적 대신 보존된 정확한 설계를 검사하려면, 먼저 독립 기준 보관 파일에서 입력 매개변수만
내보냅니다.

```python
import json
from pathlib import Path
import numpy as np

with np.load('tests/references/golden_meep_multilayer.npz') as reference:
    parameters = reference['parameters'].tolist()
Path('artifacts/design.json').write_text(json.dumps({'parameters': parameters}))
```

따로 설치한 Meep 환경에서 저장소 밖에 후보 기준 파일을 다시 생성하고, 보관 파일을 교체하기 전에 비교합니다.

```shell
python tests/references/interface/meep_multilayer.py --design artifacts/design.json --output artifacts/meep-multilayer.npz
python tests/references/interface/meep_multilayer.py --design artifacts/design.json --subpixel-tol 1e-9 --output artifacts/meep-multilayer-refined.npz
```
