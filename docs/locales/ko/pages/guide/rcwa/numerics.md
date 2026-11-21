# RCWA 수치 설정 고르기

물리적 기하와 빛은 `rcwa.Problem`에 속하고, 푸리에 절단과 알고리즘은 `rcwa.Numerics`에 속합니다.

```python
from wavebit import rcwa
numerics = rcwa.Numerics(
    orders=(1, 1), algorithm="smm", factorization="nvf",
    formulation="auto", shape_nvf_grid_size=64)
```

`orders=(Mu, Mv)`는 첫 번째 격자 벡터를 따라 −Mu…Mu 차수를, 두 번째 격자 벡터를 따라 −Mv…Mv 차수를 유지합니다. 이 값은 항상 음이 아닌 두 정수의 명시적인 쌍입니다(`numerics.orders`는 정수 하나만 주면 거부합니다). 결과 배열은 파장 축을 맨 앞에, 그다음 v, 그다음 u를 둡니다. 알고리즘은 `"smm"` 또는 `"etm"`이고, 정식화는 `"auto"`, `"first_order"` 또는 `"second_order"`입니다. 인수분해 방식은 명시적으로 고르며, 재료와 기하에 대해 지원 여부를 검사합니다. 예제의 작은 설정값은 빠른 계산을 위한 것이지, 수렴을 위한 권장값이 아닙니다.

풀이 경로는 설정이 아닙니다. 계산이 순수 1D TE/TM 경로, 1D conical 경로, 2D 경로 중 무엇을 택할지는 구조와 소스가 정합니다(방법은 [빛 가이드](light.md#the-words-are-a-declaration-the-pair-is-a-value)에 있습니다). `rcwa.check(...).resolved['solve_path']`는 계산이 택할 경로를 알려 주고, `result.diagnostics['solve_path']`는 각 파장이 실제로 택한 경로를 기록합니다. 값은 `'1d_te'`, `'1d_tm'`, `'1d_conical'` 또는 `'2d'`입니다.

레일리 디튜닝(detuning)은 사용자가 켤 때만 적용됩니다. `detuning=None`은 정확한 물리 입력을 요청합니다. 양수 `detuning`은 상대 이동량을 선언하며, 결과는 바뀐 좌표를 기록합니다. 그 메타데이터를 읽지 않은 채 디튜닝된 결과를 요청한 파장 그대로의 결과로 해석하지 마세요.

`topology_check='device'`는 사용할 수 있는 배열 검사를 쓰고, 그 밖의 추적된 기하는 거부합니다.
명시적인 `'host'`는 지원되는 스칼라 영역 기하를 JAX CPU 콜백으로 받아들입니다.
vmap 아래에서는 검사가 순차적으로 실행되므로 처리량이 제한될 수 있습니다. 유효성 마스킹, 지원되는 형상, 예제는
[JAX 변환 안에서 형상 위상 검사하기](guarded-shape-jit.md)를 참고하세요.

재료가 바로 옆 반공간의 재료와 같은 층은 복소 각도에서 그 반공간에 병합됩니다. 나머지 적층을 풀고, 반사파, 투과파, 내부 파를 병합된 두께에 걸쳐 해석적으로 옮깁니다. 이것이 정확한 답이며, 층별 풀이는 이 경우 O(1)만큼 틀릴 수 있습니다. `half_space_merge_tolerance`(기본값 `0.0`, 비트 단위로 동일한 재료만)는 같은 재료로 보는 범위를 넓힙니다. 양의 상대 허용 오차를 주면 가까운 재료를 모든 각도에서 반공간의 재료로 간주하므로 문제가 바뀝니다. `result.diagnostics["half_space_merge"]`는 파장마다 각 반공간에 몇 개의 층이 병합되었고, 그중 어느 층이 허용 오차로 맞춰졌는지 기록합니다.

`wall_resolution`(기본값 `None`)은 축에 정렬된 직사각형들의 적층, 또는 원 하나나 축 정렬 타원 하나의 적층을, 좌표선이 벽 근처에 몰리는 좌표계에서 풉니다. 이렇게 하면 늘리지 않은 풀이가 수십 퍼센트 틀리는 차수에서도 금속 옆 근접장이 유한 요소 기준값에 맞춰집니다. 이 기능은 실험적이며, 산란 행렬 알고리즘에서 `factorization='li'`를 쓰고, 소멸파 차수의 진폭을 0으로 보고합니다. [늘린 좌표로 벽 분해하기](wall-resolution.md)를 참고하세요.

`exceptional_points`(기본값 `'rebuild'`)는 층 모드들이 고유벡터를 공유하는 예외점과 그 근처에 있는 층의 처리를 정합니다. 기본적으로 그런 층은 정칙 기저에서 다시 구성되며, 다시 구성할 수 없는 경우에는 풀이가 (즉시 실행에서) 경고하거나 (추적 중에는) NaN을 답합니다. `'off'`는 모든 경로에서 이 처리를 건너뛰는 고급 선택으로, 주로 컴파일된 계산의 속도를 높이려고 씁니다. 그러면 예외점과 그 근처에서 답이 경고 없이 틀릴 수 있습니다. {ref}`예외점 §7 <ep-handling-off>`을 참고하세요.

차수와 기하/NVF 분해능은 따로따로 늘리고, 실제로 필요한 출력을 비교하세요. 에너지 수지만으로는 광학적 수렴이나 그래디언트 수렴이 확립되지 않습니다. 아래에 인용한 측정값은 같은 엔진을 실행하는 이전 `wavebit.compat.legacy` 인터페이스로 얻은 것입니다.

## 각 설정이 바꾸는 것

몇 개의 키워드가 유전율을 합성곱 행렬로 바꾸는 방식과 층을 잇는 방식을 정합니다.
이들은 결과가 `orders`에 따라 얼마나 빨리 수렴하는지와 각 단계에 드는 비용을 바꿀 뿐,
무엇으로 수렴하는지는 바꾸지 않습니다. 기본값과 각 키워드가 받는 값은
[키워드 레퍼런스](../../reference/keywords.md#the-numerics)에 있습니다.

NVF 옵션의 이름은 그 범위를 나타냅니다. `raster_normal_method`는 픽셀로부터 방향장을 만들고,
`shape_nvf_grid_size`는 셀의 축마다 형상 투영자 표본 수를 정하며,
`nvf_projector`는 두 NVF 경로 모두에서 투영자 구성 방식을 고릅니다. `normal_method`, `normal_grid` 또는 `projector`로
작성한 스크립트를 고칠 때는
[NVF 키워드 이전 안내](../../reference/keywords.md#nvf-names-changed-on-2026-09-21)를 참고하세요.

## How the coefficients are obtained, and how the products are closed

두 키워드는 서로 다른 단계를 기술하며, 지원되는 조합은
구조와 재료에 따라 다릅니다.

| `fourier_coefficients` | |
| --- | --- |
| `'sampled'` | 격자 위 재료 표본으로부터 얻는 이산 푸리에 계수 |
| `'integrated'` | 형상 영역에 대한 적분, 또는 스칼라 래스터와 텐서 Li 래스터의 경우 조각별 상수 픽셀에 대한 적분 |

| `factorization` | |
| --- | --- |
| `'li'` | 격자 v, 그다음 u 방향을 따라 차례로 적용하는 Li 규칙(기본 `li_nesting`, 아래 참고). 카르테시안 y와 x에 정렬되어 있어야 합니다 |
| `'nvf'` | 법선 벡터장(normal vector field). 재료 경계면을 따라 인수분해합니다. 계단 근사가 느리게 수렴하는 곡선 경계와 기울어진 경계를 위해 만들어졌습니다 |
| `'laurent'` | 모든 곱에 직접 규칙을 씁니다. 두 면내 필드 성분에는 `[[eps]]`, `E_z`에는 그 역행렬을 씁니다. 사용자가 켤 때만 쓰며 기본값이 되는 일은 없습니다. 필드 성분이 유전율과 함께 불연속이 되는 곳마다 느리게 수렴하며, fmmax의 `Formulation.FFT`처럼 이 규칙을 쓰는 솔버와 비교하기 위해 있습니다 |

**`fourier_coefficients`와 `factorization`은 모델링 경로마다 결정되며**,
`rcwa.check(...).resolved`는 요청한 값이 아니라 *실제로 적용되는* 값을 보고합니다.

| 입력 | 기본 `fourier_coefficients` | 기본 `factorization` |
| --- | --- | --- |
| 스칼라 래스터 | `'sampled'` | `'li'` |
| 텐서 래스터 | `'sampled'` | `'nvf'` |
| 스칼라 또는 텐서 재료의 형상 | `'integrated'` | `'nvf'` |

형상 적분은 등방성 재료와 이방성 재료 모두에 적용됩니다.
서로 겹치지 않는 원, 타원, 직사각형은 해석적인 영역 푸리에
계수를 씁니다. 칠하기는 겹침을 다각형 영역으로 분해하며, 곡선 겹침에는
다각형 근사를 씁니다. `Polygon`을 담은 층은 항상 이 방식으로 분해되며,
스칼라 재료만 쓸 수 있습니다. 스칼라 래스터의 적분 계수는 재구성한 매끄러운 경계가 아니라
주어진 조각별 상수 픽셀을 기술합니다.

명시적인 스칼라 또는 텐서 형상의
`factorization='li'`는 양의 축 정렬 직사각형 셀 안에 완전히 들어 있는 축 정렬 직사각형을
지원합니다. 일반 형상은 NVF를 씁니다. 그 경로가 구현하지 않은 값을 지정하면, 구현된 값으로 조용히
실행하는 대신 거부합니다(`UnsupportedFeatureError`, 코드 `rcwa.unsupported`. `rcwa.check`는 이를 이슈로 보고합니다).
`factorization='laurent'`는 스칼라 래스터(두 계열 모두), 칠한 겹침을 포함한 모든 스칼라 형상,
그리고 Li가 거부되는 기울어진 셀에서 구현되어 있습니다. 이 규칙은 축을 지정하지 않기 때문입니다.
텐서 재료는 이 규칙을 명시적으로 거부합니다. 축 정렬 직사각형 셀의 스칼라 래스터에서는 네 조합이
모두 동작합니다. 텐서 래스터는 Li에서 표본 계수와 적분 계수를 모두 지원하고, NVF는 표본 계수와 선형
투영자를 씁니다. 텐서 입력은 NVF를 기본값으로 유지합니다. 텐서의 제한 사항은
[텐서 Li 구성](../../theory/rcwa/tensor-li-factorization.md)과
[재료 지원](../../migration/materials-v1.md#anisotropic-materials)을 참고하세요.

**`raster_normal_method`는 `factorization='nvf'`에서 래스터 법선장을 고릅니다**.
그 경로에서 `nvf_projector`는 네 가지 필드 구성 방식 모두와 조합됩니다. 형상 NVF는
`nvf_projector='linear'` 또는 `'circular'`도 받습니다. 원형 형상 구성은
직사각형, 원, 타원과 이들의 다각형 분해 칠하기를 지원하며,
`Polygon`은 거부합니다.
복소 재료 입력을 가진 서로 겹치지 않는 형상에서는, 기본 조각 필드가 재료 차이의 위상을 담고
에르미트 투영자를 씁니다. 그 위상은 각 경계면에서 상쇄됩니다.
실수 재료 입력은 실수 연산을 유지하며, 이는 별도의 `'circular'` 다이애드 구성을 고르는 것이 아닙니다.
투영자 선택에는 NVF가 필요하며, `raster_normal_method`는 래스터 옵션으로 남습니다.
[형상 투영자 상세](parametric-modeling.md#complex-nvf-projection-for-rectangular-contacts)를 참고하세요.

`'distance'`는 기본값이자 넷 중 가장 부정확한 방식입니다. 60도 셀에서 `'contour'`의 0.42도에
비해 평균 법선 오차가 18.5도입니다. 래스터 `'circular'`는 결함을 고치는 것이 아니라 맞바꾸는
것이므로 기본으로 꺼져 있습니다. 중심축(medial axis)의 임의성을 없애는 대신, 측정한 모든 셀에서
정확도와 정확한 거울 대칭을 잃습니다.
[`../nvf-normal-methods.md`](../../theory/rcwa/nvf-normal-methods.md)를 참고하세요.

**`li_nesting`은 다른 규칙에 속합니다. 스칼라 Li 규칙을 만드는 곳이면 어디서든 읽히며**,
`factorization='li'`의 래스터와 그 설정의 축 정렬 직사각형이 그렇습니다. 아무것도 읽지 않는 곳에서는
거부합니다. `'nvf'`나 `'laurent'`에서의 모든 값, 그리고 Li 구성이 고정 중첩뿐인 텐서 재료에서의
`'symmetric'`이 그렇습니다. 교차 격자에는 두 축 모두를 따른 Li 규칙이 필요한데, 두 개의 1차원
규칙은 하나를 다른 하나 안에 넣는 식으로만 적용할 수 있습니다. 기본값인 `'fixed'`는 두 면내 성분
모두에 대해 `v`를 `u` 안에 넣습니다. 지금까지의 모든 `'li'` 결과는 래스터, 직사각형, 텐서를
가리지 않고 이 방식으로 계산되었습니다. `'symmetric'`은 각 성분 자신의 역규칙 축을 안쪽에 넣으므로
(Li 1997), 구성이 x와 y를 똑같이 다룹니다. 두 방식은 같은 답으로 수렴하며 유한한 `orders`에서는
서로 다릅니다. 연산자는 `Nesting` docstring에 있습니다.

고정 중첩의 비용은, 편광 분리가 없어야 할 구조에서 편광 분리가 생긴다는 점입니다.
C4 대칭 Au 나노홀 배열에서 차수 5일 때 x 편광의 투과는 0.0172이고
y 편광은 0.0728입니다. 대칭 중첩은 둘 다 0.0725를 줍니다. 전치 대칭인
유전체 L 형상에서 분리는 1.5e-3에서 4.2e-3이며, 대칭 중첩에서는 1e-13 이하입니다. 각 중첩은
RETICOLO가 같은 중첩으로 계산한 결과와 4.2e-11까지 일치합니다(`tests/references/test_reticolo_li_nesting.py`).
이전의 어떤 수치도 바뀌지 않도록 기본값은 `'fixed'`로 남습니다. 선택을 위한 측정값과
기본값을 바꾸면 무엇이 달라지는지는
[`../factorization-terms.md`](../../theory/rcwa/factorization-terms.md) §6에 있습니다.

**`'laurent'`는 정확도가 아니라 비교를 위한 것입니다.** 모든 곱을 직접 규칙으로 닫는 솔버를
재현합니다. fmmax의 표본 계수를 정확한 계수로 외삽하면, 유전체 막대와 육각 십자에서
fmmax의 `Formulation.FFT`와 1e-12까지 일치합니다(`tests/references/test_fmmax_laurent_golden.py`).
대가는 수렴입니다. eps 12.25인 1차원 TM 격자에서 오차는 차수를 두 배로 늘릴 때마다 절반으로
줄어듭니다. 차수 80에서 9.0e-4이며, 이때 Li는 4.9e-7입니다. 350 x 245 nm 막대에서는
차수 14에서도 p 편광이 4.1e-3만큼 틀리며, 이때 Li는 2.7e-5입니다. 1차원 격자의 TE에서는 두 규칙이
같은 행렬을 만들기 때문에 반올림 오차 수준에서 Li의 답과 같습니다. 이 규칙은 법선장을 만들지 않고
규칙을 중첩하지도 않으므로, 함께 쓰면 `raster_normal_method`, `nvf_projector`, `shape_nvf_grid_size`, `li_nesting`을
거부합니다. 측정값과 슬롯 선택은
[`../factorization-terms.md`](../../theory/rcwa/factorization-terms.md) §9에 있습니다.

두 연결 알고리즘은 같은 고유 모드와 경계 규약을 공유하며
~1e-14까지 일치합니다. SMM은 비용이 ~1.1-1.4x이고, 독립적으로 재사용할 수 있는 층 응답을 연쇄하며,
`rcwa.Outputs(scattering=True)`로 요청하는 `result.scattering`을 반환하는 유일한 알고리즘입니다.
[`../connecting-algorithms.md`](../../theory/rcwa/connecting-algorithms.md)를 참고하세요.

SMM은 check 단계에서 한 종류의 층을 거부합니다. 면내 유전율의 크기가 모든 곳에서 계산의 기계
엡실론의 제곱근보다 작은 층(배정밀도에서 1.5e-8, 단정밀도에서 3.5e-4)이며, 평면파의 면내 파수 벡터에
y 성분이 있거나 소스가 평면파가 아닌 경우입니다. SMM은 층의 자기 모드를 역변환하는데, 그 층에서는
p 열이 `eps` 크기이므로, s와 p가 섞이면 상대 정확도를 약 `1/|eps|`만큼 잃습니다.
`theta = 0.3`, `phi = 0.4`에서 `eps = 1e-13`인 균일 층은 `R + T - 1 = -1.1e-6`을 주었고,
`eps = 1e-16`은 `+0.09`를 주었습니다. 이슈 코드는 `rcwa.near_zero_permittivity_smm`입니다.
이런 층에는 `algorithm="etm"`을 쓰세요. 배정밀도에서 ETM은 `eps = 1e-16`까지 에너지를 4e-16 이내로
보존했고, 그 `R`은 `eps -> 0` 극한으로 선형으로 다가갑니다. 파수 벡터에 y 성분이 없으면 SMM은 정확하며
아무것도 거부하지 않습니다. 추적된 유전율은 읽을 수 없으므로 거부하지 않습니다.

모든 알고리즘은 check 단계에서 `eps_zz`가 정확히 0이거나, 역수가 계산의 부동소수점 형식을
넘칠 만큼 작은 표본을 가진 층(균일 층, 텐서의 zz 성분, 래스터 픽셀 또는 영역)을
`rcwa.zero_permittivity`로 거부합니다. RCWA는 `eps_zz`를, 그리고 역규칙에 따라 각 표본을 역변환하므로,
그런 층에는 계산할 모드가 없습니다. 이전에는 NumPy와 JAX에서 "SVD did not
converge"나 "Singular matrix"를, PyTorch에서는 날것의 선형대수 오류를 냈습니다. 작은
0이 아닌 값은 거부하지 않습니다. 배정밀도에서 `eps = 1e-18`인 균일 층은 에너지를 보존하며
`eps -> 0` 극한과 일치합니다. 단정밀도에서는 약 1e-9보다 작은 균일 층이 반올림 때문에
실패할 수 있습니다(`phi = 0.4`의 ETM은 1e-9, 1e-12, 1e-13에서 실패했고 3e-12와
1e-16에서는 풀렸습니다). 풀이 중에 실패한 인수분해는 모든 백엔드에서 `rcwa.singular`로
보고되며, 배정밀도를 쓰면 피할 수 있습니다.

## 법선장 표본 추출

`shape_nvf_grid_size`는 법선장을 읽어 내는 축당 격자 크기이며, 여기서 유일하게 미분 단계의
비용을 좌우할 수 있는 수입니다. 차수 3에서 기본값은 필드를 만드는 데 풀이 자체보다 더 많은
비용을 씁니다. 기본값은 `max(1024, 4*half_width+1)`입니다. 이 시작 분해능이 값, 미분,
맞춘 매개변수에 대해 요구한 정확도를 보장하지는 않습니다. 그런 양은 세분화하면서
확인하세요. 그 수렴은 단조롭지 않을 수 있습니다.
[자동 격자/재맞춤 작업 흐름](fitting-grid-convergence.md)은 매개변수 변화와 같은 점에서의
스펙트럼 차이를 따로 확인합니다.

이 값을 낮추는 한계는 취향이 아니라 정확도가 정합니다.
[`../nvf-upsampling.md`](../../validation/rcwa/nvf-upsampling.md)와 [`../backend-choice.md`](../../validation/rcwa/backend-choice.md)를 참고하세요.

## NVF 에너지 클로저 직접 공급하기

`rcwa.operators.ConvMatrices.from_energy(change, gram, longitudinal)`은 주어진
필드 기저와 재료 적분을, 조립된 연산자 대신 `rcwa.operators.solve`가 받는
다섯 개의 행렬로 변환합니다. 유지한 푸리에 차수가 H개일 때, 모양이 `(layers, 2*H, 2*H)`인
Q = C(B)와 G = C(B.H epsilon B), 그리고 모양이 `(layers, H, H)`인 C(epsilon_zz)를
주세요. 모든 입력은 솔버의 카르테시안 좌표계와 푸리에 차수 배치를 따라야 하며,
Q와 G에서는 x 성분 차수 전부 다음에 y 성분 차수 전부가 와야 합니다. 층 축이 없는 입력도
받지만, `operators.solve`에는 층 축이 필요합니다. 푸리에 계수는 셀 중심을 기준으로 주세요.
`from_energy()`는 입력의 원점을 그대로 유지하며 추가 평행 이동을 적용하지 않습니다.
이 다섯 슬롯 클로저는 xz, yz, zx, zy 재료 결합이 없다고 가정합니다.

빌더는 `Q^(-H) G Q^(-1)`을 선형 풀이로 계산하고, 종방향 행렬을 역변환하며,
비대각 부호를 채웁니다. 복소 재료 손실과 NumPy/JAX/PyTorch 배열의 정밀도 및 미분을
보존합니다. 모양이 맞지 않거나 즉시 실행 입력이 유한하지 않으면 거부하며, 잘못된 추적 풀이는 NaN을 냅니다.
B를 공급하는 일과 그 근사를 분해하는 일은 여전히 호출자의 책임입니다.

다음 완전한 예제는 B가 항등 행렬인 균일 유전체 층을 씁니다.

```python
import numpy as np
from wavebit import Material, Units, rcwa
from wavebit.rcwa import operators

order = 1
count = (2*order+1)**2
change = np.eye(2*count, dtype=complex)[None]
gram = 4.0*change
longitudinal = 4.0*np.eye(count, dtype=complex)[None]
conv = operators.ConvMatrices.from_energy(change, gram, longitudinal)
numerics = rcwa.Numerics(orders=(order, order))
transmitted = []
for polarization in ('p', 's'):
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(
            period=(700., 700.), top=Material(epsilon=1.), bottom=Material(epsilon=1.44),
            layers=(rcwa.Layer(thickness=177., background=Material(epsilon=4.)),)),
        excitation=rcwa.Excitation(wavelength=830., sources=(rcwa.PlaneWave(
            side='top', polarization=polarization, theta=.23, phi=.37),)))
    # The supplied closure replaces the assembled layer operators.
    assembled = operators.assemble(problem, numerics=numerics).replace(epsilon=(conv,))
    transmitted.append(operators.solve(problem, assembled).T.sum())
print(*transmitted)  # p, then s
```

[`rcwa.experimental.ConformingNVF`](conforming-nvf.md)는 이 표준 층 적층을
일반적인 직사각형 층 딕셔너리로부터 직접 구성합니다. 직교 셀의 어느 축을 따라서든
겹치지 않는 직사각형 열을 지원하며, 층마다 방향이 달라도 되고, 살아 있는 재료, 평행 이동,
두 물리적 주기, 공통 격자 회전, 그리고 JAX 재실행 중에 열리거나 닫힐 수 있는 내부 배경 구간을
지원합니다. 수동 복소 재료는 실수부가 양수이고 허수부가 음이 아니라면 모든 형상과
배경에서 서로 독립적인 손실을 가질 수 있습니다. 정수 재기저화(rebasing)는 같은 직교 점 격자를
기술하는 비직교 기저 벡터를 지원하며, 원래 유지한 푸리에 집합을 보존합니다. 일반적인
에너지 클로저 API는 외부에서 구성한 필드도 받습니다. 임의 형상 보정기는 여전히
직사각형 빌더의 범위 밖입니다.
