# 두 직사각형의 유한한 오른쪽 열림 응답

실험적인 `CartesianPairOpeningPlan`과 `CartesianPairOpeningCell`은 수평으로 놓인 높이가 같은
두 직사각형 사이에 물리적인 틈이 열릴 때의 접촉 값과 해석적인 오른쪽 미분을, NumPy 전용의
별도 경로로 제공합니다. `wavebit.rcwa._engine.conforming.cartesian_pair_opening`에서 가져오세요.
plan은 `wavebit.rcwa.experimental`에서도 내보냅니다. 이 인터페이스는
[`CartesianRectanglePairNVF`](cartesian-rectangle-pair-nvf.md)의 접촉을 보존하는 JAX 계약을
바꾸지 않습니다.

```python
import numpy as np
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import (
    CartesianPairOpeningCell,
    CartesianPairOpeningPlan,
)

geometry = dict(
    basis=np.array([[700.0, 315.0], [0.0, 490.0]]),
    widths=(140.0, 193.0),
    height=210.0,
    offset=(49.0, -21.0),
)
plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=1)
cell = CartesianPairOpeningCell(plan, order=(1, 2))
settings = dict(geometry, epsilon=(4.0 + 0.8j, 9.0 + 0.3j), background=1.0 + 0.1j)
coefficients, derivative, metadata, details = cell.coefficients(**settings)
slots, slot_derivative, closure_metadata, closure_details = cell.closure(**settings)

assert coefficients.shape == derivative.shape == (9, 9, 5)
assert all(slot.shape == (1, 15, 15) for slot in slots)
assert all(slot.shape == (1, 15, 15) for slot in slot_derivative)
```

## 기하와 plan

모든 길이는 같은 물리 단위를 써야 하며, 예제는 나노미터 단위로 읽을 수 있습니다. `basis`는 열이
격자 벡터인 유한한 실수 2×2 행렬(2-by-2)이며, 행렬식은 유한한 양수여야 합니다. 양수인 두 `widths`와
`epsilon` 값은 왼쪽/오른쪽 쌍입니다. 양수 `height` 하나를 함께 씁니다. `offset`은
`basis @ (.5, .5)`로부터의 카르테시안 변위이며, 적분 셀 전체를 평행 이동합니다. 직사각형은 물리적인
카르테시안 모서리를 그대로 유지합니다.

응답은 정확한 접촉에서 평가합니다. 틈 g는 입력이 아니라 미분 변수입니다. 주어진 모든 설정을 고정한 채
왼쪽 직사각형은 `(-g/2, 0)`만큼, 오른쪽 직사각형은 `(g/2, 0)`만큼 움직입니다. 오른쪽 미분은
**호출자 길이 단위당** 값으로 반환됩니다. 이는 일반적인 제약 없는 기하 JVP가 아니며, `xp`나 JAX
모드는 없습니다.

`from_geometry`는 양의 순환 메시 차트 네 개 중 하나를 고릅니다. `shift=0`, `1`, `2` 또는 `3`은
구체적인 차트를 명시적으로 요청합니다. `refinement`의 기본값은 0이며 0부터 3까지의 정수를 받습니다.
넓어진 유한 범위와 광학적 한계는 [세분화 제어](../../validation/rcwa/nvf-pair-refinement.md)를
참고하세요. 고정된 plan은 shift와 refinement만 저장합니다. 재실행(replay)은 현재 물리 설정을 사용하며,
엄격한 포함, 양의 정칙 삼각형과 양의 열림 면적 기울기를 요구합니다. 다른 차트가 필요하면 명시적으로
다시 계획해야 합니다.

## 계수와 닫힘

`order`는 0을 포함해 음이 아닌 정수 또는 `(x, y)` 쌍을 받습니다. 아홉 개의 계수 필드는
`Bxx, Bxy, Byx, Byy, Gxx, Gxy, Gyy,
epsilon, Gyx` 순서입니다. 배열 모양은 `(9, 4*y+1, 4*x+1)`이며, 축은 성분, y 레이블, x 레이블
순서입니다. 레이블은 `-2*y`부터 `2*y`까지, `-2*x`부터 `2*x`까지입니다. 푸리에 위상은 물리적 면적
정규화를 적용한 `exp(-2*pi*1j * [m,n] @ inv(basis) @ position)`입니다.

두 메서드 모두 `(value, derivative, metadata, details)`를 반환합니다. 닫힘(closure)의 값과 미분은 각각
일반적인 다섯 슬롯 순서의 `ConvMatrices` 튜플이며, 층 축이 하나 있습니다. 닫힘은 역행렬을 구하기
전에 요청된 필드 차이를 선택하고, 항상 현재 설정에서 계수를 계산합니다. 캐시된 계수로 덮어쓰는 것은
받지 않습니다. 메타데이터에는 원래의 약형(weak) 잔차, 강압성(coercivity)과 조건수 정보, 개수와 미분
단위가 들어 있습니다. 닫힘은 저장 에너지, 부호 있는 소산과 역행렬 잔차 검사를 추가합니다.
details에는 진단 필드, 변환과 원래의 Laurent 연산자가 들어 있으며, 이것들이 연속 극한이나 광학 오차의
상한을 뜻하지는 않습니다.

`epsilon`과 `background`는 실수부가 엄격히 양수이고 허수부가 음이 아닌 유한한 수치 스칼라 유전율이어야
합니다. 기하는 실수 수치 데이터입니다. 복소 기하는 허수부가 0이어도 거부합니다. 유효하지 않은
기하/재료, 지원하지 않는 차수와 세분화, 추가 키워드는 오류를 냅니다. 정확한 접촉, 공통 높이 하나와
수평 직사각형은 이 인터페이스에 내재된 조건입니다. 각도, 수직 이동, 겹침, 여러 쌍과 텐서는 지원하지
않습니다.

독립적인 유한 공간 적격성 검사와 약한 극한의 유도는
[방법 노트](../../validation/rcwa/nvf-cartesian-pair-opening.md)에 있습니다. 유한 계수도, 그 오른쪽
미분도 패턴 층의 광학적 정확도를 보증하지 않습니다.

선택적인 생성자 인수 `coercivity_angle`은 공통의 복소 반평면에 있는 수동 금속도 허용합니다. 이
증명서(certificate)는 허용 여부와 진단에 영향을 주며, 물리 방정식은 원래 재료를 그대로 유지합니다.
실행 가능한 예제, 독립적인 금속 틈 검사와 아직 해결되지 않은 광학적 한계는
[재료 섹터](../../validation/rcwa/nvf-pair-material-sector.md)를 참고하세요.
