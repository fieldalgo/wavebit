# 유한 쌍 틈 미분하기

실험적인 `CartesianPairGapEvaluator`는 NumPy 또는 JAX x64로 정확한 유한 쌍 계열을 평가합니다.
이 평가기는 값만 계산하며, 정확한 접촉에서의 물리적 오른쪽 미분을 포함한 틈 미분은 일반적인 JAX
연산이 제공합니다. 평가기는 하위 모듈에서 가져오고, 추적하기 전에 구체적인
`CartesianPairOpeningPlan`을 주세요.

```python
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from wavebit.rcwa._engine.conforming.cartesian_pair_gap_autodiff import CartesianPairGapEvaluator
from wavebit.rcwa.experimental import CartesianPairOpeningPlan

geometry = dict(
    basis=np.array([[700.0, 315.0], [0.0, 490.0]]),
    widths=(140.0, 193.0),
    height=210.0,
    offset=(49.0, -21.0),
)
plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=1)
cell = CartesianPairGapEvaluator(plan, order=(1, 2))
settings = dict(geometry, epsilon=(4.0 + 0.8j, 9.0 + 0.3j), background=1.0 + 0.1j)

@jax.jit
def response(gap):
    coefficients, slots, _, _ = cell.evaluate(gap=gap, **settings, xp=jnp)
    return coefficients, jnp.stack(slots)

length = np.sqrt(np.linalg.det(geometry["basis"]))
for gap in (0.0, length / 64):
    values, derivatives = jax.jvp(
        response, (jnp.asarray(gap),), (jnp.asarray(1.0),)
    )
    assert values[0].shape == derivatives[0].shape == (9, 9, 5)
    assert values[1].shape == derivatives[1].shape == (5, 1, 15, 15)
    assert all(np.isfinite(np.asarray(value)).all() for value in (*values, *derivatives))
```

JAX는 선택 사항입니다. `cell.evaluate(gap=gap, **settings)`는 NumPy를 사용하며 JAX를 가져오지 않고
값을 반환합니다. 별도의 [`CartesianPairGapCell`](cartesian-pair-gap.md)은 해석적 미분도 함께 반환하는
NumPy 인터페이스로 계속 남아 있습니다.

## 기하와 미분의 의미

모든 길이는 호출자의 단위 하나를 사용합니다. `basis`는 열이 격자 벡터인 유한한 실수 2×2(2-by-2)
행렬이며, 행렬식은 유한한 양수여야 합니다. `widths`는 양의 왼쪽/오른쪽 폭을 줍니다. 수평 직사각형들은
양수 `height` 하나를 함께 씁니다. 사각이 아닌 격자 위에서도 물리적 카르테시안 모서리가 보존됩니다.

`center = basis @ (.5, .5) + offset`이라 하면 두 직사각형의 중심은
`center + (-(right_width + gap)/2, 0)`와
`center + ((left_width + gap)/2, 0)`입니다. 물리적 틈을 늘리면 바깥 셀은 고정한 채 각 직사각형이
그 변화량의 절반씩 바깥쪽으로 움직입니다. `offset`은 적분 셀 전체를 평행 이동합니다. `epsilon`은
대응하는 왼쪽/오른쪽 재료를 주며, 기본적으로 이 값과 `background`는 유한한 양의 실수부와 음이 아닌
허수부를 가져야 합니다. 생성자에 명시적인 `coercivity_angle`을 주면 공통의 증명된 반평면에 있는 수동
금속도 허용합니다. [재료 섹터](../../validation/rcwa/nvf-pair-material-sector.md)를 참고하세요. 이 각도는
추적 전에 정해지는 정적인 값이며, 물리 방정식이나 재료 값을 바꾸지 않습니다.

평가기는 유한한 실수 `gap >= 0`을 받습니다. 0에서 양의 틈 접선(tangent)은 물리적 오른쪽 미분입니다.
0에서 음의 접선은 형식적인 국소 대수적 확장일 뿐이며, 음의 틈 평가는 거부합니다.
단위 틈 접선은 **호출자 길이 단위당** 미분을 반환합니다. 내부에서 `sqrt(det(basis))`로 정규화해도 그
의미는 바뀌지 않습니다.

이번 단계에서 적격성을 확인한 것은 틈 미분뿐입니다. 기하, 자세(pose)와 재료는 실제 인수로 남아 고정된
차트에서 다시 실행할 수 있지만, 그 임의의 그래디언트, 고차 미분과 혼합 미분은 여기서 적격성을 확인하지
않았습니다. 일반 [`CartesianRectanglePairNVF`](cartesian-rectangle-pair-nvf.md)는 접촉 차트에서 여전히
접촉을 보존하는 접선을 요구합니다.

## 고정된 차트와 반환값

`CartesianPairOpeningPlan.from_geometry`는 구체적인 접촉 기하를 계획하며, `refinement`는 0부터 3까지,
선택적인 순환 `shift`는 0부터 3까지 받습니다.
[세분화 노트](../../validation/rcwa/nvf-pair-refinement.md)는 더 큰 유한 공간, JAX 틈 대조 검사, 줄어든
조립 저장 공간과 아직 해결되지 않은 광학적 한계를 다룹니다.
평가기는 그 불변 위상과, 정수 또는 `(x, y)` 튜플로 주는 음이 아닌 푸리에 반폭을 저장합니다. 푸리에
차수에는 폭 0도 허용됩니다. 추적 중의 동적 계획, 물리 계수 캐시, 자동 차트 변경은 없습니다.

매번 다시 실행할 때 엄격한 범위 포함, 양의 정칙 삼각형 면적과 양의 붕괴 삼각형 열림 기울기를
검사합니다. 셀 안에 들어 있는 직사각형 쌍이라도 차트의 삼각형 가드를 통과하지 못할 수 있습니다.
지원하지 않는 기하에는 서로 다른 높이, 수직 이동, 회전, 겹침, 추가 쌍과 텐서가 있습니다.
불리언이나 복소 기하는 복소 허수부가 0이어도 거부합니다. 지원하지 않는 네임스페이스, x64가 꺼진 JAX,
호환되지 않는 모양과 추가 키워드는 거부합니다. 유효하지 않은 수치 값은 즉시 실행에서 오류를 냅니다.
JIT 아래에서는 안전한 중간 계산을 거친 뒤 계수와 닫힘(closure) 출력이 모두 NaN이 됩니다. 유효 영역 밖에서의
미분은 유효성을 보장하지 않습니다.

`evaluate`는 `(coefficients, slots, metadata, details)`를 반환합니다. 계수의
모양은 `(9, 4*y+1, 4*x+1)`이고 필드 순서는 `Bxx, Bxy, Byx, Byy, Gxx, Gxy,
Gyy, epsilon, Gyx`입니다. 위상은 물리적 면적 정규화를 적용한
`exp(-2*pi*1j * [m,n] @ inv(basis) @ position)`입니다.
다섯 개의 `ConvMatrices` 슬롯은 각각 모양이 `(1, H, H)`이며, 여기서
`H=(2*x+1)*(2*y+1)`입니다. 역행렬을 구하기 전에 요청한 차이를 정확히 선택하며, 서로 다른 차수와 0차도
포함됩니다.

메타데이터에는 물리적/정규화된 틈, 정규화 길이, 공간 개수, 사영된 약형 잔차와 원래의 약형 잔차,
대각 블록 에너지 검사, 조건수, 그리고 닫힘의 저장 에너지, 부호 있는 소산과 역행렬 검사가 들어
있습니다. details에는 정규화된 행렬, 하중(load), 해, 꼭짓점, 필드, 푸리에 가중치와 원래의 약형
플럭스/하중이 들어 있습니다. 이 진단은 값일 뿐, 추가적인 해석적 미분 API가 아닙니다. 예제처럼 미분에
필요한 출력을 반환하는 함수를 사용하세요.

[방법과 적격성 노트](../../validation/rcwa/nvf-cartesian-pair-gap-autodiff.md)에 독립적인 기준, 자원
관찰과 한계가 기록되어 있습니다. 이 유한 구성(constitutive) 평가는 연속체나 패턴 층의 광학적 정확도를
보증하지 않습니다.
