# 02. 경계조건부터: E 전체가 연속인 것은 아니다

:::{note}
이 교재의 실행 예제는 원래 수치 기록을 재현할 수 있도록 `wavebit.compat.legacy` 인터페이스를 명시적으로 사용합니다. 새 작업은 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 참고하세요. 기존 측정값은 이번 문서 이행에서 다시 계산하지 않았습니다.
:::

스칼라 상대유전율 ε인 재료에서 `D = ε₀ ε E`입니다. 계산을 간단히 쓰기 위해 아래에서는 `d = D/ε₀`로 정규화합니다. n은 재료 경계에 수직인 단위 벡터이며, t는 xy 평면에서 경계에 접하는 방향입니다. 이 장의 그림은 횡방향 전기장 E_xy를 두 방향으로 나눈 것입니다. z 성분도 이런 층 내부의 수직 경계에 접하며, 5장에서 별도 슬롯으로 다룹니다.

일반적인 재료 경계에서 접선 전기장 `Eₜ`는 연속입니다. 자유 표면 전하가 없다면 법선 전속밀도 `Dₙ`도 연속입니다. 하지만 법선 전기장 `Eₙ`은 재료의 ε가 다르면 보통 불연속입니다. 이 교재는 이런 통상적인 경계조건을 사용합니다.

```text
E_xy = Eₙ n + Eₜ t
Eₜ₁ = Eₜ₂
dₙ₁ = dₙ₂  →  ε₁Eₙ₁ = ε₂Eₙ₂
```

```{figure} ../../figures/learn/nvf/02-boundary.svg
:alt: 접선 전기장은 같고, 법선 전기장은 유전율에 따라 달라진다
:figclass: learning-figure

그림 2-1. 접선 전기장은 같고, 법선 전기장은 유전율에 따라 달라진다
```

`ε₁=1`, `ε₂=4`이고 `dₙ=1`, `Eₜ=1`인 국소 예를 생각해 봅시다.

| 성분 | 재료 1 | 재료 2 | 경계에서의 성질 |
| --- | --- | --- | --- |
| 접선 Eₜ | 1 | 1 | 연속 |
| 법선 dₙ | 1 | 1 | 연속 |
| 법선 Eₙ = dₙ/ε | 1 | 0.25 | 불연속 |
| 접선 dₜ = εEₜ | 1 | 4 | 불연속 |

그래서 모든 전기장 성분에 똑같은 곱 규칙을 쓰기보다, **법선과 접선이 서로 다른 연속성을 가진다는 사실을 행렬 구성에 반영**하는 것이 핵심입니다. 이 표는 한 경계에서 성립하는 관계를 보여 주며, 실제 구조 전체의 장을 구한 결과는 아닙니다.

```{raw} html
<section class="exercise" aria-labelledby="boundary-title"><h3 id="boundary-title">직접 바꿔 보기 · ε를 바꾸면 어느 성분이 달라질까?</h3><p class="explain">ε₁=1, Eₜ=1, Dₙ/ε₀=1로 고정한 국소 경계조건 예입니다. 단위는 정규화했으며 RCWA를 실행하지 않습니다.</p><div class="controls"><label for="eps2">오른쪽 상대유전율 ε₂ <output id="eps2-value" for="eps2">4.0</output><input id="eps2" type="range" min="1" max="12" step="0.1" value="4"></label></div><p class="explain">아래 막대 길이 = 오른쪽 Eₙ / 왼쪽 Eₙ</p><div class="bar-shell"><div class="bar" id="normal-bar"></div></div><div id="boundary-result" class="readout" aria-live="polite">ε₂=4일 때 Eₙ₂=0.25, Eₜ₂=1, Dₙ₂/ε₀=1입니다.</div><noscript><p>2장의 표로 같은 계산을 확인할 수 있습니다.</p></noscript></section>
```
