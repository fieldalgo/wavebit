(complex-kx-lossy-gain)=
# 부록 B. 흡수·이득 매질로 넓히기: 흡수 입사 매질, 떠 있는 branch point, 이득 매질

:::{note}
이 교재의 실행 예제는 원래 수치 기록을 재현할 수 있도록 `wavebit.compat.legacy` 인터페이스를 명시적으로 사용합니다. 새 작업은 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 참고하세요. 기존 측정값은 이번 문서 이행에서 다시 계산하지 않았습니다.
:::

{ref}`부록 A <complex-kx-roots>`는 무손실 매질 하나만 놓고, 반공간의 {math}`k_z` 두 근 가운데 무엇을 고르는지를 끝까지 설명했습니다. 이 부록은 부록 A의 {ref}`전제 2 <ckr-premises>`(무손실)를 풀고, 입사 매질이 빛을 흡수할 때, 근을 고르는 반공간이 흡수할 때, 반공간이 빛을 키울 때(이득) 무엇이 달라지는지 봅니다. 두 근, branch point, branch cut, 경로 D·E, 네 규칙(경로 D, 경로 E, Maystre, {math}`\operatorname{Im}k_z>0`)은 부록 A에서 설명한 뜻 그대로 씁니다.

흡수 입사 매질에서 각도를 넣을 때({ref}`8장의 예 <absorbing-incidence-example>`), 흡수 매질을 반공간에 두고 복소 `theta`를 쓸 때, 이득 매질을 반공간에 둘 때, 흡수 매질에서 다른 솔버와 결과를 맞춰 볼 때 필요한 내용입니다.

2026년 10월 2일 기준 Wavebit의 동작을 설명합니다. Wavebit은 이날 반공간의 규칙을 경로 D에서 Maystre 규칙({math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0`)으로 바꿨습니다({ref}`부록 A <complex-kx-roots>`). 수동 매질의 실수 {math}`k_x`에서는 두 규칙이 비트 단위로 같고, 이 부록에서 답이 달라지는 곳은 복소 {math}`k_x`와 이득 반공간의 전반사 쪽입니다. 경로 D는 "2026년 10월 2일 전의 Wavebit" 규칙으로 남겨 둡니다. 바꾼 근거와 측정값은 [분기 처리](../../theory/rcwa/branch-cuts.md)의 8절에 있습니다.

그림은 `tools/docs/learning/draw_complex_kx_roots.py`로 그렸고, Wavebit으로 푼 수치는 `docs/validation/learning-records/complex-kx-roots.json`에 기록한 값입니다. 기록에는 계산에 쓴 Wavebit 커밋이 함께 남아 있습니다. 경로 D로 푼 Wavebit 수치는 규칙을 바꾸기 전의 커밋에서 잰 값입니다.

(ckr-b-relaxed)=
## 부록 A의 어느 전제를 푸나

부록 A는 전제 넷을 깔았습니다. (1) 시간 규약 {math}`e^{-i\omega t}`, (2) 무손실 유전체, (3) 비자성·등방성, (4) 근은 반공간에서만 고른다는 것입니다. 이 부록은 **전제 2만 풉니다.** 나머지 셋은 그대로입니다. 그래서 절마다 "전제 2가 어디서 어떻게 풀리고, 그래서 부록 A의 무엇이 바뀌나"를 씁니다.

- {ref}`흡수 입사 매질 <ckr-absorbing-incidence>`: 입사 매질이 흡수합니다. 각도 `theta`를 {math}`k_x`로 바꾸는 규칙이 둘로 갈립니다. 부록 A의 "복소 `theta`는 실수 `theta`를 살짝 비튼 입력"이 깨집니다.
- {ref}`흡수 반공간 <ckr-lossy>`: 근을 고르는 반공간이 흡수합니다. 실수 {math}`k_x`에서 근이 1사분면 안쪽으로 들어가 비김이 없어지고, branch point가 실수축에서 뜹니다. 부록 A에서 실수축 위에 있던 branch cut이 실수축을 떠납니다.
- {ref}`이득 매질 <ckr-gain>`: 반공간이 빛을 키웁니다. 수동(빛을 키우지 않음, {ref}`아래 <ckr-b-passive>`)이라는 더 넓은 전제까지 깨져서, 실수 {math}`k_x`에서도 두 기준이 서로 다른 근을 가리키고 규칙이 필요해집니다.
- {ref}`코드 차이 <ckr-b-codes>`: 흡수나 이득이 있어야 드러나는 솔버 사이의 차이입니다.

전제 2에는 사실 조건이 둘 들어 있습니다. {math}`\operatorname{Im}\varepsilon=0`(무손실)과 {math}`\varepsilon>0`(유전체)입니다. 이 부록은 앞의 조건을 풀어 {math}`\operatorname{Im}\varepsilon\ne0`인 매질을 봅니다. 그때 이 부록의 식과 설명은 {math}`\operatorname{Re}\varepsilon`의 부호와 상관없이 성립하므로, {math}`\operatorname{Re}\varepsilon<0`인 금속에도 그대로 씁니다. 다만 수치 예와 무작위 표본은, 감사 사례의 금 기판을 빼면 모두 {math}`\operatorname{Re}\varepsilon>0`입니다. 두 부록 모두 다루지 않는 것은 {math}`\operatorname{Im}\varepsilon=0`이면서 {math}`\varepsilon<0`인 매질, 곧 손실을 뺀 이상화한 금속 하나입니다. 그때는 {math}`n=\sqrt\varepsilon`이 순허수라 branch point가 실수축이 아니라 허수축 위에 놓여서, 부록 A의 그림이 그대로 성립하지 않습니다.

(ckr-b-passive)=
### 수동과 능동: 무손실, 흡수, 이득

**수동(passive)** 매질은 스스로 에너지를 만들지 않으므로 빛을 흡수할 수는 있어도 키울 수는 없습니다. 흡수되는 전력 밀도 {math}`\tfrac12\omega\varepsilon_0\operatorname{Im}\varepsilon\,|\mathbf E|^2`가 0 이상이어야 하므로 {math}`\operatorname{Im}\varepsilon\ge0`입니다({ref}`13장 <interference-absorption-origin>`). 곧 수동은 빛의 에너지를 **줄이거나 그대로 두는** 매질을 모두 가리킵니다.

| 매질 | {math}`\operatorname{Im}\varepsilon` ({math}`e^{-i\omega t}` 규약) | 빛의 에너지 | 분류 |
| --- | --- | --- | --- |
| 무손실 (공기, 이상적인 유리) | {math}`=0` | 그대로 | 수동 |
| 흡수 (금속, 흡수체) | {math}`>0` | 줄어듦 | 수동 |
| 이득 (레이저 매질) | {math}`<0` | 커짐 | 능동 (active) |

부록 A의 무손실은 등호({math}`\operatorname{Im}\varepsilon=0`) 자리, 곧 흡수와 이득 사이의 경계에 있습니다. 엄밀히 말하면 수동성은 모든 주파수에 걸친 성질이고, 진공이 아닌 한 모든 주파수에서 흡수가 전혀 없는 물질은 없습니다. 분산이 있으면 Kramers–Kronig 관계 때문에 어딘가의 주파수에서 흡수가 생기므로, 무손실은 이상화입니다. 다만 RCWA는 한 주파수씩 풀기 때문에, 그 주파수에서 {math}`\operatorname{Im}\varepsilon\ge0`이면 수동으로 봅니다.

RCWA가 다루는 유전체, 금속, 흡수체 대부분이 수동입니다. 흡수 입사 매질과 흡수 반공간은 수동 안에서 넓히는 것이고, **이득 매질만 수동 전제까지 풉니다.**

**자성 매질은 여전히 뺍니다({ref}`부록 A의 전제 3 <ckr-premises>`).** 자성이 있으면 {math}`k_z^2=\varepsilon\mu-k_x^2`가 되는데, 수동이라도 {math}`\operatorname{Im}(\varepsilon\mu)`가 음수일 수 있습니다. 음의 굴절률 매질이 그렇습니다. 예를 들어 {math}`\varepsilon=\mu=-1+0.1i`이면 {math}`\varepsilon\mu=0.99-0.2i`입니다. 이런 반공간에서는 Maystre 규칙이 커지는 근을 고르므로, Wavebit은 2026-10-05부터 이중 음 매질(Re ε<0, Re μ<0)의 반공간에 그 거울(Im kz − Re kz > 0)을 씁니다([음굴절률 매질](../../theory/rcwa/negative-index-media.md)). 이 부록은 그 경우를 다루지 않습니다.

(ckr-b-roles)=
### 매질의 두 역할

**어느 매질이 흡수하느냐에 따라 걸리는 질문이 다릅니다.** 매질은 두 가지 역할을 하고, 무손실과 흡수는 역할마다 따로 봐야 합니다.

| 어느 매질 | 걸리는 질문 | 무손실이면 (부록 A) | 흡수 매질이면 (이 부록) |
| --- | --- | --- | --- |
| 입사 매질 | 각도 `theta`를 {math}`k_x`로 어떻게 바꾸나 | 실수 `theta`와 복소 `theta`가 이어집니다. 허수부가 작은 복소 `theta`는 실수 `theta`를 살짝 비튼 것입니다 | 실수 `theta`(위상각 규약: 위상이 나아가는 방향이 `theta`)와 복소 `theta`(형식적인 {math}`n\sin\theta`)가 다른 {math}`k_x`를 줍니다. 복소 `theta`는 따로 떨어진 형식적인 계산입니다({ref}`아래 <ckr-absorbing-incidence>`) |
| 근을 고르는 반공간(위: 반사 차수, 아래: 투과 차수) | 두 근 중 어느 것이냐 | branch point가 실수축 위에 있어, 규칙마다 실수 {math}`k_x` 근처에서 답이 갈립니다 | branch point가 실수축에서 떠 있어, 실수 {math}`k_x` 근처에서는 네 규칙이 같습니다({ref}`아래 <ckr-lossy>`) |

위쪽 매질은 입사 매질이면서 반사 차수의 반공간이라 두 질문에 모두 걸립니다. 아래쪽 매질은 투과 차수의 근 고르기에만 걸립니다. 이득 매질은 이 표 밖이며, {ref}`따로 <ckr-gain>` 봅니다.

(ckr-absorbing-incidence)=
## 흡수 입사 매질: 각도를 kx로 바꾸는 두 규칙

**푸는 전제.** 부록 A의 전제 2를 **입사 매질**에서 풉니다. 위쪽 매질이 {math}`\operatorname{Im}\varepsilon>0`입니다. 그 매질이 반사 차수의 반공간으로서 근을 고를 때 생기는 일은 {ref}`다음 절 <ckr-lossy>`에서 봅니다.

부록 A에서 복소 {math}`k_x`는 Wavebit에서 복소 `theta`로만 온다고 했습니다. 입사 매질이 무손실이면 허수부가 작은 복소 `theta`는 실수 `theta`를 살짝 비튼 입력이라, 연속성 잣대를 `theta`로 말해도 같습니다. 입사 매질이 흡수하면 이것이 깨집니다. Wavebit은 각도를 {math}`k_x`로 바꾸는 규칙을 두 가지 쓰고, 흡수 매질에서는 두 규칙이 서로 다른 {math}`k_x`를 줍니다.

- **실수 `theta`: 위상각 규약.** 흡수 매질 안의 평면파는 파수 벡터가 복소수라, 위상이 나아가는 방향({math}`\operatorname{Re}\mathbf k`)과 세기가 줄어드는 방향({math}`\operatorname{Im}\mathbf k`)이 다릅니다. 그래서 "각도"가 저절로 하나로 정해지지 않습니다. 위상각 규약은 위상이 나아가는 방향이 법선과 정확히 `theta`를 이루는 실수 {math}`k_x`를 찾습니다. 곧 {math}`\tan\theta=k_x/\operatorname{Re}k_z`를 만족하는 실수 {math}`k_x`를 분산 관계와 함께 풉니다({ref}`8장 <absorbing-incidence-example>`). {math}`k_x`가 실수라 표면을 따라 세기가 고른 조명입니다. 흡수 입사 매질이면 `theta`의 크기가 π/2보다 작아야 합니다. `theta`가 90°에 가까워지면 이 {math}`k_x`가 끝없이 커져서(아래 그림 B-1), 스치는 각도에서는 표면 세기가 고른 유한한 {math}`k_x`가 없습니다. 그래서 Wavebit이 그 입력을 거부합니다({ref}`8장 <absorbing-incidence>`).
- **복소 `theta`: 형식적인 {math}`n\sin\theta`.** 부록 A에서 본 {ref}`형식적인 식 <ckr-complex-kx>` {math}`k_x=n\sin\theta\cos\phi`를 그대로 쓰되, 입사 매질의 굴절률 {math}`n`을 복소수 그대로 곱합니다. 이때 각도는 방향이 아니라 계산용 매개변수입니다.

8장의 예와 같은 {math}`\varepsilon=4+3i`({math}`n=2.121+0.707i`)에서 Wavebit에 물어보면 이렇습니다.

| `theta` | 읽는 규칙 | {math}`k_x/k_0` |
| --- | --- | --- |
| `0.4` | 위상각 규약 | 0.833318 |
| `0.4 + 1e-12j` | 형식적인 {math}`n\sin\theta` | 0.826081 + 0.275360i |
| `0.4 - 1e-12j` | 형식적인 {math}`n\sin\theta` | 0.826081 + 0.275360i |

첫 줄이 위상각 규약인지는 직접 확인할 수 있습니다. 이 {math}`k_x`에서 {math}`k_z=1.971+0.761i`이고, 위상 방향은 {math}`\arctan(0.8333/1.971)=22.9^\circ`, 곧 0.4 rad입니다.

그런데 `theta`에 허수부를 {math}`10^{-12}`만 줘도 {math}`k_x`의 허수부가 0에서 0.275로 뜁니다. 복소 `theta`는 형식적인 식으로 읽는데, `theta`가 거의 실수라 {math}`\sin\theta\approx\sin0.4=0.3894`이므로 다음과 같기 때문입니다.

```{math}
k_x\approx n\times0.3894=(2.1213+0.7071i)\times0.3894\approx0.826+0.275i .
```

곧 이 0.275는 `theta`의 허수부가 아니라 {math}`n`의 허수부에서 옵니다({math}`0.7071\times0.3894\approx0.275`). `theta`의 허수부가 +이든 −이든 값이 같으므로, 끊기는 곳은 `theta`가 정확히 실수인 한 점뿐입니다. 곧 흡수 매질에서 복소 `theta`는 실수 `theta`를 살짝 비튼 입력이 아니라 다른 입력입니다([좌표·부호 규약](../../reference/common/conventions.md#real-angle-incidence)).

```{figure} ../../figures/learn/learn/b01-theta-to-kx.svg
:alt: kx 평면 두 장이다. (a) 흡수 입사 매질 ε=4+3i에서 청록 실선은 실수 θ를 0에서 키울 때 위상각 규약이 주는 kx로, 실수축 위에 있고 θ가 90도에 가까워지면 끝없이 커진다고 적혀 있으며 θ=0.2, 0.4, …, 1.2의 눈금이 있다. 주황 파선은 실수 θ에 형식적인 n·sinθ를 적용한 값으로, 원점에서 branch point n=2.121+0.707i(θ=90도)까지 가는 선분이고, 그 위의 작은 주황 점이 θ=0.2, 0.4, …, 1.2이며 θ=0.8과 1.2에 이름이 붙어 있다. 실수 θ=0.4의 청록 점 0.833318과, θ=0.4±10⁻¹²i라고 적힌 주황 점 0.826081+0.275360i가 화살표로 이어져 있고, θ에 허수부 10⁻¹²만 줘도 뛴다고 적혀 있다. (b) 무손실 ε=4(n=2)에서는 청록 실선과 주황 파선이 실수축 위의 0에서 branch point 2까지 겹치고, θ=0.4에서 0.778837, θ=0.4+10⁻¹²i에서 0.778837+1.8×10⁻¹²i라고 적혀 있다.
:figclass: learning-figure

그림 B-1. 각도 {math}`\theta`를 {math}`k_x`로 바꾸는 두 규칙입니다. 흡수 입사 매질에서는 실수 {math}`\theta`의 위상각 규약(실수축 위)과 형식적인 {math}`n\sin\theta`(원점에서 {math}`n`까지 가는 선분 위)가 갈리고, 무손실이면 두 규칙이 겹칩니다. `tools/docs/learning/draw_complex_kx_roots.py`로 계산해 그렸습니다.
```

그림 B-1은 이것을 {math}`k_x` 평면에 그린 것입니다.

- **(a) 흡수 입사 매질 {math}`\varepsilon=4+3i`.** 청록 실선은 실수 `theta`를 0에서 키울 때 위상각 규약이 주는 {math}`k_x`로, 실수축 위에 있고 `theta`가 90°에 가까워지면 끝없이 커집니다. 주황 파선은 실수 `theta`에 형식적인 {math}`n\sin\theta`를 적용한 값으로, 원점에서 branch point {math}`n`까지 가는 선분이며 90°에서 {math}`n`에 닿습니다. `theta`에 허수부를 아주 조금 주면 {math}`k_x`는 청록 실선에서 주황 파선 위의 점으로 옮겨 갑니다. 그래서 `theta = 0.4`에 허수부 {math}`10^{-12}`만 줘도 청록 점에서 주황 점으로 뜁니다.
- **(b) 무손실 {math}`\varepsilon=4`({math}`n=2`).** {math}`n`이 실수라 위상각 규약이 곧 {math}`n\sin\theta`입니다. 두 선이 실수축 위에서 겹치고, `theta = 0.4`에서 {math}`k_x/k_0=0.778837`, `theta = 0.4 + 1e-12j`에서 {math}`0.778837+1.8\times10^{-12}i`로 {math}`10^{-12}` 규모만 움직입니다.

그래서 부록 A의 연속성 잣대는 흡수 입사 매질에서도 {math}`k_x`에 대해서는 그대로이지만, `theta`로 옮겨 말할 수는 없습니다. `theta`의 허수부를 조금 주는 것이 {math}`k_x`를 살짝 비트는 것이 되는 때는 입사 매질이 무손실일 때(예: 공기)뿐이기 때문입니다. 흡수 입사 매질에서 실수 {math}`k_x`를 살짝 비틀려면, 원하는 복소 {math}`k_x`에서 거꾸로 {math}`\theta=\arcsin(k_x/n)`을 구해 복소 `theta`로 넣습니다({ref}`8장 <complex-transverse-wavevector>`).

그리는 스크립트는 그리기 전에 세 가지를 확인합니다. 위상각 규약의 {math}`k_x`로 잰 위상 방향이 `theta`와 같은지, `theta`에 {math}`\pm10^{-12}i`를 준 형식적인 값이 실수 `theta`의 {math}`n\sin\theta`와 {math}`10^{-12}` 규모로만 다른지, 무손실이면 두 규칙이 같은지입니다.

**왜 규칙이 둘인가.** 예전에는 실수 `theta`에도 {math}`n\sin\theta`를 썼습니다. 그러면 흡수 매질에서는 실수 각도도 복소 {math}`k_x`가 되었습니다. [시간 규약 감사](../../validation/rcwa/time-convention-audit.md)가 찾은 사례가 거기서 나왔습니다. 금 위의 손실 유전체에서 각도를 {math}`6.7\times10^{-5}` rad 움직였더니 반사 진폭이 2.96 튀었습니다. 그 사례가 어느 branch cut에 걸렸는지는 {ref}`다음 절 <ckr-lossy>`에서 봅니다. 지금은 실수 `theta`를 물리적인 방향으로만 읽고, 형식적인 계산은 복소 `theta`로만 합니다.

**소멸파 입사.** 입사파 자체가 입사 매질 안에서 소멸파인 경우입니다. 부록 A에서 본 대로 {math}`\theta=\pi/2+i\alpha` 꼴은 {math}`\sin\theta=\cosh\alpha`가 실수입니다. 입사 매질이 무손실이면 {math}`k_x=n\cosh\alpha`도 실수로 남습니다. 입사 매질이 흡수 매질이면 {math}`n`이 복소수라 {math}`k_x`도 복소수가 되고, `ComplexIncidenceWarning`이 나옵니다.

(ckr-lossy)=
## 흡수 반공간: branch point가 뜹니다

**푸는 전제.** 부록 A의 전제 2를 **근을 고르는 반공간**에서 풉니다. 그 반공간이 {math}`\operatorname{Im}\varepsilon>0`이며, 흡수 매질이므로 여전히 수동입니다. 달라지는 것은 두 가지입니다. 실수 {math}`k_x`의 근이 1사분면 안쪽으로 들어가고, branch point가 실수축에서 뜹니다.

부호를 읽을 때 주의할 점이 하나 있습니다. 이 부록도 {ref}`부록 A <ckr-premises>`처럼 반공간의 장을 구조에서 떨어진 거리 {math}`d`로 {math}`e^{ik_zd}`라 씁니다. 8장은 위쪽이 +인 전역 {math}`z`로 써서, 아래쪽 흡수 매질로 투과해 줄어드는 파의 {math}`k_z` 허수부가 음수로 보였습니다({ref}`8장 <absorbing-transmission>`). 이 부록에서는 같은 파가 {math}`\operatorname{Im}k_z>0`입니다.

**실수 {math}`k_x`에서는 여전히 근이 저절로 정해집니다.** 부록 A의 {ref}`② <ckr-real-kx>`에서 {math}`k_x`가 실수이면 {math}`\operatorname{Im}k_z^2=\operatorname{Im}\varepsilon`이고, 무손실이라 0이었습니다. 흡수 매질이면 이 값이 양수라 {math}`k_z^2`이 위쪽 반의 **안쪽**(0°와 180° 사이)에 있습니다. 제곱근은 각도를 절반으로 만드므로 고르는 근도 1사분면의 **안쪽**에 있어서 "나감"과 "줄어듦"을 둘 다 만족하고, 반대편 근은 3사분면 안쪽이라 둘 다 어깁니다. 무손실처럼 경계에 걸리지 않으니 비김도 없습니다.

```{figure} ../../figures/learn/learn/b02-real-kx-absorbing.svg
:alt: 두 패널이다. (a)는 kz² 평면으로, 허수부가 양수인 위쪽 반이 옅은 청록으로 칠해져 있다. 흡수 매질 ε=2.25+0.5i에서 실수 kx를 0에서 2.5로 키우면 kz²은 Im(kz²)=0.5인 수평선을 따라 2.25+0.5i(kx=0)에서 0.50i(kx=1.5)를 지나 −4.00+0.50i(kx=2.5)로 가고, 이 진한 선에 왼쪽 화살표가 있다. 같은 실수부의 무손실 매질 ε=2.25는 실수축 위의 회색 점선이다. (b)는 네 사분면을 칠한 kz 평면으로, 고른 근이 1.509+0.166i(kx=0)에서 0.500+0.500i(kx=1.5)를 지나 0.125+2.004i(kx=2.5)로 1사분면 안쪽을 지나는 진한 선이다. 버린 근 −kz는 3사분면 안쪽의 흐린 점선이고, 무손실 ε=2.25의 근은 실수축과 양의 허수축, 곧 1사분면의 경계를 따라가는 회색 점선이다.
:figclass: learning-figure

그림 B-2. 흡수 매질의 실수 {math}`k_x`입니다. {math}`k_z^2`은 {math}`\operatorname{Im}k_z^2=\operatorname{Im}\varepsilon`인 수평선을 따라 위쪽 반의 안쪽을 지나고, 고른 근은 1사분면 안쪽을 지납니다. 회색 점선은 비교용으로 그린 같은 실수부의 무손실 매질로, 부록 A처럼 실수축과 1사분면의 경계를 따라갑니다. `tools/docs/learning/draw_complex_kx_roots.py`로 계산해 그렸습니다.
```

그림 B-2의 {math}`k_x=0` 점을 표로 옮기면 이렇습니다.

| 경우 | {math}`k_z^2` (각도) | 고르는 근 (각도) | 반대편 근 |
| --- | --- | --- | --- |
| 흡수 매질 {math}`\varepsilon=2.25+0.5i`, {math}`k_x=0` | 2.25+0.5i (12.5°) | 1.509+0.166i (6.3°) | −1.509−0.166i |

부록 A의 무손실 1496개와 한 번에 뽑은 무작위 표본 3000개 가운데 흡수하는 1504개({math}`\operatorname{Re}\varepsilon`은 1에서 6, {math}`\operatorname{Im}\varepsilon`은 0.01에서 3, 실수 {math}`k_x`는 0에서 3 사이)에서도 네 규칙은 모두 **1504개 전부에서 같은 근**을 고릅니다.

**복소 {math}`k_x`에서는** {math}`k_z^2`의 허수부가 다음과 같습니다. 부록 A의 식에 {math}`\operatorname{Im}\varepsilon`이 더해진 것입니다.

```{math}
\operatorname{Im}k_z^2=\operatorname{Im}\varepsilon-2\operatorname{Re}k_x\operatorname{Im}k_x .
```

이 값이 음수가 되는 곳에서만 두 기준이 갈립니다. 무손실에서는 {math}`\operatorname{Re}k_x\operatorname{Im}k_x>0`이기만 하면 음수였지만, 흡수 매질에서는 {math}`2\operatorname{Re}k_x\operatorname{Im}k_x`가 {math}`\operatorname{Im}\varepsilon`보다 커야 음수가 됩니다.

부록 A의 {ref}`"커지는 근" 풀이 <ckr-complex-kx>`도 조금 바뀝니다. 거기서 본 대로 파가 가는 방향과 세기가 변하는 방향의 내적 {math}`\operatorname{Re}\mathbf k\cdot\operatorname{Im}\mathbf k`는 분산 관계에서 {math}`\tfrac12\operatorname{Im}\varepsilon`과 같습니다. 무손실이면 0이라 두 방향이 수직이었지만, 흡수 매질에서는 이 값이 양수라 수직이 아닙니다. 그래서 광선을 따라가도 세기가 일정하지 않고, 매질이 흡수하는 만큼 줄어듭니다.

```{figure} ../../figures/learn/learn/b03-air-vs-lossy.svg
:alt: 두 개의 kx 평면이다. (a) 공기 ε=1에서는 branch point n=1이 실수축 위에 있고, 두 기준이 갈리는 곳(Im kz²<0, 옅은 주황)이 kx 평면의 1사분면과 3사분면이다. 경로 D의 branch cut은 실수축의 kx>1 부분, fmmax의 branch cut은 1 왼쪽의 실수축(−1<kx<1)과 허수축, 경로 E는 1에서 곧게 위, Maystre는 1에서 휘어 올라가는 곡선이다. (b) 흡수 매질 ε=4+3i에서는 branch point n=2.121+0.707i가 실수축 위로 떠 있고, 옅은 주황 영역은 쌍곡선 Re kx·Im kx=1.5의 바깥쪽이며 실수축은 그 밖에 있다. 경로 D의 branch cut은 원점과 branch point를 잇는 직선을 branch point 너머로 늘인 반직선이고, fmmax의 branch cut은 branch point에서 왼쪽 위로, 뒤집기 전 주근의 선은 오른쪽 아래로 경계 쌍곡선을 따라간다.
:figclass: learning-figure

그림 B-3. 흡수 매질에서는 branch point가 실수축에서 떠 있어, 실수 {math}`k_x` 근처에서 네 규칙이 모두 같은 근을 고릅니다. `tools/docs/learning/draw_complex_kx_roots.py`로 계산해 그렸습니다.
```

그림 B-3(a)는 (b)와 같은 방식으로 공기에서 그린 지도로, 부록 A의 {ref}`네 branch cut <ckr-branch-cut>`을 한 장에 겹쳤습니다. 그림 B-3(b)는 앞 절의 입사 매질 {math}`\varepsilon=4+3i`를 이번에는 근을 고르는 반공간으로 본 것입니다. branch point는 {math}`n=\sqrt\varepsilon=2.121+0.707i`로 **실수축에서 떠 있습니다.**

- **옅은 주황**은 {math}`\operatorname{Im}k_z^2<0`, 곧 두 기준이 갈리는 곳입니다. 앞의 식에서 {math}`\operatorname{Im}k_z^2<0`은 {math}`\operatorname{Re}k_x\operatorname{Im}k_x>\operatorname{Im}\varepsilon/2`와 같으므로, 쌍곡선 {math}`\operatorname{Re}k_x\operatorname{Im}k_x=\operatorname{Im}\varepsilon/2=1.5`의 위쪽(원점에서 먼 쪽)입니다. **실수축은 통째로 이 영역 밖에 있습니다.**
- 주황 영역 밖에서는 네 규칙이 **모두 같은 근**을 고릅니다. 그리는 스크립트가 격자 전체에서 확인합니다.
- branch point는 그 경계 쌍곡선 위의 한 점입니다({math}`k_z^2=0`). 경계 위에서는 {math}`k_z^2`이 실수입니다. branch point에서 왼쪽 위로는 {math}`k_z^2>0`이라 {math}`\operatorname{Im}k_z>0` 규칙이 튀는 선이고, 오른쪽 아래(회색 점선)로는 {math}`k_z^2<0`이라 뒤집기 전 주근이 튀는 선입니다. 부록 A의 {ref}`계산식 <ckr-half-plane-rules>` 그대로입니다. 경로 D, 경로 E, Maystre의 선은 branch point에서 주황 영역 안쪽으로 뻗습니다.
- 경로 D의 선은 **원점과 branch point를 잇는 직선을 branch point 너머로 늘인 반직선**입니다(빨강). 원점에서 곧게 가는 경로 D가 branch point를 정확히 밟게 되는 목적지들이 바로 이 반직선이기 때문입니다. 무손실이면 이 반직선이 곧 실수축이 되며, 부록 A에서 본 {ref}`비김 <ckr-tie>`이 그것입니다.

**떠 있는 높이는 흡수가 정합니다.** 실수 {math}`k_x=a`({math}`a>0`)에서 허수부를 올려 주황 영역에 닿으려면, 쌍곡선의 식대로 허수부가 {math}`\operatorname{Im}\varepsilon/(2a)`까지 커져야 합니다. 그러니 허수부가 이보다 작으면 네 규칙이 모두 같습니다. 흡수가 작으면 이 틈도 작습니다. {math}`\varepsilon=2.25+10^{-6}i`이면 {math}`k_x=1.6\pm10^{-3}i`에서 경로 D가 무손실일 때처럼 튑니다({math}`0.0029+0.5568i\to0.0029-0.5568i`).

**숫자로 보면.** 부록 A의 두 표를, 같은 무작위 표본의 흡수 매질로 다시 세면 이렇습니다.

- 실수 {math}`k_x`에 {math}`\pm10^{-8}i`를 더했을 때 근이 뒤집힌 수는 흡수 매질 1504개에서 네 규칙 모두 **0**입니다(무손실 1496개에서는 경로 D 597, fmmax 899). 표본의 {math}`\operatorname{Im}\varepsilon`이 0.01 이상이고 비틀기가 {math}`10^{-8}`이라서, 비틀어도 위의 틈을 넘지 못하기 때문입니다.
- 같은 흡수 표본 가운데 312개에 허수부(−0.5에서 0.5 사이)를 붙여 복소 {math}`k_x`로 만들고, 각 규칙이 두 따라가기와 같은 근을 고른 수를 세면 이렇습니다.

| 규칙 | 경로 D와 같음 | 경로 E와 같음 |
| --- | ---: | ---: |
| 경로 D (2026년 10월 2일 전의 Wavebit) | 312/312 | 304/312 |
| Maystre (Wavebit) | 304/312 | **312/312** |
| {math}`\operatorname{Im}k_z>0` (fmmax) | 289/312 | 297/312 |

무손실 288개(경로 D와 E가 238개, Maystre와 E가 286개, fmmax와 D가 130개에서 같음)보다 규칙끼리 같은 근을 고르는 일이 훨씬 잦습니다. 주황 영역 밖에서는 네 규칙이 같으므로, 흡수가 규칙 사이의 차이를 줄입니다.

**2026년 10월 2일 전 Wavebit의 연속성 보정은 여기서 일했습니다.**

- 주근만 쓰면 branch cut이 그림 B-3(b)의 회색 점선 자리에 있고, 이 선은 실수축 쪽으로 내려옵니다.
- {ref}`앞 절 <ckr-absorbing-incidence>`의 감사 사례({math}`6.7\times10^{-5}` rad에 반사 진폭 2.96)는 금 기판, 곧 투과 차수의 근을 고르는 아래쪽 반공간에서 주근의 이런 선에 걸렸습니다. 당시에는 흡수 입사 매질에 실수 각도를 넣으면 {math}`k_x=n_{\mathrm{top}}\sin\theta`({math}`n_{\mathrm{top}}`: 입사 매질의 굴절률)가 복소수였습니다.
- 보정은 branch cut을 경로 D의 반직선으로 옮겼습니다. 곧 수동 매질에서 2026년 10월 2일 전의 Wavebit은 주근 + 연속성 보정 = 경로 D였습니다(이득 매질의 복소 {math}`k_x`는 {ref}`아래 <ckr-gain-complex-kx>`).
- 실수 `theta`가 실수 {math}`k_x`를 주게 된 뒤로는 실수 각도로 복소 {math}`k_x`가 생기지 않았고, 보정은 복소 `theta`를 쓸 때만 일했습니다.
- 보정이 켜진 경로 D는 2사분면 근을 고르기도 합니다. 무손실 유전체에서 경로 D(= 주근)는 {math}`\operatorname{Re}k_z<0`을 절대 고르지 않지만, 흡수 매질에서는 다릅니다.
- 그때의 `boundary_kz`로 재면, 위의 흡수 매질 312개에서 경로 D와 312개 모두, 주근과는 309개가 같았습니다. 나머지 3개가 보정이 켜진 자리입니다. 무손실 288개에서는 주근과 288개 모두 같았습니다({ref}`부록 A <ckr-continuation>`).
- 지금의 Wavebit은 보정 없이 Maystre 규칙 한 줄로 고릅니다. 지금의 `boundary_kz`는 같은 312개에서 Maystre와 312개 모두, 경로 D와 304개, 주근과 301개가 같습니다. 주황 영역 밖에서는 경로 D와 같은 근입니다.

::::{dropdown} 계산식: 연속성 보정의 닫힌 식
{math}`k_z^2` 평면에서 경로 D는 {math}`\varepsilon`에서 {math}`\varepsilon-k_x^2`로 가는 곧은 선분입니다.

```{math}
k_z^2(t)=\varepsilon-t\,k_x^2,\qquad t:0\to1 .
```

주근({math}`\operatorname{Re}k_z>0`인 근, 실수부가 0이면 비김 규칙으로 {math}`\operatorname{Im}k_z>0`)은 {math}`k_z^2`의 음의 실수축에서만 끊깁니다. 그러니 선분이 그 축을 건너는지만 보면 됩니다. 허수부 {math}`\operatorname{Im}\varepsilon-t\operatorname{Im}k_x^2`는 {math}`t`에 대해 일차식이므로 실수축을 많아야 한 번, {math}`t^*=\operatorname{Im}\varepsilon/\operatorname{Im}k_x^2`에서 지납니다. 다음 세 조건이 모두 맞으면 주근을 {math}`-k_z`로 뒤집습니다.

```{math}
\operatorname{Im}k_x^2>0,\qquad t^*<1,\qquad \operatorname{Re}\varepsilon-t^*\operatorname{Re}k_x^2<0 .
```

이 뒤집기가 2026년 10월 2일 전 Wavebit의 연속성 보정(`_off_principal_branch`)이었습니다. 이 식은 {math}`\operatorname{Im}\varepsilon\ge0`(수동)을 전제로 합니다. {math}`\operatorname{Im}k_x^2>0`인 쪽만 보고, {math}`t^*`가 0 이상인지는 따로 보지 않습니다. 이득 매질이면 이 전제가 깨집니다({ref}`복소 kx의 이득 반공간 <ckr-gain-complex-kx>`).

무손실이면 {math}`t^*=0`이라 셋째 조건이 {math}`\operatorname{Re}\varepsilon<0`이 되고, 유전체({math}`\varepsilon>0`)에서는 맞지 않습니다. 부록 A에서 "무손실 유전체에서는 보정이 켜지지 않는다"고 한 것이 이것입니다. 거꾸로 손실 없는 음의 {math}`\varepsilon`(두 부록 모두 다루지 않는 경우)에서는 {math}`\operatorname{Im}k_x^2>0`인 곳에서 보정이 켜집니다. 경로 E도 같은 방식으로 닫힌 식을 만들 수 있습니다. 두 닫힌 식은 따로 뽑은 무작위 표본 600개 가운데, 두 경로 중 하나라도 branch point에서 0.01 안쪽을 지나는 5개를 뺀 595개(무손실 287개, 흡수 308개)에서 한 걸음씩 따라간 결과와 모두 같았습니다.
::::

(ckr-gain)=
## 이득 매질: 수동 전제까지 풉니다

(ckr-gain-premise)=
### 전제가 깨집니다

이 절에서만 **수동 전제를 풉니다.** 레이저 매질처럼 빛을 키우는 **이득 매질**은 {math}`\operatorname{Im}\varepsilon<0`입니다. 그러면 부록 A의 {ref}`실수 kx의 ② <ckr-real-kx>`가 뒤집힙니다. {math}`k_x`가 실수여도 다음과 같습니다.

```{math}
\operatorname{Im}k_z^2=\operatorname{Im}\varepsilon<0 .
```

{math}`k_z^2`이 아래쪽 반으로 내려가고, 두 근이 2·4사분면에 앉습니다. 복소 {math}`k_x`가 아닌데도 두 기준이 서로 다른 근을 가리킵니다. **실수 {math}`k_x`에서도 어느 근을 쓸지 규칙으로 정해야 합니다.**

### 무엇이 답을 정하나: 인과율

결과가 원인보다 먼저 올 수 없다는 조건입니다. 계산으로는 부록 A의 {ref}`따라가기 <ckr-continuation>`를 {math}`k_x` 대신 주파수 {math}`\omega`로 하는 것이 됩니다. 출발점은 {math}`\omega`의 허수부가 아주 큰 곳입니다.

- 시간 인자가 {math}`e^{-i\omega t}`이므로, 허수부가 큰 {math}`\omega`는 시간에 따라 빠르게 커지는 신호입니다. 빛은 유한한 속도로 퍼지므로, 구조에서 {math}`d`만큼 떨어진 곳의 장은 빛이 그 거리를 오는 데 걸린 시간만큼 이전의, 더 약했던 원인에서 온 것입니다. 그래서 장은 멀어질수록 줄어들어야 합니다.
- 게다가 {math}`\omega`가 아주 크면 어떤 매질도 진공처럼 {math}`\varepsilon\to1`이 되어, 두 근이 {math}`k_z\approx\pm\omega/c`입니다(여기만 {math}`k_0`로 나누지 않은 값). {math}`\omega`의 허수부가 양수이므로 줄어드는 근은 {math}`+\omega/c`이고, 이것이 답입니다.

거기서 복소 주파수의 위쪽 반 안에서만 움직여 실제 {math}`\omega`까지 끊기지 않게 따라온 근이 인과적인 근입니다(Skaar 2006; Nistad와 Skaar 2008). 길이 여러 주파수의 {math}`\varepsilon(\omega)`를 지나가므로, 한 주파수의 {math}`\varepsilon`만으로는 답이 정해지지 않습니다. 같은 주파수에서 {math}`\varepsilon`이 같아도 분산이 다르면 반대 근이 나오는 예가 있습니다.

### 보통의 약한 이득이면: 이득을 0에서 켜서 이은 근

반전 분포를 가진 Lorentz 이득처럼 흔한 약한 이득이면, 임계각에서 떨어진 각도에서는 인과적인 근이 "{math}`k_x`는 그대로 두고 이득만 0에서 켜며 끊기지 않게 따라간 근"과 같습니다. 이득이 0이면 답이 확실하므로 따라가기와 같은 발상입니다. {math}`k_z^2=X-ig`로 씁니다. {math}`X=\operatorname{Re}\varepsilon-k_x^2`는 이득이 없을 때의 {math}`k_z^2`이고, {math}`g=-\operatorname{Im}\varepsilon>0`이 이득입니다. 이득을 켜는 길은 {math}`k_z^2` 평면에서 {math}`X`에서 곧장 아래로 내려가는 길입니다.

- **전파({math}`X>0`)**: 양의 실수축에서 아래로 내려가므로 주근이 그대로 이어집니다. 4사분면, 곧 **나가면서 커지는 파**입니다. 이득 매질 속을 진행하며 증폭되는 것이므로 자연스럽습니다(Skaar 2006).
- **전반사({math}`X<0`, 소멸)**: 출발 근이 {math}`+i\sqrt{|X|}`(줄어드는 근)이고, 음의 실수축 바로 아래로 내려가며 2사분면으로 갑니다. **멀어질수록 줄어들지만 에너지는 계면 쪽으로 들어오는 파**입니다. 이득 매질이 계면 근처의 소멸파를 키워 계면에 에너지를 보태는 셈입니다. 그래서 전반사율이 1을 넘습니다. 이를 증폭 전반사(evanescent gain)라고 합니다(Grepstad와 Skaar 2011).

### 예: 유리에서 이득 매질로

유리({math}`n=1.5`)에서 이득 매질 {math}`\varepsilon=1-0.05i`로 TE 빛을 비춥니다. 임계각은 41.81°입니다.

| 입사각 | 이득을 켜서 이은 근 | 이때 R | Wavebit (Maystre) | 경로 D (2026년 10월 2일 전의 Wavebit, 주근) | fmmax ({math}`\operatorname{Im}k_z>0`) |
| --- | --- | ---: | --- | --- | --- |
| 0° | 1.0003 − 0.0250i (4) | 0.0400 | 같음 | 같음 | 반대 근, R = 24.98 |
| 30° (전파) | 0.6625 − 0.0377i (4) | 0.1056 | 같음 | 같음 | 반대 근, R = 9.467 |
| 41.81° | 0.1581 − 0.1581i (4) | 0.5723 | 같음 | 같음 | 반대 근, R = 1.747 |
| 41.82° | −0.1575 + 0.1587i (2) | 1.7434 | 같음 | **반대 근, R = 0.5736** | 같음 |
| 45° (전반사) | −0.0694 + 0.3603i (2) | **1.2646** | 같음 | **반대 근, R = 0.7907** | 같음 |

괄호 안 숫자는 사분면입니다. TE 반사계수는 {math}`r=(k_{z1}-k_{z2})/(k_{z1}+k_{z2})`({math}`k_{z1}`: 유리, {math}`k_{z2}`: 이득 매질)이므로, {math}`k_{z2}`를 반대 근 {math}`-k_{z2}`로 바꾸면 {math}`r`이 {math}`1/r`이 되고 R도 서로 역수입니다. 41.81°와 41.82° 두 줄은 이득을 켜서 이은 근일 뿐, 인과적인 근이라고 장담할 수 없습니다. 임계각 바로 옆에서는 인과적인 답이 튀는 각도가 분산에 따라 달라지기 때문입니다(아래 "조심할 것" 1). 그림 B-4는 표의 30°와 45° 줄을 {math}`k_z` 평면에 옮긴 것입니다.

```{figure} ../../figures/learn/learn/b04-gain.svg
:alt: 두 개의 kz 평면이다. (a) 30도 전파에서는 두 근이 ±(0.6625−0.0377i)이고, 이득을 켜서 이은 근(보통의 약한 이득에서 인과적인 근) 0.6625−0.0377i(나감·커짐, R=0.1056)는 Wavebit(Maystre)과 경로 D도 고르며, 반대 근 −0.6625+0.0377i(R=9.4669)는 fmmax가 고른다. (b) 45도 전반사에서는 인과적인 근 −0.0694+0.3603i(줄어들지만 계면으로 들어옴, R=1.2646)를 Wavebit(Maystre)과 fmmax가 고르고, 반대 근 0.0694−0.3603i(R=0.7907)를 경로 D(2026년 10월 2일 전의 Wavebit)가 고른다.
:figclass: learning-figure

그림 B-4. 이득 반공간(유리에서 {math}`\varepsilon=1-0.05i`로, TE)의 두 근과 인과적인 근입니다. 보통의 약한 이득이면, 전파에서는 나가면서 커지는 근이, 전반사에서는 줄어들지만 계면으로 들어오는 근이 인과적인 근입니다. `tools/docs/learning/draw_complex_kx_roots.py`로 계산해 그렸습니다.
```

### 규칙마다 보면 (실수 kx, 이득 반공간)

- **경로 D(2026년 10월 2일 전의 Wavebit), 곧 주근**: 전파는 맞고, 전반사는 커지는 소멸파를 골라 틀립니다. 실수 {math}`k_x`이면 경로 D의 선분({math}`\varepsilon`에서 {math}`\varepsilon-k_x^2`)이 {math}`k_z^2` 평면에서 {math}`\operatorname{Im}=-0.05`인 수평선이라 음의 실수축을 건너지 않습니다. 그래서 연속성 보정이 켜지지 않고 주근 그대로입니다. 전반사에서는 {math}`k_z^2`이 음의 실수축 바로 아래(각도 −180° 근처)에 있어서, 각도를 절반으로 줄이면 −90° 근처, 곧 커지는 근입니다.
- **fmmax**: 전반사는 맞고, 전파는 들어오는 파를 골라 틀립니다. 수직 입사에서도 R = 24.98입니다.
- **Maystre(2026년 10월 2일부터의 Wavebit)**: "이득을 켜서 이은 근"과 이득 세기에 상관없이 **정확히** 같습니다. 전파이면 {math}`k_z^2`의 각도가 −90°에서 0° 사이라서 주근의 각도가 −45°에서 0° 사이이고, {math}`\operatorname{Re}+\operatorname{Im}>0`이므로 4사분면 근을 그대로 둡니다. 전반사이면 {math}`k_z^2`의 각도가 −180°에서 −90° 사이라서 주근의 각도가 −90°에서 −45° 사이이고, 2사분면 근으로 뒤집습니다. 그래서 보통의 약한 이득이면 임계각에서 떨어진 곳에서 인과적인 답과 맞습니다.

다만 한 주파수의 값만 보는 규칙은 어느 것도 모든 이득 매질에 맞을 수는 없습니다. 임계각에서 멀리 떨어진 곳까지 답을 뒤집는 매질도 만들 수 있는데, 그러려면 분산이 특이해야 합니다(Grepstad와 Skaar 2011의 설계된 매질).

**경로 E**는 실수 {math}`k_x`의 답을 출발점으로 쓰는데, 이득 매질에서는 실수 {math}`k_x`에서도 두 기준이 갈라지므로 출발점의 답부터 저절로 정해지지 않습니다. 대조한 솔버 가운데 경로 E를 쓰는 것은 없습니다.

(ckr-gain-complex-kx)=
### 복소 kx의 이득 반공간

위의 표와 비교는 모두 실수 {math}`k_x`에서입니다. {math}`k_x`도 복소수이면 한 가지가 더 있었습니다. 2026년 10월 2일 전 Wavebit의 연속성 보정({ref}`흡수 반공간 <ckr-lossy>`의 닫힌 식)은 {math}`\operatorname{Im}\varepsilon\ge0`을 전제로 하므로, 이득 반공간에서는 경로 D를 한 걸음씩 걸은 결과와 다를 수 있습니다.

- {math}`\operatorname{Im}k_x^2>0`이면 {math}`t^*=\operatorname{Im}\varepsilon/\operatorname{Im}k_x^2`가 음수, 곧 선분 밖인데도 셋째 조건으로 뒤집을 수 있습니다.
- {math}`\operatorname{Im}k_x^2<0`이면 선분이 음의 실수축을 건너도 닫힌 식은 보지 않습니다.

무작위 이득 매질({math}`\operatorname{Re}\varepsilon`은 1에서 6, {math}`\operatorname{Im}\varepsilon`은 −3에서 −0.01 사이)과 복소 {math}`k_x`(실수부 0에서 3, 허수부 −0.5에서 0.5 사이) 399개에서 닫힌 식이 걸은 경로 D와 14개가 달랐습니다(앞의 경우 6개, 뒤의 경우 8개). 예를 들어 {math}`\varepsilon=1-0.05i`에서는 이렇습니다.

| {math}`k_x` | 닫힌 식 (2026년 10월 2일 전 Wavebit의 연속성 보정) | 걸은 경로 D |
| --- | --- | --- |
| 0.1 + 5i | −5.0991 + 0.1030i | 5.0991 − 0.1030i |
| 3 − 0.05i | 0.0442 + 2.8283i | −0.0442 − 2.8283i |

이득 반공간에서 인과적인 근이 경로 D를 따르는지부터가 이 부록의 범위 밖이므로, 여기서는 "닫힌 식 = 경로 D"라는 등식이 이득 매질의 복소 {math}`k_x`에서 깨진다는 것만 적어 둡니다. 지금의 Wavebit에는 이 닫힌 식이 없습니다. Maystre 규칙은 이득 매질에서도 같은 한 줄이고, 그 근은 반공간에 흡수를 크게 줬다가 실제 값까지 줄이며 이은 근입니다([분기 처리](../../theory/rcwa/branch-cuts.md)의 8절).

::::{dropdown} 계산식: 두꺼운 이득 층의 극한은 판정 기준이 못 됩니다
이득 매질을 반공간 대신 두께 {math}`h`인 층으로 두면(유리 | 이득 층 | 공기, 파장 1 µm) 층 안에서는 {math}`\pm k_z`가 둘 다 들어가서 근을 고를 필요가 없습니다. Airy 공식으로 푼 값이 Wavebit의 층 계산과 9자리까지 같습니다(45°, {math}`h=2` µm에서 둘 다 1.264637712). 유한한 층에서는 근의 부호가 답을 바꾸지 않는다는 것도 알려져 있습니다(Lakhtakia 외 2007). 근 고르기가 문제가 되는 것은 반무한 반공간뿐입니다({ref}`부록 A의 전제 4 <ckr-premises>`).

그렇다고 {math}`h`를 한없이 키운 극한을 반공간의 답으로 쓸 수는 없습니다. {math}`h`를 키우면 Airy 식의 {math}`e^{2ik_zh}`가 {math}`\operatorname{Im}k_z>0`인 근으로 쓰면 0으로, 반대 근으로 쓰면 무한대로 갑니다. 어느 쪽으로 써도 극한은 {math}`\operatorname{Im}k_z>0`인 근으로 계산한 계면 하나의 반사계수입니다. 곧 두꺼운 층의 극한은 식의 모양 때문에 늘 줄어드는 근(fmmax 쪽)을 고르며, 모든 각도에서 그렇습니다.

- 45°: {math}`h=2` µm에서 R = 1.264638, {math}`h=5` µm부터 1.264630입니다. 인과적인 값과 같지만, 규칙이 겹친 것일 뿐 독립 증거가 아닙니다.
- 30°: 약 9.9 µm를 넘으면 층 안 왕복 이득이 1을 넘는 발진 영역입니다. 그 뒤 R은 3.05(10 µm), 9.488(20 µm), 9.467(100 µm)로 fmmax 쪽 값에 갑니다. 인과적이지 않은 값입니다(Skaar 2006).

인과적인 답의 근거는 복소 주파수로 이어 붙이는 계산입니다. 이 부록을 쓰면서 두 가지 분산 모형으로, 복소 주파수 평면의 서로 다른 네 길을 따라 따로 확인했습니다. 이득이 좁은 대역에만 있는 모형(복소 주파수의 위쪽 반에 {math}`k_z`의 branch point가 없는 모형)에서는 네 길이 모두 같은 근을 주었습니다. 단일 Lorentz 모형에서는 위쪽 반에 branch point가 있어서, 그 반대편으로 도는 길만 반대 근을 주었습니다. 그 까닭은 아래 "조심할 것" 1의 불안정성입니다. 이 확인은 `tools/docs/learning/draw_complex_kx_roots.py`에는 들어 있지 않습니다.
::::

### 조심할 것 셋

1. **임계각 근처에서는 평면파 답이 뜻을 잃습니다.** 위 표에서 이득을 켜서 이은 근이 41.81°와 41.82° 사이에서 4사분면에서 2사분면으로 튑니다. 임계각에서는 {math}`X=0`이라 이득을 켜는 길이 branch point({math}`k_z^2=0`)에서 출발하기 때문입니다. 부록 A에서 본 {ref}`"branch point를 밟는" 상황 <ckr-tie>`입니다. 인과적인 답도 임계각 근처에서 튀는데, 튀는 각도가 정확히 임계각은 아니고 분산에 따라 조금씩 달라집니다. 그래서 임계각 바로 옆에서는 어느 규칙이 맞다고 말할 수 없습니다. 물리적으로는 더 근본적인 문제가 있습니다. {math}`k_x`를 고정하면, 투과 쪽 파가 계면을 스치듯 나란히 달리는({math}`k_z=0`) 주파수가 있습니다. 그 주파수에서 매질이 이득을 주면, 그 파는 무한히 넓은 계면을 따라 달리며 끝없이 증폭됩니다. 수식으로는 복소 주파수의 위쪽 반에 {math}`k_z`의 branch point가 생기는 것이고, 그러면 위쪽 반에서 이어 붙이는 계산도 도는 길에 따라 다른 근을 줍니다. 단일 Lorentz 이득선은 모든 주파수에서 이득이라 이 불안정성이 늘 있고, 이득이 좁은 대역에만 있으면 임계각 근처의 좁은 각도 범위에서만 생깁니다. 그 범위는 이득 대역에 따라 다릅니다. 그래서 반무한 이득 반공간의 평면파 답은 제한된 시간 동안의 준정상 상태로만 뜻이 있고, 임계각에서 떨어진 곳에서만 믿을 수 있습니다(Nistad와 Skaar 2008; Grepstad와 Skaar 2011).

   **빔으로 보면.** 무한 평면파 대신 폭이 {math}`W`인 가우스 빔의 정상 상태를 계산하면 그림이 분명해집니다. 스침 근처, 곧 {math}`|\operatorname{Re}k_z^2|<|\operatorname{Im}k_z^2|` 쯤 안에서는 빔 중심의 장이 빔 폭과 함께 끝없이 커집니다. 유리에서 {math}`\varepsilon=1-0.05i`로 42°에 비추면, 빔 폭이 이득 길이 {math}`L_g=2\sqrt{\operatorname{Re}\varepsilon}/(k_0|\operatorname{Im}\varepsilon|)`의 2.5, 5, 7.5배일 때 {math}`|E_r/E_\text{inc}|^2`이 2.13, 181, 1.1e8입니다. 그 밖에서는 빔이 이득 길이의 몇 배만 넓으면 분산과 상관없이 Maystre 값으로 수렴합니다. 분산에 따라 움직이는 고정 {math}`k_x` 뒤집힘 각도는 빔에는 보이지 않습니다. 다만 가장자리가 날카로운 빔은 어느 각도에서나 가장자리에서 자라는 파가 생기므로, 이 결론은 매끄러운 빔에 대한 것입니다. Wavebit은 {math}`|\operatorname{Re}k_z^2|<2|\operatorname{Im}k_z^2|`인 차수마다 `GainCutoffWarning`을 냅니다. 수치와 근거는 [분기 처리](../../theory/rcwa/branch-cuts.md)의 "Gain near grazing"에 있습니다.
2. **이득이 크면 정상 상태 해가 없습니다.** {math}`\varepsilon(\omega)`가 복소 주파수의 위쪽 반에 극이나 홀수 차 영점을 가지면 매질 자체가 불안정해 스스로 발진합니다(절대 불안정). 한 주파수의 {math}`\varepsilon`만 보는 솔버는 이것을 알아챌 수 없습니다.
3. **문헌에도 반론이 있습니다.** 반무한 반공간에서 증폭 전반사가 정말 일어나느냐를 두고 의견이 갈립니다. 반대하는 쪽은 Siegman(2010), T. S. Mansuripur와 M. Mansuripur(2014)입니다. 에너지가 계면에서 멀어지는 근을 고르면 전반사에서는 커지는 근이 되어 R < 1이며, 2026년 10월 2일 전의 Wavebit이 내던 값이 이쪽입니다. 찬성하는 쪽은 Willis 외(2008, FDTD), Grepstad와 Skaar(2011), M. Mansuripur와 Jakobsen(2021, 인과적인 파속 해석)입니다. 이 부록은 인과율로 계산하는 쪽(Grepstad와 Skaar)을 따르고, 지금의 Wavebit(Maystre)도 이쪽 값을 냅니다.

**실용적인 결론: 이득은 유한한 층에 넣고, 위아래 반공간은 수동 매질로 둡니다.** 그러면 근 고르기 문제가 생기지 않습니다. Wavebit이 지금 이득 반공간을 어떻게 다루는지는 {ref}`다음 절 <ckr-b-codes>`에 있습니다.

(ckr-b-codes)=
## 흡수나 이득이 있어야 드러나는 코드 차이

부록 A의 {ref}`코드 비교 <ckr-codes>`는 근 규칙과 복소 {math}`k_x` 입력, 무손실에서의 branch cut을 모았습니다. 흡수 입사 매질과 이득 반공간에서만 드러나는 차이를 여기 모읍니다. 이득 반공간 열은 실수 {math}`k_x`에서의 이야기입니다.

| 코드 | 흡수 입사 매질의 각도를 {math}`k_x`로 | 이득 반공간 |
| --- | --- | --- |
| Wavebit | 실수 `theta`는 위상각 규약(실수 {math}`k_x`), 복소 `theta`는 형식적인 {math}`n\sin\theta` | Maystre(2026년 10월 2일부터): 보통의 약한 이득에 맞음. 그 전에는 늘 주근: 전파 맞음, 전반사 틀림 |
| RETICOLO V10 | 사용자가 `beta0`를 직접 준다 | 규칙으로 따지면 보통의 약한 이득에 맞음 |
| S4 | {math}`\operatorname{Re}(n)\sin\theta`로 실수 {math}`k_x` | 재지 않음 |
| fmmax 1.7.1 | 각도 도우미는 {math}`\sqrt{\operatorname{Re}\varepsilon}\sin\theta` | 전파 틀림, 전반사 맞음 |

**같은 "30°"가 코드마다 다른 {math}`k_x`.** 흡수 매질 {math}`\varepsilon=4+3i` 안에서 30°를 주면 이렇게 됩니다. Wavebit의 값은 {ref}`8장의 예 <absorbing-incidence-example>`에서 유도했습니다.

| 코드 | {math}`k_x/k_0` | 방식 |
| --- | --- | --- |
| Wavebit | 1.077700 | 위상각 규약: 실수 {math}`k_x`와 분산 관계를 함께 푼다 |
| S4 | 1.060660 | {math}`\operatorname{Re}(n)\sin\theta` |
| fmmax 도우미 | 1.000000 | {math}`\sqrt{\operatorname{Re}\varepsilon}\sin\theta`, Wavebit 규약으로는 27.72° |
| RETICOLO | 사용자가 준 `beta0` | 형식적인 {math}`n\sin\theta`를 주면 1.060660 + 0.353553i |

그러니 흡수 입사 매질에서 솔버끼리 맞대 볼 때는 각도가 아니라 {math}`k_x`를 맞춰야 합니다.

**Wavebit.**

- 흡수 입사 매질의 실수 `theta`는 위상각 규약으로 실수 {math}`k_x`가 되고, 복소 `theta`는 형식적인 {math}`n\sin\theta`입니다({ref}`흡수 입사 매질 <ckr-absorbing-incidence>`).
- 이득 매질을 통째로 거부하는 곳은 실험적인 `ConformingNVF` 클래스([20장](code-map.md))뿐입니다. 일반 풀이는 이득 층도 이득 반공간도 풉니다. 반공간으로 거부하는 것은 이득이 있는 이중 음 매질({math}`\operatorname{Re}\varepsilon<0`, {math}`\operatorname{Re}\mu<0`) 가운데 빛이 전파하는({math}`\operatorname{Re}(\varepsilon\mu)>0`) 것 하나로, 한 주파수의 값만으로는 근을 정할 수 없기 때문입니다. 같은 매질도 층으로 넣으면 풉니다. 반공간의 근은 2026년 10월 2일부터 Maystre 규칙이므로, 보통의 약한 이득이면 임계각에서 떨어진 곳에서 인과적인 답과 맞습니다. 임계각 바로 옆은 어느 규칙으로도 정답이 없습니다({ref}`이득 매질 <ckr-gain>`의 "조심할 것" 1).
- R + T가 입사 전력을 넘어도 경고하지 않습니다. 이득 매질에서는 그것이 맞는 답일 수 있기 때문입니다. 2026년 10월 3일 전에는 일반 경고 `UnphysicalResultWarning`이 이득 층과 증폭 전반사(R > 1)를 내는 이득 반공간에서 떴지만, 맞는 답을 실패로 부른다는 까닭으로 지워졌습니다([결과 검사](../../reference/common/result-guards.md)). `docs/validation/learning-records/complex-kx-roots.json`은 그 전인 `6bb364b`에서 잰 것이라 경고 목록에 이 이름이 남아 있습니다.
- 이득 반공간에 따로 뜨는 경고는 `GainCutoffWarning`입니다. 어떤 차수가 스치는 각도 근처, 곧 {math}`|\operatorname{Re}k_z^2|<2|\operatorname{Im}k_z^2|`에 있으면 그 반공간마다 뜨고, `solver.gain_cutoff_band()`가 위·아래 반공간별로 어느 차수인지 알려 줍니다. 위 예에서는 약 39.2°부터 44.4°까지가 이 띠이고, 30°와 45°에서는 경고가 없습니다.
- 위 예를 Wavebit으로 직접 풀면 30°에서 0.105632, 45°에서 1.264630입니다. 같은 크기의 손실 기판({math}`1+0.05i`)은 30°에서 0.105632, 45°에서 0.790745입니다. 전파 쪽에서는 이득과 손실이 같은 R을 내고, 전반사 쪽에서는 다릅니다.
- 2026년 10월 2일 전에는 실수 {math}`k_x`에서 늘 주근이라, 이득 기판의 R이 손실 기판과 반올림 오차 안에서 같았습니다(45°에서도 0.790745). 주근은 {math}`\varepsilon`을 켤레로 바꾸면 {math}`k_z`도 켤레로 바꾸고, 위 매질이 무손실이면 {math}`r`도 켤레가 되기 때문입니다. 곧 그때의 Wavebit은 이득 반공간을 같은 크기의 손실 반공간처럼 풀었고, 경고도 없었습니다. 복소 {math}`k_x`의 이득 반공간에서는 연속성 보정의 닫힌 식이 경로 D와도 달랐습니다({ref}`위 <ckr-gain-complex-kx>`).
- 이득은 유한한 층에 넣는 편이 안전합니다.

**RETICOLO.**

- 흡수 입사 매질에서도 `res2`는 기본값으로 빈 목록을 냅니다. 부록 A의 {ref}`코드 비교 <ckr-codes>`에 적은 `tolh`, `tolb` 함정과 같고, 푸는 법도 같습니다.
- 각도에서 `beta0`를 {math}`n\sin\theta`로 만들면, 입사 매질이 흡수 매질일 때 조용히 복소 `beta0`가 됩니다. 이 값은 Wavebit이 실수 `theta`에 쓰는 위상각 규약의 {math}`k_x`가 아니라, 그림 B-1의 주황 선, 곧 형식적인 {math}`n\sin\theta`입니다.
- 이득 반공간에서는, 규칙(Maystre)으로 따지면 "이득을 켜서 이은 근"과 같습니다.

**S4.** 각도에서 {math}`\operatorname{Re}n`, {math}`\operatorname{Re}\omega`로 실수 {math}`k_x`를 만듭니다. 흡수 입사 매질에서는 위상각 규약과 다른 값입니다(위 표). 이득 반공간은 재지 않았습니다.

**fmmax.** 각도 도우미 `plane_wave_in_plane_wavevector`는 {math}`\sqrt{\operatorname{Re}\varepsilon}\sin\theta`를 씁니다. 흡수 매질에서는 교차검증용 {math}`k_x`를 직접 넘기는 편이 안전합니다. 이득 반공간에서는 {math}`\operatorname{Im}k_z>0` 규칙이라 전파에서는 들어오는 파를 고르고, 전반사에서는 이득을 켜서 이은 근과 같은 근을 고릅니다.

(ckr-b-summary)=
## 한 장 요약

- 푸는 전제: 부록 A의 전제 2(무손실). 흡수 입사 매질과 흡수 반공간은 수동 안에서 넓히는 것이고, 이득 매질만 수동 전제까지 풉니다.
- **흡수 입사 매질**에서는 실수 `theta`(위상각 규약, 실수 {math}`k_x`)와 복소 `theta`(형식적인 {math}`n\sin\theta`)가 다른 {math}`k_x`를 줍니다. `theta`에 허수부를 {math}`10^{-12}`만 줘도 {math}`k_x`의 허수부가 0에서 {math}`\operatorname{Im}(n)\sin\theta`로 뜁니다. 그래서 흡수 입사 매질의 복소 `theta`는 실수 `theta`를 비튼 입력이 아닙니다. 무손실 입사 매질이면 두 규칙이 겹칩니다.
- **흡수 반공간**에서는 branch point가 떠 있어서, {math}`\operatorname{Im}k_x`가 {math}`\operatorname{Im}\varepsilon/(2\operatorname{Re}k_x)`보다 작으면 네 규칙이 모두 같습니다. 실수 {math}`k_x`에서는 근이 1사분면 안쪽이라 비김도 없습니다. 흡수가 아주 작으면 무손실처럼 튑니다.
- 2026년 10월 2일 전 Wavebit의 **연속성 보정**은 무손실 유전체에서는 켜지지 않고 흡수 반공간에서 켜져서, 주근의 branch cut을 경로 D의 반직선으로 옮겼습니다. 그 닫힌 식은 수동 매질을 전제로 해서, 이득 반공간의 복소 {math}`k_x`에서는 경로 D와도 달랐습니다. 지금의 Wavebit은 보정 없이 Maystre 규칙 한 줄을 씁니다.
- **이득 매질**에서는 실수 {math}`k_x`에서도 두 기준이 갈립니다. 답은 인과율이 정하고 한 주파수의 {math}`\varepsilon`만으로는 다 정해지지 않습니다. 보통의 약한 이득이면 임계각에서 떨어진 곳에서 전파는 4사분면(증폭), 전반사는 2사분면(증폭 전반사, R > 1)입니다. Maystre는 이것과 맞고, Wavebit도 2026년 10월 2일부터 Maystre를 씁니다. 그 전의 Wavebit은 전반사 쪽에서 R < 1을 냈습니다. 임계각 근처와 강한 이득에서는 평면파 답 자체가 뜻을 잃습니다. 이득은 유한한 층에 넣습니다.
- 흡수 입사 매질에서는 **같은 30°가 코드마다 다른 {math}`k_x`가 됩니다.** 교차검증은 {math}`k_x`를 맞춰서 합니다.

## 용어

| 용어 | 이 부록에서의 뜻 | 다른 이름 |
| --- | --- | --- |
| 수동 매질 | 스스로 에너지를 만들지 않는 매질, {math}`\operatorname{Im}\varepsilon\ge0` | passive medium |
| 무손실 / 흡수 / 이득 | {math}`\operatorname{Im}\varepsilon=0` / {math}`>0` / {math}`<0` | lossless / absorbing (lossy) / gain |
| 위상각 규약 | 실수 `theta`를 위상이 나아가는 방향의 각도로 읽어 실수 {math}`k_x`를 구하는 것 | phase-angle incidence |
| 형식적인 {math}`n\sin\theta` | 복소 `theta`의 {math}`\sin\theta`에 입사 매질의 복소 굴절률 {math}`n`을 그대로 곱해 {math}`k_x`를 만드는 것 | formal complex angle |
| 연속성 보정 | 경로 D를 닫힌 식으로 만들려고 주근을 뒤집는 것, {math}`\operatorname{Im}\varepsilon\ge0`을 전제로 함, 무손실 유전체에서는 켜지지 않고 흡수 반공간에서 켜짐, 2026년 10월 2일에 Wavebit에서 지움 | `_off_principal_branch` |
| 이득 매질 | {math}`\operatorname{Im}\varepsilon<0`, 빛을 키우는 매질 | gain / active medium |
| 인과적인 근 | 복소 주파수의 위쪽 반에서 이어 붙인 근 | causal root |
| 이득을 켜서 이은 근 | {math}`k_x`를 고정하고 이득을 0에서 켜며 따라간 근, 보통의 약한 이득에서 인과적인 근 | — |
| 증폭 전반사 | 이득 매질 쪽 전반사에서 R > 1 | evanescent gain, amplified TIR |

나머지 용어(나감·줄어듦, 주근, branch point, branch cut, 경로 D·E, Maystre 규칙 등)는 부록 A의 용어 표에 있습니다.

(ckr-b-confusing)=
## 헷갈리기 쉬운 문장

- **"흡수 입사 매질에서 `theta = 0.4 + 1e-12j`는 `theta = 0.4`를 살짝 비튼 것이다."** 아닙니다. 실수 `theta`는 위상각 규약, 복소 `theta`는 형식적인 {math}`n\sin\theta`라서 {math}`k_x`의 허수부가 0에서 0.275로 뜁니다({ref}`흡수 입사 매질 <ckr-absorbing-incidence>`).
- **"흡수 매질이면 근 문제는 없다."** 흡수 크기만큼만 그렇습니다. 흡수가 아주 작으면 무손실처럼 튑니다({ref}`흡수 반공간 <ckr-lossy>`).
- **"이득 매질도 1사분면 근을 고르면 된다."** 아닙니다. 이득이 있으면 실수 {math}`k_x`에서도 1사분면 근이 없습니다({ref}`이득 매질 <ckr-gain>`).
- **"이득 반공간에서 Wavebit은 전반사를 R < 1로 낸다."** 2026년 10월 2일 전의 이야기입니다. 지금의 Wavebit은 Maystre 규칙이라, 보통의 약한 이득이면 전반사 쪽에서 증폭 전반사(R > 1)를 냅니다. 임계각 근처는 어느 규칙도 믿을 수 없습니다({ref}`이득 매질 <ckr-gain>`).
- **"두꺼운 이득 층으로 풀어 보면 반공간의 답을 알 수 있다."** 아닙니다. 그 극한은 모든 각도에서 {math}`\operatorname{Im}k_z>0` 규칙의 값입니다({ref}`이득 매질 <ckr-gain>`).
- **"각도만 맞추면 솔버끼리 같은 빛이다."** 흡수 입사 매질에서는 아닙니다. {math}`\varepsilon=4+3i`의 30°가 Wavebit에서 1.077700, S4에서 1.060660, fmmax 도우미에서 1.000000입니다({ref}`코드 차이 <ckr-b-codes>`).

(ckr-b-references)=
## 참고문헌과 이 부록의 검증 범위

이득 매질:

- [J. Skaar, “Fresnel equations and the refractive index of active media,” Phys. Rev. E 73, 026605 (2006)](https://doi.org/10.1103/PhysRevE.73.026605). 이득 매질에서 굴절률과 {math}`k_z`의 부호를 인과율로 정하는 방법입니다.
- [B. Nistad, J. Skaar, “Causality and electromagnetic properties of active media,” Phys. Rev. E 78, 036603 (2008)](https://doi.org/10.1103/PhysRevE.78.036603). 파수 선택과 절대 불안정성입니다.
- [J. O. Grepstad, J. Skaar, “Total internal reflection and evanescent gain,” Opt. Express 19, 21404 (2011)](https://doi.org/10.1364/OE.19.021404). 이득 매질 쪽 전반사의 인과적인 근과 증폭 전반사입니다.
- [A. E. Siegman, “Fresnel Reflection, Lenserf Reflection and Evanescent Gain,” Opt. Photonics News 21(1), 38 (2010)](https://doi.org/10.1364/OPN.21.1.000038). 증폭 전반사에 반대하는 견해입니다.
- [K. J. Willis, J. B. Schneider, S. C. Hagness, “Amplified total internal reflection: theory, analysis, and demonstration of existence via FDTD,” Opt. Express 16, 1903 (2008)](https://doi.org/10.1364/OE.16.001903). FDTD로 본 증폭 전반사입니다.
- [T. S. Mansuripur, M. Mansuripur, “Fresnel reflection from a cavity with net roundtrip gain,” Appl. Phys. Lett. 104, 121106 (2014)](https://doi.org/10.1063/1.4869578). 단일 계면에서의 증폭 전반사에 반대하는 견해입니다.
- [M. Mansuripur, P. K. Jakobsen, “Theoretical analysis of Fresnel reflection and transmission in the presence of gain media,” Opt. Rev. 28, 626 (2021)](https://doi.org/10.1007/s10043-021-00690-4). 인과적인 파속 해석으로 본 증폭 전반사입니다.
- [A. Lakhtakia, J. B. Geddes III, T. G. Mackay, “When does the choice of the refractive index of a linear, homogeneous, isotropic, active, dielectric medium matter?,” Opt. Express 15, 17709 (2007)](https://doi.org/10.1364/OE.15.017709). 유한한 층에서는 근의 부호가 답을 바꾸지 않는다는 결과입니다.

대조한 솔버(RETICOLO, S4, fmmax)의 문헌은 {ref}`부록 A <complex-kx-roots>`의 참고문헌에 있습니다.

Siegman(2010)은 원문을 읽지 못해, 그 입장은 인용한 논문들을 통해서만 확인했습니다. `theta = 0.4`를 {math}`k_x`로 바꾼 Wavebit 값과 이득 기판·이득 층의 Wavebit 값은 `docs/validation/learning-records/complex-kx-roots.json`에 기록했습니다. 2026년 10월 2일 전의 Wavebit 값(경로 D)은 기록의 `path_d`에 그 커밋과 함께 남아 있습니다. "30°" 표의 Wavebit 값은 그리는 스크립트가 위상각 규약의 식으로 계산합니다. 그림의 수치는 모두 `tools/docs/learning/draw_complex_kx_roots.py`가 그리기 전에 스스로 검산하며, 경로 D·E의 닫힌 식과 한 걸음씩 따라가기, 각 규칙의 branch cut 식과 격자에서 실제로 튀는 자리({math}`\varepsilon=4+3i` 포함), 두 각도 규칙이 말하는 바가 맞는지 확인한 뒤에만 그림을 씁니다.
