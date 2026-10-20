(power-and-efficiency)=
# 13. 전력 흐름과 회절 효율

:::{note}
다운로드 가능한 교육 스크립트는 현재 물리 Problem/Job API를 사용합니다. 일부 설명 코드와 그림에는 이전 API 표기가 남아 있으므로 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 함께 참고하세요. 과거 측정 파일과 원본 출처는 보존하며, 이전한 스크립트의 검증과 구분합니다.
:::

반사율·투과율은 **전력의 비율**입니다. 복소 전기장 진폭 계수의 크기를 제곱하면 전기장 크기의 제곱 비율을 얻지만, 이를 전력 비율로 바꾸려면 자기장과 진행 방향도 알아야 합니다. 또한 입사파와 반사파가 겹치는 경계에서는 두 파를 합친 전체 장으로 순유입 전력을 계산합니다.

이 장에서는 전력의 정의와 기준면을 먼저 정한 뒤, 반사율·투과율을 유도하고 두 매질의 경계면에 수직 입사하는 예로 확인합니다. 이어서 입사·반사파의 간섭을 포함한 순유입 전력과 쌓은 층 전체의 흡수율을 구합니다. **각 파의 전력을 따로 구하는 단계**와 **두 파를 합쳐 경계의 순전력을 구하는 단계**를 차례로 살펴봅니다.

**한쪽에서 입사하고, 위·아래 반무한 매질은 등방성이며, 횡파수 `kx`, `ky`는 실수인 경우**를 다룹니다. 양쪽에서 입사하거나 서로 다른 횡파수의 파를 합치는 경우는 [9장](multiple-beams.md)에서 설명합니다.

## 전기장과 자기장에서 시간평균 전력 구하기

### 전력 밀도 S와 전력 P

빛의 전기장과 자기장은 시간에 따라 진동합니다. 한 위치에서 시간 {math}`t`에 갖는 전기장과 자기장 값을 각각 {math}`\mathbf e(t)`, {math}`\mathbf h(t)`로 나타내겠습니다. 두 벡터의 외적을 **포인팅 벡터** {math}`\mathbf S(t)`라고 하며, 그 순간의 단위 면적당 전력 흐름과 방향을 나타냅니다.

```{math}
\mathbf S(t)=\mathbf e(t)\times\mathbf h(t).
```

한 주파수로 진동하는 장을 계산할 때는 **진동의 크기와 위상을 담은 복소 진폭**을 사용하면 편리합니다. 전기장과 자기장의 복소 진폭을 각각 {math}`\mathbf E`, {math}`\mathbf H`라 하면, 시간에 따라 변하는 실제 장은 다음과 같이 표현합니다. 여기서 {math}`\omega`는 빛의 각주파수입니다.

```{math}
\mathbf e(t)=\operatorname{Re}[\mathbf E e^{-i\omega t}],
\qquad
\mathbf h(t)=\operatorname{Re}[\mathbf H e^{-i\omega t}].
```

즉, 대문자 {math}`\mathbf E`, {math}`\mathbf H`에 시간에 따른 진동을 나타내는 {math}`e^{-i\omega t}`를 곱하고 실수부를 취하면, 소문자로 쓴 실제 장 {math}`\mathbf e(t)`, {math}`\mathbf h(t)`를 얻습니다. **매질이 무손실이어도 위상을 표현하기 위해 복소 진폭을 사용합니다.**

순간적인 전력 흐름 {math}`\mathbf S(t)`도 시간에 따라 변합니다. 반사율·투과율을 구할 때는 이를 한 주기 동안 평균한 값을 사용하며, 복소 진폭으로 다음과 같이 계산할 수 있습니다.

```{math}
\langle\mathbf S\rangle
=\frac12\operatorname{Re}[\mathbf E\times\mathbf H^*].
```

별표 {math}`{}^*`는 복소켤레, 꺾쇠괄호 {math}`\langle\;\rangle`는 시간평균입니다. 이 식의 자기장 {math}`\mathbf H`는 A/m 단위의 물리적 자기장입니다. Wavebit가 반환하는 필드 배열의 자기장 성분은 {math}`Z_0\mathbf H`이므로, 이 식에 직접 넣을 때는 진공 임피던스 {math}`Z_0`로 나눕니다. 배열의 단위와 변환은 {ref}`14장 <magnetic-field-units>`에서 설명합니다.

면을 통과하는 전력은 전력 밀도의 **면에 수직인 성분**을 적분해서 구합니다. 면의 법선 단위벡터를 {math}`\hat{\mathbf n}`이라 하면:

```{math}
P=\int_{\text{셀 단면}}
\langle\mathbf S\rangle\cdot\hat{\mathbf n}\,dA.
```

| 물리량 | 단위 | 의미 |
| --- | --- | --- |
| {math}`\langle\mathbf S\rangle` | W/m² | 단위 면적당 시간평균 전력 흐름과 그 방향 |
| {math}`P` | W | 단위 셀의 단면을 통과하는 전력 |
| {math}`R`, {math}`T` | 무차원 | 반사·투과 전력을 입사 전력으로 나눈 비율 |

장에 물리적 진폭을 부여했을 때 위 단위를 사용합니다. 효율만 구할 때는 같은 진폭 기준을 분자·분모에 적용하면 됩니다. 표면을 따라 전력 밀도가 일정한 파끼리 같은 면적에서 비교하면 면적이 약분되므로, 전력의 비는 법선 방향 전력 밀도의 비와 같습니다. 전력 밀도가 위치에 따라 달라지면 먼저 면에 걸쳐 적분합니다. 무한 주기 구조에서는 전체 면적 대신 **단위 셀 하나의 전력**을 사용합니다.

### E와 H의 위상차가 전력에 미치는 영향

전기장이 x 방향, 자기장이 y 방향인 단일 평면파를 예로 들겠습니다. 한 위치에서 두 장의 성분을 각각 {math}`e(t)`, {math}`h(t)`라고 쓰면, +z 방향의 순간 전력 밀도는 {math}`S_z(t)=e(t)h(t)`입니다.

이 위치의 전기장이 다음과 같이 진동하도록 시간의 기준을 잡겠습니다. {math}`E_0`는 전기장 진폭의 크기인 양의 실수입니다.

```{math}
e(t)=E_0\cos\omega t,
\qquad E=E_0.
```

자기장도 같은 주파수로 진동하지만, 전기장과 위상이 다를 수 있습니다. 여기서 {math}`E`, {math}`H`는 각각 전기장 x 성분과 자기장 y 성분의 **복소 진폭**입니다. 벡터 전체가 아니라 성분 하나를 나타내므로 각각 복소 스칼라이며, 두 성분의 비를 다음처럼 놓습니다.

```{math}
Y=\frac{H}{E}=|Y|e^{i\delta},
\qquad H=YE=|Y|E_0e^{i\delta}.
```

{math}`Y`는 선택한 전기장 성분과 자기장 성분을 연결하는 **파동 어드미턴스**입니다. {math}`|Y|=|H|/|E|`는 자기장과 전기장 진폭 크기의 비, {math}`\delta=\arg H-\arg E`는 두 성분의 위상차입니다. 이 비의 값은 매질·진행 방향·편광에 따라 달라지며, 맥스웰 방정식으로 구합니다. 위의 시간 규약 {math}`e^{-i\omega t}`를 적용하면 실제 자기장은:

```{math}
\begin{aligned}
h(t)
&=\operatorname{Re}[H e^{-i\omega t}]\\
&=\operatorname{Re}[|Y|E_0e^{i(\delta-\omega t)}]\\
&=|Y|E_0\cos(\delta-\omega t)\\
&=|Y|E_0\cos(\omega t-\delta).
\end{aligned}
```

```{figure} ../../figures/learn/learn/12-time-average.svg
:alt: E와 H의 위상차가 0도일 때와 45도일 때의 시간 신호 및 두 신호의 곱을 비교한다. 정규화한 곱의 시간평균은 각각 0.5와 약 0.354이다.
:figclass: learning-figure

그림 13-1. 전기장과 자기장의 위상차에 따른 시간평균. 각 신호를 자신의 진폭으로 나누어 표시했으며, 아래 그래프는 정규화한 두 신호의 곱입니다. 특정 구조의 계산 결과가 아닌 설명용 그래프입니다.
```

#### 두 신호를 곱하고 한 주기 평균하기

**먼저 두 장을 곱합니다.**

```{math}
e(t)h(t)=|Y|E_0^2\cos\omega t\cos(\omega t-\delta).
```

여기에 삼각함수의 곱을 합으로 바꾸는 식

```{math}
\cos\alpha\cos\beta
=\frac12\left[\cos(\alpha-\beta)+\cos(\alpha+\beta)\right]
```

을 사용합니다. {math}`\alpha=\omega t`, {math}`\beta=\omega t-\delta`를 대입하면:

```{math}
\begin{aligned}
e(t)h(t)
&=\frac{|Y|E_0^2}{2}
\left[\cos\bigl(\omega t-(\omega t-\delta)\bigr)
+\cos\bigl(\omega t+(\omega t-\delta)\bigr)\right]\\[4pt]
&=\frac{|Y|E_0^2}{2}
\left[\cos\delta+\cos(2\omega t-\delta)\right].
\end{aligned}
```

괄호 안의 첫째 항 {math}`\cos\delta`는 시간에 따라 변하지 않습니다. 둘째 항 {math}`\cos(2\omega t-\delta)`는 두 장의 각주파수 {math}`\omega`보다 두 배 빠른 {math}`2\omega`로 진동합니다.

**이제 한 주기 동안 평균합니다.** 전기장과 자기장의 한 주기를 {math}`\tau=2\pi/\omega`라고 하면, 시간평균은 한 주기 동안 적분한 값을 그 주기의 길이로 나눈 것입니다.

```{math}
\begin{aligned}
\langle e h\rangle
&=\frac1{\tau}\int_0^{\tau}e(t)h(t)\,dt\\[4pt]
&=\frac{|Y|E_0^2}{2}
\left[
\frac1{\tau}\int_0^{\tau}\cos\delta\,dt
+\frac1{\tau}\int_0^{\tau}\cos(2\omega t-\delta)\,dt
\right].
\end{aligned}
```

시간에 따라 변하지 않는 첫째 항의 평균은 그 값과 같습니다.

```{math}
\frac1{\tau}\int_0^{\tau}\cos\delta\,dt
=\frac1{\tau}[t\cos\delta]_0^{\tau}
=\frac{\tau\cos\delta}{\tau}
=\cos\delta.
```

둘째 항은 다음처럼 평균이 0이 됩니다.

```{math}
\begin{aligned}
\frac1{\tau}\int_0^{\tau}\cos(2\omega t-\delta)\,dt
&=\frac1{\tau}
\left[\frac{\sin(2\omega t-\delta)}{2\omega}\right]_0^{\tau}\\[4pt]
&=\frac{\sin(2\omega\tau-\delta)-\sin(-\delta)}{2\omega\tau}\\[4pt]
&=\frac{\sin(4\pi-\delta)-\sin(-\delta)}{4\pi}\\[4pt]
&=0.
\end{aligned}
```

마지막 줄에서는 {math}`\sin(4\pi-\delta)=\sin(-\delta)`를 사용했습니다. {math}`2\omega`로 진동하는 항은 시간 {math}`\tau` 동안 정확히 두 주기를 지나므로 양·음의 기여가 상쇄됩니다. 순간 전력에서 이 항이 없어지는 것이 아니라, **한 주기 평균에 기여하는 값이 0**입니다.

따라서 남는 평균 전력 밀도는:

```{math}
\langle S_z\rangle=\langle e h\rangle
=\frac{|Y|E_0^2}{2}[\cos\delta+0]
=\frac12|Y|E_0^2\cos\delta.
```

이를 {math}`Y`의 실수부로도 쓸 수 있습니다. 오일러 공식으로 {math}`Y`를 펼치면:

```{math}
Y=|Y|e^{i\delta}
=|Y|\cos\delta+i|Y|\sin\delta,
\qquad
\operatorname{Re}Y=|Y|\cos\delta.
```

그러므로:

```{math}
\boxed{\langle S_z\rangle
=\frac12|Y|E_0^2\cos\delta
=\frac12E_0^2\operatorname{Re}Y.}
```

자기장의 진폭 크기를 {math}`H_0=|Y|E_0`라고 쓰면 이 결과는 {math}`\langle S_z\rangle=\tfrac12 E_0H_0\cos\delta`입니다. 즉, **전기장과 자기장의 진폭 크기뿐 아니라 두 장의 위상차도 평균 전력을 정합니다.**

앞에서 소개한 복소 진폭의 식도 같은 결과를 줍니다. {math}`E=E_0`, {math}`H=YE_0`를 대입하면:

```{math}
\frac12\operatorname{Re}[EH^*]
=\frac12\operatorname{Re}[E_0(YE_0)^*]
=\frac12 E_0^2\operatorname{Re}[Y^*]
=\frac12 E_0^2\operatorname{Re}Y.
```

## 전력을 읽는 경계면과 입사 기준

입사 전력 {math}`P_{\mathrm{in}}`는 **입사 반무한 매질과 첫 번째 층이 만나는 경계면에서, 입사파만으로 계산한 전력**입니다. 아래쪽 입사에서는 빛이 아래에서 처음 만나는 층과의 경계가 이 기준면입니다. 반사파의 전력 {math}`P_r`도 같은 면에서 읽고, 투과파의 전력 {math}`P_t`는 반대쪽의 마지막 층과 출사 반무한 매질 사이에서 읽습니다. 이 장에서 아래첨자 in·r·t는 각각 입사파(또는 입사 매질)·반사파·투과파를 뜻하며, i는 허수 단위로만 씁니다.

각 파의 전력은 해당 파가 진행하는 쪽을 양수로 잡습니다. 전체 반사율·투과율은 다음과 같습니다.

```{math}
R_{\mathrm{total}}=\frac{P_r}{P_{\mathrm{in}}},
\qquad
T_{\mathrm{total}}=\frac{P_t}{P_{\mathrm{in}}}.
```

모든 전력을 {math}`P_{\mathrm{in}}`로 나누면 기준 입사 전력은 1이 됩니다. 이는 실제 광원의 전력을 1 W로 설정한다는 뜻이 아닙니다. 효율을 정의하려면 {math}`P_{\mathrm{in}}>0`이어야 하며, 법선 전력을 전달하지 않는 소멸파만을 입력한 경우에는 이 정규화를 사용할 수 없습니다.

입사 경계면에는 입사파와 반사파가 함께 있습니다. 이 경계면을 통해 층 안으로 들어가는 **순유입 전력**을 구하려면, 두 파의 **전기장 복소 진폭끼리, 자기장 복소 진폭끼리** 더합니다.

```{math}
\mathbf E=\mathbf E_{\mathrm{in}}+\mathbf E_r,
\qquad
\mathbf H=\mathbf H_{\mathrm{in}}+\mathbf H_r.
```

이렇게 합친 {math}`\mathbf E`, {math}`\mathbf H`로 {math}`\tfrac12\operatorname{Re}(\mathbf E\times\mathbf H^*)`를 계산하고, 층 안으로 향하는 법선 성분을 경계면의 단위 셀에 걸쳐 적분합니다. 그림 13-2의 {math}`P_{\mathrm{enter}}`는 이 순유입 전력을 {math}`P_{\mathrm{in}}`로 나눈 비율입니다. 두 파를 합친 장으로 계산하면 간섭항도 포함됩니다. 자세한 전개와 무손실·손실 매질의 차이는 {ref}`입사파와 반사파를 합쳐 순유입 전력 구하기 <incident-reflected-power>`에서 설명합니다.

```{figure} ../../figures/learn/learn/12-reference-planes.svg
:alt: 위쪽 입사에서 보라색 선으로 표시한 ② 입사 경계면은 top과 첫 번째 층 사이, ③ 출사 경계면은 마지막 층과 bot 사이이다. 입사 경계면의 순유입 전력에서 출사 경계면의 투과 전력을 빼면 쌓은 층 전체의 흡수 전력이다.
:figclass: learning-figure

그림 13-2. 위쪽 입사에서 전력을 읽는 경계면과 흡수를 구하는 범위. 그림의 {math}`P_{\mathrm{enter}}`, {math}`T`, {math}`A`는 입사파만의 전력 {math}`P_{\mathrm{in}}`로 나눈 값입니다.
```

흡수성 매질에서는 전파하면서 진폭이 줄어들므로 **어느 면에서 전력을 읽었는지**가 중요합니다. 입사파는 기준면에 도달했을 때의 전력을 사용하며, 그 전에 입사 매질에서 흡수된 전력은 포함하지 않습니다. 투과율도 출사 경계면을 막 통과하는 전력을 뜻하며, 이후 반무한 매질 안에서 흡수되는 양은 포함하지 않습니다.

순유입 전력은 쌓은 층 안으로 들어가는 방향을 양수로 잡습니다. Wavebit의 +z는 위쪽이므로, 위쪽 입사에서는 −z, 아래쪽 입사에서는 +z가 양의 순유입 방향입니다.

(amplitude-to-efficiency)=
## 복소 진폭 계수에서 반사율·투과율 구하기

먼저 입사파·반사파·투과파의 전력을 **각각 따로** 구합니다. 여기서는 단일 평면파가 입사하며, 각 출력 성분의 전기장 진폭은 같은 입사 기준 진폭으로 나눕니다. 결과 배열과 복소 계수의 크기·위상을 읽는 방법은 {ref}`12장 <diffraction-amplitude-coefficients>`에 있습니다.

(refractive-index-and-angle)=
### 무손실 매질에서 굴절률과 각도가 들어가는 이유

아래의 전력 밀도 식, 투과율 유도, 수직 입사 예제와 차수별 효율 식은 **입사·출사 반무한 매질이 무손실·등방성·비자성이며, 굴절률이 양의 실수인 경우**를 다룹니다. 각도와 전력 식은 그 매질에서 전파하는 파에 적용합니다. 손실 매질은 이어지는 {ref}`입사·출사 매질에 손실이 있을 때 <lossy-amplitude-to-efficiency>`에서 따로 설명합니다.

굴절률이 {math}`n`인 매질에서 파의 진행 방향을 {math}`\hat{\mathbf k}`라고 하면:

```{math}
\mathbf H=\frac{n}{Z_0}\hat{\mathbf k}\times\mathbf E,
\qquad
\langle\mathbf S\rangle
=\frac{n}{2Z_0}|\mathbf E|^2\hat{\mathbf k}.
```

{math}`Z_0\approx376.73\,\Omega`는 진공 임피던스이며, {math}`|\mathbf E|^2=\mathbf E\cdot\mathbf E^*`입니다. 같은 전기장 크기라도 굴절률에 따라 자기장 크기가 달라집니다.

첫 식의 {math}`\times`는 외적입니다. 진행 방향 {math}`\hat{\mathbf k}`와 전기장 {math}`\mathbf E`의 순서로 외적하면 자기장 방향을 얻습니다. 실제 장의 전력 흐름은 {math}`\mathbf e(t)\times\mathbf h(t)`로 구하므로 **전기장이 먼저**입니다. 두 순서는 오른손 법칙의 같은 순환 관계를 나타내며, 외적의 순서를 뒤집으면 결과의 방향도 반대가 됩니다.

진행 방향과 **파가 통과하는 쪽의 표면 법선** 사이 각도를 {math}`\theta`라 하면, 표면에 수직인 시간평균 전력 밀도는:

```{math}
I_\perp=\frac{n\cos\theta}{2Z_0}|\mathbf E|^2.
```

굴절률 {math}`n`은 전기장과 자기장의 관계에서, {math}`\cos\theta`는 전력 흐름의 법선 성분을 구하는 과정에서 나옵니다.

```{figure} ../../figures/learn/learn/13-wave-directions.svg
:alt: 왼쪽은 위로 진행하는 선형 편광 평면파로, 한 시각의 전기장은 +x, 자기장은 화면 안쪽인 +y를 향한다. 오른쪽은 top, 쌓은 층들, bot을 함께 그린 xz 단면이다. bot에서 위로 입사하는 파가 처음 만나는 층은 구조체의 맨 아래층이며, 둘 사이의 선이 xy 평면인 입사 경계면이다. bot 안의 초록색 입사 전력 흐름과 보라색 법선 성분 사이 각도는 theta이고, 법선 성분은 전력 흐름 크기의 cos theta 배이다.
:figclass: learning-figure
:name: wave-directions-and-normal-power

그림 13-3. 전기장·자기장 방향과 아래쪽 입사의 전력 흐름. 오른쪽은 bot과 맨 아래층 사이의 입사 경계면을 표시한 xz 단면입니다.
```

그림 왼쪽은 Wavebit의 **아래쪽 입사**를 예로 듭니다. ⊗는 화면 안쪽을 향하는 벡터이며, 선형 편광의 한 시각을 나타냅니다. 전기장이 +x, 자기장이 +y를 향하면 전력은 +z로 흐릅니다. 위쪽 입사에서도 외적의 순서는 같으며, 전기장을 같은 +x 방향으로 잡으면 자기장이 −y를 향하여 전력은 −z로 흐릅니다.

오른쪽의 **맨 아래층**은 두께를 지정해 쌓은 층 중 가장 아래에 놓인 층입니다. 그 아래에는 반무한 매질 `bot`이 있습니다. 아래쪽에서 입사하면 빛은 `bot`에서 맨 아래층을 먼저 만나며, **`bot`과 맨 아래층 사이가 입사 경계면**입니다. 이 면은 x·y 방향으로 펼쳐져 있지만, xz 단면으로 옆에서 보았기 때문에 가로선으로 보입니다. 구조 안쪽을 향하는 법선은 +z 방향입니다.

초록색 화살표는 `bot` 안의 **입사파만의 시간평균 전력 흐름**이고, 보라색 화살표는 그 전력 흐름의 **경계면에 수직인 성분**입니다. θ는 두 방향 사이 각도입니다. 이 그림에서는 입사파의 전력을 읽는 방법을 보여 주며, 반사파와 층 내부의 장은 생략했습니다.

(transmission-amplitude-to-power)=
### 전기장 진폭의 비에서 투과율 구하기

앞에서는 **한 평면파가 경계면을 통과하는 전력 밀도**를 구했습니다. 이제 그 식을 입사파와, 구조를 지나 반대쪽으로 나온 투과파에 각각 적용하겠습니다. 두 전력을 비교하면 입사 전력을 기준으로 얼마나 많은 전력이 투과했는지, 즉 **투과율**을 알 수 있습니다.

먼저 입사파와 투과파가 각각 하나이고, 각 파에 편광 성분이 하나인 경우를 보겠습니다. 두 파의 값을 다음처럼 구분합니다. 아래첨자 in은 입사파, t는 투과파를 뜻하며, out은 출사 매질을 뜻합니다.

| 물리량 | 입사파 | 투과파 |
| --- | --- | --- |
| 파가 있는 매질의 굴절률 | {math}`n_{\mathrm{in}}` | {math}`n_{\mathrm{out}}` |
| 진행 방향과 통과하는 쪽의 표면 법선 사이 각도 | {math}`\theta_{\mathrm{in}}` | {math}`\theta_t` |
| 전기장의 복소 진폭 | {math}`E_{\mathrm{in}}` | {math}`E_t` |
| 표면에 수직인 시간평균 전력 밀도 | {math}`I_{\perp,\mathrm{in}}` | {math}`I_{\perp,t}` |

**진폭 투과계수** {math}`t`는 투과파의 전기장 복소 진폭을 입사파의 전기장 복소 진폭으로 나눈 값입니다.

```{math}
t=\frac{E_t}{E_{\mathrm{in}}},
\qquad
|t|^2=\frac{|E_t|^2}{|E_{\mathrm{in}}|^2}.
```

RCWA는 경계조건을 만족하는 전기장·자기장을 풀어 이 진폭 계수를 구합니다. 여기에 앞의 전력 밀도 식을 적용하면 **전력의 비인 투과율** {math}`T`를 얻습니다.

투과율은 투과 전력을 입사 전력으로 나눈 값입니다. 평면파의 세기는 경계면을 따라 균일하므로, 입사·출사 경계면에서 같은 단위 셀 면적을 통과하는 전력을 비교하면 면적이 약분됩니다. 따라서 법선 전력 밀도의 비로도 쓸 수 있습니다.

```{math}
T=\frac{P_t}{P_{\mathrm{in}}}
 =\frac{I_{\perp,t}}{I_{\perp,\mathrm{in}}}.
```

입사 전력은 입사파만으로 구합니다. 앞에서 얻은 {math}`I_\perp=n\cos\theta\,|E|^2/(2Z_0)`에 **각 파의 굴절률·각도·전기장 진폭**을 넣으면:

```{math}
\begin{aligned}
I_{\perp,\mathrm{in}}
 &=\frac{n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}{2Z_0}
   |E_{\mathrm{in}}|^2,\\[5pt]
I_{\perp,t}
 &=\frac{n_{\mathrm{out}}\cos\theta_t}{2Z_0}|E_t|^2.
\end{aligned}
```

이 두 식을 투과율의 분자와 분모에 대입합니다. 두 전력 밀도에 공통으로 들어 있는 {math}`1/(2Z_0)`가 약분되어:

```{math}
\begin{aligned}
T
&=\frac{\dfrac{n_{\mathrm{out}}\cos\theta_t}{2Z_0}|E_t|^2}
        {\dfrac{n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}{2Z_0}|E_{\mathrm{in}}|^2}\\[4pt]
&=\frac{n_{\mathrm{out}}\cos\theta_t}
        {n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}
  \frac{|E_t|^2}{|E_{\mathrm{in}}|^2}.
\end{aligned}
```

마지막 진폭 제곱의 비를 위에서 정의한 {math}`|t|^2`로 바꾸면:

```{math}
\boxed{T=\frac{n_{\mathrm{out}}\cos\theta_t}
              {n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}|t|^2.}
```

여기서 {math}`|t|^2`는 **전기장 크기 제곱의 비**입니다. 같은 전기장 크기라도 굴절률에 따라 자기장 크기가 달라지고, 진행 각도에 따라 경계면에 수직인 전력 성분이 달라집니다. 그래서 **전력의 비**를 얻으려면 굴절률·코사인의 비까지 곱해야 합니다. 입사·출사 매질의 굴절률과 두 각도가 각각 같으면 이 인자가 1이 되어 {math}`T=|t|^2`가 됩니다.

(normal-incidence-interface-example)=
### 예제: 굴절률 1에서 1.5로 수직 입사

굴절률이 1인 매질과 1.5인 매질이 평평한 경계면에서 바로 맞닿아 있다고 합시다. 중간에 다른 층은 없으며, 빛은 굴절률 1인 쪽에서 1.5인 쪽으로 경계면에 수직으로 들어옵니다. 두 매질은 모두 무손실·등방성·비자성으로 가정합니다.

#### 경계조건으로 진폭 투과계수 구하기

프레넬 식은 두 매질의 평평한 경계에서 반사·투과파의 전기장 복소 진폭을 구하는 식입니다. **먼저 경계조건을 풀어 진폭 투과계수를 구하고, 그다음 전력의 비로 환산합니다.**

경계면에서 입사파·반사파·투과파의 전기장 복소 진폭을 각각 {math}`E_{\mathrm{in}}`, {math}`E_r`, {math}`E_t`라고 하겠습니다. 세 값은 모두 **경계면에 평행한 같은 방향의 전기장 성분**으로 읽습니다. 진폭 투과계수는 다음 비로 정의합니다.

```{math}
t=\frac{E_t}{E_{\mathrm{in}}}.
```

이것은 {math}`t`의 **정의**입니다. 이제 경계조건으로 이 비의 값을 구하겠습니다. 그림 13-4는 이 절에서 사용하는 세 파와 두 경계조건, 그리고 두 조건을 풀어 얻는 결과를 한 번에 보여 줍니다.

```{figure} ../../figures/learn/learn/13-fresnel-interface.svg
:alt: 왼쪽은 입사 매질과 출사 매질이 만나는 경계면이다. 입사파와 투과파는 아래로, 반사파는 위로 진행하며, 각 파에 전기장과 자기장 진폭이 표시되어 있다. 오른쪽은 경계면에서 연결하는 두 조건으로, 전기장 연속 E_in + E_r = E_t와 자기장 연속 n_in(E_in − E_r) = n_out E_t이다. 반사파는 진행 방향이 반대이므로 H_r = −(n_in / Z_0) E_r이며, 두 조건을 함께 풀면 t = E_t / E_in = 2 n_in / (n_in + n_out)이다.
:figclass: learning-figure

그림 13-4. 수직 입사의 프레넬 경계조건. 입사파와 반사파를 합친 접선 전기장·자기장이 각각 투과파의 값과 같아야 합니다. 반사파는 진행 방향이 반대라서 자기장 변환식에 마이너스가 붙습니다.
```

**전기장의 경계조건:** 수직 입사에서는 전기장이 경계면에 평행합니다. 경계면 양쪽에서 전체 접선 전기장이 같아야 하므로, 입사 쪽의 입사파와 반사파를 합한 값이 출사 쪽 투과파의 값과 같습니다.

```{math}
E_{\mathrm{in}}+E_r=E_t.
```

**자기장의 경계조건:** 앞에서 구한 평면파의 관계 {math}`\mathbf H=(n/Z_0)\hat{\mathbf k}\times\mathbf E`를 사용합니다. 자기장도 세 파 모두 같은 방향의 성분으로 읽되, 입사파의 자기장 방향을 양수로 잡겠습니다. 반사파는 진행 방향이 반대이므로, 전기장에서 자기장을 구하는 식에 마이너스 부호가 붙습니다.

```{math}
H_{\mathrm{in}}=\frac{n_{\mathrm{in}}}{Z_0}E_{\mathrm{in}},
\qquad
H_r=-\frac{n_{\mathrm{in}}}{Z_0}E_r,
\qquad
H_t=\frac{n_{\mathrm{out}}}{Z_0}E_t.
```

별도의 표면 전류가 없는 이 경계에서는 전체 접선 자기장도 양쪽에서 같아야 합니다. 따라서:

```{math}
\begin{aligned}
H_{\mathrm{in}}+H_r&=H_t,\\[4pt]
\frac{n_{\mathrm{in}}}{Z_0}E_{\mathrm{in}}
-\frac{n_{\mathrm{in}}}{Z_0}E_r
&=\frac{n_{\mathrm{out}}}{Z_0}E_t,\\[4pt]
n_{\mathrm{in}}(E_{\mathrm{in}}-E_r)&=n_{\mathrm{out}}E_t.
\end{aligned}
```

마지막 줄은 양변에 {math}`Z_0`를 곱한 것입니다. 이제 전기장 조건에서 얻은 {math}`E_r=E_t-E_{\mathrm{in}}`을 대입하여 반사파의 진폭을 제거하면:

```{math}
\begin{aligned}
n_{\mathrm{in}}\bigl[E_{\mathrm{in}}-(E_t-E_{\mathrm{in}})\bigr]
 &=n_{\mathrm{out}}E_t,\\[4pt]
2n_{\mathrm{in}}E_{\mathrm{in}}-n_{\mathrm{in}}E_t
 &=n_{\mathrm{out}}E_t,\\[4pt]
2n_{\mathrm{in}}E_{\mathrm{in}}
 &=(n_{\mathrm{in}}+n_{\mathrm{out}})E_t.
\end{aligned}
```

양변을 {math}`(n_{\mathrm{in}}+n_{\mathrm{out}})E_{\mathrm{in}}`으로 나누면, 수직 입사의 진폭 투과계수를 얻습니다.

```{math}
\boxed{t=\frac{E_t}{E_{\mathrm{in}}}
       =\frac{2n_{\mathrm{in}}}{n_{\mathrm{in}}+n_{\mathrm{out}}}.}
```

즉, 첫 번째 등호는 진폭 투과계수의 정의이고, 두 번째 등호는 전기장과 자기장의 경계조건을 풀어 얻은 결과입니다.

#### 굴절률을 대입하고 투과율로 환산하기

이 프레넬 식에 {math}`n_{\mathrm{in}}=1`, {math}`n_{\mathrm{out}}=1.5`를 대입합니다.

```{math}
t=\frac{2n_{\mathrm{in}}}{n_{\mathrm{in}}+n_{\mathrm{out}}}
 =\frac{2\times1}{1+1.5}
 =0.8.
```

프레넬 식에 두 굴절률을 대입해 얻은 값이 {math}`t=0.8`입니다. 즉, 투과파의 전기장 진폭 크기는 입사파의 80%입니다.

**다음으로 이 값을 전력의 비인 투과율로 환산합니다.** 앞 절에서 구한 투과율 식에 {math}`t=0.8`을 대입합니다. 수직 입사이므로 입사각과 투과각은 모두 0°이고, 두 코사인은 모두 1입니다.

```{math}
T=\frac{n_{\mathrm{out}}\cos\theta_t}
        {n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}|t|^2
 =\frac{1.5\times1}{1\times1}|0.8|^2
 =0.96.
```

따라서 전기장 진폭 크기는 입사파의 80%지만, 투과 전력은 입사 전력의 96%입니다. 두 매질의 굴절률이 다르므로 전기장 크기 제곱의 비인 {math}`|t|^2=0.64`를 그대로 투과율로 읽을 수 없습니다. 이 예에서는 흡수가 없으므로 입사 전력의 나머지 4%는 반사됩니다.

### 각 회절차수의 s·p 성분

{math}`(m,n)`차에 s·p 성분이 모두 있으면 서로 직교하는 두 편광 성분의 전력을 더합니다.

```{math}
\begin{aligned}
T_{mn}
&=\frac{n_{\mathrm{out}}\cos\theta_{t,mn}}
       {n_{\mathrm{in}}\cos\theta_{\mathrm{in}}}
\left(|t_s^{(m,n)}|^2+|t_p^{(m,n)}|^2\right),\\[5pt]
R_{mn}
&=\frac{\cos\theta_{r,mn}}{\cos\theta_{\mathrm{in}}}
\left(|r_s^{(m,n)}|^2+|r_p^{(m,n)}|^2\right).
\end{aligned}
```

반사파는 입사파와 같은 매질에 있어 굴절률이 약분됩니다. 그러나 반사 차수의 방향은 입사 방향과 다를 수 있으므로 코사인 비는 남습니다. {math}`\theta_{r,mn}`, {math}`\theta_{t,mn}`은 각 출력 파와 **구조에서 나가는 쪽의 법선** 사이 각도입니다.

차수별 반사율·투과율을 더하면 {math}`R_{\mathrm{total}}`, {math}`T_{\mathrm{total}}`이 됩니다. 실수 횡파수를 사용하는 한 계산 묶음에서는 서로 다른 출력 차수 사이의 교차항이 단위 셀 적분에서 상쇄됩니다. 무손실 출사 매질의 소멸파처럼 법선 전력이 0인 차수는 [12장](results.md)에서 설명한 대로 읽습니다.

(lossy-amplitude-to-efficiency)=
### 입사·출사 매질에 손실이 있을 때

손실 매질에서도 전력은 {math}`\tfrac12\operatorname{Re}(\mathbf E\times\mathbf H^*)`로 구합니다. 복소 재료값으로 각 파의 전기장과 자기장을 구한 뒤, **입사 전력과 출력 전력을 각각 계산해서 나눕니다.** 무손실 식의 굴절률·코사인 비에 복소수를 넣어 비를 먼저 만든 뒤 실수부를 취하면 다른 값이 됩니다. 실수부는 각 파의 전력을 구할 때 따로 취하며, 그 형태도 편광마다 다릅니다. 아래 s 편광에서는 각 파 어드미턴스의 실수부 {math}`\operatorname{Re}Y`가 전력 인자입니다. p 편광은 전기장 진폭을 기준으로 할 때 {math}`\operatorname{Re}(\kappa/\varepsilon_r)`에 {math}`|n/\mu_r|^2`를 곱한 값이 전력 인자이며, {math}`\kappa`는 법선 파수를 진공 파수로 나눈 값입니다. 두 인자의 출처는 이 장 끝의 '코드에서 사용하는 s·p 경계 진폭'에서 설명합니다.

#### 접선 성분과 어드미턴스 정의하기

선형·등방성인 두 매질이 만나는 **회절 없는 평면 경계의 s 편광**을 예로 들겠습니다. 위쪽 매질에서 아래쪽 매질로 입사하고, 입사면은 xz 평면입니다. 세 파 모두 전기장은 y 성분, 자기장은 접선 성분인 x 성분을 같은 좌표 방향으로 읽습니다.

경계면에서 각 파의 성분을 {math}`E_j=E_{y,j}`, {math}`H_j=H_{x,j}`라고 쓰겠습니다. 아래첨자 {math}`j=\mathrm{in},r,t`는 각각 입사파·반사파·투과파를 뜻합니다. **이 값들은 벡터 전체가 아니라 성분 하나의 복소 진폭**입니다. s 편광의 전기장은 {math}`\mathbf E_j=E_j\hat{\mathbf y}`이고, 여기서 {math}`E_j`는 복소 스칼라입니다.

진폭 반사계수 {math}`r`와 진폭 투과계수 {math}`t`는 다음 비로 정의합니다.

```{math}
r=\frac{E_r}{E_{\mathrm{in}}},
\qquad
t=\frac{E_t}{E_{\mathrm{in}}}.
```

이 정의에 따라 {math}`E_r=rE_{\mathrm{in}}`, {math}`E_t=tE_{\mathrm{in}}`입니다. 이제 각 파의 전력을 구해 입사 전력으로 나누겠습니다.

각 파의 **파동 어드미턴스**는 접선 자기장과 접선 전기장의 비로 정의합니다.

```{math}
Y_{\mathrm{in}}=\frac{H_{\mathrm{in}}}{E_{\mathrm{in}}},\qquad
Y_r=\frac{H_r}{E_r},\qquad
Y_t=\frac{H_t}{E_t}.
```

이 평면 경계에서 입사파와 반사파는 같은 횡파수를 갖고, 법선 파수의 부호가 반대입니다. 이 반사 조건을 맥스웰 방정식에 적용하면 다음 관계를 얻습니다.

```{math}
Y_r=-Y_{\mathrm{in}},
\qquad
H_{\mathrm{in}}=Y_{\mathrm{in}}E_{\mathrm{in}},\quad H_r=-Y_{\mathrm{in}}E_r,\quad H_t=Y_tE_t.
```

이후 반사파의 전력을 구할 때는 **같은 좌표 성분으로 읽은 관계** {math}`Y_r=-Y_{\mathrm{in}}`를 대입해 {math}`Y_{\mathrm{in}}`로 표현합니다. 부호가 바뀌는 근거는 아래에서 확인할 수 있습니다.

::::{dropdown} 유도: 반사파의 어드미턴스 부호가 바뀌는 이유

**1. 평면 경계에서는 주파수와 횡파수가 보존됩니다.**

시간에 따라 변하지 않고 x·y 방향으로 균일한 경계에서는 입사파와 반사파가 같은 주파수와 같은 횡파수를 갖습니다. 경계면의 모든 위치와 시간에서 경계조건을 만족해야 하기 때문입니다. 지금 선택한 xz 입사면에서는 {math}`k_y=0`이며:

```{math}
\omega_r=\omega_{\mathrm{in}}=\omega,
\qquad
k_{x,r}=k_{x,\mathrm{in}}=k_x.
```

입사파와 반사파는 모두 위쪽 매질에 있으므로 그 매질의 유전율 {math}`\varepsilon`과 투자율 {math}`\mu`도 같습니다. 여기서는 상대값이 아니라 {math}`\varepsilon=\varepsilon_0\varepsilon_r`, {math}`\mu=\mu_0\mu_r`를 씁니다.

**2. 분산관계에서 법선 파수를 구합니다.**

등방성 매질의 평면파가 만족하는 분산관계는:

```{math}
k_x^2+k_z^2=\omega^2\varepsilon\mu,
\qquad
k_z^2=\omega^2\varepsilon\mu-k_x^2.
```

입사파와 반사파는 오른쪽의 모든 값이 같으므로 {math}`k_{z,r}^2=k_{z,\mathrm{in}}^2`입니다. 같은 제곱값을 갖는 두 근 중 경계로 들어오는 해를 입사파, 경계에서 나가는 해를 반사파로 선택하면:

```{math}
k_{z,r}=-k_{z,\mathrm{in}}.
```

따라서 **법선 성분의 부호만 바뀌고, 접선 성분은 유지**됩니다. 비스듬한 입사에서 반사파가 입사파의 경로를 그대로 거슬러 간다는 뜻은 아닙니다. 손실 매질에서는 {math}`k_z`가 복소수일 수 있으며, 이때도 두 근의 부호가 반대라는 관계가 성립합니다. 위 분산관계의 {math}`k_z^2`는 복소수의 제곱으로, {math}`|k_z|^2`와 구별합니다.

**3. 맥스웰 방정식으로 자기장과 전기장의 비를 구합니다.**

시간 규약 {math}`e^{-i\omega t}`에서 패러데이 법칙은 다음과 같습니다.

```{math}
\nabla\times\mathbf E=i\omega\mu\mathbf H.
```

왼쪽의 {math}`\nabla\times`는 **회전 연산**(curl)으로, 전기장 성분들을 위치에 대해 미분하여 계산합니다. 평면파의 전기장을 넣어 이 연산을 차례로 구하겠습니다.

**평면파의 위치 의존성.** 균일한 매질 안에서 평면파의 복소 전기장은 다음처럼 쓸 수 있습니다.

```{math}
\mathbf E(x,y,z)
=\mathbf E_0e^{i(k_xx+k_yy+k_zz)}.
```

{math}`\mathbf E_0`는 위치에 무관한 복소 진폭 벡터이고, {math}`\mathbf k=(k_x,k_y,k_z)`는 파수 벡터입니다. 지수함수는 위치에 따른 위상 변화와, 파수가 복소수인 경우의 진폭 변화를 나타냅니다. 시간에 따른 진동을 나타내는 {math}`e^{-i\omega t}`는 이 복소 전기장 표현에서 분리해 두었습니다.

**각 좌표로 미분하기.** 먼저 x로 미분합니다. {math}`\mathbf E_0`는 상수이고, 지수의 x 미분은 {math}`ik_x`이므로:

```{math}
\begin{aligned}
\frac{\partial\mathbf E}{\partial x}
&=\mathbf E_0\frac{\partial}{\partial x}
  e^{i(k_xx+k_yy+k_zz)}\\[5pt]
&=ik_x\mathbf E_0e^{i(k_xx+k_yy+k_zz)}\\[5pt]
&=ik_x\mathbf E.
\end{aligned}
```

y와 z로 미분해도 같은 방식으로 각 파수 성분이 곱해집니다.

```{math}
\frac{\partial\mathbf E}{\partial y}=ik_y\mathbf E,
\qquad
\frac{\partial\mathbf E}{\partial z}=ik_z\mathbf E.
```

벡터를 미분한다는 것은 각 성분을 미분한다는 뜻입니다. 예를 들어 {math}`\partial E_z/\partial y=ik_yE_z`, {math}`\partial E_y/\partial z=ik_zE_y`입니다. 손실 매질에서 파수가 복소수여도 지수함수의 미분 방법은 같습니다.

**회전 연산 계산하기.** 회전 연산을 성분으로 쓰면 다음과 같습니다.

```{math}
\nabla\times\mathbf E=
\begin{pmatrix}
\dfrac{\partial E_z}{\partial y}-\dfrac{\partial E_y}{\partial z}\\[8pt]
\dfrac{\partial E_x}{\partial z}-\dfrac{\partial E_z}{\partial x}\\[8pt]
\dfrac{\partial E_y}{\partial x}-\dfrac{\partial E_x}{\partial y}
\end{pmatrix}.
```

각 미분에 위에서 구한 평면파의 미분 결과를 넣고, 공통 인자 {math}`i`를 묶습니다.

```{math}
\begin{aligned}
\nabla\times\mathbf E
&=\begin{pmatrix}
ik_yE_z-ik_zE_y\\[3pt]
ik_zE_x-ik_xE_z\\[3pt]
ik_xE_y-ik_yE_x
\end{pmatrix}\\[8pt]
&=i\begin{pmatrix}
k_yE_z-k_zE_y\\[3pt]
k_zE_x-k_xE_z\\[3pt]
k_xE_y-k_yE_x
\end{pmatrix}\\[8pt]
&=i\mathbf k\times\mathbf E.
\end{aligned}
```

마지막 괄호 안의 세 성분은 외적 {math}`\mathbf k\times\mathbf E`의 성분입니다. 따라서 이 평면파의 회전 연산은 **파수 벡터와 전기장의 외적에 i를 곱한 값**으로 표현됩니다.

**패러데이 법칙에 대입하기.** 이제 {math}`\nabla\times\mathbf E=i\mathbf k\times\mathbf E`를 처음 식의 왼쪽에 넣습니다.

```{math}
i\mathbf k\times\mathbf E=i\omega\mu\mathbf H.
```

양변을 {math}`i`로 나누고 자기장에 대해 정리하면:

```{math}
\mathbf k\times\mathbf E=\omega\mu\mathbf H,
\qquad
\mathbf H=\frac{\mathbf k\times\mathbf E}{\omega\mu}.
```

**s 편광의 접선 성분 구하기.** 이 예에서는 {math}`\mathbf E=(0,E_y,0)`, {math}`\mathbf k=(k_x,0,k_z)`이므로 외적을 계산하면:

```{math}
\mathbf k\times\mathbf E=(-k_zE_y,\,0,\,k_xE_y),
\qquad
H_x=-\frac{k_z}{\omega\mu}E_y.
```

따라서 지금 선택한 접선 성분의 어드미턴스는:

```{math}
Y=\frac{H_x}{E_y}=-\frac{k_z}{\omega\mu}.
```

입사파와 반사파에 각각 적용하고 {math}`k_{z,r}=-k_{z,\mathrm{in}}`를 대입하면:

```{math}
Y_{\mathrm{in}}=-\frac{k_{z,\mathrm{in}}}{\omega\mu},
\qquad
Y_r=-\frac{k_{z,r}}{\omega\mu}
   =\frac{k_{z,\mathrm{in}}}{\omega\mu}
   =-Y_{\mathrm{in}}.
```

위 관계는 같은 매질에서 같은 횡파수를 갖는 입사파와 반사파 한 쌍에 적용합니다. 격자의 서로 다른 회절차수는 횡파수가 다르므로, 각 차수의 법선 파수와 어드미턴스를 따로 구합니다.

::::

#### 각 파의 전력 밀도 구하기

먼저 입사파 하나의 전력을 구합니다. 지금의 성분 선택에서는 {math}`(\mathbf E\times\mathbf H^*)_z=-E_yH_x^*`입니다. 따라서 **입사 방향인 −z 쪽**의 법선 전력 밀도는 {math}`\tfrac12\operatorname{Re}(E_{\mathrm{in}}H_{\mathrm{in}}^*)`이고, 여기에 {math}`H_{\mathrm{in}}=Y_{\mathrm{in}}E_{\mathrm{in}}`를 대입하면:

```{math}
\begin{aligned}
I_{\mathrm{in}}
&=\frac12\operatorname{Re}(E_{\mathrm{in}}H_{\mathrm{in}}^*)\\[4pt]
&=\frac12\operatorname{Re}\!\left[E_{\mathrm{in}}(Y_{\mathrm{in}}E_{\mathrm{in}})^*\right]\\[4pt]
&=\frac12\operatorname{Re}\!\left[Y_{\mathrm{in}}^*E_{\mathrm{in}}E_{\mathrm{in}}^*\right]\\[4pt]
&=\frac12\operatorname{Re}(Y_{\mathrm{in}}^*)|E_{\mathrm{in}}|^2\\[4pt]
&=\frac12\operatorname{Re}(Y_{\mathrm{in}})|E_{\mathrm{in}}|^2.
\end{aligned}
```

여기서 {math}`E_{\mathrm{in}}`, {math}`Y_{\mathrm{in}}`, {math}`E_{\mathrm{in}}^*`는 모두 복소 스칼라이므로 곱셈 순서를 바꿀 수 있습니다. 벡터 외적의 순서를 바꾼 것이 아닙니다. 또한 {math}`E_{\mathrm{in}}E_{\mathrm{in}}^*=|E_{\mathrm{in}}|^2`는 실수이므로 실수부 연산 밖으로 꺼낼 수 있고, 복소켤레를 취해도 실수부는 바뀌지 않아 {math}`\operatorname{Re}(Y_{\mathrm{in}}^*)=\operatorname{Re}(Y_{\mathrm{in}})`입니다.

반사파의 {math}`I_r`는 **입사 매질로 돌아가는 +z 쪽**을 양수로 읽습니다. 법선 방향이 입사파와 반대이므로 전력 식에 마이너스가 붙습니다. 여기에 {math}`H_r=Y_rE_r`와 {math}`Y_r=-Y_{\mathrm{in}}`를 차례로 넣으면:

```{math}
\begin{aligned}
I_r
&=-\frac12\operatorname{Re}(E_rH_r^*)\\[4pt]
&=-\frac12\operatorname{Re}(Y_r)|E_r|^2\\[4pt]
&=\frac12\operatorname{Re}(Y_{\mathrm{in}})|E_r|^2.
\end{aligned}
```

즉, **전력을 읽는 법선 방향의 부호**와 **어드미턴스의 부호**가 함께 바뀌어 마지막 식에는 마이너스가 남지 않습니다. 투과파의 전력은 입사파처럼 −z 쪽 법선을 기준으로 읽고, 출사 매질의 {math}`Y_t`를 사용합니다. 세 파가 각각 경계면을 통과하는 쪽을 양수로 읽은 결과는:

```{math}
\begin{aligned}
I_{\mathrm{in}}&=\frac12\operatorname{Re}(Y_{\mathrm{in}})|E_{\mathrm{in}}|^2,\\[3pt]
I_r&=\frac12\operatorname{Re}(Y_{\mathrm{in}})|E_r|^2,\\[3pt]
I_t&=\frac12\operatorname{Re}(Y_t)|E_t|^2.
\end{aligned}
```

#### 입사 전력으로 나누어 반사율·투과율 구하기

같은 경계면 면적을 통과하는 전력을 비교하면 면적이 약분되므로, 반사율·투과율은 이 전력 밀도의 비로 구할 수 있습니다. 여기에 위에서 정의한 {math}`r`, {math}`t`를 사용하면:

```{math}
\begin{aligned}
R&=\frac{I_r}{I_{\mathrm{in}}}
  =\frac{|E_r|^2}{|E_{\mathrm{in}}|^2}
  =|r|^2,\\[6pt]
T&=\frac{I_t}{I_{\mathrm{in}}}
  =\frac{\operatorname{Re}(Y_t)}{\operatorname{Re}(Y_{\mathrm{in}})}
   \frac{|E_t|^2}{|E_{\mathrm{in}}|^2}\\[4pt]
 &=\frac{\operatorname{Re}(Y_t)}{\operatorname{Re}(Y_{\mathrm{in}})}|t|^2.
\end{aligned}
```

분모의 입사 전력은 양수라고 가정합니다. 일반적으로 {math}`\operatorname{Re}(Y_t/Y_{\mathrm{in}})`와 {math}`\operatorname{Re}(Y_t)/\operatorname{Re}(Y_{\mathrm{in}})`는 다르므로, **비를 만든 뒤 실수부를 취하는 것과 각 전력을 구한 뒤 나누는 것을 구별해야 합니다.** 다른 차수와 편광에서도 해당 파의 전기장·자기장 관계를 사용합니다.

이렇게 얻은 반사율은 반사파만의 전력 비입니다. 다음 절에서는 입사파와 반사파가 동시에 있는 경계의 순전력이 왜 단순히 {math}`1-R`과 같지 않을 수 있는지 살펴봅니다.

(incident-reflected-power)=
## 입사파와 반사파를 합쳐 순유입 전력 구하기

앞 절에서는 입사파·반사파·투과파의 전력을 각각 구했습니다. 이제 입사 경계면에 함께 존재하는 **입사파와 반사파를 먼저 합친 뒤**, 그 면을 통과하는 순전력을 구합니다. 두 파의 어드미턴스 부호 관계는 그대로 사용하지만, 전력을 계산하는 대상이 각 파에서 전체 장으로 바뀝니다.

### 두 파의 전기장과 자기장

등방성 매질의 수직입사를 예로 들겠습니다. 부호를 명확히 하기 위해 **아래쪽 입사**를 택합니다. 입사파는 +z, 반사파는 −z로 진행하며, +z가 쌓은 층 안으로 들어가는 방향입니다.

기준면의 한 위치에서 입사·반사 전기장의 x 성분에 대한 복소 진폭을 각각 {math}`A_{\mathrm{in}}`, {math}`A_r`라고 쓰겠습니다. 이 위치의 전체 전기장 x 성분과 자기장 y 성분은 다음과 같습니다.

```{math}
E_x=A_{\mathrm{in}}+A_r,
\qquad
H_y=Y(A_{\mathrm{in}}-A_r).
```

여기서 {math}`Y=1/Z`는 **입사파를 기준으로 한 어드미턴스**이며, {math}`Z`는 그 파의 임피던스입니다. 앞 절의 {math}`Y_{\mathrm{in}}`에 해당하는 값을 이 예에서는 {math}`Y`로 씁니다. 전기장은 x 성분, 자기장은 y 성분으로 골랐으므로 입사파의 자기장은 {math}`Y A_{\mathrm{in}}`, 반사파의 자기장은 {math}`-Y A_r`입니다. 반사파에 {math}`Y_r=-Y`를 적용한 뒤 두 자기장을 더하면 위의 {math}`H_y=Y(A_{\mathrm{in}}-A_r)`를 얻습니다. **이 전기장·자기장 합성식은 무손실·손실 매질에 모두 적용됩니다.**

### 전체 장을 곱하면 나오는 간섭항

시간평균 순전력 밀도는 전체 장의 곱에서 구합니다.

```{math}
\begin{aligned}
\langle S_z\rangle
&=\frac12\operatorname{Re}(E_x H_y^*)\\[3pt]
&=\frac12\operatorname{Re}\!\left[
Y^*(A_{\mathrm{in}}+A_r)(A_{\mathrm{in}}^*-A_r^*)\right]\\[3pt]
&=\frac12\operatorname{Re}\!\left[
Y^*\left(|A_{\mathrm{in}}|^2-|A_r|^2+A_rA_{\mathrm{in}}^*-A_{\mathrm{in}}A_r^*\right)\right].
\end{aligned}
```

괄호 안의 앞 두 항은 각 파를 단독으로 계산할 때의 기여이고, 뒤 두 항은 서로 다른 파의 전기장과 자기장이 곱해진 **교차항**입니다. {math}`A_rA_{\mathrm{in}}^*-A_{\mathrm{in}}A_r^*=2i\operatorname{Im}(A_{\mathrm{in}}^*A_r)`를 사용하면:

```{math}
\langle S_z\rangle
=\frac{\operatorname{Re}Y}{2}
\left(|A_{\mathrm{in}}|^2-|A_r|^2\right)
+\operatorname{Im}Y\,\operatorname{Im}(A_{\mathrm{in}}^*A_r).
```

입사파와 반사파 각각의 전력 밀도를 {math}`I_{\mathrm{in}}`, {math}`I_r`, 교차항의 기여를 {math}`I_{\mathrm{cross}}`라고 이름 붙이면:

```{math}
\begin{aligned}
I_{\mathrm{in}}&=\frac{\operatorname{Re}Y}{2}|A_{\mathrm{in}}|^2,
& I_r&=\frac{\operatorname{Re}Y}{2}|A_r|^2,\\[3pt]
I_{\mathrm{cross}}&=\operatorname{Im}Y\,\operatorname{Im}(A_{\mathrm{in}}^*A_r),
& \langle S_z\rangle&=I_{\mathrm{in}}-I_r+I_{\mathrm{cross}}.
\end{aligned}
```

{math}`I_r`는 반사파가 나가는 쪽의 전력 밀도를 양수로 읽은 값이므로, 안쪽 순전력에서는 빼 줍니다. {math}`I_{\mathrm{cross}}`는 두 파가 동시에 존재할 때 전체 장의 곱에 포함되는 기여입니다. **별도의 광원이나 세 번째 파가 생기는 것은 아닙니다.**

### 무손실 매질에서 상쇄되는 것

보통의 무손실 매질에서 전파하는 파의 {math}`Y`는 실수이므로 {math}`\operatorname{Im}Y=0`입니다. 따라서:

```{math}
I_{\mathrm{cross}}=0,
\qquad
\langle S_z\rangle=I_{\mathrm{in}}-I_r.
```

그러나 전체 전기장 진폭의 제곱에는 교차항이 남습니다.

```{math}
|E_x|^2=|A_{\mathrm{in}}|^2+|A_r|^2+2\operatorname{Re}(A_{\mathrm{in}}^*A_r).
```

이 항은 위치에 따라 전기장이 강해지거나 약해지는 간섭 무늬를 만듭니다. **무손실에서 상쇄되는 것은 표면에 수직한 시간평균 전력의 교차항입니다.** 전기장과 자기장의 간섭 자체가 없어지는 것은 아닙니다.

### 손실 매질에서 남는 것과 그 부호

손실 매질에서는 {math}`Y`가 복소수일 수 있어 전력의 간섭항이 남을 수 있습니다. 두 파의 진폭이 0이 아닐 때, 상대 위상을 {math}`\Delta\varphi=\arg A_r-\arg A_{\mathrm{in}}`라 하면:

```{math}
I_{\mathrm{cross}}
=\operatorname{Im}Y\,|A_{\mathrm{in}}|\,|A_r|\sin\Delta\varphi.
```

간섭항의 크기와 부호는 매질의 {math}`\operatorname{Im}Y`뿐 아니라 **두 파의 상대 위상**에도 달려 있습니다. {math}`\operatorname{Im}Y=0`이거나 {math}`\sin\Delta\varphi=0`이면 이 항은 사라집니다. 한 파의 진폭이 0인 경우에도 마찬가지입니다.

| 간섭항 | 기준면의 순유입 전력 밀도 |
| --- | --- |
| {math}`I_{\mathrm{cross}}>0` | {math}`I_{\mathrm{in}}-I_r`보다 큼 |
| {math}`I_{\mathrm{cross}}<0` | {math}`I_{\mathrm{in}}-I_r`보다 작음 |
| {math}`I_{\mathrm{cross}}=0` | {math}`I_{\mathrm{in}}-I_r`와 같음 |

**간섭항이 양수라고 이득 매질인 것은 아니며, 음수라고 흡수 매질인 것도 아닙니다.** 매질이 장에 전력을 공급하는지, 장의 전력을 흡수하는지와 간섭항의 부호는 서로 다른 판단입니다. 이 장의 {ref}`계산 예제 <boundary-power-example>`에서도 흡수성 입사 매질의 간섭항이 양수입니다. 이때 늘어난 전력은 입사 매질이 기준면 근처에서 덜 흡수한 몫이며, {ref}`1을 넘은 전력은 어디에서 오는가 <interference-absorption-origin>`에서 높이별로 확인합니다.

### 입사 전력으로 나눈 순유입 전력

경계면을 통해 층 안으로 들어가는 순전력이 **입사파만의 전력에 비해 얼마인지**를 나타내는 값을 {math}`P_{\mathrm{enter}}`라고 쓰겠습니다. 분자와 분모는 모두 입사 경계면에서 읽습니다.

```{math}
P_{\mathrm{enter}}
=\frac{\text{입사·반사파를 합쳐 구한 순유입 전력}}
       {\text{입사파만으로 구한 전력}}.
```

{math}`P_{\mathrm{enter}}`는 전력을 전력으로 나눈 **단위 없는 비율**입니다. 이 예는 아래쪽에서 수직으로 입사하므로, 층 안쪽인 +z 방향의 시간평균 전력 밀도 {math}`\langle S_z\rangle`를 사용합니다. 입사파만의 전력 밀도는 {math}`I_{\mathrm{in}}`이며, 두 값의 단위는 모두 W/m²입니다. 경계면을 따라 전력 밀도가 일정하므로, 같은 면적을 통과하는 전력을 나누면 면적이 약분됩니다.

앞에서 구한 {math}`\langle S_z\rangle=I_{\mathrm{in}}-I_r+I_{\mathrm{cross}}`를 넣고, 반사율의 정의 {math}`R=I_r/I_{\mathrm{in}}`를 사용하면:

```{math}
\begin{aligned}
P_{\mathrm{enter}}
&=\frac{\langle S_z\rangle}{I_{\mathrm{in}}}\\[4pt]
&=\frac{I_{\mathrm{in}}-I_r+I_{\mathrm{cross}}}{I_{\mathrm{in}}}\\[4pt]
&=1-\frac{I_r}{I_{\mathrm{in}}}+\frac{I_{\mathrm{cross}}}{I_{\mathrm{in}}}\\[4pt]
&=1-R+\frac{I_{\mathrm{cross}}}{I_{\mathrm{in}}}.
\end{aligned}
```

이 비율을 정의하려면 {math}`I_{\mathrm{in}}>0`이어야 합니다. 오른쪽의 **1은 입사파만의 전력을 기준으로 삼은 값**, {math}`R`은 되돌아가는 반사파의 전력 비입니다. 마지막 항은 입사파와 반사파의 간섭이 순유입 전력에 기여하는 양을 같은 입사 전력 기준으로 나타낸 것입니다.

입사 매질이 무손실이고 입사파가 전파파이면 {math}`I_{\mathrm{cross}}=0`이므로 {math}`P_{\mathrm{enter}}=1-R`입니다. 손실 매질에서는 이 간섭항이 남을 수 있어, **반사율만으로 순유입 전력을 정할 수 없습니다.** 입사파와 반사파를 합친 전기장·자기장으로 계산해야 합니다.

같은 차수의 입사파와 반사파는 같은 횡파수를 갖습니다. 두 파의 횡방향 위상 인자가 교차항 안에서 서로 상쇄되므로, 이 간섭항은 경계면을 따라 평균해도 유지됩니다.

Wavebit에서는 모든 차수와 편광의 기여를 포함한 순유입 전력 비를 `result.power_entering`에서 읽습니다. 다음 절에서는 이 값과 투과율을 사용해 쌓은 층 전체의 흡수율을 구합니다.

::::{dropdown} 반사 진폭 계수로 순유입 전력 비 계산하기

위 수직 입사 예제에서 입사파와 반사파의 **같은 x 방향 전기장 성분**을 비교한 진폭 반사계수를 {math}`r`라고 하면:

```{math}
r=\frac{A_r}{A_{\mathrm{in}}},
\qquad A_r=rA_{\mathrm{in}},
\qquad R=|r|^2.
```

간섭항에 들어가는 두 진폭의 곱은 다음처럼 쓸 수 있습니다.

```{math}
A_{\mathrm{in}}^*A_r=A_{\mathrm{in}}^*(rA_{\mathrm{in}})=r|A_{\mathrm{in}}|^2.
```

{math}`|A_{\mathrm{in}}|^2`는 실수이므로 {math}`\operatorname{Im}(A_{\mathrm{in}}^*A_r)=|A_{\mathrm{in}}|^2\operatorname{Im}r`입니다. 따라서 간섭항과 입사파만의 전력 밀도는:

```{math}
\begin{aligned}
I_{\mathrm{cross}}
&=\operatorname{Im}Y\,\operatorname{Im}(A_{\mathrm{in}}^*A_r)\\[4pt]
&=\operatorname{Im}Y\,|A_{\mathrm{in}}|^2\operatorname{Im}r,\\[8pt]
I_{\mathrm{in}}&=\frac12\operatorname{Re}Y\,|A_{\mathrm{in}}|^2.
\end{aligned}
```

두 값을 나누면 공통 인자 {math}`|A_{\mathrm{in}}|^2`가 약분됩니다.

```{math}
\begin{aligned}
\frac{I_{\mathrm{cross}}}{I_{\mathrm{in}}}
&=\frac{\operatorname{Im}Y\,|A_{\mathrm{in}}|^2\operatorname{Im}r}
        {\tfrac12\operatorname{Re}Y\,|A_{\mathrm{in}}|^2}\\[5pt]
&=2\frac{\operatorname{Im}Y}{\operatorname{Re}Y}\operatorname{Im}r.
\end{aligned}
```

이를 순유입 전력 비에 대입하면:

```{math}
\begin{aligned}
P_{\mathrm{enter}}
&=1-R+\frac{I_{\mathrm{cross}}}{I_{\mathrm{in}}}\\[4pt]
&=1-|r|^2
+2\frac{\operatorname{Im}Y}{\operatorname{Re}Y}\operatorname{Im}r.
\end{aligned}
```

마지막 항은 반사계수의 크기뿐 아니라 위상에도 영향을 받습니다. {math}`r=|r|e^{i\varphi}`로 쓰면 {math}`\operatorname{Im}r=|r|\sin\varphi`이기 때문입니다.

이 식은 이 장의 {ref}`계산 예제 <boundary-power-example>`처럼 위쪽에서 비스듬히 입사하는 s 편광에도 그대로 적용됩니다. 그때는 {math}`r`을 두 파의 접선 전기장 y 성분의 비로, {math}`Y`를 {ref}`손실 매질의 s 편광 절 <lossy-amplitude-to-efficiency>`의 {math}`Y_{\mathrm{in}}`으로 읽습니다. 그 절에서도 두 파의 접선 자기장이 {math}`Y_{\mathrm{in}}(E_{\mathrm{in}}-E_r)`로 합쳐지므로 전개가 같습니다. 예제의 간섭 기여 0.158416도 이 식의 값이며, 예제 스크립트가 같은 식으로 검산합니다.

이 식의 {math}`r`은 두 파를 **같은 고정된 x 방향**으로 읽은 전기장 성분의 비입니다. Wavebit의 `r_p`는 각 파의 진행 방향에 맞춘 p 기저의 계수이므로, 위 식에 대입할 때는 기저와 부호를 맞춰야 합니다. 결과의 순유입 전력 비는 `result.power_entering`에서 바로 읽을 수 있습니다.

::::

(half-space-power)=
## 쌓은 층 전체의 흡수율과 결과 읽기

정상상태에서는 **입사 경계면을 통해 층 안으로 들어오는 순전력**에서 **반대쪽 경계면으로 나가는 투과 전력**을 빼면, 층 전체에서 흡수한 전력을 얻습니다. 이를 입사파만의 전력으로 나눈 흡수율을 {math}`A`라 하면:

```{math}
A=P_{\mathrm{enter}}-T_{\mathrm{total}}.
```

여기서 {math}`P_{\mathrm{enter}}`는 W 단위의 전력이 아니라 **입사 전력으로 나눈 순유입 전력 비**입니다. Wavebit의 결과는 다음처럼 읽습니다.

| 결과 | 의미 |
| --- | --- |
| `result.R` | 입사 쪽으로 나가는 반사파의 차수별 전력 비 |
| `result.T` | 반대쪽 경계면으로 나가는 투과파의 차수별 전력 비 |
| `result.power_entering` | 입사·반사파를 합친 전체 장으로 계산한 순유입 전력 비 |
| `result.absorption` | 두께를 지정해 쌓은 모든 층의 흡수 전력 비 |

```python
R_total = result.R.sum()
T_total = result.T.sum()
P_enter = result.power_entering
A = result.absorption             # P_enter - T_total
```

입사 매질이 무손실이고 입사파가 전파파이면 {math}`P_{\mathrm{enter}}=1-R_{\mathrm{total}}`이므로:

```{math}
A=1-R_{\mathrm{total}}-T_{\mathrm{total}}.
```

입사 매질만 무손실이라고 {math}`A=0`인 것은 아닙니다. 쌓은 층 중 흡수하는 재료가 있으면 {math}`A>0`일 수 있습니다. 층들까지 모두 무손실이면 {math}`A=0`이고, 이때는 손실 입사 매질에서도 {math}`P_{\mathrm{enter}}=T_{\mathrm{total}}`입니다.

흡수율의 대상은 층 수와 관계없이 **`structure`에 넣고 두께를 지정한 모든 층**입니다. `top`·`bot` 반무한 매질 자체의 흡수는 포함하지 않습니다. 출사 매질이 흡수성이어도 {math}`T_{\mathrm{total}}`은 출사 경계면을 통과하는 전력 비로 읽습니다.

`answers="efficiency"`에서도 순유입 전력과 흡수율을 얻을 수 있습니다. 이 두 값은 경계의 복소 진폭으로 계산하므로, 층 내부의 필드를 별도로 재구성할 필요가 없습니다. 횡파수(`kx`, `ky`) 중 하나라도 0이 아닌 허수부를 가지면, 차수별 전력을 단순히 더한 값이 셀 전체에서 적분한 전력과 같지 않을 수 있습니다. 이 경우 `power_entering`, `absorption`은 `NaN`으로 표시합니다. 입사파가 소멸파일 때({ref}`8장 <complex-transverse-wavevector>`)도 `R`, `T`와 함께 이 두 값이 `NaN`입니다. 나눌 입사 전력이 없기 때문입니다. 이 장에서 사용하는 실수 횡파수의 입사 조건은 {ref}`8장 <absorbing-incidence>`에서 설명합니다.

(boundary-power-example)=
## 계산 예제로 확인하기

### 흡수성 입사 매질에서 T가 1보다 큰 경우

입사 매질 `top`에 흡수가 있을 때와 없을 때의 전력 흐름을 비교합니다. 두 계산 모두 진공 파장 1,000 nm의 s 편광을 위쪽에서 입사시킵니다. 입사 매질에서 위상이 진행하는 방향과 아래쪽 법선 사이의 각도는 30°로 지정합니다.

`top`과 `bot` 사이에는 두께 100 nm인 균일한 층 하나를 둡니다. 이 층과 `bot`은 모두 상대유전율 {math}`\varepsilon_r=2.25`인 무손실 재료이며, 모든 매질과 층의 상대투자율은 {math}`\mu_r=1`입니다. 층과 `bot`의 재료가 같으므로 두 영역 사이에서는 반사가 생기지 않습니다.

이 설정에서 `top`의 상대유전율만 바꿔 두 번 계산합니다. 첫 번째는 {math}`\varepsilon_r=4`인 무손실 매질, 두 번째는 {math}`\varepsilon_r=4+3i`인 흡수성 매질입니다. 두 계산 모두 층 자체에는 흡수가 없으므로, 입사 매질의 손실에 따라 반사율·투과율과 순유입 전력이 어떻게 달라지는지 비교할 수 있습니다.

각도는 두 경우 모두 30°이지만, `top`의 재료가 달라 횡파수 `kx`는 서로 다릅니다. 이 예제는 입사면을 xz 평면으로 잡아 `ky=0`으로 계산합니다. 각 경우의 파수는 {ref}`8장의 각도 입력 방법 <absorbing-incidence>`으로 구합니다.

```{figure} ../../figures/learn/learn/12-power-balance.svg
:alt: 입사 전력으로 정규화하면 무손실 입사에서는 1에서 R을 뺀 값이 순유입 전력이다. 손실 입사 예제에서는 간섭 기여 0.158416을 포함해 순유입 전력과 투과율이 모두 1.013189가 되고, 층의 흡수율은 0이다.
:figclass: learning-figure

그림 13-5. 입사 매질에 따른 순유입 전력과 투과 전력의 비교. 쌓은 층과 출사 매질은 두 경우 모두 무손실입니다. 모든 값은 입사 기준면에서의 입사파만의 전력으로 나눈 비율입니다.
```

| 값 | 무손실 입사: {math}`\varepsilon_r=4` | 손실 입사: {math}`\varepsilon_r=4+3i` |
| --- | ---: | ---: |
| `R_total` | 0.046414 | 0.145227 |
| `1 - R_total` | 0.953586 | 0.854773 |
| 정규화한 간섭 기여 | 0 | 0.158416 |
| `power_entering` | 0.953586 | 1.013189 |
| `T_total` | 0.953586 | 1.013189 |
| `absorption` | 0 | 0 |
| `1 - R_total - T_total` | 0 | −0.158416 |

값은 소수점 여섯 자리로 반올림했습니다. 계산에서 0인 흡수율에는 부동소수점 오차가 남을 수 있습니다.

흡수성 입사 매질의 경우 {math}`T_{\mathrm{total}}>1`이지만, 실제 순유입 전력 비와 투과 전력 비는 모두 1.013189로 같습니다. 따라서 층의 흡수율은 0입니다. 입사파만의 전력을 기준 1로 삼았고, 경계의 순유입 전력에는 입사·반사 간섭도 기여하기 때문에 가능한 결과입니다. 이때 음수인 {math}`1-R_{\mathrm{total}}-T_{\mathrm{total}}`을 층의 흡수율로 읽으면 안 됩니다.

(interference-absorption-origin)=
#### 1을 넘은 전력은 어디에서 오는가

입사파만의 전력을 1로 삼았는데 층으로 1.013189가 들어갔다면, 그 차이가 어디에서 왔는지 물을 수 있습니다. 답은 **입사 매질 `top` 자신의 흡수**에 있습니다. 반사파가 있으면 `top`이 기준면 근처에서 흡수하는 전력이 달라지고, 덜 흡수한 만큼 더 많은 전력이 기준면에 도착합니다.

**먼저, 1은 광원이 준 전력 전체가 아닙니다.** 전력을 읽는 경계면을 정할 때 설명했듯이, 1은 기준면에 도달한 입사파만의 전력이며 그 전에 `top`에서 흡수된 전력은 포함하지 않습니다. 이 예제의 `top`은 흡수가 커서, 기준면에서 약 112.45 nm 위에서는 입사파만의 전력이 3.112882입니다. 입사파 성분만 따로 보면 그 높이에서 기준면까지 내려오는 동안 2.112882가 줄어듭니다. **이 값은 입사파 성분만의 감소이며, 그 구간에서 `top`이 실제로 흡수한 전력이 아닙니다.** 실제 흡수에는 반사파와 간섭의 몫도 들어가며, 아래에서 나누어 구합니다.

**흡수는 전체 전기장의 제곱을 따릅니다.** 손실 매질이 단위 부피에서 흡수하는 시간평균 전력은:

```{math}
p_{\mathrm{abs}}
=\frac{\omega\varepsilon_0\operatorname{Im}\varepsilon_r}{2}|\mathbf E|^2.
```

이 예제의 모든 매질은 {math}`\mu_r=1`이므로 자기장에 의한 흡수는 없습니다. 높이 {math}`z`에서 입사파와 반사파의 전기장 y 성분을 {math}`E_{\mathrm{in}}(z)`, {math}`E_r(z)`라 하면, 전체 전기장의 제곱에는 {ref}`순유입 전력 절 <incident-reflected-power>`의 무손실 매질 설명에서 본 교차항이 있습니다.

```{math}
|E_{\mathrm{in}}+E_r|^2=|E_{\mathrm{in}}|^2+|E_r|^2+2\operatorname{Re}(E_{\mathrm{in}}^*E_r).
```

이 교차항은 전기장이 강해지는 **밝은 줄**과 약해지는 **어두운 줄**을 만듭니다. 무손실 매질에서는 {math}`\operatorname{Im}\varepsilon_r=0`이라 이 무늬가 흡수를 바꾸지 않습니다. 흡수성 매질에서는 밝은 줄에서 더 흡수하고, 어두운 줄에서 덜 흡수합니다.

**전력 수지를 높이별로 나눕니다.** 기준면({math}`z=0`)과 그 위 높이 {math}`z`인 면 사이의 `top`을 생각합니다. 정상상태에서 위쪽 면으로 들어온 순전력과 기준면으로 나간 순전력의 차이는 그 사이에서 흡수한 전력입니다. 순전력을 앞 절처럼 입사파만·반사파만·간섭 기여로 나누고 흡수도 같은 방식으로 나누면, 세 부분이 각각 따로 맞습니다. 간섭 기여에 해당하는 부분은:

```{math}
I_{\mathrm{cross}}(0)
=I_{\mathrm{cross}}(z)
-\int_0^{z}\frac{\omega\varepsilon_0\operatorname{Im}\varepsilon_r}{2}\,
2\operatorname{Re}\!\left[E_{\mathrm{in}}^*(z')E_r(z')\right]dz'.
```

{math}`I_{\mathrm{cross}}(z)`는 높이 {math}`z`인 면에서 앞 절과 같은 방식으로 구한 간섭 기여입니다. 입사파는 위로 갈수록 커지고 반사파는 작아지므로 두 진폭의 곱은 높이에 무관하며, 간섭 기여는 높이에 따라 부호를 바꾸며 진동합니다.

```{figure} ../../figures/learn/learn/13-interference-absorption.svg
:alt: 흡수성 입사 예제에서 기준면 위 0에서 200 nm까지 높이별로 그린 두 패널. 왼쪽은 top이 흡수하는 전력으로, 점선은 입사파와 반사파를 따로 계산한 값, 실선은 합친 전체 장으로 계산한 값이다. 0에서 약 45.49 nm까지는 실선이 커서 더 흡수하고, 약 45.49에서 179.42 nm까지는 실선이 작아 덜 흡수한다. 오른쪽은 높이별 간섭 기여로, 약 112.45 nm에서 0, 약 45.49 nm에서 +0.328, 기준면에서 +0.158이다. 약 112.45 nm 높이에서 입사파만의 전력은 3.11이다.
:figclass: learning-figure

그림 13-6. 흡수성 입사 매질에서 간섭 기여가 생기는 곳. 가로축의 값은 모두 기준면에서 입사파만의 전력으로 나눈 비율이며, 왼쪽은 높이 100 nm당 흡수 전력입니다. 배경의 주황·청록 띠는 간섭 때문에 각각 더·덜 흡수하는 높이입니다.
```

::::{dropdown} 높이별 흡수 전력 계산하기

그림 13-6 왼쪽의 곡선은 높이마다 `top` 안의 전기장을 구해, 앞에서 본 흡수 밀도 {math}`p_{\mathrm{abs}}`에 넣은 값입니다.

**1. `top` 안의 장을 높이의 함수로 씁니다.** 이 예제는 층과 `bot`의 재료가 같아 회절이 없으므로, `top` 안의 장은 입사파와 반사파 두 평면파뿐입니다. 기준면에서 입사파의 접선 전기장을 1로 두면, 높이 {math}`z`에서 두 파의 전기장 y 성분은:

```{math}
E_{\mathrm{in}}(z)=e^{-iqk_0z},
\qquad
E_r(z)=r\,e^{iqk_0z}.
```

{math}`k_0=2\pi/\lambda`는 진공 파수이고, {math}`q`는 입사파의 법선 파수를 진행 방향인 −z 쪽으로 읽어 {math}`k_0`로 나눈 값입니다({math}`k_{z,\mathrm{in}}=-qk_0`). {math}`q`는 파수 관계식 {math}`q^2=\varepsilon_r\mu_r-(k_x/k_0)^2`의 두 해 중 아래로 진행하며 감쇠하는 쪽이고, 여기에 쓰는 {math}`k_x`는 {ref}`8장의 예시 <absorbing-incidence-example>`처럼 위상각 30°의 조건과 이 관계식을 함께 풀어 정합니다({math}`k_x/k_0=1.077700`). 예제가 기록한 값은 {math}`q=1.866632+0.803587i`, {math}`r=0.333729+0.183990i`입니다. {math}`\operatorname{Im}q>0`이므로 입사파는 위로 갈수록 커지고, 반사파는 위로 갈수록 작아집니다.

**2. 흡수 밀도에 넣습니다.** 실선은 두 파를 합친 장의 {math}`|E_{\mathrm{in}}+E_r|^2`을, 점선은 두 파를 따로 계산한 {math}`|E_{\mathrm{in}}|^2+|E_r|^2`을 {math}`p_{\mathrm{abs}}`의 {math}`|\mathbf E|^2` 자리에 넣은 값입니다. 두 선의 차이가 교차항 {math}`2\operatorname{Re}(E_{\mathrm{in}}^*E_r)`입니다.

**3. 입사파만의 전력으로 나눕니다.** 구조가 옆으로 균일하므로, 단위 면적으로 보면 {math}`p_{\mathrm{abs}}`는 높이 1 nm당 흡수한 전력입니다. 이를 기준면에서 입사파만의 전력 {math}`I_{\mathrm{in}}(0)=\tfrac12\operatorname{Re}(q)/Z_0`로 나누고 {math}`\omega\varepsilon_0Z_0=k_0`를 쓰면:

```{math}
\frac{p_{\mathrm{abs}}(z)}{I_{\mathrm{in}}(0)}
=\frac{k_0\operatorname{Im}\varepsilon_r}{\operatorname{Re}q}\,|E(z)|^2
=2k_0\operatorname{Im}q\,|E(z)|^2.
```

두 번째 등호는 횡파수가 실수라서 {math}`q^2=\varepsilon_r-(k_x/k_0)^2`의 허수부에서 {math}`\operatorname{Im}\varepsilon_r=2\operatorname{Re}q\operatorname{Im}q`가 되기 때문입니다. 이 예제에서는 {math}`2k_0\operatorname{Im}q=0.010098` /nm이므로, 그림의 가로축인 **높이 100 nm당 흡수는 {math}`1.009817\,|E(z)|^2`**입니다.

입사파만의 전력으로 검산할 수 있습니다. 입사파만의 전력은 {math}`I_{\mathrm{in}}(z)/I_{\mathrm{in}}(0)=e^{2k_0\operatorname{Im}q\,z}`이고, 이를 {math}`z`로 미분하면 {math}`2k_0\operatorname{Im}q\,|E_{\mathrm{in}}(z)|^2`입니다. 입사파가 높이마다 잃는 전력이 자기 장으로 계산한 흡수 밀도와 같습니다.

**4. 몇 높이의 값입니다.** 모두 높이 100 nm당 흡수이며, 기준면에서 입사파만의 전력으로 나눈 값입니다.

| 높이 {math}`z` | 실제: 합친 장 | 두 파를 따로 | 간섭의 효과 |
| ---: | ---: | ---: | --- |
| 0 (기준면) | 1.830479 | 1.156470 | 더 흡수 |
| 약 45.49 nm | 1.691179 | 1.691179 | 밝은 줄과 어두운 줄의 경계 |
| 약 112.45 nm | 2.420896 | 3.190552 | 덜 흡수 |

**5. 구간의 흡수는 적분입니다.** 기준면부터 약 112.45 nm까지 실선을 적분하면, 이 구간에서 `top`이 실제로 흡수한 2.053040이 나옵니다. 이 값은 두 높이에서 구한 순전력의 차이와 같으며, 그림 스크립트가 이 등식을 수치 적분으로 확인합니다.

이번에는 `top`이 균일한 매질이고 장이 두 평면파뿐이라 닫힌 식으로 계산했습니다. 패턴이 있는 층 안에서는 유전율이 x·y 위치에 따라 달라지므로, [14장](fields.md)처럼 필드를 구한 뒤 등방성 재료의 흡수 밀도 {math}`\tfrac12\omega\varepsilon_0\operatorname{Im}\varepsilon_r(x,y)\,|\mathbf E|^2`을 단위 셀 면적에 걸쳐 적분합니다. 그 값이 그 높이의 흡수이고, 높이로 한 번 더 적분하면 구간의 흡수입니다. 투자율에도 손실이 있으면 {math}`\tfrac12\omega\mu_0\operatorname{Im}\mu_r\,|\mathbf H|^2`을 더합니다. 쌓은 층 전체의 흡수만 필요하면 이 적분 없이, 두 경계면의 순전력 차이인 `absorption`을 읽습니다.

::::

그림의 오른쪽에서 간섭 기여는 약 112.45 nm 높이에서 0입니다. 위 식에 이 높이를 넣으면, 기준면의 간섭 기여는 **기준면과 이 높이 사이에서 두 파를 따로 계산할 때보다 덜 흡수한 전력**과 같습니다. 약 45.49–112.45 nm의 어두운 줄에서 덜 흡수하며 +0.328까지 쌓이고, 0–45.49 nm의 밝은 줄에서 일부를 다시 더 흡수해 +0.158416으로 기준면에 도착합니다.

같은 0–112.45 nm 구간에서 `top`이 흡수하는 전력을 세 부분으로 나누면:

| 입사파 성분 | 반사파 성분 | 간섭 | 실제 총흡수 |
| ---: | ---: | ---: | ---: |
| 2.112882 | 0.098574 | −0.158416 | 2.053040 |

입사파 성분은 앞에서 본 입사파만의 감소이고, 반사파 성분은 반사파가 기준면에서 이 높이까지 올라가며 잃는 전력입니다. 실제 총흡수 2.053040은 이 높이와 기준면에서 구한 순전력의 차이와 같습니다. 기준면에서 정리하면:

| 입사파만의 전력 | 반사파가 되돌아감 | 간섭으로 `top`이 덜 흡수 | 기준면을 지나는 순전력 |
| ---: | ---: | ---: | ---: |
| 1 | −0.145227 | +0.158416 | 1.013189 |

층과 `bot`은 무손실이므로 기준면을 지난 1.013189가 그대로 {math}`T_{\mathrm{total}}`이 됩니다. **1을 넘은 0.013189를 포함한 간섭 기여는 새로 생긴 에너지가 아니라, 두 파를 따로 계산할 때 `top`이 흡수했을 몫 중 층 쪽으로 넘어온 전력입니다.** 간섭 기여의 부호는 반사계수의 위상에 달려 있으므로, 반사 위상이 다르면 `top`이 오히려 더 흡수하여 순유입 전력이 {math}`1-R`보다 작아질 수도 있습니다.

아래에서 `top`의 손실을 바꿔 보면, 전기장의 간섭 무늬와 `top`의 흡수, 간섭 기여가 함께 달라지는 것을 볼 수 있습니다. 층과 `bot`이 같은 이 구조에서는 무손실 출사 매질로 전파파가 나가므로 {math}`\operatorname{Im}r>0`이고, 간섭 기여는 음수가 되지 않습니다. 층이 `bot`과 다르면 층의 두께에 따라 반사 위상이 달라져, 간섭 기여가 음수가 되는 경우도 생깁니다.

<section class="exercise power-flux-exercise" aria-labelledby="power-flux-title">
<h3 id="power-flux-title">직접 바꿔보기 · top의 손실을 바꾸면 간섭 기여는?</h3>
<p class="concept">개념 조작 · 이 절의 예제와 같은 단일 경계를 닫힌 식으로 계산합니다. Im ε<sub>r</sub> = 3이면 위 표의 Wavebit 결과와 같고, 0이면 무손실 입사입니다. 왼쪽 패널의 장은 시간에 따라 진동하는 순간값입니다.</p>
<div class="controls">
<label for="pf-loss">top의 Im ε<sub>r</sub> (Re ε<sub>r</sub> = 4) <output id="pf-loss-out" for="pf-loss">3.0</output><input id="pf-loss" type="range" min="0" max="6" step="0.1" value="3"></label>
<button id="pf-play" class="pf-play" type="button">일시정지</button>
</div>
<div class="svg-scroll"><svg id="pf-fig" viewBox="0 0 680 400" role="img" aria-label="기준면에서 잰 높이 z를 공유하는 세 패널. 왼쪽은 top 안의 입사파, 반사파, 둘을 합친 전기장의 순간값, 가운데는 top이 높이별로 흡수하는 전력을 간섭이 있을 때와 없을 때로 비교, 오른쪽은 높이 z에서의 간섭 기여이다."></svg></div>
<p class="pf-legend" id="pf-legend"><span class="pf-swatch pf-more"></span>밝은 줄: 간섭 때문에 더 흡수<span class="pf-swatch pf-less"></span>어두운 줄: 간섭 때문에 덜 흡수</p>
<div class="readout" id="pf-readout" aria-live="polite"><p>기준면: 입사파만 1 − 반사 0.145227 + 간섭 기여 0.158416 = 순유입 1.013189 = T</p></div>
<noscript><p>JavaScript를 켜면 top의 손실을 바꾸고 전기장의 움직임을 볼 수 있습니다.</p></noscript>
</section>

### 반사가 없어도 층에서 흡수하는 경우

이번에는 `top`, 두께를 지정한 층, `bot`에 모두 같은 {math}`\varepsilon_r=2.16+0.9i`, {math}`\mu_r=1`을 넣습니다. 재료가 바뀌는 경계가 없어 반사율은 0입니다. 진공 파장 500 nm, 수직입사에서 두께 75 nm 구간을 통과하는 전력을 비교하면:

| {math}`R_{\mathrm{total}}` | {math}`P_{\mathrm{enter}}` | {math}`T_{\mathrm{total}}` | 흡수율 {math}`A` |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0.568084 | 0.431916 |

이 재료의 복소 굴절률은 {math}`\widetilde n=1.5+0.3i`입니다. 두께 {math}`d`를 지나는 동안의 전력 감쇠는 다음 식과 일치합니다.

```{math}
T_{\mathrm{total}}
=\exp\!\left[-\frac{4\pi\operatorname{Im}(\widetilde n)d}{\lambda}\right].
```

반사는 없지만 지정한 두께 안에서 입사 전력의 약 43.2%가 흡수됩니다. `absorption`은 이처럼 두 경계면 사이에서 흡수한 양을 나타냅니다.

### 예제 실행

실행 파일: {download}`10_boundary_power.py <../../../examples/rcwa/learning/learn/10_boundary_power.py>`

```sh
python 10_boundary_power.py
```

예제는 Wavebit 결과를 출력하고, 단일 경계의 Fresnel 계수로 구한 전력, 실제 시간 신호의 곱을 한 주기 평균한 순전력, 균일 손실 매질의 지수 감쇠와 비교합니다. 이 절의 표와 그림 13-5는 이 예제의 계산값을 사용합니다. 그림 13-6은 이 예제가 기록한 반사계수와 입사 매질의 법선 파수로 `top` 안의 두 파를 펼쳐 그렸으며, 그림 스크립트가 높이별 전력 수지를 수치 적분으로 다시 확인합니다.

## 계산 결과를 비교할 때의 기준

다른 계산과 비교할 때는 재료뿐 아니라 실제 입사 횡파수, 진폭을 읽는 s·p 기저, 전력 기준면과 정규화 분모를 맞춥니다. 특히 손실 매질에서는 같은 숫자의 각도를 넣었더라도 각도 정의에 따라 다른 파가 될 수 있습니다. 비교하는 값이 **반사파만의 전력인지, 입사·반사파를 합친 순전력인지**도 확인해야 합니다.

```{dropdown} 코드에서 사용하는 s·p 경계 진폭
`src/wavebit/observations/power.py`의 `wave_power`는 한 방향 파만의 전력을, `net_power`는 같은 경계면의 입사·반사파를 합친 순전력을 구합니다.

솔버 내부는 s 채널에서 접선 전기장, p 채널에서 정규화한 접선 자기장 `Z₀H`를 주 진폭으로 사용합니다. 해당 채널에서 사용하는 법선 파수를 진공 파수로 나눈 값을 `κ`라 하면:

| 내부 채널 | 주 진폭 | 전력 환산에 쓰는 계수 |
| --- | --- | --- |
| s (TE) | 접선 전기장 E | `κ/μr` |
| p (TM) | 접선 자기장 Z₀H | `κ/εr` |

두 채널은 공통 전력 계수 `1/(2Z₀)`를 생략해 계산하며, 이 계수는 효율의 분자·분모에서 약분됩니다. p 편광의 입력 전기장 계수가 `aₚ`이면 내부 자기장 계수는 `(n/μr)aₚ`이므로, 입사 전력에 `|n/μr|²`가 반영됩니다. 여기서 `n²=εr μr`입니다.

공개 결과의 `r_s`, `r_p`, `t_s`, `t_p`는 전기장 진폭 계수입니다. 내부 p 채널의 자기장 진폭과 구별해야 합니다.
```

흡수성 입사 매질에서 경계의 순유입 전력이 {math}`1-R`과 다른 이유는 [Byrnes, *Multilayer optical calculations*, §4.3–4.4와 Appendix B](https://arxiv.org/html/1603.02720v5#A2)에서도 설명합니다. Wavebit의 수치 검증 범위는 [반무한 매질 전력 검증 기록](../../validation/rcwa/half-space-power-audit.md)을 참고합니다.
