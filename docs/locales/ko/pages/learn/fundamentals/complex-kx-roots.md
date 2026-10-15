(complex-kx-roots)=
# 부록 A. 무손실 매질에서 kz 고르기: 두 근, branch point, branch cut

:::{note}
이 교재의 실행 예제는 원래 수치 기록을 재현할 수 있도록 `wavebit.compat.legacy` 인터페이스를 명시적으로 사용합니다. 새 작업은 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 참고하세요. 기존 측정값은 이번 문서 이행에서 다시 계산하지 않았습니다.
:::

반공간(위아래의 반무한 매질)에서 파의 수직 파수 {math}`k_z`는 제곱근으로 정해지므로 후보가 늘 둘입니다. 그중 어느 것이 구조에서 멀어지는 파인지 골라야 합니다. 무손실 매질에서 횡파수 {math}`k_x`가 실수이면 둘 중 무엇을 쓸지 저절로 정해집니다. 그러나 복소 `theta`로 {math}`k_x`가 복소수가 되면 **규칙을 따로 정해야** 합니다. 이 부록은 무손실 매질 하나만 놓고, 그 규칙이 왜 필요한지, 규칙마다 무엇이 다른지, Wavebit과 다른 솔버가 각각 어떤 규칙을 쓰는지를 처음부터 설명합니다. 흡수 매질과 이득 매질에서 무엇이 달라지는지는 {ref}`부록 B <complex-kx-lossy-gain>`에서 이어서 봅니다.

복소 `theta`를 쓸 때({ref}`8장 <complex-transverse-wavevector>`)와 다른 솔버와 결과를 맞춰 볼 때 필요한 내용입니다. 실수 각도로만 계산한다면 {ref}`실수 kx에서는 왜 근이 저절로 정해지나 <ckr-real-kx>`까지만 읽어도 충분합니다.

2026년 10월 2일 기준 Wavebit의 반공간 근 선택(`boundary_kz`)을 설명합니다. Wavebit은 이날 반공간의 규칙을 뒤에서 볼 **경로 D**에서 **Maystre 규칙**({math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0`)으로 바꿨습니다. 실수 {math}`k_x`의 수동 매질에서는 두 규칙이 비트 단위로 같으므로, 실수 각도로 푼 결과는 하나도 바뀌지 않았습니다. 이 부록은 경로 D를 "2026년 10월 2일 전의 Wavebit" 규칙으로 남겨 두고, 두 규칙이 어디서 왜 갈리는지를 함께 봅니다. 바꾼 근거와 측정값은 [분기 처리](../../theory/rcwa/branch-cuts.md)의 8절에 있습니다.

그림은 `tools/docs/learning/draw_complex_kx_roots.py`로 그렸고, Wavebit이나 RETICOLO로 푼 수치는 `docs/validation/learning-records/complex-kx-roots.json`에 기록한 값입니다. 기록에는 계산에 쓴 Wavebit 커밋이 함께 남아 있습니다. 경로 D로 푼 Wavebit 수치는 규칙을 바꾸기 전의 커밋에서 잰 값입니다.

(ckr-premises)=
## 이 부록의 전제

처음부터 끝까지 아래 넷을 전제로 합니다. 하나라도 깨지면 결론이 달라집니다.

1. **시간 규약은 {math}`e^{-i\omega t}`입니다.** {ref}`13장 <power-and-efficiency>`과 같은 규약입니다. 아래에서 부호로 방향을 읽는 규칙({math}`\operatorname{Im}k_z>0`이면 줄어듦 등)도 모두 이 규약 기준입니다. 공학에서 쓰는 {math}`e^{+j\omega t}` 규약이면 허수부의 부호가 모두 반대입니다.
2. **매질은 무손실 유전체입니다.** {math}`\varepsilon`이 양의 실수({math}`\operatorname{Im}\varepsilon=0`, {math}`\varepsilon>0`)라서 빛을 흡수하지도 키우지도 않습니다. 공기와 이상적인 유리가 그렇습니다. 굴절률 {math}`n=\sqrt\varepsilon`도 양의 실수이므로, 뒤에서 볼 branch point {math}`k_x=\pm n`이 실수축 위에 놓입니다. 흡수 매질과 이득 매질은 이 전제를 푸는 {ref}`부록 B <complex-kx-lossy-gain>`에서 봅니다.
3. **자성이 없고({math}`\mu=1`) 등방성입니다.** 그래서 굴절률은 {math}`n=\sqrt\varepsilon`이고 {math}`k_z^2=\varepsilon-k_x^2`입니다. 자성 매질은 다루지 않습니다.
4. **근을 골라야 하는 곳은 위아래의 반무한 반공간뿐입니다.** 두께가 유한한 층 안에서는 {math}`+k_z`와 {math}`-k_z`가 둘 다 계산에 들어가므로, 어느 쪽을 "나가는 파"라고 부르든 답이 같습니다([시간 규약 감사](../../validation/rcwa/time-convention-audit.md)). 이 부록의 {math}`k_z`는 모두 반공간의 {math}`k_z`입니다.

**방향을 읽는 기준.** 이 부록은 반공간의 장을 {math}`e^{ik_zd}`로 씁니다. {math}`d`는 구조에서 떨어진 거리입니다. 그래서 {math}`\operatorname{Re}k_z>0`이면 구조에서 멀어지는 파이고, {math}`\operatorname{Im}k_z>0`이면 멀어질수록 줄어드는 파입니다. 8장처럼 위쪽이 +인 전역 {math}`z`로 쓰면, 아래쪽 반공간에서는 {math}`d=-z`이므로 {math}`k_z`의 부호가 반대로 보입니다.

## 1편. 개념: 두 근에서 branch cut까지

(ckr-two-roots)=
### kz는 왜 둘인가

균일한 매질 하나 안의 평면파를 생각합니다. 파수는 모두 {math}`k_0=2\pi/\lambda`로 나눈 값으로 쓰고, 거리 {math}`d`에는 거꾸로 {math}`k_0`를 곱합니다. 굴절률이 {math}`n`인 매질 안에서 평면파의 파수 벡터는 길이가 {math}`n`으로 정해져 있습니다. 그래서 성분 사이에 다음 관계가 있습니다. 이것을 분산 관계라고 부릅니다.

```{math}
k_x^2+k_z^2=\varepsilon .
```

오른쪽의 {math}`\varepsilon`은 {math}`n^2`입니다. {math}`k_y`가 있으면 {math}`k_x^2` 자리에 {math}`k_x^2+k_y^2`를 넣습니다. {math}`k_x`는 경계면을 따라가는 성분입니다. 경계면을 따라서는 모든 층의 위상이 맞아야 하므로(위상 정합) {math}`k_x`는 입사광과 격자 주기가 정하고, 매질이 바뀌어도 그대로입니다. 그러니 매질마다 남은 {math}`k_z`만 새로 정하면 됩니다. 분산 관계에서 {math}`k_z`를 풀면 다음과 같습니다.

```{math}
k_z=\pm\sqrt{\varepsilon-k_x^2}.
```

제곱근이므로 답이 **둘**입니다. 두 근은 같은 {math}`k_x`를 가진 서로 다른 두 파동입니다. 장을 {math}`e^{ik_zd}`로 쓰면 두 성질이 읽힙니다.

- {math}`\operatorname{Re}k_z>0`이면 위상이 {math}`d`를 따라 앞으로 갑니다. 시간 인자 {math}`e^{-i\omega t}`와 함께 보면, 같은 위상의 면이 시간이 지나며 {math}`d`가 커지는 쪽으로 움직입니다. 파가 구조에서 **멀어집니다**(나감).
- {math}`\operatorname{Im}k_z>0`이면 {math}`|e^{ik_zd}|=e^{-\operatorname{Im}k_z\,d}`입니다. 멀어질수록 **줄어듭니다**.

```{figure} ../../figures/learn/learn/a01-two-roots.svg
:alt: 왼쪽은 kz 평면이다. kx가 0.5일 때의 두 근 +0.866과 −0.866이 실수축 위에, kx가 1.5일 때의 두 근 +1.118i와 −1.118i가 허수축 위에 있고, 각 쌍은 원점을 지나는 점선으로 이어져 원점 대칭이다. +0.866과 +1.118i는 채운 점, −0.866과 −1.118i는 빈 점이다. 오른쪽 네 칸은 떨어진 거리 k0·d에 따른 장이다. +0.866은 조금 뒤의 흐린 곡선이 바깥쪽으로 밀려 구조에서 멀어지고, −0.866은 구조 쪽으로 다가온다. +1.118i는 거리에 따라 지수적으로 줄고, −1.118i는 지수적으로 커진다.
:figclass: learning-figure

그림 A-1. 같은 {math}`k_x`에 근이 둘 있습니다. 공기({math}`\varepsilon=1`)에서 {math}`k_x=0.5`이면 {math}`k_z=\pm0.866`, {math}`k_x=1.5`이면 {math}`k_z=\pm1.118i`입니다. 작은 그래프는 각 근의 장이 구조에서 멀어지면서 어떻게 변하는지 보여 줍니다. 개념 설명용 그림입니다.
```

그림 A-1은 공기({math}`\varepsilon=1`)의 두 경우입니다.

- {math}`k_x=0.5`이면 {math}`k_z=\pm0.866`으로 실수입니다. {math}`+0.866`은 구조에서 멀어지며 나아가고, {math}`-0.866`은 구조 쪽으로 다가옵니다.
- {math}`k_x=1.5`이면 {math}`k_z=\pm1.118i`로 순허수입니다. {math}`+1.118i`는 멀어질수록 줄어들고, {math}`-1.118i`는 멀어질수록 커집니다.

두 근은 언제나 원점 대칭({math}`k_z`와 {math}`-k_z`)입니다. 할 일은 둘 중 **하나를 고르는 것**입니다. Wavebit의 `boundary_kz`도 같은 물음, 곧 "반공간에서 구조로부터 멀어지는 파의 {math}`k_z`는 무엇인가"에 답합니다.

(ckr-real-kx)=
### 실수 kx에서는 왜 근이 저절로 정해지나

물리가 요구하는 기준은 두 가지입니다.

- **나감**: {math}`\operatorname{Re}k_z>0`. 바깥 반공간에는 광원이 없으므로 에너지는 구조에서 멀어지는 쪽으로만 흐릅니다(방사 조건).
- **줄어듦**: {math}`\operatorname{Im}k_z>0`. 광원이 없는 곳에서 장이 무한히 커지면 안 됩니다.

{math}`k_z` 평면을 네 사분면으로 나누면 두 기준이 한눈에 보입니다. 그림에서 사분면을 칠할 때는 1사분면을 청록, 2사분면을 파랑, 3사분면을 회색, 4사분면을 주황으로 칠하며, 이 약속은 부록 끝까지 같습니다. 칠한 색의 뜻이 다른 그림은 그림 A-8 하나이고, 거기서 따로 밝힙니다.

| 사분면 | 나감 | 줄어듦 |
| --- | --- | --- |
| 1사분면 | O | O |
| 2사분면 | X (들어옴) | O |
| 3사분면 | X (들어옴) | X (커짐) |
| 4사분면 | O | X (커짐) |

실수 {math}`k_x`에서는 두 근 중 하나가 늘 1사분면(경계 포함)에 있습니다. 세 단계로 보면 이렇습니다.

**① 복소수는 화살표이고, 제곱근은 각도를 절반으로 만듭니다.** 복소수 하나는 원점에서 뻗은 화살표 하나이며, 길이와 각도(실수축에서 반시계로 잰 각)를 가집니다. 제곱하면 길이는 제곱이 되고 각도는 두 배가 됩니다. 그러니 제곱근을 구하면 각도는 **절반**이 됩니다. 180°짜리 수, 곧 음수의 제곱근은 90°짜리, 곧 순허수입니다.

**② {math}`k_x`가 실수이면 {math}`k_z^2`은 실수축 위에 있습니다.** {math}`k_z^2=\varepsilon-k_x^2`입니다. {ref}`전제 2 <ckr-premises>`로 {math}`\varepsilon`이 실수이고, {math}`k_x`가 실수이면 {math}`k_x^2`도 실수입니다. 실수에서 실수를 뺐으니 {math}`k_z^2`도 허수부가 없는 실수입니다.

```{math}
\operatorname{Im}k_z^2=\operatorname{Im}(\varepsilon-k_x^2)=\operatorname{Im}\varepsilon=0 .
```

이 허수부 0은 계산 규칙에서 저절로 나오는 것이 아니라, {math}`\varepsilon`에 허수부가 없다는 **전제 2**(무손실)에서 옵니다.

실수축 위에서도 부호에 따라 두 경우로 나뉩니다. {math}`n^2=\varepsilon`이므로 {math}`k_z^2=n^2-k_x^2`입니다.

- **전파 차수**({math}`|k_x|<n`)이면 {math}`k_z^2>0`, 곧 양의 실수축(각도 0°)입니다. 공기에서 {math}`k_x=0.5`이면 {math}`k_z^2=1-0.25=0.75`입니다.
- **소멸 차수**({math}`|k_x|>n`)이면 {math}`k_z^2<0`, 곧 음의 실수축(각도 180°)입니다. 공기에서 {math}`k_x=1.5`이면 {math}`k_z^2=1-2.25=-1.25`입니다.

**③ 절반으로 줄이면 1사분면의 경계에 떨어집니다.**

- 0°의 절반은 0°, 곧 양의 실수축이고, 180°의 절반은 90°, 곧 양의 허수축입니다. 둘 다 **1사분면**의 경계입니다.
- 다른 근은 부호만 반대({math}`-k_z`)이므로 정확히 맞은편(180°나 270°)에 놓입니다.
- 1사분면(경계 포함)은 실수부와 허수부가 모두 0 이상이므로 "나감"과 "줄어듦"을 어기지 않습니다. 맞은편 근은 한 기준을 어깁니다.

그래서 1사분면 쪽 근을 고르면 됩니다. 두 기준이 서로 다른 근을 가리키는 일이 없으므로, **규칙을 따로 정하지 않아도 근이 저절로 정해집니다.**

```{figure} ../../figures/learn/learn/a02-half-angle.svg
:alt: 왼쪽은 공기의 kz² 평면으로, 실수축이 청록 굵은 선으로 강조되어 있고 오른쪽 절반에 전파(kz²>0, 0도), 왼쪽 절반에 소멸(kz²<0, 180도)이라고 적혀 있다. kx가 0.5일 때의 kz²=0.75(동그라미)와 kx가 1.5일 때의 −1.25(네모)가 찍혀 있고, 원점 둘레에 180도 호가 있다. 가운데 화살표에 제곱근, 각도 나누기 2라고 적혀 있다. 오른쪽은 네 사분면을 칠한 kz 평면으로, 양의 실수축과 양의 허수축, 곧 1사분면의 경계가 청록 굵은 선이고 90도 호가 있다. 고르는 근 0.866(0도)과 1.118i(90도)는 그 경계 위에 채운 점으로, 반대 근 −0.866과 −1.118i는 맞은편에 빈 점으로 찍혀 있다.
:figclass: learning-figure

그림 A-2. 제곱근은 각도를 절반으로 만듭니다. 무손실 매질에서 실수 {math}`k_x`이면 {math}`k_z^2`은 실수축 위(0°나 180°)에 있고, 그 제곱근은 1사분면의 경계(0°나 90°)에 있는 근과 맞은편의 반대 근 한 쌍이 됩니다. 개념 설명용 그림입니다.
```

그림 A-2의 두 예를 표로 옮기면 이렇습니다.

| 경우 | {math}`k_z^2` (각도) | 고르는 근 (각도) | 반대편 근 |
| --- | --- | --- | --- |
| 공기, {math}`k_x=0.5` | 0.75 (0°) | +0.866 (0°) | −0.866 (180°) |
| 공기, {math}`k_x=1.5` | −1.25 (180°) | +1.118i (90°) | −1.118i (270°) |

```{figure} ../../figures/learn/learn/a03-real-kx.svg
:alt: kz 평면을 네 사분면으로 칠한 그림이고, 오른쪽에 범례와 짧은 설명이 있다. 진한 선은 공기에서 kx를 0에서 2로 키울 때 고른 근이 지나는 길로, kx가 1보다 작을 때는 실수축을 따라 1에서 0으로 오고 kx가 1을 넘으면 허수축을 따라 1.73i까지 올라간다. 곧 1사분면의 경계를 따라간다. 버린 근 −kz의 길은 그 맞은편, 음의 실수축과 음의 허수축에 흐린 점선으로 그려져 있다.
:figclass: learning-figure

그림 A-3. 무손실 매질에서 실수 {math}`k_x`에서 고른 근은 실수축과 허수축, 곧 1사분면의 경계를 따라갑니다. `tools/docs/learning/draw_complex_kx_roots.py`로 계산해 그렸습니다.
```

**경계에서는 비깁니다.** 무손실 매질이면 근이 1사분면의 경계에 놓이므로, 한 기준은 두 근을 가르지 못합니다.

- 전파(0°)는 근이 실수축 위라 허수부가 정확히 0입니다. "줄어듦"만으로는 두 근이 비기므로 "나감"으로 {math}`+0.866`을 고릅니다.
- 소멸(180°)은 거꾸로입니다. 근이 허수축 위(90°)라 실수부가 정확히 0입니다. "나감"만으로는 비기므로 "줄어듦"으로 {math}`+1.118i`를 고릅니다.

어느 쪽이든 두 기준이 서로 다른 근을 가리키지는 않습니다. 한 기준이 비기면 다른 기준이 정합니다. 소멸의 비김은 뒤의 {ref}`실수 kx에서 따라가기가 비기는 경우 <ckr-tie>`에서, 전파의 비김은 {ref}`줄어드는 근만 고르는 규칙 <ckr-branch-cut>`에서 다시 나옵니다. 전파와 소멸의 구분은 [12장](results.md)과 같습니다.

(ckr-complex-kx)=
### 복소 kx에서는 무엇이 달라지나

**복소 {math}`k_x`는 어디서 오나.** Wavebit에서는 복소 `theta`에서만 옵니다({ref}`8장 <complex-transverse-wavevector>`). 실수 `theta`는 빛이 법선에서 기울어진 각도이고, 그 방향에서 {math}`k_x=n\sin\theta\cos\phi`가 나옵니다({math}`\phi`: 방위각). 복소 `theta`에는 이런 뜻이 없습니다. `0.2 + 0.1i` rad만큼 기울어진 방향은 공간에 없기 때문입니다. 그래도 {math}`\sin`은 복소수에도 계산할 수 있으므로, 같은 식에 복소수 `theta`를 그대로 넣어 {math}`k_x`를 구합니다. 이렇게 **원래의 뜻(방향)은 버리고 식의 모양만 빌려 쓰는 것을 '형식적'이라고 합니다.** 이때 `theta`는 방향이 아니라 원하는 복소 {math}`k_x`를 만들어 내는 계산용 숫자입니다. 예를 들어 공기에서 {math}`\theta=0.2+0.1i`이면 {math}`k_x`가 복소수입니다. 반면 Wavebit에 {math}`k_x`를 직접 복소수로 넣으면 거부하고, 복소 `theta`로 가는 법을 안내합니다.

**복소 `theta`를 넣으면 {math}`k_x`가 어떻게 되나.** {ref}`전제 2 <ckr-premises>`로 입사 매질의 {math}`n`도 실수이므로, {math}`k_x`의 허수부는 `theta`의 허수부에서만 생깁니다. 차례로 계산해 보면 이렇습니다.

1. 입사면을 {math}`x`–{math}`z` 평면으로 두면 {math}`\phi=0`, {math}`\cos\phi=1`이므로 {math}`k_x=n\sin\theta`입니다.
2. `theta`를 실수부 {math}`a`와 허수부 {math}`b`로 나눠 {math}`\theta=a+ib`로 씁니다. 복소수의 사인과 코사인은 지수 함수로 정의합니다.

   ```{math}
   \sin z=\frac{e^{iz}-e^{-iz}}{2i},\qquad \cos z=\frac{e^{iz}+e^{-iz}}{2}.
   ```

   여기에 {math}`z=ib`를 넣으면 쌍곡선 함수 {math}`\cosh b=(e^{b}+e^{-b})/2`, {math}`\sinh b=(e^{b}-e^{-b})/2`가 나옵니다. 곧 {math}`\cos(ib)=\cosh b`, {math}`\sin(ib)=i\sinh b`입니다. 사인의 덧셈 공식은 복소수에서도 그대로 성립하므로, 두 값을 넣으면 다음과 같습니다.

   ```{math}
   \sin(a+ib)=\sin a\cos(ib)+\cos a\sin(ib)=\sin a\cosh b+i\cos a\sinh b .
   ```

   {math}`b`가 작으면 {math}`\cosh b\approx1`, {math}`\sinh b\approx b`입니다.
3. 여기에 {math}`n`을 곱하면 {math}`k_x`의 실수부와 허수부가 따로 나옵니다.

   ```{math}
   \operatorname{Re}k_x=n\sin a\cosh b,\qquad \operatorname{Im}k_x=n\cos a\sinh b .
   ```

여기서 두 가지가 나옵니다.

- **허수부가 작으면 {math}`k_x`도 조금만 움직입니다.** {math}`b`가 작으면 {math}`\operatorname{Re}k_x\approx n\sin a`로 실수 `theta`의 값 그대로이고, {math}`\operatorname{Im}k_x\approx n\cos a\cdot b`로 {math}`b`에 비례해서 조금만 생깁니다. {math}`n=2`이면 `theta = 0.4`에서 {math}`k_x/k_0=0.778837`, `theta = 0.4 + 1e-12j`에서 {math}`0.778837+1.8\times10^{-12}i`입니다({math}`2\cos0.4\approx1.8`). 곧 허수부가 작은 복소 `theta`는 실수 `theta`를 살짝 비튼 입력입니다.
- **실수부가 π/2이면 허수부가 얼마이든 {math}`k_x`가 실수입니다.** {math}`\cos(\pi/2)=0`이라 {math}`\operatorname{Im}k_x=0`이고, {math}`\sin(\pi/2)=1`이라 {math}`\operatorname{Re}k_x=n\cosh b`입니다. {math}`\theta=\pi/2+i\alpha` 꼴이 그렇고, {math}`\cosh\alpha>1`이므로 {math}`k_x=n\cosh\alpha`는 {math}`n`보다 큰 실수입니다. 그러면 입사 매질 안에서도 {math}`k_z^2=n^2-k_x^2<0`이라 입사파 자체가 소멸파입니다. 이것을 **소멸파 입사**라고 부르며, 복소 `theta`를 넣었는데도 {math}`k_x`가 복소수가 되지 않는 경우입니다.

복소 {math}`k_x`는 입사파의 세기 자체가 표면을 따라 지수적으로 변한다는 뜻입니다. 입사파 {math}`e^{ik_xx}`의 크기가 {math}`e^{-\operatorname{Im}k_x\,x}`이기 때문입니다. 실험실의 빛이 아니라 형식적인 입사입니다. 그래서 `R`, `T`는 계산은 되지만 '입사 전력 중 몇 %'라는 뜻이 없는 형식적인 비율로만 읽습니다. `power_entering`과 `absorption`은 `NaN`이고, 풀이마다 `ComplexIncidenceWarning`이 나옵니다.

**무엇이 달라지나.** {math}`k_x`에 허수부가 있으면 {math}`k_z^2=\varepsilon-k_x^2`에 허수부가 생깁니다. {math}`k_x`를 제곱하면 허수부가 {math}`\operatorname{Im}k_x^2=2\operatorname{Re}k_x\operatorname{Im}k_x`이고, {ref}`전제 2 <ckr-premises>`로 {math}`\varepsilon`은 실수이므로 다음과 같습니다.

```{math}
\operatorname{Im}k_z^2=-\operatorname{Im}k_x^2=-2\operatorname{Re}k_x\operatorname{Im}k_x .
```

{math}`\operatorname{Re}k_x\operatorname{Im}k_x>0`이면 이 값이 음수가 되어 {math}`k_z^2`이 아래쪽 반(180°에서 360° 사이)으로 내려갑니다. 각도를 절반으로 줄이면 한 근은 90°에서 180° 사이(2사분면)에, 부호가 반대인 다른 근은 그 맞은편 270°에서 360° 사이(4사분면)에 놓입니다. 두 근이 **2사분면과 4사분면**에 앉습니다. 반대로 {math}`\operatorname{Re}k_x\operatorname{Im}k_x<0`이면 {math}`k_z^2`이 위쪽 반에 있어서, 두 근이 1·3사분면에 놓입니다. 이때는 1사분면 근이 두 기준을 다 만족하므로, 실수 {math}`k_x`처럼 그 근으로 저절로 정해집니다.

```{figure} ../../figures/learn/learn/a04-complex-pairs.svg
:alt: 위쪽에 kz 평면 두 개, 아래쪽에 x–z 평면 하나가 있다. (a)는 kx=1.5−0.3i로, 두 근 ±(0.393+1.146i)가 1사분면과 3사분면에 있고 한 근이 두 기준을 다 만족한다고 적혀 있다. (b)는 kx=1.5+0.3i로, 두 근 ±(0.393−1.146i)가 4사분면과 2사분면에 있고 나가는 근은 커지고 줄어드는 근은 들어온다고 적혀 있다. 두 그림 모두 kx=1.5일 때의 쌍 ±1.118i가 회색 ×로 찍혀 있다. (c)는 (b)의 나가는 근 kz=0.393−1.146i로 푼 장의 세기를 x–z 평면에 그린 것이다. 가로축은 표면을 따라가는 k0·x(−5에서 3까지), 세로축은 구조에서 떨어진 거리 k0·z(0에서 2.6까지)이고, 배경색은 log10 세기로 옅을수록 약하고 진한 주황일수록 세다. 맨 아래 회색 띠는 구조이며, 표면의 세기는 왼쪽일수록 세다고 적혀 있다(∝exp(−0.6x)). 표면에서 기울기 0.26으로 비스듬히 뻗는 검은 광선 다섯 줄은 색 띠의 경계와 나란하고, 광선을 따라서는 세기가 일정하다고 적혀 있다. x=1.5에는 위쪽 화살표가 달린 세로 점선이 있고, 위로 갈수록 세기가 커지는 것은 더 왼쪽에서 출발한 광선을 만나기 때문이라고 적혀 있다. 이 점선 위 z=0.5와 z=1.5의 파란 점에서 파란 점선이 각자의 광선을 따라 표면의 x=−0.41과 x=−4.23까지 내려가며, 두 끝의 색이 같다.
:figclass: learning-figure

그림 A-4. 복소 {math}`k_x`에서는 두 근이 1·3사분면에 있을 수도, 2·4사분면에 있을 수도 있습니다. 2·4사분면이면 두 기준이 서로 다른 근을 가리킵니다. (c)는 (b)의 나가는 근이 멀어질수록 커지는 까닭을 실제 공간에서 보여 줍니다. 광선을 따라서는 세기가 일정하고, 멀어질수록 표면의 더 왼쪽, 곧 더 센 곳에서 출발한 광선을 만납니다. `tools/docs/learning/draw_complex_kx_roots.py`로 계산해 그렸습니다.
```

그림 A-4의 (a)와 (b)는 공기에서 {math}`k_x=1.5-0.3i`와 {math}`k_x=1.5+0.3i`입니다.

- {math}`k_x=1.5-0.3i`: {math}`\operatorname{Im}k_z^2=+0.9`입니다. 두 근 {math}`\pm(0.393+1.146i)`가 1·3사분면에 있습니다. 1사분면의 {math}`0.393+1.146i`는 나감과 줄어듦을 둘 다 만족하고, 3사분면의 {math}`-0.393-1.146i`는 둘 다 어깁니다. 그러니 1사분면 근으로 저절로 정해집니다.
- {math}`k_x=1.5+0.3i`: {math}`\operatorname{Im}k_z^2=-0.9`입니다. 두 근 {math}`\pm(0.393-1.146i)`가 2·4사분면에 있습니다. 나가는 근({math}`0.393-1.146i`)은 커지고, 줄어드는 근({math}`-0.393+1.146i`)은 들어옵니다. **기준 하나를 버려야 합니다.**

**커지는 근이 곧 틀린 답은 아닙니다.** 그림 A-4(b)의 나가는 근 {math}`k_z=0.393-1.146i`로 장 {math}`e^{i(k_xx+k_zz)}`를 풀어 봅니다. 여기서 {math}`z`는 앞의 {math}`d`처럼 구조에서 떨어진 거리이며, 8장의 전역 {math}`z`와는 다릅니다. 파수를 벡터 {math}`\mathbf k=(k_x,\ k_z)`로 보고 실수부와 허수부로 나누면 다음과 같습니다.

```{math}
\operatorname{Re}\mathbf k=(1.5,\ 0.393),\qquad \operatorname{Im}\mathbf k=(0.3,\ -1.146).
```

1. **세기는 {math}`\operatorname{Im}\mathbf k` 방향으로만 변합니다.** 세기는 진폭의 제곱 {math}`|E|^2`이며, 전력 밀도에 비례합니다.

   ```{math}
   \left|e^{i(k_xx+k_zz)}\right|^2=e^{-2(\operatorname{Im}k_x\,x+\operatorname{Im}k_z\,z)} .
   ```

   지수에는 파수의 허수부만 들어 있습니다. 그래서 세기는 {math}`\operatorname{Im}\mathbf k` 방향으로 갈수록 줄고, 그에 수직인 방향으로는 변하지 않습니다. 이 예에서는 표면을 따라 {math}`+x`로 갈수록 {math}`e^{-0.6x}`로 줄고, {math}`\operatorname{Im}k_z`가 음수라 구조에서 멀어질수록 커집니다.
2. **파는 {math}`\operatorname{Re}\mathbf k` 방향으로 갑니다.** 위상이 나아가는 방향이 {math}`\operatorname{Re}\mathbf k=(1.5,\ 0.393)`입니다. 거의 {math}`x` 방향으로 누워서, 구조에서 조금씩 멀어지는 파입니다.
3. **두 방향은 서로 수직입니다.** 두 벡터가 수직인지는 내적이 0인지로 봅니다.

   ```{math}
   \operatorname{Re}\mathbf k\cdot\operatorname{Im}\mathbf k
   =\operatorname{Re}k_x\operatorname{Im}k_x+\operatorname{Re}k_z\operatorname{Im}k_z
   =1.5\times0.3+0.393\times(-1.146)\approx0.45-0.45=0 .
   ```

   반올림하지 않은 값으로 계산하면 정확히 0입니다. 이 예에서만 우연히 그런 것이 아닙니다. 분산 관계 {math}`k_x^2+k_z^2=\varepsilon`의 양변에서 허수부만 뽑으면, 복소수 제곱의 허수부가 {math}`\operatorname{Im}(a^2)=2\operatorname{Re}a\operatorname{Im}a`이므로 다음이 됩니다.

   ```{math}
   2\,(\operatorname{Re}k_x\operatorname{Im}k_x+\operatorname{Re}k_z\operatorname{Im}k_z)=\operatorname{Im}\varepsilon .
   ```

   괄호 안이 바로 위의 내적이고, 무손실 매질에서는 {math}`\operatorname{Im}\varepsilon=0`입니다. 그래서 무손실 매질에서는 {math}`\operatorname{Re}\mathbf k`와 {math}`\operatorname{Im}\mathbf k`가 늘 수직입니다. 1과 합치면, 파가 가는 방향으로는 세기가 변하지 않습니다. 파를 따라 한 줄기(광선)를 타고 가면 세기가 일정하고, 세기는 광선과 광선 사이에서만 다릅니다.
4. **그래서 멀어질수록 커집니다.** 한 곳 {math}`x_0`에서 구조로부터 곧게 멀어지며 세기를 재 봅니다. 광선은 {math}`z`가 1 늘 때 {math}`x`가 {math}`1.5/0.393=3.82`만큼 느는 기울기로 뻗으므로, 높이 {math}`z`에서 만나는 광선은 표면의 {math}`x_0-3.82z`에서 출발한 광선입니다. 그런데 표면의 세기는 이미 왼쪽일수록 셉니다(1의 {math}`e^{-0.6x}`). 그러니 멀어질수록 더 왼쪽의, 더 센 곳에서 출발한 광선을 만납니다. 커지는 비율도 맞습니다. 높이가 1 늘면 출발점이 3.82만큼 왼쪽으로 가므로 세기가 {math}`e^{0.6\times3.82}=e^{2.29}`배가 됩니다. 1의 식으로 곧장 계산한 값 {math}`e^{2\times1.146}=e^{2.29}`배와 같습니다.

곧 {math}`z` 방향으로 커지는 것은 무언가가 빛을 키워서가 아니라, 표면을 따라 {math}`x` 방향으로 줄어드는 세기가 비스듬한 광선을 따라 비쳐 보이는 것입니다(그림 A-4(c)). 입사파부터 표면을 따라 세기가 변하는 형식적인 파라서 생기는 일이므로, 차수들이 {math}`z` 방향으로 커지는 것은 이상하지 않습니다([시간 규약 감사](../../validation/rcwa/time-convention-audit.md)의 2절).

그래서 "줄어듦"은 이 문제의 판정 기준이 되지 못합니다. 그렇다고 "나감"만 따르면, 곧 {math}`\operatorname{Re}k_z>0`인 근(**주근**)을 늘 고르면 되는 것도 아닙니다.

**그러면 무엇으로 판정하나.** 이 부록의 잣대는 **연속성**입니다. 실제 상황(실수 {math}`k_x`)에서 살짝 벗어나면 답도 살짝만 변해야 합니다. 근거는 복소 {math}`k_x`가 무엇이냐에 있습니다. 실제 광원이 만드는 장을 표면 방향으로 푸리에 분해하면 실수 {math}`k_x` 성분만 나오고, 그 성분마다 근은 앞 절처럼 저절로 정해집니다. 복소 {math}`k_x`는 그런 성분이 아니라, 실수 {math}`k_x`에서 정해진 답을 복소수 쪽으로 넓혀 본 형식적인 계산입니다. 그러니 넓힌 답은 실수축에 다가갈수록 실수 {math}`k_x`의 답으로 이어져야 합니다([시간 규약 감사](../../validation/rcwa/time-convention-audit.md)의 3절). 입사 매질이 무손실이면 앞에서 본 대로 `theta`의 허수부가 작을 때 {math}`k_x`도 조금만 움직이므로, 이 잣대는 `theta`로 말해도 같습니다. 주근이 이 잣대를 어기는 자리는 뒤의 {ref}`실수 kx에서 따라가기가 비기는 경우 <ckr-tie>`와 {ref}`두 가지 훑기 <ckr-sweeps>`에서 봅니다.

(ckr-branch-point)=
### branch point란 무엇이고, 왜 문제인가

**무엇을 고르나.** {math}`k_x` 하나마다 {math}`k_z`의 후보는 {math}`+k_z`와 {math}`-k_z` 둘이고, 계산에는 그중 하나만 씁니다. 그래서 {math}`k_x`가 주어질 때마다 둘 중 어느 것을 반공간의 파로 쓸지 정해야 합니다. 이 부록에서 "근을 고른다"는 말은 모두 이 뜻입니다. 앞 절의 연속성 잣대는 이 고르기에 조건을 하나 붙입니다. {math}`k_x`가 조금 변하면 고른 {math}`k_z`도 조금만 변해야 합니다. 그렇게 고르는 방법은 간단합니다. {math}`k_x`를 조금씩 옮기면서, 매번 두 후보 가운데 바로 전에 고른 값에 더 가까운 쪽을 고르면 됩니다. 이것을 "끊기지 않게 따라간다"고 부르겠습니다.

**두 근이 만나는 점.** 공기({math}`\varepsilon=1`)에서 실수 {math}`k_x`를 0부터 키우며 두 후보를 보면 이렇습니다.

| {math}`k_x` | 두 후보 {math}`\pm\sqrt{1-k_x^2}` |
| --- | --- |
| 0 | ±1 |
| 0.9 | ±0.436 |
| 0.99 | ±0.141 |
| 1 | 둘 다 0 |
| 1.01 | ±0.142i |
| 1.5 | ±1.118i |

두 후보는 부호만 반대라 늘 원점을 사이에 두고 마주 봅니다. {math}`k_x`가 1에 다가가면 {math}`k_z^2=1-k_x^2`이 0에 다가가므로 두 후보가 원점 쪽으로 모여 서로 가까워지고, {math}`k_x=1`에서는 둘 다 0, 곧 **같은 수**가 됩니다. 1을 넘으면 허수축 위에서 다시 위아래로 벌어집니다. "두 근이 한 점에서 만난다"는 말은 이렇게 두 후보가 0에서 하나로 겹친다는 뜻입니다.

일반적으로 {math}`k_z^2=n^2-k_x^2`이므로 {math}`k_x=\pm n`에서 이런 일이 생기고, 이 {math}`k_x`를 **branch point**(분기점, 가지점)라고 부릅니다. {ref}`전제 2 <ckr-premises>`로 {math}`n`이 양의 실수이므로 branch point는 실수축 위에 있습니다. {math}`k_z=0`이면 파가 구조에서 멀어지지 않고 계면을 따라 나란히 달리므로, 격자에서는 차수 하나가 막 생기거나 사라지는 Rayleigh 점입니다([19장](convergence.md)).

**왜 문제인가.** 두 가지 때문입니다.

1. **가까우면 헷갈립니다.** branch point에서 멀면 두 후보가 멀리 떨어져 있어서, 바로 전 값에 가까운 쪽이 어느 것인지 분명합니다. branch point에 가까울수록 두 후보가 붙어 있어 헷갈리기 쉽고, branch point 위에서는 아예 같아집니다. 실수축을 따라 branch point를 정확히 지나가면 가까운 쪽을 정할 수 없는 경우가 {ref}`뒤에서 <ckr-tie>` 나옵니다.
2. **한 바퀴 돌면 뒤바뀝니다.** {math}`k_x`를 복소평면에서 branch point를 감싸는 동그라미를 따라 한 바퀴 옮기면서 끊기지 않게 따라가면, 출발한 {math}`k_x`로 돌아왔을 때 처음에 고른 후보가 아니라 **다른 후보**를 쥐게 됩니다. {math}`k_x` 값 하나가 입사 조건 하나이므로, 입사 조건을 조금씩 바꿔 한 바퀴 돌아 제자리로 오는 계산 실험입니다. 복소 {math}`k_x`를 지나므로 실험실이 아니라 계산으로만 할 수 있습니다.

```{figure} ../../figures/learn/learn/a05-loop.svg
:alt: 왼쪽은 kx 평면이다. branch point kx=1을 감싸는 빨간 고리 A(중심 1, 반지름 0.35, 0.65에서 출발)와 감싸지 않는 파란 고리 B(중심 0.3, 반지름 0.2)가 있다. 오른쪽은 kz 평면에서 따라간 근의 궤적이다. 고리 A는 +0.760에서 출발해 위쪽으로 반원을 그리며 −0.760에 도착하고, 고리 B는 0.995 근처의 작은 고리를 그리며 제자리로 돌아온다.
:figclass: learning-figure

그림 A-5. branch point를 감싸는 고리를 한 바퀴 돌면 근이 반대로 바뀝니다. 감싸지 않는 고리는 제자리로 돌아옵니다. `tools/docs/learning/draw_complex_kx_roots.py`로 계산해 그렸습니다.
```

그림 A-5가 둘째 성질을 보여 주는 실험입니다(공기, branch point {math}`k_x=1`).

- 고리 A(빨강)는 branch point를 감쌉니다. {math}`k_x=0.65`에서 후보 {math}`+0.760`을 골라 출발해 한 바퀴 돌면, 같은 {math}`k_x=0.65`에서 {math}`-0.760`을 쥐고 돌아옵니다.
- 고리 B(파랑)는 branch point를 감싸지 않습니다. {math}`+0.995`를 골라 출발해 {math}`+0.995`로 돌아옵니다.

고리 A를 90°씩 네 걸음으로 나눠 보면 이렇습니다. 매 걸음 두 후보 가운데 바로 전 값에 가까운 쪽을 골랐을 뿐인데, 고른 근이 원점 둘레를 조금씩 돌아 반대편에 닿습니다.

| 고리를 돈 각도 | {math}`k_x` | 두 후보 | 따라간 근 | 따라간 근의 각도 |
| --- | --- | --- | --- | ---: |
| 0° (출발) | 0.650 | ±0.760 | +0.760 | 0° |
| 90° | 1.000 − 0.350i | ±(0.645 + 0.542i) | +0.645 + 0.542i | 40° |
| 180° | 1.350 | ±0.907i | +0.907i | 90° |
| 270° | 1.000 + 0.350i | ±(0.645 − 0.542i) | **−0.645 + 0.542i** | 140° |
| 360° (제자리) | 0.650 | ±0.760 | **−0.760** | 180° |

까닭은 "각도 절반"과 같습니다.

1. {math}`k_z^2=(n-k_x)(n+k_x)`입니다. {math}`k_x`가 branch point {math}`n` 근처에 있으면 {math}`(n+k_x)\approx2n`은 거의 그대로이므로, {math}`k_z^2`은 {math}`(k_x-n)`에 비례합니다.
2. {math}`k_x`가 branch point 하나만 감싸는 고리를 한 바퀴(360°) 돌면 {math}`k_z^2`도 0을 한 바퀴 돕니다. 두 branch point를 함께 감싸는 고리라면 {math}`(n-k_x)`와 {math}`(n+k_x)`가 한 바퀴씩 돌므로 {math}`k_z^2`은 0을 두 바퀴 돕니다.
3. 제곱근은 각도를 절반으로 만듭니다. {math}`k_z^2`이 한 바퀴 돌면 {math}`k_z`는 반 바퀴(180°)만 돌아서 {math}`-k_z`에 도착하고, {math}`k_z^2`이 두 바퀴 돌아야 {math}`k_z`가 제자리로 옵니다. 그래서 branch point 하나를 감싼 고리는 두 번 돌아야 제자리이고, 두 branch point를 함께 감싼 고리는 한 번만 돌아도 제자리입니다.

비유하면 branch point는 **두 층짜리 나선형 주차장**의 가운데 기둥과 같습니다. 두 근은 1층과 2층입니다. 가운데 기둥을 한 바퀴 돌면 2층에 가고, 2층에서 한 바퀴 더 돌면 다시 1층으로 옵니다. 보통 주차장과 달리 3층이 없습니다. 수학 용어로 두 층은 Riemann 면의 두 장(sheet)입니다.

**그래서.** 같은 {math}`k_x`라도 branch point의 어느 쪽으로 돌아왔느냐에 따라 끊기지 않게 따라간 답이 다릅니다. 곧 "끊기지 않게 따라가기"만으로는 {math}`k_x`마다 답을 하나로 정할 수 없고, 어딘가에서는 반대 후보로 한 번 건너뛰기로 정해 둬야 합니다. 그 자리를 어디에 두느냐가 규칙마다 다르며, {ref}`branch cut <ckr-branch-cut>`에서 다룹니다. 아래 그림에서 직접 돌려 볼 수 있습니다.

```{raw} html
<section class="exercise ckr-exercise" aria-labelledby="ckr-title">
<h3 id="ckr-title">직접 바꿔보기 · branch point를 한 바퀴 돌면?</h3>
<p class="concept">개념 조작 · 공기(ε = 1)에서 두 후보 중 바로 전 값에 가까운 쪽을 따라갑니다. 실제 RCWA 계산은 아닙니다.</p>
<div class="controls">
<label for="ckr-angle">고리를 따라 돈 각도 <output id="ckr-angle-out" for="ckr-angle">0°</output><input id="ckr-angle" type="range" min="0" max="720" step="1" value="0" list="ckr-turns"><datalist id="ckr-turns"><option value="0"></option><option value="360"></option><option value="720"></option></datalist></label>
<fieldset class="ckr-loop"><legend>고리</legend>
<label><input type="radio" name="ckr-loop" id="ckr-loop-a" value="a" checked>고리 A · branch point를 감쌈 (중심 1, 반지름 0.35)</label>
<label><input type="radio" name="ckr-loop" id="ckr-loop-b" value="b">고리 B · 감싸지 않음 (중심 0.3, 반지름 0.2)</label>
</fieldset>
<button id="ckr-play" class="ckr-play" type="button">일시정지</button>
</div>
<div class="svg-scroll"><svg id="ckr-fig" viewBox="0 0 680 340" role="img" aria-label="두 패널입니다. 왼쪽 kx 평면에는 점선 고리, branch point(kx = 1), 출발점(작은 사각형), 지금까지 지나온 길과 지금의 kx 점이 있습니다. 오른쪽 kz 평면에는 지금의 두 후보 ±√(1 − kx²)(속이 빈 원), 따라간 근(채운 점)과 그 자취, 출발한 근(작은 사각형)이 있습니다."></svg></div>
<div class="readout" id="ckr-readout" aria-live="polite" aria-atomic="true">
<p id="ckr-kx">고리 A · 0° · kx = 0.650</p>
<p id="ckr-cands">두 후보 ±√(1 − kx²): +0.760, −0.760</p>
<p id="ckr-roots">따라간 근 +0.760 · 출발한 근 +0.760</p>
<p id="ckr-status" class="ckr-status">출발점입니다. 한 바퀴(360°)를 채우면 출발한 근과 비교합니다.</p>
</div>
<noscript><p>JavaScript를 켜면 고리를 돌면서 두 후보 중 어느 쪽을 따라가는지 볼 수 있습니다.</p></noscript>
</section>
```

(ckr-continuation)=
### "따라가기"는 어떻게 근을 정하나

앞 절에서 본 대로, 끊기지 않게 따라간 답은 어느 길로 왔느냐에 따라 달라질 수 있습니다. 그렇다면 출발점과 길을 미리 하나로 정해 두면 {math}`k_x`마다 답이 하나로 정해집니다. 이것이 **따라가기** 규칙입니다.

1. 답이 **확실한 곳**에서 출발합니다.
2. {math}`k_x`를 목적지까지 **조금씩** 옮깁니다.
3. 매 걸음 **바로 전 값에 더 가까운 근**을 고릅니다. 갑자기 반대편으로 튀지 않게 합니다.

목적지에서 손에 든 근이 답입니다. 규칙은 **출발점**을 어디로 잡느냐로 갈립니다.

- **경로 D**: 수직 입사 {math}`k_x=0`에서 출발해 {math}`k_x` 평면의 곧은 선으로 목적지까지 갑니다. 수직 입사에서는 {math}`k_z=n`이므로 이견이 없습니다. **2026년 10월 2일 전의 Wavebit 규칙**입니다.
- **경로 E**: 목적지와 실수부가 같은 실수 {math}`k_x`, 곧 **가장 가까운 실제 상황**에서 출발해 허수부만 목적지 값까지 바꿉니다.

D·E는 이 부록에서 붙인 이름입니다. [시간 규약 감사](../../validation/rcwa/time-convention-audit.md)와 [분기 처리](../../theory/rcwa/branch-cuts.md)에도 알파벳으로 부르는 경로가 나오지만, 같은 글자라도 뜻이 다릅니다. 두 문서의 "path D"는 가장 가까운 실수 {math}`k_x`에서 이어 붙이는 길이라 이 부록의 경로 E에 해당합니다.

```{figure} ../../figures/learn/learn/a06-paths-d-e.svg
:alt: 왼쪽은 kx 평면이다. 경로 D는 원점(수직 입사)에서 목적지 1.5+0.3i까지 곧은 빨간 선이며 branch point kx=1의 위쪽을 지난다. 경로 E는 실수 kx=1.5에서 목적지까지 곧게 올라가는 보라색 선이다. 오른쪽은 kz 평면에서 따라간 근이다. 경로 D는 1에서 출발해 4사분면으로 내려가 0.393−1.146i(나감·커짐)에 도착하고, 경로 E는 +1.118i에서 출발해 2사분면의 −0.393+1.146i(들어옴·줄어듦)에 도착한다. 두 도착점은 서로 반대 근이다.
:figclass: learning-figure

그림 A-6. 출발점이 다른 두 길은 같은 목적지에서 서로 반대 근에 도착합니다. 목적지의 허수부 0.3은 보기 쉽게 크게 잡은 값입니다. `tools/docs/learning/draw_complex_kx_roots.py`로 계산해 그렸습니다.
```

그림 A-6은 목적지 {math}`k_x=1.5+0.3i`입니다. 허수부가 0.05나 {math}`10^{-9}`이어도 같은 일이 일어납니다.

- 경로 D(빨강)는 branch point의 **위쪽**을 지나 {math}`0.393-1.146i`에 도착합니다(나감·커짐).
- 경로 E(보라)는 branch point를 지나지 않고 {math}`-0.393+1.146i`에 도착합니다(들어옴·줄어듦).

두 답은 서로 **반대 근**입니다. 까닭은 앞 절의 고리입니다. 경로 E가 출발하는 값, 곧 실수 {math}`k_x=1.5`의 답 {math}`+1.118i`는 원점에서 branch point의 **아래쪽**으로 돌아 {math}`k_x=1.5`에 온 값과 같습니다({ref}`다음 절 <ckr-tie>`의 그림 A-7). 아래쪽({math}`\operatorname{Im}k_x<0`)을 지나는 동안은 {math}`\operatorname{Re}k_x\operatorname{Im}k_x<0`이라, 앞에서 본 대로 {math}`k_z^2`이 위쪽 반에 있고 근은 1사분면에 머뭅니다. 그래서 근은 1사분면 안에서 0°({math}`k_x=0`의 {math}`+1`)에서 90°로 돌아 {math}`+1.118i`에 닿습니다. 경로 D는 branch point의 **위쪽**으로 지나갑니다. 그러니 두 길을 이으면 branch point를 한 바퀴 감싸는 고리가 되고, 고리를 한 바퀴 돌면 근이 바뀌므로 두 도착점이 서로 반대 근입니다.

**경로 D는 한 걸음씩 걷지 않아도 됩니다.** 무손실 유전체에서는 경로 D가 곧 주근({math}`\operatorname{Re}k_z>0`인 근)이라서 주근을 바로 쓰면 됩니다. 2026년 10월 2일 전의 Wavebit도 그렇게 했습니다. 까닭을 두 단계로 보면 이렇습니다.

1. **주근은 {math}`k_z^2`의 음의 실수축에서만 끊깁니다.** 주근은 {math}`k_z^2`의 각도를 −180°에서 180° 사이로 재고 그 절반, 곧 −90°에서 90° 사이의 각도를 근에 주는 규칙입니다. 그래서 근은 늘 오른쪽 반에 있습니다. {math}`k_z^2`이 음의 실수축을 위에서 아래로 건너면 각도가 180°에서 −180°로 넘어가고, 주근의 각도는 90°에서 −90°로 뛰어 반대 근이 됩니다. 다른 곳에서는 매끄럽게 이어집니다.
2. **경로 D는 그 축을 건너지 않습니다.** 경로 D는 {math}`k_x` 평면에서 0부터 목적지까지 가는 곧은 길이고, 그 위의 점은 목적지 {math}`k_x`에 0에서 1까지 커지는 수 {math}`s`를 곱한 {math}`s\,k_x`입니다. 이 길을 {math}`k_z^2` 평면으로 옮기면, {math}`t=s^2`으로 두어 다음과 같습니다.

   ```{math}
   k_z^2(t)=\varepsilon-t\,k_x^2,\qquad t:0\to1 .
   ```

   {math}`\varepsilon`에서 출발해 {math}`\varepsilon-k_x^2`에 닿는 곧은 선분입니다. 이 선분의 허수부는 {math}`-t\operatorname{Im}k_x^2`라서, {math}`\operatorname{Im}k_x^2\ne0`이면 출발점({math}`t=0`)에서만 0입니다. 곧 선분이 실수축에 닿는 곳은 출발점 하나뿐이고, 출발점 {math}`\varepsilon`은 양의 실수입니다. 그러니 선분은 음의 실수축을 건너지 않고, 주근이 끊기지 않고 이어집니다.

남는 경우와 2026년 10월 2일 전 코드의 사정은 이렇습니다.

- {math}`\operatorname{Im}k_x^2=0`이면 선분이 실수축 위에 눕습니다. {math}`k_x`가 실수이고 {math}`n`보다 크면 도중에 branch point({math}`k_z^2=0`)를 밟아 따라가기가 비깁니다. 이 비김은 {ref}`다음 절 <ckr-tie>`에서 다루며, 그때의 Wavebit은 거기에 따로 정한 **비김 규칙**을 썼습니다.
- 그래서 무손실 유전체에서 **2026년 10월 2일 전의 Wavebit = 주근 + 비김 규칙**({math}`|\operatorname{Re}k_z|\le10^{-12}`이면 {math}`\operatorname{Im}k_z>0` 쪽)이었습니다.
- 그때의 코드에는 선분이 음의 실수축을 건널 때 주근을 {math}`-k_z`로 뒤집는 **연속성 보정**(`_off_principal_branch`)도 들어 있었지만, 무손실 유전체에서는 위의 까닭으로 한 번도 켜지지 않았습니다. 이 보정은 규칙을 Maystre로 바꾸면서 지웠습니다.

그리는 스크립트는 경로 D를 걷지 않고 한 번에 계산하는 식, 곧 주근에 연속성 보정을 더한 **경로 D의 닫힌 식**을 씁니다. 이 식이 한 걸음씩 따라간 결과와 같은지 따로 확인하고, 무손실 유전체 셋({math}`\varepsilon=1,\ 2.25,\ 4`)의 격자와 무작위 점, 모두 834603개에서 주근과 같은지(보정이 한 번도 켜지지 않는지) 확인한 뒤에 그림을 그립니다. Wavebit의 `boundary_kz`를 {ref}`뒤에서 <ckr-branch-cut>` 쓰는 무작위 무손실 표본의 복소 {math}`k_x` 288개에서 재면, 2026년 10월 2일 전에는 288개 모두 주근과 같았고, 지금은 288개 모두 Maystre 규칙과 같습니다.

(ckr-tie)=
### 실수 kx에서 경로 D는 왜 답을 못 정하나

두 조건이 겹칠 때 생기는 일입니다.

- 매질이 **무손실**입니다({ref}`전제 2 <ckr-premises>`). 그래야 branch point가 실수축 위({math}`k_x=n`)에 있습니다.
- 목적지가 **실수이고 {math}`k_x>n`인 경우**입니다. 곧 실수 각도에서의 소멸 차수입니다.

경로 D는 0에서 목적지까지 곧은 선입니다. 목적지가 실수이면 길이 실수축을 따라가고, 도중에 {math}`k_x=n`을 **정확히 밟습니다.** branch point 옆을 스치는 것이 아니라 한가운데를 밟고 지나갑니다.

```{figure} ../../figures/learn/learn/a07-through-branch-point.svg
:alt: (a)는 kx 평면에서 원점에서 출발하는 세 경로 D이다. 목적지 1.5+0.3i로 가는 주황 선은 branch point kx=1의 위쪽을, 1.5−0.3i로 가는 청록 선은 아래쪽을 지나고, 실수 1.5로 가는 검은 선은 branch point를 밟고 지난다. (b)는 kz 평면에서 도착한 근이다. 주황은 0.393−1.146i, 청록은 0.393+1.146i에 도착하고, 검은 선은 실수축을 따라 1에서 0까지 온 뒤 +1.118i와 −1.118i로 갈라지는 두 점선과 물음표로 그려져 있다. (c)는 branch point 바로 옆의 확대로, kx=0.9999에서 kz=0.01414이고 kx=1.0001의 두 후보 +0.01414i와 −0.01414i가 모두 거리 0.0200만큼 떨어져 있다.
:figclass: learning-figure

그림 A-7. 경로 D가 branch point를 정확히 밟으면 "가까운 쪽"을 고를 수 없습니다. 목적지의 허수부만 다른 세 경로와 branch point 바로 옆의 확대입니다. `tools/docs/learning/draw_complex_kx_roots.py`로 계산해 그렸습니다.
```

그림 A-7(a)는 목적지의 허수부만 다른 세 경로이고, (b)는 각 경로가 도착한 근입니다.

- 허수부 {math}`+`(주황): branch point **위쪽**으로 지나 {math}`0.393-1.146i`, 커지는 근에 도착합니다.
- 허수부 {math}`-`(청록): branch point **아래쪽**으로 지나 {math}`0.393+1.146i`, 줄어드는 근에 도착합니다.
- 허수부 0(검정): branch point를 **밟습니다.** 그러면 어디로 가야 할까요?

그림 A-7(c)가 branch point 바로 옆을 확대한 것입니다. branch point에서는 두 후보가 모두 0으로 같아집니다.

- 한 걸음 전 {math}`k_x=0.9999`에서 {math}`k_z=0.01414`입니다(실수).
- 한 걸음 뒤 {math}`k_x=1.0001`의 후보는 {math}`+0.01414i`와 {math}`-0.01414i`입니다.
- 둘 다 바로 전 값에서 **정확히 0.0200** 떨어져 있습니다.

"더 가까운 근"이라는 규칙이 결정을 못 합니다. 걸음을 어떻게 잡아도 마찬가지입니다. branch point 앞의 값은 실수이고, 뒤의 두 후보는 {math}`+`순허수와 {math}`-`순허수입니다. 실수 하나에서 두 후보까지의 거리는 늘 정확히 같습니다. 걷는 방식으로는 이 비김을 풀 수 없습니다. {ref}`앞에서 본 "경계에서는 비깁니다" <ckr-real-kx>`가 바로 이것입니다.

결국 비김은 코드의 사소한 세부가 정합니다. 직접 해 보면 부등호 한 글자가 답을 뒤집습니다.

```python
import cmath

root = 1.0                                   # kx = 0 에서 kz = 1
for i in range(1, 3001):
    k = i * 1.5 / 3000                       # 실수축을 따라 1.5까지 (i = 2000에서 branch point)
    v = cmath.sqrt(1 - k * k)
    root = v if abs(v - root) <= abs(-v - root) else -v   # <= 를 < 로 바꿔 보세요
print(root)                                  # 1.118j, < 로 바꾸면 -1.118j
```

그래서 2026년 10월 2일 전의 Wavebit은 이 비김({math}`\operatorname{Re}k_z`가 0인 경우)을 경로 D로 풀지 않고 비김 규칙으로 풀었습니다. 실수부의 크기가 `1e-12` 이하이면 {math}`\operatorname{Im}k_z>0` 쪽, 곧 **줄어드는 근**을 고르는 규칙입니다. 물리적으로 옳은 답이고, 경로 E의 답과 같습니다. 그때의 `boundary_kz`와 지금의 `boundary_kz`에 직접 물어보면 이렇습니다(공기).

| {math}`k_x` | 2026년 10월 2일 전의 {math}`k_z` | 무엇이 정했나 | 지금(Maystre)의 {math}`k_z` |
| --- | --- | --- | --- |
| {math}`1.5-10^{-9}i` | +1.118034i (줄어듦) | branch point 아래쪽으로 지난 경로 D | +1.118034i (줄어듦) |
| {math}`1.5` | +1.118034i (줄어듦) | 비김이라 따로 정한 규칙 | +1.118034i (줄어듦) |
| {math}`1.5+10^{-9}i` | −1.118034i (커짐) | branch point 위쪽으로 지난 경로 D | +1.118034i (줄어듦) |

**그때의 Wavebit은 실수축 위에서는 경로 E처럼 답하고, 조금만 위로 벗어나면 경로 D를 따랐습니다.** 그래서 실수축 바로 위와 아래에서 답이 갈렸습니다. 경로 D는 실수축의 소멸 구간을 위아래로 건너는 순간 반대 근으로 튑니다. 이렇게 건너면 근이 튀는 선을 **branch cut**(분기선)이라고 부르며, {ref}`뒤에서 <ckr-branch-cut>` 자세히 봅니다. 곧 경로 D의 branch cut은 실수축의 소멸 구간 위에 놓여 있습니다. 무손실 유전체의 경로 D는 주근이므로, 이 선은 곧 주근의 선입니다.

지금의 Wavebit은 세 줄 모두 줄어드는 근을 고릅니다. 실수축 위의 소멸 차수도 비김 없이 정해지므로 비김 규칙이 필요 없습니다. Maystre 규칙의 branch cut은 실수축을 branch point에서만 만나기 때문이며, {ref}`뒤에서 <ckr-half-plane-rules>` 봅니다.

**전파 차수**({math}`|k_x|<n`)에서는 이 일이 생기지 않습니다. 길이 branch point에 닿기 전에 끝납니다.

(ckr-half-plane-rules)=
### 한 번에 정하는 규칙은 어떻게 생겼나

따라가기 말고, {math}`k_z` 평면에서 **어느 쪽 반에 있는 근**인지로 바로 고르는 규칙도 있습니다. 원점을 지나는 경계선을 하나 긋고 한쪽을 고르면, 원점 대칭인 두 근 중 정확히 하나가 그쪽에 있습니다. 대표적인 규칙이 셋입니다.

- **주근**: 오른쪽 반({math}`\operatorname{Re}k_z>0`)을 고릅니다. 무손실 유전체에서는 경로 D, 곧 2026년 10월 2일 전의 Wavebit 규칙과 같습니다({ref}`따라가기 <ckr-continuation>`).
- **Maystre 규칙**: −45° 선의 오른쪽 위({math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0`)를 고릅니다. 2026년 10월 2일부터의 Wavebit과 RETICOLO, S4가 이 규칙으로 답합니다.
- **{math}`\operatorname{Im}k_z>0` 규칙**: 위쪽 반, 곧 줄어드는 근만 고릅니다. fmmax가 씁니다.

그림 A-8은 {math}`k_x=1.5+0.3i`의 두 근에 세 규칙을 적용한 것입니다. 이 그림만은 칠한 색이 사분면이 아니라 각 규칙이 고르는 쪽입니다. 색은 규칙을 가리키며(경로 D와 주근은 빨강, Maystre는 파랑, fmmax는 회색, 앞의 경로 E는 보라), 뒤의 그림 A-9와 A-10에서도 선 색은 같은 규칙을 가리킵니다.

```{figure} ../../figures/learn/learn/a08-half-planes.svg
:alt: 세 개의 kz 평면이다. 왼쪽 주근(무손실 매질에서 경로 D, 2026년 10월 2일 전의 Wavebit)은 Re kz>0인 오른쪽 반을 칠하고 경계선은 허수축이며, 고른 근은 0.393−1.146i(나감·커짐)이다. 가운데 Maystre(Wavebit·RETICOLO·S4)는 Re kz+Im kz>0인 쪽을 칠하고 경계선은 −45도 선이며, 고른 근은 −0.393+1.146i(들어옴·줄어듦)이다. 오른쪽 fmmax는 Im kz>0인 위쪽 반을 칠하고 경계선은 실수축이며, 고른 근은 −0.393+1.146i이다. 고른 근은 채운 점, 버린 근은 빈 점이다.
:figclass: learning-figure

그림 A-8. 세 규칙은 {math}`k_z` 평면에 원점을 지나는 경계선 하나를 긋고 한쪽을 고릅니다. 두 근이 경계선 위에 오면 비기고, 근이 경계선을 건너면 고른 근이 반대편으로 바뀝니다. `tools/docs/learning/draw_complex_kx_roots.py`로 계산해 그렸습니다.
```

| 규칙 | 고르는 쪽 | 경계선 | {math}`k_x=1.5+0.3i`에서 고른 근 |
| --- | --- | --- | --- |
| 주근 (= 경로 D, 2026년 10월 2일 전의 Wavebit) | {math}`\operatorname{Re}k_z>0` | 허수축 | 0.393 − 1.146i (나감·커짐) |
| Maystre (Wavebit·RETICOLO·S4) | {math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0` | −45° 선 | −0.393 + 1.146i (들어옴·줄어듦) |
| {math}`\operatorname{Im}k_z>0` (fmmax) | {math}`\operatorname{Im}k_z>0` | 실수축 | −0.393 + 1.146i (들어옴·줄어듦) |

각 솔버의 자세한 사정은 {ref}`코드마다 무엇을 하나 <ckr-codes>`에 모았습니다.

::::{dropdown} 계산식: 각 규칙이 튀는 자리를 kx 평면으로 옮기기
규칙은 근이 경계선 위에 올 때 튑니다. 그 조건을 {math}`k_x^2`로 옮기면 다음과 같습니다({math}`u\ge0`).

```{math}
\begin{aligned}
\text{주근:}\quad & k_z\ \text{가 허수축 위} && \Rightarrow\ k_z^2\le0 && \Rightarrow\ k_x^2=\varepsilon+u,\\
\text{Maystre:}\quad & k_z\ \text{가 }-45^\circ\text{ 선 위} && \Rightarrow\ k_z^2\ \text{가 음의 허수} && \Rightarrow\ k_x^2=\varepsilon+iu,\\
\operatorname{Im}k_z>0:\quad & k_z\ \text{가 실수축 위} && \Rightarrow\ k_z^2\ge0 && \Rightarrow\ k_x^2=\varepsilon-u .
\end{aligned}
```

공기({math}`\varepsilon=1`)에 넣으면 이렇게 됩니다.

- 주근은 {math}`k_x^2=1+u`, 곧 {math}`|k_x|\ge1`인 **실수축**입니다. 앞 절과 같은 결론입니다.
- Maystre는 {math}`\operatorname{Re}k_x^2=1`이면서 {math}`\operatorname{Im}k_x^2\ge0`인 쌍곡선의 반쪽입니다. branch point +1에서 오른쪽 위로, −1에서 왼쪽 아래로 휘어 나갑니다.
- {math}`\operatorname{Im}k_z>0`은 {math}`k_x^2=1-u`, 곧 **{math}`|k_x|\le1`인 실수축과 허수축 전체**입니다.

경로 E는 {math}`k_z` 평면에 경계선을 긋는 규칙이 아니므로 이런 식으로 쓸 수 없습니다. 경로 E의 선은 +n에서 곧게 위로, −n에서 곧게 아래로 뻗습니다. {math}`\operatorname{Re}k_x=n`을 사이에 두고 출발점이 전파 쪽({math}`<n`)에서 소멸 쪽({math}`>n`)으로 바뀌면, 허수부를 양으로 올린 뒤의 근이 서로 반대가 되기 때문입니다. 허수부를 음으로 내린 쪽에서는 두 출발점이 같은 근으로 이어져 선이 없습니다.
::::

(ckr-branch-cut)=
### branch cut은 왜 없앨 수 없나

```{figure} ../../figures/learn/learn/a09-cut-maps.svg
:alt: 공기에서 네 규칙의 kx 평면 지도 네 장이다. 배경색은 그 kx에서 고른 근의 사분면(1사분면 청록, 2사분면 파랑, 4사분면 주황, 3사분면 회색)이고, 굵은 선이 branch cut이다. 경로 D(2026년 10월 2일 전의 Wavebit)는 branch point ±1에서 바깥쪽 실수축(kx>1, kx<−1)에, 경로 E는 +1에서 곧게 위로와 −1에서 곧게 아래로, Maystre(Wavebit·RETICOLO·S4)는 +1에서 오른쪽 위로와 −1에서 왼쪽 아래로 휘는 쌍곡선 반쪽에, fmmax는 두 branch point 사이 실수축(−1<kx<1)과 허수축 전체에 branch cut이 있다.
:figclass: learning-figure

그림 A-9. 네 규칙이 {math}`k_x` 평면에서 고르는 근(색)과 branch cut(굵은 선)입니다(공기). 색 경계라도 굵은 선이 없으면 근이 축을 부드럽게 지나갈 뿐입니다. `tools/docs/learning/draw_complex_kx_roots.py`가 격자에서 실제로 튀는 자리를 찾아 식과 맞대 본 뒤 그렸습니다.
```

그림 A-9는 공기에서 네 규칙, 곧 앞 절의 세 규칙에 경로 E를 더한 넷의 지도입니다. **색**은 그 {math}`k_x`에서 고른 근이 어느 사분면에 있는지이고(사분면 색은 앞과 같음), **굵은 선**은 그 선을 건너면 고른 근이 반대 근({math}`k_z`에서 {math}`-k_z`로)으로 튀는 자리, 곧 branch cut입니다. 선 색은 규칙을 가리킵니다.

**선은 왜 반드시 있나.** 앞에서 본 대로 branch point 하나만 감싸는 고리를 한 바퀴 돌면 근이 바뀝니다. 그런데 규칙은 {math}`k_x` 하나에 답 하나를 정해야 합니다. 그러니 그런 고리 위 어딘가에서 한 번은 반대편으로 튀어야 합니다. 튀는 자리를 이으면 branch point에서 출발하는 선, 곧 branch cut이 됩니다. **branch cut은 없앨 수 없고, 어디에 둘지만 고를 수 있습니다.**

::::{dropdown} 유도: branch cut이 무한까지 뻗어야 하는 이유
두 branch point를 함께 감싸는 고리는 한 바퀴에 두 번 뒤집혀 제자리로 오므로, 수학만 보면 두 branch point를 잇는 짧은 선으로도 됩니다. 예를 들어 {math}`k_z=i\,k_x\sqrt{1-\varepsilon/k_x^2}`가 그렇습니다. 그런데 이 규칙은 {math}`k_x=1.5`에서 {math}`+1.118i`, {math}`k_x=-1.5`에서 {math}`-1.118i`를 줍니다. {math}`k_x`와 {math}`-k_x`는 거울상이라 답이 같아야 합니다. 그러니 물리적인 규칙은 {math}`k_x^2`에만 기대야 합니다.

{math}`w=k_x^2`로 두면 {math}`k_z=\pm\sqrt{\varepsilon-w}`이고, {math}`w` 평면의 branch point는 {math}`w=\varepsilon` 하나뿐입니다. {math}`\varepsilon`을 한 번 감싸는 고리는 아무리 커도 근을 뒤집습니다. 선이 유한한 점에서 끝나면, 그 끝까지 함께 감싸는 큰 고리가 선을 한 번도 건너지 않고 근을 뒤집게 됩니다. 그래서 선은 {math}`\varepsilon`에서 무한까지 가야 합니다. {math}`k_x` 평면으로 옮기면 두 branch point에서 하나씩 무한으로 뻗는 선이 됩니다(경로 D, 경로 E, Maystre).

fmmax의 선도 {math}`w` 평면에서는 {math}`\varepsilon`에서 왼쪽으로 무한까지 가는 반직선({math}`w=\varepsilon-u`) 하나입니다. 다만 이 반직선은 {math}`w=0`을 지납니다. {math}`k_x` 평면으로 옮기면 {math}`0\le w\le\varepsilon` 부분은 두 branch point를 잇는 실수축 구간이 되고, {math}`w<0` 부분은 {math}`k_x=0`에서 위아래로 뻗는 허수축 전체가 됩니다.
::::

| 규칙 | branch cut (공기) | 실제 상황(실수 {math}`k_x`)과의 관계 |
| --- | --- | --- |
| 경로 D (2026년 10월 2일 전의 Wavebit) | branch point에서 바깥으로 실수축 ({math}`k_x>1`, {math}`k_x<-1`) | **모든 소멸 차수가 선 위** |
| 경로 E | +1에서 곧게 위로, −1에서 곧게 아래로 | branch point 한 점에서만 닿음 |
| Maystre (Wavebit·RETICOLO·S4) | 쌍곡선의 반쪽: +1에서 오른쪽 위로, −1에서 왼쪽 아래로 | branch point 한 점에서만 닿음 |
| {math}`\operatorname{Im}k_z>0` (fmmax) | 두 branch point 사이 실수축 ({math}`-1<k_x<1`) + 허수축 | **모든 전파 차수가 선 위** |

**색이 바뀐다고 튀는 것은 아닙니다.** 굵은 선이 없는 색 경계에서는 근이 {math}`k_z` 평면의 축을 **부드럽게 지나갈** 뿐입니다. 예를 들어 {math}`k_x=0.5+0.05i`는 전파 쪽의 복소 {math}`k_x`입니다. 경로 D, 경로 E, Maystre는 모두 {math}`0.8679-0.0288i`를 고릅니다. 살짝 커지는 근이지만, 실수 {math}`k_x=0.5`의 답 {math}`0.866`과 부드럽게 이어집니다. {math}`\operatorname{Im}k_z>0` 규칙만 {math}`-0.8679+0.0288i`(들어오는 파)를 고릅니다. 실수 {math}`k_x=0.5`가 그 규칙의 선 위에 있어서, 허수부를 조금만 줘도 선을 넘기 때문입니다.

지도의 색은 각 규칙이 무엇을 포기하는지도 보여 줍니다.

- 공기(무손실 유전체)에서 경로 D는 {math}`\operatorname{Re}k_z<0`을 절대 고르지 않습니다(청록과 주황뿐). 무손실 유전체의 경로 D는 주근이기 때문입니다.
- {math}`\operatorname{Im}k_z>0` 규칙은 {math}`\operatorname{Im}k_z<0`을 절대 고르지 않습니다(청록과 파랑뿐).
- 경로 E와 Maystre는 자리마다 다릅니다.

**숫자로 보면.** 무손실 매질({math}`\varepsilon`은 1에서 6 사이의 실수)과 실수 {math}`k_x`(0에서 3 사이)를 무작위로 뽑은 쌍 1496개에서는 네 규칙이 모두 **1496개 전부에서 같은 근**을 고릅니다. 실수 {math}`k_x`에서는 근이 저절로 정해진다는 앞의 결론 그대로입니다. 그런데 여기에 {math}`\pm10^{-8}i`만 더해도 근이 뒤집히는 규칙이 있습니다. 뒤집힌 수는 이렇습니다.

| 규칙 | 뒤집힌 수 (1496개 중) | 뒤집힌 차수 |
| --- | ---: | --- |
| 경로 D (2026년 10월 2일 전의 Wavebit) | **597** | 소멸 차수 전부 |
| 경로 E | 0 | |
| Maystre (Wavebit) | 0 | |
| {math}`\operatorname{Im}k_z>0` (fmmax) | **899** | 전파 차수 전부 |

1496개 가운데 소멸 차수가 597개, 전파 차수가 899개입니다. 곧 경로 D는 소멸 차수를 하나도 빠짐없이, fmmax는 전파 차수를 하나도 빠짐없이 뒤집습니다.

경로 E와 Maystre는 실수축 근처에서 **거의 같은 규칙**입니다. 같은 무손실 표본 가운데 288개에 허수부(−0.5에서 0.5 사이)를 붙여 복소 {math}`k_x`로 만들고, 각 규칙이 경로 D·경로 E를 한 걸음씩 따라간 결과와 같은 근을 고른 수를 세면 이렇습니다.

| 규칙 | 경로 D와 같음 | 경로 E와 같음 |
| --- | ---: | ---: |
| 경로 D (2026년 10월 2일 전의 Wavebit) | 288/288 | 238/288 |
| Maystre (Wavebit) | 240/288 | **286/288** |
| {math}`\operatorname{Im}k_z>0` (fmmax) | 130/288 | 180/288 |

(ckr-sweeps)=
### branch cut의 자리가 실제로 무엇을 바꾸나

두 가지로 훑어 보면 드러납니다.

```{figure} ../../figures/learn/learn/a10-sweeps.svg
:alt: (a)는 branch point kx=1 근처의 kx 평면으로, 경로 D의 branch cut(실수축의 kx>1 부분), 경로 E의 branch cut(1에서 곧게 위), Maystre의 branch cut(1에서 휘어 올라가는 곡선)과 두 훑기가 그려져 있다. 훑기 A는 허수부 0.05에서 실수부를 0.8에서 1.2로 옮기는 가로 화살표, 훑기 B는 실수부 1.5에서 허수부를 −0.05에서 +0.05로 옮기는 세로 화살표이다. (b)는 훑기 A에서 각 규칙의 Im kz로, 경로 D는 매끄럽게 −0.07에서 −0.67로 내려가고 경로 E와 Maystre는 Re kx≈1에서 −0.22에서 +0.22로 뛴다. (c)는 훑기 B에서의 Im kz로, 경로 D만 허수부 0에서 +1.118에서 −1.118로 뛰고 경로 E와 Maystre는 +1.118로 매끄럽다.
:figclass: learning-figure

그림 A-10. 복소 각도로 각도를 훑을 때(훑기 A)는 경로 E와 Maystre가 Rayleigh 점 근처에서 한 번 튀고, 실제 상황을 살짝 비틀 때(훑기 B)는 경로 D만 튑니다. `tools/docs/learning/draw_complex_kx_roots.py`로 계산해 그렸습니다.
```

**훑기 A**: 허수부를 0.05로 고정하고 실수부를 0.8에서 1.2로 옮깁니다. 복소 각도의 허수부를 고정한 채 각도를 훑는 상황이며, 이 길은 branch point **위쪽**을 가로지릅니다.

- 경로 D는 튀지 않습니다. 경로 D의 branch cut은 실수축에 있어서 이 길과 만나지 않습니다.
- 경로 E는 {math}`\operatorname{Re}k_x=1.0000`과 {math}`1.0005` 사이에서 튑니다({math}`0.226-0.221i\to-0.225+0.222i`).
- Maystre는 {math}`1.0010`과 {math}`1.0015` 사이에서 튑니다. 정확한 자리는 쌍곡선 위의 {math}`\sqrt{1+0.05^2}=1.00125`입니다. 그림 A-10(a)처럼 branch point 근처에서는 경로 E의 선과 거의 겹칩니다.

**훑기 B**: 실수부를 1.5로 고정하고 허수부를 −0.05에서 +0.05로 옮깁니다. 실제 상황({math}`k_x=1.5`)을 살짝 비트는 상황입니다.

- 경로 D만 허수부 0에서 튑니다({math}`+1.118i\to-1.118i`).
- 경로 E와 Maystre는 매끄럽습니다.

정리하면 이렇습니다.

- branch cut을 실수축 밖으로 빼면(경로 E, Maystre), 각도를 훑다가 **Rayleigh 점을 지날 때** 한 번 튑니다. Rayleigh 점은 실수 {math}`k_x`에서도 원래 결과가 꺾이는 자리(Wood 이상)이므로, 새 특이점이 생기는 것이 아니라 튀는 자리가 원래 있던 특이점 바로 옆에 놓이는 셈입니다.
- branch cut이 실수축에 있으면(경로 D) 실수축의 소멸 구간 전체가 "살짝만 비틀어도 튀는" 자리가 됩니다. 격자 계산에는 높은 차수가 늘 소멸 차수로 들어가므로, 거의 모든 격자 계산에서 어떤 차수가 이 선 위에 있습니다. {ref}`실제 격자 <ckr-grating>`에서 확인합니다.
- Wavebit이 2026년 10월 2일에 경로 D에서 Maystre 규칙으로 바꾼 까닭이 여기에 있습니다. 훑기 A의 튐은 Rayleigh 점 바로 옆에 남지만, 훑기 B의 튐은 없어집니다.

## 2편. Wavebit과 다른 솔버

(ckr-wavebit-rule)=
### Wavebit의 규칙과 주의점

지금까지의 내용을 Wavebit의 동작으로 정리하면 이렇습니다(2026년 10월 2일부터).

- **규칙**: 반공간의 {math}`k_z`는 **Maystre 규칙**({math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0`)입니다. 무손실 유전체의 실수 {math}`k_x`에서는 주근과 같습니다. 2026년 10월 2일 전에는 경로 D였습니다.
- **실수 {math}`k_x`**: 네 규칙이 모두 같은 근을 고르므로, 근 선택 때문에 다른 솔버와 결과가 갈리지는 않습니다.
- **복소 `theta`와 무손실 반공간의 소멸 차수**: Maystre의 branch cut은 실수축을 Rayleigh 점에서만 만납니다. 그래서 `theta`의 허수부를 조금 줘도 소멸 차수의 근이 뒤집히지 않습니다. {ref}`실제 격자 <ckr-grating>`에서 경로 D(2026년 10월 2일 전의 Wavebit)는 허수부 1e-9에도 튀지만, 지금의 Wavebit은 RETICOLO와 같은 값을 냅니다. Rayleigh 점 바로 옆에서는 여전히 한 번 튑니다({ref}`두 가지 훑기 <ckr-sweeps>`). 복소 `theta`는 형식적인 계산이라는 점({ref}`8장 <complex-transverse-wavevector>`)은 그대로입니다.
- **`UnresolvedBranchWarning`**: 부동소수점 정밀도(특히 complex64) 때문에 근을 고르는 값의 부호를 판단할 수 없을 때 나옵니다([분기 처리](../../theory/rcwa/branch-cuts.md)의 4절과 8절). 경로 D에서는 이 문제가 branch cut과 이어져 있었습니다. complex64가 {math}`k_x`에 남기는 작은 허수부 때문에, 실수축 위에 놓인 경로 D의 선을 건넜기 때문입니다. 지금은 실수 {math}`k_x`에서 반공간의 근 때문에 이 경고가 나오지 않고, 층 모드의 라벨 때문에만 나옵니다.

더 자세한 배경은 [분기 처리](../../theory/rcwa/branch-cuts.md)와 [시간 규약 감사](../../validation/rcwa/time-convention-audit.md)에 있습니다.

(ckr-codes)=
### 코드마다 무엇을 하나

지금까지 본 규칙이 어느 코드에 들어 있는지 모으면 이렇습니다. 경로 E를 쓰는 코드는 없습니다.

| 코드 | 근 규칙 | 복소 {math}`k_x` 입력 | branch cut (무손실) |
| --- | --- | --- | --- |
| Wavebit | Maystre(2026년 10월 2일부터). 그 전에는 경로 D, 무손실 유전체에서는 주근 + 비김 규칙 | 직접 입력은 거부하고 안내, 복소 `theta`로만 | 쌍곡선의 반쪽(그 전에는 실수축 소멸 구간) |
| RETICOLO V10 | 평면파에 Maystre, `retsqrt(x, -1)` | `beta0 = n sin(theta)`를 직접 준다 | 쌍곡선의 반쪽 |
| S4 | 실행 결과로는 Maystre와 같음 | 받지 않는다 | Maystre와 같음 |
| fmmax 1.7.1 | {math}`\operatorname{Im}k_z>0`, `_select_eigenvalues_sign` | `in_plane_wavevector`를 복소 그대로 쓴다 | 실수축 전파 구간 + 허수축 |

**RETICOLO.** 반공간의 평면파에 `retsqrt(x, -1)`을 씁니다. `retsqrt`의 셋째 인자 `tet`는 {math}`k_z^2` 평면에서 제곱근을 끊는 선의 각도이고, 기본값 π/2에서는 그 선이 {math}`k_z^2`의 음의 허수축에 놓입니다. 소스 주석은 이것을 "coupure de Maystre"라고 부르며, 결과는 Maystre 규칙과 같습니다. `tet`를 π로 바꾸면 주근이 됩니다. 둘째 인자를 생략하면(`k = 0`) "줄어듦"을 먼저 보는 "coupure de Petit"({math}`\operatorname{Im}k_z>0`, 허수부가 0이면 {math}`\operatorname{Re}k_z\ge0`)가 되어 fmmax의 규칙과 같습니다. 비교할 때 함정이 둘 있습니다.

- `res2`는 기본값으로 전파 차수만 보고합니다(`tolh`, `tolb`가 1e-6). 복소 `beta0`에서는 빈 목록이 나옵니다. 모든 차수를 받으려면 `parm`을 `res2`의 **셋째 인자**로 주고 허용치를 크게 잡습니다. `res1`의 `parm`에 넣으면 무시됩니다.
- 모든 차수를 보고하게 하면, 정확히 실수 `theta`에서 소멸 차수에도 {math}`|\text{진폭}|^2`을 효율로 적습니다. 그래서 효율의 합이 그 한 점에서만 튑니다({ref}`실제 격자 <ckr-grating>`).

**S4.** 복소 {math}`k_x`를 받지 않습니다. 대신 {math}`k_x`를 각도와 주파수의 실수부로 만들므로, 주파수를 복소수로 주면 {math}`k_x`는 실수로 남고 {math}`k_0=\omega/c`만 복소수가 되어 {math}`k_x/k_0`가 복소수가 됩니다. 이렇게 돌려 본 결과가 Maystre 규칙과 같았습니다. 근을 고르는 코드는 읽지 않고 실행 결과로만 확인했습니다.

**fmmax.** `s21[0, 0]`은 실수 {math}`k_x`에서도 `Ey_ref/Ey_inc`와 부호가 반대이므로, 반사 진폭은 재구성한 장에서 읽습니다.

(ckr-grating)=
### 실제 격자에서는 얼마나 달라지나

실제 격자로 끝까지 풀어 봅니다.

- 위쪽이 공기, 아래쪽이 기판({math}`\varepsilon=2.25`)이고, 그 사이에 한 층짜리 격자가 있습니다.
- 격자 층은 한 주기의 절반이 {math}`\varepsilon=9`, 나머지 절반이 {math}`\varepsilon=1`입니다.
- 두께 0.15, 주기 0.7, 파장 0.9(길이 단위는 모두 같음), TE 편광, 차수 ±5입니다.
- {math}`\theta=0.2+i\operatorname{Im}\theta`로 비춥니다.

위아래 반공간(공기, {math}`\varepsilon=2.25`)이 모두 무손실이므로 이 부록의 전제 안에 있습니다. 공기에서 {math}`k_x`는 0차가 0.199, +1차가 1.484, −1차가 −1.087입니다. **±1차가 둘 다 소멸 차수**라서, 실수 `theta`에서는 둘 다 경로 D의 branch cut 위에 앉습니다.

| {math}`\operatorname{Im}\theta` | 경로 D r0 | 경로 D R0 | Wavebit·RETICOLO r0 | Wavebit·RETICOLO R0 |
| ---: | --- | ---: | --- | ---: |
| −1e-3 | 0.091499 − 0.257051i | 0.074447 | 0.151070 + 0.092571i | 0.031392 |
| −1e-9 | 0.092037 − 0.256094i | 0.074055 | 0.149508 + 0.094586i | 0.031299 |
| 0 | 0.149508 + 0.094586i | 0.031299 | 0.149508 + 0.094586i | 0.031299 |
| +1e-9 | −0.006161 − 0.062574i | 0.003953 | 0.149508 + 0.094586i | 0.031299 |
| +1e-6 | −0.006163 − 0.062574i | 0.003954 | 0.149507 + 0.094588i | 0.031299 |
| +1e-3 | −0.007288 − 0.063145i | 0.004040 | 0.147873 + 0.096525i | 0.031184 |

r0는 0차 반사 진폭, R0는 0차 반사 효율입니다. 복소 `theta`에서는 형식적인 비율입니다. 경로 D 열은 2026년 10월 2일 전의 Wavebit으로 푼 값입니다. 지금의 Wavebit(Maystre 규칙)은 여섯 줄 모두 RETICOLO와 {math}`4\times10^{-15}` 안에서 같아서 한 열에 적었습니다.

- 실수 `theta`에서는 세 값이 같습니다(0.149508 + 0.094586i).
- 경로 D는 허수부를 **1e-9만** 줘도 튑니다. 허수부가 +이면 양의 소멸 차수들(공기의 +1차 등)이, −이면 음의 소멸 차수들(공기의 −1차 등)이 선을 건넙니다. 차수마다 {math}`\operatorname{Re}k_x\operatorname{Im}k_x`의 부호가 다르기 때문입니다. 부호마다 다른 차수가 뒤집히므로 **양쪽 모두에서** 튀고, 0차 효율이 0.031에서 0.004나 0.074로 바뀝니다.
- 지금의 Wavebit과 RETICOLO는 양쪽 다 매끄럽습니다. 1e-3을 줘도 조금만 변합니다.
- Wavebit의 테스트는 같은 격자를 {math}`\operatorname{Im}\theta=\pm0.1`에서도 RETICOLO와 맞춰 보며, {math}`2\times10^{-15}`까지 맞습니다.

RETICOLO에 모든 차수를 보고하게 하면 {math}`\operatorname{Im}\theta=0`인 줄에서만 반사 효율의 합이 0.182로 튑니다(다른 줄은 0.031 안팎). 소멸 차수의 {math}`|\text{진폭}|^2`까지 효율로 적기 때문이며, 근 선택과는 상관없습니다. 그래서 이 표는 0차만 비교합니다.

(ckr-beyond-lossless)=
## 무손실 전제를 풀면

이 부록의 결론은 {ref}`전제 2 <ckr-premises>`(무손실)에 기댑니다. {ref}`부록 B <complex-kx-lossy-gain>`는 이 전제 하나만 풀고 나머지 셋은 그대로 둔 채, 무엇이 달라지는지 봅니다.

- **입사 매질이 흡수하면**, 실수 `theta`(위상각 규약)와 복소 `theta`(형식적인 {math}`n\sin\theta`)가 서로 다른 {math}`k_x`를 줍니다. 복소 `theta`가 더는 실수 `theta`를 살짝 비튼 입력이 아닙니다({ref}`흡수 입사 매질 <ckr-absorbing-incidence>`).
- **근을 고르는 반공간이 흡수하면**, branch point가 실수축에서 떠서 실수 {math}`k_x` 근처에서는 네 규칙이 모두 같아집니다. 2026년 10월 2일 전 Wavebit의 연속성 보정이 일했던 곳도 여기입니다({ref}`흡수 반공간 <ckr-lossy>`).
- **반공간이 빛을 키우면(이득)**, 실수 {math}`k_x`에서도 두 기준이 서로 다른 근을 가리켜 규칙이 필요해지고, 답은 인과율이 정합니다({ref}`이득 매질 <ckr-gain>`).
- 흡수나 이득이 있어야 드러나는 코드 차이는 {ref}`부록 B의 코드 비교 <ckr-b-codes>`에 있습니다.

## 한 장 요약

- 전제: {math}`e^{-i\omega t}` 규약, 무손실 유전체({math}`\varepsilon>0`), 비자성·등방성, 반공간의 {math}`k_z`.
- {math}`k_z`는 제곱근이라 근이 둘입니다. 실수 {math}`k_x`에서는 근이 1사분면의 경계(실수축이나 허수축)에 놓여, 한 기준이 비기면 다른 기준이 정합니다. 두 기준이 서로 다른 근을 가리키지 않으므로 근이 저절로 정해집니다. 제곱근이 각도를 절반으로 만들기 때문입니다.
- 복소 {math}`k_x`에서 {math}`\operatorname{Re}k_x\operatorname{Im}k_x>0`이면 {math}`\operatorname{Im}k_z^2<0`이라 두 기준이 갈립니다. 커지는 근이 곧 틀린 답은 아니며, 판정 잣대는 실수 {math}`k_x`의 답과 이어지느냐, 곧 **연속성**입니다.
- 두 후보는 branch point({math}`k_x=\pm n`)에서 0으로 같아지고, branch point 하나를 감싸는 고리를 끊기지 않게 한 바퀴 따라가면 서로 바뀝니다. 그래서 어떤 규칙이든 **branch point에서 출발하는 branch cut**이 있고, 규칙마다 다른 것은 그 선을 어디에 두느냐입니다.
- 경로 D(2026년 10월 2일 전의 Wavebit)는 무손실 유전체에서 곧 주근이고, 연속성 보정이 한 번도 켜지지 않습니다. 그 branch cut은 **실수축의 소멸 구간**입니다. 실수축 위의 소멸 차수에서는 경로 D가 branch point를 정통으로 밟아 비기므로 비김 규칙이 줄어드는 근을 주지만, 조금만 위로 벗어나면 경로 D가 반대 근을 줍니다. 그래서 허수부 1e-9에도 소멸 차수가 뒤집힙니다.
- fmmax의 branch cut은 **실수축의 전파 구간**(과 허수축)입니다. Maystre(Wavebit·RETICOLO·S4)와 경로 E의 branch cut은 실수축 밖으로 빠지며(+n에서는 위로, −n에서는 아래로), 둘은 실수축 근처의 무손실 표본에서 286/288 같습니다.
- Wavebit은 2026년 10월 2일부터 Maystre 규칙을 씁니다. 실수 {math}`k_x`에서는 경로 D와 비트 단위로 같고, 실수축을 살짝 비틀어도 튀지 않으며, 실제 격자에서 RETICOLO와 같은 답을 냅니다.

## 용어

| 용어 | 이 부록에서의 뜻 | 다른 이름 |
| --- | --- | --- |
| 나감 / 들어옴 | {math}`\operatorname{Re}k_z>0` / {math}`<0`: 구조에서 멀어지는 파 / 다가오는 파 | outgoing / incoming (방사 조건) |
| 줄어듦 / 커짐 | {math}`\operatorname{Im}k_z>0` / {math}`<0`: 멀어질수록 약해지는 파 / 세지는 파 | decaying / growing |
| 전파 / 소멸 | 실수 {math}`k_x`의 크기가 {math}`n`보다 작은 / 큰 차수 | propagating / evanescent order |
| 무손실 유전체 | {math}`\varepsilon`이 양의 실수인 매질, 빛을 흡수하지도 키우지도 않음 | lossless dielectric |
| 형식적 | 원래의 뜻은 버리고 식의 모양만 빌려 쓴 것. 복소 `theta`는 방향이 아니라 {math}`k_x`를 만드는 계산용 숫자이고, 그때의 `R`, `T`는 입사 전력에 대한 비율이 아님 | formal |
| 주근 | {math}`\operatorname{Re}k_z>0`인 근, 실수부가 0인 비김은 비김 규칙이 {math}`\operatorname{Im}k_z>0` 쪽으로 풂 | principal root |
| 비김 규칙 | {math}`\lvert\operatorname{Re}k_z\rvert\le10^{-12}`이면 {math}`\operatorname{Im}k_z>0` 쪽을 고르는 규칙, 2026년 10월 2일 전의 Wavebit이 반공간에 씀 | tie-break |
| branch point | 두 후보 {math}`\pm k_z`가 모두 0이 되어 같아지는 {math}`k_x`({math}`=\pm n`) | 분기점, 가지점 |
| branch cut | 규칙이 반대 근으로 튀는 자리, branch point에서 출발 | 분기선, 가지 자름선 |
| Rayleigh 점 | 차수가 막 생기거나 사라지는 {math}`k_x` | Rayleigh anomaly (차수가 light line 위에 놓이는 점) |
| 경로 D / 경로 E | 수직 입사에서 / 가장 가까운 실수 {math}`k_x`에서 이어 붙이기 | (이 부록의 이름) |
| 연속성 보정 | 경로 D를 닫힌 식으로 만들려고 주근을 뒤집는 것, 무손실 유전체에서는 켜지지 않음, 2026년 10월 2일에 Wavebit에서 지움 | `_off_principal_branch` |
| Maystre 규칙 | {math}`\operatorname{Re}k_z+\operatorname{Im}k_z>0`, 2026년 10월 2일부터의 Wavebit | coupure de Maystre |
| {math}`\operatorname{Im}k_z>0` 규칙 | 줄어드는 근을 늘 고르는 규칙 (fmmax) | coupure de Petit |

## 헷갈리기 쉬운 문장

- **"경로 D가 branch point를 스친다."** 스치는 것이 아니라 한가운데를 밟습니다. 밟으면 따라가기가 비깁니다({ref}`경로 D <ckr-tie>`).
- **"경로 D 기준으로 채점하면 2026년 10월 2일 전의 Wavebit은 다 정답이다."** 맞습니다. 경로 D를 정답으로 정해 두고 그대로 구현했으니 288/288은 당연합니다. 물어야 할 것은 경로 D를 정답으로 삼아도 되느냐였고({ref}`branch cut <ckr-branch-cut>`, {ref}`훑기 <ckr-sweeps>`), Wavebit은 Maystre 규칙으로 바꿨습니다.
- **"무손실 매질에서도 2026년 10월 2일 전의 Wavebit이 연속성 보정으로 근을 뒤집었다."** 아닙니다. 무손실 유전체에서 보정은 한 번도 켜지지 않았습니다. 실수축 바로 위의 소멸 차수가 뒤집힌 것은 보정이 아니라 주근 자체의 선입니다({ref}`따라가기 <ckr-continuation>`, {ref}`경로 D <ckr-tie>`).
- **"Maystre로 바꿨으니 실수 각도의 결과도 바뀌었다."** 아닙니다. 실수 {math}`k_x`의 수동 매질에서는 Maystre 규칙과 경로 D가 비트 단위로 같습니다. 답이 바뀐 곳은 복소 `theta`와 이득 반공간뿐입니다({ref}`Wavebit의 규칙 <ckr-wavebit-rule>`).
- **"커지는 근을 고르면 틀린 것이다."** {math}`k_x=0.5+0.05i`에서는 경로 D, 경로 E, Maystre가 모두 살짝 커지는 근을 고릅니다. 그것이 실수 답과 이어지는 근입니다({ref}`복소 kx <ckr-complex-kx>`, {ref}`branch cut <ckr-branch-cut>`).
- **"fmmax에 복소 `kx`를 넣으면 실수부만 쓴다."** 아닙니다. fmmax 1.7.1은 `in_plane_wavevector`를 복소 그대로 쓰고, 근은 {math}`\operatorname{Im}k_z>0`으로 고릅니다.

## 참고문헌과 이 부록의 검증 범위

대조한 솔버:

- [J. P. Hugonin, P. Lalanne, *RETICOLO software for grating analysis*, Institut d'Optique, Orsay, France (2005)](https://arxiv.org/abs/2101.00901).
- [V. Liu, S. Fan, “S4: A free electromagnetic solver for layered periodic structures,” Comput. Phys. Commun. 183, 2233–2244 (2012)](https://doi.org/10.1016/j.cpc.2012.04.026).
- [M. F. Schubert, A. M. Hammond, *Fourier modal method for inverse design of metasurface-enhanced micro-LEDs* (2023)](https://arxiv.org/abs/2308.08573).

RETICOLO 수치는 RETICOLO V10을 Octave로 실행해 기록했고, S4는 실행 결과로만 비교했습니다. 경로 D로 푼 Wavebit 수치(격자 표의 경로 D 열, {ref}`비김 <ckr-tie>` 표의 "2026년 10월 2일 전" 열)는 규칙을 바꾸기 전의 Wavebit으로 잰 값이며, 기록의 `path_d`에 그 커밋과 함께 남아 있습니다. 그림의 수치는 모두 `tools/docs/learning/draw_complex_kx_roots.py`가 그리기 전에 스스로 검산하며, 경로 D·E의 닫힌 식과 한 걸음씩 따라가기, 각 규칙의 branch cut 식과 격자에서 실제로 튀는 자리, 무손실 유전체에서 경로 D와 주근이 맞는지 확인한 뒤에만 그림을 씁니다.
