(multiple-incident-waves)=
# 09. 여러 입사파와 내부 소스

> **과거 API 재현:** 이 페이지에서 `wavebit.compat`를 쓰는 코드·설명은 현재 설치의 API가 아닙니다.
> 해당 이름은 [퇴역 전 소스 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)에 있습니다.
> 측정 재현에는 본문에 적힌 원래 커밋·환경을 우선 사용하세요. 원본 측정값은 갱신하지 않았습니다.


:::{note}
다운로드 가능한 교육 스크립트는 현재 물리 Problem/Job API를 사용합니다. 일부 설명 코드와 그림에는 이전 API 표기가 남아 있으므로 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 함께 참고하세요. 과거 측정 파일과 원본 출처는 보존하며, 이전한 스크립트의 검증과 구분합니다.
:::

[8장](light.md)에서는 입사파 하나의 방향과 편광을 지정했습니다. 이 장에서는 여러 입사파가 함께 만드는 장과 출력 전력을 계산합니다. 먼저 외부 입사와 내부 소스의 공통 계산 흐름을 살펴본 뒤, 합산의 전제와 입력 방법을 설명합니다. 같은 쪽 입사·양면 입사·차수별 입력 예제로 이를 확인하고, 가우시안 빔과 층 내부의 전기 다이폴로 이어갑니다.


(coherent-illumination-flow)=
## 여러 소스의 계산 흐름

계산할 빛을 만드는 입력을 **소스**라고 부릅니다. 외부에서 들어오는 평면파·가우시안 빔과 층 내부에서 빛을 내는 전기 다이폴을 함께 지정할 수 있습니다. 아래 흐름은 **같은 파장과 선형 재료**, 그리고 각 소스의 크기와 위상을 정한 계산에 해당합니다.

외부 입사파는 **위·아래 경계의 입사 진폭**으로, 내부 다이폴은 **소스 위치의 전류 항**으로 들어갑니다. 계산에서 얻은 복소 전기장 E와 자기장 H를 각각 합하고, 합친 장으로 전력을 구한다는 원리는 같습니다.

```{figure} ../../_static/learning/09-source-calculation.svg
:figclass: learning-figure
:alt: 외부 평면파와 가우시안 빔은 경계 입사 진폭으로, 내부 다이폴은 전류 항으로 넣는다. 같은 푸리에 배열을 공유하는 성분끼리 RCWA를 계산하고 복소 장을 합산한다. 전력을 계산한 뒤 외부 입사만 있으면 입사 전력으로 정규화하고, 다이폴이 포함되면 방향별 출력 전력을 읽는다.

그림 9-1. 외부 입사와 내부 소스의 공통 계산 흐름. 한쪽·양면 입사와 다이폴을 함께 다루되, 입력 위치와 출력 전력을 읽는 방법을 구분합니다.
```

**내부 다이폴은 전력을 직접 공급하므로 외부 입사 전력만을 기준으로 반사율·투과율을 정의하지 않습니다.** 다이폴이 포함된 결과의 `output_top`, `output_bottom`, `absorption`은 `None`이며, `result.power`에서 방향별 전력을 읽습니다. 단위와 사용 예는 {ref}`다이폴 절 <electric-dipole-source>`에서 설명합니다.

(source-calculation-groups)=
### 어떤 성분을 함께 계산할까?

두 평면파의 횡파수 차이가 역격자 벡터의 정수배 조합이면, 같은 회절 차수 배열에 놓고 한 번의 RCWA 계산으로 처리할 수 있습니다. 이렇게 함께 계산할 수 있는 성분들을 **계산 묶음**이라고 부릅니다. 1D 격자에서는 `kx` 차이가 `2π/Px`의 정수배인 경우입니다. 역격자 벡터와 회절 차수의 관계는 [6장](lattice.md)에서 설명했습니다.

**같은 쪽에서 입사해도 횡파수 조건이 다르면 따로 계산합니다.** 이 장의 {ref}`같은 쪽 입사 예제 <same-side-beams>`에서는 수직 입사와 20° 입사가 서로 다른 묶음입니다. 반면 {ref}`0차와 +1차를 지정한 예제 <incident-order-inputs>`는 하나의 묶음입니다. 위·아래에서 들어오는 파도 이 횡파수 조건으로 묶습니다.

가우시안 빔은 여러 방향의 평면파 성분으로 구성합니다. 주기적으로 반복되는 다이폴 배열은 이웃 셀 사이의 위상을 정하는 `bloch_k`를 사용하고, 고립된 다이폴은 여러 Bloch 파수의 결과를 합하여 근사합니다. 이 성분들도 같은 푸리에 배열을 공유하면 외부 입사파와 함께 계산합니다.

묶음별로 따로 계산하더라도 **공통 좌표와 위상 기준에서 복소 장을 합산하므로 소스 사이의 간섭이 포함됩니다.** 아래에서는 먼저 장을 합산할 수 있는 이유를 짚고, 각 소스의 입력 방법을 살펴보겠습니다.

(linear-source-superposition)=
### 여러 파를 합산하는 조건

Wavebit는 **선형 재료**를 다룹니다. 주어진 파장에서 유전율·투자율이 빛의 세기에 따라 달라지지 않고, 전기장과 재료 응답의 관계가 선형이라는 뜻입니다. 예를 들어 전기장 **E**와 전기변위 **D**는 다음 관계를 따릅니다.

```{math}
\mathbf D=\varepsilon_0\boldsymbol{\varepsilon}_r\mathbf E.
```

여기서 ε₀는 진공 유전율, **ε**<sub>r</sub>은 상대 유전율입니다. 재료값이 고정되어 있으므로 두 전기장의 합에 대한 응답은 각 전기장에 대한 응답의 합과 같습니다. **같은 구조에서 각 소스로 계산한 전기장끼리, 자기장끼리 더하면 동시에 소스를 넣었을 때의 장을 얻을 수 있습니다.** 유전율이 복소수인 손실 재료나 이방성 텐서인 재료도 이 선형 조건을 만족할 수 있습니다. 빛의 세기에 따라 유전율이 달라지는 비선형 재료는 이 계산의 범위에 포함되지 않습니다.

한 `excitation`에 함께 넣은 소스는 모두 같은 진동수로 진동하며, 소스 사이의 상대 위상이 시간에 따라 일정하다고 가정합니다. 예를 들어 두 소스의 위상을 0°와 +90°로 지정하면, 그 90°의 차이를 유지하며 진동하는 상태를 계산합니다. 이처럼 일정한 위상관계를 갖는 소스를 **결맞는 소스**라고 합니다.

`wavelength`에는 이 공통 진동수에 해당하는 **진공 파장**을 넣습니다.

장은 이렇게 더할 수 있지만, **전력에는 두 장 사이의 간섭이 반영됩니다.** 따라서 결맞는 소스들을 함께 계산할 때는 장을 합친 다음 전력을 구합니다. 서로 비결맞는 입사파의 전력을 합산하는 방법은 뒤의 {ref}`전력 설명 <coherent-and-incoherent>`에서 구분합니다.

(source-types)=
## 소스 종류와 입력 형식

바깥에서 들어오는 입사파는 `PlaneWave` 또는 `GaussianBeam`으로, 층 안에서 빛을 내는 전기 다이폴은 `ElectricDipole`로 지정합니다. 이 소스 객체들은 `Source`라는 공통 타입에 속하며, `excitation`에 하나 또는 목록으로 넣습니다.

| 소스 | 지정하는 물리량 | 계산에 들어가는 위치 |
| --- | --- | --- |
| `PlaneWave` | 입사 방향·편광·복소 진폭 | 위·아래 바깥 경계 |
| `GaussianBeam` | 중심 방향·편광·빔 폭·초점 위치·복소 진폭 | 여러 평면파로 분해하여 바깥 경계에 입력 |
| `ElectricDipole` | 위치와 복소 전기 쌍극자 모멘트 | 해당 층 내부의 전류 항 |
| `OrderExcitation` | 기준 방향과 차수별 복소 진폭 | 여러 입사 평면파를 차수로 지정 |

평면파의 입사 쪽은 각 객체의 `side`로 지정합니다. 목록에서 생략하면 `"top"`입니다. 위·아래 동시 입사는 다음처럼 쓸 수 있습니다.

```python
from wavebit.compat import legacy as wavebit

sources = [
    wavebit.PlaneWave(side="top", kx=0.002, ky=0., amplitude=1.),
    wavebit.PlaneWave(side="bottom", kx=0.002, ky=0., amplitude=0.5j),
]
# rcwa(..., excitation=sources)에 전달합니다.
```

`excitation={"top": [...], "bottom": [...]}`처럼 바깥 경계별로 묶는 형식도 사용할 수 있습니다. 내부 다이폴은 `top`이나 `bottom`에 속하지 않으므로 소스 목록에 넣습니다. 위치 좌표는 다른 장과 같은 xyz 기준이며, z=0은 첫째 층의 바닥이고 +z는 위쪽입니다.

(id3)=
### 평면파 입력 형식 고르기

평면파부터 살펴보겠습니다. 입사파마다 각도와 편광을 따로 정하려면 `PlaneWave`를, 격자의 입사 차수로 방향을 고르려면 차수별 복소 진폭을 사용합니다.

| 원하는 입력 | 형식 | 예제 |
| --- | --- | --- |
| 같은 쪽에서 입사파마다 각도·편광을 따로 지정 | `excitation={"top": [wavebit.PlaneWave(...), ...]}` | {ref}`같은 쪽에 여러 입사파 <same-side-beams>` |
| 위·아래에서 동시에 입사 | `excitation={"top": ..., "bottom": ...}` | {ref}`양면 입사 <two-sided-illumination>` |
| 한 기준 방향에서 입사 차수를 선택 | `excitation={0: 1., 1: 0.5j}` | {ref}`차수별 입력 <incident-order-inputs>` |
| 차수별 입력을 소스 목록에 포함 | `wavebit.OrderExcitation(orders={0: 1., 1: 0.5j}, ...)` | {ref}`입사 차수 묶기 <beams-detail-id31>` |

`PlaneWave`의 `side` 또는 바깥 경계 사전의 `"top"`·`"bottom"` 키로 입사 쪽을 정합니다. `excitation={0: ..., 1: ...}`처럼 차수별 진폭만 넣는 형식에서는 `incidence_side`로 한쪽을 선택합니다. 입력 형식마다 위상을 정하는 기준점과 결과 객체가 다르므로, 아래 예제에서 함께 확인합니다.

(excitation-amplitude-phase)=
## 진폭·위상과 빛의 세기

**복소 진폭은 전기장 진폭의 크기와 위상을 함께 나타냅니다.** `PlaneWave`에서는 `amplitude`에, 차수별 입력에서는 각 차수의 값에 넣습니다. 코드에서는 `크기 * np.exp(1j * 위상)`으로 씁니다.

- 앞의 **크기**는 전기장 진폭의 크기를 정합니다.
- `np.exp(1j * 위상)`은 크기가 1인 복소수로, **위상만 지정**합니다. 위상은 라디안 단위이므로, 도 단위의 각도는 `np.deg2rad()`로 변환합니다.
- Python에서 `1j`는 허수 단위입니다. 위상이 +90°이면 `np.exp(1j * np.deg2rad(90.))`가 수학적으로 `1j`와 같습니다.

예를 들어 크기 0.5·위상 +45°인 값은 `0.5 * np.exp(1j * np.deg2rad(45.))`입니다. 크기를 그대로 두고 위상각만 바꾸면 입사파의 진폭 크기는 유지되고 다른 입사파와의 상대 위상이 바뀝니다.

**빛의 세기**는 단위 면적을 통과하는 시간평균 전력이며, 단위는 W/m²입니다. 전기장 진폭은 전기장이 진동하는 크기를 나타내고 단위는 V/m입니다. 같은 매질·방향·편광의 평면파에서는 세기가 전기장 진폭 크기의 제곱에 비례합니다. 따라서 진폭을 두 배로 하면 세기는 네 배, 절반으로 하면 세기는 1/4이 됩니다. 서로 다른 방향이나 매질에서 경계면을 지나는 전력을 비교할 때는 입사각과 매질의 영향도 반영해야 합니다.

외부 입사파만 계산할 때는 진폭을 1에 대한 상대값으로 정하고 출력 전력의 비율을 읽을 수 있습니다. 절대 전력이나 다이폴과의 합산이 필요하면 전기장 진폭에 V/m 단위의 크기를 부여합니다.

(same-side-beams)=
## 같은 쪽에 여러 입사파를 넣기

위에서 여러 입사파를 넣으려면 `excitation={"top": [PlaneWave(...), PlaneWave(...)]}`처럼 **한쪽에 입사파 목록**을 지정합니다. 각 `PlaneWave`에서 방향·편광·진폭·위상을 따로 정합니다. 모든 입사파는 같은 구조에 입사하며, `rcwa()`의 `wavelength`를 공유합니다.

`"top"`만 적으면 위에서만 입사합니다. 아래에서만 여러 입사파를 넣으려면 목록의 키를 `"bottom"`으로 정합니다. 이때도 `top`과 `bot`에는 위·아래 매질의 재료값을 모두 넣습니다.

(beams-detail-id30)=
### 각 입사파의 값을 정하는 방법

| 설정 | 지정하는 내용 |
| --- | --- |
| `theta`, `phi` | 해당 입사파의 입사각과 방위각입니다. 단위는 라디안이며, 각 입사파의 입사 매질 안에서 읽습니다. |
| `kx`, `ky` | 각도 대신 횡파수를 직접 지정합니다. 길이 단위가 nm이면 단위는 nm⁻¹입니다. 같은 `PlaneWave`에 `theta`, `phi`와 함께 넣지 않습니다. |
| `polarization=(a_s, a_p)` | 그 입사파의 s·p 전기장 성분 비율과 상대 위상을 지정합니다. `(1., 0.)`은 s, `(0., 1.)`은 p 편광입니다. 두 성분은 크기의 제곱합이 1이 되도록 정규화됩니다. |
| `psi`, `chi` | `polarization` 대신 편광 타원의 각도로 지정할 수 있습니다. {ref}`8장의 편광 설명 <incident-polarization>`과 같은 규약을 사용합니다. |
| `amplitude` | 정규화된 편광 계수에 곱할 진폭 크기와 위상입니다. `A * np.exp(1j * np.deg2rad(delta))`에서 `A`는 크기, `delta`는 위상각(°)입니다. 생략하면 `1.`입니다. |

`polarization`은 입사파 안의 s·p 성분 사이 관계를, `amplitude`는 입사파 전체의 크기와 다른 입사파에 대한 위상을 정합니다.

방향을 생략한 입사파는 `rcwa()`에 공통 `kx`, `ky`가 있으면 그 값을 사용하고, 없으면 수직 입사합니다. 편광도 공통 설정을 따르며, 공통 설정도 없으면 s 편광입니다. 아래 예제에서는 각 입사파의 조건을 직접 적습니다.

(beams-detail-id29)=
### 예제: 수직 입사와 20° 입사를 함께 넣기

공기에서 흡수성 평판으로 두 s 편광 입사파를 넣습니다. 첫째 입사파는 수직으로, 둘째 입사파는 +x 방향으로 20° 기울어져 들어옵니다. 둘째 입사파의 전기장 진폭 크기는 첫째의 절반이고, 기준점에서의 상대 위상은 +90°입니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

beams = [
    wavebit.PlaneWave(
        theta=0., phi=0.,
        polarization=(1., 0.),          # s 편광
        amplitude=1.,                  # 진폭 크기 1, 위상 0°
    ),
    wavebit.PlaneWave(
        theta=np.deg2rad(20.), phi=0.,
        polarization=(1., 0.),          # s 편광
        amplitude=0.5 * np.exp(1j * np.deg2rad(90.)),
    ),
]

same_side = wavebit.rcwa(
    wavelength=600.,                    # 길이 단위: nm
    period=(900., 850.),
    top=1., bot=2.25,
    structure=[dict(background=4.+0.6j, shapes=[])],
    thickness=[100.],
    truncation_order=0,                 # 패턴이 없는 균일 평판
    excitation={"top": beams},
)
same_side_result = same_side.solve(power_region="infinite")

print("반사율:", same_side_result.output_top)
print("투과율:", same_side_result.output_bottom)
print("흡수율:", same_side_result.absorption)
```

이 예제의 반사율은 약 `0.154612`, 투과율은 `0.624306`, 흡수율은 `0.221082`이며, 합은 1입니다.

| 입력 | 입사 방향 | 편광 | 진폭 크기 | 기준점에서의 위상 |
| --- | --- | --- | --- | --- |
| `beams[0]` | 수직으로 아래쪽 | s | 1 | 0° |
| `beams[1]` | 아래쪽으로 진행하며 +x 쪽으로 20° 기울어짐 | s | 0.5 | +90° |

`beams[0]`, `beams[1]`의 숫자는 **목록에서 입사파를 구별하는 번호**입니다. 회절 차수 0차·1차를 뜻하지 않습니다. 목록에 `PlaneWave`를 추가하면 셋 이상의 입사파도 넣을 수 있습니다.

위에서만 입사하므로 `output_top`은 전체 반사율, `output_bottom`은 전체 투과율입니다. 둘 다 두 입사파가 함께 만드는 입사파만의 전력을 기준으로 읽습니다. 아래에서만 입사한다면 `output_bottom`이 반사율, `output_top`이 투과율입니다.

위에서 넣는 두 입사파는 모두 **위 경계면의 셀 중심**인 `(x, y) = (0, 0)`에서 위상을 지정합니다. 따라서 예제의 +90°는 이 점에서 둘째 입사파가 첫째 입사파에 대해 갖는 상대 위상입니다. 두 입사파는 진행 방향이 달라 표면의 다른 위치에서는 상대 위상도 달라집니다. 한 위치에서 두 파 사이의 위상차가 시간에 따라 유지되는 결맞는 입사입니다.

진폭 크기 0.5가 입사 전력의 비율 0.5를 뜻하지는 않습니다. 이 예제처럼 같은 무손실 등방성 매질에서 전기장 진폭을 지정하면, 경계면에 수직으로 들어오는 전력은 `진폭 크기² × cos(theta)`에 비례합니다. 둘째 입사파의 입사 전력은 첫째의 `0.5² × cos(20°) ≈ 0.235`배입니다.

이 예제는 두 입사파의 방향이 달라 전력을 평균할 영역을 `power_region`으로 명시했습니다. 어떤 입사파들을 함께 계산하는지는 {ref}`계산 흐름 <coherent-illumination-flow>`에서 설명합니다. `power_region="infinite"`에서는 서로 다른 횡파수 사이의 간섭항이 공간 평균으로 사라지지만, 위치별 필드에는 간섭이 남습니다. 평균 영역별 차이는 {ref}`전력 읽기 <illumination-power-region>`에서 설명합니다.

(two-sided-illumination)=
## 위·아래에서 동시에 입사시키기

양쪽에서 빛을 넣으려면 `excitation` 안에 `"top"`과 `"bottom"`을 적고, 각 쪽에 `wavebit.PlaneWave(...)`을 지정합니다. 같은 쪽 입사 예제와 마찬가지로 각 입사파의 방향·편광·진폭·위상을 따로 정합니다.

`top`과 `bot`은 위·아래 반무한 매질의 재료값입니다. 빛이 들어오는 쪽은 `excitation`의 키로 정하므로 이 형식에서는 `incidence_side`를 함께 적지 않습니다. 층은 아래에서 위 순서로 입력합니다.

(beams-detail-id19)=
### 예제: 두 경계면에서 위상차를 90°로 지정하기

아래 예제는 위쪽이 공기, 아래쪽이 유리이고 그 사이에 흡수성 평판이 있는 경우입니다. 두 입사파의 `kx`, `ky`를 공통으로 정하고, 아래쪽 입사파에 +90°의 위상을 줍니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

two_sided = wavebit.rcwa(
    wavelength=600.,                    # 길이 단위: nm
    period=(900., 850.),
    top=1., bot=2.25,                   # 위·아래 상대 유전율
    structure=[dict(background=4.+0.6j, shapes=[])],
    thickness=[100.],
    truncation_order=0,                 # 패턴이 없는 균일 평판 예제
    kx=0.002, ky=0.,                    # 공통 횡파수, 단위: nm⁻¹
    answers="field",
    excitation={
        "top": wavebit.PlaneWave(polarization=(1., 0.)),
        "bottom": wavebit.PlaneWave(
            polarization=(1., 0.),
            amplitude=1.0 * np.exp(1j * np.deg2rad(90.)),
        ),
    },
)
two_sided_result = two_sided.solve()

print("위쪽으로 나가는 전력 비율:", two_sided_result.output_top)
print("아래쪽으로 나가는 전력 비율:", two_sided_result.output_bottom)
print("평판에서 흡수되는 전력 비율:", two_sided_result.absorption)
```

`amplitude`는 정규화된 s·p 편광 계수에 곱하는 복소수입니다. 앞의 `1.0`이 크기, `90.`이 위상각입니다. **위쪽 입사파는 위 경계면의 셀 중심에서, 아래쪽 입사파는 아래 경계면의 셀 중심에서** 진폭과 위상을 지정합니다. 따라서 이 예제의 90°는 그 두 기준점 사이의 상대 위상입니다.

위·아래 매질이 다르면 같은 `kx`, `ky`에서도 입사각이 다릅니다. 이 예제에서 위쪽 입사각은 약 11.01°, 아래쪽은 약 7.31°입니다. 진폭의 크기가 둘 다 1이어도 매질과 각도가 다르므로, 경계면을 통과하는 입사 전력은 같지 않습니다.

아래쪽 입사파의 위상만 바꾸어 계산하면 다음과 같습니다.

| 아래쪽 입사파의 위상 | 위쪽 출력 비율 | 아래쪽 출력 비율 | 흡수 비율 |
| --- | --- | --- | --- |
| 0° | 0.481181 | 0.242506 | 0.276313 |
| 90° | 0.739880 | 0.059363 | 0.200757 |
| 180° | 0.393406 | 0.411418 | 0.195176 |

구조와 각 입사파의 입사 전력은 같지만, 두 입사파의 간섭으로 출력과 평판 내부의 장이 달라집니다. 이 표의 모든 비율은 **양쪽에서 들어오는 입사파만의 전력을 더한 값**으로 나눈 결과입니다.

(beams-detail-id20)=
### 위·아래 입사각을 각각 정하고 싶다면

각 `PlaneWave`에 `theta`, `phi`를 직접 적으면 됩니다. 각도는 그 입사파가 들어오는 매질 안에서 읽습니다. 또는 각 입사파에 `kx`, `ky`를 직접 줄 수 있습니다. 한 입사파 안에서 두 입력 방식을 섞지는 않습니다.

```python
two_sided_result = two_sided.solve(
    excitation={
        "top": wavebit.PlaneWave(theta=np.deg2rad(20.), phi=0.),
        "bottom": wavebit.PlaneWave(
            theta=np.deg2rad(35.), phi=0.,
            amplitude=0.7 * np.exp(1j * np.deg2rad(90.)),
        ),
    },
    power_region="infinite",
)
print(two_sided_result.group_count)  # 이 예제에서는 2
```

입사각을 각각 정하면 두 입사파가 서로 다른 계산 묶음에 놓일 수 있어 `power_region`을 지정합니다. 묶음을 나누는 기준과 전력 평균은 {ref}`계산 흐름 <coherent-illumination-flow>`에서 설명합니다.

(incident-order-inputs)=
## 입사 차수로 여러 입사파를 지정하기

`excitation={0: 1., 1: 0.5j}`는 같은 쪽에서 입사하는 파들을 회절 차수로 고르는 형식입니다. 각 차수의 방향은 기준 입사 방향과 격자 주기로 결정되며, 모든 파가 하나의 편광 설정을 공유합니다.

(beams-detail-id13)=
### 차수 번호가 입사 방향을 정하는 방법

먼저 `theta`·`phi` 또는 `kx`·`ky`로 기준 입사파를 정합니다. 이 파를 **0차 입사파**라고 부릅니다. x 방향 주기가 P<sub>x</sub>인 1D 격자에서는 x 방향 파수가 `2π/Px`씩 다른 파들을 정수 m으로 구분합니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>k<sub>x,m</sub> = k<sub>x,0</sub> + m · 2π/P<sub>x</sub></p>
</div>
```

**m=0은 기준 방향, m=+1은 x 방향 파수가 한 간격 더 큰 방향**입니다. `excitation`의 `0`, `1`은 이 차수 번호입니다. 파 번호마다 임의의 각도를 따로 넣는 방식이 아니라, 같은 주기 격자의 파수 간격에 놓인 입사파들을 고릅니다. 파수 간격과 격자의 관계는 [6장](lattice.md)에서 설명합니다.

(beams-detail-id15)=
### 예제: 수직 입사파와 비스듬한 입사파를 함께 넣기

아래 예제는 공기에서 파장 600 nm의 빛을 주기 1500 nm의 격자에 넣습니다. `theta=0`이므로 0차는 수직 입사입니다. +1차는 `kx/k0 = 600/1500 = 0.4`이고, 입사각은 `arcsin(0.4)`, 약 23.58°입니다. 두 파 모두 위에서 아래로 진행하며, +1차는 +x 방향 성분도 가집니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

amplitude_1 = 0.5                 # +1차 전기장 진폭의 크기
phase_1 = np.deg2rad(90.)          # 0차에 대한 +1차의 상대 위상

order_solver = wavebit.rcwa(
    backend='numpy',
    top=1., bot=2.25,
    incidence_side="top",
    wavelength=600., period=[1500., 1500.],  # 길이 단위: nm
    theta=0., phi=0.,                      # 0차 입사파의 방향
    structure=np.array([[[1., 1., 4., 4.]]]),
    thickness=[200.],
    truncation_order=dict(x=4, y=0),
    polarization=(1., 0.),                 # 두 입사파 모두 s 편광
    excitation={
        0: 1.,                            # 크기 1, 위상 0°
        1: amplitude_1 * np.exp(1j * phase_1),  # 크기 0.5, 위상 +90°
    },
)
order_result = order_solver.solve()
print("두 입사파를 함께 넣었을 때의 반사율:", order_result.R.sum())
print("두 입사파를 함께 넣었을 때의 투과율:", order_result.T.sum())
```

앞 코드의 두 입사파를 크기와 위상으로 나누어 읽으면 다음과 같습니다.

| 차수 번호 | 이 예제의 입사 방향 | 진폭의 크기 | 0차에 대한 상대 위상 |
| --- | --- | --- | --- |
| `0` | 수직 입사 | 1 | 0° |
| `1` | +x 쪽으로 약 23.58° 기울어진 입사 | 0.5 | +90° |

+1차의 `amplitude_1 * np.exp(1j * phase_1)`은 이 예제에서 `0.5j`와 같습니다. **크기 0.5와 위상 +90°를 한 복소수로 표현한 값**입니다. `j`의 크기는 1이므로 `0.5j`의 크기는 0.5입니다. 진폭의 크기는 전력 비율을 뜻하는 값은 아닙니다. 표면을 통과하는 입사 전력에는 진폭 크기의 제곱뿐 아니라 각 파의 입사 방향과 매질도 반영됩니다. `R.sum()`과 `T.sum()`은 **두 입사파의 총 입사 전력**을 기준으로 정규화된 결과입니다.

이 차수별 진폭 형식의 위상 기준은 입사 쪽 경계면에서 푸리에 셀의 시작점입니다. 직교 셀에서는 `(x, y)=(-Px/2, -Py/2)`이며, 사선 셀에서는 `−(a₁+a₂)/2`입니다. `PlaneWave` 형식의 진폭은 셀 중심을 기준으로 합니다. 두 파의 `kx`가 다르므로 표면을 따라 이동하면 상대 위상이 달라지고, 합쳐진 입사장의 세기도 위치에 따라 달라질 수 있습니다. [8장](light.md)에서 설명한 단일 입사파는 각각의 세기가 표면을 따라 일정합니다. 여러 파를 합한 세기에는 파 사이의 간섭이 반영됩니다.

같은 크기 0.5에 **상대 위상 +45°를 지정하려면** 다음과 같이 입력합니다. 앞에서 만든 `order_solver`의 두 입사파 방향과 재료 설정을 사용하여 다시 계산합니다.

```python
order_solver.excitation = {
    0: 1.,
    1: 0.5 * np.exp(1j * np.deg2rad(45.)),  # 크기 0.5, 위상 +45°
}
result_45 = order_solver.solve()
```

이때 +1차의 복소 진폭은 약 `0.353553 + 0.353553j`입니다. 실수부와 허수부가 함께 달라져도 진폭의 크기는 0.5이고, 0차에 대한 상대 위상은 +45°입니다.

(beams-detail-id17)=
### 차수별 진폭을 한 사전으로 입력할 때의 범위

- **`excitation={0: 1., 1: 0.5j}`처럼 차수별 진폭을 적는 형식은 같은 파장·입사 쪽·편광 설정을 공유합니다.** `excitation`은 각 파의 차수와 진폭을 정하고, `polarization` 또는 `psi`·`chi`는 모든 입사파에 공통으로 적용됩니다. 각 파의 s·p 방향은 그 파의 진행 방향을 기준으로 정합니다.
- **지정한 입사 차수는 계산 차수 범위 안에 있어야 합니다.** 예를 들어 +1차를 넣으려면 `truncation_order`의 x 범위가 1 이상이어야 합니다. 계산 정확도에 필요한 차수 선택은 [10장](fourier-orders.md)에서 설명합니다.
- **입사 쪽 매질에서 구조를 향하는 전력을 가져야 합니다.** 파장·주기·기준 입사 방향에 따라 허용되는 차수가 달라지며, 전파하지 않는 소멸파나 입사 전력이 0인 차수를 지정하면 오류가 발생합니다.

(excitation-order-pairs)=
(d-m-n)=
### 2D 입사 차수: `(m, n)`의 순서

**x·y축에 나란한 직교 격자에서는 `(m, n)`을 `(x 방향 차수, y 방향 차수)` 순서로 씁니다.** 첫 번째 숫자 `m`은 x 방향, 두 번째 숫자 `n`은 y 방향의 차수입니다. 이 숫자들은 차수를 구별하는 정수 번호이며, `kx`, `ky`의 파수 값 자체는 아닙니다.

| `excitation`의 키 | x 방향 차수 `m` | y 방향 차수 `n` | 기준 입사파의 횡파수에 더하는 값 `(Δkx, Δky)` |
| --- | --- | --- | --- |
| `(0, 0)` | 0 | 0 | `(0, 0)` — 기준 입사파 |
| `(1, 0)` | +1 | 0 | `(2π/Px, 0)` |
| `(0, 1)` | 0 | +1 | `(0, 2π/Py)` |
| `(-1, 1)` | −1 | +1 | `(−2π/Px, 2π/Py)` |

여기서 `Px`, `Py`는 각각 x·y 방향의 주기입니다. 예를 들어 `(1, 0)`은 기준 입사파의 `kx`에 `2π/Px`를 더하고, `ky`는 그대로 둔 입사파를 뜻합니다. `(0, 0)`은 `theta`·`phi` 또는 `kx`·`ky`로 지정한 기준 입사 방향을 따르므로, 수직 입사일 때에만 두 횡파수 성분이 모두 0입니다.

다음은 기준 입사파와 x 방향 +1차 입사파를 함께 지정하는 값입니다. 진폭의 크기와 위상은 1D와 같은 방법으로 입력합니다.

```python
excitation = {
    (0, 0): 1.,  # 기준 입사파: 크기 1, 위상 0°
    (1, 0): 0.5 * np.exp(1j * np.deg2rad(90.)),  # x +1차, y 0차
}
```

**사선 격자에서는 `m`, `n`이 첫째·둘째 역격자 벡터에 곱하는 차수입니다.** [6장](lattice.md)에서 설명한 두 역격자 벡터를 **b**<sub>1</sub>, **b**<sub>2</sub>라고 하면, 각 입사 차수의 횡방향 파수 벡터는 다음과 같습니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p><b>k</b><sub>∥,(m,n)</sub> = <b>k</b><sub>∥,(0,0)</sub> + m <b>b</b><sub>1</sub> + n <b>b</b><sub>2</sub></p>
</div>
```

즉 `(1, 0)`은 기준 입사파의 횡파수에 **b**<sub>1</sub>을, `(0, 1)`은 **b**<sub>2</sub>를 더합니다. 사선 격자에서는 역격자 벡터가 x·y축과 나란하지 않을 수 있어, `m` 하나를 바꾸어도 `kx`, `ky`가 함께 달라질 수 있습니다. 직교 격자에서는 **b**<sub>1</sub> = `(2π/Px, 0)`, **b**<sub>2</sub> = `(0, 2π/Py)`이므로 위 표의 x·y 차수 순서로 읽으면 됩니다.

서로 다른 차수의 입력 방식과 계산 범위는 [여러 입사파 상세 설명(영문)](../../guide/rcwa/multimode-incidence.md), 전력 정규화는 {ref}`13장 <half-space-power>`에서 다룹니다.

(beams-detail-id31)=
### 입사 차수를 묶어서 지정하기

`OrderExcitation`은 한 기준 방향에서 **역격자 벡터만큼 떨어진 여러 입사 평면파**를 묶어 지정합니다. `orders`에 차수별 복소 진폭을 넣고, 모든 차수에 공통 s·p 편광 계수를 적용합니다. `PlaneWave` 하나는 평면파 하나를 지정합니다.

```python
beam_order_result = same_side.solve(
    truncation_order=dict(x=1, y=0),    # 입력할 +1차를 포함
    excitation={
        "top": wavebit.OrderExcitation(
            theta=0., phi=0.,
            polarization=(1., 0.),
            amplitude=1.,
            orders={
                0: 1.,
                1: 0.5 * np.exp(1j * np.deg2rad(90.)),
            },
        ),
    },
    power_region="cell",
)
```

여기서 `orders`의 `0`, `1`은 **x 방향 입사 차수**입니다. 이 예제는 `ky=0`이고, `kx`는 기준 방향의 값에 `m × 2π/Px`를 더합니다. 공기, 파장 600 nm, 주기 900 nm이므로 0차는 수직 입사, +1차는 +x 쪽으로 약 41.81° 기울어진 입사입니다. {ref}`같은 쪽 입사 예제 <same-side-beams>`의 20° 입사파와는 다른 방향입니다. 2D에서는 `(m, n)`을 사용하며, 두 차수는 각각 첫째·둘째 역격자 벡터에 곱하는 정수입니다.

각 차수의 복소 계수는 `OrderExcitation.amplitude × orders[차수]`이며, 여기에 공통 s·p 편광 계수를 곱합니다. 차수마다 다른 편광을 넣으려면 별도의 `PlaneWave`로 지정합니다. 입력 차수는 `truncation_order` 범위 안에 있어야 하고, 입사 매질에서 구조 쪽으로 전력을 운반할 수 있어야 합니다.

한 계산 묶음에서는 각 입력 주변의 회절 차수를 포함하도록 계산 배열을 넓힙니다. 따라서 한 묶음에 멀리 떨어진 입사 차수를 넣으면 배열 크기와 계산량도 늘어납니다.

(illumination-output-power)=
## 합친 장에서 출력 전력 구하기

앞의 예제에서 구한 `PlaneWave`·`OrderExcitation` 결과는 나가는 쪽에 따라 `output_top`, `output_bottom`으로 읽습니다. 위에서만 입사하면 각각 반사율·투과율이고, 아래에서만 입사하면 각각 투과율·반사율입니다.

위쪽으로 나가는 전기장에는 **위에서 넣은 입사파의 반사 성분과 아래에서 넣은 입사파의 투과 성분**이 함께 들어갑니다. 아래쪽 출력도 같은 방식으로 두 입력의 기여를 합칩니다. 그러므로 양면 입사의 결과는 하나의 `R`, `T`로 부르지 않고, 나가는 쪽에 따라 `output_top`, `output_bottom`으로 읽습니다.

```{math}
\mathbf E=\sum_j\mathbf E_j,\qquad
\mathbf H=\sum_j\mathbf H_j,\qquad
\langle\mathbf S\rangle=\frac12\operatorname{Re}(\mathbf E\times\mathbf H^*).
```

여기서 j는 각 계산의 기여를 구별하는 번호이며, 앞에서 복소수에 쓴 Python의 `1j`와는 별개입니다. **S**는 전력 흐름 밀도를 나타내는 포인팅 벡터, ⟨ ⟩는 시간평균, 별표는 복소켤레입니다. 식의 **H**는 실제 자기장입니다. Wavebit가 반환하는 Z₀H로 계산한다면 식의 결과를 진공 임피던스 Z₀로 나눕니다.

전기장끼리, 자기장끼리 더한 뒤 두 전체 장으로 전력을 구하면 두 파 사이의 간섭항도 포함됩니다. 전력 흐름 밀도의 경계면에 수직인 성분을 면적에 걸쳐 적분하면 그 면을 통과하는 전력이 됩니다. **출력 전력에는 바깥 경계면의 필드만 있으면 됩니다.** 층 내부의 모든 위치에서 필드를 먼저 구할 필요는 없습니다.

(coherent-and-incoherent)=
### 서로 비결맞는 입사파의 전력

앞의 예제들은 상대 위상을 정한 결맞는 입사입니다. **별도로 계산한 결과를 합칠 때도 결맞는 입사라면 복소 전기장·자기장을 먼저 합산해야 합니다.** 각 계산의 반사율·투과율만 더하면 두 파 사이의 간섭을 반영할 수 없습니다.

**서로 비결맞는 입사파는 상대 위상이 불규칙하게 변하여, 두 파 사이의 간섭항이 관측 시간 동안의 평균에서 사라집니다.** 이 경우에는 각 입사파를 따로 계산한 뒤, 반사 전력끼리 더하고 투과 전력끼리 더하면 됩니다. 각 입사파가 구조 안에서 겪는 다중반사와 간섭은 각각의 RCWA 계산에 이미 포함되어 있습니다. 시간평균에서 사라지는 것은 서로 비결맞는 두 입사파 사이의 간섭항입니다.

이때 전력과 효율을 구분합니다.

- **반사·투과 전력:** 각 입사파가 만든 전력을 더합니다.
- **반사율·투과율:** 더한 출력 전력을 총 입사 전력으로 나눕니다. 이는 각 입사파의 입사 전력 비율로 가중 평균하는 것과 같습니다.

두 입사파의 입사 전력을 P<sub>1</sub>, P<sub>2</sub>라고 하고, 따로 계산한 전체 반사율을 R<sub>1</sub>, R<sub>2</sub>, 전체 투과율을 T<sub>1</sub>, T<sub>2</sub>라고 하면 다음과 같습니다. 한 입사파씩 일반 `rcwa()` 입력으로 계산한 전체 반사율과 투과율은 `result.R.sum()`, `result.T.sum()`으로 읽습니다.

```{raw} html
<div class="power-equations" style="text-align:center">
<p>R = (P<sub>1</sub> R<sub>1</sub> + P<sub>2</sub> R<sub>2</sub>) / (P<sub>1</sub> + P<sub>2</sub>)</p>
<p>T = (P<sub>1</sub> T<sub>1</sub> + P<sub>2</sub> T<sub>2</sub>) / (P<sub>1</sub> + P<sub>2</sub>)</p>
</div>
```

다음은 서로 비결맞는 두 입사파의 반사율을 각각 20%, 60%로 가정하여 가중 평균을 계산하는 예시입니다.

| 입사파 | 입사 전력 | 따로 계산한 반사율 | 반사 전력 |
| --- | --- | --- | --- |
| 1 | 3 W | 20% | 3 W × 0.2 = 0.6 W |
| 2 | 1 W | 60% | 1 W × 0.6 = 0.6 W |

총 입사 전력은 4 W이고, 총 반사 전력은 1.2 W이므로 합친 반사율은 1.2 / 4 = 30%입니다. 두 입사파의 입사 전력이 같을 때는 반사율의 단순 평균과 같습니다. 투과율도 같은 방법으로 구합니다.

`excitation`에 복소 진폭을 지정하면 두 파의 위상차가 고정된 상태를 계산합니다. 따라서 위의 비결맞는 입사를 계산하려면 각 입사파의 RCWA 결과를 따로 구한 뒤 전력을 합산합니다.

(illumination-power-region)=
(beams-detail-id22)=
### 전력을 어느 면적에서 평균할까?

서로 다른 계산 묶음의 파를 합치면 세기 분포가 구조의 한 셀마다 똑같이 반복되지 않을 수 있습니다. 따라서 두 묶음 이상일 때는 `power_region`으로 전력을 평균할 영역을 지정합니다.

| 설정 | 평균하는 영역 |
| --- | --- |
| `power_region="cell"` | 셀 중심을 기준으로 한 격자 셀 하나. 계산 묶음이 하나일 때의 기본값입니다. |
| `power_region="infinite"` | 표면 전체에 대한 무한 면적 평균. 서로 다른 횡파수의 간섭항은 이 공간 평균에서 사라집니다. |
| `power_region=((-400., 400.), (-300., 300.))` | x가 −400∼400, y가 −300∼300인 직사각형. 단위는 구조의 길이 단위와 같습니다. |

무한 면적 평균에서 간섭항이 사라지는 것은 **공간에 대한 평균** 때문입니다. 두 입사파가 비결맞는 빛으로 바뀌는 것은 아니며, 위치별 필드에는 간섭이 남습니다. 유한한 직사각형에서는 그 영역에 걸친 간섭까지 반영한 전력을 구합니다.

양쪽 바깥 매질이 무손실이고 셀마다 장의 세기가 반복되거나 무한 면적 평균을 사용하면, `absorption = 1 - output_top - output_bottom`입니다. 바깥 매질에 손실이 있으면 입사파와 출력파 사이의 간섭도 순 전력에 기여할 수 있으므로 `result.absorption`으로 읽습니다. 이 전력 기준은 {ref}`13장 <half-space-power>`에서 설명합니다.

유한한 관측 영역은 옆면으로도 전력이 드나들 수 있습니다. 따라서 직사각형 평균이나 여러 묶음의 한 셀 평균에서는 위·아래 경계의 전력 차이를 전체 흡수로 부르지 않으며, `absorption`은 `None`입니다. 이때도 `result.power.net_into_stack`으로 아래 경계의 전체 +z 전력 흐름에서 위 경계의 전체 +z 전력 흐름을 뺀 값을 읽을 수 있습니다. 유한 영역의 방향별 전력은 간섭에 따라 음수가 될 수도 있어, 그 비율을 항상 0∼1인 효율로 해석하지 않습니다.

{ref}`같은 쪽의 두 입사파 예제 <same-side-beams>`에서 구한 필드를 그대로 두고 관측 영역만 바꾸려면 다음처럼 읽습니다. 이 비율의 분모도 해당 직사각형을 지나는 입사파만의 평균 전력 흐름입니다. 이처럼 계산 후 관측 영역을 바꾸려면 `answers="amplitude"`(기본값) 또는 `"field"`로 계산해 두어야 합니다. `"efficiency"`에서는 처음 지정한 영역의 전력만 `result.power`에 보관합니다.

```python
window_power = same_side_result.boundary_power(
    ((-450., 450.), (-425., 425.)),     # x 범위, y 범위 (nm)
)
print(window_power.output_top, window_power.output_bottom)
```

(source-expansions)=
## 가우시안 빔과 내부 소스로 확장하기

앞에서는 입사 평면파를 직접 지정했습니다. `GaussianBeam`은 폭과 초점으로 지정한 빔을 여러 평면파로 구성하고, `ElectricDipole`은 층 내부의 전류원으로 빛을 만듭니다. 입력이 들어가는 위치는 다르지만, 구한 복소 전기장·자기장을 합하여 전력을 계산한다는 원리는 같습니다.

(gaussian-beam-source)=
### 폭이 있는 가우시안 빔 넣기

가우시안 빔은 여러 방향의 평면파를 합하여, 중심에서 진폭이 크고 옆으로 갈수록 작아지는 입사장을 만듭니다. `GaussianBeam`은 중심 방향 주변의 여러 평면파에 가우시안 가중치를 주고, 각 파의 전기장이 그 파의 진행 방향에 수직이 되도록 구성합니다. 각 성분을 계산한 뒤 복소 E·H를 합칩니다.

| 설정 | 의미 |
| --- | --- |
| `side`, 방향·편광 | `PlaneWave`와 같은 규약. 방향은 빔의 중심 방향입니다. |
| `waist` | 초점에서 스칼라 가우시안 포락선의 진폭이 중심의 1/e가 되는 반지름. 구조와 같은 길이 단위입니다. |
| `focus=(x,y,z)` | 초점의 전체 구조 기준 좌표입니다. |
| `amplitude` | 구조가 없다고 보았을 때 초점의 복소 전기장 벡터 크기와 공통 위상. `1.`은 크기 1입니다. |
| `sampling=(Nx,Ny)` | 중심 방향에 수직인 두 파수 방향을 몇 점씩 나눌지 정합니다. 기본값은 `(9,9)`입니다. |
| `spectral_extent` | 두 파수 방향을 각각 `±spectral_extent/waist`까지 포함합니다. 기본값은 `4.`입니다. |

평면파 성분의 가중치는 `exp(-waist**2 * (ku**2 + kv**2) / 4)`에 비례합니다. 여기서 `ku`, `kv`는 빔의 중심 방향에 수직인 두 파수 성분입니다. 소스는 입사 매질에서 전파할 수 있고 구조 쪽으로 향하는 성분만 사용합니다. 무손실 양의 유전율·투자율을 가진 등방성 입사 매질을 지원합니다. 아주 좁거나 크게 기울어진 빔은 이 성분 선택과 전기장의 수직 조건 때문에 스칼라 가우시안 모양과 차이가 생길 수 있습니다.

```python
from wavebit.compat import legacy as wavebit

gaussian_solver = wavebit.rcwa(
    wavelength=600., period=(900., 850.),
    top=1., bot=2.25,
    structure=[dict(background=4., shapes=[])],
    thickness=[100.], truncation_order=0,
    excitation=[wavebit.GaussianBeam(
        side="top", theta=0., phi=0., polarization=(1., 0.),
        waist=1500., focus=(0., 0., 100.), sampling=(9, 9),
    )],
)
gaussian_result = gaussian_solver.solve(
    power_region=((-3000., 3000.), (-3000., 3000.)),
)
print(gaussian_result.output_top, gaussian_result.output_bottom)
```

`GaussianBeam`에는 유한한 직사각형 `power_region`을 지정합니다. 이 결과는 지정한 직사각형에서 모은 출력 전력을 같은 직사각형의 입사 전력으로 나눈 값입니다. 관측 영역 밖으로 빠져나간 빛까지 포함한 전체 빔의 반사율·투과율을 구하려면 영역을 넓히며 수렴을 확인해야 합니다. `sampling`과 `spectral_extent`도 함께 확인합니다. 유한한 파수 간격은 공간에서 반복되는 빔의 근사가 되므로, 샘플 수를 늘려 이웃 복사본의 영향을 줄입니다.

**설계 참고: FMMAX**

Wavebit의 가우시안 빔 계산은 FMMAX의 [Brillouin zone integration 설명](https://github.com/facebookresearch/fmmax/blob/main/README.md#brillouin-zone-integration)과 [FMMAX 가우시안 빔 예제](https://github.com/facebookresearch/fmmax/blob/main/examples/crystal.py)를 참고하여 설계했습니다. 참고한 핵심은 **여러 파수 성분을 계산한 뒤 복소 장을 합산하여 폭이 있는 빔을 표현하는 방식**입니다.

(electric-dipole-source)=
### 층 내부에 전기 다이폴 놓기

층 내부의 한 점에서 빛을 내는 전기 쌍극자는 `ElectricDipole`로 지정합니다. 사용자가 모멘트의 크기·방향·위상을 정하는 소스입니다. 재료의 반응으로 모멘트가 자동 결정되는 입자를 뜻하지 않습니다. 시간 의존성을 `exp(-iωt)`로 쓰면 전류원은 `J = -iω p δ(r-r₀)`이며, `p`가 `moment`입니다.

| 설정 | 의미 |
| --- | --- |
| `position=(x,y,z)` | 구조 전체의 좌표로 지정한 위치. 층 내부에 있어야 합니다. |
| `moment=(px,py,pz)` | xyz 방향의 복소 전기 쌍극자 모멘트. 단위는 C·m이며, 각 성분에 진폭과 위상이 들어갑니다. |
| `repetition="periodic"` | 격자 셀마다 반복되는 다이폴 배열입니다. |
| `bloch_k=(kx,ky)` | 반복 배열에서 이웃 셀 사이의 위상을 정합니다. 격자 이동 R에 대해 모멘트에 `exp(i k∥·R)`을 곱한 위상 관계를 가집니다. 생략하면 같은 위상입니다. |
| `repetition="isolated"` | 여러 Bloch 파수의 결과를 적분하여 고립된 다이폴을 근사합니다. |
| `sampling=(Nx,Ny)` | 고립 소스의 Bloch 파수 적분 격자. 기본값은 `(5,5)`이며 수렴을 확인해야 합니다. |
| `rcwa(length_unit=...)` | 구조 길이 한 단위를 미터로 환산한 값. nm이면 `1e-9`, μm이면 `1e-6`입니다. |

소스가 있는 층은 `shapes=[]`로 지정하거나, 모든 공간 칸에 같은 재료를 넣은 래스터로 표현합니다. 다른 층에는 패턴이 있어도 됩니다. 균일한 이방성·자성 층도 사용할 수 있습니다. 계산기는 소스 높이에서 층을 나누고 전류 항을 넣으므로 사용자가 층을 직접 나누지 않습니다. 소스를 층 경계에 놓거나 소스와 같은 높이의 장을 읽는 경우는 받지 않습니다. 점 소스의 특이장과 경계에 놓인 소스는 별도의 정의가 필요하기 때문입니다.

```python
import numpy as np
from wavebit.compat import legacy as wavebit

dipole_solver = wavebit.rcwa(
    wavelength=600., period=(900., 850.),
    top=1., bot=2.25,
    structure=[dict(background=1., shapes=[])],
    thickness=[300.], truncation_order=dict(x=1, y=1),
    length_unit=1e-9, answers="field",
    excitation=[wavebit.ElectricDipole(
        position=(0., 0., 150.),
        moment=(0., 1e-29, 0.),
        repetition="periodic", bloch_k=(0., 0.),
    )],
)
dipole_result = dipole_solver.solve(power_region="cell")
Z0 = 376.730313668
print("위쪽 평균 전력 밀도 (W/m²):", dipole_result.power.outgoing_top / Z0)
print("아래쪽 평균 전력 밀도 (W/m²):", dipole_result.power.outgoing_bottom / Z0)
dipole_field = dipole_result.reconstruct_field(
    sample_x=np.linspace(-400., 400., 41), sample_y=[0.],
    sample_z=[50., 100., 200., 250.],
)
```

이 예제는 셀마다 같은 위상으로 진동하는 y 방향 다이폴 배열입니다. 실제로 한 개만 있는 소스를 근사하려면 `repetition="isolated"`로 바꾸고 `bloch_k`를 생략합니다. 이때 `power_region`에는 유한한 관측 직사각형을 명시하고, 적분 격자와 Fourier 차수를 늘리며 결과를 확인합니다. 유한한 적분 격자는 큰 주기 셀에 반복된 소스와 같으므로 그 반복 간격보다 작은 영역에서 고립 소스의 근사를 확인합니다.

다이폴 입력의 필드는 E가 V/m, 마지막 세 성분은 Z₀H입니다. `result.power.outgoing_top`·`outgoing_bottom`은 Z₀로 나누기 전의 평균 전력 밀도입니다. 셀 전체의 출력 전력 W를 구하려면 Z₀로 나눈 값에 셀의 면적(m²)을 곱합니다. 관측 직사각형의 경우에는 그 직사각형의 면적을 사용합니다.

내부 소스가 있으면 외부 입사 전력만으로 전체 출력을 정규화할 수 없으므로 `output_top`, `output_bottom`, `absorption`은 `None`입니다. `result.power`의 방향별 전력과 `net_into_stack`은 읽을 수 있습니다. 이때 경계의 순 전력은 흡수와 소스의 공급 전력을 함께 반영하므로 흡수율로 해석하지 않습니다. 소스가 공급하는 전력, 자가장 정규화, Purcell factor는 이 API에서 반환하지 않습니다.

`PlaneWave`와 `ElectricDipole`을 같은 소스 목록에 넣으면 두 소스의 복소 장이 합쳐집니다. 이 경우 평면파의 `amplitude`도 V/m 기준으로 지정하여 단위를 맞춥니다.

(beam-input-support)=
(planewave)=
## 지원 범위와 필드 읽기

`PlaneWave`, `OrderExcitation`, `GaussianBeam`, `ElectricDipole` 소스 객체를 `excitation`에 넣는 방식은 **1차 정식화(`first_order`, 옛 이름 4ff)와 SMM**으로 계산합니다. NumPy·JAX·PyTorch의 일반 실행을 지원하며, `jit`·`vmap`, 자동미분과 `JobSpec` 저장은 지원 범위에 포함되지 않습니다. `torch.compile` 안에서는 컴파일된 호출자 안에서 즉시 실행(eager)됩니다. 위·아래 반무한 매질은 등방성이어야 합니다. 가우시안 빔의 입사 매질과 다이폴이 놓이는 층의 조건은 각 소스 절에서 설명한 범위를 따릅니다.

증폭 매질의 해 선택은 이 소스 API에서 별도로 정의하지 않습니다. 계산 차수가 정확히 스침 입사 조건에 놓여 행렬이 특이해지면 파장을 자동으로 바꾸지 않으므로, 가까운 파장이나 입사 조건에서 결과의 수렴을 확인합니다. [상세 API와 계산 정의(영문)](../../guide/rcwa/coherent-illumination.md)에서 입력값과 결과 객체를 더 자세히 확인할 수 있습니다.

(id20)=
### 경계면과 층 내부의 장 읽기

이 장에서는 소스를 지정하고 출력 전력을 읽었습니다. 경계면에서 입사파·출력파를 따로 보거나 층 내부의 장을 구하려면 {ref}`14장의 PlaneWave 필드 예제 <illumination-fields>`로 이어갑니다. 복소 필드 배열의 성분, 좌표 입력과 자기장 단위도 그 장에서 설명합니다. 전력 흐름과 회절 효율의 정의는 {ref}`13장 <half-space-power>`에서 다룹니다.
