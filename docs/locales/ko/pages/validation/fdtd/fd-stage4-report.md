# 4단계 보고: 모양과 정확도 (C0-C12)

작성: 2026-10-27

대상: `src/wavebit/fdtd/_shapes/__init__.py`, `src/wavebit/fdtd/_prepare/occupancy.py`,
`src/wavebit/fdtd/_prepare/coefficients.py`, `src/wavebit/fdtd/_prepare/constitutive.py`,
`src/wavebit/fdtd/_prepare/checks.py`, `src/wavebit/fdtd/_prepare/junction.py`,
`src/wavebit/fdtd/_prepare/clip.py`, `src/wavebit/fdtd/_prepare/stencil.py`,
`src/wavebit/fdtd/_prepare/shape_ranges.py`, `tests/fdtd/test_fd_shape_protocol.py`,
`tests/fdtd/test_fd_shapes.py`, `tests/fdtd/test_fd_occupancy.py`, `tests/fdtd/test_fd_bodies.py`,
`tests/fdtd/test_fd_clip.py`.

범용 FDTD 엔진(`wavebit.fdtd`)의 4단계 목표는 휘어지고 회전한 물체를 칸 아래 평균
(subpixel smoothing)으로 다루고, 모서리를 미분할 수 있게 하고, 흡수층까지 이어지는 물체를
다루는 것이었으며, 정확한 Mie 이론(합격선)과 RCWA 대비 메타표면으로 채점했습니다. 커밋
열세 개(C0~C12)로 계획했습니다. 이 문서는 각 커밋의 지금 상태와 근거의 위치를 적습니다.
**완료 주장이 아니라 상태 기록입니다. 등록한 관문 두 개(C7, C10의 이량체 갈래)를 통과하지
못했고, C3은 선택 사항인 스텐실로만 통과했으며, 커밋 세 개는 일부만 되었고, 하나는 소유자에게
남겼습니다.**

## 한눈에 보는 상태

| | 내용 | 상태 | 근거 |
| --- | --- | --- | --- |
| C0 | 모든 모양이 답하는 모양 규약(지지 영역, 채운 칸 또는 물체) | 끝 | `src/wavebit/fdtd/_legacy_geometry.py`의 규약; `tests/fdtd/test_fd_shape_protocol.py` |
| C1 | 기하 계산부 `_shapes/`: 정확한 채움, `<n n^T>` 모멘트, 거리 위의 CSG(내부용) | 끝 | [smoothed occupancy](fd-subpixel-occupancy.md); `tests/fdtd/test_fd_shapes.py` |
| C2 | Yee 성분별 점유, `Sphere`, 섞는 규칙 | 끝 | [smoothed occupancy](fd-subpixel-occupancy.md) |
| C3 | **합격선: 구 하나 대 Mie** | **`discretisation='stencil'`일 때만 통과**; 기본 복셀 규칙은 원기둥에서 실패 | [Mie 검증](fd-mie-validation.md) |
| C4 | `Cylinder`, `Prism`, 회전한 `Box`, `Box`/`Region` 이름 정리 | 끝 | [smoothed occupancy, C4](fd-subpixel-occupancy.md#c4-the-extruded-bodies-2026-09-26) |
| C5 | 거부 규칙과 `check` 출력; star lift | 끝 | [smoothed occupancy, C5와 star lift](fd-subpixel-occupancy.md) |
| -- | 맞붙은 접합(`junction='split'`, 기본값) | 끝 | [smoothed occupancy, 접합](fd-subpixel-occupancy.md#the-junction-a-body-standing-flush-on-a-face) |
| C6 | 미분 가능한 모서리(`range=`), 자동 미분 대 중심 차분 | 끝 | [smoothed occupancy, C6 절](fd-subpixel-occupancy.md); [미분 가이드](../../guide/fdtd/fd-differentiation.md) |
| C7 | **RCWA 대비 메타표면** | **불합격** (M3 차수: 진폭 1.64 / 1.75, 효율 1.51 / 1.62, 1.8 미만) | [메타표면 검증](fd-metasurface-validation.md) |
| C8 | 분산·손실 재료의 칸 평균 | **일부**: 평평한 막과 떨어진 막 쌓기만 | 아래 C8 막·쌓기 문서 |
| C9 | 흡수층을 지나가는 구조 | **좁은 길만** | [정합 흡수](fd-constitutive-absorber.md); [기판 연장](../../guide/fdtd/substrate-gradients.md) |
| C10 | 한 칸의 두 매끈한 표면(다면체 자르기) | **일부**: 한 표면이 평평한 곳에서 답함; L 갈래 통과, **이량체 관문 실패**, 두 곡면은 계속 거부 | [다면체 자르기](fd-polytope-clip.md) |
| C11 | 대각 밖 구성 칸 | **작성 안 함; 규칙으로 닫히지 않음; 소유자에게 남김** | [C11 기록](fd-stage4-c11.md) |
| C12 | 이 문서들과 보고 | 이 문서 | -- |

## C0-C2: 규약, 계산부, 규칙

모양은 각 장 성분의 자기 Yee 점마다, 그 성분의 칸을 얼마나 채우는지와 그곳 표면 법선의
이차 모멘트를 답합니다. 계수는 Farjadpour/Kottke 유효 역텐서의 대각으로 물체의 재료를 칸에
섞습니다. 기하 계산부의 채움은 독립 기준과 대조한 정확한 닫힌 꼴(공-상자 자르기, 잘린 원과
다각형 넓이, 정육면체-반공간 자르기)입니다. CSG 노드(`Union`, `Intersection`, `Difference`)는
`_shapes/` 안에 있고, 정확한 자르기가 없는 칸에서는 보고되는 한계값으로 답합니다. **공개되지
않았고**, C10도 이를 바꾸지 않습니다(아래).

## C3: Mie 합격선 -- 스텐실로만 통과

등록한 관문은 `fd.Structure(..., discretisation='stencil')`에서 통과합니다. 유전율 12 구는
파장당 40/56/80칸, 세 위치에서 평균 / RMS 차수 2.2442819 / 2.2443152(흩어짐 1.48%)로
수렴하고, TE 유전율 12 원기둥은 2.137819입니다. **기본 복셀 규칙은 통과하지 못합니다.** 구의
모든 갈래는 통과하지만(유전율 12에서 1.8268534, 80칸에서 -6.18%), TE 유전율 12 원기둥에서 차수
1.6033652로 실패합니다. 스텐실은 한 번에 `Sphere` 하나의 장면에만 적용되고, 일반 숫자만 받으며
(미분 없음), `PlaneWave`나 `TFSF` 광원이 필요합니다. 유리 구를 1.714에서 2.023으로 올린 것은
기본 규칙의 수반 교차 결합(공통 마디의 대각 밖 항, C11 참고)입니다.

## C4-C6: 물체, 거부, 미분

`Cylinder`, `Prism`, 회전한 `Box`는 정확한 채움으로 나옵니다. 한 물체 안의 꺾인 곳은 모두
매질 둘이고 정확합니다. 거부는 무엇을 거부하는지 이름을 대고, 문을 열어 두었을 때 장면이 읽는
수를 함께 적습니다. star lift는 매끈한 연산자를 구성상 양의 정부호로 만듭니다. 기하 미분
(`range=`)은 구, 원기둥, 상자, 프리즘에서, 평면파 조명 아래에서, 떨어진 여러 물체와 기판 위
기둥에서 쓸 수 있으며, 각각 JAX와 Torch에서 중심 차분과 대조했습니다.

맞붙은 접합 -- 회전하지 않은 매끈한 `Box`의 면 위에 선 `Cylinder`나 회전하지 않은 `Box` -- 은
자체 측정 규칙(`split`, 기본값; 평평한 막 기준에서 2차, 기둥 장면 하나에서 테두리 차수 미해결)과
대안(`staircase`, `refuse`)을 가집니다.

## C7: RCWA 대비 메타표면 -- 불합격

등록한 규칙으로 2026-10-03 측정은 통과하지 못합니다. M1(유전율 4 원기둥)은 두 편광 모두
통과합니다(차수 1.90 s, 1.97 p). **M3(유전율 12 상자, 원뿔 입사)는 차수 기준에 실패합니다.**
진폭 차수 1.64(s), 1.75(p), 효율 1.51 / 1.62로 1.8 미만이며, 54칸 모두에서 RCWA와는
일치합니다. M2 p는 27칸 중 한 칸(RCWA가 대칭 때문에 0으로 두는 교차 편광 항)에서 실패하고,
M2 s는 그 RCWA 사다리로는 판정할 수 없습니다. 등록한 후속 측정은 M3의 낮은 차수가 기둥-기판
테두리를 필요로 하지 않음(유리에서 띄운 상자도 같은 차수로 수렴)을 보였고, xz 단면의
모서리에서는 그것을 찾지 못했으며, y 면은 느슨한 용의자로 남겼습니다. 원인은 밝혀지지
않았고, 기록은 이를 한계로 적습니다.

## C8: 분산·손실 재료의 칸 평균 -- 일부

각각 자체 범위 안에서 구현하고 측정한 것: 평평한 전도 막 하나([전도 막](fd-conductive-film.md)),
떨어진 전도 막들([전도 쌓기](fd-conductive-stack.md))과 정확히 맞닿은 두 막
([전도 접촉](fd-conductive-contact.md)); 평평한 Drude 또는 Lorentz 막 하나
([단일 극](fd-single-pole-film.md)); Debye 막 하나([Debye 막](fd-debye-film.md)), Debye 극
둘([두 Debye](fd-two-debye-film.md)), 유한한 Debye 합([Debye 합](fd-debye-sum-film.md)); 유한한
Drude/Lorentz 합([진동자 합](fd-oscillator-sum-film.md)); Debye와 진동자를 섞은 막
([혼합 막](fd-hybrid-film.md)); 떨어진 Debye 쌓기와 단일 진동자 쌓기
([Debye 쌓기](fd-debye-stack.md), [진동자 쌓기](fd-oscillator-stack.md)). 각 문서는 미해결
칸을 적습니다. 각 범위는 구체값(추적된 재료 값 없음)이고, 축에 나란하며, 손실 없는 유전체
안에 있고, 흡수층에서 떨어져 있습니다. **거부**: 휘거나 회전한 물체 위의 전도·분산 재료 --
매뉴얼의 "평평한 막 밖에서는 매끈한 손실·분산 물체 없음" -- 와 접합의 손실 매질. 그래서
매끈한 곡면 금속(예: 금속 나노입자)은 지금 쓸 수 없습니다.

## C9: 흡수층을 지나가는 구조 -- 좁은 길만

길은 둘입니다. 결합 유전체 지지 영역은 모든 보간 연결이 같은 감쇠율을 이을 때 전도
`Absorber` 안에 놓일 수 있습니다([정합 흡수](fd-constitutive-absorber.md)). 그 물리 보정 실행은
모두 가라앉지 않았고 수렴 결과는 주장하지 않습니다. 회전하지 않고 가로 방향으로 균일한 고정
`Box` 기판은 `Structure.continuation`을 선언하면 `Open()` 면을 지나 이어질 수 있습니다
([기판 연장](../../guide/fdtd/substrate-gradients.md)). **유한하거나 휘어진 매끈한 물체가
PML을 지나가는 것은 거부합니다.**

## C10: 한 칸의 두 매끈한 표면

C10은 5번 문의 거부 일부를 답으로 바꿉니다([기록](fd-polytope-clip.md)). 두 매끈한 물체의
표면이 한 칸을 함께 지나가고 그 칸에서 공유하는 부피가 정확한 곳 -- 떨어져 있음이 증명되거나,
둘 다 평평하거나, 한쪽이 회전하지 않은 `Box`의 면 -- 에서는 각 매질의 채움이 정확하고, 두
경계면에 걸친 투영 하나로 세 매질을 섞습니다. 평행한 층에서는 정확한 적층이고, 한 매질이 없으면
두 매질 규칙과 비트까지 같습니다. 합격 기준은 코드보다 먼저 등록했고, 그것이 바꾸는 측정 전에
세 번 수정했습니다.

* 관문 1, 3, 4 통과: 다면체·구적 기준 대비 채움과 경계면 모멘트, 기존 두 매질 장면 전부 바이트
  동일과 네 기준 결과 재현, 백엔드 일치와 중심 차분 대비 재료 미분.
* **관문 2의 L 갈래 통과**: 모든 격자에서 틈이 칸을 공유하는 평면 적층이 전달 행렬 대비 차수
  1.866(s)과 1.817(p)로 수렴합니다(흩어짐 50%, 63%). 자르기 칸이 없는 대조는 1.852 / 1.856.
* **관문 2의 D 갈래는 두 기준 모두 실패**: S7 이량체(유전체 구 둘, 지름의 0.05 틈; S7의 두 금속은
  아예 매끈하게 다룰 수 없음)는 자르기가 쓰는 거친 격자에서 해상도 문에 거부되고, 실행되는
  곳에서는 오차가 더 촘촘한 격자로 맞춘 법칙의 0.08에서 0.49배로 등록한 2배 범위 밖입니다.
  **등록한 대로, 두 곡면이 함께 지나가는 칸은 계속 거부합니다.**
* S7의 맞붙은 기둥은 계속 접합이 맡습니다: 자르기 기록이 없고 계수는 바이트까지 같습니다.

그래서 C10은 `Box` 면에 놓이거나 바로 위에 뜨거나 잠긴 물체와 나란한 상자·프리즘을 답하고,
곡면 물체의 이량체, 접합이 다루지 않는 맞붙음, 곡면의 겹침, 같은 재료의 합집합은 거부합니다.
**공개 CSG(`fd.Union`, `fd.Difference`)는 내부에 그대로 둡니다.** C10은 CSG 노드를 정확하게
만들지 않으므로(두 자식의 표면이 만나는 곳에서는 여전히 한계값으로 답함) 그로부터 따라 나오는
것이 없습니다.

## C11: 대각 밖 구성 칸 -- 소유자에게 남김

이 칸은 측정 S1과 S2가 대각 밖 항이 차수를 올리지 못한다고 보이면 작성하지 않도록 등록했습니다.
S1(기울어진 평평한 경계면)은 그렇다고 말하고, S2(2차원 Mie 원기둥)는 대비 12의 TE 갈래에서
대각 규칙이 실패함을 보이며, 교차항은 유리 구를 1.714에서 2.023으로 올렸습니다. 그래서 규칙의
전제가 충족되지 않고, C11은 그 규칙으로 닫히지 않으며, 아무것도 만들지 않았습니다. 등방성
교차 결합은 (C3 작업에서) 이미 엔진에 있습니다. C11이 여전히 뜻할 수 있는 것 -- 매끈한 칸 안의
이방성 재료, 또는 고대비 원기둥을 위한 다른 대각 밖 구현 -- 은 측정되지 않았고
[C11 기록](fd-stage4-c11.md)에 적었습니다.

## 4단계가 남기는 것

* 기본 복셀 규칙의 TE 유전율 12 원기둥(교차항으로 차수 1.603, 1.8 미만);
* C7 M3의 차수와 밝혀지지 않은 원인;
* 평평한 막과 쌓기를 넘는 분산·손실 칸 평균(곡면 금속);
* PML을 지나가는 유한하거나 휘어진 물체;
* C11 결정;
* 한 칸의 두 곡면(C10의 실패한 이량체 관문), C10 기록에 적은 다른 거부들, 그리고 공개 CSG.
