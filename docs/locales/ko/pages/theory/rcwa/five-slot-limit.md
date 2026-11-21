# 아홉 칸 중 다섯 칸 — 모든 2차 RCWA가 갖는 형태와 그 경계에서 일어나는 일

작성: 2026-08-11. CPU, float64에서 측정했습니다. 이 솔버는 NumPy 2.5.1로, 각 기준 솔버는 현재 릴리스
(fmmax 1.7.1, inkstone 0.3.15, EMpy 2.2.3, RETICOLO V10 2025-01)로 실행했으며, 버전은 메모가 아니라
PyPI와 Zenodo에서 확인했습니다.

대상:
- `src/wavebit/rcwa/_engine/solve_2d.py::layer_modes_2d` — 이 함수가 읽는 다섯 블록
- `src/wavebit/rcwa/_engine/solve_2d_first_order.py` — 아홉 칸을 모두 읽는 정식화
- `tests/rcwa/test_anisotropic_berreman.py::test_voigt_geometry_matches_the_oracle` — 그리고 짝을 이루는
  `::test_a_z_row_is_refused_by_the_formulation_that_cannot_hold_it`
- `tests/references/README.md` — §4에 요약한 솔버별 측정

인접한 질문에 답하는 두 문서의 짝 문서입니다.
[`anisotropic-materials.md`](../../guide/rcwa/anisotropic-materials.md)는 솔버가 무엇을 받아들이고 각 공백이
어떤 비용을 치르는지 묻습니다. [`off-diagonal-slot-convention.md`](off-diagonal-slot-convention.md)는 칸에
손으로 무엇을 넣어야 하는지 묻습니다. **이 문서는 둘 다 묻지 않는 질문을 다룹니다. 다섯 칸 형태는 Wavebit의
결정이 아니라 모든 2차 RCWA가 갖는 것입니다. 그렇다면 나머지 네 성분은 무엇을 뜻하며, 각 솔버는 그것을
건네받았을 때 무엇을 할까요?**

---

## 요약

1. 유전율 텐서에는 성분이 아홉 개 있습니다. 2차 RCWA는 그중 **다섯 개**, 즉 `xx, xy, yx, yy` 블록과
   `zz`를 읽습니다. 나머지 넷 — `xz, yz, zx, zy` — 에는 칸이 없으며, 이는 누락이 아니라 구조적인 것입니다.
2. `zz`는 *역수로* 남습니다. `E_z`를 소거하려면 `eps_zz⁻¹`가 필요하기 때문입니다. 이 소거는
   **`xz`와 `yz`가 0일 때에만** 2차 방정식으로 닫힙니다. 그렇지 않으면 1차 미분 항이 남고, 쓸 수 있는
   2차 연산자가 없습니다.
3. 빠진 네 성분은 **Voigt 기하** — 자화가 면 안에 놓인 경우 — 이며, 평범한 자화된 박막이 바로 이
   경우입니다.
4. 기준 솔버 네 개를 현재 릴리스로 실행했습니다. **완전한 3×3을 지원한다고 주장하는 네 개 중 세 개는 실제로
   그것을 제공하지 않으며**, 실패 방식도 거부, 조용한 누락, 발산으로 세 가지가 서로 다릅니다.
5. **에너지 보존으로는 조용한 누락을 잡아낼 수 없습니다.** 축소된 텐서도 유효한 무손실 매질이므로
   `R + T = 1`은 기계 정밀도까지 성립하지만 답은 최대 6e-2만큼 틀립니다. Wavebit이 결코 성분을 버리지 않는
   이유가 이것입니다. 네 성분이 있으면 1차 정식화가 선택되며, z 행에 대해 2차 정식화를 요청하는 유일한
   방법(`formulation='second_order'`)은 예외를 발생시킵니다.

---

## 1. 왜 다섯이며, 왜 하필 그 다섯인가

`e^{-iwt}`와 `d/dx -> i k0 Kx` 아래에서 `curl H = -i k0 eps E`의 `z` 성분은 다음을 줍니다.

```
    D_z = ezx Ex + ezy Ey + ezz Ez  =  Ky Hx - Kx Hy
```

따라서 `E_z`는 독립 미지수가 아닙니다. 이것을 풀어서 대입해 없앨 수 있습니다.

```
    Ez = ezz⁻¹ (Ky Hx - Kx Hy - ezx Ex - ezy Ey)
                 └──────┬──────┘  └───────┬──────┘
              always present         only if the z row is populated
```

다섯 번째 칸은 여기서 나오며, 그것이 역수로 저장되는(`epz_conv_i`) 이유도 여기에 있습니다. 솔버는
`eps_zz`가 아니라 `eps_zz⁻¹`만 필요로 하고, 그 역수는 Li의 inverse 규칙이 적용되는 자리입니다. 다시
대입하면 다음이 남습니다.

```
                        E_x         E_y         E_z
                   ┌───────────┬───────────┬───────────┐
           D_x     │    epx    │   epxy    │     ○     │
                   ├───────────┼───────────┼───────────┤
           D_y     │   epyx    │    epy    │     ○     │
                   ├───────────┼───────────┼───────────┤
           D_z     │     ○     │     ○     │  epz_i⁻¹  │
                   └───────────┴───────────┴───────────┘

    named  = a slot layer_modes_2d reads      ○ = assumed zero, no slot exists
```

`○` 칸이 0이면 대입 결과는 `d²E_t/dz² = Omega2 E_t`가 됩니다. 2차이고, 1차 미분 항이 없으며, 순방향과
역방향 모드가 하나의 고유벡터를 공유합니다. **마지막 성질이 2차 계산 체계 전체가 딛고 선 토대이며**, 빠진 네
성분이 빼앗아 가는 것이 바로 이것입니다.

## 2. 빠진 넷은 무엇인가

`eps_xz`와 `eps_yz`는 면내 필드를 적층 법선에 결합합니다. 자기광학 매질에서는
`eps_ij = eps·δ_ij + i·ε_ijk·g_k`이므로, 자화 방향이 어느 칸이 채워지는지를 정합니다.

| 자화 | 채워지는 칸 | 2차로 가능? | 이름 |
|---|---|---|---|
| `z`(적층 법선) 방향 | `xy`, `yx` | 예 | Faraday |
| 면 안(`x` 또는 `y`) | `yz, zy` 또는 `zx, xz` | **아니요** | Voigt |
| 기울어짐 | 비대각 여섯 개 전부 | **아니요** | — |

가운데 행이 평범한 경우입니다. 자화된 박막은 수직 자기 이방성이 있거나 외부 자기장이 자화를 면 밖으로 붙잡아
두지 않는 한 *자기 자신의 면 안으로* 자화됩니다. 그러므로 2차 정식화가 표현할 수 없는 기하는 특이한 것이
아니라 기본값입니다. 물리는 `anisotropic-materials.md` §6에 있고, 이 문서는 경계 그 자체를 다룹니다.

## 3. 이 형태는 우리 것이 아니다

다섯 칸을 Wavebit의 단순화로 읽고 싶어지기 쉽습니다. 그렇지 않습니다. 이것은 2차 정식화의 *본질* 이며, 2차
정식화를 쓰는 곳이라면 어디에서나 같은 다섯 칸이 나타납니다.

| | 다섯 칸이 나타나는 방식 |
|---|---|
| Wavebit 2차 | `solve_conv(epx, epy, epz_i, epxy, epyx)` |
| fmmax | `eigensolve_anisotropic_media(..., permittivity_xx, xy, yx, yy, zz, ...)` |
| S4 | `xy` 블록과 `zz` |

fmmax의 `eigensolve_general_anisotropic_media`는 `mu`를 위한 다섯 개를 더할 뿐 z 행을 위한 것은 없습니다.
**그곳의 "General"은 z 결합이 아니라 자성을 뜻합니다.** 이 저장소의 자체 조사가 한때 이를 "완전 이방성"으로
기록했고 한동안 틀린 채로 있었기 때문에 밝혀 둘 가치가 있습니다.

## 4. 경계를 만나는 네 가지 방식

아래의 각 솔버에는 현재 릴리스에서 z 행이 채워진 텐서를 건넸습니다. 모든 행은 두 가지 검사로 판정되며,
어느 쪽도 다른 솔버의 동의를 필요로 하지 않습니다.

- **무손실.** 실수 대칭 `eps`는 에르미트이므로 무손실이고, 따라서 입사 편광마다 `R + T = 1`이 *정리에 의해*
  성립합니다. 실수 대칭 행렬은 전치와 어떤 축 순열 아래에서도 실수 대칭으로 남으므로, 규약 차이로는 위반을
  설명할 수 없습니다.
- **사용 여부.** `xz, zx, yz, zy`를 0으로 두고 비교합니다. 출력이 같으면 그 성분들은 무시된 것입니다.

| 솔버 | 동작 | 근거 |
|---|---|---|
| **Wavebit 2차** | 거부합니다 | 칸 이름을 밝히는 `ValueError`, `formulation='second_order'`를 요청할 때에만 도달 |
| **Wavebit 패턴(PN)** | 거부했으나 2026-09-24부터 처리함 | 2026-08-27부터 z 행을 밝히는 `ValueError`. 그 전에는 이를 **받아들이고 조용히 버렸으며**, 이 표의 바로 아래 행과 같은 동작을 이 솔버가 했던 것입니다. 2026-09-24부터 z 결합 패턴 텐서는 PN 또는 텐서 Li 아래에서 1차 경로를 탑니다([전체 텐서 패턴](../../guide/rcwa/full-tensor-patterns.md)) |
| **fmmax** 1.7.1 | 요청할 수 없음 | `xz/yz/zx/zy` 인수가 존재하지 않음 |
| **inkstone** 0.3.15 | **받아들이고 조용히 버림** | `(3,3)` 배열을 받지만, 출력이 z 행을 0으로 둔 경우와 비트 단위로 같음 |
| **EMpy** 2.2.3 | 읽지만 발산함 | z 행을 쓰지만, 수직 입사에서 평범한 Faraday 텐서에 대해 `R + T = 1.6e13`을 반환 |
| **RETICOLO** V10 | 한계 안에서 처리함 | `delta0 = 0`에서 비대각 쌍 하나에 대해 정확하지만, 그 너머에서는 `R + T`가 1에서 2.5e-3 ~ 1.2e-2만큼 벗어남 |
| **Wavebit 1차** | 처리하며, 여기서는 기본값 | `solve_2d_first_order.py`, RETICOLO의 한계 안에서 RETICOLO와 1e-15까지 일치 — 직접 호출뿐 아니라 공개 API를 통해서도 |

솔버별 수치와 각각을 확립한 방법은
[`tests/references/README.md`](https://github.com/fieldalgo/wavebit/blob/main/tests/references/README.md)에 있습니다.

## 5. 왜 조용한 누락이 위험한가

z 행을 버린다고 엉터리 결과가 나오지는 않습니다. **다른, 그러나 완벽하게 유효한 문제**가 만들어지며 — 원래
매질이 무손실이었다면 무손실 매질입니다 — 호출자가 돌려 볼 만한 값싼 검사는 모두 깨끗하게 통과합니다.

| 검사 | z 행을 버린 Voigt 텐서에서 |
|---|---|
| 풀이가 끝남 | 예 |
| `R + T = 1` | 예, 기계 정밀도까지 |
| 상반성 | 예 |
| 답 | **2e-2 ~ 6e-2만큼 틀림** |

`test_a_z_row_is_refused_by_the_formulation_that_cannot_hold_it`가 존재하는 이유가 이 표이며, 이제 그 기하를
거부하는 대신 *답하는* 테스트에 같은 수치가 인용되는 이유도 이것입니다. 막으려는 실패는 충돌(crash)이
아닙니다. 이는 inkstone이 하는 것으로 측정된 동작과도 정확히 같습니다. 무손실 구성 서른 개가 에너지를
1.3e-15까지 보존하고, z 결합이 없는 경우는 모두 Wavebit과 1e-15까지 일치하며, z 결합이 있는 경우만 조용히
4e-4 ~ 3e-2만큼 벗어났습니다. 거부가 없었다면 Wavebit도 똑같아 보였을 것입니다.

**에너지 보존은 필요조건 검사이지 충분조건 검사가 아닙니다.** 같은 주의가
`tests/rcwa/test_anisotropic_berreman.py`와
[`off-diagonal-slot-convention.md`](off-diagonal-slot-convention.md) §3에도 나오며, 그곳에서는 잘못된 *부호* 가
마찬가지로 모든 전력 효율에서 보이지 않습니다.

## 6. 경계를 넘는 비용

아홉 칸을 모두 읽으려면 2차 축소를 버리고 다음의 1차 계를 써야 합니다.

```
    d/dz (Ex, Ey, Hx, Hy) = i k0 M (Ex, Ey, Hx, Hy)          M is 4N x 4N for N orders
```

이는 층 풀이에서 대략 **8x** 이며(`eig`는 세제곱 비용이고 문제 크기가 두 배가 됩니다), 모든 2차 사용처가
가정하는 공짜 `±` 모드 짝짓기를 포기하게 합니다. 순방향과 역방향이 진정으로 서로 다른 모드 집합이 됩니다.
`src/wavebit/rcwa/_engine/solve_2d_first_order.py`가 이를 구현하며, 2026-08-13부터는 공개 진입점이 실제로
여기에 도달합니다. `solve_conv`는 다섯 칸이 아니라 아홉 칸을 받고, z 행의 네 칸 중 하나라도 주면 1차 경로가
선택됩니다(`formulation='first_order'`는 이를 강제하고, `'second_order'`는 z 행을 버리는 대신 거부합니다).
경계를 넘는 데 8x 외에 들었던 비용은 `mu`와 `layer_algorithm='SMM'`이었습니다. 둘 다 2026-09-24부터 1차
경로에서 동작하며, 무엇을 지원하는지는 모듈의 docstring이 유지합니다.
필드 재구성은 2026-08-14까지 그 목록에 있었으나 지금은 아닙니다.
격자 유형으로는 두 경로를 구분할 수 없기 때문에 `reconstruct_field`가 정식화를 읽습니다.

일반적인 z 결합 경우를 다루는 외부 솔버가 없으므로, 1차 경로는 부분적으로 오라클이 아니라 정리로 판정합니다.
텐서와 입사 방위각을 적층 법선을 축으로 함께 돌리면 효율은 변하지 않아야 합니다
(`tests/rcwa/test_solve_2d_first_order.py::test_rotating_the_whole_problem_about_z_changes_nothing`).

## 참고 문헌

- D. W. Berreman, "Optics in Stratified and Anisotropic Media: 4×4-Matrix Formulation",
  *J. Opt. Soc. Am.* **62**(4), 502 (1972). doi:10.1364/JOSA.62.000502
- L. Li, "Fourier modal method for crossed anisotropic gratings with arbitrary permittivity and
  permeability tensors", *J. Opt. A* **5**, 345 (2003).
- J. P. Hugonin and P. Lalanne, "RETICOLO software for grating analysis", arXiv:2101.00901.
- V. Liu and S. Fan, "S4: A free electromagnetic solver for layered periodic structures",
  *Comput. Phys. Commun.* **183**, 2233 (2012).
