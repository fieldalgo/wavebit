# 코드 자체의 규약으로 유도한 정식화

작성일: 2026-08-11.

대상:
- `src/wavebit/rcwa/_engine/solve_1d.py` — `layer_modes_1d`, `layer_modes_1d_conical`, `solve_1d`,
  `final_matching`
- `src/wavebit/rcwa/_engine/solve_2d.py` — `layer_modes_2d`
- `src/wavebit/rcwa/_engine/factorization.py` — `_circulant_index`, `fourier2d_piecewise_constant`
- `src/wavebit/_common/conventions.py` — 아래의 모든 제곱근
- 논문 초고(`e^{+jωt}`), §10의 대응표를 통해

---

## 요약

Wavebit의 대수는 원래 공학 규약 `e^{+jωt}`로 쓰인 논문 초고를 기준으로 작성되었고, 나중에
RETICOLO와 맞추기 위해 `e^{-iωt}`로 뒤집었습니다. 이 뒤집기는 문제 전체의 복소 켤레이며, 이를
식 하나하나 따라가며 하는 것이야말로 부호 오류가 생겨나는 방식입니다 — 세 개는 이득 매질 버그로,
하나는 분지 절단 결함으로 배포되었습니다(`docs/validation/rcwa/time-convention-audit.md`). 이 문서는
번역 단계를 없앱니다. **맥스웰 방정식에서 회절 효율까지의 전체 유도를 처음부터 `e^{-iωt}`로**
제시하며, 모든 중간량에는 `src/wavebit/rcwa/_engine/`에서 그 값을 담는 변수의 이름을 붙였습니다.
상자로 표시한 각 결과는 이 문서를 작성한 날 코드와 대조해 확인했으며, 그 확인 내용을 결과 옆에
적었습니다.

규약 전체를 먼저 정리합니다.

| 양 | 규약 |
| --- | --- |
| 시간 인자 | `e^{-iωt}`; 필드는 `Re[E(r) e^{-iωt}]` |
| 손실 | `Im(n) > 0`, `Im(eps) > 0` (데이터베이스가 표로 싣는 `n + ik`) |
| 위쪽(+z)으로 가는 파 | `exp(+i k0 kz z)`, `Re(kz) > 0` |
| +z 방향으로의 감쇠 | `Im(kz) > 0` |
| 평면파 위상 | `exp(+i k0 (kx x + ky y + kz z))` |
| `eps`의 푸리에 급수 | `eps(x) = Σ_p c_p exp(+i 2π p x / Λ)` |
| 차수 벡터 | `kx_m = n_I sinθ cosφ + m λ/Λx`, `m = -M..M` (`ky_n`도 같음); `n_I`는 빛이 들어오는 매질의 부호 있는 굴절률(음굴절률 매질에서는 음수)이며, 흡수하는 입사 매질에서는 대신 위상 굴절률에 `s`를 곱한 값을 씁니다. `s = sign(Re(n_I))` (`docs/reference/common/conventions.md`, "Real-angle incidence") |
| 정규화한 H | `H̃ = Z0 H`, `Z0 = sqrt(μ0/ε0)` |
| 짝 변수 | E가 주 변수일 때 `U = +i H̃_t`, H가 주 변수일 때 `Q = -i E_t` |

아래의 모든 내용은 단위가 없습니다. 길이는 `k0`로 정규화하므로(`z̃ = k0 z`) `d/dz` 대신 `d/dz̃`를
쓰고, 파수 벡터는 `k0` 단위입니다.

**어느 쪽이 아래인가.** 이 유도는 전파 커널 자체의 용어를 그대로 씁니다. "superstrate"는 커널의
입사 매질, `z̃ = 0`은 그 입사면이고, "아래"는 적층 안쪽을 향하는 `+z̃`입니다. 2026-09-21부터 커널의
축은 *곧* 위를 가리키는 공개 `+z`이며, 커널의 입사 포트는 공개 **아래쪽(bottom)** 매질에 묶여
있습니다(`src/wavebit/rcwa/_engine/engine.py`, `_kernel_incidence_side`). 공개적으로 위에서 입사하는
경우는 커널의 거울상 경로로 실행됩니다. 따라서 "아래"는 `+z̃`로, "superstrate"는 거울상이 아닌 그
풀이에서 빛이 들어오는 매질로 읽어야 하며, 공개 용어로 읽으면 안 됩니다. 공개 좌표계가 바뀌어도
대수는 바뀌지 않았고, 그 주변의 이름만 바뀌었습니다.

---

## 1. `e^{-iωt}` 아래의 맥스웰 방정식

`Re[E e^{-iωt}]`에 `∂/∂t → -iω`를 적용하면 다음과 같습니다.

```
∇ × E = -∂B/∂t = +iω μ0 H
∇ × H = +∂D/∂t = -iω ε0 ε E
```

`H̃ = Z0 H`와 `k0 = ω/c`를 대입하면 다음과 같습니다.

```
∇ × E  = +i k0 H̃                                    (M1)
∇ × H̃  = -i k0 ε E                                  (M2)
```

기준점 점검: 진공 평면파 `E = ŷ exp(i k0 z)`, 즉 `+z`로 진행하는 파를 생각합니다. (M1)은
`H̃x = -exp(i k0 z)`를 주고, 시간 평균 포인팅 플럭스는
`S_z = ½ Re(E × H̃*)_z = ½ Re(-Ey H̃x*) = +½`입니다 — 에너지가 `+z`로 흐릅니다. 쌍
`(Ey, H̃x) = (1, -kz)`는 `tests/rcwa/test_anisotropic_berreman.py::_vacuum_modes`가 쓰는 값이며,
이것이 (M1)/(M2) 자체에 대한 코드 쪽 확인입니다.

**`reconstruct_field`는 바로 이 변수들을 반환합니다.** 따라서 이 기준점은 풀이뿐 아니라 그 출력에
대한 진술이기도 합니다. 이 점은 문제가 생기기 전까지는 굳이 적을 필요가 없었습니다. 이 문서에는
필드 재구성에 대한 언급이 없었고, 재구성 쪽에도 이 문서에 대한 언급이 없었으며, 재작성 시점부터
2026-08-14까지 재구성은 `E_z`, `H̃x`, `H̃y`의 부호를 뒤집어 반환했습니다 — `+z` 진공파가
`H̃x = +1`로, 에너지가 `-z`로 흐르는 상태로 돌아와 위의 기준점과 모순되었는데도 테스트 모음의 모든
검사는 통과 상태였습니다. `docs/reference/common/conventions.md`와
`tests/rcwa/test_field_core.py::test_every_path_returns_a_maxwell_field_in_the_declared_frame`를
보세요.

손실 점검: `eps = 1 + iδ`이면 `kz = sqrt(eps) ≈ 1 + iδ/2`이므로 `exp(i k0 kz z)`는 `+z` 쪽으로
**감쇠**합니다. (M1)/(M2)의 이 부호 아래에서 `Im(eps) > 0`은 흡수입니다 — 두 진술은 두 개의 규약이
아니라 하나의 규약입니다.

---

## 2. 푸리에 도구

### 2.1 차수와 결합 부호

주기 층 안의 필드는 차수 `m`에 대한 블로흐 합입니다(1D를 보였으며, `y`도 똑같습니다).

```
Ey(x̃, z̃) = Σ_m  S_m(z̃) exp(+i kx_m x̃),      kx_m = n_I sinθ cosφ + m λ/Λx
```

여기서 `m`은 `-M..M`을 지나며, 이때 `m`의 부호와 위상의 `+i`는 하나의 선택입니다.
코드: `_separable_kxy`가 정확히 이 벡터를 만듭니다(`orders_u = arange(cx - hx, cx + hx + 1)`,
기본 창에서는 차수 0을 중심으로 한 `-M..M`).

유전율도 **같은** 부호로 전개합니다.

```
eps(x) = Σ_p c_p exp(+i 2π p x/Λx)
```

그래서 `eps · E`는 `c_{m-n}`를 통해 차수 `n`을 차수 `m`에 결합합니다. 따라서 합성곱("Toeplitz")
행렬은 다음과 같습니다.

```
[E]_{m,n} = c_{m-n}                                  (F1)
```

코드: `_circulant_index`는 `(r - c)`를 반환하며, 그 docstring은 이 부호가 왜 함정인지 기록합니다.
(F1)을 반대 부호의 차수 벡터와 짝지으면 **에너지는 보존하면서 회절 차수가 거울상으로 뒤집히므로**,
격자 기준 솔버(RETICOLO)만이 이를 잡아낼 수 있습니다. 이 문서는 그 사실을 바꾸지 않으며, 유도가
부호를 명시적으로 지니도록 적어 둡니다.

### 2.2 인수분해 규칙

불연속 인자들의 곱은 절단할 때 나쁘게 수렴합니다(Li). `x` 방향으로 변하는 층의 필드 성분별 규칙은
다음과 같습니다.

| 성분 | `x` 방향 불연속면을 가로지르는 연속성 | 규칙 | 코드 행렬 |
| --- | --- | --- | --- |
| `Ey`, `Ez`가 없는 곱 | 연속 | Laurent (direct) | `epy_conv` |
| `Ex` | 불연속 (`Dx`는 연속) | inverse 규칙 | `epx_conv` |
| `Dz = -Kx H̃y` 형태의 소거를 통한 `Ez` | — | `eps` 합성곱 행렬의 역 | `epz_conv_i` |

`factorization='laurent'`는 `Ex` 행에도 direct 규칙을 넣습니다 -- 두 면내 칸 모두에 `[[eps]]`,
`epz_conv_i`는 그대로 -- 이것이 이 표가 바로잡는 정식화입니다.
`docs/theory/rcwa/factorization-terms.md` §9를 보세요.

2D 패턴 층에서는 같은 논리로 다섯 칸 `(epx, epy, epz_i, epxy, epyx)`가 나옵니다. 비대각 칸
규약(칸에 `-eps_xy`)은 `docs/theory/rcwa/off-diagonal-slot-convention.md`에 설명되어 있으며 여기서
다시 유도하지 않습니다. 이 빌더들이 읽는 `eps`는 **쓰인 그대로의** 층 재료입니다 — 2026-08-20부터
입구가 상대 유전율을 받으므로 `factorization._layer_eps`, `parametric.assemble_eps_coeffs`,
`boundary.conv_mats`에 `n → eps` 변환이 더는 남아 있지 않고, **켤레도 없습니다**(셋 중 마지막
것은 2026-08-11까지 켤레를 담고 있었습니다. `docs/validation/rcwa/time-convention-audit.md` §6).
손실은 `e^{-iωt}`에서 쓰인 그대로 `Im(eps) > 0`입니다.

---

## 3. 1D 층 하나: TE와 TM

기하: `eps = eps(x)`, `y` 방향 변화 없음, `ky = 0`. 푸리에 공간에서 `∂x̃ → i Kx`(`kx_m`의 대각
행렬), `∂z → d/dz̃`입니다.

### 3.1 TE (`tetm_index = 0`)

필드: `(Ey, H̃x, H̃z)`. 주 변수 `S = Ey`(차수에 대한 벡터). (M1)로부터 다음을 얻습니다.

```
-S' = i H̃x            (x-row)          H̃z = Kx S          (z-row)
```

짝 변수를 `U ≡ +i H̃x = -S'`로 정의합니다. (M2)의 `y` 행
`H̃x' - i Kx H̃z = -i εy S`(`εy =` `epy_conv`)로부터 다음을 얻습니다.

```
S'' = (Kx² - εy) S  ≡  A S                            (TE1)
```

`A = W diag(q²) W⁻¹`로 고유분해하고, `Re(q) ≥ 0`인 `q`를 택합니다(`mode_q`; 이 이름표는 자유롭게
정할 수 있으며 — `docs/validation/rcwa/time-convention-audit.md` §5 — 이 선택은 `exp(-q d̃)`를
유계로 유지합니다). 일반해와 그 짝은 다음과 같습니다.

```
S(z̃) = W [ e^{-q z̃} c⁺ + e^{-q(d̃-z̃)} c⁻ ]
U(z̃) = V [ e^{-q z̃} c⁺ - e^{-q(d̃-z̃)} c⁻ ],        V = W q          (TE2)
```

(`U = -S'`를 항별로 적용.) **코드 확인**: `layer_modes_1d`, `tetm_index == 0`:
`A = diag(kx)² - epy_conv`, `V = W * q`. 그대로 일치합니다.

균질한 반공간에서 아래로 가는 파 `S = t exp(+i kz z̃)`는 `U = -i kz S`를 가집니다 — 이것이 기판
초기값입니다.

```
G = -i Kz_bot                                          (TE3)
```

(코드: `G = -1j * _diag(xp, kz_bot)`), 그리고 균질 매질에서의 대응은 `q = -i kz`입니다.

### 3.2 TM (`tetm_index = 1`)

필드: `(H̃y, Ex, Ez)`. 주 변수 `P = H̃y`. (M2)로부터 다음을 얻습니다.

```
P' = i εx Ex     →  Ex = -i εx⁻¹ P'          (x-row, εx = epx_conv, inverse rule)
i Kx P = -i εz Ez →  Ez = -εz⁻¹ Kx P          (z-row, εz⁻¹ = epz_conv_i)
```

두 식을 모두 (M1)의 `y` 행 `Ex' - i Kx Ez = i P`에 대입하면 다음과 같습니다.

```
P'' = εx (Kx εz⁻¹ Kx - I) P  ≡  εx B P                (TM1)
```

짝 변수는 `Q ≡ -i Ex = -εx⁻¹ P'`이므로, `P = W e^{-q z̃} c⁺`에 대해 다음을 얻습니다.

```
V = εx⁻¹ W q                                           (TM2)
```

그리고 `P = t e^{+i kz z̃}`로부터 기판 초기값은 `Q = -i (kz/εbot) P`, 즉 다음과 같습니다.

```
G = -i Kz_bot / eps_bot                                (TM3)
```

**코드 확인**: `layer_modes_1d`, `tetm_index == 1`: `B = kx εz⁻¹ kx - I`, `eig(epx_conv @ B)`,
`V = inv(epx_conv) @ W * q`; `solve_1d` 1단계: `G = -1j * diag(kz_bot / eps_bot)`.
`εx`가 어느 쪽에서 곱해지는지까지 포함해 그대로 일치합니다.

두 경우를 하나의 패턴으로 만드는 대칭에 주목하세요. 주 변수 `(S | P)`, 짝 변수
`(+i H̃x | -i Ex)`이며, 두 경우 모두 짝 변수는 `-(mass matrix)⁻¹ d(primary)/dz̃`와 같고, 질량
행렬은 TE에서 `I`, TM에서 `εx`입니다. conical 계와 2D 계는 바로 이 대칭을 블록으로 묶습니다.

---

## 4. 2D 층 하나: 1차 연립계

기하: `eps = eps(x, y)`, 두 횡방향 필드 성분이 결합합니다. `S = [Ex; Ey]`와 `U = +i [H̃x; H̃y]`로
모읍니다(각 블록은 `n_orders_v·n_orders_u` 차수 격자에 대한 벡터). (M1)/(M2)의 `z` 행을 써서 종방향
성분을 소거합니다.

```
H̃z = Kx Ey - Ky Ex
Ez  = -εz⁻¹ (Kx H̃y - Ky H̃x) = i εz⁻¹ (Kx Uy - Ky Ux)        (using H̃_t = -i U)
```

남은 네 행에 대입하고 `B = Kx εz⁻¹ Kx - I`,
`D = Ky εz⁻¹ Ky - I`, `C = Kx εz⁻¹ Ky`, `E = Ky εz⁻¹ Kx`(코드의 `B, D, Cc, E`)로 쓰면 다음과
같습니다.

```
dS/dz̃ = -Ω_L U          Ω_L = [[ -C,      B     ],
                                [ -D,      E     ]]           (2D1)

dU/dz̃ = -Ω_R S          Ω_R = [[ -Kx Ky + εyx̂,   Kx² - εy ],
                                [ εx - Ky²,        Ky Kx - εxŷ ]]   (2D2)
```

(`εyx̂`/`εxŷ`는 비대각 칸이며, 등방성 층에서는 0입니다.) `U`를 소거하면 다음과 같습니다.

```
d²S/dz̃² = Ω_L Ω_R S ≡ Ω²_LR S                          (2D3)
```

풀어 쓰면 `Ω²_LR = [[Ky² + B εx - C εyx̂, C εy - Kx Ky - B εxŷ],
[E εx - Ky Kx - D εyx̂, Kx² + D εy - E εxŷ]]`이며 — `layer_modes_2d`에 있는 코드의
`Omega2_LR`와 항 하나하나 일치합니다. 모드 행렬은 `Ω²_LR = W diag(q²) W⁻¹`이고, 짝 변수는
`S = W e^{-q z̃}`에 (2D2)를 적용해 얻습니다.

```
V = Ω_R W / q                                           (2D4)
```

**코드 확인**: `layer_modes_2d`는 정확히 (2D3)과 (2D4)를 계산하며, `Ω_R` 네 블록의 부호 패턴이
줄 단위로 일치합니다. 등방성 극한에서의 점검: 스칼라 `eps`, 평면파 — `Ω²_LR`는
`(kx² + ky² - eps) I₂`로 줄어들므로 `q² = k_par² - eps`, 즉 `q = -i kz`입니다. `Ω²_LR`와 `Ω_R`
사이의 비대각 순서(`εyx̂`를 어느 열과 짝짓는가)는 자유롭게 정할 수 **없습니다** —
`docs/theory/rcwa/off-diagonal-slot-convention.md` §4를 보세요.

`H̃_t`가 아니라 `U = +i H̃_t`를 쓰는 이유: 이렇게 하면 모든 전파 인자가 실수 양수
(`exp(-q z̃)`)가 되고 모든 어드미턴스가 `-i (…)` 꼴로 나오므로, `i`가 재귀 전체에 흩어지는 대신
정확히 두 곳 — 초기값 (TE3)/(TM3)과 최종 정합(§7) — 에만 남습니다. 이것은 Moharam의 `-j` 정규화에
대응하는 `e^{-iωt}` 판입니다.

---

## 5. conical 1D 층

기하: 다시 `eps = eps(x)`이지만 `ky = ky0 ≠ 0`입니다(차수 하나; `Ky = ky0 I`는 모든 것과
교환됩니다). 그러면 2D 계 (2D1)–(2D2)는 두 개의 스칼라 헬름홀츠 계열로 분리됩니다.

```
family 1 (E-like):   d²S1/dz̃² = (Ky² + A) S1,      A = Kx² - εy       (C1)
family 2 (H-like):   d²S2/dz̃² = (Ky² + B εx) S2,   B = Kx εz⁻¹ Kx - I  (C2)
```

— `ky0 = 0`에서 이들은 (TE1)과 (TM1)입니다. 짝 변수는 분리되지 않습니다. 각 계열의 `U`는 다른
계열의 `Ky`에 비례하는 성분을 섞어 받으며, 네 결합 블록은 여기서 나옵니다. `Ky = ky0 I`로 (2D2)를
끝까지 전개하면 다음과 같습니다.

```
V_11 = A⁻¹ W1 q1              V_12 = Ky A⁻¹ Kx W2
V_21 = Ky B⁻¹ Kx εz⁻¹ W1      V_22 = B⁻¹ W2 q2                    (C3)
```

**코드 확인**: `layer_modes_1d_conical` — `w1, W1 = eig(Ky² + A)`,
`w2, W2 = eig(Ky² + B @ epx_conv)`, 그리고 네 `V` 블록이 그대로 일치합니다. ((C2)는 `B εx`로,
1D TM의 `εx B`와 곱 순서가 바뀌어 있다는 점에 유의하세요. 두 행렬은 닮은 행렬이어서 `q`가 같고,
고유벡터 규약의 차이는 `V` 블록이 흡수합니다. 레거시 경로와의 비트 단위 일치와 RETICOLO conical
스윕이 이를 함께 고정합니다.)

경계는 s/p 기저에서 작동하며, 차수마다 `(kx, ky)`의 방향인 `varphi`만큼 회전합니다 --
`atan2(ky, kx)`; `order_azimuth`는 2026-08-16까지 이를 `atan(ky/kx)`로 썼습니다
(`docs/theory/rcwa/amplitude-sp-frame.md`) -- 이때 `F_c = diag(cos varphi)`,
`F_s = diag(sin varphi)`입니다. `sp_modes_1d_conical` / `sp_modes_2d`는 (C3)/(2D4)로부터
`big_W`(모드 → 경계 주 변수)와 `big_V`(모드 → 경계 짝 변수)를 조립합니다. 이 회전은 순수한
기하이며, §2.1의 `ky` 차수 벡터가 정하는 `varphi`의 부호 외에는 아무런 규약 내용도 담지 않습니다.

---

## 6. 층 연결: 전달 재귀

계면마다의 상태: 아래쪽 필드가 (아직 모르는) 투과 진폭 `t`에 대해 `S = F t`,
`U = G t`가 되도록 하는 쌍 `(F, G)`입니다. 초기값: 기판,
`F = I`, `G`는 (TE3)/(TM3)에서 (또는 그 블록 형태 `big_F`, `big_G`).

두께 `d̃`인 층 하나를 위에 더합니다. 층의 해 (TE2)를 써서 층의 **아래** 계면에서 `(S, U)`의
연속성을 적으면 다음과 같습니다.

```
W (X c⁺ + c⁻) = F t          V (X c⁺ - c⁻) = G t,        X = e^{-q d̃}
```

다음을 정의합니다.

```
a = ½ (W⁻¹ F + V⁻¹ G)         b = ½ (W⁻¹ F - V⁻¹ G)
```

그러면 `X c⁺ = a t`, `c⁻ = b t`입니다. `t = a⁻¹ X t'`로 다시 매개변수화합니다(모든 지수 인자를
감쇠하는 쪽으로 유지하는 단계입니다 — `X⁻¹`은 결코 나타나지 않습니다).

```
c⁺ = t'          c⁻ = b a⁻¹ X t'
```

그리고 층의 **위** 계면에서의 필드가 새 상태가 됩니다.

```
F ← W (I + Y)        G ← V (I - Y)        Y = X b a⁻¹ X        T ← T a⁻¹ X    (J1)
```

`T`는 재매개변수화를 누적하여 마지막에 기판 진폭이 `T t_top`이 되게 합니다. **코드 확인**:
`solve_1d` 2단계는 기호 하나하나까지 (J1)이며(`a_i = a⁻¹`), conical/2D 루프는 그 블록 형태입니다.
(J1)에는 규약 부호가 전혀 없습니다 — 모든 `i`가 `a`와 `b`에서 상쇄되었습니다 — 그래서 재귀는
규약 뒤집기를 손대지 않은 채 견뎌냈습니다. 초기값과 최종 정합은 그렇지 못했고, 과거의 네 버그가
모두 바로 거기에 있었습니다.

---

## 7. 연립계 닫기: 입사파

superstrate, `z̃ < 0`: 입사파(아래쪽)와 반사파(위쪽)의 합입니다.

```
S(z̃) = δ e^{+i kz_inc z̃} + R e^{-i Kz_top z̃}
U(z̃) = -i kz_inc δ e^{+i kz_inc z̃} + i Kz_top R e^{-i Kz_top z̃}
```

`δ`는 주 변수의 직교 좌표 성분(TE에서는 `E_y`, TM에서는 `H_y`)으로 쓴 입사 열입니다. 입사 차수에서는
`sigma_inc`, `excitation` 아래에서는 빛을 받는 각 차수에서 `sigma_j c_j w_j`입니다. 여기서
`sigma = sign(Re kx)`는 `ky == 0` 행에서의 `s_hat . y_hat`(`order_frame_sign`)이고,
`c_j = (-1)**(m + n)`은 셀 중심을 기준으로 한 공개 가중치를, 커널 연산자가 만들어지는 푸리에 원점인
셀 모서리를 기준으로 다시 표현합니다(공개된 합성곱 행렬은 그 연산자들을 중심 기준으로 `D C D`로
다시 표현합니다)(`corner_column`; 공통 인자 `exp(-1j k_inc . (a1 + a2)/2)`는 필드에 맡깁니다).
보고되는 진폭은 `C D R`과 `C D T T1`이며 `D = diag(sigma)`, `C = diag(c)`입니다(TM 투과는 임피던스
비를 적용한 뒤). 이로써 모든 차수가 결합 경로들과 같은 자기 자신의 s/p 좌표계에
놓이고(`docs/theory/rcwa/amplitude-sp-frame.md` §6), 중심을 기준으로 합니다(같은 문서 §7). 엔진은
입출구에서 `C`를 적용하므로 커널 자체는 `D`만 봅니다. 재구성에 넘기는 TM 필드 상태는
`T1 · n_inc/mu_inc`로, 단위 입사 E입니다. `kz_inc`는 `Kz_top`을 만드는 **바로 그 벡터에서 읽어야
합니다**(`incident_kz`). `n_top cosθ`라는 표기는 실수 `θ`에서는 항등식이지만 복소 `θ`에서는 잘못된
가지입니다(`docs/theory/rcwa/branch-cuts.md`). `(F T1, G T1)`에 대해 `z̃ = 0`에서의 연속성은
다음과 같습니다.

```
δ + R = F T1
-i kz_inc δ + i Kz_top R = G T1
```

`R`을 소거하면 다음과 같습니다.

```
(G - i Kz_top F) T1 = -i Kz_top δ - i kz_inc δ                     (TE match)
(G - i (Kz_top/n_top²) F) T1 = -i (Kz_top/n_top²) δ - i (kz_inc/n_top²) δ   (TM match)
```

**코드 확인**: `solve_1d` 3단계, 두 분기 모두 그대로 일치합니다(`inc = -1j*kz_inc*δ`).
그다음 `R = F T1 - δ`이고 기판 진폭은 `T T1`입니다.

conical/2D 최종 정합(`final_matching`)은 s/p 블록 기저에서의 같은 소거이며, 하나의
`4·n_orders` 연립계(`final_A`, `final_B`)로 풉니다. 여기서 쓰는 경계 변수는 **레거시 경로에서
이어받은 정의**입니다 — s는 전기 진폭(`E_s`)을, p는 자기 진폭을 담습니다(단위 E의 입사 p파는
`H̃ = n_top`을 가집니다). 그래서 `final_B`의 행이
`(-sinψ, cosψ kz_inc/n_top, i sinψ kz_inc, i n_top cosψ)`로 읽힙니다. 자기 p 변수는
`/(-i n_top)`을 거쳐 프레넬 `r_p`가 됩니다. 2026-08-12부터 이 변환은 결과를 묶는 지점에 있으므로,
공개 `r_p`/`t_p`는 프레넬 정규화되어 있고 진폭 기준 비교가 이들을 직접 비교합니다. 이 사실들은
경로/알고리즘 조합 여섯 가지에 걸쳐 `test_reflection_amplitude`와 `test_transmission_amplitude`로
고정되어 있으며, 이것이 s/p 부호 규약이 *유도*되는 곳이 아니라 *확인*되는 곳에 대한 정직한
진술입니다.

---

## 8. 효율

한 차수의 시간 평균 `z` 플럭스는 `S_z = ½ Re(E × H̃*)_z`로부터 다음과 같습니다.

- 전기 진폭이 `R_m`인 TE 차수: `S_z ∝ |R_m|² Re(kz_m)`
- 자기 진폭이 `R_m`인 TM 차수: `S_z ∝ |R_m|² Re(kz_m / eps)`

입사 플럭스(단위 입사 E, 실수 `θ`에서 `n_top cosθ`)로 정규화하면 다음과 같습니다.

```
de_ri[m] = |R_m|² · Re(kz_top,m) / Re(n_top cosθ)
de_ti[m] = |T_m|² · Re(kz_bot,m) / Re(n_top cosθ)              (TE)
de_ti[m] = |T_m|² · Re(kz_bot,m / n_bot²) / Re(cosθ / n_top)   (TM)
```

**코드 확인**: `solve_1d` 3단계와 `sp_efficiencies`, 그대로 일치합니다. 적어 둘 만한 규약 사실이
두 가지 있습니다. `Re(conj(kz)/eps)`가 **아니라** `Re(kz/eps)`입니다 — 켤레는 옛 정식화의
잔재이며 음의 투과 전력을 골든 값에 넣었습니다(`test_lossy_media` 모듈 docstring). 그리고 복소
`n_top`에서는 입사 전력 정규화가 규약과 무관하지 않으므로, 그 경우 `de_ri`는 어떤 값과도 비교하지
않습니다(복소 진폭은 비교합니다. `tests/rcwa/test_lossy_media.py`, 진폭 기준 비교).

---

## 9. 제곱근

위의 모든 `sqrt`는 물리적인 결정이며, `src/wavebit/_common/conventions.py`에 모여 있습니다.

- **층** `q`: SMM에서는 `Re(q) ≥ 0`(`mode_q`)으로, 자유로운 이름표에 대한 조건수 관점의 선택입니다.
  ETM에서는 층을 가로지르며 `eps**-0.25`보다 덜 커지는 곳마다 반공간의 근을 쓰며, 층을 조각으로
  자를 때는 같은 층들이 이어진 구간 전체를 가로지르며(`sqrt(eps)`와 같음) 판단합니다
  (`etm_mode_q`, `run_k0_d`). 그 재귀는 투과를 역으로 풀기 때문에 그 이름표가 자유롭지 않기
  때문입니다(`docs/theory/rcwa/connecting-algorithms.md` §10). 1차 분리는 층 연산자의 불변 블록
  하나 안에서 모든 순방향/역방향 쌍의 왕복을 그 상한의 제곱인 `eps**-0.5`로 제한하지만
  (`solve_2d_first_order._cap_round_trips`), 블록 사이에서는 그렇게 할 수 없습니다
  (`docs/theory/rcwa/connecting-algorithms.md` §10). 균일한 이중 음성 층은 두 규칙 중 하나의 거울상을
  택하고, 이중 음성을 담은 패턴 층은 그 모드들의 전력 플럭스를 따릅니다
  (`layer_label_rule`, `docs/theory/rcwa/connecting-algorithms.md` §11).
- **반공간** `kz`: Maystre의 절단 `Re(kz) + Im(kz) > 0`(`boundary_kz`, 2026-10-02부터).
  수동 매질의 모든 실수 `k_par`에서 이것은 주근(`Re > 0`, 같으면 `Im > 0`)입니다.
  복소 각도나 이득 매질로 도달하는 `kz**2`의 실수축 아래에서는 극한 흡수가 도달하는 근입니다.
  이중 음성(`Re(eps) < 0`이고 `Re(mu) < 0`)은 그 절단의 거울상 `Im(kz) - Re(kz) > 0`, 즉 그 자신의
  극한 흡수를 택하며(`docs/theory/rcwa/negative-index-media.md` §5), 그 굴절률 `n` -- 이 유도의 모든
  `n` -- 은 `sqrt(eps)*sqrt(mu)`이고, 거기서는 음수입니다(§6).

유도와 측정은 `docs/validation/rcwa/time-convention-audit.md`와
`docs/theory/rcwa/branch-cuts.md`에 있으며, 여기서 반복하지 않습니다.

---

## 10. `e^{+jωt}` 논문 초고와의 대응

이 저장소 밖에 보관된 별도의 논문 초고는 공학 규약으로 쓰여 있습니다. 두 정식화는 정확히 서로의
복소 켤레(`j ↔ -i`)이며, 모든 물리적 크기는 같습니다. 교차해 읽을 때 쓰세요.

| | 논문 (`e^{+jωt}`) | 이 문서 / 코드 (`e^{-iωt}`) |
| --- | --- | --- |
| 맥스웰 | `∇×E = -jωμ0H` | `∇×E = +ik0H̃` |
| +z 방향 파 | `exp(-j kz z̃)` | `exp(+i kz z̃)` |
| 손실 | `n - jκ`, `Im(eps) < 0` | `n + iκ`, `Im(eps) > 0` |
| +z 방향 감쇠 | `Im(kz) < 0` | `Im(kz) > 0` |
| 짝 변수 정규화 | `U = -j Z0 H_t` | `U = +i Z0 H_t` |
| 층 근 | `Re(q) ≥ 0` | `Re(q) ≥ 0` — **동일**, 켤레에 불변 |
| 복소 `r`, `t`, 필드 | 코드 값의 켤레 | |
| `\|r\|²`, 효율, `\|E\|` | 동일 | |

실무상 결과: Wavebit으로 계산한 효율과 세기 수치는 그대로 논문에 넣을 수 있습니다. 반면 **복소
진폭이나 필드 위상**은 모두 켤레를 취해야 합니다(또는 논문의 규약 진술을 바꿔야 합니다). 그리고
현재 논문은 자신의 규약을 어디에도 밝히지 않습니다 — 이론 장 첫 문단에 "we adopt `e^{+jωt}`; a
lossy medium has `Im(ε) < 0`"이라는 한 문장을 넣는 것이 이 코드베이스의 역사가 내놓을 수 있는 가장
값싼 버그 퇴치제입니다.

이 유도의 LaTeX 판(논문 문체, 코드 참조 없음)은 이 저장소 밖에 보관되어 있으며, 논문 쪽을 코드에
맞춰 뒤집는 경우를 대비해 논문에 `\input`할 수 있도록 준비되어 있습니다.
