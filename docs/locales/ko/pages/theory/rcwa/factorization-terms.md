# Laurent, direct, inverse, Toeplitz: 어느 단어가 규칙이고 어느 단어가 행렬인가

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 있어야 실행됩니다.
> 이 네임스페이스는 현재 설치본에는 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행하지 않았습니다.


작성: 2026-08-12. 아래의 구조에 관한 주장은 단언이 아니라 측정한 것입니다. 검사 방법은 §3에 인용했으며 몇
초 안에 재현됩니다. §6은 2026-09-25에 다시 썼습니다. 이때 래스터 Li 생성기를 Li의 논문으로부터 다시 만들면서
두 번째 중첩 방식이 추가되었습니다. §9는 2026-09-27에 추가했습니다. 이때 Laurent 규칙이 `factorization=`의
값 하나가 되었습니다.

대상:
- `src/wavebit/rcwa/_engine/parametric.py::conv_matrix_2d` — 이름 충돌이 있던 생성기로, 이 문서를 쓰기
  전까지는 `laurent_conv_2d`라고 불렸습니다
- `src/wavebit/rcwa/_engine/parametric.py::_grid`, `src/wavebit/rcwa/_engine/parametric.py::assemble_eps_coeffs` —
  생성기가 소비하는 두 배 크기의 역격자 격자, 그리고 그것이 두 배인 이유
- `src/wavebit/rcwa/_engine/factorization.py::fourier2d_piecewise_constant` — 래스터 쪽의 쌍둥이, 그리고
  `inverse_rule`이 규칙을 고르는 곳
- `tests/rcwa/test_parametric.py` — 두 생성기가 공유하는 순서를 고정하는
  `test_conv_matrix_matches_grid_convention`
- `src/wavebit/rcwa/_engine/factorization.py::_li_slot`과 `src/wavebit/rcwa/_engine/enums.py::Nesting` — Li
  규칙의 두 가지 중첩 방식, 그리고 둘 중 하나를 고르는 키워드(§6)
- `src/wavebit/rcwa/_engine/factorization.py::laurent_conv_matrices`와
  `src/wavebit/rcwa/_engine/parametric.py::laurent_from_coeffs` — 래스터와 형상에서 인수분해 선택지로서의
  Laurent 규칙(§9)

---

## 요약

쓰이는 단어는 네 개이며, 서로 다른 두 축에 속합니다. **Laurent**와 **direct**는 같은 *인수분해 규칙*을
가리키고, **inverse**(Li)는 다른 하나를 가리킵니다. **Toeplitz**는 *행렬의 모양*을 가리키며 어떤 규칙도
가리키지 않습니다.

예전에는 함수 하나가 두 축을 섞었습니다. `conv_matrix_2d`는 원래 `laurent_conv_2d`였는데, 두 규칙 모두에
쓰이는 생성기에 규칙 이름을 붙인 것이었습니다. §5에 이름 변경과 남겨 둔 것을 기록했습니다.

이 부분을 고치기 전에 알아 둘 것이 두 가지 있습니다.

1. `conv_matrix_2d`는 **규칙 중립적**입니다. 두 줄 간격으로 `eps`와 `1/eps`에 대해 호출되고, 세 개의
   법선장 곱에 대해서도 호출됩니다. 이 함수가 실행될 때는 아직 어떤 규칙도 정해지지 않았습니다.
2. 이 함수가 반환하는 행렬은 **Toeplitz가 아닙니다**. BTTB — Toeplitz 블록으로 된 블록 Toeplitz — 이며, 이
   구별은 현학이 아니라 검사할 수 있는 것입니다(§3).

그리고 규칙 자체에 관해 하나 더. 2D에서 Li 규칙은 한 축을 다른 축 안에 넣어 적용하며, 중첩하는 방법은 두
가지입니다. x와 y를 똑같이 다루는 것은 Li 자신의 방식뿐입니다. 이 경로가 줄곧 만들어 왔고 지금도 기본값으로
만드는 방식은 그렇지 않습니다(§6).

2026-09-27부터 규칙 이름은 *값*이기도 합니다. `factorization='laurent'`는 모든 곱을 direct 규칙으로 닫으며,
그 규칙을 쓰는 솔버와 비교하기 위해 명시적으로 선택해야 하는 옵션입니다. 무엇을 만드는지, 어디서 거부되는지,
정확도 면에서 어떤 대가가 있는지는 §9에 있습니다.

## 1. 두 규칙

실공간의 곱셈은 푸리에 공간의 컨볼루션이므로, `D = eps E`는 `E`의 계수 벡터에 작용하는 행렬이 됩니다. 어떤
행렬인지는 두 인자가 같은 곳에서 불연속인지에 달려 있습니다.

| 규칙 | 다른 이름 | 쓰는 행렬 | 올바른 규칙이 되는 경우 |
| --- | --- | --- | --- |
| direct | Laurent, "plain Laurent" | `<eps>` | 두 인자가 동시에 불연속이 아닐 때 |
| inverse | Li 규칙 | `<1/eps>^-1` | 동시에 불연속일 때 — 그 방향을 따라서 |

첫 번째 규칙의 두 이름은 이 저장소에서 모두 쓰이고 있으며 둘 다 맞습니다. `docs/theory/rcwa/formulation.md`는
`Laurent (direct)`라고 쓰고, `docs/guide/rcwa/parametric-modeling.md`는 `Laurent (direct-rule)`이라고 쓰며,
`src/wavebit/rcwa/_engine/factorization.py`는 코드에서 이 쌍을 `direct` / `inverse`로 씁니다(`ep_dir`,
`ep_inv`). 하나를 쓰고 다른 하나를 괄호 안에 넣는 병기 방식이 새 글에서 따라 할 형태입니다. 한 이름만 아는
독자도 뜻을 알아볼 수 있기 때문입니다.

`fourier2d_piecewise_constant`의 `inverse_rule`이 스위치입니다. `'u'` 또는 `'v'`는 규칙이 불연속을 닫는
방향을 가리키고, `None`은 양쪽 모두 direct 규칙을 요청합니다. 이것은 실험실 좌표가 아니라 *셀*의 두
방향입니다 — 둘은 직사각형 셀에서만 일치하므로, `'x'`와 `'y'`는 다르게 해석하지 않고 이름으로 거부합니다.
다른 어떤 것도 이 규칙을 고르지 않으며, `fourier2d_pointwise`는 이 선택지를 아예 제공하지 않습니다 — 구성상
연속인 필드에는 inverse 규칙이 닫을 불연속이 없습니다.

## 2. `conv_matrix_2d`가 실제로 무엇인가

이 함수는 푸리에 계수의 격자를 받아 컨볼루션 행렬로 배치합니다.

```
M[(r,s),(c,d)] = coeff(ky = r - c, kx = s - d)
```

하는 일은 그것이 전부입니다. 유전율을 보는 일도 없고, 호출자가 어떤 규칙을 만들고 있는지도 모릅니다.
`to_conv_mat_parametric_nvf`의 마지막 단계인 `nvf_from_coeffs`에서는 연달아 다섯 번 호출됩니다 — `eps`,
`1/eps`, 그리고 세 개의 법선장 곱 `nx^2`, `nx ny`, `ny^2`에 대해:

```python
ep_dir = conv_matrix_2d(xp, eps_c,     half_width_u, half_width_v)
ep_inv = conv_matrix_2d(xp, inv_eps_c, half_width_u, half_width_v)
...
delta_ep = ep_dir - xp.linalg.inv(ep_inv)
```

inverse 규칙의 *역행렬 계산*은 호출자의 몫으로, 바로 다음의 `_nvf_assemble`에서 이루어집니다. 생성기는
공유됩니다.

예전 이름이 틀린 점이 이것입니다. 아직 어떤 규칙도 정해지지 않은 곳에 규칙 이름을 가지고 들어왔으므로,
`ep_inv = laurent_conv_2d(...)`는 실제로는 모순이 아닌데도 모순처럼 읽혔습니다.

인덱스 차 `r - c`와 `s - d`는 각각 `-2*truncation_order .. +2*truncation_order` 범위에 걸치며, 그래서 계수
격자는 **두 배** 크기의 역격자 범위로 들어옵니다 — `_grid`를 보세요. 축마다 `2*truncation_order+1`개뿐인
격자로는 행렬의 모서리를 채울 수 없습니다.

## 3. 이 행렬은 Toeplitz가 아니라 BTTB입니다

Toeplitz 행렬은 대각선을 따라 일정합니다. `T[i,j]`는 `i - j`에만 의존합니다. 여기의 행렬은 이를 만족하지
않습니다. `i = r*n_orders_u + s`, `j = c*n_orders_u + d`로 평탄화하면,

```
i - j = (r - c) * n_orders_u + (s - d)
```

이고 `s - d`는 `-(n_orders_u-1) .. (n_orders_u-1)` 범위에 걸치는데, 이는 `n_orders_u`보다 넓습니다. 따라서
`(r-c, s-d)`를 `i - j`로부터 복원할 수 없습니다. 쌍 `(k, m)`과 `(k+1, m - n_orders_u)`는 서로 다른 값을
가지면서 같은 대각선에 놓입니다.

`half_width_u, half_width_v = 2, 3`(35x35 행렬, 무작위 복소 계수)에서 직접 확인한 결과:

| 성질 | 성립 여부 |
| --- | --- |
| 평탄화한 인덱스에서 Toeplitz — `M[i,j]`가 `i-j`에만 의존 | **아니요** |
| 블록 Toeplitz — 블록 `B[r,c]`가 `r-c`에만 의존 | 예 |
| 모든 블록이 그 자체로 Toeplitz | 예 |

마지막 두 성질을 함께 갖는 것이 **BTTB**, 즉 Toeplitz 블록으로 된 블록 Toeplitz의 정의입니다.
`block-Toeplitz`만 쓰는 것이 흔한 약칭이며 `src/wavebit/lattice.py`, `docs/theory/rcwa/formulation.md`,
`docs/guide/rcwa/parametric-modeling.md`, `docs/theory/rcwa/oblique-lattice.md`에 나옵니다. 블록 Toeplitz
행렬은 임의의 블록을 가질 수 있으므로 이 표현은 틀렸다기보다 덜 구체적입니다. 글에서는 해가 없지만, 함수
이름의 근거로 삼기에는 너무 약합니다.

## 4. 어디에 어떤 단어를 쓸까

| 다루는 대상 | 쓸 말 | 피할 말 |
| --- | --- | --- |
| 인수분해 선택 | `direct` / `inverse`, 또는 `Laurent (direct)` | 행렬을 뜻할 수도 있는 곳에서 `Laurent`만 단독으로 |
| 생성기가 반환하는 행렬 | `convolution matrix` | `Toeplitz matrix` — 그것이 아님 |
| 그 구조, 정확하게 | `BTTB` | — |
| 그 구조, 지나가며 | `block-Toeplitz` | — |
| 생성기 자체 | 나오는 모양이 아니라 하는 일로 이름 붙이기 | 규칙 이름 |

## 5. 이름 변경, 그리고 바뀌지 않은 것

`laurent_conv_2d`는 `conv_matrix_2d`가 되었습니다. 구조를 따라 이름 짓는 것도 검토했지만 기각했습니다.
`toeplitz_`는 거짓이고, `block_toeplitz_`는 덜 구체적이며, `bttb_`는 읽기 어렵습니다. 남은 것은 만들어 내는
것 — 컨볼루션 행렬 — 을 따라 이름 짓는 것이었고, 이는 코드의 나머지 부분이 이미 쓰는
어휘입니다(`nvf_conv_matrices`, `to_conv_mat_parametric_nvf`, `epx_conv`).

첫 릴리스 전, `_core`가 비공개이고 그 바깥에서 이 이름을 가져다 쓰는 곳이 없을 때 바꾸었으므로, 별칭을
남기지 않았고 남길 의무도 없습니다.

| 바뀐 것 | 개수 | 위치 |
| --- | --- | --- |
| `laurent_conv_2d` → `conv_matrix_2d` | 14 | `src/wavebit/rcwa/_engine/parametric.py`, `tests/rcwa/test_parametric.py`, `parametric-modeling.md` |
| `test_laurent_matches_dfs2d_convention` → `test_conv_matrix_matches_grid_convention` | 4 | `tests/rcwa/test_parametric.py`, `src/wavebit/rcwa/_engine/parametric.py`, `parametric-modeling.md` |
| 구조를 뜻하던 "Laurent matrix" → "convolution matrix" | 2 | `src/wavebit/rcwa/_engine/parametric.py` |

**규칙을 뜻하는 `Laurent`는 15곳 모두에 그대로 남았습니다.** 그곳에서는 맞는 말이며, 그중 하나는 아예 글이
아닙니다. `studies/rcwa/benchmark_truncation_shape/benchmark_truncation_shape.py`의
`modes=('nvf', 'laurent')`는 사용자가 입력하는 값입니다. 저장소에서 이 단어를 쓸어 낸 이름 변경이었다면 이것을
망가뜨리고 아무것도 고치지 못했을 것입니다.

## 6. 중첩된 규칙, 그리고 중첩하는 두 가지 방법

2026-09-25에 다시 썼습니다. 그전까지 이 절은 `li` 생성기의 x/y 비대칭이 "Li의 교차 격자 방식에
본질적"이라고 말했습니다. 그렇지 않습니다. 이 비대칭은 **두 필드 성분 모두에 하나의 중첩 순서를 쓰는 것**에서
오며, 성분마다 순서를 고르는 Li 자신의 구성에는 이 비대칭이 없습니다. 이제 둘 다 만들며, `li_nesting=`이
고릅니다(`src/wavebit/rcwa/_engine/enums.py::Nesting`).

2D에서 규칙을 적용한다는 것은 규칙을 중첩한다는 뜻입니다(L. Li, JOSA A 14, 2758, 1997). 한 축은 안쪽에서
셀 한 줄씩 변환되고, 그 결과는 다른 축의 행렬 값 함수가 되며, 그 다른 축은 바깥쪽에서 변환됩니다. 각 축을
따른 Toeplitz 행렬을 `T_u` / `T_v`라 하면:

| 칸 | `li_nesting='fixed'` (기본값) | `li_nesting='symmetric'` |
| --- | --- | --- |
| `epx` (`E_x`에 작용) | `( T_u( [T_v(eps)]^-1 ) )^-1` | `T_v( [T_u(1/eps)]^-1 )` |
| `epy` (`E_y`에 작용) | `T_u( [T_v(1/eps)]^-1 )` | 같음 |
| `epz_i` | `( T_uv(eps) )^-1` | 같음 |

`'fixed'`는 두 칸 모두 `v`를 안쪽에 둡니다. 래스터 경로가 이 날짜까지 만들던 방식이며, 데카르트 형상 Li와
텐서 Li 빌드가 쓰인 순서이고(텐서 Li는 이것 말고는 만들지 않습니다), RETICOLO의 `parm.res1.li=1`입니다.
`'symmetric'`은 각 칸 자신의 inverse 규칙 축을 안쪽에 둡니다 — Li의 구성, RETICOLO의 `li=0` — 그래서
`epx`는 `epy`의 x<->y 거울상입니다. 계단 근사 위에서는 둘 다 Li 규칙이며 주어진 구조에 대해 정확하고, 같은
답으로 수렴합니다. 다른 것은 절단하는 방식입니다. y 방향으로 불변인 층에서는 둘이 일치하며, 둘 다 `epx`가
Li의 1D inverse 규칙으로 줄어듭니다(`tests/rcwa/test_li_nesting.py::test_a_y_invariant_layer_reduces_to_the_1d_rules_in_both_nestings`).

어느 중첩에서든 Li 규칙에 *정말로* 본질적인 것은, 경계가 격자 축을 따라 놓인 곳에서만 정확하다는
점입니다. 곡선이거나 기울어진 경계는 이 규칙에게는 계단입니다 — Liu & Fan은 이를 S4가 벡터장 정식화를
제공하는 이유로 듭니다 — 그리고 NVF 경로가 존재하는 이유가 이것입니다. 이름 재지정에 대한 비대칭은 별개의
결함이며, fixed 중첩에 속합니다.

**대칭.** 정사각형 셀에서 `eps(x, y) = eps(y, x)`인 구조를 수직 입사로 비춘다고 합시다. 입사 편광을 90도
돌리면 문제가 자기 자신으로 사상되므로, 회절 지도는 전치되어 돌아와야 합니다.
`tests/rcwa/test_factorization_core.py`의 L자 형상(전치 대칭이며, 일부러 4중 회전 대칭이 아니게 만든 것)에서
96x96으로 측정한 `max|T(psi=0) - T(psi=90)ᵀ|`:

| truncation_order | `li`, `'fixed'` | `li`, `'symmetric'` | `nvf` |
| --- | --- | --- | --- |
| 4 | 4.2e-3 | 2.8e-14 | 4.4e-14 |
| 8 | 2.9e-3 | 1.2e-13 | 4.0e-13 |
| 12 | 1.5e-3 | 9.7e-13 | 1.5e-13 |

이 절이 예전에 담고 있던 표(`li`에 대해 5.3e-3, 1.4e-3, 5.9e-4)는 정확한 표본 계수와 반 픽셀 위상
이전의 것이며, 대체되었습니다. NVF 열은 대조군입니다. 양쪽 모두 direct 규칙을 쓰므로 중첩을 아예 고르지
않습니다.

움직이는 부분을 모두 살려도 같은 결과가 성립합니다. 그 파일의 키랄 L을 비스듬한 입사로, 서로 다른 주기와
서로 다른 절단으로 두고 4분의 1 바퀴 돌렸습니다(두 가지 표기 모두 — `psi -> -psi`를 곁들인 전치, 그리고
단순한 `np.rot90`). symmetric 중첩은 `truncation_order` (5, 3), (8, 6), (11, 9)에서 이를 1.2e-15에서
1.9e-14 이내로 지키고, fixed 중첩은 1.1e-2, 1.4e-3, 4.5e-4만큼 벗어납니다.

**RETICOLO와의 비교.** RETICOLO는 두 중첩을 모두 구현합니다. 장면은 532 nm에서 `eps = 1.46**2` 위에 놓인,
325 nm 정사각형 셀 안의 200 x 150 nm, 두께 600 nm인 `eps = 6` 직사각형입니다. 각 중첩은 같은 중첩으로
실행한 RETICOLO V10과 전체 투과에서 4.2e-11까지 일치합니다 — 두 편광, `truncation_order` 3과 5, 두 계수
경로 모두에서 — 반면 두 중첩끼리는 1.7e-5(TE)에서 1.7e-3(TM)만큼 어긋납니다. 직사각형을 4분의 1 바퀴
돌리면 symmetric 중첩의 TE는 반올림 오차 수준에서 예전 TM과 같습니다. fixed 중첩은 절단 3에서 1.4e-3,
5에서 7.1e-4만큼 벗어납니다(`tests/references/test_reticolo_li_nesting.py`).

**가장 크게 드러나는 곳: 금속.** 이전 세션(2026-09)에서 측정했으며, 여기의 테스트로 고정되어 있지
않습니다. 다시 실행할 수 있도록 장면을 모두 적어 둡니다.

- *Au 나노홀 배열.* 정사각 주기 500 nm, 두께 150 nm의 Au(700 nm에서 Johnson–Christy,
  `eps ≈ -16.49 + 1.06i`)에 뚫린 200 nm 공기 구멍, 유리 기판 `eps = 2.25`, 수직 입사, 래스터
  N = 512. 구조가 C4 대칭이므로 두 편광의 투과는 같아야 합니다. fixed 중첩은 `truncation_order` 5에서
  T(x-pol) = 0.0172, T(y-pol) = 0.0728을 주고, 15에서도 여전히 0.0239 대 0.0427입니다. symmetric 중첩은 두
  편광에 하나의 값을 주며, 그 값은 절단에 따라 진동합니다. 13에서 0.0428, 15에서 0.0488입니다. 비교
  기준으로, fmmax의 Jones-direct 정식화는 0.0432(항 1605개)를 주고, 매개변수 원 위의 wavebit NVF는
  15에서 0.0420을 줍니다.

  2026-09-25에 다시 쓴 생성기로 다시 실행했습니다. 구멍을 중심에 두고, 픽셀 중심이 구멍 안에 있으면 그
  픽셀을 공기로 셌습니다. fixed 중첩은 네 수치를 모두 인용한 자릿수까지 재현합니다. 5에서 0.0172와
  0.0728, 15에서 0.023885와 0.042708입니다. symmetric 중첩은 5에서 두 편광 모두 0.0725, 13에서
  0.042744, 15에서 0.048913을 줍니다. 이는 이전 세션의 수치와 5.6e-5, 1.1e-4 차이인데, 이전 세션은 구성을
  기록하지 않았습니다. 여기의 생성기는 위의 직사각형에서 RETICOLO 자신의 `li=0`과 4.2e-11까지 일치합니다.
  래스터 NVF는 5에서 둘 다 0.0597을 줍니다.
- *Au 나노디스크 배열.* 주기 450 nm, 원판 D = 140 nm, h = 50 nm, `n = 1.5` 환경. `truncation_order`
  10에서 fixed 중첩은 두 편광의 플라즈몬 공명을 약 25 nm 떨어뜨려 놓습니다(795 nm와 770 nm 근처). NVF는
  둘 다 780 nm 근처에 놓습니다.

symmetric 중첩은 인위적 결과인 편광 분리를 없앱니다. 계단은 없애지 못합니다. 이런 곡선 금속 경계에서 그
답은 여전히 절단에 따라 움직이며, NVF가 바로 이런 경계를 위해 만든 정식화입니다.

**기본값을 바꾸는 대가.** fixed 중첩이 기본값으로 남으므로 골든 값은 하나도 움직이지 않습니다. 테스트
모음이 고정하는 장면에서 fixed 대 symmetric(2026-09-25, 차수별 `|ΔR|`와 `|ΔT|` 중 큰 쪽, 그리고 전체
투과의 변화. 층을 아래에서 위로 쌓기 전인 2026-09-11의 코드에서 측정했습니다 -- `2d_2layer` 행은 어느
쪽이든 같은 물리적 장면입니다):

| 장면 | 차수별 | 전체 T |
| --- | --- | --- |
| `golden_legacy.npz` `2d_normal` (무작위 5x5, 절단 2) | 2.0e-1 | 1.5e-1 |
| `golden_legacy.npz` `2d_oblique` | 5.7e-2 | 2.9e-2 |
| `golden_legacy.npz` `2d_2layer` | 8.7e-2 | 2.0e-1 |
| `golden_legacy.npz` `2d_cfs` (integrated) | 5.1e-2 | 4.5e-2 |
| `golden_field.npz` `2d`, `2d_2layer`, `2d_first_order` | 5.7e-2에서 8.7e-2 | 2.5e-3에서 9.5e-2 |
| 테스트 자체의 48x48에서의 L자 형상, 절단 4와 6 | 4.8e-3에서 1.2e-2 | 4.8e-3에서 1.2e-2 |
| 키랄 L, 절단 (5, 3)에서 (11, 9) | 4.3e-4에서 1.6e-3 | 1.9e-4에서 2.1e-3 |
| `test_optimization_smoke.py`의 유전율 경사 | 2.5e-5 | 3.9e-5 |

모든 장면에서 두 중첩 모두 에너지를 6.8e-14 이내로 보존합니다. 큰 수치는 절단 2의 무작위 래스터에서 나온
것이며, 그곳에서는 어떤 인수분해 선택이든 차이가 큽니다. 이 차이는 수렴합니다. `2d_normal`에서 전체 T의
차이는 절단 2, 4, 8, 12, 16, 20에서 1.5e-1, 1.4e-2, 6.3e-4, 5.1e-4, 4.4e-4, 2.6e-4입니다. 따라서 기본값을
뒤집으면 `golden_legacy.npz`와 `golden_field.npz`의 모든 2D `li` 골든 값이 허용 오차 1e-12를 훨씬 넘어
움직이며, 레거시 파일은 그 자리에서 다시 고정할 수 없습니다 — `tests/rcwa/test_golden_legacy_parity.py`가
퇴역 후 다시 고정하는 절차를 설명합니다. 1D 골든 값은 하나도 움직이지 않습니다. 1D RETICOLO 경우 열두 개에서
두 중첩은 마지막 비트까지 같은 효율을 반환합니다. 그 측정 이후 추가된 고정 테스트 세 개도 기본값에 묶여
있습니다. `tests/rcwa/test_tensor_li.py::test_scalar_tensor_is_exactly_the_scalar_li_operator`는 스칼라
텐서에서 텐서 Li를 스칼라 Li 연산자와 3e-12 이내로 맞추며, 텐서 Li는 fixed 중첩만 만들므로 텐서 Li가
symmetric 중첩을 얻지 않는 한 기본값을 뒤집으면 이 테스트가 깨집니다.
`tests/rcwa/test_rectilinear_li.py`는 데카르트 형상 Li를 칠한 래스터와 비교하는데 양쪽 모두 기본값을
거치므로, 모든 형상 Li 수치가 움직이는 동안에도 여전히 통과할 것입니다 -- 그 수치 중에는
`studies/rcwa/nvf_seam_contrast/nvf_seam_contrast.py`와
`studies/rcwa/nvf_seam_contrast/records/nvf_seam_contrast.log`의 명시적 Li 기준(`li_T`)도 있습니다.
`tests/references/test_tensor_li_golden.py`는 움직이지 않습니다. 텐서 Li는 중첩을 읽지 않습니다. 뒤집자는
논거는 공변성, 그리고 RETICOLO 자신의 `li=0`과의 일치입니다. 반대 논거는 지금까지 측정한 모든 2D `li`
수치가 fixed 중첩에서 얻은 것이라는 점입니다. 이는 관리자가 결정할 일입니다.

`li` 결과를 읽는 사람에게는 두 가지가 따라옵니다.

- 남은 차이는 에너지 오차가 *아닙니다*. 두 중첩 모두 1e-13 이내로 보존하므로, 어떤 보존 검사도 이를
  드러내지 않습니다. 이 차이는 일치해야 할 두 편광 사이의 **전체** 투과 전력 차이로 나타납니다.
- 테스트 모음은 이를 양방향으로 고정합니다. `test_nvf_does_not_tell_x_and_y_apart`와
  `test_li_does_not_tell_x_and_y_apart_in_the_symmetric_nesting`이 불변량입니다.
  `test_li_tells_x_and_y_apart_and_this_is_expected`는 fixed 중첩의 대가를 위*와* 아래로 제한해 기록하므로,
  대가를 악화시킨 회귀는 실패하고, 이를 조용히 고쳐 이 절을 낡은 채로 남긴 변경도 실패합니다.
  `test_li_is_covariant_when_the_whole_scene_is_turned_in_the_symmetric_nesting`은 돌린 장면을 고정합니다.
  네 테스트 모두 `tests/rcwa/test_factorization_core.py`에 있습니다. 행렬 수준의 진술 — 전치한 층의 `epx`는
  이름을 바꾼 `epy`이다 — 는
  `tests/rcwa/test_li_nesting.py::test_the_symmetric_nesting_is_covariant_under_relabelling_x_and_y`이고,
  RETICOLO 비교는 `tests/references/test_reticolo_li_nesting.py`입니다.

**코드의 출처.** 2026-09-25까지 여기의 래스터 생성기는 meent의 `dfs2d`와 `cfs2d`를 옮긴 것이었고, meent의
2차원 Li 규칙 코드는 RETICOLO V9의 MATLAB 소스(J.P. Hugonin and P. Lalanne,
[arXiv:2101.00901](https://arxiv.org/abs/2101.00901))를 참고해 `retcouche.m`의 `ret2li`, `retf.m`,
`rettoeplitz.m`을 따라 작성되었습니다. 이후 이 절의 생성기는 RETICOLO나 meent의 소스, 또는 교체 대상인
함수 본문을 참고하지 않은 구현자가 Li의 논문으로부터 다시 작성했고, `compression_indices` /
`compress_cell`은 이전 코드를 본 적 없는 구현자가 이 함수들이 무엇을 반환하는지에 대한 서면 명세로부터
다시 구현했습니다(명세 작성자는 이전 함수와 `ret2li`를 본 적이 있었습니다). `LICENSE`가 그 방법을 기록하며,
이 저장소의 이력에 아직 남아 있는 이전 버전에 대해 RETICOLO 출처 표시(CC BY 4.0, V10 기록에서)를
제공합니다. 재작성이 반영되기 전에 그 옆에 추가된 두 Li 빌드도 재작성된 코드 위에서 동작합니다. 데카르트
형상 Li(`rectilinear.py`)는 `fourier2d_integrated`를 호출하고, `_fourier1d_integrated`, `_wrap`,
`_circulant_gather`를 직접 호출하던 텐서 Li의 행 변환(`tensor_li.py`)은 재작성을 그 변경 위로 리베이스할 때
`_raster_series` / `_segment_series`로 옮겨졌습니다. fixed 중첩이 기본값으로 남은 것은 여기의 모든 골든
값이 그것으로 계산되었기 때문이며, 그것은 RETICOLO의 기본값이기도 합니다. 재작성은 이전 생성기를 반올림
오차 수준으로 재현하며, 균일한 층에서는 정확합니다. 예전의 표본 경로는 2의 거듭제곱이 아닌 격자에서 대각선
밖에 1.3e-16을 남겼습니다.

## 7. 두 가지 일을 하던 단어 하나 더: 격자 생성기

`dfs2d(..., piecewise_constant=False)` — 나중의 `_fourier2d_grid`이며, 2026-09-25 재작성(§6) 이후
없어졌습니다 — 는 `laurent_conv_2d`와 같은 결함을 한 단계 아래에서 갖고 있었습니다. 이 플래그는 선호를
표현하지 않습니다 — **격자가 무엇을 담는가**를 진술하며, 답은 정확히 두 가지이고 그것을 쓰는 소비자도 정확히
두 가지입니다.

| 진입점 | 격자가 담는 것 | 호출처 |
| --- | --- | --- |
| `fourier2d_piecewise_constant` | 자기 픽셀 전체에서 유지되는 값 — `eps`, `1/eps` | `wavebit/` 안에 5곳 |
| `fourier2d_pointwise` | 연속 필드의 점 표본 — NVF 법선장 | `wavebit/` 안에 2곳 |

개수는 호출처가 추가되면서 달라지므로 믿지 말고 다시 확인합니다. 이 글을 쓸 때는 5와 3이었고, 2026-08-17에
한 곳이 pointwise에서 piecewise로 옮겨졌으며(`ee715df`, PN z 칸), 2026-09-25 전에 마지막으로 셌을 때는 7과
5였습니다. 그날 Li 생성기가 한 칸짜리 진입점을 거치지 않게 되었고(§6), 텐서 Li의 `zz` 칸(2026-09-24)은
오늘의 다섯 곳 중 하나입니다. 달라지지 않는 것은 이 나뉨 자체입니다.

호출자가 결코 헷갈릴 리 없는 값은 호출자가 다시 말해야 하는 매개변수가 되어서는 안 되며, 기본값이 이를 더
나쁘게 만들었습니다. 변경 당시 `False`는 8곳 중 3곳이 원하는 답이었으므로, 잊어버렸을 때 얻게 되는 것이
*소수* 경우였습니다. 그 뒤 `_fourier2d_grid`를 비공개로 만들고 플래그를 기본값 없는 키워드 전용 인자로
바꾸었으므로, 직접 호출하더라도 어느 쪽인지 밝혀야 했습니다. 2026-09-25부터는 넘길 플래그가 남아 있지
않습니다. 각 진입점은 자기 데이터에 필요한 보정을 `_raster_series`에 넘깁니다 — 계단에는
`_pixel_correction`, 점 표본에는 `_half_pixel_phase` — 그리고 그 선택이 둘 사이의 차이 전부입니다.

`fourier2d_pointwise`는 규칙 인자도 기본값을 두는 대신 아예 없앱니다. 저장소의 모든 호출은 direct 규칙을
요청했습니다 — 우연이 아닙니다. inverse 규칙은 *동시에 불연속인* 인자들의 곱을 닫는 것인데, 구성상 연속인
단위 벡터장에는 그 규칙이 닫을 불연속이 없습니다. 이것은 누구에게 있던 선택이 아니라, 틀릴 수 있는 방법일
뿐이었습니다.

같은 추론이 2026-08-12에 규칙 인자 자체에도 적용되었습니다. 그 인자는 `conti_x` / `conti_y`였는데, 각
방향이 *연속*(`1`)인지 아닌지(`0`)를 표시하는 정수 쌍이었습니다 — 세 가지 의미에 네 가지 상태였습니다.
`(0, 0)`은 여기서 쓰는 Li 방식의 일부가 아니어서 어디에도 나오지 않았지만, 아무것도 이를 거부하지
않았습니다. 이름도 거꾸로였습니다. `conti_y=0`은 "y가 inverse 규칙을 받는다"는 뜻이었으므로, 값을 *지정한*
플래그가 그 플래그가 가리키는 축에 대해 참이 아닌 것을 말했습니다. `None` / `'u'` / `'v'` 중 하나를 받는
`inverse_rule` 하나는 같은 내용을 독자가 생각하는 방향으로 말하며, 네 번째 상태를 단지 쓰이지 않는 것이
아니라 표현할 수 없는 것으로 만듭니다.

문서화하는 대신 이렇게 고칠 가치가 있었던 이유는, 두 실수가 대칭이 아니며 위험한 쪽이 조용하기
때문입니다. 유전율 래스터를 표본 빌드에 넣으면 여유 있는 격자(N=64, truncation_order=4)에서 차수당
5.7e-4, 빡빡한 격자(N=24, truncation_order=10)에서 4.9e-3의 비용이 들고, 대략 `N < 2*truncation_order`
아래에서는 inverse 규칙 행렬이 특이해져 계산이 `LinAlgError`를 일으킵니다(측정: N=16,
truncation_order=8). 그 모든 경우에 에너지는 1e-14 이내로 보존되므로, 하류의 어떤 것도 아무것도 보고하지
않습니다. 반대 실수 — 표본 필드를 계단으로 만드는 것 — 는 굴절률이 맞는 기판으로의 수직 입사에서 1e-5,
대비가 있거나 conical 입사에서 2e-4에서 2e-3이며, 결코 구조적인 문제가 되지 않습니다.

`tests/rcwa/test_nvf_normal_field.py::test_the_projector_is_built_from_the_sampled_transform`이 이 선택을
하는 유일한 제품 코드 위치에서 그 선택을 고정합니다. 이 테스트에는 두 절반이 필요합니다. 그 파일의
나머지가 검사하는 사영 연산자 항등식 `C_xx + C_yy = I`는 *어느* 빌드에서도 성립하기 때문입니다. 표본 빌드가
실행된다는 단언 하나, 그리고 두 빌드가 충분히 멀리 떨어져 있어서(여기서는 1.1e-2) 첫 단언이 실제로 무언가를
고정한다는 단언 하나입니다.

`tests/rcwa/test_parametric.py`의 규약 탐침은 세 번째 이유로 `fourier2d_pointwise`를 호출합니다. 원시
`np.fft.fft2`와 비교하므로 픽셀 보정이 끼어드는 것을 원하지 않습니다. 이름은 여전히 맞습니다 — 보정되지
않은 빌드를 요청하는 것은 곧 입력이 표본이라고 *선언하는* 것입니다 — 그러나 그곳의 의도는 "이것은 측정된
필드다"가 아니라 "평범한 DFT를 보여 달라"입니다.

## 8. 끝까지 `sampled`와 `integrated`

`DFS`와 `CFS`는 예전에 층의 푸리에 계수를 얻는 두 방법의 약칭이었고, `sampled` / `integrated`는 같은
선택의 공개 표기였습니다. 한 가지에 두 어휘가 있었고, 내부 어휘는 이 모듈에서 가장 큰 무게를 지는 단어와
충돌했습니다. 바로 **continuous**(연속)입니다. Li 규칙은 어느 필드 성분이 경계면을 가로질러 *연속*인지에
따라 정해지므로, `CFS = continuous Fourier series`는 바로 옆의 물리가 필드의 성질을 가리키는 데 쓰는 그
단어를 방법에 붙인 셈이었습니다. `conti_x` / `conti_y`는 없어지기 전에 이미 이 단어를 세 번째 의미로 가져다
쓰고 있었습니다.

공개 어휘가 모든 곳에서 이겼습니다.

| 예전 | 지금 |
| --- | --- |
| `cfs2d`, `_cfs` | `fourier2d_integrated`, `_fourier1d_integrated` (2026-09-25부터 `_segment_series`) |
| `to_conv_mat_raster_discrete` | `to_conv_mat_raster_sampled` |
| `to_conv_mat_raster_continuous`, `..._continuous_nvf` | `..._integrated`, `..._integrated_nvf` |
| `nvf_conv_matrices_continuous` | `nvf_conv_matrices_integrated` |
| 글, 표, 연구 모드 이름의 `DFS` / `CFS` | `sampled` / `integrated` |

`NVF`는 남습니다. *normal vector field*(법선 벡터장)의 약자로, 방법이 아니라 사물을 가리키며, 다른 축에
놓입니다. `sampled`와 `integrated`는 `fourier_coefficients`의 두 값이고, `nvf`는 `factorization`의 값 중
하나입니다(`li`와 함께; 2026-09-27부터 `laurent`가 세 번째 값, §9). `sampled` / `integrated`와 `li` / `nvf`의 네 가지 조합은 모두 실제 생성기이며, 그래서 `to_conv_mat_raster_integrated_nvf`에는 두 단어가
모두 들어 있습니다.

이 작업을 반복하려는 사람에게 주의할 점 하나. `DFS`와 `CFS`는 명사였고, `sampled`와 `integrated`는
형용사입니다. 하나씩 그대로 바꾸면 "the sampled/sampled path", "match the legacy integrated", "Raster with
sampled" 같은 표현이 나옵니다 — 여기서 무심코 일괄 치환했을 때 실제로 이런 표현이 모두 생겼고, 나중에
명사구로 다시 썼습니다. 통하는 규칙은 *the sampled path*, *integrated coefficients*, *the integrated
build*처럼 쓰는 것입니다.

**이 어휘가 알려 주지 않지만 알려 주어야 할 것 하나.** `fourier_coefficients`는 *유전율*만 다스립니다 —
법선장은 결코 아닙니다. 그러나 그 이유는 경로마다 다르며, 둘 중 하나만 짐작할 만한 이유입니다.

**래스터** 경로에서는 두 값 모두 받아들여지며 법선장은 영향을 받지 않습니다. `_nvf_projector`는
`nvf_conv_matrices`에서도 `nvf_conv_matrices_integrated`에서도 똑같이 `fourier2d_pointwise`를 호출합니다.
표본한 필드에는 적분할 닫힌 형태가 없기 때문입니다. 따라서 `('integrated', 'nvf')`는 `eps`와 `1/eps`를
정확히 적분하면서도 사영 연산자는 여전히 표본합니다. 구현의 빈틈이 아니라 법선장이 원래 그런 것입니다.

**형상** 경로(매개변수 층 dict)에서는 이 질문이 아예 생기지 않습니다.
`fourier_coefficients='sampled'`는 `NotImplementedError`로 바로 거부됩니다 — 매개변수 기하에는 이산 급수를
만들 표본이 없습니다 — 따라서 그곳에서는 `'integrated'`가 유일한 값입니다.

두 경로는 필드의 격자가 어디서 오는지도 다르며, 이는 쉽게 당연시하고 넘어가는 부분입니다. 32x32 래스터와
원형 형상 층에서 호출을 세면:

| 경로 | `lattice.sample_grid` 호출 수 |
| --- | --- |
| 래스터, `'sampled'` | 0 |
| 래스터, `'integrated'` | 0 |
| 형상 (해석적 dict) | 1 |

`sample_grid`는 래스터가 아닌 생성기만을 위한 것입니다. 래스터 법선장은 인덱스 기반이며, 셀 좌표를 한 번도
묻지 않고 `factorization.normal_vector_field`를 거칩니다.

## 9. 선택지로서의 Laurent 규칙

2026-09-27에 추가되었습니다. `factorization='laurent'`는 유전율과 필드 성분의 **모든** 곱을 direct 규칙으로
닫습니다 -- 경계면을 가로질러 무엇이 연속이든 평범한 컨볼루션 행렬 `[[eps]]`를 씁니다. 어떤 것도 자동으로
이 규칙으로 정해지지 않습니다. 래스터는 여전히 Li가, 형상은 NVF가 기본값입니다. 다른 도구들이 이 규칙을
쓰기 때문에 제공합니다 -- fmmax의 `Formulation.FFT`, 그리고 RCWA가 처음 출발한 정식화 -- 그리고 그들과
같은 조건으로 비교하려면 이쪽에서도 같은 규칙이 필요합니다.

### 칸

§3의 BTTB 컨볼루션 행렬을 `[[f]]`라 하면, 양쪽 모두 direct 규칙으로:

| 칸 | `'laurent'` | 비교용 `'li'`, `li_nesting='fixed'` |
| --- | --- | --- |
| `epx` (`E_x`에 작용) | `[[eps]]` | `( T_u( [T_v(eps)]^-1 ) )^-1` |
| `epy` (`E_y`에 작용) | `[[eps]]` | `T_u( [T_v(1/eps)]^-1 )` |
| `epz_i` | `[[eps]]^-1` | `[[eps]]^-1`, 같음 |
| `epxy`, `epyx` | `None` | `None` |

자성 층은 `(E, H, eps, mu) -> (H, -E, mu, eps)` 쌍대성에 따라 같은 규칙을 받습니다. 두 면내 칸에는
`[[mu]]`, z 칸에는 `[[mu]]^-1`입니다. 두 면내 칸은 한 번 만든 *하나의* 행렬이므로 비트 단위로 같습니다.

**z 칸은 두 규칙이 일치하는 곳이며, 그 값은 `[[eps]]^-1`이지 `[[1/eps]]`가 아닙니다.** 계산에는 `D_z`로부터
`E_z`가 필요합니다. `E_z`는 모든 수직 경계면에 접선 방향이고 연속이므로, `D_z = eps E_z`는 direct 규칙이
올바르게 닫는 곱이며 -- Li 규칙도 이렇게 다룹니다 -- 이를 역행렬로 바꾸면 `[[eps]]^-1`이 됩니다.
`[[1/eps]]`는 두 인자가 함께 뛰는 `E_z = (1/eps) D_z`에 direct 규칙을 적용한 것입니다. 이는 측정 가능할
만큼 다른 행렬이고(`tests/rcwa/test_laurent_factorization.py`의 교차 격자에서 1e-2 넘게 차이), 아래의 어느
출처도 쓰지 않는 행렬입니다. Moharam과 Gaylord의 정식화는 그 자리에서 유전율의 Toeplitz 행렬을 역행렬로
바꾸며, fmmax의 고유값 문제도 그 `z_permittivity_matrix`에 대해 풀므로 마찬가지입니다. fmmax의 별도
`[[1/eps]]`(`inverse_z_permittivity_matrix`)는 필드 재구성을 위해 보관되며, 고유값 문제나 산란 행렬에는
들어가지 않습니다.

규칙 사이에 남는 차이는 면내 쌍입니다. x 방향으로만 변하는 격자에서 TE는 `epy`만 읽고, 그곳에서는 두 규칙
모두 `[[eps]]`를 만듭니다. 답은 모든 차수에서 반올림 오차 수준으로 같습니다. TM은 `epx`를 읽는데, Li는
`[[1/eps]]^-1`을, Laurent는 `[[eps]]`를 만듭니다 -- 느리게 수렴하는 경우입니다.

출처: M. G. Moharam and T. K. Gaylord, JOSA **71**, 811 (1981), 정식화;
M. G. Moharam, E. B. Grann, D. A. Pommet and T. K. Gaylord, JOSA A **12**, 1068 (1995), 이진 격자에 대해
두 편광 모두로 이를 기술; L. Li, JOSA A **13**, 1870 (1996), 느리게 수렴하는 이유 -- 경계면에서 뛰는 필드
성분이 같은 곳에서 뛰는 유전율과 만나면, 절단한 급수들의 곱은 곱의 급수로 수렴하지 않습니다. 코드는
`laurent_conv_matrices`에서 이들을 인용합니다.

### 어디서 만들고, 어디서 거부하는가

| 입력 | `'laurent'` | 이유 |
| --- | --- | --- |
| 스칼라 래스터, `'sampled'` | 만듦 | 정확한 계단 계수. 아무것도 결정하지 않으므로 고정된 기하 없이도 추적(trace)됨 |
| 스칼라 래스터, `'integrated'` | 만듦 | Li의 integrated 빌드가 기록하는 것과 같은 블록 압축 |
| 형상, 서로 떨어진 경우 | 만듦 | NVF가 출발점으로 삼는 닫힌 형태 `[[eps]]`에서 법선장을 뺀 것 |
| 형상, 칠한 겹침 | 만듦 | 겹침을 정리한 다각형의 `[[eps]]`. 곡선 기본 도형은 NVF에서처럼 꼭짓점 256개로 다각형화 |
| 기울어진 셀 | 만듦 | 이 규칙은 축을 지정하지 않음 -- 두 칸에 하나의 스칼라 컨볼루션 -- Li는 거부되는 곳 |
| 1D TE / TM, conical | 만듦 | TE는 반올림 오차 수준으로 Li와 같음. TM은 Li (1996)의 느린 경우 |
| `formulation='first_order'` | 만듦 | 같은 다섯 칸. 2차 정식화와 1차 정식화가 9e-15까지 일치 |
| 텐서 재료 | **이름으로 거부** | 텐서 생성기는 텐서 Li와 Popov–Nevière NVF입니다. Laurent 텐서 빌드는 작성되지 않았고 대조한 기준도 없습니다. 균일한 텐서는 어느 규칙에서도 같으므로, 그곳에서는 `factorization`을 지정하지 않아도 잃는 것이 없습니다 |
| `li_nesting`, `raster_normal_method`, `nvf_projector`, `shape_nvf_grid_size` | **이름으로 거부** | Laurent는 규칙을 중첩하지 않고 법선장을 만들지 않으므로, 이들을 읽는 것이 없습니다 |

**고정된 기하에는 그 규칙이 찍혀 있습니다.** `StaticGeometry.factorization`은 기록이 고정될 때의 규칙을
기록하며, 다른 규칙으로 재생하면 즉시 실행(eager)에서도 `jax.jit` 아래에서도 이름으로 거부됩니다. 형상의
Laurent 기록은 칠하기를 계획하지만 경계면 집합은 계획하지 않으며, NVF 재생은 이를 "계획 없음(plan
blind)"으로 읽을 것입니다. integrated 래스터의 Li 기록에는 법선장이 없으며, NVF 빌드는 즉시 실행에서는
이를 다시 만들고, 추적 중에는 아예 만들 수 없을 것입니다. 손으로 만든 기록에는 표시가 없으며 믿고
재생됩니다.

### 어떻게 수렴하는가

`studies/rcwa/laurent_convergence/laurent_convergence.py`이며,
`studies/rcwa/laurent_convergence/records/laurent_convergence.log`에 기록되어 있습니다. 파장 900 nm, 두께
250 nm, 위쪽 1, 아래쪽 2.25, 공기 중 eps 12.25.

1차원 이진 격자, 주기 700 nm, 채움비 50%, theta 10도. `|T - T_ref|`이며, 기준은 차수 400의 Li입니다.

| 차수 | TE, `li` | TE, `laurent` | TM, `li` | TM, `laurent` |
| --- | --- | --- | --- | --- |
| 5 | 1.15e-3 | 1.15e-3 | 3.43e-3 | 1.80e-2 |
| 10 | 3.11e-4 | 3.11e-4 | 4.90e-4 | 7.93e-3 |
| 20 | 4.11e-5 | 4.11e-5 | 7.57e-5 | 3.78e-3 |
| 40 | 5.22e-6 | 5.22e-6 | 8.26e-6 | 1.83e-3 |
| 80 | 6.52e-7 | 6.52e-7 | 4.93e-7 | 9.00e-4 |

TE는 모든 차수에서 Li의 답입니다. Laurent에서 TM은 차수를 두 배로 할 때마다 절반이 되고 -- 절단에 대해
1차 -- Li는 두 배마다 7배에서 17배 줄어듭니다. Li의 1996년 결과를 재현한 것입니다.

700 nm 정사각형 셀 안의 350 x 245 nm 막대를 40 x 40 래스터의 픽셀 위에 그려 모든 규칙이 같은 계단을 보게
하고, theta 10도, phi 20도로 둡니다. 편광별 `|T - T_ref|`이며, 기준은 차수 18의 Li입니다(축 정렬 직사각형에서는
Li가 정확한 규칙입니다).

| 차수 | `li` s | `li` p | `nvf` s | `nvf` p | `laurent` s | `laurent` p |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 3.3e-4 | 1.0e-3 | 3.7e-4 | 3.6e-4 | 7.6e-4 | 1.1e-2 |
| 8 | 8.5e-5 | 2.4e-6 | 1.2e-4 | 6.4e-4 | 2.3e-4 | 6.6e-3 |
| 12 | 1.9e-5 | 1.4e-5 | 3.3e-5 | 2.9e-4 | 3.2e-4 | 4.7e-3 |
| 14 | 1.3e-5 | 2.7e-5 | 2.6e-5 | 2.7e-4 | 3.1e-4 | 4.1e-3 |

Laurent는 수렴합니다 -- 그리고 이 모든 계산에서 에너지를 보존하므로 느린 수렴은 손실 때문이 아닙니다(로그에
가장 큰 `|R + T - 1|`이 기록되어 있습니다) -- 그러나 차수 14에서도 p에서는 Li보다 답에서 150배 더 멀리
있습니다. 형상 경로의 400 nm 원판에서는 비교할 Li가 없으며, Laurent는 차수 4에서 NVF보다 7.4e-2 /
5.4e-2(s / p), 차수 12에서 2.0e-2 / 1.3e-2만큼 아래에 있습니다. 위의 TM 열에서처럼 차수를 두 배로 할 때마다
차이가 대략 절반으로 줄어듭니다.

**절약되는 것은 측정할 만한 것이 없습니다.** 차수 12에서 막대 계산 한 번은 Li로 2.68 s, NVF로 2.74 s,
Laurent로 2.72 s가 걸리고, 원판은 NVF로 2.71 s, Laurent로 2.70 s가 걸립니다(세 번의 중앙값, 다른 작업과
공유하는 기계에서). 고유값 계산이 대부분을 차지하며, Laurent가 건너뛰는 법선장은 그 1% 미만입니다.
Laurent는 비교용 규칙이지 더 빠른 규칙이 아닙니다.

### fmmax와의 비교

`tests/references/test_fmmax_laurent_golden.py`. fmmax의 `Formulation.FFT`는 같은 연산자를 조립하므로,
계수를 일치시키고 나면 비교는 항등성 확인이 됩니다. fmmax의 DFT는 Wavebit이 각 픽셀에서 정확히 계산하는
푸리에 적분에 대한 중점 규칙이며, 둘 사이의 인자는 축마다 `y / sin(y)`이고 픽셀당 표본 수 `R`에 대해
짝함수이므로, 네 가지 표본화에 걸친 Richardson 외삽으로 이를 제거합니다. 700 x 560 셀 안의 실리콘 같은
막대, 구멍 뚫린 손실 금속 막, 육각 셀 위의 십자 -- Li가 거부되는 곳 -- 에서 Wavebit의 `'laurent'`는
외삽 결과와 전체에서 6.1e-13, 3.2e-11, 4.8e-13까지 일치하고, 네다섯 개의 전파 차수에서 차수별로도 같은
정도로 일치합니다. 거울상 차수 인덱스로는 1.3e-2에서 7.4e-2만큼, Li나 NVF로는 4e-2에서 2e-1만큼
벗어납니다.

이는 픽셀 중심 위상을 배정밀도로 계산한 fmmax와의 비교입니다. fmmax 1.7.1은 그 위상을 단정밀도 주파수
격자(`fmmax.fft._fft2`)에서 계산하는데, 이는 정확한 평행 이동을 차수에 의존하는 평행 이동으로 바꾸어 이
효율들에 1e-8에서 1e-7을 남깁니다. 생성기는 기본(stock) 실행도 고정해 두며, 테스트는 Wavebit을 두 실행
사이 차이의 두 배 이내로 그 실행에 맞춥니다. 패턴 층에서 대략 1e-7보다 나은 일치를 기대하는 fmmax FFT
정식화와의 비교는 먼저 그 위상을 고려해야 합니다.
