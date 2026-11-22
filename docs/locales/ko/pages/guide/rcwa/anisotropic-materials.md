# 이방성·자성 재료 — 솔버가 이미 받는 것과 각 공백의 비용

**현재 입력 API(2026-09-22):** `Material(epsilon=[exx, eyy, ezz])`
또는 3×3 텐서를 `rcwa.Layer` 배경이나 `rcwa.Region` 재료에 직접 지정하고 문제를
계산합니다(`rcwa.run`, 또는 `rcwa.prepare` 후 `prepared.run`).
텐서 형상은 이제 기본적으로 [연속 기하 적분](tensor-shape-integration.md)을 쓰며,
해석적 기본 도형 지시 함수와 표본 기반 법선 확장을 사용합니다.
재료 축은 `Material.rotated(angle, axis)`로 회전하고, 텐서 래스터는 모양이 `(Ny, Nx, 3, 3)`인
`rcwa.Layer(epsilon=...)` 배열입니다.
[이전 재료 설명서](../../migration/materials-v1.md#anisotropic-materials)에
텐서 표와 텐서 래스터 시연이 있습니다. 아래 논의는
과거의 유도와 개발 기록이며, 현재의 입력 가이드가 아닙니다.


작성: 2026-08-09. 당시에는 범위 검토만 한 문서였습니다. 2026-08-11부터 z 결합 행이
구현되었고, 2026-08-13부터는 *도달할 수 있습니다* — `solve_conv`가 아홉 개의 슬롯을 받고
솔버가 정식화를 고릅니다(§8, 더 이상 조건부가 아닙니다). 그 경로가 하는 일과 어디까지
검증되었는지는 [`first-order-formulation.md`](../../theory/rcwa/first-order-formulation.md)를 참고하세요.
이 문서는 범위 검토의 질문, 곧 *남은* 각 공백의 비용이 무엇인지를 그대로 다룹니다.

대상:
- `src/wavebit/rcwa/_engine/solve_2d.py::layer_modes_2d` — 모드 연산자가 읽는 다섯 개의 유전율 슬롯
- `src/wavebit/rcwa/_engine/factorization.py` — `epx` / `epy` / `epz_i`가 실제로 무엇인지(보이는 것과 다릅니다)
- `src/wavebit/_common/conventions.py::boundary_kz` — 층 외에 `mu`가 들어가야 할 곳
- `src/wavebit/rcwa/_engine/engine.py::solve_conv` — `structure`를 우회하는 주입 지점이자 정식화를 고르는 곳
- `src/wavebit/rcwa/_engine/solve_2d_first_order.py::solve_2d_first_order` — 2D 풀이의 아홉 슬롯 쌍둥이

---

## 요약

| 경우 | 상태 | 비용 |
|---|---|---|
| 균일(패턴 없는) 이방성 층, 면내 텐서 | **현재 동작** | 얇은 래퍼 |
| `eps_xz` / `eps_yz`(z 결합), 균일 | **현재 동작** | 완료 — 별도의 솔버 [`first-order-formulation.md`](../../theory/rcwa/first-order-formulation.md)로 자동 분기되며 비용은 ~8x |
| 패턴 이방성 층, 면내 텐서 | **현재 동작** | 완료 — Popov–Nevière, [`patterned-anisotropic-factorization.md`](../../theory/rcwa/patterned-anisotropic-factorization.md). 검토한 대로 두 솔버 모두 바뀌지 않음 |
| 패턴 **이면서** z 결합 | 지원 | epsilon과 mu에 대한 완전한 Li/NVF 닫힘과 1차 정식화. [검증](full-tensor-patterns.md) |
| `mu != 1`, 층과 반공간 | **현재 동작** | 완료 — [`magnetic-materials.md`](../../theory/rcwa/magnetic-materials.md). 층은 다섯 슬롯의 E/H 쌍대이고, 반공간은 어드미턴스 한 쌍 |

이 목록에 불가능한 것은 없습니다. 이 중 정확히 하나만이 코드의 내용이 아니라 *모양*을
바꾸었습니다. z 결합 행이며, 그래서 두 번째 솔버가 필요했고 실제로 생겼습니다. 나머지는
내용입니다. 같은 기계 장치에 들어가는 새 행렬일 뿐입니다.

패턴 행은 §3이 예측한 대로 들어왔습니다. 인수분해 하나이고, 두 솔버는 손대지 않았습니다.
그 아래 행은 둘의 조합이며, 어려움 때문이 아니라 근거가 없어서 미뤄졌습니다. 패턴 경우에 도달하는
솔버는 fmmax뿐이었는데 fmmax에는 z 행 인수가 없어서, 답을 비교할 대상이 없었습니다.
2026-09-24부터 지원하며, 검증은 [패턴 층 안의 전체 텐서 재료](full-tensor-patterns.md)에 있습니다.

## 1. 모드 연산자가 이미 받는 것

`layer_modes_2d`는 세 개가 아니라 다섯 개의 합성곱 행렬을 받습니다.

```python
def layer_modes_2d(xp, Kx, Ky, I, epx_conv, epy_conv, epz_conv_i, epxy_conv, epyx_conv, C):
    B  = Kx @ epz_conv_i @ Kx - I
    D  = Ky @ epz_conv_i @ Ky - I
    Cc = Kx @ epz_conv_i @ Ky
    E  = Ky @ epz_conv_i @ Kx

    Omega2_LR = _block(xp, [
        [Ky @ Ky + B @ epx_conv - Cc @ epyx_conv,  Cc @ epy_conv - Kx @ Ky - B @ epxy_conv],
        [E @ epx_conv - Ky @ Kx - D @ epyx_conv,   Kx @ Kx + D @ epy_conv - E @ epxy_conv],
    ])
```

이 다섯 슬롯은 유전율 텐서에 정확히 다음과 같이 대응합니다.

```
                    E_x        E_y        E_z
               ┌──────────┬──────────┬──────────┐
        D_x    │   epx    │   epxy   │    ·     │
               ├──────────┼──────────┼──────────┤
        D_y    │   epyx   │   epy    │    ·     │
               ├──────────┼──────────┼──────────┤
        D_z    │    ·     │    ·     │  epz_i⁻¹ │
               └──────────┴──────────┴──────────┘

   named cell = a slot the mode operator already reads
   ·          = assumed zero, no slot exists
```

따라서 풀리는 텐서 부류는 다음과 같습니다.

```
        [ exx  exy   0  ]
  eps = [ eyx  eyy   0  ]
        [  0    0   ezz ]
```

`epxy`와 `epyx`는 별도의 인수이지만, **실제로 올바르게 풀리는 것은 대칭인 쌍뿐입니다**.
§6.1을 참고하세요. `ezz` 성분은 역수로만
들어갑니다. `E_z`가 `E_z = ezz⁻¹ D_z`로 대수적으로 소거되기 때문이며, 필드
재구성이 `epz_conv_i` 외에는 아무것도 읽지 않는 이유도 같습니다(`fields.py::_field_coupled`).

슬롯은 모든 백엔드에서 `rcwa.operators`를 통해 사용자에게 닿습니다. `assemble`이 슬롯을 반환하고,
`solve`가 다시 받아 엔진의 `solve_conv`에 넘깁니다. 손으로 만든 패턴
슬롯은 조립된 슬롯처럼 셀 중심에서 잰 푸리에 좌표를 씁니다.

```python
from wavebit.rcwa import operators

ops = operators.assemble(problem, numerics=numerics)
conv = operators.ConvMatrices(epx_conv_all, epy_conv_all, epz_conv_i_all,
                              epxy_conv_all, epyx_conv_all)
res = operators.solve(problem, ops.replace(epsilon=(conv,)))   # one tuple per wavelength
```

1D 경로와 1D conical 경로는 다섯 개가 아니라 세 개를 받으므로(`engine.py::_solve_1d`, `::_solve_1d_conical`),
면내 비대각 항은 2D 경로에서만 쓸 수 있습니다.

## 2. 함정: `epx` / `epy` / `epz_i`는 텐서 성분이 아닙니다

이 셋은 **하나의 스칼라 유전율**에 서로 다른 세 가지 인수분해 규칙을 적용한 것입니다.

```
             structure   (one relative permittivity per pixel)
                                 │
                        eps = structure          (no squaring since 2026-08-20)
                                 │
          ┌──────────────────────┼──────────────────────┐
  ..._piecewise_constant  ..._piecewise_constant  ..._piecewise_constant
      (eps, 0, 1)            (eps, 1, 0)            (eps, 1, 1)
  inverse rule along x  inverse rule along y      direct rule
          │                      │                      │
       epx_conv               epy_conv               epz_conv ──inv──> epz_conv_i
```

(`inverse_rule`은 역규칙이 그 불연속을 닫는 셀 방향을 가리키며, 두 방향 모두 직접 규칙이면
`None`입니다.) NVF도 마찬가지입니다. `ep_dir = ⟦eps⟧`와
`ep_inv = ⟦1/eps⟧⁻¹`은 모두 같은 하나의 스칼라에서 나오며, `_nvf_assemble`이 반환하는 텐서는
경계면 투영자의 *유효* 텐서이지, 재료 텐서가 아닙니다.

**따라서 "`exx`를 `epx` 슬롯에 넣는다"는 올바른 치환이 아닙니다.** 역규칙은 슬롯의
성질이 아니라, 어떤 곱을 인수분해하느냐에 따른 결과입니다.

## 3. 치환이 실패하는 정확한 이유

Li 규칙은 경계면에서의 연속성에서 나옵니다.

```
      material A  │  material B
                  │
     E_x  ────────┤ ┌──────      normal to the interface -> jumps
                  │ │
     D_x  ─────────────────      normal component of D  -> continuous
                  │
              x ──┴──>           x-normal interface
```

연속인 양을 불연속인 두 양의 곱으로 쓴다는 사실이 `D_x = eps · E_x`에 역규칙을
강제합니다. 텐서에서는 연속인 양이 **세 곱의 합**입니다.

```
  D_x = exx·E_x  +  exy·E_y  +  exz·E_z
```

그리고 항마다 적용할 규칙이 없습니다. 표준적인 해결책(Popov–Nevière)은 구성 관계를 다시 정리해
연속인 양 — 접선 방향 `E`, 법선 방향 `D` — 을 한쪽에 모으고, 그 블록을 **실공간에서 점별로**
역변환한 다음, 전체에 직접 규칙을 적용하는 것입니다.
그 결과로 나오는 `eps̃` 행렬은 모든 성분이 채워져 있습니다.

**완료, 2026-08-14** — `factorization.py::pn_eps_conv_matrices`이며, 적층에는
`to_conv_mat_raster_pn_eps`를 씁니다. 전체 설명은
[`patterned-anisotropic-factorization.md`](../../theory/rcwa/patterned-anisotropic-factorization.md)에 있습니다. 그중 두 가지는
여기로 다시 가져올 가치가 있습니다. 둘 다 이 절이 가정했던 것과 어긋나기 때문입니다.

* "그 블록을 실공간에서 점별로 역변환한다"는 보기보다 많은 일을 합니다. 투영자
  대수는 무엇이든 변환하기 *전에* 끝나야 합니다. `⟦T⟧`, `⟦N⟧`과 전달 행렬
  `⟦M⟧`을 따로 만들어 행렬끼리 곱하는 것은 점별로는 같은 대수이지만 틀립니다. 법선장은
  내부로 확장한 단위 벡터장이므로 그 자체가 불연속이고, 따라서 그 곱은 동시에 일어나는
  불연속을 가로질러 Laurent 규칙을 적용한 것이 됩니다. 에너지를 보존하고, 회전에
  공변이며, 그럴듯하게 fmmax 근처에 있지만, **균일 슬래브에서 0.5만큼 틀립니다**.
* 전체 텐서는 닫는 단계에서 세 성분을 모두 씁니다. 자세한 내용은
  [패턴 층 안의 전체 텐서 재료](full-tensor-patterns.md)에 있습니다.

## 4. 균일 이방성 층은 현재 동작합니다

패턴이 없는 층에는 경계면이 없으므로 인수분해 문제가 생기지 않습니다. 모든
합성곱 행렬이 항등 행렬의 상수배입니다. 코드가 직접 이렇게 말합니다.

> Uniform layers cannot expose it at all — their convolution matrix is `eps * I`.
> (`factorization.py::_circulant_index`)

따라서 복굴절 박막, 단축 기판, LC 층은 다섯 슬롯을 직접 채워서 지금 바로 `develop`에서
풀 수 있습니다. 이는 주장이 아니라 검증된 사실입니다.
`tests/rcwa/test_uniform_anisotropic.py`는 아래의 `operators.solve`가 슬롯을 넘기는 엔진의
`solve_conv`를 구동해, 고유 편광마다 해석적인 Airy
슬래브와 Jones 파장판에 대해 비교하며, ~1e-16까지 일치합니다.

```python
import numpy as np
from wavebit import Units, rcwa
from wavebit.rcwa import operators

# One unpatterned layer; its assembled slots are replaced below.
problem = rcwa.Problem(units=Units(length='nm'),
                       stack=rcwa.Stack(period=(500., 500.), layers=(rcwa.Layer(thickness=220.),)),
                       excitation=rcwa.Excitation(wavelength=700., sources=(rcwa.PlaneWave(),)))
ops = operators.assemble(problem, numerics=rcwa.Numerics(orders=(half_width_u, half_width_v)))

n_orders = (2 * half_width_u + 1) * (2 * half_width_v + 1)
I = np.eye(n_orders, dtype=np.complex128)

# eps = [[exx, exy, 0], [eyx, eyy, 0], [0, 0, ezz]] for a single uniform layer
epx   = np.stack([exx * I])
epy   = np.stack([eyy * I])
epxy  = np.stack([-exy * I])      # slot convention is (-exy, -eyx); see _nvf_assemble
epyx  = np.stack([-eyx * I])
epz_i = np.stack([np.linalg.inv(ezz * I)])

conv = operators.ConvMatrices(epx, epy, epz_i, epxy, epyx)
res = operators.solve(problem, ops.replace(epsilon=(conv,)))
```

전체 텐서는 **같은** 부호 규칙 아래에서 네 개를 더합니다. 한 호출의 가운데에서 두 규칙으로
나뉘는 것이 아니라, 모든 비대각 슬롯에 규칙 하나를 씁니다.

```python
epxz, epyz = np.stack([-exz * I]), np.stack([-eyz * I])
epzx, epzy = np.stack([-ezx * I]), np.stack([-ezy * I])

conv = operators.TensorConvMatrices(epx, epy, epz_i, epxy, epyx,
                                    epxz, epyz, epzx, epzy)   # nine slots: first-order formulation
res = operators.solve(problem, ops.replace(epsilon=(conv,)))
```

이제 균일 층에는 둘 다 필요하지 않습니다. 같은 텐서를 층 배경의 `Material`로 주면,
`rcwa.Layer(thickness=220., background=Material(epsilon=[[exx, exy, exz], ...]))`가
이 슬롯들을 스스로 만들고, 그 효율은 손으로 채운 경우와 비트 단위로 같습니다.

주의할 점이 두 가지 있습니다.

- **슬롯 부호 규약**은 `(epx, epy, epz_i, epxy, epyx) = (exx, eyy, ⟦eps⟧⁻¹, -exy, -eyx)`이며,
  z 행도 이를 그대로 확장합니다: `(epxz, epyz, epzx, epzy) = (-exz, -eyz, -ezx, -ezy)`.
  이 규약은 모드 연산자가 아니라 `_nvf_assemble`이 정하며, 거꾸로 써도 거의
  알아챌 수 없습니다. `+eps_xy`는 광축을 `x`에 대해 거울 반사시키는데, **모든 전력 효율이
  비트 단위로 같게 남습니다** — 같은 편광, 교차 편광, 전체 모두 그렇습니다. 복소 진폭만
  움직여서, `T_p`는 부호가 뒤집히고 `T_s`는 뒤집히지 않습니다. 논증이 아니라 측정한 결과입니다.
  `test_uniform_anisotropic.py::test_off_diagonal_sign_shows_up_in_the_amplitude_only`를 참고하세요.
  손으로 채운 텐서를 무언가와 비교한다면 진폭을 비교하세요.
- **2D 경로를 선택해야 합니다**. `engine.py::_assign_grating_type`은 `structure`의 모양으로 경로를 고르므로,
  `exy != 0`인 순수 다층 적층은 2D 풀이로 보내야 합니다. 1D 경로와 conical
  경로에는 채울 비대각 슬롯이 없습니다. z 행은 스스로 이렇게 하지만(§8) 면내 쌍은
  그렇지 않으며, 이것이 기억해 둘 비대칭입니다. 현재 생애주기에서는 `background`로 선언한 층은
  (`regions`가 있든 없든) 항상 2D 경로에 있고, 텐서 `Material`을 담은 층도 마찬가지입니다.
  직사각형 셀에서 한 행짜리 `epsilon` 래스터가 1D conical 경로로 가는 경우입니다.

## 5. 자성 재료: 층과 반공간의 `mu`

**층: 완료, 2026-08-14.** `layer_modes_2d`는 선택적인 `mu`를 받습니다. 유전율과 같은 다섯 슬롯을
같은 규약으로 받으며, `solve_2d` / `smm_2d`는 이를 층마다 가지고 갑니다.
유도와 검증 방법은 [`magnetic-materials.md`](../../theory/rcwa/magnetic-materials.md)에 있습니다.

이 절은 예전에 다음의 두 `- I` 항을 가리켰습니다.

```python
B = Kx @ epz_conv_i @ Kx - I        #  <- this I is mu_conv
D = Ky @ epz_conv_i @ Ky - I        #  <- and this one
```

위치는 맞았지만 내용은 세 가지 면에서 틀렸고, 구현은 그 셋을 모두 바로잡아야 했습니다.

* 짝이 **엇갈립니다** — `B`는 `mu_yy`를, `D`는 `mu_xx`를 받습니다. 스케치를 따랐다면
  등방성 `mu`에서는 정확히 맞았겠지만 단축 투자율은 조용히 전치되었을 것입니다.
* 슬롯이 두 개 더 있습니다. `Cc`와 `E`는 비대각 `mu`를 받고, `mu_zz`는 `Omega_R`에서
  `K` 인자 사이에 들어갑니다. `epz_conv_i`가 `Omega_L`에서 놓이는 자리를 거울처럼 따릅니다.
* `Omega2_LR`을 손으로 전개한 형태는 **재사용할 수 없습니다**. 그 안의 네 개의 맨 `K @ K` 항은
  곱셈에서 살아남은 기하처럼 보이지만, 실제로는 `K` 인자가 1인 `muz_conv_i`를 지나 교환될 때
  다른 두 항이 줄어든 결과입니다. 이를 재사용하면 스칼라 `mu`에서는 에너지가 보존되었지만
  자이로자성 `mu`에서는 완전히 깨졌습니다(`R + T = 1.035`).

**반공간: 이것도 완료되었습니다**. 원래의 경고가 다룬 것이 바로 이 부분입니다. 반공간은 예전에
구성상 등방성이고 비자성이었습니다.

```python
kz_top = boundary_kz(xp, n_top ** 2, k_par_sq)          # takes eps alone
big_F  = _block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]])   # p-channel admittance
```

`big_F`의 `/ eps_bot`이 바로 `mu`가 들어가야 할 어드미턴스입니다. 여기의 지침은
예전에 "골든 값을 손에 들고 바꾸라"였고, 실제로 그렇게 되었습니다.
`golden_fmmax_mu_ambient.npz`를 먼저 고정했는데, 그때는 아직 `mu`를 넣을 곳이 없었습니다.
그리고 그 위에 **닫힌 형태의 해**도 담았습니다. 자성 층의 골든 값은 쓸 수 없었는데,
그 안의 모든 경우가 진공 사이에 있기 때문이며, 바로 그 점 덕분에 층 연산자를 단독으로
검증할 수 있었습니다.

알고 보니 `boundary_kz`는 전혀 바꿀 필요가 없었습니다. 이미 `eps`라는 이름으로 `n^2`를 받고
있었고, `n_top` / `n_bot`은 굴절률로 남았습니다. 지금도 그렇습니다 — 2026-08-20부터 이들은
`top` / `bot` 재료에서 *유도되며*(`n = sqrt(eps*mu)`) 읽기 전용입니다. 그래서 입구에서
유전율로 바꾼 변경이 코드의 이 절반에는 손대지 않았습니다
([`conventions.md`](../../reference/common/conventions.md)). 바뀐 것은 어드미턴스 한 쌍,
s의 `kz/mu`와 p의 `kz/eps`입니다 —
[`magnetic-materials.md`](../../theory/rcwa/magnetic-materials.md) §5를 참고하세요.

## 6. z 결합: 솔버를 바꾸는 유일한 공백

### 무엇인가

`z` 첨자를 가진 텐서 성분 `exz`, `ezx`, `eyz`, `ezy`입니다. `exz`는 "`E_z`가
`D_x`에 기여한다"는 뜻이고, `ezx`는 "`E_x`가 `D_z`에 기여한다"는 뜻입니다. §1 그림의 `·` 칸이 이것입니다.

이 성분들은 **광축이 xy 평면 밖으로 기울어질 때마다** 나타납니다.

```
     axis ∥ z                    axis ∥ x                  axis tilted by a
        z                           z                          z
        │ ↑                         │                          │    ↗
        │ │                         │                          │  ↗
        └─┴──── x                   └───→──── x                └────────── x

  diag(eo, eo, ee)            diag(ee, eo, eo)         R_y(a)·diag(eo,eo,ee)·R_y(a)ᵀ
  no z-coupling               no z-coupling            exz = ezx = (ee-eo)·sin a·cos a
                              (in-plane rotation                    ≠ 0
                               only makes exy)
```

실제로 이것은 특이한 경우가 아닙니다. 프리틸트가 있는 LC 방향자는 거의 정의상 기울어져 있고,
축에서 벗어나게 자른 복굴절 결정이나 비스듬히 자화된 자기광학 박막도 마찬가지입니다.

### 2차 형식이 깨지는 이유

`E_z`와 `H_z`를 소거하면 횡방향 필드에 대한 `z`의 1차 연립 방정식이 남습니다.

```
   d psi / dz = M psi,        psi = [E_x, E_y, H_x, H_y]      (4N x 4N for N orders)
```

z 결합이 없으면 `M`은 블록 반대각입니다.

```
   no z-coupling                            with z-coupling

   M = [  0    P  ]                         M = [ M11    P  ]
       [  Q    0  ]                             [  Q    M22 ]

   dEt/dz depends on H only                  dEt/dz depends on E too
   dHt/dz depends on E only

   -> d²Et/dz² = P·Q·Et    (second-order)   -> a first-derivative term survives;
      no first-derivative term                  no clean second-order reduction
```

`P·Q`를 전개한 형태가 *바로* `Omega2_LR`입니다. 오른쪽 인자는 `V = (Omega_R @ W) / q`로 `H`를
되찾을 수 있도록 `Omega_R`로 따로 보관합니다. 그리고 방정식이 순수한 2차이므로,
`exp(+q z)`와 `exp(-q z)`는 **같은 고유벡터**로 방정식을 풉니다 — `±` 쌍이 공짜로 얻어집니다.

```
   second-order eigenproblem            first-order eigenproblem
   ─────────────────────────            ────────────────────────
   eig(Omega2_LR) -> (W, q²)            eig(M) -> 4N eigenpairs
   down: W·exp(-k0 q d)                 forward and backward are genuinely
   up:   W·exp(+k0 q d)   same W        different mode sets; sort by decay
                                        and energy flux, not by sign
```

이 `±` 짝짓기는 모든 사용처에 박혀 있습니다. `big_W (I + Y)` / `big_V (I - Y)` 구성
(`solve_2d.py::solve_2d`), 층마다 저장하는 하나의 `[epz_conv_i, W, V, q, d, big_A_i, big_B]`,
`sp_modes_2d`의 네 사분면 분할, `conventions.py::mode_q`의 가지 규칙, 그리고 `fields.py`의 필드
재구성입니다. 이 모두가 2차 형식을 가정합니다.

비용은 구조적인 것만이 아닙니다. `eig`는 `O(N³)`이므로, 2차에서 1차로 가면 층 풀이 비용이 대략 **8x**가 됩니다.

**이 중 일부는 이제 존재합니다.** `src/wavebit/rcwa/_engine/solve_2d_first_order.py`는 1차
연산자 `M`, 순방향/역방향 분할, 적층 풀이를 담고 있으며,
`tests/rcwa/test_solve_2d_first_order.py`가 이를 2차 경로(그 경로가 유효한 곳에서
고유값이 `lam = i q`에서 1e-15까지 일치)와 Berreman 오라클(2차 경로로는 전혀 표현할 수 없는
텐서에서 성분 하나하나가 1e-16까지 일치)에 대해 고정하고,
`tests/references/test_first_order_external_goldens.py`가 RETICOLO와 fmmax에 대해 고정합니다(24개 경우, 최악
2.4e-15, 규약 맞춤 없음). 그중 세 개는 이를 위해 생성한 Voigt 텐서입니다 — **이 모듈이 존재하는
목적인 기하를 외부 솔버가 처음으로 판정한 것입니다.**

이 외부 검증의 범위에는 뚜렷한 경계가 있으며, 이는 가정이 아니라 측정한 것입니다. RETICOLO는
대각 텐서에 대해서는 어떤 입사 배치에서든, *단 하나의* 비대각 쌍에 대해서는 고전적 배치에서 정확합니다.
그 밖에서는 손실이 없음이 증명된 텐서에서 `R + T`가 1에서 최대 1.2e-2만큼 벗어난 값을 반환하며,
이는 수렴했고 절단과 무관합니다. 기울어진 자화 — xy와 yz가 함께 채워진 경우로, 실제 자화된 박막이
이렇습니다 — 는 그 경계 밖에 있으므로, 여전히 우리 자신의 Berreman에 의존합니다. 측정값과
추론은 `tests/references/README.md`에 있습니다.

이 글을 쓸 때는 아무것도 이 경로로 분기하지 않았지만, 2026-08-13부터는 솔버가 분기합니다(§8). 이 경로는
두께와 상관없이 안정적입니다. 층은 향상 투과율(enhanced-transmittance) 재귀
(`rcwa.Numerics(algorithm='etm')`, 2차 경로와 같은 문자열이자 같은 기법)
또는 산란 행렬(`'smm'`, 기본값)로 이어지며, 둘 다 모드 지수 2500까지 ~1e-13
상대 정확도를 유지하고, 거기서 닫힌 형태의 단일 경계면 Fresnel 반사율을 4e-16까지 재현합니다.
안정화하지 않은 전달은 교차 확인용으로 `'TMM'`으로 남겨 두었으며, 정확한 범위를 넘으면
거부합니다. 220 nm 층에서는 25개 푸리에 차수부터 그렇습니다. 라이트 라인(light line) 위의 회절 차수는
두께와 상관없이 강하게 소멸하기 때문입니다. 이것이 RCWA에 향상된 형식이 필요했던 과거의 이유이며,
두 재귀 사이의 선호 문제가 아닙니다. 블록 배치는 9개와 25개 차수에서 검증했습니다. 아직
없는 것은 차수 *결합*으로, §3의 패턴 인수분해가 필요합니다(2026-09-24부터
존재합니다. [패턴 층 안의 전체 텐서 재료](full-tensor-patterns.md)를 참고하세요). 필드 재구성은
2026-08-14에 들어왔습니다(`docs/theory/rcwa/first-order-formulation.md` §4b). 그 `E_z`는 2차 경로가
빌려줄 수 없었던 유일한 공식입니다. z 행이 채워지면 소거 과정에 항이 두 개 더 남기 때문입니다.

이를 만들면서 얻은 발견이 두 가지 있으며, 둘 다 해당 경우가 생기기 전에는 보이지 않았습니다.
위의 `±` 짝짓기는 1차 형식에서 단지 불필요해지는 것이 아니라 *틀리게* 됩니다. 수직 입사에서는
역방향 집합이 부호를 바꾼 순방향 집합과 여전히 1e-16까지 일치하지만, 35°에서 기울어진 자화의 경우
둘은 9e-2만큼 갈라집니다. z 결합과 비스듬한 입사가 함께 있어야 가정이 깨지며, 그래서
아무도 알아채지 못했습니다. 그리고 방향은 모드마다 그 모드 자신의 노름에 대해 정해야 합니다.
플럭스 문턱값을 집합에서 가장 큰 플럭스로 스케일하면 모든 전파하는 경우에서는 동작하지만,
모든 모드가 소멸하게 되면 — 라이트 라인 위의 모든 회절 차수가 그렇습니다 — 1e-17의
잡음을 실제 에너지 흐름으로 격상시킵니다.

## 6.1 자이로트로피: 두 블록이 쌍의 순서에 대해 엇갈립니다

*여기서는 요약만 합니다. 전체 유도, 그림, 모든 측정은
[`off-diagonal-slot-convention.md`](../../theory/rcwa/off-diagonal-slot-convention.md)에 있습니다.*

면내 반대칭(`exy = -eyx`, 곧 Faraday형 자기광학)은 2차 형식을 **깨지 않습니다**. 2차 형식을
깨는 것은 면내 비대칭이 아니라 z 결합입니다. 따라서 자이로트로피에는 새 정식화가
필요 없습니다. 아래의 유도는 비대칭을 포함한 임의의 면내 텐서에 대해 기하에만 의존하는 `Omega_L`을
만들어 냅니다. 부딪히는 문제는 그 형식의 **구현에 있는 순서 오류**입니다 — 한 블록이
비대각 쌍을 다른 블록과 반대 순서로 담습니다.

`layer_modes_2d`는 두 블록을 만들며, 둘은 비대각 쌍을 **반대 순서로** 담습니다.

```
   Omega2_LR   ...  - Cc @ epyx   ... - B @ epxy    (eigenvalues q², and W)
   Omega_R     ...  + epxy        ... - epyx        (the H-field, V = Omega_R W / q)
```

인수분해 `Omega2_LR = Omega_L @ Omega_R`은 `Omega_L`이 기하만으로 만들어질 때에만 의미가 있습니다.
기호적으로 보면 그렇지 않습니다 — 배포된 순서에서는 `Omega_L`이 `epx`, `epy`, `epxy`,
`epyx`에 의존하게 됩니다. `Omega_R` 안에서**만** 쌍을 맞바꾸면 이는 다음으로 줄어듭니다.

```
   Omega_L = [[-Kx Ky epz_i,      Kx² epz_i - 1],
              [-Ky² epz_i + 1,    Kx Ky epz_i  ]]
```

임의의 텐서와 임의의 입사각에 대해 그렇습니다. 수정 전에는 두 블록이 쌍을
같은 순서로 담았으므로, 비대칭 텐서는 고유값을 한 매질에서, 자기장을 그 매질의 전치에서
얻었습니다.

자이로트로픽 슬래브(`eps = [[eo, i·g], [-i·g, eo]]`, g = 0.45)에서 원편광 고유 상태로 얻은 답과
비교해 측정했습니다 — 각 원편광 성분은
`n± = sqrt(eo ± g)`인 독립적인 Fabry–Pérot입니다.

| | 투과 Jones 행렬의 최대 절대 오차 |
|---|---|
| 수정 전 | 3.6e-2 — `g`에 대해 1차이므로 상쇄되지 않았습니다 |
| `Omega_R`에서 쌍을 맞바꾼 경우 | 6.5e-16 |

결과는 여전히 맞는 것처럼 *보입니다*. Jones 행렬은 대각에서 축퇴되고 비대각에서는 반대칭이며,
에너지는 1e-12까지 보존되고, 회전 방향도 맞습니다. 크기만
틀립니다.

이는 기준값의 시간 규약이 만든 인공물이 아닙니다. 해석적인 쪽의 손잡이 방향이나 `e^{-iwt}` 부호
오류는 `g -> -g`로 이름만 바꿀 수 있으므로, `g`를 연속적으로 훑으면 결론이 납니다.
배포된 출력이 *어디서든* 가장 잘 맞는 값은 3.18e-2에서 `g = +0.4264`이고 — 어떤 `g`에 대해서도
정확한 자이로트로픽 답이 아닙니다 — 맞바꾼 버전은 6.47e-16에서 `g = +0.45000`에 정확히 놓입니다.
위의 `Omega_L` 논증은 아예 규약과 무관합니다. 물리가 전혀 들어 있지 않은, 코드의
대수적 항등식이기 때문입니다.

여전히 열려 있으며, 이 중 어느 것으로도 *결론 나지 않은* 문제가 있습니다. 이 저장소의 `e^{-iwt}`,
`Im(n) > 0` 규약에서 `[[eo, i·g], [-i·g,
eo]]`가 어떤 자화 방향을 나타내느냐는 것입니다. 검증된 것은 솔버가 주어진 텐서를 재현한다는 점이지,
그 텐서가 물리적으로 무엇을 뜻하는지가 아닙니다.

**발견 당시에는 배포된 어떤 것도 여기에 닿을 수 없었습니다.** `_nvf_assemble`은 `-eps_xy`를
*두 번* 반환했고, 그 옆의 두 경로도 마찬가지였습니다 — `boundary.py`, 그리고 당시 `analytic.py`에서
닫기 과정의 자체 사본을 가졌던 `nvf_from_coeffs`(지금은 `parametric.py`이며
`_nvf_assemble`에 위임합니다)입니다. 그래서 라이브러리가 만든 모든 텐서는 대칭이었고, 대칭인 쌍에서는
맞바꿈이 아무 효과가 없습니다 — 비트 단위로 같습니다. 이 결함은 `solve_conv`를 통해 손으로만
닿을 수 있었는데, 자이로트로피 실험이 맨 먼저 하는 일이 바로 그것입니다. 그 뒤에 추가된 두 경로,
`nvf_projector='circular'`와 텐서 재료는 실제로 비대칭 쌍을 만들며, 아래의 수정에 의존합니다.

**수정되었습니다.** `Omega_R`은 이제 `epxy`를 받던 곳에서 `epyx`를, 그 반대도 마찬가지로 받습니다.
모든 대칭 텐서는 어느 쪽이든 비트 단위로 같으므로, 라이브러리가 만드는 것 중 바뀐 것은 없습니다.
`test_uniform_anisotropic.py`, `test_anisotropic_berreman.py`(Berreman 4x4 오라클),
`tests/references/test_reticolo_anisotropic_golden.py`(RETICOLO, 고정된 골든 값)가 이를 고정합니다.

## 7. 구성 방식 고르기

균일한 재료 지정은 정확한 블록을 씁니다. 패턴이 있는 재료 지정은
기하와 선택한 옵션에 따라 카르테시안 Li 또는 선형 NVF를 씁니다.
epsilon이나 mu에 z 결합이 있으면 완전한 1차 솔버를 자동으로 고릅니다.
입력 형식은 [재료 가이드](../common/materials.md)에 설명되어 있고, 지원하는 기하, 검증,
수렴 한계는 [패턴 층 안의 전체 텐서
재료](full-tensor-patterns.md)에 있습니다.

## 8. 솔버가 고르는 방식

위의 모든 내용은 무엇이 빠져 있는지를 말합니다. 이 절은 *실제로* 존재하는 조각들이 어떻게 선택되는지를
말합니다. 그 결정은 사용자가 내릴 일이 아니기 때문입니다. 이 절은 분기 구현이 존재하기 전에
설계로 작성되었습니다. 원칙, 비용, 재정의, 요건이라는 그 순서를 그대로 두고, 각 부분에
2026-08-13에 배포된 내용을 적어 두었습니다. 앞으로의 축(`mu`)에 필요한 것은 그 추론이며,
코드만으로는 그것이 전달되지 않기 때문입니다.

**원칙은 이미 코드베이스에 있습니다.** `engine.py::_assign_grating_type`은 `structure`의
모양, 입사의 선언(`tetm`, 또는 편광 단어 `'s'` / `'p'`), `phi`와 `ky`를 읽고,
문제를 표현할 수 있는 가장 저렴한 정식화를 고릅니다 — 그
docstring은 *"Pick the formulation this scene needs: pure 1D TE/TM, 1D conical, or 2D."*로 시작합니다.
1D TE/TM은 스칼라로 나뉘고, 1D conical은 나뉘지 않으며, 2D는 모든 것을 결합합니다. 이방성은
새로운 종류의 결정을 더하는 것이 아니라 같은 결정에 축을 더합니다.

| 축 | 읽는 곳 | 분기 |
|---|---|---|
| 격자 차원 *(존재)* | `structure` 모양, `tetm` 또는 편광 단어, `phi`, `ky` | 1D TE/TM, 1D conical, 2D |
| **텐서의 z 행** *(존재)* | z 행 슬롯 네 개가 주어졌는지 여부 | 2차, 1차 |
| **투자율** | `mu != 1` | 기존 연산자, 자성 연산자 |
| **패턴 또는 균일** | 층에 면내 구조가 있는지 | 인수분해, 또는 항등 행렬의 상수배 |

z 행 행에 관해서는 한 가지가 스케치와 다르게 나왔습니다. 이 축은 슬롯이 0이 아닌지가 아니라
슬롯이 **주어졌는지**를 읽습니다. 분기는 제어 흐름을 정하므로, 읽을 것이 아무것도 없는 JAX 트레이서에서도
답할 수 있어야 합니다. 그리고 "모두 0인 행렬을 넘겼다"는 것은 텐서에 대한 주장이 아니라
호출자에 대한 주장입니다. 따라서 0 행렬 네 개를 넘기면 8x의 비용이 들고, 저렴한 경로는
인수 하나를 생략하면 됩니다.

z 행 축은 차원 축에도 영향을 미칩니다. 1D 연산자에는 z 행 슬롯이 없으므로,
z 행 — 유전율의 `ep{xz,yz,zx,zy}` 또는 투자율의 `mu[5:]` — 이 있으면 한 행짜리
래스터라도 1차 경로가 작성된 2D 배치로 옮겨지며,
`formulation='first_order'`는 z 행이 없어도 똑같이 합니다. 그 배치는 두 편광을 모두 담으므로,
그 둘 중 하나와 함께 쓴 `tetm`은 경로를 바꾸는 대신 이름을 들어 거부합니다. 바꿔 보낼 경로는
`tetm`이 없다고 선언한 입사 편광을 읽기 때문입니다. 2026-09-27까지는 경로를 바꿔 보냈습니다. 생성 시에
준 `tetm`은 풀이 깊숙한 곳에서 `TypeError`로 드러났고, 나중에 지정한 것은 둘 중 무엇을
가리키든 s광에 대한 답을 냈습니다. 오류 메시지는 같은 빛을 기술하는 쌍인
`phi = 0`에서의 `polarization=(1, 0)` / `(0, 1)`을 알려 주며,
`tests/rcwa/test_tetm_off_the_pure_path.py`가 실제로 그런지 확인합니다. 모두 0인 z 행과 함께,
그리고 강제한 `'first_order'`에서, 그 쌍은 순수 `tetm` 풀이를 재현합니다. `tetm`은
이전 인터페이스에만 속하므로, 현재 생애주기는 이 거부를 만나지 않습니다. 현재 생애주기는 대신
편광 단어(`PlaneWave(polarization='s')` / `'p'`, 2026-11-03부터)로 순수 경로에 닿습니다.
편광 단어는 편광이 없다고 선언하는 것이 아니라 고정되어 있다고 선언하므로 결코 거부되지 않습니다.
z 행과 함께, `'first_order'`에서, 기울어진 셀에서, 또는 방위각이 자유로운 경우, 편광 단어는
자신이 가리키는 쌍으로서 그 쌍을 담는 경로에서 풀립니다(`engine.py::_polarization_word_holds`).

### 왜 자동이어야 하는가

비용 차이는 기본값이 틀리면 실제 부담이 될 만큼 크고, 흔한 경우에는 아무도 그 비용을 치르는 것을
알아채지 못할 만큼 작습니다. RETICOLO에서 441개 차수로, 같은 크기의 등방성 층과 비교해
측정했습니다.

| 층 | 비용 |
|---|---|
| 대각 이방성 | 2.8x |
| 면내 텐서, 빈 z 행(Faraday) | 3.3x |
| **z 결합(Voigt)** | **9.5x** |

9.5x는 1차 형식으로의 도약입니다 — 고유값 문제가 두 배가 되고 `eig`는 세제곱입니다. Faraday
적층을 쓰는 사용자는 채워진 `eps_yz`가 여덟 배의 비용을 들게 한다는 것을 알 이유가 없고,
이 문서를 읽지 않고는 자신의 텐서에 그런 성분이 있는지 확인할 방법도 없습니다. `auto` 기본값이
있는 이유가 바로 그것입니다.

### 그래도 재정의가 중요한 이유

```python
class Formulation(str, Enum):
    AUTO         = 'auto'           # read the slots and decide           (default)
    SECOND_ORDER = 'second_order'   # force second-order; refuse z-coupling
    FIRST_ORDER  = 'first_order'    # force first-order; always possible, ~8x
```

이것이 `wavebit.rcwa._engine.enums.Formulation`이며,
`rcwa.Numerics(formulation=...)`로 씁니다. 이전 인터페이스에서는 `solver.formulation`입니다 —
생성자 키워드가 아니라 속성인 이유는, JAX 솔버가 pytree 자식들을 위치 인수로 넘기므로
새 키워드가 생기면 그 위치가 밀리기 때문입니다.

필요하지 않은 문제에 `'first_order'`를 강제하는 것이 1차 솔버를 **검증하는** 방법입니다.
z 결합이 없는 경우를 두 방식으로 계산하면 일치해야 하며, 이는 어떤 외부 오라클도 제공하지 않는
내부 교차 확인입니다. 그것이 `tests/rcwa/test_formulation_dispatch.py`이며, 이를 담당하는 경우는
conical 입사의 *패턴* 격자입니다 — 이때 두 정식화는 완전히 다른 행렬의 고유값을 구하면서도
4e-14까지 일치하는데, 균일 층의 상수배 항등 행렬로는 이를 보여 줄 수 없었을 것입니다. `layer_algorithm`을 선택할 수 있는 것도 같은 이유이며,
`test_anisotropic_berreman.py`가 ETM뿐 아니라 SMM도 구동하는 것도 같은 이유입니다. "수정이 다른
경로에도 공짜로 닿는다"는 것이야말로 틀린 것으로 드러나곤 하는 종류의 주장이기 때문입니다.

z 결합 텐서에 `'second_order'`를 강제하면 z 행을 조용히 버리는 것이 아니라 **오류를 내야** 합니다.
그 실패 양상은 이미 측정되어 있습니다. z 행을 버리면 효율이 2e-2에서 6e-2만큼 움직이지만 `R + T`는
기계 정밀도까지 1로 남으므로, 호출자가 돌릴 법한 모든 확인이 여전히 통과합니다. 이 방어 장치는
예전에 `test_anisotropic_berreman.py`의 텐서 어댑터에 있었습니다. 지금은 솔버에 있으며,
솔버가 텐서를 직접 받을 수 있게 되면 그곳에 있어야 한다고 이 문단이 말했던 자리입니다.

### 분기 자체에 대한 한 가지 요건

**어느 경로가 실행되었는지가 결과에 드러나야 합니다.** `grating_type`은 이미 정확히 이 이유로
노출되어 있습니다. 소리 없는 8x는 느린 기계와 구별할 수 없으며, 어느 정식화를 측정했는지
말할 수 없는 벤치마크는 벤치마크가 아닙니다. 해당 속성은
`solver.formulation_assigned`이며, 값은 `'second_order'` 또는 `'first_order'`이고, 모든 2D 풀이가 설정하며
첫 풀이 전에는 `None`입니다. 둘 다 이전 인터페이스 솔버의 속성입니다. 현재
`rcwa.Result`는 각 파장의 풀이가 택한 경로를 `diagnostics['solve_path']`
(`'1d_te'`, `'1d_tm'`, `'1d_conical'` 또는 `'2d'`)로 보고하고, check 보고서는 계산이 택할
경로를 `resolved['solve_path']`로 알려 줍니다. 2D 풀이가 어느 정식화를 배정했는지는 아직 보고하지 않습니다.
`resolved['formulation']`은 배정된 값이 아니라 요청한 값(기본값 `'auto'`)입니다.

## 참고 문헌

- L. Li, *Fourier modal method for crossed anisotropic gratings with arbitrary permittivity and
  permeability tensors*, J. Opt. A: Pure Appl. Opt. **5**(4), 345 (2003).
  [doi:10.1088/1464-4258/5/4/307](https://doi.org/10.1088/1464-4258/5/4/307) — §5와 §6을 한 논문에서
  함께 다루는 1차 정식화.
- E. Popov and M. Nevière, *Maxwell equations in Fourier space: fast-converging formulation for
  diffraction by arbitrary shaped, periodic, anisotropic media*, J. Opt. Soc. Am. A **18**(11),
  2886 (2001). [doi:10.1364/JOSAA.18.002886](https://doi.org/10.1364/JOSAA.18.002886) — §3에서 설명한
  재배열.
- T. Schuster, J. Ruoff, N. Kerwien, S. Rafler and W. Osten, *Normal vector method for convergence
  improvement using the RCWA for crossed gratings*, J. Opt. Soc. Am. A **24**(9), 2880 (2007).
  [doi:10.1364/JOSAA.24.002880](https://doi.org/10.1364/JOSAA.24.002880) — 이 저장소가 구현하는
  등방성 NVF.
- P. Götz, T. Schuster, K. Frenner, S. Rafler and W. Osten, *Normal vector method for the RCWA with
  automated vector field generation*, Opt. Express **16**(22), 17295 (2008).
  [doi:10.1364/OE.16.017295](https://doi.org/10.1364/OE.16.017295) — 필드를 자동으로 생성하는
  후속 연구. 위의 논문과는 별개의 논문입니다.
