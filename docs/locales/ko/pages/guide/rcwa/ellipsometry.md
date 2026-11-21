# 타원 계측 출력: Jones, Ψ/Δ, Mueller

엘립소미터(타원 계측기)는 p 편광과 s 편광 반사 계수의 비 `rho = r_p / r_s = tan(Psi) exp(i Delta)`를 재고, Mueller 행렬 장비는 4×4 세기 전달 행렬을 잽니다. `rcwa.run_ellipsometry`는 어떤 RCWA 문제에서든 준비 한 번과 풀이 두 번으로 둘 다 돌려줍니다.

```python
import numpy as np
from wavebit import Material, Units, materials, rcwa

problem = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(period=(300., 300.), bottom=materials.load('Si'),
                     layers=(rcwa.Layer(thickness=100., background=materials.load('SiO2')),)),
    excitation=rcwa.Excitation(wavelength=np.arange(400., 801., 10.),
                               sources=(rcwa.PlaneWave(theta=np.radians(70.), phi=0.),)))
result = rcwa.run_ellipsometry(problem, numerics=rcwa.Numerics(orders=(0, 0)))

angles = result.reflection.ellipsometry()        # 0차, 모든 파장
psi, delta = np.degrees(angles.psi), np.degrees(angles.delta)
jones = result.reflection.jones()                # (wavelength, 2, 2)
mueller = result.reflection.mueller()            # (wavelength, 4, 4), m_11로 정규화
```

문제의 `PlaneWave` 하나를 한 번 준비한 뒤 `polarization=(1, 0)`과 `(0, 1)`로 두 번 풉니다. 소스에 적힌 `polarization` 단어는 자리만 차지하고, Jones 쌍은 버리지 않고 오류로 거부합니다. `result.s`와 `result.p`는 두 번의 풀이가 내놓은 보통의 `rcwa.Result`이고, 단일 풀이가 가진 모든 것을 그대로 가집니다. `result.reflection`과 `result.transmission`은 그 두 결과의 회절 차수를 나란히(`.s`, `.p`) 들고 있으면서 결합 출력 세 가지를 더합니다. 기본은 0차이고, 차수 이름으로 하나를 고르거나(`jones(order=(1, 0))`), 모든 차수를 한 번에 받을 수 있습니다(`jones(order=None)`, `DiffractionOrders`와 같은 `(order_v, order_u)` 축). 같은 함수들은 `wavebit.ellipsometry`에서 어떤 Jones 배열에도 쓸 수 있고, NumPy·JAX·PyTorch에서 같이 돌며, 풀이가 미분되는 곳에서 미분됩니다(`jax.grad`와 PyTorch autograd는 즉시 실행으로; `run_ellipsometry`는 안에서 준비를 하므로 `jax.jit` 아래에서는 [실행 절차 안내](../common/lifecycle.md)처럼 두 문제를 변환 밖에서 준비합니다).

[`examples/rcwa/ellipsometry_film.py`](https://github.com/fieldalgo/wavebit/blob/main/examples/rcwa/ellipsometry_film.py)가 실리콘 위 SiO₂ 막의 완전한 스펙트럼 예제이고, Airy 공식과 대조합니다.

## 규약 한 표

엘립소미터가 보고하는 숫자는 교과서와 장비 소프트웨어마다 다르게 정하는 세 가지 선택에 달려 있습니다. Wavebit의 선택은 다음과 같이 고정되어 있고 `tests/rcwa/test_ellipsometry.py`가 고정합니다. `result.reflection.reference`가 데이터 옆에 같은 내용을 되풀이합니다.

| 선택 | Wavebit | 나타나는 곳 |
| --- | --- | --- |
| Jones 기저와 첨자 | `PlaneWave(polarization=(a_s, a_p))`와 같은 `(s, p)`; `J[out, in]`이라 `j_ps = J[1, 0]`은 s 입사의 p 진폭 | `jones()` |
| 반사파의 `p_hat` | 면내 성분이 반사 빔에서 `+cos(theta) k_hat_par`, 입사·투과 빔에서 `-cos(theta) k_hat_par`([규약](../../reference/common/conventions.md)). 수직 입사 단일 경계에서 `r_p = -r_s`: 타원 계측의 **Nebraska 규약**(Muller 1969; Azzam and Bashara 1977)이라, 등방성 시료는 수직 입사에서 `Delta = 180°`, 투과는 `Delta = 0` | `jones()`, `delta` |
| 시간 의존성 | 진폭은 `exp(-i omega t)` 페이저이고 손실은 `Im(n) > 0`. 엘립소미터는 `exp(+i omega t)`로 쓰므로 모든 진폭이 복소 켤레 | `rho` |
| `delta` | **장비 규약**: `tan(psi) exp(i delta) = conj(j_pp / j_ss)`, `delta`는 `[0, 2 pi)`, `psi`는 `[0, pi/2]`, 라디안. 흡수 기판은 측정값처럼 `0 < delta < pi`이고 각이 커질수록 `pi`에서 내려옴. `rho = j_pp / j_ss`는 고유 페이저 그대로이고 그 편각이 `-delta` | `ellipsometry()` |
| 일반화 비 | `conj(j_sp / j_pp)`(p 입사, s 출사)의 `psi_ps`, `delta_ps`와 `conj(j_ps / j_ss)`(s 입사, p 출사)의 `psi_sp`, `delta_sp`, Schubert (1996)를 따름; 등방성 스택에서 0 | `ellipsometry()` |
| Mueller 기저 | Stokes `(I_p + I_s, I_p - I_s, I_+45 - I_-45, I_R - I_L)`, p가 먼저; `M = A (J (x) J*) A^-1`, `normalize=False`가 아니면 `m_11`로 정규화. 등방성 시료는 `[[1, -N, 0, 0], [-N, 1, 0, 0], [0, 0, C, S], [0, 0, -S, C]]`, `N = cos 2psi`, `C = sin 2psi cos delta`, `S = sin 2psi sin delta` | `mueller()` |

장비와 비교하기 전에 알아둘 좌표계의 결과가 둘 있습니다. **아래쪽**에서 비추면 `(s, p, k)` 삼조의 손잡이가 반대라서(이유는 규약 페이지에 있습니다) `j_pp / j_ss`와 `psi`, `delta`는 그대로지만 교차항과 Mueller 행렬의 비대각 블록은 장비와 부호가 반대가 됩니다. 여기서는 어떤 부호도 고쳐 쓰지 않습니다. 그리고 `ky = 0` 행에서 `kx < 0`인 차수는 뒤집힌 좌표계로 보고되는데, 이는 `J` 전체의 부호 하나라 어떤 비도, 어떤 Mueller 원소도 보지 못합니다.

## 장비의 부호 규약에 맞추기

- **`Delta`의 부호가 반대다.** 소프트웨어가 `exp(-i omega t)` 각도를 보고하거나, 시료를 아래에서 비추면서 교차항을 비교하는 경우입니다. `-delta mod 2 pi`를 쓰거나, 고유 편각이 필요하면 `np.angle(angles.rho)`를 씁니다.
- **수직 입사의 맨 유전체에서 `Delta`가 `180°`가 아니라 `0`이다.** 소프트웨어가 다른 반사 `p_hat`(흔히 Verdet 규약)을 쓰는 경우로, 거기서는 `r_p`와 `j_ps`, `j_sp`의 부호가 반대입니다. Jones 행렬을 비교하기 전에 `J`의 둘째 행 부호를 바꾸면 됩니다. `psi`는 영향이 없습니다.
- **`Delta`가 `(-180, 180]`로 보고된다.** `delta > pi`인 곳에서 `delta - 2 pi`로 옮깁니다.
- **스펙트럼을 따라 `Delta`가 `360°` 뛴다.** 연속한 위상의 가지이고, 어느 범위를 골라도 어딘가에서 감깁니다. `exp(i delta)`를 비교하거나 `np.unwrap`으로 펼칩니다.
- **Mueller 원소의 마지막 행이나 열의 부호가 다르다.** 장비의 `S_3`가 `I_L - I_R`이거나, 장비의 Jones 기저가 s가 먼저인 `(s, p)`인 경우입니다. 앞의 경우는 `diag(1, 1, 1, -1)`로, 뒤의 경우는 `diag(1, -1, -1, 1)`로 닮음 변환합니다.

## 풀이의 Mueller 행렬이 될 수 있는 것과 없는 것

Jones 행렬에서 만든 행렬은 탈편광하지 않습니다. `wavebit.ellipsometry.depolarization_index`가 1이고(Gil and Bernabeu 1986) `wavebit.ellipsometry.cloude_eigenvalues`는 0이 아닌 값이 하나뿐입니다(Cloude 1986). 지수가 더 작은 측정 행렬은 풀이가 고정해 둔 조건에 대한 세기 평균입니다. 두꺼운 기판의 뒷면 반사([비간섭 기판](../../learn/fundamentals/incoherent-substrate.md)), 분광기의 대역폭, 집속 빔의 각 분산, 측정 스팟 안의 두께 불균일이 그것입니다. 이를 모델링하려면 그 조건들에 대해 `mueller(normalize=False)`를 평균한 뒤 정규화합니다. Jones 행렬을 평균하면 안 됩니다. 그러면 측정이 평균해 없애는 간섭을 더하게 됩니다. 탈편광 자체는 이 기능의 범위 밖입니다.

수치에 관한 주의 둘. 무손실 스택의 Brewster 각에서는 `|j_pp|`가 정확히 0이라 `psi`가 0이고, 각도에 대한 그 미분은 특이하며, `delta`는 `pi`만큼 계단을 이룹니다. 그리고 무손실 스택의 `delta`는 정확히 `0` 또는 `pi`인데, 반올림이 보고값을 `0 = 2 pi`의 어느 쪽에든 놓을 수 있으므로 시험은 `exp(i delta)`를 비교해야 합니다.

## 참고문헌

- R. H. Muller, "Definitions and conventions in ellipsometry," *Surf. Sci.* **16**, 14 (1969).
- R. M. A. Azzam and N. M. Bashara, *Ellipsometry and Polarized Light* (North-Holland, 1977).
- H. Fujiwara, *Spectroscopic Ellipsometry: Principles and Applications* (Wiley, 2007).
- M. Schubert, "Polarization-dependent optical parameters of arbitrarily anisotropic homogeneous layered systems," *Phys. Rev. B* **53**, 4265 (1996).
- J. J. Gil and E. Bernabeu, "Depolarization and polarization indices of an optical system," *Optica Acta* **33**, 185 (1986).
- S. R. Cloude, "Group theory and polarisation algebra," *Optik* **75**, 26 (1986).
- J. J. Gil and R. Ossikovski, *Polarized Light and the Mueller Matrix Approach* (CRC Press, 2016).
