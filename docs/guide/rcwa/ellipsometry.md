# Ellipsometry outputs: Jones, Psi/Delta and Mueller

An ellipsometer measures the ratio of the p- and s-polarized reflection coefficients, `rho = r_p / r_s = tan(Psi) exp(i Delta)`, and a Mueller-matrix instrument the 4x4 intensity transfer matrix. `rcwa.run_ellipsometry` returns both for any RCWA problem, from one preparation and two solves:

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

angles = result.reflection.ellipsometry()        # zeroth order, every wavelength
psi, delta = np.degrees(angles.psi), np.degrees(angles.delta)
jones = result.reflection.jones()                # (wavelength, 2, 2)
mueller = result.reflection.mueller()            # (wavelength, 4, 4), normalized by m_11
```

The problem's single `PlaneWave` is prepared once and run twice, lit with `polarization=(1, 0)` and `(0, 1)`; the `polarization` word it carries is only a placeholder, and a Jones pair is refused rather than discarded. `result.s` and `result.p` are the two plain `rcwa.Result`s, with everything a single solve has. `result.reflection` and `result.transmission` hold those results' diffraction orders side by side (`.s`, `.p`) and add the three combined outputs, for the zeroth order by default, for one order by its labels (`jones(order=(1, 0))`) or for every order at once (`jones(order=None)`, with the `(order_v, order_u)` axes of `DiffractionOrders`). The same functions are available on any Jones array in `wavebit.ellipsometry`, on NumPy, JAX and PyTorch alike, and differentiate wherever the solve does (eagerly under `jax.grad` and PyTorch autograd; `run_ellipsometry` prepares inside, so under `jax.jit` prepare the two problems outside the transform as the [lifecycle guide](../common/lifecycle.md) shows).

[`examples/rcwa/ellipsometry_film.py`](https://github.com/fieldalgo/wavebit/blob/main/examples/rcwa/ellipsometry_film.py) is the complete SiO2-on-silicon spectrum, checked against an Airy formula.

## The conventions, in one table

The numbers an ellipsometer reports depend on three choices that textbooks and instrument software make differently. Wavebit's are fixed as follows and pinned by `tests/rcwa/test_ellipsometry.py`; `result.reflection.reference` repeats them beside the data.

| Choice | Wavebit | Where it shows |
| --- | --- | --- |
| Jones basis and index | `(s, p)`, as `PlaneWave(polarization=(a_s, a_p))`; `J[out, in]`, so `j_ps = J[1, 0]` is the p amplitude for s incidence | `jones()` |
| Reflected `p_hat` | In-plane part `+cos(theta) k_hat_par` on the reflected beam, `-cos(theta) k_hat_par` on the incident and transmitted ones ([conventions](../../reference/common/conventions.md)). A single interface at normal incidence has `r_p = -r_s`: the **Nebraska convention** of ellipsometry (Muller 1969; Azzam and Bashara 1977), so an isotropic sample has `Delta = 180 deg` at normal incidence, and transmission `Delta = 0` | `jones()`, `delta` |
| Time dependence | Amplitudes are `exp(-i omega t)` phasors, `Im(n) > 0` is loss. Ellipsometers write `exp(+i omega t)`, where every amplitude is the complex conjugate | `rho` |
| `delta` | **Instrument convention**: `tan(psi) exp(i delta) = conj(j_pp / j_ss)`, `delta` in `[0, 2 pi)`, `psi` in `[0, pi/2]`, radians. An absorbing substrate gives `0 < delta < pi`, falling from `pi` as the angle grows, as in a measurement. `rho = j_pp / j_ss` keeps the native phasor, whose argument is `-delta` | `ellipsometry()` |
| Generalized ratios | `psi_ps`, `delta_ps` from `conj(j_sp / j_pp)` (p in, s out) and `psi_sp`, `delta_sp` from `conj(j_ps / j_ss)` (s in, p out), after Schubert (1996); zero on an isotropic stack | `ellipsometry()` |
| Mueller basis | Stokes `(I_p + I_s, I_p - I_s, I_+45 - I_-45, I_R - I_L)`, p first; `M = A (J (x) J*) A^-1`, normalized by `m_11` unless `normalize=False`. An isotropic sample gives `[[1, -N, 0, 0], [-N, 1, 0, 0], [0, 0, C, S], [0, 0, -S, C]]` with `N = cos 2psi`, `C = sin 2psi cos delta`, `S = sin 2psi sin delta` | `mueller()` |

Two consequences of the frame are worth knowing before comparing with an instrument. From the **bottom** side the `(s, p, k)` triad has the opposite handedness (the conventions page says why), so `j_pp / j_ss` -- and `psi`, `delta` -- are unchanged but the cross terms and the off-diagonal blocks of the Mueller matrix change sign against an instrument's; nothing is re-signed here. And on the `ky = 0` row an order with `kx < 0` is quoted in the reversed frame, a sign on the whole of `J` that no ratio and no Mueller entry sees.

## Matching an instrument's sign convention

- **`Delta` has the other sign.** The software reports `exp(-i omega t)` angles, or the sample is lit from the bottom and the cross terms are being compared. Use `-delta mod 2 pi`, or `np.angle(angles.rho)` for the native argument.
- **`Delta` is `0` instead of `180 deg` for a bare dielectric at normal incidence.** The software uses the other reflected `p_hat` (sometimes called the Verdet convention): `r_p` and `j_ps`, `j_sp` have the opposite sign there. Negate the second row of `J` before comparing Jones matrices; `psi` is unaffected.
- **`Delta` is reported in `(-180, 180]`.** Shift: `delta - 2 pi` where `delta > pi`.
- **`Delta` jumps by `360 deg` along a spectrum.** It is the branch of a continuous phase; both ranges wrap somewhere. Compare `exp(i delta)`, or unwrap with `np.unwrap`.
- **Mueller elements differ in sign in the last row or column.** The instrument's `S_3` is `I_L - I_R`, or its Jones basis is `(s, p)` with s first. Conjugate by `diag(1, 1, 1, -1)` for the former, by `diag(1, -1, -1, 1)` for the latter.

## What the Mueller matrix of a solve can and cannot be

A matrix built from a Jones matrix is non-depolarizing: `wavebit.ellipsometry.depolarization_index` is 1 (Gil and Bernabeu 1986) and `wavebit.ellipsometry.cloude_eigenvalues` has one nonzero entry (Cloude 1986). A measured matrix with a smaller index is an intensity average over conditions the solve holds fixed: the back-surface reflection of a thick substrate ([incoherent substrate](../../learn/fundamentals/incoherent-substrate.md)), the spectrometer's bandwidth, a focused beam's angular spread, thickness nonuniformity under the spot. To model one, average `mueller(normalize=False)` over those conditions -- never the Jones matrices, which would add the interference the measurement averages out -- and normalize the sum. Depolarization itself is outside this feature.

Two numerical notes. At a lossless stack's Brewster angle `|j_pp|` is exactly zero: `psi` is zero, its derivative with respect to the angle is singular, and `delta` steps by `pi`. And `delta` of a lossless stack is exactly `0` or `pi`, where rounding may put the reported value on either side of `0 = 2 pi`: a test should compare `exp(i delta)`.

## References

- R. H. Muller, "Definitions and conventions in ellipsometry," *Surf. Sci.* **16**, 14 (1969).
- R. M. A. Azzam and N. M. Bashara, *Ellipsometry and Polarized Light* (North-Holland, 1977).
- H. Fujiwara, *Spectroscopic Ellipsometry: Principles and Applications* (Wiley, 2007).
- M. Schubert, "Polarization-dependent optical parameters of arbitrarily anisotropic homogeneous layered systems," *Phys. Rev. B* **53**, 4265 (1996).
- J. J. Gil and E. Bernabeu, "Depolarization and polarization indices of an optical system," *Optica Acta* **33**, 185 (1986).
- S. R. Cloude, "Group theory and polarisation algebra," *Optik* **75**, 26 (1986).
- J. J. Gil and R. Ossikovski, *Polarized Light and the Mueller Matrix Approach* (CRC Press, 2016).
