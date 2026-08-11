import numpy as np
import pytest

import fieldwave


def test_nvf_circular_grating():
    print("==================================================")
    print("Test: NVF (Normal Vector Field) Circular Grating")
    print("==================================================")

    # 2D Circular Cylinder Grating (n_cylinder=2.0, n_bg=1.0)
    period = [1000, 1000]
    wavelength = 600.0

    grid_y, grid_x = np.ogrid[:100, :100]
    center_y, center_x = 50, 50
    radius = 30
    dist_from_center = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)

    ucell_2d = np.ones((1, 100, 100), dtype=complex) * 1.0
    ucell_2d[0, dist_from_center <= radius] = 2.0  # Cylinder with n=2.0

    option_nvf = dict(
        pol=1, n_top=1.0, n_bot=1.5, theta=10 * np.pi / 180, phi=20 * np.pi / 180,
        fto=[3, 3], wavelength=wavelength, period=period,
        ucell=ucell_2d, thickness=[200], factorization='nvf'
    )

    mee_nvf = fieldwave.rcwa(backend=0, **option_nvf)
    res_nvf = mee_nvf.conv_solve()

    assert res_nvf.R is not None and res_nvf.T is not None
    R_nvf = float(res_nvf.R.sum())
    T_nvf = float(res_nvf.T.sum())

    print(f"[NVF] Reflectance R = {R_nvf:.6f}, Transmittance T = {T_nvf:.6f}, R+T = {R_nvf + T_nvf:.6f}")

    assert 0.0 <= R_nvf <= 1.0, f"Unphysical reflectance R = {R_nvf}"
    assert abs(R_nvf + T_nvf - 1.0) < 1e-3, f"Energy conservation failed in NVF: R+T = {R_nvf + T_nvf}"

    print("SUCCESS: NVF Circular Grating Test Passed!\n")


def test_nvf_energy_convergence():
    """Lossless circular rod at normal incidence: |R+T-1| must ->0 as the Fourier
    order rises. This is the structural correctness check for the NVF anisotropic
    factorization; the pre-fix implementation violated energy conservation by
    O(0.1-0.5) here (wrong epz convention, non-Hermitian projector product)."""
    print("==================================================")
    print("Test: NVF energy conservation / convergence")
    print("==================================================")

    N = 120
    x = (np.arange(N) + 0.5) / N - 0.5
    X, Y = np.meshgrid(x, x)
    ucell = np.ones((1, N, N), dtype=complex)
    ucell[0, X ** 2 + Y ** 2 <= 0.3 ** 2] = 2.5  # lossless dielectric rod

    cfg = dict(n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, pol=0.0,
               wavelength=0.5, period=[1.0, 1.0], thickness=[0.3])

    errs = {}
    for fto in (3, 5, 7):
        res = fieldwave.rcwa(backend=0, ucell=ucell, fto=[fto, fto],
                              factorization='nvf', **cfg).conv_solve()
        R = float(np.array(res.R).real.sum())
        T = float(np.array(res.T).real.sum())
        errs[fto] = abs(R + T - 1.0)
        print(f"[NVF] fto={fto:2d}  |R+T-1| = {errs[fto]:.2e}")

    # Tight, order-independent bound the buggy implementation could never meet.
    assert errs[3] < 1e-6, f"NVF energy error too large at fto=3: {errs[3]}"
    assert errs[7] < 1e-5, f"NVF energy error too large at fto=7: {errs[7]}"

    print("SUCCESS: NVF energy conservation test passed!\n")


if __name__ == "__main__":
    test_nvf_circular_grating()
    test_nvf_energy_convergence()


@pytest.mark.parametrize('kind', ['te', 'tm', 'conical'])
def test_nvf_on_a_one_cell_axis_reduces_to_li(kind):
    """A 1D grating has no interface along y, so NVF must *be* Li there — and must run.

    ``np.gradient`` needs three cells, so the distance-transform normal field used to raise on
    a ``(layers, 1, N)`` ucell and NVF was simply unusable on 1D. The fix states the normal
    directly: with a one-cell axis every normal points along the other one, and a constant
    normal collapses the NVF projector onto Li's rules (``C_xx = I``, ``C_xy = C_yy = 0``).

    So this asserts equality, not closeness — anything else means the collapse is not exact.
    """
    ucell = np.array([[[3., 3., 1., 1., 3., 1., 1., 1.]]])
    cfg = dict(backend=0, ucell=ucell, period=[0.77], thickness=[0.2], theta=0.25,
               wavelength=0.777, n_top=1.0, n_bot=1.5, fto=[10, 0])
    cfg |= dict(phi=0.3, psi=0.4) if kind == 'conical' else dict(pol=0 if kind == 'te' else 1)

    def total(**kw):
        res = fieldwave.rcwa(**kw, **cfg).conv_solve()
        return float(np.sum(np.asarray(res.R)))

    assert total(factorization='nvf') == pytest.approx(total(factorization='li'), abs=1e-12)
    # ...and the collapse is a property of the projector, so it holds whichever way the
    # coefficients are obtained -- including the pairing the old integer could not spell.
    assert (total(fourier_coefficients='integrated', factorization='nvf')
            == pytest.approx(total(fourier_coefficients='integrated'), abs=1e-12))
