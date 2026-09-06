"""Crossed-grating scenes shared by ``gen_goldens_meep_crossed.py`` and
``test_meep_crossed_golden.py``.

They live here rather than in the test module for the reason the fmmax pair gives: the
two run in *different environments*. Meep is GPL and brings its own compiled stack, so
the generator runs in a separate ``meep`` env that has no ``wavebit`` in it, and the
test runs in the project env that has no ``meep``. Neither can import the other's
module. This one imports numpy and nothing else, so both can read it.

**Lengths are micrometres**, which is a deliberate departure from the nanometres
``tests/test_fdtd.py`` uses. Both solvers are scale-free and the FDTD module has had
exactly one defect that was not — a bare ``0.05`` in the CPML that was a reciprocal
*length* (``docs/fdtd.md`` §3) — so stating this oracle in the other unit costs nothing
and is the one thing that would notice such a constant coming back.

**Materials are stated as a refractive index**, not as a permittivity. Meep's
``mp.Medium`` takes ``index=`` and wavebit takes ``eps``; a case file that stated one of
the two would have to be squared or rooted on one side, and the direction is easy to get
wrong silently. ``eps_of`` below is the one conversion, read by the test.

The two cells:

* ``square_18deg`` — a 0.84 x 0.84 cell. Three propagating orders above and eight below.
* ``rect_28deg`` — 0.80 x 1.20, a **rectangular** cell with propagating ``l != 0``
  orders in *both* half-spaces. That pair of conditions — ``Px != Py`` and a radiating y
  order — is what a y ladder written on the x period survives, since ``l*wl/Px`` is
  ``l*wl/Py`` on a square cell.

Both rasters are asymmetric in x, asymmetric in y, **and** asymmetric under transpose.
The last one is not decoration: every external oracle this project has used was once
blind to a y row-order defect because the 2D rasters it was handed were y-symmetric
(``tests/references/README.md``, the fmmax ``patch`` case).
"""
import numpy as np


def eps_of(index):
    """The one index -> permittivity conversion. Meep takes the index, wavebit the
    permittivity; a case file that stated only one of the two invites a stray square."""
    return np.asarray(index, dtype=float) ** 2


def _raster(n_y, n_x, blocks, background=1.0):
    """``(1, n_y, n_x)`` refractive index. ``blocks`` are ``(j0, j1, i0, i1, index)``
    half-open in the array's own ``[y, x]`` order — v-first, the rule
    ``docs/conventions.md`` states for arrays."""
    a = np.full((n_y, n_x), float(background))
    for j0, j1, i0, i1, n in blocks:
        a[j0:j1, i0:i1] = float(n)
    return a[None, :, :]


#: name -> the scene. ``orders_u`` / ``orders_v`` are the propagating window both solvers
#: report on; they are written out rather than derived so that a change in either
#: solver's window is a *visible* diff against a frozen list rather than a quiet reshape.
#:
#: ``meep_*`` are the settings the golden was produced with, kept beside the scene
#: because a frozen number without the grid it was taken on is not reproducible.
CASES = {
    'square_18deg': dict(
        wavelength=0.80, period=(0.84, 0.84), thickness=(0.20,),
        n_top=1.0, n_bot=1.6, theta_deg=18.0, phi_deg=40.0,
        # 7 x 7. Three indices and air, in three disjoint rectangles: the footprint is
        # not a rectangle, not mirror-symmetric on either axis, and not its own
        # transpose. Three propagating orders above and eight below, the smallest
        # Rayleigh margin 0.38.
        index=_raster(7, 7, [(0, 3, 0, 4, 2.0), (3, 7, 0, 2, 1.7), (5, 7, 4, 7, 1.4)]),
        orders_u=(-1, 0, 1), orders_v=(-1, 0, 1),
        meep_resolution=50, meep_until=200.0, meep_dpml=0.6, meep_pad=0.7),
    'rect_28deg': dict(
        # 0.30 rather than 0.24: the generator refuses a thickness that is not an integer
        # number of pixels, because rounding one moves the plane the transmitted phase is
        # referenced to. At 0.24 and resolution 60 that was 14.4 pixels, and the specular
        # amplitude came out 4.6e-2 from the converged RCWA answer with R + T at 0.9954.
        wavelength=0.85, period=(0.80, 1.20), thickness=(0.30,),
        n_top=1.0, n_bot=1.45, theta_deg=28.0, phi_deg=35.0,
        # 12 rows x 8 columns -- the array is v-first, so this is Ny = 12 along the
        # LONG period and Nx = 8 along the short one. Four propagating orders above and
        # eight below, including l != 0 on both sides; smallest Rayleigh margin 0.33.
        index=_raster(12, 8, [(0, 5, 0, 5, 2.0), (5, 12, 0, 2, 1.7), (8, 12, 5, 8, 1.4)]),
        orders_u=(-1, 0, 1), orders_v=(-2, -1, 0, 1, 2),
        meep_resolution=60, meep_until=200.0, meep_dpml=0.6, meep_pad=0.7),
}

#: The three leg signs are the canonical frame rather than a convention, and both sides
#: derive them from the same two definitions -- ``s_hat = z_hat x k_hat_par`` and
#: ``p_hat = k_hat x s_hat`` -- so nothing here is fitted. Written down because the
#: generator and ``wavebit/_core/fdtd.py`` reach them independently:
#:
#:     E . p_hat = -+ (n / kz) (cos(varphi) Ex + sin(varphi) Ey)
#:
#: with the upper sign for a DOWNGOING wave. Incident -1, reflected +1, transmitted -1.
SIGMA = (-1.0, 1.0, -1.0)


def in_plane_k(case, m, l):
    """``(kx, ky)`` of order ``(m, l)``, normalised by ``k0``."""
    th, ph = np.deg2rad(case['theta_deg']), np.deg2rad(case['phi_deg'])
    px, py = case['period']
    wl = case['wavelength']
    return (case['n_top'] * np.sin(th) * np.cos(ph) + m * wl / px,
            case['n_top'] * np.sin(th) * np.sin(ph) + l * wl / py)


def margin(case, m, l, n):
    """``n**2 - |k_par|**2``: positive where order ``(m, l)`` propagates in index ``n``.

    Compared against a *margin* rather than against zero wherever amplitudes are: an
    order within round-off of a Rayleigh point converges as ``1/kz`` and the p channel
    multiplies by ``n/kz`` again, so both codes are slowest exactly there.
    """
    kx, ky = in_plane_k(case, m, l)
    return float(n ** 2 - (kx ** 2 + ky ** 2))
