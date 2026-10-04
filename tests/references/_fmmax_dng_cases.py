"""Cases shared by ``gen_goldens_fmmax_dng.py`` and ``test_fmmax_dng_golden.py``.

Same split as the other fmmax pairs: the generator runs in ``fwref`` (fmmax, no wavebit) and the
test in the project env (the reverse), so what they share imports numpy and nothing else.

**What this oracle is for.** Gratings resting on, containing, or lit from a **lossy
double-negative** medium (``Re(eps) < 0`` and ``Re(mu) < 0``): the half-space root, the signed
index and the layer labels this solver chose for such media (``docs/branch-cuts.md`` §9),
held to a solver that made none of those choices. fmmax keeps each layer's modes with
``Im(kz) > 0`` (decaying along its forward direction), which is the physical choice for **every
lossy medium, of either sign of index** -- the limiting-absorption argument that picks the root
needs no sign of ``Re(n)`` once there is loss. In a *lossless* double negative that rule has no
loss to read and picks by round-off, so every negative-index medium here is lossy, and every
one has ``|mu| != 1`` so that a solver confusing ``eps`` with ``eps*mu`` cannot pass.

**Identity, not convergence.** Wavebit is run with ``factorization='laurent'``, which assembles
``[[eps]]`` and ``[[mu]]`` in the in-plane slots and their inverses in the z slots -- what fmmax's
``Formulation.FFT`` builds for both tensors. The one remaining difference is fmmax's midpoint
rule for the pixel integral, removed by extrapolating over the samples per pixel exactly as
``_fmmax_laurent_cases.py`` derives it (``richardson`` and ``upsampled`` are imported from
there), plus fmmax 1.7.1's single-precision centring phase, removed by the generator's second
run.

**What is compared is basis-free.** A lossy negative-index medium's two order-0 modes come out
of a general eigensolver in whatever mix of polarizations it returns, so neither code's ``s`` and
``p`` can be matched one to one. Both sides instead report the **unpolarized efficiency** of
every order: the mean over two incident polarizations that are orthonormal in incident flux,
which is ``trace(H_in^-1 H_out) / 2`` for the Hermitian flux forms of the inputs and outputs and
does not depend on which two were chosen. Wavebit's ``s`` and ``p`` are flux-orthogonal in an
isotropic medium of any loss (their ``E x H*`` cross terms have no ``z`` component), so there it
is ``(R_s + R_p) / 2`` order by order.

**The incidence is stated as an in-plane wavevector**, ``(kx, ky) / k0``, which both codes take
as given. For the rows lit from glass it comes from ``kx = n sin(theta) cos(phi)``; for the row
lit **from** the double negative it is computed below from ``theta`` and ``phi`` with the rule
fixed for a negative-index lit side in 2026-10 (``engine.CoreSolveMixin._incident_kxy``): the
angle names the direction the light travels, so ``k_par = s*N*sin(theta)``, ``N`` the lit
medium's phase index and ``s = -1`` the sign of its index. The golden holds the wavevector, not
the rule; the test asks Wavebit's ``theta`` door separately for the same answer.

The raster convention is this project's ``[v, u]`` (``[y, x]``); the generator transposes for
fmmax. Nothing below is square or centred, and the incidence is oblique in both angles, so a
transposed, mirrored or order-flipped layer changes the answer.
"""
import numpy as np

WL = 500.0
THICK = 120.0
LEVELS = (16, 32, 64, 128)   # fmmax samples per Wavebit pixel, per axis

GLASS = (2.25 + 0j, 1.0 + 0j)
#: Lossy double negatives, ``(eps, mu)``. Each has ``|mu| != 1``.
DNG_A = (-3.0 + 0.1j, -0.5 + 0.05j)       # n = sqrt(eps)*sqrt(mu) = -1.226 + 0.082j
DNG_B = (-6.0 + 0.3j, -1.5 + 0.05j)       # n = -3.002 + 0.125j

_G1 = np.array([[2.25, 4.0, 4.0, 1.0, 3.0, 1.0, 1.0, 2.0]], dtype=complex)        # [1, x]
_G2 = np.array([[2.25, 4.0, 1.0, 1.0], [4.0, 4.0, 1.0, 3.0],
                [1.0, 2.0, 2.0, 1.0], [1.0, 1.0, 3.0, 2.25]], dtype=complex)      # [y, x]
#: Where the third case puts double-negative pixels into ``_G2``: off-centre, not symmetric.
_MASK = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]], dtype=bool)


#: name -> the scene. ``period`` is ``(Px,)`` for a 1D grating (rows uniform along y) and
#: ``(Px, Py)`` for a 2D one; ``order`` is ``truncation_order`` per axis; ``eps`` / ``mu`` are the
#: layer rasters ``[y, x]`` (``mu=None``: a non-magnetic layer, solved as one by both codes);
#: ``top`` lights the stack, ``bot`` receives it.
CASES = {
    # A 1D dielectric grating on a double-negative substrate, conical incidence.
    'grating1d_on_dng': dict(period=(1000.0,), order=(4, 0), eps=_G1, mu=None,
                             top=GLASS, bot=DNG_A, theta=0.3, phi=0.6),
    # A 2D dielectric grating on a stronger double negative, |mu| = 1.5.
    'grating2d_on_dng': dict(period=(1000.0, 900.0), order=(2, 2), eps=_G2, mu=None,
                             top=GLASS, bot=DNG_B, theta=0.55, phi=2.4),
    # Double-negative pixels inside the grating, on a double-negative substrate: patterned
    # eps *and* mu, the layer whose modes this solver labels by their power flux.
    'dng_pattern_on_dng': dict(period=(1000.0, 900.0), order=(2, 2),
                               eps=np.where(_MASK, DNG_A[0], _G2),
                               mu=np.where(_MASK, DNG_A[1], 1.0 + 0j),
                               top=GLASS, bot=DNG_A, theta=0.3, phi=0.6),
    # Lit FROM the double negative, through a 1D grating into glass.
    'lit_from_dng_1d': dict(period=(1000.0,), order=(4, 0), eps=_G1, mu=None,
                            top=DNG_A, bot=GLASS, theta=0.35, phi=0.6),
}


def phase_index(eps_mu, theta):
    """``N = |Re(k)|/k0`` of a plane wave of real angle ``theta`` with real ``k_par``:
    ``N**2 = (a + hypot(a, b/cos(theta)))/2`` for ``eps*mu = a + ib`` (``docs/conventions.md``)."""
    a, b = eps_mu.real, eps_mu.imag
    return np.sqrt((a + np.hypot(a, b / np.cos(theta))) / 2)


def incident_kxy(name):
    """``(kx, ky) / k0`` of the case's incident wave, written out independently of the solver.

    ``s`` is the sign of ``Re(sqrt(eps)*sqrt(mu))`` with principal roots: ``+1`` for glass,
    ``-1`` for a double negative, whose phase advances against its power.
    """
    case = CASES[name]
    eps, mu = (complex(v) for v in case['top'])
    s = -1.0 if (np.sqrt(eps) * np.sqrt(mu)).real < 0 else 1.0
    radial = s * phase_index(eps * mu, case['theta']) * np.sin(case['theta'])
    return radial * np.cos(case['phi']), radial * np.sin(case['phi'])


def orders(name):
    """``(m, n)`` of every retained order, the zeroth first (the incident amplitude's index)."""
    mx, my = CASES[name]['order']
    pairs = [(m, n) for m in range(-mx, mx + 1) for n in range(-my, my + 1)]
    return sorted(pairs, key=lambda mn: (mn != (0, 0), abs(mn[0]) + abs(mn[1]), mn))


def lossless(name):
    """No case here is: every negative-index medium is lossy on purpose (see the module text)."""
    return False
