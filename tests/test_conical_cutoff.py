"""A 1D conical order at a layer's x-cutoff solves exactly, and nothing off the cutoff moves.

**The point.** On the conical path (``ky != 0``) a layer's modes come in two families, one with
``E_x = 0`` normalised to ``H_x`` and one with ``H_x = 0`` normalised to ``E_x``. Where an order's
``kx**2`` meets the layer's permittivity (``wA = q**2 - ky**2 = 0``) both families tend to the one
field with ``E_x = H_x = 0`` -- the exception in L. Li's uniqueness proof (J. Mod. Opt. 40, 553,
1993, appendix A) -- so their columns grow like ``1/wA``, turn parallel, and the interface
inverses amplify rounding by ``1/wA**2``. It is *not* a Rayleigh point: the order's ``kz`` is
``|ky|``, not 0.

* In a **uniform** layer the point is an ordinary double root: only the basis collapses. The
  family construction refused where ``wA`` came out exactly zero and was silently wrong one ULP
  off (2.4e-3 .. 13 in ``R`` over eight scenes; over a census of round inputs SMM's ``R + T``
  reached 2.4e3); the ``auto_detune`` retry landed 2.6e-2 .. 0.1 off.
* In a **patterned** layer it is a genuine 2x2 Jordan block (the two roots cross ``|ky|`` with
  different slopes); ``wA`` is never exactly zero there, so nothing refused at all: ``R`` was off
  by up to 11, ``R + T`` by up to 48 (68 offsets within relative 1e-4).

:func:`~wavebit._core.solve_1d._regular_conical_modes` now rebuilds such a layer in a regular
basis with an upper-triangular propagator, gated by
:func:`~wavebit._core.solve_1d._cutoff_cluster`. What this file pins:

1. uniform scenes at the exact point, one ULP either side and ``+1e-12``, against the 2D path
   (a y-invariant raster, which never forms the families and is exact there);
2. the gate itself, on synthetic eigenvalues;
3. patterned exceptional points against an in-test polynomial fit of the family construction
   far enough off the point (``|rel| >= 1e-4``) to be accurate -- no golden file;
4. a wide cluster (both families routed) against the family construction at the same point;
5. the rebuilt layer's two first-order relations, and its divisor guards beside a zero root;
6. derivatives (``jax.grad``, ``jvp``, ``jacfwd``, torch reverse and forward mode), along the
   raster, the wavelength, the thickness and the angles (``theta``, ``phi``: they move ``ky``);
7. fields, against the 2D path and SMM against ETM, values and gradients, on one layer and two,
   lit from either side; and an eager record differentiated under ``jax.jit``;
8. inertness off the band, and under an abstract trace -- ``jax.jit`` and ``jax.vmap`` (the
   documented gap: no rebuild there);
9. the ``ky == 0`` Rayleigh contract, untouched;
11. complex64; 12. a cutoff and a near-graze in one layer; 13. ``excitation=`` and SMM's
    ``scattering``. (Item 10 lives in ``test_rayleigh_guard.py``, ``[1D ky]``.)

Sizes stay at ``truncation_order=5``. ``docs/rayleigh-points.md`` section 7 has the story and
the measurements behind every tolerance below.
"""
import contextlib
import functools
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core import modal_gradient, solve_1d

PERIOD_Y = 700.0          # the 2D oracle's y period; a y-invariant raster makes it immaterial
HALF_PI = np.pi / 2


def _pillar(n=20, lo=6, hi=14, eps=12.25):
    p = np.ones((1, 1, n))
    p[0, 0, lo:hi] = eps
    return p


def _stack(spacer, slices=1):
    """``slices`` uniform spacer layers of permittivity ``spacer`` over the pillar."""
    return np.concatenate([np.full((slices, 1, 20), spacer, dtype=float), _pillar()], axis=0)


#: Scene R1: an air spacer (300 nm) under a pillar grating (200 nm), wl 500, P 1000, phi pi/2,
#: so ``kx_m = m/2`` exactly and order 2 sits at ``kx = 1`` on the spacer.
R1 = dict(structure=_stack(1.0), wavelength=500., period=1000., thickness=[300., 200.],
          top=1.0, bot=2.25, theta=0.3)

_MU_SPACER = np.concatenate([np.full((1, 1, 20), 1.5), np.ones((1, 1, 20))])
_MU_DNG = np.concatenate([np.full((1, 1, 20), -1.5), np.ones((1, 1, 20))])
_PSI_PHI = dict(R1, psi=0.7, phi=1.1)
_PSI_PHI['wavelength'] = (1 - np.sin(0.3) * np.cos(1.1)) * 1000. / 2   # kx_2 = kx_0 + 2 wl/P = 1

UNIFORM = {
    'R1 air spacer': R1,
    'R2 air, wl 600, theta 0.1': dict(R1, wavelength=600., period=1200., theta=0.1),
    'R3 glass 1000 nm': dict(R1, structure=_stack(2.25), thickness=[1000., 200.]),
    'R4 glass, bot 1': dict(R1, structure=_stack(2.25), bot=1.0),
    'R5 glass 300 nm': dict(R1, structure=_stack(2.25)),
    # inputs that rounded within an ULP of the point and were silent before
    'water 1.7689 literal': dict(R1, structure=_stack(1.7689), wavelength=665.),
    'eps 2 at 1000 sqrt2/2': dict(R1, structure=_stack(2.0), wavelength=1000 * np.sqrt(2) / 2),
    'arange element (um)': dict(R1, wavelength=float(np.arange(0.4, 0.8, 0.001)[100]),
                                period=1.0, thickness=[0.3, 0.2]),
    'magnetic spacer': dict(R1, structure=_stack(1.5), mu=_MU_SPACER),
    'lossless double negative': dict(R1, structure=_stack(-1.5), mu=_MU_DNG),
    'backside': dict(R1, incidence_side='bottom'),
    'theta 1.4': dict(R1, theta=1.4),
    'psi 0.7, phi 1.1': _PSI_PHI,
}

_MU_PILLAR = np.ones((1, 1, 20))
_MU_PILLAR[0, 0, 6:14] = 2.0
_STEPPED = np.ones((1, 1, 24))
_STEPPED[0, 0, 2:9] = 6.0
_STEPPED[0, 0, 9:13] = 3.0

#: Patterned layers at their own x-cutoff, root-found on the family construction (``min|wA|``
#: at the stated wavelength is 2.7e-16 .. 2.7e-15): genuine 2x2 Jordan blocks.
PATTERNED = {
    'PA616': dict(structure=_pillar(), wavelength=616.515732741827, period=1000.,
                  thickness=[300.], top=1.0, bot=2.1025, theta=0.3),
    'PD574': dict(structure=_STEPPED, wavelength=574.819358670326, period=800., thickness=[300.],
                  top=1.0, bot=2.1025, theta=0.25, phi=1.2, truncation=6),
    'MU614': dict(structure=_pillar(), mu=_MU_PILLAR, wavelength=614.0618580721513,
                  period=1000., thickness=[300.], top=1.0, bot=2.1025, theta=0.4, phi=1.0),
}


def _solver(scene, *, algo='ETM', backend='numpy', two_d=False, **over):
    """A ``wavebit.rcwa`` for ``scene``: the 1D conical solve, or its 2D twin (y-invariant)."""
    sc = {**scene, **over}
    kw = dict(backend=backend, wavelength=sc['wavelength'], top=sc['top'], bot=sc['bot'],
              theta=sc['theta'], phi=sc.get('phi', HALF_PI), psi=sc.get('psi', HALF_PI),
              thickness=sc['thickness'], layer_algorithm=algo)
    for key in ('incidence_side', 'excitation', 'complex_dtype'):
        if key in sc:
            kw[key] = sc[key]
    to = sc.get('truncation', 5)
    structure, mu = sc['structure'], sc.get('mu')
    if two_d:
        kw.update(period=[sc['period'], PERIOD_Y], truncation_order=[to, 1],
                  structure=np.repeat(structure, 2, axis=1))
        if mu is not None:
            kw['mu'] = np.repeat(mu, 2, axis=1)
    else:
        kw.update(period=[sc['period']], truncation_order=[to, 0], structure=structure)
        if mu is not None:
            kw['mu'] = mu
    return wavebit.rcwa(**kw)


def _np(a):
    if hasattr(a, 'detach'):
        a = a.detach().cpu()
    return np.asarray(a)


def _numbers(result, two_d=False):
    """``R``, ``T`` per order (real) and the ``xyz`` amplitudes, the ``v = 0`` row on 2D."""
    R, T, r, t = (_np(getattr(result, k)) for k in ('R', 'T', 'r_xyz', 't_xyz'))
    if two_d:
        row = list(_np(result.orders_v)).index(0)
        R, T, r, t = R[row], T[row], r[row], t[row]
    return dict(R=np.real(R).ravel(), T=np.real(T).ravel(), r=r.ravel(), t=t.ravel())


def _vector(result):
    """``(sum R, sum T, r_xyz..., t_xyz...)`` as one complex vector, for fits and comparisons."""
    n = _numbers(result)
    return np.concatenate([[n['R'].sum(), n['T'].sum()], n['r'], n['t']])


def _errors(got, ref):
    return (abs(got['R'].sum() - ref['R'].sum()),
            max(np.max(np.abs(got['r'] - ref['r'])), np.max(np.abs(got['t'] - ref['t']))))


def _strict(call):
    """Run ``call`` with every warning an error, RuntimeWarning included."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        return call()


@contextlib.contextmanager
def _family_construction():
    """The gate switched off: every layer takes the family construction it had before."""
    saved = solve_1d._CUTOFF_REL
    solve_1d._CUTOFF_REL = 0.0
    try:
        yield
    finally:
        solve_1d._CUTOFF_REL = saved


@pytest.fixture
def rebuilds(monkeypatch):
    """Every call of the rebuild, eager or under autodiff (the twin calls it for its value)."""
    calls = []
    original = solve_1d._regular_conical_modes

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(solve_1d, '_regular_conical_modes', counted)
    return calls


def _offsets(wavelength):
    return {'exact': wavelength, '-ulp': np.nextafter(wavelength, 0.),
            '+ulp': np.nextafter(wavelength, 2 * wavelength), '+1e-12': wavelength * (1 + 1e-12)}


# ---------------------------------------------------------------- 1. uniform layers, vs 2D

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(UNIFORM))
def test_a_uniform_cutoff_matches_the_2d_path(name, algo, rebuilds):
    """At the exact point, one ULP either side and ``+1e-12``: the 2D path's answer, silently.

    The 2D path solves for the tangential fields directly and divides only by ``q = |ky|``
    there, so it is exact at a uniform cutoff (its own spread is 1e-14). Measured here:
    ``|dR| <= 1.2e-14`` and amplitudes ``<= 4.8e-14``; before, the exact point refused and its
    neighbours were 2.4e-3 .. 13 off with no warning. Every warning is an error, RuntimeWarning
    included, and the default solve must be the ``auto_detune=False`` one to the bit: there is
    nothing left to retry. Catches a gate that never engages and a rebuild that keeps the
    singular ``1/wA`` term.
    """
    scene = UNIFORM[name]
    for label, wavelength in _offsets(scene['wavelength']).items():
        ref = _numbers(_solver(scene, two_d=True, wavelength=wavelength)
                       .solve(auto_detune=False), True)
        del rebuilds[:]
        exact = _strict(lambda: _solver(scene, algo=algo, wavelength=wavelength)
                        .solve(auto_detune=False))
        assert rebuilds, f'{name} {label}: the rebuild did not engage'
        default = _strict(lambda: _solver(scene, algo=algo, wavelength=wavelength).solve())
        got = _numbers(exact)
        dR, amp = _errors(got, ref)
        assert dR < 1e-13 and amp < 1e-12, f'{name} {label}: dR {dR:.1e}, amplitudes {amp:.1e}'
        for key, value in _numbers(default).items():
            assert np.array_equal(value, got[key]), f'{name} {label}: the default moved {key}'


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_uniform_cutoff_on_jax_and_torch(backend, algo, rebuilds):
    """The same exact point, eager JAX and PyTorch: the 2D path's answer (measured <= 1.9e-14)."""
    pytest.importorskip(backend)
    ref = _numbers(_solver(R1, two_d=True).solve(auto_detune=False), True)
    got = _numbers(_solver(R1, algo=algo, backend=backend).solve(auto_detune=False))
    assert rebuilds
    dR, amp = _errors(got, ref)
    assert dR < 1e-13 and amp < 1e-12, f'dR {dR:.1e}, amplitudes {amp:.1e}'


# ------------------------------------------------------------------------------ 2. the gate

def _synthetic(wA, wB, ky, q1=None, q2=None):
    wA, wB = np.asarray(wA, dtype=complex), np.asarray(wB, dtype=complex)
    q1 = np.sqrt(wA + ky ** 2) if q1 is None else np.asarray(q1, dtype=complex)
    q2 = np.sqrt(wB + ky ** 2) if q2 is None else np.asarray(q2, dtype=complex)
    return solve_1d._cutoff_cluster(np, wA, wB, q1, q2, np.asarray(ky, dtype=complex))


def _gate(ky):
    return min(solve_1d._CUTOFF_REL * abs(ky) ** 1.5, abs(ky) ** 2 / 2)


def test_the_gate_never_engages_at_ky_zero():
    """``ky == 0`` leaves the gate at 0 and the comparison strict: an exact zero stays out.

    Real roots there are ``sqrt(wA)``, which the root floor would refuse anyway; the last case
    hands it roots well above the floor, so the strict comparison is pinned on its own.
    """
    assert _synthetic([0.0, 0.3], [0.0, 0.5], 0.0) is None
    assert _synthetic([1e-300, 0.3], [-1e-300, 0.5], 0.0) is None
    assert _synthetic([0.0, 0.3], [0.0, 0.5], 0.0, q1=[0.5, 0.6], q2=[0.5, 0.7]) is None


@pytest.mark.parametrize('ky', [0.3, 1.2])
def test_the_gate_engages_inside_its_edge_and_not_outside(ky):
    """``0.9 g`` is in, ``1.1 g`` is out, and only the modes inside are marked."""
    g = _gate(ky)
    masks = _synthetic([0.9 * g, 0.7], [-0.9 * g, 0.5], ky)
    assert masks is not None
    assert masks[0].tolist() == [True, False] and masks[1].tolist() == [True, False]
    assert _synthetic([1.1 * g, 0.7], [-1.1 * g, 0.5], ky) is None


def test_the_rayleigh_bound_takes_over_at_small_ky():
    """At ``ky = 1e-4`` the ``ky**2/2`` bound decides: ``1.1 ky**2/2`` is still far inside the
    ``|ky|**1.5`` law (4e-8) and must stay out, which keeps ``|q| >= |ky|/sqrt(2)``."""
    ky = 1e-4
    assert _gate(ky) == ky ** 2 / 2 < solve_1d._CUTOFF_REL * ky ** 1.5
    assert _synthetic([0.9 * ky ** 2 / 2, 0.5], [0.9 * ky ** 2 / 2, 0.5], ky) is not None
    assert _synthetic([1.1 * ky ** 2 / 2, 0.5], [1.1 * ky ** 2 / 2, 0.5], ky) is None


def test_the_gate_refuses_a_root_under_the_floor_and_opposite_labels():
    """A cluster root at the floor, or a pair labelled up and down, keeps the old construction."""
    ky, g = 0.3, _gate(0.3)
    wA, wB = [0.5 * g, 0.7], [0.5 * g, 0.5]
    q = np.sqrt(np.asarray(wA, dtype=complex) + ky ** 2)
    assert _synthetic(wA, wB, ky, q1=q, q2=q) is not None
    assert _synthetic(wA, wB, ky, q1=[1e-10, q[1]], q2=q) is None
    assert _synthetic(wA, wB, ky, q1=q, q2=[-q[0], q[1]]) is None


# ------------------------------------------------------- 3. patterned exceptional points

_FIT_OFFSETS = np.array([s * h for h in (1e-4, 1.5e-4, 2e-4, 3e-4, 4e-4, 6e-4, 8e-4, 1e-3)
                         for s in (-1, 1)])


@functools.lru_cache(maxsize=None)
def _fit(name):
    """Degree-8 polynomial in ``x = 1e3 (wl/wl0 - 1)`` through the family construction at 16
    offsets ``|rel|`` in [1e-4, 1e-3], where its rounding error (``~C u / wA**2``) is 1e-12 or
    less; the answer is analytic through the point (the layer propagator is entire in the
    operator), so this is the exceptional point's value with no reference to the code under
    test. Measured residual <= 1.7e-12; ETM and SMM fits agree to 1.2e-11.
    """
    scene = PATTERNED[name]
    with _family_construction(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Y = np.array([_vector(_solver(scene, wavelength=scene['wavelength'] * (1 + h))
                              .solve(auto_detune=False)) for h in _FIT_OFFSETS])
    return np.polyfit(1e3 * _FIT_OFFSETS, Y, 8)


@pytest.mark.parametrize('name', list(PATTERNED))
def test_a_patterned_exceptional_point_is_exact(name, monkeypatch):
    """At ``wl0`` and ``rel`` 0, +-1e-15, +-1e-12, +-1e-9: the fit's value, ETM = SMM, silently.

    The family construction was a lottery here (``R`` off by up to 11, ``R + T`` by 48, no
    warning). Measured now: amplitudes 1.4e-13 (PA616, PD574) and 1.2e-12 (MU614, the fit's own
    accuracy) from the fit. Catches, through the amplitudes: ``tau := 0`` (5.3e-3), ``tau`` of
    the wrong sign (1.1e-2), ``U_y`` of the coupled columns without ``tau`` (4.4e-3), ``N`` without
    ``mu_x^-1`` (1.5e-3, MU614) and the own-family ``V_21`` without ``muz_i`` (5.5e-2, MU614),
    and the singular ``1/wA`` term kept (0.87). The premise -- that ``wl0`` *is* a cutoff -- is
    read off the gate's own input.
    """
    scene = PATTERNED[name]
    coeffs = _fit(name)
    seen = []
    gate = solve_1d._cutoff_cluster

    def spy(xp, wA, wB, *rest):
        seen.append(min(np.min(np.abs(_np(wA))), np.min(np.abs(_np(wB)))))
        return gate(xp, wA, wB, *rest)
    monkeypatch.setattr(solve_1d, '_cutoff_cluster', spy)
    for rel in (0.0, 1e-15, -1e-15, 1e-12, -1e-12, 1e-9, -1e-9):
        wavelength = scene['wavelength'] * (1 + rel)
        truth = np.polyval(coeffs, 1e3 * (wavelength / scene['wavelength'] - 1))
        del seen[:]
        etm = _vector(_strict(lambda: _solver(scene, wavelength=wavelength)
                              .solve(auto_detune=False)))
        if rel == 0.0:
            assert min(seen) < 1e-13, f'{name}: wl0 is not a cutoff (min|w| {min(seen):.1e})'
        smm = _vector(_strict(lambda: _solver(scene, algo='SMM', wavelength=wavelength)
                              .solve(auto_detune=False)))
        assert np.max(np.abs(etm - truth)) < 1e-10, (
            f'{name} rel {rel:g}: {np.max(np.abs(etm - truth)):.1e} from the fit')
        assert np.max(np.abs(smm - etm)) < 1e-13, (
            f'{name} rel {rel:g}: SMM {np.max(np.abs(smm - etm)):.1e} from ETM')


# ------------------------------------------------------------------- 4. a wide cluster

@pytest.mark.parametrize('name', ['PA616', 'MU614'])
def test_a_wide_cluster_agrees_with_the_family_construction(name, monkeypatch):
    """At ``rel = +1e-3`` both families route, and the family construction is still accurate.

    The rebuild is an exact change of basis for any masks, so the two agree (measured 4.7e-14).
    The only value check that sees the ``tau`` term in the coupled columns' ``U_x``, which is
    ``O(wA tau)`` at the point itself (1.1e-4 here when dropped).
    """
    scene = PATTERNED[name]
    wavelength = scene['wavelength'] * (1 + 1e-3)
    masks = []
    gate = solve_1d._cutoff_cluster

    def spy(*args):
        out = gate(*args)
        if out is not None:
            masks.append(out)
        return out
    monkeypatch.setattr(solve_1d, '_cutoff_cluster', spy)
    got = _vector(_solver(scene, wavelength=wavelength).solve(auto_detune=False))
    assert masks and all(bool(np.any(_np(m))) for m in masks[0])
    with _family_construction():
        old = _vector(_solver(scene, wavelength=wavelength).solve(auto_detune=False))
    assert np.max(np.abs(got - old)) < 1e-10, f'{np.max(np.abs(got - old)):.1e}'


# --------------------------------------------------------- 5. the first-order relations

@pytest.mark.parametrize('name, rel', [('PA616', 0.0), ('PA616', 1e-3), ('MU614', 0.0),
                                       ('MU614', 1e-3)])
def test_the_rebuilt_layer_satisfies_both_first_order_relations(name, rel, monkeypatch):
    """``Q E = U q`` and ``P U = E q`` on the rebuilt layer, relative to ``|Q||E|`` / ``|P||U|``.

    Measured 2.9e-16; a coupled column's ``U_x`` without ``tau`` gives 7.5e-6.
    """
    scene = PATTERNED[name]
    captured = []
    build = solve_1d.layer_modes_1d_conical

    def spy(xp, Kx, Ky, Kx2, Ky2, I, epx, epy, epz_i, C, mu=None, **kw):
        out = build(xp, Kx, Ky, Kx2, Ky2, I, epx, epy, epz_i, C, mu=mu, **kw)
        if len(out) == 3:
            captured.append((Kx, Ky, Kx2, I, epx, epy, epz_i, mu, out))
        return out
    monkeypatch.setattr(solve_1d, 'layer_modes_1d_conical', spy)
    _solver(scene, wavelength=scene['wavelength'] * (1 + rel)).solve(auto_detune=False)
    assert captured
    for Kx, Ky, Kx2, I, epx, epy, epz_i, mu, (E, U, q) in captured:
        mux, muy, muz_i = (I, I, None) if mu is None else mu
        A = (Kx2 if muz_i is None else Kx @ muz_i @ Kx) - epy
        B = Kx @ epz_i @ Kx - muy
        L = solve_1d._conical_first_order(np, Kx, Ky[0, 0], epx, epz_i, A, B, mux, muz_i)
        n2 = E.shape[0]
        P, Q = L[:n2, n2:], L[n2:, :n2]
        rq = np.linalg.norm(Q @ E - U @ q) / (np.linalg.norm(Q) * np.linalg.norm(E))
        rp = np.linalg.norm(P @ U - E @ q) / (np.linalg.norm(P) * np.linalg.norm(U))
        assert max(rq, rp) < 1e-12, f'{name} rel {rel}: QE-Uq {rq:.1e}, PU-Eq {rp:.1e}'


def test_a_zero_root_outside_the_cluster_forms_no_inf():
    """A root exactly 0 beside a cluster: no division warning, finite modes, both relations hold.

    An air layer with ``ky = 0.3`` and orders ``kx = 1`` (at its cutoff, ``wA = 0``),
    ``sqrt(1 - ky**2)`` (``q = 0``: ``|wA| = ky**2``, far outside the gate) and 2. The rebuild's
    ``wA/q1`` and the coupled columns' ``/q2`` sit in lanes the masks discard for that order;
    they must divide by the guard's 1, or NumPy warns, which every warning an error turns into a
    raise -- the class of input the guards exist for, and none of the scenes above builds it.
    """
    ky = 0.3
    Kx = np.diag([1.0, np.sqrt(1 - ky ** 2), 2.0]).astype(complex)
    I = np.eye(3, dtype=complex)
    A = Kx @ Kx - I                 # = B on an air layer, and diagonal, so W1 = W2 = I
    w = np.array([0.0, -ky ** 2, 3.0], dtype=complex)            # wA = wB
    q = np.array([ky, 0.0, np.sqrt(3.0 + ky ** 2)], dtype=complex)
    mask = np.array([True, False, False])
    E, U, T = _strict(lambda: solve_1d._regular_conical_modes(
        np, Kx, ky, I, I, A, I, None, False, I, I, I, w, w, q, q, mask, mask))
    assert all(np.all(np.isfinite(x)) for x in (E, U, T))
    L = solve_1d._conical_first_order(np, Kx, ky, I, I, A, A, I, None)
    P, Q = L[:6, 6:], L[6:, :6]
    assert np.linalg.norm(Q @ E - U @ T) < 1e-14 * np.linalg.norm(Q) * np.linalg.norm(E)
    assert np.linalg.norm(P @ U - E @ T) < 1e-14 * np.linalg.norm(P) * np.linalg.norm(U)


# ----------------------------------------------------------------------- 6. derivatives

def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@functools.lru_cache(maxsize=None)
def _reference_gradients():
    """``jax.grad`` of the 2D path at R1's exact point: raster, wavelength, first thickness."""
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    s = _solver(R1, backend='jax', two_d=True)

    def f(u, w, t):
        r = s.solve(structure=jnp.repeat(u, 2, axis=1), wavelength=w,
                    thickness=jnp.stack([t, jnp.asarray(200.)]), auto_detune=False)
        return jnp.sum(jnp.real(r.R))
    g = jax.grad(f, argnums=(0, 1, 2))(jnp.asarray(R1['structure']), jnp.asarray(500.),
                                       jnp.asarray(300.))
    return tuple(np.asarray(x) for x in g)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_uniform_cutoff_derivatives_on_jax(algo, rebuilds):
    """``jax.grad``, ``jax.jvp`` and ``jax.jacfwd`` at R1's exact point match the 2D path.

    Measured: raster 3.9e-14, wavelength 3.7e-13 .. 4.2e-13, thickness 4e-15 (relative); before,
    the point refused and the default's gradient was off by 1e4.
    """
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    g_raster, g_wl, g_th = _reference_gradients()
    s = _solver(R1, algo=algo, backend='jax')

    def f(u, w, t):
        r = s.solve(structure=u, wavelength=w, thickness=jnp.stack([t, jnp.asarray(200.)]),
                    auto_detune=False)
        return jnp.sum(jnp.real(r.R))
    args = (jnp.asarray(R1['structure']), jnp.asarray(500.), jnp.asarray(300.))
    gu, gw, gt = jax.grad(f, argnums=(0, 1, 2))(*args)
    assert rebuilds
    assert _rel(gu, g_raster) < 1e-11 and _rel(gw, g_wl) < 1e-11 and _rel(gt, g_th) < 1e-11, (
        _rel(gu, g_raster), _rel(gw, g_wl), _rel(gt, g_th))
    jvp_t = jax.jvp(lambda t: f(args[0], args[1], t), (args[2],), (jnp.asarray(1.0),))[1]
    assert _rel(jvp_t, g_th) < 1e-11
    jac = jax.jacfwd(lambda u: f(u, args[1], args[2]))(args[0])
    assert _rel(jac, g_raster) < 1e-11


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_uniform_cutoff_derivatives_on_torch(algo):
    """torch reverse (raster) and ``forward_ad`` (wavelength) at R1's exact point vs the 2D path."""
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    g_raster, g_wl, _ = _reference_gradients()
    u = torch.tensor(R1['structure'], dtype=torch.float64, requires_grad=True)
    _solver(R1, algo=algo, backend='torch', structure=u).solve(auto_detune=False).R.real.sum().backward()
    assert _rel(u.grad.numpy(), g_raster) < 1e-11
    with forward_ad.dual_level():
        w = forward_ad.make_dual(torch.tensor(500., dtype=torch.float64),
                                 torch.tensor(1., dtype=torch.float64))
        r = _solver(R1, algo=algo, backend='torch', wavelength=w).solve(auto_detune=False)
        tangent = forward_ad.unpack_dual(r.R.real.sum()).tangent
    assert _rel(float(tangent), g_wl) < 1e-11


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', ['PA616', 'MU614'])
def test_patterned_derivatives_match_the_fit(name, algo):
    """``dR/dwl`` at the exceptional point is the fit's slope; JAX and PyTorch agree.

    Measured against the fit 5.3e-11 (PA616) and 5.8e-9 (MU614, the fit's own accuracy) on
    ``jax.grad`` and torch reverse; raster gradients JAX vs torch 3.2e-14 .. 8.5e-14. Before:
    1e11 .. 1e28. Catches a frozen subspace (``X := 0`` in the autodiff twin, 0.17) and a
    coupled exponential whose tangent is the divided-difference formula's (6.4e-3) -- the only
    check that sees the second.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    jnp = jax.numpy
    scene = PATTERNED[name]
    slope = np.real(_fit(name)[-2][0]) * 1e3 / scene['wavelength']
    s = _solver(scene, algo=algo, backend='jax')
    g = float(jax.grad(lambda w: jnp.sum(jnp.real(s.solve(wavelength=w, auto_detune=False).R)))(
        jnp.asarray(scene['wavelength'])))
    assert abs(g - slope) < 1e-6 * abs(slope), f'jax.grad {abs(g - slope) / abs(slope):.1e}'
    w = torch.tensor(scene['wavelength'], dtype=torch.float64, requires_grad=True)
    r = _solver(scene, algo=algo, backend='torch', wavelength=w).solve(auto_detune=False)
    r.R.real.sum().backward()
    assert abs(float(w.grad) - slope) < 1e-6 * abs(slope)
    u0 = np.asarray(scene['structure'], dtype=float)
    gj = np.asarray(jax.grad(
        lambda u: jnp.sum(jnp.real(s.solve(structure=u, auto_detune=False).R)))(jnp.asarray(u0)))
    u = torch.tensor(u0, dtype=torch.float64, requires_grad=True)
    r = _solver(scene, algo=algo, backend='torch', structure=u).solve(auto_detune=False)
    r.R.real.sum().backward()
    assert _rel(u.grad.numpy(), gj) < 1e-12


#: Derivatives that move ``ky``. ``wA = w1 - ky**2`` carries ``ky``'s tangent, so an angle is the
#: one parameter that sees whether the twin's value is built from stopped inputs; the raster,
#: the wavelength and the thickness above leave ``ky`` alone. ``R1 rel 1e-3`` is inside the band
#: (``|wA| = 2e-3`` against a gate of 6.4e-3) but off the point, where the error was as large.
_ANGLE_SCENES = {
    'R1 exact, theta': (R1, 'theta'),
    'R1 rel 1e-3, theta': (dict(R1, wavelength=500. * (1 + 1e-3)), 'theta'),
    'psi 0.7, phi 1.1, phi': (_PSI_PHI, 'phi'),
}


@functools.lru_cache(maxsize=None)
def _reference_angle_gradient(name):
    """``jax.grad`` of the 2D path's ``sum R`` (its ``v = 0`` row) in the scene's angle."""
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    scene, var = _ANGLE_SCENES[name]
    s = _solver(scene, backend='jax', two_d=True)
    row = list(_np(s.solve(auto_detune=False).orders_v)).index(0)
    return float(jax.grad(lambda x: jnp.sum(jnp.real(
        s.solve(**{var: x}, auto_detune=False).R[row])))(jnp.asarray(scene[var])))


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(_ANGLE_SCENES))
def test_angle_derivatives_on_a_rebuilt_layer(name, algo, rebuilds):
    """``d(sum R)/dtheta`` and ``d/dphi`` through a rebuilt layer: the 2D path's ``jax.grad``.

    On ``jax.grad`` and torch reverse alike. Measured 4.3e-15 .. 8.2e-14. With ``wA`` and ``wB``
    left live in the twin's value build, the value's ``q1/wA``, ``wA/q1`` and ``1/wB`` took a
    tangent of their own: 5.0e-1 off at R1's point, 5.2e-1 at ``rel 1e-3`` (where the family
    construction had been right to 6.7e-13), 6.2e-2 in ``phi``.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    jnp = jax.numpy
    scene, var = _ANGLE_SCENES[name]
    ref = _reference_angle_gradient(name)
    s = _solver(scene, algo=algo, backend='jax')
    g = float(jax.grad(lambda x: jnp.sum(jnp.real(s.solve(**{var: x}, auto_detune=False).R)))(
        jnp.asarray(scene[var])))
    assert rebuilds
    assert abs(g - ref) < 1e-11 * abs(ref), f'jax.grad {abs(g - ref) / abs(ref):.1e}'
    x = torch.tensor(scene[var], dtype=torch.float64, requires_grad=True)
    r = _solver(scene, algo=algo, backend='torch', **{var: x}).solve(auto_detune=False)
    r.R.real.sum().backward()
    assert abs(float(x.grad) - ref) < 1e-11 * abs(ref), (
        f'torch reverse {abs(float(x.grad) - ref) / abs(ref):.1e}')


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', ['PA616', 'MU614'])
def test_patterned_angle_derivatives_match_a_central_difference(name, algo, rebuilds):
    """``d(sum R)/dtheta`` at a patterned exceptional point against a central difference.

    The 2D path is no oracle there (``docs/rayleigh-points.md`` section 7: 1e-8 .. 4e-5 off), so
    the reference is the conical solve's own values at ``theta +- 1e-5``, which the rebuild makes
    exact (amplitudes <= 1.2e-12 from the fit above). Measured 8.5e-10 .. 2.0e-8 (the difference's
    own error); with ``wA`` and ``wB`` live in the twin's value build, 2.2e-1 (PA616) and 4.2e-1
    (MU614).
    """
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    scene = PATTERNED[name]
    s = _solver(scene, algo=algo, backend='jax')
    g = float(jax.grad(lambda x: jnp.sum(jnp.real(s.solve(theta=x, auto_detune=False).R)))(
        jnp.asarray(scene['theta'])))
    assert rebuilds
    h = 1e-5
    side = [np.real(_numbers(_solver(scene, algo=algo, theta=scene['theta'] + sign * h)
                             .solve(auto_detune=False))['R']).sum() for sign in (1, -1)]
    fd = (side[0] - side[1]) / (2 * h)
    assert abs(g - fd) < 1e-6 * abs(fd), f'{abs(g - fd) / abs(fd):.1e}'


# ----------------------------------------------------------------------------- 7. fields

def _field(scene, two_d=False, **over):
    r = _solver(scene, two_d=two_d, **over).solve(auto_detune=False, answers='field')
    return _np(r.reconstruct_field(sample_x=12, sample_y=1, sample_z=8))


_SLICED = dict(R1, structure=_stack(1.0, slices=3), thickness=[100., 100., 100., 200.])


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name, scene', [
    ('R1 top lit', R1), ('R1 bottom lit', dict(R1, incidence_side='bottom')),
    ('sliced spacer', _SLICED), ('magnetic spacer (H_z)', UNIFORM['magnetic spacer'])])
def test_a_uniform_cutoff_field_matches_the_2d_path(name, scene, algo, rebuilds):
    """All six components at the exact point against the 2D field, its cell-centre phase removed.

    The 2D cell's centre sits ``Py/2`` up in ``y``, which puts one global phase on its field.
    Measured <= 1.2e-14 of ``max|F|``. Catches field reconstruction reading a rebuilt layer
    (``W`` ``2n x 2n``) in the family layout.
    """
    ref = _field(scene, two_d=True)
    got = _field(scene, algo=algo)
    assert rebuilds
    phase = np.exp(1j * np.angle(np.vdot(got.ravel(), ref.ravel())))
    err = np.max(np.abs(got * phase - ref)) / np.max(np.abs(ref))
    assert err < 1e-12, f'{name} {algo}: {err:.1e}'


_GLASS_OVER_PA616 = dict(PATTERNED['PA616'], thickness=[150., 300.], structure=np.concatenate(
    [np.full((1, 1, 20), 1.7), PATTERNED['PA616']['structure']], axis=0))

#: Patterned exceptional points for the SMM-vs-ETM field check. ``PA616 500 nm``: two of its
#: modes (``|q|`` 0.112 and 0.124) stay inside the field state's ``|k0 d q| < 1`` threshold while
#: the cutoff pair (``|k0 d q| = 1.5``) is dropped -- a kept row beside a dropped coupling column.
#: The two-layer stacks put glass over the pillar and light them from either side, so the
#: per-layer ``coupled`` flags must follow the mirrored scene the field state is built on (a
#: top-lit solve is the mirrored one). Lit from below, ``theta`` is taken in the substrate, so
#: ``ky = 1.45 sin 0.3``: the pillar stays at its exceptional point (``wA`` and ``wB`` do not
#: depend on ``ky``), but its cutoff pair moves out to ``|k0 d q| = 1.31`` and is dropped as well.
_PATTERNED_FIELDS = {
    'PA616': PATTERNED['PA616'],
    'MU614': PATTERNED['MU614'],
    'PA616 500 nm': dict(PATTERNED['PA616'], thickness=[500.]),
    'glass over PA616, top lit': _GLASS_OVER_PA616,
    'glass over PA616, bottom lit': dict(_GLASS_OVER_PA616, incidence_side='bottom'),
}


@pytest.mark.parametrize('name', list(_PATTERNED_FIELDS))
def test_a_patterned_cutoff_field_and_its_gradient_agree_across_algorithms(name):
    """SMM's field and field gradient at an exceptional point are the ETM's.

    The ETM's record does not go through ``two_face``; SMM's does. The gradient is the whole
    raster's and the wavelength's. Measured: fields <= 2.2e-15 of ``max|F|``, gradients
    <= 1.6e-14 relative (on PA616 and MU614 one raster pixel and the wavelength were also within
    1e-10 of a central difference). Catches the field state reading a coupled layer with plain
    ``two_face`` (6.8e-3) or dropping only the coupled rows without the term they leave out
    (2.6e-2); the dropped-column branch of that term replaced by the kept-column one (raster
    8.3e-4 at ``PA616 500 nm``, 1.8e-3 bottom lit); and the mirrored scene handed its flags
    unreversed, which marks the glass instead of the pillar (top lit: raster 3.9e-4, wavelength
    1.3e-5). Before these scenes neither of the last two failed any test.
    """
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    scene = _PATTERNED_FIELDS[name]
    fields = {algo: _field(scene, algo=algo) for algo in ('ETM', 'SMM')}
    scale = np.max(np.abs(fields['ETM']))
    assert np.max(np.abs(fields['SMM'] - fields['ETM'])) < 1e-12 * scale
    xs, zs = np.array([-200., 0., 130.]), np.array([0.13, 0.5, 0.87]) * sum(scene['thickness'])
    grads = {}
    for algo in ('ETM', 'SMM'):
        s = _solver(scene, algo=algo, backend='jax')

        def F(u, w):
            r = s.solve(structure=u, wavelength=w, auto_detune=False, answers='field')
            f = r.reconstruct_field(sample_x=xs, sample_y=[0.], sample_z=zs)
            return jnp.sum(jnp.abs(f[..., :3]) ** 2)
        gu, gw = jax.grad(F, argnums=(0, 1))(jnp.asarray(scene['structure'], dtype=float),
                                            jnp.asarray(scene['wavelength']))
        grads[algo] = (np.asarray(gu), float(gw))
    (gu_etm, gw_etm), (gu_smm, gw_smm) = grads['ETM'], grads['SMM']
    raster = np.max(np.abs(gu_smm - gu_etm)) / np.max(np.abs(gu_etm))
    wavelength = abs(gw_smm - gw_etm) / abs(gw_etm)
    assert raster < 1e-10 and wavelength < 1e-10, (
        f'{name}: raster {raster:.1e}, wavelength {wavelength:.1e}')


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_an_eager_field_record_differentiates_alike_under_jit(algo):
    """A PA616 record built eagerly, passed into ``jax.jit(jax.grad(...))``: the eager derivative.

    Passed in, the record's ``q`` is a tracer, so the coupled exponential takes its tangent from
    ``_expm`` with an abstract norm; eagerly ``q`` is a constant and the formula's exact ``scale``
    derivative is used. The traced squaring count keeps the two together (measured 4.1e-16 ..
    8.0e-16 in ``d/dz``); falling back to ``jax.scipy.linalg.expm``'s own floor count there was
    6.4e-11 off.
    """
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    result = _solver(PATTERNED['PA616'], algo=algo, backend='jax').solve(auto_detune=False,
                                                                       answers='field')

    def F(r, z):
        f = r.reconstruct_field(sample_x=jnp.asarray([0., 130.]), sample_y=jnp.asarray([0.]),
                                sample_z=z)
        return jnp.sum(jnp.abs(f[..., :3]) ** 2)
    z = jnp.asarray([40., 150., 260.])
    eager = np.asarray(jax.grad(F, argnums=1)(result, z))
    jitted = np.asarray(jax.jit(jax.grad(F, argnums=1))(result, z))
    err = np.max(np.abs(jitted - eager)) / np.max(np.abs(eager))
    assert err < 1e-13, f'jit(grad) {err:.1e} from eager grad'


# ---------------------------------------------------------------- 8. inert off the band

def _random_scenes(count, seed=20261010):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        n_layers = int(rng.integers(1, 3))
        st = np.empty((n_layers, 1, 20))
        for layer in range(n_layers):
            st[layer] = rng.uniform(1.0, 3.0)
            a, b = sorted(rng.integers(0, 20, 2))
            st[layer, 0, a:b + 1] = rng.uniform(1.5, 12.0)
        out.append(dict(structure=st, wavelength=float(rng.uniform(400, 1500)),
                        period=float(rng.uniform(300, 1500)),
                        thickness=list(rng.uniform(50, 500, n_layers)), top=1.0,
                        bot=float(rng.choice([1.0, 1.45, 2.25])),
                        theta=float(rng.uniform(0.1, 1.2)),
                        phi=float(rng.uniform(0.2, 1.4)), psi=float(rng.uniform(0, np.pi))))
    return out


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_off_the_band_nothing_moves(algo, rebuilds):
    """R1 at ``rel = 5e-2`` and random scenes: no rebuild, and every bit the gate-off solve's."""
    for scene in [dict(R1, wavelength=500. * 1.05)] + _random_scenes(4):
        got = _numbers(_solver(scene, algo=algo).solve(auto_detune=False))
        with _family_construction():
            old = _numbers(_solver(scene, algo=algo).solve(auto_detune=False))
        for key in got:
            assert np.array_equal(got[key], old[key]), key
    assert not rebuilds


def test_under_an_abstract_trace_the_family_construction_ships(rebuilds):
    """The documented gap: a traced solve cannot read the gate, so no layer is rebuilt.

    The gate's masks decide the layer's *structure*, so wherever its values are abstract the old
    construction runs, bit for bit -- including its NaN at R1's exact zero
    (``docs/rayleigh-points.md`` section 7). Not only ``jax.jit``: ``jax.vmap`` over the
    wavelength, the common sweep idiom, is abstract too (measured: NaN at the point, 2.6e-2 off
    at ``+-1e-12``, no warning), and so are a ``jax.lax.map`` body, ``jax.checkpoint`` and
    ``torch.func.vmap`` (2.5e-2..2.7e-2). If this starts failing because a trace got the
    rebuild, update the docs with it.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    jnp = jax.numpy
    wls = [500. * (1 - 1e-12), 500., 500. * (1 + 1e-12)]

    def traced():
        s = _solver(R1, backend='jax')
        r = jax.jit(lambda u: s.solve(structure=u, auto_detune=False))(jnp.asarray(R1['structure']))
        sweep = jnp.asarray(wls)
        one = lambda w: s.solve(wavelength=w, auto_detune=False).R
        v = jax.vmap(one)(sweep)
        m = jax.lax.map(one, sweep)
        c = [jax.checkpoint(one)(w) for w in sweep]
        st = _solver(R1, backend='torch')
        t = torch.func.vmap(lambda w: st.solve(wavelength=w, auto_detune=False).R)(
            torch.tensor(wls, dtype=torch.float64))
        return (np.asarray(r.R), np.asarray(r.r_xyz), np.asarray(v), np.asarray(m),
                np.asarray(c), t.detach().numpy())
    got = traced()
    assert not rebuilds
    with _family_construction():
        old = traced()
    for a, b in zip(got, old):
        np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ 9. ky = 0 untouched

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_the_ky_zero_rayleigh_point_still_refuses(algo, rebuilds, monkeypatch):
    """At ``theta = phi = 0`` the same order is a Rayleigh point: refused, never rebuilt.

    ``kx_2 = 1`` on an air spacer whose operator is exactly diagonal, so the zero is exact on
    any LAPACK. The gate is consulted and declines (``ky == 0``), the refusal is the one
    ``check_finite`` always gave, and the default ladder still rescues it with a notice.
    """
    from wavebit._core.backend import AutoDetuneWarning
    asked = []
    gate = solve_1d._cutoff_cluster

    def spy(*args):
        out = gate(*args)
        asked.append(out)
        return out
    monkeypatch.setattr(solve_1d, '_cutoff_cluster', spy)
    scene = dict(R1, theta=0.0, phi=0.0)
    with pytest.raises(FloatingPointError):
        _solver(scene, algo=algo).solve(auto_detune=False)
    assert asked and all(out is None for out in asked)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _solver(scene, algo=algo).solve()
    assert any(issubclass(r.category, AutoDetuneWarning) for r in rec)
    assert not rebuilds


# --------------------------------------------------------------------------- 11. complex64

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_uniform_cutoff_at_complex64(algo):
    """No refusal at single precision, and the 2D path's complex64 answer (measured 3.8e-7).

    Before, the exact point refused and the default's 1e-12 step could not move a float32
    wavelength, so the refusal surfaced.
    """
    ref = _numbers(_solver(dict(R1, complex_dtype=1), two_d=True).solve(auto_detune=False), True)
    for auto in (False, True):
        got = _numbers(_solver(dict(R1, complex_dtype=1), algo=algo).solve(auto_detune=auto))
        dR, amp = _errors(got, ref)
        assert amp < 1e-6, f'auto_detune={auto}: amplitudes {amp:.1e}'


# ------------------------------------------------------- 12. a cutoff beside a near-graze

_COINCIDE = dict(R1, theta=np.pi / 3)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_cutoff_beside_a_near_graze(algo):
    """``theta = pi/3``: order 2 at the cutoff, order 1 at ``k_par**2 ~ 1``, in one layer.

    The near-graze is a Rayleigh-type sensitivity (its amplitude carries ~sqrt of a 1e-16
    crumb), which caps the exact point at the 1e-9 level: measured 3.8e-9 against the 2D path,
    8.9e-13 at ``rel = 1e-9``. SMM's field must stay with the ETM's (measured 1.3e-9): a field
    state that read every mode of the rebuilt layer off its bottom face took ``two_face``'s
    protection from the near-grazing order and lost 1.5e-2 here.
    """
    for rel, tol in ((0.0, 1e-8), (1e-9, 1e-11)):
        wavelength = 500. * (1 + rel)
        ref = _numbers(_solver(_COINCIDE, two_d=True, wavelength=wavelength)
                       .solve(auto_detune=False), True)
        got = _numbers(_solver(_COINCIDE, algo=algo, wavelength=wavelength)
                       .solve(auto_detune=False))
        dR, amp = _errors(got, ref)
        assert amp < tol, f'rel {rel}: amplitudes {amp:.1e}'
    if algo == 'SMM':
        etm, smm = _field(_COINCIDE, algo='ETM'), _field(_COINCIDE, algo='SMM')
        err = np.max(np.abs(smm - etm)) / np.max(np.abs(etm))
        assert err < 1e-8, f'SMM field {err:.1e} from the ETM field'


# ----------------------------------------------- 13. excitation= and SMM's scattering

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_multimode_excitation_at_a_cutoff(algo):
    """``excitation={0: 1, 1: 0.6-0.8j}`` at R1's exact point: the 2D answer (measured 6e-15)."""
    scene = dict(R1, excitation={0: 1.0, 1: 0.6 - 0.8j})
    ref = _numbers(_solver(scene, two_d=True).solve(auto_detune=False), True)
    got = _numbers(_strict(lambda: _solver(scene, algo=algo).solve(auto_detune=False)))
    dR, amp = _errors(got, ref)
    assert dR < 1e-13 and amp < 1e-12, f'dR {dR:.1e}, amplitudes {amp:.1e}'


def test_the_scattering_matrix_is_finite_and_continuous_at_a_cutoff():
    """SMM's ``scattering`` at R1's exact point is finite and moves ~1e-7 for ``rel = +-1e-9``.

    Measured 9.3e-8 on both sides -- the physical shift, not the old construction's 1e-2
    plateau.
    """
    def S(rel):
        s = _solver(R1, algo='SMM', wavelength=500. * (1 + rel))
        s.solve(auto_detune=False, answers='amplitude')
        M = s.scattering
        return np.concatenate([np.ravel(_np(getattr(M, k))) for k in ('S11', 'S12', 'S21', 'S22')])
    at = S(0.0)
    assert np.all(np.isfinite(at))
    for rel in (1e-9, -1e-9):
        assert np.max(np.abs(at - S(rel))) < 1e-6


# ---------------------------------------------------------- the coupled exponential itself

@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_the_coupled_exponential_is_the_matrix_exponential(backend):
    """``exponential(coupled=True)``: the value from the formula, the derivative from ``expm``.

    A propagator shaped like a rebuilt layer's (two cluster pairs coupled, at ``|A|_1 ~ 10``,
    where ``jax.scipy.linalg.expm`` unscaled was 4e-9 off), differentiated in a direction with
    entries everywhere, against a central difference of SciPy's ``expm``. The formula's own
    derivative misses the coupling chained behind the variation; ``coupled=False`` must show it.
    """
    scipy_linalg = pytest.importorskip('scipy.linalg')
    rng = np.random.default_rng(7)
    n = 6
    q = np.diag(rng.uniform(0.3, 1.8, n) + 1j * rng.uniform(0, 0.4, n))
    q[0, 3], q[1, 4] = 0.7 - 0.2j, -0.4 + 0.5j          # rows {0, 1}, columns {3, 4}
    q[1, 1] = q[4, 4] = 0.9 + 0.1j                      # one confluent pair, as in a uniform layer
    direction = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    scale = -5.0
    h = 1e-6
    fd = (scipy_linalg.expm(scale * (q + h * direction))
          - scipy_linalg.expm(scale * (q - h * direction))) / (2 * h)
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jnp = jax.numpy

        def run(coupled):
            def f(t):
                return modal_gradient.exponential(
                    jnp, jnp.asarray(q) + t * jnp.asarray(direction), scale, coupled=coupled)
            value, tangent = jax.jvp(f, (jnp.asarray(0.0, dtype=jnp.complex128),),
                                     (jnp.asarray(1.0, dtype=jnp.complex128),))
            return np.asarray(value), np.asarray(tangent)
    else:
        torch = pytest.importorskip('torch')
        from torch.autograd import forward_ad
        import array_api_compat.torch as xp

        def run(coupled):
            with forward_ad.dual_level():
                Q = forward_ad.make_dual(torch.tensor(q), torch.tensor(direction))
                out = forward_ad.unpack_dual(modal_gradient.exponential(xp, Q, scale,
                                                                        coupled=coupled))
                return out.primal.numpy(), out.tangent.numpy()
    value, tangent = run(True)
    exact = scipy_linalg.expm(scale * q)
    assert np.max(np.abs(value - exact)) < 1e-13 * np.max(np.abs(exact))
    assert np.max(np.abs(tangent - fd)) < 1e-8 * np.max(np.abs(fd))
    _, formula = run(False)
    assert np.max(np.abs(formula - fd)) > 1e-4 * np.max(np.abs(fd))
