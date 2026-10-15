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

:func:`~wavebit.rcwa._engine.solve_1d._regular_conical_modes` now rebuilds such a layer in a regular
basis with an upper-triangular propagator, gated by
:func:`~wavebit.rcwa._engine.solve_1d._cutoff_cluster`. What this file pins:

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
8. inertness off the band (eager: every bit; under ``jax.jit``: equal accuracy), and the
   rebuild under every abstract trace -- ``jax.jit``, ``jax.vmap``, ``jax.lax.map``,
   ``jax.checkpoint``, ``torch.func.vmap`` -- with the gate's masks as data: values, first and
   second derivatives (``torch.autograd`` and ``forward_ad`` through ``torch.func.vmap``
   among them), fields and field gradients, and the gate itself under ``jax.jit``;
9. the ``ky == 0`` Rayleigh contract, untouched;
11. complex64; 12. a cutoff and a near-graze in one layer; 13. ``excitation=`` and SMM's
    ``scattering``; and the coupled exponential itself: its eager ``expm`` tangent, its traced
    closed-form Frechet tangent against SciPy's ``expm_frechet``, and which one runs where.
    (Item 10 lives in ``test_rayleigh_guard.py``, ``[1D ky]``.)

Sizes stay at ``truncation_order=5``. ``docs/theory/rcwa/rayleigh-points.md`` section 7 has the story and
the measurements behind every tolerance below.
"""

from wavebit._common.backend import ComplexIncidenceWarning as _legacy_wavebit_ComplexIncidenceWarning

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import contextlib
import functools
import warnings

import numpy as np
import pytest

from wavebit.rcwa._engine import modal_gradient, solve_1d

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
    return _legacy_wavebit_rcwa(**kw)


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


def test_the_gate_refuses_a_root_under_the_floor_and_takes_opposite_labels():
    """A cluster root at the floor keeps the old construction; a pair labelled up and down is
    taken, and the rebuild puts both on the deepest member's side (:func:`_one_side`)."""
    ky, g = 0.3, _gate(0.3)
    wA, wB = [0.5 * g, 0.7], [0.5 * g, 0.5]
    q = np.sqrt(np.asarray(wA, dtype=complex) + ky ** 2)
    assert _synthetic(wA, wB, ky, q1=q, q2=q) is not None
    assert _synthetic(wA, wB, ky, q1=[1e-10, q[1]], q2=q) is None
    masks = _synthetic(wA, wB, ky, q1=q, q2=[-q[0], q[1]])
    assert masks is not None and masks[0].tolist() == [True, False]
    q1, q2 = solve_1d._one_side(np, np.asarray(wA, dtype=complex), np.asarray(wB, dtype=complex),
                                q, np.asarray([-q[0], q[1]]), *masks)
    assert q1.tolist() == q.tolist() and q2.tolist() == [q[0], q[1]]


def test_one_side_follows_the_deepest_member_and_flips_nothing_else():
    """The reference is the member with the smallest ``|w|``, whichever family it is in; modes
    outside the masks keep their labels whatever they are; with no disagreement the output is
    the input, bit for bit."""
    q = np.asarray([0.3 + 0.01j, 0.31 + 0.0j, -0.29 - 0.02j], dtype=complex)
    wA = np.asarray([1e-3, 2e-3, 0.5], dtype=complex)
    wB = np.asarray([1e-9, 3e-3, 0.4], dtype=complex)
    mJ = np.asarray([True, True, False])
    mK = np.asarray([True, True, False])
    # family 2's first member is the deepest: its label (-) wins, and family 1 follows
    q1, q2 = solve_1d._one_side(np, wA, wB, q, np.asarray([-q[0], -q[1], q[2]]), mJ, mK)
    assert q1.tolist() == [-q[0], -q[1], q[2]] and q2.tolist() == [-q[0], -q[1], q[2]]
    same = solve_1d._one_side(np, wA, wB, q, q, mJ, mK)
    assert all(np.array_equal(a, q) for a in same)


def _gate_cases():
    """``(name, engages, (wA, wB, ky, q1, q2))`` for the cases above; ``q`` from
    ``sqrt(w + ky**2)`` where ``None``. ``engages`` is the expected answer, not the eager one."""
    out = []
    for ky in (0.3, 1.2):
        g = _gate(ky)
        out.append((f'inside, ky {ky}', True, ([0.9 * g, 0.7], [-0.9 * g, 0.5], ky, None, None)))
        out.append((f'outside, ky {ky}', False, ([1.1 * g, 0.7], [-1.1 * g, 0.5], ky, None, None)))
    ky = 1e-4
    out.append(('Rayleigh bound, inside', True,
                ([0.9 * ky ** 2 / 2, 0.5], [0.9 * ky ** 2 / 2, 0.5], ky, None, None)))
    out.append(('Rayleigh bound, outside', False,
                ([1.1 * ky ** 2 / 2, 0.5], [1.1 * ky ** 2 / 2, 0.5], ky, None, None)))
    ky, g = 0.3, _gate(0.3)
    wA, wB = [0.5 * g, 0.7], [0.5 * g, 0.5]
    q = list(np.sqrt(np.asarray(wA, dtype=complex) + ky ** 2))
    out.append(('a cluster', True, (wA, wB, ky, q, q)))
    out.append(('a root under the floor', False, (wA, wB, ky, [1e-10, q[1]], q)))
    out.append(('opposite labels', True, (wA, wB, ky, q, [-q[0], q[1]])))
    out.append(('ky 0, exact zero', False, ([0.0, 0.3], [0.0, 0.5], 0.0, None, None)))
    out.append(('ky 0, +-1e-300', False, ([1e-300, 0.3], [-1e-300, 0.5], 0.0, None, None)))
    out.append(('ky 0, roots above the floor', False,
                ([0.0, 0.3], [0.0, 0.5], 0.0, [0.5, 0.6], [0.5, 0.7])))
    return out


@pytest.mark.parametrize('name, engages, case', _gate_cases(),
                         ids=[name for name, _, _ in _gate_cases()])
def test_the_gate_under_jit_matches_the_eager_gate(name, engages, case):
    """Under ``jax.jit`` the gate hands its masks on as data: the eager masks where the gate
    engages (opposite labels included), all-False wherever it must not -- outside the edge, a
    root under the floor, ``ky == 0``.

    The only check that sees the refusal fold (a refused cluster's masks dropped to empty): the
    solver-level tests never build a refused cluster. Opposite labels are no longer a refusal
    (the rebuild relabels them), so they keep their masks under a trace as eagerly. Also pins the
    strict comparison at ``ky == 0`` under a trace, where nothing refuses afterwards.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    jnp = jax.numpy
    wA, wB, ky, q1, q2 = case
    wA, wB = np.asarray(wA, dtype=complex), np.asarray(wB, dtype=complex)
    q1 = np.sqrt(wA + ky ** 2) if q1 is None else np.asarray(q1, dtype=complex)
    q2 = np.sqrt(wB + ky ** 2) if q2 is None else np.asarray(q2, dtype=complex)
    ky = np.asarray(ky, dtype=complex)
    eager = solve_1d._cutoff_cluster(np, wA, wB, q1, q2, ky)
    traced = jax.jit(lambda *a: solve_1d._cutoff_cluster(jnp, *a))(wA, wB, q1, q2, ky)
    assert traced is not None, 'under a trace the gate must hand its masks on, never None'
    got = [np.asarray(m).tolist() for m in traced]
    if engages:
        assert eager is not None, f'{name}: the eager gate did not engage'
        want = [np.asarray(m).tolist() for m in eager]
        assert any(want[0] + want[1])
    else:
        want = [[False] * len(wA), [False] * len(wB)]
    assert got == want, f'{name}: traced {got}, expected {want}'


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


# ------------------------------- 3b. a cluster whose labels split (complex theta, double negative)

def _dng_grating(eps_bg, eps_fg, mu_bg, mu_fg):
    eps, mu = np.full((1, 1, 20), eps_bg), np.full((1, 1, 20), mu_bg)
    eps[0, 0, 6:14], mu[0, 0, 6:14] = eps_fg, mu_fg
    return dict(structure=eps, mu=mu)


_DNG_HOST = _dng_grating(-2.0, 2.25, -2.0, 1.0)     # glass pillars in a double-negative host
_SI_HOST = _dng_grating(12.25, -2.0, 1.0, -1.5)     # double-negative pillars in silicon
_SPLIT_BASE = dict(period=1000., thickness=[300.], top=1.0, bot=2.1025)

#: Lossless patterned double negatives at their own x-cutoff (root-found on the layer's own
#: matrices, ``min|wA|`` below 1e-14), lit at a complex ``theta``. With ``phi = pi/2`` the cutoff
#: does not depend on ``ky``, so the pair sits at ``q = +-ky``. Before 2026-10-12 the gate refused
#: each and kept the family construction, silently (largest error in ``(sum R, sum T, r, t)``
#: against the fit): a band member the flux labels the other way (ETM; 2.3 and 8.1), every root
#: a flux-broken tie at an imaginary ``theta`` (both algorithms; 9.3e-3), the cutoff pair itself
#: split by its flux labels at a relative 1e-12 (ETM; 6.6e-4).
SPLIT = {
    'DNG host, theta 0.02+1.2j': dict(_DNG_HOST, **_SPLIT_BASE, wavelength=589.736193646378,
                                      theta=0.02 + 1.2j, rel=0.0, algos=('ETM',)),
    'Si host, theta 0.02+1.2j': dict(_SI_HOST, **_SPLIT_BASE, wavelength=1396.5775562391,
                                     theta=0.02 + 1.2j, rel=0.0, algos=('ETM',)),
    'Si host, theta 1.2j': dict(_SI_HOST, **_SPLIT_BASE, wavelength=1396.5775562391,
                                theta=1.2j, rel=1e-12, algos=('ETM', 'SMM')),
    'DNG host, theta 0.02+0.8j': dict(_DNG_HOST, **_SPLIT_BASE, wavelength=589.736193646378,
                                      theta=0.02 + 0.8j, rel=1e-12, algos=('ETM',)),
}


def _quiet_complex(call):
    """Run ``call`` with every warning an error but the complex-incidence notice."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warnings.simplefilter('ignore', _legacy_wavebit_ComplexIncidenceWarning)
        return call()


@functools.lru_cache(maxsize=None)
def _split_fit(name):
    """:func:`_fit` for a :data:`SPLIT` scene (ETM family construction, the same 16 offsets)."""
    scene = SPLIT[name]
    with _family_construction(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Y = np.array([_vector(_solver(scene, wavelength=scene['wavelength'] * (1 + h))
                              .solve(auto_detune=False)) for h in _FIT_OFFSETS])
    return np.polyfit(1e3 * _FIT_OFFSETS, Y, 8)


def _split_spy(monkeypatch):
    """Spy on the gate: per call, whether two roots inside its band are labelled up and down."""
    seen = []
    gate = solve_1d._cutoff_cluster

    def spy(xp, wA, wB, q1, q2, ky):
        v1, v2 = (_np(q if q.ndim == 1 else np.diagonal(_np(q))) for q in (q1, q2))
        g = _gate(complex(_np(ky)))
        mJ, mK = np.abs(_np(wA)) < g, np.abs(_np(wB)) < g
        opposite = np.abs(v1[:, None] - v2[None, :]) >= np.abs(v1[:, None] + v2[None, :])
        seen.append(bool(np.any(opposite & mJ[:, None] & mK[None, :])))
        return gate(xp, wA, wB, q1, q2, ky)
    monkeypatch.setattr(solve_1d, '_cutoff_cluster', spy)
    return seen


@pytest.mark.parametrize('name', list(SPLIT))
def test_a_split_cluster_is_rebuilt_on_one_side(name, monkeypatch, rebuilds):
    """The rebuild takes a cluster whose labels disagree, puts it on one side, and is exact.

    The premise -- that the gate's band holds a pair labelled up and down -- is read off the
    gate's own input. Measured: 1.0e-10 .. 2.1e-9 from the fit (its own residual is 3.7e-10 ..
    3.5e-9 at these angles), ETM = SMM to 4.6e-14 .. 1.2e-11. Catches the refusal kept (2.3 ..
    8.1) and the rebuild without the relabelling (the pair-split scenes).
    """
    scene = SPLIT[name]
    truth = np.polyval(_split_fit(name), 1e3 * scene['rel'])
    split = _split_spy(monkeypatch)
    wavelength = scene['wavelength'] * (1 + scene['rel'])
    got = {}
    for algo in scene['algos']:
        del split[:], rebuilds[:]
        got[algo] = _vector(_quiet_complex(lambda: _solver(scene, algo=algo, wavelength=wavelength)
                                           .solve(auto_detune=False)))
        assert any(split), f'{name} {algo}: no split cluster -- the premise is gone'
        assert rebuilds, f'{name} {algo}: the rebuild did not engage'
        err = np.max(np.abs(got[algo] - truth))
        assert err < 1e-8, f'{name} {algo}: {err:.1e} from the fit'
    if len(got) == 2:
        assert np.max(np.abs(got['ETM'] - got['SMM'])) < 1e-11


@pytest.mark.parametrize('name', ['DNG host, theta 0.02+1.2j', 'Si host, theta 1.2j'])
def test_a_split_cluster_under_jit_and_vmap(name):
    """Under ``jax.jit`` and ``jax.vmap`` the masks travel as data and the relabelling with them.

    Before, a split cluster under a trace folded to empty masks: the family construction in the
    general layout, as wrong as eagerly (6.7e-3 and 38 from the eager solve). Measured: jitted
    values 3.9e-13 and 8.4e-13 from eager.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    jnp = jax.numpy
    scene = SPLIT[name]
    wl = scene['wavelength'] * (1 + scene['rel'])
    algo = scene['algos'][-1]
    s = _solver(scene, algo=algo, backend='jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', _legacy_wavebit_ComplexIncidenceWarning)
        eager = _vector(s.solve(wavelength=wl, auto_detune=False))
        jitted = _vector(jax.jit(lambda w: s.solve(wavelength=w, auto_detune=False))(jnp.asarray(wl)))
        lanes = jax.vmap(lambda w: jnp.sum(jnp.real(s.solve(wavelength=w, auto_detune=False).R)))(
            jnp.asarray([wl, wl * (1 + 1e-12)]))
    assert np.max(np.abs(jitted - eager)) < 1e-10
    assert abs(float(lanes[0]) - eager[0].real) < 1e-10


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

    The 2D path is no oracle there (``docs/theory/rcwa/rayleigh-points.md`` section 7: 1e-8 .. 4e-5 off), so
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

    Passed in, the record's ``q`` is a tracer, so the coupled exponential's norm is abstract and
    its tangent is the closed-form Frechet derivative
    (:func:`~wavebit.rcwa._engine.modal_gradient._coupled_frechet`); eagerly ``q`` is a constant and the
    formula's exact ``scale`` derivative is used. Measured 9.9e-17 in ``d/dz``. Before, ``_expm``
    formed its squaring count traced there (4.1e-16 .. 8.0e-16), and before that fell back to
    ``jax.scipy.linalg.expm``'s own floor count (6.4e-11).
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


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_uniform_cutoff_field_under_jit_matches_the_2d_path(algo, rebuilds):
    """R1's field from ``jax.jit(solve + reconstruct_field)`` at the exact point: the 2D field.

    Before, the traced solve kept the family construction and its field was NaN here. Measured
    <= 8.6e-15 of ``max|F|``, the cell-centre phase removed as in the eager test above.
    """
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    ref = _field(R1, two_d=True)
    s = _solver(R1, algo=algo, backend='jax')
    got = np.asarray(jax.jit(lambda w: s.solve(wavelength=w, auto_detune=False, answers='field')
                             .reconstruct_field(sample_x=12, sample_y=1, sample_z=8))(
        jnp.asarray(500.)))
    assert rebuilds
    phase = np.exp(1j * np.angle(np.vdot(got.ravel(), ref.ravel())))
    err = np.max(np.abs(got * phase - ref)) / np.max(np.abs(ref))
    assert np.isfinite(err) and err < 1e-12, f'{algo}: {err:.1e}'


@pytest.mark.parametrize('name', list(_PATTERNED_FIELDS))
def test_a_patterned_cutoff_field_under_jit(name):
    """``jax.jit(solve + reconstruct_field)`` at an exceptional point is the eager field, SMM's
    is the ETM's, and the field gradient under ``jax.jit(jax.grad)`` is the eager one.

    Under a trace every layer is rebuilt, so SMM's field state reads it with
    ``two_face(coupled=True)`` and the fields path takes the coupled exponential's closed-form
    tangent. Measured: fields <= 5.1e-14 of ``max|F|`` (MU614; the others <= 6.3e-15), SMM vs
    ETM <= 3.2e-15, gradients in the raster and the wavelength <= 1.2e-13; before, 8.6e-3 .. 2.5,
    1.6 .. 2.0 and 7e11 .. 3e28.
    Catches the field state reading a coupled traced layer with plain ``two_face``
    (2.1e-3 .. 4.1e-3) and the fields path taking the formula's tangent.
    """
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    scene = _PATTERNED_FIELDS[name]
    xs = jnp.asarray([-200., 0., 130.])
    zs = jnp.asarray(np.array([0.13, 0.5, 0.87]) * sum(scene['thickness']))
    u0 = jnp.asarray(scene['structure'], dtype=float)
    w0 = jnp.asarray(scene['wavelength'])
    jitted = {}
    for algo in ('ETM', 'SMM'):
        s = _solver(scene, algo=algo, backend='jax')

        def field(u, w):
            r = s.solve(structure=u, wavelength=w, auto_detune=False, answers='field')
            return r.reconstruct_field(sample_x=xs, sample_y=jnp.asarray([0.]), sample_z=zs)

        def F(u, w):
            return jnp.sum(jnp.abs(field(u, w)[..., :3]) ** 2)
        eager = np.asarray(field(u0, w0))
        jitted[algo] = np.asarray(jax.jit(field)(u0, w0))
        err = np.max(np.abs(jitted[algo] - eager)) / np.max(np.abs(eager))
        assert np.isfinite(err) and err < 1e-12, f'{algo} jit field: {err:.1e} from eager'
        g_eager = jax.grad(F, argnums=(0, 1))(u0, w0)
        g_jit = jax.jit(jax.grad(F, argnums=(0, 1)))(u0, w0)
        for var, a, b in zip(('raster', 'wavelength'), g_jit, g_eager):
            a, b = np.asarray(a), np.asarray(b)
            err = np.max(np.abs(a - b)) / np.max(np.abs(b))
            assert np.isfinite(err) and err < 1e-11, f'{algo} jit(grad) {var}: {err:.1e}'
    scale = np.max(np.abs(jitted['ETM']))
    err = np.max(np.abs(jitted['SMM'] - jitted['ETM'])) / scale
    assert err < 1e-12, f'jitted SMM field {err:.1e} from the jitted ETM field'


# ------------------------------------------- 8. off the band, and under every abstract trace

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
    """R1 at ``rel = 5e-2`` and random scenes: eagerly no rebuild and every bit the gate-off
    solve's; under ``jax.jit`` the rebuild in the general layout, at the eager answer.

    A traced conical layer cannot be gated, so it takes the general layout whatever its masks
    (all empty off the band) and is rounded differently from the eager solve: under a trace the
    test pins accuracy, not bits (measured <= 1.6e-14 of the largest output). The rebuild must
    run there -- a traced gate that answered ``None``, the old behaviour, fails it.
    """
    scenes = [dict(R1, wavelength=500. * 1.05)] + _random_scenes(4)
    for scene in scenes:
        got = _numbers(_solver(scene, algo=algo).solve(auto_detune=False))
        with _family_construction():
            old = _numbers(_solver(scene, algo=algo).solve(auto_detune=False))
        for key in got:
            assert np.array_equal(got[key], old[key]), key
    assert not rebuilds
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    for index, scene in enumerate(scenes):
        ref = _flat(_numbers(_solver(scene, algo=algo).solve(auto_detune=False)))
        s = _solver(scene, algo=algo, backend='jax')
        del rebuilds[:]
        r = jax.jit(lambda u: s.solve(structure=u, auto_detune=False))(
            jnp.asarray(scene['structure'], dtype=float))
        assert rebuilds, f'scene {index}: no rebuild under jax.jit'
        err = np.max(np.abs(_flat(_numbers(r)) - ref)) / np.max(np.abs(ref))
        assert err < 1e-12, f'scene {index}: jit {err:.1e} from the eager solve'


def _flat(n):
    return np.concatenate([n['R'], n['T'], n['r'], n['t']])


def test_under_an_abstract_trace_the_layer_is_rebuilt(rebuilds):
    """Every abstract trace rebuilds R1's spacer: the gate hands its masks on as data.

    ``jax.jit``, ``jax.vmap`` (the sweep idiom), ``jit(vmap)``, ``jax.lax.map``,
    ``jax.checkpoint`` and ``torch.func.vmap``, at the exact point, one ULP either side,
    ``+1e-12`` and ``+1e-3`` (off the band, in the same trace), against the 2D path. Measured
    <= 4.4e-14 in amplitudes; before, the family construction shipped under every one of them:
    NaN at the point and 0.27 .. 0.78 one ULP away, silently. One ``jit`` trace serves every
    wavelength. Catches the gate answering ``None`` under a trace (NaN), ``torch.func.vmap``
    left on the family construction, and the refusal fold applied to every layer, i.e. masks
    always empty (NaN).
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    jnp = jax.numpy
    wls = list(_offsets(500.).values()) + [500. * (1 + 1e-3)]
    for algo in ('ETM', 'SMM'):
        ref = np.stack([_flat(_numbers(_solver(R1, algo=algo, two_d=True, wavelength=w)
                                       .solve(auto_detune=False), two_d=True)) for w in wls])
        s = _solver(R1, algo=algo, backend='jax')
        traces = []

        def one(w):
            traces.append(1)
            r = s.solve(wavelength=w, auto_detune=False)
            return jnp.concatenate([jnp.real(r.R).ravel(), jnp.real(r.T).ravel(),
                                    r.r_xyz.ravel(), r.t_xyz.ravel()])
        sweep = jnp.asarray(wls)
        jf, ck = jax.jit(one), jax.checkpoint(one)
        got = {'jit': np.stack([np.asarray(jf(w)) for w in sweep])}
        assert len(traces) == 1, 'one jit trace serves on- and off-band wavelengths'
        got.update({'vmap': np.asarray(jax.vmap(one)(sweep)),
                    'jit(vmap)': np.asarray(jax.jit(jax.vmap(one))(sweep)),
                    'lax.map': np.asarray(jax.lax.map(one, sweep)),
                    'checkpoint': np.stack([np.asarray(ck(w)) for w in sweep])})
        st = _solver(R1, algo=algo, backend='torch')
        got['torch.func.vmap'] = torch.func.vmap(lambda w: (lambda r: torch.cat([
            r.R.real.reshape(-1), r.T.real.reshape(-1), r.r_xyz.reshape(-1),
            r.t_xyz.reshape(-1)]))(st.solve(wavelength=w, auto_detune=False)))(
            torch.tensor(wls, dtype=torch.float64)).numpy()
        for name, g in got.items():
            err = np.max(np.abs(g - ref))
            assert np.isfinite(err) and err < 1e-12, f'{algo} {name}: {err:.1e}'
    assert rebuilds


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', ['PA616', 'MU614'])
def test_a_patterned_cutoff_under_an_abstract_trace(name, algo, rebuilds):
    """At a patterned exceptional point every trace gives the eager answer, derivatives too.

    Values under ``jit`` and ``jit(vmap)`` at ``rel`` 0 and ``+-1e-12`` against the eager NumPy
    rebuild (measured <= 3.6e-14; before 3.9e-3 .. 1.5). ``d(sum R)`` in the wavelength, the
    raster and the thickness under ``jit(grad)`` and ``grad(checkpoint)``, ``jit(jvp)`` in the
    wavelength, and the sweep idioms ``jit(grad(sum vmap))`` and ``jit(vmap(grad))`` per lane,
    against eager ``jax.grad`` (measured <= 2.0e-12, MU614's thickness under ``jit``; PA616
    <= 3.3e-14; before 1e10 .. 3e28). The thickness is the one parameter that reaches the
    coupled exponential's ``scale``: without the traced rule's ``ds q exp(s q)`` term every
    thickness derivative here is lost (relative error 1.0). Also
    catches the closed form replaced by the formula's own tangent (6.1e-4 PA616, 6.4e-3 MU614),
    either second-order term dropped (3.1e-4 .. 3.3e-2), divided differences by the series
    alone (4.7e-3 .. 28) and a coupling cap of 0 (NaN).
    """
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    scene = PATTERNED[name]
    wl0 = scene['wavelength']
    wls = [wl0, wl0 * (1 - 1e-12), wl0 * (1 + 1e-12)]
    ref = np.stack([_flat(_numbers(_solver(scene, algo=algo, wavelength=w)
                                   .solve(auto_detune=False))) for w in wls])
    s = _solver(scene, algo=algo, backend='jax')

    def one(w):
        r = s.solve(wavelength=w, auto_detune=False)
        return jnp.concatenate([jnp.real(r.R).ravel(), jnp.real(r.T).ravel(),
                                r.r_xyz.ravel(), r.t_xyz.ravel()])
    sweep = jnp.asarray(wls)
    jf = jax.jit(one)
    del rebuilds[:]
    values = {'jit': np.stack([np.asarray(jf(w)) for w in sweep]),
              'jit(vmap)': np.asarray(jax.jit(jax.vmap(one))(sweep))}
    assert rebuilds
    for label, got in values.items():
        err = np.max(np.abs(got - ref))
        assert np.isfinite(err) and err < 1e-12, f'{label}: {err:.1e}'

    u0 = jnp.asarray(scene['structure'], dtype=float)
    t0 = jnp.asarray(scene['thickness'][0])

    def F(w, u, t):
        r = s.solve(wavelength=w, structure=u, thickness=jnp.stack([t]), auto_detune=False)
        return jnp.sum(jnp.real(r.R))
    args = (jnp.asarray(wl0), u0, t0)
    eager = jax.grad(F, argnums=(0, 1, 2))(*args)
    traced = {'jit(grad)': jax.jit(jax.grad(F, argnums=(0, 1, 2)))(*args),
              'grad(checkpoint)': jax.grad(jax.checkpoint(F), argnums=(0, 1, 2))(*args)}
    for label, g in traced.items():
        for var, a, b in zip(('wavelength', 'raster', 'thickness'), g, eager):
            err = _rel(a, b)
            assert np.isfinite(err) and err < 1e-11, f'{label} {var}: {err:.1e}'
    jvp = jax.jit(lambda w: jax.jvp(lambda x: F(x, u0, t0), (w,), (jnp.asarray(1.0),))[1])(
        args[0])
    err = _rel(jvp, eager[0])
    assert np.isfinite(err) and err < 1e-11, f'jit(jvp) wavelength: {err:.1e}'

    def Fw(w):
        return F(w, u0, t0)
    lanes = np.array([float(jax.grad(Fw)(w)) for w in sweep])
    sweeps = {'jit(grad(sum vmap))': jax.jit(jax.grad(lambda ws: jnp.sum(jax.vmap(Fw)(ws))))(sweep),
              'jit(vmap(grad))': jax.jit(jax.vmap(jax.grad(Fw)))(sweep)}
    for label, g in sweeps.items():
        err = np.max(np.abs(np.asarray(g) - lanes) / np.abs(lanes))
        assert np.isfinite(err) and err < 1e-11, f'{label}: {err:.1e}'


#: One ``torch.func.vmap`` batch per scene: the lane off the band (R1 at 520 nm, a patterned
#: point at ``rel +1e-2``) and the raster pixel the derivative takes (R1's in its rebuilt
#: spacer, a patterned one at the pillar's edge).
_TORCH_VMAP_SCENES = {
    'R1': (R1, 520., (0, 0, 3)),
    'PA616': (PATTERNED['PA616'], PATTERNED['PA616']['wavelength'] * (1 + 1e-2), (0, 0, 6)),
    'MU614': (PATTERNED['MU614'], PATTERNED['MU614']['wavelength'] * (1 + 1e-2), (0, 0, 6)),
}


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(_TORCH_VMAP_SCENES))
def test_torch_autograd_through_vmap_at_a_cutoff(name, algo, monkeypatch):
    """``torch.autograd`` through ``torch.func.vmap`` at a cutoff is eager autograd, lane by lane.

    One ``vmap`` over the wavelength at ``rel`` -1e-12, 0, +1e-12 and a lane off the band, a
    raster pixel ``p`` captured with ``requires_grad``. ``vmap(f)(wl).sum().backward()`` must
    give each lane's eager ``dR/dwl`` and the lanes' summed eager ``dR/dp``; ``forward_ad``
    through the ``vmap`` -- a batched dual wavelength, then a captured dual pixel -- each lane's
    eager tangent. Under the ``vmap`` the masks are data, so every lane takes the rebuild's AD
    build (``_regular_conical_modes_ad``) and its coupled exponential's ``expm`` tangent,
    batched; both are spied under the transform. Measured <= 7.3e-13 against eager; with the
    eigen-derivative missing under ``vmap``, as before 2026-10-11, 0.2 .. 3.3 relative off in
    band, silently, and ``forward_ad`` through ``vmap`` raised. Catches the coupled exponential
    taking the formula's tangent under a ``torch.func`` transform -- PA616 6.1e-4 (wavelength)
    and 1.0e-2 .. 1.1e-2 (pixel), MU614 6.4e-3 and 7.1e-3 .. 1.0e-2 -- which every other test
    here and in ``test_torch_vmap_autograd.py`` passes. R1's uniform spacer couples nothing, so
    the formula's tangent is exact there and only the spy sees it.
    """
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    f64 = torch.float64
    scene, off_band, pixel = _TORCH_VMAP_SCENES[name]
    wl0 = scene['wavelength']
    lanes = [wl0 * (1 - 1e-12), wl0, wl0 * (1 + 1e-12), off_band]
    base = torch.tensor(np.asarray(scene['structure'], dtype=float), dtype=f64)
    bump = torch.zeros_like(base)
    bump[pixel] = 1.0
    s = _solver(scene, algo=algo, backend='torch')

    def f(w, p):
        return s.solve(wavelength=w, structure=base + p * bump, auto_detune=False).R.real.sum()

    def scalar(x, grad=False):
        return torch.tensor(x, dtype=f64, requires_grad=grad)

    eager_rev, eager_fwd = [], []
    for w in lanes:
        wt, pt = scalar(w, True), scalar(0., True)
        f(wt, pt).backward()
        eager_rev.append([float(wt.grad), float(pt.grad)])
        tangents = []
        for which in (0, 1):
            with forward_ad.dual_level():
                args = [scalar(w), scalar(0.)]
                args[which] = forward_ad.make_dual(args[which], scalar(1.))
                tangents.append(float(forward_ad.unpack_dual(f(*args)).tangent))
        eager_fwd.append(tangents)
    eager_rev, eager_fwd = np.array(eager_rev), np.array(eager_fwd)

    under = {'_regular_conical_modes_ad': 0, '_expm': 0}
    for module, attr in ((solve_1d, '_regular_conical_modes_ad'), (modal_gradient, '_expm')):
        def spy(*args, _attr=attr, _original=getattr(module, attr), **kwargs):
            under[_attr] += bool(torch._C._are_functorch_transforms_active())
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, attr, spy)

    wl = torch.tensor(lanes, dtype=f64, requires_grad=True)
    p = scalar(0., True)
    torch.func.vmap(lambda w: f(w, p))(wl).sum().backward()
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(torch.tensor(lanes, dtype=f64), torch.ones(4, dtype=f64))
        fwd_wl = forward_ad.unpack_dual(
            torch.func.vmap(lambda w: f(w, scalar(0.)))(dual)).tangent.numpy()
    with forward_ad.dual_level():
        captured = forward_ad.make_dual(scalar(0.), scalar(1.))
        fwd_p = forward_ad.unpack_dual(torch.func.vmap(lambda w: f(w, captured))(
            torch.tensor(lanes, dtype=f64))).tangent.numpy()

    def lane_rel(got, want):
        return float(np.max(np.abs(np.asarray(got) - want) / np.abs(want)))
    errors = {
        'vmap + backward, wavelength per lane': lane_rel(wl.grad.numpy(), eager_rev[:, 0]),
        'vmap + backward, captured pixel, lanes summed': lane_rel(float(p.grad),
                                                                   eager_rev[:, 1].sum()),
        'forward_ad + vmap, wavelength per lane': lane_rel(fwd_wl, eager_fwd[:, 0]),
        'forward_ad + vmap, captured pixel per lane': lane_rel(fwd_p, eager_fwd[:, 1]),
    }
    off = {label: f'{err:.1e}' for label, err in errors.items()
           if not (np.isfinite(err) and err < 1e-11)}
    assert not off, off
    assert all(under.values()), f'calls under the transform: {under}'


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_second_derivatives_through_a_traced_cutoff(algo):
    """PA616 under ``jax.jit``: ``d2(sum R)/dt2`` and the mixed thickness x raster derivative in
    both orders (``jacfwd`` of a ``grad``, ``grad`` of a ``jvp``) and reverse over reverse
    (``jacrev`` of a ``grad``), against eager.

    The traced rule's tangent is ``L(s q, s dq) + ds q exp(s q)`` with ``L`` at the stopped
    ``q``, so an outer derivative must still find every term: an outer ``scale`` moves ``L``
    along ``q``, an outer ``q`` reaches ``q exp(s q)``. Measured <= 7.7e-14; without the
    ``ds q exp(s q)`` term 0.57 .. 1.0.

    Reverse over reverse across the two variables -- the raster inside, the thickness outside --
    is the one probe here that sends a cotangent into the closed form's divided differences
    through ``scale``, to every lane, the exactly confluent ones included (``F1``'s diagonal,
    the padded ``J = K = 0`` sets): measured 1.5e-14 .. 1.7e-14, and NaN with either divisor
    guard dropped (``_dd_from_subsets``'s, ``F1``'s), where the other three stay finite and
    exact. (``d2/dt2`` is reverse over reverse too, but in ``t`` alone the variation ``dq`` is
    zero, so no cotangent reaches ``L``.)
    """
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    scene = PATTERNED['PA616']
    s = _solver(scene, algo=algo, backend='jax')
    u0 = jnp.asarray(scene['structure'], dtype=float)
    t0 = jnp.asarray(scene['thickness'][0])
    one = jnp.asarray(1.0)

    def F(u, t):
        r = s.solve(structure=u, thickness=jnp.stack([t]), auto_detune=False)
        return jnp.sum(jnp.real(r.R))
    probes = {
        'd2/dt2': lambda u, t: jax.grad(jax.grad(F, argnums=1), argnums=1)(u, t),
        'd/dt (d/du), jacfwd(grad)': lambda u, t: jax.jacfwd(
            lambda x: jax.grad(F, argnums=0)(u, x))(t),
        'd/du (d/dt), grad(jvp)': lambda u, t: jax.grad(
            lambda v: jax.jvp(lambda x: F(v, x), (t,), (one,))[1])(u),
        'd/dt (d/du), jacrev(grad)': lambda u, t: jax.jacrev(
            lambda x: jax.grad(F, argnums=0)(u, x))(t),
    }
    for label, probe in probes.items():
        eager = np.asarray(probe(u0, t0))
        jitted = np.asarray(jax.jit(probe)(u0, t0))
        err = np.max(np.abs(jitted - eager)) / np.max(np.abs(eager))
        assert np.isfinite(err) and err < 1e-11, f'{label}: {err:.1e}'


# ------------------------------------------------------------------ 9. ky = 0 untouched

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_the_ky_zero_rayleigh_point_still_refuses(algo, rebuilds, monkeypatch):
    """At ``theta = phi = 0`` the same order is a Rayleigh point: refused, never rebuilt.

    ``kx_2 = 1`` on an air spacer whose operator is exactly diagonal, so the zero is exact on
    any LAPACK. The gate is consulted and declines (``ky == 0``), the refusal is the one
    ``check_finite`` always gave, and the default ladder still rescues it with a notice.
    """
    from wavebit._common.backend import AutoDetuneWarning
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

@pytest.mark.parametrize('backend', ['jax', 'jax-jit', 'torch'])
def test_the_coupled_exponential_is_the_matrix_exponential(backend):
    """``exponential(coupled=True)``: the value from the formula, the derivative exact.

    A propagator shaped like a rebuilt layer's (two cluster pairs coupled, at ``|A|_1 ~ 10``,
    where ``jax.scipy.linalg.expm`` unscaled was 4e-9 off), differentiated in a direction with
    entries everywhere, against a central difference of SciPy's ``expm``: eager JAX and PyTorch
    take the tangent from ``expm``, ``jax.jit`` from the closed-form Frechet derivative. The
    formula's own derivative misses the coupling chained behind the variation;
    ``coupled=False`` must show it, traced or not.
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
    if backend.startswith('jax'):
        jax = pytest.importorskip('jax')
        jnp = jax.numpy

        def run(coupled):
            def f(t):
                return modal_gradient.exponential(
                    jnp, jnp.asarray(q) + t * jnp.asarray(direction), scale, coupled=coupled)

            def pushed(t):
                return jax.jvp(f, (t,), (jnp.asarray(1.0, dtype=jnp.complex128),))
            if backend == 'jax-jit':
                pushed = jax.jit(pushed)
            value, tangent = pushed(jnp.asarray(0.0, dtype=jnp.complex128))
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


#: Couplings of a synthetic ``q`` (``m = 10``): ``(rows x columns)`` entries, and which diagonal
#: entry of a column is moved next to a row's (``(k, j, gap)``: ``q_kk = q_jj + gap``).
_COUPLED_SHAPES = {
    'tau = 0': ([], []),
    'one pair, gap 0.3': ([(1, 6)], [(6, 1, 0.3)]),
    'an exact confluent pair': ([(1, 6)], [(6, 1, 0.0)]),
    'a pair at a gap of 2e-15': ([(1, 6)], [(6, 1, 2e-15 * (1 + 1j))]),
    'two pairs, a full 2x2 coupling': ([(0, 5), (0, 6), (1, 5), (1, 6)],
                                       [(5, 0, 1e-9), (6, 1, 0.0)]),
    'four rows (the cap)': ([(j, k) for j in range(4) for k in range(4, 8)],
                            [(4, 0, 0.0), (5, 1, 0.1)]),
    'five rows (past the cap)': ([(j, 5 + j) for j in range(5)], []),
    'a row that is also a column': ([(1, 5), (5, 6)], []),
}

#: The shapes the closed form refuses, each a half of its ``ok``: more coupled rows than
#: :data:`~wavebit.rcwa._engine.modal_gradient._COUPLED_CAP`, and its premise broken (row 1 couples to
#: column 5, and 5 couples on to column 6: a row that is also a column, two couplings chained).
_REFUSED_SHAPES = ('five rows (past the cap)', 'a row that is also a column')


def _coupled_q(shape, m=10, seed=5):
    """A diagonal-plus-coupling ``q`` and a direction ``E`` with entries everywhere."""
    rng = np.random.default_rng(seed)
    q = np.diag(rng.uniform(0.2, 1.6, m) + 1j * rng.uniform(0, 0.3, m))
    entries, moves = _COUPLED_SHAPES[shape]
    for k, j, gap in moves:
        q[k, k] = q[j, j] + gap
    for j, k in entries:
        q[j, k] = rng.normal() * 0.6 + 1j * rng.normal() * 0.4
    return q, rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))


def _coupled_tangent(q, E, scale, jit):
    """``(value, tangent)`` of ``exponential(q + t E, scale, coupled=True)`` at ``t = 0``."""
    import jax
    jnp = jax.numpy

    def pushed(t):
        return jax.jvp(lambda x: modal_gradient.exponential(
            jnp, jnp.asarray(q) + x * jnp.asarray(E), scale, coupled=True),
            (t,), (jnp.asarray(1.0 + 0j),))
    value, tangent = (jax.jit(pushed) if jit else pushed)(jnp.asarray(0j))
    return np.asarray(value), np.asarray(tangent)


@pytest.mark.parametrize('shape', [s for s in _COUPLED_SHAPES if s not in _REFUSED_SHAPES])
def test_the_coupled_exponential_tangent_under_a_trace_is_the_frechet_derivative(shape):
    """Under ``jax.jit`` the coupled exponential's tangent is ``L(s q, s E)``, SciPy's
    ``expm_frechet``, to 1e-13 of its largest entry -- confluent, near-confluent and spread
    roots, one coupling to the cap of four -- at ``s`` = -5, -0.7, 0 and -12+3i and for a
    batched ``s``; and it is the eager (``expm``) tangent to the same 1e-13. Measured
    <= 1.2e-15 against SciPy, <= 8.7e-16 against the eager tangent.

    The closed form's fourth term (coupling, direction, coupling) is invisible at solver level,
    since a physical tangent's ``K x J`` entries vanish with the root gap: only this check and
    the ``jax-jit`` case of ``test_the_coupled_exponential_is_the_matrix_exponential`` see it
    dropped (1.3e-3). Also catches divided differences by the recurrence alone (6.8e9 at a pair
    2e-15 apart) or by the series alone (0.15).
    """
    jax = pytest.importorskip('jax')
    scipy_linalg = pytest.importorskip('scipy.linalg')
    _legacy_wavebit_enable_x64()
    jnp = jax.numpy
    q, E = _coupled_q(shape)
    for s in (-5.0, -0.7, 0.0, -12 + 3j):
        value, traced = _coupled_tangent(q, E, s, jit=True)
        ref = scipy_linalg.expm_frechet(s * q, s * E, compute_expm=False)
        exact = scipy_linalg.expm(s * q)
        assert np.max(np.abs(value - exact)) <= 1e-13 * np.max(np.abs(exact)), f's {s}: value'
        err = np.max(np.abs(traced - ref))
        assert err <= 1e-13 * np.max(np.abs(ref)), (
            f's {s}: {err / max(np.max(np.abs(ref)), 1e-300):.1e} from expm_frechet')
        _, eager = _coupled_tangent(q, E, s, jit=False)
        assert np.max(np.abs(traced - eager)) <= 1e-13 * np.max(np.abs(ref)), f's {s}: eager'
    batch = np.array([-5.0, 0.0, -0.7, -12 + 3j])
    _, traced = _coupled_tangent(q, E, jnp.asarray(batch), jit=True)
    for lane, s in enumerate(batch):
        ref = scipy_linalg.expm_frechet(s * q, s * E, compute_expm=False)
        assert np.max(np.abs(traced[lane] - ref)) <= 1e-13 * np.max(np.abs(ref)), f'lane {s}'


@pytest.mark.parametrize('shape', _REFUSED_SHAPES)
def test_a_refused_coupling_gives_a_nan_traced_tangent(shape):
    """A coupling the closed form refuses gives a traced tangent NaN in every entry -- loud,
    never a silently partial derivative -- and leaves the value alone.

    The two halves of the closed form's ``ok``. Five coupled rows (the cap is four): the value
    is still exact. A row that is also a column (``_coupled_frechet``'s premise broken; no
    rebuilt layer forms one): the value is the divided-difference formula's, finite, and no
    longer ``exp`` (a product of two couplings survives: 5.7e-2 off), so it is checked against
    the formula. Nothing else in this file reaches either refusal: a refusal made silent, or
    the premise check dropped, fails here alone.
    """
    jax = pytest.importorskip('jax')
    scipy_linalg = pytest.importorskip('scipy.linalg')
    _legacy_wavebit_enable_x64()
    assert modal_gradient._COUPLED_CAP == 4
    q, E = _coupled_q(shape)
    value, tangent = _coupled_tangent(q, E, -5.0, jit=True)
    assert np.all(np.isfinite(value))
    if shape == 'a row that is also a column':
        expected = modal_gradient.exponential(np, q, -5.0)
        off = np.max(np.abs(scipy_linalg.expm(-5.0 * q) - expected))
        assert off > 1e-2 * np.max(np.abs(expected)), 'premise: two couplings chain'
    else:
        expected = scipy_linalg.expm(-5.0 * q)
    assert np.max(np.abs(value - expected)) <= 1e-13 * np.max(np.abs(expected))
    assert np.all(np.isnan(tangent))


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_eager_takes_expm_and_a_trace_the_closed_form(algo, monkeypatch):
    """Which tangent rule runs where, at PA616: eager ``jax.grad`` calls ``_expm`` and never the
    closed form, ``jax.jit(jax.grad)`` the closed form and never ``_expm``.

    The two rules give eager gradients 2.4e-14 apart, below every tolerance here, so a dispatch
    reversed, or the closed form serving eager solves too, would move eager bits at every rebuilt
    point and fail nothing else.
    """
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    calls = {'_expm': 0, '_coupled_frechet': 0}
    for name in calls:
        def spy(*args, _name=name, _original=getattr(modal_gradient, name), **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(modal_gradient, name, spy)
    scene = PATTERNED['PA616']
    s = _solver(scene, algo=algo, backend='jax')

    def F(w):
        return jnp.sum(jnp.real(s.solve(wavelength=w, auto_detune=False).R))
    w0 = jnp.asarray(scene['wavelength'])
    jax.grad(F)(w0)
    assert calls['_expm'] and not calls['_coupled_frechet'], f'eager: {calls}'
    calls.update({name: 0 for name in calls})
    jax.jit(jax.grad(F))(w0)
    assert calls['_coupled_frechet'] and not calls['_expm'], f'jit: {calls}'
