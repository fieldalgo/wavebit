"""The exact return of one PML face (``src/wavebit/fdtd/_prepare/layer_return.py``), held to the kernel.

The grazing rule judges a stabilized face by this model, so the model has to be the kernel and
not an approximation of it. Three gates, each with a mutant it catches:

**(a) It is the kernel's own steady state.** A 1 x 1 column along z (``Bloch`` or ``Axis1D``
across x, the two z faces the layer under test, an ``eps = 2.25`` background), its one-step
matrix ``M`` (``fd_helpers.one_step_matrix``: E, H and every psi, stepped once through
``_kernel.step``), driven at ``exp(-i omega dt)`` by a unit ``E_x`` and ``E_y`` at the middle
node: ``X = (z I - M)^-1 f``. Between the source and each face the column is source-free and
uniform, so its tangential E is ``A lambda^k + B lambda^-k`` with the interior's discrete
``lambda``; the ratio of the outgoing to the incoming amplitude at the layer's inner face is the
kernel's return. Measured at 0 / 30 / 60 / 80 degrees, 12 and 24 cells, ``rho`` 0 and 3e-3,
both faces, TE (``E_y``) and TM (``E_x``), a Bloch and an Axis1D axis: the model agrees to
4.7e-14 of the incident wave or better -- 3.4e-11 relative or better wherever the return is
above 1e-4, and 6e-8 relative at worst, on the 24-cell layer's 2.1e-8 at normal incidence, which
is the driven solve's round-off (1e-15 of the incident wave) and not the model. Also at
an off-design wavelength, a layer matched to a medium it is not in, a magnetic medium, and an
order with both ``k_x`` and ``k_y``. The gate line is ``1e-9 * max(|r|, 1e-4)``. Mutants,
measured at 12 cells, 60 degrees, ``rho = 3e-3`` (return 1.4e-3 / 1.5e-3 TE / TM): the loss put
on the tangential components too moves the model by 2.1e-3; the wall a cell off, by 3.0e-3
(on the pure stretch it returns 3.3e-3 where the layer returns 1.0e-4); the CFS pole dropped
from the model alone, by 8.2e-4 -- each 5e8 times the gate line or more.

**(b) It reproduces the doubling tables** of ``tests/fdtd/test_fd_pml.py`` (``TABLE`` and
``STABILIZED``, every entry) within that probe's own settling error. The probe's window is
finite, and the part of the packet's band that arrives at a worse angle than the column's
leaks into the carrier bin: at the default window the tables sit 0.81 .. 1.45 times the model
(worst: 16 cells, 60 degrees, pure, 1.26e-4 against 8.68e-5). Lengthening the probe's window
(``tail=6``, ``extent=3.5``, ``delay=8``) moves every probed entry onto the model: 0.95 .. 1.02
(12 / 16 cells at 60 degrees pure, 24 cells at 30 / 60 degrees stabilized, 12 cells at 0) --
so the gap is the probe settling, and :data:`SETTLE` is its line. Mutant: the CFS pole dropped
puts the 12-cell normal-incidence entry at 7.3e-7 against the table's 9.2e-5.

**(c) For the pure stretch it approaches the continuum ``R_c`` from below** as ``dx -> 0`` at a
fixed physical thickness (1.2 wavelengths), first order, as ``studies/fdtd/grazing_return/fdtd_grazing_return.py``
measured it: at 1e-12 and ``cos 0.26``, 6.21e-4 / 9.09e-4 / 1.110e-3 at 20 / 40 / 80 cells per
wavelength against the 6.1e-4 / 9.1e-4 / 1.11e-3 ``tests/fdtd/test_fd_grazing.py`` measures on a
run, and 1.325e-3 at 640 against ``R_c`` 1.359e-3. Mutant: the CFS pole dropped converges to
``reflection ** cos theta`` instead (8.3e-3 at 1e-8, ``cos 0.26``, against ``R_c`` 1.23e-2).

**(d) Many orders in one march.** ``K`` as an array (a face's orders at one wavelength, as the
grazing rule passes them) gives each order's one-order result to 2.1e-11 relative, at 0.76 ms
for 1000 orders against 0.09 ms for one; an order the grid makes evanescent is refused.
"""
import cmath
import functools
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._prepare import cpml, grazing
from wavebit.fdtd._prepare import layer_return as model
from wavebit.fdtd._prepare.layer_return import grid_k, layer_return
from tests.fixtures.fdtd.fd_helpers import doubling_reflection, one_step_matrix, packet_layout

DX = 0.05
EPS = 2.25
ANGLES = (0, 30, 60, 80)
#: The model against the kernel: relative, with an absolute floor at the driven solve's
#: round-off (1e-15 of the incident wave, measured) where the return itself is tiny.
EXACT = 1e-9
FLOOR = 1e-4
#: How far the doubling probe's default window may sit from the model: measured 0.81 .. 1.45.
SETTLE = 1.6


# ---------------------------------------------------------------------------------------------
# (a) The kernel's own steady state
# ---------------------------------------------------------------------------------------------

def _column(cells, rho, axis1d, kx, ky=0.0, eps=EPS, mu=None, match=None, interior=16):
    """The 1 x 1 column along z: both z faces ``PML(cells, stabilize=rho)``, the medium given."""
    nz = 2 * cells + interior
    if axis1d:
        x = fd.Axis1D(k=kx, k_max=max(abs(kx), 1.0))
    else:
        x = fd.Bloch(phase=kx * DX) if kx else fd.Periodic()
    y = fd.Bloch(phase=ky * DX) if ky else fd.Periodic()
    pml = fd.PML(cells=cells, wavelength=1.0, stabilize=rho or None, background=match)
    return fd.Scene(domain=fd.Domain(size=(DX, DX, nz * DX)),
                    boundaries=fd.Boundaries(x=x, y=y, z=pml),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=1),
                    background=fd.Material(eps=eps, mu=mu)), nz


@functools.lru_cache(maxsize=None)
def _matrix(cells, rho, axis1d, kx, ky=0.0, eps=EPS, mu=None, match=None):
    scene, nz = _column(cells, rho, axis1d, kx, ky, eps, mu, match)
    M, dt = one_step_matrix(scene)
    return M, dt, nz


def kernel_return(cells, rho, axis1d, kx, ky=0.0, eps=EPS, mu=None, match=None,
                  wavelength=1.0):
    """``({(side, 'TE'|'TM'): r}, dt)``: the kernel's steady-state return at each z face."""
    M, dt, nz = _matrix(cells, rho, axis1d, kx, ky, eps, mu, match)
    omega = 2 * math.pi / wavelength
    source = nz // 2
    f = np.zeros(M.shape[0], complex)
    f[0 * nz + source] = 1.0                           # E_x and E_y at the middle node
    f[1 * nz + source] = 1.0
    X = np.linalg.solve(cmath.exp(-1j * omega * dt) * np.eye(M.shape[0]) - M, f)
    E = X[:3 * nz].reshape(3, nz)
    Ku, Kv = grid_k(kx, DX, axis1d), grid_k(ky, DX)
    K = math.hypot(Ku, Kv)
    # TE is E transverse to k_par, TM in its plane. A stored value is the wave at its own Yee
    # point, so each component is referred back to the node first: E_x sits half a cell along
    # x, E_y half a cell along y (on an Axis1D axis there is no stagger, and nothing to undo).
    # With k_par along one axis the two never mix and the phase cancels in r.
    Ex = E[0] * (1.0 if axis1d else cmath.exp(-0.5j * kx * DX))
    Ey = E[1] * cmath.exp(-0.5j * ky * DX)
    te = (-Kv * Ex + Ku * Ey) / K if K else Ey
    tm = (Ku * Ex + Kv * Ey) / K if K else Ex
    n2 = eps * (1.0 if mu is None else mu)
    W = 2 * math.sin(omega * dt / 2) / dt
    lam = cmath.exp(2j * cmath.asin(cmath.sqrt(DX ** 2 * (n2 * W ** 2 - K ** 2)) / 2))
    out = {}
    for side in '-+':
        k0 = cells if side == '-' else nz - cells     # the layer's inner face: depth 0
        nodes = np.arange(k0, source - 1) if side == '-' else np.arange(source + 2, k0 + 1)
        basis = np.stack([lam ** (nodes - k0), lam ** (-(nodes - k0))], axis=1)
        for pol, e in (('TE', te), ('TM', tm)):
            (a, b), *_ = np.linalg.lstsq(basis, e[nodes], rcond=None)
            out[side, pol] = b / a if side == '+' else a / b    # outgoing over incoming
    return out, dt


def model_return(side, cells, rho, dt, kx, axis1d, ky=0.0, eps=EPS, mu=None, match=None,
                 wavelength=1.0):
    K = math.hypot(grid_k(kx, DX, axis1d), grid_k(ky, DX))
    eps_r = eps * (1.0 if mu is None else mu) if match is None else match
    te, tm = layer_return(cells, DX, dt, 1.0, eps_r, side, 1e-8, rho, wavelength, K, eps=eps,
                          mu=1.0 if mu is None else mu)
    return {'TE': te, 'TM': tm}


def _kx(angle, n2=EPS, wavelength=1.0):
    return math.sqrt(n2) * 2 * math.pi / wavelength * math.sin(math.radians(angle))


def _check(got, dt, cells, rho, kx, axis1d, **kwargs):
    worst = 0.0
    for side in '-+':
        want = model_return(side, cells, rho, dt, kx, axis1d, **kwargs)
        for pol in ('TE', 'TM'):
            k, m = got[side, pol], want[pol]
            assert abs(m - k) <= EXACT * max(abs(k), FLOOR), (side, pol, k, m, abs(m - k))
            worst = max(worst, abs(m - k))
    return worst


@pytest.mark.parametrize('angle', ANGLES)
@pytest.mark.parametrize('rho', [0.0, 3e-3])
@pytest.mark.parametrize('cells', [12, 24])
@pytest.mark.parametrize('axis1d', [False, True], ids=['bloch', 'axis1d'])
def test_the_model_is_the_kernels_steady_state(axis1d, cells, rho, angle):
    """Both faces, TE and TM, in ``eps = 2.25``: measured 1e-15 .. 5e-14 of the incident wave."""
    kx = _kx(angle)
    got, dt = kernel_return(cells, rho, axis1d, kx)
    _check(got, dt, cells, rho, kx, axis1d)


@pytest.mark.parametrize('case', [
    dict(wavelength=1.1), dict(wavelength=0.9, rho=1e-2),
    dict(match=1.0), dict(eps=1.5, mu=1.5), dict(ky=0.7 * math.sqrt(EPS) * 2 * math.pi * 0.4),
], ids=['off-design', 'off-design-strong-loss', 'unmatched', 'magnetic', 'kx-and-ky'])
def test_the_model_is_the_kernels_steady_state_off_the_simple_case(case):
    """The CFS pole at another wavelength than its design's, a layer matched to vacuum in
    ``eps = 2.25``, a magnetic medium (``eps = mu = 1.5``) and an order with both ``k_x`` and
    ``k_y`` (the rotated TE/TM frame) -- at 12 cells and 50 degrees."""
    case = dict(case)
    rho = case.pop('rho', 3e-3)
    wavelength = case.get('wavelength', 1.0)
    n2 = case.get('eps', EPS) * (case.get('mu') or 1.0)
    kx = _kx(50, n2, wavelength) * (0.6 if 'ky' in case else 1.0)
    matrix = {k: v for k, v in case.items() if k != 'wavelength'}
    got, dt = kernel_return(12, rho, False, kx, **matrix, wavelength=wavelength)
    _check(got, dt, 12, rho, kx, False, **case)


def _tangential_loss(pol, cells, t_E, t_H, t_EP, t_HP, j2):
    """Mutant: the loss on the tangential components as well as the normal one."""
    if pol == 'TE':
        return list(t_HP), [te - j2 / tp for te, tp in zip(t_EP, t_HP)], t_H
    return [-tp + j2 / te for tp, te in zip(t_HP, t_EP)], [-te for te in t_EP], -t_H + j2 / t_E


def _wall_a_cell_off(side, cells, alpha, beta, S_half, S_node, alpha0, beta0, lam):
    """Mutant: the PEC wall one cell into the layer (the outermost cell dropped)."""
    if side == '+':
        args = (alpha[:-1], beta[:-1], S_half[:-1], S_node[:-1])
    else:
        args = (alpha[1:], beta[1:], S_half[1:], S_node[1:])
    return _MARCH(side, cells - 1, *args, alpha0, beta0, lam)


_MARCH = model._march


@pytest.mark.parametrize('mutant', ['tangential loss', 'wall a cell off', 'no CFS pole'])
def test_mutants_of_the_model_leave_the_kernel(mutant, monkeypatch):
    """Measured at 12 cells, 60 degrees, ``rho = 3e-3``: 2.1e-3, 3.0e-3 and 8.2e-4 off, against
    a gate line of 1.5e-12."""
    kx = _kx(60)
    got, dt = kernel_return(12, 3e-3, False, kx)            # the kernel, before any patch
    if mutant == 'tangential loss':
        monkeypatch.setattr(model, '_polarisation', _tangential_loss)
    elif mutant == 'wall a cell off':
        monkeypatch.setattr(model, '_march', _wall_a_cell_off)
    else:
        monkeypatch.setattr(cpml, 'ALPHA_RELATIVE', 0.0)
    want = model_return('+', 12, 3e-3, dt, kx, False)
    off = max(abs(want[p] - got['+', p]) for p in ('TE', 'TM'))
    assert off > 1e5 * EXACT * max(abs(got['+', p]) for p in ('TE', 'TM')), (mutant, off)


# ---------------------------------------------------------------------------------------------
# (b) The doubling tables
# ---------------------------------------------------------------------------------------------

def _doubling_model(cells, angle, rho, **layout):
    """The model at the doubling probe's own carrier: its dx, dt, discrete omega and Bloch kx."""
    L = packet_layout(cells, angle, **layout)
    K = grid_k(2 * math.pi * math.sin(math.radians(angle)), L['dx'])
    return max(abs(r) for r in layer_return(cells, L['dx'], L['dt'], 1.0, 1.0, '+', 1e-8, rho,
                                            2 * math.pi / L['omega'], K))


def test_the_model_reproduces_the_doubling_tables():
    """Every entry of ``TABLE`` and ``STABILIZED``: measured 0.81 .. 1.45 of the model."""
    from tests.fdtd.test_fd_pml import RHO, STABILIZED, TABLE
    ratios = []
    for table, rho in ((TABLE, 0.0), (STABILIZED, RHO)):
        for cells, row in table.items():
            for angle, measured in row.items():
                ratio = measured / _doubling_model(cells, angle, rho)
                ratios.append(ratio)
                assert 1 / SETTLE < ratio < SETTLE, (cells, angle, rho, measured, ratio)
    assert len(ratios) == 40


def test_mutant_no_cfs_pole_leaves_the_doubling_table(monkeypatch):
    """The CFS pole dropped from the model: 7.3e-7 at 12 cells and 0 degrees, table 9.2e-5."""
    from tests.fdtd.test_fd_pml import TABLE
    monkeypatch.setattr(cpml, 'ALPHA_RELATIVE', 0.0)
    assert _doubling_model(12, 0, 0.0) < TABLE[12][0] / (10 * SETTLE)


#: The probe with a longer window: measured 0.972 / 0.954 / 0.998 / 1.017 / 1.000 of the model.
SETTLED = dict(extent=3.5, tail=6.0, delay=8.0)


@pytest.mark.parametrize('cells, angle, rho', [
    (12, 0, 0.0), (24, 30, 3e-3),
    pytest.param(12, 60, 0.0, marks=pytest.mark.slow),
    pytest.param(16, 60, 0.0, marks=pytest.mark.slow),
    pytest.param(24, 60, 3e-3, marks=pytest.mark.slow)])
def test_a_settled_probe_lands_on_the_model(cells, angle, rho):
    """The tables' gap to the model is the probe's window: a longer one closes it to 5%."""
    measured, _, layout = doubling_reflection(cells, float(angle), stabilize=rho or None,
                                              **SETTLED)
    assert layout['reference_peak'] > 100 * layout['reference_initial'], 'nothing arrived'
    want = _doubling_model(cells, angle, rho, **SETTLED)
    assert abs(measured / want - 1) < 0.06, (cells, angle, rho, measured, want)


# ---------------------------------------------------------------------------------------------
# (c) The continuum limit of the pure stretch
# ---------------------------------------------------------------------------------------------

def _thick(reflection, cos, cpw, thickness=1.2):
    """The pure stretch, ``thickness`` wavelengths deep, at ``cpw`` cells per wavelength."""
    dx = 1.0 / cpw
    K = grid_k(2 * math.pi * math.sqrt(1 - cos ** 2), dx, axis1d=True)
    return max(abs(r) for r in layer_return(round(thickness * cpw), dx, 0.5 * dx / math.sqrt(3),
                                            1.0, 1.0, '+', reflection, 0.0, 1.0, K))


CPW = (20, 40, 80, 160, 320, 640)


@pytest.mark.parametrize('reflection, cos', [(1e-8, 0.26), (1e-12, 0.26), (1e-16, 0.26),
                                             (1e-12, 0.3), (1e-8, 0.5)])
def test_the_pure_stretch_approaches_the_continuum_from_below(reflection, cos):
    """Rising, below ``R_c``, first order, and within 5% of it at 640 cells per wavelength
    (measured 0.955 .. 0.989)."""
    R_c = grazing.continuum_return(reflection, cos)
    got = [_thick(reflection, cos, cpw) for cpw in CPW]
    assert all(a < b for a, b in zip(got, got[1:])), got
    assert got[-1] < R_c and got[-1] > 0.94 * R_c, (got, R_c)
    gaps = [R_c - r for r in got]
    assert all(0.4 < b / a < 0.65 for a, b in zip(gaps[2:], gaps[3:])), gaps   # first order


def test_the_coarse_grids_are_the_runs_the_grazing_gate_measures():
    """1.2 wavelengths of 1e-12 at ``cos 0.26``: the model 6.21e-4 / 9.09e-4 / 1.110e-3 at
    20 / 40 / 80 cells per wavelength, the run (``tests/fdtd/test_fd_grazing.py``) 6.1e-4 / 9.1e-4 /
    1.11e-3."""
    for cpw, measured in ((20, 6.1e-4), (40, 9.1e-4), (80, 1.11e-3)):
        assert abs(_thick(1e-12, 0.26, cpw) / measured - 1) < 0.03, cpw


def test_mutant_no_cfs_pole_converges_to_the_wrong_limit(monkeypatch):
    """Without the pole the grading's ``g`` is 1: 8.3e-3 at 1e-8, ``cos 0.26``, not 1.23e-2."""
    monkeypatch.setattr(cpml, 'ALPHA_RELATIVE', 0.0)
    got = _thick(1e-8, 0.26, CPW[-1])
    assert got < 0.8 * grazing.continuum_return(1e-8, 0.26), got


# ---------------------------------------------------------------------------------------------
# What the model is not allowed to be: slow
# ---------------------------------------------------------------------------------------------

def test_the_model_costs_a_fraction_of_a_millisecond():
    """Measured 0.105 / 0.131 ms stabilized at 12 / 24 cells (0.073 / 0.102 pure); the line
    is 0.5 ms, about 4x the 24-cell call, which a shared machine's load does not reach on the
    best of five."""
    import time
    best = float('inf')
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(50):
            layer_return(24, 0.05, 0.0144, 1.0, 1.0, '+', 1e-8, 3e-3, 1.0, 3.0)
        best = min(best, (time.perf_counter() - t0) / 50)
    assert best < 5e-4, best


def test_a_propagating_order_is_never_amplified():
    """``|r| < 1`` everywhere on a sweep: a lossy layer returns less than it is sent."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for cells in (12, 24):
            for rho in (0.0, 3e-3, 2e-2):
                for angle in range(0, 90, 5):
                    K = grid_k(_kx(angle, 1.0), DX)
                    r = layer_return(cells, DX, 0.0144, 1.0, 1.0, '-', 1e-8, rho, 1.0, K)
                    assert max(abs(x) for x in r) < 1.0, (cells, rho, angle, r)


# ---------------------------------------------------------------------------------------------
# Many orders in one march
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('side', '-+')
@pytest.mark.parametrize('rho', [0.0, 3e-3])
def test_an_array_of_orders_is_each_order_marched_alone(side, rho):
    """``K`` as an array (what the grazing rule passes for a face's orders at one wavelength)
    gives what each order gives alone: measured 2.1e-11 relative at worst over 200 orders from
    normal incidence to grazing, 12 cells (round-off: the one-order call is Python scalar
    arithmetic, the array NumPy's). The gate line is the model's own, :data:`EXACT`."""
    K = np.linspace(0.0, 0.98 * 2 * math.pi, 200)
    te, tm = layer_return(12, DX, 0.0144, 1.0, 1.0, side, 1e-8, rho, 1.0, K)
    assert te.shape == tm.shape == K.shape
    for i, k in enumerate(K):
        one = layer_return(12, DX, 0.0144, 1.0, 1.0, side, 1e-8, rho, 1.0, float(k))
        for got, want in zip((te[i], tm[i]), one):
            assert abs(got - want) <= EXACT * abs(want), (k, got, want)


def test_an_array_of_orders_costs_about_one_order():
    """Measured (best of 5 x 20, Windows): 0.25 / 0.31 / 0.76 ms for 1 / 100 / 1000 orders
    in one call, against 0.09 ms for one order alone. The line is 3 ms for 1000 orders --
    a thousand one-order calls would take 90."""
    import time
    K = np.linspace(0.0, 5.0, 1000)
    best = float('inf')
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(20):
            layer_return(12, 0.05, 0.0144, 1.0, 1.0, '+', 1e-8, 3e-3, 1.0, K)
        best = min(best, (time.perf_counter() - t0) / 20)
    assert best < 3e-3, best


def test_an_order_the_grid_makes_evanescent_is_refused():
    """Only a propagating order is modelled (the rule never asks for another --
    ``tests/fdtd/test_fd_grazing.py::test_a_grid_evanescent_order_is_never_modelled``). An array
    with one such order in it is refused whole."""
    dt = 0.0144
    W = 2 * math.sin(math.pi * dt) / dt
    with pytest.raises(ValueError, match='evanescent on this grid'):
        layer_return(12, DX, dt, 1.0, 1.0, '+', 1e-8, 3e-3, 1.0, 1.001 * W)
    with pytest.raises(ValueError, match='evanescent on this grid'):
        layer_return(12, DX, dt, 1.0, 1.0, '+', 1e-8, 3e-3, 1.0, np.array([0.5 * W, 1.001 * W]))
    layer_return(12, DX, dt, 1.0, 1.0, '+', 1e-8, 3e-3, 1.0, 0.999 * W)


def test_an_order_past_the_grids_nyquist_along_the_axis_is_refused():
    """``4 sin^2(kappa/2) = dx^2 (n^2 W^2 - K^2) > 4``: the wave is shorter along the face's
    axis than two cells, and the grid carries it as a decaying sign-alternating field. The
    review's case -- ``dx = 1/13``, wavelength 0.235 in vacuum, ``K`` 1.51 and 5.44 -- was a bare
    'math domain error' on one order and a silent NaN on an array; both paths now refuse it by
    name (the plan refuses such a Diffraction wavelength before the rule reads it:
    ``tests/fdtd/test_fd_grazing.py::test_a_diffraction_wavelength_the_grid_cannot_carry_is_refused``).
    """
    dx = 1.0 / 13
    dt = 0.5 * dx / math.sqrt(3)
    for K in (1.51, np.array([1.51, 5.44])):
        with pytest.raises(ValueError, match="past the grid's Nyquist limit"):
            layer_return(12, dx, dt, 1.0, 1.0, '-', 1e-8, 3e-3, 0.235, K)
        with pytest.raises(ValueError, match="past the grid's Nyquist limit"):
            layer_return(12, dx, dt, 1.0, 1.0, '-', 1e-8, 0.0, 0.235, K)
    # At 8 cells per wavelength, the floor the plan holds every Diffraction wavelength to, the
    # normal order is far inside: dx^2 n^2 W^2 <= (2 pi / 8)^2 = 0.62.
    r = layer_return(12, dx, dt, 1.0, 1.0, '-', 1e-8, 3e-3, 8 * dx, np.array([0.0, 1.0]))
    assert np.all(np.isfinite(r[0])) and np.all(np.isfinite(r[1]))
