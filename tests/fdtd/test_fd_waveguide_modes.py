"""``fdtd.solve_modes``: modes of the discrete Yee system against independent answers.

The criteria are the ones registered in ``docs/validation/fdtd/fd-waveguide-modes.md`` before
the solver was first run (M1-M6); every limit here is quoted from there. The oracles are
closed forms (the Yee dispersion of a uniform medium, the hollow metal guide on the grid), the
exact TE/TM dispersion relations of a symmetric slab, and the published dielectric-corner
benchmark of G. R. Hadley, J. Lightwave Technol. 20, 1210 (2002).
"""
import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

from wavebit import Material, Units, fdtd
from wavebit.errors import WavebitError
from wavebit.materials import ConductiveMaterial


def _quiet():
    warnings.simplefilter('ignore')


def _problem(domain, center, boundaries, structures=(), background=1.0):
    return fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=domain, center=center),
                         background=Material(epsilon=background), boundaries=boundaries,
                         structures=tuple(structures)))


def _numerics(dx, wavelength=1.0):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=dx), duration=fdtd.Duration(steps=1),
                         absorber=fdtd.PML(wavelength=wavelength))


def _omega_d(problem, numerics, wavelength):
    """``Omega = (2/dt) sin(omega dt/2)`` from the time step ``fdtd.check`` reports."""
    from wavebit.units import C0
    seconds = fdtd.check(problem, numerics=numerics).resolved['time_step_seconds']
    dt = seconds * C0 / problem.units.length_unit_meters
    return 2.0 / dt * math.sin(math.pi / wavelength * dt)


# ---------------------------------------------------------------------------------------------
# M1: homogeneous control
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('invariant', [False, True])
def test_m1_homogeneous_medium_gives_its_index_and_the_yee_dispersion(invariant):
    _quiet()
    dx, lam = 0.05, 1.0
    y = fdtd.InvariantAxis() if invariant else fdtd.Periodic()
    problem = _problem((0.6, dx if invariant else 0.4, 4 * dx), (0., 0., 0.),
                       fdtd.Boundaries(x=fdtd.Periodic(), y=y, z=fdtd.Periodic()), background=2.25)
    modes = fdtd.solve_modes(problem, numerics=_numerics(dx), plane=fdtd.Plane(z=0.),
                             wavelength=lam, count=1)
    Omega = _omega_d(problem, _numerics(dx), lam)
    q = modes[0].metadata['q']
    assert abs(q / Omega - 1.5) <= 1e-12 * 1.5
    k0 = 2 * math.pi / lam
    closed = (2.0 / (dx * k0)) * math.asin(1.5 * Omega * dx / 2.0)
    assert abs(modes[0].effective_index - closed) <= 1e-12 * closed


# ---------------------------------------------------------------------------------------------
# M2: hollow metal guide
# ---------------------------------------------------------------------------------------------

def _hollow(dx):
    problem = _problem((2., 1., 4 * dx), (1., .5, 0.),
                       fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.Periodic()),
                       background=2.25)
    modes = fdtd.solve_modes(problem, numerics=_numerics(dx), plane=fdtd.Plane(z=0.),
                             wavelength=1.0, count=5)
    return modes, _omega_d(problem, _numerics(dx), 1.0)


#: (m, n) of TE10, TE20, TE01, TE11 and TM11 in a 2 x 1 guide, sorted by decreasing index.
_HOLLOW = ((1, 0), (2, 0), (0, 1), (1, 1), (1, 1))


def _hollow_q2(dx, Omega, a=2.0, b=1.0):
    return sorted((Omega ** 2 * 2.25 - (2 / dx) ** 2 * (math.sin(m * math.pi * dx / (2 * a)) ** 2
                                                        + math.sin(n * math.pi * dx / (2 * b)) ** 2)
                   for m, n in _HOLLOW), reverse=True)


def _hollow_continuum(lam=1.0, a=2.0, b=1.0):
    return sorted((math.sqrt(2.25 - (lam / 2) ** 2 * ((m / a) ** 2 + (n / b) ** 2))
                   for m, n in _HOLLOW), reverse=True)


def test_m2_hollow_metal_guide_matches_the_discrete_closed_form_and_converges():
    _quiet()
    errors = []
    for dx in (0.05, 0.025, 0.0125):
        modes, Omega = _hollow(dx)
        q2 = [m.metadata['q'] ** 2 for m in modes]
        exact = _hollow_q2(dx, Omega)
        for got, want in zip(q2, exact):
            assert abs(got - want) <= 1e-10 * abs(want)
        errors.append([abs(m.effective_index - c)
                       for m, c in zip(modes, _hollow_continuum())])
    errors = np.array(errors)
    order = np.log2(errors[:-1] / errors[1:])
    assert np.all(order >= 1.8), order


# ---------------------------------------------------------------------------------------------
# M3: symmetric slab against the exact dispersion relation
# ---------------------------------------------------------------------------------------------

N1, N2, THICKNESS, LAM = 2.0, 1.444, 1.5, 1.55


def _slab_exact(order, polarization):
    """The exact effective index of mode ``order`` of the symmetric slab (TE or TM)."""
    k0, a = 2 * math.pi / LAM, THICKNESS / 2
    r = 1.0 if polarization == 'TE' else (N1 / N2) ** 2

    def f(n):
        kappa = k0 * math.sqrt(N1 ** 2 - n ** 2)
        gamma = k0 * math.sqrt(n ** 2 - N2 ** 2)
        return 2 * kappa * a - order * math.pi - 2 * math.atan(r * gamma / kappa)
    return brentq(f, N2 + 1e-12, N1 - 1e-12, xtol=1e-15, rtol=1e-15)


def _slab_problem(dx, width=9.):
    return _problem((width, dx, 4 * dx), (0., 0., 0.),
                       fdtd.Boundaries(x=fdtd.Open(), y=fdtd.InvariantAxis(), z=fdtd.Periodic()),
                       structures=(fdtd.Structure('core', shape=fdtd.SnappedSlab(x=(-.75, .75)),
                                                  material=Material(epsilon=N1 ** 2)),),
                       background=N2 ** 2)


def _slab_modes(dx):
    return fdtd.solve_modes(_slab_problem(dx), numerics=_numerics(dx, LAM), plane=fdtd.Plane(z=0.),
                            wavelength=LAM, count=6)


def _split(modes):
    """TE (E along the invariant y) and TM modes, each by decreasing index."""
    te = [m for m in modes if m.polarization['y'] > 0.5]
    tm = [m for m in modes if m.polarization['y'] <= 0.5]
    return te, tm


def test_m3_slab_converges_to_the_exact_te_and_tm_dispersion():
    _quiet()
    exact = [_slab_exact(m, 'TE') for m in range(3)] + [_slab_exact(m, 'TM') for m in range(3)]
    errors, signed = [], []
    for dx in (0.05, 0.025, 0.0125):
        modes = _slab_modes(dx)
        te, tm = _split(modes)
        assert len(te) == 3 and len(tm) == 3
        got = [m.effective_index for m in te + tm]
        signed.append([g - e for g, e in zip(got, exact)])
        errors.append([abs(g - e) for g, e in zip(got, exact)])
        # Orthogonality and normalization at every spacing.
        _check_orthonormal(modes, dx)
    errors, signed = np.array(errors), np.array(signed)
    order = np.log2(errors[:-1] / errors[1:])
    assert np.all(order >= 1.8), order
    assert np.all(errors[-1] <= 1e-3), errors[-1]
    richardson = (4 * (signed[-1] + np.array(exact)) - (signed[-2] + np.array(exact))) / 3
    assert np.all(np.abs(richardson - np.array(exact)) <= 1e-4), richardson - np.array(exact)


def _check_orthonormal(modes, dx):
    """Cross product of distinct modes <= 1e-10 of their norms; flux power 1 to 1e-12.

    The plane is across z, so u = x and v = y; y is invariant, so the power is per metre of y
    and each sample stands for dx of x. H is in A/m and E in V/m.
    """
    L = 1e-6
    for i, a in enumerate(modes):
        Ex, Ey = np.asarray(a.E['Ex']), np.asarray(a.E['Ey'])
        Hx, Hy = np.asarray(a.H['Hx']), np.asarray(a.H['Hy'])
        cos = math.cos(a.beta_per_meter * L * dx / 2)
        power = 0.5 * np.sum(np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))) * dx * L * cos
        assert abs(power - 1.0) <= 1e-12
        for b in modes[i + 1:]:
            Hbx, Hby = np.asarray(b.H['Hx']), np.asarray(b.H['Hy'])
            cross = np.sum(Ex * Hby - Ey * Hbx)
            na = math.sqrt(np.sum(np.abs(Ex) ** 2 + np.abs(Ey) ** 2))
            nb = math.sqrt(np.sum(np.abs(Hbx) ** 2 + np.abs(Hby) ** 2))
            assert abs(cross) <= 1e-10 * na * nb


# ---------------------------------------------------------------------------------------------
# M4: Hadley's dielectric corners
# ---------------------------------------------------------------------------------------------

#: (core eps, cladding eps, PEC sides, published n_eff): Hadley 2002, stated error < 1e-8.
_CORNERS = ((2.25, 1.0, ('left', 'right'), 1.27627404),
            (8.0, 1.0, ('left', 'right'), 2.65679692),
            (1.0, 2.25, ('left', 'top'), 1.387926425),
            (1.0, 8.0, ('left', 'top'), 2.761465320))


def _corner_modes(core, clad, pec, N):
    """The unit square with its walls on the square's edges, at the spacing that puts them there.

    A PEC wall sits on its face; a PMC wall half a cell outside a low face and half a cell
    inside a high one (``src/wavebit/fdtd/_kernel/faces.py``). With ``s = 1/2`` for a PMC side and 0 for PEC, the
    low wall is at ``lo - s_low dx`` and the high one at ``lo + n dx - s_high dx``; putting them
    at 0 and 1 gives ``lo = s_low dx`` and ``dx = 1 / (n + s_low - s_high)``. ``N`` cells along x
    fix ``dx``; y then needs a whole number of cells, which both benchmark layouts give.
    """
    edges = {'left': 'x-', 'right': 'x+', 'bottom': 'y-', 'top': 'y+'}
    pmc = {face: side not in pec for side, face in edges.items()}
    s = {face: (0.5 if is_pmc else 0.0) for face, is_pmc in pmc.items()}
    dx = 1.0 / (N + s['x-'] - s['x+'])
    ny = 1.0 / dx - s['y-'] + s['y+']
    assert abs(ny - round(ny)) < 1e-9
    ny = int(round(ny))
    low = (s['x-'] * dx, s['y-'] * dx)
    size = (N * dx, ny * dx)
    center = (low[0] + size[0] / 2, low[1] + size[1] / 2, 0.)
    face = lambda f: fdtd.PMC() if pmc[f] else fdtd.PEC()       # noqa: E731
    problem = _problem((size[0], size[1], 4 * dx), center,
                       fdtd.Boundaries(x=(face('x-'), face('x+')), y=(face('y-'), face('y+')),
                                       z=fdtd.Periodic()),
                       structures=(fdtd.Structure('core', shape=fdtd.Box(size=(1.5, 1.5, 1.),
                                                                          center=(-.25, -.25, 0.)),
                                                  material=Material(epsilon=core)),),
                       background=clad)
    modes = fdtd.solve_modes(problem, numerics=_numerics(dx, 1.5), plane=fdtd.Plane(z=0.),
                             wavelength=1.5, count=1)
    return modes, dx


@pytest.mark.parametrize('case', _CORNERS, ids=['2.25-1-LR', '8-1-LR', '1-2.25-LT', '1-8-LT'])
def test_m4_dielectric_corners_reach_hadleys_published_index(case):
    _quiet()
    core, clad, pec, reference = case
    errors = []
    for N in (40, 80, 160):
        modes, dx = _corner_modes(core, clad, pec, N)
        errors.append(abs(modes[0].effective_index - reference))
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[2] <= 1e-3, errors


# ---------------------------------------------------------------------------------------------
# M5: refusals by name
# ---------------------------------------------------------------------------------------------

def _guide(dx=0.05, *, core=None, boundaries=None, background=1.0, structures=None):
    """A 1 x 0.5 core of epsilon 4 in a 3 x 3 PEC box, propagating along z."""
    core = core if core is not None else Material(epsilon=4.0)
    if structures is None:
        structures = (fdtd.Structure('core', shape=fdtd.SnappedBox(size=(1., .5, 10.)),
                                     material=core),)
    return _problem((3., 3., 0.4), (0., 0., 0.),
                    boundaries or fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.Periodic()),
                    structures=structures, background=background)


def _solve(problem, dx=0.05, **kw):
    kw.setdefault('plane', fdtd.Plane(z=0.))
    kw.setdefault('wavelength', 1.0)
    return fdtd.solve_modes(problem, numerics=_numerics(dx), **kw)


def _refused(code, problem, **kw):
    _quiet()
    with pytest.raises(WavebitError) as info:
        _solve(problem, **kw)
    assert info.value.code == code, (info.value.code, str(info.value))
    return info.value


def test_m5_the_guide_itself_is_admitted():
    _quiet()
    modes = _solve(_guide(), count=2)
    assert len(modes) == 2 and modes[0].effective_index > modes[1].effective_index > 1.0


def test_m5_conductive_or_dispersive_medium_in_the_window_is_refused():
    from wavebit.materials import DispersiveMaterial, Lorentz
    _refused('mode.material', _guide(core=ConductiveMaterial(epsilon=4.0, conductivity=1e3)))
    lorentz = DispersiveMaterial(epsilon_inf=2.0, poles=(Lorentz(strength=1.0, omega_0=3e15, gamma=1e13),))
    _refused('mode.material', _guide(core=lorentz))


def test_m5_a_medium_that_changes_along_the_axis_is_refused():
    ends = (fdtd.Structure('core', shape=fdtd.SnappedBox(size=(1., .5, .2), center=(0., 0., .1)),
                           material=Material(epsilon=4.0)),)
    error = _refused('mode.nonuniform', _guide(structures=ends))
    assert 'uniform' in str(error)


@pytest.mark.parametrize('face', [fdtd.Bloch(phase=0.3), fdtd.MirrorSymmetry(parity=1)],
                         ids=['Bloch', 'MirrorSymmetry'])
def test_m5_bloch_and_mirror_transverse_faces_are_refused(face):
    boundaries = fdtd.Boundaries(x=face if isinstance(face, fdtd.Bloch) else (face, fdtd.PEC()),
                                 y=fdtd.PEC(), z=fdtd.Periodic())
    _refused('mode.boundary', _guide(boundaries=boundaries))


def test_m5_invariant_axis_with_a_wavevector_is_refused():
    boundaries = fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.InvariantAxis(k=0.5), z=fdtd.Periodic())
    slab = (fdtd.Structure('core', shape=fdtd.SnappedSlab(x=(-.5, .5)), material=Material(epsilon=4.)),)
    problem = _problem((3., .05, .4), (0., 0., 0.), boundaries, structures=slab)
    _refused('mode.boundary', problem)


def test_m5_coupled_smoothed_material_is_refused():
    rod = (fdtd.Structure('core', shape=fdtd.Cylinder(radius=.4, height=10.), material=Material(epsilon=4.)),)
    _refused('mode.material', _guide(structures=rod))


def test_m5_traced_inputs_are_refused_by_name():
    import jax
    import torch
    jax.config.update('jax_enable_x64', True)

    def through_jax(eps):
        return _solve(_guide(core=Material(epsilon=eps)))[0].effective_index
    with pytest.raises(WavebitError) as info:
        jax.grad(through_jax)(4.0)
    assert info.value.code == 'mode.traced'
    eps = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    _refused('mode.traced', _guide(core=Material(epsilon=eps)))
    _refused('mode.traced', _guide(), wavelength=torch.tensor(1.0, dtype=torch.float64, requires_grad=True))


def test_m5_plane_outside_or_across_an_invariant_axis_is_refused():
    _refused('mode.outside_domain', _guide(), plane=fdtd.Plane(z=0.5))
    boundaries = fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.InvariantAxis())
    problem = _problem((3., 3., .05), (0., 0., 0.), boundaries)
    _refused('mode.plane', problem)


def test_m5_more_confined_modes_than_exist_are_refused():
    # An Open box: only the guided modes are confined; the slab of M3 guides three of each.
    error = _refused('mode.count', _slab_problem(0.05), plane=fdtd.Plane(z=0.), wavelength=LAM,
                     count=7)
    assert 'found 6 confined mode(s)' in str(error)


def test_m5_a_mode_reaching_a_truncating_edge_is_not_admitted():
    # The same slab in a window too narrow for its third TM mode's tail.
    problem = _slab_problem(0.05, width=3.0)
    error = _refused('mode.count', problem, plane=fdtd.Plane(z=0.), wavelength=LAM, count=6)
    assert 'outermost cells' in str(error)


# ---------------------------------------------------------------------------------------------
# M6: backends
# ---------------------------------------------------------------------------------------------

def test_m6_backends_return_the_same_bits():
    import jax
    jax.config.update('jax_enable_x64', True)
    _quiet()
    results = {b: _solve(_guide(), count=2, backend=b) for b in ('numpy', 'jax', 'torch')}
    for b in ('jax', 'torch'):
        assert type(results[b][0].E['Ex']).__module__.split('.')[0] in ('jax', 'jaxlib', 'torch')
        for ref, got in zip(results['numpy'], results[b]):
            assert got.effective_index == ref.effective_index
            for name in ref.E:
                np.testing.assert_array_equal(np.asarray(got.E[name]), ref.E[name])
                np.testing.assert_array_equal(np.asarray(got.H['H' + name[1]]), ref.H['H' + name[1]])


# ---------------------------------------------------------------------------------------------
# M7: the mode is a mode of the run
# ---------------------------------------------------------------------------------------------

def _bloch_guide(kind, dx, k):
    """The M7 guides with a Bloch propagation axis of eight cells and wavevector ``k``."""
    z = fdtd.Bloch(k=k) if k else fdtd.Periodic()
    if kind in ('box', 'smoothed'):
        shape = (fdtd.SnappedBox(size=(1., .5, 10.)) if kind == 'box'
                 else fdtd.Box(size=(1.03, .51, 10.)))
        return _problem((3., 3., 8 * dx), (0., 0., 0.),
                        fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=z),
                        structures=(fdtd.Structure('core', shape=shape,
                                                   material=Material(epsilon=4.0)),))
    return _problem((9., dx, 8 * dx), (0., 0., 0.),
                    fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.InvariantAxis(), z=z),
                    structures=(fdtd.Structure('core', shape=fdtd.SnappedSlab(x=(-.75, .75)),
                                               material=Material(epsilon=N1 ** 2)),),
                    background=N2 ** 2)


@pytest.mark.parametrize('kind,lam', [('box', 1.0), ('slab', LAM), ('smoothed', 1.0)])
@pytest.mark.parametrize('index', [0, 1])
def test_m7_a_solved_mode_is_an_eigenmode_of_the_time_stepper(kind, lam, index):
    _quiet()
    from wavebit.units import C0
    dx, steps = 0.05, 100
    modes = fdtd.solve_modes(_bloch_guide(kind, dx, 0.0), numerics=_numerics(dx, lam),
                             plane=fdtd.Plane(z=0.), wavelength=lam, count=2)
    mode = modes[index]
    beta = mode.beta_per_meter * 1e-6                       # rad per um
    problem = _bloch_guide(kind, dx, beta)
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=dx), duration=fdtd.Duration(steps=steps),
                             absorber=fdtd.PML(wavelength=lam))
    report = fdtd.check(problem, numerics=numerics)
    dt = report.resolved['time_step_seconds'] * C0 / 1e-6   # in um
    omega = 2 * math.pi / lam
    from wavebit.fdtd._kernel.grid import OFFSETS
    nz = 8
    zlow = -4 * dx
    grid_shape = (3, nz) + np.shape(mode.E['Ex'])

    def volume(field, values):
        out = np.zeros(grid_shape, dtype=complex)
        for c, axis in enumerate('xyz'):
            z = zlow + (np.arange(nz) + 0.5 * OFFSETS[field][c][2]) * dx
            out[c] = np.asarray(values[field + axis])[None] * np.exp(1j * beta * z)[:, None, None]
        return out
    E0, H0 = volume('E', mode.E), volume('H', mode.H)
    initial = fdtd.InitialField(E=E0, H=H0 * np.exp(1j * omega * dt / 2))
    run = fdtd.run(problem.replace(initial=initial, outputs=fdtd.Outputs(final_fields=True)),
                   numerics=numerics)
    final = run.final_fields.values
    expected_E = E0 * np.exp(-1j * omega * steps * dt)
    expected_H = H0 * np.exp(-1j * omega * (steps - 0.5) * dt)
    for got, want in ((np.asarray(final['E']), expected_E), (np.asarray(final['H']), expected_H)):
        scale = np.max(np.abs(want))
        assert np.max(np.abs(got - want)) <= 1e-10 * scale, np.max(np.abs(got - want)) / scale
