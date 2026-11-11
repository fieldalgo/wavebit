"""The in-cell field with a **patterned permeability**, against RETICOLO, under both rules.

``reconstruction='lalanne'`` is RETICOLO's ``champ = 1`` on ``H`` as well as ``E``: across a wall
``mu H`` is continuous and ``H`` normal to it jumps, the dual of ``eps E``, and RETICOLO builds
``fhx``/``fhy`` from ``mu`` as it builds ``fex``/``fey`` from ``eps``. The Fourier sum smooths that
jump just as it does ``E``'s, so ``champ = 0`` is the sum on the four transverse components.

The golden reaches each path the rule has an ``H`` on (``_reticolo_magnetic_field_cases.py``).
Nothing is fitted: on 2D ``res3``'s cell is this project's public one with ``x`` and ``y`` swapped,
as ``test_reticolo_2d_field_golden.py`` explains, and on the 1D paths, where RETICOLO's origin is
the cell's corner, the one stated phase is the incident wave's across half a period. A flipped sign
on ``B_x`` is O(1) here.
"""
import numpy as np
import pytest

from tests.references._reticolo_magnetic_field_cases import (
    ANGLE_1D, ANGLE_2D, ANGLE_CONICAL, EINC, MATERIALS, NAMES, NPTS, N_BOT, N_TOP, ORDER_1D, ORDER_2D,
    PERIOD_1D, PERIOD_2D, STACK_1D, STACK_2D, WL, fields_1d, grid_fields, key)
from tools.repository_paths import repository_path
from wavebit import Material, Units
from wavebit import rcwa as rcwa_api
from wavebit.rcwa._factory import rcwa
from wavebit.units import Z0

GOLDEN = np.load(repository_path('tests/references/golden_reticolo_magnetic_field.npz'))

# Measured over every comparison: the worst gap is 3.0e-11. Round-off room above that, not slack.
TOL = 1e-10
SLOTS_1D = {'te': (('Ey', 0), ('Hx', 1), ('Hz', 2)), 'tm': (('Hy', 0), ('Ex', 1), ('Ez', 2))}
X, Y, X_1D = GOLDEN['x_2d'], GOLDEN['y_2d'], GOLDEN['x_1d']
RULES = {'lalanne': 1, 'plain': 0}
_LAYERS = len(STACK_2D)


def _heights(path):
    """The layers' rows of ``res3``'s cell, each layer's top row moved just inside it."""
    rows = np.arange(NPTS, NPTS * (_LAYERS + 1))
    z = GOLDEN[f'z_{path}'][rows].copy()
    z[np.arange(NPTS - 1, NPTS * _LAYERS, NPTS)] -= 1e-9
    return rows, z


def _reticolo(path, pol, champ):
    rows, _ = _heights(path)
    return GOLDEN[key(path, pol, champ)][rows]


def _gap(ours, theirs):
    return np.abs(ours - theirs).max() / np.abs(theirs).max()


def _material(index):
    """An index outside :data:`MATERIALS` names no ``mu``, so its layer has no magnetic line at all."""
    if index not in MATERIALS:
        return Material(index ** 2)
    eps, mu = MATERIALS[index]
    return Material(eps, mu=mu)


def _shape_structure():
    return [dict(background=_material(layer['background']),
                 shapes=[dict(type='rectangle', cx=cx, cy=cy, lx=dx, ly=dy, material=_material(n))
                         for cx, cy, dx, dy, n in layer['blocks']])
            for layer in STACK_2D[::-1]]


def _solved_2d(pol, structure, formulation='auto', **kw):
    a_p, a_s = EINC[pol]
    s = rcwa(backend='numpy', top=N_TOP ** 2, bot=N_BOT ** 2, period=list(PERIOD_2D),
             truncation_order=[ORDER_2D] * 2, wavelength=WL,
             thickness=[layer['thickness'] for layer in STACK_2D[::-1]], structure=structure,
             theta=np.deg2rad(ANGLE_2D[0]), phi=np.deg2rad(ANGLE_2D[1]), polarization=(a_s, a_p),
             factorization='li', answers='field', **kw)
    s.formulation = formulation
    s.solve()
    return s


def _solved_raster(pol, formulation='auto'):
    eps, mu = grid_fields()
    return _solved_2d(pol, eps, formulation=formulation, mu=mu, fourier_coefficients='integrated')


def _solved_1d(path, pol):
    eps, mu = fields_1d()
    conical = path == 'conical'
    kw = dict(theta=np.deg2rad(ANGLE_CONICAL[0]), phi=np.deg2rad(ANGLE_CONICAL[1]),
              psi=np.pi / 2 if pol == 's' else 0.0) if conical else dict(
        theta=np.deg2rad(ANGLE_1D), phi=None, tetm=path)
    s = rcwa(backend='numpy', top=N_TOP ** 2, bot=N_BOT ** 2, period=[PERIOD_1D], truncation_order=[ORDER_1D],
             wavelength=WL, thickness=[layer['thickness'] for layer in STACK_1D[::-1]], structure=eps, mu=mu,
             factorization='li', fourier_coefficients='integrated', answers='field', **kw)
    s.solve()
    return s


def _corner_phase(path):
    """RETICOLO's 1D wave is unit at the cell's corner, this project's at its centre."""
    theta = ANGLE_CONICAL[0] if path == 'conical' else ANGLE_1D
    kx = N_TOP * np.sin(np.deg2rad(theta))
    if path == 'conical':
        kx = kx * np.cos(np.deg2rad(ANGLE_CONICAL[1]))
    return np.exp(-2j * np.pi / WL * kx * PERIOD_1D / 2)


@pytest.mark.parametrize('rule,champ', RULES.items())
@pytest.mark.parametrize('pol', list(EINC))
@pytest.mark.parametrize('path', ['shapes', 'raster'])
def test_two_dimensions(path, pol, rule, champ):
    s = _solved_2d(pol, _shape_structure()) if path == 'shapes' else _solved_raster(pol)
    ours = np.asarray(s.reconstruct_field(X, Y, _heights(path)[1], reconstruction=rule))[0]
    theirs = np.transpose(_reticolo(path, pol, champ), (0, 2, 1, 3))
    # RETICOLO leaves E_z and H_z unset inside a layer under champ = 0.
    gaps = {NAMES[k]: _gap(ours[..., k], theirs[..., k]) for k in (range(6) if champ else (0, 1, 3, 4))}
    assert all(gap < TOL for gap in gaps.values()), gaps


@pytest.mark.parametrize('rule,champ', RULES.items())
@pytest.mark.parametrize('pol', list(EINC))
@pytest.mark.parametrize('path', ['shapes', 'raster'])
def test_the_first_order_path_has_no_rule_of_its_own(path, pol, rule, champ):
    """RETICOLO has no formulation to ask, so this is the same golden read from the other path."""
    s = (_solved_2d(pol, _shape_structure(), formulation='first_order') if path == 'shapes'
         else _solved_raster(pol, formulation='first_order'))
    ours = np.asarray(s.reconstruct_field(X, Y, _heights(path)[1], reconstruction=rule))[0]
    theirs = np.transpose(_reticolo(path, pol, champ), (0, 2, 1, 3))
    gaps = {NAMES[k]: _gap(ours[..., k], theirs[..., k]) for k in (range(6) if champ else (0, 1, 3, 4))}
    assert all(gap < TOL for gap in gaps.values()), gaps


@pytest.mark.parametrize('path,pol', [('te', 's'), ('tm', 'p')])
def test_the_pure_1d_paths(path, pol):
    """``res3``'s pure 1D branch rebuilds the wall's normal component under either ``champ``.

    So ``champ = 0`` cannot speak for the sum there except on the component along every wall, and
    the sum's own difference from the rule is what shows the reference can tell them apart.
    """
    s = _solved_1d(path, pol)
    x, z = X_1D - PERIOD_1D / 2, _heights(path)[1]
    rule = np.asarray(s.reconstruct_field(x, 1, z, reconstruction='lalanne'))[:, 0]
    plain = np.asarray(s.reconstruct_field(x, 1, z))[:, 0]
    one, zero = (_reticolo(path, pol, champ) * _corner_phase(path) for champ in (1, 0))
    gaps = {name: _gap(rule[..., k], one[..., k]) for name, k in SLOTS_1D[path]}
    assert all(gap < TOL for gap in gaps.values()), gaps
    assert _gap(plain[..., 0], zero[..., 0]) < TOL
    np.testing.assert_array_equal(zero[..., 1], one[..., 1])
    assert _gap(plain[..., 1], one[..., 1]) > 0.1


@pytest.mark.parametrize('rule,champ', RULES.items())
@pytest.mark.parametrize('pol', list(EINC))
def test_the_conical_path(pol, rule, champ):
    s = _solved_1d('conical', pol)
    ours = np.asarray(s.reconstruct_field(X_1D - PERIOD_1D / 2, np.array([0.0]), _heights('conical')[1],
                                          reconstruction=rule))[0][:, 0]
    theirs = _reticolo('conical', pol, champ)[:, :, 0] * _corner_phase('conical')
    gaps = {NAMES[k]: _gap(ours[..., k], theirs[..., k]) for k in (range(6) if champ else (0, 1, 3, 4))}
    assert all(gap < TOL for gap in gaps.values()), gaps


@pytest.mark.parametrize('path', ['shapes', 'raster', 'conical'])
@pytest.mark.parametrize('pol', list(EINC))
def test_each_reference_tells_the_two_rules_apart_on_h(path, pol):
    one, zero = _reticolo(path, pol, 1)[..., 3], _reticolo(path, pol, 0)[..., 3]
    assert _gap(zero, one) > 0.1


def _regions(layer):
    return tuple(rcwa_api.Region(geometry=rcwa_api.Rectangle(size=(dx, dy), center=(cx, cy)), material=_material(n))
                 for cx, cy, dx, dy, n in layer['blocks'])


@pytest.mark.parametrize('pol', list(EINC))
def test_a_result_of_the_current_api_is_champ_1_on_shapes(pol):
    layers = tuple(rcwa_api.Layer(thickness=layer['thickness'], background=_material(layer['background']),
                                  regions=_regions(layer)) for layer in STACK_2D[::-1])
    problem = rcwa_api.Problem(
        units=Units(length='nm'),
        stack=rcwa_api.Stack(period=tuple(PERIOD_2D), top=Material(N_TOP ** 2), bottom=Material(N_BOT ** 2), layers=layers),
        excitation=rcwa_api.Excitation(wavelength=WL, sources=(rcwa_api.PlaneWave(
            theta=np.deg2rad(ANGLE_2D[0]), phi=np.deg2rad(ANGLE_2D[1]), polarization=pol),)),
        outputs=rcwa_api.Outputs(fields=True))
    result = rcwa_api.run(problem, numerics=rcwa_api.Numerics(orders=(ORDER_2D, ORDER_2D), factorization='li'))
    ours = np.array(result.field(x=X, y=Y, z=_heights('shapes')[1], reconstruction='lalanne').values[0])
    ours[..., 3:] *= Z0
    theirs = np.transpose(_reticolo('shapes', pol, 1), (0, 2, 1, 3))
    gaps = {NAMES[k]: _gap(ours[..., k], theirs[..., k]) for k in range(6)}
    assert all(gap < TOL for gap in gaps.values()), gaps


@pytest.mark.parametrize('path,pol', [('raster', 'p'), ('te', 's'), ('conical', 's'), ('conical', 'p')])
def test_the_field_state_answers_what_the_solver_answers(path, pol):
    """The rule's second body, on each path: what a ``Result`` reads, against the solver's door."""
    s = _solved_raster(pol) if path == 'raster' else _solved_1d(path, pol)
    x, y = (X, Y) if path == 'raster' else (X_1D - PERIOD_1D / 2, np.array([0.0]))
    z = _heights(path)[1]
    door = np.asarray(s.reconstruct_field(x, y, z, reconstruction='lalanne'))
    ours = np.asarray(s.solve()._field_state.reconstruct(x, y, z, 'incident', 'lalanne'))
    if path == 'te':
        expected, ours = door[..., 1], ours[..., 3]
    else:
        expected, ours = door[0, ..., 3:5], ours[..., 3:5]
    assert np.abs(ours - expected).max() < 1e-11 * np.abs(expected).max()
