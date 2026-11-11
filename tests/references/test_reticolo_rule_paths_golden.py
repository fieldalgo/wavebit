"""The in-cell field off the Lalanne rule's first paths, against RETICOLO, under both rules.

``reconstruction='lalanne'`` is ``champ = 1`` and ``'plain'`` is ``champ = 0``
(``test_reticolo_2d_field_golden.py``), here on what the scalar goldens do not reach: a diagonal
tensor in a patterned layer, a uniform tensor beside one on either formulation, layers merged into
a half-space, one retained order along an axis, incidence from below, and a dipole. A diagonal
tensor is ``eps_xx`` and ``mu_xx`` on the rows and ``eps_yy`` and ``mu_yy`` on the columns, as RETICOLO's
``fex``, ``fhx``, ``fey`` and ``fhy`` are, so crossing them is O(1). Nothing is fitted but a dipole's overall
scale: RETICOLO's source carries a factor of its own, read off the sum and held for the rule.
"""
import numpy as np
import pytest

from tests.references._reticolo_rule_paths_cases import (
    ANGLE_1D, ANGLE_2D, ANGLE_CONICAL, EINC, MERGED_1D, MERGED_2D, MERGE_NUDGE, MERGE_TOLERANCE, N_BOT, N_TOP, NAMES,
    NPTS, ONE_ORDER_2D, ORDER_1D, ORDER_2D, PERIOD_1D, PERIOD_2D, SOURCE_HOST, SOURCE_HOST_INDEX, SOURCE_MOMENTS,
    SOURCE_ORDER, SOURCE_PATTERNED, SOURCE_POSITION, SOURCE_ROW, TENSOR_1D, TENSOR_2D, UNIFORM_ANGLE, UNIFORM_BLOCK,
    UNIFORM_TENSOR, UNIFORM_THICKNESS, WL, SOURCE_BLOCK, key, material_of, raster_1d, raster_2d, rectangle_of)
from tools.repository_paths import repository_path
from wavebit import Material, Units
from wavebit import rcwa as rcwa_api
from wavebit.rcwa._factory import rcwa
from wavebit.units import Z0

GOLDEN = np.load(repository_path('tests/references/golden_reticolo_rule_paths.npz'))

# Measured over every comparison: the worst gap is 5.9e-11. Round-off room above that, not slack.
TOL = 1e-10
RULES = {'lalanne': 1, 'plain': 0}
X, Y, X_1D = GOLDEN['x_2d'], GOLDEN['y_2d'], GOLDEN['x_1d']
SLOTS_1D = {'te': ((1, 0), (3, 1), (5, 2)), 'tm': ((4, 0), (0, 1), (2, 2))}


def _rows(path, n_layers):
    """The layers' rows of ``res3``'s cell and their heights, both faces just inside each layer.

    ``res3`` repeats each interface once per layer, and its heights are a sum that can round to
    either side of this project's, which would read the row from the layer next door.
    """
    rows = np.arange(NPTS, NPTS * (n_layers + 1))
    z = GOLDEN[f'z_{path}'][rows].copy()
    z[np.arange(NPTS - 1, NPTS * n_layers, NPTS)] -= 1e-9
    z[np.arange(0, NPTS * n_layers, NPTS)] += 1e-9
    return rows, z


def _gap(ours, theirs):
    return np.abs(ours - theirs).max() / np.abs(theirs).max()


def _material(name, nudge=0.0):
    e, m = material_of(name)
    eps = e[0] * (1 + nudge) if e[0] == e[1] == e[2] else list(e)
    if m == (1.0, 1.0, 1.0):
        return Material(eps)
    return Material(eps, mu=m[0] if m[0] == m[1] == m[2] else list(m))


def _shapes(stack):
    return [dict(background=_material(layer['background'], MERGE_NUDGE if layer.get('merges') else 0.0),
                 shapes=[dict(zip(('cx', 'cy', 'lx', 'ly'), rectangle_of(block)), type='rectangle',
                              material=_material(block[4])) for block in layer['blocks']])
            for layer in stack[::-1]]


def _solved(pol, structure, stack, angle=ANGLE_2D, orders=(ORDER_2D, ORDER_2D), formulation='auto', **kw):
    a_p, a_s = EINC[pol]
    s = rcwa(backend='numpy', top=N_TOP ** 2, bot=N_BOT ** 2, period=list(PERIOD_2D), truncation_order=list(orders),
             wavelength=WL, thickness=[layer['thickness'] for layer in stack[::-1]], structure=structure,
             theta=np.deg2rad(angle[0]), phi=np.deg2rad(angle[1]), polarization=(a_s, a_p), factorization='li',
             answers='field', **kw)
    s.formulation = formulation
    s.solve()
    return s


def _compare_2d(s, path, pol, rule, champ, n_layers):
    rows, z = _rows(path, n_layers)
    ours = np.asarray(s.reconstruct_field(X, Y, z, reconstruction=rule))[0]
    theirs = np.transpose(GOLDEN[key(path, pol, champ)][rows], (0, 2, 1, 3))
    # RETICOLO leaves E_z and H_z unset inside a patterned layer under champ = 0.
    gaps = {NAMES[k]: _gap(ours[..., k], theirs[..., k]) for k in (range(6) if champ else (0, 1, 3, 4))}
    assert all(gap < TOL for gap in gaps.values()), gaps


@pytest.mark.parametrize('formulation', ['second_order', 'first_order'])
@pytest.mark.parametrize('rule,champ', RULES.items())
@pytest.mark.parametrize('pol', list(EINC))
@pytest.mark.parametrize('kind', ['shapes', 'raster'])
def test_a_diagonal_tensor(kind, pol, rule, champ, formulation):
    if kind == 'shapes':
        s = _solved(pol, _shapes(TENSOR_2D), TENSOR_2D, formulation=formulation)
    else:
        eps, mu = raster_2d(TENSOR_2D)
        s = _solved(pol, eps, TENSOR_2D, mu=mu, fourier_coefficients='integrated', formulation=formulation)
    assert s.formulation_assigned.value == formulation
    _compare_2d(s, 'tensor_2d', pol, rule, champ, len(TENSOR_2D))


def _solved_1d(path, stack, pol, tensor=True, **kw):
    conical = path == 'conical'
    eps, mu = raster_1d(stack)
    if not tensor:
        eps, mu = eps[..., 0, 0], mu[..., 0, 0]
    if conical:
        incidence = dict(theta=np.deg2rad(ANGLE_CONICAL[0]), phi=np.deg2rad(ANGLE_CONICAL[1]))
    else:
        incidence = dict(theta=np.deg2rad(ANGLE_1D), phi=0.0 if tensor else None)
    if tensor or conical:
        incidence['polarization'] = (1.0, 0.0) if pol == 's' else (0.0, 1.0)
    else:
        incidence['tetm'] = path
    s = rcwa(backend='numpy', top=N_TOP ** 2, bot=N_BOT ** 2, period=[PERIOD_1D], truncation_order=[ORDER_1D],
             wavelength=WL, thickness=[layer['thickness'] for layer in stack[::-1]], structure=eps, mu=mu,
             factorization='li', fourier_coefficients='integrated', answers='field', **incidence, **kw)
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
@pytest.mark.parametrize('path,pol', [('te', 's'), ('tm', 'p')])
def test_a_diagonal_tensor_on_a_grating_at_phi_zero(path, pol, rule, champ):
    """A tensor scene takes the 2D layout, so this is ``field_2d`` with one y order."""
    s = _solved_1d(path, TENSOR_1D, pol)
    rows, z = _rows(f'tensor_{path}', len(TENSOR_1D))
    ours = np.asarray(s.reconstruct_field(X_1D - PERIOD_1D / 2, np.array([0.0]), z, reconstruction=rule))[0][:, 0]
    theirs = GOLDEN[key(f'tensor_{path}', pol, champ)][rows] * _corner_phase(path)
    # `res3`'s pure 1D branch rebuilds the wall's normal component under either champ.
    pairs = SLOTS_1D[path] if champ else SLOTS_1D[path][:1]
    gaps = {NAMES[ours_k]: _gap(ours[..., ours_k], theirs[..., k]) for ours_k, k in pairs}
    assert all(gap < TOL for gap in gaps.values()), gaps


@pytest.mark.parametrize('rule,champ', RULES.items())
@pytest.mark.parametrize('pol', list(EINC))
def test_a_diagonal_tensor_on_a_conical_grating(pol, rule, champ):
    s = _solved_1d('conical', TENSOR_1D, pol)
    rows, z = _rows('tensor_conical', len(TENSOR_1D))
    ours = np.asarray(s.reconstruct_field(X_1D - PERIOD_1D / 2, np.array([0.0]), z, reconstruction=rule))[0][:, 0]
    theirs = GOLDEN[key('tensor_conical', pol, champ)][rows][:, :, 0] * _corner_phase('conical')
    gaps = {NAMES[k]: _gap(ours[..., k], theirs[..., k]) for k in (range(6) if champ else (0, 1, 3, 4))}
    assert all(gap < TOL for gap in gaps.values()), gaps


@pytest.mark.parametrize('pol', list(EINC))
@pytest.mark.parametrize('pair,formulation', [('xy', 'second_order'), ('xz', 'first_order')])
def test_a_uniform_tensor_beside_a_patterned_layer(pair, formulation, pol):
    """It has no walls, so it keeps the sum whichever rule its neighbour reads, and RETICOLO's own ``champ = 1``."""
    stack = [dict(thickness=UNIFORM_THICKNESS[0], blocks=[UNIFORM_BLOCK]), dict(thickness=UNIFORM_THICKNESS[1])]
    structure = [dict(background=Material(UNIFORM_TENSOR[pair]), shapes=[]),
                 dict(background=Material(1.0), shapes=[dict(zip(('cx', 'cy', 'lx', 'ly'), rectangle_of(UNIFORM_BLOCK)),
                                                             type='rectangle', material=_material(UNIFORM_BLOCK[4]))])]
    s = _solved(pol, structure, stack, angle=UNIFORM_ANGLE)
    assert s.formulation_assigned.value == formulation
    path = f'uniform_{pair}'
    rows, z = _rows(path, 2)
    below = z < UNIFORM_THICKNESS[1]
    rule = np.asarray(s.reconstruct_field(X, Y, z, reconstruction='lalanne'))[0]
    plain = np.asarray(s.reconstruct_field(X, Y, z))[0]
    one, zero = (np.transpose(GOLDEN[key(path, pol, champ)][rows], (0, 2, 1, 3)) for champ in (1, 0))
    gaps = {NAMES[k]: _gap(rule[..., k], one[..., k]) for k in range(6)}
    gaps.update({f'plain {NAMES[k]}': _gap(plain[~below][..., k], zero[~below][..., k]) for k in (0, 1, 3, 4)})
    assert all(gap < TOL for gap in gaps.values()), gaps
    np.testing.assert_array_equal(rule[below], plain[below])


def _merged_raster(stack):
    eps, mu = raster_2d(stack)
    eps, mu = eps[..., 0, 0].copy(), mu[..., 0, 0]
    for i, layer in enumerate(stack[::-1]):
        if layer['merges']:
            eps[i] = eps[i] * (1 + MERGE_NUDGE)
    return eps, mu


@pytest.mark.parametrize('rule,champ', RULES.items())
@pytest.mark.parametrize('pol', list(EINC))
@pytest.mark.parametrize('kind', ['shapes', 'raster'])
def test_layers_merged_into_a_half_space(kind, pol, rule, champ):
    if kind == 'shapes':
        structure, kw = _shapes(MERGED_2D), {}
    else:
        structure, mu = _merged_raster(MERGED_2D)
        kw = dict(mu=mu, fourier_coefficients='integrated')
    s = _solved(pol, structure, MERGED_2D, half_space_merge_tolerance=MERGE_TOLERANCE, **kw)
    assert s.solve()._field_state.merged[:2] == (1, 2)
    _compare_2d(s, 'merged_2d', pol, rule, champ, len(MERGED_2D))
    rows, z = _rows('merged_2d', len(MERGED_2D))
    state = np.asarray(s.solve()._field_state.reconstruct(X, Y, z, 'incident', rule))
    door = np.asarray(s.reconstruct_field(X, Y, z, reconstruction=rule))[0]
    np.testing.assert_array_equal(state, door)


@pytest.mark.parametrize('path,pol', [('te', 's'), ('tm', 'p'), ('conical', 's'), ('conical', 'p')])
def test_layers_merged_into_a_half_space_on_a_grating(path, pol):
    stack = [dict(layer) for layer in MERGED_1D]
    eps, mu = raster_1d(stack)
    eps, mu = eps[..., 0, 0].copy(), mu[..., 0, 0]
    for i, layer in enumerate(stack[::-1]):
        if layer['merges']:
            eps[i] = eps[i] * (1 + MERGE_NUDGE)
    conical = path == 'conical'
    a = (1.0, 0.0) if pol == 's' else (0.0, 1.0)
    incidence = (dict(theta=np.deg2rad(ANGLE_CONICAL[0]), phi=np.deg2rad(ANGLE_CONICAL[1]), polarization=a) if conical
                 else dict(theta=np.deg2rad(ANGLE_1D), phi=None, tetm=path))
    s = rcwa(backend='numpy', top=N_TOP ** 2, bot=N_BOT ** 2, period=[PERIOD_1D], truncation_order=[ORDER_1D],
             wavelength=WL, thickness=[layer['thickness'] for layer in stack[::-1]], structure=eps, mu=mu,
             factorization='li', fourier_coefficients='integrated', answers='field',
             half_space_merge_tolerance=MERGE_TOLERANCE, **incidence)
    state = s.solve()._field_state
    assert state.merged[:2] == (1, 2)
    golden_path = f'merged_{path}'
    rows, z = _rows(golden_path, len(stack))
    x = X_1D - PERIOD_1D / 2
    field = np.asarray(state.reconstruct(x, np.array([0.0]), z, 'incident', 'lalanne'))[:, 0]
    theirs = GOLDEN[key(golden_path, pol, 1)][rows] * _corner_phase(path)
    if conical:
        gaps = {NAMES[k]: _gap(field[..., k], theirs[:, :, 0, k]) for k in range(6)}
    else:
        gaps = {NAMES[ours]: _gap(field[..., ours], theirs[..., k]) for ours, k in SLOTS_1D[path]}
    assert all(gap < TOL for gap in gaps.values()), gaps
    door = np.asarray(s.reconstruct_field(x, np.array([0.0]), z, reconstruction='lalanne'))
    if conical:
        np.testing.assert_array_equal(door[0, :, 0], field)
    else:
        np.testing.assert_array_equal(door[:, 0], field[..., [ours for ours, _ in SLOTS_1D[path]]])


@pytest.mark.parametrize('rule,champ', RULES.items())
@pytest.mark.parametrize('pol', list(EINC))
@pytest.mark.parametrize('axis,orders', [('x', (ORDER_2D, 0)), ('y', (0, ORDER_2D))])
@pytest.mark.parametrize('kind', ['shapes', 'raster'])
def test_one_order_along_an_axis(kind, axis, orders, pol, rule, champ):
    """The lines still read the walls the solve cannot resolve along that axis, and so do RETICOLO's."""
    if kind == 'shapes':
        structure, kw = _shapes(ONE_ORDER_2D), {}
    else:
        eps, mu = raster_2d(ONE_ORDER_2D)
        structure, kw = eps[..., 0, 0], dict(mu=mu[..., 0, 0], fourier_coefficients='integrated')
    s = _solved(pol, structure, ONE_ORDER_2D, orders=orders, **kw)
    _compare_2d(s, f'one_{axis}', pol, rule, champ, len(ONE_ORDER_2D))


@pytest.mark.parametrize('rule,champ', RULES.items())
@pytest.mark.parametrize('pol', list(EINC))
def test_incidence_from_below(pol, rule, champ):
    k_par = N_TOP * np.sin(np.deg2rad(ANGLE_2D[0]))
    angle = (np.rad2deg(np.arcsin(k_par / N_BOT)), ANGLE_2D[1])
    # `sens = -1` keeps `s` and turns `p` over: the amplitude's sign, measured, is all of it.
    a_p, a_s = EINC[pol]
    s = rcwa(backend='numpy', top=N_TOP ** 2, bot=N_BOT ** 2, period=list(PERIOD_2D),
             truncation_order=[ORDER_2D, ORDER_2D], wavelength=WL,
             thickness=[layer['thickness'] for layer in TENSOR_2D[::-1]], structure=_shapes(TENSOR_2D),
             theta=np.deg2rad(angle[0]), phi=np.deg2rad(angle[1]), polarization=(a_s, -a_p), factorization='li',
             answers='field', incidence_side='bottom')
    s.solve()
    _compare_2d(s, 'bottom_2d', pol, rule, champ, len(TENSOR_2D))


def _dipole(component):
    host = rcwa_api.Layer(thickness=SOURCE_HOST[1] + SOURCE_ROW + SOURCE_HOST[0],
                          background=Material(SOURCE_HOST_INDEX ** 2))
    cx, cy, lx, ly, n = SOURCE_BLOCK
    patterned = rcwa_api.Layer(thickness=SOURCE_PATTERNED, background=Material(N_TOP ** 2), regions=(rcwa_api.Region(
        geometry=rcwa_api.Rectangle(size=(lx, ly), center=(cx, cy)), material=Material(n ** 2)),))
    source = rcwa_api.ElectricDipole(position=(*SOURCE_POSITION, SOURCE_HOST[1] + SOURCE_ROW / 2),
                                     moment=SOURCE_MOMENTS[component])
    problem = rcwa_api.Problem(
        units=Units(length='nm'),
        stack=rcwa_api.Stack(period=tuple(PERIOD_2D), top=Material(N_TOP ** 2), bottom=Material(N_BOT ** 2),
                             layers=(host, patterned)),
        excitation=rcwa_api.Excitation(wavelength=WL, sources=(source,)),
        outputs=rcwa_api.Outputs(fields=True, diffraction=False))
    return rcwa_api.run(problem, numerics=rcwa_api.Numerics(orders=(SOURCE_ORDER, SOURCE_ORDER), factorization='li'))


@pytest.mark.parametrize('component', list(SOURCE_MOMENTS))
def test_a_dipole_under_a_patterned_layer(component):
    result = _dipole(component)
    z = GOLDEN['z_source'].copy()
    z[[5, 8]] -= 1e-9
    z = z[3:]
    plain = np.array(result.field(x=X, y=Y, z=z).values[0])
    rule = np.array(result.field(x=X, y=Y, z=z, reconstruction='lalanne').values[0])
    plain[..., 3:] *= Z0
    rule[..., 3:] *= Z0
    zero, one = (np.transpose(GOLDEN[f'source_{component}_{c}'][3:], (0, 2, 1, 3)) for c in (0, 1))
    comps = (0, 1, 3, 4)
    scale = np.vdot(plain[..., comps], zero[..., comps]) / np.vdot(plain[..., comps], plain[..., comps])
    gaps = {f'plain {NAMES[k]}': _gap(scale * plain[..., k], zero[..., k]) for k in comps}
    gaps.update({NAMES[k]: _gap(scale * rule[..., k], one[..., k]) for k in range(6)})
    assert all(gap < TOL for gap in gaps.values()), gaps
    assert _gap(one[3:, ..., 0], zero[3:, ..., 0]) > 1e-2
