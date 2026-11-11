"""The Lalanne rule off its first paths, where no reference is needed or none is left to ask.

``tests/references/test_reticolo_rule_paths_golden.py`` pins these paths against RETICOLO. This file
holds what RETICOLO has no word on: a merged solve at a complex angle, which RETICOLO cannot light,
against the same stack with the layers deleted; the first-order path against the second-order one;
a source against the ordinary solve of the same wave, with several Bloch groups; derivatives, the
backends, and what stays refused.
"""
import pickle

import numpy as np
import pytest

from wavebit import Material, Units
from wavebit import rcwa as rcwa_api
from wavebit.rcwa._factory import rcwa
from wavebit.units import Z0

PX, PY, WL = 400.0, 360.0, 620.0
X = np.array([-171.3, -40.7, 33.1, 120.9])
Y = np.array([-133.1, 17.3, 101.9])
OFF = np.array([[3.0, 0.3, 0.0], [0.3, 3.0, 0.0], [0.0, 0.0, 2.5]], dtype=complex)


def _rect(cx=-20.0, cy=30.0, lx=150.0, ly=110.0, material=4.0):
    return dict(type='rectangle', cx=cx, cy=cy, lx=lx, ly=ly, material=material)


def _raster(values, n_y=9, n_x=8):
    out = np.ones((len(values), n_y, n_x), dtype=complex)
    for i, value in enumerate(values):
        if value is None:
            out[i] = 1.0
            out[i, 2:6, 1:5] = 4.0
        else:
            out[i] = value
    return out


def _solver(structure, thickness, **kw):
    args = dict(backend='numpy', top=1.0, bot=2.25, period=[PX, PY], truncation_order=[3, 3], wavelength=WL,
                thickness=thickness, structure=structure, theta=0.3, phi=0.2, polarization=(0.6, 0.8),
                factorization='li', answers='field')
    args.update(kw)
    if 'lattice' in kw:
        del args['period']
    return rcwa(**args)


def _solved(structure, thickness, formulation='auto', **kw):
    s = _solver(structure, thickness, **kw)
    s.formulation = formulation
    s.solve()
    return s


# ---------------------------------------------------------------------------------- first order

@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('kind', ['raster', 'shapes', 'magnetic'])
def test_the_first_order_rule_is_the_second_order_rule(kind, side):
    structure = {'raster': _raster([None, 2.25]),
                 'shapes': [dict(background=1.0, shapes=[_rect()]), dict(background=2.25, shapes=[])],
                 'magnetic': [dict(background=1.0, shapes=[_rect(material=Material(4.0, mu=2.25))]),
                              dict(background=2.25, shapes=[])]}[kind]
    z = np.array([33.0, 90.0, 140.0, 170.0])
    second, first = (_solved(structure, [120.0, 60.0], formulation=f, incidence_side=side)
                     for f in ('second_order', 'first_order'))
    assert first.solve()._field_state.kind == 'first_order'
    a, b = (np.asarray(s.reconstruct_field(X, Y, z, reconstruction='lalanne')) for s in (second, first))
    assert np.abs(a - b).max() < 1e-10 * np.abs(a).max()
    assert np.abs(b - np.asarray(first.reconstruct_field(X, Y, z))).max() > 1e-3
    state = np.asarray(first.solve()._field_state.reconstruct(X, Y, z, 'incident', 'lalanne'))
    assert np.abs(state - b[0]).max() < 1e-12 * np.abs(b).max()


def test_a_skewed_first_order_solve_is_refused_before_a_line_is_read():
    import wavebit
    s = rcwa(backend='numpy', top=1.0, bot=1.0, lattice=wavebit.Lattice.hexagonal(PX), truncation_order=[2, 2],
             wavelength=WL, thickness=[100.0], structure=_raster([None]), factorization='nvf', answers='field')
    s.formulation = 'first_order'
    s.solve()
    with pytest.raises(NotImplementedError, match='skewed lattice'):
        s.reconstruct_field(reconstruction='lalanne')


# ------------------------------------------------------------------------------------ one order

@pytest.mark.parametrize('orders', [(3, 0), (0, 3)])
def test_the_lines_read_walls_the_solve_keeps_one_order_across(orders):
    """A shape and the raster that draws it answer alike under the rule's own Li, where they agree."""
    px, py = PX / 8, PY / 9
    shape = _rect(cx=3 * px - PX / 2, cy=4 * py - PY / 2, lx=4 * px, ly=4 * py)
    a = _solved(_raster([None]), [100.0], truncation_order=list(orders), fourier_coefficients='integrated')
    b = _solved([dict(background=1.0, shapes=[shape])], [100.0], truncation_order=list(orders))
    z = np.array([33.0, 66.0])
    one, two = (np.asarray(s.reconstruct_field(X, Y, z, reconstruction='lalanne')) for s in (a, b))
    assert np.abs(one - two).max() < 1e-11 * np.abs(one).max()
    # The axis with one order has no phase to tell its samples apart: only the lines do.
    across = one[0, 0, :, :, 0] if orders[1] == 0 else one[0, 0, :, :, 1].T
    assert np.abs(across[0] - across[1]).max() + np.abs(across[0] - across[2]).max() > 1e-3


# -------------------------------------------------------------------------------------- merging

def _grating(kind, values, thickness, side, **kw):
    grid = (len(values), 9, 8) if kind == '2d' else (len(values), 1, 8)
    structure = np.ones(grid, dtype=complex)
    mu = np.ones(grid, dtype=complex)
    for i, value in enumerate(values):
        if value is None:
            structure[i, ..., :4] = 4.0
            mu[i, ..., 2:6] = 2.25
        else:
            structure[i] = value
    args = dict(backend='numpy', top=1.0, bot=2.25, theta=0.9 + 0.25j, wavelength=WL, thickness=thickness,
                structure=structure, mu=mu, incidence_side=side, answers='field', factorization='li')
    if kind == '2d':
        args.update(polarization=(0.6, 0.8), phi=0.3, truncation_order=[2, 2], period=[PX, PY])
    elif kind == 'conical':
        args.update(polarization=(0.6, 0.8), phi=0.3, truncation_order=[3], period=[PX])
    else:
        args.update(tetm=kind, truncation_order=[3], period=[PX])
    args.update(kw)
    return rcwa(**args)


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('kind', ['te', 'tm', 'conical', '2d'])
def test_a_merged_rule_is_the_deleted_stacks_times_one_phase(kind, side):
    """RETICOLO cannot light a complex angle, so the reference is the theory of a merge itself.

    The kept layer's amplitudes are the deleted stack's times the phase the incident wave gathers
    crossing the merged layers, which the sum states and the rule must then hold. Below, and off by
    a row, the 1D rule is 0.25 to 0.31 away.
    """
    th = [300.0, 220.0, 150.0]
    full = _grating(kind, [2.25, None, 1.0], th, side)
    alone = _grating(kind, [None], th[1:2], side)
    state, single = full.solve(auto_detune=False)._field_state, alone.solve(auto_detune=False)._field_state
    assert state.merged[:2] == (1, 1)
    x, y, z = X, (Y if kind == '2d' else np.array([0.0])), np.array([41.0, 150.0])
    plain_full, plain_alone = (np.asarray(s.reconstruct(x, y, h, 'incident')) for s, h in (
        (state, z + th[0]), (single, z)))
    rule_full, rule_alone = (np.asarray(s.reconstruct(x, y, h, 'incident', 'lalanne')) for s, h in (
        (state, z + th[0]), (single, z)))
    phase = plain_full.ravel()[np.argmax(np.abs(plain_alone))] / plain_alone.ravel()[np.argmax(np.abs(plain_alone))]
    assert np.abs(plain_full - phase * plain_alone).max() < 1e-11 * np.abs(plain_full).max()
    assert np.abs(rule_full - phase * rule_alone).max() < 1e-11 * np.abs(rule_full).max()
    assert np.abs(rule_full - plain_full).max() > 1e-3 * np.abs(rule_full).max()
    merged = np.array([100.0, 600.0])
    np.testing.assert_array_equal(state.reconstruct(x, y, merged, 'incident', 'lalanne'),
                                  state.reconstruct(x, y, merged, 'incident'))


def test_a_stack_that_merged_whole_has_nothing_for_the_rule_to_change():
    s = _grating('2d', [2.25, 1.0], [300.0, 150.0], 'top')
    state = s.solve(auto_detune=False)._field_state
    assert state.merged[:2] == (1, 1) and not state.data['groups']
    z = np.array([100.0, 400.0])
    np.testing.assert_array_equal(state.reconstruct(X, Y, z, 'incident', 'lalanne'), state.reconstruct(X, Y, z, 'incident'))
    np.testing.assert_array_equal(s.reconstruct_field(X, Y, z, reconstruction='lalanne'), s.reconstruct_field(X, Y, z))


def test_a_curve_in_a_merged_layer_is_read_by_nothing():
    """Drawn in the half-space's own material it is uniform, and the refusal keeps the public index."""
    circle = dict(type='circle', cx=0.0, cy=0.0, l=120.0, material=2.25 * (1 + 3e-10))
    kept = _rect()
    s = _solver([dict(background=2.25, shapes=[circle]), dict(background=1.0, shapes=[kept])], [80.0, 100.0],
                half_space_merge_tolerance=1e-9, factorization='nvf')
    state = s.solve()._field_state
    assert state.merged[:2] == (1, 0)
    assert np.isfinite(np.asarray(state.reconstruct(X, Y, np.array([120.0]), 'incident', 'lalanne'))).all()
    t = _solver([dict(background=2.25, shapes=[circle]), dict(background=1.0, shapes=[
        dict(type='circle', cx=0.0, cy=0.0, l=120.0, material=4.0)])], [80.0, 100.0], half_space_merge_tolerance=1e-9,
                factorization='nvf')
    state = t.solve()._field_state
    with pytest.raises(NotImplementedError, match='Layer 1 shape 0'):
        state.reconstruct(X, Y, np.array([120.0]), 'incident', 'lalanne')
    with pytest.raises(NotImplementedError, match='Layer 1 shape 0'):
        t.reconstruct_field(X, Y, np.array([120.0]), reconstruction='lalanne')


def test_a_jitted_run_merges_and_answers_what_the_eager_one_answers():
    """A trace cannot ask whether a real angle stays clear of the knife edge, so jit merges where eager did not."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)

    def problem(cx):
        return rcwa_api.Problem(
            units=Units(length='nm'),
            stack=rcwa_api.Stack(period=(PX, PY), top=Material(1.0), bottom=Material(2.25), layers=(
                rcwa_api.Layer(thickness=80.0, background=Material(2.25)),
                rcwa_api.Layer(thickness=100.0, background=Material(1.0), regions=(rcwa_api.Region(
                    geometry=rcwa_api.Rectangle(size=(150.0, 110.0), center=(cx, 30.0)), material=Material(4.0)),)))),
            excitation=rcwa_api.Excitation(wavelength=WL, sources=(rcwa_api.PlaneWave(theta=0.2, polarization='p'),)),
            outputs=rcwa_api.Outputs(fields=True, diffraction=False))
    numerics = rcwa_api.Numerics(orders=(2, 2), factorization='li')
    eager = rcwa_api.run(problem(-20.0), numerics=numerics, backend='jax')
    assert eager._field_records[0].merged is None
    prepared = rcwa_api.prepare(problem(-20.0), numerics=numerics, backend='jax')
    jitted = jax.jit(lambda cx: prepared.run(problem(cx)))(-20.0)
    assert jitted._field_records[0].merged is not None
    z = np.array([120.0, 160.0])
    a, b = (np.asarray(r.field(x=X, y=Y, z=z, reconstruction='lalanne').values) for r in (eager, jitted))
    assert np.abs(a - b).max() < 1e-10 * np.abs(a).max()


def test_a_merged_thickness_keeps_its_derivative_through_the_rule():
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    def power(d):
        s = _grating('2d', [2.25, None, 1.0], jnp.array([300.0, 220.0, 150.0]) + jnp.array([d, 0.0, 0.0]), 'top',
                     backend='jax')
        state = s.solve(auto_detune=False)._field_state
        return jnp.sum(jnp.abs(state.reconstruct(X, Y, np.array([341.0]), 'incident', 'lalanne')[..., 0]) ** 2)
    grad = float(jax.grad(power)(0.0))
    h = 1e-3
    fd = float((power(h) - power(-h)) / (2 * h))
    assert grad == pytest.approx(fd, rel=1e-5)
    assert abs(grad) > 0


# -------------------------------------------------------------------------------------- tensors

@pytest.mark.parametrize('formulation', ['second_order', 'first_order'])
def test_eps_times_the_identity_is_the_scalar_rule(formulation):
    z = np.array([33.0, 90.0])
    scalar = _solved([dict(background=1.0, shapes=[_rect(material=Material(4.0, mu=2.25))])], [120.0],
                     formulation=formulation)
    tensor = _solved([dict(background=Material([1.0, 1.0, 1.0]), shapes=[
        _rect(material=Material([4.0, 4.0, 4.0], mu=[2.25, 2.25, 2.25]))])], [120.0], formulation=formulation)
    a, b = (np.asarray(s.reconstruct_field(X, Y, z, reconstruction='lalanne')) for s in (scalar, tensor))
    assert np.abs(a - b).max() < 1e-11 * np.abs(a).max()


def test_each_axis_reads_its_own_entry():
    """Swapping ``eps_xx`` and ``eps_yy`` moves the answer, so a test of them could fail."""
    kept = _solved([dict(background=1.0, shapes=[_rect(material=Material([4.0, 9.0, 2.25]))])], [120.0])
    ops_x, _, eps_x, eps_y = kept._lalanne_rows(X, Y, True)[0].electric
    np.testing.assert_array_equal(np.unique(eps_x), [1.0, 4.0])
    np.testing.assert_array_equal(np.unique(eps_y), [1.0, 9.0])
    assert ops_x.shape == (Y.size, 7, 7)
    swapped = _solved([dict(background=1.0, shapes=[_rect(material=Material([9.0, 4.0, 2.25]))])], [120.0])
    a, b = (np.asarray(s.reconstruct_field(X, Y, np.array([33.0]), reconstruction='lalanne')) for s in (kept, swapped))
    assert np.abs(a - b).max() > 1e-2


@pytest.mark.parametrize('make', [
    lambda: [dict(background=1.0, shapes=[_rect(material=Material(OFF))])],
    lambda: _raster_tensor(off_in=True),
    lambda: [dict(background=Material(OFF), shapes=[_rect()])],
    lambda: [dict(background=1.0, shapes=[_rect(material=Material(4.0, mu=OFF))])],
    lambda: [dict(background=Material(1.0, mu=OFF), shapes=[_rect()])],
], ids=['shape', 'raster', 'background', 'shape_mu', 'background_mu'])
def test_a_patterned_layer_with_off_diagonals_is_refused(make):
    s = _solved(make(), [120.0])
    with pytest.raises(NotImplementedError, match='off-diagonal'):
        s.reconstruct_field(X, Y, np.array([33.0]), reconstruction='lalanne')


def _raster_tensor(off_in=False):
    grid = np.broadcast_to(np.eye(3, dtype=complex), (1, 9, 8, 3, 3)).copy()
    grid[0, 2:6, 1:5] = OFF if off_in else np.diag([4.0, 9.0, 2.25])
    return grid


def test_a_diagonal_raster_takes_the_rule_and_a_zero_off_diagonal_is_read_by_value():
    s = _solved(_raster_tensor(), [120.0], fourier_coefficients='integrated')
    field = np.asarray(s.reconstruct_field(X, Y, np.array([33.0]), reconstruction='lalanne'))
    assert np.abs(field - np.asarray(s.reconstruct_field(X, Y, np.array([33.0])))).max() > 1e-3


def test_a_differentiated_tensor_raster_is_refused_since_a_zero_can_carry_a_derivative():
    torch = pytest.importorskip('torch')
    leaf = torch.tensor(np.real(_raster_tensor()), dtype=torch.float64, requires_grad=True)
    s = _solved(leaf, [120.0], backend='torch', fourier_coefficients='integrated')
    with pytest.raises(NotImplementedError, match='off-diagonal'):
        s.reconstruct_field(X, Y, np.array([33.0]), reconstruction='lalanne')


def test_principal_values_keep_their_derivative_through_the_rule():
    jax = pytest.importorskip('jax')
    def power(e):
        s = _solver([dict(background=1.0, shapes=[_rect(material=Material([e, 9.0, 2.25]))])], [120.0], backend='jax')
        s.solve()
        cell = s.reconstruct_field(X, Y, np.array([33.0]), reconstruction='lalanne')
        return jax.numpy.sum(jax.numpy.abs(cell[..., 0]) ** 2)
    grad = float(jax.grad(power)(4.0))
    h = 1e-4
    assert grad == pytest.approx(float((power(4.0 + h) - power(4.0 - h)) / (2 * h)), rel=1e-5)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_a_diagonal_tensor_answers_alike_on_each_backend(backend):
    pytest.importorskip(backend)
    structure = [dict(background=Material([1.0, 2.0, 3.0]), shapes=[_rect(material=Material([4.0, 9.0, 2.25], mu=[1.2, 1.3, 1.4]))])]
    z = np.array([33.0])
    a, b = (np.asarray(_solved(structure, [120.0], backend=name).reconstruct_field(X, Y, z, reconstruction='lalanne'))
            for name in ('numpy', backend))
    assert np.abs(a - b).max() < 1e-9 * np.abs(a).max()


def test_a_sampled_layer_with_no_mu_beside_a_tensor_layer_has_the_identity():
    """All nine entries used to read one, so the solve itself saw an off-diagonal ``mu``."""
    epsilon = np.broadcast_to(np.eye(3, dtype=complex), (9, 8, 3, 3)).copy()
    epsilon[2:6, 1:5] = np.diag([4.0, 9.0, 2.25])
    mu = np.broadcast_to(np.eye(3, dtype=complex), (9, 8, 3, 3)).copy()
    mu[2:6, 1:5] = np.diag([1.2, 1.3, 1.4])
    problem = rcwa_api.Problem(
        units=Units(length='nm'),
        stack=rcwa_api.Stack(period=(PX, PY), top=Material(1.0), bottom=Material(2.25), layers=(
            rcwa_api.Layer(thickness=100.0, epsilon=epsilon), rcwa_api.Layer(thickness=80.0, epsilon=epsilon, mu=mu))),
        excitation=rcwa_api.Excitation(wavelength=WL, sources=(rcwa_api.PlaneWave(theta=0.2, polarization='p'),)),
        outputs=rcwa_api.Outputs(fields=True, diffraction=True))
    answer = rcwa_api.run(problem, numerics=rcwa_api.Numerics(orders=(2, 2), factorization='li'))
    twin = problem.replace(stack=problem.stack.replace(layers=(
        rcwa_api.Layer(thickness=100.0, epsilon=epsilon, mu=np.broadcast_to(np.eye(3, dtype=complex), (9, 8, 3, 3))),
        rcwa_api.Layer(thickness=80.0, epsilon=epsilon, mu=mu))))
    named = rcwa_api.run(twin, numerics=rcwa_api.Numerics(orders=(2, 2), factorization='li'))
    assert np.abs(np.asarray(answer.field(x=X, y=Y, z=np.array([33.0])).values)
                  - np.asarray(named.field(x=X, y=Y, z=np.array([33.0])).values)).max() < 1e-12


# -------------------------------------------------------------------------------------- sources

def _two_layers(region=True, mu=None):
    regions = (rcwa_api.Region(geometry=rcwa_api.Rectangle(size=(150.0, 110.0), center=(-20.0, 30.0)),
                               material=Material(4.0, mu=mu) if mu is not None else Material(4.0)),) if region else ()
    return (rcwa_api.Layer(thickness=200.0, background=Material(1.69)),
            rcwa_api.Layer(thickness=120.0, background=Material(1.0), regions=regions))


def _problem(*sources, layers=None):
    return rcwa_api.Problem(
        units=Units(length='nm'),
        stack=rcwa_api.Stack(period=(PX, PY), top=Material(1.0), bottom=Material(2.25),
                             layers=_two_layers() if layers is None else layers),
        excitation=rcwa_api.Excitation(wavelength=WL, sources=sources),
        outputs=rcwa_api.Outputs(fields=True, diffraction=False, power_region='cell'))


NUMERICS = rcwa_api.Numerics(orders=(3, 3), factorization='li')
Z = np.array([20.0, 260.0, 300.0])


@pytest.mark.parametrize('orders', [(3, 3), (3, 1), (1, 3)])
@pytest.mark.parametrize('mu', [None, 2.25])
def test_a_wave_as_a_source_answers_what_the_ordinary_solve_answers(mu, orders):
    """Two waves take the source's path, and the second carries nothing. Unequal orders hold the axes apart."""
    layers = _two_layers(mu=mu)
    numerics = rcwa_api.Numerics(orders=orders, factorization='li')
    lit = rcwa_api.PlaneWave(side='top', theta=0.2, phi=0.3, polarization='p')
    dark = rcwa_api.PlaneWave(side='bottom', theta=0.1, phi=0.3, polarization='p', amplitude=0.0)
    ordinary = rcwa_api.run(_problem(lit, layers=layers), numerics=numerics)
    source = rcwa_api.run(_problem(lit, dark, layers=layers), numerics=numerics)
    for rule in ('plain', 'lalanne'):
        a, b = (np.asarray(r.field(x=X, y=Y, z=Z, reconstruction=rule).values) for r in (ordinary, source))
        assert np.abs(a - b).max() < 1e-10 * np.abs(a).max(), rule
    assert np.abs(np.asarray(source.field(x=X, y=Y, z=Z, reconstruction='lalanne').values)
                  - np.asarray(source.field(x=X, y=Y, z=Z).values)).max() > 1e-3


def test_several_bloch_groups_add_up_under_the_rule():
    waves = (rcwa_api.PlaneWave(side='top', theta=0.2, phi=0.3, polarization='p'),
             rcwa_api.PlaneWave(side='top', theta=0.05, phi=0.3, polarization='s'))
    both = rcwa_api.run(_problem(*waves), numerics=NUMERICS)
    one, other = (rcwa_api.run(_problem(wave, rcwa_api.PlaneWave(side='bottom', theta=0.1, polarization='p', amplitude=0.0)),
                               numerics=NUMERICS) for wave in waves)
    get = lambda r: np.asarray(r.field(x=X, y=Y, z=Z, reconstruction='lalanne').values)
    assert np.abs(get(both) - get(one) - get(other)).max() < 1e-10 * np.abs(get(both)).max()


def test_a_dipole_reads_the_rule_off_a_raster_and_in_a_result_that_outlives_its_solve():
    dipole = rcwa_api.ElectricDipole(position=(31.0, -22.0, 100.0), moment=(1.0, 0.0, 0.0))
    grid = np.ones((9, 8), dtype=complex)
    grid[2:6, 1:5] = 4.0
    layers = (rcwa_api.Layer(thickness=200.0, epsilon=np.full((9, 8), 1.69 + 0j)), rcwa_api.Layer(thickness=120.0, epsilon=grid))
    result = rcwa_api.run(_problem(dipole, layers=layers), numerics=rcwa_api.Numerics(
        orders=(3, 3), factorization='li', fourier_coefficients='integrated'))
    field = result.field(x=X, y=Y, z=Z, reconstruction='lalanne')
    assert field.metadata['reconstruction'] == 'lalanne'
    plain = np.asarray(result.field(x=X, y=Y, z=Z).values)
    held = np.asarray(field.values)
    assert np.abs(held[:, 1:] - plain[:, 1:]).max() > 1e-3 * np.abs(plain).max()
    np.testing.assert_array_equal(held[:, 0], plain[:, 0])      # the uniform host keeps the sum
    state = pickle.loads(pickle.dumps(result._field_records[0]))
    np.testing.assert_array_equal(state.reconstruct(X, Y, Z, 'incident', 'lalanne'),
                                  result._field_records[0].reconstruct(X, Y, Z, 'incident', 'lalanne'))


def test_a_curve_in_a_source_solve_is_refused_by_name_and_a_skew_cell_too():
    import wavebit
    from wavebit import UnsupportedFeatureError
    dipole = rcwa_api.ElectricDipole(position=(31.0, -22.0, 100.0), moment=(1.0, 0.0, 0.0))
    curve = (rcwa_api.Layer(thickness=200.0, background=Material(1.69)),
             rcwa_api.Layer(thickness=120.0, background=Material(1.0), regions=(
                 rcwa_api.Region(geometry=rcwa_api.Circle(radius=80.0), material=Material(4.0)),)))
    result = rcwa_api.run(_problem(dipole, layers=curve), numerics=rcwa_api.Numerics(orders=(2, 2), factorization='nvf'))
    with pytest.raises(UnsupportedFeatureError, match='slanted or curved'):
        result.field(x=X, y=Y, z=Z, reconstruction='lalanne')
    skew = _problem(dipole).replace(stack=rcwa_api.Stack(lattice=wavebit.Lattice.hexagonal(PX), top=Material(1.0),
                                                         bottom=Material(2.25), layers=_two_layers(region=False)))
    result = rcwa_api.run(skew, numerics=rcwa_api.Numerics(orders=(2, 2), factorization='nvf'))
    with pytest.raises(UnsupportedFeatureError, match='skewed lattice'):
        result.field(x=X, y=Y, z=Z, reconstruction='lalanne')


def test_a_raster_mu_reaches_the_rule_on_a_source():
    grid = np.ones((9, 8), dtype=complex)
    grid[2:6, 1:5] = 4.0
    mu = np.ones((9, 8), dtype=complex)
    mu[2:6, 1:5] = 2.25
    layers = (rcwa_api.Layer(thickness=200.0, epsilon=np.full((9, 8), 1.69 + 0j)),
              rcwa_api.Layer(thickness=120.0, epsilon=grid, mu=mu))
    numerics = rcwa_api.Numerics(orders=(3, 3), factorization='li', fourier_coefficients='integrated')
    lit = rcwa_api.PlaneWave(side='top', theta=0.2, phi=0.3, polarization='p')
    dark = rcwa_api.PlaneWave(side='bottom', theta=0.1, phi=0.3, polarization='p', amplitude=0.0)
    a, b = (np.asarray(rcwa_api.run(_problem(*waves, layers=layers), numerics=numerics).field(
        x=X, y=Y, z=Z, reconstruction='lalanne').values) for waves in ((lit,), (lit, dark)))
    assert np.abs(a[..., 3] - b[..., 3]).max() < 1e-10 * np.abs(a[..., 3]).max()


def test_the_coherent_solvers_door_takes_the_rule():
    from wavebit.rcwa._engine.sources import PlaneWave
    s = rcwa('numpy', top=1.0, bot=2.25, period=[PX, PY], truncation_order=[3, 3], wavelength=WL, thickness=[120.0],
             structure=_raster([None]), factorization='li', answers='field',
             excitation={'top': PlaneWave(theta=0.25, phi=0.6, polarization=(1.0, 0.0))})
    result = s.solve()
    z = np.array([33.0])
    rule = np.asarray(s.reconstruct_field(X, Y, z, reconstruction='lalanne'))
    np.testing.assert_array_equal(rule, result.reconstruct_field(X, Y, z, reconstruction='lalanne'))
    assert np.abs(rule - np.asarray(s.reconstruct_field(X, Y, z))).max() > 1e-3


def test_a_source_lattice_asks_what_the_solve_did():
    """Rounding in a rectangular cell's columns is not a skew, and a matrix under a derivative can shear."""
    import wavebit
    from wavebit import UnsupportedFeatureError
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    lit = rcwa_api.PlaneWave(side='top', theta=0.2, phi=0.3, polarization='p')
    dark = rcwa_api.PlaneWave(side='bottom', theta=0.1, phi=0.3, polarization='p', amplitude=0.0)
    numerics = rcwa_api.Numerics(orders=(2, 2), factorization='li')
    rounded = _problem(lit, dark).replace(stack=rcwa_api.Stack(
        lattice=wavebit.Lattice.oblique(PX, PY, angle=np.pi / 2), top=Material(1.0), bottom=Material(2.25),
        layers=_two_layers()))
    result = rcwa_api.run(rounded, numerics=numerics)
    plain = rcwa_api.run(_problem(lit, dark), numerics=numerics)
    a, b = (np.asarray(r.field(x=X, y=Y, z=Z, reconstruction='lalanne').values) for r in (result, plain))
    assert np.abs(a - b).max() < 1e-9 * np.abs(b).max()

    def sheared(s):
        problem = _problem(lit, dark).replace(stack=rcwa_api.Stack(
            lattice=jax.numpy.array([[PX, s], [0.0, PY]]), top=Material(1.0), bottom=Material(2.25),
            layers=_two_layers(region=False)))
        return jax.numpy.sum(jax.numpy.abs(jax.numpy.asarray(rcwa_api.run(
            problem, numerics=rcwa_api.Numerics(orders=(1, 1), factorization='nvf'), backend='jax').field(
                x=X, y=Y, z=Z, reconstruction='lalanne').values)) ** 2)
    with pytest.raises(UnsupportedFeatureError, match='skewed lattice'):
        jax.grad(sheared)(0.0)


def test_a_layer_of_no_thickness_changes_nothing_on_a_source():
    """Its midpoint used to fall in the next layer, whose field was then added twice."""
    lit = rcwa_api.PlaneWave(side='top', theta=0.2, phi=0.3, polarization='p')
    dark = rcwa_api.PlaneWave(side='bottom', theta=0.1, phi=0.3, polarization='p', amplitude=0.0)
    dipole = rcwa_api.ElectricDipole(position=(31.0, -22.0, 100.0), moment=(1.0, 0.0, 0.0))
    first, second = _two_layers()
    empty = rcwa_api.Layer(thickness=0.0, background=Material(9.0))
    for sources in ((lit, dark), (dipole,)):
        with_it, without = (rcwa_api.run(_problem(*sources, layers=layers), numerics=NUMERICS)
                            for layers in ((first, empty, second), (first, second)))
        for rule in ('plain', 'lalanne'):
            a, b = (np.asarray(r.field(x=X, y=Y, z=Z, reconstruction=rule).values) for r in (with_it, without))
            assert np.abs(a - b).max() < 1e-12 * np.abs(b).max(), rule


def test_a_source_field_keeps_a_layers_thickness_on_its_derivative():
    """The heights were read off concrete segments: the sign of this derivative came out wrong."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    numerics = rcwa_api.Numerics(orders=(1, 1), factorization='li')

    def power(thickness):
        layers = (rcwa_api.Layer(thickness=thickness, background=Material(1.69)), _two_layers()[1])
        lit = rcwa_api.PlaneWave(side='top', theta=0.2, phi=0.3, polarization='p')
        dark = rcwa_api.PlaneWave(side='bottom', theta=0.1, phi=0.3, polarization='p', amplitude=0.0)
        result = rcwa_api.run(_problem(lit, dark, layers=layers), numerics=numerics, backend='jax')
        return jax.numpy.sum(jax.numpy.abs(result.field(x=X, y=Y, z=np.array([260.0]),
                                                        reconstruction='lalanne').values[..., 0]) ** 2)
    h = 1e-3
    grad = float(jax.grad(power)(200.0))
    assert grad == pytest.approx(float((power(200.0 + h) - power(200.0 - h)) / (2 * h)), rel=1e-5)


# ---------------------------------------------------------------------- tensors, by what they spell

@pytest.mark.parametrize('rotated,diagonal', [
    (lambda: Material([4.0, 9.0, 2.25]).rotated(np.pi / 2, 'z'), [9.0, 4.0, 2.25]),
    (lambda: Material([4.0, 4.0, 2.25]).rotated(0.7, 'z'), [4.0, 4.0, 2.25]),
], ids=['quarter_turn', 'uniaxial'])
def test_the_rounding_a_rotation_leaves_is_not_an_off_diagonal(rotated, diagonal):
    z = np.array([33.0])
    a, b = (np.asarray(_solved([dict(background=1.0, shapes=[_rect(material=m)])], [120.0]).reconstruct_field(
        X, Y, z, reconstruction='lalanne')) for m in (rotated(), Material(diagonal)))
    assert np.abs(a - b).max() < 1e-11 * np.abs(b).max()
    grid = np.broadcast_to(np.eye(3, dtype=complex), (1, 9, 8, 3, 3)).copy()
    grid[0, 2:6, 1:5] = np.asarray(rotated().epsilon)
    s = _solved(grid, [120.0], fourier_coefficients='integrated')
    assert np.isfinite(np.asarray(s.reconstruct_field(X, Y, z, reconstruction='lalanne'))).all()


def test_a_uniform_off_diagonal_raster_keeps_the_sum_beside_a_patterned_layer():
    grid = np.broadcast_to(np.eye(3, dtype=complex), (2, 9, 8, 3, 3)).copy()
    grid[0] = OFF
    grid[1, 2:6, 1:5] = np.diag([4.0, 9.0, 2.25])
    s = _solved(grid, [120.0, 100.0], fourier_coefficients='integrated')
    below, above = np.array([33.0]), np.array([160.0])
    np.testing.assert_array_equal(s.reconstruct_field(X, Y, below, reconstruction='lalanne'),
                                  s.reconstruct_field(X, Y, below))
    assert np.abs(np.asarray(s.reconstruct_field(X, Y, above, reconstruction='lalanne'))
                  - np.asarray(s.reconstruct_field(X, Y, above))).max() > 1e-3


def test_a_tensor_uniform_across_a_layer_answers_alike_on_shapes_and_on_a_raster():
    """A patterned ``eps`` under a uniform off-diagonal ``mu``, and the reverse: the other field has the walls."""
    px, py = PX / 8, PY / 9
    wall = dict(cx=3 * px - PX / 2, cy=4 * py - PY / 2, lx=4 * px, ly=4 * py)
    z = np.array([33.0])
    shapes = [dict(background=Material(1.0, mu=OFF), shapes=[_rect(material=Material(4.0, mu=OFF), **wall)])]
    eps = _raster([None])
    mu = np.broadcast_to(OFF, (1, 9, 8, 3, 3)).copy()
    a = _solved(shapes, [120.0])
    b = _solved(eps, [120.0], mu=mu, fourier_coefficients='integrated')
    one, two = (np.asarray(s.reconstruct_field(X, Y, z, reconstruction='lalanne')) for s in (a, b))
    assert np.abs(one - two).max() < 1e-11 * np.abs(two).max()
    # `mu` has no walls here, so `H` is the sum, to the rounding of a division by a constant.
    assert np.abs(one[..., 3:] - np.asarray(a.reconstruct_field(X, Y, z))[..., 3:]).max() < 1e-13
    both = _solved([dict(background=Material(OFF), shapes=[_rect(material=Material(OFF))])], [120.0])
    assert np.isfinite(np.asarray(both.reconstruct_field(X, Y, z, reconstruction='lalanne'))).all()


def test_a_differentiated_whole_tensor_is_refused_for_what_it_hides_and_names_the_way_out():
    torch = pytest.importorskip('torch')
    e = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    tensor = torch.diag(torch.stack([e, e + 5.0, e - 1.0])).to(torch.complex128)
    s = _solved([dict(background=1.0, shapes=[_rect(material=Material(tensor))])], [120.0], backend='torch')
    with pytest.raises(NotImplementedError, match='principal values'):
        s.reconstruct_field(X, Y, np.array([33.0]), reconstruction='lalanne')


@pytest.mark.parametrize('kind', ['raster', 'material'])
def test_a_concrete_diagonal_tensor_read_inside_a_jit_is_read_by_value(kind):
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    structure = (_raster_tensor() if kind == 'raster' else
                 [dict(background=1.0, shapes=[_rect(material=Material(np.diag([4.0, 9.0, 2.25]).astype(complex)))])])
    extra = dict(fourier_coefficients='integrated') if kind == 'raster' else {}
    state = _solved(structure, [120.0], backend='jax', **extra).solve()._field_state
    z = np.array([33.0])
    eager = np.asarray(state.reconstruct(X, Y, z, 'incident', 'lalanne'))
    jitted = np.asarray(jax.jit(lambda: state.reconstruct(X, Y, z, 'incident', 'lalanne'))())
    assert np.abs(eager - jitted).max() < 1e-12 * np.abs(eager).max()


# ------------------------------------------------------------------------- what a sample holds

def _singular_layer():
    """A row of the middle layer whose ``1/eps`` has a zero mean, at y = -120."""
    grid = np.ones((3, 8, 8), dtype=complex)
    grid[0] = 1.69
    grid[1] = 1.5
    grid[1, 1:4, :4] = -1.0
    grid[1, 1:4, 4:] = 1.0
    grid[2] = 1.3
    return grid


def test_a_singular_line_in_another_layer_fails_nothing():
    s = _solved(_singular_layer(), [100.0, 120.0, 90.0], factorization='laurent', truncation_order=[2, 2])
    y = np.array([-120.0])
    with pytest.raises(np.linalg.LinAlgError):
        s.reconstruct_field(X, y, np.array([160.0]), reconstruction='lalanne')
    for z in (50.0, 270.0):
        np.testing.assert_array_equal(s.reconstruct_field(X, y, np.array([z]), reconstruction='lalanne'),
                                      s.reconstruct_field(X, y, np.array([z])))
    state = s.solve()._field_state
    np.testing.assert_array_equal(state.reconstruct(X, y, np.array([50.0]), 'incident', 'lalanne'),
                                  state.reconstruct(X, y, np.array([50.0]), 'incident'))


def test_a_singular_line_in_another_layer_leaves_the_gradient_alone():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    y = np.array([-120.0])

    def power(d):
        s = _solver(_singular_layer(), jax.numpy.array([100.0 + d, 120.0, 90.0]), backend='jax',
                    factorization='laurent', truncation_order=[1, 1], theta=0.3)
        state = s.solve()._field_state
        return jax.numpy.sum(jax.numpy.abs(state.reconstruct(X, y, np.array([250.0]), 'incident', 'lalanne')) ** 2)
    grad = float(jax.grad(power)(0.0))
    h = 1e-3
    assert grad == pytest.approx(float((power(h) - power(-h)) / (2 * h)), rel=1e-4)


@pytest.mark.parametrize('sx,sy', [(-1, 1), (1, -1), (-1, -1)])
def test_a_negative_side_is_refused_at_the_solvers_door_too(sx, sy):
    """The solver's own door gives the same refusal as the public one, before reading a line."""
    columns = np.array([[sx * PX, 0.0], [0.0, sy * PY]])
    s = _solved(_raster([None]), [120.0], lattice=columns, truncation_order=[1, 1], factorization='laurent')
    z = np.array([33.0])
    assert np.all(np.isfinite(np.asarray(s.reconstruct_field(X, Y, z))))
    with pytest.raises(NotImplementedError, match='positive sides'):
        s.reconstruct_field(X, Y, z, reconstruction='lalanne')


def test_a_shape_layer_no_sample_is_in_reads_no_line(monkeypatch):
    from wavebit.rcwa._engine import lalanne
    calls = []
    read = lalanne.read_lines
    monkeypatch.setattr(lalanne, 'read_lines', lambda *a, **k: calls.append(a[1]) or read(*a, **k))
    layers = [dict(background=1.0, shapes=[_rect()]), dict(background=2.25, shapes=[_rect(material=9.0)])]
    s = _solved(layers, [120.0, 60.0])
    state = s.solve()._field_state
    for z, read_of in ((33.0, [layers[0]]), (150.0, [layers[1]]), (np.array([33.0, 150.0]), layers)):
        z = np.atleast_1d(z)
        for ask in (lambda: s.reconstruct_field(X, Y, z, reconstruction='lalanne'),
                    lambda: state.reconstruct(X, Y, z, 'incident', 'lalanne')):
            calls.clear()
            ask()
            assert [line['background'] for line in calls] == [
                layer['background'] for layer in read_of for _ in 'xy']


def test_a_sample_rounded_across_an_interface_by_the_mirror_still_has_its_row(monkeypatch):
    """total - z lands past the mirror's own sum of the same layers, in the layer below the one z is in."""
    from wavebit.rcwa._engine import engine, lalanne
    thickness = [60.0, 67.4, 130.3]
    assert sum(thickness) - thickness[0] > thickness[2] + thickness[1]
    grid = _raster([None, 2.25, None])
    grid[1, 3:7, 2:6] = 9.0
    s = _solved(grid, thickness, fourier_coefficients='integrated')
    z = np.array([thickness[0]])
    assert lalanne.layers_holding(z, thickness) == {0, 1}
    state = s.solve()._field_state
    found = (np.asarray(s.reconstruct_field(X, Y, z, reconstruction='lalanne')),
             np.asarray(state.reconstruct(X, Y, z, 'incident', 'lalanne')))
    monkeypatch.setattr(engine, 'layers_holding', lambda *_: None)
    monkeypatch.setattr(lalanne, 'layers_holding', lambda *_: None)
    every_row = (np.asarray(s.reconstruct_field(X, Y, z, reconstruction='lalanne')),
                 np.asarray(state.reconstruct(X, Y, z, 'incident', 'lalanne')))
    np.testing.assert_array_equal(found[0], every_row[0])
    np.testing.assert_array_equal(found[1], every_row[1])
    assert np.abs(found[0] - np.asarray(s.reconstruct_field(X, Y, z))).max() > 1e-3


def test_a_concrete_tensor_solved_inside_a_jit_is_still_read_by_value():
    """The solve's copy is what the refusals read: a tracer there would be refused for a derivative it has not."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    grid = jax.numpy.array(_raster_tensor())

    def field(height):
        s = _solver(grid, jax.numpy.array([height]), backend='jax', fourier_coefficients='integrated')
        s.solve()
        return s.reconstruct_field(4, 3, 2, reconstruction='lalanne')
    eager, jitted = np.asarray(field(120.0)), np.asarray(jax.jit(field)(120.0))
    assert np.abs(eager - jitted).max() < 1e-11 * np.abs(eager).max()


def test_a_source_reads_the_structure_its_solve_saw():
    from wavebit.rcwa._engine.sources import PlaneWave
    s = rcwa('numpy', top=1.0, bot=2.25, period=[PX, PY], truncation_order=[3, 3], wavelength=WL, thickness=[120.0],
             structure=_raster([None]), factorization='li', answers='field',
             excitation={'top': PlaneWave(theta=0.25, phi=0.6, polarization=(1.0, 0.0))})
    result = s.solve()
    z = np.array([33.0])
    before = np.asarray(result.reconstruct_field(X, Y, z, reconstruction='lalanne'))
    s.structure[...] = 9.0
    np.testing.assert_array_equal(result.reconstruct_field(X, Y, z, reconstruction='lalanne'), before)


@pytest.mark.parametrize('columns', [[[-PX, 100.0], [0.0, PY]], [[PX, 100.0], [0.0, -PY]]])
def test_a_negative_diagonal_on_a_skew_cell_walks_its_own_basis(columns):
    """A skewed cell samples along its basis from the corner that basis names, so its phase takes the signed column."""
    columns = np.array(columns)
    uniform = dict(bot=1.0, truncation_order=[1, 1], fourier_coefficients='integrated')
    skewed = _solved(np.ones((1, 9, 8), dtype=complex), [120.0], lattice=columns, factorization='nvf', **uniform)
    straight = _solved(np.ones((1, 9, 8), dtype=complex), [120.0], factorization='li', **uniform)
    z = np.array([33.0])
    field = np.asarray(skewed.reconstruct_field(3, 2, z))[0]
    for v, along_v in enumerate(np.linspace(-0.5, 0.5, 2)):
        for u, along_u in enumerate(np.linspace(-0.5, 0.5, 3)):
            x, y = columns @ np.array([along_u, along_v])
            plane = np.asarray(straight.reconstruct_field(np.array([x]), np.array([y]), z))[0]
            assert np.abs(field[0, v, u] - plane[0, 0, 0]).max() < 1e-11
