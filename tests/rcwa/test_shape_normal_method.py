"""``shape_normal_method``: which normal field a layer of shapes builds under NVF.

``'per_shape'`` (the default) builds a layer of disjoint rectangles, circles and ellipses in
closed form, shape by shape, and a layer the painting resolves from the interfaces it leaves.
The two fields differ, so one structure written as tiles and as a painting reads two ways, and
a layer steps where its shapes cross the routing tolerance (``OVERLAP_TOL_REL``).
``'interfaces'`` (experimental) builds every shape layer the second way.

What is pinned here: the option is honoured where it is read and refused where it is not; the
default is unchanged bit for bit; under ``'interfaces'`` the two descriptions agree to
round-off and the routing step is gone; a frozen record says which method it was taken under
and refuses the other; the option survives a job and a JAX trace.
"""
import json

import numpy as np
import pytest

from wavebit import Job, Material, Units, rcwa
from wavebit.errors import UnsupportedFeatureError
from wavebit.rcwa._engine.geometry import StaleGeometryError, StaticGeometry
from wavebit.rcwa._engine.parametric import OVERLAP_TOL_REL
from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

#: Every test but the one about it reads past the warning `'interfaces'` emits.
pytestmark = pytest.mark.filterwarnings('ignore::wavebit.rcwa.diagnostics.ExperimentalWarning')

ORDERS = (3, 3)
INTERFACES = rcwa.Numerics(orders=ORDERS, shape_normal_method='interfaces')


def region(geometry, epsilon):
    return rcwa.Region(geometry=geometry, material=Material(epsilon))


def problem(*regions):
    layer = rcwa.Layer(thickness=.4, background=Material(1.), regions=regions)
    return rcwa.Problem(units=Units(length='um'),
                        stack=rcwa.Stack(layers=(layer,), bottom=Material(2.25), period=(1., 1.)),
                        excitation=rcwa.Excitation(wavelength=1.3, sources=(
                            rcwa.PlaneWave(theta=.35, phi=.6, polarization='s'),)))


def pair(overlap):
    """Two 0.3 x 0.4 rectangles side by side, the right one painted over the left by ``overlap``."""
    return problem(region(rcwa.Rectangle(size=(.3, .4), center=(-.15, 0.)), 4.),
                   region(rcwa.Rectangle(size=(.3, .4), center=(.15 - overlap, 0.)), 9.))


#: One structure, two descriptions: a 0.3 x 0.4 block of 4 beside one of 9, written as two
#: touching tiles (disjoint: the closed form under the default) and as a 0.6 x 0.4 block of 4
#: with the right half painted 9 (overlapping: the boundary route under both methods).
TILES = pair(0.)
PAINTING = problem(region(rcwa.Rectangle(size=(.6, .4)), 4.),
                   region(rcwa.Rectangle(size=(.3, .4), center=(.15, 0.)), 9.))


def answers(p, numerics):
    result = rcwa.run(p, numerics=numerics)
    return np.asarray(result.R)[0], np.asarray(result.T)[0]


def gap(a, b):
    return max(np.abs(x - y).max() for x, y in zip(a, b))


def test_under_interfaces_tiles_and_a_painting_of_one_structure_agree():
    """The description-invariance the default lacks: measured 2.9e-15 under ``'interfaces'``
    and 1.9e-4 in ``T`` under the default on this pair (2026-11-18)."""
    default = rcwa.Numerics(orders=ORDERS)
    assert gap(answers(TILES, default), answers(PAINTING, default)) > 1e-5
    tiles, painting = answers(TILES, INTERFACES), answers(PAINTING, INTERFACES)
    assert gap(tiles, painting) < 1e-12
    # and the painting itself was always built on the interfaces: the switch does not move it
    assert gap(painting, answers(PAINTING, default)) < 1e-12
    assert abs(np.sum(tiles[0]) + np.sum(tiles[1]) - 1.) < 1e-12


def test_under_interfaces_the_routing_tolerance_leaves_no_step():
    """Across ``OVERLAP_TOL_REL`` the default changes builders and steps (5.4e-4 measured on
    the 2026-11-17 audit's pair); under ``'interfaces'`` both sides take the boundary route and
    the change is the overlap's own, of order the 2e-7 moved."""
    below, above = pair(.9 * OVERLAP_TOL_REL), pair(1.1 * OVERLAP_TOL_REL)
    default = rcwa.Numerics(orders=ORDERS)
    assert gap(answers(below, default), answers(above, default)) > 1e-5
    assert gap(answers(below, INTERFACES), answers(above, INTERFACES)) < 1e-7


@pytest.mark.parametrize('p', [TILES, problem(region(rcwa.Circle(radius=.25), 6.25))],
                         ids=['tiles', 'circle'])
def test_an_explicit_per_shape_is_the_default_bit_for_bit(p):
    """``'per_shape'`` names what the path has always done; naming it changes nothing."""
    plain = answers(p, rcwa.Numerics(orders=(2, 2)))
    named = answers(p, rcwa.Numerics(orders=(2, 2), shape_normal_method='per_shape'))
    assert all(np.array_equal(a, b) for a, b in zip(plain, named))


def test_interfaces_warns_that_it_is_experimental_and_the_default_does_not():
    import warnings

    from wavebit.rcwa.diagnostics import ExperimentalWarning
    small = lambda **kw: rcwa.Numerics(orders=(1, 1), **kw)
    with pytest.warns(ExperimentalWarning, match="shape_normal_method='interfaces' is experimental"):
        rcwa.run(TILES, numerics=small(shape_normal_method='interfaces'))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ExperimentalWarning)
        rcwa.run(TILES, numerics=small())
        rcwa.run(TILES, numerics=small(shape_normal_method='per_shape'))
        # a refused scene raises without a warning
        with pytest.raises(UnsupportedFeatureError):
            rcwa.run(TILES, numerics=small(shape_normal_method='interfaces',
                                           factorization='laurent'))


def test_the_report_resolves_the_method_where_it_is_read():
    resolved = lambda **kw: rcwa.check(TILES, numerics=rcwa.Numerics(orders=(1, 1), **kw)).resolved
    assert resolved()['shape_normal_method'] == 'per_shape'
    assert resolved(shape_normal_method='interfaces')['shape_normal_method'] == 'interfaces'
    assert resolved(factorization='laurent')['shape_normal_method'] is None
    assert resolved(nvf_projector='circular')['shape_normal_method'] is None


@pytest.mark.parametrize('method', ['per_shape', 'interfaces'])
@pytest.mark.parametrize('numerics, where', [
    (dict(factorization='laurent'), "factorization='laurent' builds no normal field"),
    (dict(factorization='li'), "factorization='li' builds no normal field"),
    (dict(nvf_projector='circular'), "nvf_projector='circular' builds its own"),
], ids=['laurent', 'li', 'circular'])
def test_an_explicit_method_without_a_reader_is_refused(method, numerics, where):
    """Even the default's own name: an explicit value is a request, and these builds would run
    to completion with it ignored."""
    with pytest.raises(UnsupportedFeatureError) as caught:
        rcwa.run(TILES, numerics=rcwa.Numerics(orders=(1, 1), shape_normal_method=method,
                                               **numerics))
    assert caught.value.code == 'rcwa.unsupported'
    assert f'shape_normal_method={method!r} has no reader here' in caught.value.message
    assert where in caught.value.message


def test_tensor_materials_refuse_it():
    tensor = Material(np.diag([2.1, 2.5, 3.]).astype(complex))
    p = problem(region(rcwa.Rectangle(size=(.3, .4)), 4.),
                rcwa.Region(geometry=rcwa.Rectangle(size=(.2, .2), center=(.3, 0.)),
                            material=tensor))
    with pytest.raises(UnsupportedFeatureError) as caught:
        rcwa.run(p, numerics=rcwa.Numerics(orders=(1, 1), shape_normal_method='interfaces'))
    assert caught.value.code == 'rcwa.unsupported'
    assert 'tensor materials are built by their own shape builders' in caught.value.message


def test_a_raster_refuses_it():
    solver = _legacy_wavebit_rcwa(structure=np.ones((1, 8, 8)), period=[1., 1.], thickness=[.4],
                                  wavelength=1.3, truncation_order=[1, 1], top=1., bot=1.,
                                  factorization='nvf', shape_normal_method='interfaces')
    with pytest.raises(NotImplementedError, match='a raster builds its normal field from samples'):
        solver.solve()


#: The legacy door, for the frozen record: two disjoint rectangles, in micrometres.
_LAYERS = [{'background': 1.0, 'shapes': [
    {'type': 'rectangle', 'cx': -.15, 'cy': 0., 'lx': .3, 'ly': .4, 'material': 4.},
    {'type': 'rectangle', 'cx': .2, 'cy': 0., 'lx': .2, 'ly': .4, 'material': 9.}]}]
_KW = dict(structure=_LAYERS, period=[1., 1.], thickness=[.4], wavelength=1.3,
           truncation_order=[2, 2], top=1., bot=2.25, theta=.35, phi=.6, psi=0.)


def _T(solver, **kw):
    return np.asarray(solver.solve(**kw).T)


def test_a_record_says_which_method_it_was_taken_under_and_refuses_the_other():
    """A ``None`` plan reads as "sum this layer", so the method decides which layers carry one,
    and a record replayed under the other method would build the other field without a word."""
    default = _legacy_wavebit_rcwa(**_KW)
    record = default.freeze_geometry()
    assert record.shape_normal_method == 'per_shape' and record.boundary_plans == [None]

    interfaces = _legacy_wavebit_rcwa(shape_normal_method='interfaces', **_KW)
    taken = interfaces.freeze_geometry()
    assert taken.shape_normal_method == 'interfaces' and taken.boundary_plans[0] is not None
    assert taken != record                    # a jitted call compiles each separately
    np.testing.assert_allclose(_T(interfaces, geometry=taken), _T(interfaces), atol=1e-14, rtol=0)

    for solver, other, held in ((interfaces, record, 'per_shape'),
                                (default, taken, 'interfaces')):
        with pytest.raises(StaleGeometryError, match=f"taken under shape_normal_method='{held}'"):
            solver.solve(geometry=other)


def test_a_hand_built_record_with_no_plan_cannot_serve_interfaces():
    """A record built by hand carries no stamp and is replayed on trust -- but a ``None`` plan
    on a shape layer is an answer the method cannot take, so the builder refuses it itself."""
    solver = _legacy_wavebit_rcwa(shape_normal_method='interfaces', **_KW)
    taken = solver.freeze_geometry()
    bare = StaticGeometry(boundary_plans=[None], gamma_plans=taken.gamma_plans)
    with pytest.raises(StaleGeometryError, match='records layer 0 as summed'):
        solver.solve(geometry=bare)


def test_the_method_survives_a_job_and_a_default_job_keeps_its_key():
    numerics = rcwa.Numerics(orders=(1, 1))
    default = Job(solver='rcwa', problem=TILES, numerics=numerics)
    assert 'shape_normal_method' not in json.loads(default.to_json())['numerics']['fields']
    job = default.replace(numerics=numerics.replace(shape_normal_method='interfaces'))
    restored = Job.from_json(job.to_json())
    assert restored.numerics.shape_normal_method == 'interfaces'
    assert restored.key == job.key != default.key


def test_a_disjoint_layer_on_the_interfaces_differentiates_through_a_frozen_record():
    """The boundary route carries a derivative to the vertices, so a layer it builds only
    because the method asked keeps a shape gradient, eager and jitted."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit._common.backend import _enable_x64
    _enable_x64()

    def make(width):
        return [{'background': 1.0, 'shapes': [
            {'type': 'rectangle', 'cx': -.15, 'cy': 0., 'lx': width, 'ly': .4, 'material': 4.},
            {'type': 'rectangle', 'cx': .2, 'cy': 0., 'lx': .2, 'ly': .4, 'material': 9.}]}]

    kw = {k: v for k, v in _KW.items() if k != 'structure'}
    x = jnp.asarray(.3)
    frozen = _legacy_wavebit_rcwa(backend='jax', structure=make(x), shape_normal_method='interfaces',
                                  **kw).freeze_geometry()

    def total(v):
        solver = _legacy_wavebit_rcwa(backend='jax', structure=make(v),
                                      shape_normal_method='interfaces', **kw)
        return jnp.sum(solver.solve(geometry=frozen).T).real

    eager = float(total(x))
    assert float(jax.jit(total)(x)) == pytest.approx(eager, rel=1e-11)
    grad = float(jax.grad(total)(x))
    h = 1e-5
    fd = (float(total(x + h)) - float(total(x - h))) / (2 * h)
    assert grad == pytest.approx(fd, abs=1e-6), (grad, fd)
