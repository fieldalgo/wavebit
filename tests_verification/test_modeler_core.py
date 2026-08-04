"""Vector geometry modeler (fieldwave._core.modeler) against the frozen legacy output.

The modeler decomposes parametric shapes — rectangles and ellipses, optionally rotated — into
the piecewise-constant blocks the vector factorization consumes. Its reference values were
captured from the legacy per-backend implementation before that was deleted
(gen_goldens_modeler.py -> golden_legacy_modeler.npz).
"""
import pathlib

import numpy as np
import pytest

import fieldwave
from fieldwave._core.modeler import Modeling

GOLDEN = np.load(pathlib.Path(__file__).parent / 'golden_legacy_modeler.npz')

PERIOD = [300.0, 300.0]
SI, SIO2, SI3N4 = 3.638751670074983, 1.4518, 2.0056

# Must stay identical to gen_goldens_modeler.py.
INSTRUCTIONS = {
    'rect_1d': [[SI3N4, [['rectangle', 50, 150, 31, 300, SI, 0, 0, 0]]], [SI, []]],
    'rect_2d': [[SI3N4, [['rectangle', 80, 120, 60, 90, SI, 0, 0, 0],
                         ['rectangle', 200, 180, 50, 70, SIO2, 0, 0, 0]]], [SI, []]],
    'rect_rot': [[SI3N4, [['rectangle', 150, 150, 80, 60, SI, 0.4, 4, 4]]], [SIO2, []]],
    'ellipse': [[SI3N4, [['ellipse', 150, 150, 90, 60, SI, 0.0, 4, 4]]], [SIO2, []]],
    'ellipse_rot': [[SI3N4, [['ellipse', 140, 160, 100, 50, SI, 0.5, 4, 4]]], [SIO2, []]],
}
SOLVE_KW = dict(period=PERIOD, thickness=[205.0, 100.0], fto=[3, 3],
                n_top=1.0, n_bot=1.0, wavelength=900.0, pol=0.5)


@pytest.mark.parametrize('name', list(INSTRUCTIONS))
def test_vector_blocks_match_legacy_golden(name):
    """The decomposed blocks and their edges, layer by layer."""
    info = Modeling(period=PERIOD).modeling_vector_instruction(INSTRUCTIONS[name])
    assert len(info) == int(GOLDEN[f'{name}__n_layers'])
    for i, (cell, x, y) in enumerate(info):
        for tag, got in (('cell', cell), ('x', x), ('y', y)):
            want = GOLDEN[f'{name}__L{i}__{tag}']
            got = np.asarray(got)
            assert got.shape == want.shape, (i, tag, want.shape, got.shape)
            assert np.array_equal(want, got), (i, tag, np.max(np.abs(want - got)))


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('name', list(INSTRUCTIONS))
def test_vector_solve_matches_legacy_golden(name, backend):
    """End to end: modeled geometry -> factorization -> efficiencies, on every backend."""
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    result = fieldwave.rcwa(backend=backend, ucell=INSTRUCTIONS[name], **SOLVE_KW).conv_solve()
    for field in ('de_ri', 'de_ti'):
        want = GOLDEN[f'{name}__{field}']
        got = np.asarray(getattr(result.res, field))
        tol = 1e-12   # not bit-exact: the vector path's inverses do not commute
                      # with the 2026-08 order-index flip to the last digit
        assert np.allclose(want, got, atol=tol, rtol=0), (field, np.max(np.abs(want - got)))


def test_rectangle_no_approximation_and_draw():
    """The exact-rectangle helper and ``draw`` (used directly, not via instructions)."""
    modeler = Modeling(period=[1000.0, 1000.0])
    element = modeler.rectangle_no_approximation(cx=500, cy=500, lx=200, ly=200, base=2.0)
    for i, (cell, x, y) in enumerate(modeler.draw([[1.0, element]])):
        for tag, got in (('cell', cell), ('x', x), ('y', y)):
            want = GOLDEN[f'draw__L{i}__{tag}']
            assert np.array_equal(want, np.asarray(got)), (i, tag)


def test_torch_device_code_is_not_a_device_index():
    """``device=0`` is fieldwave's code for CPU, not torch device index 0.

    Passing it through raw selects MPS on a Mac, which rejects float64 outright.
    """
    torch = pytest.importorskip('torch')
    modeler = Modeling(period=PERIOD, type_complex=torch.complex128, device=0,
                       xp=__import__('array_api_compat').array_namespace(torch.zeros(1)))
    info = modeler.modeling_vector_instruction(INSTRUCTIONS['rect_rot'])
    assert np.array_equal(GOLDEN['rect_rot__L0__cell'], np.asarray(info[0][0]))


def test_jax_device_setter_round_trips():
    """``solver.device = solver.device`` must be accepted on the jax backend.

    The getter returns resolved ``jax.Device`` objects, and several paths re-assign the
    property, so rejecting them breaks ``conv_solve_field``. The old check compared the
    repr of the implementation class, which stopped matching when jaxlib moved ``Device``
    from ``xla_extension`` to ``_jax``.
    """
    jax = pytest.importorskip('jax')

    solver = fieldwave.rcwa(
        backend=1, pol=0, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, fto=[5],
        wavelength=900.0, period=[1000.0], thickness=[500.0],
        ucell=np.array([[[1.0, 3.0, 3.0, 1.0]]]), device=0)

    solver.device = solver.device                      # the round-trip that used to raise
    solver.device = jax.devices('cpu')[0]              # a bare device, not a list
    assert all(isinstance(d, jax.Device) for d in solver.device)

    with pytest.raises(ValueError):
        solver.device = 'not-a-device'

    # The caller-visible symptom was here: @jax_device_set re-assigns the property, so the
    # rejected round-trip surfaced as a bare ValueError out of conv_solve_field. The jitted
    # variant is still blocked by a separate same-material/tracer problem, so pin the
    # no-jit one, which goes through the same decorator.
    solver.conv_solve_field_no_jit(res_x=10, res_y=1, res_z=10)
