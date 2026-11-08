"""``torch.func`` transforms of a public Torch FDTD run.

``torch.func.grad``, ``jacrev`` and ``vjp`` of ``Prepared.run`` raised PyTorch's raw "must
override the setup_context staticmethod" ``RuntimeError``: the in-place slab Functions and the
checkpoint ``Segment`` were written in the old ``forward(ctx, ...)`` style. Once they ran, a
checkpointed run's ``vjp`` read 0 with no error. ``torch.func.vmap`` failed with a raw in-place
``RuntimeError``. Each derivative is checked against eager ``torch.autograd.grad`` on the same
scalar (to rounding) and against a central difference of NumPy runs.
"""
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.errors import UnsupportedFeatureError
from wavebit.units import C0

torch = pytest.importorskip('torch')

CENTER = (-.073, .017, -.009)
RADIUS, EPSILON = .22, 2.25


def problem(radius=RADIUS, epsilon=EPSILON, open_z=False):
    body = fdtd.Structure('ball', shape=fdtd.Sphere(radius=radius, center=CENTER),
                          material=Material(epsilon=epsilon, mu=1.2))
    z = fdtd.Open() if open_z else fdtd.PEC()
    return fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(1.6,)*3),
                         boundaries=fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=z),
                         structures=(body,)),
        sources=(fdtd.CurrentDipole('d', center=(-.55, .03, -.02), current_moment=1e-16,
                                    pulse=fdtd.GaussianPulse(wavelength=3., frequency_sigma_hz=.03*C0/1e-6),
                                    delay_seconds=-18e-6/C0),),
        outputs=fdtd.Outputs(final_fields=True))


def numerics(**changes):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.1), duration=fdtd.Duration(steps=12),
                         geometry_ranges={'ball': ((-.36, .36),)*3}).replace(**changes)


def quiet(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args)


def objectives(backend, field, open_z=False, **changes):
    """``(scalar, vector)`` objectives of ``field`` on one prepared run of ``backend``."""
    prepared = quiet(lambda: fdtd.prepare(problem(open_z=open_z), numerics=numerics(**changes),
                                          backend=backend))

    def fields(value):
        E = quiet(prepared.run, problem(open_z=open_z, **{field: value})).final_fields.values['E']
        return E

    def scalar(value):
        return (fields(value)**2).sum()

    def vector(value):
        E = fields(value)
        return (E**2).reshape(3, -1).sum(-1)       # one energy-like sum per component
    return scalar, vector


def central(field, h):
    scalar, _ = objectives('numpy', field)
    x = {'radius': RADIUS, 'epsilon': EPSILON}[field]
    return (float(scalar(x + h)) - float(scalar(x - h)))/(2*h)


def t(value):
    return torch.tensor(value, dtype=torch.float64)


def eager(fn, x):
    leaf = t(x).requires_grad_()
    out = fn(leaf)
    if out.ndim == 0:
        return torch.autograd.grad(out, leaf)[0]
    return torch.stack([torch.autograd.grad(o, leaf, retain_graph=True)[0] for o in out])


@pytest.mark.parametrize('field,x,h', [('radius', RADIUS, 1e-6), ('epsilon', EPSILON, 1e-5)])
def test_torch_func_grad_jacrev_and_vjp_match_autograd_and_numpy_central_difference(field, x, h):
    scalar, vector = objectives('torch', field)
    reference = eager(scalar, x)
    expected = central(field, h)
    assert abs(expected) > 1e-6
    _, pullback = torch.func.vjp(scalar, t(x))
    got = {'grad': torch.func.grad(scalar)(t(x)),
           'jacrev': torch.func.jacrev(scalar)(t(x)),
           'vjp': pullback(t(1.))[0]}
    for name, value in got.items():
        assert float(value) == pytest.approx(float(reference), rel=1e-12, abs=0), name
        assert float(value) == pytest.approx(expected, rel=2e-6), name
    rows = torch.func.jacrev(vector)(t(x))
    torch.testing.assert_close(rows, eager(vector, x), rtol=1e-12, atol=0)


def test_torch_func_grad_through_the_cpml_matches_autograd():
    scalar, _ = objectives('torch', 'radius', open_z=True, absorber=fdtd.PML(cells=3, allow_thin=True))
    reference = eager(scalar, RADIUS)
    assert float(torch.func.grad(scalar)(t(RADIUS))) == pytest.approx(float(reference), rel=1e-12)
    assert float(torch.func.jacrev(scalar)(t(RADIUS))) == pytest.approx(float(reference), rel=1e-12)


def test_torch_func_grad_and_vjp_through_checkpoint_segments_match_autograd():
    # 12 steps in blocks of 2 nest segments inside segments.
    scalar, _ = objectives('torch', 'epsilon', checkpoint_steps=2)
    reference = eager(scalar, EPSILON)
    _, pullback = torch.func.vjp(scalar, t(EPSILON))
    for value in (torch.func.grad(scalar)(t(EPSILON)), pullback(t(1.))[0]):
        assert float(value) == pytest.approx(float(reference), rel=1e-12, abs=0)
    with pytest.raises(UnsupportedFeatureError) as caught:
        torch.func.jacrev(scalar)(t(EPSILON))
    assert caught.value.code == 'fdtd.torch_transform'
    assert caught.value.path == 'numerics.checkpoint_steps'


@pytest.mark.parametrize('field,x', [('radius', RADIUS), ('epsilon', EPSILON)])
@pytest.mark.parametrize('transform', ['vmap', 'jacfwd'])
def test_torch_func_vmap_of_a_run_is_a_structured_refusal(field, x, transform):
    scalar, _ = objectives('torch', field)
    call = {'vmap': lambda: torch.func.vmap(scalar)(torch.stack([t(x), t(x*1.01)])),
            'jacfwd': lambda: torch.func.jacfwd(scalar)(t(x))}[transform]
    with pytest.raises(UnsupportedFeatureError) as caught:
        call()
    assert caught.value.code == 'fdtd.torch_transform' and caught.value.path == 'problem'
    # jvp is forward mode without vmap and is not refused.
    tangent = torch.func.jvp(scalar, (t(x),), (t(1.),))[1]
    assert float(tangent) == pytest.approx(float(eager(scalar, x)), rel=1e-12)
