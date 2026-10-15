"""``torch.autograd`` through ``torch.func.vmap``: the eager derivative, lane by lane.

A batch of wavelengths, angles or designs under ``torch.func.vmap``, differentiated by
``backward()`` (and ``torch.autograd.grad``) or by ``torch.autograd.forward_ad``, is checked
against eager autograd run one lane at a time (to 1e-11 relative) and, at the first lane, against
``test_forward_mode``'s central difference on NumPy, extrapolated in the step (to 1e-6) -- on the
1D TE and TM paths, the conical path on both layer algorithms, the 2D Li and Laurent
factorizations, the z-coupled (``'first_order'``) formulation, analytic shapes, the scattering
matrix, an ``excitation``, and disks of a lossy dielectric and of gold, where the NVF inverse-rule
floor binds. Every batch has two distinct lanes, ``p`` and ``p + 100 h``.

Until 2026-10-11 every one of them was silently wrong. ``vmap`` hands the solve ``BatchedTensor``
wrappers, and a wrapper reads ``requires_grad`` False even when the tensor it batches requires
grad, so each dispatch that asked it (``backend.carries_autodiff``) sent its layer down the
constant branch: the modal rule never ran, the eigen-derivative was dropped, and ``backward()``
returned a number with no error. Measured at 806661e4 against eager autograd: ``dR/d(wavelength)``
of a conical grating -0.003033 where eager and a central difference read -0.004225, the 2D Li
raster's pixel derivative 4.1 relative off, the scattering matrix's 18, the gold disks' 0.27; a
census of 34 scenes under ten transforms found 60 of 340 cells wrong. ``create_graph=True``
through ``vmap`` returned a partial Hessian where eager refuses it, and ``forward_ad`` through
``vmap`` raised PyTorch's "Batching rule not implemented for aten::_unpack_dual".

What else is pinned here, each against the change that would break it:

* a parameter captured with ``requires_grad`` beside a batched wavelength -- a spectral sweep of
  one design -- gets the sum of the lanes' eager gradients, the NVF material raster included;
* a parameter that moves no modal operator, a thickness, stays exact under every transform,
  ``torch.func`` and second orders included, while the operator carries no derivative; with a
  captured ``requires_grad`` permittivity in the operator it stays exact under each single
  ``torch.func`` transform, and a later ``backward()`` to that permittivity is the eager mixed
  derivative: the door counts a transform's own level only where it is live there;
* every ``torch.func`` grad or jvp level through a moving operator is refused by name -- nested
  levels by their type, the mixed partials included that read a tangent beneath another level
  and returned a number without the eigen-derivative -- and every second order through the
  first-order rules is refused, through ``vmap`` as well;
* the NVF floor on lanes that disagree about binding: the lane where it does not bind is handed
  back bit for bit;
* the two ``vmap`` rules alone, in every batching of their inputs, where no solve reaches;
* the geometry readers' refusals under ``vmap``;
* the private ``torch._C._functorch`` calls the detection rests on, by name and by meaning, and
  that eager PyTorch never calls them while the transforms-active probe exists (the probe
  itself is read on every call);
* a ``vmap`` with no derivative runs no first-order rule; PyTorch's own vectorized Jacobian and
  batched gradients stay right.
"""

from wavebit.compat import legacy as _legacy_wavebit
import functools

import numpy as np
import pytest

import wavebit
from tests.rcwa.test_forward_mode import GOLD, GRATING, GRATING_DIRECTION, JAX_ONLY, ONE_D, PATHS, PIXELS, RASTER, SQUARE, TWO_D, _close, _difference, _lift, _read_out, _weights, gold_disks, raster, shape

torch = pytest.importorskip('torch')
from torch.autograd import forward_ad  # noqa: E402

pytestmark = pytest.mark.filterwarnings('ignore')

func = torch.func


def _t(value):
    return torch.tensor(value, dtype=torch.float64)


def _lanes(at, h):
    return [at, at + 100 * h]


# ------------------------------------------------------------------------------- the scenes

PILLAR = np.ones((1, 1, 20))
PILLAR[0, 0, 6:14] = 12.25
REPORTED = dict(period=[1000.], truncation_order=[5, 0], thickness=[300., 200.], top=1.0,
                bot=2.25, theta=0.3, phi=np.pi / 2, psi=np.pi / 2)


def reported_conical(b, p):
    """The conical grating the defect was reported on: a pillar under a spacer, the wavelength the
    parameter. Its ``dR/d(wavelength)`` read -0.003033 through ``vmap`` and -0.004225 eager."""
    structure = _lift(b, np.concatenate([np.ones((1, 1, 20)), PILLAR]))
    solver = _legacy_wavebit.rcwa(backend=b, structure=structure, wavelength=520., **REPORTED)
    return _read_out(b, solver.solve(wavelength=p, auto_detune=False), 'r_s', 5)


#: name: (read-out, point, difference step). A name of ``test_forward_mode.PATHS`` is taken from
#: there, with its cached difference.
_OWN = {
    'conical, reported grating, wavelength': (reported_conical, 520.0, 1e-3),
    '2d Li, pixel eps': (raster('eps'), 0.0, 1e-4),
    'dielectric disks, floor idle, Re eps': (gold_disks('Re eps'), 4.0, 1e-4),
    # The width moves the rectangle's normal field, sampled 1024 x 1024 by default. Under `vmap`
    # with a backward pass that grid took one worker of the Linux CI runner (7.9 GB, two workers)
    # from 2.3 to 6.1 GB in five seconds and the runner was shut down, three runs out of three
    # (2026-10-14/15). Each test here compares a transform with eager on the same function, so
    # 256 per axis asks the same question: 0.55 GiB and 1.0 s instead of 2.49 GiB and 6.8 s
    # (macOS), its gradient within 1% of the 1024 one.
    'shape, rectangle lx, grid 256': (shape('lx', shape_nvf_grid_size=256), 0.385, 1e-5),
}
SCENES = {name: PATHS[name] for name in (
    '1d TE, eps, ETM', '1d TM, theta, SMM', 'conical, eps, ETM', 'conical, mu, SMM',
    '2d Li, theta', '2d Laurent, pixel eps', 'tensor first-order, theta', 'shape, disc wavelength',
    'excitation, eps', 'scattering matrix, pixel eps',
    'gold disks, floor binds, Re eps')}
SCENES.update(_OWN)
ALL = {**SCENES, **JAX_ONLY}


@functools.lru_cache(maxsize=None)
def _finite(name):
    """The NumPy read-out at the scene's point and its central difference, extrapolated in h."""
    if name in PATHS or name in JAX_ONLY:
        return _difference(name)
    f, at, h = ALL[name]
    near = (f('numpy', at + h) - f('numpy', at - h)) / (2 * h)
    far = (f('numpy', at + 2 * h) - f('numpy', at - 2 * h)) / (4 * h)
    return f('numpy', at), (4 * near - far) / 3


def _scalar(name):
    """The scene's weighted read-out, a scalar function of its parameter, on PyTorch."""
    f = ALL[name][0]
    w = torch.as_tensor(_weights(_finite(name)[0].size))
    return lambda p: (w * f('torch', p)).sum()


def _eager_gradient(g, p):
    x = _t(p).requires_grad_(True)
    g(x).backward()
    return float(x.grad)


def _eager_tangent(f, p):
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(_t(p), _t(1.0))
        return forward_ad.unpack_dual(f('torch', dual)).tangent.numpy()


# ------------------------------------------------------------------------------- first order

@pytest.mark.parametrize('name', list(SCENES))
def test_backward_through_vmap_is_the_eager_derivative(name):
    """``vmap(g)(x).sum().backward()`` against ``g(x_i).backward()`` lane by lane. At 806661e4
    each of these returned a gradient without the eigen-derivative, with no error."""
    _, at, h = SCENES[name]
    g = _scalar(name)
    lanes = _lanes(at, h)
    x = _t(lanes).requires_grad_(True)
    func.vmap(g)(x).sum().backward()
    assert _close(x.grad, [_eager_gradient(g, p) for p in lanes], 1e-11), x.grad


def test_vmap_of_vmap_is_the_eager_derivative():
    """Two nested batches, a 2 x 2 grid of permittivities, differentiated by one ``backward()``."""
    name = '1d TE, eps, ETM'
    g = _scalar(name)
    grid = [[0.0, 0.01], [0.005, 0.015]]
    x = _t(grid).requires_grad_(True)
    func.vmap(func.vmap(g))(x).sum().backward()
    assert _close(x.grad, [[_eager_gradient(g, p) for p in row] for row in grid], 1e-11), x.grad


@pytest.mark.parametrize('name', list(SCENES))
def test_forward_ad_through_vmap_is_the_eager_tangent(name):
    """``forward_ad`` through ``vmap``, the whole read-out's tangent, against eager ``forward_ad``
    lane by lane and against the central difference at the first lane. At 806661e4 it raised
    PyTorch's "Batching rule not implemented for aten::_unpack_dual": ``forward_tangent`` handed
    ``unpack_dual`` the ``BatchedTensor`` itself."""
    f, at, h = SCENES[name]
    lanes = _lanes(at, h)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(_t(lanes), torch.ones(2, dtype=torch.float64))
        tangent = forward_ad.unpack_dual(func.vmap(lambda p: f('torch', p))(dual)).tangent.numpy()
    eager = np.stack([_eager_tangent(f, p) for p in lanes])
    assert _close(tangent, eager, 1e-11), (tangent, eager)
    assert _close(tangent[0], _finite(name)[1], 1e-6), (tangent[0], _finite(name)[1])


@pytest.mark.parametrize('name', ['2d Li, theta', 'gold disks, floor binds, Re eps'])
def test_forward_ad_through_vmap_of_vmap_is_the_eager_tangent(name):
    """``forward_ad`` through two nested batches, a 2 x 2 grid, against eager ``forward_ad`` per
    lane: the dual sits under two ``BatchedTensor`` levels, and ``forward_tangent`` has to walk
    both. One that peeled a single level read no tangent on these 2D scenes and returned the
    constant branch's -- 0.76 relative off on the Li raster's angle, 0.33 on the gold disks --
    with no error (on the 1D TE path the same walk raised a raw ``TypeError`` instead)."""
    f, at, h = SCENES[name]
    grid = [[at, at + 100 * h], [at + 50 * h, at + 150 * h]]
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(_t(grid), torch.ones(2, 2, dtype=torch.float64))
        out = func.vmap(func.vmap(lambda p: f('torch', p)))(dual)
        tangent = forward_ad.unpack_dual(out).tangent.numpy()
    eager = np.stack([np.stack([_eager_tangent(f, p) for p in row]) for row in grid])
    assert _close(tangent, eager, 1e-11), (tangent, eager)


def _one_d_captured(b, q, wavelength):
    eps = _lift(b, GRATING) + q * _lift(b, GRATING_DIRECTION)
    cfg = dict(ONE_D, layer_algorithm='ETM', tetm='te', wavelength=wavelength)
    solver = _legacy_wavebit.rcwa(backend=b, structure=eps, **cfg)
    return _read_out(b, solver.solve(), 'r_s', 3, point=([57.], [0.], [83.]))


def _raster_captured(direction, **options):
    def f(b, q, wavelength):
        cfg = dict(TWO_D, wavelength=wavelength, **options)
        structure = _lift(b, RASTER) + q * _lift(b, direction)
        solver = _legacy_wavebit.rcwa(backend=b, structure=structure, **cfg)
        return _read_out(b, solver.solve(), 'r_s', 6, point=([0.1], [-0.05], [0.07]),
                         counted=(3, 2, [0.07]))
    return f


def _gold_captured(b, q, wavelength):
    piece = dict(type='circle', cx=0.0, cy=0.0, lx=140.0, material=q + 1j * GOLD.imag)
    solver = _legacy_wavebit.rcwa(backend=b, structure=[dict(background=2.25, shapes=[piece])],
                          top=2.25, bot=2.25, theta=0.3, phi=0.4, psi=0.7, wavelength=wavelength,
                          period=(450.0, 450.0), thickness=(50.0,), truncation_order=(4, 4))
    return _read_out(b, solver.solve(), 'r_s', 40)


#: name: (read-out of (parameter, wavelength), parameter, wavelengths).
CAPTURED = {
    '1d TE, eps': (_one_d_captured, 0.0, [900.0, 910.0]),
    '2d Li, pixel eps': (_raster_captured(PIXELS), 0.0, [0.777, 0.787]),
    # The NVF material raster's own door under vmap: batching the raster raises at its
    # data-dependent material test, so the material rides in captured.
    '2d NVF raster, material': (_raster_captured(SQUARE, factorization='nvf'), 0.0, [0.777, 0.787]),
    'gold disks, floor binds, Re eps': (_gold_captured, GOLD.real, [852.0, 862.0]),
}


@pytest.mark.parametrize('name', list(CAPTURED))
def test_a_captured_parameter_under_a_wavelength_vmap(name):
    """One design, a batch of wavelengths: the design parameter is captured, ``requires_grad``,
    and its gradient through ``vmap`` is the sum of the lanes' eager gradients. A captured leaf is
    not batched, but everything it touches beside a batched wavelength is, and reads
    ``requires_grad`` False at the wrapper."""
    f, at, wavelengths = CAPTURED[name]
    w = torch.as_tensor(_weights(f('numpy', at, wavelengths[0]).size))
    q = _t(at).requires_grad_(True)
    func.vmap(lambda wl: (w * f('torch', q, wl)).sum())(_t(wavelengths)).sum().backward()
    eager = 0.0
    for wl in wavelengths:
        p = _t(at).requires_grad_(True)
        (w * f('torch', p, _t(wl))).sum().backward()
        eager += float(p.grad)
    assert float(q.grad) == pytest.approx(eager, rel=1e-11)


# ------------------------------------------------------------------- what moves no operator

def _first(transform, g, lanes):
    x = _t(lanes)
    if transform == 'vmap + backward':
        x.requires_grad_(True)
        func.vmap(g)(x).sum().backward()
        return x.grad.numpy()
    if transform == 'grad(vmap)':
        return func.grad(lambda y: func.vmap(g)(y).sum())(x).numpy()
    if transform == 'vmap(grad)':
        return func.vmap(func.grad(g))(x).numpy()
    if transform == 'torch.func.grad':
        return np.array([float(func.grad(g)(_t(p))) for p in lanes])
    if transform == 'torch.func.jvp':
        return np.array([float(func.jvp(g, (_t(p),), (_t(1.0),))[1]) for p in lanes])
    if transform == 'jvp(vmap)':
        return func.jvp(func.vmap(g), (x,), (torch.ones_like(x),))[1].numpy()
    assert transform == 'forward_ad + vmap'
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(x, torch.ones_like(x))
        return forward_ad.unpack_dual(func.vmap(g)(dual)).tangent.numpy()


FIRST = ['vmap + backward', 'grad(vmap)', 'vmap(grad)', 'torch.func.grad', 'torch.func.jvp',
         'jvp(vmap)', 'forward_ad + vmap']


@functools.lru_cache(maxsize=None)
def _eager_orders(name):
    """Eager first and second derivatives of the weighted read-out, lane by lane."""
    _, at, h = ALL[name]
    g = _scalar(name)
    first, second = [], []
    for p in _lanes(at, h):
        x = _t(p).requires_grad_(True)
        gx, = torch.autograd.grad(g(x), x, create_graph=True)
        first.append(float(gx))
        second.append(float(torch.autograd.grad(gx, x)[0]))
    return np.array(first), np.array(second)


@pytest.mark.parametrize('transform', FIRST + ['hessian', 'create_graph through vmap'])
@pytest.mark.parametrize('name', list(JAX_ONLY))
def test_a_parameter_that_moves_no_operator_is_exact_under_every_transform(name, transform):
    """A thickness moves no layer's modal operator, only its exponential, so no first-order rule
    is differentiated in it and nothing refuses it: with a constant permittivity every
    ``torch.func`` transform and both second orders return the eager derivative -- a detector
    that counted a ``torch.func`` wrapper by its type rather than by its level's own flag refused
    five of these first orders and both second orders. With a captured ``requires_grad``
    permittivity in the operator: :func:`test_a_captured_parameter_leaves_a_thickness_exact`.
    Not ``jvp(jvp)``, which carries ``torch.linalg.solve``'s own forward-over-forward error
    (torch 2.13)."""
    _, at, h = JAX_ONLY[name]
    g = _scalar(name)
    lanes = _lanes(at, h)
    first, second = _eager_orders(name)
    if transform in FIRST:
        assert _close(_first(transform, g, lanes), first, 1e-11)
    elif transform == 'hessian':
        assert float(func.hessian(g)(_t(at))) == pytest.approx(second[0], rel=1e-11)
    else:
        x = _t(lanes).requires_grad_(True)
        gx, = torch.autograd.grad(func.vmap(g)(x).sum(), x, create_graph=True)
        assert _close(torch.autograd.grad(gx.sum(), x)[0], second, 1e-11)


def _one_d_thickness(b, q, d):
    """``test_forward_mode``'s 1D TM SMM thickness scene, its permittivity moved by ``q``."""
    eps = _lift(b, GRATING) + q * _lift(b, GRATING_DIRECTION)
    solver = _legacy_wavebit.rcwa(backend=b, structure=eps,
                          **dict(ONE_D, layer_algorithm='SMM', tetm='tm', thickness=(d,)))
    return _read_out(b, solver.solve(), 'r_p', 3, point=([57.], [0.], [83.]))


def _gold_thickness(b, q, d):
    """The gold disks, ``Re eps`` ``q`` and thickness ``d``: the NVF floor binds."""
    piece = dict(type='circle', cx=0.0, cy=0.0, lx=140.0, material=q + 1j * GOLD.imag)
    solver = _legacy_wavebit.rcwa(backend=b, structure=[dict(background=2.25, shapes=[piece])],
                          top=2.25, bot=2.25, theta=0.3, phi=0.4, psi=0.7, wavelength=852.0,
                          period=(450.0, 450.0), thickness=(d,), truncation_order=(4, 4))
    return _read_out(b, solver.solve(), 'r_s', 40)


#: name: (read-out of (permittivity, thickness), permittivity, thickness lanes).
THICKNESS = {'1d TM, SMM': (_one_d_thickness, 0.0, [170.0, 175.0]),
             'gold disks, floor binds': (_gold_thickness, GOLD.real, [50.0, 52.0])}


def _thickness_scalar(name):
    f, q, lanes = THICKNESS[name]
    w = torch.as_tensor(_weights(f('numpy', q, lanes[0]).size))
    return lambda e, d: (w * f('torch', e, d)).sum()


@functools.lru_cache(maxsize=None)
def _thickness_eager(name):
    """Eager ``d/d(thickness)`` per lane, and the mixed ``d2/(d eps d thickness)`` at the first."""
    g = _thickness_scalar(name)
    _, q, lanes = THICKNESS[name]
    first = []
    for d in lanes:
        x = _t(d).requires_grad_(True)
        g(_t(q), x).backward()
        first.append(float(x.grad))
    e, x = _t(q).requires_grad_(True), _t(lanes[0]).requires_grad_(True)
    gx, = torch.autograd.grad(g(e, x), x, create_graph=True)
    return np.array(first), float(torch.autograd.grad(gx, e)[0])


def _captured_route(route, g, q, lanes):
    """``route``'s derivative in the thickness, lane by lane, the permittivity ``q`` captured."""
    h = lambda d: g(q, d)  # noqa: E731
    x = _t(lanes)
    each = {'torch.func.grad': lambda d: func.grad(h)(d),
            'torch.func.jvp': lambda d: func.jvp(h, (d,), (_t(1.0),))[1],
            'jacrev': lambda d: func.jacrev(h)(d), 'jacfwd': lambda d: func.jacfwd(h)(d)}
    if route in each:
        return torch.stack([each[route](_t(d)) for d in lanes])
    if route == 'vmap(grad)':
        return func.vmap(func.grad(h))(x)
    if route == 'grad(vmap)':
        return func.grad(lambda y: func.vmap(h)(y).sum())(x)
    assert route == 'jvp(vmap)'
    return func.jvp(func.vmap(h), (x,), (torch.ones_like(x),))[1]


#: A transform over the thickness, then ``backward()`` to the captured permittivity.
THEN_BACKWARD = {
    'torch.func.grad': lambda h, d: func.grad(h)(_t(d)),
    'torch.func.jvp': lambda h, d: func.jvp(h, (_t(d),), (_t(1.0),))[1],
    'jacfwd': lambda h, d: func.jacfwd(h)(_t(d)),
    'jvp(vmap)': lambda h, d: func.jvp(func.vmap(h), (_t([d]),), (_t([1.0]),))[1].sum(),
}
CAPTURED_ROUTES = (['torch.func.grad', 'torch.func.jvp', 'jacrev', 'jacfwd', 'vmap(grad)',
                    'grad(vmap)', 'jvp(vmap)']
                   + [f'{transform}, then backward()' for transform in THEN_BACKWARD]
                   + ['forward_ad dual outside torch.func.grad'])


@pytest.mark.parametrize('route', CAPTURED_ROUTES)
@pytest.mark.parametrize('name', list(THICKNESS))
def test_a_captured_parameter_leaves_a_thickness_exact(name, route):
    """A ``torch.func`` transform over the thickness while a permittivity captured with
    ``requires_grad`` -- an ``nn.Parameter`` -- feeds the modal operator: every value computed
    inside the transform carries its ``GradTrackingTensor`` wrapper, the operator's included,
    whose own ``requires_grad`` is False while the level under it requires grad. The transform
    differentiates no operator, so its derivative is eager's; and the mixed derivative to the
    captured permittivity, by a later ``backward()`` or by a ``forward_ad`` dual outside
    ``torch.func.grad``, is eager's ``create_graph`` one, on ETM and SMM alike. After
    ``torch.func.jvp``, ``jacfwd`` or ``jvp(vmap)`` that backward pass is reverse over forward,
    and every solve it differentiates goes through ``backend.linear_solve``; through
    ``torch.linalg.solve`` it read 0.12 relative off on the SMM scene's field read-out and 0.018
    on the gold disks, with no error, until 2026-10-12. A door that counted the
    transform's level by its type refused all of these where 806661e4 returned the first orders
    exactly and the mixed ones 0.15 relative off, without the eigen-derivative (the dual one
    through a tangent read beneath the grad level, which reads None unless read at its own
    level)."""
    g = _thickness_scalar(name)
    _, q, lanes = THICKNESS[name]
    first, mixed = _thickness_eager(name)
    if route.endswith(', then backward()'):
        e = _t(q).requires_grad_(True)
        THEN_BACKWARD[route[:-len(', then backward()')]](lambda d: g(e, d), lanes[0]).backward()
        assert float(e.grad) == pytest.approx(mixed, rel=1e-11)
    elif route == 'forward_ad dual outside torch.func.grad':
        with forward_ad.dual_level():
            e = forward_ad.make_dual(_t(q), _t(1.0))
            got = forward_ad.unpack_dual(func.grad(lambda d: g(e, d))(_t(lanes[0]))).tangent
        assert float(got) == pytest.approx(mixed, rel=1e-11)
    else:
        got = _captured_route(route, g, _t(q).requires_grad_(True), lanes).detach()
        assert _close(got, first, 1e-11), (got, first)


NESTED_ROUTES = {
    'hessian in the thickness, the permittivity captured':
        lambda g, q, d: (lambda e: func.hessian(lambda t: g(e, t))(_t(d)))(
            _t(q).requires_grad_(True)),
    'jacfwd over the permittivity of torch.func.grad over the thickness':
        lambda g, q, d: func.jacfwd(lambda e: func.grad(lambda t: g(e, t))(_t(d)))(_t(q)),
    'jvp over the permittivity of torch.func.grad over the thickness':
        lambda g, q, d: func.jvp(lambda e: func.grad(lambda t: g(e, t))(_t(d)), (_t(q),),
                                 (_t(1.0),)),
    'jvp over the permittivity of torch.func.jvp over the thickness':
        lambda g, q, d: func.jvp(lambda e: func.jvp(lambda t: g(e, t), (_t(d),), (_t(1.0),))[1],
                                 (_t(q),), (_t(1.0),)),
}


@pytest.mark.parametrize('route', list(NESTED_ROUTES))
@pytest.mark.parametrize('name', list(THICKNESS))
def test_nested_levels_over_a_moving_operator_are_refused(name, route):
    """Nested ``torch.func`` grad or jvp levels where the operator carries a derivative: a level
    beneath the transform's own counts by its type at the door, so these are refused by name.
    The mixed partials -- ``jacfwd`` or ``jvp`` over the permittivity of ``torch.func.grad`` or
    ``jvp`` over the thickness -- went down the constant branch at 806661e4 and returned 0.15
    relative off on the 1D TM scene, with no error: the permittivity's tangent sits on a jvp level
    beneath the thickness's, and read from above it reads None (a ``GradTrackingTensor`` level is
    lifted by the interpreter on top). It is read at its own level now, and the solve reaches the
    door. The ``hessian`` with a captured permittivity is a refusal of what would be exact -- the
    price of not reading a nested level's liveness at the door."""
    g = _thickness_scalar(name)
    _, q, lanes = THICKNESS[name]
    with pytest.raises(NotImplementedError, match='not differentiable under torch.func'):
        NESTED_ROUTES[route](g, q, lanes[0])


# ------------------------------------------------------------------------------- refusals

REFUSED_SCENES = ['1d TE, eps, ETM', '2d Li, pixel eps', 'gold disks, floor binds, Re eps']

FUNC_ROUTES = {
    'grad': lambda g, x, xs: func.grad(g)(x),
    'vjp': lambda g, x, xs: func.vjp(g, x)[1](torch.ones_like(x)),
    'jvp': lambda g, x, xs: func.jvp(g, (x,), (torch.ones_like(x),)),
    'jacrev': lambda g, x, xs: func.jacrev(g)(x),
    'jacfwd': lambda g, x, xs: func.jacfwd(g)(x),
    'hessian': lambda g, x, xs: func.hessian(g)(x),
    'vmap(grad)': lambda g, x, xs: func.vmap(func.grad(g))(xs),
    'grad(vmap)': lambda g, x, xs: func.grad(lambda y: func.vmap(g)(y).sum())(xs),
    'jacrev(vmap)': lambda g, x, xs: func.jacrev(func.vmap(g))(xs),
    'jacfwd(vmap)': lambda g, x, xs: func.jacfwd(func.vmap(g))(xs),
    'jvp(vmap)': lambda g, x, xs: func.jvp(func.vmap(g), (xs,), (torch.ones_like(xs),)),
}


@pytest.mark.parametrize('route', list(FUNC_ROUTES))
@pytest.mark.parametrize('name', REFUSED_SCENES)
def test_torch_func_grad_and_jvp_levels_are_refused_by_name(name, route):
    """Every ``torch.func`` level that could differentiate a first-order rule -- each of the six
    transforms, and each composed with ``vmap`` in either order -- raises ``NotImplementedError``
    pointing at ``torch.autograd``: the modal rule's, or where the NVF floor binds (or, under
    ``vmap``, might) the floor's. The rules have ``vmap`` rules now; what they cannot do is tell a
    ``torch.func`` grad level from ``create_graph=True``, or guard a ``jvp`` at all. Without the
    door ``torch.func.jvp`` returned a value, ``torch.func.grad`` raised the ``create_graph``
    refusal, and ``jvp(jvp)`` returned a partial Hessian."""
    _, at, h = ALL[name]
    g = _scalar(name)
    with pytest.raises(NotImplementedError, match='not differentiable under torch.func'):
        FUNC_ROUTES[route](g, _t(at), _t(_lanes(at, h)))


def _create_graph_through_vmap(g, lanes):
    x = _t(lanes).requires_grad_(True)
    gx, = torch.autograd.grad(func.vmap(g)(x).sum(), x, create_graph=True)
    return torch.autograd.grad(gx.sum(), x)


def _forward_over_backward_through_vmap(g, lanes):
    x = _t(lanes).requires_grad_(True)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(x, torch.ones_like(x))
        gradient, = torch.autograd.grad(func.vmap(g)(dual).sum(), x)
        return forward_ad.unpack_dual(gradient)


def _backward_over_forward_through_vmap(g, lanes):
    x = _t(lanes).requires_grad_(True)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(x, torch.ones_like(x))
        tangent = forward_ad.unpack_dual(func.vmap(g)(dual)).tangent
        return torch.autograd.grad(tangent.sum(), x)


SECOND_ROUTES = {
    'create_graph through vmap': (_create_graph_through_vmap, RuntimeError, 'first order'),
    'forward_ad over backward, through vmap': (_forward_over_backward_through_vmap, RuntimeError,
                                               'first order'),
    'backward over forward_ad, through vmap': (_backward_over_forward_through_vmap, RuntimeError,
                                               'first order'),
    'jvp(jvp)': (lambda g, lanes: func.jvp(lambda y: func.jvp(g, (y,), (_t(1.0),))[1],
                                           (_t(lanes[0]),), (_t(1.0),)),
                 NotImplementedError, 'not differentiable under torch.func'),
    'hessian': (lambda g, lanes: func.hessian(g)(_t(lanes[0])), NotImplementedError,
                'not differentiable under torch.func'),
}


@pytest.mark.parametrize('route', list(SECOND_ROUTES))
@pytest.mark.parametrize('name', ['1d TE, eps, ETM', 'conical, reported grating, wavelength',
                                  '2d Li, pixel eps', 'gold disks, floor binds, Re eps'])
def test_second_orders_through_the_rules_are_refused(name, route):
    """The first-order rules hold the basis and the floor's factors fixed, so a derivative of
    their derivative is a partial Hessian. Through ``vmap`` as eager, a second order where the
    operator moves under both derivatives raises: ``create_graph=True`` and both mixed orders
    with ``RuntimeError`` naming the first-order limit, ``jvp(jvp)`` and ``hessian`` at the
    ``torch.func`` door. At 806661e4 ``create_graph=True`` through ``vmap`` returned a partial
    Hessian on 12 of the 13 scenes measured, with no error."""
    _, at, h = ALL[name]
    run, error, words = SECOND_ROUTES[route]
    with pytest.raises(error, match=words):
        run(_scalar(name), _lanes(at, h))


# ------------------------------------------------------------------------------- the NVF floor

FLOOR_LANES = {'gold, dielectric': [GOLD.real, 4.0], 'dielectric, gold': [4.0, GOLD.real],
               'dielectric, dielectric': [4.0, 4.5], 'gold, gold': [GOLD.real, GOLD.real]}


@pytest.mark.parametrize('lanes', list(FLOOR_LANES))
def test_floor_lanes_that_disagree_about_binding(lanes):
    """Gold disks bind the NVF floor; at a ``Re eps`` of 4.0 or 4.5 it does not. Batched together,
    whether it binds is a per-lane answer the solve cannot read, so the floor runs on every lane
    and a per-lane ``where`` hands each lane where it does not bind ``B`` itself. The read-out,
    ``backward()`` and ``forward_ad`` through ``vmap`` are eager's per lane. The values are compared
    to 1e-13, not to the bit: on Linux (OpenBLAS) the batched solve itself differs from the eager one
    by 1.6e-15 even where nothing binds, so the ``where`` is checked bit for bit on the floor alone
    (:func:`test_a_lane_where_the_floor_does_not_bind_gets_its_matrix_back_bit_for_bit`)."""
    f = gold_disks('Re eps')
    points = FLOOR_LANES[lanes]
    w = torch.as_tensor(_weights(4))
    eager_values, eager_gradients, eager_tangents = [], [], []
    for p in points:
        x = _t(p).requires_grad_(True)
        value = f('torch', x)
        (w * value).sum().backward()
        eager_values.append(value.detach())
        eager_gradients.append(float(x.grad))
        eager_tangents.append(_eager_tangent(f, p))
    x = _t(points).requires_grad_(True)
    values = func.vmap(lambda p: f('torch', p))(x)
    for got, want in zip(values.detach(), eager_values):
        assert _close(got, want, 1e-13), (got - want)
    (values * w).sum().backward()
    assert _close(x.grad, eager_gradients, 1e-11)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(_t(points), torch.ones(2, dtype=torch.float64))
        tangent = forward_ad.unpack_dual(func.vmap(lambda p: f('torch', p))(dual)).tangent
    assert _close(tangent, np.stack(eager_tangents), 1e-11)


def test_a_lane_where_the_floor_does_not_bind_gets_its_matrix_back_bit_for_bit():
    """The per-lane ``where`` of :func:`_floor_singular_values` under ``vmap``, alone: a batch of
    one matrix the floor binds on (smallest singular value 1e-3 under a floor of 0.1) and one it
    does not (smallest 0.5), each requiring grad. The second lane is that matrix itself, to the
    bit, on every platform: the ``where`` selects it, nothing recomputes it. Without the ``where``
    it came back rebuilt from its SVD, a few ULPs off -- the defect the end-to-end test above can
    only see to its tolerance. The first lane is the eager floor's."""
    from wavebit._common.backend import array_namespace
    from wavebit.rcwa._engine.factorization import _floor_singular_values

    xp = array_namespace(torch.zeros(1))
    rng = np.random.default_rng(7)

    def unitary():
        q, _ = np.linalg.qr(rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5)))
        return q

    def with_singular_values(s):
        return unitary() @ np.diag(s) @ unitary().conj().T

    B = torch.as_tensor(np.stack([with_singular_values([2.0, 1.5, 1.0, 0.7, 1e-3]),
                                  with_singular_values([2.0, 1.5, 1.0, 0.7, 0.5])])).requires_grad_(True)
    floor = 0.1
    got = func.vmap(lambda b: _floor_singular_values(xp, b, floor))(B)
    assert torch.equal(got[1].detach(), B[1].detach())
    eager = _floor_singular_values(xp, B[0], floor)
    assert float((got[0] - eager).abs().max()) <= 1e-13 * float(eager.abs().max())
    assert not torch.equal(got[0].detach(), B[0].detach())        # premise: it binds there


# --------------------------------------------------------------------------- the rules alone

def _complex(rng, *shape):
    return torch.as_tensor(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


BATCHINGS = ['operator and basis batched', 'operator batched, basis captured',
             'operator captured, basis batched', 'forward_ad']


@pytest.mark.parametrize('batching', BATCHINGS)
def test_the_modal_rule_under_vmap_in_every_batching(batching):
    """``modal_gradient._variation`` alone, under ``vmap``, against an eager loop: its ``vmap``
    rule expands whichever input is not batched, so the output is batched whenever either is.
    A generated rule returned an unbatched zero beside a batched operator, and the cotangent
    summed over the lanes landed on every lane: three times too large on three lanes, silently,
    in the batching no solve reaches (the operator batched, the basis captured)."""
    from wavebit.rcwa._engine.modal_gradient import _variation
    rng = np.random.default_rng(7)
    n, k = 3, 3
    A, V, C = _complex(rng, n, k, k), _complex(rng, n, k, k), _complex(rng, k, k)

    def loss(out):
        return (C * out).real.sum()

    if batching == 'forward_ad':
        T = _complex(rng, n, k, k)
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(A, T)
            got = forward_ad.unpack_dual(func.vmap(_variation)(dual, V)).tangent
        want = torch.stack([torch.linalg.solve(V[i], T[i] @ V[i]) for i in range(n)])
    elif batching == 'operator captured, basis batched':
        a = A[0].clone().requires_grad_(True)
        func.vmap(lambda v: loss(_variation(a * 1.0, v)))(V).sum().backward()
        got, want = a.grad, torch.zeros_like(A[0])
        for i in range(n):
            x = A[0].clone().requires_grad_(True)
            loss(_variation(x, V[i])).backward()
            want = want + x.grad
    else:
        captured = batching == 'operator batched, basis captured'
        a = A.clone().requires_grad_(True)
        if captured:
            func.vmap(lambda x: loss(_variation(x, V[0])))(a).sum().backward()
        else:
            func.vmap(lambda x, v: loss(_variation(x, v)))(a, V).sum().backward()
        got, want = a.grad, []
        for i in range(n):
            x = A[i].clone().requires_grad_(True)
            loss(_variation(x, V[0] if captured else V[i])).backward()
            want.append(x.grad)
        want = torch.stack(want)
    assert _close(torch.view_as_real(got), torch.view_as_real(want), 1e-13)


def _floor_batch():
    """Three lanes of 3 x 3 matrices -- as many lanes as singular values -- and a floor per lane:
    lanes 0 and 2 bind, lane 1 does not."""
    rng = np.random.default_rng(7)
    n = k = 3
    U = torch.linalg.qr(_complex(rng, n, k, k))[0]
    W = torch.linalg.qr(_complex(rng, n, k, k))[0]
    s = _t([[1.0, 0.5, 0.01], [1.0, 0.8, 0.6], [2.0, 0.02, 0.001]])
    B = U @ torch.diag_embed(s.to(torch.complex128)) @ W.mH
    return B, _t([0.1, 0.1, 0.05]), _complex(rng, k, k)


def test_the_floor_under_vmap_when_the_batch_matches_the_singular_values():
    """``factorization._floor_singular_values`` alone, under ``vmap``, three lanes of three
    singular values. Its ``vmap`` rule gives each lane's floor singleton axes so it broadcasts as
    the eager 0-d floor does; without them it broadcast along the singular values, an error
    where the lengths differ and, where they agree as here, wrong by 0.37 in ``d/dB``, 1.2 in
    ``d/dfloor`` and 0.34 in the tangent, with no error. The lane that does not bind is ``B``."""
    from wavebit._common.backend import array_namespace
    from wavebit.rcwa._engine.factorization import _floor_singular_values
    B, floors, C = _floor_batch()
    xp = array_namespace(B)

    def loss(out):
        return (C * out).real.sum()

    b, f = B.clone().requires_grad_(True), floors.clone().requires_grad_(True)
    values = func.vmap(lambda x, y: _floor_singular_values(xp, x, y))(b, f)
    func.vmap(loss)(values).sum().backward()
    want_b, want_f, want_values = [], [], []
    for i in range(3):
        x, y = B[i].clone().requires_grad_(True), floors[i].clone().requires_grad_(True)
        value = _floor_singular_values(xp, x, y)
        loss(value).backward()
        want_values.append(value.detach())
        want_b.append(x.grad)
        want_f.append(torch.zeros_like(y) if y.grad is None else y.grad)
    assert _close(torch.view_as_real(b.grad), torch.view_as_real(torch.stack(want_b)), 1e-13)
    assert _close(f.grad, torch.stack(want_f), 1e-13)
    assert torch.equal(values[1].detach(), B[1])
    for i in (0, 2):
        assert float((values[i].detach() - want_values[i]).abs().max()) < 1e-14

    T = _complex(np.random.default_rng(8), 3, 3, 3)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(B, T)
        got = forward_ad.unpack_dual(
            func.vmap(lambda x, y: _floor_singular_values(xp, x, y))(dual, floors)).tangent
    want = []
    for i in range(3):
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(B[i], T[i])
            want.append(forward_ad.unpack_dual(_floor_singular_values(xp, dual, floors[i])).tangent)
    assert _close(torch.view_as_real(got), torch.view_as_real(torch.stack(want)), 1e-13)


@pytest.mark.parametrize('route', ['jvp', 'grad', 'vmap(grad)', 'grad(vmap), nothing binds',
                                   'grad over the floor alone'])
def test_the_floor_refuses_torch_func_by_itself(route):
    """The floor's own ``torch.func`` door, with no modal rule behind it to refuse first: in a
    solve the modal door masks a floor that lets ``torch.func.jvp`` through. Where it binds every
    grad or jvp level is refused; under ``vmap`` with such a level so is a batch where nothing
    binds, since that is a per-lane answer that cannot be read. The door asks the floor as well
    as ``B``: with ``B`` constant and the floor the transform's input -- a direct call, since in
    a solve both come from the permittivity -- a door that asked ``B`` alone let the level reach
    the rule's backward."""
    from wavebit._common.backend import array_namespace
    from wavebit.rcwa._engine.factorization import _floor_singular_values
    B, floors, C = _floor_batch()
    xp = array_namespace(B)

    def loss(x, floor=floors[0]):
        return (C * _floor_singular_values(xp, x, floor)).real.sum()

    run = {'jvp': lambda: func.jvp(loss, (B[0],), (B[0],)),
           'grad': lambda: func.grad(loss)(B[0]),
           'vmap(grad)': lambda: func.vmap(func.grad(loss))(B),
           'grad(vmap), nothing binds':
               lambda: func.grad(lambda b: func.vmap(lambda x: loss(x, _t(1e-4)))(b).sum())(B),
           'grad over the floor alone': lambda: func.grad(lambda f: loss(B[0], f))(floors[0])}
    with pytest.raises(NotImplementedError, match='floor is not differentiable under torch.func'):
        run[route]()


@pytest.mark.parametrize('mode', ['reverse over forward_ad', 'forward_ad over reverse'])
def test_the_floor_alone_refuses_a_mixed_second_order_through_vmap(mode):
    """``test_nvf_metal_inverse_rule``'s mixed second orders through the floor alone, on two lanes
    under ``vmap``: the floor's ``vmap`` rule hands the Function the ``dual`` flag it was given,
    and its backward refuses any pass through a floor built from a dual. A rule that passed
    ``dual=False`` returned -0.0075 per lane reverse over forward, against the exact +0.036047 at
    ``t = 0.3`` (a central difference of the exact first derivative), and 0.0137 the other way,
    with no error. In a whole solve the modal rule's own guard fires first and masks it; this is
    the route ``eps_conv_matrices()`` takes."""
    from tests.rcwa.test_nvf_metal_inverse_rule import _floor_test_matrices, _namespace
    from wavebit.rcwa._engine.factorization import _floor_singular_values
    xp = _namespace('torch')
    B0, D, R = _floor_test_matrices(True, [0.004, 0.01, 0.3, 1.0])

    def loss(t):
        B = xp.asarray(B0) + 0.002 * xp.sin(t) * xp.asarray(D)
        floored = _floor_singular_values(xp, B, 0.02 + 0.006 * t * t)
        return xp.sum(xp.real(xp.conj(xp.asarray(R)) * floored))

    x = _t([0.3, 0.31]).requires_grad_(True)
    with forward_ad.dual_level(), pytest.raises(RuntimeError, match='first order'):
        out = func.vmap(loss)(forward_ad.make_dual(x, torch.ones(2, dtype=torch.float64)))
        if mode == 'reverse over forward_ad':
            torch.autograd.grad(forward_ad.unpack_dual(out).tangent.sum(), x)
        else:
            torch.autograd.grad(out.sum(), x)


def test_eig_through_vmap():
    """``backend.eig`` under ``vmap`` on a live batched operator: its dispatch asks the operator's
    own liveness, so a batched tensor that requires grad goes to the native ``torch.linalg.eig``
    and ``backward()`` through ``vmap`` is eager's per lane; a ``forward_ad`` tangent goes to the
    old-style ``_Eig``, which has no functorch rule, and PyTorch refuses it by its
    ``setup_context`` error -- loud, and no solve hands ``eig`` a live operator. A dispatch on
    the peeling ``carries_autodiff`` sent the requires-grad batch to ``_Eig`` too, where it
    raised."""
    from wavebit._common import backend
    rng = np.random.default_rng(0)
    A = _complex(rng, 2, 4, 4)

    def loss(a):
        w, _ = backend.eig(a)
        return (w.real ** 2).sum()

    a = A.clone().requires_grad_(True)
    func.vmap(loss)(a).sum().backward()
    want = []
    for i in range(2):
        x = A[i].clone().requires_grad_(True)
        loss(x).backward()
        want.append(x.grad)
    assert _close(torch.view_as_real(a.grad), torch.view_as_real(torch.stack(want)), 1e-12)
    with forward_ad.dual_level(), pytest.raises(RuntimeError, match='setup_context'):
        func.vmap(loss)(forward_ad.make_dual(A, torch.ones_like(A)))


# ------------------------------------------------------------------------ the geometry readers

def _sampled_tensor_disc(lx=270.0, period=600.0):
    return _legacy_wavebit.rcwa(
        backend='torch', wavelength=700., period=(period, 600.), thickness=[170.],
        structure=[dict(background=1., shapes=[dict(type='circle', lx=lx,
                                                      material=wavebit.Material([2., 3., 4.]))])],
        truncation_order=(1, 1), shape_nvf_grid_size=24, phi=.1, fourier_coefficients='sampled')


@pytest.mark.parametrize('moved, words', [('lx', 'geometry derivatives'),
                                          ('period', 'lattice derivatives')])
def test_a_sampled_tensor_shape_refuses_a_geometry_derivative_under_vmap(moved, words):
    """Sampled tensor shapes paint a fixed mask, so a geometry derivative through them would be a
    plausible zero, and they refuse one by name -- eager, and under ``vmap`` now too. At
    806661e4 the batched diameter or period read as constant to those checks and went on to
    PyTorch's "Cannot access data pointer of Tensor that doesn't have storage". (A rectangle's
    width under ``vmap``, which reaches the shape cache's key, is exact:
    :func:`test_backward_through_vmap_is_the_eager_derivative` on ``shape, rectangle lx, grid 256``.)"""
    x = _t([270.0, 280.0] if moved == 'lx' else [600.0, 610.0]).requires_grad_(True)
    with pytest.raises(ValueError, match=words):
        func.vmap(lambda p: _sampled_tensor_disc(**{moved: p}).solve().R.real.sum())(x)


# ---------------------------------------------------------------------- the private torch API

def _inside(transform, probe, x):
    """``probe`` of the value a transform hands its function, read out from inside it."""
    seen = []

    def f(y):
        seen.append(probe(y))
        return (y * 1.0).sum()

    transform(f)(x)
    return seen[0]


def test_the_private_functorch_calls_exist():
    """``torch._C._functorch``'s eight calls and the transforms-active probe, which are private
    and which the detection rests on: a PyTorch that renamed one is refused by name inside a
    transform (see below), and this names it before anyone gets there."""
    from wavebit._common.backend import _FUNCTORCH_CALLS
    assert callable(torch._C._are_functorch_transforms_active)
    assert set(_FUNCTORCH_CALLS) == {'is_functorch_wrapped_tensor', 'get_unwrapped',
                                     'is_batchedtensor', 'is_gradtrackingtensor',
                                     'maybe_get_level', 'maybe_current_level',
                                     'pop_dynamic_layer_stack', 'push_dynamic_layer_stack'}
    for name in _FUNCTORCH_CALLS:
        assert callable(getattr(torch._C._functorch, name)), name


def test_the_private_functorch_calls_mean_what_the_detection_reads():
    """What the walks read of those calls, on toy tensors, so a PyTorch that kept the names and
    changed their meaning fails here rather than silently in a solve: the probe is True inside
    ``vmap`` and ``torch.func.grad`` and False outside;
    :func:`~wavebit._common.backend.torch_autodiff` finds a ``requires_grad`` under a batch, under
    ``grad(vmap)``, in a batched value times a captured leaf, and a ``forward_ad`` tangent under a
    batch, and finds nothing in a batch with nothing differentiated nor in a value detached inside
    ``torch.func.grad`` (a level's own flag, not its type); it finds a tangent beneath another
    grad level -- a jvp level's under ``torch.func.grad``, and a ``forward_ad`` dual's made
    outside it -- which a plain ``unpack_dual`` reads as None from above (the tangent is read at
    its own level, popping the interpreters above it, and the stack is whole again after);
    :func:`~wavebit._common.backend.functorch_differentiates` finds a live grad or jvp level, alone
    or beside ``vmap`` either way round, and not ``vmap`` alone, nor a transform's level that a
    captured ``requires_grad`` leaf's value merely carries; a level nested beneath another counts
    by its type."""
    from wavebit._common.backend import _functorch_active, functorch_differentiates, torch_autodiff
    x = _t([1.0, 2.0])
    live = x.clone().requires_grad_(True)
    leaf = _t(3.0).requires_grad_(True)
    active = lambda y: _functorch_active(torch)  # noqa: E731
    vmap, grad = func.vmap, func.grad
    assert not _functorch_active(torch)
    assert _inside(vmap, active, x) and _inside(grad, active, _t(1.0))

    assert _inside(vmap, torch_autodiff, live)
    assert _inside(lambda f: grad(lambda y: vmap(f)(y).sum()), torch_autodiff, x)
    assert _inside(vmap, lambda y: torch_autodiff(y * leaf), x)
    assert not _inside(vmap, torch_autodiff, x)
    assert not _inside(grad, lambda y: torch_autodiff(y.detach()), _t(1.0))

    assert _inside(grad, functorch_differentiates, _t(1.0))
    assert _inside(lambda f: lambda y: func.jvp(f, (y,), (torch.ones_like(y),)),
                   functorch_differentiates, _t(1.0))
    assert _inside(lambda f: vmap(grad(f)), functorch_differentiates, x)
    assert _inside(lambda f: grad(lambda y: vmap(f)(y).sum()), functorch_differentiates, x)
    assert not _inside(vmap, functorch_differentiates, live)
    assert not _inside(grad, lambda y: functorch_differentiates(leaf * 2.0), _t(1.0))
    assert _inside(grad, lambda y: torch_autodiff(leaf * 2.0), _t(1.0))
    assert _inside(lambda f: grad(grad(f)), lambda y: functorch_differentiates(leaf * 2.0),
                   _t(1.0))

    # A value that depends on `e` only, read inside `torch.func.grad` over `d`: `e` a jvp level
    # beneath it, a `forward_ad` dual made outside it, or a constant under another grad level.
    depth = torch._C._functorch.get_dynamic_layer_stack_depth
    seen = []

    def beneath(e):
        def inner(d):
            y = e * 2.0
            seen.append((torch_autodiff(y), functorch_differentiates(y), depth()))
            return (e * d * d).sum()
        return func.grad(inner)(_t(2.0))
    func.jvp(beneath, (_t(1.5),), (_t(1.0),))
    with forward_ad.dual_level():
        beneath(forward_ad.make_dual(_t(1.5), _t(1.0)))
    func.grad(lambda d: beneath(_t(1.5)) * d)(_t(1.0))
    assert seen == [(True, True, 2), (True, False, 1), (False, True, 2)], seen

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(x, torch.ones_like(x))
        assert _inside(vmap, torch_autodiff, dual)
        assert not _inside(vmap, torch_autodiff, x)


def test_a_missing_private_call_is_refused_by_name(monkeypatch):
    """A PyTorch without one of the calls is refused inside a transform with a ``RuntimeError``
    naming it -- not an ``AttributeError`` deep in a solve, and not a tensor taken for one with
    no derivative. Where the transforms-active probe itself is missing the walk runs anyway, so
    that refusal is reached, and eager answers do not change."""
    from wavebit._common import backend
    monkeypatch.setattr(backend, '_FUNCTORCH_CALLS', backend._FUNCTORCH_CALLS + ('no_such_call',))
    with pytest.raises(RuntimeError, match=r'torch\._C\._functorch\.no_such_call, which PyTorch'):
        func.vmap(lambda y: torch.tensor(backend.torch_autodiff(y)))(_t([1.0, 2.0]))
    assert backend.torch_autodiff(_t(1.0).requires_grad_(True))
    monkeypatch.undo()
    monkeypatch.delattr(torch._C, '_are_functorch_transforms_active')
    assert backend._functorch_active(torch)
    assert not backend.torch_autodiff(_t(1.0))
    assert backend.torch_autodiff(_t(1.0).requires_grad_(True))


def test_eager_autodiff_never_calls_torch_c_functorch(monkeypatch):
    """Outside a transform there are no wrappers to walk, so eager PyTorch -- reverse mode and
    ``forward_ad``, through the modal rule and the NVF floor -- never calls ``torch._C._functorch``
    while the transforms-active probe exists: an eager solve's derivative does not depend on its
    calls. The probe itself, ``torch._C._are_functorch_transforms_active``, is read on every call,
    and on a PyTorch without it the walk runs eagerly too
    (:func:`test_a_missing_private_call_is_refused_by_name`)."""
    from wavebit._common import backend

    def unreachable(torch):
        raise AssertionError('eager PyTorch read torch._C._functorch')

    monkeypatch.setattr(backend, '_functorch', unreachable)
    for name in ('1d TE, eps, ETM', 'gold disks, floor binds, Re eps'):
        f, at, _ = SCENES[name]
        g = _scalar(name)
        gradient = _eager_gradient(g, at)
        w = _weights(_finite(name)[0].size)
        assert float(w @ _eager_tangent(f, at)) == pytest.approx(gradient, rel=1e-11)
        assert gradient == pytest.approx(float(w @ _finite(name)[1]), rel=1e-6)


# ------------------------------------------------------------------- what vmap alone leaves

@pytest.mark.parametrize('name, lanes', [('1d TE, eps, ETM', [0.0, 0.01]),
                                         ('2d Li, pixel eps', [0.0, 0.01]),
                                         ('gold disks, floor binds, Re eps', [GOLD.real, 4.0])])
def test_a_vmap_with_no_derivative_runs_no_first_order_rule(name, lanes, monkeypatch):
    """A forward sweep under ``vmap``, nothing differentiated: neither the modal rule nor the NVF
    floor's Function is applied, so the values are the constant branch's, as they were before
    this detection saw through the batch. A detector that counted every wrapper as live ran the
    modal rule here and moved the 2D raster's values by 8.8e-15."""
    from wavebit.rcwa._engine import factorization, modal_gradient
    calls = []
    for module, name_ in ((modal_gradient, '_torch_variation'), (factorization, '_torch_floor')):
        original = getattr(module, name_)
        monkeypatch.setattr(module, name_, lambda original=original, name_=name_:
                            calls.append(name_) or original())
    f = ALL[name][0]
    values = func.vmap(lambda p: f('torch', p))(_t(lanes))
    assert calls == []
    assert _close(values, torch.stack([f('torch', _t(p)) for p in lanes]), 1e-12)


def test_vectorized_jacobian_and_batched_gradients_stay_right():
    """PyTorch's own batching of a derivative -- ``torch.autograd.functional.jacobian(...,
    vectorize=True)`` and ``torch.autograd.grad(..., is_grads_batched=True)`` -- runs the rule's
    ``backward`` under ``vmap``, which was right before this change and is pinned here."""
    f, at, _ = SCENES['1d TE, eps, ETM']
    tangent = _eager_tangent(f, at)
    jacobian = torch.autograd.functional.jacobian(lambda p: f('torch', p), _t(at), vectorize=True)
    assert _close(jacobian, tangent, 1e-12)
    x = _t(at).requires_grad_(True)
    y = f('torch', x)
    rows, = torch.autograd.grad(y, x, torch.eye(y.shape[0], dtype=y.dtype), is_grads_batched=True)
    assert _close(rows, tangent, 1e-12)
