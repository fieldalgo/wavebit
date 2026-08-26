"""Every design parameter, every path, against a finite difference.

The suite had gradient tests for the raster ``structure`` and for a few shape parameters, and
they all passed while ``wavelength`` was 38-85% wrong, the superstrate 49% wrong, and
``theta`` / ``phi`` / ``psi`` / ``period`` / a shape's ``material`` / a layer's
``background`` / ``angle`` carried no gradient at all on PyTorch. Nothing was checking the
scalars.

Two failure modes to keep apart, because only one of them announces itself:

* a severed graph raises on PyTorch (``element 0 of tensors does not require grad``) and
  silently returns zero on JAX;
* a *partially* severed graph returns a plausible number. ``wavelength`` used to come back as
  exactly the ``k0 = 2*pi/lambda`` part of its own derivative, with the ``kx``/``ky`` path cut
  — no error, no zero, just wrong by a factor.

So each case compares against a central difference rather than a stored value: the claim is
that the derivative is the real one. Any parameter whose finite difference is itself zero is
reported as such rather than counted as a pass.
"""

import numpy as np
import pytest

import wavebit

RASTER = np.ones((1, 8, 8))
RASTER[0, 2:6, 2:6] = 9.0             # permittivity: 3.0**2

BASE = dict(period=[0.77, 0.77], thickness=[0.2], theta=0.2, phi=0.3, psi=0.4,
            wavelength=0.777, top=1.0, bot=2.25, truncation_order=[3, 3])

SHAPE = dict(type='rectangle', cx=0.385, cy=0.385, lx=0.385, ly=0.231, angle=0.4,
             material=12.25)                            # 3.5**2
ANALYTIC_BASE = dict(period=[0.77, 0.77], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
                     wavelength=0.777, top=1.0, bot=2.25, truncation_order=[4, 4])

def _frozen_raster():
    global FROZEN
    if FROZEN is None:
        FROZEN = wavebit.rcwa(backend=0, structure=RASTER, **BASE).freeze_geometry()
    return FROZEN


FROZEN = None

VECTOR_BASE = dict(period=[300.0, 300.0], thickness=[205.0], theta=0.1, truncation_order=[3, 3],
                   wavelength=900.0, top=1.0, bot=1.0, psi=np.pi / 4)


def _backend(name):
    pytest.importorskip(name)
    if name == 'jax':
        wavebit.enable_x64()
        import jax.numpy as jnp
        return 1, (lambda v: jnp.asarray(v, dtype=jnp.float64))
    import torch
    return 2, (lambda v: torch.tensor(v, dtype=torch.float64))


def _derivative(backend, objective, value, rel_step=1e-6):
    """(autograd, central difference) of ``objective`` at ``value``.

    ``rel_step`` matters more than it looks on the smooth path: the normal field is sampled
    on a grid of ``period / normal_grid``, so the objective carries a sawtooth of that
    period. A difference step that straddles one of those steps reports the jump, not the
    derivative — at ``cx = period/2``, which sits exactly on a grid line, ``h = 1e-6`` gives
    3.8 where the derivative is 3.2e-3. Stay well inside one cell.
    """
    if backend == 2:
        import torch
        p = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        objective(p).backward()
        auto = float(p.grad)
    else:
        import jax
        auto = float(jax.grad(lambda v: objective(v).real)(value))

    h = rel_step * max(abs(value), 1.0)
    scalar = (lambda v: float(objective(_backend_scalar(backend)(v))))
    return auto, (scalar(value + h) - scalar(value - h)) / (2 * h)


def _backend_scalar(backend):
    if backend == 2:
        import torch
        return lambda v: torch.tensor(v, dtype=torch.float64)
    import jax.numpy as jnp
    return lambda v: jnp.asarray(v, dtype=jnp.float64)


def _total(res):
    import torch
    d = res.T
    return torch.sum(d) if hasattr(d, 'backward') or hasattr(d, 'grad_fn') else d.sum()


# --------------------------------------------------------------------------- raster (topology)

RASTER_SCALARS = ['theta', 'phi', 'psi', 'wavelength', 'top', 'bot', 'thickness', 'period']


@pytest.mark.parametrize('name', ['torch', 'jax'])
@pytest.mark.parametrize('param', RASTER_SCALARS)
def test_raster_scalar_gradients(param, name):
    backend, cast = _backend(name)

    def objective(p):
        cfg = dict(BASE)
        if param == 'thickness':
            cfg['thickness'] = [p]
        elif param == 'period':
            cfg['period'] = [p, cast(0.77)]
        else:
            cfg[param] = p
        solver = wavebit.rcwa(backend=backend, structure=cast(RASTER), **cfg)
        # No frozen geometry: the default ('sampled', 'li') build has no discrete decision to
        # take, so it traces as is. It used to need one, for a same-material check that no
        # longer exists.
        return solver.solve().T.sum()

    # `top` / `bot` are permittivities now, so the gradient measured here is d/d(eps) --
    # a different number from the old d/dn, and the finite difference moves with it.
    start = {'theta': 0.2, 'phi': 0.3, 'psi': 0.4, 'wavelength': 0.777, 'top': 1.0,
             'bot': 2.25, 'thickness': 0.2, 'period': 0.77}[param]
    auto, fd = _derivative(backend, objective, start)
    assert abs(fd) > 1e-8, f'the objective does not depend on {param}; test is vacuous'
    assert abs(auto - fd) / abs(fd) < 1e-4, (param, auto, fd)


@pytest.mark.parametrize('name', ['torch', 'jax'])
def test_raster_per_cell_gradient_reaches_every_cell(name):
    """Sampled coefficients are what topology optimization needs: one gradient per cell.

    The integrated (closed-form) series legitimately gives 9 of 64 — it compresses the raster
    into blocks and the other cells genuinely stop being variables — which is why the choice
    is not an implementation detail, and why ``fourier_coefficients`` defaults to ``'sampled'``.
    Changing that default silently breaks every topology-optimization user.
    """
    backend, cast = _backend(name)

    def cells(geometry=None, **strategy):
        if backend == 2:
            import torch
            uc = torch.tensor(RASTER, dtype=torch.float64, requires_grad=True)
            res = wavebit.rcwa(backend=2, structure=uc, **strategy,
                                 **BASE).solve(geometry=geometry)
            torch.sum(res.T).backward()
            return int((uc.grad.abs() > 1e-14).sum())
        import jax, jax.numpy as jnp
        g = jax.grad(lambda u: jnp.sum(wavebit.rcwa(backend=1, structure=u, **strategy,
                     **BASE).solve(geometry=geometry).T).real)
        return int(np.sum(np.abs(np.asarray(g(jnp.asarray(RASTER)))) > 1e-14))

    # No strategy keywords at all: this is the *default* that has to carry every cell.
    assert cells() == RASTER.size

    integrated = dict(fourier_coefficients='integrated')
    solver = wavebit.rcwa(backend=backend, structure=cast(RASTER), **integrated, **BASE)
    assert cells(solver.freeze_geometry(), **integrated) < RASTER.size


# ------------------------------------------------------------------------- analytic (shape/OCD)

ANALYTIC_PARAMS = ['lx', 'ly', 'angle', 'material', 'background',
                   'theta', 'wavelength', 'thickness']


@pytest.mark.parametrize('name', ['torch', 'jax'])
@pytest.mark.parametrize('param', ANALYTIC_PARAMS)
def test_analytic_shape_gradients(param, name):
    backend, cast = _backend(name)

    def objective(p):
        shape, cfg, background = dict(SHAPE), dict(ANALYTIC_BASE), 1.0
        if param == 'background':
            background = p
        elif param == 'thickness':
            cfg['thickness'] = [p]
        elif param in shape:
            shape[param] = p
        else:
            cfg[param] = p
        res = wavebit.rcwa(backend=backend,
                             structure=[dict(background=background, shapes=[shape])],
                             **cfg).solve()
        return res.R.sum()

    start = dict(SHAPE, background=1.0, theta=0.35, wavelength=0.777, thickness=0.25)[param]
    auto, fd = _derivative(backend, objective, float(start), rel_step=1e-8)
    assert abs(fd) > 1e-8, f'the objective does not depend on {param}; test is vacuous'
    assert abs(auto - fd) / abs(fd) < 1e-3, (param, auto, fd)


# ------------------------------------------------------------------------------ vector (OCD)

@pytest.mark.parametrize('name', ['torch', 'jax'])
@pytest.mark.parametrize('param,index,start', [('lx', 3, 80.0), ('ly', 4, 60.0),
                                               ('angle', 6, 0.4)])
def test_vector_edge_gradients(param, index, start, name):
    """OCD differentiates the edges: CD, position, sidewall angle."""
    backend, cast = _backend(name)

    def objective(p):
        rect = ['rectangle', cast(150.0), cast(150.0), cast(80.0), cast(60.0), 4.0, cast(0.4)]
        rect[index] = p
        res = wavebit.rcwa(backend=backend, structure=[[12.25, [rect]]],
                             **VECTOR_BASE).solve()
        return res.T.sum()

    auto, fd = _derivative(backend, objective, start)
    assert abs(fd) > 1e-8, f'the objective does not depend on {param}; test is vacuous'
    assert abs(auto - fd) / abs(fd) < 1e-3, (param, auto, fd)


# ------------------------------------------------------------------- shape gradients near zero

def test_rectangle_is_built_the_same_way_on_both_sides_of_zero_angle():
    """A rectangle at -1e-6 rad must be built exactly like one at +1e-6.

    ``angle`` used to be wrapped into [0, 2*pi) and then tested against a list of aligned
    targets that stopped at ``3*pi/2``, so a small *negative* angle arrived as ``2*pi - eps``,
    missed every target, and fell through to the rotated decomposition while the positive side
    got an exact single block. The two disagreed by 2e-7 in efficiency — a step sitting exactly
    on the unrotated case, which is where any sidewall-angle fit starts.

    There is no aligned-target test left to get wrong: the angle folds onto ``[0, pi/2)`` and
    both sides run the same code. So the check is on the geometry the two sides produce, which
    is now identical rather than merely close, and on the unrotated case still sitting between
    them.
    """
    m = wavebit.rcwa(backend=0, period=[300.0, 300.0]).modeler
    blocks = [m.rectangle(150.0, 150.0, 80.0, 60.0, 4.0, a) for a in (-1e-6, 0.0, 1e-6)]
    f = lambda v: float(np.real(np.asarray(v).reshape(-1)[0]))

    def bbox(bs):
        ys = [f(b[0][0]) for b in bs] + [f(b[1][0]) for b in bs]
        xs = [f(b[0][1]) for b in bs] + [f(b[1][1]) for b in bs]
        return min(xs), max(xs), min(ys), max(ys)

    def area(bs):
        return sum((f(b[1][0]) - f(b[0][0])) * (f(b[1][1]) - f(b[0][1])) for b in bs)

    assert len(blocks[0]) == len(blocks[2]), [len(b) for b in blocks]
    assert bbox(blocks[0]) == bbox(blocks[2])
    for bs in blocks:
        assert area(bs) == pytest.approx(80.0 * 60.0, rel=1e-12)

    # The rotated pair sits within the rotation's own reach of the unrotated one.
    for a, b in zip(bbox(blocks[0]), bbox(blocks[1])):
        assert abs(a - b) < 1e-4, (bbox(blocks[0]), bbox(blocks[1]))


def test_analytic_angle_is_smooth_through_normal():
    """The smooth path is the one that can fit a sidewall angle near normal; the vector one cannot.

    Rotating a rectangle moves its Fourier coefficients analytically, so the efficiency is
    smooth in ``angle`` and the two one-sided slopes at zero must agree. They do, on the
    smooth path, and that is all this pins.

    The block path is not held to it. It is a staircase at any angle but zero, and at zero it
    stops being one — the two triangles appear the moment the angle leaves it — so the two
    sides of zero are not built from the same number of blocks and its one-sided slopes are
    free to differ. A central difference hides that, so it only bites an optimizer, which sees
    one side at a time.
    """
    base = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[5, 5], top=1.0, bot=1.0,
                wavelength=900.0, psi=np.pi / 4, theta=0.1, phi=0.3)

    def analytic(angle):
        shape = dict(type='rectangle', cx=150.0, cy=150.0, lx=80.0, ly=60.0,
                     angle=angle, material=4.0)
        res = wavebit.rcwa(backend=0, structure=[dict(background=12.25, shapes=[shape])],
                             **base).solve()
        return float(np.sum(res.T))

    h = 1e-5
    f0, fp, fm = analytic(0.0), analytic(h), analytic(-h)
    left, right = (f0 - fm) / h, (fp - f0) / h
    assert abs(right - left) < 0.02 * abs(left), (left, right)


def test_translation_leaves_nothing_behind_within_one_normal_field_cell():
    """Sliding a single layer along x cannot change the efficiency, so d/dcx is exactly zero.

    The sub-grid-cell sweep is the historical shape of this test: the normal field used to be
    sampled at the shape's place, so its cell size set a noise floor under every shape gradient
    on this path — 5e-4 at ``normal_grid=256``, which is why the default rose to 1024.
    A lone shape's field is now translated by a phase instead
    (:func:`wavebit._core.analytic.normal_projector_coeffs`) and the floor is gone: this reads
    5.8e-13, and ``test_shape_position_gradients`` is where the full statement lives. The bound
    stays loose here because the sweep no longer probes anything that varies.
    """
    base = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[5, 5], top=1.0, bot=1.0,
                wavelength=900.0, psi=np.pi / 4, theta=0.1, phi=0.3)
    solver = wavebit.rcwa(backend=0, period=[300.0, 300.0])
    cell = 300.0 / solver.normal_grid

    def total(cx):
        shape = dict(type='rectangle', cx=cx, cy=150.0, lx=80.0, ly=60.0,
                     angle=0.4, material=4.0)
        return float(np.sum(wavebit.rcwa(backend=0, structure=[dict(background=12.25, shapes=[shape])],
                                           **base).solve().T))

    values = [total(150.0 + f * cell) for f in (0.05, 0.35, 0.65, 0.95)]
    assert np.ptp(values) < 5e-5, np.ptp(values)
