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

SHAPE = dict(type='rectangle', cx=0.0, cy=0.0, lx=0.385, ly=0.231, angle=0.4,
             material=12.25)                            # 3.5**2
ANALYTIC_BASE = dict(period=[0.77, 0.77], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
                     wavelength=0.777, top=1.0, bot=2.25, truncation_order=[4, 4])

def _frozen_raster():
    global FROZEN
    if FROZEN is None:
        FROZEN = wavebit.rcwa(backend='numpy', structure=RASTER, **BASE).freeze_geometry()
    return FROZEN


FROZEN = None

def _backend(name):
    pytest.importorskip(name)
    if name == 'jax':
        wavebit.enable_x64()
        import jax.numpy as jnp
        return 'jax', (lambda v: jnp.asarray(v, dtype=jnp.float64))
    import torch
    return 'torch', (lambda v: torch.tensor(v, dtype=torch.float64))


def _derivative(backend, objective, value, rel_step=1e-6):
    """(autograd, central difference) of ``objective`` at ``value``.

    ``rel_step`` matters more than it looks on the parametric path: the normal field is sampled
    on a grid of ``period / shape_nvf_grid_size``, so the objective carries a sawtooth of that
    period. A difference step that straddles one of those steps reports the jump, not the
    derivative — at ``cx = period/2``, which sits exactly on a grid line, ``h = 1e-6`` gives
    3.8 where the derivative is 3.2e-3. Stay well inside one cell.
    """
    if backend == 'torch':
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
    if backend == 'torch':
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
    """Both Fourier strategies preserve the independent raster design variables.

    Integrated compression used to leave only 9 of these 64 gradients nonzero. Equality of
    the current values does not tie their independent tangents; the per-pixel finite-difference
    tests in test_integrated_raster_gradients.py establish the derivatives themselves.
    """
    backend, cast = _backend(name)

    def cells(geometry=None, **strategy):
        if backend == 'torch':
            import torch
            uc = torch.tensor(RASTER, dtype=torch.float64, requires_grad=True)
            res = wavebit.rcwa(backend='torch', structure=uc, **strategy,
                                 **BASE).solve(geometry=geometry)
            torch.sum(res.T).backward()
            return int((uc.grad.abs() > 1e-14).sum())
        import jax, jax.numpy as jnp
        g = jax.grad(lambda u: jnp.sum(wavebit.rcwa(backend='jax', structure=u, **strategy,
                     **BASE).solve(geometry=geometry).T).real)
        return int(np.sum(np.abs(np.asarray(g(jnp.asarray(RASTER)))) > 1e-14))

    # No strategy keywords at all: this is the *default* that has to carry every cell.
    assert cells() == RASTER.size

    integrated = dict(fourier_coefficients='integrated')
    solver = wavebit.rcwa(backend=backend, structure=cast(RASTER), **integrated, **BASE)
    assert cells(solver.freeze_geometry(), **integrated) == RASTER.size


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

# ------------------------------------------------------------------- shape gradients near zero

def test_analytic_angle_is_smooth_through_normal():
    """The shape path fits a sidewall angle through normal, and the slopes agree at zero.

    Rotating a rectangle moves its Fourier coefficients analytically, so the efficiency is
    smooth in ``angle`` and the two one-sided slopes at zero must agree.

    The block path could not be held to this, which is part of why it is gone: it was a
    staircase at every angle but zero, and at zero it stopped being one -- the two triangles
    appeared the moment the angle left it -- so the two sides of zero were built from different
    numbers of blocks and its one-sided slopes were free to differ. A central difference hides
    that, so it only bit an optimizer, which sees one side at a time.
    """
    base = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[5, 5], top=1.0, bot=1.0,
                wavelength=900.0, psi=np.pi / 4, theta=0.1, phi=0.3)

    def analytic(angle):
        shape = dict(type='rectangle', cx=0.0, cy=0.0, lx=80.0, ly=60.0,
                     angle=angle, material=4.0)
        res = wavebit.rcwa(backend='numpy', structure=[dict(background=12.25, shapes=[shape])],
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
    on this path — 5e-4 at ``shape_nvf_grid_size=256``, which is why the default rose to 1024.
    A lone shape's field is now translated by a phase instead
    (:func:`wavebit._core.parametric.normal_projector_coeffs`) and the floor is gone: this reads
    5.8e-13, and ``test_shape_position_gradients`` is where the full statement lives. The bound
    stays loose here because the sweep no longer probes anything that varies.
    """
    base = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[5, 5], top=1.0, bot=1.0,
                wavelength=900.0, psi=np.pi / 4, theta=0.1, phi=0.3)
    solver = wavebit.rcwa(backend='numpy', period=[300.0, 300.0])
    cell = 300.0 / solver.shape_nvf_grid_size

    def total(cx):
        shape = dict(type='rectangle', cx=cx, cy=0.0, lx=80.0, ly=60.0,
                     angle=0.4, material=4.0)
        return float(np.sum(wavebit.rcwa(backend='numpy', structure=[dict(background=12.25, shapes=[shape])],
                                           **base).solve().T))

    values = [total(f * cell) for f in (0.05, 0.35, 0.65, 0.95)]
    assert np.ptp(values) < 5e-5, np.ptp(values)
