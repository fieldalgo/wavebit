"""``wall_resolution`` under derivatives, ``jax.jit`` and ``jax.vmap``, with the geometry as numbers.

What is pinned: material, wavelength and angle derivatives against central differences on both
coordinate systems (rectangles, ``Stretch``; a circle, ``Matched``), on JAX and PyTorch; a jitted or
vmapped ``Prepared.run`` against the eager run; the touching-rectangles rule under a trace; and the
refusal of every traced or differentiated geometry by its path. The measurements behind the
criteria are in ``docs/validation/rcwa/wall-resolution.md`` ("Derivatives and JAX transforms").
"""
import dataclasses

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import UnsupportedFeatureError

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
jnp = jax.numpy

P, H = 400.0, 150.0
METAL = -20 + 1.5j


def _problem(shape='P1', eps=METAL, wavelength=900.0, theta=0.35, size=200.0, center=0.0,
             radius=110.0, period=P, bottom=None):
    geometry = (rcwa.Rectangle(size=(size, 200.0), center=(center, 0.0)) if shape == 'P1'
                else rcwa.Circle(radius=radius, center=(center, 0.0)))
    media = {} if bottom is None else {'bottom': Material(epsilon=bottom)}
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(period, P), layers=[rcwa.Layer(thickness=H, regions=[
            rcwa.Region(geometry=geometry, material=Material(epsilon=eps))])], **media),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=[rcwa.PlaneWave(
            side='top', polarization='p', theta=theta)]))


def _numerics(orders=(3, 3)):
    return rcwa.Numerics(orders=orders, factorization='li', wall_resolution=0.02)


def _vary(shape, param, x):
    if param == 'eps':
        return _problem(shape, eps=x + 1j * METAL.imag)
    if param == 'wavelength':
        return _problem(shape, wavelength=x)
    return _problem(shape, theta=x)


START = {'eps': METAL.real, 'wavelength': 900.0, 'theta': 0.35}
STEP = {'eps': 2e-4, 'wavelength': 9e-3, 'theta': 1e-5}


def _central(shape, param, numerics):
    x, h = START[param], STEP[param]
    value = lambda t: float(np.sum(np.asarray(rcwa.run(_vary(shape, param, t), numerics=numerics).R)))
    return (value(x + h) - value(x - h)) / (2 * h)


# --------------------------------------------------------------------------- derivatives


@pytest.mark.parametrize('param', ['wavelength', 'theta'])
def test_an_incidence_derivative_turns_the_kept_half_space_modes(param):
    """The half-space keeps its evanescent modes and replaces the propagating ones; under a
    derivative the kept modes' span turns, and that turn rides on the modal generator's entries
    across the two sets. Without it this read 0.39% (theta) and 0.11% (wavelength) off the
    central difference, on JAX and PyTorch alike (2026-11-28); measured 1.6e-8 and 1.2e-8."""
    torch = pytest.importorskip('torch')
    numerics = _numerics()
    x = torch.tensor(START[param], dtype=torch.float64, requires_grad=True)
    total = rcwa.run(_vary('P1', param, x), numerics=numerics, backend='torch').R.sum()
    (gradient,) = torch.autograd.grad(total, x)
    central = _central('P1', param, numerics)
    assert abs(gradient.item() - central) <= 1e-6 * abs(central)


@pytest.mark.slow
@pytest.mark.parametrize('shape', ['P1', 'C1'])
@pytest.mark.parametrize('param', ['eps', 'wavelength', 'theta'])
def test_jax_and_torch_gradients_match_central_differences(shape, param):
    torch = pytest.importorskip('torch')
    numerics = _numerics()
    central = _central(shape, param, numerics)
    run = lambda t: jnp.sum(rcwa.run(_vary(shape, param, t), numerics=numerics, backend='jax').R)
    from_jax = float(jax.grad(run)(START[param]))
    x = torch.tensor(START[param], dtype=torch.float64, requires_grad=True)
    total = rcwa.run(_vary(shape, param, x), numerics=numerics, backend='torch').R.sum()
    (from_torch,) = torch.autograd.grad(total, x)
    # Measured at most 7.9e-8 of the difference at this order (docs/validation, criterion K1).
    assert abs(from_jax - central) <= 1e-6 * abs(central)
    assert abs(from_torch.item() - central) <= 1e-6 * abs(central)


def test_a_substrate_permittivity_derivative_is_taken_eagerly():
    """The half-space's own modes take its permittivity: an eager derivative runs through them.
    Measured 3.7e-9 off the central difference."""
    numerics = _numerics()
    total = lambda b: jnp.sum(rcwa.run(_problem(bottom=b), numerics=numerics, backend='jax').T)
    central = (float(total(2.25 + 1e-4)) - float(total(2.25 - 1e-4))) / 2e-4
    assert abs(float(jax.grad(total)(2.25)) - central) <= 1e-6 * abs(central)


# --------------------------------------------------------------------------- jit and vmap


@pytest.mark.slow
@pytest.mark.parametrize('shape', ['P1', 'C1'])
def test_jit_and_vmap_answer_what_the_eager_run_answers(shape):
    numerics = _numerics(orders=(2, 2))
    prepared = rcwa.prepare(_problem(shape), numerics=numerics, backend='jax')

    def powers(wavelength, eps):
        result = prepared.run(_problem(shape, eps=eps + 1j * METAL.imag, wavelength=wavelength))
        return jnp.concatenate([jnp.ravel(result.R), jnp.ravel(result.T)])

    points = [(900.0, -20.0), (850.0, -18.0), (950.0, -22.0)]
    eager = np.stack([np.asarray(powers(*p)) for p in points])
    jitted = jax.jit(powers)
    compiled = np.stack([np.asarray(jitted(*p)) for p in points])
    mapped = np.asarray(jax.vmap(powers)(*(jnp.asarray(c) for c in zip(*points))))
    scale = np.maximum(1.0, np.abs(eager))
    # Measured below 1e-13 (criteria K2, K3).
    assert np.all(np.abs(compiled - eager) <= 1e-12 * scale)
    assert np.all(np.abs(mapped - eager) <= 1e-12 * scale)
    angle = lambda t: jnp.sum(prepared.run(_problem(shape, theta=t)).R)
    assert float(jax.jit(jax.grad(angle))(0.35)) == pytest.approx(float(jax.grad(angle)(0.35)),
                                                                 rel=1e-10)


def _halves(first, second):
    regions = [rcwa.Region(geometry=rcwa.Rectangle(size=(100.0, 200.0), center=(x, 0.0)),
                           material=material)
               for x, material in ((-50.0, first), (50.0, second))]
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(P, P), layers=[rcwa.Layer(thickness=H, regions=regions)]),
        excitation=rcwa.Excitation(wavelength=900.0, sources=[rcwa.PlaneWave(
            side='top', polarization='p', theta=0.35)]))


@pytest.mark.slow
def test_touching_rectangles_of_one_traced_value_make_no_wall_under_jit():
    """Eagerly two touching rectangles of one material make no wall. Under jit the values are
    unknown: one traced value written for both is one material (identity), two traced values are
    two, and where those turn out equal the run answers NaN rather than a solve in coordinates
    the eager run would not take."""
    prepared = rcwa.prepare(_halves(Material(epsilon=METAL), Material(epsilon=METAL)),
                            numerics=_numerics(orders=(2, 2)), backend='jax')
    eager = float(jnp.sum(prepared.run(_halves(Material(epsilon=METAL), Material(epsilon=METAL))).R))

    def shared(e):
        material = Material(epsilon=e + 1j * METAL.imag)
        return jnp.sum(prepared.run(_halves(material, material)).R)

    assert float(jax.jit(shared)(METAL.real)) == pytest.approx(eager, rel=1e-12)

    def apart(a, b):
        return prepared.run(_halves(Material(epsilon=a + 1j * METAL.imag),
                                    Material(epsilon=b + 1j * METAL.imag)))

    equal = jax.jit(apart)(METAL.real, METAL.real)
    assert np.all(np.isnan(np.asarray(equal.R)))
    assert not bool(equal.validity.checks['numerical_finite'])
    unequal = float(jnp.sum(jax.jit(apart)(METAL.real, -19.0).R))
    reference = float(jnp.sum(prepared.run(_halves(Material(epsilon=METAL),
                                                   Material(epsilon=-19 + 1.5j))).R))
    assert unequal == pytest.approx(reference, rel=1e-12)


def test_the_report_claims_the_transforms_on_jax():
    report = rcwa.check(_problem(), numerics=_numerics(), backend='jax')
    assert report.capabilities['jit'] and report.capabilities['vmap']
    assert report.capabilities['material_gradients']


# --------------------------------------------------------------------------- refusals


GEOMETRY = [('P1', 'size', 'stack.layers[0].regions[0].geometry.size'),
            ('P1', 'center', 'stack.layers[0].regions[0].geometry.center'),
            ('C1', 'radius', 'stack.layers[0].regions[0].geometry.radius'),
            ('C1', 'period', 'stack.cell')]


@pytest.mark.parametrize('shape, name, path', GEOMETRY)
@pytest.mark.parametrize('transform', ['grad', 'jit', 'vmap', 'torch'])
def test_a_traced_geometry_is_refused_by_its_path(shape, name, path, transform):
    value = {'size': 200.0, 'center': 10.0, 'radius': 110.0, 'period': P}[name]
    numerics = _numerics(orders=(2, 2))
    with pytest.raises(UnsupportedFeatureError) as caught:
        if transform == 'torch':
            torch = pytest.importorskip('torch')
            leaf = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            rcwa.run(_problem(shape, **{name: leaf}), numerics=numerics, backend='torch')
        elif transform == 'grad':
            jax.grad(lambda t: jnp.sum(rcwa.run(_problem(shape, **{name: t}), numerics=numerics,
                                                backend='jax').R))(value)
        else:
            prepared = rcwa.prepare(_problem(shape), numerics=numerics, backend='jax')
            run = lambda t: jnp.sum(prepared.run(_problem(shape, **{name: t})).R)
            (jax.jit(run)(value) if transform == 'jit'
             else jax.vmap(run)(jnp.asarray([value, value + 1.0])))
    assert caught.value.code == 'rcwa.unsupported' and caught.value.path == path
    assert 'wall_resolution builds its coordinates from the walls as numbers' in str(caught.value)


def test_a_traced_half_space_is_refused_under_jit():
    numerics = _numerics(orders=(2, 2))
    prepared = rcwa.prepare(_problem(bottom=2.25), numerics=numerics, backend='jax')
    with pytest.raises(UnsupportedFeatureError, match="bottom half-space's permittivity is traced"):
        jax.jit(lambda b: jnp.sum(prepared.run(_problem(bottom=b)).R))(2.25)


@pytest.mark.slow
@pytest.mark.parametrize('shape', ['P1', 'C1'])
def test_a_field_takes_a_material_derivative_eagerly_and_is_refused_under_jit(shape):
    """The circle's field reading took its materials through ``complex()`` until 2026-11-28, which
    a trace refuses and which dropped their derivative. Measured 2.7e-8 (block), 2.8e-8 (circle)."""
    numerics = _numerics(orders=(2, 2))
    with_fields = lambda e: dataclasses.replace(_problem(shape, eps=e), outputs=rcwa.Outputs(fields=True))
    prepared = rcwa.prepare(with_fields(METAL), numerics=numerics, backend='jax')
    x = y = np.linspace(-150.0, 150.0, 3)

    def energy(e):
        field = prepared.run(with_fields(e + 1j * METAL.imag)).field(x=x, y=y, z=[75.0])
        return jnp.sum(jnp.abs(field.values) ** 2)

    central = (float(energy(METAL.real + 2e-3)) - float(energy(METAL.real - 2e-3))) / 4e-3
    assert abs(float(jax.grad(energy)(METAL.real)) - central) <= 1e-6 * abs(central)
    with pytest.raises(UnsupportedFeatureError, match='reads its sample points as numbers'):
        jax.jit(energy)(METAL.real)
