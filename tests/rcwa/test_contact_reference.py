"""Validate the independent thin-gap instrument before using its derivative."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import array_api_compat.numpy as xp
import pytest
from wavebit.rcwa._engine import factorization as F
from wavebit.lattice import as_lattice
from tests.fixtures.rcwa.contact_instruments import rectangular_li


@pytest.mark.parametrize('height', [350.,525.,700.])
def test_nonuniform_li_matches_a_commensurate_raster(height):
    y,x = np.meshgrid((np.arange(16)+.5)*700/16,(np.arange(16)+.5)*700/16,indexing='ij')
    eps = np.where((abs(x-350)<175)&(abs(y-350)<height/2),12.25,1.)[None]
    reference=F.to_conv_mat_raster_integrated(eps,3,3)
    got=rectangular_li(xp,[dict(type='rectangle',lx=350.,ly=height,material=12.25)],
                       1.,3,3,as_lattice(period=(700.,700.)),xp.complex128)
    for value,want in zip(got[:3],reference[:3]):
        np.testing.assert_allclose(value,want[0],rtol=0,atol=2e-12)


def test_reference_retains_a_subpixel_gap_and_its_derivative(monkeypatch):
    from wavebit.rcwa._engine import parametric as P
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    monkeypatch.setattr(P,'to_conv_mat_parametric_nvf',rectangular_li)
    def solver(height,backend):
        return _legacy_wavebit.rcwa(backend=backend,period=(700.,700.),wavelength=900.,top=1.,bot=2.25,
                           theta=.2,phi=0.,psi=0.,thickness=(300.,),truncation_order=(3,3),
                           structure=[dict(background=1.,shapes=[dict(type='rectangle',cx=0.,
                           cy=0.,lx=350.,ly=height,angle=0.,material=12.25)])])
    center=700.-1e-4
    geometry=solver(center,'numpy').freeze_geometry()
    def loss(height):
        return jnp.sum(solver(height,'jax').solve(geometry=geometry).T).real
    derivative=float(jax.jit(jax.grad(loss))(jnp.asarray(center)))
    assert np.isfinite(derivative) and abs(derivative)>1e-4
    for h in (1e-5,1e-6):
        fd=(float(np.sum(solver(center+h,'numpy').solve().T))-float(np.sum(solver(center-h,'numpy').solve().T)))/(2*h)
        assert derivative==pytest.approx(fd,rel=2e-4,abs=1e-7)


def test_partial_derivatives_reconstruct_the_live_nvf_gradient(monkeypatch):
    """Attribute sensitivity without substituting a frozen-normal gradient."""
    from wavebit.rcwa._engine import parametric as P
    from tests.fixtures.rcwa.contact_instruments import split_nvf_derivative
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    center = 700. - 1e-4

    def solver(height, backend):
        return _legacy_wavebit.rcwa(
            backend=backend, period=(700., 700.), wavelength=900., top=1., bot=2.25,
            theta=.2, phi=0., psi=0., thickness=(300.,), truncation_order=(3, 3),
            shape_nvf_grid_size=256, structure=[dict(background=1., shapes=[dict(
                type='rectangle', cx=0., cy=0., lx=350., ly=height,
                angle=0., material=12.25)])])

    geometry = solver(center, 'numpy').freeze_geometry()
    original = P.to_conv_mat_parametric_nvf
    values, derivatives = {}, {}
    try:
        for component in ('full', 'material', 'normal'):
            builder = original if component == 'full' else split_nvf_derivative(center, component)
            monkeypatch.setattr(P, 'to_conv_mat_parametric_nvf', builder)
            P._PROJECTOR_CACHE.clear()

            def loss(height):
                return jnp.sum(solver(height, 'jax').solve(geometry=geometry).T).real

            value, derivative = jax.jit(jax.value_and_grad(loss))(jnp.asarray(center))
            values[component], derivatives[component] = float(value), float(derivative)
            for h in (1e-5, 1e-6):
                P._PROJECTOR_CACHE.clear()
                plus = float(np.sum(solver(center+h, 'numpy').solve().T))
                P._PROJECTOR_CACHE.clear()
                minus = float(np.sum(solver(center-h, 'numpy').solve().T))
                # The wider secant has visible curvature; the narrower one must
                # agree more tightly. No topology switch is crossed by either.
                tolerance = 3e-3 if h == 1e-5 else 4e-5
                assert derivatives[component] == pytest.approx(
                    (plus-minus)/(2*h), rel=tolerance, abs=2e-8)
        assert values['material'] == pytest.approx(values['full'], rel=0, abs=2e-12)
        assert values['normal'] == pytest.approx(values['full'], rel=0, abs=2e-12)
        assert derivatives['material'] + derivatives['normal'] == pytest.approx(
            derivatives['full'], rel=2e-10, abs=2e-10)
    finally:
        P._PROJECTOR_CACHE.clear()
