"""Independent material tangents, two ports and two-face fields of uniform stacks."""
import json
from pathlib import Path

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.units import Z0

REFERENCE = json.loads((Path(__file__).parents[1] / 'fixtures/rcwa/uniform_interfaces/values.json').read_text(encoding='utf-8'))


def problem(eps, *, side='bottom', pol='s', thickness=20000.):
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(700.,700.),
                         bottom=Material(9. if side == 'bottom' else 1.),
                         top=Material(9. if side == 'top' else 1.),
                         layers=(rcwa.Layer(thickness=thickness, background=Material(eps)),)),
        excitation=rcwa.Excitation(wavelength=1000., sources=(
            rcwa.PlaneWave(theta=.9+.2j, side=side, polarization=pol),)),
        outputs=rcwa.Outputs(fields=True))


@pytest.mark.parametrize('algorithm', ['etm', 'smm'])
@pytest.mark.parametrize('side', ['bottom', 'top'])
@pytest.mark.parametrize('pol', ['s', 'p'])
def test_independent_slab_values_fields_and_tangents(algorithm, side, pol):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    with jax.enable_x64(True):
        prepared = rcwa.prepare(problem(9., side=side, pol=pol), backend='jax',
                                numerics=rcwa.Numerics(orders=(2,0), algorithm=algorithm))
        z = np.asarray(REFERENCE['z'])
        def run(e):
            result = prepared.run(problem(e, side=side, pol=pol))
            assert result.diagnostics['half_space_merge'][0] is None
            r = getattr(result.reflection, 'amplitude_'+pol).ravel()[2]
            t = getattr(result.transmission, 'amplitude_'+pol).ravel()[2]
            field = result.field(x=[0.], y=[0.], z=z if side == 'bottom' else 20000-z).values.reshape((-1,6))
            if pol == 's':
                f = jnp.stack((field[:,1], field[:,3]*Z0*(1 if side=='bottom' else -1), field[:,5]*Z0), axis=-1)
            else:
                # Public p_hat gives negative H_y for upward propagation.
                # The oracle uses positive magnetic primary amplitude.
                f = -jnp.stack((field[:,0], field[:,2]*(1 if side=='bottom' else -1), field[:,4]*Z0*(1 if side=='bottom' else -1)), axis=-1)
            return jnp.concatenate((jnp.asarray([r,t]), f.ravel()))
        pair = jax.jit(lambda e: jax.jvp(run, (e,), (1.,)))
        for row in REFERENCE['rows']:
            if row['polarization'] != pol:
                continue
            value, tangent = pair(row['epsilon'])
            for got, key in ((value,'values'), (tangent,'derivatives')):
                a = np.asarray(row[key]); expected = a[:,0]+1j*a[:,1]
                np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize('algorithm', ['etm', 'smm'])
@pytest.mark.parametrize('side', ['bottom', 'top'])
@pytest.mark.parametrize('pol', ['s', 'p'])
def test_two_layers_keep_fields_power_and_both_scattering_ports(algorithm, side, pol):
    jax = pytest.importorskip('jax')
    with jax.enable_x64(True):
        def scene(e):
            p = problem(e, side=side, pol=pol, thickness=150.)
            return p.replace(
                stack=p.stack.replace(bottom=Material(2.25), top=Material(1.), layers=(
                    rcwa.Layer(thickness=150., background=Material(e)),
                    rcwa.Layer(thickness=210., background=Material(3.2)))),
                excitation=rcwa.Excitation(wavelength=1000., sources=(
                    rcwa.PlaneWave(theta=.3, phi=.2, side=side, polarization=pol),)),
                outputs=rcwa.Outputs(fields=True, scattering=(algorithm == 'smm')))
        numerics = rcwa.Numerics(orders=(1,1), algorithm=algorithm)
        prepared = rcwa.prepare(scene(4.), numerics=numerics, backend='jax')
        def select(r):
            arrays = (r.reflection.amplitude_s, r.reflection.amplitude_p,
                      r.transmission.amplitude_s, r.transmission.amplitude_p,
                      r.field(x=[0.], y=[0.], z=[10.,149.999999,150.000001,350.]).values)
            if algorithm == 'smm':
                arrays += tuple(getattr(r.scattering[0], key) for key in ('S11','S12','S21','S22'))
            return arrays
        expected = select(rcwa.run(scene(4.), numerics=numerics, backend='numpy'))
        got = jax.jit(lambda e: select(prepared.run(scene(e))))(4.)
        for a, b in zip(got, expected):
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-13)
        # Tangential E/H continuity at the internal material interface.
        field = np.asarray(got[4]).reshape((-1,6))
        np.testing.assert_allclose(field[1,[0,1,3,4]], field[2,[0,1,3,4]], rtol=1e-7, atol=1e-10)


def test_independent_material_reverse_derivative_and_batch():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    with jax.enable_x64(True):
        prepared = rcwa.prepare(problem(9.), numerics=rcwa.Numerics(orders=(2,0)), backend='jax')
        def run(e):
            return prepared.run(problem(e)).reflection.amplitude_s.ravel()[2]
        rows = REFERENCE['rows'][:5]
        eps = jnp.asarray([r['epsilon'] for r in rows])
        value = jax.jit(jax.vmap(run))(eps)
        expected = np.array([complex(*r['values'][0]) for r in rows])
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=0)
        grad = jax.jit(jax.grad(lambda e: jnp.real(run(e)/1e52)))(9.)
        np.testing.assert_allclose(grad, rows[0]['derivatives'][0][0]/1e52, rtol=1e-12)


@pytest.mark.parametrize('pol', ['s','p'])
def test_torch_independent_material_keeps_its_tangent(pol):
    torch = pytest.importorskip('torch')
    row = next(r for r in REFERENCE['rows'] if r['polarization'] == pol)
    e = torch.tensor(9., dtype=torch.float64, requires_grad=True)
    result = rcwa.run(problem(e, pol=pol), numerics=rcwa.Numerics(orders=(2,0)), backend='torch')
    r = getattr(result.reflection, 'amplitude_'+pol).ravel()[2]
    t = getattr(result.transmission, 'amplitude_'+pol).ravel()[2]
    np.testing.assert_allclose([r.detach().numpy(),t.detach().numpy()],
                               [complex(*v) for v in row['values'][:2]], rtol=1e-12, atol=0)
    grad, = torch.autograd.grad(r.real/1e52, e)
    np.testing.assert_allclose(grad.detach().numpy(), row['derivatives'][0][0]/1e52, rtol=1e-12)
