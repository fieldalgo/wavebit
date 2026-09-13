"""Material limits of the painted NVF path, without changing the physical geometry."""
import numpy as np
import pytest

import wavebit
from wavebit._core import boundary as B


CELL = ((700., 0.), (0., 700.))
KW = dict(lattice=CELL, wavelength=900., thickness=(300.,), top=1., bot=2.25,
          theta=.2, phi=.17, psi=0., truncation_order=(2, 2), normal_grid=64)


def shapes(left=9., right=9.):
    return [dict(type='rectangle', cx=-150., cy=0., lx=300., ly=300., angle=.3, material=left),
            dict(type='rectangle', cx=150., cy=0., lx=300., ly=300., angle=.3, material=right)]


def layer(left=9., right=9.):
    return [dict(background=1., shapes=shapes(left, right))]


def total(left=9., right=9., **kwargs):
    return float(np.sum(wavebit.rcwa(backend=0, structure=layer(left, right),
                                     **dict(KW, **kwargs)).solve().T))


@pytest.mark.parametrize('base,direction', [(9., 1.), (4.+.4j, 1j), (-4.+.3j, 1.)])
@pytest.mark.parametrize('grid', [64, 127, 128])
def test_material_equality_has_a_ramp_instead_of_a_fixed_step(base, direction, grid):
    zero = total(base, base, normal_grid=grid)
    delta = [abs(total(base+h*direction, base, normal_grid=grid)-zero)
             for h in (1e-5, 1e-7)]
    assert delta[1] < .02*delta[0] + 2e-12
    assert delta[1] < 1e-7


@pytest.mark.parametrize('materials', [(4., 9.), (4.+.2j, 9.+.3j), (12.25, -10.+1j)])
def test_painted_materials_preserve_reflection_and_rotation(materials):
    original = shapes(*materials)
    reflected = [dict(s, cy=-s['cy'], angle=-s['angle']) for s in original]
    angle = .37
    rotated = [dict(s, cx=np.cos(angle)*s['cx']-np.sin(angle)*s['cy'],
                    cy=np.sin(angle)*s['cx']+np.cos(angle)*s['cy'], angle=s['angle']+angle)
               for s in original]
    # A physical rotation also rotates the lattice; reflection may keep the same square.
    lattice = 700.*np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
    values = []
    for shape, kwargs in ((original, {}), (reflected, dict(phi=-KW['phi'])),
                          (rotated, dict(lattice=lattice, phi=KW['phi']+angle))):
        result = wavebit.rcwa(backend=0, structure=[dict(background=1., shapes=shape)],
                              **dict(KW, **kwargs)).solve()
        values.append(float(np.sum(result.T)))
        power = float(np.sum(result.R+result.T))
        if all(np.isreal(m) for m in materials):
            assert power == pytest.approx(1., abs=2e-11)
        else:
            assert 0. <= power <= 1.+2e-11
    np.testing.assert_allclose(values, values[0], rtol=0., atol=2e-10)


def test_complex_painted_projector_is_hermitian_rank_one_and_chunk_independent():
    vertices = [B.shape_vertices(np, s, CELL, 256) for s in shapes(4.+.2j, 9.+.3j)]
    recs, regions, transform, plan = B.plan_layer(shapes(4.+.2j, 9.+.3j), CELL)
    vals = B._clipper.eval_table(np, recs, vertices)
    regs = [(s['material'], [B._clipper.loop_verts(np, loop, vals) for loop in reg])
            for s, reg in zip(shapes(4.+.2j, 9.+.3j), regions)]
    data = B.replay_interfaces(np, regs, 1., CELL, transform, plan)
    contrasts = B.interface_contrasts(np, regs, 1., plan, np.float64)
    a = B.material_projector_fields(np, (31, 37), CELL, np.float64, data, contrasts, plan,
                                    transform, chunk=113)
    b = B.material_projector_fields(np, (31, 37), CELL, np.float64, data, contrasts, plan,
                                    transform, chunk=31*37)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    xx, xy, yy = a
    np.testing.assert_allclose(xx+yy, 1., atol=1e-14)
    assert all(np.isrealobj(v) for v in a)
    np.testing.assert_allclose(xx*yy-xy*xy, 0., atol=1e-14)


@pytest.mark.parametrize('right', [4.+1e-3, 4.+1e-8, 4.+.5j, -4.+1j])
def test_weak_and_complex_interfaces_keep_the_exact_normal(right, monkeypatch):
    shape = shapes(4., right)
    vertices = [B.shape_vertices(np, s, CELL, 256) for s in shape]
    recs, regions, transform, plan = B.plan_layer(shape, CELL)
    vals = B._clipper.eval_table(np, recs, vertices)
    regs = [(s['material'], [B._clipper.loop_verts(np, loop, vals) for loop in reg])
            for s, reg in zip(shape, regions)]
    data = B.replay_interfaces(np, regs, 1., CELL, transform, plan)
    starts, edges, _, active = data
    midpoints = starts[active]+edges[active]/2
    # All samples are in the interior of straight material-interface intervals;
    # their normal dyad is known independently of the off-interface extension.
    normals = edges[active][:, ::-1]*[1., -1.]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    monkeypatch.setattr(B, 'sample_grid', lambda *args: (midpoints[:, 0][None, :],
                                                        midpoints[:, 1][None, :]))
    fields = B.material_projector_fields(
        np, (1, len(midpoints)), CELL, np.float64, data,
        B.interface_contrasts(np, regs, 1., plan, np.float64), plan, transform)
    expected = [normals[:, 0]**2, normals[:, 0]*normals[:, 1], normals[:, 1]**2]
    np.testing.assert_allclose(np.asarray(fields)[:, 0], expected, atol=2e-12, rtol=0.)


def test_jit_and_material_gradient_through_painted_equality():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()
    solver = wavebit.rcwa(backend=1, structure=layer(), **KW)
    geometry = solver.freeze_geometry()
    def response(d):
        return jnp.sum(solver.solve(structure=layer(9.+d, 9.), geometry=geometry).T).real
    value, derivative = jax.jit(jax.value_and_grad(response))(0.)
    assert float(value) == pytest.approx(total(), abs=2e-11)
    secant = (float(response(1e-5))-float(response(-1e-5)))/2e-5
    assert np.isfinite(derivative)
    assert float(derivative) == pytest.approx(secant, rel=2e-5, abs=1e-8)


def test_torch_material_gradient_and_cross_backend_value():
    torch = pytest.importorskip('torch')
    d = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    result = wavebit.rcwa(backend=2, structure=layer(9.+d, 9.), **KW).solve().T.sum()
    result.backward()
    assert float(result.detach()) == pytest.approx(total(), abs=2e-11)
    secant = (total(9.+1e-5)-total(9.-1e-5))/2e-5
    assert float(d.grad) == pytest.approx(secant, rel=2e-5, abs=1e-8)
