"""``reconstruct_field(discontinuous='continuity')`` on the 1D TM and conical paths.

Inside a patterned layer ``E_x`` jumps at every wall (``eps_x E_x`` is what is continuous), so the
Fourier sum of its coefficients rings at any truncation. The continuity rule sums the continuous
``D_x`` series instead and divides by ``eps(x)`` at each sample -- the near-field rule of
P. Lalanne and M. P. Jurek, J. Mod. Opt. 45, 1357 (1998), which RETICOLO's ``res3`` applies.

The RETICOLO goldens pin the rule to 1e-12 on gratings
(``tests/references/test_reticolo_field_golden.py``, ``..._conical_field_golden.py``). This file
holds what they cannot: a label-free check that the reconstructed ``eps E_x`` is continuous across
every wall, on both incidence sides (the goldens are lit from the top only, and the layer order
the rule reads eps in is the one thing that differs between sides); the layers and components the
rule must leave alone; the backends; and the refusals.
"""
import numpy as np
import pytest

from wavebit.rcwa._factory import rcwa

PERIOD, NX = 400.0, 40


def _bar(lo, hi, eps_bg, eps_hi):
    row = np.full(NX, eps_bg, dtype=complex)
    row[int(lo / 10):int(hi / 10)] = eps_hi
    return row


# Two patterned layers whose walls sit at different x, over a substrate: a layer read with the
# other layer's permittivity puts a wall where there is none and misses one where there is.
STRUCTURE = np.stack([_bar(50, 230, 1.0, 4.0), _bar(130, 330, 1.69, 6.25)])[:, None, :]
THICKNESS = [150.0, 260.0]
WALLS = {0: (50.0, 230.0), 1: (130.0, 330.0)}
PATHS = {'tm': dict(tetm='tm', phi=None), 'conical': dict(phi=0.4, psi=0.3)}


def _solver(path, side='top', backend='numpy', structure=STRUCTURE, **kw):
    s = rcwa(backend=backend, top=1.0, bot=2.25, period=[PERIOD], truncation_order=12,
             wavelength=620.0, thickness=THICKNESS, structure=structure, theta=0.3,
             fourier_coefficients='integrated', answers='field', incidence_side=side,
             **PATHS[path], **kw)
    s.solve()
    return s


def _ex(cell, path):
    """``E_x`` out of a cell: the pure TM layout is ``(Hy, Ex, Ez)``, the conical one six-wide."""
    cell = np.asarray(cell)
    return cell[..., 1] if path == 'tm' else cell[0, ..., 0]


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('path', list(PATHS))
def test_normal_d_is_continuous_across_every_wall(path, side):
    """``eps E_x`` a hair either side of each wall agrees, in both layers, lit from either side.

    The plain Fourier sum fails this by the Gibbs overshoot, and the rule reading the wrong
    layer's eps fails it by 0.75 (measured by swapping the layer order in
    ``_continuity_rows``); the residual here is the field's slope over the 2e-6 offset.
    """
    s = _solver(path, side)
    for layer, (z0, d) in enumerate(((0.0, 150.0), (150.0, 260.0))):
        z = np.array([z0 + 0.37 * d])
        for wall in WALLS[layer]:
            x = np.array([wall - 1e-6, wall + 1e-6]) - PERIOD / 2
            ex = _ex(s.reconstruct_field(sample_x=x, sample_y=np.array([0.0]), sample_z=z,
                                         discontinuous='continuity'), path).ravel()
            eps = STRUCTURE[layer, 0, [int((wall - 1e-3) / 10), int((wall + 1e-3) / 10)]]
            d_x = eps * ex
            assert abs(d_x[0] - d_x[1]) < 1e-6 * np.abs(d_x).max(), (layer, wall)
            plain = _ex(s.reconstruct_field(sample_x=x, sample_y=np.array([0.0]), sample_z=z),
                        path).ravel()
            assert abs(eps[0] * plain[0] - eps[1] * plain[1]) > 1e-2 * np.abs(d_x).max()


@pytest.mark.parametrize('path', list(PATHS))
def test_only_ex_in_patterned_layers_changes(path):
    """Every other component, and every component of a uniform layer, is the Fourier sum, bit for bit."""
    uniform = np.concatenate([STRUCTURE[:1], np.full((1, 1, NX), 2.25 + 0j)])
    s = _solver(path, structure=uniform)
    a = np.asarray(s.reconstruct_field(sample_x=17, sample_y=1, sample_z=5))
    b = np.asarray(s.reconstruct_field(sample_x=17, sample_y=1, sample_z=5, discontinuous='continuity'))
    ex_slot = 1 if path == 'tm' else 0
    others = [c for c in range(a.shape[-1]) if c != ex_slot]
    np.testing.assert_array_equal(a[..., others], b[..., others])
    upper = slice(5, 10) if path == 'tm' else (slice(None), slice(5, 10))   # the uniform layer
    np.testing.assert_array_equal(a[upper], b[upper])
    assert not np.allclose(a[..., ex_slot], b[..., ex_slot])


def test_te_has_nothing_to_change():
    s = rcwa(backend='numpy', top=1.0, bot=2.25, period=[PERIOD], truncation_order=12,
             wavelength=620.0, thickness=THICKNESS, structure=STRUCTURE, theta=0.3, phi=None,
             tetm='te', fourier_coefficients='integrated', answers='field')
    s.solve()
    np.testing.assert_array_equal(
        np.asarray(s.reconstruct_field(sample_x=17, sample_y=1, sample_z=5)),
        np.asarray(s.reconstruct_field(sample_x=17, sample_y=1, sample_z=5, discontinuous='continuity')))


@pytest.mark.parametrize('path', list(PATHS))
def test_explicit_points_are_the_count_grid(path):
    s = _solver(path)
    x = np.linspace(-PERIOD / 2, PERIOD / 2, 11)
    y = np.array([-PERIOD / 2])       # a count of one samples the cell's edge row, not its centre
    counted = np.asarray(s.reconstruct_field(sample_x=11, sample_y=1, sample_z=4,
                                             discontinuous='continuity'))
    z = np.concatenate([np.linspace(0, 150, 4)[:-1], [150 - 1e-9], np.linspace(150, 410, 4)[:-1],
                        [410 - 1e-9]])
    explicit = np.asarray(s.reconstruct_field(sample_x=x, sample_y=y, sample_z=z,
                                              discontinuous='continuity'))
    np.testing.assert_allclose(_ex(explicit, path), _ex(counted, path), rtol=0, atol=1e-9)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('path', list(PATHS))
def test_backends_agree(path, backend):
    pytest.importorskip(backend)
    want = _ex(_solver(path).reconstruct_field(sample_x=17, sample_y=1, sample_z=5,
                                               discontinuous='continuity'), path)
    got = _solver(path, backend=backend).reconstruct_field(sample_x=17, sample_y=1, sample_z=5,
                                                          discontinuous='continuity')
    got = got.detach().cpu().numpy() if hasattr(got, 'detach') else np.asarray(got)
    np.testing.assert_allclose(_ex(got, path), want, rtol=0, atol=1e-10 * np.abs(want).max())


def test_gradient_through_the_raster_values():
    """``d|E_x|^2 / d eps`` of a raster cell matches a central difference (the division and
    the gather are on the AD graph; the column choice is not, and needs not be)."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def power(scale):
        structure = jnp.asarray(STRUCTURE).at[0, 0, 10].multiply(scale)
        s = _solver('tm', backend='jax', structure=structure)
        ex = s.reconstruct_field(sample_x=13, sample_y=1, sample_z=3, discontinuous='continuity')[..., 1]
        return jnp.sum(jnp.abs(ex) ** 2)

    grad = float(jax.grad(power)(1.0))
    h = 1e-4          # the Fourier sum's own AD-vs-FD gap is the same 1e-7 at this step
    fd = float((power(1.0 + h) - power(1.0 - h)) / (2 * h))
    assert grad == pytest.approx(fd, rel=1e-6)


def test_refusals_name_what_is_missing():
    s = _solver('tm')
    with pytest.raises(ValueError, match="'fourier' or 'continuity'"):
        s.reconstruct_field(discontinuous='li')
    two_d = rcwa(backend='numpy', top=1.0, bot=1.0, period=[PERIOD, PERIOD], truncation_order=[2, 2],
                 wavelength=620.0, thickness=[100.0], structure=np.full((1, 4, 4), 4.0 + 0j).copy(),
                 answers='field')
    two_d.structure[0, 0, 0] = 1.0
    two_d.solve()
    with pytest.raises(NotImplementedError, match='1D gratings'):
        two_d.reconstruct_field(discontinuous='continuity')
    shaped = rcwa(backend='numpy', top=1.0, bot=1.0, period=[PERIOD], truncation_order=4,
                  wavelength=620.0, thickness=[100.0], phi=0.3, psi=0.2, answers='field',
                  structure=[dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0,
                                                                lx=150.0, ly=PERIOD, material=4.0)])])
    shaped.solve()
    with pytest.raises(NotImplementedError, match='1D gratings|raster'):
        shaped.reconstruct_field(discontinuous='continuity')
