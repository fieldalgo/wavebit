"""The inverse rule on a metal: the pole in ``<1/eps>^-1`` and the floor that closes it.

Where ``1/eps`` changes sign across a boundary -- gold beside air or glass -- ``<1/eps>`` has an
eigenvalue that passes the origin at some truncation, and the inverse rule answers with
``1 / Im(1/eps)``, 400 on gold. The NVF tensor then carries a plasmon of the *smeared* interface
that the sharp one does not have: a narrow spurious feature that moves with the truncation, and
through the symmetrized closing also gain, ``R + T`` up to 1.27 on a gold hole array. Found on
2026-09-27 comparing wavebit against fmmax, S4 and RETICOLO on gold nanoholes and nanodisks;
the mechanism and the numbers are in ``_nvf_assemble``'s docstring.

The fix is a floor on the singular values of ``<1/eps>`` at ``1 / (INVERSE_RULE_BOUND * max|eps|)``,
which binds only where the hull of ``1/eps`` passes the origin. What these tests pin:

* on a passive dielectric the floor never binds and the tensor is **bit-identical** to the
  unfloored one, so every dielectric number in this repository still stands; a dielectric layer
  that mixes gain with loss is not covered, and can bind;
* on gold it binds, on a handful of modes, and the gold disk array that returned ``R + T = 1.13``
  at 121 harmonics returns less than it was given;
* every NVF builder -- shapes, painted overlaps, both raster coefficient routes, the circular
  projector -- hands the floor its layer's largest ``|eps|``, whether the metal is a shape or
  the background, so it binds on every one of them;
* on a lossless metal at the pole itself, where ``<1/eps>`` is singular, the floored tensor is
  finite and the unfloored one is not;
* the floored closing's anti-Hermitian part is bounded where it was not -- measured, not proved,
  which is why the bound asserted is the measured one with room, not zero;
* the floor is differentiated exactly, in a shape length and in the metal's permittivity, on
  JAX and Torch alike, in reverse and in forward mode, where it binds on modes that reach the
  far field and through degenerate singular values; ``torch.func``'s grad and jvp transforms,
  which its first-order guard cannot tell from a second derivative, are refused by name where it
  binds (``torch.autograd`` through ``torch.func.vmap`` is exact: ``test_torch_vmap_autograd.py``);
  and a second derivative through it, mixed orders included, is refused where it binds, rather
  than returned without the floor's curvature.

The metal permittivities are Johnson & Christy gold, interpolated linearly in ``n`` and ``k``.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core import factorization as F
from wavebit._core import parametric as P
from wavebit._core.backend import array_namespace
from wavebit._core.factorization import (INVERSE_RULE_BOUND, _floor_singular_values,
                                         _nvf_assemble)
from wavebit._core.lattice import Lattice

AU_852 = -28.443519 + 1.753958j     # Johnson & Christy, 852 nm
AU_807 = -24.635184 + 1.546042j     # 807 nm


def disks(eps_disk, wl):
    """Case B of the comparison: Au disks 140 nm across, 50 nm thick, 450 nm pitch, in n = 1.5."""
    return ([dict(background=2.25, shapes=[dict(type='circle', cx=0.0, cy=0.0, lx=140.0, material=eps_disk)])],
            dict(top=2.25, bot=2.25, theta=0.0, phi=0.0, psi=0.0, wavelength=wl,
                 period=(450.0, 450.0), thickness=(50.0,)))


def holes(eps_film, wl):
    """Case A: a 150 nm Au film with 200 nm air holes on a 500 nm pitch, air above, glass below."""
    return ([dict(background=eps_film, shapes=[dict(type='circle', cx=0.0, cy=0.0, lx=200.0, material=1.0)])],
            dict(top=1.0, bot=2.25, theta=0.0, phi=0.0, psi=0.0, wavelength=wl,
                 period=(500.0, 500.0), thickness=(150.0,)))


def matrices(par, cfg, M):
    """``E``, ``<1/eps>`` and the projector block of the layer, as the shape path builds them."""
    lat = Lattice.rectangular(*cfg['period'])
    lay = par[0]
    eps_c, inv_c = P.assemble_eps_coeffs(np, lay['shapes'], lay['background'], M, M, lat, np.complex128)
    E = P.conv_matrix_2d(np, eps_c, M, M)
    B = P.conv_matrix_2d(np, inv_c, M, M)
    C = tuple(P.conv_matrix_2d(np, c, M, M) for c in P.normal_projector_coeffs(
        np, lay['shapes'], (256, 256), lat, M, M, np.complex128, plan=None))
    return E, B, C


def largest_eps(par):
    lay = par[0]
    return max([abs(lay['background'])] + [abs(s['material']) for s in lay['shapes']])


def solve_with(par, cfg, M, slots):
    s = wavebit.rcwa(backend='numpy', structure=par, truncation_order=(M, M), **cfg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = s.solve_conv(*(m[None] for m in slots))
    return float(np.sum(r.R).real) + float(np.sum(r.T).real)


def anti_hermitian_minimum(slots):
    epx, epy, _, epxy, epyx = slots
    Z = np.block([[epx, -epxy], [-epyx, epy]])
    return float(np.linalg.eigvalsh((Z - Z.conj().T) / 2j).min())


# ------------------------------------------------------------------ where the floor binds

@pytest.mark.parametrize('disk, background', [
    pytest.param(4.0 + 0.3j, 2.25, id='an absorbing disk in glass'),
    pytest.param(0.01 + 4.0j, 4.0 + 0.01j, id='loss angles of 89.9 and 0.1 degrees'),
])
def test_the_floor_never_binds_on_a_passive_dielectric_and_the_tensor_is_bit_identical(disk,
                                                                                        background):
    """A passive ``1/eps`` with ``Re eps > 0`` lies in a sector at most 90 degrees wide, at least
    ``1 / max|eps|`` from the origin, so the hull of its values -- where ``<1/eps>``'s numerical
    range lies -- stays ``cos(45 deg) / max|eps|`` away, above the floor's ``0.5 / max|eps|``. The
    second case is as wide as a passive pair gets: 0.717 at these 81 harmonics, 0.709 at 289.

    Bit-identical, not close: :func:`_floor_singular_values` returns the matrix it was given
    where nothing is below the floor, so a dielectric layer is exactly the tensor it was before
    the floor existed. That is what keeps every dielectric golden in this repository valid.
    """
    par, cfg = disks(disk, 852.0)
    par[0]['background'] = background
    E, B, C = matrices(par, cfg, 4)
    assert np.linalg.svd(B, compute_uv=False).min() * largest_eps(par) >= np.cos(np.pi / 4)
    floored = _nvf_assemble(np, E, B, *C, eps_bound=largest_eps(par))
    bare = _nvf_assemble(np, E, B, *C)
    for a, b in zip(floored, bare):
        assert np.array_equal(a, b)


def test_a_dielectric_layer_mixing_gain_and_loss_is_not_covered():
    """The hull argument needs passive materials. With gain beside loss the values ``1/eps``
    spread over both sides of the real axis, the sector opens towards 180 degrees, and the floor
    can bind with every ``Re eps > 0``: a ``0.2 + 2i`` disk in ``0.2 - 2i`` brings the smallest
    singular value to ``0.212 / max|eps|`` at 121 harmonics, and 6 modes are clipped. Pinned
    because the documentation says *passive* dielectric, and this is why.
    """
    par, cfg = disks(0.2 + 2.0j, 852.0)
    par[0]['background'] = 0.2 - 2.0j
    _, B, _ = matrices(par, cfg, 5)
    s = np.linalg.svd(B, compute_uv=False)
    assert s.min() * largest_eps(par) == pytest.approx(0.212, abs=1e-3)
    assert (s < 1.0 / (INVERSE_RULE_BOUND * largest_eps(par))).sum() == 6


def test_the_floor_binds_on_gold_and_on_a_few_modes_only():
    """On gold ``<1/eps>``'s smallest singular value sits at ``Im(1/eps)``, well under the floor.

    A few modes, not many: the floor is a repair of the pole, not a different rule. Measured
    6 of 441 at 807 nm on the hole array, and 6 of the 169 here. The count is asserted exactly,
    which is what pins ``INVERSE_RULE_BOUND``: a bound of 3 clips 3 of these 169, and a bound
    of 1 clips 137.
    """
    par, cfg = holes(AU_807, 807.0)
    E, B, C = matrices(par, cfg, 6)
    floor = 1.0 / (INVERSE_RULE_BOUND * largest_eps(par))
    s = np.linalg.svd(B, compute_uv=False)
    assert s.min() < floor
    assert s.min() == pytest.approx(abs((1.0 / AU_807).imag), rel=0.5)     # the pole's residue
    assert (s < floor).sum() == 6
    lifted = _floor_singular_values(np, B, floor)
    assert np.linalg.svd(lifted, compute_uv=False).min() == pytest.approx(floor)
    assert not np.array_equal(lifted, B)


# --------------------------------------------------------------- what it does to the solve

def test_gold_disks_return_less_than_they_receive():
    """The array that returned ``R + T = 1.1259`` at 121 harmonics, 852 nm, before the floor.

    Both closings are run on the same matrices through ``solve_conv``, so the only difference
    between the two numbers is the floor. The unfloored one is asserted as well: it is the
    defect this file exists for, and a test that only shows the fixed number would pass on a
    build where the inverse rule had no pole to begin with.
    """
    par, cfg = disks(AU_852, 852.0)
    E, B, C = matrices(par, cfg, 5)
    before = solve_with(par, cfg, 5, _nvf_assemble(np, E, B, *C))
    after = solve_with(par, cfg, 5, _nvf_assemble(np, E, B, *C, eps_bound=largest_eps(par)))
    assert before > 1.1, before
    assert after <= 1.0, after

    # ...and the front door takes the same route.
    res = wavebit.rcwa(backend='numpy', structure=par, truncation_order=(5, 5), **cfg).solve()
    assert float(np.sum(res.R).real + np.sum(res.T).real) == pytest.approx(after, abs=1e-3)


# Two scenes, each built by every NVF route at 121 harmonics. In the disks of the test above the
# gold is a shape in a glass background; in the holes of case A the gold is the background -- a
# film -- and the shapes are air. A builder that read its bound from the shapes alone, or from the
# background alone, would be right on one scene and wrong on the other. The raster is the scene
# drawn on 64 x 64 pixels; the painted overlap is two rectangles of the shape material sharing a
# strip, which the boundary path resolves into one region before it closes the tensor.
_N = 64
SCENES = {'gold disks in glass': (disks, AU_852, 852.0),
          'gold film with air holes': (holes, AU_807, 807.0)}


def _raster(pitch, radius, inside, outside):
    centres = (np.arange(_N) + 0.5) / _N * pitch - pitch / 2
    x, y = np.meshgrid(centres, centres, indexing='xy')
    return np.where(x**2 + y**2 <= radius**2, inside, outside + 0j)[None]


def _halves(dx, lx, ly, material):
    return [dict(type='rectangle', cx=-dx, cy=0.0, lx=lx, ly=ly, material=material),
            dict(type='rectangle', cx=dx, cy=0.0, lx=lx, ly=ly, material=material)]


def _routes(scene, background, circle, halves, raster):
    """Every NVF builder on one scene, as ``(scene, solver keywords)`` cases."""
    def layer(shapes):
        return [dict(background=background, shapes=shapes)]
    return [
        pytest.param(scene, dict(structure=layer(circle)), id=f'{scene}: shape'),
        pytest.param(scene, dict(structure=layer(circle), nvf_projector='circular'),
                     id=f'{scene}: shape, circular projector'),
        pytest.param(scene, dict(structure=layer(halves)), id=f'{scene}: painted overlap'),
        pytest.param(scene, dict(structure=layer(halves), nvf_projector='circular'),
                     id=f'{scene}: painted overlap, circular projector'),
        pytest.param(scene, dict(structure=raster, factorization='nvf'), id=f'{scene}: raster, sampled'),
        pytest.param(scene, dict(structure=raster, factorization='nvf', fourier_coefficients='integrated'),
                     id=f'{scene}: raster, integrated'),
    ]


BUILDERS = (
    _routes('gold disks in glass', 2.25,
            [dict(type='circle', cx=0.0, cy=0.0, lx=140.0, material=AU_852)],
            _halves(30.0, 80.0, 120.0, AU_852), _raster(450.0, 70.0, AU_852, 2.25))
    + _routes('gold film with air holes', AU_807,
              [dict(type='circle', cx=0.0, cy=0.0, lx=200.0, material=1.0)],
              _halves(40.0, 120.0, 160.0, 1.0), _raster(500.0, 100.0, 1.0, AU_807)))


@pytest.mark.parametrize('scene, route', BUILDERS)
def test_every_nvf_builder_floors_its_metal(monkeypatch, scene, route):
    """Each builder computes the bound itself -- the shape path from its materials, the boundary
    path region by region, the raster builders from the raster -- so each one can drop it, or
    take it from the wrong material, without any other builder noticing; and the metal can be a
    shape or the background. Recorded at the floor: called once, with
    ``1 / (INVERSE_RULE_BOUND * |eps_gold|)``, and binding.

    Measured 2026-10-02 on the disks: with ``eps_bound`` dropped, the raster's ``R + T`` goes
    0.860686 -> 1.034581, the gain the floor exists to remove, on either raster route; the
    painted overlap's 0.706320 -> 0.710354, and its circular-projector branch's 0.757487 ->
    0.764659; with the boundary path's bound taken from the background alone, the floor binds on
    9 modes rather than 1 and ``R + T`` reads 0.884471. On the holes, where only the background
    is gold: with the shape path's bound taken from the shapes alone, the floor is 0.5 rather
    than 0.0203 and clips 108 of the 121 modes rather than 4, and ``T`` goes 0.005872 ->
    0.027615; with the boundary path's bound started at 0 rather than at the background's
    ``|eps|``, 109 rather than 2, and the painted overlap's ``R + T`` goes 0.924263 -> 0.955543.
    Neither defect moves the disks.
    """
    calls = []
    floor_alone = F._floor_singular_values

    def recorder(xp, B, floor):
        calls.append((np.linalg.svd(np.asarray(B), compute_uv=False), float(floor)))
        return floor_alone(xp, B, floor)

    monkeypatch.setattr(F, '_floor_singular_values', recorder)
    geometry, metal, wavelength = SCENES[scene]
    _, cfg = geometry(metal, wavelength)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = wavebit.rcwa(backend='numpy', truncation_order=(5, 5), **cfg, **route).solve()
    assert len(calls) == 1
    s, floor = calls[0]
    assert floor == pytest.approx(1.0 / (INVERSE_RULE_BOUND * abs(metal)), rel=1e-12)
    assert (s < floor).any()
    assert float(np.sum(res.R).real + np.sum(res.T).real) <= 1.0


def test_the_anti_hermitian_residual_is_bounded_where_it_was_not():
    """Gain in the tensor is a negative eigenvalue of its anti-Hermitian part.

    Before the floor: -25.4 on these disks, of the order of the pole times ``1/8`` (the Jordan
    product's worst case). After: measured -2.09, the symmetrized closing's own residual, which
    is not zero and is documented as such in ``_nvf_assemble``. Asserted with room -- the point
    is the order of magnitude, and that it no longer scales with the pole.
    """
    par, cfg = disks(AU_852, 852.0)
    E, B, C = matrices(par, cfg, 5)
    assert anti_hermitian_minimum(_nvf_assemble(np, E, B, *C)) < -10.0
    assert anti_hermitian_minimum(_nvf_assemble(np, E, B, *C, eps_bound=largest_eps(par))) > -4.0


# ------------------------------------------------------------------ what it does to a gradient
#
# The floor is the one piece of the closing that is new to every backend at once: an SVD, a
# ``where`` on a scalar condition, and a lift whose derivative is supplied in closed form rather
# than taken through the SVD. These tests are placed where the floor binds *and* the clipped
# modes reach the far field. At truncation order 4 on the disks two modes are clipped, and at
# normal incidence, where symmetry keeps them dark, ``R + T`` does not see them: there, the lift
# held as a detached constant -- which returns the unfloored closing's derivative -- agreed with
# a finite difference to 4e-9, and a check placed there could not have failed. At oblique
# incidence the same two modes are lit and the detached lift is 20% off, which is why one case
# below is oblique.

#: Relative tolerance of an autodiff derivative against the extrapolated central difference.
#: The worst agreement measured on 2026-10-02 in the tests below is 1.2e-8 (the dielectric's
#: diameter derivative), so this leaves 80x. The smallest error of a defect these tests catch is
#: 1.5e-4 -- the floor's own derivative dropped, on the uniform, all-clipped, rotated case --
#: 150x over it.
RELATIVE = 1e-6

#: The oblique incidence of the cases that light what normal incidence leaves dark.
OBLIQUE = dict(theta=0.3, phi=0.4, psi=0.7)


GRADIENT_CASES = [
    pytest.param(disks, AU_852, 852.0, 5, 140.0, id='gold disks, 121 harmonics, 1 mode clipped'),
    pytest.param(holes, AU_807, 807.0, 6, 200.0, id='gold holes, 169 harmonics, 6 modes clipped'),
]


def _scene(geometry, eps, wl, lx, order, angles=None):
    """``geometry``'s layer with its metal (the disk, or the film the holes are cut in) at ``eps``
    and its circle ``lx`` across, at truncation ``order``; at normal incidence unless ``angles``
    says otherwise."""
    par, cfg = geometry(eps, wl)
    par[0]['shapes'][0]['lx'] = lx
    cfg.update(angles or {})
    return dict(structure=par, truncation_order=(order, order), **cfg)


def _power(backend, scene):
    """``R + T`` summed over orders: a float on NumPy, the backend's own scalar on JAX / Torch."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = wavebit.rcwa(backend=backend, **scene).solve()
    if backend == 'numpy':
        return float(np.sum(r.R).real + np.sum(r.T).real)
    return (r.R.sum() + r.T.sum()).real


def _difference(f, x, h):
    """Central difference of ``f`` at ``x``, Richardson-extrapolated in the step: error O(h^4).

    A plain central difference is not enough here. ``R + T`` curves sharply on the disks, and
    at ``h = 1e-3`` its O(h^2) error is 1.2e-4 of the slope -- more than what is checked.
    """
    near = (f(x + h) - f(x - h)) / (2 * h)
    far = (f(x + 2 * h) - f(x - 2 * h)) / (4 * h)
    return (4 * near - far) / 3


def _diameter_derivatives(geometry, eps, wl, order, lx, angles=None):
    """``R + T`` on three backends and its derivative in the circle's diameter on JAX, Torch and
    by finite difference on NumPy."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    want = _power('numpy', _scene(geometry, eps, wl, lx, order, angles))
    value_jax, grad_jax = jax.value_and_grad(
        lambda x: _power('jax', _scene(geometry, eps, wl, x, order, angles)))(jnp.asarray(lx))
    lx_t = torch.tensor(lx, dtype=torch.float64, requires_grad=True)
    value_torch = _power('torch', _scene(geometry, eps, wl, lx_t, order, angles))
    grad_torch, = torch.autograd.grad(value_torch, lx_t)
    finite = _difference(lambda x: _power('numpy', _scene(geometry, eps, wl, x, order, angles)), lx, 1e-3)
    return dict(want=want, value_jax=float(value_jax), value_torch=float(value_torch.detach()),
                jax=float(grad_jax), torch=float(grad_torch), finite=finite)


def _permittivity_derivatives(geometry, eps, wl, order, lx):
    """``d(R + T) / d Re eps`` and ``d / d Im eps`` of the metal, on JAX, Torch and by finite
    difference on NumPy, each as a ``(Re, Im)`` pair."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    a, b = eps.real, eps.imag
    grad_jax = jax.grad(lambda x, y: _power('jax', _scene(geometry, x + 1j * y, wl, lx, order)),
                        argnums=(0, 1))(jnp.asarray(a), jnp.asarray(b))
    a_t = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    grad_torch = torch.autograd.grad(
        _power('torch', _scene(geometry, torch.complex(a_t, b_t), wl, lx, order)), (a_t, b_t))
    power = lambda x, y: _power('numpy', _scene(geometry, complex(x, y), wl, lx, order))
    finite = (_difference(lambda x: power(x, b), a, 1e-4), _difference(lambda y: power(a, y), b, 1e-4))
    return dict(jax=tuple(float(g) for g in grad_jax), torch=tuple(float(g) for g in grad_torch),
                finite=finite)


@pytest.mark.parametrize('geometry, eps, wl, order, lx, angles', [
    pytest.param(*case.values, None, id=case.id) for case in GRADIENT_CASES] + [
    pytest.param(disks, AU_852, 852.0, 4, 140.0, OBLIQUE,
                 id='gold disks, 81 harmonics, 2 modes clipped, oblique'),
    pytest.param(disks, 4.0 + 0.3j, 852.0, 5, 140.0, None, id='lossy dielectric disks, floor inactive')])
def test_the_floor_is_the_same_on_every_backend_and_differentiates_exactly(geometry, eps, wl, order, lx,
                                                                           angles):
    """Forward, the three backends agree to the last digits they share; backward, the derivative
    of ``R + T`` in the diameter matches a central difference where the floor binds as well as
    where it does not.

    Measured 2026-10-01, JAX / Torch against the extrapolated difference: gold disks 2.969526e-04,
    gold holes 3.065110e-04, dielectric -7.436556e-05, relative errors 1.2e-8 or below, and JAX
    against Torch 4e-12 or below. The lift held as a detached constant gave -1.142507e-02 on the
    disks (39x, the wrong sign) and -1.842889e-03 on the holes; holding the clipped modes at the
    floor with the singular vectors frozen -- the lift's magnitude differentiated, its rotation
    not -- gave -7.105029e-03 (24x) and 1.130864e-04 (0.37x), so the rotation of the singular
    vectors is not a correction but most of the answer.

    The oblique case, measured 2026-10-02: -4.314524e-03, where the detached lift gives
    -5.198912e-03, 20% off. The same disks at normal incidence cannot tell the two apart.
    """
    got = _diameter_derivatives(geometry, eps, wl, order, lx, angles)
    assert got['value_jax'] == pytest.approx(got['want'], abs=1e-9)
    assert got['value_torch'] == pytest.approx(got['want'], abs=1e-9)
    for backend in ('jax', 'torch'):
        assert np.isfinite(got[backend]), backend
        assert got[backend] == pytest.approx(got['finite'], rel=RELATIVE), (backend, got)
    assert got['jax'] == pytest.approx(got['torch'], rel=1e-9), got


@pytest.mark.parametrize('geometry, eps, wl, order, lx', GRADIENT_CASES)
def test_a_metal_permittivity_differentiates_exactly_through_the_floor(geometry, eps, wl, order, lx):
    """The floor ``1 / (INVERSE_RULE_BOUND * max|eps|)`` is itself a function of the metal's
    permittivity, and every clipped mode sits on it: a derivative in ``eps`` has to carry the
    floor's own derivative on those modes as well as ``<1/eps>``'s.

    Measured 2026-10-01, JAX and Torch alike, against the extrapolated difference: on the disks
    ``(d/d Re, d/d Im) = (1.073187e-02, -1.061561e-02)``, on the holes
    ``(1.253546e-03, -2.128232e-02)``, relative errors 8e-9 or below. Without the floor's term the
    disks read ``(6.054986e-03, -1.032721e-02)``, 44% off in ``Re``, and the holes
    ``(1.347759e-03, -2.128823e-02)``, 7.5%; the lift held as a detached constant,
    ``(-2.995603e-04, -3.050163e-03)`` and ``(2.658234e-03, -2.145442e-02)``.
    """
    got = _permittivity_derivatives(geometry, eps, wl, order, lx)
    for backend in ('jax', 'torch'):
        for part, value, finite in zip(('Re', 'Im'), got[backend], got['finite']):
            assert np.isfinite(value), (backend, part)
            assert value == pytest.approx(finite, rel=RELATIVE), (backend, part, got)


def _unitary(rng, n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _degenerate_derivative(backend, rotated, values):
    """``d/dt`` of a fixed linear read-out of ``B(t)`` floored at ``floor(t)``, on ``backend`` and
    by central difference on NumPy, with ``B(0) = U diag(values) V^H``. ``backend`` is ``jax`` or
    ``torch`` in reverse mode, or ``jax jvp`` or ``torch forward_ad`` in forward mode."""
    pytest.importorskip(backend.split()[0])
    rng = np.random.default_rng(7)
    n = len(values)
    if rotated:
        U, V = _unitary(rng, n), _unitary(rng, n)
    else:
        U, V = np.diag(np.exp(2j * np.pi * rng.random(n))), np.eye(n)
    B0 = (U * np.asarray(values)) @ V.conj().T
    direction = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    read_out = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

    def loss(t):
        xp = array_namespace(t)
        B = xp.asarray(B0) + t * xp.asarray(direction)
        lifted = _floor_singular_values(xp, B, 0.02 + 0.006 * t)
        return xp.sum(xp.real(xp.conj(xp.asarray(read_out)) * lifted))

    if backend.startswith('jax'):
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        if backend == 'jax':
            got = float(jax.grad(loss)(jnp.asarray(0.0)))
        else:
            got = float(jax.jvp(loss, (jnp.asarray(0.0),), (jnp.asarray(1.0),))[1])
    elif backend == 'torch':
        import torch
        t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        got = float(torch.autograd.grad(loss(t), t)[0])
    else:
        import torch
        from torch.autograd import forward_ad
        with forward_ad.dual_level():
            t = forward_ad.make_dual(torch.tensor(0.0, dtype=torch.float64),
                                     torch.tensor(1.0, dtype=torch.float64))
            got = float(forward_ad.unpack_dual(loss(t)).tangent)
    return got, _difference(lambda t: float(loss(np.asarray(t))), 0.0, 1e-5)


@pytest.mark.parametrize('backend', ['jax', 'jax jvp', 'torch', 'torch forward_ad'])
@pytest.mark.parametrize('rotated', [False, True], ids=['diagonal', 'rotated'])
@pytest.mark.parametrize('values', [
    pytest.param([0.004, 0.004, 0.01, 0.3, 0.3, 0.3, 1.0], id='clipped pair, a third clipped, unclipped triple'),
    pytest.param([0.5] * 6, id='uniform, floor inactive'),
    pytest.param([0.01] * 6, id='uniform, every value clipped'),
])
def test_the_floor_differentiates_through_degenerate_singular_values(backend, rotated, values):
    """Repeated singular values are the ordinary case, not an edge: a uniform layer has nothing
    else, and a two-material layer two clusters of them. The SVD's derivative divides by their
    differences and is NaN there, and a divided difference written naively is 0/0. This floors
    ``B = U diag(values) V^H`` at 0.02 -- exactly degenerate when ``U`` and ``V`` are diagonal
    phases, degenerate to rounding when they are random unitaries -- and moves both ``B`` and the
    floor along a fixed direction ``t``: the derivative of a fixed linear read-out must be finite
    and equal to a central difference of the forward on NumPy, in reverse mode and in forward
    mode alike.

    Measured 2026-10-01: relative error 1.8e-9 or below in every case, JAX and Torch alike, and
    on 2026-10-02 the same in forward mode, where without the PyTorch ``jvp`` every case with a
    clipped value fails. On
    every case with a clipped value the lift held as a detached constant is 18% to 180% off, and
    the floor's own derivative dropped 1.5e-4 to 2e-3. Divided differences written for every
    pair return NaN wherever two computed singular values come out equal (2026-10-02): five of
    these six cases on JAX, where the NaN reaches the gradient through the ``where`` even with
    the floor inactive, and three on Torch, which hands a layer the floor does not touch back
    as it was.
    """
    got, finite = _degenerate_derivative(backend, rotated, values)
    assert np.isfinite(got)
    assert got == pytest.approx(finite, rel=RELATIVE)


def _floor_test_matrices(rotated, values):
    """``B0 = U diag(values) V^H`` (diagonal phases, or random unitaries), a direction ``D`` and a
    read-out ``R``, from one seed."""
    rng = np.random.default_rng(11)
    n = len(values)
    if rotated:
        U, V = _unitary(rng, n), _unitary(rng, n)
    else:
        U, V = np.diag(np.exp(2j * np.pi * rng.random(n))), np.eye(n)
    B0 = (U * np.asarray(values)) @ V.conj().T
    D = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    R = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return B0, D, R


def _namespace(backend):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        return array_namespace(jnp.asarray(0.0))
    torch = pytest.importorskip('torch')
    return array_namespace(torch.tensor(0.0, dtype=torch.float64))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('rotated', [False, True], ids=['diagonal', 'rotated'])
def test_a_singular_value_exactly_at_the_floor_is_not_clipped(backend, rotated):
    """``s < floor`` clips; ``s == floor`` does not, so there the floored matrix is ``B`` and its
    derivative ``dB``'s. Pinned with the floor set to the smallest singular value itself, as the
    backend that floors computes it: the derivative of ``<R, floored(B0 + t D)>`` must be
    ``<R, D>``. Counting the value at the floor as clipped (``s <= floor``) holds that mode's
    magnitude and drops the Hermitian part of its diagonal entry; measured 2026-10-02, that
    moves the derivative here by 24% (rotated) and 75% (diagonal).

    In a solve the computed singular value lands on one side of the floor or the other by
    rounding, so either one-sided slope can come back at a crossing; this pins only the rule.
    """
    xp = _namespace(backend)
    B0, D, R = _floor_test_matrices(rotated, [0.004, 0.01, 0.3, 1.0])
    floor = float(xp.linalg.svd(xp.asarray(B0))[1][-1])

    def loss(t):
        floored = _floor_singular_values(xp, xp.asarray(B0) + t * xp.asarray(D), floor)
        return xp.sum(xp.real(xp.conj(xp.asarray(R)) * floored))

    if backend == 'jax':
        import jax
        got = float(jax.grad(loss)(xp.asarray(0.0)))
    else:
        import torch
        t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        got = float(torch.autograd.grad(loss(t), t)[0])
    assert got == pytest.approx(float(np.sum(np.real(np.conj(R) * D))), rel=1e-12)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_an_exactly_singular_matrix_floors_finitely_and_has_no_derivative(backend):
    """Two exact zeros among the singular values: a lossless metal at its pole, in miniature.

    The floored matrix is finite, and traced it is the NumPy one: the forward pass does not read
    the derivative term. The derivative is NaN -- not an invented finite value. Between the two
    zeros ``D2 = (g_i + g_j) / (s_i + s_j)`` is ``2 floor / 0``; on the way in it grows as
    ``floor / s``, and at zero the clipped modes' left and right vectors have no relative phase.
    """
    xp = _namespace(backend)
    B0, D, R = _floor_test_matrices(False, [0.0, 0.0, 0.01, 0.3, 1.0])
    want = _floor_singular_values(np, B0, 0.02)
    assert np.linalg.svd(want, compute_uv=False).min() == pytest.approx(0.02, rel=1e-12)

    def floored(t):
        return _floor_singular_values(xp, xp.asarray(B0) + t * xp.asarray(D), 0.02 + 0.006 * t)

    if backend == 'jax':
        import jax
        value, pullback = jax.vjp(floored, xp.asarray(0.0))
        derivative, = pullback(xp.asarray(R))
    else:
        import torch
        t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        value = floored(t)
        derivative, = torch.autograd.grad(xp.sum(xp.real(xp.conj(xp.asarray(R)) * value)), t)
        value = value.detach()
    assert np.all(np.isfinite(np.asarray(value)))
    np.testing.assert_allclose(np.asarray(value), want, rtol=0, atol=1e-14)
    assert not np.isfinite(float(derivative))


@pytest.mark.parametrize('mode', ['reverse over forward_ad', 'forward_ad over reverse'])
def test_the_floor_alone_refuses_a_mixed_second_order_on_torch(mode):
    """PyTorch can differentiate a ``forward_ad`` tangent in reverse mode, and carry a tangent
    through a backward pass; either differentiates the floor's derivative, a linear map with
    coefficients from the detached SVD, and comes back without the floor's curvature. Both pass
    through the floor's own backward, which refuses any backward pass through a floor built from
    a dual -- here on the floor alone, with a read-out linear in the floored matrix, as through
    ``eps_conv_matrices()`` below.

    The path is nonlinear, ``B0 + 0.002 sin(t) D`` floored at ``0.02 + 0.006 t^2`` with two modes
    clipped at ``t = 0.3``, so its second derivative is not the linear map's. Measured
    2026-10-02 with the refusal removed: -0.007473 in either order, where a central difference
    of the exact first derivative reads +0.036047.
    """
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    xp = _namespace('torch')
    B0, D, R = _floor_test_matrices(True, [0.004, 0.01, 0.3, 1.0])

    def loss(t):
        B = xp.asarray(B0) + 0.002 * xp.sin(t) * xp.asarray(D)
        floored = _floor_singular_values(xp, B, 0.02 + 0.006 * t * t)
        return xp.sum(xp.real(xp.conj(xp.asarray(R)) * floored))

    x = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    with forward_ad.dual_level(), pytest.raises(RuntimeError, match='first order'):
        dual = forward_ad.make_dual(x, torch.tensor(1.0, dtype=torch.float64))
        if mode == 'reverse over forward_ad':
            torch.autograd.grad(forward_ad.unpack_dual(loss(dual)).tangent, x)
        else:
            torch.autograd.grad(loss(dual), x)


#: The conv-matrix read-out of the tests below, ``Re sum(conj(W) * eps_xx)`` of
#: ``eps_conv_matrices()`` on the disks at 81 harmonics: the floor binds on 2 modes on gold and
#: on none on the dielectric. ``solve()`` refuses a second derivative at the modal operator
#: whatever the floor does, so this is the route on which the floor's own rules show.
CONV_ORDER = 4
GOLD, DIELECTRIC = AU_852, 4.0 + 0.3j


def _conv_solver(backend, material, lx=140.0):
    par, cfg = disks(material, 852.0)
    par[0]['shapes'][0]['lx'] = lx
    return wavebit.rcwa(backend=backend, structure=par, truncation_order=(CONV_ORDER, CONV_ORDER),
                        **cfg)


def _conv_read_out(backend, material, parameter='diameter', geometry=None):
    """``x -> Re sum(conj(W) * eps_xx)`` on ``backend`` (0, 1 or 2), with ``x`` the disk's
    diameter or the real part of its permittivity (``parameter``), the other held."""
    n = (2 * CONV_ORDER + 1) ** 2
    rng = np.random.default_rng(3)
    W = rng.standard_normal((1, n, n)) + 1j * rng.standard_normal((1, n, n))

    def read_out(x):
        if parameter == 'diameter':
            solver = _conv_solver(backend, material, x)
        elif backend == 'torch':
            import torch
            solver = _conv_solver(backend, torch.complex(x, torch.tensor(material.imag, dtype=x.dtype)))
        else:
            solver = _conv_solver(backend, x + 1j * material.imag)
        epx = solver.eps_conv_matrices(**({} if geometry is None else dict(geometry=geometry)))[0]
        xp = array_namespace(epx)
        return xp.sum(xp.real(xp.conj(xp.asarray(W)) * epx))
    return read_out


def _conv_difference(material, parameter='diameter'):
    """The read-out's central difference on NumPy, extrapolated in the step."""
    read_out = _conv_read_out('numpy', material, parameter)
    at, step = (140.0, 1e-3) if parameter == 'diameter' else (material.real, 1e-4)
    return _difference(lambda x: float(read_out(x)), at, step)


@pytest.mark.parametrize('mode', ['torch forward_ad', 'jax jvp'])
@pytest.mark.parametrize('parameter', ['diameter', 'Re eps'])
def test_forward_mode_through_the_floor_is_the_exact_derivative(mode, parameter):
    """Forward mode gets the floor's exact derivative, as reverse mode does: ``jax.jvp`` goes
    through the same ``custom_jvp`` rule as ``jax.grad``, and on PyTorch the floor's
    ``autograd.Function`` carries a ``jvp``. ``Re eps`` moves the floor itself, so that case
    needs the floor's own derivative as well.

    A ``torch.autograd.forward_ad`` dual has ``requires_grad`` False, which is all the floor
    used to ask about, so it went down the plain branch, where the floored part is a constant
    and its tangent ``dB - dB = 0``. Measured 2026-10-02 on this read-out, forward mode gave
    -0.538854 in the diameter and +3.423643 in ``Re eps``, where reverse mode and a central
    difference read -27.626431 and -17.338962, with no error.
    """
    at = 140.0 if parameter == 'diameter' else GOLD.real
    finite = _conv_difference(GOLD, parameter)
    if mode == 'jax jvp':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        read_out = _conv_read_out('jax', GOLD, parameter)
        forward = float(jax.jvp(read_out, (jnp.asarray(at),), (jnp.asarray(1.0),))[1])
        reverse = float(jax.grad(read_out)(jnp.asarray(at)))
    else:
        torch = pytest.importorskip('torch')
        from torch.autograd import forward_ad
        read_out = _conv_read_out('torch', GOLD, parameter)
        with forward_ad.dual_level():
            x = forward_ad.make_dual(torch.tensor(at, dtype=torch.float64),
                                     torch.tensor(1.0, dtype=torch.float64))
            forward = float(forward_ad.unpack_dual(read_out(x)).tangent)
        x = torch.tensor(at, dtype=torch.float64, requires_grad=True)
        reverse = float(torch.autograd.grad(read_out(x), x)[0])
    assert forward == pytest.approx(finite, rel=RELATIVE)
    assert forward == pytest.approx(reverse, rel=1e-12)


@pytest.mark.parametrize('transform', ['jvp', 'grad'])
@pytest.mark.parametrize('metal', [True, False], ids=['gold, floor binds', 'dielectric'])
def test_torch_func_is_refused_by_name_where_the_floor_binds(transform, metal):
    """Where the floor binds, ``torch.func``'s grad and jvp transforms are refused by name,
    pointing at ``torch.autograd``, which it supports. The floor's ``autograd.Function`` has a
    functorch ``vmap`` rule, so ``torch.autograd`` through ``torch.func.vmap`` is exact; a grad
    or jvp level would reach its first-order guard, which cannot tell one from a second
    derivative, so it is turned away at the door. Before, ``torch.func.jvp`` took the plain
    branch and returned -0.538854 where the derivative is -27.626431, and ``torch.func.grad``
    raised PyTorch's own error about ``setup_context``. Where the floor does not bind it is never
    reached, and ``torch.func`` stays exact (without ``vmap``; under it, binding is per lane).
    """
    torch = pytest.importorskip('torch')
    material = GOLD if metal else DIELECTRIC
    read_out = _conv_read_out('torch', material)
    x = torch.tensor(140.0, dtype=torch.float64)
    if transform == 'jvp':
        def run():
            return torch.func.jvp(read_out, (x,), (torch.ones_like(x),))[1]
    else:
        def run():
            return torch.func.grad(read_out)(x)
    if metal:
        with pytest.raises(NotImplementedError, match='torch.func'):
            run()
    else:
        assert float(run()) == pytest.approx(_conv_difference(material), rel=RELATIVE)


@pytest.mark.parametrize('backend', ['jax', 'jit', 'jax.checkpoint, grad of grad',
                                     'jax.checkpoint, hessian', 'torch',
                                     'torch, reverse over forward_ad',
                                     'torch, forward_ad over reverse'])
@pytest.mark.parametrize('metal', [True, False], ids=['gold, floor binds', 'dielectric'])
def test_a_second_derivative_through_the_floor_is_refused_where_it_binds(backend, metal):
    """The floor's derivative is linear in ``dB`` with coefficients from the detached SVD, so a
    second derivative taken through it would lack the floor's curvature: on the gold disks at 81
    harmonics (2 modes clipped), with the read-out ``Re sum(conj(W) * eps_xx)`` of
    ``eps_conv_matrices()``, it read +4.193117 on JAX and Torch alike where a central difference
    of the exact first derivative reads -17.546 -- the wrong sign, and no error. Refused instead,
    as the modal operator's second derivative is: ``NotImplementedError`` on JAX,
    ``RuntimeError`` for ``create_graph=True`` on Torch, and ``RuntimeError`` for PyTorch's two
    mixed orders -- a backward pass through a ``forward_ad`` tangent, and a ``forward_ad``
    tangent carried through a backward pass -- which, unrefused, read the same +4.193117.

    Where the floor does not bind the map is the identity and the Hessian stays exact (-2.1e-3
    on the dielectric), in every mode. Under ``jax.jit`` and ``jax.checkpoint`` whether it binds
    is not known when the trace is built, so a second derivative through any NVF layer is refused
    there.

    ``jax.checkpoint`` stages the rule's tangent and partially evaluates it before the outer
    derivative arrives, and that pass drops whatever the tangent computes and does not read. The
    refusal was such a call until 2026-10-02, so under ``jax.checkpoint`` it was dropped and the
    partial Hessian came back: -0.8494 on the gold through ``jax.hessian`` and through
    ``jax.grad`` of ``jax.grad`` alike, where a difference of the first derivative reads -17.546.
    """
    material = GOLD if metal else DIELECTRIC
    staged = backend == 'jit' or 'checkpoint' in backend
    if backend.startswith('torch'):
        torch = pytest.importorskip('torch')
        from torch.autograd import forward_ad
        read_out = _conv_read_out('torch', material)

        def tangent(at, requires_grad=False):
            x = torch.tensor(at, dtype=torch.float64, requires_grad=requires_grad)
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(x, torch.tensor(1.0, dtype=torch.float64))
                return forward_ad.unpack_dual(read_out(dual)).tangent, x

        def first(at, create_graph=False):
            if backend == 'torch, reverse over forward_ad':
                return float(tangent(at)[0])
            x = torch.tensor(at, dtype=torch.float64, requires_grad=True)
            g, = torch.autograd.grad(read_out(x), x, create_graph=create_graph)
            return (g, x) if create_graph else float(g)

        def second(at):
            if backend == 'torch, reverse over forward_ad':
                t, x = tangent(at, requires_grad=True)
                return float(torch.autograd.grad(t, x)[0])
            if backend == 'torch, forward_ad over reverse':
                x = torch.tensor(at, dtype=torch.float64, requires_grad=True)
                with forward_ad.dual_level():
                    dual = forward_ad.make_dual(x, torch.tensor(1.0, dtype=torch.float64))
                    g, = torch.autograd.grad(read_out(dual), x)
                    return float(forward_ad.unpack_dual(g).tangent)
            g, x = first(at, create_graph=True)
            return float(torch.autograd.grad(g, x)[0])
        refusal = RuntimeError
    else:
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        geometry = _conv_solver('jax', material).freeze_geometry() if staged else None
        read_out = _conv_read_out('jax', material, geometry=geometry)
        if 'checkpoint' in backend:
            read_out = jax.checkpoint(read_out)
        gradient = jax.grad(read_out)
        hessian = jax.hessian(read_out) if 'hessian' in backend else jax.grad(gradient)
        first = lambda x: float(gradient(jnp.asarray(x)))
        second = lambda x: float(hessian(jnp.asarray(x)))
        refusal = NotImplementedError

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if backend == 'jit':                  # the frozen geometry holds at 140 nm only
            compiled = jax.jit(gradient)(jnp.asarray(140.0))
            assert float(compiled) == pytest.approx(first(140.0), rel=1e-12)
            with pytest.raises(refusal, match='first order'):
                jax.jit(hessian)(jnp.asarray(140.0))
            return
        if staged:
            # `jax.checkpoint` stages the read-out as `jax.jit` does, so whether the floor binds
            # is not known there either, and the dielectric is refused as well.
            assert first(140.0) == pytest.approx(_conv_difference(material), rel=RELATIVE)
            with pytest.raises(refusal, match='first order'):
                second(140.0)
            return
        assert first(140.0) == pytest.approx(_conv_difference(material), rel=RELATIVE)
        if metal:
            with pytest.raises(refusal, match='first order'):
                second(140.0)
        else:
            assert second(140.0) == pytest.approx(_difference(first, 140.0, 1e-3), rel=RELATIVE)


def test_a_lossless_metal_no_longer_divides_by_zero():
    """A metal given without its loss makes ``<1/eps>`` Hermitian with a sign change, and at the
    pole itself singular: the inverse rule then returns infinities. The floor turns that into a
    finite, if extreme, tensor -- which is the least one can ask of a solve on a real material
    given without its loss.

    The metal is put *at* the pole, not near it: ``<1/eps> = (1/eps_d) I + (1/eps_m - 1/eps_d) K``
    with ``K`` the disk's indicator, so the eigenvalue on a transition mode of ``K`` -- one with
    ``kappa`` of the dielectric, ``1 - kappa`` of the metal -- vanishes at
    ``eps_m = eps_d (1 - 1/kappa)``. Anywhere else a lossless metal leaves ``<1/eps>`` regular:
    at ``eps = -25`` its smallest singular value is 3.8e-3, and the unfloored closing is finite
    (``max|eps_xx|`` 4.43), so a test placed there could not fail. At the pole, measured
    2026-10-02 at 121 harmonics: ``kappa = 0.0747``, ``eps_m = -27.857``, smallest singular
    value 4.6e-16, ``max|eps_xx|`` 2.9e13 unfloored and 2.35 floored.
    """
    par, cfg = disks(-25.0 + 0j, 852.0)
    E, _, _ = matrices(par, cfg, 5)
    K_disk = (E - 2.25 * np.eye(len(E))) / (-25.0 - 2.25)
    kappa = 1.0 - np.linalg.eigvalsh((K_disk + K_disk.conj().T) / 2)      # of the dielectric
    kappa_j = kappa[(kappa > 0.05) & (kappa < 0.12)][0]                  # a transition mode
    eps_m = 2.25 * (1.0 - 1.0 / kappa_j)                                 # about -28: gold-like
    par, cfg = disks(eps_m + 0j, 852.0)
    E, B, C = matrices(par, cfg, 5)
    limit = 10 * INVERSE_RULE_BOUND * abs(eps_m)

    assert np.linalg.svd(B, compute_uv=False).min() < 1e-12            # singular, to rounding
    with np.errstate(all='ignore'):
        try:
            bare = _nvf_assemble(np, E, B, *C)
            assert not (np.all(np.isfinite(bare[0])) and np.abs(bare[0]).max() < limit)
        except np.linalg.LinAlgError:
            pass                                                         # singular outright

    slots = _nvf_assemble(np, E, B, *C, eps_bound=largest_eps(par))
    assert all(np.all(np.isfinite(m)) for m in slots)
    assert np.abs(slots[0]).max() < limit
