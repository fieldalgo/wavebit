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
  which is why the bound asserted is the measured one with room, not zero.

The metal permittivities are Johnson & Christy gold, interpolated linearly in ``n`` and ``k``.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core import factorization as F
from wavebit._core import parametric as P
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
    s = wavebit.rcwa(backend=0, structure=par, truncation_order=(M, M), **cfg)
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = wavebit.rcwa(backend=0, structure=par, truncation_order=(5, 5), **cfg).solve()
    assert float(np.sum(res.R).real + np.sum(res.T).real) == pytest.approx(after, abs=1e-3)
    assert not any('Unphysical' in str(w.message) for w in caught)


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
        res = wavebit.rcwa(backend=0, truncation_order=(5, 5), **cfg, **route).solve()
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
