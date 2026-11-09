"""Physical checks for typed sources, angular spectra and internal emission."""

from wavebit.rcwa._engine.sources import Beam as _legacy_wavebit_Beam

from wavebit.rcwa._engine.sources import ElectricDipole as _legacy_wavebit_ElectricDipole

from wavebit.rcwa._engine.sources import GaussianBeam as _legacy_wavebit_GaussianBeam

from wavebit.rcwa._engine.sources import PlaneWave as _legacy_wavebit_PlaneWave

from wavebit.rcwa._engine.sources import Source as _legacy_wavebit_Source

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
from dataclasses import replace

import numpy as np
import pytest

import wavebit
from wavebit.rcwa._engine.source_expansion import EPSILON_0


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def vacuum(**updates):
    options = dict(wavelength=1., period=(2.3, 2.7), top=1., bot=1.,
                   structure=np.ones((1, 1, 1)), thickness=[.8],
                   truncation_order=(1, 1), answers='field')
    return options | updates


def test_source_list_and_order_adapter_keep_boundary_physics():
    top = _legacy_wavebit_PlaneWave(side='top', kx=.3, polarization=(.6, .8j))
    bottom = _legacy_wavebit_PlaneWave(side='bottom', kx=.3, amplitude=.4j)
    options = vacuum(structure=np.array([[[4., 1.]]]))
    a = _legacy_wavebit_rcwa(**options, excitation=[top, bottom]).solve()
    b = _legacy_wavebit_rcwa(**options, excitation={'top': top, 'bottom': bottom}).solve()
    c = _legacy_wavebit_rcwa(**options, excitation={'top': _legacy_wavebit_Beam(kx=.3, polarization=(.6, .8j)),
                                           'bottom': _legacy_wavebit_Beam(kx=.3, amplitude=.4j)}).solve()
    for r in (b, c):
        np.testing.assert_allclose(r.boundary_fields('top', [.1]), a.boundary_fields('top', [.1]), atol=1e-12)
    assert isinstance(top, _legacy_wavebit_Source)
    assert not isinstance(_legacy_wavebit_GaussianBeam(), _legacy_wavebit_PlaneWave)
    with pytest.raises(TypeError):
        _legacy_wavebit_PlaneWave(orders={0: 1.})
    with pytest.raises(ValueError, match='conflicts'):
        _legacy_wavebit_rcwa(**options, excitation={'bottom': top})


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('moment', [(1e-29, 0., 0.), (0., 1e-29j, 0.), (0., 0., 1e-29)])
def test_dipole_matches_homogeneous_dyadic_green_spectrum(backend, moment):
    """E_G = i k0^2/(2 eps0 A kz) (I-kk/k0^2) p, independently of SMM."""
    position, q = np.array([.12, -.08, .31]), np.array([.21, .13])
    source = _legacy_wavebit_ElectricDipole(position=tuple(position), moment=moment, bloch_k=tuple(q))
    r = _legacy_wavebit_rcwa(**vacuum(backend=backend), excitation=[source], length_unit=1e-6).solve()
    k0 = 2*np.pi
    area = 2.3*2.7
    xy = np.array([.07, -.12])
    for z, side in [(.8, 'top'), (0., 'bottom')]:
        expected = np.zeros(6, complex)
        sign = 1 if z > position[2] else -1
        for m in range(-1, 2):
            for n in range(-1, 2):
                kt = q + 2*np.pi*np.array([m/2.3, n/2.7])
                kz = np.sqrt(k0*k0-kt@kt+0j)
                k = np.r_[kt, sign*kz]
                p = np.array(moment)
                electric = 1j*k0*k0/(2*EPSILON_0*area*1e-18*kz)*(p-k*np.dot(k, p)/(k0*k0))
                magnetic = np.cross(k, electric)/k0
                phase = np.exp(1j*(kt@(xy-position[:2])+kz*abs(z-position[2])))
                expected += phase*np.r_[electric, magnetic]
        np.testing.assert_allclose(host(r.boundary_fields(side, [xy[0]], [xy[1]]))[0, 0], expected,
                                   atol=2e-12, rtol=2e-12)
    assert r.output_top is None and r.absorption is None
    assert host(r.power.outgoing_top) > 0
    np.testing.assert_allclose(host(r.power.outgoing_top), host(r.power.outgoing_bottom), rtol=1e-12)


def test_dipole_source_jump_and_outer_continuity_with_reflecting_stack():
    source = _legacy_wavebit_ElectricDipole(position=(.11, .07, .25), moment=(1e-29, .4e-29j, 0.), bloch_k=(.2, .1))
    options = vacuum(structure=[dict(background=4., shapes=[]), dict(background=1., shapes=[])],
                     thickness=[.5, .3], top=2.25, bot=1., truncation_order=0)
    r = _legacy_wavebit_rcwa(**options, excitation=source, length_unit=1e-6).solve()
    x, y = [.03], [.05]
    near = r.reconstruct_field(x, y, [.25-1e-8, .25+1e-8])[:, 0, 0]
    phase = np.exp(1j*(.2*(x[0]-.11)+.1*(y[0]-.07)))
    p = np.array(source.moment)/(EPSILON_0*(2.3*2.7)*1e-18)
    expected = phase*np.array([0., 0., -2j*np.pi*p[1], 2j*np.pi*p[0]])
    np.testing.assert_allclose((near[1]-near[0])[[0, 1, 3, 4]], expected, atol=4e-7)
    edge = r.reconstruct_field(x, y, [0., .8])[:, 0, 0]
    for i, side in enumerate(('bottom', 'top')):
        np.testing.assert_allclose(edge[i, [0, 1, 3, 4]],
                                   r.boundary_fields(side, x, y)[0, 0, [0, 1, 3, 4]], atol=2e-12)
    with pytest.raises(ValueError, match='singular'):
        r.reconstruct_field(x, y, [.25])


def test_multiple_internal_and_external_sources_superpose_fields():
    dipole = _legacy_wavebit_ElectricDipole(position=(0., 0., .25), moment=(0., 1e-29, .3e-29j), bloch_k=(.2, .1))
    second = replace(dipole, position=(.1, .1, .55), moment=(1e-29j, 0., 0.))
    plane = _legacy_wavebit_PlaneWave(side='bottom', kx=.2, ky=.1, amplitude=.7j)
    options = vacuum(structure=[dict(background=4.+.2j, shapes=[])], length_unit=1e-6)
    separate = [_legacy_wavebit_rcwa(**options, excitation=s).solve() for s in (dipole, second, plane)]
    joint = _legacy_wavebit_rcwa(**options, excitation=[dipole, second, plane]).solve()
    points = ([.13], [.02], [.1, .4, .7])
    np.testing.assert_allclose(joint.reconstruct_field(*points),
                               sum(r.reconstruct_field(*points) for r in separate), atol=2e-12)
    for side in ('top', 'bottom'):
        np.testing.assert_allclose(joint.boundary_fields(side, [.13], [.02]),
                                   sum(r.boundary_fields(side, [.13], [.02]) for r in separate), atol=2e-12)
    assert joint.absorption is None and joint.output_top is None


def test_dipole_moment_units_are_independent_of_scene_length_units():
    source = _legacy_wavebit_ElectricDipole(position=(.1, .1, .3), moment=(1e-29, 0., 0.))
    a = _legacy_wavebit_rcwa(**vacuum(), excitation=source, length_unit=1e-6).solve()
    b = _legacy_wavebit_rcwa(**vacuum(wavelength=1000., period=(2300., 2700.), thickness=[800.]),
                     excitation=replace(source, position=(100., 100., 300.)), length_unit=1e-9).solve()
    np.testing.assert_allclose(a.boundary_fields('top', [.2], [.1]),
                               b.boundary_fields('top', [200.], [100.]), rtol=1e-12, atol=1e-12)


def test_isolated_dipole_quadrature_matches_supercell_representation():
    """A 3x3 Bloch grid with 1x1 orders tiles the 9x9 supercell spectrum."""
    source = _legacy_wavebit_ElectricDipole(position=(.1, -.1, .3), moment=(0., 1e-29, 0.),
                                   repetition='isolated', sampling=(3, 3))
    box = ((-2., 2.), (-2., 2.))
    a = _legacy_wavebit_rcwa(**vacuum(), excitation=source, length_unit=1e-6).solve(power_region=box)
    b = _legacy_wavebit_rcwa(**vacuum(period=(6.9, 8.1), truncation_order=(4, 4)),
                     excitation=replace(source, repetition='periodic'), length_unit=1e-6).solve(power_region=box)
    assert a.group_count == 9
    for side in ('top', 'bottom'):
        np.testing.assert_allclose(a.boundary_fields(side, [-.3, .2], [-.1, .2]),
                                   b.boundary_fields(side, [-.3, .2], [-.1, .2]), atol=2e-11)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_gaussian_one_component_limit_and_focus_phase(side):
    source = _legacy_wavebit_GaussianBeam(side=side, waist=2., focus=(.1, -.2, .4), sampling=(1, 1),
                                  theta=.2, phi=.3, polarization=(.6, .8j), amplitude=.7j)
    options = vacuum(truncation_order=0)
    solver = _legacy_wavebit_rcwa(**options, excitation=source)
    component = solver._gaussian_waves(replace(source, side=side))[0]
    a = solver.solve(power_region=((-2., 2.), (-2., 2.)))
    b = _legacy_wavebit_rcwa(**options, excitation=component).solve()
    np.testing.assert_allclose(a.boundary_fields(side, [.1]), b.boundary_fields(side, [.1]), atol=1e-12)
    # At its declared focus, the electric field norm is the requested amplitude.
    field = a.reconstruct_field([.1], [-.2], [.4])[0, 0, 0, :3]
    np.testing.assert_allclose(np.linalg.norm(field), .7, atol=1e-12)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_gaussian_focal_envelope_and_vacuum_transmission(backend):
    source = _legacy_wavebit_GaussianBeam(waist=8., focus=(0., 0., .4), sampling=(11, 11), spectral_extent=5.)
    options = vacuum(backend=backend, period=(2.31, 2.73), truncation_order=0)
    r = _legacy_wavebit_rcwa(**options, excitation=source).solve(power_region=((-24., 24.), (-24., 24.)))
    x = np.linspace(-8., 8., 9)
    field = host(r.reconstruct_field(x, [0.], [.4]))[0, 0]
    np.testing.assert_allclose(field[:, 1].real, np.exp(-(x/8.)**2), atol=6e-4)
    np.testing.assert_allclose(np.linalg.norm(field[4, :3]), 1., atol=1e-12)
    np.testing.assert_allclose(host(r.output_top), 0., atol=1e-24)
    # Symmetric observation planes about the focus give equal collected power.
    np.testing.assert_allclose(host(r.output_bottom), 1., atol=1e-11)


def test_dipole_input_limits_are_explicit():
    source = _legacy_wavebit_ElectricDipole(position=(0., 0., .3), moment=(0., 1e-29, 0.))
    with pytest.raises(ValueError, match='length_unit'):
        _legacy_wavebit_rcwa(**vacuum(), excitation=source).solve()
    with pytest.raises(ValueError, match='uniform host'):
        _legacy_wavebit_rcwa(**vacuum(structure=np.array([[[1., 4.]]])), excitation=source, length_unit=1e-6).solve()
    # Keeping only the zeroth Fourier order must not disguise a patterned host.
    with pytest.raises(ValueError, match='uniform host'):
        _legacy_wavebit_rcwa(**vacuum(structure=np.array([[[1., 4.]]]), truncation_order=0),
                     excitation=source, length_unit=1e-6).solve()
    with pytest.raises(ValueError, match='strictly inside'):
        _legacy_wavebit_rcwa(**vacuum(), excitation=replace(source, position=(0., 0., 0.)), length_unit=1e-6).solve()
    with pytest.raises(ValueError, match='omit bloch_k'):
        replace(source, repetition='isolated', bloch_k=(0., 0.))
    with pytest.raises(ValueError, match='sampling'):
        _legacy_wavebit_GaussianBeam(sampling=(0, 2))
    with pytest.raises(ValueError, match='lossless'):
        _legacy_wavebit_rcwa(**vacuum(top=1.+.1j), excitation=_legacy_wavebit_GaussianBeam()).solve(
            power_region=((-2., 2.), (-2., 2.)))
    with pytest.raises(ValueError, match='finite rectangle'):
        _legacy_wavebit_rcwa(**vacuum(), excitation=_legacy_wavebit_GaussianBeam()).solve(power_region='infinite')


def test_emission_balance_with_tensor_host_and_patterned_neighbor():
    material = wavebit.Material([2.4, 3.1, 4.2]).rotated(.3, 'y')
    source = _legacy_wavebit_ElectricDipole(position=(.1, -.07, .4), moment=(1e-29, .2e-29j, .3e-29),
                                   bloch_k=(.2, .1))
    options = vacuum(structure=[dict(background=1., shapes=[
        dict(type='rectangle', lx=.8, ly=.7, material=4.)]),
        dict(background=material, shapes=[])], thickness=[.2, .6],
        factorization='li', length_unit=1e-6)
    r = _legacy_wavebit_rcwa(**options, excitation=source).solve()
    x = np.linspace(-1.15, 1.15, 11, endpoint=False)
    y = np.linspace(-1.35, 1.35, 11, endpoint=False)
    f = r.reconstruct_field(x, y, [.4-1e-8, .4+1e-8])
    flux = .5*np.real(f[..., 0]*f[..., 4].conj()-f[..., 1]*f[..., 3].conj()).mean(axis=(1, 2))
    np.testing.assert_allclose(flux[1]-flux[0], r.power.outgoing_top+r.power.outgoing_bottom,
                               rtol=2e-8, atol=1e-10)


def _outgoing(result, x=(.13, -.4), y=(-.07, .3)):
    return np.concatenate([host(result.boundary_fields(side, list(x), list(y), part='outgoing')).ravel()
                           for side in ('top', 'bottom')])


@pytest.mark.parametrize('lattice,shift', [
    (None, (1, 0)), (None, (3, 0)), (None, (-2, 1)),
    (((2.3, 0.), (.7, 2.5)), (1, 0)), (((2.3, 0.), (.7, 2.5)), (2, -3))])
def test_dipole_bloch_k_and_a_reciprocal_shift_are_one_source(lattice, shift):
    """exp(i k.R) = exp(i (k+G).R) on every lattice translation R, so bloch_k and bloch_k + G
    describe one periodic dipole array and must radiate the same fields. Its order window was
    centred on bloch_k as written: one G away the fields were off by 62 %, three by 181 %."""
    options = vacuum(structure=[dict(background=4., shapes=[]), dict(background=1.7, shapes=[])],
                     thickness=[.43, .35], bot=2.25, answers='amplitude', length_unit=1e-6)
    if lattice is not None:
        options.pop('period')
        options['lattice'] = np.array(lattice).T
    columns = options.get('lattice', np.diag(options.get('period', (2.3, 2.7))))
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(columns, float)).T
    source = _legacy_wavebit_ElectricDipole(position=(.1, -.07, .2), moment=(1e-29, .2e-29j, .3e-29),
                                            bloch_k=(.2, .1))
    shifted = replace(source, bloch_k=tuple(np.array([.2, .1]) + reciprocal @ np.array(shift)))
    a, b = (_legacy_wavebit_rcwa(**options, excitation=s).solve(power_region='infinite')
            for s in (source, shifted))
    expected = _outgoing(a)
    np.testing.assert_allclose(_outgoing(b), expected, rtol=0, atol=1e-13*np.max(np.abs(expected)))
    for name in ('outgoing_top', 'outgoing_bottom'):
        np.testing.assert_allclose(getattr(b.power, name), getattr(a.power, name), rtol=1e-13)


def test_dipole_bloch_k_outside_the_retained_orders_still_radiates():
    """Only the zone orders propagate here. With bloch_k three reciprocal vectors out and one
    retained order each side, every retained order was evanescent: 6.6e-18 for 0.156."""
    options = vacuum(wavelength=1.5, structure=[dict(background=4., shapes=[])], thickness=[.5],
                     answers='amplitude', length_unit=1e-6)
    source = _legacy_wavebit_ElectricDipole(position=(.1, -.07, .2), moment=(1e-29, .2e-29j, .3e-29),
                                            bloch_k=(.2 + 3*2*np.pi/2.3, .1))
    far = _legacy_wavebit_rcwa(**options, excitation=source).solve(power_region='infinite')
    near = _legacy_wavebit_rcwa(**options, excitation=replace(source, bloch_k=(.2, .1))).solve(
        power_region='infinite')
    assert host(near.power.outgoing_top) > .1
    for name in ('outgoing_top', 'outgoing_bottom'):
        np.testing.assert_allclose(getattr(far.power, name), getattr(near.power, name), rtol=1e-13)


@pytest.mark.parametrize('order', [1, 2])
def test_dipole_one_reciprocal_vector_away_superposes_with_a_plane_wave(order):
    """A plane wave at q and a dipole at q + G share one Bloch sector. Run together they must
    give the sum of their separate runs: before, the dipole's own run centred its orders on
    q + G and the joint run on q, and the sum was off by 61 % (orders 1) and 36 % (orders 2)."""
    options = vacuum(structure=[dict(background=4., shapes=[]), dict(background=1.7, shapes=[])],
                     thickness=[.43, .35], bot=2.25, truncation_order=(order, order),
                     answers='amplitude', length_unit=1e-6)
    dipole = _legacy_wavebit_ElectricDipole(position=(.1, -.07, .2), moment=(1e-29, .2e-29j, .3e-29),
                                            bloch_k=(.2 + 2*np.pi/2.3, .1))
    plane = _legacy_wavebit_PlaneWave(side='bottom', kx=.2, ky=.1, amplitude=.7j)
    joint, alone, lit = (_legacy_wavebit_rcwa(**options, excitation=e).solve(power_region='infinite')
                         for e in ([dipole, plane], dipole, plane))
    assert joint.group_count == 1
    expected = _outgoing(alone) + _outgoing(lit)
    np.testing.assert_allclose(_outgoing(joint), expected, rtol=0, atol=1e-13*np.max(np.abs(expected)))


@pytest.mark.parametrize('host_structure', [np.full((1, 4, 4), 2.25),
                                            [dict(background=2.25, shapes=[])]],
                         ids=['raster', 'shapes'])
@pytest.mark.parametrize('kw', [dict(bloch_k=(.21, .13)),
                                dict(repetition='isolated', sampling=(3, 3))],
                         ids=['bloch_k', 'isolated'])
def test_single_precision_dipole_off_the_zero_bloch_k_tracks_double(host_structure, kw):
    """Single precision accepts the uniform host double accepts, and agrees to float32 accuracy.

    Off ``bloch_k = 0`` the centre-origin gauge leaves ``|phase|**2 = 1 +- 1e-8`` on a uniform
    block's diagonal in complex64, and a float64 tolerance called that host patterned. The
    isolated mode samples nonzero Bloch vectors itself, so it met the same refusal.
    """
    source = _legacy_wavebit_ElectricDipole(position=(.12, -.08, .31),
                                            moment=(1e-29, 1e-29j, 1e-29), **kw)
    box = ((-2., 2.), (-2., 2.))
    fields = {}
    for dtype in (np.complex128, np.complex64):
        r = _legacy_wavebit_rcwa(**vacuum(structure=host_structure), excitation=source,
                                 length_unit=1e-6, complex_dtype=dtype).solve(power_region=box)
        fields[dtype] = np.stack([host(r.boundary_fields(side, [.07, -.3], [-.12, .2]))
                                  for side in ('top', 'bottom')])
    # Evaluation promotes the returned field; the solve itself ran in complex64.
    assert not np.array_equal(fields[np.complex64], fields[np.complex128])
    scale = np.abs(fields[np.complex128]).max()
    # Measured 2.4e-7 of the field scale; float32 resolution is 1.2e-7.
    np.testing.assert_allclose(fields[np.complex64], fields[np.complex128], rtol=0, atol=2e-6*scale)


@pytest.mark.parametrize('contrast', [3., 1e-3])
def test_single_precision_still_refuses_a_patterned_dipole_host(contrast):
    raster = np.full((1, 1, 2), 1.)
    raster[..., 1] += contrast
    source = _legacy_wavebit_ElectricDipole(position=(.12, -.08, .31), moment=(1e-29, 0., 0.),
                                            bloch_k=(.21, .13))
    with pytest.raises(ValueError, match='uniform host'):
        _legacy_wavebit_rcwa(**vacuum(structure=raster), excitation=source, length_unit=1e-6,
                             complex_dtype=np.complex64).solve()
