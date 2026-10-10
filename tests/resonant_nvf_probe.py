"""Separate Fourier truncation and normal quadrature in the tilted contact fixture.

The three Wavebit NVF extensions use exact rectangle transforms and the same solve.
``wavebit-direct`` removes only the NVF correction to check that solve against S4.
It is a scoped diagnostic, not a production factorization option.
The optional S4 direct rule is an independent implementation, also subject to
Fourier truncation. Its actual reciprocal orders must match Wavebit's square cap.
Run: PYTHONPATH=$PWD python tests/resonant_nvf_probe.py --orders 6 10 14 18
Add --methods circular s4-direct to compare only the new NVF and S4.
"""
import argparse
import json
import time
from unittest.mock import patch
import numpy as np
import wavebit
from wavebit._core import boundary as B, factorization as F, parametric as P


def polygon_extension(xp, shapes, background, hu, hv, lattice, dtype,
                      shape_nvf_grid_size=256, plan=None):
    return B.to_conv_mat_boundary_layer(xp, shapes, background, hu, hv, lattice,
                                       dtype, shape_nvf_grid_size=shape_nvf_grid_size)


def fixture_shape():
    angle = np.arctan2(1., 2.)
    return dict(type='rectangle', cx=0., cy=0., lx=640.,
                ly=700.*np.sin(angle), angle=angle, material=12.25)


def make_solver(order, grid, method='pieces', shape=None):
    return wavebit.rcwa(
        backend='numpy', period=(700., 700.), wavelength=900.,
        top=1., bot=2.25, theta=.2, phi=0., psi=0., thickness=(300.,),
        truncation_order=(order, order), shape_nvf_grid_size=grid,
        nvf_projector='circular' if method == 'circular' else None,
        structure=[dict(background=1., shapes=[fixture_shape() if shape is None else shape])])


def s4_solve(order, shape=None, *, basis=None):
    """Independent S4 solve on a checked square order set; None selects direct rule."""
    import S4
    shape = fixture_shape() if shape is None else shape
    s = S4.New(Lattice=((700., 0.), (0., 700.)), NumBasis=(2*order+1)**2)
    if basis not in (None, 'Default', 'Normal', 'Jones'):
        raise ValueError(f'unknown S4 polarization basis: {basis!r}')
    options = dict(LatticeTruncation='Parallelogramic', PolarizationDecomposition=basis is not None)
    if basis is not None:
        options['PolarizationBasis'] = basis
    s.SetOptions(**options)
    for name, epsilon in (('air', 1.), ('substrate', 2.25), ('inclusion', float(shape['material']))):
        s.SetMaterial(Name=name, Epsilon=epsilon)
    s.AddLayer(Name='top', Thickness=0., Material='air')
    uniform = shape['lx'] == 700. and shape['ly'] == 700. and shape.get('angle', 0.) == 0.
    # A full-cell rectangle is a homogeneous periodic layer. Stating it as such
    # avoids S4's Default/Jones zero-vector failure on this validation control.
    s.AddLayer(Name='film', Thickness=300., Material='inclusion' if uniform else 'air')
    if not uniform:
        s.SetRegionRectangle(Layer='film', Material='inclusion',
                            Center=(float(shape.get('cx', 0.)), float(shape.get('cy', 0.))),
                            Angle=float(np.rad2deg(shape.get('angle', 0.))),
                            Halfwidths=(float(shape['lx'])/2, float(shape['ly'])/2))
    s.AddLayer(Name='bottom', Thickness=0., Material='substrate')
    actual = set(s.GetBasisSet())
    expected = {(m, n) for m in range(-order, order+1) for n in range(-order, order+1)}
    if actual != expected:
        raise ValueError('S4 reciprocal-order set differs from the requested square truncation')
    s.SetFrequency(1./900.)
    s.SetExcitationPlanewave(IncidenceAngles=(float(np.rad2deg(.2)), 0.),
                            sAmplitude=0., pAmplitude=1.)
    incoming, reflected = s.GetPowerFlux(Layer='top')
    transmitted, _ = s.GetPowerFlux(Layer='bottom')
    fluxes = np.asarray([incoming, reflected, transmitted], dtype=complex)
    if not np.isfinite(fluxes).all() or incoming.real <= 0:
        raise FloatingPointError('S4 returned invalid fluxes; this solve is not a reference')
    return float(transmitted.real/incoming.real), float(-reflected.real/incoming.real)


def s4_direct(order, shape=None):
    return s4_solve(order, shape)


def direct_from_coeffs(xp, eps_c, inv_eps_c, c_xx, c_xy, c_yy,
                       half_width_u, half_width_v, c_yx=None, eps_bound=None):
    """Keep the exact material series, with Laurent's rule in both transverse slots.

    ``eps_bound`` is accepted to match :func:`wavebit._core.parametric.nvf_from_coeffs` and
    ignored: the inverse rule it bounds is not used here."""
    ep_dir = P.conv_matrix_2d(xp, eps_c, half_width_u, half_width_v)
    zero = xp.zeros_like(ep_dir)
    return F.ConvMatrices(ep_dir, ep_dir, xp.linalg.inv(ep_dir), zero, zero)


def solve_case(order, grid, method='pieces', shape=None):
    """Return total (T, R), restoring every diagnostic override before returning."""
    s4_bases = {'s4-direct': None, 's4-default': 'Default',
                's4-normal': 'Normal', 's4-jones': 'Jones'}
    if method in s4_bases:
        return s4_solve(order, shape, basis=s4_bases[method])
    if method not in ('pieces', 'polygon', 'circular', 'wavebit-direct'):
        raise ValueError(f'unknown comparison method: {method!r}')
    if method == 'wavebit-direct':
        target, builder = 'nvf_from_coeffs', direct_from_coeffs
    else:
        target = 'to_conv_mat_parametric_nvf'
        builder = polygon_extension if method == 'polygon' else P.to_conv_mat_parametric_nvf
    with patch.object(P, target, builder):
        result = make_solver(order, grid, method, shape).solve()
    return float(np.sum(result.T)), float(np.sum(result.R))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=[6, 10, 14])
    parser.add_argument('--grids', nargs='+', type=int, default=[256, 512])
    parser.add_argument('--methods', nargs='+',
                        choices=['pieces', 'polygon', 'circular', 'wavebit-direct',
                                 's4-direct', 's4-default', 's4-normal', 's4-jones'],
                        default=['pieces', 'polygon', 'circular'])
    args = parser.parse_args()
    try:
        for order in args.orders:
            for method in args.methods:
                for grid in ([None] if method.startswith('s4-') else args.grids):
                    P._PROJECTOR_CACHE.clear()
                    start = time.monotonic()
                    transmission, reflection = solve_case(order, grid, method)
                    print(json.dumps(dict(order=order, grid=grid, method=method,
                                          T=transmission, R=reflection,
                                          energy_error=abs(transmission+reflection-1.),
                                          seconds=time.monotonic()-start)), flush=True)
    finally:
        P._PROJECTOR_CACHE.clear()


if __name__ == '__main__':
    main()
