"""Eigenvector-free propagation inside the unchanged NVF mirror sectors.

Electric and magnetic parity spaces have different component sizes. A unitary
permutation puts their flux coordinates in one ordering before the shared
short-slice/scattering-doubling kernel. Exterior matching uses sector-sized
Schur solves, without constructing a doubled boundary matrix.
"""
import numpy as np
from scipy.linalg import solve
from scipy.sparse import eye

from studies.nvf_harmonic_energy import THETA
from studies.nvf_mirror_convergence import exterior_admittance, mirror_sector, sector_geometry
from studies.nvf_scattering_doubling import canonical_factors, reference_coordinates, symmetric_scattering


def sector_factors(sector, xp=np):
    """First-order L/R and cancellation-free LR/RL in their own parity bases."""
    ka, kb, ya, yb, y = sector_geometry(sector)
    na, nb = len(ka), len(kb)
    e, z = sector['effective'], sector['z_inverse']
    ex, ey, xy, yx = e[:na, :na], e[na:, na:], -e[:na, na:], -e[na:, :na]
    a = ka[:, None]*z*ka[None]-np.eye(na)
    d = y.T@z@y-np.eye(nb)
    c, f = ka[:, None]*(z@y), (y.T@z)*ka[None]
    left = xp.block([[-c, a], [-d, f]])
    right = xp.block([[-kb[:, None]*y.T+yx, np.diag(kb**2)-ey],
                       [ex-np.diag(ya**2), y*kb[None]-xy]])
    lr = xp.block([[np.diag(ya**2)+a@ex-c@yx, c@ey-ka[:, None]*y-a@xy],
                   [f@ex-kb[:, None]*y.T-d@yx, np.diag(kb**2)+d@ey-f@xy]])
    upper = (ey@y.T-yx*ka[None])@z
    lower = (ex*ka[None]-xy@y.T)@z
    rl = xp.block([[np.diag(kb**2)-ey+upper@y, kb[:, None]*y.T-yx-upper*ka[None]],
                   [y*kb[None]-xy-lower@y, np.diag(ya**2)-ex+lower*ka[None]]])
    return left, right, lr, rl


def sector_doubling_power(sector, progress=None, *, xp=np, **settings):
    """Return p or s power with the existing constitutive sector unchanged."""
    na, nb = len(sector['a']['n']), len(sector['b']['n'])
    size = na+nb
    permutation, phase = reference_coordinates(na, nb)
    identity = eye(size, format='csc', dtype=complex)
    def port(current_sector, epsilon):
        admittance = exterior_admittance(current_sector, epsilon)[permutation].multiply(phase[:, None]).tocsc()
        return (identity+admittance)/2, (identity-admittance)/2, admittance
    pt, mt, yt = port(sector, 1.)
    pb, mb, yb = port(sector, 2.25)
    if xp is not np:
        pt, mt, yt, pb, mb, yb = [xp.asarray(matrix.toarray()) for matrix in (pt, mt, yt, pb, mb, yb)]
    linear_solve = solve if xp is np else xp.linalg.solve
    def dense(matrix):
        return matrix.toarray() if xp is np else matrix
    polarization = sector['polarization']
    basis = sector['a'] if polarization == 'p' else sector['b']
    origin = np.flatnonzero((basis['n'] == 0) & (basis['m'] == 0))[0]
    incoming = np.zeros(size, complex)
    incoming[origin if polarization == 'p' else na+origin] = np.cos(THETA) if polarization == 'p' else 1.
    incoming = xp.asarray(incoming)
    factors = canonical_factors(*sector_factors(sector, xp), na, nb, xp)
    del sector
    if progress:
        progress('factors_ready')
    scattering, _ = symmetric_scattering(*factors, xp, progress=progress, **settings)
    del factors
    reflection, transmission = scattering[:2]
    outgoing = linear_solve(dense(pb)-reflection@mb, transmission)
    roundtrip = reflection+transmission@(mb@outgoing)
    reflected = linear_solve(dense(pt)-roundtrip@mt, roundtrip@(pt@incoming)-mt@incoming)
    transmitted = outgoing@(pt@incoming+mt@reflected)
    if progress:
        progress('matching_ready')
    def flux(field, admittance):
        return xp.vdot(field, admittance@field).real/np.cos(THETA)
    return flux(transmitted, yb), flux(reflected, yt)


def mirror_doubling_powers(coefficients, order, xp=np, **settings):
    """Differentiable p/s T and R, with the original strict symmetry guards.

    The coefficient field must stay mirror symmetric as parameters move.
    Asymmetric JIT inputs produce NaNs rather than being symmetrized.
    """
    powers = [sector_doubling_power(mirror_sector(coefficients, order, pol, xp), xp=xp, **settings)
              for pol in ('p', 's')]
    t, r = xp.asarray(powers).T
    error = xp.max(abs(t+r-1))
    valid = xp.all(xp.isfinite(t)) & xp.all(xp.isfinite(r)) & (error < 1e-9)
    if xp is np and not valid:
        raise FloatingPointError(f'invalid mirror power balance: {error}')
    return xp.where(valid, t, xp.nan), xp.where(valid, r, xp.nan), dict(energy_error=error)


def main():
    """Time a JIT value and material-scale derivative on the same saved cell."""
    import argparse
    import json
    import time

    import jax
    import jax.numpy as jnp

    from studies.nvf_gap_convergence import load_projection, projected_matrices
    from studies.nvf_harmonic_energy import polarized_powers
    from studies.nvf_scattering_doubling import doubling_powers
    from studies.nvf_thin_gap_energy import ThinGapEnergyModel

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--projection', required=True, help='saved mixed 17-nm-gap, 350-nm-width, 4/9 cell at seam 31')
    parser.add_argument('--orders', nargs='+', type=int, default=[4, 8])
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    jax.config.update('jax_enable_x64', True)
    coefficients, metadata = load_projection(args.projection, 17., 350., [31.], [4., 9.])
    for order in args.orders:
        cap = (coefficients.shape[-1]-1)//2
        if order < 0 or 2*order > cap:
            parser.error('orders must be nonnegative and covered by the projection cap')
        c = coefficients[:, cap-2*order:cap+2*order+1, cap-2*order:cap+2*order+1]
        c_jax = jnp.asarray(c)
        adapter = ThinGapEnergyModel(order, rows=2)
        # Uniformly scaling all layer permittivities leaves the cell potential
        # unchanged. Its energy and material rows scale, including background.
        # This isolates optical closure/propagation cost from a new cell solve.
        def scaled(scale, xp):
            base = c if xp is np else c_jax
            return xp.concatenate([base[:4], scale*base[4:]])
        controls = [polarized_powers(order, projected_matrices(scaled(scale, np), order))[0]
                    for scale in (1., 1.+1e-5, 1.-1e-5)]
        secant = (controls[1]-controls[2])/2e-5
        scale = jnp.asarray(1.)
        for method in ('full', 'mirror'):
            def objective(scale):
                field = scaled(scale, jnp)
                if method == 'mirror':
                    return mirror_doubling_powers(field, order, jnp)[0]
                slots, _ = adapter._matrices_from_coefficients(field, {}, jnp)
                return doubling_powers(order, slots, jnp)[0]
            derivative = jax.jacfwd(objective)
            function = jax.jit(lambda scale: (objective(scale), derivative(scale)))
            start = time.monotonic()
            value, gradient = function(scale)
            value.block_until_ready()
            gradient.block_until_ready()
            compile_seconds = time.monotonic()-start
            np.testing.assert_allclose(value, controls[0], atol=2e-10, rtol=0)
            np.testing.assert_allclose(gradient, secant, atol=2e-9, rtol=3e-6)
            timings = []
            for _ in range(args.repeats):
                start = time.monotonic()
                value, gradient = function(scale)
                value.block_until_ready()
                gradient.block_until_ready()
                timings.append(time.monotonic()-start)
            print(json.dumps(dict(kind='mirror_doubling_jax_timing', order=order, method=method,
                                  parameter='all_layer_permittivity_scale', projection=metadata,
                                  T=np.asarray(value).tolist(), derivative=np.asarray(gradient).tolist(),
                                  modal_secant=secant.tolist(), compile_and_first_seconds=compile_seconds,
                                  median_seconds=float(np.median(timings)), samples_seconds=timings)), flush=True)
            jax.clear_caches()


if __name__ == '__main__':
    main()
