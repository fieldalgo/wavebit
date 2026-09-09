"""Eigenvector-free single-layer propagation for the NVF research instruments.

A bounded Taylor transfer on a short slice is converted to a scattering
matrix, then composed by doubling. The reference ports have unit positive
flux. No large growing transfer matrix, modal branch selection or custom
eigenvector derivative enters this opt-in NumPy/JAX calculation.
"""
import math

import numpy as np
from scipy.linalg import solve

from studies.nvf_fourier_orders import order_grid
from studies.nvf_harmonic_energy import PERIOD, THETA, THICKNESS, WAVELENGTH
from studies.nvf_polynomial_body import PolynomialBodyGapModel


def incident_orders(order, theta=THETA, azimuth=0., xp=np):
    """Live transverse wavevectors in the local square-lattice coordinates.

    Theta is signed and lies strictly between -pi/2 and pi/2. The incident
    polarization frame follows azimuth continuously through normal incidence.
    """
    if xp is not np and xp.asarray(0.).dtype != np.dtype('float64'):
        raise ValueError('the scattering-doubling study requires JAX x64')
    if (xp.asarray(theta).ndim != 0 or xp.asarray(azimuth).ndim != 0
            or xp.iscomplexobj(theta) or xp.iscomplexobj(azimuth)):
        raise ValueError('incidence angles must be real scalars')
    valid = xp.isfinite(theta) & xp.isfinite(azimuth) & (xp.abs(theta) < np.pi/2)
    if xp is np and not valid:
        raise ValueError('incidence angles must be finite with abs(theta) < pi/2')
    n, m = order_grid(order)
    kx = xp.sin(theta)*xp.cos(azimuth)+WAVELENGTH/PERIOD*m.ravel()
    ky = xp.sin(theta)*xp.sin(azimuth)+WAVELENGTH/PERIOD*n.ravel()
    return xp.where(valid, kx, xp.nan), xp.where(valid, ky, xp.nan)


def maxwell_factors(order, slots, xp=np, *, theta=THETA, azimuth=0.):
    """The existing nonmagnetic first-order Maxwell factors and both products."""
    kx, ky = incident_orders(order, theta, azimuth, xp)
    identity = xp.eye(len(kx), dtype=xp.complex128)
    xx, yy, z, xy, yx = slots
    b = kx[:, None]*z*kx[None]-identity
    d = ky[:, None]*z*ky[None]-identity
    c, e = kx[:, None]*z*ky[None], ky[:, None]*z*kx[None]
    left = xp.block([[-c, b], [-d, e]])
    right = xp.block([[-xp.diag(kx*ky)+yx, xp.diag(kx*kx)-yy],
                       [xx-xp.diag(ky*ky), xp.diag(ky*kx)-xy]])
    # Expanded products avoid subtracting O(k^4) terms to obtain O(k^2).
    lr = xp.block([[xp.diag(ky*ky)+b@xx-c@yx, c@yy-xp.diag(kx*ky)-b@xy],
                    [e@xx-xp.diag(ky*kx)-d@yx, xp.diag(kx*kx)+d@yy-e@xy]])
    upper = yy*ky[None]-yx*kx[None]
    lower = xx*kx[None]-xy*ky[None]
    rl = xp.block([[xp.diag(kx*kx)-yy+(upper@z)*ky[None],
                     xp.diag(kx*ky)-yx-(upper@z)*kx[None]],
                    [xp.diag(ky*kx)-xy-(lower@z)*ky[None],
                     xp.diag(ky*ky)-xx+(lower@z)*kx[None]]])
    return left, right, lr, rl


def star(left, right, xp=np):
    """Cascade two scattering matrices in [top-out, bottom-out] ordering."""
    r1, u1, t1, b1 = left
    r2, u2, t2, b2 = right
    linear_solve = solve if xp is np else xp.linalg.solve
    identity = xp.eye(len(r1), dtype=xp.complex128)
    down = linear_solve(identity-b1@r2, t1)
    up = linear_solve(identity-r2@b1, u2)
    return r1+u1@r2@down, u1@up, t2@down, b2+t2@b1@up


def reference_coordinates(na, nb):
    """Map magnetic [Hx_b, Hy_a] to electric [Ex_a, Ey_b] flux coordinates."""
    return np.r_[nb+np.arange(na), np.arange(nb)], np.r_[-1j*np.ones(na), 1j*np.ones(nb)]


def canonical_factors(left, right, lr, rl, na, nb, xp=np):
    """Use Hc=Y^{-1}H so both canonical fields have the electric ordering."""
    permutation, phase = reference_coordinates(na, nb)
    phase = xp.asarray(phase)
    return (left[:, permutation]/phase[None], phase[:, None]*right[permutation], lr,
            phase[:, None]*rl[permutation][:, permutation]/phase[None])


def symmetric_scattering(left, right, lr, rl, xp=np, *, doublings=6, terms=12,
                         thickness=THICKNESS, progress=None):
    """Entire short-slice series and doubling in unit-flux coordinates.

    The off-diagonal Maxwell generator is invariant under reversing z and H.
    With identical reference ports this gives equal top/bottom reflections
    and equal forward/backward transmissions, even for noncommuting factors.
    """
    if not isinstance(doublings, int) or not 0 <= doublings <= 16:
        raise ValueError('doublings must be an integer between zero and sixteen')
    if not isinstance(terms, int) or not 3 <= terms <= 18:
        raise ValueError('terms must be an integer between three and eighteen')
    if xp is not np and xp.asarray(0.).dtype != np.dtype('float64'):
        raise ValueError('the scattering-doubling study requires JAX x64')
    if xp is np and (not np.isfinite(thickness) or thickness < 0):
        raise ValueError('thickness must be finite and nonnegative')
    linear_solve = solve if xp is np else xp.linalg.solve
    size = len(left)
    identity = xp.eye(size, dtype=xp.complex128)
    tau = 2*np.pi/WAVELENGTH*thickness/2**doublings
    p, q = tau*tau*lr, tau*tau*rl
    def norm(matrix):
        return xp.max(xp.sum(xp.abs(matrix), axis=1))
    alpha = xp.maximum(norm(p), norm(q))
    ratio = alpha/((2*terms+3)*(2*terms+4))
    bound = ((1+xp.abs(tau)*xp.maximum(norm(left), norm(right)))
             *alpha**(terms+1)/float(math.factorial(2*terms+2))/xp.maximum(1-ratio, 1e-300))
    valid = ((ratio < 1) & xp.isfinite(bound) & (bound*2**doublings < 1e-13)
             & xp.isfinite(thickness) & (thickness >= 0))
    if xp is np and not valid:
        raise FloatingPointError('Taylor tail is unresolved; increase doublings or terms')
    ep, eq, op, oq = identity, identity, identity, identity
    term_p, term_q = identity, identity
    for k in range(1, terms+1):
        term_p, term_q = term_p@p/((2*k-1)*(2*k)), term_q@q/((2*k-1)*(2*k))
        ep, eq = ep+term_p, eq+term_q
        op, oq = op+term_p/(2*k+1), oq+term_q/(2*k+1)
    f00, f11, f01, f10 = ep, eq, -tau*left@oq, -tau*right@op
    del left, right, lr, rl, p, q, ep, eq, op, oq, term_p, term_q
    c, d = (f00+f01-f10-f11)/2, (f00-f01-f10+f11)/2
    del f00, f11, f01, f10
    solution = linear_solve(d, xp.concatenate([identity, -c], axis=1))
    del c, d
    transmission, reflection = solution[:, :size], solution[:, size:]
    transmission, reflection = [xp.where(valid, matrix, xp.nan) for matrix in (transmission, reflection)]
    del solution
    if progress:
        progress('slice_ready', taylor_tail_bound=float(bound), doublings=doublings, terms=terms)
    for index in range(doublings):
        common = linear_solve(identity-reflection@reflection, transmission)
        reflection, transmission = (reflection+(transmission@reflection)@common, transmission@common)
        if progress:
            progress('doubling_ready', doubling=index+1)
    return (reflection, transmission, transmission, reflection), bound


def slice_scattering(order, slots, xp=np, *, doublings=6, terms=12, thickness=THICKNESS,
                     theta=THETA, azimuth=0.):
    """Short-slice exponential, with a conservative local Taylor-tail check."""
    factors = maxwell_factors(order, slots, xp, theta=theta, azimuth=azimuth)
    half = len(factors[0])//2
    factors = canonical_factors(*factors, half, half, xp)
    scattering, bound = symmetric_scattering(*factors, xp,
                                              doublings=doublings, terms=terms, thickness=thickness)
    small_identity, zero = xp.eye(half, dtype=xp.complex128), xp.zeros((half, half), dtype=xp.complex128)
    reference_y = 1j*xp.block([[zero, -small_identity], [small_identity, zero]])
    return scattering, reference_y, bound


def doubling_response(order, slots, xp=np, *, doublings=6, terms=12, thickness=THICKNESS,
                       theta=THETA, azimuth=0.):
    """Complex outgoing amplitudes and p/s powers, with live incidence angles.

    Amplitude rows are [s orders, p orders]; columns are p and s incidence.
    Every order, including evanescent orders, is retained in the amplitudes.
    """
    linear_solve = solve if xp is np else xp.linalg.solve
    kx, ky = incident_orders(order, theta, azimuth, xp)
    scattering, reference_y, bound = slice_scattering(order, slots, xp, doublings=doublings,
                                                       terms=terms, thickness=thickness, theta=theta, azimuth=azimuth)
    r, up, down, bottom = scattering
    n, m = order_grid(order)
    count = len(kx)
    identity = xp.eye(count, dtype=xp.complex128)
    origin = (n.ravel() == 0) & (m.ravel() == 0)
    # atan2(0,0) and sqrt(kx^2+ky^2) have undefined derivatives at normal
    # incidence. The central port uses the chosen incident frame instead.
    transverse = xp.sqrt(xp.where(origin, 1., kx*kx+ky*ky))
    co = xp.where(origin, xp.cos(azimuth), kx/transverse)
    si = xp.where(origin, xp.sin(azimuth), ky/transverse)
    def exterior(epsilon):
        kz = xp.sqrt(xp.asarray(epsilon-kx*kx-ky*ky, dtype=xp.complex128))
        valid = xp.all(xp.isfinite(kz)) & (xp.min(xp.abs(kz)) > 1e-12)
        if xp is np and not valid:
            raise FloatingPointError('doubling requires finite nongrazing exterior orders')
        index = np.sqrt(epsilon)
        w = xp.block([[-identity*si, identity*(co*kz/index)],
                      [identity*co, identity*(si*kz/index)]])
        v = 1j*xp.block([[-identity*(co*kz), -identity*(si*index)],
                         [-identity*(si*kz), identity*(co*index)]])
        return xp.where(valid, (w+reference_y@v)/2, xp.nan), xp.where(valid, (w-reference_y@v)/2, xp.nan), kz
    pt, mt, kt = exterior(1.)
    pb, mb, kb = exterior(2.25)
    incoming = np.zeros((2*count, 2), complex)
    incoming[count+count//2, 0], incoming[count//2, 1] = 1., 1.
    incoming = xp.asarray(incoming)
    system = xp.block([[pt-r@mt, -up@mb], [-down@mt, pb-bottom@mb]])
    forcing = xp.concatenate([(r@pt-mt)@incoming, down@pt@incoming])
    result = linear_solve(system, forcing)
    reflected, transmitted = result[:2*count], result[2*count:]
    t = xp.sum(xp.tile(kb.real, 2)[:, None]*xp.abs(transmitted)**2, axis=0)/xp.cos(theta)
    r = xp.sum(xp.tile(kt.real, 2)[:, None]*xp.abs(reflected)**2, axis=0)/xp.cos(theta)
    error = xp.max(xp.abs(t+r-1))
    valid = xp.all(xp.isfinite(t)) & xp.all(xp.isfinite(r)) & (error < 1e-9)
    if xp is np and not valid:
        raise FloatingPointError(f'invalid doubling power balance: {error}')
    return dict(T=xp.where(valid, t, xp.nan), R=xp.where(valid, r, xp.nan),
                reflected=xp.where(valid, reflected, xp.nan), transmitted=xp.where(valid, transmitted, xp.nan),
                energy_error=error, taylor_tail_bound=bound)


def doubling_powers(order, slots, xp=np, **settings):
    """Return p/s T, R and diagnostics, with ordinary NumPy/JAX derivatives."""
    result = doubling_response(order, slots, xp, **settings)
    return result['T'], result['R'], {key: result[key] for key in ('energy_error', 'taylor_tail_bound')}


class ScatteringDoublingGapModel(PolynomialBodyGapModel):
    """Opt-in polynomial-body NVF model with the same live cell and closure.

    Only slab propagation changes. This adapter retains the parent's limited
    horizontal rectangle-tiling geometry; it is not a production dispatch.
    ``polarization='both'`` returns vectors in p, s order.
    """

    def _transmission(self, slots, metadata, xp, polarization, settings):
        if polarization not in ('p', 's', 'both'):
            raise ValueError('polarization must be p, s or both')
        t, r, diagnostics = doubling_powers(self.order, slots, xp, **settings)
        if polarization != 'both':
            index = 0 if polarization == 'p' else 1
            t, r = t[index], r[index]
        return dict(T=t, R=r, **metadata, **diagnostics)

    def transmission(self, gap, epsilon=12.25, width=350., xp=np, *, polarization='p', **settings):
        return self._transmission(*self.matrices(gap, epsilon, width, xp), xp, polarization, settings)

    def shape_transmission(self, shapes, xp=np, *, polarization='p', **settings):
        return self._transmission(*self.shape_matrices(shapes, xp), xp, polarization, settings)


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--gap', type=float, default=17.)
    parser.add_argument('--width', type=float, default=350.)
    parser.add_argument('--seams', type=float, nargs='*', default=[31.])
    parser.add_argument('--values', type=float, nargs='+', default=[4., 9.])
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--subdivisions', type=int, default=1)
    parser.add_argument('--body-degree', type=int, default=4)
    parser.add_argument('--gap-degree', type=int, default=6)
    parser.add_argument('--edge-fraction', type=float, default=1e-5)
    parser.add_argument('--doublings', type=int, default=6)
    parser.add_argument('--terms', type=int, default=12)
    parser.add_argument('--jax', action='store_true', help='also report JIT gap, seam and material derivatives')
    parser.add_argument('--mirror', action='store_true', help='restrict the same live closure and propagation to mirror sectors')
    args = parser.parse_args()
    model = ScatteringDoublingGapModel(args.order, rows=args.rows, x_subdivisions=args.subdivisions,
                                     body_degree=args.body_degree, gap_degree=args.gap_degree,
                                     edge_fraction=args.edge_fraction)
    parameters = np.array([args.gap, *args.seams, *args.values])
    seam_count = len(args.seams)
    def evaluate(parameters, xp):
        profile = dict(seams=parameters[1:1+seam_count], values=parameters[1+seam_count:])
        if args.mirror:
            from studies.nvf_mirror_doubling import mirror_doubling_powers
            coefficients, metadata = model.coefficients(parameters[0], profile, args.width, xp)
            t, r, diagnostics = mirror_doubling_powers(coefficients, args.order, xp,
                                                       doublings=args.doublings, terms=args.terms)
            return dict(T=t, R=r, **metadata, **diagnostics)
        return model.transmission(parameters[0], profile, args.width, xp, polarization='both',
                                  doublings=args.doublings, terms=args.terms)
    result = evaluate(parameters, np)
    record = dict(kind='scattering_doubling', settings=vars(args),
                  **{key: np.asarray(value).tolist() for key, value in result.items()})
    if args.jax:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        objective = lambda p: evaluate(p, jnp)['T']
        value = jax.jit(objective)(jnp.asarray(parameters))
        derivative = jax.jit(jax.jacrev(objective))(jnp.asarray(parameters))
        if not (np.isfinite(value).all() and np.isfinite(derivative).all()):
            raise FloatingPointError('nonfinite JIT power or derivative')
        if np.max(np.abs(np.asarray(value)-result['T'])) > 2e-10:
            raise FloatingPointError('NumPy and JIT power disagree')
        record.update(jit_T=np.asarray(value).tolist(), jacobian=np.asarray(derivative).tolist(),
                      parameters=['gap', *[f'seam_{i}' for i in range(seam_count)],
                                  *[f'epsilon_{i}' for i in range(len(args.values))]])
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
