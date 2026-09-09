"""Rigid pose and live incidence for the conforming rectangular NVF study.

Rotation moves the entire pattern AND its square periodic lattice in the
laboratory plane. Local rectangle dimensions and material seams remain local
coordinates. This does not rotate independent primitives inside a fixed cell.
"""
import numpy as np

from studies.nvf_harmonic_energy import PERIOD, THETA
from studies.nvf_polynomial_body import PolynomialBodyGapModel
from studies.nvf_scattering_doubling import doubling_response


def translated_coefficients(coefficients, center=(0., 0.), rotation=0., xp=np):
    """Translate by a laboratory vector while retaining local lattice axes."""
    center = xp.asarray(center)
    if (center.shape != (2,) or xp.asarray(rotation).ndim != 0
            or xp.iscomplexobj(center) or xp.iscomplexobj(rotation)):
        raise ValueError('center must have two real components and rotation must be a real scalar')
    valid = xp.all(xp.isfinite(center)) & xp.isfinite(rotation)
    if xp is np and not valid:
        raise ValueError('center and rotation must be finite')
    cosine, sine = xp.cos(rotation), xp.sin(rotation)
    local_x = cosine*center[0]+sine*center[1]
    local_y = -sine*center[0]+cosine*center[1]
    cap = (coefficients.shape[-1]-1)//2
    modes = xp.arange(-cap, cap+1)
    phase = xp.exp(-2j*np.pi/PERIOD*(modes[:, None]*local_y+modes[None]*local_x))
    return xp.where(valid, coefficients*phase[None], xp.nan)


class PosedGapModel:
    """The same live polynomial-body cell, observed at a rigid lattice pose.

    Angles are radians. Center is a laboratory-plane translation in nm.
    Azimuth is the incident-wave azimuth in laboratory coordinates; rotation
    is the laboratory orientation of the cell's local x axis. A signed polar
    angle theta makes the polarization frame continuous at normal incidence.
    """

    def __init__(self, order, **cell_settings):
        self.cell = PolynomialBodyGapModel(order, **cell_settings)
        self.order = order

    def response(self, gap, epsilon=12.25, width=350., xp=np, *, center=(0., 0.), rotation=0.,
                 theta=THETA, azimuth=0., **settings):
        coefficients, metadata = self.cell.coefficients(gap, epsilon, width, xp)
        coefficients = translated_coefficients(coefficients, center, rotation, xp)
        slots, metadata = self.cell._matrices_from_coefficients(coefficients, metadata, xp)
        response = doubling_response(self.order, slots, xp, theta=theta, azimuth=azimuth-rotation, **settings)
        return dict(**response, **metadata)

    def transmission(self, *args, polarization='p', **kwargs):
        if polarization not in ('p', 's', 'both'):
            raise ValueError('polarization must be p, s or both')
        response = self.response(*args, **kwargs)
        response = {key: value for key, value in response.items() if key not in ('reflected', 'transmitted')}
        if polarization != 'both':
            index = 0 if polarization == 'p' else 1
            response['T'], response['R'] = response['T'][index], response['R'][index]
        return response


def contact_comparison(model, epsilon, width=350., *, center=(0., 0.), rotation=0., theta=THETA, azimuth=0.):
    """Report a finite-order contact control, without certifying convergence."""
    from studies.nvf_gap_convergence import profile_li
    from studies.nvf_harmonic_energy import polarized_powers

    profile = epsilon if isinstance(epsilon, dict) else dict(seams=[], values=[epsilon])
    gaps, candidate, control = [0., 1e-4, 5e-5], [], []
    for gap in gaps:
        candidate.append(model.transmission(gap, epsilon, width, center=center, rotation=rotation,
                                             theta=theta, azimuth=azimuth, polarization='both')['T'])
        slots = profile_li(model.order, gap, width, profile['seams'], profile['values'])
        control.append(polarized_powers(model.order, slots, theta=theta, azimuth=azimuth-rotation)[0])
    candidate, control = np.asarray(candidate), np.asarray(control)
    def slope(values):
        return 2*(values[2]-values[0])/gaps[2]-(values[1]-values[0])/gaps[1]
    return dict(gaps=gaps, candidate_T=candidate.tolist(), cartesian_T=control.tolist(),
                candidate_slope=slope(candidate).tolist(), cartesian_slope=slope(control).tolist(),
                slope_difference=(slope(candidate)-slope(control)).tolist())


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--gap', type=float, default=17.)
    parser.add_argument('--width', type=float, default=350.)
    parser.add_argument('--seams', type=float, nargs='*', default=[31.])
    parser.add_argument('--values', type=float, nargs='+', default=[4., 9.])
    parser.add_argument('--center', type=float, nargs=2, default=[17., -21.])
    parser.add_argument('--rotation', type=float, default=.31)
    parser.add_argument('--theta', type=float, default=THETA)
    parser.add_argument('--azimuth', type=float, default=.47)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--subdivisions', type=int, default=1)
    parser.add_argument('--body-degree', type=int, default=4)
    parser.add_argument('--gap-degree', type=int, default=6)
    parser.add_argument('--edge-fraction', type=float, default=1e-5)
    parser.add_argument('--grading-strength', type=float, default=8.)
    parser.add_argument('--grading-radius', type=float, default=.01)
    parser.add_argument('--flux-basis', choices=('global', 'local'), default='global')
    parser.add_argument('--jax', action='store_true')
    parser.add_argument('--contact-check', action='store_true', help='also report the finite-order Cartesian contact comparison')
    args = parser.parse_args()
    model = PosedGapModel(args.order, rows=args.rows, x_subdivisions=args.subdivisions,
                          body_degree=args.body_degree, gap_degree=args.gap_degree, edge_fraction=args.edge_fraction,
                          grading_strength=args.grading_strength, grading_radius=args.grading_radius,
                          flux_basis=args.flux_basis)
    parameters = np.array([args.gap, args.width, *args.center, args.rotation, args.theta, args.azimuth,
                           *args.seams, *args.values])
    seam_count = len(args.seams)
    def evaluate(p, xp):
        return model.transmission(p[0], dict(seams=p[7:7+seam_count], values=p[7+seam_count:]), p[1], xp,
                                   center=p[2:4], rotation=p[4], theta=p[5], azimuth=p[6], polarization='both')
    result = evaluate(parameters, np)
    record = dict(kind='posed_gap', settings=vars(args),
                  **{key: np.asarray(value).tolist() for key, value in result.items()})
    if args.jax:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        objective = lambda p: evaluate(p, jnp)['T']
        value = jax.jit(objective)(jnp.asarray(parameters))
        derivative = jax.jit(jax.jacrev(objective))(jnp.asarray(parameters))
        if not (np.isfinite(value).all() and np.isfinite(derivative).all()):
            raise FloatingPointError('nonfinite posed JIT value or derivative')
        np.testing.assert_allclose(value, result['T'], atol=2e-10, rtol=0)
        record.update(jit_T=np.asarray(value).tolist(), jacobian=np.asarray(derivative).tolist(),
                      parameters=['gap', 'width', 'center_x_lab', 'center_y_lab', 'lattice_rotation', 'theta',
                                  'lab_azimuth', *[f'seam_{i}' for i in range(seam_count)],
                                  *[f'epsilon_{i}' for i in range(len(args.values))]])
    if args.contact_check:
        record['contact_control'] = contact_comparison(model, dict(seams=args.seams, values=args.values), args.width,
                                                       center=args.center, rotation=args.rotation,
                                                       theta=args.theta, azimuth=args.azimuth)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
