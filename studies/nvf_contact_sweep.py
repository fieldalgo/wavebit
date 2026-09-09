"""Independent finite-order contact controls across material profiles and incidence.

The same maximal cell projection supplies every retained Fourier space. A
separate Cartesian factorization supplies each one-sided control. Agreement
at finite order is a contact gate, not an optical convergence certificate.
"""
import argparse
import json
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning

from studies.nvf_gap_convergence import profile_li, projected_matrices
from studies.nvf_harmonic_energy import polarized_powers
from studies.nvf_polynomial_body import PolynomialBodyGapModel
from studies.nvf_scattering_doubling import doubling_powers


PROFILES = {
    'mixed': dict(seams=[31.], values=[4., 9.]),
    'strong': dict(seams=[-73.], values=[2.25, 12.25]),
    'multiple': dict(seams=[-73., 31.], values=[4., 9., 6.25]),
    'equal': dict(seams=[31.], values=[6.25, 6.25]),
}


def sweep(orders=(2, 4, 7), profiles=('mixed', 'strong', 'multiple'), azimuths=(0., .37, 1.1),
          *, width=350., theta=.2, step=1e-4, **cell_settings):
    """Yield gate results; never route the candidate through its control."""
    if (not orders or any(not isinstance(order, int) or order < 0 for order in orders)
            or not np.isfinite(step) or step <= 0):
        raise ValueError('orders must be nonnegative integers and step must be positive')
    gaps = np.array([0., step, step/2])
    model = PolynomialBodyGapModel(max(orders), **cell_settings)
    def slope(values):
        return 2*(values[2]-values[0])/(step/2)-(values[1]-values[0])/step

    for name in profiles:
        profile = PROFILES[name]
        # Retain one maximal Fourier projection per gap, not a separately
        # optimized cell for each outer order or angle. Warnings are failures.
        with warnings.catch_warnings():
            warnings.simplefilter('error', LinAlgWarning)
            fields = [model.coefficients(gap, profile, width) for gap in gaps]
        for order in orders:
            candidate_slots = [projected_matrices(field[0], order) for field in fields]
            control_slots = [profile_li(order, gap, width, **profile) for gap in gaps]
            for azimuth in azimuths:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', LinAlgWarning)
                    candidates = [doubling_powers(order, slots, theta=theta, azimuth=azimuth)
                                  for slots in candidate_slots]
                    controls = [polarized_powers(order, slots, theta=theta, azimuth=azimuth)
                                for slots in control_slots]
                t = np.array([result[0] for result in candidates])
                control = np.array([result[0] for result in controls])
                energy = max(float(np.max(abs(result[0]+result[1]-1)))
                             for result in [*candidates, *controls])
                derivative, expected = slope(t), slope(control)
                difference = derivative-expected
                contact_error = float(np.max(abs(t[0]-control[0])))
                finite = np.isfinite(np.r_[t.ravel(), control.ravel(), difference]).all()
                passed = bool(finite and energy < 1e-9 and contact_error < 2e-10
                              and np.max(abs(difference)) < 1e-7)
                yield dict(kind='nvf_contact_sweep', profile=name, **profile, width=width, theta=theta,
                           azimuth=azimuth, order=order, projection_order=max(orders), gaps=gaps.tolist(),
                           cell_settings=cell_settings, degrees_of_freedom=fields[0][1]['degrees_of_freedom'],
                           cell_residual=max(float(field[1]['cell_residual']) for field in fields),
                           candidate_T=t.tolist(), cartesian_T=control.tolist(),
                           candidate_slope=derivative.tolist(), cartesian_slope=expected.tolist(),
                           slope_difference=difference.tolist(), contact_error=contact_error,
                           energy_error=energy, contact_gate=passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=[2, 4, 7])
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=['mixed', 'strong', 'multiple'])
    parser.add_argument('--azimuths', nargs='+', type=float, default=[0., .37, 1.1])
    parser.add_argument('--theta', type=float, default=.2)
    parser.add_argument('--width', type=float, default=350.)
    parser.add_argument('--step', type=float, default=1e-4)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--body-degree', type=int, default=1)
    parser.add_argument('--subdivisions', type=int, default=6)
    parser.add_argument('--gap-degree', type=int, default=6)
    parser.add_argument('--edge-fraction', type=float, default=1e-5)
    parser.add_argument('--grading-radius', type=float, default=.0001)
    parser.add_argument('--grading-strength', type=float, default=8.)
    args = parser.parse_args()
    passed = True
    for record in sweep(args.orders, args.profiles, args.azimuths, theta=args.theta, width=args.width, step=args.step,
                        rows=args.rows, body_degree=args.body_degree, x_subdivisions=args.subdivisions,
                        gap_degree=args.gap_degree, edge_fraction=args.edge_fraction,
                        grading_radius=args.grading_radius, grading_strength=args.grading_strength, flux_basis='local'):
        print(json.dumps(record), flush=True)
        passed &= record['contact_gate']
    if not passed:
        raise SystemExit('At least one finite-order contact gate is unresolved; see the recorded differences.')


if __name__ == '__main__':
    main()
