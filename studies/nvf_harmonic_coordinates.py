"""Material-harmonic coordinates in the partitioned thin-gap study.

Within a fixed x element, the nodal coordinate is the cumulative integral of
rho=1/epsilon divided by its whole-element integral. Its derivative follows a
moving material jump without moving the mesh. Material-jump telescoping keeps
redundant partitions neutral, including their physical-parameter derivatives.
This is an opt-in research model, not a production factorization method.
"""
import argparse
import json

import numpy as np

from studies.nvf_partitioned_gap_energy import PartitionedGapEnergyModel, interval_moments, moment_matrices
from studies.nvf_thin_gap_energy import _line_mass, _line_stiffness


def _polynomials(weight, offset, slope, xp):
    """Coefficients in physical local r for weight * coordinate**(0,1,2)."""
    zero = xp.zeros_like(weight)
    return xp.stack([xp.stack([weight, zero, zero], axis=-1),
                     xp.stack([weight*offset, weight*slope, zero], axis=-1),
                     xp.stack([weight*offset**2, 2*weight*offset*slope, weight*slope**2], axis=-1)], axis=-2)


def coordinate_integrals(nodes, lengths, seams, values, inside, modes, xp=np):
    """Exact mass/derivative/stiffness moments of the harmonic nodal basis.

    Each polynomial jump is integrated on its suffix. An equal-material seam
    changes neither polynomial coefficients nor reciprocal material moments.
    This is algebraic cancellation, with no material-equality predicate.
    """
    raw = interval_moments(nodes, lengths, xp.zeros_like(nodes), xp.ones_like(nodes), modes, xp)
    base = xp.where(inside, values[0], 1.)
    rho_base = 1/base
    average = rho_base
    locations, increments = [], []
    for i in range(len(seams)):
        location = xp.clip((seams[i]-nodes)/lengths, 0., 1.)
        jump = xp.where(inside, 1/values[i+1]-1/values[i], 0.)
        average = average+jump*(1-location)
        locations.append(location)
        increments.append(interval_moments(nodes, lengths, location, xp.ones_like(nodes), modes, xp))

    def integrate(moments, polynomials):
        return xp.einsum('mei,eji->mej', moments, polynomials)

    offset = xp.zeros_like(nodes)
    slope = rho_base/average
    body_poly = _polynomials(base, offset, slope, xp)
    gap_poly = _polynomials(xp.ones_like(base), offset, slope, xp)
    body_mass, gap_mass = integrate(raw, body_poly), integrate(raw, gap_poly)
    rho_moment, rho_squared = raw[..., 0]*rho_base, raw[..., 0]*rho_base**2
    for i, (location, increment) in enumerate(zip(locations, increments)):
        epsilon = xp.where(inside, values[i+1], 1.)
        previous = xp.where(inside, values[i], 1.)
        jump = 1/epsilon-1/previous
        offset = offset-jump*location/average
        slope = (1/epsilon)/average
        next_body = _polynomials(epsilon, offset, slope, xp)
        next_gap = _polynomials(xp.ones_like(epsilon), offset, slope, xp)
        body_mass = body_mass+integrate(increment, next_body-body_poly)
        gap_mass = gap_mass+integrate(increment, next_gap-gap_poly)
        rho_moment = rho_moment+jump*increment[..., 0]
        rho_squared = rho_squared+(1/epsilon**2-1/previous**2)*increment[..., 0]
        body_poly, gap_poly = next_body, next_gap

    scale = average*lengths
    body = dict(mass=body_mass, derivative=raw[..., 0]/scale,
                stiffness=rho_moment/scale**2)
    gap = dict(mass=gap_mass, derivative=rho_moment/scale,
               stiffness=rho_squared/scale**2)
    for pack in (body, gap):
        pack.update(lift=pack['derivative']*lengths,
                    lift_squared=pack['stiffness']*lengths**2,
                    lift_derivative=pack['stiffness']*lengths)
    # Reference arithmetic with the same grouping makes the corrections exactly
    # zero for an unpartitioned material, including its smallest mesh elements.
    base_scale = rho_base*lengths
    body_base = dict(mass=raw*base[None, :, None], derivative=raw[..., 0]/base_scale,
                     stiffness=raw[..., 0]*rho_base/base_scale**2)
    gap_base = dict(mass=raw, derivative=raw[..., 0]*rho_base/base_scale,
                    stiffness=raw[..., 0]*rho_base**2/base_scale**2)
    return body, gap, body_base, gap_base, base


class HarmonicCoordinateGapModel(PartitionedGapEnergyModel):
    """A fixed mesh with a live material-defined coordinate in each element.

    The geometry/profile/shape API is inherited from PartitionedGapEnergyModel.
    The field jump is represented inside an element by the coordinate itself;
    no material seam is promoted to a mesh node or discarded at equality.
    """

    def coefficients(self, gap, epsilon=12.25, width=350., xp=np):
        g, edges, values, nodes, lengths, valid = self._profile_geometry(gap, epsilon, width, xp)
        body, background, body_base, gap_base, base_eps = coordinate_integrals(
            nodes, lengths, edges[1:-1], values, np.arange(self.nx)//self.per_region == 1,
            self.modes, xp)
        center = 2*self.order

        def correction(pack, base):
            mass, _, load, _ = (v[center].real for v in moment_matrices(pack['mass']-base['mass'], lengths, xp))
            stiffness = _line_stiffness((pack['stiffness'][center]-base['stiffness'][center]).real, xp)
            derivative = (pack['derivative'][center]-base['derivative'][center]).real
            return mass, stiffness, load, xp.roll(derivative, 1)-derivative

        mb_delta, kb_delta, lb_delta, _ = correction(background, gap_base)
        me_delta, ke_delta, le_delta, _ = correction(body, body_base)
        mb = _line_mass(lengths, xp)+mb_delta
        kb = _line_stiffness(1/lengths, xp)+kb_delta
        lb = (lengths+xp.roll(lengths, 1))/2+lb_delta
        weighted = lengths*base_eps
        me = _line_mass(weighted, xp)+me_delta
        ke = _line_stiffness(base_eps/lengths, xp)+ke_delta
        le = (weighted+xp.roll(weighted, 1))/2+le_delta
        # The affine lift is L=length*coordinate', so its forcing pairs with
        # coordinate' through length times the element stiffness integral.
        body_force = base_eps+(body['stiffness'][center]-body_base['stiffness'][center]).real*lengths
        gap_force = (background['stiffness'][center]-gap_base['stiffness'][center]).real*lengths
        de, dg = xp.roll(body_force, 1)-body_force, xp.roll(gap_force, 1)-gap_force
        v, a, residual = self._solve_cell(g, me, ke, le, de, mb, kb, lb, xp, gap_derivative_load=dg)
        v, a = xp.where(valid, v, xp.nan), xp.where(valid, a, xp.nan)
        return self._project_cell(g, lengths, background, body, v, a, residual, xp)


def main(*, model_class=HarmonicCoordinateGapModel, kind='harmonic_coordinate', gap_enrichment=False,
         mesh_grading=False, body_enrichment=False, flux_enrichment=False):
    parser = argparse.ArgumentParser(description=model_class.__doc__ or __doc__)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--gaps', type=float, nargs='+', default=[0., 17.])
    parser.add_argument('--seams', type=float, nargs='+', default=[0., 31.])
    parser.add_argument('--values', type=float, nargs=2, default=[4., 9.])
    parser.add_argument('--width', type=float, default=350.)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--subdivisions', type=int, default=1)
    parser.add_argument('--y-subdivisions', type=int, default=1)
    parser.add_argument('--edge-fraction', type=float, default=1e-4)
    parser.add_argument('--static', action='store_true', help='compare the closed static laminate energy and derivative')
    if gap_enrichment:
        parser.add_argument('--gap-degree', type=int, default=4)
    if mesh_grading:
        parser.add_argument('--grading-strength', type=float, default=8.)
        parser.add_argument('--grading-radius', type=float, default=.01)
    if body_enrichment:
        parser.add_argument('--body-degree', type=int, default=2)
    if flux_enrichment:
        parser.add_argument('--flux-degree', type=int, default=2)
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    settings = dict(rows=args.rows, edge_fraction=args.edge_fraction, x_subdivisions=args.subdivisions,
                    y_subdivisions=args.y_subdivisions)
    if gap_enrichment:
        settings['gap_degree'] = args.gap_degree
    if mesh_grading:
        settings.update(grading_strength=args.grading_strength, grading_radius=args.grading_radius)
    if body_enrichment:
        settings['body_degree'] = args.body_degree
    if flux_enrichment:
        settings['flux_degree'] = args.flux_degree
    model = model_class(0 if args.static else args.order, **settings)
    report = dict(y_subdivisions=args.y_subdivisions)
    if gap_enrichment:
        report['gap_degree'] = args.gap_degree
    if mesh_grading:
        report.update(grading_strength=args.grading_strength, grading_radius=args.grading_radius)
    if body_enrichment:
        report['body_degree'] = args.body_degree
    if flux_enrichment:
        report['flux_degree'] = args.flux_degree
    if args.static:
        objective = jax.jit(jax.value_and_grad(lambda seam: model.coefficients(
            0., dict(seams=jnp.array([seam]), values=jnp.asarray(args.values)), args.width, jnp)[0][4, 0, 0].real))
        e0, e1 = args.values
        for seam in args.seams:
            value, derivative = objective(seam)
            if not np.isfinite([value, derivative]).all():
                raise FloatingPointError('nonfinite data in static coordinate study')
            expected = 1/(1-args.width/700+(args.width/2+seam)/700/e0+(args.width/2-seam)/700/e1)
            slope = -expected**2*(1/e0-1/e1)/700
            print(json.dumps(dict(kind=kind+'_static', seam=seam, width=args.width, values=args.values, **report,
                                  energy=float(value), derivative=float(derivative),
                                  exact_energy=expected, exact_derivative=slope)), flush=True)
        return
    for pol in ('p', 's'):
        def objective(p):
            return model.transmission(p[0], dict(seams=p[1:2], values=p[2:]), args.width, jnp, polarization=pol)['T']
        run = jax.jit(jax.value_and_grad(objective))
        for gap in args.gaps:
            for seam in args.seams:
                value, gradient = run(jnp.array([gap, seam, *args.values]))
                if not np.isfinite(np.r_[value, gradient]).all():
                    raise FloatingPointError('nonfinite data in harmonic-coordinate study')
                h = 1e-4
                def reference(s):
                    return model.transmission(gap, dict(seams=[s], values=args.values),
                                                args.width, polarization=pol)['T']
                center = reference(seam)
                left, right = (center-reference(seam-h))/h, (reference(seam+h)-center)/h
                print(json.dumps(dict(kind=kind, order=args.order, width=args.width, **report,
                                      polarization=pol, gap=gap, seam=seam, T=float(value),
                                      values=args.values, rows=args.rows, subdivisions=args.subdivisions,
                                      edge_fraction=args.edge_fraction, numpy_T=float(center),
                                      seam_step=h, seam_secants=[float(left), float(right)],
                                      gradient_gap_seam_left_right=np.asarray(gradient).tolist())), flush=True)
        jax.clear_caches()


if __name__ == '__main__':
    main()
