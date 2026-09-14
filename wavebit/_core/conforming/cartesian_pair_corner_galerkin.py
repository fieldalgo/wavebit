"""Explicit real outer-corner enrichment of the complete pair NVF tensor."""
from numbers import Integral
import numpy as np
from .cartesian_pair_opening import real_array
from .cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from .cartesian_pair_seam_galerkin import _trace_change, _free_fourier_energy, _check_response
from .pair_corner_fields import PairCornerFields
from .pair_corner_assembly import augment
from .pair_corner_quadrature import quadrature as sample_quadrature
from .pair_corner_contact import solve as solve_contact
from .pair_corner_positive import response as solve_positive, verify as verify_energy
from .real_trace_coupling import coupled_positive


class CartesianPairCornerGalerkinGapCell:
    """Cubic/quartic pair potentials, outer-corner powers and real gap traces.

    This explicit NumPy family accepts the same fixed Cartesian opening plan
    and geometry as the polynomial pair cell. Gap and passive complex
    materials may vary together; geometry is fixed in a JVP. At zero gap the
    derivative is the right-opening derivative of this same enriched family.

    Exponents are distinct members of (0.6, 0.8), fixed independently of the
    materials. They enrich only the four outer body corners. The empty tuple
    is the original polynomial/continuous-trace space. All original physical
    guards apply. Choosing a field space or quadrature does not establish
    mesh/Fourier/optical convergence or change ordinary NVF routing.
    """

    def __init__(self, plan, order=1, *, degree=3, quadrature=48,
                 exponents=(.6, .8), coercivity_angle=0.):
        if (isinstance(degree, (bool, np.bool_)) or not isinstance(degree, Integral)
                or degree not in (3, 4)):
            raise ValueError('corner pair polynomial degree must be 3 or 4')
        self._base = CartesianPairPolynomialGalerkinGapCell(
            plan, order, degree=degree, coercivity_angle=coercivity_angle)
        self.order = self._base.order
        if (isinstance(quadrature, (bool, np.bool_)) or not isinstance(quadrature, Integral)
                or quadrature < max(24, 2*max(self.order)+4)):
            raise ValueError('quadrature must be an integer at least max(24, 2*max(order)+4)')
        self.exponents = tuple(exponents)
        PairCornerFields(self._base, self.exponents)
        self.quadrature = int(quadrature)

    def closure(self, *, gap, basis, widths, height, epsilon, background=1.,
                offset=(0., 0.), gap_tangent=1., epsilon_tangent=(0., 0.),
                background_tangent=0.):
        gap = real_array(gap, (), 'gap').item()
        settings = dict(gap=gap, basis=basis, widths=widths, height=height,
                        epsilon=epsilon, background=background, offset=offset,
                        gap_tangent=gap_tangent, epsilon_tangent=epsilon_tangent,
                        background_tangent=background_tangent)
        order = max(self.quadrature, 4*max(self.order)+16)
        if gap == 0:
            order = max(48, order)
            value, tangent, metadata = solve_contact(
                self._base, settings, quadrature=order, exponents=self.exponents)
        else:
            base = augment(self._base, settings, exponents=self.exponents, quadrature=order)
            transform = _trace_change(self._base, settings, order)
            quadrature = lambda split=True: sample_quadrature(
                self._base, base, settings, order, transform, split, False)
            normalized = lambda: sample_quadrature(
                self._base, base, settings, order, transform, True, True)
            weighted = lambda: sample_quadrature(
                self._base, base, settings, order, transform, True, True, weighted=True)
            free = _free_fourier_energy(self.order, settings)
            response = lambda q, dq, blocks: solve_positive(
                self._base, settings, base, order, weighted, free, q, dq, blocks)
            energy = lambda x, dx, u: verify_energy(
                self._base, settings, base, order, normalized, u, x, dx)
            value, tangent, metadata = coupled_positive(
                self._base, base, quadrature, free, self.order[1],
                response_solver=response, normalized_quadrature=normalized,
                energy_verifier=energy)
        _check_response(value, tangent, metadata)
        metadata.update(closure='real_coupled_pair_corner_galerkin', order=self.order,
                        refinement=self._base.plan.refinement, shift=self._base.plan.shift,
                        degree=self._base.degree, quadrature=self.quadrature,
                        integration_quadrature=order, gap=gap,
                        corner_exponents=self.exponents, corner_controls=(4, 6, 7, 9))
        return value, tangent, metadata
