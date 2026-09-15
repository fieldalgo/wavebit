"""Explicit real corner and junction enrichment of the complete pair NVF tensor."""
from numbers import Integral
import numpy as np
from .cartesian_pair_opening import real_array, validate_geometry
from ..factorization import ConvMatrices
from .cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from .pair_symmetric_mesh import SymmetricPairPolynomialCell
from .cartesian_pair_seam_galerkin import _trace_change, _free_fourier_energy, _check_response
from .pair_corner_fields import PairCornerFields
from .pair_corner_assembly import augment
from .pair_corner_quadrature import quadrature as sample_quadrature
from .pair_junction_quadrature import quadrature as junction_quadrature
from .pair_corner_contact import solve as solve_contact
from .pair_corner_positive import response as solve_positive, verify as verify_energy
from .real_trace_coupling import coupled_positive


class CartesianPairCornerGalerkinGapCell:
    """Cubic/quartic pair potentials, outer-corner powers and real gap traces.

    This explicit NumPy family accepts the same fixed Cartesian opening plan
    and geometry as the polynomial pair cell. Gap and passive complex
    materials may vary together; geometry is fixed in a JVP. At zero gap the
    derivative is the right-opening derivative of this same enriched family.

    Exponents are distinct members of (0.55, 0.6, 0.8, 0.95), fixed independently
    of the materials. The default (0.6, 0.8) keeps fifth-root evaluation.
    Including 0.55 or 0.95 uses stable normalized radial coordinates and
    twentieth-root integration with at least 20*max(degree,4) points. These powers
    enrich the four outer body corners. With junctions=True, both
    closing-gap material junctions are included as well. Angular orders 1 through 4
    add nested homogeneous bubble powers around each corner. The default 0
    preserves the original linear/quadratic angular space. The empty tuple
    is the original polynomial/continuous-trace space. All original physical
    guards apply. Choosing a field space or quadrature does not establish
    mesh/Fourier/optical convergence or change ordinary NVF routing.

    angular_quadrature optionally sets a separate angular Gauss order for
    two-dimensional corner and clipped-trace integrals. It must be at least
    max(12, 2*max(order)+4). None uses the effective radial order in both
    directions. Radial grading, one-dimensional moving-edge quadrature and
    geometry-coordinate preparation retain their established orders.

    oscillatory_radial=True appends the fixed real cosine/sine pair with
    radial power 19/20 and logarithmic frequency 1/2. It requires the default
    real exponents (0.6, 0.8), uses twentieth-root integration, and retains
    both real directions independently of the materials. The default False
    preserves the selected nonoscillatory field space. On the symmetric mesh,
    complete radial/angular coordinate expressions are evaluated with
    compensated arithmetic before rounding their values and gradients.

    mesh='symmetric' selects a reflection-invariant triangle trial space on
    shift-zero Cartesian plans. Exact polynomial identities remove redundant
    enrichment coordinates while retaining their complete potential span.
    Its physical equations are evaluated with the cell center at zero; an
    invertible Fourier translation returns every value and JVP slot to the
    caller's offset. This avoids cancellation in translated mesh coordinates.
    The default 'original' preserves the established mesh and field sampler.
    Positive states with amplified original equation defects receive complete
    pointwise weak refinement. Metadata reports its correction count as
    pointwise_refinement_steps; every original physical guard still applies.
    """

    _polynomial_degrees = (3, 4)

    def __init__(self, plan, order=1, *, degree=3, quadrature=48,
                 exponents=(.6, .8), junctions=False, angular_order=0, coercivity_angle=0.,
                 mesh='original', oscillatory_radial=False, angular_quadrature=None):
        if (isinstance(degree, (bool, np.bool_)) or not isinstance(degree, Integral)
                or degree not in self._polynomial_degrees):
            choices=' or '.join(map(str,self._polynomial_degrees))
            raise ValueError(f'corner pair polynomial degree must be {choices}')
        if not isinstance(mesh, str) or mesh not in ('original', 'symmetric'):
            raise ValueError("corner mesh must be 'original' or 'symmetric'")
        self.mesh=mesh
        base_type = (SymmetricPairPolynomialCell if mesh=='symmetric'
                     else CartesianPairPolynomialGalerkinGapCell)
        self._base = base_type(
            plan, order, degree=degree, coercivity_angle=coercivity_angle)
        self.order = self._base.order
        if (isinstance(quadrature, (bool, np.bool_)) or not isinstance(quadrature, Integral)
                or quadrature < max(24, 2*max(self.order)+4)):
            raise ValueError('quadrature must be an integer at least max(24, 2*max(order)+4)')
        if angular_quadrature is not None and (isinstance(angular_quadrature,(bool,np.bool_))
                or not isinstance(angular_quadrature,Integral)
                or angular_quadrature < max(12,2*max(self.order)+4)):
            raise ValueError('angular_quadrature must be None or an integer at least max(12, 2*max(order)+4)')
        self.angular_quadrature=None if angular_quadrature is None else int(angular_quadrature)
        self.exponents = tuple(exponents)
        fields=PairCornerFields(self._base, self.exponents, junctions=junctions, angular_order=angular_order,
                                oscillatory_radial=oscillatory_radial,angular_quadrature=self.angular_quadrature)
        self.oscillatory_radial=fields.oscillatory_radial
        self.root_order=fields.root_order
        if self.root_order==20 and quadrature<48:
            raise ValueError('expanded radial exponents require quadrature at least 48')
        self.angular_order=fields.angular_order
        self.junctions=fields.junctions
        self.corner_controls=fields.controls
        self._fields=fields
        self.quadrature = int(quadrature)

    def closure(self, *, gap, basis, widths, height, epsilon, background=1.,
                offset=(0., 0.), gap_tangent=1., epsilon_tangent=(0., 0.),
                background_tangent=0.):
        gap = real_array(gap, (), 'gap').item()
        settings = dict(gap=gap, basis=basis, widths=widths, height=height,
                        epsilon=epsilon, background=background, offset=offset,
                        gap_tangent=gap_tangent, epsilon_tangent=epsilon_tangent,
                        background_tangent=background_tangent)
        if self.mesh != 'symmetric':
            return self._closure_at_offset(settings)
        geometry=validate_geometry(basis,widths,height,offset)
        # Preserve validation of the caller's complete physical chart before
        # selecting an equivalent origin for the same translated cell.
        self._base._chart_geometry(gap,geometry)
        center_offset=-geometry['basis'].sum(axis=1)/2
        value,tangent,metadata=self._closure_at_offset(dict(settings,offset=center_offset))
        if not np.array_equal(geometry['offset'],center_offset):
            nx,ny=self.order
            modes=np.array([(i,j) for j in range(-ny,ny+1) for i in range(-nx,nx+1)])
            fractional=np.linalg.solve(geometry['basis'],geometry['offset'])
            # The half-cell shift is exactly +/-1 in Fourier coordinates.
            # Do not approximate it by evaluating exp(i*pi*integer).
            phase=(-1.)**modes.sum(axis=1)*np.exp(-2j*np.pi*(modes@fractional))
            change=phase[:,None]*phase[None,:].conj()
            value,tangent=(ConvMatrices(*(slot*change for slot in array))
                           for array in (value,tangent))
        metadata['coordinate_origin']='cell_center'
        _check_response(value,tangent,metadata)
        return value,tangent,metadata

    def _closure_at_offset(self,settings,*,prepared_fields=None):
        """Solve and independently verify the complete state at one origin."""
        gap=settings['gap']
        fields=self._fields if prepared_fields is None else prepared_fields
        order = max(self.quadrature, 4*max(self.order)+16, self.root_order*max(self._base.degree,4))
        if gap == 0:
            order = max(48, order)
            value, tangent, metadata = solve_contact(
                self._base, settings, quadrature=order, exponents=self.exponents, junctions=self.junctions,
                angular_order=self.angular_order,prepared_fields=fields,
                oscillatory_radial=self.oscillatory_radial,angular_quadrature=self.angular_quadrature)
        else:
            base = augment(self._base, settings, exponents=self.exponents, quadrature=order, junctions=self.junctions,
                           angular_order=self.angular_order,prepared_fields=fields,
                           oscillatory_radial=self.oscillatory_radial,angular_quadrature=self.angular_quadrature)
            sample = junction_quadrature if self.junctions or self.root_order==20 else sample_quadrature
            transform = _trace_change(self._base, settings, order)
            quadrature = lambda split=True: sample(
                self._base, base, settings, order, transform, split, False)
            normalized = lambda: sample(
                self._base, base, settings, order, transform, True, True)
            weighted = lambda: sample(
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
                        corner_exponents=self.exponents, corner_controls=self.corner_controls,
                        junctions=self.junctions, angular_order=self.angular_order,
                        radial_root_order=self.root_order)
        if self.angular_quadrature is not None:
            metadata['angular_integration_quadrature']=self.angular_quadrature
        if self.oscillatory_radial:
            metadata.update(oscillatory_radial=True,oscillatory_radial_power=.95,
                            oscillatory_radial_frequency=.5)
        metadata['mesh']=self.mesh
        metadata['pointwise_refinement_steps']=(base[-1].get('pointwise_refinement_steps',0)
                                                if gap>0 else 0)
        return value, tangent, metadata
