"""Explicit pair NVF with fixed local transmission trial functions."""
from numbers import Integral
from wavebit._common.conventions import swap_fourier_origin
import numpy as np
from wavebit.rcwa._engine.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
from wavebit.rcwa._engine.conforming.pair_transmission_fields import PairTransmissionFields
from wavebit._common.backend import eager_under_torch_compile


@eager_under_torch_compile
class CartesianPairTransmissionGalerkinGapCell(CartesianPairCornerGalerkinGapCell):
    """Experimental complete tensor/JVP with a fixed reference corner space.

    reference=(left, right, background) specifies auxiliary transmission
    coefficients, independently of the physical materials passed to closure.
    They never change with a material JVP. branches=1 through 4 adds successive
    noninteger growing roots lambda, 2-lambda, 2+lambda, 4-lambda at all six
    corners. Each complex mode contributes both real directions. A real mode
    with real relative coefficients contributes its single real direction. Straight
    reference interfaces add no noninteger direction. Junction modes require
    Re(lambda) > 1/2 for finite opening energy; outer modes require Re(lambda) > 0.
    An inadmissible reference bank is rejected as a whole.

    Every mode is multiplied by its fixed corner hat and transported with the
    same symmetric polynomial mesh of degree 3 through 8. Degrees 5 through 8
    retain complete compensated common and free Fourier contact equations,
    and a full geometric coordinate change at positive gaps. Independent
    physical Fourier integrals check positive-gap retained constraints.
    All original energy, weak, retained and flux tolerances apply. Contact and positive gap use the same field space.
    Choosing branches, mesh or quadrature does not establish optical accuracy.
    At degrees 5 through 8, cutoff_order=1 or 2 retains h*(1-h)**m times every
    selected mode for all m from zero through cutoff_order. These nested
    spaces use three-word common-field arithmetic before final two-word
    assembly. Degrees 7 and 8 also use this arithmetic for cutoff zero,
    including flat reference materials with no added transmission direction.
    The default zero keeps the original h-weighted field space.
    The original fixed-power family and ordinary NVF routing are unchanged.
    """

    _polynomial_degrees = (3, 4, 5, 6, 7, 8)

    def __init__(self, plan, order=1, *, reference, branches=2, degree=3,
                 quadrature=80, angular_quadrature=None, coercivity_angle=0.,
                 cutoff_order=0):
        coefficients = np.asarray(reference, complex)
        if (coefficients.shape != (3,) or not np.all(np.isfinite(coefficients))
                or np.any(coefficients == 0)):
            raise ValueError('transmission reference requires three finite nonzero coefficients')
        if (isinstance(branches, (bool, np.bool_)) or not isinstance(branches, Integral)
                or not 1 <= branches <= 4):
            raise ValueError('transmission branches must be an integer from 1 through 4')
        if (isinstance(cutoff_order, (bool, np.bool_)) or not isinstance(cutoff_order, Integral)
                or not 0 <= cutoff_order <= 2):
            raise ValueError('transmission cutoff_order must be an integer from 0 through 2')
        if cutoff_order and degree not in (5, 6, 7, 8):
            raise ValueError('nested transmission cutoffs require degree 5 through 8')
        super().__init__(plan, order, degree=degree, quadrature=quadrature,
                         exponents=(), junctions=True, mesh='symmetric',
                         angular_quadrature=angular_quadrature,
                         coercivity_angle=coercivity_angle)
        if self.quadrature < 48:
            raise ValueError('transmission modes require quadrature at least 48')
        self.reference = tuple(coefficients)
        self.branches = int(branches)
        self.cutoff_order = int(cutoff_order)
        self.root_order = 20

    def closure(self, *, gap, basis, widths, height, epsilon, background=1.,
                offset=(0., 0.), gap_tangent=1., epsilon_tangent=(0., 0.),
                background_tangent=0.):
        """Return centre-frame value and tangent slots for ``solve_conv``.

        ``offset`` places the pair relative to the public cell centre. The
        private parent computes in its existing chart; both exported operators
        are restated at this door. ``coordinate_origin`` also names their
        Fourier origin here, as well as the chart used in the construction.
        """
        value, tangent, metadata = super().closure(
            gap=gap, basis=basis, widths=widths, height=height, epsilon=epsilon,
            background=background, offset=offset, gap_tangent=gap_tangent,
            epsilon_tangent=epsilon_tangent, background_tangent=background_tangent)
        window = tuple((0, n) for n in self.order)
        return (swap_fourier_origin(value, *window),
                swap_fourier_origin(tangent, *window), metadata)

    def _closure_at_offset(self, settings):
        # Geometry is fixed in a JVP. Construct locally, so separate closure
        # calls cannot overwrite each other's material-independent field bank.
        fields = PairTransmissionFields(self._base, settings, self.reference,
                                        self.branches, self.angular_quadrature,
                                        self.cutoff_order)
        value, tangent, metadata = super()._closure_at_offset(settings, prepared_fields=fields)
        metadata.update(closure='fixed_transmission_pair_galerkin',
                        transmission_reference=self.reference,
                        transmission_branches=self.branches,
                        transmission_cutoff_order=self.cutoff_order,
                        transmission_directions=fields.count)
        return value, tangent, metadata
