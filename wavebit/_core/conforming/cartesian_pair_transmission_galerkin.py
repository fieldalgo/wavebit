"""Explicit pair NVF with fixed local transmission trial functions."""
from numbers import Integral
import numpy as np
from .cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
from .pair_transmission_fields import PairTransmissionFields


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
    same symmetric polynomial mesh of degree 3 through 6. Degrees 5 and 6
    retain complete compensated common and free Fourier contact equations,
    and a full geometric coordinate change at positive gaps. Independent
    physical Fourier integrals check positive-gap retained constraints.
    All original energy, weak, retained and flux tolerances apply. Contact and positive gap use the same field space.
    Choosing branches, mesh or quadrature does not establish optical accuracy.
    The original fixed-power family and ordinary NVF routing are unchanged.
    """

    _polynomial_degrees = (3, 4, 5, 6)

    def __init__(self, plan, order=1, *, reference, branches=2, degree=3,
                 quadrature=80, angular_quadrature=None, coercivity_angle=0.):
        coefficients = np.asarray(reference, complex)
        if (coefficients.shape != (3,) or not np.all(np.isfinite(coefficients))
                or np.any(coefficients == 0)):
            raise ValueError('transmission reference requires three finite nonzero coefficients')
        if (isinstance(branches, (bool, np.bool_)) or not isinstance(branches, Integral)
                or not 1 <= branches <= 4):
            raise ValueError('transmission branches must be an integer from 1 through 4')
        super().__init__(plan, order, degree=degree, quadrature=quadrature,
                         exponents=(), junctions=True, mesh='symmetric',
                         angular_quadrature=angular_quadrature,
                         coercivity_angle=coercivity_angle)
        if self.quadrature < 48:
            raise ValueError('transmission modes require quadrature at least 48')
        self.reference = tuple(coefficients)
        self.branches = int(branches)
        self.root_order = 20

    def _closure_at_offset(self, settings):
        # Geometry is fixed in a JVP. Construct locally, so separate closure
        # calls cannot overwrite each other's material-independent field bank.
        fields = PairTransmissionFields(self._base, settings, self.reference,
                                        self.branches, self.angular_quadrature)
        value, tangent, metadata = super()._closure_at_offset(settings, prepared_fields=fields)
        metadata.update(closure='fixed_transmission_pair_galerkin',
                        transmission_reference=self.reference,
                        transmission_branches=self.branches,
                        transmission_directions=fields.count)
        return value, tangent, metadata
