"""Experimental RCWA formulations with narrower, explicit physical admission.

These research builders preserve their existing numerical implementation and
validation contracts. They construct convolution operators; they do not replace
``rcwa.Problem`` or silently change its factorization. Use their published
cell-centred operators with ``rcwa.operators`` when the retained order window
matches. Their API may evolve independently of the primary lifecycle.
"""
from .._engine.conforming import ConformingNVF
from .._engine.conforming.affine_profile import AffineProfileNVF
from .._engine.conforming.cartesian_rectangle import CartesianRectangleNVF
from .._engine.conforming.cartesian_pair import CartesianRectanglePairNVF
from .._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from .._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell

__all__ = ['ConformingNVF', 'AffineProfileNVF', 'CartesianRectangleNVF',
           'CartesianRectanglePairNVF', 'CartesianPairOpeningPlan',
           'CartesianPairTransmissionGalerkinGapCell']
