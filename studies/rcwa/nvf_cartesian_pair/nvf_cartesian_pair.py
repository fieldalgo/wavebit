"""Compatibility access to the qualified physical Cartesian pair finite cell.

The implementation is packaged with Wavebit. This import preserves the study
and independent-oracle interface without a runtime dependency on studies.
"""
from wavebit.rcwa._engine.conforming.cartesian_pair import CartesianPairCell, CartesianPairPlan

__all__ = ['CartesianPairPlan', 'CartesianPairCell']
