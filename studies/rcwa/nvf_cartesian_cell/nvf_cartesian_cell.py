"""Compatibility imports for the experimental physical Cartesian P1 cell.

The implementation lives in the installed package. Independent weak-system
and moving-triangle controls remain in tests/rcwa/test_nvf_cartesian_cell.py.
"""
from wavebit.rcwa._engine.conforming.cartesian_rectangle import CartesianRectangleCell, CartesianRectanglePlan

__all__ = ['CartesianRectangleCell', 'CartesianRectanglePlan']
