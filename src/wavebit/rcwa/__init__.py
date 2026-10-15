"""Rigorous coupled-wave analysis: immutable problems and explicit preparation."""
from .problem import Problem, Stack, Layer, Region, Rectangle, Circle, Ellipse, Polygon, Excitation, Outputs
from .sources import PlaneWave, OrderExcitation, GaussianBeam, ElectricDipole
from .numerics import Numerics
from .results import Result, BoundaryPower
from ._lifecycle import Prepared, check, prepare, run
from ._structure import sample_structure

__all__ = ['Problem', 'Stack', 'Layer', 'Region', 'Rectangle', 'Circle', 'Ellipse', 'Polygon',
           'Excitation', 'Outputs', 'PlaneWave', 'OrderExcitation', 'GaussianBeam', 'ElectricDipole',
           'Numerics', 'Result', 'BoundaryPower', 'Prepared', 'check', 'prepare', 'run',
           'sample_structure']
