"""FDTD physical problems, numerical settings, and immutable prepared execution."""
from .problem import Problem, Scene
from .geometry import Domain, Shape, Slab, Region, Plane, Sphere, Box, Cylinder, Prism, Structure
from .boundaries import Open, Periodic, Bloch, PEC, PMC, Symmetry, Axis1D, Boundaries
from .sources import GaussianPulse, PlaneWave, Dipole, TFSF, InitialField
from .monitors import Record, Field, Flux, Diffraction, FarField, Outputs
from .numerics import UniformGrid, Duration, PML, Absorber, Decay, Numerics
from .media import SwitchedMaterial
from ._api import Prepared, check, prepare, run
from .result import Result, SpectralFlux, FarFieldSpectrum
from . import native, periodic
__all__ = ['Problem','Scene','Domain','Shape','Slab','Region','Plane','Sphere','Box','Cylinder','Prism',
           'Structure','Open','Periodic','Bloch','PEC','PMC','Symmetry','Axis1D','Boundaries',
           'GaussianPulse','PlaneWave','Dipole','TFSF','InitialField','Record','Field','Flux',
           'Diffraction','FarField','Outputs','UniformGrid','Duration','PML','Absorber','Decay','Numerics',
           'SwitchedMaterial','Prepared','check','prepare','run','Result','SpectralFlux','FarFieldSpectrum','native','periodic']
