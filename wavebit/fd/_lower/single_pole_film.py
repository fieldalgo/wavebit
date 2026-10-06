"""One flat film with one passive Drude or Lorentz pole in a plain dielectric.

For film fraction f, g=1-f, high-frequency film permittivity a and host b,
write eps_d=a+S/L with L=s*s+gamma*s+omega0_squared and s=-i*omega.
The tangential arithmetic mixture has eps_inf=f*a+g*b and strength=f*S.
The normal harmonic mixture is
    b*(a*L+S)/(A*L+g*S), A=f*b+g*a,
or eps_inf=a*b/A plus one oscillator with strength=f*b*b*S/(A*A)
and resonance_squared=omega0_squared+g*S/A. Damping is unchanged.
These are direct rational reductions of the diagonal interface response in
Deinega and Valuev, Optics Letters 32, 3429 (2007), Eqs. 1-2:
https://doi.org/10.1364/OL.32.003429 . No pole fitting or tensor coupling is used.

The existing centered second-order ADE advances this spatial oscillator without
an instantaneous permittivity shift. Its local energy weights must be supplied
directly: reconstructing them from the recurrence loses precision and confuses
an oscillator at gamma*dt=2 (hold=0) with a Debye pole.

No extra interface CFL rate is needed. With q=sqrt(S_eff/eps_inf), the normalized
Maxwell/oscillator coupling is B=[[C_eps,-q],[0,Omega]]. Its squared norm is at
most ||C_eps||**2+max(Omega**2+q**2), by splitting B B* into the curl block and
cell-local outer products (-q,Omega)(-q,Omega)*. The normal local sum is exactly
omega0_squared+S/a; the tangential sum is no larger. This is the existing material
rate. Centered nonnegative damping dissipates the leapfrog energy; positivity
requires dt*||B||<2. This statement does not extend the existing PML contract.

Only the concrete singleton, unturned, unranged, nonmagnetic film classification
below is supported. In particular, this is not arbitrary dispersive smoothing.
"""
import dataclasses
import math

import numpy as np

from ..boundaries import Absorbing, FACES
from ..geometry import Box
from ..materials import Dispersive, Drude, Lorentz, Material, as_material, concrete
from .._kernel.masks import Region, Slab
from .._kernel.poles import PoleTerm
from .conductive_film import _compact, _fill, _isotropic

__all__ = ['Film', 'DispersiveFilmTerm', 'descriptor', 'terms', 'material_profiles',
           'pole_params', 'refuse_mismatch']


@dataclasses.dataclass(frozen=True)
class Film:
    """Concrete material and geometry identity of the supported one-pole film."""

    axis: int
    low: float
    high: float
    eps: float
    host_eps: float
    omega0_squared: float
    gamma: float
    strength: float
    box: tuple
    signature: tuple


@dataclasses.dataclass(frozen=True)
class DispersiveFilmTerm(PoleTerm):
    """One spatial oscillator, reserved even for a zero-strength Lorentz film."""

    signature: tuple


def descriptor(scene, grid):
    """Return the bounded supported scene, or None for the existing refusal paths."""
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) != 1 or getattr(scene, 'initial', None) is not None:
        return None
    structure = structures[0]
    shape, material = structure.shape, structure.material
    if (type(shape) is not Box or shape.angle != 0 or shape.range is not None
            or structure.discretisation != 'voxel' or type(material) is not Dispersive
            or len(material.poles) != 1):
        return None
    pole = material.poles[0]
    if type(pole) not in (Drude, Lorentz):
        return None
    host = as_material(getattr(scene, 'background', None))
    if type(host) is not Material:
        return None
    # concrete() alone can detach Torch forward-mode material values.
    from .shape_ranges import materials_traced, traced
    if materials_traced(scene) or traced(scene):
        return None
    a = _isotropic(material.eps, positive=True)
    b = _isotropic(host.eps, positive=True)
    mu_a = _isotropic(material.mu_value, positive=True)
    mu_b = _isotropic(host.mu_value, positive=True)
    values = tuple(concrete(getattr(pole, field.name))
                   for field in dataclasses.fields(pole))
    if (None in (a, b) or mu_a != 1.0 or mu_b != 1.0
            or any(v is None or not math.isfinite(v) or v < 0 for v in values)):
        return None
    gamma = concrete(pole.gamma)
    if type(pole) is Drude:
        frequency = concrete(pole.omega_p)
        omega0_squared, strength = 0.0, frequency*frequency
    else:
        frequency = concrete(pole.omega_0)
        omega0_squared, strength = frequency*frequency, concrete(pole.strength)
    if (not math.isfinite(omega0_squared) or not math.isfinite(strength)
            or frequency*frequency == 0):
        raise ValueError('single-pole-film derived oscillator exceeds finite float64 range; '
                         'rescale the physical units or use less extreme material values.')
    uniform = tuple(shape.uniform_along(k, (0, n), grid)
                    for k, n in enumerate(grid.n))
    if sum(uniform) != 2:
        return None
    axis = uniform.index(False)
    if shape.size[axis] < grid.dx:
        return None                           # no two surfaces inside one E dual cell
    box = tuple(tuple(int(v) for v in pair) for pair in shape.support(grid))
    lo, hi = box[axis]
    if lo <= 0 or hi >= grid.n[axis] or lo >= hi:
        return None                           # also excludes normal wall half-voxels
    for name in FACES:
        face = scene.boundaries.face(name)
        if not isinstance(face, Absorbing):
            continue
        k = 'xyz'.index(name[0])
        start, end = box[k]
        if (start < face.cells if name[1] == '-' else end > grid.n[k]-face.cells):
            return None
    low = float(shape.center[axis] - shape.size[axis]/2)
    high = float(shape.center[axis] + shape.size[axis]/2)
    signature = ('single-pole-film-v1', tuple(float(v) for v in shape.size),
                 tuple(float(v) for v in shape.center), float(shape.angle), shape.axis,
                 a, b, type(pole).__name__, values, material.mu is not None,
                 host.mu is not None, structure.discretisation, structure.junction,
                 tuple(grid.n), float(grid.dx), tuple(float(v) for v in grid.low))
    return Film(axis, low, high, a, b, omega0_squared, gamma, strength, box, signature)


def terms(scene, grid):
    """One interface term replacing the film's ordinary, spatially uniform PoleTerm."""
    film = descriptor(scene, grid)
    if film is None:
        return ()
    region = Region(z=film.box[2], y=film.box[1], x=film.box[0])
    slabs = tuple(Slab(c=c, z=region.z, y=region.y, x=region.x) for c in range(3))
    return (DispersiveFilmTerm(name='dispersive_film[0]', region=region, slabs=slabs,
                               n_poles=1, signature=film.signature),)


def refuse_mismatch(scene, plan):
    """A prepared plan cannot change the film or silently omit its spatial memory."""
    held = tuple(t for t in plan.terms
                 if isinstance(t, DispersiveFilmTerm)
                 or getattr(t, 'name', '').startswith('dispersive_film['))
    expected = terms(scene, plan.grid)
    if not expected and any(s.discretisation == 'voxel' and isinstance(s.material, Dispersive)
                            and s.shape.body() is not None
                            for s in (getattr(scene, 'structures', ()) or ())):
        raise ValueError('dispersive smoothing is limited to one concrete axis-aligned film '
                         'with one passive Drude or Lorentz pole in a plain lossless host, '
                         'separated interfaces outside absorbing layers, and no initial '
                         'fields or differentiation. A prepared non-film plan cannot '
                         'bypass this bounded C8 scope.')
    all_poles = tuple(t for t in plan.terms if hasattr(t, 'n_poles'))
    if held != expected or (expected and all_poles != expected):
        raise ValueError('the prepared single-pole-film plan does not match this scene. '
                         'Changed or traced geometry/material values and missing/extra '
                         'interface memory need a new supported fd.plan(scene).')


def _profiles(film, grid, dt):
    """Finite compact coefficient/energy profiles, retaining every positive fill."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('single-pole-film coefficients need a finite positive time step.')
    n = grid.n[film.axis]
    eps = np.full((3, n), film.host_eps)
    sigma, gain, hold, drive, inverse, potential, active = (np.zeros((3, n))
                                                         for _ in range(7))
    a, b, w0, gamma, strength = (film.eps, film.host_eps, film.omega0_squared,
                                film.gamma, film.strength)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            dt_squared = dt*dt
            half = gamma*dt/2
            denominator = 1+half
            for component in range(3):
                f = _fill(film, grid, component)
                g = 1-f
                if component != film.axis:
                    eps[component] = f*a+g*b
                    effective_strength = f*strength
                    resonance = np.full(n, w0)
                else:
                    # Evaluate the rational formula only on mixed cells: at a pure
                    # endpoint an otherwise unused ratio can overflow for extreme eps.
                    mixed = (f > 0) & (f < 1)
                    fm, gm = f[mixed], g[mixed]
                    A = fm*b+gm*a
                    ratio = b/A
                    eps[component, mixed] = a*ratio
                    effective_strength = np.zeros(n)
                    effective_strength[mixed] = (fm*b/A)*ratio*strength
                    resonance = np.full(n, w0)
                    resonance[mixed] = w0+gm*strength/A
                # Exact geometric endpoints; no epsilon-wide snapping of mixed cells.
                eps[component, f == 0] = b
                eps[component, f == 1] = a
                effective_strength[f == 0] = 0
                effective_strength[f == 1] = strength
                resonance[f == 1] = w0
                live = (f > 0) & (strength > 0)
                s, omega_squared = effective_strength[live], resonance[live]
                gain[component, live] = (2-omega_squared*dt_squared)/denominator
                hold[component, live] = (1-half)/denominator
                drive[component, live] = s*dt_squared/denominator
                inverse[component, live] = 1/s
                potential[component, live] = omega_squared/s
                active[component, live] = 1
    except (FloatingPointError, OverflowError) as error:
        raise ValueError('single-pole-film derived coefficients exceed finite float64 range; '
                         'rescale the physical units or use less extreme material values.') from error
    arrays = (eps, sigma, gain, hold, drive, inverse, potential, active)
    live = active != 0
    if (not all(np.isfinite(v).all() for v in arrays) or not np.all(eps > 0)
            or np.any(drive[live] <= 0) or np.any(inverse[live] <= 0)
            or np.any(potential < 0) or np.any(np.abs(hold) > 1)):
        raise ValueError('single-pole-film derived coefficients are not finite and passive; '
                         'rescale the physical units or use less extreme material values.')
    return tuple(_compact(v, film.axis) for v in arrays)


def material_profiles(scene, plan, rt):
    """None, or compact high-frequency permittivity and zero conductivity."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return None
    eps, sigma, *_ = _profiles(film, plan.grid, plan.dt)
    return rt.array(eps, 'float64'), rt.array(sigma, 'float64')


def pole_params(scene, plan, rt):
    """One spatial oscillator and its local energy weights, cropped to term support."""
    film = descriptor(scene, plan.grid)
    if film is None:
        return {}
    _, _, gain, hold, drive, inverse, potential, active = _profiles(film, plan.grid, plan.dt)
    cut = [slice(None)]*4
    cut[3-film.axis] = slice(*film.box[film.axis])
    cut = tuple(cut)
    array = lambda value: rt.array(value[cut], 'float64')
    return {'dispersive_film[0]': {
        'step_gain': (array(gain),), 'hold': (array(hold),), 'drive': (array(drive),),
        'dx_over_dt': rt.array(plan.dx/plan.dt, 'float64'),
        'energy_inverse_strength': (array(inverse),),
        'energy_resonance_over_strength': (array(potential),),
        'energy_active': (array(active),),
    }}
