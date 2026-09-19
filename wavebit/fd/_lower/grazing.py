"""The grazing-order rule: which orders a PML face may be asked to absorb, measured first.

**What was measured** (``studies/fdtd_grazing_return.py``, 2026-09-19, its log in
``studies/results/fdtd_grazing_return.log``): a ``PlaneWave`` in a Bloch cell leaving toward a
PEC-backed PML at exit ``cos theta`` 0.15 .. 0.7, doubling method, over the layer's cells, its
``reflection=`` target and the run length. The PEC backing does **not** remove the grazing
return: it makes the return exactly the round trip the grading is designed for. In the continuum
a PEC-backed graded CFS-PML reflects

    R_c = reflection ** (g cos theta),   g = (m + 1) int_0^1 f^m / (1 + (a (1 - f))^2) df

(``m = ORDER = 4``, ``a = ALPHA_RELATIVE = 1.5`` of ``wavebit/fd/_lower/cpml.py``: ``g =
0.919`` at the face's design wavelength), independent of the thickness. The discrete return
converges to ``R_c`` as the grid refines at a fixed physical thickness, roughly first order but **upward and from below** -- a finer grid is
worse on a grazing order, never better (at 1.2 wavelengths, 1e-8, ``cos 0.26``: 6.9e-3 / 9.7e-3 /
1.1e-2 at 20 / 40 / 80 cells per wavelength against ``R_c`` 1.2e-2). The measured return is the
sum of that round trip, which only the target moves, and a discretisation reflection, which
only cells move (12 cells at 1e-16: 2.6e-4 at ``cos 0.7``, against 2.2e-5 for the default
layer) -- which is why "neither knob alone" was the legacy engine's experience.

**``g`` depends on the wavelength.** The CFS shift is fixed at the face's design wavelength
(``alpha_max = ALPHA_RELATIVE 2 pi / wavelength`` in ``cpml.py``), so at another wavelength the
round trip is ``exp(-2 cos theta int sigma / (1 + (alpha / omega)^2) dz)`` with ``alpha / omega
= ALPHA_RELATIVE (lambda / lambda_design) (1 - f)``: the same law with ``a`` scaled by ``lambda /
lambda_design``. ``g`` is 0.931 / 0.919 / 0.906 / 0.893 at 0.9 / 1.0 / 1.1 / 1.2 of the design
wavelength, and ``cos_min(1e-8)`` 0.403 / 0.408 / 0.414 / 0.420. Every row is judged with the
``g`` of its own wavelength against its face's design wavelength (``plan.pml_wavelengths``; the
column's padded face is matched to the band centre). The study measured the band centre only,
so the other wavelengths rest on the continuum law, the part of the measurement that holds at
every setting measured (the grid approaches ``R_c`` from below). The figures quoted below
(0.41, 0.27, 0.204) are the design wavelength's.

**The rule**, with the amplitude tolerance ``tau = 1e-3``: a face whose target is ``R`` reaches
down to ``cos_min(R) = ln(tau) / (g ln R)`` -- 0.41 for the default ``PML(cells=12,
reflection=1e-8)``. So, for every propagating order that leaves the cell through an absorbing
face along the propagation axis, at every wavelength a ``Diffraction`` monitor asks for, and for
the ``PlaneWave``'s own incidence (its column returns and re-injects at the same size, on the
column's two faces):

* ``cos theta >= cos_min(1e-8) = 0.41`` and ``>= cos_min(R)`` of the face: allowed.
* below 0.41: allowed only if that face declares ``PML(cells >= 24, reflection=R)`` with ``cos
  theta >= cos_min(R)`` **and** the run is ``Run(periods >= 160)`` (of the PlaneWave's band
  centre). 1e-12 reaches 0.27 (measured at ``cos 0.3``: 1.5e-4 / 2.9e-4 / 3.8e-4 at 20 / 40 / 80
  cpw against ``R_c`` 4.9e-4); 1e-16 reaches 0.204 (at ``cos 0.26``: 2.1e-5 / 6.4e-5 / 1.0e-4
  against ``R_c`` 1.5e-4). The 24 cells keep the non-grazing orders from getting worse than
  under the default layer (1e-8 or less at ``cos 0.7``); the 160 periods are needed because at
  40 the band's window error swamps every layer (5e-3 to 1.2e-2 at ``cos 0.26``). Cost: 12 more
  cells on that face and 4x the steps of a 40-period run.
* below ``cos_min(1e-16) = 0.204``: refused outright. No measured setting meets ``tau`` there
  (at ``cos 0.15``, 1e-16 returns 5.2e-3 at 80 cpw against ``R_c`` 6.2e-3); targets past 1e-16
  were not measured.

What is left when an order is allowed: its return up to ``R_c <= 1e-3`` (the grid gives less,
approaching from below), 1e-8 or less of discretisation on the other orders, anywhere from 0
to 2x the return at a monitor between a PlaneWave sheet and the layer (the echo), and a window
term falling as ``T^-1.5``, at or below 3e-4 at ``cos >= 0.26`` after 160 periods for this band.

Only ``PML`` faces are judged: an ``Absorber`` is matched at normal incidence only and absorbs
hopelessly at grazing whatever it is told (``boundaries.Absorber``), and a PEC, PMC or Symmetry
face along the axis is a mirror the scene asked for. An order is judged in the medium of the
absorbing layer it enters (each material filling that layer; the lowest index where the medium
is anisotropic), with ``k_par`` from the cell's faces (a Bloch ``k``, ``0`` on a Periodic axis,
an ``Axis1D`` ``k`` with the single order 0 along it). A medium or a Bloch phase that is traced
cannot be counted; that face is skipped and ``survey`` says so.
"""
import functools
import math

import numpy as np

from ..boundaries import AXES, Axis1D, Bloch, PML, Periodic
from ..geometry import cell_box
from ..materials import as_material, concrete
from .cpml import ALPHA_RELATIVE, ORDER

__all__ = ['GRAZING_TOLERANCE', 'GRAZING_EXPONENT', 'exponent', 'GRAZING_CELLS', 'GRAZING_PERIODS',
           'GRAZING_TABLE', 'cos_min', 'continuum_return', 'survey', 'enforce']

#: The amplitude a PML return may reach on an order that is reported.
GRAZING_TOLERANCE = 1e-3


def _exponent(m=ORDER, a=ALPHA_RELATIVE, samples=200001):
    """``g = (m + 1) int_0^1 f^m / (1 + (a (1 - f))^2) df`` (the module docstring), by Simpson."""
    f = np.linspace(0.0, 1.0, samples)
    y = f ** m / (1.0 + (a * (1.0 - f)) ** 2)
    h = 1.0 / (samples - 1)
    return float((m + 1) * h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


#: ``g`` of ``R_c = reflection ** (g cos theta)``, from the grading's own constants: 0.919.
GRAZING_EXPONENT = _exponent()


@functools.lru_cache(maxsize=512)
def exponent(ratio=1.0):
    """``g`` at ``ratio = wavelength / design wavelength``: the grading's ``a`` scaled by it."""
    if ratio == 1.0:
        return GRAZING_EXPONENT
    return _exponent(a=ALPHA_RELATIVE * ratio)

#: The default target's reach, and the measured floor (the lowest target measured, 1e-16).
DEFAULT_REFLECTION = 1e-8
FLOOR_REFLECTION = 1e-16

#: What a face must declare, and the run must last, below the default face's reach.
GRAZING_CELLS = 24
GRAZING_PERIODS = 160

#: The measurement the refusal quotes: ``R_taper`` at the band centre, 640 periods, s, the
#: layer's own return (``studies/results/fdtd_grazing_return.log``), and the continuum law.
GRAZING_TABLE = """\
  measured PML return at f0 (640 periods, s), columns = exit cos theta:
                        0.15     0.2      0.26     0.3      0.4      0.5      0.7
  20 cpw 12 cells 1e-8  3.3e-02  1.4e-02  4.8e-03  2.3e-03  3.9e-04  7.6e-05  2.2e-05
  20 cpw 24 cells 1e-12 7.3e-03  2.0e-03  4.3e-04  1.5e-04  1.0e-05  7.2e-07  1.0e-08
  20 cpw 24 cells 1e-16 1.1e-03  1.8e-04  2.1e-05  5.3e-06  1.6e-07  5.8e-09  9.3e-09
  80 cpw 1.2 wl   1e-8  7.6e-02  3.2e-02  1.1e-02  5.6e-03  1.0e-03  1.8e-04  5.7e-06
  continuum R_c   1e-8  7.9e-02  3.4e-02  1.2e-02  6.2e-03  1.1e-03  2.1e-04  7.2e-06
  window error alone, 40 / 160 periods (20 cpw):
                        3.5e-02  2.3e-02  1.5e-02  1.1e-02  5.4e-03  2.8e-03  3.0e-04
                        4.3e-03  2.6e-03  1.7e-03  1.2e-03  6.4e-04  2.9e-04  3.1e-05
  R_c = reflection**(0.919 cos theta); cos_min(R) = ln(1e-3) / (0.919 ln R):
  0.41 at 1e-8 (the default), 0.27 at 1e-12, 0.204 at 1e-16 (the floor)
  (g = 0.919 at the face's design wavelength; 0.931 / 0.906 at 0.9x / 1.1x of it)"""


def cos_min(reflection, tolerance=GRAZING_TOLERANCE, g=GRAZING_EXPONENT):
    """The lowest exit ``cos theta`` a face with target ``reflection`` returns ``<= tolerance`` at."""
    return math.log(tolerance) / (g * math.log(reflection))


def continuum_return(reflection, cos, g=GRAZING_EXPONENT):
    """``R_c = reflection ** (g cos theta)``: a PEC-backed graded CFS-PML's round trip."""
    return reflection ** (g * cos)


def reflection_for(cos, tolerance=GRAZING_TOLERANCE, g=GRAZING_EXPONENT):
    """The largest target whose ``cos_min`` is ``cos``: ``tolerance ** (1 / (g cos))``."""
    return tolerance ** (1.0 / (g * cos))


# ---------------------------------------------------------------------------------------------
# What the scene asks the faces to absorb
# ---------------------------------------------------------------------------------------------

def _transverse_k(scene, grid, a):
    """``(k, period, single)`` along transverse axis ``a``; ``None`` if not a periodic axis."""
    face = scene.boundaries.face(AXES[a] + '-')
    if isinstance(face, Periodic):
        return 0.0, grid.size[a], False
    if isinstance(face, Axis1D):
        k = concrete(face.k)
        return (None if k is None else k), None, True
    if isinstance(face, Bloch):
        if face.k is not None:
            k = concrete(face.k)
        else:
            phase = concrete(face.phase)
            k = None if phase is None else phase / grid.size[a]
        return k, grid.size[a], False
    return None


def _layer_media(scene, grid, P, side, cells):
    """``[(label, index or None), ...]``: the materials filling the absorbing layer's cells."""
    k = cells - 1 if side == '-' else grid.n[P] - cells
    background = as_material(scene.background, 'Scene(background=)')
    owners = {}
    for index, s in enumerate(scene.structures):
        box = cell_box(s.shape, grid)
        if box[P][0] <= k < box[P][1]:
            owners[index] = s.material
    media = [('the background', background)] if not any(
        all(cell_box(scene.structures[i].shape, grid)[a] == (0, grid.n[a])
            for a in range(3) if a != P) for i in owners) else []
    media += [(f'structures[{i}]', m) for i, m in owners.items()]
    return [(label, m.index_bound()) for label, m in media]


def _periods(scene, plan, pw):
    wavelength = pw.band.concrete_wavelength
    return None if wavelength is None else plan.steps * plan.dt / wavelength


def _orders(kpar, n, wavelength):
    """``[((m_u, m_v), cos), ...]``: every order propagating in index ``n`` at ``wavelength``."""
    k0 = n * 2.0 * math.pi / wavelength
    axes = []
    for k, L, single in kpar:
        if single or L is None:
            axes.append([(0, k)])
            continue
        g = 2.0 * math.pi / L
        m_max = int(math.ceil((k0 + abs(k)) / g)) + 1
        axes.append([(m, k + m * g) for m in range(-m_max, m_max + 1)])
    out = []
    for mu, ku in axes[0]:
        for mv, kv in axes[1]:
            s2 = (ku * ku + kv * kv) / (k0 * k0)
            if s2 < 1.0:
                out.append(((mu, mv), math.sqrt(1.0 - s2)))
    return out


def survey(scene, plan):
    """``[row, ...]``: the lowest exit ``cos theta`` each judged face is asked to absorb.

    One row per (PML face, what leaves through it): ``{'face', 'what', 'medium', 'order',
    'wavelength', 'cos', 'R_c', 'cos_min', 'cells', 'reflection', 'periods', 'verdict'}`` where
    ``what`` is ``'orders'`` (a Diffraction scene's propagating orders in that layer's medium)
    or ``'incidence'`` (the PlaneWave's own angle, on its column's faces), and ``verdict`` is
    ``'ok'``, ``'needs'`` (below the default reach, and the face or the run falls short),
    ``'weak'`` (below the face's own ``cos_min``) or ``'floor'`` (below 0.204). A face that
    cannot be counted (a traced medium, band or Bloch phase) has ``verdict='skipped'``.
    """
    from ..monitors import Diffraction
    from .monitors import plane_waves
    waves = plane_waves(scene)
    if len(waves) != 1:
        return []
    pw = waves[0]
    P = pw.axis
    transverse = [a for a in range(3) if a != P]
    kpar = [_transverse_k(scene, plan.grid, a) for a in transverse]
    if any(k is None for k in kpar):
        return []
    diffraction = [m for m in scene.monitors if isinstance(m, Diffraction)]
    wavelengths = sorted({w for m in diffraction for w in m.wavelengths})
    periods = _periods(scene, plan, pw)
    design = dict(plan.pml_wavelengths)
    rows = []

    def judge(row, face, face_wavelength):
        """``row`` with its verdict, judged with the ``g`` of its own wavelength."""
        c = row['cos']
        R = face.reflection
        g = exponent(row['wavelength'] / face_wavelength)
        low = cos_min(R, g=g)
        row.update(cells=face.cells, reflection=R, periods=periods, g=g,
                   design_wavelength=face_wavelength, R_c=continuum_return(R, c, g=g),
                   cos_min=low)
        if c < cos_min(FLOOR_REFLECTION, g=g):
            row['verdict'] = 'floor'
        elif c < low:
            row['verdict'] = 'weak'
        elif c < cos_min(DEFAULT_REFLECTION, g=g) and (
                face.cells < GRAZING_CELLS or periods is None
                or periods < GRAZING_PERIODS * (1.0 - 1e-9)):
            row['verdict'] = 'needs'
        else:
            row['verdict'] = 'ok'
        return row

    def worst(candidates):
        """The candidate furthest from being accepted: the verdict first, then ``cos/cos_min``."""
        rank = {'ok': 0, 'needs': 1, 'weak': 2, 'floor': 3}
        return max(candidates, key=lambda r: (rank[r['verdict']], -r['cos'] / r['cos_min']))

    # The orders leaving through each PML face along P, in that layer's medium, judged at every
    # wavelength a Diffraction monitor asks for (the lowest exit cos at each), the worst kept.
    if diffraction:
        for side in ('-', '+'):
            name = AXES[P] + side
            face = scene.boundaries.face(name)
            if not isinstance(face, PML):
                continue
            for label, n in _layer_media(scene, plan.grid, P, side, face.cells):
                if n is None or any(k[0] is None for k in kpar):
                    rows.append(dict(face=name, what='orders', medium=label, verdict='skipped',
                                     why='a traced medium or Bloch k cannot be counted'))
                    continue
                candidates = []
                for w in wavelengths:
                    orders = _orders(kpar, n, w)
                    if orders:
                        order, c = min(orders, key=lambda oc: oc[1])
                        candidates.append(judge(dict(face=name, what='orders', medium=label,
                                                     order=order, wavelength=w, cos=c),
                                                face, design[name]))
                if candidates:
                    rows.append(worst(candidates))
    # The PlaneWave's own incidence, on its column's two faces (a padded column face is the
    # default PML the column carries past a non-absorbing scene face, matched to the band
    # centre).
    n_bg = as_material(scene.background, 'Scene(background=)').index_bound()
    centre = pw.band.concrete_wavelength
    if n_bg is not None and centre is not None and not any(k[0] is None for k in kpar):
        checked = sorted(set(wavelengths) | {centre})
        angles = []
        for w in checked:
            k0 = n_bg * 2.0 * math.pi / w
            s2 = sum(k[0] ** 2 for k in kpar) / (k0 * k0)
            if s2 < 1.0:
                angles.append((w, math.sqrt(1.0 - s2)))
        if angles:
            from ..boundaries import Absorbing
            for side in ('-', '+'):
                name = AXES[P] + side
                face = scene.boundaries.face(name)
                if not isinstance(face, PML):
                    if isinstance(face, Absorbing):
                        continue
                    face = PML()                     # the column's own padded layer
                    label = 'the background (the column\'s padded PML)'
                    face_wavelength = centre
                else:
                    label = 'the background'
                    face_wavelength = design[name]
                rows.append(worst([judge(dict(face=name, what='incidence', medium=label,
                                              order=(0, 0), wavelength=w, cos=c),
                                         face, face_wavelength) for w, c in angles]))
    return rows


def _message(row, pw):
    c = row['cos']
    g = row['g']
    need = reflection_for(c, g=g)
    what = (f"order {row['order']} leaving through {row['face']!r} in {row['medium']}"
            if row['what'] == 'orders' else
            f"the incidence of PlaneWave({pw.name!r}) itself, whose column returns and "
            f"re-injects through {row['face']!r}")
    head = (f"Grazing order refused: {what} at wavelength {row['wavelength']:g} has exit cos "
            f"theta = {c:.3f}. That face is PML(cells={row['cells']}, "
            f"reflection={row['reflection']:g}): its continuum round trip on this order is R_c = "
            f"{row['reflection']:g}**({g:.3f}*{c:.3f}) = {row['R_c']:.2e} (g = {g:.3f} at "
            f"{row['wavelength'] / row['design_wavelength']:.3g} of the face's design wavelength "
            f"{row['design_wavelength']:g}), and it reaches down to "
            f"cos_min = {row['cos_min']:.3f} for the amplitude tolerance "
            f"{GRAZING_TOLERANCE:g}. ")
    if row['verdict'] == 'floor':
        fix = (f'Below cos theta = {cos_min(FLOOR_REFLECTION, g=g):.3f} no measured setting meets '
               f'the tolerance (at cos 0.15, reflection=1e-16 returns 5.2e-3 at 80 cells per '
               f'wavelength against R_c 6.2e-3) and targets past 1e-16 were not measured, so '
               f'this is refused outright: change the period, the angle or the wavelengths so '
               f'the order is less grazing, or leaves the window as evanescent.')
    else:
        fix = (f'Below cos theta = {cos_min(DEFAULT_REFLECTION, g=g):.3f} (the default face\'s reach) '
               f'an order is allowed only when that face declares PML(cells>={GRAZING_CELLS}, '
               f'reflection<={need:.1e}) -- the target whose cos_min is this order\'s cos '
               f'theta -- and the run is Run(periods>={GRAZING_PERIODS}) (this one: '
               + ('unknown' if row['periods'] is None else f"{row['periods']:.1f}")
               + f'). Cost: {GRAZING_CELLS - 12} more cells on that face and 4x the steps of a '
               f'40-period run. What is left then: the return up to R_c <= 1e-3, 1e-8 or less '
               f'on the other orders, up to 2x the return between a sheet and the layer (the '
               f'echo), and a window term at or below 3e-4 after 160 periods.')
    return (head + fix + '\n' + GRAZING_TABLE
            + '\n(studies/fdtd_grazing_return.py; wavebit/fd/_lower/grazing.py derives the law.)')


def enforce(scene, plan):
    """Refuse the first row of :func:`survey` that is not ``'ok'`` or ``'skipped'``."""
    from .monitors import plane_waves
    for row in survey(scene, plan):
        if row['verdict'] in ('ok', 'skipped'):
            continue
        raise ValueError(_message(row, plane_waves(scene)[0]))
