"""The stage-1 scenes, samples and lowered arrays that ``fd_golden_stage1.npz`` and its test share.

One module, imported by both ``interface/freeze_fd_stage1.py`` (which writes the golden) and
``tests/fdtd/test_fd_stage1_backends.py`` (which reads it), so the two cannot disagree about which
scene a key names. Nothing here solves anything. The stage-0 counterpart
(``_fd_stage0_cases.py``) is imported rather than copied: the sampling rule, the capture, the
content digest and the relative gap are the same definitions, and a second copy of them would
be a second thing to keep in step.

**The six scenes**, one per stage-1 feature, each small enough to run in well under a second
on every backend and long enough that the feature is actually exercised -- the pulse reaches
the layer, the slab, the metal:

``pml_box``
    A 20^3 vacuum box with a ``PML`` on all **six** faces (6 cells, ``allow_thin``), so every
    face's slab geometry, all 24 psi arrays and the corner overlaps are in the answer. Real.
``pml_oblique``
    A 1 x 1 x 120 line carrying a 30-degree travelling packet under a ``Bloch`` phase across
    x, with a 12-cell ``PML`` on z: the same stretch at the other dtype, at an angle.
``slab``
    A lossless dielectric layer (``eps = 2.25``) between two PMLs: the ``be`` array, built
    only along z, and a cell-snapped interface.
``dispersive``
    A Drude + Lorentz half-space running out through an ``Absorber`` face (a PML cannot be
    matched to a dispersive medium, and refuses it by name): the ADE term, two poles, and the
    graded damping in the same scene.
``conductor``
    A ``Conductor`` slab (``sigma = 0.5``) in the path of a 20-degree packet, Bloch across x:
    the ``ae`` and ``be`` loss coefficients, complex.
``absorber``
    Vacuum with an ``Absorber`` on both z faces and no PML at all: the damping profile on its
    own, with no stretch to hide behind.

**Every scene empties itself**, which is what lets the gate say the frozen numbers are an
answer and not only a number: each one is driven by a field with no static (longitudinal)
part, so nothing is left sitting in the box that an absorber could never have removed. That
costs a different excitation per geometry -- the curl of a scalar blob in the cube, a
:class:`~fd_helpers.Packet` of exact travelling eigenmodes in the oblique lines, and a pair of
plain Gaussians in the normal-incidence lines, where the single transverse cell makes a pulse
on Ex or Ey divergence free already.

**What is frozen.** Per scene and backend, the same small set stage 0 freezes -- the probe
series, the final E and H at eight cells per component, and the whole-volume energy. Per
scene and *once*, from the NumPy lowering, the stage-1 numbers that are new: the four update
coefficient arrays, every CPML profile (``b``, ``a``, ``1/kappa``), and every pole's
recurrence coefficients. Those are what a stage-1 defect moves first, they are the same on
every backend by construction, and freezing them separates "the profile changed" from "the
kernel drifted" in the failure message.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import math
import sys

import numpy as np

HERE = repository_path('tests/references')
sys.path.insert(0, str(HERE))                          # tests/references, for the stage-0 cases
sys.path.insert(0, str(HERE.parent))                   # tests/, for fd_helpers

import wavebit.fdtd._legacy as fd                                                       # noqa: E402
from tests.references._fd_stage0_cases import (BACKENDS as BACKENDS, capture as capture,
                                              content_digest as content_digest,
                                              relative_gap as relative_gap)
from tests.fixtures.fdtd.fd_helpers import Packet                                                 # noqa: E402

#: The committed golden.
GOLDEN = HERE / 'fd_golden_stage1.npz'

#: Gate 7's budget, the same 1e-12 of an array's peak gate 5 holds the stage-0 scenes to.
#: Measured 2026-09-18 on these scenes: Torch is bit-identical to NumPy on the four real ones
#: and 4.7e-15 off on the two complex ones; JAX sits 4.2e-15 to 1.2e-13 off both. The large
#: end of that range is on the scenes whose field has decayed by three decades before the last
#: step -- the scale of this measure is the array's *own* peak, so a field that has been
#: absorbed away is compared against a correspondingly smaller number. The budget is 8x above
#: the worst of it and orders below what a moved profile, coefficient or pole does (the two
#: mutants in the gate land at 1e-2 to 3).
TOL = 1e-12

#: The grid every scene below uses, and the wavelength its PMLs are matched at.
DX = 0.05
WAVELENGTH = 1.0
COURANT = 0.5

#: The resolution is stated as cells per wavelength rather than as ``dx`` (the same 0.05) so
#: that the scene carries a wavelength of its own. Without one, the cells-per-wavelength check
#: has nothing to measure a material against and says so -- and it would say it loudly in the
#: PEC control of :func:`sealed`, where taking the absorbing faces away also takes away the
#: only wavelength the scene had.
RESOLUTION = fd.Resolution(cells_per_wavelength=WAVELENGTH / DX, wavelength=WAVELENGTH)


def _pulse(centre, width, carrier=None):
    """A Gaussian in z (times a carrier, which is what makes a complex scene complex).

    An initial E with no H is a standing pulse that splits into two travelling halves, so one
    excitation reaches both faces of the axis. Nothing here is a physics claim: these scenes
    are a drift detector, and what they have to do is drive every stage-1 slot.
    """
    def fn(x, y, z):
        g = np.exp(-((z - centre) / width) ** 2)
        if carrier is not None:
            g = g * np.exp(2j * np.pi * (z - centre) / carrier)
        return g
    return fn


def _line(nz, boundaries, structures=(), steps=300, centre=None, width=0.15, carrier=None,
          probes=()):
    """A 1 x 1 x ``nz`` box whose low z corner is 0, with a pulse on **both** Ex and Ey.

    Both, and not Ey alone, because of what a one-cell transverse box does to the other
    components: ``d/dy`` is identically zero across a single periodic cell, so an Ey pulse
    alone leaves ``Ex`` exactly zero for the whole run -- and with it the ``<face>:E:0`` and
    ``<face>:H:0`` psi of every z face, which is half of the CPML this gate exists to pin.
    (Measured: the profile mutant below moved nothing at all until this was fixed.) The two
    pulses are given different widths and centres so that the scene does not merely run the
    same field twice under another name.
    """
    z0 = nz * DX / 2 if centre is None else centre
    E = (_pulse(z0 + 3 * DX, 1.3 * width, carrier), _pulse(z0, width, carrier), None)
    return fd.Scene(domain=fd.Domain(size=(DX, DX, nz * DX), center=(0.0, 0.0, nz * DX / 2)),
                    boundaries=boundaries, resolution=RESOLUTION,
                    run=fd.Run(steps=steps, courant=COURANT),
                    initial=fd.InitialField(E=E),
                    structures=structures,
                    monitors=tuple(fd.Record(name, at=(0.0, 0.0, z), component=component)
                                   for name, z, component in probes))


def _curl_blob(w):
    """``E = curl(0, 0, exp(-r^2/w^2)) * w^2/2``: a 3-D blob with no static part.

    **A blob on Ey alone would not do**, and the reason is a property of the update and not of
    this file: the leapfrog conserves the discrete divergence of E exactly, so the
    longitudinal part of an initial field is a static charge distribution that never
    propagates and never reaches a layer. Measured on the 20^3 box: an Ey Gaussian leaves 25%
    of its energy sitting in the box after 200 steps, which would make "the PML absorbed it"
    untestable. The curl of a scalar blob is divergence free in the continuum and to
    discretisation error on the grid, and leaves 0.2%.
    """
    def fx(x, y, z):
        return -y * np.exp(-(x ** 2 + y ** 2 + z ** 2) / w ** 2)

    def fy(x, y, z):
        return x * np.exp(-(x ** 2 + y ** 2 + z ** 2) / w ** 2)
    return fx, fy, None


def _packet_line(nz, angle, z_faces, structures=(), steps=480, start=0.28, probes=()):
    """A ``1 x 1 x nz`` box carrying one :class:`~fd_helpers.Packet` at ``angle``, Bloch across x.

    The oblique scenes use the packet rather than a Gaussian on a component for the same
    reason :func:`_curl_blob` exists: under a Bloch phase a pulse on one component has a
    longitudinal part that stays where it is put. The packet is an explicit superposition of
    exact discrete travelling eigenmodes, each transverse to its own wavevector, so the box
    empties -- and it is complex, which is the dtype these two scenes are here for.
    """
    dt = COURANT * DX / math.sqrt(3.0)
    packet = Packet(WAVELENGTH, angle, DX, dt, z0=start * nz * DX, real=False)
    return fd.Scene(
        domain=fd.Domain(size=(DX, DX, nz * DX), center=(0.0, 0.0, nz * DX / 2)),
        boundaries=fd.Boundaries(x=fd.Bloch(phase=packet.kx * DX), y=fd.Periodic(), z=z_faces),
        resolution=RESOLUTION, run=fd.Run(steps=steps, courant=COURANT),
        initial=packet.initial(), structures=structures,
        monitors=tuple(fd.Record(name, at=(0.0, 0.0, z), component=component)
                       for name, z, component in probes))


#: The two-pole metal of the ``dispersive`` scene. Drude + Lorentz, both well inside the CFL
#: bound at this dx (rate = 144 + 76, bound 0.0282 against dt 0.0144), and at wavelength 1.0
#: it is ``eps = 1.177 + 0.097j`` -- 18 cells per wavelength inside it, so no mesh warning.
METAL = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=4.0, gamma=1.0),
                                          fd.Lorentz(omega_0=12.0, gamma=1.0, strength=60.0)))


def scenes():
    """``{name: scene}`` -- the six stage-1 scenes; see the module docstring for what each is for."""
    out = {}
    out['pml_box'] = fd.Scene(
        domain=fd.Domain(size=(20 * DX,) * 3),
        boundaries=fd.Boundaries.open(fd.PML(cells=6, wavelength=WAVELENGTH, allow_thin=True)),
        resolution=RESOLUTION, run=fd.Run(steps=120, courant=COURANT),
        initial=fd.InitialField(E=_curl_blob(4 * DX)),
        monitors=(fd.Record('Ey', at=(0.1, 0.0, -0.05), component='Ey'),
                  fd.Record('Hx', at=(0.1, 0.0, -0.05), component='Hx')))

    out['pml_oblique'] = _packet_line(
        120, 30.0, fd.PML(cells=12, wavelength=WAVELENGTH),
        probes=(('Ey', 4.4, 'Ey'), ('Hx', 2.0, 'Hx')))

    out['slab'] = _line(
        60, fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                          z=fd.PML(cells=12, wavelength=WAVELENGTH)),
        structures=(fd.Structure(fd.Slab(z=(1.7, 2.0)), fd.Material(eps=2.25)),),
        centre=1.2, probes=(('front', 1.4, 'Ey'), ('back', 2.2, 'Ey')))

    out['dispersive'] = _line(
        60, fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                          z=(fd.PML(cells=12, wavelength=WAVELENGTH), fd.Absorber(cells=12))),
        structures=(fd.Structure(fd.Slab(z=(1.8, 3.0)), METAL),),
        centre=1.2, probes=(('front', 1.6, 'Ey'), ('inside', 2.0, 'Ey')))

    out['conductor'] = _packet_line(
        120, 20.0, fd.PML(cells=12, wavelength=WAVELENGTH),
        structures=(fd.Structure(fd.Slab(z=(3.4, 3.9)),
                                 fd.Conductor(eps=2.25, sigma=0.5)),),
        probes=(('front', 3.2, 'Ey'), ('back', 4.1, 'Ey')))

    out['absorber'] = _line(
        48, fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Absorber(cells=12)),
        probes=(('Ey', 1.0, 'Ey'), ('Hx', 1.4, 'Hx')))
    return out


#: The scenes whose only loss is the absorbing face: sealing them must keep the energy.
BOUNDARY_LOSS_ONLY = ('pml_box', 'pml_oblique', 'slab', 'absorber')

#: The scenes with a lossy *material*: sealing them is not a control, and :func:`lossless` is.
MATERIAL_LOSS = ('dispersive', 'conductor')


def sealed(scene):
    """The same scene with every absorbing face replaced by ``PEC``: the energy control.

    A frozen final energy far below the initial one says nothing on its own -- a pulse can
    leave a window, and a number that only ever falls is a number a broken kernel produces
    too. The same box with PEC walls keeps its energy (the leapfrog conserves a nearby
    quadratic form, so it wanders by a few percent and does not decay), so the pair is what
    makes the decay attributable to the absorber.
    """
    faces = {}
    for axis in ('x', 'y', 'z'):
        low, high = getattr(scene.boundaries, axis)
        pair = tuple(fd.PEC() if isinstance(f, fd.Absorbing) else f for f in (low, high))
        faces[axis] = pair[0] if pair[0] is pair[1] else pair
    return scene.replace(boundaries=fd.Boundaries(**faces))


def sample_indices(shape):
    """``(S, 4)`` cell indices ``(c, k, j, i)`` to freeze of a ``(3, nz, ny, nx)`` field.

    Stage 0's picks are written for a cube and index ``(1, 2, 3)``; four of these six scenes
    are ``1 x 1 x N`` lines, where that is out of bounds. So the picks here are stated along
    the long axis and clipped into range on the other two: the two ends, a cell **inside**
    each absorbing layer (3 and 4 from the ends, where a psi write that missed its slab would
    show), a cell just inside each layer's inner face (6 and 14 from the ends, which is where
    a profile graded from the wrong end differs most), the centre, and one cell a third of the
    way in. Eight per component, as stage 0 freezes.
    """
    _, nz, ny, nx = shape

    def take(n, values):
        return [min(max(v if v >= 0 else n + v, 0), n - 1) for v in values]

    ks = take(nz, (0, 3, 6, 13, nz // 2, -14, -4, -1))
    js = take(ny, (0, 1, ny // 2, ny - 2, ny // 2, 1, 0, ny - 1))
    xs = take(nx, (0, nx - 1, nx // 2, 1, nx // 3, nx // 2, nx - 2, nx // 4))
    return np.array([(c, k, j, i) for c in range(3) for k, j, i in zip(ks, js, xs)],
                    dtype=np.int64)


def lossless(scene):
    """The same scene with every structure's material replaced by a lossless one of its ``eps``.

    The control for the two scenes whose loss is in the *medium*: a ``Conductor(eps=2.25,
    sigma=0.5)`` becomes ``Material(eps=2.25)`` and the two-pole metal becomes its own
    ``eps_inf``. Sealed, the control conserves energy and the original does not, which is what
    says the decay came from the conductivity and the poles rather than from the walls.
    """
    structures = tuple(s.replace(material=fd.Material(eps=s.material.eps))
                       for s in scene.structures)
    return scene.replace(structures=structures)


def psi_tag(key):
    """``'z+:E:0'`` as an archive key: no ``:`` or ``+`` in a name that lands in a zip entry."""
    face, field, component = key.split(':')
    return f'{face[0]}{"hi" if face[1] == "+" else "lo"}_{field}{component}'


def lowered(scene, plan, samples):
    """The arrays frozen once per scene from the NumPy lowering: what stage 1 added to params.

    The initial field at the sampled cells (as stage 0 freezes it, so a moved excitation reads
    as a moved scene and not as kernel drift), the four update coefficients in full, every
    CPML profile, and every pole's three recurrence coefficients. All of it is small: the
    largest array here is ``(3, 60, 1, 1)``.
    """
    params = fd.lower(scene, plan, 'numpy')
    idx = tuple(samples.T)
    out = {}
    for field in ('E', 'H'):
        F0 = params[f'{field}0']
        if F0 is not None:
            out[f'initial__{field}'] = np.ascontiguousarray(np.asarray(F0)[idx])
    for name in ('ae', 'be', 'ah', 'bh'):
        out[f'coefficient__{name}'] = _frozen(params[name])
    for key, profile in params['cpml'].items():
        for which in ('b', 'a', 'ik'):
            out[f'cpml__{psi_tag(key)}__{which}'] = np.ascontiguousarray(
                np.asarray(profile[which]).reshape(-1))
    for term, coefficients in params['poles'].items():
        tag = term.replace('[', '_').replace(']', '')
        for which in ('step_gain', 'hold', 'drive'):
            out[f'poles__{tag}__{which}'] = np.array(
                [float(v) for v in coefficients[which]], dtype=np.float64)
        out[f'poles__{tag}__dx_over_dt'] = _frozen(coefficients['dx_over_dt'])
    return out


def _frozen(x):
    """``x`` as a contiguous array, **keeping a scalar 0-dimensional**.

    ``np.ascontiguousarray`` promotes a 0-d array to shape ``(1,)``, which would hide the one
    thing the coefficient rows are here to show: a vacuum scene keeps the stage-0 *scalars*,
    and a scene that quietly started building ``(3, 1, 1, 1)`` arrays for vacuum would freeze
    under the same shape as before.
    """
    a = np.asarray(x)
    return np.ascontiguousarray(a) if a.ndim else a.copy()
