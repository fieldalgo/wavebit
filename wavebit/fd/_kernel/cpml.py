"""The CPML stretch: what a ``PML`` face is inside the curl, and where its memory lives.

A PML face is two things and no more (``boundaries.py``): the face's neighbour rule is the
zero pad of a PEC wall, and a slab of ``cells`` cells inside it stretches the derivative
along that face's axis. Nothing else in the kernel knows a PML exists.

**Which differences are stretched, and why there are exactly four psi per face.** Component
``c`` of a curl is ``D_a F[b] - D_b F[a]`` with ``a = c+1``, ``b = c+2`` (mod 3), so the
difference ``D_a F[a]`` -- along an axis, of the component along that axis -- never appears.
For an absorbing axis ``a`` the stretched differences are therefore ``D_a F[b]`` for the two
``b != a``, in each of the two curls: two components times two curls is **four psi arrays per
face**, each of shape ``(cells, <full transverse>)``.

**Where psi lives: in the slab, on every backend.** Measured
(``studies/fdtd_cpml_slab_cost.py``, 2026-09-18, CPU, jax 0.11.0). It is not a trade. At
128^3 with six faces the slab layout is 36 MB against 192 MB for psi over the full volume
(the fields themselves are 96 MB), *and* it is faster on all three backends: 1.11x against
2.04x (NumPy), 1.13x against 1.90x (Torch), 2.27x against 2.79x (JAX) over the no-absorber
baseline. The one corner it loses is Torch at 64^3 with a single axis absorbing (1.85x
against 1.52x) -- eight small in-place ops losing to four large ones while a 12 MB field sits
in cache -- which closes by 96^3 and inverts at 128^3. Not a reason to carry a second layout,
but it is why a tiny one-axis benchmark shows the CPML "costing more than it should" on
Torch.

**How the recurrence is written** is per backend and is `rt.cpml_stretch` (``_runtime.py``),
because the writing, not the layout, is what the study actually had to settle:

* NumPy fuses the four operations in place on the slab *view* of the difference, in the order
  the formula is written, with one slab-shaped scratch: 1.11x against 1.24x for the
  gather-correct-add-back form at 128^3 / 6 faces.
* Torch does the same four in place and needs no scratch at all.
* JAX writes ``d.at[slices].add(...)`` -- a dynamic-update-slice, **deliberately the opposite
  of the stage-0 ``rt.slab_add`` primitive**, and not a contradiction of it. The stage-0 study
  rejected the slice add for a small cube at an arbitrary origin, which lowers to a
  full-volume select loop LLVM unswitches only on aligned origins; a CPML slab has full
  extent on two axes, is contiguous on the third and is anchored at 0 or ``N - cells``, so
  the loop bounds are exact and there is nothing to unswitch. It is the cheapest writing at
  every N and on every axis, the only one that beats the full layout at every N, the flattest
  across axes, and the only one needing no index arrays at all (a scatter's three int32 per
  cell cost 1.5x the float64 psi they address). ``rt.slab_add`` and the CPML write are two
  different primitives; do not unify them.

Both XLA findings are version-dependent (jax 0.11.0) and must be re-measured when JAX moves,
exactly as the slab-add study's own must. GPU is stage 6 and nothing above transfers to one;
the memory figures are exact and do, and matter more there.

**psi is carried, never recomputed.** It is a recurrence, so it is its own entry of the scan
carry (``state['psi']``), a dict keyed by :func:`psi_key`. The slab geometry is static and
lives on the plan; ``b``, ``a`` and ``1/kappa`` are traced leaves and live in ``Params``, as
1-D profiles along the absorbing axis, reshaped once to broadcast and never materialised at
slab shape.

**Where the profile is sampled.** ``curl E`` is a forward difference, so ``D_a E[b]`` lands
half a cell along ``a`` -- the H points -- and reads the *half* profile. ``curl H`` is a
backward difference, so ``D_a H[b]`` lands on the nodes and reads the *node* profile. The two
are half a cell apart and are built separately (``wavebit/fd/_lower/cpml.py``); sampling one of them
twice is a mutant the reflection gate sees.

**The stabilizing loss** (:func:`damp`). A stretch backed by a PEC wall feeds energy into an
evanescent field whose tail reaches the wall, so a lossless structure with a bound mode near
the light line grows under a pure CPML (``wavebit/fd/_lower/cpml.py::stabilizing_rate`` has the
mechanism and the numbers). The cure is a small matched loss on the component normal to the
face, inside the slab. It is written here, as two slab multiplies on that one component
around each update -- the curl by ``lift = 1/(1 - h)`` before it and the updated field by
``keep = (1 - h)/(1 + h)`` after it, which is ``keep F + drive cF`` with ``drive = 1/(1 +
h)`` (:func:`damp`) -- and not folded into ``ae``/``be``/``ah``/``bh`` the way an ``Absorber``
is: folded, a vacuum box with a PML on every face would turn five scalar coefficients into five
full-volume arrays (the rates of three axes do not factor into one broadcast), which is 2.5x
the fields' own memory for a loss that lives in the slabs. Written here it costs four
one-component slab multiplies per face per step (on JAX a broadcast multiply XLA fuses into the
update) and no memory beyond two 1-D profiles. Measured 2026-09-19 (CPU, 4 threads, a vacuum
box with a PML on all six faces, ms per step pure -> stabilized): NumPy 7.0 -> 8.2 at 64^3 and
22.1 -> 24.9 at 96^3, Torch 6.8 -> 7.8 at 64^3, JAX 1.11 -> 1.34 at 64^3 and 9.0 -> 9.2 at
96^3 -- the slabs are a larger share of a small box. It is **opt-in** (``PML(stabilize=3e-3)``): a
plain :class:`CpmlFace` is skipped statically, so the default is the pre-stabilization plan,
skeleton and arithmetic exactly, and only a :class:`StabilizedFace` pays for the loss.
"""
import dataclasses

__all__ = ['CpmlFace', 'StabilizedFace', 'psi_key', 'stretch', 'damp', 'loss_key']


@dataclasses.dataclass(frozen=True)
class CpmlFace:
    """One PML face, statically: its name, its axis, the cells it owns, and the two psi shapes.

    ``axis`` is 0, 1 or 2 (x, y, z) and ``side`` is ``'-'`` or ``'+'``; ``region`` is the
    slab of a single-component array the stretch writes. Every field is a Python primitive or
    a frozen dataclass of them, so the face lives in the plan and in its skeleton hash.
    """

    name: str
    axis: int
    side: str
    cells: int
    region: object          # masks.Region

    #: The stabilizing loss ``rho``; 0 on a pure stretch. :class:`StabilizedFace` sets it.
    stabilize = 0.0

    @property
    def normal_slab(self):
        """The slab of the component normal to this face -- where the stabilizing loss acts."""
        from .masks import Slab
        r = self.region
        return Slab(c=self.axis, z=r.z, y=r.y, x=r.x)

    def keys(self):
        """The four psi keys of this face: two curls times the two transverse components."""
        return tuple(psi_key(self.name, field, c)
                     for field in ('E', 'H') for c in range(3) if c != self.axis)


@dataclasses.dataclass(frozen=True)
class StabilizedFace(CpmlFace):
    """A PML face that also carries the stabilizing loss ``stabilize = rho > 0`` (:func:`damp`).

    A subclass rather than a field of :class:`CpmlFace`, so that a pure stretch -- the default
    -- lowers to exactly the plan, and the skeleton, it lowered to before the loss existed: a
    new field would have moved every PML scene's frozen skeleton for a value that is zero.
    """

    stabilize: float = 0.0


def psi_key(face, field, comp):
    """The state key of the psi behind ``face`` for ``D_axis field[comp]``.

    A string, because the state is a plain dict and a dict is a pytree on JAX with no
    registration. ``field`` is the field being differenced (``'E'`` in the curl that feeds
    the H update, ``'H'`` in the one that feeds E), which is also what decides whether the
    half or the node profile is read.
    """
    return f'{face}:{field}:{comp}'


def stretch(d, field, comp, axis_name, plan, params, psi, rt, out=None):
    """``d`` stretched by every PML face on ``axis_name``, and the psi entries it advanced.

    ``d`` is one component's bare difference along ``axis_name``, at full array shape. The
    return is ``(d, updates)`` with ``updates`` a dict of the psi entries this call replaced
    -- the caller merges them, so the JAX path stays functional and the NumPy and Torch paths
    (where the arrays are written through) return the same objects they were given.
    """
    updates = {}
    for face in plan.cpml:
        if face.name[0] != axis_name or comp == face.axis:
            continue
        key = psi_key(face.name, field, comp)
        coefficients = params['cpml'][key]
        d, updates[key] = rt.cpml_stretch(d, face.region, psi[key], coefficients['b'],
                                          coefficients['a'], coefficients['ik'], key)
    return d, updates


def loss_key(face, field):
    """The params key of ``face``'s stabilizing factors for the update of ``field``."""
    return f'{face}:{field}'


def damp(F, factor, field, plan, params, rt):
    """``F`` with every stabilized face's ``factor`` applied to its normal component's slab.

    The matched-loss update of ``wavebit/fd/_lower/coefficients.py`` in product form is
    ``alpha keep F + beta drive cF`` on the slab, with ``keep = (1 - h)/(1 + h)`` and ``drive =
    1/(1 + h)``. The step writes it as ``keep * (alpha F + beta lift cF)`` with ``lift = drive
    / keep = 1/(1 - h)``: the curl is multiplied by ``lift`` before the update
    (``factor='lift'``, ``F`` is the curl) and the updated field by ``keep`` after it
    (``factor='keep'``). Same arithmetic up to one rounding, it needs neither ``alpha`` nor
    ``beta`` (which a material can make arrays), and it never scales the field the step was
    *handed*. How ``rt.slab_scale`` writes the multiply is per backend: in place on the slab
    view on NumPy and Torch, a broadcast multiply on JAX, where every slab write tried first
    cost full-volume copies in the compiled body (``tests/test_fd_pml.py`` pins that it adds
    none).

    ``field`` is the field being updated, ``'E'`` or ``'H'``; it picks the profile (the normal
    E component sits half a cell along the axis, the normal H one on the node). Called after
    the terms, so the loss acts on the whole curl as an ``Absorber``'s ``be`` does. The
    component is the face's axis, so two faces on different axes never touch the same
    component and their corner needs no rule.
    """
    for face in plan.cpml:
        if not face.stabilize:
            continue
        F = rt.slab_scale(F, face.normal_slab,
                          params['cpml_loss'][loss_key(face.name, field)][factor])
    return F
