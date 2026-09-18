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
"""
import dataclasses

__all__ = ['CpmlFace', 'psi_key', 'stretch']


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

    def keys(self):
        """The four psi keys of this face: two curls times the two transverse components."""
        return tuple(psi_key(self.name, field, c)
                     for field in ('E', 'H') for c in range(3) if c != self.axis)


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
