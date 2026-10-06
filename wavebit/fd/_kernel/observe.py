"""Observers: what the step reads after the updates, into accumulators the state carries.

An observer has an accumulator that is part of the run's state -- a ``(steps,)`` array for
a point record -- and ``observe(acc, E, H, n, params, rt)`` returns the accumulator with
step ``n`` written. Written, not appended: the JAX scan carries the accumulator as an
ordinary array and sets one element per step, so the same protocol runs unchanged inside
``lax.scan``, a Python loop, and a checkpointed block. Observers never write to a field.

Time levels: an E component is read after the E update of step ``n``, at ``(n + 1) dt``; an H
component after the H update, at ``(n + 1/2) dt``. ``times`` says which for a given observer,
and the result carries them.

**Stage 3's one frequency-domain observer, ``DFTRegion``.** A running DFT of named components
over a region at a static list of frequencies,

    F^(f) = sum_n F(t_n) exp(+i 2 pi f t_n) dt,

the positive-frequency phasor of the ``e^{-i w t}`` convention (a real ``Re(A e^{-iwt})`` running
for a time ``T`` gives ``~ A T / 2``). **Each component is transformed at its own time level**:
E at ``t_n = (n + 1) dt`` and H at ``t_n = (n + 1/2) dt``, the times the observer reads them
at. That is what makes the E and H phasors co-timed without interpolation: the leapfrog's
``H(n+1/2) - H(n-1/2) = -dt curl E(n)`` becomes, summed against ``exp(i w t)`` over a run that
starts and ends at rest, ``-i Omega H^ = -curl E^`` with ``Omega = 2 sin(w dt/2)/dt`` exactly,
and the same for Ampere's law -- the discrete Maxwell equations in the frequency domain, which
every post-processing identity (a flux that closes, an order split into its two directions)
rests on. A run that stops before the field has decayed breaks those identities by the
amount still in the region, which is what the decay diagnostic measures.

The region is one :class:`Slab` per component (each component its own Yee points), the
frequencies are static (in the plan and its skeleton) and the accumulator is one complex
``(nf, *slab.shape)`` array per component, carried in the scan; ``check()`` prints the bytes.
Beside it, ``energy`` records ``sum |F|^2`` over the region once per step -- ``(steps,)``
reals -- which is what the always-on decay diagnostic reads. It carries no gradient (it is a
diagnostic, and it would otherwise drag the whole region into the backward pass).

**The interior's energy, :class:`InteriorEnergy`** -- no monitor's: the field energy of
everything but the PML layers (an ``Absorber``'s included), read every :data:`ENERGY_STRIDE`
steps, which is what the growth diagnostic judges (``wavebit/fd/_assemble.py::decay_report``).
It lives beside the observers (``plan.energy``, ``state['energy']``), not among them, so a
plan's observers are still exactly its monitors', and it reads in its own slots of the step --
around the H update, and not with the observers after the E update (the class says why).
"""
import dataclasses

import numpy as np

__all__ = ['PointRecord', 'DFTRegion', 'InteriorEnergy', 'ENERGY_STRIDE']

#: Every how many steps :class:`InteriorEnergy` reads the interior. The read is a weighted
#: ``sum |F|^2`` over the whole interior and one inner product, which on NumPy costs 0.15 ms at
#: 44^3 and 0.57 ms at 62^3 against a 2.6 and a 6.4 ms step -- 6 to 9% if it read every step.
#:
#: **What the always-on diagnostic costs**, measured 2026-09-21 tree against tree: each
#: checkout timed on its own kernel loop, no stub and no monkeypatch anywhere, since a stub is
#: what got this wrong before (below). ``fd_helpers.dipole_box`` with an ``eps`` Region and a
#: Lorentz Region, so ``ae``/``be``/``ah``/``bh`` are ``(3, N, N, N)`` and a pole is stepped;
#: 320 steps, best of seven, ms per step, against the commit before this observer existed:
#:
#:                                        NumPy        Torch        JAX (jit)
#:     44^3   no observer at all          2.594        3.025        0.3185
#:            the reading replaced        +2.3%        +1.8%        +3.1%
#:            this reading, split         +1.4%        +3.3%        +11.0%
#:            this reading, in one                     +3.8%        +18.4%
#:     62^3   no observer at all          6.404        6.030        1.391
#:            the reading replaced        -0.4%        +2.3%        +5.4%
#:            this reading, split         +0.2%        +3.3%        +9.4%
#:            this reading, in one                     +2.6%        +43.0%
#:
#: So it is **9 to 11% of a JAX step** -- about three and a half times the reading it replaced
#: at 44^3, and the number a large run should be sized with -- and inside the run-to-run spread
#: on NumPy and Torch, whose steps are 8 to 20 times longer here: repeating the whole table on a
#: busier machine moved JAX's rows by a point or two and moved NumPy's and Torch's between -8%
#: and +6%, which is what "inside the spread" means. (NumPy reads in one point by construction:
#: its update destroys ``H^(n-1/2)``, so it has no "in one" row to separate.) An earlier table
#: read -1.0% and -0.5% for JAX and called the reading free; it was taken against a stub of
#: ``observe``, which is the entry point JAX and Torch never call in this split design, so both
#: sides of it ran the same program (review, 2026-09-20).
#:
#: **"In one" against "split"** is where the exact reading is taken
#: (``InteriorEnergy.observe_electric``): both halves at one point, which on a compiled backend
#: is a reduction over ``E^n`` -- one of the scan's carries -- next to one over ``H^(n-1/2)``,
#: against the two taken apart (the E half at the end of the previous step, over that step's
#: own outputs; the H half after the H update, over the carry and the new field). The split is
#: still the choice, and by more than the old numbers claimed for it: 43.0% -> 9.4% at 62^3 and
#: 18.4% -> 11.0% at 44^3 (repeated: 35.0% -> 10.5% and 21.5% -> 12.0%). It is not the sums,
#: the weights or the poles: with the weights 1 and the poles 0 the cost was the same, and each
#: reduction alone cost under 0.05 ms (``scratchpad`` probes, 2026-09-20). Torch, which
#: compiles nothing here, is inside its spread either way.
#:
#: **What costs is the branch, not the sample.** The same tree with the stride so long that a
#: 320-step run reads two rows instead of twenty-one runs at 0.3611 against 0.3672 ms a step on
#: JAX at 44^3 (+10.1% against +12.0% over no observer) -- the stride's own value, within its
#: spread. ``rt.when`` is a ``lax.cond`` in the scan's body and the body carries it at every
#: step, sampling or not; so the stride buys NumPy's time and the accumulator's rows, and on
#: JAX it buys nothing.
#: Rejected, measured the same way: handing the step's curl to the reading -- NumPy's own
#: shape, given to every backend -- is the most expensive shape measured, JAX +25.1% at 44^3
#: and +41.6% at 62^3 against no observer, and it moves the JAX fields by a rounding -- 1.1e-16
#: in E and 5.3e-16 in H against the same run with no observer at all, where this split is bit
#: for bit that run, XLA having no reason to stop fusing the update -- and every frozen golden
#: with them. As pairs against the same probe's committed shape, and buying nothing:
#: recomputing the curl inside the sampled branch (2.72 against 1.97 ms at 62^3) and moving
#: the H update into that branch (0.384 against 0.378 ms at 44^3).
#: A NumPy sample allocates 0.11-0.13 MiB whatever the grid (einsum's own fixed buffer; the
#: weights go into one scratch per part, ``rt.energy_weight``), and the steps between samples
#: allocate nothing and cost 1 us.
#:
#: The statistic takes means over quarters of at least one period each, and what this observer
#: reads is the leapfrog's exactly conserved energy (the class), with no ripple to sample, so
#: the samples need not resolve the carrier: at 20 cells per wavelength a 3-D period is
#: ``20 sqrt(3) / courant`` steps -- 69 at the default courant 0.5 (4.3 samples a period), 35
#: at courant 1. A first version read ``eps E . E + mu (|H^(n-1/2)|^2 + |H^(n+1/2)|^2)/2``,
#: which is off by ``dt^2 |curl E|^2 / (2 mu)`` -- a ripple at twice the carrier, 4e-3 of the
#: energy at 20 cells per wavelength but up to 0.57 near 2 .. 4 cells per wavelength -- and
#: wherever ``2 omega * stride dt`` sat near a multiple of ``2 pi`` the stride sampled that
#: ripple as a slow beat the quarter means did not average out: the exact Yee modes (14, 12,
#: 12) and (14, 13, 11) of a closed 16^3 PEC box (``fd_helpers.cavity_box``) read 1.165 and
#: 1.38 and one of them warned growing (review, 2026-09-20). They now read 1.0000 at every
#: stride and courant measured (``tests/test_fd_decay.py`` pins both).
ENERGY_STRIDE = 16


@dataclasses.dataclass(frozen=True)
class PointRecord:
    """``Record`` lowered: ``field`` ``'E'``/``'H'``, component ``c``, indices ``k, j, i`` (z, y, x)."""

    name: str
    field: str
    c: int
    k: int
    j: int
    i: int

    def init(self, plan, rt):
        return rt.zeros((plan.steps,), plan.dtype)

    def observe(self, acc, E, H, n, params, rt):
        F = E if self.field == 'E' else H
        return rt.record(acc, n, F[self.c, self.k, self.j, self.i])

    def times(self, plan):
        """The sample times, ``(steps,)``."""
        shift = 1.0 if self.field == 'E' else 0.5
        return (np.arange(plan.steps) + shift) * plan.dt


@dataclasses.dataclass(frozen=True)
class DFTRegion:
    """A running DFT of ``parts`` at ``freqs`` -- the lowering of every stage-3 monitor.

    ``parts`` is ``((label, field, Slab), ...)``: the accumulator's key, ``'E'`` or ``'H'``, and
    that component's region. ``monitor`` is the public monitor it belongs to (a box monitor
    lowers to one region per face, named ``monitor::face``). ``dt`` is the plan's. All static;
    the angular frequencies ``2 pi f`` are a params leaf (``params['dft'][name]``) only so that
    each backend holds them as its own array.
    """

    name: str
    monitor: str
    freqs: tuple
    parts: tuple
    dt: float

    def init(self, plan, rt):
        nf = len(self.freqs)
        acc = {label: rt.zeros((nf,) + tuple(slab.shape), 'complex128')
               for label, _, slab in self.parts}
        acc['energy'] = rt.zeros((plan.steps,), 'float64')
        return acc

    def observe(self, acc, E, H, n, params, rt):
        omega, dt = params['dft'][self.name], self.dt
        # Each field at its own time level, the dt of the sum folded into the phase.
        phase = {'E': rt.xp.exp(1j * (omega * ((n + 1.0) * dt))) * dt,
                 'H': rt.xp.exp(1j * (omega * ((n + 0.5) * dt))) * dt}
        out, energy = {}, None
        for label, field, slab in self.parts:
            view = (E if field == 'E' else H)[slab.slices]
            out[label] = rt.dft_add(acc[label], view, phase[field], f'dft[{self.name}]{label}')
            e = rt.energy(view)
            energy = e if energy is None else energy + e
        out['energy'] = rt.record(acc['energy'], n, energy)
        return out

    def nbytes(self, plan):
        """The accumulator's bytes: complex128 per region point per frequency, plus ``energy``."""
        size = sum(int(np.prod(slab.shape)) for _, _, slab in self.parts)
        return 16 * len(self.freqs) * size + 8 * plan.steps


@dataclasses.dataclass(frozen=True)
class InteriorEnergy:
    """The energy of everything but the PML layers, every ``stride`` steps: ``(rows,)``.

    ``lo`` and ``hi`` are the judged cells' range ``(z, y, x)``: every cell but a PML's. A PML's
    field is no physical energy (the stretch is a change of coordinates), and what it hands
    back is exactly what the growth diagnostic is there to catch. An ``Absorber``'s is: the
    layer is a matched electric and magnetic conductivity backed by its PEC wall, a passive
    medium whose energy only falls -- and it holds what it has not yet absorbed and gives part
    of it back, so leaving it out made an ``Absorber(32)`` column read 2.57 and warn (review,
    2026-09-20: the wave that went in comes back out after the layer's round trip, 3.2
    wavelengths deep, into an interior the pulse had already left; counted, see
    ``tests/test_fd_decay.py``).

    **The weights** are ``eps`` for E and ``mu`` for H per Yee component, read back off the
    coefficients the step multiplies by (``wavebit/fd/_lower/coefficients.py``): with
    ``ae = (1 - q)/(1 + q)`` and ``be = (dt/dx)/(eps (1 + q))`` for the loss ``q = dt rate/2``,

        eps = (dt/dx) (1 + ae) / (2 be),      mu = (dt/dx) (1 + ah) / (2 bh)

    exactly, in a lossy cell as in a lossless one (``1 + ae = 2/(1 + q)``), and 1 in vacuum.
    A first version read ``(dt/dx)/be``, which is ``eps (1 + q)``: exact only where there is
    no loss. They are read at every sample (into one scratch per field on NumPy,
    ``rt.energy_weight``) rather than stored: stored, they were two more interior-sized
    arrays -- two more fields of memory in any scene with a structure -- held for a
    diagnostic. Unweighted, a pulse leaving a dielectric would gain energy: a wave in ``eps``
    carries ``|H|^2 = eps |E|^2``, so ``|E|^2 + |H|^2`` is ``(1 + eps)/(2 eps)`` of its energy,
    and it rises 1.85x as a pulse leaves silicon (``eps = 12``).

    **What is read, and why it is flat.** The leapfrog conserves, in a closed lossless box,

        W^n = eps E^n . E^n + mu H^(n-1/2) . H^(n+1/2)

    exactly (``Re`` of the conjugate products in a complex run): ``W^(n+1) - W^n`` is
    ``eps (E^(n+1) - E^n) . (E^(n+1) + E^n)`` plus ``mu H^(n+1/2) . (H^(n+3/2) - H^(n-1/2))``;
    the E update makes the first ``dt (E^(n+1) + E^n) . curl H^(n+1/2)`` and two H updates make
    the second ``-dt H^(n+1/2) . curl (E^(n+1) + E^n)``, and summed over a closed grid the two
    cancel, because the discrete curls of E and of H are each other's transpose (summation by
    parts; the same identity is what bounds the Yee scheme under its CFL limit). A loss only
    takes from it. **It is read where the step holds all of it**, with no field kept from one
    step to the next -- in two halves whose split is what brings it down from 43% of a JAX step
    at 62^3 to 9% (``ENERGY_STRIDE``, :meth:`observe_electric`):

        the E half   ``eps E^n . E^n`` and every pole's energy, at the END of step ``n - 1``
                     (``step.py``), where ``E^n`` and the advanced pole memory are that step's
                     own outputs -- or, where the update is in place (NumPy), at the start of
                     step ``n`` with the rest
        the H half   ``mu H^(n-1/2) . H^(n+1/2)`` after the H update of step ``n``, from the
                     carry ``H^(n-1/2)`` and the field the update has just made; and where the
                     update is in place, before it, from the curl ``cE`` it is about to use:
                     ``mu ah |H^(n-1/2)|^2 + mu minus_bh H^(n-1/2) . cE``, the same product
                     written with ``H^(n+1/2) = ah H^(n-1/2) + minus_bh cE``

    Whatever the terms add to the curl (a source, a TFSF face) is in ``cE``, so it is in the
    product exactly as the update applies it, and the two writings agree to round-off. That is
    a claim about a term's drive, so it is gated on a scene that has one inside the judged
    cells: a **magnetic** ``Dipole``, whose ``M`` is what ``before_H`` writes into the curl
    (``tests/test_fd_decay.py`` holds NumPy's every-step series against JAX's and Torch's
    there, to 1.5e-15 of the run's peak, and off the bare curl the row is out by 11%; an
    electric ``Dipole`` drives ``before_E`` and a closed box has no term at all, which is why
    the cross-backend gate alone did not reach this). The masks, the stabilizing ``lift`` and
    the ``keep`` in the same gap change nothing it sums: a wall mask zeroes cells that were
    zeroed after the previous update too (and at ``t = 0``), so their ``H^(n-1/2)`` is 0 and so
    is their product, and ``lift`` and ``keep`` scale one face's ``normal_slab``, which is that
    face's own PML cells (``cpml.py``) -- outside ``lo .. hi`` by construction, so neither
    needs a gate. Measured (``tests/test_fd_decay.py``, 2026-09-20), read at every step
    in closed boxes of random grid-scale field: 1.3e-15 .. 1.4e-14 (max / min - 1) with PEC,
    PMC and ``Symmetry`` faces, three mirrors meeting, a Bloch box of complex field, courant 0.5
    and 0.9; 4e-13 for the stride's alias modes over 3000 steps; round-off in lossless pole
    cavities and in half-filled ``eps = 12`` and ``mu = 12`` columns. A first version kept the
    pair ``eps E . E + mu (|H^(n-1/2)|^2 + |H^(n+1/2)|^2)/2`` across two steps, which is ``W^n
    + dt^2 |curl E^n|^2 / (2 mu)``: off by a term second order in ``omega dt`` that oscillates at
    twice the carrier, and which the stride aliased into a steady rise for grid-scale content
    (the exact Yee mode (14, 12, 12) of a closed 16^3 PEC box read 1.165, spread 1.36, and
    warned growing at 3000 steps; review, 2026-09-20). No ripple is left to alias. (The naive
    ``eps |E^n|^2 + mu |H^(n-1/2)|^2`` is off at first order: 8.2e-2 in a closed cavity.)

    **A mirror plane halves what lies on it.** A ``Symmetry`` face's half box is half of a full
    box whose energy is twice the half box's off-plane cells plus the plane's once, so a
    component lying on the plane (unmasked: tangential E and the normal H on a low face's node
    plane of parity +1, the normal E and tangential H on a high face's half plane of parity -1)
    weighs 1/2 -- 1/4 or 1/8 where two or three planes meet. ``planes`` is ``((coefficient,
    field, c, box), ...)`` (``wavebit/fd/_lower/plan.py::_mirror_planes``): parts read like the
    whole and added with their coefficients. Counted whole, the half box's reading swings 4e-2.
    Each part is one component, so it takes that component's own ``eps`` or ``mu``
    (:meth:`_cut`'s ``comps``) -- a different number from component 0's only where the plane
    cuts an anisotropic structure, which is the one configuration that gates it: an
    ``eps = (2, 3, 5)`` slab flush against the plane reads 4.7e-15 .. 7.5e-15 as written and
    swings 1.3e-2 .. 1.7e-2 off component 0 (``tests/test_fd_decay.py``, 2026-09-21).

    **The rows** are steps ``1, 1 + stride, 1 + 2 stride, ...`` and the run's last step
    (:meth:`sampled`); row ``k`` is ``T^n`` at ``t = n dt``, the time of ``E^n``, so the series
    reaches the end of the run. A row is written by the two halves above, the E half first
    (at the end of the step before), so a run stopped between them would leave the last row
    half written -- it cannot: the last step sampled is ``steps - 1``, whose E half was written
    at ``steps - 2``. Not step 0: the pole memory starts at rest (``poles.py``), which
    drops one step's drive, so ``T^0`` is off the identity's path in a scene whose initial field
    overlaps a dispersive structure (a Debye cavity kicked in its field rises 6.0e-4 from
    ``T^0`` to ``T^1``, then falls at every step).

    **A dispersive structure's own energy is in it too, exactly.** Write the second-order pole
    (Drude, Lorentz; ``poles.py``) as the kernel steps it, ``P^(m+1) = g P^m - h P^(m-1) +
    d E^m`` with the E update ``eps (E^(m+1) - E^m) = dt curl H^(m+1/2) - (P^(m+1) - P^m)``.
    It is central differences of ``P'' + gamma P' + omega0^2 P = s E`` with

        a = 2 / (dt^2 (1 + h)),   s = d a,   omega0^2 = 2/dt^2 - g a,   gamma = (1 - h) a dt,

    so the pole constants are read back off the recurrence's own coefficients
    (``params['poles']``) and a traced pole needs nothing else. With ``J^(m+1/2) = (P^(m+1) -
    P^m)/dt`` the field energy moves by ``F^(m+1) - F^m = -dt (E^(m+1) + E^m) . J^(m+1/2)``,
    and the pole's

        V^(m+1/2) = (|J^(m+1/2)|^2 + omega0^2 P^(m+1) . P^m) / s

    by ``V^(m+1/2) - V^(m-1/2) = dt E^m . (J^(m+1/2) + J^(m-1/2)) - (gamma dt / (2 s))
    |J^(m+1/2) + J^(m-1/2)|^2`` (substitute the recurrence). The two do not cancel term by term;
    their sum does once the coupling ``- E^m . (P^(m+1) - P^m)`` is added, so

        T^m = F^m + V^(m+1/2) - E^m . (P^(m+1) - P^m)

    is conserved exactly when ``gamma = 0`` and never rises when it is not -- the pole's
    analogue of the ``H^(n-1/2) . H^(n+1/2)`` above: a product of consecutive values and a
    coupling term, no midpoint (derived here from the recurrence, by the same summation the
    field's identity takes; the review of 2026-09-20 set it out).

    **The time levels, checked against the kernel.** ``advance`` runs at the end of step
    ``n - 1``, when ``E`` holds ``E^n``, and leaves ``mem['P'] = P^(n+1)`` and
    ``mem['previous'] = P^n`` (``poles.py``, "the one-step shift"): step ``n`` adds
    ``-(P^(n+1) - P^n)`` to ``E^n``. So where this observer reads, at the start of step ``n``,
    every term of ``T^n = F^n + V^(n+1/2) - E^n . (P^(n+1) - P^n)`` is at hand. Measured on
    closed 1-D PEC cavities (``tests/test_fd_decay.py`` pins them; 2026-09-20): a Lorentz at
    ``omega0 = 2 pi`` with strength 1, 10 and 40 ``omega0^2``, one at ``4 omega0``, a Drude at
    ``omega_p`` 1, 3 and 6 x ``2 pi``, dx 0.05 and 0.1, courant 0.5 and 0.9 -- ``T`` holds to
    7e-16 .. 1.1e-14 (max / min - 1), and a damped Lorentz's never rises past round-off; read by
    this observer at every step, the strong ones hold to 1.3e-15 .. 2.9e-15. A first version
    read the pole as ``(|P'|^2 + omega0^2 |P|^2)/s`` with ``P'`` and ``P`` the difference and
    the mean of the last two values -- half a step late and without the coupling, first order
    in ``dt`` -- and swung 9e-2 .. 2.1 on the same cavities (0.80 for the strong Lorentz at the
    default courant, which read 1.47 and warned; review, 2026-09-20). Dropping only the coupling
    swings as much.

    **Debye** (``hold = 0``) runs the split ``P = P~ + c E`` (``materials.Debye``): the memory
    holds ``P~`` and ``c = delta_eps (1 - g)/2`` is in the E update's permittivity,
    ``eps_inf + c``, with ``delta_eps = 2 d / (1 - g^2)``. The trapezoid it steps makes
    ``F^(m+1) - F^m`` (with ``eps_inf``) ``= -(2 tau / (dt delta_eps)) |P^(m+1) - P^m|^2 -
    (|P^(m+1)|^2 - |P^m|^2) / delta_eps``, so ``T^m = F^m + |P^m|^2 / delta_eps`` never rises.
    ``P~`` is at the kernel's shift too, so ``P^n = mem['previous'] + c E^n``; the E weight
    above carries ``eps_inf + c``, so the pole adds ``|P^n|^2 / delta_eps - c |E^n|^2``.
    Measured on the same cavity (tau 0.01 and 0.05, delta_eps 1 and 5): ``T`` never rises (the
    largest step is a fall); read at ``mem['P']``, a step late, it rose by up to 10% in one
    step. ``poles`` is ``((term name, n_poles, cells, offsets), ...)``: the part
    of each dispersive structure inside the judged cells, and where that part starts within the
    structure's own slab.

    It carries no gradient -- the sums are ``rt.weighted_energy`` and ``rt.real_inner``, which
    stop it, and ``rt.constant`` stops it on the pole coefficients, which are leaves a gradient
    can ask for -- and ``rt.when`` keeps the step free of a branch on ``n``, which is traced
    under JAX: a Python ``if`` on NumPy and Torch, a ``lax.cond`` on JAX, so the steps between
    samples compute nothing. The weights are read at every sample (``rt.energy_weight``, with
    ``ah`` and ``minus_bh`` folded in for the product) into one scratch per part on NumPy.
    """

    lo: tuple
    hi: tuple
    stride: int
    dt: float
    ratio: float
    steps: int = 0
    poles: tuple = ()
    planes: tuple = ()
    constitutive: tuple = ()

    def sampled(self, steps):
        """The steps read: every ``stride``-th from step 1, and the run's last."""
        return sorted(set(range(1, steps, self.stride)) | ({steps - 1} if steps > 1 else set()))

    def rows(self, plan):
        return len(self.sampled(plan.steps))

    def init(self, plan, rt):
        return rt.zeros((self.rows(plan),), 'float64')

    @property
    def box(self):
        """The judged cells, ``((z0, z1), (y0, y1), (x0, x1))``."""
        return tuple(zip(self.lo, self.hi))

    @staticmethod
    def _cut(b, comps, box):
        """A coefficient cut to ``comps`` (a slice) and ``box`` on the axes it varies along:
        ``b`` is a scalar or a ``(3, nz, ny, nx)`` array that is 1 long on an axis it does not
        vary on."""
        if getattr(b, 'ndim', 0) != 4:
            return b
        return b[(comps if b.shape[0] > 1 else slice(None),)
                 + tuple(slice(lo, hi) if b.shape[a + 1] > 1 else slice(None)
                         for a, (lo, hi) in enumerate(box))]

    def _weight(self, params, field, rt, comps=slice(None), box=None, times=None, key=''):
        """``eps`` (``field`` ``'E'``) or ``mu`` (``'H'``) over ``box`` (the judged cells by
        default), times the coefficient ``times`` names (``'ah'``, ``'minus_bh'``): see the
        class."""
        box = self.box if box is None else box
        a, b = ('ae', 'be') if field == 'E' else ('ah', 'bh')
        scale = 1.0 if times is None else self._cut(params[times], comps, box)
        return rt.energy_weight(self.ratio, self._cut(params[a], comps, box),
                                self._cut(params[b], comps, box),
                                f'energy[{field}{times or ""}{key}]', scale)

    def electric(self, E, mem, params, rt):
        """``eps E^n . E^n`` over the judged cells plus every pole's stored energy at that
        level (the class): the E half of ``T^n``."""
        total = self._electric(E, mem, params, rt, slice(None), self.box, '')
        for coefficient, field, c, box in self.planes:
            if field == 'E':
                total = total + coefficient * self._electric(E, mem, params, rt,
                                                             slice(c, c + 1), box, f'{c}{box}')
        for term in self.constitutive:
            if term.field == 'E':
                total = total + self._constitutive(term, E, mem, params, rt)
        return total

    def magnetic(self, H, new, cE, params, rt, mem=None):
        """``mu H^(n-1/2) . H^(n+1/2)`` over the judged cells: the H half of ``T^n``, from
        ``new = H^(n+1/2)`` (after the H update) or from the curl ``cE`` the update is about to
        use (before it, ``new`` ``None``)."""
        total = self._magnetic(H, new, cE, params, rt, slice(None), self.box, '')
        for coefficient, field, c, box in self.planes:
            if field == 'H':
                total = total + coefficient * self._magnetic(H, new, cE, params, rt,
                                                             slice(c, c + 1), box, f'{c}{box}')
        for term in self.constitutive:
            if term.field == 'H':
                total = total + self._constitutive(term, H, mem, params, rt, new, cE)
        return total

    def _constitutive(self, term, field, mem, params, rt, new=None, curl=None):
        """Replace diagonal energy by E.D or H_old.B_new in a lossless region.

        The plan keeps the region outside PML and refuses nonzero cross
        coefficients on a mirror. Any admitted mirror intersection is diagonal;
        applying the same mirror weights to this correction preserves that rule.
        Magnetic flux memory has already advanced before either reading.
        """
        parts = [(1., slice(None), term.box)]
        for coefficient, kind, c, box in self.planes:
            if kind == term.field:
                cut = tuple((max(a, u), min(b, v))
                            for (a, b), (u, v) in zip(term.box, box))
                if all(b > a for a, b in cut):
                    parts.append((coefficient, slice(c, c+1), cut))
        total = 0.
        for coefficient, comps, box in parts:
            view = (comps,) + tuple(slice(a, b) for a, b in box)
            local = (comps,) + tuple(slice(a-base[0], b-base[0])
                                    for (a, b), base in zip(box, term.box))
            actual = rt.real_inner(field[view], mem[term.name][local], 1.)
            if term.field == 'E':
                diagonal = rt.weighted_energy(field[view], self._weight(
                    params, 'E', rt, comps, box, key=term.name + str(comps)))
            else:
                diagonal = self._magnetic(field, new, curl, params, rt, comps, box,
                                           term.name + str(comps))
            total = total + coefficient * (actual-diagonal)
        return total

    def _electric(self, E, mem, params, rt, comps, box, key):
        """:meth:`electric` over components ``comps`` of ``box``."""
        view = (comps,) + tuple(slice(lo, hi) for lo, hi in box)
        total = rt.weighted_energy(E[view], self._weight(params, 'E', rt, comps, box, key=key))
        for name, n_poles, cells, offsets in self.poles:
            spans = tuple((max(c0, lo), min(c1, hi)) for (c0, c1), (lo, hi) in zip(cells, box))
            if all(b > a for a, b in spans):
                shifted = tuple(o + (a - c0) for o, (a, _), (c0, _) in
                                zip(offsets, spans, cells))
                total = total + self._pole(mem[name], E, params['poles'][name], n_poles,
                                           spans, shifted, self.dt, rt, comps)
        return total

    def _magnetic(self, H, new, cE, params, rt, comps, box, key):
        """:meth:`magnetic` over components ``comps`` of ``box``."""
        view = (comps,) + tuple(slice(lo, hi) for lo, hi in box)
        old = H[view]
        if new is not None:
            return rt.real_inner(old, new[view], self._weight(params, 'H', rt, comps, box,
                                                              key=key))
        # The same product with H^(n+1/2) = ah H^(n-1/2) + minus_bh cE, the update about to
        # run: two sums over the old H, none over a field that does not exist yet.
        return (rt.weighted_energy(old, self._weight(params, 'H', rt, comps, box, 'ah', key))
                + rt.real_inner(old, cE[view],
                                self._weight(params, 'H', rt, comps, box, 'minus_bh', key)))

    def _row(self, n):
        """``(row, sampled)`` for step ``n``: which row it belongs to and whether it is read.
        The row index is static under a trace only through ``n``; a run of one step has no row
        at all, and the callers return before indexing an accumulator of no rows."""
        return ((n + self.stride - 2) // self.stride,
                (n >= 1) & (n < self.steps)
                & (((n - 1) % self.stride == 0) | (n == self.steps - 1)))

    def observe(self, acc, E, H, cE, mem, n, params, rt):
        """Row ``ceil((n - 1) / stride)`` gets the whole of ``T^n`` -- the in-place reading, from
        the state before the H update (``step.py``; ``rt.in_place``)."""
        if self.steps < 2:
            return acc                      # a one-step run has no row
        k, when = self._row(n)
        return rt.when(when, lambda a: rt.record(
            a, k, self.electric(E, mem, params, rt) + self.magnetic(H, None, cE, params, rt, mem)),
            acc)

    def observe_electric(self, acc, E, mem, n, params, rt):
        """The E half of row ``ceil((n - 1) / stride)``, read at the END of step ``n - 1``,
        where ``E^n`` and the pole memory are the step's own outputs (``step.py``; the
        functional runtimes). On a compiled backend that is what keeps the reading down: a
        reduction over a value the step has just made is cheaper than one over the scan's carry
        -- ``E^n`` at the start of step ``n``, next to ``H^(n-1/2)`` -- which on JAX costs
        +43.0% of a step at 62^3 against this split's +9.4%, and +18.4% against +11.0% at 44^3
        (tree against tree, 2026-09-21; :data:`ENERGY_STRIDE` has the table)."""
        if self.steps < 2:
            return acc                      # a one-step run has no row
        k, when = self._row(n)
        return rt.when(when,
                       lambda a: rt.record(a, k, self.electric(E, mem, params, rt)), acc)

    def observe_magnetic(self, acc, H, new, n, params, rt, mem=None):
        """The H half of row ``ceil((n - 1) / stride)``, added to what
        :meth:`observe_electric` wrote, after the H update of step ``n`` (``step.py``)."""
        if self.steps < 2:
            return acc                      # a one-step run has no row
        k, when = self._row(n)
        return rt.when(when, lambda a: rt.record(
            a, k, a[k] + self.magnetic(H, new, None, params, rt, mem)), acc)

    @staticmethod
    def _pole(mem, E, coefficients, n_poles, cells, offsets, dt, rt, comps=slice(None)):
        """The stored energy of one dispersive structure's poles over ``cells`` (the part of
        its slab from ``offsets``), components ``comps``, at the level of ``E^n``:
        ``V^(n+1/2) - E^n . (P^(n+1) - P^n)``, or Debye's ``|P^n|^2 / delta_eps - c |E^n|^2``
        (the class)."""
        inside = (comps,) + tuple(slice(o, o + (c1 - c0))
                                  for o, (c0, c1) in zip(offsets, cells))
        field = E[(comps,) + tuple(slice(c0, c1) for c0, c1 in cells)]
        total = 0.0
        rank_one = 'energy_rank_one' in coefficients
        sum_new = sum_old = None
        for j in range(n_poles):
            if 'energy_inverse_delta' in coefficients:
                # The induced interface Debye pole varies by cell. Weight each cell
                # before reducing, including component and clipped-region selections.
                # Supplied inverse strength avoids 0/0 at inactive cells and cancellation
                # in 1-gain**2 near a pure conductor. Existing scalar poles keep their path.
                shape = mem['P'][j].shape
                inv_delta, c = (rt.xp.broadcast_to(rt.constant(coefficients[key][j]),
                                                  shape)[inside]
                                for key in ('energy_inverse_delta', 'energy_instantaneous'))
                previous = mem['previous'][j][inside]
                total = total + rt.weighted_energy(previous + c * field, inv_delta)
                total = total - rt.weighted_energy(field, c)
                continue
            if 'energy_inverse_strength' in coefficients:
                # Spatial Drude/Lorentz oscillators need local weights before reduction.
                # Explicit metadata also identifies an oscillator when gamma*dt=2 makes
                # hold zero. Inactive cells contribute nothing, including the cross term.
                shape = mem['P'][j].shape
                inverse, potential, active = (
                    rt.xp.broadcast_to(rt.constant(coefficients[key][j]), shape)[inside]
                    for key in ('energy_inverse_strength', 'energy_resonance_over_strength',
                                'energy_active'))
                new, old = mem['P'][j][inside], mem['previous'][j][inside]
                step = new-old
                total = total + rt.weighted_energy(step/dt, inverse)
                total = total + rt.real_inner(new, old, potential)
                total = total - rt.real_inner(field, step, active)
                if rank_one:
                    # Physical polarization components share one normal restoring
                    # force. Mask inactive slots before forming its single sum.
                    held_new, held_old = active*new, active*old
                    sum_new = held_new if sum_new is None else sum_new+held_new
                    sum_old = held_old if sum_old is None else sum_old+held_old
                continue
            gain, hold, drive = (rt.constant(coefficients[key][j])
                                 for key in ('step_gain', 'hold', 'drive'))
            # Which kind is static wherever the coefficients are concrete (NumPy, Torch), so
            # one branch is computed; under a JAX trace both are, and ``where`` picks. A pole
            # with no strength (``Lorentz(strength=0)``, ``Debye(delta_eps=0)`` are legal) is
            # never driven and stores nothing: skipped when concrete, and under a trace its
            # denominators are kept finite and its reading zeroed.
            kind, strength = _static(hold), _static(drive)
            if strength == 0.0:
                continue
            live = None if strength is not None else drive != 0.0
            new, old = mem['P'][j][inside], mem['previous'][j][inside]
            second = debye = None
            if kind is None or kind != 0.0:
                scale = 2.0 / (dt * dt * (1.0 + hold))
                s = drive * scale
                step = new - old
                second = ((rt.weighted_energy(step, 1.0) / (dt * dt)
                           + (2.0 / (dt * dt) - gain * scale) * rt.real_inner(new, old))
                          / (s if live is None else rt.xp.where(live, s, 1.0))
                          - rt.real_inner(field, step))
            if kind is None or kind == 0.0:
                # Debye's delta_eps divides by 1 - gain^2, which a second-order pole can zero;
                # where that is not the branch taken, the denominator is kept finite.
                one = 1.0 - gain * gain
                delta_eps = 2.0 * drive / (one if kind is not None
                                           else rt.xp.where(hold == 0.0, one, 1.0))
                c = 0.5 * delta_eps * (1.0 - gain)
                debye = (rt.weighted_energy(old + c * field, 1.0)
                         / (delta_eps if live is None else rt.xp.where(live, delta_eps, 1.0))
                         - c * rt.weighted_energy(field, 1.0))
            if second is None:
                value = debye
            elif debye is None:
                value = second
            else:
                value = rt.xp.where(hold == 0.0, debye, second)
            total = total + (value if live is None else rt.xp.where(live, value, 0.0))
        if rank_one:
            # The diagonal potential above uses the unshifted bulk resonance.
            # Add alpha/kappa Re(sum(Pnew) conj(sum(Pold))) once, after summing
            # active physical polarizations in this component/region selection.
            shape = mem['P'][0].shape
            weight = rt.xp.broadcast_to(rt.constant(coefficients['energy_rank_one']),
                                        shape)[inside]
            total = total + rt.real_inner(sum_new, sum_old, weight)
        return total

    def series(self, plan, acc):
        """``(t, W)`` from the finished accumulator (NumPy): row ``k`` is ``T^n`` at the
        ``k``-th step :meth:`sampled` names, at the time of ``E^n``."""
        W = np.asarray(acc, dtype=np.float64).reshape(-1)
        return np.asarray(self.sampled(plan.steps), dtype=np.float64) * plan.dt, W


def _static(x):
    """``x`` as a Python float, or ``None`` when it has no value yet (a JAX tracer)."""
    try:
        return float(x)
    except Exception:
        return None
