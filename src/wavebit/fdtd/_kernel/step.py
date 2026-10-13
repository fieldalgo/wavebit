"""The kernel step: one leapfrog update as a pure function of ``(plan, params, state, n)``.

With ``c = 1``, ``eps0 = mu0 = 1`` and ``H`` stored as ``Z0 H``, the vacuum leapfrog is

    H(n+1/2) = H(n-1/2) - dt * curl E(n)
    E(n+1)   = E(n)     + dt * curl H(n+1/2)

and the step writes it with the coefficients in ``params`` -- ``ah = ae = 1`` and
``bh = be = dt/dx`` in vacuum, the curl being the bare difference -- so that a material
(stage 1) changes the numbers and not the code. Between the curl and the update sit the
three slots every later feature plugs into, in this fixed order:

    cE, psi = curl E                  <- the CPML stretches the slab inside the curl
    cE = t.before_H(cE, ...)          <- terms: sources, polarisations, TFSF
    cE = damp(cE, 'lift')             <- the PML's stabilizing loss, on its slabs:
    energy = plan.energy.observe(...) <- the interior's energy (NumPy: the row, from H and cE)
    H  = ah*H - bh*cE
    H  = damp(H, 'keep')              <- the curl before the update, the field after
    H  = m.after_H(H)                 <- masks: the walls
    energy = ...observe_magnetic(...) <- (JAX, Torch: its H half, from both H)
    cH, psi = curl H
    cH = t.before_E(cH, ...)
    cH = damp(cH, 'lift')
    E  = ae*E + be*cH
    E  = damp(E, 'keep')
    E  = m.after_E(E)
    mem = t.advance(...)              <- term memory, once per step
    obs = o.observe(...)              <- observers read, last
    energy = ...observe_electric(...) <- (JAX, Torch: the E half of the NEXT step's row)

The interior's energy reads around the H update because that is the one place the step
holds everything its exact form needs -- ``E^n``, ``H^(n-1/2)``, ``H^(n+1/2)`` (or the curl
that makes it) and the pole memory at ``P^(n+1)``, ``P^n`` -- without keeping a field across
steps (``src/wavebit/fdtd/_kernel/observe.py::InteriorEnergy``). ``rt.in_place`` (static per
runtime) says how: NumPy's update destroys ``H^(n-1/2)`` and scales the curl, so it reads the
whole row before the update, from ``E``, ``H`` and the curl. JAX and Torch keep both, so they
read the H half after the update from the two fields -- the compiled step hands the curl to
nothing but its update, which is what keeps the fields bit-identical
(``tests/fdtd/test_fd_stage3_backends.py`` pins them) -- and the E half of the NEXT step's row at
the end of this one, where ``E^(n+1)`` and the advanced pole memory are the step's own
outputs, which is what holds the diagnostic to 9% of a JAX step at 62^3 where reading both
halves at one point costs 43% (``InteriorEnergy.observe_electric``, ``ENERGY_STRIDE``). The
masks, the stabilizing ``lift`` and the ``keep`` in the gap around the update change nothing
that reading sums, and whatever the terms add to ``cE`` is in it exactly as the update
applies it (``InteriorEnergy``). The monitors' observers keep their slot after the E update,
at the time levels their results carry.

The state is ``dict(E, H, psi, mem, obs, energy)``: the two fields, the CPML memory, one memory
entry per term, one accumulator per observer, and the interior-energy accumulator
(``src/wavebit/fdtd/_kernel/observe.py::InteriorEnergy``, ``plan.energy``). It is a pytree, so the
JAX scan carries it whole; on
NumPy the arrays in it are the preallocated buffers the runtime writes into, and the same
dict comes back with the same arrays. ``n`` is a Python int on NumPy and Torch and a traced
int on JAX, so nothing in the step may branch on it.

``psi`` sits beside the fields rather than inside ``mem`` because it is not a term's: the
stretch happens in the curl, before any term runs, and a term that could see it could
reorder it. It is empty (``{}``) for every scene with no ``PML`` face, which is part of what
keeps a vacuum run bit-identical to stage 0.

Stage 1 fills the coefficients too: ``ae``, ``be``, ``ah`` and ``bh`` are scalars in vacuum
and ``(3, nz, ny, nx)`` arrays once a material or an ``Absorber`` makes them so, and the step
is unchanged by which -- that is what "a material changes the numbers and not the code"
bought. The H update reads ``minus_bh`` rather than negating ``bh``, because negating an array
here is a full-volume temporary per step and the lowering can do it once
(``src/wavebit/fdtd/_prepare/coefficients.py``).
Coupled smoothed regions additionally read each source-corrected curl before the update,
integrate compact D/B memory, and add the adjoint cross-material field increment after
the diagonal update (``constitutive.py``). Energy uses E.D and H_old.B_new there. The
flux preparation precedes NumPy's destructive update; it is absent from the numerical
graph of a scene without constitutive terms.
"""
from wavebit.fdtd._kernel.cpml import damp
from wavebit.fdtd._kernel.curl import curl
from wavebit.fdtd._kernel.constitutive import prepare as prepare_constitutive

__all__ = ['step', 'initial_state']


def initial_state(plan, params, rt):
    """The state at ``n = 0``: the initial fields (or zeros), empty memory, zeroed accumulators.

    The wall masks are applied to the initial fields too. A user's initial field that is
    nonzero where a wall says zero -- tangential E on a PEC node layer, say -- would
    otherwise feed the first H update before the first mask ran, and a wall that leaks for
    one step is a source nobody declared. Projecting at ``t = 0`` makes the walls hold from
    the first update on, and makes a half box start from exactly what its full box starts
    from (the symmetry gate found this: random initial data on a PEC wall).
    """
    shape = (3, plan.nz, plan.ny, plan.nx)
    amp = params['amplitude']
    E0, H0 = params['E0'], params['H0']
    E = rt.zeros(shape, plan.dtype) if E0 is None else rt.fresh(E0 * amp)
    H = rt.zeros(shape, plan.dtype) if H0 is None else rt.fresh(H0 * amp)
    for m in plan.masks:
        E = m.after_E(E, rt)
        H = m.after_H(H, rt)
    mem = {t.name: t.init_mem(plan, rt) for t in plan.terms}
    obs = {o.name: o.init(plan, rt) for o in plan.observers}
    psi = {key: rt.zeros(face.region.shape, plan.dtype)
           for face in plan.cpml for key in face.keys()}
    energy = plan.energy.init(plan, rt)
    return dict(E=E, H=H, psi=psi, mem=mem, obs=obs, energy=energy)


def step(plan, params, state, n, rt):
    """One leapfrog step; see the module docstring for the slot order."""
    E, H, mem, obs, psi = state['E'], state['H'], state['mem'], state['obs'], state['psi']

    cE, psi = curl(E, 'E', plan, params, psi, rt)
    for t in plan.terms:
        cE = t.before_H(cE, E, mem[t.name], n, params, rt)
    cE = damp(cE, 'lift', 'H', plan, params, rt)
    mem, pending = prepare_constitutive(plan.constitutive, 'H', cE, mem, params, rt)
    energy = state['energy']
    if rt.in_place:
        # NumPy's update overwrites H and scales cE in place: read the whole row from them
        # first (the E half was read into it at the end of the previous step elsewhere).
        energy = plan.energy.observe(energy, E, H, cE, mem, n, params, rt)
    old = H
    H = rt.update(H, cE, params['ah'], params['minus_bh'])
    for term, delta in pending:
        H = term.add(H, delta, rt)
    H = damp(H, 'keep', 'H', plan, params, rt)
    for m in plan.masks:
        H = m.after_H(H, rt)
    if not rt.in_place:
        # JAX and Torch keep the old H: the H half is the product of the two, and the curl is
        # handed to nothing but its update -- a compiled step that also reads the curl fuses
        # the update otherwise and moves the fields by a rounding
        # (``tests/fdtd/test_fd_stage3_backends.py`` pins them).
        energy = plan.energy.observe_magnetic(energy, old, H, n, params, rt, mem=mem)

    cH, psi = curl(H, 'H', plan, params, psi, rt)
    for t in plan.terms:
        cH = t.before_E(cH, H, mem[t.name], n, params, rt)
    cH = damp(cH, 'lift', 'E', plan, params, rt)
    mem, pending = prepare_constitutive(plan.constitutive, 'E', cH, mem, params, rt)
    E = rt.update(E, cH, params['ae'], params['be'])
    for term, delta in pending:
        E = term.add(E, delta, rt)
    E = damp(E, 'keep', 'E', plan, params, rt)
    for m in plan.masks:
        E = m.after_E(E, rt)

    mem = {t.name: t.advance(mem[t.name], E, H, n, params, rt) for t in plan.terms}
    obs = {o.name: o.observe(obs[o.name], E, H, n, params, rt) for o in plan.observers}
    if not rt.in_place:
        # And the E half of the NEXT step's row, from this step's own outputs: see there.
        energy = plan.energy.observe_electric(energy, E, mem, n + 1, params, rt)
    return dict(E=E, H=H, psi=psi, mem=mem, obs=obs, energy=energy)
