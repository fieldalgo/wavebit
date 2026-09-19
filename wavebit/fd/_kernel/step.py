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
    H  = ah*H - bh*cE
    H  = damp(H, 'keep')              <- the curl before the update, the field after
    H  = m.after_H(H)                 <- masks: the walls
    cH, psi = curl H
    cH = t.before_E(cH, ...)
    cH = damp(cH, 'lift')
    E  = ae*E + be*cH
    E  = damp(E, 'keep')
    E  = m.after_E(E)
    mem = t.advance(...)              <- term memory, once per step
    obs = o.observe(...)              <- observers read, last

The state is ``dict(E, H, psi, mem, obs)``: the two fields, the CPML memory, one memory entry
per term, one accumulator per observer. It is a pytree, so the JAX scan carries it whole; on
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
(``wavebit/fd/_lower/coefficients.py``).
"""
from .cpml import damp
from .curl import curl

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
    return dict(E=E, H=H, psi=psi, mem=mem, obs=obs)


def step(plan, params, state, n, rt):
    """One leapfrog step; see the module docstring for the slot order."""
    E, H, mem, obs, psi = state['E'], state['H'], state['mem'], state['obs'], state['psi']

    cE, psi = curl(E, 'E', plan, params, psi, rt)
    for t in plan.terms:
        cE = t.before_H(cE, E, mem[t.name], n, params, rt)
    cE = damp(cE, 'lift', 'H', plan, params, rt)
    H = rt.update(H, cE, params['ah'], params['minus_bh'])
    H = damp(H, 'keep', 'H', plan, params, rt)
    for m in plan.masks:
        H = m.after_H(H, rt)

    cH, psi = curl(H, 'H', plan, params, psi, rt)
    for t in plan.terms:
        cH = t.before_E(cH, H, mem[t.name], n, params, rt)
    cH = damp(cH, 'lift', 'E', plan, params, rt)
    E = rt.update(E, cH, params['ae'], params['be'])
    E = damp(E, 'keep', 'E', plan, params, rt)
    for m in plan.masks:
        E = m.after_E(E, rt)

    mem = {t.name: t.advance(mem[t.name], E, H, n, params, rt) for t in plan.terms}
    obs = {o.name: o.observe(obs[o.name], E, H, n, params, rt) for o in plan.observers}
    return dict(E=E, H=H, psi=psi, mem=mem, obs=obs)
