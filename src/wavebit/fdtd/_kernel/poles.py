"""The ADE term: a dispersive structure's polarisation, as one term with its own memory.

One :class:`~wavebit.fdtd._legacy.Dispersive` structure becomes one :class:`PoleTerm`, whose support is
that structure's snapped slab, so it writes with ``rt.slab_add`` -- the primitive the stage-0
study chose (``studies/fdtd/slab_add_cost/fdtd_slab_add_cost.py``): a 4-D index scatter on JAX, an in-place
``add_`` under a ``mark_dirty`` Function on Torch, ``np.add(view, v, out=view)`` on NumPy.
(The CPML's write is a *different* primitive; see ``cpml.py`` for why the two must not be
unified.)

**The recurrence**, central differences on both derivatives so that ``P^{n+1}`` reads ``E^n``
and never ``E^{n+1}`` -- nothing has to be solved simultaneously (the ``_PoleSet`` seed in
``src/wavebit/fdtd/_periodic/legacy.py``)::

    P^{n+1} = step_gain * P^n - hold * P^{n-1} + drive * E^n

The three coefficients come from the pole type (``materials.py``): Drude and Lorentz derive
them from ``P'' + gamma P' + omega0^2 P = strength E``, Debye from its own first-order ODE
with ``hold = 0``.

**Where it sits in the step, and what the ``dx`` is doing.** Ampere's law with a polarisation
is ``eps dE/dt + sigma E = curl H - dP/dt``, so the E update carries an extra
``-(dP/dt) (dt/eps)/(1 + dt sigma/(2 eps))`` -- exactly the factor ``be`` already carries,
except that ``be`` also folds in the ``1/dx`` the curl leaves out. The kernel therefore adds

    -(dx/dt) * (P^{n+1} - P^n)

into ``cH`` in the ``before_E`` slot, over the structure's slab, and ``be`` turns it into the
right thing. Forgetting the ``dx`` is a mutant that scales with the resolution and so hides
at one ``dx``.

**The one-step shift.** ``advance`` runs at the end of step ``n``, when ``E`` is already
``E^{n+1}``, and it produces the polarisation increment the *next* step's ``before_E`` adds.
Writing ``Q_n`` for what the memory holds after step ``n``, ``Q_n = P^{n+1}``: ``advance``
computes ``Q_{n+1} = step_gain Q_n - hold Q_{n-1} + drive E^{n+1}``, which is
``P^{n+2} = step_gain P^{n+1} - hold P^n + drive E^{n+1}``, and step ``n+1`` (which takes
``E^{n+1}`` to ``E^{n+2}``) adds ``-(P^{n+2} - P^{n+1})/dt``. That is the standard leapfrog
ADE, index for index. The memory starts at rest, which drops one step's drive at ``t = 0``;
it is exactly zero for any scene whose initial field does not overlap the dispersive
structure, which is every scene a packet gate builds.
"""
import dataclasses

__all__ = ['PoleTerm']


@dataclasses.dataclass(frozen=True)
class PoleTerm:
    """The polarisation of one dispersive structure: ``slabs`` (one per component), ``n_poles``.

    Static, a frozen dataclass of primitives, so it lives in the plan and in its skeleton
    hash. Its numbers -- ``step_gain``, ``hold``, ``drive``, and the ``dx/dt`` the update
    needs -- are traced leaves and live in ``params['poles'][name]``, which is what makes a
    gradient with respect to a pole strength flow.
    """

    name: str
    region: object          # masks.Region: the cells, without the component axis
    slabs: tuple            # three masks.Slab, one per component, over the same cells
    n_poles: int

    def init_mem(self, plan, rt):
        """``(P, P_previous, delta)`` per pole, at rest: three slab-shaped arrays each.

        ``delta`` is carried rather than recomputed for the same reason ``psi`` is: it is the
        output of a recurrence that the next step's ``before_E`` reads before the next
        ``advance`` runs, and a term's memory is the only place a value survives a step.
        """
        shape = self.region.shape
        zeros = [rt.zeros((3,) + shape, plan.dtype) for _ in range(3 * self.n_poles)]
        return {'P': tuple(zeros[:self.n_poles]),
                'previous': tuple(zeros[self.n_poles:2 * self.n_poles]),
                'delta': tuple(zeros[2 * self.n_poles:])}

    def before_H(self, cE, E, mem, n, params, rt):
        return cE

    def before_E(self, cH, H, mem, n, params, rt):
        """Add ``-(dx/dt) * sum(delta)`` into ``cH`` over the slab, component by component."""
        scale = -params['poles'][self.name]['dx_over_dt']
        for c, slab in enumerate(self.slabs):
            total = None
            for delta in mem['delta']:
                total = delta[c] if total is None else total + delta[c]
            cH = rt.slab_add(cH, slab, scale * total)
        return cH

    def advance(self, mem, E, H, n, params, rt):
        """One step of every pole's recurrence, driven by ``E`` over the slab."""
        coefficients = params['poles'][self.name]
        # One view of all three components at once -- the slabs differ only in ``c`` -- so
        # the drive costs no copy on NumPy or Torch and no gather on JAX.
        drive_field = E[(slice(None),) + self.region.slices]
        P, previous, delta = [], [], []
        for k in range(self.n_poles):
            step_gain = coefficients['step_gain'][k]
            hold = coefficients['hold'][k]
            drive = coefficients['drive'][k]
            nxt = step_gain * mem['P'][k] - hold * mem['previous'][k] + drive * drive_field
            P.append(nxt)
            previous.append(mem['P'][k])
            delta.append(nxt - mem['P'][k])
        return {'P': tuple(P), 'previous': tuple(previous), 'delta': tuple(delta)}
