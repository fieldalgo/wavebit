"""The extended-stencil material term: what the step reads once an operator is prepared.

The term is a :class:`~wavebit.fdtd._kernel.constitutive.ConstitutiveTerm`, and that is the
whole of its wiring: ``step`` already calls ``prepare`` on every constitutive term before each
update and adds what it returns after it, and ``InteriorEnergy`` already replaces the diagonal
energy of a constitutive region by ``E . D``. So this class overrides one method and inherits
its slot, its flux memory and its energy reading.

**What it consumes.** Four arrays, prepared once on the host at plan time
(``src/wavebit/fdtd/_prepare/stencil.py``) and split the way every other stage splits:

* ``index`` (int32, ``(active rows, width)``), ``output`` (int32, one entry per row),
  ``diagonal`` (float64) and ``value`` (float64, the off-diagonal coefficients) are
  **static** -- they are fields of this term, so they are in the plan and live exactly as long
  as it does, and they are declared ``metadata={'digest': True}`` so the skeleton records
  their content by SHA-256 instead of writing tens of megabytes into the hash input (commit
  95140d8);
* at lowering, the off-diagonal coefficients are handed to the kernel through
  ``params['stencil'][name]``, beside the existing ``params['constitutive']`` keys, because
  that is where an array the kernel multiplies by is read from.

``value`` used to be held beside the plan in a module-level dict keyed by the operator's
digest, which nothing ever released: every operator prepared in a process lived until the
process did (0.9 MB on the test fixture, 51 MB on the 56-cell research sphere). It is a field
now, so dropping the plan drops the coefficients
(``tests/fdtd/test_fd_stencil.py::test_a_prepared_operator_is_released_with_its_plan``).

``diagonal`` is a field here and is **also** patched into ``be`` at lowering. It has to be in
both: ``be`` is what the step multiplies the curl by and what the interior-energy diagnostic
recovers the medium from, so an effective medium that reached only a private key would be
invisible to the diagnostic that is supposed to see it
(``tests/fdtd/test_fd_stencil.py::test_the_prepared_diagonal_reaches_be_and_the_energy_weight``).
The field is the record of what was prepared; ``be`` is the copy the arithmetic reads.

**The product, and why it is written as a gather.** ``padded`` stores only the rows that have
an off-diagonal entry, each padded to the widest of them, with an exact zero slot at index 0
of the output map -- so a row that couples to nothing gathers the zero and contributes
nothing, with no mask and no branch. One ``take``, one multiply, one sum along the padded
axis, and one scatter back through ``output``. That is the same four lines on NumPy, JAX and
Torch, which is the rule this package is built on.
"""
import dataclasses

from wavebit.fdtd._kernel.constitutive import ConstitutiveTerm

__all__ = ['StencilTerm']


@dataclasses.dataclass(frozen=True, eq=False)
class StencilTerm(ConstitutiveTerm):
    """One prepared extended-stencil region. See the module docstring.

    ``eq=False`` so that the inherited ``__eq__`` and ``__hash__`` -- ConstitutiveTerm's, over
    its own five fields -- keep working: a generated pair over the fields below would compare
    NumPy arrays, and ``plan == other`` would raise rather than answer while ``hash(plan)``
    would fail outright. The arrays are still part of the plan's *content*, through the
    skeleton, which is the thing that has to see them.
    """

    #: Which of the three E components the operator was assembled over. ``(0, 1, 2)`` for a
    #: ball; the research two-dimensional case is ``(1, 2)`` and is not reachable publicly.
    components: tuple = ()
    #: The selector's own word, so the skeleton carries the spelling and not only its effect.
    discretisation: str = 'stencil'
    index: object = dataclasses.field(default=None, metadata={'digest': True})
    output: object = dataclasses.field(default=None, metadata={'digest': True})
    diagonal: object = dataclasses.field(default=None, metadata={'digest': True})
    value: object = dataclasses.field(default=None, metadata={'digest': True})
    #: Which of the scene's structures the operator was prepared for, so that lowering can ask
    #: the same doors of the scene it is handed (``src/wavebit/fdtd/_prepare/stencil.py::lower``).
    structure: int = 0
    #: SHA-256 over the exact arguments :func:`~wavebit.fdtd._prepare.stencil.geometry` was
    #: called with (``stencil.source_digest``). The operator is a function of those and of
    #: nothing else, so this is what lets ``fd.run(scene, plan=...)`` reuse a prepared
    #: operator instead of solving it again, and what lets ``fd.lower`` refuse a scene whose
    #: sphere or permittivity is not the one the operator was fitted for.
    source: str = ''
    #: The assembled operator's spectral upper bound (``stencil.energy_bounds``): what the
    #: time-step door reads, kept so that a reused operator is checked against the run it is
    #: reused for without the matrix it came from.
    spectral_upper_bound: float = 0.0

    def prepare(self, curl, mem, params, rt):
        """The flux memory and the off-diagonal increment, from the source-corrected curl.

        Read before NumPy's diagonal update scales the curl in place, exactly as
        :meth:`ConstitutiveTerm.prepare` is.
        """
        xp = rt.xp
        local = curl[self.view]
        table = params['stencil'][self.name]
        compact = xp.stack(tuple(local[c] for c in self.components)).reshape(-1)
        value = xp.sum(compact[table['index']] * table['value'], axis=1)
        value = xp.concat((xp.zeros_like(compact[:1]), value))[table['output']]
        value = value.reshape((len(self.components),) + self.region.shape)
        delta = xp.stack(tuple(value[self.components.index(c)] if c in self.components
                               else xp.zeros_like(local[c]) for c in range(3)))
        return mem + self.ratio * local, self.ratio * delta
