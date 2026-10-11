"""Typed domain objects at the API boundary -- ``Material`` / ``Layer`` / ``Stack`` /
``Incidence``.

Phase 2 of the rewrite: the geometry and illumination a user
configures become typed objects instead of a bare ``structure`` array plus loose
``thickness`` / ``theta`` / ``psi`` scalars. These are an *adapt-in* layer -- the same
keyword arguments :func:`wavebit.rcwa` takes, and a ``Stack`` round-trips losslessly to and
from them (``Stack.from_structure(structure, thickness)`` rebuilds the exact ``structure``).

A ``structure`` is layer-major on its first axis in every encoding the solver accepts, so a
single layer is exactly ``(thickness[i], structure[i])``:

  - raster   : ndarray ``(n_layers, Ny, Nx)`` of PERMITTIVITY -- one sample per pixel
  - parametric : list of layer dicts -- one dict per layer

There are three places in there that say what a region is made of, and two of them -- the
parametric layer's ``background`` and the parametric shape's ``material`` -- are
**assignments**: a name for the stuff that fills a region. :class:`Material` is that name, and
it carries permittivity and permeability together because a medium has both. The raster is the
third and stays arrays: it is not an assignment but a *sampled field*, one number per pixel,
which is exactly why topology optimisation reaches for it -- so it keeps a separate array per
quantity rather than a grid of objects (``structure`` for permittivity, ``mu`` beside it).

``Stack`` therefore holds the ordered layers plus the surrounding media (``top``, ``bot``,
``period``); ``Incidence`` holds the illumination. Numerics and backend choice (``truncation_order``,
``fourier_coefficients``, ``factorization``, ``backend``, ``complex_dtype``) stay as
:func:`wavebit.rcwa` arguments -- they are not part of the physical scene.

Backend-agnostic throughout: a raster grid, and equally a :class:`Material`'s ``epsilon``, may
be a NumPy / JAX / PyTorch value; stacking goes through the Array-API namespace and
:class:`Material` is a JAX pytree, so gradients survive both.
"""
import sys
from cmath import isfinite as _complex_isfinite, sqrt as _complex_sqrt
from dataclasses import dataclass, field, fields
from functools import cache
from typing import Any, Optional

from .conventions import (_COMPLEX_CHI, _DIRECTION_CLASH, incidence_clashes, reject_complex_angle,
                          transverse_component_problem, wavelength_problem)
from .backend import _concrete, array_namespace, principal_sqrt
from .backend import eager_under_torch_compile
from .enums import Tetm  # noqa: F401  (re-exported for spec's tetm parsing)


def _is_raster(value: Any) -> bool:
    """True if ``value`` is an array grid (raster) rather than a shape recipe."""
    return not isinstance(value, (list, tuple, dict)) and hasattr(value, 'shape')


# --------------------------------------------------------------------------- material


_PYTREE_REGISTERED = False


def _register_pytree_if_jax_loaded():
    """Register :class:`Material` as a JAX pytree, if JAX is in the process already.

    A material is a legitimate *design variable* -- the parametric path already tolerates a
    tracer in a shape's coordinates, and there is no reason permittivity should be the one
    parameter you cannot differentiate. For that to work JAX has to be able to see inside a
    ``Material``: an unregistered class is an opaque **leaf**, so ``jax.grad`` has nothing to
    put a tangent on and ``tree_map`` walks straight past the number it was aimed at. That is
    the same defect class as the aux_data bugs already fixed on this branch, where a value
    that should have been a child was carried as static metadata.

    Registration is lazy for the reason ``backends/jax_backend.py`` lifts x64 on import: ``import
    wavebit`` must not import JAX on behalf of a caller who will never use it, and importing
    JAX has real side effects here (that module flips ``jax_enable_x64`` process-wide). So
    this reads ``sys.modules`` and never imports -- the same "act only if JAX is already
    there" pattern as
    :func:`wavebit._enable_jax_x64_if_already_imported`. :class:`Material` cannot use the
    other pattern in the codebase, the one :mod:`wavebit._core.result` uses, because that one
    registers a *backend subclass* (``ResultJax``) while the class users actually construct
    here is the shared one.

    Called from ``Material.__post_init__``, so the class is registered by the time any
    ``Material`` exists in a process that has JAX. The one ordering this cannot cover is
    building a ``Material`` before ``import jax`` and then never building another before
    handing that one to JAX; there the class is still a leaf, and JAX says so loudly
    (``TypeError: ... is not a valid JAX type``) rather than silently dropping the gradient.
    Constructing any ``Material`` afterwards -- which every wavebit ingest path does --
    registers the class and repairs the objects already built, since registration is per
    class and not per instance.
    """
    global _PYTREE_REGISTERED
    if _PYTREE_REGISTERED:
        return
    jax = sys.modules.get('jax')
    if jax is None:
        return
    tree_util = getattr(jax, 'tree_util', None)
    if tree_util is None:
        return                      # jax is mid-import; the next Material will try again
    try:
        tree_util.register_pytree_node(
            Material, Material.tree_flatten, Material.tree_unflatten)
    except ValueError:
        pass                        # already registered; idempotent by design
    _PYTREE_REGISTERED = True


@eager_under_torch_compile
@dataclass(frozen=True)
class Material:
    """What fills a region: relative permittivity, and the permeability that goes with it.

    ``Material(epsilon, mu=None)``. Both are **relative** to vacuum and both are complex in
    general; ``Im(epsilon) > 0`` is loss in this project's ``e^{-iwt}`` convention, which is
    also what refractive-index tables mean by ``n + 1j*k`` (``docs/conventions.md``).

    Each quantity can be scalar, three diagonal entries ``[xx, yy, zz]``, or a
    physical 3x3 tensor in the public xyz frame. Assign tensors to a layer
    background or shape and call ``solve()``; internal convolution signs are
    handled automatically. ``rotated(angle, axis)`` rotates material axes.
    Uniform full epsilon tensors and in-plane tensor patterns are
    supported. Patterned z coupling and anisotropic half-spaces are not.

    **Why permittivity rather than index.** Everything past the front door already works in
    permittivity -- the Fourier factorizations, the tensor builders and ``solve_conv`` all
    take it -- and the front door used to square an index on the way in. One quantity had to
    win, and an index cannot carry the pair: ``mu`` was always taken raw, so a single material
    argument spelled as an index would have meant "index" for one of its two numbers and
    "permeability" for the other. It is also the direction with no branch cut in it. ``eps =
    n**2`` is unambiguous where ``n = sqrt(eps)`` is a choice (:func:`index_from_eps`), and the
    half-space square root is on this project's short list of recurring traps, so the boundary
    is a bad place to have to make that choice repeatedly.

    The flip is not free and is not silent-safe: the same literal
    ``3.5`` read as an index gives ``R = 0.3086`` and read as a permittivity ``R = 0.0920``,
    both energy-conserving, which is why the retired spellings are refused by name rather than
    quietly accepted, and why ``tests/test_material_convention_anchor.py`` pins the convention
    against closed-form Fresnel instead of against a regenerated golden.

    **Scalar shorthand.** Anywhere a material is accepted, a bare number means
    ``Material(epsilon=x)``. Non-magnetic is the overwhelming default and must cost nothing to
    write, so ``material=12.25`` and ``Material(12.25)`` are the same input -- and, by the
    paragraph below, that shorthand leaves ``mu`` at ``None``. The coercion happens in exactly
    one function, :func:`as_material`; a scalar is *not* ``==`` to its Material (see below),
    because the shorthand is an input convenience and not a claim that a number and a medium
    are the same kind of thing.

    **``mu = None`` is how "non-magnetic" is spelled, and it is the default** -- not
    ``mu = 1.0``. Every reader of ``mu`` outside the arithmetic is asking one question, "must
    this solve build the permeability path?", and that question has to be answerable under
    ``jax.jit``. ``mu`` is a pytree *child* (it is a material value, and material values are
    what gradients are for), so inside a trace it is a tracer and ``mu != 1`` is a traced
    boolean that no ``if`` can take. ``None`` is an **empty pytree subtree**: it contributes no
    leaf, so it comes through a trace as itself, and :attr:`is_magnetic` stays a question about
    *structure* rather than about a value -- answerable at trace time, always. This file
    already spells absence that way twice, at :attr:`Incidence.polarization` and at the raster
    :attr:`Layer.mu`, for the same reason.

    ``Material(4.0)`` and ``Material(4.0, mu=1.0)`` are therefore **different materials**: the
    first names no permeability, the second names the value one. They compare unequal, they
    flatten to different treedefs, and the second gets the permeability path built for it --
    which computes the numbers the non-magnetic path would, only slower. The distinction is
    kept crisp deliberately: reading a literal ``1.0`` back as absence would make the treedef,
    and so the jit cache key, depend on whether the caller happened to write the default, and
    that is the sort of coupling that surfaces a long way from where it was introduced.
    :attr:`mu_value` is there for the consumers that need the number rather than the question.

    **Frozen, and a JAX pytree** whose children are ``(epsilon, mu)`` -- see
    :func:`_register_pytree_if_jax_loaded` for why registration matters and why it is lazy.
    Frozen because a material handed to a solve must not change under it, and because a pytree
    that can be mutated after flattening is a cache-coherence bug waiting to happen.

    **Values are passed through untouched** -- no ``float()``, no ``complex()``, no array
    conversion. A ``Material`` must be able to hold a ``jax`` tracer, and any coercion here
    would either drop the tangent or fail outright at trace time. It follows that a
    ``Material`` is exactly as well-behaved as the values in it, which is the whole story on
    equality and hashing:

    * ``Material(2.0) == Material(2.0)`` and their hashes agree, so materials work as dict
      keys and dedup in a set. Identity semantics were the alternative and would have made two
      spellings of one medium compare unequal, defeating any such dedup silently.
    * With an array ``epsilon``, ``==`` inherits that array's semantics (a size>1 NumPy array
      raises the usual ambiguous-truth ``ValueError``) and ``hash`` inherits its
      unhashability. Deliberately not papered over: a container that reports a clean answer
      for values that cannot give one would be lying about the values.

    Two conversions live beside this class for callers who hold the other quantity:
    :func:`eps_from_index` and :func:`index_from_eps`.
    """

    epsilon: Any
    mu: Any = None
    _wavelength_shape: Any = field(default=None, repr=False, compare=False, kw_only=True)

    def __post_init__(self):
        _register_pytree_if_jax_loaded()

    def at(self, wavelength):
        """Return this constant material unchanged at any wavelength.

        For wavelength-dependent values, use :class:`wavebit.TabulatedMaterial`
        or ``wavebit.materials.load(name)``.
        """
        return self

    @property
    def is_anisotropic(self):
        """Whether epsilon or mu is supplied as three principal values or a 3x3 tensor.

        This selects tensor arithmetic structurally, including an isotropic diagonal
        tensor. Scalars retain the ordinary scalar material path.
        """
        from .anisotropy import material_is_tensor
        return material_is_tensor(self)

    def rotated(self, angle, axis='z'):
        """Return a material actively rotated about x, y or z by radians.

        Tensor components are in Wavebit's xyz axes. Both epsilon and mu rotate
        as ``R @ tensor @ R.T``; scalars are unchanged. Array backend and material
        and angle derivatives are retained. This rotates the material axes, not
        the shape to which the material is assigned.
        """
        from .anisotropy import rotated_material
        return rotated_material(self, angle, axis)

    @property
    def is_magnetic(self) -> bool:
        """Whether the solve must build the permeability path for this material.

        A **structural** question -- "was a permeability named?" -- rather than the physical
        one, "is this permeability different from one?". That is exactly what makes it
        answerable inside a trace: ``self.mu is not None`` reads the pytree's *shape*, which
        JAX keeps concrete, where ``self.mu != 1`` would read its *value*, which under
        ``jax.jit`` is a tracer whose truth cannot be taken.

        The property exists so that no reader has to write that comparison inline, and the
        reason it is worth a property is that the three backends' ``solve_conv`` did write
        it: ``mu_top != 1.0`` raised ``TracerBoolConversionError`` for *every* traced solve,
        magnetic or not, which took the ``jax.jit``-decorated ``solve_field`` out of
        service for every scene there is. A predicate that can only be evaluated eagerly is not
        a predicate a jitted solver can use.

        The consequence to keep in view: ``Material(2.25, mu=1.0)`` answers ``True``. It named
        a permeability, so it gets the permeability path -- the same numbers, more work.
        Absence is ``None``, absence is the default, and the scalar shorthand produces it, so
        no scene that never mentions permeability pays for this.
        """
        return self.mu is not None

    @property
    def mu_value(self):
        """The permeability as a *number*: ``mu`` if one was named, otherwise ``1.0``.

        Absence turned back into the value it stands for, once, for the consumers that
        *multiply* by ``mu`` rather than branch on it. The one today is the permeability
        convolution build (:meth:`wavebit._core.engine.CoreFactorizationMixin._mu_structure`),
        which runs the permittivity's own builder over a structure whose material slots hold
        permeabilities: in a scene something else made magnetic, a region that named none is
        still a region, and ``1.0`` is what it contributes. A ``None`` there would reach the
        arithmetic and raise.

        It is deliberately *not* what :attr:`is_magnetic` reads, and the two must not be
        collapsed into one accessor: the whole point of the ``None`` spelling is that the
        question ("was one named?") stays answerable inside a trace while the value need not
        be. Consumers that only ask the question -- the routing in ``solve_conv``, the
        ``sqrt(eps*mu)`` of :func:`wavebit._core.engine._half_space_index` -- read
        :attr:`is_magnetic` and then ``mu`` itself, so a non-magnetic medium's index stays
        ``sqrt(eps)`` to the bit rather than being multiplied by a manufactured one.
        """
        return 1.0 if self.mu is None else self.mu

    # pytree protocol -- registered lazily; see _register_pytree_if_jax_loaded.
    # Both values are children rather than aux_data: aux_data is hashed into the jit cache
    # key, so a swept permittivity carried there would retrace on every step and a traced one
    # could not be carried at all. A `mu` of `None` is an empty subtree and contributes no
    # leaf, so a non-magnetic material flattens to the single number it is -- which is what
    # keeps `is_magnetic` readable on the way back out of a trace.
    def tree_flatten(self):
        return (self.epsilon, self.mu), self._wavelength_shape

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, _wavelength_shape=aux_data)


def as_material(value, where='material') -> Material:
    """A :class:`Material` from whatever a caller wrote: a Material, a number, or nothing.

    **The single coercion point.** Every path that accepts a material calls this, so the
    scalar shorthand is defined once and cannot drift between the parametric layer's
    ``background`` and the parametric shape's ``material`` key -- the two spellings that
    :class:`Material` exists to unify. ``where`` names the slot in the error message
    (``'stack.top'``, ``'shapes[2].material'``), matching how ``_core/spec.py`` reports paths.

    ``None`` is the *absent* material and becomes vacuum, ``Material(1.0)`` -- permittivity
    one, and no permeability named, so it is non-magnetic by :attr:`Material.is_magnetic`'s
    structural reading. Writing ``Material(1.0, 1.0)`` here instead would make the default
    material of every unfilled slot request the permeability path, which is the opposite of
    what "nothing was said here" means. This spells the default; it does not decide whether
    absence is legal, which is the caller's business -- a slot where a material is mandatory
    should reject ``None`` before it gets here.

    Containers are refused rather than unpacked. A ``str`` is the tempting one: there is no
    named-material lookup in wavebit, and quietly accepting ``'Si'`` somewhere would create
    the expectation of one. A ``list`` / ``tuple`` / ``dict`` at a material slot is almost
    always a shape or layer recipe that landed one level too deep, and it is worth saying so.
    Arrays and tracers are *not* refused: shape inspection cannot tell a vmapped tracer from a
    raster grid, and refusing both to catch one would break the design-variable case this type
    was built for.
    """
    if isinstance(value, Material):
        if value._wavelength_shape is not None:
            raise ValueError(f'{where}: a material evaluated at an array of wavelengths is a spectrum; '
                             'evaluate at one wavelength or assign the material model')
        return value
    from .materials import TabulatedMaterial
    if isinstance(value, TabulatedMaterial):
        raise TypeError(f'{where}: evaluate a TabulatedMaterial with .at(wavelength), '
                        'or assign it directly to an RCWA material slot')
    if value is None:
        return Material(1.0)                    # vacuum, and mu unnamed -- see the docstring
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        raise TypeError(
            f'{where}: expected a Material or a number (the scalar shorthand for '
            f'Material(epsilon=x)), got {type(value).__name__}. '
            + ('Materials are numeric; wavebit has no lookup from a material name.'
               if isinstance(value, (str, bytes)) else
               'A list or dict here is usually a shape or layer recipe nested one level '
               'too deep.'))
    return Material(value)


@eager_under_torch_compile
def eps_from_index(n, *, mu=None):
    """Convert isotropic refractive index to relative permittivity.

    Return ``n**2`` when ``mu`` is omitted (relative permeability one), or
    ``n**2 / mu`` when it is supplied. Both quantities may be complex and must
    describe the same wavelength. Scalars and elementwise array batches are
    supported; this is not a conversion from an anisotropic index tensor.
    NumPy/JAX/PyTorch values retain their backend and differentiation graph.

    Supplied permeability must be finite and nonzero. Invalid values raise
    ``ValueError`` eagerly; traced JAX arrays produce NaN at invalid entries.
    Assign the same permeability to the resulting material, for example
    ``Material(eps_from_index(n, mu=mu), mu=mu)``.
    """
    if mu is None:
        return n ** 2
    message = 'index conversion requires finite, nonzero relative permeability mu'
    if not hasattr(mu, 'dtype'):
        if not _complex_isfinite(mu) or mu == 0:
            raise ValueError(message)
        return n ** 2 / mu
    xp = array_namespace(mu)
    valid = xp.isfinite(mu) & (mu != 0)
    if _concrete(xp.all(valid)) is False:
        raise ValueError(message)
    # Avoid dividing by zero even in unselected traced branches.
    safe_mu = xp.where(valid, mu, xp.ones_like(mu))
    return xp.where(valid, n ** 2 / safe_mu, float('nan'))


@eager_under_torch_compile
def index_from_eps(eps):
    """Refractive index of a medium with relative permittivity ``eps`` -- the passive root.

    The square root of a complex number has two values and only one of them is a physical
    index here. This takes the **principal** root, and that is not a choice made by this
    function so much as one verified against the convention: a passive medium has
    ``Im(eps) >= 0`` (loss, in ``e^{-iwt}``), which puts ``eps`` in the closed upper half
    plane, whose principal root lies in the first quadrant -- so ``Im(n) >= 0``, loss again.
    The same argument is what makes :func:`wavebit._core.conventions.boundary_kz` principal
    for real ``k_par``, and it is the reason no branch rule is invented here.

    **The hazard is the negative side of zero.** The principal cut runs along the negative
    real axis, and IEEE signed zero decides which lip a value sits on: ``complex(-2, +0.0)``
    gives ``+1.4142j`` (an evanescent, decaying medium -- correct) while ``complex(-2, -0.0)``
    gives ``-1.4142j``, a *gain* medium, from an imaginary part that is not merely small but
    identically zero with a sign bit set. Spell the zero with ``complex()``: whether the
    expression ``-2 - 0j`` carries ``-0.0`` depends on the Python version (``+0.0`` on 3.12 and
    3.13; ``-0.0`` from 3.14, which converts the real operand of mixed real-complex arithmetic
    to a float rather than to a complex, CPython gh-69639). The same trap at a smaller scale: a
    lossless value that has been through arithmetic can come out as ``4 - 1e-18j``, and its
    root ``2 - 2.5e-19j`` is nominally active. Neither is repaired here, because a helper cannot
    tell round-off from a deliberately active medium, and clamping would quietly make gain
    media unrepresentable. Callers that build a permittivity by subtraction and then take a
    root should keep ``Im`` non-negative themselves.

    **Always complex**, even for ``eps = 2.25``, which returns ``1.5+0j``. Two reasons. A real
    ``sqrt`` of a metal's negative permittivity is ``nan`` -- silent, and metals are a
    supported material class. And the alternative, returning a real index when the value
    happens to be real and non-negative, makes the *dtype* depend on the value: it would flip
    part-way through a sweep that crosses zero, and under ``jax.jit`` the sign cannot be read
    at all, so eager and traced calls would disagree about the output type. Uniform beats
    convenient; a caller that has verified its medium is lossless can take ``.real``.

    **Not a magnetic medium's index.** That is
    :func:`~wavebit._core.conventions.signed_index`, ``sqrt(eps)*sqrt(mu)``: the principal root
    of ``eps*mu`` for an ordinary medium and its negation for a negative-phase-velocity one,
    every passive double negative and passive magnetic metal among them
    (``docs/negative-index-media.md`` section 6). This takes no ``mu`` on purpose: the sign
    needs both factors, and their product has lost it.

    Backend-agnostic: arrays and tracers go through their own namespace's ``sqrt`` (with a
    width-preserving promotion to complex -- ``float32`` in gives ``complex64`` out), and
    Python scalars go through :mod:`cmath`. All of them take the lip a signed zero names --
    JAX's ``sqrt`` takes the upper one for both zeros, and
    :func:`~wavebit._core.backend.principal_sqrt` turns it there -- so a value on the cut gets
    one root on every backend. Elsewhere they are each their library's ``sqrt``, which need not
    agree to the last bit: :mod:`cmath` and NumPy differ by one or two ulp in a component on 45
    of 2,000 random products ``eps*mu`` (``studies/negative_index_media.py``, section 13).
    """
    if hasattr(eps, 'dtype'):
        xp = array_namespace(eps)
        if not xp.isdtype(eps.dtype, 'complex floating'):
            # Multiplying by a Python complex 1 promotes without naming a width, so a
            # single-precision caller is not silently widened to complex128. Exact: the real
            # part is multiplied by 1.0 and the (zero) imaginary part contributes nothing.
            eps = eps * (1 + 0j)
        return principal_sqrt(xp, eps)
    return _complex_sqrt(complex(eps))


# --------------------------------------------------------------------------- the attribute door


@cache
def _declared_fields(cls):
    """The field names of dataclass ``cls``, once per class: the hot path of every assignment."""
    return frozenset(f.name for f in fields(cls))


@eager_under_torch_compile
class _DeclaredFieldsOnly:
    """Base of the mutable spec dataclasses: an assignment to a name that is no field is refused.

    ``inc = Incidence(...); inc.thet = 0.2`` used to be stored as a new attribute, and
    ``inc.as_kwargs()`` still wrote ``theta=0`` -- the solve answered for the angle that was
    *not* changed, with nothing to notice. The solver has refused that since 2026-08-27
    (``CoreFactorizationMixin.__setattr__``) and the excitation facade since its own door was
    closed; these six (``Layer``, ``Stack``, ``Incidence`` here, ``Numerics``, ``Sweep``,
    ``JobSpec`` in ``_core/spec.py``) were the objects a spec is built from and kept the hole.
    ``Material`` and ``TabulatedMaterial`` are frozen and refuse every assignment already.

    The rule is the solver's, so all three doors predict alike: a declared field, a leading
    underscore, or a name already on the class (which keeps the descriptor protocol in charge
    -- ``stack.thickness = ...`` gets Python's own "has no setter", not "did you mean", and a
    method stays stubbable on an instance). The refusal is the solver's too,
    :func:`wavebit._core.engine._unknown_attribute`, with its near-match hint drawn from the
    fields -- so ``Numerics.nesting`` is pointed at ``li_nesting``, as ``solver.nesting`` is.

    **A base class rather than** ``slots=True``. Slots refuse the typo too, but measured on a
    slotted twin they answer ``'Incidence' object has no attribute 'thet'`` -- no hint, and not
    the solver's message; they refuse ``_note`` and an instance stub (``object attribute 'check'
    is read-only``) that the solver allows, so the doors would disagree about which names are
    checked; and ``vars(inc)`` raises for want of a ``__dict__``. ``__init__``,
    ``__post_init__`` and ``dataclasses.replace`` assign declared fields, so they pass the first
    branch; pickle and deepcopy restore ``__dict__`` directly and never come through here.

    Branches on ``name`` and the class only: no instance state is read (``__init__`` assigns
    into an empty ``__dict__``) and no value is inspected (a tracer passes as any value does).
    """

    def __setattr__(self, name, value):
        cls = type(self)
        if name in _declared_fields(cls) or name.startswith('_') or hasattr(cls, name):
            super().__setattr__(name, value)
            return
        from .engine import _unknown_attribute     # lazily: engine imports this module
        raise _unknown_attribute(cls.__name__, name, _declared_fields(cls), kind='fields')


#: The sentence every retired material spelling has to carry. The rename is the *smaller* half
#: of what changed, and a message that only said "use X instead of Y" would send a user to a
#: keyword that accepts their old number and answers a different question with it.
#:
#: The figures are the README quickstart solved twice -- once as written, once with every
#: material literal left at the index it used to be (the layer's 3.5 and the substrate's 1.5
#: read as permittivities). They are deliberately *close*: a migration that broke loudly would
#: not need refusing. 0.2053 against 0.1753 is the size of a truncation-order wobble, which is
#: what a user would take it for. Re-measure them if the quickstart scene ever changes.
@eager_under_torch_compile
@dataclass
class Layer(_DeclaredFieldsOnly):
    """One layer of a :class:`Stack`: a thickness paired with the structure filling it.

    ``structure`` is the layer's entry of the stack's ``structure`` in either encoding -- a
    raster grid array ``(Ny, Nx)`` of permittivity, or a parametric layer dict.

    It is a *structure*, not a material, which is why the field is no longer called ``eps``:
    the raster case is a single quantity end to end, and even there it is one sample per pixel
    rather than one value. The materials live inside the other encoding, where they are
    :class:`Material` (or a number, via the scalar shorthand).

    ``mu`` is the raster's field-form companion and is ``None`` everywhere else: on the two
    shape encodings a permeability is part of the :class:`Material` that names the region, so
    a second slot here would be a second place to say it.

    **What the values mean is decided by the ingest layer, not here.** This class transports
    them: whatever is written into ``structure`` is what :attr:`Stack.structure` hands to
    :func:`wavebit.rcwa`, unread and unconverted. That is deliberate -- a second place that
    coerced a material would be a second place a convention could be decided, and the flip to
    permittivity is exactly the kind of change that survives only if there is one such place
    (:func:`as_material` for the assignment paths, ``docs/conventions.md`` for the quantity
    itself).
    """
    thickness: float
    structure: Any
    mu: Any = None

    @property
    def is_raster(self) -> bool:
        return _is_raster(self.structure)


@eager_under_torch_compile
@dataclass
class Stack(_DeclaredFieldsOnly):
    """An ordered list of :class:`Layer` plus the media that bound it.

    Layer order is bottom-to-top, matching ``structure``'s first axis (the same order the
    solver stacks them). ``top`` / ``bot`` are the bounding superstrate / substrate and
    ``period`` is the in-plane period, and ``lattice`` the basis when the cell is not
    axis-aligned -- ``period`` alone cannot describe one, since it reports the two edge lengths
    and the angle is already gone.

    The two half-space slots take a :class:`Material` or a number, and -- like
    :attr:`Layer.structure` -- **pass it through untouched** to :func:`wavebit.rcwa` via
    :meth:`as_kwargs`. They are not coerced here. A domain object that converted would be a
    second reader of the material convention sitting in front of the real one, so that the
    same ``Stack`` could mean different physics depending on which door it went through; the
    conversion belongs to ingest, which is also where the retired spellings are refused by
    name. The field names are the ones ``_core/spec.py`` serializes, so they are the ingest
    layer's to rename.
    """
    layers: list = field(default_factory=list)
    top: Any = 1.0
    bot: Any = 1.0
    period: Any = (1.0, 1.0)
    #: The cell basis, columns ``a1`` / ``a2``, or ``None`` for the axis-aligned cell
    #: ``period`` spells. Carried because :class:`~wavebit._core.spec.JobSpec` calls itself a
    #: *complete* description of a run and could not describe an oblique cell without it --
    #: ``period`` reports ``(|a1|, |a2|)`` there, which does not determine the basis. Kept as
    #: the plain ``(2, 2)`` rather than a :class:`~wavebit._core.lattice.Lattice` for the same
    #: reason the half-spaces stay uncoerced: this layer passes through, and ingest converts.
    #: A ``Lattice`` is accepted too and writes through its array form -- it grew ``tolist``
    #: for exactly this boundary, since being accepted here and refused by ``to_json`` was the
    #: worst of the three options.
    lattice: Any = None

    @classmethod
    def from_structure(cls, structure, thickness, top=1.0, bot=1.0, period=(1.0, 1.0),
                       mu=None, lattice=None) -> "Stack":
        """Adapt a ``(structure, thickness)`` pair into a typed ``Stack``.

        ``structure`` may be either encoding (raster ndarray / parametric list); it is
        layer-major on its first axis, so layer *i* is ``(thickness[i], structure[i])``. The
        inverse of :attr:`structure` — ``Stack.from_structure(s, t).structure`` reproduces
        ``s``. ``mu`` is the raster path's optional permeability field, layer-major the same
        way, and ``lattice`` the cell basis for a skewed cell -- give that rather than
        ``period``, which on one reports only the two edge lengths.
        """
        if structure is None:
            layers = []
        else:
            layers = [Layer(t, structure[i], None if mu is None else mu[i])
                      for i, t in enumerate(thickness)]
        return cls(layers=layers, top=top, bot=bot, period=period, lattice=lattice)

    @property
    def thickness(self) -> list:
        """Per-layer thicknesses, in the list form the solver accepts."""
        return [layer.thickness for layer in self.layers]

    @property
    def structure(self):
        """Rebuild the stack-wide ``structure`` for the solver.

        Raster layers (array grids) stack into an ``(n_layers, Ny, Nx)`` array through the
        Array-API namespace, preserving backend and dtype; parametric layers return a
        plain per-layer list. An empty stack is ``None`` (matching the ``structure`` default).
        """
        return self._stacked([layer.structure for layer in self.layers])

    @property
    def mu(self):
        """The raster permeability field, or ``None`` if no layer carries one.

        All-or-nothing on purpose: the solve takes one ``(n_layers, Ny, Nx)`` array, so a
        stack where only some layers named a ``mu`` has no array to hand over. That is a
        refusal rather than a silent 1.0 fill, because the fill would be indistinguishable
        from a caller who meant vacuum.
        """
        mus = [layer.mu for layer in self.layers]
        if all(m is None for m in mus):
            return None
        if any(m is None for m in mus):
            missing = [i for i, m in enumerate(mus) if m is None]
            raise ValueError(
                f'layers {missing} carry no mu while others do. The raster permeability is '
                f'one (n_layers, Ny, Nx) field, so it is given for every layer or for none; '
                f'write mu=1.0 grids for the non-magnetic layers to say so explicitly.')
        return self._stacked(mus)

    def _stacked(self, per_layer):
        if not per_layer:
            return None
        if all(_is_raster(e) for e in per_layer):
            xp = array_namespace(*per_layer)
            return xp.stack(per_layer, axis=0)
        return per_layer

    def check(self):
        """Every reason this stack cannot be solved, as messages, or an empty tuple.

        Same contract as :meth:`Incidence.check`: all of them, none raised, and the rules are
        called rather than restated. ``as_material`` and ``_refuse_shaped_half_space`` are the
        solver's own, and ``as_lattice`` is what the cell has always been parsed by.

        A ``Layer`` is checked for the pair a raster has to be -- a permeability field is a
        second array on one grid, so it shares a shape with the permittivity exactly -- which
        is the check ``_revalidate_mu`` performs on the solver, from whichever side lands
        second. Here both sides are in hand at once and there is no order to worry about.

        What it cannot see: the modeling path. Whether a shape recipe is well formed, whether
        its overlaps resolve, whether the shapes fit the cell -- those are read
        by the builders and need the whole scene. A clean check here is not a promise that the
        solve will succeed; it is a promise that these particular mistakes are not present.
        """
        from .engine import _refuse_shaped_half_space
        from .materials import TabulatedMaterial
        problems = []
        for where, value in (('top', self.top), ('bot', self.bot)):
            try:
                if isinstance(value, TabulatedMaterial):
                    value = value.at(value.wavelengths[0])
                _refuse_shaped_half_space(as_material(value, where), where)
            except (ValueError, TypeError) as exc:
                problems.append(str(exc))
        for index, layer in enumerate(self.layers):
            eps, mu = layer.structure, layer.mu
            if mu is None or not hasattr(eps, 'shape') or not hasattr(mu, 'shape'):
                continue
            def grid_shape(a):
                return tuple(a.shape[:-2]) if a.ndim == 4 and a.shape[-2:] == (3, 3) else tuple(a.shape)
            if grid_shape(eps) != grid_shape(mu):
                problems.append(
                    f'layer {index}: mu is {tuple(mu.shape)} and structure is '
                    f'{tuple(eps.shape)}. They are two fields sampled on one raster, so they '
                    f'share a shape exactly.')
        if self.lattice is not None:
            try:
                from .lattice import as_lattice
                as_lattice(lattice=self.lattice)
            except (ValueError, TypeError) as exc:
                problems.append(f'lattice: {exc}')
        elif self.period is not None:
            try:
                pair = tuple(self.period) if hasattr(self.period, '__len__') else (self.period,)
            except TypeError:
                pair = None
            if pair is not None and len(pair) not in (1, 2):
                problems.append(
                    f'period: expected 1 or 2 lengths, got {len(pair)}. It is the '
                    f'axis-aligned spelling of the cell; a skewed one is `lattice=`.')
        return tuple(problems)

    def as_kwargs(self) -> dict:
        """The geometry keyword arguments for :func:`wavebit.rcwa`.

        ``mu`` is omitted rather than passed as ``None`` when no layer names one, so a stack
        of shape layers -- where ``mu`` is refused *by name* because it belongs inside each
        :class:`Material` -- does not trip that refusal just by having been written as a
        ``Stack``.
        """
        mu = self.mu
        # `period` only when there is no basis: the two are one cell in two spellings, and
        # naming both is refused at the solver's door. A `Stack` legitimately holds both --
        # `period` is a derived readout on a skewed cell, and `to_dict` writes it into every
        # document -- so the dropping happens here, on the way out, rather than by making the
        # container lossy. On a skewed cell it was never honoured anyway: the solver assigns
        # `lattice` after `period` and the basis overwrites the slot.
        return dict(structure=self.structure, thickness=self.thickness,
                    top=self.top, bot=self.bot,
                    **({'period': self.period} if self.lattice is None
                       else {'lattice': self.lattice}),
                    **({} if mu is None else {'mu': mu}))

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index):
        return self.layers[index]


@eager_under_torch_compile
@dataclass
class Incidence(_DeclaredFieldsOnly):
    """The incident illumination: wavelength, angles, polarization, and which side it hits.

    The polarization comes in three kinds, and they are not three spellings of one thing.
    ``tetm`` (``'te'`` / ``'tm'``) is a **declaration** that the polarization and the azimuth
    are not degrees of freedom here, which is what buys the pure 1D TE/TM formulation; ``psi``
    is a **linear angle**, and ``chi`` beside it is the ellipticity angle that makes the two
    of them the canonical coordinates of any state; ``polarization`` is the **complex pair**
    below. ``tetm`` refuses to
    coexist with either of the others, and so do those two with each other. These map
    one-to-one onto the constructor arguments.

    ``incidence_side`` is ``'top'`` or ``'bottom'`` — illumination is part of the incidence,
    not of the scene, which is why it lives here rather than on :class:`Stack`. ``theta`` /
    ``phi`` are measured in the incidence medium (``bot``'s normal for ``'bottom'``).
    Real angles are those of ``s*Re(k)``: ``k_par = s*N*sin(theta)``, ``N`` the incidence
    medium's phase index and ``s`` the sign of its index -- ``+1`` except in a
    negative-phase-velocity medium, where the phase runs against the light
    (:meth:`~wavebit._core.engine.CoreSolveMixin._incident_kxy`). In a lossless medium that is
    exactly the direction the incident light travels; in an absorbing one the intensity is
    uniform along the surface and the Poynting vector departs from it by an amount set by the
    loss and the polarization (``docs/conventions.md``). A nonzero imaginary angle
    instead specifies the formal complex-angle wavevector ``n*sin(theta)``, with the same
    signed ``n``.

    Alternatively, give real Cartesian ``kx`` and/or ``ky`` in inverse length
    units (nm**-1 when wavelength is in nm). The omitted component is zero.
    These replace ``theta``/``phi`` and stay fixed during a wavelength sweep.
    They are the incident wavevector, before reciprocal-lattice order offsets.
    At kx=ky=0 the polarization basis uses azimuth zero.

    ``polarization`` is the complex amplitude pair ``(a_s, a_p)`` — the general (elliptical)
    polarization ``psi`` cannot spell. It supersedes ``psi``: giving both is an error, and
    ``psi = v`` is exactly ``polarization = (sin(v), cos(v))``. See ``docs/conventions.md``.
    """
    wavelength: Any = 1.0
    theta: Any = None
    phi: Optional[Any] = None
    tetm: Optional[Any] = None
    psi: Optional[Any] = None
    chi: Optional[Any] = None
    incidence_side: str = 'top'
    polarization: Optional[Any] = None
    kx: Optional[Any] = None
    ky: Optional[Any] = None

    def __post_init__(self):
        if self.theta is None and self.kx is None and self.ky is None:
            self.theta = 0.0

    def check(self):
        """Every reason this illumination cannot be solved, as messages, or an empty tuple.

        All of them at once. The solver refuses the first problem it reaches and stops, which
        is right for a setter and wrong for a caller holding a whole scene: a spec with three
        mistakes takes three round trips to find out. Nothing is raised here, so a caller
        decides whether to raise, print, or collect across a sweep.

        The rules are called, not restated. ``reject_complex_angle`` and
        ``incidence_clashes`` live in :mod:`wavebit._core.conventions` precisely so that this
        and the solver's setters are two callers of one rule rather than two rules -- which
        is what ``truncation_order`` was until `0dde6c5`, and the two had disagreed for years.

        What it cannot see: anything that needs the scene. ``phi`` on a 1D grating, a ``psi``
        against a structure that has no polarization freedom, an order count against a cell --
        those are :class:`Stack` and :class:`Numerics` crossed with this, and a check that
        knows only the illumination cannot reach them.
        """
        problems = []
        if (self.kx is not None or self.ky is not None) and (
                self.theta is not None or self.phi is not None):
            problems.append(_DIRECTION_CLASH)
        for name in ('kx', 'ky'):
            message = transverse_component_problem(getattr(self, name), name)
            if message is not None:
                problems.append(message)
        message = wavelength_problem(self.wavelength)
        if message is not None:
            problems.append(message)
        for name, value, note in (('psi', self.psi, None), ('chi', self.chi, _COMPLEX_CHI)):
            if value is None:
                continue
            try:
                reject_complex_angle(value, note)
            except ValueError as exc:
                problems.append(str(exc))
        problems.extend(incidence_clashes(self.tetm, self.psi, self.chi, self.polarization))
        if self.incidence_side not in ('top', 'bottom'):
            problems.append(
                f"incidence_side: expected 'top' or 'bottom', got {self.incidence_side!r}. "
                f"It names which half-space the light arrives from, and theta and phi are "
                f"measured in that medium.")
        return tuple(problems)

    def as_kwargs(self) -> dict:
        """The illumination keyword arguments for :func:`wavebit.rcwa`."""
        return dict(wavelength=self.wavelength, theta=self.theta, phi=self.phi,
                    tetm=self.tetm, psi=self.psi, chi=self.chi,
                    incidence_side=self.incidence_side,
                    polarization=self.polarization, kx=self.kx, ky=self.ky)
