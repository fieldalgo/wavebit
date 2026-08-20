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
  - block    : list ``[[background, [objects...]], ...]`` -- one entry per layer
  - analytic : list of layer dicts -- one dict per layer

There are four places in there that say what a region is made of, and three of them -- the
block background, the block object's own slot, the analytic shape's ``material`` -- are
**assignments**: a name for the stuff that fills a region. :class:`Material` is that name, and
it carries permittivity and permeability together because a medium has both. The raster is the
fourth and stays arrays: it is not an assignment but a *sampled field*, one number per pixel,
which is exactly why topology optimisation reaches for it -- so it keeps a separate array per
quantity rather than a grid of objects (``structure`` for permittivity, ``mu`` beside it).

``Stack`` therefore holds the ordered layers plus the surrounding media (``top``,
``bot``, ``period``); ``Incidence`` holds the illumination. Numerics and backend choice
(``fto``, ``fourier_coefficients``, ``factorization``, ``backend``, ``type_complex``) stay as
:func:`wavebit.rcwa` arguments -- they are not part of the physical scene.

Backend-agnostic throughout: a raster grid, and equally a :class:`Material`'s ``epsilon``, may
be a NumPy / JAX / PyTorch value; stacking goes through the Array-API namespace and
:class:`Material` is a JAX pytree, so gradients survive both.
"""
import sys
from cmath import sqrt as _complex_sqrt
from dataclasses import dataclass, field
from typing import Any, Optional

from .backend import array_namespace
from .enums import Pol  # noqa: F401  (re-exported for spec's tetm parsing)


def _is_raster(value: Any) -> bool:
    """True if ``value`` is an array grid (raster) rather than a block/analytic recipe."""
    return not isinstance(value, (list, tuple, dict)) and hasattr(value, 'shape')


# --------------------------------------------------------------------------- material


_PYTREE_REGISTERED = False


def _register_pytree_if_jax_loaded():
    """Register :class:`Material` as a JAX pytree, if JAX is in the process already.

    A material is a legitimate *design variable* -- the analytic path already tolerates a
    tracer in a shape's coordinates, and there is no reason permittivity should be the one
    parameter you cannot differentiate. For that to work JAX has to be able to see inside a
    ``Material``: an unregistered class is an opaque **leaf**, so ``jax.grad`` has nothing to
    put a tangent on and ``tree_map`` walks straight past the number it was aimed at. That is
    the same defect class as the aux_data bugs already fixed on this branch, where a value
    that should have been a child was carried as static metadata.

    Registration is lazy for the reason ``on_jax/__init__.py`` exists at all: ``import
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


@dataclass(frozen=True)
class Material:
    """What fills a region: relative permittivity, and the permeability that goes with it.

    ``Material(epsilon, mu=None)``. Both are **relative** to vacuum and both are complex in
    general; ``Im(epsilon) > 0`` is loss in this project's ``e^{-iwt}`` convention, which is
    also what refractive-index tables mean by ``n + 1j*k`` (``docs/conventions.md``).

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

    def __post_init__(self):
        _register_pytree_if_jax_loaded()

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
        return (self.epsilon, self.mu), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def as_material(value, where='material') -> Material:
    """A :class:`Material` from whatever a caller wrote: a Material, a number, or nothing.

    **The single coercion point.** Every path that accepts a material calls this, so the
    scalar shorthand is defined once and cannot drift between the block background, the
    vector object slot and the analytic ``material`` key -- the three spellings that
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
        return value
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


def eps_from_index(n):
    """Relative permittivity of a non-magnetic medium with refractive index ``n``.

    ``n**2``, and nothing else -- no dtype handling, no namespace, so it passes a NumPy value,
    a JAX tracer or a Torch tensor straight through and keeps whatever gradient was attached.

    Public because the flip to permittivity leaves every index-shaped datasheet, every
    ``n + 1j*k`` table and every existing script holding the other quantity. ``Material(
    eps_from_index(3.5))`` is the migration, and it is deliberately more to type than ``3.5``
    was: an index silently reread as a permittivity is a legal, energy-conserving, wrong
    answer, so the conversion is worth seeing at the call site.

    For a magnetic medium the index is ``sqrt(eps*mu)`` and this is not its inverse; that
    case is ``Material(eps, mu)`` written directly.
    """
    return n ** 2


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
    real axis, and IEEE signed zero decides which lip a value sits on: ``eps = -2 + 0j`` gives
    ``+1.4142j`` (an evanescent, decaying medium -- correct) while ``eps = -2 - 0j`` gives
    ``-1.4142j``, a *gain* medium, from an imaginary part that is not merely small but
    identically zero with a sign bit set. The same trap at a smaller scale: a lossless value
    that has been through arithmetic can come out as ``4 - 1e-18j``, and its root
    ``2 - 2.5e-19j`` is nominally active. Neither is repaired here, because a helper cannot
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

    For a magnetic medium the index is ``index_from_eps(eps * mu)``. Not a keyword argument
    here on purpose -- written out, the multiplication is visible at the call site, which is
    where a forgotten ``mu`` would otherwise hide.

    Backend-agnostic: arrays and tracers go through their own namespace's ``sqrt`` (with a
    width-preserving promotion to complex -- ``float32`` in gives ``complex64`` out), and
    Python scalars go through :mod:`cmath`, which agrees with NumPy bit for bit including the
    signed-zero case above.
    """
    if hasattr(eps, 'dtype'):
        xp = array_namespace(eps)
        if not xp.isdtype(eps.dtype, 'complex floating'):
            # Multiplying by a Python complex 1 promotes without naming a width, so a
            # single-precision caller is not silently widened to complex128. Exact: the real
            # part is multiplied by 1.0 and the (zero) imaginary part contributes nothing.
            eps = eps * (1 + 0j)
        return xp.sqrt(eps)
    return _complex_sqrt(complex(eps))


#: The sentence every retired material spelling has to carry. The rename is the *smaller* half
#: of what changed, and a message that only said "use X instead of Y" would send a user to a
#: keyword that accepts their old number and answers a different question with it.
#:
#: The figures are the README quickstart solved twice -- once as written, once with every
#: material literal left at the index it used to be (the layer's 3.5 and the substrate's 1.5
#: read as permittivities). They are deliberately *close*: a migration that broke loudly would
#: not need refusing. 0.2053 against 0.1753 is the size of a truncation-order wobble, which is
#: what a user would take it for. Re-measure them if the quickstart scene ever changes.
_VALUE_CHANGED_TOO = (
    'THE VALUE CHANGED TOO: wavebit now takes relative PERMITTIVITY where it took a '
    'refractive index, so renaming the keyword and keeping the number solves a different '
    'structure. n = 3.5 is eps = 12.25 -- write wavebit.eps_from_index(3.5), or '
    'wavebit.Material(12.25). The same 3.5 read as a permittivity is a perfectly legal, '
    "energy-conserving medium, which is why this is refused by name: on the README's "
    'quickstart scene, keeping the old numbers moves sum(R) from 0.2053 to 0.1753 -- a '
    'plausible number, the size of a truncation wobble, and nothing warns. '
    'wavebit.index_from_eps is the way back.')

#: ``old spelling -> (new spelling, what it names)``. One table, read by the constructors (to
#: refuse a retired keyword argument), by the analytic path's strict-key funnel and by
#: ``_core/spec.py`` (to refuse a retired dict key), so they cannot drift into saying different
#: things about one rename.
#:
#: ``ucell`` and ``eps`` name a *structure* rather than a material, and they are in this table
#: for the same reason the material spellings are: the raster each of them usually held is a
#: grid of numbers whose quantity moved with the rename. A layer written ``eps=[[1, 1, 3.5,
#: 3.5]]`` respelled to ``structure=`` and left at 3.5 is a legal, energy-conserving grating
#: made of a different glass, so the rename alone is not the whole migration and the message
#: must not say it is.
RETIRED_MATERIAL_NAMES = {
    'ucell': ('structure', 'the layer stack: an (n_layers, Ny, Nx) permittivity raster, or a '
                           'per-layer list of shape recipes'),
    'eps': ('structure', "one layer's structure: a (Ny, Nx) permittivity raster, a block "
                         'recipe, or an analytic layer dict'),
    'n_top': ('top', 'the superstrate, as a Material or a bare permittivity'),
    'n_bot': ('bot', 'the substrate, as a Material or a bare permittivity'),
    'n_base': ('background', "an analytic layer's background material"),
    'base': ('background', "an analytic layer's background material"),
    'n_index': ('material', "an analytic shape's material"),
}


def retired_material_name(old, where=''):
    """The refusal text for one retired spelling -- the rename *and* the value change."""
    new, what = RETIRED_MATERIAL_NAMES[old]
    prefix = f'{where}: ' if where else ''
    return (f'{prefix}{old!r} is retired; it is {new!r} now, naming {what}. '
            f'{_VALUE_CHANGED_TOO}')


@dataclass
class Layer:
    """One layer of a :class:`Stack`: a thickness paired with the structure filling it.

    ``structure`` is the layer's entry of the stack's ``structure`` in any of the three
    encodings -- a raster grid array ``(Ny, Nx)`` of permittivity, a block
    ``[background, [objects...]]``, or an analytic layer dict.

    It is a *structure*, not a material, which is why the field is no longer called ``eps``:
    only the raster case is a single quantity end to end, and even there it is one sample per
    pixel rather than one value. The materials live inside the other two encodings, where they
    are :class:`Material` (or a number, via the scalar shorthand).

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


@dataclass
class Stack:
    """An ordered list of :class:`Layer` plus the media that bound it.

    Layer order is top-to-bottom, matching ``structure``'s first axis (the same order the
    solver stacks them). ``top`` / ``bot`` are the bounding superstrate / substrate and
    ``period`` is the in-plane period.

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

    @classmethod
    def from_structure(cls, structure, thickness, top=1.0, bot=1.0, period=(1.0, 1.0),
                       mu=None) -> "Stack":
        """Adapt a ``(structure, thickness)`` pair into a typed ``Stack``.

        ``structure`` may be any encoding (raster ndarray / block list / analytic list); it is
        layer-major on its first axis, so layer *i* is ``(thickness[i], structure[i])``. The
        inverse of :attr:`structure` — ``Stack.from_structure(s, t).structure`` reproduces
        ``s``. ``mu`` is the raster path's optional permeability field, layer-major the same
        way.
        """
        if structure is None:
            layers = []
        else:
            layers = [Layer(t, structure[i], None if mu is None else mu[i])
                      for i, t in enumerate(thickness)]
        return cls(layers=layers, top=top, bot=bot, period=period)

    @property
    def thickness(self) -> list:
        """Per-layer thicknesses, in the list form the solver accepts."""
        return [layer.thickness for layer in self.layers]

    @property
    def structure(self):
        """Rebuild the stack-wide ``structure`` for the solver.

        Raster layers (array grids) stack into an ``(n_layers, Ny, Nx)`` array through the
        Array-API namespace, preserving backend and dtype; block / analytic layers return a
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

    def as_kwargs(self) -> dict:
        """The geometry keyword arguments for :func:`wavebit.rcwa`.

        ``mu`` is omitted rather than passed as ``None`` when no layer names one, so a stack
        of shape layers -- where ``mu`` is refused *by name* because it belongs inside each
        :class:`Material` -- does not trip that refusal just by having been written as a
        ``Stack``.
        """
        mu = self.mu
        return dict(structure=self.structure, thickness=self.thickness,
                    top=self.top, bot=self.bot, period=self.period,
                    **({} if mu is None else {'mu': mu}))

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index):
        return self.layers[index]


@dataclass
class Incidence:
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

    ``polarization`` is the complex amplitude pair ``(a_s, a_p)`` — the general (elliptical)
    polarization ``psi`` cannot spell. It supersedes ``psi``: giving both is an error, and
    ``psi = v`` is exactly ``polarization = (sin(v), cos(v))``. See ``docs/conventions.md``.
    """
    wavelength: Any = 1.0
    theta: Any = 0.0
    phi: Optional[Any] = None
    tetm: Optional[Any] = None
    psi: Optional[Any] = None
    chi: Optional[Any] = None
    incidence_side: str = 'top'
    polarization: Optional[Any] = None

    def as_kwargs(self) -> dict:
        """The illumination keyword arguments for :func:`wavebit.rcwa`."""
        return dict(wavelength=self.wavelength, theta=self.theta, phi=self.phi,
                    tetm=self.tetm, psi=self.psi, chi=self.chi,
                    incidence_side=self.incidence_side,
                    polarization=self.polarization)
