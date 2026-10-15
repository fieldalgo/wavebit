"""Constitutive values and the constant material model, independent of either solver."""
import sys
from cmath import isfinite as _complex_isfinite, sqrt as _complex_sqrt
from dataclasses import dataclass, field, replace
from typing import Any
from wavebit._common.backend import _concrete, array_namespace, principal_sqrt, eager_under_torch_compile

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

    Registration is lazy for the reason ``src/wavebit/rcwa/_backends/jax_backend.py`` lifts x64 on import: ``import
    wavebit`` must not import JAX on behalf of a caller who will never use it, and importing
    JAX has real side effects here (that module flips ``jax_enable_x64`` process-wide). So
    this reads ``sys.modules`` and never imports -- the same "act only if JAX is already
    there" pattern as
    :func:`wavebit._enable_jax_x64_if_already_imported`. :class:`Material` cannot use the
    other pattern in the codebase, the one :mod:`wavebit.observations.diffraction` uses, because that one
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
    from wavebit._common.pytrees import _STATIC_FIELDS
    _STATIC_FIELDS[Material] = ('_wavelength_shape',)
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
    also what refractive-index tables mean by ``n + 1j*k`` (``docs/reference/common/conventions.md``).

    Each quantity can be scalar, three diagonal entries ``[xx, yy, zz]``, or a
    physical 3x3 tensor in the public xyz frame. Assign tensors to a layer
    background or shape and call ``solve()``; internal convolution signs are
    handled automatically. ``rotated(angle, axis)`` rotates material axes.
    Uniform full tensors, in-plane tensor patterns and patterned full tensors
    (a filled z row, since 2026-09-24) are supported; anisotropic half-spaces
    are not. ``wavebit.manifest.MATERIALS`` states each case and its limits.

    **Why permittivity rather than index.** Everything past the front door already works in
    permittivity -- the Fourier factorizations, the tensor builders and ``solve_conv`` all
    take it -- and the front door used to square an index on the way in. One quantity had to
    win, and an index cannot carry the pair: ``mu`` was always taken raw, so a single material
    argument spelled as an index would have meant "index" for one of its two numbers and
    "permeability" for the other. It is also the direction with no branch cut in it. ``eps =
    n**2`` is unambiguous where ``n = sqrt(eps)`` is a choice (:func:`refractive_index_from_eps`),
    and the half-space square root is on this project's short list of recurring traps, so the
    boundary is a bad place to have to make that choice repeatedly.

    The flip is not free and is not silent-safe: the same literal
    ``3.5`` read as an index gives ``R = 0.3086`` and read as a permittivity ``R = 0.0920``,
    both energy-conserving, which is why the retired index spellings (``ucell``, ``n_top=``,
    ``n_bot=``) are refused -- as unknown keywords, since their by-name messages were deleted
    on 2026-08-26 -- rather than quietly accepted, and why
    ``tests/materials/test_material_convention_anchor.py`` pins the convention
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
    :func:`eps_from_refractive_index` and :func:`refractive_index_from_eps`.
    """

    epsilon: Any
    mu: Any = None
    _wavelength_shape: Any = field(default=None, repr=False, compare=False, kw_only=True)

    def __post_init__(self):
        _register_pytree_if_jax_loaded()

    def replace(self, **changes):
        return replace(self, **changes)

    def at(self, wavelength, *, units=None):
        """Return this constant material unchanged at any wavelength.

        For wavelength-dependent values, use :class:`wavebit.TabulatedMaterial`
        or ``wavebit.materials.catalog.load(name)``.
        """
        if units is None:
            return self  # Explicit legacy calls use already normalized numerical values.
        from wavebit.units import require_units
        require_units(units)
        from .models import constant_response
        return constant_response(self.epsilon, self.mu, wavelength, units=units)

    @property
    def is_anisotropic(self):
        """Whether epsilon or mu is supplied as three principal values or a 3x3 tensor.

        This selects tensor arithmetic structurally, including an isotropic diagonal
        tensor. Scalars retain the ordinary scalar material path.
        """
        from wavebit.materials.tensors import material_is_tensor
        return material_is_tensor(self)

    def rotated(self, angle, axis='z'):
        """Return a material actively rotated about x, y or z by radians.

        Tensor components are in Wavebit's xyz axes. Both epsilon and mu rotate
        as ``R @ tensor @ R.T``; scalars are unchanged. Array backend and material
        and angle derivatives are retained. This rotates the material axes, not
        the shape to which the material is assigned.
        """
        from wavebit.materials.tensors import rotated_material
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
        convolution build (:meth:`wavebit.rcwa._engine.engine.CoreFactorizationMixin._mu_structure`),
        which runs the permittivity's own builder over a structure whose material slots hold
        permeabilities: in a scene something else made magnetic, a region that named none is
        still a region, and ``1.0`` is what it contributes. A ``None`` there would reach the
        arithmetic and raise.

        It is deliberately *not* what :attr:`is_magnetic` reads, and the two must not be
        collapsed into one accessor: the whole point of the ``None`` spelling is that the
        question ("was one named?") stays answerable inside a trace while the value need not
        be. Consumers that only ask the question -- the routing in ``solve_conv``, the
        ``sqrt(eps*mu)`` of :func:`wavebit.rcwa._engine.engine._half_space_index` -- read
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
    (``'stack.top'``, ``'shapes[2].material'``), matching how ``src/wavebit/_common/_jobs_v1.py`` reports paths.

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
    if isinstance(value, MaterialResponse):
        value = Material(value.epsilon, value.mu, _wavelength_shape=value._wavelength_shape)
    if isinstance(value, Material):
        if value._wavelength_shape is not None:
            raise ValueError(f'{where}: a material evaluated at an array of wavelengths is a spectrum; '
                             'evaluate at one wavelength or assign the material model')
        return value
    from wavebit.materials.tabulated import TabulatedMaterial
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
def eps_from_refractive_index(n, *, mu=None):
    """Convert isotropic refractive index to relative permittivity.

    Return ``n**2`` when ``mu`` is omitted (relative permeability one), or
    ``n**2 / mu`` when it is supplied. Both quantities may be complex and must
    describe the same wavelength. Scalars and elementwise array batches are
    supported; this is not a conversion from an anisotropic index tensor.
    NumPy/JAX/PyTorch values retain their backend and differentiation graph.

    Supplied permeability must be finite and nonzero. Invalid values raise
    ``ValueError`` eagerly; traced JAX arrays produce NaN at invalid entries.
    Assign the same permeability to the resulting material, for example
    ``Material(eps_from_refractive_index(n, mu=mu), mu=mu)``.
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
def refractive_index_from_eps(eps):
    """Refractive index of a medium with relative permittivity ``eps`` -- the passive root.

    The square root of a complex number has two values and only one of them is a physical
    index here. This takes the **principal** root, and that is not a choice made by this
    function so much as one verified against the convention: a passive medium has
    ``Im(eps) >= 0`` (loss, in ``e^{-iwt}``), which puts ``eps`` in the closed upper half
    plane, whose principal root lies in the first quadrant -- so ``Im(n) >= 0``, loss again.
    The same argument is what makes :func:`wavebit._common.conventions.boundary_kz` principal
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
    :func:`~wavebit._common.conventions.signed_index`, ``sqrt(eps)*sqrt(mu)``: the principal root
    of ``eps*mu`` for an ordinary medium and its negation for a negative-phase-velocity one,
    every passive double negative and passive magnetic metal among them
    (``docs/theory/rcwa/negative-index-media.md`` section 6). This takes no ``mu`` on purpose: the sign
    needs both factors, and their product has lost it.

    Backend-agnostic: arrays and tracers go through their own namespace's ``sqrt`` (with a
    width-preserving promotion to complex -- ``float32`` in gives ``complex64`` out), and
    Python scalars go through :mod:`cmath`. All of them take the lip a signed zero names --
    JAX's ``sqrt`` takes the upper one for both zeros, and
    :func:`~wavebit._common.backend.principal_sqrt` turns it there -- so a value on the cut gets
    one root on every backend. Elsewhere they are each their library's ``sqrt``, which need not
    agree to the last bit: :mod:`cmath` and NumPy differ by one or two ulp in a component on 45
    of 2,000 random products ``eps*mu`` (``studies/rcwa/negative_index_media/negative_index_media.py``, section 13).
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



@eager_under_torch_compile
@dataclass(frozen=True)
class MaterialResponse:
    """Evaluated relative constitutive values; not a broadband model."""
    epsilon: Any
    mu: Any = None
    _wavelength_shape: Any = field(default=None, repr=False, compare=False, kw_only=True)

    def __post_init__(self):
        from .models import _register
        _register(type(self), ('epsilon', 'mu'), ('_wavelength_shape',))

    def replace(self, **changes):
        return replace(self, **changes)

    @property
    def is_magnetic(self):
        return self.mu is not None

    @property
    def mu_value(self):
        return 1.0 if self.mu is None else self.mu
