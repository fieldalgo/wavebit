"""Frozen keyword-only specs, and the one refusal they all share.

Every public constructor in :mod:`wavebit.fdtd._legacy` -- ``Domain``, ``Run``, ``PEC``, ``Scene`` and
the rest -- is a frozen dataclass whose arguments are keyword-only. Two things follow from
making that a decorator rather than a convention:

* **An unknown keyword is refused with the nearest valid name.** A dataclass already raises
  ``TypeError`` for one, but its message names nothing; the refusal here says which of the
  legal names is closest, in the wording ``wavebit.rcwa._engine.parametric`` uses for a shape key.
  The reason it is refused rather than dropped is the same everywhere in this repository: a
  keyword nobody reads is silently ignored, so the run answers the scene you had rather
  than the one you asked for.
* **``replace(**changes)`` is on every spec**, so ``scene.replace(run=fd.Run(steps=800))`` is
  the way to vary one thing and every validation runs again on the new object (a frozen
  spec cannot be edited into an invalid state; it can only be rebuilt, and rebuilding checks).

Positional arguments are refused outright: ``fd.Domain((1, 1, 1))`` would read as a size to
one reader and a centre to another, and the constructors here have no natural order.

**A static value is concretised here, and a traced one refused by name.** ``Domain.size``,
``Resolution.dx``, ``Run.courant``, ``Record.at`` fix the cell counts, the scan length and
the skeleton, so they cannot be leaves of a gradient; the leaves are listed once, in
:data:`TRACED_LEAVES`. A bare ``float()`` would not say so: on JAX it raises
a ``ConcretizationTypeError`` that names the ``float`` function and not the field, and on
Torch it *succeeds* on a tensor with ``requires_grad`` -- the value is silently detached,
the run completes, and the gradient is simply absent. ``static_float`` and ``static_int``
refuse both cases with the field's name and the rule -- and Torch's forward mode and
``torch.func`` wrappers too (:func:`torch_carried`), which set no ``requires_grad`` and
whose primal ``float()`` reads just as silently.
"""
import dataclasses
import difflib
import inspect
import operator
import sys

__all__ = ['spec', 'unknown_keyword_message', 'static_float', 'static_int', 'is_jax_typed',
           'torch_carried']


def is_jax_typed(v):
    """Whether ``v`` is a JAX value (an array or a tracer), without importing JAX."""
    return type(v).__module__.split('.')[0] in ('jax', 'jaxlib') and hasattr(v, 'aval')


def torch_carried(v):
    """Whether ``v`` is a Torch tensor with a derivative or a batch riding on it.

    ``float()`` of such a value reads its primal and drops what rides on it. Reverse mode says
    so through ``requires_grad``, which the static doors always asked; forward mode and
    ``torch.func`` do not. Measured through public ``fd.run(scene, 'torch')`` on Torch 2.13
    (2026-09-25), ``fd.Sphere(radius=)`` on ``b24a708``: ``torch.func.jvp`` gave a derivative
    of exactly 0.0 and a ``forward_ad`` dual no tangent at all, where the central differences
    are 1.97e-7 (stencil) and 1.84e-7 (voxel) (``docs/design/fd-symmetric-stencil.md``, third
    review).

    **First, can a tangent be seen here at all?** Not while a ``torch.func`` interpreter that
    runs on autograd -- Grad or Jvp -- sits above the lowest open forward-mode level. There,
    ``forward_ad.unpack_dual`` reads every forward tangent beneath it as absent, on a wrapper
    and on a plain dual alike, while the derivative still flows. Measured on Torch 2.13:
    ``jvp`` over ``grad`` of ``x * a``, ``a`` captured by closure, is 1.0 without a door and
    0.0 when a door reads ``float(a)``; ``jvp`` over ``jvp`` the same (a Grad layer was found
    first, in the fifth round; a second Jvp layer in the sixth). So in that state every Torch
    tensor is refused, whatever it carries: a forward level with no ``jvp`` interpreter is a
    user's ``dual_level`` (or ``linearize``), beneath every interpreter, so any Grad layer
    hides it; with ``jvp`` interpreters, a Grad or a second Jvp layer above the lowest of
    them hides it. Inference mode under an open forward level is refused the same way --
    ``unpack_dual`` raises there. That refuses some values that carry nothing -- under
    ``torch.func.hessian``, forward over forward, or a constant captured into a ``grad``
    inside a ``jvp`` -- which ``jax`` admits. It is loud, and a Python number is the
    spelling that passes. (Functorch's private ``temporarily_pop_interpreter_stack``
    recovers the hidden tangents, a review measured; a check that edits the interpreter
    stack it is asking about is not one this door will be.)

    **Otherwise, walk the layers.** Each ``torch.func`` wrapper, from the outside in, is asked
    the two questions that dispatch nothing -- does it require grad, is it a ``vmap`` batch --
    and a layer that answers no is unwrapped and the one inside asked. Only when every layer
    has answered no is each asked for a forward tangent: ``unpack_dual`` dispatches, and on a
    layer over a batch (a ``functionalize`` wrapper over a ``vmap`` one, under an open forward
    level) it raises Torch's own error instead of letting the door refuse by name. So a
    wrapper that carries nothing is admitted -- ``torch.func.grad(f, argnums=0)(eps, w)``
    wraps ``w`` too, and ``jax.grad`` admits the same program -- and an outer ``grad``'s leaf
    handed to an inner ``grad`` as an argument it does not differentiate, wrapped again with
    nothing on the new layer, is not.

    **A live wrapper from a stack this thread cannot see.** The interpreter stack is per
    thread, and functorch lowers it to nothing inside an ``autograd.Function``'s ``forward``.
    A live ``jvp`` primal read there -- in a worker thread of a pool, or captured into a
    custom Function -- sits at a level no interpreter here owns, and its tangent reads as
    absent; measured, a spec built in a worker from such a primal kept a derivative of 0.5
    where the identity hand-back keeps 1.0, while the same build in the calling thread is
    refused (and jax refuses it in the worker too). So a live, non-functional wrapper whose
    level is not a Grad or Jvp level on this thread's stack is refused. A dead wrapper -- one
    that escaped its transform -- is unwrapped as before.

    The forward level itself is one global for the process, not per thread: while another
    thread holds a ``dual_level`` or is inside a ``jvp``, this thread's ``grad`` or inference
    mode counts as blind and refuses a constant too. Loud, and a Python number passes.

    ``tests/fdtd/test_fd_stage1_grad.py::test_torch_carried_answers_every_transform`` asks a real
    door about 58 ways of making a Torch value; the designs before this one each dropped a
    derivative silently on some of them (the counts are in its docstring), this on none.

    ``unpack_dual`` is public. ``forward_ad._current_level`` is private -- the module global
    that ``dual_level`` keeps -- and so is everything read from ``torch._C._functorch``, the
    module ``_shapes`` already reads ``get_unwrapped`` from. None is wrapped in a ``try``:
    a rename makes a static door handed a Torch tensor raise rather than open. A change of
    meaning without a rename would not raise; the table's captured and ``argnums`` rows are
    what would notice. A value that is not a ``torch.Tensor`` is asked nothing, and Torch is
    not imported to say so -- any subclass, defined anywhere, is one.
    """
    torch = sys.modules.get('torch')
    if torch is None or not isinstance(v, torch.Tensor):
        return False
    from torch.autograd import forward_ad
    functorch = torch._C._functorch
    stack = functorch.get_interpreter_stack() or ()
    kinds = (functorch.TransformType.Grad, functorch.TransformType.Jvp)
    if forward_ad._current_level >= 0:
        if torch.is_inference_mode_enabled():
            return True
        autograd = [layer.level() for layer in stack if layer.key() in kinds]
        jvps = [layer.level() for layer in stack if layer.key() == functorch.TransformType.Jvp]
        lowest = min(jvps) if jvps else -1
        if any(level > lowest for level in autograd):
            return True
    here = {layer.level() for layer in stack if layer.key() in kinds}
    layers = []
    while True:
        if v.requires_grad or functorch.is_batchedtensor(v):
            return True
        layers.append(v)
        if not functorch.is_functorch_wrapped_tensor(v):
            break
        if (not torch._is_functional_tensor(v) and not functorch.is_dead_tensor_wrapper(v)
                and functorch.maybe_get_level(v) not in here):
            return True
        v = functorch.get_unwrapped(v)
    return any(forward_ad.unpack_dual(layer).tangent is not None for layer in layers)


#: What *is* a leaf of a gradient, in the order the stages added it. One tuple, read by every
#: static-door refusal below, so a stage that adds a leaf adds it here and every door names it
#: (the stage-0 sentence was a fixed string and went stale twice: stage 1's material values
#: and stage 2's source leaves were missing from it).
TRACED_LEAVES = (
    'InitialField.amplitude and the initial fields',                        # stage 0
    'the Bloch phase or k',                                                 # stage 0
    'material values (a permittivity, a conductivity, a pole strength)',     # stage 1
    "each source's amplitude, phase and delay",                             # stage 2
    'its Band(wavelength=..., df=...)',                                     # stage 2
    "a Dipole's center, within the range= it declares",                     # stage 2
    'an Axis1D k, within the k_max= it declares',                           # stage 3
    'an isolated Sphere radius and center within range=, with a concrete plan',  # C6
    'an isolated Cylinder radius, height and center within range=, with a concrete plan',
    'an isolated Box size and center within range=, with a concrete plan and static orientation',
    'vertices, height and center of an isolated Prism within range=, with a concrete plan, '
    'static vertex count and static orientation',
    'multiple ranged bodies together when their full interpolation halos are disjoint',
)


def _refuse_traced(where, name):
    return ValueError(
        f'{where}({name}=...) is static: it fixes the grid, the scan and the skeleton, so a '
        f'traced value is refused -- it is not a leaf. The traced leaves are '
        f'{"; ".join(TRACED_LEAVES)}.')


def static_float(where, name, v):
    """``v`` as a Python float, or a refusal naming ``where(name=...)``.

    A Torch tensor with ``requires_grad``, or one a forward-mode tangent or a ``vmap`` batch
    rides on (:func:`torch_carried`), is refused before ``float()`` reads it (that read
    succeeds, and silently drops the graph or the tangent); a JAX tracer is refused when
    ``float()`` fails on a JAX value. Anything else ``float()`` cannot read is refused as not a
    number.
    """
    if getattr(v, 'requires_grad', False) or torch_carried(v):
        raise _refuse_traced(where, name)
    if isinstance(v, (bool, complex, str)):
        raise ValueError(f'{where}({name}={v!r}): a real number is needed, not a '
                         f'{type(v).__name__}.')
    try:
        return float(v)
    except Exception as err:
        if is_jax_typed(v):
            raise _refuse_traced(where, name) from err
        raise ValueError(f'{where}({name}={v!r}): a real number is needed.') from err


def static_int(where, name, v):
    """``v`` as a Python int through ``__index__`` (so ``numpy`` integers count), or a refusal.

    A bool is refused: ``True`` indexes as 1 but nobody means a step count or a parity by
    it. A traced value is refused by name as ``static_float`` does.
    """
    if isinstance(v, bool):
        raise ValueError(f'{where}({name}={v!r}): an integer is needed, not a bool.')
    if getattr(v, 'requires_grad', False) or torch_carried(v):
        raise _refuse_traced(where, name)
    try:
        return operator.index(v)
    except Exception as err:
        if is_jax_typed(v):
            raise _refuse_traced(where, name) from err
        raise ValueError(f'{where}({name}={v!r}): an integer is needed.') from err


def unknown_keyword_message(where, unknown, known):
    """The refusal text for keywords ``unknown`` at ``where``; ``known`` are the legal names."""
    bits = []
    for name in unknown:
        near = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
        bits.append(f'{name!r}' + (f' (did you mean {near[0]!r}?)' if near else ''))
    return (f'{where} does not take {", ".join(bits)}. It takes {list(known)}. A keyword nobody '
            f'reads is silently ignored, which is why this is refused rather than dropped.')


def renamed_keyword_message(where, gone, renamed):
    """The refusal text for removed keywords ``gone`` at ``where``.

    ``renamed`` maps an old keyword to the new one, or to ``(new, note)`` when the value is not
    carried over as it is (a width that became a spectral sigma, steps that became seconds): the
    note says how to convert it.
    """
    bits, notes = [], []
    for old in gone:
        new, note = renamed[old] if isinstance(renamed[old], tuple) else (renamed[old], None)
        bits.append(f'{where}({old}=...) is now {where}({new}=...)')
        if note and note not in notes:
            notes.append(note)
    text = ', '.join(bits) + ''.join(f'; {note}' for note in notes)
    return (f'{text}. The old keyword was removed, not aliased; see the public-name migration '
            f'guide, docs/migration/public-names.md.')


class Answered:
    """The ``obj`` of a tombstone's AttributeError: the message already names the replacement,
    so Python's "Did you mean" (computed from ``dir(obj)``) has nothing to add."""

    def __dir__(self):
        return []


def removed(owner, name, replacement, note=None):
    """The AttributeError for ``owner.name``, a public name that ``replacement`` replaced.

    A tombstone, never an alias: reading the old name raises and names the new one. ``note``
    says how to convert a value that is not carried over as it is.
    """
    text = f'{owner}.{name} is now {replacement}'
    return AttributeError(f'{text}{"; " + note if note else ""}. The old name was removed, not '
                          f'aliased; see the public-name migration guide, '
                          f'docs/migration/public-names.md.', name=name, obj=Answered())


def _signature(cls, leading):
    """The signature ``inspect`` and the API reference show for a spec's ``__init__``.

    The wrapper itself takes ``*args, **kwargs``; this says what it accepts: the ``leading``
    fields positionally or by keyword, every other field by keyword only, with its default.
    """
    def default(f):
        if f.default is not dataclasses.MISSING:
            return f.default
        if f.default_factory is not dataclasses.MISSING:
            return f.default_factory()
        return inspect.Parameter.empty

    fields_ = {f.name: f for f in dataclasses.fields(cls) if f.init}
    params = [inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    params += [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default(fields_[n]))
               for n in leading if n in fields_]
    params += [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=default(f))
               for n, f in fields_.items() if n not in leading]
    return inspect.Signature(params)


def spec(cls=None, *, positional=(), renamed=None):
    """Make ``cls`` a frozen, keyword-only dataclass that refuses unknown keywords by name.

    ``positional`` names the leading fields that may also be given positionally -- the one
    use is a declaration's ``name``, as in ``TimeSeriesMonitor('probe', ...)``, where the name
    is what a reader looks for first. ``renamed`` maps a removed keyword to the one that
    replaced it, or to ``(new, note)`` when the value must be converted: the old spelling is
    refused with the new one named (a tombstone, never an alias). Messages name ``type(self)``,
    so a public subclass of an engine declaration is refused under its own name.
    """
    if cls is None:
        return lambda c: spec(c, positional=positional, renamed=renamed)
    cls = dataclasses.dataclass(frozen=True, kw_only=True)(cls)
    generated = cls.__init__
    names = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    required = tuple(f.name for f in dataclasses.fields(cls) if f.init
                     and f.default is dataclasses.MISSING
                     and f.default_factory is dataclasses.MISSING)
    leading = tuple(positional)
    removed = dict(renamed or {})

    def __init__(self, *args, **kwargs):
        where = type(self).__name__
        if len(args) > len(leading):
            if leading:
                raise TypeError(f'{where} takes {", ".join(leading)} positionally and everything '
                                f'else by keyword: {list(names)}.')
            raise TypeError(f'{where} takes keyword arguments only: {list(names)}. '
                            f'Positional arguments have no natural order here.')
        for name, value in zip(leading, args):
            if name in kwargs:
                raise TypeError(f'{where}: {name!r} given twice, positionally and by keyword.')
            kwargs[name] = value
        unknown = [k for k in kwargs if k not in names]
        if unknown:
            gone = [k for k in unknown if k in removed]
            if gone:
                raise TypeError(renamed_keyword_message(where, gone, removed))
            raise TypeError(unknown_keyword_message(where, unknown, names))
        missing = [k for k in required if k not in kwargs]
        if missing:
            # The generated __init__ would name the class it was generated for -- an engine
            # declaration's, for a public subclass -- so the refusal is made here instead.
            raise TypeError(f'{where} needs {" and ".join(map(repr, missing))}. It takes '
                            f'{list(names)}.')
        generated(self, **kwargs)

    __init__.__qualname__ = generated.__qualname__
    __init__.__doc__ = generated.__doc__
    __init__.__signature__ = _signature(cls, leading)
    cls.__init__ = __init__

    def replace(self, **changes):
        """A new spec with ``changes`` applied; every validation runs again."""
        return dataclasses.replace(self, **changes)

    cls.replace = replace
    return eager_under_torch_compile(cls)


from dataclasses import fields
from functools import cache
from wavebit._common.backend import eager_under_torch_compile

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
    ``JobSpec`` in ``src/wavebit/_common/_jobs_v1.py``) were the objects a spec is built from and kept the hole.
    ``Material`` and ``TabulatedMaterial`` are frozen and refuse every assignment already.

    The rule is the solver's, so all three doors predict alike: a declared field, a leading
    underscore, or a name already on the class (which keeps the descriptor protocol in charge
    -- ``stack.thickness = ...`` gets Python's own "has no setter", not "did you mean", and a
    method stays stubbable on an instance). The refusal is the solver's too,
    :func:`wavebit._common.validation._unknown_attribute`, with its near-match hint drawn from the
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
        from wavebit._common.validation import _unknown_attribute     # lazily: engine imports this module
        raise _unknown_attribute(cls.__name__, name, _declared_fields(cls), kind='fields')



def _merge_domain(kwargs, domain_kwargs, source, solver='rcwa'):
    """Fold a domain object's keyword arguments into ``kwargs``, rejecting duplicates.

    Passing e.g. both ``stack=`` and ``structure=`` is ambiguous, so it is an explicit error
    rather than a silent override.

    ``solver`` names the door in the message. It is a parameter rather than a constant because
    :func:`fdtd` is the second caller, and a duplicate keyword on the time-domain entry point
    was being reported as a mistake made to ``rcwa``.
    """
    clash = set(kwargs) & set(domain_kwargs)
    if clash:
        raise ValueError(
            f"{source} already sets {sorted(clash)}; do not also pass them to {solver}.")
    merged = dict(kwargs)
    merged.update(domain_kwargs)
    return merged



@eager_under_torch_compile
class Declaration:
    """Replace protocol for structurally immutable public dataclasses.

    Use ``@dataclass(frozen=True, kw_only=True)`` on a subclass. Numerical arrays
    are borrowed leaves; freezing the declaration does not freeze caller buffers.
    """
    def replace(self, **changes):
        return dataclasses.replace(self, **changes)
