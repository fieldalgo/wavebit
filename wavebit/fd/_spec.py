"""Frozen keyword-only specs, and the one refusal they all share.

Every public constructor in :mod:`wavebit.fd` -- ``Domain``, ``Run``, ``PEC``, ``Scene`` and
the rest -- is a frozen dataclass whose arguments are keyword-only. Two things follow from
making that a decorator rather than a convention:

* **An unknown keyword is refused with the nearest valid name.** A dataclass already raises
  ``TypeError`` for one, but its message names nothing; the refusal here says which of the
  legal names is closest, in the wording ``wavebit._core.parametric`` uses for a shape key.
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
    of exactly 0.0 and a ``forward_ad`` dual no tangent at all, on either discretisation,
    where the central difference is 1.97e-7 (``docs/fd-symmetric-stencil.md``, third
    review). ``torch.func.jacfwd`` and a dual of a ``torch.Tensor`` subclass dropped theirs
    the same way. A ``vmap`` batch never passed silently -- ``float()`` of it raises -- but it
    was refused as "a real number is needed", which names no rule.

    Each ``torch.func`` wrapper layer is asked, from the outside in, the three questions that
    mean something rides on it: does it require grad, does it carry a forward tangent, is it
    a vmap batch. A layer that answers no is unwrapped and the one inside it asked, so a
    wrapper that carries nothing is admitted: ``torch.func.grad(f, argnums=0)(eps, w)`` wraps
    ``w`` too, with nothing on it, and ``jax.grad`` admits the same program. ``d46935a``
    refused every wrapper, which made the two libraries disagree there.

    One layer cannot be seen through. From inside a ``torch.func.grad`` layer, a forward
    tangent on the tensor it wraps reads as absent -- measured: a ``forward_ad`` dual, or a
    ``torch.func.jvp`` primal, handed to ``grad`` as an argument it does not differentiate,
    unwraps to a tensor whose ``unpack_dual`` tangent is ``None``, while ``grad``'s output
    still carries the outer tangent, so ``float()`` there would drop it silently. So while a
    forward-mode level is open, a ``grad`` layer that answers no is refused rather than
    unwrapped. That refuses one wrapper that carries nothing too (a ``grad``-wrapped constant
    inside ``jvp``); it is loud, and a Python number is the spelling that passes.

    The 27 cases this was chosen against are
    ``tests/test_fd_stage1_grad.py::test_torch_carried_answers_every_transform``: ``b24a708``
    was wrong on 12 of them (9 silently: ``float()`` read the primal), ``d46935a`` on 5 (1
    silently: the subclass dual), this on none. Two need the walk -- an outer ``grad``'s
    leaf handed to an inner ``grad`` or ``jacfwd`` as an argument it does not differentiate
    is wrapped again with nothing on the new layer.

    ``unpack_dual`` is public, and so is its ``_current_level`` (the open level); everything
    else is ``torch._C._functorch``'s, the module ``_shapes`` already reads ``get_unwrapped``
    from. None is wrapped in a ``try``: if a Torch release renames one, a static door handed
    a Torch tensor raises rather than opening. A value that is not a ``torch.Tensor`` is asked
    nothing, and Torch is not imported to say so -- any subclass, defined anywhere, is one.
    """
    torch = sys.modules.get('torch')
    if torch is None or not isinstance(v, torch.Tensor):
        return False
    from torch.autograd import forward_ad
    functorch = torch._C._functorch
    forward = forward_ad._current_level >= 0
    grads = {layer.level() for layer in functorch.get_interpreter_stack() or ()
             if layer.key() == functorch.TransformType.Grad}
    while True:
        if (v.requires_grad or forward_ad.unpack_dual(v).tangent is not None
                or functorch.is_batchedtensor(v)):
            return True
        if not functorch.is_functorch_wrapped_tensor(v):
            return False
        if forward and functorch.maybe_get_level(v) in grads:
            return True
        v = functorch.get_unwrapped(v)


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


def spec(cls=None, *, positional=()):
    """Make ``cls`` a frozen, keyword-only dataclass that refuses unknown keywords by name.

    ``positional`` names the leading fields that may also be given positionally -- the one
    use is ``Record('probe', ...)``, where the name is what a reader looks for first.
    """
    if cls is None:
        return lambda c: spec(c, positional=positional)
    cls = dataclasses.dataclass(frozen=True, kw_only=True)(cls)
    generated = cls.__init__
    names = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    leading = tuple(positional)

    def __init__(self, *args, **kwargs):
        if len(args) > len(leading):
            raise TypeError(f'{cls.__name__} takes keyword arguments only: {list(names)}. '
                            f'Positional arguments have no natural order here.')
        for name, value in zip(leading, args):
            if name in kwargs:
                raise TypeError(f'{cls.__name__}: {name!r} given twice, positionally and by keyword.')
            kwargs[name] = value
        unknown = [k for k in kwargs if k not in names]
        if unknown:
            raise TypeError(unknown_keyword_message(cls.__name__, unknown, names))
        generated(self, **kwargs)

    __init__.__qualname__ = generated.__qualname__
    __init__.__doc__ = generated.__doc__
    cls.__init__ = __init__

    def replace(self, **changes):
        """A new spec with ``changes`` applied; every validation runs again."""
        return dataclasses.replace(self, **changes)

    cls.replace = replace
    return cls
