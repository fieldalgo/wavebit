"""The face types, and the six-face declaration every scene must make.

**Boundaries are required per face.** There is no default: a box with an unnamed face is a
box whose physics nobody chose, and the run would answer a question the user did not ask.
``Boundaries(x=..., y=..., z=...)`` takes, per axis, one face type for both faces or a
``(low, high)`` pair. Faces are named ``'x-'``, ``'x+'``, ``'y-'``, ``'y+'``, ``'z-'``,
``'z+'`` and never ordered.

Stage 0 knows five face types. What each one *is* on the Yee grid -- the neighbour rule and
the mask it lowers to, and where the wall physically sits -- is decided in
``wavebit/fd/_kernel/faces.py``; this module only says what a user may declare:

``Periodic()``
    Both faces of the axis wrap. Must be given to both faces (one object for the axis).
``Bloch(phase=phi)`` or ``Bloch(k=k)``
    Both faces wrap with the phase ``phi`` radians across the period: the field at ``x + L``
    is ``exp(+i*phi)`` times the field at ``x``, in the ``e^{-i w t}`` time convention this
    project uses everywhere (a wave travelling toward ``+x`` is ``exp(+i k x)``). ``k`` is the
    wavevector component along the axis, so ``phi = k * L`` with ``L`` the snapped box size.
    A Bloch face makes the whole run complex, doubling every byte and every flop, so a phase
    that is concretely ``0`` is refused in favour of ``Periodic()`` rather than lowered to it
    silently -- a user who wrote ``Bloch`` and got a real run would rightly ask why.
``PEC()``
    Perfect electric conductor: tangential E vanishes on the wall.
``PMC()``
    Perfect magnetic conductor: tangential H vanishes on the wall.
``Symmetry(parity=+1)`` / ``Symmetry(parity=-1)``
    A mirror plane on that face; the box is one half of a symmetric problem. ``parity`` is
    the eigenvalue of the field under the mirror, stated for the **tangential E**: ``+1``
    means tangential E is even across the plane, which makes the plane PMC-like (tangential
    H odd, hence zero, on it); ``-1`` means tangential E is odd, PEC-like. Allowed on at most
    one face of an axis, with ``PEC`` or ``PMC`` on the other. Where the plane sits relative
    to the face is fixed by the Yee layout (a node plane on a low face, a half-cell plane on
    a high face) and recorded per face by the plan; ``faces.py`` derives it and the
    symmetry gate is the proof.

Presets: ``Boundaries.periodic(z=...)`` for a periodic cell with the given z faces, and
``Boundaries.open(...)``, which is stage 1's absorber and raises ``NotImplementedError``
naming stage 1 -- a named refusal, not a silent fallback.
"""
import numbers

from ._spec import is_jax_typed, spec

__all__ = ['Boundaries', 'Periodic', 'Bloch', 'PEC', 'PMC', 'Symmetry', 'FACES', 'AXES']

#: The three axes in the order the public API spells a triple, and the six face names.
AXES = ('x', 'y', 'z')
FACES = ('x-', 'x+', 'y-', 'y+', 'z-', 'z+')


class FaceType:
    """Base of the five face types; carries nothing, exists so ``isinstance`` has a name."""

    __slots__ = ()


@spec
class Periodic(FaceType):
    """Both faces of the axis wrap; real dtype."""


def _concrete_scalar(what, v):
    """``v`` as a float when its value is readable now; ``None`` for a JAX tracer.

    The phase is a traced leaf, so a tracer must pass; but a tracer is the *only* thing that
    passes without being read. Everything else ``float()`` cannot read -- a string, a list,
    a complex number, an array of two phases -- is refused here by name, not left for the
    lowering to trip over with a NumPy message that names neither ``Bloch`` nor ``phase``.
    """
    if isinstance(v, (bool, complex, str)) or v is None:
        raise ValueError(f'Bloch({what}={v!r}): a real scalar in radians (or a traced scalar '
                         f'of jax or torch); got {type(v).__name__}.')
    shape = getattr(v, 'shape', None)
    if shape is not None and tuple(shape) != ():
        raise ValueError(f'Bloch({what}=...): one real scalar per axis; got shape '
                         f'{tuple(shape)}. A phase per axis is one Bloch(...) per axis.')
    detach = getattr(v, 'detach', None)
    if detach is not None:
        v = detach()
    try:
        return float(v)
    except Exception as err:
        if is_jax_typed(v):
            return None
        raise ValueError(f'Bloch({what}={v!r}): a real scalar in radians (or a traced scalar '
                         f'of jax or torch); got {type(v).__name__}.') from err


@spec
class Bloch(FaceType):
    """Both faces wrap with a phase; complex dtype. Exactly one of ``phase`` or ``k``.

    ``phase`` is in radians across the period; ``k`` is the wavevector component along the
    axis and is turned into a phase with the snapped box size when the scene is lowered.
    Either may be a traced scalar (a ``jax`` or ``torch`` value); it is a leaf of the traced
    parameters, not part of the static plan, so differentiating through it is allowed.
    """

    phase: object = None
    k: object = None

    def __post_init__(self):
        if (self.phase is None) == (self.k is None):
            raise ValueError('Bloch takes exactly one of phase= (radians across the period) '
                             'or k= (the wavevector component along the axis).')
        what = 'phase' if self.k is None else 'k'
        value = _concrete_scalar(what, self.phase if self.k is None else self.k)
        if value is not None and value == 0.0:
            raise ValueError(
                f'Bloch({what}=0) is Periodic(): the field repeats with no phase. Say '
                f'Periodic() -- a Bloch face makes the whole run complex, which doubles every '
                f'byte and every flop, and a phase of zero buys nothing for that price.')


@spec
class PEC(FaceType):
    """Perfect electric conductor: tangential E vanishes on the wall."""


@spec
class PMC(FaceType):
    """Perfect magnetic conductor: tangential H vanishes on the wall."""


@spec
class Symmetry(FaceType):
    """A mirror plane on this face. ``parity`` is the tangential-E eigenvalue: +1 or -1."""

    parity: int

    def __post_init__(self):
        # Normalised to a Python int at the door: the parity is a value in the static plan
        # and its skeleton, so ``1``, ``1.0`` and ``numpy.int64(1)`` must be the one mirror
        # (the skeleton encodes the type of a value, and a float ``1.0`` would hash apart).
        # A bool is not one: ``True`` is not a mirror eigenvalue anyone meant.
        p = self.parity
        if isinstance(p, float) and p in (1.0, -1.0):
            p = int(p)
        if isinstance(p, bool) or not isinstance(p, numbers.Integral) or int(p) not in (1, -1):
            raise ValueError(
                f'Symmetry(parity={self.parity!r}): parity is +1 (tangential E even across '
                f'the plane, PMC-like) or -1 (tangential E odd, PEC-like); nothing else is a '
                f'mirror eigenvalue.')
        object.__setattr__(self, 'parity', int(p))


_WRAPPING = (Periodic, Bloch)


def _pair(axis, value):
    """The ``(low, high)`` pair an axis declaration means, validated."""
    if isinstance(value, FaceType):
        if isinstance(value, Symmetry):
            raise ValueError(
                f'Boundaries({axis}=Symmetry(...)): a mirror on both faces of one axis is not '
                f'a half box, and stage 0 has no rule for it. Give a (Symmetry, PEC) or '
                f'(Symmetry, PMC) pair, mirror side first or last as the geometry needs.')
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, FaceType) for v in value):
        lo, hi = value
        if isinstance(lo, _WRAPPING) or isinstance(hi, _WRAPPING):
            try:
                same = type(lo) is type(hi) and (lo is hi or bool(lo == hi))
            except Exception:          # a traced phase cannot be compared; one object can
                same = False
            if not same:
                raise ValueError(
                    f'Boundaries({axis}=({lo}, {hi})): Periodic and Bloch wrap the axis, so '
                    f'both faces must carry the same declaration. Give one object for the '
                    f'axis, {axis}={lo if isinstance(lo, _WRAPPING) else hi}.')
        if isinstance(lo, Symmetry) and isinstance(hi, Symmetry):
            raise ValueError(
                f'Boundaries({axis}=(Symmetry, Symmetry)): a mirror on both faces of one axis '
                f'is not a half box; Symmetry goes on at most one face, with PEC or PMC on '
                f'the other.')
        return (lo, hi)
    raise TypeError(
        f'Boundaries({axis}={value!r}): an axis takes one face type (both faces) or a '
        f'(low, high) pair of face types -- Periodic(), Bloch(phase=...), PEC(), PMC(), '
        f'Symmetry(parity=...). Every axis is required; there is no default face.')


@spec
class Boundaries:
    """The six faces, declared per axis as one face type or a ``(low, high)`` pair."""

    x: object
    y: object
    z: object

    def __post_init__(self):
        for axis in AXES:
            object.__setattr__(self, axis, _pair(axis, getattr(self, axis)))

    def face(self, name):
        """The face type on ``name`` (``'x-'`` .. ``'z+'``)."""
        if name not in FACES:
            raise ValueError(f'no face named {name!r}; the faces are {FACES}')
        return getattr(self, name[0])[0 if name[1] == '-' else 1]

    @classmethod
    def periodic(cls, *, z):
        """``x`` and ``y`` Periodic, ``z`` as given: the periodic-cell preset."""
        return cls(x=Periodic(), y=Periodic(), z=z)

    @classmethod
    def open(cls, *args, **kwargs):
        """Absorbing faces all round -- stage 1. Named here so the call fails by name."""
        raise NotImplementedError(
            'Boundaries.open(...) is the absorber (PML) preset and belongs to stage 1; stage 0 '
            'has Periodic, Bloch, PEC, PMC and Symmetry only.')
