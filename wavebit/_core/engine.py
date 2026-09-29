"""Bridge from the solver classes to the single-source core.

``CoreSolveMixin`` reimplements the three ``_BaseRCWA`` solve methods on top of
:mod:`wavebit._core.solve_1d` / :mod:`wavebit._core.solve_2d`. It keeps their exact contract —
same arguments, same returned result mapping, and the same ``layer_info_list`` / ``T1``
side-effects that field reconstruction reads — so mixing it in swaps the *implementation*
without touching the public API.

Because the core is backend-agnostic, one mixin serves NumPy, JAX and PyTorch alike; the
per-backend ``transfer_method`` chains it replaces existed in triplicate.
"""
import contextlib as _contextlib
import functools as _functools

import numpy as _np

from . import parametric as _parametric
from . import rectilinear as _rectilinear
from . import boundary as _boundary
from . import factorization as _fac
from . import anisotropy as _anisotropy
from . import lattice as _lattice
from .backend import _concrete, _device_of, array_namespace, as_array, check_input_precision, checked_float
from .conventions import (_CHI_POLARIZATION_CLASH, _COMPLEX_CHI, _DIRECTION_CLASH,
                          _PSI_POLARIZATION_CLASH, _TETM_INCIDENCE_CLASH,
                          centre_parity, corner_column, swap_fourier_origin, incident_phase_index,
                          incident_sp_amplitudes, lit_orders, order_window,
                          parse_order_pair, reject_complex_angle, transverse_component_problem,
                          warn_if_xy_on_a_skewed_cell)
from .domain import Material, as_material, index_from_eps
from .materials import TabulatedMaterial, evaluate_material, evaluate_structure
from .enums import (Answers, DEFAULT_ANSWERS, DEFAULT_FACTORIZATION, DEFAULT_FORMULATION,
                    DEFAULT_FOURIER_COEFFICIENTS, DEFAULT_NESTING, DEFAULT_NORMAL_METHOD,
                    DEFAULT_PROJECTOR, answers_of, nesting_of, normal_method_of, projector_of,
                    near_match, near_match_hint, nvf_option_rename_hint, quantity_note,
                    Factorization, Formulation, FourierCoefficients, ModelingPath, Nesting,
                    Tetm, coefficients_of,
                    factorization_of, formulation_of, resolve_fourier_options)
from .fields import (field_1d, field_1d_conical, field_2d, field_2d_4ff,
                     is_count, to_mirrored_z, to_scene_frame)
from .geometry import StaticGeometry
from .result import Result, ResultSub
from .scattering import smm_1d as _smm_1d, smm_1d_conical as _smm_1d_conical, smm_2d as _smm_2d
from .solve_1d import solve_1d as _etm_1d, solve_1d_conical as _etm_1d_conical
from .solve_2d import solve_2d as _etm_2d
from .solve_2d_4ff import solve_2d_4ff as _solve_2d_4ff

# The four convolution slots that hold the permittivity's z row and column, in the order the
# solve methods take them. Naming them once keeps the dispatch message specific about which
# one was given -- see `CoreSolveMixin._formulation_for`.
_Z_ROW_SLOTS = ('epxz_conv_all', 'epyz_conv_all', 'epzx_conv_all', 'epzy_conv_all')


def _z_row_given(z_row, mu=None):
    """The z-row slots a ``solve_conv`` call was handed, by name: permittivity, then ``mu``.

    ``z_row`` is the four ``ep{xz,yz,zx,zy}_conv_all`` arguments and ``mu`` the permeability
    slots, whose z row is ``mu[5:]``. *Given*, not nonzero -- see
    :meth:`CoreSolveMixin._formulation_for`, which decides the formulation on this list, and
    :meth:`CoreSolveMixin._refuse_tetm_off_the_pure_path`, which names it.
    """
    given = [name for name, stack in zip(_Z_ROW_SLOTS, z_row) if stack is not None]
    if mu is not None:
        given += [name for name, stack in zip(('muxz', 'muyz', 'muzx', 'muzy'), tuple(mu)[5:])
                  if stack is not None]
    return given


# The relative wavelength step an auto_detune retry applies. Measured on the 2D detune
# ladder (see backend._CUTOFF_REL): the RayleighPointWarning band clears by ~1e-13 relative
# and the efficiency error is <= ~2e-6 from ~4e-13 out, so 1e-12 stands clear of the band by
# an order while staying physically negligible (5e-10 nm at 500 nm). Sized for double
# precision — at complex64 the step vanishes below the dtype's resolution.
_AUTO_DETUNE_REL = 1e-12

# Connecting algorithm -> the core solve for each grating type. ETM propagates an enhanced
# transmittance matrix (Moharam et al. 1995); SMM cascades per-layer scattering matrices. Same
# modes, same boundary conventions, same answers (~1e-14), both able to reconstruct fields; SMM
# costs ~1.1-1.4x on efficiencies and a little more once it also recovers the per-layer
# amplitudes. The two solves in each row take identical arguments, which is what makes the
# dispatch a lookup.
_SOLVES = {
    'ETM': {'1d': _etm_1d, 'conical': _etm_1d_conical, '2d': _etm_2d},
    'SMM': {'1d': _smm_1d, 'conical': _smm_1d_conical, '2d': _smm_2d},
}


#: "the caller did not name a cell" -- distinct from naming ``period=(1., 1.)``, which is a real
#: cell and the coherent partner of the ``wavelength=1.`` default in normalized units. The two
#: were one value until 2026-09-01, so a scene that forgot its cell was indistinguishable from
#: one written in normalized units and simply solved, on a cell 900x smaller than its own
#: wavelength. Only the raster path could reach that: a parametric shape carries absolute
#: lengths, so it is already refused by "shape is larger than the period".
_CELL_UNSET = object()


def _concrete_period(value):
    """``period`` as two plain floats, or ``None`` if it has none to give.

    ``period`` is a differentiable parameter (see ``tests/test_gradients.py``), so
    it arrives in three shapes and all three are legitimate. Plain numbers convert outright. A
    grad-tracking torch tensor has to be detached first. A ``jax.grad`` tracer refuses both a
    whole-array conversion *and* ``float()`` — unlike the boolean narrowing
    :func:`~wavebit._core.factorization._as_bools` relies on, there is no way to read a
    scalar out of one — so the honest answer there is ``None``.

    ``None`` costs nothing when a frozen geometry is present, because
    :meth:`CoreFactorizationMixin._geometry_lattice_for` recovers it from the record. Without
    one it falls back to the legacy index-space normal, unchanged.
    """
    import numpy as np
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach()
    try:
        p = np.atleast_1d(np.asarray(value, dtype=float))
    except Exception:
        try:
            items = value if isinstance(value, (list, tuple)) else list(value)
            p = np.asarray([float(v.detach() if hasattr(v, 'detach') else v) for v in items],
                           dtype=float)
        except Exception:             # a jax tracer: no primal to read
            return None
    if p.size == 1:
        return float(p[0]), float(p[0])
    return float(p[0]), float(p[1])


def _concrete_2x2(A):
    """A ``(2, 2)`` cell as a concrete array for a check, or ``None`` when it has no value.

    Read through :func:`~wavebit._core.backend.checked_float`, so a ``jax.grad`` tracer is read
    by its primal and a ``jax.jit`` or ``jax.vmap`` one is ``None``. Indexed, so a
    :class:`Lattice`, an array and a nested tuple read alike.
    """
    try:
        return _np.array([[checked_float(A[i][j]) for j in (0, 1)] for i in (0, 1)])
    except TypeError:
        return None


def _holds_tracer(value):
    """Whether ``value`` is, or carries, a JAX tracer -- a value that cannot outlive its trace.

    Structural, never a read: ``isinstance`` against ``jax.core.Tracer`` over the containers a
    condition arrives in -- a shape layer's dicts and lists, a :class:`Lattice`'s two vectors,
    a :class:`DrawnOn` raster, a :class:`Material`'s two slots. A tracer under ``jax.grad``
    counts as much as one under ``jax.jit``: both are dead the moment the traced call
    returns, and a value that answers a check today (the grad tracer's primal) still raises
    ``UnexpectedTracerError`` from every array op tomorrow. A PyTorch leaf under autograd is
    not one -- it converts, detaches and reads at any time -- and stays where it was put.
    ``False`` whenever JAX is not in the process, which is what makes this free to ask on the
    other two backends.
    """
    import sys
    jax = sys.modules.get('jax')
    if jax is None:
        return False
    tracer = jax.core.Tracer
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, tracer):
            return True
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        elif isinstance(v, _lattice.Lattice):
            stack.extend((v.u, v.v))
        elif isinstance(v, _lattice.DrawnOn):
            stack.extend((v.values, v.lattice))
        elif isinstance(v, Material):
            stack.extend((v.epsilon, v.mu))
    return False


def _refuse_shaped_half_space(material, where):
    """A half-space is one uniform material, so its numbers must be scalars.

    Checked here rather than in :func:`~wavebit._core.domain.as_material`, which is
    deliberately permissive: it cannot tell a vmapped tracer from a raster grid, and refusing
    both to catch one would break the design-variable case (``tests/test_material.py::
    test_arrays_and_tracers_are_not_refused``). This slot can be precise where that one cannot,
    because *half-space* is a stronger claim than *material* -- and because the test is a
    **shape**, never a value. A shape is static under ``jit``, ``grad`` and ``vmap`` alike, and
    a vmapped scalar reads ``()`` inside the trace, so nothing traced is caught by this.

    Without it a shaped half-space travelled as far as ``_separable_kxy``, where it met the
    order range and raised ``operands could not be broadcast together with shapes (1, 8) and
    (5,)`` -- loud, but naming neither the argument at fault nor what a half-space is.
    """
    for name, value in (('epsilon', material.epsilon), ('mu', material.mu)):
        shape = _anisotropy.tensor_shape(value)
        size = 1
        for dim in shape:
            size *= int(dim)
        if size > 1:
            raise ValueError(
                f'`{where}` is a half-space, so its {name} must be a single value, but it has '
                f'shape {shape}. A half-space is semi-infinite and homogeneous: a patterned or '
                'graded medium is a *layer*, which goes in `structure` with a thickness of its '
                'own. An anisotropic half-space is not supported either -- its modes are the '
                'plane waves that `sqrt(eps*mu)` alone defines.')


def _half_space_index(material, where='top'):
    """``sqrt(eps*mu)`` -- the one place a half-space material becomes an index.

    **The branch is verified, not chosen.** A passive medium has ``Im(eps) >= 0`` in this
    project's ``e^{-iwt}`` convention, so ``eps*mu`` lies in the closed upper half plane, whose
    principal square root lies in the first quadrant: ``Im(n) >= 0``, which is loss again. That
    is the same argument :func:`~wavebit._core.conventions.boundary_kz` rests on, and it is why
    :func:`~wavebit._core.domain.index_from_eps` -- which is the principal root and nothing
    else -- is already the right function here. No branch rule is invented.

    ``mu`` multiplies in rather than being dropped: the index of a magnetic medium is
    ``sqrt(eps*mu)``, and every ``kz``, every Snell relation and the whole order grid are
    written in terms of it. The two admittances the boundary needs -- ``kz/eps`` and ``kz/mu``
    -- are recovered from ``mu`` separately (see :func:`wavebit._core.solve_2d.solve_2d`), so
    nothing here has to keep them apart.

    An **unnamed** ``mu`` is left out of the product entirely rather than folded in as a
    literal one. That is not a micro-optimisation: it keeps ``sqrt(eps)`` bit-for-bit what it
    was for every non-magnetic medium, which is all of them by default, so no golden moves by a
    rounding step for a feature the scene does not use. The test is
    :attr:`~wavebit._core.domain.Material.is_magnetic`, so it is "was a permeability named"
    rather than "does it equal one" -- which is why this reads the same eagerly and under
    ``jax.jit`` where the value would be a tracer. A material that names ``mu = 1`` takes the
    second line and multiplies by an exact one, so the two answers agree anyway.
    """
    _refuse_shaped_half_space(material, where)
    if not material.is_magnetic:
        return index_from_eps(material.epsilon)
    return index_from_eps(material.epsilon * material.mu)


def _recorded_lattice(normal_field_source):
    """The cell basis a frozen record was taken on, or ``None`` if it does not name one.

    The record stores the basis scaled to ``|a1| == 1``
    (:func:`~wavebit._core.factorization._lattice_key`), because that is all the normal field
    depends on — a uniform rescale of the cell leaves it identical. That scaled basis is itself
    a perfectly good basis to hand back: it re-derives to exactly the key the staleness check
    compares against, so a replay can never be refused for having reconstructed it differently.

    Layers disagreeing, or a record predating the basis being an input, both answer ``None``:
    neither names a single cell, and inventing one would be worse than falling back.
    """
    import numpy as np
    if not normal_field_source:
        return None
    entries = [e for e in normal_field_source if len(e) > 2]
    if len(entries) != len(normal_field_source):
        return None
    keys = {e[2] for e in entries}
    if len(keys) != 1:
        return None
    key = keys.pop()
    if key is None:
        return None
    return np.asarray(key, dtype=float).reshape(2, 2)


#: Raised by :meth:`CoreSolveMixin._require_field_declaration` — the refusal the ``answers``
#: declaration exists to be able to give. It names the keyword to change, because the thing
#: that went wrong is a declaration and not a missing call: by the time ``reconstruct_field``
#: runs, the state it wants was never built, and no argument here can bring it back.
#: Why ``solver.scattering`` has nothing, one message per reason. Modelled on
#: :data:`_FIELD_NOT_DECLARED` and split for the same stated reason: "run a solve first" is
#: false advice for a solver whose solves cannot produce one, and running another would change
#: nothing. A bare ``None`` said all three of these at once, and then surfaced as
#: ``'NoneType' object has no attribute 'S11'``.
_SCATTERING_WRONG_ALGORITHM = (
    "there is no scattering matrix because layer_algorithm={got!r} forms none. The ETM fixes "
    "one illumination's boundary condition and solves it directly -- it never builds a "
    "both-sides operator, which is exactly why it is the cheaper default -- so solving again "
    "with this solver changes nothing. Pass layer_algorithm='SMM', whose cascade builds one to "
    "answer at all and now keeps it. Everything else the two algorithms produce is identical, "
    "so this is the one thing that choice buys."
)
_SCATTERING_NOT_DECLARED = (
    "there is no scattering matrix because this solver declared answers={got!r}, which keeps "
    "power only. An S *is* amplitude information -- its blocks applied to the incident vector "
    "are the reported amplitudes -- so it appears from answers='amplitude' up. Declare that "
    "(the default) or higher and solve again; it costs no arithmetic and four matrices, "
    "whatever the stack's depth."
)
_SCATTERING_NOT_SOLVED = (
    "there is no scattering matrix yet: run solve() -- or solve_conv(), if you built the "
    "convolution matrices yourself -- first. The declaration is right for one "
    "(layer_algorithm='SMM', answers='amplitude' or higher), so a solve will fill it."
)

_SCATTERING_SOLVED_ELSEWHERE = (
    "there is no scattering matrix on this solver: the last call that moved its conditions was "
    "the jitted solve_field(...), which keeps its keywords on the solver but solves inside its "
    "own jax.jit trace, so this solver holds none for the scene it now reports. Call solve() "
    "to build this scene's."
)

_FIELD_NOT_DECLARED = (
    "reconstruct_field needs the per-layer modes and amplitudes the solve keeps as it goes, "
    "and this solver declared answers={got!r}, which does not keep them. That is a "
    "construction-time declaration rather than an argument to this call on purpose: the "
    "state has to be retained *during* the solve, so there is nothing left to reconstruct "
    "from once it has returned. Set answers='field' and solve again.\n"
    "The default is 'amplitude' because the field state is 4.25 matrices of side "
    "2*n_orders_u*n_orders_v per layer on the 2D path (3.75 conical, 5 pure 1D) -- 21.7 MB at "
    "truncation_order=(8, 8), 240 MB at (15, 15), 733 MB at (20, 20), per layer -- while the amplitudes "
    "it does keep are boundary vectors. An optimization against a fixed objective needs the "
    "amplitudes and never reconstructs a field.")


def _centred(a, odd):
    """``a`` with every entry ``odd`` marks negated: an exact sign bit, traced or not.

    ``odd`` is a host ``bool`` array with ``a``'s size -- a parity grid read by label, or the
    pair grid of a scattering block. See :meth:`CoreSolveMixin._finish`.
    """
    xp = array_namespace(a)
    mask = xp.reshape(as_array(xp, odd, device=_device_of(a)), a.shape)
    return xp.where(mask, -a, a)


def _efficiency_only(de_ri, de_ti, power_entering):
    """The ``ResultSub`` payload of a solve declared ``answers='efficiency'``.

    Every amplitude slot is ``None`` rather than absent, because ``ResultSub`` has a fixed
    field list and a fixed pytree structure — ``None`` is an empty subtree, so a result from
    this rung flattens to fewer leaves without becoming a different kind of object. The s/p
    split of the efficiencies is ``None`` for a plainer reason: this rung never formed it.
    """
    return {'r_s': None, 'r_p': None, 't_s': None, 't_p': None,
            'R': de_ri, 'R_s': None, 'R_p': None, 'T': de_ti, 'T_s': None, 'T_p': None,
            'power_entering': power_entering, 'r_xyz': None, 't_xyz': None}


#: ``pol`` was three things at once, and this is the rename. See :attr:`CoreSolveMixin.tetm`.
#: What NumPy answers to `grad` and `fit`. A named refusal rather than the `AttributeError` a
#: missing method raises: "'SolverNumpy' object has no attribute 'grad'" is true and says
#: nothing -- not that the absence is deliberate, not that two other backends have it, not that
#: the scene needs no change to move. NumPy has no autodiff, and that is a fact about NumPy.
_NO_AUTODIFF = (
    "{name} needs automatic differentiation and the NumPy backend has none. This is a "
    "property of the backend rather than of the scene: the same solver on JAX or PyTorch "
    "differentiates it, and `backend=` is the only thing that changes.\n"
    "    solver = wavebit.rcwa(backend=1, ...)   # JAX, then `optax` for `fit`\n"
    "    solver = wavebit.rcwa(backend=2, ...)   # PyTorch, then `torch.optim`\n"
    "Ask before building, with `type(solver)._DIFFERENTIABLE` or the manifest's "
    "`backends` section, which publishes the same fact.")


def _refuse_unknown_keywords(kwargs):
    """Refuse a constructor keyword that is not a condition.

    The constructors collect the leftovers through ``**unknown`` rather than letting Python
    raise, because the catch-all is what makes a typo reachable at all: without a refusal here
    it would be swallowed silently.

    This used to also name the retired spellings -- ``ucell``, ``n_top``, ``n_bot``, ``pol``,
    ``ucell_materials`` -- and say what replaced each and that the *quantity* changed with the
    name. Those messages are gone with the rest of the retired machinery: wavebit has never
    been released, so the reader they were written for does not exist yet. The renames belong
    in the first release's notes instead.
    """
    if kwargs:
        raise TypeError(
            f'unexpected keyword argument(s) {sorted(kwargs)!r}.'
            f'{nvf_option_rename_hint(kwargs)}')


def _unknown_attribute(owner, name, known, kind='conditions'):
    """The refusal of an assignment to a public name nothing reads, as an exception to raise.

    One function because there are three attribute doors and they must refuse the same
    mistake the same way: the solver's own ``__setattr__`` (``CoreFactorizationMixin``), the
    source-excitation facade's (``illumination.CoherentSolver``), which forwards the scene's
    conditions and used to store every other public name on itself in silence --
    ``s.thicknes = [900.0]`` there was accepted and changed nothing, when the solver had
    refused it since 2026-08-27 -- and the spec dataclasses' (``domain._DeclaredFieldsOnly``),
    where ``inc.thet = 0.2`` was stored and ``as_kwargs`` still wrote ``theta=0``. ``owner`` is
    the class name the caller assigned on; ``known`` is what that class can set, and the
    near-match hint (:func:`~wavebit._core.enums.near_match_hint`) is drawn from it alone.
    ``kind`` names what ``known`` holds: a solver's conditions, a dataclass's fields.
    """
    known = sorted(known)
    hint = near_match_hint([name], known)
    return AttributeError(
        f'{owner} has no attribute {name!r} to set.{hint} Refused rather '
        f'than stored, because storing it is silent: the solve reads the attribute it '
        f'knows about, so the answer comes back for the value you meant to change and '
        f'nothing says otherwise.\n'
        f'  Settable {kind}: {", ".join(known)}.\n'
        f'  To attach state of your own, use a name starting with an underscore, or '
        f'declare it as a class attribute on a subclass.')


#: ``mu`` is a *field*, and only the raster states a layer as fields. Refused by name on the
#: two shape paths rather than ignored, because a permeability that reached no solve would be
#: exactly the unread knob ``_reject_unread_normal_method`` exists to refuse one level down.
_MU_ON_A_SHAPE_PATH = (
    'mu= is the raster path\'s permeability field and has no meaning here: this structure is '
    'written as shapes, where the permeability belongs to the material that fills each region '
    '-- wavebit.Material(epsilon, mu). A layer-shaped mu beside a shape recipe has no region '
    'to attach itself to, so it is refused rather than dropped. The two fields are one pair '
    'and either may be assigned second, so this is also what assigning a shape structure over '
    'a solver that already holds a mu raises -- clear the mu first if the swap is intended.')

#: `nextafter(float32(pi/2), 0)` and `float32(pi/2)`, as plain Python floats. They are the
#: clamp and the threshold `_incident_kxy` uses, and they are constants: computing them in each
#: backend's own namespace was the reason the method looked as though it had to live there.
_HALF_PI_F32 = float(_np.float32(_np.pi / 2))
_HALF_PI_F32_BELOW = float(_np.nextafter(_np.float32(_np.pi / 2), _np.float32(0)))


def device_scoped(func):
    """Run ``func`` inside the solver's own device scope.

    Replaces a `jax_device_set` that lived in the JAX backend and named `jax.default_device`
    directly. The methods it decorates all have counterparts on the other two backends, so
    the decorator has to stop being one backend's before those bodies can become one body.

    The scope itself is :meth:`CoreSolveMixin._device_scope`, which is a declared member: the
    base answers `nullcontext` and JAX overrides it. Declared rather than duck-typed because a
    missing device scope is invisible -- everything still computes, on the wrong device, and a
    CPU-only test machine cannot tell the difference.

    That last sentence was a prediction, and on 2026-09-10 it came true. Five internal methods
    carried this decorator and the public entry points did not -- `solve`,
    `eps_conv_matrices`, `mu_conv_matrices`, `freeze_geometry`, `order_kxy` -- so on a CUDA
    jaxlib a `device='cpu'` solve ran on the GPU while `solver.device` reported the CPU, and
    omitting `device` did the same. Decorating them is one part of the fix. The second is
    `_array_device` in the JAX backend: the stored arrays are built by setters that no
    decorator reaches, and once the builders committed the convolution matrices to where the
    structure happened to live, the scope around `solve_conv` was open and overruled. The
    third is :meth:`CoreSolveMixin._placed`, for the arrays a caller hands in rather than a
    setter builds. The scope places what a call builds from scalars; the arrays carry where
    the work runs.
    `tests/test_jax_device_placement.py` checks both on two host devices, which any machine
    has, and counts GPU allocations where there is a GPU.
    """
    @_functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        with self._device_scope():
            return func(self, *args, **kwargs)
    return wrapped


class CoreSolveMixin:
    """Provides ``_solve_1d`` / ``_solve_1d_conical`` / ``_solve_2d`` backed by the core.

    Private since the public entry point became :meth:`CoreFactorizationMixin.solve`. All
    three take convolution matrices already built and each does strictly less than ``solve``
    does, so leaving them public put three longer ``solve_*`` names beside the shortest one
    that does the most. Handing in matrices you built yourself is still a supported thing to
    do -- that is ``solve_conv``, which routes to these.
    """

    #: Whether this backend can differentiate through a solve -- i.e. whether `grad` and `fit`
    #: are here at all. `False` is the base answer and NumPy keeps it; JAX and PyTorch override
    #: it beside `_BACKEND_ID`. Declared rather than discovered so a caller can *ask* before
    #: building a solver it cannot optimize, and so the manifest can publish it and be checked
    #: against it. See `_NO_AUTODIFF`.
    _DIFFERENTIABLE = False

    _excitation = None

    def __init__(self, top=1., bot=1., theta=None, phi=None, psi=None, chi=None, tetm=None,
                 truncation_order=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), layer_algorithm='ETM', excitation=None,
                 device=0, complex_dtype=0, polarization=None, kx=None, ky=None):
        """The scene, set once through the setters that maintain it.

        One body for three backends. The three it replaced were textually identical apart from
        two defaults and a stale comment, and identical in effect too: measured before the fold,
        all three ended construction with the same 37 attribute names and the same `__dict__`
        insertion order (digest `e46a77c5060e` on each), which is to say the same setters called
        in the same order. Everything a backend does differently is already behind a declared
        member -- `xp`, `_ARRAY_TYPES`, `_array_device`, `_device_scope`, `_resolve_device`,
        `_resolve_precision` -- so there is nothing here to specialise.

        The two defaults are written in wavebit's own vocabulary rather than a namespace's.
        `0` is what `_resolve_precision` and `_resolve_device` already accept on every backend,
        and it resolves to exactly what `np.complex128` / `jnp.complex128` /
        `torch.complex128` and `'cpu'` did -- the three spellings were three ways to say the
        thing this one says once.

        On JAX this runs **under tracing** as well as eagerly: `SolverJax._tree_unflatten`
        rebuilds by `cls(**named, **aux_data)`, so `solve_field` re-enters it on every call
        with tracers in the child slots. The one value-shaped test below is safe because it is
        structural -- `None` against present, never a comparison against a traced number. See
        `docs/` on the pytree, and do not add a branch here that reads one.
        """
        self.device = device

        # The width, once. `_resolve_precision` is not the setter, because the setter re-casts
        # every stored array and none of them exist yet.
        self._complex_dtype, self._float_dtype, self._int_dtype = (
            self._resolve_precision(complex_dtype))

        # Materials in, an index out: the two setters store the Material and derive the
        # `sqrt(eps*mu)` every solve reads. See `CoreSolveMixin.top`.
        self.wavelength = wavelength
        self.top = top
        self.bot = bot

        self.theta = 0. if theta is None and kx is None and ky is None else theta
        self.phi = phi
        self.kx = kx
        self.ky = ky
        # Order matters: `tetm` refuses to coexist with a named incidence, so it goes first
        # and the two spellings below raise against it rather than silently winning.
        self.tetm = tetm
        self.psi = psi
        self.chi = chi
        self.polarization = polarization
        if self._tetm is None and self._psi is None and polarization is None:
            self._default_psi()

        self.truncation_order = truncation_order
        self.period = period
        self.thickness = thickness
        self.layer_algorithm = layer_algorithm
        self.excitation = excitation

        self.layer_info_list = []
        self.T1 = None
        # SMM only, from `answers='amplitude'` up; see `CoreSolveMixin._finish`.
        self._scattering = None

    @device_scoped
    def _separable_kxy(self, wavelength):
        """The in-plane wavevector on each retained order, one axis at a time.

        The last body that was written three times. They differed in the namespace's `arange`,
        in `.astype` against Torch's `.type`, and in Torch alone naming `device=` and
        `dtype=` -- every one of which the array API already absorbs.

        `dtype=` is *not* carried over, and that is measured rather than assumed. It looked
        load-bearing at single precision, where an untyped integer `arange` multiplied by a
        float32 gives a float64 intermediate on NumPy. Torch promotes that product to float32
        regardless, so its `dtype=` never changed a width -- dropping it left each backend's
        arrays byte-identical to what that backend returned before, at both widths. `device=` is
        the half that was doing the work, and it stays, spelled `_array_device` (`None` where
        nothing is placed).

        NOTE, separate from this move: at ``complex_dtype=1`` NumPy still computes these offsets
        through a float64 intermediate while JAX and Torch stay in float32, so NumPy's kx/ky
        differ from theirs in the last bits. Pre-existing, and not something a fold should
        decide -- the declared width governs the answer, which
        ``test_the_declared_width_is_the_width_answered_in`` checks, and it is met either way.

        Nor are the three byte-identical at float64. Torch evaluates
        ``wavelength / self.period[0]`` (a Python float over a tensor) as ``reciprocal()``
        times the float, which can land one ulp off: ``0.6 / 1.6`` is ``0.375`` on Torch and
        ``0.37499999999999994`` on NumPy and JAX, and with a direct ``kx = 2*pi/P`` at those
        numbers order ``-1``'s ``kx``, zero in exact arithmetic, comes out ``-5.55e-17`` on
        Torch and ``+0.0`` on the other two. What that does to the per-order s/p frame at an
        exact ``kx = 0`` is the ``ky == 0`` bullet of ``docs/conventions.md``.
        """
        xp = self.xp
        (cx, hx), (cy, hy) = self.order_window

        orders_u = xp.arange(cx - hx, cx + hx + 1, device=self._array_device)
        orders_v = xp.arange(cy - hy, cy + hy + 1, device=self._array_device)

        k_inc_x, k_inc_y = self._incident_kxy(wavelength)

        kx = xp.astype(k_inc_x + orders_u * (wavelength / self.period[0]), self.complex_dtype)
        ky = xp.astype(k_inc_y + orders_v * (wavelength / self.period[1]), self.complex_dtype)

        return kx, ky

    def _lit(self):
        """The excitation resolved onto the retained window, or ``None`` if none was named.

        Resolved here, at the boundary, and handed to the solves already done -- the same
        arrangement :meth:`_incidence_pair` has for the polarization. The solves never see
        ``excitation`` itself, so they never have to know which window it widened.

        These are the **public** weights, unit at ``x = y = 0``: what the boundary spectra's
        incoming wave is built from. The kernels take :meth:`_kernel_lit` instead.
        """
        excitation = self._excitation
        if excitation is None and (self.kx is not None or self.ky is not None):
            # Direct k can request a zero-power evanescent incident wave. Give the
            # default order the same positive-power check as explicit excitation.
            excitation = {(0, 0): 1.0}
        return lit_orders(excitation, *self.order_window)

    def _kernel_lit(self):
        """:meth:`_lit` restated for the kernels, whose Fourier origin is the cell corner.

        A public weight is referenced at the cell centre; about the corner the same wave carries
        ``exp(-1j * k_inc . o) * (-1)**(m + n)``, ``o = (a1 + a2)/2``. The kernels are handed the
        sign (:func:`~wavebit._core.conventions.corner_column`), which is exact, and the scalar
        -- a normalisation of the whole field, cancelling from every amplitude and efficiency --
        is put back on the field by :meth:`_centre_phase`. :meth:`_finish` signs the kernels'
        amplitudes back onto the centre.
        """
        return corner_column(self._lit(), *self.order_window)

    @property
    def order_window(self):
        """Which orders this solve retains: ``(centre, half_width)`` per axis.

        ``(0, truncation_order)`` unless an :attr:`excitation` widens it -- see
        :func:`~wavebit._core.conventions.order_window` for why lighting an off-centre order
        moves the window rather than leaving that beam short of neighbours.
        """
        return order_window(self.truncation_order[0], self.truncation_order[1], self._excitation)

    @property
    def excitation(self):
        """Which diffraction orders the illumination arrives on, and with what amplitude.

        ``None`` — the default — is one plane wave on the specular order, which is what this
        library has always solved and what every other incidence keyword describes. Naming a
        dict lights several orders at once::

            rcwa(..., excitation={0: 1.0, 1: 0.6 - 0.8j})        # 1D: keyed by the x order
            rcwa(..., excitation={(0, 0): 1.0, (1, 0): 0.5j})    # 2D: by the pair

        The weights are *spatial*: :attr:`psi` / :attr:`polarization` / :attr:`tetm` still say
        which polarization arrives, and every lit order carries that same one, **in its own
        s/p frame**: a weight multiplies that order's ``s_hat`` / ``p_hat``, so on a 1D solve at
        ``phi = 0`` a lit order with ``kx < 0`` arrives polarized along ``-y`` (TE). The pure 1D
        path means the same since 2026-09-27; before, it multiplied ``+y`` into every lit
        order. Each weight is the amplitude of its plane wave **at the public origin**,
        ``x = y = 0`` on the lit face -- the cell centre every coordinate is measured from --
        so these are the same weights ``OrderExcitation(orders=...)`` takes, and the same light
        (``docs/conventions.md``). Until 2026-09-27 they were referenced at the cell corner,
        which negated every lit order with ``m + n`` odd relative to the rest. They may be
        traced -- an incident field is a legitimate design variable -- while the **keys** may
        not, because which entries of the incident column are non-zero decides array
        structure rather than a value.

        What it is for is a coherent superposition that a single solve cannot otherwise
        express. A *single* non-specular order is not that: exciting order ``m`` at ``theta``
        is the same plane wave as exciting the specular order at the angle whose ``kx``
        matches, so it buys nothing but a relabelling (on the ``ky == 0`` row, up to the sign
        of an order whose ``kx`` is exactly zero in one spelling and a rounding away from it in
        the other -- its frame is ``sign(kx) * y_hat``). Several at once is different, and the
        alternative -- one solve per beam, indices realigned, amplitudes added -- costs a solve
        per beam for the same answer.

        **Every lit order must propagate in the incidence medium.** One past the light line
        carries no time-averaged power, so it would drive the structure while contributing
        nothing to the flux every efficiency is divided by; it is refused rather than silently
        redefining what "efficiency" means. An evanescent excitation is a near-field source,
        which is a different problem than an incident plane wave.
        """
        return self._excitation

    @excitation.setter
    def excitation(self, value):
        if value is None:
            self._excitation = None
            return
        if not isinstance(value, dict):
            raise TypeError(
                f'excitation must be a dict of order -> complex amplitude, got '
                f'{type(value).__name__}. A 1D scene keys it by the x order alone '
                '({0: 1.0, 1: 0.5j}); a 2D one by the pair ({(0, 0): 1.0, (1, 0): 0.5j}). '
                'Leave it unset for the ordinary single wave on the specular order.')
        if not value:
            raise ValueError(
                'excitation is empty, so nothing would be lit and every answer would be zero. '
                'Leave it unset for the ordinary single wave on the specular order.')
        # Which orders are legal depends on `truncation_order`, which may still change; that check
        # belongs where the solve reads them (`conventions.lit_orders`). This one is the shape.
        self._excitation = {order: self._placed(amplitude) for order, amplitude in value.items()}

    _incidence_side = 'top'

    @property
    def incidence_side(self):
        """Which medium the illumination arrives from: ``'top'`` (default) or ``'bottom'``.

        ``top`` / ``bot`` / ``structure`` order are *scene* labels and never move — this
        selects the illumination only. Results are incidence-relative (the port model):
        ``R`` is power back into the incidence medium, ``T`` power through the far side,
        the Fresnel amplitudes keep their contract, and ``theta`` / ``phi`` are measured
        against the incidence medium's normal (``n_bot``'s, for ``'bottom'``). User
        post-processing is therefore identical for both sides. See ``docs/conventions.md``.
        """
        return self._incidence_side

    @incidence_side.setter
    def incidence_side(self, value):
        if value not in ('top', 'bottom'):
            raise ValueError(
                f"incidence_side must be 'top' or 'bottom', got {value!r}")
        self._incidence_side = value

    @property
    def _kernel_incidence_side(self):
        """Select the entry port of the positive-z propagation kernel.

        Public layers run from the bottom (z=0) upwards. The private kernels' historical
        ``top``/``bottom`` port names mean start/end of their positive-z list, respectively;
        they are not scene labels. Bind the lower medium to the start port and the upper
        medium to the end port at the four call sites below. No geometry or tensor is
        reflected at this boundary: the kernel and the public Cartesian axes coincide.
        """
        return 'top' if self.incidence_side == 'bottom' else 'bottom'

    _answers = DEFAULT_ANSWERS

    @property
    def answers(self):
        """What this solve keeps the state to answer: ``'amplitude'`` (default) or ``'field'``
        — or ``'efficiency'``. A *declaration*, in the sense :attr:`tetm` is one.

        ``tetm`` declares a degree of freedom away; this declares a *question* away. Saying
        ``answers='amplitude'`` is not a request to be economical, it is a statement that a
        near field is not something this problem asks about — and the per-layer state that
        would answer one is then never built, rather than built and discarded.

        ====================  ======================================================
        ``'efficiency'``      ``R`` / ``T`` per order and nothing else
        ``'amplitude'``       **default** — the Fresnel amplitudes, their s/p-resolved
                              efficiencies, and ``s_inc`` / ``p_inc``
        ``'field'``           what :meth:`reconstruct_field` walks
        ====================  ======================================================

        Each rung answers everything below it, so ``'field'`` is a superset and not a mode.

        The default is ``'amplitude'`` because that is where the cost cliff is. The
        amplitudes are boundary vectors already solved for; the field state is 4.25 matrices
        of side ``2*n_orders_u*n_orders_v`` per layer on the 2D path — 240 MB per layer at
        ``truncation_order=(15, 15)``, 733 MB at ``(20, 20)``. The two differ by three orders of
        magnitude, and the runs that are large enough to care are optimization sweeps
        against a fixed objective, which never call :meth:`reconstruct_field` at all.

        Why it is settled at construction rather than passed to the solve: this is a claim
        about the problem, which is knowable when the problem is set up; it rides in the JAX
        pytree's ``aux_data``, where it is static under ``jit`` instead of a traced value a
        branch cannot read; and asking for a field it excludes is a contradiction, which the
        code refuses by name (see :meth:`_require_field_declaration`) rather than silently
        satisfying at every scale.

        Assigning it after a solve is legal and takes effect on the *next* one — the state is
        kept as the solve runs, so nothing here can recover what an earlier one did not keep.
        """
        return self._answers

    @answers.setter
    def answers(self, value):
        self._answers = answers_of(value)

    def _require_field_declaration(self):
        """Refuse :meth:`reconstruct_field` on a solver that declared no field, by name.

        Called before the state is read, so the failure names the declaration to change
        rather than surfacing as ``_require_solved``'s "run a solve first" — which would be
        false advice here, since running another solve of the same solver changes nothing.
        """
        if not self._answers.covers(Answers.FIELD):
            raise RuntimeError(_FIELD_NOT_DECLARED.format(got=self._answers.value))

    # ------------------------------------------------------------------ the two half-spaces
    #
    # Stored as Materials, read by the solves as an index. The conversion happens **once, at
    # ingest**, and that asymmetry with the layers is deliberate rather than an inconsistency
    # left behind. A layer's material is squared nowhere any more -- the factorizers take
    # permittivity and now get it. A half-space is not: `n_top` appears to the *first* power
    # at about sixteen sites where the quantity genuinely is an amplitude ratio (the
    # incident-power normalisation, the efficiency denominators, the Fresnel `-i*n`
    # conversion, the TM `n_ref/n_trn` step, `check_physical`'s argument), and flipping those
    # would put a complex square root, with a branch choice, at each one -- in a codebase whose
    # own trap list names "the half-space sqrt branch". So the setter takes the root once,
    # where the branch can be argued about in one place, and the ~43 internal readers keep
    # working unchanged.
    _top = Material(1.0)
    _bot = Material(1.0)
    _n_top = index_from_eps(1.0)
    _n_bot = index_from_eps(1.0)

    @property
    def top(self):
        """The superstrate, as a :class:`~wavebit._core.domain.Material`.

        Takes a ``Material`` or a bare number (the scalar shorthand for
        ``Material(epsilon=x)``) -- relative permittivity, with ``Im(eps) > 0`` for loss.
        A ``TabulatedMaterial`` is retained and reevaluated when wavelength changes;
        reading this property returns its evaluated numerical ``Material``.
        Assigning it also refreshes :attr:`n_top`, which is what the solves read.
        """
        return self._top

    @top.setter
    def top(self, value):
        """Both, or neither. ``_half_space_index`` refuses, and it used to refuse too late.

        This wrote ``_top`` and *then* derived ``_n_top``, and the derivation is where
        ``_refuse_shaped_half_space`` lives. So a caller who handed over a shaped epsilon --
        a raster where a half-space belongs -- caught the ``ValueError`` and was left with
        ``top`` holding the Material the guard had just refused while ``n_top`` still held the
        previous medium's index. The two disagreed, and the solve reads the index, so it went
        on answering the old question under the new label. With a ``mu`` named in the refused
        Material it was worse: ``is_magnetic`` flipped, which selects a different arithmetic
        path entirely.
        """
        material = self._placed(as_material(self._material_at(value, self.wavelength), 'top'))
        self._n_top = _half_space_index(material, 'top')
        self._top = material
        self._top_dispersion = value if isinstance(value, TabulatedMaterial) else None

    @property
    def bot(self):
        """The substrate, as a :class:`~wavebit._core.domain.Material`. See :attr:`top`."""
        return self._bot

    @bot.setter
    def bot(self, value):
        """Both, or neither -- see :attr:`top`, which had the same order and the same defect."""
        material = self._placed(as_material(self._material_at(value, self.wavelength), 'bot'))
        self._n_bot = _half_space_index(material, 'bot')
        self._bot = material
        self._bot_dispersion = value if isinstance(value, TabulatedMaterial) else None

    @property
    def n_top(self):
        """The superstrate's refractive index, ``sqrt(eps*mu)`` -- **derived, read-only**.

        Every solve reads this rather than :attr:`top`, and it is not settable: two spellings
        of one medium that could disagree is exactly the state this type exists to remove.
        Assign :attr:`top` instead.
        """
        return self._n_top

    @property
    def n_bot(self):
        """The substrate's refractive index, ``sqrt(eps*mu)`` -- derived, read-only.

        See :attr:`n_top`.
        """
        return self._n_bot

    @property
    def n_inc(self):
        """The incidence medium's index — the one ``theta`` / ``phi`` / ``kx`` live in.

        Real angles use the phase direction with uniform incident intensity along
        the surface. In a lossless medium this reduces to ``n_inc*sin(theta)``;
        absorbing media use the full material in :meth:`_incident_kxy`.
        """
        return self.n_bot if self._incidence_side == 'bottom' else self.n_top

    def _psi_set_explicitly(self):
        """Called by the backend ``psi`` setters on a non-``None`` assignment.

        Refuses the assignment while a ``polarization`` pair is set (the two spell the same
        physical input), and otherwise records that ``psi`` was *given* rather than derived
        from ``pol`` — the distinction the ``polarization`` setter's own refusal reads.
        """
        if self._polarization is not None:
            raise ValueError(_PSI_POLARIZATION_CLASH)
        if self._tetm is not None:
            raise ValueError(_TETM_INCIDENCE_CLASH)
        self._psi_given = True

    def _solve_fn(self, grating_type):
        """The core solve for this instance's ``layer_algorithm`` and the given grating type."""
        try:
            return _SOLVES[self.layer_algorithm][grating_type]
        except KeyError:
            raise ValueError(
                f'layer_algorithm {self.layer_algorithm!r} is not supported '
                f'(expected one of {sorted(_SOLVES)})') from None

    _frozen_geometry = None
    _lattice_basis = None
    _polarization = None
    _tetm = None
    _chi = None
    #: ``None`` is a real state now, not an uninitialised one: it is what a solver whose
    #: polarization was declared away by ``tetm`` reads back. The absence is meant to be
    #: visible — a declared-away degree of freedom should not answer with a number.
    _psi = None

    @property
    def tetm(self):
        """``'te'`` / ``'tm'`` — or ``None``, the default. A *declaration*, not a value.

        It says the incidence has **no polarization and no azimuth degree of freedom**: not
        that they happen to be TE and zero right now, but that neither is a parameter of this
        problem. That is what makes the pure 1D TE/TM formulation legal, and that formulation
        is 2 to 4 times faster than the conical one it replaces (measured across ``truncation_order``;
        they agree to 1e-13).

        The distinction is the reason this is not a second spelling of ``polarization=(1, 0)``.
        Those two describe the same light and give the same numbers -- amplitude for amplitude
        in the same per-order s/p frame, field for field, the same ``excitation`` weights, and
        a scattering matrix that is the conical one's s-s / p-p block -- to ~1e-14 (since
        2026-09-27; before, the amplitudes differed by ``sign(kx_m) * sign(kx_inc)`` per order,
        the incident wave was ``+y`` at every angle, and the TM field was ``mu_inc/n_inc`` of
        the conical p field). They differ in what is left free. ``polarization=(1, 0)`` keeps the polarization a live parameter — you can
        differentiate it, sweep it, optimize it, and you pay conical cost for the privilege.
        ``tetm='te'`` gives that up in exchange for the speed. A declared-away degree of
        freedom is *absent*, so ``psi`` reads back as ``None`` rather than as a number whose
        gradient is quietly zero.

        Legal only where the formulation exists: a 1D structure, ``truncation_order[1] == 0``, no azimuth,
        and no ``psi`` or ``polarization`` beside it. Everywhere else it is refused rather
        than ignored — ``_assign_grating_type`` does that check, and says which condition
        failed. "A 1D structure" means a one-row raster: a shape structure, a tensor material,
        a z row handed to :meth:`solve_conv` and ``formulation='4ff'`` all solve on the 2D
        layout, and a solver holding no structure on the layout its last solve took (2D if
        none); each is refused by name in :meth:`solve_conv` unless that layout is the pure one
        (:meth:`_refuse_tetm_off_the_pure_path`, since 2026-09-27; until then all but the
        tensor raised a ``TypeError`` from deep in the solve, or -- with ``tetm`` assigned after
        construction -- answered for s light whichever it named). A kept pure layout re-checks
        the declaration as the raster that chose it was checked -- ``phi``, ``ky``, the cell's
        skew and ``truncation_order[1]`` -- since the scene may have moved after that solve;
        until the same date it asked nothing, and answered the ``phi = 0`` scene for
        ``phi = 0.4``.
        """
        return self._tetm

    @tetm.setter
    def tetm(self, value):
        """Declare, or withdraw, and leave ``psi`` where the constructor would have.

        The constructor sets ``tetm`` before ``psi`` and falls back to pure s only when nothing
        was declared, so a solver built with ``tetm`` holds ``psi = None``. Assignment has to
        reach that same state, and until 2026-09-27 it did not, in either direction (measured
        on 0937b35). Declaring on a built solver left its default ``psi = pi/2`` behind: ``psi``
        read back a number the declaration says is absent, and on JAX the solver could no
        longer cross a ``jit`` boundary -- the rebuild hands the constructor that ``psi`` beside
        ``tetm`` and is refused for naming two incidences. Withdrawing left ``psi = None`` with
        nothing in its place, so the next solve, now on the conical path, had no polarization
        at all and raised a ``TypeError`` from :meth:`_incidence_pair`. So a declaration clears
        the default and a withdrawal restores it: pure s, what a solver built without ``tetm``
        reads. Nothing else can be lost either way -- while ``tetm`` is set, a ``psi``, ``chi``
        or ``polarization`` is refused rather than stored. The routing follows at the next
        solve: a solver holding no structure keeps its last layout, and a kept pure one is
        re-routed as its raster would be undeclared (:meth:`_assign_grating_type`) -- the
        pure kernel cannot run without the declaration.

        The restored default is built under :meth:`_eager`. Inside ``jax.jit`` it was otherwise
        staged, a tracer, and :meth:`_conditions` puts a slot left holding a tracer back to what
        it held before the call -- ``None`` -- while the concrete ``tetm = None`` stayed: after
        a jitted ``solve(tetm=None)`` the next eager solve raised the same ``TypeError``.
        """
        if value is None:
            if self._tetm is not None:
                with self._eager():
                    self._default_psi()
            self._tetm = None
            return
        if isinstance(value, str):
            key = value.lower()
        elif isinstance(value, Tetm):
            key = value.name.lower()
        else:
            raise ValueError(
                f"tetm must be 'te', 'tm' or None, got {value!r}. It is a declaration, not "
                f'a quantity: there is no fraction between the two, and a linear angle '
                f'between them is psi.')
        if key not in ('te', 'tm'):
            raise ValueError(f"tetm must be 'te', 'tm' or None, got {value!r}")
        if self._psi_given or self._polarization is not None or self._chi is not None:
            raise ValueError(_TETM_INCIDENCE_CLASH)
        self._tetm = key
        self._psi = None

    @property
    def chi(self):
        """The ellipticity angle — the second of the two coordinates, or ``None``.

        ``psi`` alone reaches only the linear states, because it is one coordinate on a set
        that has two. ``chi`` is the other: ``tan(chi)`` is the polarization ellipse's
        minor/major axis ratio, and its sign is the direction the field turns.

        ============  ===============================================
        ``chi = 0``   linear — exactly the ``psi`` spelling
        ``|chi|=pi/4``  circular
        in between    elliptical, at orientation ``psi``
        ============  ===============================================

        Together ``(psi, chi)`` are the canonical coordinates of the polarization state:
        ``2*psi`` and ``2*chi`` are longitude and latitude on the Poincaré sphere, so the
        two of them cover every state exactly once, with no normalization to maintain and
        nothing redundant to optimize over. That last part is why they exist beside
        ``polarization``: the pair is four real numbers of which two are structurally null
        (its overall scale is divided out, and its overall phase cannot move any power), so
        an optimizer over the raw pair spends half its dimensions on directions that cannot
        change the answer. Measured on a 2D grating: ``d(sum T)/d(scale) = 2.4e-16`` and
        ``d/d(phase) = -1.8e-16`` against ``d/d psi = +0.049`` and ``d/d chi = +0.075``.

        The pair keeps what the chart drops — an overall phase — so it stays the normative
        spelling and the two are mutually exclusive, as are ``chi`` and ``tetm``.

        ``chi = 0`` is the ``psi`` spelling exactly — bit-identical, not merely equal. Both
        angle forms are unit by construction, so neither is normalized; see
        :func:`~wavebit._core.conventions.incident_sp_amplitudes`, which is where all three
        spellings become one pair.
        """
        return self._chi

    @chi.setter
    def chi(self, value):
        if value is None:
            self._chi = None
            return
        if self._polarization is not None:
            raise ValueError(_CHI_POLARIZATION_CLASH)
        if self._tetm is not None:
            raise ValueError(_TETM_INCIDENCE_CLASH)
        reject_complex_angle(value, _COMPLEX_CHI)
        self._chi = self._as_real_angle(value)

    def _incidence_pair(self):
        """The incident amplitudes the solves take — resolved here, once, at the boundary.

        Whatever the user wrote becomes the same thing: a unit ``(a_s, a_p)``. The solves are
        handed that and nothing else, so "which spelling was this?" is a question asked in one
        place rather than in each of them.

        ``tetm`` is the exception, and not by omission: it declares that there is no
        polarization to resolve, and the pure 1D path it selects takes ``tetm_index`` instead
        -- and builds the identical ``(1, 0)`` / ``(0, 1)`` incidence in its own variables, each
        lit order along its own ``s_hat`` / ``p_hat`` (``conventions.order_frame_sign``).
        """
        xp = array_namespace(*(self._polarization if self._polarization is not None
                               else (self._psi,)))
        return incident_sp_amplitudes(xp, self.complex_dtype, self._psi,
                                      chi=self._chi, polarization=self._polarization)

    @property
    def tetm_index(self):
        """``0`` for TE, ``1`` for TM — what the pure 1D solves and ``field_1d`` dispatch on.

        The core takes the index rather than the word because it indexes into the two
        formulations rather than naming them; the word is the user's spelling.
        """
        return None if self._tetm is None else int(self._tetm == 'tm')

    def _refuse_y_orders_on_a_y_invariant_cell(self, one_dimensional):
        """A one-row raster cannot diffract out of plane, so it cannot keep ``y`` orders.

        Both 1D formulations retain the single ``n = 0`` order, which is not a limitation but
        the physics: with nothing varying in ``y`` there is no grating vector to couple
        different ``n``, and the orders being asked for are identically zero. Measured on the
        2D path, which does retain them, for the same structure written with two identical
        rows: the ``n = ±1`` rows sum to 5e-29, and the ``n = 0`` row reproduces
        ``truncation_order=[m, 0]`` to 3e-15.

        Until this check the request reached ``layer_modes_1d_conical``, where ``Ky2`` is built
        over the full ``n_orders_u * n_orders_v`` order set and the convolution matrices over
        ``n_orders_u`` alone, and surfaced as ``operands could not be broadcast together with
        shapes (5,5) (15,15)`` — an error with nothing in it that names the mistake.
        """
        if not one_dimensional or self.truncation_order[1] == 0:
            return
        raise ValueError(
            f'truncation_order={list(self.truncation_order)} asks for y diffraction orders, but this '
            f'structure has one row in y, so the structure is invariant along it and every '
            f'n != 0 order is identically zero. Both 1D formulations keep the single n = 0 '
            f'order for that reason. Use truncation_order=[{self.truncation_order[0]}, 0]; if you meant a '
            f'structure that varies in y, '
            f'give a structure with more than one row (two identical rows is the 2D path solving '
            f'the same physics, and it returns those zeros explicitly).')

    def _refuse_illegal_tetm(self, one_dimensional, one_row=None):
        """``tetm`` is a claim about the problem; where it cannot hold, say which part failed.

        Called from ``_assign_grating_type``, because that is where the structure is known.
        The old ``pol`` had no such check: on a 2D structure it was accepted and then ignored,
        and the fast path it was reaching for silently did not happen.

        ``one_row`` separates the two ways the claim fails, because they need different
        advice. A 2D cell has no pure TE/TM at all. A **one-row raster on a skewed cell** is a
        1D grating, but its grooves run along ``a2`` rather than along y, so ``E`` along y --
        which is what ``tetm`` names -- is a *mixture* of that grating's two decoupled
        polarizations rather than one of them. Telling that caller "this is not a 1D
        structure" would be answering a question they did not ask.

        The ``truncation_order[1]`` half lives in :meth:`_refuse_y_orders_on_a_y_invariant_cell`
        instead — it is not a fact about the declaration but about the structure, and it
        applies to every spelling of the incidence.
        """
        if self._tetm is None:
            return
        if self.ky is not None and _concrete(self.ky == 0) is not True:
            raise ValueError(
                'tetm requires ky=0; omit ky for a pure 1D TE/TM solve. '
                'Use polarization for a nonzero or traced ky on the conical path.')
        if not one_dimensional:
            if one_row:
                raise ValueError(
                    'tetm was given on a one-row raster with a skewed lattice. That is a 1D '
                    'grating, but its grooves run along a2, not along y -- so the pure TE/TM '
                    'pair tetm names is not this grating\'s pair, and E along y is a mixture '
                    'of both. Give the incidence as psi (linear) or '
                    'polarization=(a_s, a_p), which carries the mixture; the solve then takes '
                    'the 2D path, which is the one whose k layout a skewed cell has.')
            raise ValueError(
                'tetm is only for a 1D structure: the pure TE/TM formulation it selects does '
                'not exist on a 2D cell, where s and p couple. Drop tetm and give the '
                'incidence as psi (linear) or polarization=(a_s, a_p).')
        if self.phi is not None:
            import numpy as np
            try:
                off_axis = complex(np.asarray(self.phi)) != 0
            except (TypeError, ValueError):
                # A traced azimuth is a parameter being differentiated or swept, which is
                # precisely what the declaration says this problem does not have.
                off_axis = True
            if off_axis:
                raise ValueError(
                    'tetm and a nonzero phi both given: tetm declares that the azimuth is '
                    'not a degree of freedom of this problem, so an off-axis plane of '
                    'incidence contradicts it. Drop tetm for the conical path, which is '
                    'what carries an azimuth.')

    def _refuse_tetm_off_the_pure_path(self, z_row_given, routed):
        """``tetm`` on a solve that is not taking the pure 1D path: name what sent it away.

        :meth:`_refuse_illegal_tetm` answers for a raster, where the structure alone decides.
        Five other routes leave the pure path with ``tetm`` still set. A z row handed to
        :meth:`solve_conv` (the permittivity's or the permeability's) and ``formulation='4ff'``
        move any structure onto the 2D layout the first-order path is written on; a tensor
        material and a shape structure are always solved there; and a solver holding no
        structure solves on the layout its last solve took, or the 2D one if it never solved --
        so it is refused unless that solve was a pure one. The tensor route is refused by name
        earlier, where the convolution matrices are built
        (:func:`~wavebit._core.anisotropy.validate_options`), and is here for matrices built
        before ``tetm`` was set. The other four reached :meth:`_incidence_pair`, and what they
        did there depended on the door ``tetm`` came in by. Declared at construction it leaves
        ``psi`` unset, and the solve raised ``TypeError: array_namespace requires at least one
        non-scalar array input``, naming neither input. Assigned afterwards it left the
        constructor's default ``psi = pi/2`` behind, and the solve *answered* -- for s light,
        whichever of ``'te'`` / ``'tm'`` had been declared (measured on 0937b35: ``tetm='tm'``
        under ``formulation='4ff'`` returned the TE reflectance).

        ``z_row_given`` is :func:`_z_row_given` of the call, and ``routed`` the layout the
        structure chose before a z row or ``'4ff'`` moved it. Where two routes meet, the message
        names the one whose fix is complete: the z row first, since dropping ``tetm`` is the
        whole fix whatever else holds; ``'4ff'`` only where the structure took the pure path,
        since only there does "leave formulation at 'auto'" keep it; then the structure's own
        reason. Asked before the structure, ``'4ff'`` told a shape, and a solver with no
        structure, that ``'auto'`` would keep a pure path neither has. Every test is structural
        -- was a slot given, which formulation was set, which type the structure is -- so the
        refusal holds under ``jax.jit`` and ``jax.grad`` exactly as it does eagerly.
        """
        tetm = self._tetm
        instead = (f"polarization={'(1, 0)' if tetm == 'te' else '(0, 1)'} at phi=0, the "
                   f'light tetm={tetm!r} describes')
        if z_row_given:
            raise ValueError(
                f"tetm={tetm!r} and a z row ({', '.join(z_row_given)}) were both given. A z "
                'row couples E_z to the in-plane field, which only the first-order (4ff) path '
                'expresses, and that path solves on the 2D layout with both polarizations: '
                'the pure TE/TM formulation tetm selects has no slot for it. Drop tetm and give '
                f'the incidence as {instead}; the solve then takes the 4ff path. See '
                'docs/anisotropic-materials.md section 8.')
        if routed == 0 and self._formulation == Formulation.FOUR_FIELD:
            raise ValueError(
                f"tetm={tetm!r} and formulation='4ff' were both given. '4ff' forces the "
                'first-order path, which solves on the 2D layout with both polarizations, so '
                'the pure TE/TM solve tetm selects cannot run on it. To cross-check the two '
                f'formulations on this structure, drop tetm and give the incidence as '
                f"{instead}; to keep the pure 1D path, leave formulation at 'auto'.")
        if self.structure is None:
            layout = 'the conical 1D one' if routed == 1 else 'the 2D one'
            raise ValueError(
                f'tetm={tetm!r} was given to a solver that holds no structure. Only a 1D '
                'raster routes a solve to the pure TE/TM path tetm selects; with no structure, '
                'solve_conv keeps the layout the last solve took, or the 2D one if there was '
                f'none -- here {layout}, which carries both polarizations. Assign the raster '
                'the convolution matrices were built from, or drop tetm and give the incidence '
                f'as {instead}.')
        if self._has_tensor_materials():
            raise ValueError(_anisotropy.TETM_REFUSAL)
        if self.modeling_path == ModelingPath.PARAMETRIC:
            raise ValueError(
                f'tetm={tetm!r} was given with a shape structure. Shapes are always solved on '
                'the 2D layout, where s and p couple, even when no shape varies along y -- so '
                'the pure TE/TM formulation tetm selects does not exist there. Drop tetm and '
                f'give the incidence as {instead}, or draw the grating as a one-row raster to '
                'keep the pure 1D path.')
        # Unreachable today: a raster that is not 1D is refused by `_refuse_illegal_tetm`.
        # Kept so that a sixth route to the 2D layout meets a sentence rather than the
        # TypeError the five above used to.
        raise ValueError(
            f'tetm={tetm!r} cannot be honoured: this solve is not on the pure 1D TE/TM path. '
            f'Drop tetm and give the incidence as {instead}.')

    #: Whether ``psi`` was set explicitly, rather than being the pure-s default a solver
    #: falls back to when no incidence was named. It is what lets ``polarization`` refuse to
    #: coexist with a *given* ``psi`` without also refusing that default.
    _psi_given = False

    @property
    def polarization(self):
        """The incident polarization as a complex amplitude pair ``(a_s, a_p)`` — or ``None``.

        The pair lives in the same ``(s, p)`` basis as ``psi`` and the returned amplitudes:
        ``(1, 0)`` is pure s (``psi = pi/2``), ``(0, 1)`` pure p (``psi = 0``), and
        ``psi = v`` is exactly ``(sin(v), cos(v))``. What the pair adds is the relative
        phase ``psi`` cannot spell — ``(1, -1j)/sqrt(2)`` and ``(1, +1j)/sqrt(2)`` are the
        circular states. The overall magnitude is divided out (efficiencies are per unit
        incident flux, amplitudes per unit incident E-field); the overall phase is kept and
        carries into the complex amplitudes. See
        :func:`~wavebit._core.conventions.incident_sp_amplitudes`.

        ``polarization`` and an explicit ``psi`` are mutually exclusive — setting both is an
        error, not an override. A 1D structure given a pair is solved on the conical path,
        which carries both polarizations; the pure 1D TE/TM path is single-polarization by
        construction.
        """
        return self._polarization

    @polarization.setter
    def polarization(self, polarization):
        if polarization is None:
            self._polarization = None
            return
        if self._psi_given:
            raise ValueError(_PSI_POLARIZATION_CLASH)
        if self._chi is not None:
            raise ValueError(_CHI_POLARIZATION_CLASH)
        if self._tetm is not None:
            raise ValueError(_TETM_INCIDENCE_CLASH)
        pair = tuple(polarization)
        if len(pair) != 2:
            raise ValueError(
                f'polarization must be a pair (a_s, a_p) of complex amplitudes, '
                f'got {len(pair)} value(s)')
        try:
            if complex(pair[0]) == 0 and complex(pair[1]) == 0:
                raise ValueError(
                    'polarization = (0, 0) carries no incident field; give at least one '
                    'nonzero amplitude (the pair is normalized to unit norm, so its overall '
                    'scale is free)')
        except TypeError:
            pass                      # traced values: concreteness is not theirs to prove
        # Cast on the way in. The pair is a pytree child, and JAX refuses to differentiate a
        # tree with an integer leaf -- so `polarization=(1, -1j)`, the most natural way to
        # write a circular state, used to raise `grad requires real- or complex-valued
        # inputs` while `(1.0+0j, -1j)` worked. A dtype is not something the caller should
        # have to know to get a gradient.
        self._polarization = tuple(self._as_complex(v) for v in pair)

    @property
    def lattice(self):
        """The unit cell's basis, always — :class:`~wavebit._core.lattice.Lattice`, never ``None``.

        A cell given as ``period`` materialises here as ``diag(period)``, so there is one place
        to ask what the cell is and one type to get back. It returned ``None`` for the
        axis-aligned default until 2026-08-23, on the stated grounds that "a basis has to be
        concrete (it decides control flow and reaches SciPy)" and ``period`` may be a tracer.
        That stopped being true when ``Lattice`` split its numbers from its shape: the numbers
        are pytree children and trace, while ``axis_aligned`` -- the part that decides control
        flow -- is aux_data and is known from *which keyword was used* without reading one.

        The `None` was load-bearing in the meantime, which is how you can tell it cost
        something: :func:`~wavebit._core.draw.slice_structure` reaches past this property to
        ``_geometry_lattice_for`` precisely to get the materialised basis, "instead of a
        lookalike".

        Assigning a basis selects the non-orthogonal path; ``period`` then reports the
        basis-vector lengths (see :func:`wavebit._core.lattice.periods_of`), and the raster
        is read in lattice coordinates — ``structure[layer][j, i]`` sampling
        ``r = (i+0.5)/Nx * a1 + (j+0.5)/Ny * a2``.
        """
        if self._lattice_basis is not None:
            return self._lattice_basis
        return _lattice.as_lattice(period=self.period)

    @lattice.setter
    def lattice(self, value):
        if value is None:
            self._lattice_basis = None
            return
        A = _lattice.as_lattice(lattice=value)
        self._cell_named = True          # an assignment names it, as a keyword does
        self._lattice_basis = A
        self._set_period_slot(_lattice.periods_of(A))   # derived, not chosen

    #: Whether a cell was ever *named* by the caller. ``True`` here so that anything built
    #: by a route other than the constructor is not accused of omitting one; the constructor
    #: sets the real answer.
    _cell_named = True

    #: The basis a ``Lattice.drawn(...)`` structure declared, or ``None`` for a bare array.
    #: A bare array is the lattice frame and is portable by definition, so ``None`` is not a
    #: missing value -- it is the other answer.
    _structure_drawn_on = None

    @property
    def structure_drawn_on(self):
        """The cell this structure declared through :meth:`~wavebit._core.lattice.Lattice.drawn`.

        ``None`` for a bare array, and that is an answer rather than a gap: a bare array is the
        *lattice* frame, portable across cells by construction, so there is nothing for it to
        have declared. Read-only -- the declaration is made where the structure is built.
        """
        return self._structure_drawn_on

    def _refuse_a_raster_drawn_on_another_cell(self):
        """A structure that named its cell may not be solved on a different one.

        This is the Cartesian frame's invariant and the only one either frame has. A raster is
        legal on any basis, so nothing downstream can notice the mismatch: measured, a
        radius-100 circle drawn on a 640 square and solved on ``hexagonal(640.)`` gives
        ``sum(T) = 0.252202`` against the 0.352836 a redrawn array gives -- silent, large, and
        indistinguishable from the lattice effect the caller was asking for.

        Only a declared structure is checked. A bare array stays accepted on every cell,
        because that is the *lattice* frame rather than an undeclared Cartesian one, and it is
        what every pre-existing caller passes.

        Both sides are read for a value: a ``jax.grad`` cell by its primal, and a ``jax.jit``
        one not at all. On the JAX solver the cell and the declaration are pytree children, so
        inside a trace they are traced; the flatten that started the trace compared them while
        they were concrete (``SolverJax._declaration_checked_for_the_trace``), and a rebuilt copy
        trusts that for as long as it holds the same objects. A cell or declaration assigned
        inside a ``jax.jit`` trace was compared by nothing and is refused, by name -- the one
        case this refuses that answered while the declaration rode as static aux data, when a
        concrete cell assigned in the trace could still be read (``docs/oblique-lattice.md``).
        """
        if self._structure_drawn_on is None:
            return
        # Both sides are read the same way, and both before anything else: on the JAX solver
        # the declaration is a pytree child beside the cell, so in a rebuilt copy either can be
        # a tracer, and `np.asarray` on one raises rather than answering `None`.
        drawn, here = _concrete_2x2(self._structure_drawn_on), self._read_cell()
        if drawn is None or here is None:
            # A side with no number -- traced under `jax.jit` or `jax.vmap`. The JAX solver's
            # cell and declaration are pytree children, so they are traced in every rebuilt
            # copy; the flatten that built this copy compared them while they were concrete,
            # and nothing needs asking again while the copy holds those same objects
            # (`_cell_checked_when_flattened`). Anything else traced has been compared by
            # nothing, and a moved one here is the silent answer this check exists to stop.
            if self._cell_checked_when_flattened():
                return
            side = ('the cell it is being solved on' if here is None else
                    'the cell it was drawn on (the solver itself is an argument of the trace, '
                    'so its declaration is a traced leaf)')
            raise ValueError(
                f'this structure was declared with Lattice.drawn(...), and {side} has no value '
                'here -- it is traced, under jax.jit or jax.vmap -- so the two cannot be '
                'compared, and the comparison is the only thing that catches a raster solved '
                'on a cell it was not drawn on. Differentiate the cell under jax.grad, where it '
                'has a value; or keep the cell concrete and trace the rest, as a jitted '
                'solve_field(..., geometry=...) on a declared solver does, assigning the cell '
                'before the trace rather than inside it; or pass the bare array, the lattice '
                'frame, which declares no cell.')
        if _np.allclose(drawn, here, rtol=1e-10, atol=1e-10):
            return
        raise ValueError(
            'this structure was drawn on a different cell than the one it is being solved '
            'on.\n'
            f'    drawn on : {drawn.tolist()}\n'
            f'    solved on: {here.tolist()}\n'
            'A raster holds pixel indices, so it is legal on any basis and nothing further '
            'down can catch this -- the answer moves, and it moves the way a lattice effect '
            'would. Redraw it on the cell you are solving:\n'
            '    X, Y = lattice.meshgrid_xy(grid_shape)\n'
            '    structure = lattice.drawn(...)\n'
            'Or, if the pattern is meant as fractions of the cell rather than as lengths, it '
            'is in the lattice frame: build it from `meshgrid_uv` and pass the bare array, '
            'which is portable across cells by definition.')

    def _read_cell(self):
        """The cell as a concrete ``(2, 2)`` array for a check, or ``None`` when it has no value.

        Read through :func:`~wavebit._core.backend.checked_float`, so a ``jax.grad`` tracer is
        read by its primal -- the checked side of the line every geometry check here draws --
        and a ``jax.jit`` or ``jax.vmap`` one is ``None``. ``np.asarray`` refused both, which
        refused every gradient through the cell of a declared structure.
        """
        return _concrete_2x2(self.lattice)

    def _cell_objects(self):
        """The slots the drawn-cell check compares, as objects: what a rebuilt copy was checked
        against -- the two a cell is held in, and the declaration beside them."""
        return (self.__dict__.get('_lattice_basis'), self.__dict__.get('_period'),
                self.__dict__.get('_structure_drawn_on'))

    def _cell_checked_when_flattened(self):
        """Whether this is a rebuilt JAX copy still holding the cell and declaration its flatten
        compared.

        ``SolverJax._tree_flatten`` compares a declared structure against the concrete cell and
        ``_tree_unflatten`` records the objects it rebuilt; a cell or structure assigned inside
        the trace replaces one, and then this is ``False`` and the check refuses a traced side.
        The declaration is in the tuple because it is a child too: with only the cell there, a
        structure redeclared on another cell inside the trace was trusted and answered.
        """
        held = self.__dict__.get('_drawn_checked_against')
        return held is not None and all(a is b for a, b in zip(held, self._cell_objects()))

    def _refuse_a_structure_with_no_cell(self):
        """A structure to solve, and no cell ever stated — refuse rather than assume ``(1, 1)``.

        ``period=(1., 1.)`` is a real cell and the coherent partner of ``wavelength=1.``, so
        the default cannot simply be dropped: 35 call sites in this repository rely on it, all
        of them building a solver with **no structure** to exercise the spec plumbing. What is
        not defensible is solving a *structure* on a cell nobody chose. A raster carries no
        length of its own, so nothing downstream can notice: measured on a hex scene whose
        ``lattice=`` was forgotten, ``sum(T)`` came back **0.989790** against the correct
        **0.927664**, with no error and energy conserved.

        The parametric path is already safe here and this does not duplicate it — a shape's
        lengths are absolute, so the same omission raises "shape is larger than the period".
        The raster was the only silent one.

        :func:`wavebit.meshgrid_xy` has refused exactly this since it was written
        (``as_lattice: needs one of period or lattice``), so the two doors into a raster now
        say the same thing.
        """
        if self._cell_named or self._structure is None:
            return
        raise ValueError(
            'no cell was given: this solver has a structure to solve but neither `period` '
            'nor `lattice` was named, so the cell would silently be the (1, 1) default -- '
            'which is the normalized-units cell, not a guess at yours. State one:\n'
            '    period=[700., 1100.]                 # a rectangular cell\n'
            '    lattice=wavebit.Lattice.hexagonal(400.)      # or oblique(...) / a 2x2\n'
            'A raster carries no length of its own, so nothing further down can catch this: '
            'measured on a hex scene with `lattice=` forgotten, sum(T) read 0.989790 against '
            'the correct 0.927664, energy conserving throughout. `wavebit.meshgrid_xy` has always '
            'refused the same omission.')

    @property
    def lattice_is_axis_aligned(self):
        """Whether the cell is ``diag(Lx, Ly)`` — i.e. whether the separable k-layout applies."""
        return self._lattice_basis is None or _lattice.is_axis_aligned(self._lattice_basis)

    def _geometry_lattice_for(self, geometry):
        """The cell basis the *geometry* steps should use — live, or recovered from the record.

        One accessor rather than two, because a skewed basis and a non-square pixel are the
        same question asked twice: both ask what one raster step means in real space, and a 2x2
        answers it in either case. An axis-aligned cell gets ``diag(period)`` — it is *not*
        withheld, because the metric it enables fixes a real defect there too (a raster whose
        pixels are not square in real space had its normal up to ~18 degrees off; see
        :func:`~wavebit._core.factorization.normal_vector_field`).

        ``period`` is readable except when it is the variable under differentiation, which is
        exactly when a caller is most likely to also be replaying a frozen geometry. Freezing
        happens eagerly on a concrete ``period``, so the record was built on a real pixel shape;
        letting an unreadable one arrive as ``None`` would make the same structure look like a
        different raster and the replay would be refused
        (:func:`~wavebit._core.factorization._check_normal_field_still_fits`). Taking it back
        out of the record rather than storing it twice keeps the two in step, and is the honest
        answer besides — the field being replayed *is* the one built on that pixel shape,
        whatever ``period`` has since become.
        """
        if not self.lattice_is_axis_aligned:
            return self._lattice_basis
        period = _concrete_period(self.period)
        if period is not None:
            return _lattice.as_lattice(period=period)
        if geometry is not None:
            recorded = _recorded_lattice(geometry.normal_field_source)
            if recorded is not None:
                return recorded
        # A traced `period` and no record. This used to return `None`, and `None` downstream is
        # not "unknown" -- it is the identity metric, i.e. *square pixels*. So a `jax.grad` over
        # `period` on a raster whose pixels are not square silently solved a different structure:
        # measured on a 200 x 400 cell at 32 x 32, the value inside `value_and_grad` was
        # 0.962499142191 against an eager 0.964225033797, and bit-identical to what this method
        # returning `None` gives. Not a missing derivative -- a wrong forward answer, and the
        # derivative of the wrong function.
        #
        # `as_lattice` has taken a traced `period` since the basis became a design variable: the
        # shape is still diagonal, so `axis_aligned` is known without reading a number, which is
        # exactly why that flag is aux_data. Nothing here needed building; this line needed
        # calling.
        return _lattice.as_lattice(period=self.period)

    # ---------------------------------------------------------------- the incidence angles
    #
    # One copy. These were three, differing only in how a value became an array: `np.array`,
    # `jnp.array`, and a local `_as_tensor` that was a second copy of `_core.backend.as_array`.
    # `as_array` is what all three became, for the two reasons that hold on this repo's floor:
    # it normalises strides, which Torch needs and the other two do not, and it honours
    # `device=` for a value already in the namespace. Not for autograd -- `torch.asarray` did
    # flip a leaf's `requires_grad` off in place, but pytorch#170897 changed that in 2.12.0 and
    # the floor here is 2.13. For a scalar angle the two spellings are measurably equivalent
    # today, which is why a mutation swapping them survives the suite: a fact about the floor,
    # not a gap in the tests.
    #
    # Torch's `psi` setter also checked `torch.is_tensor` before taking a real part where the
    # other two called `np.real` unconditionally. That divergence dies here: `xp.real` is the
    # namespace's own, and it is correct for all three.

    @property
    def theta(self):
        """Phase polar angle in radians, measured from the inward surface normal.

        Real ``theta`` and ``phi`` specify uniform incident intensity along the
        surface, including in absorbing media. Complex angles instead retain
        the formal ``n*sin(theta)`` parameterization for advanced illumination.
        """
        return self._theta

    @theta.setter
    def theta(self, theta):
        if theta is not None and (self.kx is not None or self.ky is not None):
            raise ValueError(_DIRECTION_CLASH)
        if theta is None:
            self._theta = None
        else:
            check_input_precision('theta', theta, self.complex_dtype)
            self._theta = self._as_complex(theta)

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi is not None and (self.kx is not None or self.ky is not None):
            raise ValueError(_DIRECTION_CLASH)
        if phi is None:
            self._phi = None
        else:
            check_input_precision('phi', phi, self.complex_dtype)
            self._phi = self._as_complex(phi)

    _kx = None
    _ky = None

    def _set_transverse_component(self, name, value):
        if value is None:
            setattr(self, '_' + name, None)
            return
        if self.theta is not None or self.phi is not None:
            raise ValueError(_DIRECTION_CLASH)
        problem = transverse_component_problem(value, name)
        if problem is not None:
            raise ValueError(problem)
        check_input_precision(name, value, self.complex_dtype)
        xp = self.xp
        candidate = self._as_complex(value)
        valid = xp.isfinite(candidate) & (xp.imag(candidate) == 0)
        real = xp.real(candidate)
        # Invalid traced values must not silently become a different real wave.
        setattr(self, '_' + name, xp.where(valid, real, xp.full_like(real, float('nan'))))

    @property
    def kx(self):
        """Incident Cartesian x wavevector in inverse length units, or ``None``.

        Direct ``kx``/``ky`` input replaces ``theta``/``phi``. An omitted component
        is zero. Components are finite real scalars, expressed in the fixed xyz
        frame, including on oblique lattices and for bottom incidence. For lengths
        in nm, these inputs are in nm**-1 (angular wavenumber, not cycles per nm).
        At kx=ky=0 the s/p basis uses azimuth zero: s along +y and p in xz.
        ``order_kxy()`` instead returns components normalized by k0=2*pi/wavelength.
        """
        return self._kx

    @kx.setter
    def kx(self, value):
        self._set_transverse_component('kx', value)

    @property
    def ky(self):
        """Incident Cartesian y wavevector in inverse length units; see :attr:`kx`."""
        return self._ky

    @ky.setter
    def ky(self, value):
        self._set_transverse_component('ky', value)

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        """Refuse before recording, because the record outlives the refusal.

        This read ``_psi_set_explicitly()`` first, and that call *latches* ``_psi_given``. So a
        ``psi`` refused for being complex left the angle at whatever it was and the flag at
        True -- and ``_psi_given`` is what the ``polarization`` setter reads to decide the two
        spellings clash. A caller who tried a complex ``psi``, caught the refusal, and then
        reached for ``polarization=(1, -1j)`` -- which is the pair that *does* express the
        state they were after, and what the refusal message points them to -- was told
        "psi and polarization both given". The refusal had claimed a spelling nobody used.

        Order now: reject, build, record, write. Every raising step precedes every write, and
        the only statement after the latch is an assignment that cannot fail.
        ``_psi_set_explicitly`` still does the clash checks itself, so those keep raising
        before it latches. :attr:`chi` had this order all along, in the same file.
        """
        if psi is not None:
            reject_complex_angle(psi)
            value = self._as_real_angle(psi)
            self._psi_set_explicitly()
            self._psi = value

    @property
    def device(self):
        """Where this solver's arrays live, in whatever this backend calls a device.

        The three spell it differently and the difference is real, so it is reported rather
        than normalised: NumPy answers ``0`` because it has one device and does not name it,
        JAX answers a *list* of :class:`jax.Device` (which is what ``jax.default_device``
        takes), and Torch answers a :class:`torch.device`. Assigning
        ``solver.device = solver.device`` round-trips on all three.

        Accepts wavebit's own codes -- ``0`` / ``'cpu'`` and ``1`` / ``'gpu'`` / ``'cuda'`` --
        as well as the backend's own device object. The codes are wavebit's, not device
        indices: handing ``0`` straight to Torch selects device *index* 0, which on a Mac is
        MPS and refuses float64.

        **On JAX this is where the solve runs, not only what is reported.** Every array the
        solver holds is committed to this device and every public entry point opens its scope,
        so the solve, its convolution matrices, its field and the library's own `grad` / `fit`
        run here whatever JAX's default device is. Until 2026-09-11 this getter was the one
        thing that honoured the request: on a CUDA jaxlib `device='cpu'` -- and the default,
        which is the CPU -- ran on the GPU while this read `[CpuDevice(id=0)]`, since it reports
        the request, not the placement. A transformation the caller writes *around* a solve
        stays the caller's: `jax.grad` builds its seed and a few zeros on JAX's default device
        (58 small GPU allocations around a CPU solve that made none, measured), and a gradient
        comes back on the device of the input it was taken with respect to.

        **Assigning it moves the solver**, on JAX and Torch alike: every array the solver holds
        -- the ones its setters built and the ones a caller handed in -- goes to the new device
        (:meth:`_move_to_device`), so a solver moved to ``X`` answers what one built on ``X``
        answers, bit for bit. ``solve(device=X)`` moves it too, and like any concrete condition
        it stays moved -- inside ``jax.jit`` and ``jax.grad`` as well, where the move is carried
        out when the trace runs rather than staged into it. A field reconstruction from a solve taken before the move is refused as
        a changed scene; solve again first.
        """
        return self._device

    @device.setter
    def device(self, value):
        new = self._resolve_device(value)
        # The constructor's assignment is the first and has nothing to move. Read from the
        # instance, so a class attribute could never pass for a device this solver was on.
        moving = '_device' in self.__dict__ and self._device != new
        self._device = new
        if moving:
            self._move_to_device()

    def _move_to_device(self):
        """Put every array this solver holds on its device -- what assigning ``device`` means.

        The setters commit what they store to the device current when they run, and a
        committed operand outranks the device scope. So a device assigned *after* them used to
        leave every stored array behind: measured 2026-09-11 on two host devices, a solver
        built on ``cpu:1`` and moved to ``cpu:0`` raised ``Received incompatible devices`` on
        its next solve (a new-device order range against the old-device period), by the
        attribute and by ``solve(device=...)`` alike. Before the arrays were committed at all,
        the same call answered -- on JAX's default device, whatever ``device`` said.

        The instance is walked rather than its slots listed, for the reason
        :meth:`_conditions` gives: a list goes stale on the first new slot. The walk enters
        the containers :func:`_holds_tracer` enters -- dicts, lists, tuples (the solve's
        :class:`~wavebit._core.scattering.ScatteringMatrix` among them), a
        :class:`~wavebit._core.lattice.Lattice`'s two vectors and a
        :class:`~wavebit._core.domain.Material`'s two slots -- and moves each native array
        with :meth:`_placed`, so a moved solver holds what one built on the new device would.
        Containers are rebuilt, never mutated: a caller may hold the same list or dict.

        Inside a trace the walk runs under :meth:`_eager`, so a concrete slot is moved for real
        rather than staged. Staged, every slot came back a tracer: measured 2026-09-11, a jitted
        ``solve(device=X)`` on a TE solver at ``phi = 0`` was refused as "tetm and a nonzero
        phi", because the azimuth check could no longer read ``phi``. A tracer is moved too --
        a keyword's value that arrived before ``device`` in the same call, which its setter
        placed on the old device. Under ``jax.grad`` that move is eager and keeps the
        derivative; under ``jax.jit`` it is a transfer in the program. Left where it was,
        ``jax.grad`` of ``solve(wavelength=w, device=X)`` raised ``Received incompatible
        devices`` while the reverse keyword order answered.

        On Torch the move is ``.to()``: differentiable, but a new tensor, so an optimizer that
        holds the old leaf no longer reaches the solver -- as with any solver rebuilt on
        another device.
        """
        with self._eager():
            for slot, value in list(vars(self).items()):
                moved = self._moved(value)
                if moved is not value:
                    object.__setattr__(self, slot, moved)

    def _moved(self, value):
        """``value`` with every native array in it on this solver's device, or ``value`` itself.

        Itself -- the same object -- when nothing in it moved, so a slot already in place is not
        rebuilt and the walk costs nothing on a solver that never changes device.
        """
        if isinstance(value, dict):
            items = {k: self._moved(v) for k, v in value.items()}
            return value if all(items[k] is value[k] for k in value) else type(value)(items)
        if isinstance(value, (list, tuple)):
            items = [self._moved(v) for v in value]
            if all(a is b for a, b in zip(items, value)):
                return value
            if hasattr(value, '_fields'):                  # a NamedTuple
                return type(value)(*items)
            return type(value)(items)
        if isinstance(value, _lattice.Lattice):
            u, v = self._moved(value.u), self._moved(value.v)
            if u is value.u and v is value.v:
                return value
            import dataclasses
            return dataclasses.replace(value, u=u, v=v)
        return self._placed(value)       # a native array or a Material; anything else is itself

    def _resolve_device(self, value):
        """A device request as this backend's own device object. One device here.

        NumPy names no device and places nothing, so a CPU request is the identity and a GPU
        request is refused. It used to `print` -- for *every* assignment, including
        ``device = 0``, which is what it already was -- and store nothing. A print is not a
        refusal: it does not stop the caller, it does not reach a log, and in a notebook it
        lands somewhere nobody reads. A solver that reports `0` after being asked for a GPU
        has answered the question wrongly and said so where it cannot be heard.
        """
        if value is None or value in (0, 'cpu'):
            return 0
        raise ValueError(
            f'device: the NumPy backend has one device and it is the CPU; got {value!r}. '
            f'Use backend=1 (JAX) or backend=2 (Torch) for anything else.')

    #: Where the setters place what they build -- `_as_complex`, `_as_real_angle`,
    #: `_as_lengths`, `_coerce_raster`, `_separable_kxy`'s order ranges -- and where `_placed`
    #: moves what the caller hands in. `None` on NumPy,
    #: which has no device to name. Torch and JAX override it with the solver's device: Torch
    #: because a tensor is placed at creation, JAX because a committed array is the one
    #: placement a later computation cannot overrule (it was `None` there too until
    #: 2026-09-11, and a `device='cpu'` solve ran on the GPU -- see the JAX override). Declared
    #: for the same reason `_device_scope` is: a wrong placement computes nearly the right
    #: numbers in the wrong place, and a CPU-only machine cannot tell.
    _array_device = None

    def _as_complex(self, value):
        """A user-supplied value as a complex array of this solver's width."""
        return as_array(self.xp, value, dtype=self.complex_dtype, device=self._array_device)

    def _as_real_angle(self, value):
        """The real part, as a real array. `psi` and `chi` are angles, not complex numbers."""
        xp = self.xp
        real = xp.real(value) if hasattr(value, 'dtype') else _np.real(value)
        return as_array(xp, real, dtype=self.float_dtype, device=self._array_device)

    def _placed(self, value):
        """``value`` on this solver's device if it is one of this backend's arrays, or a
        :class:`~wavebit._core.domain.Material` holding one. Anything else comes back as itself.

        The setters above place what they build, but some values reach a solve as the caller
        wrote them: a Material's permittivity and permeability, a shape's coordinates, an
        excitation amplitude. A native array among them on another device -- one the caller
        built on JAX's default device, or that `jax.grad` built there for them -- was
        *followed*: the shape builder places its order grid at the background's `.device`,
        which committed it to the GPU under a CPU solver, and the build raised
        `Received incompatible devices for jitted computation` (measured 2026-09-11,
        `test_analytic_shape_gradients[background-jax]` on a CUDA box).

        A move, not a conversion: the dtype is kept and the move is differentiable. It is the
        identity on NumPy, for every Python number, and for an array already where it belongs,
        so a scene that holds no foreign array reaches the builders object for object as before.
        """
        if self._array_device is None:
            return value
        if isinstance(value, Material):
            epsilon = self._placed(value.epsilon)
            mu = None if value.mu is None else self._placed(value.mu)
            if epsilon is value.epsilon and mu is value.mu:
                return value
            return Material(epsilon, mu)
        if isinstance(value, self._ARRAY_TYPES):
            return as_array(self.xp, value, device=self._array_device)
        return value

    def _default_psi(self):
        """The unstated polarization: pure s, represented by ``psi = pi/2``.

        Set rather than defaulted-at-read because `psi` is a traced leaf: it has to exist as an
        array before anything differentiates through it, and `_psi_given` -- not its value --
        is what records whether the caller named it.
        """
        self._psi = self._as_real_angle(_np.pi / 2)

    # ------------------------------------------------------------------- the precision
    #
    # One copy. The width resolution was written *seven* times: a `complex_dtype` setter and an
    # inline block in each backend's `_BaseRCWA.__init__`, plus a third inside
    # `SolverTorch.__init__`. No per-backend table is needed for any of it -- every Array-API
    # namespace exposes `complex128`, `complex64`, `float64`, `float32`, `int64` and `int32`,
    # and on all three they are *the same objects* as the native spellings (measured), so
    # `xp.complex128` under Torch is `torch.complex128` itself.

    def _resolve_precision(self, value):
        """``(complex, float, int)`` dtypes for a requested width, in this solver's namespace.

        ``0`` or a complex128 spelling asks for double, ``1`` or a complex64 spelling for
        single. Both the namespace's own spelling and NumPy's are accepted on every backend,
        which is what the copies did between them.

        The three are returned **together** rather than derived from the first afterwards.
        The copies derived them with ``is not``, and that is a live bug: for a ``np.dtype``
        instance ``np.dtype('complex64') == np.complex64`` is True while
        ``np.dtype('complex64') is np.complex64`` is False, so NumPy's setter accepted the
        spelling, selected complex64 arithmetic, and paired it with ``float64`` and ``int64``.
        A mixed width that nothing downstream could see.
        """
        xp = self.xp
        if value in (0, xp.complex128, _np.complex128):
            return xp.complex128, xp.float64, xp.int64
        if value in (1, xp.complex64, _np.complex64):
            return xp.complex64, xp.float32, xp.int32
        raise ValueError(
            f'complex_dtype: 0 or 1, or a complex128 / complex64 dtype -- got {value!r}. '
            f'0 and complex128 are double precision, 1 and complex64 single.')

    @property
    def complex_dtype(self):
        return self._complex_dtype

    @complex_dtype.setter
    def complex_dtype(self, complex_dtype):
        self._complex_dtype, self._float_dtype, self._int_dtype = (
            self._resolve_precision(complex_dtype))
        self._recast_to_precision()

    @property
    def float_dtype(self):
        return self._float_dtype

    @property
    def int_dtype(self):
        return self._int_dtype

    def _recast_to_precision(self):
        """Rebuild every stored array at the current width.

        Only reached by *assigning* :attr:`complex_dtype`; the constructors resolve the width
        before any of these exist and never call the setter. So this is the whole of what makes
        ``solver.complex_dtype = 1`` mean the same thing as building the solver at ``1``, and
        before this commit it did not: ``period``, ``psi`` and ``structure`` were left at the
        old width on all three backends. NumPy and JAX then solved at a *mixed* width and
        answered, and the tell is that they answered *identically*: both returned
        ``R + T = 1.000000080317048``, where solvers born at single precision return
        ``1.0000002160668373`` and ``1.0000000447034836``. Two backends agreeing to the last
        bit is not what single precision looks like; it is what double precision looks like.
        Torch did not answer at all -- it raised ``expected m1 and m2 to have the same dtype``
        from inside ``layer_modes_2d``. One call, three outcomes, none of them the request.

        ``structure`` is the one that carries it: re-casting it alone restores the born
        solver's answer bit for bit on both NumPy and Torch, while ``period`` or ``psi`` alone
        restores neither. They are still rebuilt here, because "the attribute reads back at
        the width you asked for" is the contract, and an attribute whose staleness happens to
        be absorbed downstream today is a defect waiting for the path that does not absorb it.

        ``psi`` is assigned privately and cast by hand, which looks like an oversight and is
        not. The public setter calls ``_psi_set_explicitly``, which would (a) flip
        ``_psi_given`` from False to True on a solver whose ``psi`` came from the 45-degree
        default, silently converting a derived value into a declared one, and (b) *raise*
        outright once ``polarization`` or ``tetm`` is set. The copies knew this and wrote
        ``self._psi = self.psi`` -- which reads the property and writes it straight back, so it
        re-cast nothing at all. Same object before and after, on every backend.
        """
        self.theta = self.theta
        self.phi = self.phi
        self.kx = self.kx
        self.ky = self.ky
        if self._psi is not None:
            self._psi = self._as_real_angle(self._psi)
        self.truncation_order = self.truncation_order
        self._set_period_slot(self._period)            # a re-cast, not a new cell
        self.thickness = self.thickness
        # Guarded because `_structure` is set by the `RCWA*` subclass, after `_BaseRCWA`.
        if getattr(self, '_structure', None) is not None:
            self.structure = self._material_input('structure')
        if any(getattr(self, f'_{name}_dispersion', None) is not None
               for name in ('top', 'bot')):
            self.wavelength = self.wavelength

    # ------------------------------------------------------------- the cell and the stack
    #
    # One copy each. These were three, differing in which constructor built the array --
    # `np.array`, `jnp.array`, `_as_tensor` -- and in which array types each was willing to
    # accept alongside `list` and `tuple`. The second difference is a real per-backend fact,
    # so it survives as data rather than as three copies of a branch.

    #: Array types these setters accept beyond `list`, `tuple` and `np.ndarray`. Empty here:
    #: NumPy's own array is already in that trio. JAX and Torch each name theirs, which is what
    #: lets a caller hand a solver a traced or a grad-carrying period. Declared rather than
    #: sniffed, because a generic "has dtype and shape" test would widen all three at once --
    #: NumPy refuses a torch tensor today, and changing that is not this commit's business.
    _ARRAY_TYPES = ()

    def _as_lengths(self, value, pair):
        """A length, or a list of lengths, as a real 1-D array of this solver's width.

        `pair=True` is :attr:`period`, where a lone number means a square cell and a length-1
        sequence means the same; `pair=False` is :attr:`thickness`, where a lone number means a
        single layer and nothing is duplicated.

        Three things here are not incidental:

        * `copy=True`. All three backends copied a NumPy input before this was one function,
          and `as_array` is built around *not* copying -- sharing is how it keeps a graph
          alive. Folding onto it without saying so would have made every solver alias the
          array it was configured from, on all three backends at once.
        * a sequence holding this namespace's own arrays is **stacked**, not rebuilt from, or
          the caller's graph is lost. Each element is cast first, so nothing is created at
          single precision and widened after: `xp.asarray(0.4)` is `float32` under Torch, and
          folding `_incident_kxy` measured exactly that at 1.5e-07 on 13 of 18 probes.
        * the type tests are `type(...) in`, not `isinstance`, everywhere except
          :attr:`_ARRAY_TYPES`. That is what the three copies did, and loosening it would
          start accepting subclasses that are refused today.
        """
        xp = self.xp
        if type(value) in (int, float):
            value = [value] * (2 if pair else 1)
        elif type(value) in (list, tuple, _np.ndarray) or isinstance(value, self._ARRAY_TYPES):
            if pair and len(value) == 1:
                value = [value[0], value[0]]
        else:
            raise ValueError
        if isinstance(value, (list, tuple)) and any(
                isinstance(v, self._ARRAY_TYPES) for v in value):
            return xp.stack([as_array(xp, v, dtype=self.float_dtype,
                                      device=self._array_device) for v in value])
        return as_array(xp, value, dtype=self.float_dtype, device=self._array_device, copy=True)

    @property
    def truncation_order(self):
        return self._max_order

    @truncation_order.setter
    def truncation_order(self, truncation_order):
        self._max_order, self._truncation_spelling = parse_order_pair(truncation_order)

    @property
    def period(self):
        """The cell's two edge lengths — **syntactic sugar for the axis-aligned basis**.

        ``period=(Lx, Ly)`` is a shorthand spelling of ``lattice=diag(Lx, Ly)`` and nothing
        more: it is the same cell, written the short way for the case that has no angle to
        state. :attr:`lattice` materialises it, so a solver given only a period still answers
        with a basis.

        **Reading it back on a skewed cell gives something else.** There it is
        ``periods_of(A) = (|a1|, |a2|)`` -- the two *edge lengths*, with the angle already
        gone, so a 60-degree hex of side 400 and a 400x400 square both report ``(400, 400)``.
        Correct for a caller wanting the physical extent along each raster axis, wrong for one
        wanting a bounding box, and wrong for anything doing geometry: a function handed this
        pair and no basis can only compute a rectangle's answer. Two live defects came through
        exactly that gap on 2026-08-23, which is why
        ``tests/test_cell_shape_gate.py`` now refuses the shape of code that did.

        So: **sugar on the way in, a derived readout on the way out.** :attr:`lattice` is the
        cell; this is a convenience over it.
        """
        return self._period

    @period.setter
    def period(self, period):
        if isinstance(period, dict):
            keys = set(period)
            if keys != {'x', 'y'}:
                raise ValueError(
                    f"period: a mapping needs exactly {{'x', 'y'}}; got {sorted(keys)}. "
                    f"`period` is the axis-aligned spelling of the cell, so x and y are what "
                    f"it has -- a skewed cell is given as `lattice=` instead, and its two "
                    f"directions are u and v.")
            period = [period['x'], period['y']]

        # A cell basis outranks this. `period` is `(|a1|, |a2|)`, which drops the angle, so on
        # a skewed cell it cannot say what the new cell is -- and the sync only ever ran one
        # way: `lattice` derives `period`, `period` never touched `_lattice_basis`. So an
        # assignment here moved `_period` and the read-back while every geometry consumer kept
        # reading the old basis. Measured on a hex cell: `solver.period = [600., 600.]` left T
        # bit-identical at 0.868418046552 while `solver.period` reported [600., 600.].
        #
        # An axis-aligned basis has no angle to lose, so it is updated to match rather than
        # refused. A skewed one is refused, which is this setter's own stance one message up:
        # `period` is the axis-aligned spelling of the cell.
        if self._lattice_basis is not None and not self.lattice_is_axis_aligned:
            raise ValueError(
                'period: this solver was given a skewed cell basis, and `period` is the '
                'axis-aligned spelling -- it is (|a1|, |a2|), so it cannot say what the new '
                'cell is without the angle. Assigning it used to move the read-back and '
                'nothing else. Set the basis instead:\n'
                '    solver.lattice = wavebit.Lattice.oblique(lx, ly, angle)\n'
                'or `wavebit.Lattice.hexagonal(a)` / a 2x2 column_stack((a1, a2)); `period` follows '
                'from it.')

        self._cell_named = True          # an assignment names it, as a keyword does
        self._set_period_slot(period)
        if self._lattice_basis is not None:
            # Axis-aligned, by the guard above: keep the two spellings from drifting apart.
            self._lattice_basis = _lattice.as_lattice(period=self._period)

    def _set_period_slot(self, period):
        """The store, without the cell-basis question -- for the two callers that own it.

        `lattice` derives `period` from a basis it just set, and the precision cascade re-casts
        the value already there. Neither is a caller *choosing* a period, so neither should meet
        a refusal about which spelling to use.
        """
        self._period = self._as_lengths(period, pair=True)

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = self._as_lengths(thickness, pair=False)

    @property
    def wavelength(self):
        """The vacuum wavelength, in the unit ``period`` and ``thickness`` share.

        Stored as given -- no cast, no conversion -- as it was when this was a plain attribute.
        The setter places a backend array on the solver's device and reevaluates
        tabulated half-space and shape materials. All table evaluations must succeed
        before the wavelength or any evaluated material is changed.
        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        wavelength = self._placed(wavelength)
        updates = {}
        for name in ('top', 'bot'):
            source = getattr(self, f'_{name}_dispersion', None)
            if source is not None:
                material = self._placed(self._material_at(source, wavelength))
                updates[f'_n_{name}'] = _half_space_index(material, name)
                updates[f'_{name}'] = material
        source = getattr(self, '_structure_dispersion', None)
        if source is not None:
            updates['_structure'] = self._placed_layers(self._structure_at(source, wavelength))
        # Resolve every table before changing anything: a range error must leave
        # wavelength, boundary indices and layer values describing the same scene.
        self._wavelength = wavelength
        for name, value in updates.items():
            setattr(self, name, value)

    def _material_at(self, value, wavelength):
        return evaluate_material(value, wavelength, xp=self.xp,
                                 dtype=self._complex_dtype, device=self._array_device)

    def _structure_at(self, value, wavelength):
        return evaluate_structure(value, wavelength, xp=self.xp,
                                  dtype=self._complex_dtype, device=self._array_device)

    def _material_input(self, name):
        """Original table-bearing input for a lossless solver pytree round trip."""
        source = getattr(self, f'_{name}_dispersion', None)
        return getattr(self, name) if source is None else source

    def _device_scope(self):
        """Where this solver's arrays are placed, as a context manager.

        `nullcontext` on every backend that does not need one -- NumPy has no device and
        Torch places tensors at creation. JAX overrides it, because `jax.default_device` is
        the only way to say it there and the answer is scoped rather than an argument.
        """
        return _contextlib.nullcontext()

    def _eager(self):
        """Evaluate what is concrete now, even inside a trace -- as a context manager.

        `nullcontext` on NumPy and Torch, which stage nothing. JAX overrides it with
        `jax.ensure_compile_time_eval`. :meth:`_move_to_device` runs under it, because a move
        is a fact about the solver rather than a step of the traced program: staged, it leaves
        a tracer in every slot it touches.
        """
        return _contextlib.nullcontext()

    def _incident_kxy(self, wavelength):
        """The incident in-plane wavevector, ``(k_inc_x, k_inc_y)``, normalized by ``k0``.

        Real angles describe the phase direction and impose real transverse k.
        ``incident_phase_index`` obtains its magnitude from the full complex
        epsilon*mu of the illuminated half-space. Complex angles retain the
        formal n*sin(theta) parameterization; no imaginary component is discarded.

        The historical grazing clamp remains for lossless and complex-angle
        inputs. Absorbing real-angle incidence must have abs(theta) < pi/2:
        at grazing no finite wavevector satisfies uniform surface intensity.
        Invalid traced inputs yield NaNs when their value cannot be checked eagerly.

        Direct kx/ky inputs are physical wavenumbers and are divided by k0 here.
        In angle mode dispersive materials have already been evaluated at the
        current wavelength.
        """
        xp, C = self.xp, self.complex_dtype
        if self.kx is not None or self.ky is not None:
            scale = wavelength / (2 * _np.pi)
            return tuple(self._as_complex(0. if value is None else value * scale)
                         for value in (self.kx, self.ky))
        theta = self._as_complex(0.0 if self.theta is None else self.theta)
        phi = self._as_complex(0.0 if self.phi is None else self.phi)
        material = self.bot if self._incidence_side == 'bottom' else self.top
        epsilon_mu = material.epsilon * material.mu if material.is_magnetic else material.epsilon
        epsilon_mu = self._as_complex(epsilon_mu)
        real_angles = (xp.imag(theta) == 0) & (xp.imag(phi) == 0)
        a, b = xp.real(epsilon_mu), xp.imag(epsilon_mu)
        invalid = real_angles & (((b != 0) & (xp.abs(xp.real(theta)) >= _np.pi / 2))
                                 | ((b == 0) & (a <= 0)))
        if _concrete(xp.any(invalid)):
            raise ValueError(
                'phase-angle incidence requires a propagating incident wave; in an '
                'absorbing half-space, abs(theta) must be less than pi/2 radians')
        # Mask before arithmetic so an unused branch cannot contaminate AD with NaNs.
        use_phase = real_angles & ~invalid
        phase_medium = xp.where(use_phase, epsilon_mu, xp.ones_like(epsilon_mu))
        phase_theta = xp.where(use_phase, xp.real(theta), xp.zeros_like(xp.real(theta)))
        phase_index = incident_phase_index(xp, phase_medium, phase_theta)
        index = xp.where(real_angles, phase_index, self.n_inc)
        index = xp.where(invalid, xp.full_like(index, float('nan')), index)
        clamped = xp.asarray(_HALF_PI_F32_BELOW, dtype=C) + xp.imag(theta) * 1j
        sin_theta = xp.sin(xp.where(xp.real(theta) >= _HALF_PI_F32, clamped, theta))
        return (xp.astype(index * sin_theta * xp.cos(phi), C),
                xp.astype(index * sin_theta * xp.sin(phi), C))

    def _centre_phase(self):
        """``exp(-1j * k_inc . o)``, ``o = (a1 + a2)/2``: the kernels' field to the public one.

        The kernels are handed :meth:`_kernel_lit`'s column, which describes the public incidence
        divided by this scalar: their incident wave is unit at the cell corner, the public one at
        the centre. Amplitudes and efficiencies are ratios and never see it; a field does, once,
        here. One complex number, a pure phase for a real angle. For a complex ``theta`` its
        magnitude is ``exp(Im(k_inc) . o)`` -- 7.7 on a TE scene at ``theta = 0.3 + 0.1j`` in a
        lossy superstrate -- because an evanescent wave is not uniform along the surface and the
        public definition fixes *where* it is unit: at ``x = y = 0``, the one point every public
        coordinate names, and an expression holomorphic in the angle, the wavelength and the
        cell, so a gradient through a complex ``theta`` is the continuation of the real one.

        ``k_inc`` comes from :meth:`_incident_kxy`, the backend's own spelling of the incidence,
        never read back out of an order grid (whose centre entry is the window's, not the
        specular order's), and ``o`` from the basis, never from ``period``, which drops the
        angle.
        """
        xp = self.xp
        k_x, k_y = self._incident_kxy(self.wavelength)
        A = as_array(xp, self.lattice.columns, dtype=xp.real(k_x).dtype, device=_device_of(k_x))
        k0 = 2 * _np.pi / self.wavelength
        phase = -0.5j * k0 * (k_x * (A[0, 0] + A[0, 1]) + k_y * (A[1, 0] + A[1, 1]))
        return xp.astype(xp.exp(phase), self.complex_dtype)

    @device_scoped
    def order_kxy(self):
        """Per-order in-plane wavevectors ``(kx, ky)``, normalized by ``k0``.

        What the 2D solve uses, in whichever layout the lattice admits. Axis-aligned cells get
        the two 1-D vectors, lengths ``(n_orders_u,)`` and ``(n_orders_v,)`` — handed straight
        back from the separable build behind it, untouched, so on a rectangular cell this *is*
        that build and nothing downstream can tell the lattice step is there.
        A skewed cell cannot be written that way and gets the ``(n_orders_v, n_orders_u)``
        Cartesian components instead; both spellings are what
        :func:`wavebit._core.solve_2d.order_grid_kxy` accepts.

        The 1D and conical operators take the 1-D pair and ``diag`` it, so they read the
        separable build directly (:meth:`_solve_1d`) -- which is the same thing on an
        axis-aligned cell, and is what they have always done on a skewed one.

        **This is public because the separable half is a trap on its own.** Until 2026-08-20
        the public accessor was ``get_kx_ky_vector``, which builds only the 1-D pair, and on a
        skewed cell that pair describes the rectangular problem rather than the one being
        solved -- measured on a 60-degree hexagonal cell at ``truncation_order=(2, 2)``, ``ky`` is out
        by up to ``0.805 k0`` across a retained band of about ``+-2``, silently, with nothing
        in the docs saying so. The correct answer was the private one. They swapped places.

        The incident components come from the backend's :meth:`_incident_kxy`, so its own
        ``sin(theta)`` handling — which differs between NumPy, JAX and Torch for tracing
        reasons — stays the single source for incidence, and the lattice basis is the only new
        arithmetic here. They used to be read back out of the 1-D vectors instead, at the index
        this docstring called ``m = n = 0`` and which was in fact the *window centre*. Those
        agree only while the window is centred on zero; under ``excitation`` it follows the lit
        orders, and the entry carried a ``cx * lambda / |a1|`` that the lattice term below then
        added a second time. Measured at ``excitation={(2, 0): 1}`` on a sheared 2000 x 2200
        cell, ``kx`` for the lit order came out ``1.183`` against a true ``0.683`` -- enough to
        put a propagating order outside the light line, so the run refused as evanescent rather
        than answering. See ``tests/test_oblique_lattice.py``.

        The skewed branch does not call :meth:`_separable_kxy` at all, which is the point rather
        than a saving: that build divides by ``period``, and ``period`` on a skewed cell is
        ``(|a1|, |a2|)`` with the angle gone. Keeping it off this path is what lets
        ``tests/test_cell_shape_gate.py`` exempt it on reachability instead of on a promise that
        its result goes unused.

        Takes no wavelength, for the reason ``solve_conv`` takes none: the solver has one, and
        every caller passed exactly that.
        """
        wavelength = self.wavelength
        A = self._lattice_basis
        if A is None or _lattice.is_axis_aligned(A):
            return self._separable_kxy(wavelength)
        k_inc_x, k_inc_y = self._incident_kxy(wavelength)
        (cx, half_width_u), (cy, half_width_v) = self.order_window
        xp = array_namespace(k_inc_x)
        return _lattice.order_kxy(xp, A, half_width_u, half_width_v, wavelength,
                                  k_inc_x, k_inc_y, centre=(cx, cy),
                                  dtype=k_inc_x.dtype, device=_device_of(k_inc_x))

    @property
    def scattering(self):
        """The SMM cascade's own :class:`~wavebit._core.scattering.ScatteringMatrix`.

        Present from ``answers='amplitude'`` up on an ``'SMM'`` solve, and *absent* otherwise
        -- which raises, naming which of the four reasons applies, rather than handing back a
        ``None`` that says all three at once and then fails as an attribute error on it. The
        split is the one :meth:`_require_field_declaration` already makes for
        :meth:`reconstruct_field`, for the same reason: the advice differs, and two of the
        three are declarations rather than states, so "solve again" would be false there.
        """
        if self._scattering is not None:
            return self._scattering
        if self.layer_algorithm != 'SMM':
            raise RuntimeError(
                _SCATTERING_WRONG_ALGORITHM.format(got=self.layer_algorithm))
        if not self._answers.covers(Answers.AMPLITUDE):
            raise RuntimeError(_SCATTERING_NOT_DECLARED.format(got=self._answers.value))
        # The fourth reason, and a state like the third: JAX's jitted `solve_field` kept a
        # keyword and dropped the matrix of the scene before it (`_forget_the_solve`). "No
        # scattering matrix yet" would say no solve had run, when one ran and is gone.
        if self._solved_scene is getattr(type(self), '_SOLVED_ELSEWHERE', None):
            raise RuntimeError(_SCATTERING_SOLVED_ELSEWHERE)
        raise RuntimeError(_SCATTERING_NOT_SOLVED)

    def _finish(self, out):
        """Store the field-reconstruction state and return just the result mapping.

        All three slots are *reset*, not merely filled: below ``answers='field'`` the core hands
        back a mapping without them, and clearing them here is what stops a later
        reconstruction from walking the state of some earlier, unrelated solve. The
        The ``'efficiency'`` rung returns ``(R, T, power_entering)`` without retained
        amplitudes. It is converted into the shared result layout here.

        It is also the one door every kernel answer passes on its way out -- all three paths,
        both formulations, both algorithms -- so it is where the kernels' corner-referenced
        amplitudes and scattering matrix are signed onto the cell centre
        (:func:`~wavebit._core.conventions.centre_parity`). The kernels were handed the column
        :meth:`_kernel_lit` built, ``C w`` with ``C = diag((-1)**(m + n))``; what they return is
        about the corner, so the public amplitude is ``C`` times it, ``C L C w`` in all, and
        the scattering matrix ``C S C`` block by block: every port amplitude, both ports, takes
        the same sign, which commutes with the pure 1D path's port change and with the
        backside's block reversal. ``where`` rather than a multiply by ``-1``: an exact sign bit
        on every entry, no complex product. ``T1`` and ``layer_info_list`` stay the kernels'
        own state in their own frame; :meth:`reconstruct_field` and the result's field adapter
        convert what they describe.
        """
        if not isinstance(out, dict):
            out = {'incident': _efficiency_only(*out)}
        self.layer_info_list = out.pop('layer_info_list', [])
        self.T1 = out.pop('T1', None)
        #: The scene this solve was posed with, so `reconstruct_field` can tell whether it is
        #: still being asked about the same one. See `_reconstruction_scene`.
        self._solved_scene = self._reconstruction_scene()
        odd = centre_parity(*self.order_window)
        for key in ('incident', 's_inc', 'p_inc'):
            sub = out.get(key)
            if sub is not None and sub['r_s'] is not None:
                out[key] = dict(sub, **{name: _centred(sub[name], odd)
                                        for name in ('r_s', 'r_p', 't_s', 't_p')})
        #: The stack's scattering matrix -- both sides, every order -- on this shelf rather than in
        #: result: :class:`~wavebit._core.result.Result` is a registered pytree with fixed
        #: children, so carrying matrices there would change its structure and drag them
        #: across every ``jit`` / ``grad`` boundary. ``None`` after an ETM solve -- not empty
        #: but *absent*, because the ETM fixes one illumination's boundary condition and
        #: never forms a both-sides operator at all.
        scattering = out.pop('scattering', None)
        if scattering is not None:
            # One sign per port amplitude, s and p halves alike on the coupled layout, so a
            # block entry takes the product of its row's and its column's.
            port = odd.reshape(-1)
            if scattering.S11.shape[-1] == 2 * port.size:
                port = _np.concatenate([port, port])
            pair = port[:, None] ^ port[None, :]
            scattering = scattering._replace(**{name: _centred(getattr(scattering, name), pair)
                                                for name in ('S11', 'S12', 'S21', 'S22')})
        self._scattering = scattering
        return out

    _formulation = DEFAULT_FORMULATION
    _formulation_assigned = None

    @property
    def formulation(self):
        """Which layer formulation the 2D solve uses: ``'auto'`` (default), ``'2ff'``, ``'4ff'``.

        ``'2ff'`` is the second-order form the library has always used, one field pair per
        order. ``'4ff'`` carries the full transverse state and is what a populated
        permittivity ``z`` row — the Voigt / magneto-optic geometry — needs; it costs about 8x,
        because the eigenproblem doubles and ``eig`` is cubic.

        The default reads the problem rather than asking: a solve handed any of the four
        ``ep{xz,yz,zx,zy}_conv_all`` slots takes ``'4ff'``, everything else ``'2ff'``. That
        decision is the solver's because a caller with a Faraday stack has no way to know that
        one populated ``eps_yz`` is what would cost them eight times more.

        Setting it overrides that, in the two directions that mean something:

        * ``'4ff'`` forces the first-order path on a problem that does not need it. This is
          how the two formulations are cross-checked against each other — the same scene run
          both ways must agree, and no external oracle provides that. The first-order path
          is written on the 2D layout, so ``tetm`` beside it is refused: give the incidence
          as ``polarization`` for the cross-check.
        * ``'2ff'`` refuses a z row instead of dropping it. Dropping it is the dangerous
          failure: the reduced tensor is a perfectly good medium, so the solve succeeds and
          ``R + T`` stays at 1 to machine precision while the efficiencies have moved by 2e-2
          to 6e-2.

        Which one actually ran is on :attr:`formulation_assigned` afterwards.
        See ``docs/anisotropic-materials.md`` §8.
        """
        return self._formulation

    @formulation.setter
    def formulation(self, value):
        self._formulation = formulation_of(value)

    @property
    def formulation_assigned(self):
        """The formulation the last solve actually ran on — ``None`` before the first.

        The companion to :attr:`grating_type`, and exposed for the same reason: a
        silent 8x is indistinguishable from a slow machine, and a benchmark that cannot say
        which formulation it measured is not a benchmark.

        Every solve sets it, including the 1D ones, which are second-order and have no choice
        to make. Leaving those alone would be the worse answer: the attribute would then keep
        whatever a *previous* 2D solve put there, and a stale ``'4ff'`` reads exactly like a
        fresh one.
        """
        return self._formulation_assigned

    def _formulation_for(self, z_row, mu=None):
        """Settle :attr:`formulation` against the z-row slots this call was handed.

        ``z_row`` is the four ``ep{xz,yz,zx,zy}_conv_all`` arguments. The test is whether a
        slot was *given*, not whether it is nonzero: the branch decides control flow, so it has
        to be answerable on a JAX tracer, and "passed all-zero matrices" is a claim about the
        caller rather than about the tensor. Leave them ``None`` — the default — to stay on the
        cheap path.
        """
        given = _z_row_given(z_row, mu)
        if self._formulation == Formulation.TWO_FIELD and given:
            raise ValueError(
                f"formulation='2ff' cannot express the z row, and {', '.join(given)} "
                f"{'was' if len(given) == 1 else 'were'} given. "
                'The second-order form d2E/dz2 = Omega2 E exists only while the '
                'epsilon and mu z off-diagonals are absent; dropping them would answer for the '
                'reduced tensor, which conserves energy to machine precision while the '
                "efficiencies move by 2e-2 to 6e-2. Use formulation='auto' (the default) or "
                "'4ff'. See docs/anisotropic-materials.md section 6.")
        if self._formulation == Formulation.AUTO:
            return Formulation.FOUR_FIELD if given else Formulation.TWO_FIELD
        return self._formulation

    # ------------------------------------------------------------------ path selection

    def _assign_grating_type(self):
        """Pick the formulation this scene needs: pure 1D TE/TM, 1D conical, or 2D.

        Reads the structure and the declaration, and nothing about the backend -- which is why
        it lives here rather than in triplicate on the three solver classes, where it sat until
        2026-08-21. It had been copied verbatim into all three, so the skewed-cell routing bug
        `3c7d166` fixed existed three times and had to be fixed three times. Same argument as
        the one that put the solve methods here (see this class's docstring).

        * ``0`` -- pure 1D TE/TM: one grating vector, no azimuth, and a ``tetm`` declaration
          that says the other polarization is not a degree of freedom of this problem.
        * ``1`` -- 1D conical: the same one grating vector with the azimuth and the
          polarization both live. ``phi=None`` is what distinguishes it from ``phi=0``; a
          stated zero azimuth is still a stated one.
        * ``2`` -- 2D: everything else, and every scene whose k layout is not separable.
        """
        self._refuse_a_structure_with_no_cell()
        self._refuse_a_raster_drawn_on_another_cell()
        if self.structure is None:
            if self._grating_type is None:
                self._grating_type = 2
            elif self._grating_type == 0:
                # The pure layout is kept only from a one-row raster, so what the raster branch
                # below asked of that raster is asked again: the scene may have moved since.
                # Asked nothing, a kept pure layout answered for the wrong light -- phi = 0.4
                # solved as phi = 0, a nonzero ky as ky = 0, a skewed cell as the rectangle --
                # or crashed on truncation_order[1] != 0 with a bare broadcast error (measured
                # on 0937b35). The refusals come before any write, so the routing stays put.
                one_dimensional = self.lattice_is_axis_aligned
                self._refuse_y_orders_on_a_y_invariant_cell(True)
                self._refuse_illegal_tetm(one_dimensional, one_row=True)
                if self._tetm is None:
                    # Withdrawn since: the pure kernel cannot run undeclared (it raised
                    # "tetm_index must be 0 (TE) or 1 (TM), got None"), so route as the raster
                    # branch routes that raster undeclared -- conical on an axis-aligned cell,
                    # the solve a never-declared solver gives bit for bit, and 2D on a skewed
                    # one, where the conical layout would answer the rectangle.
                    self._grating_type = 1 if one_dimensional else 2
            return

        if self._has_tensor_materials():
            self._grating_type = 2
            return

        if self.modeling_path == ModelingPath.RASTER:
            one_row = self.structure.shape[1] == 1
            self._refuse_y_orders_on_a_y_invariant_cell(one_row)
            # One raster row means invariant along `a2`, which is a 1D grating -- but only on
            # an axis-aligned cell is it one the 1D operators can express. A skewed cell's
            # `b1` is perpendicular to `a2` rather than parallel to `a1`, so order `m` carries
            # `m*b1_y` of `ky` that the separable layout has no slot for, and the grooves do
            # not run along y. Routed to the 1D path anyway it answers the *rectangular*
            # problem: measured on a 50% binary grating, R out by 0.132 and T by 0.090 at a
            # 60-degree skew, energy conserving to 1e-15 throughout. See
            # `tests/test_oblique_lattice.py::test_a_one_row_skewed_cell_is_the_grating_it_draws`.
            one_dimensional = one_row and self.lattice_is_axis_aligned
            warn_if_xy_on_a_skewed_cell(
                getattr(self, '_truncation_spelling', None), self.lattice_is_axis_aligned)
            self._refuse_illegal_tetm(one_dimensional, one_row=one_row)
            if one_dimensional:
                # The declaration selects the path, and nothing else does. A named incidence
                # -- psi, or a pair, which mixes s and p the pure path cannot carry -- leaves
                # the polarization a live parameter, and the conical formulation is the 1D
                # one that carries it.
                self._grating_type = 0 if self._tetm is not None else 1
            else:
                self._grating_type = 2

        elif self.modeling_path == ModelingPath.PARAMETRIC:   # staircase-free
            self._grating_type = 2

    @property
    def grating_type(self):
        """Which formulation the routing picked: ``0`` pure 1D TE/TM, ``1`` conical, ``2`` 2D.

        ``None`` before a structure has been assigned. The companion to
        :attr:`formulation_assigned`, exposed for the reason that one gives -- a path that
        costs 3.75 blocks per layer and one that costs 4.25 are not distinguishable from the
        outside, and a scene that silently reroutes is compared against the right numbers
        through the wrong code. ``tests/data/golden_field.npz`` stores it beside every case
        for exactly that.

        Read-only, and named without the ``_assigned`` suffix
        :attr:`formulation_assigned` carries: that suffix earns its place by separating what
        the solver chose from what the caller asked for, and ``formulation`` is settable while
        this is not. There is no ``grating_type`` to set -- the structure and the declaration
        decide it -- so a suffix here would be guarding against nothing.
        """
        return self._grating_type

    @property
    def modeling_path(self):
        """Which modeling path the structure selected, as a :class:`.ModelingPath`.

        ``None`` before a structure has been assigned. The *type of the object* passed as
        ``structure`` is the choice -- there is no separate mode switch -- so this is the only
        place that says which one was read, and it is what ``_core.draw`` reads to label a
        picture with the path that drew it. Read-only.

        A ``str`` enum, so ``solver.modeling_path == 'raster'`` holds without importing
        anything. It was a bare integer (``modeling_type``, ``0`` and ``2``) until the paths
        were named: the missing ``1`` was the block path, removed in ``ad415b4``, and an
        integer that has to be looked up is a poor thing to print in a refusal.
        """
        return self._modeling_path

    def plot_field(self, field_cell):
        """Draw a reconstructed cell. See :func:`wavebit._core.fields.plot_field`."""
        from .fields import plot_field
        plot_field(field_cell, self.tetm_index)

    @staticmethod
    def _mu_1d(mu):
        """``solve_conv``'s five permeability slots narrowed to the 1D operators' three.

        The 2D layout is ``(x, y, z_i, xy, yx)`` and the 1D one is its first three: those
        operators have no off-diagonal position, so the last two slots have nowhere to go.
        Slicing here rather than at each call site keeps the *caller-facing* order one order
        -- the property ``tests/test_slot_order.py`` pins for every builder -- while the
        narrowing happens once. ``solve_conv`` has already refused a non-``None`` entry in
        either dropped slot, so this only ever discards ``None``.
        """
        return None if mu is None else tuple(mu)[:3]

    def _solve_1d(self, epx_conv_all, epy_conv_all, epz_conv_i_all,
                  mu=None, mu_top=None, mu_bot=None, eps_top=None, eps_bot=None):
        if self._polarization is not None:
            # Unreachable through solve — `_assign_grating_type` sends a structure with a
            # pair to the conical path, which carries both polarizations — so this only
            # answers a direct call.
            raise ValueError(
                'the pure 1D TE/TM path is single-polarization by construction and cannot '
                'take a polarization pair; use _solve_1d_conical, which solve() selects on '
                'its own when polarization is set')
        solve = self._solve_fn('1d')
        self._formulation_assigned = Formulation.TWO_FIELD
        wavelength = self.wavelength
        # The separable pair, not :meth:`order_kxy`: the 1D operator takes a 1-D ``kx`` and
        # ``diag``s it, and the two are the same object here -- `_assign_grating_type` reaches
        # this path only under `one_row and lattice_is_axis_aligned`, so the cell is a rectangle
        # by the time the call is made. A skewed one-row cell is routed to `_solve_2d` instead,
        # because its `b1` is perpendicular to `a2` rather than parallel to `a1` and order `m`
        # carries a `ky` the separable layout has no slot for.
        kx, _ = self._separable_kxy(wavelength)
        # `mu_top` / `mu_bot` are `None` when no half-space permeability was named -- the
        # structural spelling `solve_conv` explains -- and the numeric default lives at the one
        # place that divides by it, so absence resolves to `1.0` exactly once, here.
        return self._finish(solve(
            self.tetm_index, wavelength, kx, self.n_bot, self.n_top, self.theta,
            self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            incidence_side=self._kernel_incidence_side, answers=self._answers,
            lit=self._kernel_lit(), mu=self._mu_1d(mu),
            mu_top=1.0 if mu_bot is None else mu_bot,
            mu_bot=1.0 if mu_top is None else mu_top,
            eps_top=eps_bot, eps_bot=eps_top))

    def _solve_1d_conical(self, epx_conv_all, epy_conv_all, epz_conv_i_all,
                          mu=None, mu_top=None, mu_bot=None, eps_top=None, eps_bot=None):
        solve = self._solve_fn('conical')
        self._formulation_assigned = Formulation.TWO_FIELD
        wavelength = self.wavelength
        kx, ky = self._separable_kxy(wavelength)   # 1-D pair; see :meth:`_solve_1d`
        return self._finish(solve(
            wavelength, kx, ky, self.n_bot, self.n_top, self.theta, self._incidence_pair(), self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            phi=self.phi, incidence_side=self._kernel_incidence_side,
            answers=self._answers, lit=self._kernel_lit(), mu=self._mu_1d(mu),
            mu_top=1.0 if mu_bot is None else mu_bot,
            mu_bot=1.0 if mu_top is None else mu_top,
            eps_top=eps_bot, eps_bot=eps_top))

    def _solve_2d(self, epx_conv_all, epy_conv_all, epz_conv_i_all,
                 epxy_conv_all=None, epyx_conv_all=None,
                 epxz_conv_all=None, epyz_conv_all=None,
                 epzx_conv_all=None, epzy_conv_all=None, mu=None, mu_top=None, mu_bot=None,
                 eps_top=None, eps_bot=None):
        # None means no explicit magnetic half-space; numeric defaults resolve
        # only at the kernel boundary, so tracing never branches on mu values.
        wavelength = self.wavelength
        kx, ky = self.order_kxy()
        z_row = (epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all)
        self._formulation_assigned = self._formulation_for(z_row, mu)
        if self._formulation_assigned == Formulation.FOUR_FIELD:
            return self._finish(_solve_2d_4ff(
                wavelength, kx, ky, self.n_bot, self.n_top, self.theta, self._incidence_pair(),
                self.thickness, epx_conv_all, epy_conv_all, epz_conv_i_all,
                epxy_conv_all=epxy_conv_all, epyx_conv_all=epyx_conv_all,
                epxz_conv_all=epxz_conv_all, epyz_conv_all=epyz_conv_all,
                epzx_conv_all=epzx_conv_all, epzy_conv_all=epzy_conv_all,
                phi=self.phi, incidence_side=self._kernel_incidence_side,
                answers=self._answers, layer_algorithm=self.layer_algorithm,
                lit=self._kernel_lit(), mu=mu,
                mu_top=1.0 if mu_bot is None else mu_bot,
                mu_bot=1.0 if mu_top is None else mu_top,
                eps_top=eps_bot, eps_bot=eps_top))
        solve = self._solve_fn('2d')
        return self._finish(solve(
            wavelength, kx, ky, self.n_bot, self.n_top, self.theta, self._incidence_pair(), self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            epxy_conv_all=epxy_conv_all, epyx_conv_all=epyx_conv_all,
            phi=self.phi, incidence_side=self._kernel_incidence_side,
            answers=self._answers, lit=self._kernel_lit(), mu=mu,
            mu_top=1.0 if mu_bot is None else mu_bot,
            mu_bot=1.0 if mu_top is None else mu_top,
            eps_top=eps_bot, eps_bot=eps_top))


# The rules a modeling path builds on every structure it accepts, its default first.
# `ModelingPath.RASTER` is absent because it implements all of them and so takes the option
# as given. Shapes default to NVF and take Laurent's rule on any shape -- it reads the same
# closed-form `[[eps]]` NVF starts from; explicit Cartesian Li is conditional on the shapes
# and handled on read, so it is not listed.
_FACTORIZATION_IMPLEMENTED_BY = {
    ModelingPath.PARAMETRIC: (Factorization.NVF, Factorization.LAURENT),
}

# General parametric shapes use NVF or Laurent; Cartesian rectangles have an explicit Li route.
_FACTORIZATION_REASON = {
    ModelingPath.PARAMETRIC: "general shapes take factorization='nvf' (the default) or "
                             "'laurent'; explicit Li supports axis-aligned rectangles",
}

# Why every NVF option is refused under Laurent's rule, in one spelling for the four refusals
# that give it -- raster_normal_method, nvf_projector, shape_nvf_grid_size, and the tensor one.
_LAURENT_BUILDS_NO_NORMAL = ("factorization='laurent' builds no normal field at all -- it takes "
                             "the direct rule for every product")

# The other Fourier axis, exactly the same shape of fact. The shape path describes a layer as
# geometry rather than as samples, and integrates the permittivity over that geometry in
# closed form; it has no raster underneath to run a discrete series along.
# `ModelingPath.RASTER` is again absent because it is the one that implements both.
_COEFFICIENTS_IMPLEMENTED_BY = {
    ModelingPath.PARAMETRIC: FourierCoefficients.INTEGRATED,
}

# Why, phrased for the person who asked for 'sampled'. The parametric path's missing series is
# not merely unbuilt but unbuildable -- a parametric shape has no samples to take.
_COEFFICIENTS_REASON = {
    ModelingPath.PARAMETRIC: (
        "the parametric path transforms geometry in closed form -- each "
        "shape in turn (wavebit._core.parametric.shape_ft), or, for a painted layer whose "
        "shapes overlap, the disjoint polygons the painting resolves into "
        "(wavebit._core.boundary.to_conv_mat_boundary_layer). Either transform *is* the "
        "integral of eps over the cell, evaluated in closed form. Parametric geometry has no "
        "samples to take a discrete "
        "series of, so the sampled coefficients are not a series that path has yet to grow -- "
        "they are not one it can form. The integrated coefficients are the exact ones anyway, "
        "so nothing is lost: drop the keyword, or pass fourier_coefficients='integrated' to "
        "say so"),
}


class CoreFactorizationMixin:
    """Provides ``solve`` — factorization through the core, then the core solve.

    Dispatches the same way the per-backend ``solve`` implementations did (modeling type,
    then the Fourier options) but builds every convolution matrix with
    :mod:`wavebit._core.factorization`. Two consequences beyond de-duplication:

    * NVF works on every backend — the per-backend builders had it only in NumPy, so JAX and
      PyTorch used to raise here.
    * The eager geometry steps (NVF normal field, the integrated path's block compression) are computed inside
      the builders, which needs concrete values. For JAX-traced use, call the core builders
      directly with a precomputed ``normal_field`` / ``compression``.

    The parametric path (``modeling_path`` ``PARAMETRIC``) is built from parametric shapes rather
    than an array, so there is no input array to read the namespace from — it comes from the
    backend code instead.

    Also owns the two Fourier option attributes, so the three backend constructors resolve them
    identically. *Neither* axis is a free choice on every path — see :attr:`factorization` and
    :attr:`fourier_coefficients` — so both are stored as *what was asked for* (``None`` when
    nobody asked) and resolved against the modeling path on read.
    """

    def __init__(self,
                 top=1.,                     # superstrate: a Material, or a bare relative
                 # permittivity (the scalar shorthand for Material(epsilon=x)). Was `n_top`
                 # and held a refractive index; both the name and the quantity changed.
                 bot=1.,                     # substrate, same reading as `top`.
                 theta=None,
                 phi=None,
                 psi=None,
                 period=_CELL_UNSET,   # (Lx, Ly); the sentinel is 'not named', not a value.
                 # See `_CELL_UNSET` and the two refusals at the end of this constructor.
                 wavelength=1.,
                 structure=None,             # the layer stack. A raster is an
                 # (n_layers, Ny, Nx) array of relative PERMITTIVITY; the two shape spellings
                 # name a Material per region instead. Was `ucell`, and held index samples.
                 thickness=(0.,),
                 backend=None,               # the integer `rcwa(backend=...)` takes.
                 # `None` means this class's own `_BACKEND_ID` -- the one default that cannot
                 # be a shared literal. It stays an accepted keyword because
                 # `SolverJax._tree_unflatten` passes it back in on every jitted call.
                 tetm=None,
                 truncation_order=(0, 0),
                 layer_algorithm='ETM',
                 excitation=None,   # which orders the illumination arrives on; see
                 # `CoreSolveMixin.excitation`. None is one wave on the specular order.
                 device=0,                    # `0` / `'cpu'`, `1` / `'gpu'` / `'cuda'`, or
                 # the backend's own device object. wavebit's vocabulary rather than a
                 # namespace's, which is what lets one signature serve three backends; each
                 # resolves it through `_resolve_device`. NumPy has one device and refuses a
                 # GPU request rather than printing about it.
                 complex_dtype=0,            # `0` (double) / `1` (single), or any of the
                 # three namespaces' own complex spellings. Same argument as `device`.
                 fourier_coefficients=None,  # 'sampled' | 'integrated'
                 # (closed-form series); the default resolves per modeling path --
                 # 'sampled' on a raster, 'integrated' on the shape path, which is
                 # the only series it can form. Naming the other one raises.
                 factorization=None,         # 'li' | 'nvf' (normal vector field) |
                 # 'laurent' (the direct rule for every product -- opt-in, for comparing with
                 # solvers that use it, never a default); the default resolves per modeling
                 # path -- 'li' on a raster, 'nvf' on parametric shapes. The shape path builds
                 # 'li' only on axis-aligned rectangles in an axis-aligned rectangular cell
                 # (`rectilinear`), and refuses it on any other shape; tensor materials refuse
                 # 'laurent'.
                 shape_nvf_grid_size=None,  # normal-field sampling, per axis; None resolves to
                 # max(_NORMAL_GRID_DEFAULT, 4*half_width+1) -- 1024, lifted to the aliasing
                 # floor. This comment said 256 until 2026-09-02, which is a number that was
                 # tried and reverted: it is a rectangle's *angle* that sets the grid, and 256
                 # sits 9.7% from the converged gradient there. Read by the shape paths only --
                 # the normal field has no closed-form transform, so it is sampled and read off
                 # this grid. See `_resolve_normal_grid` for the measurements.
                 lattice=None,               # 2x2 cell basis (columns a1, a2); None = diag(period).
                 # New parameters go at the end on purpose: SolverJax._tree_unflatten passes
                 # the children positionally, so one added earlier silently shifts them.
                 incidence_side='top',       # 'top' | 'bottom': which medium the light
                 # arrives from. Scene labels (top / bot / structure order) never move;
                 # results are incidence-relative and theta/phi are measured in the
                 # incidence medium. See docs/conventions.md.
                 chi=None,                   # ellipticity angle; with psi it is the
                 # canonical (psi, chi) chart of the polarization state. Also at the
                 # end, for the reason above.
                 polarization=None,          # complex incident amplitude pair (a_s, a_p);
                 # supersedes psi (naming both raises). See docs/conventions.md.
                 raster_normal_method=None,  # raster NVF normal-field construction:
                 # 'distance' (default) | 'contour' | 'spectral' | 'smooth'. Read only
                 # where a raster normal field is built (factorization='nvf' on a raster);
                 # explicitly set anywhere else it is refused rather than ignored.
                 # See docs/nvf-normal-methods.md for the measurements that rank them.
                 nvf_projector=None,         # how P is formed for raster or shape NVF:
                 # 'linear' (default, P = n n^T) | 'circular' (Antos' complex polarization
                 # basis, direction-free as |n| -> 0 -- the medial axis). A second axis
                 # rather than a fifth raster_normal_method: it composes with all four
                 # raster constructions and is also supported by the shape NVF path.
                 answers='amplitude',        # what this solve keeps the state to answer:
                 # 'efficiency' | 'amplitude' (default) | 'field'. A declaration, like tetm --
                 # reconstruct_field is refused below 'field' rather than served, because the
                 # per-layer state it walks is 4.25 matrices of side 2*n_orders_u*n_orders_v per
                 # layer (240 MB at truncation_order=(15, 15)) and is kept as the solve runs or not at
                 # all. See CoreSolveMixin.answers.
                 mu=None,                    # raster path only: the permeability field,
                 # (n_layers, Ny, Nx) beside `structure`. On the two shape paths a
                 # permeability belongs to the Material filling each region, so naming it
                 # here is refused rather than ignored. New parameters go at the end -- see
                 # the note on `lattice` above.
                 kx=None, ky=None,           # real Cartesian components in inverse length
                 li_nesting=None,            # factorization='li' only: the order Li's rules
                 # are nested in. 'fixed' (default -- v inside u for every slot, what every
                 # 'li' number here was measured on) | 'symmetric' (each slot's inverse-rule
                 # axis inside, Li 1997: covariant under relabelling x and y). Read by scalar
                 # Li, on a raster and on axis-aligned rectangles; explicitly set where it has
                 # no reader it is refused rather than ignored -- under 'nvf' or 'laurent', and
                 # 'symmetric' on tensor materials, whose Li build is the fixed nesting only. See
                 # docs/factorization-terms.md section 6.
                 **unknown,                  # collected rather than left to Python, so the
                 # refusal comes from `_refuse_unknown_keywords` and a typo cannot be
                 # swallowed by the catch-all itself.
                 ):
        _refuse_unknown_keywords(unknown)

        # `period` and `lattice` are two spellings of one cell, not two cells, so naming both
        # is refused rather than resolved. It resolved silently until 2026-09-01 -- `lattice`
        # won and `period` was dropped, which reads as a lattice effect and is not one.
        #
        # This has to be the *named* keyword and not the value, because three internal paths
        # legitimately hold both and hand them on: `SolverJax._tree_unflatten` (period is a
        # pytree child, lattice is aux data), `Stack.as_kwargs`, and the spec round trip, which
        # writes `period` on every document. All three now drop `period` when a basis is
        # present -- provably a no-op, since `self.lattice = lattice` below overwrites the
        # period slot with `periods_of(A)` whenever a basis arrives.
        if period is not _CELL_UNSET and lattice is not None:
            raise ValueError(
                'period and lattice both given. `period=(Lx, Ly)` is the axis-aligned '
                '*spelling* of a cell, not a second cell -- on a skewed one it is '
                '(|a1|, |a2|), with the angle already gone, so the two cannot both be '
                'honoured. Pass one:\n'
                '    lattice=wavebit.Lattice.hexagonal(400.)      # or oblique(...) / a 2x2\n'
                '    period=[700., 1100.]                 # the rectangular cell\n'
                'Until now `lattice` won and `period` was ignored in silence.')

        super().__init__(top=top, bot=bot, theta=theta, phi=phi, psi=psi, chi=chi, tetm=tetm,
                         truncation_order=truncation_order,
                         period=(1., 1.) if period is _CELL_UNSET else period,
                         wavelength=wavelength,
                         thickness=thickness, layer_algorithm=layer_algorithm,
                         excitation=excitation,
                         device=device, complex_dtype=complex_dtype,
                         polarization=polarization, kx=kx, ky=ky)
        self.incidence_side = incidence_side
        self.answers = answers

        self._modeling_path = None
        self._grating_type = None
        self._structure = None

        self.structure = structure
        # After `structure`: `mu` is refused on the two shape paths, and which path this is
        # only becomes known when the structure is assigned.
        self.mu = mu

        self.backend = self._BACKEND_ID if backend is None else backend
        self._set_fourier_options(fourier_coefficients, factorization)
        self.raster_normal_method = raster_normal_method
        self.nvf_projector = nvf_projector
        self.li_nesting = li_nesting
        self._normal_grid = shape_nvf_grid_size
        self.lattice = lattice      # after period: assigning a basis overwrites period

        # Whether a cell was ever *named*. Read by `_refuse_a_structure_with_no_cell`, which
        # runs here and again at the solve -- a structure can arrive by assignment long after
        # construction, and the door has to be closed on both.
        self._cell_named = period is not _CELL_UNSET or lattice is not None
        self._refuse_a_structure_with_no_cell()


    #: Which of the three modeling paths the current ``structure`` selected; the concrete
    #: solver classes assign it. Defaulted here because both option properties resolve against
    #: it, so the mixin would otherwise only work inside a class that happens to set it first.
    _modeling_path = None
    #: The raster permeability field; ``None`` is non-magnetic. Read through :attr:`mu`.
    _mu = None
    #: What was *asked for*, with ``None`` meaning "nobody said". Read through the
    #: :attr:`factorization` property, never directly: only two of the three modeling paths
    #: implement both rules, so the requested value and the effective one are not the same
    #: thing, and telling an explicit request from an untouched default is what decides
    #: whether the mismatch is worth refusing.
    _factorization = None
    #: Likewise, and for the same reason — see :attr:`fourier_coefficients`. Read through the
    #: property.
    _fourier_coefficients = None
    #: Likewise unset-vs-default, though the resolution is simpler: ``'distance'`` is the
    #: default everywhere, and the keyword is only *read* where a raster normal field is
    #: built. What the None distinguishes is an explicit request on a build that would never
    #: read it, which :meth:`_reject_unread_normal_method` refuses instead of ignoring.
    _normal_method = None
    #: Likewise, preserve whether the NVF projector was explicitly requested. On a raster,
    #: ``raster_normal_method`` and this select how the field is built and projected. The
    #: shape NVF path also reads this option, but builds its field directly from geometry.
    _projector = None
    #: The same unset-vs-default slot on the other rule: ``'fixed'`` is the default everywhere,
    #: and the keyword is read where scalar Li's rules are built (a raster, or axis-aligned
    #: rectangles). :meth:`_reject_unread_nesting` refuses an explicit value it cannot honour.
    _nesting = None

    def _set_fourier_options(self, fourier_coefficients, factorization):
        """Settle ``fourier_coefficients`` / ``factorization`` from a constructor's arguments."""
        self._fourier_coefficients, self._factorization = resolve_fourier_options(
            fourier_coefficients, factorization, keep_unset=True)

    # ------------------------------------------------------- the layer stack, as it was given
    #
    # One copy. The three `structure` setters differed in a single `isinstance` check, and
    # `_coerce_raster` in four places: that same check, the accepted-dtype tuples, the error
    # messages, and the conversion at the end. All four collapse onto things `_core` already
    # declares -- `_ARRAY_TYPES`, the namespace's own dtype names, and `as_array`.

    @classmethod
    def _condition_names(cls):
        """Every keyword the solve entry points accept: a public property with a setter.

        Derived rather than listed, so it cannot drift. A condition is something with a
        validating setter -- which is exactly what a `property` with an `fset` is here.

        This is **not** the same set as "every attribute that may be written", and the two must
        not be merged. `backend`, `T1` and `layer_info_list` are public attributes the library
        writes on itself and no caller may pass as a keyword; they are in
        `_ASSIGNABLE_NON_CONDITIONS` and `tests/test_solve_keywords.py` refuses them at the
        keyword door on purpose. Folding them in here would make `solve(backend=1)` legal for
        an effect it does not have.

        Cached per class, because the three backends have different sets and the answer cannot
        change after import.
        """
        cached = cls.__dict__.get('_CONDITION_NAMES')
        if cached is None:
            import inspect

            # Two sources, because neither alone is the set. Properties miss `layer_algorithm`,
            # a plain attribute and exactly what a sweep moves -- deriving from properties alone
            # once refused a wavelength sweep, when `wavelength` was a plain attribute too, and
            # that is the most ordinary call this library has. The constructor misses the
            # retired spellings, which have no parameter and must stay in so their own setters
            # can say what replaced them.
            # Every `__init__` in the MRO, not `cls.__init__`: the public wrapper is
            # `(*args, **kwargs)`, so asking the class itself returns nothing at all -- which
            # is how a first version of this refused a wavelength sweep. The wrappers
            # contribute nothing here because variadics are skipped.
            declared = {name
                        for klass in cls.__mro__
                        for name, parameter in inspect.signature(
                            klass.__init__).parameters.items()
                        if name != 'self'
                        and parameter.kind not in (parameter.VAR_KEYWORD,
                                                   parameter.VAR_POSITIONAL)}
            settable = {name
                        for klass in cls.__mro__
                        for name, member in vars(klass).items()
                        if isinstance(member, property) and member.fset is not None
                        # Prevention rather than a filter: there are no private properties
                        # with setters today. It states that writing a private slot from here
                        # is not a supported way to set a condition, whatever appears later.
                        and not name.startswith('_')}

            # `backend` is a constructor parameter that cannot take effect afterwards: it
            # selected the class, and construction is over. Assigning it changes nothing about
            # the solve and only moves what the JAX pytree hashes into its aux_data, so it
            # would be one more keyword accepted for an effect it does not have.
            cls._CONDITION_NAMES = cached = frozenset(declared | settable) - {'backend'}
        return cached

    #: Public attributes the library writes on itself that are not conditions. `backend` is
    #: subtracted from `_condition_names` above because assigning it cannot take effect;
    #: `layer_info_list` and `T1` are per-solve state written by `__init__` and `_finish`.
    #: Every one is refused as a *keyword* (`tests/test_solve_keywords.py`) and must still be
    #: writable as an *attribute* -- the two rules are different rules, and this is the
    #: difference. Without them `__setattr__` below refuses `rcwa()` itself, at the
    #: `self.layer_info_list = []` in `CoreSolveMixin.__init__`.
    _ASSIGNABLE_NON_CONDITIONS = frozenset({'backend', 'T1', 'layer_info_list'})

    def __setattr__(self, name, value):
        """Refuse a public attribute nothing reads, instead of storing it.

        `solver.thicknes = [900.0]` used to be a new attribute and a solve that answered for
        the *old* thickness -- bit-identically, so there was nothing to notice. Direct
        assignment is the idiom the tutorials and examples teach, and it was the one door into
        the solver with no check on it: the constructor and the five `**kwargs` entry points
        have been guarded since 2026-08-26 (`_refuse_unknown_keywords`,
        `_refuse_unknown_conditions`), which is what made the gap easy to miss -- the same typo
        is caught in `rcwa(thicknes=...)` and was not in `solver.thicknes = ...`.

        Four constraints, each of which a plausible implementation gets wrong:

        * **A name already on the class goes through untouched**, and this branch is not a
          convenience. Without it the guard *preempts the descriptor protocol*: a read-only
          property would be reported as a missing attribute, which is the one thing it is not,
          and the near-match hint made that worse -- `solver.n_top = 1.5` came back "did you
          mean 'top'?", pointing a refractive index at a permittivity slot, which is exactly
          the quantity error `docs/conventions.md` exists to prevent. Delegating instead lets
          Python say `property 'n_top' of 'SolverNumpy' object has no setter`. The same branch
          keeps a method stubbable (`solver.solve = fake`, `monkeypatch.setattr(solver, ...)`),
          which the first version refused with a message claiming `solve` did not exist.
        * **`AttributeError`, not `TypeError`.** `tests/test_domain.py` asks for
          `AttributeError` on a read-only property, and a caller cannot tell the two refusals
          apart if the types differ. `TypeError` is what `_refuse_unknown_conditions` raises at
          the keyword door; it is the wrong type here.
        * **No instance state may be read.** This fires on `self.device = device`, the first
          statement of `CoreSolveMixin.__init__`, when `__dict__` is still empty.
          `_condition_names()` is a classmethod cached on the class, and `type(self)` reaches
          the class without touching the instance, which is why both are safe. It also has to
          live on the class that *owns* `_condition_names`: on `CoreSolveMixin`, which does not
          inherit it, `_BaseRCWA()` dies inside the guard on its own first assignment.
        * **No branch on the value.** `SolverJax._tree_unflatten` rebuilds through `__init__`,
          so under a jitted `solve_field` every slot arrives as a tracer. This branches on
          `name`, a `str`, and on nothing else.

        A leading underscore passes. Pinning the private slots by name instead was measured --
        35 of them -- and rejected: it buys nothing a caller can reach by accident and goes
        stale on every new slot. `solver._thicknes` is a way to break this, and it is spelled
        like one.

        **A subclass adding state of its own is refused, deliberately.** Declaring the name as
        a class attribute (`label = None` on the subclass) opts it in through the first branch,
        and a private name never needed to. The alternative -- allowing any new name on a
        subclass -- would make whether a name is checked depend on which class wrote it, which
        is not a rule anyone could predict.
        """
        if (name.startswith('_')
                or name in self._condition_names()
                or name in self._ASSIGNABLE_NON_CONDITIONS
                # Last, and it is the branch that keeps the descriptor protocol in charge:
                # a settable property is set as before, a read-only one raises Python's own
                # message, and a method can still be stubbed on an instance.
                or hasattr(type(self), name)):
            object.__setattr__(self, name, value)
            return

        raise _unknown_attribute(type(self).__name__, name, self._condition_names())

    def _refuse_unknown_conditions(self, kwargs):
        """Refuse a keyword that is not a condition, before anything is assigned.

        The name check of :meth:`_conditions`, which every `**kwargs` entry point goes through.
        These entry points take `**kwargs` so an optimizer can move a design variable and
        re-solve in one call, and every one of them used to end with a bare

            [setattr(self, k, v) for k, v in kwargs.items()]

        which accepts anything. A misspelled condition was stored as a brand-new attribute,
        the real one never moved, and the **unchanged** answer came back -- no exception, no
        warning, nothing in the result to tell it apart from a change that happened to do
        nothing. Measured: `solve(thicknes=[900.])` returned T = 0.851185066096, bit-identical
        to the base, while the real `thickness=[900.]` moves T by thirteen points.

        The constructor already refuses an unknown keyword by name (`_refuse_retired`), and
        `tests/test_domain.py` pins it with this exact typo. The decision was made; it just
        never reached the entry points a sweep actually calls.

        It also closes a sharper hole than a typo. `solve(_tetm='tm')` reached the private slot
        behind the property and returned the TM answer for a solver configured TE -- the clash
        check that refuses `tetm` beside a named incidence lives in the setter, and writing the
        slot goes around it. A private name is neither a constructor parameter nor a public
        property, so it is now simply unknown.

        It was one half of `_assign_conditions`, whose other half assigned the keywords and
        kept every one -- tracers included -- until the last caller, `solve_field`, moved to
        :meth:`_conditions` on 2026-09-24.
        """
        if not kwargs:
            return

        legal = self._condition_names()
        unknown = sorted(name for name in kwargs if name not in legal)
        if unknown:
            hints = []
            for name in unknown:
                # `near_match`, not difflib: `solve(n_top=1.5)` was pointed at `top`.
                near = near_match(name, legal, cutoff=0.7)
                hints.append(f'{name!r}' + (f' (did you mean {near!r}?)' if near else ''))
            raise TypeError(
                f'unexpected keyword argument(s): {", ".join(hints)}.'
                f'{nvf_option_rename_hint(unknown)}{quantity_note(unknown, legal)} '
                f'These entry points take '
                f'conditions to set before solving, and a name that is not one used to be '
                f'stored as an attribute and silently ignored -- the solve then answered the '
                f'scene you had, not the one you asked for. Settable conditions: '
                f'{", ".join(sorted(legal))}.')

    @_contextlib.contextmanager
    def _conditions(self, kwargs):
        """Assign the keyword conditions for one call, and put every traced value back.

        The entry points take conditions so an optimizer can move a design variable and
        re-solve in one call, and the value has to be *on the solver* while the call runs --
        `order_kxy`, the geometry, the field build all read it there. What it must not do is
        stay: a ``jax.jit`` tracer assigned by ``solve(lattice=L)`` outlived the call in
        ``solver.lattice``, and everything eager after it read a dead value. Measured on the
        sheared 500 x 700 box and the lone 350 box after the traced-basis refusal of
        2026-09-06, following the recipe that refusal prints: ``freeze_geometry()`` took its
        record off the tracer and recorded ``gamma_plans = [None]`` in silence, the jitted
        replay with that record was refused again with the same message, and every eager
        ``solve()`` on the solver raised ``UnexpectedTracerError`` until the lattice was
        reassigned by hand. The same leak predates the refusal -- after a *successful* jitted
        ``solve(lattice=L)`` the eager solves raised too -- but nobody was told to freeze
        afterwards then.

        The first fix put back the cell and the structure slots by name and left the scalar
        conditions, on the argument that a thickness, a wavelength or an angle is read by no
        eager decision before the next keyword overwrites it. The eager ``solve()`` *is* such
        a decision: after ``jax.jit(lambda w: solver.solve(geometry=g, wavelength=w))(900.0)``
        the plain ``solver.solve()`` raised ``UnexpectedTracerError`` from the tracer in
        ``solver.wavelength`` -- the same failure on a different slot -- and so did
        ``thickness`` and ``theta``. And the snapshot was taken outside the ``try``, so a
        setter that raised after the traced lattice was assigned (``solve(lattice=L,
        structure='junk')`` inside a trace) left the tracer on the solver with the ``finally``
        never reached.

        So the keywords are assigned one at a time with the instance ``__dict__``
        snapshotted around each, and every slot whose setter left it holding a tracer
        (:func:`_holds_tracer`, structural) -- whatever the slot is called: the private store
        behind a property, a plain attribute, a slot the setter re-derives -- is restored in
        the ``finally`` to the object it held before; a slot the setter created is deleted.
        The test is on what the slot *holds*, not on what came in, because the two differ
        under a trace: ``thickness=[310.0]`` arrives as a concrete list and is stored by
        ``as_array`` as a staged constant -- a tracer -- when the JAX setter runs inside
        ``jax.jit``, and left the same ``UnexpectedTracerError`` on the next eager solve.
        Slots are keyed by identity, and the first snapshot wins, so a slot two setters touch
        goes back to what it held before either.

        **What goes back is the setter's whole write, not the tracer alone.** A setter that put
        a tracer where there was none has every slot it changed put back with it, concrete ones
        included:
        the flag ``psi`` latches beside the angle (``_psi_given``), the path a ``structure``
        selects (``_modeling_path``), the cell a raster declared (``_structure_drawn_on``). The
        tracer alone used to go back, which left the solver half in each scene. Measured
        2026-09-24: after a jitted ``solve(psi=p)`` the angle read its default while
        ``_psi_given`` said it was given, so ``solver.polarization = (1, 1j)`` was refused as
        naming the incidence twice. A ``jax.grad`` of ``solve(structure=shapes)`` on a raster
        solver put the raster back under ``_modeling_path = PARAMETRIC``, and every eager
        ``solve()`` after it raised ``TypeError``. A traced raster through a solver whose
        structure came from ``Lattice.drawn`` dropped the declaration, so a moved cell was
        answered where it is refused. A setter that only *moves* a tracer an earlier keyword put
        there -- ``device``, which carries every slot to the new device -- is not one of these:
        its own write is concrete and stays, and only the traced slots go back, as before.

        The assignment itself runs inside the ``try``: a setter that raises still has every
        traced write before it put back. A concrete one stays, as it would have had the call
        gone on -- the call failing does not make it traced. Read by the diff rather than listed
        by name because a list goes stale on every new slot -- the first fix's two tuples missed
        ``wavelength`` the day they were written.

        A value that is concrete once stored (a float, a NumPy or backend array, a torch leaf
        under autograd; anything at all outside a JAX trace) is assigned and stays, exactly
        as before, in the order the keywords were given. Nothing traced outlives the call.
        """
        absent = object()
        held = {}
        device_before = self._device
        try:
            self._refuse_unknown_conditions(kwargs)
            for name, value in kwargs.items():
                before = dict(self.__dict__)
                setattr(self, name, value)
                changed = [slot for slot in before.keys() | self.__dict__.keys()
                           if self.__dict__.get(slot, absent) is not before.get(slot, absent)]
                traced = [slot for slot in changed
                          if _holds_tracer(self.__dict__.get(slot, absent))]
                # Its whole write when it put a tracer where there was none; only the traced
                # slots when it moved one an earlier keyword put there -- `device` does that,
                # and its own write is concrete and stays.
                introduced = any(not _holds_tracer(before.get(slot, absent)) for slot in traced)
                for slot in (changed if introduced else traced):
                    held.setdefault(slot, before.get(slot, absent))
                    if slot in ('_top', '_bot', '_structure'):
                        # A traced material and its dispersion definition are one assignment.
                        source = slot + '_dispersion'
                        held.setdefault(source, before.get(source, absent))
            if self._device != device_before:
                # `solve(device=X)`: the decorator opened the scope of the device this solver
                # was on, before the keyword was assigned. The new device's opens inside it.
                with self._device_scope():
                    yield
            else:
                yield
        finally:
            # A slot put back after `device` moved the solver goes back on the device it is on
            # now: its snapshot was taken before the move. Put back as it was, a jitted
            # `solve(device=X)` left the solver reporting X over the old device's arrays, and
            # every eager solve after it raised `Received incompatible devices`.
            moved = self._device != device_before
            with self._eager():
                for slot, value in held.items():
                    if value is absent:
                        self.__dict__.pop(slot, None)
                    else:
                        object.__setattr__(self, slot, self._moved(value) if moved else value)

    def _raster_types(self):
        """The array types a raster may arrive as: NumPy's, plus this backend's own.

        NumPy's is universal -- every backend takes a NumPy raster -- and `_ARRAY_TYPES` is the
        per-backend half, already declared for the geometry setters. The two `structure`
        setters that were not NumPy's differed from it in exactly this tuple.
        """
        return (_np.ndarray,) + self._ARRAY_TYPES

    @property
    def structure(self):
        """The layer stack, in whichever of the two encodings it was given in.

        A raster is an ``(n_layers, Ny, Nx)`` array of **relative permittivity**; the parametric
        spelling is a per-layer list that names a
        :class:`~wavebit._core.domain.Material` per region. Assigning it settles
        :attr:`modeling_path`, which is the only thing that selects the modeling
        path -- there is no separate mode switch.
        """
        return self._structure

    @structure.setter
    def structure(self, structure):
        """Settle the encoding and the modeling type together, or leave both alone.

        Every branch decides into locals and the pair is written once, at the end. It used to
        write as it went, and there are three ways to raise after the first write: the raster
        coercion, and both of ``_revalidate_mu``'s refusals. A caller who caught any of them
        was left holding the state the refusal exists to prevent -- measured, assigning a shape
        recipe over a raster that had a ``mu``: ``ValueError`` raised, and afterwards
        ``modeling_path == PARAMETRIC`` with ``mu`` still attached, which is verbatim the
        silently-dropped knob :data:`_MU_ON_A_SHAPE_PATH` is worded to stop. The mismatched-grid
        refusal was worse: it left a ``(1, 2, 8)`` permittivity beside a ``(1, 2, 4)``
        permeability and the next ``solve()`` *answered*.

        ``_revalidate_mu`` still runs after the write, because it checks the new pair and has
        to see it. So the write is undone if it refuses. Rolling back is not the same as
        checking first: the rules for what a ``mu`` may sit beside belong to the ``mu`` setter,
        and a copy of them here would be a second thing to keep in step -- which is the reason
        ``_revalidate_mu`` re-assigns rather than re-checking inline, given in its own docstring.
        """
        # `Lattice.drawn(...)` arrives here and is taken apart on the spot: the array goes on
        # exactly as it always did and the basis it was drawn on goes in a slot beside it. The
        # wrapper never travels past this line -- no backend, builder or document sees it, which
        # is the whole reason it is a wrapper rather than an ndarray subclass (a subclass
        # survives only NumPy; jax and torch copy the numbers and drop the attribute).
        drawn_on = None
        dispersion = None
        if isinstance(structure, _lattice.DrawnOn):
            structure, drawn_on = structure.values, _np.asarray(structure.lattice, dtype=float)

        if isinstance(structure, self._raster_types()):     # Raster
            kind, value = ModelingPath.RASTER, self._coerce_raster('structure', structure)
        elif type(structure) is list:
            # Analytic: one dict(background=..., shapes=[...]) per layer. Shape
            # coordinates are absolute lengths in the units of `period`, not fractions
            # of it. Shapes that overlap paint, later over earlier, and are routed to
            # the boundary builder by the overlap itself. See wavebit._core.parametric.
            # The block spelling -- a per-layer `[background, [[shape, ...], ...]]` -- used
            # to be the `else` of this line, so every list that was not the parametric one became
            # a block. That path is gone; the shape it accepted has to be refused by name here,
            # or it falls through to the parametric builder and dies as
            # `TypeError: unhashable type: 'list'` from somewhere with no bearing on the input.
            if not structure or not isinstance(structure[0], dict):
                raise ValueError(
                    'structure: a list of layers has to be the parametric spelling -- one '
                    'dict(background=..., shapes=[...]) per layer. The other list spelling, a '
                    'per-layer [background, [[shape, ...], ...]], was the block modeller, and '
                    'that path is gone: it decomposed shapes into axis-aligned rectangles and '
                    "applied Li's rule to them, which measured bit-identical to a raster under "
                    'the same rule on a commensurate 1D grating and behind NVF everywhere '
                    'else. Write the shapes as dicts for the parametric path, or sample them '
                    'onto a raster.')
            resolved = self._structure_at(structure, self.wavelength)
            dispersion = structure if resolved is not structure else None
            kind, value = ModelingPath.PARAMETRIC, self._placed_layers(resolved)
        elif structure is None:
            kind, value = None, None
        else:
            raise ValueError(
                f'structure: expected a raster array, a list of layers, or None -- got '
                f'{type(structure).__name__}')

        previous = (self._structure, self._modeling_path)
        self._structure, self._modeling_path = value, kind
        self._structure_drawn_on = drawn_on
        try:
            # `mu` and `structure` are one raster pair and either can be assigned second, so
            # the pair is checked from whichever side completes it -- see `_revalidate_mu`.
            self._revalidate_mu()
        except Exception:
            self._structure, self._modeling_path = previous
            raise
        self._structure_dispersion = dispersion

    def _coerce_raster(self, where, grid):
        """A raster field on this backend at the solve's precision.

        Shared by ``structure`` and ``mu`` -- they are two fields of the same shape sampled on
        the same grid, so a second copy of this would be a second chance for them to disagree
        about dtype.

        The dtype sets are built from the namespace rather than listed, which reproduces each
        backend's own set exactly: ``xp.float64`` under Torch *is* ``torch.float64``, and
        NumPy's four are accepted everywhere because a NumPy raster is accepted everywhere.

        ``copy=True`` is the one deliberate behaviour change in the fold, and it settles a
        divergence rather than introducing one. NumPy copied the caller's array
        (``grid.astype``) and Torch copied it (``torch.tensor``), but JAX's ``jnp.asarray``
        zero-copies from a NumPy buffer *sometimes*: five identical trials gave
        ``[False, True, False, True, False]`` for "did a later mutation of the caller's array
        reach ``solver.structure``". Not aliasing and not copying -- either would be a contract.
        It is the same rule the geometry setters already keep: a NumPy array you pass in is
        copied, a native array you pass in is kept, and the second half is what lets a Torch
        optimizer's in-place step reach the solver it configured.

        Torch's copy used to be ``_as_tensor``, a second copy of :func:`as_array` kept alive
        only for this caller. Both are gone.
        """
        xp = self.xp
        if not isinstance(grid, self._raster_types()):
            raise ValueError(
                f'{where}: expected one of '
                f'{", ".join(t.__name__ for t in self._raster_types())} on this backend, got '
                f'{type(grid).__name__}')
        if grid.dtype in (xp.complex128, xp.complex64, _np.complex128, _np.complex64):
            dtype = self.complex_dtype
        elif grid.dtype in (xp.float64, xp.float32, xp.int64, xp.int32,
                            _np.float64, _np.float32, _np.int64, _np.int32):
            dtype = self.float_dtype
        else:
            raise ValueError(f'{where}: unsupported dtype {grid.dtype}')
        return as_array(xp, grid, dtype=dtype, device=self._array_device, copy=True)

    # ------------------------------------------------- the solve, and the field it leaves
    #
    # One copy each. With the comments stripped, NumPy's and Torch's bodies were **byte
    # identical** -- 40 lines and 55 lines, same sha -- and JAX differed by `@device_scoped`
    # and by naming its own two `Result` subclasses. Nothing else. The three sets of comments
    # said the same things in different words and at different lengths; the fullest of each
    # pair survives here.

    #: The result containers this backend hands back. JAX overrides them with its two empty
    #: subclasses, which exist only to be registered as pytrees so a jitted method can return
    #: one. Declared rather than imported directly because that registration is the whole
    #: difference between the three bodies -- everything else about `solve_conv` and
    #: `reconstruct_field` is shared.
    #:
    #: Losing the override is caught, and loudly: dropping either line fails 14 tests across
    #: `test_solve_field_jax` and `test_pytree_round_trip`, because an unregistered class
    #: cannot cross a `jax.jit` boundary at all. Measured rather than assumed -- the danger a
    #: declared member usually guards against is the silent kind, and this one is not it.
    _RESULT = Result
    _RESULT_SUB = ResultSub

    @device_scoped
    def solve_conv(self, epx_conv_all, epy_conv_all, epz_conv_i_all,
                   epxy_conv_all=None, epyx_conv_all=None,
                   epxz_conv_all=None, epyz_conv_all=None,
                   epzx_conv_all=None, epzy_conv_all=None, mu=None,
                   mu_top=None, mu_bot=None, eps_top=None, eps_bot=None):
        """Solve a stack whose convolution matrices are already built.

        The nine permittivity slots in :meth:`eps_conv_matrices` order -- the in-plane ``(epx,
        epy, epz_i, epxy, epyx)``, then the z row ``(epxz, epyz, epzx, epzy)`` -- followed by
        ``mu`` and the half-space terms. ``solver.solve_conv(*solver.eps_conv_matrices())`` is
        ``solver.solve()``; see :mod:`wavebit._core.factorization` for why ``epz_i`` sits third
        rather than last. Every input slot is referenced at the cell centre, the frame
        :meth:`eps_conv_matrices` and :meth:`mu_conv_matrices` publish. Hand-built Fourier
        coefficients must use coordinates measured from that centre too. Each slot is
        restated once for the kernels as ``D C D``, ``D = diag((-1)**(m+n))`` by order label.
        Private builders return corner-frame matrices for :meth:`_solve_conv` instead.

        Takes no wavelength. The solver already has one, and a second spelling of it was a
        second thing that could disagree: the k vectors, the field reconstruction and the
        auto-detune ladder all read ``self.wavelength``, so an argument that differed would
        have answered one problem while the rest of the object described another.
        """

        # Reject a retained one-row layout's invalid order request before the
        # centre conversion checks the dimensions of matrices built earlier.
        if self.structure is None and self._grating_type == 0:
            self._refuse_y_orders_on_a_y_invariant_cell(True)
        window = self.order_window
        eps = swap_fourier_origin(
            (epx_conv_all, epy_conv_all, epz_conv_i_all, epxy_conv_all, epyx_conv_all,
             epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all), *window)
        mu = swap_fourier_origin(mu, *window)
        return self._solve_conv(
            *eps, mu=mu,
            mu_top=mu_top, mu_bot=mu_bot, eps_top=eps_top, eps_bot=eps_bot)

    @device_scoped
    def _solve_conv(self, epx_conv_all, epy_conv_all, epz_conv_i_all,
                   epxy_conv_all=None, epyx_conv_all=None,
                   epxz_conv_all=None, epyz_conv_all=None,
                   epzx_conv_all=None, epzy_conv_all=None, mu=None,
                   mu_top=None, mu_bot=None, eps_top=None, eps_bot=None):
        """Solve kernel-frame matrices, as returned by the private Fourier builders."""
        before = self._grating_type
        self._assign_grating_type()
        z_row = (epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all)
        four_field = self._formulation_for(z_row, mu) == Formulation.FOUR_FIELD
        if self._tetm is not None and (four_field or self._grating_type != 0):
            # Every route off the pure path reads an incidence pair `tetm` declared absent. The
            # routing goes back before the refusal: `reconstruct_field` dispatches on it, and
            # left at 2 it took the pure solve answered before this one for a 2D one (IndexError).
            routed, self._grating_type = self._grating_type, before
            self._refuse_tetm_off_the_pure_path(_z_row_given(z_row, mu), routed)
        if four_field:
            # The 1D paths have no z-row slots to fill, so a z row is also a choice of path:
            # the first-order solve is written on the 2D layout (a 1D structure is n_orders_v == 1
            # there, the same physics the conical path solves at conical cost).
            self._grating_type = 2

        # Every permeability question here is **structural** -- "was a slot named?", which is
        # what `Material.is_magnetic` answers -- and never "does the value differ from one?".
        # `None` means absent; any value, 1.0 included, means present. The value comparison is
        # not merely untidy: under `jax.jit` a named `mu` is a tracer, so `mu_top != 1.0`
        # produces a traced boolean and `or` raises TracerBoolConversionError on every traced
        # solve, magnetic and non-magnetic alike. `None` is an empty pytree subtree and comes
        # through a trace as itself, which is the whole reason it is how absence is spelled.
        #
        # The 1D and conical operators take three permeability slots, not five: they have no
        # off-diagonal position to put `mu_xy` / `mu_yx` in, exactly as they have none for
        # `eps_xy` / `eps_yx`. Refused by name rather than dropped -- an off-diagonal quietly
        # discarded returns a plausible *uniaxial* answer for a gyromagnetic layer. The
        # diagonal three route through, as of 2026-09-01; until then the whole feature was
        # refused here, and `fields.field_1d`'s `H_z` was correct *because* it was.
        if (self._grating_type != 2 and mu is not None
                and any(s is not None for s in tuple(mu)[3:])):
            raise NotImplementedError(
                'an off-diagonal permeability (mu_xy / mu_yx) was given and this structure '
                'routed to a 1D solve, whose operator has no off-diagonal slot -- the same '
                'reason it takes no eps_xy / eps_yx. The diagonal slots (mu_xx, mu_yy, '
                'mu_zz) do work here. Pass None in the last two slots for a layer that has '
                'no off-diagonal permeability -- absence is None, not a zero matrix, the '
                'same structural spelling every other slot uses. A tensor that genuinely '
                'needs the off-diagonals needs the 2D path, which a structure varying in y '
                'routes to.')

        if self._grating_type == 0:
            result_dict = self._solve_1d(epx_conv_all, epy_conv_all, epz_conv_i_all,
                                         mu=mu, mu_top=mu_top, mu_bot=mu_bot,
                                         eps_top=eps_top, eps_bot=eps_bot)
        elif self._grating_type == 1:
            result_dict = self._solve_1d_conical(epx_conv_all, epy_conv_all, epz_conv_i_all,
                                                 mu=mu, mu_top=mu_top, mu_bot=mu_bot,
                                                 eps_top=eps_top, eps_bot=eps_bot)
        else:
            result_dict = self._solve_2d(epx_conv_all, epy_conv_all, epz_conv_i_all,
                                         epxy_conv_all, epyx_conv_all, *z_row, mu=mu,
                                         mu_top=mu_top, mu_bot=mu_bot,
                                         eps_top=eps_top, eps_bot=eps_bot)

        # The boundary spectra and the per-order Cartesian vectors are one arithmetic
        # (`ordinary_spectra`), read off the packed amplitudes before any `ResultSub` exists, so
        # `r_xyz` / `t_xyz` ride in the result mappings like every other amplitude.
        from .field_result import cartesian_amplitudes, ordinary_spectra, ordinary_state
        keys = {'incident': 'incident', 's': 's_inc', 'p': 'p_inc'}
        spectra = ordinary_spectra(
            self, {name: result_dict.get(key) for name, key in keys.items()},
            eps_top=eps_top, eps_bot=eps_bot, mu_top=mu_top, mu_bot=mu_bot)
        if spectra is not None:
            for name, key in keys.items():
                if key in result_dict:
                    result_dict[key] = dict(result_dict[key], **cartesian_amplitudes(
                        spectra, name, result_dict[key]['r_s'].shape))

        incident = self._RESULT_SUB(**result_dict['incident']) if 'incident' in result_dict else None
        s_inc = self._RESULT_SUB(**result_dict['s_inc']) if 's_inc' in result_dict else None
        p_inc = self._RESULT_SUB(**result_dict['p_inc']) if 'p_inc' in result_dict else None

        result = self._RESULT(incident, s_inc, p_inc, order_window=self.order_window)
        result._field_state = ordinary_state(self, spectra)
        self._warn_complex_incidence()

        return result

    def _warn_complex_incidence(self):
        """Say once per solve that a complex incident ``kx`` / ``ky`` made R and T formal.

        Only a complex ``theta`` / ``phi`` gets here -- direct components are refused as
        non-real -- and :func:`~wavebit._core.power.power_for_real_k` has already set
        ``power_entering`` and ``absorption`` to NaN on the same test. What it cannot mark is
        R and T, which stay finite and read like efficiencies; see
        :class:`~wavebit._core.backend.ComplexIncidenceWarning`. Eager only: a traced solve has
        no value to test, the line every guard in this class draws.
        """
        import warnings
        from .backend import ComplexIncidenceWarning
        from .power import transverse_k_is_real
        k_inc_x, k_inc_y = self._incident_kxy(self.wavelength)
        xp = array_namespace(k_inc_x)
        if _concrete(transverse_k_is_real(xp, k_inc_x, k_inc_y)) is not False:
            return
        try:
            where = f' (kx/k0 = {complex(k_inc_x):.6g}, ky/k0 = {complex(k_inc_y):.6g})'
        except Exception:                      # a value that will not become a Python number
            where = ''
        warnings.warn(
            f'the incident transverse wavevector is complex{where}, from a complex theta/phi. '
            'Its intensity varies along the surface, so R and T are formal per-order ratios '
            'rather than fractions of an incident power -- their sum can exceed 1 on a '
            'lossless stack -- and power_entering and absorption are NaN. For an absorbing '
            'incident medium, give a real theta: the phase angle keeps kx and ky real. See '
            'docs/conventions.md#phase-incidence.',
            ComplexIncidenceWarning, stacklevel=3)

    #: The scene of the last solve: a tuple of numbers, ``None`` when a condition it reads was
    #: traced or under autograd at the solve, and this sentinel while no solve has run. The
    #: two non-scenes answer differently in `_reject_stale_field_scene`.
    _solved_scene = _UNSOLVED = object()
    #: The third non-scene: this solver's conditions moved in a call whose solve stayed inside
    #: it -- JAX's compiled `solve_field`. Set by `_forget_the_solve`.
    _SOLVED_ELSEWHERE = object()

    def _forget_the_solve(self):
        """Drop the solve state, because the scene it belonged to is no longer this solver's.

        For JAX's jitted ``solve_field``, which keeps a concrete keyword on the solver, as
        :meth:`solve` does, but solves on the copy of ``self`` its trace rebuilds -- so the
        modes, amplitudes and scattering matrix on the live solver are still the last eager
        solve's while the scene has moved. Walking them would reconstruct a field of neither
        scene. Clear the obsolete state even though ``_reconstruction_scene`` also guards
        reconstruction against changed conditions. Emptied rather than refreshed: the compiled
        call's own state is a traced value inside it, and bringing it out would put it on the
        live solver under an outer ``jax.jit``. JAX's ``fit`` also discards retained state:
        its last traced solve precedes the final optimizer update, and cannot be used to
        reconstruct the final design, including after a failed iteration.
        """
        self.layer_info_list = []
        self.T1 = None
        self._scattering = None
        self._solved_scene = self._SOLVED_ELSEWHERE

    def _reconstruction_scene(self):
        """The live conditions a reconstruction reads, as concrete numbers -- or ``None``.

        `reconstruct_field` walks the modes and amplitudes the last solve left behind, but it
        re-reads `wavelength`, `thickness`, the cell, the incidence angles, `incidence_side`,
        the half-space media and the retained order window from the solver as it goes. So a
        condition changed after the solve and before the reconstruction produces a field that
        belongs to *neither* scene.

        ``None`` when any of it is traced: concreteness is not a tracer's to prove, and a jitted
        reconstruction cannot compare anyway -- the same rule `StaticGeometry`'s staleness
        checks follow, and for the same reason.
        """
        try:
            lattice = self.lattice
            def real_component(value):
                if value is None:
                    return None
                return float(value.detach() if hasattr(value, 'detach') else value)
            return (
                complex(self.wavelength),
                tuple(complex(v) for v in _np.asarray(self.thickness).reshape(-1)),
                None if self.theta is None else complex(self.theta),
                None if self.phi is None else complex(self.phi),
                real_component(self.kx),
                real_component(self.ky),
                tuple(complex(v) for v in _np.asarray(self.period).reshape(-1)),
                None if lattice is None else tuple(
                    complex(v) for v in _np.asarray(lattice).reshape(-1)),
                self.incidence_side,
                self.tetm_index,
                # The width, because narrowing it re-casts the *conditions* and leaves the
                # solve's `T1` and `layer_info_list` at the width they were computed in. That
                # is a mix too. Held explicitly rather than left to fall out of the numbers:
                # `complex_dtype=1` moves `theta=0.1` and leaves `theta=0.0` alone, so without
                # this the refusal would depend on whether a value happens to survive float32.
                self._complex_dtype,
                # The device, for the width's reason: a move carries the conditions over, and a
                # field is a claim about one solve on one device -- the moved solver's next
                # solve is the one that answers there. As text, so a device's own equality
                # (a JAX list, a Torch object) never decides a refusal.
                str(self._device),
                # The half-space media. The incident wavevector is read live -- by `order_kxy`
                # and by the centre phase -- through the lit medium's index, so a `top` changed
                # after the solve returned a field of neither scene (a 1D TE grating, `top`
                # 1.0 -> 1.3: 0.19 from the field before and 0.11 from the re-solved one). Both
                # media and both constants, because either is the scene.
                tuple(None if value is None else
                      tuple(complex(v) for v in _np.asarray(value).reshape(-1))
                      for medium in (self._top, self._bot)
                      for value in (medium.epsilon, medium.mu)),
                # The retained window, which `order_kxy` also reads live. An `excitation` whose
                # keys move its centre keeps its size, so no shape mismatch catches it: the modes
                # are read one order off (the same grating, `excitation` None -> {1: 1.0}: 1.87
                # from the field before, 1.76 from the re-solved one).
                self.order_window,
            )
        except (TypeError, ValueError, RuntimeError):
            # Three ways a condition declines to be a number, and all three mean the same
            # thing here. `TypeError` is a JAX tracer refusing `complex()`; `ValueError` an
            # array that is not a scalar; `RuntimeError` is Torch's "Can't call numpy() on
            # Tensor that requires grad", which is a *differentiated* leaf rather than a
            # traced one -- the case the whole optimizer path runs in, and the one that
            # taught this line its third exception.
            return None

    def _reject_stale_field_scene(self):
        """Refuse a reconstruction of a scene that was never solved.

        Measured on a 1D grating: solve, then move `wavelength` from 900 to 600 without
        re-solving, then reconstruct. The field **moved** -- max|d| = 1.105 against the first
        reconstruction, so the new wavelength was read -- and it did **not** match the field
        after actually re-solving either, max|d| = 0.582. It belonged to no scene at all, while
        the docstring below promises "the state left behind by the last solve".

        Eager only, like every other staleness check here. Under a trace both keys are ``None``
        and the comparison is skipped.

        **A solve whose scene could not be read, reconstructed on a solver that reads now, is
        refused too.** ``solve()`` puts a traced cell back after its call (:meth:`_conditions`),
        so after ``jax.jit(lambda L: solver.solve(lattice=L))`` the modes and amplitudes
        belong to the traced cell and ``solver.lattice`` is the concrete one it held before --
        the same "neither scene" mix as a wavelength moved after the solve, reached from the
        other side. The old comparison skipped it (one key ``None``), and inside a trace the
        reconstruction would have read the put-back cell in silence. Now the record of a
        traced solve against a concrete live scene is a refusal that names the two ways out:
        ``solve_field``, which solves and reconstructs in one traced call and assigns its
        conditions outside ``solve()``, or an eager solve first. A torch leaf under autograd
        reads the same way -- its scene is ``None`` at the solve -- and reconstructs as before
        while it stays on the solver, since the live scene is ``None`` then too.
        """
        if self._solved_scene is self._UNSOLVED:
            return
        if self._solved_scene is self._SOLVED_ELSEWHERE:
            raise RuntimeError(
                'nothing to reconstruct from: the last call that moved this solver\'s conditions '
                'was a jitted solve_field(...) or fit(...) call. These compute inside a '
                'jax.jit trace, so this solver holds no modes or '
                'amplitudes for the scene it now reports. Call solve() to reconstruct from this '
                'solver, or take the field a solve_field(...) call returns.')
        now = self._reconstruction_scene()
        if self._solved_scene is None:
            if now is None:
                return
            raise RuntimeError(
                'the solve whose state this would reconstruct from was posed with a condition '
                'that could not be read as a number -- a traced cell, wavelength or angle, or '
                'a leaf under autograd -- and this solver holds concrete conditions now, so '
                'the modes and amplitudes belong to a scene it no longer describes. `solve()` '
                'puts a traced cell or structure back after its call, which is one way here. '
                'Reconstruct inside the traced call with solve_field(..., geometry=...), which '
                'solves and reconstructs the one scene, or call solve() again eagerly first.')
        if now is None or now == self._solved_scene:
            return
        raise RuntimeError(
            'the scene has changed since the solve whose state this would reconstruct from. '
            'The modes and amplitudes are the old scene\'s while the wavelength, thickness, '
            'cell, incidence, half-space media and retained orders are read live, so the '
            'field would belong to neither -- '
            'measured at max|d| = 1.105 against the field before the change and 0.582 against '
            'the field after re-solving. Call solve() again first.')

    @device_scoped
    def reconstruct_field(self, sample_x=20, sample_y=20, sample_z=20, incidences=('incident',)):
        """Reconstruct the field from the state left behind by the last solve.

        ``sample_x`` / ``sample_y`` / ``sample_z`` each take either a **count** -- a uniform
        grid -- or the **coordinates
        themselves**, as a 1-D array. The reconstruction is pointwise in all three, so
        explicit points cost the same per point as grid ones and involve no interpolation.
        They are named for taking either; a count is one way to say where to sample.

        ``sample_x`` / ``sample_y`` are lengths in ``period``'s units, measured from the
        **centre of the cell** -- the origin a parametric shape's ``cx`` uses since 2026-08-31 --
        and are otherwise unconstrained; a count spans the cell, which is ``[-period/2,
        +period/2]`` in that frame and the same physical grid it always was.
        ``sample_z`` is absolute, measured upward from the bottom of the stack, must lie inside it, and
        assigns an interface to the layer above it. The normal components ``E_z`` and ``H_z``
        may jump across material interfaces. With explicit ``sample_z`` the returned z axis is
        exactly the request, in the caller's order, rather than ``sample_z * n_layers``.

        ``incidences`` names which of the incidences the solve carries to reconstruct, one
        leading axis entry each, **in the order asked for**: ``'incident'`` is the incidence the
        solve was posed with (``psi``, or the ``polarization`` pair), and ``'s'`` / ``'p'``
        are the pure variants the coupled paths produce for free -- the same three the result
        object calls ``incident`` / ``s_inc`` / ``p_inc``. It was a positional triple of booleans
        (``set_field_input``) until 2026-08-20; booleans are now refused by name, since a
        reconstruction of the wrong incidence is a complete and plausible field cell.

        On the pure 1D TE/TM path only ``('incident',)`` is answerable -- ``tetm`` declared the
        other degree of freedom away, so that solve carries one incidence and the cell it
        returns has no axis to put a second on. Naming ``'s'`` or ``'p'`` there is refused
        rather than ignored, which is what it was until 2026-08-20.

        The field is the one the solve's incidence makes, **unit at ``x = y = 0`` on the lit
        face**: an ordinary solve's field is the ``PlaneWave`` field of the same angles, and an
        ``excitation``'s the ``OrderExcitation`` field of the same weights. That has been so
        since 2026-09-27; before, the incidence was unit at the cell corner, a factor
        ``exp(1j * k_inc . (a1 + a2)/2)`` away wherever ``k_inc`` is not zero.

        Takes no ``geometry``: it never calls ``solve``, and every discrete decision it
        makes -- the grating type, and which incidences ``incidences`` selects -- is
        already concrete by the time it runs. It reads ``self.T1`` / ``self.layer_info_list``,
        so whichever geometry the preceding solve used is the one in force here. Those two are
        the kernels' own state, in their own frame -- the cell corner is their Fourier origin --
        and describe the public field divided by :meth:`_centre_phase`, which is put back here.

        Explicit ``sample_z`` is the one thing here that cannot be traced: bucketing a z into a
        layer needs the thicknesses as numbers. It raises rather than guessing.

        After JAX's jitted ``solve_field`` kept a keyword, there is no state to reconstruct
        from: that call solved inside its own trace, and the state before it belonged to the
        scene before the keyword. This refuses, naming :meth:`solve` and the field that call
        returned, until a solve runs again.
        """
        self._require_field_declaration()
        self._reject_stale_field_scene()
        # Top illumination walks downward, so restate public heights in that propagation
        # frame before assigning samples to layers. `backside` is the kernels' word -- their
        # 'bottom' port -- and since 2026-09-21 it means public TOP incidence; public bottom
        # incidence walks the scene as given (see `_kernel_incidence_side`).
        backside = self._kernel_incidence_side == 'bottom'
        counted_z = is_count(sample_z)
        if backside:
            sample_z = to_mirrored_z(sample_z, self.thickness)
        kx, ky = self.order_kxy()

        if self._grating_type == 0:
            sample_y = 1
            field_cell = field_1d(self.wavelength, kx, self.T1, self.layer_info_list,
                                  self.lattice, self.tetm_index,
                                  sample_x=sample_x, sample_y=sample_y, sample_z=sample_z,
                                  incidences=incidences,
                                  interface_side='left' if backside else 'right')
        elif self._grating_type == 1:
            field_cell = field_1d_conical(self.wavelength, kx, ky, self.T1,
                                          self.layer_info_list, self.lattice,
                                          sample_x=sample_x, sample_y=sample_y,
                                          sample_z=sample_z, incidences=incidences,
                                          interface_side='left' if backside else 'right')
        elif self._formulation_assigned == Formulation.FOUR_FIELD:
            # The grating type cannot answer this: the first-order path sets it to 2 as well,
            # so branching on it alone would hand a 4ff solve to the 2ff reconstruction, which
            # reads the same state slots and returns a plausible, wrong cell. Static under jit
            # for the same reason the side flip below is -- the formulation is settled eagerly.
            field_cell = field_2d_4ff(self.wavelength, kx, ky, self.T1,
                                      self.layer_info_list, self.lattice,
                                      sample_x=sample_x, sample_y=sample_y,
                                      sample_z=sample_z, incidences=incidences,
                                      interface_side='left' if backside else 'right')
        else:
            field_cell = field_2d(self.wavelength, kx, ky, self.T1,
                                  self.layer_info_list, self.lattice,
                                  sample_x=sample_x, sample_y=sample_y,
                                  sample_z=sample_z, incidences=incidences,
                                  interface_side='left' if backside else 'right')

        # The kernels' state describes the field of their corner column (`_kernel_lit`), whose
        # incident wave is unit at the cell corner; the public incidence is unit at the centre.
        field_cell = field_cell * self._centre_phase()

        if backside:
            # The kernels' backside solve -- public top incidence -- is their top-side solve of
            # the z-mirrored scene, so its cell is in that mirror -- axis and components both.
            # `to_scene_frame` undoes both; see it for why the component half is not optional,
            # and `to_scene_frame`'s `flip_z` for why an explicit z has no axis left to
            # reverse. Static under jit either way: `incidence_side` is aux data, never a tracer.
            field_cell = to_scene_frame(field_cell, pure_1d=self._grating_type == 0,
                                        tetm_index=self.tetm_index, flip_z=counted_z)

        return field_cell

    def _solve_and_field(self, sample_x, sample_y, sample_z, incidences, geometry, **kwargs):
        """Solve, then reconstruct, sharing nothing with the caller but the arguments.

        The body all three backends had. Undecorated on purpose: both entry points below open
        the device scope themselves, and JAX's jitted `solve_field` would otherwise nest
        `jax.default_device` inside itself for no gain.
        """
        self._require_field_declaration()
        # An optimizer's loop sets a design variable and re-solves in one call, under the rule
        # `solve` keeps: a condition that is concrete once stored stays assigned, and every
        # slot a keyword's setter left holding a tracer is put back when the call returns. This
        # used to assign and keep everything, tracers included -- `jax.jit` (or `jax.grad`) of
        # `solver.solve_field_eager(..., wavelength=w)` left a tracer in `solver.wavelength`,
        # and every eager `solve()` after it raised `UnexpectedTracerError` (measured
        # 2026-09-24). That is the leak `solve` closed on 2026-09-06, still open one door over.
        # The reconstruction runs inside the same scope, so it reads the scene that was solved.
        with self._conditions(kwargs):
            result = self.solve(geometry=geometry)
            return result, self.reconstruct_field(sample_x, sample_y, sample_z, incidences)

    @device_scoped
    def solve_field(self, sample_x=20, sample_y=20, sample_z=20, incidences=('incident',),
                    geometry=None, **kwargs):
        """Solve and reconstruct the field in one call, eagerly.

        ``geometry`` is optional: omit it and each discrete decision is auto-detected from the
        concrete structure. It is accepted so that backend-agnostic code can hand over the same
        frozen geometry JAX's jitted override *requires* -- see
        :meth:`wavebit.backends.jax_backend.RCWAJax.solve_field`, which overrides this, and
        ``solve_field_eager`` there, which is this method under its other name.

        Naming the field here does not declare one: ``answers='field'`` still has to be on the
        solver, because it is the *solve* that either keeps the per-layer state or does not.
        Checked before that solve rather than after, so the refusal does not arrive on the far
        side of the expensive part.

        ``kwargs`` are assigned to the solver first, which is how an optimizer moves a design
        variable and re-solves in one call. NumPy did not accept them and the other two did,
        so the same sweep loop ran on two backends and raised ``TypeError`` on the third.
        They follow :meth:`solve`'s rule, on every backend and on JAX's jitted override too: a
        condition that is concrete once stored stays on the solver, and nothing traced does
        (:meth:`_conditions`). The one difference is the solve state. Here it is the call's
        own, so :meth:`reconstruct_field` afterwards reads the scene just solved; JAX's jitted
        override solves inside its compiled call, so a keyword it keeps drops the solver's
        earlier solve state instead, and :meth:`reconstruct_field` refuses until :meth:`solve`
        runs again (``docs/guide/fields.md``).
        """
        return self._solve_and_field(sample_x, sample_y, sample_z, incidences, geometry,
                                     **kwargs)

    @property
    def mu(self):
        """The raster path's relative permeability field — ``(n_layers, Ny, Nx)``, or ``None``.

        The companion of ``structure``, and the *only* place a permeability is written as a
        field. That asymmetry is the point of the whole material type: three of the four
        modeling paths state a layer as **assignments** ("this region is made of that"), and
        an assignment carries both quantities together as
        :class:`~wavebit._core.domain.Material`. The raster states a layer as a **sampled
        field**, one number per pixel, which is exactly why topology optimisation reaches for
        it -- and a field has one quantity in it, so a second quantity needs a second array.

        Naming it on a shape path is refused rather than ignored (:data:`_MU_ON_A_SHAPE_PATH`):
        there is no region for a layer-shaped array to attach itself to there, and a
        permeability the build silently dropped would be a knob with no reader.

        ``None`` is not ``1.0``. It means non-magnetic, and the solve takes the arithmetic that
        existed before ``mu`` did -- bit-identical, not merely equal.
        """
        return self._mu

    @mu.setter
    def mu(self, value):
        if value is None:
            self._mu = None
            return
        if self._modeling_path == ModelingPath.PARAMETRIC:
            raise ValueError(_MU_ON_A_SHAPE_PATH)
        field = self._coerce_raster('mu', value)
        # Same grid or nothing. The two fields are sampled on one raster and are transformed
        # layer by layer against each other, so a mismatch is not a broadcast to resolve --
        # it means the two describe different structures. Left to the builders it surfaces
        # much later as a shape error inside a Fourier transform, naming neither field.
        def grid_shape(a):
            return tuple(a.shape[:3]) if a.ndim == 5 and a.shape[-2:] == (3, 3) else tuple(a.shape)
        if self._structure is not None and grid_shape(field) != grid_shape(self._structure):
            raise ValueError(
                f'mu is {tuple(field.shape)} and structure is '
                f'{tuple(self._structure.shape)}: the permittivity and the permeability are '
                f'two fields sampled on one raster, so they share a shape exactly. Resample '
                f'whichever is coarser, or write the uniform one as a full grid.')
        self._mu = field

    def _revalidate_mu(self):
        """Re-run the ``mu`` setter's checks against a ``structure`` that just replaced one.

        The two fields are one pair, and **either** setter can be the one that completes it.
        The ``mu`` setter checks the pair when ``mu`` lands second -- the shape-path refusal,
        and the shared-grid shape -- and until this existed the ``structure`` setter checked
        nothing when ``structure`` landed second, so the same mistake was refused or accepted
        depending only on the order the two lines were written in. Assigning a shape recipe
        over a raster left ``solver.mu`` set and reaching no build: ``solve()`` returned
        results bit-identical to the same scene with no ``mu`` at all, which is exactly the
        silently-dropped knob :data:`_MU_ON_A_SHAPE_PATH` is worded to prevent.

        It re-*assigns* rather than re-checking inline on purpose: the rules for what a ``mu``
        may sit beside belong to the setter, and a second copy of them here would be a second
        thing to keep in step with the first. The cost is one re-coercion of a field that is
        already coerced, which is a value-preserving ``astype`` and only happens on a solver
        that named a ``mu`` in the first place.

        Called by each backend's ``structure`` setter, at the end -- after
        ``_modeling_type`` is settled, since that is the thing being checked against.
        """
        if self._mu is not None:
            # This only revalidates the existing value. Staging its coercion inside an
            # unrelated structure trace would hide otherwise concrete magnetic interfaces.
            with self._eager():
                self.mu = self._mu

    @property
    def fourier_coefficients(self):
        """How each layer's Fourier coefficients are obtained: ``'sampled'`` | ``'integrated'``.

        Resolved against the modeling path, exactly like :attr:`factorization` and for the same
        reason. Scalar shapes integrate their geometry. Tensor shapes also default to
        ``'integrated'``: analytic region indicators are combined with a sampled normal
        extension. Tensor shapes additionally accept ``'sampled'`` to rasterize the material
        geometry explicitly. Tensor rasters use ``'sampled'`` under NVF and support
        both coefficient methods under Li; scalar rasters support both.

        Raises:
            NotImplementedError: when the series was *explicitly* asked for and this modeling
                path cannot form it. Same exemption for the default, and for the same reason:
                ``'sampled'`` is the module-wide default, so a smooth-path user who never
                touched the keyword is not asking for anything and keeps the closed-form
                series that path always ran.
        """
        if self._has_tensor_materials():
            _anisotropy.validate_options(self)
            return ((FourierCoefficients.INTEGRATED if isinstance(self.structure,list)
                     else FourierCoefficients.SAMPLED) if self._fourier_coefficients is None
                    else self._fourier_coefficients)
        implemented = _COEFFICIENTS_IMPLEMENTED_BY.get(self._modeling_path)
        if implemented is None:                       # raster, or no structure yet: free choice
            return (DEFAULT_FOURIER_COEFFICIENTS if self._fourier_coefficients is None
                    else self._fourier_coefficients)
        if self._fourier_coefficients is not None and self._fourier_coefficients != implemented:
            # `!=` rather than `is not`, and `coefficients_of` in the message, for the reason
            # spelled out in `factorization`: this is a str enum, so a raw string written onto
            # the private attribute compares equal and must still format.
            raise NotImplementedError(
                f"fourier_coefficients="
                f"{coefficients_of(self._fourier_coefficients).value!r} is not available "
                f"here: {_COEFFICIENTS_REASON[self._modeling_path]}.")
        return implemented

    @fourier_coefficients.setter
    def fourier_coefficients(self, value):
        self._fourier_coefficients = coefficients_of(value)

    @property
    def factorization(self):
        """Which rule closes the discontinuous products: ``'li'`` | ``'nvf'`` | ``'laurent'``.

        Shapes default to NVF. Explicit Li uses exact Cartesian intervals for
        axis-aligned rectangles in a rectangular cell; other shapes are refused.
        Scalar rasters default to Li. Tensor inputs default to NVF and accept
        explicit Li on axis-aligned rectangular cells with no patterned z coupling.

        ``'laurent'`` is opt-in and never a default: the direct rule for every product,
        ``[[eps]]`` in both in-plane slots and ``[[eps]]^-1`` in the z slot -- the original
        RCWA formulation (Moharam & Gaylord, JOSA 71, 811, 1981), which converges slowly
        wherever a field component jumps with the permittivity (L. Li, JOSA A 13, 1870, 1996).
        It is here to compare with solvers that use it, fmmax's ``Formulation.FFT`` among them.
        Scalar rasters take it with either coefficient series, shapes take it on any shape --
        painted overlaps included -- and both take it on a skewed cell, since the rule names no
        axis. Tensor materials refuse it by name. See
        :func:`~wavebit._core.factorization.laurent_conv_matrices` for the slots.
        """
        if self._has_tensor_materials():
            _anisotropy.validate_options(self)
            return Factorization.NVF if self._factorization is None else self._factorization
        if self._modeling_path == ModelingPath.PARAMETRIC and self._factorization == Factorization.LI:
            _rectilinear.validate(self.structure, self.lattice)
            return Factorization.LI
        implemented = _FACTORIZATION_IMPLEMENTED_BY.get(self._modeling_path)
        if implemented is None:                       # raster, or no structure yet: free choice
            return DEFAULT_FACTORIZATION if self._factorization is None else self._factorization
        if self._factorization is None:
            return implemented[0]
        if self._factorization not in implemented:
            # `in` compares by `==`, not identity: `Factorization` is a str enum, so a value
            # written straight onto the private attribute compares equal to the member it
            # spells, and the formatting below tolerates one rather than dying inside the
            # error path.
            raise NotImplementedError(
                f"factorization={factorization_of(self._factorization).value!r} is not "
                f"available here: {_FACTORIZATION_REASON[self._modeling_path]}.")
        return factorization_of(self._factorization)

    @factorization.setter
    def factorization(self, value):
        self._factorization = factorization_of(value)

    @property
    def raster_normal_method(self):
        """How the raster NVF normal field is constructed: ``'distance'`` | ``'contour'`` |
        ``'spectral'`` | ``'smooth'``.

        Read only where a raster normal field is built — the raster path under
        ``factorization='nvf'`` (and the tensor-raster builders, called directly). Unlike
        :attr:`factorization` there is nothing to resolve per path: ``'distance'`` is the
        default everywhere, so this just reports it when nothing was asked. What *is*
        refused is an explicit request on a build that would never read it — see
        :meth:`_reject_unread_normal_method`, which fires from the same place the other
        refusals do, because only the build knows whether a normal field exists at all.
        """
        return DEFAULT_NORMAL_METHOD if self._normal_method is None else self._normal_method

    @raster_normal_method.setter
    def raster_normal_method(self, value):
        self._normal_method = None if value is None else normal_method_of(value)

    @property
    def nvf_projector(self):
        """NVF projection: ``'linear'`` (default) or ``'circular'``.

        Raster circular projection uses field magnitude. For shapes,
        circular projection lifts the sum of positive normal dyads into a complex
        basis. It avoids cancelling opposed normals through contact, and supports
        rotations and live geometry/material gradients. Disjoint circles and
        ellipses use a regular periodic level-set extension; curved paintings
        correct it near their polygon chords to follow the resolved geometry.
        Both are NVF rules;
        selecting a projector with Li factorization is refused.
        """
        return DEFAULT_PROJECTOR if self._projector is None else self._projector

    @nvf_projector.setter
    def nvf_projector(self, value):
        self._projector = None if value is None else projector_of(value)

    @property
    def li_nesting(self):
        """The order Li's rules are nested in: ``'fixed'`` | ``'symmetric'``.

        Named for its scope, as :attr:`nvf_projector` and :attr:`raster_normal_method` are:
        only Li reads it.

        ``'fixed'`` nests ``v`` inside ``u`` for every slot -- what the raster path built before
        the choice existed, and the default so that nothing measured on it moves.
        ``'symmetric'`` puts each slot's own inverse-rule axis inside (Li, JOSA A 14, 2758,
        1997), which makes the build covariant under relabelling x and y. See
        :class:`~wavebit._core.enums.Nesting` for the operators and
        ``docs/factorization-terms.md`` §6 for what separates them, measured.

        Read by scalar Li: the raster path under ``factorization='li'``, and the parametric
        path's axis-aligned rectangles under it. Like :attr:`raster_normal_method` there is
        nothing to resolve per path, so this reports the default when nothing was asked; what
        is refused is an explicit value on a build that would never read it
        (:meth:`_reject_unread_nesting`) -- any value under NVF or Laurent, and ``'symmetric'``
        on a tensor material, whose Li build is the fixed nesting only. The frozen geometry does
        not record it: the records a Li build reads -- the raster integrated path's
        compression, the rectangles' ``rectilinear_plans``, tensor Li's ``tensor_geometry`` --
        say where the blocks are, and the nesting only how they are combined, so one record
        serves both.
        """
        return DEFAULT_NESTING if self._nesting is None else self._nesting

    @li_nesting.setter
    def li_nesting(self, value):
        self._nesting = None if value is None else nesting_of(value)

    #: What `shape_nvf_grid_size=None` means, before the `4*truncation_order+1` floor lifts it.
    #:
    #: **1024, and it survived an attempt to drop it to 256.** On the forward value 1024 is
    #: plainly over-specified: 256 moves `sum(T)` by 1.4e-9 to 7.2e-7 on a circle, a tilted
    #: rectangle and a pair of circles at truncation_order 4 / 8 / 12, where the truncation
    #: error on those same rows is 4.2e-7 to 1.2e-4 -- one to four orders larger. Judged on
    #: values alone, this is 16x the pixels for the seventh digit.
    #:
    #: It is not judged on values. A *rotated rectangle's angle gradient* is what sets this,
    #: and it converges far later than the value does -- `d(sum T)/d(angle)` at 128 / 256 /
    #: 512 / 1024 / 2048 is 0.002050 / 0.002593 / 0.002342 / 0.002372 / 0.002364, so 256 is
    #: **9.7% off** the converged answer and 512 is 0.9% off. Non-monotone, too: the AD-vs-FD
    #: gap runs 1.3e-3 / 1.6e-2 / 3.4e-3 / 1.6e-3 / 4.4e-4 across those grids, so a coarser
    #: grid can look better than a finer one and no single check catches it.
    _NORMAL_GRID_DEFAULT = 1024

    @property
    def shape_nvf_grid_size(self):
        """Shape NVF projector sampling points per cell axis, resolved rather than ``None``.

        Reports the grid the *current* solve will use, the way :attr:`factorization` reports the
        rule it will use: a stated number is returned as stated, and ``None`` resolves against
        the truncation order. A caller that reads this to size something -- and there is one in
        the suite, dividing a period by it -- gets a number rather than the absence of one.

        Assigning ``None`` is how you ask for the default back after setting it.
        """
        want = self._normal_grid
        if want is not None:
            return want
        try:
            (_, hu), (_, hv) = self.order_window
        except Exception:                              # noqa: BLE001 - no order assigned yet
            return self._NORMAL_GRID_DEFAULT
        return self._resolve_normal_grid(hu, hv)

    @shape_nvf_grid_size.setter
    def shape_nvf_grid_size(self, value):
        self._normal_grid = value

    def _resolve_normal_grid(self, half_width_u, half_width_v):
        """`shape_nvf_grid_size`, with `None` meaning "enough, and not more".

        A *stated* number is taken as stated and checked against the aliasing floor by
        :func:`~wavebit._core.factorization.check_normal_grid`, which refuses below it. The
        default must never be the thing that trips that check, so it is resolved against the
        same floor rather than being a literal -- at `truncation_order=64` a flat 256 would
        refuse a solve nobody configured.

        **The default's stated reason was the right kind of reason and the wrong measurement.**
        It read "sets the floor on shape gradients (a translation-invariant structure should
        give d/dcx == 0 exactly; the sampling leaves a sawtooth instead -- 5e-4 at 256, 4.2e-5
        at 1024)". That *translation* number was a multi-shape one written as a general one: a
        lone shape is built at the centre of the cell and carried to its position by a phase
        (`parametric.normal_projector_coeffs`), so it was translation-exact at every grid, and
        since 2026-09-11 several shapes are built with their first shape at the centre and
        carried the same way. Peak-to-peak `sum(T)` over a one-pixel rigid translation, `truncation_order=6`:

            shape_nvf_grid_size        128       256       512      1024
            one shape      1.1e-13   7.0e-14   1.3e-13   8.1e-14      flat, machine precision
            two shapes     2.8e-06   6.4e-07   2.6e-07   1.3e-07      converged, until 2026-09-11
            two shapes     1.5e-13   2.9e-13   2.8e-13   9.8e-14      flat since (two 250 x 300
                                                                      boxes, the placement study's)

        But the conclusion that looks like -- that the default is over-specified and could be
        256 -- is wrong, and 256 was tried and reverted. It is a rectangle's **angle** that
        sets the grid, not anyone's position, and there 256 is 9.7% from the converged
        gradient. See `_NORMAL_GRID_DEFAULT` for the numbers. The general lesson is the one
        `docs/` already states about this field: a *value* cannot judge a normal field, and a
        default chosen on values will be chosen too coarse.
        """
        want = self._normal_grid
        floor = max(4 * half_width_u + 1, 4 * half_width_v + 1)
        if want is None:
            return max(self._NORMAL_GRID_DEFAULT, floor)
        return want

    def _reject_unread_normal_method(self, factorization):
        """Refuse an explicit ``raster_normal_method`` on a build that constructs no raster normal
        field — before it silently decides nothing.

        The keyword is read exactly when the raster path builds a normal field, which is
        ``factorization='nvf'`` on the raster path. Everywhere else — Li on a raster,
        either shape path — the build would run to completion with the request ignored,
        which is the same silent non-choice the ``enhanced_dfs`` parameter used to be
        (``docs/nvf-upsampling.md``): a knob whose reader does not exist. The default is
        exempt for the same reason as everywhere else: an untouched keyword is not a request.
        """
        if self._normal_method is None:
            return
        if self._modeling_path == ModelingPath.RASTER and factorization == Factorization.NVF:
            return
        if factorization == Factorization.LAURENT:
            where = _LAURENT_BUILDS_NO_NORMAL
        elif self._modeling_path == ModelingPath.PARAMETRIC:
            where = 'the parametric path builds its normal field from the geometry itself'
        else:
            where = "factorization='li' builds no normal field at all"
        raise NotImplementedError(
            f"raster_normal_method={normal_method_of(self._normal_method).value!r} has no reader "
            f"here: {where}. It selects the raster NVF normal-field construction, so it "
            f"needs a raster structure and factorization='nvf'.")

    def _reject_unread_projector(self, factorization):
        """Reject projector requests without an implementation on this structure."""
        if self._projector is None:
            return
        if self._modeling_path == ModelingPath.RASTER and factorization == Factorization.NVF:
            return
        if self._modeling_path == ModelingPath.PARAMETRIC and factorization == Factorization.NVF:
            if self.nvf_projector == 'circular':
                for layer in self.structure:
                    _parametric.check_dyadic_shapes(layer.get('shapes', []))
            return
        if factorization == Factorization.LAURENT:
            where = _LAURENT_BUILDS_NO_NORMAL
        elif self._modeling_path == ModelingPath.PARAMETRIC:
            where = "Cartesian shape factorization='li' builds no normal projector"
        else:
            where = "factorization='li' builds no normal field at all"
        raise NotImplementedError(
            f"nvf_projector={projector_of(self._projector).value!r} has no reader here: {where}. "
            f"It selects the NVF projection for raster or shape structures, so it "
            f"needs factorization='nvf'.")

    def _reject_unread_nesting(self, factorization):
        """Refuse an explicit ``li_nesting`` on a build that cannot honour it.

        Its readers are the two scalar Li builds: the raster path under ``factorization='li'``,
        and the parametric path's axis-aligned rectangles under it
        (:mod:`~wavebit._core.rectilinear`), which run on the same integrated builder. NVF takes
        the direct rule both ways and closes the discontinuity with the normal-field projector
        instead, and Laurent takes the direct rule and stops there, so under either there is no
        inner inverse for an order to matter to, and any explicit value is refused --
        :meth:`_reject_unread_normal_method` and
        :meth:`_reject_unread_projector` read the other way round. Tensor materials are the
        third case: tensor Li (:mod:`~wavebit._core.tensor_li`) builds the fixed nesting and
        nothing else, so an explicit ``'fixed'`` is honoured and ``'symmetric'`` refused. A
        scalar tensor gives exactly the scalar Li operator (``tests/test_tensor_li.py``), and
        answering a ``'symmetric'`` request with the fixed build would break that silently.
        Tensor scenes never reach :meth:`_check_before_building`, so
        :func:`~wavebit._core.anisotropy.validate_options` calls this for them. Separate from the
        NVF pair rather than folded into it so that a refusal names the keyword the caller
        wrote. The default is exempt, as everywhere: an untouched keyword is not a request.
        """
        if self._nesting is None:
            return
        nesting = nesting_of(self._nesting)
        if factorization == Factorization.LI:
            if nesting == Nesting.FIXED or not self._has_tensor_materials():
                return
            raise NotImplementedError(
                f"li_nesting={nesting.value!r} has no reader here: tensor factorization='li' "
                f"nests its rules in the fixed order only (v inside u for every slot). Leave "
                f"li_nesting unset or 'fixed' on a tensor material; the scalar Li builds read "
                f"both nestings.")
        if factorization == Factorization.LAURENT:
            where = ("factorization='laurent' nests no rules -- it takes the direct rule for "
                     "every product, so there is no inverse for an order to matter to")
        elif self._modeling_path == ModelingPath.PARAMETRIC:
            where = ("the parametric path's factorization='nvf' nests no rules; its "
                     "factorization='li' on axis-aligned rectangles is what reads it")
        else:
            where = ("factorization='nvf' nests no rules -- it takes the direct rule both ways "
                     "and closes the discontinuity with the normal-field projector")
        raise NotImplementedError(
            f"li_nesting={nesting.value!r} has no reader here: {where}. It selects the order Li's "
            f"rules are nested in, so it needs factorization='li'.")

    def _reject_unread_normal_grid(self):
        """Refuse an explicit ``shape_nvf_grid_size`` on a build that samples no normal field.

        Exactly the argument :meth:`_reject_unread_normal_method` makes for its sibling, and
        the asymmetry between them was not a decision. ``shape_nvf_grid_size`` is read by the shape
        paths and by nothing else: on a raster the field is built at the raster's own
        resolution (lifted to the aliasing floor), so ``shape_nvf_grid_size=64`` and
        ``shape_nvf_grid_size=4096`` return the same answer to the last bit, under Li *and* under NVF.
        Measured before this check: 0.898147036010 for both at 64 and at 4096.

        More urgently than for `raster_normal_method`, in fact: this one's own signature comment
        promises a raster user something about shape gradients it cannot deliver for them.

        The default is exempt for the same reason as everywhere else -- an untouched keyword is
        not a request -- which is what `None` now means (see :meth:`_resolve_normal_grid`).
        """
        if self._normal_grid is None:
            return
        if self._modeling_path == ModelingPath.PARAMETRIC:
            factorization = self.factorization
            if factorization == Factorization.NVF:
                return
            if factorization == Factorization.LAURENT:
                raise NotImplementedError(
                    f'shape_nvf_grid_size={self._normal_grid!r} has no reader here: '
                    f'{_LAURENT_BUILDS_NO_NORMAL}. It sizes the grid shape NVF samples its '
                    f"normal field on, so it needs factorization='nvf'.")
            raise NotImplementedError("shape_nvf_grid_size has no reader with Cartesian shape factorization='li'")
        raise NotImplementedError(
            f'shape_nvf_grid_size={self._normal_grid!r} has no reader here: the raster path builds its '
            f"normal field at the raster's own resolution, lifted to the 4*truncation_order+1 "
            f'floor, so this changes nothing. It sizes the grid the *shape* paths sample their '
            f'analytic normal field on, and needs a structure of shape dicts.')

    @device_scoped
    def freeze_geometry(self, **kwargs):
        """Take this structure's discrete geometry decisions once, eagerly.

        Captures whatever the active Fourier strategy decides from the concrete ``structure``.
        Hand the result back to :meth:`solve` as ``geometry=`` to make the call
        traceable. The default ('sampled', 'li') build decides nothing, so its frozen geometry
        is empty — passing it is still what keeps ``solve`` off the eager path.

        The record describes *this* structure. Replaying it onto one whose geometry has moved
        is refused rather than answered, eagerly — see :mod:`wavebit._core.geometry` for what
        each record checks and why a traced replay cannot.

        **A solver holding a tracer it cannot read is refused, by name.** The freeze is the
        eager pass, and a cell or a shape layer that is a ``jax.jit`` tracer gives it nothing to
        decide from: the record it took held ``gamma_plans = [None]`` and replayed blind, which
        is the guess it exists to avoid (:meth:`_refuse_a_freeze_on_a_tracer`). ``solve()``
        puts a traced cell or structure back after its call, so a solver reaches that refusal
        only when the tracer was assigned to it directly, inside a trace.
        """
        with self._conditions(kwargs):
            return self._freeze_geometry()

    def _freeze_geometry(self):
        """:meth:`freeze_geometry` past its conditions."""
        if self._has_tensor_materials():
            geometry = _anisotropy.freeze(self)
            # Stamped like the scalar records below, so one check covers both kinds of scene.
            geometry.factorization = self.factorization.value
            return geometry
        # A record of a stack that cannot be solved is a record of nothing, and it would be
        # replayed rather than rebuilt -- so the mismatch would surface later, from the replay,
        # further from the assignment that caused it. Only this one refusal, not all of
        # `_check_before_building`: whether a freeze should also refuse an unavailable rule is
        # a separate question about what a geometry record is for.
        self._reject_layer_count_mismatch()
        # Same two refusals as `_eps_conv_matrices`, in the same order, for the same reason: a
        # record of a build that cannot happen is worth nothing, and a structure that gets one
        # diagnosis from `freeze_geometry` and a different one from `solve` sends the
        # reader to the wrong place depending on which they called.
        self._reject_unsupported_lattice()
        factorization = self.factorization
        coefficients = self.fourier_coefficients
        self._reject_unread_normal_method(factorization)
        self._reject_unread_projector(factorization)
        self._reject_unread_nesting(factorization)
        self._reject_unread_normal_grid()
        # The *window's* half-width, not the declared `truncation_order`: an excitation can
        # widen the retained set, and a convolution matrix has to be as wide as the
        # `kx` it will meet. Its entries do not otherwise care where the window sits
        # -- `(i, j)` is the harmonic `i - j` -- so only the size travels here.
        (_, half_width_u), (_, half_width_v) = self.order_window
        basis = self._geometry_lattice_for(None)
        self._refuse_a_freeze_on_a_tracer(basis)

        # The two axes decide independently what has to be frozen: NVF needs the normal
        # field, the integrated series needs the block compression, and ('integrated', 'nvf')
        # -- the combination the old integer could not spell -- needs both. Li and Laurent
        # need no normal field; on a raster their records are the compression or nothing.
        normal_field = normal_field_source = compression = None
        mu_normal_field = mu_normal_field_source = mu_compression = None
        if self._modeling_path == ModelingPath.RASTER:
            if factorization == Factorization.NVF:
                normal_field = _fac.normal_field_for_raster(
                    self.structure, half_width_u, half_width_v, lattice=basis,
                    method=self.raster_normal_method, nvf_projector=self.nvf_projector)
                normal_field_source = _fac.normal_field_source_for_raster(
                    self.structure, half_width_u, half_width_v, lattice=basis, method=self.raster_normal_method)
                if self._mu is not None:
                    # A separate record, off `mu`'s own raster. A magnetic inclusion need not
                    # sit where the dielectric one does, and the NVF rearrangement is written
                    # around *one* interface normal, so each tensor gets the normal of its own
                    # -- the same argument `to_conv_mat_raster_pn_mu` makes. Sharing the
                    # permittivity's field would be right on every case where the two patterns
                    # coincide, which is most of them, and silently wrong on the rest.
                    mu_normal_field = _fac.normal_field_for_raster(
                        self._mu, half_width_u, half_width_v, lattice=basis,
                        method=self.raster_normal_method, nvf_projector=self.nvf_projector)
                    mu_normal_field_source = _fac.normal_field_source_for_raster(
                        self._mu, half_width_u, half_width_v, lattice=basis, method=self.raster_normal_method)
            if coefficients == FourierCoefficients.INTEGRATED:
                compression = [_fac.compression_indices(self.structure[i])
                               for i in range(self.structure.shape[0])]
                if self._mu is not None:
                    mu_compression = [_fac.compression_indices(self._mu[i])
                                      for i in range(self._mu.shape[0])]

        geometry = StaticGeometry(normal_field=normal_field, compression=compression,
                                  normal_field_source=normal_field_source,
                                  mu_normal_field=mu_normal_field,
                                  mu_normal_field_source=mu_normal_field_source,
                                  mu_compression=mu_compression,
                                  # What this record was taken under, so that a replay under
                                  # another rule is refused rather than read -- see
                                  # `_refuse_a_record_of_another_rule`.
                                  factorization=factorization.value)
        if self._modeling_path == ModelingPath.PARAMETRIC:
            if factorization == Factorization.LI:
                geometry.rectilinear_plans = [_rectilinear.plan_layer(layer.get('shapes', []), basis)
                                              for layer in self.structure]
                return geometry
            # The layer refusals run here, not only on the solve. This *is* the eager pass --
            # it is where the shapes are concrete by construction -- and a record taken over a
            # layer that would be refused is a record that lets the refusal be replayed past:
            # freeze a rod of `lx = 340` on a period of 300, hand the record to a traced solve,
            # and it builds, giving T = 0.9715 where the slab it actually describes gives
            # 0.9590 and a `dT/dlx` of 9.8e-4 where the true derivative is exactly zero. Both
            # `check_layer_shapes` docstrings named this call as the one that still sees them;
            # until 2026-08-27 it did not make it.
            for index, layer in enumerate(self.structure):
                _parametric.check_layer_spec(layer, index)
                _parametric.check_layer_shapes(layer, basis, index)
            # Only a layer that actually overlaps has a plan to freeze. One with nothing
            # overlapping means the same structure either way and stays on the closed-form
            # path, so its slot stays None and the replay never looks.
            # The plan records the basis reduction beside its loops (`boundary.plan_layer`),
            # taken here on concrete numbers so that a jitted replay enumerates its periodic
            # images on the same reduced cell.
            cell = self._geometry_lattice_for(None)
            plans = []
            for index, layer in enumerate(self.structure):
                if _parametric.layer_needs_resolving(layer, cell):
                    plans.append(_boundary.plan_layer(layer.get('shapes', []), cell))
                else:
                    plans.append(None)
            # Recorded whether or not anything needed a plan. The entry *is* the decision:
            # `None` here means "looked at while the numbers were concrete, and it does not
            # overlap". Setting the attribute only when something overlapped made those two
            # cases indistinguishable downstream, so a disjoint painted layer under a trace
            # had to re-take the overlap test it had already passed -- on values that can no
            # longer answer it.
            geometry.boundary_plans = plans
            if factorization == Factorization.LAURENT:
                # Laurent's rule reads the painting and nothing else: it builds no normal
                # field, so there is no interface set to plan. The stamp is what keeps this
                # record from reaching an NVF build, which would take the missing plans as
                # "plan blind" rather than as "never asked".
                return geometry
            # The interface set of every layer -- which faces lie on which image's faces over
            # a run, which are dead, whether anything is left -- by name, with no float in it.
            # Recorded for every layer whether or not the closed form builds it: a plan is
            # microseconds. It used to be `None` for a layer whose shapes were `jax.jit`
            # tracers at the freeze; that solver is refused above now, so every entry here is
            # a plan (a `jax.grad` primal plans, as it always did).
            geometry.gamma_plans = [_parametric.plan_gamma(layer.get('shapes', []), basis)
                                    for layer in self.structure]
        return geometry

    def _refuse_a_freeze_on_a_tracer(self, basis):
        """Refuse to take a record off a cell or a shape layer that is a tracer with no primal.

        A ``jax.grad`` tracer carries its primal and every geometry read below takes it
        (:func:`~wavebit._core.backend.checked_float`), so a freeze inside a gradient plans
        the same plan the eager call would. A ``jax.jit`` tracer has no number: the layer
        checks skipped, the painting question raised only for two or more shapes, and the
        interface set was recorded as ``None`` -- a record that says nothing and, handed to
        the jitted replay, was refused again with the very message that had prescribed it.
        Measured on the sheared 500 x 700 box and the lone 350 box after a jitted
        ``solve(lattice=L)`` left its tracer on the solver: ``gamma_plans = [None]``, the
        replay refused, the eager solves raising ``UnexpectedTracerError``.

        The parametric structure is asked the same question, for the same reason: a lone
        ellipse of traced size under ``jax.jit`` is refused by the blind plan since
        2026-09-06, and a record taken off the tracer would send its recipe round the same
        loop. A raster is not asked -- the NVF freeze cannot read a traced raster at all and
        says so on its own, and the sampled Li build decides nothing, so an empty record of
        it is an honest one.
        """
        try:
            _parametric._cell_floats(basis)
        except Exception:                                  # noqa: BLE001 - no primal to read
            what = 'cell basis (`lattice`, or the `period` it is spelled as)'
        else:
            what = None
            if self._modeling_path == ModelingPath.PARAMETRIC:
                for index, layer in enumerate(self.structure):
                    if _parametric._primitives(layer.get('shapes', []), basis) is None:
                        what = f'structure (layer {index} holds a traced shape parameter)'
                        break
        if what is None:
            return
        raise ValueError(
            f"freeze_geometry: this solver's {what} is a traced value with no number to read "
            f'-- a `jax.jit` tracer -- so there is nothing to take the geometry decisions from. '
            f'A record taken here would hold no plan and replay blind, which is the guess the '
            f'freeze exists to avoid. Freeze on a concrete solver, outside the trace, and hand '
            f'the record to the traced call:\n'
            f'    solver.lattice = A                   # or solver.structure = ..., as numbers\n'
            f'    geometry = solver.freeze_geometry()\n'
            f'    jax.jit(lambda L: solver.solve(lattice=L, geometry=geometry))(A)\n'
            f'`solve()` puts a concrete cell or structure back after a call that traced one, so '
            f'a solver holds a tracer here only when it was assigned directly, inside a trace.')

    @device_scoped
    def solve(self, geometry=None, auto_detune=True, **kwargs):
        """Build the convolution matrices, then solve.

        ``geometry`` is an optional :class:`~wavebit._core.geometry.StaticGeometry` — the
        discrete decisions frozen by :meth:`freeze_geometry`. Supply it to keep the whole call
        traceable under ``jax.jit``; omit it and each decision is taken from the values in
        front of it, which needs a number to read.

        **``jax.grad`` needs no freeze, since 2026-08-28.** "Concrete" is a property of the
        *value*, not of the framework: an un-jitted grad tracer carries a primal, so every
        decision can be taken at the point the gradient is being taken -- and a decision taken
        there cannot be stale, which a held one can. Under ``jax.jit`` and ``vmap`` there is no
        primal to read and the freeze is what stands in for it. PyTorch has always been on the
        readable side, and the two now agree to 4.0e-13 on a shape gradient.

        ``auto_detune`` is what happens when the requested point is numerically singular — an
        order exactly on a light line (``kx_m == n``, a Rayleigh point). On by default: if the
        solve refuses with a degeneracy (``LinAlgError`` / ``FloatingPointError``) or fires
        :class:`~wavebit._core.backend.RayleighPointWarning`, it is re-run up a two-rung
        ladder — ``wavelength * (1 + 1e-12)`` first, and if the condition survives (the
        ``m == 0`` order carries no wavelength) additionally ``theta * (1 + 1e-12)``, which
        moves every order at once. Off the point the answer is measured to agree with the
        one-sided limit to ~1e-6. :class:`~wavebit._core.backend.AutoDetuneWarning` states
        exactly which knobs were moved and the values actually used; the solver keeps them,
        so field reconstruction and everything else downstream describe one consistent
        problem. Nothing moves silently, which is what separates this from the
        ``perturbation`` knob this library removed. The knobs multiply, so a differentiable
        ``wavelength`` / ``theta`` keeps its autodiff graph through the retry — the gradient
        returned is the nearby problem's, which is also the only one that exists (the exact
        point is a cusp of R(lambda)). Pass ``False`` to solve exactly the requested point
        and let the refusal or the warning through; pass a float to choose the relative step
        (the default is sized for double precision — complex64 resolves nothing below ~1e-7,
        use ~1e-5 there). A traced solve (``jax.jit``) skips every guard and therefore never
        retries — the same line the guards themselves draw.

        **Nothing traced stays on the solver.** ``solve(lattice=L)`` inside ``jax.jit`` has
        to hold the tracer while it runs; it used to keep it afterwards, so the recipe the
        traced-basis refusal prints -- freeze, then replay -- froze a tracer and recorded no
        plan, and a jitted ``solve(wavelength=w)`` left every eager ``solve()`` after it
        raising ``UnexpectedTracerError``. Every slot a keyword's setter left holding a
        tracer -- the cell, the structure, a thickness, a wavelength, an angle -- is put back
        to what it held once the call returns, even when a later keyword's setter raises
        (:meth:`_conditions`); a value that is concrete once stored stays as assigned.

        Raises:
            StaleGeometryError: eagerly, when ``geometry`` no longer describes the structure it
                is being replayed onto. A traced call cannot check — the comparison is the
                thing the record exists to avoid — so a long fit should re-freeze on a schedule
                rather than wait to be told.
        """
        with self._conditions(kwargs):    # an optimizer moves a design variable here
            self._frozen_geometry = geometry
            return self._solve_detuned(geometry, auto_detune)

    def _solve_detuned(self, geometry, auto_detune):
        """:meth:`solve` past its conditions: the build, with the Rayleigh-point ladder around it."""
        if not auto_detune:
            return self._solve_built(geometry)

        import warnings
        from numpy.linalg import LinAlgError
        from .backend import AutoDetuneWarning, RayleighPointWarning
        step = _AUTO_DETUNE_REL if auto_detune is True else float(auto_detune)
        original = None
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                res = self._solve_built(geometry)
            if not any(issubclass(r.category, RayleighPointWarning) for r in rec):
                for r in rec:                  # not ours to handle: replay for the caller
                    warnings.warn_explicit(r.message, r.category, r.filename, r.lineno)
                return res
            cause = 'RayleighPointWarning fired'
        except (FloatingPointError, LinAlgError, RuntimeError) as exc:
            # RuntimeError only for the backend whose LinAlgError spells itself that way
            # (torch._C._LinAlgError); anything else is not a degeneracy and not ours to retry.
            if isinstance(exc, RuntimeError) and 'linalg' not in type(exc).__name__.lower():
                raise
            cause = f'the solve refused ({type(exc).__name__})'
            original = exc

        # The retry ladder. Wavelength first: it moves every m != 0 order and is the knob a
        # sweep is already varying. It cannot move the m == 0 order — kx_0 = n_top*sin(theta)
        # has no wavelength in it — so a condition that survives the first rung gets the
        # incidence angle nudged as well; theta moves every order at once. theta == 0 has no
        # second rung and needs none: kx_0 == 0 there, which grazes nothing. The knobs
        # multiply rather than being replaced, so a differentiable wavelength/theta keeps its
        # autodiff graph through the retry.
        knobs = ['wavelength']
        try:
            if self.theta is not None and bool(self.theta != 0):
                knobs.append('theta')
        except Exception:
            pass                               # unreadable theta: wavelength rung only
        saved = {k: getattr(self, k) for k in knobs}
        candidate = None if original is not None else (res, rec, [])
        applied = []
        for knob in knobs:
            setattr(self, knob, saved[knob] * (1 + step))
            applied.append(knob)
            try:
                with warnings.catch_warnings(record=True) as rung_rec:
                    warnings.simplefilter('always')
                    rung_res = self._solve_built(geometry)
            except (FloatingPointError, LinAlgError, RuntimeError) as exc:
                if isinstance(exc, RuntimeError) and 'linalg' not in type(exc).__name__.lower():
                    for k in applied:
                        setattr(self, k, saved[k])
                    raise                      # not a degeneracy: restore and surface as-is
                setattr(self, knob, saved[knob])   # this rung did not take
                applied.pop()
                continue
            candidate = (rung_res, rung_rec, list(applied))
            if not any(issubclass(r.category, RayleighPointWarning) for r in rung_rec):
                break                          # off the condition: this is the answer
        if candidate is None:
            # Every rung refused as well (applied is empty — each failure restored its knob).
            # The asked-for point is the honest thing to report, message and all.
            raise original from None

        res, rung_rec, moved = candidate
        for r in rung_rec:                     # the returned solve's own warnings, replayed
            warnings.warn_explicit(r.message, r.category, r.filename, r.lineno)
        cleared = not any(issubclass(r.category, RayleighPointWarning) for r in rung_rec)
        if moved:
            where = ', '.join(f'{k}*(1+{step:g}) = {getattr(self, k)!r}' for k in moved)
            tail = ('The solver keeps these values. Pass auto_detune=False to solve exactly '
                    'the requested point.' if cleared else
                    'The condition did not clear — it does not depend on the moved knobs; '
                    'the RayleighPointWarning above stands. Nudge an index or the period '
                    'instead.')
            warnings.warn(f'auto_detune: {cause}; re-solved at {where}. {tail}',
                          AutoDetuneWarning, stacklevel=2)
        else:
            warnings.warn(
                f'auto_detune: {cause}, and every detuned retry refused; returning the '
                'exact-point result, warnings and all. Nudge an index or the period instead.',
                AutoDetuneWarning, stacklevel=2)
        return res

    @device_scoped
    def eps_conv_matrices(self, geometry=None, **kwargs):
        """The permittivity convolution matrices for this structure, without solving.

        Returns a :class:`~wavebit._core.factorization.ConvMatrices` — the five slots ``(epx,
        epy, epz_i, epxy, epyx)``, which are exactly :meth:`solve_conv`'s leading parameters, so
        ``solver.solve_conv(*solver.eps_conv_matrices())`` is ``solve()``. It is a ``NamedTuple``:
        splatting, unpacking, indexing and ``len`` all behave as they did when this returned a
        bare tuple, and ``conv.epz_i`` now reaches one slot by name rather than by an index that
        is correct only until somebody reorders. Each slot is ``(n_layers, n_orders, n_orders)``
        with ``n_orders = (2*half_width_u+1) * (2*half_width_v+1)``; ``epxy`` and ``epyx`` are
        ``None`` on the paths that produce three rather than five (see
        :func:`~._core.factorization.nvf_conv_matrices`).

        A full material tensor, uniform or patterned, may instead return
        :class:`wavebit.TensorConvMatrices`, which appends ``epxz, epyz, epzx,
        epzy`` for the nine-slot ``solve_conv`` interface. Physical tensor inputs
        use ordinary signs; off-diagonal convolution slots use negative signs.

        **This exists because the boundary was one-way.** ``solve_conv`` has always accepted
        these matrices, so a caller could put them *in*; nothing returned them, so anyone who
        wanted to look at what a modeling path actually built had to reach into ``_core`` and
        reassemble it. That is not hypothetical: three separate attempts to hand-rebuild this
        quantity against internals produced three wrong numbers in one day, because the
        reassembly has to reproduce the doubled reciprocal grid's layout and the order-index
        sign convention, and both fail silently when guessed.

        It is the right object to compare two modeling paths with, for the same reason it is the
        one the solve consumes: every path produces it, so a comparison is between the same
        quantity rather than between two reconstructions of it. Note what it *includes*, though
        — on an NVF path these carry the normal-field projector as well as the geometry, so a
        difference between two paths is not attributable to their geometry alone. Isolating the
        geometry means comparing the permittivity coefficients instead, which is a narrower
        question and is left to the caller that has it.

        **Permittivity only**, which since 2026-08-20 the name says: this was ``conv_matrices``
        beside a ``mu_matrices``, one named for the operation and one for the quantity, so the
        pair did not read as a pair and the unmarked half had to carry this paragraph instead.
        A magnetic structure also builds permeability slots and they are not returned here
        -- widening the tuple would change what every existing caller unpacks.
        :meth:`mu_conv_matrices` returns them in the same five- or nine-slot layout.

        **Fourier operators about the cell centre**, ``x = y = 0``, as :meth:`solve_conv`
        reads them. Since 2026-09-28 each kernel-frame slot is restated as ``D C D``, with
        ``D = diag((-1)**(m+n))`` by order label. A centred denser stripe has a positive first
        harmonic. See ``docs/conventions.md``, "One origin: the cell is centred on it".
        This allocates an array per slot and a negated temporary, at quadratic cost in the
        retained order count. Ordinary :meth:`solve` uses the private builders directly and
        pays no conversion cost.

        ``geometry`` behaves as in :meth:`solve`, and so do the conditions: a traced cell or
        structure serves this call and is put back after it.
        """
        with self._conditions(kwargs):
            return swap_fourier_origin(self._eps_conv_matrices(geometry), *self.order_window)

    @device_scoped
    def mu_conv_matrices(self, geometry=None, **kwargs):
        """The permeability convolution matrices, or ``None`` for a non-magnetic structure.

        ``(mux_conv, muy_conv, muz_conv_i, muxy_conv, muyx_conv)`` — the same five slots as
        :meth:`eps_conv_matrices`, on the other side of the
        ``(E, H, eps, mu) -> (H, -E, mu, eps)`` duality, which is why
        one layout serves both, and why this is a
        :class:`~wavebit._core.factorization.ConvMatrices` as well: its field names are the
        permittivity's, and the duality is what makes reading ``mu`` out of ``.epx`` correct
        rather than a mislabelling. ``None`` rather than five identity matrices when nothing is
        magnetic: the solve reads that ``None`` as "take the non-magnetic arithmetic", which
        is bit-identical to the code that existed before ``mu`` did.

        Full tensors append muxz, muyz, muzx and muzy in the same negative
        off-diagonal convention, returning :class:`wavebit.TensorConvMatrices`.
        A coupling in either epsilon or mu selects 4ff automatically. Every slot is about
        the cell centre, like :meth:`eps_conv_matrices` and the :meth:`solve_conv` input.

        ``geometry`` behaves as in :meth:`solve`, and so do the conditions: a traced cell or
        structure serves this call and is put back after it.
        """
        with self._conditions(kwargs):
            return swap_fourier_origin(self._mu_conv_matrices(geometry), *self.order_window)

    def slice_structure(self, layer=0, n=512, show='eps', band_limited=False, *, component=None):
        """One layer's structure as an array. See :func:`wavebit._core.draw.slice_structure`.

        Verb-first to match :meth:`plot_structure`, which is this call with ``plot()`` around
        it -- same arguments, one returning the array and one drawing it. It was
        ``structure_slice`` beside a ``structure_plot`` until 2026-08-20, a matched noun-first
        pair; moving only the plotting half left the two halves of one pair reading in
        opposite directions. The field pair says the same thing the same way
        (:meth:`reconstruct_field` / :meth:`plot_field`).
        """
        from .draw import slice_structure
        return slice_structure(self, layer=layer, n=n, show=show, band_limited=band_limited,
                               component=component)

    def plot_structure(self, layer=0, n=512, show='eps', ax=None, band_limited=False, *, component=None):
        """Draw one layer of the structure. See :func:`wavebit._core.draw.plot_structure`.

        Here rather than on the three backend solvers because the picture is NumPy either way,
        which is the argument ``plot_field`` already makes; the difference is that this one
        reads the *solver* rather than an array it was handed, so the per-backend wrapper had
        nothing left to do.
        """
        from .draw import plot_structure
        return plot_structure(self, layer=layer, n=n, show=show, ax=ax,
                              band_limited=band_limited, component=component)

    def _reject_unsupported_lattice(self):
        """Refuse the build paths a skewed cell has no implementation for — before they answer.

        Every one of these *runs* on a non-orthogonal cell and returns a plausible, wrong
        number: the raster is still a raster, the transforms are still transforms, and the NVF
        assembly stays Hermitian so energy still conserves to 1e-15. Nothing downstream is in a
        position to notice, which is why the refusal has to live here.

        What is left is specific, not general. ``factorization='li'`` applies the inverse rule
        along the *raster axes* (see
        :func:`~wavebit._core.factorization.to_conv_mat_raster_sampled`), and the solve
        consumes the result as Cartesian ``eps_xx`` / ``eps_yy``. On a skewed cell those axes
        are ``a1`` / ``a2``, not ``x`` / ``y``. Repairing that is Li's covariant formulation
        (JOSA A 14(10), 2758) — a separate piece of work, and one worth costing before doing,
        since NVF already converges faster than Li on the curved boundaries this feature exists
        for.

        Three things that were refused here are not any more. Two of them because the
        assumption behind the refusal turned out to be wrong rather than because they were
        implemented:

        * The **integrated** coefficients are computed by ``fourier2d_integrated`` as an integral over the
          *lattice* cell, where the Jacobian cancels against the ``1/area`` normalisation.
          There is no metric in it, so it was always lattice-general.
        * The **parametric** path (``modeling_path='parametric'``) states shape lengths as absolute
          Cartesian quantities, so nothing about a shape deforms under a skew — a rectangle
          stays a rectangle. Only the grid of frequencies its transform is sampled at moves,
          which is one expression in :func:`~wavebit._core.parametric.shape_ft`. Overlapping
          *painted* layers were refused inside that path for a while, on the belief that
          resolving an overlap clips against a rectangular cell. It does not: the clipper does
          Cartesian polygon booleans and never sees the cell, and ``exp(-i G.r)`` carries the
          periodicity for a polygon straddling an edge. What actually needed the basis was the
          periodic-image enumeration in
          :func:`~wavebit._core.overlap.overlapping_pairs`, which decides whether a layer has
          an overlap at all. See ``test_a_painted_overlap_survives_a_unimodular_shear``.

        The third was implemented, on 2026-08-21. Every ``raster_normal_method`` other than
        ``'distance'`` used to be refused here, on the grounds that only ``'distance'``
        carried the ``J^-T`` pullback that turns an index-space gradient into a Cartesian
        normal. True of the code, and not of the constructions: two of the three never take
        an index-space difference at all — ``'contour'`` measures between Cartesian points,
        ``'spectral'`` multiplies by a Cartesian wavevector — so what they needed was the
        *coordinates*, not a pullback, and the third needed its two scipy calls swapped for
        tools that can express a metric which is not diagonal. The refusal was costing
        accuracy rather than protecting it, because ``'distance'`` is the least accurate of
        the four and the skew makes it worse: on a disc at a 60-degree skew it sits at 18.5
        degrees of mean angular error against ``'contour'``'s 0.42
        (``docs/nvf-normal-methods.md``).

        The Li check below reads the *resolved* :attr:`factorization`, which is why a skewed
        run on the parametric path no longer has to spell ``factorization='nvf'`` out: that
        path resolves to NVF on its own, and demanding the keyword was this refusal reading the
        unresolved default and mistaking it for a request.

        **Laurent's rule is not refused here**, and that is a property of the rule rather than
        an oversight: it names no axis. Both in-plane slots are the one scalar convolution
        ``[[eps]]`` on the lattice's own reciprocal grid, so nothing in it is taken along ``a1``
        or ``a2`` and read back as ``x`` or ``y`` -- the thing this refusal exists for. The
        coefficients are lattice-general for the reason the integrated ones are (above).
        """
        if self.lattice_is_axis_aligned:
            return
        common = ('The cell basis is non-orthogonal (set via .lattice). ')
        factorization = self.factorization
        if factorization == Factorization.LI:
            raise NotImplementedError(
                common + f"factorization={factorization.value!r} applies the inverse "
                f'rule along the raster axes, which are not x and y once the cell is skewed, '
                f"so it would answer plausibly and wrongly. Use factorization='nvf'.")

    def _eps_conv_matrices(self, geometry=None):
        """Convolution matrices for the current structure: ``(epx, epy, epz_i, epxy, epyx)``."""
        if self._has_tensor_materials():
            self._refuse_a_record_of_another_rule(geometry)
            return _anisotropy.build(self, geometry)
        self._check_before_building()
        self._refuse_a_record_of_another_rule(geometry)
        return self._slots_for(self.structure, geometry, quantity='permittivity')

    #: What each rule's frozen record holds, for the refusal below -- named so that the message
    #: says what the other build would have found missing, not only that the names differ.
    _RECORD_HOLDS = {
        'li': 'the block compression of an integrated raster, or the Cartesian rectangle plans',
        'nvf': "the normal field or the shapes' interface set, beside the painting's plan",
        'laurent': "the block compression of an integrated raster, or the painting's plan -- "
                   'no normal field and no interface set',
    }

    def _refuse_a_record_of_another_rule(self, geometry):
        """Refuse a frozen geometry taken under a different factorization than this build's.

        A record holds what its own rule's build decides and nothing else, and the builds read
        an empty slot as an answer rather than as a gap: a raster's missing normal field sends
        NVF back to the eager construction -- which under ``jax.jit`` has no number to construct
        from -- and a Li record of rectangles handed to the shape NVF build carries no interface
        set, so the NVF replay would plan blind. A Laurent record of shapes is the same case: it
        plans the painting and no interface set. Until this check the parametric path refused the
        Li/NVF pair by testing ``rectilinear_plans`` for ``None``, and the raster path refused
        nothing. The stamp (:attr:`~wavebit._core.geometry.StaticGeometry.factorization`) is a
        Python string, so this runs the same eagerly and inside a trace. A record built by hand
        carries no stamp and is replayed on trust, as it always was.

        Called for both quantities and for tensor scenes, before anything is read from the
        record, so a mismatch is reported as itself rather than as whichever slot the other
        build happened to find empty first.
        """
        held = None if geometry is None else getattr(geometry, 'factorization', None)
        if held is None:
            return
        now = self.factorization
        if held == now:
            return
        raise _fac.StaleGeometryError(
            f"this frozen geometry was taken under factorization={held!r} and is being replayed "
            f"under factorization={now.value!r}. A record holds what its own rule's build "
            f"decides -- under {held!r}, {self._RECORD_HOLDS.get(held, 'its own decisions')} -- "
            f"and the {now.value!r} build would read what is missing as an answer and replay it "
            f"blind. Call freeze_geometry() again under factorization={now.value!r} (NVF, Li and "
            f"Laurent records are not interchangeable), or drop geometry= to rebuild.")

    def _mu_conv_matrices(self, geometry=None):
        """The permeability slots, or ``None`` when nothing in the scene is magnetic.

        The permeability is built by the *same* builder as the permittivity, on the same
        geometry, and that is the whole content of this method. Maxwell's two curl equations
        map onto each other under ``(E, H, eps, mu) -> (H, -E, mu, eps)``, so a factorization
        rule derived for one holds for the other term for term -- see
        :func:`~wavebit._core.factorization.pn_mu_conv_matrices`, which is
        :func:`~wavebit._core.factorization.pn_eps_conv_matrices` under a different name for
        exactly this reason. Rebuilding the rules for ``mu`` would be rederiving them.

        Two things the shared build gives that a separate one would have to re-earn: the two
        quantities land on one grid (the shape path replays the same boundary plans, so the
        geometry cannot drift between them), and every
        refusal that applies to the permittivity build applies here once rather than twice.

        Where ``mu`` *cannot* be expressed the refusal is by name and lives elsewhere, close
        to what cannot express it: the 1D and conical operators hold the diagonal three of
        these five and have no off-diagonal position for the other two, and ``solve_conv``
        says so. Full tensors in either epsilon or mu select the first-order
        (4ff) solve, including longitudinal magnetic-field reconstruction. Nothing drops mu,
        which is the whole reason it is wired at all: a material property the solve ignored
        would be the unread knob :meth:`_reject_unread_normal_method` exists to refuse.

        A magnetic *half-space* over ordinary layers answers ``None`` here and enters through
        ``mu_top`` / ``mu_bot`` instead; see :meth:`_layers_are_magnetic`.
        """
        if not self._layers_are_magnetic():
            return None
        if self._has_tensor_materials():
            self._refuse_a_record_of_another_rule(geometry)
            return _anisotropy.build(self, geometry, quantity='mu')
        self._check_before_building()
        self._refuse_a_record_of_another_rule(geometry)
        return self._slots_for(self._mu_structure(), geometry, quantity='permeability')

    def _check_before_building(self):
        """The refusals every build goes through, in the order they have to be read in."""
        # Lattice first, then the rule -- the order is load-bearing. Where a cell is refused
        # outright, reporting the *keyword* would send the user to change the one thing that
        # cannot help. The other way round is safe:
        # `_reject_unsupported_lattice` reads the resolved `factorization` itself, so a skewed
        # cell asking for a rule its path cannot express still gets told about the rule.
        self._reject_unsupported_lattice()
        # Then read both, unconditionally: this is the one call every build goes through, so it
        # is where an option the chosen path cannot express has to be refused rather than
        # quietly replaced by the one that path does implement. The factorization is read first
        # so that a smooth-path call naming *both* unavailable values is told about the rule --
        # the axis a user is far likelier to have meant something by.
        factorization = self.factorization
        self.fourier_coefficients   # read for the refusal it may raise, not for its value
        self._reject_unread_normal_method(factorization)
        self._reject_unread_projector(factorization)
        self._reject_unread_nesting(factorization)
        self._reject_unread_normal_grid()
        self._reject_layer_count_mismatch()

    def _reject_layer_count_mismatch(self):
        """The stack has to be one stack: as many thicknesses as the structure has layers.

        Every solve sizes its layer loop off `len(thickness)`, so the two ways of getting this
        wrong failed in opposite directions and neither said what was wrong:

        * **more layers than thicknesses** -- the extra convolution slots were simply never
          indexed. A two-layer structure with one thickness answered T = 0.851185066096, which
          is *bit-identical* to the one-layer structure with one thickness. The second layer
          was built, paid for, and dropped, in silence.
        * **more thicknesses than layers** -- a raw `IndexError: index 1 is out of bounds for
          axis 0 with size 1` out of the middle of the solve, which says where it died and not
          what the caller did.

        Checked at build time rather than in the setters because the two arrive in either
        order: assigning a two-layer structure to a solver that still holds one thickness is a
        legal half-step. This is the same reason `excitation`'s legality is read here and not
        in its setter -- the condition it depends on may still change.

        `len` covers both spellings: a raster is `(n_layers, Ny, Nx)`, and the two shape
        spellings are a list with one entry per layer.
        """
        structure = getattr(self, '_structure', None)
        if structure is None:
            return
        layers, thicknesses = len(structure), len(self._thickness)
        if layers == thicknesses:
            return
        raise ValueError(
            f'the structure declares {layers} layer(s) and `thickness` has {thicknesses} '
            f'entry(ies); a stack needs one thickness per layer. '
            + ('The extra layers used to be built and then never indexed -- the solve loops '
               'over `thickness` -- so the answer came back identical to the shorter stack.'
               if layers > thicknesses else
               'The extra thicknesses used to reach past the end of the structure and raise '
               'an IndexError from inside the solve.'))

    def _refuse_a_stale_geometry_record(self, geometry):
        """Check replayable geometry while the caller's scene is still concrete.

        Compiled field calls hide the live shape coordinates and tensor interfaces from
        the builders. Use those builders' own validation before entering the trace, without
        assembling Fourier operators or solving. Scalar shape checks inspect coordinates,
        so a traced material alone must not suppress their topology validation.
        """
        if geometry is None:
            return
        self._refuse_a_record_of_another_rule(geometry)
        if self._has_tensor_materials():
            with self._eager():
                _anisotropy.check_geometry(self, geometry)
            return
        if self._modeling_path == ModelingPath.RASTER:
            with self._eager():
                self._refuse_a_stale_raster_record(geometry)
            return
        if self._modeling_path != ModelingPath.PARAMETRIC:
            return
        self._reject_layer_count_mismatch()
        if self.factorization == Factorization.LI:
            _rectilinear.validate(self.structure, self.lattice)
            plans = geometry.rectilinear_plans
            if plans is None or len(plans) != len(self.structure):
                raise _fac.StaleGeometryError('Cartesian Li layer plans changed; freeze_geometry() again')
            for layer, plan in zip(self.structure, plans):
                _rectilinear.check_plan(layer.get('shapes', []), self.lattice, plan)
            return
        plans, gammas = geometry.boundary_plans, geometry.gamma_plans
        for records in (plans, gammas if self.factorization == Factorization.NVF else None):
            if records is not None and len(records) != len(self.structure):
                raise _fac.StaleGeometryError('shape layer count changed; freeze_geometry() again')
        lattice = self._geometry_lattice_for(geometry)
        for index, layer in enumerate(self.structure):
            _parametric.check_layer_spec(layer, index)
            _parametric.check_layer_shapes(layer, lattice, index)
            plan = None if plans is None else plans[index]
            if plans is not None:
                if plan is None:
                    _parametric.check_no_plan_still_fits(layer, lattice, index)
                else:
                    _boundary.check_plan_still_fits(layer.get('shapes', []), lattice, plan)
            if (self.factorization == Factorization.NVF and gammas is not None
                    and (plan is None or self.nvf_projector == 'circular')):
                _parametric.check_gamma_plan_still_fits(
                    layer, lattice, gammas[index], index, dyadic=self.nvf_projector == 'circular')

    def _refuse_a_stale_raster_record(self, geometry):
        """Run the raster builders' record checks now, without building -- for a traced build.

        A frozen normal field or compression is compared with the layer in front of it by the
        builder that replays it, and only eagerly: under a trace the layer has no value. JAX's
        jitted ``solve_field`` builds only under its trace, so it replayed every record on
        trust -- a normal field frozen on one oblique cell and replayed on another answered
        ``sum(T)`` wrong by up to 0.86 (measured 2026-09-25), where ``solve()`` refuses the same
        record. It calls this on the live solver first. The records and paths mirror
        :meth:`_slots_for` exactly: a normal-field source on an NVF path, a compression on an
        integrated one, and the permeability's own records beside the permittivity's. The
        parametric path does not use raster normal-field or compression records, so it
        needs no check here.
        """
        if geometry is None or self._modeling_path != ModelingPath.RASTER:
            return
        nvf = self.factorization != Factorization.LI
        integrated = self.fourier_coefficients != FourierCoefficients.SAMPLED
        (_, half_width_u), (_, half_width_v) = self.order_window
        passes = [(self.structure, geometry.normal_field, geometry.normal_field_source,
                   geometry.compression)]
        if self._layers_are_magnetic():
            passes.append((self._mu_structure(), geometry.mu_normal_field,
                           geometry.mu_normal_field_source, geometry.mu_compression))
        concrete_cell = not any(_holds_tracer(v) for v in (self._lattice_basis, self.period))
        for structure, field, source, compression in passes:
            # One differentiated material cannot hide the other quantity's concrete pattern.
            # Compression depends only on pixel cuts; normal fields also need the cell.
            if _holds_tracer(structure):
                continue
            _fac.check_raster_records_still_fit(
                structure, half_width_u, half_width_v,
                normal_field_source=source if nvf and field is not None and concrete_cell else None,
                compression=compression if integrated else None,
                lattice=self._geometry_lattice_for(geometry), normal_method=self.raster_normal_method)

    def _slots_for(self, structure, geometry, quantity):
        """Five convolution slots for one *quantity* of the current structure.

        ``quantity`` is ``'permittivity'`` or ``'permeability'`` and selects nothing about the
        arithmetic -- the duality means both are the same build -- only which frozen records
        are replayed, since a magnetic pattern need not sit where the dielectric one does.
        Paths that return three slots are padded with ``None``, which every consumer already
        reads as "this rule does not couple x and y".
        """
        factorization = self.factorization
        coefficients = self.fourier_coefficients
        # The *window's* half-width, not the declared `truncation_order`: an excitation can
        # widen the retained set, and a convolution matrix has to be as wide as the
        # `kx` it will meet. Its entries do not otherwise care where the window sits
        # -- `(i, j)` is the harmonic `i - j` -- so only the size travels here.
        (_, half_width_u), (_, half_width_v) = self.order_window
        mu_pass = quantity == 'permeability'

        if self._modeling_path == ModelingPath.RASTER:
            # The two axes are dispatched independently. Splitting on the factorization
            # first is not arbitrary: Li's rules return three matrices and NVF returns five
            # (it couples x and y, so epxy / epyx stop being zero).
            sampled = coefficients == FourierCoefficients.SAMPLED
            if geometry is None:
                normal_field = normal_field_source = compression = None
            elif mu_pass:
                normal_field = geometry.mu_normal_field
                normal_field_source = geometry.mu_normal_field_source
                compression = geometry.mu_compression
            else:
                normal_field = geometry.normal_field
                normal_field_source = geometry.normal_field_source
                compression = geometry.compression

            if factorization == Factorization.LI:
                if sampled:
                    conv = _fac.to_conv_mat_raster_sampled(
                        structure, half_width_u, half_width_v, nesting=self.li_nesting)
                else:
                    conv = _fac.to_conv_mat_raster_integrated(
                        structure, half_width_u, half_width_v, compression=compression,
                        nesting=self.li_nesting)
            elif factorization == Factorization.LAURENT:
                if sampled:
                    conv = _fac.to_conv_mat_raster_laurent(structure, half_width_u, half_width_v)
                else:
                    conv = _fac.to_conv_mat_raster_integrated_laurent(
                        structure, half_width_u, half_width_v, compression=compression)
            else:
                if sampled:
                    conv = _fac.to_conv_mat_raster_nvf(
                        structure, half_width_u, half_width_v,
                        normal_field=normal_field,
                        normal_field_source=normal_field_source,
                        lattice=self._geometry_lattice_for(geometry),
                        raster_normal_method=self.raster_normal_method,
                        nvf_projector=self.nvf_projector)
                else:
                    conv = _fac.to_conv_mat_raster_integrated_nvf(
                        structure, half_width_u, half_width_v,
                        normal_field=normal_field,
                        normal_field_source=normal_field_source,
                        compression=compression,
                        lattice=self._geometry_lattice_for(geometry),
                        raster_normal_method=self.raster_normal_method,
                        nvf_projector=self.nvf_projector)

        elif self._modeling_path == ModelingPath.PARAMETRIC:
            if factorization == Factorization.LI:
                if geometry is not None and geometry.rectilinear_plans is None:
                    raise _fac.StaleGeometryError("freeze_geometry() with factorization='li' before replay")
                conv = _rectilinear.layer_stack(
                    self.xp, structure, half_width_u, half_width_v,
                    self.lattice, self.complex_dtype,
                    plans=None if geometry is None else geometry.rectilinear_plans,
                    nesting=self.li_nesting)
            elif factorization == Factorization.LAURENT:
                # The same funnel as NVF -- every layer checked, overlaps painted -- with the
                # normal field dropped: Laurent reads the painting's plan and nothing else.
                conv = _parametric.to_conv_mat_parametric_layers(
                    structure, half_width_u, half_width_v,
                    self._geometry_lattice_for(geometry),
                    complex_dtype=self.complex_dtype, xp=self.xp,
                    boundary_plans=None if geometry is None else geometry.boundary_plans,
                    factorization=Factorization.LAURENT)
            else:
                if geometry is not None and geometry.rectilinear_plans is not None:
                    raise _fac.StaleGeometryError("Cartesian Li geometry cannot be replayed with NVF; freeze_geometry() again")
                xp = self.xp
                conv = _parametric.to_conv_mat_parametric_layers(
                    structure, half_width_u, half_width_v,
                    self._geometry_lattice_for(geometry),
                    shape_nvf_grid_size=self._resolve_normal_grid(half_width_u, half_width_v),
                    # The declared width, not a literal. This read `xp.complex128` with
                    # `self.complex_dtype` in scope, so a caller asking for single precision on
                    # this path got double: NumPy and JAX returned float64 from a solver that
                    # reported complex64, and Torch raised `expected m1 and m2 to have the same
                    # dtype` as soon as anything upstream honoured the request.
                    complex_dtype=self.complex_dtype, xp=xp,
                    boundary_plans=None if geometry is None else geometry.boundary_plans,
                    # The same record for both quantities: it is geometric, and the permeability
                    # view's own material slots make the seam signs per quantity at build time.
                    gamma_plans=None if geometry is None else geometry.gamma_plans,
                    **({} if self._projector is None else {'nvf_projector': self.nvf_projector}))

        else:
            raise ValueError(
                f'no modeling path is assigned (modeling_path={self._modeling_path!r}), so '
                f'there is nothing to build convolution matrices from. Assign `structure`: '
                f'its type is what selects the path.')

        # Returned as the builder made it. Every builder in `_core.factorization` and
        # `_core.parametric` answers with a `ConvMatrices`, the Li-rule ones carrying `None` in
        # the off-diagonal pair, so there is no padding step here to get the arity wrong.
        return conv

    # ------------------------------------------------------------------------- permeability

    def _is_magnetic(self):
        """Whether anything in the scene -- layers or half-spaces -- carries a permeability.

        Structural rather than physical, and answered the way
        :attr:`~wavebit._core.domain.Material.is_magnetic` answers it: a permeability that was
        *named* counts, whatever its value, and one that was never named does not. Nothing here
        reads a number, which is what lets the same call decide the same way eagerly and inside
        a ``jax.jit`` trace -- the property this method most needs, since it chooses which
        branch of :meth:`_solve_built` runs and a branch cannot be taken on a tracer.

        A scene that never mentions permeability therefore stays on exactly the arithmetic it
        had before ``mu`` existed, and pays nothing for it.
        """
        return (self._top.is_magnetic or self._bot.is_magnetic
                or self._layers_are_magnetic())

    def _layers_are_magnetic(self):
        """Whether any *layer* is magnetic -- the narrower question the slot build asks.

        Separate from :meth:`_is_magnetic` because the two halves of the permeability enter
        the solve by different doors. A magnetic half-space is two scalars on the boundary
        admittances; magnetic *layers* need five convolution matrices apiece. A scene with a
        magnetic substrate over ordinary layers must pass the scalars and no matrices, and
        building five identities for it instead would be an eigenproblem's worth of work to
        express "nothing here is magnetic".
        """
        if self._modeling_path == ModelingPath.RASTER:
            return self._mu is not None
        return any(m.is_magnetic for m in self._structure_materials())

    def _structure_materials(self):
        """Every :class:`~wavebit._core.domain.Material` named inside a shape structure."""
        out = []
        for layer in (self.structure or []):
            out.append(as_material(layer.get('background', 1.0), 'layer background'))
            for shape in layer.get('shapes', []):
                out.append(as_material(shape.get('material', 1.0), 'shape material'))
        return out

    def _has_tensor_materials(self):
        return _anisotropy.scene_is_tensor(getattr(self, '_structure', None),
                                          getattr(self, '_mu', None))

    def _mu_structure(self):
        """The current structure with every material's ``mu`` where its ``epsilon`` was.

        A *view*, built fresh each call and never stored: it exists only to be handed to the
        same builder the permittivity went through, which is what makes the permeability slots
        the E/H dual of the permittivity ones by construction rather than by a second
        derivation. Geometry is shared verbatim -- the shape dicts are copied key for key and
        only the material slot is swapped -- so the two passes cannot describe different
        shapes.

        The swap reads :attr:`~wavebit._core.domain.Material.mu_value`, not ``.mu``: this is a
        *consumer* of the number, and the builder it feeds does arithmetic on every slot. A
        region whose material named no permeability is non-magnetic vacuum-permeability
        material sitting in a scene something else made magnetic, and ``1.0`` is what that
        means; handing the builder a ``None`` would raise, and defaulting it here to anything
        else would be a second, quieter statement of a convention the type already owns.
        """
        if self._modeling_path == ModelingPath.RASTER:
            return self._mu
        out = []
        for layer in self.structure:
            shapes = [dict(shape, material=as_material(shape.get('material', 1.0)).mu_value)
                      for shape in layer.get('shapes', [])]
            out.append(dict(layer,
                            background=as_material(layer.get('background', 1.0)).mu_value,
                            shapes=shapes))
        return out

    def _placed_layers(self, structure):
        """A shape structure with every value in it passed through :meth:`_placed`.

        What the `structure` setter stores for the shape spelling, for the reason the raster
        branch goes through `_coerce_raster`: a background, a material or a coordinate the
        caller built on another device would otherwise be held there, and followed -- see
        :meth:`_placed`. The permeability pass reads the stored structure, so it inherits the
        placement. A layer or shape dict is rebuilt only when one of its values actually moved,
        so a scene written in Python numbers is stored as the very list it was handed.
        """
        def placed(entry):
            out = {key: ([placed(shape) for shape in value] if key == 'shapes'
                         else self._placed(value))
                   for key, value in entry.items()}
            moved = any(out[key] is not value if key != 'shapes'
                        else any(a is not b for a, b in zip(out[key], value))
                        for key, value in entry.items())
            return out if moved else entry

        if self._array_device is None:
            return structure
        layers = [placed(layer) for layer in structure]
        return layers if any(a is not b for a, b in zip(layers, structure)) else structure

    def _solve_built(self, geometry=None):
        """Build this structure's convolution matrices and hand them to the solve.

        The one place the two halves meet. The half-space quantities ride together as keywords
        or not at all: ``solve_conv`` recovers ``eps_top`` from ``n_top**2 / mu_top`` when
        it is not told, so supplying one of the pair without the other would leave the solve
        reading a permittivity that disagrees with the material the caller set. A non-magnetic
        scene passes none of them and takes that recovery, where dividing by ``mu_top = 1.0``
        is exact -- so the boundary admittances are the ones computed before ``mu`` existed, to
        the bit. One term is not: the incident wave's tangential-E row is now spelled from
        ``eps_top`` rather than from the index, so that it and the reflected column read one
        medium, and the two spellings agree only to a last bit for a general ``n_top``. See
        :func:`~wavebit._core.solve_1d.final_matching`, which measures the difference and says
        why it is the right way round.

        ``mu`` itself is ``None`` whenever no *layer* is magnetic, even in a scene whose
        substrate is -- see :meth:`_layers_are_magnetic`.

        The two half-space permeabilities go over as ``Material.mu`` and not
        :attr:`~wavebit._core.domain.Material.mu_value`, which is to say ``None`` when the
        material named none. ``solve_conv`` asks the same structural question of them that
        :meth:`_is_magnetic` asked of the scene, so the "was it named" bit has to survive the
        call; resolving it to ``1.0`` here would tell the backend a permeability was named for
        a half-space that never mentioned one. That is not merely untidy bookkeeping: the
        numeric default is applied at the one place that divides by it, so a ``1.0`` invented
        here would put a non-magnetic scene on the arithmetic written for a magnetic one and
        cost the bit-identity the whole feature is built to keep.

        Permittivity first, and it matters on the shape path: without a frozen geometry the
        permeability pass replays the boundary plans the permittivity pass has just recorded,
        so the order is what puts the two quantities on one grid.
        """
        eps = self._eps_conv_matrices(geometry)
        if not self._is_magnetic():
            return self._solve_conv(*eps)
        return self._solve_conv(*eps, mu=self._mu_conv_matrices(geometry),
                               mu_top=self._top.mu, mu_bot=self._bot.mu,
                               eps_top=self._top.epsilon, eps_bot=self._bot.epsilon)
