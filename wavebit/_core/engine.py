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

from . import analytic as _analytic
from . import boundary as _boundary
from . import factorization as _fac
from . import lattice as _lattice
from .backend import _device_of, array_namespace, as_array, check_input_precision
from .conventions import (_CHI_POLARIZATION_CLASH, _COMPLEX_CHI,
                          _PSI_POLARIZATION_CLASH, _TETM_INCIDENCE_CLASH,
                          incident_sp_amplitudes, lit_orders, order_window,
                          parse_order_pair, reject_complex_angle,
                          warn_if_xy_on_a_skewed_cell)
from .domain import (Material, RETIRED_MATERIAL_NAMES, as_material, index_from_eps,
                     retired_material_name)
from .enums import (Answers, DEFAULT_ANSWERS, DEFAULT_FACTORIZATION, DEFAULT_FORMULATION,
                    DEFAULT_FOURIER_COEFFICIENTS, DEFAULT_NORMAL_METHOD, answers_of,
                    normal_method_of,
                    Factorization, Formulation, FourierCoefficients, Tetm, coefficients_of,
                    factorization_of, formulation_of, resolve_fourier_options)
from .fields import (field_1d, field_1d_conical, field_2d, field_2d_4ff,
                     is_count, to_mirrored_z, to_scene_frame)
from .geometry import StaticGeometry
from .modeler import Modeling
from .result import Result, ResultSub
from .scattering import smm_1d as _smm_1d, smm_1d_conical as _smm_1d_conical, smm_2d as _smm_2d
from .solve_1d import solve_1d as _etm_1d, solve_1d_conical as _etm_1d_conical
from .solve_2d import solve_2d as _etm_2d
from .solve_2d_4ff import solve_2d_4ff as _solve_2d_4ff

# The four convolution slots that hold the permittivity's z row and column, in the order the
# solve methods take them. Naming them once keeps the dispatch message specific about which
# one was given -- see `CoreSolveMixin._formulation_for`.
_Z_ROW_SLOTS = ('epxz_conv_all', 'epyz_conv_all', 'epzx_conv_all', 'epzy_conv_all')

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
        shape = tuple(getattr(value, 'shape', ()))
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


def _efficiency_only(de_ri, de_ti):
    """The ``ResultSub`` payload of a solve declared ``answers='efficiency'``.

    Every amplitude slot is ``None`` rather than absent, because ``ResultSub`` has a fixed
    field list and a fixed pytree structure — ``None`` is an empty subtree, so a result from
    this rung flattens to fewer leaves without becoming a different kind of object. The s/p
    split of the efficiencies is ``None`` for a plainer reason: this rung never formed it.
    """
    return {'r_s': None, 'r_p': None, 't_s': None, 't_p': None,
            'R': de_ri, 'R_s': None, 'R_p': None, 'T': de_ti, 'T_s': None, 'T_p': None}


#: ``pol`` was three things at once, and this is the rename. See :attr:`CoreSolveMixin.tetm`.
_POL_RETIRED = (
    "pol is gone; use tetm='te' or tetm='tm'. The old name said three things at once: "
    'which polarization, which solver path, and (through a derived psi) a value that a '
    'fractional pol -- pol=0.3 was psi = 0.35*pi -- turned into a third spelling of the '
    'angle. tetm says only the first two, which are the ones that travel together: it '
    'declares that the incidence has no polarization and no azimuth degree of freedom, '
    'and that declaration is what buys the pure 1D TE/TM formulation. For a linear angle '
    'use psi, and for anything elliptical use polarization=(a_s, a_p).')

#: ``ucell_materials`` had a reader once and lost it. Deleted rather than renamed, so the
#: message says that rather than pointing at a replacement that does not exist.
_UCELL_MATERIALS_DELETED = (
    'ucell_materials is gone. It was assigned in all three backends and carried in the JAX '
    "pytree's aux_data -- where it took part in the jit cache key -- and read by nothing at "
    'all. A material is named where it is used now: wavebit.Material(epsilon, mu) in a shape '
    'or block slot, and permittivity samples in the raster itself.')

def _refuse_retired(kwargs):
    """Refuse the retired constructor keywords by name, before anything else runs.

    The three solver constructors collect these through ``**retired`` rather than listing
    them, because what has to be said is not "no such argument" -- Python says that already --
    but *which* argument replaced it and that the number it carried has to change too. A
    ``TypeError: unexpected keyword argument 'n_top'`` sends a reader to rename the keyword and
    keep the value, and a refractive index reread as a permittivity is a legal,
    energy-conserving, wrong medium that nothing downstream can flag.

    Anything else still raises ``TypeError``, from here rather than from Python, so a typo in a
    live keyword is not silently swallowed by the catch-all this refusal needs.
    """
    if not kwargs:
        return
    for name in kwargs:
        if name == 'pol':
            raise TypeError(_POL_RETIRED)
        if name == 'ucell_materials':
            raise TypeError(_UCELL_MATERIALS_DELETED)
        if name in RETIRED_MATERIAL_NAMES:
            raise TypeError(retired_material_name(name))
    raise TypeError(
        f'unexpected keyword argument(s) {sorted(kwargs)!r}')


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
    directly. The five methods it decorates all have counterparts on the other two backends,
    so the decorator has to stop being one backend's before those bodies can become one body.

    The scope itself is :meth:`CoreSolveMixin._device_scope`, which is a declared member: the
    base answers `nullcontext` and JAX overrides it. Declared rather than duck-typed because a
    missing device scope is invisible -- everything still computes, on the wrong device, and a
    CPU-only test machine cannot tell the difference.
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

    _excitation = None

    def __init__(self, top=1., bot=1., theta=0., phi=None, psi=None, chi=None, tetm=None,
                 truncation_order=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), layer_algorithm='ETM', excitation=None,
                 device=0, complex_dtype=0, polarization=None):
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
        self.top = top
        self.bot = bot

        self.theta = theta
        self.phi = phi
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
        self.wavelength = wavelength
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
        regardless, so its `dtype=` never changed a width -- dropping it leaves all three
        returning byte-identical arrays at both widths. `device=` is the half that was doing
        the work, and it stays, spelled `_array_device` (`None` where nothing is placed).

        NOTE, separate from this move: at ``complex_dtype=1`` NumPy still computes these offsets
        through a float64 intermediate while JAX and Torch stay in float32, so NumPy's kx/ky
        differ from theirs in the last bits. Pre-existing, and not something a fold should
        decide -- the declared width governs the answer, which
        ``test_the_declared_width_is_the_width_answered_in`` checks, and it is met either way.
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
        """
        return lit_orders(self._excitation, *self.order_window)

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
        which polarization arrives, and every lit order carries that same one. They may be
        traced -- an incident field is a legitimate design variable -- while the **keys** may
        not, because which entries of the incident column are non-zero decides array structure
        rather than a value.

        What it is for is a coherent superposition that a single solve cannot otherwise
        express. A *single* non-specular order is not that: exciting order ``m`` at ``theta``
        is the same plane wave as exciting the specular order at the angle whose ``kx``
        matches, so it buys nothing but a relabelling. Several at once is different, and the
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
        self._excitation = dict(value)

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
        material = as_material(value, 'top')
        self._n_top = _half_space_index(material, 'top')
        self._top = material

    @property
    def bot(self):
        """The substrate, as a :class:`~wavebit._core.domain.Material`. See :attr:`top`."""
        return self._bot

    @bot.setter
    def bot(self, value):
        """Both, or neither -- see :attr:`top`, which had the same order and the same defect."""
        material = as_material(value, 'bot')
        self._n_bot = _half_space_index(material, 'bot')
        self._bot = material

    @property
    def n_top(self):
        """The superstrate's refractive index, ``sqrt(eps*mu)`` -- **derived, read-only**.

        Every solve reads this rather than :attr:`top`, and it is not settable: two spellings
        of one medium that could disagree is exactly the state this type exists to remove.
        Assign :attr:`top` instead.
        """
        return self._n_top

    @n_top.setter
    def n_top(self, value):
        raise AttributeError(retired_material_name('n_top'))

    @property
    def n_bot(self):
        """The substrate's refractive index, ``sqrt(eps*mu)`` -- derived, read-only.

        See :attr:`n_top`.
        """
        return self._n_bot

    @n_bot.setter
    def n_bot(self, value):
        raise AttributeError(retired_material_name('n_bot'))

    @property
    def n_inc(self):
        """The incidence medium's index — the one ``theta`` / ``phi`` / ``kx`` live in.

        The in-plane wavevector build (each backend's ``_separable_kxy``) reads this
        rather than ``n_top``, so ``kx_m = n_inc*sin(theta)*cos(phi) + m*wl/period`` holds
        for either illumination side with one spelling.
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
        Those two describe the same light and give the same numbers; they differ in what is
        left free. ``polarization=(1, 0)`` keeps the polarization a live parameter — you can
        differentiate it, sweep it, optimize it, and you pay conical cost for the privilege.
        ``tetm='te'`` gives that up in exchange for the speed. A declared-away degree of
        freedom is *absent*, so ``psi`` reads back as ``None`` rather than as a number whose
        gradient is quietly zero.

        Legal only where the formulation exists: a 1D structure, ``truncation_order[1] == 0``, no azimuth,
        and no ``psi`` or ``polarization`` beside it. Everywhere else it is refused rather
        than ignored — ``_assign_grating_type`` does that check, and says which condition
        failed.
        """
        return self._tetm

    @tetm.setter
    def tetm(self, value):
        if value is None:
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
        polarization to resolve, and the pure 1D path it selects takes ``tetm_index`` instead.
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

    @property
    def pol(self):
        raise AttributeError(_POL_RETIRED)

    @pol.setter
    def pol(self, value):
        raise AttributeError(_POL_RETIRED)

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
                    'nonzero amplitude (the pair is normalized to unit flux, so its overall '
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
        self._lattice_basis = A
        self._set_period_slot(_lattice.periods_of(A))   # derived, not chosen

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
        return self._theta

    @theta.setter
    def theta(self, theta):
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
        if phi is None:
            self._phi = None
        else:
            check_input_precision('phi', phi, self.complex_dtype)
            self._phi = self._as_complex(phi)

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
        """
        return self._device

    @device.setter
    def device(self, value):
        self._device = self._resolve_device(value)
        if getattr(self, 'modeler', None) is not None:
            # `_array_device`, not `self._device`: the modeler wants the argument an array
            # constructor takes, which is `None` on the two namespaces that place their own
            # arrays. Torch's two spellings are the same object, which is why the line it
            # replaced could conflate them and still be right there.
            self.modeler.device = self._array_device

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

    #: Where `_as_complex` and `_as_real_angle` place what they build. `None` on the two
    #: namespaces that have no device to name; Torch overrides it with `self.device`, because
    #: there a tensor is placed at creation and there is nowhere else to say it. Declared for
    #: the same reason `_device_scope` is: a wrong placement computes the right numbers in the
    #: wrong place, and a CPU-only machine cannot tell.
    _array_device = None

    def _as_complex(self, value):
        """A user-supplied value as a complex array of this solver's width."""
        return as_array(self.xp, value, dtype=self.complex_dtype, device=self._array_device)

    def _as_real_angle(self, value):
        """The real part, as a real array. `psi` and `chi` are angles, not complex numbers."""
        xp = self.xp
        real = xp.real(value) if hasattr(value, 'dtype') else _np.real(value)
        return as_array(xp, real, dtype=self.float_dtype, device=self._array_device)

    def _default_psi(self):
        """The unstated polarization: 45 degrees, so `s` and `p` arrive with equal amplitude.

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
        if self._psi is not None:
            self._psi = self._as_real_angle(self._psi)
        self.truncation_order = self.truncation_order
        self._set_period_slot(self._period)            # a re-cast, not a new cell
        self.thickness = self.thickness
        # Guarded because `_structure` is set by the `RCWA*` subclass, after `_BaseRCWA`.
        if getattr(self, '_structure', None) is not None:
            self.structure = self.structure
        if getattr(self, 'modeler', None) is not None:
            self.modeler.complex_dtype = self._complex_dtype

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
                '    solver.lattice = wavebit.oblique(lx, ly, angle)\n'
                'or `wavebit.hexagonal(a)` / a 2x2 column_stack((a1, a2)); `period` follows '
                'from it.')

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
        if getattr(self, 'modeler', None) is not None:
            self.modeler.period = self._period

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = self._as_lengths(thickness, pair=False)

    def _device_scope(self):
        """Where this solver's arrays are placed, as a context manager.

        `nullcontext` on every backend that does not need one -- NumPy has no device and
        Torch places tensors at creation. JAX overrides it, because `jax.default_device` is
        the only way to say it there and the answer is scoped rather than an argument.
        """
        return _contextlib.nullcontext()

    def _incident_kxy(self, wavelength):
        """The incident in-plane wavevector, ``(k_inc_x, k_inc_y)``, normalized by ``k0``.

        One body for all three backends. It was three, under a docstring saying "the
        ``sin(theta)`` clamp differs between NumPy, JAX and Torch for tracing reasons, and this
        keeps it the single source" -- a sentence that was in four places and was false. NumPy
        and Torch branched with `if`, JAX with `jnp.where` on the *result*; putting the `where`
        on the *argument* instead reproduces each backend's own output byte for byte, checked
        across nine `theta` values including `pi/2`, `pi/2 + 0.5j` and `float32(pi/2) + 0.5j`.

        Three things that are not decoration:

        * the dtype is given at creation, not applied after. `xp.asarray(0.4)` is `float32` on
          Torch, and an `astype` afterwards widens a number that has already lost its digits --
          measured at 1.5e-07 on 13 of 18 probes, which is exactly the kind of quiet narrowing
          this repo hunts.
        * `theta` and `phi` go through `_as_complex`, not `xp.asarray`. They are *user* values
          and may be optimizer leaves; the constants below are the core's own, and
          `xp.asarray` is right for those. `as_array`'s docstring has the split.
        * the clamp and the threshold are Python floats computed once in NumPy. Building them
          per backend is what made the method look backend-shaped in the first place.

        ``theta >= pi/2`` is clamped one float32 ulp below it because ``sin`` at exactly
        ``pi/2`` in single precision walks off the branch NumPy documents at
        numpy/numpy#27306; the imaginary part rides through untouched, so an evanescent
        incidence keeps its decay.

        Takes ``wavelength`` for the signature it has always had, and does not read it: the
        incidence direction is a direction. Kept so that the three call sites did not have to
        change in the same commit that moved the body.
        """
        xp, C = self.xp, self.complex_dtype
        theta = self._as_complex(0.0 if self.theta is None else self.theta)
        # `phi is None` is the 1D TE/TM case, where the azimuth is not a degree of freedom.
        phi = self._as_complex(0.0 if self.phi is None else self.phi)
        # A constant the core builds, so `xp.asarray` is the right spelling here and
        # `_as_complex` is the right one above -- see `as_array`'s docstring on the split.
        clamped = xp.asarray(_HALF_PI_F32_BELOW, dtype=C) + xp.imag(theta) * 1j
        sin_theta = xp.sin(xp.where(xp.real(theta) >= _HALF_PI_F32, clamped, theta))
        # n_inc, not n_top: theta lives in the incidence medium, which incidence_side picks.
        return (xp.astype(self.n_inc * sin_theta * xp.cos(phi), C),
                xp.astype(self.n_inc * sin_theta * xp.sin(phi), C))

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
        -- which raises, naming which of the three reasons applies, rather than handing back a
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
        raise RuntimeError(_SCATTERING_NOT_SOLVED)

    def _finish(self, out):
        """Store the field-reconstruction state and return just the result mapping.

        All three slots are *reset*, not merely filled: below ``answers='field'`` the core hands
        back a mapping without them, and clearing them here is what stops a later
        reconstruction from walking the state of some earlier, unrelated solve. The
        ``'efficiency'`` rung returns a bare ``(de_ri, de_ti)`` pair — the core's oldest
        contract, kept because a caller who wants only power should not have to unpack a
        mapping — so it is widened into one here, at the boundary, once.
        """
        if not isinstance(out, dict):
            out = {'given': _efficiency_only(*out)}
        self.layer_info_list = out.pop('layer_info_list', [])
        self.T1 = out.pop('T1', None)
        #: The stack's scattering matrix -- both sides, every order -- on this shelf rather than in
        #: result: :class:`~wavebit._core.result.Result` is a registered pytree with fixed
        #: children, so carrying matrices there would change its structure and drag them
        #: across every ``jit`` / ``grad`` boundary. ``None`` after an ETM solve -- not empty
        #: but *absent*, because the ETM fixes one illumination's boundary condition and
        #: never forms a both-sides operator at all.
        self._scattering = out.pop('scattering', None)
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
          both ways must agree, and no external oracle provides that.
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

    def _formulation_for(self, z_row):
        """Settle :attr:`formulation` against the z-row slots this call was handed.

        ``z_row`` is the four ``ep{xz,yz,zx,zy}_conv_all`` arguments. The test is whether a
        slot was *given*, not whether it is nonzero: the branch decides control flow, so it has
        to be answerable on a JAX tracer, and "passed all-zero matrices" is a claim about the
        caller rather than about the tensor. Leave them ``None`` — the default — to stay on the
        cheap path.
        """
        given = [name for name, stack in zip(_Z_ROW_SLOTS, z_row) if stack is not None]
        if self._formulation == Formulation.TWO_FIELD and given:
            raise ValueError(
                f"formulation='2ff' cannot express the z row, and {', '.join(given)} "
                f"{'was' if len(given) == 1 else 'were'} given. "
                'The second-order form d2E/dz2 = Omega2 E exists only while the '
                'permittivity z row and column are empty; dropping them would answer for the '
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
        if self.structure is None:
            if self._grating_type is None:
                self._grating_type = 2
            return

        if self.modeling_type == 0:                      # raster
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

        elif self.modeling_type == 1:                    # block
            self._grating_type = 2                       # TODO: 1d conical case

        elif self.modeling_type == 2:                    # analytic, staircase-free
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
    def modeling_type(self):
        """Which modeling path the structure selected: ``0`` raster, ``1`` block, ``2`` shapes.

        ``None`` before a structure has been assigned. The *type of the object* passed as
        ``structure`` is the choice -- there is no separate mode switch -- so this is the only
        place that says which one was read, and it is what ``_core.draw`` reads to label a
        picture with the path that drew it. Read-only, and spelled the way every refusal in
        this package already spells the concept (``modeling_type=0`` is how the messages in
        ``_FACTORIZATION_REASON`` name a raster).
        """
        return self._modeling_type

    # -------------------------------------------------------------- modeller pass-throughs

    def modeling_block_instruction(self, *args, **kwargs):
        return self.modeler.modeling_block_instruction(*args, **kwargs)

    def rectangle(self, *args, **kwargs):
        return self.modeler.rectangle(*args, **kwargs)

    def ellipse(self, *args, **kwargs):
        return self.modeler.ellipse(*args, **kwargs)

    @property
    def ucell(self):
        raise AttributeError(retired_material_name('ucell'))

    @ucell.setter
    def ucell(self, value):
        raise AttributeError(retired_material_name('ucell'))

    def plot_field(self, field_cell):
        """Draw a reconstructed cell. See :func:`wavebit._core.fields.plot_field`."""
        from .fields import plot_field
        plot_field(field_cell, self.tetm_index)

    def _solve_1d(self, epx_conv_all, epy_conv_all, epz_conv_i_all):
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
        return self._finish(solve(
            self.tetm_index, wavelength, kx, self.n_top, self.n_bot, self.theta,
            self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            incidence_side=self.incidence_side, answers=self._answers,
            lit=self._lit()))

    def _solve_1d_conical(self, epx_conv_all, epy_conv_all, epz_conv_i_all):
        solve = self._solve_fn('conical')
        self._formulation_assigned = Formulation.TWO_FIELD
        wavelength = self.wavelength
        kx, ky = self._separable_kxy(wavelength)   # 1-D pair; see :meth:`_solve_1d`
        return self._finish(solve(
            wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self._incidence_pair(), self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            phi=self.phi, incidence_side=self.incidence_side,
            answers=self._answers, lit=self._lit()))

    def _solve_2d(self, epx_conv_all, epy_conv_all, epz_conv_i_all,
                 epxy_conv_all=None, epyx_conv_all=None,
                 epxz_conv_all=None, epyz_conv_all=None,
                 epzx_conv_all=None, epzy_conv_all=None, mu=None, mu_top=None, mu_bot=None,
                 eps_top=None, eps_bot=None):
        # `mu_top` / `mu_bot` are `None` when no half-space permeability was named, the same
        # spelling `Material.mu` uses and for the same reason: the refusal just below is a
        # structural question, and `mu_top != 1.0` on a traced permeability is a traced boolean
        # no `if` can take. The numeric default lives at the one place that multiplies by it,
        # at the bottom of this method -- absence is resolved to `1.0` exactly once.
        wavelength = self.wavelength
        kx, ky = self.order_kxy()
        z_row = (epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all)
        self._formulation_assigned = self._formulation_for(z_row)
        if self._formulation_assigned == Formulation.FOUR_FIELD:
            if (mu is not None and any(s is not None for s in mu)) or \
                    mu_top is not None or mu_bot is not None:
                # Name the reason this call is on the 4ff path, not a reason it could have
                # been: `formulation='4ff'` reaches here with an empty z row, and blaming a
                # z-coupled permittivity then sends the user to look for one that is not
                # there. This refusal is also what makes
                # `wavebit._core.fields.field_2d_4ff`'s `H_z = Kx Ey - Ky Ex` correct -- that
                # line holds only at `mu = 1`, and holds because nothing else can arrive.
                why = ('a z-coupled permittivity was given, which routes to'
                       if any(s is not None for s in z_row) else
                       "formulation='4ff' was asked for, which routes to")
                raise NotImplementedError(
                    f'a permeability was given and {why} the first-order (4ff) solve, and mu '
                    'is not implemented there -- neither in the layer operator nor in the '
                    'field reconstruction, whose H_z carries no mu_zz. Stay on the 2ff path, '
                    'which does take mu: drop the z row, or leave formulation at its default.')
            return self._finish(_solve_2d_4ff(
                wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self._incidence_pair(),
                self.thickness, epx_conv_all, epy_conv_all, epz_conv_i_all,
                epxy_conv_all=epxy_conv_all, epyx_conv_all=epyx_conv_all,
                epxz_conv_all=epxz_conv_all, epyz_conv_all=epyz_conv_all,
                epzx_conv_all=epzx_conv_all, epzy_conv_all=epzy_conv_all,
                phi=self.phi, incidence_side=self.incidence_side,
                answers=self._answers, layer_algorithm=self.layer_algorithm,
                lit=self._lit()))
        solve = self._solve_fn('2d')
        return self._finish(solve(
            wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self._incidence_pair(), self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            epxy_conv_all=epxy_conv_all, epyx_conv_all=epyx_conv_all,
            phi=self.phi, incidence_side=self.incidence_side,
            answers=self._answers, lit=self._lit(), mu=mu,
            mu_top=1.0 if mu_top is None else mu_top,
            mu_bot=1.0 if mu_bot is None else mu_bot,
            eps_top=eps_top, eps_bot=eps_bot))


# The rule each modeling path actually implements, for the two paths that implement exactly
# one. The raster path (0) is absent because it implements both and so takes the option as
# given; on these two the option is a claim to check rather than a choice to make.
_FACTORIZATION_IMPLEMENTED_BY = {
    1: Factorization.LI,
    2: Factorization.NVF,
}

# Why each of those two is the only rule its path has, phrased for the person who asked for
# the other one. The distinction matters: one is unbuilt, the other is unbuildable.
_FACTORIZATION_REASON = {
    1: ("the block path (modeling_type=1) applies Li's rules to the rectangles its modeler "
        "paints (wavebit._core.factorization.to_conv_mat_block); no normal field is computed "
        "there, so NVF is not implemented on it. Use a raster structure (modeling_type=0) or "
        "smooth shapes (modeling_type=2), which both have it"),
    2: ("the smooth shape path (modeling_type=2) builds eps from an analytic normal field -- "
        "the shapes' own (wavebit._core.analytic.to_conv_mat_smooth_nvf), or, for a painted "
        "layer whose shapes overlap, that of the disjoint polygons the painting resolves into "
        "(wavebit._core.boundary.to_conv_mat_boundary_layer). There is no raster "
        "axis to run Li's nested 1D rule along, so Li's rules are not something that path has "
        "yet to grow -- they are not something it can express. NVF is a superset of Li anyway "
        "(a constant normal collapses the projector onto Li's rules exactly), so nothing is "
        "lost: drop the keyword, or pass factorization='nvf' to say so"),
}

# The other Fourier axis, exactly the same shape of fact. Both shape paths describe a layer as
# geometry rather than as samples, and integrate the permittivity over that geometry in closed
# form; neither has a raster underneath to run a discrete series along. The raster path (0) is
# again absent because it is the one that implements both.
_COEFFICIENTS_IMPLEMENTED_BY = {
    1: FourierCoefficients.INTEGRATED,
    2: FourierCoefficients.INTEGRATED,
}

# Why, phrased for the person who asked for 'sampled'. Same distinction as above: the block
# path's missing series is unbuilt, the smooth path's is unbuildable.
_COEFFICIENTS_REASON = {
    1: ("the block path (modeling_type=1) hands the factorizer explicit rectangles and their "
        "edges, and integrates eps over them in closed form "
        "(wavebit._core.factorization.fourier2d_integrated, via to_conv_mat_block). The edges are the "
        "geometry, at whatever real coordinates the modeler put them, so there is no raster "
        "underneath to sample -- rasterising one first is what modeling_type=0 is. Use a "
        "raster structure, which implements both"),
    2: ("the smooth shape path (modeling_type=2) transforms geometry analytically -- each "
        "shape in turn (wavebit._core.analytic.shape_ft), or, for a painted layer whose "
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

    The smooth shape path (``modeling_type == 2``) is built from parametric shapes rather
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
                 theta=0.,
                 phi=None,
                 psi=None,
                 period=(1., 1.),
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
                 # 'sampled' on a raster, 'integrated' on both shape paths, which is
                 # the only series either can form. Naming the other one raises.
                 factorization=None,         # 'li' | 'nvf' (normal vector field); the
                 # default resolves per modeling path -- 'li' on a raster and on the
                 # block modeler, 'nvf' on analytic shapes, which is the only rule
                 # each of those last two implements. Naming the other one raises.
                 normal_grid=1024,  # normal-field sampling; sets the floor on shape gradients
                 # (a translation-invariant structure should give d/dcx == 0 exactly; the
                 #  sampling leaves a sawtooth instead -- 5e-4 at 256, 4.2e-5 at 1024, for
                 #  ~1.2-2x the runtime. Raise further to fit positions below that.)
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
                 normal_method=None,         # raster NVF normal-field construction:
                 # 'distance' (default) | 'contour' | 'spectral' | 'smooth'. Read only
                 # where a raster normal field is built (factorization='nvf' on a raster);
                 # explicitly set anywhere else it is refused rather than ignored.
                 # See docs/nvf-normal-methods.md for the measurements that rank them.
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
                 **retired,                  # ucell / n_top / n_bot / ucell_materials, so
                 # each can say what replaced it *and* that the quantity changed with the
                 # name. A TypeError alone would send a reader to rename the keyword and keep
                 # the number, which is a different, legal, energy-conserving structure.
                 ):
        _refuse_retired(retired)

        super().__init__(top=top, bot=bot, theta=theta, phi=phi, psi=psi, chi=chi, tetm=tetm,
                         truncation_order=truncation_order, period=period, wavelength=wavelength,
                         thickness=thickness, layer_algorithm=layer_algorithm,
                         excitation=excitation,
                         device=device, complex_dtype=complex_dtype,
                         polarization=polarization)
        self.incidence_side = incidence_side
        self.answers = answers

        # `xp=` and the *resolved* device, because `Modeling.__init__` resolves its own
        # `device` one line before it learns its namespace: `_xp` starts at the NumPy
        # fallback and only becomes this backend's when the `period` setter infers it from
        # the array. `_resolve_device` returns `None` for any non-Torch namespace, so a
        # Torch device handed in was discarded on arrival -- the modeler then built its own
        # arrays on the host while the solver was elsewhere. The solver knows its namespace;
        # it should not be guessed from an argument.
        self.modeler = Modeling(period=self.period, xp=self.xp,
                                device=self._array_device, complex_dtype=complex_dtype)

        self._modeling_type = None
        self._grating_type = None
        self._structure = None

        self.structure = structure
        # After `structure`: `mu` is refused on the two shape paths, and which path this is
        # only becomes known when the structure is assigned.
        self.mu = mu

        self.backend = self._BACKEND_ID if backend is None else backend
        self._set_fourier_options(fourier_coefficients, factorization)
        self.normal_method = normal_method
        self.normal_grid = normal_grid
        self.lattice = lattice      # after period: assigning a basis overwrites period


    #: Which of the three modeling paths the current ``structure`` selected; the concrete
    #: solver classes assign it. Defaulted here because both option properties resolve against
    #: it, so the mixin would otherwise only work inside a class that happens to set it first.
    _modeling_type = None
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
        validating setter -- which is exactly what a `property` with an `fset` is here -- and
        the retired spellings (`ucell`, `n_top`, `n_bot`, `pol`) are in the set on purpose:
        they are properties whose setters raise by name, so they keep saying what replaced
        them instead of being reported as unknown.

        Cached per class, because the three backends have different sets and the answer cannot
        change after import.
        """
        cached = cls.__dict__.get('_CONDITION_NAMES')
        if cached is None:
            import inspect

            # Two sources, because neither alone is the set. Properties miss `wavelength`,
            # `layer_algorithm` and `normal_grid`, which are plain attributes and are exactly
            # what a sweep moves -- deriving from properties alone refused a wavelength sweep,
            # which is the most ordinary call this library has. The constructor misses the
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

    def _assign_conditions(self, kwargs):
        """Set conditions from keywords, refusing a name that is not one.

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
        """
        if not kwargs:
            return

        legal = self._condition_names()
        unknown = sorted(name for name in kwargs if name not in legal)
        if unknown:
            import difflib

            hints = []
            for name in unknown:
                near = difflib.get_close_matches(name, sorted(legal), n=1, cutoff=0.7)
                hints.append(f'{name!r}' + (f' (did you mean {near[0]!r}?)' if near else ''))
            raise TypeError(
                f'unexpected keyword argument(s): {", ".join(hints)}. These entry points take '
                f'conditions to set before solving, and a name that is not one used to be '
                f'stored as an attribute and silently ignored -- the solve then answered the '
                f'scene you had, not the one you asked for. Settable conditions: '
                f'{", ".join(sorted(legal))}.')

        for name, value in kwargs.items():
            setattr(self, name, value)

    def _raster_types(self):
        """The array types a raster may arrive as: NumPy's, plus this backend's own.

        NumPy's is universal -- every backend takes a NumPy raster -- and `_ARRAY_TYPES` is the
        per-backend half, already declared for the geometry setters. The two `structure`
        setters that were not NumPy's differed from it in exactly this tuple.
        """
        return (_np.ndarray,) + self._ARRAY_TYPES

    @property
    def structure(self):
        """The layer stack, in whichever of the three encodings it was given in.

        A raster is an ``(n_layers, Ny, Nx)`` array of **relative permittivity**; the block
        and analytic spellings are per-layer lists that name a
        :class:`~wavebit._core.domain.Material` per region. Assigning it settles
        :attr:`modeling_type`, which is the only thing that selects the modeling
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
        ``modeling_type == 2`` with ``mu`` still attached, which is verbatim the
        silently-dropped knob :data:`_MU_ON_A_SHAPE_PATH` is worded to stop. The mismatched-grid
        refusal was worse: it left a ``(1, 2, 8)`` permittivity beside a ``(1, 2, 4)``
        permeability and the next ``solve()`` *answered*.

        ``_revalidate_mu`` still runs after the write, because it checks the new pair and has
        to see it. So the write is undone if it refuses. Rolling back is not the same as
        checking first: the rules for what a ``mu`` may sit beside belong to the ``mu`` setter,
        and a copy of them here would be a second thing to keep in step -- which is the reason
        ``_revalidate_mu`` re-assigns rather than re-checking inline, given in its own docstring.
        """
        if isinstance(structure, self._raster_types()):     # Raster
            kind, value = 0, self._coerce_raster('structure', structure)
        elif type(structure) is list:
            # Analytic: one dict(background=..., shapes=[...], combine=...) per layer.
            # Shape coordinates are absolute lengths in the units of `period`, not
            # fractions of it, and `combine` picks what overlapping shapes mean --
            # this path adds them where the block path paints. See
            # wavebit._core.analytic; overlaps are refused until it is set.
            # An empty list is a Block, not a missing structure -- `None` is that.
            kind = 2 if (len(structure) > 0 and isinstance(structure[0], dict)) else 1
            value = structure
        elif structure is None:
            kind, value = None, None
        else:
            raise ValueError(
                f'structure: expected a raster array, a list of layers, or None -- got '
                f'{type(structure).__name__}')

        previous = (self._structure, self._modeling_type)
        self._structure, self._modeling_type = value, kind
        try:
            # `mu` and `structure` are one raster pair and either can be assigned second, so
            # the pair is checked from whichever side completes it -- see `_revalidate_mu`.
            self._revalidate_mu()
        except Exception:
            self._structure, self._modeling_type = previous
            raise

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
        rather than last.

        Takes no wavelength. The solver already has one, and a second spelling of it was a
        second thing that could disagree: the k vectors, the field reconstruction and the
        auto-detune ladder all read ``self.wavelength``, so an argument that differed would
        have answered one problem while the rest of the object described another.
        """

        self._assign_grating_type()
        z_row = (epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all)
        if self._formulation_for(z_row) == Formulation.FOUR_FIELD:
            # The 1D paths have no z-row slots to fill, so a z row is also a choice of path:
            # the first-order solve is written on the 2D layout (a 1D structure is n_orders_v == 1
            # there, the same physics the conical path solves at conical cost).
            self._grating_type = 2

        # "Was a permeability named?", the structural question `Material.is_magnetic` answers,
        # rather than "does it differ from one?". `None` means no half-space permeability was
        # given; any value, 1.0 included, means one was. It used to read `mu_top != 1.0`, and
        # under `jax.jit` a named `mu` is a tracer, so that comparison produced a traced boolean
        # and `or` raised TracerBoolConversionError on every traced solve -- taking the jitted
        # `solve_field` out of service for magnetic and non-magnetic scenes alike. `None` is an
        # empty pytree subtree and comes through a trace as itself, which is the whole reason it
        # is how absence is spelled.
        magnetic = ((mu is not None and any(m is not None for m in mu))
                    or mu_top is not None or mu_bot is not None)
        if magnetic and self._grating_type != 2:
            raise NotImplementedError(
                'mu is implemented on the 2D path only; this structure routed to a 1D solve. '
                'The 1D and conical operators carry no permeability slot at all.')

        if self._grating_type == 0:
            result_dict = self._solve_1d(epx_conv_all, epy_conv_all, epz_conv_i_all)
        elif self._grating_type == 1:
            result_dict = self._solve_1d_conical(epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            result_dict = self._solve_2d(epx_conv_all, epy_conv_all, epz_conv_i_all,
                                         epxy_conv_all, epyx_conv_all, *z_row, mu=mu,
                                         mu_top=mu_top, mu_bot=mu_bot,
                                         eps_top=eps_top, eps_bot=eps_bot)

        given = self._RESULT_SUB(**result_dict['given']) if 'given' in result_dict else None
        s_inc = self._RESULT_SUB(**result_dict['s_inc']) if 's_inc' in result_dict else None
        p_inc = self._RESULT_SUB(**result_dict['p_inc']) if 'p_inc' in result_dict else None

        result = self._RESULT(given, s_inc, p_inc, order_window=self.order_window)

        return result

    @device_scoped
    def reconstruct_field(self, sample_x=20, sample_y=20, sample_z=20, incidences=('given',)):
        """Reconstruct the field from the state left behind by the last solve.

        ``sample_x`` / ``sample_y`` / ``sample_z`` each take either a **count** -- a uniform
        grid, the long-standing behaviour and unchanged to the bit -- or the **coordinates
        themselves**, as a 1-D array. The reconstruction is pointwise in all three, so
        explicit points cost the same per point as grid ones and involve no interpolation.
        They are named for taking either; a count is one way to say where to sample.

        ``sample_x`` / ``sample_y`` are lengths in ``period``'s units and are unconstrained.
        ``sample_z`` is absolute, measured from the top of the stack, must lie inside it, and
        assigns an interface to the layer below it -- which is visible only in ``E_z``, the one
        component genuinely two-valued there. With explicit ``sample_z`` the returned z axis is
        exactly the request, in the caller's order, rather than ``sample_z * n_layers``.

        ``incidences`` names which of the incidences the solve carries to reconstruct, one
        leading axis entry each, **in the order asked for**: ``'given'`` is the incidence the
        solve was posed with (``psi``, or the ``polarization`` pair), and ``'s'`` / ``'p'``
        are the pure variants the coupled paths produce for free -- the same three the result
        object calls ``given`` / ``s_inc`` / ``p_inc``. It was a positional triple of booleans
        (``set_field_input``) until 2026-08-20; booleans are now refused by name, since a
        reconstruction of the wrong incidence is a complete and plausible field cell.

        On the pure 1D TE/TM path only ``('given',)`` is answerable -- ``tetm`` declared the
        other degree of freedom away, so that solve carries one incidence and the cell it
        returns has no axis to put a second on. Naming ``'s'`` or ``'p'`` there is refused
        rather than ignored, which is what it was until 2026-08-20.

        Takes no ``geometry``: it never calls ``solve``, and every discrete decision it
        makes -- the grating type, and which incidences ``incidences`` selects -- is
        already concrete by the time it runs. It reads ``self.T1`` / ``self.layer_info_list``,
        so whichever geometry the preceding solve used is the one in force here.

        Explicit ``sample_z`` is the one thing here that cannot be traced: bucketing a z into a
        layer needs the thicknesses as numbers. It raises rather than guessing.
        """
        self._require_field_declaration()
        # An explicit z is stated in scene coordinates on both sides, so a backside request is
        # restated in the mirror the solve walks before anything buckets it.
        backside = self.incidence_side == 'bottom'
        counted_z = is_count(sample_z)
        if backside:
            sample_z = to_mirrored_z(sample_z, self.thickness)
        kx, ky = self.order_kxy()

        if self._grating_type == 0:
            sample_y = 1
            field_cell = field_1d(self.wavelength, kx, self.T1, self.layer_info_list,
                                  self.lattice, self.tetm_index,
                                  sample_x=sample_x, sample_y=sample_y, sample_z=sample_z,
                                  incidences=incidences)
        elif self._grating_type == 1:
            field_cell = field_1d_conical(self.wavelength, kx, ky, self.T1,
                                          self.layer_info_list, self.lattice,
                                          sample_x=sample_x, sample_y=sample_y,
                                          sample_z=sample_z, incidences=incidences)
        elif self._formulation_assigned == Formulation.FOUR_FIELD:
            # The grating type cannot answer this: the first-order path sets it to 2 as well,
            # so branching on it alone would hand a 4ff solve to the 2ff reconstruction, which
            # reads the same state slots and returns a plausible, wrong cell. Static under jit
            # for the same reason the side flip below is -- the formulation is settled eagerly.
            field_cell = field_2d_4ff(self.wavelength, kx, ky, self.T1,
                                      self.layer_info_list, self.lattice,
                                      sample_x=sample_x, sample_y=sample_y,
                                      sample_z=sample_z, incidences=incidences)
        else:
            field_cell = field_2d(self.wavelength, kx, ky, self.T1,
                                  self.layer_info_list, self.lattice,
                                  sample_x=sample_x, sample_y=sample_y,
                                  sample_z=sample_z, incidences=incidences)

        if backside:
            # The backside solve is the top-side solve of the z-mirrored scene, so its cell is
            # in that mirror -- axis and components both. `to_scene_frame` undoes both; see it
            # for why the component half is not optional, and `to_scene_frame`'s `flip_z` for
            # why an explicit z has no axis left to reverse. Static under jit either way:
            # `incidence_side` is aux data, never a tracer.
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
        # An optimizer's loop sets a design variable and re-solves in one call. It was a list
        # comprehension evaluated for its side effects in two of the three copies, which is a
        # `for` loop written to look like a value.
        self._assign_conditions(kwargs)
        result = self.solve(geometry=geometry)
        return result, self.reconstruct_field(sample_x, sample_y, sample_z, incidences)

    @device_scoped
    def solve_field(self, sample_x=20, sample_y=20, sample_z=20, incidences=('given',),
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
        if self._modeling_type in (1, 2):
            raise ValueError(_MU_ON_A_SHAPE_PATH)
        field = self._coerce_raster('mu', value)
        # Same grid or nothing. The two fields are sampled on one raster and are transformed
        # layer by layer against each other, so a mismatch is not a broadcast to resolve --
        # it means the two describe different structures. Left to the builders it surfaces
        # much later as a shape error inside a Fourier transform, naming neither field.
        if self._structure is not None and tuple(field.shape) != tuple(self._structure.shape):
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
            self.mu = self._mu

    @property
    def fourier_coefficients(self):
        """How each layer's Fourier coefficients are obtained: ``'sampled'`` | ``'integrated'``.

        Resolved against the modeling path, exactly like :attr:`factorization` and for the same
        reason. Only the raster path takes both: the two shape paths state a layer as geometry
        and integrate over it in closed form, so the integrated series is the only one either
        can form, and the keyword there is a claim about the build rather than a choice within
        it.

        Raises:
            NotImplementedError: when the series was *explicitly* asked for and this modeling
                path cannot form it. Same exemption for the default, and for the same reason:
                ``'sampled'`` is the module-wide default, so a block- or smooth-path user who
                never touched the keyword is not asking for anything and keeps the closed-form
                series both paths always ran.
        """
        implemented = _COEFFICIENTS_IMPLEMENTED_BY.get(self._modeling_type)
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
                f"here: {_COEFFICIENTS_REASON[self._modeling_type]}.")
        return implemented

    @fourier_coefficients.setter
    def fourier_coefficients(self, value):
        self._fourier_coefficients = coefficients_of(value)

    @property
    def factorization(self):
        """Which rule closes the discontinuous products: ``'li'`` | ``'nvf'``.

        This reports the rule the *current structure* will be built with, which is not always
        the one that was asked for, because only the raster path implements both. The block
        path has only Li's rules and the smooth shape path only NVF, so on those two the
        keyword is a claim about the build rather than a choice within it, and this resolves
        it against the modeling path instead of repeating it back.

        Raises:
            NotImplementedError: when the rule was *explicitly* asked for and this modeling
                path does not implement it. Refusing rather than substituting is the point --
                a smooth-path circle used to answer to ``factorization='li'`` with NVF numbers
                and no way to tell. The default is exempt because it is not a request: it is
                resolved per path, so a user who never touched the keyword gets Li on a raster
                and NVF on smooth shapes, which is what each path always did.
        """
        implemented = _FACTORIZATION_IMPLEMENTED_BY.get(self._modeling_type)
        if implemented is None:                       # raster, or no structure yet: free choice
            return DEFAULT_FACTORIZATION if self._factorization is None else self._factorization
        if self._factorization is not None and self._factorization != implemented:
            # `!=`, not `is not`: `Factorization` is a str enum, so a value written straight
            # onto the private attribute compares equal to the member it spells, and the
            # formatting below tolerates one rather than dying inside the error path.
            raise NotImplementedError(
                f"factorization={factorization_of(self._factorization).value!r} is not "
                f"available here: {_FACTORIZATION_REASON[self._modeling_type]}.")
        return implemented

    @factorization.setter
    def factorization(self, value):
        self._factorization = factorization_of(value)

    @property
    def normal_method(self):
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

    @normal_method.setter
    def normal_method(self, value):
        self._normal_method = None if value is None else normal_method_of(value)

    def _reject_unread_normal_method(self, factorization):
        """Refuse an explicit ``normal_method`` on a build that constructs no raster normal
        field — before it silently decides nothing.

        The keyword is read exactly when the raster path builds a normal field, which is
        ``factorization='nvf'`` on ``modeling_type == 0``. Everywhere else — Li on a raster,
        either shape path — the build would run to completion with the request ignored,
        which is the same silent non-choice the ``enhanced_dfs`` parameter used to be
        (``docs/nvf-upsampling.md``): a knob whose reader does not exist. The default is
        exempt for the same reason as everywhere else: an untouched keyword is not a request.
        """
        if self._normal_method is None:
            return
        if self._modeling_type == 0 and factorization == Factorization.NVF:
            return
        if self._modeling_type == 1:
            where = "the block path applies Li's rules only and builds no normal field"
        elif self._modeling_type == 2:
            where = 'the smooth shape path builds its normal field from the geometry itself'
        else:
            where = "factorization='li' builds no normal field at all"
        raise NotImplementedError(
            f"normal_method={normal_method_of(self._normal_method).value!r} has no reader "
            f"here: {where}. It selects the raster NVF normal-field construction, so it "
            f"needs a raster structure and factorization='nvf'.")

    def freeze_geometry(self, **kwargs):
        """Take this structure's discrete geometry decisions once, eagerly.

        Captures whatever the active Fourier strategy decides from the concrete ``structure``.
        Hand the result back to :meth:`solve` as ``geometry=`` to make the call
        traceable. The default ('sampled', 'li') build decides nothing, so its frozen geometry
        is empty — passing it is still what keeps ``solve`` off the eager path.

        The record describes *this* structure. Replaying it onto one whose geometry has moved
        is refused rather than answered, eagerly — see :mod:`wavebit._core.geometry` for what
        each record checks and why a traced replay cannot.
        """
        self._assign_conditions(kwargs)
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
        # The *window's* half-width, not the declared `truncation_order`: an excitation can
        # widen the retained set, and a convolution matrix has to be as wide as the
        # `kx` it will meet. Its entries do not otherwise care where the window sits
        # -- `(i, j)` is the harmonic `i - j` -- so only the size travels here.
        (_, half_width_u), (_, half_width_v) = self.order_window

        # The two axes decide independently what has to be frozen: NVF needs the normal
        # field, the integrated series needs the block compression, and ('integrated', 'nvf')
        # -- the combination the old integer could not spell -- needs both.
        normal_field = normal_field_source = compression = None
        mu_normal_field = mu_normal_field_source = mu_compression = None
        if self._modeling_type == 0:
            if factorization == Factorization.NVF:
                basis = self._geometry_lattice_for(None)
                normal_field = _fac.normal_field_for_raster(
                    self.structure, half_width_u, half_width_v, lattice=basis, method=self.normal_method)
                normal_field_source = _fac.normal_field_source_for_raster(
                    self.structure, half_width_u, half_width_v, lattice=basis, method=self.normal_method)
                if self._mu is not None:
                    # A separate record, off `mu`'s own raster. A magnetic inclusion need not
                    # sit where the dielectric one does, and the NVF rearrangement is written
                    # around *one* interface normal, so each tensor gets the normal of its own
                    # -- the same argument `to_conv_mat_raster_pn_mu` makes. Sharing the
                    # permittivity's field would be right on every case where the two patterns
                    # coincide, which is most of them, and silently wrong on the rest.
                    mu_normal_field = _fac.normal_field_for_raster(
                        self._mu, half_width_u, half_width_v, lattice=basis, method=self.normal_method)
                    mu_normal_field_source = _fac.normal_field_source_for_raster(
                        self._mu, half_width_u, half_width_v, lattice=basis, method=self.normal_method)
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
                                  mu_compression=mu_compression)
        if self._modeling_type == 1:
            # The block modeler records its plans while it builds, so run the build once to
            # get them -- the *modeler*, not the whole factorization. Freezing used to go
            # through `_eps_conv_matrices` and throw the convolution matrices away, which it paid
            # for only because the same-material detector read them; that detector is gone.
            # Dropping them takes a freeze from 25.6 ms to 10.4 ms, which is what makes
            # re-freezing every step of a fit cheaper than nursing one along.
            #
            # One record serves the permeability build too, and has to: the mu pass paints the
            # *same* shapes at the *same* coordinates, so replaying these plans is what
            # guarantees the two quantities land on one grid rather than on two that happen to
            # agree.
            self.modeling_block_instruction(self.structure)
            geometry.block_grid = self.modeler.grid_plans
            geometry.shape_choices = self.modeler.shape_choices
        if self._modeling_type == 2:
            # Only a painted layer that actually overlaps has a plan to freeze. A summed one
            # has no discrete decision in it, and a painted one with nothing overlapping means
            # the same structure either way -- both stay on the closed-form path, so their
            # slots stay None and the replay never looks.
            plans = [_boundary.plan_layer(layer.get('shapes', []),
                                          self._geometry_lattice_for(None))
                     if _analytic.layer_needs_resolving(
                         layer, self._geometry_lattice_for(None)) else None
                     for layer in self.structure]
            # Recorded whether or not anything needed a plan. The entry *is* the decision:
            # `None` here means "looked at while the numbers were concrete, and it does not
            # overlap". Setting the attribute only when something overlapped made those two
            # cases indistinguishable downstream, so a disjoint painted layer under a trace
            # had to re-take the overlap test it had already passed -- on values that can no
            # longer answer it.
            geometry.boundary_plans = plans
        return geometry

    def solve(self, geometry=None, auto_detune=True, **kwargs):
        """Build the convolution matrices, then solve.

        ``geometry`` is an optional :class:`~wavebit._core.geometry.StaticGeometry` — the
        discrete decisions frozen by :meth:`freeze_geometry`. Supply it to keep the whole call
        traceable under ``jax.jit`` / ``jax.grad``; omit it and each decision is auto-detected
        eagerly, which needs concrete values.

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

        Raises:
            StaleGeometryError: eagerly, when ``geometry`` no longer describes the structure it
                is being replayed onto. A traced call cannot check — the comparison is the
                thing the record exists to avoid — so a long fit should re-freeze on a schedule
                rather than wait to be told.
        """
        self._assign_conditions(kwargs)   # an optimizer moves a design variable here
        self._frozen_geometry = geometry
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
        A magnetic structure also builds five permeability slots and they are not returned here
        -- widening the tuple would change what every existing caller unpacks.
        :meth:`mu_conv_matrices` returns the other five, in the same five-slot layout.

        ``geometry`` behaves as in :meth:`solve`.
        """
        self._assign_conditions(kwargs)
        return self._eps_conv_matrices(geometry)

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

        ``geometry`` behaves as in :meth:`solve`.
        """
        self._assign_conditions(kwargs)
        return self._mu_conv_matrices(geometry)

    def slice_structure(self, layer=0, n=512, show='eps', band_limited=False):
        """One layer's structure as an array. See :func:`wavebit._core.draw.slice_structure`.

        Verb-first to match :meth:`plot_structure`, which is this call with ``plot()`` around
        it -- same arguments, one returning the array and one drawing it. It was
        ``structure_slice`` beside a ``structure_plot`` until 2026-08-20, a matched noun-first
        pair; moving only the plotting half left the two halves of one pair reading in
        opposite directions. The field pair says the same thing the same way
        (:meth:`reconstruct_field` / :meth:`plot_field`).
        """
        from .draw import slice_structure
        return slice_structure(self, layer=layer, n=n, show=show, band_limited=band_limited)

    def plot_structure(self, layer=0, n=512, show='eps', ax=None, band_limited=False):
        """Draw one layer of the structure. See :func:`wavebit._core.draw.plot_structure`.

        Here rather than on the three backend solvers because the picture is NumPy either way,
        which is the argument ``plot_field`` already makes; the difference is that this one
        reads the *solver* rather than an array it was handed, so the per-backend wrapper had
        nothing left to do.
        """
        from .draw import plot_structure
        return plot_structure(self, layer=layer, n=n, show=show, ax=ax,
                              band_limited=band_limited)

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

        The block modeler (``modeling_type=1``) rasterises shapes onto a rectangular grid of
        its own and is not ported.

        Three things that were refused here are not any more. Two of them because the
        assumption behind the refusal turned out to be wrong rather than because they were
        implemented:

        * The **integrated** coefficients are computed by ``fourier2d_integrated`` as an integral over the
          *parametric* cell, where the Jacobian cancels against the ``1/area`` normalisation.
          There is no metric in it, so it was always lattice-general.
        * The **smooth shape** path (``modeling_type=2``) states shape lengths as absolute
          Cartesian quantities, so nothing about a shape deforms under a skew — a rectangle
          stays a rectangle. Only the grid of frequencies its transform is sampled at moves,
          which is one expression in :func:`~wavebit._core.analytic.shape_ft`. Overlapping
          *painted* layers were refused inside that path for a while, on the belief that
          resolving an overlap clips against a rectangular cell. It does not: the clipper does
          Cartesian polygon booleans and never sees the cell, and ``exp(-i G.r)`` carries the
          periodicity for a polygon straddling an edge. What actually needed the basis was the
          periodic-image enumeration in
          :func:`~wavebit._core.overlap.overlapping_pairs`, which decides whether a layer has
          an overlap at all. See ``test_a_painted_overlap_survives_a_unimodular_shear``.

        The third was implemented, on 2026-08-21. Every ``normal_method`` other than
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
        run on the smooth shape path no longer has to spell ``factorization='nvf'`` out: that
        path resolves to NVF on its own, and demanding the keyword was this refusal reading the
        unresolved default and mistaking it for a request.
        """
        if self.lattice_is_axis_aligned:
            return
        common = ('The cell basis is non-orthogonal (set via .lattice). ')
        if self._modeling_type == 1:
            raise NotImplementedError(
                common + 'the block modeler (modeling_type=1) rasterises onto a rectangular '
                'grid, which is not implemented yet for a skewed cell. Use a raster structure '
                '(modeling_type=0) or smooth shapes (modeling_type=2).')
        factorization = self.factorization
        if factorization != Factorization.NVF:
            raise NotImplementedError(
                common + f"factorization={factorization.value!r} applies the inverse "
                f'rule along the raster axes, which are not x and y once the cell is skewed, '
                f"so it would answer plausibly and wrongly. Use factorization='nvf'.")

    def _eps_conv_matrices(self, geometry=None):
        """Convolution matrices for the current structure: ``(epx, epy, epz_i, epxy, epyx)``."""
        self._check_before_building()
        return self._slots_for(self.structure, geometry, quantity='permittivity')

    def _mu_conv_matrices(self, geometry=None):
        """The five permeability slots, or ``None`` when nothing in the scene is magnetic.

        The permeability is built by the *same* builder as the permittivity, on the same
        geometry, and that is the whole content of this method. Maxwell's two curl equations
        map onto each other under ``(E, H, eps, mu) -> (H, -E, mu, eps)``, so a factorization
        rule derived for one holds for the other term for term -- see
        :func:`~wavebit._core.factorization.pn_mu_conv_matrices`, which is
        :func:`~wavebit._core.factorization.pn_eps_conv_matrices` under a different name for
        exactly this reason. Rebuilding the rules for ``mu`` would be rederiving them.

        Two things the shared build gives that a separate one would have to re-earn: the two
        quantities land on one grid (the block path replays the *same* plans, the shape paths
        the same boundary plans, so the geometry cannot drift between them), and every
        refusal that applies to the permittivity build applies here once rather than twice.

        Where ``mu`` *cannot* be expressed the refusal is by name and lives elsewhere, close
        to what cannot express it: the 1D and conical operators carry no permeability slot at
        all and ``solve_conv`` says so, and the first-order (4ff) solve does not implement
        one -- :meth:`CoreSolveMixin._solve_2d` says so. Nothing here drops a ``mu`` quietly,
        which is the whole reason it is wired at all: a material property the solve ignored
        would be the unread knob :meth:`_reject_unread_normal_method` exists to refuse.

        A magnetic *half-space* over ordinary layers answers ``None`` here and enters through
        ``mu_top`` / ``mu_bot`` instead; see :meth:`_layers_are_magnetic`.
        """
        if not self._layers_are_magnetic():
            return None
        self._check_before_building()
        return self._slots_for(self._mu_structure(), geometry, quantity='permeability')

    def _check_before_building(self):
        """The refusals every build goes through, in the order they have to be read in."""
        # Lattice first, then the rule -- the order is load-bearing. A skewed cell on the block
        # path is unbuildable whatever rule is named, so reporting the *keyword* there would
        # send the user to change the one thing that cannot help. The other way round is safe:
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

        if self._modeling_type == 0:                      # raster
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
                        structure, half_width_u, half_width_v)
                else:
                    conv = _fac.to_conv_mat_raster_integrated(
                        structure, half_width_u, half_width_v, compression=compression)
            else:
                if sampled:
                    conv = _fac.to_conv_mat_raster_nvf(
                        structure, half_width_u, half_width_v,
                        normal_field=normal_field,
                        normal_field_source=normal_field_source,
                        lattice=self._geometry_lattice_for(geometry),
                        normal_method=self.normal_method)
                else:
                    conv = _fac.to_conv_mat_raster_integrated_nvf(
                        structure, half_width_u, half_width_v,
                        normal_field=normal_field,
                        normal_field_source=normal_field_source,
                        compression=compression,
                        lattice=self._geometry_lattice_for(geometry),
                        normal_method=self.normal_method)

        elif self._modeling_type == 1:      # block -- Li on integrated coefficients
            # No branch on either option here, and that is not an oversight: this path has one
            # rule and one series (see _FACTORIZATION_IMPLEMENTED_BY and
            # _COEFFICIENTS_IMPLEMENTED_BY) -- `to_conv_mat_block` integrates eps over the
            # painted rectangles with `fourier2d_integrated`, unconditionally. Asking it for the other of
            # either was refused above rather than answered with this.
            #
            # The permeability pass replays the plans the permittivity pass recorded (or the
            # frozen ones), never re-decides them: the two passes paint the same shapes at the
            # same coordinates, so a second independent gridding could only agree by luck.
            plans = None if geometry is None else geometry.block_grid
            choices = None if geometry is None else geometry.shape_choices
            if mu_pass and plans is None:
                plans, choices = self.modeler.grid_plans, self.modeler.shape_choices
            structure_block = self.modeling_block_instruction(
                structure, plans=plans, shape_choices=choices)
            conv = _fac.to_conv_mat_block(structure_block, half_width_u, half_width_v)

        elif self._modeling_type == 2:      # smooth shapes -- NVF on exact transforms
            # Likewise fixed on both axes: the epsilon tensor comes from the shapes' analytic
            # normal field, so five matrices always, and each shape's transform is the exact
            # integral over the cell, so the coefficients are integrated ones. Naming
            # `factorization='li'` or `fourier_coefficients='sampled'` was refused above.
            xp = self.xp
            conv = _analytic.to_conv_mat_smooth_layers(
                structure, half_width_u, half_width_v,
                self._geometry_lattice_for(geometry),
                normal_grid=self.normal_grid,
                # The declared width, not a literal. This read `xp.complex128` with
                # `self.complex_dtype` in scope, so a caller asking for single precision on
                # this path got double: NumPy and JAX returned float64 from a solver that
                # reported complex64, and Torch raised `expected m1 and m2 to have the same
                # dtype` as soon as anything upstream honoured the request.
                complex_dtype=self.complex_dtype, xp=xp,
                boundary_plans=None if geometry is None else geometry.boundary_plans)

        else:
            raise ValueError(
                f'no modeling path is assigned (modeling_type={self._modeling_type!r}), so '
                f'there is nothing to build convolution matrices from. Assign `structure`: '
                f'its type is what selects the path.')

        # Returned as the builder made it. Every builder in `_core.factorization` and
        # `_core.analytic` answers with a `ConvMatrices`, the Li-rule ones carrying `None` in
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
        if self._modeling_type == 0:
            return self._mu is not None
        return any(m.is_magnetic for m in self._structure_materials())

    def _block_material_slots(self, layer):
        """``(background, [object materials...])`` of one block layer, as raw slot values.

        One reader for the two places that take a block layer apart, so the positional layout
        -- ``[background, [[name, cx, cy, lx, ly, material, angle], ...], combine]`` -- is
        written down once. A shape written without its material slot is left to the modeler to
        complain about; answering "vacuum" here would turn a missing material into a silent
        one, and this method only ever asks whether something is magnetic.
        """
        objects = layer[1] if len(layer) > 1 else []
        return layer[0], [(obj[5] if len(obj) > 5 else 1.0) for obj in objects]

    def _structure_materials(self):
        """Every :class:`~wavebit._core.domain.Material` named inside a shape structure."""
        out = []
        for layer in (self.structure or []):
            if self._modeling_type == 1:
                background, materials = self._block_material_slots(layer)
                out.append(as_material(background, 'block layer background'))
                out.extend(as_material(m, 'block shape material') for m in materials)
            else:
                out.append(as_material(layer.get('background', 1.0), 'layer background'))
                for shape in layer.get('shapes', []):
                    out.append(as_material(shape.get('material', 1.0), 'shape material'))
        return out

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
        if self._modeling_type == 0:
            return self._mu
        out = []
        for layer in self.structure:
            if self._modeling_type == 1:
                objects = [list(obj[:5]) + [as_material(obj[5]).mu_value] + list(obj[6:])
                           for obj in (layer[1] if len(layer) > 1 else [])]
                out.append([as_material(layer[0]).mu_value, objects] + list(layer[2:]))
            else:
                shapes = [dict(shape, material=as_material(shape.get('material', 1.0)).mu_value)
                          for shape in layer.get('shapes', [])]
                out.append(dict(layer,
                                background=as_material(
                                    layer.get('background', 1.0)).mu_value,
                                shapes=shapes))
        return out

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
        a half-space that never mentioned one, and on a scene routed to a 1D path that is the
        difference between solving it and refusing it.

        Permittivity first, and it matters on the block path: without a frozen geometry the
        permeability pass replays the grid plans the permittivity pass has just recorded, so
        the order is what puts the two quantities on one grid.
        """
        eps = self._eps_conv_matrices(geometry)
        if not self._is_magnetic():
            return self.solve_conv(*eps)
        return self.solve_conv(*eps, mu=self._mu_conv_matrices(geometry),
                               mu_top=self._top.mu, mu_bot=self._bot.mu,
                               eps_top=self._top.epsilon, eps_bot=self._bot.epsilon)
