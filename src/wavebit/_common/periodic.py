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

from wavebit._common.declarations import _DeclaredFieldsOnly

from wavebit.materials.response import as_material
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

from wavebit._common.conventions import _COMPLEX_CHI
from wavebit._common.conventions import _DIRECTION_CLASH
from wavebit._common.conventions import incidence_clashes
from wavebit._common.conventions import reject_complex_angle
from wavebit._common.conventions import transverse_component_problem
from wavebit._common.conventions import wavelength_problem
from wavebit._common.backend import _concrete
from wavebit._common.backend import array_namespace
from wavebit._common.backend import as_array
from wavebit._common.backend import eager_under_torch_compile
from wavebit._common.enums import Tetm  # noqa: F401  (re-exported for spec's tetm parsing)


def _is_raster(value: Any) -> bool:
    """True if ``value`` is an array grid (raster) rather than a shape recipe."""
    return not isinstance(value, (list, tuple, dict)) and hasattr(value, 'shape')


# --------------------------------------------------------------------------- material














# --------------------------------------------------------------------------- the attribute door






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
    (:func:`as_material` for the assignment paths, ``docs/reference/common/conventions.md`` for the quantity
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
    name. The field names are the ones ``src/wavebit/_common/_jobs_v1.py`` serializes, so they are the ingest
    layer's to rename.
    """
    layers: list = field(default_factory=list)
    top: Any = 1.0
    bot: Any = 1.0
    period: Any = (1.0, 1.0)
    #: The cell basis, columns ``a1`` / ``a2``, or ``None`` for the axis-aligned cell
    #: ``period`` spells. Carried because :class:`~wavebit._common._jobs_v1.JobSpec` calls itself a
    #: *complete* description of a run and could not describe an oblique cell without it --
    #: ``period`` reports ``(|a1|, |a2|)`` there, which does not determine the basis. Kept as
    #: the plain ``(2, 2)`` rather than a :class:`~wavebit.lattice.Lattice` for the same
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
        from wavebit._common.validation import _refuse_shaped_half_space
        from wavebit.materials.tabulated import TabulatedMaterial
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
                from wavebit.lattice import as_lattice
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
    (:meth:`~wavebit.rcwa._engine.engine.CoreSolveMixin._incident_kxy`). In a lossless medium that is
    exactly the direction the incident light travels; in an absorbing one the intensity is
    uniform along the surface and the Poynting vector departs from it by an amount set by the
    loss and the polarization (``docs/reference/common/conventions.md``). A nonzero imaginary angle
    instead specifies the formal complex-angle wavevector ``n*sin(theta)``, with the same
    signed ``n``.

    Alternatively, give real Cartesian ``kx`` and/or ``ky`` in inverse length
    units (nm**-1 when wavelength is in nm). The omitted component is zero.
    These replace ``theta``/``phi`` and stay fixed during a wavelength sweep.
    They are the incident wavevector, before reciprocal-lattice order offsets.
    At kx=ky=0 the polarization basis uses azimuth zero.

    ``polarization`` is the complex amplitude pair ``(a_s, a_p)`` — the general (elliptical)
    polarization ``psi`` cannot spell. It supersedes ``psi``: giving both is an error, and
    ``psi = v`` is exactly ``polarization = (sin(v), cos(v))``. See ``docs/reference/common/conventions.md``.
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
        ``incidence_clashes`` live in :mod:`wavebit._common.conventions` precisely so that this
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


@eager_under_torch_compile
def layer_thicknesses(logits, total, *, minimum=0.):
    """Map design weights to positive physical lengths with a fixed total.

    Return ``minimum + (total-sum(minimum))*softmax(logits)``. The one-dimensional
    real ``logits`` array chooses NumPy, JAX or Torch and retains its floating
    dtype, device and gradient. A plain numeric list uses NumPy; integer weights
    are promoted to float64. ``total`` and ``minimum`` are static host values;
    minimum is a nonnegative scalar or one length per layer, with sum < total.

    The output is an ordinary thickness array accepted by :class:`Stack`, RCWA
    and FDTD. For FDTD, also pass static ``mesh_thickness`` with the same total.
    With positive ``layer_smoothing=h``, give the first and last layers minimum
    lengths of at least h/2 so the averaged interfaces stay inside the stack.

    Adding a constant to all logits changes nothing. Fix one logit when an
    optimizer needs independent coordinates. Very unequal logits can saturate
    the softmax and its derivatives. With a zero minimum they can underflow to
    a zero length; this is refused rather than silently creating a missing layer.
    Invalid concrete weights raise; invalid JAX-traced weights return NaN lengths.
    Total and minima are not differentiated. No clipping or custom gradient is
    applied to valid design weights.
    """
    import numpy as np
    if not hasattr(logits, 'dtype'):
        if not isinstance(logits, (list, tuple)) or not all(isinstance(v, (int, float, np.number)) for v in logits):
            raise ValueError('logits must be a nonempty one-dimensional real array or numeric list.')
        logits = np.asarray(logits)
    xp = array_namespace(logits)
    if logits.ndim != 1 or logits.shape[0] == 0 or not xp.isdtype(logits.dtype, ('real floating', 'integral')):
        raise ValueError('logits must be a nonempty one-dimensional real array or numeric list.')
    if xp.isdtype(logits.dtype, 'integral'):
        logits = xp.astype(logits, xp.float64)
    def host(value, name):
        numeric = (int, float, np.number)
        if not (isinstance(value, (*numeric, np.ndarray)) or
                isinstance(value, (list, tuple)) and all(isinstance(v, numeric) for v in value)):
            raise ValueError(f'{name} must be static real host values.')
        value = np.asarray(value)
        if value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError(f'{name} must be finite static real host values.')
        return value
    total = host(total, 'total')
    floors = host(minimum, 'minimum')
    if total.shape != () or total <= 0:
        raise ValueError('total must be a positive finite static real scalar.')
    if floors.shape not in ((), logits.shape) or np.any(floors < 0):
        raise ValueError('minimum must be nonnegative and scalar or one value per layer.')
    floors = np.broadcast_to(floors, logits.shape).astype(float)
    remaining = float(total)-float(floors.sum())
    if not np.isfinite(remaining) or remaining <= 0:
        raise ValueError('The sum of minimum thicknesses must be strictly less than total.')
    valid = xp.all(xp.isfinite(logits))
    if _concrete(valid) is False:
        raise ValueError('logits must be finite.')
    weights = xp.exp(logits-xp.max(logits))
    lengths = as_array(xp, floors, dtype=logits.dtype, device=getattr(logits, 'device', None))
    lengths = lengths+remaining*weights/xp.sum(weights)
    valid = valid & xp.all(xp.isfinite(lengths) & (lengths > 0))
    if _concrete(valid) is False:
        raise ValueError('Thicknesses are not positive and finite; reduce the logit range or choose positive minima.')
    return xp.where(valid, lengths, xp.full_like(lengths, float('nan')))
