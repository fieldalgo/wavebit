"""Typed domain objects at the API boundary — ``Layer`` / ``Stack`` / ``Incidence``.

Phase 2 of the rewrite (see ``<a scratch note outside this repository>``): the geometry and illumination a user
configures become typed objects instead of a bare ``ucell`` array plus loose
``thickness`` / ``theta`` / ``pol`` scalars. These are a non-breaking, *adapt-in* layer —
the legacy keyword arguments keep working unchanged, and a ``Stack`` round-trips losslessly
to and from them (``Stack.from_ucell(ucell, thickness)`` rebuilds the exact ``ucell``).

A ``ucell`` is layer-major on its first axis in every encoding the solver accepts, so a
single layer is exactly ``(thickness[i], ucell[i])``:

  - raster   : ndarray ``(n_layers, Ny, Nx)`` — ``ucell[i]`` is layer *i*'s permittivity grid
  - vector   : list ``[[bg_material, [objects...]], ...]`` — one entry per layer
  - analytic : list of layer dicts — one dict per layer

``Stack`` therefore holds the ordered layers plus the surrounding media (``n_top``,
``n_bot``, ``period``); ``Incidence`` holds the illumination. Numerics and backend choice
(``fto``, ``fourier_type``, ``backend``, ``type_complex``) stay as :func:`fieldwave.call_mee`
arguments — they are not part of the physical scene.

Backend-agnostic: a raster ``eps`` may be a NumPy / JAX / PyTorch array; stacking goes
through the Array-API namespace so gradients survive.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

from .backend import array_namespace
from .enums import Pol


def _is_raster(eps: Any) -> bool:
    """True if ``eps`` is an array grid (raster) rather than a vector/analytic recipe."""
    return not isinstance(eps, (list, tuple, dict)) and hasattr(eps, 'shape')


@dataclass
class Layer:
    """One layer of a :class:`Stack`: a thickness paired with its permittivity spec.

    ``eps`` is the layer's entry of a legacy ``ucell`` in any of the three encodings — a
    raster grid array ``(Ny, Nx)``, a vector ``[bg_material, [objects...]]``, or an analytic
    layer dict. Phase 2 keeps this a thin typed pairing; a richer ``Material`` / ``Shape``
    geometry model (see ``<a scratch note outside this repository>``) is deferred to a later phase.
    """
    thickness: float
    eps: Any

    @property
    def is_raster(self) -> bool:
        return _is_raster(self.eps)


@dataclass
class Stack:
    """An ordered list of :class:`Layer` plus the media that bound it.

    Layer order is top-to-bottom, matching ``ucell``'s first axis (the same order the solver
    stacks them). ``n_top`` / ``n_bot`` are the superstrate / substrate indices and
    ``period`` is the in-plane period.
    """
    layers: list = field(default_factory=list)
    n_top: Any = 1.0
    n_bot: Any = 1.0
    period: Any = (1.0, 1.0)

    @classmethod
    def from_ucell(cls, ucell, thickness, n_top=1.0, n_bot=1.0, period=(1.0, 1.0)) -> "Stack":
        """Adapt a legacy ``(ucell, thickness)`` pair into a typed ``Stack``.

        ``ucell`` may be any encoding (raster ndarray / vector list / analytic list); it is
        layer-major on its first axis, so layer *i* is ``(thickness[i], ucell[i])``. The
        inverse of :attr:`ucell` — ``Stack.from_ucell(u, t).ucell`` reproduces ``u``.
        """
        if ucell is None:
            layers = []
        else:
            layers = [Layer(t, ucell[i]) for i, t in enumerate(thickness)]
        return cls(layers=layers, n_top=n_top, n_bot=n_bot, period=period)

    @property
    def thickness(self) -> list:
        """Per-layer thicknesses, in the list form the solver accepts."""
        return [layer.thickness for layer in self.layers]

    @property
    def ucell(self):
        """Rebuild the legacy ``ucell`` for the solver.

        Raster layers (array grids) stack into an ``(n_layers, Ny, Nx)`` array through the
        Array-API namespace, preserving backend and dtype; vector / analytic layers return a
        plain per-layer list. An empty stack is ``None`` (matching the ``ucell`` default).
        """
        if not self.layers:
            return None
        eps_list = [layer.eps for layer in self.layers]
        if all(_is_raster(e) for e in eps_list):
            xp = array_namespace(*eps_list)
            return xp.stack(eps_list, axis=0)
        return eps_list

    def as_kwargs(self) -> dict:
        """The geometry keyword arguments for :func:`fieldwave.call_mee`."""
        return dict(ucell=self.ucell, thickness=self.thickness,
                    n_top=self.n_top, n_bot=self.n_bot, period=self.period)

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index):
        return self.layers[index]


@dataclass
class Incidence:
    """The incident illumination: wavelength, angles, and polarization.

    ``pol`` is the TM fraction (``Pol.TE == 0`` … ``Pol.TM == 1``); ``psi`` / ``phi`` are the
    conical polarization / azimuth angles (``None`` selects the pure 1D TE/TM path). These map
    one-to-one onto the existing constructor arguments, so precedence (an explicit ``psi``
    overriding ``pol``) is whatever the solver already does.
    """
    wavelength: Any = 1.0
    theta: Any = 0.0
    phi: Optional[Any] = None
    pol: Any = Pol.TE
    psi: Optional[Any] = None

    def as_kwargs(self) -> dict:
        """The illumination keyword arguments for :func:`fieldwave.call_mee`."""
        return dict(wavelength=self.wavelength, theta=self.theta, phi=self.phi,
                    pol=self.pol, psi=self.psi)
