# What is Wavebit

Wavebit computes how a periodic nanostructure scatters light, and differentiates through the
whole computation — so the same code that predicts a device's response can also optimize it.

For periodic layered structures, the method is **RCWA** (rigorous coupled-wave analysis, also called the Fourier modal method).
It expands the field in the plane-wave orders a periodic structure supports, solves each layer
as an eigenproblem in that basis, and joins the layers into a stack response. What comes back
is one complex amplitude and one efficiency per diffraction order.

## What it is good at

* **Periodic structures**, in one or two directions: gratings, metasurfaces, photonic crystal
  slabs, wire-grid polarizers, colour filters.
* **Layered stacks** where each layer's cross-section is constant in `z`.
* **Sweeps and optimization**, because a solve is a few dense linear-algebra operations and the
  whole thing is differentiable end to end.

## Limits of RCWA

Being honest about this saves more time than any feature list:

* **Aperiodic or finite structures.** RCWA solves one unit cell with periodic boundaries. A
  single isolated scatterer needs a supercell large enough that the images stop interacting,
  and the cost grows quickly.
* **Structures that vary continuously in `z`.** A smooth sidewall is approximated by slicing it
  into layers, and the slice count is yours to choose.
* **Broadband time-domain questions** — a pulse, a ring-down, a resonance hunt across a wide
  band in one shot. That is FDTD's shape rather than RCWA's, and it is why this package ships a
  second solver: [the `wavebit.fdtd` module](fdtd.md) answers exactly those, and a band costs one
  run rather than one run per wavelength.
* **Very high index contrast with metals at grazing incidence**, where the retained order count
  needed for convergence can become impractical.

## The three backends

The ordinary solver uses one implementation of the physics on NumPy, JAX and PyTorch behind
a thin adapter; cross-backend agreement is checked in the test suite. Experimental auxiliary
families state their narrower backend support separately. NumPy is the baseline; JAX and PyTorch add automatic differentiation and supported accelerator execution. The current public lifecycle provides structured checks and result validity metadata. Which to run is measured, not a matter of
taste: see [Backends, devices and precision](../guide/common/backends.md).

## Status

**Version 0.1.0, alpha candidate.** The API may change. The conventions have a normative
contract in [Conventions](../reference/common/conventions.md) and a test suite behind them.

The ordinary RCWA/NVF paths and the experimental contact auxiliary solver have separate
validation scopes. The auxiliary solver is explicitly selected, NumPy-only, and restricted
to one supported rectangle pair. Optical convergence, the independently checked optical
opening derivative and practical runtime remain unresolved. Passing its internal physical
checks does not certify optical accuracy. It is not called automatically by ordinary NVF;
see [method selection](../guide/rcwa/nvf-method-selection.md).
