# What is Wavebit

Wavebit computes how a periodic nanostructure scatters light, and differentiates through the
whole computation — so the same code that predicts a device's response can also optimize it.

The method is **RCWA** (rigorous coupled-wave analysis, also called the Fourier modal method).
It expands the field in the plane-wave orders a periodic structure supports, solves each layer
as an eigenproblem in that basis, and joins the layers into a stack response. What comes back
is one complex amplitude and one efficiency per diffraction order.

## What it is good at

* **Periodic structures**, in one or two directions: gratings, metasurfaces, photonic crystal
  slabs, wire-grid polarizers, colour filters.
* **Layered stacks** where each layer's cross-section is constant in `z`.
* **Sweeps and optimization**, because a solve is a few dense linear-algebra operations and the
  whole thing is differentiable end to end.

## What it is not for

Being honest about this saves more time than any feature list:

* **Aperiodic or finite structures.** RCWA solves one unit cell with periodic boundaries. A
  single isolated scatterer needs a supercell large enough that the images stop interacting,
  and the cost grows quickly.
* **Structures that vary continuously in `z`.** A smooth sidewall is approximated by slicing it
  into layers, and the slice count is yours to choose.
* **Broadband time-domain questions** — a pulse, a ring-down, a resonance hunt across a wide
  band in one shot. That is FDTD's shape, and it is on the roadmap below rather than here.
* **Very high index contrast with metals at grazing incidence**, where the retained order count
  needed for convergence can become impractical.

## The three backends

One implementation of the physics runs on NumPy, JAX and PyTorch behind a thin adapter, so a
result never depends on which you pick — cross-backend agreement is part of the test suite
rather than a claim. NumPy is the baseline and the only one that checks its own output; JAX and
PyTorch add automatic differentiation and GPU support. Which to run is measured, not a matter of
taste: see [Backends, devices and precision](backends.md).

## Status

**Version 0.1.0, unreleased.** No release has been tagged and the API is still free to move; the
things that are *pinned* are the conventions, which have a normative contract in
[Conventions](../conventions.md) and a test suite behind them.

## Roadmap

Wavebit is being built as a suite rather than a single solver:

1. **RCWA on a single-source core** — one backend-agnostic implementation of the physics behind
   a thin adapter, so NumPy, JAX and PyTorch cannot drift apart. *(done)*
2. **FDTD** — a time-domain solver sharing the same geometry and material layers, enabling
   cross-validation of the same structure between methods.
