# Tutorials

Lessons, read in order, for someone who has not used the API yet. The boundary against
[`../docs/`](../docs) is **audience, not format**: a tutorial is outgrown after one reading,
while a document in `docs/` is cited — including from code comments — and records what was
measured or why a default is what it is. A notebook whose purpose is to record a measurement
belongs there as markdown instead.

## Modeling and Electromagnetic simulation
* [Modeling and EMsolver](01-modeling-and-emsolver.ipynb)
  * modeling and Electromagnetic simulation(FieldWave uses RCWA).

## Autograd and Optimization
* [Autograd and Optimization on JAX](02-autograd-and-optimization-jax.ipynb) or [Autograd and Optimization on PyTorch](02-autograd-and-optimization-pytorch.ipynb)
  * gradient and optimization with the help of Automatic Differentiation(a.k.a. backprop).
  * separately written for JAX and PyTorch backends for user convenience.

## Device and Precision
* [Device and Precision on JAX](03-device-and-datatype-jax.ipynb) or [Device and Precision on PyTorch](03-device-and-datatype-torch.ipynb)
  * setting device(CPU or GPU)
  * setting precision(64bit or 32bit)
  * separately written for JAX and PyTorch backends for user convenience.

## Non-orthogonal lattices
* [Hexagonal and other tilted cells](05-oblique-lattice.ipynb)
  * the `lattice` keyword, and the raster convention that goes with it — a raster is read in
    *lattice* coordinates, and building it on a Cartesian grid instead describes a different
    structure with no error and no loss of energy conservation.
  * a worked metalens phase library, and the published benchmark the feature was checked against.

## Fourier factorization
* [NVF Fourier factorization](04-nvf-fourier-factorization.ipynb)
  * why a curved, high-contrast boundary converges slowly under a plain discrete Fourier sum,
    and what the normal vector field (`factorization='nvf'`) does about it.
  * also shows `freeze_geometry()`, the object that holds the normal field fixed and keeps a
    solve traceable.
