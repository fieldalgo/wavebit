# Tutorials

Lessons, read in order, for someone who has not used the API yet. The boundary against
[`../docs/`](../docs) is **audience, not format**: a tutorial is outgrown after one reading,
while a document in `docs/` is cited — including from code comments — and records what was
measured or why a default is what it is. A notebook whose purpose is to record a measurement
belongs there as markdown instead.

| Entry | Question it answers |
| --- | --- |
| [`01-modeling-and-emsolver.ipynb`](01-modeling-and-emsolver.ipynb) | How do you describe a structure and get a diffraction efficiency out of it? The starting point: modeling, and the electromagnetic solve (FieldWave uses RCWA). |
| [`02-autograd-and-optimization-jax.ipynb`](02-autograd-and-optimization-jax.ipynb), [`02-autograd-and-optimization-pytorch.ipynb`](02-autograd-and-optimization-pytorch.ipynb) | How do you differentiate a solve and optimize a structure with the gradient? Written twice, once per backend, so neither reader has to translate. |
| [`03-device-and-datatype-jax.ipynb`](03-device-and-datatype-jax.ipynb), [`03-device-and-datatype-torch.ipynb`](03-device-and-datatype-torch.ipynb) | Where does the solve run (CPU or GPU) and in what precision (64-bit or 32-bit)? Also per backend. |
| [`04-nvf-fourier-factorization.ipynb`](04-nvf-fourier-factorization.ipynb) | Why does a curved, high-contrast boundary converge so slowly under a plain discrete Fourier sum, and what does `factorization='nvf'` do about it? Also shows `freeze_geometry()`, the object that holds the normal field fixed and keeps a solve traceable. |
| [`05-oblique-lattice.ipynb`](05-oblique-lattice.ipynb) | How do you solve a hexagonal or otherwise tilted cell? The `lattice` keyword, and the raster convention that goes with it — a raster is read in *lattice* coordinates, and building it on a Cartesian grid instead describes a different structure with no error and no loss of energy conservation. Ends in a worked metalens phase library and the published benchmark the feature was checked against. |
| [`06-polarization-pair.ipynb`](06-polarization-pair.ipynb) | How do you send in circular or elliptical light, when `psi` can only name a linear state? The `polarization=(a_s, a_p)` pair: what it equals in `psi`, what its magnitude and phase each do, how to read the transmitted state back out, and the four ways it is usually got wrong. |
