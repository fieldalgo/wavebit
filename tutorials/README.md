# Tutorials

Lessons, read in order, for someone who has not used the API yet. The boundary against
[`../docs/`](../docs) is **audience, not format**: a tutorial is outgrown after one reading,
while a document in `docs/` is cited — including from code comments — and records what was
measured or why a default is what it is. A notebook whose purpose is to record a measurement
belongs there as markdown instead.

Every scene below states its materials as **relative permittivity** — `structure`, `top` and
`bot`, and the `Material(epsilon, mu=1.0)` a shape names. Where a notebook starts from a
datasheet refractive index it converts once, visibly, with `wavebit.eps_from_index(n)`. The
contract is [`../docs/conventions.md`](../docs/conventions.md); the retired spellings
(`ucell`, `n_top`, `n_bot`, `n_base`, `n_index`) are refused by name, because the quantity
changed with them and a script that only renames its keywords would solve a different
structure without saying so.

| Entry | Question it answers |
| --- | --- |
| [`01-modeling-and-emsolver.ipynb`](01-modeling-and-emsolver.ipynb) | How do you describe a structure and get a diffraction efficiency out of it? The starting point: modeling, and the electromagnetic solve (Wavebit uses RCWA). |
| [`02-autograd-and-optimization-jax.ipynb`](02-autograd-and-optimization-jax.ipynb), [`02-autograd-and-optimization-pytorch.ipynb`](02-autograd-and-optimization-pytorch.ipynb) | How do you differentiate a solve and optimize a structure with the gradient? Written twice, once per backend, so neither reader has to translate. |
| [`03-device-and-datatype-jax.ipynb`](03-device-and-datatype-jax.ipynb), [`03-device-and-datatype-torch.ipynb`](03-device-and-datatype-torch.ipynb) | Where does the solve run (CPU or GPU) and in what precision (64-bit or 32-bit)? Also per backend. |
| [`04-nvf-fourier-factorization.ipynb`](04-nvf-fourier-factorization.ipynb) | Why does a curved, high-contrast boundary converge so slowly under a plain discrete Fourier sum, and what does `factorization='nvf'` do about it? Also shows `freeze_geometry()`, the object that holds the normal field fixed and keeps a solve traceable. |
| [`05-oblique-lattice.ipynb`](05-oblique-lattice.ipynb) | How do you solve a hexagonal or otherwise tilted cell? The `lattice` keyword, and the raster convention that goes with it — a raster is read in *lattice* coordinates, and building it on a Cartesian grid instead describes a different structure with no error and no loss of energy conservation. Ends in a worked metalens phase library and the published benchmark the feature was checked against. |
| [`06-polarization-pair.ipynb`](06-polarization-pair.ipynb) | How do you send in circular or elliptical light, when `psi` can only name a linear state? The `polarization=(a_s, a_p)` pair: what it equals in `psi`, what its magnitude and phase each do, how to read the transmitted state back out, and the four ways it is usually got wrong. |
| [`07-multimode-incidence.ipynb`](07-multimode-incidence.ipynb) | How do you send light in on more than one diffraction order at once? The `excitation` dict: why a *single* non-specular order is only a relabelling of the angle, what a coherent superposition is actually for, the solve it saves against doing it beam by beam, why every lit order has to propagate, and where to go for a different polarization per order. |
| [`08-fdtd-time-domain.ipynb`](08-fdtd-time-domain.ipynb) | When is the time-domain solver the right one, and how do you know its answer is settled and refined? `wavebit.fdtd` against RCWA on the same slab, a whole band from one run, a grating order by order, the two engines behind the one function and what each can do, a derivative of a time-domain solve checked against a finite difference, and shape recipes. Ends with the edges: it is research alpha, and a grazing order measures the absorbing boundary rather than the grid. |

All layer lists and thickness arrays use **bottom-to-top** order. The stack bottom is
`z=0`, +z points upward, and `incidence_side` chooses illumination without reordering
the geometry. Notebook outputs were cleared when migrating the coordinate convention;
run the cells to generate figures and numbers in the current frame.
