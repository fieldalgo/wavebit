# Tutorials

Start with the current public API:

- [RCWA: an immutable problem and reusable preparation](rcwa/00-unified-lifecycle.ipynb).
- [FDTD: a pulse, material film and named time record](fdtd/00-unified-lifecycle.ipynb).

Both notebooks use the NumPy backend, shared physical materials and explicit units. Their code cells were executed sequentially in fresh processes; outputs remain empty so opening them does not present stale numerical output. They demonstrate execution and reuse, not a convergence claim.

After those introductions, the current API has three executable worked examples:

- [RCWA film optimization](../examples/rcwa/film_optimization.py): a prepared JAX objective, an independently checked derivative, and a quarter-wave antireflection design.
- [Dispersive FDTD fields](../examples/fdtd/dispersive_fields.py): a Lorentz film, seconds and SI material rates, raw Fourier E/H units, and live source-amplitude reuse.
- [RCWA–FDTD film and grating comparison](../examples/cross_solver/film_and_grating.py): common physical parameters, explicit amplitude reference planes, and separate spatial/time refinement.

These scripts retain their checks and need no plotting library unless the optional
FDTD `--plot` output is requested. Only the optimization example needs JAX.

## Current advanced lessons

The topics in the older notebooks have the following current entry points. They
use physical problems, explicit units and execution settings. The scripts retain
their own numerical checks; they do not reproduce the old notebook timing tables.

| Topic | Current API material |
| --- | --- |
| Modeling, diffraction and fields | [RCWA lifecycle](../tutorials/rcwa/00-unified-lifecycle.ipynb), [physical field sampling](../examples/rcwa/learning/learn/03_fields.py) |
| JAX differentiation and optimization | [film optimization](../examples/rcwa/film_optimization.py), [raster derivative](../examples/rcwa/learning/learn/05_jax_gradient.py) |
| Torch differentiation | [Torch derivative](../examples/rcwa/learning/learn/06_torch_gradient.py) |
| JAX devices and precision | [explicit execution](../docs/guide/common/backends.md) |
| Torch devices and precision | [explicit execution](../docs/guide/common/backends.md) |
| NVF and fixed numerical topology | [shape NVF](../examples/rcwa/learning/nvf/01_shape_nvf.py), [NVF gradient](../examples/rcwa/learning/nvf/04_jax_gradient.py) |
| Oblique cells and raster coordinates | [cell geometry](../docs/guide/rcwa/scene.md) |
| Polarization amplitude and phase | [physical illumination](../docs/guide/rcwa/light.md) |
| Multiple coherent sources | [coherent illumination](../examples/rcwa/coherent_illumination.py), [coherent source guide](../docs/guide/rcwa/coherent-illumination.md) |
| Periodic time-domain diffraction and derivatives | [film/grating comparison](../examples/cross_solver/film_and_grating.py), [physical periodic FDTD](../docs/guide/fdtd/periodic-problems.md), [backend derivatives](../examples/fdtd/fdtd_backends.py) |

## Historical tutorials

The ten numbered notebooks below are archived teaching records of the earlier
mutable API. **They require the separate source checkout below; installing the
current package will not make their imports work.** The two `00` notebooks above
are the current runnable introductions. No old factory is exported by the current
package, and no substitute compatibility module is needed for new calculations.

The historical notebooks keep their original calculation cells, assertions,
stored outputs and figures. Their first cell and metadata identify the source
revision. From an existing clone, create a separate worktree and environment:

```bash
git worktree add --detach ../wavebit-tutorials-v1 3c8a50f8a6506240169538957e99648a4b18ca3a
cd ../wavebit-tutorials-v1
python -m venv .venv
.venv/bin/python -m pip install '.[jax,pytorch,plot]'
```

Use that environment as the notebook kernel in your notebook editor, and open the
notebooks from that checkout. Follow its dependency floors; a recorded experiment's
own environment and source revision take precedence when reproducing stored numbers.
The device lessons skip CUDA sections when CUDA is unavailable. These instructions
preserve access to the historical workflow; they are not a new execution or timing
claim for all ten notebooks. See the [migration guide](../docs/migration/index.md).

| Archived notebook | Topic |
| --- | --- |
| [08-fdtd-time-domain.ipynb](fdtd/08-fdtd-time-domain.ipynb) | Periodic time-domain diffraction and derivatives |
| [01-modeling-and-emsolver.ipynb](rcwa/01-modeling-and-emsolver.ipynb) | Modeling, diffraction and fields |
| [02-autograd-and-optimization-jax.ipynb](rcwa/02-autograd-and-optimization-jax.ipynb) | JAX differentiation and optimization |
| [02-autograd-and-optimization-pytorch.ipynb](rcwa/02-autograd-and-optimization-pytorch.ipynb) | Torch differentiation |
| [03-device-and-datatype-jax.ipynb](rcwa/03-device-and-datatype-jax.ipynb) | JAX devices and precision |
| [03-device-and-datatype-torch.ipynb](rcwa/03-device-and-datatype-torch.ipynb) | Torch devices and precision |
| [04-nvf-fourier-factorization.ipynb](rcwa/04-nvf-fourier-factorization.ipynb) | NVF and fixed numerical topology |
| [05-oblique-lattice.ipynb](rcwa/05-oblique-lattice.ipynb) | Oblique cells and raster coordinates |
| [06-polarization-pair.ipynb](rcwa/06-polarization-pair.ipynb) | Polarization amplitude and phase |
| [07-multimode-incidence.ipynb](rcwa/07-multimode-incidence.ipynb) | Multiple coherent sources |

The historical scenes state relative permittivity and use bottom-to-top layer
order with the stack bottom at z=0 and +z upward. Their normalized source/field
conventions and mutable options belong to the pinned API. Current examples use
explicit physical units, including SI E/H; compare phase references, source
normalization and units before comparing arrays across the two interfaces.
