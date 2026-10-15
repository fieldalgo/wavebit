<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/wavebit_logo_dark.svg">
    <img src="assets/branding/wavebit_logo.svg" width="45%" alt="Wavebit">
  </picture>
</p>

# Wavebit

[![tests](https://github.com/fieldalgo/wavebit/actions/workflows/tests.yml/badge.svg)](https://github.com/fieldalgo/wavebit/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

**Differentiable electromagnetic simulation: RCWA and FDTD behind one API, on NumPy, JAX and PyTorch.**

Wavebit computes how light scatters from periodic nanostructures (RCWA) and how fields evolve in
time (FDTD). It differentiates through the RCWA calculation, so the model that predicts a device's
response is the model that optimizes it.

* **Two solvers, one lifecycle.** `rcwa` and `fdtd` share `check` → `prepare` → `run`, the same
  materials, explicit length units, radian angles, and fields in V/m and A/m.
* **One implementation of the physics** on three array backends. Cross-backend agreement is part of
  the test suite, not a promise.
* **Gradients where you need them.** Wavelength, materials, layer thicknesses and raster
  permittivities differentiate on JAX and PyTorch; shape dimensions on PyTorch and under eager
  `jax.grad`, not under `jax.jit`. The full table is in the [differentiation guide](docs/guide/common/autodiff.md#rcwa-which-inputs-differentiate-on-which-backend).
* **Checked against the outside world.** RETICOLO, fmmax and S4 for RCWA; Meep for FDTD; the Li (1997)
  crossed-grating benchmark. The numbers and how they were taken: [validation](docs/validation/index.md).

**Status: 0.0.3, alpha.** RCWA is the mature path. FDTD runs on CPU in double precision; its geometry,
material and derivative support is reported by `check` for each problem. The API may change before
1.0. An experimental RCWA solver for metal contacts is opt-in and has its own, narrower scope; read
[method selection](docs/guide/rcwa/nvf-method-selection.md) before enabling it.

## Install

Python 3.12 or newer.

```shell
pip install wavebit              # NumPy backend
pip install 'wavebit[jax]'       # or 'wavebit[pytorch]', or 'wavebit[jax,pytorch]' — differentiation backends
```

From a source checkout, for development:

```shell
git clone https://github.com/fieldalgo/wavebit.git && cd wavebit
pip install -e '.[jax,docs]'     # editable, with the documentation toolchain
```

`environment.yml` builds the full development environment, including what the studies need. The
version floors in [`pyproject.toml`](pyproject.toml) are the versions the suite is verified against;
each is commented with the reason it sits where it does.

## Quickstart

A binary grating on glass under normal incidence, on the NumPy backend:

```python
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length="nm"),
    stack=rcwa.Stack(
        period=(700., 700.),
        bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=300., epsilon=np.array([[1., 1., 12.25, 12.25]])),),
    ),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization="s"),)),
)
numerics = rcwa.Numerics(orders=(10, 0))
result = rcwa.run(problem, numerics=numerics)
print(float(result.R.sum()), float(result.T.sum()))
print(result.transmission.order(0, 0).efficiency)
```

Diffraction arrays keep `(wavelength, order_v, order_u)` axes even for a single wavelength. `R` and
`T` are efficiencies; absolute port power and its integration measure are recorded separately.

To differentiate, prepare once and run with live leaves. JAX double precision is enabled by you,
never by importing Wavebit:

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

prepared = rcwa.prepare(problem, numerics=numerics, backend="jax")

def transmitted(thickness):
    layer = problem.stack.layers[0].replace(thickness=thickness)
    live = problem.replace(stack=problem.stack.replace(layers=(layer,)))
    return jnp.sum(prepared.run(live).T)

gradient = jax.grad(transmitted)(jnp.asarray(300.))
```

The FDTD entry point works the same way; [`examples/fdtd/pulse_slab.py`](examples/fdtd/pulse_slab.py)
is the first thing to run. The conventions both solvers follow — relative permittivity,
`exp(-iωt)` phasors, bottom-to-top layers, centered cells, explicit units — are listed once in the
[documentation home](docs/index.md#seven-conventions-to-check) and defined normatively in
[`docs/reference/common/conventions.md`](docs/reference/common/conventions.md).

## Documentation

The prose is split by who is reading, the way NumPy's manual is:

| | For a reader who |
| --- | --- |
| [`docs/guide/`](docs/guide) | needs the contract for one keyword: the manual |
| [`docs/reference/`](docs/reference) | needs a signature or a guarantee |
| [`docs/design/`](docs/design), [`docs/development/`](docs/development) | is questioning a default, and wants what was measured to decide it |
| [`docs/validation/`](docs/validation) | wants the evidence behind a claim |
| [`tutorials/`](tutorials) | has not used the API yet: start with the [RCWA](tutorials/rcwa/00-unified-lifecycle.ipynb) or [FDTD](tutorials/fdtd/00-unified-lifecycle.ipynb) notebook |
| [`examples/`](examples) | wants it applied to a problem |
| [`studies/`](studies) | wants the raw measurement behind a number in `docs/` |

The same pages build as a site (`pip install -e '.[docs]'`, then
`sphinx-build -b html docs docs/_build/html`); a hosted copy is coming. Every link in `docs/` is a
relative link to a real file, and a test checks that every path a page cites exists.
`python -m wavebit.manifest` prints every option, refusal and constraint as one JSON document for
programs that need to know without reading the source
([capability manifest](docs/reference/common/capability-manifest.md)).

## Contributing

Bug reports, questions and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) opens
with the rule for where a new file belongs, and explains how changes are reviewed and landed.

## Citation

```bibtex
@software{wavebit,
    title  = {Wavebit: Differentiable Electromagnetic Simulation Suite},
    author = {Kim, Yongha},
    year   = {2026},
    url    = {https://github.com/fieldalgo/wavebit},
}
```

## License

Apache-2.0. [LICENSE](LICENSE) also carries the third-party notices this project owes.

## Contact

[dev@fieldalgo.com](mailto:dev@fieldalgo.com)
