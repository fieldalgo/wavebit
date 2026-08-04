# FieldWave

**Differentiable electromagnetic simulation suite — by [FieldAlgo](https://fieldalgo.com)**

FieldWave simulates how light interacts with nanostructures, and differentiates through the
whole calculation — so the same code that predicts a device's response can also optimize it.

* **Modeling** — describe a structure parametrically or as a raster
* **EM simulation** — RCWA solver, rigorous and fully vectorial
* **Optimization** — gradients flow end to end, from geometry to diffraction efficiency

## Lineage

FieldWave is the successor to [meent](https://github.com/kc-ml2/meent), continued by its
original lead author at FieldAlgo. It is derived from meent (MIT License, © 2022 ML2) and is
developed independently under Apache-2.0.

## Backends

One implementation runs on all three, so a result never depends on which you pick:

![alt text](images/backends.png "Backends")

* [NumPy](https://github.com/numpy/numpy) — the fundamental package for scientific computing with Python; easy and lean to use
* [JAX](https://github.com/google/jax) — autograd and XLA, brought together for high-performance machine learning research
* [PyTorch](https://github.com/pytorch/pytorch) — tensor computation with strong GPU acceleration and neural networks built on a tape-based autograd system

### When to use

|                 | NumPy | JAX | PyTorch | Description |
| --------------- | :---: | :-: | :-----: | :---------: |
| 64bit support   |   O   |  O  |    O    | Default for scientific computing |
| 32bit support   |   O   |  O  |    O    | 32bit (float32 and complex64) data type operation* |
| GPU support     |   X   |  O  |    O    | except Eigendecomposition** |
| TPU support     |   X   |  X  |    X    | Currently there is no workaround to do 32 bit eigendecomposition on TPU |
| AD support      |   X   |  O  |    O    | Automatic Differentiation (Back Propagation) |
| Parallelization |   X   |  O  |    X    | JAX pmap function |

*In 32bit operation, operations on numbers of 8>= digit difference fail without warning or error.
Use only when you do understand what you are doing.
**As of now(2023.03.19), GPU-native Eigendecomposition is not implemented in JAX and PyTorch.
It's enforced to run on CPUs and send back to GPUs.

NumPy is simple and light to use. Suggested as a baseline with small ~ medium scale optics problem.
JAX and PyTorch are recommended for cases having large scale or optimization part.
If you want parallelized computing with multiple devices (e.g., GPUs), JAX is ready for that.
But since JAX does jit compilation, it takes much time at the first run.

## How to install

```shell
pip install fieldwave
```

JAX and PyTorch are needed for advanced utilization.

## How to use

```python
import fieldwave

# backend 0 = NumPy
# backend 1 = JAX
# backend 2 = PyTorch

solver = fieldwave.rcwa(backend=1, ...)
result = solver.conv_solve()
```

To differentiate through a simulation, freeze the structure's discrete geometry once so the
rest stays traceable:

```python
import jax, jax.numpy as jnp

geometry = solver.freeze_geometry()
grad = jax.grad(lambda ucell: jnp.sum(
    solver.conv_solve(ucell=ucell, geometry=geometry).res.de_ti).real)
```

## Tutorials

Jupyter notebooks are prepared in [tutorials](tutorials) to give a brief introduction.

## Examples

Comprehensive examples of computational optics with FieldWave can be found in the `examples`
folder.

## Roadmap

FieldWave is being built as a suite rather than a single solver:

1. **RCWA on a single-source core** — one backend-agnostic implementation of the physics
   behind a thin adapter, so NumPy, JAX and PyTorch cannot drift apart. *(done)*
2. **FDTD** — a time-domain solver sharing the same geometry and material layers, enabling
   cross-validation of the same structure between methods.

## Citation

FieldWave builds on the work published as meent. If you use FieldWave in your research,
please cite:

```bibtex
@article{kim2024meent,
    title={Meent: Differentiable Electromagnetic Simulator for Machine Learning},
    author={Kim, Yongha and Jung, Anthony W. and Kim, Sanmun and
            Octavian, Kevin and Heo, Doyoung and Park, Chaejin and
            Shin, Jeongmin and Nam, Sunghyun and Park, Chanhyung and
            Park, Juho and Han, Sangjun and Lee, Jinmyoung and
            Kim, Seolho and Jang, Min Seok and Park, Chan Y.},
    journal={arXiv preprint arXiv:2406.12904},
    year={2024}
}
```

## Contact

[📩 FieldAlgo](mailto:dev@fieldalgo.com)
