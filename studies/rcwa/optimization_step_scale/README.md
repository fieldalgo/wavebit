# One optimization step at (15, 15)–(20, 20): torch, jax eager, jax jit

Measured 2026-10-27 with `optimization_step_scale.py` (this folder). The base is `2fd0229`, and the
script commits on top of it touch only this folder. The raw record is in `records/`:
`optimization_step_scale.log` holds every round as it ran, and `optimization_step_scale.jsonl`
holds one summary per run, which `--report` turns into the tables below.

**Machine.** Intel Xeon @ 2.10 GHz (AVX-512), 4 cores, 15.7 GB RAM, 13.36 GiB memory-cgroup
ceiling. Python 3.13.17, torch 2.14.1+cpu (LAPACK/BLAS: MKL 2024.2), jax/jaxlib 0.12.0.1 (eig and
solve go through jaxlib's LAPACK kernels on SciPy 1.18.1's OpenBLAS; matmul uses XLA's own dot),
NumPy 2.5.4. All arms use 4 threads (`OMP_/MKL_/OPENBLAS_NUM_THREADS=4`,
`torch.set_num_threads(4)`).

**Scene.** A 600 nm square cell with ε = 12.25 patterned layers, each 200 nm thick, in air on ε = 2.25,
lit at 850 nm with s-polarized light. The objective is the zeroth-order transmittance `T[0, n, n]`,
in complex128/float64 with jax x64 enabled first. Problems go through `rcwa.prepare` / `prepared.run`.

- **Shape:** one `Circle` per layer, with its radius as the design variable. Default modeling (NVF,
  default normal-field grid) with `topology_check='host'`, the policy that admits a traced
  circle under jit. The radius moves 0.37 nm (layer 2: −0.29 nm) every step, so the
  normal-field and Fourier caches miss every time, as they would in a real optimization.
- **Raster (control):** a fixed binary 128×128 pattern per layer, with the permittivity array as the
  design variable. Its high value moves every step.

**Method.**
- **Interleaving.** At (15,15) with one layer, all arms run in one process, one step each per
  round, with the arm order rotated every round. Everywhere else, the arms together exceeded the
  memory ceiling, so each arm ran alone in its own process, alternating torch → jit → eager.
- **Step times.** "First" is round 0, which includes compilation. Median and min are taken over
  the 3–4 rounds after it.
- **Modeling split.** Modeling is a second loss timed the same way: the gradient of a fixed
  linear functional of the convolution matrices, i.e. the Fourier coefficients and normal field,
  forward and backward. Solve is the full step minus modeling (eig, S-matrix and their backward).
- **Peak memory.** The largest resident set during a step, minus the resident set before it.
- **Compile and break-even.** Compile time is jit's first step minus its median. Break-even is
  that compile time divided by (torch median − jit median); it is "never" when jit is slower per
  step.

## Dense kernels alone (best of 3)

Eigenproblem sizes are 2·(2n+1)² square.

| orders | matrix | kernel | torch (s) | jax (s) | jax / torch |
|---|---|---|---|---|---|
| (15,15) | 1922 | eig | 5.05 | 7.45 | 1.48x |
| (15,15) | 1922 | solve | 0.33 | 0.43 | 1.28x |
| (15,15) | 1922 | matmul | 0.20 | 0.51 | 2.52x |
| (20,20) | 3362 | eig | 18.54 | 31.23 | 1.68x |
| (20,20) | 3362 | solve | 1.54 | 2.00 | 1.29x |
| (20,20) | 3362 | matmul | 1.01 | 2.66 | 2.63x |

## One step (forward + gradient)

"vs torch" is torch's median divided by the arm's median, so values below 1 mean slower than torch.

| orders | layers | path | arm | first (s) | median (s) | min (s) | modeling (s) | solve (s) | vs torch | compile (s) | break-even | peak (GB) | grad vs torch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (15,15) | 1 | shape | torch | 45.9 | 47.8 | 43.5 | 1.44 | 46.4 | | | | 4.9 | |
| (15,15) | 1 | shape | torch.compile | 44.1 | 44.9 | 44.3 | 1.71 | 43.2 | 1.06x | | | 4.4 | 0 |
| (15,15) | 1 | shape | jax eager | 122.1 | 98.9 | 96.8 | 3.66 | 95.3 | 0.48x | | | 6.3 | 4e-12 |
| (15,15) | 1 | shape | **jax jit** | 84.7 | 78.0 | 77.7 | 3.34 | 74.7 | **0.61x** | 6.7 | never | 3.9 | 2e-12 |
| (15,15) | 1 | raster | torch | 41.2 | 39.5 | 39.2 | 0.38 | 39.1 | | | | 4.7 | |
| (15,15) | 1 | raster | torch.compile | 46.9 | 39.9 | 39.1 | 0.47 | 39.5 | 0.99x | | | 4.3 | 0 |
| (15,15) | 1 | raster | jax eager | 111.4 | 89.2 | 89.1 | 0.91 | 88.3 | 0.44x | | | 6.5 | 8e-11 |
| (15,15) | 1 | raster | **jax jit** | 78.1 | 68.4 | 68.1 | 0.83 | 67.6 | **0.58x** | 9.6 | never | 3.8 | 5e-11 |
| (15,15) | 2 | shape | torch | 83.3 | 77.9 | 77.6 | 2.71 | 75.2 | | | | 7.9 | |
| (15,15) | 2 | shape | jax eager | 206.4 | 175 (1 step) | | | | ≈0.45x | | | > 9.2 | |
| (15,15) | 2 | shape | **jax jit** | 147.7 | 138.5 | 136.5 | 6.29 | 132.2 | **0.56x** | 9.2 | never | 7.3 | 2e-12 |
| (15,15) | 2 | raster | torch | 70.6 | 90.6 | 71.9 | 0.71 | 89.9 | | | | 7.5 | |
| (15,15) | 2 | raster | jax eager | 183.5 | 183.7 | 159.4 | 1.87 | 181.8 | 0.49x | | | 10.3 | 2e-11 |
| (15,15) | 2 | raster | **jax jit** | 145.0 | 129.3 | 124.1 | 1.27 | 128.0 | **0.70x** | 15.7 | never | 6.8 | 4e-11 |
| (20,20) | 1 | shape | torch | does not fit: passed 13.1 GB in its first step | | | | | | | | > 12.7 | |
| (20,20) | 1 | shape | jax eager | does not fit: passed 13.1 GB in its first step | | | | | | | | > 12.7 | |
| (20,20) | 1 | shape | **jax jit** | 394.3 | 379.9 | 377.4 | 16.95 | 362.9 | (torch does not fit) | 14.4 | | 10.9 | |
| (20,20) | 1 | raster | torch | 215.7 (1 step, at the ceiling) | | | 1.57 | | | | | 12.8 | |
| (20,20) | 1 | raster | **jax jit** | 358.1 | 344.0 | 342.4 | 2.40 | 341.6 | **≈0.63x** (vs torch's one step) | 14.1 | never | 10.4 | |
| (20,20) | 2 | shape | torch | does not fit: passed 12.7 GB in its first step; skipped | | | | | | | | > 12.3 | |
| (20,20) | 2 | raster | torch | does not fit: passed 12.7 GB in its first step; skipped | | | | | | | | > 12.3 | |

Notes on the rows:
- **Torch shape, round 3 (both layer counts) failed** with
  `torch._C._LinAlgError: linalg.svd ... failed to converge`. The failure is in
  `_floor_singular_values`, the NVF inverse-rule floor, at radius 161.48 nm, and it is
  deterministic (single-threaded too, and not at 161.11 or 161.85). The NumPy and jax arms solve the
  same point. Its median comes from the other rounds, which is why those runs used 4 rounds.
- **(15,15) two-layer raster torch** alternated between 71–73 s and 91–98 s steps, in either
  process layout. Its min (71.9 s) against jit's min (124.1 s) gives 0.58x.
- **(15,15) two-layer shape, jax eager** passed the 12.7 GB guard in its third step; its one
  warm step was 175 s.
- **(20,20) one-layer raster torch** completed one step (215.7 s, +12.8 GB, about 13.2 GiB resident)
  in the shared-process run before the next arm was OOM-killed. Run alone, it passed the
  13.1 GiB guard, so it fits only with no margin.
- **(20,20) one-layer raster, jax eager** was not run, at the user's request: on the shape path it
  passed 13.1 GB in its first step.
- **(20,20) two layers** do not fit this machine on either path; they stopped at the guard on
  torch's first step and were not run further.
- **An earlier (15,15) one-layer shape pass** (rounds 0–2 before its round-3 failure; first in
  the log) read torch 43–46 s and jit 70–81 s, the same ratio as the table.

## What it says

**At this scale jit does not pay. JAX is slower per step, jitted or not.** Jit runs at
0.56–0.70x torch's speed and eager JAX at 0.44–0.49x, so jit never breaks even, whatever its
7–16 s compile costs. The hypothesis that jit's gain shrinks toward 1x is refuted: it goes past 1x
to a loss.

The cause is that the step is almost all dense linear algebra:
- **Modeling is negligible.** It is 1–2% of a raster step and 3–5% of a moving-shape step (17 s
  of 380 s at (20,20)), so there is little left for XLA to fuse.
- **The kernels underneath are slower.** On this machine jax's `eig` is 1.5–1.7x slower than
  torch's MKL `eig`, and its matmul 2.5–2.6x slower. The step ratios sit between those kernel
  ratios.

**This reverses the small-order result in
[backend choice](../../../docs/validation/rcwa/backend-choice.md).** There, on Apple silicon, the
retired interface, and orders 3–13, jit won by 1.28–3.41x and repaid its compile within 2–13
steps. The lifecycle study measured 1.4–1.9x up to order 9. Some of that win came from fusing a
modeling half that is now under 5% of the step. The rest depended on that machine's LAPACK.

**Jit's real advantage at this scale is memory, not time.** It needs the least memory of any arm:
- **(15,15):** 3.8–3.9 GB against torch's 4.7–4.9 GB on one layer, and 6.8–7.3 GB against
  7.5–7.9 GB on two.
- **(20,20) one shape layer:** jit is the only arm that fits, at 10.9 GB, where torch and eager
  JAX pass 13 GB.

torch.compile measured 0.99–1.06x, as designed, since wavebit runs eagerly under it.

**Caveats.** These are single-host, CPU-only measurements, and the jax/torch gap is mostly a
statement about MKL against OpenBLAS/XLA-dot on an Intel CPU. A JAX build on a faster LAPACK, or a
GPU, could reorder it.
