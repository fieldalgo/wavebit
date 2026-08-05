# Benchmark
FieldWave performance analyses

## RCWA result (compared to reticolo)

## 1. Fourier transform
convergence test with both of Discrete FT and Continuous FT.

## 2. Calculation time
time to get diffraction efficiencies from EM solver.

### Backends, and what `jax.jit` is worth

`benchmarks/bench_jit.py`. Seconds per solve, mean of 5 after warm-up, one CPU (no GPU on the
machine these were taken on, which matters — the ranking below is a CPU ranking).

Every path is traceable through `freeze_geometry()` + `conv_solve(geometry=...)`, the vector
one included. Calling `conv_solve()` with no geometry is the auto-detect path: it compares
concrete values on purpose and is not a jit failure.

```
              path  fto     numpy       jax   jax+jit     torch  jit speedup
    vector ellipse    3    0.0139    0.0284    0.0130    0.0143        2.19x
  vector rectangle    3    0.0132    0.0284    0.0124    0.0149        2.28x
          analytic    3    0.0887    0.0667    0.0273    0.0857        2.44x
        raster DFS    3    0.0372    0.0327    0.0189    0.0313        1.73x
    vector ellipse    6    0.1281    0.1590    0.1194    0.1996        1.33x
  vector rectangle    6    0.1272    0.1597    0.1202    0.1940        1.33x
          analytic    6    0.2193    0.2307    0.1391    0.2698        1.66x
        raster DFS    6    0.2343    0.2194    0.1662    0.2833        1.32x
    vector ellipse   10    1.6117    1.7065    1.5277    2.3321        1.12x
  vector rectangle   10    1.6378    1.7790    1.5661    2.0000        1.14x
          analytic   10    1.7846    2.0401    1.5548    2.5410        1.31x
        raster DFS   10    2.4673    3.6570    3.3703    3.2465        1.09x
```

Three things fall out of it.

**The jit win is a small-problem win.** 2.2-2.4x at `fto` 3, 1.3x at 6, 1.1x at 10. What jit
removes is Python overhead, and by `fto` 10 the call is dense eigendecomposition, which is
LAPACK's work with or without a trace. Anything that pushes `fto` up — which is to say,
anything that needs accuracy — pushes the win towards nothing.

**NumPy is the fastest backend here, and jax+jit only just catches it.** At `fto` 10 the vector
path is 1.61s on NumPy against 1.53s jitted; below that NumPy wins outright. Choose a backend
for its gradients, not its speed — that is what jax and torch are for on this code, and torch
is the slower of the two throughout (`torch.compile` does not work on this codebase at all:
`InternalTorchDynamoError` on every path tried).

**Analytic gains the most from jit and loses the most without it.** It is the slowest path at
`fto` 3 eager (0.089s) and among the fastest jitted (0.027s), because its per-call cost is
largely the normal field sampled on a grid — arithmetic, which is what jit is good at.

For a fit loop, the ordering that follows: get `fto` right first, then pick the backend for
gradients, and reach for `jit` only if `fto` stayed small.

## 3. Eigendecomposition with CPU core limit

