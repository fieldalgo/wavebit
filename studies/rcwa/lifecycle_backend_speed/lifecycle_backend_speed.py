"""
lifecycle_backend_speed.py -- the wall time of one derivative step through the public lifecycle.

One step is ``prepared.run(problem)`` and the derivative of its summed transmission, the unit an
optimizer repeats. Three ways of taking it, against two design variables:

    torch          eager PyTorch autograd                     radius, thickness
    jax-grad       eager jax.grad, no jit                     radius, thickness
    jax-jit-grad   jax.jit(jax.grad(...))                     thickness only: the lifecycle
                                                              refuses geometry under jit

The problem is a circle of radius 160 nm in a 600 nm square cell, 200 nm thick, eps 12.25 in
air on eps 2.25, lit at 850 nm, NVF on a 256-point normal-field grid -- the problem of
tests/api/test_unified_rcwa.py's ``patterned()``. The first call (compilation and caches) is
reported apart from the median and minimum of the next five.

Run each configuration in its own process: JAX compilations, torch graphs and allocator pools
are only reliably freed when a process exits. The record in records/ came from

    for n in 3 5 7 9; do
      for cfg in "torch radius" "jax-grad radius" "torch thickness" "jax-grad thickness" \
                 "jax-jit-grad thickness"; do
        python studies/rcwa/lifecycle_backend_speed/lifecycle_backend_speed.py $cfg $n
      done
    done

with the package's ``src`` on PYTHONPATH.
"""
import sys, time, resource, statistics
import jax, jax.numpy as jnp, torch
jax.config.update("jax_enable_x64", True)
from wavebit import Material, Units, rcwa

mode, variable, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
REPEATS = 5

def problem(radius, thickness):
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(600., 600.), bottom=Material(2.25),
            layers=(rcwa.Layer(thickness=thickness, background=Material(1.),
                regions=(rcwa.Region(geometry=rcwa.Circle(radius=radius), material=Material(12.25)),)),)),
        excitation=rcwa.Excitation(wavelength=850., sources=(rcwa.PlaneWave(polarization='s'),)))

numerics = rcwa.Numerics(orders=(n, n), shape_nvf_grid_size=256)
build = (lambda v: problem(v, 200.)) if variable == 'radius' else (lambda v: problem(160., v))
x0 = 160. if variable == 'radius' else 200.

if mode == 'torch':
    prepared = rcwa.prepare(build(torch.tensor(x0, dtype=torch.float64)), numerics=numerics, backend='torch')
    def step(x):
        v = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        prepared.run(build(v)).T.sum().backward()
        return float(v.grad)
else:
    prepared = rcwa.prepare(build(x0), numerics=numerics, backend='jax')
    derivative = jax.grad(lambda v: jnp.sum(prepared.run(build(v)).T))
    if mode == 'jax-jit-grad':
        derivative = jax.jit(derivative)
    def step(x):
        return float(derivative(jnp.asarray(x)).block_until_ready())

t0 = time.perf_counter(); g0 = step(x0 + 1.); first = time.perf_counter() - t0
times = []
for k in range(REPEATS):
    t0 = time.perf_counter(); step(x0 + 1. + .1*(k + 1)); times.append(time.perf_counter() - t0)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30   # bytes on macOS
print(f'{mode:13s} {variable:9s} orders=({n},{n})  first {first:7.3f} s  median {statistics.median(times):7.3f} s'
      f'  min {min(times):7.3f} s  derivative {g0:+.10e}  maxrss {rss:.2f} GB')
