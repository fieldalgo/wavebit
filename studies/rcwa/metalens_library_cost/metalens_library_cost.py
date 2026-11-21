"""What does one point of a pillar phase library cost on a CPU, and how do the value and the
time move with the number of retained orders -- for Wavebit, and for fmmax on the same array?

The array is the worked case of ``docs/theory/rcwa/oblique-lattice.md`` and of
``examples/rcwa/metalens_phase_library.py``: TiO2 pillars (n = 2.43) on fused silica
(n = 1.46), hexagonal pitch 350 nm, height 600 nm, 532 nm, normal incidence from air. Two
diameters away from the library's resonances, 150 and 240 nm.

Both solvers keep the same orders, ``|m|, |n| <= M`` on the same 120-degree basis, for
M = 5, 8, 11, 14 (121, 289, 529, 841 orders). They differ in what they are given:

  wavebit   the circle itself: an analytic shape transform, the default NVF factorization,
            NumPy backend.
  fmmax     a raster, as fmmax takes one: 256 x 256 pixels in lattice coordinates, each holding
            the area-averaged permittivity of a 4 x 4 sub-sampling. ``Formulation.FFT`` and
            ``Formulation.JONES_DIRECT``, eager and under ``jax.jit``, float64.

Reported per row: total transmittance, the phase of the co-polarized zero-order transmitted
amplitude (reference planes at the two faces of the pillar layer for both solvers), and the
wall time of one solve as the median of ``REPEATS`` calls after one untimed call. For the jit
rows the untimed call is the compilation, and its time is printed beside the row.

The two halves run in different environments, because fmmax pins its own JAX. Neither half
imports the other's package:

    python studies/rcwa/metalens_library_cost/metalens_library_cost.py --solver wavebit
    python studies/rcwa/metalens_library_cost/metalens_library_cost.py --solver fmmax

Run from the repository root. The two printed logs, concatenated, are
``studies/rcwa/metalens_library_cost/records/metalens_library_cost.log``.
"""

import argparse
import os
import platform
import sys
import time
from importlib import metadata

import numpy as np

PITCH = 350.0
HEIGHT = 600.0
WAVELENGTH = 532.0
N_PILLAR = 2.43
N_SUBSTRATE = 1.46
DIAMETERS = (150.0, 240.0)
ORDERS = (5, 8, 11, 14)
REPEATS = 3
RASTER = 256
SUBSAMPLING = 4


def environment(packages):
    print(f'python {platform.python_version()}  {platform.system()} {platform.machine()}  '
          f'logical CPUs {os.cpu_count()}')
    model = _cpu_model()
    if model:
        print(f'cpu {model}')
    print('  '.join(f'{name} {metadata.version(name)}' for name in packages))
    threads = {key: os.environ[key] for key in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS')
               if key in os.environ}
    print(f'thread settings {threads or "library defaults"}')


def _cpu_model():
    try:
        with open('/proc/cpuinfo', encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        return None
    return platform.processor() or None


def timed(call):
    """Median wall time of ``REPEATS`` calls after one untimed call, and that first call's time."""
    start = time.perf_counter()
    value = call()
    first = time.perf_counter()-start
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        value = call()
        times.append(time.perf_counter()-start)
    return value, float(np.median(times)), first


def row(label, orders, transmittance, amplitude, seconds, first):
    phase = float(np.angle(amplitude)) % (2*np.pi)
    print(f'    {label:26s} {orders:5d}   {transmittance:.6f}   {phase:.5f}   {seconds:8.2f}   {first:8.2f}',
          flush=True)


HEADER = ('    solver                     orders   T          phase (rad)   time (s)   first call (s)')


def run_wavebit():
    from wavebit import Lattice, Material, Units, eps_from_refractive_index, rcwa

    environment(('wavebit', 'numpy', 'scipy'))
    pillar = Material(epsilon=eps_from_refractive_index(N_PILLAR))
    substrate = Material(epsilon=eps_from_refractive_index(N_SUBSTRATE))

    def problem(diameter):
        return rcwa.Problem(
            units=Units(length='nm'),
            stack=rcwa.Stack(lattice=Lattice.hexagonal(PITCH), bottom=substrate, layers=(
                rcwa.Layer(thickness=HEIGHT, regions=(
                    rcwa.Region(geometry=rcwa.Circle(radius=diameter/2), material=pillar),)),)),
            excitation=rcwa.Excitation(wavelength=WAVELENGTH,
                                       sources=(rcwa.PlaneWave(polarization='s'),)),
        )

    for diameter in DIAMETERS:
        print(f'\ndiameter {diameter:.0f} nm')
        print(HEADER)
        for order in ORDERS:
            numerics = rcwa.Numerics(orders=(order, order))
            result, seconds, first = timed(lambda: rcwa.run(problem(diameter), numerics=numerics))
            total = float(result.T.sum())
            np.testing.assert_allclose(float(result.R.sum())+total, 1., atol=1e-9, rtol=0)
            amplitude = complex(np.asarray(
                result.transmission.order(0, 0).amplitude_ratio_s).reshape(-1)[0])
            row('wavebit numpy', (2*order+1)**2, total, amplitude, seconds, first)


def pillar_raster(diameter):
    """Area-averaged permittivity on the lattice grid of the 120-degree hexagonal cell."""
    a1 = np.array([PITCH, 0.0])
    a2 = np.array([-PITCH/2, PITCH*np.sqrt(3)/2])
    count = RASTER*SUBSAMPLING
    u = (np.arange(count)+0.5)/count-0.5
    grid_u, grid_v = np.meshgrid(u, u, indexing='ij')
    nearest = np.full(grid_u.shape, np.inf)
    for i in (-1, 0, 1):          # minimum image over the neighbouring cells
        for j in (-1, 0, 1):
            x = (grid_u+i)*a1[0]+(grid_v+j)*a2[0]
            y = (grid_u+i)*a1[1]+(grid_v+j)*a2[1]
            nearest = np.minimum(nearest, x*x+y*y)
    inside = (nearest <= (diameter/2)**2).astype(float)
    fill = inside.reshape(RASTER, SUBSAMPLING, RASTER, SUBSAMPLING).mean(axis=(1, 3))
    return 1.0+(N_PILLAR**2-1.0)*fill, a1, a2


def run_fmmax():
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from fmmax import basis, fields, fmm, scattering

    environment(('fmmax', 'jax', 'jaxlib', 'numpy'))
    print(f'raster {RASTER} x {RASTER}, {SUBSAMPLING} x {SUBSAMPLING} area averaging')

    def solver(order, formulation, a1, a2):
        vectors = basis.LatticeVectors(u=jnp.asarray(a1), v=jnp.asarray(a2))
        expansion = basis.generate_expansion(
            vectors, approximate_num_terms=(2*order+1)**2,
            truncation=basis.Truncation.PARALLELOGRAMIC)
        coefficients = np.asarray(expansion.basis_coefficients)
        # The same order set Wavebit keeps, or the comparison is not at equal truncation.
        assert expansion.num_terms == (2*order+1)**2 and np.abs(coefficients).max() == order
        zero = int(np.argmax(np.all(coefficients == 0, axis=1)))
        common = dict(wavelength=jnp.asarray(WAVELENGTH), in_plane_wavevector=jnp.zeros((2,)),
                      primitive_lattice_vectors=vectors, expansion=expansion)

        def solve(permittivity):
            ambient = fmm.eigensolve_isotropic_media(
                permittivity=jnp.ones((1, 1), dtype=complex),
                formulation=fmm.Formulation.FFT, **common)
            substrate = fmm.eigensolve_isotropic_media(
                permittivity=jnp.full((1, 1), N_SUBSTRATE**2, dtype=complex),
                formulation=fmm.Formulation.FFT, **common)
            layer = fmm.eigensolve_isotropic_media(
                permittivity=permittivity, formulation=formulation, **common)
            matrix = scattering.stack_s_matrix(
                [ambient, layer, substrate],
                [jnp.asarray(0.0), jnp.asarray(HEIGHT), jnp.asarray(0.0)])
            forward = jnp.zeros((2*expansion.num_terms, 1), dtype=complex).at[zero, 0].set(1.0)
            transmitted = matrix.s11@forward       # fmmax: s11 start -> end, s21 start -> start
            reflected = matrix.s21@forward
            incident, _ = fields.amplitude_poynting_flux(forward, jnp.zeros_like(forward), ambient)
            through, _ = fields.amplitude_poynting_flux(
                transmitted, jnp.zeros_like(transmitted), substrate)
            _, back = fields.amplitude_poynting_flux(jnp.zeros_like(reflected), reflected, ambient)
            power = jnp.sum(incident).real
            return (jnp.sum(through).real/power, -jnp.sum(back).real/power, transmitted[zero, 0])
        return solve

    formulations = (('fft', fmm.Formulation.FFT), ('jones_direct', fmm.Formulation.JONES_DIRECT))
    for diameter in DIAMETERS:
        raster, a1, a2 = pillar_raster(diameter)
        permittivity = jnp.asarray(raster, dtype=complex)
        print(f'\ndiameter {diameter:.0f} nm')
        print(HEADER)
        for order in ORDERS:
            for name, formulation in formulations:
                solve = solver(order, formulation, a1, a2)
                for mode, call in (('eager', solve), ('jit', jax.jit(solve))):
                    (through, back, amplitude), seconds, first = timed(
                        lambda: jax.block_until_ready(call(permittivity)))
                    # JONES_DIRECT does not conserve power exactly at a finite truncation.
                    assert abs(float(through)+float(back)-1.0) < 1e-3
                    row(f'fmmax {name} {mode}', (2*order+1)**2, float(through),
                        complex(amplitude), seconds, first)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--solver', choices=('wavebit', 'fmmax'), required=True)
    {'wavebit': run_wavebit, 'fmmax': run_fmmax}[parser.parse_args().solver]()
    sys.stdout.flush()
