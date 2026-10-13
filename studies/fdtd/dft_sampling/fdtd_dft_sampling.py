"""Measure monitor work and quadrature error against every-step DFT."""
import platform
import time
import numpy as np
from wavebit.compat import legacy as wavebit


def main():
    steps, dt = 400, .02
    source = np.zeros((3, 32, 8, 8), complex)
    source[0, 10] = 1
    t = np.arange(steps)*dt
    pulse = np.exp(-((t-2)/.6)**2)*np.exp(-2j*np.pi*t)
    kw = dict(epsilon=np.ones((32, 8, 8)), spacing=(.1, .1, .1), dt=dt, steps=steps,
              source=source, pulse=pulse, wavelengths=np.linspace(.8, 1.2, 32),
              monitor_z=(8, 14, 20, 24), pml_cells=4)
    reference = wavebit.fdtd_grid(**kw)['electric_spectrum']
    print(f'CPU {platform.machine()}, NumPy {np.__version__}; median of three warm runs')
    print('| stride | DFT evaluations | seconds | relative spectral L2 error |')
    print('| --- | --- | --- | --- |')
    for stride in (1, 2, 4, 8):
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            result = wavebit.fdtd_grid(**kw, dft_subsample=stride)
            elapsed.append(time.perf_counter()-start)
        error = np.linalg.norm(result['electric_spectrum']-reference)/np.linalg.norm(reference)
        print(f'| {stride} | {result["dft_samples"]} | {np.median(elapsed):.4f} | {error:.6g} |', flush=True)


if __name__ == '__main__':
    main()
