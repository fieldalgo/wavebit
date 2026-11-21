"""How far is a pillar deflector's rigorous efficiency from what the local periodic approximation
predicts, and how much does gradient ascent on the diameters recover?

The pillar family is that of ``examples/rcwa/metalens_phase_library.py``: TiO2 pillars
(n = 2.43) on fused silica (n = 1.46), hexagonal pitch 350 nm, height 600 nm, 532 nm, normal
incidence from air, transmission into the substrate, diameters 80 to 260 nm in 10 nm steps.

The lattice is written with ``a1 = (pitch*sqrt(3)/2, -pitch/2)`` and ``a2 = (0, pitch)``: the
hexagonal lattice with one row of pillars along ``y``. A deflector of N pillars per ramp period
is the supercell ``(N*a1, a2)`` holding N circles at ``(k - (N-1)/2)*a1``. Its orders ``(m, 0)``
have ``kx = m*2*pi/(N*pitch*sqrt(3)/2)`` and ``ky = 0``, so the deflection is in the ``xz``
plane. ``'s'`` is then ``E`` along ``y`` (transverse to that plane) and ``'p'`` is ``E`` along
``x`` at incidence. The supercell keeps the orders ``|m| <= N*M, |n| <= M``, which are the
primitive cell's ``|m|, |n| <= M`` and everything between them.

The local periodic approximation, as used here:

  library     ``t(d)``, the co-polarized zero-order transmitted amplitude ratio of the primitive
              cell holding one pillar of diameter ``d``, at ``|m|, |n| <= LIBRARY_ORDER``.
  design      for a ramp ``exp(i*(phi0 + 2*pi*k/N))``, pillar ``k`` takes the usable entry
              (``T >= USABLE``) with the largest ``Re(t * conj(target))``; ``phi0`` is the one of
              360 offsets whose design has the largest predicted efficiency.
  prediction  the field under pillar ``k`` is taken to be ``t(d_k)`` times the incident field.
              ``c_m = (1/N) * sum_k t(d_k) * exp(-2*pi*i*m*k/N)`` and
              ``eta_m = |c_m|^2 * F_m``, with ``F_m = kz_m/k0`` for s and
              ``n_sub^2*k0/kz_m`` for p, ``kz_m`` the substrate's normal wavevector of order m.
              That is the *sampled* prediction: one field value per pillar. The *staircase*
              prediction holds ``t(d_k)`` constant over pillar k's cell and is the sampled one
              times ``sinc(m/N)^2``. Both equal the library transmittance for N equal pillars.

Three parts, each a separate process:

    python studies/rcwa/deflector_supercell/deflector_supercell.py --part library
    python studies/rcwa/deflector_supercell/deflector_supercell.py --part deflector --pillars N
    python studies/rcwa/deflector_supercell/deflector_supercell.py --part grid

``library`` prints the table and the anchor: a supercell of four equal pillars against the
primitive cell. ``deflector`` (N = 3, 4, 6, 8, 12) designs, predicts, checks one diameter
derivative against a central difference, runs ``STEPS`` steps of Adam ascent on the N
diameters with PyTorch, prints the process's peak resident memory, and only then solves the
looked-up and the ascended design at every truncation in ``ORDERS`` on NumPy, so that the peak
is the ascent's. ``grid`` measures what the ascent's coarser normal-field grid changes.
``--polarization p`` runs any part for p; ``--no-ascent`` skips the ascent.

Run from the repository root. The record,
``studies/rcwa/deflector_supercell/records/deflector_supercell.log``, is thirteen printed logs
joined in this order: for s, library, deflector for N = 3, 4, 6, 8, 12, grid; then for p,
library and deflector for N = 3, 4, 6, 8, 12.
"""

import argparse
import os
import platform
import sys
import time
from importlib import metadata

import numpy as np

from wavebit import Lattice, Material, Units, eps_from_refractive_index, rcwa

PITCH = 350.0
HEIGHT = 600.0
WAVELENGTH = 532.0
N_PILLAR = 2.43
N_SUBSTRATE = 1.46
DIAMETERS = tuple(float(d) for d in range(80, 261, 10))
D_MIN, D_MAX = DIAMETERS[0], DIAMETERS[-1]
USABLE = 0.9                 # a library entry below this transmittance is not offered to the lookup
LIBRARY_ORDER = 8
PILLARS = (3, 4, 6, 8, 12)
ORDERS = (4, 5, 6)           # M: the supercell keeps |m| <= N*M, |n| <= M
ASCENT_ORDER = 4
ASCENT_GRID = 512            # shape_nvf_grid_size during the ascent; the default is 1024
STEPS = 30
LEARNING_RATE = 2.0          # nm, Adam
FD_STEP = 1e-3               # nm

A1 = np.array([PITCH*np.sqrt(3)/2, -PITCH/2])
A2 = np.array([0.0, PITCH])
K0 = 2*np.pi/WAVELENGTH

PILLAR = Material(epsilon=eps_from_refractive_index(N_PILLAR))
SUBSTRATE = Material(epsilon=eps_from_refractive_index(N_SUBSTRATE))


def environment(packages):
    print(f'python {platform.python_version()}  {platform.system()} {platform.machine()}  '
          f'logical CPUs {os.cpu_count()}')
    model = _cpu_model()
    if model:
        print(f'cpu {model}')
    print('  '.join(f'{name} {metadata.version(name)}' for name in packages))
    threads = {key: os.environ[key] for key in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS') if key in os.environ}
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


def peak_memory_gb():
    """Peak resident memory of this process so far (``ru_maxrss``, kilobytes on Linux)."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6


def problem(diameters, polarization):
    """N circles on the supercell ``(N*a1, a2)``; one diameter is the primitive cell."""
    count = len(diameters)
    regions = tuple(
        rcwa.Region(geometry=rcwa.Circle(radius=diameter/2,
                                         center=tuple((k-(count-1)/2)*A1)), material=PILLAR)
        for k, diameter in enumerate(diameters))
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(lattice=Lattice.from_columns(np.column_stack([count*A1, A2])),
                         bottom=SUBSTRATE,
                         layers=(rcwa.Layer(thickness=HEIGHT, regions=regions),)),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH,
                                   sources=(rcwa.PlaneWave(polarization=polarization),)),
    )


def numerics(count, order, grid=None):
    return rcwa.Numerics(orders=(count*order, order), shape_nvf_grid_size=grid)


def scalar(value):
    return np.asarray(value).reshape(-1)[0]


def solve(diameters, order, polarization, grid=None):
    """One NumPy solve of the supercell, lossless power balance asserted."""
    result = rcwa.run(problem(list(diameters), polarization),
                      numerics=numerics(len(diameters), order, grid))
    reflected, transmitted = float(result.R.sum()), float(result.T.sum())
    np.testing.assert_allclose(reflected+transmitted, 1.0, atol=1e-9, rtol=0)
    return result, reflected, transmitted


def efficiency(result, m):
    return float(scalar(result.transmission.order(m, 0).efficiency))


def amplitude(result, polarization):
    order = result.transmission.order(0, 0)
    return complex(scalar(order.amplitude_ratio_s if polarization == 's'
                          else order.amplitude_ratio_p))


def flux_factor(count, m, polarization):
    """Power in substrate order ``(m, 0)`` per unit ``|c_m|^2``, relative to the incident power."""
    kx = m*2*np.pi/(count*A1[0])
    kz = np.sqrt((N_SUBSTRATE*K0)**2-kx**2)
    return kz/K0 if polarization == 's' else N_SUBSTRATE**2*K0/kz


def predicted(amplitudes, m, polarization):
    """Sampled and staircase predictions of the efficiency into substrate order ``(m, 0)``."""
    count = len(amplitudes)
    coefficient = np.mean(amplitudes*np.exp(-2j*np.pi*m*np.arange(count)/count))
    sampled = float(abs(coefficient)**2*flux_factor(count, m, polarization))
    return sampled, sampled*float(np.sinc(m/count))**2


def library(polarization, order=LIBRARY_ORDER):
    """Transmittance and co-polarized zero-order amplitude of every diameter's primitive cell."""
    transmittance, amplitudes = [], []
    for diameter in DIAMETERS:
        result, _, total = solve([diameter], order, polarization)
        transmittance.append(total)
        amplitudes.append(amplitude(result, polarization))
    return np.array(transmittance), np.array(amplitudes)


def look_up(count, transmittance, amplitudes, polarization):
    """The usable entries with the largest projection on a linear phase ramp, best offset."""
    usable = transmittance >= USABLE
    best = None
    for phi0 in np.arange(360)*2*np.pi/360:
        targets = np.exp(1j*(phi0+2*np.pi*np.arange(count)/count))
        chosen = [int(np.argmax(np.where(usable, (amplitudes*np.conj(target)).real, -np.inf)))
                  for target in targets]
        value = predicted(amplitudes[chosen], 1, polarization)[0]
        if best is None or value > best[0]:
            best = (value, chosen)
    return best[1]


def deflection_angles(count):
    sine = WAVELENGTH/(count*A1[0])
    return (float(np.degrees(np.arcsin(sine))),
            float(np.degrees(np.arcsin(sine/N_SUBSTRATE))))


# ---------------------------------------------------------------------------------------------

def run_library(polarization):
    environment(('wavebit', 'numpy', 'scipy'))
    print(f'\nlibrary, polarization {polarization}, primitive cell, '
          f'{(2*LIBRARY_ORDER+1)**2} orders')
    start = time.perf_counter()
    transmittance, amplitudes = library(polarization)
    seconds = time.perf_counter()-start
    print('    diameter (nm)   T          phase (rad)   |t|^2*n_sub   usable')
    for diameter, total, value in zip(DIAMETERS, transmittance, amplitudes):
        print(f'    {diameter:6.0f}          {total:.6f}   {np.angle(value) % (2*np.pi):.5f}       '
              f'{abs(value)**2*N_SUBSTRATE:.6f}      {"yes" if total >= USABLE else "no"}')
    print(f'    {len(DIAMETERS)} solves in {seconds:.1f} s; '
          f'{int((transmittance >= USABLE).sum())} usable entries')

    print(f'\nanchor, polarization {polarization}: four equal pillars against the primitive cell')
    print('    diameter (nm)   M   orders   T primitive   T supercell   difference   '
          'phase difference (rad)   T - prediction   largest other order')
    for diameter in (150.0, 240.0):
        for order in ORDERS:
            primitive, _, total = solve([diameter], order, polarization)
            supercell, _, total4 = solve([diameter]*4, order, polarization)
            value = amplitude(primitive, polarization)
            sampled, staircase = predicted(np.full(4, value), 0, polarization)
            assert sampled == staircase
            cross = total-abs(value)**2*N_SUBSTRATE
            others = max(efficiency(supercell, m) for m in range(-4*order, 4*order+1) if m)
            shift = np.angle(amplitude(supercell, polarization)/value)
            print(f'    {diameter:6.0f}          {order}   {(8*order+1)*(2*order+1):5d}    '
                  f'{total:.6f}      {total4:.6f}      {total4-total:+.1e}     {shift:+.5f}'
                  f'                  {total-sampled:.1e}          {others:.1e}')
            # The prediction is the co-polarized zero order exactly; the rest of T is the
            # cross-polarized zero order a finite truncation leaves.
            assert abs(total-sampled-cross) < 1e-12 and cross > -1e-12 and others < 1e-20


def run_deflector(count, polarization, ascent):
    packages = ('wavebit', 'numpy', 'scipy') + (('torch',) if ascent else ())
    environment(packages)
    in_air, in_substrate = deflection_angles(count)
    print(f'\ndeflector, polarization {polarization}, N = {count} pillars per period: order (1, 0) '
          f'leaves at {in_substrate:.2f} deg in the substrate ({in_air:.2f} deg once in air)')
    transmittance, amplitudes = library(polarization)
    chosen = look_up(count, transmittance, amplitudes, polarization)
    looked_up = np.array([DIAMETERS[j] for j in chosen])
    sampled, staircase = predicted(amplitudes[chosen], 1, polarization)
    print(f'    library at {(2*LIBRARY_ORDER+1)**2} orders, lookup among T >= {USABLE}')
    print('    looked-up diameters (nm)  ' + ' '.join(f'{d:.0f}' for d in looked_up))
    print('    library phase (rad)       '
          + ' '.join(f'{np.angle(amplitudes[j]) % (2*np.pi):.3f}' for j in chosen))
    print(f'    predicted efficiency into (1, 0): sampled {sampled:.6f}   staircase {staircase:.6f}')

    designs = [('looked-up', looked_up)]
    if ascent:
        designs.append(('ascended', run_ascent(looked_up, polarization)))
        ascended = designs[1][1]
        # The prediction for the ascended diameters, from primitive-cell solves at them.
        own = [solve([diameter], LIBRARY_ORDER, polarization) for diameter in ascended]
        own_amplitudes = np.array([amplitude(result, polarization) for result, _, _ in own])
        again = predicted(own_amplitudes, 1, polarization)
        print('    ascended diameters (nm)   ' + ' '.join(f'{d:.2f}' for d in ascended))
        print('    their primitive-cell T    ' + ' '.join(f'{total:.3f}' for _, _, total in own))
        print('    their phase (rad)         '
              + ' '.join(f'{np.angle(value) % (2*np.pi):.3f}' for value in own_amplitudes))
        print(f'    predicted efficiency into (1, 0) for them: sampled {again[0]:.6f}   '
              f'staircase {again[1]:.6f}')

    print('\n    rigorous supercell solves, NumPy, default normal-field grid')
    print('    design      M   orders   eta(1,0)   eta(0,0)   eta(-1,0)  eta(2,0)   '
          'T total    R total    time (s)')
    for name, diameters in designs:
        for order in ORDERS:
            start = time.perf_counter()
            result, reflected, total = solve(diameters, order, polarization)
            seconds = time.perf_counter()-start
            print(f'    {name:10s}  {order}   {(2*count*order+1)*(2*order+1):5d}    '
                  f'{efficiency(result, 1):.6f}   {efficiency(result, 0):.6f}   '
                  f'{efficiency(result, -1):.6f}   {efficiency(result, 2):.6f}   '
                  f'{total:.6f}   {reflected:.6f}   {seconds:7.1f}', flush=True)
    print(f'    process peak resident memory after these solves {peak_memory_gb():.2f} GB')


def run_ascent(looked_up, polarization):
    """Adam ascent on the diameters, clipped to the library's range. Returns the best iterate."""
    import torch

    count = len(looked_up)
    settings = numerics(count, ASCENT_ORDER, ASCENT_GRID)

    def value_and_gradient(diameters):
        variable = torch.tensor(diameters, dtype=torch.float64, requires_grad=True)
        result = rcwa.run(problem([variable[k] for k in range(count)], polarization),
                          numerics=settings, backend='torch')
        value = result.transmission.order(1, 0).efficiency.reshape(-1)[0]
        value.backward()
        return float(value.detach()), variable.grad.detach().numpy().copy()

    print(f'\n    ascent: PyTorch, M = {ASCENT_ORDER} '
          f'({(2*count*ASCENT_ORDER+1)*(2*ASCENT_ORDER+1)} orders), normal-field grid '
          f'{ASCENT_GRID}, Adam step {LEARNING_RATE} nm, diameters clipped to '
          f'[{D_MIN:.0f}, {D_MAX:.0f}] nm')

    # One diameter's derivative against a central difference of NumPy solves.
    probe = count//2
    _, gradient = value_and_gradient(looked_up)
    offset = np.zeros(count)
    offset[probe] = FD_STEP
    upper = efficiency(solve(looked_up+offset, ASCENT_ORDER, polarization, ASCENT_GRID)[0], 1)
    lower = efficiency(solve(looked_up-offset, ASCENT_ORDER, polarization, ASCENT_GRID)[0], 1)
    difference = (upper-lower)/(2*FD_STEP)
    relative = abs(gradient[probe]-difference)/abs(difference)
    print(f'    d eta/d diameter[{probe}]: autodiff {gradient[probe]:+.9e}   central difference '
          f'{difference:+.9e} (step {FD_STEP} nm)   relative {relative:.1e}')
    assert relative < 1e-5

    print('    step   eta(1,0)   time (s)   diameters (nm)')
    diameters = looked_up.astype(float).copy()
    first, second = np.zeros(count), np.zeros(count)
    best, times = (-np.inf, None, None), []
    for step in range(STEPS+1):
        start = time.perf_counter()
        value, gradient = value_and_gradient(diameters)
        seconds = time.perf_counter()-start
        times.append(seconds)
        print(f'    {step:4d}   {value:.6f}   {seconds:7.2f}    '
              + ' '.join(f'{d:.2f}' for d in diameters), flush=True)
        if value > best[0]:
            best = (value, step, diameters.copy())
        if step == STEPS:
            break
        first = 0.9*first+0.1*gradient
        second = 0.999*second+0.001*gradient**2
        update = (first/(1-0.9**(step+1)))/(np.sqrt(second/(1-0.999**(step+1)))+1e-12)
        diameters = np.clip(diameters+LEARNING_RATE*update, D_MIN, D_MAX)
    print(f'    best iterate: step {best[1]}, eta(1,0) {best[0]:.6f}; '
          f'{int(((best[2] == D_MIN) | (best[2] == D_MAX)).sum())} diameters on a bound')
    print(f'    one value-and-gradient step: median {np.median(times):.2f} s, '
          f'first {times[0]:.2f} s; process peak resident memory {peak_memory_gb():.2f} GB')
    return best[2]


def run_grid(polarization):
    """What the ascent's coarser normal-field grid changes in the value it climbs."""
    environment(('wavebit', 'numpy', 'scipy'))
    transmittance, amplitudes = library(polarization)
    print(f'\nnormal-field grid, polarization {polarization}, looked-up designs, M = {ASCENT_ORDER}')
    print('    N    eta(1,0) grid 256   grid 512   grid 1024   512 - 1024')
    for count in PILLARS:
        chosen = look_up(count, transmittance, amplitudes, polarization)
        looked_up = np.array([DIAMETERS[j] for j in chosen])
        values = [efficiency(solve(looked_up, ASCENT_ORDER, polarization, grid)[0], 1)
                  for grid in (256, 512, 1024)]
        print(f'    {count:2d}   {values[0]:.6f}            {values[1]:.6f}   {values[2]:.6f}    '
              f'{values[1]-values[2]:+.1e}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--part', choices=('library', 'deflector', 'grid'), required=True)
    parser.add_argument('--pillars', type=int, choices=PILLARS)
    parser.add_argument('--polarization', choices=('s', 'p'), default='s')
    parser.add_argument('--no-ascent', action='store_true')
    arguments = parser.parse_args()
    if arguments.part == 'library':
        run_library(arguments.polarization)
    elif arguments.part == 'grid':
        run_grid(arguments.polarization)
    else:
        if arguments.pillars is None:
            parser.error('--part deflector needs --pillars')
        run_deflector(arguments.pillars, arguments.polarization, not arguments.no_ascent)
    sys.stdout.flush()
