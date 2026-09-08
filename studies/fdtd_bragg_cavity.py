"""Separate long-lived Bragg cavity transients from spatial detuning."""
import numpy as np
import wavebit


def cavity():
    indices = np.r_[np.tile([2., 1.], 3), 1., np.tile([1., 2.], 3)]
    thickness = np.r_[np.tile([.125, .25], 3), .5, np.tile([.25, .125], 3)]
    return indices, thickness


def analytic_transmission(frequencies):
    """Normal-incidence lossless characteristic matrices, vacuum on both sides."""
    indices, thickness = cavity()
    frequencies = np.asarray(frequencies)
    matrix = np.broadcast_to(np.eye(2, dtype=complex), frequencies.shape+(2, 2)).copy()
    for index, depth in zip(indices, thickness):
        phase = 2*np.pi*frequencies*index*depth
        layer = np.empty_like(matrix)
        layer[..., 0, 0] = layer[..., 1, 1] = np.cos(phase)
        layer[..., 0, 1] = -1j*np.sin(phase)/index
        layer[..., 1, 0] = -1j*np.sin(phase)*index
        matrix = matrix @ layer
    return 2/matrix.sum(axis=(-1, -2))


def measure(resolution, periods):
    indices, thickness = cavity()
    # 200 steps at resolution 40; preserve a comparable window on finer grids.
    window = 200 if periods < 400 else int(5*resolution*np.sqrt(3)/.8)
    result, diagnostic = wavebit.fdtd(
        structure=(indices**2)[:, None, None], thickness=thickness, top=1., bot=1.,
        period=.05, wavelength=1., backend=0, resolution=resolution,
        n_periods=periods, truncation_order=0, tetm='te', return_diagnostics=True,
        convergence_window=window)
    c = diagnostic['convergence']
    return dict(resolution=resolution, periods=periods, window=window,
                T=result.T.item(), error=abs(result.given.t_s.item()-analytic_transmission(1.).item()),
                decay=max(float(v[k]['field_decay']) for v in c['bases'] for k in ('actual', 'reference')),
                change=float(abs(c['amplitude_change']['t_s']).max()),
                settled=bool(c['amplitudes_settled']))


def main():
    frequencies = np.linspace(.98, 1.02, 40001)
    power = abs(analytic_transmission(frequencies))**2
    middle = len(frequencies)//2
    low = np.interp(.5, power[:middle+1], frequencies[:middle+1])
    high = np.interp(.5, power[middle:][::-1], frequencies[middle:][::-1])
    print(f'Continuum linewidth Q: {1/(high-low):.6g}; t(f=1)={analytic_transmission(1.)}')
    for resolution, periods in [(40, 25), (40, 80), (40, 200), (40, 400), (80, 400)]:
        print(measure(resolution, periods), flush=True)


if __name__ == '__main__':
    main()
