"""Read-time convention changes for immutable, pre-2026-09-27 Meep goldens.

The Meep harness already projected onto each order's physical s/p vectors, but
stored amplitudes at the cell corner. Current public results use the centre.
Changing the reference multiplies amplitude ratios by (-1)**(m+n); it never
changes efficiencies or rewrites the original independent measurements.
"""
import numpy as np


AMPLITUDES = ('r_s', 'r_p', 't_s', 't_p')


def centre_phase(case):
    u = np.asarray(case['orders_u'], dtype=int)
    v = np.asarray(case.get('orders_v', [0]), dtype=int)
    return np.where((v[:, None] + u[None, :]) % 2, -1., 1.)


def centred_meep(data, cases):
    """Copy the loaded archive's readings into the current lateral reference."""
    readings = {key: data[key] for key in data.files}
    for key, value in readings.items():
        fields = key.split('__')
        if fields[0] in cases and fields[-1] in AMPLITUDES:
            readings[key] = value * centre_phase(cases[fields[0]])
    return readings
