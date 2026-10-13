"""Assign a tensor material directly; run from the repository with PYTHONPATH=."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.25])
solver = _legacy_wavebit.rcwa(
    wavelength=600., period=(600., 600.), thickness=[220.],
    structure=[{'background': crystal, 'shapes': []}],
    truncation_order=(0, 0), phi=0.,
)
result = solver.solve()
print(f'Uniform film: R_s={float(result.s_inc.R.sum()):.9f}, '
      f'R_p={float(result.p_inc.R.sum()):.9f}')

# Rotate the material axes, leaving the geometry unchanged.
rotated = crystal.rotated(np.deg2rad(30.), axis='z')
result = solver.solve(structure=[{'background': rotated, 'shapes': []}])
print(f'Rotated film: cross-polarized T_p={float(result.s_inc.T_p.sum()):.9f}')

# The table is assigned once and reevaluated at each solve wavelength.
dispersive = wavebit.TabulatedMaterial(
    wavelengths=[500., 700.], wavelength_unit='nm', epsilon=[[2., 3., 2.5], [3., 4., 2.9]],
)
solver.structure = [{'background': dispersive, 'shapes': []}]
for wavelength in (550., 600., 650.):
    result = solver.solve(wavelength=wavelength)
    print(f'{wavelength:.0f} nm: R_s={float(result.s_inc.R.sum()):.9f}')
