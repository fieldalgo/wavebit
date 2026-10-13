"""Current FDTD lifecycle: a pulse through a film with outside PML padding."""
from wavebit import Material, Units, fdtd

problem = fdtd.Problem(
    units=Units(length="um"),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(fdtd.Structure("film", fdtd.Slab(z=(-.1, .1)), Material(epsilon=2.25)),),
    ),
    sources=(fdtd.PlaneWave("illumination", plane=fdtd.Plane(z=.5),
        pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),),
    outputs=fdtd.Outputs(monitors=(fdtd.Record("probe", at=(0., 0., -.3), component="Ey"),)),
)
numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.025),
    duration=fdtd.Duration(periods=20, source="illumination"), absorber=fdtd.PML(cells=12))
report = fdtd.check(problem, numerics=numerics)
assert report.ok
result = fdtd.run(problem, numerics=numerics)
print("samples:", len(result["probe"].time), "time unit: s; field unit:", result["probe"].units)

assert len(result["probe"].time) > 1
