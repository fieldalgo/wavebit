# Coupled dielectric geometry in a matched absorber

:::{note}
`Absorber` is not in the current `wavebit.fdtd` yet: its absorbing faces are `Open()` with a `PML` (`Numerics(absorber=...)` takes only `fdtd.PML`), and a finite body may not reach them. This page's example therefore uses the earlier interface, `wavebit.compat.legacy.fd`.
:::

This page describes the implemented bounded `wavebit.compat.legacy.fd` route for
a dielectric extrusion that continues into `Absorber`. Its algebra, targeted tests,
fixed physical measurements and completed full/slow regressions are recorded, with
bit-exact freezer comparisons. The measured finite-domain differences remain
substantial, so this route is not a validated low-reflection termination for
arbitrary waveguides.

`Absorber(cells=...)` is a graded matched loss backed by a PEC wall. It is distinct
from `PML`. A coupled, smoothed dielectric can enter this absorber only when its
finite interpolation links connect exactly equal damping rates. For example, a
z-extruded cylinder has transverse xy links and can continue through z absorbers.
A nonzero interpolation link joining unequal rates is refused, including a
typical interface that ends inside the layer.

| Requirement | Contract |
| --- | --- |
| Materials | Positive isotropic, lossless `Material`; `mu=1` everywhere, including the host. |
| Geometry | Concrete voxel geometry without a shape range. Ordinary support, occupancy and wall checks still apply. Shapes alone do not bypass the exact link certificate. |
| Faces | One absorbing axis; other faces PEC, PMC, Periodic or Bloch. No PML, Symmetry or Axis1D in this first route. |
| Sources | Electric Dipoles or no sources. No magnetic source, initial field, PlaneWave or TFSF. |
| Execution | NumPy, JAX or Torch on CPU; fixed steps, time or periods. No automatic Decay, gradient or enclosing transformed solve. |
| Prepared plans | Rebuild when the material, geometry, boundary or static source/memory allocation changes. |

This compact API example uses the executed cylinder calibration geometry and
excitation, with one Record monitor. This shortened snippet has not been executed
separately. The measurement instrument additionally records both sides, magnetic
components and a DFT plane. The band and geometry
are explicit so that the example does not depend on an inferred wavelength:

```python
# Earlier interface: Absorber is not in the current wavebit.fdtd yet.
from wavebit.compat.legacy import fd

center = (.015, .0075, 0.)
scene = fd.Scene(
    domain=fd.Domain(size=(.8, .8, 3.2)),
    resolution=fd.Resolution(dx=.05),
    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.Absorber(cells=8)),
    structures=(fd.Structure(
        fd.Cylinder(radius=.16, height=3.4, axis='z', center=center),
        fd.Material(eps=2.25),
    ),),
    sources=(fd.Dipole(
        'drive', center=center, component='Ex',
        band=fd.Band(wavelength=1., df=.15, cutoff=1e-3),
    ),),
    monitors=(fd.Record('out', at=(.015, .0075, .4), component='Ex'),),
    run=fd.Run(time=16., courant=.5),
)
prepared = fd.plan(scene)
report = fd.check(scene)
result = fd.run(scene, plan=prepared, backend='numpy', device='cpu')
trace = result['out']
```

The length unit is arbitrary but shared by the domain, wavelength and time
(`c=1`). The eight absorber cells occupy each end of the declared z extent; they
are not added outside the domain. The cylinder extends past both walls to keep
its cross-section uniform through the layers. The material-relative wavelength
and transverse mesh still need adequate resolution; successful planning does
not establish field accuracy. Coarse-mesh and subpixel-accuracy warnings are
expected for this example's geometry and resolution.

The fixed 60-solve study compared finite domains and durations. Its source-
polarized whole electric traces differed by about 8.8–9.2% between short and
longer domains, with magnetic partners differing by about 16–22%; individual
spectral and late-window differences can be larger. All runs with the study's
DFT plane, including its longer-time controls, remained unsettled. These are
sensitivity measurements, not reflection coefficients or continuum error bounds;
a longer finite domain does not supply an exact reference. See the complete
[measurements and warnings](../../validation/fdtd/fd-constitutive-absorber.md).

For this new route, the energy diagnostic reads a positive staggered storage
with the magnetic-loss correction over the whole absorber. The old product of
consecutive magnetic fields alone is not that storage in damped cells. A
source-free admitted discrete system dissipates the corrected storage, subject
to its CFL bound; a driven Gaussian pulse still does work through its tail.

A scene whose coupled support stays clear of absorbing layers retains the
existing observer and field path. An equal-permittivity curved body still
reserves its new flux term when its potential support overlaps the absorber:
its fields can be compared with a homogeneous control, while the two observers'
energy values need not match.

The route does not admit coupled material in CPML or general lossy/magnetic
smoothing. See [the derivation and scoped measurements](../../validation/fdtd/fd-constitutive-absorber.md)
and [the general FDTD API](../../migration/general-fdtd-v1.md) for the remaining contracts.
