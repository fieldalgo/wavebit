# Learning FDTD

FDTD advances electric and magnetic fields in time. A useful answer depends on both the physical problem and a discretization that resolves its fastest variation. Follow this sequence before using a result in an inverse design.

## 1. Define the physical problem

Start with the [general FDTD guide](../../guide/fdtd/general-fdtd.md) and [how the solver models a problem](../../guide/fdtd/model.md): specify a finite domain, objects, source, boundary conditions and requested monitors. Distinguish the physical domain from its absorbing boundary region. For a periodic layered structure with plane-wave illumination, follow the [periodic FDTD guide](../../guide/fdtd/time-domain.md).

## 2. Resolve space and time

Choose spatial resolution from wavelength in the material and geometric feature size. Set a stable time step, then increase the run duration until the requested spectral quantity stops changing. A stable step is not a convergence result. Compare [transverse resolution](../../validation/fdtd/fdtd-transverse-resolution.md), [DFT sampling](../../validation/fdtd/fdtd-dft-sampling.md) and [pulse convergence](../../validation/fdtd/fdtd-pulse-convergence.md).

## 3. Separate sources and boundaries

Check pulse support, monitor placement and absorbing-boundary clearance independently. The [source-clearance study](../../validation/fdtd/fdtd-source-clearance.md) and [PML-strength study](../../validation/fdtd/fdtd-pml-strength.md) show why increasing only the absorber strength cannot certify a spectrum.

## 4. Read the requested observable

Field samples, Fourier amplitudes and integrated power answer different questions. Use the monitor definition and its reference surface. Before comparing with RCWA, match [coordinates, layer order, phase origins and polarization frames](../../reference/common/conventions.md), not only reflectance. The [route-parity study](../../validation/fdtd/fdtd-route-parity.md) documents the comparison scope.

## 5. Add materials and gradients

Read [constitutive response and absorbers](../../guide/fdtd/fd-constitutive-absorber.md) before assigning conductive or dispersive media. Then use the [differentiation guide](../../guide/fdtd/fd-differentiation.md): keep discrete layout choices fixed while differentiating continuous inputs, compare a derivative with a finite difference, and refine the forward solve as well as the derivative.

## 6. Choose a worked application

Continue with [radiation](../../guide/fdtd/fd-radiation.md), [metasurfaces](../../guide/fdtd/fd-metasurface.md), or the [multilayer optimization study](../../validation/fdtd/fdtd-multilayer-optimization.md). Their support and convergence statements are specific to the tested configurations.
