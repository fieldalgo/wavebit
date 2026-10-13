# RCWA and FDTD compatibility integration checkpoint

Written: 2026-10-13.
Targets: `src/wavebit/compat/legacy.py`, `src/wavebit/fdtd/_periodic/adapter.py`, `src/wavebit/fdtd/_periodic/scene.py`, `src/wavebit/fdtd/_periodic/legacy.py`, `tests/cross_solver/test_solver_integration.py`, `studies/_support/_fdtd_legacy.py`

This checkpoint combines the RCWA implementation from develop commit
`db37dbb8722b8993c2486c6af4f491e3b1b8bc13` with the FDTD implementation from
fdtd-general commit `d1119445bb2839dd03d2af0b769a2797d0be2d13`. The package now
lives under `src/wavebit/`. The periodic FDTD adapter translates its numerical
kernels to the current RCWA scene, coordinate and scattering conventions.

This page records the periodic compatibility boundary established before the
new public lifecycle. Its retained entry points are
`wavebit.compat.legacy.rcwa(...)`, `wavebit.compat.legacy.fdtd(...)` and
`wavebit.compat.legacy.fd`. The names and units in the sections below describe
that compatibility interface. Where a snippet uses `wavebit` as a short alias,
it means `from wavebit.compat import legacy as wavebit`.

The new `wavebit.rcwa` and `wavebit.fdtd` modules now provide shared physical
materials and the check/prepare/run lifecycle. Start with the
[current API guide](../getting-started/index.md) and the
[implementation checkpoint](../design/unified-api.md#13-implementation-checkpoint-and-acceptance-evidence).
In particular, the compatibility field convention described below is not the
new API's SI magnetic-field convention. Retained periodic and general-FDTD
compatibility material declarations also remain distinct from each other.

## Compatibility contracts

| Contract | Compatibility periodic behavior |
| --- | --- |
| Entry point | `rcwa(...)` returns a solver; `.solve()` returns its result. `fdtd(...)` returns its result directly. |
| Backend | `'numpy'`, `'jax'`, `'torch'` or `wavebit.Backend` members; integers are refused. Omitting the FDTD backend selects the legacy NumPy kernel; explicit `'numpy'` selects the shared scene engine. |
| Scene | A shared `Stack` and `Incidence` work within each solver's supported subset. All layer-indexed scene arrays are bottom-to-top. |
| Coordinates | Bottom interface `z=0`, top `z=sum(thickness)`, +z upward. The physical `top`/`bot` media and the layer order stay fixed when the incident side changes. |
| Incidence | Both periodic routes support top and bottom illumination. Angles are radians. Legacy theta may be signed; explicit backends require `0 <= theta < pi/2`. Cartesian `kx`/`ky` incidence is explicitly unsupported in periodic FDTD. |
| Polarization | TE/TM strings are case-insensitive and `wavebit.Tetm` members are accepted. Complex Jones pairs preserve their overall phase after normalization. |
| Scattering | `result.incident` is the requested state; unit s/p columns are supplied where the route computes them. Pure legacy TE/TM may leave `s_inc`/`p_inc` unset. Amplitudes are cell-centred and use each order's own s/p frame. Reflection refers to the illuminated face, transmission to the opposite face. |
| Result limits | Periodic FDTD supplies R/T, complex s/p amplitudes and order labels, but no Cartesian `r_xyz`/`t_xyz` or RCWA reconstruction state. Positive real exterior media permit `power_entering=1-sum(R)` and total stack absorption from the outgoing transmission. |
| Diagnostic angles | `angles` contains wavelength-dependent polar angles in radians; `azimuth` is phi. The legacy `angles_deg` compatibility key is the same polar angle in degrees. A broadband run holds physical kx and ky fixed. |

Native compatibility `fdtd_grid` coordinates and general `fd.Scene` inputs do not pass through
this periodic adapter. Read their own guides before reusing a periodic example.
The [time-domain guide](../guide/fdtd/time-domain.md) specifies the supported periodic
material, geometry, runtime and differentiation options.

## Field diagnostics are not interchangeable calibration data

Both periodic routes return fields in public coordinates and store H as `Z0 H`.
The legacy frequency monitors divide by the soft-source pulse spectrum;
explicit-backend monitors calibrate each wavelength to unit incident E at the
illuminated interface. Explicit time records instead use a constant
centre-frequency calibration, preserving a causal waveform. Read each
normalization label and use a matching reference before comparing absolute
fields from different routes.

`power_z` is a signed laboratory flux: transmission from the upper half-space
usually has negative z flux, and transmission from below positive z flux.
Diffraction T is an outgoing power fraction on either side. Component records
retain native Yee coordinates and staggered E/H times; use their own `t/z/y/x`
axes. Invariant-y records represent `y=0`, without an artificial phase depending
on an unused second period.

Use the public `source_z`, `z_nodes`, `layer_planes` and `pml_bounds` metadata.
For explicit backends, `mesh_indices` and `background` use physical top-then-bottom
half-space order; each `background` row contains `(epsilon, mu)`. Layer-indexed
arrays instead run bottom-to-top.
The legacy `diagnostics['grid']` remains an internal numerical-grid object with
its original coordinate interpretation; it is not a public field-coordinate
map. The public sampling and plotting helpers read the adapted metadata.

## Migrating a historical periodic caller

1. Replace retired `result.given` access with `result.incident`, and integer
   backend selectors with their names. Keep the distinction between an omitted
   FDTD backend and explicit `'numpy'`.
2. For a scene authored in the former top-to-bottom convention, reverse the
   layer axis together with thickness, permeability, dispersive labels and any
   other layer-indexed data. Keep the physical `top` and `bot` media names.
   Do not reverse a scene already authored for current RCWA.
3. Convert old downward z coordinates using `z_new = total_thickness - z_old`.
   Update monitor positions, record bounds and field plots together. Returned
   Cartesian vectors also need the appropriate polar-E and axial-H reflection;
   the public adapter performs it for new solves.
4. Recheck complex diffraction amplitudes, not just R/T. Moving a single-input
   order amplitude from corner to centre introduces parity `(-1)**(m+n)`;
   former fixed-frame TE/TM amplitudes and negative-tilt incident states can
   additionally require basis-sign changes. Replaying a stored field is not
   equivalent to multiplying all records by order parity.
5. Reinstall the checkout after the move to `src/`, then verify
   `wavebit.__file__`. See [source checkout and debugging](source-checkout.md).

Archived numerical experiments keep their measured inputs, coordinate frames
and results. `studies/_support/_fdtd_legacy.py` is their explicit replay boundary; it is
not the public API for new applications and does not validate the new public
conventions. Historical names or signs in a measurement record should be
identified as such instead of silently rewritten as fresh validation.

## Regression checks

`tests/cross_solver/test_solver_integration.py` checks the public boundary with independent
observables:

| Check | What it distinguishes |
| --- | --- |
| Asymmetric two-layer stacks, unequal exterior media, both sides and TE/TM | A layer-order or port reversal that a symmetric vacuum scene would hide |
| Unequal magnetic exterior media, conical/crossed routes and both ports | Electric-amplitude power conversion with the correct kz/mu admittance, checked against Fresnel powers |
| Nonzero diffraction orders | Complex phase and outgoing basis changes invisible to powers |
| Signed legacy tilt and mixed Jones states | The incident frame's sign, including conical incidence |
| Vacuum records and six-component analytic 3D fields | Public z direction, E/H reflection and lateral phase |
| Invariant-y records and frequency planes | Dependence on an unused y period |
| Scalar TM versus the equivalent Jones p state | Field normalization and the corresponding monitor flux |
| Typed declarations, defaults and unsupported Cartesian incidence | Shared input contracts and explicit capability boundaries |
| JIT material sensitivity versus a finite difference | Preservation of a live JAX material graph through the adapter |

Run these checks and the capability manifest after boundary changes:

```sh
python -m pytest tests/cross_solver/test_solver_integration.py tests/api/test_manifest.py -q
```

This set tests integration conventions. It does not replace resolution, runtime,
PML and external-reference studies for a new physical regime. The
[research-alpha scope](../guide/fdtd/fdtd-alpha.md) retains the method's evidence and limits.
