# Conforming NVF cells

| Entry | Question it answers |
| --- | --- |
| `__init__.py` | How do rebased rectangle layers preserve physical geometry and the original Cartesian frame and Fourier order set? |
| `thin_gap.py` | How is the one-sided contact limit retained in a periodic conforming cell? |
| `partitioned_gap.py` | How do material intervals retain their opening derivatives at zero width without adding interface constraints? |
| `continuous_flux.py` | How can body displacement remain continuous across auxiliary mesh nodes? |
| `material_grading.py` | How can mesh resolution follow magnitude and phase material jumps and merge coincident interfaces while parameters stay differentiable? |
| `polynomial_body.py` | How do real and complex polynomial-body cells retain their contact traces and full energy projection? |
| `affine_cell.py` | How does an affine reference profile retain the full oblique metric, exact laminated contact and stable opening derivative, physical field frame and exact directional Fourier set? |
| `affine_profile.py` | How do explicit fractional profiles on nonorthogonal lattices preserve physical materials, offsets, contact and the caller's Fourier modes? |
| `cartesian_rectangle.py` | How does a separated physical rectangle retain its Cartesian edges and finite derivatives on an oblique lattice with an explicit frozen orientation plan? |
| `cartesian_pair.py` | How do two horizontal equal-height rectangles retain physical edges, exact contact state, input identity and finite derivatives through an explicitly planned public adapter? |
| `cartesian_pair_opening.py` | How does the finite pair retain its analytical right-opening response through exact collapsed topology, Laurent weak operators and moving Fourier weights? |
| `cartesian_pair_quadratic.py` | Does a quadratic potential retain a finite contact derivative on the same geometry and periodic collapse graph? |
| `cartesian_pair_galerkin.py` | Can real linear or quadratic potential corrections for every retained field preserve reciprocity, passivity and complex-linear material/contact derivatives in the original weak equations? |
| `cartesian_pair_polynomial_galerkin.py` | Can cubic and quartic interface/interior potentials preserve the same original constraints and contact/material derivatives? |
| `triangle_polynomials.py` | Can real Bernstein gradients, products and moving Fourier means be integrated without dividing by a vanishing triangle area? |
| `periodic_rectangle_galerkin.py` | Can the same retained-field constraints cross a rectangle's periodic closing gap with a finite original contact derivative? |
| `periodic_seam_trace.py` | Can continuous positive-gap potentials supply the unrestricted normal contact trace through their original material energy? |
| `periodic_seam_galerkin.py` | Can the complete retained tensor couple those seam fields to the quadratic mesh, with finite contact and positive-gap derivatives? |
| `cartesian_pair_seam_galerkin.py` | Can real continuous internal-gap traces share the whole pair tensor and its exact contact/material derivative? |
| `cartesian_pair_corner_galerkin.py` | Can fixed real corner and optional junction powers join that whole tensor and its positive-gap/contact/material derivative? |
| `cartesian_pair_transmission_galerkin.py` | Can fixed reference transmission modes define a separate complete contact/positive-gap tensor and material JVP? |
| `pair_transmission_modes.py` | Which finite-energy sector solutions preserve local potential and normal-flux continuity with one-sided interface derivatives? |
| `pair_transmission_mode_precision.py` | How do complete characteristic roots and factored angular states preserve weak transmission contrasts and reflected field spans? |
| `pair_transmission_fields.py` | How can all selected real transmission directions share the fitted common polynomial coordinates and their moving gradients? |
| `pair_transmission_common_precision.py` | How can complete higher-degree polynomial and fixed transmission coordinates retain accurate gradients and moving-sample variations before weak projection? |
| `pair_transmission_extended_precision.py` | How do all nested transmission cutoffs retain complete polynomial and moving gradients through large cancellations? |
| `pair_complex_exponential.py` | Can complete complex sector powers and Fourier phases use real compensated exponential and trigonometric factors without losing primitive precision? |
| `pair_positive_fields.py` | Can complete moving positive-gap fields retain primitive coordinate precision, including normalized collapsing gradients? |
| `pair_positive_precision.py` | Can the full projected geometric metric condition positive-gap equations without removing a potential or shifting a pivot? |
| `pair_positive_energy_precision.py` | Do independently reconstructed complete positive-gap fields retain physical energy and Fourier constraints through coefficient cancellation? |
| `pair_corner_fields.py` | How do exact fifth-root remainders and nested angular powers preserve the complete corner field space? |
| `pair_radial_powers.py` | How can exact root factors remove both polynomial overlap and nearby-exponent cancellation without discarding a radial direction? |
| `pair_oscillatory_coefficients.py` | Which exact scalar weights and Taylor coefficients define the fixed real exponent columns? |
| `pair_oscillatory_scalar.py` | How can compensated scalar powers evaluate those complete columns without repeated dense exponentials? |
| `pair_corner_precision.py` | How are complete real coordinate operations evaluated with compensated arithmetic before rounding? |
| `pair_extended_precision.py` | How can an extra arithmetic word preserve strongly cancelling trial potentials without changing the physical equations? |
| `pair_oscillatory_polynomials.py` | How do exact angular Bernstein coefficients retain every oscillatory corner direction and its polynomial overlap? |
| `pair_oscillatory_fields.py` | How are full symmetric oscillatory fields and their moving gradients evaluated without amplifying coordinate roundoff? |
| `pair_oscillatory_radial.py` | How can a fixed real logarithmic cosine/sine pair retain both complete radial directions and their derivatives without cancelling nearby powers? |
| `pair_radial_fields.py` | How do homogeneous normalized hats preserve complete radial values and both spatial and moving-field derivatives? |
| `pair_symmetric_mesh.py` | How can the explicit pair trial mesh preserve reflection and the original exact contact quotient? |
| `pair_corner_prefactors.py` | Which enrichment coordinates are exactly redundant on the selected mesh, including identities across corner labels? |
| `pair_corner_coordinates.py` | How can every symmetric-mesh radial subset retain accurate original weak equations before material assembly? |
| `pair_corner_common_precision.py` | How can the complete polynomial and real root-20 corner transform preserve the unit-simplex potential before contact integration? |
| `pair_complex_precision.py` | How can complete complex coordinate sums, products and exponentials retain their high/low parts? |
| `triangle_exponential_precision.py` | Can complete triangle Fourier moments and their variations remain accurate for repeated or nearly repeated phases? |
| `pair_contact_fourier.py` | How can the original contact measures supply precise free Fourier blocks before large common-field cancellations? |
| `pair_complete_solve.py` | Can a complete complex factorization retain every coordinate when its leading floating-point matrix is singular or poorly conditioned? |
| `pair_contact_retained.py` | Can the full contact equations and physical variations be solved through a smaller retained block without dense common coordinate transforms? |
| `pair_contact_precision.py` | Can every contact equation and high-degree geometric coordinate remain complete before and after Fourier projection? |
| `pair_contact_normal_precision.py` | Can all disappearing polynomial and seam-trace fields preserve complete gradients through their geometric coordinate changes? |
| `pair_contact_normal_mean.py` | Can periodic potential integration retain every normal Fourier mean without cancelling large bulk and interface gradients? |
| `pair_contact_quadrature.py` | How can Gaussian moments, physical inverse maps, opening limits and Fourier phases retain enough precision for the complete contact variation? |
| `pair_contact_refinement.py` | How can the contact variation correct its original weak, retained and flux defects together in the same complete space? |
| `pair_corner_assembly.py` | How do added common corner fields retain the original C/Z equations and their physical variations? |
| `pair_corner_partition.py` | How are moving clipped polygons integrated radially from actual corners without negative or tolerance-clipped areas? |
| `pair_corner_quadrature.py` | How do original, corner and continuous trace fields share the same moving positive quadrature? |
| `pair_corner_contact.py` | How does the enlarged physical field space retain its exact contact state and right-opening derivative? |
| `pair_corner_positive.py` | How can unsplit smooth-field products and split trace products preserve the complete positive solution and independent stationary energy? |
| `pair_balanced_response.py` | How can the complete positive-gap derivative retain every coordinate without differentiating an artificial square-root balancing map? |
| `pair_positive_refinement.py` | Can pointwise original weak residuals correct large-coefficient roundoff in the complete positive value and variation without using the independent energy verifier? |
| `pair_residual_arithmetic.py` | Can original retained and flux constraints distinguish their physical defects from roundoff in large cancelling matrix products? |
| `pair_junction_partition.py` | How do all six corner charts retain exact moving geometry, including refined collapsed triangles? |
| `pair_junction_thin_partition.py` | How can both junction endpoints be resolved on every collapsing triangle, including the exterior annulus? |
| `pair_junction_transport.py` | How do fixed reference volume derivatives and moving profile-edge fluxes avoid artificial quadrature-node cancellation? |
| `pair_junction_quadrature.py` | How do complete junction, nodal and trace fields share those volume and boundary variations? |
| `polynomial_pair_trace.py` | Can analytic polynomial remainders retain every real trace column without subtracting nearly equal sampled fields? |
| `projected_pair_contact.py` | Can nearly coincident high-degree closed fields retain all original contact equations when their physical difference is assembled before its energy? |
| `projected_pair_positive.py` | Can streamed real physical gradient coordinates preserve the whole positive-gap solution and its variation? |
| `pair_contact_energy.py` | Can the complete reconstructed contact field independently verify its energy and variation without receiving a candidate tensor? |
| `pair_contact_energy_precision.py` | How can pointwise contact energy and its physical variation retain the complete solver coordinates without receiving a candidate response? |
| `galerkin_error.py` | Can conforming electric and displacement fields bound the closed constitutive error and drive refinement without accepting an unresolved mesh? |
| `equilibrated_flux.py` | Can a full weighted projection tighten the constitutive error bound while keeping the original electric tensor and admissible flux space? |
| `real_trace_coupling.py` | How do periodic and internal traces retain the same original weak solve and independently reconstructed physical energy? |
| `cartesian_pair_normal_frame.py` | Can a material-referenced normal frame retain coercivity and fitted gap/material derivatives without an auxiliary weak solve, and can conjugate-paired energies preserve reciprocity? |
| `triangle_moments.py` | How can quadratic triangle Fourier integrals and their moving derivatives remain finite at collapse and repeated phases? |
| `cartesian_pair_gap.py` | How can the finite pair cross exact contact and positive gaps without dividing its weak solve or thin fields by gap? |
| `cartesian_pair_gap_autodiff.py` | How can a separate value evaluator provide the finite physical gap right-JVP at contact using ordinary JAX x64 operations? |

These kernels build constitutive data. Optical study fixtures and their
fixed-wavelength propagators remain in `studies/`. The public entry points are
`wavebit.ConformingNVF` for physical rectangle layers and
`wavebit.AffineProfileNVF` for explicit fractional profiles. The separate
experimental `wavebit.CartesianRectangleNVF` admits one separated physical
rectangle per patterned layer on its planned mesh. The explicit experimental
`wavebit.CartesianRectanglePairNVF` admits two horizontal equal-height
rectangles with frozen input indices and separated/contact charts. Neither
builder certifies optical convergence. Strip-profile kernel
inputs retain the historical reference length; the Cartesian rectangle
and pair kernels use physical lengths. Direct kernel inputs are implementation details.

The separate NumPy-only `CartesianPairOpeningPlan` and
`CartesianPairOpeningCell` expose an explicit finite right-opening response
from their submodule. They do not change ordinary contact replay or JAX
tangents. See the [manual](../../../docs/guide/cartesian-pair-opening.md).

The separate `CartesianPairGapEvaluator` supports NumPy values and JAX x64
gap derivatives on the same explicit finite family. See the
[gap differentiation manual](../../../docs/guide/cartesian-pair-gap-autodiff.md).
Other live-parameter gradients and optical convergence remain unqualified.
