# Docs

Measurements and decisions, written so they can be cited. `fieldwave/_core/` links to several
of these from code comments — they are why a constant has the value it has, so a document here
is not finished until it states what its numbers decided.

House style: title, `Written:` date, a `Targets:` list of the files it explains, a TL;DR, then
tables. Say what was measured on what hardware, and where the result stops being trustworthy.

| Document | Question it answers | Decision |
| --- | --- | --- |
| [`backend-choice.md`](backend-choice.md) | Which backend, and whether to freeze the geometry, for a *differentiated* run? | PyTorch below `fto≈5`; `jax jit` from `fto≈7` up; re-freeze every step |
| [`analytic-modeling.md`](analytic-modeling.md) | How does the staircase-free shape path converge against the raster and block staircases? | The smooth path for tilted and curved boundaries |
| [`boundary-modeling.md`](boundary-modeling.md) | Can a layer be painted *and* keep a tilted edge exact? | Yes — resolve the painting into polygons once, then transform those |
| [`levelset-modeling.md`](levelset-modeling.md) | Can a topology-optimizable design field skip the staircase *and* give a shape derivative? | Yes — marching triangles, the segment-sum transform, and `grad phi` as the normal field |
| [`nvf-upsampling.md`](nvf-upsampling.md) | What grid should the NVF normal field be taken on? | Not the fine one: it lost accuracy in 27 of 32 combinations and cost the most |
| [`torch-compile.md`](torch-compile.md) | Is `torch.compile` worth chasing for this solver? | No — the solve is almost all LAPACK, which the compiler forwards untouched |
| [`oblique-lattice.md`](oblique-lattice.md) | What does `lattice` do to a non-orthogonal (oblique, hexagonal) cell, and what does it cost? | Implemented — ~8x cheaper than the supercell; NVF only, and the raster is read in lattice coordinates |
| [`fourier-truncation-shape.md`](fourier-truncation-shape.md) | Does S4's circular cut of the Fourier order set beat the full rectangle FieldWave keeps? | **Not implemented** — structure-dependent, within ±20%, and negative on a common case |
| [`anisotropic-materials.md`](anisotropic-materials.md) | Can an anisotropic or magnetic material be given to the solver, and what does each missing case cost? | Uniform in-plane tensors already work; only `eps_xz` needs a different solver |
| [`off-diagonal-slot-convention.md`](off-diagonal-slot-convention.md) | If I fill the five permittivity slots by hand, what goes in them — and why did an asymmetric tensor come out wrong? | The slot holds `-eps_xy`; an ordering defect on asymmetric pairs, found and fixed |

## Docs vs tutorials

Audience, not format. **`docs/` is for a reader who already knows the API** and needs to know
what was measured or why a default is what it is; [`../tutorials/`](../tutorials) is for a
reader who does not know it yet.

A tutorial is outgrown — you read it once. A document here is *cited*, including from code. So
a notebook whose purpose is to record a measurement belongs here as markdown instead: stored as
JSON it will not diff, will not review, and will carry stale outputs.
