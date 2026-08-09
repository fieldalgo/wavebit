# Docs

Measurements and decisions, written so they can be cited. `fieldwave/_core/` links to several
of these from code comments — they are why a constant has the value it has, so a document here
is not finished until it states what its numbers decided.

House style: title, `Written:` date, a `Targets:` list of the files it explains, a TL;DR, then
tables. Say what was measured on what hardware, and where the result stops being trustworthy.

| Document | Question it answers | Decision |
| --- | --- | --- |
| [`backend-choice.md`](backend-choice.md) | Which backend, and whether to freeze the geometry, for a *differentiated* run? | PyTorch below `fto≈5`; `jax jit` from `fto≈7` up; re-freeze every step |
| [`analytic-modeling.md`](analytic-modeling.md) | How does the staircase-free shape path converge against the raster and vector staircases? | Analytic for tilted and curved boundaries |
| [`boundary-modeling.md`](boundary-modeling.md) | Can a layer be painted *and* keep a tilted edge exact? | Yes — resolve the painting into polygons once, then transform those |
| [`nvf-upsampling.md`](nvf-upsampling.md) | What grid should the NVF normal field be taken on? | Not the fine one: it lost accuracy in 27 of 32 combinations and cost the most |
| [`torch-compile.md`](torch-compile.md) | Is `torch.compile` worth chasing for this solver? | No — the solve is almost all LAPACK, which the compiler forwards untouched |
| [`oblique-lattice.md`](oblique-lattice.md) | What would non-orthogonal (oblique, hexagonal) lattice support cost? | See the document |
| [`anisotropic-materials.md`](anisotropic-materials.md) | Can an anisotropic or magnetic material be given to the solver, and what does each missing case cost? | Uniform in-plane tensors already work; only `eps_xz` needs a different solver |
| [`repository-layout.md`](repository-layout.md) | Where does a new file go, and why is the tree shaped this way? | The routing rule, restated for contributors in [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |

## Docs vs tutorials

Audience, not format. **`docs/` is for a reader who already knows the API** and needs to know
what was measured or why a default is what it is; [`../tutorials/`](../tutorials) is for a
reader who does not know it yet.

A tutorial is outgrown — you read it once. A document here is *cited*, including from code. So
a notebook whose purpose is to record a measurement belongs here as markdown instead: stored as
JSON it will not diff, will not review, and will carry stale outputs.
