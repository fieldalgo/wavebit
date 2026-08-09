# Reference cases

Validation against solvers that are not ours. The expected answer is known — agreement to
~1e-13 — so these are tests, not studies. What makes them their own directory is that
*producing* the reference needs software this project does not ship.

The split is by what each piece needs:

| Piece | Needs |
| --- | --- |
| `interface/` — bridges to RETICOLO, GRCWA and TORCWA | Octave or MATLAB, and the external solver |
| `gen_goldens_reticolo.py` — freezes the reference numbers | the same; run by hand, rarely |
| `golden_reticolo.npz` — the frozen numbers, committed | nothing |
| `test_reticolo_golden.py` — the assertion | nothing; **runs in CI everywhere** |
| `reti_1D.py`, `reti_1Dc.py`, `reti_2D.py` — live side-by-side comparisons | Octave; exploratory, not asserted |

That is the whole point of the arrangement: the expensive half runs once on a machine that has
the tools, and what CI checks on every push is the committed numbers.

**Regenerating a golden is not a way to make a failing test pass.** These numbers did not come
from this codebase, which is exactly what makes them worth having.

RETICOLO is the only oracle wired up so far, and the one reference here that validates
individual diffraction *orders* rather than totals. The GRCWA and TORCWA bridges exist but have
no frozen case yet — adding one means adding files here, not changing test code.
