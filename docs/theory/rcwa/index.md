# RCWA theory

Derivations and mathematical conventions underlying the Fourier-domain solver.

These pages are dated records, written against the solver interface of their date. Their code
snippets use the earlier interface, which from 2026-10-13 was `wavebit.compat.legacy`: where a
snippet calls `wavebit.rcwa(...)`, `wavebit` means `from wavebit.compat import legacy as wavebit`.
That namespace was retired on 2026-10-15; the [pre-retirement revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a) runs them.
The public `wavebit.rcwa` module is introduced in the [RCWA guide](../../guide/rcwa/index.md).

- [The per-order s/p frame the complex amplitudes are quoted in](amplitude-sp-frame.md)
- [Branch cuts — which square root, and the two ways it went wrong](branch-cuts.md)
- [Connecting algorithms — ETM, SMM, and why a plain transfer is not on the list](connecting-algorithms.md)
- [Laurent, direct, inverse, Toeplitz: which word names a rule and which names a matrix](factorization-terms.md)
- [The first-order path — what it is judged against, when no oracle will do it](first-order-formulation.md)
- [Five slots out of nine — the shape every second-order RCWA has, and what happens at its edge](five-slot-limit.md)
- [The formulation, derived in the code's own convention](formulation.md)
- [Rectangular vs circular Fourier-order truncation — measurements](fourier-truncation-shape.md)
- [Magnetic materials — `mu` on layers and half-spaces](magnetic-materials.md)
- [Negative-index media — the signed index, the mirror root, and what `theta` means](negative-index-media.md)
- [The inverse rule on a metal: the pole in `<1/eps>^-1`, and the floor that closes it](nvf-metal-inverse-rule.md)
- [The raster normal-field constructions, and the `raster_normal_method` / `nvf_projector` options](nvf-normal-methods.md)
- [Non-orthogonal (oblique) unit cells — what `lattice` does and what it costs](oblique-lattice.md)
- [The off-diagonal permittivity slots — what they hold, and why an asymmetric tensor came out wrong](off-diagonal-slot-convention.md)
- [Patterned anisotropic layers — the Popov–Nevière factorization](patterned-anisotropic-factorization.md)
- [Rayleigh points — the exact graze, which paths solve it, and what `auto_detune` does](rayleigh-points.md)
- [Li factorization for anisotropic materials](tensor-li-factorization.md)
