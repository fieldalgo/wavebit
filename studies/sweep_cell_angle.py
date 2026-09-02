"""
sweep_cell_angle.py — sweeping a unit-cell angle, and the one way it goes wrong.

The question: what does changing a cell take, and what happens if the raster is built once
and reused across the change?

**The sweep is the instrument here, not the use case.** One change of ``lattice=`` with a
reused array is already the whole defect; sweeping an angle only makes it legible, because
the error then grows with the tilt instead of being a single wrong number that looks like a
lattice effect. Be exact about the use case too: a *continuous* angle is a design variable in
one documented family, the monoclinic chiral metasurfaces where achiral meta-atoms on a
skewed cell give circular dichroism from the lattice alone (Toftul et al., Phys. Rev. Lett.
**133**, 216901 (2024)). Hexagonal against rectangular is a *discrete* choice, not a sweep --
you pick 60 or 90 -- and it hits the same trap on the one change it makes.

**The answer is that it depends which frame the pattern was specified in, and that nothing
tells you which one you asked for.** A raster holds pixel *indices*: pixel ``(i, j)`` means the
parametric point ``u = (i+0.5)/Nx``, ``v = (j+0.5)/Ny``, whose Cartesian position is
``r = (u-0.5)*a1 + (v-0.5)*a2``. Nothing in the array records "radius 100 nm", so holding it
says the pattern is *fractions of the cell* and redrawing it says the pattern is *Cartesian
lengths*. Both are real specifications; the sweep separates them.

Three arms:

* ``redrawn``  -- ``A.grid`` at each angle: the Cartesian frame, a circle at every angle
* ``reused``   -- drawn once and passed on: the lattice frame, sheared by the basis change
* ``analytic`` -- parametric circles, Cartesian by construction and staircase-free; the control
  for the *Cartesian* frame, and not an answer key for the other one

**Neither arm is the mistake.** The lattice frame is the right answer wherever a feature is a
duty cycle or a fill fraction, and one case settles it: scale the whole design -- cell,
thickness and wavelength by ``s`` -- and Maxwell says the answer cannot move. Holding the array
gives 0.124231941 at ``s`` = 0.5, 1, 2, 4, identical to nine digits; redrawing pins the rod at
radius 100 while the cell grows and reads 0.330 / 0.124 / 0.99986 / 0.99999. There the Cartesian
frame is the wrong one.

Tilting and scaling are also not the same case. Holding the array, the pattern keeps its shape
exactly when the basis change is a **similarity**: the axis ratio of the lit pixels stays 1.000
under an isotropic 2x scale, and goes to 2.000 under ``Lx`` alone and 2.414 under a 45-degree
tilt. An angle sweep is therefore where the two frames are easiest to tell apart, which is why
this study sweeps one.

Note what this study is *not* about. Differentiating with respect to the angle is a separate
question with a different answer: redrawing a **binary** raster inside ``jax.grad`` does not
work -- the redraw flips pixels discretely -- but smoothing the indicator over about one pixel
makes it work, to 0.54% against a finite difference. ``docs/oblique-lattice.md`` records both,
in "Differentiating the cell" and "What is binary and what is not"; the short form is that a
raster's basis gradient shears the pattern with the cell unless the array is redrawn, and that
redrawing it under a gradient needs the boundary to stop being two-valued.

Run from the repo root:

    python studies/sweep_cell_angle.py
"""
import numpy as np

import wavebit

#: Equal edge lengths, so the only thing the sweep moves is the angle between them.
EDGE = 640.0
RADIUS = 100.0
N = 256
WAVELENGTH, THICKNESS, EPS_ROD = 900.0, 300.0, 12.25
TRUNCATION = 10
ANGLES = np.arange(90.0, 29.0, -10.0)
#: One angle, where the `redrawn` arm is checked against the raster refining under it.
CONVERGENCE_ANGLE = 60.0


#: Figures belong to the document that shows them, one directory per document, named for what
#: they show rather than numbered -- a number goes stale the moment a section is reordered.
#: Both are committed: `docs/` carried no images before 2026-09-02 and that was a habit rather
#: than a rule (`tutorials/` commits PNGs), and the two frames are a picture before they are a
#: table.
FIG_DIR = 'docs/figures/oblique-lattice/'


def _draw_two_frames(plt):
    """The concept figure: one pixel set, read in two frames, and redrawn in the second.

    Deliberately coarse (24 px) and drawn from the *indices* rather than from a solve, because
    what it has to show is that the lit set never changes -- only where those pixels sit.
    """
    n, radius = 24, 0.30
    t = (np.arange(n) + 0.5) / n - 0.5
    U, V = np.meshgrid(t, t)
    lit = (U ** 2 + V ** 2) < radius ** 2                 # drawn once, at 90 degrees

    def basis(deg):
        g = np.radians(deg)
        return np.array([[1.0, np.cos(g)], [0.0, np.sin(g)]])

    def panel(ax, A, mask, colour, title):
        x, y = U * A[0, 0] + V * A[0, 1], U * A[1, 0] + V * A[1, 1]
        box = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5], [-.5, -.5]]) @ A.T
        ax.plot(box[:, 0], box[:, 1], '-', color='0.55', lw=1.4)
        ax.scatter(x[~mask], y[~mask], s=4, color='0.87')
        ax.scatter(x[mask], y[mask], s=11, color=colour)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.9, 0.9)
        ax.set_title(title, fontsize=9.5)
        for sp in ax.spines.values():
            sp.set_visible(False)

    A90, A45 = basis(90), basis(45)
    x45, y45 = U * A45[0, 0] + V * A45[0, 1], U * A45[1, 0] + V * A45[1, 1]
    redrawn = (x45 ** 2 + y45 ** 2) < radius ** 2         # same expression, the 45-degree cell

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 2.9))
    panel(ax[0], A90, lit, '#2a78d6', 'drawn on the 90-degree cell\n(a circle)')
    panel(ax[1], A45, lit, 'crimson',
          'the SAME array on a 45-degree cell\n(the lattice frame: an ellipse, 2.41:1)')
    panel(ax[2], A45, redrawn, '#2a78d6',
          'redrawn on the 45-degree cell\n(the Cartesian frame: a circle again)')
    fig.tight_layout()
    fig.savefig(FIG_DIR + 'two-frames.png', dpi=140)


def raster_at(cell):
    """A genuine Cartesian circle on ``cell`` -- the same expression at every angle."""
    X, Y = (np.asarray(c) for c in cell.grid((N, N)))
    return np.where(X ** 2 + Y ** 2 < RADIUS ** 2, EPS_ROD, 1.0)


def aspect_of(lit, cell):
    """Axis ratio of the lit pixels *where they actually sit*, from their covariance.

    Read off the second moment rather than by fitting a circle, so it needs no centre and no
    threshold; 1 is round. A few percent above 1 at every angle is the pixelation of a
    finite raster, not a deformation -- the `reused` arm leaves that scale entirely.
    """
    A = np.asarray(cell)
    t = (np.arange(N) + 0.5) / N - 0.5
    U, V = np.meshgrid(t, t)
    x, y = U * A[0, 0] + V * A[0, 1], U * A[1, 0] + V * A[1, 1]
    w = np.linalg.eigvalsh(np.cov(np.stack([x[lit], y[lit]])))
    return float(np.sqrt(w[1] / w[0]))


def solve(cell, structure, **kw):
    s = wavebit.rcwa(backend=0, lattice=cell, structure=structure, thickness=[THICKNESS],
                     wavelength=WAVELENGTH, truncation_order=[TRUNCATION, TRUNCATION],
                     top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0, **kw)
    return float(np.sum(np.asarray(s.solve().T)))


def main():
    first = wavebit.Lattice.oblique(EDGE, EDGE, np.radians(ANGLES[0]))
    stale = raster_at(first)                      # drawn once, then never again

    rows = []
    for gamma in ANGLES:
        cell = wavebit.Lattice.oblique(EDGE, EDGE, np.radians(gamma))
        fresh = raster_at(cell)
        shapes = [{'background': 1.0, 'shapes': [
            {'type': 'circle', 'cx': 0.0, 'cy': 0.0, 'lx': 2 * RADIUS,
             'material': EPS_ROD}]}]
        rows.append((
            gamma,
            solve(cell, fresh[None], factorization='nvf'),
            solve(cell, stale[None], factorization='nvf'),
            solve(cell, shapes),
            aspect_of(fresh > 1.0, cell),
            aspect_of(stale > 1.0, cell),
        ))

    head = ('gamma', 'redrawn', 'reused', 'analytic', 'aspect redrawn', 'aspect reused')
    print(f'{head[0]:>7} {head[1]:>11} {head[2]:>11} {head[3]:>11} '
          f'{head[4]:>15} {head[5]:>14}')
    for g, a, b, c, ar, br in rows:
        print(f'{g:>6.0f}  {a:>11.6f} {b:>11.6f} {c:>11.6f} {ar:>15.4f} {br:>14.4f}')

    ga = np.array([r[0] for r in rows])
    red, reu, ana = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    print(f'\nredrawn vs analytic : worst {np.max(np.abs(red - ana)):.2e}')
    print(f'reused  vs analytic : worst {np.max(np.abs(reu - ana)):.2e}')

    # The gap on the `redrawn` arm is the staircase and nothing else, which is worth measuring
    # rather than asserting: it shrinks with the raster while the truncation is held.
    print('\nthe redrawn arm against the same analytic answer, as the raster is refined '
          f'(gamma = {CONVERGENCE_ANGLE:.0f}):')
    cell = wavebit.Lattice.oblique(EDGE, EDGE, np.radians(CONVERGENCE_ANGLE))
    exact = solve(cell, [{'background': 1.0, 'shapes': [
        {'type': 'circle', 'cx': 0.0, 'cy': 0.0, 'lx': 2 * RADIUS, 'material': EPS_ROD}]}])
    for n in (64, 128, 256, 512):
        X, Y = (np.asarray(c) for c in cell.grid((n, n)))
        got = solve(cell, np.where(X ** 2 + Y ** 2 < RADIUS ** 2, EPS_ROD, 1.0)[None],
                    factorization='nvf')
        print(f'   N = {n:>4}   {abs(got - exact):.1e}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n(matplotlib not installed; skipping the figure)')
        return

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].plot(ga, ana, 'k-', lw=2.2, label='analytic (staircase-free)')
    ax[0].plot(ga, red, 'o-', ms=5, color='#2a78d6', label='raster, redrawn each angle')
    ax[0].plot(ga, reu, 's--', ms=5, color='crimson', label='raster, drawn once and reused')
    ax[0].set_xlabel('angle between a1 and a2 (degrees)')
    ax[0].set_ylabel('total transmission, sum(T)')
    ax[0].set_title('A cell-angle sweep')
    ax[0].invert_xaxis(); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].axhline(1.0, color='k', lw=.8, ls=':')
    ax[1].plot(ga, [r[4] for r in rows], 'o-', ms=5, color='#2a78d6', label='redrawn')
    ax[1].plot(ga, [r[5] for r in rows], 's--', ms=5, color='crimson', label='reused')
    ax[1].set_xlabel('angle between a1 and a2 (degrees)')
    ax[1].set_ylabel('axis ratio of the lit pixels (1 = round)')
    ax[1].set_title('What the solver is actually given')
    ax[1].invert_xaxis(); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR + 'cell-angle-sweep.png', dpi=130)
    print(f'\nfigure: {FIG_DIR}cell-angle-sweep.png')

    _draw_two_frames(plt)
    print(f'figure: {FIG_DIR}two-frames.png')


if __name__ == '__main__':
    main()
