"""Experimental material-contrast field; deliberately not a production strategy.

This instrument reuses the piece distances but replaces the material sign jump
with complex contrast weights. It modifies an in-memory function copy only.
The principal real direction of the complex vector's covariance supplies a real
unit normal. Degenerate principal directions and equal-material limits need
separate investigation before this can replace the default field.
"""
import inspect

from wavebit._core import parametric as P


def material_contrast_field(background=1.):
    source = inspect.getsource(P._piece_field)
    changes = [
        ('    d_min = d[0]',
         '    d = [xp.where(slots[pc.shape] != contrast_background, dk,\n'
         '                  xp.full_like(dk, float("inf")))\n'
         '         for dk, pc in zip(d, plan.pieces)]\n'
         '    d_min = d[0]'),
        ('    safe_min = xp.where(on_gamma, one, d_min)',
         '    safe_min = xp.where(on_gamma | ~xp.isfinite(d_min), one, d_min)'),
        ('    own_w = list(w)',
         '    w = [weight * (slots[pc.shape] - contrast_background)\n'
         '         for weight, pc in zip(w, plan.pieces)]\n'
         '    own_w = list(w)'),
        ('own_w[pr.p] + signs[r] * moved', 'own_w[pr.p] - moved'),
        ('    norm = _hypot_safe(xp, sx, sy)\n    return sx / norm, sy / norm',
         '    a = xp.real(sx * xp.conj(sx) - sy * xp.conj(sy))\n'
         '    b = 2 * xp.real(sx * xp.conj(sy))\n'
         '    isotropic = (a == 0) & (b == 0)\n'
         '    angle = .5 * xp.atan2(xp.where(isotropic, zero, b),\n'
         '                           xp.where(isotropic, one, a))\n'
         '    return xp.cos(angle), xp.sin(angle)'),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise AssertionError('material-contrast instrument no longer matches the kernel')
        source = source.replace(old, new, 1)
    namespace = dict(P.__dict__, contrast_background=background)
    exec(compile(source, '<material-contrast instrument>', 'exec'), namespace)
    return namespace['_piece_field']


def main():
    import argparse
    import json
    from unittest.mock import patch
    import numpy as np
    import wavebit
    from contact_instruments import paired_rectangles_li

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=[4, 8, 12])
    parser.add_argument('--grid', type=int, default=256)
    parser.add_argument('--lossy', action='store_true')
    args = parser.parse_args()
    candidate, original = material_contrast_field(), P._piece_field
    builder = P.to_conv_mat_parametric_nvf
    materials = (4.+.2j, 4.+.7j) if args.lossy else (4., 9.)
    def total(gap, order):
        P._PROJECTOR_CACHE.clear()
        shapes = [dict(type='rectangle', cx=sign*(150.+gap/2), cy=0., lx=300.,
                       ly=300., angle=0., material=mat)
                  for sign, mat in zip((-1, 1), materials)]
        solver = wavebit.rcwa(
            backend=0, period=(700., 700.), wavelength=900., top=1., bot=2.25,
            theta=.2, phi=0., psi=0., thickness=(300.,),
            truncation_order=(order, order), normal_grid=args.grid,
            structure=[dict(background=1., shapes=shapes)])
        return float(np.sum(solver.solve().T))
    try:
        for order in args.orders:
            for mode in ('original', 'contrast', 'li'):
                with patch.object(P, '_piece_field', candidate if mode == 'contrast' else original), \
                     patch.object(P, 'to_conv_mat_parametric_nvf', paired_rectangles_li if mode == 'li' else builder):
                    contact = total(0., order)
                    offsets = {str(g): total(g, order)-contact for g in (1e-4, 1e-6)}
                    print(json.dumps(dict(order=order, grid=args.grid, lossy=args.lossy,
                                          mode=mode, contact=contact, offsets=offsets)), flush=True)
    finally:
        P._PROJECTOR_CACHE.clear()


def material_transition_width(delta, scale=1.):
    """Measure the candidate's P_xx=1/2 distance from a weakening real seam.

    This pointwise bisection has no normal-field sampling grid or Maxwell solve.
    It is a diagnostic of the rejected candidate, not a solver control.
    """
    import numpy as np
    from wavebit._core.lattice import as_lattice
    if not np.isfinite(delta) or not 0. < delta <= .01:
        raise ValueError('the diagnostic requires 0 < delta <= 0.01')
    if not np.isfinite(scale) or scale <= 0.:
        raise ValueError('scale must be finite and positive')
    cell = as_lattice(period=(700.*scale, 700.*scale))
    shapes = [dict(type='rectangle', cx=sign*150.*scale, cy=0.,
                   lx=300.*scale, ly=300.*scale, angle=0., material=material)
              for sign, material in [(-1, 4.), (1, 4.+delta)]]
    plan = P.plan_gamma(shapes, cell)
    field = material_contrast_field()
    def pxx(distance):
        nx, _ = field(np, np.asarray([350.*scale+distance]), np.asarray([400.*scale]),
                      shapes, plan, None, cell, np.float64)
        return float(nx[0]**2)
    low, high = 0., 50.*scale
    if not pxx(low) > .5 > pxx(high):
        raise AssertionError('the transition is no longer bracketed')
    for _ in range(50):
        middle = .5*(low+high)
        if pxx(middle) > .5:
            low = middle
        else:
            high = middle
    return .5*(low+high)


if __name__ == '__main__':
    main()
