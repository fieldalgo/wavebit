"""Real production mutations for every plane-wave illumination regression; restore source bytes."""
from pathlib import Path

import fdtd_sphere_grad_mutations as controls


controls.LOGS = Path(__file__).resolve().parent/'results/fdtd_traced_illumination_controls'
controls.PREFIX = 'tests/test_fd_traced_illumination_grad.py::'
controls.MUTATIONS = [
    ('detach_order_power', '_assemble.py',
     "    R = R_s + R_p\n    if monitor.side == 'reflect':",
     "    R = (R_s + R_p).detach() if hasattr(R_s, 'detach') else R_s + R_p\n"
     "    if monitor.side == 'reflect':",
     'test_a_pillar_under_a_bloch_plane_wave_differentiates_flux_and_orders[torch]'),
    ('detach_box_flux', '_assemble.py',
     '                net, parts = _post.box_flux(rt, accs, faces, grid.dx)\n'
     '                out[m.name] = FluxResult(',
     '                net, parts = _post.box_flux(rt, accs, faces, grid.dx)\n'
     "                net = net.detach() if hasattr(net, 'detach') else net\n"
     '                out[m.name] = FluxResult(',
     'test_a_body_inside_a_tfsf_box_differentiates_cross_section_and_far_field[torch]'),
    ('drop_forward_tangent', '_spec.py',
     'return any(forward_ad.unpack_dual(layer).tangent is not None for layer in layers)',
     'return False',
     'test_a_pillar_under_a_bloch_plane_wave_differentiates_flux_and_orders[torch]'),
    ('skip_column_media', '_lower/shape_ranges.py',
     "if getattr(term, 'kind', None) in ('PlaneWave', 'TFSF'):",
     'if False:',
     'test_plane_wave_media_must_lie_outside_every_traced_support'),
    ('skip_tfsf_column_media', '_lower/shape_ranges.py',
     "if getattr(term, 'kind', None) in ('PlaneWave', 'TFSF'):",
     "if getattr(term, 'kind', None) in ('PlaneWave',):",
     'test_a_tfsf_normalisation_plane_must_lie_outside_a_traced_support'),
    ('skip_diffraction_media', '_lower/shape_ranges.py',
     'if isinstance(m, Diffraction):\n            layout, _ = diffraction_layout(',
     'if False:\n            layout, _ = diffraction_layout(',
     'test_plane_wave_media_must_lie_outside_every_traced_support'),
    ('check_concrete_partners', '_lower/shape_ranges.py',
     '                continue                    # concrete partners are read at their current values',
     '                pass',
     'test_plane_wave_media_must_lie_outside_every_traced_support'),
    ('read_layer_strictly_inside', '_lower/shape_ranges.py',
     'if any(lo <= k < hi for k in cells):', 'if any(lo < k < hi for k in cells):',
     'test_plane_wave_media_must_lie_outside_every_traced_support'),
    ('read_layer_end_inclusive', '_lower/shape_ranges.py',
     'if any(lo <= k < hi for k in cells):', 'if any(lo <= k <= hi for k in cells):',
     'test_plane_wave_media_must_lie_outside_every_traced_support'),
    ('unguarded_layer_share', '_post.py',
     '    if any(_shape_traced(v) for v in leaves(shape)):\n        raise ValueError(\n'
     "            f'The medium of cell layer",
     '    if False:\n        raise ValueError(\n'
     "            f'The medium of cell layer",
     'test_a_plane_wave_medium_is_never_read_through_a_traced_shape'),
]


if __name__ == '__main__':
    controls.main()
