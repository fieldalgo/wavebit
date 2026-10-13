"""Real production mutations for every centre-tie regression; restore source bytes."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path

import studies.fdtd.sphere_grad.fdtd_sphere_grad_mutations as controls


controls.LOGS = repository_path('studies/results/fdtd_sphere_center_tie_controls')
controls.PREFIX = 'tests/fdtd/test_fd_sphere_center_tie.py::'
controls.MUTATIONS = [
    ('fold_with_abs', '_shapes/_exact.py',
     '    return s, s * v + 0.', '    return s, xp.abs(v)',
     'test_the_fold_is_abs_to_the_bit_with_its_sign_as_the_derivative_at_zero[torch]'),
    ('keep_negative_zero', '_shapes/_exact.py',
     '    return s, s * v + 0.', '    return s, s * v',
     'test_the_fold_is_abs_to_the_bit_with_its_sign_as_the_derivative_at_zero[jax]'),
    ('fold_with_abs_in_the_volume', '_shapes/_exact.py',
     '    return s, s * v + 0.', '    return s, xp.abs(v)',
     'test_a_face_through_the_centre_plane_keeps_every_derivative[y-torch]'),
    ('tie_the_rectangle_limit', '_shapes/_exact.py',
     'rect = xp.where(corner > 0, xp.minimum(corner, c), zero)',
     'rect = xp.minimum(corner, c)',
     'test_a_face_through_the_centre_plane_keeps_every_derivative[z-jax]'),
    ('tie_the_two_chord_limit', '_shapes/_exact.py',
     'both = xp.where(shorter > 0, xp.minimum(shorter, c), zero)',
     'both = xp.minimum(shorter, c)',
     'test_a_face_through_the_centre_plane_keeps_every_derivative[z-jax]'),
    ('tie_the_one_chord_limit', '_shapes/_exact.py',
     'one = xp.where(outer > 0, xp.minimum(outer, c), zero)',
     'one = xp.minimum(outer, c)',
     'test_a_face_through_the_centre_plane_keeps_every_derivative[corner-jax]'),
    ('empty_rectangle_reads_c', '_shapes/_exact.py',
     'rect = xp.where(corner > 0, xp.minimum(corner, c), zero)',
     'rect = xp.where(corner > 0, xp.minimum(corner, c), c)',
     'test_the_ball_volume_is_unchanged_to_the_bit[numpy]'),
    ('fold_with_abs_end_to_end', '_shapes/_exact.py',
     '    return s, s * v + 0.', '    return s, xp.abs(v)',
     'test_a_sphere_centred_on_half_node_planes_differentiates_like_its_difference[half-torch]'),
    ('tie_the_rectangle_limit_end_to_end', '_shapes/_exact.py',
     'rect = xp.where(corner > 0, xp.minimum(corner, c), zero)',
     'rect = xp.minimum(corner, c)',
     'test_a_sphere_centred_on_half_node_planes_differentiates_like_its_difference[half-jax]'),
]


if __name__ == '__main__':
    controls.main()
