"""A moving picture must keep coordinates, field signs and its color scale fixed."""
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

mpl = pytest.importorskip('matplotlib')
mpl.use('Agg')
import matplotlib.pyplot as plt

import wavebit


@pytest.fixture
def record():
    z = np.array([-8., -4., -2., 0., 3., 7., 12.])
    e = np.arange(3*7*2*3).reshape(3, 7, 2, 3) * (1+2j)
    return dict(field_t=np.array([0., 1., 2.]), field_z=z,
                field_x=np.array([-2., 0., 2.]), field_y=np.array([-1., 1.]),
                field_ex_x=np.array([-1., 1., 3.]), field_ex_y=np.array([-2., 0.]),
                field_e=e, field_ex=-e, field_e_component='ey', wavelengths=np.array([10.]),
                n_pml=1, grid=SimpleNamespace(k_tfsf=2, thickness=np.array([3.])))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_component_staggering_and_nonuniform_z_are_drawn_at_the_recorded_positions(record):
    fig, ax, mesh = wavebit.plot_fdtd_field(record, frame=1, component='ex', y=0.)
    coords = mesh.get_coordinates()
    np.testing.assert_allclose(coords[0, :, 0], [-2., 0., 2., 4.])
    np.testing.assert_allclose(coords[:, 0, 1], [-10., -6., -3., -1., 1.5, 5., 9.5, 14.5])
    np.testing.assert_array_equal(mesh.get_array(), record['field_ex'][1, :, 1, :].real)
    assert ax.get_ylim() == (12., -8.)
    assert 'y=0' in ax.get_title()
    tfsf = next(line for line in ax.lines if line.get_label().startswith('TFSF'))
    np.testing.assert_array_equal(tfsf.get_ydata(), [-3., -3.])
    assert fig is ax.figure


def test_real_field_keeps_its_sign_and_all_frames_share_a_color_scale(record):
    _, _, first = wavebit.plot_fdtd_field(record, frame=0, component='ex', y=0.)
    _, _, last = wavebit.plot_fdtd_field(record, frame=-1, component='ex', y=0.)
    assert np.min(first.get_array()) < 0
    assert first.get_clim() == last.get_clim() == (-125., 125.)


def test_animation_updates_values_and_time_without_rescaling(record):
    animation = wavebit.animate_fdtd_field(record, component='ex', y=0., interval=20)
    # Drive the update directly; saving is checked separately by the visual smoke run.
    mesh, title = animation._func(2)
    np.testing.assert_array_equal(mesh.get_array(), record['field_ex'][2, :, 1, :].real)
    assert 't=2 ' in title.get_text()
    assert mesh.get_clim() == (-125., 125.)
    animation._draw_was_started = True


def test_aliasing_is_reported_instead_of_hidden_by_playback_speed(record):
    record['field_t'] = np.array([0., 6., 12.])
    with pytest.warns(UserWarning, match='undersamples'):
        wavebit.plot_fdtd_field(record)


def test_missing_record_and_unrecorded_component_are_explained(record):
    with pytest.raises(ValueError, match='record_field'):
        wavebit.plot_fdtd_field({})
    with pytest.raises(ValueError, match='not recorded'):
        wavebit.plot_fdtd_field(record, component='ez')
    with pytest.raises(IndexError, match='snapshots'):
        wavebit.plot_fdtd_field(record, frame=3)


def test_importing_wavebit_does_not_import_matplotlib():
    subprocess.run([sys.executable, '-c',
                    "import sys, wavebit; assert 'matplotlib' not in sys.modules"], check=True)


def test_magnetic_plot_uses_native_time_and_z_but_physical_boundary_planes(record):
    z = record['field_z']
    record['fields'] = {'hx': dict(values=record['field_e'][:, :-1],
                                  t=record['field_t']-.5, z=(z[:-1]+z[1:])/2,
                                  x=record['field_x'], y=record['field_y'])}
    fig, ax, mesh = wavebit.plot_fdtd_field(record, component='hx', frame=1)
    coords = mesh.get_coordinates()
    # On a nonuniform mesh pcolormesh's cell-edge midpoints need not recover
    # sample centres. Compare its inferred edges instead.
    half_z = (z[:-1]+z[1:])/2
    expected = np.r_[half_z[0]-(half_z[1]-half_z[0])/2,
                     (half_z[:-1]+half_z[1:])/2,
                     half_z[-1]+(half_z[-1]-half_z[-2])/2]
    np.testing.assert_allclose(coords[:, 0, 1], expected)
    assert 't=0.5' in ax.get_title()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [-3., -3.])
    assert 'Z0 H' in fig.axes[1].get_ylabel()
