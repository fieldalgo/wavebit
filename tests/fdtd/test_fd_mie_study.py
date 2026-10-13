"""Independent manufactured radiation checks the cylinder measuring contour."""
from types import SimpleNamespace
import numpy as np
import pytest
from wavebit.fdtd._kernel.grid import Grid, OFFSETS
from studies.fdtd.subpixel_mie_cylinder.fdtd_subpixel_mie_cylinder import scattered_flux, convergence


def radiation(cpw, pol):
    h = 1/cpw
    grid = Grid(n=(1, 2*cpw, 2*cpw), dx=h, low=(-h/2, -1., -1.))
    fields, coords = {}, {}
    for component in ('Ex', 'Hy', 'Hz') if pol == 'TM' else ('Hx', 'Ey', 'Ez'):
        field, c = component[0], 'xyz'.index(component[1])
        axes = tuple(grid.axis(a, OFFSETS[field][c][a]) for a in range(3))
        coords[component] = axes
        z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
        r = np.sqrt(y*y+z*z)
        r = np.where(r == 0, 1., r)
        u = np.exp(2j*np.pi*r)/np.sqrt(r)
        if component in ('Hy', 'Ez'):
            u *= (z if component == 'Hy' else y)/r*(1+1j/(4*np.pi*r))
        elif component in ('Hz', 'Ey'):
            u *= -(y if component == 'Hz' else z)/r*(1+1j/(4*np.pi*r))
        fields[component] = u[None]
    result = {'box': SimpleNamespace(fields=fields, coords=coords)}
    reference = {'box': SimpleNamespace(fields={c: np.zeros_like(v) for c,v in fields.items()}, coords=coords)}
    return grid, result, reference


@pytest.mark.parametrize('pol', ('TM', 'TE'))
@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
def test_closed_contour_integrates_manufactured_radial_power(pol, backend):
    pytest.importorskip(backend)
    from wavebit.fdtd._runtime import runtime
    rt = runtime(backend)
    # u=exp(ikr)/sqrt(r), with its exact curl-derived companion, carries radial
    # power 1/(2r); every contour surrounding the origin therefore integrates pi.
    # Measured at 20/40/80 cells: order 1.99845 (TM), 1.99904 (TE), finest error
    # 6.061e-4. Opening endpoint weights gives order .762/.722 and error >.0065.
    errors = []
    for cpw in (20, 40, 80):
        grid, result, reference = radiation(cpw, pol)
        for sample in (result, reference):
            sample['box'].fields = {c: rt.array(v, 'complex128')
                                    for c, v in sample['box'].fields.items()}
        value = scattered_flux(result, reference, pol, .55, grid)[0]
        errors.append(value/np.pi-1)
    order = -np.polyfit(np.log((20, 40, 80)), np.log(abs(np.array(errors))), 1)[0]
    assert 1.98 < order < 2.02
    assert abs(errors[-1]) < .0007


def test_opposite_placement_errors_cannot_hide_in_the_signed_mean():
    # Exact manufactured error: +/-h at four placements. Its mean is zero,
    # whereas the RMS has order one. Averaging before squaring would hide it.
    cells = np.array((20, 40, 80))
    errors = np.array((1., -1., 1., -1.))[None, :] / cells[:, None]
    result = convergence(cells, errors)
    assert result['mean']['order'] is None
    assert result['subset_orders'] == dict(count=0, minimum=None, maximum=None, sd=None)
    np.testing.assert_allclose(result['rms_errors'], 1/cells, rtol=1e-14)
    assert abs(result['rms']['order']-1) < 1e-14
