"""Verify the integration instrument independently of any NVF candidate."""
import numpy as np
import pytest
from contact_quadrature_probe import gauss_axis, seam_grid, coefficients


@pytest.mark.parametrize('degree',[8,16])
def test_graded_quadrature_integrates_plane_wave_moments(degree):
    grid=seam_grid(1e-6,degree)
    x,wx,y,wy=grid
    # A shifted Fourier mode checks sign, axis order, cell normalization, and
    # off-diagonal complex phase; energy conservation alone cannot do that.
    field=np.exp(2j*np.pi*(2*x[None,:]-y[:,None])/700.+.3j)
    got=coefficients(field,grid,3)
    want=np.zeros((13,13),complex);want[5,8]=np.exp(.3j)
    np.testing.assert_allclose(got,want,rtol=0,atol=3e-13)
    assert abs(wx.sum()-700.)<2e-12 and abs(wy.sum()-700.)<2e-12


def test_subpixel_strip_integral_retains_its_width():
    width=1e-6
    x,wx=gauss_axis([0.,350.,350.+width,700.],16)
    y,wy=gauss_axis([0.,700.],16)
    field=np.broadcast_to(((x>350.)&(x<350.+width))[None,:],(len(y),len(x)))
    got=coefficients(field,(x,wx,y,wy),0)[0,0]
    assert got.real==pytest.approx(width/700.,rel=3e-8,abs=0.)
    assert abs(got.imag)==0.
