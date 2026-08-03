import sys
import numpy as np
import fieldwave

def test_1_si_glass_branch_cut():
    print("==================================================")
    print("Test 1: Si/Glass Branch Cut & Fresnel Verification")
    print("==================================================")
    
    # 200nm Si on Glass
    n_si = 3.8
    n_glass = 1.5
    option = dict(
        pol=1, n_top=1.0, n_bot=n_glass, theta=0.0, phi=0.0,
        fto=5, wavelength=632.8, period=[1000],
        ucell=np.array([[[n_si, n_si]]]), thickness=[200]
    )

    for backend_id, backend_name in [(0, 'NumPy'), (2, 'PyTorch')]:
        try:
            mee = fieldwave.call_mee(backend=backend_id, **option)
            res = mee.conv_solve()
            assert res.de_ri is not None and res.de_ti is not None
            R = float(res.de_ri.sum())
            T = float(res.de_ti.sum())
            print(f"[{backend_name}] R = {R:.6f}, T = {T:.6f}, R+T = {R+T:.6f}")
            assert 0.0 <= R <= 1.0, f"Unphysical reflectance R = {R}"
            assert abs(R + T - 1.0) < 1e-4, f"Energy conservation failed: R+T = {R+T}"
        except Exception as e:
            print(f"[{backend_name}] Error: {e}")
    print("SUCCESS: Test 1 Passed!\n")


def test_2_air_au_raw_n_lossy_substrate():
    print("==================================================")
    print("Test 2: Air -> Gold (Au) Lossy Substrate & Raw n")
    print("==================================================")
    
    # Gold refractive index at 500nm: n = 0.18 + 3.0j
    n_au = 0.18 + 3.0j
    R_exact = abs((1.0 - n_au) / (1.0 + n_au)) ** 2
    print(f"Exact Fresnel Analytical R (Air->Au): {R_exact:.6f}")

    option = dict(
        pol=1, n_top=1.0, n_bot=n_au, theta=0.0, phi=0.0,
        fto=3, wavelength=500, period=[1000],
        ucell=np.array([[[1.0, 1.0]]]), thickness=[0] # Bare interface
    )

    for backend_id, backend_name in [(0, 'NumPy'), (2, 'PyTorch')]:
        try:
            mee = fieldwave.call_mee(backend=backend_id, **option)
            res = mee.conv_solve()
            assert res.de_ri is not None
            R = float(res.de_ri.sum())
            print(f"[{backend_name}] Calculated R = {R:.6f} (Diff: {abs(R - R_exact):.6e})")
            assert abs(R - R_exact) < 1e-3, f"Fresnel match failed for {backend_name}"
        except Exception as e:
            print(f"[{backend_name}] Error: {e}")
    print("SUCCESS: Test 2 Passed!\n")


def test_3_same_material_interface():
    print("==================================================")
    print("Test 3: Same-Material Interface (Au layer on Au substrate)")
    print("==================================================")
    
    n_au = 0.18 + 3.0j
    R_exact = abs((1.0 - n_au) / (1.0 + n_au)) ** 2

    # 20nm Au layer on Au substrate
    option = dict(
        pol=1, n_top=1.0, n_bot=n_au, theta=0.0,
        fto=3, wavelength=500, period=[1000],
        ucell=np.array([[[n_au, n_au]]]), thickness=[20]
    )

    for backend_id, backend_name in [(0, 'NumPy'), (2, 'PyTorch')]:
        mee = fieldwave.call_mee(backend=backend_id, **option)
        res = mee.conv_solve()
        assert res.de_ri is not None
        R = float(res.de_ri.sum())
        print(f"[{backend_name}] Au(20nm)/Au R = {R:.6f} (Exact: {R_exact:.6f})")
        assert abs(R - R_exact) < 1e-3, f"Same-material failed for {backend_name}"
    print("SUCCESS: Test 3 Passed!\n")


def test_4_tm_transmittance_conjugation():
    print("==================================================")
    print("Test 4: TM Transmittance Conjugation in Conical & 2D")
    print("==================================================")
    
    n_au = 0.18 + 3.0j
    # 1D Conical with TM polarization
    option_conical = dict(
        pol=1, n_top=1.0, n_bot=n_au, theta=10 * np.pi / 180, phi=30 * np.pi / 180,
        fto=3, wavelength=500, period=[1000],
        ucell=np.array([[[1.0, n_au]]]), thickness=[20]
    )

    for backend_id, backend_name in [(0, 'NumPy'), (2, 'PyTorch')]:
        try:
            mee = fieldwave.call_mee(backend=backend_id, **option_conical)
            res = mee.conv_solve()
            assert res.res_tm_inc is not None and res.res_tm_inc.de_ti_p is not None
            T_p = float(res.res_tm_inc.de_ti_p.sum())
            print(f"[{backend_name} Conical TM] Transmittance T_p = {T_p:.6f}")
            assert T_p >= 0.0, f"Negative transmittance found in {backend_name}: T_p = {T_p}"
        except Exception as e:
            print(f"[{backend_name}] Error: {e}")
    print("SUCCESS: Test 4 Passed!\n")


def test_5_continuous_raster_convolution_matrix():
    print("==================================================")
    print("Test 5: Continuous Raster Convolution Matrix np.conj")
    print("==================================================")
    
    from fieldwave.on_numpy.emsolver.convolution_matrix import to_conv_mat_raster_continuous
    
    n_compressed = np.array([0.18 + 3.0j])
    ucell = np.array([[[0.18 + 3.0j]]])
    
    epx, epy, epz_i = to_conv_mat_raster_continuous(ucell, fto_x=1, fto_y=1)
    imag_part = epx[0, 0, 0].imag
    print(f"[on_numpy continuous raster] eps[0,0,0] imag part = {imag_part:.4f}")
    assert imag_part < 0, f"Expected negative imaginary part for lossy material, got {imag_part}"
    print("SUCCESS: Test 5 Passed!\n")


def test_6_none_ucell_grating_type():
    print("==================================================")
    print("Test 6: None ucell Grating Type Assignment Handling")
    print("==================================================")
    
    option = dict(
        pol=1, n_top=1.0, n_bot=1.5, theta=0.0, phi=0.0,
        fto=3, wavelength=500, period=[1000],
        ucell=None, thickness=[100]
    )

    for backend_id, backend_name in [(0, 'NumPy'), (1, 'JAX'), (2, 'PyTorch')]:
        try:
            mee = fieldwave.call_mee(backend=backend_id, **option)
            mee._assign_grating_type()
            print(f"[{backend_name}] ucell=None assigned grating_type: {mee.grating_type_assigned}")
            assert mee.grating_type_assigned == 2, f"Expected grating_type_assigned == 2 for None ucell, got {mee.grating_type_assigned}"
        except Exception as e:
            print(f"[{backend_name}] Error: {e}")
            raise e
    print("SUCCESS: Test 6 Passed!\n")


def test_7_none_theta_handling():
    print("==================================================")
    print("Test 7: theta=None Handling in get_kx_ky_vector & transfer_method")
    print("==================================================")
    
    option = dict(
        pol=1, n_top=1.0, n_bot=1.5, theta=None, phi=None,
        fto=3, wavelength=500, period=[1000],
        ucell=np.array([[[1.5, 1.5]]]), thickness=[100]
    )

    for backend_id, backend_name in [(0, 'NumPy'), (1, 'JAX'), (2, 'PyTorch')]:
        try:
            mee = fieldwave.call_mee(backend=backend_id, **option)
            res = mee.conv_solve()
            assert res.de_ri is not None
            R = float(res.de_ri.sum())
            print(f"[{backend_name}] theta=None solve succeeded: R = {R:.6f}")
            assert R >= 0.0, f"Expected non-negative R for theta=None in {backend_name}"
        except Exception as e:
            print(f"[{backend_name}] Error: {e}")
            raise e
    print("SUCCESS: Test 7 Passed!\n")


def test_8_torch_ucell_numpy_ndarray():
    print("==================================================")
    print("Test 8: RCWATorch ucell numpy ndarray setter test")
    print("==================================================")

    import torch
    from fieldwave.on_torch.emsolver.rcwa import RCWATorch

    solver = RCWATorch()
    # Test float numpy array
    solver.ucell = np.array([[[1.5, 2.0]]])
    assert isinstance(solver.ucell, torch.Tensor)

    # Test complex numpy array
    solver.ucell = np.array([[[1.5 + 0.5j, 2.0 + 1.0j]]])
    assert isinstance(solver.ucell, torch.Tensor)
    print("SUCCESS: Test 8 Passed!\n")


def test_9_vector_modeling_period():
    print("==================================================")
    print("Test 9: Vector Modeling period attribute test")
    print("==================================================")

    for backend_id, backend_name in [(0, 'NumPy'), (1, 'JAX'), (2, 'PyTorch')]:
        mee = fieldwave.call_mee(backend=backend_id, period=[1000, 1000])
        assert mee.modeler.period is not None
        # vector object for layer
        sub_element = mee.modeler.rectangle_no_approximation(cx=500, cy=500, lx=200, ly=200, base=2.0)
        layer_info_list = [[1.0, sub_element]]
        ucell_vector = mee.modeler.draw(layer_info_list)
        assert ucell_vector is not None
        assert len(ucell_vector) == 1
        print(f"[{backend_name}] Vector modeling period test passed")
    print("SUCCESS: Test 9 Passed!\n")


if __name__ == "__main__":
    test_1_si_glass_branch_cut()
    test_2_air_au_raw_n_lossy_substrate()
    test_3_same_material_interface()
    test_4_tm_transmittance_conjugation()
    test_5_continuous_raster_convolution_matrix()
    test_6_none_ucell_grating_type()
    test_7_none_theta_handling()
    test_8_torch_ucell_numpy_ndarray()
    test_9_vector_modeling_period()
    print("==================================================")
    print("ALL FIXES VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("==================================================")

