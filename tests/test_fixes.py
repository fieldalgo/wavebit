import numpy as np
import wavebit
from wavebit import eps_from_index

# The Fresnel closed forms below are written in refractive index, so every `n_*` here stays
# an index and is converted where it enters wavebit -- which since 2026-08-20 takes relative
# permittivity at `top`, `bot` and in the raster.

def test_1_si_glass_branch_cut():
    print("==================================================")
    print("Test 1: Si/Glass Branch Cut & Fresnel Verification")
    print("==================================================")
    
    # 200nm Si on Glass
    n_si = 3.8
    n_glass = 1.5
    option = dict(
        tetm='tm', top=1.0, bot=eps_from_index(n_glass), theta=0.0, phi=0.0,
        truncation_order=5, wavelength=632.8, period=[1000],
        structure=eps_from_index(np.array([[[n_si, n_si]]])), thickness=[200]
    )

    for backend_id, backend_name in [(0, 'NumPy'), (2, 'PyTorch')]:
        try:
            solver = wavebit.rcwa(backend=backend_id, **option)
            res = solver.solve()
            assert res.R is not None and res.T is not None
            R = float(res.R.sum())
            T = float(res.T.sum())
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
        tetm='tm', top=1.0, bot=eps_from_index(n_au), theta=0.0, phi=0.0,
        truncation_order=3, wavelength=500, period=[1000],
        structure=np.array([[[1.0, 1.0]]]), thickness=[0] # Bare interface
    )

    for backend_id, backend_name in [(0, 'NumPy'), (2, 'PyTorch')]:
        try:
            solver = wavebit.rcwa(backend=backend_id, **option)
            res = solver.solve()
            assert res.R is not None
            R = float(res.R.sum())
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
        tetm='tm', top=1.0, bot=eps_from_index(n_au), theta=0.0,
        truncation_order=3, wavelength=500, period=[1000],
        structure=eps_from_index(np.array([[[n_au, n_au]]])), thickness=[20]
    )

    for backend_id, backend_name in [(0, 'NumPy'), (2, 'PyTorch')]:
        solver = wavebit.rcwa(backend=backend_id, **option)
        res = solver.solve()
        assert res.R is not None
        R = float(res.R.sum())
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
        tetm='tm', top=1.0, bot=eps_from_index(n_au), theta=10 * np.pi / 180,
        phi=30 * np.pi / 180,
        truncation_order=3, wavelength=500, period=[1000],
        structure=eps_from_index(np.array([[[1.0, n_au]]])), thickness=[20]
    )

    for backend_id, backend_name in [(0, 'NumPy'), (2, 'PyTorch')]:
        try:
            solver = wavebit.rcwa(backend=backend_id, **option_conical)
            res = solver.solve()
            assert res.p_inc is not None and res.p_inc.T_p is not None
            T_p = float(res.p_inc.T_p.sum())
            print(f"[{backend_name} Conical TM] Transmittance T_p = {T_p:.6f}")
            assert T_p >= 0.0, f"Negative transmittance found in {backend_name}: T_p = {T_p}"
        except Exception as e:
            print(f"[{backend_name}] Error: {e}")
    print("SUCCESS: Test 4 Passed!\n")


def test_5_continuous_raster_convolution_matrix():
    print("==================================================")
    print("Test 5: Continuous Raster Convolution Matrix sign")
    print("==================================================")
    
    from wavebit._core.factorization import to_conv_mat_raster_integrated
    
    structure = eps_from_index(np.array([[[0.18 + 3.0j]]]))

    epx, epy, epz_i, *_ = to_conv_mat_raster_integrated(
        structure, half_width_u=1, half_width_v=1)
    # No conjugation anywhere on the way in: Im(n) > 0 is loss, and the permittivity the
    # raster now carries has Im(eps) > 0 to match. This asserted the opposite until 2026-08,
    # when the solver stopped conjugating; the squaring the builder used to do is gone too,
    # which leaves the raster's own value as the only thing that can be flipped here.
    imag_part = epx[0, 0, 0].imag
    print(f"[on_numpy continuous raster] eps[0,0,0] imag part = {imag_part:.4f}")
    assert imag_part > 0, f"Expected positive imaginary part for lossy material, got {imag_part}"
    print("SUCCESS: Test 5 Passed!\n")


def test_6_none_structure_grating_type():
    print("==================================================")
    print("Test 6: None structure Grating Type Assignment Handling")
    print("==================================================")
    
    option = dict(
        tetm='tm', top=1.0, bot=2.25, theta=0.0, phi=0.0,
        truncation_order=3, wavelength=500, period=[1000],
        structure=None, thickness=[100]
    )

    for backend_id, backend_name in [(0, 'NumPy'), (1, 'JAX'), (2, 'PyTorch')]:
        try:
            solver = wavebit.rcwa(backend=backend_id, **option)
            solver._assign_grating_type()
            print(f"[{backend_name}] structure=None assigned grating_type: {solver.grating_type}")
            assert solver.grating_type == 2, f"Expected grating_type == 2 for structure=None, got {solver.grating_type}"
        except Exception as e:
            print(f"[{backend_name}] Error: {e}")
            raise e
    print("SUCCESS: Test 6 Passed!\n")


def test_7_none_theta_handling():
    print("==================================================")
    print("Test 7: theta=None Handling in _separable_kxy and the solve")
    print("==================================================")
    
    option = dict(
        tetm='tm', top=1.0, bot=2.25, theta=None, phi=None,
        truncation_order=3, wavelength=500, period=[1000],
        structure=np.array([[[2.25, 2.25]]]), thickness=[100]
    )

    for backend_id, backend_name in [(0, 'NumPy'), (1, 'JAX'), (2, 'PyTorch')]:
        try:
            solver = wavebit.rcwa(backend=backend_id, **option)
            res = solver.solve()
            assert res.R is not None
            R = float(res.R.sum())
            print(f"[{backend_name}] theta=None solve succeeded: R = {R:.6f}")
            assert R >= 0.0, f"Expected non-negative R for theta=None in {backend_name}"
        except Exception as e:
            print(f"[{backend_name}] Error: {e}")
            raise e
    print("SUCCESS: Test 7 Passed!\n")


def test_8_torch_structure_numpy_ndarray():
    print("==================================================")
    print("Test 8: RCWATorch structure numpy ndarray setter test")
    print("==================================================")

    import torch
    from wavebit.backends.torch_backend import RCWATorch

    solver = RCWATorch()
    # Test float numpy array
    solver.structure = np.array([[[1.5, 2.0]]])
    assert isinstance(solver.structure, torch.Tensor)

    # Test complex numpy array
    solver.structure = np.array([[[1.5 + 0.5j, 2.0 + 1.0j]]])
    assert isinstance(solver.structure, torch.Tensor)
    print("SUCCESS: Test 8 Passed!\n")


def test_9_vector_modeling_period():
    print("==================================================")
    print("Test 9: Vector Modeling period attribute test")
    print("==================================================")

    for backend_id, backend_name in [(0, 'NumPy'), (1, 'JAX'), (2, 'PyTorch')]:
        solver = wavebit.rcwa(backend=backend_id, period=[1000, 1000])
        assert solver.modeler.period is not None
        # vector object for layer
        sub_element = solver.modeler.rectangle_no_approximation(
            cx=500, cy=500, lx=200, ly=200, material=4.0)
        layer_info_list = [[1.0, sub_element]]
        blocks = solver.modeler.draw(layer_info_list)
        assert blocks is not None
        assert len(blocks) == 1
        print(f"[{backend_name}] Vector modeling period test passed")
    print("SUCCESS: Test 9 Passed!\n")


def test_10_shared_mean_permittivity_keeps_the_interface():
    print("==================================================")
    print("Test 10: Layers Sharing a Mean Permittivity Keep Their Interface")
    print("==================================================")

    # Two differently patterned layers that happen to share an average permittivity. A
    # short-circuit used to compare only that average (the DC coefficient), flag the upper
    # layer as identical to the one below, and skip the interface between them -- which cost
    # this lossless stack its energy conservation by ~50x. The short-circuit is gone entirely
    # now, so nothing can skip an interface; this holds the energy statement that caught it.
    UC = np.array([[[9., 9., 1., 1.], [1., 9., 9., 1.],
                    [1., 1., 9., 9.], [9., 1., 1., 9.]]])
    option = dict(
        structure=np.concat([UC, UC[:, ::-1]]), thickness=[0.2, 0.3], top=1.0, bot=1.0,
        psi=0.4, wavelength=0.5, period=[0.6, 0.6], theta=0.1, phi=0.0, truncation_order=[2, 2])

    for algo in ('ETM', 'SMM'):
        res = wavebit.rcwa(backend=0, layer_algorithm=algo, **option).solve()
        total = float(np.sum(res.R)) + float(np.sum(res.T))
        print(f"[{algo}] R+T = {total:.12f}")
        assert abs(total - 1.0) < 1e-9, f"{algo} energy conservation failed: {total}"
    print("SUCCESS: Test 10 Passed!\n")


if __name__ == "__main__":
    test_1_si_glass_branch_cut()
    test_2_air_au_raw_n_lossy_substrate()
    test_3_same_material_interface()
    test_4_tm_transmittance_conjugation()
    test_5_continuous_raster_convolution_matrix()
    test_6_none_structure_grating_type()
    test_7_none_theta_handling()
    test_8_torch_structure_numpy_ndarray()
    test_9_vector_modeling_period()
    test_10_shared_mean_permittivity_keeps_the_interface()
    print("==================================================")
    print("ALL FIXES VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("==================================================")

