import jax.numpy as jnp


def get_normal_vector_field_analytical(grid_shape, shapes, period=(1000.0, 1000.0), type_float=jnp.float64):
    """
    Analytical unit normal vector field (nx, ny) from geometric shape definitions
    (jax port of the numpy version). Supports circles, ellipses, and rotated rectangles.

    ``period`` is unpacked as ``(Py, Px)`` to match the numpy implementation's convention.
    Written functionally (no in-place mutation) so it is jit-/grad-compatible.
    """
    Ny, Nx = grid_shape
    Py, Px = float(period[0]), float(period[1])

    x = (jnp.arange(Nx, dtype=type_float) + 0.5) * (Px / Nx)
    y = (jnp.arange(Ny, dtype=type_float) + 0.5) * (Py / Ny)
    X, Y = jnp.meshgrid(x, y)

    nx_total = jnp.zeros((Ny, Nx), dtype=type_float)
    ny_total = jnp.zeros((Ny, Nx), dtype=type_float)

    for s in shapes:
        stype = s.get('type', 'ellipse')
        cx, cy = s.get('cx', Px / 2.0), s.get('cy', Py / 2.0)
        lx, ly = s.get('lx', 500.0), s.get('ly', 500.0)
        angle = s.get('angle', 0.0)

        dx = X - cx
        dy = Y - cy
        cos_a = jnp.cos(jnp.asarray(angle, dtype=type_float))
        sin_a = jnp.sin(jnp.asarray(angle, dtype=type_float))

        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a

        if stype in ('ellipse', 'circle', 'disk'):
            a = lx / 2.0
            b = ly / 2.0
            gxr = 2.0 * xr / (a ** 2)
            gyr = 2.0 * yr / (b ** 2)
            gx = gxr * cos_a - gyr * sin_a
            gy = gxr * sin_a + gyr * cos_a
        elif stype in ('rectangle', 'square'):
            a = lx / 2.0
            b = ly / 2.0
            dist_x = jnp.abs(xr) - a
            dist_y = jnp.abs(yr) - b
            mask_x = dist_x >= dist_y
            gxr = jnp.where(mask_x, jnp.sign(xr), 0.0)
            gyr = jnp.where(~mask_x, jnp.sign(yr), 0.0)
            gx = gxr * cos_a - gyr * sin_a
            gy = gxr * sin_a + gyr * cos_a
        else:
            continue

        norm = jnp.hypot(gx, gy)
        norm = jnp.where(norm == 0, 1.0, norm)
        nx_total = nx_total + gx / norm
        ny_total = ny_total + gy / norm

    norm_t = jnp.hypot(nx_total, ny_total)
    norm_t = jnp.where(norm_t == 0, 1.0, norm_t)
    return nx_total / norm_t, ny_total / norm_t
