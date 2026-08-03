import torch


def get_normal_vector_field_analytical(grid_shape, shapes, period=(1000.0, 1000.0),
                                       device=torch.device('cpu'), type_float=torch.float64):
    """
    Analytical unit normal vector field (nx, ny) from geometric shape definitions
    (torch port of the numpy version). Supports circles, ellipses, and rotated rectangles.

    ``period`` is unpacked as ``(Py, Px)`` to match the numpy implementation's convention.
    """
    Ny, Nx = grid_shape
    Py, Px = float(period[0]), float(period[1])

    x = (torch.arange(Nx, device=device, dtype=type_float) + 0.5) * (Px / Nx)
    y = (torch.arange(Ny, device=device, dtype=type_float) + 0.5) * (Py / Ny)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    nx_total = torch.zeros((Ny, Nx), device=device, dtype=type_float)
    ny_total = torch.zeros((Ny, Nx), device=device, dtype=type_float)

    for s in shapes:
        stype = s.get('type', 'ellipse')
        cx, cy = s.get('cx', Px / 2.0), s.get('cy', Py / 2.0)
        lx, ly = s.get('lx', 500.0), s.get('ly', 500.0)
        angle = s.get('angle', 0.0)

        dx = X - cx
        dy = Y - cy
        cos_a = torch.cos(torch.as_tensor(angle, dtype=type_float, device=device))
        sin_a = torch.sin(torch.as_tensor(angle, dtype=type_float, device=device))

        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a

        if stype in ('ellipse', 'circle', 'disk'):
            a = lx / 2.0
            b = ly / 2.0
            gxr = 2.0 * xr / (a ** 2)
            gyr = 2.0 * yr / (b ** 2)
            gx = gxr * cos_a - gyr * sin_a
            gy = gxr * sin_a + gyr * cos_a
            norm = torch.hypot(gx, gy)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            nx_total = nx_total + gx / norm
            ny_total = ny_total + gy / norm
        elif stype in ('rectangle', 'square'):
            a = lx / 2.0
            b = ly / 2.0
            dist_x = torch.abs(xr) - a
            dist_y = torch.abs(yr) - b
            mask_x = dist_x >= dist_y
            gxr = torch.where(mask_x, torch.sign(xr), torch.zeros_like(xr))
            gyr = torch.where(~mask_x, torch.sign(yr), torch.zeros_like(yr))
            gx = gxr * cos_a - gyr * sin_a
            gy = gxr * sin_a + gyr * cos_a
            norm = torch.hypot(gx, gy)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            nx_total = nx_total + gx / norm
            ny_total = ny_total + gy / norm

    norm_t = torch.hypot(nx_total, ny_total)
    norm_t = torch.where(norm_t == 0, torch.ones_like(norm_t), norm_t)
    return nx_total / norm_t, ny_total / norm_t
