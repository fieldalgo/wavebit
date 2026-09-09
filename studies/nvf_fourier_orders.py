"""Explicit rectangular retained Fourier orders for the NVF instruments."""
from numbers import Integral

import numpy as np


def orders_xy(order):
    """Return (x, y) half widths; a scalar retains the original square set."""
    pair = (order, order) if isinstance(order, Integral) else order
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2
            or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in pair)):
        raise ValueError('order must be a nonnegative integer or an (x, y) integer pair')
    return tuple(int(n) for n in pair)


def order_grid(order):
    """Return n(y), m(x) arrays in the existing y-major Fourier ordering."""
    x, y = orders_xy(order)
    return np.meshgrid(np.arange(-y, y+1), np.arange(-x, x+1), indexing='ij')


def parse_order(value):
    """CLI spelling: N or NX:NY, always with explicit nonnegative integers."""
    parts = tuple(map(int, value.split(':')))
    order = parts[0] if len(parts) == 1 else parts
    orders_xy(order)
    return order
