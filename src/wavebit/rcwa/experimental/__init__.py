"""Experimental RCWA formulations with narrower, explicit physical admission.

These research builders preserve their existing numerical implementation and
validation contracts. They construct convolution operators; they do not replace
``rcwa.Problem`` or silently change its factorization. Use their published
cell-centered operators with ``rcwa.operators`` when the retained order window
matches. Their API may evolve independently of the primary lifecycle.

Each builder takes ``orders``, the Fourier half-widths along the two lattice directions,
spelled as in ``rcwa.Numerics(orders=...)``: a nonnegative integer for both, or a pair.
"""
import inspect
import re
from numbers import Integral

import numpy as np

from ..._common.declarations import renamed_attributes, renamed_keyword_message
from ...errors import ValidationError
from .._engine.conforming import ConformingNVF as _ConformingNVF
from .._engine.conforming.affine_profile import AffineProfileNVF as _AffineProfileNVF
from .._engine.conforming.cartesian_rectangle import CartesianRectangleNVF as _CartesianRectangleNVF
from .._engine.conforming.cartesian_pair import CartesianRectanglePairNVF as _CartesianRectanglePairNVF
from .._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from .._engine.conforming.cartesian_pair_transmission_galerkin import (
    CartesianPairTransmissionGalerkinGapCell as _TransmissionGapCell)

__all__ = ['ConformingNVF', 'AffineProfileNVF', 'CartesianRectangleNVF',
           'CartesianRectanglePairNVF', 'CartesianPairOpeningPlan',
           'CartesianPairTransmissionGalerkinGapCell']

_PREFIX = 'rcwa.experimental.'


def _orders(where, orders):
    """``orders`` as a pair of Python ints, or the refusal under its public name."""
    pair = (orders, orders) if isinstance(orders, Integral) else orders
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0
                   for n in pair)):
        raise ValidationError(f'{where}(orders=...) must be a nonnegative integer or a pair of '
                              f'them, one Fourier half-width per lattice direction.',
                              code='experimental.orders', path='orders')
    return tuple(int(n) for n in pair)


def _refuse(where, given, renamed):
    """Refuse the removed keywords in ``given`` by name, and any other unknown keyword."""
    gone = [k for k in given if k in renamed]
    if gone:
        raise TypeError(renamed_keyword_message(where, gone, renamed))
    if given:
        raise TypeError(f'{where}() does not take {sorted(given)}.')


def _build(engine, *args, **kwargs):
    """Construct the engine builder, naming its order keyword as the public one in a refusal."""
    try:
        return engine(*args, **kwargs)
    except ValueError as error:
        text = str(error)
        public = re.sub(r'\b(?:truncation_order|order)\b', 'orders', text)
        if public == text:
            raise
        raise ValueError(public) from error


def _doc(engine, note=''):
    """The engine builder's description, with its order keyword spelled ``orders``."""
    text = inspect.cleandoc(engine.__doc__).replace('``truncation_order``', '``orders``')
    return text + (f'\n\n{note}' if note else '')


class _NVFBuilder:
    """The public face of an engine NVF builder, which it holds: ``orders``, its methods."""
    _renamed = {'truncation_order': 'orders'}

    @property
    def orders(self):
        """The Fourier half-widths along the two lattice directions."""
        return self._builder.truncation_order

    def __repr__(self):
        return f'{_PREFIX}{type(self).__name__}(orders={self.orders})'

    __getattr__ = renamed_attributes(_renamed, prefix=_PREFIX)


class ConformingNVF(_NVFBuilder):
    __doc__ = _doc(_ConformingNVF)

    def __init__(self, orders=3, **cell_settings):
        _refuse('ConformingNVF', [k for k in cell_settings if k in self._renamed], self._renamed)
        self._builder = _build(_ConformingNVF, _orders('ConformingNVF', orders), **cell_settings)

    def plan(self, structure, lattice):
        """The immutable plan of concrete geometry; see the class description."""
        return self._builder.plan(structure, lattice)

    def conv_matrices(self, structure, lattice, *, plan=None, xp=np):
        """The constitutive convolution matrices; see the class description."""
        return self._builder.conv_matrices(structure, lattice, plan=plan, xp=xp)


class AffineProfileNVF(_NVFBuilder):
    __doc__ = _doc(_AffineProfileNVF)

    def __init__(self, orders=3, **cell_settings):
        _refuse('AffineProfileNVF', [k for k in cell_settings if k in self._renamed], self._renamed)
        self._builder = _build(_AffineProfileNVF, _orders('AffineProfileNVF', orders),
                               **cell_settings)

    def conv_matrices(self, structure, lattice, *, xp=np):
        """The constitutive convolution matrices; see the class description."""
        return self._builder.conv_matrices(structure, lattice, xp=xp)


class CartesianRectangleNVF(_NVFBuilder):
    __doc__ = _doc(_CartesianRectangleNVF,
                   '``orders`` is a nonnegative integer or (x, y) pair of Fourier half-widths.')

    def __init__(self, orders=1, *, refinement=0, **removed):
        _refuse('CartesianRectangleNVF', removed, self._renamed)
        self._builder = _build(_CartesianRectangleNVF, _orders('CartesianRectangleNVF', orders),
                               refinement=refinement)

    @property
    def refinement(self):
        """The refinement level the builder was made with."""
        return self._builder.refinement

    def plan(self, structure, lattice):
        """The immutable plan of concrete geometry; see the class description."""
        return self._builder.plan(structure, lattice)

    def conv_matrices(self, structure, lattice, *, plan, xp=np):
        """The constitutive convolution matrices on ``plan``; see the class description."""
        return self._builder.conv_matrices(structure, lattice, plan=plan, xp=xp)


class CartesianRectanglePairNVF(_NVFBuilder):
    __doc__ = _doc(_CartesianRectanglePairNVF,
                   '``orders`` is a nonnegative integer or (x, y) pair of Fourier half-widths.')

    def __init__(self, orders=1, *, refinement=0, **removed):
        _refuse('CartesianRectanglePairNVF', removed, self._renamed)
        self._builder = _build(_CartesianRectanglePairNVF,
                               _orders('CartesianRectanglePairNVF', orders), refinement=refinement)

    @property
    def refinement(self):
        """The refinement level the builder was made with."""
        return self._builder.refinement

    def plan(self, structure, lattice):
        """The immutable plan of concrete geometry; see the class description."""
        return self._builder.plan(structure, lattice)

    def conv_matrices(self, structure, lattice, *, plan, xp=np):
        """The constitutive convolution matrices on ``plan``; see the class description."""
        return self._builder.conv_matrices(structure, lattice, plan=plan, xp=xp)


class CartesianPairTransmissionGalerkinGapCell:
    __doc__ = _doc(_TransmissionGapCell,
                   '``plan`` is a shift-zero ``CartesianPairOpeningPlan``; ``orders`` is a '
                   'nonnegative integer or (x, y) pair of Fourier half-widths.')
    _renamed = {'order': 'orders'}

    def __init__(self, plan, orders=1, *, reference, branches=2, degree=3, quadrature=80,
                 angular_quadrature=None, coercivity_angle=0., cutoff_order=0, **removed):
        where = 'CartesianPairTransmissionGalerkinGapCell'
        _refuse(where, removed, self._renamed)
        self._cell = _build(_TransmissionGapCell, plan, _orders(where, orders), reference=reference,
                            branches=branches, degree=degree, quadrature=quadrature,
                            angular_quadrature=angular_quadrature,
                            coercivity_angle=coercivity_angle, cutoff_order=cutoff_order)

    @property
    def orders(self):
        """The Fourier half-widths ``(x, y)`` the cell retains."""
        return self._cell.order

    def closure(self, *, gap, basis, widths, height, epsilon, background=1., offset=(0., 0.),
                gap_tangent=1., epsilon_tangent=(0., 0.), background_tangent=0.):
        """``(value, tangent, metadata)``: cell-centered matrix slots for ``rcwa.operators``
        and their gap/material JVP. ``offset`` places the pair relative to the cell centre."""
        return self._cell.closure(gap=gap, basis=basis, widths=widths, height=height,
                                  epsilon=epsilon, background=background, offset=offset,
                                  gap_tangent=gap_tangent, epsilon_tangent=epsilon_tangent,
                                  background_tangent=background_tangent)

    def __repr__(self):
        return f'{_PREFIX}{type(self).__name__}(orders={self.orders})'

    __getattr__ = renamed_attributes(_renamed, prefix=_PREFIX)
