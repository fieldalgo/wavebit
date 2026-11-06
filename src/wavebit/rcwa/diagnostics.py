"""Warnings an RCWA run can emit about its answer.

A warning here says the run produced numbers that may be worse than expected, or that it
answered a nearby point; it never says the input was refused (a refusal raises). Each warning
has its own category, so a sweep can silence or escalate one without touching the others::

    import warnings
    from wavebit.rcwa import diagnostics

    warnings.simplefilter('error', diagnostics.RayleighPointWarning)
"""
from .._common.backend import (AutoDetuneWarning, ComplexIncidenceWarning, ExceptionalPointWarning,
                               GainCutoffWarning, RayleighPointWarning, SvdConvergenceWarning)

__all__ = ['AutoDetuneWarning', 'ComplexIncidenceWarning', 'ExceptionalPointWarning',
           'GainCutoffWarning', 'RayleighPointWarning', 'SvdConvergenceWarning']
