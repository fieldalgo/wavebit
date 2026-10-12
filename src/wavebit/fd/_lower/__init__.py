"""Lowering: scene -> static ``Plan`` (``plan.py``), and scene + plan -> traced ``Params``
(``params.py``, with the initial field evaluated by ``fields.py``).

The split is the whole point: everything a jitted body branches on is in the plan, and
everything a gradient may flow through is in the params.
"""
from .params import lower
from .plan import Plan, plan, skeleton_of

__all__ = ['Plan', 'plan', 'lower', 'skeleton_of']
