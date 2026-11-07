"""A plan's skeleton held to a frozen one with its floats inside a tolerance.

A plan's skeleton is a SHA-256 over its static content, with every float written in hex
(``src/wavebit/fdtd/_prepare/plan.py``). One last-digit difference anywhere moves the whole
hash, and a static value computed through ``sin``, ``cos`` or ``exp`` is not the same to the
last digit on every platform's math library. Measured 2026-11-08 on the stage-3 conical scene:
Windows' ``ColumnTerm.current_gain`` of the oblique p ``PlaneWave`` came out
``1+6.516909065175717j`` against Linux's ``1+6.5169090651757156j`` (2.7e-16 relative), so the
skeleton differed while every other static value, and every frozen array, agreed.

So a frozen skeleton is checked here in two parts. The structure -- every type, field name,
``int``, ``str``, ``bool`` and ``None`` -- must be the frozen one exactly. Each float and complex
leaf must sit within a relative tolerance of the frozen leaf at the same place. The check
writes the frozen floats back over the live ones that pass and hashes the result, which must
be the frozen skeleton itself. Every leaf of what is hashed is then either live or recorded,
so a pass means the structure is the frozen one and each live float is the frozen float or
within the tolerance of it. A wrong record can make the check fail but never pass. The record
is checked on its own too: written back whole (``rtol=math.inf``), it must rebuild the frozen
hash.
"""

import dataclasses
import hashlib
import importlib
import json
import math

__all__ = ['encoded_static', 'float_leaves', 'rebuilt_skeleton']

_SCALARS = ('none', 'bool', 'int', 'str')


def _plan_module():
    # ``wavebit.fdtd._prepare.plan`` is shadowed by the ``plan`` function its package exports.
    return importlib.import_module('wavebit.fdtd._prepare.plan')


def encoded_static(plan):
    """The encoding the skeleton hashes, rebuilt the way ``plan._finish`` builds it.

    The rebuild is checked against the plan's own skeleton before it is returned, so a change to
    ``_finish``'s omission rule fails here, loudly, and never as a tolerance question.
    """
    module = _plan_module()
    static = {f.name: getattr(plan, f.name) for f in dataclasses.fields(plan)
              if f.name != 'skeleton'}
    for name in ('junctions', 'clips', 'shape_ranges', 'decay'):
        if not getattr(plan, name):
            static.pop(name)
    encoded = module._encode(static)
    if _digest(encoded) != plan.skeleton:
        raise AssertionError('the rebuilt encoding does not hash to the plan\'s own skeleton: '
                             '_finish in src/wavebit/fdtd/_prepare/plan.py has changed what it '
                             'hashes, and this rebuild must follow it')
    return encoded


def _digest(encoded):
    return hashlib.sha256(json.dumps(encoded, sort_keys=False).encode()).hexdigest()


def _walk(node, visit):
    """Call ``visit`` on every float and complex leaf of an encoding, in order.

    The grammar is ``_encode``'s: ``[tag, payload]``, where ``'tuple'`` holds nodes, ``'dict'``
    and a dataclass's type name hold ``[name, node]`` pairs, and the rest are leaves.
    """
    tag = node[0]
    if tag in ('float', 'complex'):
        visit(node)
    elif tag == 'tuple':
        for child in node[1]:
            _walk(child, visit)
    elif tag not in _SCALARS:                       # 'dict' and every dataclass
        for _, child in node[1]:
            _walk(child, visit)


def float_leaves(encoded):
    """Every float leaf as its hex string and every complex leaf as its pair, in order."""
    out = []
    _walk(encoded, lambda node: out.append(node[1]))
    return out


def _value(leaf):
    if isinstance(leaf, str):
        return float.fromhex(leaf)
    return complex(float.fromhex(leaf[0]), float.fromhex(leaf[1]))


def rebuilt_skeleton(encoded, frozen, rtol):
    """``(skeleton, worst)``: the hash with each live float inside ``rtol`` replaced by its frozen one.

    ``frozen`` is :func:`float_leaves` of the frozen encoding. A leaf is inside when it is the
    frozen one exactly, or when both are finite, the frozen one is not zero and
    ``|live - frozen| <= rtol * |frozen|`` (``|.|`` of the whole complex number for a complex
    leaf). A leaf outside is left as it is, so the hash cannot come out as the frozen skeleton.
    ``worst`` names the largest gap found, or the count mismatch if the structure moved.
    """
    encoded = json.loads(json.dumps(encoded))           # a copy to write into
    leaves = []
    _walk(encoded, leaves.append)
    if len(leaves) != len(frozen):
        return _digest(encoded), (f'{len(leaves)} float leaves against {len(frozen)} frozen: '
                                  f'the structure moved')
    worst = (0.0, None)
    for node, was in zip(leaves, frozen):
        if node[1] == was:
            continue
        if isinstance(node[1], str) != isinstance(was, str):    # a float where a complex was
            return _digest(encoded), f'a {node[0]} leaf where the frozen one is not: the structure moved'
        live, then = _value(node[1]), _value(was)
        finite = all(math.isfinite(x) for x in (complex(live).real, complex(live).imag,
                                                 complex(then).real, complex(then).imag))
        gap = abs(live - then) / abs(then) if finite and then != 0 else math.inf
        if gap > worst[0]:
            worst = (gap, f'{node[0]} {live!r} against the frozen {then!r} ({gap:.1e} relative)')
        if gap <= rtol:
            node[1] = was
    return _digest(encoded), worst[1]
