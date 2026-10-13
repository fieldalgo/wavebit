"""The kernel: one body of code for three backends, written against the runtime primitives.

Nothing here imports numpy, jax or torch for arithmetic. Every array operation goes through
the runtime object ``rt`` (``src/wavebit/fdtd/_runtime.py``), so the curl, the step, the masks and
the observers are written once and the backend is a parameter. That is the condition under
which the three backends agree to round-off: every arithmetic is done in the same order
because it is the same code.

Layout, repeated because every module here depends on it: arrays are ``[component, z, y,
x]``; a single component ``F[c]`` is ``(nz, ny, nx)`` with spatial axes ``z = 0``, ``y = 1``,
``x = 2``. ``AX`` maps an axis name to that position.
"""

#: Position of each spatial axis in a single-component array ``(nz, ny, nx)``.
AX = {'x': 2, 'y': 1, 'z': 0}

#: Axis index (0 = x, 1 = y, 2 = z) to name.
AXIS_NAMES = ('x', 'y', 'z')
