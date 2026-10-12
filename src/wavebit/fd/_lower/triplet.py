"""The triplet baseline's local maps, one per oriented triplet of E edges at a node.

Werner, Bauer and Cary, *A more accurate, stable, FDTD algorithm for electromagnetics in
anisotropic dielectrics*, J. Comput. Phys. **255**, 436 (2013), sections 4 and 7. At each node,
an orientation picks one edge per E component -- on the side its sign names -- and the local
map ``Gamma Pi^-1`` is built from the fraction of each of those edges and of each dual face the
body holds, and from the interface normal there. The symmetric parts of those maps, averaged
over the orientations, are the baseline :func:`~wavebit.fd._lower.stencil.fit` corrects;
:func:`~wavebit.fd._lower.stencil.triplet_baseline` assembles them. A sphere has the eight
triplets of :data:`TRIPLETS`; the invariant-x plane of the TE cylinder has the four pairs of
:data:`PAIRS`, where the sphere's central plane is the circle.

**Moved, not rewritten (2026-09-26).** These are the bodies of
``studies/fdtd_subpixel_triplet_sphere.py``'s ``ORIENTATIONS``, ``local_map`` and
``sphere_maps`` and of ``studies/fdtd_subpixel_triplet_cylinder.py``'s ``ORIENTATIONS``,
``local_map``, ``circle_fraction`` and ``circle_maps``, which the public stencil's default
baseline used to import from a repository checkout at plan time. The two studies import them
from here now, under their old names. Apart from the renames (``ORIENTATIONS`` to
:data:`TRIPLETS` and :data:`PAIRS`, ``local_map`` to :func:`triplet_map` and :func:`pair_map`)
and a one-line docstring each on :func:`sphere_maps` and :func:`circle_maps`, the one edit
inside a body is that ``sphere_maps`` calls :meth:`~wavebit.fd._shapes.Sphere.exact_edge_fill`
and :meth:`~wavebit.fd._shapes.Sphere.exact_face_fill` itself, which is all the study's
``edge_fraction`` and ``face_fraction`` wrappers did. What was measured to say no number moved
is in ``docs/fd-symmetric-stencil.md``, "The seam, closed".

**NumPy on the host for the public route; a namespace for the research drivers.** The stencil
calls these once per plan with NumPy float64 arrays. :func:`triplet_map`, :func:`sphere_maps`
and :func:`circle_fraction` keep the ``xp`` argument the research drivers use to step and
differentiate on JAX and Torch (``tests/test_fd_triplet_sphere.py`` and
``tests/test_fd_triplet_research.py`` run them there); :func:`pair_map` and
:func:`circle_maps` are NumPy only, as they were.
"""
import itertools

import numpy as np

from .._shapes import Sphere

__all__ = ['PAIRS', 'TRIPLETS', 'circle_fraction', 'circle_maps', 'pair_map', 'sphere_maps',
           'triplet_map']

#: The eight oriented triplets of a sphere's nodes: one sign per component, x, y, z.
TRIPLETS = tuple(itertools.product((1, -1), repeat=3))

#: Four oriented pairs replace the eight 3D triplets when x is invariant: signs for y, z.
PAIRS = ((1,1), (1,-1), (-1,1), (-1,-1))


def triplet_map(normal, lengths, areas, eps, xp=np):
    """Gamma Pi^-1 via the rank-one inverse of Pi, then its symmetric part.

    Pi = diag(d) - diag((eps-1)*areas) n n^T, d=1+(eps-1)*areas.
    Sherman-Morrison gives K_ij=delta_ij/d_i +
    (beta_i+(1/eps-1)*length_i)*n_i*n_j/(denom*d_j),
    beta=(d-1)/d and denom=1-sum(beta*n*n). All divisions are
    regular for a unit normal, positive eps and fractions in [0,1].
    These preconditions and energy are checked once at preparation.
    """
    d = 1+(eps-1)*areas
    beta = (eps-1)*areas/d
    denom = 1-xp.sum(beta*normal*normal, axis=-1)
    coefficient = beta+(1/eps-1)*lengths
    rows = []
    for i in range(3):
        rows.append(xp.stack(tuple((1/d[..., i] if i == j else xp.zeros_like(denom))
            +coefficient[..., i]*normal[..., i]*normal[..., j]/(denom*d[..., j])
            for j in range(3)), axis=-1))
    accurate = xp.stack(rows, axis=-2)
    symmetric = xp.stack(tuple(xp.stack(tuple((accurate[..., i, j]+accurate[..., j, i])/2
        for j in range(3)), axis=-1) for i in range(3)), axis=-2)
    return accurate, symmetric


def sphere_maps(points, dx, center, radius, eps, xp=np, geometry=False):
    """Each node's eight triplet maps, accurate and symmetric; with ``geometry``, the fractions.

    The edge and face fractions are the ball's exact ones, taken per call exactly as the
    study's wrappers took them (:class:`~wavebit.fd._shapes.Sphere` is built afresh each time,
    so a traced ``radius`` enters the graph the way it always has).
    """
    relative = xp.stack(tuple(points[..., i]-center[i] for i in range(3)), axis=-1)
    norm2 = xp.sum(relative*relative, axis=-1)
    norm = xp.sqrt(xp.where(norm2>0, norm2, xp.ones_like(norm2)))
    normal = xp.stack(tuple(xp.where(norm2>0, relative[..., i]/norm,
        xp.ones_like(norm) if i == 0 else xp.zeros_like(norm)) for i in range(3)), axis=-1)
    accurate, symmetric, edges, faces = [], [], [], []
    for orientation in TRIPLETS:
        ell, area = [], []
        for i, sign in enumerate(orientation):
            p = xp.stack(tuple(relative[..., j]+(sign*dx/2 if i == j else 0.)
                for j in range(3)), axis=-1)
            ell.append(Sphere(radius).exact_edge_fill(p, dx, i))
            area.append(Sphere(radius).exact_face_fill(p, dx, i))
        ell, area = xp.stack(ell, axis=-1), xp.stack(area, axis=-1)
        a, s = triplet_map(normal, ell, area, eps, xp)
        accurate.append(a); symmetric.append(s)
        edges.append(ell); faces.append(area)
    result = xp.stack(accurate), xp.stack(symmetric)
    return result+(xp.stack(edges), xp.stack(faces)) if geometry else result


def pair_map(normal, lengths, areas, eps):
    """Gamma Pi^-1 from E edge lengths and D dual-face areas, eps2=1.

    Gamma_p = I + (1/eps_p-1) nn^T and Pi_p = eps_p Gamma_p.
    Fractions left-multiply these matrices row by row (paper Eqs. 15-32).
    """
    nn = normal[..., :, None]*normal[..., None, :]
    eye = np.eye(2)
    gamma = eye + lengths[..., :, None]*(1/eps-1)*nn
    pi = eye + areas[..., :, None]*(eps-1)*(eye-nn)
    determinant = np.linalg.det(pi)
    if not np.all(np.isfinite(determinant) & (determinant > 0)):
        raise ValueError('TripletPiRefused: nonpositive or nonfinite determinant at preparation')
    accurate = np.linalg.solve(np.swapaxes(pi,-1,-2),np.swapaxes(gamma,-1,-2))
    accurate = np.swapaxes(accurate,-1,-2)
    return accurate, (accurate+np.swapaxes(accurate,-1,-2))/2


def circle_fraction(point, dx, axis, radius, xp=np):
    """Exact length of the circle's intersection with a centered segment.

    The invariant-x plane of a sphere **is** a circle of the same radius, so
    this is `src/wavebit/fd/_shapes/_bodies.py::Sphere.exact_edge_fill` on the
    (y,z) point lifted to a triple with x exactly zero: the lifted transverse
    sum is `0*0 + t*t`, which is the `t**2` this used to write. The public form
    holds the two guards this docstring used to carry -- the segment-coordinate
    clip that keeps a whole segment exactly one (the world-coordinate spelling
    gave 1+8.88179e-16 in the radius-one, dx=.1 audit) and the guarded root.

    `xp` is what the lift is built with, and it is why this function has one at
    all: without it a JAX or Torch point is silently pulled through `np.stack`
    into NumPy, which loses the trace and the gradient with it. The 2D research
    path calls this with `axis` 0 for y and 1 for z, so the lifted component
    index is `axis+1`.
    """
    zero = xp.zeros_like(point[...,0])
    lifted = xp.stack((zero,point[...,0],point[...,1]),axis=-1)
    return Sphere(radius).exact_edge_fill(lifted,dx,axis+1)


def circle_maps(points, dx, center, radius, eps):
    """Each node's four pair maps in the invariant-x plane, accurate and symmetric."""
    relative = points-np.asarray(center)
    norm = np.linalg.norm(relative,axis=-1)
    normal = relative/np.where(norm>0,norm,1.)[...,None]
    # The center is uniform material; either normal gives the same inverse.
    normal = np.where((norm==0)[...,None],np.array((1.,0.)),normal)
    accurate,symmetric = [],[]
    for sy,sz in PAIRS:
        ell,area = [],[]
        for i,sign in enumerate((sy,sz)):
            p = relative.copy(); p[...,i] += sign*dx/2
            ell.append(circle_fraction(p,dx,i,radius))
            area.append(circle_fraction(p,dx,1-i,radius))
        a,s = pair_map(normal,np.stack(ell,axis=-1),np.stack(area,axis=-1),eps)
        accurate.append(a); symmetric.append(s)
    return np.stack(accurate),np.stack(symmetric)
