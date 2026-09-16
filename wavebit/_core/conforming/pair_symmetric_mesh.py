"""Reflection-invariant pair trial mesh with the original exact contact quotient.

The explicit NumPy corner family selects this mesh before solving. No response
entries or derivatives are projected or averaged to impose symmetry.
"""
import numpy as np
from .cartesian_pair import _UNIT
from .cartesian_pair_opening import CartesianPairOpeningPlan, _collapse_topology
from .cartesian_pair_polynomial_galerkin import (
    CartesianPairPolynomialGalerkinGapCell, _polynomial_topology,
)


def refined(contact,shift,refinement):
    """Eight-triangle rectangles and reflected annulus fans, uniformly refined."""
    if shift!=0:raise ValueError('symmetric pair mesh requires shift zero')
    nx=3 if contact else 4
    nodes=list(np.eye(2*nx+4));lookup={tuple(row):i for i,row in enumerate(nodes)}
    triangles=[];tags=[]
    def node(row):
        key=tuple(row)
        if key not in lookup:lookup[key]=len(nodes);nodes.append(row)
        return lookup[key]
    def midpoint(a,b):return node((nodes[a]+nodes[b])/2)
    def quad(a,b,c,d,tag):
        center=node((nodes[a]+nodes[b]+nodes[c]+nodes[d])/4)
        # Edge midpoints ensure every thin child has a repeated endpoint pair.
        boundary=[a,midpoint(a,b),b,midpoint(b,c),c,midpoint(c,d),d,midpoint(d,a)]
        triangles.extend((center,u,v) for u,v in zip(boundary,boundary[1:]+boundary[:1]))
        tags.extend([tag]*8)
    bottom=[4]
    for i,tag in enumerate((1,2) if contact else (1,0,2)):
        a,b,c,d=4+i,5+i,4+2*nx-2-i,4+2*nx-1-i
        quad(a,b,c,d,tag)
        bottom.extend((midpoint(a,b),b))
    left_bottom,right_bottom=4,3+nx
    left_top,right_top=3+2*nx,4+nx
    quad(0,left_bottom,left_top,3,0)
    quad(right_bottom,1,2,right_top,0)
    origin=midpoint(0,1)
    boundary=[1,midpoint(1,right_bottom),*bottom[::-1],midpoint(left_bottom,0),0]
    lower=[(origin,a,b) for a,b in zip(boundary,boundary[1:])]
    triangles.extend(lower);tags.extend([0]*len(lower))
    mirror=np.r_[[3,2,1,0],np.arange(4,4+2*nx)[::-1]]
    reflected=[]
    for a,b,c in lower:
        reflected.append(tuple(node(nodes[i][mirror]) for i in (a,c,b)))
    triangles.extend(reflected);tags.extend([0]*len(reflected))
    weights,cells,tags=np.asarray(nodes),np.asarray(triangles),np.asarray(tags)
    for _ in range(refinement):
        nodes=list(weights);mids={};children=[]
        def mid(a,b):
            key=tuple(sorted((int(a),int(b))))
            if key not in mids:mids[key]=len(nodes);nodes.append((weights[a]+weights[b])/2)
            return mids[key]
        for a,b,c in cells:
            ab,bc,ca=mid(a,b),mid(b,c),mid(c,a)
            children.extend(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)))
        weights,cells,tags=np.asarray(nodes),np.asarray(children),np.repeat(tags,4)
    lookup={};periodic=[]
    for i,row in enumerate(weights):
        key=('interior',i) if np.any(row[4:]) else ('boundary',*(row[:4]@_UNIT%1))
        if key not in lookup:lookup[key]=len(lookup)
        periodic.append(lookup[key])
    for array in (weights,cells,tags):array.flags.writeable=False
    return weights,cells,tags,np.asarray(periodic)

def topology(plan):
    """Use every original collapse, periodic, orientation and pole-graph guard."""
    opened=refined(False,plan.shift,plan.refinement)
    closed=refined(True,plan.shift,plan.refinement)
    return _collapse_topology(*opened,*closed,2**(plan.refinement+2))


class SymmetricPairPolynomialCell(CartesianPairPolynomialGalerkinGapCell):
    """Internal polynomial base whose geometry reconstruction retains its mesh."""
    _symmetric_mesh=True
    _maximum_degree=8

    @staticmethod
    def _normalization_length(basis):
        period=np.diag(basis)
        if np.any(basis!=np.diag(period)) or min(period)<=0:
            raise ValueError('positive diagonal period required')
        # A diagonal area is a direct product. Going through a logarithmic
        # determinant adds an avoidable unit-dependent rounding to the chart.
        return np.sqrt(period[0]*period[1])

    def __init__(self,plan,order=1,*,degree=3,coercivity_angle=0.):
        if not isinstance(plan,CartesianPairOpeningPlan):
            raise ValueError('an explicit CartesianPairOpeningPlan is required')
        if plan.shift!=0:
            raise ValueError('symmetric pair mesh requires shift zero')
        super().__init__(plan,order,degree=degree,coercivity_angle=coercivity_angle)
        self._data=_polynomial_topology(plan,self.degree,coarse=topology(plan))
