"""Real full-space corner coordinates formed before physical weak assembly.

The symmetric mesh's radial fields can be nearly dependent in raw
coordinates. A common Pp interpolation subtraction and a complete per-corner
QR retain the same potential space while avoiding cancellation in the weak
matrix itself. Every coordinate survives; all original physical checks apply.
This explicit NumPy geometry step depends on no material or physical length.
"""
import numpy as np
from wavebit.rcwa._engine.conforming.triangle_polynomials import multi_indices
from wavebit.rcwa._engine.conforming.triangle_polynomials import values_and_tangent
from wavebit.rcwa._engine.conforming.triangle_polynomials import gradient_coefficients


def prepare(self):
    data=self.obj.data;degree=self.obj.degree
    bary=np.asarray(multi_indices(degree))/degree
    matrix=values_and_tangent(bary,np.zeros_like(bary),degree)[0]
    count=data['Cfull'].shape[1]
    coeff=np.full((count,self.count),np.nan)
    for tri in data['triangles']:
        hats=self.hats[tri[:3]]
        singular=np.any((bary@hats)[:,self.controls]==1,axis=1)
        values=np.zeros((len(bary),self.count))
        # Every complete corner potential is zero at its singular vertex;
        # its infinite gradient is unnecessary for nodal interpolation.
        values[~singular]=self._sample(tri,bary[~singular],np.eye(2),np.zeros((2,2)))[0]
        local=np.linalg.solve(matrix,values)
        ids=data['pmap'][data['periodic'][tri]]
        for index,row in zip(ids,local):
            if np.isnan(coeff[index,0]):coeff[index]=row
            elif not np.allclose(coeff[index],row,atol=2e-13,rtol=0):
                raise ArithmeticError('corner interpolation does not share the exact common topology')
    if not np.all(np.isfinite(coeff)) or np.any(coeff[0]):
        raise ArithmeticError('corner interpolation must preserve the finite common gauge')
    # Reuse one coefficient for each common node, so the subtraction is a
    # globally continuous Pp function, including collapsing triangles.
    self.interpolant=coeff
    self.reference=gradient_coefficients(degree)
    self.column_scale=np.ones(self.count)
    norm=np.zeros(self.count)
    for tri in data['triangles']:
        for b,w in self.quadrature(tri,12):
            g=self.gradients(tri,b,np.eye(2),np.zeros((2,2)))[0]
            norm+=np.einsum('q,qai,qai->i',w,g,g)
    if not np.all(np.isfinite(norm)) or not np.all(norm>0):
        raise ArithmeticError('corner field requires a finite positive reference norm')
    # This preliminary norm only scales coordinates. It is not used as a
    # physical energy integral, error criterion or field-selection threshold.
    self.column_scale=1/np.sqrt(norm)
    self.groups=[np.array([i for i,row in enumerate(self._definitions) if row[0]==corner])
                 for corner in self.controls]
    upper=[np.zeros((0,len(group))) for group in self.groups]
    for tri in data['triangles']:
        present=np.any(self.hats[tri[:3]]!=0,axis=0)
        for bary,weight in self.quadrature(tri,max(24,self.root_order*max(degree,4))):
            gradient=self.gradients(tri,bary,np.eye(2),np.zeros((2,2)))[0]
            for index,(corner,indices) in enumerate(zip(self.controls,self.groups)):
                if not present[corner] or not len(indices):continue
                rows=(np.sqrt(weight)[:,None,None]*gradient[:,:,indices]).reshape(-1,len(indices))
                upper[index]=np.linalg.qr(np.vstack([upper[index],rows]),mode='r')
    self.transforms=[np.linalg.solve(matrix,np.eye(len(matrix))) for matrix in upper]


def change_coordinates(self,tri,bary,inverse,dinverse,bary_tangent,result):
    v,g,dg=result
    gradient,change=change_gradients(self,tri,bary,inverse,dinverse,bary_tangent,(g,dg))
    data=self.obj.data;local=self.interpolant[data['pmap'][data['periodic'][tri]]]
    db=np.zeros_like(bary) if bary_tangent is None else bary_tangent
    polynomial=values_and_tangent(bary,db,self.obj.degree)[0]
    v=(v-polynomial@local)*self.column_scale
    if hasattr(self,'transforms'):
        present=np.any(self.hats[tri[:3]]!=0,axis=0)
        for corner,indices,transform in zip(self.controls,self.groups,self.transforms):
            if present[corner] and len(indices):v[...,indices]=v[...,indices]@transform
    return v,gradient,change


def change_gradients(self,tri,bary,inverse,dinverse,bary_tangent,result):
    """Keep all gradient operations; cache only fixed interpolation coefficients."""
    g,dg=result
    cache=getattr(self,'_gradient_coefficient_cache',None)
    if cache is None:self._gradient_coefficient_cache=cache={}
    key=tuple(tri);reference=cache.get(key)
    if reference is None:
        data=self.obj.data
        local=self.interpolant[data['pmap'][data['periodic'][tri]]]
        reference=np.einsum('vna,ni->vai',self.reference,local)
        # Bound memory on refined meshes. Recomputing an evicted coefficient
        # uses exactly the same arithmetic and never removes a field.
        if len(cache)>=128:del cache[next(iter(cache))]
        cache[key]=reference
    db=np.zeros_like(bary) if bary_tangent is None else bary_tangent
    p,dp=values_and_tangent(bary,db,self.obj.degree-1)
    gradient=np.einsum('qv,vai->qai',p,reference)
    change=np.einsum('qv,vai->qai',dp,reference)
    g-=np.einsum('ab,qai->qbi',inverse,gradient)
    dg-=np.einsum('ab,qai->qbi',dinverse,gradient)+np.einsum('ab,qai->qbi',inverse,change)
    result=g*self.column_scale,dg*self.column_scale
    if hasattr(self,'transforms'):
        present=np.any(self.hats[tri[:3]]!=0,axis=0)
        for array in result:
            for corner,indices,transform in zip(self.controls,self.groups,self.transforms):
                if present[corner] and len(indices):array[...,indices]=array[...,indices]@transform
    return result
