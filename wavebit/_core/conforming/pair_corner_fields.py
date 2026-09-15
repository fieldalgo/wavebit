"""Material-independent real corner and junction potentials for the pair cell.

NumPy-only geometry evaluation, like the explicit polynomial pair family.
The default exponents 3/5 and 4/5 admit the common r=t**5 pullback.
Adding 11/20 or 19/20 uses exact normalized powers and r=t**20.
"""
import numpy as np
from fractions import Fraction
from functools import lru_cache
from math import comb
from numbers import Integral
from .pair_radial_fields import prepare as prepare_radial, sample as sample_radial

@lru_cache(None)
def remainder_polynomial(m,degree):
    numerator=[Fraction(0) for _ in range(m+5*degree+1)]
    numerator[0]=Fraction(1);coefficient=Fraction(1)
    for j in range(degree+1):
        if j:coefficient*=Fraction(5*(j-1)+m,5*j)
        for k in range(j+1):
            numerator[m+5*k]-=coefficient*comb(j,k)*(-1)**k
    for _ in range(degree+1):
        quotient=[];running=Fraction(0)
        for item in numerator[:-1]:
            running+=item;quotient.append(running)
        if numerator[-1]+running != 0:
            raise ArithmeticError('corner remainder has a nonzero exact division residual')
        numerator=quotient
    return np.asarray(numerator,float)

from numpy.polynomial.legendre import leggauss


class PairCornerFields:
    """Real corner powers modulo the existing polynomial space.

    Closed control hats define h*n*(1-h)**(alpha-1) and
    h*n*m*(1-h)**(alpha-2). The Taylor polynomial already represented by
    the cubic or quartic mesh is removed by an invertible real change of
    coordinates. Exact fifth-root factorization preserves small remainders.
    Optional angular powers multiply n*m by (n-m)**j and decrease the radial
    exponent by j, retaining the same leading radial power. Their polynomial
    overlap is removed only when it lies in the existing polynomial space.
    All fields vanish on the periodic boundary. Outer-corner fields also
    vanish on collapsing triangles; optional junction fields remain finite
    tangential potentials there.
    """
    def __init__(self, obj, exponents=(.6,.8), *, junctions=False, angular_order=0):
        if obj.degree not in (3, 4):
            raise ValueError('corner potentials require degree 3 or 4')
        exponents=tuple(exponents)
        if (any(isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating))
                or value not in (.55,.6,.8,.95) for value in exponents)
                or len(set(exponents)) != len(exponents)):
            raise ValueError('corner exponents must be distinct members of (0.55, 0.6, 0.8, 0.95)')
        if not isinstance(junctions,(bool,np.bool_)):
            raise ValueError('junctions must be a boolean')
        if (isinstance(angular_order,(bool,np.bool_)) or not isinstance(angular_order,Integral)
                or not 0 <= angular_order <= 4):
            raise ValueError('corner angular order must be an integer from 0 through 4')
        self.root_order=20 if any(a not in (.6,.8) for a in exponents) else 5
        self.angular_order=int(angular_order)
        self.exponents=exponents
        self.junctions=bool(junctions)
        self.controls=(4,5,6,7,8,9) if self.junctions else (4,6,7,9)
        self.obj=obj
        self.hats=obj.data['contact_numerators'][obj.data['mapping']]/obj.data['weight_denominator']
        triangles=obj.data['triangles'][:,:3]
        regular=~obj.data['thin']
        self.powers=[]
        self.bubbles=[]
        for corner in self.controls:
            if corner not in (5,8) and np.any(self.hats[triangles[~regular],corner]):
                raise ValueError('outer corner fields must vanish on collapsing triangles')
            active=regular & np.any(self.hats[triangles,corner]>0,axis=1)
            for neighbor in range(10):
                if neighbor==corner:continue
                if np.any(active & np.any(self.hats[triangles,neighbor]>0,axis=1)):
                    self.powers.extend((corner,neighbor,a) for a in exponents)
            for first in range(10):
                for second in range(first+1,10):
                    if corner in (first,second):continue
                    if np.any(active & np.any(self.hats[triangles,first]>0,axis=1) & np.any(self.hats[triangles,second]>0,axis=1)):
                        self.bubbles.extend((corner,first,second,a) for a in exponents)
        self.angular=[(c,n,m,a,j) for c,n,m,a in self.bubbles
                      for j in range(1,self.angular_order+1)]
        if getattr(obj, '_symmetric_mesh', False):
            from .pair_corner_prefactors import remove_duplicates
            remove_duplicates(self)
        self.count=len(self.powers)+len(self.bubbles)+len(self.angular)
        self._definitions=[(c,(n,),a-1,self.obj.degree-2,0) for c,n,a in self.powers]
        self._definitions += [(c,(n,m),a-2,self.obj.degree-3,0) for c,n,m,a in self.bubbles]
        self._definitions += [(c,(n,m),a-2-j,self.obj.degree-3-j,j) for c,n,m,a,j in self.angular]
        self._support_cache={}
        if self.root_order==20:prepare_radial(self)
        if getattr(obj, '_symmetric_mesh', False) and self.count:
            from .pair_corner_coordinates import prepare
            prepare(self)

    def sample(self, tri, bary, inverse, dinverse, bary_tangent=None):
        result=self._sample(tri,bary,inverse,dinverse,bary_tangent)
        if hasattr(self,'interpolant'):
            from .pair_corner_coordinates import change_coordinates
            return change_coordinates(self,tri,bary,inverse,dinverse,bary_tangent,result)
        return result

    def gradients(self, tri, bary, inverse, dinverse, bary_tangent=None):
        """Return the identical field and variation without transforming unused values."""
        result=self._sample(tri,bary,inverse,dinverse,bary_tangent)[1:]
        if hasattr(self,'interpolant'):
            from .pair_corner_coordinates import change_gradients
            return change_gradients(self,tri,bary,inverse,dinverse,bary_tangent,result)
        return result

    def _sample(self, tri, bary, inverse, dinverse, bary_tangent=None):
        if self.root_order==20:
            return sample_radial(self,tri,bary,inverse,dinverse,bary_tangent)
        hat_values=self.hats[tri[:3]];hv=bary@hat_values
        dhv=np.zeros_like(hv) if bary_tangent is None else bary_tangent@hat_values
        reference=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
        gradients=(reference@inverse).T@hat_values
        dgradients=(reference@dinverse).T@hat_values
        value=np.zeros((len(bary),self.count));gradient=np.zeros((len(bary),2,self.count));change=np.zeros_like(gradient)
        cache={}
        def radial(corner,beta,degree):
            key=(corner,beta,degree)
            if key in cache:return cache[key]
            h=hv[:,corner];s=bary@hat_values[:,np.arange(10)!=corner].sum(axis=1)
            if np.any(s<=0):raise ValueError('corner gradient needs a positive complementary coordinate')
            if degree < 0:
                # No part of this angular polynomial is already in the
                # base space. The complete power needs no Taylor subtraction.
                result=(s**beta,-beta*s**(beta-1),beta*(beta-1)*s**(beta-2))
                cache[key]=result
                return result
            m=int(round(-5*beta));coefficients=remainder_polynomial(m,degree)
            u=s**.2
            complement=h/(1+u+u*u+u**3+u**4)
            p=np.polynomial.polynomial.polyval(u,coefficients)
            first=np.polynomial.polynomial.polyval(u,np.arange(1,len(coefficients))*coefficients[1:])
            second=(np.polynomial.polynomial.polyval(u,np.arange(2,len(coefficients))*np.arange(1,len(coefficients)-1)*coefficients[2:]) if len(coefficients)>2 else np.zeros_like(u))
            a=p/u**m
            da=first/u**m-m*p/u**(m+1)
            dda=second/u**m-2*m*first/u**(m+1)+m*(m+1)*p/u**(m+2)
            k=degree+1
            r=complement**k*a
            ru=complement**k*da-k*complement**(k-1)*a
            ruu=complement**k*dda-2*k*complement**(k-1)*da
            if k>1:ruu+=k*(k-1)*complement**(k-2)*a
            du=-1/(5*u**4);ddu=-4/(25*u**9)
            dr=ru*du;ddr=ruu*du**2+ru*ddu
            cache[key]=(r,dr,ddr);return r,dr,ddr
        # A hat absent at all three vertices is identically zero, including
        # its gradient and variation. Skip only these exact zero products;
        # keep every column in the common field space and returned arrays.
        support=tuple(np.flatnonzero(np.any(hat_values!=0,axis=0)))
        definitions=self._support_cache.get(support)
        if definitions is None:
            active=set(support)
            definitions=[(index,definition) for index,definition in enumerate(self._definitions)
                         if definition[0] in active and all(n in active for n in definition[1])]
            self._support_cache[support]=definitions
        for index,(corner,neighbors,beta,degree,angular) in definitions:
            h,dh=hv[:,corner],dhv[:,corner]
            r,dr,ddr=radial(corner,beta,degree)
            n=neighbors[0];product=hv[:,n];dproduct=dhv[:,n]
            gp=np.broadcast_to(gradients[:,n],(len(bary),2))
            dgp=np.broadcast_to(dgradients[:,n],(len(bary),2))
            if len(neighbors)==2:
                m=neighbors[1]
                gp=hv[:,n,None]*gradients[:,m]+hv[:,m,None]*gradients[:,n]
                dgp=(dhv[:,n,None]*gradients[:,m]+hv[:,n,None]*dgradients[:,m]
                     +dhv[:,m,None]*gradients[:,n]+hv[:,m,None]*dgradients[:,n])
                product=hv[:,n]*hv[:,m];dproduct=dhv[:,n]*hv[:,m]+hv[:,n]*dhv[:,m]
            if angular:
                z=hv[:,n]-hv[:,m];dz=dhv[:,n]-dhv[:,m]
                gz=gradients[:,n]-gradients[:,m];dgz=dgradients[:,n]-dgradients[:,m]
                factor=z**angular;dfactor=angular*z**(angular-1)*dz
                gf=(angular*z**(angular-1))[:,None]*gz
                dgf=(angular*z**(angular-1))[:,None]*dgz
                if angular>1:
                    dgf+=(angular*(angular-1)*z**(angular-2)*dz)[:,None]*gz
                dgp=(dgp*factor[:,None]+gp*dfactor[:,None]
                     +dproduct[:,None]*gf+product[:,None]*dgf)
                gp=gp*factor[:,None]+product[:,None]*gf
                dproduct=dproduct*factor+product*dfactor
                product=product*factor
            a=product*(r+h*dr);b=h*r
            da=dproduct*(r+h*dr)+product*(2*dr+h*ddr)*dh;db=(r+h*dr)*dh
            value[:,index]=h*product*r
            gradient[:,:,index]=a[:,None]*gradients[:,corner]+b[:,None]*gp
            change[:,:,index]=(da[:,None]*gradients[:,corner]+a[:,None]*dgradients[:,corner]
                                +db[:,None]*gp+b[:,None]*dgp)
        return value,gradient,change

    def quadrature(self, tri, order):
        t,w=leggauss(order);t=(t+1)/2;w=w/2
        refs=np.eye(3)
        seam_hats=self.hats[tri[:3]][:,[5,8]]
        singular_edges=np.flatnonzero(np.sum(seam_hats==1,axis=0)==2)
        if self.junctions and len(singular_edges):
            edge=np.flatnonzero(seam_hats[:,singular_edges[0]]==1)
            other=next(i for i in range(3) if i not in edge)
            # Both ends can be singular after the triangle collapses.
            # Retain the small complement instead of subtracting from one.
            small=.5*t**self.root_order;measure=(self.root_order/2)*t**(self.root_order-1)
            for radial,complement in ((small,1-small),(1-small,small)):
                bary=np.zeros((order,order,3))
                bary[:,:,other]=radial[:,None]
                bary[:,:,edge[0]]=complement[:,None]*(1-t)
                bary[:,:,edge[1]]=complement[:,None]*t
                yield bary.reshape(-1,3),(complement[:,None]*measure[:,None]*w[:,None]*w).ravel()
            return
        singular=np.flatnonzero(np.any(self.hats[tri[:3]][:,self.controls]==1,axis=1))
        if len(singular)==1:
            subs=[np.roll(refs,-int(singular[0]),axis=0)];power=self.root_order
        elif len(singular)==0:
            subs=[refs];power=1
        else:
            center=np.full(3,1/3);subs=[];power=self.root_order
            for i in range(3):
                j,k=(i+1)%3,(i+2)%3
                subs.extend([np.array([refs[i],(refs[i]+refs[j])/2,center]),np.array([refs[i],center,(refs[i]+refs[k])/2])])
        radial=t**power;dr=power*t**(power-1)
        uv=np.stack([np.broadcast_to(1-radial[:,None],(order,order)),radial[:,None]*(1-t),radial[:,None]*t],axis=-1).reshape(-1,3)
        weight=(radial[:,None]*dr[:,None]*w[:,None]*w).ravel()
        for sub in subs:yield uv@sub,weight*abs(np.linalg.det(sub))
