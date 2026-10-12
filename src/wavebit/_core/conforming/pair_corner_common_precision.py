"""Complete real common coordinates before the contact weak projection.

The full polynomial and corner coordinate transform is composed in
compensated arithmetic. Exact rational prefactors retain all real radial
nodes; the unit-simplex relation is maintained before evaluating them.
"""
from collections import OrderedDict
from fractions import Fraction as F
from math import factorial
import numpy as np
from .pair_corner_precision import D
from .pair_contact_quadrature import coordinates
from . import pair_oscillatory_fields as fields_api
from .pair_oscillatory_polynomials import add,scale,multiply,power,linear,bernstein,pair,ONE
from .triangle_polynomials import multi_indices

def _prepare(fields,tri):
    assert not fields.oscillatory_radial
    data=fields.obj.data;degree=fields.obj.degree;angular_degree=2+fields.angular_order
    numbers=data['contact_numerators'][data['mapping'][tri[:3]]]
    hats=[[F(int(n),int(data['weight_denominator'])) for n in row] for row in numbers]
    local=fields.interpolant[data['pmap'][data['periodic'][tri]]]
    rows=[(*row,True) for row in fields.newton];result=[]
    for corner,ids,transform in zip(fields.controls,fields.groups,fields.transforms):
        if not any(row[corner] for row in hats):continue
        s=[1-row[corner] for row in hats]
        angular=[linear([row[n]/sv if sv else F(0) for row,sv in zip(hats,s)]) for n in range(10)]
        raw=[linear([row[n] for row in hats]) for n in range(10)]
        radial=[[[F(0) for _ in ids] for _ in multi_indices(angular_degree)] for _ in fields.exponents]
        ordinary=[[F(0) for _ in ids] for _ in multi_indices(degree)]
        for column,index in enumerate(ids):
            _,c,ns,cutoff,j,node_values,normalized=rows[index]
            assert c==corner
            p=power(ONE,0);q=power(ONE,0)
            for n in ns:p=multiply(p,angular[n]);q=multiply(q,raw[n])
            if j:
                p=multiply(p,power(add(angular[ns[0]],scale(angular[ns[1]],-1)),j))
                q=multiply(q,power(add(raw[ns[0]],scale(raw[ns[1]],-1)),j))
            k=len(ns)+j;r=max(0,cutoff+1);nodes=tuple(F(str(a)) for a in node_values)
            factor=F((-1)**(len(nodes)-1)*factorial(r+len(nodes)-1),factorial(r))
            vector=bernstein(p,angular_degree);polynomial={}
            for node,value in zip(nodes,node_values):
                lead=F(1);taylor=[]
                for n in range(r):
                    taylor.append(lead);lead*=F(k+n)-node;lead/=n+1
                weight=F(1)
                if normalized:
                    for other in nodes:
                        if other!=node:weight/=node-other
                    weight*=factor/lead
                radial_index=fields.exponents.index(value)
                for a,coefficient in enumerate(vector):radial[radial_index][a][column]+=coefficient*weight
                for n,t in enumerate(taylor):
                    polynomial=add(polynomial,scale(multiply(q,power(raw[corner],n+1)),weight*t))
            for a,value in enumerate(bernstein(polynomial,degree)):ordinary[a][column]=value
        keep=np.array([all(s[j] or not alpha[j] for j in range(3)) for alpha in multi_indices(angular_degree)])
        mapping=D(transform)
        radial=[(pair(np.asarray(row,dtype=object)[keep])*fields.column_scale[ids])@mapping for row in radial]
        ordinary=((pair(ordinary)+D(local[:,ids]))*fields.column_scale[ids])@mapping
        result.append((corner,ids,pair(s),keep,radial,ordinary))
    return result


def _power(x,n):
    result=D(np.ones_like(x.hi))
    while n:
        if n&1:result=result*x
        n//=2
        if n:x=x*x
    return result

def _root(value,n):
    x=D(value.hi**(1/n))
    for _ in range(2):
        r=_power(x,n)
        x=x+((value-r)/r)*x/n
    return x


def _kernels(s,exponents):
    value=s.parts[0];x=_root(value,20);out=[]
    for exponent in exponents:
        alpha=pair(F(str(exponent)))
        radial=_power(x,int(F(str(exponent))*20))
        out.append(fields_api.compose(s,radial,alpha*radial/value))
    return out


class CommonCoordinates:

    def __init__(self,fields,mapping):
        if fields.oscillatory_radial or fields.root_order!=20:
            raise ValueError('complete real contact coordinates require real root-20 fields')
        self.fields=fields;self.mapping=D(mapping);self.cache=OrderedDict();self.bytes=0

    def prepared(self,tri):
        key=tuple(tri)
        if key in self.cache:
            result,size=self.cache.pop(key);self.cache[key]=(result,size);return result
        data=self.fields.obj.data;nold=data['Cfull'].shape[1]-1
        ordinary=D(data['Cfull'][data['periodic'][tri],1:])@self.mapping[:nold]
        out=[]
        for corner,ids,complement,keep,radials,polynomial in _prepare(self.fields,tri):
            change=self.mapping[nold+ids]
            ordinary=ordinary-polynomial@change
            out.append((complement,keep,[radial@change for radial in radials]))
        result=ordinary,out;size=8*len(key)+ordinary.hi.nbytes*2+sum(s.hi.nbytes*2+k.nbytes+sum(r.hi.nbytes*2 for r in rs) for s,k,rs in out)
        if size<=64*1024*1024:
            while self.cache and self.bytes+size>64*1024*1024:
                _,(_,previous)=self.cache.popitem(last=False);self.bytes-=previous
            self.cache[key]=(result,size);self.bytes+=size
        return result

    def sample(self,tri,bary,inverse,dinverse):
        ordinary,rows=self.prepared(tri);ref=D(np.array([[-1.,-1.],[1.,0.],[0.,1.]]))
        barycentric=coordinates(bary)
        points=[fields_api.Jet((barycentric[i],D(0.),ref[i,0],ref[i,1],D(0.),D(0.)),True) for i in range(3)]
        potential=fields_api.mapped(fields_api.bernstein(points,self.fields.obj.degree),ordinary,True,True)
        for complement,keep,radials in rows:
            s=sum(points[i]*complement[i] for i in range(3));h=sum(points[i]*(1-complement[i]) for i in range(3))
            if np.any(s.parts[0].hi+s.parts[0].lo<=0):
                raise ValueError('corner requires a positive complement')
            angle=[fields_api.quotient(points[i]*complement[i],s) for i in range(3)]
            angular=fields_api.bernstein(angle,2+self.fields.angular_order,keep)
            kernels=_kernels(s,self.fields.exponents)
            for radial,matrix in zip(kernels,radials):potential=potential+(h*radial)[:,None]*fields_api.mapped(angular,matrix,True)
        refg=potential.parts[2:4]
        return [sum(refg[a]*D(matrix[a,b]) for a in range(2)) for matrix in (inverse,dinverse) for b in range(2)]
