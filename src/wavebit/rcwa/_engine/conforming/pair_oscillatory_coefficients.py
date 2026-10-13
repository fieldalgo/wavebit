"""Exact rational weights for the fixed real exponent-column coordinates."""
from fractions import Fraction as F
from functools import lru_cache

def add(a,b):return a[0]+b[0],a[1]+b[1]
def neg(a):return -a[0],-a[1]
def sub(a,b):return add(a,neg(b))
def mul(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def div(a,b):
    den=b[0]*b[0]+b[1]*b[1]
    return (a[0]*b[0]+a[1]*b[1])/den,(a[1]*b[0]-a[0]*b[1])/den
ONE=(F(1),F(0));ZERO=(F(0),F(0))
NODES=((F(3,5),F(0)),(F(4,5),F(0)),(F(19,20),F(1,2)),(F(19,20),F(-1,2)))

@lru_cache(None)
def coefficients(k,r,count,terms=96):
    nodes=NODES[:count];weights=[];lead=[];taylor=[]
    for i,a in enumerate(nodes):
        w=ONE
        for j,b in enumerate(nodes):
            if i!=j:w=div(w,sub(a,b))
        weights.append(w)
        c=ONE;row=[]
        for j in range(r+1):
            if j:c=div(mul(c,sub((F(k+j-1),F(0)),a)),(F(j),F(0)))
            row.append(c)
        lead.append(c);taylor.append(row[:-1])
    scale=F((-1)**(count-1))
    for j in range(count-1):scale*=r+1+j
    current=[ONE]*count;out=[]
    for n in range(terms):
        value=ZERO
        for w,c in zip(weights,current):value=add(value,mul(w,c))
        value=mul((scale,F(0)),value)
        if count!=3:assert value[1]==0
        if n<count-1:assert value==ZERO
        if n==count-1:assert value[0]==1
        out.append(value[0])
        current=[div(mul(c,sub((F(r+k+n),F(0)),a)),(F(r+1+n),F(0))) for c,a in zip(current,nodes)]
    return tuple(weights),tuple(lead),tuple(tuple(t) for t in taylor),scale,tuple(out)
