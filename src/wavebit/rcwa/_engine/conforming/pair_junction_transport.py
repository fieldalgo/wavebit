"""Reference-domain transport derivatives with explicit moving-edge flux.

Interior nodes are fixed in reference coordinates for differentiation.
Only vertical profile edges can carry an integrand jump inside a triangle:
C1 taper caps match both the potential and its gradient across moving y edges.
"""
from fractions import Fraction as F
from itertools import chain
import numpy as np
from numpy.polynomial.legendre import leggauss
from wavebit.rcwa._engine.conforming.pair_junction_partition import samples as regular_samples
from wavebit.rcwa._engine.conforming.pair_junction_partition import partitions
from wavebit.rcwa._engine.conforming.pair_junction_partition import scalar
from wavebit.rcwa._engine.conforming.pair_junction_partition import dot
from wavebit.rcwa._engine.conforming.pair_junction_partition import point
from wavebit.rcwa._engine.conforming.pair_junction_thin_partition import samples as thin_samples


def samples(obj,base,settings,chart,order,split=True):
    power=base[-1]['corner_fields'].root_order
    sf,dsf=scalar(chart[1]),scalar(base[-1]['ds'])
    for tri,local,p,dp,poly,bb,dbb,old_measure,old_linear in chain(
            regular_samples(obj,base,settings,chart,order,split),
            thin_samples(obj,base,settings,chart,order,split)):
        determinant=scalar(local[9])+sf*scalar(local[10])
        ddet=dsf*scalar(local[10])
        def measure(normalized):
            w=old_measure(normalized)[0]
            ratio=ddet/determinant-(dsf/sf if normalized else 0)
            return w,w*float(ratio)
        def linear(values,changes):
            return old_linear(values,[F(0)]*3)[0],old_linear(changes,[F(0)]*3)[0]
        yield tri,local,p,dp,poly,bb,np.zeros_like(bb),measure,linear
    if not split:return
    t,w=leggauss(order);t=(t+1)/2;w=w/2
    # Matching physical cuts share the identical canonical line nodes.
    for tri,local,p,dp,poly in partitions(obj,base,settings,chart,True):
        determinant=scalar(local[9])+sf*scalar(local[10])
        for (a,da),(b,db) in zip(poly,poly[1:]+poly[:1]):
            edge=tuple(v-u for u,v in zip(a,b))
            va=da[1]*edge[2]-da[2]*edge[1]
            vb=db[1]*edge[2]-db[2]*edge[1]
            if va==vb==0:continue
            pa,pb=point(a,p),point(b,p)
            if pa[0]!=pb[0]:
                if pa[1]!=pb[1]:raise ValueError('unexpected moving cut orientation')
                continue
            if a>b:a,b,va,vb=b,a,vb,va
            for upper in (False,True):
                q=.5*t**power
                xi=1-q if upper else q
                complement=q if upper else 1-q
                bb=complement[:,None]*np.asarray(a,float)+xi[:,None]*np.asarray(b,float)
                def measure(normalized):
                    scale=determinant/(sf if normalized else 1)
                    flux=complement*float(scale*va)+xi*float(scale*vb)
                    return np.zeros_like(w),w*(power/2)*t**(power-1)*flux
                def linear(values,changes):
                    return (complement*float(dot(a,values))+xi*float(dot(b,values)),
                            complement*float(dot(a,changes))+xi*float(dot(b,changes)))
                yield tri,local,p,dp,poly,bb,np.zeros_like(bb),measure,linear
