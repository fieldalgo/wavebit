"""Exact reference-coordinate sweep of collapsing junction triangles."""
from fractions import Fraction as F
import numpy as np
from numpy.polynomial.legendre import leggauss
from .pair_junction_partition import partitions,scalar,dot


def intersection(a,b,other,r,dr):
    v,dv=a;z,dz=b
    span=z[other]-v[other];dspan=dz[other]-dv[other]
    u=(r-v[other])/span;du=(dr-dv[other]-u*dspan)/span
    return (tuple((1-u)*x+u*y for x,y in zip(v,z)),
            tuple((1-u)*dx+u*dy+du*(y-x) for x,y,dx,dy in zip(v,z,dv,dz)))


def samples(obj,base,settings,chart,order,split=True):
    t,w=leggauss(order);t=(t+1)/2;w=w/2
    sf,dsf=scalar(chart[1]),scalar(base[-1]['ds'])
    fields=base[-1]['corner_fields']
    for tri,local,p,dp,poly in partitions(obj,base,settings,chart,split):
        if not local[11]:continue
        hats=fields.hats[tri[:3]][:,[5,8]]
        candidate=np.flatnonzero(np.sum(hats==1,axis=0)==2)
        if not len(candidate):
            # Refined thin triangles can lie away from either original edge.
            # Their closed hats remain constant across a collapsing vertex pair.
            pair=next(((i,j) for i in range(3) for j in range(i+1,3)
                       if np.array_equal(fields.hats[tri[i]],fields.hats[tri[j]])),None)
            if pair is None:raise ValueError('thin chart has no closed coincident pair')
            edge=np.array(pair)
        else:edge=np.flatnonzero(hats[:,candidate[0]]==1)
        other=next(i for i in range(3) if i not in edge);transverse=int(edge[1])
        knots={}
        for b,db in poly:
            r,dr=b[other],db[other]
            if r in knots and knots[r]!=dr:raise ValueError('coincident thin events have inconsistent derivatives')
            knots[r]=dr
        rs=sorted(knots)
        if rs[0]<F(1,2)<rs[-1] and F(1,2) not in knots:
            knots[F(1,2)]=F(0);rs=sorted(knots)
        # Fixed dyadic reference subdivisions keep smooth radial pieces
        # well resolved without moving every interior node as the gap closes.
        extra=[]
        for lo,hi in zip(rs,rs[1:]):
            upper=lo>=F(1,2)
            a,b=(1-hi,1-lo) if upper else (lo,hi)
            if a==0:continue
            cut=F(1,2)
            for _ in range(100):
                if cut<=a:break
                if cut<b:extra.append(1-cut if upper else cut)
                cut/=2
            else:raise ValueError('thin reference grading exhausted')
        for cut in extra:
            if cut not in knots:knots[cut]=F(0)
        rs=sorted(knots)
        for low,high in zip(rs,rs[1:]):
            if high==low:continue
            middle=(low+high)/2;active={}
            for a,b in zip(poly,poly[1:]+poly[:1]):
                ra,rb=a[0][other],b[0][other]
                if min(ra,rb)<middle<max(ra,rb):
                    hit=intersection(a,b,other,middle,F(0))
                    active[hit[0][transverse]]=(a,b)
            if len(active)!=2:raise ValueError('thin polygon cross-section must have two edges')
            left,right=[active[x] for x in sorted(active)]
            dlo,dhi=knots[low],knots[high]
            endpoints=[(intersection(*left,other,r,dr),intersection(*right,other,r,dr)) for r,dr in ((low,dlo),(high,dhi))]
            if middle>F(1,2):
                low,high,dlo,dhi=1-high,1-low,-dhi,-dlo
                endpoints.reverse()
            widthf=high-low;dwidthf=dhi-dlo
            if low==0:
                if dlo!=0:raise ValueError('fixed thin endpoint has a nonzero derivative')
                # The exact normalized interpolation is t**5. Differentiating
                # a quotient of fifth roots would numerically manufacture dxi.
                xi=t**5;dxi=np.zeros_like(t);complement=1-xi
                jac=5*float(high)*t**4;djac=5*float(dhi)*t**4
            else:
                xi=t;dxi=np.zeros_like(t);complement=1-t
                jac=np.full_like(t,float(widthf));djac=np.full_like(t,float(dwidthf))
            # Each endpoint contains the left and right physical section vertices.
            def interpolate(values,changes):
                lo=np.asarray(values[0],float);hi=np.asarray(values[1],float)
                dlo=np.asarray(changes[0],float);dhi=np.asarray(changes[1],float)
                val=complement[:,None]*lo+xi[:,None]*hi
                difference=np.asarray([b-a for a,b in zip(values[0],values[1])],float)
                direction=complement[:,None]*dlo+xi[:,None]*dhi+dxi[:,None]*difference
                return val,direction
            boundaries=[];dboundaries=[]
            for side in range(2):
                val,dval=interpolate([ends[side][0] for ends in endpoints],[ends[side][1] for ends in endpoints])
                boundaries.append(val);dboundaries.append(dval)
            bb=((1-t)[None,:,None]*boundaries[0][:,None,:]+t[None,:,None]*boundaries[1][:,None,:]).reshape(-1,3)
            dbb=((1-t)[None,:,None]*dboundaries[0][:,None,:]+t[None,:,None]*dboundaries[1][:,None,:]).reshape(-1,3)
            widths=[ends[1][0][transverse]-ends[0][0][transverse] for ends in endpoints]
            dwidths=[ends[1][1][transverse]-ends[0][1][transverse] for ends in endpoints]
            cross=complement*float(widths[0])+xi*float(widths[1])
            dcross=complement*float(dwidths[0])+xi*float(dwidths[1])+dxi*float(widths[1]-widths[0])
            if np.any(cross<=0):raise ValueError('nonpositive thin cross-section')
            def measure(normalized):
                factor=scalar(local[10])*(1 if normalized else sf)
                dfactor=F(0) if normalized else scalar(local[10])*dsf
                value=jac*cross;direction=djac*cross+jac*dcross
                return ((w[:,None]*w*(value*float(factor))[:,None]).ravel(),
                        (w[:,None]*w*(direction*float(factor)+value*float(dfactor))[:,None]).ravel())
            def linear(values,changes):
                val=[];dval=[]
                for side in range(2):
                    ev=[dot(ends[side][0],values) for ends in endpoints]
                    de=[dot(ends[side][1],values)+dot(ends[side][0],changes) for ends in endpoints]
                    vv=complement*float(ev[0])+xi*float(ev[1])
                    dd=complement*float(de[0])+xi*float(de[1])+dxi*float(ev[1]-ev[0])
                    val.append(vv);dval.append(dd)
                return (((1-t)[None,:]*val[0][:,None]+t[None,:]*val[1][:,None]).ravel(),
                        ((1-t)[None,:]*dval[0][:,None]+t[None,:]*dval[1][:,None]).ravel())
            yield tri,local,p,dp,poly,bb,dbb,measure,linear
