"""Exact moving clipped polygons and radial integration from actual corners.

Rational geometry resolves zero areas before floating-point integration.
Positive radial bounds are split until their angular denominators vary by
at most a factor of two. This refines quadrature; it removes no trial field.
"""
from fractions import Fraction as F
import numpy as np
from numpy.polynomial.legendre import leggauss


from .pair_corner_partition import scalar,dot,det,point,moving_point,clip,inverse3,endpoint_linear,root_sum


def partitions(obj,base,settings,chart,split):
    length,s,zero,velocity,period,widths,height,center,trace,tg=chart
    ds=base[-1]['ds'];delta,delta_slope=trace._taper(tg);ddelta=delta_slope*ds
    sf,dsf=scalar(s),scalar(ds);a,b=map(scalar,widths);h=scalar(height)
    deltaf,ddeltaf=scalar(delta),scalar(ddelta)
    planes=[(0,-a-sf/2,-dsf/2),(0,-sf/2,-dsf/2),(0,sf/2,dsf/2),(0,b+sf/2,dsf/2),
            (1,-h/2-deltaf,-ddeltaf),(1,-h/2,F(0)),(1,-h/2+deltaf,ddeltaf),
            (1,h/2-deltaf,-ddeltaf),(1,h/2,F(0)),(1,h/2+deltaf,ddeltaf)] if split else []
    for tri,local in zip(obj.data['triangles'],base[-1]['local']):
        # Reconstruct the closed affine controls exactly. Rounded intermediate
        # refined vertices must not invent a nonzero area on a collapsed line.
        px,py=map(scalar,period)
        controls=[(-(px+a-b)/2,-py/2),((px-a+b)/2,-py/2),
                  ((px-a+b)/2,py/2),(-(px+a-b)/2,py/2),
                  (-a,-h/2),(F(0),-h/2),(b,-h/2),
                  (b,h/2),(F(0),h/2),(-a,h/2)]
        numbers=obj.data['contact_numerators'][obj.data['mapping'][tri[:3]]]
        denominator=int(obj.data['weight_denominator'])
        p0=[tuple(sum(F(int(n),denominator)*control[axis] for n,control in zip(row,controls))
                  for axis in range(2)) for row in numbers]
        v=[tuple(map(scalar,x)) for x in velocity[tri[:3]]]
        p=[tuple(x+sf*speed for x,speed in zip(row,vel)) for row,vel in zip(p0,v)]
        dp=[tuple(dsf*speed for speed in row) for row in v]
        polygon=[(tuple(F(int(i==j)) for i in range(3)),(F(0),)*3) for j in range(3)]
        polygons=[polygon]
        for axis,bound,direction in planes:
            divided=[]
            for poly in polygons:
                values=[point(vertex[0],p)[axis]-bound for vertex in poly]
                if min(values)<0<max(values):
                    divided.extend(clip(poly,p,dp,axis,bound,direction,sign) for sign in (1,-1))
                else:divided.append(poly)
            polygons=divided
        for poly in polygons:
            if len(poly)<3:continue
            yield tri,local,p,dp,poly



def charts(obj,base,settings,chart,split):
    eye=[tuple(F(int(i==j)) for i in range(3)) for j in range(3)]
    for tri,local,p,dp,poly in partitions(obj,base,settings,chart,split):
        if local[11]:continue
        hats=base[-1]['corner_fields'].hats[tri[:3]][:,base[-1]['corner_fields'].controls]
        corners=np.flatnonzero(np.any(hats==1,axis=1))
        if len(corners)<=1:
            anchor=int(corners[0]) if len(corners) else 0
            B=eye[anchor:]+eye[:anchor]
            yield tri,local,p,dp,poly,B
        elif len(corners)==2:
            a,b=map(int,corners);c=next(i for i in range(3) if i not in corners)
            midpoint=tuple((x+y)/2 for x,y in zip(eye[a],eye[b]))
            pseudo=[(F(int(i==a)-int(i==b)),F(0)) for i in range(3)]
            zero=[(F(0),F(0))]*3
            for side,B in ((1,[eye[a],midpoint,eye[c]]),(-1,[eye[b],eye[c],midpoint])):
                piece=clip(poly,pseudo,zero,0,F(0),F(0),side)
                if len(piece)<3:continue
                if det(B)<0:B=[B[0],B[2],B[1]]
                yield tri,local,p,dp,piece,B
        else:
            center=tuple(F(1,3) for _ in range(3))
            zero=[(F(0),F(0))]*3
            for i in range(3):
                j,k=(i+1)%3,(i+2)%3
                mj=tuple((a+b)/2 for a,b in zip(eye[i],eye[j]))
                mk=tuple((a+b)/2 for a,b in zip(eye[i],eye[k]))
                for B in ([eye[i],mj,center],[eye[i],center,mk]):
                    piece=poly
                    for col in zip(*inverse3(B)):
                        pseudo=[(v,F(0)) for v in col]
                        piece=clip(piece,pseudo,zero,0,F(0),F(0),1)
                    if len(piece)<3:continue
                    if not any(det([piece[0][0],piece[j][0],piece[j+1][0]])!=0 for j in range(1,len(piece)-1)):continue
                    yield tri,local,p,dp,piece,B


def samples(obj,base,settings,chart,order,split=True):
    power=base[-1]['corner_fields'].root_order
    t,w=leggauss(order);t=(t+1)/2;w=w/2
    sf,dsf=scalar(chart[1]),scalar(base[-1]['ds'])
    for tri,local,p,dp,poly,B in charts(obj,base,settings,chart,split):
        inv=inverse3(B);transposed=list(zip(*inv))
        vertices=[(tuple(dot(v[0],column) for column in transposed),tuple(dot(v[1],column) for column in transposed)) for v in poly]
        angular={}
        for v,dv in vertices:
            radial=v[1]+v[2];dradial=dv[1]+dv[2]
            if radial==0:continue
            angle=v[2]/radial;direction=(dv[2]-angle*dradial)/radial
            if angle in angular and angular[angle]!=direction:
                raise ValueError('coincident angular events have inconsistent directions')
            angular[angle]=direction
        angles=sorted(angular)
        edges=list(zip(vertices,vertices[1:]+vertices[:1]))
        for left,right in zip(angles,angles[1:]):
            if right==left:continue
            dleft,dright=angular[left],angular[right]
            middle=(left+right)/2;limits={}
            for (va,dva),(vb,dvb) in edges:
                n=(vb[2]-va[2],va[1]-vb[1]);dn=(dvb[2]-dva[2],dva[1]-dvb[1])
                c=n[0]*va[1]+n[1]*va[2]
                dc=dn[0]*va[1]+dn[1]*va[2]+n[0]*dva[1]+n[1]*dva[2]
                denominator=n[0]*(1-middle)+n[1]*middle
                if denominator==0:continue
                r=c/denominator
                if r<0:continue
                j=1 if vb[1]!=va[1] else 2
                if vb[j]==va[j]:continue
                alpha=(r*((1-middle) if j==1 else middle)-va[j])/(vb[j]-va[j])
                if 0<=alpha<=1:limits[r]=(n,dn,c,dc)
            if len(limits)!=2:
                raise ValueError(f'positive ray must have two polygon intersections, got {len(limits)}')
            low,high=[limits[r] for r in sorted(limits)]
            pending=[(left,right,dleft,dright,0)];segments=[]
            while pending:
                lo,hi,dlo,dhi,depth=pending.pop()
                steep=False
                for n,dn,c,dc in (low,high):
                    if c==0:continue
                    ends=[abs(n[0]*(1-v)+n[1]*v) for v in (lo,hi)]
                    if min(ends)==0:raise ValueError('a nonzero radial bound reaches a parallel ray')
                    steep |= max(ends)>2*min(ends)
                if steep:
                    if depth>=80:raise ValueError('radial angular partition exhausted')
                    mid=(lo+hi)/2;dmid=(dlo+dhi)/2
                    pending.extend([(mid,hi,dmid,dhi,depth+1),(lo,mid,dlo,dmid,depth+1)])
                else:segments.append((lo,hi,dlo,dhi))
            for left,right,dleft,dright in segments:
                angle=(1-t)*float(left)+t*float(right)
                dangle=(1-t)*float(dleft)+t*float(dright)
                complement=(1-t)*float(1-left)+t*float(1-right)
                den=[];dden=[];radial=[];dradial=[]
                for n,dn,c,dc in (low,high):
                    d,dd=endpoint_linear(n,dn,left,dleft,right,dright,t)
                    den.append(d);dden.append(dd)
                    if c==0:
                        if dc!=0:raise ValueError('a zero radial boundary has a nonzero direction')
                        r=np.zeros_like(t);dr=np.zeros_like(t)
                    elif n[0]==n[1] and dn[0]==dn[1]:
                        # A radial reference edge has an exact angle-independent
                        # radius. Do not create a derivative from rounded ratios.
                        r=np.full_like(t,float(c/n[0]))
                        dr=np.full_like(t,float((dc*n[0]-c*dn[0])/n[0]**2))
                    else:r=float(c)/d;dr=(float(dc)-r*dd)/d
                    radial.append(r);dradial.append(dr)
                coefficients=tuple(high[2]*x-low[2]*y for x,y in zip(low[0],high[0]))
                changes=tuple(high[3]*x+high[2]*dx-low[3]*y-low[2]*dy for x,dx,y,dy in zip(low[0],low[1],high[0],high[1]))
                numerator,dnumerator=endpoint_linear(coefficients,changes,left,dleft,right,dright,t)
                denominator=den[0]*den[1];ddenominator=dden[0]*den[1]+den[0]*dden[1]
                width=numerator/denominator;dwidth=(dnumerator-width*ddenominator)/denominator
                if np.any(width<=0):raise ValueError('nonpositive radial interval')
                roots=[r**(1/power) for r in radial]
                droots=[np.divide(dr,power*u**(power-1),out=np.zeros_like(dr),where=u>0) for dr,u in zip(dradial,roots)]
                S,DS=root_sum(roots[1],roots[0],droots[1],droots[0],power)
                root_width=width/S;droot_width=(dwidth-root_width*DS)/S
                u=roots[0][:,None]+root_width[:,None]*t
                du=droots[0][:,None]+droot_width[:,None]*t
                if low[2]==0:
                    r=radial[1][:,None]*t**power
                    dr=dradial[1][:,None]*t**power
                else:r=u**power;dr=power*u**(power-1)*du
                zeta=np.stack([1-r,r*complement[:,None],r*angle[:,None]],axis=-1)
                dzeta=np.stack([-dr,dr*complement[:,None]-r*dangle[:,None],dr*angle[:,None]+r*dangle[:,None]],axis=-1)
                bb=zeta.reshape(-1,3)@np.asarray(B,float);dbb=dzeta.reshape(-1,3)@np.asarray(B,float)
                # Carry angle-span times radial-width numerator as exact
                # coefficients before dividing by a small gap.
                span=right-left;dspan=dright-dleft
                def measure(normalized):
                    factor=sf if normalized else F(1);dfactor=dsf if normalized else F(0)
                    co=tuple(span*c/factor for c in coefficients)
                    dco=tuple((dspan*c+span*dc)/factor-span*c*dfactor/factor**2 for c,dc in zip(coefficients,changes))
                    num,dnum=endpoint_linear(co,dco,left,dleft,right,dright,t)
                    pref=num/(denominator*S)
                    dpref=(dnum-pref*(ddenominator*S+denominator*DS))/(denominator*S)
                    jac=det(B)*(scalar(local[9])+sf*scalar(local[10]));djac=det(B)*dsf*scalar(local[10])
                    weighted=power*u**(2*power-1)*pref[:,None]
                    changed=power*(2*power-1)*u**(2*power-2)*du*pref[:,None]+power*u**(2*power-1)*dpref[:,None]
                    return ((w[:,None]*w*weighted*float(jac)).ravel(),
                            (w[:,None]*w*(changed*float(jac)+weighted*float(djac))).ravel())
                a,da=root_sum(u,roots[0][:,None],du,droots[0][:,None],power)
                xi=t[None,:]*a/S[:,None]
                dxi=t[None,:]*(da-xi/t[None,:]*DS[:,None])/S[:,None]
                ac,_=root_sum(roots[1][:,None],u,droots[1][:,None],du,power)
                one_minus_xi=(1-t)[None,:]*ac/S[:,None]
                if low[2]==0:
                    xi=np.broadcast_to(t**power,(order,order));dxi=np.zeros_like(xi)
                    one_minus_xi=1-xi
                def linear(values,dvalues):
                    vals=[dot(row,values) for row in B];dvals=[dot(row,dvalues) for row in B]
                    endpoints=[];directions=[]
                    for n,dn,c,dc,d,dd in zip((low[0],high[0]),(low[1],high[1]),(low[2],high[2]),(low[3],high[3]),den,dden):
                        co=tuple(vals[0]*nj+c*(vj-vals[0]) for nj,vj in zip(n,vals[1:]))
                        dco=tuple(dvals[0]*nj+vals[0]*dnj+dc*(vj-vals[0])+c*(dvj-dvals[0]) for nj,dnj,vj,dvj in zip(n,dn,vals[1:],dvals[1:]))
                        # Exact moving profile/taper boundaries are constant
                        # scalar values along this ray intersection edge.
                        pivot=0 if n[0]!=0 else 1
                        constant=co[pivot]/n[pivot]
                        delta=(dco[pivot]-constant*dn[pivot])/n[pivot]
                        if all(co[j]==constant*n[j] and dco[j]==constant*dn[j]+delta*n[j] for j in range(2)):
                            val=np.full_like(t,float(constant));change=np.full_like(t,float(delta))
                        else:
                            num,dnum=endpoint_linear(co,dco,left,dleft,right,dright,t)
                            val=num/d;change=(dnum-val*dd)/d
                        endpoints.append(val);directions.append(change)
                    value=one_minus_xi*endpoints[0][:,None]+xi*endpoints[1][:,None]
                    tangent=one_minus_xi*directions[0][:,None]+xi*directions[1][:,None]+dxi*(endpoints[1]-endpoints[0])[:,None]
                    return value.ravel(),tangent.ravel()
                yield tri,local,p,dp,poly,bb,dbb,measure,linear
