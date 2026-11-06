(()=>{
'use strict';
const byId = id => document.getElementById(id);
const listen = (ids, fn) => { if (!ids.every(id => byId(id))) return; ids.forEach(id => byId(id).addEventListener('input',fn)); fn(); };
listen(['order-x','order-y'], () => {
  const nx = +byId('order-x').value, ny = +byId('order-y').value, h = (2*nx+1)*(2*ny+1);
  byId('order-x-out').textContent = nx; byId('order-y-out').textContent = ny;
  byId('order-readout').textContent = `전체 성분 수 H = ${h}개 · 결과 배열 크기 (행, 열) = (${2*ny+1}, ${2*nx+1}) · H² = ${h*h}`;
  const group=byId('order-cells'); group.replaceChildren();
  const cols=2*nx+1,rows=2*ny+1,dx=340/cols,dy=200/rows;
  for(let j=0;j<rows;j++)for(let i=0;i<cols;i++){
    const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
    for(const [k,v] of Object.entries({x:30+i*dx,y:20+j*dy,width:Math.max(1,dx-2),height:Math.max(1,dy-2),rx:2,fill:i===nx&&j===ny?'#b56626':'#337eb0'}))rect.setAttribute(k,v);
    group.append(rect);
  }
});
listen(['pol-psi','pol-chi'], () => {
  const psiDeg=+byId('pol-psi').value,chiDeg=+byId('pol-chi').value;
  const psi=psiDeg*Math.PI/180,chi=chiDeg*Math.PI/180;
  byId('pol-psi-out').textContent=`${psiDeg}°`;
  byId('pol-chi-out').textContent=`${chiDeg}°`;
  // Same normalized electric coefficients as incident_sp_amplitudes in Wavebit.
  const sr=Math.sin(psi)*Math.cos(chi),si=Math.cos(psi)*Math.sin(chi);
  const pr=Math.cos(psi)*Math.cos(chi),pi=-Math.sin(psi)*Math.sin(chi);
  // Re[a exp(-i wt)] = Re(a) cos(wt) + Im(a) sin(wt).
  const point=t=>[200+120*(sr*Math.cos(t)+si*Math.sin(t)),
                 150-120*(pr*Math.cos(t)+pi*Math.sin(t))];
  const points=Array.from({length:241},(_,j)=>point(j*2*Math.PI/240).join(','));
  byId('pol-curve').setAttribute('points',points.join(' '));
  const arrow=Array.from({length:21},(_,j)=>point(0.45+j*0.5/20).join(','));
  byId('pol-motion').setAttribute('points',arrow.join(' '));
  const [x,y]=point(0);
  byId('pol-vector').setAttribute('x2',x);byId('pol-vector').setAttribute('y2',y);
  const linear=chiDeg===0,circular=Math.abs(chiDeg)===45;
  const type=linear?(psiDeg===0||psiDeg===180?'p 선편광':psiDeg===90?'s 선편광':'선편광'):
    circular?'원편광':'타원편광';
  byId('pol-state').textContent=type;
  byId('pol-shape-note').textContent=linear?'전기장 끝점이 직선 위에서 왕복합니다.':
    circular?'두 반지름이 같아 원을 그립니다. χ의 부호로 회전 방향을 바꿉니다.':
    `짧은 반지름 / 긴 반지름 = ${Math.abs(Math.tan(chi)).toFixed(3)}. 작은 화살표가 시간 진행 방향입니다.`;
  byId('pol-angle-code').textContent=`ψ=${psiDeg}°, χ=${chiDeg}°`;
  const complex=(re,im)=>{
    const r=Math.abs(re)<0.00005?0:re,i=Math.abs(im)<0.00005?0:im;
    if(i===0)return r.toFixed(4);
    if(r===0)return `${i.toFixed(4)}j`;
    return `${r.toFixed(4)}${i<0?'-':'+'}${Math.abs(i).toFixed(4)}j`;
  };
  byId('pol-pair-code').textContent=`polarization=(${complex(sr,si)}, ${complex(pr,pi)})`;
});
listen(['phasor-time'], () => {
  const t=+byId('phasor-time').value*Math.PI/180,phase=Math.PI/3,angle=phase-t;
  byId('phasor-time-out').textContent=`${byId('phasor-time').value}°`;
  const x=155+94*Math.cos(angle),y=125-94*Math.sin(angle);
  byId('phasor-vector').setAttribute('x2',x);byId('phasor-vector').setAttribute('y2',y);
  byId('phasor-point').setAttribute('cx',x);byId('phasor-point').setAttribute('cy',125);
  byId('phasor-drop').setAttribute('x1',x);byId('phasor-drop').setAttribute('y1',y);byId('phasor-drop').setAttribute('x2',x);
  byId('phasor-readout').textContent=`Re[E exp(−iωt)] = ${Math.cos(angle).toFixed(3)} · |E|² = 1 (시간에 따라 일정)`;
});

// Chapter 13: interference in an absorbing incidence medium. The same single s interface
// as the chapter's example (top: 4 + i·Im, layer = bot = 2.25), in closed form.
(() => {
  const svg=byId('pf-fig'),button=byId('pf-play');
  if(!svg||!button||!byId('pf-loss'))return;
  const NS='http://www.w3.org/2000/svg';
  const C=(re,im=0)=>({re,im});
  const add=(a,b)=>C(a.re+b.re,a.im+b.im),sub=(a,b)=>C(a.re-b.re,a.im-b.im);
  const mul=(a,b)=>C(a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re);
  const div=(a,b)=>{const d=b.re*b.re+b.im*b.im;return C((a.re*b.re+a.im*b.im)/d,(a.im*b.re-a.re*b.im)/d);};
  const abs2=a=>a.re*a.re+a.im*a.im;
  const cexp=a=>{const m=Math.exp(a.re);return C(m*Math.cos(a.im),m*Math.sin(a.im));};
  const k0=2*Math.PI/1000,ZMIN=-60,ZMAX=200,TOP=46,BOTTOM=372,X0=[46,262,478],W=196;
  const yz=z=>TOP+(ZMAX-z)/(ZMAX-ZMIN)*(BOTTOM-TOP);
  const zs=Array.from({length:ZMAX-ZMIN+1},(_,i)=>ZMIN+i);
  const INK='#182b3e',MUTED='#6b7d8b',GRID='#e3eaee',BLUE='#2d60a1',ORANGE='#b56626',TEAL='#117c77',BROWN='#a45b24';
  // Phase direction 30° from the normal in top: kx = tan30·Re(kz). With kz = a + ib,
  // a² − b² = Re ε − kx² and 2ab = Im ε give a quadratic in a².
  const setup=im=>{
    const re=4,t=Math.tan(Math.PI/6),s=1+t*t;
    const a=Math.sqrt((re+Math.hypot(re,Math.sqrt(s)*im))/(2*s)),kx=t*a;
    const q=C(a,im/(2*a)),qOut=Math.sqrt(2.25-kx*kx);
    const r=div(sub(q,C(qOut)),add(q,C(qOut)));
    return {im,q,qOut,r,R:abs2(r),cross:2*q.im*r.im/q.re,T:qOut*abs2(add(C(1),r))/q.re};
  };
  const el=(tag,attrs)=>{const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);svg.append(e);return e;};
  // Inline styles, because .exercise svg text sets its own size and colour.
  const text=(x,y,s,style='',anchor='start')=>{const e=el('text',{x,y,'text-anchor':anchor,style:`font-size:11px;fill:${MUTED};${style}`});e.textContent=s;return e;};
  const path=pts=>pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join('');
  const step=(max,n)=>[.01,.02,.05,.1,.2,.5,1,2,5,10,20,50].find(v=>max/v<=n)||100;
  const label=v=>v===0?'0':(v<0?'−':'')+String(+Math.abs(v).toFixed(2));
  let P,F,xm1,lines=[],phase=0,raf=0,last=0,visible=true;
  let running=!matchMedia('(prefers-reduced-motion: reduce)').matches;
  const draw=()=>{
    svg.replaceChildren();
    const lossy=P.im>0,kr=P.q.re,ki=P.q.im,phi=Math.atan2(P.r.im,P.r.re);
    F=zs.map(z=>{
      if(z<0)return {s:mul(add(C(1),P.r),cexp(C(0,-P.qOut*k0*z)))};
      const a=mul(C(0,-k0*z),P.q),i=cexp(a),r=mul(P.r,cexp(C(-a.re,-a.im)));
      return {i,r,s:add(i,r)};
    });
    // Heights where the cross term of |E|² raises (more) or lowers the absorption.
    const runs=[];
    if(lossy){let a=0,more=Math.cos(phi)>0;
      for(let z=1;z<=ZMAX;z++){const m=Math.cos(phi+2*kr*k0*z)>0;
        if(m!==more||z===ZMAX){runs.push({a,b:z,more});a=z;more=m;}}}
    for(const u of runs)for(const x of X0)el('rect',{x,y:yz(u.b),width:W,height:yz(u.a)-yz(u.b),fill:u.more?ORANGE:TEAL,'fill-opacity':.08});
    for(const x of X0){
      el('rect',{x,y:yz(0),width:W,height:yz(ZMIN)-yz(0),fill:'#8a9aa6','fill-opacity':.12});
      for(const z of [-50,0,50,100,150,200])el('line',{x1:x,x2:x+W,y1:yz(z),y2:yz(z),stroke:GRID,'stroke-width':.6});
      el('rect',{x,y:TOP,width:W,height:BOTTOM-TOP,fill:'none',stroke:'#cad7df'});
      el('line',{x1:x,x2:x+W,y1:yz(0),y2:yz(0),stroke:'#745e97','stroke-width':1.6});
    }
    for(const z of [-50,0,50,100,150,200])text(40,yz(z)+4,label(z),'','end');
    text(40,32,'z (nm)','','end');
    const yt=(yz(ZMAX)+yz(0))/2,yb=(yz(0)+yz(ZMIN))/2;
    text(12,yt,'top','','middle').setAttribute('transform',`rotate(-90 12 ${yt})`);
    text(12,yb,'층·bot','','middle').setAttribute('transform',`rotate(-90 12 ${yb})`);
    ['전기장 (순간값)','top이 흡수하는 양','간섭 기여'].forEach((s,i)=>text(X0[i],16,s,`font-size:13px;font-weight:600;fill:${INK}`));
    const key=(x,color,dash,s)=>{el('line',{x1:x,x2:x+14,y1:30,y2:30,stroke:color,'stroke-width':2,'stroke-dasharray':dash||'none'});text(x+18,34,s);};
    key(X0[0],BLUE,0,'입사');key(X0[0]+56,ORANGE,0,'반사');key(X0[0]+112,INK,0,'합');
    key(X0[1],MUTED,'3 3','간섭 없으면');key(X0[1]+100,INK,0,'실제');
    text(X0[2],34,'실제 − (입사 − 반사)');
    // Panel 1: envelope here, the moving field in frame().
    xm1=1.08*Math.max(...F.map(f=>Math.sqrt(abs2(f.s))),...F.filter(f=>f.i).map(f=>Math.sqrt(abs2(f.i))));
    const c1=X0[0]+W/2,sx1=v=>c1+v/xm1*(W/2-6);
    el('line',{x1:c1,x2:c1,y1:TOP,y2:BOTTOM,stroke:GRID});
    for(const sign of [1,-1])el('path',{d:path(zs.map((z,k)=>[sx1(sign*Math.sqrt(abs2(F[k].s))),yz(z)])),fill:'none',stroke:MUTED,'stroke-dasharray':'3 3'});
    lines=[BLUE,ORANGE,INK].map((color,k)=>el('path',{fill:'none',stroke:color,'stroke-width':k===2?2:1.25}));
    const t1=step(xm1,2.5);for(const v of [-t1,0,t1])text(sx1(v),388,label(v),'','middle');
    // Panel 2: absorbed power per 100 nm of height, 2 k0 Im(q)|E|² (kx is real).
    const x2=X0[1];
    if(lossy){
      const rate=100*2*k0*ki,top=zs.map((z,k)=>[z,k]).filter(([z])=>z>=0);
      const D=top.map(([z,k])=>[z,rate*(abs2(F[k].i)+abs2(F[k].r))]),S=top.map(([z,k])=>[z,rate*abs2(F[k].s)]);
      const xm2=1.05*Math.max(...S.map(p=>p[1]),...D.map(p=>p[1])),sx2=v=>x2+v/xm2*W;
      for(const u of runs){const a=[],b=[];
        S.forEach(([z,v],k)=>{if(z>=u.a&&z<=u.b){a.push([sx2(v),yz(z)]);b.unshift([sx2(D[k][1]),yz(z)]);}});
        el('path',{d:path(a.concat(b))+'Z',fill:u.more?ORANGE:TEAL,'fill-opacity':.3});}
      el('path',{d:path(D.map(([z,v])=>[sx2(v),yz(z)])),fill:'none',stroke:MUTED,'stroke-width':1.5,'stroke-dasharray':'4 3'});
      el('path',{d:path(S.map(([z,v])=>[sx2(v),yz(z)])),fill:'none',stroke:INK,'stroke-width':2});
      const t2=step(xm2,5);for(let j=0;j*t2<=xm2;j++)text(sx2(j*t2),388,label(j*t2),'','middle');
      el('line',{x1:x2+1,x2:x2+1,y1:yz(0),y2:yz(ZMIN),stroke:INK,'stroke-width':2});
    }else{
      el('line',{x1:x2+1,x2:x2+1,y1:yz(ZMAX),y2:yz(ZMIN),stroke:INK,'stroke-width':2});
      text(x2+W/2,yz(110),'Im εr = 0',`font-size:12px;fill:${INK}`,'middle');
      text(x2+W/2,yz(110)+18,'흡수가 없어 아낄 것도 없음','font-size:12px','middle');
      text(x2,388,'0','','middle');
    }
    text(x2+W-6,yz(ZMIN)-8,'가로축: 높이 100 nm당','','end');
    // Panel 3: the cross term at height z; its slope is the absorption difference in panel 2.
    const c3=X0[2]+W/2,amp=2*ki*Math.sqrt(P.R)/kr,xm3=Math.max(.1,1.25*amp),sx3=v=>c3+v/xm3*(W/2-8);
    el('line',{x1:c3,x2:c3,y1:TOP,y2:BOTTOM,stroke:'#aab8c2'});
    const t3=step(xm3,2.5);for(const v of [-t3,0,t3])text(sx3(v),388,(v>0?'+':'')+label(v),'','middle');
    if(lossy){
      const Sx=z=>2*ki*(P.r.im*Math.cos(2*kr*k0*z)+P.r.re*Math.sin(2*kr*k0*z))/kr;
      el('path',{d:path(zs.filter(z=>z>=0).map(z=>[sx3(Sx(z)),yz(z)])),fill:'none',stroke:BROWN,'stroke-width':2});
      const z0=(Math.PI-phi)/(2*kr*k0);
      const dot=(z,s,dy)=>{const x=sx3(Sx(z));el('circle',{cx:x,cy:yz(z),r:4,fill:BROWN,stroke:'#fff','stroke-width':2});text(x+8,yz(z)+dy,s,`font-size:12px;fill:${INK}`);};
      dot(0,'+'+P.cross.toFixed(3),-6);
      if(z0<=ZMAX)dot(z0,`0 (${z0.toFixed(2)} nm)`,4);
    }else{
      el('line',{x1:c3,x2:c3,y1:yz(ZMAX),y2:yz(0),stroke:BROWN,'stroke-width':2});
      text(c3+8,yz(110),'항상 0',`font-size:12px;fill:${INK}`);
    }
    text(X0[2]+W-6,yz(ZMIN)-8,'입사파만의 전력 = 1','','end');
  };
  // Re[E exp(−iωt)] = Re(E) cos ωt + Im(E) sin ωt.
  const frame=()=>{
    const c=Math.cos(phase),s=Math.sin(phase),re=E=>E.re*c+E.im*s,sx1=v=>X0[0]+W/2+v/xm1*(W/2-6);
    const pts=[[],[],[]];
    zs.forEach((z,k)=>{const f=F[k],y=yz(z);if(f.i){pts[0].push([sx1(re(f.i)),y]);pts[1].push([sx1(re(f.r)),y]);}pts[2].push([sx1(re(f.s)),y]);});
    lines.forEach((line,k)=>line.setAttribute('d',path(pts[k])));
  };
  const readout=()=>{
    const f=v=>v.toFixed(6),entering=1-P.R+P.cross;
    const sum=document.createElement('p'),note=document.createElement('p');
    sum.textContent=`기준면: 입사파만 1 − 반사 ${f(P.R)} + 간섭 기여 ${f(P.cross)} = 순유입 ${f(entering)} = T`;
    note.textContent=P.im===0?'Im εr = 0이면 top이 흡수하지 않으므로, 간섭 무늬가 있어도 간섭 기여는 0이고 순유입 전력은 1 − R입니다.':
      `간섭 때문에 top이 기준면 근처에서 두 파를 따로 계산할 때보다 ${f(P.cross)} 덜 흡수합니다. `+
      (entering>1?'이 몫이 반사로 되돌아간 전력보다 커서 T가 1을 넘습니다.':'반사로 되돌아간 전력이 이보다 커서 T는 1보다 작습니다.');
    byId('pf-readout').replaceChildren(sum,note);
    byId('pf-legend').style.visibility=P.im>0?'visible':'hidden';
  };
  const tick=now=>{phase+=Math.min(50,last?now-last:0)*.0025;last=now;frame();raf=requestAnimationFrame(tick);};
  const sync=()=>{
    if(running&&visible&&!raf){last=0;raf=requestAnimationFrame(tick);}
    else if(!(running&&visible)&&raf){cancelAnimationFrame(raf);raf=0;}
  };
  const labelButton=()=>{button.textContent=running?'일시정지':'재생';};
  button.addEventListener('click',()=>{running=!running;labelButton();sync();});
  new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;sync();}).observe(svg);
  labelButton();
  listen(['pf-loss'],()=>{
    const im=+byId('pf-loss').value;
    byId('pf-loss-out').textContent=im.toFixed(1);
    P=setup(im);draw();frame();readout();
  });
})();

// Appendix A: kx circles a loop in air (ε = 1, branch point kx = 1) and kz = ±√(1 − kx²) is
// followed from the start in 0.25° steps, keeping the candidate nearer the previous value.
// The same walk as `track` in the lab's _complex_kx_root_figs.py (its figure 5).
(() => {
  const svg=byId('ckr-fig'),button=byId('ckr-play'),slider=byId('ckr-angle'),readout=byId('ckr-readout');
  const loopB=byId('ckr-loop-b');
  if(!svg||!button||!slider||!readout||!loopB||!byId('ckr-loop-a')||!byId('ckr-angle-out')||
     !['kx','cands','roots','status'].every(id=>byId('ckr-'+id)))return;
  const NS='http://www.w3.org/2000/svg';
  const INK='#182b3e',MUTED='#546779',HAIR='#b8c6ce',RED='#b33948',BLUE='#2d60a1';
  const C=(re,im=0)=>({re,im});
  // Principal square root: Re ≥ 0, and Im ≥ 0 on the negative real axis.
  const csqrt=z=>{
    const m=Math.hypot(z.re,z.im);if(m===0)return C(0);
    const a=Math.sqrt((m+Math.abs(z.re))/2);
    return z.re>=0?C(a,z.im/(2*a)):C(Math.abs(z.im)/(2*a),z.im<0?-a:a);
  };
  const neg=z=>C(-z.re,-z.im),dist=(a,b)=>Math.hypot(a.re-b.re,a.im-b.im);
  // The two candidates ±√(1 − kx²), principal root first.
  const pair=kx=>{const v=csqrt(C(1-kx.re*kx.re+kx.im*kx.im,-2*kx.re*kx.im));return [v,neg(v)];};
  // Each loop starts at its left edge on the real axis and runs counterclockwise.
  const LOOPS={a:{c:1,r:.35,color:RED,name:'고리 A'},b:{c:.3,r:.2,color:BLUE,name:'고리 B'}};
  const kxAt=(L,deg)=>{const t=deg*Math.PI/180;return C(L.c-L.r*Math.cos(t),-L.r*Math.sin(t));};
  const STEP=.25;
  const follow=(L,deg)=>{
    const start=pair(kxAt(L,0))[0],kxs=[kxAt(L,0)],kzs=[start],n=Math.ceil(deg/STEP-1e-9);
    let root=start,kx=kxs[0];
    for(let j=1;j<=n;j++){
      kx=kxAt(L,Math.min(j*STEP,deg));
      const [v,w]=pair(kx);
      root=dist(v,root)<=dist(w,root)?v:w;
      if(j%4===0||j===n){kxs.push(kx);kzs.push(root);}
    }
    return {start,root,kx,kxs,kzs};
  };
  const num=(v,plus)=>(v<0?'−':plus?'+':'')+Math.abs(v).toFixed(3);
  // Rounded to three digits, a part below the last digit dropped: +0.760, −1.118i, +0.646 + 0.542i.
  const fmt=(z,plus=true)=>{
    const re=Math.abs(z.re)<5e-4?0:z.re,im=Math.abs(z.im)<5e-4?0:z.im;
    if(im===0)return num(re,plus);
    if(re===0)return num(im,plus)+'i';
    return `${num(re,plus)} ${im<0?'−':'+'} ${Math.abs(im).toFixed(3)}i`;
  };
  const el=(tag,attrs)=>{const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);svg.append(e);return e;};
  // Inline styles, because .exercise svg text sets its own size and colour.
  const text=(x,y,s,style='',anchor='start')=>{const e=el('text',{x,y,'text-anchor':anchor,style:`font-size:11px;fill:${MUTED};${style}`});e.textContent=s;return e;};
  const path=pts=>pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join('');
  const label=v=>v===0?'0':(v<0?'−':'')+String(Math.abs(v));
  // Left: kx plane, Re −0.15…1.75. Right: kz plane, ±1.15. Each panel has one scale for both axes.
  const TOP=32,BOTTOM=262,Y0=(TOP+BOTTOM)/2;
  const P1={x0:44,x1:334,s:290/1.9,ox:44+.15*290/1.9},P2={x0:412,x1:642,s:100,ox:527};
  const at=(P,z)=>[P.ox+z.re*P.s,Y0-z.im*P.s];
  const panel=(P,title,name,ticksX,ticksY)=>{
    el('rect',{x:P.x0,y:TOP,width:P.x1-P.x0,height:BOTTOM-TOP,fill:'none',stroke:HAIR});
    el('line',{x1:P.x0,x2:P.x1,y1:Y0,y2:Y0,stroke:HAIR});
    el('line',{x1:P.ox,x2:P.ox,y1:TOP,y2:BOTTOM,stroke:HAIR});
    for(const v of ticksX){const x=P.ox+v*P.s;el('line',{x1:x,x2:x,y1:BOTTOM,y2:BOTTOM+4,stroke:HAIR});text(x,BOTTOM+16,label(v),'','middle');}
    for(const v of ticksY){const y=Y0-v*P.s;el('line',{x1:P.x0-4,x2:P.x0,y1:y,y2:y,stroke:HAIR});text(P.x0-7,y+4,label(v),'','end');}
    text(P.x0,20,title,`font-size:13px;font-weight:600;fill:${INK}`);
    text((P.x0+P.x1)/2,BOTTOM+33,`Re(${name})`,'','middle');
    text(P.x0-32,Y0,`Im(${name})`,'','middle').setAttribute('transform',`rotate(-90 ${P.x0-32} ${Y0})`);
  };
  // Nested when they coincide: ring (candidate) around square (start) around dot (followed root).
  const square=([x,y])=>el('rect',{x:x-5.5,y:y-5.5,width:11,height:11,fill:'none',stroke:INK,'stroke-width':1.5});
  const dot=([x,y],color,r)=>el('circle',{cx:x,cy:y,r,fill:color,stroke:'#fff','stroke-width':1.5});
  const ring=([x,y],r=8.5)=>el('circle',{cx:x,cy:y,r,fill:'none',stroke:MUTED,'stroke-width':1.5});
  const trail=(pts,color,width)=>{if(pts.length>1)el('path',{d:path(pts),fill:'none',stroke:color,'stroke-width':width,'stroke-linecap':'round','stroke-linejoin':'round'});};
  const legend=(y,items)=>{for(const [x,mark,s] of items){mark(x+7,y-4);text(x+18,y,s);}};
  let deg=+slider.value,hold=0,raf=0,last=0,visible=true;
  let running=!matchMedia('(prefers-reduced-motion: reduce)').matches;
  const draw=()=>{
    const L=LOOPS[loopB.checked?'b':'a'],A=Math.round(deg),F=follow(L,A);
    svg.replaceChildren();
    panel(P1,'kx 평면','kx',[0,.5,1,1.5],[-.5,0,.5]);
    panel(P2,'kz 평면: 두 후보 ±√(1 − kx²)','kz',[-1,0,1],[-1,0,1]);
    // kx plane: the loop, the branch point, the path so far and kx now.
    const loop=Array.from({length:180},(_,j)=>at(P1,kxAt(L,2*j)));
    el('path',{d:path(loop)+'Z',fill:'none',stroke:L.color,'stroke-width':1.5,'stroke-dasharray':'5 4','stroke-opacity':.75});
    text(P1.ox+L.c*P1.s,Y0-L.r*P1.s-8,L.name,`font-size:12px;font-weight:600;fill:${L.color}`,'middle');
    trail(F.kxs.map(z=>at(P1,z)),L.color,2.5);
    square(at(P1,F.kxs[0]));
    const bp=at(P1,C(1));
    dot(bp,INK,4.5);
    text(bp[0],bp[1]-22,'branch point',`fill:${INK}`,'middle');
    text(bp[0],bp[1]-10,'kx = 1',`fill:${INK}`,'middle');
    dot(at(P1,F.kx),L.color,5.5);
    // kz plane: the followed root's path, both candidates now, the starting root and the followed root.
    trail(F.kzs.map(z=>at(P2,z)),L.color,2);
    pair(F.kx).forEach(z=>ring(at(P2,z)));
    square(at(P2,F.start));
    dot(at(P2,F.root),L.color,4.5);
    const line=dash=>(x,y)=>el('line',{x1:x-7,x2:x+7,y1:y,y2:y,stroke:L.color,'stroke-width':dash?1.5:2.5,'stroke-dasharray':dash?'4 3':'none'});
    const box=(x,y)=>square([x,y]),fill=r=>(x,y)=>dot([x,y],L.color,r);
    legend(326,[[44,line(true),'고리'],[96,line(false),'지나온 길'],[173,fill(5.5),'지금의 kx'],[251,box,'출발점']]);
    legend(326,[[412,(x,y)=>ring([x,y],6),'두 후보'],[489,fill(4.5),'따라간 근'],[566,box,'출발한 근']]);
    // Readout.
    const [v,w]=pair(F.kx);
    byId('ckr-angle-out').textContent=`${A}°`;
    byId('ckr-kx').textContent=`${L.name} · ${A}° · kx = ${fmt(F.kx,false)}`;
    byId('ckr-cands').textContent=`두 후보 ±√(1 − kx²): ${fmt(v)}, ${fmt(w)}`;
    byId('ckr-roots').textContent=`따라간 근 ${fmt(F.root)} · 출발한 근 ${fmt(F.start)}`;
    const same=dist(F.root,F.start)<dist(F.root,neg(F.start));
    byId('ckr-status').textContent=A===0?'출발점입니다. 한 바퀴(360°)를 채우면 출발한 근과 비교합니다.':
      A%360===0?`${A===360?'한 바퀴':'두 바퀴'}(${A}°) · `+(same?'제자리: 출발한 근과 같은 값':'반대 근: 출발한 근의 부호가 바뀌었습니다(−kz)'):
      `${A<360?'첫째':'둘째'} 바퀴를 도는 중입니다. ${A<360?360:720}°에서 출발한 근과 비교합니다.`;
  };
  // Play: 72° per second, a pause at every full turn, then back to 0° after two turns.
  const SPEED=.072,HOLD=1400;
  const tick=now=>{
    const dt=Math.min(50,last?now-last:0);last=now;
    if(hold>0)hold-=dt;
    else if(deg>=720){deg=0;hold=600;slider.value='0';draw();}
    else{
      const turn=360*(Math.floor(deg/360)+1);
      deg=Math.min(turn,deg+dt*SPEED);
      if(deg===turn)hold=HOLD;
      slider.value=String(Math.round(deg));draw();
    }
    raf=requestAnimationFrame(tick);
  };
  // While playing, the readout changes every frame, so it is not announced until paused.
  const sync=()=>{
    const on=running&&visible;
    if(on&&!raf){last=0;raf=requestAnimationFrame(tick);}
    else if(!on&&raf){cancelAnimationFrame(raf);raf=0;}
    readout.setAttribute('aria-live',on?'off':'polite');
  };
  const labelButton=()=>{button.textContent=running?'일시정지':'재생';};
  button.addEventListener('click',()=>{running=!running;labelButton();sync();});
  new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;sync();}).observe(svg);
  labelButton();
  // Moving the slider by hand pauses the play; the loop choice keeps it going.
  const update=e=>{
    if(e&&e.target===slider){deg=+slider.value;hold=0;if(running){running=false;labelButton();sync();}}
    draw();
  };
  listen(['ckr-angle','ckr-loop-a','ckr-loop-b'],update);
  // Older Safari fires only change on radio buttons.
  ['ckr-loop-a','ckr-loop-b'].forEach(id=>byId(id).addEventListener('change',update));
})();

})();
