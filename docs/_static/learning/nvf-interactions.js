(()=>{
"use strict";
const byId = id => document.getElementById(id);
function boundaryUpdate(){const e=Number(byId('eps2').value);byId('eps2-value').textContent=e.toFixed(1);byId('normal-bar').style.width=(100/e)+'%';byId('boundary-result').textContent=`Eₙ₁=1, Eₙ₂=${(1/e).toFixed(4)}. Eₜ₁=Eₜ₂=1 and ε₂Eₙ₂=${(e/e).toFixed(1)} remain unchanged. Increasing ε₂ reduces the normal electric field on the right.`;}
function projectionUpdate(){const d=Number(byId('angle').value),a=d*Math.PI/180,c=Math.cos(a),s=Math.sin(a);byId('angle-value').textContent=d+'°';const x=200+90*c,y=130-90*s;byId('normal-line').setAttribute('x2',x);byId('normal-line').setAttribute('y2',y);byId('normal-tip').setAttribute('cx',x);byId('normal-tip').setAttribute('cy',y);const clean=v=>Math.abs(v)<1e-12?'0.0000':v.toFixed(4);byId('projection-result').textContent=`Pxx=${clean(c*c)}, Pxy=Pyx=${clean(c*s)}, Pyy=${clean(s*s)}. trace(P)=${(c*c+s*s).toFixed(1)}. The opposite direction (180°) gives the same projector.`;}
function ordersUpdate(){const nx=Number(byId('nx').value),ny=Number(byId('ny').value);byId('nx-value').textContent=nx;byId('ny-value').textContent=ny;let out='';for(let m=-4;m<=4;m++){for(let n=-4;n<=4;n++){const selected=Math.abs(m)<=nx&&Math.abs(n)<=ny;out+=`<circle cx="${188+34*m}" cy="${160-30*n}" r="${selected?6:3}" fill="${selected?'#117c77':'#d9e3e0'}"><title>(${m}, ${n}) ${selected?'retained':'excluded'}</title></circle>`;}}byId('mode-dots').innerHTML=out;const k=(2*nx+1)*(2*ny+1);byId('orders-result').textContent=`M = ${2*nx+1} × ${2*ny+1} = ${k} modes. C(ε) has shape ${k} × ${k}. Electromagnetic operators combine these blocks. This describes storage size, not optical accuracy.`;}

for (const [ids, update] of [[['eps2'], boundaryUpdate], [['angle'], projectionUpdate], [['nx','ny'], ordersUpdate]]) {
  if (!ids.every(id => byId(id))) continue;
  for (const id of ids) byId(id).addEventListener('input', update);
  update();
}

})();
