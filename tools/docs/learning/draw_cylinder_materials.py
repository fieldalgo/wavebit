"""Original SVG of the constant-permittivity cylinder example; no external art.

Orthographic projection of a right-handed frame with +z up. Dimensions and
material assignments match the adjacent executable example. The substrate's
visible depth is only an illustration of a semi-infinite medium.
"""
from pathlib import Path
from math import sin, cos, pi, atan2
from html import escape
import argparse


def draw():
    out=['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 880" role="img" aria-labelledby="cylinder-title cylinder-desc">
<title id="cylinder-title">공기 속 원기둥을 기판 위에 놓기</title>
<desc id="cylinder-desc">600×600 nm 한 셀의 입체도와 평면도. 셀 중심의 원기둥은 지름 240 nm, 높이 180 nm, 상대 유전율 12.25이다. 바닥 z=0 아래는 유전율 2.25인 반무한 기판 bot이다. z=180 위는 공기 top, 패턴 층에서 원기둥 바깥은 공기 background이며 둘 다 유전율 1이다. 모든 매질의 상대 투자율은 1이다. 위에서 파장 900 nm인 빛이 수직 입사한다.</desc>
<defs>
<marker id="dim-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#52677c"/></marker>
<marker id="ray-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#c44738"/></marker>
</defs>
<style>text{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;} .halo{paint-order:stroke;stroke:white;stroke-width:6;stroke-linejoin:round;}</style>
<rect width="1200" height="880" fill="white"/>
''']
    def text(x,y,s,size=19,color='#182b3e',weight=400,extra=''):
        out.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" font-weight="{weight}" {extra}>{escape(s)}</text>')
    def poly(points,fill,stroke='#b1bdc9',width=1.5,extra=''):
        pairs=' '.join(f'{x:.2f},{y:.2f}' for x,y in points)
        out.append(f'<polygon points="{pairs}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" {extra}/>')
    def line(a,b,stroke='#52677c',width=1.5,extra=''):
        out.append(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="{stroke}" stroke-width="{width}" {extra}/>')
    def rect(x,y,w,h,fill,stroke='#dce4ea',rx=12):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    def dot(p,color='#52677c'):
        out.append(f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="3.5" fill="{color}"/>')
    def proj(x,y,z=0):
        return 375+.50*x+.24*y, 425+.18*x-.28*y-.72*z
    def circle(z,t0=0,t1=2*pi,n=100):
        return [proj(120*cos(t0+(t1-t0)*i/n),120*sin(t0+(t1-t0)*i/n),z) for i in range(n+1)]
    dim='marker-start="url(#dim-arrow)" marker-end="url(#dim-arrow)"'
    text(30,43,'코드를 구조로 읽기: 기판 위의 원기둥',30,weight=750)
    text(30,78,'길이 단위는 nm · 원기둥 하나와 주변 공기가 패턴 층 하나를 이룹니다.',20,color='#52677c')
    text(30,119,'한 셀의 입체도',20,weight=650)
    # Semi-infinite substrate: finite visible depth is only a pictorial cutaway.
    corners=[(-300,-300),(300,-300),(300,300),(-300,300)]
    a,b,c,d=[proj(x,y) for x,y in corners]
    aa,bb,cc,dd=[proj(x,y,-65) for x,y in corners]
    poly([b,c,cc,bb],'#b6c6d9')
    poly([a,b,bb,aa],'#cad7e6')
    poly([a,b,c,d],'#edf2f8')
    # Air layer outline, not solid side walls. Its upper face is z=180.
    for x,y in corners:
        line(proj(x,y),proj(x,y,180),'#a7b7c6',1.4,'stroke-dasharray="5 6"')
    poly([proj(x,y,180) for x,y in corners],'none','#a7b7c6',1.4,'stroke-dasharray="5 6"')
    # The visible half of the circular cylinder side, followed by its top face.
    tangent=atan2(.24,.50)
    front_top=circle(180,tangent+pi,tangent+2*pi,60)
    front_bottom=circle(0,tangent+pi,tangent+2*pi,60)
    poly(front_top+list(reversed(front_bottom)),'#117c77','#0f6965')
    poly(circle(180),'#47a59b','#0f6965')
    # Height belongs to the finite layer and therefore to the cylinder.
    top_edge=max(circle(180),key=lambda p:p[0])
    bottom_edge=max(circle(0),key=lambda p:p[0])
    line(top_edge,(684,top_edge[1]),'#a7b7c6',1,'stroke-dasharray="3 4"')
    line(bottom_edge,(684,bottom_edge[1]),'#a7b7c6',1,'stroke-dasharray="3 4"')
    line((668,top_edge[1]+2),(668,bottom_edge[1]-2),extra=dim)
    text(692,352,'180 nm',21,weight=650)
    text(692,380,'높이',17,color='#52677c')
    text(692,406,'thickness[0]',14,color='#52677c')
    # Incoming wave travels down from the top half-space.
    line((375,143),(375,240),'#c44738',3.5,'marker-end="url(#ray-arrow)"')
    text(401,158,'입사광 · λ = 900 nm',20,'#af372c',650)
    text(401,184,'위에서 수직 입사',17,'#af372c')
    text(73,195,'top · 공기',20,'#3c658c',650)
    # Callouts for the four distinct assignment regions, without material walls.
    text(65,350,'background',20,'#3c658c',650)
    text(65,376,'층 안의 주변 공기',17,color='#52677c')
    air=proj(-220,-30,65)
    line((217,368),air,'#6b869c');dot(air,'#6b869c')
    text(373,379,'material',18,'white',650,'text-anchor="middle"')
    text(0,0,'bot · 기판',21,'#3c526d',650,
         'text-anchor="middle" dominant-baseline="central" transform="matrix(1 .36 0 1 303 532)"')
    anchor=proj(-300,-300)
    dot(anchor)
    line(anchor,(88,anchor[1]))
    text(33,anchor[1]+6,'z=0',19,weight=650)
    top_anchor=proj(-300,-300,180)
    dot(top_anchor)
    line(top_anchor,(96,top_anchor[1]))
    text(33,top_anchor[1]+6,'z=180',17,weight=650)
    text(121,643,'기판은 아래로 계속 이어집니다.',18,color='#52677c')
    # Plan view uses the same diameter-to-period ratio (240 / 600).
    rect(815,109,360,458,'#f8fafc')
    text(837,143,'위에서 본 한 셀',21,weight=650)
    x0,y0,side=854,202,260
    rect(x0,y0,side,side,'white','#a7b7c6',0)
    cx,cy=x0+side/2,y0+side/2
    out.append(f'<circle cx="{cx}" cy="{cy}" r="52" fill="#47a59b" stroke="#0f6965" stroke-width="1.5"/>')
    line((x0,183),(x0+side,183),extra=dim)
    text(cx,174,'Px = 600',18,weight=600,extra='text-anchor="middle"')
    line((1134,y0),(1134,y0+side),extra=dim)
    text(0,0,'Py = 600',18,weight=600,extra=f'text-anchor="middle" transform="matrix(0 -1 1 0 1157 {cy})"')
    line((cx-51,cy),(cx+51,cy),'#ffffff',2,
         'marker-start="url(#dim-arrow)" marker-end="url(#dim-arrow)"')
    text(cx,cy-11,'240',19,'white',700,'text-anchor="middle"')
    text(cx,493,'셀 중심 · cx = cy = 0',18,extra='text-anchor="middle"')
    text(cx,527,'lx = 240은 지름입니다.',19,weight=650,extra='text-anchor="middle"')
    text(837,601,'한 셀이 x·y로 반복됩니다.',19,weight=600)
    text(837,630,'period = (600, 600)',19,color='#52677c')
    # Four compact entries map every material argument to its exact region.
    entries=[
        ('top', '위쪽 공기', 'z > 180', 'εr = 1', '#f2f7fc','#3c658c'),
        ('background', '층 안 · 원기둥 바깥 공기', '0 < z < 180', 'εr = 1', '#f2f7fc','#3c658c'),
        ('material', '층 안 · 원기둥 내부', '0 < z < 180', 'εr = 12.25', '#eaf5f2','#0f6965'),
        ('bot', '아래쪽 기판', 'z < 0', 'εr = 2.25', '#edf2f8','#3c526d'),
    ]
    for i,(key,region,height,value,fill,ink) in enumerate(entries):
        x=30+i*290
        rect(x,676,272,148,fill)
        text(x+18,706,key,22,ink,700)
        text(x+18,737,region,17)
        text(x+18,765,height,17,color='#52677c')
        text(x+18,802,value,24,ink,700)
    text(30,858,'모든 매질은 μr = 1 · 점선은 층의 범위이며 물체의 벽이 아닙니다 · 기판의 표시 두께는 실제 두께가 아닙니다.',17,color='#52677c')
    out.append('</svg>')
    return '\n'.join(out)+'\n'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.write_text(draw())
