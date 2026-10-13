"""Original vector illustration of Wavebit's periodic stack and top incidence.

Public xyz is right-handed with +z upward; depth is only a drawing helper. No external art is used.
"""
from pathlib import Path
from math import sin, cos, radians
from html import escape
import argparse


def draw():
    out = ['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 900" role="img" aria-labelledby="stack-title stack-desc">
<title id="stack-title">반복되는 패턴, 쌓인 층, 입사광을 한눈에 보기</title>
<desc id="stack-desc">직교 격자의 두 층을 비스듬히 본 개념도. 한 셀이 x와 y로 반복되고 첫째 층은 아래에, 둘째 층은 그 위에 쌓인다. 구조 바닥은 z=0이다. 입사광은 위에서 비스듬히 들어오며 진행 방향의 z 성분은 음수이다. theta는 빛의 진행 방향과 안쪽 법선 −z 사이의 각, phi는 xy로 투영한 진행 방향과 x 방향 사이의 각이다. 화면에 그린 옅은 입사면은 실제 재료가 아니다.</desc>
<defs>
<marker id="dark-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#42566b"/></marker>
<marker id="light-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#c44738"/></marker>
<marker id="gold-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#a86508"/></marker>
</defs>
<style>text{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;} .halo{paint-order:stroke;stroke:white;stroke-width:6;stroke-linejoin:round;} .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}</style>
<rect width="1200" height="900" fill="white"/>
''']
    def text(x,y,s,size=20,color='#182b3e',weight=400,extra=''):
        out.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" font-weight="{weight}" {extra}>{escape(s)}</text>')
    def physical(x,y,z):
        # Right-handed frame viewed from above: x right/down, y right/up, z up.
        return (430+145*x+60*y, 524+25*x-55*y-z)
    def project(x,front,depth=0):
        # Preserve the drawing's front/back layout while height grows upward.
        return physical(x,-front,124-depth)
    def poly(points,fill,stroke='#a6b6c3',extra=''):
        pts=' '.join(f'{a:.2f},{b:.2f}' for a,b in points)
        out.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.6" {extra}/>')
    def line(a,b,color='#42566b',width=2,extra=''):
        out.append(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="{color}" stroke-width="{width}" {extra}/>')
    def path(points,color='#42566b',width=2,extra=''):
        d='M'+' L'.join(f'{a:.2f},{b:.2f}' for a,b in points)
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" {extra}/>')
    def card(y,n,title,lines):
        out.append(f'<rect x="825" y="{y}" width="349" height="172" rx="14" fill="#f6f8fa" stroke="#dce4ea"/>')
        out.append(f'<circle cx="853" cy="{y+31}" r="14" fill="#243b53"/>')
        text(848,y+37,str(n),17,'white',700)
        text(878,y+38,title,22,weight=700)
        for i,s in enumerate(lines): text(846,y+77+i*27,s,18)
    text(30,44,'먼저 전체 구조를 한눈에 보기',31,weight=750)
    text(30,79,'한 셀은 옆으로 반복되고, 여러 층은 위로 쌓입니다. 위에서 비스듬히 입사하는 예시입니다.',20,color='#52677c')
    # Two finite layers; visible side faces use shading, not a material-value color scale.
    corners=[(-1.5,-1.5),(1.5,-1.5),(1.5,1.5),(-1.5,1.5)]
    for top,bottom,idx in [(62,124,0),(0,62,1)]:
        a,b,c,d=[project(x,y,top) for x,y in corners]
        ab,bb,cb,db=[project(x,y,bottom) for x,y in corners]
        poly([b,c,cb,bb],'#c1cad4',extra=f'data-layer="{idx}"')
        poly([d,c,cb,db],'#dce2e9',extra=f'data-layer="{idx}"')
    poly([project(x,y) for x,y in corners],'#f1f5f7')
    for s in [-.5,.5]:
        line(project(s,-1.5),project(s,1.5),'#b7c6d1',1.3, 'stroke-dasharray="5 5"')
        line(project(-1.5,s),project(1.5,s),'#b7c6d1',1.3, 'stroke-dasharray="5 5"')
    # Teal motifs are just the repeated material pattern; no numerical epsilon encoded.
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            poly([project(x+dx,y+dy) for dx,dy in [(-.23,-.23),(.23,-.23),(.23,.23),(-.23,.23)]], '#117c77','#0f6965',extra='data-motif="true"')
    # A complete front-left period, clearly distinguished from the motif it contains.
    cell=[(-1.5,.5),(-.5,.5),(-.5,1.5),(-1.5,1.5)]
    poly([project(x,y) for x,y in cell],'#ffefb5','#a86508','fill-opacity="0.35" style="stroke-width:3" data-unit-cell="true"')
    # Repaint its motif to preserve the shared material color.
    poly([project(-1+dx,1+dy) for dx,dy in [(-.23,-.23),(.23,-.23),(.23,.23),(-.23,.23)]], '#117c77','#0f6965')
    text(179,407,'한 셀',19,'#7f4b03',700, 'class="halo"')
    line(project(-1.46,1.40),project(-.54,1.40),'#a86508',1.8,'marker-start="url(#gold-arrow)" marker-end="url(#gold-arrow)"')
    text(175,449,'Px',19,'#7f4b03',700,'class="halo"')
    line(project(-1.42,.56),project(-1.42,1.43),'#a86508',1.8,'marker-start="url(#gold-arrow)" marker-end="url(#gold-arrow)"')
    text(137,409,'Py',19,'#7f4b03',700,'class="halo"')
    # The incidence plane spans the chosen in-plane ray direction and z.
    theta,phi=radians(35),radians(35)
    hx,hy=cos(phi),-sin(phi)
    plane=[project(-1.45*hx,-1.45*hy,-235),project(.55*hx,.55*hy,-235),project(.55*hx,.55*hy,0),project(-1.45*hx,-1.45*hy,0)]
    poly(plane,'#dfeaf5','#9aafc0','fill-opacity="0.38" stroke-dasharray="5 5"')
    text(470,165,'입사면',20,'#536f88',650)
    text(470,191,'방향을 설명하는 가상의 면',16,'#536f88')
    hit=project(0,0)
    start=project(-2.1*sin(theta)*hx,-2.1*sin(theta)*hy,-210*cos(theta))
    line(start,hit,'#c44738',3.8,'marker-end="url(#light-arrow)"')
    text(start[0]-23,start[1]-23,'입사광 k',23,'#af372c',700)
    line(project(0,0,-225),hit,'#5e7285',1.8,'stroke-dasharray="5 5"')
    text(385,158,'수직선',17,'#53677c')
    # theta compares k to the inward normal -z; the reversed rays show the same angle.
    theta_arc=[project(-1.03*sin(theta*i/40)*hx,-1.03*sin(theta*i/40)*hy,-103*cos(theta*i/40)) for i in range(41)]
    path(theta_arc,'#a86508',2.2)
    text(398,285,'θ',27,'#965d0c',650,'class="halo"')
    # Azimuth from +x to the xy projection of the incoming propagation vector.
    line(hit,project(1.58,0),'#677c8e',1.6,'stroke-dasharray="4 4" marker-end="url(#dark-arrow)"')
    text(657,436,'+x',19,'#42566b',650,'class="halo"')
    line(hit,project(0,-1.4),'#677c8e',1.6,'stroke-dasharray="4 4" marker-end="url(#dark-arrow)"')
    text(505,320,'+y',19,'#42566b',650,'class="halo"')
    line(hit,project(1.6*hx,1.6*hy),'#c44738',2.0,'stroke-dasharray="5 4" marker-end="url(#light-arrow)"')
    text(650,385,'k의 xy 성분',17,'#af372c',600,'class="halo"')
    phi_arc=[project(1.12*cos(phi*i/40),1.12*-sin(phi*i/40)) for i in range(41)]
    path(phi_arc,'#a86508',2.2)
    text(586,414,'φ',26,'#965d0c',650,'class="halo"')
    out.append(f'<circle cx="{hit[0]}" cy="{hit[1]}" r="4" fill="#c44738"/>')
    # Map the labels onto the front x-z face, centered within each finite layer.
    # The x basis projects to (145, 25); z stays vertical in the image.
    for idx in range(2):
        cx,cy=project(0,1.5,31+62*idx)
        text(0,0,f'structure[{1-idx}]',22,weight=650,
             extra=f'text-anchor="middle" dominant-baseline="central" '
                   f'transform="matrix(1 {25/145:.8f} 0 1 {cx:.2f} {cy:.2f})" '
                   f'data-layer-label="{1-idx}"')
    front=project(-1.5,1.5)
    for off,label in [(0,'d1'),(62,'d0')]:
        line((100,front[1]+off+3),(100,front[1]+off+59),'#42566b',1.6,'marker-start="url(#dark-arrow)" marker-end="url(#dark-arrow)"')
        text(65,front[1]+off+37,label,20,weight=650)
    # Anchor the height origin to the actual lower face, outside the orientation key.
    bottom_anchor = project(0, 1.5, 124)
    out.append(f'<circle cx="{bottom_anchor[0]}" cy="{bottom_anchor[1]}" r="3.5" fill="#52677c"/>')
    path([bottom_anchor, (bottom_anchor[0], 666), (374, 666)], '#8193a4', 1.5)
    text(392,674,'z = 0 · 구조 바닥',21,weight=650)

    # One aligned information strip: orientation, height, and visual legend.
    out.append('<rect x="30" y="730" width="1144" height="122" rx="12" fill="#f6f8fa" stroke="#dce4ea"/>')
    line((547,752),(547,831),'#dce4ea',1)
    line((877,752),(877,831),'#dce4ea',1)
    # The triad describes directions only; it is not the physical z=0 origin.
    origin=(94,816)
    line(origin,(163,828),extra='marker-end="url(#dark-arrow)"')
    line(origin,(139,775),extra='marker-end="url(#dark-arrow)"')
    line(origin,(94,759),extra='marker-end="url(#dark-arrow)"')
    text(173,835,'+x',18,weight=650)
    text(146,777,'+y',18,weight=650)
    text(76,751,'+z',18,weight=650)
    text(220,763,'좌표 방향',17,color='#52677c',weight=600)
    text(220,795,'+z는 위쪽',23,weight=700)
    text(220,827,'오른손 규칙: x × y = z',18)
    text(575,763,'높이 기준',17,color='#52677c',weight=600)
    text(575,795,'바닥을 z=0으로 설정',20,weight=650)
    text(575,827,'h_total = d0 + d1',18,extra='class="mono"')
    text(905,763,'색상 범례',17,color='#52677c',weight=600)
    out.append('<rect x="905" y="782" width="17" height="17" rx="3" fill="#dce2e9" stroke="#a6b6c3"/>')
    text(935,797,'회색 · 층 구획',18)
    out.append('<rect x="905" y="813" width="17" height="17" rx="3" fill="#117c77"/>')
    text(935,828,'청록 · 반복 도형',18)
    card(117,1,'옆으로 반복되는 한 셀', ['노란 테두리 = 계산하는 반복 단위', 'period = (Px, Py)', '표시한 3×3 셀 밖으로도 계속 반복'])
    card(311,2,'바닥부터 위로 쌓는 층', ['아래 structure[0] → 위 structure[1]', 'thickness = [d0, d1]', '바닥 z=0에서 위로 높이가 증가'])
    card(505,3,'위에서 들어오는 빛', ['θ · 수직 방향에서 얼마나 기울었나', 'φ · xy에서 +x 기준으로 어느 방향인가', '편광은 별도 설정 → 8장에서 설명'])
    text(30,882,'직교 셀 · 위쪽 입사 · 두 층의 개념도입니다. 색은 특정 재료값을, 크기는 실제 비율을 뜻하지 않습니다.',16,color='#52677c')
    out.append('</svg>')
    return '\n'.join(out)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.write_text(draw())
