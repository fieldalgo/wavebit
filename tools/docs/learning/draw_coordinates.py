"""Draw shared physical coordinates and layer order, with raster indexing as an example."""
from pathlib import Path
from html import escape
import argparse


def draw():
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 720" role="img" aria-labelledby="coordinates-title coordinates-desc">
<title id="coordinates-title">좌표와 층 순서를 구분해서 읽기</title>
<desc id="coordinates-desc">좌표축, 셀 중심 원점, 층 순서는 래스터와 도형 방식에 공통이다. 왼쪽 xy 좌표도에는 래스터 방식의 4행 4열 배열을 예시로 겹쳐 표시했다. 재료값은 생략하고 단색 격자로 행과 열의 위치만 표시한다. 0번 행은 음의 y 쪽이고 원점은 셀 중심이다. x는 오른쪽, y는 위쪽, z는 화면 밖인 오른손 좌표도이다. 오른쪽은 재료 무늬를 생략한 두 층의 순서도이다. 회색은 재료를 뜻하지 않으며 첫째 층 두께는 d0, 둘째 층은 d1이다. 바닥 z=0에서 위로 z가 증가한다.</desc>
<defs><marker id="coord-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#435466"/></marker></defs>
<style>
text { font-family: -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif; fill:#182b3e; }
.axis { stroke:#435466; stroke-width:2; marker-end:url(#coord-arrow); }
.dimension { stroke:#657584; stroke-width:1.6; marker-start:url(#coord-arrow); marker-end:url(#coord-arrow); }
</style>
<rect width="1000" height="720" fill="white"/>
''']
    def rect(x,y,w,h,fill,stroke='none',rx=0,extra=''):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.6" {extra}/>')
    def text(x,y,value,size=20,weight=400,anchor='start',fill='#182b3e'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escape(value)}</text>')
    def line(x1,y1,x2,y2,cls):
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"/>')
    text(28,41,'좌표와 층 순서를 구분해서 읽기',28,750)
    text(28,76,'좌표축·원점·층 순서는 래스터와 도형(쉐입) 방식에 공통입니다.',18,fill='#536678')
    rect(24,101,464,566,'#ffffff','#d6e0e5',14)
    rect(512,101,464,566,'#ffffff','#d6e0e5',14)
    text(48,136,'A · 한 층의 xy 좌표',23,700)
    text(48,166,'아래 격자는 래스터 배열의 예시입니다.',18,fill='#536678')
    text(536,136,'B · 옆에서 본 층 순서 (z)',23,700)
    text(536,166,'각 층의 재료 무늬는 생략했습니다.',18,fill='#536678')
    # Physical +y points upward. Array row zero is the bottom displayed row.
    x0,y0,cell=135,228,60
    for row in range(4):
        y=y0+(3-row)*cell
        for col in range(4):
            rect(x0+col*cell,y,cell,cell,'#f4f6f8','#a6bbc7',
                 extra=f'data-row="{row}" data-column="{col}"')
        text(115,y+37,str(row),20,700 if row==0 else 400,'middle')
    for col in range(4):
        text(x0+(col+.5)*cell,492,str(col),18,anchor='middle')
    text(97,215,'행 y',17,anchor='middle',fill='#536678')
    text(391,492,'열 x',17,fill='#536678')
    line(255,480,255,187,'axis')
    line(101,348,423,348,'axis')
    text(269,197,'+y',20,650)
    text(432,355,'+x',20,650)
    parts.append('<circle cx="255" cy="348" r="5" fill="#182b3e"/>')
    rect(266,358,98,27,'white',rx=4)
    text(315,378,'(0, 0)',18,600,'middle')
    text(48,533,'래스터 예시 · 4 × 4 배열 (재료값 생략)',18,fill='#536678')
    text(48,576,'0번 행은 아래쪽(−y)에 있습니다.',19)
    text(48,615,'래스터 배열: [층, 행 y, 열 x]',21,650)
    parts.append('<circle cx="58" cy="638" r="8" fill="none" stroke="#435466" stroke-width="1.8"/><circle cx="58" cy="638" r="2.5" fill="#435466"/>')
    text(77,644,'+z: 화면 밖 · 원점 (0, 0)은 셀 중심',17,fill='#536678')
    # The same neutral fill is used for both finite layers: no material encoding.
    sx,sw=545,285
    rect(sx,208,sw,72,'#fafbfc','#c1cdd5')
    rect(sx,280,sw,105,'#edf0f3','#a9b7c2',extra='data-layer="1"')
    rect(sx,385,sw,105,'#edf0f3','#a9b7c2',extra='data-layer="0"')
    rect(sx,490,sw,72,'#fafbfc','#c1cdd5')
    text(565,239,'top · 위쪽 매질',20,600)
    text(565,265,'반무한 공간',16,fill='#536678')
    text(565,322,'structure[1]',23,650)
    text(565,354,'두 번째로 쌓은 층',18)
    text(565,427,'structure[0]',23,650)
    text(565,459,'첫 번째로 쌓은 층',18)
    text(565,520,'bot · 아래쪽 매질',20,600)
    text(565,547,'반무한 공간',16,fill='#536678')
    line(852,284,852,381,'dimension')
    line(852,389,852,486,'dimension')
    text(868,340,'d1',19,650)
    text(868,445,'d0',19,650)
    line(934,551,934,220,'axis')
    text(838,505,'z=0',17,650)
    text(944,397,'+z',19,650)
    text(536,615,'thickness = [d0, d1]',22,650)
    text(536,644,'회색은 층 구획이며 재료값을 뜻하지 않습니다.',16,fill='#536678')
    text(28,700,'직교 셀의 개념도 · 사선 셀의 래스터 배열 축은 [층, v, u]로 읽습니다.',17,fill='#536678')
    parts.append('</svg>')
    return '\n'.join(parts)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.write_text(draw())


if __name__=='__main__':
    main()
