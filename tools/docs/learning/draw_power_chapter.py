"""Original figures for the power chapter; numeric values come from its example."""
from html import escape
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/docs/learning/ko'
DATA = json.loads((ROOT / 'docs/validation/learning-records/power-example.json').read_text())
plt.rcParams.update({
    'font.family': ['Apple SD Gothic Neo', 'DejaVu Sans'],
    'font.size': 12, 'svg.fonttype': 'none', 'axes.unicode_minus': False,
    'text.color': '#182b3e', 'axes.labelcolor': '#182b3e',
    'xtick.color': '#546779', 'ytick.color': '#546779',
})


def time_average():
    t = np.linspace(0, 1, 601)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), sharex=True,
                             gridspec_kw={'hspace': .40, 'wspace': .22})
    fig.subplots_adjust(top=.81, bottom=.14, left=.085, right=.97)
    fig.text(.5, .90, '위: 시간 신호    ·    아래: 두 신호의 곱과 한 주기 평균',
             ha='center', color='#546779', fontsize=12)
    for column, delta in enumerate([0., np.pi / 4]):
        e = np.cos(2 * np.pi * t)
        h = np.cos(2 * np.pi * t - delta)
        axes[0, column].plot(t, e, color='#2d60a1', lw=2.8, label='e(t) / E₀')
        axes[0, column].plot(t, h, color='#b56626', lw=2.3, ls='--', label='h(t) / H₀')
        axes[0, column].set_title(f'위상차 δ = {np.rad2deg(delta):.0f}°', fontsize=14, pad=10)
        axes[0, column].set_ylim(-1.2, 1.2)
        axes[0, column].legend(loc='upper center', ncol=2, fontsize=10, frameon=False)
        mean = .5 * np.cos(delta)
        axes[1, column].plot(t, e * h, color='#117c77', lw=2.5)
        axes[1, column].axhline(mean, color='#7c4731', ls='--', lw=2)
        axes[1, column].text(.52, mean + .07, f'시간평균 = {mean:.3f}',
                             color='#7c4731', fontsize=12,
                             bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2})
        axes[1, column].set_ylim(-.24, 1.1)
        axes[1, column].set_xlabel('시간 / 한 주기')
        for row in range(2):
            ax = axes[row, column]
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.axhline(0, color='#aabac4', lw=.8, zorder=0)
            ax.grid(alpha=.16)
            ax.spines[['top', 'right']].set_visible(False)
            for side in ['left', 'bottom']:
                ax.spines[side].set_color('#b8c6ce')
    axes[0, 0].set_ylabel('정규화한 장')
    axes[1, 0].set_ylabel('e(t) h(t) / (E₀H₀)')
    fig.text(.5, .025, '일정한 평균값은 남고, 2ω로 진동하는 부분은 시간평균에서 사라집니다.',
             ha='center', fontsize=12, color='#546779')
    fig.savefig(OUT / '12-time-average.svg', metadata={'Title': '전기장과 자기장의 시간평균',
                   'Description': '원래 코사인 신호, 위상차, 곱의 주기 평균을 직접 계산한 설명용 그래프.'})
    svg = OUT / '12-time-average.svg'
    svg.write_text(svg.read_text().replace('height="460.8pt"', 'height="430.8pt"')
                   .replace('viewBox="0 0 828 460.8"', 'viewBox="0 30 828 430.8"'))
    fig.savefig(OUT / '12-time-average.png', dpi=150)
    plt.close(fig)


class Diagram:
    def __init__(self, title, desc, height):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 58 1000 {height-58}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
                      '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;fill:#182b3e;font-size:20px}</style>',
                      f'<rect width="1000" height="{height}" fill="white"/>']

    def text(self, x, y, text, size=20, color='#182b3e', weight=400, anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="font-size:{size}px;fill:{color};font-weight:{weight}">{escape(text)}</text>')

    def markup(self, x, y, markup, size=20, color='#182b3e', weight=400, anchor='start'):
        """Like text(), for markup already built from sub() and plain()."""
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="font-size:{size}px;fill:{color};font-weight:{weight}">{markup}</text>')

    def rect(self, x, y, w, h, fill, radius=0, stroke='#cad7df'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

    def line(self, x1, y1, x2, y2, color='#879ba9', width=2):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')

    def arrow(self, x, y1, y2, color):
        self.line(x, y1, x, y2, color, 5)
        back = y2 - np.sign(y2 - y1) * 15
        self.parts.append(f'<polygon points="{x},{y2} {x-8},{back} {x+8},{back}" fill="{color}"/>')

    def save(self, filename):
        (OUT / filename).write_text('\n'.join(self.parts + ['</svg>']) + '\n')


def sub(symbol, index):
    """A symbol with a multi-letter subscript, such as P with 'in'."""
    return (f'<tspan baseline-shift="baseline">{escape(symbol)}</tspan>'
            f'<tspan baseline-shift="sub" style="font-size:70%">{escape(index)}</tspan>')


def plain(value):
    return f'<tspan baseline-shift="baseline">{escape(value)}</tspan>'


def reference_planes():
    d = Diagram('입사와 출사 경계에서 전력을 읽고, 그 사이의 흡수를 구합니다',
                '위쪽 입사. 보라색 선은 ② 입사 경계면과 ③ 출사 경계면이다. 입사파와 반사파의 순유입 전력, 투과파의 전력, 두 경계면 사이의 흡수를 구분한다.', 650)
    d.rect(30, 78, 555, 473, '#f6fafc', 12)
    d.text(53, 110, 'top · 입사 매질', weight=600)
    d.arrow(224, 137, 235, '#2d60a1')
    d.arrow(396, 235, 137, '#b56626')
    d.text(194, 132, '입사파', size=18)
    d.text(366, 132, '반사파', size=18)
    d.markup(140, 195, sub('P', 'in'), color='#2d60a1', size=24)
    d.markup(441, 195, sub('P', 'r'), color='#b56626', size=24)
    d.rect(53, 253, 509, 158, '#e7edf1', stroke='none')
    d.line(53, 252, 562, 252, '#745e97', 3)
    d.text(65, 239, '② 입사 경계면', size=18, color='#745e97', weight=600)
    d.line(53, 330, 562, 330, '#ccd7df', 1)
    d.text(78, 299, '두께를 지정한 층들', weight=600)
    d.text(78, 364, '층에서 흡수하는 전력', size=21)
    d.arrow(462, 279, 385, '#117c77')
    d.text(503, 352, '순전력', size=16, color='#117c77')
    d.line(53, 412, 562, 412, '#745e97', 3)
    d.text(65, 438, '③ 출사 경계면', size=18, color='#745e97', weight=600)
    d.arrow(307, 437, 499, '#2d60a1')
    d.markup(337, 474, plain('투과파 ') + sub('P', 't'), color='#2d60a1', size=21)
    d.text(53, 530, 'bot · 출사 매질', weight=600)
    d.text(651, 105, '① 입사파만의 전력을 기준으로', weight=600, size=21)
    d.markup(651, 144, sub('P', 'in') + plain('로 나누어 무차원화합니다.'), size=19)
    d.text(651, 192, '② 입사 경계면의 순유입 전력', weight=600, size=21)
    d.text(651, 224, '입사파·반사파의 복소 진폭을 합산', size=18)
    d.markup(651, 254, plain('E = ') + sub('E', 'in') + plain(' + ') + sub('E', 'r')
             + plain(',  H = ') + sub('H', 'in') + plain(' + ') + sub('H', 'r'), size=21)
    d.text(651, 284, '합친 E와 H로 전력을 계산합니다.', size=18)
    d.markup(651, 317, sub('P', 'enter') + plain(' = 순유입 전력 / ') + sub('P', 'in'),
             size=20, color='#117c77')
    d.text(651, 366, '③ 출사 경계면의 투과 전력', weight=600, size=21)
    d.markup(651, 404, plain('T = ') + sub('P', 't') + plain(' / ') + sub('P', 'in'),
             color='#2d60a1', size=23)
    d.rect(637, 443, 334, 88, '#e5f4ef', 10)
    d.text(657, 480, '쌓은 층 전체의 흡수', weight=600)
    d.markup(657, 513, plain('A = ') + sub('P', 'enter') + plain(' − T'),
             color='#117c77', size=24, weight=600)
    d.arrow(65, 610, 570, '#546779')
    d.text(88, 583, '+z', size=19)
    d.text(165, 585, '그림은 위쪽 입사: 구조 안쪽은 −z 방향입니다.', size=18, color='#546779')
    d.text(165, 617, 'top·bot 반무한 매질에서의 흡수는 A에 포함하지 않습니다.', size=18, color='#546779')
    d.save('12-reference-planes.svg')


def wave_directions():
    """Right-handed plane-wave fields and the normal projection of power flux."""
    d = Diagram('전기장과 자기장의 방향, 표면을 통과하는 전력',
                '왼쪽은 무손실 등방성 비자성 매질에서 위로 진행하는 선형 편광 평면파이다. 오른쪽은 top, 두께를 지정한 층들, bot을 함께 그린 xz 단면이다. 아래쪽 입사파는 bot에서 맨 아래층을 향하며, 두 매질 사이의 입사 경계면을 통과하는 전력 밀도는 입사 전력 흐름 크기의 cos theta 배이다.', 670)
    blue, orange, teal, purple = '#2d60a1', '#b56626', '#117c77', '#745e97'

    def vector(x1, y1, x2, y2, color, width=5):
        dx, dy = x2-x1, y2-y1
        length = (dx*dx + dy*dy)**.5
        ux, uy = dx/length, dy/length
        bx, by = x2-15*ux, y2-15*uy
        d.line(x1, y1, bx, by, color, width)
        d.parts.append(f'<polygon points="{x2},{y2} {bx-7*uy},{by+7*ux} {bx+7*uy},{by-7*ux}" fill="{color}"/>')

    d.rect(24, 78, 458, 550, '#f6fafc', 12)
    d.rect(500, 78, 476, 550, '#f6fafc', 12)
    d.text(48, 116, '① 전기장·자기장·진행 방향', size=24, weight=650)
    d.text(48, 148, '아래쪽 수직 입사 · 한 시각의 장 방향', size=18, color='#546779')
    vector(170, 355, 170, 205, teal)
    vector(170, 355, 405, 355, blue)
    d.text(170, 190, '+z', size=18, color='#546779', anchor='middle')
    d.text(420, 362, '+x', size=18, color='#546779')
    d.text(135, 362, '+y', size=18, color='#546779', anchor='end')
    d.text(192, 224, 'k, ⟨S⟩', size=23, color=teal, weight=600)
    d.text(192, 256, '위로 진행', size=19, color=teal)
    d.text(285, 333, 'e(t)', size=23, color=blue, weight=600)
    d.parts.append(f'<circle cx="170" cy="355" r="14" fill="white" stroke="{orange}" stroke-width="3"/>')
    d.line(163, 348, 177, 362, orange, 3)
    d.line(163, 362, 177, 348, orange, 3)
    d.text(91, 405, '⊗ h(t): 화면 안쪽', size=21, color=orange, weight=600)
    d.text(91, 439, 'e(t) × h(t)는 위쪽을 향합니다.', size=19)
    d.rect(48, 470, 410, 126, '#e8f1f9', 10, stroke='none')
    d.text(70, 504, '굴절률 n: 자기장의 크기를 결정', size=20, weight=600)
    d.text(253, 544, '|H| = (n / Z₀) |E|', size=26, color=blue, anchor='middle')
    d.text(253, 576, '같은 |E|라도 n이 크면 |H|가 큽니다.', size=18, anchor='middle')

    d.text(524, 116, '② 입사 경계면을 통과하는 전력', size=24, weight=650)
    d.text(524, 148, '아래쪽 비스듬한 입사 · xz 단면', size=18, color='#546779')
    d.text(620, 180, 'top', size=18, color='#546779')
    for y, fill, label in [(190, '#dce8f2', '첫 번째 층'),
                           (230, '#edf2f5', '⋯'),
                           (270, '#dce8f2', '맨 아래층')]:
        d.rect(620, y, 324, 40, fill)
        d.text(782, y+27, label, size=19, anchor='middle', weight=600)
    d.parts.append('<path d="M 614 190 H 606 V 310 H 614" fill="none" stroke="#546779" stroke-width="2"/>')
    d.text(540, 234, '구조체', size=19, weight=600)
    d.text(528, 263, '쌓은 층들', size=17, color='#546779')
    d.text(620, 340, 'bot', size=18, color='#546779')
    # Both vectors stay in bot. Their tips reach the incident boundary;
    # no continuation into the stack is implied by this projection diagram.
    vector(705, 440, 705, 310, purple)
    vector(705, 440, 855, 310, teal)
    d.text(842, 362, '⟨S⟩', size=26, color=teal, weight=600)
    d.text(816, 394, '입사파', size=20, color=teal)
    d.parts.append(f'<text x="679" y="377" text-anchor="end" style="font-size:26px;fill:{purple};font-weight:600">I<tspan baseline-shift="sub" style="font-size:70%">⊥</tspan></text>')
    d.text(679, 407, '법선 성분', size=19, color=purple, anchor='end')
    d.text(679, 433, '+z', size=18, color='#546779', anchor='end')
    # Angle from +z to the power-flow vector, with a 52 px radius.
    length = (150**2 + 130**2)**.5
    arc_x, arc_y = 705 + 52*150/length, 440 - 52*130/length
    d.parts.append(f'<path d="M 705 388 A 52 52 0 0 1 {arc_x} {arc_y}" fill="none" stroke="{teal}" stroke-width="2.5"/>')
    d.text(724, 382, 'θ', size=27, color=teal)
    d.parts.append('<path d="M 705 324 L 719 324 L 719 310" fill="none" stroke="#879ba9" stroke-width="1.5"/>')
    d.rect(524, 470, 428, 126, '#f1edf7', 10, stroke='none')
    d.text(546, 504, 'cos θ: 법선 방향으로 투영한 비율', size=20, weight=600)
    d.parts.append(f'<text x="738" y="544" text-anchor="middle" style="font-size:26px;fill:{purple}">I<tspan baseline-shift="sub" style="font-size:70%">⊥</tspan><tspan baseline-shift="baseline"> = |⟨S⟩| cos θ</tspan></text>')
    d.text(738, 576, 'θ = 0°이면 전력 흐름 전체가 통과합니다.', size=18, anchor='middle')
    d.text(500, 656, '입사 매질은 무손실·등방성·비자성 · 입사파만 표시 · 화살표 길이는 크기의 비교가 아닙니다.', size=17, color='#546779', anchor='middle')
    d.save('13-wave-directions.svg')


def fresnel_interface():
    """Connect the three waves at a normal-incidence Fresnel interface."""
    d = Diagram('수직 입사의 프레넬 경계조건',
                '수직 입사하는 두 무손실 등방성 비자성 매질의 경계. '
                '입사파와 반사파의 전체 접선 전기장과 자기장을 각각 투과파와 연결한다. '
                '반사파의 진행 방향이 반대이므로 자기장의 변환식에 마이너스가 붙는다.', 654)
    blue, orange, teal = '#2d60a1', '#b56626', '#117c77'

    def math_label(x, y, markup, size=25, color='#182b3e', anchor='middle'):
        d.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                       f'style="font-size:{size}px;fill:{color}">{markup}</text>')

    ei, er, et = sub('E', 'in'), sub('E', 'r'), sub('E', 't')
    hi, hr, ht = sub('H', 'in'), sub('H', 'r'), sub('H', 't')
    ni, no, z0 = sub('n', 'in'), sub('n', 'out'), sub('Z', '0')
    d.rect(24, 78, 458, 524, '#f6fafc', 12)
    d.rect(500, 78, 476, 524, '#f6fafc', 12)

    d.text(48, 114, '① 입사·반사·투과파', size=23, weight=650)
    d.text(48, 146, '수직 입사 · 두 매질 사이의 한 경계면', size=18, color='#546779')
    d.rect(48, 166, 410, 190, '#f3f7fa', stroke='none')
    d.rect(48, 356, 410, 192, '#e2edf6', stroke='none')
    math_label(70, 197, plain('입사 매질  ') + ni, size=20, anchor='start')
    math_label(70, 525, plain('출사 매질  ') + no, size=20, anchor='start')
    d.line(48, 356, 458, 356, '#879ba9', 2)
    d.text(440, 345, '경계면', size=17, color='#546779', anchor='end')
    d.text(160, 234, '입사파', size=20, color=blue, anchor='middle')
    d.text(350, 234, '반사파', size=20, color=orange, anchor='middle')
    d.arrow(160, 248, 344, blue)
    d.arrow(350, 344, 248, orange)
    math_label(130, 280, ei, size=26, color=blue, anchor='end')
    math_label(130, 317, hi, size=26, color=blue, anchor='end')
    math_label(379, 280, er, size=26, color=orange, anchor='start')
    math_label(379, 317, hr, size=26, color=orange, anchor='start')
    d.arrow(160, 370, 476, teal)
    d.text(198, 397, '투과파', size=20, color=teal)
    math_label(198, 434, et, size=26, color=teal, anchor='start')
    math_label(198, 471, ht, size=26, color=teal, anchor='start')
    d.text(253, 580, '화살표는 파의 진행 방향입니다.', size=18, color='#546779', anchor='middle')

    d.text(524, 114, '② 경계면에서 연결하는 두 조건', size=23, weight=650)
    d.text(524, 146, '각 파의 E와 H를 같은 방향 성분으로 읽습니다.', size=17, color='#546779')
    d.rect(524, 172, 428, 100, '#e8f1f9', 10, stroke='none')
    d.text(546, 205, '전기장 연속', size=20, weight=600)
    math_label(738, 247, ei + plain(' + ') + er + plain(' = ') + et, size=29, color=blue)
    d.rect(524, 287, 428, 137, '#e5f4ef', 10, stroke='none')
    d.text(546, 320, '자기장 연속', size=20, weight=600)
    math_label(738, 358, hi + plain(' + ') + hr + plain(' = ') + ht, size=26, color=teal)
    math_label(738, 402, ni + plain('(') + ei + plain(' − ') + er + plain(') = ') + no + et,
               size=25, color=teal)
    d.text(738, 453, '반사파는 진행 방향이 반대이므로', size=18, anchor='middle')
    math_label(738, 485, hr + plain(' = −(') + ni + plain(' / ') + z0 + plain(') ') + er,
               size=24, color=orange)
    d.text(738, 523, '두 조건을 함께 풀면', size=18, color='#546779', anchor='middle')
    # Two true fractions retain the same notation as the surrounding derivation.
    d.text(567, 574, 't =', size=26)
    math_label(646, 552, et, size=23)
    math_label(646, 586, ei, size=23)
    d.line(618, 561, 674, 561, '#182b3e', 1.6)
    d.text(689, 574, '=', size=26)
    math_label(820, 552, plain('2') + ni, size=23)
    math_label(820, 586, ni + plain(' + ') + no, size=23)
    d.line(739, 561, 901, 561, '#182b3e', 1.6)
    d.text(500, 636, '무손실·등방성·비자성 매질 · E와 H는 경계면에서 읽은 복소 진폭',
           size=17, color='#546779', anchor='middle')
    d.save('13-fresnel-interface.svg')



def power_balance():
    d = Diagram('1 − R과 실제 순유입 전력을 구별합니다',
                '무손실과 손실 입사를 비교한 전력 수지. 간섭 기여를 포함하면 순유입 전력과 투과 전력이 같아 쌓은 층 전체의 흡수는 0이다.', 560)
    for x, key, title, subtitle in [(30, 'lossless_incidence', '무손실 입사 매질', 'top의 상대유전율: 4'),
                                  (515, 'absorbing_incidence', '손실 입사 매질', 'top의 상대유전율: 4 + 3i')]:
        r = DATA[key]
        d.rect(x, 76, 455, 414, '#f8fafb', 12)
        d.text(x+24, 114, title, size=23, weight=650)
        d.text(x+24, 145, subtitle, color='#546779', size=18)
        d.text(x+24, 195, '입사 − 반사: 1 − R', size=19)
        d.text(x+427, 195, f'{1-r["R"]:.6f}', size=23, anchor='end')
        d.text(x+24, 238, '+ 간섭 기여', color='#a45b24', size=19)
        d.text(x+427, 238, f'{r["interference"]:.6f}', color='#a45b24', size=23, anchor='end')
        d.line(x+24, 261, x+427, 261)
        d.text(x+24, 303, '= 순유입 전력', color='#117c77', size=21, weight=600)
        d.text(x+427, 303, f'{r["power_entering"]:.6f}', color='#117c77', size=24, weight=600, anchor='end')
        d.text(x+24, 350, '− 투과 전력 T', color='#2d60a1', size=21)
        d.text(x+427, 350, f'{r["T"]:.6f}', color='#2d60a1', size=24, anchor='end')
        d.rect(x+22, 379, 411, 82, '#e5f4ef', 9, stroke='none')
        d.text(x+42, 427, '= 층 전체의 흡수 A', size=21, weight=600)
        d.text(x+406, 430, '0', size=29, color='#117c77', weight=650, anchor='end')
    d.text(30, 526, '층과 bot은 같은 무손실 재료 · λ = 1,000 nm · θ = 30° · s 입사', size=18, color='#546779')
    d.save('12-power-balance.svg')


def interference_absorption(preview=None):
    """Where the absorbing-incidence interference term comes from.

    Poynting's theorem between the reference plane (z = 0) and a height z in
    top: each part of the net downward power changes by the matching part of
    the absorbed power density (omega eps0 Im eps_r / 2)|E|^2, so the cross
    term at the plane is the interference part of top's absorption.
    """
    case = DATA['absorbing_incidence']
    r = complex(*case['r_electric_tangential'])
    q = complex(*case['admittance_times_Z0'])          # kz / k0 of the incident wave
    k0 = 2 * np.pi / 1000.                               # vacuum wavenumber, 1/nm
    z = np.linspace(0., 200., 4001)                      # height above the plane, nm
    e_inc = np.exp(-1j * q * k0 * z)                     # travels -z, |e_inc(0)| = 1
    e_ref = r * np.exp(1j * q * k0 * z)
    # Absorbed power per nm over the incident-only power at z = 0;
    # k0 Im(eps_r) / Re(q) = 2 k0 Im(q) because kx is real.
    rate = 2 * k0 * q.imag
    isolated = rate * (abs(e_inc) ** 2 + abs(e_ref) ** 2)
    actual = rate * abs(e_inc + e_ref) ** 2
    cross = 2 * q.imag / q.real * np.imag(np.conj(e_inc) * e_ref)
    net = abs(e_inc) ** 2 - abs(e_ref) ** 2 + cross

    def integral(density):
        steps = .5 * (density[1:] + density[:-1]) * np.diff(z)
        return np.concatenate([[0.], np.cumsum(steps)])

    np.testing.assert_allclose(cross[0], case['interference'], rtol=1e-12)
    np.testing.assert_allclose(net[0], case['power_entering'], rtol=1e-12)
    np.testing.assert_allclose(net - net[0], integral(actual), atol=1e-7)
    np.testing.assert_allclose(cross - cross[0], integral(actual - isolated), atol=1e-7)

    phase, spatial = np.angle(r), 2 * k0 * q.real
    bright_end = (np.pi / 2 - phase) / spatial           # absorption difference changes sign
    zero = (np.pi - phase) / spatial                     # cross term vanishes
    dark_end = (3 * np.pi / 2 - phase) / spatial
    peak = 2 * q.imag / q.real * abs(r)
    incident_at_zero = np.exp(2 * k0 * q.imag * zero)
    # Between the plane and the height where the cross term vanishes, top
    # absorbs exactly the cross term less than the isolated waves would.
    saved = -np.interp(zero, z, integral(actual - isolated))
    np.testing.assert_allclose(saved, cross[0], atol=1e-6)
    # The same interval's absorption, split three ways. The incident part is
    # only the incident component's decrease, not what top really absorbs.
    reflected_at_zero = abs(r) ** 2 * np.exp(-2 * k0 * q.imag * zero)
    split = {'incident': incident_at_zero - 1., 'reflected': abs(r) ** 2 - reflected_at_zero,
             'interference': -cross[0]}
    total = np.interp(zero, z, integral(actual))
    np.testing.assert_allclose(sum(split.values()), total, atol=1e-6)
    np.testing.assert_allclose(incident_at_zero - reflected_at_zero - net[0], total, atol=1e-6)

    orange, teal, ink, grey = '#b56626', '#117c77', '#182b3e', '#546779'
    fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(10.2, 6.6), sharey=True,
                                     gridspec_kw={'wspace': .10})
    fig.subplots_adjust(top=.88, bottom=.17, left=.085, right=.98)
    for ax in (ax_a, ax_c):
        for lo, hi, color in [(0, bright_end, orange), (bright_end, dark_end, teal),
                              (dark_end, 200, orange)]:
            ax.axhspan(lo, hi, color=color, alpha=.07, lw=0)
        ax.axhspan(-40, 0, color='#e7edf1', lw=0)
        ax.axhline(0, color='#745e97', lw=2.2)
        ax.set_ylim(-40, 200)
        ax.grid(alpha=.16)
        ax.spines[['top', 'right']].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color('#b8c6ce')

    per = 100.                                           # per 100 nm of height
    z, isolated, actual, cross = (a[::5] for a in (z, isolated, actual, cross))  # plot at 0.25 nm
    ax_a.fill_betweenx(z, per * isolated, per * actual, where=actual >= isolated,
                       interpolate=True, color=orange, alpha=.30, lw=0, label='간섭 때문에 더 흡수')
    ax_a.fill_betweenx(z, per * isolated, per * actual, where=actual < isolated,
                       interpolate=True, color=teal, alpha=.30, lw=0, label='간섭 때문에 덜 흡수')
    ax_a.plot(per * isolated, z, color=grey, lw=2, ls='--', label='입사파만 + 반사파만')
    ax_a.plot(per * actual, z, color=ink, lw=2.4, label='실제: 합친 전체 장')
    ax_a.plot([0, 0], [-40, 0], color=ink, lw=2.4)
    ax_a.set_xlim(0, 8.2)
    ax_a.set_title('① top 매질이 흡수하는 전력', fontsize=16, pad=10)
    ax_a.set_xlabel('높이 100 nm당 흡수 전력')
    ax_a.set_ylabel('입사 기준면에서 잰 높이 z (nm)')
    ax_a.legend(loc='upper left', fontsize=12, frameon=False)
    ax_a.text(.15, -22, '층·bot: 무손실, 흡수 없음', color=grey, fontsize=12.5)

    ax_c.axvline(0, color='#aabac4', lw=.8, zorder=0)
    ax_c.plot(cross, z, color='#a45b24', lw=2.6)
    for height, value, label in [(0, cross[0], f'기준면에 도착: +{cross[0]:.3f}'),
                                 (bright_end, peak, f'+{peak:.3f}'),
                                 (zero, 0., f'0 · 여기서 입사파만의 전력 {incident_at_zero:.2f}')]:
        ax_c.plot(value, height, 'o', color='#a45b24', ms=7, mec='white', mew=1.5)
        ax_c.text(value + .03, height + 4, label, fontsize=12.5, color=ink,
                  bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5})
    ax_c.text(.03, 190, '↓ 전력이 흐르는 방향으로 읽기', color=grey, fontsize=12.5, va='center')
    ax_c.text(-.40, (bright_end + zero) / 2, '덜 흡수한 만큼\n쌓임', color=teal, fontsize=12.5, va='center')
    ax_c.text(-.40, bright_end / 2, '더 흡수한 만큼\n줄어듦', color=orange, fontsize=12.5, va='center')
    ax_c.set_xlim(-.42, .80)
    ax_c.set_title('② 높이 z에서의 간섭 기여', fontsize=16, pad=10)
    ax_c.set_xlabel('간섭 기여 (입사파만의 전력 = 1)')
    fig.text(.5, .012, '손실 입사 예제 · top: εr = 4 + 3i · λ = 1,000 nm · θ = 30° · s 입사\n'
             '모든 값은 기준면에서 입사파만의 전력으로 나눈 비율',
             ha='center', fontsize=12, color=grey)
    name = '13-interference-absorption.svg'
    fig.savefig(OUT / name, metadata={
        'Title': '간섭 기여는 입사 매질 흡수의 변화입니다',
        'Description': '흡수성 입사 매질에서 높이에 따른 흡수 전력과 간섭 기여. '
                       '기준면 바로 위의 밝은 줄에서는 더 흡수하고 그 위의 어두운 줄에서는 덜 흡수하며, '
                       f'합하면 입사 매질이 {cross[0]:.3f}만큼 덜 흡수한다.'})
    if preview:
        fig.savefig(preview, dpi=110)
    plt.close(fig)
    return {'bright_end_nm': bright_end, 'zero_nm': zero, 'dark_end_nm': dark_end,
            'peak': peak, 'incident_at_zero': incident_at_zero, 'cross_at_plane': cross[0],
            'net_at_zero': incident_at_zero - reflected_at_zero,
            'absorbed_below_zero': dict(split, total=sum(split.values()))}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    time_average()
    reference_planes()
    wave_directions()
    fresnel_interface()
    power_balance()
    interference_absorption()
    print('Created six original power-chapter figures.')
