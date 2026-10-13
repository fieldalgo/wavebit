import re, zlib, sys

data = open(sys.argv[1], 'rb').read()
out = []
for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', data, re.S):
    try:
        s = zlib.decompress(m.group(1))
    except Exception:
        continue
    chunks = []
    for t in re.finditer(rb'\[(.*?)\]\s*TJ|\(((?:\\.|[^\\)])*)\)\s*Tj|(T\*|Td|TD|Tm)', s, re.S):
        if t.group(1) is not None:
            parts = re.findall(rb'\(((?:\\.|[^\\)])*)\)|(-?\d+\.?\d*)', t.group(1))
            line = b''
            for p, num in parts:
                if p:
                    line += p
                elif num and float(num) < -200:
                    line += b' '
            chunks.append(line)
        elif t.group(2) is not None:
            chunks.append(t.group(2))
        else:
            chunks.append(b' ')
    if chunks:
        out.append(b''.join(chunks))
txt = b'\n'.join(out).decode('latin-1')
txt = re.sub(r'\\(\d{3})', lambda m: chr(int(m.group(1), 8)), txt)
txt = txt.replace('\\(', '(').replace('\\)', ')')
open('farjadpour_text.txt', 'w', encoding='utf-8').write(txt)
print(len(txt), 'chars from', len(out), 'streams')
