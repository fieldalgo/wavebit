"""Crude PDF text extraction (zlib streams + Tj/TJ operators); no libraries."""
import re, sys, zlib
data = open(sys.argv[1], 'rb').read()
out = []
BS = b'\\'
for m in re.finditer(rb'stream\r?\n', data):
    start = m.end()
    end = data.find(b'endstream', start)
    raw = data[start:end]
    try:
        dec = zlib.decompressobj().decompress(raw)
    except Exception:
        continue
    if b'TJ' not in dec and b'Tj' not in dec:
        continue
    txt = []
    for tm in re.finditer(rb'\[(.*?)\]\s*TJ|\(((?:\\.|[^\\)])*)\)\s*Tj|(T\*|Td|TD)', dec, re.S):
        if tm.group(1) is not None:
            parts = re.findall(rb'\(((?:\\.|[^\\)])*)\)|(-?\d+\.?\d*)', tm.group(1))
            s = b''
            for p, num in parts:
                if p:
                    s += p
                elif num and float(num) < -200:
                    s += b' '
            txt.append(s)
        elif tm.group(2) is not None:
            txt.append(tm.group(2))
        else:
            txt.append(b'\n')
    t = b''.join(txt)
    t = re.sub(rb'\\(\d{3})', lambda k: bytes([int(k.group(1), 8)]), t)
    t = t.replace(BS + b'(', b'(').replace(BS + b')', b')').replace(BS + BS, BS)
    out.append(t.decode('latin-1'))
print('\n=====\n'.join(out))
