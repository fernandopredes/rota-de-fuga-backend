"""One-off diagnostic: render the map and verify the spec's claims about
path counts and stroke colors before writing the real preprocess pipeline."""
import collections
import fitz

PDF = '/home/fernando/Taka-Storm/BUZ43_BUZ92_00.pdf'

doc = fitz.open(PDF)
page = doc[0]
print('page rect:', page.rect)

# Overview render (moderate DPI so the whole A1 fits a reasonable PNG)
pix = page.get_pixmap(dpi=72)
pix.save('/home/fernando/rota-fuga-backend/debug/overview.png')
print('overview saved', pix.width, 'x', pix.height)

drawings = page.get_drawings()
print('total paths:', len(drawings))

# Group by stroke color, count paths and line segments
by_color = collections.Counter()
seg_by_color = collections.Counter()
for d in drawings:
    c = d.get('color')
    key = tuple(round(v, 2) for v in c) if c else None
    by_color[key] += 1
    seg_by_color[key] += sum(1 for it in d['items'] if it[0] == 'l')

print('\ntop stroke colors (paths / line segments):')
for color, n in by_color.most_common(20):
    print(f'  {color}: {n} paths, {seg_by_color[color]} segments')

# Text extraction sanity check (spec says only legend/stamp is real text)
text = page.get_text().strip()
print('\nreal text chars:', len(text))
print(text[:600])
