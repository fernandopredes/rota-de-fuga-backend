"""Preprocess do mapa de restrição (PDF vetorial QGIS) → artefatos em data/.

Roda uma vez por PDF. Cada estágio salva um artefato + uma imagem de debug:

  1 georef    → data/georef.json        debug/01_georef.png
  2 isobaths  → data/isobaths_raw.json  debug/02_isobaths.png
  3 labels    → data/isobaths.geojson   debug/03_labels.png
  4 grid      → data/depth_grid.npz     debug/04_depth_grid.png
  5 obstacles → data/obstacles.geojson  debug/05_obstacles.png
  6 wells     → data/wells.json         debug/06_wells.png

Uso:
  .venv/bin/python preprocess.py --stage 1
  .venv/bin/python preprocess.py --stage all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

import fitz
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.environ.get('MAP_PDF', '/home/fernando/Taka-Storm/BUZ70D_BUZ43_00.pdf')
DATA = os.path.join(BASE, 'data')
DEBUG = os.path.join(BASE, 'debug')
TESSDATA = os.path.join(BASE, 'tessdata')

os.makedirs(DATA, exist_ok=True)
os.makedirs(DEBUG, exist_ok=True)

# Grid lines are plain black with this stroke width; everything else on the
# map uses other colors/widths, so this is a safe signature.
GRID_COLOR = (0.0, 0.0, 0.0)
GRID_MIN_LEN = 800  # pts — grid lines span most of the map frame


# ── helpers ──────────────────────────────────────────────────────────────────

def open_page() -> fitz.Page:
    return fitz.open(PDF_PATH)[0]


def find_grid_lines(page: fitz.Page):
    """Long axis-aligned black lines → candidate UTM grid. Returns (xs, ys)."""
    xs, ys = [], []
    for d in page.get_drawings():
        color = d.get('color')
        if color is None or tuple(round(c, 2) for c in color) != GRID_COLOR:
            continue
        for it in d['items']:
            if it[0] != 'l':
                continue
            p1, p2 = it[1], it[2]
            dx, dy = abs(p1.x - p2.x), abs(p1.y - p2.y)
            if max(dx, dy) < GRID_MIN_LEN:
                continue
            if dx < 0.5:
                xs.append(round((p1.x + p2.x) / 2, 2))
            elif dy < 0.5:
                ys.append(round((p1.y + p2.y) / 2, 2))
    return sorted(set(xs)), sorted(set(ys))


def ocr_digits(img_gray: np.ndarray) -> list[dict]:
    """OCR digit runs in a grayscale numpy image. Returns [{text, cx, cy}] in
    image pixel coords."""
    from PIL import Image
    import tesserocr

    api = tesserocr.PyTessBaseAPI(path=TESSDATA, psm=tesserocr.PSM.SPARSE_TEXT)
    api.SetVariable('tessedit_char_whitelist', '0123456789EN')
    api.SetImage(Image.fromarray(img_gray))
    api.Recognize()
    out = []
    it = api.GetIterator()
    level = tesserocr.RIL.WORD
    if it is not None:
        while True:
            try:
                text = it.GetUTF8Text(level)
            except RuntimeError:
                break
            if text and re.fullmatch(r'\d{6,7}[EN]?', text.strip()):
                x0, y0, x1, y1 = it.BoundingBox(level)
                out.append({'text': text.strip(), 'cx': (x0 + x1) / 2, 'cy': (y0 + y1) / 2})
            if not it.Next(level):
                break
    api.End()
    return out


# ── stage 1: georeferencing ─────────────────────────────────────────────────

def _load_manual_georef() -> dict | None:
    """MANUAL_GEOREF env var: inline JSON or a path to a JSON file, shaped
    like {"easting": {"<x da linha>": <valor UTM E>, ...},
          "northing": {"<y da linha>": <valor UTM N>, ...}}.
    Keys are page coordinates — copy them from the "xs:"/"ys:" line stage 1
    prints, then read the matching UTM value off the map by eye (open
    debug/overview.png or render the border strip at high DPI). Needs ≥2
    points per axis; a 3rd is a nice cross-check."""
    raw = os.environ.get('MANUAL_GEOREF')
    if not raw:
        return None
    if os.path.exists(raw):
        with open(raw) as f:
            raw = f.read()
    data = json.loads(raw)
    return {
        'easting': {float(k): int(v) for k, v in data['easting'].items()},
        'northing': {float(k): int(v) for k, v in data['northing'].items()},
    }


def _snap_to_grid(labels: dict[float, int], coords: list[float], tol: float = 1.0) -> dict[float, int]:
    """Snap possibly-imprecise manual keys onto the actual grid-line
    coordinates so consensus_axis's exact-match lookup doesn't KeyError."""
    out = {}
    for k, v in labels.items():
        nearest = min(coords, key=lambda c: abs(c - k))
        if abs(nearest - k) <= tol:
            out[nearest] = v
        else:
            print(f'  aviso: MANUAL_GEOREF tem {k} sem linha de grade próxima '
                  f'(mais perto: {nearest}, dist {abs(nearest - k):.2f} pts) — ignorado')
    return out


def stage1_georef():
    page = open_page()
    xs, ys = find_grid_lines(page)
    print(f'grid: {len(xs)} vertical x {len(ys)} horizontal lines')
    print('  xs:', xs)
    print('  ys:', ys)

    # sanity: constant spacing
    dxs = np.diff(xs)
    dys = np.diff(ys)
    assert dxs.std() < 0.5 and dys.std() < 0.5, 'grid spacing not constant'
    spacing_pts = float(np.mean(np.concatenate([dxs, dys])))

    manual = _load_manual_georef()
    easting_by_x: dict[float, int] = {}
    northing_by_y: dict[float, int] = {}

    if manual:
        print('MANUAL_GEOREF definido — pulando OCR das bordas')
        easting_by_x = _snap_to_grid(manual['easting'], xs)
        northing_by_y = _snap_to_grid(manual['northing'], ys)
        print('easting labels (manual):', easting_by_x)
        print('northing labels (manual):', northing_by_y)
    else:
        # OCR the bottom strip (easting labels) and right strip (northing labels)
        dpi = 300
        zoom = dpi / 72

        # bottom strip: easting labels sit right below the frame corner (~y 1630-1672)
        clip = fitz.Rect(0, 1620, 1990, 1675)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        for w in ocr_digits(img):
            val = int(re.sub(r'[EN]', '', w['text']))
            page_x = clip.x0 + w['cx'] / zoom
            # match to nearest grid vertical
            nearest = min(xs, key=lambda gx: abs(gx - page_x))
            if abs(nearest - page_x) < spacing_pts / 2:
                easting_by_x[nearest] = val
        print('easting labels:', easting_by_x)

        # right strip: northing labels are plain horizontal text just inside the
        # right frame edge (white-halo'd over map content), e.g. "7287000N".
        clip = fitz.Rect(1895, 0, 1984, 1684)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        for w in ocr_digits(img):
            val = int(re.sub(r'[EN]', '', w['text']))
            page_y = clip.y0 + w['cy'] / zoom
            nearest = min(ys, key=lambda gy: abs(gy - page_y))
            if abs(nearest - page_y) < spacing_pts / 2:
                northing_by_y[nearest] = val
        print('northing labels:', northing_by_y)

    # Cross-check OCR'd (or manual) labels against constant grid spacing; keep the
    # consensus and infer the rest. Spacing in meters from the map scale.
    def consensus_axis(labels: dict[float, int], coords: list[float], sign: int):
        """Fit value = v0 + sign * k * index given OCR anchors; return values
        for every grid line. sign=+1 (easting grows with x), -1 (northing
        shrinks as page y grows)."""
        if len(labels) < 2:
            return None
        idx = {c: i for i, c in enumerate(coords)}
        pairs = [(idx[c], v) for c, v in labels.items()]
        # spacing candidates from all anchor pairs
        spacings = []
        for i, (ia, va) in enumerate(pairs):
            for ib, vb in pairs[i + 1:]:
                if ib != ia:
                    spacings.append((vb - va) / (ib - ia) * sign)
        spacing = float(np.median(spacings))
        # majority vote for the origin value
        v0s = [v - sign * spacing * i for i, v in pairs]
        v0 = float(np.median(v0s))
        vals = [v0 + sign * spacing * i for i in range(len(coords))]
        # drop anchors that disagree wildly (bad OCR) and refit once
        good = [(i, v) for i, v in pairs if abs(v - vals[i]) < spacing / 4]
        if len(good) >= 2 and len(good) < len(pairs):
            return consensus_axis({coords[i]: v for i, v in good}, coords, sign)
        return [round(v) for v in vals], spacing

    east = consensus_axis(easting_by_x, xs, +1)
    north = consensus_axis(northing_by_y, ys, -1)
    if east is None or north is None:
        raise SystemExit(
            'Pontos insuficientes para georreferenciar '
            f'(easting: {len(easting_by_x)}/2+, northing: {len(northing_by_y)}/2+).\n\n'
            'Defina a variável MANUAL_GEOREF com JSON inline ou caminho de arquivo:\n'
            '  MANUAL_GEOREF=\'{"easting": {"<x>": <E utm>, "<x>": <E utm>}, '
            '"northing": {"<y>": <N utm>, "<y>": <N utm>}}\' '
            'python preprocess.py --stage 1\n\n'
            f'Use como chave um dos valores impressos acima em "xs:" (para easting) '
            f'ou "ys:" (para northing), e leia o UTM correspondente direto no PDF — '
            f'abra debug/overview.png ou renderize a faixa da borda em alta DPI e leia '
            f'o rótulo por cima do olho. Pelo menos 2 pontos por eixo.\n'
            f'  xs: {xs}\n  ys: {ys}')
    east_vals, east_spacing = east
    north_vals, north_spacing = north
    print('easting per line: ', dict(zip(xs, east_vals)))
    print('northing per line:', dict(zip(ys, north_vals)))

    # least-squares affine: E = a*x + b ; N = c*y + d
    a, b = np.polyfit(xs, east_vals, 1)
    c, d = np.polyfit(ys, north_vals, 1)
    scale_x = a * 72 / 0.0254   # map scale denominator implied by x
    scale_y = -c * 72 / 0.0254
    print(f'affine: E = {a:.6f}*x + {b:.2f}   N = {c:.6f}*y + {d:.2f}')
    print(f'implied scale: 1:{scale_x:.0f} (x), 1:{scale_y:.0f} (y)')

    georef = {
        'pdf': os.path.basename(PDF_PATH),
        'crs': 'EPSG:31983',  # SIRGAS 2000 / UTM 23S (central meridian 45°W)
        'affine': {'a': a, 'b': b, 'c': c, 'd': d},
        'grid_xs': xs, 'grid_ys': ys,
        'easting_vals': east_vals, 'northing_vals': north_vals,
        'spacing_m': abs(east_spacing),
        'map_frame': [xs[0], ys[0], xs[-1], ys[-1]],
    }
    with open(os.path.join(DATA, 'georef.json'), 'w') as f:
        json.dump(georef, f, indent=2)

    # debug image: grid lines + assigned values over a dim render
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pix = page.get_pixmap(dpi=60)
    bg = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.imshow(bg, extent=(0, page.rect.width, page.rect.height, 0), alpha=0.5)
    for x, v in zip(xs, east_vals):
        ax.axvline(x, color='red', lw=0.8)
        ax.text(x, 30, f'{v}E', color='red', fontsize=8, ha='center', rotation=90)
    for y, v in zip(ys, north_vals):
        ax.axhline(y, color='blue', lw=0.8)
        ax.text(40, y, f'{v}N', color='blue', fontsize=8, va='center')
    ax.set_title(f'georef: 1:{scale_x:.0f}, spacing {abs(east_spacing):.0f} m')
    fig.savefig(os.path.join(DEBUG, '01_georef.png'), dpi=110, bbox_inches='tight')
    print('debug/01_georef.png saved')


# ── stage 2: isobath extraction ─────────────────────────────────────────────

def _parse_color(spec: str) -> tuple[float, float, float]:
    """Aceita '#RRGGBB' ou 'r,g,b' (0-1) — mesmo formato que o PyMuPDF usa
    internamente pros stroke colors."""
    spec = spec.strip().lstrip('#')
    if ',' in spec:
        r, g, b = (float(x) for x in spec.split(','))
    elif len(spec) == 6:
        r, g, b = (int(spec[i:i + 2], 16) / 255 for i in (0, 2, 4))
    else:
        raise ValueError(f"cor inválida: {spec!r} (use '#RRGGBB' ou 'r,g,b' 0-1)")
    return (round(r, 2), round(g, 2), round(b, 2))


# Cor das isóbatas — calibrada por mapa. Cada template QGIS/Petrobras pode
# usar uma cor diferente; em vez de travar num valor fixo, o admin informa a
# cor (opcional, hex) no upload — ISOBATH_COLOR env var, setada pelo
# jobs_api.py. Sem isso, cai no azul do primeiro mapa calibrado.
ISOBATH_COLOR = (_parse_color(os.environ['ISOBATH_COLOR'])
                  if os.environ.get('ISOBATH_COLOR') else (0.12, 0.47, 0.71))
# Map frame (the 'None'-colored border lines found during inspection); blue
# strokes outside it are legend swatches / cropped-away QGIS leftovers.
FRAME = (16.8, 11.3, 1980.4, 1672.4)


def _split_into_polylines(items, eps=1e-3):
    """A QGIS-batched path holds many disconnected features. Walk the item
    list and cut wherever the next segment doesn't start where the last one
    ended."""
    polys = []
    cur = []
    for it in items:
        if it[0] != 'l':
            # isobaths are pure line segments; 're'/'qu'/'c' items end a run
            if len(cur) >= 2:
                polys.append(cur)
            cur = []
            continue
        p1, p2 = it[1], it[2]
        if cur and (abs(cur[-1][0] - p1.x) > eps or abs(cur[-1][1] - p1.y) > eps):
            if len(cur) >= 2:
                polys.append(cur)
            cur = []
        if not cur:
            cur = [(p1.x, p1.y)]
        cur.append((p2.x, p2.y))
    if len(cur) >= 2:
        polys.append(cur)
    return polys


def _polyline_length(poly):
    a = np.asarray(poly)
    return float(np.sqrt(((a[1:] - a[:-1]) ** 2).sum(axis=1)).sum())


def _clip_to_frame(poly):
    """Keep only the portion inside the map frame (split where it exits)."""
    x0, y0, x1, y1 = FRAME
    out, cur = [], []
    for x, y in poly:
        if x0 <= x <= x1 and y0 <= y <= y1:
            cur.append((x, y))
        else:
            if len(cur) >= 2:
                out.append(cur)
            cur = []
    if len(cur) >= 2:
        out.append(cur)
    return out


def stage2_isobaths():
    page = open_page()
    polylines = []
    n_paths = 0
    for d in page.get_drawings():
        color = d.get('color')
        if color is None or tuple(round(c, 2) for c in color) != ISOBATH_COLOR:
            continue
        n_paths += 1
        for poly in _split_into_polylines(d['items']):
            polylines.extend(_clip_to_frame(poly))

    print(f'{n_paths} blue paths → {len(polylines)} raw polylines')

    # merge polylines whose endpoints touch (QGIS may split one contour
    # across paths); hash endpoints on a snapped grid. 0.1 pt snap: touching
    # pieces differ by float noise; distinct contours are never that close
    # at their endpoints.
    def key(pt):
        return (round(pt[0], 1), round(pt[1], 1))

    by_end: dict = {}
    for i, poly in enumerate(polylines):
        for pt in (poly[0], poly[-1]):
            by_end.setdefault(key(pt), []).append(i)

    used = [False] * len(polylines)
    merged = []
    for i in range(len(polylines)):
        if used[i]:
            continue
        used[i] = True
        chain = list(polylines[i])
        # extend forward/backward while a unique continuation exists
        for _ in range(2):
            while True:
                candidates = [j for j in by_end.get(key(chain[-1]), [])
                              if not used[j]]
                if len(candidates) != 1:
                    break
                j = candidates[0]
                nxt = polylines[j]
                used[j] = True
                if key(nxt[0]) == key(chain[-1]):
                    chain.extend(nxt[1:])
                else:
                    chain.extend(reversed(nxt[:-1]))
            chain.reverse()
        merged.append(chain)

    # drop tiny fragments (dashes, noise) — keep everything ≥ 8 pts (~85 m)
    kept = [p for p in merged if _polyline_length(p) >= 8]
    total_pts = sum(len(p) for p in kept)
    print(f'merged → {len(merged)} chains, kept {len(kept)} ≥8pts ({total_pts} vertices)')

    with open(os.path.join(DATA, 'isobaths_raw.json'), 'w') as f:
        json.dump({'polylines': [[[round(x, 3), round(y, 3)] for x, y in p] for p in kept]}, f)

    # debug plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 11))
    for p in kept:
        a = np.asarray(p)
        ax.plot(a[:, 0], a[:, 1], lw=0.4)
    ax.set_xlim(0, 2000); ax.set_ylim(1684, 0)
    ax.set_aspect('equal')
    ax.set_title(f'{len(kept)} isobath polylines')
    fig.savefig(os.path.join(DEBUG, '02_isobaths.png'), dpi=110, bbox_inches='tight')
    print('debug/02_isobaths.png saved')


# ── stage 3a: harvest depth samples via tile OCR ────────────────────────────
# Red equipment labels ("T-14  N=7274731m  E=761061m  LDA=-1948m") are drawn
# as horizontal curve-text all over the map: each is a georeferenced depth
# sample, independent of the page georef. Blue contour labels ("-2010") are
# collected too where they happen to be horizontal.

def stage3a_samples():
    from PIL import Image
    import tesserocr

    with open(os.path.join(DATA, 'georef.json')) as f:
        g = json.load(f)
    ga, gb, gc, gd = (g['affine'][k] for k in 'abcd')

    page = open_page()
    zoom = 300 / 72
    x0f, y0f, x1f, y1f = FRAME
    tile = 420          # pts
    overlap = 60        # pts — labels are ≤ ~55 pts wide

    api = tesserocr.PyTessBaseAPI(path=TESSDATA, psm=tesserocr.PSM.SPARSE_TEXT)
    api.SetVariable('tessedit_char_whitelist', '0123456789TNELDAm=-.')

    lines = []  # {text, cx, cy} in page coords
    xs = np.arange(x0f, x1f, tile - overlap)
    ys = np.arange(y0f, y1f, tile - overlap)
    for tx in xs:
        for ty in ys:
            clip = fitz.Rect(tx, ty, min(tx + tile, x1f), min(ty + tile, y1f))
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            r = img[:, :, 0].astype(int)
            gg = img[:, :, 1].astype(int)
            bb = img[:, :, 2].astype(int)
            # red label text only — kills contour/anchor-line interference
            mask = (r > 140) & (gg < 110) & (bb < 110)
            if mask.sum() < 40:  # no red text in this tile
                continue
            binimg = np.where(mask, 0, 255).astype(np.uint8)
            api.SetImage(Image.fromarray(binimg))
            api.Recognize()
            it = api.GetIterator()
            if it is None:
                continue
            level = tesserocr.RIL.TEXTLINE
            while True:
                try:
                    text = it.GetUTF8Text(level)
                except RuntimeError:
                    break
                if text and text.strip():
                    bx0, by0, bx1, by1 = it.BoundingBox(level)
                    lines.append({
                        'text': text.strip(),
                        'cx': clip.x0 + (bx0 + bx1) / 2 / zoom,
                        'cy': clip.y0 + (by0 + by1) / 2 / zoom,
                    })
                if not it.Next(level):
                    break
    api.End()
    print(f'OCR lines: {len(lines)}')

    # parse into typed entries; prefixes are often mangled, so fall back to
    # digit-count heuristics (N=7 digits, E=6 digits, LDA=3-4 digits)
    ents = []
    for ln in lines:
        t = ln['text'].replace(' ', '')
        m = re.search(r'N=?(\d{7})m?\b', t) or re.fullmatch(r'=?(\d{7})m?', t)
        if m and len(m.group(1)) == 7:
            ents.append(('N', int(m.group(1)), ln['cx'], ln['cy']))
            continue
        m = re.search(r'E=?(\d{6})m?\b', t) or re.fullmatch(r'=?(\d{6})m?', t)
        if m and len(m.group(1)) == 6:
            ents.append(('E', int(m.group(1)), ln['cx'], ln['cy']))
            continue
        m = re.search(r'(?:LDA|DA|A)=?-?(\d{3,4})m?\b', t)
        if m:
            ents.append(('LDA', int(m.group(1)), ln['cx'], ln['cy']))

    # cluster: one label block = N, E, LDA stacked vertically ~11 pts apart
    samples = []
    used = [False] * len(ents)
    for i, (k1, v1, x1, y1) in enumerate(ents):
        if used[i] or k1 != 'N':
            continue
        block = {'N': (v1, x1, y1)}
        for j, (k2, v2, x2, y2) in enumerate(ents):
            if used[j] or j == i or k2 in block:
                continue
            if abs(x2 - x1) < 60 and 0 < (y2 - y1) < 40:
                block[k2] = (v2, x2, y2)
        if 'E' in block and 'LDA' in block:
            used[i] = True
            n, e, lda = block['N'][0], block['E'][0], block['LDA'][0]
            # plausibility + georef cross-check: the label sits right next to
            # its anchor, so OCR'd UTM must match the label's page position
            # within ~800 m — this catches single-digit OCR errors.
            exp_e = ga * x1 + gb
            exp_n = gc * y1 + gd
            if not (740000 < e < 775000 and 7260000 < n < 7300000):
                continue
            if abs(e - exp_e) > 800 or abs(n - exp_n) > 800:
                continue
            if not (1200 < lda < 2600):
                continue
            samples.append({'E': e, 'N': n, 'depth': lda})

    # dedupe (overlapping tiles read the same label 2-4x)
    seen = set()
    uniq = []
    for s in samples:
        key = (s['E'], s['N'])
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    print(f'depth samples: {len(uniq)}')

    with open(os.path.join(DATA, 'depth_samples.json'), 'w') as f:
        json.dump(uniq, f, indent=1)

    # debug: samples over the isobaths
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(os.path.join(DATA, 'georef.json')) as f:
        g = json.load(f)
    a, b, c, d = g['affine']['a'], g['affine']['b'], g['affine']['c'], g['affine']['d']
    with open(os.path.join(DATA, 'isobaths_raw.json')) as f:
        iso = json.load(f)['polylines']
    fig, ax = plt.subplots(figsize=(16, 11))
    for p in iso:
        arr = np.asarray(p)
        ax.plot(arr[:, 0], arr[:, 1], lw=0.3, color='steelblue')
    sx = [(s['E'] - b) / a for s in uniq]
    sy = [(s['N'] - d) / c for s in uniq]
    sc = ax.scatter(sx, sy, c=[s['depth'] for s in uniq], cmap='viridis', s=14, zorder=5)
    fig.colorbar(sc, ax=ax, label='LDA (m)')
    ax.set_xlim(0, 2000); ax.set_ylim(1684, 0); ax.set_aspect('equal')
    ax.set_title(f'{len(uniq)} OCR depth samples')
    fig.savefig(os.path.join(DEBUG, '03a_samples.png'), dpi=110, bbox_inches='tight')
    print('debug/03a_samples.png saved')


# ── stage 3b: contour label OCR (vector-guided) ─────────────────────────────
# Contour depth labels are blue FILL paths (glyph outlines). Cluster them,
# estimate the text angle by PCA of the outline points, render a tight blue-
# masked crop, and OCR at candidate angles with strict validation.

def _isobath_pixel_mask(r, g, b):
    """Pixels próximos de ISOBATH_COLOR (glifos de rótulo são preenchidos
    nessa cor) — era um limiar fixo calibrado só pro azul original; agora
    deriva da cor configurada pra esse mapa."""
    tr, tg, tb = (round(c * 255) for c in ISOBATH_COLOR)
    dist2 = (r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2
    return dist2 < 80 ** 2


def stage3b_contour_labels():
    from PIL import Image
    import tesserocr

    page = open_page()
    api = tesserocr.PyTessBaseAPI(path=TESSDATA, psm=tesserocr.PSM.SINGLE_LINE)
    api.SetVariable('tessedit_char_whitelist', '0123456789-')

    paths = []
    for d in page.get_drawings():
        f = d.get('fill')
        if f is None or tuple(round(v, 2) for v in f) != ISOBATH_COLOR:
            continue
        pts = []
        for it in d['items']:
            for obj in it[1:]:
                if hasattr(obj, 'x'):
                    pts.append((obj.x, obj.y))
                elif hasattr(obj, 'x0'):
                    pts.extend([(obj.x0, obj.y0), (obj.x1, obj.y1)])
        paths.append((d['rect'], np.asarray(pts)))

    # union-find: nearby glyph paths belong to one label
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a, b = paths[i][0], paths[j][0]
            dx = max(0, max(a.x0, b.x0) - min(a.x1, b.x1))
            dy = max(0, max(a.y0, b.y0) - min(a.y1, b.y1))
            if dx < 6 and dy < 6:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(len(paths)):
        groups.setdefault(find(i), []).append(i)

    zoom = 600 / 72
    results = []
    for idxs in groups.values():
        allpts = np.vstack([paths[i][1] for i in idxs])
        x0 = min(paths[i][0].x0 for i in idxs)
        x1 = max(paths[i][0].x1 for i in idxs)
        y0 = min(paths[i][0].y0 for i in idxs)
        y1 = max(paths[i][0].y1 for i in idxs)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        u = allpts - allpts.mean(axis=0)
        evals, evecs = np.linalg.eigh(u.T @ u)
        main = evecs[:, np.argmax(evals)]
        theta = float(np.degrees(np.arctan2(main[1], main[0])))
        clip = fitz.Rect(x0 - 3, y0 - 3, x1 + 3, y1 + 3)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        r = img[:, :, 0].astype(int)
        g = img[:, :, 1].astype(int)
        b = img[:, :, 2].astype(int)
        mask = _isobath_pixel_mask(r, g, b)
        pil = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8))
        angles = [theta, theta + 180, theta - 12, theta + 12, theta + 168, theta + 192]
        angles += list(range(0, 360, 15))
        depth = None
        for ang in angles:
            api.SetImage(pil.rotate(ang, expand=True, fillcolor=255))
            t = api.GetUTF8Text().strip().replace(' ', '')
            m = re.fullmatch(r'-?(\d{3,4})', t)
            if m:
                v = int(m.group(1))
                if 1300 <= v <= 2600 and v % 10 == 0:
                    depth = v
                    break
        results.append({'x': round(cx, 2), 'y': round(cy, 2), 'depth': depth})
    api.End()

    n_ok = sum(1 for r in results if r['depth'])
    print(f'contour labels: {n_ok}/{len(results)} parsed')
    with open(os.path.join(DATA, 'contour_labels.json'), 'w') as f:
        json.dump(results, f, indent=1)


# ── stage 3c: assign depths to contours + propagate ─────────────────────────

def stage3c_assign():
    from shapely.geometry import LineString, Point
    from shapely.strtree import STRtree

    with open(os.path.join(DATA, 'isobaths_raw.json')) as f:
        polys = [np.asarray(p) for p in json.load(f)['polylines']]
    with open(os.path.join(DATA, 'contour_labels.json')) as f:
        labels = [l for l in json.load(f) if l['depth']]
    with open(os.path.join(DATA, 'georef.json')) as f:
        g = json.load(f)
    ga, gb, gc, gd = (g['affine'][k] for k in 'abcd')
    with open(os.path.join(DATA, 'depth_samples.json')) as f:
        red = json.load(f)

    lines = [LineString(p) for p in polys]
    tree = STRtree(lines)

    # 1. direct label → nearest contour (labels sit on their line)
    votes: dict[int, list] = {}
    for lab in labels:
        pt = Point(lab['x'], lab['y'])
        idxs = tree.query(pt.buffer(12))
        if len(idxs) == 0:
            continue
        best = min(idxs, key=lambda i: lines[int(i)].distance(pt))
        d = lines[int(best)].distance(pt)
        if d < 12:
            votes.setdefault(int(best), []).append(lab['depth'])

    depth: dict[int, float] = {}
    conflicts = []
    for i, vs in votes.items():
        med = float(np.median(vs))
        if any(abs(v - med) > 5 for v in vs):
            conflicts.append((i, vs))
            # majority value
            vals, counts = np.unique(vs, return_counts=True)
            med = float(vals[np.argmax(counts)])
        depth[i] = med
    print(f'directly labeled: {len(depth)}/{len(polys)} contours; conflicts: {conflicts}')

    # 2. propagate to unlabeled contours: nearest labeled neighbor by
    # perpendicular probing from several midpoints, step = ±10 m decided by
    # red LDA samples (local depth gradient direction).
    red_pts = np.array([[(s['E'] - gb) / ga, (s['N'] - gd) / gc] for s in red])
    red_d = np.array([s['depth'] for s in red])

    def sample_depth_idw(x, y, k=4):
        dd = np.hypot(red_pts[:, 0] - x, red_pts[:, 1] - y)
        idx = np.argsort(dd)[:k]
        w = 1 / np.maximum(dd[idx], 1)
        return float((red_d[idx] * w).sum() / w.sum())

    unlabeled = [i for i in range(len(polys)) if i not in depth]
    for _round in range(4):
        progressed = False
        for i in list(unlabeled):
            if i in depth:
                continue
            line = lines[i]
            # probe perpendicular from a few points along the line
            neigh_votes = []
            L = line.length
            for frac in (0.2, 0.4, 0.6, 0.8):
                p0 = line.interpolate(L * frac)
                p1 = line.interpolate(min(L, L * frac + 2))
                tx, ty = p1.x - p0.x, p1.y - p0.y
                norm = math.hypot(tx, ty) or 1
                nx, ny = -ty / norm, tx / norm
                for sgn in (+1, -1):
                    for reach in (6, 12, 20, 30):
                        q = Point(p0.x + sgn * nx * reach, p0.y + sgn * ny * reach)
                        cand = [int(j) for j in tree.query(q.buffer(2))]
                        cand = [j for j in cand if j != i and j in depth
                                and lines[j].distance(q) < 2]
                        if cand:
                            j = cand[0]
                            # deeper side? decide by IDW of red samples
                            here = sample_depth_idw(p0.x, p0.y)
                            there_pt = lines[j].interpolate(
                                lines[j].project(q))
                            there = sample_depth_idw(there_pt.x, there_pt.y)
                            step = 10 if here > there else -10
                            neigh_votes.append(depth[j] + step)
                            break
            if neigh_votes:
                vals, counts = np.unique(neigh_votes, return_counts=True)
                depth[i] = float(vals[np.argmax(counts)])
                progressed = True
        if not progressed:
            break
    still = [i for i in range(len(polys)) if i not in depth]
    print(f'after propagation: {len(depth)} labeled, {len(still)} unresolved '
          f'(lengths: {[round(lines[i].length) for i in still][:10]})')

    # 3. write GeoJSON in UTM (page→UTM affine)
    feats = []
    for i, p in enumerate(polys):
        e = ga * p[:, 0] + gb
        n = gc * p[:, 1] + gd
        coords = [[round(float(a), 1), round(float(b), 1)] for a, b in zip(e, n)]
        feats.append({
            'type': 'Feature',
            'properties': {
                'depth': depth.get(i),
                'source': ('label' if i in votes else
                           'propagated' if i in depth else 'unresolved'),
            },
            'geometry': {'type': 'LineString', 'coordinates': coords},
        })
    with open(os.path.join(DATA, 'isobaths.geojson'), 'w') as f:
        json.dump({'type': 'FeatureCollection',
                   'crs_note': 'EPSG:31983 (UTM); convert to WGS84 at API layer',
                   'features': feats}, f)
    print('data/isobaths.geojson saved')

    # debug: contours colored by depth
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 11))
    ds = [depth.get(i) for i in range(len(polys))]
    valid = [d for d in ds if d is not None]
    vmin, vmax = min(valid), max(valid)
    cmap = plt.get_cmap('viridis')
    for i, p in enumerate(polys):
        if ds[i] is None:
            ax.plot(p[:, 0], p[:, 1], color='red', lw=1.2)
        else:
            ax.plot(p[:, 0], p[:, 1],
                    color=cmap((ds[i] - vmin) / (vmax - vmin)), lw=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=ax, label='depth (m)')
    ax.set_xlim(0, 2000)
    ax.set_ylim(1684, 0)
    ax.set_aspect('equal')
    ax.set_title('isobath depths (red = unresolved)')
    fig.savefig(os.path.join(DEBUG, '03c_depths.png'), dpi=110,
                bbox_inches='tight')
    print('debug/03c_depths.png saved')


# ── stage 4: continuous depth grid ──────────────────────────────────────────

def stage4_grid():
    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt

    with open(os.path.join(DATA, 'isobaths.geojson')) as f:
        fc = json.load(f)

    pts = []
    vals = []
    for feat in fc['features']:
        d = feat['properties']['depth']
        if d is None:
            continue
        coords = feat['geometry']['coordinates']
        # subsample: contour vertices are ~2-6 m apart; every 4th is plenty
        for c in coords[::4]:
            pts.append(c)
            vals.append(d)
    pts = np.asarray(pts)
    vals = np.asarray(vals, dtype=np.float32)
    print(f'interpolation input: {len(pts)} points')

    cell = 20.0  # meters
    e0, e1 = pts[:, 0].min(), pts[:, 0].max()
    n0, n1 = pts[:, 1].min(), pts[:, 1].max()
    ge = np.arange(e0, e1 + cell, cell)
    gn = np.arange(n0, n1 + cell, cell)
    E, N = np.meshgrid(ge, gn)
    print(f'grid: {len(gn)} x {len(ge)} cells @ {cell} m')

    Z = griddata(pts, vals, (E, N), method='linear').astype(np.float32)
    # fill outside-hull cells with nearest value (map corners beyond contours)
    nanmask = np.isnan(Z)
    if nanmask.any():
        idx = distance_transform_edt(nanmask, return_distances=False,
                                     return_indices=True)
        Z = Z[tuple(idx)]
    print(f'depth range: {np.nanmin(Z):.0f} … {np.nanmax(Z):.0f} m')

    np.savez_compressed(
        os.path.join(DATA, 'depth_grid.npz'),
        depth=Z, e0=ge[0], n0=gn[0], cell=cell,
        crs='EPSG:31983')

    # validation against the red LDA samples + P-77
    with open(os.path.join(DATA, 'depth_samples.json')) as f:
        red = json.load(f)

    def read_depth(e, n):
        i = int(round((n - gn[0]) / cell))
        j = int(round((e - ge[0]) / cell))
        if 0 <= i < Z.shape[0] and 0 <= j < Z.shape[1]:
            return float(Z[i, j])
        return None

    errs = []
    for s in red:
        z = read_depth(s['E'], s['N'])
        if z is not None:
            errs.append(z - s['depth'])
    errs = np.asarray(errs)
    print(f'red-sample residuals: n={len(errs)} mean={errs.mean():+.1f} '
          f'median={np.median(errs):+.1f} p95(|e|)={np.percentile(abs(errs), 95):.1f} m')

    # Poço de referência pra validar a grade — configurável por mapa
    # (REF_WELL_E/N/LDA), com o P-77 do primeiro mapa calibrado como padrão.
    # Se o poço cair fora da grade desse mapa (área diferente), não trava —
    # os resíduos das amostras vermelhas acima já servem de validação.
    ref_e = float(os.environ.get('REF_WELL_E', 761977))
    ref_n = float(os.environ.get('REF_WELL_N', 7272834))
    ref_lda_env = os.environ.get('REF_WELL_LDA', '1980')
    ref_depth = read_depth(ref_e, ref_n)
    if ref_depth is None:
        print(f'poço de referência (E={ref_e:.0f}, N={ref_n:.0f}) fora da grade '
              f'desse mapa — checagem pulada (use os resíduos das amostras '
              f'vermelhas acima, ou informe REF_WELL_E/N/LDA de um poço deste mapa)')
    else:
        ref_lda = float(ref_lda_env)
        print(f'poço de referência check: grid={ref_depth:.1f} m, '
              f'anotação={ref_lda:.0f} m, delta={ref_depth - ref_lda:+.1f} m')

    # debug hillshade-ish image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 11))
    im = ax.imshow(Z, origin='lower', cmap='viridis',
                   extent=(ge[0], ge[-1], gn[0], gn[-1]))
    fig.colorbar(im, ax=ax, label='depth (m)')
    ax.plot(761977, 7272834, 'r*', ms=14)
    ax.annotate('P-77', (761977, 7272834), color='red',
                xytext=(6, 6), textcoords='offset points')
    ax.set_title('interpolated depth grid')
    ax.set_aspect('equal')
    fig.savefig(os.path.join(DEBUG, '04_depth_grid.png'), dpi=110,
                bbox_inches='tight')
    print('debug/04_depth_grid.png saved')


# ── stage 5: obstacles ──────────────────────────────────────────────────────
# Stroke color → category, mapped automatically from the legend swatches
# (see debug notes). `projeto` = planned-only feature (not on the seabed yet);
# excluded from the forbidden mask by default at analysis time.

OBSTACLE_CATEGORIES = {
    (1.00, 0.00, 0.00): ('duto_rigido_gasoduto', False),
    (0.99, 0.67, 0.06): ('duto_rigido_oleoduto', False),
    (1.00, 0.50, 0.00): ('duto_rigido_oleoduto_nao_mapeado', False),
    (0.65, 0.81, 0.89): ('duto_rigido_outros', False),
    (0.59, 0.75, 0.72): ('duto_rigido_outros_nao_mapeados', False),
    (0.86, 0.00, 0.02): ('duto_rigido_spool', False),
    (0.20, 0.63, 0.17): ('duto_flexivel', False),
    (0.53, 0.53, 0.53): ('duto_flexivel_nao_mapeado_ou_riser', False),
    (0.74, 0.74, 0.74): ('mangueira', False),
    (0.28, 0.48, 0.71): ('cabo_fibra_otica', False),
    (1.00, 0.00, 0.80): ('amarra', False),
    (1.00, 0.00, 0.75): ('amarra_poliester', False),   # split vs escape fans below
    (0.89, 0.10, 0.11): ('ancoragem_de_duto_ou_anm', False),
    (0.71, 0.13, 0.94): ('monoboia', False),
    (0.72, 0.72, 0.72): ('obstaculo_linear', False),
    (0.00, 0.50, 1.00): ('duto_abandonado', False),
    (0.22, 0.83, 0.00): ('projeto_duto_flexivel', True),
    (0.82, 0.00, 0.01): ('projeto_dutos_rigidos', True),
    (0.33, 0.91, 0.26): ('projeto_manifold_plem', True),
    (0.32, 0.32, 1.00): ('projeto_cabo_fibra', True),
    (0.92, 0.56, 0.74): ('outros_projetos', True),
}


def _straightness(poly):
    """Max perpendicular deviation from the chord (pts)."""
    a = np.asarray(poly)
    p0, p1 = a[0], a[-1]
    chord = p1 - p0
    L = np.hypot(*chord)
    if L < 1e-6:
        return 0.0
    n = np.array([-chord[1], chord[0]]) / L
    return float(np.abs((a - p0) @ n).max())


def stage5_obstacles():
    page = open_page()
    with open(os.path.join(DATA, 'georef.json')) as f:
        g = json.load(f)
    ga, gb, gc, gd = (g['affine'][k] for k in 'abcd')

    feats = []
    counts: dict = {}
    for d in page.get_drawings():
        color = d.get('color')
        if color is None:
            continue
        key = tuple(round(v, 2) for v in color)
        if key not in OBSTACLE_CATEGORIES:
            continue
        cat, projeto = OBSTACLE_CATEGORIES[key]
        for poly in _split_into_polylines(d['items']):
            for piece in _clip_to_frame(poly):
                if _polyline_length(piece) < 1.5:
                    continue
                # escape-route fans share the AMARRA POLIÉSTER color but are
                # long straight rays — reclassify so they never become
                # obstacles.
                pcat = cat
                if cat == 'amarra_poliester':
                    if _polyline_length(piece) > 120 and _straightness(piece) < 2.0:
                        pcat = 'rota_fuga_desenhada'
                e = [round(ga * x + gb, 1) for x, y in piece]
                n = [round(gc * y + gd, 1) for x, y in piece]
                feats.append({
                    'type': 'Feature',
                    'properties': {'category': pcat, 'projeto': projeto},
                    'geometry': {'type': 'LineString',
                                 'coordinates': [[a, b] for a, b in zip(e, n)]},
                })
                counts[pcat] = counts.get(pcat, 0) + 1

    print('obstacle features by category:')
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f'  {cat}: {n}')
    with open(os.path.join(DATA, 'obstacles.geojson'), 'w') as f:
        json.dump({'type': 'FeatureCollection',
                   'crs_note': 'EPSG:31983 (UTM)',
                   'features': feats}, f)
    print(f'data/obstacles.geojson saved ({len(feats)} features)')

    # debug plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20')
    cats = sorted(counts)
    color_of = {c: cmap(i % 20) for i, c in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(16, 11))
    for f_ in feats:
        arr = np.asarray(f_['geometry']['coordinates'])
        ax.plot(arr[:, 0], arr[:, 1], lw=0.5,
                color=color_of[f_['properties']['category']])
    handles = [plt.Line2D([0], [0], color=color_of[c], lw=2, label=f'{c} ({counts[c]})')
               for c in cats]
    ax.legend(handles=handles, fontsize=6, loc='upper right')
    ax.set_aspect('equal')
    ax.set_title('obstacles by category (UTM)')
    fig.savefig(os.path.join(DEBUG, '05_obstacles.png'), dpi=110,
                bbox_inches='tight')
    print('debug/05_obstacles.png saved')


# ── stage 6: wells ──────────────────────────────────────────────────────────
# Magenta annotation blocks ("P-77 / N=7272834m / E=761977m / LDA=-1980m")
# are OCR'd like the red ones. Target wells without an annotation on the map
# (e.g. the title wells) can be added in data/wells_manual.json:
#   [{"id": "9-BUZ-43-RJS", "name": "...", "E": ..., "N": ..., "lda": ...}]

def stage6_wells():
    from PIL import Image
    import tesserocr

    page = open_page()
    zoom = 300 / 72
    x0f, y0f, x1f, y1f = FRAME
    tile, overlap = 420, 80

    api = tesserocr.PyTessBaseAPI(path=TESSDATA, psm=tesserocr.PSM.SPARSE_TEXT)
    api.SetVariable('tessedit_char_whitelist',
                    '0123456789PFSONELDAm=-. ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    lines = []
    for tx in np.arange(x0f, x1f, tile - overlap):
        for ty in np.arange(y0f, y1f, tile - overlap):
            clip = fitz.Rect(tx, ty, min(tx + tile, x1f), min(ty + tile, y1f))
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            r = img[:, :, 0].astype(int)
            g = img[:, :, 1].astype(int)
            b = img[:, :, 2].astype(int)
            # magenta text: high R, low G, high B
            mask = (r > 150) & (g < 120) & (b > 150)
            if mask.sum() < 60:
                continue
            api.SetImage(Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)))
            api.Recognize()
            it = api.GetIterator()
            if it is None:
                continue
            level = tesserocr.RIL.TEXTLINE
            while True:
                try:
                    text = it.GetUTF8Text(level)
                except RuntimeError:
                    break
                if text and text.strip():
                    bx0, by0, bx1, by1 = it.BoundingBox(level)
                    lines.append({'text': text.strip(),
                                  'cx': clip.x0 + (bx0 + bx1) / 2 / zoom,
                                  'cy': clip.y0 + (by0 + by1) / 2 / zoom})
                if not it.Next(level):
                    break
    api.End()

    # group into blocks: name line + N/E/LDA lines below it
    ents = []
    for ln in lines:
        t = ln['text'].replace(' ', '')
        m = re.search(r'N=?(\d{7})m?\b', t)
        if m:
            ents.append(('N', int(m.group(1)), ln))
            continue
        m = re.search(r'E=?(\d{6})m?\b', t)
        if m:
            ents.append(('E', int(m.group(1)), ln))
            continue
        m = re.search(r'LDA=?-?(\d{3,4})m?\b', t)
        if m:
            ents.append(('LDA', int(m.group(1)), ln))
            continue
        name = ln['text'].strip()
        # strict well/unit name patterns: P-77, NS-38, FPSO …, 9-BUZ-43-RJS
        if re.match(r'^(P-\d+|NS-\d+|FPSO[A-Z ]+|\d-[A-Z]{2,4}-\d+[A-Z0-9-]*)$', name):
            ents.append(('NAME', name, ln))

    wells = []
    for k, v, ln in ents:
        if k != 'NAME':
            continue
        block = {}
        for k2, v2, ln2 in ents:
            if k2 in ('N', 'E', 'LDA') and k2 not in block \
               and abs(ln2['cx'] - ln['cx']) < 90 and 0 < ln2['cy'] - ln['cy'] < 60:
                block[k2] = v2
        if 'N' in block and 'E' in block:
            wells.append({
                'id': v.replace(' ', '-'),
                'name': v,
                'E': block['E'],
                'N': block['N'],
                'lda': block.get('LDA'),
                'source': 'ocr',
            })

    # dedupe by id and by coordinates (several OCR name-lines can latch onto
    # the same N/E block)
    seen = set()
    wells = [w for w in wells
             if not (w['id'] in seen or seen.add(w['id']))]
    seen_c = set()
    wells = [w for w in wells
             if not ((w['E'], w['N']) in seen_c or seen_c.add((w['E'], w['N'])))]

    manual_path = os.path.join(DATA, 'wells_manual.json')
    if os.path.exists(manual_path):
        with open(manual_path) as f:
            for w in json.load(f):
                w['source'] = 'manual'
                wells = [x for x in wells if x['id'] != w['id']] + [w]

    # fill missing LDA from the depth grid
    grid = np.load(os.path.join(DATA, 'depth_grid.npz'))
    Z, e0, n0, cell = grid['depth'], float(grid['e0']), float(grid['n0']), float(grid['cell'])
    for w in wells:
        if not w.get('lda') and w.get('N') is not None and w.get('E') is not None:
            i = int(round((w['N'] - n0) / cell))
            j = int(round((w['E'] - e0) / cell))
            if 0 <= i < Z.shape[0] and 0 <= j < Z.shape[1]:
                w['lda'] = round(float(Z[i, j]))
                w['lda_source'] = 'grid'

    print(f'wells: {len(wells)}')
    for w in wells:
        print(' ', w)
    with open(os.path.join(DATA, 'wells.json'), 'w') as f:
        json.dump(wells, f, indent=1)


STAGES = {'1': stage1_georef, '2': stage2_isobaths, '3': stage3a_samples,
          '4': stage3b_contour_labels, '5': stage3c_assign, '6': stage4_grid,
          '7': stage5_obstacles, '8': stage6_wells}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all')
    args = ap.parse_args()
    keys = sorted(STAGES) if args.stage == 'all' else [args.stage]
    for k in keys:
        print(f'\n=== stage {k} ===')
        STAGES[k]()
