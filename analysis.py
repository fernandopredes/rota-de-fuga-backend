"""Análise de rota de fuga: limiar de profundidade + buffers de obstáculos +
conectividade + setores de azimute livres.

Toda a parte pesada (extração do PDF) já foi feita pelo preprocess; aqui é
só álgebra de raster sobre a grade de profundidade — precisa responder <1 s.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt, label as ndlabel
from skimage import measure
from skimage.draw import line as sk_line

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

CONTOUR_INTERVAL_M = 10.0   # intervalo das isóbatas do mapa
BATHY_UNCERTAINTY_NOTE = (
    'Precisão batimétrica ~2% da LDA (nota do próprio mapa). Medições devem '
    'ser confirmadas com a SUB/SSUB antes de uso operacional.')
TOOL_NOTE = (
    'Ferramenta de análise/planejamento — não substitui validação oficial '
    'nem análise de riser da engenharia.')

DEFAULT_BUFFERS_M = {
    # amarras sobem na coluna d'água — buffer maior que duto assentado
    'amarra': 200.0,
    'amarra_poliester': 200.0,
    'ancoragem_de_duto_ou_anm': 150.0,
    'default': 75.0,
}


class EscapeAnalyzer:
    def __init__(self):
        self._load()

    def _load(self):
        grid = np.load(os.path.join(DATA, 'depth_grid.npz'))
        self.Z = grid['depth']
        self.e0 = float(grid['e0'])
        self.n0 = float(grid['n0'])
        self.cell = float(grid['cell'])
        with open(os.path.join(DATA, 'obstacles.geojson')) as f:
            self.obstacles = json.load(f)['features']
        with open(os.path.join(DATA, 'wells.json')) as f:
            self.wells = json.load(f)
        self._apply_well_overrides()
        self.to_wgs84 = Transformer.from_crs('EPSG:31983', 'EPSG:4326',
                                             always_xy=True)

    def reload(self):
        """Recarrega tudo do disco — chamado depois que um mapa novo (ou uma
        correção de poço) foi copiado para data/."""
        self._load()

    # ── overrides de poço (preenchidos manualmente, sem reprocessar o PDF) ──
    def _overrides_path(self) -> str:
        return os.path.join(DATA, 'wells_overrides.json')

    def _load_overrides(self) -> dict:
        path = self._overrides_path()
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _apply_well_overrides(self):
        overrides = self._load_overrides()
        by_id = {w['id']: w for w in self.wells}
        for well_id, patch in overrides.items():
            if well_id in by_id:
                by_id[well_id].update(patch)
            else:
                new_well = {'id': well_id, 'name': patch.get('name', well_id)}
                new_well.update(patch)
                self.wells.append(new_well)

    def set_well_override(self, well_id: str, patch: dict):
        """Grava/atualiza E, N e/ou lda de um poço à mão (ex.: poço-alvo sem
        anotação OCR no mapa) sem precisar rodar o preprocess de novo."""
        clean = {k: v for k, v in patch.items() if v is not None}
        overrides = self._load_overrides()
        overrides.setdefault(well_id, {}).update(clean)
        with open(self._overrides_path(), 'w') as f:
            json.dump(overrides, f, indent=2, ensure_ascii=False)
        self._apply_well_overrides()

    # ── helpers ─────────────────────────────────────────────────────────────
    def depth_at(self, e: float, n: float) -> float | None:
        i = int(round((n - self.n0) / self.cell))
        j = int(round((e - self.e0) / self.cell))
        if 0 <= i < self.Z.shape[0] and 0 <= j < self.Z.shape[1]:
            return float(self.Z[i, j])
        return None

    def well_by_id(self, well_id: str) -> dict | None:
        for w in self.wells:
            if w['id'] == well_id:
                return w
        return None

    # ── core ────────────────────────────────────────────────────────────────
    def analyze(self, e: float, n: float, lda: float | None = None,
                radius_m: float = 4000.0, margin_m: float = 50.0,
                obstacle_buffers: dict | None = None,
                include_projeto: bool = False,
                well_meta: dict | None = None) -> dict:
        buffers = dict(DEFAULT_BUFFERS_M)
        if obstacle_buffers:
            buffers.update(obstacle_buffers)

        lda_grid = self.depth_at(e, n)
        if lda is None:
            lda = lda_grid
        if lda is None:
            raise ValueError('posição fora da grade de profundidade')
        threshold = lda + margin_m

        # window the grid to the radius
        i0 = max(0, int((n - radius_m - self.n0) / self.cell))
        i1 = min(self.Z.shape[0], int((n + radius_m - self.n0) / self.cell) + 1)
        j0 = max(0, int((e - radius_m - self.e0) / self.cell))
        j1 = min(self.Z.shape[1], int((e + radius_m - self.e0) / self.cell) + 1)
        Zw = self.Z[i0:i1, j0:j1]
        H, W = Zw.shape

        # cell-center coordinate grids
        ee = self.e0 + (np.arange(j0, j1) + 0.0) * self.cell
        nn = self.n0 + (np.arange(i0, i1) + 0.0) * self.cell
        EE, NN = np.meshgrid(ee, nn)
        dist_from_well = np.hypot(EE - e, NN - n)
        inside = dist_from_well <= radius_m

        # obstacle mask per category via rasterized lines + EDT
        cat_masks: dict[str, np.ndarray] = {}
        for feat in self.obstacles:
            props = feat['properties']
            cat = props['category']
            if cat == 'rota_fuga_desenhada':
                continue  # referência visual, não é obstáculo
            if props.get('projeto') and not include_projeto:
                continue
            coords = np.asarray(feat['geometry']['coordinates'])
            # quick reject: bbox far outside the window
            if (coords[:, 0].max() < ee[0] - 500 or coords[:, 0].min() > ee[-1] + 500 or
                    coords[:, 1].max() < nn[0] - 500 or coords[:, 1].min() > nn[-1] + 500):
                continue
            mask = cat_masks.setdefault(cat, np.zeros((H, W), dtype=bool))
            jj = np.clip(((coords[:, 0] - ee[0]) / self.cell).round().astype(int), 0, W - 1)
            ii = np.clip(((coords[:, 1] - nn[0]) / self.cell).round().astype(int), 0, H - 1)
            for k in range(len(ii) - 1):
                rr, cc = sk_line(ii[k], jj[k], ii[k + 1], jj[k + 1])
                mask[rr, cc] = True

        obstacle_forbidden = np.zeros((H, W), dtype=bool)
        for cat, mask in cat_masks.items():
            if not mask.any():
                continue
            buf = buffers.get(cat, buffers['default'])
            dist = distance_transform_edt(~mask) * self.cell
            obstacle_forbidden |= dist <= buf

        # zone model:
        #   transit  — depth ≥ LDA (o riser não toca o fundo; dá para passar)
        #   safe     — depth ≥ LDA + margem (refúgio) e alcançável
        #   uncertain— faixa de ±1 isóbata em torno do limiar de refúgio
        # A conectividade é avaliada sobre o que é TRANSITÁVEL: um poço fica
        # exatamente na LDA, então exigir a margem já no primeiro metro
        # tornaria tudo inalcançável.
        passable = (Zw >= lda) & ~obstacle_forbidden
        shallow = Zw < lda
        uncertain = (Zw >= threshold) & (Zw < threshold + CONTOUR_INTERVAL_M)

        # connectivity: only what is continuously reachable from the well
        # vicinity counts (a deep basin ringed by a shallow crest is useless)
        lab, _ = ndlabel(passable)
        seed_ring = (dist_from_well <= 400) & passable
        reachable_ids = np.unique(lab[seed_ring])
        reachable_ids = reachable_ids[reachable_ids != 0]
        reachable = np.isin(lab, reachable_ids)

        zone = np.zeros((H, W), dtype=np.uint8)   # 0=outside
        zone[inside & shallow] = 1                                    # forbidden_shallow
        zone[inside & ~shallow & obstacle_forbidden] = 2              # forbidden_obstacle
        zone[inside & passable & ~reachable] = 3                      # unreachable
        zone[inside & passable & reachable & (Zw < threshold)] = 6    # transit
        zone[inside & passable & reachable & uncertain] = 4           # uncertain
        zone[inside & passable & reachable
             & (Zw >= threshold + CONTOUR_INTERVAL_M)] = 5            # safe

        # azimuth sectors: ray-march each degree while the ray stays in
        # transitable water (transit/uncertain/safe); a sector is only
        # operational if the ray actually REACHES safe water within the radius
        clearances = np.zeros(360)
        reaches_safe = np.zeros(360, dtype=bool)
        step = self.cell / 2
        for az in range(360):
            rad = math.radians(az)
            dx, dy = math.sin(rad), math.cos(rad)  # az 0 = norte, 90 = leste
            r = 0.0
            while r < radius_m:
                r += step
                pe, pn = e + dx * r, n + dy * r
                ii = int(round((pn - nn[0]) / self.cell))
                jj = int(round((pe - ee[0]) / self.cell))
                if not (0 <= ii < H and 0 <= jj < W):
                    break
                z = zone[ii, jj]
                if z not in (4, 5, 6):
                    break
                if z == 5:
                    reaches_safe[az] = True
            clearances[az] = max(0.0, r - step)

        min_clear = 500.0
        sectors = []
        az = 0
        while az < 360:
            if clearances[az] >= min_clear and reaches_safe[az]:
                start = az
                while az < 360 and clearances[az] >= min_clear and reaches_safe[az]:
                    az += 1
                sectors.append({
                    'from_az': start,
                    'to_az': az % 360,
                    'min_clearance_m': round(float(clearances[start:az].min())),
                })
            else:
                az += 1
        # merge wrap-around sector (…→360 with 0→…)
        if len(sectors) >= 2 and sectors[0]['from_az'] == 0 and sectors[-1]['to_az'] == 0:
            first, last = sectors[0], sectors.pop()
            first['from_az'] = last['from_az']
            first['min_clearance_m'] = min(first['min_clearance_m'],
                                           last['min_clearance_m'])

        zones_geojson = self._vectorize_zones(zone, ee, nn)

        return {
            'well': (well_meta or {}) | {'E': e, 'N': n, 'lda': round(lda, 1),
                                         'lda_grid': round(lda_grid, 1) if lda_grid else None},
            'threshold_depth': round(threshold, 1),
            'params': {'radius_m': radius_m, 'margin_m': margin_m,
                       'buffers_m': buffers, 'include_projeto': include_projeto,
                       'uncertain_band_m': CONTOUR_INTERVAL_M},
            'zones': zones_geojson,
            'sectors': sectors,
            'clearance_by_azimuth': [round(c) for c in clearances],
            'warnings': [BATHY_UNCERTAINTY_NOTE, TOOL_NOTE],
        }

    def _vectorize_zones(self, zone, ee, nn):
        names = {1: 'forbidden_shallow', 2: 'forbidden_obstacle',
                 3: 'unreachable', 4: 'uncertain', 5: 'safe', 6: 'transit'}
        feats = []
        for val, name in names.items():
            mask = (zone == val).astype(float)
            if mask.sum() == 0:
                continue
            padded = np.pad(mask, 1)
            for contour in measure.find_contours(padded, 0.5):
                if len(contour) < 8:
                    continue
                # padded index → UTM (contour[:,0]=row=i, [:,1]=col=j)
                utm = [(float(ee[0] + (c - 1) * (ee[1] - ee[0])),
                        float(nn[0] + (r - 1) * (nn[1] - nn[0])))
                       for r, c in contour[::3]]
                lonlat = [list(self.to_wgs84.transform(x, y)) for x, y in utm]
                feats.append({
                    'type': 'Feature',
                    'properties': {
                        'zone': name,
                        'utm_ring': [[round(x, 1), round(y, 1)] for x, y in utm],
                    },
                    'geometry': {'type': 'Polygon',
                                 'coordinates': [[[round(lo, 6), round(la, 6)]
                                                  for lo, la in lonlat]]},
                })
        return {'type': 'FeatureCollection', 'features': feats}
