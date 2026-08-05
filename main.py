"""API FastAPI da análise de rota de fuga.

  uvicorn main:app --reload --port 8000

O trabalho pesado fica no preprocess (uma vez por PDF); /analyze responde <1 s.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyproj import Transformer

from analysis import EscapeAnalyzer

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

# chave compartilhada com o server.js — só rotas que ESCREVEM dado exigem;
# GET /wells, /map/layers e POST /analyze continuam públicas (mesmo padrão
# de sempre desta API: sem estado de usuário, o server.js é quem barra quem
# pode chegar até aqui como admin).
INTERNAL_KEY = os.environ.get('ROTAFUGA_PROCESS_KEY')


def require_internal_key(x_internal_key: str | None = Header(default=None)):
    if not INTERNAL_KEY or x_internal_key != INTERNAL_KEY:
        raise HTTPException(401, 'chave interna inválida ou não configurada')


app = FastAPI(title='Rota de Fuga — análise de mapa de restrição')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],   # dev; restringir em produção
    allow_methods=['*'],
    allow_headers=['*'],
)

analyzer = EscapeAnalyzer()
_to_wgs84 = Transformer.from_crs('EPSG:31983', 'EPSG:4326', always_xy=True)
_from_wgs84 = Transformer.from_crs('EPSG:4326', 'EPSG:31983', always_xy=True)


class WellOverrideBody(BaseModel):
    id: str
    name: Optional[str] = None
    x: Optional[float] = None     # UTM E (SIRGAS2000 23S)
    y: Optional[float] = None     # UTM N
    lon: Optional[float] = None   # alternativa em WGS84
    lat: Optional[float] = None
    lda: Optional[float] = None


@app.post('/wells/override', dependencies=[Depends(require_internal_key)])
def set_well_override(body: WellOverrideBody):
    e = n = None
    if body.x is not None and body.y is not None:
        e, n = body.x, body.y
    elif body.lon is not None and body.lat is not None:
        e, n = _from_wgs84.transform(body.lon, body.lat)
    analyzer.set_well_override(body.id, {
        'name': body.name, 'E': e, 'N': n, 'lda': body.lda, 'source': 'manual',
    })
    return {'ok': True}


@app.get('/wells/incomplete')
def wells_incomplete():
    """Poços que o preprocess achou mas sem coordenada (tipicamente os
    poços-alvo do título, sem anotação OCR no mapa) — o que falta preencher
    manualmente via /wells/override."""
    return [{'id': w['id'], 'name': w.get('name'), 'note': w.get('_nota')}
            for w in analyzer.wells if w.get('E') is None]


@app.post('/reload', dependencies=[Depends(require_internal_key)])
def reload_data():
    """Recarrega data/ do disco — chamado pelo droplet de processamento
    depois de publicar um mapa novo."""
    analyzer.reload()
    return {'ok': True}


class AnalyzeBody(BaseModel):
    map_id: str = 'BUZ43_BUZ92_00'
    well_id: Optional[str] = None
    x: Optional[float] = None     # UTM E (SIRGAS2000 23S)
    y: Optional[float] = None     # UTM N
    lon: Optional[float] = None   # alternativa em WGS84
    lat: Optional[float] = None
    radius_m: float = 4000
    margin_m: float = 50
    obstacle_buffer_m: Optional[dict[str, float]] = None
    include_projeto: bool = False


@app.get('/wells')
def wells():
    out = []
    for w in analyzer.wells:
        if w.get('E') is None:
            continue
        lon, lat = _to_wgs84.transform(w['E'], w['N'])
        out.append({'id': w['id'], 'name': w['name'],
                    'x': w['E'], 'y': w['N'],
                    'lon': round(lon, 6), 'lat': round(lat, 6),
                    'lda': w.get('lda'), 'source': w.get('source')})
    return out


def _linestring_fc_wgs84(path: str, max_feats: int | None = None):
    with open(path) as f:
        fc = json.load(f)
    feats = fc['features'][:max_feats] if max_feats else fc['features']
    out = []
    for feat in feats:
        coords = feat['geometry']['coordinates']
        ll = [list(_to_wgs84.transform(x, y)) for x, y in coords[::2] or coords]
        out.append({
            'type': 'Feature',
            'properties': feat['properties'],
            'geometry': {'type': 'LineString',
                         'coordinates': [[round(lo, 6), round(la, 6)]
                                         for lo, la in ll]},
        })
    return {'type': 'FeatureCollection', 'features': out}


@app.get('/map/layers')
def map_layers():
    return {
        'crs': 'WGS84 (convertido de EPSG:31983)',
        'isobaths': _linestring_fc_wgs84(os.path.join(DATA, 'isobaths.geojson')),
        'obstacles': _linestring_fc_wgs84(os.path.join(DATA, 'obstacles.geojson')),
    }


@app.post('/analyze')
def analyze(body: AnalyzeBody):
    well_meta = {}
    if body.well_id:
        w = analyzer.well_by_id(body.well_id)
        if w is None:
            raise HTTPException(404, f'poço {body.well_id!r} não encontrado')
        if w.get('E') is None:
            raise HTTPException(422, f'poço {body.well_id!r} sem coordenadas — '
                                     'preencha data/wells_manual.json e rode '
                                     'preprocess --stage 8')
        e, n, lda = w['E'], w['N'], w.get('lda')
        well_meta = {'id': w['id'], 'name': w['name']}
    elif body.x is not None and body.y is not None:
        e, n, lda = body.x, body.y, None
    elif body.lon is not None and body.lat is not None:
        e, n = _from_wgs84.transform(body.lon, body.lat)
        lda = None
    else:
        raise HTTPException(422, 'informe well_id, x/y (UTM) ou lon/lat')

    try:
        return analyzer.analyze(
            e, n, lda=lda, radius_m=body.radius_m, margin_m=body.margin_m,
            obstacle_buffers=body.obstacle_buffer_m,
            include_projeto=body.include_projeto, well_meta=well_meta)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
