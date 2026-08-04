# Rota de Fuga — backend (FastAPI)

Análise de rota de fuga para navio sonda sobre mapa de restrição da Petrobras
(PDF vetorial QGIS). Ver `PROJETO_ROTA_DE_FUGA.md` no repositório do front
para o contexto completo.

## Setup (sem sudo)

```bash
# uv (gerenciador python userspace) — já instalado em ~/.local/bin
uv venv .venv
uv pip install --python .venv/bin/python pymupdf numpy scipy shapely pyproj \
    scikit-image matplotlib fastapi uvicorn tesserocr pillow
# tessdata (OCR): tessdata/eng.traineddata (tessdata_fast) — já baixado
```

## Pipeline

```bash
# pré-processamento (1x por PDF; ~5 min no total). MAP_PDF aponta o PDF.
.venv/bin/python preprocess.py --stage 1   # georef: grid UTM + OCR das bordas
.venv/bin/python preprocess.py --stage 2   # isóbatas (polylines azuis)
.venv/bin/python preprocess.py --stage 3   # amostras LDA (rótulos vermelhos, OCR)
.venv/bin/python preprocess.py --stage 4   # rótulos de profundidade das isóbatas
.venv/bin/python preprocess.py --stage 5   # atribuição/propagação de profundidade
.venv/bin/python preprocess.py --stage 6   # grade contínua 20 m (valida P-77)
.venv/bin/python preprocess.py --stage 7   # obstáculos por cor→categoria da legenda
.venv/bin/python preprocess.py --stage 8   # poços (OCR magenta + wells_manual.json)

# API
.venv/bin/uvicorn main:app --port 8000
```

Cada estágio salva uma imagem de inspeção em `debug/`. **Confira as imagens
antes de confiar nos números.**

## Carregando um mapa de restrição novo

O sistema tem duas partes com papéis diferentes:

- **O preprocess (pesado, roda 1x por mapa)** — lê o PDF e destila ele nos
  arquivos prontos em `data/` (grade de profundidade, isóbatas rotuladas,
  obstáculos, poços). É aqui que mora o OCR e a extração vetorial.
- **A API (leve, instantânea)** — só lê esses arquivos prontos. Nunca toca
  no PDF.

Para trocar de mapa, hoje é manual, no terminal:

```bash
MAP_PDF=/caminho/do/novo_mapa.pdf .venv/bin/python preprocess.py --stage all
# conferir as imagens em debug/ (georef, profundidades, obstáculos...)
# reiniciar a API para ela reler data/:
.venv/bin/uvicorn main:app --port 8000
```

Três ressalvas:

1. **Sobrescreve o mapa anterior.** `data/` guarda um mapa por vez — não há
   ainda "vários mapas carregados" com o front escolhendo entre eles (o
   campo `map_id` da API existe no contrato, mas é decorativo por enquanto).
2. **Mesma família de mapa tende a funcionar direto.** Outro mapa de
   restrição da Petrobras no mesmo template QGIS (mesma legenda, mesmas
   cores, mesmo layout A1 1:30.000) deve passar pelos 8 estágios sem ajuste
   — as imagens de `debug/` dizem se passou.
3. **Região/layout diferente exige ajuste de código.** Constantes como a
   moldura do mapa (`FRAME` em `preprocess.py`) e as faixas de
   plausibilidade do OCR (easting/northing de Búzios, LDA 1200–2600 m) estão
   calibradas para este PDF. Um mapa de outra bacia seria rejeitado pelos
   filtros até alguém generalizar esses números — idealmente extraindo tudo
   isso para um `map_config.json` por mapa (não implementado ainda).

Não existe upload pelo site — carregar mapa é operação de quem administra o
backend, não do usuário final do front.

## Validações do PDF atual (BUZ43_BUZ92_00)

- Georef: 12 rótulos OCR consistentes, escala implícita 1:30.000 exata
- P-77: grade interpolada = **1980.7 m** vs anotação do mapa **1980 m**
- Resíduo mediano nas 26 amostras LDA (âncoras): **−2 m**
- 100/127 rótulos de isóbata lidos; propagação fecha o resto sem conflitos

## Poços-alvo

`data/wells_manual.json` tem placeholders para 9-BUZ-43-RJS e 8-BUZ-92-RJS —
preencher E/N (SIRGAS2000 UTM 23S, EPSG:31983) e rodar `--stage 8` de novo.
`POST /analyze` também aceita `{x, y}` direto sem poço cadastrado.

## API

- `GET /wells` — poços com coordenadas (UTM + WGS84)
- `GET /map/layers` — isóbatas + obstáculos em GeoJSON WGS84
- `POST /analyze` — `{well_id | x,y, radius_m, margin_m, obstacle_buffer_m,
  include_projeto}` → zonas GeoJSON (WGS84 + anel UTM em propriedade),
  setores de azimute livres, clearance por grau, avisos obrigatórios

### Modelo de zonas

- `forbidden_shallow` — mais raso que a LDA do poço (riser tocaria)
- `transit` — ≥ LDA mas < LDA+margem: dá para passar, não é refúgio
- `uncertain` — faixa de ±1 isóbata (10 m) acima do limiar de refúgio
- `safe` — ≥ LDA+margem+10, alcançável
- `forbidden_obstacle` / `unreachable` — buffer de obstáculo / sem caminho

Setor livre = raio contínuo ≥500 m em água transitável que **alcança** água
`safe` dentro do raio de análise.

## Avisos obrigatórios (sempre presentes na resposta)

- Precisão batimétrica ~2% da LDA (nota do próprio mapa); confirmar com
  SUB/SSUB antes de uso operacional
- Ferramenta de planejamento — não substitui validação oficial nem análise
  de riser da engenharia
# rota-de-fuga-backend
