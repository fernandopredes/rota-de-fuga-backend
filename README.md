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

### Passo a passo

Se o mapa novo é da mesma família (template QGIS da Petrobras, mesmas
cores/legenda), o atalho é rodar tudo de uma vez e só conferir as imagens:

```bash
export MAP_PDF=/caminho/do/novo_mapa.pdf
.venv/bin/python preprocess.py --stage all
```

Se algo sair torto, rode estágio por estágio — cada um imprime diagnóstico e
salva uma imagem em `debug/` pra conferir antes de seguir pro próximo:

**0. Antes de começar** — anote as coordenadas UTM e a LDA de um poço que
tenha profundidade anotada no mapa novo (como o P-77 aqui). Você vai precisar
dele no passo 7 pra validar a grade interpolada.

**1. Georreferenciamento**
```bash
.venv/bin/python preprocess.py --stage 1
```
Confira `debug/01_georef.png`: a grade desenhada por cima deve alinhar com as
linhas pretas do mapa, com os valores UTM certos nas bordas.

Se os rótulos de coordenada (ex: `747000E`) estiverem em outra posição/fonte
nesse PDF, o OCR pode não achar nada — o comando falha com uma mensagem tipo:

```
Pontos insuficientes para georreferenciar (easting: 0/2+, northing: 0/2+).
  xs: [65.18, 348.64, 632.11, ...]
  ys: [281.1, 564.56, 848.03, ...]
```

Nesse caso, use o fallback manual: pegue 2+ valores de `xs:`/`ys:` (a
detecção de linhas de grade é geometria pura, não depende de OCR — sempre
funciona), abra `debug/overview.png` (ou renderize a faixa da borda em alta
DPI) e leia por olho qual UTM cada linha representa. Depois rode de novo com
`MANUAL_GEOREF`:

```bash
MANUAL_GEOREF='{"easting": {"65.18": 747000, "1765.96": 765000},
                 "northing": {"281.1": 7287000, "1414.95": 7275000}}' \
  .venv/bin/python preprocess.py --stage 1
```

As chaves são os valores de `xs`/`ys` (com tolerância de ~1pt — não precisa
ser exato); os valores são o UTM correspondente. `MANUAL_GEOREF` também
aceita um caminho de arquivo `.json` em vez de JSON inline. Quando definida,
pula o OCR das bordas inteiramente e usa só os pontos manuais — o resto do
pipeline (spacing, afim, validação) roda igual. Se o OCR funciona parcialmente
mas você não confia nele, prefira o manual: ele sempre ganha quando a
variável está definida.

**2. Isóbatas**
```bash
.venv/bin/python preprocess.py --stage 2
```
`debug/02_isobaths.png` deve parecer um mapa de contorno normal (curvas
fechadas, sem fiapos soltos). Se as isóbatas desse mapa forem de outra cor,
ajuste a constante `ISOBATH_COLOR` no topo de `preprocess.py` (hoje fixa no
azul `(0.12, 0.47, 0.71)`).

**3. Amostras de profundidade (rótulos vermelhos de equipamento)**
```bash
.venv/bin/python preprocess.py --stage 3
```
`debug/03a_samples.png` mostra pontos coloridos por profundidade. Se vier
vazio, sem problema — é só um reforço pra propagação; o mapa pode não ter
esse tipo de rótulo com N/E/LDA.

**4. OCR dos rótulos de isóbata**
```bash
.venv/bin/python preprocess.py --stage 4
```
Olhe a razão impressa no terminal (`N/M parsed`). Acima de ~70% costuma ser
suficiente pra propagação fechar o resto sozinha.

**5. Atribuir profundidade a cada isóbata**
```bash
.venv/bin/python preprocess.py --stage 5
```
`debug/03c_depths.png`: contornos em degradê contínuo do raso ao fundo.
**Linhas vermelhas = não resolvidas.** Se aparecer muita coisa vermelha, o
problema está no estágio 4 (poucos rótulos lidos).

**6. Grade contínua + validação**
```bash
.venv/bin/python preprocess.py --stage 6
```
O terminal imprime uma checagem tipo `P-77 check: grid=... delta=...` — hoje
hardcoded pro poço P-77 deste mapa. **Para um mapa novo, edite essa
validação dentro de `stage4_grid()`** trocando pelas coordenadas e LDA do
poço-referência que você anotou no passo 0. Delta de poucos metros = grade
confiável; delta grande = revisar os estágios anteriores antes de seguir.

**7. Obstáculos**
```bash
.venv/bin/python preprocess.py --stage 7
```
O terminal lista a contagem por categoria; `debug/05_obstacles.png` mostra
tudo colorido por tipo. Cores de traço que não estão em
`OBSTACLE_CATEGORIES` (topo de `preprocess.py`) ficam de fora **silenciosamente**
— se sobrar muita coisa sem categoria, use o mesmo truque que usamos aqui
(casar cada swatch de cor da legenda com o texto real ao lado dele via
`page.get_text('words')`) pra descobrir os códigos novos e adicioná-los.

**8. Poços**
```bash
.venv/bin/python preprocess.py --stage 8
```
Confira `data/wells.json`. Poços sem anotação OCR no mapa (tipicamente os
poços-alvo do título, tipo 9-BUZ-43-RJS/8-BUZ-92-RJS aqui) precisam entrar
manualmente em `data/wells_manual.json` (E/N em UTM SIRGAS2000 23S,
EPSG:31983) — depois rode o estágio 8 de novo.

**9. Subir a API**
```bash
lsof -ti:8000 -sTCP:LISTEN | xargs -r kill   # se já tiver uma rodando
.venv/bin/uvicorn main:app --port 8000
```

**10. Testar**
```bash
curl http://localhost:8000/wells
curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' -d '{"well_id": "<id-do-poco>"}'
```
E no front, abrir o módulo "Rota de Fuga" e clicar no mapa — as camadas
(`/map/layers`) devem carregar e o clique deve gerar um resultado coerente
com o mapa original.

### Ressalvas

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
