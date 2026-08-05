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

## Infraestrutura (dois droplets + upload pelo front)

O preprocess não roda nem na máquina de dev (sem capacidade local) nem no
droplet de produção do Taka-Storm — por isso existem **dois droplets**, com
papéis diferentes, e o admin pode disparar tudo pelo front sem precisar de SSH
no dia a dia:

- **Droplet pesado** (processamento) — `67.205.156.210`, Ubuntu 24.04, 1 vCPU,
  1GB RAM (+1GB swapfile), 24GB disco. Roda `jobs_api.py`
  (`rota-fuga-jobs.service`, porta 8001) — recebe o PDF, dispara
  `preprocess.py --stage all` em background e, quando termina, publica o
  resultado sozinho (ver abaixo). **Sempre ligado** (decisão consciente:
  simplicidade > economizar os ~$6-12/mês de mantê-lo parado). Toolchain
  pesado (PyMuPDF, tesserocr, scikit-image, matplotlib) — tudo via wheel
  pronto, nada precisou compilar, mesmo o `tesserocr`. `tessdata/eng.traineddata`
  também existe via pacote apt, mas o `preprocess.py` usa o `tessdata/` local
  do projeto (`TESSDATA = BASE + 'tessdata'`). zsh + oh-my-zsh por
  conveniência. Chave SSH própria (`~/.ssh/id_ed25519`, gerada nesse droplet)
  cadastrada como **deploy key** deste repo no GitHub, e também autorizada
  (`authorized_keys`) no droplet de produção — é essa segunda ligação que
  deixa o passo de publicação automático.
- **Droplet de produção** (137.184.18.204) — roda só a **API leve**
  (`main.py`+`analysis.py`, sem nenhuma lib de OCR/PDF) como
  `rota-fuga-api.service` em `127.0.0.1:8000`, atrás do nginx em
  `https://takatsugu-hub.com/fuga`. Venv com só `fastapi uvicorn numpy scipy
  shapely pyproj scikit-image` — nada de PyMuPDF/tesserocr/matplotlib lá (a
  máquina tem só 1 vCPU/~2GB RAM e o disco vive em ~90%, então o toolchain
  pesado nunca é instalado ali).

**Fluxo automático (upload pelo front do Taka-Storm, admin only):**
1. Admin sobe o PDF em `RotaFuga.tsx` → `POST /admin/rotafuga/maps` no
   `server.js` (é o portão de autenticação de verdade) → repassado pro
   droplet pesado (`POST /jobs` em `jobs_api.py`) com uma chave interna
   compartilhada (`ROTAFUGA_PROCESS_KEY`, nunca chega ao cliente).
2. `jobs_api.py` roda `preprocess.py --stage all` num subprocess em
   background (só um job por vez) e o front faz polling do status
   (`GET /admin/rotafuga/maps/:jobId`).
3. Ao terminar com sucesso, `jobs_api.py` sozinho: `rsync` de `data/` pro
   droplet de produção, depois `POST https://takatsugu-hub.com/fuga/reload`
   (mesma chave interna) pra API leve recarregar o `EscapeAnalyzer` em
   memória — sem restart manual de processo.

**Fluxo manual (mesma coisa, via SSH, útil pra depurar um mapa novo estágio
por estágio antes de confiar no upload direto):**
```bash
ssh root@67.205.156.210
cd /root/rota-fuga-backend
git pull                         # ou: rsync direto da máquina de dev
                                  # (o repo local pode estar à frente do origin —
                                  # nesse caso rsync é mais confiável que pull)
MAP_PDF=/root/rota-fuga-backend/nome_do_mapa.pdf .venv/bin/python preprocess.py --stage all
rsync -az data/ puppeteeruser@137.184.18.204:~/rota-fuga-backend/data/
curl -X POST https://takatsugu-hub.com/fuga/reload -H "X-Internal-Key: $ROTAFUGA_PROCESS_KEY"
```

Nenhum dos dois droplets guarda estado que não esteja também no repo Git ou
nos PDFs de origem — em tese o droplet pesado poderia voltar a ser
descartável mais tarde, mas hoje é permanente por simplicidade.

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
poços-alvo do título, tipo 9-BUZ-43-RJS/8-BUZ-92-RJS aqui) ficam com
`E`/`N` nulos — **não precisa mais editar `wells_manual.json` à mão**: depois
que o mapa é publicado, esses poços aparecem em `GET /wells/incomplete` e o
admin preenche lon/lat direto no painel "Poços sem coordenada" do front
(`RotaFuga.tsx`), que salva via `POST /wells/override` sem reprocessar o PDF
(ver seção **API** abaixo).

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

Upload pelo site existe e é admin-only (ver acima) — mas continua sem
revisão humana das imagens de `debug/` antes de publicar (decisão
consciente: o usuário sempre manda o PDF no formato/template certo, então
pulamos essa etapa).

## Validações do PDF atual (BUZ70D_BUZ43_00)

- Georef: escala implícita 1:30.000 exata
- P-77: grade interpolada = **1980.3 m** vs anotação do mapa **1980 m**
- 125/157 rótulos de isóbata lidos; propagação fecha o resto (6 trechos
  curtos ficam sem rótulo, comprimentos [42,116,327,63,383,38])
- 5 poços extraídos (P-74, P-7, P-77 com coordenada; os dois poços-alvo do
  título ficam para preencher via `/wells/override`)

## Poços sem coordenada (poços-alvo do título)

Não é mais preenchido editando `data/wells_manual.json` e reprocessando —
isso ficaria acoplado ao pipeline pesado por nada, já que coordenada de poço
não depende de OCR/geometria. Em vez disso, a API leve guarda um arquivo à
parte (`data/wells_overrides.json`, na API de **produção**, não no droplet
pesado) que é mesclado por cima de `wells.json` em tempo de leitura:

- `GET /wells/incomplete` — lista poços sem `E`/`N` (id, nome, nota) — é o
  que alimenta o painel "Poços sem coordenada" no front
- `POST /wells/override` *(protegido por `X-Internal-Key`)* — `{id, name?,
  lon, lat, lda?}` (ou `x`/`y` em UTM) — grava/atualiza sem reprocessar nada;
  `EscapeAnalyzer` já reflete a mudança na resposta seguinte
- `POST /analyze` também aceita `{x, y}` ou `{lon, lat}` direto, sem poço
  cadastrado, pra quem não quer nem cadastrar o poço

## API

- `GET /wells` — poços com coordenadas (UTM + WGS84), já com overrides
  aplicados
- `GET /wells/incomplete` — poços sem coordenada (ver seção acima)
- `POST /wells/override` *(protegido)* — grava coordenada/LDA de um poço
- `GET /map/layers` — isóbatas + obstáculos em GeoJSON WGS84
- `POST /analyze` — `{well_id | x,y, radius_m, margin_m, obstacle_buffer_m,
  include_projeto}` → zonas GeoJSON (WGS84 + anel UTM em propriedade),
  setores de azimute livres, clearance por grau, avisos obrigatórios
- `POST /reload` *(protegido)* — recarrega tudo de `data/` sem reiniciar o
  processo; chamado automaticamente pelo droplet pesado após publicar um
  mapa novo

Rotas marcadas *(protegido)* exigem o header `X-Internal-Key` com o valor de
`ROTAFUGA_PROCESS_KEY` — a própria API não tem conceito de usuário/sessão;
quem decide *quem* pode chegar a essas rotas é o `server.js` (`requireAuth` +
`requireAdmin`), que é o único lugar que deveria conhecer essa chave.

## API de jobs (`jobs_api.py`, só no droplet pesado)

Não roda no droplet de produção — só existe onde o preprocess de fato roda.
Toda protegida por `X-Internal-Key`; a porta (8001) hoje não tem firewall
restringindo por IP (ver "Pendências de hardening" abaixo).

- `POST /jobs` — multipart, campo `file` (PDF) → `{job_id}`; recusa (409) se
  já houver um job rodando (só um processamento por vez, a máquina é de
  1 vCPU)
- `GET /jobs/{id}` — `{status: running|done|error, error, log_tail,
  started_at, finished_at}`

## Pendências de hardening

- Porta 8001 (`jobs_api.py`) aberta pra qualquer origem, só protegida pela
  chave — o ideal é `ufw allow from <ip-produção> to any port 8001` (não fiz
  ainda: ativar `ufw` num droplet sem console configurado tem risco real de
  travar o próprio SSH se a regra de porta 22 não entrar antes)
- `main.py` mantém `CORS allow_origins=['*']` — aceitável porque as rotas de
  escrita já exigem a chave interna, mas vale revisar se o escopo crescer

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
