"""API de jobs do droplet de PROCESSAMENTO (pesado).

Só existe aqui — nunca roda no droplet de produção. Recebe um PDF, dispara
o preprocess.py inteiro em background e, quando termina com sucesso, publica
o resultado (rsync de data/) no droplet de produção e pede pra API leve
recarregar.

  uvicorn jobs_api:app --port 8001

Protegida por uma chave compartilhada simples (não há usuários aqui — quem
autentica o pedido de verdade é o server.js, que é o único que deveria saber
essa chave). Não é exposta pelo nginx: só o droplet de produção deve
conseguir alcançar essa porta (ver firewall/ufw).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, 'uploads')
os.makedirs(UPLOADS, exist_ok=True)

INTERNAL_KEY = os.environ.get('ROTAFUGA_PROCESS_KEY')
PROD_HOST = os.environ.get('ROTAFUGA_PROD_HOST')            # ex: puppeteeruser@137.184.18.204
PROD_DATA_PATH = os.environ.get('ROTAFUGA_PROD_DATA_PATH')  # ex: /home/puppeteeruser/rota-fuga-backend/data/
PROD_RELOAD_URL = os.environ.get('ROTAFUGA_PROD_RELOAD_URL')  # ex: http://localhost:8000/reload (via túnel/ssh) ou URL interna

app = FastAPI(title='Rota de Fuga — jobs de processamento (droplet dedicado)')


def require_internal_key(x_internal_key: Optional[str] = Header(default=None)):
    if not INTERNAL_KEY or x_internal_key != INTERNAL_KEY:
        raise HTTPException(401, 'chave interna inválida ou não configurada')


@dataclass
class Job:
    id: str
    pdf_path: str
    log_path: str
    isobath_color: Optional[str] = None  # '#RRGGBB' — None usa o padrão calibrado
    status: str = 'running'     # running | done | error
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _push_to_production():
    if not (PROD_HOST and PROD_DATA_PATH):
        raise RuntimeError('ROTAFUGA_PROD_HOST/ROTAFUGA_PROD_DATA_PATH não configurados')
    result = subprocess.run(
        ['rsync', '-az', '--delete', os.path.join(BASE, 'data') + '/',
         f'{PROD_HOST}:{PROD_DATA_PATH}'],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f'rsync falhou: {result.stderr[-2000:]}')

    if PROD_RELOAD_URL:
        import urllib.request
        req = urllib.request.Request(
            PROD_RELOAD_URL, method='POST',
            headers={'X-Internal-Key': INTERNAL_KEY or ''})
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status >= 300:
                raise RuntimeError(f'reload da API leve falhou: HTTP {resp.status}')


def _run_job(job: Job):
    env = os.environ.copy()
    env['MAP_PDF'] = job.pdf_path
    if job.isobath_color:
        env['ISOBATH_COLOR'] = job.isobath_color
    else:
        env.pop('ISOBATH_COLOR', None)
    with open(job.log_path, 'w') as logf:
        proc = subprocess.Popen(
            [sys.executable, 'preprocess.py', '--stage', 'all'],
            cwd=BASE, env=env, stdout=logf, stderr=subprocess.STDOUT,
        )
        returncode = proc.wait()
    job.finished_at = time.time()
    if returncode != 0:
        job.status = 'error'
        job.error = f'preprocess saiu com código {returncode} — ver log'
        return
    try:
        _push_to_production()
        job.status = 'done'
    except Exception as exc:
        job.status = 'error'
        job.error = f'preprocess ok, mas falhou ao publicar no droplet de produção: {exc}'


@app.post('/jobs', dependencies=[Depends(require_internal_key)])
async def create_job(file: UploadFile = File(...),
                      isobath_color: Optional[str] = Form(default=None)):
    if not (file.filename or '').lower().endswith('.pdf'):
        raise HTTPException(422, 'esperado um arquivo .pdf')

    with JOBS_LOCK:
        if any(j.status == 'running' for j in JOBS.values()):
            raise HTTPException(409, 'já existe um processamento em andamento')
        job_id = uuid.uuid4().hex[:12]
        pdf_path = os.path.join(UPLOADS, f'{job_id}.pdf')
        job = Job(id=job_id, pdf_path=pdf_path,
                  log_path=os.path.join(UPLOADS, f'{job_id}.log'),
                  isobath_color=isobath_color or None)
        JOBS[job_id] = job

    with open(pdf_path, 'wb') as f:
        f.write(await file.read())

    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {'job_id': job_id}


@app.get('/jobs/{job_id}', dependencies=[Depends(require_internal_key)])
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'job não encontrado')
    tail = ''
    if os.path.exists(job.log_path):
        with open(job.log_path) as f:
            tail = ''.join(f.readlines()[-30:])
    return {
        'status': job.status, 'error': job.error, 'log_tail': tail,
        'started_at': job.started_at, 'finished_at': job.finished_at,
    }
