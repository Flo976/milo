"""
Claude Code HTTP Proxy — expose `claude -p` sur le reseau local.

Usage:
    python3 claude-proxy.py

Endpoints:
    POST /prompt  — envoie un prompt a Claude Code, retourne la reponse
    GET  /        — page de test
"""

import asyncio
import json
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PROJECT_DIR = "/mnt/c/Users/Florent Didelot/Desktop/milo"
CLAUDE_BIN = "/home/florent/.local/bin/claude"
PORT = 8001

app = FastAPI(
    title="Claude Code Proxy",
    description="Proxy HTTP pour editer le projet Milo via Claude Code (`claude -p`)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Instruction pour Claude Code")
    model: str = Field(default="sonnet", description="Modele: sonnet, opus, haiku")
    timeout: int = Field(default=120, description="Timeout en secondes")


class PromptResponse(BaseModel):
    response: str
    model: str
    elapsed_s: float


@app.post("/prompt", response_model=PromptResponse, summary="Envoyer un prompt a Claude Code")
async def run_prompt(req: PromptRequest):
    """Execute un prompt via `claude -p` dans le repertoire du projet Milo.

    Claude Code a acces a tous les fichiers du projet et peut les lire/editer.

    **Exemples :**
    - `{"prompt": "Lis le fichier .env et montre-moi la config"}`
    - `{"prompt": "Ajoute un endpoint GET /api/v1/version qui retourne la version"}`
    - `{"prompt": "Corrige le bug dans stt.py ligne 38"}`
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)  # avoid nested session check

    cmd = [
        CLAUDE_BIN, "-p",
        "--model", req.model,
        "--dangerously-skip-permissions",
        req.prompt,
    ]

    start = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=PROJECT_DIR,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=req.timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=504, detail=f"Claude timed out after {req.timeout}s")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.perf_counter() - start

    output = stdout.decode().strip()
    if proc.returncode != 0 and not output:
        error = stderr.decode().strip()
        raise HTTPException(status_code=500, detail=f"Claude error: {error}")

    return PromptResponse(
        response=output,
        model=req.model,
        elapsed_s=round(elapsed, 1),
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return """<!DOCTYPE html>
<html><head><title>Claude Code Proxy</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }
  h1 { color: #f97316; margin-bottom: 0.5rem; }
  .sub { color: #94a3b8; margin-bottom: 2rem; }
  textarea { width: 100%; height: 120px; background: #1e293b; color: #e2e8f0; border: 1px solid #334155;
             border-radius: 8px; padding: 1rem; font-size: 14px; font-family: monospace; resize: vertical; }
  button { background: #f97316; color: white; border: none; padding: 0.75rem 2rem; border-radius: 8px;
           font-size: 16px; cursor: pointer; margin-top: 1rem; }
  button:hover { background: #ea580c; }
  button:disabled { opacity: 0.5; cursor: wait; }
  #output { margin-top: 2rem; background: #1e293b; border-radius: 8px; padding: 1.5rem;
            white-space: pre-wrap; font-family: monospace; font-size: 13px; min-height: 100px;
            border: 1px solid #334155; }
  .meta { color: #64748b; font-size: 12px; margin-top: 0.5rem; }
  select { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px;
           padding: 0.5rem; margin-left: 1rem; }
</style></head>
<body>
  <h1>Claude Code Proxy</h1>
  <p class="sub">Edite le projet Milo via Claude Code</p>
  <textarea id="prompt" placeholder="Ex: Montre-moi la structure du projet..."></textarea>
  <div>
    <button onclick="send()" id="btn">Envoyer</button>
    <select id="model"><option value="sonnet">Sonnet</option><option value="opus">Opus</option><option value="haiku">Haiku</option></select>
  </div>
  <div id="output">En attente...</div>
  <div class="meta" id="meta"></div>
<script>
async function send() {
  const btn = document.getElementById('btn');
  const out = document.getElementById('output');
  const meta = document.getElementById('meta');
  btn.disabled = true; btn.textContent = 'En cours...';
  out.textContent = 'Claude reflechit...';
  meta.textContent = '';
  try {
    const r = await fetch('/prompt', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: document.getElementById('prompt').value,
        model: document.getElementById('model').value
      })
    });
    const d = await r.json();
    if (r.ok) {
      out.textContent = d.response;
      meta.textContent = d.model + ' | ' + d.elapsed_s + 's';
    } else {
      out.textContent = 'Erreur: ' + (d.detail || JSON.stringify(d));
    }
  } catch(e) { out.textContent = 'Erreur: ' + e.message; }
  btn.disabled = false; btn.textContent = 'Envoyer';
}
document.getElementById('prompt').addEventListener('keydown', e => { if (e.ctrlKey && e.key === 'Enter') send(); });
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    print(f"Claude Code Proxy on http://0.0.0.0:{PORT}")
    print(f"Project: {PROJECT_DIR}")
    print(f"Docs: http://0.0.0.0:{PORT}/docs")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
