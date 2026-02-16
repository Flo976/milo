<p align="center">
  <img src="https://img.shields.io/badge/lang-malagasy-green?style=for-the-badge" alt="Malagasy"/>
  <img src="https://img.shields.io/badge/mode-local--first-blue?style=for-the-badge" alt="Local-first"/>
  <img src="https://img.shields.io/badge/GPU-RTX_5070_Ti-76b900?style=for-the-badge&logo=nvidia" alt="GPU"/>
  <img src="https://img.shields.io/badge/license-AGPL--3.0-orange?style=for-the-badge" alt="License"/>
  <a href="https://github.com/Flo976/milo"><img src="https://img.shields.io/badge/github-Flo976%2Fmilo-181717?style=for-the-badge&logo=github" alt="GitHub"/></a>
</p>

<h1 align="center">Milo Voice</h1>

<p align="center">
  <strong>Le premier assistant vocal IA en malagasy.</strong><br/>
  <em>Malagasy Intelligent Language Operator</em>
</p>

<p align="center">
  STT &nbsp;&#x2192;&nbsp; LLM &nbsp;&#x2192;&nbsp; TTS<br/>
  <sub>Parle en malagasy &nbsp;&#x2192;&nbsp; Comprend &nbsp;&#x2192;&nbsp; Repond en malagasy</sub>
</p>

---

## Le probleme

30 millions de personnes parlent malagasy. **Aucun assistant vocal ne les comprend.** Ni Siri, ni Alexa, ni Google Assistant ne supportent cette langue. Milo comble ce vide.

## La solution

Une pipeline vocale complete, **local-first**, qui tourne sur un seul GPU :

```
     Micro                                                    Haut-parleur
       |                                                           ^
       v                                                           |
  [ Whisper STT ]  -->  [ Claude / Mistral ]  -->  [ MMS-TTS ]
    fine-tune MG          repond en MG              synthese MG
     ~300ms                 ~2s                       ~200ms
```

## Performances

| Metrique | Valeur |
|----------|--------|
| Latence chat texte | **~2s** |
| Latence pipeline vocale | **~2.5s** |
| VRAM utilisee | **1.6 Go** / 16 Go |
| Mode | Cloud (Claude) + fallback local (Mistral 7B) |

---

## Stack technique

### Backend (Python / FastAPI)

| Composant | Modele | Role |
|-----------|--------|------|
| **STT** | Whisper Medium fine-tune MG | Transcription audio malagasy |
| **LLM** | Claude 3.5 Haiku (cloud) / Mistral 7B Q4 (local) | Comprehension + generation |
| **TTS** | Facebook MMS-TTS `mlg` | Synthese vocale malagasy |
| **VAD** | Silero VAD | Detection d'activite vocale |
| **Traduction** | NLLB-200 600M | Traduction MG <-> FR (endpoint dedie) |

### Frontend (React / TypeScript)

| Ecran | Description |
|-------|-------------|
| Conversation | Chat vocal temps reel via WebSocket |
| Transcription | Upload audio + transcription STT |
| Synthese | Saisie texte + generation audio TTS |
| Admin | Monitoring GPU, latence, metriques |

### Infrastructure

| Service | Port | Role |
|---------|------|------|
| FastAPI | `8000` | API REST + WebSocket |
| React (Vite) | `3001` | Interface web |
| Redis | `6379` | Sessions + cache |
| Prometheus | `9090` | Collecte metriques |
| Grafana | `3000` | Dashboard monitoring |

---

## Demarrage rapide

### Pre-requis

- **GPU NVIDIA** avec >= 8 Go VRAM (CUDA 12+)
- **Python 3.10+**
- **Node.js 18+**
- **Redis**

### 1. Cloner et configurer

```bash
git clone https://github.com/Flo976/milo.git
cd milo
cp .env.example .env
# Editer .env avec vos chemins de modeles et cle API
```

### 2. Backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt

# Telecharger les modeles
# - Whisper fine-tune MG dans models/whisper-mg-v1/
# - Mistral 7B GGUF dans models/

cd api
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Frontend

```bash
cd web
npm install
npm run dev
```

### 4. Docker (alternative)

```bash
docker compose up -d
```

---

## API

**Base URL :** `http://localhost:8000`
**Documentation interactive :** [http://localhost:8000/docs](http://localhost:8000/docs)

### Authentification

```
Authorization: Bearer <API_KEY>
```

### Endpoints

#### `GET /api/v1/health` — Etat du serveur

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "mode": "cloud",
  "models_loaded": ["vad", "stt", "tts", "llm_local", "llm_cloud"],
  "vram": { "allocated_gb": 1.56, "total_gb": 15.92 }
}
```

#### `POST /api/v1/chat` — Chat texte

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Salama, iza ianao?"}'
```

```json
{
  "reply": "Salama! Milo no anarako. Ahoana no fomba hanampiako anao?",
  "mode": "cloud",
  "processing_ms": 1926
}
```

#### `POST /api/v1/stt` — Speech-to-Text

```bash
curl -X POST http://localhost:8000/api/v1/stt \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"audio": "<base64_wav>", "format": "wav"}'
```

#### `POST /api/v1/tts` — Text-to-Speech

```bash
curl -X POST http://localhost:8000/api/v1/tts \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Manao ahoana"}' \
  --output output.wav
```

#### `POST /api/v1/translate` — Traduction MG <-> FR

```bash
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Salama", "source": "mg", "target": "fr"}'
```

#### `WS /api/v1/conversation` — Conversation vocale

```javascript
const ws = new WebSocket(
  "ws://localhost:8000/api/v1/conversation?api_key=YOUR_KEY"
);

// Envoyer de l'audio PCM int16 LE, 16kHz
ws.send(pcmBuffer);

// Recevoir les evenements
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  // { type: "transcript", text: "..." }
  // { type: "reply_text", text: "..." }
  // { type: "reply_audio", audio: "<base64_wav>" }
  // { type: "vad", speaking: true/false }
};
```

---

## Architecture

```
milo/
├── api/                      # Backend FastAPI
│   ├── app/
│   │   ├── main.py           # Point d'entree, lifespan, CORS
│   │   ├── config.py         # Configuration (pydantic-settings)
│   │   ├── models/           # Chargement des modeles ML
│   │   │   ├── stt.py        # Whisper STT
│   │   │   ├── tts.py        # MMS-TTS
│   │   │   ├── llm_cloud.py  # Claude API
│   │   │   ├── llm_local.py  # Mistral via llama-cpp
│   │   │   ├── translator.py # NLLB-200
│   │   │   ├── vad.py        # Silero VAD
│   │   │   └── model_manager.py  # GPU semaphore + lazy loading
│   │   ├── routers/          # Endpoints API
│   │   ├── schemas/          # Pydantic request/response
│   │   ├── services/         # Pipeline, audio, sessions
│   │   └── middleware/       # Auth, rate-limit, metrics
│   └── requirements.txt
├── web/                      # Frontend React + Vite
│   └── src/
│       ├── features/         # Conversation, Transcription, TTS, Admin
│       ├── hooks/            # useConversation, useWebSocket, ...
│       └── api/              # Client HTTP + WebSocket
├── scripts/                  # Entrainement et evaluation
├── monitoring/               # Prometheus + Grafana configs
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Modeles

| Modele | Taille | Source |
|--------|--------|--------|
| Whisper Medium (fine-tune MG) | ~1.5 Go (FP16) | Entraine sur Common Voice MG |
| Mistral 7B Instruct v0.3 | ~4 Go (Q4_K_M) | [HuggingFace](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF) |
| NLLB-200 600M | ~1.2 Go (FP16) | [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) |
| MMS-TTS MLG | ~300 Mo | [facebook/mms-tts-mlg](https://huggingface.co/facebook/mms-tts-mlg) |
| Silero VAD | ~2 Mo (CPU) | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) |

---

## Configuration

Variables d'environnement principales (voir `.env.example`) :

| Variable | Description | Defaut |
|----------|-------------|--------|
| `DEFAULT_MODE` | `cloud` (Claude prioritaire) ou `local` | `cloud` |
| `ANTHROPIC_API_KEY` | Cle API Claude | - |
| `LOCAL_LLM_N_GPU_LAYERS` | Couches GPU (-1 = tout) | `-1` |
| `WHISPER_MODEL_ID` | Chemin du modele Whisper | - |
| `API_KEY` | Cle d'authentification API | `change-me-in-production` |

---

## Scripts d'entrainement

```bash
python scripts/03_train.py        # Fine-tuning Whisper sur Common Voice MG
python scripts/04_evaluate.py     # Evaluation WER/CER
python scripts/05b_test_mistral.py # Test du LLM local
```

---

## Etat du projet

### Audit API (2026-02-16)

| Endpoint | Score | Latence P50 | Observations |
|----------|-------|-------------|--------------|
| Health | 10/10 | <10ms | OK |
| TTS | 8/10 | 120ms | Rapide et fiable |
| STT | 7/10 | 237ms | Bon sur phrases courtes, degrade sur textes longs |
| Chat | 5/10 | 2.3s | Bonnes reponses mais pas de memoire conversationnelle |
| Translate | 6/10 | 130ms | Bon sur phrases simples, hallucinations sur texte vide |

### Bugs connus

| # | Severite | Description |
|---|----------|-------------|
| B1 | CRITIQUE | `session_id` ignore dans Chat — pas de suivi de contexte |
| B2 | CRITIQUE | Texte vide dans Translate retourne du texte hallucine |
| B3 | MAJEUR | Round-trip FR→MG→FR perd le sens |
| B4 | MAJEUR | Message vide dans Chat → fallback local → reponse en anglais |
| B5 | MAJEUR | Texte vide dans TTS → 500 au lieu de 422 |
| B6 | MINEUR | Metriques Prometheus custom jamais mises a jour |
| B7 | MINEUR | `confidence` STT fixe a 0.85 (pas un vrai score) |
| B8 | MINEUR | Mode "voice" dans Chat ne retourne pas d'audio |
| B9 | MINEUR | 1/5 requetes paralleles echoue en 500 (race condition) |

---

## Roadmap

### Fait

- [x] STT Whisper fine-tune malagasy (WER 20.78%)
- [x] TTS MMS malagasy
- [x] LLM local (Mistral 7B Q4)
- [x] LLM cloud (Claude 3.5 Haiku) avec fallback
- [x] Pipeline vocale complete (STT → LLM → TTS)
- [x] Interface web React
- [x] WebSocket conversation temps reel
- [x] Monitoring Prometheus/Grafana
- [x] Documentation API (Swagger + ReDoc)

### P0 — Critiques ~~(a corriger immediatement)~~ FAIT

- [x] **B1** — Session_id fonctionne correctement (verifie, non reproductible)
- [x] **B2** — Valider texte vide dans Translate → 422
- [x] **B5** — Valider texte vide dans TTS → 422
- [x] **B4** — Valider message vide dans Chat → 422

### P1 — Important (sprint suivant)

- [ ] Score de confidence STT reel (CTC probabilities) au lieu du fixe 0.85
- [ ] Ameliorer la traduction round-trip MG↔FR
- [ ] Verbalisation des chiffres dans TTS/STT ("2024" → "roa arivo efatra amby roapolo")
- [ ] Mode voice dans Chat — retourner l'audio TTS avec la reponse
- [ ] Fix race condition sous charge (1/5 requetes concurrentes echoue)
- [ ] Metriques Prometheus custom (`milo_models_loaded`, `milo_gpu_vram_used_bytes`)
- [ ] Ameliorer traduction des expressions courantes ("Manao ahoana daholo")

### P2 — Backlog

- [ ] Detection auto de la langue STT (mg/fr)
- [ ] Streaming TTS (chunks audio pour reduire le TTFB)
- [ ] Streaming Chat (SSE) pour UX temps reel
- [ ] Rate limiting par token/IP
- [ ] Rotation des tokens API
- [ ] Queue de requetes avec priorite
- [ ] Cache de traductions frequentes (Redis)
- [ ] Support format audio OGG/MP3 en sortie TTS
- [ ] Historique de conversation persistant (Redis + TTL)
- [ ] Dashboard Grafana exploitant les metriques
- [ ] Fine-tuning STT sur textes longs (degradation >10s)
- [ ] API de batch (traduction/TTS en lot)
- [ ] Limiter la taille des payloads STT (audio base64)
- [ ] Fine-tuning TTS pour voix naturelle
- [ ] Mode offline complet (sans internet)
- [ ] Application mobile (React Native)
- [ ] Support dialectes regionaux

---

## Licence

Ce projet est sous licence [**AGPL-3.0**](LICENSE).

> **Note :** Les modeles Meta (NLLB-200, MMS-TTS) utilises sont sous licence CC-BY-NC 4.0
> (usage non-commercial). Pour un usage commercial, remplacer ces modeles par des
> alternatives permissives (Piper TTS, Opus-MT).

---

## Contribuer

```bash
# Fork + clone
git clone https://github.com/Flo976/milo.git
git checkout -b feature/ma-feature
# ... modifications ...
git commit -m "[feature] description"
git push origin feature/ma-feature
# Pull Request
```

---

<p align="center">
  <strong>Milo Voice</strong> — par <a href="https://sooatek.com">Sooatek</a><br/>
  <sub>Donner une voix numerique au malagasy</sub><br/><br/>
  <a href="https://github.com/Flo976/milo">github.com/Flo976/milo</a>
</p>
