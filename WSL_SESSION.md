# WSL_SESSION.md — Briefing pour Claude Code dans WSL2

> Ce fichier est un résumé pour que Claude Code, lancé dans WSL2,
> sache exactement où on en est et quoi faire. À lire en premier.

## Objectif de cette session

**Tester la pipeline E2E complète** : Backend FastAPI + Frontend React.
Pipeline vocale : Audio MG → Whisper STT → NLLB MG→FR → LLM → NLLB FR→MG → MMS-TTS → Audio MG

## État actuel — tout est prêt SAUF le test pipeline

| Composant | État | Chemin |
|-----------|------|--------|
| ASR (Whisper) | fine-tuné, pausé ckpt-7000 | `/home/florent/milo/models/whisper-mg-v1/checkpoint-7000/` |
| TTS (MMS-TTS) | fine-tuné, modèle final | `/home/florent/milo/models/mms-tts-mg-ft/final/` |
| LLM (Mistral 7B) | GGUF téléchargé, PAS TESTÉ | `/home/florent/milo/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` |
| Traduction (NLLB) | fonctionne | HF cache (facebook/nllb-200-distilled-600M) |
| Backend FastAPI | code complet, pas lancé | `/mnt/c/Users/Florent Didelot/Desktop/milo/api/` |
| Frontend React | code complet, node_modules OK | `/mnt/c/Users/Florent Didelot/Desktop/milo/web/` |
| Redis | installé WSL2 | port 6379 |

## IMPORTANT — Fichiers Windows vs WSL2

- **Code backend/frontend** : sur Windows → `/mnt/c/Users/Florent Didelot/Desktop/milo/`
- **Modèles, venv, scripts** : dans WSL2 → `/home/florent/milo/`
- Ce sont des **copies séparées**, pas des symlinks
- Le `.env` est sur Windows : `/mnt/c/Users/Florent Didelot/Desktop/milo/.env`

## Bugs de config à corriger AVANT de lancer

### 1. Chemin GGUF Mistral (casse incorrecte)
Dans `.env` : `LOCAL_LLM_MODEL_PATH=/home/florent/milo/models/mistral-7b-instruct-v0.3-q4_k_m.gguf` (minuscules)
Fichier réel : `/home/florent/milo/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` (majuscules)
→ **Corriger le `.env` ou renommer le fichier** (Linux est case-sensitive)

### 2. TTS pointe vers baseline, pas le fine-tuné
Dans `.env` : `MMS_TTS_MODEL_ID=facebook/mms-tts-mlg` (baseline HuggingFace)
Modèle fine-tuné : `/home/florent/milo/models/mms-tts-mg-ft/final/`
→ **Mettre à jour si on veut tester le modèle fine-tuné**

### 3. Mode cloud sans clé API valide
Dans `.env` : `DEFAULT_MODE=cloud` et `ANTHROPIC_API_KEY=sk-ant-xxxxx` (placeholder)
→ **Mettre `DEFAULT_MODE=local`** pour utiliser Mistral, ou mettre une vraie clé Claude

### 4. LLM local utilise llama-cpp-python (binding Python), PAS llama-cli
Le fichier `api/app/models/llm_local.py` importe `from llama_cpp import Llama`.
C'est le package `llama-cpp-python`, pas le binaire `llama-cli` qu'on a compilé.
→ **Vérifier que `llama-cpp-python` est installé dans le venv** :
```bash
source /home/florent/milo/venv/bin/activate
pip show llama-cpp-python  # Si absent, pip install llama-cpp-python
```
Si absent et difficile à installer, alternative : modifier `llm_local.py` pour utiliser
llama-server en HTTP (le binaire compilé avec CUDA qu'on a déjà).

### 5. Versions torch/transformers
Le `requirements.txt` demande `torch==2.5.1, transformers==4.47.1`
Le venv WSL2 a `torch 2.11+cu128, transformers 5.1`
→ Ne PAS réinstaller depuis requirements.txt. Le code doit être compatible
avec les versions du venv. Appliquer les fixes connus si besoin.

## Plan de test — étape par étape

### Étape 1 : Corriger la config
```bash
# Éditer le .env
nano /mnt/c/Users/Florent\ Didelot/Desktop/milo/.env
# Corriger :
#   LOCAL_LLM_MODEL_PATH=/home/florent/milo/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
#   MMS_TTS_MODEL_ID=/home/florent/milo/models/mms-tts-mg-ft/final/
#   DEFAULT_MODE=local
```

### Étape 2 : Vérifier les dépendances
```bash
source /home/florent/milo/venv/bin/activate

# Vérifier les packages nécessaires pour le backend
python3 -c "import fastapi; print('fastapi', fastapi.__version__)"
python3 -c "import uvicorn; print('uvicorn OK')"
python3 -c "import redis; print('redis OK')"
python3 -c "import llama_cpp; print('llama-cpp-python OK')"
python3 -c "import pydantic_settings; print('pydantic-settings OK')"
python3 -c "import prometheus_client; print('prometheus OK')"
python3 -c "import prometheus_fastapi_instrumentator; print('instrumentator OK')"
python3 -c "from transformers import VitsModel, WhisperForConditionalGeneration; print('models OK')"

# Installer les manquants avec pip install <package>
```

### Étape 3 : Démarrer Redis
```bash
sudo service redis-server start
redis-cli ping  # Doit répondre PONG
```

### Étape 4 : Lancer le backend
```bash
source /home/florent/milo/venv/bin/activate
cd /mnt/c/Users/Florent\ Didelot/Desktop/milo
uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload
```
**Note** : le `main.py` est dans `api/app/main.py`, le module Python est `api.app.main`.

### Étape 5 : Tester les endpoints individuellement
```bash
# Health check
curl http://localhost:8000/api/v1/health -H "Authorization: Bearer change-me-in-production"

# TTS seul
curl -X POST http://localhost:8000/api/v1/tts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me-in-production" \
  -d '{"text": "Manao ahoana", "language": "mg"}'

# Chat texte (sans audio)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me-in-production" \
  -d '{"message": "Manao ahoana tompoko", "language": "mg"}'
```

### Étape 6 : Lancer le frontend
```bash
cd /mnt/c/Users/Florent\ Didelot/Desktop/milo/web
# Si node est dans Windows, utiliser depuis Windows :
# Ouvrir un terminal Windows et faire : cd Desktop\milo\web && npm run dev
# Ou si node est dispo dans WSL2 :
npx vite --port 3001
```
Le frontend proxie `/api` vers `http://localhost:8000` (voir `vite.config.ts`).
Ouvrir http://localhost:3001 dans le navigateur.

## Architecture backend (résumé)

```
api/app/
├── main.py           # FastAPI app, lifespan, CORS, routers
├── config.py         # Settings (pydantic-settings, lit .env)
├── dependencies.py   # Redis pool, SessionManager, auth
├── models/
│   ├── model_manager.py  # Lazy-load + GPU semaphore
│   ├── stt.py            # WhisperSTT (transcribe)
│   ├── tts.py            # MMSTTS (synthesize → WAV bytes)
│   ├── translator.py     # NLLBTranslator (translate mg↔fr)
│   ├── llm_local.py      # LocalLLM via llama-cpp-python
│   ├── llm_cloud.py      # Claude Haiku via anthropic SDK
│   └── vad.py            # Silero VAD (speech detection)
├── routers/
│   ├── health.py     # GET /api/v1/health
│   ├── stt.py        # POST /api/v1/stt (base64 audio → text)
│   ├── tts.py        # POST /api/v1/tts (text → WAV)
│   ├── chat.py       # POST /api/v1/chat (text → text, avec traduction)
│   ├── translate.py  # POST /api/v1/translate
│   └── conversation.py  # WS /api/v1/conversation (pipeline vocale complète)
├── services/
│   ├── pipeline.py   # run_stt, run_tts, run_translate, run_llm, run_voice_pipeline
│   ├── audio.py      # decode/encode audio, base64, mono 16kHz
│   ├── session.py    # SessionManager (Redis)
│   └── fallback.py   # Cloud/local mode switching
└── middleware/
    ├── auth.py       # API key auth
    ├── metrics.py    # Prometheus metrics
    └── rate_limit.py # Rate limiting
```

## Architecture frontend (résumé)

```
web/src/
├── App.tsx, main.tsx, routes.tsx
├── api/          # client.ts (fetch wrapper), endpoints.ts, types.ts, ws.ts
├── audio/        # recorder.ts, player.ts, encoder.ts
├── features/
│   ├── conversation/  # ConversationScreen, MicButton, ChatBubble, SoundWave...
│   ├── transcription/ # TranscriptionScreen, DropZone, ExportMenu
│   ├── tts/           # TtsScreen, TextInput, AudioPreview
│   └── admin/         # AdminDashboard, GpuMonitor, LatencyChart
├── hooks/        # useConversation, useAudioRecorder, useHealth, useTts, useWebSocket
├── store/        # Zustand stores (audio, conversation, settings)
├── components/   # Layout, Sidebar, StatusBar
└── styles/       # globals.css (Tailwind)
```

Stack : React 18 + Vite + Tailwind + Zustand + Recharts. PWA-ready.
4 écrans : Conversation (WebSocket vocal), Transcription (upload), TTS (texte), Admin.

## Fixes connus (transformers 5.1 + torch 2.11)

- `total_mem` → `total_memory` dans torch.cuda.get_device_properties (déjà géré dans model_manager.py)
- `AutoTokenizer` cassé pour NLLB → `NllbTokenizer` (déjà OK dans translator.py)
- Whisper checkpoint en fp32 : ne PAS forcer `torch_dtype=float16` (OK dans stt.py)

## Hardware

- **GPU** : RTX 5070 Ti, 16 Go VRAM, compute capability 12.0 (Blackwell)
- **CUDA** : 13.1, drivers 591.86
- **VRAM estimée** : ~5.6 Go avec STT+NLLB+TTS, ~10 Go avec Mistral en plus
- **Venv** : `/home/florent/milo/venv/` (torch 2.11+cu128, transformers 5.1)

## Docs de référence

- `CLAUDE.md` : Conventions du projet
- `memory.md` : Suivi d'avancement détaillé
- `soul.md` : Identité et principes de Milo
- `guide/guide.md` : Guide 10 étapes complet
