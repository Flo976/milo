# Memory — Projet Milo

## Projet

- **Objectif** : Fine-tuner Whisper pour le malagasy (ASR) + ajouter TTS malagasy
- **Contexte** : Un bot conversationnel fonctionne déjà en FR/EN sur un autre poste
- **Dossier** : `C:\Users\Florent Didelot\Desktop\milo`
- **Démarrage** : 2026-02-14, dossier vide au départ

## Langue malagasy — Points clés

- Langue austronésienne (PAS indo-européenne), ~25M de locuteurs
- Code ISO 639-1 : `mg` — Code Common Voice : `mg`
- Alphabet latin, phonologie relativement régulière
- Variantes dialectales : merina (officiel), côtier, etc.
- Ressources numériques limitées (low-resource language)

## Hardware

- **GPU** : RTX 5070 Ti — 16 Go VRAM, CUDA 13.1, drivers 591.86
- **Environnement** : WSL2 Ubuntu 22.04 sur PC Windows (GPU visible dans WSL2)
- Pas besoin du PC Linux séparé, tout dans WSL2

## Décisions techniques

- **ASR** : Whisper-medium fine-tuné sur malagasy (HuggingFace Transformers)
- **TTS** : MMS-TTS-mlg (facebook/mms-tts-mlg) fine-tuné sur voix native
- **LLM local** : Mistral 7B Q4_K_M via llama.cpp (local-first, offline)
- **LLM cloud** : Claude Haiku (optionnel, boost si internet)
- **Traduction** : NLLB-200 600M distilled (MG↔FR)
- **Architecture** : Local-first — tout fonctionne sans internet
- Données : `badrex/malagasy-speech-full` (~166h, 28K train)

## État d'avancement

### ASR (Whisper)
- [x] Dataset complet : 28371 train / 3099 val / 3101 test (~166h audio)
- [x] Baseline Whisper-medium : WER 106.3%, CER 43.1% (attendu)
- [x] Fine-tuning lancé, loss 1.3 → 0.23 (excellent)
- [x] **Pausé à epoch ~4.5 (checkpoint-7000)** — ~5.5 epochs restantes
- [ ] Reprendre training (`--resume auto`)
- [ ] Évaluation formelle (script 04_evaluate.py corrigé et prêt)

### TTS (MMS-TTS-mlg)
- [x] Baseline testé : 10 phrases + long test (manakara.wav, 35.8s)
- [x] Qualité baseline correcte, génération ultra-rapide (~0.2s/phrase)
- [x] Scripts prêts : 06_prepare_tts_data.py + 07_finetune_tts.py
- [x] Données single-speaker préparées (speaker ff5x6S32, 41 samples)
- [x] **Fine-tuning terminé** — 600 steps, 200 epochs, loss 115 → convergé
- [x] Modèle sauvé : `/home/florent/milo/models/mms-tts-mg-ft/final/` (317M)
- [x] Test audio OK : `test_output.wav` (3.70s, "Manao ahoana tompoko...")

### LLM local (Mistral 7B)
- [x] Scripts prêts : 05_setup_mistral.sh + 05b_test_mistral.py
- [x] Modèle choisi : bartowski/Mistral-7B-Instruct-v0.3-GGUF (Q4_K_M, ~4.5 Go)
- [x] llama.cpp compilé avec CUDA (`/home/florent/milo/llama.cpp/build/bin/`)
- [x] Modèle téléchargé : `/home/florent/milo/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` (4.1G)
- [ ] Tester inference

### Infra
- [x] API status sur réseau local (port 5555) + endpoint /fix (Claude headless)
- [x] Claude Code installé dans WSL2 (`/usr/bin/claude` v2.1.42)
- [x] PRD mis à jour (local-first, Mistral 7B, MMS-TTS-mlg fine-tuné)
- [x] **Interface Milo Voice complète** — ~60 fichiers créés :
  - Backend FastAPI : `api/` (config, models, routers, schemas, services, middleware)
  - Frontend React : `web/` (Vite + Tailwind + Zustand + Recharts)
  - Docker Compose + monitoring (Prometheus + Grafana)
  - 4 écrans : Conversation, Transcription, TTS, Admin Dashboard
  - WebSocket conversation temps réel avec VAD
  - Pipeline vocal complet : STT → NLLB MG→FR → LLM → NLLB FR→MG → TTS

### Tests validés (2026-02-15)
- [x] `npm install` + `npm run build` : OK (0 vulnérabilités, PWA générée)
- [x] Backend FastAPI dans venv WSL2 : health endpoint OK
- [x] STT (Whisper checkpoint-7000) : chargé GPU, transcription ~1s
- [x] TTS (MMS-TTS-mlg) : audio généré en ~700ms
- [x] Translate NLLB bidirectionnel MG↔FR : OK (~70-400ms)
- [x] VRAM : 5.61 GB / 15.92 GB avec 3 modèles (10 GB libres)
- [x] Auth middleware, Prometheus metrics, Swagger docs : OK

### Fixes appliqués
- `torch.cuda.get_device_properties(0).total_mem` → `total_memory` (torch 2.11)
- `AutoTokenizer` → `NllbTokenizer` pour NLLB (transformers 5.1 breaking change)
- Whisper : chargement en fp32 (checkpoint entraîné en fp32), cast input au dtype du modèle
- **TTS finetune-hf-vits** (transformers 5.1 compat, dans WSL2) :
  - `VitsDiscriminator.__init__()` → ajout `self.post_init()` (manquant, cause `all_tied_weights_keys` error)
  - `plot.py` : `tostring_rgb()` → `buffer_rgba()` + slice RGBA→RGB (matplotlib breaking change)
  - `getattr(config, "pad_token_id", 0)` pour éviter AttributeError
  - `run_vits_finetuning.py` : telemetry import, overwrite_output_dir, load_from_disk, logging_dir

### À faire
- [ ] Tester Mistral 7B inference via llama-cli
- [ ] Reprendre ASR training (checkpoint-7000, --resume auto)
- [ ] Évaluation formelle ASR (04_evaluate.py)
- [ ] Tester pipeline E2E : STT → NLLB MG→FR → LLM → NLLB FR→MG → TTS
- [ ] Tester le WebSocket /conversation
- [ ] Frontend : `npm run dev` et navigation entre les 4 écrans
- [ ] Intégration dans le bot existant

## Pour reprendre le training

```bash
# Dans WSL2 :
tmux new -s milo
source /home/florent/milo/venv/bin/activate
python3 /home/florent/milo/scripts/03_train.py \
  --epochs 10 --batch-size 4 --grad-accum 4 \
  --eval-steps 1000 --save-steps 1000 \
  --resume auto \
  2>&1 | tee /home/florent/milo/training.log

# Puis l'orchestrateur :
tmux new -s orchestrator
source /home/florent/milo/venv/bin/activate
python3 /home/florent/milo/scripts/orchestrator.py
```
