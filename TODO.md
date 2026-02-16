# Milo Voice — TODO

Suivi detaille des taches techniques. Voir [README.md](README.md) pour la roadmap produit.

---

## Fait

### P0 — Critiques

- [x] **B1** — `session_id` ignore dans Chat → verifie, fonctionne correctement
- [x] **B2** — Texte vide dans Translate retourne du texte hallucine → validation 422
- [x] **B4** — Message vide dans Chat → fallback local → reponse en anglais → validation 422
- [x] **B5** — Texte vide dans TTS → 500 → validation 422

### P1 — Important

- [x] **B7** — Score de confidence STT fixe a 0.85 → log-probabilities Whisper reelles
- [x] **B8** — Mode `voice` dans Chat ne retournait pas d'audio → TTS integre (`mode=voice`)
- [x] **B9** — 1/5 requetes paralleles echoue en 500 → per-model loading lock (double-check pattern)
- [x] **B6** — Metriques Prometheus custom jamais mises a jour → background task toutes les 15s
- [x] Verbalisation des chiffres TTS ("2024" → "efatra amby roapolo amby roa arivo")
- [x] Concurrent translate 500 → `threading.Lock` sur `tokenizer.src_lang` (etat mutable partage)
- [x] Memoire chat perdue au tour 3 → supprime troncature `history[-8:]`, `n_ctx` 1024 → 4096
- [x] Metrics toujours a 0 → `_metrics_updater()` background task dans le lifespan

### Optimisations appliquees

- [x] `DEFAULT_MODE=local` → `cloud` (supprime 2s de timeout inutile)
- [x] `n_gpu_layers=-1` (full GPU offload)
- [x] Whisper en float16 (VRAM 3 Go → 1.5 Go)
- [x] Prechargement de tous les modeles au startup
- [x] GPU semaphore `Semaphore(1)` → `Semaphore(2)`
- [x] Supprime NLLB de la pipeline chat/voice (LLM repond directement en malagasy)
- [x] `verbose=False` sur le LLM local

---

## A faire

### Traduction

- [ ] **B3** — Round-trip FR → MG → FR perd le sens (qualite NLLB insuffisante)
- [ ] Ameliorer traduction des expressions courantes ("Manao ahoana daholo")
- [ ] Cache de traductions frequentes (Redis)
- [ ] API de batch traduction/TTS en lot

### STT

- [ ] Detection auto de la langue STT (mg / fr)
- [ ] Fine-tuning STT sur textes longs (degradation au-dela de 10s)
- [ ] Limiter la taille des payloads STT (audio base64 trop gros)

### TTS

- [ ] Streaming TTS (chunks audio pour reduire le TTFB)
- [ ] Support format audio OGG / MP3 en sortie
- [ ] Fine-tuning TTS pour voix plus naturelle

### Chat / LLM

- [ ] Streaming Chat (SSE) pour UX temps reel
- [ ] Historique de conversation persistant (Redis + TTL configurable)

### Infra / Securite

- [ ] Rate limiting par token / IP
- [ ] Rotation des tokens API
- [ ] Queue de requetes avec priorite
- [ ] Dashboard Grafana exploitant les metriques custom
- [ ] Containerisation production-ready (healthchecks, restart, GPU passthrough)

---

## Audit API (2026-02-16)

Scores avant corrections P0/P1 :

| Endpoint | Score | Latence P50 | Observations |
|----------|-------|-------------|--------------|
| Health | 10/10 | <10ms | OK |
| TTS | 8/10 | 120ms | Rapide et fiable |
| STT | 7/10 | 237ms | Bon sur phrases courtes, degrade sur textes longs |
| Chat | 5/10 | 2.3s | Bonnes reponses mais pas de memoire conversationnelle |
| Translate | 6/10 | 130ms | Bon sur phrases simples, hallucinations sur texte vide |
