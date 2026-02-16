Version : 1.0  |  Date : 15 février 2026  |  Auteur : Sooatek  |  Statut : Draft

1. Vision & Objectif

Nom du produit

Milo Voice

Vision one-liner

Donner une voix numérique au malagasy — le premier assistant vocal local-first pour 30 millions de malgachophones, qui fonctionne même sans internet.

Problème résolu

30 millions de personnes parlent malagasy. Aucun assistant vocal ne les comprend. Ni Siri, ni Alexa, ni Google Assistant ne supportent cette langue. Milo Voice comble ce vide en assemblant des composants existants (STT, TTS, LLM, traduction) dans un pipeline vocal unifié, conçu local-first pour fonctionner partout — y compris sans internet, essentiel pour les zones rurales de Madagascar.

Objectifs mesurables (KPIs)

KPI

Cible v1.0

Cible v2.0 (6 mois)

Latence conversation (local)

< 2s

< 1.5s

Latence conversation (cloud boost)

< 1.5s

< 1s

WER STT malagasy

~20%

< 15%

Conversations simultanées

50

200

Beta testeurs actifs

10

100

Disponibilité

99.5%

99.9%

Clients pilotes

1

5

2. Personas utilisateurs

🧑 Utilisateur final malgache (B2C)

Profil : Malgachophone, smartphone ou PC, connectivité variable

Besoin : Poser des questions, obtenir des informations, dicter du texte — en malagasy

Friction actuelle : Obligé de passer par le français ou l'anglais pour utiliser un assistant vocal

🏢 Entreprise / Call center (B2B)

Profil : Service client, centre d'appels, entreprise malgache

Besoin : Transcrire les appels, automatiser les réponses vocales en malagasy

Friction actuelle : Transcription manuelle, pas d'automatisation vocale en MG

🏛 ONG / Institution publique (B2G)

Profil : ONG, ministères, services publics malgaches

Besoin : Diffuser de l'information vocale en malagasy (santé, éducation, agriculture)

Friction actuelle : Contenus disponibles uniquement en français ou en texte écrit

👩‍💻 Développeur (B2D)

Profil : Développeur intégrant des capacités vocales malagasy dans son app

Besoin : API simple pour STT, TTS, conversation et traduction malagasy

Friction actuelle : Aucune API vocale malagasy n'existe

3. User Stories

Conversation vocale

En tant qu'utilisateur final, je veux parler en malagasy à Milo et recevoir une réponse vocale, afin d'obtenir de l'aide sans taper.

En tant qu'utilisateur final, je veux que la conversation continue en mode offline si internet coupe, afin de ne pas être interrompu.

Transcription

En tant qu'entreprise, je veux transcrire un fichier audio malagasy en texte, afin d'archiver et analyser mes appels clients.

En tant qu'ONG, je veux transcrire des témoignages audio en malagasy, afin de les documenter pour mes rapports.

Traduction

En tant qu'utilisateur final, je veux traduire du malagasy vers le français et inversement par la voix, afin de communiquer avec des non-malgachophones.

En tant qu'institution publique, je veux convertir des documents français en audio malagasy, afin de toucher les populations rurales.

TTS

En tant que développeur, je veux convertir du texte malagasy en audio via API, afin d'ajouter la synthèse vocale malagasy à mon application.

Accessibilité

En tant qu'utilisateur malvoyant, je veux interagir avec Milo entièrement par la voix, afin d'accéder à l'information sans interface visuelle.

Mode offline

En tant qu'utilisateur en zone rurale, je veux utiliser Milo même sans internet, afin d'avoir un assistant vocal fiable partout à Madagascar.

4. Architecture technique

Flux principal (local-first — offline)

┌─────────┐   ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌───────────────┐
│  Audio   │──▶│ Silero  │──▶│  Whisper  │──▶│  NLLB    │──▶│  Mistral 7B   │
│  MG      │   │  VAD    │   │  STT MG   │   │  MG→FR   │   │  local (GPU)  │
└─────────┘   └─────────┘   └───────────┘   └──────────┘   └───────┬───────┘
                                                                     │
┌─────────┐   ┌───────────┐   ┌──────────┐                          │
│  Audio   │◀──│  MMS-TTS  │◀──│  NLLB    │◀─────────────────────────┘
│  MG      │   │  MG       │   │  FR→MG   │
└─────────┘   └───────────┘   └──────────┘

Latence cible : ~1.85s | VRAM : ~7.5 Go | Fonctionne sans internet

Flux secondaire (cloud boost — internet)

┌─────────┐   ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌───────────────┐
│  Audio   │──▶│ Silero  │──▶│  Whisper  │──▶│  NLLB    │──▶│ Claude Haiku  │
│  MG      │   │  VAD    │   │  STT MG   │   │  MG→FR   │   │  API (cloud)  │
└─────────┘   └─────────┘   └───────────┘   └──────────┘   └───────┬───────┘
                                                                     │
┌─────────┐   ┌───────────┐   ┌──────────┐                          │
│  Audio   │◀──│  MMS-TTS  │◀──│  NLLB    │◀─────────────────────────┘
│  MG      │   │  MG       │   │  FR→MG   │
└─────────┘   └───────────┘   └──────────┘

Latence cible : ~1.5s | VRAM : ~3 Go (LLM déchargé vers le cloud)

Mécanisme de basculement (local-first)

Le LLM local (Mistral 7B) traite toutes les requêtes par défaut — aucun internet requis

Si internet disponible ET mode cloud activé → requête envoyée à Claude Haiku API avec timeout de 2s

Si timeout ou erreur cloud → retour transparent vers Mistral 7B local (déjà chargé en VRAM)

L'utilisateur ne perçoit pas la différence (même pipeline STT/TTS/traduction)

Avantage : fonctionne partout à Madagascar, y compris sans connexion

Métrique : taux d'utilisation cloud vs local loggé pour monitoring

Stack technique détaillée

Composant

Technologie

Modèle / Version

Rôle

VAD

Silero VAD

v5

Détection activité vocale

STT

Whisper

medium, fine-tuné MG

Speech-to-text malagasy

Traduction

NLLB-200

600M distilled

MG↔FR bidirectionnel

LLM local (principal)

Mistral 7B

GGUF Q4 (~4.5 Go VRAM)

Génération de réponses (local-first, offline)

LLM cloud (optionnel)

Claude

3.5 Haiku

Génération de réponses (boost cloud si internet)

TTS

MMS-TTS-mlg

Meta MMS, fine-tuné MG

Text-to-speech malagasy (fine-tuné sur voix native)

Backend

FastAPI

Python 3.11+

API REST + WebSocket

Frontend

React

18+

Interface web

Inference

llama.cpp (Mistral 7B) + vLLM

—

Serving des modèles (local-first)

GPU

NVIDIA RTX 5070 Ti

16 Go VRAM

Serveur local

VRAM budget par mode

Composant

Mode Local (principal)

Mode Cloud (boost)

Whisper medium (fine-tuné MG)

1.5 Go

1.5 Go

NLLB-200 600M

1.2 Go

1.2 Go

MMS-TTS-mlg (fine-tuné)

0.3 Go

0.3 Go

Silero VAD

~0 (CPU)

~0 (CPU)

Mistral 7B Q4

4.5 Go

4.5 Go

Claude Haiku

—

0 (API)

Total

~7.5 Go

~7.5 Go

Marge

8.5 Go libres

8.5 Go libres

Latence budget par étape

Étape

Local (principal)

Cloud (boost)

VAD

50ms

50ms

Whisper STT

300ms

300ms

NLLB MG→FR

100ms

100ms

LLM

1000ms (Mistral 7B local)

600ms (Claude API)

NLLB FR→MG

100ms

100ms

MMS-TTS

200ms

200ms

Réseau

0ms

150ms

Total

~1.85s (offline)

~1.5s (cloud)

5. Spécifications fonctionnelles

Interface web

Technologie : React SPA + WebSocket pour le streaming audio

Capture audio : Web Audio API (MediaRecorder, 16kHz mono WAV)

Feedback temps réel : Indicateur d'activité vocale, état du pipeline, affichage du texte transcrit en live

API REST

Endpoint

Méthode

Description

/api/v1/stt

POST

Audio → Texte malagasy

/api/v1/tts

POST

Texte malagasy → Audio

/api/v1/chat

POST

Message texte → Réponse texte (avec contexte)

/api/v1/translate

POST

Texte MG↔FR

/api/v1/conversation

WebSocket

Conversation vocale temps réel

/api/v1/health

GET

État du système, VRAM, mode actif

Modes de fonctionnement

Conversation : Pipeline complet voix → voix (mode principal)

Transcription seule : Audio MG → Texte MG (STT uniquement)

TTS seul : Texte MG → Audio MG

Traduction : Texte/audio MG ↔ FR

Limites (configurables)

Audio STT : max 30 secondes par requête

Texte TTS : max 500 caractères par requête

Contexte conversation : 10 derniers échanges

Taille fichier upload : max 10 Mo

Langues

Primaire : Malagasy (officiel)

Secondaire : Français

Direction traduction : MG→FR, FR→MG

Gestion des sessions

Session identifiée par session_id (UUID)

Contexte de conversation stocké en mémoire (Redis)

TTL session : 30 minutes d'inactivité

Pas de stockage audio par défaut (opt-in pour analytics)

6. Spécifications non-fonctionnelles

Critère

Spécification

Latence

< 2s mode local (principal), < 1.5s mode cloud (end-to-end voix→voix)

Disponibilité

99.5% (local-first = pas de dépendance internet)

Capacité

50 conversations simultanées sur RTX 5070 Ti

Sécurité

HTTPS obligatoire, API keys, pas de stockage audio par défaut

Vie privée

RGPD-compatible, pas de conservation des données vocales sauf opt-in

Accessibilité

WCAG 2.1 AA, navigation 100% clavier, lecteur d'écran

Compatibilité

Chrome 90+, Firefox 90+, Safari 15+, Edge 90+

Mobile

PWA responsive, fonctionne sur Android 10+ et iOS 15+

7. API Reference

Authentification

Toutes les requêtes API nécessitent un header Authorization: Bearer <API_KEY>.

Rate Limiting

Free tier : 100 requêtes/heure

Pro tier : 10 000 requêtes/heure

Enterprise : illimité

POST /api/v1/stt

Request :

{
  "audio": "<base64_encoded_wav>",
  "format": "wav",
  "sample_rate": 16000
}

Response :

{
  "text": "Manao ahoana tompoko",
  "language": "mg",
  "confidence": 0.87,
  "duration_ms": 2300,
  "processing_ms": 310
}

POST /api/v1/tts

Request :

{
  "text": "Manao ahoana tompoko",
  "language": "mg",
  "format": "wav"
}

Response : Content-Type: audio/wav — Binary audio data

POST /api/v1/chat

Request :

{
  "message": "Inona ny vaovao androany?",
  "session_id": "uuid-optional",
  "language": "mg",
  "mode": "text"
}

Response :

{
  "reply": "Ireto ny vaovao farany...",
  "session_id": "uuid",
  "mode": "local|cloud",
  "processing_ms": 1200
}

POST /api/v1/translate

Request :

{
  "text": "Manao ahoana",
  "source": "mg",
  "target": "fr"
}

Response :

{
  "translation": "Bonjour",
  "source": "mg",
  "target": "fr",
  "processing_ms": 95
}

WebSocket /api/v1/conversation

Protocole :

Client ouvre la connexion avec ?api_key=<KEY>&session_id=<UUID>

Client envoie des frames audio binaires (chunks de 320ms, 16kHz mono PCM)

Serveur envoie des messages JSON pour les événements

Client peut envoyer {"type": "stop"} pour interrompre

Événements serveur :

{"type": "vad", "speaking": true}
{"type": "transcript", "text": "...", "partial": true}
{"type": "reply_text", "text": "..."}
{"type": "reply_audio", "audio": "<base64>"}
{"type": "mode", "value": "local|cloud"}

8. Interfaces utilisateur

Écran Conversation (principal)

Grand bouton micro central (push-to-talk ou mains libres)

Indicateur visuel d'écoute active (onde sonore animée)

Historique des échanges (bulles : utilisateur à droite, Milo à gauche)

Texte transcrit affiché en temps réel

Bouton replay sur chaque réponse audio

Indicateur du mode (💻 local / 🌐 cloud boost)

Écran Transcription

Zone de drag & drop pour upload audio

Enregistrement direct depuis le micro

Résultat texte avec timestamps

Export en TXT, SRT, JSON

Écran TTS

Zone de saisie texte

Sélection de langue (MG / FR)

Bouton "Générer" → lecture audio inline

Téléchargement WAV/MP3

Dashboard Admin

Graphiques temps réel : latence, requêtes/min, taux de fallback

Logs des conversations (si opt-in activé)

VRAM et GPU usage

Liste des API keys actives

Mobile

PWA responsive, installable

Interface tactile optimisée pour le bouton micro

Fonctionne en mode offline (fallback local via service worker + API locale)

9. Infrastructure & Déploiement

Phase 1 — GPU local (Semaines 1-6)

Serveur : 192.168.3.102, RTX 5070 Ti 16 Go

Exposition : Cloudflare Tunnel (HTTPS, domaine milo.sooatek.com)

Stack : Docker Compose

Containers :

milo-api : FastAPI backend

milo-web : React frontend (Nginx)

milo-models : Inference server (Whisper + NLLB + MMS-TTS-mlg + Mistral 7B)

redis : Cache sessions

prometheus + grafana : Monitoring

Phase 2 — Hybride cloud + local (Semaines 7-12)

Cloud : VPS avec GPU (RunPod ou Hetzner GPU) pour la scalabilité

Local : GPU 192.168.3.102 comme fallback et développement

Load balancing : Traefik ou Caddy

CDN : Cloudflare pour le frontend statique

Docker Compose (simplifié)

services:
  api:
    build: ./api
    ports: ["8000:8000"]
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
  web:
    build: ./web
    ports: ["80:80"]
  redis:
    image: redis:7-alpine
  prometheus:
    image: prom/prometheus
  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]

CI/CD

GitHub Actions : Build, test, deploy sur push main

Tests : Unit tests + integration tests audio pipeline

Deploy : docker compose pull && docker compose up -d

10. Métriques & Analytics

Métrique

Méthode

Alerte si

Latence p50 / p95 / p99

Prometheus histogram

p95 > 3s

Taux de fallback local

Counter fallback/total

> 30%

WER en production

Échantillonnage 5% + review humaine

> 25%

Conversations / jour

Counter

< 10 (adoption faible)

Satisfaction (👍/👎)

Boutons dans l'UI

Ratio < 70% positif

VRAM utilisée

nvidia-smi export

> 14 Go

Erreurs API

Log 4xx/5xx

Taux > 5%

Temps de réponse STT

Timer par étape

> 500ms

11. Roadmap

v0.1 — Sprint 1 (Semaines 1-2) : API STT + TTS + LLM local

Setup Docker avec Whisper + MMS-TTS-mlg (fine-tuné) + Mistral 7B (llama.cpp)

Endpoints /stt et /tts fonctionnels

Pipeline local complet opérationnel dès le Sprint 1 (offline-ready)

Tests unitaires pipeline audio

Documentation API de base

v0.2 — Sprint 2 (Semaines 3-4) : Mode conversation

Intégration NLLB MG↔FR

Pipeline complet : Audio MG → Texte FR → Claude → Texte MG → Audio MG

Endpoint /chat avec gestion de session

Redis pour le contexte conversationnel

v0.3 — Sprint 3 (Semaines 5-6) : Interface web + WebSocket

React SPA : écran conversation, transcription, TTS

WebSocket streaming audio bidirectionnel

PWA installable

Cloudflare Tunnel pour accès externe

v0.4 — Sprint 4 (Semaines 7-8) : Cloud boost + monitoring

Intégration Claude Haiku API comme boost optionnel (Mistral 7B local tourne déjà depuis Sprint 1)

Mécanisme de basculement cloud ↔ local (timeout 2s)

Prometheus + Grafana dashboards

Alertes sur métriques critiques

v1.0 — Sprint 5 (Semaines 9-10) : Production ready

Tests de charge (50 conversations simultanées)

Documentation complète (API, onboarding, admin)

Onboarding 10 beta testeurs

Landing page + démo publique

1 client pilote identifié

12. Risques & Mitigations

#

Risque

Impact

Probabilité

Mitigation

1

Latence réseau élevée

Expérience dégradée

Moyenne

Fallback local automatique < 2s

2

Qualité TTS insuffisante

Adoption faible

Faible

MMS-TTS-mlg fine-tuné dès v1 sur voix native malagasy (80-150 samples)

3

Coût API Claude explose

Budget dépassé

Faible

Cache réponses fréquentes, rate limiting

4

GPU local tombe en panne

Service down

Faible

Phase 2 : cloud backup, monitoring proactif

5

Google ajoute le malagasy

Concurrence directe

Faible

Différenciation : personnalisation, API ouverte, offline

6

WER trop élevé en production

Frustration utilisateur

Moyenne

Collecte data + re-fine-tuning continu

7

Adoption lente

ROI insuffisant

Moyenne

Cibler B2B/B2G d'abord (valeur immédiate)

13. Budget

Développement (heures estimées)

Sprint

Tâches principales

Heures estimées

Sprint 1

Docker, STT, TTS, Mistral 7B local, tests

80h

Sprint 2

NLLB, LLM, chat, sessions

80h

Sprint 3

React, WebSocket, PWA, tunnel

100h

Sprint 4

Cloud boost (Claude API), basculement, monitoring

80h

Sprint 5

Tests charge, docs, onboarding

60h

Total



400h

Infrastructure (coûts mensuels)

Poste

Phase 1

Phase 2

GPU local (électricité)

~30 €/mois

~30 €/mois

Cloudflare (free/pro)

0-20 €/mois

20 €/mois

VPS Cloud GPU

—

~150-300 €/mois

Domaine

10 €/an

10 €/an

Total mensuel

~35 €

~220-350 €

API Claude (coûts estimés)

Volume

Coût estimé / mois

1 000 conversations/mois

~5 €

10 000 conversations/mois

~50 €

100 000 conversations/mois

~500 €

Basé sur Claude 3.5 Haiku : ~0.25$/1M input tokens, ~1.25$/1M output tokens

14. Success Criteria

Critère

Mesure

Deadline

v1.0 livrée

Code en production, tous les endpoints actifs

Semaine 10

Latence < 2s local

p95 mesuré sur 1000 requêtes

Semaine 10

Latence < 1.5s cloud

p95 mesuré sur 1000 requêtes

Semaine 10

10 beta testeurs actifs

≥ 10 utilisateurs avec ≥ 5 conversations chacun

Semaine 12

1 client pilote

Contrat ou LOI signé

Semaine 14

Satisfaction > 70%

Ratio thumbs up sur total votes

Semaine 12

Documentation complète

API docs + guide onboarding + admin guide

Semaine 10

Annexe : Glossaire

Terme

Définition

STT

Speech-to-Text — conversion de la parole en texte

TTS

Text-to-Speech — conversion du texte en parole

LLM

Large Language Model — modèle de langage génératif

VAD

Voice Activity Detection — détection d'activité vocale

NLLB

No Language Left Behind — modèle de traduction Meta

MMS

Massively Multilingual Speech — modèle vocal Meta

WER

Word Error Rate — taux d'erreur de reconnaissance vocale

VRAM

Video RAM — mémoire du GPU

MG

Malagasy (code ISO 639-1)

FR

Français (code ISO 639-1)

PWA

Progressive Web App