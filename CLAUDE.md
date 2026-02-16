# CLAUDE.md — Projet Milo

## Contexte

Projet de fine-tuning Whisper pour la transcription audio en **malagasy**, avec objectif final de réponse vocale (TTS) en malagasy. Le bot conversationnel existe déjà en français/anglais sur un autre poste.

## Structure du projet

```
milo/
├── CLAUDE.md          # Ce fichier — instructions pour Claude Code
├── soul.md            # Identité et principes du projet
├── memory.md          # Notes persistantes inter-sessions
├── guide/             # Documentation et marche à suivre
│   └── guide.md       # Guide principal étape par étape
├── data/              # Datasets audio (pas versionné)
│   ├── raw/           # Données brutes téléchargées
│   ├── processed/     # Données nettoyées et formatées
│   └── augmented/     # Données augmentées
├── scripts/           # Scripts d'entraînement et de traitement
├── models/            # Modèles fine-tunés (pas versionné)
├── configs/           # Fichiers de configuration d'entraînement
└── evaluation/        # Résultats d'évaluation et métriques
```

## Conventions

- **Langue de travail** : Français pour la documentation, anglais pour le code
- **Nommage des scripts** : snake_case, préfixés par étape (`01_download_data.py`, `02_preprocess.py`...)
- **Configs** : YAML pour les hyperparamètres, JSON pour les métadonnées de datasets
- **Commits** : format `[étape] description` (ex: `[data] ajout script de nettoyage Common Voice`)

## Règles

1. Ne jamais committer de fichiers audio ou de modèles dans git
2. Toujours documenter les résultats d'évaluation (WER, CER) dans `evaluation/`
3. Chaque script doit être exécutable indépendamment avec des arguments CLI
4. Favoriser HuggingFace Transformers/Datasets pour la compatibilité
5. Tester sur un petit sous-ensemble avant de lancer un entraînement complet
6. Référencer `soul.md` pour le ton et les principes directeurs

## Stack technique cible

- **ASR** : OpenAI Whisper (fine-tuning via HuggingFace)
- **TTS** : Piper TTS ou VITS (à déterminer selon résultats)
- **Framework** : PyTorch, HuggingFace Transformers & Datasets
- **Audio** : librosa, soundfile, ffmpeg
- **Évaluation** : jiwer (WER/CER)

## Données principales

- Mozilla Common Voice (malagasy — `mg`)
- OpenSLR (si datasets malagasy disponibles)
- Données collectées manuellement (à venir)
