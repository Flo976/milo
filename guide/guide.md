# Guide — Fine-tuning Whisper pour le Malagasy

> Objectif : obtenir un modèle Whisper capable de transcrire de l'audio malagasy en local, puis ajouter la synthèse vocale (TTS) en malagasy.

---

## Vue d'ensemble

```
Phase 1 — ASR (Speech-to-Text)          Phase 2 — TTS (Text-to-Speech)
┌─────────────────────────────┐          ┌─────────────────────────────┐
│ 1. Environnement            │          │ 7. Collecte données TTS     │
│ 2. Collecte des données     │          │ 8. Entraînement TTS         │
│ 3. Prétraitement            │          │ 9. Intégration vocale       │
│ 4. Fine-tuning Whisper      │          │ 10. Pipeline complet        │
│ 5. Évaluation               │          └─────────────────────────────┘
│ 6. Intégration ASR au bot   │
└─────────────────────────────┘
```

---

## Phase 1 — ASR : Transcription du malagasy

### Étape 1 — Préparer l'environnement

**Objectif** : Installer tout ce qu'il faut pour entraîner Whisper.

**Prérequis matériel** :
- GPU avec >= 8 Go VRAM (whisper-small) ou >= 16 Go (whisper-medium)
- Si pas de GPU : on utilisera Google Colab ou un serveur distant
- ~20 Go d'espace disque pour les données + modèles

**Installation** :

```bash
# Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install librosa soundfile jiwer
pip install tensorboard evaluate
```

**Fichier `requirements.txt`** à créer :
```
torch>=2.1
torchaudio>=2.1
transformers>=4.36
datasets>=2.16
accelerate>=0.25
librosa>=0.10
soundfile>=0.12
jiwer>=3.0
tensorboard>=2.15
evaluate>=0.4
```

**Vérification** :
```python
import torch
print(f"CUDA disponible : {torch.cuda.is_available()}")
print(f"GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun'}")
```

---

### Étape 2 — Collecter les données

**Objectif** : Rassembler le maximum d'audio malagasy transcrit.

#### Source principale : Mozilla Common Voice

Common Voice a un dataset malagasy (`mg`). C'est notre base.

```python
from datasets import load_dataset

# Télécharger le dataset Common Voice malagasy
# Nécessite un token HuggingFace (accepter les conditions sur le site)
dataset = load_dataset(
    "mozilla-foundation/common_voice_16_1",
    "mg",
    token="TON_TOKEN_HF"
)

print(f"Train : {len(dataset['train'])} exemples")
print(f"Test  : {len(dataset['test'])} exemples")
```

> **Note** : Le dataset malagasy est petit (probablement quelques milliers d'exemples). C'est normal pour une langue low-resource. On compensera avec de l'augmentation de données.

#### Sources complémentaires

| Source | Description | Accès |
|--------|-------------|-------|
| OpenSLR | Datasets speech open-source | openslr.org |
| FLEURS | Google, 102 langues dont mg | HuggingFace |
| Bible.is | Lectures de la Bible en malagasy | bible.is |
| Radio malagasy en ligne | Données non transcrites (semi-supervisé) | Divers |

#### Collecte manuelle (optionnel mais très utile)

Si on a accès à des locuteurs malagasy :
1. Enregistrer des phrases courtes (5-15 secondes)
2. Format : WAV, 16kHz, mono
3. Transcrire chaque audio dans un fichier CSV : `audio_path,transcription`
4. Viser la diversité : hommes/femmes, âges, dialectes

---

### Étape 3 — Prétraiter les données

**Objectif** : Mettre toutes les données au format attendu par Whisper.

**Format cible** :
- Audio : 16kHz, mono, float32
- Texte : nettoyé, normalisé (minuscules, ponctuation standardisée)

```python
# Script de prétraitement — scripts/02_preprocess.py

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def preprocess(example):
    # Charger et resampler l'audio à 16kHz
    audio = example["audio"]

    # Extraire les features mel-spectrogram
    input_features = processor(
        audio["array"],
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features[0]

    # Tokenizer le texte cible
    labels = processor.tokenizer(example["sentence"]).input_ids

    return {
        "input_features": input_features,
        "labels": labels
    }

# Appliquer le prétraitement
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
```

**Nettoyage du texte malagasy** :
```python
import re

def clean_malagasy_text(text):
    # Minuscules
    text = text.lower()
    # Garder les caractères malagasy (alphabet latin de base)
    text = re.sub(r"[^a-zàâäéèêëïîôùûüÿçñ\s'-]", "", text)
    # Normaliser les espaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

**Augmentation des données** (crucial pour un petit dataset) :
- **Speed perturbation** : accélérer/ralentir l'audio (0.9x, 1.1x)
- **Ajout de bruit** : bruit ambiant léger
- **Pitch shifting** : modifier légèrement la hauteur
- **SpecAugment** : masquer des bandes de fréquences (intégré dans Whisper)

---

### Étape 4 — Fine-tuner Whisper

**Objectif** : Adapter Whisper au malagasy.

**Choix du modèle de base** :

| Modèle | Paramètres | VRAM min | Recommandation |
|--------|-----------|----------|----------------|
| whisper-tiny | 39M | 2 Go | Tests rapides uniquement |
| whisper-base | 74M | 3 Go | Prototypage |
| **whisper-small** | **244M** | **8 Go** | **Bon compromis pour du local** |
| whisper-medium | 769M | 16 Go | Si GPU suffisant |

**Configuration d'entraînement** (`configs/training_config.yaml`) :

```yaml
model:
  name: "openai/whisper-small"
  language: "mg"
  task: "transcribe"

training:
  output_dir: "./models/whisper-mg-v1"
  num_train_epochs: 30
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 2
  learning_rate: 1e-5
  warmup_steps: 500
  logging_steps: 25
  eval_steps: 500
  save_steps: 500
  fp16: true

  # Crucial pour les petits datasets
  gradient_checkpointing: true

evaluation:
  metric: "wer"  # Word Error Rate
```

**Script d'entraînement principal** :

```python
# scripts/03_train.py

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

# Charger le modèle pré-entraîné
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Forcer la langue malagasy et la tâche de transcription
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="mg", task="transcribe"
)
model.config.suppress_tokens = []

# Data collator pour le padding dynamique
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

# Arguments d'entraînement
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/whisper-mg-v1",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=30,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    fp16=True,
    gradient_checkpointing=True,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to=["tensorboard"],
)

# Métrique WER
import evaluate
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Lancer l'entraînement
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

# Sauvegarder le modèle final
trainer.save_model("./models/whisper-mg-final")
processor.save_pretrained("./models/whisper-mg-final")
```

**Astuces pour un petit dataset** :
1. **Commencer par whisper-small** — il généralise mieux que tiny/base avec peu de données
2. **Learning rate bas** (1e-5) — on veut affiner, pas casser les poids pré-entraînés
3. **Beaucoup d'epochs** (30+) — avec peu de données, le modèle a besoin de plus de passes
4. **Early stopping** — surveiller le WER sur le set de validation, arrêter si plateau
5. **Gradient checkpointing** — économise la VRAM au prix d'un peu de vitesse

---

### Étape 5 — Évaluer le modèle

**Objectif** : Mesurer la qualité de la transcription.

**Métriques** :
- **WER** (Word Error Rate) : taux d'erreur au niveau des mots — métrique principale
- **CER** (Character Error Rate) : taux d'erreur au niveau des caractères — utile pour le malagasy

**Cibles réalistes** :

| Niveau | WER | Qualité |
|--------|-----|---------|
| Baseline (sans fine-tuning) | 80-100%+ | Inutilisable |
| Après fine-tuning v1 | 40-60% | Compréhensible avec corrections |
| Objectif intermédiaire | 25-35% | Utilisable avec post-traitement |
| Objectif final | < 20% | Bon pour production |

> **Attention** : Pour une langue low-resource, un WER de 30% est déjà un très bon résultat. Ne pas se comparer aux scores EN/FR.

**Script d'évaluation** :

```python
# scripts/04_evaluate.py

from transformers import pipeline
from jiwer import wer, cer

# Charger le modèle fine-tuné
pipe = pipeline(
    "automatic-speech-recognition",
    model="./models/whisper-mg-final",
    device="cuda:0"
)

# Évaluer sur le set de test
results = []
for example in dataset["test"]:
    prediction = pipe(example["audio"]["array"])["text"]
    reference = example["sentence"]
    results.append({
        "prediction": prediction,
        "reference": reference,
        "wer": wer(reference, prediction),
        "cer": cer(reference, prediction)
    })

avg_wer = sum(r["wer"] for r in results) / len(results)
avg_cer = sum(r["cer"] for r in results) / len(results)

print(f"WER moyen : {avg_wer:.2%}")
print(f"CER moyen : {avg_cer:.2%}")
```

---

### Étape 6 — Intégrer l'ASR dans le bot existant

**Objectif** : Remplacer/ajouter le malagasy comme langue de transcription dans le bot.

**Option A — Via l'API HuggingFace locale** :

```python
from transformers import pipeline

# Charger le modèle au démarrage du bot
asr_mg = pipeline(
    "automatic-speech-recognition",
    model="./models/whisper-mg-final",
    device="cuda:0"  # ou "cpu"
)

def transcribe_malagasy(audio_path):
    result = asr_mg(audio_path)
    return result["text"]
```

**Option B — Via Whisper.cpp (plus performant pour du local)** :

```bash
# Convertir le modèle au format ggml pour whisper.cpp
python convert-hf-to-gguf.py ./models/whisper-mg-final --outfile whisper-mg.bin

# Utiliser avec whisper.cpp
./main -m whisper-mg.bin -f audio.wav -l mg
```

**Option C — Via faster-whisper (bon compromis)** :

```python
from faster_whisper import WhisperModel

model = WhisperModel("./models/whisper-mg-final", device="cuda", compute_type="float16")

segments, info = model.transcribe("audio.wav", language="mg")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

---

## Phase 2 — TTS : Synthèse vocale en malagasy

> Cette phase démarre une fois que l'ASR fonctionne correctement.

### Étape 7 — Collecter des données TTS

**Objectif** : Obtenir un corpus audio monolocuteur de qualité en malagasy.

**Exigences** :
- **Un seul locuteur** (ou quelques-uns, bien séparés)
- **Qualité audio** : enregistrement propre, pas de bruit de fond
- **Durée** : minimum 2h pour un résultat correct, idéalement 5-10h
- **Diversité de phrases** : phonèmes variés, intonations variées

**Sources possibles** :
1. **Enregistrement dédié** — Le mieux : un locuteur natif qui lit des textes
2. **Audiobooks malagasy** — Si disponibles en licence ouverte
3. **Bible audio malagasy** — Souvent bien lue, monolocuteur
4. **Filtrage de Common Voice** — Isoler les contributions d'un même locuteur

**Format** :
```
data/tts/
├── wavs/
│   ├── mg_001.wav
│   ├── mg_002.wav
│   └── ...
└── metadata.csv  # format: filename|transcription
```

---

### Étape 8 — Entraîner un modèle TTS

**Options de modèles** :

| Modèle | Qualité | Données min | Complexité |
|--------|---------|-------------|------------|
| **Piper TTS** | Bonne | 2-5h | Faible |
| VITS | Très bonne | 5-10h | Moyenne |
| Coqui TTS | Très bonne | 5-10h | Moyenne |
| Bark | Variable | Fine-tuning limité | Élevée |

**Recommandation** : Commencer par **Piper TTS** — il est conçu pour tourner en local, est léger, et fonctionne bien avec peu de données.

```bash
# Installation de Piper
pip install piper-tts

# L'entraînement de Piper utilise un processus dédié
# Documentation : https://github.com/rhasspy/piper
```

---

### Étape 9 — Intégrer la réponse vocale

**Objectif** : Le bot répond avec un audio en malagasy.

**Pipeline complet** :
```
Audio malagasy → [Whisper fine-tuné] → Texte malagasy
                                            ↓
                                     [Logique du bot]
                                            ↓
                                     Texte de réponse
                                            ↓
                              [Piper TTS malagasy] → Audio de réponse
```

---

### Étape 10 — Pipeline complet et optimisation

**Objectif** : Tout faire tourner de manière fluide en local.

**Optimisations** :
- Quantization du modèle Whisper (INT8 via CTranslate2)
- Modèle TTS optimisé ONNX
- Streaming audio (transcription en temps réel)
- Cache des réponses fréquentes
- Détection automatique de la langue (mg/fr/en)

---

## Checklist globale

- [ ] **Env** : Python + CUDA + dépendances installées
- [ ] **Data** : Common Voice mg téléchargé et exploré
- [ ] **Data** : Données nettoyées et prétraitées
- [ ] **Data** : Augmentation de données appliquée
- [ ] **ASR** : Baseline Whisper évaluée (sans fine-tuning)
- [ ] **ASR** : Premier fine-tuning complété
- [ ] **ASR** : WER < 40% atteint
- [ ] **ASR** : Modèle intégré dans le bot
- [ ] **TTS** : Données TTS collectées (>= 2h)
- [ ] **TTS** : Modèle Piper entraîné
- [ ] **TTS** : Réponse vocale intégrée dans le bot
- [ ] **Final** : Pipeline complet fonctionnel en local
