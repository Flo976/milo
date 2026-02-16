import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.model_manager import model_manager
from app.routers import chat, conversation, health, stt, translate, tts

logger = logging.getLogger("milo")


async def _metrics_updater():
    """Background task: update Prometheus gauge metrics every 15s."""
    import torch
    from app.middleware.metrics import MODELS_LOADED, GPU_VRAM_USED

    while True:
        try:
            MODELS_LOADED.set(len(model_manager.loaded_models()))
            if torch.cuda.is_available():
                GPU_VRAM_USED.set(torch.cuda.memory_allocated())
        except Exception as e:
            logger.debug("Metrics updater error: %s", e)
        await asyncio.sleep(15)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Milo Voice starting up")
    logger.info("Mode: %s", settings.default_mode)

    # Preload VAD at startup (lightweight, CPU-only, avoids WS timeout)
    from app.models.vad import load_vad
    await asyncio.to_thread(load_vad)

    # Preload GPU models at startup (slower startup, but first request is instant)
    # NLLB translator not preloaded — no longer in the chat/voice pipeline
    from app.models.stt import load_stt
    from app.models.tts import load_tts
    from app.models.llm_local import load_llm_local
    from app.models.llm_cloud import load_llm_cloud

    logger.info("Preloading GPU models...")
    await asyncio.to_thread(load_stt)
    await asyncio.to_thread(load_tts)
    await asyncio.to_thread(load_llm_local)
    load_llm_cloud()
    logger.info("All models loaded: %s", model_manager.loaded_models())
    logger.info("VRAM: %s", model_manager.vram_usage())

    # Start background metrics updater
    metrics_task = None
    if settings.prometheus_enabled:
        metrics_task = asyncio.create_task(_metrics_updater())

    yield

    if metrics_task:
        metrics_task.cancel()
    logger.info("Milo Voice shutting down")
    model_manager.unload_all()


API_DESCRIPTION = """
# Milo Voice API

Assistant vocal malagasy propulse par IA.

## Pipeline

**Texte :** Message MG → LLM (Claude / Mistral local) → Reponse MG

**Voix :** Audio → STT (Whisper) → LLM → TTS (MMS) → Audio

## Authentification

Tous les endpoints (sauf `/health`) necessitent un header :

```
Authorization: Bearer <API_KEY>
```

Le WebSocket `/conversation` utilise un query parameter `api_key`.

## Langues supportees

| Code | Langue |
|------|--------|
| `mg` | Malagasy |
| `fr` | Francais |
"""

app = FastAPI(
    title="Milo Voice API",
    description=API_DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health", "description": "Etat du serveur et des modeles"},
        {"name": "chat", "description": "Chat texte avec Milo (malagasy direct)"},
        {"name": "stt", "description": "Speech-to-Text — transcription audio malagasy"},
        {"name": "tts", "description": "Text-to-Speech — synthese vocale malagasy"},
        {"name": "translate", "description": "Traduction Malagasy ↔ Francais (NLLB)"},
        {"name": "conversation", "description": "Conversation vocale temps reel (WebSocket)"},
    ],
)

# CORS — allow local network access
cors_origins = list(settings.cors_origins) + ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
if settings.prometheus_enabled:
    from app.middleware.metrics import setup_metrics

    setup_metrics(app)

# Auth middleware
from app.middleware.auth import AuthMiddleware

app.add_middleware(AuthMiddleware)

# Routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(stt.router, prefix="/api/v1", tags=["stt"])
app.include_router(tts.router, prefix="/api/v1", tags=["tts"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(translate.router, prefix="/api/v1", tags=["translate"])
app.include_router(conversation.router, prefix="/api/v1", tags=["conversation"])
