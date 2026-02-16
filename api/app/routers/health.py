import logging

from fastapi import APIRouter

from app.dependencies import get_redis
from app.models.model_manager import model_manager
from app.schemas.health import HealthResponse
from app.services.fallback import fallback_manager

logger = logging.getLogger("milo")
router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    """Verifie l'etat du serveur, des modeles charges et de la VRAM GPU.

    Pas d'authentification requise.
    """
    redis_ok = False
    try:
        r = await get_redis()
        await r.ping()
        redis_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if redis_ok else "degraded",
        version="0.1.0",
        mode=fallback_manager.mode,
        models_loaded=model_manager.loaded_models(),
        vram=model_manager.vram_usage(),
        redis_connected=redis_ok,
    )
