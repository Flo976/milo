import logging

from fastapi import APIRouter, HTTPException

from app.schemas.translate import TranslateRequest, TranslateResponse
from app.services.pipeline import run_translate

logger = logging.getLogger("milo")
router = APIRouter()

VALID_LANGS = {"mg", "fr"}


@router.post("/translate", response_model=TranslateResponse, summary="Traduction MG ↔ FR")
async def translate(req: TranslateRequest):
    """Traduit du texte entre malagasy et francais avec NLLB-200.

    Langues supportees : `mg` (malagasy), `fr` (francais).

    **Exemple :** `{"text": "Salama", "source": "mg", "target": "fr"}`
    """
    if req.source not in VALID_LANGS or req.target not in VALID_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Supported languages: {VALID_LANGS}",
        )
    if req.source == req.target:
        raise HTTPException(
            status_code=400,
            detail="Source and target must differ",
        )

    translation, processing_ms = await run_translate(
        req.text, req.source, req.target
    )

    return TranslateResponse(
        translation=translation,
        source=req.source,
        target=req.target,
        processing_ms=round(processing_ms),
    )
